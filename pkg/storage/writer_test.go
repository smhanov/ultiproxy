package storage

import (
	"bytes"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestWALModeAndSchemaInit(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_wal.db")

	writer, err := NewWriter(dbPath)
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}
	defer writer.Close()

	// Verify WAL mode
	var journalMode string
	err = writer.DB().QueryRow("PRAGMA journal_mode;").Scan(&journalMode)
	if err != nil {
		t.Fatalf("failed to query journal_mode: %v", err)
	}
	if journalMode != "wal" {
		t.Errorf("expected journal_mode wal, got %s", journalMode)
	}

	// Verify tables exist
	tables := []string{"requests", "request_attempts", "usage", "quota_observations"}
	for _, table := range tables {
		var name string
		err := writer.DB().QueryRow("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", table).Scan(&name)
		if err != nil {
			t.Errorf("expected table %s to exist, error: %v", table, err)
		}
	}
}

func TestEnqueue10kRecordsAndFlush(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_10k.db")

	writer, err := NewWriter(dbPath, WithQueueCapacity(16384), WithBatchSize(100))
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	nRecords := 10000

	start := time.Now()
	for i := 1; i <= nRecords; i++ {
		err := writer.TrackRequest(RequestRecord{
			ID:             int64(i),
			ClientKeyHash:  "key_hash_abc",
			LogicalID:      fmt.Sprintf("req-%d", i),
			RequestedModel: "claude-3-5-sonnet",
			ResolvedModel:  "claude-3-5-sonnet-20241022",
			Provider:       "anthropic",
			CreatedAt:      "2025-01-01T00:00:00Z",
			CompletedAt:    "2025-01-01T00:00:01Z",
			FinishReason:   "end_turn",
			StreamComplete: 1,
			TTFTMs:         150,
			TotalMs:        800,
		})
		if err != nil {
			t.Fatalf("failed to track request record %d: %v", i, err)
		}
	}

	// Close flushes all queued records
	if err := writer.Close(); err != nil {
		t.Fatalf("failed to close writer: %v", err)
	}
	t.Logf("Enqueued and flushed 10k records in %v", time.Since(start))

	// Reopen db to query count
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	var count int
	err = db.QueryRow("SELECT COUNT(*) FROM requests;").Scan(&count)
	if err != nil {
		t.Fatalf("failed to query count: %v", err)
	}
	if count != nRecords {
		t.Errorf("expected %d records in database, got %d", nRecords, count)
	}
}

func TestFullQueueReturnsErrQueueFullWithoutBlocking(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_queue_full.db")

	// Very small queue capacity (2 items) and huge batch size so items stay in queue
	// We can test by filling queue while worker is busy or stopped
	writer, err := NewWriter(dbPath, WithQueueCapacity(2), WithBatchSize(10))
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}
	defer writer.Close()

	// Enqueue in rapid succession without blocking
	var queueFullCount int
	var lock sync.Mutex
	start := time.Now()

	// Rapidly send 50 records
	for i := 1; i <= 50; i++ {
		err := writer.TrackUsage(UsageRecord{
			RequestID:    int64(i),
			PromptTokens: 100,
		})
		if errors.Is(err, ErrQueueFull) {
			lock.Lock()
			queueFullCount++
			lock.Unlock()
		}
	}

	duration := time.Since(start)
	// Must be non-blocking (takes less than 100ms for 50 non-blocking channel attempts)
	if duration > 100*time.Millisecond {
		t.Errorf("enqueuing took %v, expected instantaneous non-blocking return", duration)
	}

	t.Logf("Queue full encountered %d times (non-blocking)", queueFullCount)
}

func TestAllRecordTypes(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_all_types.db")

	writer, err := NewWriter(dbPath)
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	// 1. Request
	_ = writer.TrackRequest(RequestRecord{
		ID:             1,
		ClientKeyHash:  "hash1",
		LogicalID:      "log1",
		RequestedModel: "gpt-4o",
		ResolvedModel:  "gpt-4o",
		Provider:       "openai",
		CreatedAt:      "2025-01-01T00:00:00Z",
		CompletedAt:    "2025-01-01T00:00:01Z",
		FinishReason:   "stop",
		StreamComplete: 1,
		TTFTMs:         100,
		TotalMs:        500,
	})

	// 2. Attempt
	_ = writer.TrackAttempt(AttemptRecord{
		RequestID:         1,
		Attempt:           1,
		Provider:          "openai",
		Model:             "gpt-4o",
		UpstreamRequestID: "req_up_1",
		StatusCode:        200,
	})

	// 3. Usage
	_ = writer.TrackUsage(UsageRecord{
		RequestID:        1,
		PromptTokens:     50,
		CompletionTokens: 25,
		ReasoningTokens:  10,
		CachedTokens:     5,
		Cost:             0.0005,
	})

	// 4. Quota Observation
	_ = writer.TrackQuotaObservation(QuotaObservationRecord{
		Provider:   "openai",
		Label:      "tier-1-rpm",
		UsedPct:    0.35,
		Remaining:  650,
		Limit:      1000,
		Unit:       "requests",
		ObservedAt: "2025-01-01T00:00:00Z",
		Source:     "response_headers",
	})

	_ = writer.Close()

	// Query verify
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open err: %v", err)
	}
	defer db.Close()

	var reqCount, attCount, usgCount, qtaCount int
	_ = db.QueryRow("SELECT COUNT(*) FROM requests;").Scan(&reqCount)
	_ = db.QueryRow("SELECT COUNT(*) FROM request_attempts;").Scan(&attCount)
	_ = db.QueryRow("SELECT COUNT(*) FROM usage;").Scan(&usgCount)
	_ = db.QueryRow("SELECT COUNT(*) FROM quota_observations;").Scan(&qtaCount)

	if reqCount != 1 || attCount != 1 || usgCount != 1 || qtaCount != 1 {
		t.Errorf("counts mismatch: req=%d, att=%d, usg=%d, qta=%d", reqCount, attCount, usgCount, qtaCount)
	}
}

// --- T023: close-safe enqueue and honest SQLite error reporting ---

// TestWriter_ConcurrentCloseAndEnqueueNeverPanics stresses the enqueue/Close
// lifecycle: producers keep enqueueing while Close drains and closes the queue.
// Send-on-closed-channel must be impossible, so the test (and the process) must
// survive without a panic.
func TestWriter_ConcurrentCloseAndEnqueueNeverPanics(t *testing.T) {
	tempDir := t.TempDir()

	for round := 0; round < 40; round++ {
		dbPath := filepath.Join(tempDir, fmt.Sprintf("race_%d.db", round))

		// Tiny queue and batch so the worker cannot keep up and the window
		// between the closed check and the channel send stays wide open.
		writer, err := NewWriter(dbPath, WithQueueCapacity(2), WithBatchSize(1))
		if err != nil {
			t.Fatalf("round %d: failed to create writer: %v", round, err)
		}

		var wg sync.WaitGroup
		for g := 0; g < 8; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := 0; i < 100; i++ {
					// RequestID stays 0 so the stress stays about the channel
					// lifecycle and not about foreign-key noise.
					_ = writer.TrackUsage(UsageRecord{PromptTokens: 1})
					_ = writer.TrackRequest(RequestRecord{ID: int64(i), LogicalID: fmt.Sprintf("req-%d", i)})
				}
			}()
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := writer.Close(); err != nil {
				t.Errorf("round %d: Close: %v", round, err)
			}
		}()

		wg.Wait()
	}
}

// TestWriter_EnqueueAfterCloseReturnsTypedError proves a closed writer answers
// producers with the sentinel ErrClosed (and not ErrQueueFull, and not a panic).
func TestWriter_EnqueueAfterCloseReturnsTypedError(t *testing.T) {
	tempDir := t.TempDir()
	writer, err := NewWriter(filepath.Join(tempDir, "closed.db"))
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	err = writer.TrackRequest(RequestRecord{LogicalID: "after-close"})
	if err == nil {
		t.Fatalf("TrackRequest after Close returned nil, want an error")
	}
	if !errors.Is(err, ErrClosed) {
		t.Errorf("TrackRequest after Close = %v, want ErrClosed", err)
	}
	if errors.Is(err, ErrQueueFull) {
		t.Errorf("closed writer must not be reported as ErrQueueFull, got %v", err)
	}

	err = writer.TrackUsage(UsageRecord{PromptTokens: 1})
	if !errors.Is(err, ErrClosed) {
		t.Errorf("TrackUsage after Close = %v, want ErrClosed", err)
	}

	// Closing twice must stay a no-op, not panic.
	if err := writer.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

// TestWriter_SQLiteExecErrorIsSurfaced injects a failing statement into SQLite
// (a BEFORE INSERT trigger that aborts) and requires the writer to report the
// failure instead of silently dropping the record.
func TestWriter_SQLiteExecErrorIsSurfaced(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "exec_error.db")

	writer, err := NewWriter(dbPath, WithQueueCapacity(16), WithBatchSize(8))
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	if _, err := writer.DB().Exec(`CREATE TRIGGER fail_requests_before_insert
BEFORE INSERT ON requests
BEGIN
    SELECT RAISE(ABORT, 'injected_exec_failure');
END;`); err != nil {
		t.Fatalf("failed to install failing trigger: %v", err)
	}

	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer func() { log.SetOutput(os.Stderr) }()

	_ = writer.TrackRequest(RequestRecord{ID: 7, LogicalID: "req-7", Provider: "prov"})

	// The failed flush must also be returned by Close, not only logged.
	if err := writer.Close(); err == nil {
		t.Errorf("Close returned nil although the batch insert failed")
	} else if !strings.Contains(err.Error(), "injected_exec_failure") {
		t.Errorf("Close error = %v, want it to carry the injected exec failure", err)
	}

	out := buf.String()
	if !strings.Contains(out, "injected_exec_failure") {
		t.Errorf("SQLite exec error was dropped: writer log = %q", out)
	}
	if !strings.Contains(strings.ToLower(out), "request") {
		t.Errorf("log line does not identify the affected telemetry kind: %q", out)
	}

	// The row really did not land.
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	defer db.Close()
	var rows int
	if err := db.QueryRow("SELECT COUNT(*) FROM requests;").Scan(&rows); err != nil {
		t.Fatalf("count requests: %v", err)
	}
	if rows != 0 {
		t.Errorf("requests rows = %d, want 0 (trigger aborts the insert)", rows)
	}
}

// TestWriter_SQLiteBeginErrorIsSurfaced makes even the transaction begin fail
// (the schema is dropped out from under the writer) and requires a log line.
func TestWriter_SQLiteBeginErrorIsSurfaced(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "begin_error.db")

	writer, err := NewWriter(dbPath, WithQueueCapacity(16), WithBatchSize(8))
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	if _, err := writer.DB().Exec(`DROP TABLE requests;`); err != nil {
		t.Fatalf("failed to drop table: %v", err)
	}

	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer func() { log.SetOutput(os.Stderr) }()

	_ = writer.TrackRequest(RequestRecord{ID: 9, LogicalID: "req-9"})
	if err := writer.Close(); err == nil {
		t.Errorf("Close returned nil although the batch could not be written")
	}

	logged := strings.ToLower(buf.String())
	if !strings.Contains(logged, "prepare") && !strings.Contains(logged, "begin") {
		t.Errorf("SQLite begin/prepare error was dropped: writer log = %q", buf.String())
	}
}
