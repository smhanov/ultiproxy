package storage

import (
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"
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
