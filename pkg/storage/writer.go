package storage

import (
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"log"
	"sync"

	_ "modernc.org/sqlite"
)

//go:embed schema.sql
var schemaSQL string

var (
	ErrQueueFull = errors.New("storage telemetry queue is full")
	ErrClosed    = errors.New("storage telemetry writer is closed")
)

// Record types matching telemetry schema

type RequestRecord struct {
	ID             int64
	ClientKeyHash  string
	LogicalID      string
	RequestedModel string
	ResolvedModel  string
	Provider       string
	CreatedAt      string
	CompletedAt    string
	FinishReason   string
	StreamComplete int
	ErrorClass     string
	TTFTMs         int64
	TotalMs        int64
}

type AttemptRecord struct {
	ID                int64
	RequestID         int64
	Attempt           int
	Provider          string
	Model             string
	UpstreamRequestID string
	StatusCode        int
	ErrorClass        string
	RetryAfterSeconds int
	ResetAt           string
}

type UsageRecord struct {
	ID               int64
	RequestID        int64
	PromptTokens     int64
	CompletionTokens int64
	ReasoningTokens  int64
	CachedTokens     int64
	Cost             float64
}

type QuotaObservationRecord struct {
	ID         int64
	Provider   string
	Label      string
	UsedPct    float64
	Remaining  float64
	Limit      float64
	Unit       string
	ResetAt    string
	ObservedAt string
	Source     string
}

// Option configures Writer.
type Option func(*Writer)

// WithQueueCapacity configures writer bounded buffer capacity.
func WithQueueCapacity(cap int) Option {
	return func(w *Writer) {
		w.queueCap = cap
	}
}

// WithBatchSize configures max items in one transaction.
func WithBatchSize(size int) Option {
	return func(w *Writer) {
		w.batchSize = size
	}
}

// Writer is an asynchronous SQLite telemetry writer.
type Writer struct {
	db        *sql.DB
	queueCap  int
	batchSize int
	queue     chan any

	// logger receives every telemetry write failure (begin/prepare/exec/commit)
	// so accounting gaps stay observable. It defaults to the standard logger.
	logger *log.Logger

	mu sync.Mutex
	// closed is read and written only while mu is held, and the queue send in
	// enqueue happens under the same mutex, so a producer can never observe
	// closed == false and then send on a channel Close already closed.
	closed bool
	done   chan struct{}

	// flushErr is the first batch error of the worker's final flush, published
	// before done is closed and read by Close after done, so no lock is needed.
	flushErr error
}

// NewWriter opens SQLite database at dbPath, applies schema and PRAGMAs, and starts background worker.
func NewWriter(dbPath string, opts ...Option) (*Writer, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite database: %w", err)
	}

	// PRAGMAs: WAL mode, busy_timeout=5000, foreign_keys ON
	pragmas := []string{
		"PRAGMA journal_mode=WAL;",
		"PRAGMA busy_timeout=5000;",
		"PRAGMA foreign_keys=ON;",
	}
	for _, pragma := range pragmas {
		if _, err := db.Exec(pragma); err != nil {
			db.Close()
			return nil, fmt.Errorf("failed to execute pragma %q: %w", pragma, err)
		}
	}

	// Apply schema migrations
	if _, err := db.Exec(schemaSQL); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to apply schema migrations: %w", err)
	}

	w := &Writer{
		db:        db,
		queueCap:  4096,
		batchSize: 100,
		done:      make(chan struct{}),
		logger:    log.Default(),
	}

	for _, opt := range opts {
		opt(w)
	}

	w.queue = make(chan any, w.queueCap)
	go w.worker()

	return w, nil
}

// DB returns the underlying *sql.DB connection (e.g. for testing or reading).
func (w *Writer) DB() *sql.DB {
	return w.db
}

// Ready reports whether the writer can still accept telemetry. A closed or
// nil writer is not ready.
func (w *Writer) Ready() bool {
	if w == nil {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return !w.closed && w.db != nil
}

// enqueue admits one record without blocking. The whole admission - the closed
// check and the channel send - happens under mu, and Close closes the queue
// under the same mutex, so the two can never interleave into a send on a closed
// channel (which would panic the process).
func (w *Writer) enqueue(item any) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return ErrClosed
	}

	select {
	case w.queue <- item:
		return nil
	default:
		return ErrQueueFull
	}
}

// logf reports a telemetry write failure instead of dropping it silently.
func (w *Writer) logf(format string, args ...any) {
	if w.logger == nil {
		return
	}
	w.logger.Printf("ultiproxy/storage: "+format, args...)
}

// TrackRequest enqueues a request record non-blockingly.
//
// A record with a positive ID is upserted, so a caller may open the row early
// (before any attempt/usage rows that reference it) and write the terminal
// outcome later.
func (w *Writer) TrackRequest(req RequestRecord) error {
	return w.enqueue(req)
}

// TrackAttempt enqueues a request attempt record non-blockingly.
func (w *Writer) TrackAttempt(attempt AttemptRecord) error {
	return w.enqueue(attempt)
}

// TrackUsage enqueues a token/cost usage record non-blockingly.
func (w *Writer) TrackUsage(usage UsageRecord) error {
	return w.enqueue(usage)
}

// TrackQuotaObservation enqueues a quota observation non-blockingly.
func (w *Writer) TrackQuotaObservation(quota QuotaObservationRecord) error {
	return w.enqueue(quota)
}

// Close flushes all queued telemetry records, commits them, and closes the database.
func (w *Writer) Close() error {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return nil
	}
	w.closed = true
	close(w.queue)
	w.mu.Unlock()

	<-w.done
	dbErr := w.db.Close()
	if w.flushErr != nil {
		// A failed final flush is a data loss the caller must see; the database
		// close error is kept alongside it rather than dropped.
		return errors.Join(fmt.Errorf("telemetry flush failed: %w", w.flushErr), dbErr)
	}
	return dbErr
}

func (w *Writer) worker() {
	var firstErr error
	keepErr := func(err error) {
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	defer func() {
		// Published before done closes; Close reads it only after <-w.done.
		w.flushErr = firstErr
		close(w.done)
	}()

	batch := make([]any, 0, w.batchSize)

	for {
		item, ok := <-w.queue
		if !ok {
			// Queue closed: write any remaining in batch
			if len(batch) > 0 {
				keepErr(w.writeBatch(batch))
			}
			return
		}

		batch = append(batch, item)

		// Drain up to batchSize without blocking
	drainLoop:
		for len(batch) < w.batchSize {
			select {
			case nextItem, nextOk := <-w.queue:
				if !nextOk {
					keepErr(w.writeBatch(batch))
					return
				}
				batch = append(batch, nextItem)
			default:
				break drainLoop
			}
		}

		keepErr(w.writeBatch(batch))
		batch = batch[:0]
	}
}

// writeBatch persists one batch inside a single transaction and reports the
// first failure. Every layer - begin, prepare, exec, commit - is logged, so a
// telemetry gap is always observable, and the error is returned so Close can
// surface a failed final flush.
func (w *Writer) writeBatch(batch []any) error {
	if len(batch) == 0 {
		return nil
	}

	tx, err := w.db.Begin()
	if err != nil {
		w.logf("dropping %d telemetry record(s): begin transaction: %v", len(batch), err)
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	// The requests row is written as an upsert: callers open it (id only, plus
	// what is known at dispatch start) before recording attempts/usage against
	// it, then re-write it with the terminal outcome. request_attempts and
	// usage reference requests(id) with a foreign key, so the parent has to be
	// in place before the children even when a batch flushes one item at a
	// time. Rows without an explicit id get a fresh auto rowid and never
	// conflict.
	reqStmt, err := tx.Prepare(`
INSERT INTO requests (id, client_key_hash, logical_id, requested_model, resolved_model, provider, created_at, completed_at, finish_reason, stream_complete, error_class, ttft_ms, total_ms)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    client_key_hash = excluded.client_key_hash,
    logical_id = excluded.logical_id,
    requested_model = excluded.requested_model,
    resolved_model = excluded.resolved_model,
    provider = excluded.provider,
    created_at = excluded.created_at,
    completed_at = excluded.completed_at,
    finish_reason = excluded.finish_reason,
    stream_complete = excluded.stream_complete,
    error_class = excluded.error_class,
    ttft_ms = excluded.ttft_ms,
    total_ms = excluded.total_ms;
`)
	if err != nil {
		w.logf("dropping %d telemetry record(s): prepare requests statement: %v", len(batch), err)
		return fmt.Errorf("prepare requests statement: %w", err)
	}
	defer reqStmt.Close()

	attStmt, err := tx.Prepare(`
INSERT INTO request_attempts (id, request_id, attempt, provider, model, upstream_request_id, status_code, error_class, retry_after_seconds, reset_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
`)
	if err != nil {
		w.logf("dropping %d telemetry record(s): prepare request_attempts statement: %v", len(batch), err)
		return fmt.Errorf("prepare request_attempts statement: %w", err)
	}
	defer attStmt.Close()

	usgStmt, err := tx.Prepare(`
INSERT INTO usage (id, request_id, prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, cost)
VALUES (?, ?, ?, ?, ?, ?, ?);
`)
	if err != nil {
		w.logf("dropping %d telemetry record(s): prepare usage statement: %v", len(batch), err)
		return fmt.Errorf("prepare usage statement: %w", err)
	}
	defer usgStmt.Close()

	qtaStmt, err := tx.Prepare(`
INSERT INTO quota_observations (id, provider, label, used_pct, remaining, "limit", unit, reset_at, observed_at, source)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
`)
	if err != nil {
		w.logf("dropping %d telemetry record(s): prepare quota_observations statement: %v", len(batch), err)
		return fmt.Errorf("prepare quota_observations statement: %w", err)
	}
	defer qtaStmt.Close()

	var firstErr error
	exec := func(kind, stmtName string, stmt *sql.Stmt, args ...any) {
		if _, err := stmt.Exec(args...); err != nil {
			w.logf("telemetry %s insert failed (%s): %v", kind, stmtName, err)
			if firstErr == nil {
				firstErr = fmt.Errorf("insert %s: %w", kind, err)
			}
		}
	}

	for _, item := range batch {
		switch r := item.(type) {
		case RequestRecord:
			var idArg any = nil
			if r.ID > 0 {
				idArg = r.ID
			}
			exec("request", fmt.Sprintf("id=%d logical_id=%q", r.ID, r.LogicalID), reqStmt,
				idArg, r.ClientKeyHash, r.LogicalID, r.RequestedModel, r.ResolvedModel, r.Provider, r.CreatedAt, r.CompletedAt, r.FinishReason, r.StreamComplete, r.ErrorClass, r.TTFTMs, r.TotalMs)
		case AttemptRecord:
			var idArg any = nil
			if r.ID > 0 {
				idArg = r.ID
			}
			var reqIDArg any = nil
			if r.RequestID > 0 {
				reqIDArg = r.RequestID
			}
			exec("request_attempt", fmt.Sprintf("request_id=%d attempt=%d provider=%q", r.RequestID, r.Attempt, r.Provider), attStmt,
				idArg, reqIDArg, r.Attempt, r.Provider, r.Model, r.UpstreamRequestID, r.StatusCode, r.ErrorClass, r.RetryAfterSeconds, r.ResetAt)
		case UsageRecord:
			var idArg any = nil
			if r.ID > 0 {
				idArg = r.ID
			}
			var reqIDArg any = nil
			if r.RequestID > 0 {
				reqIDArg = r.RequestID
			}
			exec("usage", fmt.Sprintf("request_id=%d", r.RequestID), usgStmt,
				idArg, reqIDArg, r.PromptTokens, r.CompletionTokens, r.ReasoningTokens, r.CachedTokens, r.Cost)
		case QuotaObservationRecord:
			var idArg any = nil
			if r.ID > 0 {
				idArg = r.ID
			}
			exec("quota_observation", fmt.Sprintf("provider=%q label=%q", r.Provider, r.Label), qtaStmt,
				idArg, r.Provider, r.Label, r.UsedPct, r.Remaining, r.Limit, r.Unit, r.ResetAt, r.ObservedAt, r.Source)
		}
	}

	if err := tx.Commit(); err != nil {
		w.logf("telemetry commit failed for %d record(s): %v", len(batch), err)
		if firstErr == nil {
			firstErr = fmt.Errorf("commit: %w", err)
		}
	}
	return firstErr
}
