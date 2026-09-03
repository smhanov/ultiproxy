package storage

import (
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
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

	mu     sync.Mutex
	closed bool
	done   chan struct{}
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

func (w *Writer) enqueue(item any) error {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return ErrClosed
	}
	w.mu.Unlock()

	select {
	case w.queue <- item:
		return nil
	default:
		return ErrQueueFull
	}
}

// TrackRequest enqueues a request record non-blockingly.
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
	return w.db.Close()
}

func (w *Writer) worker() {
	defer close(w.done)

	batch := make([]any, 0, w.batchSize)

	for {
		item, ok := <-w.queue
		if !ok {
			// Queue closed: write any remaining in batch
			if len(batch) > 0 {
				w.writeBatch(batch)
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
					w.writeBatch(batch)
					return
				}
				batch = append(batch, nextItem)
			default:
				break drainLoop
			}
		}

		w.writeBatch(batch)
		batch = batch[:0]
	}
}

func (w *Writer) writeBatch(batch []any) {
	if len(batch) == 0 {
		return
	}

	tx, err := w.db.Begin()
	if err != nil {
		return
	}
	defer tx.Rollback()

	reqStmt, _ := tx.Prepare(`
INSERT INTO requests (id, client_key_hash, logical_id, requested_model, resolved_model, provider, created_at, completed_at, finish_reason, stream_complete, error_class, ttft_ms, total_ms)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
`)
	if reqStmt != nil {
		defer reqStmt.Close()
	}

	attStmt, _ := tx.Prepare(`
INSERT INTO request_attempts (id, request_id, attempt, provider, model, upstream_request_id, status_code, error_class, retry_after_seconds, reset_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
`)
	if attStmt != nil {
		defer attStmt.Close()
	}

	usgStmt, _ := tx.Prepare(`
INSERT INTO usage (id, request_id, prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, cost)
VALUES (?, ?, ?, ?, ?, ?, ?);
`)
	if usgStmt != nil {
		defer usgStmt.Close()
	}

	qtaStmt, _ := tx.Prepare(`
INSERT INTO quota_observations (id, provider, label, used_pct, remaining, "limit", unit, reset_at, observed_at, source)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
`)
	if qtaStmt != nil {
		defer qtaStmt.Close()
	}

	for _, item := range batch {
		switch r := item.(type) {
		case RequestRecord:
			if reqStmt != nil {
				var idArg any = nil
				if r.ID > 0 {
					idArg = r.ID
				}
				_, _ = reqStmt.Exec(idArg, r.ClientKeyHash, r.LogicalID, r.RequestedModel, r.ResolvedModel, r.Provider, r.CreatedAt, r.CompletedAt, r.FinishReason, r.StreamComplete, r.ErrorClass, r.TTFTMs, r.TotalMs)
			}
		case AttemptRecord:
			if attStmt != nil {
				var idArg any = nil
				if r.ID > 0 {
					idArg = r.ID
				}
				var reqIDArg any = nil
				if r.RequestID > 0 {
					reqIDArg = r.RequestID
				}
				_, _ = attStmt.Exec(idArg, reqIDArg, r.Attempt, r.Provider, r.Model, r.UpstreamRequestID, r.StatusCode, r.ErrorClass, r.RetryAfterSeconds, r.ResetAt)
			}
		case UsageRecord:
			if usgStmt != nil {
				var idArg any = nil
				if r.ID > 0 {
					idArg = r.ID
				}
				var reqIDArg any = nil
				if r.RequestID > 0 {
					reqIDArg = r.RequestID
				}
				_, _ = usgStmt.Exec(idArg, reqIDArg, r.PromptTokens, r.CompletionTokens, r.ReasoningTokens, r.CachedTokens, r.Cost)
			}
		case QuotaObservationRecord:
			if qtaStmt != nil {
				var idArg any = nil
				if r.ID > 0 {
					idArg = r.ID
				}
				_, _ = qtaStmt.Exec(idArg, r.Provider, r.Label, r.UsedPct, r.Remaining, r.Limit, r.Unit, r.ResetAt, r.ObservedAt, r.Source)
			}
		}
	}

	_ = tx.Commit()
}
