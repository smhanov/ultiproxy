package freebuff

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestLockContentionAndTimeout(t *testing.T) {
	tempDir := t.TempDir()
	lockPath := filepath.Join(tempDir, "freebuff.lock")

	actor1, err := NewFreebuffAccountActor(lockPath, nil, "token-1")
	if err != nil {
		t.Fatalf("failed to create actor 1: %v", err)
	}
	defer actor1.Close()

	actor2, err := NewFreebuffAccountActor(lockPath, nil, "token-2")
	if err != nil {
		t.Fatalf("failed to create actor 2: %v", err)
	}
	defer actor2.Close()

	// Actor 1 acquires
	if err := actor1.Acquire(context.Background()); err != nil {
		t.Fatalf("actor 1 failed to acquire: %v", err)
	}

	// Goroutine 2 contends with a short timeout -> must time out cleanly
	ctxTimeout, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	err2 := actor2.Acquire(ctxTimeout)
	duration := time.Since(start)

	if err2 == nil {
		t.Fatalf("actor 2 should not have acquired lock held by actor 1")
	}
	if !errors.Is(err2, context.DeadlineExceeded) && !errors.Is(err2, ErrLeaseConflict) {
		t.Errorf("expected DeadlineExceeded or lease conflict, got %v", err2)
	}
	if duration < 30*time.Millisecond {
		t.Errorf("expected wait around 50ms, got %v", duration)
	}
}

func TestLeaseConflictRejection(t *testing.T) {
	tempDir := t.TempDir()
	lockPath := filepath.Join(tempDir, "freebuff.lock")

	actor1, err := NewFreebuffAccountActor(lockPath, nil, "token-1")
	if err != nil {
		t.Fatalf("failed to create actor 1: %v", err)
	}
	defer actor1.Close()

	actor2, err := NewFreebuffAccountActor(lockPath, nil, "token-2")
	if err != nil {
		t.Fatalf("failed to create actor 2: %v", err)
	}
	defer actor2.Close()

	// Actor 1 acquires lock
	if err := actor1.TryAcquire(); err != nil {
		t.Fatalf("actor 1 failed to acquire lock: %v", err)
	}

	// Actor 2 attempts non-blocking acquire on same lock file -> rejected with ErrLeaseConflict
	err = actor2.TryAcquire()
	if !errors.Is(err, ErrLeaseConflict) {
		t.Fatalf("expected ErrLeaseConflict, got %v", err)
	}

	// Now actor 1 releases
	if err := actor1.Release(); err != nil {
		t.Fatalf("actor 1 release error: %v", err)
	}

	// Now actor 2 can acquire
	if err := actor2.TryAcquire(); err != nil {
		t.Fatalf("actor 2 failed to acquire after actor 1 released: %v", err)
	}
}

func TestTwoGoroutinesContendOneWaitsOneAcquires(t *testing.T) {
	tempDir := t.TempDir()
	lockPath := filepath.Join(tempDir, "contend.lock")

	actor1, err := NewFreebuffAccountActor(lockPath, nil, "t1")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer actor1.Close()

	actor2, err := NewFreebuffAccountActor(lockPath, nil, "t2")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer actor2.Close()

	if err := actor1.Acquire(context.Background()); err != nil {
		t.Fatalf("actor 1 acquire failed: %v", err)
	}

	var actor2Acquired atomic.Bool
	var wg sync.WaitGroup
	wg.Add(1)

	// Goroutine for actor2 waiting up to 500ms
	go func() {
		defer wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()
		if err := actor2.Acquire(ctx); err == nil {
			actor2Acquired.Store(true)
			_ = actor2.Release()
		}
	}()

	// Hold lock briefly, then release
	time.Sleep(60 * time.Millisecond)
	_ = actor1.Release()

	wg.Wait()

	if !actor2Acquired.Load() {
		t.Errorf("expected actor 2 to acquire lock once actor 1 released it")
	}
}

func TestActorLifecycleAndEndpoints(t *testing.T) {
	var sessionCalled, bindCalled, runCalled, streamCalled atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/freebuff/session":
			sessionCalled.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(Session{
				InstanceID: "inst-999",
				Model:      "claude-3-5-sonnet",
			})

		case r.Method == http.MethodPost && r.URL.Path == "/freebuff/session":
			bindCalled.Add(1)
			var payload map[string]string
			_ = json.NewDecoder(r.Body).Decode(&payload)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(Session{
				InstanceID: "inst-999",
				Model:      payload["model"],
			})

		case r.Method == http.MethodPost && r.URL.Path == "/agent-runs":
			runCalled.Add(1)
			var payload map[string]string
			_ = json.NewDecoder(r.Body).Decode(&payload)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(AgentRun{
				RunID:      "run-001",
				AgentID:    payload["agent_id"],
				InstanceID: payload["instance_id"],
				Status:     "running",
			})

		case r.Method == http.MethodPost && r.URL.Path == "/agent-runs/stream":
			streamCalled.Add(1)
			body, _ := io.ReadAll(r.Body)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("stream-response-for:" + string(body)))

		case r.Method == http.MethodDelete && r.URL.Path == "/freebuff/session":
			w.WriteHeader(http.StatusOK)

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	tempDir := t.TempDir()
	lockPath := filepath.Join(tempDir, "freebuff.lock")

	actor, err := NewFreebuffAccountActor(
		lockPath,
		server.Client(),
		"mock-token",
		WithBaseURL(server.URL),
		WithQueueCapacity(5),
	)
	if err != nil {
		t.Fatalf("failed to create actor: %v", err)
	}
	defer actor.Close()

	// 1. Reconcile
	if err := actor.Reconcile(); err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}
	if actor.InstanceID() != "inst-999" {
		t.Errorf("expected inst-999, got %s", actor.InstanceID())
	}
	if actor.BoundModel() != "claude-3-5-sonnet" {
		t.Errorf("expected claude-3-5-sonnet, got %s", actor.BoundModel())
	}

	// 2. Bind
	if err := actor.Bind("gpt-4o"); err != nil {
		t.Fatalf("bind failed: %v", err)
	}
	if actor.BoundModel() != "gpt-4o" {
		t.Errorf("expected gpt-4o, got %s", actor.BoundModel())
	}

	// 3. StartRun
	run, err := actor.StartRun("agent-alpha")
	if err != nil {
		t.Fatalf("start run failed: %v", err)
	}
	if run.RunID != "run-001" || run.AgentID != "agent-alpha" {
		t.Errorf("unexpected run info: %+v", run)
	}

	// 4. Stream FIFO through queue
	if actor.QueueSize() != 5 {
		t.Errorf("expected queue capacity 5, got %d", actor.QueueSize())
	}

	rc, err := actor.Stream(strings.NewReader("query-1"))
	if err != nil {
		t.Fatalf("stream failed: %v", err)
	}
	respBytes, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		t.Fatalf("failed to read stream resp: %v", err)
	}
	if string(respBytes) != "stream-response-for:query-1" {
		t.Errorf("unexpected stream response: %s", string(respBytes))
	}

	// 5. Delete session
	if err := actor.DeleteSession(); err != nil {
		t.Fatalf("delete session failed: %v", err)
	}
	if actor.InstanceID() != "" || actor.BoundModel() != "" {
		t.Errorf("expected cleared session after delete")
	}
}

func TestStreamQueueCancellationAndBounded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()

	tempDir := t.TempDir()
	lockPath := filepath.Join(tempDir, "freebuff.lock")

	// Queue size 2
	actor, err := NewFreebuffAccountActor(
		lockPath,
		server.Client(),
		"tok",
		WithBaseURL(server.URL),
		WithQueueCapacity(2),
	)
	if err != nil {
		t.Fatalf("failed to create actor: %v", err)
	}
	defer actor.Close()

	// Fill queue with canceled request
	ctxCancel, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err = actor.Stream(ctxCancel, strings.NewReader("canceled"))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}
