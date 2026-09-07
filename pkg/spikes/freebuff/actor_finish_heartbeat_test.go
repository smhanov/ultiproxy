package freebuff

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestActor_FinishRun_DispatchesCorrectShape(t *testing.T) {
	var finishPayload map[string]any
	var authHeader, userHeader string
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch r.URL.Path {
		case "/me":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"usr-test-123"}`))
		case "/agent-runs":
			if r.Method == "POST" {
				authHeader = r.Header.Get("Authorization")
				userHeader = r.Header.Get("x-freebuff-acting-user-id")
				_ = json.NewDecoder(r.Body).Decode(&finishPayload)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"status":"ok"}`))
				return
			}
			http.NotFound(w, r)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	actor, err := NewFreebuffAccountActor("", server.Client(), "test-token", WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("NewFreebuffAccountActor: %v", err)
	}
	defer actor.Close()

	err = actor.FinishRun(context.Background(), "run-abc-123", "completed", nil)
	if err != nil {
		t.Fatalf("FinishRun failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	if authHeader != "Bearer test-token" {
		t.Errorf("expected Authorization Bearer test-token, got %q", authHeader)
	}
	if userHeader != "usr-test-123" {
		t.Errorf("expected x-freebuff-acting-user-id usr-test-123, got %q", userHeader)
	}
	if finishPayload["action"] != "FINISH" {
		t.Errorf("expected action=FINISH, got %v", finishPayload["action"])
	}
	if finishPayload["runId"] != "run-abc-123" {
		t.Errorf("expected runId=run-abc-123, got %v", finishPayload["runId"])
	}
	if finishPayload["status"] != "completed" {
		t.Errorf("expected status=completed, got %v", finishPayload["status"])
	}
	if finishPayload["totalSteps"] != float64(1) {
		t.Errorf("expected totalSteps=1, got %v", finishPayload["totalSteps"])
	}
}

func TestActor_Heartbeat_CompactSessionHeaderAndInterval(t *testing.T) {
	var sessionGetCount int32
	var lastCompactHeader string
	var lastInstanceHeader string
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/freebuff/session" && r.Method == "GET" {
			atomic.AddInt32(&sessionGetCount, 1)
			mu.Lock()
			lastCompactHeader = r.Header.Get("x-freebuff-compact-session")
			lastInstanceHeader = r.Header.Get("x-freebuff-instance-id")
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"status": "active",
				"instanceId": "test-inst-1",
				"model": "mimo/mimo-v2.5",
				"remainingMs": 3600000
			}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	actor, err := NewFreebuffAccountActor("", server.Client(), "test-token",
		WithBaseURL(server.URL),
		WithInstanceID("test-inst-1"),
		WithHeartbeatInterval(50*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("NewFreebuffAccountActor: %v", err)
	}
	defer actor.Close()

	actor.StartHeartbeat()

	// Wait for at least 2 heartbeat ticks
	deadline := time.Now().Add(1 * time.Second)
	for time.Now().Before(deadline) {
		if atomic.LoadInt32(&sessionGetCount) >= 2 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	actor.StopHeartbeat()

	count := atomic.LoadInt32(&sessionGetCount)
	if count < 2 {
		t.Fatalf("expected at least 2 session GET heartbeat calls, got %d", count)
	}

	mu.Lock()
	defer mu.Unlock()
	if lastCompactHeader != "1" {
		t.Errorf("expected x-freebuff-compact-session: 1, got %q", lastCompactHeader)
	}
	if lastInstanceHeader != "test-inst-1" {
		t.Errorf("expected x-freebuff-instance-id: test-inst-1, got %q", lastInstanceHeader)
	}
}
