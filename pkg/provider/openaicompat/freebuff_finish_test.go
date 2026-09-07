package openaicompat

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

type fakeFinishAndHeartbeatActor struct {
	fakeAdoptingActor
	actingUserID   string
	mu             sync.Mutex
	finishedRuns   []string
	finishStatuses []string
	heartbeatCount int
}

func (f *fakeFinishAndHeartbeatActor) ActingUserID(ctx context.Context) string {
	return f.actingUserID
}

func (f *fakeFinishAndHeartbeatActor) StartRun(ctx context.Context, agentID string) (any, error) {
	return "run-test-xyz", nil
}

func (f *fakeFinishAndHeartbeatActor) FinishRun(ctx context.Context, runID, status string, err error) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.finishedRuns = append(f.finishedRuns, runID)
	f.finishStatuses = append(f.finishStatuses, status)
	return nil
}

func (f *fakeFinishAndHeartbeatActor) StartHeartbeat() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.heartbeatCount++
}

func TestFreebuff_FinishRunAndHeartbeat_Generate(t *testing.T) {
	var mu sync.Mutex
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var payload map[string]any
		_ = json.Unmarshal(body, &payload)
		switch {
		case strings.HasSuffix(r.URL.Path, "/me"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"usr-999","email":"test@test.com"}`))
		case strings.HasSuffix(r.URL.Path, "/chat/completions"):
			mu.Lock()
			gotBody = payload
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"1","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	actor := &fakeFinishAndHeartbeatActor{actingUserID: "usr-999"}
	actor.minted = "fb-inst-test"
	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-key",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       actor,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "ping"}}}}
	// Request carries stream_options in extra body to verify stripping
	opts := []provider.Option{
		provider.WithModel("mimo"),
		provider.WithExtraBody(map[string]any{
			"stream_options": map[string]any{"include_usage": true},
		}),
	}

	_, err = p.Generate(context.Background(), msgs, opts...)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	actor.mu.Lock()
	finCount := len(actor.finishedRuns)
	lastRun := ""
	lastStatus := ""
	if finCount > 0 {
		lastRun = actor.finishedRuns[0]
		lastStatus = actor.finishStatuses[0]
	}
	hbCount := actor.heartbeatCount
	actor.mu.Unlock()

	if finCount != 1 {
		t.Errorf("expected 1 FinishRun call, got %d", finCount)
	}
	if lastRun != "run-test-xyz" {
		t.Errorf("expected FinishRun runId=run-test-xyz, got %q", lastRun)
	}
	if lastStatus != "completed" {
		t.Errorf("expected FinishRun status=completed, got %q", lastStatus)
	}
	if hbCount < 1 {
		t.Errorf("expected StartHeartbeat to be called, got %d", hbCount)
	}

	mu.Lock()
	reqBody := gotBody
	mu.Unlock()

	if _, exists := reqBody["stream_options"]; exists {
		t.Errorf("expected stream_options to be stripped from freebuff request body, got: %#v", reqBody["stream_options"])
	}
}

func TestFreebuff_FinishRun_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/me"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"usr-999","email":"test@test.com"}`))
		case strings.HasSuffix(r.URL.Path, "/chat/completions"):
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"P\"}}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ONG\"}}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	actor := &fakeFinishAndHeartbeatActor{actingUserID: "usr-999"}
	actor.minted = "fb-inst-test"
	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-key",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       actor,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "ping"}}}}
	ch, err := p.Stream(context.Background(), msgs, provider.WithModel("mimo"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	// Drain stream
	for range ch {
	}

	actor.mu.Lock()
	finCount := len(actor.finishedRuns)
	lastRun := ""
	lastStatus := ""
	if finCount > 0 {
		lastRun = actor.finishedRuns[0]
		lastStatus = actor.finishStatuses[0]
	}
	actor.mu.Unlock()

	if finCount != 1 {
		t.Errorf("expected 1 FinishRun call on stream completion, got %d", finCount)
	}
	if lastRun != "run-test-xyz" {
		t.Errorf("expected FinishRun runId=run-test-xyz, got %q", lastRun)
	}
	if lastStatus != "completed" {
		t.Errorf("expected FinishRun status=completed, got %q", lastStatus)
	}
}
