package openaicompat

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// The upstream signals free-capacity pressure with 428 waiting_room_required;
// the official CLI queues (honoring retry-after, exponential backoff) and
// retries. The lane must do the same instead of surfacing a raw 428.

func waitingRoomStub(t *testing.T, failsBeforeOK int, retryAfter string, calls *atomic.Int32) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/agent-runs"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"runId":"run-wr-1"}`))
		case strings.HasSuffix(r.URL.Path, "/chat/completions"):
			n := calls.Add(1)
			if int(n) <= failsBeforeOK {
				if retryAfter != "" {
					w.Header().Set("Retry-After", retryAfter)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(freebuffStatusWaitingRoom)
				_, _ = w.Write([]byte(`{"error":"waiting_room_required","message":"Your free session has ended. Send your message again to start a new one."}`))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"wr-1","choices":[{"index":0,"message":{"role":"assistant","content":"WR PONG"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		default:
			http.NotFound(w, r)
		}
	}))
}

func freebuffWRActor() *fakeAdoptingActor {
	a := &fakeAdoptingActor{minted: "fb-wr-inst"}
	return a
}

// TestOpenAICompat_FreebuffWaitingRoomRetry: a single 428 is retried and the
// chat succeeds once capacity admits the request.
func TestOpenAICompat_FreebuffWaitingRoomRetry(t *testing.T) {
	var calls atomic.Int32
	server := waitingRoomStub(t, 1, "1", &calls)
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-token",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       freebuffWRActor(),
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	resp, err := p.Generate(context.Background(), msgs, provider.WithModel("glm"))
	if err != nil {
		t.Fatalf("Generate should succeed after waiting-room retry, got: %v", err)
	}
	if !strings.Contains(respText(resp), "WR PONG") {
		t.Errorf("unexpected response: %v", respText(resp))
	}
	if calls.Load() != 2 {
		t.Errorf("expected exactly 2 chat attempts (1x428 + 1xOK), got %d", calls.Load())
	}
}

// TestOpenAICompat_FreebuffWaitingRoomExhausted: persistent capacity pressure
// surfaces an honest error after bounded retries (llmhub's fixed policy: 4
// retries + initial = 5 attempts) — not an infinite loop, and the error
// carries the upstream diagnosis.
func TestOpenAICompat_FreebuffWaitingRoomExhausted(t *testing.T) {
	var calls atomic.Int32
	server := waitingRoomStub(t, 100, "1", &calls)
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-token",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       freebuffWRActor(),
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	_, err = p.Generate(context.Background(), msgs, provider.WithModel("glm"))
	if err == nil {
		t.Fatal("expected an error after exhausted retries")
	}
	if !strings.Contains(err.Error(), "waiting_room_required") {
		t.Errorf("error = %v, want a waiting-room diagnosis", err)
	}
	if calls.Load() != 5 {
		t.Errorf("expected 5 bounded attempts (1 + 4 retries), got %d", calls.Load())
	}
}

func respText(r *ir.Response) string {
	if r == nil || r.Message == nil {
		return ""
	}
	var sb strings.Builder
	for _, b := range r.Message.Blocks {
		if tb, ok := b.(ir.TextBlock); ok {
			sb.WriteString(tb.Text)
		}
	}
	return sb.String()
}
