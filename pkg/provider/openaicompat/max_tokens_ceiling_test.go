package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// chatStub returns a server that captures the last chat completion payload.
func chatPayloadRecorder(t *testing.T) (*httptest.Server, func() map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var last map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var p map[string]any
		_ = json.Unmarshal(body, &p)
		mu.Lock()
		last = p
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-1",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
			},
		})
	}))
	t.Cleanup(srv.Close)
	return srv, func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return last
	}
}

// TestOpenAICompat_MaxTokensByModelIsACeiling (AC2): max_tokens_by_model is an
// upper bound the upstream accepts, not a default that only applies when the
// client omits max_tokens. A client asking for more than the lane can honor
// must be clamped to the matched limit instead of being forwarded verbatim.
func TestOpenAICompat_MaxTokensByModelIsACeiling(t *testing.T) {
	srv, lastPayload := chatPayloadRecorder(t)

	p, err := New(Config{
		BaseURL:    srv.URL,
		APIKey:     "k",
		HTTPClient: srv.Client(),
		Quirks:     Quirks{MaxTokensByModel: map[string]int{"glm": 8192}},
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}

	if _, err := p.Generate(context.Background(), msgs,
		provider.WithModel("glm"), provider.WithMaxTokens(100000)); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if mt, ok := lastPayload()["max_tokens"].(float64); !ok || int(mt) != 8192 {
		t.Fatalf("client max_tokens 100000 against max_tokens_by_model glm:8192 must be sent as 8192, got %v", lastPayload()["max_tokens"])
	}

	// Below the ceiling the client's own value wins.
	if _, err := p.Generate(context.Background(), msgs,
		provider.WithModel("glm"), provider.WithMaxTokens(512)); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if mt, ok := lastPayload()["max_tokens"].(float64); !ok || int(mt) != 512 {
		t.Fatalf("client max_tokens 512 under the ceiling must pass through, got %v", lastPayload()["max_tokens"])
	}

	// No client value at all still falls back to the lane ceiling.
	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("glm")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if mt, ok := lastPayload()["max_tokens"].(float64); !ok || int(mt) != 8192 {
		t.Fatalf("omitted max_tokens must default to the lane ceiling 8192, got %v", lastPayload()["max_tokens"])
	}
}

// TestOpenAICompat_MaxTokensByModelDeterministicWinner (AC3): overlapping
// patterns must resolve to one documented winner regardless of Go map
// iteration order.
//
// Rule: an exact model match wins; otherwise the longest pattern contained in
// the model id wins; equal-length patterns are broken lexicographically
// (ascending) so the result never depends on map ordering.
func TestOpenAICompat_MaxTokensByModelDeterministicWinner(t *testing.T) {
	// Same map declared with opposite insertion orders.
	orderA := map[string]int{}
	orderA["glm"] = 4096
	orderA["glm-5.3"] = 8192

	orderB := map[string]int{}
	orderB["glm-5.3"] = 8192
	orderB["glm"] = 4096

	newProvider := func(limits map[string]int) *Provider {
		t.Helper()
		p, err := New(Config{Quirks: Quirks{MaxTokensByModel: limits}})
		if err != nil {
			t.Fatalf("New failed: %v", err)
		}
		return p
	}

	// Repeat the probe: Go randomizes map iteration per runtime, so a
	// nondeterministic matcher flips within a handful of rounds.
	for round := 0; round < 50; round++ {
		for name, limits := range map[string]map[string]int{"A": orderA, "B": orderB} {
			p := newProvider(limits)

			// "glm-5.3-flash" contains both "glm" and "glm-5.3" -> longest wins.
			if got := p.resolveMaxTokens("glm-5.3-flash", 0); got != 8192 {
				t.Fatalf("round %d order %s: resolveMaxTokens(glm-5.3-flash) = %d, want 8192 (longest pattern glm-5.3)", round, name, got)
			}
			// Exact match wins.
			if got := p.resolveMaxTokens("glm", 0); got != 4096 {
				t.Fatalf("round %d order %s: resolveMaxTokens(glm) = %d, want 4096 (exact match glm)", round, name, got)
			}
			// Unrelated model: no ceiling at all.
			if got := p.resolveMaxTokens("kimi-k3", 0); got != 0 {
				t.Fatalf("round %d order %s: resolveMaxTokens(kimi-k3) = %d, want 0 (no match)", round, name, got)
			}
		}
	}

	// Equal-length overlapping patterns resolve lexicographically (glm < gpt).
	tie := newProvider(map[string]int{"glm": 111, "gpt": 222})
	for i := 0; i < 25; i++ {
		if got := tie.resolveMaxTokens("glm-and-gpt", 0); got != 111 {
			t.Fatalf("equal-length tie must resolve to the lexicographically smaller pattern: got %d, want 111", got)
		}
	}

	// The ceiling also clamps client values for the deterministic winner.
	p := newProvider(orderA)
	if got := p.resolveMaxTokens("glm-5.3-flash", 100000); got != 8192 {
		t.Fatalf("clamped winner = %d, want 8192", got)
	}
}

// TestOpenAICompat_FreebuffCancelReleasesActor (AC4): a cancelled Freebuff
// request must release the serialized actor even when the caller stopped
// draining the event channel. Regression: the forwarding goroutine blocked on
// `actorCh <- ev` and never re-checked ctx.Done(), so the lane's single-slot
// lock stayed held forever (every later freebuff request deadlocked).
func TestOpenAICompat_FreebuffCancelReleasesActor(t *testing.T) {
	release := make(chan struct{})
	var once sync.Once

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		// More events than the forwarding channel's buffer, so the provider
		// goroutine is guaranteed to be parked on the send when we cancel.
		for i := 0; i < 500; i++ {
			fmt.Fprintf(w, "data: {\"id\":\"fb-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n")
		}
		if flusher != nil {
			flusher.Flush()
		}
		<-r.Context().Done()
		once.Do(func() { close(release) })
	}))
	defer srv.Close()

	actor := &fakeActor{instanceID: "fb-inst-1"}
	p, err := New(Config{
		BaseURL:    srv.URL,
		APIKey:     "k",
		HTTPClient: srv.Client(),
		Quirks:     Quirks{FreebuffActor: actor, FreebuffDefaultTool: true},
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := p.Stream(ctx, []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	_ = ch // abandoned on purpose: nobody drains the channel

	// Abandon the stream: nobody drains the channel.
	time.Sleep(200 * time.Millisecond)
	cancel()

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		actor.mu.Lock()
		active := actor.active
		actor.mu.Unlock()
		if active == 0 {
			// The lane must be usable again: a second acquire succeeds.
			ctx2, cancel2 := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel2()
			ch2, err := p.Stream(ctx2, []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "again"}}}})
			if err != nil {
				t.Fatalf("Stream after cancel failed: %v", err)
			}
			for range ch2 {
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("actor lock still held 3s after context cancellation (stuck lock)")
}
