package xai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestGRPCWebFrameDecodeFixture(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "xai", "credits.grpcweb.bin")
	fixtureData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	snap, err := ParseGrokCreditsResponse(fixtureData, time.Unix(1788000000, 0).UTC())
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse failed: %v", err)
	}

	if len(snap.Windows) < 2 {
		t.Fatalf("expected at least 2 windows (5 hour and Weekly), got %d: %+v", len(snap.Windows), snap.Windows)
	}

	w5h := snap.Windows[0]
	if w5h.Label != "5 hour" {
		t.Errorf("w5h.Label = %q, want '5 hour'", w5h.Label)
	}
	if w5h.UsedPct != 15.0 {
		t.Errorf("w5h.UsedPct = %g, want 15.0", w5h.UsedPct)
	}
	if w5h.Remaining != 85.0 {
		t.Errorf("w5h.Remaining = %g, want 85.0", w5h.Remaining)
	}

	wWeekly := snap.Windows[1]
	if wWeekly.Label != "Weekly" {
		t.Errorf("wWeekly.Label = %q, want 'Weekly'", wWeekly.Label)
	}
	if wWeekly.UsedPct != 45.0 {
		t.Errorf("wWeekly.UsedPct = %g, want 45.0", wWeekly.UsedPct)
	}
	if wWeekly.Remaining != 55.0 {
		t.Errorf("wWeekly.Remaining = %g, want 55.0", wWeekly.Remaining)
	}
}

func TestStreamingReasoningOrder(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "xai", "stream-reasoning.sses")
	fixtureData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(fixtureData)
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		StaticToken: "test-token",
		HTTPClient:  srv.Client(),
	})

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Explain reasoning order"}}},
	}

	eventCh, err := p.Stream(context.Background(), msgs, provider.WithModel("grok-beta"))
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var kinds []string
	var reasoningText, outputText string
	for ev := range eventCh {
		kinds = append(kinds, ev.EventKind())
		switch e := ev.(type) {
		case ir.EventReasoningDelta:
			reasoningText += e.Text
		case ir.EventTextDelta:
			outputText += e.Text
		}
	}

	if len(kinds) == 0 {
		t.Fatal("expected events, got none")
	}

	// Verify reasoning_delta comes BEFORE text_delta
	firstReasoningIdx := -1
	firstTextIdx := -1
	for idx, k := range kinds {
		if k == "reasoning_delta" && firstReasoningIdx == -1 {
			firstReasoningIdx = idx
		}
		if k == "text_delta" && firstTextIdx == -1 {
			firstTextIdx = idx
		}
	}

	if firstReasoningIdx == -1 {
		t.Fatal("expected at least one reasoning_delta")
	}
	if firstTextIdx == -1 {
		t.Fatal("expected at least one text_delta")
	}
	if firstReasoningIdx >= firstTextIdx {
		t.Fatalf("reasoning_delta (idx %d) must precede text_delta (idx %d)", firstReasoningIdx, firstTextIdx)
	}

	if reasoningText != "Thinking deeply about the answer..." {
		t.Errorf("reasoningText = %q, want 'Thinking deeply about the answer...'", reasoningText)
	}
	if outputText != "Answer begins with clear logic." {
		t.Errorf("outputText = %q, want 'Answer begins with clear logic.'", outputText)
	}

	// Verify usage is delivered
	lastKind := kinds[len(kinds)-1]
	secondLastKind := kinds[len(kinds)-2]
	if lastKind != "usage_update" && secondLastKind != "usage_update" {
		t.Errorf("expected usage_update at end, got %v", kinds)
	}
}

func TestEffortPassthroughInPayload(t *testing.T) {
	var capturedBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"id": "grok-123",
			"choices": [{"message": {"role": "assistant", "content": "Done"}}]
		}`))
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		StaticToken: "test-tok",
		HTTPClient:  srv.Client(),
	})

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Solve problem"}}},
	}

	for _, effort := range []string{"low", "medium", "high", "xhigh"} {
		_, err := p.Generate(context.Background(), msgs,
			provider.WithModel("grok-3"),
			provider.WithReasoningEffort(effort),
		)
		if err != nil {
			t.Fatalf("Generate with effort %q failed: %v", effort, err)
		}

		if capturedBody["reasoning_effort"] != effort {
			t.Errorf("reasoning_effort = %v, want %q", capturedBody["reasoning_effort"], effort)
		}
	}
}

func TestStreamSynchronousErrorOnNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Unauthorized xAI key", http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		StaticToken: "test-tok",
		HTTPClient:  srv.Client(),
	})

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Hello"}}},
	}

	_, err := p.Stream(context.Background(), msgs, provider.WithModel("grok-beta"))
	if err == nil {
		t.Fatal("expected synchronous error on 401 status code, got nil")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("expected 401 in error message, got %v", err)
	}
}

func TestTolerantErrorHandlingOnBadBilling(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	p := New(Config{
		BillingURL:  srv.URL,
		StaticToken: "test-tok",
		HTTPClient:  srv.Client(),
	})

	snap, err := p.Quota(context.Background())
	if err != nil {
		t.Fatalf("Quota should tolerate non-200 and not return hard error: %v", err)
	}
	if !strings.Contains(snap.Detail, "unknown") && !strings.Contains(snap.Detail, "503") {
		t.Errorf("expected detail mentioning unknown status or 503, got %q", snap.Detail)
	}
}

func TestRegistry(t *testing.T) {
	r := provider.NewRegistry()
	p := New(Config{StaticToken: "test"})
	p.Register(r)
	got, ok := r.Get("xai")
	if !ok || got.Inference == nil || got.Quota == nil || got.Auth == nil {
		t.Fatalf("xai registry get failed: %+v", got)
	}
	if !got.Capabilities.Chat || !got.Capabilities.Tools || !got.Capabilities.Reasoning || !got.Capabilities.Streaming {
		t.Fatalf("unexpected capabilities: %+v", got.Capabilities)
	}
}
