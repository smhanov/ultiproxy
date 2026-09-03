package copilot

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestDetermineInitiator(t *testing.T) {
	// Case 1: First proxied human turn -> "user"
	userMsgs := []*ir.Message{
		{Role: "system", Blocks: []ir.Block{ir.TextBlock{Text: "You are a helpful assistant"}}},
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Hello"}}},
	}
	if got := DetermineInitiator(userMsgs, nil); got != "user" {
		t.Errorf("expected 'user', got %q", got)
	}

	// Case 2: Subsequent agent continuation with prior assistant message -> "agent"
	agentMsgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Hello"}}},
		{Role: "assistant", Blocks: []ir.Block{ir.TextBlock{Text: "Hi there!"}}},
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "What can you do?"}}},
	}
	if got := DetermineInitiator(agentMsgs, nil); got != "agent" {
		t.Errorf("expected 'agent', got %q", got)
	}

	// Case 3: Continuation with tool result message -> "agent"
	toolMsgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Run command"}}},
		{Role: "tool", Blocks: []ir.Block{ir.ToolResultBlock{ToolCallID: "call_1", Content: "output"}}},
	}
	if got := DetermineInitiator(toolMsgs, nil); got != "agent" {
		t.Errorf("expected 'agent', got %q", got)
	}

	// Case 4: Explicit client-supplied header OVERRIDES inference to "user"
	headersOverride := map[string]string{"X-Ultiproxy-Initiator": "user"}
	if got := DetermineInitiator(agentMsgs, headersOverride); got != "user" {
		t.Errorf("expected 'user' with header override, got %q", got)
	}

	// Case 5: Explicit message meta override
	metaMsgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Hello"}}},
		{Role: "assistant", Blocks: []ir.Block{ir.TextBlock{Text: "Hi"}}},
		{
			Role:   "user",
			Blocks: []ir.Block{ir.TextBlock{Text: "Follow-up"}},
			Meta:   map[string]string{"x-initiator": "user"},
		},
	}
	if got := DetermineInitiator(metaMsgs, nil); got != "user" {
		t.Errorf("expected 'user' with meta override, got %q", got)
	}
}

func TestEditorHeadersAndInitiatorPresent(t *testing.T) {
	var capturedHeaders http.Header

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"id": "resp_test_1",
			"choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}]
		}`))
	}))
	defer srv.Close()

	p := New(Config{
		Token:      "gho_testtoken123",
		BaseURL:    srv.URL,
		HTTPClient: srv.Client(),
	})

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Hello"}}},
	}

	resp, err := p.Generate(context.Background(), msgs, provider.WithModel("gpt-4o"))
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp == nil || resp.ID != "resp_test_1" {
		t.Fatalf("unexpected response: %+v", resp)
	}

	// Verify mandatory editor headers
	if got := capturedHeaders.Get("Editor-Version"); got != EditorVersion {
		t.Errorf("Editor-Version = %q, want %q", got, EditorVersion)
	}
	if got := capturedHeaders.Get("Editor-Plugin-Version"); got != EditorPluginVersion {
		t.Errorf("Editor-Plugin-Version = %q, want %q", got, EditorPluginVersion)
	}
	if got := capturedHeaders.Get("Copilot-Integration-Id"); got != CopilotIntegrationID {
		t.Errorf("Copilot-Integration-Id = %q, want %q", got, CopilotIntegrationID)
	}
	if got := capturedHeaders.Get("User-Agent"); got != CopilotUserAgent {
		t.Errorf("User-Agent = %q, want %q", got, CopilotUserAgent)
	}
	if got := capturedHeaders.Get("Authorization"); got != "Bearer gho_testtoken123" {
		t.Errorf("Authorization = %q, want 'Bearer gho_testtoken123'", got)
	}
	if got := capturedHeaders.Get("X-Initiator"); got != "user" {
		t.Errorf("X-Initiator = %q, want 'user'", got)
	}
}

func TestResponsesBridgeRoundTripOnFixture(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "copilot", "responses-stream.sses")
	fixtureData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/responses") {
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(fixtureData)
	}))
	defer srv.Close()

	p := New(Config{
		Token:      "gho_testtok",
		BaseURL:    srv.URL,
		HTTPClient: srv.Client(),
	})

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Solve this"}}},
	}

	// Model gpt-5.4 routes to /responses
	eventsCh, err := p.Stream(context.Background(), msgs, provider.WithModel("gpt-5.4"))
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var events []ir.Event
	for ev := range eventsCh {
		events = append(events, ev)
	}

	if len(events) == 0 {
		t.Fatal("expected streamed events, got none")
	}

	// Check that we got message_start, reasoning_delta, text_delta, usage_update, message_stop
	var kinds []string
	var reasoningText, outputText string
	for _, ev := range events {
		kinds = append(kinds, ev.EventKind())
		switch e := ev.(type) {
		case ir.EventReasoningDelta:
			reasoningText += e.Text
		case ir.EventTextDelta:
			outputText += e.Text
		}
	}

	if reasoningText != "Thinking through solution." {
		t.Errorf("reasoningText = %q, want 'Thinking through solution.'", reasoningText)
	}
	if outputText != "Hello from Copilot!" {
		t.Errorf("outputText = %q, want 'Hello from Copilot!'", outputText)
	}

	// Check non-streaming responses parsing
	nonStreamJSON := `{
		"id": "resp_non_stream_01",
		"model": "gpt-5.4",
		"status": "completed",
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "Non-streaming answer"}]
			}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
	}`
	resp, err := TranslateResponsesJSONToResponse([]byte(nonStreamJSON))
	if err != nil {
		t.Fatalf("TranslateResponsesJSONToResponse failed: %v", err)
	}
	if resp.ID != "resp_non_stream_01" {
		t.Errorf("resp.ID = %q, want resp_non_stream_01", resp.ID)
	}
	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(resp.Message.Blocks))
	}
	tb, ok := resp.Message.Blocks[0].(ir.TextBlock)
	if !ok || tb.Text != "Non-streaming answer" {
		t.Errorf("unexpected block: %+v", resp.Message.Blocks[0])
	}
}

func TestQuotaParse(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "copilot", "quota-user.json")
	fixtureData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "token gho_testtoken" {
			http.Error(w, "bad auth header", http.StatusUnauthorized)
			return
		}
		if r.Header.Get("User-Agent") != CopilotUserAgent {
			http.Error(w, "missing user agent", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(fixtureData)
	}))
	defer srv.Close()

	p := New(Config{
		Token:        "gho_testtoken",
		QuotaBaseURL: srv.URL,
		HTTPClient:   srv.Client(),
	})

	snap, err := p.Quota(context.Background())
	if err != nil {
		t.Fatalf("Quota failed: %v", err)
	}

	if snap == nil {
		t.Fatal("expected non-nil snapshot")
	}

	if len(snap.Windows) < 3 {
		t.Fatalf("expected at least 3 windows, got %d", len(snap.Windows))
	}

	premWin := snap.Windows[0]
	if premWin.Label != "Premium requests" {
		t.Errorf("label = %q, want 'Premium requests'", premWin.Label)
	}
	if premWin.Remaining != 280 {
		t.Errorf("remaining = %g, want 280", premWin.Remaining)
	}
	if premWin.Limit != 300 {
		t.Errorf("limit = %g, want 300", premWin.Limit)
	}
	// 20/300 used = ~6.67%
	if premWin.UsedPct < 6.6 || premWin.UsedPct > 6.7 {
		t.Errorf("usedPct = %g, want ~6.67", premWin.UsedPct)
	}
	if !strings.Contains(snap.Detail, "Premium 280/300 remaining") {
		t.Errorf("detail = %q, want containing 'Premium 280/300 remaining'", snap.Detail)
	}
}

func TestStreamSynchronousErrorOnNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Quota exceeded", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	p := New(Config{
		Token:      "gho_testtok",
		BaseURL:    srv.URL,
		HTTPClient: srv.Client(),
	})

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Hello"}}},
	}

	_, err := p.Stream(context.Background(), msgs, provider.WithModel("gpt-4o"))
	if err == nil {
		t.Fatal("expected synchronous error on 429 status code, got nil")
	}
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("expected 429 in error message, got %v", err)
	}
}

func TestRegistry(t *testing.T) {
	r := provider.NewRegistry()
	p := New(Config{Token: "test"})
	p.Register(r)
	got, ok := r.Get("copilot")
	if !ok || got.Inference == nil || got.Quota == nil || got.Auth == nil {
		t.Fatalf("copilot registry get failed: %+v", got)
	}
	if !got.Capabilities.Chat || !got.Capabilities.Tools || !got.Capabilities.Reasoning || !got.Capabilities.Streaming || got.Capabilities.Vision {
		t.Fatalf("unexpected capabilities: %+v", got.Capabilities)
	}
}
