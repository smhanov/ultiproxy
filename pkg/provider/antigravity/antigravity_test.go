package antigravity

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

func TestStrictToolSchemaSanitizer(t *testing.T) {
	// Case 1: Valid schema with type object and required field defined in properties
	validSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{
				"type": "string",
			},
		},
		"required": []string{"city"},
	}
	if err := ValidateToolSchema(validSchema); err != nil {
		t.Fatalf("expected valid schema to pass, got: %v", err)
	}

	// Case 2: Reject node with properties but missing type: object
	missingTypeSchema := map[string]any{
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
	}
	if err := ValidateToolSchema(missingTypeSchema); err == nil {
		t.Fatal("expected error for schema with properties but missing type object")
	} else if !strings.Contains(err.Error(), "400") {
		t.Fatalf("expected HTTP 400 error, got: %v", err)
	}

	// Case 3: Reject node with type 'string' but having properties
	wrongTypeSchema := map[string]any{
		"type": "string",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
	}
	if err := ValidateToolSchema(wrongTypeSchema); err == nil {
		t.Fatal("expected error for non-object type with properties")
	} else if !strings.Contains(err.Error(), "400") {
		t.Fatalf("expected HTTP 400 error, got: %v", err)
	}

	// Case 4: Reject required field not defined in properties
	missingReqFieldSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
		"required": []string{"city", "country"},
	}
	if err := ValidateToolSchema(missingReqFieldSchema); err == nil {
		t.Fatal("expected error for required field not in properties")
	} else if !strings.Contains(err.Error(), "400") {
		t.Fatalf("expected HTTP 400 error, got: %v", err)
	}

	// Case 5: Nested invalid property rejected
	nestedInvalidSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{
				"required": []string{"lat"},
				// missing type: object
			},
		},
	}
	if err := ValidateToolSchema(nestedInvalidSchema); err == nil {
		t.Fatal("expected error for nested node violating rules")
	}
}

func TestUserAgentHeaderSet(t *testing.T) {
	var capturedUA string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedUA = r.Header.Get("User-Agent")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}}]
		}`))
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		StaticToken: "test-token",
		HTTPClient:  srv.Client(),
	})

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Hi"}}},
	}

	_, err := p.Generate(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if capturedUA != UserAgent {
		t.Fatalf("User-Agent = %q, want %q", capturedUA, UserAgent)
	}
}

func TestQuotaParse(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "antigravity", "quota.json")
	fixtureData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("User-Agent") != UserAgent {
			http.Error(w, "missing required UA", http.StatusForbidden)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(fixtureData)
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		StaticToken: "test-token",
		HTTPClient:  srv.Client(),
	})

	snap, err := p.Quota(context.Background())
	if err != nil {
		t.Fatalf("Quota failed: %v", err)
	}

	if len(snap.Windows) != 2 {
		t.Fatalf("expected 2 windows, got %d", len(snap.Windows))
	}

	w1 := snap.Windows[0]
	if w1.Label != "Gemini Models · 5h" {
		t.Errorf("label = %q, want 'Gemini Models · 5h'", w1.Label)
	}
	if w1.Remaining != 85.0 {
		t.Errorf("remaining = %g, want 85.0", w1.Remaining)
	}
	if w1.UsedPct != 15.0 {
		t.Errorf("usedPct = %g, want 15.0", w1.UsedPct)
	}

	w2 := snap.Windows[1]
	if w2.Label != "Claude and GPT Models · weekly" {
		t.Errorf("label = %q, want 'Claude and GPT Models · weekly'", w2.Label)
	}
	if w2.Remaining != 25.0 {
		t.Errorf("remaining = %g, want 25.0", w2.Remaining)
	}
	if w2.UsedPct != 75.0 {
		t.Errorf("usedPct = %g, want 75.0", w2.UsedPct)
	}
}

func TestStreamingThinkingDeltas(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "antigravity", "stream-thinking.sses")
	fixtureData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("User-Agent") != UserAgent {
			http.Error(w, "missing required UA", http.StatusForbidden)
			return
		}
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
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Explain reasoning"}}},
	}

	eventCh, err := p.Stream(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var reasoningDeltas []string
	var textDeltas []string
	var signatures []string
	var gotUsage bool
	var gotStop bool

	for ev := range eventCh {
		switch e := ev.(type) {
		case ir.EventReasoningDelta:
			reasoningDeltas = append(reasoningDeltas, e.Text)
		case ir.EventTextDelta:
			textDeltas = append(textDeltas, e.Text)
		case ir.EventReasoningSignature:
			signatures = append(signatures, e.Signature)
		case ir.EventUsageUpdate:
			gotUsage = true
			if e.TotalTokens != 35 {
				t.Errorf("TotalTokens = %d, want 35", e.TotalTokens)
			}
		case ir.EventMessageStop:
			gotStop = true
		}
	}

	fullReasoning := strings.Join(reasoningDeltas, "")
	expectedReasoning := "Thinking step 1: analyze requirement. Thinking step 2: formulate answer."
	if fullReasoning != expectedReasoning {
		t.Errorf("reasoning = %q, want %q", fullReasoning, expectedReasoning)
	}

	fullText := strings.Join(textDeltas, "")
	if fullText != "Here is the solution to your request." {
		t.Errorf("text = %q, want 'Here is the solution to your request.'", fullText)
	}

	if len(signatures) == 0 || signatures[0] != "sig_opaque_12345" {
		t.Errorf("expected signature 'sig_opaque_12345', got %v", signatures)
	}

	if !gotUsage {
		t.Error("expected usage event")
	}
	if !gotStop {
		t.Error("expected stop event")
	}
}

func TestStreamSynchronousErrorOnNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Forbidden without correct license", http.StatusForbidden)
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

	_, err := p.Stream(context.Background(), msgs)
	if err == nil {
		t.Fatal("expected synchronous error on 403 status code, got nil")
	}
	if !strings.Contains(err.Error(), "403") {
		t.Errorf("expected 403 in error message, got %v", err)
	}
}

func TestGenerateWithInvalidToolSchemaRejection(t *testing.T) {
	p := New(Config{
		StaticToken: "test-tok",
	})

	invalidTool := map[string]any{
		"name": "calc",
		"parameters": map[string]any{
			"properties": map[string]any{"x": map[string]any{"type": "number"}},
			// Missing type: "object"
		},
	}

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Calculate"}}},
	}

	_, err := p.Generate(context.Background(), msgs, provider.WithExtraBody(map[string]any{
		"tools": []any{invalidTool},
	}))
	if err == nil {
		t.Fatal("expected error rejecting invalid tool schema, got nil")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("expected 400 error, got: %v", err)
	}
}

func TestRegistry(t *testing.T) {
	r := provider.NewRegistry()
	p := New(Config{StaticToken: "test"})
	p.Register(r)
	got, ok := r.Get("antigravity")
	if !ok || got.Inference == nil || got.Quota == nil || got.Auth == nil {
		t.Fatalf("antigravity registry get failed: %+v", got)
	}
	if !got.Capabilities.Chat || !got.Capabilities.Tools || !got.Capabilities.Reasoning || !got.Capabilities.Streaming {
		t.Fatalf("unexpected capabilities: %+v", got.Capabilities)
	}
}
