package antigravity

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

func antigravityRecorder(t *testing.T) (*httptest.Server, func() map[string]any) {
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
		_, _ = w.Write([]byte(`{"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}}`))
	}))
	t.Cleanup(srv.Close)
	return srv, func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return last
	}
}

// TestGenerate_ImageBlocksForwarded (AC5): image content blocks must reach the
// Antigravity upstream request. Regression: buildRequest had no image case, so
// vision prompts lost their images.
func TestGenerate_ImageBlocksForwarded(t *testing.T) {
	srv, lastPayload := antigravityRecorder(t)
	p := New(Config{BaseURL: srv.URL, StaticToken: "tok", HTTPClient: srv.Client()})

	dataURL := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "What is in these?"},
				ir.ImageBlock{URL: dataURL, Detail: "high"},
				ir.ImageBlock{URL: "https://example.com/photo.jpg"},
			},
		},
	}

	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("gemini-3.7-flash-high")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := lastPayload()
	inner, _ := payload["request"].(map[string]any)
	if inner == nil {
		t.Fatalf("request payload missing: %v", payload)
	}
	contents, _ := inner["contents"].([]any)
	if len(contents) != 1 {
		t.Fatalf("expected 1 content entry, got %v", inner["contents"])
	}
	first, _ := contents[0].(map[string]any)
	parts, _ := first["parts"].([]any)
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts (text + 2 images), got %d: %v", len(parts), parts)
	}

	// Data URLs become inlineData (base64) parts.
	p1, _ := parts[1].(map[string]any)
	inline, _ := p1["inlineData"].(map[string]any)
	if inline == nil {
		t.Fatalf("expected inlineData on part 1, got %v", parts[1])
	}
	if inline["mimeType"] != "image/png" || inline["data"] != "iVBORw0KGgoAAAANSUhEUg==" {
		t.Errorf("inlineData = %v, want mimeType image/png with the base64 payload", inline)
	}

	// Remote URLs become fileData parts.
	p2, _ := parts[2].(map[string]any)
	fileData, _ := p2["fileData"].(map[string]any)
	if fileData == nil {
		t.Fatalf("expected fileData on part 2, got %v", parts[2])
	}
	if fileData["fileUri"] != "https://example.com/photo.jpg" {
		t.Errorf("fileData.fileUri = %v, want the remote URL", fileData)
	}
}

// TestGenerate_SystemRolePreserved (AC6): system messages must not be rewritten
// as user turns. Regression: every non-assistant role (including system) was
// mapped to role "user", silently corrupting the prompt shape.
func TestGenerate_SystemRolePreserved(t *testing.T) {
	srv, lastPayload := antigravityRecorder(t)
	p := New(Config{BaseURL: srv.URL, StaticToken: "tok", HTTPClient: srv.Client()})

	msgs := []*ir.Message{
		{Role: "system", Blocks: []ir.Block{ir.TextBlock{Text: "You are a terse coding agent."}}},
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hello"}}},
	}

	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("gemini-3.7-flash-high")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := lastPayload()
	inner, _ := payload["request"].(map[string]any)
	if inner == nil {
		t.Fatalf("request payload missing: %v", payload)
	}

	contents, _ := inner["contents"].([]any)
	if len(contents) != 1 {
		t.Fatalf("expected the system turn to be lifted out of contents, got %v", inner["contents"])
	}
	only, _ := contents[0].(map[string]any)
	if role, _ := only["role"].(string); role != "user" {
		t.Fatalf("contents[0].role = %q, want user (the only remaining turn)", role)
	}

	sys, ok := inner["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("system prompt must map to the protocol system-instruction field, got %v", inner["systemInstruction"])
	}
	sysParts, _ := sys["parts"].([]any)
	if len(sysParts) != 1 {
		t.Fatalf("expected 1 system part, got %v", sys["parts"])
	}
	sp0, _ := sysParts[0].(map[string]any)
	if sp0["text"] != "You are a terse coding agent." {
		t.Errorf("systemInstruction text = %v, want the system prompt", sp0)
	}

	// No content entry may claim the system role as a user turn.
	for i, c := range contents {
		cm, _ := c.(map[string]any)
		if r, _ := cm["role"].(string); strings.EqualFold(r, "system") {
			t.Errorf("contents[%d] carries role %q: the system turn belongs in systemInstruction", i, r)
		}
	}
}

// TestGenerate_MultipleSystemMessagesMerge: several system turns merge into one
// systemInstruction in order, and the assistant role still maps to "model".
func TestGenerate_MultipleSystemMessagesMerge(t *testing.T) {
	srv, lastPayload := antigravityRecorder(t)
	p := New(Config{BaseURL: srv.URL, StaticToken: "tok", HTTPClient: srv.Client()})

	msgs := []*ir.Message{
		{Role: "system", Blocks: []ir.Block{ir.TextBlock{Text: "first"}}},
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "q"}}},
		{Role: "assistant", Blocks: []ir.Block{ir.TextBlock{Text: "a"}}},
		{Role: "system", Blocks: []ir.Block{ir.TextBlock{Text: "second"}}},
	}

	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("gemini-3.7-flash-high")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	inner, _ := lastPayload()["request"].(map[string]any)
	sys, _ := inner["systemInstruction"].(map[string]any)
	if sys == nil {
		t.Fatalf("systemInstruction missing: %v", inner)
	}
	parts, _ := sys["parts"].([]any)
	if len(parts) != 2 {
		t.Fatalf("expected 2 merged system parts, got %v", sys["parts"])
	}
	if p0, _ := parts[0].(map[string]any); p0["text"] != "first" {
		t.Errorf("system part 0 = %v, want first", parts[0])
	}
	if p1, _ := parts[1].(map[string]any); p1["text"] != "second" {
		t.Errorf("system part 1 = %v, want second", parts[1])
	}

	contents, _ := inner["contents"].([]any)
	if len(contents) != 2 {
		t.Fatalf("expected 2 remaining turns (user + model), got %v", inner["contents"])
	}
	if r, _ := contents[0].(map[string]any)["role"].(string); r != "user" {
		t.Errorf("contents[0].role = %q, want user", r)
	}
	if r, _ := contents[1].(map[string]any)["role"].(string); r != "model" {
		t.Errorf("contents[1].role = %q, want model", r)
	}
}
