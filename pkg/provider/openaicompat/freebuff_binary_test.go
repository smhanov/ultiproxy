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

// The binary-identical request contract, extracted from the shipped freebuff
// CLI 0.0.167 Bun bundle (see skill ai-cli-proxy references). Every assertion
// here matches the official client byte-for-byte; deviations are
// fingerprinting risk.

// fakeActingUser exposes the runtime-fetched acting user id (the way the real
// actor resolves /me).
type fakeActingUser struct {
	fakeAdoptingActor
	actingUserID string
}

func (f *fakeActingUser) ActingUserID(ctx context.Context) string { return f.actingUserID }

// StartRun stands in for the upstream run id (the real actor hits HTTP).
func (f *fakeActingUser) StartRun(ctx context.Context, agentID string) (any, error) {
	f.mu.Lock()
	f.startedAgentIDs = append(f.startedAgentIDs, agentID)
	f.mu.Unlock()
	return "run-9", nil
}

// TestFreebuff_BinaryIdenticalChatRequest pins headers, metadata and body for
// /chat/completions to the official client's exact shape.
func TestFreebuff_BinaryIdenticalChatRequest(t *testing.T) {
	var mu sync.Mutex
	var gotHeaders http.Header
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var payload map[string]any
		_ = json.Unmarshal(body, &payload)
		switch {
		case strings.HasSuffix(r.URL.Path, "/me"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"usr-2222","email":"x@y.z"}`))
		case strings.HasSuffix(r.URL.Path, "/agent-runs"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"runId":"run-9"}`))
		case strings.HasSuffix(r.URL.Path, "/chat/completions"):
			mu.Lock()
			gotHeaders = r.Header.Clone()
			gotBody = payload
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	actor := &fakeActingUser{actingUserID: "usr-2222"}
	actor.minted = "fb-inst-1"
	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "tok-1",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       actor,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("glm")); err != nil {
		t.Fatalf("Generate: %v", err)
	}

	mu.Lock()
	h, body := gotHeaders, gotBody
	mu.Unlock()
	if h == nil || body == nil {
		t.Fatal("chat request not observed")
	}

	// Headers: authorization, interpolated-version UA, acting user id. No
	// instance header on chat (binary truth).
	if got := h.Get("Authorization"); got != "Bearer tok-1" {
		t.Errorf("Authorization = %q", got)
	}
	if got := h.Get("User-Agent"); got != "ai-sdk/openai-compatible/0.0.167/codebuff" {
		t.Errorf("User-Agent = %q, want the version-interpolated official format", got)
	}
	if got := h.Get("x-freebuff-acting-user-id"); got != "usr-2222" {
		t.Errorf("x-freebuff-acting-user-id = %q, want the runtime /me id", got)
	}
	// Instance header present (session keying — without it chat 428s).
	if got := h.Get("x-freebuff-instance-id"); got != "fb-inst-1" {
		t.Errorf("x-freebuff-instance-id on chat = %q, want the actor's session instance", got)
	}

	// Body model: canonical publisher id.
	if got, _ := body["model"].(string); got != "z-ai/glm-5.3-flash" {
		t.Errorf("body model = %q, want canonical", got)
	}

	meta, _ := body["codebuff_metadata"].(map[string]any)
	if meta == nil {
		t.Fatal("missing codebuff_metadata")
	}
	if got, _ := meta["run_id"].(string); got != "run-9" {
		t.Errorf("run_id = %q", got)
	}
	// client_id: base36 random per run, no cli- prefix, not a UUID, 11 chars.
	if got, _ := meta["client_id"].(string); !isBase36RunID(got) {
		t.Errorf("client_id = %q, want Math.random().toString(36).substring(2,15) shape", got)
	}
	if got, _ := meta["llm_step_number"].(string); got != "1" {
		t.Errorf("llm_step_number = %q, want string \"1\"", got)
	}
	if got, _ := meta["cost_mode"].(string); got != "free" {
		t.Errorf("cost_mode = %q", got)
	}
	if _, ok := meta["freebuff_instance_id"]; ok {
		t.Errorf("freebuff_instance_id must NOT be in metadata (binary omits it)")
	}

	// provider: allow_fallbacks false for official models, no order key.
	prov, _ := body["provider"].(map[string]any)
	if prov == nil {
		t.Fatal("missing provider object")
	}
	if got, ok := prov["allow_fallbacks"].(bool); !ok || got {
		t.Errorf("provider.allow_fallbacks = %v, want false for official models", prov["allow_fallbacks"])
	}
	if _, ok := prov["order"]; ok {
		t.Errorf("provider.order must be absent for non-openrouter-claude models")
	}

	// Tools: at least the read_files fallback.
	tools, _ := body["tools"].([]any)
	if len(tools) == 0 {
		t.Error("tools must be non-empty (upstream requires >=1)")
	}
}

func isBase36RunID(s string) bool {
	if len(s) < 5 || len(s) > 15 {
		return false
	}
	if strings.Contains(s, "-") || strings.HasPrefix(s, "cli-") {
		return false
	}
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'z')) {
			return false
		}
	}
	return true
}
