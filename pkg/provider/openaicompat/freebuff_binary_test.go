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

	// Headers: authorization, CLI composite UA, acting user id. The instance
	// header is ABSENT on chat (the current CLI carries the instance in
	// codebuff_metadata instead; captured 2026-09-06).
	if got := h.Get("Authorization"); got != "Bearer tok-1" {
		t.Errorf("Authorization = %q", got)
	}
	if got := h.Get("User-Agent"); got != freebuffCLIUserAgent {
		t.Errorf("User-Agent = %q, want the CLI composite UA", got)
	}
	if got := h.Get("x-freebuff-acting-user-id"); got != "usr-2222" {
		t.Errorf("x-freebuff-acting-user-id = %q, want the runtime /me id", got)
	}
	if got := h.Get("x-freebuff-instance-id"); got != "" {
		t.Errorf("x-freebuff-instance-id on chat = %q, want ABSENT (metadata carries it)", got)
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
	// The instance id now rides the metadata (captured CLI truth).
	if got, _ := meta["freebuff_instance_id"].(string); got != "fb-inst-1" {
		t.Errorf("freebuff_instance_id = %q, want the actor's instance (metadata carries it)", got)
	}
	if got, _ := meta["trace_session_id"].(string); got == "" {
		t.Error("trace_session_id missing; CLI sends it")
	}
	if got, _ := meta["repo_snapshot"].(string); !strings.Contains(got, "gitAvailable") {
		t.Errorf("repo_snapshot = %q, want the CLI's stringified repo stats JSON", got)
	}

	// provider: {data_collection:deny} — the CLI's shape (captured).
	prov, _ := body["provider"].(map[string]any)
	if prov == nil {
		t.Fatal("missing provider object")
	}
	if got, _ := prov["data_collection"].(string); got != "deny" {
		t.Errorf("provider.data_collection = %v, want \"deny\"", prov["data_collection"])
	}
	if _, ok := prov["allow_fallbacks"]; ok {
		t.Error("provider.allow_fallbacks must be absent (CLI sends data_collection only)")
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
