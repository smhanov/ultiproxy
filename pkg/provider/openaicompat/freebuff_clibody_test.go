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

// fakeInstActor supplies a fixed instance id so metadata assertions are deterministic.
type fakeInstActor struct {
	fakeActingUser
	instID string
}

func (f *fakeInstActor) InstanceID() string { return f.instID }
func (f *fakeInstActor) ActingUserID(ctx context.Context) string {
	return f.actingUserID
}

// TestFreebuff_ChatBodyMatchesCLI pins the freebuff chat body to the official CLI's
// captured wire shape (mitmdump capture 2026-09-06, /tmp/cli_chat_ground_truth.json).
// Upstream rejects (428 waiting_room_required) bodies that deviate from this shape.
func TestFreebuff_ChatBodyMatchesCLI(t *testing.T) {
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

	actor := &fakeInstActor{fakeActingUser: fakeActingUser{actingUserID: "usr-2222"}, instID: "inst-777"}
	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "fb-tok",
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
	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("mimo")); err != nil {
		t.Fatalf("Generate: %v", err)
	}

	mu.Lock()
	h, body := gotHeaders, gotBody
	mu.Unlock()
	if h == nil || body == nil {
		t.Fatal("chat request not observed")
	}

	// --- UA: the current CLI's ai-sdk factory string (captured 2026-09-06).
	if got := h.Get("User-Agent"); got != "ai-sdk/openai-compatible/0.0.0-test/codebuff ai-sdk/provider-utils/3.0.25 runtime/browser" {
		t.Errorf("User-Agent = %q, want the CLI 0.0.0-test composite UA", got)
	}

	// --- NO x-freebuff-instance-id header on chat (metadata carries the instance).
	if got := h.Get("X-Freebuff-Instance-Id"); got != "" {
		t.Errorf("X-Freebuff-Instance-Id header = %q, want absent (CLI does not send it on chat)", got)
	}

	// --- codebuff_metadata: instance id + trace id now ride the BODY.
	meta, _ := body["codebuff_metadata"].(map[string]any)
	if meta == nil {
		t.Fatal("codebuff_metadata missing")
	}
	for _, k := range []string{"freebuff_instance_id", "trace_session_id", "repo_snapshot", "llm_step_number", "run_id", "client_id", "cost_mode"} {
		if v, _ := meta[k].(string); v == "" {
			t.Errorf("codebuff_metadata.%s missing/empty, want set (CLI sends it)", k)
		}
	}
	if got := meta["freebuff_instance_id"]; got != "inst-777" {
		t.Errorf("metadata.freebuff_instance_id = %v, want the actor's instance id", got)
	}
	if got := meta["cost_mode"]; got != "free" {
		t.Errorf("metadata.cost_mode = %v, want free", got)
	}

	// --- provider: data_collection deny (NOT allow_fallbacks).
	prov, _ := body["provider"].(map[string]any)
	if prov == nil {
		t.Fatal("provider missing")
	}
	if got := prov["data_collection"]; got != "deny" {
		t.Errorf("provider.data_collection = %v, want \"deny\"", got)
	}
	if _, has := prov["allow_fallbacks"]; has {
		t.Errorf("provider.allow_fallbacks present; CLI sends provider={data_collection:deny} only")
	}

	// --- NO stream / stop fields: llmhub auto-adds stream_options on streams
	// (the CLI never sends stream_options), so the lane uses the non-stream
	// body — live-verified 2026-09-06 (200 FLORB via curl replay).
	if _, has := body["stream"]; has {
		t.Errorf("stream present; lane must use the non-stream body")
	}
	if _, has := body["stop"]; has {
		t.Errorf("stop present; the proven non-stream variant omits it")
	}

	// --- tool_choice: auto.
	if got := body["tool_choice"]; got != "auto" {
		t.Errorf("tool_choice = %v, want \"auto\"", got)
	}

	// --- NO temperature (CLI does not set it).
	if _, has := body["temperature"]; has {
		t.Errorf("temperature present; CLI omits it")
	}

	// --- tools: the 15 CLI tools as the base (request carried none).
	tools, _ := body["tools"].([]any)
	wantTools := []string{"read_files", "str_replace", "write_file", "run_terminal_command",
		"code_search", "glob", "list_directory", "write_todos", "web_search", "read_url",
		"ask_user", "suggest_followups", "gravity_index", "render_ui", "skill"}
	if len(tools) != len(wantTools) {
		t.Fatalf("tools count = %d, want %d (the CLI's toolset)", len(tools), len(wantTools))
	}
	for i, tv := range tools {
		tm, _ := tv.(map[string]any)
		fn, _ := tm["function"].(map[string]any)
		if fn["name"] != wantTools[i] {
			t.Errorf("tools[%d] name = %v, want %s", i, fn["name"], wantTools[i])
		}
	}

	// --- caller tools are APPENDED after the CLI base (arbitrary tools must
	// survive the lane; replacing the base 404s at upstream).
	msgs2 := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	extra := map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "get_weather",
			"description": "w",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
			},
		},
	}
	if _, err := p.Generate(context.Background(), msgs2,
		provider.WithModel("mimo"),
		provider.WithExtraBody(map[string]any{"tools": []any{extra}})); err != nil {
		t.Fatalf("Generate with caller tools: %v", err)
	}
	mu.Lock()
	freshBody := gotBody
	mu.Unlock()
	tools2, _ := freshBody["tools"].([]any)
	if len(tools2) != len(wantTools)+1 {
		t.Fatalf("caller tools appended: got %d tools, want %d", len(tools2), len(wantTools)+1)
	}
	last, _ := tools2[len(tools2)-1].(map[string]any)
	lfn, _ := last["function"].(map[string]any)
	if lfn["name"] != "get_weather" {
		t.Errorf("last tool = %v, want get_weather appended last", lfn["name"])
	}
}
