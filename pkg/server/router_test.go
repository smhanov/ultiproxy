package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
)

// --- fixtures ---------------------------------------------------------------

// newLaneRegistry registers one fake inference lane per name.
func newLaneRegistry(names ...string) (*provider.Registry, map[string]*fakeInferenceProvider) {
	registry := provider.NewRegistry()
	lanes := make(map[string]*fakeInferenceProvider, len(names))
	for _, name := range names {
		lane := &fakeInferenceProvider{name: name}
		lanes[name] = lane
		registry.Register(provider.Provider{Inference: lane})
	}
	return registry, lanes
}

// newAliasLifecycleServer builds a server with one fake lane ("vllm"), the
// given config aliases and a state manager, so alias-lifecycle tests drive the
// real router + catalog + state + MCP wiring end to end.
func newAliasLifecycleServer(t *testing.T, aliases map[string]ModelAlias) (*Server, *fakeInferenceProvider) {
	t.Helper()

	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.Server.Models = aliases

	lane := &fakeInferenceProvider{name: "vllm"}
	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: lane})

	srv := NewServer(cfg, registry, WithStateManager(state.NewStateManager()))
	return srv, lane
}

// mcpToolCall posts one JSON-RPC tools/call request to the server's MCP
// surface and returns the concatenated tool result text.
func mcpToolCall(t *testing.T, srv *Server, id int, tool, args string) string {
	t.Helper()

	body := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":"tools/call","params":{"name":%q,"arguments":%s}}`, id, tool, args)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("%s: HTTP %d: %s", tool, rec.Code, rec.Body.String())
	}

	var resp struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
		Result struct {
			IsError bool `json:"isError"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("%s: decode response: %v (%s)", tool, err, rec.Body.String())
	}
	if resp.Error != nil {
		t.Fatalf("%s: JSON-RPC error: %s", tool, resp.Error.Message)
	}
	var text strings.Builder
	for _, c := range resp.Result.Content {
		text.WriteString(c.Text)
	}
	if resp.Result.IsError {
		t.Fatalf("%s: tool reported an error: %s", tool, text.String())
	}
	return text.String()
}

// chat posts a non-streaming chat completion for model and returns status +
// body, along with the error "type" field when the body carries one.
func chat(t *testing.T, srv *Server, model string) (int, string, string) {
	t.Helper()

	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"Hi"}]}`, model)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	errType := ""
	var parsed struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(rec.Body.Bytes(), &parsed) == nil && parsed.Error.Type != "" {
		errType = parsed.Error.Type
	}
	return rec.Code, rec.Body.String(), errType
}

func (f *fakeInferenceProvider) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// --- AC1: disabled aliases must not route ----------------------------------

// TestRouter_DisabledStateEntryStopsRouting proves the router treats a state
// entry with Enabled=false as terminal: no catalog fallback, no provider-name
// prefix fallback. This is what makes toggle_model(false) an actual kill
// switch instead of a discovery hint.
func TestRouter_DisabledStateEntryStopsRouting(t *testing.T) {
	registry, lanes := newLaneRegistry("vllm")

	sm := state.NewStateManager()
	sm.Update(func(snap *state.RuntimeSnapshot) {
		snap.Models["alias-disabled"] = state.ModelRuntime{ID: "alias-disabled", Provider: "vllm", Enabled: false}
		snap.Models["vllm/lane-disabled"] = state.ModelRuntime{ID: "vllm/lane-disabled", Provider: "vllm", Enabled: false}
	})

	// The catalog still knows the alias, and the lane name is a prefix of the
	// second id: both fallbacks must be suppressed by the disabled entry.
	catalog, err := NewModelCatalog(map[string]ModelAlias{
		"alias-disabled": {Provider: "vllm", Upstream: "Qwen/Qwen3.8-Instruct-AWQ"},
	}, "")
	if err != nil {
		t.Fatalf("NewModelCatalog: %v", err)
	}

	router := NewRegistryRouter(registry, sm, catalog)
	for _, model := range []string{"alias-disabled", "vllm/lane-disabled"} {
		name, err := router.Route(context.Background(), model)
		if err == nil {
			t.Fatalf("model %q: expected routing to be refused, got lane %q", model, name)
		}
		var uerr *UnknownModelError
		if !errors.As(err, &uerr) {
			t.Errorf("model %q: error must classify as unknown_model, got %T %v", model, err, err)
		}
	}

	for name, lane := range lanes {
		if got := lane.callCount(); got != 0 {
			t.Errorf("lane %q was called %d times for disabled models", name, got)
		}
	}
}

// TestRouter_DisabledAliasEndToEnd is AC1 on the wire: toggle_model(false)
// over MCP, then POST /v1/chat/completions with that alias must answer
// 404 unknown_model with zero lane calls - not a successful route.
func TestRouter_DisabledAliasEndToEnd(t *testing.T) {
	srv, lane := newAliasLifecycleServer(t, map[string]ModelAlias{
		"qwenpoint-3.8": {Provider: "vllm", Upstream: "Qwen/Qwen3.8-Instruct-AWQ"},
	})

	// Baseline: the alias routes, proving the failure below is caused by the
	// toggle and not by a broken fixture.
	if code, body, _ := chat(t, srv, "qwenpoint-3.8"); code != http.StatusOK {
		t.Fatalf("baseline chat with enabled alias: got %d (%s)", code, body)
	}
	if got := lane.callCount(); got != 1 {
		t.Fatalf("baseline lane calls = %d, want 1", got)
	}

	mcpToolCall(t, srv, 1, "toggle_model", `{"model_id":"qwenpoint-3.8","enabled":false}`)

	code, body, errType := chat(t, srv, "qwenpoint-3.8")
	if code != http.StatusNotFound {
		t.Fatalf("chat with disabled alias: got %d (%s), want 404", code, body)
	}
	if errType != "unknown_model" {
		t.Fatalf("chat with disabled alias: error type = %q, want unknown_model (%s)", errType, body)
	}
	if got := lane.callCount(); got != 1 {
		t.Fatalf("lane calls after disable = %d, want 1 (no additional dispatch)", got)
	}

	// Re-enabling must restore routing: the toggle is reversible by contract.
	mcpToolCall(t, srv, 2, "toggle_model", `{"model_id":"qwenpoint-3.8","enabled":true}`)
	if code, body, _ := chat(t, srv, "qwenpoint-3.8"); code != http.StatusOK {
		t.Fatalf("chat after re-enable: got %d (%s), want 200", code, body)
	}
}

// --- AC2: removing an alias must purge it immediately ----------------------

// TestRouter_RemovedAliasStopsRoutingWithoutRestart is AC2 on the wire:
// remove_model_alias over MCP, then an immediate request for that alias
// answers 404 unknown_model in the same process (no restart, no other
// state-mutating call in between).
func TestRouter_RemovedAliasStopsRoutingWithoutRestart(t *testing.T) {
	srv, lane := newAliasLifecycleServer(t, map[string]ModelAlias{
		"keep": {Provider: "vllm", Upstream: "Qwen/Qwen3.8-Instruct-AWQ"},
	})

	mcpToolCall(t, srv, 1, "set_model_alias", `{"alias":"drain-me","provider":"vllm","upstream":"Qwen/Qwen3.8-Instruct-AWQ"}`)
	if code, body, _ := chat(t, srv, "drain-me"); code != http.StatusOK {
		t.Fatalf("baseline chat with runtime alias: got %d (%s)", code, body)
	}
	before := lane.callCount()

	mcpToolCall(t, srv, 2, "remove_model_alias", `{"alias":"drain-me"}`)

	code, body, errType := chat(t, srv, "drain-me")
	if code != http.StatusNotFound {
		t.Fatalf("chat with removed alias: got %d (%s), want 404", code, body)
	}
	if errType != "unknown_model" {
		t.Fatalf("chat with removed alias: error type = %q, want unknown_model (%s)", errType, body)
	}
	if got := lane.callCount(); got != before {
		t.Fatalf("lane calls after removal = %d, want %d", got, before)
	}

	// The stale entry must not linger in the router's first resolution source.
	if _, ok := srv.sm.Snapshot().Models["drain-me"]; ok {
		t.Fatal("removed alias still present in the state snapshot")
	}
	if _, ok := srv.catalog.Get("drain-me"); ok {
		t.Fatal("removed alias still present in the catalog")
	}

	// Discovery agrees: the removed alias is no longer advertised.
	for id := range getModels(t, srv).ids() {
		if id == "drain-me" {
			t.Fatal("GET /v1/models still advertises the removed alias")
		}
	}

	// An unrelated alias keeps routing.
	if code, body, _ := chat(t, srv, "keep"); code != http.StatusOK {
		t.Fatalf("chat with surviving alias: got %d (%s)", code, body)
	}
}

// --- AC3: administering one alias must not re-enable another ---------------

// TestRouter_EditingAliasKeepsOtherDisabled is AC3: disable alias A, edit
// alias B (and remove a third alias), then A stays disabled.
func TestRouter_EditingAliasKeepsOtherDisabled(t *testing.T) {
	srv, lane := newAliasLifecycleServer(t, map[string]ModelAlias{
		"alias-a": {Provider: "vllm", Upstream: "Qwen/Qwen3.8-Instruct-AWQ"},
		"alias-b": {Provider: "vllm", Upstream: "meta-llama/Llama-3-8B"},
		"alias-c": {Provider: "vllm", Upstream: "microsoft/phi-4"},
	})

	mcpToolCall(t, srv, 1, "toggle_model", `{"model_id":"alias-a","enabled":false}`)
	if code, _, _ := chat(t, srv, "alias-a"); code != http.StatusNotFound {
		t.Fatalf("alias-a before edits: got %d, want 404", code)
	}

	// Editing another alias must not flip alias-a back on.
	mcpToolCall(t, srv, 2, "set_model_alias", `{"alias":"alias-b","provider":"vllm","upstream":"meta-llama/Llama-3.3-70B","max_output":512}`)

	snap := srv.sm.Snapshot()
	if m, ok := snap.Models["alias-a"]; !ok || m.Enabled {
		t.Fatalf("alias-a must stay disabled after editing alias-b, got %+v (present=%v)", m, ok)
	}
	if m, ok := snap.Models["alias-b"]; !ok || !m.Enabled {
		t.Fatalf("alias-b must stay enabled, got %+v (present=%v)", m, ok)
	}
	if m, ok := snap.Models["alias-b"]; ok && m.MaxOutput != 512 {
		t.Fatalf("alias-b edit did not land: %+v", m)
	}

	// Removing a third alias must not re-enable alias-a either.
	mcpToolCall(t, srv, 3, "remove_model_alias", `{"alias":"alias-c"}`)
	if m, ok := srv.sm.Snapshot().Models["alias-a"]; !ok || m.Enabled {
		t.Fatalf("alias-a must stay disabled after removing alias-c, got %+v (present=%v)", m, ok)
	}

	code, body, errType := chat(t, srv, "alias-a")
	if code != http.StatusNotFound || errType != "unknown_model" {
		t.Fatalf("chat with alias-a after administering other aliases: got %d %q (%s), want 404 unknown_model", code, errType, body)
	}
	if got := lane.callCount(); got != 0 {
		t.Fatalf("lane calls = %d, want 0 (alias-a never dispatched)", got)
	}

	// The edited alias still routes.
	if code, body, _ := chat(t, srv, "alias-b"); code != http.StatusOK {
		t.Fatalf("chat with edited alias-b: got %d (%s)", code, body)
	}
}

// --- AC4: lane matching is exact id or "<lane>/" prefix, never a substring --

func TestRouter_LaneMatchIsExactOrPrefix(t *testing.T) {
	registry, lanes := newLaneRegistry("zai", "vllm")
	router := NewRegistryRouter(registry, nil, nil)

	cases := []struct {
		model string
		want  string // "" = expect unknown_model
	}{
		{model: "zai", want: "zai"},                         // exact lane id
		{model: "zai/glm-5.3-flash", want: "zai"},           // genuine prefix
		{model: "ZAI/glm-5.3-flash", want: "zai"},           // case-insensitive prefix
		{model: "vllm", want: "vllm"},                       // exact lane id
		{model: "vllm/Qwen/Qwen3.8-Instruct", want: "vllm"}, // nested upstream id
		{model: "amazai-gpt-4o", want: ""},                  // substring of "zai", not a prefix
		{model: "prevllm/qwen", want: ""},                   // substring of "vllm", not a prefix
		{model: "xaizai", want: ""},                         // contains "zai" twice, still no prefix
		{model: "zai-model", want: ""},                      // dash is not a separator
	}

	for _, tc := range cases {
		got, err := router.Route(context.Background(), tc.model)
		if tc.want == "" {
			var uerr *UnknownModelError
			if err == nil {
				t.Errorf("model %q: expected unknown_model, routed to %q", tc.model, got)
			} else if !errors.As(err, &uerr) {
				t.Errorf("model %q: expected unknown_model, got %v", tc.model, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("model %q: unexpected error: %v", tc.model, err)
			continue
		}
		if got != tc.want {
			t.Errorf("model %q: routed to %q, want %q", tc.model, got, tc.want)
		}
	}

	for name, lane := range lanes {
		if got := lane.callCount(); got != 0 {
			t.Errorf("lane %q was called %d times by the router", name, got)
		}
	}
}

// TestRouter_SubstringModelIdDoesNotReachLane is AC4 end to end: a model id
// that merely contains a lane name answers 404 unknown_model and the lane
// never sees the request (so it cannot receive an unstripped model name).
func TestRouter_SubstringModelIdDoesNotReachLane(t *testing.T) {
	srv, lane := newAliasLifecycleServer(t, nil)

	code, body, errType := chat(t, srv, "prevllm/qwen")
	if code != http.StatusNotFound || errType != "unknown_model" {
		t.Fatalf("substring model id: got %d %q (%s), want 404 unknown_model", code, errType, body)
	}
	if got := lane.callCount(); got != 0 {
		t.Fatalf("lane calls = %d, want 0", got)
	}

	// Control: the genuine prefix form of the same lane does route.
	if code, body, _ := chat(t, srv, "vllm/Qwen/Qwen3.8-Instruct"); code != http.StatusOK {
		t.Fatalf("prefixed lane model: got %d (%s), want 200", code, body)
	}
}

// --- persistence / lifecycle -------------------------------------------------

// TestRouter_AliasLifecycleAcrossRestart pins what survives a restart:
// removing an alias is permanent (the catalog no longer knows it, so a fresh
// process never installs it again), while the disabled flag is memory-only by
// contract and therefore re-enables on restart.
func TestRouter_AliasLifecycleAcrossRestart(t *testing.T) {
	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.Server.Models = map[string]ModelAlias{
		"config-alias": {Provider: "vllm", Upstream: "Qwen/Qwen3.8-Instruct-AWQ"},
	}

	boot := func() (*Server, *fakeInferenceProvider) {
		lane := &fakeInferenceProvider{name: "vllm"}
		registry := provider.NewRegistry()
		registry.Register(provider.Provider{Inference: lane})
		return NewServer(cfg, registry, WithStateManager(state.NewStateManager())), lane
	}

	srv1, _ := boot()
	mcpToolCall(t, srv1, 1, "set_model_alias", `{"alias":"runtime-alias","provider":"vllm","upstream":"meta-llama/Llama-3-8B"}`)
	mcpToolCall(t, srv1, 2, "set_model_alias", `{"alias":"doomed-alias","provider":"vllm","upstream":"microsoft/phi-4"}`)
	mcpToolCall(t, srv1, 3, "toggle_model", `{"model_id":"runtime-alias","enabled":false}`)
	mcpToolCall(t, srv1, 4, "remove_model_alias", `{"alias":"doomed-alias"}`)

	if _, ok := srv1.catalog.Get("doomed-alias"); ok {
		t.Fatal("catalog still carries the removed alias")
	}

	// Restart from the same DataDir with a fresh process image.
	srv2, lane2 := boot()

	code, body, errType := chat(t, srv2, "doomed-alias")
	if code != http.StatusNotFound || errType != "unknown_model" {
		t.Fatalf("removed alias after restart: got %d %q (%s), want 404 unknown_model", code, errType, body)
	}
	if got := lane2.callCount(); got != 0 {
		t.Fatalf("lane calls after restart = %d, want 0", got)
	}

	if _, ok := srv2.sm.Snapshot().Models["doomed-alias"]; ok {
		t.Fatal("removed alias reinstalled into state after restart")
	}

	// The toggle flag is memory-only: a restart re-enables the alias.
	if code, body, _ := chat(t, srv2, "runtime-alias"); code != http.StatusOK {
		t.Fatalf("alias disabled before restart must be routable after it (toggle is memory-only): got %d (%s)", code, body)
	}
}

func TestRouter_ConcurrentAliasAdministrationKeepsDisabledState(t *testing.T) {
	srv, lane := newAliasLifecycleServer(t, map[string]ModelAlias{
		"frozen": {Provider: "vllm", Upstream: "Qwen/Qwen3.8-Instruct-AWQ"},
	})

	mcpToolCall(t, srv, 1, "toggle_model", `{"model_id":"frozen","enabled":false}`)

	// Concurrent administration: runtime toggles on unrelated ids interleave
	// with catalog/state reconciliation. None of it may re-enable "frozen",
	// and the shared bookkeeping must stay race-free.
	//
	// (Concurrent set_model_alias / remove_model_alias calls are deliberately
	// not driven here: ModelCatalog.persist races on its single .tmp path,
	// which is the separate persistence-hardening task.)
	const workers = 8
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := fmt.Sprintf("lane-id-%d", i)
			mcpToolCall(t, srv, 100+i, "toggle_model", fmt.Sprintf(`{"model_id":%q,"enabled":true}`, id))
			srv.syncCatalogToState()
			mcpToolCall(t, srv, 200+i, "toggle_model", fmt.Sprintf(`{"model_id":%q,"enabled":false}`, id))
			srv.syncCatalogToState()
		}(i)
	}
	wg.Wait()

	snap := srv.sm.Snapshot()
	if m, ok := snap.Models["frozen"]; !ok || m.Enabled {
		t.Fatalf("concurrent alias administration re-enabled the disabled alias: %+v (present=%v)", m, ok)
	}
	code, body, errType := chat(t, srv, "frozen")
	if code != http.StatusNotFound || errType != "unknown_model" {
		t.Fatalf("chat with frozen alias: got %d %q (%s), want 404 unknown_model", code, errType, body)
	}
	if got := lane.callCount(); got != 0 {
		t.Fatalf("lane calls = %d, want 0", got)
	}
	// Reconciliation must not have dropped the toggle-only rows either.
	for i := 0; i < workers; i++ {
		if _, ok := snap.Models[fmt.Sprintf("lane-id-%d", i)]; !ok {
			t.Fatalf("toggle-only model lane-id-%d was dropped by reconciliation", i)
		}
	}
}
