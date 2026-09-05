package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/antigravity"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/state"
)

// fakeModelUpstream stands in for an OpenAI-compatible lane upstream: it serves
// /v1/models (model discovery) and /v1/chat/completions (capturing the request
// body so tests can assert what ultiproxy actually sent).
type fakeModelUpstream struct {
	srv      *httptest.Server
	models   int32 // requests to /v1/models
	chats    int32 // requests to /v1/chat/completions
	lastBody map[string]any
}

func newFakeModelUpstream(t *testing.T, modelIDs ...string) *fakeModelUpstream {
	t.Helper()
	f := &fakeModelUpstream{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			atomic.AddInt32(&f.models, 1)
			data := make([]map[string]any, 0, len(modelIDs))
			for _, id := range modelIDs {
				data = append(data, map[string]any{"id": id})
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
		case "/v1/chat/completions":
			atomic.AddInt32(&f.chats, 1)
			_ = json.NewDecoder(r.Body).Decode(&f.lastBody)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "cmpl-fake",
				"choices": []map[string]any{{
					"index":         0,
					"message":       map[string]any{"role": "assistant", "content": "ok"},
					"finish_reason": "stop",
				}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeModelUpstream) modelRequests() int32 { return atomic.LoadInt32(&f.models) }

type modelsResponseShape struct {
	Object string `json:"object"`
	Data   []struct {
		ID              string `json:"id"`
		Object          string `json:"object"`
		Created         int64  `json:"created"`
		OwnedBy         string `json:"owned_by"`
		ContextLength   int    `json:"context_length"`
		MaxOutputTokens int    `json:"max_output_tokens"`
	} `json:"data"`
}

func getModels(t *testing.T, srv *Server) modelsResponseShape {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/models: %d %s", rec.Code, rec.Body.String())
	}
	var out modelsResponseShape
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode /v1/models: %v (%s)", err, rec.Body.String())
	}
	return out
}

func (r modelsResponseShape) ids() map[string]int {
	out := make(map[string]int, len(r.Data))
	for _, e := range r.Data {
		out[e.ID] = e.MaxOutputTokens
	}
	return out
}

// TestHandleModels_IncludesRuntimeLanes verifies the aggregated /v1/models
// response is not alias-only: runtime lanes added through the MCP surface
// appear too, as "<lane>/<model>" ids only. A passthrough lane contributes one
// entry per cached discovered upstream model (the form routing accepts); lanes
// without model discovery (anthropic, codex, custom kinds) and without a
// default model contribute nothing - the bare lane name is never advertised,
// because "model": "<lane>" is a legacy routing prefix, not a model id; and
// listing never re-probes the upstream.
func TestHandleModels_IncludesRuntimeLanes(t *testing.T) {
	dir := t.TempDir()
	upstream := newFakeModelUpstream(t, "Qwen/Qwen3.8-27B-Instruct", "meta-llama/Llama-3-8B")

	cfg := DefaultConfig()
	cfg.DataDir = dir
	cfg.Server.Models = map[string]ModelAlias{
		"qwenpoint-3.8": {Provider: "vllm-local", Upstream: "Qwen/Qwen3.8-27B-Instruct"},
	}

	store := NewRuntimeProviderStore(filepath.Join(dir, "providers.json"))
	store.DefaultDataDir = dir
	if err := store.Add(openaicompat.Config{
		Name:    "vllm-local",
		BaseURL: upstream.srv.URL,
		Quirks:  openaicompat.Quirks{ModelListPassthrough: true},
	}); err != nil {
		t.Fatalf("add passthrough lane: %v", err)
	}
	// A custom-kind lane (anthropic/codex style): registered in the store, no
	// model discovery surface at all.
	if err := store.AddCustom("anthropic", "anthropic", ""); err != nil {
		t.Fatalf("add custom lane: %v", err)
	}

	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: &fakeInferenceProvider{name: "anthropic"}})

	srv := NewServer(cfg, registry,
		WithStateManager(state.NewStateManager()),
		WithRuntimeProviderStore(store),
	)

	discovered := upstream.modelRequests()
	if discovered != 1 {
		t.Fatalf("expected exactly 1 startup model discovery request, got %d", discovered)
	}

	resp := getModels(t, srv)
	ids := resp.ids()

	for _, want := range []string{
		"vllm-local/Qwen/Qwen3.8-27B-Instruct", // discovered upstream models
		"vllm-local/meta-llama/Llama-3-8B",
		"qwenpoint-3.8", // catalog alias
	} {
		if _, ok := ids[want]; !ok {
			t.Errorf("model %q missing from /v1/models: %v", want, ids)
		}
	}
	// The bare lane name is a routing prefix, not a model: it must not be
	// advertised, neither for the discovery lane nor for the lane that has no
	// model surface at all.
	for _, lane := range []string{"vllm-local", "anthropic"} {
		if _, ok := ids[lane]; ok {
			t.Errorf("bare lane name %q advertised as a model id: %v", lane, ids)
		}
	}
	for id := range ids {
		if strings.HasPrefix(id, "anthropic/") {
			t.Errorf("invented model id for a lane without model discovery: %q", id)
		}
	}
	if _, ok := ids["vllm-local/default"]; ok {
		t.Errorf("invented placeholder model id for an empty passthrough cache: %q", "vllm-local/default")
	}

	// Listing models must never fan out to the lane upstream.
	if got := upstream.modelRequests(); got != discovered {
		t.Errorf("GET /v1/models re-probed the upstream: %d -> %d", discovered, got)
	}
}

// TestHandleModels_EmptyPassthroughCacheInventsNothing: a passthrough lane
// whose discovery failed (upstream unreachable / empty list) and has no
// default model contributes nothing at all - no bare lane name, no invented
// placeholder id.
func TestHandleModels_EmptyPassthroughCacheInventsNothing(t *testing.T) {
	dir := t.TempDir()
	upstream := newFakeModelUpstream(t) // serves an empty model list

	cfg := DefaultConfig()
	cfg.DataDir = dir

	store := NewRuntimeProviderStore(filepath.Join(dir, "providers.json"))
	store.DefaultDataDir = dir
	if err := store.Add(openaicompat.Config{
		Name:    "vllm-local",
		BaseURL: upstream.srv.URL,
		Quirks:  openaicompat.Quirks{ModelListPassthrough: true},
	}); err != nil {
		t.Fatalf("add passthrough lane: %v", err)
	}

	srv := NewServer(cfg, provider.NewRegistry(),
		WithStateManager(state.NewStateManager()),
		WithRuntimeProviderStore(store),
	)

	resp := getModels(t, srv)
	if len(resp.Data) != 0 {
		t.Fatalf("expected no entries for a lane with empty discovery and no default, got %+v", resp.Data)
	}
}

// TestHandleModels_DefaultModelEscapeHatch: a lane whose discovery cache is
// empty but which has a real default model (quirks.default_model, or
// antigravity's compiled-in default) still advertises exactly one routable id,
// "<lane>/<default>".
func TestHandleModels_DefaultModelEscapeHatch(t *testing.T) {
	dir := t.TempDir()
	upstream := newFakeModelUpstream(t) // serves an empty model list

	cfg := DefaultConfig()
	cfg.DataDir = dir

	store := NewRuntimeProviderStore(filepath.Join(dir, "providers.json"))
	store.DefaultDataDir = dir
	if err := store.Add(openaicompat.Config{
		Name:    "augure-like",
		BaseURL: upstream.srv.URL,
		Quirks: openaicompat.Quirks{
			ModelListPassthrough: true,
			DefaultModel:         "tofino-3",
		},
	}); err != nil {
		t.Fatalf("add lane with a default model: %v", err)
	}

	srv := NewServer(cfg, provider.NewRegistry(),
		WithStateManager(state.NewStateManager()),
		WithRuntimeProviderStore(store),
	)

	ids := getModels(t, srv).ids()
	if _, ok := ids["augure-like/tofino-3"]; !ok {
		t.Fatalf("lane/default missing from /v1/models: %v", ids)
	}
	for id := range ids {
		if id == "augure-like" {
			t.Errorf("bare lane name advertised: %v", ids)
		}
	}

	// The escape hatch must not fan out to the upstream either.
	if got := upstream.modelRequests(); got == 0 {
		t.Fatal("startup discovery never ran")
	}
	before := upstream.modelRequests()
	_ = getModels(t, srv)
	if got := upstream.modelRequests(); got != before {
		t.Errorf("GET /v1/models re-probed the upstream: %d -> %d", before, got)
	}
}

// TestHandleModels_AliasLimits: catalog alias ContextLimit/MaxOutput are
// surfaced on the alias entry (ContextLimit is advisory metadata only).
func TestHandleModels_AliasLimits(t *testing.T) {
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.DataDir = dir
	cfg.Server.Models = map[string]ModelAlias{
		"qwenpoint-3.8": {
			Provider:     "vllm",
			Upstream:     "Qwen/Qwen3.8-27B-Instruct",
			ContextLimit: 200000,
			MaxOutput:    77,
		},
	}

	srv := NewServer(cfg, provider.NewRegistry(), WithStateManager(state.NewStateManager()))

	resp := getModels(t, srv)
	if len(resp.Data) != 1 || resp.Data[0].ID != "qwenpoint-3.8" {
		t.Fatalf("expected the alias entry only, got %+v", resp.Data)
	}
	if resp.Data[0].ContextLength != 200000 {
		t.Errorf("context_length = %d, want 200000", resp.Data[0].ContextLength)
	}
	if resp.Data[0].MaxOutputTokens != 77 {
		t.Errorf("max_output_tokens = %d, want 77", resp.Data[0].MaxOutputTokens)
	}
}

// TestAliasMaxOutputClampsMaxTokens: an alias MaxOutput is enforced on the
// request path \u2014 the upstream sees max_tokens clamped to the alias limit, both
// when the client asks for more and when the client asks for nothing.
func TestAliasMaxOutputClampsMaxTokens(t *testing.T) {
	dir := t.TempDir()
	upstream := newFakeModelUpstream(t)

	cfg := DefaultConfig()
	cfg.DataDir = dir
	cfg.Server.Models = map[string]ModelAlias{
		"tiny": {Provider: "vllmlane", Upstream: "Qwen/Qwen3.8-27B-Instruct", MaxOutput: 77},
	}

	registry := provider.NewRegistry()
	p, err := openaicompat.New(openaicompat.Config{
		Name:       "vllmlane",
		BaseURL:    upstream.srv.URL,
		APIKey:     "test-key",
		HTTPClient: upstream.srv.Client(),
	})
	if err != nil {
		t.Fatalf("openaicompat.New: %v", err)
	}
	registry.Register(p.Provider())

	srv := NewServer(cfg, registry, WithStateManager(state.NewStateManager()))

	post := func(body string) {
		t.Helper()
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("POST /v1/chat/completions: %d %s", rec.Code, rec.Body.String())
		}
	}

	sentMaxTokens := func() float64 {
		v, _ := upstream.lastBody["max_tokens"].(float64)
		return v
	}

	post(`{"model":"tiny","max_tokens":5000,"messages":[{"role":"user","content":"hi"}]}`)
	if got := sentMaxTokens(); got != 77 {
		t.Errorf("max_tokens sent upstream = %v, want 77 (clamped from 5000)", got)
	}
	if got := upstream.lastBody["model"]; got != "Qwen/Qwen3.8-27B-Instruct" {
		t.Errorf("upstream model = %v, want the alias upstream id", got)
	}

	post(`{"model":"tiny","messages":[{"role":"user","content":"hi"}]}`)
	if got := sentMaxTokens(); got != 77 {
		t.Errorf("max_tokens sent upstream = %v, want 77 (alias cap applied when unset)", got)
	}
}

// TestUpstreamOptions_MaxOutputClamp unit-tests the option builder directly.
func TestUpstreamOptions_MaxOutputClamp(t *testing.T) {
	alias := ModelAlias{Provider: "lane", Upstream: "m", MaxOutput: 77}

	got := provider.NewRequestConfig(upstreamOptions(nil, "m", alias, true, "hash")...)
	if got.MaxTokens != 77 {
		t.Errorf("no request max_tokens: MaxTokens = %d, want 77", got.MaxTokens)
	}
	if got.Model != "m" || got.ClientKeyHash != "hash" {
		t.Errorf("unexpected config: %+v", got)
	}

	got = provider.NewRequestConfig(upstreamOptions([]provider.Option{provider.WithMaxTokens(5000)}, "m", alias, true, "hash")...)
	if got.MaxTokens != 77 {
		t.Errorf("request max_tokens 5000: MaxTokens = %d, want clamped 77", got.MaxTokens)
	}

	got = provider.NewRequestConfig(upstreamOptions([]provider.Option{provider.WithMaxTokens(50)}, "m", alias, true, "hash")...)
	if got.MaxTokens != 50 {
		t.Errorf("request max_tokens 50: MaxTokens = %d, want 50 (below the alias cap, untouched)", got.MaxTokens)
	}

	// No alias: the request's own max_tokens must survive.
	got = provider.NewRequestConfig(upstreamOptions([]provider.Option{provider.WithMaxTokens(5000)}, "m", ModelAlias{}, false, "hash")...)
	if got.MaxTokens != 5000 {
		t.Errorf("no alias: MaxTokens = %d, want 5000", got.MaxTokens)
	}
}

// fakeListableLane is a registered lane whose model surface the /v1/models
// handler can read offline: a discovery cache (modelsCacheProvider) and/or a
// lane default model (defaultModelProvider). It exists so tests can build the
// three lane shapes the listing contract distinguishes - discovery, default,
// and neither - without any upstream.
type fakeListableLane struct {
	*fakeInferenceProvider
	cached []string
	def    string
}

func (f *fakeListableLane) CachedModels() []string { return f.cached }
func (f *fakeListableLane) DefaultModel() string   { return f.def }

// TestHandleModels_AdvertisesOnlyRoutableIDs is the core T031 contract:
//
//   - AC1: no id equals a registered lane name (a lane name is a routing
//     prefix, not a model);
//   - AC2: every id is either a catalog alias or a "<lane>/<model>" id;
//   - AC3: a lane with discovery lists "<lane>/<discovered>", a lane with a
//     default model lists "<lane>/<default>", a lane with neither lists
//     nothing at all.
func TestHandleModels_AdvertisesOnlyRoutableIDs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.Server.Models = map[string]ModelAlias{
		"myalias": {Provider: "discovered", Upstream: "m1"},
	}

	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: &fakeListableLane{
		fakeInferenceProvider: &fakeInferenceProvider{name: "discovered"},
		cached:                []string{"m2", "m1"},
	}})
	registry.Register(provider.Provider{Inference: &fakeListableLane{
		fakeInferenceProvider: &fakeInferenceProvider{name: "defaultonly"},
		def:                   "tofino-3",
	}})
	registry.Register(provider.Provider{Inference: &fakeInferenceProvider{name: "neither"}})

	srv := NewServer(cfg, registry, WithStateManager(state.NewStateManager()))
	resp := getModels(t, srv)
	ids := resp.ids()

	// AC3: the routable ids that must be advertised.
	for _, want := range []string{"myalias", "discovered/m1", "discovered/m2", "defaultonly/tofino-3"} {
		if _, ok := ids[want]; !ok {
			t.Errorf("model %q missing from /v1/models: %v", want, ids)
		}
	}

	// AC1: zero bare lane names.
	for _, lane := range registry.Names() {
		if _, ok := ids[lane]; ok {
			t.Errorf("AC1 violated: bare lane name %q is advertised as a model id", lane)
		}
	}

	// AC2: every advertised id is an alias or a "<lane>/<model>" id.
	aliases := map[string]bool{"myalias": true}
	for id := range ids {
		if aliases[id] {
			continue
		}
		if !strings.Contains(id, "/") {
			t.Errorf("AC2 violated: id %q is neither an alias nor a <lane>/<model> id", id)
		}
	}

	// AC3 (negative): a lane with neither discovery nor a default model
	// contributes nothing - no invented ids.
	for id := range ids {
		if strings.HasPrefix(id, "neither/") {
			t.Errorf("AC3 violated: lane without discovery or default invented id %q", id)
		}
	}
}

// TestHandleModels_HideTestLanes: ULTIPROXY_HIDE_TEST_LANES=1 removes clearly
// test lanes (probe/fake/...) from the listing while real lanes stay listed.
func TestHandleModels_HideTestLanes(t *testing.T) {
	newServer := func() *Server {
		cfg := DefaultConfig()
		cfg.DataDir = t.TempDir()
		registry := provider.NewRegistry()
		registry.Register(provider.Provider{Inference: &fakeListableLane{
			fakeInferenceProvider: &fakeInferenceProvider{name: "probe"},
			cached:                []string{"m1"},
		}})
		registry.Register(provider.Provider{Inference: &fakeListableLane{
			fakeInferenceProvider: &fakeInferenceProvider{name: "reallane"},
			cached:                []string{"m1"},
		}})
		return NewServer(cfg, registry, WithStateManager(state.NewStateManager()))
	}

	t.Run("hidden", func(t *testing.T) {
		t.Setenv("ULTIPROXY_HIDE_TEST_LANES", "1")
		ids := getModels(t, newServer()).ids()
		for id := range ids {
			if id == "probe" || strings.HasPrefix(id, "probe/") {
				t.Errorf("AC4 violated: test lane id %q listed with ULTIPROXY_HIDE_TEST_LANES=1: %v", id, ids)
			}
		}
		if _, ok := ids["reallane/m1"]; !ok {
			t.Errorf("hiding test lanes also hid a real lane: %v", ids)
		}
	})

	t.Run("visible by default", func(t *testing.T) {
		t.Setenv("ULTIPROXY_HIDE_TEST_LANES", "")
		ids := getModels(t, newServer()).ids()
		if _, ok := ids["probe/m1"]; !ok {
			t.Errorf("default (flag unset) must keep listing the probe lane's models: %v", ids)
		}
	})
}

// TestHandleModels_AntigravityDefaultModel: the antigravity lane is the
// real-world escape hatch - no model discovery surface at all, but a compiled-in
// default model - so /v1/models advertises exactly
// antigravity/gemini-3.7-flash-high and never the bare lane name.
func TestHandleModels_AntigravityDefaultModel(t *testing.T) {
	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()

	lane := antigravity.New(antigravity.Config{})
	registry := provider.NewRegistry()
	registry.Register(lane.ProviderBundle())

	ids := getModels(t, NewServer(cfg, registry, WithStateManager(state.NewStateManager()))).ids()

	if _, ok := ids["antigravity/gemini-3.7-flash-high"]; !ok {
		t.Errorf("antigravity/<default> missing from /v1/models: %v", ids)
	}
	if _, ok := ids["antigravity"]; ok {
		t.Errorf("bare lane name %q advertised as a model id: %v", "antigravity", ids)
	}
	if len(ids) != 1 {
		t.Errorf("antigravity lane should contribute exactly one id, got %v", ids)
	}
}

// TestHandleModels_MatchesListModelsTool is the cross-surface parity guard
// (AC5): the ids GET /v1/models advertises and the ids the list_models MCP tool
// advertises must be the same set, so the two surfaces cannot drift. Both must
// exclude bare lane names, and both must honour ULTIPROXY_HIDE_TEST_LANES.
func TestHandleModels_MatchesListModelsTool(t *testing.T) {
	build := func() *Server {
		cfg := DefaultConfig()
		cfg.DataDir = t.TempDir()
		cfg.Server.Models = map[string]ModelAlias{
			"myalias": {Provider: "discovered", Upstream: "m1"},
		}
		registry := provider.NewRegistry()
		registry.Register(provider.Provider{Inference: &fakeListableLane{
			fakeInferenceProvider: &fakeInferenceProvider{name: "discovered"},
			cached:                []string{"m1", "m2"},
		}})
		registry.Register(provider.Provider{Inference: &fakeListableLane{
			fakeInferenceProvider: &fakeInferenceProvider{name: "defaultonly"},
			def:                   "tofino-3",
		}})
		registry.Register(provider.Provider{Inference: &fakeInferenceProvider{name: "neither"}})
		registry.Register(provider.Provider{Inference: &fakeListableLane{
			fakeInferenceProvider: &fakeInferenceProvider{name: "probe"},
			cached:                []string{"m1"},
		}})
		return NewServer(cfg, registry, WithStateManager(state.NewStateManager()))
	}

	httpIDs := func(srv *Server) []string {
		resp := getModels(t, srv)
		out := make([]string, 0, len(resp.Data))
		for _, e := range resp.Data {
			out = append(out, e.ID)
		}
		sort.Strings(out)
		return out
	}

	mcpIDs := func(srv *Server) []string {
		body := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_models","arguments":{}}}`
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("POST /mcp list_models: %d %s", rec.Code, rec.Body.String())
		}
		var rpc struct {
			Result struct {
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"result"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &rpc); err != nil {
			t.Fatalf("decode list_models reply: %v (%s)", err, rec.Body.String())
		}
		if len(rpc.Result.Content) == 0 {
			t.Fatal("list_models returned no content")
		}
		ids := map[string]struct {
			Enabled bool `json:"enabled"`
		}{}
		if err := json.Unmarshal([]byte(rpc.Result.Content[0].Text), &ids); err != nil {
			t.Fatalf("decode list_models payload: %v (%s)", err, rpc.Result.Content[0].Text)
		}
		out := make([]string, 0, len(ids))
		for id, m := range ids {
			if !m.Enabled {
				continue // /v1/models omits disabled ids; list_models keeps them for toggle_model
			}
			out = append(out, id)
		}
		sort.Strings(out)
		return out
	}

	for _, tc := range []struct {
		name string
		env  string
		want []string
	}{
		{name: "default", env: "", want: []string{"defaultonly/tofino-3", "discovered/m1", "discovered/m2", "myalias", "probe/m1"}},
		{name: "hide test lanes", env: "1", want: []string{"defaultonly/tofino-3", "discovered/m1", "discovered/m2", "myalias"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("ULTIPROXY_HIDE_TEST_LANES", tc.env)
			srv := build()
			if got := httpIDs(srv); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("GET /v1/models ids = %v, want %v", got, tc.want)
			}
			if got := mcpIDs(srv); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("list_models ids = %v, want %v", got, tc.want)
			}
		})
	}
}
