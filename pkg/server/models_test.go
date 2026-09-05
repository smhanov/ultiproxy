package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
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
// appear too. A passthrough lane contributes one entry per cached discovered
// upstream model (as "<lane>/<model>", the form routing accepts) plus one
// entry named exactly "<lane>"; lanes without model discovery (anthropic,
// codex, custom kinds) contribute only the "<lane>" entry, with no invented
// model ids; and listing never re-probes the upstream.
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
		"vllm-local",                           // the lane itself (prefix-routable)
		"vllm-local/Qwen/Qwen3.8-27B-Instruct", // discovered upstream models
		"vllm-local/meta-llama/Llama-3-8B",
		"anthropic",     // lane with no model discovery: static entry only
		"qwenpoint-3.8", // catalog alias
	} {
		if _, ok := ids[want]; !ok {
			t.Errorf("model %q missing from /v1/models: %v", want, ids)
		}
	}
	for id := range ids {
		if id == "anthropic" {
			continue
		}
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
// whose discovery failed (upstream unreachable / empty list) contributes its
// lane entry and nothing else.
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
	if len(resp.Data) != 1 || resp.Data[0].ID != "vllm-local" {
		t.Fatalf("expected only the lane entry, got %+v", resp.Data)
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
