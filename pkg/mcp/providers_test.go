package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// modelsTestLane is a minimal InferenceProvider so the registry has something
// real to hold while add_provider runs.
type modelsTestLane struct{ name string }

func (l modelsTestLane) Name() string { return l.name }
func (l modelsTestLane) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	return &ir.Response{}, nil
}
func (l modelsTestLane) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	return nil, nil
}

// newModelsUpstream serves an OpenAI-compatible /v1/models and counts requests.
func newModelsUpstream(t *testing.T, ids ...string) (*httptest.Server, *int32) {
	t.Helper()
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		atomic.AddInt32(&calls, 1)
		data := make([]map[string]any, 0, len(ids))
		for _, id := range ids {
			data = append(data, map[string]any{"id": id})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

// TestMCPAddProviderRunsDiscovery: the success path runs model discovery
// synchronously before replying, so the reply states how many upstream models
// the new lane serves and no second call is needed.
func TestMCPAddProviderRunsDiscovery(t *testing.T) {
	upstream, calls := newModelsUpstream(t, "deepseek-chat", "deepseek-reasoner")

	dir := t.TempDir()
	store := newFileProviderStore(filepath.Join(dir, "providers.json"))
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store))

	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"deepseek","base_url":"`+upstream.URL+`","api_key":"sk-test"}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}
	text := res.Content[0].Text
	if !strings.Contains(text, "discovered 2 models") {
		t.Errorf("add_provider reply = %q, want it to state \"discovered 2 models\"", text)
	}
	if !strings.Contains(text, `"registered": true`) {
		t.Errorf("add_provider reply = %q, want registered: true", text)
	}

	bundle, ok := registry.Get("deepseek")
	if !ok {
		t.Fatal("deepseek not registered")
	}
	cacher, ok := bundle.Inference.(interface{ CachedModels() []string })
	if !ok {
		t.Fatalf("lane has no model cache: %T", bundle.Inference)
	}
	if got := cacher.CachedModels(); len(got) != 2 {
		t.Errorf("cached models = %v, want the two upstream ids", got)
	}
	if got := atomic.LoadInt32(calls); got < 1 {
		t.Errorf("discovery never reached the upstream (%d calls)", got)
	}
}

// TestMCPAddProviderModelDiscoveryOptOut: quirks.model_list_passthrough:false
// is the explicit opt-out - the lane is registered, nothing is probed, and the
// reply says so instead of inventing a model list.
func TestMCPAddProviderModelDiscoveryOptOut(t *testing.T) {
	upstream, calls := newModelsUpstream(t, "deepseek-chat")

	dir := t.TempDir()
	store := newFileProviderStore(filepath.Join(dir, "providers.json"))
	srv := NewServer(provider.NewRegistry(), nil, WithProviderStore(store))

	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"deepseek","base_url":"`+upstream.URL+`","api_key":"sk-test","quirks":{"model_list_passthrough":false}}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}
	text := res.Content[0].Text
	if !strings.Contains(text, "discovered 0 models") {
		t.Errorf("add_provider reply = %q, want it to state \"discovered 0 models\"", text)
	}
	if !strings.Contains(text, "model discovery disabled") {
		t.Errorf("add_provider reply = %q, want it to say discovery is disabled", text)
	}
	if got := atomic.LoadInt32(calls); got != 0 {
		t.Errorf("opted-out lane was probed %d times, want 0", got)
	}
	if strings.Contains(text, "sk-test") {
		t.Errorf("add_provider reply leaked the api key: %s", text)
	}
}

// stubCredsStore is a daemon-owned credential store stub for T003 AC1: it
// serves a valid token so construction-time discovery succeeds and live ==
// stored can be observed through the cached model list.
type stubCredsStore struct{}

func (stubCredsStore) Get(ctx context.Context, key string) (auth.Credential, error) {
	return auth.Credential{
		Provider:    "xai",
		AccessToken: "stub-access-tok",
		ClientID:    key,
	}, nil
}
func (stubCredsStore) Peek(key string) (auth.Credential, bool) {
	cred, _ := stubCredsStore{}.Get(context.Background(), key)
	return cred, true
}
func (stubCredsStore) Store(ctx context.Context, key string, cred auth.Credential) error {
	return nil
}
func (stubCredsStore) Invalidate(key, accessToken string) bool { return false }

// TestMCPAddProviderOAuthLiveEqualsStored (T003 AC1): after add_provider with
// auth_via_oauth_manager:true the live registered lane carries the same
// credential store and resolved quirks as the stored config.
func TestMCPAddProviderOAuthLiveEqualsStored(t *testing.T) {
	upstream, _ := newModelsUpstream(t, "grok-4")

	dir := t.TempDir()
	store := newFileProviderStore(filepath.Join(dir, "providers.json"))
	registry := provider.NewRegistry()
	creds := stubCredsStore{}
	srv := NewServer(registry, nil, WithProviderStore(store), WithCredentialStore(creds))

	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"xai","base_url":"`+upstream.URL+`","quirks":{"auth_via_oauth_manager":true}}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}

	stored, ok := store.List()["xai"]
	if !ok {
		t.Fatal("xai not in store after add_provider")
	}
	if stored.Creds == nil {
		t.Fatal("stored config has no injected Creds")
	}
	if !sameInterfaceValue(stored.Creds, any(creds)) {
		t.Errorf("stored Creds = %T, want the daemon-owned stub store", stored.Creds)
	}
	if !stored.Quirks.AuthViaOAuthManager {
		t.Error("stored config lost auth_via_oauth_manager")
	}
	if !stored.Quirks.ModelListPassthrough {
		t.Errorf("stored discovery flag = false, want true (OAuth lanes discover)")
	}

	lane, ok := registry.Get("xai")
	if !ok {
		t.Fatal("xai not registered in the live registry")
	}
	if lane.Auth == nil {
		t.Error("live lane has no Auth surface (OAuth TokenSource missing)")
	}
	if lane.Inference == nil {
		t.Fatal("live lane has no inference surface")
	}
	if en, ok := lane.Inference.(interface{ ModelDiscoveryEnabled() bool }); !ok {
		t.Fatalf("live lane has no ModelDiscoveryEnabled: %T", lane.Inference)
	} else if got := en.ModelDiscoveryEnabled(); got != stored.Quirks.ModelListPassthrough {
		t.Errorf("live discovery = %v, stored = %v (live != stored)", got, stored.Quirks.ModelListPassthrough)
	}
	if cacher, ok := lane.Inference.(interface{ CachedModels() []string }); ok {
		if got := cacher.CachedModels(); len(got) != 1 || got[0] != "grok-4" {
			t.Errorf("live cached models = %v, want [grok-4] from the enriched build", got)
		}
	}
}

// TestMCPAddProviderReAddReplacesLiveLane (T003 AC2): re-adding an existing
// lane (the key-rotation path) replaces the live lane, not just the stored one.
func TestMCPAddProviderReAddReplacesLiveLane(t *testing.T) {
	upstream1, _ := newModelsUpstream(t, "model-a")
	upstream2, _ := newModelsUpstream(t, "model-b")

	dir := t.TempDir()
	store := newFileProviderStore(filepath.Join(dir, "providers.json"))
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store))

	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"rot","base_url":"`+upstream1.URL+`","api_key":"sk-one"}`)
	if res.IsError {
		t.Fatalf("first add_provider failed: %s", res.Content[0].Text)
	}
	lane1, ok := registry.Get("rot")
	if !ok {
		t.Fatal("rot not registered after first add")
	}
	cacher1, ok := lane1.Inference.(interface{ CachedModels() []string })
	if !ok {
		t.Fatalf("lane has no model cache: %T", lane1.Inference)
	}
	if got := cacher1.CachedModels(); len(got) != 1 || got[0] != "model-a" {
		t.Fatalf("first live cached = %v, want [model-a]", got)
	}

	res2 := callMCPTool(t, srv, 2, "add_provider",
		`{"name":"rot","base_url":"`+upstream2.URL+`","api_key":"sk-two"}`)
	if res2.IsError {
		t.Fatalf("second add_provider failed: %s", res2.Content[0].Text)
	}
	stored := store.List()["rot"]
	if stored.BaseURL != upstream2.URL {
		t.Fatalf("stored base_url = %q, want second upstream %q", stored.BaseURL, upstream2.URL)
	}
	if stored.APIKey != "sk-two" {
		t.Fatalf("stored api_key = %q, want rotated sk-two", stored.APIKey)
	}
	lane2, ok := registry.Get("rot")
	if !ok {
		t.Fatal("rot not registered after second add")
	}
	cacher2, ok := lane2.Inference.(interface{ CachedModels() []string })
	if !ok {
		t.Fatalf("re-added lane has no model cache: %T", lane2.Inference)
	}
	if got := cacher2.CachedModels(); len(got) != 1 || got[0] != "model-b" {
		t.Errorf("re-added live cached = %v, want [model-b] (stale live lane survived)", got)
	}
}

// failingAddStore fails persist so AC4 (register only after validation+persist)
// can be verified at the tool layer.
type failingAddStore struct {
	*fileProviderStore
}

func (s *failingAddStore) Add(cfg openaicompat.Config) error {
	return errors.New("persist boom")
}

// TestMCPAddProviderPersistFailureRegistersNothing (T003 AC4): when persist
// fails the lane must not become live.
func TestMCPAddProviderPersistFailureRegistersNothing(t *testing.T) {
	upstream, _ := newModelsUpstream(t, "m1")

	dir := t.TempDir()
	inner := newFileProviderStore(filepath.Join(dir, "providers.json"))
	store := &failingAddStore{fileProviderStore: inner}
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store))

	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"vllm","base_url":"`+upstream.URL+`","api_key":"sk-x"}`)
	if !res.IsError {
		t.Fatalf("expected error when persist fails: %s", res.Content[0].Text)
	}
	if _, ok := registry.Get("vllm"); ok {
		t.Fatal("lane was registered although persist failed")
	}
}

// TestLaneConfigsDrifted covers the generalized drift assertion that subsumed
// the freebuff-only branch: identical configs do not drift, enrichment fields
// do.
func TestLaneConfigsDrifted(t *testing.T) {
	base := openaicompat.Config{
		Name:    "x",
		BaseURL: "http://127.0.0.1:1/v1",
		APIKey:  "sk",
		DataDir: "/data/credentials/x",
		Quirks:  openaicompat.Quirks{ModelListPassthrough: true},
	}
	if laneConfigsDrifted(base, base) {
		t.Fatal("identical configs reported as drifted")
	}
	other := base
	other.DataDir = "/other"
	if !laneConfigsDrifted(base, other) {
		t.Error("DataDir change not detected as drift")
	}
	other = base
	other.Quirks.ModelListPassthrough = false
	if !laneConfigsDrifted(base, other) {
		t.Error("discovery flag change not detected as drift")
	}
	marked := base
	marked.Quirks.FreebuffActor = struct{}{}
	real := base
	real.Quirks.FreebuffActor = &wireFreebuffFake{}
	if !laneConfigsDrifted(marked, real) {
		t.Error("marker -> real actor change not detected as drift")
	}
	if laneConfigsDrifted(real, real) {
		t.Error("same real actor reported as drifted")
	}
}

// wireFreebuffFake is a minimal FreebuffActor for drift tests.
type wireFreebuffFake struct{}

func (wireFreebuffFake) Acquire(ctx context.Context) error { return nil }
func (wireFreebuffFake) Release()                          {}
