package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
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
