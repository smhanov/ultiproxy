package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// modelsUpstream serves GET /v1/models with the given ids (500 when fail).
func rebuildModelsUpstream(t *testing.T, ids []string, fail bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" && r.URL.Path != "/models" {
			http.NotFound(w, r)
			return
		}
		if fail {
			http.Error(w, "upstream down", http.StatusInternalServerError)
			return
		}
		data := make([]map[string]any, 0, len(ids))
		for _, id := range ids {
			data = append(data, map[string]any{"id": id})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestRebuildLaneSuccess covers the happy path: a stored lane rebuilds,
// swaps into the registry and reports its fresh discovery count.
func TestRebuildLaneSuccess(t *testing.T) {
	upstream := rebuildModelsUpstream(t, []string{"m1", "m2"}, false)
	dir := t.TempDir()
	store := NewRuntimeProviderStore(filepath.Join(dir, "providers.json"))
	store.DefaultDataDir = dir
	if err := store.Add(openaicompat.Config{
		Name:       "vllm",
		BaseURL:    upstream.URL,
		APIKey:     "sk-test",
		HTTPClient: upstream.Client(),
	}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	registry := provider.NewRegistry()
	stored, ok := store.List()["vllm"]
	if !ok {
		t.Fatal("vllm not in store")
	}
	initial, err := openaicompat.New(stored)
	if err != nil {
		t.Fatalf("initial New: %v", err)
	}
	registry.Register(initial.Provider())

	n, err := store.RebuildLane(registry, "vllm")
	if err != nil {
		t.Fatalf("RebuildLane: %v", err)
	}
	if n != 2 {
		t.Fatalf("discovered = %d, want 2", n)
	}
	bundle, ok := registry.Get("vllm")
	if !ok {
		t.Fatal("vllm not in registry after rebuild")
	}
	cacher, ok := bundle.Inference.(interface{ CachedModels() []string })
	if !ok {
		t.Fatalf("rebuilt lane has no model cache: %T", bundle.Inference)
	}
	if got := cacher.CachedModels(); len(got) != 2 {
		t.Fatalf("rebuilt cached = %v, want 2 models", got)
	}
}

// TestRebuildLaneNotStored maps to AC3: a lane with no stored record returns
// ErrProviderNotStored so the caller can report honestly instead of a false
// "usable".
func TestRebuildLaneNotStored(t *testing.T) {
	dir := t.TempDir()
	store := NewRuntimeProviderStore(filepath.Join(dir, "providers.json"))
	registry := provider.NewRegistry()
	if _, err := store.RebuildLane(registry, "ghost"); err == nil {
		t.Fatal("expected error for unknown lane")
	} else if !strings.Contains(err.Error(), "not present in runtime store") {
		t.Fatalf("error = %q, want it to wrap ErrProviderNotStored", err)
	}
}

// TestRebuildLanePreservesOldOnFailure (AC4): when the fresh build cannot
// discover (upstream rejects immediately), the previous lane stays registered.
func TestRebuildLanePreservesOldOnFailure(t *testing.T) {
	good := rebuildModelsUpstream(t, []string{"keep-me"}, false)
	dir := t.TempDir()
	store := NewRuntimeProviderStore(filepath.Join(dir, "providers.json"))
	store.DefaultDataDir = dir
	if err := store.Add(openaicompat.Config{
		Name:       "rot",
		BaseURL:    good.URL,
		APIKey:     "sk-one",
		HTTPClient: good.Client(),
	}); err != nil {
		t.Fatalf("Add good: %v", err)
	}
	registry := provider.NewRegistry()
	stored, _ := store.List()["rot"]
	initial, err := openaicompat.New(stored)
	if err != nil {
		t.Fatalf("initial New: %v", err)
	}
	registry.Register(initial.Provider())

	// Rotate the STORED config to a dead upstream (live registry still good).
	bad := rebuildModelsUpstream(t, nil, true)
	if err := store.Add(openaicompat.Config{
		Name:       "rot",
		BaseURL:    bad.URL,
		APIKey:     "sk-two",
		HTTPClient: bad.Client(),
	}); err != nil {
		t.Fatalf("Add bad: %v", err)
	}

	if _, err := store.RebuildLane(registry, "rot"); err == nil {
		t.Fatal("expected rebuild failure against the dead upstream")
	}
	// Old lane preserved with its cache.
	bundle, ok := registry.Get("rot")
	if !ok {
		t.Fatal("rot missing from registry after failed rebuild (old not preserved)")
	}
	cacher, ok := bundle.Inference.(interface{ CachedModels() []string })
	if !ok {
		t.Fatalf("preserved lane has no model cache: %T", bundle.Inference)
	}
	if got := cacher.CachedModels(); len(got) != 1 || got[0] != "keep-me" {
		t.Fatalf("preserved cached = %v, want [keep-me]", got)
	}
}

// TestRebuildLaneCustomKind is the antigravity-parity guard: custom-wire lanes
// are reported explicitly instead of rebuilt through the openaicompat path.
func TestRebuildLaneCustomKind(t *testing.T) {
	dir := t.TempDir()
	store := NewRuntimeProviderStore(filepath.Join(dir, "providers.json"))
	store.DefaultDataDir = dir
	if err := store.AddCustom("antigravity", "antigravity", ""); err != nil {
		t.Fatalf("AddCustom: %v", err)
	}
	registry := provider.NewRegistry()
	if _, err := store.RebuildLane(registry, "antigravity"); err == nil {
		t.Fatal("expected error for custom-wire lane")
	} else if !strings.Contains(err.Error(), "custom-wire lane") {
		t.Fatalf("error = %q, want custom-wire explanation", err)
	}
}

// TestRebuildLaneOrderPreserved ensures the swap relies on Register's
// replace-in-place (no Unregister reorder, no Register-semantics change).
func TestRebuildLaneOrderPreserved(t *testing.T) {
	upstream := rebuildModelsUpstream(t, []string{"m1"}, false)
	dir := t.TempDir()
	store := NewRuntimeProviderStore(filepath.Join(dir, "providers.json"))
	store.DefaultDataDir = dir
	for _, name := range []string{"aaa", "bbb"} {
		if err := store.Add(openaicompat.Config{
			Name:       name,
			BaseURL:    upstream.URL,
			HTTPClient: upstream.Client(),
		}); err != nil {
			t.Fatalf("Add %s: %v", name, err)
		}
	}
	registry := provider.NewRegistry()
	for _, name := range []string{"aaa", "bbb"} {
		cfg, _ := store.List()[name]
		p, err := openaicompat.New(cfg)
		if err != nil {
			t.Fatalf("New %s: %v", name, err)
		}
		registry.Register(p.Provider())
	}
	before := registry.Names()
	if _, err := store.RebuildLane(registry, "aaa"); err != nil {
		t.Fatalf("RebuildLane aaa: %v", err)
	}
	after := registry.Names()
	if len(before) != len(after) || before[0] != after[0] || before[1] != after[1] {
		t.Fatalf("registry order changed by rebuild: %v -> %v", before, after)
	}
	_ = context.Background()
}
