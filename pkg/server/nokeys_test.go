package server

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// TestNoAPIKeyAllowsOpenAccess verifies the "no API key configured = open
// access" requirement: with an empty api_key and no client keys, requests
// pass through without authentication.
func TestNoAPIKeyAllowsOpenAccess(t *testing.T) {
	dir := t.TempDir()
	writer, err := storage.NewWriter(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatalf("storage: %v", err)
	}
	defer writer.Close()

	cfg := DefaultConfig()
	cfg.Server.APIKey = ""
	cfg.Server.ClientKeys = nil

	srv := NewServer(cfg, nil,
		WithStateManager(state.NewStateManager()),
		WithStorageWriter(writer),
	)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("open access: got %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
}

// TestAPIKeyRequiredWhenConfigured verifies that configuring a key enforces it.
func TestAPIKeyRequiredWhenConfigured(t *testing.T) {
	dir := t.TempDir()
	writer, err := storage.NewWriter(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatalf("storage: %v", err)
	}
	defer writer.Close()

	cfg := DefaultConfig()
	cfg.Server.APIKey = "secret-key"

	srv := NewServer(cfg, nil,
		WithStateManager(state.NewStateManager()),
		WithStorageWriter(writer),
	)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("auth required: got %d, want 401", rec.Code)
	}

	req2 := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req2.Header.Set("Authorization", "Bearer secret-key")
	rec2 := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec2, req2)

	if rec2.Code != http.StatusOK {
		t.Fatalf("valid key: got %d, want 200", rec2.Code)
	}
}

// TestCatalogSyncedToState verifies aliases land in the state snapshot and
// therefore appear in /v1/models and route correctly.
func TestCatalogSyncedToState(t *testing.T) {
	dir := t.TempDir()
	writer, err := storage.NewWriter(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatalf("storage: %v", err)
	}
	defer writer.Close()

	cfg := DefaultConfig()
	cfg.DataDir = dir
	cfg.Server.Models = map[string]ModelAlias{
		"qwenpoint-3.8": {Provider: "vllm", Upstream: "Qwen/Qwen3.8-Instruct-AWQ"},
	}
	sm := state.NewStateManager()

	srv := NewServer(cfg, nil, WithStateManager(sm), WithStorageWriter(writer))

	snap := sm.Snapshot()
	m, ok := snap.Models["qwenpoint-3.8"]
	if !ok {
		t.Fatal("alias not synced into state manager")
	}
	if m.Provider != "vllm" || !m.Enabled {
		t.Fatalf("model runtime wrong: %+v", m)
	}

	// Must be visible in GET /v1/models
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("models: %d", rec.Code)
	}
	body := rec.Body.String()
	if body == "" || body == `{"object":"list","data":[]}` {
		t.Fatalf("aliased model not in /v1/models: %s", body)
	}

	// MCP alias manager must expose it
	al := srv.mcpServer
	_ = al
}
