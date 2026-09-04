package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

func TestRuntimeProviderStore_PersistRestart(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")

	s1 := NewRuntimeProviderStore(path)
	if err := s1.Add(openaicompat.Config{
		Name:    "vllm",
		BaseURL: "http://127.0.0.1:1/v1",
		APIKey:  "sk-local",
		Quirks:  openaicompat.Quirks{ModelListPassthrough: true},
	}); err != nil {
		t.Fatalf("add vllm: %v", err)
	}
	if err := s1.Add(openaicompat.Config{
		Name:    "openrouter",
		BaseURL: "https://openrouter.ai/api/v1",
		APIKey:  "sk-or",
		Quirks: openaicompat.Quirks{
			MaxTokensByModel: map[string]int{"glm-4.5-air": 98304},
		},
	}); err != nil {
		t.Fatalf("add openrouter: %v", err)
	}

	// The file exists and holds both lanes, secrets included (0600, like aliases.json).
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("providers.json not written: %v", err)
	}
	var onDisk map[string]map[string]any
	if err := json.Unmarshal(data, &onDisk); err != nil {
		t.Fatalf("providers.json not valid JSON: %v\n%s", err, data)
	}
	if len(onDisk) != 2 {
		t.Fatalf("expected 2 persisted providers, got %d: %s", len(onDisk), data)
	}
	if onDisk["vllm"]["base_url"] != "http://127.0.0.1:1/v1" {
		t.Fatalf("vllm base_url not persisted: %v", onDisk["vllm"])
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("providers.json perms = %v, want 0600", info.Mode().Perm())
	}

	// Simulated restart: a second store over the same file round-trips both.
	s2 := NewRuntimeProviderStore(path)
	loaded, err := s2.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded) != 2 {
		t.Fatalf("Load returned %d providers, want 2: %v", len(loaded), loaded)
	}
	got, ok := loaded["vllm"]
	if !ok || got.BaseURL != "http://127.0.0.1:1/v1" || got.APIKey != "sk-local" {
		t.Fatalf("vllm did not round-trip: %+v", got)
	}
	if !got.Quirks.ModelListPassthrough {
		t.Errorf("quirks did not round-trip: %+v", got.Quirks)
	}
	or, ok := loaded["openrouter"]
	if !ok || !or.Quirks.CodingPlanPath == false || or.Quirks.MaxTokensByModel["glm-4.5-air"] != 98304 {
		t.Fatalf("openrouter quirks did not round-trip: %+v", or.Quirks)
	}

	// Validation.
	if err := s1.Add(openaicompat.Config{Name: "", BaseURL: "https://x/v1"}); err == nil {
		t.Error("expected error for empty name")
	}
	if err := s1.Add(openaicompat.Config{Name: "x", BaseURL: ""}); err == nil {
		t.Error("expected error for empty base_url")
	}
	if err := s1.Add(openaicompat.Config{Name: "Bad/Name", BaseURL: "https://x/v1"}); err == nil {
		t.Error("expected error for invalid name pattern")
	}
	if err := s1.Add(openaicompat.Config{Name: "UPPER", BaseURL: "https://x/v1"}); err == nil {
		t.Error("expected error for uppercase name")
	}
	if len(s1.List()) != 2 {
		t.Fatalf("invalid adds must not change the store, got %d", len(s1.List()))
	}

	// Remove updates the file.
	if err := s1.Remove("vllm"); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if s1.Has("vllm") {
		t.Fatal("vllm still present after Remove")
	}
	data, _ = os.ReadFile(path)
	if strings.Contains(string(data), `"vllm"`) {
		t.Fatalf("providers.json still contains vllm: %s", data)
	}
	if err := s1.Remove("ghost"); err == nil {
		t.Fatal("expected error removing an unknown lane")
	}

	// In-memory store: no persistence, no path.
	mem := NewRuntimeProviderStore("")
	if mem.Path() != "" {
		t.Fatalf("in-memory store path = %q", mem.Path())
	}
	if err := mem.Add(openaicompat.Config{Name: "m", BaseURL: "https://m/v1"}); err != nil {
		t.Fatalf("in-memory add: %v", err)
	}
	if _, err := mem.Load(); err != nil {
		t.Fatalf("in-memory load: %v", err)
	}
	if len(mem.List()) != 1 {
		t.Fatalf("in-memory list = %d", len(mem.List()))
	}
}

// TestRuntimeProviderRestoreAcrossRestart covers the fresh-install bootstrap:
// a lane added at runtime is served again after a restart from the same
// DataDir, with no config file anywhere.
func TestRuntimeProviderRestoreAcrossRestart(t *testing.T) {
	dir := t.TempDir()

	cfg := DefaultConfig()
	cfg.DataDir = dir
	cfg.Server.APIKey = ""
	cfg.Server.ClientKeys = nil

	// First boot: no providers, empty DataDir.
	registry := provider.NewRegistry()
	srv := NewServer(cfg, registry)
	if registry.Len() != 0 {
		t.Fatalf("fresh install must register zero lanes, got %v", registry.Names())
	}

	// add_provider through the MCP surface mounted on the HTTP server.
	addBody := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add_provider","arguments":{"name":"zai","base_url":"http://127.0.0.1:1/v1","api_key":"sk-zai"}}}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(addBody)))
	if rec.Code != http.StatusOK {
		t.Fatalf("add_provider: HTTP %d: %s", rec.Code, rec.Body.String())
	}
	var rpcResp struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &rpcResp); err != nil {
		t.Fatalf("decode add_provider response: %v", err)
	}
	if rpcResp.Result.IsError || len(rpcResp.Result.Content) == 0 ||
		!strings.Contains(rpcResp.Result.Content[0].Text, `"registered": true`) {
		t.Fatalf("add_provider response: %s", rec.Body.String())
	}
	if _, ok := registry.Get("zai"); !ok {
		t.Fatal("zai not registered in the live registry")
	}

	// No config file was written; the lane lives in providers.json only.
	if _, err := os.Stat(filepath.Join(dir, "providers.json")); err != nil {
		t.Fatalf("providers.json missing: %v", err)
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.yaml"))
	if len(matches) != 0 {
		t.Fatalf("unexpected config files written: %v", matches)
	}

	// Restart from the same DataDir: the lane comes back before any traffic.
	registry2 := provider.NewRegistry()
	_ = NewServer(cfg, registry2)
	if _, ok := registry2.Get("zai"); !ok {
		t.Fatalf("zai not restored after restart, registry has %v", registry2.Names())
	}
	if registry2.Len() != 1 {
		t.Fatalf("expected exactly 1 restored lane, got %v", registry2.Names())
	}

	// And the restored lane is a real inference provider.
	p, _ := registry2.Get("zai")
	if p.Inference == nil || p.Inference.Name() != "zai" {
		t.Fatalf("restored lane is not an inference provider: %+v", p)
	}
}
