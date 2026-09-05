package server

import (
	"context"
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

// TestRuntimeProviderStore_LegacyFieldsIgnored covers the removal of the
// obsolete provider options (data_dir, workspace_id, session_cookie,
// refresh_url, token_file, device_auth_url, token_url and the
// auth_via_workspace_cookie quirk): a providers.json written by an older build
// still loads (unknown JSON fields are ignored), the lane keeps its real
// identity, credential state is re-derived from the server DataDir, and the
// next persist drops the legacy keys entirely.
func TestRuntimeProviderStore_LegacyFieldsIgnored(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")
	legacy := `{
  "opencode": {
    "name": "opencode",
    "base_url": "https://opencode.ai/zen/go/v1",
    "api_key": "sk-opencode",
    "data_dir": "/old/state",
    "workspace_id": "ws-legacy",
    "session_cookie": "sess-legacy",
    "refresh_url": "https://legacy.example/refresh",
    "token_file": "/old/state/augure-auth.json",
    "device_auth_url": "https://legacy.example/device",
    "token_url": "https://legacy.example/token",
    "quirks": {
      "auth_via_workspace_cookie": true,
      "model_list_passthrough": true
    }
  }
}`
	if err := os.WriteFile(path, []byte(legacy), 0o600); err != nil {
		t.Fatalf("write providers.json: %v", err)
	}

	s := NewRuntimeProviderStore(path)
	s.DefaultDataDir = dir
	loaded, err := s.Load()
	if err != nil {
		t.Fatalf("Load legacy providers.json: %v", err)
	}
	got, ok := loaded["opencode"]
	if !ok {
		t.Fatalf("legacy lane not loaded: %v", loaded)
	}
	if got.BaseURL != "https://opencode.ai/zen/go/v1" || got.APIKey != "sk-opencode" {
		t.Fatalf("lane identity lost: %+v", got)
	}
	if !got.Quirks.ModelListPassthrough {
		t.Errorf("model_list_passthrough did not survive the legacy load: %+v", got.Quirks)
	}
	if want := filepath.Join(dir, "credentials", "opencode"); got.DataDir != want {
		t.Errorf("credential dir = %q, want %q (legacy data_dir must not win)", got.DataDir, want)
	}

	// Re-persisting (add_provider is how a lane is rotated) drops the legacy keys.
	if err := s.Add(got); err != nil {
		t.Fatalf("re-add lane: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}
	for _, legacyKey := range []string{
		"data_dir", "workspace_id", "session_cookie", "refresh_url",
		"token_file", "device_auth_url", "token_url", "auth_via_workspace_cookie",
	} {
		if strings.Contains(string(data), `"`+legacyKey+`"`) {
			t.Errorf("providers.json still carries the removed key %q: %s", legacyKey, data)
		}
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

// laneBuilderCall records one invocation of RuntimeProviderStore.LaneBuilder.
type laneBuilderCall struct {
	name, kind, dataDir, apiKey string
}

// TestRuntimeProviderStore_CustomLaneAPIKeyAcrossRestart covers the custom-lane
// plumbing for key-authenticated kinds (anthropic): AddCustom persists the
// api_key and Restore hands name, kind, DataDir and the key to LaneBuilder so
// the lane rebuilds identically after a restart.
func TestRuntimeProviderStore_CustomLaneAPIKeyAcrossRestart(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")

	var calls []laneBuilderCall
	laneBuilder := func(name, kind, dataDir, apiKey string) (provider.Provider, error) {
		calls = append(calls, laneBuilderCall{name: name, kind: kind, dataDir: dataDir, apiKey: apiKey})
		return provider.Provider{Inference: &fakeInferenceProvider{name: name}}, nil
	}
	byKind := func() map[string]laneBuilderCall {
		out := map[string]laneBuilderCall{}
		for _, c := range calls {
			out[c.kind] = c
		}
		return out
	}

	s1 := NewRuntimeProviderStore(path)
	s1.DefaultDataDir = dir
	s1.LaneBuilder = laneBuilder
	if err := s1.AddCustom("anthropic", "anthropic", "sk-ant-test"); err != nil {
		t.Fatalf("AddCustom anthropic: %v", err)
	}
	if err := s1.AddCustom("codex", "codex", ""); err != nil {
		t.Fatalf("AddCustom codex: %v", err)
	}

	onDisk, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}
	if !strings.Contains(string(onDisk), "sk-ant-test") || !strings.Contains(string(onDisk), `"kind": "anthropic"`) {
		t.Fatalf("providers.json missing the anthropic key/kind: %s", onDisk)
	}

	registry := provider.NewRegistry()
	if got := s1.Restore(registry); len(got) != 2 {
		t.Fatalf("Restore registered %v, want [anthropic codex]", got)
	}
	if len(calls) != 2 {
		t.Fatalf("LaneBuilder called %d times, want 2: %+v", len(calls), calls)
	}
	if c := byKind()["anthropic"]; c.apiKey != "sk-ant-test" || c.dataDir != dir || c.name != "anthropic" {
		t.Fatalf("anthropic LaneBuilder call unexpected: %+v", c)
	}
	if c := byKind()["codex"]; c.apiKey != "" || c.name != "codex" {
		t.Fatalf("codex LaneBuilder call unexpected: %+v", c)
	}

	// Restart: a fresh store over the same file rebuilds the lanes with the
	// persisted key.
	s2 := NewRuntimeProviderStore(path)
	s2.DefaultDataDir = dir
	s2.LaneBuilder = laneBuilder
	calls = nil
	registry2 := provider.NewRegistry()
	if got := s2.Restore(registry2); len(got) != 2 {
		t.Fatalf("second Restore registered %v, want [anthropic codex]", got)
	}
	if c := byKind()["anthropic"]; c.apiKey != "sk-ant-test" {
		t.Fatalf("anthropic key not restored: %+v", c)
	}
	if _, ok := registry2.Get("anthropic"); !ok {
		t.Fatal("anthropic lane not registered after restart")
	}
	if _, ok := registry2.Get("codex"); !ok {
		t.Fatal("codex lane not registered after restart")
	}
}

// freebuffMarkerActor is the non-actor marker an MCP add_provider call can
// carry: it marks the lane as freebuff without being a usable actor.
type freebuffMarkerActor struct{}

// fakeFreebuffActor stands in for the real adapter the cmd installs.
type fakeFreebuffActor struct{}

func (fakeFreebuffActor) Acquire(ctx context.Context) error { return nil }
func (fakeFreebuffActor) Release()                          {}

// TestRuntimeProviderStore_FreebuffActorEnrichment covers a freebuff lane
// added over MCP: the stored marker actor is swapped for the real one through
// ActorBuilder immediately (so the live lane works), the flag persists as
// quirks.freebuff_actor=true, and a restart rebuilds the actor again.
func TestRuntimeProviderStore_FreebuffActorEnrichment(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")

	built := 0
	builder := func(cfg openaicompat.Config) any {
		built++
		return any(fakeFreebuffActor{})
	}

	s1 := NewRuntimeProviderStore(path)
	s1.DefaultDataDir = dir
	s1.ActorBuilder = builder

	if err := s1.Add(openaicompat.Config{
		Name:    "freebuff",
		BaseURL: "https://www.codebuff.com/api/v1",
		APIKey:  "fb-key",
		Quirks: openaicompat.Quirks{
			FreebuffActor:       freebuffMarkerActor{},
			FreebuffDefaultTool: true,
		},
	}); err != nil {
		t.Fatalf("Add freebuff: %v", err)
	}

	// The marker was replaced with the actor the builder produced.
	stored := s1.List()["freebuff"]
	if _, ok := stored.Quirks.FreebuffActor.(fakeFreebuffActor); !ok {
		t.Fatalf("stored actor = %T, want fakeFreebuffActor", stored.Quirks.FreebuffActor)
	}
	if built != 1 {
		t.Errorf("ActorBuilder called %d times, want 1", built)
	}

	// The persisted projection is the boolean flag, never the actor.
	onDisk, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}
	if !strings.Contains(string(onDisk), `"freebuff_actor": true`) {
		t.Fatalf("providers.json missing freebuff_actor: %s", onDisk)
	}
	if strings.Contains(string(onDisk), "fakeFreebuffActor") {
		t.Fatalf("providers.json leaked the actor: %s", onDisk)
	}

	// Restart: the flag rebuilds the actor for the restored lane.
	s2 := NewRuntimeProviderStore(path)
	s2.ActorBuilder = builder
	loaded, err := s2.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if _, ok := loaded["freebuff"].Quirks.FreebuffActor.(fakeFreebuffActor); !ok {
		t.Fatalf("restored actor = %T, want fakeFreebuffActor", loaded["freebuff"].Quirks.FreebuffActor)
	}
	if !loaded["freebuff"].Quirks.FreebuffDefaultTool {
		t.Error("freebuff_default_tool did not round-trip")
	}
}

// TestRuntimeProviderStore_RestoreDoesNotAttachFreebuffActorToOrdinaryLanes
// covers the restart path for an ordinary OpenAI-compatible lane: the persisted
// quirks.freebuff_actor boolean is the only Freebuff discriminator, so a lane
// stored without it must come back with no actor at all -- even though the
// runtime ActorBuilder is installed -- and must not invoke that builder. The
// in-memory actor being nil is never evidence of Freebuff.
func TestRuntimeProviderStore_RestoreDoesNotAttachFreebuffActorToOrdinaryLanes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")

	s1 := NewRuntimeProviderStore(path)
	s1.DefaultDataDir = dir
	if err := s1.Add(openaicompat.Config{
		Name:    "vllm",
		BaseURL: "http://127.0.0.1:1/v1",
		APIKey:  "sk-local",
	}); err != nil {
		t.Fatalf("add vllm: %v", err)
	}
	onDisk, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}
	if strings.Contains(string(onDisk), "freebuff") {
		t.Fatalf("ordinary lane persisted a freebuff marker: %s", onDisk)
	}

	// Restart the way cmd does it: the store loads providers.json while
	// ActorBuilder is still nil, and the builder is installed afterwards.
	s2 := NewRuntimeProviderStore(path)
	s2.DefaultDataDir = dir
	if cfg := s2.List()["vllm"]; cfg.Quirks.FreebuffActor != nil {
		t.Fatalf("ordinary lane loaded with an actor: %T", cfg.Quirks.FreebuffActor)
	}
	built := 0
	var builtFor []string
	s2.ActorBuilder = func(cfg openaicompat.Config) any {
		built++
		builtFor = append(builtFor, cfg.Name)
		return any(fakeFreebuffActor{})
	}

	registry := provider.NewRegistry()
	if got := s2.Restore(registry); len(got) != 1 || got[0] != "vllm" {
		t.Fatalf("Restore registered %v, want [vllm]", got)
	}
	if built != 0 {
		t.Fatalf("ActorBuilder invoked for %v (actor == nil must not mean Freebuff)", builtFor)
	}
	if cfg := s2.List()["vllm"]; cfg.Quirks.FreebuffActor != nil {
		t.Fatalf("ordinary lane gained a Freebuff actor on restore: %T", cfg.Quirks.FreebuffActor)
	}
	p, ok := registry.Get("vllm")
	if !ok {
		t.Fatal("vllm not registered by Restore")
	}
	if p.Quota != nil {
		t.Errorf("ordinary restored lane exposes a quota provider: %T", p.Quota)
	}
	if p.Capabilities.MaxConcurrentRequests == 1 || p.Capabilities.SessionAffinity || p.Capabilities.Queueing {
		t.Errorf("ordinary restored lane gained freebuff capabilities: %+v", p.Capabilities)
	}
}

// TestRuntimeProviderStore_RestoreMixedLanes covers one providers.json holding
// both an ordinary OpenAI-compatible lane and a freebuff lane: after a restart
// only the lane persisted with quirks.freebuff_actor true gets an actor, and the
// builder runs exactly once.
func TestRuntimeProviderStore_RestoreMixedLanes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")
	file := `{
  "freebuff": {
    "kind": "openaicompat",
    "name": "freebuff",
    "base_url": "https://www.codebuff.com/api/v1",
    "api_key": "fb-key",
    "quirks": {"freebuff_actor": true, "freebuff_default_tool": true}
  },
  "ordinary": {
    "kind": "openaicompat",
    "name": "ordinary",
    "base_url": "http://127.0.0.1:1/v1",
    "api_key": "sk-ord",
    "quirks": {}
  }
}`
	if err := os.WriteFile(path, []byte(file), 0o600); err != nil {
		t.Fatalf("write providers.json: %v", err)
	}

	s := NewRuntimeProviderStore(path)
	s.DefaultDataDir = dir
	built := 0
	var builtFor []string
	s.ActorBuilder = func(cfg openaicompat.Config) any {
		built++
		builtFor = append(builtFor, cfg.Name)
		return any(fakeFreebuffActor{})
	}

	registry := provider.NewRegistry()
	got := s.Restore(registry)
	if len(got) != 2 || got[0] != "freebuff" || got[1] != "ordinary" {
		t.Fatalf("Restore registered %v, want [freebuff ordinary]", got)
	}
	if built != 1 {
		t.Fatalf("ActorBuilder called %d times for %v, want exactly once (freebuff only)", built, builtFor)
	}
	if len(builtFor) != 1 || builtFor[0] != "freebuff" {
		t.Fatalf("ActorBuilder built for %v, want [freebuff]", builtFor)
	}

	fb, ok := registry.Get("freebuff")
	if !ok {
		t.Fatal("freebuff lane not registered")
	}
	if fb.Quota == nil {
		t.Error("freebuff restored lane lost its quota provider")
	}
	if fb.Capabilities.MaxConcurrentRequests != 1 || !fb.Capabilities.SessionAffinity {
		t.Errorf("freebuff restored lane lost actor capabilities: %+v", fb.Capabilities)
	}
	ord, ok := registry.Get("ordinary")
	if !ok {
		t.Fatal("ordinary lane not registered")
	}
	if ord.Quota != nil {
		t.Errorf("ordinary restored lane exposes a quota provider: %T", ord.Quota)
	}
	if ord.Capabilities.MaxConcurrentRequests == 1 || ord.Capabilities.SessionAffinity || ord.Capabilities.Queueing {
		t.Errorf("ordinary restored lane gained freebuff capabilities: %+v", ord.Capabilities)
	}

	// The loaded freebuff lane still has a nil in-memory actor (the builder only
	// ran for the restore copy), so re-persisting the file -- which any later
	// add_provider does -- must not drop the persisted flag.
	if err := s.Add(openaicompat.Config{Name: "extra", BaseURL: "http://127.0.0.1:2/v1", APIKey: "sk-extra"}); err != nil {
		t.Fatalf("add extra lane: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}
	if !strings.Contains(string(data), `"freebuff_actor": true`) {
		t.Fatalf("re-persist dropped the freebuff flag: %s", data)
	}
}
