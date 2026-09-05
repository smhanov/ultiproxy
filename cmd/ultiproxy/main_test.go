package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/server"
)

func TestExampleConfigValid(t *testing.T) {
	cfg, err := server.LoadConfig("config.example.yaml")
	if err != nil {
		t.Fatalf("failed to load config.example.yaml: %v", err)
	}

	if cfg.Server.Addr != "0.0.0.0:9050" && cfg.Server.Addr != "127.0.0.1:9050" {
		t.Errorf("expected Addr 0.0.0.0:9050 or 127.0.0.1:9050, got %s", cfg.Server.Addr)
	}
	// Open-access default: no api_key configured means no auth required.
	if cfg.Server.APIKey != "" {
		t.Errorf("expected empty APIKey (open access), got %q", cfg.Server.APIKey)
	}
	// Model aliases present from config across requested providers.
	if cfg.Server.Models == nil {
		t.Fatalf("expected models in config example")
	}
	expected := map[string]string{
		"gemini-3.8-flash-high":      "antigravity",
		"glm-5.3-flash":              "zai",
		"glm-5.3":                    "zai",
		"opencode-deepseek-v4-flash": "opencode",
		"gpt-5.6-luna":               "freebuff",
		"freebuff-deepseek-v4-flash": "freebuff",
		"grok-4.6":                   "xai",
	}
	for alias, provider := range expected {
		entry, ok := cfg.Server.Models[alias]
		if !ok {
			t.Errorf("expected alias %q in config models", alias)
			continue
		}
		if provider != "" && entry.Provider != provider {
			t.Errorf("alias %q: expected provider %q, got %q", alias, provider, entry.Provider)
		}
	}
	// Timeouts include slow-lane defaults.
	if cfg.Server.Timeouts == nil || cfg.Server.Timeouts["vllm"] == "" {
		t.Errorf("expected vllm timeout in config example, got %+v", cfg.Server.Timeouts)
	}
	if cfg.Storage.DBPath == "" {
		t.Errorf("expected DBPath set, got empty")
	}
}

// TestRuntimeLaneBuilderAnthropicAndCodex covers the custom-wire kinds the MCP
// add_provider tool accepts: kind=anthropic builds a Claude lane from the
// persisted api_key, and kind=codex builds the OAuth device lane (registering
// before login so quota stays readable). Both must register into a registry.
func TestRuntimeLaneBuilderAnthropicAndCodex(t *testing.T) {
	dir := t.TempDir()
	registry := provider.NewRegistry()

	anthropic, err := runtimeLaneBuilder("anthropic", "anthropic", dir, "sk-ant-test")
	if err != nil {
		t.Fatalf("anthropic lane: %v", err)
	}
	if anthropic.Inference == nil || anthropic.Inference.Name() != "anthropic" {
		t.Fatalf("anthropic lane is not an inference provider: %+v", anthropic)
	}
	registry.Register(anthropic)
	if _, ok := registry.Get("anthropic"); !ok {
		t.Fatal("anthropic lane not registered")
	}

	// A missing api_key must fail loudly instead of registering a dead lane.
	if _, err := runtimeLaneBuilder("anthropic", "anthropic", dir, ""); err == nil {
		t.Fatal("expected an error for anthropic without an api_key")
	}

	codexLane, err := runtimeLaneBuilder("codex", "codex", dir, "")
	if err != nil {
		t.Fatalf("codex lane: %v", err)
	}
	if codexLane.Inference == nil || codexLane.Inference.Name() != "codex" {
		t.Fatalf("codex lane is not an inference provider: %+v", codexLane)
	}
	registry.Register(codexLane)
	if _, ok := registry.Get("codex"); !ok {
		t.Fatal("codex lane not registered")
	}
	// The codex credential store is created under <dataDir>/credentials/codex.
	if _, err := os.Stat(filepath.Join(dir, "credentials", "codex")); err != nil {
		t.Fatalf("codex credential dir missing: %v", err)
	}

	copilotLane, err := runtimeLaneBuilder("copilot", "copilot", dir, "gho_test_token")
	if err != nil {
		t.Fatalf("copilot lane: %v", err)
	}
	if copilotLane.Inference == nil || copilotLane.Inference.Name() != "copilot" {
		t.Fatalf("copilot lane is not an inference provider: %+v", copilotLane)
	}
	registry.Register(copilotLane)
	if _, ok := registry.Get("copilot"); !ok {
		t.Fatal("copilot lane not registered")
	}

	// Unknown kinds are rejected.
	if _, err := runtimeLaneBuilder("nope", "nope", dir, ""); err == nil {
		t.Fatal("expected an error for an unsupported kind")
	}
}

// TestRuntimeLaneBuilderFreebuff covers kind=freebuff: the lane is the
// OpenAI-compatible Codebuff wire plus the serialized-request actor, so it must
// build from the api_key handed to add_provider (and persist that key for later
// runs), register with a quota surface and serialize requests.
func TestRuntimeLaneBuilderFreebuff(t *testing.T) {
	// Neutralize a developer shell so the assertions below only see the
	// explicit api_key / the temp-dir state.
	t.Setenv("ULTIPROXY_FREEBUFF_TOKEN", "")
	t.Setenv("FREEBUFF_TOKEN", "")

	dir := t.TempDir()
	registry := provider.NewRegistry()

	lane, err := runtimeLaneBuilder("freebuff", "freebuff", dir, "fb-token-1")
	if err != nil {
		t.Fatalf("freebuff lane: %v", err)
	}
	if lane.Inference == nil || lane.Inference.Name() != "freebuff" {
		t.Fatalf("freebuff lane is not an inference provider: %+v", lane)
	}
	// The actor-backed quota surface must be attached: Freebuff quota comes
	// from the account actor, not from a credits observer.
	if lane.Quota == nil {
		t.Fatal("expected a Quota provider on the freebuff lane")
	}
	// Freebuff requests are serialized through the session actor.
	if lane.Capabilities.MaxConcurrentRequests != 1 {
		t.Errorf("MaxConcurrentRequests = %d, want 1 (serialized session)", lane.Capabilities.MaxConcurrentRequests)
	}
	if !lane.Capabilities.SessionAffinity {
		t.Error("SessionAffinity = false, want true (freebuff session affinity)")
	}
	registry.Register(lane)
	if _, ok := registry.Get("freebuff"); !ok {
		t.Fatal("freebuff lane not registered")
	}

	// The explicit key is persisted so a restart rebuilds the actor without
	// the caller re-supplying it.
	tok, err := os.ReadFile(filepath.Join(dir, "freebuff_token"))
	if err != nil {
		t.Fatalf("freebuff_token not persisted: %v", err)
	}
	if got := strings.TrimSpace(string(tok)); got != "fb-token-1" {
		t.Errorf("freebuff_token = %q, want %q", got, "fb-token-1")
	}

	// The lane is also usable from a lane name other than "freebuff".
	named, err := runtimeLaneBuilder("bufflane", "freebuff", dir, "fb-token-2")
	if err != nil {
		t.Fatalf("freebuff lane (bufflane): %v", err)
	}
	if named.Inference == nil || named.Inference.Name() != "bufflane" {
		t.Fatalf("freebuff lane (bufflane) has the wrong name: %+v", named)
	}
	if named.Quota == nil {
		t.Fatal("expected a Quota provider on the bufflane freebuff lane")
	}
	// An already-persisted token is never clobbered by a later key.
	tok2, err := os.ReadFile(filepath.Join(dir, "freebuff_token"))
	if err != nil {
		t.Fatalf("freebuff_token disappeared: %v", err)
	}
	if got := strings.TrimSpace(string(tok2)); got != "fb-token-1" {
		t.Errorf("freebuff_token = %q, want the original %q", got, "fb-token-1")
	}

	// Without any key at all the lane still builds (it registers and reports
	// an honest quota instead of disappearing), just without the actor.
	bare, err := runtimeLaneBuilder("freebuff-bare", "freebuff", t.TempDir(), "")
	if err != nil {
		t.Fatalf("freebuff lane without a token: %v", err)
	}
	if bare.Inference == nil || bare.Inference.Name() != "freebuff-bare" {
		t.Fatalf("tokenless freebuff lane is not an inference provider: %+v", bare)
	}
	if bare.Quota != nil {
		t.Errorf("tokenless freebuff lane advertises a quota surface: %+v", bare.Quota)
	}
}

// TestNewFreebuffActorExplicitToken verifies the actor builder itself: an
// explicit token wins over discovery, is persisted once, and yields an actor
// that satisfies the adapter surface used by the openaicompat quirks.
func TestNewFreebuffActorExplicitToken(t *testing.T) {
	t.Setenv("ULTIPROXY_FREEBUFF_TOKEN", "")
	t.Setenv("FREEBUFF_TOKEN", "")

	dir := t.TempDir()
	if got := newFreebuffActor(dir, ""); got != nil {
		t.Fatalf("newFreebuffActor with no token anywhere = %T, want nil", got)
	}

	actor := newFreebuffActor(dir, "fb-key-1")
	if actor == nil {
		t.Fatal("newFreebuffActor with an explicit token returned nil")
	}
	// The adapter must satisfy the surfaces the openaicompat quirks assert.
	if _, ok := actor.(interface {
		Acquire(context.Context) error
		Release()
	}); !ok {
		t.Fatalf("actor %T does not implement the freebuff lock interface", actor)
	}
	if _, ok := actor.(interface {
		FetchUsage(context.Context, string) ([]byte, error)
	}); !ok {
		t.Fatalf("actor %T does not implement FetchUsage", actor)
	}
	tok, err := os.ReadFile(filepath.Join(dir, "freebuff_token"))
	if err != nil {
		t.Fatalf("freebuff_token not written: %v", err)
	}
	if got := strings.TrimSpace(string(tok)); got != "fb-key-1" {
		t.Errorf("freebuff_token = %q, want fb-key-1", got)
	}

	// A second, different key never overwrites the stored one.
	_ = newFreebuffActor(dir, "fb-key-2")
	tok2, _ := os.ReadFile(filepath.Join(dir, "freebuff_token"))
	if got := strings.TrimSpace(string(tok2)); got != "fb-key-1" {
		t.Errorf("freebuff_token = %q, want the first key fb-key-1", got)
	}

	// The runtime hook hands the lane's own key to the builder.
	hook := runtimeFreebuffActorBuilder(dir)
	if got := hook(openaicompat.Config{APIKey: "fb-key-3", DataDir: t.TempDir()}); got == nil {
		t.Fatal("runtimeFreebuffActorBuilder returned nil for a lane with an api_key")
	}
	if got := hook(openaicompat.Config{}); got == nil {
		t.Fatal("runtimeFreebuffActorBuilder returned nil for a lane falling back to the state token")
	}
}
