package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
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
