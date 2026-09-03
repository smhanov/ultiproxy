package main

import (
	"testing"

	"github.com/smhanov/ultiproxy/pkg/server"
)

func TestExampleConfigValid(t *testing.T) {
	cfg, err := server.LoadConfig("config.example.yaml")
	if err != nil {
		t.Fatalf("failed to load config.example.yaml: %v", err)
	}

	if cfg.Server.Addr != "127.0.0.1:9050" {
		t.Errorf("expected Addr 127.0.0.1:9050, got %s", cfg.Server.Addr)
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
