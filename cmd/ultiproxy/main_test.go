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
	// Model aliases present from config.
	if cfg.Server.Models == nil || cfg.Server.Models["zai-flash"].Upstream != "glm-5.3-flash" {
		t.Errorf("expected zai-flash alias mapping to glm-5.3-flash, got %+v", cfg.Server.Models)
	}
	if cfg.Storage.DBPath == "" {
		t.Errorf("expected DBPath set, got empty")
	}
}
