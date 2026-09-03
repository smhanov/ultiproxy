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

	if cfg.Server.Addr != "127.0.0.1:8317" {
		t.Errorf("expected Addr 127.0.0.1:8317, got %s", cfg.Server.Addr)
	}
	if cfg.Server.APIKey != "admin-secret-key" {
		t.Errorf("expected APIKey admin-secret-key, got %s", cfg.Server.APIKey)
	}
	if cfg.Server.ClientKeys["cursor"] != "sk-cursor-secret" {
		t.Errorf("expected cursor client key, got %v", cfg.Server.ClientKeys)
	}
	if cfg.Storage.DBPath != "ultiproxy.db" {
		t.Errorf("expected DBPath ultiproxy.db, got %s", cfg.Storage.DBPath)
	}
}
