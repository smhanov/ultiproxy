package server

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelCatalogCRUDAndPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "aliases.json")

	c, err := NewModelCatalog(map[string]ModelAlias{
		"qwenpoint-3.8": {Provider: "vllm", Upstream: "Qwen/Qwen3.8-Instruct-AWQ", PricingTag: "local"},
	}, path)
	if err != nil {
		t.Fatalf("NewModelCatalog: %v", err)
	}

	up, ok := c.UpstreamName("qwenpoint-3.8")
	if !ok || up != "Qwen/Qwen3.8-Instruct-AWQ" {
		t.Fatalf("UpstreamName = %q, %v; want Qwen/Qwen3.8-Instruct-AWQ, true", up, ok)
	}

	if err := c.Set("claude-sonnet", ModelAlias{Provider: "copilot", Upstream: "claude-sonnet-4-6"}); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if _, ok := c.UpstreamName("missing"); ok {
		t.Fatal("unexpected alias present")
	}

	// Reload from persisted file: runtime alias must survive.
	c2, err := NewModelCatalog(map[string]ModelAlias{}, path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if up, ok := c2.UpstreamName("claude-sonnet"); !ok || up != "claude-sonnet-4-6" {
		t.Fatalf("persisted alias not restored: %q, %v", up, ok)
	}
	if _, ok := c2.UpstreamName("qwenpoint-3.8"); !ok {
		t.Fatal("config alias missing after reload")
	}

	if err := c.Remove("claude-sonnet"); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if _, ok := c.UpstreamName("claude-sonnet"); ok {
		t.Fatal("alias still present after remove")
	}

	if err := c.Set("", ModelAlias{Provider: "x", Upstream: "y"}); err == nil {
		t.Fatal("expected error for empty alias")
	}
}

func TestModelCatalogPersistUses060Perms(t *testing.T) {
	path := filepath.Join(t.TempDir(), "aliases.json")
	c, _ := NewModelCatalog(nil, path)
	if err := c.Set("a", ModelAlias{Provider: "p", Upstream: "u"}); err != nil {
		t.Fatalf("Set: %v", err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("perm = %o; want 600", fi.Mode().Perm())
	}
}
