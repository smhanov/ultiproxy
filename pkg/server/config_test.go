package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

// writeTempConfig writes body to a YAML file inside a fresh temp dir and
// returns its path. Tests must never touch the developer working directory.
func writeTempConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return path
}

// TestLoadConfig_RejectsUnknownFields pins the fail-closed parsing contract:
// legacy keys from the old installer template (server.listen, server.api_keys)
// and the removed routing/providers/accounting blocks must abort startup
// instead of silently disabling authentication.
func TestLoadConfig_RejectsUnknownFields(t *testing.T) {
	cases := map[string]string{
		"server.listen": `
server:
  listen: "127.0.0.1:9050"
  api_key: "sk-up-keep"
`,
		"server.api_keys": `
server:
  addr: "127.0.0.1:9050"
  api_keys:
    - "sk-up-local-agent-key"
`,
		"top-level routing": `
server:
  addr: "127.0.0.1:9050"
routing:
  strategy: "quota-priority"
`,
		"top-level providers": `
server:
  addr: "127.0.0.1:9050"
providers:
  copilot:
    enabled: true
`,
		"top-level accounting": `
server:
  addr: "127.0.0.1:9050"
accounting:
  enabled: true
`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			cfg, err := LoadConfig(writeTempConfig(t, body))
			if err == nil {
				t.Fatalf("LoadConfig accepted unknown key(s) %q: %+v", name, cfg)
			}
			if !strings.Contains(err.Error(), "parse config file") {
				t.Errorf("expected parse config file error, got %v", err)
			}
		})
	}
}

// TestLoadConfig_CurrentSchemaSetsAPIKey proves a current-schema file loads
// and that the single bearer key lands in Server.APIKey.
func TestLoadConfig_CurrentSchemaSetsAPIKey(t *testing.T) {
	dir := t.TempDir()
	body := `
server:
  addr: "127.0.0.1:19050"
  api_key: "sk-up-installer-generated"
data_dir: "` + filepath.ToSlash(dir) + `"
storage:
  db_path: "` + filepath.ToSlash(filepath.Join(dir, "ultiproxy.db")) + `"
`
	cfg, err := LoadConfig(writeTempConfig(t, body))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Server.APIKey != "sk-up-installer-generated" {
		t.Errorf("Server.APIKey = %q, want %q", cfg.Server.APIKey, "sk-up-installer-generated")
	}
	if cfg.Server.Addr != "127.0.0.1:19050" {
		t.Errorf("Server.Addr = %q, want 127.0.0.1:19050", cfg.Server.Addr)
	}
	if want := filepath.FromSlash(filepath.Join(dir, "ultiproxy.db")); cfg.Storage.DBPath != want {
		t.Errorf("Storage.DBPath = %q, want %q", cfg.Storage.DBPath, want)
	}
	if cfg.DataDir != dir {
		t.Errorf("DataDir = %q, want %q", cfg.DataDir, dir)
	}
}

// TestLoadConfig_ZeroConfigDefaults covers the zero-config/MCP path: no key
// configured means open access, and defaults are still filled in.
func TestLoadConfig_ZeroConfigDefaults(t *testing.T) {
	cfg, err := LoadConfig("")
	if err != nil {
		t.Fatalf("LoadConfig(empty): %v", err)
	}
	if cfg.Server.Addr == "" {
		t.Error("expected default addr to be filled in")
	}
	if cfg.Storage.DBPath == "" {
		t.Error("expected default db_path to be filled in")
	}
}

// TestLLMsTxtServedFromEmbedWhenFileMissing reproduces the packaged-binary
// case: there is no llms.txt next to the process, so GET /llms.txt must be
// served from the copy embedded at build time instead of 404ing.
func TestLLMsTxtServedFromEmbedWhenFileMissing(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	if _, err := os.Stat(filepath.Join(dir, "llms.txt")); err == nil {
		t.Fatal("test setup: temp dir unexpectedly contains llms.txt")
	}

	cfg := DefaultConfig()
	cfg.DataDir = dir
	srv := NewServer(cfg, provider.NewRegistry())
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/llms.txt")
	if err != nil {
		t.Fatalf("GET /llms.txt: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /llms.txt status = %d, want 200", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if len(body) == 0 {
		t.Fatal("embedded llms.txt fallback served an empty body")
	}
	if !strings.HasPrefix(string(body), "# Ultiproxy") {
		head := string(body)
		if i := strings.IndexByte(head, '\n'); i >= 0 {
			head = head[:i]
		}
		t.Errorf("embedded body does not look like llms.txt: %q", head)
	}
}

// TestLLMsTxtFilesystemOverrideWins documents the precedence rule: an
// llms.txt found on disk still wins over the embedded copy, so an operator
// can override the served document without rebuilding.
func TestLLMsTxtFilesystemOverrideWins(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	if err := os.WriteFile(filepath.Join(dir, "llms.txt"), []byte("# local override\n"), 0o644); err != nil {
		t.Fatalf("write override: %v", err)
	}

	cfg := DefaultConfig()
	cfg.DataDir = dir
	srv := NewServer(cfg, provider.NewRegistry())
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/llms.txt")
	if err != nil {
		t.Fatalf("GET /llms.txt: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /llms.txt status = %d, want 200", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if string(body) != "# local override\n" {
		t.Errorf("body = %q, want the on-disk override", string(body))
	}
}
