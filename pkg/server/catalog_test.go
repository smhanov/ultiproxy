package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
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

// TestModelCatalog_ConcurrentSetsBothSurviveOnDisk (T022 AC1): two or more
// concurrent Set calls must both survive on disk, and the final file must be
// valid JSON reflecting the last completed mutation -- no writer may rename a
// shared .tmp file out from under another, and no older snapshot may overwrite
// a newer mutation.
func TestModelCatalog_ConcurrentSetsBothSurviveOnDisk(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "aliases.json")
	c, err := NewModelCatalog(map[string]ModelAlias{
		"seed": {Provider: "vllm", Upstream: "seed-upstream"},
	}, path)
	if err != nil {
		t.Fatalf("NewModelCatalog: %v", err)
	}

	const workers = 32
	errs := make([]error, workers)
	names := make([]string, workers)
	for i := range names {
		names[i] = fmt.Sprintf("runtime-%02d", i)
	}
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			errs[i] = c.Set(names[i], ModelAlias{
				Provider: "vllm",
				Upstream: fmt.Sprintf("upstream-%02d", i),
			})
		}(i)
	}
	close(start)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent Set(%q) reported a persistence error: %v", names[i], err)
		}
	}

	// The last completed mutation must be on disk with everything else intact.
	if err := c.Set("final", ModelAlias{Provider: "vllm", Upstream: "final-upstream"}); err != nil {
		t.Fatalf("Set(final): %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read aliases.json: %v", err)
	}
	var onDisk map[string]ModelAlias
	if err := json.Unmarshal(data, &onDisk); err != nil {
		t.Fatalf("aliases.json is not valid JSON (torn write): %v\n%s", err, data)
	}
	want := append([]string{"seed", "final"}, names...)
	for _, name := range want {
		if _, ok := onDisk[name]; !ok {
			t.Errorf("alias %q lost from aliases.json after concurrent writes: %s", name, data)
		}
	}
	if onDisk["final"].Upstream != "final-upstream" {
		t.Errorf("last completed mutation not on disk: %+v", onDisk["final"])
	}

	// No shared/leftover temp file may remain behind.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Errorf("leftover temp file %q next to aliases.json", e.Name())
		}
	}

	// A restart from the same file restores every concurrent mutation.
	c2, err := NewModelCatalog(nil, path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	for _, name := range want {
		if _, ok := c2.UpstreamName(name); !ok {
			t.Errorf("alias %q missing after reload", name)
		}
	}
}

// TestModelCatalog_PersistFailureLeavesLiveStateUnchanged (T022 AC2): when the
// disk write fails, the alias must not become live and pre-existing aliases
// must survive untouched, while the caller still gets the error.
func TestModelCatalog_PersistFailureLeavesLiveStateUnchanged(t *testing.T) {
	dir := t.TempDir()
	// A regular file where a directory is needed: MkdirAll fails with ENOTDIR,
	// which is a deterministic persist failure even when running as root.
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o600); err != nil {
		t.Fatalf("write blocker: %v", err)
	}

	c, err := NewModelCatalog(map[string]ModelAlias{
		"keep": {Provider: "vllm", Upstream: "keep-upstream"},
	}, filepath.Join(dir, "aliases.json"))
	if err != nil {
		t.Fatalf("NewModelCatalog: %v", err)
	}
	c.persistPath = filepath.Join(blocker, "aliases.json")

	if err := c.Set("new", ModelAlias{Provider: "vllm", Upstream: "new-upstream"}); err == nil {
		t.Fatal("Set must fail when persistence fails")
	}
	if _, ok := c.Get("new"); ok {
		t.Error("alias became live although persistence failed")
	}
	if _, ok := c.Get("keep"); !ok {
		t.Error("pre-existing alias lost after a failed Set")
	}
	if got := len(c.List()); got != 1 {
		t.Errorf("catalog holds %d aliases after failed Set; want 1", got)
	}

	if err := c.Remove("keep"); err == nil {
		t.Fatal("Remove must fail when persistence fails")
	}
	if _, ok := c.Get("keep"); !ok {
		t.Error("Remove was applied to live state although persistence failed")
	}
}

// TestModelCatalog_CorruptPersistenceFileFailsConstruction (T022 AC3): a
// truncated or garbage aliases.json must be reported as an error instead of
// silently succeeding with an empty runtime overlay.
func TestModelCatalog_CorruptPersistenceFileFailsConstruction(t *testing.T) {
	for name, content := range map[string]string{
		"truncated": `{"claude-sonnet": {"provider": "copilot"`,
		"garbage":   "\x00not json at all",
		"empty":     "",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "aliases.json")
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Fatalf("write: %v", err)
			}
			if _, err := NewModelCatalog(nil, path); err == nil {
				t.Fatalf("NewModelCatalog silently accepted corrupt %s persistence file", name)
			}
		})
	}
}

// TestRouter_ConcurrentSetModelAliasAllSurvive (T022 AC1): concurrent MCP
// set_model_alias calls must all be acknowledged, all survive in
// data_dir/aliases.json, and leave valid JSON of the last completed mutation
// behind -- no shared .tmp path, no lost alias, no tool error.
func TestRouter_ConcurrentSetModelAliasAllSurvive(t *testing.T) {
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.DataDir = dir

	lane := &fakeInferenceProvider{name: "vllm"}
	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: lane})

	srv := NewServer(cfg, registry, WithStateManager(state.NewStateManager()))

	const workers = 16
	results := make([]string, workers)
	aliases := make([]string, workers)
	for i := range aliases {
		aliases[i] = fmt.Sprintf("runtime-%02d", i)
	}
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			args := fmt.Sprintf(`{"alias":%q,"provider":"vllm","upstream":%q}`, aliases[i], fmt.Sprintf("up-%02d", i))
			body := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":"tools/call","params":{"name":"set_model_alias","arguments":%s}}`, i, args)
			rec := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body)))
			results[i] = fmt.Sprintf("HTTP %d: %s", rec.Code, rec.Body.String())
		}(i)
	}
	close(start)
	wg.Wait()

	for i, r := range results {
		// A successful tools/call result reports ok:true and carries no
		// "isError":true; a persistence failure would surface as an error.
		if strings.Contains(r, `"isError":true`) || !strings.Contains(r, `\"ok\": true`) {
			t.Errorf("set_model_alias(%q) did not succeed: %s", aliases[i], r)
		}
	}

	path := filepath.Join(dir, "aliases.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read aliases.json: %v", err)
	}
	var onDisk map[string]ModelAlias
	if err := json.Unmarshal(data, &onDisk); err != nil {
		t.Fatalf("aliases.json is not valid JSON (torn write): %v\n%s", err, data)
	}
	for _, a := range aliases {
		if _, ok := onDisk[a]; !ok {
			t.Errorf("alias %q lost from aliases.json: %s", a, data)
		}
	}
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Errorf("leftover temp file %q next to aliases.json", e.Name())
		}
	}

	// The state manager must agree with the catalog after reconciliation.
	srv.syncCatalogToState()
	snap := srv.sm.Snapshot().Models
	for _, a := range aliases {
		if _, ok := snap[a]; !ok {
			t.Errorf("alias %q missing from state manager after reconciliation", a)
		}
	}
}

// TestServer_CorruptPersistenceFilesAreLoudAtStartup (T022 AC3): a damaged
// data_dir must not come up as a silent empty store. Each corrupt control-plane
// file is named in the startup log, no lane/alias from it is loaded, and the
// server still has a usable timeout manager (a nil one would panic on the first
// request, since the request path reads it unconditionally-ish).
func TestServer_CorruptPersistenceFilesAreLoudAtStartup(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"aliases.json", "providers.json", "timeouts.json"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(`{"truncated": `), 0o600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	cfg := DefaultConfig()
	cfg.DataDir = dir

	var buf bytes.Buffer
	log.SetOutput(&buf)
	srv := NewServer(cfg, provider.NewRegistry(), WithStateManager(state.NewStateManager()))
	log.SetOutput(os.Stderr)

	if srv == nil {
		t.Fatal("NewServer returned nil")
	}
	out := buf.String()
	for _, want := range []string{"aliases.json", "providers.json", "timeouts.json"} {
		if !strings.Contains(out, want) {
			t.Errorf("startup never named the corrupt %s file; log:\n%s", want, out)
		}
	}
	if got := strings.Count(out, "corrupt"); got < 3 {
		t.Errorf("corruption reported %d times; want 3 (aliases, providers, timeouts). log:\n%s", got, out)
	}
	if len(srv.catalog.Sorted()) != 0 {
		t.Errorf("corrupt aliases.json contributed %d aliases; want none", len(srv.catalog.Sorted()))
	}
	if len(srv.providers.List()) != 0 {
		t.Errorf("corrupt providers.json contributed %d lanes; want none", len(srv.providers.List()))
	}
	if srv.timeouts == nil {
		t.Fatal("nil timeout manager: the request path would panic")
	}
	if got := srv.timeouts.Timeout("any-lane"); got != DefaultRequestTimeout {
		t.Errorf("timeout after corrupt file = %v; want default %v", got, DefaultRequestTimeout)
	}
}
