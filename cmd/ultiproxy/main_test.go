package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/server"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// TestIsAddrInUse classifies bind-conflict errors for the startup hint (T005):
// raw EADDRINUSE, wrapped EADDRINUSE, wrapped other error, nil.
func TestIsAddrInUse(t *testing.T) {
	if !isAddrInUse(syscall.EADDRINUSE) {
		t.Error("isAddrInUse(raw EADDRINUSE) = false, want true")
	}
	if !isAddrInUse(fmt.Errorf("listen: %w", syscall.EADDRINUSE)) {
		t.Error("isAddrInUse(wrapped EADDRINUSE) = false, want true")
	}
	// Portable fallback: a non-errno error carrying the bind-conflict text
	// (e.g. from a platform without syscall.EADDRINUSE) still counts.
	if !isAddrInUse(fmt.Errorf("listen tcp 127.0.0.1:9050: bind: address already in use")) {
		t.Error("isAddrInUse(substring fallback) = false, want true")
	}
	if isAddrInUse(fmt.Errorf("listen: %w", syscall.EACCES)) {
		t.Error("isAddrInUse(wrapped EACCES) = true, want false")
	}
	if isAddrInUse(fmt.Errorf("tls: certificate invalid")) {
		t.Error("isAddrInUse(non-bind error) = true, want false")
	}
	if isAddrInUse(nil) {
		t.Error("isAddrInUse(nil) = true, want false")
	}
}

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

// TestEnsureServeDirsNestedParent (T007): a hand-written config pointing at
// fresh nested paths must start — the helper creates the SQLite parent and
// the data dir so storage.NewWriter then succeeds.
func TestEnsureServeDirsNestedParent(t *testing.T) {
	base := t.TempDir()
	dbPath := filepath.Join(base, "fresh", "nested", "ultiproxy.db")
	dataDir := filepath.Join(base, "fresh", "state")

	if err := ensureServeDirs(dbPath, dataDir); err != nil {
		t.Fatalf("ensureServeDirs: %v", err)
	}
	if _, err := os.Stat(filepath.Join(base, "fresh", "nested")); err != nil {
		t.Fatalf("DB parent not created: %v", err)
	}
	if _, err := os.Stat(dataDir); err != nil {
		t.Fatalf("data dir not created: %v", err)
	}

	w, err := storage.NewWriter(dbPath)
	if err != nil {
		t.Fatalf("NewWriter after ensureServeDirs: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(dbPath); err != nil {
		t.Fatalf("DB file not created at %s: %v", dbPath, err)
	}
}

// TestEnsureServeDirsNewWriterAloneFails documents why the helper exists:
// storage.NewWriter opens the path directly (SQLite creates the file, never
// missing parents), so without the serve-layer MkdirAll a fresh nested path
// fails. Storage semantics are intentionally unchanged by T007.
func TestEnsureServeDirsNewWriterAloneFails(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "no-such-parent", "ultiproxy.db")
	w, err := storage.NewWriter(dbPath)
	if err == nil {
		w.Close()
		t.Fatalf("NewWriter(%s) succeeded without the parent dir, want failure", dbPath)
	}
}

// TestEnsureServeDirsNoops covers the default/relative case: a bare filename
// (parent ".") and "."/empty data dir perform no filesystem writes and never
// fail.
func TestEnsureServeDirsNoops(t *testing.T) {
	for _, dataDir := range []string{".", ""} {
		if err := ensureServeDirs("just-a-file.db", dataDir); err != nil {
			t.Errorf("ensureServeDirs(just-a-file.db, %q) = %v, want nil", dataDir, err)
		}
	}
}

// TestEnsureServeDirsFileInTheWay covers the fail-loud path (AC2): a regular
// file where a directory should be makes MkdirAll fail with ENOTDIR —
// effective even when running as root, unlike permission-bit tricks — and the
// error names the offending path so the runServe log.Fatalf line is actionable.
func TestEnsureServeDirsFileInTheWay(t *testing.T) {
	base := t.TempDir()
	blocker := filepath.Join(base, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatalf("write blocker file: %v", err)
	}
	err := ensureServeDirs(filepath.Join(blocker, "ultiproxy.db"), filepath.Join(base, "state"))
	if err == nil {
		t.Fatal("ensureServeDirs with a file in place of the parent = nil, want error")
	}
	if !strings.Contains(err.Error(), blocker) {
		t.Errorf("error %q does not name the offending path %q", err, blocker)
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

// TestResolveProviderStateDirPrecedence (T008): explicit env wins, then the
// configured data_dir, then the home default. "" and "." (the zero-config
// default) both mean "no custom dir" and fall through to the home default so
// default runs stay byte-identical.
func TestResolveProviderStateDirPrecedence(t *testing.T) {
	fakeHome := t.TempDir()
	t.Setenv("HOME", fakeHome)
	def := filepath.Join(fakeHome, ".local", "state", "ultiproxy")

	cases := []struct {
		name       string
		dataEnv    string
		stateEnv   string
		configured string
		want       string
	}{
		{"dataDirEnvWins", "/env-data", "/env-state", "/cfg", "/env-data"},
		{"stateDirEnvSecond", "", "/env-state", "/cfg", "/env-state"},
		{"configuredWins", "", "", "/cfg", "/cfg"},
		{"emptyFallsThrough", "", "", "", def},
		{"dotFallsThrough", "", "", ".", def},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("ULTIPROXY_DATA_DIR", tc.dataEnv)
			t.Setenv("ULTIPROXY_STATE_DIR", tc.stateEnv)
			if got := resolveProviderStateDir(tc.configured); got != tc.want {
				t.Errorf("resolveProviderStateDir(%q) = %q, want %q", tc.configured, got, tc.want)
			}
		})
	}
}

// TestRegisterProvidersFollowsConfiguredDataDir (T008 AC1/AC3): with no env
// override, the compile-time lanes resolve credential state under the passed
// configured dir (augure_token probe); with "" configured they fall back to
// the home default.
func TestRegisterProvidersFollowsConfiguredDataDir(t *testing.T) {
	fakeHome := t.TempDir()
	t.Setenv("HOME", fakeHome)
	t.Setenv("ULTIPROXY_DATA_DIR", "")
	t.Setenv("ULTIPROXY_STATE_DIR", "")
	// Neutralize every env-token source except the augure probe so the test
	// only observes the configured dir vs the home default.
	for _, k := range []string{
		"ZAI_API_KEY", "ULTIPROXY_ZAI_API_KEY",
		"DEEPSEEK_API_KEY", "ULTIPROXY_DEEPSEEK_API_KEY",
		"ULTIPROXY_ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY",
		"OPENROUTER_API_KEY", "ULTIPROXY_OPENROUTER_API_KEY",
		"ULTIPROXY_VLLM_BASE_URL", "VLLM_BASE_URL",
		"OPENCODE_API_KEY", "ULTIPROXY_OPENCODE_API_KEY",
		"AUGURE_TOKEN", "ULTIPROXY_XAI_TOKEN",
		"ULTIPROXY_CODEX_TOKEN", "ULTIPROXY_COPILOT_TOKEN",
		"COPILOT_GITHUB_TOKEN", "GH_TOKEN",
		"ULTIPROXY_FREEBUFF_TOKEN", "FREEBUFF_TOKEN",
	} {
		t.Setenv(k, "")
	}
	t.Setenv("ULTIPROXY_AUGURE_TOKEN", "aug-probe-token")

	configured := t.TempDir()
	registerProviders(provider.NewRegistry(), configured)

	raw, err := os.ReadFile(filepath.Join(configured, "augure_token"))
	if err != nil {
		t.Fatalf("augure_token not written under configured dir: %v", err)
	}
	if got := strings.TrimSpace(string(raw)); got != "aug-probe-token" {
		t.Errorf("augure_token = %q, want %q", got, "aug-probe-token")
	}
	if _, err := os.Stat(filepath.Join(fakeHome, ".local")); !os.IsNotExist(err) {
		t.Errorf("fake HOME touched (err = %v), want no state under the home default", err)
	}

	// Default path: "" configured resolves under the home default.
	registerProviders(provider.NewRegistry(), "")
	raw, err = os.ReadFile(filepath.Join(fakeHome, ".local", "state", "ultiproxy", "augure_token"))
	if err != nil {
		t.Fatalf("augure_token not written under home default: %v", err)
	}
	if got := strings.TrimSpace(string(raw)); got != "aug-probe-token" {
		t.Errorf("default augure_token = %q, want %q", got, "aug-probe-token")
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
	// The openaicompat freebuff quirk duck-types the session-lifecycle surface
	// before every chat (reconcile + bind); if the adapter drops any of these
	// methods the assertion silently fails and every chat 428s with
	// waiting_room_required.
	if _, ok := actor.(interface {
		Reconcile(...context.Context) error
		BoundModel() string
		DeleteSession(...context.Context) error
		Bind(any, ...string) error
	}); !ok {
		t.Fatalf("actor %T does not implement the freebuff session-lifecycle interface", actor)
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
