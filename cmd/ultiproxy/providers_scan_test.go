package main

import (
	"bytes"
	"context"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/antigravity"
	"github.com/smhanov/ultiproxy/pkg/provider/codex"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/server"
)

// clearStartupEnv neutralizes every env-token source registerProviders reads,
// so scan tests observe only fixture credentials.
func clearStartupEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"ULTIPROXY_DATA_DIR", "ULTIPROXY_STATE_DIR",
		"ZAI_API_KEY", "ULTIPROXY_ZAI_API_KEY",
		"DEEPSEEK_API_KEY", "ULTIPROXY_DEEPSEEK_API_KEY",
		"ULTIPROXY_ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY",
		"OPENROUTER_API_KEY", "ULTIPROXY_OPENROUTER_API_KEY",
		"ULTIPROXY_VLLM_BASE_URL", "VLLM_BASE_URL",
		"OPENCODE_API_KEY", "ULTIPROXY_OPENCODE_API_KEY",
		"ULTIPROXY_AUGURE_TOKEN", "AUGURE_TOKEN",
		"ULTIPROXY_XAI_TOKEN",
		"ULTIPROXY_CODEX_TOKEN", "ULTIPROXY_COPILOT_TOKEN",
		"COPILOT_GITHUB_TOKEN", "GH_TOKEN",
		"ULTIPROXY_FREEBUFF_TOKEN", "FREEBUFF_TOKEN",
	} {
		t.Setenv(k, "")
	}
}

// captureStartupLog redirects the standard logger for the duration of a test.
func captureStartupLog(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	return &buf
}

// seedScanCred stores one fixture credential under credDir for clientID.
func seedScanCred(t *testing.T, credDir, clientID, token string) {
	t.Helper()
	mgr, err := auth.NewManager(credDir, nil)
	if err != nil {
		t.Fatalf("auth.NewManager(%s): %v", credDir, err)
	}
	if err := mgr.Store(context.Background(), clientID, auth.Credential{
		Provider:    "scan-test",
		AccessToken: token,
		ExpiresAt:   time.Now().Add(time.Hour),
		ClientID:    clientID,
	}); err != nil {
		t.Fatalf("seed credential: %v", err)
	}
}

// scanToken serves the lane's live token (empty when the bundle has no Auth).
func scanToken(t *testing.T, bundle provider.Provider) string {
	t.Helper()
	if bundle.Auth == nil {
		return ""
	}
	tok, err := bundle.Auth.Token(context.Background())
	if err != nil {
		t.Fatalf("lane Token(): %v", err)
	}
	return tok
}

// TestLaneScanDescriptorsTable (T005 AC3): the vendor blocks are declarative
// descriptors + a shared loop. A fixture credential per vendor yields the same
// lane the old block built: same name, credential-scan source, and a working
// token served from daemon-owned state.
func TestLaneScanDescriptorsTable(t *testing.T) {
	clearStartupEnv(t)

	if len(laneScanOrder) != 3 {
		t.Fatalf("laneScanOrder = %v, want [xai codex antigravity]", laneScanOrder)
	}
	for _, name := range laneScanOrder {
		d, ok := laneScanDescriptors[name]
		if !ok {
			t.Errorf("laneScanDescriptors[%q] missing", name)
			continue
		}
		if d.name != name || d.credSubdir != name {
			t.Errorf("descriptor %q: name=%q credSubdir=%q", name, d.name, d.credSubdir)
		}
		if d.build == nil {
			t.Errorf("descriptor %q has no build func", name)
		}
	}

	t.Run("xai", func(t *testing.T) {
		clearStartupEnv(t)
		dir := t.TempDir()
		seedScanCred(t, filepath.Join(dir, "credentials", "xai"), xaiDefaultClientID, "xai-scan-tok")
		xaiMgr, err := newOAuthManager(filepath.Join(dir, "credentials", "xai"))
		if err != nil {
			t.Fatalf("newOAuthManager: %v", err)
		}
		bundle, source, err := laneScanDescriptors["xai"].build(laneScanContext{stateDir: dir, home: dir, xaiMgr: xaiMgr})
		if err != nil {
			t.Fatalf("xai build: %v", err)
		}
		if source != "credential-scan" {
			t.Errorf("xai source = %q, want credential-scan", source)
		}
		if bundle.Inference == nil || bundle.Inference.Name() != "xai" {
			t.Fatalf("xai lane is not an inference provider: %+v", bundle)
		}
		if bundle.Quota == nil {
			t.Error("xai credential lane has no Quota surface (credits observer dropped?)")
		}
		if got := scanToken(t, bundle); got != "xai-scan-tok" {
			t.Errorf("xai Token() = %q, want the fixture credential", got)
		}
	})

	t.Run("codex", func(t *testing.T) {
		clearStartupEnv(t)
		dir := t.TempDir()
		seedScanCred(t, filepath.Join(dir, "credentials", "codex"), codex.DefaultClientID, "codex-scan-tok")
		bundle, source, err := laneScanDescriptors["codex"].build(laneScanContext{stateDir: dir, home: dir})
		if err != nil {
			t.Fatalf("codex build: %v", err)
		}
		if source != "credential-scan" {
			t.Errorf("codex source = %q, want credential-scan", source)
		}
		if bundle.Inference == nil || bundle.Inference.Name() != "codex" {
			t.Fatalf("codex lane is not an inference provider: %+v", bundle)
		}
		if got := scanToken(t, bundle); got != "codex-scan-tok" {
			t.Errorf("codex Token() = %q, want the fixture credential", got)
		}
	})

	t.Run("antigravity", func(t *testing.T) {
		clearStartupEnv(t)
		dir := t.TempDir()
		seedScanCred(t, filepath.Join(dir, "credentials", "antigravity"), antigravity.DefaultClientID, "ag-scan-tok")
		bundle, source, err := laneScanDescriptors["antigravity"].build(laneScanContext{stateDir: dir, home: dir})
		if err != nil {
			t.Fatalf("antigravity build: %v", err)
		}
		if source != "credential-scan" {
			t.Errorf("antigravity source = %q, want credential-scan", source)
		}
		if bundle.Auth == nil {
			t.Fatal("antigravity lane has no Auth surface")
		}
		if got := scanToken(t, bundle); got != "ag-scan-tok" {
			t.Errorf("antigravity Token() = %q, want the fixture credential", got)
		}
	})
}

// TestLaneScanDescriptorsEnvFallbacks covers the descriptor alternates: with
// no credential on disk an env token still builds the same-named lane via the
// env-token source; antigravity has no alternate and stays absent.
func TestLaneScanDescriptorsEnvFallbacks(t *testing.T) {
	clearStartupEnv(t)

	t.Run("xai", func(t *testing.T) {
		clearStartupEnv(t)
		t.Setenv("ULTIPROXY_XAI_TOKEN", "xai-env-tok")
		dir := t.TempDir()
		xaiMgr, err := newOAuthManager(filepath.Join(dir, "credentials", "xai"))
		if err != nil {
			t.Fatalf("newOAuthManager: %v", err)
		}
		bundle, source, err := laneScanDescriptors["xai"].build(laneScanContext{stateDir: dir, home: dir, xaiMgr: xaiMgr})
		if err != nil {
			t.Fatalf("xai env build: %v", err)
		}
		if source != "env-token" {
			t.Errorf("xai source = %q, want env-token", source)
		}
		if bundle.Inference == nil || bundle.Inference.Name() != "xai" {
			t.Fatalf("xai env lane is not an inference provider: %+v", bundle)
		}
		if bundle.Auth != nil {
			t.Error("xai env-token lane should not carry the OAuth Auth surface")
		}
	})

	t.Run("codex", func(t *testing.T) {
		clearStartupEnv(t)
		t.Setenv("ULTIPROXY_CODEX_TOKEN", "codex-env-tok")
		bundle, source, err := laneScanDescriptors["codex"].build(laneScanContext{stateDir: t.TempDir()})
		if err != nil {
			t.Fatalf("codex env build: %v", err)
		}
		if source != "env-token" {
			t.Errorf("codex source = %q, want env-token", source)
		}
		if bundle.Inference == nil || bundle.Inference.Name() != "codex" {
			t.Fatalf("codex env lane is not an inference provider: %+v", bundle)
		}
	})

	t.Run("antigravity", func(t *testing.T) {
		clearStartupEnv(t)
		dir := t.TempDir()
		if _, _, err := laneScanDescriptors["antigravity"].build(laneScanContext{stateDir: dir, home: dir}); err == nil {
			t.Fatal("antigravity build without a credential succeeded, want the silent-skip signal")
		}
	})
}

// TestScanCredentialLanesSilentWhenAbsent (T005 AC2): with no credential and
// no env token, the scan registers zero lanes and logs nothing about them
// (the "fresh install registers zero lanes" behavior).
func TestScanCredentialLanesSilentWhenAbsent(t *testing.T) {
	clearStartupEnv(t)
	dir := t.TempDir()
	buf := captureStartupLog(t)

	xaiMgr, err := newOAuthManager(filepath.Join(dir, "credentials", "xai"))
	if err != nil {
		t.Fatalf("newOAuthManager: %v", err)
	}
	registry := provider.NewRegistry()
	scanCredentialLanes(registry, laneScanContext{stateDir: dir, home: dir, xaiMgr: xaiMgr})

	if got := registry.Len(); got != 0 {
		t.Errorf("registry holds %d lanes %v, want zero", got, registry.Names())
	}
	if out := buf.String(); out != "" {
		t.Errorf("scan logged %q with no credential/env present, want silence", out)
	}
}

// TestScanCredentialLanesConflictWinner (T005 AC1): a lane that is already
// registered is replaced deterministically by the scan build and the
// replacement is logged — never two conflicting registrations.
func TestScanCredentialLanesConflictWinner(t *testing.T) {
	clearStartupEnv(t)
	dir := t.TempDir()
	seedScanCred(t, filepath.Join(dir, "credentials", "xai"), xaiDefaultClientID, "xai-scan-tok")
	xaiMgr, err := newOAuthManager(filepath.Join(dir, "credentials", "xai"))
	if err != nil {
		t.Fatalf("newOAuthManager: %v", err)
	}

	// Simulate a stale/runtime xai lane already in the registry (static key,
	// discovery opted out so construction never dials the network).
	registry := provider.NewRegistry()
	stale, err := openaicompat.New(openaicompat.Config{
		Name:                       "xai",
		BaseURL:                    xaiDefaultBaseURL,
		APIKey:                     "stale-runtime-key",
		OptOutModelListPassthrough: true,
		Quirks: openaicompat.Quirks{
			CreditsQuotaObserver: xaiDefaultBillingURL,
		},
	})
	if err != nil {
		t.Fatalf("stale xai lane: %v", err)
	}
	registry.Register(stale.Provider())

	buf := captureStartupLog(t)
	scanCredentialLanes(registry, laneScanContext{stateDir: dir, home: dir, xaiMgr: xaiMgr})

	if names := registry.Names(); len(names) != 1 || names[0] != "xai" {
		t.Fatalf("registry = %v, want exactly [xai] (one winner, no duplicates)", names)
	}
	bundle, _ := registry.Get("xai")
	if got := scanToken(t, bundle); got != "xai-scan-tok" {
		t.Errorf("xai Token() = %q, want the credential-scan build (deterministic winner)", got)
	}
	out := buf.String()
	if !strings.Contains(out, "replacing existing registration with credential-scan build") {
		t.Errorf("no conflict-replacement log line, got %q", out)
	}
	if !strings.Contains(out, "registered xai via credential-scan") {
		t.Errorf("no credential-scan source log line, got %q", out)
	}
}

// TestStartupRestoreConflictSingleLane (T005 AC1/AC4, production order):
// registerProviders runs BEFORE providerStore.Restore (runServe), so a
// providers.json runtime lane with the same name wins deterministically via
// Register's replace semantics. Both log lines appear in order — scan first,
// runtime second — and the last line is the live lane.
func TestStartupRestoreConflictSingleLane(t *testing.T) {
	clearStartupEnv(t)
	stateDir := t.TempDir()
	seedScanCred(t, filepath.Join(stateDir, "credentials", "xai"), xaiDefaultClientID, "xai-scan-tok")

	buf := captureStartupLog(t)
	registry := provider.NewRegistry()

	// Phase 1: compile-time scan (registerProviders) finds the credential.
	xaiMgr := registerProviders(registry, stateDir)
	if xaiMgr == nil {
		t.Fatal("registerProviders returned a nil xai store")
	}
	if names := registry.Names(); len(names) != 1 || names[0] != "xai" {
		t.Fatalf("after registerProviders registry = %v, want exactly [xai]", names)
	}

	// Phase 2: providers.json holds a same-named runtime lane (operator-added
	// via MCP). Discovery is opted out so Restore never dials the network.
	store := server.NewRuntimeProviderStore(filepath.Join(stateDir, "providers.json"))
	store.DefaultDataDir = stateDir
	store.Creds = xaiMgr
	store.LaneBuilder = runtimeLaneBuilder
	if err := store.Add(openaicompat.Config{
		Name:                       "xai",
		BaseURL:                    xaiDefaultBaseURL,
		OptOutModelListPassthrough: true,
		Quirks: openaicompat.Quirks{
			AuthViaOAuthManager:  true,
			CreditsQuotaObserver: xaiDefaultBillingURL,
		},
	}); err != nil {
		t.Fatalf("store.Add runtime xai: %v", err)
	}
	restored := store.Restore(registry)
	store.WaitForRestoreDiscovery()
	if len(restored) != 1 || restored[0] != "xai" {
		t.Fatalf("Restore registered %v, want [xai]", restored)
	}

	// One lane, no duplicates; the runtime (providers.json) build is live.
	if names := registry.Names(); len(names) != 1 || names[0] != "xai" {
		t.Fatalf("after Restore registry = %v, want exactly [xai]", names)
	}
	bundle, _ := registry.Get("xai")
	disabler, ok := bundle.Inference.(interface{ ModelDiscoveryEnabled() bool })
	if !ok {
		t.Fatalf("xai lane %T has no ModelDiscoveryEnabled surface", bundle.Inference)
	}
	if disabler.ModelDiscoveryEnabled() {
		t.Error("live xai lane has discovery enabled: the scan build survived, want the runtime (opted-out) build")
	}

	// AC4: the boot log states each lane's source, in production order.
	out := buf.String()
	scanIdx := strings.Index(out, "registered xai via credential-scan")
	runtimeIdx := strings.Index(out, "registered runtime xai")
	if scanIdx < 0 {
		t.Errorf("no credential-scan source line in boot log:\n%s", out)
	}
	if runtimeIdx < 0 {
		t.Errorf("no providers.json (runtime) source line in boot log:\n%s", out)
	}
	if scanIdx >= 0 && runtimeIdx >= 0 && scanIdx > runtimeIdx {
		t.Errorf("boot log order inverted (runtime before scan):\n%s", out)
	}
}
