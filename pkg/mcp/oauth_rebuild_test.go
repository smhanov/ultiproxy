package mcp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// fakeInteractiveAuth completes login immediately for T004 handler tests.
type fakeInteractiveAuth struct {
	name        string
	completeErr error
}

func (f *fakeInteractiveAuth) Name() string { return f.name }
func (f *fakeInteractiveAuth) Login(ctx context.Context) error {
	return f.completeErr
}
func (f *fakeInteractiveAuth) Token(ctx context.Context) (string, error) { return "fake-tok", nil }
func (f *fakeInteractiveAuth) Refresh(ctx context.Context) error         { return nil }
func (f *fakeInteractiveAuth) StartLogin(ctx context.Context) (*provider.LoginStartInfo, error) {
	return &provider.LoginStartInfo{
		Kind:            provider.LoginFlowDevice,
		VerificationURI: "https://example.test/device",
		UserCode:        "TEST-1234",
		ExpiresIn:       300,
	}, nil
}
func (f *fakeInteractiveAuth) CompleteLogin(ctx context.Context, code string) error {
	return f.completeErr
}

// wrapWithFakeAuth replaces the registry lane's Auth with a fake interactive
// flow that completes immediately, keeping the same Inference surface so the
// stored config stays rebuildable.
func wrapWithFakeAuth(t *testing.T, registry *provider.Registry, name string, auth provider.AuthProvider) {
	t.Helper()
	bundle, ok := registry.Get(name)
	if !ok {
		t.Fatalf("%s not in registry", name)
	}
	registry.Register(provider.Provider{
		Inference:    bundle.Inference,
		Quota:        bundle.Quota,
		Auth:         auth,
		Capabilities: bundle.Capabilities,
	})
}

func decodeToolText(t *testing.T, res CallToolResult) map[string]any {
	t.Helper()
	if len(res.Content) == 0 {
		t.Fatalf("tool result has no content: %+v", res)
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(res.Content[0].Text), &out); err != nil {
		t.Fatalf("decode tool reply %q: %v", res.Content[0].Text, err)
	}
	return out
}

// TestCheckOAuthLoginRebuildsLaneAndDiscovers (AC1/AC2): a completed device
// flow rebuilds the lane and the reply carries the fresh discovery count.
func TestCheckOAuthLoginRebuildsLaneAndDiscovers(t *testing.T) {
	upstream, _ := newModelsUpstream(t, "grok-4", "grok-4-fast")
	dir := t.TempDir()
	store := newFileProviderStore(dir + "/providers.json")
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store))

	add := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"xai","base_url":"`+upstream.URL+`"}`)
	if add.IsError {
		t.Fatalf("add_provider failed: %s", add.Content[0].Text)
	}
	wrapWithFakeAuth(t, registry, "xai", &fakeInteractiveAuth{name: "xai"})

	res := callMCPTool(t, srv, 2, "check_oauth_login", `{"provider":"xai"}`)
	if res.IsError {
		t.Fatalf("check_oauth_login failed: %s", res.Content[0].Text)
	}
	reply := decodeToolText(t, res)
	if reply["status"] != "completed" {
		t.Fatalf("status = %v, want completed: %v", reply["status"], reply)
	}
	if got, _ := reply["discovered_models"].(float64); got != 2 {
		t.Fatalf("discovered_models = %v, want 2: %v", got, reply)
	}
	if note, _ := reply["note"].(string); note != "" {
		t.Fatalf("note = %q, want empty on clean rebuild: %v", note, reply)
	}
	// Registry serves the rebuilt lane with a fresh cache, no restart.
	bundle, ok := registry.Get("xai")
	if !ok {
		t.Fatal("xai missing from registry after rebuild")
	}
	cacher, ok := bundle.Inference.(interface{ CachedModels() []string })
	if !ok {
		t.Fatalf("rebuilt lane has no model cache: %T", bundle.Inference)
	}
	if got := cacher.CachedModels(); len(got) != 2 {
		t.Fatalf("rebuilt cached = %v, want 2 models", got)
	}
}

// TestCheckOAuthLoginNotRegistered (AC3): login for a lane with no stored
// record still succeeds and reports honestly instead of a false "usable".
func TestCheckOAuthLoginNotRegistered(t *testing.T) {
	registry := provider.NewRegistry()
	registry.Register(provider.Provider{
		Inference: modelsTestLane{name: "ghost"},
		Auth:      &fakeInteractiveAuth{name: "ghost"},
	})
	dir := t.TempDir()
	store := newFileProviderStore(dir + "/providers.json")
	srv := NewServer(registry, nil, WithProviderStore(store))

	res := callMCPTool(t, srv, 1, "check_oauth_login", `{"provider":"ghost"}`)
	if res.IsError {
		t.Fatalf("check_oauth_login for unstored lane must not error: %s", res.Content[0].Text)
	}
	reply := decodeToolText(t, res)
	if reply["status"] != "completed" {
		t.Fatalf("status = %v, want completed: %v", reply["status"], reply)
	}
	note, _ := reply["note"].(string)
	if note != "credential stored; lane not registered — call add_provider" {
		t.Fatalf("note = %q, want AC3 wording: %v", note, reply)
	}
	if got, _ := reply["discovered_models"].(float64); got != 0 {
		t.Fatalf("discovered_models = %v, want 0 for unstored lane: %v", got, reply)
	}
}

// TestCheckOAuthLoginRebuildFailurePreservesOld (AC4): when the fresh build
// cannot discover, the reply is honest and the previous lane stays live.
func TestCheckOAuthLoginRebuildFailurePreservesOld(t *testing.T) {
	upstream, _ := newModelsUpstream(t, "keep-me")
	dir := t.TempDir()
	store := newFileProviderStore(dir + "/providers.json")
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store))

	add := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"rot","base_url":"`+upstream.URL+`"}`)
	if add.IsError {
		t.Fatalf("add_provider failed: %s", add.Content[0].Text)
	}
	wrapWithFakeAuth(t, registry, "rot", &fakeInteractiveAuth{name: "rot"})

	// Break the STORED config only (live registry still serves keep-me).
	// An unroutable base URL makes the fresh build's discovery fail while the
	// live lane stays good.
	if err := store.Add(openaicompat.Config{
		Name:    "rot",
		BaseURL: "http://127.0.0.1:1/v1",
		APIKey:  "sk-dead",
	}); err != nil {
		t.Fatalf("store.Add dead config: %v", err)
	}

	res := callMCPTool(t, srv, 2, "check_oauth_login", `{"provider":"rot"}`)
	if res.IsError {
		t.Fatalf("rebuild failure must not be a tool error (credential IS stored): %s", res.Content[0].Text)
	}
	reply := decodeToolText(t, res)
	if reply["status"] != "completed" {
		t.Fatalf("status = %v, want completed with honest note: %v", reply["status"], reply)
	}
	note, _ := reply["note"].(string)
	if !strings.Contains(note, "previous lane preserved") {
		t.Fatalf("note = %q, want it to say the previous lane is preserved: %v", note, reply)
	}
	// Old lane preserved: still serves keep-me.
	bundle, ok := registry.Get("rot")
	if !ok {
		t.Fatal("rot missing from registry after failed rebuild")
	}
	// The preserved lane is the pre-login wrapper (same Inference object with
	// the original cache), not an empty rebuilt lane.
	if cacher, ok := bundle.Inference.(interface{ CachedModels() []string }); ok {
		if got := cacher.CachedModels(); len(got) != 1 || got[0] != "keep-me" {
			t.Fatalf("preserved cached = %v, want [keep-me]", got)
		}
	}
}

// TestSubmitOAuthCodeRebuildsLane covers the auth-code completion branch: the
// same rebuild + discovery fields appear on submit_oauth_code.
func TestSubmitOAuthCodeRebuildsLane(t *testing.T) {
	upstream, _ := newModelsUpstream(t, "a1", "a2", "a3")
	dir := t.TempDir()
	store := newFileProviderStore(dir + "/providers.json")
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store))

	add := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"ag","base_url":"`+upstream.URL+`"}`)
	if add.IsError {
		t.Fatalf("add_provider failed: %s", add.Content[0].Text)
	}
	wrapWithFakeAuth(t, registry, "ag", &fakeInteractiveAuth{name: "ag"})

	res := callMCPTool(t, srv, 2, "submit_oauth_code", `{"provider":"ag","code":"past-code-123"}`)
	if res.IsError {
		t.Fatalf("submit_oauth_code failed: %s", res.Content[0].Text)
	}
	reply := decodeToolText(t, res)
	if reply["status"] != "completed" {
		t.Fatalf("status = %v, want completed: %v", reply["status"], reply)
	}
	if got, _ := reply["discovered_models"].(float64); got != 3 {
		t.Fatalf("discovered_models = %v, want 3: %v", got, reply)
	}
}

// TestInitiateOAuthLoginLegacyRebuilds covers the legacy blocking Login path:
// after Login the reply still states the rebuilt discovery count.
func TestInitiateOAuthLoginLegacyRebuilds(t *testing.T) {
	upstream, _ := newModelsUpstream(t, "lm1")
	dir := t.TempDir()
	store := newFileProviderStore(dir + "/providers.json")
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store))

	add := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"leg","base_url":"`+upstream.URL+`"}`)
	if add.IsError {
		t.Fatalf("add_provider failed: %s", add.Content[0].Text)
	}
	// Non-interactive auth forces the legacy Login branch.
	bundle, _ := registry.Get("leg")
	registry.Register(provider.Provider{
		Inference:    bundle.Inference,
		Auth:         &fakeAuthProvider{name: "leg"},
		Capabilities: bundle.Capabilities,
	})

	res := callMCPTool(t, srv, 2, "initiate_oauth_login", `{"provider":"leg"}`)
	if res.IsError {
		t.Fatalf("initiate_oauth_login (legacy) failed: %s", res.Content[0].Text)
	}
	reply := decodeToolText(t, res)
	if reply["status"] != "initiated" {
		t.Fatalf("status = %v, want initiated: %v", reply["status"], reply)
	}
	if got, _ := reply["discovered_models"].(float64); got != 1 {
		t.Fatalf("discovered_models = %v, want 1: %v", got, reply)
	}
}

// Ensure the ir import is used (stubCustomLane in server_test.go already does,
// but this file must compile standalone).
var _ = ir.Response{}
