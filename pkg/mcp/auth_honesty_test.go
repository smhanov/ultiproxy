package mcp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// emptyOAuthStore is a credential store holding nothing: Peek always misses so
// an OAuth lane reports login_required.
type emptyOAuthStore struct{}

func (emptyOAuthStore) Get(ctx context.Context, key string) (auth.Credential, error) {
	return auth.Credential{}, auth.ErrNotFound
}
func (emptyOAuthStore) Peek(key string) (auth.Credential, bool) { return auth.Credential{}, false }
func (emptyOAuthStore) Store(ctx context.Context, key string, cred auth.Credential) error {
	return nil
}
func (emptyOAuthStore) Invalidate(key, accessToken string) bool { return false }

// expiringOAuthStore holds one credential with a real expiry for AC2.
type expiringOAuthStore struct {
	cred auth.Credential
}

func (s expiringOAuthStore) Get(ctx context.Context, key string) (auth.Credential, error) {
	return s.cred, nil
}
func (s expiringOAuthStore) Peek(key string) (auth.Credential, bool) { return s.cred, true }
func (s expiringOAuthStore) Store(ctx context.Context, key string, cred auth.Credential) error {
	return nil
}
func (s expiringOAuthStore) Invalidate(key, accessToken string) bool { return false }

// TestAddProviderOAuthLoginRequired (T006 AC1, no-credential branch): an OAuth
// lane added with no stored credential skips the doomed 401 and replies with
// login_required + next_action naming initiate_oauth_login.
func TestAddProviderOAuthLoginRequired(t *testing.T) {
	upstream, _ := newModelsUpstream(t, "grok-4")

	dir := t.TempDir()
	store := newFileProviderStore(dir + "/providers.json")
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store), WithCredentialStore(emptyOAuthStore{}))

	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"xai","base_url":"`+upstream.URL+`","quirks":{"auth_via_oauth_manager":true}}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}
	reply := decodeToolText(t, res)
	if reply["login_required"] != true {
		t.Fatalf("login_required = %v, want true: %v", reply["login_required"], reply)
	}
	if reply["next_action"] != "initiate_oauth_login" {
		t.Fatalf("next_action = %v, want initiate_oauth_login: %v", reply["next_action"], reply)
	}
	if got, _ := reply["discovered_models"].(float64); got != 0 {
		t.Fatalf("discovered_models = %v, want 0 when login is required: %v", got, reply)
	}
	note, _ := reply["note"].(string)
	if !strings.Contains(note, "initiate_oauth_login") {
		t.Fatalf("note = %q, want it to name initiate_oauth_login: %v", note, reply)
	}
	summary, _ := reply["summary"].(string)
	if !strings.Contains(summary, "login required") {
		t.Fatalf("summary = %q, want login-required wording, not a bare 401: %v", summary, reply)
	}
	if strings.Contains(res.Content[0].Text, "401") {
		t.Fatalf("reply mentions a bare 401 instead of the actionable login_required: %s", res.Content[0].Text)
	}
	// The lane IS registered so initiate_oauth_login has an Auth surface.
	if _, ok := registry.Get("xai"); !ok {
		t.Fatal("xai not registered although the reply says registered:true")
	}
}

// TestAddProviderOAuthWithCredentialDiscovers (T006 AC1, credential branch):
// with a credential present the reply is the normal discovery summary.
func TestAddProviderOAuthWithCredentialDiscovers(t *testing.T) {
	upstream, _ := newModelsUpstream(t, "grok-4", "grok-4-fast")

	dir := t.TempDir()
	store := newFileProviderStore(dir + "/providers.json")
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store), WithCredentialStore(stubCredsStore{}))

	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"xai","base_url":"`+upstream.URL+`","quirks":{"auth_via_oauth_manager":true}}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}
	reply := decodeToolText(t, res)
	if _, hasLogin := reply["login_required"]; hasLogin {
		t.Fatalf("reply carries login_required although a credential is stored: %v", reply)
	}
	if got, _ := reply["discovered_models"].(float64); got != 2 {
		t.Fatalf("discovered_models = %v, want 2 (normal discovery): %v", got, reply)
	}
	if summary, _ := reply["summary"].(string); !strings.Contains(summary, "discovered 2 models") {
		t.Fatalf("summary = %q, want normal discovery summary: %v", summary, reply)
	}
}

// TestListProvidersLoggedIn (T006 AC2): OAuth rows carry logged_in +
// token_expires_at via Peek; no token material leaks.
func TestListProvidersLoggedIn(t *testing.T) {
	upstream, _ := newModelsUpstream(t, "grok-4")

	// Empty store: logged_in:false, no expiry.
	dir1 := t.TempDir()
	store1 := newFileProviderStore(dir1 + "/providers.json")
	srv1 := NewServer(provider.NewRegistry(), nil, WithProviderStore(store1), WithCredentialStore(emptyOAuthStore{}))
	add1 := callMCPTool(t, srv1, 1, "add_provider",
		`{"name":"xai","base_url":"`+upstream.URL+`","quirks":{"auth_via_oauth_manager":true}}`)
	if add1.IsError {
		t.Fatalf("add_provider (empty store) failed: %s", add1.Content[0].Text)
	}
	list1 := callMCPTool(t, srv1, 2, "list_providers", `{}`)
	if list1.IsError {
		t.Fatalf("list_providers failed: %s", list1.Content[0].Text)
	}
	var out1 struct {
		Providers []map[string]any `json:"providers"`
	}
	if err := json.Unmarshal([]byte(list1.Content[0].Text), &out1); err != nil {
		t.Fatalf("decode list_providers: %v (%s)", err, list1.Content[0].Text)
	}
	if len(out1.Providers) != 1 {
		t.Fatalf("providers = %v, want 1 row", out1.Providers)
	}
	row1 := out1.Providers[0]
	if row1["logged_in"] != false {
		t.Fatalf("logged_in = %v, want false for empty store: %v", row1["logged_in"], row1)
	}
	if _, hasExp := row1["token_expires_at"]; hasExp {
		t.Fatalf("token_expires_at present with no credential: %v", row1)
	}

	// Populated store with expiry: logged_in:true + RFC3339 expiry.
	expiry := time.Now().Add(time.Hour).UTC().Truncate(time.Second)
	populated := expiringOAuthStore{cred: auth.Credential{
		Provider:    "xai",
		AccessToken: "super-secret-access",
		ExpiresAt:   expiry,
		ClientID:    xaiOAuthClientID,
	}}
	dir2 := t.TempDir()
	store2 := newFileProviderStore(dir2 + "/providers.json")
	srv2 := NewServer(provider.NewRegistry(), nil, WithProviderStore(store2), WithCredentialStore(populated))
	add2 := callMCPTool(t, srv2, 1, "add_provider",
		`{"name":"xai","base_url":"`+upstream.URL+`","quirks":{"auth_via_oauth_manager":true}}`)
	if add2.IsError {
		t.Fatalf("add_provider (populated store) failed: %s", add2.Content[0].Text)
	}
	list2 := callMCPTool(t, srv2, 2, "list_providers", `{}`)
	var out2 struct {
		Providers []map[string]any `json:"providers"`
	}
	if err := json.Unmarshal([]byte(list2.Content[0].Text), &out2); err != nil {
		t.Fatalf("decode list_providers: %v (%s)", err, list2.Content[0].Text)
	}
	row2 := out2.Providers[0]
	if row2["logged_in"] != true {
		t.Fatalf("logged_in = %v, want true for populated store: %v", row2["logged_in"], row2)
	}
	expRaw, _ := row2["token_expires_at"].(string)
	if expRaw == "" {
		t.Fatalf("token_expires_at missing for populated store: %v", row2)
	}
	if _, err := time.Parse(time.RFC3339, expRaw); err != nil {
		t.Fatalf("token_expires_at = %q, not RFC3339: %v", expRaw, err)
	}
	// No token material anywhere in the listing.
	if strings.Contains(list2.Content[0].Text, "super-secret-access") {
		t.Fatalf("list_providers leaked the access token: %s", list2.Content[0].Text)
	}

	// Missing store: fields omitted rather than lied about.
	dir3 := t.TempDir()
	store3 := newFileProviderStore(dir3 + "/providers.json")
	srv3 := NewServer(provider.NewRegistry(), nil, WithProviderStore(store3))
	// Add a non-OAuth lane (no creds needed) so listing has a row.
	add3 := callMCPTool(t, srv3, 1, "add_provider",
		`{"name":"plain","base_url":"`+upstream.URL+`"}`)
	if add3.IsError {
		t.Fatalf("add_provider (plain) failed: %s", add3.Content[0].Text)
	}
	list3 := callMCPTool(t, srv3, 3, "list_providers", `{}`)
	var out3 struct {
		Providers []map[string]any `json:"providers"`
	}
	if err := json.Unmarshal([]byte(list3.Content[0].Text), &out3); err != nil {
		t.Fatalf("decode list_providers: %v", err)
	}
	for _, row := range out3.Providers {
		if _, has := row["logged_in"]; has {
			t.Fatalf("non-OAuth lane carries logged_in (should omit): %v", row)
		}
		if _, has := row["token_expires_at"]; has {
			t.Fatalf("non-OAuth lane carries token_expires_at (should omit): %v", row)
		}
	}
}

// TestRefreshModelsSchemaProviderAlias (T006 AC3 schema half): tools/list
// documents provider (required) plus name as a deprecated alias.
func TestRefreshModelsSchemaProviderAlias(t *testing.T) {
	// Read the live schema through standardTools (same surface tools/list serves).
	var found *InputSchema
	for i := range standardTools {
		if standardTools[i].Name == "refresh_models" {
			found = standardTools[i].InputSchema
			break
		}
	}
	if found == nil {
		t.Fatal("refresh_models tool not found")
	}
	if _, ok := found.Properties["provider"]; !ok {
		t.Fatalf("refresh_models schema missing provider: %+v", found.Properties)
	}
	if _, ok := found.Properties["name"]; !ok {
		t.Fatalf("refresh_models schema missing deprecated name alias: %+v", found.Properties)
	}
	hasProviderRequired := false
	for _, r := range found.Required {
		if r == "provider" {
			hasProviderRequired = true
		}
	}
	if !hasProviderRequired {
		t.Fatalf("refresh_models schema required = %v, want provider required going forward", found.Required)
	}
}

// TestRefreshModelsProviderAlias (T006 AC3): provider and name behave
// identically.
func TestRefreshModelsProviderAlias(t *testing.T) {
	registry := provider.NewRegistry()
	lane := &fakeModelFetchLane{name: "opencode", models: []string{"a", "b"}}
	registry.Register(provider.Provider{Inference: lane})
	srv := NewServer(registry, newStubStateSource())

	viaProvider := callMCPTool(t, srv, 1, "refresh_models", `{"provider":"opencode"}`)
	if viaProvider.IsError {
		t.Fatalf("refresh_models with provider failed: %s", viaProvider.Content[0].Text)
	}
	viaName := callMCPTool(t, srv, 2, "refresh_models", `{"name":"opencode"}`)
	if viaName.IsError {
		t.Fatalf("refresh_models with name failed: %s", viaName.Content[0].Text)
	}
	if viaProvider.Content[0].Text != viaName.Content[0].Text {
		t.Fatalf("provider vs name replies differ:\nprovider: %q\nname: %q",
			viaProvider.Content[0].Text, viaName.Content[0].Text)
	}
	if !strings.Contains(viaProvider.Content[0].Text, "2 models cached for lane opencode") {
		t.Fatalf("reply = %q, want cached-model summary", viaProvider.Content[0].Text)
	}
}
