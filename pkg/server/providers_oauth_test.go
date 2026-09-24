package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// fakeXAIDeviceFlow returns an httptest server speaking the xAI device flow:
// /device issues a device_code, /token approves immediately with the given
// access token.
func fakeXAIDeviceFlow(t *testing.T, accessToken string) (*httptest.Server, string, *http.Client) {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/device", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"device_code": "dev-ac1-123",
			"user_code": "AC1-1234",
			"verification_uri": "https://accounts.x.ai/oauth2/device",
			"expires_in": 300,
			"interval": 0
		}`))
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"access_token": "` + accessToken + `",
			"refresh_token": "ac1-refresh-tok",
			"expires_in": 3600
		}`))
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts, ts.URL, ts.Client()
}

// AC1: an OAuth lane built from the daemon-owned store completes device login
// and the credential file appears under <dataDir>/credentials — never under
// os.TempDir().
func TestOAuthCredentialUnderDataDir(t *testing.T) {
	fallback := filepath.Join(os.TempDir(), "ultiproxy-xai-auth")
	_ = os.RemoveAll(fallback)

	dataDir := t.TempDir()
	credDir := filepath.Join(dataDir, "credentials", "xai")
	mgr, err := auth.NewManager(credDir, nil)
	if err != nil {
		t.Fatalf("auth.NewManager: %v", err)
	}

	store := NewRuntimeProviderStore(filepath.Join(dataDir, "providers.json"))
	store.DefaultDataDir = dataDir
	store.Creds = mgr

	_, fakeURL, fakeClient := fakeXAIDeviceFlow(t, "ac1-access-tok")
	if err := store.Add(openaicompat.Config{
		Name:                       "xai",
		BaseURL:                    "https://api.x.ai",
		HTTPClient:                 fakeClient,
		DeviceAuthURL:              fakeURL + "/device",
		TokenURL:                   fakeURL + "/token",
		OptOutModelListPassthrough: true,
		Quirks:                     openaicompat.Quirks{AuthViaOAuthManager: true},
	}); err != nil {
		t.Fatalf("Add xai lane: %v", err)
	}

	stored, ok := store.List()["xai"]
	if !ok {
		t.Fatal("xai lane not in store after Add")
	}
	if stored.Creds == nil {
		t.Fatal("injectCreds did not supply Config.Creds for the OAuth lane")
	}

	p, err := openaicompat.New(stored)
	if err != nil {
		t.Fatalf("openaicompat.New: %v", err)
	}
	if err := p.Login(context.Background()); err != nil {
		t.Fatalf("Login: %v", err)
	}

	// The token is served from the injected store.
	tok, err := p.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if tok != "ac1-access-tok" {
		t.Fatalf("Token = %q, want ac1-access-tok", tok)
	}

	// The file lives under <dataDir>/credentials (daemon-owned), never /tmp.
	matches, err := filepath.Glob(filepath.Join(dataDir, "credentials", "xai", "*.json"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("credential files under <dataDir>/credentials/xai = %v, want exactly 1", matches)
	}
	raw, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatalf("read credential file: %v", err)
	}
	if !strings.Contains(string(raw), "ac1-access-tok") {
		t.Fatalf("credential file does not hold the login token: %s", raw)
	}
	if _, err := os.Stat(fallback); !os.IsNotExist(err) {
		t.Fatalf("TempDir fallback %q exists (err = %v): credential escaped the daemon data dir", fallback, err)
	}

	// The store itself Peeks the token (T004/T006 handle reach it the same way).
	if cred, ok := mgr.Peek(xaiClientID); !ok || cred.AccessToken != "ac1-access-tok" {
		t.Fatalf("Peek = %+v, %v; want the login token", cred, ok)
	}
}

// AC3: restart restores the lane reading the same credential — a cold manager
// over the same <dataDir>/credentials dir Peeks the token, and a restored lane
// serves it without re-login.
func TestOAuthRestartRestoresCredential(t *testing.T) {
	dataDir := t.TempDir()
	credDir := filepath.Join(dataDir, "credentials", "xai")
	providersPath := filepath.Join(dataDir, "providers.json")

	mgr1, err := auth.NewManager(credDir, nil)
	if err != nil {
		t.Fatalf("auth.NewManager: %v", err)
	}
	store1 := NewRuntimeProviderStore(providersPath)
	store1.DefaultDataDir = dataDir
	store1.Creds = mgr1
	if err := store1.Add(openaicompat.Config{
		Name:                       "xai",
		BaseURL:                    "https://api.x.ai",
		OptOutModelListPassthrough: true,
		Quirks:                     openaicompat.Quirks{AuthViaOAuthManager: true},
	}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	seed := auth.Credential{
		Provider:    "xai",
		AccessToken: "restart-tok-1",
		ExpiresAt:   time.Now().Add(time.Hour),
		Generation:  1,
		ClientID:    xaiClientID,
	}
	if err := mgr1.Store(context.Background(), xaiClientID, seed); err != nil {
		t.Fatalf("seed credential: %v", err)
	}

	// Simulated restart: a cold manager over the same dir, a fresh store over
	// the same providers.json. NewRuntimeProviderStore loads before Creds is
	// set, so Load again after wiring the store (Restore would also re-inject).
	mgr2, err := auth.NewManager(credDir, nil)
	if err != nil {
		t.Fatalf("auth.NewManager (restart): %v", err)
	}
	if cred, ok := mgr2.Peek(xaiClientID); !ok || cred.AccessToken != "restart-tok-1" {
		t.Fatalf("cold Peek after restart = %+v, %v; want the seeded token", cred, ok)
	}

	store2 := NewRuntimeProviderStore(providersPath)
	store2.DefaultDataDir = dataDir
	store2.Creds = mgr2
	if _, err := store2.Load(); err != nil {
		t.Fatalf("Load after restart: %v", err)
	}
	restored, ok := store2.List()["xai"]
	if !ok {
		t.Fatal("xai lane not restored after restart")
	}
	if restored.Creds == nil {
		t.Fatal("restored lane has no injected Creds")
	}
	p, err := openaicompat.New(restored)
	if err != nil {
		t.Fatalf("openaicompat.New (restored): %v", err)
	}
	tok, err := p.Token(context.Background())
	if err != nil {
		t.Fatalf("Token (restored): %v", err)
	}
	if tok != "restart-tok-1" {
		t.Fatalf("restored Token = %q, want restart-tok-1 (no re-login)", tok)
	}

	// And Restore registers the lane into a fresh registry.
	registry := provider.NewRegistry()
	if got := store2.Restore(registry); len(got) != 1 || got[0] != "xai" {
		t.Fatalf("Restore registered %v, want [xai]", got)
	}
	if _, ok := registry.Get("xai"); !ok {
		t.Fatal("xai not in registry after Restore")
	}
}
