package openaicompat

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/auth"
)

// AC2: building an OAuth lane with no injected store fails with an explicit
// error naming the missing dependency, and creates no directory outside the
// daemon data dir (in particular, no os.TempDir() fallback).
func TestXAIRequiresCredentialStore(t *testing.T) {
	fallback := filepath.Join(os.TempDir(), "ultiproxy-xai-auth")
	// Ensure a clean slate so os.Stat below proves New created nothing.
	_ = os.RemoveAll(fallback)

	_, err := New(Config{
		Name:                       "xai",
		BaseURL:                    "http://127.0.0.1:1",
		OptOutModelListPassthrough: true,
		Quirks:                     Quirks{AuthViaOAuthManager: true},
	})
	if err == nil {
		t.Fatal("New with AuthViaOAuthManager and no Creds succeeded, want hard error")
	}
	if !strings.Contains(err.Error(), "credential store not injected") {
		t.Fatalf("New error = %q, want it to name the missing credential store", err)
	}
	if _, statErr := os.Stat(fallback); !os.IsNotExist(statErr) {
		t.Fatalf("TempDir fallback %q exists after failed New (err = %v): no directory may be created outside the daemon data dir", fallback, statErr)
	}
}

// AC2 (completeXAI half, with a real pending flow): StartLogin against a fake
// device endpoint, then CompleteLogin with Creds == nil must fail naming the
// store and must not create the /tmp fallback.
func TestXAICompleteLoginRequiresStore(t *testing.T) {
	fallback := filepath.Join(os.TempDir(), "ultiproxy-xai-auth")
	_ = os.RemoveAll(fallback)

	mux := http.NewServeMux()
	mux.HandleFunc("/device", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"device_code": "dev-creds-test",
			"user_code": "CREDS-0000",
			"verification_uri": "https://accounts.x.ai/oauth2/device",
			"expires_in": 300,
			"interval": 0
		}`))
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"access_token": "should-never-persist",
			"refresh_token": "should-never-persist-rt",
			"expires_in": 3600
		}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	mgr, err := auth.NewManager(t.TempDir(), nil)
	if err != nil {
		t.Fatalf("auth.NewManager: %v", err)
	}
	p, err := New(Config{
		Name:                       "xai",
		BaseURL:                    "http://127.0.0.1:1",
		Creds:                      mgr,
		HTTPClient:                 srv.Client(),
		DeviceAuthURL:              srv.URL + "/device",
		TokenURL:                   srv.URL + "/token",
		OptOutModelListPassthrough: true,
		Quirks:                     Quirks{AuthViaOAuthManager: true},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := p.StartLogin(context.Background()); err != nil {
		t.Fatalf("StartLogin: %v", err)
	}
	// Strip the store after StartLogin to simulate a lane that reached login
	// without injection (e.g. an MCP-added lane whose store was never wired).
	p.cfg.Creds = nil
	err = p.CompleteLogin(context.Background(), "")
	if err == nil {
		t.Fatal("CompleteLogin with nil Creds succeeded, want hard error")
	}
	if !strings.Contains(err.Error(), "credential store not injected") {
		t.Fatalf("CompleteLogin error = %q, want it to name the missing credential store", err)
	}
	if _, statErr := os.Stat(fallback); !os.IsNotExist(statErr) {
		t.Fatalf("TempDir fallback %q exists after failed CompleteLogin (err = %v)", fallback, statErr)
	}
}
