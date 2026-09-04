package antigravity

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/oauth"
)

func TestGetTokenUsesAuthManagerNotStaticWhenBothSet(t *testing.T) {
	dir := t.TempDir()
	mgr, err := auth.NewManager(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	cred := auth.Credential{
		Provider:    "antigravity",
		AccessToken: "manager-token",
		ExpiresAt:   time.Now().Add(time.Hour),
		ClientID:    DefaultClientID,
		ProjectID:   "proj-from-manager",
	}
	if err := mgr.Store(context.Background(), DefaultClientID, cred); err != nil {
		t.Fatal(err)
	}

	p := New(Config{
		AuthManager: mgr,
		StaticToken: "stale-static",
		ClientID:    DefaultClientID,
	})
	tok, err := p.getToken(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tok != "manager-token" {
		t.Fatalf("got %q, want manager-token (static must not win)", tok)
	}
	if p.ProjectID() != "proj-from-manager" {
		t.Fatalf("project = %q, want proj-from-manager", p.ProjectID())
	}
}

func TestGetTokenRefreshesViaManager(t *testing.T) {
	var refreshCount atomic.Int32
	dir := t.TempDir()
	ref := func(ctx context.Context, cred auth.Credential) (auth.Credential, error) {
		refreshCount.Add(1)
		cred.AccessToken = "refreshed-token"
		cred.ExpiresAt = time.Now().Add(time.Hour)
		return cred, nil
	}
	mgr, err := auth.NewManager(dir, ref)
	if err != nil {
		t.Fatal(err)
	}
	if err := mgr.Store(context.Background(), DefaultClientID, auth.Credential{
		Provider:     "antigravity",
		AccessToken:  "almost-dead",
		RefreshToken: "rt",
		ExpiresAt:    time.Now().Add(30 * time.Second), // within 5m refresh window
		ClientID:     DefaultClientID,
	}); err != nil {
		t.Fatal(err)
	}

	p := New(Config{AuthManager: mgr, ClientID: DefaultClientID, Refresher: ref})
	tok, err := p.getToken(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tok != "refreshed-token" {
		t.Fatalf("got %q, want refreshed-token", tok)
	}
	if refreshCount.Load() != 1 {
		t.Fatalf("refresh count = %d, want 1", refreshCount.Load())
	}
}

func TestLoginAuthCodePersistsTokensAndProject(t *testing.T) {
	var sawExchange, sawAssist bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/token"):
			sawExchange = true
			body, _ := io.ReadAll(r.Body)
			form := string(body)
			if !strings.Contains(form, "grant_type=authorization_code") || !strings.Contains(form, "code=4%2F0A-live") {
				t.Errorf("unexpected token form: %s", form)
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token":  "new-access",
				"refresh_token": "new-refresh",
				"expires_in":    3600,
				"token_type":    "Bearer",
			})
		case strings.Contains(r.URL.Path, "loadCodeAssist"):
			sawAssist = true
			if got := r.Header.Get("Authorization"); got != "Bearer new-access" {
				t.Errorf("assist auth = %q", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"cloudaicompanionProject": "aicode-consumers",
			})
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	dir := t.TempDir()
	mgr, err := auth.NewManager(dir, oauth.MakeRefresher(srv.Client(), srv.URL+"/token", DefaultClientID, DefaultClientSecret))
	if err != nil {
		t.Fatal(err)
	}

	var printed string
	p := New(Config{
		BaseURL:      srv.URL,
		TokenURL:     srv.URL + "/token",
		AuthURL:      "https://accounts.google.com/o/oauth2/auth",
		RedirectURI:  DefaultRedirectURI,
		ClientID:     DefaultClientID,
		ClientSecret: DefaultClientSecret,
		AuthManager:  mgr,
		HTTPClient:   srv.Client(),
		OnAuthURL:    func(u string) { printed = u },
		ReadCode:     func() (string, error) { return "4/0A-live", nil },
	})
	if err := p.Login(context.Background()); err != nil {
		t.Fatalf("Login: %v", err)
	}
	if printed == "" || !strings.Contains(printed, "accounts.google.com") {
		t.Fatalf("did not print auth url: %q", printed)
	}
	if !sawExchange || !sawAssist {
		t.Fatalf("exchange=%v assist=%v", sawExchange, sawAssist)
	}
	tok, err := p.getToken(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tok != "new-access" {
		t.Fatalf("token = %q", tok)
	}
	if p.ProjectID() != "aicode-consumers" {
		t.Fatalf("project = %q", p.ProjectID())
	}
	disk, err := mgr.LoadFromDisk(DefaultClientID)
	if err != nil {
		t.Fatal(err)
	}
	if disk.RefreshToken != "new-refresh" || disk.ProjectID != "aicode-consumers" {
		t.Fatalf("disk cred: %+v", disk)
	}
}

func TestGenerateUsesManagerToken(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"PONG"}]},"finishReason":"STOP"}]}}`))
	}))
	defer srv.Close()

	dir := t.TempDir()
	mgr, err := auth.NewManager(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	_ = mgr.Store(context.Background(), DefaultClientID, auth.Credential{
		AccessToken: "mgr-tok",
		ExpiresAt:   time.Now().Add(time.Hour),
		ClientID:    DefaultClientID,
		ProjectID:   "p1",
	})
	p := New(Config{BaseURL: srv.URL, AuthManager: mgr, HTTPClient: srv.Client(), ClientID: DefaultClientID})
	resp, err := p.Generate(context.Background(), []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}},
	}, provider.WithModel("gemini-3.8-flash-high"))
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer mgr-tok" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if resp == nil {
		t.Fatal("nil response")
	}
}
