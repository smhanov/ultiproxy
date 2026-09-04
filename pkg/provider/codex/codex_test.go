package codex

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestMaxCompletionTokensConversion(t *testing.T) {
	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Write hello world"}}},
	}

	// Case 1: Newer gpt-5 model converts max_tokens to max_completion_tokens
	cfgGPT5 := provider.NewRequestConfig(
		provider.WithModel("gpt-5.4"),
		provider.WithMaxTokens(1024),
	)
	payload1, err := BuildPayload(msgs, cfgGPT5, false)
	if err != nil {
		t.Fatalf("BuildPayload failed: %v", err)
	}
	if _, ok := payload1["max_tokens"]; ok {
		t.Errorf("max_tokens should NOT be present for gpt-5.4")
	}
	if val, ok := payload1["max_completion_tokens"]; !ok || val != 1024 {
		t.Errorf("max_completion_tokens = %v, want 1024", val)
	}

	// Case 2: Older / other model keeps max_tokens
	cfgGPT4 := provider.NewRequestConfig(
		provider.WithModel("gpt-4o"),
		provider.WithMaxTokens(2048),
	)
	payload2, err := BuildPayload(msgs, cfgGPT4, false)
	if err != nil {
		t.Fatalf("BuildPayload failed: %v", err)
	}
	if val, ok := payload2["max_tokens"]; !ok || val != 2048 {
		t.Errorf("max_tokens = %v, want 2048", val)
	}
	if _, ok := payload2["max_completion_tokens"]; ok {
		t.Errorf("max_completion_tokens should NOT be present for gpt-4o")
	}
}

func TestTokenRefreshOnceOn401(t *testing.T) {
	var attempts int32
	var authTokens []string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		att := atomic.AddInt32(&attempts, 1)
		authHeader := r.Header.Get("Authorization")
		authTokens = append(authTokens, authHeader)

		if att == 1 {
			// First attempt returns 401 Unauthorized
			http.Error(w, "Unauthorized token expired", http.StatusUnauthorized)
			return
		}

		// Second attempt succeeds
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"id": "codex-retry-ok",
			"choices": [{"message": {"role": "assistant", "content": "Success after refresh"}, "finish_reason": "stop"}]
		}`))
	}))
	defer srv.Close()

	// Mock auth manager & refresher
	tempDir := t.TempDir()
	refreshed := false
	refresher := func(ctx context.Context, cred auth.Credential) (auth.Credential, error) {
		refreshed = true
		cred.AccessToken = "new-refreshed-token"
		cred.ExpiresAt = time.Now().Add(1 * time.Hour)
		cred.Generation++
		return cred, nil
	}

	mgr, err := auth.NewManager(tempDir, refresher)
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	initialCred := auth.Credential{
		ClientID:     "app_EMoamEEZ73f0CkXaXp7hrann",
		AccessToken:  "initial-expired-token",
		RefreshToken: "valid-refresh-token",
		ExpiresAt:    time.Now().Add(10 * time.Minute),
		Generation:   1,
	}
	if err := mgr.Store(context.Background(), "app_EMoamEEZ73f0CkXaXp7hrann", initialCred); err != nil {
		t.Fatalf("Store failed: %v", err)
	}

	p := New(Config{
		BaseURL:     srv.URL,
		AuthManager: mgr,
		Refresher:   refresher,
		HTTPClient:  srv.Client(),
	})

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Test"}}},
	}

	resp, err := p.Generate(context.Background(), msgs, provider.WithModel("gpt-5.5"))
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.ID != "codex-retry-ok" {
		t.Errorf("resp.ID = %q, want 'codex-retry-ok'", resp.ID)
	}

	if atomic.LoadInt32(&attempts) != 2 {
		t.Fatalf("expected 2 attempts, got %d", attempts)
	}
	if !refreshed {
		t.Error("expected auth refresher to be called")
	}
	if len(authTokens) != 2 {
		t.Fatalf("expected 2 auth tokens recorded, got %d", len(authTokens))
	}
	if authTokens[0] != "Bearer initial-expired-token" {
		t.Errorf("token 0 = %q, want 'Bearer initial-expired-token'", authTokens[0])
	}
	if authTokens[1] != "Bearer new-refreshed-token" {
		t.Errorf("token 1 = %q, want 'Bearer new-refreshed-token'", authTokens[1])
	}
}

func TestWhamUsageParse(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "codex", "wham-usage.json")
	fixtureData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/backend-api/wham/usage") {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(fixtureData)
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		StaticToken: "test-tok",
		HTTPClient:  srv.Client(),
	})

	snap, err := p.Quota(context.Background())
	if err != nil {
		t.Fatalf("Quota failed: %v", err)
	}

	if len(snap.Windows) != 2 {
		t.Fatalf("expected 2 windows, got %d", len(snap.Windows))
	}

	w1 := snap.Windows[0]
	if w1.Label != "5 hour" {
		t.Errorf("w1.Label = %q, want '5 hour'", w1.Label)
	}
	if w1.UsedPct != 18.0 {
		t.Errorf("w1.UsedPct = %g, want 18.0", w1.UsedPct)
	}
	if w1.Remaining != 82.0 {
		t.Errorf("w1.Remaining = %g, want 82.0", w1.Remaining)
	}
	if w1.SecondsRemaining != 2741 {
		t.Errorf("w1.SecondsRemaining = %d, want 2741", w1.SecondsRemaining)
	}

	w2 := snap.Windows[1]
	if w2.Label != "Weekly" {
		t.Errorf("w2.Label = %q, want 'Weekly'", w2.Label)
	}
	if w2.UsedPct != 42.3 {
		t.Errorf("w2.UsedPct = %g, want 42.3", w2.UsedPct)
	}
	if w2.Remaining != 57.7 {
		t.Errorf("w2.Remaining = %g, want 57.7", w2.Remaining)
	}
	if w2.SecondsRemaining != 589541 {
		t.Errorf("w2.SecondsRemaining = %d, want 589541", w2.SecondsRemaining)
	}

	if !strings.Contains(snap.Detail, "plus") {
		t.Errorf("detail = %q, want plan 'plus'", snap.Detail)
	}
}

func TestSupportedModelsList(t *testing.T) {
	p := New(Config{})
	models := p.Models()

	expected := []string{"gpt-5.5", "gpt-5.6-luna", "gpt-5.4", "gpt-5.4-mini", "gpt-5.6-sol", "gpt-5.6-terra"}
	for _, m := range expected {
		found := false
		for _, sm := range models {
			if sm == m {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected model %q in supported models list %v", m, models)
		}
	}
}

func TestStreamSynchronousErrorOnNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Bad gateway", http.StatusBadGateway)
	}))
	defer srv.Close()

	p := New(Config{
		BaseURL:     srv.URL,
		StaticToken: "test-tok",
		HTTPClient:  srv.Client(),
	})

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "Hello"}}},
	}

	_, err := p.Stream(context.Background(), msgs, provider.WithModel("gpt-5.5"))
	if err == nil {
		t.Fatal("expected synchronous error on 502 status code, got nil")
	}
	if !strings.Contains(err.Error(), "502") {
		t.Errorf("expected 502 in error message, got %v", err)
	}
}

func TestGetTokenPrefersAuthManagerOverStatic(t *testing.T) {
	dir := t.TempDir()
	mgr, err := auth.NewManager(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := mgr.Store(context.Background(), DefaultClientID, auth.Credential{
		Provider:    "codex",
		AccessToken: "manager-token",
		ExpiresAt:   time.Now().Add(time.Hour),
		ClientID:    DefaultClientID,
	}); err != nil {
		t.Fatal(err)
	}
	p := New(Config{AuthManager: mgr, StaticToken: "stale-static", ClientID: DefaultClientID})
	tok, err := p.getToken(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tok != "manager-token" {
		t.Fatalf("got %q, want manager-token", tok)
	}
}

func TestRegistry(t *testing.T) {
	r := provider.NewRegistry()
	p := New(Config{StaticToken: "test"})
	p.Register(r)
	got, ok := r.Get("codex")
	if !ok || got.Inference == nil || got.Quota == nil || got.Auth == nil {
		t.Fatalf("codex registry get failed: %+v", got)
	}
	if !got.Capabilities.Chat || !got.Capabilities.Tools || !got.Capabilities.Reasoning || !got.Capabilities.Streaming {
		t.Fatalf("unexpected capabilities: %+v", got.Capabilities)
	}
}
