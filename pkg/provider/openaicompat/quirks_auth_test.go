package openaicompat

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
)

// newCountingRefresherManager builds an auth.Manager whose refresher hands out a
// fresh access token on every call, so tests can tell a real refresh from a
// cached read.
func newCountingRefresherManager(t *testing.T, calls *atomic.Int64) *auth.Manager {
	t.Helper()
	base := time.Now().UTC()
	mgr, err := auth.NewManager(t.TempDir(), func(ctx context.Context, cred auth.Credential) (auth.Credential, error) {
		n := calls.Add(1)
		return auth.Credential{
			Provider:     "xai",
			AccessToken:  fmt.Sprintf("token-%d", n),
			RefreshToken: fmt.Sprintf("rt-%d", n),
			ExpiresAt:    base.Add(time.Hour),
			Generation:   cred.Generation + 1,
			ClientID:     defaultXAIClientID,
		}, nil
	})
	if err != nil {
		t.Fatalf("auth.NewManager: %v", err)
	}
	return mgr
}

// AC2: Invalidate(token) followed by Token() must return a newly refreshed
// token, never the still-unexpired cached one.
func TestOAuthManagerTokenSource_InvalidateForcesRefresh(t *testing.T) {
	var refreshes atomic.Int64
	mgr := newCountingRefresherManager(t, &refreshes)

	seed := auth.Credential{
		Provider:    "xai",
		AccessToken: "token-0",
		ExpiresAt:   time.Now().UTC().Add(time.Hour), // well outside the refresh window
		Generation:  1,
		ClientID:    defaultXAIClientID,
	}
	if err := mgr.Store(context.Background(), defaultXAIClientID, seed); err != nil {
		t.Fatalf("store seed credential: %v", err)
	}

	src := NewOAuthManagerTokenSource(mgr, defaultXAIClientID)
	ctx := context.Background()

	first, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("first Token: %v", err)
	}
	if first.AccessToken != "token-0" {
		t.Fatalf("first Token = %q, want the stored %q", first.AccessToken, "token-0")
	}

	src.Invalidate("token-0")

	second, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token after Invalidate: %v", err)
	}
	if second.AccessToken == "token-0" {
		t.Fatalf("Token after Invalidate returned the invalidated token %q", second.AccessToken)
	}
	if second.AccessToken != "token-1" {
		t.Errorf("Token after Invalidate = %q, want the refreshed %q", second.AccessToken, "token-1")
	}
	if got := refreshes.Load(); got != 1 {
		t.Errorf("refresher calls = %d, want 1", got)
	}

	// The refreshed token is cached: a further read must not refresh again.
	third, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("third Token: %v", err)
	}
	if third.AccessToken != "token-1" {
		t.Errorf("third Token = %q, want the cached %q", third.AccessToken, "token-1")
	}
	if got := refreshes.Load(); got != 1 {
		t.Errorf("refresher calls after a cached read = %d, want 1", got)
	}
}

// provider.Refresh() calls Invalidate("") because it has no token in hand; that
// must force a refresh too.
func TestOAuthManagerTokenSource_InvalidateEmptyTokenForcesRefresh(t *testing.T) {
	var refreshes atomic.Int64
	mgr := newCountingRefresherManager(t, &refreshes)

	if err := mgr.Store(context.Background(), defaultXAIClientID, auth.Credential{
		Provider:    "xai",
		AccessToken: "token-0",
		ExpiresAt:   time.Now().UTC().Add(time.Hour),
		Generation:  1,
		ClientID:    defaultXAIClientID,
	}); err != nil {
		t.Fatalf("store seed credential: %v", err)
	}

	src := NewOAuthManagerTokenSource(mgr, defaultXAIClientID)
	ctx := context.Background()

	if tok, err := src.Token(ctx); err != nil || tok.AccessToken != "token-0" {
		t.Fatalf("first Token = %q, %v", tok, err)
	}

	src.Invalidate("")

	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token after Invalidate(\"\"): %v", err)
	}
	if tok.AccessToken != "token-1" {
		t.Errorf("Token after Invalidate(\"\") = %q, want the refreshed %q", tok.AccessToken, "token-1")
	}
}

// Invalidating a token that is no longer the current one must not throw away a
// healthy credential (mirrors llmhub's InvalidatableTokenSource semantics).
func TestOAuthManagerTokenSource_InvalidateStaleTokenKeepsCurrent(t *testing.T) {
	var refreshes atomic.Int64
	mgr := newCountingRefresherManager(t, &refreshes)

	if err := mgr.Store(context.Background(), defaultXAIClientID, auth.Credential{
		Provider:    "xai",
		AccessToken: "token-0",
		ExpiresAt:   time.Now().UTC().Add(time.Hour),
		Generation:  1,
		ClientID:    defaultXAIClientID,
	}); err != nil {
		t.Fatalf("store seed credential: %v", err)
	}

	src := NewOAuthManagerTokenSource(mgr, defaultXAIClientID)
	ctx := context.Background()

	src.Invalidate("an-old-revoked-token")

	tok, err := src.Token(ctx)
	if err != nil {
		t.Fatalf("Token after invalidating a stale token: %v", err)
	}
	if tok.AccessToken != "token-0" {
		t.Errorf("Token after invalidating a stale token = %q, want the still-valid %q", tok.AccessToken, "token-0")
	}
	if got := refreshes.Load(); got != 0 {
		t.Errorf("refresher calls = %d, want 0", got)
	}
}
