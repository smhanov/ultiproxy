package server

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// recordingNextHandler captures the auth context values the middleware injects.
func recordingNextHandler(t *testing.T, name *string, hash *string) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*name = ClientNameFromContext(r.Context())
		*hash = ClientKeyHashFromContext(r.Context())
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	})
}

func keyHashHex(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// AC1: a request carrying only the Anthropic-style x-api-key header (no
// Authorization header at all) must be authenticated, not rejected with 401.
func TestAuthMiddleware_APIKeyHeaderAuthenticates(t *testing.T) {
	m := NewAuthMiddleware("", map[string]string{"claude-code": "anthropic-style-key"})

	var gotName, gotHash string
	h := m.Wrap(recordingNextHandler(t, &gotName, &gotHash))

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{}`))
	req.Header.Set("x-api-key", "anthropic-style-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("POST /v1/messages with x-api-key only = %d: %s", rec.Code, rec.Body.String())
	}
	if gotName != "claude-code" {
		t.Errorf("authenticated client name = %q, want %q", gotName, "claude-code")
	}
	if want := keyHashHex("anthropic-style-key"); gotHash != want {
		t.Errorf("authenticated key hash = %q, want %q", gotHash, want)
	}
}

// The static admin key is accepted through x-api-key too, and identifies the
// caller as "admin".
func TestAuthMiddleware_APIKeyHeaderAdminKey(t *testing.T) {
	m := NewAuthMiddleware("admin-secret", map[string]string{"claude-code": "anthropic-style-key"})

	var gotName, gotHash string
	h := m.Wrap(recordingNextHandler(t, &gotName, &gotHash))

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{}`))
	req.Header.Set("x-api-key", "admin-secret")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("POST /v1/messages with admin x-api-key = %d: %s", rec.Code, rec.Body.String())
	}
	if gotName != "admin" {
		t.Errorf("authenticated client name = %q, want %q", gotName, "admin")
	}
	if want := keyHashHex("admin-secret"); gotHash != want {
		t.Errorf("authenticated key hash = %q, want %q", gotHash, want)
	}
}

// A wrong x-api-key is still a 401 with the OpenAI-shaped error body.
func TestAuthMiddleware_APIKeyHeaderWrongKeyRejected(t *testing.T) {
	m := NewAuthMiddleware("", map[string]string{"claude-code": "anthropic-style-key"})

	var gotName, gotHash string
	h := m.Wrap(recordingNextHandler(t, &gotName, &gotHash))

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{}`))
	req.Header.Set("x-api-key", "not-the-configured-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("POST /v1/messages with a wrong x-api-key = %d, want 401", rec.Code)
	}
	if gotName != "" || gotHash != "" {
		t.Errorf("next handler ran for a rejected key (name=%q hash=%q)", gotName, gotHash)
	}
	if !strings.Contains(rec.Body.String(), "invalid_api_key") {
		t.Errorf("expected invalid_api_key error body, got:\n%s", rec.Body.String())
	}
}

// When both headers are present the Authorization: Bearer credential wins, so
// OpenAI-style clients keep their existing semantics.
func TestAuthMiddleware_BearerTakesPrecedenceOverAPIKeyHeader(t *testing.T) {
	m := NewAuthMiddleware("", map[string]string{
		"claude-code": "anthropic-style-key",
		"openai-ish":  "bearer-style-key",
	})

	var gotName, gotHash string
	h := m.Wrap(recordingNextHandler(t, &gotName, &gotHash))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer bearer-style-key")
	req.Header.Set("x-api-key", "anthropic-style-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("POST /v1/chat/completions with both headers = %d: %s", rec.Code, rec.Body.String())
	}
	if gotName != "openai-ish" {
		t.Errorf("authenticated client name = %q, want the Bearer identity %q", gotName, "openai-ish")
	}
}
