package oauth

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestNewPKCE(t *testing.T) {
	a, err := NewPKCE()
	if err != nil {
		t.Fatalf("NewPKCE: %v", err)
	}
	b, err := NewPKCE()
	if err != nil {
		t.Fatalf("NewPKCE: %v", err)
	}
	if a.Verifier == "" || a.Challenge == "" || a.State == "" {
		t.Fatalf("empty pkce: %+v", a)
	}
	if a.Verifier == b.Verifier || a.State == b.State || a.Challenge == b.Challenge {
		t.Fatal("PKCE values must be unique per call")
	}
	if strings.ContainsAny(a.Verifier, "+/=") || strings.ContainsAny(a.Challenge, "+/=") {
		t.Fatalf("pkce must be raw url-safe base64, got verifier=%q challenge=%q", a.Verifier, a.Challenge)
	}
}

func TestAuthorizationURL(t *testing.T) {
	pkce := PKCE{Verifier: "v", Challenge: "chal", State: "st"}
	got := AuthorizationURL(AuthCodeConfig{
		ClientID:    "cid",
		AuthURL:     "https://accounts.google.com/o/oauth2/auth",
		RedirectURI: "https://antigravity.google/oauth-callback",
		Scope:       "https://www.googleapis.com/auth/cloud-platform email",
	}, pkce)
	u, err := url.Parse(got)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	q := u.Query()
	want := map[string]string{
		"client_id":             "cid",
		"redirect_uri":          "https://antigravity.google/oauth-callback",
		"response_type":         "code",
		"access_type":           "offline",
		"prompt":                "consent",
		"code_challenge":        "chal",
		"code_challenge_method": "S256",
		"state":                 "st",
	}
	for k, v := range want {
		if q.Get(k) != v {
			t.Errorf("%s = %q, want %q", k, q.Get(k), v)
		}
	}
	if !strings.Contains(q.Get("scope"), "cloud-platform") {
		t.Errorf("scope missing cloud-platform: %q", q.Get("scope"))
	}
}

func TestExchangeCode(t *testing.T) {
	var gotForm url.Values
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotForm, _ = url.ParseQuery(string(body))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "acc",
			"refresh_token": "ref",
			"expires_in":    3600,
			"token_type":    "Bearer",
		})
	}))
	defer srv.Close()

	tr, err := ExchangeCode(context.Background(), AuthCodeConfig{
		ClientID:     "cid",
		ClientSecret: "sec",
		TokenURL:     srv.URL,
		RedirectURI:  "https://antigravity.google/oauth-callback",
		HTTPClient:   srv.Client(),
	}, "4/0A-code", "verifier")
	if err != nil {
		t.Fatalf("ExchangeCode: %v", err)
	}
	if tr.AccessToken != "acc" || tr.RefreshToken != "ref" {
		t.Fatalf("tokens: %+v", tr)
	}
	if gotForm.Get("grant_type") != "authorization_code" {
		t.Errorf("grant_type = %q", gotForm.Get("grant_type"))
	}
	if gotForm.Get("code") != "4/0A-code" {
		t.Errorf("code = %q", gotForm.Get("code"))
	}
	if gotForm.Get("code_verifier") != "verifier" {
		t.Errorf("code_verifier = %q", gotForm.Get("code_verifier"))
	}
	if gotForm.Get("client_id") != "cid" || gotForm.Get("client_secret") != "sec" {
		t.Errorf("client fields: %v", gotForm)
	}
	if gotForm.Get("redirect_uri") != "https://antigravity.google/oauth-callback" {
		t.Errorf("redirect_uri = %q", gotForm.Get("redirect_uri"))
	}
}

func TestExchangeCodeRejectsEmpty(t *testing.T) {
	_, err := ExchangeCode(context.Background(), AuthCodeConfig{TokenURL: "http://127.0.0.1:1"}, "", "")
	if err == nil {
		t.Fatal("expected error for empty code")
	}
}

func TestExchangeCodeHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"invalid_grant","error_description":"Malformed auth code"}`))
	}))
	defer srv.Close()
	_, err := ExchangeCode(context.Background(), AuthCodeConfig{
		TokenURL:   srv.URL,
		HTTPClient: srv.Client(),
	}, "bad", "v")
	if err == nil || !strings.Contains(err.Error(), "invalid_grant") {
		t.Fatalf("want invalid_grant error, got %v", err)
	}
}
