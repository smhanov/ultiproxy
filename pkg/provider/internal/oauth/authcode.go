package oauth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// AuthCodeConfig configures an OAuth 2.0 authorization-code flow (with PKCE).
type AuthCodeConfig struct {
	ClientID     string
	ClientSecret string
	AuthURL      string
	TokenURL     string
	RedirectURI  string
	Scope        string
	HTTPClient   *http.Client
}

// PKCE holds a PKCE verifier/challenge pair plus CSRF state.
type PKCE struct {
	Verifier  string
	Challenge string
	State     string
}

// NewPKCE generates a S256 PKCE pair and random state.
func NewPKCE() (PKCE, error) {
	verifier, err := randomURLString(32)
	if err != nil {
		return PKCE{}, err
	}
	state, err := randomURLString(16)
	if err != nil {
		return PKCE{}, err
	}
	sum := sha256.Sum256([]byte(verifier))
	return PKCE{
		Verifier:  verifier,
		Challenge: base64.RawURLEncoding.EncodeToString(sum[:]),
		State:     state,
	}, nil
}

func randomURLString(n int) (string, error) {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

// AuthorizationURL builds the browser consent URL.
func AuthorizationURL(cfg AuthCodeConfig, pkce PKCE) string {
	q := url.Values{}
	q.Set("client_id", cfg.ClientID)
	q.Set("redirect_uri", cfg.RedirectURI)
	q.Set("response_type", "code")
	q.Set("access_type", "offline")
	q.Set("prompt", "consent")
	q.Set("include_granted_scopes", "true")
	if cfg.Scope != "" {
		q.Set("scope", cfg.Scope)
	}
	if pkce.State != "" {
		q.Set("state", pkce.State)
	}
	if pkce.Challenge != "" {
		q.Set("code_challenge", pkce.Challenge)
		q.Set("code_challenge_method", "S256")
	}
	sep := "?"
	if strings.Contains(cfg.AuthURL, "?") {
		sep = "&"
	}
	return cfg.AuthURL + sep + q.Encode()
}

// ExchangeCode swaps an authorization code for tokens.
func ExchangeCode(ctx context.Context, cfg AuthCodeConfig, code, verifier string) (*TokenResponse, error) {
	if code == "" {
		return nil, fmt.Errorf("oauth: empty authorization code")
	}
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("code", code)
	form.Set("client_id", cfg.ClientID)
	form.Set("redirect_uri", cfg.RedirectURI)
	if cfg.ClientSecret != "" {
		form.Set("client_secret", cfg.ClientSecret)
	}
	if verifier != "" {
		form.Set("code_verifier", verifier)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.TokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("oauth: failed to create code exchange request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("oauth: code exchange failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("oauth: failed to read code exchange response: %w", err)
	}

	var tr TokenResponse
	if err := json.Unmarshal(body, &tr); err != nil {
		return nil, fmt.Errorf("oauth: failed to parse code exchange response: %w", err)
	}
	if resp.StatusCode != http.StatusOK || tr.AccessToken == "" {
		if tr.Error != "" {
			return nil, fmt.Errorf("oauth error: %s (%s)", tr.Error, tr.ErrorDesc)
		}
		return nil, fmt.Errorf("oauth: code exchange returned HTTP %d: %s", resp.StatusCode, string(body))
	}
	return &tr, nil
}
