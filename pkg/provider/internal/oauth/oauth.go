package oauth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
)

// DeviceCodeResponse holds the response from an OAuth device authorization endpoint.
type DeviceCodeResponse struct {
	DeviceCode              string `json:"device_code"`
	UserCode                string `json:"user_code"`
	VerificationURI         string `json:"verification_uri"`
	VerificationURIComplete string `json:"verification_uri_complete,omitempty"`
	ExpiresIn               int    `json:"expires_in"`
	Interval                int    `json:"interval"`
}

// TokenResponse represents a standard OAuth token endpoint response.
type TokenResponse struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	RefreshToken string `json:"refresh_token,omitempty"`
	ExpiresIn    int    `json:"expires_in,omitempty"`
	Scope        string `json:"scope,omitempty"`
	Error        string `json:"error,omitempty"`
	ErrorDesc    string `json:"error_description,omitempty"`
}

// DeviceFlowConfig configures an OAuth 2.0 device authorization flow.
type DeviceFlowConfig struct {
	ClientID      string
	ClientSecret  string
	Scope         string
	Audience      string
	DeviceAuthURL string
	TokenURL      string
	HTTPClient    *http.Client
}

// RequestDeviceCode initiates the device authorization request.
func RequestDeviceCode(ctx context.Context, cfg DeviceFlowConfig) (*DeviceCodeResponse, error) {
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	form := url.Values{}
	form.Set("client_id", cfg.ClientID)
	if cfg.Scope != "" {
		form.Set("scope", cfg.Scope)
	}
	if cfg.Audience != "" {
		form.Set("audience", cfg.Audience)
	}
	if cfg.ClientSecret != "" {
		form.Set("client_secret", cfg.ClientSecret)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.DeviceAuthURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("oauth: failed to create device code request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("oauth: device code request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("oauth: failed to read device code response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("oauth: device code request returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	var dcr DeviceCodeResponse
	if err := json.Unmarshal(body, &dcr); err != nil {
		return nil, fmt.Errorf("oauth: failed to parse device code response: %w", err)
	}

	if dcr.Interval <= 0 {
		dcr.Interval = 5
	}
	return &dcr, nil
}

// PollToken polls the token endpoint until a token is returned, expired, or access denied.
func PollToken(ctx context.Context, cfg DeviceFlowConfig, deviceCode string, interval int) (*TokenResponse, error) {
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	if interval <= 0 {
		interval = 5
	}
	pollDuration := time.Duration(interval) * time.Second

	ticker := time.NewTicker(pollDuration)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			form := url.Values{}
			form.Set("client_id", cfg.ClientID)
			form.Set("grant_type", "urn:ietf:params:oauth:grant-type:device_code")
			form.Set("device_code", deviceCode)
			if cfg.ClientSecret != "" {
				form.Set("client_secret", cfg.ClientSecret)
			}

			req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.TokenURL, strings.NewReader(form.Encode()))
			if err != nil {
				return nil, err
			}
			req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			req.Header.Set("Accept", "application/json")

			resp, err := client.Do(req)
			if err != nil {
				return nil, err
			}

			body, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				return nil, err
			}

			var tr TokenResponse
			if err := json.Unmarshal(body, &tr); err != nil {
				return nil, fmt.Errorf("oauth: failed to parse token response: %w", err)
			}

			if resp.StatusCode == http.StatusOK && tr.AccessToken != "" {
				return &tr, nil
			}

			switch tr.Error {
			case "authorization_pending":
				// continue polling
				continue
			case "slow_down":
				pollDuration += 5 * time.Second
				ticker.Reset(pollDuration)
				continue
			case "expired_token":
				return nil, errors.New("oauth: device code expired")
			case "access_denied":
				return nil, errors.New("oauth: user denied authorization")
			default:
				if tr.Error != "" {
					return nil, fmt.Errorf("oauth error: %s (%s)", tr.Error, tr.ErrorDesc)
				}
				return nil, fmt.Errorf("oauth token request returned HTTP %d: %s", resp.StatusCode, string(body))
			}
		}
	}
}

// RefreshToken exchanges a refresh_token for a new access_token.
func RefreshToken(ctx context.Context, client *http.Client, tokenURL, clientID, clientSecret, refreshToken string) (*TokenResponse, error) {
	if client == nil {
		client = http.DefaultClient
	}

	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("refresh_token", refreshToken)
	form.Set("client_id", clientID)
	if clientSecret != "" {
		form.Set("client_secret", clientSecret)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("oauth: failed to create refresh request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("oauth: refresh request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("oauth: failed to read refresh response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("oauth: refresh returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	var tr TokenResponse
	if err := json.Unmarshal(body, &tr); err != nil {
		return nil, fmt.Errorf("oauth: failed to parse refresh response: %w", err)
	}

	if tr.AccessToken == "" {
		return nil, errors.New("oauth: refresh returned no access token")
	}

	return &tr, nil
}

// MakeRefresher returns an auth.Refresher that uses RefreshToken.
func MakeRefresher(client *http.Client, tokenURL, clientID, clientSecret string) auth.Refresher {
	return func(ctx context.Context, cred auth.Credential) (auth.Credential, error) {
		tr, err := RefreshToken(ctx, client, tokenURL, clientID, clientSecret, cred.RefreshToken)
		if err != nil {
			return cred, err
		}

		cred.AccessToken = tr.AccessToken
		if tr.RefreshToken != "" {
			cred.RefreshToken = tr.RefreshToken
		}
		expiresIn := tr.ExpiresIn
		if expiresIn <= 0 {
			expiresIn = 3600
		}
		cred.ExpiresAt = time.Now().Add(time.Duration(expiresIn) * time.Second)
		cred.Generation++
		return cred, nil
	}
}
