package augure

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/openai"
	"golang.org/x/sync/singleflight"
)

const (
	ProviderName       = "augure"
	DefaultBaseURL     = "https://api.augureai.ca/v1"
	DefaultSupabaseURL = "https://jsulegwnacbyntfqzfiq.supabase.co/auth/v1/token?grant_type=refresh_token"
	DefaultModel       = "tofino-3"
)

var KnownModels = map[string]bool{
	"tofino-3":      true,
	"ossington-5":   true,
	"ossington-4":   true,
	"ossington-4-1": true,
	"auto":          true,
}

// TokenData represents stored Supabase OAuth tokens.
type TokenData struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresAt    int64  `json:"expires_at"` // Unix epoch in seconds
}

// Config configures the Augure provider.
type Config struct {
	BaseURL     string
	RefreshURL  string
	TokenFile   string
	AnonKeyFile string
	AnonKey     string
	HTTPClient  *http.Client
}

// Provider implements provider.InferenceProvider and provider.AuthProvider.
type Provider struct {
	cfg        Config
	httpClient *http.Client
	tokenFile  string
	anonKey    string
	refreshURL string
	baseURL    string
	sf         singleflight.Group
	mu         sync.RWMutex
}

// New creates a new Augure provider.
func New(cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")

	if cfg.RefreshURL == "" {
		cfg.RefreshURL = DefaultSupabaseURL
	}

	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}

	tokenFile := cfg.TokenFile
	if tokenFile == "" {
		home, err := os.UserHomeDir()
		if err == nil {
			tokenFile = filepath.Join(home, ".augure", "augure-auth.json")
		} else {
			tokenFile = filepath.Join(os.TempDir(), ".augure", "augure-auth.json")
		}
	}

	anonKey := cfg.AnonKey
	if anonKey == "" {
		anonKeyFile := cfg.AnonKeyFile
		if anonKeyFile == "" {
			home, err := os.UserHomeDir()
			if err == nil {
				anonKeyFile = filepath.Join(home, ".augure", "anon.key")
			}
		}
		if anonKeyFile != "" {
			if data, err := os.ReadFile(anonKeyFile); err == nil {
				anonKey = strings.TrimSpace(string(data))
			}
		}
	}

	return &Provider{
		cfg:        cfg,
		httpClient: cfg.HTTPClient,
		tokenFile:  tokenFile,
		anonKey:    anonKey,
		refreshURL: cfg.RefreshURL,
		baseURL:    cfg.BaseURL,
	}, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return ProviderName
}

// Capabilities returns Augure capabilities (vision false, text only).
func Capabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:      true,
		Streaming: true,
		Tools:     true,
		Vision:    false,
	}
}

// Provider returns the provider.Provider bundle.
func (p *Provider) Provider() provider.Provider {
	return provider.Provider{
		Inference:    p,
		Auth:         p,
		Capabilities: Capabilities(),
	}
}

// Login is not interactive in the adapter.
func (p *Provider) Login(ctx context.Context) error {
	return fmt.Errorf("%w: login via CLI or device flow (e.g. augure-cli or populate %s)", provider.ErrNotImplemented, p.tokenFile)
}

// Token returns a valid access token, refreshing if necessary.
func (p *Provider) Token(ctx context.Context) (string, error) {
	tok, err := p.readTokenFile()
	if err != nil {
		return "", err
	}

	// If token expires within 5 minutes or is already expired, refresh
	now := time.Now().Unix()
	if tok.ExpiresAt-now < 300 {
		if err := p.Refresh(ctx); err != nil {
			// If already expired, fail
			if tok.ExpiresAt <= now {
				return "", fmt.Errorf("token expired and refresh failed: %w", err)
			}
		} else {
			// Re-read refreshed token
			tok, err = p.readTokenFile()
			if err != nil {
				return "", err
			}
		}
	}

	return tok.AccessToken, nil
}

// Refresh always re-reads file (for rotating refresh tokens), atomic write, singleflight.
func (p *Provider) Refresh(ctx context.Context) error {
	_, err, _ := p.sf.Do("refresh", func() (any, error) {
		// 1. Always re-read file from disk
		tok, err := p.readTokenFile()
		if err != nil {
			return nil, fmt.Errorf("failed to read token file for refresh: %w", err)
		}

		if tok.RefreshToken == "" {
			return nil, fmt.Errorf("no refresh_token present in %s", p.tokenFile)
		}

		// 2. Call Supabase refresh endpoint
		reqBody, err := json.Marshal(map[string]string{
			"refresh_token": tok.RefreshToken,
		})
		if err != nil {
			return nil, err
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.refreshURL, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		if p.anonKey != "" {
			req.Header.Set("apikey", p.anonKey)
		}

		resp, err := p.httpClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("refresh request failed: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return nil, fmt.Errorf("refresh failed with status %d", resp.StatusCode)
		}

		var result struct {
			AccessToken  string `json:"access_token"`
			RefreshToken string `json:"refresh_token"`
			ExpiresIn    int64  `json:"expires_in"`
			ExpiresAt    int64  `json:"expires_at"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, fmt.Errorf("failed to decode refresh response: %w", err)
		}

		expiresAt := result.ExpiresAt
		if expiresAt == 0 && result.ExpiresIn > 0 {
			expiresAt = time.Now().Unix() + result.ExpiresIn
		}

		newTok := TokenData{
			AccessToken:  result.AccessToken,
			RefreshToken: result.RefreshToken,
			ExpiresAt:    expiresAt,
		}

		// 3. Atomic write to token file
		if err := p.writeTokenFileAtomic(newTok); err != nil {
			return nil, fmt.Errorf("failed to atomically write token file: %w", err)
		}

		return nil, nil
	})

	return err
}

func (p *Provider) readTokenFile() (TokenData, error) {
	data, err := os.ReadFile(p.tokenFile)
	if err != nil {
		return TokenData{}, err
	}

	var raw struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresAt    any    `json:"expires_at"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return TokenData{}, fmt.Errorf("corrupt token file: %w", err)
	}

	var exp int64
	switch v := raw.ExpiresAt.(type) {
	case float64:
		exp = int64(v)
	case int64:
		exp = v
	case string:
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			exp = t.Unix()
		}
	}

	return TokenData{
		AccessToken:  raw.AccessToken,
		RefreshToken: raw.RefreshToken,
		ExpiresAt:    exp,
	}, nil
}

func (p *Provider) writeTokenFileAtomic(tok TokenData) error {
	dir := filepath.Dir(p.tokenFile)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}

	tmpFile, err := os.CreateTemp(dir, "augure-auth-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmpFile.Name()
	defer os.Remove(tmpName)

	data, err := json.MarshalIndent(tok, "", "  ")
	if err != nil {
		tmpFile.Close()
		return err
	}

	if _, err := tmpFile.Write(data); err != nil {
		tmpFile.Close()
		return err
	}
	if err := tmpFile.Sync(); err != nil {
		tmpFile.Close()
		return err
	}
	_ = tmpFile.Chmod(0600)
	if err := tmpFile.Close(); err != nil {
		return err
	}

	return os.Rename(tmpName, p.tokenFile)
}

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	model := reqConfig.Model
	if model == "" {
		model = DefaultModel
	}

	token, err := p.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("auth error: %w", err)
	}

	// Vision is false (text-only)
	chatMsgs := openai.ConvertMessages(msgs, openai.ConvertOptions{AllowVision: false})

	reqBody := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    chatMsgs,
		Stream:      false,
		MaxTokens:   reqConfig.MaxTokens,
		Temperature: reqConfig.Temperature,
		Extra:       reqConfig.ExtraBody,
	}

	bodyReader, err := openai.BuildRequestBody(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bodyReader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	for k, v := range reqConfig.Headers {
		req.Header.Set(k, v)
	}

	return openai.ExecuteGenerate(ctx, p.httpClient, req)
}

// Stream implements provider.InferenceProvider.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	model := reqConfig.Model
	if model == "" {
		model = DefaultModel
	}

	token, err := p.Token(ctx)
	if err != nil {
		return nil, fmt.Errorf("auth error: %w", err)
	}

	chatMsgs := openai.ConvertMessages(msgs, openai.ConvertOptions{AllowVision: false})

	reqBody := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    chatMsgs,
		Stream:      true,
		MaxTokens:   reqConfig.MaxTokens,
		Temperature: reqConfig.Temperature,
		Extra:       reqConfig.ExtraBody,
	}

	bodyReader, err := openai.BuildRequestBody(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bodyReader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	for k, v := range reqConfig.Headers {
		req.Header.Set(k, v)
	}

	return openai.ExecuteStream(ctx, p.httpClient, req)
}
