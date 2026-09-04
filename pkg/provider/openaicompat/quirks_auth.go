package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	llmauth "github.com/smhanov/llmhub/auth"
	"github.com/smhanov/ultiproxy/pkg/auth"
	"golang.org/x/sync/singleflight"
)

const (
	defaultAugureRefreshURL = "https://jsulegwnacbyntfqzfiq.supabase.co/auth/v1/token?grant_type=refresh_token"
	defaultXAIIssuer        = "https://auth.x.ai"
	defaultXAIClientID      = "b1a00492-073a-47ea-816f-4c329264a828"
)

// SupabaseTokenData represents stored Supabase OAuth tokens.
type SupabaseTokenData struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresAt    int64  `json:"expires_at"` // Unix epoch in seconds
}

// SupabaseTokenSource implements llmauth.InvalidatableTokenSource with token file persistence
// and Supabase refresh endpoint calls on expiry or 401 invalidation.
type SupabaseTokenSource struct {
	mu           sync.RWMutex
	httpClient   *http.Client
	refreshURL   string
	tokenFile    string
	anonKey      string
	accessToken  string
	refreshToken string
	expiresAt    time.Time
	invalidated  bool
	sf           singleflight.Group
}

// NewSupabaseTokenSource creates a new SupabaseTokenSource.
func NewSupabaseTokenSource(client *http.Client, refreshURL, tokenFile, initialAccess, initialRefresh string) *SupabaseTokenSource {
	if client == nil {
		client = http.DefaultClient
	}
	if refreshURL == "" {
		refreshURL = defaultAugureRefreshURL
	}
	src := &SupabaseTokenSource{
		httpClient:   client,
		refreshURL:   refreshURL,
		tokenFile:    tokenFile,
		accessToken:  initialAccess,
		refreshToken: initialRefresh,
	}
	if tokenFile != "" {
		_ = src.loadFromFile()
	}
	return src
}

func (s *SupabaseTokenSource) loadFromFile() error {
	if s.tokenFile == "" {
		return nil
	}
	data, err := os.ReadFile(s.tokenFile)
	if err != nil {
		return err
	}
	var td SupabaseTokenData
	if err := json.Unmarshal(data, &td); err != nil {
		return err
	}
	s.mu.Lock()
	if td.AccessToken != "" {
		s.accessToken = td.AccessToken
	}
	if td.RefreshToken != "" {
		s.refreshToken = td.RefreshToken
	}
	if td.ExpiresAt > 0 {
		s.expiresAt = time.Unix(td.ExpiresAt, 0)
	}
	s.mu.Unlock()
	return nil
}

func (s *SupabaseTokenSource) saveToFile(td SupabaseTokenData) error {
	if s.tokenFile == "" {
		return nil
	}
	dir := filepath.Dir(s.tokenFile)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}
	tmpFile, err := os.CreateTemp(dir, "auth-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmpFile.Name()
	defer os.Remove(tmpName)

	data, err := json.MarshalIndent(td, "", "  ")
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
	return os.Rename(tmpName, s.tokenFile)
}

// Token returns the current access token, refreshing if invalidated or expired.
func (s *SupabaseTokenSource) Token(ctx context.Context) (*llmauth.Token, error) {
	s.mu.RLock()
	curAccess := s.accessToken
	curRefresh := s.refreshToken
	curExp := s.expiresAt
	invalid := s.invalidated
	s.mu.RUnlock()

	needsRefresh := invalid || curAccess == ""
	if !curExp.IsZero() && time.Now().After(curExp.Add(-30*time.Second)) {
		needsRefresh = true
	}

	if !needsRefresh && curAccess != "" {
		return &llmauth.Token{
			AccessToken:  curAccess,
			RefreshToken: curRefresh,
			Expiry:       curExp,
		}, nil
	}

	res, err, _ := s.sf.Do("refresh", func() (any, error) {
		// Re-check after acquiring singleflight lock
		s.mu.RLock()
		if !s.invalidated && s.accessToken != "" && (s.expiresAt.IsZero() || time.Now().Before(s.expiresAt.Add(-30*time.Second))) {
			tok := &llmauth.Token{
				AccessToken:  s.accessToken,
				RefreshToken: s.refreshToken,
				Expiry:       s.expiresAt,
			}
			s.mu.RUnlock()
			return tok, nil
		}
		rt := s.refreshToken
		s.mu.RUnlock()

		reqBody, err := json.Marshal(map[string]string{
			"refresh_token": rt,
		})
		if err != nil {
			return nil, err
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.refreshURL, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		if s.anonKey != "" {
			req.Header.Set("apikey", s.anonKey)
		}

		resp, err := s.httpClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("supabase refresh request failed: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return nil, fmt.Errorf("supabase refresh failed with status %d", resp.StatusCode)
		}

		var parsed struct {
			AccessToken  string `json:"access_token"`
			RefreshToken string `json:"refresh_token"`
			ExpiresIn    int64  `json:"expires_in"`
			ExpiresAt    int64  `json:"expires_at"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
			return nil, fmt.Errorf("decode supabase refresh response: %w", err)
		}

		expSec := parsed.ExpiresAt
		if expSec == 0 && parsed.ExpiresIn > 0 {
			expSec = time.Now().Unix() + parsed.ExpiresIn
		}
		expTime := time.Unix(expSec, 0)

		s.mu.Lock()
		s.accessToken = parsed.AccessToken
		if parsed.RefreshToken != "" {
			s.refreshToken = parsed.RefreshToken
		}
		s.expiresAt = expTime
		s.invalidated = false
		s.mu.Unlock()

		_ = s.saveToFile(SupabaseTokenData{
			AccessToken:  parsed.AccessToken,
			RefreshToken: s.refreshToken,
			ExpiresAt:    expSec,
		})

		return &llmauth.Token{
			AccessToken:  parsed.AccessToken,
			RefreshToken: s.refreshToken,
			Expiry:       expTime,
		}, nil
	})

	if err != nil {
		return nil, err
	}
	return res.(*llmauth.Token), nil
}

// Invalidate flags the token as invalid so the next Token call executes a refresh.
func (s *SupabaseTokenSource) Invalidate(accessToken string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if accessToken == "" || s.accessToken == accessToken {
		s.invalidated = true
	}
}

var _ llmauth.InvalidatableTokenSource = (*SupabaseTokenSource)(nil)

// OAuthManagerTokenSource wraps an auth.Manager to implement llmauth.InvalidatableTokenSource.
type OAuthManagerTokenSource struct {
	mgr      *auth.Manager
	clientID string
}

// NewOAuthManagerTokenSource wraps an auth.Manager.
func NewOAuthManagerTokenSource(mgr *auth.Manager, clientID string) *OAuthManagerTokenSource {
	if clientID == "" {
		clientID = defaultXAIClientID
	}
	return &OAuthManagerTokenSource{
		mgr:      mgr,
		clientID: clientID,
	}
}

// Token returns a valid token from auth.Manager.
func (s *OAuthManagerTokenSource) Token(ctx context.Context) (*llmauth.Token, error) {
	if s.mgr == nil {
		return nil, errors.New("auth.Manager is nil")
	}
	cred, err := s.mgr.Get(ctx, s.clientID)
	if err != nil {
		return nil, err
	}
	return &llmauth.Token{
		AccessToken:  cred.AccessToken,
		RefreshToken: cred.RefreshToken,
		Expiry:       cred.ExpiresAt,
	}, nil
}

// Invalidate invalidates cached tokens.
func (s *OAuthManagerTokenSource) Invalidate(accessToken string) {
	// auth.Manager will refresh when Get is called if expired or invalidated
}

var _ llmauth.InvalidatableTokenSource = (*OAuthManagerTokenSource)(nil)
