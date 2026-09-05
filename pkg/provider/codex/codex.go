package codex

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/oauth"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/sse"
)

const (
	DefaultBaseURL   = "https://chatgpt.com"
	DefaultClientID  = "app_EMoamEEZ73f0CkXaXp7hrann"
	DefaultDeviceURL = "https://auth.openai.com/device"
	DefaultTokenURL  = "https://auth.openai.com/oauth/token"
)

// SupportedModels is the fixture list of models surfaced by the OpenAI Codex backend.
var SupportedModels = []string{
	"gpt-5.5",
	"gpt-5.6-luna",
	"gpt-5.4",
	"gpt-5.4-mini",
	"gpt-5.6-sol",
	"gpt-5.6-terra",
}

// Config configures the OpenAI Codex backend adapter.
type Config struct {
	BaseURL       string
	ClientID      string
	ClientSecret  string
	DeviceAuthURL string
	TokenURL      string
	AuthManager   *auth.Manager
	Refresher     auth.Refresher
	StaticToken   string
	HTTPClient    *http.Client
}

// Provider implements InferenceProvider, QuotaProvider, and AuthProvider for Codex.
type Provider struct {
	baseURL       string
	clientID      string
	clientSecret  string
	deviceAuthURL string
	tokenURL      string
	authManager   *auth.Manager
	refresher     auth.Refresher
	staticToken   string
	httpClient    *http.Client

	mu        sync.RWMutex
	liveToken string
}

// New creates a new OpenAI Codex provider adapter.
func New(cfg Config) *Provider {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")

	clientID := cfg.ClientID
	if clientID == "" {
		clientID = DefaultClientID
	}

	deviceURL := cfg.DeviceAuthURL
	if deviceURL == "" {
		deviceURL = DefaultDeviceURL
	}

	tokenURL := cfg.TokenURL
	if tokenURL == "" {
		tokenURL = DefaultTokenURL
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	if cfg.Refresher == nil && cfg.AuthManager != nil {
		cfg.Refresher = oauth.MakeRefresher(client, tokenURL, clientID, cfg.ClientSecret)
	}
	if cfg.AuthManager != nil && cfg.Refresher != nil {
		cfg.AuthManager.SetRefresher(cfg.Refresher)
	}

	return &Provider{
		baseURL:       baseURL,
		clientID:      clientID,
		clientSecret:  cfg.ClientSecret,
		deviceAuthURL: deviceURL,
		tokenURL:      tokenURL,
		authManager:   cfg.AuthManager,
		refresher:     cfg.Refresher,
		staticToken:   cfg.StaticToken,
		httpClient:    client,
	}
}

// Name implements InferenceProvider, QuotaProvider, and AuthProvider.
func (p *Provider) Name() string {
	return "codex"
}

// Models returns the list of models surfaced from the fixture list.
func (p *Provider) Models() []string {
	out := make([]string, len(SupportedModels))
	copy(out, SupportedModels)
	return out
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:      true,
		Tools:     true,
		Reasoning: true,
		Streaming: true,
		Vision:    true,
	}
}

// ProviderBundle returns a provider.Provider bundle.
func (p *Provider) ProviderBundle() provider.Provider {
	return provider.Provider{
		Inference:    p,
		Quota:        p,
		Auth:         p,
		Capabilities: p.Capabilities(),
	}
}

// Register registers this provider in a registry.
func (p *Provider) Register(r *provider.Registry) {
	r.Register(p.ProviderBundle())
}

// -----------------------------------------------------------------------------
// Request Helper with 401 Refresh-Once
// -----------------------------------------------------------------------------

func (p *Provider) getToken(ctx context.Context) (string, error) {
	if p.authManager != nil {
		cred, err := p.authManager.Get(ctx, p.clientID)
		if err == nil && cred.AccessToken != "" {
			p.mu.Lock()
			p.liveToken = cred.AccessToken
			p.mu.Unlock()
			return cred.AccessToken, nil
		}
	}
	p.mu.RLock()
	if p.liveToken != "" {
		t := p.liveToken
		p.mu.RUnlock()
		return t, nil
	}
	p.mu.RUnlock()
	if p.staticToken != "" {
		return p.staticToken, nil
	}
	return "", errors.New("codex: no access token available")
}

// doRequestWithRefreshOnce executes an HTTP request, refreshing token on 401 exactly once.
func (p *Provider) doRequestWithRefreshOnce(ctx context.Context, makeReq func(tok string) (*http.Request, error)) (*http.Response, error) {
	tok, err := p.getToken(ctx)
	if err != nil {
		return nil, err
	}

	req, err := makeReq(tok)
	if err != nil {
		return nil, err
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusUnauthorized {
		return resp, nil
	}

	// 401 Unauthorized -> Refresh token exactly once
	resp.Body.Close()

	if err := p.Refresh(ctx); err != nil {
		return nil, fmt.Errorf("codex: 401 received and refresh failed: %w", err)
	}

	newTok, err := p.getToken(ctx)
	if err != nil {
		return nil, err
	}

	retryReq, err := makeReq(newTok)
	if err != nil {
		return nil, err
	}

	return p.httpClient.Do(retryReq)
}

// -----------------------------------------------------------------------------
// Payload & Model Translation
// -----------------------------------------------------------------------------

// BuildPayload constructs the JSON payload for OpenAI Codex completions,
// translating max_tokens -> max_completion_tokens for newer models (gpt-5*).
func BuildPayload(msgs []*ir.Message, cfg *provider.RequestConfig, stream bool) (map[string]any, error) {
	payload := make(map[string]any)
	payload["model"] = cfg.Model
	payload["stream"] = stream

	if strings.HasPrefix(cfg.Model, "gpt-5") {
		if cfg.MaxTokens > 0 {
			payload["max_completion_tokens"] = cfg.MaxTokens
		}
	} else {
		if cfg.MaxTokens > 0 {
			payload["max_tokens"] = cfg.MaxTokens
		}
	}

	if cfg.Temperature != nil {
		payload["temperature"] = *cfg.Temperature
	}

	var chatMsgs []map[string]any
	for _, m := range msgs {
		if m == nil {
			continue
		}
		var textParts []string
		var toolCalls []map[string]any
		var toolCallID string

		for _, blk := range m.Blocks {
			if blk == nil {
				continue
			}
			switch b := blk.(type) {
			case ir.TextBlock:
				textParts = append(textParts, b.Text)
			case *ir.TextBlock:
				textParts = append(textParts, b.Text)
			case ir.ToolCallBlock:
				toolCalls = append(toolCalls, map[string]any{
					"id":   b.ID,
					"type": "function",
					"function": map[string]any{
						"name":      b.Name,
						"arguments": b.Arguments,
					},
				})
			case *ir.ToolCallBlock:
				toolCalls = append(toolCalls, map[string]any{
					"id":   b.ID,
					"type": "function",
					"function": map[string]any{
						"name":      b.Name,
						"arguments": b.Arguments,
					},
				})
			case ir.ToolResultBlock:
				toolCallID = b.ToolCallID
				textParts = append(textParts, b.Content)
			case *ir.ToolResultBlock:
				toolCallID = b.ToolCallID
				textParts = append(textParts, b.Content)
			}
		}

		entry := map[string]any{
			"role":    m.Role,
			"content": strings.Join(textParts, "\n"),
		}
		if len(toolCalls) > 0 {
			entry["tool_calls"] = toolCalls
		}
		if toolCallID != "" {
			entry["role"] = "tool"
			entry["tool_call_id"] = toolCallID
		}
		chatMsgs = append(chatMsgs, entry)
	}

	payload["messages"] = chatMsgs

	for k, v := range cfg.ExtraBody {
		payload[k] = v
	}

	return payload, nil
}

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	cfg := provider.NewRequestConfig(opts...)

	payloadMap, err := BuildPayload(msgs, cfg, false)
	if err != nil {
		return nil, fmt.Errorf("codex: build payload failed: %w", err)
	}

	bodyBytes, err := json.Marshal(payloadMap)
	if err != nil {
		return nil, fmt.Errorf("codex: failed to marshal payload: %w", err)
	}

	endpoint := p.baseURL + "/backend-api/codex/completions"
	makeReq := func(tok string) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+tok)
		req.Header.Set("Content-Type", "application/json")
		for k, v := range cfg.Headers {
			req.Header.Set(k, v)
		}
		return req, nil
	}

	resp, err := p.doRequestWithRefreshOnce(ctx, makeReq)
	if err != nil {
		return nil, fmt.Errorf("codex: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("codex: failed to read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("codex: completions returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	var chatResp struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("codex: failed to parse completions response: %w", err)
	}

	irResp := &ir.Response{
		ID:         chatResp.ID,
		UpstreamID: chatResp.ID,
	}

	if len(chatResp.Choices) > 0 {
		choice := chatResp.Choices[0]
		irResp.FinishReason = choice.FinishReason

		var blocks []ir.Block
		if choice.Message.Content != "" {
			blocks = append(blocks, ir.TextBlock{Text: choice.Message.Content})
		}
		for _, tc := range choice.Message.ToolCalls {
			blocks = append(blocks, ir.ToolCallBlock{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
		irResp.Message = &ir.Message{
			Role:   choice.Message.Role,
			Blocks: blocks,
		}
	}

	if chatResp.Usage != nil {
		irResp.Usage = &ir.Usage{
			PromptTokens:     chatResp.Usage.PromptTokens,
			CompletionTokens: chatResp.Usage.CompletionTokens,
			TotalTokens:      chatResp.Usage.TotalTokens,
		}
	}

	return irResp, nil
}

// Stream implements provider.InferenceProvider.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	cfg := provider.NewRequestConfig(opts...)

	payloadMap, err := BuildPayload(msgs, cfg, true)
	if err != nil {
		return nil, fmt.Errorf("codex: build payload failed: %w", err)
	}

	bodyBytes, err := json.Marshal(payloadMap)
	if err != nil {
		return nil, fmt.Errorf("codex: failed to marshal payload: %w", err)
	}

	endpoint := p.baseURL + "/backend-api/codex/completions"
	makeReq := func(tok string) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+tok)
		req.Header.Set("Content-Type", "application/json")
		for k, v := range cfg.Headers {
			req.Header.Set(k, v)
		}
		return req, nil
	}

	resp, err := p.doRequestWithRefreshOnce(ctx, makeReq)
	if err != nil {
		return nil, fmt.Errorf("codex: stream request failed: %w", err)
	}

	// Synchronous check: return error synchronously on non-2xx
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("codex: upstream completions returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	outCh := make(chan ir.Event, 64)

	go func() {
		defer close(outCh)
		defer resp.Body.Close()

		scanner := sse.NewScanner(resp.Body)
		started := false
		var responseID string

		type streamChunk struct {
			ID      string `json:"id"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}

		for scanner.Scan() {
			ev := scanner.Event()
			data := bytes.TrimSpace(ev.Data)
			if len(data) == 0 {
				continue
			}
			if string(data) == "[DONE]" {
				break
			}

			var chunk streamChunk
			if err := json.Unmarshal(data, &chunk); err != nil {
				continue
			}

			if !started && chunk.ID != "" {
				responseID = chunk.ID
				outCh <- ir.EventMessageStart{ID: chunk.ID}
				started = true
			}

			if chunk.Usage != nil {
				outCh <- ir.EventUsageUpdate{
					PromptTokens:     chunk.Usage.PromptTokens,
					CompletionTokens: chunk.Usage.CompletionTokens,
					TotalTokens:      chunk.Usage.TotalTokens,
				}
			}

			for _, c := range chunk.Choices {
				if c.Delta.ReasoningContent != "" {
					outCh <- ir.EventReasoningDelta{Index: c.Index, Text: c.Delta.ReasoningContent}
				}
				if c.Delta.Content != "" {
					outCh <- ir.EventTextDelta{Index: c.Index, Text: c.Delta.Content}
				}
				if c.FinishReason != nil && *c.FinishReason != "" {
					outCh <- ir.EventMessageStop{FinishReason: *c.FinishReason, UpstreamID: responseID}
				}
			}
		}

		if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
			outCh <- ir.EventUpstreamError{
				Kind:      "stream_error",
				Message:   err.Error(),
				Permanent: false,
			}
		}
	}()

	return outCh, nil
}

// -----------------------------------------------------------------------------
// QuotaProvider (wham/usage)
// -----------------------------------------------------------------------------

type whamUsageResponse struct {
	PlanType  string `json:"plan_type"`
	RateLimit struct {
		Allowed       bool `json:"allowed"`
		LimitReached  bool `json:"limit_reached"`
		PrimaryWindow struct {
			UsedPercent        float64 `json:"used_percent"`
			LimitWindowSeconds int64   `json:"limit_window_seconds"`
			ResetAfterSeconds  int64   `json:"reset_after_seconds"`
			ResetAt            int64   `json:"reset_at"`
		} `json:"primary_window"`
		SecondaryWindow struct {
			UsedPercent        float64 `json:"used_percent"`
			LimitWindowSeconds int64   `json:"limit_window_seconds"`
			ResetAfterSeconds  int64   `json:"reset_after_seconds"`
			ResetAt            int64   `json:"reset_at"`
		} `json:"secondary_window"`
	} `json:"rate_limit"`
	Credits struct {
		HasCredits bool   `json:"has_credits"`
		Balance    string `json:"balance"`
	} `json:"credits"`
}

// Quota implements provider.QuotaProvider.
//
// When the lane has no access token at all (never logged in: no credential
// store entry, no static token) there is nothing to query, so the result is
// reported as an observation with empty windows instead of an error — an
// error here surfaces to MCP get_quota_status as "error getting quota:
// codex: wham usage request failed: codex: no access token available".
func (p *Provider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	if _, err := p.getToken(ctx); err != nil {
		return &provider.QuotaSnapshot{
			ObservedAt: time.Now().UTC(),
			Windows:    []provider.QuotaWindow{},
			Detail:     "not logged in — start a login with the MCP initiate_oauth_login tool (provider \"codex\")",
		}, nil
	}

	endpoint := p.baseURL + "/backend-api/wham/usage"
	makeReq := func(tok string) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+tok)
		return req, nil
	}

	resp, err := p.doRequestWithRefreshOnce(ctx, makeReq)
	if err != nil {
		return nil, fmt.Errorf("codex: wham usage request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("codex: failed to read wham response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("codex: wham usage endpoint returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	return ParseWhamUsageJSON(body)
}

// ParseWhamUsageJSON parses the /backend-api/wham/usage JSON into QuotaSnapshot.
func ParseWhamUsageJSON(data []byte) (*provider.QuotaSnapshot, error) {
	var resp whamUsageResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("codex: failed to parse wham usage json: %w", err)
	}

	now := time.Now().UTC()
	var windows []provider.QuotaWindow

	// Primary window ("5 hour")
	pw := resp.RateLimit.PrimaryWindow
	pwReset := time.Unix(pw.ResetAt, 0).UTC()
	pwRemaining := math.Max(0, 100.0-pw.UsedPercent)
	windows = append(windows, provider.QuotaWindow{
		Label:            "5 hour",
		UsedPct:          pw.UsedPercent,
		Remaining:        pwRemaining,
		Limit:            100,
		Unit:             "%",
		ResetAt:          pwReset,
		SecondsRemaining: pw.ResetAfterSeconds,
	})

	// Secondary window ("Weekly")
	sw := resp.RateLimit.SecondaryWindow
	swReset := time.Unix(sw.ResetAt, 0).UTC()
	swRemaining := math.Max(0, 100.0-sw.UsedPercent)
	windows = append(windows, provider.QuotaWindow{
		Label:            "Weekly",
		UsedPct:          sw.UsedPercent,
		Remaining:        swRemaining,
		Limit:            100,
		Unit:             "%",
		ResetAt:          swReset,
		SecondsRemaining: sw.ResetAfterSeconds,
	})

	detail := fmt.Sprintf("Plan: %s · 5 hour %.1f%% used · Weekly %.1f%% used",
		resp.PlanType, pw.UsedPercent, sw.UsedPercent)

	return &provider.QuotaSnapshot{
		ObservedAt: now,
		Windows:    windows,
		Detail:     detail,
	}, nil
}

// -----------------------------------------------------------------------------
// AuthProvider
// -----------------------------------------------------------------------------

func (p *Provider) Login(ctx context.Context) error {
	cfg := oauth.DeviceFlowConfig{
		ClientID:      p.clientID,
		ClientSecret:  p.clientSecret,
		DeviceAuthURL: p.deviceAuthURL,
		TokenURL:      p.tokenURL,
		HTTPClient:    p.httpClient,
	}

	dcr, err := oauth.RequestDeviceCode(ctx, cfg)
	if err != nil {
		return fmt.Errorf("codex: login device code failed: %w", err)
	}

	tokResp, err := oauth.PollToken(ctx, cfg, dcr.DeviceCode, dcr.Interval)
	if err != nil {
		return fmt.Errorf("codex: login token poll failed: %w", err)
	}

	p.mu.Lock()
	p.liveToken = tokResp.AccessToken
	p.mu.Unlock()

	if p.authManager != nil {
		expiresIn := tokResp.ExpiresIn
		if expiresIn <= 0 {
			expiresIn = 3600
		}
		cred := auth.Credential{
			Provider:     "codex",
			AccessToken:  tokResp.AccessToken,
			RefreshToken: tokResp.RefreshToken,
			ExpiresAt:    time.Now().Add(time.Duration(expiresIn) * time.Second),
			ClientID:     p.clientID,
		}
		_ = p.authManager.Store(ctx, p.clientID, cred)
	}

	return nil
}

func (p *Provider) Token(ctx context.Context) (string, error) {
	return p.getToken(ctx)
}

func (p *Provider) Refresh(ctx context.Context) error {
	var cred auth.Credential
	if p.authManager != nil {
		c, err := p.authManager.LoadFromDisk(p.clientID)
		if err == nil {
			cred = c
		}
	}

	ref := p.refresher
	if ref == nil {
		ref = oauth.MakeRefresher(p.httpClient, p.tokenURL, p.clientID, p.clientSecret)
	}

	newCred, err := ref(ctx, cred)
	if err != nil {
		return err
	}

	p.mu.Lock()
	p.liveToken = newCred.AccessToken
	p.mu.Unlock()

	if p.authManager != nil {
		_ = p.authManager.Store(ctx, p.clientID, newCred)
	}
	return nil
}
