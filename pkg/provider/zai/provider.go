package zai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/openai"
)

const (
	ProviderName    = "zai"
	DefaultBaseURL  = "https://api.z.ai/api/coding/paas/v4"
	DefaultQuotaURL = "https://api.z.ai/api/monitor/usage/quota/limit"
	DefaultModel    = "glm-5.3-flash"

	MaxOutputTokensStandard = 131072
	MaxOutputTokensAir      = 98304 // 98k for 4.5-air
)

var KnownModels = map[string]bool{
	"glm-5.3-flash": true,
	"glm-5.3":       true,
	"glm-5.2":       true,
	"glm-5.1":       true,
	"glm-5-turbo":   true,
	"glm-4.7":       true,
	"glm-4.5-air":   true,
}

// Config configures the Z.ai provider.
type Config struct {
	BaseURL    string
	QuotaURL   string
	APIKey     string
	HTTPClient *http.Client
}

// Provider implements provider.InferenceProvider and provider.QuotaProvider.
type Provider struct {
	cfg        Config
	httpClient *http.Client
	baseURL    string
	quotaURL   string
	apiKey     string
}

// New creates a new Z.ai provider directly against the coding PAAS endpoint.
func New(cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")

	if cfg.QuotaURL == "" {
		cfg.QuotaURL = DefaultQuotaURL
	}

	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("ZAI_API_KEY")
	}

	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}

	return &Provider{
		cfg:        cfg,
		httpClient: cfg.HTTPClient,
		baseURL:    cfg.BaseURL,
		quotaURL:   cfg.QuotaURL,
		apiKey:     cfg.APIKey,
	}, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return ProviderName
}

// Capabilities returns Z.ai capabilities.
func Capabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:      true,
		Streaming: true,
		Reasoning: true,
		Vision:    true,
		Tools:     true,
	}
}

// Provider returns the provider.Provider bundle.
func (p *Provider) Provider() provider.Provider {
	return provider.Provider{
		Inference:    p,
		Quota:        p,
		Capabilities: Capabilities(),
	}
}

func (p *Provider) resolveMaxTokens(model string, requested int) int {
	if requested > 0 {
		return requested
	}
	if strings.Contains(model, "4.5-air") {
		return MaxOutputTokensAir
	}
	return MaxOutputTokensStandard
}

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	model := reqConfig.Model
	if model == "" {
		model = DefaultModel
	}

	maxTokens := p.resolveMaxTokens(model, reqConfig.MaxTokens)

	// Vision is true (glm-5.3-flash etc. support image_url parts)
	chatMsgs := openai.ConvertMessages(msgs, openai.ConvertOptions{
		AllowVision:   true,
		EchoReasoning: true,
	})

	reqBody := openai.ChatCompletionRequest{
		Model:           model,
		Messages:        chatMsgs,
		Stream:          false,
		MaxTokens:       maxTokens,
		Temperature:     reqConfig.Temperature,
		ReasoningEffort: reqConfig.ReasoningEffort,
		Extra:           reqConfig.ExtraBody,
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
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
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

	maxTokens := p.resolveMaxTokens(model, reqConfig.MaxTokens)

	chatMsgs := openai.ConvertMessages(msgs, openai.ConvertOptions{
		AllowVision:   true,
		EchoReasoning: true,
	})

	reqBody := openai.ChatCompletionRequest{
		Model:           model,
		Messages:        chatMsgs,
		Stream:          true,
		MaxTokens:       maxTokens,
		Temperature:     reqConfig.Temperature,
		ReasoningEffort: reqConfig.ReasoningEffort,
		Extra:           reqConfig.ExtraBody,
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
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	for k, v := range reqConfig.Headers {
		req.Header.Set(k, v)
	}

	return openai.ExecuteStream(ctx, p.httpClient, req)
}

// Quota implements provider.QuotaProvider.
func (p *Provider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.quotaURL, nil)
	if err != nil {
		return nil, err
	}
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("quota request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("quota limit request returned status %d", resp.StatusCode)
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return ParseQuotaLimits(bodyBytes, time.Now())
}

type quotaLimitItem struct {
	Type       string  `json:"type"`
	Label      string  `json:"label"`
	Name       string  `json:"name"`
	Unit       string  `json:"unit"`
	Used       float64 `json:"used"`
	Limit      float64 `json:"limit"`
	Remaining  float64 `json:"remaining"`
	Percentage float64 `json:"percentage"`
	ResetAt    string  `json:"reset_at"`
}

// ParseQuotaLimits parses Z.ai quota limit response into QuotaSnapshot.
func ParseQuotaLimits(data []byte, now time.Time) (*provider.QuotaSnapshot, error) {
	var top struct {
		Code int `json:"code"`
		Data struct {
			Limits []quotaLimitItem `json:"limits"`
		} `json:"data"`
		Limits []quotaLimitItem `json:"limits"`
	}

	if err := json.Unmarshal(data, &top); err != nil {
		return nil, fmt.Errorf("failed to parse Z.ai quota json: %w", err)
	}

	items := top.Data.Limits
	if len(items) == 0 {
		items = top.Limits
	}

	var windows []provider.QuotaWindow

	for _, item := range items {
		name := item.Type
		if name == "" {
			name = item.Label
		}
		if name == "" {
			name = item.Name
		}

		unit := item.Unit
		if unit == "" {
			unit = "credits"
		}

		label := name
		lower := strings.ToLower(name)
		if strings.Contains(lower, "5-hour") || strings.Contains(lower, "5h") || strings.Contains(lower, "five_hour") {
			label = "5-hour"
		} else if strings.Contains(lower, "week") {
			label = "Weekly"
		}

		usedPct := item.Percentage
		if usedPct == 0 && item.Limit > 0 {
			usedPct = (item.Used / item.Limit) * 100.0
		}

		rem := item.Remaining
		if rem == 0 && item.Limit > 0 && item.Used > 0 {
			rem = item.Limit - item.Used
		}

		var resetAt time.Time
		var secRem int64
		if item.ResetAt != "" {
			if t, err := time.Parse(time.RFC3339, item.ResetAt); err == nil {
				resetAt = t
				secRem = int64(t.Sub(now).Seconds())
				if secRem < 0 {
					secRem = 0
				}
			}
		}

		windows = append(windows, provider.QuotaWindow{
			Label:            label,
			UsedPct:          usedPct,
			Remaining:        rem,
			Limit:            item.Limit,
			Unit:             unit,
			ResetAt:          resetAt,
			SecondsRemaining: secRem,
		})
	}

	return &provider.QuotaSnapshot{
		ObservedAt: now,
		Windows:    windows,
	}, nil
}
