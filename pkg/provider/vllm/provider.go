package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/openai"
)

const (
	ProviderName   = "vllm"
	DefaultBaseURL = "http://localhost:8000/v1"
	DefaultModel   = "default"
)

// Config configures the vLLM provider.
type Config struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

// Provider implements provider.InferenceProvider for vLLM and generic OpenAI-compatible backends.
type Provider struct {
	cfg        Config
	httpClient *http.Client
	baseURL    string
	apiKey     string

	mu     sync.RWMutex
	models []string
}

// New creates a new vLLM provider and attempts to load models from /v1/models.
func New(cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")

	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("VLLM_API_KEY")
	}

	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}

	p := &Provider{
		cfg:        cfg,
		httpClient: cfg.HTTPClient,
		baseURL:    cfg.BaseURL,
		apiKey:     cfg.APIKey,
	}

	// Fetch models with short timeout; non-fatal on startup if server is not yet live
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, _ = p.FetchModels(ctx)

	return p, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return ProviderName
}

// Models returns the cached list of models discovered from /v1/models.
func (p *Provider) Models() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]string, len(p.models))
	copy(out, p.models)
	return out
}

// FetchModels queries GET /v1/models (or /models if baseURL ends in /v1) and updates cached models.
func (p *Provider) FetchModels(ctx context.Context) ([]string, error) {
	endpoint := p.baseURL + "/models"
	if !strings.HasSuffix(p.baseURL, "/v1") {
		endpoint = p.baseURL + "/v1/models"
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("models endpoint returned status %d", resp.StatusCode)
	}

	var raw struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}

	var models []string
	for _, m := range raw.Data {
		if m.ID != "" {
			models = append(models, m.ID)
		}
	}

	p.mu.Lock()
	p.models = models
	p.mu.Unlock()

	return models, nil
}

// Capabilities returns vLLM capabilities.
func Capabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:      true,
		Streaming: true,
		Tools:     true,
		Vision:    true,
	}
}

// Provider returns the provider.Provider bundle.
func (p *Provider) Provider() provider.Provider {
	return provider.Provider{
		Inference:    p,
		Capabilities: Capabilities(),
	}
}

func (p *Provider) resolveModel(requested string) string {
	if requested != "" {
		return requested
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.models) > 0 {
		return p.models[0]
	}
	return DefaultModel
}

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	model := p.resolveModel(reqConfig.Model)

	chatMsgs := openai.ConvertMessages(msgs, openai.ConvertOptions{
		AllowVision:   true,
		EchoReasoning: true,
	})

	reqBody := openai.ChatCompletionRequest{
		Model:           model,
		Messages:        chatMsgs,
		Stream:          false,
		MaxTokens:       reqConfig.MaxTokens,
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
	model := p.resolveModel(reqConfig.Model)

	chatMsgs := openai.ConvertMessages(msgs, openai.ConvertOptions{
		AllowVision:   true,
		EchoReasoning: true,
	})

	reqBody := openai.ChatCompletionRequest{
		Model:           model,
		Messages:        chatMsgs,
		Stream:          true,
		MaxTokens:       reqConfig.MaxTokens,
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
