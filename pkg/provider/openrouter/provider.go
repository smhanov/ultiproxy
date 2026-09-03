package openrouter

import (
	"context"
	"net/http"
	"os"
	"strings"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/openai"
)

const (
	ProviderName   = "openrouter"
	DefaultBaseURL = "https://openrouter.ai/api/v1"
	DefaultModel   = "openai/gpt-4o"
)

// Config configures the OpenRouter provider.
type Config struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

// Provider implements provider.InferenceProvider.
type Provider struct {
	cfg        Config
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

// New creates a new OpenRouter aggregator provider.
func New(cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")

	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("OPENROUTER_API_KEY")
	}

	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}

	return &Provider{
		cfg:        cfg,
		httpClient: cfg.HTTPClient,
		baseURL:    cfg.BaseURL,
		apiKey:     cfg.APIKey,
	}, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return ProviderName
}

// Capabilities returns OpenRouter capabilities.
func Capabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:      true,
		Streaming: true,
		Reasoning: true,
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

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	model := reqConfig.Model
	if model == "" {
		model = DefaultModel
	}

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
	model := reqConfig.Model
	if model == "" {
		model = DefaultModel
	}

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
