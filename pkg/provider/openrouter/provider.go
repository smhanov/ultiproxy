package openrouter

import (
	"context"
	"net/http"
	"os"
	"strings"

	"github.com/smhanov/llmhub"
	hubopenrouter "github.com/smhanov/llmhub/providers/openrouter"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/hublane"
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
	adapter    *hublane.Adapter
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

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
		cfg.HTTPClient = client
	}

	p := &Provider{
		cfg:        cfg,
		httpClient: client,
		baseURL:    cfg.BaseURL,
		apiKey:     cfg.APIKey,
	}

	var err error
	hubOpts := []llmhub.Option{llmhub.WithHTTPClient(client)}
	if cfg.BaseURL != "" {
		hubOpts = append(hubOpts, llmhub.WithBaseURL(cfg.BaseURL))
	}
	hubProv, herr := hubopenrouter.New(cfg.APIKey, hubOpts...)
	if herr != nil {
		return nil, err
	}

	p.adapter = hublane.New(hubProv, hublane.WithCapabilities(Capabilities()))
	return p, nil
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
	opts = append(opts, provider.WithModel(model))

	return p.adapter.Generate(ctx, msgs, opts...)
}

// Stream implements provider.InferenceProvider.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	model := reqConfig.Model
	if model == "" {
		model = DefaultModel
	}
	opts = append(opts, provider.WithModel(model))

	return p.adapter.Stream(ctx, msgs, opts...)
}
