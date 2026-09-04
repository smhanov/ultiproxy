// Package anthropichub adapts llmhub's Anthropic provider to the ultiproxy
// provider registry via the shared hublane bridge.
package anthropichub

import (
	"context"
	"net/http"

	"github.com/smhanov/llmhub"
	hubanthropic "github.com/smhanov/llmhub/providers/anthropic"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/hublane"
)

const (
	ProviderName   = "anthropic"
	DefaultBaseURL = "https://api.anthropic.com"
)

// Config configures the Anthropic (first-party API) lane.
type Config struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

// Provider implements provider.InferenceProvider, backed by llmhub.
type Provider struct {
	adapter *hublane.Adapter
}

// New creates the Anthropic lane via llmhub + hublane. Returns an error when
// the API key is empty (registration is opt-in upstream).
func New(cfg Config) (*Provider, error) {
	opts := []llmhub.Option{}
	if cfg.BaseURL != "" {
		opts = append(opts, llmhub.WithBaseURL(cfg.BaseURL))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, llmhub.WithHTTPClient(cfg.HTTPClient))
	}
	hubProv, err := hubanthropic.New(cfg.APIKey, opts...)
	if err != nil {
		return nil, err
	}
	caps := provider.Capabilities{
		Chat:      true,
		Reasoning: true,
		Vision:    true,
		Tools:     true,
		Streaming: true,
	}
	adapter := hublane.New(hubProv, hublane.WithCapabilities(caps))
	return &Provider{adapter: adapter}, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string { return ProviderName }

// Generate forwards to the hublane adapter.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	return p.adapter.Generate(ctx, msgs, opts...)
}

// Stream forwards to the hublane adapter.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	return p.adapter.Stream(ctx, msgs, opts...)
}

// Provider returns the registry bundle.
func (p *Provider) Provider() provider.Provider {
	return p.adapter.ProviderBundle()
}
