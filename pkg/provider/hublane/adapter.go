package hublane

import (
	"context"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// AdapterOption configures an Adapter.
type AdapterOption func(*Adapter)

// WithCapabilities sets the provider capabilities advertised by the adapter.
func WithCapabilities(caps provider.Capabilities) AdapterOption {
	return func(a *Adapter) { a.caps = caps }
}

// WithQuotaProvider attaches an optional quota provider to the adapter.
func WithQuotaProvider(qp provider.QuotaProvider) AdapterOption {
	return func(a *Adapter) { a.quota = qp }
}

// WithAuthProvider attaches an optional auth provider to the adapter.
func WithAuthProvider(ap provider.AuthProvider) AdapterOption {
	return func(a *Adapter) { a.auth = ap }
}

// Adapter wraps an llmhub.Provider so it implements provider.InferenceProvider.
// Optional quota and auth providers can be attached for bundle registration.
type Adapter struct {
	hub   llmhub.Provider
	quota provider.QuotaProvider
	auth  provider.AuthProvider
	caps  provider.Capabilities
}

// New creates a new HubLane adapter around the supplied llmhub provider.
func New(hub llmhub.Provider, opts ...AdapterOption) *Adapter {
	a := &Adapter{
		hub:  hub,
		caps: defaultCapabilities(),
	}
	for _, o := range opts {
		if o != nil {
			o(a)
		}
	}
	return a
}

func defaultCapabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:      true,
		Streaming: true,
		Tools:     true,
		Reasoning: true,
		Vision:    true,
	}
}

// Name returns the provider identifier from the underlying llmhub provider.
func (a *Adapter) Name() string {
	if a == nil || a.hub == nil {
		return "hublane"
	}
	return a.hub.Name()
}

// Capabilities returns the adapter's advertised capabilities.
func (a *Adapter) Capabilities() provider.Capabilities {
	if a == nil {
		return provider.Capabilities{}
	}
	return a.caps
}

// Generate implements provider.InferenceProvider by forwarding the request to
// the underlying llmhub provider and converting the response back to IR.
func (a *Adapter) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	if a == nil || a.hub == nil {
		return nil, provider.ErrProviderNotFound
	}

	cfg := provider.NewRequestConfig(opts...)
	if cfg.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.Timeout)
		defer cancel()
	}

	prompt := IRToHubPrompt(msgs)
	hubOpts := a.hubOpts(cfg)

	resp, err := a.hub.Generate(ctx, prompt, hubOpts...)
	if err != nil {
		return nil, err
	}
	return HubResponseToIR(resp, cfg.Model), nil
}

// Stream implements provider.InferenceProvider by forwarding the request to the
// underlying llmhub provider and bridging the chunk stream to IR events.
func (a *Adapter) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	if a == nil || a.hub == nil {
		return nil, provider.ErrProviderNotFound
	}

	cfg := provider.NewRequestConfig(opts...)
	prompt := IRToHubPrompt(msgs)
	hubOpts := a.hubOpts(cfg)

	hubChan, err := a.hub.Stream(ctx, prompt, hubOpts...)
	if err != nil {
		return nil, err
	}
	return StreamBridge(ctx, hubChan), nil
}

// hubOpts maps the normalized request config into llmhub functional options.
func (a *Adapter) hubOpts(cfg *provider.RequestConfig) []llmhub.Option {
	var opts []llmhub.Option
	if cfg.Model != "" {
		opts = append(opts, llmhub.WithModel(cfg.Model))
	}
	if cfg.MaxTokens > 0 {
		opts = append(opts, llmhub.WithMaxTokens(cfg.MaxTokens))
	}
	if cfg.Temperature != nil {
		opts = append(opts, llmhub.WithTemperature(*cfg.Temperature))
	}
	if tools := ToolsFromExtra(cfg); len(tools) > 0 {
		opts = append(opts, llmhub.WithTools(tools...))
	}
	for k, v := range cfg.Headers {
		opts = append(opts, llmhub.WithHeader(k, v))
	}
	return opts
}

// Quota returns the optional quota provider attached to the adapter.
func (a *Adapter) Quota() provider.QuotaProvider {
	if a == nil {
		return nil
	}
	return a.quota
}

// Auth returns the optional auth provider attached to the adapter.
func (a *Adapter) Auth() provider.AuthProvider {
	if a == nil {
		return nil
	}
	return a.auth
}

// ProviderBundle returns a provider.Provider bundle including any optional
// quota/auth providers that were attached via AdapterOption.
func (a *Adapter) ProviderBundle() provider.Provider {
	if a == nil {
		return provider.Provider{}
	}
	return provider.Provider{
		Inference:    a,
		Quota:        a.quota,
		Auth:         a.auth,
		Capabilities: a.caps,
	}
}

var _ provider.InferenceProvider = (*Adapter)(nil)
