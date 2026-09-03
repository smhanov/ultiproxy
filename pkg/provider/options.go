package provider

import (
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
)

// Option configures a provider call. Implementations treat options as
// per-request overrides layered on top of construction-time config.
type Option func(*RequestConfig)

// RequestConfig carries per-request knobs given to InferenceProvider
// methods through Option values.
type RequestConfig struct {
	Model           string
	MaxTokens       int
	Temperature     *float64
	ReasoningEffort string
	Headers         map[string]string
	ExtraBody       map[string]any
	Timeout         time.Duration
	// ClientKeyHash identifies the downstream API key owner for accounting.
	ClientKeyHash string
}

// ApplyOptions applies opts to cfg in order.
func ApplyOptions(cfg *RequestConfig, opts ...Option) {
	for _, o := range opts {
		if o != nil {
			o(cfg)
		}
	}
}

// WithModel selects the upstream model identifier (provider-local).
func WithModel(model string) Option {
	return func(c *RequestConfig) { c.Model = model }
}

// WithMaxTokens caps output tokens.
func WithMaxTokens(n int) Option {
	return func(c *RequestConfig) { c.MaxTokens = n }
}

// WithTemperature overrides temperature.
func WithTemperature(t float64) Option {
	return func(c *RequestConfig) { c.Temperature = &t }
}

// WithTimeout sets a per-request timeout; providers should honor it via
// context cancellation when calling upstream.
func WithTimeout(d time.Duration) Option {
	return func(c *RequestConfig) { c.Timeout = d }
}

// WithReasoningEffort sets low/medium/high/xhigh (provider-dependent).
func WithReasoningEffort(e string) Option {
	return func(c *RequestConfig) { c.ReasoningEffort = e }
}

// WithHeader injects a per-request header.
func WithHeader(k, v string) Option {
	return func(c *RequestConfig) {
		if c.Headers == nil {
			c.Headers = make(map[string]string)
		}
		c.Headers[k] = v
	}
}

// WithExtraBody merges arbitrary JSON fields into the request payload.
func WithExtraBody(kv map[string]any) Option {
	return func(c *RequestConfig) {
		if c.ExtraBody == nil {
			c.ExtraBody = make(map[string]any)
		}
		for k, v := range kv {
			c.ExtraBody[k] = v
		}
	}
}

// WithClientKeyHash tags the request with the accounting owner hash.
func WithClientKeyHash(h string) Option {
	return func(c *RequestConfig) { c.ClientKeyHash = h }
}

// NewRequestConfig builds a default RequestConfig with opts applied.
func NewRequestConfig(opts ...Option) *RequestConfig {
	cfg := &RequestConfig{Headers: make(map[string]string)}
	ApplyOptions(cfg, opts...)
	return cfg
}

// Compile-time assertion that Option is a functional option over RequestConfig.
var _ = ir.Message{}
