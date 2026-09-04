// Package provider defines the provider contract interfaces that all
// upstream adapters (Copilot, Antigravity, Codex, xAI, Freebuff, Z.ai,
// Augure, DeepSeek, Anthropic, OpenCode Go, OpenRouter, vLLM, ...) must
// implement, plus a thread-safe registry of registered providers.
//
// This is the "contract freeze" boundary: worktrees implement adapters
// against these interfaces, and the core (server/router/quota) consumes
// them. Adapters are not allowed to change these signatures.
package provider

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
)

// ErrProviderNotFound is returned by Registry.Get when no provider with the
// requested name is registered.
var ErrProviderNotFound = errors.New("provider: not found")

// ErrNotImplemented is returned by adapters for optional interface methods
// they do not support.
var ErrNotImplemented = errors.New("provider: not implemented")

// QuotaWindow is one quota/rate-limit pool reported by a QuotaProvider.
type QuotaWindow struct {
	Label            string    `json:"label"`
	UsedPct          float64   `json:"used_pct"` // 0-100
	Remaining        float64   `json:"remaining,omitempty"`
	Limit            float64   `json:"limit,omitempty"`
	Unit             string    `json:"unit,omitempty"` // %, credits, requests
	ResetAt          time.Time `json:"reset_at,omitempty"`
	SecondsRemaining int64     `json:"seconds_remaining,omitempty"`
}

// QuotaSnapshot is the normalized quota view returned by a QuotaProvider.
type QuotaSnapshot struct {
	ObservedAt time.Time     `json:"observed_at"`
	Windows    []QuotaWindow `json:"windows"`
	Detail     string        `json:"detail,omitempty"`
}

// InferenceProvider generates and streams responses from an upstream model
// using the protocol-neutral IR. Stream MUST return an error synchronously
// when the upstream replies with a non-2xx status (so failover can happen
// before downstream headers are committed); mid-stream errors are delivered
// as ir.EventUpstreamError events on the channel followed by stream close.
type InferenceProvider interface {
	Name() string
	Generate(ctx context.Context, msgs []*ir.Message, opts ...Option) (*ir.Response, error)
	Stream(ctx context.Context, msgs []*ir.Message, opts ...Option) (<-chan ir.Event, error)
}

// QuotaProvider reports live quota/rate-limit headroom for an upstream.
type QuotaProvider interface {
	Name() string
	Quota(ctx context.Context) (*QuotaSnapshot, error)
}

// AuthProvider manages an upstream subscription credential: OAuth device
// login, token access, and refresh. Implementations own token persistence
// and must be safe for concurrent use (singleflight refresh).
type AuthProvider interface {
	Name() string
	Login(ctx context.Context) error
	Token(ctx context.Context) (string, error)
	Refresh(ctx context.Context) error
}

// Capabilities describes what an InferenceProvider supports, used by the
// router for capability negotiation before dispatch.
type Capabilities struct {
	Chat          bool `json:"chat"`
	Messages      bool `json:"messages"` // Anthropic Messages surface
	Reasoning     bool `json:"reasoning"`
	Vision        bool `json:"vision"`
	Tools         bool `json:"tools"`
	Streaming     bool `json:"streaming"`
	PromptCaching bool `json:"prompt_caching"`
	// MaxConcurrentRequests <= 0 means unlimited. Freebuff uses 1.
	MaxConcurrentRequests int  `json:"max_concurrent_requests,omitempty"`
	SessionAffinity       bool `json:"session_affinity,omitempty"`
	Queueing              bool `json:"queueing,omitempty"`
}

// Provider bundles the optional capabilities every adapter can implement.
// An adapter implementing only Name() is invalid; it must implement at
// least one of InferenceProvider / QuotaProvider / AuthProvider.
type Provider struct {
	Inference    InferenceProvider
	Quota        QuotaProvider
	Auth         AuthProvider
	Capabilities Capabilities
}

// Registry is a thread-safe collection of registered providers.
type Registry struct {
	mu        sync.RWMutex
	providers map[string]Provider
	order     []string
}

// NewRegistry creates an empty provider registry.
func NewRegistry() *Registry {
	return &Registry{providers: make(map[string]Provider)}
}

// Register adds or replaces a provider under its Name().
func (r *Registry) Register(p Provider) {
	name := ""
	if p.Inference != nil {
		name = p.Inference.Name()
	} else if p.Quota != nil {
		name = p.Quota.Name()
	} else if p.Auth != nil {
		name = p.Auth.Name()
	}
	if name == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.providers[name]; !exists {
		r.order = append(r.order, name)
	}
	r.providers[name] = p
}

// Get looks up a provider by name.
func (r *Registry) Get(name string) (Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.providers[name]
	return p, ok
}

// Names returns all registered provider names in registration order.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, len(r.order))
	copy(out, r.order)
	return out
}

// Len returns the number of registered providers.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.providers)
}

// Unregister removes a provider from the registry by lane name, keeping the
// registration order slice consistent. It reports whether the name was
// registered. Runtime removal (MCP remove_provider) uses this so a lane stops
// being routed immediately, in the same process lifetime.
func (r *Registry) Unregister(name string) bool {
	if name == "" {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.providers[name]; !ok {
		return false
	}
	delete(r.providers, name)
	for i, n := range r.order {
		if n == name {
			r.order = append(r.order[:i], r.order[i+1:]...)
			break
		}
	}
	return true
}
