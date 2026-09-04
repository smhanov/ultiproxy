package server

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
)

// Router routes a requested model to a registered provider bundle name.
type Router interface {
	Route(ctx context.Context, model string) (string, error)
}

type excludedProvidersKey struct{}

// ContextWithExcludedProviders injects excluded provider names for failover before commit.
func ContextWithExcludedProviders(ctx context.Context, excluded map[string]bool) context.Context {
	return context.WithValue(ctx, excludedProvidersKey{}, excluded)
}

// ExcludedProvidersFromContext returns excluded provider names.
func ExcludedProvidersFromContext(ctx context.Context) map[string]bool {
	if m, ok := ctx.Value(excludedProvidersKey{}).(map[string]bool); ok {
		return m
	}
	return nil
}

// RegistryRouter routes models based on provider registry and state snapshot.
type RegistryRouter struct {
	registry *provider.Registry
	sm       *state.StateManager
}

// NewRegistryRouter creates a default registry-backed router.
func NewRegistryRouter(registry *provider.Registry, sm *state.StateManager) *RegistryRouter {
	return &RegistryRouter{
		registry: registry,
		sm:       sm,
	}
}

// Route resolves a model name to an available provider bundle name.
func (r *RegistryRouter) Route(ctx context.Context, model string) (string, error) {
	excluded := ExcludedProvidersFromContext(ctx)

	// 1. Check state snapshot for explicit model mapping
	if r.sm != nil {
		snap := r.sm.Snapshot()
		if snap != nil && snap.Models != nil {
			if mr, ok := snap.Models[model]; ok && mr.Enabled {
				if !excluded[mr.Provider] {
					if pr, ok := snap.Providers[mr.Provider]; ok {
						if pr.IsAvailable() {
							return mr.Provider, nil
						}
					} else {
						return mr.Provider, nil
					}
				}
				return "", fmt.Errorf("provider %q for model %q is unavailable or failed", mr.Provider, model)
			}
		}
	}

	// 2. Check registry
	if r.registry == nil || r.registry.Len() == 0 {
		return "", errors.New("no providers registered")
	}

	names := r.registry.Names()

	// Direct or prefix match with provider name
	for _, name := range names {
		if strings.Contains(strings.ToLower(model), strings.ToLower(name)) {
			if !excluded[name] {
				return name, nil
			}
			return "", fmt.Errorf("provider %q for model %q is unavailable or failed", name, model)
		}
	}

	// Fallback to first non-excluded provider (for generic unmapped models)
	for _, name := range names {
		if !excluded[name] {
			return name, nil
		}
	}

	return "", errors.New("no available provider for model")
}
