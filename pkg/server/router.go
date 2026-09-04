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
	catalog  *ModelCatalog
}

// NewRegistryRouter creates a default registry-backed router. The catalog is
// optional; when set, unknown models are rejected instead of falling back to
// an arbitrary provider (the "10-lane failover walk" bug).
func NewRegistryRouter(registry *provider.Registry, sm *state.StateManager, catalog *ModelCatalog) *RegistryRouter {
	return &RegistryRouter{
		registry: registry,
		sm:       sm,
		catalog:  catalog,
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

	// 3. Catalog mapping (alias -> provider) when the state snapshot missed it.
	if r.catalog != nil {
		if entry, ok := r.catalog.Get(model); ok && entry.Provider != "" {
			if !excluded[entry.Provider] {
				if _, registered := r.registry.Get(entry.Provider); registered {
					return entry.Provider, nil
				}
			}
			return "", fmt.Errorf("provider %q for model %q is unavailable or failed", entry.Provider, model)
		}
	}

	// 4. Direct or prefix match with provider name (e.g. "zai/glm-5.3-flash").
	for _, name := range names {
		if strings.Contains(strings.ToLower(model), strings.ToLower(name)) {
			if !excluded[name] {
				return name, nil
			}
			return "", fmt.Errorf("provider %q for model %q is unavailable or failed", name, model)
		}
	}

	// 5. Unknown model: reject with unknown_model instead of silently
	// routing to the first registered provider (which produced the
	// "all candidate providers failed" 10-lane walk).
	return "", &UnknownModelError{Model: model}
}

// UnknownModelError indicates the requested model has no mapping to any lane.
type UnknownModelError struct {
	Model string
}

func (e *UnknownModelError) Error() string {
	return fmt.Sprintf("unknown model %q: no catalog alias or provider prefix match", e.Model)
}
