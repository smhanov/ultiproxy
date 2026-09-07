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
//
// The request path accepts exactly two id shapes:
//   - a user-defined alias from the alias catalog (an explicit name the
//     operator mapped to a lane + upstream), resolved by the catalog; and
//   - a canonical "<lane>/<model>" id, resolved by the "<lane>/" prefix.
//
// Everything else is unknown_model. A bare lane name is a routing prefix, not
// a model, so an exact lane-name match routes nothing. A bare id that is not a
// registered user alias (the old "implicit catalog alias" form) is likewise
// unknown_model. A disabled state row is a terminal kill switch checked before
// either resolution path.
func (r *RegistryRouter) Route(ctx context.Context, model string) (string, error) {
	excluded := ExcludedProvidersFromContext(ctx)

	// 1. Kill switch: a disabled state row is terminal. toggle_model(false) is
	// a kill switch, not a discovery hint: refuse here instead of falling
	// through to the alias catalog or a lane prefix, either of which would
	// happily route the very model the operator just disabled. A model is
	// disabled when any of its related state rows is disabled: the id as
	// given, plus, for a user alias, its bare name and canonical <lane>/<model>
	// form both. So disabling either name disables the model by either name.
	if r.sm != nil {
		if snap := r.sm.Snapshot(); snap != nil && snap.Models != nil {
			if modelIsDisabled(snap.Models, model, r.catalog) {
				return "", &DisabledModelError{Model: model}
			}
		}
	}

	// 2. Check registry
	if r.registry == nil || r.registry.Len() == 0 {
		return "", errors.New("no providers registered")
	}

	names := r.registry.Names()

	// 3. User-defined alias (catalog): an explicit name mapped to a lane.
	// Aliases resolve before prefix routing and are the only bare ids that
	// route; everything else bare is unknown_model.
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

	// 4. "<lane>/" prefix match (e.g. "zai/glm-5.3-flash"). The match is a
	// genuine "<lane>/" prefix - never a substring and never the bare lane id
	// itself: a model id that merely embeds a lane name ("amazai-gpt-4o")
	// belongs to no lane, and a bare lane name is a routing prefix, not a
	// model, so both fall through to unknown_model.
	lowerModel := strings.ToLower(model)
	for _, name := range names {
		if !strings.HasPrefix(lowerModel, strings.ToLower(name)+"/") {
			continue
		}
		if !excluded[name] {
			return name, nil
		}
		return "", fmt.Errorf("provider %q for model %q is unavailable or failed", name, model)
	}

	// 5. Unknown model: reject with unknown_model instead of silently
	// routing to the first registered provider (which produced the
	// "all candidate providers failed" 10-lane walk).
	return "", &UnknownModelError{Model: model}
}

// disabledStateRow reports whether the state snapshot carries a row for id
// (or its canonical form) with Enabled=false.
func disabledStateRow(models map[string]state.ModelRuntime, id string) bool {
	mr, ok := models[id]
	return ok && !mr.Enabled
}

// UnknownModelError indicates the requested model has no mapping to any lane.
type UnknownModelError struct {
	Model string
}

func (e *UnknownModelError) Error() string {
	return fmt.Sprintf("unknown model %q: no catalog alias or provider prefix match", e.Model)
}

// DisabledModelError indicates the model is known but currently disabled
// (toggle_model enabled=false, or an alias disabled at runtime). It unwraps to
// UnknownModelError so the HTTP surface answers with the same 404
// unknown_model contract clients already handle for ids that map to no lane.
type DisabledModelError struct {
	Model string
}

func (e *DisabledModelError) Error() string {
	return fmt.Sprintf("model %q is disabled (toggle_model enabled=false); refusing to route", e.Model)
}

// Unwrap makes DisabledModelError classify as unknown_model on the wire.
func (e *DisabledModelError) Unwrap() error {
	return &UnknownModelError{Model: e.Model}
}
