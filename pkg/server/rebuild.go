package server

import (
	"context"
	"fmt"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// RebuildLane rebuilds one runtime lane from its stored config after a
// credential change (T004: post-login), so "completed" means usable without a
// restart.
//
// It loads the enriched config a restart would restore (Enrich: daemon-owned
// DataDir + Creds, resolved discovery flag, real freebuff actor), builds a
// fresh lane with openaicompat.New (which closes over the injected credential
// store, so the new TokenSource reads the just-stored credential), runs one
// bounded discovery pass and swaps it into the registry.
//
// Ordering and failure semantics:
//   - Registry.Register replaces in place preserving order, so the swap is a
//     single Register on success. The previous lane is preserved on any
//     failure: nothing is unregistered before the new lane is known good.
//   - A lane that is not in the runtime store (login-first ordering,
//     compile-time lane) is NOT an error for the caller to hide: it returns
//     ErrProviderNotStored so the MCP completion branch can report
//     "credential stored; lane not registered — call add_provider".
//   - A custom-wire lane (kind != openaicompat) cannot be rebuilt through this
//     path; it returns an explicit error so the caller reports honestly
//     instead of a false "usable".
//   - Discovery runs synchronously under modelDiscoveryBudget (the same budget
//     registration, startup and the schedule use). A discovery failure
//     preserves the previous lane and is returned as an error.
func (s *RuntimeProviderStore) RebuildLane(registry *provider.Registry, name string) (int, error) {
	if s == nil {
		return 0, fmt.Errorf("runtime provider store not configured")
	}
	if registry == nil {
		return 0, fmt.Errorf("provider registry not configured")
	}
	if name == "" {
		return 0, fmt.Errorf("provider name is required")
	}

	cfg, ok := s.Get(name)
	if !ok {
		s.mu.RLock()
		sp, custom := s.custom[name]
		s.mu.RUnlock()
		if custom {
			return 0, fmt.Errorf("lane %q is a custom-wire lane (kind %q): rebuild not supported via the OpenAI-compatible path", name, sp.Kind)
		}
		return 0, fmt.Errorf("%w: %q", ErrProviderNotStored, name)
	}

	// Single build path (T003): enrich so the rebuilt lane is exactly what a
	// restart would restore, closing over the daemon-owned credential store.
	cfg = s.Enrich(cfg)

	p, err := openaicompat.New(cfg)
	if err != nil {
		return 0, fmt.Errorf("rebuild lane %q: %w (previous lane preserved)", name, err)
	}

	discovered := len(p.CachedModels())
	if discovered == 0 && p.ModelDiscoveryEnabled() {
		// Construction discovery came up empty (the lane was built while its
		// upstream was unreachable, or the fresh credential was not yet
		// stored). Retry once under the standard budget so the reply never
		// understates a lane that just came up; a failure preserves the
		// previous lane.
		fetchCtx, cancel := context.WithTimeout(context.Background(), modelDiscoveryBudget)
		defer cancel()
		models, ferr := p.FetchModels(fetchCtx)
		if ferr != nil {
			return 0, fmt.Errorf("rebuild lane %q: model discovery failed: %w (previous lane preserved)", name, ferr)
		}
		discovered = len(models)
	}

	// Replace in place (Register replaces preserving order). Only reached on
	// success, so a failure never leaves the lane missing.
	registry.Register(p.Provider())
	return discovered, nil
}
