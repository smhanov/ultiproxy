package server

import (
	"context"
	"fmt"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// RebuildLane rebuilds one runtime lane from its stored config after a
// credential change (T004: post-login; T011: custom-wire parity), so
// "completed" means usable without a restart.
//
// OpenAI-compatible lanes load the enriched config a restart would restore
// (Enrich: daemon-owned TokenFile + Creds, resolved discovery flag, real
// freebuff actor), build a fresh lane with openaicompat.New (which closes
// over the injected credential store, so the new TokenSource reads the
// just-stored credential), run one bounded discovery pass and swap it into
// the registry.
//
// Custom-wire lanes (kind != openaicompat, e.g. antigravity) have no
// openaicompat config and no model discovery. "Rebuild" for them means
// re-resolve through the store's LaneBuilder — the same kind-specific
// builder Restore uses (antigravity: NewFromState over the server's general
// DataDir, ProviderBundle) — and re-register. It reports discovered 0;
// models stay addressed as <lane>/<model> (plus the lane's default id).
//
// Ordering and failure semantics:
//   - Registry.Register replaces in place preserving order, so the swap is a
//     single Register on success. The previous lane is preserved on any
//     failure: nothing is unregistered before the new lane is known good.
//   - A lane that is not in the runtime store (login-first ordering,
//     compile-time lane) is NOT an error for the caller to hide: it returns
//     ErrProviderNotStored so the MCP completion branch can report
//     "credential stored; lane not registered — call add_provider".
//   - A custom-wire lane with no LaneBuilder cannot be rebuilt; it returns an
//     explicit "custom-wire lane" error so the caller reports honestly
//     instead of a false "usable". A custom builder failure is a different
//     error ("rebuild lane %q via kind %q builder ... (previous lane
//     preserved)") — it deliberately omits the "custom-wire lane" substring
//     so the MCP completion branch maps it to the failure note, not the
//     honest no-builder note.
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
		builder := s.LaneBuilder
		dataDir := s.DefaultDataDir
		s.mu.RUnlock()
		if custom {
			if builder == nil {
				return 0, fmt.Errorf("lane %q is a custom-wire lane (kind %q): no LaneBuilder to rebuild it", name, sp.Kind)
			}
			// T011: re-resolve through the kind-specific builder (the same
			// constructor Restore uses) without holding the store lock, then
			// swap in place. Only success reaches Register, so a builder
			// failure preserves the previous lane.
			bundle, berr := builder(name, sp.Kind, dataDir, sp.APIKey)
			if berr != nil {
				return 0, fmt.Errorf("rebuild lane %q via kind %q builder: %w (previous lane preserved)", name, sp.Kind, berr)
			}
			registry.RegisterWithSource(bundle, "rebuild")
			return 0, nil
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
	registry.RegisterWithSource(p.Provider(), "rebuild")
	return discovered, nil
}
