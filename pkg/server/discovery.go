package server

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

// Automatic model discovery.
//
// /v1/models must be correct without any manual step, so every lane that can
// discover its upstream catalog does so:
//
//   - at registration: openaicompat.New discovers while the lane is built
//     (quirks.model_list_passthrough defaults to true), and the MCP
//     add_provider success path re-runs discovery synchronously so its reply
//     can state what the lane serves;
//   - at startup: RuntimeProviderStore.Restore re-runs discovery for restored
//     lanes whose cache is still empty (a lane built while its upstream was
//     unreachable must not stay empty until someone calls refresh_models), and
//     the server's startup pass heals whatever is still empty once Restore's
//     pool has settled;
//   - on a schedule: a ticker (DefaultModelRefreshInterval, configurable with
//     WithModelRefreshInterval, 0 disables) re-runs discovery for ALL
//     discovery lanes, so upstream catalog changes surface without a restart.
//
// None of this touches the request path: the aggregated /v1/models handler
// keeps reading CachedModels() only, so listing never fans out to an upstream
// and a mid-refresh list serves the previous cache until the swap.
//
// Lanes whose wire is not an OpenAI model list (antigravity, anthropic, codex,
// copilot, freebuff) stay non-discoverable: they do not implement FetchModels
// (or explicitly opt out), so no ids are invented for them.

const (
	// DefaultModelRefreshInterval is how often every discovery lane re-fetches
	// its upstream model list.
	DefaultModelRefreshInterval = 6 * time.Hour

	// modelDiscoveryBudget bounds a single FetchModels call. It is the budget
	// openaicompat.New, Restore's pool, the startup pass and the schedule all
	// use, so one slow upstream can never stall a lane's registration or a
	// whole refresh round.
	modelDiscoveryBudget = 5 * time.Second

	// modelDiscoveryConcurrency bounds the discovery goroutine pool: a slow
	// upstream delays at most this many other lanes, and startup is never
	// blocked (the pool runs in the background).
	modelDiscoveryConcurrency = 2
)

// modelsFetcher is the optional lane capability that fetches and caches the
// upstream model list. It is asserted off bundle.Inference so this package
// stays independent of the concrete adapter packages.
type modelsFetcher interface {
	FetchModels(ctx context.Context) ([]string, error)
}

// modelDiscoveryOptOuter is implemented by lanes that can opt out of automatic
// discovery (openaicompat: quirks.model_list_passthrough resolves to false).
// Lanes that do not implement it are discovered whenever they can fetch a list.
type modelDiscoveryOptOuter interface {
	ModelDiscoveryEnabled() bool
}

// discoverLane runs ONE lane's model discovery under a per-call budget and logs
// the count. Lanes without a FetchModels surface, and lanes that opted out of
// discovery, are skipped (0, nil): nothing is invented for them.
func discoverLane(ctx context.Context, name string, bundle provider.Provider, budget time.Duration, reason string) (int, error) {
	if bundle.Inference == nil {
		return 0, nil
	}
	fetcher, ok := bundle.Inference.(modelsFetcher)
	if !ok {
		return 0, nil
	}
	if opt, ok := bundle.Inference.(modelDiscoveryOptOuter); ok && !opt.ModelDiscoveryEnabled() {
		log.Printf("[discovery] model discovery lane=%s skipped (%s): disabled for this lane", name, reason)
		return 0, nil
	}
	if budget <= 0 {
		budget = modelDiscoveryBudget
	}
	fetchCtx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	models, err := fetcher.FetchModels(fetchCtx)
	if err != nil {
		log.Printf("[discovery] model discovery lane=%s failed (%s): %v", name, reason, err)
		return 0, err
	}
	log.Printf("[discovery] model discovery lane=%s count=%d (%s)", name, len(models), reason)
	return len(models), nil
}

// discoveryTarget is one lane selected for a discovery pass.
type discoveryTarget struct {
	name   string
	bundle provider.Provider
}

// discoveryTargetsFor picks the lanes a discovery pass should cover. Lanes
// without a FetchModels surface are never targets. When onlyEmpty is set, lanes
// that already hold a cached model list are skipped too: that is the heal pass
// (registration/startup), which must not re-dial lanes construction already
// populated.
func discoveryTargetsFor(registry *provider.Registry, names []string, onlyEmpty bool) []discoveryTarget {
	if registry == nil {
		return nil
	}
	targets := make([]discoveryTarget, 0, len(names))
	for _, name := range names {
		bundle, ok := registry.Get(name)
		if !ok || bundle.Inference == nil {
			continue
		}
		if _, ok := bundle.Inference.(modelsFetcher); !ok {
			continue
		}
		if opt, ok := bundle.Inference.(modelDiscoveryOptOuter); ok && !opt.ModelDiscoveryEnabled() {
			continue
		}
		if onlyEmpty {
			if cacher, ok := bundle.Inference.(modelsCacheProvider); ok && len(cacher.CachedModels()) > 0 {
				continue
			}
		}
		targets = append(targets, discoveryTarget{name: name, bundle: bundle})
	}
	return targets
}

// discoveryTargets snapshots the registry's discovery lanes for a pass.
func (s *Server) discoveryTargets(onlyEmpty bool) []discoveryTarget {
	if s == nil || s.registry == nil {
		return nil
	}
	return discoveryTargetsFor(s.registry, s.registry.Names(), onlyEmpty)
}

// discoverLanes runs a discovery pass through a bounded goroutine pool and
// returns when every lane of the pass has finished.
func discoverLanes(ctx context.Context, targets []discoveryTarget, reason string) {
	if len(targets) == 0 {
		return
	}
	sem := make(chan struct{}, modelDiscoveryConcurrency)
	var wg sync.WaitGroup
	for _, target := range targets {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			_, _ = discoverLane(ctx, target.name, target.bundle, modelDiscoveryBudget, reason)
		}()
	}
	wg.Wait()
}

// modelTicker abstracts *time.Ticker so the refresh schedule is observable (and
// drivable) from tests with a fake clock instead of sleeping for hours.
type modelTicker interface {
	C() <-chan time.Time
	Stop()
}

type realModelTicker struct{ ticker *time.Ticker }

func (r realModelTicker) C() <-chan time.Time { return r.ticker.C }
func (r realModelTicker) Stop()               { r.ticker.Stop() }

// withModelTickerFactory is the (unexported, test-only) hook that replaces the
// real ticker with a fake clock.
func withModelTickerFactory(factory func(d time.Duration) modelTicker) Option {
	return func(s *Server) {
		s.newModelTicker = factory
	}
}

// startModelDiscovery launches the background discovery loop:
//
//  1. startup heal pass - after Restore's own pool has settled, re-run
//     discovery for every lane whose cache is still empty;
//  2. schedule - re-run discovery for ALL discovery lanes on every tick.
//
// The loop owns its context; Shutdown cancels it.
func (s *Server) startModelDiscovery() {
	if s.registry == nil {
		return
	}
	interval := s.modelRefreshInterval
	if interval < 0 {
		interval = DefaultModelRefreshInterval
	}

	ctx, cancel := context.WithCancel(context.Background())
	s.discoveryCancel = cancel

	go func() {
		// Let Restore's own pool finish first, so the heal pass does not race
		// it into duplicate upstream calls.
		if s.providers != nil {
			s.providers.WaitForRestoreDiscovery()
		}
		discoverLanes(ctx, s.discoveryTargets(true), "startup")

		if interval <= 0 {
			return
		}
		newTicker := s.newModelTicker
		if newTicker == nil {
			newTicker = func(d time.Duration) modelTicker { return realModelTicker{ticker: time.NewTicker(d)} }
		}
		ticker := newTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C():
				discoverLanes(ctx, s.discoveryTargets(false), "scheduled")
			}
		}
	}()
}
