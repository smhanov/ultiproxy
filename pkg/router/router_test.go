package router

import (
	"context"
	"errors"
	"math/rand"
	"sync"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
)

type mockInferenceProvider struct {
	name string
}

func (m *mockInferenceProvider) Name() string { return m.name }
func (m *mockInferenceProvider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	return &ir.Response{}, nil
}
func (m *mockInferenceProvider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	ch := make(chan ir.Event)
	close(ch)
	return ch, nil
}

type staticQuotaStore struct {
	mu        sync.RWMutex
	snapshots map[string]*provider.QuotaSnapshot
}

func newStaticQuotaStore() *staticQuotaStore {
	return &staticQuotaStore{snapshots: make(map[string]*provider.QuotaSnapshot)}
}

func (s *staticQuotaStore) Get(name string) (*provider.QuotaSnapshot, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	snap, ok := s.snapshots[name]
	return snap, ok
}

func (s *staticQuotaStore) Set(name string, snap *provider.QuotaSnapshot) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshots[name] = snap
}

// 1. Test Weighted Selection Determinism with seeded rand
func TestWeightedSelectionDeterminism(t *testing.T) {
	reg := provider.NewRegistry()
	reg.Register(provider.Provider{
		Inference:    &mockInferenceProvider{name: "provider-a"},
		Capabilities: provider.Capabilities{Vision: true, Tools: true},
	})
	reg.Register(provider.Provider{
		Inference:    &mockInferenceProvider{name: "provider-b"},
		Capabilities: provider.Capabilities{Vision: true, Tools: true},
	})

	sm := state.NewStateManager()
	sm.SetProvider("provider-a", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})
	sm.SetProvider("provider-b", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})

	// Add model benchmark score difference
	sm.Update(func(snap *state.RuntimeSnapshot) {
		snap.Models["provider-a/claude-sonnet-4-6"] = state.ModelRuntime{
			ID:              "claude-sonnet-4-6",
			Provider:        "provider-a",
			Enabled:         true,
			BenchmarkScores: map[string]float64{"coding": 92.5},
		}
		snap.Models["provider-b/claude-sonnet-4-6"] = state.ModelRuntime{
			ID:              "claude-sonnet-4-6",
			Provider:        "provider-b",
			Enabled:         true,
			BenchmarkScores: map[string]float64{"coding": 80.0},
		}
	})

	qStore := newStaticQuotaStore()
	qStore.Set("provider-a", &provider.QuotaSnapshot{
		Windows: []provider.QuotaWindow{{Label: "reqs", UsedPct: 20, Remaining: 80, Limit: 100}},
	})
	qStore.Set("provider-b", &provider.QuotaSnapshot{
		Windows: []provider.QuotaWindow{{Label: "reqs", UsedPct: 10, Remaining: 90, Limit: 100}},
	})

	aliases := map[string][]string{
		"claude-sonnet-4-6": {"provider-a/claude-sonnet-4-6", "provider-b/claude-sonnet-4-6"},
	}

	// Run multiple trials with identical seed -> must yield identical decisions
	runDecisions := func(seed int64) []string {
		r := NewRouter(sm, reg, qStore, RouterConfig{
			SafetyMargin:      5.0,
			Aliases:           aliases,
			BenchmarkCategory: "coding",
			Rand:              rand.New(rand.NewSource(seed)),
		})

		var decisions []string
		for i := 0; i < 5; i++ {
			dec, err := r.Route(context.Background(), RouteRequest{
				Model:     "claude-sonnet-4-6",
				RouteMode: RouteModeAuto,
				Messages: []*ir.Message{
					{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hello world"}}},
				},
			})
			if err != nil {
				t.Fatalf("Route failed: %v", err)
			}
			decisions = append(decisions, dec.Provider)
			dec.Reservation.Release()
		}
		return decisions
	}

	run1 := runDecisions(12345)
	run2 := runDecisions(12345)

	if len(run1) != len(run2) {
		t.Fatalf("run lengths differ: %d vs %d", len(run1), len(run2))
	}
	for i := range run1 {
		if run1[i] != run2[i] {
			t.Fatalf("determinism violated at step %d: %s != %s", i, run1[i], run2[i])
		}
	}
}

// 2. Test Canonical Alias vs Qualified Routing
func TestAliasVsQualifiedRouting(t *testing.T) {
	reg := provider.NewRegistry()
	reg.Register(provider.Provider{
		Inference: &mockInferenceProvider{name: "copilot"},
	})
	reg.Register(provider.Provider{
		Inference: &mockInferenceProvider{name: "antigravity"},
	})

	sm := state.NewStateManager()
	// copilot is healthy initially
	sm.SetProvider("copilot", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})
	// antigravity is healthy
	sm.SetProvider("antigravity", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})

	aliases := map[string][]string{
		"claude-sonnet-4-6": {"copilot/claude-sonnet-4-6", "antigravity/claude-sonnet-4-6"},
	}

	r := NewRouter(sm, reg, nil, RouterConfig{
		Aliases: aliases,
		Rand:    rand.New(rand.NewSource(1)),
	})

	ctx := context.Background()

	// 2a. Bare alias with RouteModeAuto: succeeds and can use copilot or antigravity
	dec, err := r.Route(ctx, RouteRequest{
		Model:     "claude-sonnet-4-6",
		RouteMode: RouteModeAuto,
	})
	if err != nil {
		t.Fatalf("expected RouteModeAuto to succeed: %v", err)
	}
	dec.Reservation.Release()

	// 2b. Now degrade copilot (e.g. QuotaExhausted)
	sm.Update(func(snap *state.RuntimeSnapshot) {
		p := snap.Providers["copilot"]
		p.Quota = state.QuotaExhausted
		snap.Providers["copilot"] = p
	})

	// Bare alias with RouteModeAuto should route to antigravity!
	decAuto, err := r.Route(ctx, RouteRequest{
		Model:     "claude-sonnet-4-6",
		RouteMode: RouteModeAuto,
	})
	if err != nil {
		t.Fatalf("expected RouteModeAuto to route to antigravity: %v", err)
	}
	if decAuto.Provider != "antigravity" {
		t.Errorf("expected auto routing to migrate to antigravity, got %s", decAuto.Provider)
	}
	decAuto.Reservation.Release()

	// 2c. Qualified model: "copilot/claude-sonnet-4-6" MUST NEVER migrate silently!
	_, errQualified := r.Route(ctx, RouteRequest{
		Model:     "copilot/claude-sonnet-4-6",
		RouteMode: RouteModeAuto,
	})
	if errQualified == nil {
		t.Fatalf("expected error for qualified unavailable copilot, but got nil")
	}
	if !errors.Is(errQualified, ErrProviderUnavailable) {
		t.Errorf("expected ErrProviderUnavailable, got %v", errQualified)
	}

	// 2d. Bare alias WITHOUT RouteModeAuto (caller did not opt in): does not cross-route!
	_, errNoAuto := r.Route(ctx, RouteRequest{
		Model:     "claude-sonnet-4-6",
		RouteMode: RouteModeExact, // or empty
	})
	if errNoAuto == nil {
		t.Fatalf("expected bare alias without RouteModeAuto to NOT silently cross-route when primary is exhausted")
	}
}

// 3. Test Reservation Underflow Protection
func TestReservationUnderflowProtection(t *testing.T) {
	rm := NewReservationManager()

	// Attempt to release without reserving
	res := &Reservation{
		provider: "test-p",
		tokens:   1000,
		slots:    2,
		mgr:      rm,
	}

	// Underflow protection: counters should remain 0
	res.Release()
	if rm.ReservedTokens("test-p") != 0 {
		t.Errorf("expected ReservedTokens 0, got %d", rm.ReservedTokens("test-p"))
	}
	if rm.ReservedSlots("test-p") != 0 {
		t.Errorf("expected ReservedSlots 0, got %d", rm.ReservedSlots("test-p"))
	}

	// Try multiple releases (idempotency)
	res.Release()
	res.Release()
	if rm.ReservedTokens("test-p") != 0 || rm.ReservedSlots("test-p") != 0 {
		t.Errorf("expected counters to remain 0 after multiple releases")
	}

	// Reserve 500 tokens, 1 slot
	r1, err := rm.TryReserve("test-p", 500, 1, 100, 5, 10)
	if err != nil {
		t.Fatalf("TryReserve: %v", err)
	}
	if rm.ReservedTokens("test-p") != 500 {
		t.Errorf("expected 500 tokens, got %d", rm.ReservedTokens("test-p"))
	}

	// Commit with more tokens than reserved
	r1.Commit(1000)
	if rm.ReservedTokens("test-p") != 0 {
		t.Errorf("expected 0 tokens after commit, got %d", rm.ReservedTokens("test-p"))
	}
	if rm.ReservedSlots("test-p") != 0 {
		t.Errorf("expected 0 slots after commit, got %d", rm.ReservedSlots("test-p"))
	}
}

// 4. Test Stampede Bounds:
// 100 concurrent goroutines choose among 3 providers — assert no provider exceeds observed headroom by more than the safety margin.
func TestStampedeBounds(t *testing.T) {
	reg := provider.NewRegistry()
	providers := []string{"provider-1", "provider-2", "provider-3"}
	for _, p := range providers {
		reg.Register(provider.Provider{
			Inference: &mockInferenceProvider{name: p},
		})
	}

	sm := state.NewStateManager()
	for _, p := range providers {
		sm.SetProvider(p, state.ProviderRuntime{
			Admin:      state.AdminEnabled,
			Health:     state.HealthHealthy,
			Quota:      state.QuotaHealthy,
			Circuit:    state.CircuitClosed,
			Credential: state.CredentialValid,
		})
	}

	// Set limited quota for each provider:
	// Total headroom = 20 requests each. Safety margin = 5.
	// Maximum allowed reservations = 20 - 5 = 15 requests each!
	observedHeadroom := map[string]float64{
		"provider-1": 20.0,
		"provider-2": 25.0,
		"provider-3": 15.0,
	}

	qStore := newStaticQuotaStore()
	for p, limit := range observedHeadroom {
		qStore.Set(p, &provider.QuotaSnapshot{
			Windows: []provider.QuotaWindow{
				{Label: "reqs", UsedPct: 0, Remaining: limit, Limit: limit, Unit: "requests"},
			},
		})
	}

	safetyMargin := 5.0
	r := NewRouter(sm, reg, qStore, RouterConfig{
		SafetyMargin: safetyMargin,
		Aliases: map[string][]string{
			"model-stampede": {
				"provider-1/model-stampede",
				"provider-2/model-stampede",
				"provider-3/model-stampede",
			},
		},
	})

	const numGoroutines = 100
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	startBarrier := make(chan struct{})
	var activeReservations sync.Map

	for i := 0; i < numGoroutines; i++ {
		reqID := i
		go func() {
			defer wg.Done()
			<-startBarrier // stampede release

			dec, err := r.Route(context.Background(), RouteRequest{
				Model:     "model-stampede",
				RouteMode: RouteModeAuto,
			})
			if err == nil && dec != nil {
				activeReservations.Store(reqID, dec.Reservation)
			}
		}()
	}

	// Release all 100 goroutines simultaneously
	close(startBarrier)
	wg.Wait()

	// Assert no provider exceeds observed headroom by more than safety margin!
	// In fact, active reserved slots must be <= observedHeadroom - safetyMargin.
	for _, p := range providers {
		reservedSlots := r.Reservations().ReservedSlots(p)
		limit := observedHeadroom[p]
		maxAllowed := int64(limit - safetyMargin)

		t.Logf("Provider %s: reserved slots = %d (limit=%.0f, safetyMargin=%.0f, maxAllowed=%d)",
			p, reservedSlots, limit, safetyMargin, maxAllowed)

		if reservedSlots > maxAllowed {
			t.Fatalf("STAMPEDE VIOLATION: provider %s reserved %d slots, exceeding max allowed (%d = limit %.0f - margin %.0f)",
				p, reservedSlots, maxAllowed, limit, safetyMargin)
		}

		// Also assert never exceeded observed headroom (even without safety margin)
		if float64(reservedSlots) > limit {
			t.Fatalf("STAMPEDE VIOLATION: provider %s reserved %d slots, exceeding observed limit %.0f",
				p, reservedSlots, limit)
		}
	}

	// Clean up reservations
	activeReservations.Range(func(key, val any) bool {
		if res, ok := val.(*Reservation); ok {
			res.Release()
		}
		return true
	})

	// After release, all counters must be 0
	for _, p := range providers {
		if r.Reservations().ReservedSlots(p) != 0 {
			t.Errorf("expected 0 reserved slots after cleanup for %s", p)
		}
	}
}
