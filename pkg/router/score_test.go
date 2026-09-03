package router

import (
	"context"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
)

func TestRouterCapabilityFiltering(t *testing.T) {
	reg := provider.NewRegistry()
	reg.Register(provider.Provider{
		Inference:    &mockInferenceProvider{name: "text-only"},
		Capabilities: provider.Capabilities{Vision: false, Tools: false},
	})
	reg.Register(provider.Provider{
		Inference:    &mockInferenceProvider{name: "multi-modal"},
		Capabilities: provider.Capabilities{Vision: true, Tools: true},
	})

	sm := state.NewStateManager()
	sm.SetProvider("text-only", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})
	sm.SetProvider("multi-modal", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})

	aliases := map[string][]string{
		"omni-model": {"text-only/omni-model", "multi-modal/omni-model"},
	}

	r := NewRouter(sm, reg, nil, RouterConfig{
		Aliases: aliases,
	})

	ctx := context.Background()

	// Request requiring Vision: should filter out text-only and route to multi-modal
	dec, err := r.Route(ctx, RouteRequest{
		Model:     "omni-model",
		RouteMode: RouteModeAuto,
		RequiredCapabilities: provider.Capabilities{
			Vision: true,
		},
	})
	if err != nil {
		t.Fatalf("Route failed: %v", err)
	}
	if dec.Provider != "multi-modal" {
		t.Errorf("expected multi-modal, got %s", dec.Provider)
	}
	dec.Reservation.Release()
}

func TestScoreCostClassAndBenchmark(t *testing.T) {
	weights := DefaultScoringWeights()

	subCand := CandidateMeta{
		Provider:          "sub-provider",
		EffectiveHeadroom: 50,
		IsSubscription:    true,
		BenchmarkScore:    85.0,
		Runtime: state.ProviderRuntime{
			Health: state.HealthHealthy,
		},
	}

	paygCand := CandidateMeta{
		Provider:          "payg-provider",
		EffectiveHeadroom: 50,
		IsSubscription:    false,
		BenchmarkScore:    85.0,
		Runtime: state.ProviderRuntime{
			Health: state.HealthHealthy,
		},
	}

	subScore := CalculateScore(subCand, weights)
	paygScore := CalculateScore(paygCand, weights)

	if subScore <= paygScore {
		t.Errorf("expected subscription candidate score (%.2f) to be higher than payg (%.2f)", subScore, paygScore)
	}
}
