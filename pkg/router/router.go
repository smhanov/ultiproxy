package router

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
)

var (
	ErrNoHealthyProvider   = errors.New("router: no healthy provider available for request")
	ErrProviderUnavailable = errors.New("router: requested qualified provider is unavailable")
	ErrCapabilityMismatch  = errors.New("router: candidate lacks required capabilities")
)

// RouteMode determines whether cross-provider routing is permitted.
type RouteMode string

const (
	RouteModeAuto  RouteMode = "auto"
	RouteModeExact RouteMode = "exact"
)

// QuotaSnapshotGetter abstracts reading live quota snapshots.
type QuotaSnapshotGetter interface {
	Get(providerName string) (*provider.QuotaSnapshot, bool)
}

// RouterConfig configures the model router.
type RouterConfig struct {
	SafetyMargin      float64
	Weights           ScoringWeights
	Aliases           map[string][]string // bare alias -> ["provider1/model", "provider2/model"]
	BenchmarkCategory string              // e.g. "coding"
	Rand              *rand.Rand
	UsePowerOfTwo     bool
}

// RouteRequest defines the input to a routing decision.
type RouteRequest struct {
	Model                string
	Messages             []*ir.Message
	RouteMode            RouteMode
	RequiredCapabilities provider.Capabilities
	Options              []provider.Option
	ClientKeyHash        string
	SessionID            string
}

// RouteDecision contains the selected provider, resolved model, and acquired reservation.
type RouteDecision struct {
	Provider        string             `json:"provider"`
	Model           string             `json:"model"`
	Reservation     *Reservation       `json:"-"`
	CandidateScores map[string]float64 `json:"candidate_scores,omitempty"`
}

// Router orchestrates model selection and in-flight reservations.
type Router struct {
	mu           sync.RWMutex
	sm           *state.StateManager
	registry     *provider.Registry
	quotaGetter  QuotaSnapshotGetter
	reservations *ReservationManager
	cfg          RouterConfig
	rand         *rand.Rand
}

// NewRouter constructs a Router.
func NewRouter(sm *state.StateManager, reg *provider.Registry, quotaGetter QuotaSnapshotGetter, cfg RouterConfig) *Router {
	if cfg.SafetyMargin <= 0 {
		cfg.SafetyMargin = 5.0
	}
	if cfg.Weights == (ScoringWeights{}) {
		cfg.Weights = DefaultScoringWeights()
	}
	if cfg.Aliases == nil {
		cfg.Aliases = make(map[string][]string)
	}
	r := cfg.Rand
	if r == nil {
		r = rand.New(rand.NewSource(42))
	}

	return &Router{
		sm:           sm,
		registry:     reg,
		quotaGetter:  quotaGetter,
		reservations: NewReservationManager(),
		cfg:          cfg,
		rand:         r,
	}
}

// Reservations returns the underlying ReservationManager.
func (r *Router) Reservations() *ReservationManager {
	return r.reservations
}

// Route selects the best provider and model, acquiring an in-flight reservation.
func (r *Router) Route(ctx context.Context, req RouteRequest) (*RouteDecision, error) {
	if r.registry == nil {
		return nil, errors.New("router: provider registry is nil")
	}

	snap := r.currentSnapshot()

	// 1. Determine candidate provider/model targets
	candidates, isQualified, err := r.resolveCandidates(req, snap)
	if err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		if isQualified {
			return nil, ErrProviderUnavailable
		}
		return nil, ErrNoHealthyProvider
	}

	// 2. Filter and score candidates
	type candidateEval struct {
		provider          string
		model             string
		meta              CandidateMeta
		observedRemaining float64
		maxConcurrency    int
		score             float64
	}

	var eligible []candidateEval
	scores := make(map[string]float64)

	r.mu.Lock()
	randGen := r.rand
	r.mu.Unlock()

	for _, cand := range candidates {
		candKey := cand.provider + "/" + cand.model

		// Check registry
		p, exists := r.registry.Get(cand.provider)
		if !exists || p.Inference == nil {
			continue
		}

		// Capability matching
		if !matchCapabilities(p.Capabilities, req.RequiredCapabilities) {
			continue
		}

		// Availability checking from RuntimeSnapshot
		pr, hasRuntime := snap.Providers[cand.provider]
		if hasRuntime && !isProviderAvailable(pr) {
			continue
		}

		// Concurrency limit check
		activeSlots := r.reservations.ReservedSlots(cand.provider)
		if p.Capabilities.MaxConcurrentRequests > 0 && activeSlots >= int64(p.Capabilities.MaxConcurrentRequests) {
			continue
		}

		// Headroom check
		observedRem := r.getObservedRemaining(cand.provider)
		effectiveHeadroom := observedRem
		if observedRem != math.MaxFloat64 {
			effectiveHeadroom = r.reservations.EffectiveHeadroom(cand.provider, observedRem, r.cfg.SafetyMargin)
			if effectiveHeadroom < 1.0 {
				continue
			}
		}

		// Model metadata
		var mRuntime *state.ModelRuntime
		var benchScore float64
		if snap.Models != nil {
			if mr, ok := snap.Models[candKey]; ok {
				mRuntime = &mr
				if mr.BenchmarkScores != nil {
					benchScore = mr.BenchmarkScores[r.cfg.BenchmarkCategory]
					if benchScore == 0 {
						for _, score := range mr.BenchmarkScores {
							benchScore = score
							break
						}
					}
				}
			}
		}

		var randomFactor float64
		r.mu.Lock()
		if randGen != nil {
			randomFactor = randGen.Float64()*2 - 1
		}
		r.mu.Unlock()

		meta := CandidateMeta{
			Provider:          cand.provider,
			Model:             cand.model,
			Runtime:           pr,
			ModelRuntime:      mRuntime,
			EffectiveHeadroom: effectiveHeadroom,
			ActiveSlots:       activeSlots,
			BenchmarkScore:    benchScore,
			RandomFactor:      randomFactor,
		}

		score := CalculateScore(meta, r.cfg.Weights)
		scores[candKey] = score

		eligible = append(eligible, candidateEval{
			provider:          cand.provider,
			model:             cand.model,
			meta:              meta,
			observedRemaining: observedRem,
			maxConcurrency:    p.Capabilities.MaxConcurrentRequests,
			score:             score,
		})
	}

	if len(eligible) == 0 {
		if isQualified {
			return nil, ErrProviderUnavailable
		}
		return nil, ErrNoHealthyProvider
	}

	// 3. Selection: pick top or power-of-two choices
	best := eligible[0]
	if len(eligible) > 1 && r.cfg.UsePowerOfTwo {
		r.mu.Lock()
		idx1 := r.rand.Intn(len(eligible))
		idx2 := r.rand.Intn(len(eligible))
		r.mu.Unlock()
		if eligible[idx2].score > eligible[idx1].score {
			best = eligible[idx2]
		} else {
			best = eligible[idx1]
		}
	} else {
		for _, e := range eligible[1:] {
			if e.score > best.score {
				best = e
			}
		}
	}

	// 4. Reserve in-flight resources
	estimatedTokens := EstimateTokens(req.Messages, req.Options...)
	res, err := r.reservations.TryReserve(
		best.provider,
		estimatedTokens,
		1,
		best.observedRemaining,
		r.cfg.SafetyMargin,
		best.maxConcurrency,
	)
	if err != nil {
		return nil, fmt.Errorf("router: reservation failed on %s: %w", best.provider, err)
	}

	return &RouteDecision{
		Provider:        best.provider,
		Model:           best.model,
		Reservation:     res,
		CandidateScores: scores,
	}, nil
}

type providerModelTarget struct {
	provider string
	model    string
}

func (r *Router) resolveCandidates(req RouteRequest, snap *state.RuntimeSnapshot) ([]providerModelTarget, bool, error) {
	// Qualified check: "copilot/claude-sonnet-4-6"
	if strings.Contains(req.Model, "/") {
		parts := strings.SplitN(req.Model, "/", 2)
		target := providerModelTarget{
			provider: parts[0],
			model:    parts[1],
		}
		return []providerModelTarget{target}, true, nil
	}

	// Bare alias: "claude-sonnet-4-6"
	bareModel := req.Model

	// If route_mode == "auto", route across candidates
	if req.RouteMode == RouteModeAuto {
		targets := r.lookupAliasTargets(bareModel, snap)
		return targets, false, nil
	}

	// If route_mode != "auto", caller did NOT opt in to cross-provider migration!
	// Select only the default/primary provider for this alias.
	targets := r.lookupAliasTargets(bareModel, snap)
	if len(targets) > 0 {
		return []providerModelTarget{targets[0]}, false, nil
	}

	return nil, false, nil
}

func (r *Router) lookupAliasTargets(bareModel string, snap *state.RuntimeSnapshot) []providerModelTarget {
	var out []providerModelTarget

	// Check configured router aliases
	if targets, ok := r.cfg.Aliases[bareModel]; ok {
		for _, t := range targets {
			if strings.Contains(t, "/") {
				parts := strings.SplitN(t, "/", 2)
				out = append(out, providerModelTarget{provider: parts[0], model: parts[1]})
			} else {
				out = append(out, providerModelTarget{provider: t, model: bareModel})
			}
		}
		return out
	}

	// Search RuntimeSnapshot.Models
	if snap != nil && snap.Models != nil {
		for _, m := range snap.Models {
			if m.ID == bareModel && m.Enabled {
				out = append(out, providerModelTarget{provider: m.Provider, model: m.ID})
			}
		}
		if len(out) > 0 {
			return out
		}
	}

	// Search Registry
	if r.registry != nil {
		for _, pName := range r.registry.Names() {
			if p, ok := r.registry.Get(pName); ok && p.Inference != nil {
				out = append(out, providerModelTarget{provider: pName, model: bareModel})
			}
		}
	}

	return out
}

func (r *Router) currentSnapshot() *state.RuntimeSnapshot {
	if r.sm != nil {
		return r.sm.Snapshot()
	}
	return &state.RuntimeSnapshot{
		Providers: make(map[string]state.ProviderRuntime),
		Models:    make(map[string]state.ModelRuntime),
	}
}

func (r *Router) getObservedRemaining(providerName string) float64 {
	if r.quotaGetter == nil {
		return math.MaxFloat64
	}
	snap, ok := r.quotaGetter.Get(providerName)
	if !ok || snap == nil || len(snap.Windows) == 0 {
		return math.MaxFloat64
	}

	// Return remaining requests from the most constrained window
	minRemaining := math.MaxFloat64
	hasLimit := false
	for _, w := range snap.Windows {
		if w.Limit > 0 {
			hasLimit = true
			if w.Remaining < minRemaining {
				minRemaining = w.Remaining
			}
		}
	}
	if hasLimit {
		return minRemaining
	}
	return math.MaxFloat64
}

func matchCapabilities(available provider.Capabilities, required provider.Capabilities) bool {
	if required.Vision && !available.Vision {
		return false
	}
	if required.Tools && !available.Tools {
		return false
	}
	if required.Reasoning && !available.Reasoning {
		return false
	}
	if required.Streaming && !available.Streaming {
		return false
	}
	if required.Messages && !available.Messages {
		return false
	}
	return true
}

func isProviderAvailable(pr state.ProviderRuntime) bool {
	return pr.Admin == state.AdminEnabled &&
		pr.Circuit != state.CircuitOpen &&
		pr.Credential == state.CredentialValid &&
		pr.Quota != state.QuotaExhausted &&
		pr.Health != state.HealthUnavailable
}
