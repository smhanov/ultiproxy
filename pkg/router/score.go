package router

import (
	"math"

	"github.com/smhanov/ultiproxy/pkg/state"
)

// ScoringWeights defines the weight multipliers for candidate scoring.
type ScoringWeights struct {
	Headroom    float64
	Health      float64
	Concurrency float64
	QueueDelay  float64
	Recent429s  float64
	CostClass   float64
	Benchmark   float64
	Jitter      float64
}

// DefaultScoringWeights returns balanced default weights.
func DefaultScoringWeights() ScoringWeights {
	return ScoringWeights{
		Headroom:    10.0,
		Health:      1.0,
		Concurrency: 5.0,
		QueueDelay:  2.0,
		Recent429s:  25.0,
		CostClass:   50.0,
		Benchmark:   100.0,
		Jitter:      1.0,
	}
}

// CandidateMeta bundles candidate status for scoring.
type CandidateMeta struct {
	Provider          string
	Model             string
	Runtime           state.ProviderRuntime
	ModelRuntime      *state.ModelRuntime
	EffectiveHeadroom float64
	ActiveSlots       int64
	QueueDepth        int64
	BenchmarkScore    float64
	IsSubscription    bool
	RandomFactor      float64 // [-1.0, 1.0] from seeded rand
}

// CalculateScore computes a composite candidate score according to policy weights.
func CalculateScore(c CandidateMeta, w ScoringWeights) float64 {
	score := 0.0

	// 1. Headroom score: log(1 + effective_headroom)
	if c.EffectiveHeadroom > 0 {
		score += math.Log1p(c.EffectiveHeadroom) * w.Headroom
	}

	// 2. Health score
	switch c.Runtime.Health {
	case state.HealthHealthy:
		score += 100.0 * w.Health
	case state.HealthDegraded:
		score += 20.0 * w.Health
	default:
		// Unavailable candidates are filtered prior to scoring
	}

	// 3. Concurrency penalty: fewer active slots preferred
	score -= float64(c.ActiveSlots) * w.Concurrency

	// 4. Queue delay penalty
	score -= float64(c.QueueDepth) * w.QueueDelay

	// 5. Recent 429s / failures penalty
	score -= float64(c.Runtime.ConsecutiveFailure) * w.Recent429s

	// 6. Cost class: subscription_zero_marginal bonus
	if c.IsSubscription {
		score += w.CostClass
	} else if c.ModelRuntime != nil && c.ModelRuntime.PricingTag == "subscription_zero_marginal" {
		score += w.CostClass
	}

	// 7. Benchmark preference
	score += c.BenchmarkScore * w.Benchmark

	// 8. Random jitter / tie breaking
	score += c.RandomFactor * w.Jitter

	return score
}
