package state

import "time"

// 5-Dimension Provider State Types

type AdminState string

const (
	AdminEnabled  AdminState = "enabled"
	AdminDisabled AdminState = "disabled"
)

type HealthState string

const (
	HealthHealthy     HealthState = "healthy"
	HealthDegraded    HealthState = "degraded"
	HealthUnavailable HealthState = "unavailable"
)

type QuotaState string

const (
	QuotaHealthy   QuotaState = "healthy"
	QuotaLow       QuotaState = "low"
	QuotaExhausted QuotaState = "exhausted"
	QuotaUnknown   QuotaState = "unknown"
)

type CircuitState string

const (
	CircuitClosed   CircuitState = "closed"
	CircuitOpen     CircuitState = "open"
	CircuitHalfOpen CircuitState = "half_open"
)

type CredentialState string

const (
	CredentialValid      CredentialState = "valid"
	CredentialRefreshing CredentialState = "refreshing"
	CredentialExpired    CredentialState = "expired"
	CredentialRevoked    CredentialState = "revoked"
)

// ProviderRuntime represents the runtime status across all 5 dimensions.
type ProviderRuntime struct {
	Admin       AdminState      `json:"admin"`
	Health      HealthState     `json:"health"`
	Quota       QuotaState      `json:"quota"`
	Circuit     CircuitState    `json:"circuit"`
	Credential  CredentialState `json:"credential"`
	LastAttempt time.Time       `json:"last_attempt"`
	LastSuccess time.Time       `json:"last_success"`
	ObservedAt  time.Time       `json:"observed_at"`
	ValidUntil  time.Time       `json:"valid_until"`
	Source      string          `json:"source"`
	Error       string          `json:"error,omitempty"`

	// Internal circuit breaker bookkeeping
	CircuitOpenedAt    time.Time `json:"circuit_opened_at,omitempty"`
	ProbeInFlight      bool      `json:"probe_in_flight,omitempty"`
	ConsecutiveSuccess int       `json:"consecutive_success,omitempty"`
	ConsecutiveFailure int       `json:"consecutive_failure,omitempty"`
}

// IsAvailable returns true if the provider can serve traffic.
func (p ProviderRuntime) IsAvailable() bool {
	return p.Admin == AdminEnabled &&
		p.Circuit != CircuitOpen &&
		p.Credential == CredentialValid &&
		p.Quota != QuotaExhausted &&
		p.Health != HealthUnavailable
}

// ModelRuntime represents metadata and runtime availability of a model.
type ModelRuntime struct {
	ID              string             `json:"id"`
	Provider        string             `json:"provider"`
	Enabled         bool               `json:"enabled"`
	ContextLimit    int                `json:"context_limit"`
	MaxOutput       int                `json:"max_output"`
	BenchmarkScores map[string]float64 `json:"benchmark_scores,omitempty"`
	PricingTag      string             `json:"pricing_tag,omitempty"`
}

// RuntimeSnapshot holds a point-in-time immutable snapshot of all providers and models.
type RuntimeSnapshot struct {
	Version   uint64                     `json:"version"`
	Providers map[string]ProviderRuntime `json:"providers"`
	Models    map[string]ModelRuntime    `json:"models"`
}

// Clone returns a deep copy of the RuntimeSnapshot.
func (s *RuntimeSnapshot) Clone() *RuntimeSnapshot {
	if s == nil {
		return &RuntimeSnapshot{
			Providers: make(map[string]ProviderRuntime),
			Models:    make(map[string]ModelRuntime),
		}
	}

	cloned := &RuntimeSnapshot{
		Version:   s.Version,
		Providers: make(map[string]ProviderRuntime, len(s.Providers)),
		Models:    make(map[string]ModelRuntime, len(s.Models)),
	}

	for k, v := range s.Providers {
		cloned.Providers[k] = v
	}

	for k, v := range s.Models {
		m := v
		if v.BenchmarkScores != nil {
			scores := make(map[string]float64, len(v.BenchmarkScores))
			for sk, sv := range v.BenchmarkScores {
				scores[sk] = sv
			}
			m.BenchmarkScores = scores
		}
		cloned.Models[k] = m
	}

	return cloned
}
