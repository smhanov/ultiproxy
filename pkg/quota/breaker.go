package quota

import (
	"math/rand"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/state"
)

// CircuitBreakerConfig configures the quota circuit breaker.
type CircuitBreakerConfig struct {
	DefaultOpenDuration time.Duration
	FailureThreshold    int
	JitterRatio         float64
	NowFn               func() time.Time
	RandFn              func() float64
}

// BreakerState holds local circuit tracking per provider.
type providerBreakerState struct {
	circuit            state.CircuitState
	openedAt           time.Time
	cooldown           time.Duration
	probeInFlight      bool
	consecutiveSuccess int
	consecutiveFailure int
}

// CircuitBreaker manages provider circuit breaker transitions and probes.
type CircuitBreaker struct {
	mu     sync.RWMutex
	sm     *state.StateManager
	cfg    CircuitBreakerConfig
	states map[string]*providerBreakerState
	nowFn  func() time.Time
	randFn func() float64
}

// NewCircuitBreaker constructs a new CircuitBreaker.
func NewCircuitBreaker(sm *state.StateManager, cfg CircuitBreakerConfig) *CircuitBreaker {
	if cfg.DefaultOpenDuration <= 0 {
		cfg.DefaultOpenDuration = 30 * time.Second
	}
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = 5
	}
	if cfg.JitterRatio < 0 {
		cfg.JitterRatio = 0
	}
	nowFn := cfg.NowFn
	if nowFn == nil {
		if sm != nil {
			nowFn = sm.Now
		} else {
			nowFn = time.Now
		}
	}
	randFn := cfg.RandFn
	if randFn == nil {
		randFn = rand.Float64
	}

	return &CircuitBreaker{
		sm:     sm,
		cfg:    cfg,
		states: make(map[string]*providerBreakerState),
		nowFn:  nowFn,
		randFn: randFn,
	}
}

// SetNow overrides the current clock function (useful in tests).
func (cb *CircuitBreaker) SetNow(fn func() time.Time) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.nowFn = fn
	if cb.sm != nil {
		cb.sm.SetNow(fn)
	}
}

// SetRand overrides the random generator (useful in tests).
func (cb *CircuitBreaker) SetRand(fn func() float64) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.randFn = fn
}

func (cb *CircuitBreaker) now() time.Time {
	if cb.nowFn != nil {
		return cb.nowFn()
	}
	return time.Now()
}

func (cb *CircuitBreaker) getOrCreateState(providerID string) *providerBreakerState {
	st, ok := cb.states[providerID]
	if !ok {
		st = &providerBreakerState{
			circuit: state.CircuitClosed,
		}
		cb.states[providerID] = st
	}
	return st
}

// AllowProbe checks if a request or probe is allowed for the provider.
// Closed: true
// Open: transitions to HalfOpen after cooldown + jitter, allowing 1 probe.
// HalfOpen: allows exactly 1 probe in flight.
// AdminDisabled: always false.
func (cb *CircuitBreaker) AllowProbe(providerID string) bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := cb.now()

	// Check admin state from StateManager if available
	if cb.sm != nil {
		snap := cb.sm.Snapshot()
		if p, ok := snap.Providers[providerID]; ok {
			if p.Admin == state.AdminDisabled {
				return false
			}
		}
	}

	st := cb.getOrCreateState(providerID)

	switch st.circuit {
	case state.CircuitClosed:
		return true

	case state.CircuitOpen:
		cooldown := st.cooldown
		if cooldown <= 0 {
			cooldown = cb.cfg.DefaultOpenDuration
		}
		if cb.cfg.JitterRatio > 0 {
			jitter := 1.0 + (cb.randFn()*2-1)*cb.cfg.JitterRatio
			cooldown = time.Duration(float64(cooldown) * jitter)
		}

		if now.Sub(st.openedAt) >= cooldown {
			// Transition to HalfOpen and grant the single probe slot
			st.circuit = state.CircuitHalfOpen
			st.probeInFlight = true
			st.consecutiveSuccess = 0
			cb.syncToStateManager(providerID, state.CircuitHalfOpen, true)
			return true
		}
		return false

	case state.CircuitHalfOpen:
		if !st.probeInFlight {
			st.probeInFlight = true
			cb.syncToStateManager(providerID, state.CircuitHalfOpen, true)
			return true
		}
		return false

	default:
		return true
	}
}

// RecordSuccess records a successful interaction.
// If HalfOpen -> transitions to Closed.
// Auto-recovery strictly preserves AdminState.
func (cb *CircuitBreaker) RecordSuccess(providerID string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	st := cb.getOrCreateState(providerID)
	st.consecutiveSuccess++
	st.consecutiveFailure = 0
	st.probeInFlight = false

	if st.circuit == state.CircuitHalfOpen {
		st.circuit = state.CircuitClosed
	}

	if cb.sm != nil {
		cb.sm.RecordSuccess(providerID)
	}
}

// RecordFailure records a failure for a provider.
// If HalfOpen -> transitions back to Open immediately.
// If Closed and threshold reached -> transitions to Open.
// AdminState is never modified.
func (cb *CircuitBreaker) RecordFailure(providerID string, err error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := cb.now()
	st := cb.getOrCreateState(providerID)
	st.consecutiveFailure++
	st.consecutiveSuccess = 0

	if st.circuit == state.CircuitHalfOpen {
		// Single probe failed: return to Open immediately
		st.circuit = state.CircuitOpen
		st.openedAt = now
		st.probeInFlight = false
		if st.cooldown <= 0 {
			st.cooldown = cb.cfg.DefaultOpenDuration
		}
		cb.syncToStateManager(providerID, state.CircuitOpen, false)
	} else if st.circuit == state.CircuitClosed {
		if st.consecutiveFailure >= cb.cfg.FailureThreshold {
			st.circuit = state.CircuitOpen
			st.openedAt = now
			st.cooldown = cb.cfg.DefaultOpenDuration
			cb.syncToStateManager(providerID, state.CircuitOpen, false)
		}
	}

	if cb.sm != nil {
		cb.sm.RecordFailure(providerID, err)
	}
}

// RecordLimitError handles a typed LimitError and updates breaker and state.
func (cb *CircuitBreaker) RecordLimitError(limitErr LimitError) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	providerID := limitErr.Provider
	if providerID == "" {
		return
	}

	now := cb.now()
	st := cb.getOrCreateState(providerID)

	cooldown := limitErr.RetryAfter
	if cooldown <= 0 {
		cooldown = cb.cfg.DefaultOpenDuration
	}
	st.cooldown = cooldown
	st.openedAt = now
	st.probeInFlight = false

	switch limitErr.Kind {
	case LimitKindSpendLimit, LimitKindAbuseGuard:
		st.circuit = state.CircuitOpen
		if cb.sm != nil {
			cb.sm.Update(func(snap *state.RuntimeSnapshot) {
				p := snap.Providers[providerID]
				p.Quota = state.QuotaExhausted
				p.Circuit = state.CircuitOpen
				p.CircuitOpenedAt = now
				p.Health = state.HealthUnavailable
				p.ObservedAt = now
				p.Error = limitErr.Error()
				// NEVER touch Admin!
				snap.Providers[providerID] = p
			})
		}

	case LimitKindQuotaWindow:
		if cb.sm != nil {
			cb.sm.Update(func(snap *state.RuntimeSnapshot) {
				p := snap.Providers[providerID]
				p.Quota = state.QuotaExhausted
				if limitErr.RetryAt != nil {
					p.ValidUntil = *limitErr.RetryAt
				}
				p.ObservedAt = now
				p.Error = limitErr.Error()
				// NEVER touch Admin!
				snap.Providers[providerID] = p
			})
		}

	case LimitKindBurstRate, LimitKindCapacity, LimitKindTokenRate, LimitKindUnknown:
		st.circuit = state.CircuitOpen
		cb.syncToStateManager(providerID, state.CircuitOpen, false)
		if cb.sm != nil {
			cb.sm.Update(func(snap *state.RuntimeSnapshot) {
				p := snap.Providers[providerID]
				p.Circuit = state.CircuitOpen
				p.CircuitOpenedAt = now
				p.Health = state.HealthDegraded
				p.ObservedAt = now
				p.Error = limitErr.Error()
				// NEVER touch Admin!
				snap.Providers[providerID] = p
			})
		}

	default:
		// Default failure tracking
		st.consecutiveFailure++
		if st.consecutiveFailure >= cb.cfg.FailureThreshold {
			st.circuit = state.CircuitOpen
			cb.syncToStateManager(providerID, state.CircuitOpen, false)
		}
	}
}

// TripCircuit directly trips a provider's circuit to Open with given cooldown.
func (cb *CircuitBreaker) TripCircuit(providerID string, cooldown time.Duration) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := cb.now()
	st := cb.getOrCreateState(providerID)
	st.circuit = state.CircuitOpen
	st.openedAt = now
	st.probeInFlight = false
	if cooldown > 0 {
		st.cooldown = cooldown
	} else {
		st.cooldown = cb.cfg.DefaultOpenDuration
	}

	cb.syncToStateManager(providerID, state.CircuitOpen, false)
}

func (cb *CircuitBreaker) syncToStateManager(providerID string, circ state.CircuitState, probeInFlight bool) {
	if cb.sm == nil {
		return
	}
	now := cb.now()
	cb.sm.Update(func(snap *state.RuntimeSnapshot) {
		p, ok := snap.Providers[providerID]
		if !ok {
			return
		}
		p.Circuit = circ
		p.ProbeInFlight = probeInFlight
		p.ObservedAt = now
		if circ == state.CircuitOpen {
			p.CircuitOpenedAt = now
		}
		// NEVER touch Admin!
		snap.Providers[providerID] = p
	})
}
