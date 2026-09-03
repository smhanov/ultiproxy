package state

import (
	"math/rand"
	"sync/atomic"
	"time"
)

// StateManager coordinates provider states, atomic snapshot swapping, and circuit breaker logic.
type StateManager struct {
	snapshot atomic.Pointer[RuntimeSnapshot]
	nowFn    func() time.Time

	// Circuit breaker configuration
	openDuration     time.Duration
	failureThreshold int
	jitterRatio      float64
}

// Option configures StateManager.
type Option func(*StateManager)

// WithNow sets the clock function.
func WithNow(fn func() time.Time) Option {
	return func(sm *StateManager) {
		sm.nowFn = fn
	}
}

// WithOpenDuration sets the duration a circuit remains open before entering half-open.
func WithOpenDuration(d time.Duration) Option {
	return func(sm *StateManager) {
		sm.openDuration = d
	}
}

// WithFailureThreshold sets consecutive failure count to trip circuit open.
func WithFailureThreshold(threshold int) Option {
	return func(sm *StateManager) {
		sm.failureThreshold = threshold
	}
}

// WithJitterRatio sets jitter ratio for circuit retry (0.0 to 1.0).
func WithJitterRatio(ratio float64) Option {
	return func(sm *StateManager) {
		sm.jitterRatio = ratio
	}
}

// NewStateManager creates a new StateManager with an initial empty snapshot.
func NewStateManager(opts ...Option) *StateManager {
	sm := &StateManager{
		nowFn:            time.Now,
		openDuration:     30 * time.Second,
		failureThreshold: 5,
		jitterRatio:      0.2,
	}

	for _, opt := range opts {
		opt(sm)
	}

	initSnap := &RuntimeSnapshot{
		Version:   1,
		Providers: make(map[string]ProviderRuntime),
		Models:    make(map[string]ModelRuntime),
	}
	sm.snapshot.Store(initSnap)

	return sm
}

// Snapshot returns the current immutable RuntimeSnapshot.
func (sm *StateManager) Snapshot() *RuntimeSnapshot {
	return sm.snapshot.Load()
}

// SetSnapshot stores a new snapshot atomically.
func (sm *StateManager) SetSnapshot(snap *RuntimeSnapshot) {
	sm.snapshot.Store(snap)
}

// Update executes a copy-on-write mutation of the snapshot atomically, bumping Version.
func (sm *StateManager) Update(fn func(*RuntimeSnapshot)) *RuntimeSnapshot {
	for {
		current := sm.snapshot.Load()
		cloned := current.Clone()
		fn(cloned)
		cloned.Version = current.Version + 1
		if sm.snapshot.CompareAndSwap(current, cloned) {
			return cloned
		}
	}
}

// Now returns the current time from the injected clock.
func (sm *StateManager) Now() time.Time {
	if sm.nowFn != nil {
		return sm.nowFn()
	}
	return time.Now()
}

// SetNow overrides the current clock function (useful in tests).
func (sm *StateManager) SetNow(fn func() time.Time) {
	sm.nowFn = fn
}

// SetProvider configures or initializes a provider runtime.
func (sm *StateManager) SetProvider(id string, p ProviderRuntime) {
	sm.Update(func(snap *RuntimeSnapshot) {
		snap.Providers[id] = p
	})
}

// SetAdminState updates AdminState for a provider. Only explicit admin actions should call this.
func (sm *StateManager) SetAdminState(id string, admin AdminState) {
	sm.Update(func(snap *RuntimeSnapshot) {
		p, ok := snap.Providers[id]
		if !ok {
			return
		}
		p.Admin = admin
		snap.Providers[id] = p
	})
}

// AllowProbe checks if a request/probe is permitted to proceed for the provider.
// For Closed: always true.
// For Open: transitions to HalfOpen if backoff elapsed, allowing a single probe.
// For HalfOpen: allows exactly one single probe in flight.
func (sm *StateManager) AllowProbe(providerID string) bool {
	now := sm.Now()

	var allowed bool
	sm.Update(func(snap *RuntimeSnapshot) {
		p, ok := snap.Providers[providerID]
		if !ok {
			allowed = false
			return
		}

		// Admin-disabled providers cannot probe
		if p.Admin == AdminDisabled {
			allowed = false
			return
		}

		switch p.Circuit {
		case CircuitClosed:
			allowed = true

		case CircuitOpen:
			cooldown := sm.openDuration
			if sm.jitterRatio > 0 {
				// Jitter between [1 - jitterRatio, 1 + jitterRatio]
				jitter := 1.0 + (rand.Float64()*2-1)*sm.jitterRatio
				cooldown = time.Duration(float64(cooldown) * jitter)
			}

			if now.Sub(p.CircuitOpenedAt) >= sm.openDuration {
				// Transition to HalfOpen and claim the probe slot
				p.Circuit = CircuitHalfOpen
				p.ProbeInFlight = true
				p.ConsecutiveSuccess = 0
				p.ObservedAt = now
				snap.Providers[providerID] = p
				allowed = true
			} else {
				allowed = false
			}

		case CircuitHalfOpen:
			if !p.ProbeInFlight {
				// Allow single probe request
				p.ProbeInFlight = true
				snap.Providers[providerID] = p
				allowed = true
			} else {
				allowed = false
			}
		}
	})

	return allowed
}

// RecordSuccess records a successful provider interaction.
// If circuit was HalfOpen, it closes the circuit.
// IMPORTANT: Auto-recovery only touches Circuit, Quota, and Health — NEVER Admin!
func (sm *StateManager) RecordSuccess(providerID string) {
	now := sm.Now()
	sm.Update(func(snap *RuntimeSnapshot) {
		p, ok := snap.Providers[providerID]
		if !ok {
			return
		}

		p.LastSuccess = now
		p.LastAttempt = now
		p.ObservedAt = now
		p.Error = ""
		p.ConsecutiveFailure = 0
		p.ConsecutiveSuccess++

		if p.Health == HealthUnavailable || p.Health == HealthDegraded {
			p.Health = HealthHealthy
		}

		if p.Circuit == CircuitHalfOpen {
			p.Circuit = CircuitClosed
			p.ProbeInFlight = false
		} else if p.Circuit == CircuitClosed {
			p.ProbeInFlight = false
		}

		// Strictly preserve existing p.Admin (NEVER touched by auto-recovery)
		snap.Providers[providerID] = p
	})
}

// RecordFailure records a failure for a provider.
// If failures exceed threshold, circuit opens.
// If circuit was HalfOpen, single probe failed -> circuit returns to Open.
// IMPORTANT: Recovery/failure handling NEVER modifies Admin!
func (sm *StateManager) RecordFailure(providerID string, err error) {
	now := sm.Now()
	sm.Update(func(snap *RuntimeSnapshot) {
		p, ok := snap.Providers[providerID]
		if !ok {
			return
		}

		p.LastAttempt = now
		p.ObservedAt = now
		p.ConsecutiveSuccess = 0
		p.ConsecutiveFailure++
		if err != nil {
			p.Error = err.Error()
		}

		if p.Circuit == CircuitHalfOpen {
			// Single probe failed: return to Open
			p.Circuit = CircuitOpen
			p.CircuitOpenedAt = now
			p.ProbeInFlight = false
			p.Health = HealthUnavailable
		} else if p.Circuit == CircuitClosed {
			if p.ConsecutiveFailure >= sm.failureThreshold {
				p.Circuit = CircuitOpen
				p.CircuitOpenedAt = now
				p.Health = HealthDegraded
			}
		}

		// Strictly preserve existing p.Admin (NEVER touched)
		snap.Providers[providerID] = p
	})
}

// ResetQuota updates Quota state back to Healthy.
// IMPORTANT: Recovery only touches Quota, never Admin!
func (sm *StateManager) ResetQuota(providerID string) {
	now := sm.Now()
	sm.Update(func(snap *RuntimeSnapshot) {
		p, ok := snap.Providers[providerID]
		if !ok {
			return
		}
		p.Quota = QuotaHealthy
		p.ObservedAt = now
		// Admin is strictly preserved
		snap.Providers[providerID] = p
	})
}

// RecoverCircuit manually or automatically closes the circuit.
// IMPORTANT: Recovery only touches Circuit/Health, never Admin!
func (sm *StateManager) RecoverCircuit(providerID string) {
	now := sm.Now()
	sm.Update(func(snap *RuntimeSnapshot) {
		p, ok := snap.Providers[providerID]
		if !ok {
			return
		}
		p.Circuit = CircuitClosed
		p.ProbeInFlight = false
		p.ConsecutiveFailure = 0
		p.ObservedAt = now
		if p.Health == HealthUnavailable {
			p.Health = HealthHealthy
		}
		// Admin is strictly preserved
		snap.Providers[providerID] = p
	})
}
