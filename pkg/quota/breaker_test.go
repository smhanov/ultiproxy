package quota

import (
	"errors"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/state"
)

func TestCircuitBreakerTransitions(t *testing.T) {
	currTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	nowFn := func() time.Time { return currTime }

	sm := state.NewStateManager(
		state.WithNow(nowFn),
		state.WithOpenDuration(30*time.Second),
		state.WithFailureThreshold(3),
	)

	sm.SetProvider("test-provider", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})

	cb := NewCircuitBreaker(sm, CircuitBreakerConfig{
		DefaultOpenDuration: 30 * time.Second,
		FailureThreshold:    3,
		JitterRatio:         0, // 0 jitter for deterministic timing
		NowFn:               nowFn,
	})

	// 1. Initial state: Closed, probe allowed
	if !cb.AllowProbe("test-provider") {
		t.Fatalf("expected AllowProbe to be true for Closed circuit")
	}

	// 2. Trip the circuit open
	cb.TripCircuit("test-provider", 30*time.Second)

	// Immediately after tripping: Open, probe NOT allowed
	if cb.AllowProbe("test-provider") {
		t.Fatalf("expected AllowProbe to be false immediately after tripping")
	}

	// 10s elapsed: still Open, probe NOT allowed
	currTime = currTime.Add(10 * time.Second)
	if cb.AllowProbe("test-provider") {
		t.Fatalf("expected AllowProbe to be false after 10s")
	}

	// 30s elapsed (cooldown met): transitions to HalfOpen, 1 probe allowed
	currTime = currTime.Add(20 * time.Second)
	if !cb.AllowProbe("test-provider") {
		t.Fatalf("expected AllowProbe to transition to HalfOpen and allow 1 probe")
	}

	// Second concurrent probe while one probe in flight: NOT allowed
	if cb.AllowProbe("test-provider") {
		t.Fatalf("expected second concurrent probe to be rejected in HalfOpen")
	}

	// Probe succeeds: circuit closes
	cb.RecordSuccess("test-provider")

	// Now Closed: multiple probes allowed again
	if !cb.AllowProbe("test-provider") {
		t.Fatalf("expected AllowProbe to be true after success closed circuit")
	}
	if !cb.AllowProbe("test-provider") {
		t.Fatalf("expected subsequent probes to be true")
	}

	// 3. Test probe failure in HalfOpen -> returns to Open
	cb.TripCircuit("test-provider", 30*time.Second)
	currTime = currTime.Add(30 * time.Second)

	// Transition to HalfOpen, grant probe
	if !cb.AllowProbe("test-provider") {
		t.Fatalf("expected probe allowed after cooldown")
	}

	// Probe fails
	cb.RecordFailure("test-provider", errors.New("upstream timeout"))

	// Should be back to Open: probe not allowed
	if cb.AllowProbe("test-provider") {
		t.Fatalf("expected circuit to be Open again after probe failure")
	}

	// Cooldown has reset! Advancing only 10s should still reject
	currTime = currTime.Add(10 * time.Second)
	if cb.AllowProbe("test-provider") {
		t.Fatalf("expected circuit to still be Open after 10s")
	}

	// After full 30s cooldown from failure: HalfOpen again
	currTime = currTime.Add(20 * time.Second)
	if !cb.AllowProbe("test-provider") {
		t.Fatalf("expected HalfOpen probe allowed after full cooldown from probe failure")
	}
}

func TestAdminDisabledNeverReEnabledByRecovery(t *testing.T) {
	currTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	nowFn := func() time.Time { return currTime }

	sm := state.NewStateManager(state.WithNow(nowFn))
	sm.SetProvider("disabled-provider", state.ProviderRuntime{
		Admin:      state.AdminDisabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})

	cb := NewCircuitBreaker(sm, CircuitBreakerConfig{
		DefaultOpenDuration: 30 * time.Second,
		NowFn:               nowFn,
	})

	// Probe must NEVER be allowed for admin disabled provider
	if cb.AllowProbe("disabled-provider") {
		t.Fatalf("AllowProbe must be false for AdminDisabled")
	}

	// Even if RecordSuccess is invoked
	cb.RecordSuccess("disabled-provider")

	snap := sm.Snapshot()
	if snap.Providers["disabled-provider"].Admin != state.AdminDisabled {
		t.Fatalf("AdminState was mutated by RecordSuccess! Got %v", snap.Providers["disabled-provider"].Admin)
	}

	// Even if cooldown passes
	currTime = currTime.Add(100 * time.Hour)
	if cb.AllowProbe("disabled-provider") {
		t.Fatalf("AllowProbe must still be false for AdminDisabled after time passes")
	}

	snap = sm.Snapshot()
	if snap.Providers["disabled-provider"].Admin != state.AdminDisabled {
		t.Fatalf("AdminState was corrupted! Got %v", snap.Providers["disabled-provider"].Admin)
	}
}

func TestCircuitBreakerJitter(t *testing.T) {
	currTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	nowFn := func() time.Time { return currTime }

	cb := NewCircuitBreaker(nil, CircuitBreakerConfig{
		DefaultOpenDuration: 30 * time.Second,
		JitterRatio:         0.2, // 20% jitter
		NowFn:               nowFn,
		RandFn: func() float64 {
			return 1.0 // Maximum jitter factor: 1 + (1*2-1)*0.2 = 1.2 -> 36s
		},
	})

	cb.TripCircuit("p1", 30*time.Second)

	// At 31s, should still be Open because jitter made cooldown 36s
	currTime = currTime.Add(31 * time.Second)
	if cb.AllowProbe("p1") {
		t.Fatalf("expected probe to be rejected at 31s with jitter cooldown of 36s")
	}

	// At 36s, should be HalfOpen
	currTime = currTime.Add(5 * time.Second)
	if !cb.AllowProbe("p1") {
		t.Fatalf("expected probe to be allowed at 36s")
	}
}
