package state

import (
	"errors"
	"sync"
	"testing"
	"time"
)

func TestAtomicSnapshotAndCopyOnWrite(t *testing.T) {
	sm := NewStateManager()
	snap0 := sm.Snapshot()
	if snap0.Version != 1 {
		t.Fatalf("expected initial version 1, got %d", snap0.Version)
	}

	// Update with model and provider
	snap1 := sm.Update(func(s *RuntimeSnapshot) {
		s.Providers["anthropic"] = ProviderRuntime{
			Admin:      AdminEnabled,
			Health:     HealthHealthy,
			Quota:      QuotaHealthy,
			Circuit:    CircuitClosed,
			Credential: CredentialValid,
		}
		s.Models["claude-3-5-sonnet"] = ModelRuntime{
			ID:           "claude-3-5-sonnet",
			Provider:     "anthropic",
			Enabled:      true,
			ContextLimit: 200000,
		}
	})

	if snap1.Version != 2 {
		t.Errorf("expected version 2 after update, got %d", snap1.Version)
	}

	// Verify old snapshot is unmodified (copy-on-write immutability)
	if len(snap0.Providers) != 0 {
		t.Errorf("expected snap0 to remain empty, got %d providers", len(snap0.Providers))
	}
	if len(snap1.Providers) != 1 {
		t.Errorf("expected snap1 to have 1 provider, got %d", len(snap1.Providers))
	}

	// Concurrent updates should bump version correctly without races
	var wg sync.WaitGroup
	numWorkers := 20
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			sm.Update(func(s *RuntimeSnapshot) {
				s.Models["dynamic-model"] = ModelRuntime{
					ID:        "dynamic-model",
					Provider:  "anthropic",
					MaxOutput: idx,
				}
			})
		}(i)
	}
	wg.Wait()

	finalSnap := sm.Snapshot()
	expectedMinVersion := uint64(2 + numWorkers)
	if finalSnap.Version != expectedMinVersion {
		t.Errorf("expected version %d, got %d", expectedMinVersion, finalSnap.Version)
	}
}

func TestCircuitBreakerOpenToHalfOpenToClosed(t *testing.T) {
	currTime := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)
	fakeClock := func() time.Time { return currTime }

	sm := NewStateManager(
		WithNow(fakeClock),
		WithOpenDuration(10*time.Second),
		WithFailureThreshold(3),
		WithJitterRatio(0), // deterministic for testing
	)

	provider := "openai"
	sm.SetProvider(provider, ProviderRuntime{
		Admin:      AdminEnabled,
		Health:     HealthHealthy,
		Quota:      QuotaHealthy,
		Circuit:    CircuitClosed,
		Credential: CredentialValid,
	})

	// 1. Trip circuit open with 3 consecutive failures
	errSample := errors.New("upstream timeout")
	sm.RecordFailure(provider, errSample)
	sm.RecordFailure(provider, errSample)
	sm.RecordFailure(provider, errSample)

	snap := sm.Snapshot()
	if snap.Providers[provider].Circuit != CircuitOpen {
		t.Fatalf("expected circuit Open, got %s", snap.Providers[provider].Circuit)
	}

	// Requests before openDuration expires must NOT be allowed
	if sm.AllowProbe(provider) {
		t.Fatalf("probe should NOT be allowed while circuit is Open before cooldown")
	}

	// Advance clock by 10s -> cooldown expires
	currTime = currTime.Add(10 * time.Second)

	// Single probe request MUST now be allowed (transitions to HalfOpen)
	if !sm.AllowProbe(provider) {
		t.Fatalf("first probe should be allowed after cooldown")
	}

	snap = sm.Snapshot()
	if snap.Providers[provider].Circuit != CircuitHalfOpen {
		t.Fatalf("expected circuit HalfOpen, got %s", snap.Providers[provider].Circuit)
	}

	// Second concurrent probe while one is in-flight MUST be rejected (single probe only)
	if sm.AllowProbe(provider) {
		t.Fatalf("second concurrent probe must be rejected in HalfOpen")
	}

	// Single probe succeeds -> circuit closes!
	sm.RecordSuccess(provider)

	snap = sm.Snapshot()
	if snap.Providers[provider].Circuit != CircuitClosed {
		t.Fatalf("expected circuit Closed after successful probe, got %s", snap.Providers[provider].Circuit)
	}
	if snap.Providers[provider].Health != HealthHealthy {
		t.Fatalf("expected health healthy after probe success, got %s", snap.Providers[provider].Health)
	}

	// Now standard requests are allowed
	if !sm.AllowProbe(provider) {
		t.Fatalf("requests should be allowed in Closed circuit")
	}
}

func TestCircuitBreakerOpenToHalfOpenToOpen(t *testing.T) {
	currTime := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)
	fakeClock := func() time.Time { return currTime }

	sm := NewStateManager(
		WithNow(fakeClock),
		WithOpenDuration(15*time.Second),
		WithFailureThreshold(1),
		WithJitterRatio(0),
	)

	provider := "freebuff"
	sm.SetProvider(provider, ProviderRuntime{
		Admin:      AdminEnabled,
		Health:     HealthHealthy,
		Quota:      QuotaHealthy,
		Circuit:    CircuitClosed,
		Credential: CredentialValid,
	})

	// Trip open
	sm.RecordFailure(provider, errors.New("initial failure"))
	if sm.Snapshot().Providers[provider].Circuit != CircuitOpen {
		t.Fatalf("expected circuit Open")
	}

	// Advance time past cooldown
	currTime = currTime.Add(16 * time.Second)

	// Single probe allowed -> HalfOpen
	if !sm.AllowProbe(provider) {
		t.Fatalf("probe should be allowed")
	}
	if sm.Snapshot().Providers[provider].Circuit != CircuitHalfOpen {
		t.Fatalf("expected circuit HalfOpen")
	}

	// Single probe FAILS -> transitions back to Open!
	sm.RecordFailure(provider, errors.New("probe failure"))

	snap := sm.Snapshot()
	if snap.Providers[provider].Circuit != CircuitOpen {
		t.Fatalf("expected circuit back to Open after failed probe, got %s", snap.Providers[provider].Circuit)
	}
	if snap.Providers[provider].Health != HealthUnavailable {
		t.Fatalf("expected health unavailable, got %s", snap.Providers[provider].Health)
	}

	// Immediately following requests must be rejected
	if sm.AllowProbe(provider) {
		t.Fatalf("probe should be rejected after transitioning back to Open")
	}
}

func TestAdminDisabledNeverReEnabledByAutoRecovery(t *testing.T) {
	// CRITICAL TEST: an admin-disabled provider must NEVER be re-enabled by
	// auto-recovery (quota reset / circuit close) — assert that recovery
	// only touches Circuit/Quota/Health, not Admin.

	currTime := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)
	sm := NewStateManager(
		WithNow(func() time.Time { return currTime }),
		WithOpenDuration(5*time.Second),
		WithFailureThreshold(2),
		WithJitterRatio(0),
	)

	provider := "custom-disabled"
	sm.SetProvider(provider, ProviderRuntime{
		Admin:      AdminDisabled, // Explicitly disabled by admin
		Health:     HealthUnavailable,
		Quota:      QuotaExhausted,
		Circuit:    CircuitOpen,
		Credential: CredentialValid,
	})

	// Verify initially disabled and unavailable
	initSnap := sm.Snapshot().Providers[provider]
	if initSnap.Admin != AdminDisabled {
		t.Fatalf("expected AdminDisabled")
	}
	if initSnap.IsAvailable() {
		t.Fatalf("disabled provider must NOT be available")
	}

	// 1. Trigger Quota Reset
	sm.ResetQuota(provider)
	snapAfterQuota := sm.Snapshot().Providers[provider]
	if snapAfterQuota.Quota != QuotaHealthy {
		t.Errorf("quota should recover to healthy, got %s", snapAfterQuota.Quota)
	}
	if snapAfterQuota.Admin != AdminDisabled {
		t.Fatalf("CRITICAL VIOLATION: ResetQuota re-enabled Admin! Got %s", snapAfterQuota.Admin)
	}
	if snapAfterQuota.IsAvailable() {
		t.Fatalf("provider still admin-disabled, should not be available")
	}

	// 2. Trigger Circuit Recovery
	sm.RecoverCircuit(provider)
	snapAfterCircuit := sm.Snapshot().Providers[provider]
	if snapAfterCircuit.Circuit != CircuitClosed {
		t.Errorf("circuit should recover to closed, got %s", snapAfterCircuit.Circuit)
	}
	if snapAfterCircuit.Admin != AdminDisabled {
		t.Fatalf("CRITICAL VIOLATION: RecoverCircuit re-enabled Admin! Got %s", snapAfterCircuit.Admin)
	}
	if snapAfterCircuit.IsAvailable() {
		t.Fatalf("provider still admin-disabled, should not be available")
	}

	// 3. Trigger RecordSuccess (which closes half-open circuit and marks health healthy)
	sm.RecordSuccess(provider)
	snapAfterSuccess := sm.Snapshot().Providers[provider]
	if snapAfterSuccess.Health != HealthHealthy {
		t.Errorf("health should be healthy, got %s", snapAfterSuccess.Health)
	}
	if snapAfterSuccess.Admin != AdminDisabled {
		t.Fatalf("CRITICAL VIOLATION: RecordSuccess re-enabled Admin! Got %s", snapAfterSuccess.Admin)
	}
	if snapAfterSuccess.IsAvailable() {
		t.Fatalf("provider still admin-disabled, should not be available")
	}

	// 4. Probes must be refused for admin-disabled provider
	if sm.AllowProbe(provider) {
		t.Fatalf("CRITICAL VIOLATION: AllowProbe allowed request for admin-disabled provider")
	}

	// 5. Only explicit SetAdminState can enable it
	sm.SetAdminState(provider, AdminEnabled)
	snapAfterAdminEnable := sm.Snapshot().Providers[provider]
	if snapAfterAdminEnable.Admin != AdminEnabled {
		t.Fatalf("expected AdminEnabled after explicit SetAdminState")
	}
	if !snapAfterAdminEnable.IsAvailable() {
		t.Fatalf("provider should now be available after explicit admin enable")
	}
}
