package quota

import (
	"context"
	"errors"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

type mockQuotaProvider struct {
	name      string
	quotaFn   func(ctx context.Context) (*provider.QuotaSnapshot, error)
	callCount atomic.Int64
}

func (m *mockQuotaProvider) Name() string { return m.name }
func (m *mockQuotaProvider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	m.callCount.Add(1)
	if m.quotaFn != nil {
		return m.quotaFn(ctx)
	}
	return &provider.QuotaSnapshot{
		ObservedAt: time.Now().UTC(),
		Windows: []provider.QuotaWindow{
			{
				Label:     "requests",
				UsedPct:   50.0,
				Remaining: 50,
				Limit:     100,
				Unit:      "requests",
			},
		},
	}, nil
}

func TestSchedulerTicksWithFakeClock(t *testing.T) {
	startTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	currTime := startTime

	nowFn := func() time.Time { return currTime }

	reg := provider.NewRegistry()
	p1 := &mockQuotaProvider{
		name: "provider-alpha",
		quotaFn: func(ctx context.Context) (*provider.QuotaSnapshot, error) {
			return &provider.QuotaSnapshot{
				ObservedAt: currTime,
				Windows: []provider.QuotaWindow{
					{
						Label:     "5-hour",
						UsedPct:   85.0, // Should be QuotaLow
						Remaining: 15,
						Limit:     100,
						Unit:      "requests",
					},
				},
				Detail: "test alpha",
			}, nil
		},
	}
	p2 := &mockQuotaProvider{
		name: "provider-beta",
		quotaFn: func(ctx context.Context) (*provider.QuotaSnapshot, error) {
			return &provider.QuotaSnapshot{
				ObservedAt: currTime,
				Windows: []provider.QuotaWindow{
					{
						Label:     "daily",
						UsedPct:   100.0, // Should be QuotaExhausted
						Remaining: 0,
						Limit:     1000,
						Unit:      "requests",
					},
				},
			}, nil
		},
	}

	reg.Register(provider.Provider{Quota: p1})
	reg.Register(provider.Provider{Quota: p2})

	sm := state.NewStateManager(state.WithNow(nowFn))
	sm.SetProvider("provider-alpha", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})
	sm.SetProvider("provider-beta", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "quota_telemetry.db")
	w, err := storage.NewWriter(dbPath, storage.WithBatchSize(1))
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer w.Close()

	store := NewQuotaStore()

	sched := NewScheduler(reg, SchedulerConfig{
		ConcurrencyBudget: 2,
		DefaultPolicy: PollPolicy{
			Interval:       60 * time.Second,
			JitterRatio:    0,
			Deadline:       5 * time.Second,
			InitialBackoff: 2 * time.Second,
			MaxBackoff:     30 * time.Second,
			BackoffFactor:  2.0,
			StaggerDelay:   200 * time.Millisecond,
		},
		NowFn: nowFn,
		RandFn: func() float64 {
			return 0.5
		},
	},
		WithStateManager(sm),
		WithStorageWriter(w),
		WithStore(store),
		WithClock(nowFn),
	)

	ctx := context.Background()

	// Initial tick at T0:
	// provider-alpha has stagger 0s, provider-beta has stagger 200ms
	if err := sched.Tick(ctx, currTime); err != nil {
		t.Fatalf("Tick at T0 failed: %v", err)
	}

	// provider-alpha should have been polled (1), provider-beta not yet (0)
	if p1.callCount.Load() != 1 {
		t.Errorf("expected p1 callCount=1, got %d", p1.callCount.Load())
	}
	if p2.callCount.Load() != 0 {
		t.Errorf("expected p2 callCount=0 due to stagger, got %d", p2.callCount.Load())
	}

	// Advance time by 200ms: p2 is now due
	currTime = currTime.Add(200 * time.Millisecond)
	if err := sched.Tick(ctx, currTime); err != nil {
		t.Fatalf("Tick at T0+200ms failed: %v", err)
	}
	if p2.callCount.Load() != 1 {
		t.Errorf("expected p2 callCount=1, got %d", p2.callCount.Load())
	}

	// Check state manager updates
	snap := sm.Snapshot()
	alphaState := snap.Providers["provider-alpha"]
	if alphaState.Quota != state.QuotaLow {
		t.Errorf("expected provider-alpha QuotaLow (85%% used), got %v", alphaState.Quota)
	}
	if alphaState.Health != state.HealthHealthy {
		t.Errorf("expected provider-alpha HealthHealthy, got %v", alphaState.Health)
	}

	betaState := snap.Providers["provider-beta"]
	if betaState.Quota != state.QuotaExhausted {
		t.Errorf("expected provider-beta QuotaExhausted (100%% used), got %v", betaState.Quota)
	}

	// Check store has snapshots
	snapAlpha, okAlpha := store.Get("provider-alpha")
	if !okAlpha || snapAlpha.Detail != "test alpha" {
		t.Errorf("expected store to contain provider-alpha snapshot")
	}

	// Advance time by 30s: neither is due yet (interval is 60s)
	currTime = currTime.Add(30 * time.Second)
	if err := sched.Tick(ctx, currTime); err != nil {
		t.Fatalf("Tick at T0+30s failed: %v", err)
	}
	if p1.callCount.Load() != 1 || p2.callCount.Load() != 1 {
		t.Errorf("expected no additional polls before interval, got p1=%d p2=%d", p1.callCount.Load(), p2.callCount.Load())
	}

	// Advance time past 60s: p1 is due again
	currTime = currTime.Add(35 * time.Second) // total +65.2s
	if err := sched.Tick(ctx, currTime); err != nil {
		t.Fatalf("Tick at T0+65s failed: %v", err)
	}
	if p1.callCount.Load() != 2 {
		t.Errorf("expected p1 to be polled again, got %d", p1.callCount.Load())
	}
}

func TestSchedulerBackoffOnFailure(t *testing.T) {
	currTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	nowFn := func() time.Time { return currTime }

	var returnErr atomic.Bool
	returnErr.Store(true)

	reg := provider.NewRegistry()
	pFail := &mockQuotaProvider{
		name: "failing-provider",
		quotaFn: func(ctx context.Context) (*provider.QuotaSnapshot, error) {
			if returnErr.Load() {
				return nil, errors.New("upstream connection reset")
			}
			return &provider.QuotaSnapshot{
				ObservedAt: currTime,
				Windows: []provider.QuotaWindow{
					{Label: "reqs", UsedPct: 10, Remaining: 90, Limit: 100},
				},
			}, nil
		},
	}
	reg.Register(provider.Provider{Quota: pFail})

	sm := state.NewStateManager(state.WithNow(nowFn))
	sm.SetProvider("failing-provider", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})

	sched := NewScheduler(reg, SchedulerConfig{
		DefaultPolicy: PollPolicy{
			Interval:       60 * time.Second,
			JitterRatio:    0,
			Deadline:       5 * time.Second,
			InitialBackoff: 2 * time.Second,
			MaxBackoff:     16 * time.Second,
			BackoffFactor:  2.0,
			StaggerDelay:   0,
		},
		NowFn: nowFn,
	},
		WithStateManager(sm),
		WithClock(nowFn),
	)

	ctx := context.Background()

	// Poll 1: Fails -> Next poll at currTime + 2s (initial backoff)
	_ = sched.Tick(ctx, currTime)
	if pFail.callCount.Load() != 1 {
		t.Fatalf("expected callCount 1, got %d", pFail.callCount.Load())
	}

	// Advance 1s: not due yet
	currTime = currTime.Add(1 * time.Second)
	_ = sched.Tick(ctx, currTime)
	if pFail.callCount.Load() != 1 {
		t.Fatalf("expected callCount still 1 at +1s, got %d", pFail.callCount.Load())
	}

	// Advance another 1.1s (total 2.1s): Poll 2 fails -> Backoff 2s * 2 = 4s
	currTime = currTime.Add(1100 * time.Millisecond)
	_ = sched.Tick(ctx, currTime)
	if pFail.callCount.Load() != 2 {
		t.Fatalf("expected callCount 2 at +2.1s, got %d", pFail.callCount.Load())
	}

	snap := sm.Snapshot()
	if snap.Providers["failing-provider"].Health != state.HealthDegraded {
		t.Errorf("expected HealthDegraded after 2 failures, got %v", snap.Providers["failing-provider"].Health)
	}

	// Advance 3s: not due yet (needs 4s)
	currTime = currTime.Add(3 * time.Second)
	_ = sched.Tick(ctx, currTime)
	if pFail.callCount.Load() != 2 {
		t.Fatalf("expected callCount still 2 before 4s backoff, got %d", pFail.callCount.Load())
	}

	// Advance 1.1s: Poll 3 fails -> Backoff 4s * 2 = 8s
	currTime = currTime.Add(1100 * time.Millisecond)
	_ = sched.Tick(ctx, currTime)
	if pFail.callCount.Load() != 3 {
		t.Fatalf("expected callCount 3, got %d", pFail.callCount.Load())
	}

	// Now provider recovers
	returnErr.Store(false)

	// Advance 8.1s: Poll 4 succeeds!
	currTime = currTime.Add(8100 * time.Millisecond)
	_ = sched.Tick(ctx, currTime)
	if pFail.callCount.Load() != 4 {
		t.Fatalf("expected callCount 4 on recovery, got %d", pFail.callCount.Load())
	}

	snap = sm.Snapshot()
	if snap.Providers["failing-provider"].Health != state.HealthHealthy {
		t.Errorf("expected HealthHealthy after success, got %v", snap.Providers["failing-provider"].Health)
	}
	if snap.Providers["failing-provider"].Quota != state.QuotaHealthy {
		t.Errorf("expected QuotaHealthy, got %v", snap.Providers["failing-provider"].Quota)
	}
}
