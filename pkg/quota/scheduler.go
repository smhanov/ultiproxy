package quota

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// PollPolicy defines polling parameters for a provider.
type PollPolicy struct {
	Interval       time.Duration
	JitterRatio    float64
	Deadline       time.Duration
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	BackoffFactor  float64
	StaggerDelay   time.Duration
}

// DefaultPollPolicy provides standard defaults.
func DefaultPollPolicy() PollPolicy {
	return PollPolicy{
		Interval:       60 * time.Second,
		JitterRatio:    0.1,
		Deadline:       10 * time.Second,
		InitialBackoff: 2 * time.Second,
		MaxBackoff:     60 * time.Second,
		BackoffFactor:  2.0,
		StaggerDelay:   500 * time.Millisecond,
	}
}

// SchedulerConfig holds configuration for the background QuotaScheduler.
type SchedulerConfig struct {
	ConcurrencyBudget int
	DefaultPolicy     PollPolicy
	ProviderPolicies  map[string]PollPolicy
	NowFn             func() time.Time
	RandFn            func() float64
}

// providerSchedule tracks next execution and backoff state per provider.
type providerSchedule struct {
	policy       PollPolicy
	failures     int
	nextPollAt   time.Time
	lastPolledAt time.Time
}

// Scheduler periodically polls registered QuotaProviders, persists observations to storage,
// and updates the runtime snapshot in StateManager.
type Scheduler struct {
	mu        sync.Mutex
	registry  *provider.Registry
	store     *QuotaStore
	sm        *state.StateManager
	writer    *storage.Writer
	cfg       SchedulerConfig
	sem       chan struct{}
	schedules map[string]*providerSchedule
	nowFn     func() time.Time
	randFn    func() float64

	running  bool
	stopChan chan struct{}
	doneChan chan struct{}
}

// Option configures Scheduler.
type Option func(*Scheduler)

// WithStorageWriter sets the telemetry writer.
func WithStorageWriter(w *storage.Writer) Option {
	return func(s *Scheduler) { s.writer = w }
}

// WithStateManager sets the state manager.
func WithStateManager(sm *state.StateManager) Option {
	return func(s *Scheduler) { s.sm = sm }
}

// WithStore sets the QuotaStore.
func WithStore(store *QuotaStore) Option {
	return func(s *Scheduler) { s.store = store }
}

// WithClock sets a custom clock function for deterministic testing.
func WithClock(nowFn func() time.Time) Option {
	return func(s *Scheduler) {
		s.nowFn = nowFn
	}
}

// WithRand sets a custom random function for deterministic testing.
func WithRand(randFn func() float64) Option {
	return func(s *Scheduler) {
		s.randFn = randFn
	}
}

// NewScheduler creates a background quota scheduler.
func NewScheduler(reg *provider.Registry, cfg SchedulerConfig, opts ...Option) *Scheduler {
	if cfg.ConcurrencyBudget <= 0 {
		cfg.ConcurrencyBudget = 4
	}
	if cfg.DefaultPolicy.Interval <= 0 {
		cfg.DefaultPolicy = DefaultPollPolicy()
	}
	if cfg.ProviderPolicies == nil {
		cfg.ProviderPolicies = make(map[string]PollPolicy)
	}
	nowFn := cfg.NowFn
	if nowFn == nil {
		nowFn = time.Now
	}
	randFn := cfg.RandFn
	if randFn == nil {
		randFn = rand.Float64
	}

	s := &Scheduler{
		registry:  reg,
		cfg:       cfg,
		sem:       make(chan struct{}, cfg.ConcurrencyBudget),
		schedules: make(map[string]*providerSchedule),
		nowFn:     nowFn,
		randFn:    randFn,
	}

	for _, opt := range opts {
		opt(s)
	}

	if s.store == nil {
		s.store = NewQuotaStore()
	}

	s.initSchedules()
	return s
}

func (s *Scheduler) now() time.Time {
	if s.nowFn != nil {
		return s.nowFn()
	}
	return time.Now()
}

func (s *Scheduler) initSchedules() {
	if s.registry == nil {
		return
	}
	names := s.registry.Names()
	now := s.now()

	for i, name := range names {
		p, ok := s.registry.Get(name)
		if !ok || p.Quota == nil {
			continue
		}
		policy, hasPolicy := s.cfg.ProviderPolicies[name]
		if !hasPolicy {
			policy = s.cfg.DefaultPolicy
		}

		stagger := time.Duration(i) * policy.StaggerDelay
		s.schedules[name] = &providerSchedule{
			policy:     policy,
			nextPollAt: now.Add(stagger),
		}
	}
}

// Store returns the underlying QuotaStore.
func (s *Scheduler) Store() *QuotaStore {
	return s.store
}

// PollProvider polls a single named provider immediately, bypassing scheduled wait.
func (s *Scheduler) PollProvider(ctx context.Context, name string) error {
	if s.registry == nil {
		return fmt.Errorf("scheduler: registry is nil")
	}
	p, ok := s.registry.Get(name)
	if !ok || p.Quota == nil {
		return fmt.Errorf("scheduler: provider %q has no QuotaProvider", name)
	}

	// Acquire concurrency budget
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		return ctx.Err()
	}

	s.mu.Lock()
	sched, exists := s.schedules[name]
	if !exists {
		policy, hasPolicy := s.cfg.ProviderPolicies[name]
		if !hasPolicy {
			policy = s.cfg.DefaultPolicy
		}
		sched = &providerSchedule{policy: policy}
		s.schedules[name] = sched
	}
	s.mu.Unlock()

	pollCtx, cancel := context.WithTimeout(ctx, sched.policy.Deadline)
	defer cancel()

	snap, err := p.Quota.Quota(pollCtx)
	now := s.now()

	s.mu.Lock()
	defer s.mu.Unlock()

	sched.lastPolledAt = now

	if err != nil {
		sched.failures++
		backoff := s.computeBackoff(sched.policy, sched.failures)
		sched.nextPollAt = now.Add(backoff)

		s.store.SetLastRefreshError(err)

		if s.sm != nil {
			s.sm.Update(func(rs *state.RuntimeSnapshot) {
				pr, ok := rs.Providers[name]
				if !ok {
					pr = state.ProviderRuntime{Admin: state.AdminEnabled}
				}
				pr.ObservedAt = now
				pr.Error = err.Error()
				if sched.failures >= 5 {
					pr.Health = state.HealthUnavailable
				} else if sched.failures >= 2 {
					pr.Health = state.HealthDegraded
				}
				// Admin NEVER touched!
				rs.Providers[name] = pr
			})
		}
		return err
	}

	// Success
	sched.failures = 0
	interval := s.computeIntervalWithJitter(sched.policy)
	sched.nextPollAt = now.Add(interval)

	if snap.ObservedAt.IsZero() {
		snap.ObservedAt = now
	}
	s.store.Set(name, snap)
	s.store.SetLastRefreshError(nil)

	// Persist observations to storage
	if s.writer != nil {
		for _, w := range snap.Windows {
			var resetAtStr string
			if !w.ResetAt.IsZero() {
				resetAtStr = w.ResetAt.Format(time.RFC3339)
			}
			_ = s.writer.TrackQuotaObservation(storage.QuotaObservationRecord{
				Provider:   name,
				Label:      w.Label,
				UsedPct:    w.UsedPct,
				Remaining:  w.Remaining,
				Limit:      w.Limit,
				Unit:       w.Unit,
				ResetAt:    resetAtStr,
				ObservedAt: snap.ObservedAt.Format(time.RFC3339),
				Source:     "scheduler",
			})
		}
	}

	// Update StateManager
	if s.sm != nil {
		qState := computeQuotaState(snap)
		s.sm.Update(func(rs *state.RuntimeSnapshot) {
			pr, ok := rs.Providers[name]
			if !ok {
				pr = state.ProviderRuntime{Admin: state.AdminEnabled}
			}
			pr.ObservedAt = snap.ObservedAt
			pr.Source = "scheduler"
			pr.Error = ""
			pr.Quota = qState
			pr.Health = state.HealthHealthy
			// Admin NEVER touched!
			rs.Providers[name] = pr
		})
	}

	return nil
}

// PollAll polls all registered QuotaProviders in parallel up to concurrency budget.
func (s *Scheduler) PollAll(ctx context.Context) error {
	if s.registry == nil {
		return nil
	}

	now := s.now()
	s.store.SetRefreshing(true, &now)
	defer s.store.SetRefreshing(false, nil)

	names := s.registry.Names()
	var wg sync.WaitGroup
	var firstErr error
	var errMu sync.Mutex

	for _, name := range names {
		p, ok := s.registry.Get(name)
		if !ok || p.Quota == nil {
			continue
		}

		providerName := name
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := s.PollProvider(ctx, providerName); err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				errMu.Unlock()
			}
		}()
	}

	wg.Wait()
	return firstErr
}

// Tick evaluates all scheduled providers against currentTime, polling those due.
// Useful for fake-clock deterministic unit tests.
func (s *Scheduler) Tick(ctx context.Context, currentTime time.Time) error {
	s.mu.Lock()
	s.nowFn = func() time.Time { return currentTime }

	var dueProviders []string
	for name, sched := range s.schedules {
		if !currentTime.Before(sched.nextPollAt) {
			dueProviders = append(dueProviders, name)
		}
	}
	s.mu.Unlock()

	var firstErr error
	for _, name := range dueProviders {
		if err := s.PollProvider(ctx, name); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// Start launches the background scheduler loop.
func (s *Scheduler) Start(ctx context.Context) {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return
	}
	s.running = true
	s.stopChan = make(chan struct{})
	s.doneChan = make(chan struct{})
	s.mu.Unlock()

	go s.loop(ctx)
}

// Stop terminates the scheduler background loop.
func (s *Scheduler) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	s.running = false
	close(s.stopChan)
	s.mu.Unlock()

	<-s.doneChan
}

func (s *Scheduler) loop(ctx context.Context) {
	defer close(s.doneChan)
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-s.stopChan:
			return
		case t := <-ticker.C:
			_ = s.Tick(ctx, t)
		}
	}
}

func (s *Scheduler) computeBackoff(p PollPolicy, failures int) time.Duration {
	if failures <= 0 {
		return p.Interval
	}
	factor := p.BackoffFactor
	if factor <= 1.0 {
		factor = 2.0
	}
	backoff := float64(p.InitialBackoff) * math.Pow(factor, float64(failures-1))
	if backoff > float64(p.MaxBackoff) {
		backoff = float64(p.MaxBackoff)
	}
	return time.Duration(backoff)
}

func (s *Scheduler) computeIntervalWithJitter(p PollPolicy) time.Duration {
	interval := p.Interval
	if p.JitterRatio > 0 {
		r := s.randFn()
		// Jitter between [1 - JitterRatio, 1 + JitterRatio]
		factor := 1.0 + (r*2-1)*p.JitterRatio
		interval = time.Duration(float64(interval) * factor)
	}
	return interval
}

func computeQuotaState(snap *provider.QuotaSnapshot) state.QuotaState {
	if snap == nil || len(snap.Windows) == 0 {
		return state.QuotaHealthy
	}

	var worstUsedPct float64
	for _, w := range snap.Windows {
		if w.UsedPct > worstUsedPct {
			worstUsedPct = w.UsedPct
		}
		if w.Limit > 0 && w.Remaining <= 0 {
			return state.QuotaExhausted
		}
	}

	if worstUsedPct >= 100.0 {
		return state.QuotaExhausted
	} else if worstUsedPct >= 80.0 {
		return state.QuotaLow
	}
	return state.QuotaHealthy
}
