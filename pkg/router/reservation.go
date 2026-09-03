package router

import (
	"errors"
	"math"
	"sync"
	"sync/atomic"
)

var (
	ErrHeadroomExceeded    = errors.New("router: effective headroom exceeded")
	ErrConcurrencyExceeded = errors.New("router: provider concurrency budget exceeded")
)

// ProviderCounters tracks atomic in-flight token and slot usage for a provider.
type ProviderCounters struct {
	tokens atomic.Int64
	slots  atomic.Int64
}

// ReservationManager manages in-flight reservations across providers.
type ReservationManager struct {
	mu       sync.RWMutex
	counters map[string]*ProviderCounters
}

// NewReservationManager constructs an empty ReservationManager.
func NewReservationManager() *ReservationManager {
	return &ReservationManager{
		counters: make(map[string]*ProviderCounters),
	}
}

func (rm *ReservationManager) getCounters(providerID string) *ProviderCounters {
	rm.mu.RLock()
	c, ok := rm.counters[providerID]
	rm.mu.RUnlock()
	if ok {
		return c
	}

	rm.mu.Lock()
	defer rm.mu.Unlock()
	c, ok = rm.counters[providerID]
	if !ok {
		c = &ProviderCounters{}
		rm.counters[providerID] = c
	}
	return c
}

// ReservedTokens returns current in-flight tokens for a provider.
func (rm *ReservationManager) ReservedTokens(providerID string) int64 {
	c := rm.getCounters(providerID)
	return c.tokens.Load()
}

// ReservedSlots returns current in-flight slots/requests for a provider.
func (rm *ReservationManager) ReservedSlots(providerID string) int64 {
	c := rm.getCounters(providerID)
	return c.slots.Load()
}

// EffectiveHeadroom computes: observed_remaining - reserved - safety_margin.
func (rm *ReservationManager) EffectiveHeadroom(providerID string, observedRemaining float64, safetyMargin float64) float64 {
	reserved := float64(rm.ReservedSlots(providerID))
	return observedRemaining - reserved - safetyMargin
}

// EffectiveTokenHeadroom computes token-level effective headroom.
func (rm *ReservationManager) EffectiveTokenHeadroom(providerID string, observedRemainingTokens float64, safetyMargin float64) float64 {
	reserved := float64(rm.ReservedTokens(providerID))
	return observedRemainingTokens - reserved - safetyMargin
}

// TryReserve atomically checks headroom and reserves tokens and a concurrency slot.
func (rm *ReservationManager) TryReserve(
	providerID string,
	tokens int64,
	slots int64,
	observedRemaining float64,
	safetyMargin float64,
	maxConcurrency int,
) (*Reservation, error) {
	c := rm.getCounters(providerID)

	rm.mu.Lock()
	defer rm.mu.Unlock()

	currentSlots := c.slots.Load()
	if maxConcurrency > 0 && currentSlots+slots > int64(maxConcurrency) {
		return nil, ErrConcurrencyExceeded
	}

	// Check headroom if observedRemaining is not infinite
	if observedRemaining != math.MaxFloat64 {
		// effective_headroom = observed_remaining - reserved - safety_margin
		effectiveHeadroom := observedRemaining - float64(currentSlots) - safetyMargin
		if effectiveHeadroom < float64(slots) {
			return nil, ErrHeadroomExceeded
		}
	}

	c.tokens.Add(tokens)
	c.slots.Add(slots)

	res := &Reservation{
		provider: providerID,
		tokens:   tokens,
		slots:    slots,
		mgr:      rm,
	}
	return res, nil
}

// Reservation represents an active in-flight resource allocation.
type Reservation struct {
	provider string
	tokens   int64
	slots    int64
	mgr      *ReservationManager
	released atomic.Bool
}

// Provider returns the reserved provider ID.
func (r *Reservation) Provider() string {
	return r.provider
}

// Tokens returns the estimated token allocation.
func (r *Reservation) Tokens() int64 {
	return r.tokens
}

// Slots returns the reserved slot count.
func (r *Reservation) Slots() int64 {
	return r.slots
}

// Release returns reserved resources to the pool without recording usage.
// Protected against underflow and idempotent.
func (r *Reservation) Release() {
	if r == nil || !r.released.CompareAndSwap(false, true) {
		return
	}
	r.mgr.release(r.provider, r.tokens, r.slots)
}

// Commit releases the in-flight reservation and optionally accounts for actual tokens.
// Protected against underflow and idempotent.
func (r *Reservation) Commit(actualTokens int64) {
	if r == nil || !r.released.CompareAndSwap(false, true) {
		return
	}
	r.mgr.release(r.provider, r.tokens, r.slots)
}

func (rm *ReservationManager) release(providerID string, tokens int64, slots int64) {
	c := rm.getCounters(providerID)

	// Underflow protection for tokens
	for {
		cur := c.tokens.Load()
		next := cur - tokens
		if next < 0 {
			next = 0
		}
		if c.tokens.CompareAndSwap(cur, next) {
			break
		}
	}

	// Underflow protection for slots
	for {
		cur := c.slots.Load()
		next := cur - slots
		if next < 0 {
			next = 0
		}
		if c.slots.CompareAndSwap(cur, next) {
			break
		}
	}
}

// ForceReleaseAll resets counters for a provider (useful in tests/cleanups).
func (rm *ReservationManager) ForceReleaseAll(providerID string) {
	c := rm.getCounters(providerID)
	c.tokens.Store(0)
	c.slots.Store(0)
}
