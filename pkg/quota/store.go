package quota

import (
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

// QuotaStore provides thread-safe in-memory caching of QuotaSnapshots and refresh metadata.
type QuotaStore struct {
	mu                 sync.RWMutex
	snapshots          map[string]*provider.QuotaSnapshot
	refreshing         bool
	refreshStartedAt   *time.Time
	refreshMinInterval time.Duration
	lastRefreshError   *string
	lastFetchedAt      time.Time
}

// NewQuotaStore initializes an empty QuotaStore.
func NewQuotaStore() *QuotaStore {
	return &QuotaStore{
		snapshots:          make(map[string]*provider.QuotaSnapshot),
		refreshMinInterval: 30 * time.Second,
	}
}

// Get returns the latest QuotaSnapshot for a provider.
func (s *QuotaStore) Get(name string) (*provider.QuotaSnapshot, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	snap, ok := s.snapshots[name]
	return snap, ok
}

// Set stores the QuotaSnapshot for a provider.
func (s *QuotaStore) Set(name string, snap *provider.QuotaSnapshot) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshots[name] = snap
	if snap != nil && !snap.ObservedAt.IsZero() {
		if snap.ObservedAt.After(s.lastFetchedAt) {
			s.lastFetchedAt = snap.ObservedAt
		}
	} else {
		s.lastFetchedAt = time.Now()
	}
}

// All returns a copy of all current snapshots.
func (s *QuotaStore) All() map[string]*provider.QuotaSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]*provider.QuotaSnapshot, len(s.snapshots))
	for k, v := range s.snapshots {
		out[k] = v
	}
	return out
}

// SetRefreshing marks whether a refresh cycle is in flight.
func (s *QuotaStore) SetRefreshing(refreshing bool, startedAt *time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refreshing = refreshing
	s.refreshStartedAt = startedAt
}

// SetLastRefreshError records the error from the most recent refresh.
func (s *QuotaStore) SetLastRefreshError(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err == nil {
		s.lastRefreshError = nil
	} else {
		msg := err.Error()
		s.lastRefreshError = &msg
	}
}

// Metadata returns current refresh bookkeeping.
func (s *QuotaStore) Metadata() (refreshing bool, startedAt *time.Time, minInterval time.Duration, lastErr *string, lastFetched time.Time) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.refreshing, s.refreshStartedAt, s.refreshMinInterval, s.lastRefreshError, s.lastFetchedAt
}

// SetLastFetchedAt sets the last fetch time explicitly (useful in tests).
func (s *QuotaStore) SetLastFetchedAt(t time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastFetchedAt = t
}
