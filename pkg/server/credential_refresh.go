package server

import (
	"context"
	"log"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

// Automatic credential refresh (the proactive half).
//
// A lazily-refreshed credential dies while nobody is asking for it: auth.Manager
// refreshes on Get, Get only runs on the request path, so an idle lane wakes up
// with an expired OAuth token and 401/403s - exactly the overnight xai failure.
// This loop closes that gap:
//
//   - every DefaultCredentialRefreshInterval (WithCredentialRefresh, 0 disables)
//     it walks the registry and asks each lane when its credential expires;
//   - a lane inside the CredentialRefreshLead window (10 minutes of life left)
//     is refreshed RIGHT THEN, with no inbound request at all;
//   - the refresh action is the lane's own provider.AuthProvider.Refresh, so
//     singleflight coordination and atomic persistence stay in one place;
//   - the reactive half (a 401/403 answered by one invalidate + retry) lives in
//     the openaicompat request path, strictly before the first byte.
//
// Lanes without a credential surface (a static api_key, or a token source with
// no expiry) are skipped, not refreshed and not logged about.

const (
	// DefaultCredentialRefreshInterval is how often the refresher walks the
	// registry looking for soon-to-expire credentials.
	DefaultCredentialRefreshInterval = 5 * time.Minute

	// CredentialRefreshLead is how far ahead of expiry a credential is
	// refreshed. It must stay comfortably above the tick interval: with a 5m
	// tick and a 10m lead, a credential is seen at least once before it dies.
	// It is deliberately wider than auth.Manager's own lazy 5-minute window,
	// which is what makes an idle lane refresh before its token is even close.
	CredentialRefreshLead = 10 * time.Minute

	// credentialRefreshBudget bounds one lane's refresh call so a hung token
	// endpoint delays neither the rest of the round nor shutdown.
	credentialRefreshBudget = 15 * time.Second
)

// credentialExpirer is the optional lane surface that reports when the lane's
// credential expires (openaicompat.Provider implements it over its token
// source). It is asserted off bundle.Inference so this package stays
// independent of the concrete adapter packages, and so custom kinds can adopt
// it later without touching the refresher.
type credentialExpirer interface {
	TokenExpiresAt() (time.Time, bool)
}

// credentialRefresher refreshes a lane's credential (provider.AuthProvider).
type credentialRefresher interface {
	Refresh(ctx context.Context) error
}

// withCredentialClock is the (unexported, test-only) hook that replaces the
// clock the lead-window decision runs on, so tests can hold a credential one
// minute from expiry for as long as they need.
func withCredentialClock(now func() time.Time) Option {
	return func(s *Server) {
		s.credentialNow = now
	}
}

// withCredentialTickerFactory is the (unexported, test-only) hook that replaces
// the real ticker with a fake clock-driven one.
func withCredentialTickerFactory(factory func(d time.Duration) modelTicker) Option {
	return func(s *Server) {
		s.newCredentialTicker = factory
	}
}

// WithCredentialRefresh sets how often the proactive credential refresher walks
// the registry. Zero disables the schedule entirely (credentials are then
// refreshed only lazily on request, plus the reactive retry); a negative value
// restores the default, DefaultCredentialRefreshInterval.
func WithCredentialRefresh(d time.Duration) Option {
	return func(s *Server) {
		s.credentialRefreshInterval = d
	}
}

// startCredentialRefresh launches the background refresh loop. The loop owns
// its context; Shutdown cancels it, and a disabled schedule (interval 0) never
// starts a goroutine at all.
func (s *Server) startCredentialRefresh() {
	if s == nil || s.registry == nil {
		return
	}
	interval := s.credentialRefreshInterval
	if interval < 0 {
		interval = DefaultCredentialRefreshInterval
	}
	if interval <= 0 {
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	s.credentialRefreshCancel = cancel

	now := s.credentialNow
	if now == nil {
		now = time.Now
	}
	newTicker := s.newCredentialTicker
	if newTicker == nil {
		newTicker = func(d time.Duration) modelTicker {
			return realModelTicker{ticker: time.NewTicker(d)}
		}
	}

	log.Printf("[auth] credential refresher started interval=%s lead=%s", interval, CredentialRefreshLead)

	go func() {
		ticker := newTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C():
				s.refreshExpiringCredentials(ctx, now())
			}
		}
	}()
}

// refreshExpiringCredentials runs one round: every lane whose credential is
// inside the lead window gets a fresh one. Lanes that cannot report an expiry,
// or have no refresh surface, are skipped.
func (s *Server) refreshExpiringCredentials(ctx context.Context, now time.Time) {
	if s == nil || s.registry == nil {
		return
	}
	for _, name := range s.registry.Names() {
		bundle, ok := s.registry.Get(name)
		if !ok {
			continue
		}
		if !credentialNeedsRefresh(bundle, now, CredentialRefreshLead) {
			continue
		}
		s.refreshLaneCredential(ctx, name, bundle)
	}
}

// credentialNeedsRefresh reports whether the lane's credential is inside the
// lead window. Reads only: the probe must not mint a token, and a lane with no
// expiry surface (or no auth surface to refresh it with) is never a target.
func credentialNeedsRefresh(bundle provider.Provider, now time.Time, lead time.Duration) bool {
	if bundle.Inference == nil || bundle.Auth == nil {
		return false
	}
	probe, ok := bundle.Inference.(credentialExpirer)
	if !ok {
		return false
	}
	expiresAt, ok := probe.TokenExpiresAt()
	if !ok || expiresAt.IsZero() {
		return false
	}
	return expiresAt.Sub(now) < lead
}

// refreshLaneCredential refreshes ONE lane under a per-call budget and logs the
// outcome. Log lines carry the lane name and the new expiry only - never an
// access or refresh token.
func (s *Server) refreshLaneCredential(ctx context.Context, name string, bundle provider.Provider) {
	refreshCtx, cancel := context.WithTimeout(ctx, credentialRefreshBudget)
	defer cancel()

	refresher, ok := bundle.Auth.(credentialRefresher)
	if !ok {
		return
	}
	if err := refresher.Refresh(refreshCtx); err != nil {
		log.Printf("[auth] credential refresh failed lane=%s: %v", name, err)
		return
	}

	if probe, ok := bundle.Inference.(credentialExpirer); ok {
		if expiresAt, ok := probe.TokenExpiresAt(); ok && !expiresAt.IsZero() {
			log.Printf("[auth] credential refreshed lane=%s new_expiry=%s", name, expiresAt.UTC().Format(time.RFC3339))
			return
		}
	}
	log.Printf("[auth] credential refreshed lane=%s", name)
}
