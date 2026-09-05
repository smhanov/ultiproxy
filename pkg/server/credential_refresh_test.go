package server

import (
	"bytes"

	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// xaiClientID is openaicompat's compiled-in xAI OAuth client id (the credential
// key the OAuth manager token source uses). It is duplicated here because the
// constant is unexported; the value is public knowledge (the xAI CLI ships it).
const xaiClientID = "b1a00492-073a-47ea-816f-4c329264a828"

// fakeCredentialClock is the fake clock the proactive refresher and the
// credential manager both run on, so a test can hold a credential "one minute
// from expiry" for as long as it likes.
type fakeCredentialClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeCredentialClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeCredentialClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	c.mu.Unlock()
}

// fakeCredentialTicker implements modelTicker over a channel the test fires.
type fakeCredentialTicker struct {
	ch chan time.Time
}

func (f fakeCredentialTicker) C() <-chan time.Time { return f.ch }
func (f fakeCredentialTicker) Stop()               {}

// countingUpstream is the lane's upstream. The proactive refresher must never
// talk to it: every request it sees is a violation of "no inbound request, no
// upstream dial".
type countingUpstream struct {
	srv   *httptest.Server
	calls atomic.Int64
}

func newCountingUpstream(t *testing.T) *countingUpstream {
	t.Helper()
	u := &countingUpstream{}
	u.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u.calls.Add(1)
		http.NotFound(w, r)
	}))
	t.Cleanup(u.srv.Close)
	return u
}

// expiringLane builds a real openaicompat lane over a real auth.Manager whose
// refresher mints a fresh credential from the fake clock, so a proactive
// refresh is observable as a generation bump in the persisted file.
type expiringLane struct {
	provider  *openaicompat.Provider
	manager   *auth.Manager
	clock     *fakeCredentialClock
	refreshes *atomic.Int64
}

func newExpiringLane(t *testing.T, name, baseURL string, client *http.Client, clock *fakeCredentialClock, refreshes *atomic.Int64) *expiringLane {
	t.Helper()
	dir := t.TempDir()
	mgr, err := auth.NewManager(dir, func(ctx context.Context, cred auth.Credential) (auth.Credential, error) {
		n := refreshes.Load() + 1
		refreshes.Store(n)
		return auth.Credential{
			Provider:     "xai",
			AccessToken:  fmt.Sprintf("token-%d", n),
			RefreshToken: fmt.Sprintf("rt-%d", n),
			// Six hours of new life, measured on the fake clock.
			ExpiresAt:  clock.Now().Add(6 * time.Hour),
			Generation: cred.Generation + 1,
			ClientID:   xaiClientID,
		}, nil
	}, auth.WithNow(clock.Now))
	if err != nil {
		t.Fatalf("auth.NewManager: %v", err)
	}

	p, err := openaicompat.New(openaicompat.Config{
		Name:        name,
		BaseURL:     baseURL,
		HTTPClient:  client,
		DataDir:     dir,
		TokenSource: openaicompat.NewOAuthManagerTokenSource(mgr, xaiClientID),
		// Discovery would dial the upstream at construction time; the
		// assertion under test is "zero upstream calls", so opt out.
		OptOutModelListPassthrough: true,
		Quirks:                     openaicompat.Quirks{AuthViaOAuthManager: true},
	})
	if err != nil {
		t.Fatalf("openaicompat.New(%q): %v", name, err)
	}
	return &expiringLane{provider: p, manager: mgr, clock: clock, refreshes: refreshes}
}

// seed stores a credential expiring d from the fake clock's now, so the test
// controls exactly how close to death the lane is.
func (l *expiringLane) seed(t *testing.T, d time.Duration) {
	t.Helper()
	if err := l.manager.Store(context.Background(), xaiClientID, auth.Credential{
		Provider:    "xai",
		AccessToken: "token-0",
		ExpiresAt:   l.clock.Now().Add(d),
		Generation:  1,
		ClientID:    xaiClientID,
	}); err != nil {
		t.Fatalf("seed credential: %v", err)
	}
}

// persisted reads the credential file back from disk.
func (l *expiringLane) persisted(t *testing.T) auth.Credential {
	t.Helper()
	cred, ok := l.persistedCredential()
	if !ok {
		t.Fatalf("no persisted credential for %q", xaiClientID)
	}
	return cred
}

// persistedCredential reports the on-disk credential without failing when it is
// not there (yet), so a test can poll for the refresh to land durably - the
// refresh only counts once auth.Manager has persisted and published it.
func (l *expiringLane) persistedCredential() (auth.Credential, bool) {
	cred, err := l.manager.LoadFromDisk(xaiClientID)
	if err != nil {
		return auth.Credential{}, false
	}
	return cred, true
}

// waitForGeneration blocks until the persisted credential carries the wanted
// generation.
func (l *expiringLane) waitForGeneration(t *testing.T, what string, gen uint64) {
	t.Helper()
	waitFor(t, what, func() bool {
		cred, ok := l.persistedCredential()
		return ok && cred.Generation == gen
	})
}

// newCredentialRefreshServer wires a server whose credential-refresher schedule
// is driven by a fake ticker on a fake clock.
func newCredentialRefreshServer(t *testing.T, registry *provider.Registry, interval time.Duration, clock *fakeCredentialClock, tick chan time.Time, factoryCalled *atomic.Bool) *Server {
	t.Helper()
	cfg := newDiscoveryTestConfig(t)
	return NewServer(cfg, registry,
		WithCredentialRefresh(interval),
		withCredentialClock(clock.Now),
		withCredentialTickerFactory(func(d time.Duration) modelTicker {
			if factoryCalled != nil {
				factoryCalled.Store(true)
			}
			return fakeCredentialTicker{ch: tick}
		}),
	)
}

// fireTick delivers one tick. With no refresh loop running (a disabled or
// not-yet-started schedule) nothing consumes the channel, and a test must fail
// with a clear message instead of blocking forever on the send.
func fireTick(t *testing.T, tick chan time.Time, now time.Time) {
	t.Helper()
	select {
	case tick <- now:
	case <-time.After(5 * time.Second):
		t.Fatalf("tick was never consumed: no credential refresh loop is running")
	}
}

// AC1: a credential one minute from expiry is refreshed by the background
// refresher when its tick fires - with NO inbound client request, NO upstream
// dial, and the refreshed credential durably persisted (generation + expiry
// bumped in the on-disk file).
func TestCredentialRefresh_ProactiveRefreshBeforeExpiry(t *testing.T) {
	clock := &fakeCredentialClock{now: time.Date(2026, 9, 6, 10, 0, 0, 0, time.UTC)}
	up := newCountingUpstream(t)
	var refreshes atomic.Int64
	lane := newExpiringLane(t, "xai", up.srv.URL, up.srv.Client(), clock, &refreshes)
	lane.seed(t, time.Minute) // one minute of life left

	registry := provider.NewRegistry()
	registry.Register(lane.provider.ProviderBundle())

	tick := make(chan time.Time)
	srv := newCredentialRefreshServer(t, registry, DefaultCredentialRefreshInterval, clock, tick, nil)
	defer func() { _ = srv.Shutdown(context.Background()) }()

	if got := refreshes.Load(); got != 0 {
		t.Fatalf("refreshes before the tick = %d, want 0", got)
	}

	fireTick(t, tick, clock.Now())
	lane.waitForGeneration(t, "proactive refresh to persist", 2)
	if got := refreshes.Load(); got != 1 {
		t.Errorf("credential refreshes = %d, want 1", got)
	}

	cred := lane.persisted(t)
	if cred.Generation != 2 {
		t.Errorf("persisted generation = %d, want 2 (bumped by the proactive refresh)", cred.Generation)
	}
	if cred.AccessToken != "token-1" {
		t.Errorf("persisted access token = %q, want the refreshed credential", cred.AccessToken)
	}
	wantExpiry := clock.Now().Add(6 * time.Hour)
	if !cred.ExpiresAt.Equal(wantExpiry) {
		t.Errorf("persisted expiry = %v, want %v (bumped by the proactive refresh)", cred.ExpiresAt, wantExpiry)
	}
	if got := up.calls.Load(); got != 0 {
		t.Errorf("upstream requests during a proactive refresh = %d, want 0 (no Generate/Stream call happened)", got)
	}
}

// AC1 lead window: the refresher acts one full lead window (10 min) ahead of
// expiry - and ONLY then. A credential 7 minutes out (inside the lead window
// but well outside the manager's own lazy 5-minute window) is refreshed; one
// with hours left is left alone; the same lane refreshed on one tick is not
// refreshed again on the next.
func TestCredentialRefresh_LeadWindow(t *testing.T) {
	clock := &fakeCredentialClock{now: time.Date(2026, 9, 6, 10, 0, 0, 0, time.UTC)}
	up := newCountingUpstream(t)
	var soonRefreshes, muchRefreshes atomic.Int64
	soon := newExpiringLane(t, "xai", up.srv.URL, up.srv.Client(), clock, &soonRefreshes)
	much := newExpiringLane(t, "other", up.srv.URL, up.srv.Client(), clock, &muchRefreshes)
	soon.seed(t, 7*time.Minute) // inside the 10-minute lead, outside the lazy 5-minute window
	much.seed(t, 2*time.Hour)

	registry := provider.NewRegistry()
	registry.Register(soon.provider.ProviderBundle())
	registry.Register(much.provider.ProviderBundle())

	tick := make(chan time.Time)
	srv := newCredentialRefreshServer(t, registry, DefaultCredentialRefreshInterval, clock, tick, nil)
	defer func() { _ = srv.Shutdown(context.Background()) }()

	fireTick(t, tick, clock.Now())
	soon.waitForGeneration(t, "lead-window refresh to persist", 2)

	if got := muchRefreshes.Load(); got != 0 {
		t.Errorf("lane with 2h of life left was refreshed %d times, want 0", got)
	}
	if cred := soon.persisted(t); cred.Generation != 2 {
		t.Errorf("soon lane generation = %d, want 2", cred.Generation)
	}
	if cred := much.persisted(t); cred.Generation != 1 {
		t.Errorf("healthy lane generation = %d, want 1 (untouched)", cred.Generation)
	}

	// The refreshed credential is good for another 6h on the fake clock: a
	// second tick must not refresh it again.
	clock.Advance(5 * time.Minute)
	fireTick(t, tick, clock.Now())
	waitFor(t, "second tick to settle", func() bool {
		// Nothing observable changes; give the loop a moment to (not) act.
		time.Sleep(20 * time.Millisecond)
		return true
	})
	if got := soonRefreshes.Load(); got != 1 {
		t.Errorf("refreshes after a second tick = %d, want 1 (a freshly refreshed lane is left alone)", got)
	}

	// 5h55m later the refreshed credential is back inside the lead window and
	// the schedule refreshes it again - the "overnight idle" case. The other
	// lane's 2h credential is long dead by then, so the same round heals it.
	clock.Advance(5*time.Hour + 55*time.Minute)
	fireTick(t, tick, clock.Now())
	soon.waitForGeneration(t, "second expiry-window refresh to persist", 3)
	much.waitForGeneration(t, "expired lane to be healed by the same round", 2)
	if got := soonRefreshes.Load(); got != 2 {
		t.Errorf("refreshes after the second expiry window = %d, want 2", got)
	}
	if got := muchRefreshes.Load(); got != 1 {
		t.Errorf("expired lane refreshes = %d, want 1", got)
	}
}

// WithCredentialRefresh(0) disables the schedule outright: no ticker is ever
// built and an expiring credential is never refreshed in the background.
func TestCredentialRefresh_DisabledWithZero(t *testing.T) {
	clock := &fakeCredentialClock{now: time.Now().UTC()}
	up := newCountingUpstream(t)
	var refreshes atomic.Int64
	lane := newExpiringLane(t, "xai", up.srv.URL, up.srv.Client(), clock, &refreshes)
	lane.seed(t, time.Minute)

	registry := provider.NewRegistry()
	registry.Register(lane.provider.ProviderBundle())

	var factoryCalled atomic.Bool
	tick := make(chan time.Time)
	srv := newCredentialRefreshServer(t, registry, 0, clock, tick, &factoryCalled)
	defer func() { _ = srv.Shutdown(context.Background()) }()

	if factoryCalled.Load() {
		t.Error("a refresh ticker was built despite WithCredentialRefresh(0)")
	}
	if srv.credentialRefreshCancel != nil {
		t.Error("a refresh loop was started despite WithCredentialRefresh(0)")
	}

	// A credential one minute from expiry stays exactly as stored: with the
	// schedule disabled the only refresh path left is a lazy one on request.
	clock.Advance(30 * time.Second)
	if got := refreshes.Load(); got != 0 {
		t.Errorf("background refreshes with the refresher disabled = %d, want 0", got)
	}
	if cred := lane.persisted(t); cred.Generation != 1 {
		t.Errorf("generation = %d, want 1 (untouched)", cred.Generation)
	}
}

// AC3: refresh log lines carry the lane name and the new expiry only. A grep
// over everything logged during a proactive refresh must find no token-shaped
// string - not the access token, not the refresh token, no bearer headers, no
// JWT, no long secret-looking run.
func TestCredentialRefresh_LogsNoSecrets(t *testing.T) {
	clock := &fakeCredentialClock{now: time.Date(2026, 9, 6, 10, 0, 0, 0, time.UTC)}
	up := newCountingUpstream(t)
	var refreshes atomic.Int64
	lane := newExpiringLane(t, "xai", up.srv.URL, up.srv.Client(), clock, &refreshes)
	lane.seed(t, time.Minute)

	registry := provider.NewRegistry()
	registry.Register(lane.provider.ProviderBundle())

	tick := make(chan time.Time)
	srv := newCredentialRefreshServer(t, registry, DefaultCredentialRefreshInterval, clock, tick, nil)
	defer func() { _ = srv.Shutdown(context.Background()) }()

	var buf syncLogBuffer
	log.SetOutput(&buf)
	log.SetFlags(0)
	defer func() {
		log.SetOutput(os.Stderr)
		log.SetFlags(log.LstdFlags)
	}()

	fireTick(t, tick, clock.Now())
	lane.waitForGeneration(t, "proactive refresh to persist", 2)
	if got := refreshes.Load(); got != 1 {
		t.Errorf("credential refreshes = %d, want 1", got)
	}

	assertNoTokenShapedStrings(t, buf.String(), "proactive refresh logs")
	if !strings.Contains(buf.String(), "lane=xai") {
		t.Errorf("refresh log lines do not name the lane; log was:\n%s", buf.String())
	}
}

// syncLogBuffer is a mutex-guarded log sink: the background refresher and the
// discovery loop log from their own goroutines while the test reads.
type syncLogBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncLogBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncLogBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// tokenShapedPatterns are the shapes a leaked credential takes in a log line.
var tokenShapedPatterns = []*regexp.Regexp{
	regexp.MustCompile(`eyJ[A-Za-z0-9_-]{8,}`),                // JWT
	regexp.MustCompile(`(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}`), // Authorization header
	regexp.MustCompile(`sk-[A-Za-z0-9]{8,}`),                  // API key
	regexp.MustCompile(`[0-9a-f]{32,}`),                       // long hex
	regexp.MustCompile(`[A-Za-z0-9+/]{40,}={0,2}`),            // long base64
}

// assertNoTokenShapedStrings fails when any log line mentions a secret that was
// in play, or matches a generic token shape.
func assertNoTokenShapedStrings(t *testing.T, logged, what string) {
	t.Helper()
	secrets := []string{"token-0", "token-1", "rt-0", "rt-1", "supabase-access"}
	for _, line := range strings.Split(logged, "\n") {
		if !strings.Contains(line, "refresh") {
			continue
		}
		for _, secret := range secrets {
			if strings.Contains(line, secret) {
				t.Errorf("%s leak: line %q contains the in-play secret %q", what, line, secret)
			}
		}
		for _, re := range tokenShapedPatterns {
			if re.MatchString(line) {
				t.Errorf("%s leak: line %q matches token shape %v", what, line, re)
			}
		}
	}
}

// A lane whose kind has no credential surface (a static-key lane, no token
// source) is skipped by the refresher instead of erroring.
func TestCredentialRefresh_SkipsLanesWithoutACredential(t *testing.T) {
	clock := &fakeCredentialClock{now: time.Now().UTC()}
	up := newCountingUpstream(t)

	static, err := openaicompat.New(openaicompat.Config{
		Name:                       "plain",
		BaseURL:                    up.srv.URL,
		APIKey:                     "sk-static-key-value",
		HTTPClient:                 up.srv.Client(),
		OptOutModelListPassthrough: true,
	})
	if err != nil {
		t.Fatalf("openaicompat.New(static): %v", err)
	}

	registry := provider.NewRegistry()
	registry.Register(static.ProviderBundle())

	tick := make(chan time.Time)
	srv := newCredentialRefreshServer(t, registry, DefaultCredentialRefreshInterval, clock, tick, nil)
	defer func() { _ = srv.Shutdown(context.Background()) }()

	// Nothing to assert except that firing the tick neither panics nor dials.
	fireTick(t, tick, clock.Now())
	waitFor(t, "tick to settle", func() bool { time.Sleep(20 * time.Millisecond); return true })
	if got := up.calls.Load(); got != 0 {
		t.Errorf("upstream requests = %d, want 0", got)
	}
}

// The refresher must stop with the server: after Shutdown a delivered tick is
// left unconsumed in the ticker channel and refreshes nothing.
func TestCredentialRefresh_ShutdownStopsLoop(t *testing.T) {
	clock := &fakeCredentialClock{now: time.Now().UTC()}
	up := newCountingUpstream(t)
	var refreshes atomic.Int64
	lane := newExpiringLane(t, "xai", up.srv.URL, up.srv.Client(), clock, &refreshes)
	lane.seed(t, time.Minute)

	registry := provider.NewRegistry()
	registry.Register(lane.provider.ProviderBundle())

	// Buffered on purpose: a tick nobody consumes stays readable, which is how
	// the test proves the loop is gone.
	tick := make(chan time.Time, 1)
	srv := newCredentialRefreshServer(t, registry, DefaultCredentialRefreshInterval, clock, tick, nil)
	if srv.credentialRefreshCancel == nil {
		t.Fatal("no credential refresh loop was started")
	}

	// First prove the loop is live, so the second half cannot pass just
	// because the goroutine had not started yet.
	fireTick(t, tick, clock.Now())
	lane.waitForGeneration(t, "first refresh", 2)
	if got := refreshes.Load(); got != 1 {
		t.Fatalf("refreshes before Shutdown = %d, want 1", got)
	}

	if err := srv.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	// The refreshed credential is 6h out on the fake clock; pull the clock
	// forward so a live loop WOULD refresh again, then deliver a tick. The
	// channel is buffered and nobody should take it, so this send must not
	// block (and must not be consumed).
	clock.Advance(5 * time.Hour)
	tick <- clock.Now()
	waitFor(t, "loop to be gone", func() bool { time.Sleep(50 * time.Millisecond); return true })

	// The tick is still sitting in the channel: nothing consumed it.
	select {
	case <-tick:
	default:
		t.Fatal("the tick was consumed after Shutdown; the refresh schedule is still live")
	}
	if got := refreshes.Load(); got != 1 {
		t.Errorf("refreshes after Shutdown = %d, want 1 (no further refresh)", got)
	}
	if cred := lane.persisted(t); cred.Generation != 2 {
		t.Errorf("generation after Shutdown = %d, want 2 (untouched)", cred.Generation)
	}
}
