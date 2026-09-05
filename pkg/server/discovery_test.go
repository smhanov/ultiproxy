package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/state"
)

// discoveryUpstream is a fake OpenAI-compatible upstream whose /v1/models
// catalog can change (and fail) while the test runs, so tests can prove what
// the discovery machinery does at registration, startup and on the schedule.
type discoveryUpstream struct {
	srv *httptest.Server

	mu     sync.Mutex
	models []string
	fail   bool

	// failFirst is the number of leading /v1/models requests that must fail,
	// so a lane can be built with an empty cache (the state this task exists
	// to heal automatically).
	failFirst atomic.Int32
	calls     atomic.Int32
}

func newDiscoveryUpstream(t *testing.T, models ...string) *discoveryUpstream {
	t.Helper()
	u := &discoveryUpstream{models: models}
	u.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		n := u.calls.Add(1)
		u.mu.Lock()
		models, fail := u.models, u.fail
		failFirst := u.failFirst.Load()
		u.mu.Unlock()
		if fail || (failFirst > 0 && n <= failFirst) {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		data := make([]map[string]any, 0, len(models))
		for _, id := range models {
			data = append(data, map[string]any{"id": id})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	}))
	t.Cleanup(u.srv.Close)
	return u
}

func (u *discoveryUpstream) URL() string { return u.srv.URL }

func (u *discoveryUpstream) setModels(models ...string) {
	u.mu.Lock()
	u.models = models
	u.mu.Unlock()
}

func (u *discoveryUpstream) setFail(fail bool) {
	u.mu.Lock()
	u.fail = fail
	u.mu.Unlock()
}

// setFailFirst makes the next n /v1/models requests fail, so a lane built right
// afterwards starts with an empty discovery cache.
func (u *discoveryUpstream) setFailFirst(n int32) { u.failFirst.Store(n) }

func (u *discoveryUpstream) modelCalls() int32 { return int32(u.calls.Load()) }

// waitFor polls cond until it holds or the deadline expires.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func newDiscoveryTestConfig(t *testing.T) *Config {
	t.Helper()
	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.Server.APIKey = ""
	cfg.Server.ClientKeys = nil
	return cfg
}

func callAddProvider(t *testing.T, srv *Server, args string) string {
	t.Helper()
	body := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add_provider","arguments":` + args + `}}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("add_provider: HTTP %d: %s", rec.Code, rec.Body.String())
	}
	var rpcResp struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &rpcResp); err != nil {
		t.Fatalf("decode add_provider response: %v (%s)", err, rec.Body.String())
	}
	if rpcResp.Result.IsError || len(rpcResp.Result.Content) == 0 {
		t.Fatalf("add_provider tool error: %s", rec.Body.String())
	}
	return rpcResp.Result.Content[0].Text
}

// AC1: a lane registered through add_provider with NO quirks discovers its
// upstream catalog before the tool replies - the reply states what was found
// and GET /v1/models serves the lane-prefixed ids with zero further upstream
// calls.
func TestDiscovery_AddProviderDiscoversAndReports(t *testing.T) {
	up := newDiscoveryUpstream(t, "deepseek-chat", "deepseek-reasoner")
	cfg := newDiscoveryTestConfig(t)
	registry := provider.NewRegistry()
	srv := NewServer(cfg, registry, WithStateManager(state.NewStateManager()))

	reply := callAddProvider(t, srv,
		`{"name":"deepseek","base_url":"`+up.URL()+`","api_key":"sk-test"}`)
	if !strings.Contains(reply, "discovered 2 models") {
		t.Errorf("add_provider reply = %q, want it to state \"discovered 2 models\"", reply)
	}
	if _, ok := registry.Get("deepseek"); !ok {
		t.Fatal("deepseek not registered")
	}
	if got := up.modelCalls(); got < 1 {
		t.Fatalf("expected discovery to hit the upstream, got %d calls", got)
	}

	resp := getModels(t, srv)
	ids := resp.ids()
	for _, want := range []string{"deepseek", "deepseek/deepseek-chat", "deepseek/deepseek-reasoner"} {
		if _, ok := ids[want]; !ok {
			t.Errorf("model %q missing from /v1/models: %v", want, ids)
		}
	}

	// Listing is cache-only: no extra upstream calls, ever.
	before := up.modelCalls()
	for i := 0; i < 3; i++ {
		getModels(t, srv)
	}
	if got := up.modelCalls(); got != before {
		t.Errorf("GET /v1/models fanned out to the upstream: %d -> %d", before, got)
	}
}

// AC2: a lane whose discovery cache is empty at startup is backfilled by the
// startup hook, without any MCP call.
func TestDiscovery_StartupBackfillsEmptyCache(t *testing.T) {
	up := newDiscoveryUpstream(t, "glm-5.3", "glm-5.3-air")
	// Construction-time discovery fails (upstream slow / daemon starting
	// before the network), so the lane comes up with an empty cache.
	up.setFailFirst(1)

	p, err := openaicompat.New(openaicompat.Config{
		Name:    "zai",
		BaseURL: up.URL(),
		APIKey:  "sk-zai",
	})
	if err != nil {
		t.Fatalf("openaicompat.New: %v", err)
	}
	if got := p.CachedModels(); len(got) != 0 {
		t.Fatalf("precondition: cache must start empty, got %v", got)
	}

	registry := provider.NewRegistry()
	registry.Register(p.Provider())

	srv := NewServer(newDiscoveryTestConfig(t), registry, WithStateManager(state.NewStateManager()))

	waitFor(t, "the startup hook to backfill the lane cache", func() bool {
		return len(p.CachedModels()) == 2
	})

	ids := getModels(t, srv).ids()
	for _, want := range []string{"zai/glm-5.3", "zai/glm-5.3-air"} {
		if _, ok := ids[want]; !ok {
			t.Errorf("model %q missing from /v1/models: %v", want, ids)
		}
	}
}

// AC5: a lane restored from providers.json with an empty cache is backfilled
// on restart, with no add_provider anywhere (todo #15's manual refresh step,
// automated).
func TestDiscovery_RestoreBackfillsEmptyCacheFromProvidersJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")
	up := newDiscoveryUpstream(t, "deepseek-chat", "deepseek-reasoner")

	// Boot 1: the lane is persisted while its upstream is unreachable.
	store1 := NewRuntimeProviderStore(path)
	store1.DefaultDataDir = dir
	if err := store1.Add(openaicompat.Config{
		Name:    "deepseek",
		BaseURL: up.URL(),
		APIKey:  "sk-deepseek",
	}); err != nil {
		t.Fatalf("persist lane: %v", err)
	}

	// Boot 2: restore from disk. Construction-time discovery still fails (the
	// first request), so only the restore/startup discovery hook can fill the
	// cache.
	up.setFailFirst(1)
	cfg := newDiscoveryTestConfig(t)
	cfg.DataDir = dir
	registry := provider.NewRegistry()
	srv := NewServer(cfg, registry,
		WithStateManager(state.NewStateManager()),
		WithRuntimeProviderStore(NewRuntimeProviderStore(path)))

	if _, ok := registry.Get("deepseek"); !ok {
		t.Fatalf("lane not restored: %v", registry.Names())
	}
	waitFor(t, "the restore hook to backfill the lane cache", func() bool {
		return len(cachedModelsOf(t, registry, "deepseek")) == 2
	})
	if got := up.modelCalls(); got < 2 {
		t.Errorf("expected a construction attempt plus a hook attempt, got %d upstream calls", got)
	}

	ids := getModels(t, srv).ids()
	for _, want := range []string{"deepseek/deepseek-chat", "deepseek/deepseek-reasoner"} {
		if _, ok := ids[want]; !ok {
			t.Errorf("model %q missing from /v1/models: %v", want, ids)
		}
	}
}

func cachedModelsOf(t *testing.T, registry *provider.Registry, name string) []string {
	t.Helper()
	bundle, ok := registry.Get(name)
	if !ok || bundle.Inference == nil {
		return nil
	}
	cacher, ok := bundle.Inference.(interface{ CachedModels() []string })
	if !ok {
		return nil
	}
	return cacher.CachedModels()
}

// fakeModelTicker drives the refresh schedule from the test: a tick is an
// explicit send, so no test ever sleeps to wait for the 6h default.
type fakeModelTicker struct {
	c     chan time.Time
	stops atomic.Int32
}

func newFakeModelTicker() *fakeModelTicker { return &fakeModelTicker{c: make(chan time.Time, 1)} }

func (f *fakeModelTicker) C() <-chan time.Time { return f.c }
func (f *fakeModelTicker) Stop()               { f.stops.Add(1) }

func (f *fakeModelTicker) tick() {
	select {
	case f.c <- time.Now():
	default:
	}
}

// fakeDiscoveryLane is an InferenceProvider with a controllable FetchModels, so
// discovery can be observed without an HTTP upstream.
type fakeDiscoveryLane struct {
	fakeInferenceProvider
	models      []string
	err         error
	fetches     atomic.Int32
	inFlight    atomic.Int32
	maxInFlight atomic.Int32
	blockUntil  chan struct{}
}

func (f *fakeDiscoveryLane) FetchModels(ctx context.Context) ([]string, error) {
	f.fetches.Add(1)
	cur := f.inFlight.Add(1)
	for {
		max := f.maxInFlight.Load()
		if cur <= max || f.maxInFlight.CompareAndSwap(max, cur) {
			break
		}
	}
	defer f.inFlight.Add(-1)
	if f.blockUntil != nil {
		select {
		case <-f.blockUntil:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return f.models, f.err
}

// AC4: a ticker tick re-runs discovery, so a model added upstream between ticks
// shows up without a restart - including for lanes whose cache was already
// populated (the schedule refreshes ALL discovery lanes).
func TestDiscovery_ScheduledRefreshPicksUpNewModels(t *testing.T) {
	up := newDiscoveryUpstream(t, "glm-5.3")

	registry := provider.NewRegistry()
	p, err := openaicompat.New(openaicompat.Config{
		Name:    "zai",
		BaseURL: up.URL(),
		APIKey:  "sk-zai",
	})
	if err != nil {
		t.Fatalf("openaicompat.New: %v", err)
	}
	registry.Register(p.Provider())

	ticker := newFakeModelTicker()
	var requestedInterval atomic.Int64
	srv := NewServer(newDiscoveryTestConfig(t), registry,
		WithStateManager(state.NewStateManager()),
		WithModelRefreshInterval(time.Hour),
		withModelTickerFactory(func(d time.Duration) modelTicker {
			requestedInterval.Store(int64(d))
			return ticker
		}),
	)

	waitFor(t, "the refresh schedule to be built", func() bool {
		return requestedInterval.Load() != 0
	})
	if got := requestedInterval.Load(); got != int64(time.Hour) {
		t.Errorf("refresh interval = %v, want the configured 1h", time.Duration(got))
	}

	ids := getModels(t, srv).ids()
	if _, ok := ids["zai/glm-5.3"]; !ok {
		t.Fatalf("precondition: %q missing from /v1/models: %v", "zai/glm-5.3", ids)
	}
	if _, ok := ids["zai/glm-5.4"]; ok {
		t.Fatalf("precondition: %q must not be advertised yet: %v", "zai/glm-5.4", ids)
	}

	// The upstream ships a new model; only a refresh can pick it up.
	up.setModels("glm-5.3", "glm-5.4")
	before := up.modelCalls()
	ticker.tick()

	waitFor(t, "the scheduled refresh to pick up glm-5.4", func() bool {
		_, ok := getModels(t, srv).ids()["zai/glm-5.4"]
		return ok
	})
	if got := up.modelCalls(); got <= before {
		t.Errorf("the tick did not re-run discovery: %d -> %d upstream calls", before, got)
	}
}

// WithModelRefreshInterval(0) disables the schedule: no ticker is ever built
// and no refresh happens, while the startup backfill still runs.
func TestDiscovery_RefreshIntervalZeroDisablesSchedule(t *testing.T) {
	up := newDiscoveryUpstream(t, "glm-5.3")
	up.setFailFirst(1) // the lane starts with an empty cache

	registry := provider.NewRegistry()
	p, err := openaicompat.New(openaicompat.Config{Name: "zai", BaseURL: up.URL()})
	if err != nil {
		t.Fatalf("openaicompat.New: %v", err)
	}
	registry.Register(p.Provider())

	var tickersBuilt atomic.Int32
	ticker := newFakeModelTicker()
	srv := NewServer(newDiscoveryTestConfig(t), registry,
		WithStateManager(state.NewStateManager()),
		WithModelRefreshInterval(0),
		withModelTickerFactory(func(d time.Duration) modelTicker {
			tickersBuilt.Add(1)
			return ticker
		}),
	)

	// Startup backfill still heals the empty cache.
	waitFor(t, "the startup backfill", func() bool { return len(p.CachedModels()) == 1 })

	// But no schedule exists to pick up upstream changes.
	up.setModels("glm-5.3", "glm-5.4")
	ticker.tick()
	ids := getModels(t, srv).ids()
	if _, ok := ids["zai/glm-5.4"]; ok {
		t.Errorf("a disabled schedule still refreshed the lane: %v", ids)
	}
	if got := tickersBuilt.Load(); got != 0 {
		t.Errorf("refresh schedule was built %d times, want none", got)
	}
}

// Custom-wire lanes stay non-discoverable: a lane with no FetchModels surface is
// never probed and never gets invented ids, on startup or on a tick.
func TestDiscovery_LanesWithoutFetchModelsAreLeftAlone(t *testing.T) {
	registry := provider.NewRegistry()
	lane := &fakeInferenceProvider{name: "codex"}
	registry.Register(provider.Provider{Inference: lane})

	ticker := newFakeModelTicker()
	srv := NewServer(newDiscoveryTestConfig(t), registry,
		WithStateManager(state.NewStateManager()),
		withModelTickerFactory(func(d time.Duration) modelTicker { return ticker }),
	)

	ticker.tick()
	ticker.tick()

	ids := getModels(t, srv).ids()
	if _, ok := ids["codex"]; !ok {
		t.Errorf("the lane entry is missing from /v1/models: %v", ids)
	}
	for id := range ids {
		if strings.HasPrefix(id, "codex/") {
			t.Errorf("invented model id for a lane without model discovery: %q", id)
		}
	}
}

// A lane that opted out of discovery (quirks.model_list_passthrough:false) is
// skipped by every automatic pass, and the opt-out survives providers.json.
func TestDiscovery_OptedOutLaneIsNotDiscovered(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")
	up := newDiscoveryUpstream(t, "glm-5.3")

	// Persisted with an explicit opt-out.
	store := NewRuntimeProviderStore(path)
	store.DefaultDataDir = dir
	if err := store.Add(openaicompat.Config{
		Name:                       "zai",
		BaseURL:                    up.URL(),
		OptOutModelListPassthrough: true,
	}); err != nil {
		t.Fatalf("persist lane: %v", err)
	}
	onDisk, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}
	if !strings.Contains(string(onDisk), `"model_list_passthrough": false`) {
		t.Fatalf("the explicit opt-out was not persisted: %s", onDisk)
	}

	// Restart: the lane comes back, still opted out, and nothing probes it.
	ticker := newFakeModelTicker()
	cfg := newDiscoveryTestConfig(t)
	cfg.DataDir = dir
	registry := provider.NewRegistry()
	srv := NewServer(cfg, registry,
		WithStateManager(state.NewStateManager()),
		WithRuntimeProviderStore(NewRuntimeProviderStore(path)),
		withModelTickerFactory(func(d time.Duration) modelTicker { return ticker }),
	)
	ticker.tick()

	if _, ok := registry.Get("zai"); !ok {
		t.Fatal("opted-out lane was not restored")
	}
	ids := getModels(t, srv).ids()
	if _, ok := ids["zai/glm-5.3"]; ok {
		t.Errorf("an opted-out lane advertised a discovered model: %v", ids)
	}
	// Give every automatic pass (startup, both ticks) a chance to misbehave.
	time.Sleep(200 * time.Millisecond)
	if got := up.modelCalls(); got != 0 {
		t.Errorf("the opted-out lane was probed %d times, want 0", got)
	}
}

// The discovery pool is bounded: at most modelDiscoveryConcurrency lanes are
// probed at the same time, so one slow upstream cannot stampede the rest.
func TestDiscovery_PoolIsBounded(t *testing.T) {
	const lanes = 5
	targets := make([]discoveryTarget, 0, lanes)
	release := make(chan struct{})
	for i := 0; i < lanes; i++ {
		lane := &fakeDiscoveryLane{blockUntil: release}
		lane.name = fmt.Sprintf("lane-%d", i)
		targets = append(targets, discoveryTarget{
			name:   lane.name,
			bundle: provider.Provider{Inference: lane},
		})
	}

	done := make(chan struct{})
	go func() {
		discoverLanes(context.Background(), targets, "test")
		close(done)
	}()

	// Let the pool fill, then drain it: nobody may exceed the bound.
	time.Sleep(50 * time.Millisecond)
	for _, target := range targets {
		lane := target.bundle.Inference.(*fakeDiscoveryLane)
		if got := lane.maxInFlight.Load(); got > modelDiscoveryConcurrency {
			t.Errorf("lane %s ran %d concurrent discoveries, want at most %d",
				target.name, got, modelDiscoveryConcurrency)
		}
	}
	close(release)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the discovery pool did not finish")
	}
}

// Without WithModelRefreshInterval the schedule uses the documented 6h default.
func TestDiscovery_DefaultRefreshInterval(t *testing.T) {
	ticker := newFakeModelTicker()
	var requestedInterval atomic.Int64
	srv := NewServer(newDiscoveryTestConfig(t), provider.NewRegistry(),
		WithStateManager(state.NewStateManager()),
		withModelTickerFactory(func(d time.Duration) modelTicker {
			requestedInterval.Store(int64(d))
			return ticker
		}),
	)
	waitFor(t, "the default refresh schedule to be built", func() bool {
		return requestedInterval.Load() != 0
	})
	if got := time.Duration(requestedInterval.Load()); got != DefaultModelRefreshInterval {
		t.Errorf("default refresh interval = %v, want %v", got, DefaultModelRefreshInterval)
	}
	if got := time.Duration(requestedInterval.Load()); got != 6*time.Hour {
		t.Errorf("DefaultModelRefreshInterval = %v, want 6h", got)
	}
	_ = srv
}
