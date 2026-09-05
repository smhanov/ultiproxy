package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/smhanov/ultiproxy/pkg/mcp"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// Server represents the Ultiproxy dual HTTP surface server.
type Server struct {
	cfg       *Config
	registry  *provider.Registry
	router    Router
	sm        *state.StateManager
	writer    *storage.Writer
	mcpServer *mcp.Server
	auth      *AuthMiddleware
	catalog   *ModelCatalog
	timeouts  *TimeoutManager
	providers *RuntimeProviderStore
	// catalogOwned remembers which state model entries the last catalog sync
	// installed, so the next sync can delete the ones the catalog no longer
	// serves (remove_model_alias) without touching entries that came from
	// elsewhere - notably toggle_model-created rows for lane-prefixed ids.
	// catalogOwnedMu guards it: alias administration can arrive concurrently
	// on the MCP surface.
	catalogOwnedMu sync.Mutex
	catalogOwned   map[string]bool
	// mcpLaneBuilder bridges the runtime store's LaneBuilder into the MCP
	// add_provider custom-kind path (antigravity, anthropic, codex).
	mcpLaneBuilder func(name, kind, apiKey string) (provider.Provider, error)
	httpServer     *http.Server
	mux            *http.ServeMux
	// modelRefreshInterval is the scheduled model-discovery cadence. -1
	// (NewServer's default) means DefaultModelRefreshInterval; 0 disables the
	// schedule (the startup backfill still runs).
	modelRefreshInterval time.Duration
	// newModelTicker builds the refresh ticker. It is nil in production (a real
	// *time.Ticker) and replaced by the test hook so the schedule can be driven
	// with a fake clock.
	newModelTicker func(d time.Duration) modelTicker
	// discoveryCancel stops the background model-discovery loop (Shutdown).
	discoveryCancel context.CancelFunc
	// credentialRefreshInterval is the proactive credential-refresher cadence.
	// -1 (NewServer's default) means DefaultCredentialRefreshInterval; 0
	// disables the schedule (credentials are then refreshed lazily on request
	// and reactively after an upstream 401/403).
	credentialRefreshInterval time.Duration
	// newCredentialTicker builds the credential-refresh ticker. It is nil in
	// production (a real *time.Ticker) and replaced by the test hook so the
	// schedule can be driven with a fake clock.
	newCredentialTicker func(d time.Duration) modelTicker
	// credentialNow is the clock the refresh lead window is measured on. nil in
	// production (time.Now), replaced by the test hook.
	credentialNow func() time.Time
	// credentialRefreshCancel stops the background credential refresher
	// (Shutdown).
	credentialRefreshCancel context.CancelFunc
	// requestIDSeq allocates the request ids that tie the request, attempt and
	// usage telemetry rows of one dispatch together. See nextRequestID.
	requestIDSeq atomic.Int64
}

// nextRequestID returns a unique positive request id.
//
// Ids are minted by the proxy instead of by SQLite because the attempt and
// usage rows of a dispatch are recorded before its terminal request row (the
// outcome is only known at the end) and must reference it. The sequence is
// seeded from the wall clock, so ids stay unique across restarts even though
// the requests table also auto-assigns small rowids for id-less rows.
func (s *Server) nextRequestID() int64 {
	return s.requestIDSeq.Add(1)
}

// Option configures Server.
type Option func(*Server)

// WithRouter overrides default router.
func WithRouter(r Router) Option {
	return func(s *Server) {
		s.router = r
	}
}

// WithStateManager sets state manager.
func WithStateManager(sm *state.StateManager) Option {
	return func(s *Server) {
		s.sm = sm
	}
}

// WithStorageWriter sets telemetry writer.
func WithStorageWriter(w *storage.Writer) Option {
	return func(s *Server) {
		s.writer = w
	}
}

// WithRuntimeProviderStore sets the runtime provider store used by the MCP
// add_provider / remove_provider / list_providers tools. When not provided,
// NewServer builds one from data_dir/providers.json.
func WithRuntimeProviderStore(store *RuntimeProviderStore) Option {
	return func(s *Server) {
		if store != nil {
			s.providers = store
		}
	}
}

// WithMCPServer sets MCP server instance.
func WithMCPServer(m *mcp.Server) Option {
	return func(s *Server) {
		s.mcpServer = m
	}
}

// WithModelRefreshInterval sets how often every discovery-capable lane
// re-fetches its upstream model list. Zero disables the schedule (lanes are
// still discovered at registration and backfilled at startup); a negative value
// restores the default, DefaultModelRefreshInterval.
func WithModelRefreshInterval(d time.Duration) Option {
	return func(s *Server) {
		if d < 0 {
			d = DefaultModelRefreshInterval
		}
		s.modelRefreshInterval = d
	}
}

// NewServer creates a new Server.
func NewServer(cfg *Config, registry *provider.Registry, opts ...Option) *Server {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	s := &Server{
		cfg:      cfg,
		registry: registry,
		mux:      http.NewServeMux(),
		// -1 means "no explicit interval": startModelDiscovery then uses
		// DefaultModelRefreshInterval. (0 is a valid, explicit "disabled".)
		modelRefreshInterval: -1,
		// Same convention for the credential refresher.
		credentialRefreshInterval: -1,
	}
	s.requestIDSeq.Store(time.Now().UnixNano())

	for _, opt := range opts {
		opt(s)
	}

	// Model alias catalog (config + runtime-persisted aliases). Must exist
	// before the MCP server is built (the bridge references it).
	catalog, catalogErr := NewModelCatalog(cfg.Server.Models, filepath.Join(cfg.DataDir, "aliases.json"))
	if catalogErr != nil {
		log.Printf("[server] model catalog error (continuing with config only): %v", catalogErr)
		cfg.Server.Models = nil
		catalog, _ = NewModelCatalog(cfg.Server.Models, "")
	}
	s.catalog = catalog

	// Runtime provider store (MCP add_provider / remove_provider /
	// list_providers). Persisted lanes must be registered BEFORE the router or
	// the model catalog resolve them, so a restart from the same DataDir serves
	// the same lanes with no config file anywhere.
	if s.providers == nil {
		s.providers = NewRuntimeProviderStore(filepath.Join(cfg.DataDir, "providers.json"))
	}
	if s.providers.DefaultDataDir == "" {
		s.providers.DefaultDataDir = cfg.DataDir
	}
	// Wire the custom-lane builder (antigravity, anthropic, codex, ...) into
	// MCP add_provider / restore. The store's LaneBuilder and the MCP custom
	// builder both come from the same constructor; the router/catalog resolve
	// lanes after this.
	if s.providers.LaneBuilder != nil {
		laneBuilder := s.providers.LaneBuilder
		s.mcpLaneBuilder = func(name, kind, apiKey string) (provider.Provider, error) {
			return laneBuilder(name, kind, s.providers.DefaultDataDir, apiKey)
		}
	}
	s.providers.Restore(registry)

	// Default router needs the catalog so unknown models can be rejected
	// instead of silently routed to the first registered provider.
	if s.router == nil {
		s.router = NewRegistryRouter(registry, s.sm, catalog)
	}

	// Register aliases into the state manager so the router and /v1/models
	// resolve them without touching the catalog on every request.
	if s.sm != nil {
		s.syncCatalogToState()
	}

	// Per-provider request timeouts (config + runtime overrides).
	timeouts, _ := NewTimeoutManager(cfg.Server.Timeouts, DefaultRequestTimeout, filepath.Join(cfg.DataDir, "timeouts.json"))
	s.timeouts = timeouts

	// Create MCP server if not provided
	if s.mcpServer == nil {
		var stateSrc mcp.StateSource
		if s.sm != nil {
			stateSrc = &stateManagerSourceAdapter{sm: s.sm}
		}
		mcpOpts := []mcp.Option{
			mcp.WithAliasManager(&catalogBridge{catalog: s.catalog, server: s}),
			mcp.WithTimeoutManager(&timeoutBridge{timeouts: s.timeouts}),
		}
		if s.providers != nil {
			mcpOpts = append(mcpOpts, mcp.WithProviderStore(s.providers))
		}
		if s.mcpLaneBuilder != nil {
			mcpOpts = append(mcpOpts, mcp.WithCustomLaneBuilder(s.mcpLaneBuilder))
		}
		if s.writer != nil {
			// get_client_usage answers from the SQLite telemetry store instead
			// of synthetic zeros.
			mcpOpts = append(mcpOpts, mcp.WithUsageSource(newStorageUsageSource(s.writer, cfg.Server.ClientKeys)))
		}
		s.mcpServer = mcp.NewServer(registry, stateSrc, mcpOpts...)
	}

	// Auth middleware
	s.auth = NewAuthMiddleware(cfg.Server.APIKey, cfg.Server.ClientKeys)

	// Proactive credential refresh: keep OAuth lanes alive while they sit idle.
	// Started before the discovery loop so lane credentials are healthy before
	// anything dials an upstream to enumerate models.
	s.startCredentialRefresh()

	// Background model discovery: backfill lanes whose cache is still empty
	// after Restore, then refresh every discovery lane on a schedule. Never
	// blocks startup and never touches the request path (/v1/models keeps
	// reading the cache only).
	s.startModelDiscovery()

	// Mount routes
	s.setupRoutes()

	return s
}

func (s *Server) setupRoutes() {
	// OpenAI Chat Completions
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	// Anthropic Messages
	s.mux.HandleFunc("POST /v1/messages", s.handleMessages)
	// OpenAI Models list
	s.mux.HandleFunc("GET /v1/models", s.handleModels)
	// Health check
	s.mux.HandleFunc("GET /healthz", s.handleHealthz)
	// LLMs.txt
	s.mux.HandleFunc("GET /llms.txt", s.handleLLMsTxt)

	// Quota Dashboard routes
	s.mux.HandleFunc("GET /api/quota", s.handleQuotaDashboard)
	s.mux.HandleFunc("GET /api/stats/summary", s.handleStatsSummary)
	s.mux.HandleFunc("GET /api/stats/by-client", s.handleStatsByClient)
	s.mux.HandleFunc("GET /quota.txt", s.handleQuotaText)
	s.mux.HandleFunc("GET /quota.md", s.handleQuotaMarkdown)

	// MCP endpoints
	s.mux.HandleFunc("/mcp", s.handleMCP)
	s.mux.HandleFunc("/mcp/sse", s.handleMCPSSE)
}

// Handler returns the root http.Handler wrapped with authentication middleware.
func (s *Server) Handler() http.Handler {
	return s.auth.Wrap(s.mux)
}

// Start begins listening on configured address.
func (s *Server) Start() error {
	s.httpServer = &http.Server{
		Addr:    s.cfg.Server.Addr,
		Handler: s.Handler(),
	}
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully stops the HTTP server, the background model-discovery
// loop and the proactive credential refresher. In-flight discovery and refresh
// calls are cancelled through their per-call budget contexts; the loops are not
// waited on, so a slow upstream cannot delay shutdown.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.discoveryCancel != nil {
		s.discoveryCancel()
	}
	if s.credentialRefreshCancel != nil {
		s.credentialRefreshCancel()
	}
	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// storageUsageSource adapts the SQLite telemetry writer to mcp.UsageSource, so
// the MCP get_client_usage tool answers from the recorded requests/usage rows
// instead of hardcoded zeros.
type storageUsageSource struct {
	writer *storage.Writer
	// clientHashes maps configured client key names to the sha256 hex digest
	// stored in requests.client_key_hash, so callers may pass either a key
	// name ("alice") or the digest itself.
	clientHashes map[string]string
}

// newStorageUsageSource builds a usage source over a telemetry writer. The
// client key map only resolves names to digests; it is never used to
// authenticate anything.
func newStorageUsageSource(w *storage.Writer, clientKeys map[string]string) *storageUsageSource {
	hashes := make(map[string]string, len(clientKeys))
	for name, key := range clientKeys {
		if key == "" {
			continue
		}
		sum := sha256.Sum256([]byte(key))
		hashes[name] = hex.EncodeToString(sum[:])
	}
	return &storageUsageSource{writer: w, clientHashes: hashes}
}

// GetClientUsage implements mcp.UsageSource. An empty clientID asks for the
// aggregate over every client (including rows with no client key hash); a
// non-empty one is matched as a configured client key name, or verbatim as the
// key hash. Windows and totals come from the storage layer; an unreachable or
// empty database yields zeros rather than an error.
func (s *storageUsageSource) GetClientUsage(ctx context.Context, clientID, window string) (any, error) {
	if s == nil || s.writer == nil || s.writer.DB() == nil {
		return storage.ClientUsageTotals{ClientID: clientID, Window: window}, nil
	}

	// An empty client id means "every client"; otherwise resolve a configured
	// client key name to its digest, falling back to treating the value as the
	// digest itself.
	hash := clientID
	if resolved, ok := s.clientHashes[clientID]; ok {
		hash = resolved
	}

	totals, err := s.writer.GetClientUsage(ctx, hash, window)
	if err != nil {
		return nil, err
	}
	totals.ClientID = clientID
	return totals, nil
}

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

// handleLLMsTxt serves the agent-facing documentation. An llms.txt found at
// the configured path wins (operators can override the document without
// rebuilding); when that path is missing -- the normal case for a binary
// installed by dist/install.sh, which copies no documentation next to the
// executable -- the copy embedded at build time is served instead.
func (s *Server) handleLLMsTxt(w http.ResponseWriter, r *http.Request) {
	path := s.cfg.Server.LLMsTxtPath
	if path == "" {
		path = "llms.txt"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		data = embeddedLLMsTxt()
		if len(data) == 0 {
			http.NotFound(w, r)
			return
		}
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write(data)
}

func (s *Server) handleMCP(w http.ResponseWriter, r *http.Request) {
	s.mcpServer.ServeHTTP(w, r)
}

func (s *Server) handleMCPSSE(w http.ResponseWriter, r *http.Request) {
	s.mcpServer.HandleSSE(w, r)
}

// stateManagerSourceAdapter adapts state.StateManager to mcp.StateSource
type stateManagerSourceAdapter struct {
	sm *state.StateManager
}

func (a *stateManagerSourceAdapter) Snapshot() *state.RuntimeSnapshot {
	if a.sm == nil {
		return nil
	}
	return a.sm.Snapshot()
}

func (a *stateManagerSourceAdapter) ToggleModel(modelID string, enabled bool) error {
	if a.sm == nil {
		return nil
	}
	a.sm.Update(func(snap *state.RuntimeSnapshot) {
		if snap.Models == nil {
			snap.Models = make(map[string]state.ModelRuntime)
		}
		m, ok := snap.Models[modelID]
		if !ok {
			m = state.ModelRuntime{ID: modelID}
		}
		m.Enabled = enabled
		snap.Models[modelID] = m
	})
	return nil
}

// syncCatalogToState reconciles the state manager's Models map with the alias
// catalog so routing and /v1/models resolve the same table the catalog owns.
//
// It is a reconciliation, not an upsert: aliases the catalog no longer serves
// (remove_model_alias) are dropped from state, otherwise they keep routing
// until the process restarts. Entries the catalog still serves keep their
// runtime Enabled bit, so administering one alias never re-enables another the
// operator disabled with toggle_model. Only aliases previously installed by
// this sync are deleted, so toggle_model rows for ids the catalog never owned
// (lane-prefixed ids like "zai/glm-5.3-flash") survive alias administration.
func (s *Server) syncCatalogToState() {
	if s.sm == nil {
		return
	}

	aliases := s.catalog.List()

	s.catalogOwnedMu.Lock()
	var gone []string
	for alias := range s.catalogOwned {
		if _, still := aliases[alias]; !still {
			gone = append(gone, alias)
		}
	}
	s.catalogOwnedMu.Unlock()

	s.sm.Update(func(snap *state.RuntimeSnapshot) {
		if snap.Models == nil {
			snap.Models = make(map[string]state.ModelRuntime)
		}
		for _, alias := range gone {
			delete(snap.Models, alias)
		}
		for alias, entry := range aliases {
			prev, known := snap.Models[alias]
			next := state.ModelRuntime{
				ID:              alias,
				Provider:        entry.Provider,
				Enabled:         true,
				ContextLimit:    entry.ContextLimit,
				MaxOutput:       entry.MaxOutput,
				PricingTag:      entry.PricingTag,
				BenchmarkScores: entry.BenchmarkScores,
			}
			if known {
				// Preserve the runtime toggle state of an alias the operator
				// already switched off.
				next.Enabled = prev.Enabled
			}
			snap.Models[alias] = next
		}
	})

	owned := make(map[string]bool, len(aliases))
	for alias := range aliases {
		owned[alias] = true
	}
	s.catalogOwnedMu.Lock()
	s.catalogOwned = owned
	s.catalogOwnedMu.Unlock()
}

// catalogBridge adapts server.ModelCatalog to the mcp.AliasManager interface
// and re-syncs the state manager after runtime mutations.
type catalogBridge struct {
	catalog *ModelCatalog
	server  *Server
}

func (b *catalogBridge) List() map[string]mcp.ModelAlias {
	out := make(map[string]mcp.ModelAlias)
	for alias, entry := range b.catalog.List() {
		out[alias] = mcp.ModelAlias{
			Provider:        entry.Provider,
			Upstream:        entry.Upstream,
			ContextLimit:    entry.ContextLimit,
			MaxOutput:       entry.MaxOutput,
			PricingTag:      entry.PricingTag,
			InputCost:       entry.InputCost,
			OutputCost:      entry.OutputCost,
			BenchmarkScores: entry.BenchmarkScores,
		}
	}
	return out
}

func (b *catalogBridge) Sorted() []string { return b.catalog.Sorted() }

func (b *catalogBridge) Set(alias string, entry mcp.ModelAlias) error {
	if err := b.catalog.Set(alias, ModelAlias{
		Provider:        entry.Provider,
		Upstream:        entry.Upstream,
		ContextLimit:    entry.ContextLimit,
		MaxOutput:       entry.MaxOutput,
		PricingTag:      entry.PricingTag,
		InputCost:       entry.InputCost,
		OutputCost:      entry.OutputCost,
		BenchmarkScores: entry.BenchmarkScores,
	}); err != nil {
		return err
	}
	b.server.syncCatalogToState()
	return nil
}

func (b *catalogBridge) Remove(alias string) error {
	if err := b.catalog.Remove(alias); err != nil {
		return err
	}
	b.server.syncCatalogToState()
	return nil
}

// timeoutBridge adapts server.TimeoutManager to the mcp.TimeoutManager
// interface. Durations are exchanged as Go duration strings.
type timeoutBridge struct {
	timeouts *TimeoutManager
}

func (b *timeoutBridge) Timeout(provider string) string {
	return b.timeouts.Timeout(provider).String()
}

func (b *timeoutBridge) Set(provider string, timeout string) error {
	d, err := time.ParseDuration(timeout)
	if err != nil || d <= 0 {
		return fmt.Errorf("invalid timeout %q: %w", timeout, errTimeoutInvalid)
	}
	return b.timeouts.Set(provider, d)
}

func (b *timeoutBridge) Remove(provider string) error { return b.timeouts.Remove(provider) }

func (b *timeoutBridge) List() map[string]string { return b.timeouts.List() }
