package server

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
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
	// mcpLaneBuilder bridges the runtime store's LaneBuilder into the MCP
	// add_provider custom-kind path (antigravity, anthropic, codex).
	mcpLaneBuilder func(name, kind, apiKey string) (provider.Provider, error)
	httpServer     *http.Server
	mux            *http.ServeMux
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

// NewServer creates a new Server.
func NewServer(cfg *Config, registry *provider.Registry, opts ...Option) *Server {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	s := &Server{
		cfg:      cfg,
		registry: registry,
		mux:      http.NewServeMux(),
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
		s.mcpServer = mcp.NewServer(registry, stateSrc, mcpOpts...)
	}

	// Auth middleware
	s.auth = NewAuthMiddleware(cfg.Server.APIKey, cfg.Server.ClientKeys)

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

// Shutdown gracefully stops the HTTP server.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func (s *Server) handleLLMsTxt(w http.ResponseWriter, r *http.Request) {
	path := s.cfg.Server.LLMsTxtPath
	if path == "" {
		path = "llms.txt"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		http.NotFound(w, r)
		return
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

// syncCatalogToState copies every alias from the catalog into the state
// manager's Models map so routing and /v1/models resolve them.
func (s *Server) syncCatalogToState() {
	if s.sm == nil {
		return
	}
	aliases := s.catalog.List()
	s.sm.Update(func(snap *state.RuntimeSnapshot) {
		if snap.Models == nil {
			snap.Models = make(map[string]state.ModelRuntime)
		}
		for alias, entry := range aliases {
			snap.Models[alias] = state.ModelRuntime{
				ID:              alias,
				Provider:        entry.Provider,
				Enabled:         true,
				ContextLimit:    entry.ContextLimit,
				MaxOutput:       entry.MaxOutput,
				PricingTag:      entry.PricingTag,
				BenchmarkScores: entry.BenchmarkScores,
			}
		}
	})
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
