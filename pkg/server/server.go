package server

import (
	"context"
	"net/http"
	"os"

	"github.com/smhanov/ultiproxy/pkg/mcp"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// Server represents the Ultiproxy dual HTTP surface server.
type Server struct {
	cfg        *Config
	registry   *provider.Registry
	router     Router
	sm         *state.StateManager
	writer     *storage.Writer
	mcpServer  *mcp.Server
	auth       *AuthMiddleware
	httpServer *http.Server
	mux        *http.ServeMux
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

	for _, opt := range opts {
		opt(s)
	}

	if s.router == nil {
		s.router = NewRegistryRouter(registry, s.sm)
	}

	// Create MCP server if not provided
	if s.mcpServer == nil {
		var stateSrc mcp.StateSource
		if s.sm != nil {
			stateSrc = &stateManagerSourceAdapter{sm: s.sm}
		}
		s.mcpServer = mcp.NewServer(registry, stateSrc)
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
