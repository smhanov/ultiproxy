package mcp

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

// Server implements an MCP server supporting Streamable HTTP on /mcp and legacy /mcp/sse.
type Server struct {
	registry    *provider.Registry
	stateSource StateSource
	usageSource UsageSource
	aliases     AliasManager
	timeouts    TimeoutManager
	name        string
	version     string
	mu          sync.RWMutex
}

// Option configures MCP server.
type Option func(*Server)

// WithUsageSource configures usage source.
func WithUsageSource(u UsageSource) Option {
	return func(s *Server) {
		s.usageSource = u
	}
}

// WithAliasManager configures the model alias catalog.
func WithAliasManager(am AliasManager) Option {
	return func(s *Server) {
		s.aliases = am
	}
}

// WithTimeoutManager configures per-provider timeouts.
func WithTimeoutManager(tm TimeoutManager) Option {
	return func(s *Server) {
		s.timeouts = tm
	}
}

// NewServer creates a new MCP server.
func NewServer(registry *provider.Registry, stateSource StateSource, opts ...Option) *Server {
	s := &Server{
		registry:    registry,
		stateSource: stateSource,
		name:        "ultiproxy",
		version:     "0.1.0",
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// ServeHTTP handles both Streamable HTTP POST/GET requests.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		s.handlePost(w, r)
	case http.MethodGet:
		s.handleGetSSE(w, r)
	default:
		w.Header().Set("Allow", "GET, POST")
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// HandleSSE provides legacy /mcp/sse endpoint support.
func (s *Server) HandleSSE(w http.ResponseWriter, r *http.Request) {
	s.handleGetSSE(w, r)
}

func (s *Server) handlePost(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONRPCError(w, nil, CodeParseError, "failed to read request body")
		return
	}

	var req JSONRPCRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSONRPCError(w, nil, CodeParseError, "invalid JSON")
		return
	}

	// Handle notification (no ID)
	if req.ID == nil {
		if req.Method == "notifications/initialized" {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		// For other notifications, return 204
		w.WriteHeader(http.StatusNoContent)
		return
	}

	var res any
	var rpcErr *JSONRPCError

	switch req.Method {
	case "initialize":
		res = map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities": map[string]any{
				"tools": map[string]any{},
			},
			"serverInfo": map[string]any{
				"name":    s.name,
				"version": s.version,
			},
		}

	case "tools/list":
		res = s.handleListTools()

	case "tools/call":
		res, rpcErr = s.handleCallTool(r.Context(), req.Params)

	default:
		rpcErr = &JSONRPCError{
			Code:    CodeMethodNotFound,
			Message: fmt.Sprintf("method not found: %s", req.Method),
		}
	}

	resp := JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result:  res,
		Error:   rpcErr,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleGetSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Emit initial endpoint event indicating where to POST JSON-RPC messages
	_, _ = fmt.Fprintf(w, "event: endpoint\ndata: /mcp\n\n")
	flusher.Flush()

	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			_, err := fmt.Fprintf(w, "event: ping\ndata: {}\n\n")
			if err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

func writeJSONRPCError(w http.ResponseWriter, id any, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error: &JSONRPCError{
			Code:    code,
			Message: msg,
		},
	})
}
