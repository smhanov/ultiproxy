package mcp

import (
	"context"
	"encoding/json"

	"github.com/smhanov/ultiproxy/pkg/state"
)

// JSONRPCRequest represents a JSON-RPC 2.0 request.
type JSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// JSONRPCResponse represents a JSON-RPC 2.0 response.
type JSONRPCResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      any           `json:"id"`
	Result  any           `json:"result,omitempty"`
	Error   *JSONRPCError `json:"error,omitempty"`
}

// JSONRPCError defines a JSON-RPC error.
type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// Standard JSON-RPC error codes
const (
	CodeParseError     = -32700
	CodeInvalidRequest = -32600
	CodeMethodNotFound = -32601
	CodeInvalidParams  = -32602
	CodeInternalError  = -32603
)

// MCP Tool Definition
type Tool struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	InputSchema *InputSchema `json:"inputSchema"`
}

// InputSchema defines JSON schema for a tool.
type InputSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]PropertyDef `json:"properties,omitempty"`
	Required   []string               `json:"required,omitempty"`
}

// PropertyDef describes a single parameter.
type PropertyDef struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// CallToolParams is the parameter payload for tools/call.
type CallToolParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// CallToolResult is the standard MCP tool execution result.
type CallToolResult struct {
	Content []ToolContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// ToolContent holds a single content item in a tool result.
type ToolContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// StateSource provides state snapshot and model manipulation for MCP tools.
type StateSource interface {
	Snapshot() *state.RuntimeSnapshot
	ToggleModel(modelID string, enabled bool) error
}

// UsageSource provides client usage stats for MCP tools.
type UsageSource interface {
	GetClientUsage(ctx context.Context, clientID string, window string) (any, error)
}
