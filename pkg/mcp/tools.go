package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

var standardTools = []Tool{
	{
		Name:        "list_models",
		Description: "List available models and their status",
		InputSchema: &InputSchema{
			Type:       "object",
			Properties: map[string]PropertyDef{},
		},
	},
	{
		Name:        "get_quota_status",
		Description: "Get quota status for an upstream provider",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider name"},
			},
			Required: []string{"provider"},
		},
	},
	{
		Name:        "toggle_model",
		Description: "Enable or disable a model",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"model_id": {Type: "string", Description: "Model ID"},
				"enabled":  {Type: "boolean", Description: "Enable (true) or disable (false)"},
			},
			Required: []string{"model_id", "enabled"},
		},
	},
	{
		Name:        "get_client_usage",
		Description: "Get token and request usage for a client",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"client_id": {Type: "string", Description: "Client ID or key hash"},
				"window":    {Type: "string", Description: "Time window (e.g. 1h, 24h, 7d)"},
			},
		},
	},
	{
		Name:        "initiate_oauth_login",
		Description: "Initiate OAuth login flow for a provider",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider name"},
			},
			Required: []string{"provider"},
		},
	},
}

func (s *Server) handleListTools() any {
	return map[string]any{
		"tools": standardTools,
	}
}

func (s *Server) handleCallTool(ctx context.Context, rawParams json.RawMessage) (*CallToolResult, *JSONRPCError) {
	var params CallToolParams
	if err := json.Unmarshal(rawParams, &params); err != nil {
		return nil, &JSONRPCError{
			Code:    CodeInvalidParams,
			Message: fmt.Sprintf("invalid call tool params: %v", err),
		}
	}

	switch params.Name {
	case "list_models":
		return s.toolListModels(ctx)
	case "get_quota_status":
		return s.toolGetQuotaStatus(ctx, params.Arguments)
	case "toggle_model":
		return s.toolToggleModel(ctx, params.Arguments)
	case "get_client_usage":
		return s.toolGetClientUsage(ctx, params.Arguments)
	case "initiate_oauth_login":
		return s.toolInitiateOAuthLogin(ctx, params.Arguments)
	default:
		return nil, &JSONRPCError{
			Code:    CodeMethodNotFound,
			Message: fmt.Sprintf("unknown tool: %q", params.Name),
		}
	}
}

func (s *Server) toolListModels(ctx context.Context) (*CallToolResult, *JSONRPCError) {
	var models any
	if s.stateSource != nil {
		snap := s.stateSource.Snapshot()
		if snap != nil {
			models = snap.Models
		}
	}
	if models == nil {
		models = map[string]any{}
	}

	b, _ := json.MarshalIndent(models, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{
			{
				Type: "text",
				Text: string(b),
			},
		},
	}, nil
}

func (s *Server) toolGetQuotaStatus(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	var args struct {
		Provider string `json:"provider"`
	}
	_ = json.Unmarshal(argsRaw, &args)
	if args.Provider == "" {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: provider argument is required"}},
			IsError: true,
		}, nil
	}

	if s.registry != nil {
		prov, ok := s.registry.Get(args.Provider)
		if ok && prov.Quota != nil {
			snap, err := prov.Quota.Quota(ctx)
			if err != nil {
				return &CallToolResult{
					Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error getting quota: %v", err)}},
					IsError: true,
				}, nil
			}
			b, _ := json.MarshalIndent(snap, "", "  ")
			return &CallToolResult{
				Content: []ToolContent{{Type: "text", Text: string(b)}},
			}, nil
		}
	}

	// Fallback to state snapshot
	if s.stateSource != nil {
		snap := s.stateSource.Snapshot()
		if snap != nil && snap.Providers != nil {
			if pr, ok := snap.Providers[args.Provider]; ok {
				b, _ := json.MarshalIndent(map[string]any{
					"provider": args.Provider,
					"quota":    pr.Quota,
					"admin":    pr.Admin,
					"health":   pr.Health,
				}, "", "  ")
				return &CallToolResult{
					Content: []ToolContent{{Type: "text", Text: string(b)}},
				}, nil
			}
		}
	}

	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("provider %q not found or quota not available", args.Provider)}},
		IsError: true,
	}, nil
}

func (s *Server) toolToggleModel(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	var args struct {
		ModelID string `json:"model_id"`
		Enabled bool   `json:"enabled"`
	}
	if err := json.Unmarshal(argsRaw, &args); err != nil || args.ModelID == "" {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: model_id and enabled parameters are required"}},
			IsError: true,
		}, nil
	}

	if s.stateSource == nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: state source not configured"}},
			IsError: true,
		}, nil
	}

	if err := s.stateSource.ToggleModel(args.ModelID, args.Enabled); err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("failed to toggle model: %v", err)}},
			IsError: true,
		}, nil
	}

	resp := map[string]any{
		"model_id": args.ModelID,
		"enabled":  args.Enabled,
		"status":   "updated",
	}
	b, _ := json.MarshalIndent(resp, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}, nil
}

func (s *Server) toolGetClientUsage(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	var args struct {
		ClientID string `json:"client_id"`
		Window   string `json:"window"`
	}
	_ = json.Unmarshal(argsRaw, &args)

	if s.usageSource != nil {
		usage, err := s.usageSource.GetClientUsage(ctx, args.ClientID, args.Window)
		if err != nil {
			return &CallToolResult{
				Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error retrieving client usage: %v", err)}},
				IsError: true,
			}, nil
		}
		b, _ := json.MarshalIndent(usage, "", "  ")
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: string(b)}},
		}, nil
	}

	// Default response if usage source not configured
	res := map[string]any{
		"client_id":      args.ClientID,
		"window":         args.Window,
		"total_requests": 0,
		"total_tokens":   0,
		"cost":           0.0,
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}, nil
}

func (s *Server) toolInitiateOAuthLogin(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	var args struct {
		Provider string `json:"provider"`
	}
	_ = json.Unmarshal(argsRaw, &args)
	if args.Provider == "" {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: provider argument is required"}},
			IsError: true,
		}, nil
	}

	if s.registry == nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: provider registry not configured"}},
			IsError: true,
		}, nil
	}

	prov, ok := s.registry.Get(args.Provider)
	if !ok || prov.Auth == nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("provider %q has no auth provider", args.Provider)}},
			IsError: true,
		}, nil
	}

	if err := prov.Auth.Login(ctx); err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("oauth login error: %v", err)}},
			IsError: true,
		}, nil
	}

	res := map[string]any{
		"status":   "initiated",
		"provider": args.Provider,
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}, nil
}

// Ensure unused import compiles cleanly
var _ = provider.ErrProviderNotFound
