package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

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
		Description: "Start the OAuth login flow for a provider. Returns the sign-in URL (and user code for device flows) WITHOUT blocking; then call check_oauth_login to poll device flows or submit_oauth_code to finish auth-code flows.",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider name"},
			},
			Required: []string{"provider"},
		},
	},
	{
		Name:        "check_oauth_login",
		Description: "Poll a pending OAuth login (device flows: xai etc.) until the user approves; returns completed or pending. Also finalizes auth-code flows whose token exchange can complete server-side.",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider name"},
			},
			Required: []string{"provider"},
		},
	},
	{
		Name:        "submit_oauth_code",
		Description: "Submit the authorization code from the browser (auth-code flows: antigravity) to finish OAuth login.",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider name"},
				"code":     {Type: "string", Description: "Authorization code from the browser callback/success page"},
			},
			Required: []string{"provider", "code"},
		},
	},
	{
		Name:        "list_model_aliases",
		Description: "List all model aliases (client-visible name -> provider lane + upstream id)",
		InputSchema: &InputSchema{Type: "object", Properties: map[string]PropertyDef{}},
	},
	{
		Name:        "set_model_alias",
		Description: "Set a model alias mapping a client-visible name to a provider lane and upstream model id. Persists across restarts.",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"alias":         {Type: "string", Description: "Client-visible model name, e.g. qwenpoint-3.8"},
				"provider":      {Type: "string", Description: "Provider lane name, e.g. vllm, zai"},
				"upstream":      {Type: "string", Description: "Upstream model id, e.g. Qwen/Qwen3.8-Instruct-AWQ"},
				"context_limit": {Type: "number", Description: "Optional context window size"},
				"max_output":    {Type: "number", Description: "Optional max output tokens"},
				"pricing_tag":   {Type: "string", Description: "Optional pricing label"},
			},
			Required: []string{"alias", "provider", "upstream"},
		},
	},
	{
		Name:        "remove_model_alias",
		Description: "Remove a model alias mapping",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"alias": {Type: "string", Description: "Alias to remove"},
			},
			Required: []string{"alias"},
		},
	},
	{
		Name:        "get_provider_timeouts",
		Description: "List per-provider request timeouts (Go duration strings)",
		InputSchema: &InputSchema{Type: "object", Properties: map[string]PropertyDef{}},
	},
	{
		Name:        "set_provider_timeout",
		Description: "Set a per-provider request timeout, e.g. {\"provider\":\"vllm\",\"timeout\":\"10m\"}. Persists across restarts.",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider lane name, e.g. vllm, freebuff"},
				"timeout":  {Type: "string", Description: "Duration string, e.g. 3m30s, 10m, 1h"},
			},
			Required: []string{"provider", "timeout"},
		},
	},
	{
		Name:        "remove_provider_timeout",
		Description: "Clear a provider's explicit timeout (falls back to default)",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider lane name"},
			},
			Required: []string{"provider"},
		},
	},
	{
		Name:        "add_provider",
		Description: "Register an OpenAI-compatible provider lane at runtime (no config file). Persists across restarts in providers.json.",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"name":            {Type: "string", Description: "Lane name, lowercase [a-z0-9_-], e.g. vllm, zai, openrouter"},
				"base_url":        {Type: "string", Description: "Upstream base URL, e.g. https://api.deepseek.com/v1"},
				"api_key":         {Type: "string", Description: "Optional static API key"},
				"data_dir":        {Type: "string", Description: "Optional data dir for token/credential storage"},
				"workspace_id":    {Type: "string", Description: "Workspace id for auth_via_workspace_cookie (opencode)"},
				"session_cookie":  {Type: "string", Description: "Session cookie for auth_via_workspace_cookie (opencode)"},
				"refresh_url":     {Type: "string", Description: "Supabase refresh URL for auth_via_supabase_refresh (augure)"},
				"token_file":      {Type: "string", Description: "Token file for auth_via_supabase_refresh (augure)"},
				"device_auth_url": {Type: "string", Description: "Device authorization URL for auth_via_oauth_manager"},
				"token_url":       {Type: "string", Description: "Token URL for auth_via_oauth_manager"},
				"quirks": {
					Type:        "object",
					Description: "Vendor quirks: {coding_plan_path, max_tokens_by_model, echo_reasoning, model_list_passthrough, auth_via_workspace_cookie, auth_via_oauth_manager, credits_quota_observer, auth_via_supabase_refresh, freebuff_default_tool, default_model}",
				},
			},
			Required: []string{"name", "base_url"},
		},
	},
	{
		Name:        "remove_provider",
		Description: "Unregister a provider lane: drops it from the live registry and from providers.json.",
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"name": {Type: "string", Description: "Lane name to remove, e.g. vllm"},
			},
			Required: []string{"name"},
		},
	},
	{
		Name:        "list_providers",
		Description: "List runtime-registered provider lanes (base URL + quirks; secrets are redacted) and the live registry lanes.",
		InputSchema: &InputSchema{Type: "object", Properties: map[string]PropertyDef{}},
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
	case "check_oauth_login":
		return s.toolCheckOAuthLogin(ctx, params.Arguments)
	case "submit_oauth_code":
		return s.toolSubmitOAuthCode(ctx, params.Arguments)
	case "list_model_aliases":
		return s.toolListModelAliases(ctx)
	case "set_model_alias":
		return s.toolSetModelAlias(ctx, params.Arguments)
	case "remove_model_alias":
		return s.toolRemoveModelAlias(ctx, params.Arguments)
	case "get_provider_timeouts":
		return s.toolGetProviderTimeouts(ctx)
	case "set_provider_timeout":
		return s.toolSetProviderTimeout(ctx, params.Arguments)
	case "remove_provider_timeout":
		return s.toolRemoveProviderTimeout(ctx, params.Arguments)
	case "add_provider":
		return s.toolAddProvider(ctx, params.Arguments)
	case "remove_provider":
		return s.toolRemoveProvider(ctx, params.Arguments)
	case "list_providers":
		return s.toolListProviders(ctx)
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

	// Two-phase interactive flow: return the sign-in URL immediately so an
	// agent (or FC browser) can present it without blocking.
	if interactive, ok := prov.Auth.(provider.InteractiveAuthProvider); ok {
		info, err := interactive.StartLogin(ctx)
		if err != nil {
			return &CallToolResult{
				Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("oauth login error: %v", err)}},
				IsError: true,
			}, nil
		}
		res := map[string]any{
			"status":             "awaiting_user",
			"provider":           args.Provider,
			"kind":               info.Kind,
			"url":                info.VerificationURI,
			"expires_in_seconds": info.ExpiresIn,
		}
		if info.UserCode != "" {
			res["user_code"] = info.UserCode
		}
		b, _ := json.MarshalIndent(res, "", "  ")
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: string(b)}},
		}, nil
	}

	// Legacy blocking providers: run the full flow (may block on stdin).
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

// toolCheckOAuthLogin polls a pending device flow (CompleteLogin with a short
// context). It does NOT block forever: returns pending when the user has not
// approved yet, completed when the token is stored.
func (s *Server) toolCheckOAuthLogin(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
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
	interactive, ok := prov.Auth.(provider.InteractiveAuthProvider)
	if !ok {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("provider %q does not support interactive login", args.Provider)}},
			IsError: true,
		}, nil
	}

	// Bounded poll: device flows typically take the human 10-120s after
	// opening the URL. CompleteLogin internally polls with short sleeps; a
	// ctx timeout keeps one MCP call from hanging indefinitely.
	pollCtx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()
	// CompleteLogin for device flows ignores the code and polls; for
	// auth-code flows the code is required, so do not finalize here.
	if err := interactive.CompleteLogin(pollCtx, ""); err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			b, _ := json.MarshalIndent(map[string]any{
				"status":   "pending",
				"provider": args.Provider,
			}, "", "  ")
			return &CallToolResult{Content: []ToolContent{{Type: "text", Text: string(b)}}}, nil
		}
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("oauth login error: %v", err)}},
			IsError: true,
		}, nil
	}

	b, _ := json.MarshalIndent(map[string]any{
		"status":   "completed",
		"provider": args.Provider,
	}, "", "  ")
	return &CallToolResult{Content: []ToolContent{{Type: "text", Text: string(b)}}}, nil
}

// toolSubmitOAuthCode finishes an auth-code flow (antigravity) with the code
// the user copied from the browser after consent.
func (s *Server) toolSubmitOAuthCode(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	var args struct {
		Provider string `json:"provider"`
		Code     string `json:"code"`
	}
	_ = json.Unmarshal(argsRaw, &args)
	if args.Provider == "" || args.Code == "" {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: provider and code are required"}},
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
	interactive, ok := prov.Auth.(provider.InteractiveAuthProvider)
	if !ok {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("provider %q does not support interactive login", args.Provider)}},
			IsError: true,
		}, nil
	}
	if err := interactive.CompleteLogin(ctx, args.Code); err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("oauth login error: %v", err)}},
			IsError: true,
		}, nil
	}
	b, _ := json.MarshalIndent(map[string]any{
		"status":   "completed",
		"provider": args.Provider,
	}, "", "  ")
	return &CallToolResult{Content: []ToolContent{{Type: "text", Text: string(b)}}}, nil
}

// Ensure unused import compiles cleanly
var _ = provider.ErrProviderNotFound

func (s *Server) toolListModelAliases(ctx context.Context) (*CallToolResult, *JSONRPCError) {
	if s.aliases == nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: model alias manager not configured"}},
			IsError: true,
		}, nil
	}
	aliases := s.aliases.List()
	if aliases == nil {
		aliases = map[string]ModelAlias{}
	}
	b, _ := json.MarshalIndent(aliases, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}, nil
}

func (s *Server) toolSetModelAlias(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	if s.aliases == nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: model alias manager not configured"}},
			IsError: true,
		}, nil
	}
	var args struct {
		Alias        string             `json:"alias"`
		Provider     string             `json:"provider"`
		Upstream     string             `json:"upstream"`
		ContextLimit int                `json:"context_limit"`
		MaxOutput    int                `json:"max_output"`
		PricingTag   string             `json:"pricing_tag"`
		Benchmarks   map[string]float64 `json:"benchmarks"`
	}
	if err := json.Unmarshal(argsRaw, &args); err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error: %v", err)}},
			IsError: true,
		}, nil
	}
	entry := ModelAlias{
		Provider:        args.Provider,
		Upstream:        args.Upstream,
		ContextLimit:    args.ContextLimit,
		MaxOutput:       args.MaxOutput,
		PricingTag:      args.PricingTag,
		BenchmarkScores: args.Benchmarks,
	}
	if args.Alias == "" || entry.Provider == "" || entry.Upstream == "" {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: alias, provider and upstream are required"}},
			IsError: true,
		}, nil
	}
	if err := s.aliases.Set(args.Alias, entry); err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error: %v", err)}},
			IsError: true,
		}, nil
	}
	res := map[string]any{"ok": true, "alias": args.Alias, "entry": entry}
	b, _ := json.MarshalIndent(res, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}, nil
}

func (s *Server) toolRemoveModelAlias(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	if s.aliases == nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: model alias manager not configured"}},
			IsError: true,
		}, nil
	}
	var args struct {
		Alias string `json:"alias"`
	}
	_ = json.Unmarshal(argsRaw, &args)
	if args.Alias == "" {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: alias argument is required"}},
			IsError: true,
		}, nil
	}
	if err := s.aliases.Remove(args.Alias); err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error: %v", err)}},
			IsError: true,
		}, nil
	}
	res := map[string]any{"ok": true, "removed": args.Alias}
	b, _ := json.MarshalIndent(res, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}, nil
}

func (s *Server) toolGetProviderTimeouts(ctx context.Context) (*CallToolResult, *JSONRPCError) {
	if s.timeouts == nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: timeout manager not configured"}},
			IsError: true,
		}, nil
	}
	timeouts := s.timeouts.List()
	if timeouts == nil {
		timeouts = map[string]string{}
	}
	b, _ := json.MarshalIndent(timeouts, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}, nil
}

func (s *Server) toolSetProviderTimeout(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	if s.timeouts == nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: timeout manager not configured"}},
			IsError: true,
		}, nil
	}
	var args struct {
		Provider string `json:"provider"`
		Timeout  string `json:"timeout"`
	}
	if err := json.Unmarshal(argsRaw, &args); err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error: %v", err)}},
			IsError: true,
		}, nil
	}
	if args.Provider == "" || args.Timeout == "" {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: provider and timeout are required"}},
			IsError: true,
		}, nil
	}
	if err := s.timeouts.Set(args.Provider, args.Timeout); err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error: %v", err)}},
			IsError: true,
		}, nil
	}
	res := map[string]any{"ok": true, "provider": args.Provider, "timeout": args.Timeout}
	b, _ := json.MarshalIndent(res, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}, nil
}

func (s *Server) toolRemoveProviderTimeout(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	if s.timeouts == nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: timeout manager not configured"}},
			IsError: true,
		}, nil
	}
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
	if err := s.timeouts.Remove(args.Provider); err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error: %v", err)}},
			IsError: true,
		}, nil
	}
	res := map[string]any{"ok": true, "removed": args.Provider}
	b, _ := json.MarshalIndent(res, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}, nil
}
