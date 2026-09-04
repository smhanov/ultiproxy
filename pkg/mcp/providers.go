package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// ProviderStore is implemented by the http Server's RuntimeProviderStore
// (pkg/server/providers.go). The interface lives here because mcp cannot
// import server (import cycle); satisfaction is structural.
type ProviderStore interface {
	// Add validates + stores a lane config and persists it (providers.json).
	Add(cfg openaicompat.Config) error
	// AddCustom stores a non-OpenAI-compatible lane (kind, plus the api_key
	// for kinds that need a static key) and persists it. Used for
	// antigravity-like compile-time-wired lanes that the store can hold but
	// only the server's LaneBuilder can reconstruct. Storage lives in the
	// server's general DataDir — lanes never carry their own data dir. apiKey
	// is empty for kinds that authenticate through a credential store instead.
	AddCustom(name, kind, apiKey string) error
	// Remove deletes a persisted lane.
	Remove(name string) error
	// List returns the stored lane configs keyed by lane name.
	List() map[string]openaicompat.Config
}

// providerQuirksArgs is the JSON-able subset of openaicompat.Quirks accepted by
// the add_provider tool. Quirks.FreebuffActor is deliberately absent: it is an
// injected *FreebuffAccountActor (not JSON-serializable), so freebuff lanes stay
// compile-time registrations for now.
type providerQuirksArgs struct {
	CodingPlanPath         bool           `json:"coding_plan_path"`
	MaxTokensByModel       map[string]int `json:"max_tokens_by_model"`
	EchoReasoning          bool           `json:"echo_reasoning"`
	ModelListPassthrough   bool           `json:"model_list_passthrough"`
	AuthViaWorkspaceCookie bool           `json:"auth_via_workspace_cookie"`
	AuthViaOAuthManager    bool           `json:"auth_via_oauth_manager"`
	CreditsQuotaObserver   string         `json:"credits_quota_observer"`
	AuthViaSupabaseRefresh bool           `json:"auth_via_supabase_refresh"`
	FreebuffDefaultTool    bool           `json:"freebuff_default_tool"`
	DefaultModel           string         `json:"default_model"`
}

type addProviderArgs struct {
	Name    string             `json:"name"`
	Kind    string             `json:"kind"` // "" or "openaicompat" | "antigravity" | "anthropic" | "codex"
	BaseURL string             `json:"base_url"`
	APIKey  string             `json:"api_key"`
	Quirks  providerQuirksArgs `json:"quirks"`
}

func (a addProviderArgs) config() openaicompat.Config {
	return openaicompat.Config{
		Name:    a.Name,
		BaseURL: a.BaseURL,
		APIKey:  a.APIKey,
		Quirks: openaicompat.Quirks{
			CodingPlanPath:         a.Quirks.CodingPlanPath,
			MaxTokensByModel:       a.Quirks.MaxTokensByModel,
			EchoReasoning:          a.Quirks.EchoReasoning,
			ModelListPassthrough:   a.Quirks.ModelListPassthrough,
			AuthViaWorkspaceCookie: a.Quirks.AuthViaWorkspaceCookie,
			AuthViaOAuthManager:    a.Quirks.AuthViaOAuthManager,
			CreditsQuotaObserver:   a.Quirks.CreditsQuotaObserver,
			AuthViaSupabaseRefresh: a.Quirks.AuthViaSupabaseRefresh,
			FreebuffDefaultTool:    a.Quirks.FreebuffDefaultTool,
			DefaultModel:           a.Quirks.DefaultModel,
		},
	}
}

func toolError(format string, args ...any) *CallToolResult {
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: "error: " + fmt.Sprintf(format, args...)}},
		IsError: true,
	}
}

func toolJSON(payload any) *CallToolResult {
	b, _ := json.MarshalIndent(payload, "", "  ")
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: string(b)}},
	}
}

// toolAddProvider registers an OpenAI-compatible lane at runtime: validate by
// constructing the adapter, register into the live registry, persist to
// providers.json so the lane survives a restart.
func (s *Server) toolAddProvider(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	if s.providers == nil {
		return toolError("runtime provider store not configured"), nil
	}
	var args addProviderArgs
	if err := json.Unmarshal(argsRaw, &args); err != nil {
		return toolError("invalid arguments: %v", err), nil
	}
	if args.Name == "" {
		return toolError("name is required"), nil
	}
	// Custom-wire lanes (antigravity, anthropic, codex) do not go through the
	// openai-compat path; route them to the injected builder instead.
	if args.Kind != "" && args.Kind != "openaicompat" {
		return s.toolAddCustomProvider(ctx, args)
	}
	if args.BaseURL == "" {
		return toolError("base_url is required"), nil
	}

	cfg := args.config()

	// Validate by building the adapter. New() does not dial the upstream
	// (except the optional model-list passthrough discovery, whose failure is
	// non-fatal), so a bad base URL still has to look structurally sound.
	p, err := openaicompat.New(cfg)
	if err != nil {
		return toolError("provider %q failed validation: %v", args.Name, err), nil
	}

	if err := s.providers.Add(cfg); err != nil {
		return toolError("%v", err), nil
	}
	if s.registry != nil {
		s.registry.Register(p.Provider())
	}

	return toolJSON(map[string]any{
		"registered": true,
		"name":       cfg.Name,
		"lane":       cfg.Name,
	}), nil
}

// customProviderKindsRequireAPIKey lists custom lane kinds whose upstream
// authenticates with a static API key handed in through the add_provider call;
// every other custom kind uses the ultiproxy-owned credential store instead.
var customProviderKindsRequireAPIKey = map[string]bool{
	"anthropic": true,
}

// validateCustomProviderArgs enforces the per-kind requirements of custom
// lanes before the lane is built or persisted.
func validateCustomProviderArgs(args addProviderArgs) error {
	if customProviderKindsRequireAPIKey[args.Kind] && strings.TrimSpace(args.APIKey) == "" {
		return fmt.Errorf("kind %q requires api_key", args.Kind)
	}
	return nil
}

// toolAddCustomProvider registers a runtime lane that is not
// OpenAI-compatible (kind=antigravity, anthropic, codex) via the injected
// builder. The lane persists in providers.json alongside openai-compatible
// lanes; the api_key of key-authenticated kinds (anthropic) is persisted too,
// so the builder can reconstruct the lane after a restart.
func (s *Server) toolAddCustomProvider(ctx context.Context, args addProviderArgs) (*CallToolResult, *JSONRPCError) {
	if args.Name == "" {
		return toolError("name is required"), nil
	}
	if err := validateCustomProviderArgs(args); err != nil {
		return toolError("%v", err), nil
	}
	if s.customLaneBuilder == nil {
		return toolError("custom lanes (kind=%q) are not wired on this server", args.Kind), nil
	}
	bundle, err := s.customLaneBuilder(args.Name, args.Kind, args.APIKey)
	if err != nil {
		return toolError("provider %q failed to build: %v", args.Name, err), nil
	}
	if err := s.providers.AddCustom(args.Name, args.Kind, args.APIKey); err != nil {
		return toolError("%v", err), nil
	}
	if s.registry != nil {
		s.registry.Register(bundle)
	}
	return toolJSON(map[string]any{
		"registered": true,
		"name":       args.Name,
		"lane":       args.Name,
		"kind":       args.Kind,
	}), nil
}

// toolRemoveProvider unregisters a lane: drop it from the live registry and
// from providers.json. Compile-time lanes (env/credential-file registered) are
// not in the store, but removing them from the live registry still works and is
// reported as not persisted.
func (s *Server) toolRemoveProvider(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	if s.providers == nil {
		return toolError("runtime provider store not configured"), nil
	}
	var args struct {
		Name string `json:"name"`
	}
	_ = json.Unmarshal(argsRaw, &args)
	if args.Name == "" {
		return toolError("name argument is required"), nil
	}

	persisted := true
	stored := s.providers.List()
	_, inStore := stored[args.Name]
	if err := s.providers.Remove(args.Name); err != nil {
		if !inStore {
			persisted = false // not a runtime-registered lane
		} else {
			return toolError("%v", err), nil
		}
	}

	unregistered := false
	if s.registry != nil {
		unregistered = s.registry.Unregister(args.Name)
	}

	if !inStore && !unregistered {
		return toolError("provider %q is not registered", args.Name), nil
	}

	return toolJSON(map[string]any{
		"ok":           true,
		"removed":      args.Name,
		"persisted":    persisted,
		"was_stored":   inStore,
		"unregistered": unregistered,
	}), nil
}

// toolListProviders lists runtime-registered lanes. Secrets (api_key,
// session_cookie) are never returned, only presence booleans.
func (s *Server) toolListProviders(ctx context.Context) (*CallToolResult, *JSONRPCError) {
	stored := map[string]openaicompat.Config{}
	if s.providers != nil {
		stored = s.providers.List()
	}

	names := make([]string, 0, len(stored))
	for name := range stored {
		names = append(names, name)
	}
	sort.Strings(names)

	lanes := make([]map[string]any, 0, len(stored))
	for _, name := range names {
		cfg := stored[name]
		lanes = append(lanes, map[string]any{
			"name":               name,
			"base_url":           cfg.BaseURL,
			"data_dir":           cfg.DataDir,
			"has_api_key":        cfg.APIKey != "",
			"has_session_cookie": cfg.SessionCookie != "",
			"has_workspace_id":   cfg.WorkspaceID != "",
			"auth_via_oauth":     cfg.Quirks.AuthViaOAuthManager || cfg.Quirks.AuthViaSupabaseRefresh || cfg.Quirks.AuthViaWorkspaceCookie,
			"quirks": map[string]any{
				"coding_plan_path":          cfg.Quirks.CodingPlanPath,
				"echo_reasoning":            cfg.Quirks.EchoReasoning,
				"model_list_passthrough":    cfg.Quirks.ModelListPassthrough,
				"auth_via_workspace_cookie": cfg.Quirks.AuthViaWorkspaceCookie,
				"auth_via_oauth_manager":    cfg.Quirks.AuthViaOAuthManager,
				"auth_via_supabase_refresh": cfg.Quirks.AuthViaSupabaseRefresh,
				"credits_quota_observer":    cfg.Quirks.CreditsQuotaObserver,
				"freebuff_default_tool":     cfg.Quirks.FreebuffDefaultTool,
				"default_model":             cfg.Quirks.DefaultModel,
				"max_tokens_by_model":       cfg.Quirks.MaxTokensByModel,
			},
		})
	}

	registryLanes := []string{}
	if s.registry != nil {
		registryLanes = s.registry.Names()
	}

	return toolJSON(map[string]any{
		"providers":      lanes,
		"count":          len(lanes),
		"registry_lanes": registryLanes,
	}), nil
}
