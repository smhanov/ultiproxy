package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
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
// the add_provider tool. Quirks.FreebuffActor cannot cross the MCP boundary (it
// is an injected *FreebuffAccountActor, not JSON-serializable), so freebuff
// lanes are declared with the freebuff_actor boolean: the placeholder it
// installs marks the lane as freebuff so the provider store persists
// quirks.freebuff_actor=true and rebuilds the real actor from the lane's
// api_key / state token (see RuntimeProviderStore.Add and Restore).
type providerQuirksArgs struct {
	CodingPlanPath   bool           `json:"coding_plan_path"`
	MaxTokensByModel map[string]int `json:"max_tokens_by_model"`
	EchoReasoning    bool           `json:"echo_reasoning"`
	// ModelListPassthrough is a pointer so "not specified" (discovery is the
	// default for OpenAI-compatible lanes) stays distinguishable from an
	// explicit false opt-out.
	ModelListPassthrough   *bool  `json:"model_list_passthrough"`
	AuthViaOAuthManager    bool   `json:"auth_via_oauth_manager"`
	CreditsQuotaObserver   string `json:"credits_quota_observer"`
	AuthViaSupabaseRefresh bool   `json:"auth_via_supabase_refresh"`
	FreebuffActor          bool   `json:"freebuff_actor"`
	FreebuffDefaultTool    bool   `json:"freebuff_default_tool"`
	DefaultModel           string `json:"default_model"`
}

type addProviderArgs struct {
	Name    string             `json:"name"`
	Kind    string             `json:"kind"` // "" or "openaicompat" | "antigravity" | "anthropic" | "codex"
	BaseURL string             `json:"base_url"`
	APIKey  string             `json:"api_key"`
	Quirks  providerQuirksArgs `json:"quirks"`
}

func (a addProviderArgs) config() openaicompat.Config {
	cfg := openaicompat.Config{
		Name:    a.Name,
		BaseURL: a.BaseURL,
		APIKey:  a.APIKey,
		Quirks: openaicompat.Quirks{
			CodingPlanPath:         a.Quirks.CodingPlanPath,
			MaxTokensByModel:       a.Quirks.MaxTokensByModel,
			EchoReasoning:          a.Quirks.EchoReasoning,
			AuthViaOAuthManager:    a.Quirks.AuthViaOAuthManager,
			CreditsQuotaObserver:   a.Quirks.CreditsQuotaObserver,
			AuthViaSupabaseRefresh: a.Quirks.AuthViaSupabaseRefresh,
			FreebuffDefaultTool:    a.Quirks.FreebuffDefaultTool,
			DefaultModel:           a.Quirks.DefaultModel,
		},
	}
	if a.Quirks.ModelListPassthrough != nil && !*a.Quirks.ModelListPassthrough {
		// quirks.model_list_passthrough is the opt-OUT: absent keeps the
		// default (discovery ON), false disables it. The effective flag is
		// resolved by openaicompat.New and the provider store, never here.
		cfg.OptOutModelListPassthrough = true
	}
	if a.Quirks.FreebuffActor {
		// Non-nil placeholder: it marks the lane as a freebuff lane (serialized
		// requests, session affinity, actor-backed quota) so the store persists
		// freebuff_actor=true. The store swaps it for the real actor built from
		// the lane's api_key / state token; without one it stays a marker and
		// quota reports the missing actor honestly.
		cfg.Quirks.FreebuffActor = struct{}{}
	}
	return cfg
}

// modelDiscoveryBudget bounds the synchronous model discovery add_provider runs
// before replying. It matches the budget the server registration, startup and
// scheduled discovery passes use, so a slow upstream cannot hang the tool call.
const modelDiscoveryBudget = 5 * time.Second

// modelsCacher is the cache side of a lane discovery capability: the ids the
// lane already holds without contacting the upstream.
type modelsCacher interface {
	CachedModels() []string
}

// discoverModelsForReply runs (or reuses) one lane model discovery and returns
// the count plus an explanatory note for the add_provider reply.
// openaicompat.New already discovers while validating the lane, so a lane whose
// construction discovery succeeded reports that result instead of dialling the
// upstream a second time; an empty cache (construction discovery failed) is
// retried here so the reply never understates a lane that just came up.
func discoverModelsForReply(ctx context.Context, name string, bundle provider.Provider) (int, string) {
	if bundle.Inference == nil {
		return 0, "lane has no inference surface"
	}
	if opt, ok := bundle.Inference.(interface{ ModelDiscoveryEnabled() bool }); ok && !opt.ModelDiscoveryEnabled() {
		return 0, "model discovery disabled for this lane (quirks.model_list_passthrough=false)"
	}
	fetcher, ok := bundle.Inference.(modelsFetcher)
	if !ok {
		return 0, "lane does not support model discovery, its models are addressed as <lane>/<model>"
	}
	if cacher, ok := bundle.Inference.(modelsCacher); ok {
		if cached := cacher.CachedModels(); len(cached) > 0 {
			return len(cached), ""
		}
	}
	fetchCtx, cancel := context.WithTimeout(ctx, modelDiscoveryBudget)
	defer cancel()
	models, err := fetcher.FetchModels(fetchCtx)
	if err != nil {
		return 0, fmt.Sprintf("model discovery failed: %v", err)
	}
	return len(models), ""
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
	// The store may enrich what was added (the freebuff actor cannot cross the
	// MCP boundary): re-read the stored config and rebuild when it changed so
	// the lane registered right now matches what a restart would restore.
	if stored, ok := s.providers.List()[cfg.Name]; ok {
		if stored.Quirks.FreebuffActor != nil && cfg.Quirks.FreebuffActor != stored.Quirks.FreebuffActor {
			if rebuilt, err := openaicompat.New(stored); err == nil {
				p = rebuilt
			}
		}
	}
	if s.registry != nil {
		s.registry.Register(p.Provider())
	}

	// Synchronous model discovery, BEFORE replying: the reply then states what
	// the lane serves (its "<lane>/<model>" ids are already routable and
	// already on /v1/models) and the caller needs no refresh_models follow-up.
	discovered, note := discoverModelsForReply(ctx, cfg.Name, p.Provider())
	summary := fmt.Sprintf("discovered %d models", discovered)
	if note != "" {
		summary += " (" + note + ")"
	}
	return toolJSON(map[string]any{
		"registered":        true,
		"name":              cfg.Name,
		"lane":              cfg.Name,
		"discovered_models": discovered,
		"summary":           summary,
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
	// Custom-wire lanes have no model discovery: their model set is addressed
	// as <lane>/<model> and expressed through aliases, never invented here.
	return toolJSON(map[string]any{
		"registered":        true,
		"name":              args.Name,
		"lane":              args.Name,
		"kind":              args.Kind,
		"discovered_models": 0,
		"summary":           "discovered 0 models (custom-wire lane: no model discovery, address models as <lane>/<model> or set_model_alias)",
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

// toolListProviders lists runtime-registered lanes. Secrets (the api_key)
// are never returned, only a presence boolean.
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
			"name":           name,
			"base_url":       cfg.BaseURL,
			"has_api_key":    cfg.APIKey != "",
			"auth_via_oauth": cfg.Quirks.AuthViaOAuthManager || cfg.Quirks.AuthViaSupabaseRefresh,
			"quirks": map[string]any{
				"coding_plan_path":          cfg.Quirks.CodingPlanPath,
				"echo_reasoning":            cfg.Quirks.EchoReasoning,
				"model_list_passthrough":    cfg.Quirks.ModelListPassthrough,
				"auth_via_oauth_manager":    cfg.Quirks.AuthViaOAuthManager,
				"auth_via_supabase_refresh": cfg.Quirks.AuthViaSupabaseRefresh,
				"credits_quota_observer":    cfg.Quirks.CreditsQuotaObserver,
				"freebuff_actor":            cfg.Quirks.FreebuffActor != nil,
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
