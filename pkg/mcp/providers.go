package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"reflect"
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
	// Enrich returns the lane config a restart would restore (credential
	// injection + resolved discovery flag + real freebuff actor). toolAddProvider
	// builds the live lane from the enriched config so live == stored; Add
	// re-applies it as a no-op.
	Enrich(cfg openaicompat.Config) openaicompat.Config
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

// laneRebuilder is implemented by the real RuntimeProviderStore
// (pkg/server/rebuild.go) to rebuild one lane from its stored config after a
// credential change. It is asserted dynamically so this package never imports
// server (import cycle); stores that do not implement it use the local
// fallback in rebuildLaneAfterLogin, which mirrors the same Enrich → New →
// Register → bounded-discovery path through the ProviderStore interface.
type laneRebuilder interface {
	RebuildLane(registry *provider.Registry, name string) (int, error)
}

// hasLane reports whether the store holds any record (openai-compatible or
// custom) for name. ProviderStore.List covers only openai-compatible lanes;
// the real store also answers Has for custom kinds.
func hasLane(store ProviderStore, name string) bool {
	if store == nil {
		return false
	}
	if _, ok := store.List()[name]; ok {
		return true
	}
	if h, ok := store.(interface{ Has(string) bool }); ok {
		return h.Has(name)
	}
	return false
}

// rebuildLaneAfterLogin rebuilds one lane after CompleteLogin stored a fresh
// credential, so the completion reply can state the real routable state.
//
// It returns the post-login discovery count, an explanatory note ("" on a
// clean rebuild) and whether the lane is not held by the runtime store. The
// three outcomes map to the T004 acceptance criteria:
//   - success: (n, "", false) — registry serves the rebuilt lane, list_models
//     is fresh without a restart (AC1, AC2);
//   - not in store: (0, "", true) — the caller reports "credential stored;
//     lane not registered — call add_provider" (AC3), not an error and not a
//     false "usable";
//   - rebuild failure: (0, "lane rebuild failed: ... (previous lane
//     preserved)", false) — the previous lane is untouched (AC4).
//
// The preferred path delegates to the daemon store's RebuildLane (the same
// Enrich helper and discovery budget T003 established). Test doubles that do
// not implement it fall back to the identical path built from the
// ProviderStore interface.
func (s *Server) rebuildLaneAfterLogin(ctx context.Context, name string) (int, string, bool) {
	if s == nil || s.registry == nil || s.providers == nil {
		return 0, "lane rebuild skipped: provider store or registry not configured (previous lane preserved)", false
	}
	// Preferred: the daemon-owned store rebuilds from what a restart would
	// restore.
	if rb, ok := s.providers.(laneRebuilder); ok {
		discovered, err := rb.RebuildLane(s.registry, name)
		if err == nil {
			return discovered, "", false
		}
		msg := err.Error()
		if strings.Contains(msg, "not present in runtime store") {
			return 0, "", true
		}
		if strings.Contains(msg, "custom-wire lane") {
			return 0, "credential stored; custom-wire lane has no post-login rebuild (models addressed as <lane>/<model>)", false
		}
		return 0, fmt.Sprintf("lane rebuild failed: %v (previous lane preserved)", err), false
	}
	// Fallback for stores without RebuildLane (tests): same path through the
	// interface. List covers openai-compatible lanes; Has also catches custom
	// kinds so a stored custom lane is not misreported as unregistered.
	stored, ok := s.providers.List()[name]
	if !ok {
		if hasLane(s.providers, name) {
			return 0, "credential stored; custom-wire lane has no post-login rebuild (models addressed as <lane>/<model>)", false
		}
		return 0, "", true
	}
	cfg := s.providers.Enrich(stored)
	// T002 is preserved: OAuth lanes authenticate from the daemon-owned
	// store, never a TempDir fallback (mirrors toolAddProvider).
	if cfg.Quirks.AuthViaOAuthManager && cfg.Creds == nil && s.creds != nil {
		cfg.Creds = s.creds
	}
	p, err := openaicompat.New(cfg)
	if err != nil {
		return 0, fmt.Sprintf("lane rebuild failed: %v (previous lane preserved)", err), false
	}
	discovered, note := discoverModelsForReply(ctx, name, p.Provider())
	if strings.Contains(note, "failed") {
		return 0, fmt.Sprintf("lane rebuild failed: %s (previous lane preserved)", note), false
	}
	s.registry.Register(p.Provider())
	return discovered, note, false
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

// sameInterfaceValue reports whether two interface values are equal without
// panicking on non-comparable dynamics (e.g. an actor holding a slice). A
// non-comparable pair is treated as different so the drift path rebuilds.
func sameInterfaceValue(a, b any) (equal bool) {
	defer func() {
		if recover() != nil {
			equal = false
		}
	}()
	equal = (a == b)
	return equal
}

// isRealFreebuffActor reports whether v is a usable actor (not the MCP
// placeholder marker, which is a non-nil struct{} that implements nothing).
func isRealFreebuffActor(v any) bool {
	if v == nil {
		return false
	}
	_, ok := v.(openaicompat.FreebuffActor)
	return ok
}

// laneConfigsDrifted reports whether the config the live lane was built from
// differs from what the store persisted for the same lane. Enrich-before-build
// plus Add-as-no-op should make this false; a true result means a future
// enrichment changed one side only, so the caller rebuilds the live lane from
// the stored config (what a restart would restore).
func laneConfigsDrifted(built, stored openaicompat.Config) bool {
	if built.Name != stored.Name ||
		built.BaseURL != stored.BaseURL ||
		built.APIKey != stored.APIKey ||
		built.DataDir != stored.DataDir ||
		built.OptOutModelListPassthrough != stored.OptOutModelListPassthrough {
		return true
	}
	if (built.Creds == nil) != (stored.Creds == nil) {
		return true
	}
	if built.Creds != nil && stored.Creds != nil && !sameInterfaceValue(built.Creds, stored.Creds) {
		return true
	}
	bq, sq := built.Quirks, stored.Quirks
	if bq.CodingPlanPath != sq.CodingPlanPath ||
		bq.EchoReasoning != sq.EchoReasoning ||
		bq.ModelListPassthrough != sq.ModelListPassthrough ||
		bq.AuthViaOAuthManager != sq.AuthViaOAuthManager ||
		bq.CreditsQuotaObserver != sq.CreditsQuotaObserver ||
		bq.AuthViaSupabaseRefresh != sq.AuthViaSupabaseRefresh ||
		bq.FreebuffDefaultTool != sq.FreebuffDefaultTool ||
		bq.DefaultModel != sq.DefaultModel {
		return true
	}
	if !reflect.DeepEqual(bq.MaxTokensByModel, sq.MaxTokensByModel) {
		return true
	}
	if (bq.FreebuffActor == nil) != (sq.FreebuffActor == nil) {
		return true
	}
	if isRealFreebuffActor(bq.FreebuffActor) != isRealFreebuffActor(sq.FreebuffActor) {
		return true
	}
	if bq.FreebuffActor != nil && sq.FreebuffActor != nil && !sameInterfaceValue(bq.FreebuffActor, sq.FreebuffActor) {
		return true
	}
	return false
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

	// Single lane build path (T003): enrich first so the live lane is built
	// from exactly what a restart would restore. T002 is preserved: OAuth lanes
	// authenticate from the daemon-owned store, never a TempDir fallback. The
	// MCP handle (s.creds) is injected before Enrich so the store sees it; if
	// the store carries no Creds the handle is applied again after Enrich, and
	// a missing store still fails loudly in New with "credential store not
	// injected".
	cfgRaw := args.config()
	if cfgRaw.Quirks.AuthViaOAuthManager && cfgRaw.Creds == nil && s.creds != nil {
		cfgRaw.Creds = s.creds
	}
	cfg := s.providers.Enrich(cfgRaw)
	if cfg.Quirks.AuthViaOAuthManager && cfg.Creds == nil && s.creds != nil {
		cfg.Creds = s.creds
	}

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
	// Add re-applies Enrich as a no-op, so the stored config should equal the
	// built one. Re-read as a drift assertion: any future enrichment that
	// changes one side only rebuilds the live lane from the stored config
	// instead of shipping the drift (this subsumes the old freebuff-only and
	// Creds-only rebuild branches).
	if stored, ok := s.providers.List()[cfg.Name]; ok {
		if laneConfigsDrifted(cfg, stored) {
			log.Printf("[mcp] add_provider drift for %q: built config differs from stored, rebuilding live lane from stored", cfg.Name)
			if rebuilt, err := openaicompat.New(stored); err == nil {
				p = rebuilt
				cfg = stored
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
