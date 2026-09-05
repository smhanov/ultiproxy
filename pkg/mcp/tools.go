package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
)

var standardTools = []Tool{
	{
		Name: "list_models",
		Description: `List the client-visible model ids Ultiproxy can route right now.

Returns a JSON object mapping every model id to its metadata:
- id: the exact string a client passes as "model" in POST /v1/chat/completions and POST /v1/messages
- provider: the lane serving it (antigravity, xai, copilot, codex, freebuff, zai, or any lane added with add_provider)
- enabled: whether the id is routable right now (flip it with toggle_model)
- context_limit: context window in tokens; advisory metadata surfaced as context_length on GET /v1/models, never enforced against the prompt
- max_output: hard cap on generated tokens; a request's max_tokens is clamped down to it
- pricing_tag, benchmark_scores: pricing / quality labels carried by the alias
- source: where the id comes from - "alias" (set_model_alias catalog), "discovery" (the lane's cached upstream model list, exposed as "<lane>/<model>"), or "default" (a lane's default model, exposed as "<lane>/<default>", e.g. antigravity/gemini-3.7-flash-high)

This is the same id set GET /v1/models advertises. Only routable ids appear: a bare lane name is never listed, because a lane name is a routing prefix, not a model. Lanes whose discovery cache is empty and that have no default model contribute nothing (refresh_models refills the cache). Ids disabled with toggle_model stay listed with "enabled": false so they can be switched back on, while GET /v1/models omits them. Setting ULTIPROXY_HIDE_TEST_LANES=1 additionally drops clearly-test lanes such as "probe" or "fake" from both surfaces.

One routing shape is deliberately NOT advertised but still works, for backward compatibility: "model": "<lane>" with no slash routes to that lane and lets it pick its own upstream model. Prefer the listed ids. Use list_providers for lane-level inventory/health and list_model_aliases for the raw alias table.`,
		InputSchema: &InputSchema{
			Type:       "object",
			Properties: map[string]PropertyDef{},
		},
	},
	{
		Name: "get_quota_status",
		Description: `Fetch the live quota / credit state of ONE upstream provider lane.

provider must be a lane name exactly as list_providers reports it, e.g. "antigravity", "xai", "copilot", "codex", "freebuff", "zai", or a lane registered with add_provider.

Returns the lane's normalized quota snapshot as JSON:
- observed_at: when the upstream was last asked
- windows[]: one entry per quota pool with label, used_pct (0-100), remaining, limit, unit ("%", "credits", "requests"), reset_at and seconds_remaining (0 when the pool has no reset)
- detail: human note, including why no pool could be read

Typical shapes: antigravity and copilot report per-model-group percentage windows, codex reports 5-hour and weekly sliding windows, xai reports credit pools, freebuff reports a credits balance, zai reports the dominant coding-plan window. Lanes whose upstream exposes no quota endpoint answer "has no quota mechanism"; a lane that is not logged in (e.g. codex) answers with a detail telling you how to log in first.

Monitoring only: Ultiproxy NEVER reroutes or fails over on its own because a quota is exhausted - routing decisions belong to the operator/agent. Read this tool (or GET /api/quota and /quota.txt), choose the model explicitly, then send the request.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider lane name, e.g. antigravity, xai, copilot, codex, freebuff, zai (see list_providers)"},
			},
			Required: []string{"provider"},
		},
	},
	{
		Name: "toggle_model",
		Description: `Enable or disable a client-visible model id at runtime WITHOUT deleting its mapping.

- model_id: an existing alias id (see list_models / list_model_aliases)
- enabled: true makes the id routable again; false makes requests for it fail with unknown_model while the alias, its provider lane and all its metadata stay intact

Use it as a soft kill switch: a model that is misbehaving, an alias being drained before maintenance, temporarily hiding a lane's models from clients. Re-enabling is instant. The flag lives in memory only, so a restart re-enables every alias - remove_model_alias is the permanent operation.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"model_id": {Type: "string", Description: "Model id (alias) to toggle, exactly as listed by list_models"},
				"enabled":  {Type: "boolean", Description: "Enable (true) or disable (false) the model id; disabling keeps the alias mapping"},
			},
			Required: []string{"model_id", "enabled"},
		},
	},
	{
		Name: "get_client_usage",
		Description: `Read back the token and request accounting Ultiproxy records, for one client key or overall.

- client_id: the client key identity to report on; leave it empty for the overall / aggregate view
- window: lookback window such as "1h", "24h" or "7d"; leave it empty for the default window

The report covers prompt (input) tokens, completion (output) tokens, cached prompt tokens, total tokens, request counts and the estimated USD cost derived from each model's pricing (set_model_alias input_cost / output_cost). Streaming and non-streaming requests are accounted the same way. When the running daemon has no usage source attached the totals come back as zero - use GET /api/stats/summary for the SQL-backed aggregate in that case.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"client_id": {Type: "string", Description: "Client key name or key hash to report on; omit / empty for overall usage"},
				"window":    {Type: "string", Description: "Lookback window, e.g. 1h, 24h, 7d; omit for the default window"},
			},
		},
	},
	{
		Name: "initiate_oauth_login",
		Description: `Start an OAuth login against a subscription provider lane WITHOUT blocking the call.

provider is a lane with an interactive auth surface, e.g. "antigravity" (auth-code flow) or "xai" (device flow).

Returns immediately with:
- status "awaiting_user", the provider and the flow kind
- url: the sign-in page to open in a browser (hand it to the human or open it yourself)
- user_code: for device flows, the code the user must confirm on that page
- expires_in_seconds: how long the flow stays valid

Then finish it: for device flows poll check_oauth_login until it reports "completed"; for auth-code flows take the authorization code from the redirect/callback URL and pass it to submit_oauth_code. Lanes that only implement the legacy blocking flow run it inline and answer "initiated". Tokens are written to the daemon's credential store for that lane and are never returned to clients.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider lane to log in to, e.g. antigravity (auth-code flow) or xai (device flow)"},
			},
			Required: []string{"provider"},
		},
	},
	{
		Name: "check_oauth_login",
		Description: `Poll a pending OAuth login (device flows, e.g. xai) until the user approves it, or complete a flow whose token exchange can finish server-side.

provider: the lane whose login was started with initiate_oauth_login.

One call waits at most ~90 seconds and answers either:
- status "pending": nobody has approved yet - simply call again (device flows normally need a few polls while the human types the user_code)
- status "completed": tokens are stored, the lane is usable

Anything else is returned as an error. Auth-code flows whose token exchange happens server-side also finish here; flows that need the code copied out of the browser redirect must use submit_oauth_code instead.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider lane with a pending login, e.g. xai"},
			},
			Required: []string{"provider"},
		},
	},
	{
		Name: "submit_oauth_code",
		Description: `Finish an auth-code OAuth flow by handing over the authorization code the browser ended up with.

- provider: the lane whose login was started with initiate_oauth_login (e.g. antigravity)
- code: the authorization code from the redirect callback URL (http://localhost:<port>/oauth-callback?code=...&state=... - copy only the code value) or from the provider's success page

On success the code is exchanged for tokens server-side and the answer is status "completed". Codes are single-use and short-lived: if the exchange fails, start over with initiate_oauth_login. A browser running on the same machine as the daemon is captured automatically by the loopback listener, so this tool is the path for remote or headless browsers.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider lane finishing an auth-code flow, e.g. antigravity"},
				"code":     {Type: "string", Description: "Authorization code copied from the browser callback URL (the code= parameter)"},
			},
			Required: []string{"provider", "code"},
		},
	},
	{
		Name: "list_model_aliases",
		Description: `List the alias table: client-visible model name -> provider lane + upstream model id, with limits and pricing.

Returns a JSON object keyed by alias; every value carries provider, upstream, context_limit, max_output, pricing_tag, input_cost / output_cost (USD per 1M tokens) and benchmarks. This is the same data list_models reports per id, but complete and unfiltered. Every alias here is a valid "model" value for POST /v1/chat/completions and POST /v1/messages. Mutate the table with set_model_alias and remove_model_alias.`,
		InputSchema: &InputSchema{Type: "object", Properties: map[string]PropertyDef{}},
	},
	{
		Name: "set_model_alias",
		Description: `Create or update one model alias: a client-visible name mapped to a provider lane and an upstream model id. The mapping applies to new requests immediately and persists across restarts (data_dir/aliases.json).

- alias (required): the name clients will send as "model", e.g. "qwenpoint-3.8" or "sonnet-big"
- provider (required): a registered lane name, e.g. "vllm", "zai", "antigravity", "xai", "copilot", "codex", "freebuff" (see list_providers)
- upstream (required): the model id that lane understands, e.g. "Qwen/Qwen3.8-Instruct-AWQ" or "claude-sonnet-4-5"
- context_limit (optional): context window in tokens - advisory metadata surfaced as context_length on GET /v1/models, not enforced
- max_output (optional): hard cap on completion tokens; a request's max_tokens is clamped to it
- pricing_tag (optional): pricing label such as "flat-subscription" or "paid-api"
- input_cost / output_cost (optional): USD per 1M prompt / completion tokens, used for cost accounting when the upstream does not price itself
- benchmarks (optional): map of benchmark name -> score, informational

Calling it again with the same alias replaces the whole entry. Clients ask for the alias alone - no lane prefix needed. Example: {"alias":"qwenpoint-3.8","provider":"vllm","upstream":"Qwen/Qwen3.8-Instruct-AWQ","context_limit":131072,"max_output":8192,"pricing_tag":"local-gpu"}.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"alias":         {Type: "string", Description: "Client-visible model name clients send as model, e.g. qwenpoint-3.8"},
				"provider":      {Type: "string", Description: "Provider lane name the alias routes to, e.g. vllm, zai, antigravity, xai"},
				"upstream":      {Type: "string", Description: "Upstream model id on that lane, e.g. Qwen/Qwen3.8-Instruct-AWQ"},
				"context_limit": {Type: "number", Description: "Optional context window size in tokens (advisory, surfaced as context_length on /v1/models)"},
				"max_output":    {Type: "number", Description: "Optional max output tokens; a request's max_tokens is clamped to it"},
				"pricing_tag":   {Type: "string", Description: "Optional pricing label, e.g. flat-subscription"},
				"input_cost":    {Type: "number", Description: "Optional input price in US dollars per 1M prompt tokens (drives cost accounting)"},
				"output_cost":   {Type: "number", Description: "Optional output price in US dollars per 1M completion tokens (drives cost accounting)"},
				"benchmarks":    {Type: "object", Description: "Optional map of benchmark name -> score, informational only"},
			},
			Required: []string{"alias", "provider", "upstream"},
		},
	},
	{
		Name: "remove_model_alias",
		Description: `Delete one alias mapping by name.

The client-visible id disappears from list_models and GET /v1/models immediately and stops routing (requests for it fail with unknown_model). Deletion is permanent across restarts because data_dir/aliases.json is rewritten. The provider lane itself is untouched, and re-creating the mapping is just another set_model_alias call. Use toggle_model instead when the mapping should survive but be temporarily unroutable.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"alias": {Type: "string", Description: "Alias to remove, exactly as listed by list_model_aliases"},
			},
			Required: []string{"alias"},
		},
	},
	{
		Name: "get_provider_timeouts",
		Description: `List the per-provider request timeouts currently in force, as a JSON map of lane name -> Go duration string (e.g. {"vllm":"10m"}).

Lanes absent from the map use the server default of 120s. A timeout bounds one upstream request, streaming included, so slow lanes (large local models, long reasoning chains) need a bigger value or long generations get cut off. Change an entry with set_provider_timeout and drop it back to the default with remove_provider_timeout.`,
		InputSchema: &InputSchema{Type: "object", Properties: map[string]PropertyDef{}},
	},
	{
		Name: "set_provider_timeout",
		Description: `Configure the request timeout for one provider lane. It applies to new requests immediately and persists across restarts (data_dir/timeouts.json).

- provider: a registered lane name, e.g. "vllm", "freebuff", "zai"
- timeout: a Go duration string such as "10m", "3m30s", "45s" or "1h"

Raise it to give slow lanes headroom (local vLLM models, long agentic generations), lower it to fail fast on a lane that hangs. Anything above the 120s default must be set explicitly. Invalid or non-positive durations are rejected. Reset to the default with remove_provider_timeout. Example: {"provider":"vllm","timeout":"10m"}.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider lane name, e.g. vllm, freebuff"},
				"timeout":  {Type: "string", Description: "Go duration string, e.g. 3m30s, 10m, 1h"},
			},
			Required: []string{"provider", "timeout"},
		},
	},
	{
		Name: "remove_provider_timeout",
		Description: `Reset a provider lane's timeout to the server default (120s) by dropping its explicit override.

provider: the lane whose override should be removed. The change is immediate and persisted (data_dir/timeouts.json). Removing a timeout that was never set is not an error.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"provider": {Type: "string", Description: "Provider lane name to reset to the default timeout"},
			},
			Required: []string{"provider"},
		},
	},
	{
		Name: "add_provider",
		Description: `Register a new upstream provider lane at runtime: no restart, nothing to hand-edit. The lane is built, validated and added to the live registry immediately, and persisted to data_dir/providers.json so it comes back on restart.

Parameters:
- name (required): lane name, lowercase [a-z0-9_-], e.g. "vllm", "zai", "deepseek", "openrouter". Its models are reachable as "<name>/<upstream_model>" and through aliases (set_model_alias).
- kind: "" or "openaicompat" (default) for any OpenAI-compatible upstream; "antigravity", "anthropic", "codex" or "freebuff" for the custom-wire lanes.
- base_url: upstream base URL, required for OpenAI-compatible lanes, e.g. "https://api.deepseek.com/v1". Custom-wire lanes ignore it - they know their endpoints.
- api_key: static key. Required for kind=anthropic; optional elsewhere (public or local lanes work without one).
- the reply reports how many upstream models the new lane serves ("discovered N models"); discovery runs automatically, so no follow-up call is needed
- quirks (object, OpenAI-compatible lanes only): vendor behaviour switches
  - coding_plan_path: route through a coding-plan path (zai-style coding subscriptions)
  - max_tokens_by_model: map of upstream model id -> max_tokens that upstream accepts, e.g. {"glm-4.6":8192}
  - echo_reasoning: repeat reasoning tokens as visible content (upstreams that hide them)
  - model_list_passthrough: model discovery (GET <base>/v1/models so "<name>/<model>" ids show up on /v1/models). ON by default for OpenAI-compatible lanes; set it to false to opt out. Discovery runs when the lane is registered, again at daemon startup when a lane cache is still empty, and every 6h afterwards
  - auth_via_oauth_manager: authenticate with the lane's stored OAuth credential (xai-style) instead of a static key
  - credits_quota_observer: upstream credits endpoint to poll so get_quota_status has something to report (e.g. the xai billing URL)
  - auth_via_supabase_refresh: refresh credentials through a Supabase session
  - freebuff_actor: mark the lane as a Codebuff/freebuff lane (serialized requests, session affinity, actor-backed quota); the real actor is rebuilt from the lane's api_key / stored state token
  - freebuff_default_tool: force the default tool behaviour on freebuff chats
  - default_model: upstream model id to use when a request carries no explicit model

Examples: a local vLLM lane {"name":"vllm","base_url":"http://127.0.0.1:8000/v1","quirks":{"model_list_passthrough":true}} (then set_model_alias {"alias":"qwen-coder","provider":"vllm","upstream":"Qwen/Qwen2.5-Coder-32B-Instruct"}); a keyed API lane {"name":"deepseek","base_url":"https://api.deepseek.com/v1","api_key":"sk-..."}; custom lanes {"name":"my-claude","kind":"anthropic","api_key":"sk-ant-..."}, {"name":"antigravity","kind":"antigravity"} followed by initiate_oauth_login, {"name":"codex","kind":"codex"}, {"name":"freebuff","kind":"freebuff"}.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"name":     {Type: "string", Description: "Lane name, lowercase [a-z0-9_-], e.g. vllm, zai, openrouter; models route as <name>/<model>"},
				"kind":     {Type: "string", Description: "Lane kind: empty or openaicompat for OpenAI-compatible lanes, antigravity, anthropic (requires api_key), codex or freebuff (Codebuff account lane; api_key optional, falls back to the ultiproxy-owned state token)"},
				"base_url": {Type: "string", Description: "Upstream base URL, e.g. https://api.deepseek.com/v1 (required for OpenAI-compatible lanes)"},
				"api_key":  {Type: "string", Description: "Static API key (required for kind=anthropic, optional otherwise)"},
				"quirks": {
					Type: "object",
					Description: `Vendor quirks, OpenAI-compatible lanes only: {coding_plan_path, max_tokens_by_model, echo_reasoning, model_list_passthrough, auth_via_oauth_manager, credits_quota_observer, auth_via_supabase_refresh, freebuff_actor, freebuff_default_tool, default_model}.

coding_plan_path (bool): route through a coding-plan path (zai-style coding subscriptions). max_tokens_by_model (object): upstream model id -> max_tokens cap the upstream accepts. echo_reasoning (bool): repeat reasoning tokens as visible content. model_list_passthrough (bool): discover GET <base>/v1/models so <lane>/<model> ids appear on /v1/models. auth_via_oauth_manager (bool): use the lane's stored OAuth credential instead of a static key. credits_quota_observer (string): upstream credits endpoint polled for quota. auth_via_supabase_refresh (bool): refresh credentials through a Supabase session. freebuff_actor (bool): mark a Codebuff/freebuff lane - serialized requests, session affinity, actor-backed quota rebuilt from the lane's api_key / state token. freebuff_default_tool (bool): force the default tool behaviour on freebuff chats. default_model (string): upstream model id used when a request carries no model.`,
				},
			},
			// base_url is validated per kind by the tool itself: custom kinds
			// (antigravity, anthropic, codex) do not need one.
			Required: []string{"name"},
		},
	},
	{
		Name: "remove_provider",
		Description: `Unregister a provider lane: it is dropped from the live registry immediately (no request routes to it any more) and deleted from data_dir/providers.json, so it does not come back on restart.

Aliases pointing at the lane are left in place but stop resolving - repoint them with set_model_alias or delete them with remove_model_alias. Compile-time lanes (registered from env/OAuth credentials rather than add_provider) can only be removed from the live registry; the result then reports persisted=false. name must match the lane exactly (see list_providers).`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"name": {Type: "string", Description: "Lane name to remove, e.g. vllm (see list_providers)"},
			},
			Required: []string{"name"},
		},
	},
	{
		Name: "list_providers",
		Description: `List every provider lane the proxy knows about, in two groups:

- providers[]: lanes registered at runtime through add_provider, each with name, base_url, has_api_key (a boolean - the key itself is never returned), auth_via_oauth and the full quirks set
- registry_lanes[]: every lane in the live registry, including compiled/in-memory lanes that live only in the process (env-credential or OAuth-credential lanes)

Secrets are always redacted. Use this to discover valid provider names before calling add_provider / set_model_alias / get_quota_status / set_provider_timeout, and to confirm a lane you just added is actually live. Per-lane health and quota live in get_quota_status.`,
		InputSchema: &InputSchema{Type: "object", Properties: map[string]PropertyDef{}},
	},
	{
		Name: "refresh_models",
		Description: `Re-fetch and cache a lane's upstream model list (GET <base_url>/v1/models) so those ids show up as "<lane>/<model>" on GET /v1/models and can be routed directly or aliased.

name: the lane to refresh, e.g. "opencode", "vllm", "deepseek".

Discovery is automatic: every OpenAI-compatible lane discovers when it is registered (add_provider reports the count), lanes whose cache is still empty are re-discovered at daemon startup, and every discovery lane is refreshed every 6h (model_list_passthrough:false opts a lane out of all of it). This tool is the manual override on top of that schedule (10s budget): use it to pick up an upstream change right now instead of waiting for the next tick. It answers "N models cached for lane <name>". Lanes without model discovery (custom-wire lanes such as antigravity, codex, anthropic) answer "does not support model discovery", though prefix routing "<lane>/<model>" still works for them.`,
		InputSchema: &InputSchema{
			Type: "object",
			Properties: map[string]PropertyDef{
				"name": {Type: "string", Description: "Provider lane name to re-discover models for, e.g. opencode, vllm"},
			},
			Required: []string{"name"},
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
	case "refresh_models":
		return s.toolRefreshModels(ctx, params.Arguments)
	default:
		return nil, &JSONRPCError{
			Code:    CodeMethodNotFound,
			Message: fmt.Sprintf("unknown tool: %q", params.Name),
		}
	}
}

func (s *Server) toolListModels(ctx context.Context) (*CallToolResult, *JSONRPCError) {
	b, _ := json.MarshalIndent(s.listedModels(), "", "  ")
	return &CallToolResult{
		Content: []ToolContent{
			{
				Type: "text",
				Text: string(b),
			},
		},
	}, nil
}

// listedModel is one list_models entry. The id set is the same set GET
// /v1/models advertises; the value carries the metadata an agent needs to pick
// a model or flip its enabled state.
type listedModel struct {
	ID              string             `json:"id"`
	Provider        string             `json:"provider"`
	Enabled         bool               `json:"enabled"`
	ContextLimit    int                `json:"context_limit,omitempty"`
	MaxOutput       int                `json:"max_output,omitempty"`
	PricingTag      string             `json:"pricing_tag,omitempty"`
	BenchmarkScores map[string]float64 `json:"benchmarks,omitempty"`
	// Source is where the id comes from: "alias" (catalog / state model map),
	// "discovery" (cached upstream catalog) or "default" (the lane's default
	// model).
	Source string `json:"source"`
}

// modelsCacheProvider, defaultModelProvider, EnvHideTestLanes, hideTestLanes
// and isTestLane mirror the pkg/server surfaces (mcp cannot import server
// without an import cycle). Keep them in sync: list_models must apply exactly
// the filtering GET /v1/models applies, and the tests on both sides assert
// that agreement.
type modelsCacheProvider interface {
	CachedModels() []string
}

type defaultModelProvider interface {
	DefaultModel() string
}

const EnvHideTestLanes = "ULTIPROXY_HIDE_TEST_LANES"

func hideTestLanes() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(EnvHideTestLanes))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func isTestLane(name string) bool {
	switch name {
	case "probe", "fake", "mock", "test":
		return true
	}
	for _, prefix := range []string{"probe-", "fake-", "mock-", "test-"} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// listedModels builds the id -> entry map list_models returns, mirroring the
// GET /v1/models filtering: aliases plus "<lane>/<model>" ids (discovered or
// lane default), never a bare lane name, and never a test lane when
// ULTIPROXY_HIDE_TEST_LANES is set. One deliberate difference: ids disabled at
// runtime (toggle_model) stay listed with "enabled": false so an agent can see
// them and switch them back on; /v1/models omits them entirely.
func (s *Server) listedModels() map[string]listedModel {
	out := make(map[string]listedModel)
	set := func(id string, e listedModel) {
		if id == "" {
			return
		}
		if _, exists := out[id]; exists {
			return
		}
		e.ID = id
		out[id] = e
	}

	var snap *state.RuntimeSnapshot
	if s.stateSource != nil {
		snap = s.stateSource.Snapshot()
	}

	// 1. State snapshot model map (aliases synced from the catalog, plus any
	//    runtime toggle_model entries), Enabled reported as-is.
	if snap != nil && snap.Models != nil {
		keys := make([]string, 0, len(snap.Models))
		for k := range snap.Models {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			m := snap.Models[k]
			set(m.ID, listedModel{
				Provider:        m.Provider,
				Enabled:         m.Enabled,
				ContextLimit:    m.ContextLimit,
				MaxOutput:       m.MaxOutput,
				PricingTag:      m.PricingTag,
				BenchmarkScores: m.BenchmarkScores,
				Source:          "alias",
			})
		}
	}

	// 2. Alias catalog (covers servers built without a state source).
	if s.aliases != nil {
		entries := s.aliases.List()
		for _, alias := range s.aliases.Sorted() {
			entry, ok := entries[alias]
			if !ok {
				continue
			}
			enabled := true
			if snap != nil && snap.Models != nil {
				if mr, ok := snap.Models[alias]; ok {
					enabled = mr.Enabled
				}
			}
			set(alias, listedModel{
				Provider:        entry.Provider,
				Enabled:         enabled,
				ContextLimit:    entry.ContextLimit,
				MaxOutput:       entry.MaxOutput,
				PricingTag:      entry.PricingTag,
				BenchmarkScores: entry.BenchmarkScores,
				Source:          "alias",
			})
		}
	}

	// 3. Registered lanes: "<lane>/<model>" ids only, read from the discovery
	//    cache (never a live fetch) plus the lane default model.
	if s.registry != nil {
		hideTest := hideTestLanes()
		for _, name := range s.registry.Names() {
			if hideTest && isTestLane(name) {
				continue
			}
			bundle, ok := s.registry.Get(name)
			if !ok || bundle.Inference == nil {
				continue
			}
			if cacher, ok := bundle.Inference.(modelsCacheProvider); ok {
				discovered := append([]string(nil), cacher.CachedModels()...)
				sort.Strings(discovered)
				for _, m := range discovered {
					set(name+"/"+m, listedModel{Provider: name, Enabled: true, Source: "discovery"})
				}
			}
			if def, ok := bundle.Inference.(defaultModelProvider); ok {
				if m := def.DefaultModel(); m != "" {
					set(name+"/"+m, listedModel{Provider: name, Enabled: true, Source: "default"})
				}
			}
		}
	}

	return out
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
		if ok {
			if prov.Quota == nil {
				// Lane is registered but implements no quota mechanism (plain
				// openaicompat lanes): say so instead of the generic
				// "not found" message, which reads as a routing bug.
				return &CallToolResult{
					Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("provider %q has no quota mechanism (lane registered, upstream exposes no quota/usage endpoint)", args.Provider)}},
					IsError: true,
				}, nil
			}
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
		InputCost    float64            `json:"input_cost"`
		OutputCost   float64            `json:"output_cost"`
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
		InputCost:       args.InputCost,
		OutputCost:      args.OutputCost,
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

// modelsFetcher is the optional lane capability that pulls the upstream model
// list and caches it (OpenAI-compatible lanes implement it via
// GET <base>/models). It is asserted off bundle.Inference so the MCP layer
// stays independent of the concrete adapter packages.
type modelsFetcher interface {
	FetchModels(ctx context.Context) ([]string, error)
}

// toolRefreshModels backfills the model cache of a running lane. Lanes
// registered before startup model discovery existed have an empty cache, so
// the aggregated /v1/models handler only lists the bare "<lane>" id. This tool
// re-runs discovery on demand instead of forcing the lane to be re-added.
func (s *Server) toolRefreshModels(ctx context.Context, argsRaw json.RawMessage) (*CallToolResult, *JSONRPCError) {
	var args struct {
		Name string `json:"name"`
	}
	_ = json.Unmarshal(argsRaw, &args)
	if args.Name == "" {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "error: name argument is required"}},
			IsError: true,
		}, nil
	}

	notFound := func() *CallToolResult {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error: lane %s not found", args.Name)}},
			IsError: true,
		}
	}
	if s.registry == nil {
		return notFound(), nil
	}
	bundle, ok := s.registry.Get(args.Name)
	if !ok {
		return notFound(), nil
	}

	// A lane with no inference surface has nothing to discover models for.
	fetcher, ok := bundle.Inference.(modelsFetcher)
	if !ok {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error: lane %s does not support model discovery", args.Name)}},
			IsError: true,
		}, nil
	}

	fetchCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	models, err := fetcher.FetchModels(fetchCtx)
	if err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("error: fetching models for lane %s: %v", args.Name, err)}},
			IsError: true,
		}, nil
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "%d models cached for lane %s\n", len(models), args.Name)
	for _, m := range models {
		sb.WriteString(m)
		sb.WriteByte('\n')
	}
	return &CallToolResult{
		Content: []ToolContent{{Type: "text", Text: sb.String()}},
	}, nil
}
