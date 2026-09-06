package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/smhanov/llmhub"
	llmauth "github.com/smhanov/llmhub/auth"
	hubopenai "github.com/smhanov/llmhub/providers/openai"
	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/modelmeta"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/hublane"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/oauth"
)

// xaiPendingFlow carries the in-flight xAI device authorization between the
// two-phase MCP login calls.
type xaiPendingFlow struct {
	cfg        oauth.DeviceFlowConfig
	deviceCode string
	interval   int
}

// modelDiscoveryBudget bounds the construction-time model discovery call (and
// the on-demand Models() fallback). It matches the budget the server's
// registration/startup/scheduled discovery passes use.
const modelDiscoveryBudget = 5 * time.Second

// ModelListPassthroughEnabled resolves whether a lane config participates in
// upstream model discovery. Discovery is the DEFAULT for OpenAI-compatible
// lanes - a single GET <base>/models, cached afterwards - so an upstream that
// has no such endpoint simply caches nothing. Two things opt out:
//   - OptOutModelListPassthrough (MCP quirks.model_list_passthrough:false, or
//     the same field persisted in providers.json);
//   - freebuff lanes: their wire is not an OpenAI model list, and their model
//     set is subscription knowledge, so nothing is discovered or invented.
func ModelListPassthroughEnabled(cfg Config) bool {
	return !cfg.OptOutModelListPassthrough &&
		cfg.Quirks.FreebuffActor == nil &&
		!cfg.Quirks.FreebuffDefaultTool
}

// Provider implements provider.InferenceProvider and optionally provider.QuotaProvider and provider.AuthProvider.
type Provider struct {
	adapter    *hublane.Adapter
	cfg        Config
	httpClient *http.Client
	baseURL    string
	apiKey     string
	name       string

	mu        sync.RWMutex
	models    []string
	modelInfo []provider.ModelInfo
	// pendingXAI holds the in-flight device authorization for a two-phase
	// MCP-driven login. Guarded by mu.
	pendingXAI *xaiPendingFlow
}

// New creates a new OpenAI-compatible provider.
func New(cfg Config) (*Provider, error) {
	name := cfg.Name
	if name == "" {
		name = "openaicompat"
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
		cfg.HTTPClient = client
	}

	// Quirks: CodingPlanPath default baseURL
	if cfg.Quirks.CodingPlanPath && cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.z.ai/api/coding/paas/v4"
	} else if strings.Contains(cfg.BaseURL, "coding") {
		cfg.Quirks.CodingPlanPath = true
	}

	// Quirks: AuthViaSupabaseRefresh TokenSource initialization
	if cfg.Quirks.AuthViaSupabaseRefresh && cfg.TokenSource == nil {
		refreshURL := cfg.RefreshURL
		if refreshURL == "" {
			if strings.Contains(cfg.BaseURL, "127.0.0.1") || strings.Contains(cfg.BaseURL, "localhost") {
				refreshURL = strings.TrimSuffix(cfg.BaseURL, "/") + "/auth/v1/token?grant_type=refresh_token"
			} else {
				refreshURL = defaultAugureRefreshURL
			}
		}
		tokenFile := cfg.TokenFile
		if tokenFile == "" && cfg.DataDir != "" {
			tokenFile = filepath.Join(cfg.DataDir, "augure-auth.json")
		}
		cfg.TokenSource = NewSupabaseTokenSource(client, refreshURL, tokenFile, cfg.APIKey, "")
	}

	// Quirks: AuthViaOAuthManager TokenSource initialization
	if cfg.Quirks.AuthViaOAuthManager && cfg.TokenSource == nil {
		dataDir := cfg.DataDir
		if dataDir == "" {
			dataDir = filepath.Join(os.TempDir(), "ultiproxy-xai-auth")
		}
		refresher := oauth.MakeRefresher(client, "https://auth.x.ai/oauth2/token", defaultXAIClientID, "")
		mgr, err := auth.NewManager(dataDir, refresher)
		if err == nil {
			cfg.TokenSource = NewOAuthManagerTokenSource(mgr, defaultXAIClientID)
		}
	}

	// Model discovery is on by default for OpenAI-compatible lanes (see
	// ModelListPassthroughEnabled) and opted out with
	// quirks.model_list_passthrough:false. Resolving the effective flag BEFORE
	// the provider struct is built means p.cfg, Models(), resolveModel(), the
	// runtime provider store and the MCP surface all agree on the same value
	// for the life of the lane.
	cfg.Quirks.ModelListPassthrough = ModelListPassthroughEnabled(cfg)

	hubKey := cfg.APIKey
	if hubKey == "" {
		hubKey = "openaicompat"
	}

	hubOpts := []llmhub.Option{llmhub.WithHTTPClient(client), llmhub.WithRetryOnStatus(http.StatusTooManyRequests, false)}
	// Freebuff: the upstream signals free-capacity pressure with
	// 428 waiting_room_required; the official CLI queues and retries
	// (honoring Retry-After). Opt the lane's HTTP layer into 428 retries.
	if cfg.Quirks.FreebuffDefaultTool {
		hubOpts = append(hubOpts, llmhub.WithRetryOnStatus(freebuffStatusWaitingRoom, true))
	}
	if cfg.BaseURL != "" {
		hubOpts = append(hubOpts, llmhub.WithBaseURL(cfg.BaseURL))
	}
	if cfg.TokenSource != nil {
		// The lane's token source is handed to llmhub WITHOUT its Invalidate
		// method. llmhub's own OpenAI client retries a 401 by itself when the
		// source implements auth.InvalidatableTokenSource, which would stack a
		// second retry on top of the one this package owns below: a rejected
		// credential would then cost up to four upstream requests and a
		// persistent failure would be retried twice. Ultiproxy keeps the whole
		// refresh-and-retry policy in this package (ONE retry, covering 401 AND
		// 403, and never after the first byte of a stream), so the narrowing
		// adapter below is deliberately NOT an invalidator.
		hubOpts = append(hubOpts, llmhub.WithTokenSource(hubTokenSource{inner: cfg.TokenSource}))
	}

	// ALWAYS build the underlying llmhub provider using the openai client, NEVER vendor providers
	hubProv, err := hubopenai.New(hubKey, hubOpts...)
	if err != nil {
		return nil, err
	}

	p := &Provider{
		cfg:        cfg,
		httpClient: client,
		baseURL:    strings.TrimSuffix(cfg.BaseURL, "/"),
		apiKey:     cfg.APIKey,
		name:       name,
	}

	// Model discovery for this lane. The flag was already resolved above, so
	// p.cfg carries the same decision Models(), resolveModel() and the
	// server-side discovery loop see.
	if cfg.Quirks.ModelListPassthrough {
		ctx, cancel := context.WithTimeout(context.Background(), modelDiscoveryBudget)
		defer cancel()
		_, _ = p.FetchModels(ctx)
	}

	caps := p.Capabilities()
	hublaneOpts := []hublane.AdapterOption{
		hublane.WithCapabilities(caps),
	}
	if cfg.Quirks.CreditsQuotaObserver != "" || cfg.Quirks.FreebuffActor != nil {
		hublaneOpts = append(hublaneOpts, hublane.WithQuotaProvider(p))
	}
	if cfg.Quirks.AuthViaOAuthManager || cfg.Quirks.AuthViaSupabaseRefresh {
		hublaneOpts = append(hublaneOpts, hublane.WithAuthProvider(p))
	}

	p.adapter = hublane.New(hubProv, hublaneOpts...)
	return p, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return p.name
}

// Capabilities returns the provider's advertised capabilities.
func (p *Provider) Capabilities() provider.Capabilities {
	caps := provider.Capabilities{
		Chat:          true,
		Messages:      true,
		Reasoning:     true,
		Vision:        true,
		Tools:         true,
		Streaming:     true,
		PromptCaching: false,
	}
	if p.cfg.Quirks.FreebuffActor != nil {
		caps.MaxConcurrentRequests = 1
		caps.SessionAffinity = true
		caps.Queueing = true
	}
	return caps
}

// ProviderBundle returns a provider.Provider bundle including any optional quota/auth.
func (p *Provider) ProviderBundle() provider.Provider {
	bundle := provider.Provider{
		Inference:    p,
		Capabilities: p.Capabilities(),
	}
	if p.cfg.Quirks.CreditsQuotaObserver != "" || p.cfg.Quirks.FreebuffActor != nil {
		bundle.Quota = p
	}
	if p.cfg.Quirks.AuthViaOAuthManager || p.cfg.Quirks.AuthViaSupabaseRefresh {
		bundle.Auth = p
	}
	return bundle
}

// Provider returns the provider.Provider bundle.
func (p *Provider) Provider() provider.Provider {
	return p.ProviderBundle()
}

func (p *Provider) freebuffActor() FreebuffActor {
	if p.cfg.Quirks.FreebuffActor == nil {
		return nil
	}
	if a, ok := p.cfg.Quirks.FreebuffActor.(FreebuffActor); ok {
		return a
	}
	return nil
}

func (p *Provider) resolveModel(requested string) string {
	if requested != "" {
		return requested
	}
	if p.cfg.Quirks.DefaultModel != "" {
		return p.cfg.Quirks.DefaultModel
	}
	if p.cfg.Quirks.ModelListPassthrough {
		p.mu.RLock()
		defer p.mu.RUnlock()
		if len(p.models) > 0 {
			return p.models[0]
		}
		return "default"
	}
	return ""
}

// resolveMaxTokens returns the max_tokens value to send upstream.
//
// max_tokens_by_model is a CEILING the upstream accepts, not a default: a
// client value above the matched limit is clamped down to it, a client value
// at or below the limit passes through untouched, and an omitted value falls
// back to the limit itself.
//
// Pattern matching is deterministic and never depends on Go map iteration
// order:
//
//  1. an exact model id match wins;
//  2. otherwise the longest pattern the model id contains wins (so "glm-5.3"
//     beats "glm" for "glm-5.3-flash");
//  3. equal-length patterns are broken lexicographically (ascending).
func (p *Provider) resolveMaxTokens(model string, requested int) int {
	if ceiling := p.matchMaxTokensPattern(model); ceiling > 0 {
		return clampMaxTokens(requested, ceiling)
	}
	if p.cfg.Quirks.CodingPlanPath {
		// Legacy zai coding-plan defaults when no pattern table is set.
		if len(p.cfg.Quirks.MaxTokensByModel) == 0 && strings.Contains(model, "4.5-air") {
			return clampMaxTokens(requested, 98304)
		}
		return clampMaxTokens(requested, 131072)
	}
	return requested
}

// clampMaxTokens applies ceiling as an upper bound: values above it (or unset)
// become the ceiling, values below it pass through.
func clampMaxTokens(requested, ceiling int) int {
	if requested > ceiling {
		return ceiling
	}
	if requested <= 0 {
		return ceiling
	}
	return requested
}

// matchMaxTokensPattern returns the max_tokens limit configured for model, or
// 0 when no pattern matches. Candidates are sorted, never read in map order.
func (p *Provider) matchMaxTokensPattern(model string) int {
	if len(p.cfg.Quirks.MaxTokensByModel) == 0 || model == "" {
		return 0
	}
	// Exact match first: nothing can outrank the model's own id.
	if limit, ok := p.cfg.Quirks.MaxTokensByModel[model]; ok {
		return limit
	}
	patterns := make([]string, 0, len(p.cfg.Quirks.MaxTokensByModel))
	for pattern := range p.cfg.Quirks.MaxTokensByModel {
		if pattern != "" && strings.Contains(model, pattern) {
			patterns = append(patterns, pattern)
		}
	}
	if len(patterns) == 0 {
		return 0
	}
	sort.Slice(patterns, func(i, j int) bool {
		if len(patterns[i]) != len(patterns[j]) {
			return len(patterns[i]) > len(patterns[j]) // longest first
		}
		return patterns[i] < patterns[j] // then lexicographic
	})
	return p.cfg.Quirks.MaxTokensByModel[patterns[0]]
}

// freebuffStatusWaitingRoom is the HTTP status upstream returns when the
// free tier is at capacity (428 Precondition Required upstream-side).
const freebuffStatusWaitingRoom = 428

// freebuffClientSessionID mirrors the shipped CLI's clientSessionId:
// Math.random().toString(36).substring(2,15) — up to 11 lowercase base36
// chars, no prefix.
func freebuffClientSessionID() string {
	const charset = "0123456789abcdefghijklmnopqrstuvwxyz"
	n := 11
	b := make([]byte, n)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
	}
	return string(b)
}

// freebuffCanonicalModel maps a request model (any alias form) to the
// canonical publisher-qualified upstream model id used for session binding
// (x-freebuff-model). Unknown models pass through unchanged.
func freebuffCanonicalModel(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	for canonical, aliases := range freebuffCanonicalByAlias {
		for _, a := range aliases {
			if normalized == a {
				return canonical
			}
		}
	}
	return model
}

// freebuffCanonicalByAlias: canonical publisher model -> accepted aliases
// (lowercased). Source: freebuff modelMap + live session rateLimitsByModel.
var freebuffCanonicalByAlias = map[string][]string{
	"z-ai/glm-5.3-flash":         {"z-ai/glm-5.3-flash", "glm-5.3-flash", "glm"},
	"deepseek/deepseek-v4-flash": {"deepseek/deepseek-v4-flash", "deepseek-v4-flash", "deepseek"},
	"openai/gpt-5.6-luna":        {"openai/gpt-5.6-luna", "gpt-5.6-luna", "luna"},
	"minimax/minimax-m3":         {"minimax/minimax-m3", "minimax-m3", "minimax"},
	"mimo/mimo-v2.5":             {"mimo/mimo-v2.5", "mimo-v2.5", "mimo"},
	"upstage/solar-pro4":         {"upstage/solar-pro4", "solar-pro-4", "solar"},
	"crof/kimi-k3-eco":           {"crof/kimi-k3-eco", "kimi-k3-eco", "kimi"},
}

// freebuffAgentID translates a request model into the free-mode agentId the
// /agent-runs endpoint expects (mirrors the old node bridge's modelMap.json,
// ~/workspace/freebuff-proxy/src/modelMap.json). Unknown models pass through
// unchanged (fail-open): upstream, not the proxy, owns the canonical list.
func freebuffAgentID(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	if normalized == "" {
		return model
	}
	if strings.HasPrefix(normalized, "base3-free-") {
		return model // already an agentId
	}
	if id, ok := freebuffAgentByModel[normalized]; ok {
		return id
	}
	// Also accept "<publisher>/<model>" for a known bare model and vice versa.
	if _, rest, found := strings.Cut(normalized, "/"); found {
		if id, ok := freebuffAgentByModel[rest]; ok {
			return id
		}
	}
	return model
}

// freebuffAgentByModel maps free-tier model ids (bare and publisher-qualified
// forms) to their base3-free agent ids. Source of truth: the free session's
// rateLimitsByModel on a live account + modelMap.json from the Sept 3 bridge.
var freebuffAgentByModel = map[string]string{
	"deepseek/deepseek-v4-flash": "base3-free-deepseek-flash",
	"deepseek-v4-flash":          "base3-free-deepseek-flash",
	"deepseek":                   "base3-free-deepseek-flash",
	"z-ai/glm-5.3-flash":         "base3-free-glm-5-3-flash",
	"glm-5.3-flash":              "base3-free-glm-5-3-flash",
	"glm":                        "base3-free-glm-5-3-flash",
	"openai/gpt-5.6-luna":        "base3-free-luna",
	"gpt-5.6-luna":               "base3-free-luna",
	"luna":                       "base3-free-luna",
	"minimax/minimax-m3":         "base3-free-minimax-m3",
	"minimax-m3":                 "base3-free-minimax-m3",
	"minimax":                    "base3-free-minimax-m3",
	"mimo/mimo-v2.5":             "base3-free-mimo",
	"mimo-v2.5":                  "base3-free-mimo",
	"mimo":                       "base3-free-mimo",
	"upstage/solar-pro4":         "base3-free-solar-pro4",
	"solar-pro-4":                "base3-free-solar-pro4",
	"solar":                      "base3-free-solar-pro4",
	"crof/kimi-k3-eco":           "base3-free-kimi-k3-eco",
	"kimi-k3-eco":                "base3-free-kimi-k3-eco",
	"kimi":                       "base3-free-kimi-k3-eco",
	"google/gemini-3.8-flash":    "base3-free-gemini-3-8-flash",
	"gemini-3.8-flash":           "base3-free-gemini-3-8-flash",
}

func (p *Provider) applyRequestTransforms(ctx context.Context, msgs []*ir.Message, opts []provider.Option) (context.Context, []*ir.Message, []provider.Option, error) {
	reqConfig := provider.NewRequestConfig(opts...)

	// 1. Model resolution
	model := p.resolveModel(reqConfig.Model)
	if model != "" {
		// Freebuff: normalize aliases to the canonical publisher-qualified id
		// so the body model, session bind, and agent run all agree.
		if p.cfg.Quirks.FreebuffDefaultTool {
			model = freebuffCanonicalModel(model)
		}
		opts = append(opts, provider.WithModel(model))
	}

	// 2. MaxTokens resolution (CodingPlanPath / MaxTokensByModel)
	maxTokens := p.resolveMaxTokens(model, reqConfig.MaxTokens)
	if maxTokens > 0 {
		opts = append(opts, provider.WithMaxTokens(maxTokens))
	}

	// 3. Reasoning sanitization (EchoReasoning)
	msgs = sanitizeReasoning(msgs, p.cfg.Quirks.EchoReasoning)

	// 4+5. Freebuff session lifecycle BEFORE header construction, then the
	// freebuff headers and default tool/codebuff_metadata transform.
	if p.cfg.Quirks.FreebuffActor != nil {
		// Lifecycle first: reconcile, bind to the canonical publisher model
		// when unbound (fresh account), delete+re-bind on model switch. Bind
		// may adopt an upstream-minted instance id, so this must run before
		// the x-freebuff-instance-id header is built.
		if p.cfg.Quirks.FreebuffDefaultTool {
			if sm, ok := p.cfg.Quirks.FreebuffActor.(interface {
				Reconcile(...context.Context) error
				BoundModel() string
				DeleteSession(...context.Context) error
				Bind(any, ...string) error
			}); ok {
				canonical := freebuffCanonicalModel(model)
				if err := sm.Reconcile(ctx); err != nil {
					return nil, nil, nil, fmt.Errorf("freebuff session check: %w", err)
				}
				switch bound := sm.BoundModel(); {
				case bound == "":
					if err := sm.Bind(ctx, canonical); err != nil {
						return nil, nil, nil, err
					}
				case bound != canonical:
					if err := sm.DeleteSession(ctx); err != nil {
						return nil, nil, nil, fmt.Errorf("freebuff session switch: %w", err)
					}
					if err := sm.Bind(ctx, canonical); err != nil {
						return nil, nil, nil, err
					}
				}
			}
		}
		// Headers, binary-aligned: Authorization, version-interpolated UA,
		// runtime acting-user id (no hardcoded ids). The instance header is
		// kept on chat: upstream keys the free session to it (empirically
		// required; without it chat 428s waiting_room_required even with an
		// active session), and the working CLI-era bridge sent it too.
		opts = append(opts, provider.WithHeader("User-Agent", "ai-sdk/openai-compatible/0.0.167/codebuff"))
		if au, ok := p.cfg.Quirks.FreebuffActor.(interface{ ActingUserID(context.Context) string }); ok {
			if id := au.ActingUserID(ctx); id != "" {
				opts = append(opts, provider.WithHeader("x-freebuff-acting-user-id", id))
			}
		}
		if inst, ok := p.cfg.Quirks.FreebuffActor.(freebuffInstanceIDer); ok {
			if id := inst.InstanceID(); id != "" {
				opts = append(opts, provider.WithHeader("x-freebuff-instance-id", id))
			}
		}
		tok := p.cfg.APIKey
		if tok == "" {
			if ts := p.cfg.TokenSource; ts != nil {
				if t, err := ts.Token(ctx); err == nil && t != nil {
					tok = t.AccessToken
				}
			}
		}
		if tok != "" {
			opts = append(opts, provider.WithHeader("Authorization", "Bearer "+tok))
		}
	}

	// 6. FreebuffDefaultTool (Buffy system prompt + default tool + codebuff_metadata)
	if p.cfg.Quirks.FreebuffDefaultTool {
		hasSystem := false
		for _, m := range msgs {
			if m != nil && m.Role == "system" {
				hasSystem = true
				break
			}
		}
		if !hasSystem {
			sysMsg := &ir.Message{
				Role:   "system",
				Blocks: []ir.Block{ir.TextBlock{Text: "You are Buffy, the coding agent behind Codebuff."}},
			}
			msgs = append([]*ir.Message{sysMsg}, msgs...)
		}

		defaultTool := map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "read_files",
				"description": "Read files from project",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"paths": map[string]any{
							"type":  "array",
							"items": map[string]any{"type": "string"},
						},
					},
				},
			},
		}

		var finalTools []any
		if reqConfig.ExtraBody != nil {
			if tools, ok := reqConfig.ExtraBody["tools"].([]any); ok && len(tools) > 0 {
				finalTools = append(finalTools, defaultTool)
				finalTools = append(finalTools, tools...)
			}
		}
		if len(finalTools) == 0 {
			finalTools = []any{defaultTool}
		}

		runID := "run-default"
		if rp, ok := p.cfg.Quirks.FreebuffActor.(interface {
			StartRun(context.Context, string) (any, error)
		}); ok {
			// Free mode only accepts specific agent/model combos: /agent-runs
			// wants the base3-free-* agentId, not the raw model id (upstream
			// 403s free_mode_invalid_agent_model otherwise). Unknown models
			// pass through unchanged so newly listed upstream models still work.
			if r, err := rp.StartRun(ctx, freebuffAgentID(model)); err == nil {
				if s, ok := r.(string); ok && s != "" {
					runID = s
				} else if rID, ok := r.(interface{ GetRunID() string }); ok {
					if id := rID.GetRunID(); id != "" {
						runID = id
					}
				}
			}
		}

		// codebuff_metadata + provider, binary-identical to the shipped
		// client: run_id from START, client_id = base36 random per run (the
		// CLI's Math.random().toString(36).substring(2,15)), stringified step
		// number, free cost mode. No freebuff_instance_id (the instance rides
		// the session API), and provider.allow_fallbacks=false for official
		// models (inverted vs the old bridge).
		meta := map[string]any{
			"run_id":          runID,
			"cost_mode":       "free",
			"client_id":       freebuffClientSessionID(),
			"llm_step_number": "1",
		}

		extraBody := make(map[string]any)
		for k, v := range reqConfig.ExtraBody {
			extraBody[k] = v
		}
		extraBody["tools"] = finalTools
		extraBody["codebuff_metadata"] = meta
		extraBody["provider"] = map[string]any{"allow_fallbacks": false}
		opts = append(opts, provider.WithExtraBody(extraBody))
	}

	return ctx, msgs, opts, nil
}

// hubTokenSource narrows a lane token source down to plain Token() for the
// llmhub client (see the wiring comment in New). Refreshes still flow through
// the very same source - Provider.Token, Provider.Refresh and the reactive
// retry below all hold the original, invalidatable one.
type hubTokenSource struct {
	inner interface {
		Token(ctx context.Context) (*llmauth.Token, error)
	}
}

// Token implements llmhub's auth.TokenSource.
func (h hubTokenSource) Token(ctx context.Context) (*llmauth.Token, error) {
	return h.inner.Token(ctx)
}

// upstreamStatusRe matches the status-bearing errors llmhub's OpenAI-compatible
// client returns ("<provider>: http <code>: <body>"). The status is the FIRST
// such segment, so a request body that happens to quote another status cannot
// be mistaken for the response status.
var upstreamStatusRe = regexp.MustCompile(`: http (\d{3}):`)

// upstreamStatus extracts the upstream HTTP status code from an llmhub request
// error, or 0 when the error is not a status error.
func upstreamStatus(err error) int {
	if err == nil {
		return 0
	}
	m := upstreamStatusRe.FindStringSubmatch(err.Error())
	if m == nil {
		return 0
	}
	code, convErr := strconv.Atoi(m[1])
	if convErr != nil {
		return 0
	}
	return code
}

// isCredentialRejection reports whether err is an upstream "your credential is
// bad" answer: 401 always, and 403 with it because that is exactly how xai
// reports a dead OAuth access token (403 unauthenticated:bad-credentials).
// A non-credential 403 (model access, quota) costs at most one extra attempt
// before the error is returned honestly, so the classification stays simple.
func isCredentialRejection(err error) bool {
	switch upstreamStatus(err) {
	case http.StatusUnauthorized, http.StatusForbidden:
		return true
	default:
		return false
	}
}

// retryWithFreshCredential implements the reactive half of automatic token
// refresh: Invalidate the credential the upstream just rejected, mint a fresh
// one, and report whether the caller should send the request again.
//
// It is called strictly BEFORE the first byte of a response or stream reaches
// the client, the same boundary hublane failover honours. A lane without a
// token source (static api_key) has nothing to refresh and never retries; a
// refresh that fails leaves the original error for the caller.
func (p *Provider) retryWithFreshCredential(ctx context.Context, err error) bool {
	if p.cfg.TokenSource == nil || !isCredentialRejection(err) {
		return false
	}

	if inv, ok := p.cfg.TokenSource.(interface{ Invalidate(string) }); ok {
		inv.Invalidate("")
	}
	if _, refreshErr := p.cfg.TokenSource.Token(ctx); refreshErr != nil {
		log.Printf("[auth] reactive credential refresh failed lane=%s: %v", p.name, refreshErr)
		return false
	}
	if exp, ok := p.TokenExpiresAt(); ok {
		// Lane + new expiry only: never the token itself.
		log.Printf("[auth] credential refreshed lane=%s reason=upstream_rejection new_expiry=%s", p.name, exp.UTC().Format(time.RFC3339))
		return true
	}
	log.Printf("[auth] credential refreshed lane=%s reason=upstream_rejection", p.name)
	return true
}

// TokenExpiresAt reports when this lane's credential expires, so the server's
// proactive refresher can refresh it before it dies. The bool is false for
// lanes that hold no expiring credential (a static api_key, or a token source
// without an expiry surface).
func (p *Provider) TokenExpiresAt() (time.Time, bool) {
	if p == nil || p.cfg.TokenSource == nil {
		return time.Time{}, false
	}
	expiring, ok := p.cfg.TokenSource.(ExpiringTokenSource)
	if !ok {
		return time.Time{}, false
	}
	return expiring.ExpiresAt()
}

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	if actor := p.freebuffActor(); actor != nil {
		if err := actor.Acquire(ctx); err != nil {
			return nil, err
		}
		defer actor.Release()
	}

	var err error
	ctx, msgs, opts, err = p.applyRequestTransforms(ctx, msgs, opts)
	if err != nil {
		return nil, err
	}

	resp, err := p.adapter.Generate(ctx, msgs, opts...)
	if err != nil {
		// Reactive refresh: one retry with a fresh credential when the
		// upstream rejected the one this request carried. A second consecutive
		// rejection falls through and is returned to the caller honestly.
		if p.retryWithFreshCredential(ctx, err) {
			resp, err = p.adapter.Generate(ctx, msgs, opts...)
		}
		if err != nil {
			return nil, err
		}
	}

	return filterReasoningResponse(resp, p.cfg.Quirks.EchoReasoning), nil
}

// Stream implements provider.InferenceProvider.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	actor := p.freebuffActor()
	if actor != nil {
		if err := actor.Acquire(ctx); err != nil {
			return nil, err
		}
	}

	var err error
	ctx, msgs, opts, err = p.applyRequestTransforms(ctx, msgs, opts)
	if err != nil {
		if actor != nil {
			actor.Release()
		}
		return nil, err
	}

	inCh, err := p.adapter.Stream(ctx, msgs, opts...)
	if err != nil {
		// Reactive refresh at the same pre-first-byte boundary as Generate.
		// Once a stream has produced its first byte there is no retry at all:
		// the failure surfaces as an event on the channel instead.
		if p.retryWithFreshCredential(ctx, err) {
			inCh, err = p.adapter.Stream(ctx, msgs, opts...)
		}
		if err != nil {
			if actor != nil {
				actor.Release()
			}
			return nil, err
		}
	}

	ch := inCh
	if actor != nil {
		actorCh := make(chan ir.Event, 64)
		go func() {
			defer close(actorCh)
			defer actor.Release()
			for {
				select {
				case <-ctx.Done():
					return
				case ev, ok := <-inCh:
					if !ok {
						return
					}
					// The send must stay cancellable: a caller that stops
					// draining the stream (client disconnect, failover) would
					// otherwise park this goroutine on the channel send and
					// hold the actor's single-slot lock forever.
					select {
					case <-ctx.Done():
						return
					case actorCh <- ev:
					}
				}
			}
		}()
		ch = actorCh
	}

	return filterReasoningStream(ctx, ch, p.cfg.Quirks.EchoReasoning), nil
}

func sanitizeReasoning(msgs []*ir.Message, echoReasoning bool) []*ir.Message {
	if echoReasoning {
		return msgs
	}
	out := make([]*ir.Message, 0, len(msgs))
	for _, m := range msgs {
		if m == nil {
			continue
		}
		var newBlocks []ir.Block
		for _, b := range m.Blocks {
			switch b.(type) {
			case ir.ReasoningBlock, *ir.ReasoningBlock:
				// Drop reasoning when EchoReasoning is false
			default:
				newBlocks = append(newBlocks, b)
			}
		}
		cp := *m
		cp.Blocks = newBlocks
		out = append(out, &cp)
	}
	return out
}

func filterReasoningResponse(resp *ir.Response, echoReasoning bool) *ir.Response {
	if echoReasoning || resp == nil || resp.Message == nil {
		return resp
	}
	var newBlocks []ir.Block
	for _, b := range resp.Message.Blocks {
		switch b.(type) {
		case ir.ReasoningBlock, *ir.ReasoningBlock:
			// Drop reasoning when EchoReasoning is false
		default:
			newBlocks = append(newBlocks, b)
		}
	}
	resp.Message.Blocks = newBlocks
	return resp
}

func filterReasoningStream(ctx context.Context, in <-chan ir.Event, echoReasoning bool) <-chan ir.Event {
	if echoReasoning {
		return in
	}
	out := make(chan ir.Event, 64)
	go func() {
		defer close(out)
		for {
			select {
			case <-ctx.Done():
				return
			case ev, ok := <-in:
				if !ok {
					return
				}
				if _, ok := ev.(ir.EventReasoningDelta); ok {
					continue
				}
				out <- ev
			}
		}
	}()
	return out
}

// Models returns the cached list of models discovered from /v1/models.
func (p *Provider) Models() []string {
	p.mu.RLock()
	if len(p.models) > 0 {
		out := make([]string, len(p.models))
		copy(out, p.models)
		p.mu.RUnlock()
		return out
	}
	p.mu.RUnlock()

	if p.cfg.Quirks.ModelListPassthrough {
		ctx, cancel := context.WithTimeout(context.Background(), modelDiscoveryBudget)
		defer cancel()
		models, err := p.FetchModels(ctx)
		if err == nil {
			return models
		}
	}
	return nil
}

// FetchModels queries GET /v1/models (or /models if baseURL ends in /v1) and updates cached models.
func (p *Provider) FetchModels(ctx context.Context) ([]string, error) {
	endpoint := p.baseURL + "/models"
	if !strings.HasSuffix(p.baseURL, "/v1") {
		endpoint = p.baseURL + "/v1/models"
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	if p.apiKey != "" && p.apiKey != "openaicompat" && p.apiKey != "placeholder" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("models endpoint returned status %d", resp.StatusCode)
	}

	// Upstream model rows have no OpenAI standard beyond id, so every extra
	// field ultiproxy understands is declared here: vLLM (max_model_len),
	// OpenRouter (context_length, top_provider.*, architecture.*),
	// LiteLLM/OpenAI-compatible gateways (max_output_tokens,
	// max_completion_tokens, supports_vision, top-level modality arrays),
	// llama.cpp (meta.n_ctx / meta.n_ctx_train) and Groq (context_window).
	var raw struct {
		Data []struct {
			ID            string   `json:"id"`
			MaxModelLen   int      `json:"max_model_len"`
			ContextLength int      `json:"context_length"`
			ContextWindow int      `json:"context_window"`
			MaxOutputTok  int      `json:"max_output_tokens"`
			MaxComplTok   int      `json:"max_completion_tokens"`
			SupportsVis   bool     `json:"supports_vision"`
			InModalities  []string `json:"input_modalities"`
			OutModalities []string `json:"output_modalities"`
			TopProvider   struct {
				ContextLength    int `json:"context_length"`
				MaxCompletionTok int `json:"max_completion_tokens"`
			} `json:"top_provider"`
			Meta struct {
				ContextLength int `json:"context_length"`
				NCtx          int `json:"n_ctx"`
				NCtxTrain     int `json:"n_ctx_train"`
			} `json:"meta"`
			Architecture struct {
				InModalities  []string `json:"input_modalities"`
				OutModalities []string `json:"output_modalities"`
				Modality      string   `json:"modality"`
			} `json:"architecture"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}

	var models []string
	var info []provider.ModelInfo
	for _, m := range raw.Data {
		if m.ID == "" {
			continue
		}
		models = append(models, m.ID)
		info = append(info, provider.ModelInfo{
			ID: m.ID,
			ContextLength: pickContextLength(
				m.MaxModelLen,
				m.ContextLength,
				m.TopProvider.ContextLength,
				m.Meta.ContextLength,
				m.Meta.NCtx,
				m.ContextWindow,
				m.Meta.NCtxTrain,
			),
			MaxOutput: pickContextLength(
				m.TopProvider.MaxCompletionTok,
				m.MaxOutputTok,
				m.MaxComplTok,
			),
			InputModalities:  pickInputModalities(m.Architecture.InModalities, m.InModalities, m.Architecture.Modality, m.SupportsVis),
			OutputModalities: pickOutputModalities(m.Architecture.OutModalities, m.OutModalities, m.Architecture.Modality),
		})
	}

	p.mu.Lock()
	p.models = models
	p.modelInfo = info
	p.mu.Unlock()

	return models, nil
}

// pickContextLength returns the first positive value from the upstream field
// variants it is handed. It is used for the context window (vLLM max_model_len,
// OpenRouter context_length / top_provider.context_length, llama.cpp
// meta.context_length / meta.n_ctx, Groq context_window, trained
// meta.n_ctx_train last) and for the output cap (top_provider
// max_completion_tokens, then max_output_tokens / max_completion_tokens). Zero
// means the upstream did not say, and is never advertised.
func pickContextLength(values ...int) int {
	for _, v := range values {
		if v > 0 {
			return v
		}
	}
	return 0
}

// pickInputModalities resolves the advertised input modalities of one upstream
// row: explicit arrays win (architecture first, then top level), then an
// HF-style architecture.modality string ("text+image->text"), then a bare
// supports_vision flag, which advertises image input without naming the rest.
// nil means unknown - never an empty array standing in for "unknown".
func pickInputModalities(architecture, topLevel []string, modalityString string, supportsVision bool) []string {
	if in := modelmeta.NormalizeModalities(architecture); len(in) > 0 {
		return in
	}
	if in := modelmeta.NormalizeModalities(topLevel); len(in) > 0 {
		return in
	}
	if modalityString != "" {
		in, _ := modelmeta.ParseModalityString(modalityString)
		if len(in) > 0 {
			return in
		}
	}
	if supportsVision {
		return []string{modelmeta.ModalityImage}
	}
	return nil
}

// pickOutputModalities resolves the advertised output modalities of one
// upstream row: explicit arrays first (architecture, then top level), then the
// output side of an architecture.modality string.
func pickOutputModalities(architecture, topLevel []string, modalityString string) []string {
	if out := modelmeta.NormalizeModalities(architecture); len(out) > 0 {
		return out
	}
	if out := modelmeta.NormalizeModalities(topLevel); len(out) > 0 {
		return out
	}
	if modalityString != "" {
		_, out := modelmeta.ParseModalityString(modalityString)
		if len(out) > 0 {
			return out
		}
	}
	return nil
}

// Quota implements provider.QuotaProvider.
func (p *Provider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	// Freebuff routes quota through its account actor (FetchUsage), not the
	// grpc-web credits observer — exactly like the pre-F2 freebuff lane did.
	if p.cfg.Quirks.FreebuffActor != nil {
		return freebuffQuota(ctx, p.cfg.Quirks.FreebuffActor)
	}
	if p.cfg.Quirks.CreditsQuotaObserver == "" {
		return nil, errors.New("quota observer not enabled")
	}
	tok, err := p.Token(ctx)
	if err != nil && p.apiKey != "" && p.apiKey != "openaicompat" && p.apiKey != "placeholder" {
		tok = p.apiKey
	}
	return fetchCreditsQuota(ctx, p.httpClient, p.cfg.Quirks.CreditsQuotaObserver, tok)
}

// Login implements provider.AuthProvider.
//
// Quirk-routed (ported from the pre-F2 xai and freebuff lanes):
//   - AuthViaOAuthManager: xai device-OAuth flow (RequestDeviceCode -> PollToken ->
//     persist via auth.Manager under DataDir so Token() can serve it later).
//   - FreebuffActor != nil: persist the configured/web token (env or api_key)
//     into ultraproxy state and push it into the actor.
//   - otherwise: provider.ErrNotImplemented (plain openai endpoint needs no login).
func (p *Provider) Login(ctx context.Context) error {
	if p.cfg.Quirks.AuthViaOAuthManager {
		return p.loginXAI(ctx)
	}
	if p.cfg.Quirks.FreebuffActor != nil {
		return p.loginFreebuff(ctx)
	}
	return provider.ErrNotImplemented
}

// loginXAI runs the full xAI device-authorization flow to completion. Used by
// the legacy blocking Login(); MCP-driven flows call StartLogin then
// CompleteLogin instead.
func (p *Provider) loginXAI(ctx context.Context) error {
	if _, err := p.startXAI(ctx); err != nil {
		return err
	}
	return p.completeXAI(ctx)
}

// startXAI requests the device authorization code and returns the sign-in URL
// + user code without blocking on the human. The pending flow is kept so
// CompleteLogin can finish it later.
func (p *Provider) startXAI(ctx context.Context) (*provider.LoginStartInfo, error) {
	deviceURL := p.cfg.DeviceAuthURL
	if deviceURL == "" {
		deviceURL = defaultXAIDeviceURL
	}
	tokenURL := p.cfg.TokenURL
	if tokenURL == "" {
		tokenURL = defaultXAITokenURL
	}
	cfg := oauth.DeviceFlowConfig{
		ClientID:      defaultXAIClientID,
		DeviceAuthURL: deviceURL,
		TokenURL:      tokenURL,
		Scope:         defaultXAIScope,
		HTTPClient:    p.httpClient,
	}

	dcr, err := oauth.RequestDeviceCode(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: xai login device code failed: %w", err)
	}

	p.mu.Lock()
	p.pendingXAI = &xaiPendingFlow{cfg: cfg, deviceCode: dcr.DeviceCode, interval: dcr.Interval}
	p.mu.Unlock()

	return &provider.LoginStartInfo{
		Kind:            provider.LoginFlowDevice,
		VerificationURI: dcr.VerificationURI,
		UserCode:        dcr.UserCode,
		ExpiresIn:       dcr.ExpiresIn,
	}, nil
}

// completeXAI polls the device authorization until the human approves, then
// persists the credential into the auth manager.
func (p *Provider) completeXAI(ctx context.Context) error {
	p.mu.RLock()
	pending := p.pendingXAI
	p.mu.RUnlock()
	if pending == nil {
		return errors.New("openaicompat: xai: no pending device flow — call StartLogin first")
	}

	tokResp, err := oauth.PollToken(ctx, pending.cfg, pending.deviceCode, pending.interval)
	if err != nil {
		return fmt.Errorf("openaicompat: xai login token poll failed: %w", err)
	}
	p.mu.Lock()
	p.pendingXAI = nil
	p.mu.Unlock()

	// Persist so Token() (via the OAuth manager TokenSource) can serve it later.
	dataDir := p.cfg.DataDir
	if dataDir == "" {
		dataDir = filepath.Join(os.TempDir(), "ultiproxy-xai-auth")
	}
	refresher := oauth.MakeRefresher(p.httpClient, pending.cfg.TokenURL, defaultXAIClientID, "")
	mgr, err := auth.NewManager(dataDir, refresher)
	if err != nil {
		return fmt.Errorf("openaicompat: xai login credential store: %w", err)
	}
	expiresIn := tokResp.ExpiresIn
	if expiresIn <= 0 {
		expiresIn = 3600
	}
	cred := auth.Credential{
		Provider:     "xai",
		AccessToken:  tokResp.AccessToken,
		RefreshToken: tokResp.RefreshToken,
		ExpiresAt:    time.Now().Add(time.Duration(expiresIn) * time.Second),
		ClientID:     defaultXAIClientID,
	}
	if err := mgr.Store(ctx, defaultXAIClientID, cred); err != nil {
		return fmt.Errorf("openaicompat: xai login credential store: %w", err)
	}
	return nil
}

// loginFreebuff persists the freebuff token from the provider's own
// configuration (APIKey set via env, MCP add_provider, or an explicit token)
// into ultiproxy state, then pushes it into the actor. It never reads an
// external CLI credential store.
func (p *Provider) loginFreebuff(ctx context.Context) error {
	tok := p.cfg.APIKey
	if tok == "" {
		tok = os.Getenv("ULTIPROXY_FREEBUFF_TOKEN")
	}
	if tok == "" {
		tok = os.Getenv("FREEBUFF_TOKEN")
	}
	if tok == "" {
		return errors.New("freebuff: no token configured — set ULTIPROXY_FREEBUFF_TOKEN or pass api_key to add_provider")
	}
	if setter, ok := p.cfg.Quirks.FreebuffActor.(freebuffTokenSetter); ok {
		setter.SetToken(tok)
	}
	p.apiKey = tok
	p.cfg.APIKey = tok

	dataDir := p.cfg.DataDir
	if dataDir != "" {
		if err := os.MkdirAll(dataDir, 0755); err != nil {
			return fmt.Errorf("freebuff: create data dir: %w", err)
		}
		tokenFile := filepath.Join(dataDir, "freebuff_token")
		if err := os.WriteFile(tokenFile, []byte(tok+"\n"), 0600); err != nil {
			return fmt.Errorf("freebuff: persist token: %w", err)
		}
	}
	return nil
}

// StartLogin implements provider.InteractiveAuthProvider: returns the xAI
// device sign-in URL + user code without blocking (used by MCP tools). No
// pending flow is kept for freebuff/plain lanes — they return ErrNotImplemented.
func (p *Provider) StartLogin(ctx context.Context) (*provider.LoginStartInfo, error) {
	if p.cfg.Quirks.AuthViaOAuthManager {
		return p.startXAI(ctx)
	}
	return nil, provider.ErrNotImplemented
}

// CompleteLogin implements provider.InteractiveAuthProvider. For xai it
// polls the device authorization until approval; authorizationCode is unused.
func (p *Provider) CompleteLogin(ctx context.Context, authorizationCode string) error {
	if p.cfg.Quirks.AuthViaOAuthManager {
		return p.completeXAI(ctx)
	}
	if p.cfg.Quirks.FreebuffActor != nil {
		return p.loginFreebuff(ctx)
	}
	return provider.ErrNotImplemented
}

// Token implements provider.AuthProvider.
func (p *Provider) Token(ctx context.Context) (string, error) {
	if p.cfg.TokenSource != nil {
		tok, err := p.cfg.TokenSource.Token(ctx)
		if err != nil {
			return "", err
		}
		return tok.AccessToken, nil
	}
	if p.apiKey != "" && p.apiKey != "openaicompat" && p.apiKey != "placeholder" {
		return p.apiKey, nil
	}
	return "", errors.New("no token available")
}

// Refresh implements provider.AuthProvider.
func (p *Provider) Refresh(ctx context.Context) error {
	if inv, ok := p.cfg.TokenSource.(interface{ Invalidate(string) }); ok {
		inv.Invalidate("")
		_, err := p.cfg.TokenSource.Token(ctx)
		return err
	}
	return nil
}

var (
	_ provider.InferenceProvider = (*Provider)(nil)
	_ provider.QuotaProvider     = (*Provider)(nil)
	_ provider.AuthProvider      = (*Provider)(nil)
)

// ModelDiscoveryEnabled reports whether this lane participates in automatic
// model discovery (the resolved quirks.model_list_passthrough flag). The
// server's discovery loop uses it to leave opted-out lanes alone.
func (p *Provider) ModelDiscoveryEnabled() bool {
	return p.cfg.Quirks.ModelListPassthrough
}

// DefaultModel returns the lane's configured default upstream model
// (quirks.default_model, e.g. augure's "tofino-3"), "" when the lane has
// none. Callers that must list routable ids without contacting the upstream
// (the aggregated /v1/models handler and the list_models MCP tool) use it to
// advertise "<lane>/<default>" for lanes that have no discovery cache.
func (p *Provider) DefaultModel() string {
	return p.cfg.Quirks.DefaultModel
}

// CachedModels returns the upstream model ids discovered so far (startup
// discovery for ModelListPassthrough lanes, or the last successful
// FetchModels) WITHOUT contacting the upstream. Callers that must never fan
// out (the aggregated /v1/models handler) read this instead of Models().
func (p *Provider) CachedModels() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.models) == 0 {
		return nil
	}
	out := make([]string, len(p.models))
	copy(out, p.models)
	return out
}

// CachedModelInfo returns discovered ids plus context windows without
// contacting the upstream. ContextLength is 0 when the upstream omitted it.
func (p *Provider) CachedModelInfo() []provider.ModelInfo {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.modelInfo) == 0 {
		return nil
	}
	out := make([]provider.ModelInfo, len(p.modelInfo))
	copy(out, p.modelInfo)
	return out
}
