package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/smhanov/llmhub"
	hubopenai "github.com/smhanov/llmhub/providers/openai"
	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/ir"
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

// Provider implements provider.InferenceProvider and optionally provider.QuotaProvider and provider.AuthProvider.
type Provider struct {
	adapter    *hublane.Adapter
	cfg        Config
	httpClient *http.Client
	baseURL    string
	apiKey     string
	name       string

	mu     sync.RWMutex
	models []string
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

	// Quirks: AuthViaWorkspaceCookie environment fallbacks
	if cfg.Quirks.AuthViaWorkspaceCookie {
		if cfg.WorkspaceID == "" {
			cfg.WorkspaceID = os.Getenv("OPENCODE_WORKSPACE_ID")
		}
		if cfg.SessionCookie == "" {
			cfg.SessionCookie = os.Getenv("OPENCODE_SESSION_COOKIE")
		}
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

	hubKey := cfg.APIKey
	if hubKey == "" {
		hubKey = "openaicompat"
	}

	hubOpts := []llmhub.Option{llmhub.WithHTTPClient(client), llmhub.WithRetryOnStatus(http.StatusTooManyRequests, false)}
	if cfg.BaseURL != "" {
		hubOpts = append(hubOpts, llmhub.WithBaseURL(cfg.BaseURL))
	}
	if cfg.TokenSource != nil {
		hubOpts = append(hubOpts, llmhub.WithTokenSource(cfg.TokenSource))
	}

	// Attach headers for AuthViaWorkspaceCookie if available at construction time
	if cfg.Quirks.AuthViaWorkspaceCookie {
		cookieHeader := cfg.SessionCookie
		if cookieHeader != "" && !strings.Contains(cookieHeader, "=") {
			cookieHeader = "session=" + cookieHeader
		}
		if cookieHeader != "" {
			hubOpts = append(hubOpts, llmhub.WithHeader("Cookie", cookieHeader))
		}
		if cfg.WorkspaceID != "" {
			hubOpts = append(hubOpts, llmhub.WithHeader("X-Workspace-ID", cfg.WorkspaceID))
		}
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

	// Quirk: ModelListPassthrough startup model discovery
	if cfg.Quirks.ModelListPassthrough {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
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

func (p *Provider) resolveMaxTokens(model string, requested int) int {
	if requested > 0 {
		return requested
	}
	if p.cfg.Quirks.CodingPlanPath || len(p.cfg.Quirks.MaxTokensByModel) > 0 {
		if len(p.cfg.Quirks.MaxTokensByModel) > 0 {
			for pattern, maxTok := range p.cfg.Quirks.MaxTokensByModel {
				if strings.Contains(model, pattern) || model == pattern {
					return maxTok
				}
			}
			if p.cfg.Quirks.CodingPlanPath {
				return 131072
			}
		} else if p.cfg.Quirks.CodingPlanPath {
			if strings.Contains(model, "4.5-air") {
				return 98304
			}
			return 131072
		}
	}
	return 0
}

func (p *Provider) applyRequestTransforms(ctx context.Context, msgs []*ir.Message, opts []provider.Option) (context.Context, []*ir.Message, []provider.Option, error) {
	reqConfig := provider.NewRequestConfig(opts...)

	// 1. Model resolution
	model := p.resolveModel(reqConfig.Model)
	if model != "" {
		opts = append(opts, provider.WithModel(model))
	}

	// 2. MaxTokens resolution (CodingPlanPath / MaxTokensByModel)
	maxTokens := p.resolveMaxTokens(model, reqConfig.MaxTokens)
	if maxTokens > 0 {
		opts = append(opts, provider.WithMaxTokens(maxTokens))
	}

	// 3. Reasoning sanitization (EchoReasoning)
	msgs = sanitizeReasoning(msgs, p.cfg.Quirks.EchoReasoning)

	// 4. OpenCode Workspace / Session-Cookie auth
	if p.cfg.Quirks.AuthViaWorkspaceCookie {
		sessCookie := p.cfg.SessionCookie
		wsID := p.cfg.WorkspaceID

		if p.cfg.TokenSource != nil {
			if wts, ok := p.cfg.TokenSource.(interface {
				WorkspaceID() string
				SessionCookie() string
			}); ok {
				if wsID == "" {
					wsID = wts.WorkspaceID()
				}
				if sessCookie == "" {
					sessCookie = wts.SessionCookie()
				}
			} else if tok, err := p.cfg.TokenSource.Token(ctx); err == nil && tok != nil {
				if sessCookie == "" {
					sessCookie = tok.AccessToken
				}
			}
		}

		if sessCookie != "" {
			cookieHeader := sessCookie
			if !strings.Contains(cookieHeader, "=") {
				cookieHeader = "session=" + cookieHeader
			}
			opts = append(opts, provider.WithHeader("Cookie", cookieHeader))
		}
		if wsID != "" {
			opts = append(opts, provider.WithHeader("X-Workspace-ID", wsID))
		}
	}

	// 5. Freebuff headers (x-freebuff-instance-id, x-freebuff-acting-user-id,
	// User-Agent) — ported from the pre-F2 freebuff lane's setHeaders. Codebuff
	// requires these; without them the Actor rejects the request.
	if p.cfg.Quirks.FreebuffActor != nil {
		opts = append(opts, provider.WithHeader("User-Agent", "ai-sdk/openai-compatible/0.0.0-test/codebuff ai-sdk/provider-utils/3.0.25 runtime/node.js/v22.23.2"))
		if inst, ok := p.cfg.Quirks.FreebuffActor.(freebuffInstanceIDer); ok {
			if id := inst.InstanceID(); id != "" {
				opts = append(opts, provider.WithHeader("x-freebuff-instance-id", id))
			}
		}
		opts = append(opts, provider.WithHeader("x-freebuff-acting-user-id", "adcc6f59-fffd-4735-8c09-703eb3158941"))
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

		inst := ""
		if ip, ok := p.cfg.Quirks.FreebuffActor.(interface{ InstanceID() string }); ok {
			inst = ip.InstanceID()
		}
		runID := "run-default"
		if rp, ok := p.cfg.Quirks.FreebuffActor.(interface {
			StartRun(context.Context, string) (any, error)
		}); ok {
			if r, err := rp.StartRun(ctx, model); err == nil {
				if s, ok := r.(string); ok && s != "" {
					runID = s
				} else if rID, ok := r.(interface{ GetRunID() string }); ok {
					if id := rID.GetRunID(); id != "" {
						runID = id
					}
				}
			}
		}

		meta := map[string]any{
			"run_id":               runID,
			"freebuff_instance_id": inst,
			"cost_mode":            "free",
			"client_id":            "cli-" + inst,
			"llm_step_number":      "1",
		}

		extraBody := make(map[string]any)
		for k, v := range reqConfig.ExtraBody {
			extraBody[k] = v
		}
		extraBody["tools"] = finalTools
		extraBody["codebuff_metadata"] = meta
		opts = append(opts, provider.WithExtraBody(extraBody))
	}

	return ctx, msgs, opts, nil
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
		return nil, err
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
		if actor != nil {
			actor.Release()
		}
		return nil, err
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
					actorCh <- ev
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
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
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

	var raw struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}

	var models []string
	for _, m := range raw.Data {
		if m.ID != "" {
			models = append(models, m.ID)
		}
	}

	p.mu.Lock()
	p.models = models
	p.mu.Unlock()

	return models, nil
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
