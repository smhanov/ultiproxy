package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/anthropichub"
	"github.com/smhanov/ultiproxy/pkg/provider/antigravity"
	"github.com/smhanov/ultiproxy/pkg/provider/codex"
	"github.com/smhanov/ultiproxy/pkg/provider/copilot"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	spikesfreebuff "github.com/smhanov/ultiproxy/pkg/spikes/freebuff"
)

const (
	xaiDefaultClientID   = "b1a00492-073a-47ea-816f-4c329264a828"
	xaiDefaultBillingURL = "https://grok.com/grok_api_v2.GrokBuildBilling/GetGrokCreditsConfig"
	xaiDefaultBaseURL    = "https://api.x.ai"
)

type freebuffActorAdapter struct {
	actor *spikesfreebuff.FreebuffAccountActor
}

func (a *freebuffActorAdapter) Acquire(ctx context.Context) error {
	return a.actor.Acquire(ctx)
}

func (a *freebuffActorAdapter) Release() {
	_ = a.actor.Release()
}

func (a *freebuffActorAdapter) InstanceID() string {
	return a.actor.InstanceID()
}

func (a *freebuffActorAdapter) StartRun(ctx context.Context, model string) (any, error) {
	return a.actor.StartRun(ctx, model)
}

// SetToken pushes a token into the actor (used by the freebuff login flow).
func (a *freebuffActorAdapter) SetToken(tok string) {
	a.actor.SetToken(tok)
}

// FetchUsage forwards to the underlying actor for the freebuff quota path.
func (a *freebuffActorAdapter) FetchUsage(ctx context.Context, fingerprintID string) ([]byte, error) {
	return a.actor.FetchUsage(ctx, fingerprintID)
}

// SessionInfo returns the current session's instance id + model for quota Detail.
func (a *freebuffActorAdapter) SessionInfo(ctx context.Context) (string, string, error) {
	sess, err := a.actor.GetSession(ctx)
	if err != nil {
		return "", "", err
	}
	return sess.InstanceID, sess.Model, nil
}

// The openaicompat freebuff quirk duck-types this session-lifecycle surface
// before every chat (reconcile, bind to the canonical model, delete+re-bind on
// model switch). The adapter must forward all of it — dropping any method
// silently disables the lifecycle and every chat 428s waiting_room_required.

func (a *freebuffActorAdapter) Reconcile(ctxs ...context.Context) error {
	return a.actor.Reconcile(ctxs...)
}

func (a *freebuffActorAdapter) BoundModel() string {
	return a.actor.BoundModel()
}

func (a *freebuffActorAdapter) DeleteSession(ctxs ...context.Context) error {
	return a.actor.DeleteSession(ctxs...)
}

func (a *freebuffActorAdapter) Bind(ctxOrModel any, optionalModel ...string) error {
	return a.actor.Bind(ctxOrModel, optionalModel...)
}

// ActingUserID forwards the live /me-resolved account id for the
// x-freebuff-acting-user-id header (binary-identical header set).
func (a *freebuffActorAdapter) ActingUserID(ctx context.Context) string {
	return a.actor.ActingUserID(ctx)
}

// readJSONField loads a key from a JSON file, returning "" if unavailable.
func readJSONField(path string, fields ...string) (string, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return "", false
	}
	cur := any(m)
	for _, f := range fields {
		m2, ok := cur.(map[string]any)
		if !ok {
			return "", false
		}
		v, ok := m2[f]
		if !ok {
			return "", false
		}
		cur = v
	}
	s, ok := cur.(string)
	if !ok || s == "" {
		return "", false
	}
	return s, true
}

// firstEnv returns the first non-empty environment variable.
func firstEnv(keys ...string) string {
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}

func newOAuthManager(credDir string) (*auth.Manager, error) {
	return auth.NewManager(credDir, nil)
}

func managerHasToken(mgr *auth.Manager, key string) bool {
	if mgr == nil {
		return false
	}
	cred, err := mgr.LoadFromDisk(key)
	return err == nil && cred.AccessToken != ""
}

// registerProviders wires upstream adapters into the registry. Registration is
// opt-in per provider and driven by environment variables OR ultiproxy-owned
// credential stores. Antigravity never reads external CLI credential stores.
func registerProviders(registry *provider.Registry) {
	home, _ := os.UserHomeDir()
	stateDir := firstEnv("ULTIPROXY_DATA_DIR", "ULTIPROXY_STATE_DIR")
	if stateDir == "" {
		stateDir = filepath.Join(home, ".local", "state", "ultiproxy")
	}

	add := func(name string, bundle provider.Provider) {
		registry.Register(bundle)
		log.Printf("[providers] registered %s", name)
	}

	// Z.ai Coding Plan (GLM) — zero marginal cost on subscription.
	zaiKey := firstEnv("ZAI_API_KEY", "ULTIPROXY_ZAI_API_KEY")
	if zaiKey != "" {
		if p, err := openaicompat.New(openaicompat.Config{
			Name:   "zai",
			APIKey: zaiKey,
			Quirks: openaicompat.Quirks{
				CodingPlanPath: true,
			},
		}); err == nil {
			add("zai", p.Provider())
		} else {
			log.Printf("[providers] zai: %v", err)
		}
	}

	// DeepSeek (metered API).
	if key := firstEnv("DEEPSEEK_API_KEY", "ULTIPROXY_DEEPSEEK_API_KEY"); key != "" {
		baseURL := firstEnv("DEEPSEEK_BASE_URL", "ULTIPROXY_DEEPSEEK_BASE_URL")
		if baseURL == "" {
			baseURL = "https://api.deepseek.com/v1"
		}
		if p, err := openaicompat.New(openaicompat.Config{
			Name:    "deepseek",
			BaseURL: baseURL,
			APIKey:  key,
			Quirks: openaicompat.Quirks{
				EchoReasoning: true,
			},
		}); err == nil {
			add("deepseek", p.Provider())
		} else {
			log.Printf("[providers] deepseek: %v", err)
		}
	}

	// Anthropic (first-party API) — via llmhub, opt-in by env key.
	if key := firstEnv("ULTIPROXY_ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY"); key != "" {
		if p, err := anthropichub.New(anthropichub.Config{APIKey: key}); err == nil {
			add("anthropic", p.Provider())
		} else {
			log.Printf("[providers] anthropic: %v", err)
		}
	}

	// OpenRouter (metered fallback aggregator).
	if key := firstEnv("OPENROUTER_API_KEY", "ULTIPROXY_OPENROUTER_API_KEY"); key != "" {
		baseURL := firstEnv("OPENROUTER_BASE_URL", "ULTIPROXY_OPENROUTER_BASE_URL")
		if baseURL == "" {
			baseURL = "https://openrouter.ai/api/v1"
		}
		if p, err := openaicompat.New(openaicompat.Config{
			Name:    "openrouter",
			BaseURL: baseURL,
			APIKey:  key,
		}); err == nil {
			add("openrouter", p.Provider())
		} else {
			log.Printf("[providers] openrouter: %v", err)
		}
	}

	// Local vLLM — ULTIPROXY_VLLM_BASE_URL, optional key.
	if base := firstEnv("ULTIPROXY_VLLM_BASE_URL", "VLLM_BASE_URL"); base != "" {
		key := firstEnv("ULTIPROXY_VLLM_API_KEY", "VLLM_API_KEY")
		if p, err := openaicompat.New(openaicompat.Config{
			Name:    "vllm",
			BaseURL: base,
			APIKey:  key,
			Quirks: openaicompat.Quirks{
				ModelListPassthrough: true,
			},
		}); err == nil {
			add("vllm", p.Provider())
		} else {
			log.Printf("[providers] vllm: %v", err)
		}
	}

	// OpenCode Go — env only (ultiproxy self-contained): standard Bearer API
	// key (opencode.ai/auth); no external CLI config reads.
	ocKey := firstEnv("OPENCODE_API_KEY", "ULTIPROXY_OPENCODE_API_KEY")
	if ocKey != "" {
		if p, err := openaicompat.New(openaicompat.Config{
			Name:    "opencode",
			BaseURL: "https://opencode.ai/zen/go/v1",
			APIKey:  ocKey,
		}); err == nil {
			add("opencode", p.Provider())
		} else {
			log.Printf("[providers] opencode: %v", err)
		}
	}

	// Augure AI — ultiproxy-owned token file (stateDir/augure_token) or env only.
	// No ~/.augure reads; login writes the token into ultiproxy state.
	augTok := firstEnv("ULTIPROXY_AUGURE_TOKEN", "AUGURE_TOKEN")
	augTokenFile := filepath.Join(stateDir, "augure_token")
	if augTok == "" {
		if data, err := os.ReadFile(augTokenFile); err == nil {
			augTok = strings.TrimSpace(string(data))
		}
	}
	if augTok != "" {
		_ = os.MkdirAll(stateDir, 0755)
		_ = os.WriteFile(augTokenFile, []byte(augTok+"\n"), 0600)
		if p, err := openaicompat.New(openaicompat.Config{
			Name:      "augure",
			BaseURL:   "https://api.augureai.ca/v1",
			APIKey:    augTok,
			TokenFile: augTokenFile,
			DataDir:   stateDir,
			Quirks: openaicompat.Quirks{
				AuthViaSupabaseRefresh: true,
				DefaultModel:           "tofino-3",
			},
		}); err == nil {
			add("augure", p.Provider())
		} else {
			log.Printf("[providers] augure: %v", err)
		}
	}

	// xAI Grok — ultiproxy-owned credentials first; env, then ~/.grok (xAI CLI).
	{
		credDir := filepath.Join(stateDir, "credentials", "xai")
		mgr, mgrErr := newOAuthManager(credDir)
		if mgrErr == nil && managerHasToken(mgr, xaiDefaultClientID) {
			if p, err := openaicompat.New(openaicompat.Config{
				Name:    "xai",
				BaseURL: xaiDefaultBaseURL,
				DataDir: credDir,
				Quirks: openaicompat.Quirks{
					AuthViaOAuthManager:  true,
					CreditsQuotaObserver: xaiDefaultBillingURL,
				},
			}); err == nil {
				add("xai", p.Provider())
			} else {
				log.Printf("[providers] xai: %v", err)
			}
		} else if tok := firstEnv("ULTIPROXY_XAI_TOKEN"); tok != "" {
			if p, err := openaicompat.New(openaicompat.Config{
				Name:    "xai",
				BaseURL: xaiDefaultBaseURL,
				APIKey:  tok,
				Quirks: openaicompat.Quirks{
					CreditsQuotaObserver: xaiDefaultBillingURL,
				},
			}); err == nil {
				add("xai", p.Provider())
			} else {
				log.Printf("[providers] xai: %v", err)
			}
		} else if mgrErr == nil && managerHasToken(mgr, xaiDefaultClientID) {
			if p, err := openaicompat.New(openaicompat.Config{
				Name:    "xai",
				BaseURL: xaiDefaultBaseURL,
				DataDir: credDir,
				Quirks: openaicompat.Quirks{
					AuthViaOAuthManager:  true,
					CreditsQuotaObserver: xaiDefaultBillingURL,
				},
			}); err == nil {
				add("xai", p.Provider())
			} else {
				log.Printf("[providers] xai: %v", err)
			}
		}
	}

	// Codex — ultiproxy-owned credentials or env token only (no ~/.codex reads).
	{
		credDir := filepath.Join(stateDir, "credentials", "codex")
		mgr, mgrErr := newOAuthManager(credDir)
		if mgrErr == nil && managerHasToken(mgr, codex.DefaultClientID) {
			p := codex.New(codex.Config{AuthManager: mgr, ClientID: codex.DefaultClientID})
			add("codex", p.ProviderBundle())
		} else if tok := firstEnv("ULTIPROXY_CODEX_TOKEN"); tok != "" {
			p := codex.New(codex.Config{StaticToken: tok})
			add("codex", p.ProviderBundle())
		} else if mgrErr == nil && managerHasToken(mgr, codex.DefaultClientID) {
			p := codex.New(codex.Config{AuthManager: mgr, ClientID: codex.DefaultClientID})
			add("codex", p.ProviderBundle())
		}
	}

	// Antigravity — ultiproxy-owned OAuth only; registers only when a real token
	// exists (a fresh install registers zero lanes).
	if p := antigravity.NewFromState(home, stateDir, nil); p != nil && p.HasToken() {
		add("antigravity", p.ProviderBundle())
	}

	// Copilot — env token only (ultiproxy self-contained; no gh CLI shell-out).
	copTok := firstEnv("ULTIPROXY_COPILOT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN")
	if copTok != "" {
		p := copilot.New(copilot.Config{Token: copTok})
		add("copilot", p.ProviderBundle())
	}

	// Freebuff: env or ultiproxy-owned state token only (self-contained).
	fbTok := freebuffToken(stateDir)
	if fbTok != "" {
		if err := os.MkdirAll(stateDir, 0755); err != nil {
			log.Printf("[providers] freebuff: %v", err)
		} else {
			fbActor := newFreebuffActor(stateDir, "")
			if fbActor == nil {
				log.Printf("[providers] freebuff: actor unavailable")
			} else {
				if p, err := openaicompat.New(openaicompat.Config{
					Name:    "freebuff",
					BaseURL: "https://www.codebuff.com/api/v1",
					APIKey:  fbTok,
					DataDir: stateDir,
					Quirks: openaicompat.Quirks{
						FreebuffActor:       fbActor,
						FreebuffDefaultTool: true,
					},
				}); err == nil {
					add("freebuff", p.Provider())
				} else {
					log.Printf("[providers] freebuff: %v", err)
				}
			}
		}
	}
}

// freebuffToken discovers the Codebuff/Freebuff token from ultiproxy-owned
// sources only: env, then the state-dir token file. Never reads an external
// CLI store.
func freebuffToken(stateDir string) string {
	if tok := firstEnv("ULTIPROXY_FREEBUFF_TOKEN", "FREEBUFF_TOKEN"); tok != "" {
		return tok
	}
	if stateDir != "" {
		if data, err := os.ReadFile(filepath.Join(stateDir, "freebuff_token")); err == nil {
			if tok := strings.TrimSpace(string(data)); tok != "" {
				return tok
			}
		}
	}
	return ""
}

// newFreebuffActor builds the serialized-request actor for a freebuff lane,
// reusing the persisted instance id (if any). An explicit token (the api_key
// handed to add_provider, or a lane's stored key) wins; without one the token
// is discovered from the ultiproxy-owned sources (env, then the state-dir
// token file). It returns nil when no token is available at all. Used by the
// compile-time lane above, by the runtime provider store hook (providers.json
// quirks.freebuff_actor=true) and by runtimeLaneBuilder for kind=freebuff.
func newFreebuffActor(stateDir, token string) any {
	fbTok := strings.TrimSpace(token)
	if fbTok == "" {
		fbTok = freebuffToken(stateDir)
	}
	if fbTok == "" {
		return nil
	}
	persistFreebuffToken(stateDir, fbTok)
	instanceID := ""
	if data, err := os.ReadFile(filepath.Join(stateDir, "freebuff_instance_id")); err == nil {
		instanceID = strings.TrimSpace(string(data))
	}
	if strings.HasPrefix(instanceID, "fb-inst-") {
		instanceID = ""
	}
	fbActor, err := spikesfreebuff.NewFreebuffAccountActor(
		"",
		http.DefaultClient,
		fbTok,
		spikesfreebuff.WithBaseURL("https://www.codebuff.com/api/v1"),
		spikesfreebuff.WithInstanceID(instanceID),
	)
	if err != nil {
		log.Printf("[providers] freebuff actor: %v", err)
		return nil
	}
	return &freebuffActorAdapter{actor: fbActor}
}

// persistFreebuffToken records an explicitly supplied freebuff token in the
// ultiproxy-owned state dir so a runtime lane keeps working after a restart
// even if its key is not re-supplied. An existing token file is never
// overwritten (rotation goes through login or a new add_provider token).
func persistFreebuffToken(stateDir, token string) {
	if token == "" || stateDir == "" {
		return
	}
	tokenFile := filepath.Join(stateDir, "freebuff_token")
	if _, err := os.Stat(tokenFile); err == nil {
		return
	}
	if err := os.MkdirAll(stateDir, 0o755); err != nil {
		log.Printf("[providers] freebuff token persist: %v", err)
		return
	}
	if err := os.WriteFile(tokenFile, []byte(token+"\n"), 0o600); err != nil {
		log.Printf("[providers] freebuff token persist: %v", err)
	}
}

// runtimeFreebuffActorBuilder adapts newFreebuffActor to the runtime provider
// store hook: the lane's own key (cfg.APIKey, e.g. an add_provider api_key)
// wins, the lane's own DataDir comes next, otherwise fall back to the daemon
// state dir.
func runtimeFreebuffActorBuilder(fallbackStateDir string) func(openaicompat.Config) any {
	return func(cfg openaicompat.Config) any {
		dir := cfg.DataDir
		if dir == "" {
			dir = fallbackStateDir
		}
		return newFreebuffActor(dir, cfg.APIKey)
	}
}
