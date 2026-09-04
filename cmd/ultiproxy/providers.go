package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/antigravity"
	"github.com/smhanov/ultiproxy/pkg/provider/augure"
	"github.com/smhanov/ultiproxy/pkg/provider/codex"
	"github.com/smhanov/ultiproxy/pkg/provider/copilot"
	"github.com/smhanov/ultiproxy/pkg/provider/deepseek"
	"github.com/smhanov/ultiproxy/pkg/provider/freebuff"
	"github.com/smhanov/ultiproxy/pkg/provider/opencode"
	"github.com/smhanov/ultiproxy/pkg/provider/openrouter"
	"github.com/smhanov/ultiproxy/pkg/provider/vllm"
	"github.com/smhanov/ultiproxy/pkg/provider/xai"
	"github.com/smhanov/ultiproxy/pkg/provider/zai"
)

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

// readNestedToken walks a JSON file: try the exact field path first; if that
// fails, return the "access_token" field of the first dict value that has one.
// This handles files where credentials are nested under an opaque scope key
// (e.g. ~/.grok/auth.json keyed by OAuth scope).
func readNestedToken(path string, fields ...string) (string, bool) {
	if tok, ok := readJSONField(path, fields...); ok {
		return tok, true
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return "", false
	}
	for _, v := range m {
		if entry, ok := v.(map[string]any); ok {
			if tok, ok := entry["access_token"].(string); ok && tok != "" {
				return tok, true
			}
			if tok, ok := entry["key"].(string); ok && tok != "" {
				return tok, true
			}
		}
	}
	if tok, ok := m["access_token"].(string); ok && tok != "" {
		return tok, true
	}
	return "", false
}

// execGhAuthToken shells out to `gh auth token` (best effort, 8s timeout).
func execGhAuthToken() (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "gh", "auth", "token").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// envFile reads a KEY=value pair from one or more dotenv files.
func envFile(envFile, key string) string {
	data, err := os.ReadFile(envFile)
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, key+"=") {
			return strings.Trim(strings.TrimPrefix(line, key+"="), `"'`)
		}
	}
	return ""
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
// credential stores. Antigravity never reads ~/.cli-proxy-api.
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
		if p, err := zai.New(zai.Config{APIKey: zaiKey}); err == nil {
			add("zai", p.Provider())
		} else {
			log.Printf("[providers] zai: %v", err)
		}
	}

	// DeepSeek (metered API).
	if key := firstEnv("DEEPSEEK_API_KEY", "ULTIPROXY_DEEPSEEK_API_KEY"); key != "" {
		if p, err := deepseek.New(deepseek.Config{APIKey: key}); err == nil {
			add("deepseek", p.Provider())
		} else {
			log.Printf("[providers] deepseek: %v", err)
		}
	}

	// OpenRouter (metered fallback aggregator).
	if key := firstEnv("OPENROUTER_API_KEY", "ULTIPROXY_OPENROUTER_API_KEY"); key != "" {
		if p, err := openrouter.New(openrouter.Config{APIKey: key}); err == nil {
			add("openrouter", p.Provider())
		} else {
			log.Printf("[providers] openrouter: %v", err)
		}
	}

	// Local vLLM (megos-style) — ULTIPROXY_VLLM_BASE_URL, optional key.
	if base := firstEnv("ULTIPROXY_VLLM_BASE_URL", "VLLM_BASE_URL"); base != "" {
		key := firstEnv("ULTIPROXY_VLLM_API_KEY", "VLLM_API_KEY")
		if p, err := vllm.New(vllm.Config{BaseURL: base, APIKey: key}); err == nil {
			add("vllm", p.Provider())
		} else {
			log.Printf("[providers] vllm: %v", err)
		}
	}

	// OpenCode Go — workspace id + session cookie (from dashboard .env) + API key (from opencode.json).
	ocKey := firstEnv("OPENCODE_API_KEY", "ULTIPROXY_OPENCODE_API_KEY")
	if ocKey == "" {
		if k, ok := readNestedToken(filepath.Join(home, ".config", "opencode", "opencode.json"), "provider", "opencode-go", "options", "apiKey"); ok {
			ocKey = k
		}
	}
	if ocKey == "" {
		// check old backup files if opencode.json was overwritten
		matches, _ := filepath.Glob(filepath.Join(home, ".config", "opencode", "opencode.json.bak-*"))
		for _, m := range matches {
			if k, ok := readNestedToken(m, "provider", "opencode-go", "options", "apiKey"); ok && k != "" {
				ocKey = k
				break
			}
		}
	}

	ocWorkspace := firstEnv("OPENCODE_WORKSPACE_ID", "ULTIPROXY_OPENCODE_WORKSPACE")
	ocCookie := firstEnv("OPENCODE_SESSION_COOKIE", "ULTIPROXY_OPENCODE_COOKIE")
	if ocWorkspace == "" {
		ocWorkspace = envFile(filepath.Join(home, "ai-quota-dashboard", ".env"), "OPENCODE_WORKSPACE_ID")
		ocCookie = envFile(filepath.Join(home, "ai-quota-dashboard", ".env"), "OPENCODE_SESSION_COOKIE")
	}
	if ocKey != "" || (ocWorkspace != "" && ocCookie != "") {
		if p, err := opencode.New(opencode.Config{
			APIKey:        ocKey,
			WorkspaceID:   ocWorkspace,
			SessionCookie: ocCookie,
		}); err == nil {
			add("opencode", p.Provider())
		} else {
			log.Printf("[providers] opencode: %v", err)
		}
	}

	// Augure AI — Supabase OAuth tokens in ~/.augure.
	authFile := filepath.Join(home, ".augure", "augure-auth.json")
	if _, err := os.Stat(authFile); err == nil {
		if p, err := augure.New(augure.Config{TokenFile: authFile}); err == nil {
			add("augure", p.Provider())
		} else {
			log.Printf("[providers] augure: %v", err)
		}
	}

	// xAI Grok — ultiproxy-owned credentials first; env, then ~/.grok (xAI CLI).
	{
		credDir := filepath.Join(stateDir, "credentials", "xai")
		mgr, mgrErr := newOAuthManager(credDir)
		if mgrErr == nil && managerHasToken(mgr, xai.DefaultClientID) {
			p := xai.New(xai.Config{AuthManager: mgr, ClientID: xai.DefaultClientID})
			add("xai", p.ProviderBundle())
		} else if tok := firstEnv("ULTIPROXY_XAI_TOKEN"); tok != "" {
			p := xai.New(xai.Config{StaticToken: tok})
			add("xai", p.ProviderBundle())
		} else if tok, ok := readNestedToken(filepath.Join(home, ".grok", "auth.json")); ok {
			p := xai.New(xai.Config{StaticToken: tok})
			add("xai", p.ProviderBundle())
		} else if mgrErr == nil {
			p := xai.New(xai.Config{AuthManager: mgr, ClientID: xai.DefaultClientID})
			add("xai", p.ProviderBundle())
		}
	}

	// Codex — ultiproxy-owned credentials first; env, then ~/.codex (Codex CLI).
	{
		credDir := filepath.Join(stateDir, "credentials", "codex")
		mgr, mgrErr := newOAuthManager(credDir)
		if mgrErr == nil && managerHasToken(mgr, codex.DefaultClientID) {
			p := codex.New(codex.Config{AuthManager: mgr, ClientID: codex.DefaultClientID})
			add("codex", p.ProviderBundle())
		} else if tok := firstEnv("ULTIPROXY_CODEX_TOKEN"); tok != "" {
			p := codex.New(codex.Config{StaticToken: tok})
			add("codex", p.ProviderBundle())
		} else if tok, ok := readJSONField(filepath.Join(home, ".codex", "auth.json"), "tokens", "access_token"); ok {
			p := codex.New(codex.Config{StaticToken: tok})
			add("codex", p.ProviderBundle())
		} else if tok, ok := readJSONField(filepath.Join(home, ".codex", "auth.json"), "access_token"); ok {
			p := codex.New(codex.Config{StaticToken: tok})
			add("codex", p.ProviderBundle())
		} else if mgrErr == nil {
			p := codex.New(codex.Config{AuthManager: mgr, ClientID: codex.DefaultClientID})
			add("codex", p.ProviderBundle())
		}
	}

	// Antigravity — ultiproxy-owned OAuth only. Never read ~/.cli-proxy-api.
	if p := antigravity.NewFromState(home, stateDir, nil); p != nil {
		add("antigravity", p.ProviderBundle())
	}

	// Copilot — env, gh auth token, or gh CLI output.
	copTok := firstEnv("ULTIPROXY_COPILOT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN")
	if copTok == "" {
		if out, err := execGhAuthToken(); err == nil && out != "" {
			copTok = out
		}
	}
	if copTok != "" {
		p := copilot.New(copilot.Config{Token: copTok})
		add("copilot", p.ProviderBundle())
	}

	// Freebuff — CLI credentials (~/.config/manicode/credentials.json), then env.
	// Never reads ~/workspace/freebuff-proxy/.env.
	fbTok := firstEnv("ULTIPROXY_FREEBUFF_TOKEN", "FREEBUFF_TOKEN")
	if fbTok == "" {
		if tok, _, _, err := freebuff.ReadCLIToken(); err == nil {
			fbTok = tok
		}
	}
	if fbTok != "" {
		if p, err := freebuff.New(freebuff.Config{Token: fbTok, DataDir: stateDir}); err == nil {
			add("freebuff", p.Provider())
		} else {
			log.Printf("[providers] freebuff: %v", err)
		}
	}
}
