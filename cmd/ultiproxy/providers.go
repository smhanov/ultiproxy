package main

import (
	"log"
	"os"
	"path/filepath"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/antigravity"
	"github.com/smhanov/ultiproxy/pkg/provider/augure"
	"github.com/smhanov/ultiproxy/pkg/provider/codex"
	"github.com/smhanov/ultiproxy/pkg/provider/copilot"
	"github.com/smhanov/ultiproxy/pkg/provider/deepseek"
	"github.com/smhanov/ultiproxy/pkg/provider/freebuff"
	"github.com/smhanov/ultiproxy/pkg/provider/openrouter"
	"github.com/smhanov/ultiproxy/pkg/provider/vllm"
	"github.com/smhanov/ultiproxy/pkg/provider/xai"
	"github.com/smhanov/ultiproxy/pkg/provider/zai"
)

// registerProviders wires upstream adapters into the registry. Registration is
// opt-in per provider and driven by environment variables, so the binary
// boots cleanly in CI and on machines without any subscription credentials.
func registerProviders(registry *provider.Registry) {
	home, _ := os.UserHomeDir()

	add := func(name string, bundle provider.Provider) {
		registry.Register(bundle)
		log.Printf("[providers] registered %s", name)
	}

	// Z.ai Coding Plan (GLM) — zero marginal cost on subscription.
	if os.Getenv("ZAI_API_KEY") != "" {
		if p, err := zai.New(zai.Config{}); err == nil {
			add("zai", p.Provider())
		} else {
			log.Printf("[providers] zai: %v", err)
		}
	}

	// DeepSeek (metered API).
	if os.Getenv("DEEPSEEK_API_KEY") != "" {
		if p, err := deepseek.New(deepseek.Config{}); err == nil {
			add("deepseek", p.Provider())
		} else {
			log.Printf("[providers] deepseek: %v", err)
		}
	}

	// OpenRouter (metered fallback aggregator).
	if os.Getenv("OPENROUTER_API_KEY") != "" {
		if p, err := openrouter.New(openrouter.Config{}); err == nil {
			add("openrouter", p.Provider())
		} else {
			log.Printf("[providers] openrouter: %v", err)
		}
	}

	// Local vLLM (megos-style) — ULTIPROXY_VLLM_BASE_URL, optional key.
	if base := os.Getenv("ULTIPROXY_VLLM_BASE_URL"); base != "" {
		if p, err := vllm.New(vllm.Config{BaseURL: base, APIKey: os.Getenv("ULTIPROXY_VLLM_API_KEY")}); err == nil {
			add("vllm", p.Provider())
		} else {
			log.Printf("[providers] vllm: %v", err)
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

	// xAI Grok — static Bearer token (ULTIPROXY_XAI_TOKEN) or OAuth via auth manager.
	if tok := os.Getenv("ULTIPROXY_XAI_TOKEN"); tok != "" {
		p := xai.New(xai.Config{StaticToken: tok})
		add("xai", p.ProviderBundle())
	}

	// Codex — static Bearer token (ULTIPROXY_CODEX_TOKEN).
	if tok := os.Getenv("ULTIPROXY_CODEX_TOKEN"); tok != "" {
		p := codex.New(codex.Config{StaticToken: tok})
		add("codex", p.ProviderBundle())
	}

	// Antigravity — static Bearer token (ULTIPROXY_ANTIGRAVITY_TOKEN).
	if tok := os.Getenv("ULTIPROXY_ANTIGRAVITY_TOKEN"); tok != "" {
		p := antigravity.New(antigravity.Config{StaticToken: tok})
		add("antigravity", p.ProviderBundle())
	}

	// Copilot — static GitHub token (ULTIPROXY_COPILOT_TOKEN) or gh auth.
	if tok := os.Getenv("ULTIPROXY_COPILOT_TOKEN"); tok != "" {
		p := copilot.New(copilot.Config{Token: tok})
		add("copilot", p.ProviderBundle())
	}

	// Freebuff — static token + data dir for the account actor lock.
	if tok := os.Getenv("ULTIPROXY_FREEBUFF_TOKEN"); tok != "" {
		datadir := os.Getenv("ULTIPROXY_DATA_DIR")
		if datadir == "" {
			datadir = filepath.Join(home, ".local", "state", "ultiproxy")
		}
		if p, err := freebuff.New(freebuff.Config{Token: tok, DataDir: datadir}); err == nil {
			add("freebuff", p.Provider())
		} else {
			log.Printf("[providers] freebuff: %v", err)
		}
	}
}
