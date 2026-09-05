package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/anthropichub"
	"github.com/smhanov/ultiproxy/pkg/provider/antigravity"
	"github.com/smhanov/ultiproxy/pkg/provider/codex"
	"github.com/smhanov/ultiproxy/pkg/provider/copilot"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/server"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

const version = "0.1.0"

func main() {
	// The daemon serves by default (`ultiproxy --config ...` or the explicit
	// `ultiproxy serve --config ...`). There is no administration CLI:
	// every operator knob (logins, lanes, aliases, quotas) is an MCP tool at
	// /mcp. Go's flag package stops parsing at the first non-flag argument,
	// so we extract the subcommand before parsing flags.
	args := os.Args[1:]
	cmd := "serve"
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		cmd = args[0]
		args = args[1:]
	}

	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	configPath := fs.String("config", "", "Path to YAML configuration file")
	dataDir := fs.String("data-dir", "", "Data directory path")
	if err := fs.Parse(args); err != nil {
		log.Fatalf("flag parse error: %v", err)
	}

	switch cmd {
	case "version":
		fmt.Printf("ultiproxy v%s\n", version)
		os.Exit(0)

	case "serve":
		runServe(*configPath, *dataDir)

	default:
		fmt.Fprintf(os.Stderr, "unknown command %q (available: serve, version; administration goes through MCP tools at /mcp)\n", cmd)
		os.Exit(1)
	}
}

func runServe(configPath, dataDir string) {
	cfg, err := server.LoadConfig(configPath)
	if err != nil {
		log.Fatalf("failed to load configuration: %v", err)
	}

	if dataDir != "" {
		cfg.DataDir = dataDir
		if !filepath.IsAbs(cfg.Storage.DBPath) {
			cfg.Storage.DBPath = filepath.Join(dataDir, cfg.Storage.DBPath)
		}
	}

	// Initialize storage writer
	writer, err := storage.NewWriter(cfg.Storage.DBPath)
	if err != nil {
		log.Fatalf("failed to initialize SQLite telemetry storage: %v", err)
	}

	registry := provider.NewRegistry()
	stateManager := state.NewStateManager()

	registerProviders(registry)

	// Runtime-registered lanes (MCP add_provider) persist to
	// <data_dir>/providers.json and must be loaded BEFORE the router / model
	// catalog resolve provider names, so a restart from the same DataDir
	// restores them without any config file.
	providerStore := server.NewRuntimeProviderStore(filepath.Join(cfg.DataDir, "providers.json"))
	providerStore.DefaultDataDir = cfg.DataDir
	providerStore.ActorBuilder = runtimeFreebuffActorBuilder(cfg.DataDir)
	// Custom-wire lanes (antigravity, anthropic, codex) reconstruct from the
	// server's general DataDir: credential state lives at
	// <data_dir>/credentials/<lane> exactly like compile-time lanes, so a
	// runtime-registered lane behaves identically across restarts with no
	// per-lane config.
	providerStore.LaneBuilder = runtimeLaneBuilder
	providerStore.Restore(registry)

	srv := server.NewServer(cfg, registry,
		server.WithStateManager(stateManager),
		server.WithStorageWriter(writer),
		server.WithRuntimeProviderStore(providerStore),
	)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("Ultiproxy v%s starting on %s", version, cfg.Server.Addr)
		if err := srv.Start(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	<-stop
	log.Println("Shutting down Ultiproxy gracefully...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP shutdown error: %v", err)
	}

	if err := writer.Close(); err != nil {
		log.Printf("Storage close error: %v", err)
	}

	log.Println("Shutdown complete.")
}

// runtimeLaneBuilder reconstructs compile-time-wired lane kinds that are not
// OpenAI-compatible from their stored identity. It is the RuntimeProviderStore
// LaneBuilder used by MCP add_provider (custom kinds) and by Restore after a
// restart: name is the lane name, dataDir the server's general DataDir
// (credential state lives at <dataDir>/credentials/<kind>) and apiKey the
// persisted static key for kinds that need one ("" otherwise).
func runtimeLaneBuilder(name, kind, dataDir, apiKey string) (provider.Provider, error) {
	switch kind {
	case "antigravity":
		credDir := filepath.Join(dataDir, "credentials", "antigravity")
		if err := os.MkdirAll(credDir, 0o700); err != nil {
			return provider.Provider{}, fmt.Errorf("antigravity: create credential store: %w", err)
		}
		p := antigravity.NewFromState(dataDir, credDir, nil)
		if p == nil {
			return provider.Provider{}, fmt.Errorf("antigravity: could not create provider under %s", credDir)
		}
		return p.ProviderBundle(), nil
	case "anthropic":
		// Claude first-party API lane. The API key comes from the add_provider
		// call (not from a credential store), so it is persisted with the lane
		// and replayed here on restore. The lane is the same Anthropic-wire
		// adapter the compile-time registration uses, so /v1/models routing
		// and the codec path behave identically.
		p, err := anthropichub.New(anthropichub.Config{
			BaseURL: anthropichub.DefaultBaseURL,
			APIKey:  apiKey,
		})
		if err != nil {
			return provider.Provider{}, fmt.Errorf("anthropic: %w", err)
		}
		return p.Provider(), nil
	case "codex":
		// Codex OAuth device lane. Credentials live in the ultiproxy-owned
		// store at <dataDir>/credentials/codex, exactly like the compile-time
		// lane (see registerProviders in providers.go), so a runtime lane and
		// a compile-time lane share one token. The lane registers even before
		// login so quota stays readable. codex does NOT implement
		// provider.InteractiveAuthProvider (no StartLogin/CompleteLogin), so
		// do not build an interactive login flow here: login goes through
		// `ultiproxy login codex`, and MCP initiate_oauth_login falls back to
		// the legacy blocking Login() path, which cannot run over MCP.
		mgr, err := newOAuthManager(filepath.Join(dataDir, "credentials", "codex"))
		if err != nil {
			return provider.Provider{}, fmt.Errorf("codex: create credential store: %w", err)
		}
		p := codex.New(codex.Config{AuthManager: mgr, ClientID: codex.DefaultClientID})
		return p.ProviderBundle(), nil
	case "copilot":
		tok := apiKey
		if tok == "" {
			tok = firstEnv("ULTIPROXY_COPILOT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN")
		}
		p := copilot.New(copilot.Config{Token: tok})
		return p.ProviderBundle(), nil
	case "freebuff":
		// Freebuff (Codebuff) lane: the OpenAI-compatible wire at
		// https://www.codebuff.com/api/v1 plus the serialized-request actor
		// that owns the session lock, the default tool injection and the
		// usage quota. The key comes from the add_provider call (persisted
		// with the lane) or, failing that, from the ultiproxy-owned state
		// token; newFreebuffActor persists an explicit key for later runs.
		// The lane still builds without any token so it can register (and
		// report an honest quota) before a login/key exists.
		fbActor := newFreebuffActor(dataDir, apiKey)
		if fbActor == nil {
			log.Printf("[providers] freebuff: no token for runtime lane %q (set api_key or ULTIPROXY_FREEBUFF_TOKEN)", name)
		}
		p, err := openaicompat.New(openaicompat.Config{
			Name:    name,
			BaseURL: "https://www.codebuff.com/api/v1",
			APIKey:  apiKey,
			DataDir: dataDir,
			Quirks: openaicompat.Quirks{
				FreebuffActor:       fbActor,
				FreebuffDefaultTool: true,
			},
		})
		if err != nil {
			return provider.Provider{}, fmt.Errorf("freebuff: %w", err)
		}
		return p.Provider(), nil
	default:
		return provider.Provider{}, fmt.Errorf("unsupported runtime lane kind %q", kind)
	}
}
