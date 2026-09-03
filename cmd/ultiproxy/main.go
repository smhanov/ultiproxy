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
	"github.com/smhanov/ultiproxy/pkg/server"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

const version = "0.1.0"

func main() {
	// Subcommand-first CLI: `ultiproxy serve --config ...`. Go's flag
	// package stops parsing at the first non-flag argument, so we extract
	// the subcommand before parsing flags.
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
		fmt.Fprintf(os.Stderr, "unknown command %q (available: serve, version)\n", cmd)
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

	srv := server.NewServer(cfg, registry,
		server.WithStateManager(stateManager),
		server.WithStorageWriter(writer),
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
