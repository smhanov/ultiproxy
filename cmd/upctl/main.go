package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type CLIConfig struct {
	URL  string
	Key  string
	Args []string
}

func parseCLIArgs(rawArgs []string) CLIConfig {
	cfg := CLIConfig{
		URL: os.Getenv("ULTIPROXY_URL"),
		Key: os.Getenv("ULTIPROXY_KEY"),
	}
	if cfg.URL == "" {
		cfg.URL = "http://127.0.0.1:8317"
	}

	for i := 0; i < len(rawArgs); i++ {
		arg := rawArgs[i]
		if arg == "--url" && i+1 < len(rawArgs) {
			cfg.URL = rawArgs[i+1]
			i++
		} else if strings.HasPrefix(arg, "--url=") {
			cfg.URL = strings.TrimPrefix(arg, "--url=")
		} else if arg == "--key" && i+1 < len(rawArgs) {
			cfg.Key = rawArgs[i+1]
			i++
		} else if strings.HasPrefix(arg, "--key=") {
			cfg.Key = strings.TrimPrefix(arg, "--key=")
		} else {
			cfg.Args = append(cfg.Args, arg)
		}
	}

	cfg.URL = strings.TrimRight(cfg.URL, "/")
	return cfg
}

func main() {
	cfg := parseCLIArgs(os.Args[1:])

	if len(cfg.Args) == 0 {
		printUsage()
		os.Exit(1)
	}

	client := &http.Client{Timeout: 10 * time.Second}
	cmd := cfg.Args[0]

	switch cmd {
	case "status":
		runStatus(client, cfg)
	case "models":
		runModels(client, cfg)
	case "quota":
		runQuota(client, cfg)
	case "usage":
		runUsage(client, cfg)
	case "login":
		if len(cfg.Args) < 2 {
			fmt.Fprintln(os.Stderr, "error: provider argument required for login (e.g. upctl login copilot)")
			os.Exit(1)
		}
		runLogin(client, cfg, cfg.Args[1])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %q\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`upctl - CLI client for Ultiproxy

Usage:
  upctl [flags] <command> [arguments]

Commands:
  status             Check health status of the server (GET /healthz)
  models             List available models (GET /v1/models)
  quota              Display upstream quota and rate limit status (GET /api/quota)
  usage              Display request and token usage statistics (GET /api/stats/summary)
  login <provider>   Initiate OAuth login flow for a provider (POST /mcp)

Flags:
  --url URL          Ultiproxy URL (default: $ULTIPROXY_URL or http://127.0.0.1:8317)
  --key KEY          API key for authentication (default: $ULTIPROXY_KEY)`)
}

func doRequest(client *http.Client, req *http.Request, key string) ([]byte, error) {
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("server error (HTTP %d): %s", resp.StatusCode, string(body))
	}

	return body, nil
}

func runStatus(client *http.Client, cfg CLIConfig) {
	req, err := http.NewRequest(http.MethodGet, cfg.URL+"/healthz", nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating request: %v\n", err)
		os.Exit(1)
	}

	body, err := doRequest(client, req, cfg.Key)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(string(body))
}

func runModels(client *http.Client, cfg CLIConfig) {
	req, err := http.NewRequest(http.MethodGet, cfg.URL+"/v1/models", nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating request: %v\n", err)
		os.Exit(1)
	}

	body, err := doRequest(client, req, cfg.Key)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	var formatted bytes.Buffer
	if err := json.Indent(&formatted, body, "", "  "); err == nil {
		fmt.Println(formatted.String())
	} else {
		fmt.Println(string(body))
	}
}

func runQuota(client *http.Client, cfg CLIConfig) {
	req, err := http.NewRequest(http.MethodGet, cfg.URL+"/api/quota", nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating request: %v\n", err)
		os.Exit(1)
	}

	body, err := doRequest(client, req, cfg.Key)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	var formatted bytes.Buffer
	if err := json.Indent(&formatted, body, "", "  "); err == nil {
		fmt.Println(formatted.String())
	} else {
		fmt.Println(string(body))
	}
}

func runUsage(client *http.Client, cfg CLIConfig) {
	req, err := http.NewRequest(http.MethodGet, cfg.URL+"/api/stats/summary", nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating request: %v\n", err)
		os.Exit(1)
	}

	body, err := doRequest(client, req, cfg.Key)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	var formatted bytes.Buffer
	if err := json.Indent(&formatted, body, "", "  "); err == nil {
		fmt.Println(formatted.String())
	} else {
		fmt.Println(string(body))
	}
}

func runLogin(client *http.Client, cfg CLIConfig, providerName string) {
	payload := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "initiate_oauth_login",
			"arguments": map[string]any{
				"provider": providerName,
			},
		},
	}

	data, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, cfg.URL+"/mcp", bytes.NewReader(data))
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating request: %v\n", err)
		os.Exit(1)
	}
	req.Header.Set("Content-Type", "application/json")

	body, err := doRequest(client, req, cfg.Key)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	var formatted bytes.Buffer
	if err := json.Indent(&formatted, body, "", "  "); err == nil {
		fmt.Println(formatted.String())
	} else {
		fmt.Println(string(body))
	}
}
