package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider/antigravity"
	"github.com/smhanov/ultiproxy/pkg/provider/codex"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	spikesfreebuff "github.com/smhanov/ultiproxy/pkg/spikes/freebuff"
)

func runLogin(providerName, dataDir string) {
	providerName = strings.ToLower(strings.TrimSpace(providerName))
	if providerName == "" {
		providerName = "antigravity"
	}
	home, _ := os.UserHomeDir()
	if dataDir == "" {
		dataDir = firstEnv("ULTIPROXY_DATA_DIR", "ULTIPROXY_STATE_DIR")
	}
	if dataDir == "" {
		dataDir = filepath.Join(home, ".local", "state", "ultiproxy")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	switch providerName {
	case "antigravity", "gemini", "google":
		p := antigravity.NewFromState(home, dataDir, nil)
		if urlFile := os.Getenv("ULTIPROXY_OAUTH_URL_FILE"); urlFile != "" {
			codeFile := os.Getenv("ULTIPROXY_OAUTH_CODE_FILE")
			if codeFile == "" {
				log.Fatal("ULTIPROXY_OAUTH_CODE_FILE required when ULTIPROXY_OAUTH_URL_FILE is set")
			}
			mgr, err := newOAuthManager(filepath.Join(dataDir, "credentials", "antigravity"))
			if err != nil {
				log.Fatalf("antigravity credential store: %v", err)
			}
			p = antigravity.New(antigravity.Config{
				AuthManager: mgr,
				OnAuthURL: func(u string) {
					_ = os.WriteFile(urlFile, []byte(u+"\n"), 0o600)
				},
				ReadCode: func() (string, error) {
					deadline := time.Now().Add(8 * time.Minute)
					for time.Now().Before(deadline) {
						b, err := os.ReadFile(codeFile)
						if err == nil {
							s := strings.TrimSpace(string(b))
							if s != "" {
								return s, nil
							}
						}
						select {
						case <-ctx.Done():
							return "", ctx.Err()
						case <-time.After(400 * time.Millisecond):
						}
					}
					return "", fmt.Errorf("timed out waiting for authorization code in %s", codeFile)
				},
			})
		}
		if p == nil {
			log.Fatalf("antigravity: could not create credential store under %s", dataDir)
		}
		fmt.Fprintln(os.Stderr, "Ultiproxy Antigravity login (does not use cliproxy).")
		if err := p.Login(ctx); err != nil {
			log.Fatalf("antigravity login failed: %v", err)
		}
		fmt.Fprintf(os.Stderr, "Logged in. project=%s credentials=%s\n", p.ProjectID(), filepath.Join(dataDir, "credentials", "antigravity"))
	case "xai", "grok":
		p, err := openaicompat.New(openaicompat.Config{
			Name:    "xai",
			BaseURL: "https://api.x.ai",
			DataDir: filepath.Join(dataDir, "credentials", "xai"),
			Quirks: openaicompat.Quirks{
				AuthViaOAuthManager: true,
			},
		})
		if err != nil {
			log.Fatalf("xai: %v", err)
		}
		fmt.Fprintln(os.Stderr, "Ultiproxy xAI device login.")
		if err := p.Login(ctx); err != nil {
			log.Fatalf("xai login failed: %v", err)
		}
		fmt.Fprintln(os.Stderr, "Logged in to xAI.")
	case "codex", "openai":
		mgr, err := newOAuthManager(filepath.Join(dataDir, "credentials", "codex"))
		if err != nil {
			log.Fatalf("codex credential store: %v", err)
		}
		p := codex.New(codex.Config{AuthManager: mgr, ClientID: codex.DefaultClientID})
		fmt.Fprintln(os.Stderr, "Ultiproxy Codex device login.")
		if err := p.Login(ctx); err != nil {
			log.Fatalf("codex login failed: %v", err)
		}
		fmt.Fprintln(os.Stderr, "Logged in to Codex.")
	case "freebuff", "codebuff":
		instanceIDFile := filepath.Join(dataDir, "freebuff_instance_id")
		instanceID := ""
		if data, err := os.ReadFile(instanceIDFile); err == nil {
			instanceID = strings.TrimSpace(string(data))
		}
		if strings.HasPrefix(instanceID, "fb-inst-") {
			instanceID = ""
		}
		fbActor, err := spikesfreebuff.NewFreebuffAccountActor(
			"",
			http.DefaultClient,
			"",
			spikesfreebuff.WithBaseURL("https://www.codebuff.com/api/v1"),
			spikesfreebuff.WithInstanceID(instanceID),
		)
		if err != nil {
			log.Fatalf("freebuff: %v", err)
		}
		p, err := openaicompat.New(openaicompat.Config{
			Name:    "freebuff",
			BaseURL: "https://www.codebuff.com/api/v1",
			DataDir: dataDir,
			Quirks: openaicompat.Quirks{
				FreebuffActor:       &freebuffActorAdapter{actor: fbActor},
				FreebuffDefaultTool: true,
			},
		})
		if err != nil {
			log.Fatalf("freebuff: %v", err)
		}
		fmt.Fprintln(os.Stderr, "Ultiproxy Freebuff login (imports ~/.config/manicode/credentials.json).")
		if err := p.Login(ctx); err != nil {
			log.Fatalf("freebuff login failed: %v", err)
		}
		fmt.Fprintln(os.Stderr, "Imported Freebuff CLI token. Restart ultiproxy serve to use it.")
	default:
		fmt.Fprintf(os.Stderr, "unknown login provider %q (available: antigravity, xai, codex, freebuff)\n", providerName)
		os.Exit(1)
	}
}
