package opencode

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/server"
)

// fakeOAuthModelsUpstream speaks the xAI device flow plus an auth-gated
// OpenAI-compatible model list: /v1/models answers 401 unless the request
// carries the just-issued access token. That proves the rebuilt lane's
// TokenSource reads the just-stored credential (T004 AC1).
func fakeOAuthModelsUpstream(t *testing.T, accessToken string, models []string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/device", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"device_code": "dev-t004-123",
			"user_code": "T004-1234",
			"verification_uri": "https://accounts.x.ai/oauth2/device",
			"expires_in": 300,
			"interval": 0
		}`))
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"access_token": "` + accessToken + `",
			"refresh_token": "t004-refresh-tok",
			"expires_in": 3600
		}`))
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+accessToken {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": "unauthorized"}})
			return
		}
		data := make([]map[string]any, 0, len(models))
		for _, id := range models {
			data = append(data, map[string]any{"id": id})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	})
	// The chat wire is not exercised here, but a default keeps the mux honest.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// getModelsIDs fetches GET /v1/models from the downstream test server.
func getModelsIDs(t *testing.T, client *http.Client, url string) []string {
	t.Helper()
	resp, err := client.Get(url + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/models = %d: %s", resp.StatusCode, body)
	}
	var decoded struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode /v1/models %q: %v", body, err)
	}
	out := make([]string, 0, len(decoded.Data))
	for _, m := range decoded.Data {
		out = append(out, m.ID)
	}
	return out
}

// TestLoginRebuild_DeviceFlowLeavesLaneUsable (T004 AC1): the full device flow
// — StartLogin → CompleteLogin → RebuildLane — leaves the registry serving a
// lane whose TokenSource reads the just-stored credential, with list_models
// fresh without a restart.
func TestLoginRebuild_DeviceFlowLeavesLaneUsable(t *testing.T) {
	const accessToken = "t004-access-tok"
	fake := fakeOAuthModelsUpstream(t, accessToken, []string{"grok-4", "grok-4-fast"})

	dataDir := t.TempDir()
	credDir := filepath.Join(dataDir, "credentials", "xai")
	mgr, err := auth.NewManager(credDir, nil)
	if err != nil {
		t.Fatalf("auth.NewManager: %v", err)
	}
	store := server.NewRuntimeProviderStore(filepath.Join(dataDir, "providers.json"))
	store.DefaultDataDir = dataDir
	store.Creds = mgr
	if err := store.Add(openaicompat.Config{
		Name:          "xai",
		BaseURL:       fake.URL,
		HTTPClient:    fake.Client(),
		DeviceAuthURL: fake.URL + "/device",
		TokenURL:      fake.URL + "/token",
		Quirks:        openaicompat.Quirks{AuthViaOAuthManager: true},
	}); err != nil {
		t.Fatalf("Add xai lane: %v", err)
	}

	registry := provider.NewRegistry()
	cfg := server.DefaultConfig()
	cfg.DataDir = dataDir
	srv := server.NewServer(cfg, registry,
		server.WithRuntimeProviderStore(store),
		server.WithModelRefreshInterval(0),
		server.WithCredentialRefresh(0),
	)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Drive the device flow on the live lane (initiate → complete).
	lane, ok := registry.Get("xai")
	if !ok {
		t.Fatalf("xai not registered after Restore: %v", registry.Names())
	}
	interactive, ok := lane.Auth.(provider.InteractiveAuthProvider)
	if !ok {
		t.Fatalf("xai lane has no interactive auth surface: %+v", lane)
	}
	if _, err := interactive.StartLogin(context.Background()); err != nil {
		t.Fatalf("StartLogin: %v", err)
	}
	if err := interactive.CompleteLogin(context.Background(), ""); err != nil {
		t.Fatalf("CompleteLogin: %v", err)
	}
	if cred, ok := mgr.Peek("b1a00492-073a-47ea-816f-4c329264a828"); !ok || cred.AccessToken != accessToken {
		t.Fatalf("credential store Peek = %+v, %v; want the login token", cred, ok)
	}

	// Rebuild from the stored config with the fresh credential in place.
	discovered, err := store.RebuildLane(registry, "xai")
	if err != nil {
		t.Fatalf("RebuildLane: %v", err)
	}
	if discovered != 2 {
		t.Fatalf("discovered = %d, want 2", discovered)
	}

	// The rebuilt lane's TokenSource serves the just-stored credential.
	rebuilt, ok := registry.Get("xai")
	if !ok {
		t.Fatal("xai missing from registry after rebuild")
	}
	if rebuilt.Auth == nil {
		t.Fatal("rebuilt lane has no Auth surface (OAuth TokenSource missing)")
	}
	tok, err := rebuilt.Auth.Token(context.Background())
	if err != nil {
		t.Fatalf("rebuilt Token: %v", err)
	}
	if tok != accessToken {
		t.Fatalf("rebuilt Token = %q, want %q", tok, accessToken)
	}

	// And /v1/models reflects the fresh discovery without a restart.
	ids := getModelsIDs(t, ts.Client(), ts.URL)
	for _, want := range []string{"xai/grok-4", "xai/grok-4-fast"} {
		found := false
		for _, id := range ids {
			if id == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("/v1/models missing %q after rebuild: %v", want, ids)
		}
	}
	if len(ids) == 0 || !strings.HasPrefix(ids[0], "xai/") {
		t.Errorf("/v1/models after rebuild = %v, want xai/-prefixed ids", ids)
	}
}
