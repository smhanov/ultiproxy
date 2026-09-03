package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseCLIArgs(t *testing.T) {
	args := []string{"--url", "http://localhost:9999", "status", "--key=my-key"}
	cfg := parseCLIArgs(args)

	if cfg.URL != "http://localhost:9999" {
		t.Errorf("expected URL http://localhost:9999, got %s", cfg.URL)
	}
	if cfg.Key != "my-key" {
		t.Errorf("expected key my-key, got %s", cfg.Key)
	}
	if len(cfg.Args) != 1 || cfg.Args[0] != "status" {
		t.Errorf("expected args ['status'], got %v", cfg.Args)
	}
}

func TestUpctlCommands_MockServer(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret-token" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"gpt-4o"}]}`))
	})
	mux.HandleFunc("/api/quota", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"summary":{"total":1,"ok":1}}`))
	})
	mux.HandleFunc("/api/stats/summary", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"total_requests":100}`))
	})
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"login initiated"}]}}`))
	})

	ts := httptest.NewServer(mux)
	defer ts.Close()

	client := ts.Client()
	cfg := CLIConfig{
		URL: ts.URL,
		Key: "secret-token",
	}

	// 1. Test status
	reqStatus, _ := http.NewRequest(http.MethodGet, cfg.URL+"/healthz", nil)
	bodyStatus, err := doRequest(client, reqStatus, cfg.Key)
	if err != nil || !strings.Contains(string(bodyStatus), `"status":"ok"`) {
		t.Errorf("doRequest /healthz failed: %v, body: %s", err, string(bodyStatus))
	}

	// 2. Test models
	reqModels, _ := http.NewRequest(http.MethodGet, cfg.URL+"/v1/models", nil)
	bodyModels, err := doRequest(client, reqModels, cfg.Key)
	if err != nil || !strings.Contains(string(bodyModels), "gpt-4o") {
		t.Errorf("doRequest /v1/models failed: %v, body: %s", err, string(bodyModels))
	}

	// 3. Test quota
	reqQuota, _ := http.NewRequest(http.MethodGet, cfg.URL+"/api/quota", nil)
	bodyQuota, err := doRequest(client, reqQuota, cfg.Key)
	if err != nil || !strings.Contains(string(bodyQuota), `"total":1`) {
		t.Errorf("doRequest /api/quota failed: %v, body: %s", err, string(bodyQuota))
	}

	// 4. Test usage
	reqUsage, _ := http.NewRequest(http.MethodGet, cfg.URL+"/api/stats/summary", nil)
	bodyUsage, err := doRequest(client, reqUsage, cfg.Key)
	if err != nil || !strings.Contains(string(bodyUsage), "100") {
		t.Errorf("doRequest /api/stats/summary failed: %v, body: %s", err, string(bodyUsage))
	}

	// 5. Test login
	loginData, _ := json.Marshal(map[string]any{"jsonrpc": "2.0"})
	reqLogin, _ := http.NewRequest(http.MethodPost, cfg.URL+"/mcp", strings.NewReader(string(loginData)))
	bodyLogin, err := doRequest(client, reqLogin, cfg.Key)
	if err != nil || !strings.Contains(string(bodyLogin), "login initiated") {
		t.Errorf("doRequest /mcp failed: %v, body: %s", err, string(bodyLogin))
	}
}
