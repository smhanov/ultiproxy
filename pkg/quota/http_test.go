package quota

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

func TestGoldenDashboardJSONContract(t *testing.T) {
	goldenPath := filepath.Join("..", "..", "testdata", "quota", "dashboard.json")
	goldenData, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("failed to read golden fixture: %v", err)
	}

	fixedTime := time.Unix(1788436800, 0).UTC() // 2026-09-03T12:00:00Z
	resetTime := time.Date(2026, 9, 3, 14, 0, 0, 0, time.UTC)
	updatedTime := time.Date(2026, 9, 3, 16, 0, 0, 0, time.UTC)

	reg := provider.NewRegistry()
	p := &mockQuotaProvider{name: "copilot"}
	reg.Register(provider.Provider{Quota: p})

	sm := state.NewStateManager(state.WithNow(func() time.Time { return fixedTime }))
	sm.SetProvider("copilot", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
		ObservedAt: updatedTime,
	})

	store := NewQuotaStore()
	store.Set("copilot", &provider.QuotaSnapshot{
		ObservedAt: updatedTime,
		Windows: []provider.QuotaWindow{
			{
				Label:            "Premium requests",
				UsedPct:          42.5,
				Remaining:        575,
				Limit:            1000,
				Unit:             "requests",
				ResetAt:          resetTime,
				SecondsRemaining: 3600,
			},
		},
		Detail: "Premium 57.5% remaining · resets 2026-09-03 · chat+completions included/unlimited",
	})
	store.SetLastFetchedAt(fixedTime)

	handler := NewHandler(HandlerConfig{
		StateManager: sm,
		Registry:     reg,
		Store:        store,
		NowFn:        func() time.Time { return fixedTime },
		Metadata: map[string]ProviderMetadata{
			"copilot": {
				Name: "GitHub Copilot",
				Plan: "Pro (annual)",
			},
		},
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/quota", nil)
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rec.Code)
	}

	var expected DashboardResponse
	if err := json.Unmarshal(goldenData, &expected); err != nil {
		t.Fatalf("failed to unmarshal golden fixture: %v", err)
	}

	var actual DashboardResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &actual); err != nil {
		t.Fatalf("failed to unmarshal actual response: %v\nBody: %s", err, rec.Body.String())
	}

	if !reflect.DeepEqual(actual, expected) {
		actJSON, _ := json.MarshalIndent(actual, "", "  ")
		expJSON, _ := json.MarshalIndent(expected, "", "  ")
		t.Fatalf("dashboard JSON mismatch!\nActual:\n%s\nExpected:\n%s", string(actJSON), string(expJSON))
	}
}

func TestDashboardTextAndMarkdown(t *testing.T) {
	fixedTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	reg := provider.NewRegistry()
	p := &mockQuotaProvider{name: "copilot"}
	reg.Register(provider.Provider{Quota: p})

	sm := state.NewStateManager(state.WithNow(func() time.Time { return fixedTime }))
	sm.SetProvider("copilot", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})

	store := NewQuotaStore()
	store.Set("copilot", &provider.QuotaSnapshot{
		ObservedAt: fixedTime,
		Windows: []provider.QuotaWindow{
			{
				Label:     "requests",
				UsedPct:   50.0,
				Remaining: 50,
				Limit:     100,
				Unit:      "requests",
			},
		},
		Detail: "half consumed",
	})
	store.SetLastFetchedAt(fixedTime)

	handler := NewHandler(HandlerConfig{
		StateManager: sm,
		Registry:     reg,
		Store:        store,
		NowFn:        func() time.Time { return fixedTime },
	})

	// /quota.txt
	recTxt := httptest.NewRecorder()
	reqTxt := httptest.NewRequest(http.MethodGet, "/quota.txt", nil)
	handler.ServeHTTP(recTxt, reqTxt)
	if recTxt.Code != http.StatusOK {
		t.Errorf("expected 200 for /quota.txt, got %d", recTxt.Code)
	}
	if recTxt.Header().Get("Content-Type") != "text/plain; charset=utf-8" {
		t.Errorf("expected text/plain for /quota.txt, got %s", recTxt.Header().Get("Content-Type"))
	}

	// /quota.md
	recMd := httptest.NewRecorder()
	reqMd := httptest.NewRequest(http.MethodGet, "/quota.md", nil)
	handler.ServeHTTP(recMd, reqMd)
	if recMd.Code != http.StatusOK {
		t.Errorf("expected 200 for /quota.md, got %d", recMd.Code)
	}
	if recMd.Header().Get("Content-Type") != "text/markdown; charset=utf-8" {
		t.Errorf("expected text/markdown for /quota.md, got %s", recMd.Header().Get("Content-Type"))
	}

	// /healthz
	recHz := httptest.NewRecorder()
	reqHz := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	handler.ServeHTTP(recHz, reqHz)
	if recHz.Code != http.StatusOK {
		t.Errorf("expected 200 for /healthz, got %d", recHz.Code)
	}

	// /llms.txt
	recLLMs := httptest.NewRecorder()
	reqLLMs := httptest.NewRequest(http.MethodGet, "/llms.txt", nil)
	handler.ServeHTTP(recLLMs, reqLLMs)
	if recLLMs.Code != http.StatusOK {
		t.Errorf("expected 200 for /llms.txt, got %d", recLLMs.Code)
	}
}

func TestDashboardStatsAPI(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "stats_test.db")
	w, err := storage.NewWriter(dbPath, storage.WithBatchSize(1))
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer w.Close()

	now := time.Now().UTC()
	_ = w.TrackRequest(storage.RequestRecord{
		ID:            10,
		ClientKeyHash: "client-key-1",
		CreatedAt:     now.Format(time.RFC3339),
	})
	_ = w.TrackUsage(storage.UsageRecord{
		RequestID:        10,
		PromptTokens:     100,
		CompletionTokens: 50,
		Cost:             0.015,
	})

	time.Sleep(50 * time.Millisecond)

	handler := NewHandler(HandlerConfig{
		Storage: w,
	})

	// /api/stats/summary
	recSum := httptest.NewRecorder()
	reqSum := httptest.NewRequest(http.MethodGet, "/api/stats/summary", nil)
	handler.ServeHTTP(recSum, reqSum)
	if recSum.Code != http.StatusOK {
		t.Fatalf("expected 200 for /api/stats/summary, got %d", recSum.Code)
	}

	var sumResp map[string]any
	if err := json.Unmarshal(recSum.Body.Bytes(), &sumResp); err != nil {
		t.Fatalf("unmarshal summary: %v", err)
	}
	if _, ok := sumResp["summary"]; !ok {
		t.Errorf("missing 'summary' key in stats summary")
	}

	// /api/stats/by-client?window=7d
	recClient := httptest.NewRecorder()
	reqClient := httptest.NewRequest(http.MethodGet, "/api/stats/by-client?window=7d", nil)
	handler.ServeHTTP(recClient, reqClient)
	if recClient.Code != http.StatusOK {
		t.Fatalf("expected 200 for /api/stats/by-client, got %d", recClient.Code)
	}

	var clientResp map[string]any
	if err := json.Unmarshal(recClient.Body.Bytes(), &clientResp); err != nil {
		t.Fatalf("unmarshal by-client: %v", err)
	}
	if clientResp["window"] != "7d" {
		t.Errorf("expected window=7d, got %v", clientResp["window"])
	}
}
