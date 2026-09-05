package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/storage"
	"gopkg.in/yaml.v3"
)

// statsServerFixture is a Server wired to a throwaway SQLite telemetry store and
// a configured client key, so the stats endpoints can be exercised behind real
// auth middleware.
type statsServerFixture struct {
	srv    *Server
	writer *storage.Writer
}

func newStatsServer(t *testing.T) *statsServerFixture {
	t.Helper()

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "telemetry.db")
	writer, err := storage.NewWriter(dbPath)
	if err != nil {
		t.Fatalf("storage.NewWriter: %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })

	cfg := DefaultConfig()
	cfg.DataDir = dir
	cfg.Storage.DBPath = dbPath
	cfg.Server.ClientKeys = map[string]string{"alice": "alice-key"}

	registry := provider.NewRegistry()
	return &statsServerFixture{
		srv:    NewServer(cfg, registry, WithStorageWriter(writer)),
		writer: writer,
	}
}

// seedStatsRow inserts one terminal request row plus its usage row, stamped now.
func (f *statsServerFixture) seedStatsRow(t *testing.T, clientKeyHash, prov string, prompt, completion, cached int64, cost float64, errorClass string) {
	t.Helper()
	f.seedStatsRowAt(t, clientKeyHash, prov, prompt, completion, cached, cost, errorClass, time.Now().UTC())
}

// seedStatsRowAt inserts one terminal request row plus its usage row at an
// explicit timestamp, so window filtering can be exercised deterministically.
func (f *statsServerFixture) seedStatsRowAt(t *testing.T, clientKeyHash, prov string, prompt, completion, cached int64, cost float64, errorClass string, createdAt time.Time) {
	t.Helper()

	res, err := f.writer.DB().Exec(`
INSERT INTO requests (client_key_hash, logical_id, requested_model, resolved_model, provider, created_at, completed_at, finish_reason, stream_complete, error_class, ttft_ms, total_ms)
VALUES (?, 'log-1', 'm', 'm', ?, ?, ?, 'stop', 1, ?, 10, 20)`,
		clientKeyHash, prov, createdAt.Format(time.RFC3339), createdAt.Format(time.RFC3339), errorClass)
	if err != nil {
		t.Fatalf("insert request row: %v", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		t.Fatalf("request row id: %v", err)
	}
	if _, err := f.writer.DB().Exec(`
INSERT INTO usage (request_id, prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, cost)
VALUES (?, ?, ?, 0, ?, ?)`, id, prompt, completion, cached, cost); err != nil {
		t.Fatalf("insert usage row: %v", err)
	}
}

func (f *statsServerFixture) get(t *testing.T, path, key string) (*httptest.ResponseRecorder, map[string]json.RawMessage) {
	t.Helper()

	req := httptest.NewRequest(http.MethodGet, path, nil)
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	rec := httptest.NewRecorder()
	f.srv.Handler().ServeHTTP(rec, req)

	var body map[string]json.RawMessage
	if len(rec.Body.Bytes()) > 0 && rec.Body.Bytes()[0] == '{' {
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("decode %s body %q: %v", path, rec.Body.String(), err)
		}
	}
	return rec, body
}

// requiredFields loads the `required` list of one schema from
// docs/openapi.yaml, so the assertions below can never drift away from the
// published contract.
func requiredFields(t *testing.T, schema string) []string {
	t.Helper()

	spec := loadOpenAPISpec(t)
	entry, ok := spec.Components.Schemas[schema]
	if !ok {
		t.Fatalf("docs/openapi.yaml has no components/schemas/%s", schema)
	}
	if len(entry.Required) == 0 {
		t.Fatalf("components/schemas/%s declares no required fields", schema)
	}
	return entry.Required
}

// AC3: GET /api/stats/summary answers with the fields the published OpenAPI
// schema advertises, populated from the telemetry store.
func TestStatsSummary_MatchesPublishedSchema(t *testing.T) {
	f := newStatsServer(t)

	aliceHash := keyHashHex("alice-key")
	bobHash := keyHashHex("bob-key")

	f.seedStatsRow(t, aliceHash, "copilot", 100, 40, 7, 0.01, "")
	f.seedStatsRow(t, aliceHash, "codex", 10, 5, 0, 0.02, "")
	f.seedStatsRow(t, bobHash, "copilot", 1, 2, 0, 0.03, "generate_error")

	statsSummaryRequiredFields := requiredFields(t, "StatsSummaryResponse")

	rec, body := f.get(t, "/api/stats/summary", "alice-key")
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/stats/summary = %d: %s", rec.Code, rec.Body.String())
	}
	for _, field := range statsSummaryRequiredFields {
		if _, ok := body[field]; !ok {
			t.Errorf("response is missing the advertised required field %q: %s", field, rec.Body.String())
		}
	}

	var summary struct {
		TotalRequests         int64   `json:"total_requests"`
		TotalPromptTokens     int64   `json:"total_prompt_tokens"`
		TotalCompletionTokens int64   `json:"total_completion_tokens"`
		TotalCachedTokens     int64   `json:"total_cached_tokens"`
		ActiveClients         int64   `json:"active_clients"`
		EstimatedCostSaved    float64 `json:"estimated_cost_saved_usd"`
		ProviderBreakdown     map[string]struct {
			Requests int64 `json:"requests"`
			Tokens   int64 `json:"tokens"`
			Errors   int64 `json:"errors"`
		} `json:"provider_breakdown"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &summary); err != nil {
		t.Fatalf("decode stats summary against the published schema: %v (%s)", err, rec.Body.String())
	}

	if summary.TotalRequests != 3 {
		t.Errorf("total_requests = %d, want 3", summary.TotalRequests)
	}
	if summary.TotalPromptTokens != 111 {
		t.Errorf("total_prompt_tokens = %d, want 111", summary.TotalPromptTokens)
	}
	if summary.TotalCompletionTokens != 47 {
		t.Errorf("total_completion_tokens = %d, want 47", summary.TotalCompletionTokens)
	}
	if summary.TotalCachedTokens != 7 {
		t.Errorf("total_cached_tokens = %d, want 7", summary.TotalCachedTokens)
	}
	if summary.ActiveClients != 2 {
		t.Errorf("active_clients = %d, want 2 (alice and bob)", summary.ActiveClients)
	}
	if diff := summary.EstimatedCostSaved - 0.06; diff < -1e-12 || diff > 1e-12 {
		t.Errorf("estimated_cost_saved_usd = %v, want 0.06", summary.EstimatedCostSaved)
	}

	copilot, ok := summary.ProviderBreakdown["copilot"]
	if !ok {
		t.Fatalf("provider_breakdown has no copilot entry: %+v", summary.ProviderBreakdown)
	}
	if copilot.Requests != 2 || copilot.Tokens != 143 {
		t.Errorf("copilot breakdown = %+v, want 2 requests / 143 tokens", copilot)
	}
	if copilot.Errors != 1 {
		t.Errorf("copilot breakdown errors = %d, want 1", copilot.Errors)
	}
	codex, ok := summary.ProviderBreakdown["codex"]
	if !ok || codex.Requests != 1 || codex.Tokens != 15 || codex.Errors != 0 {
		t.Errorf("codex breakdown = %+v, want 1 request / 15 tokens / 0 errors", codex)
	}
}

// An empty telemetry store still answers with every advertised field, so the
// contract holds from the first request onwards.
func TestStatsSummary_EmptyStoreStillAdvertisesFields(t *testing.T) {
	f := newStatsServer(t)

	rec, body := f.get(t, "/api/stats/summary", "alice-key")
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/stats/summary = %d: %s", rec.Code, rec.Body.String())
	}
	for _, field := range requiredFields(t, "StatsSummaryResponse") {
		if _, ok := body[field]; !ok {
			t.Errorf("empty store response is missing the advertised required field %q: %s", field, rec.Body.String())
		}
	}
}

// A daemon built without storage (writer nil) must still answer the contract.
func TestStatsSummary_NoStorageStillAdvertisesFields(t *testing.T) {
	statsSummaryRequiredFields := requiredFields(t, "StatsSummaryResponse")

	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.Server.ClientKeys = map[string]string{"alice": "alice-key"}
	srv := NewServer(cfg, provider.NewRegistry())

	req := httptest.NewRequest(http.MethodGet, "/api/stats/summary", nil)
	req.Header.Set("Authorization", "Bearer alice-key")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/stats/summary = %d: %s", rec.Code, rec.Body.String())
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	for _, field := range statsSummaryRequiredFields {
		if _, ok := body[field]; !ok {
			t.Errorf("no-storage response is missing the advertised required field %q: %s", field, rec.Body.String())
		}
	}
}

// AC4: /api/stats/by-client is documented, so it must be mounted and answer with
// the published ClientUsageRecord fields.
func TestStatsByClient_MountedAndMatchesPublishedSchema(t *testing.T) {
	f := newStatsServer(t)

	aliceHash := keyHashHex("alice-key")
	f.seedStatsRow(t, aliceHash, "copilot", 100, 40, 7, 0.01, "")
	f.seedStatsRow(t, aliceHash, "codex", 10, 5, 0, 0.02, "rate_limited")
	f.seedStatsRow(t, keyHashHex("bob-key"), "copilot", 1, 2, 0, 0.03, "")

	clientUsageRecordRequiredFields := requiredFields(t, "ClientUsageRecord")

	rec, _ := f.get(t, "/api/stats/by-client", "alice-key")
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/stats/by-client = %d: %s", rec.Code, rec.Body.String())
	}

	var records []map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &records); err != nil {
		t.Fatalf("decode by-client array: %v (%s)", err, rec.Body.String())
	}
	if len(records) != 2 {
		t.Fatalf("by-client returned %d records, want 2: %s", len(records), rec.Body.String())
	}
	for i, record := range records {
		for _, field := range clientUsageRecordRequiredFields {
			if _, ok := record[field]; !ok {
				t.Errorf("by-client record %d is missing the advertised required field %q: %v", i, field, record)
			}
		}
	}

	var typed []struct {
		ClientID         string `json:"client_id"`
		ClientName       string `json:"client_name"`
		RequestCount     int64  `json:"request_count"`
		PromptTokens     int64  `json:"prompt_tokens"`
		CompletionTokens int64  `json:"completion_tokens"`
		CachedTokens     int64  `json:"cached_tokens"`
		ErrorCount       int64  `json:"error_count"`
		RateLimitedCount int64  `json:"rate_limited_count"`
		LastActive       string `json:"last_active"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &typed); err != nil {
		t.Fatalf("decode by-client records: %v", err)
	}
	// Sorted by request count descending: alice first.
	if typed[0].ClientID != "alice" || typed[0].ClientName != "alice" {
		t.Errorf("first record client = %q/%q, want alice/alice", typed[0].ClientID, typed[0].ClientName)
	}
	if typed[0].RequestCount != 2 || typed[0].PromptTokens != 110 || typed[0].CompletionTokens != 45 {
		t.Errorf("alice record = %+v, want 2 requests / 110 prompt / 45 completion", typed[0])
	}
	if typed[0].CachedTokens != 7 {
		t.Errorf("alice cached tokens = %d, want 7", typed[0].CachedTokens)
	}
	if typed[0].ErrorCount != 1 || typed[0].RateLimitedCount != 1 {
		t.Errorf("alice error counts = %d/%d, want 1 error / 1 rate limited", typed[0].ErrorCount, typed[0].RateLimitedCount)
	}
	if typed[0].LastActive == "" {
		t.Errorf("alice last_active is empty")
	}
	// An unknown key hash is reported by its digest rather than a configured name.
	if typed[1].ClientID != keyHashHex("bob-key") {
		t.Errorf("unknown client id = %q, want the key digest", typed[1].ClientID)
	}
}

// The documented window query parameter is honoured (not silently ignored).
func TestStatsByClient_WindowParameterAccepted(t *testing.T) {
	f := newStatsServer(t)
	aliceHash := keyHashHex("alice-key")

	f.seedStatsRow(t, aliceHash, "copilot", 100, 40, 0, 0.01, "")
	f.seedStatsRowAt(t, aliceHash, "copilot", 1, 1, 0, 0.001, "", time.Now().UTC().Add(-30*24*time.Hour))

	rec, _ := f.get(t, "/api/stats/by-client?window=7d", "alice-key")
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/stats/by-client?window=7d = %d: %s", rec.Code, rec.Body.String())
	}
	var records []map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &records); err != nil {
		t.Fatalf("decode by-client array: %v (%s)", err, rec.Body.String())
	}
	if len(records) != 1 {
		t.Errorf("window=7d returned %d records, want 1 (the 30-day-old row excluded): %s", len(records), rec.Body.String())
	}

	// An unknown window value is a 400, not a silently ignored default.
	rec, _ = f.get(t, "/api/stats/by-client?window=not-a-window", "alice-key")
	if rec.Code != http.StatusBadRequest {
		t.Errorf("GET /api/stats/by-client?window=not-a-window = %d, want 400", rec.Code)
	}
}

// AC4: every /api/ path documented in docs/openapi.yaml must be served by the
// mux, and every mounted /api/ stats route must be documented.
func TestOpenAPI_StatsRoutesMatchMux(t *testing.T) {
	spec := loadOpenAPISpec(t)
	f := newStatsServer(t)

	for path := range spec.Paths {
		if !strings.HasPrefix(path, "/api/") {
			continue
		}
		rec, _ := f.get(t, path, "alice-key")
		if rec.Code == http.StatusNotFound {
			t.Errorf("docs/openapi.yaml documents %s but the mux does not serve it", path)
		}
	}

	if _, ok := spec.Paths["/api/stats/by-client"]; !ok {
		t.Error("docs/openapi.yaml does not document /api/stats/by-client although the mux serves it")
	}
}

// AC5: /api/quota requires the client key whenever keys are configured, so the
// published security requirement must not claim the route is public.
func TestOpenAPI_QuotaSecurityMatchesMiddleware(t *testing.T) {
	spec := loadOpenAPISpec(t)
	entry, ok := spec.Paths["/api/quota"]
	if !ok {
		t.Fatal("docs/openapi.yaml does not document /api/quota")
	}
	if entry.Get == nil {
		t.Fatal("docs/openapi.yaml does not document GET /api/quota")
	}
	if entry.Get.isPublic() {
		t.Error("docs/openapi.yaml marks GET /api/quota as `security: []` (public) but the auth middleware protects it")
	}

	f := newStatsServer(t)
	rec, _ := f.get(t, "/api/quota", "")
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("GET /api/quota without a key = %d, want 401 (auth-required)", rec.Code)
	}
	rec, _ = f.get(t, "/api/quota", "alice-key")
	if rec.Code != http.StatusOK {
		t.Errorf("GET /api/quota with a client key = %d, want 200: %s", rec.Code, rec.Body.String())
	}
}

// --- OpenAPI spec loader -------------------------------------------------

type openAPIPathItem struct {
	Get *openAPIOperation `yaml:"get"`
}

type openAPIOperation struct {
	// Security is nil when the operation omits the key (it then inherits the
	// document-level requirement) and a non-nil empty slice for an explicit
	// `security: []`, which opts the route out of authentication.
	Security *[]map[string][]string `yaml:"security"`
}

// isPublic reports whether the operation explicitly opts out of authentication.
// OpenAPI spells that `security: []`: an empty list of requirements, which decodes
// to a non-nil empty slice here. A missing key or a non-empty list (named
// schemes such as bearerAuth) means the route requires authentication.
func (o *openAPIOperation) isPublic() bool {
	if o == nil || o.Security == nil {
		return false
	}
	return len(*o.Security) == 0
}

type openAPISpec struct {
	Paths      map[string]openAPIPathItem `yaml:"paths"`
	Components openAPIComponents          `yaml:"components"`
}

type openAPIComponents struct {
	Schemas map[string]openAPISchema `yaml:"schemas"`
}

type openAPISchema struct {
	Required []string `yaml:"required"`
}

func loadOpenAPISpec(t *testing.T) *openAPISpec {
	t.Helper()

	data, err := os.ReadFile(filepath.Join("..", "..", "docs", "openapi.yaml"))
	if err != nil {
		t.Fatalf("read docs/openapi.yaml: %v", err)
	}
	var spec openAPISpec
	if err := yaml.Unmarshal(data, &spec); err != nil {
		t.Fatalf("parse docs/openapi.yaml: %v", err)
	}
	if len(spec.Paths) == 0 {
		t.Fatal("docs/openapi.yaml parsed to zero paths")
	}
	return &spec
}
