package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// newAccountingServer builds a Server whose telemetry goes to a throwaway
// SQLite database, so tests can read the recorded rows back.
func newAccountingServer(t *testing.T, prov *fakeInferenceProvider, models map[string]ModelAlias) (*Server, *storage.Writer) {
	t.Helper()

	dir := t.TempDir()
	writer, err := storage.NewWriter(filepath.Join(dir, "telemetry.db"), storage.WithBatchSize(1))
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })

	cfg := DefaultConfig()
	cfg.DataDir = dir
	cfg.Storage.DBPath = filepath.Join(dir, "telemetry.db")
	cfg.Server.Models = models

	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: prov})
	return NewServer(cfg, registry, WithStorageWriter(writer)), writer
}

// waitForTerminalRequest polls until query returns a row (telemetry is flushed
// asynchronously, one item per transaction in these tests).
func waitForTerminalRequest(t *testing.T, w *storage.Writer, query string, dest ...any) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		err := w.DB().QueryRow(query, dest...).Scan(dest...)
		if err == nil {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for telemetry row (%s): %v", query, err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// recordedRequestUsage reads the single request/usage pair a dispatch produced,
// asserting they are associated through request_id.
func recordedRequestUsage(t *testing.T, w *storage.Writer) (reqID int64, prompt, completion, cached int64, cost float64) {
	t.Helper()

	// Wait for the terminal request row (error_class is empty once the outcome
	// is written) and for its usage row to land.
	deadline := time.Now().Add(3 * time.Second)
	for {
		rows, err := w.DB().Query(`
SELECT r.id, COALESCE(u.prompt_tokens, 0), COALESCE(u.completion_tokens, 0), COALESCE(u.cached_tokens, 0), COALESCE(u.cost, 0.0)
FROM requests r
LEFT JOIN usage u ON r.id = u.request_id
WHERE r.error_class NOT IN ('in_flight')`)
		if err != nil {
			t.Fatalf("query requests/usage: %v", err)
		}
		type pair struct {
			id                         int64
			prompt, completion, cached int64
			cost                       float64
		}
		var found []pair
		for rows.Next() {
			var p pair
			if err := rows.Scan(&p.id, &p.prompt, &p.completion, &p.cached, &p.cost); err != nil {
				t.Fatalf("scan: %v", err)
			}
			found = append(found, p)
		}
		rows.Close()
		if len(found) == 1 && (found[0].prompt > 0 || found[0].completion > 0) {
			return found[0].id, found[0].prompt, found[0].completion, found[0].cached, found[0].cost
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for a terminal request row joined to usage (got %d rows)", len(found))
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// 1. A non-streaming dispatch writes RequestRecord and UsageRecord that share
// the request id, and the attempt row points at the same request.
func TestAccounting_NonStreaming_JoinsRequestAndUsage(t *testing.T) {
	prov := &fakeInferenceProvider{
		name: "prov",
		generateFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
			return &ir.Response{
				ID:           "resp-1",
				FinishReason: "stop",
				Usage: &ir.Usage{
					PromptTokens:             100,
					CompletionTokens:         40,
					TotalTokens:              140,
					ReasoningTokens:          12,
					CacheReadInputTokens:     7,
					CacheCreationInputTokens: 3,
				},
			}, nil
		},
	}

	srv, writer := newAccountingServer(t, prov, nil)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"prov/gpt-4o","messages":[]}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rec.Code, rec.Body.String())
	}

	reqID, prompt, completion, cached, _ := recordedRequestUsage(t, writer)
	if prompt != 100 || completion != 40 {
		t.Errorf("usage tokens mismatch: prompt=%d completion=%d, want 100/40", prompt, completion)
	}
	if cached != 10 {
		t.Errorf("cached tokens = %d, want 10 (cache read 7 + cache creation 3)", cached)
	}

	var attempts int64
	waitForTerminalRequest(t, writer, `SELECT COUNT(*) FROM request_attempts WHERE request_id = `+int64String(reqID), &attempts)
	if attempts != 1 {
		t.Errorf("request_attempts rows linked to request %d = %d, want 1", reqID, attempts)
	}
}

// 2. A streaming dispatch writes RequestRecord and UsageRecord that share the
// request id; several cumulative usage events collapse into one usage row.
func TestAccounting_Streaming_JoinsRequestAndUsage(t *testing.T) {
	prov := &fakeInferenceProvider{
		name: "prov",
		streamFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
			ch := make(chan ir.Event, 6)
			ch <- ir.EventMessageStart{ID: "msg-1"}
			ch <- ir.EventTextDelta{Index: 0, Text: "Hello"}
			ch <- ir.EventUsageUpdate{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}
			ch <- ir.EventUsageUpdate{PromptTokens: 30, CompletionTokens: 8, TotalTokens: 38}
			ch <- ir.EventUsageUpdate{PromptTokens: 60, CompletionTokens: 20, TotalTokens: 80, Cost: 0.0042}
			ch <- ir.EventMessageStop{FinishReason: "stop"}
			close(ch)
			return ch, nil
		},
	}

	srv, writer := newAccountingServer(t, prov, nil)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"prov/gpt-4o","messages":[],"stream":true}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rec.Code, rec.Body.String())
	}

	reqID, prompt, completion, _, cost := recordedRequestUsage(t, writer)
	if prompt != 60 || completion != 20 {
		t.Errorf("usage tokens mismatch: prompt=%d completion=%d, want the last cumulative event 60/20", prompt, completion)
	}
	if cost != 0.0042 {
		t.Errorf("usage cost = %v, want the upstream-reported 0.0042", cost)
	}

	var usageRows int64
	waitForTerminalRequest(t, writer, `SELECT COUNT(*) FROM usage WHERE request_id = `+int64String(reqID), &usageRows)
	if usageRows != 1 {
		t.Errorf("usage rows for request %d = %d, want 1 (usage events are cumulative)", reqID, usageRows)
	}
}

// 3. An alias priced with input_cost/output_cost carries provider.WithCost down
// to the provider and prices the recorded usage when the upstream reports none.
func TestAccounting_AliasCostRatesPriceUsage(t *testing.T) {
	prov := &fakeInferenceProvider{
		name: "prov",
		generateFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
			return &ir.Response{
				ID:           "resp-1",
				FinishReason: "stop",
				Usage:        &ir.Usage{PromptTokens: 1000, CompletionTokens: 2000, TotalTokens: 3000},
			}, nil
		},
	}

	srv, writer := newAccountingServer(t, prov, map[string]ModelAlias{
		"priced": {Provider: "prov", Upstream: "upstream-model", InputCost: 2, OutputCost: 5},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"priced","messages":[]}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rec.Code, rec.Body.String())
	}

	cfg := provider.NewRequestConfig(prov.capturedOpts...)
	if cfg.InputCostPerMillion != 2 || cfg.OutputCostPerMillion != 5 {
		t.Errorf("provider received cost rates input=%v output=%v, want 2/5", cfg.InputCostPerMillion, cfg.OutputCostPerMillion)
	}
	if cfg.Model != "upstream-model" {
		t.Errorf("provider received model %q, want the alias upstream id", cfg.Model)
	}

	// 1000 prompt tokens at $2/1M + 2000 completion tokens at $5/1M = $0.012
	_, prompt, completion, _, cost := recordedRequestUsage(t, writer)
	if prompt != 1000 || completion != 2000 {
		t.Fatalf("usage tokens mismatch: prompt=%d completion=%d", prompt, completion)
	}
	if diff := cost - 0.012; diff < -1e-12 || diff > 1e-12 {
		t.Errorf("usage cost = %v, want alias-priced 0.012", cost)
	}
}

// 3b. A cost reported by the upstream always wins over the alias rates.
func TestAccounting_UpstreamCostWinsOverAliasRates(t *testing.T) {
	prov := &fakeInferenceProvider{
		name: "prov",
		generateFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
			return &ir.Response{
				ID:           "resp-1",
				FinishReason: "stop",
				Usage:        &ir.Usage{PromptTokens: 1000, CompletionTokens: 2000, TotalTokens: 3000, Cost: 0.03},
			}, nil
		},
	}

	srv, writer := newAccountingServer(t, prov, map[string]ModelAlias{
		"priced": {Provider: "prov", Upstream: "upstream-model", InputCost: 2, OutputCost: 5},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"priced","messages":[]}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rec.Code, rec.Body.String())
	}

	_, _, _, _, cost := recordedRequestUsage(t, writer)
	if cost != 0.03 {
		t.Errorf("usage cost = %v, want the upstream-reported 0.03", cost)
	}
}

// 4. /api/stats/summary reports the recorded requests, tokens and cost.
func TestAccounting_StatsSummaryAggregates(t *testing.T) {
	prov := &fakeInferenceProvider{
		name: "prov",
		generateFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
			return &ir.Response{
				ID:           "resp-1",
				FinishReason: "stop",
				Usage:        &ir.Usage{PromptTokens: 100, CompletionTokens: 40, TotalTokens: 140},
			}, nil
		},
	}
	srv, writer := newAccountingServer(t, prov, map[string]ModelAlias{
		"priced": {Provider: "prov", Upstream: "upstream-model", InputCost: 1, OutputCost: 1},
	})

	do := func() {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"priced","messages":[]}`))
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200 OK, got %d: %s", rec.Code, rec.Body.String())
		}
	}
	do()
	do()

	waitForRows(t, writer, 2)

	// The daily summary joins usage back to requests through request_id.
	summary, err := storage.ListUsageSummary(context.Background(), writer.DB())
	if err != nil {
		t.Fatalf("ListUsageSummary: %v", err)
	}
	if len(summary) != 1 {
		t.Fatalf("ListUsageSummary rows = %d, want 1 (one client, one day): %+v", len(summary), summary)
	}
	if summary[0].Requests != 2 {
		t.Errorf("summary requests = %d, want 2", summary[0].Requests)
	}
	if summary[0].PromptTokens != 200 || summary[0].CompletionTokens != 80 {
		t.Errorf("summary tokens = %d/%d, want 200/80", summary[0].PromptTokens, summary[0].CompletionTokens)
	}
	if summary[0].TotalTokens != 280 {
		t.Errorf("summary total tokens = %d, want 280", summary[0].TotalTokens)
	}
	// 2 x (100 prompt + 40 completion) tokens at $1/1M = $0.00028
	if diff := summary[0].Cost - 0.00028; diff < -1e-12 || diff > 1e-12 {
		t.Errorf("summary cost = %v, want 0.00028", summary[0].Cost)
	}

	byClient, err := storage.ListUsageByClient(context.Background(), writer.DB(), "7d")
	if err != nil {
		t.Fatalf("ListUsageByClient: %v", err)
	}
	if len(byClient) != 1 || byClient[0].Requests != 2 || byClient[0].TotalTokens != 280 {
		t.Fatalf("ListUsageByClient = %+v, want one client with 2 requests / 280 tokens", byClient)
	}

	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/stats/summary", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/stats/summary = %d: %s", rec.Code, rec.Body.String())
	}
	var stats struct {
		TotalRequests int64   `json:"total_requests"`
		TotalTokens   int64   `json:"total_tokens"`
		TotalCost     float64 `json:"total_cost"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &stats); err != nil {
		t.Fatalf("decode stats summary: %v (%s)", err, rec.Body.String())
	}
	if stats.TotalRequests != 2 {
		t.Errorf("stats total_requests = %d, want 2", stats.TotalRequests)
	}
	if stats.TotalTokens != 280 {
		t.Errorf("stats total_tokens = %d, want 280", stats.TotalTokens)
	}
	if diff := stats.TotalCost - 0.00028; diff < -1e-12 || diff > 1e-12 {
		t.Errorf("stats total_cost = %v, want 0.00028", stats.TotalCost)
	}
}

// 5. A dispatch where every provider fails before commit still closes its
// request row, so the recorded attempts are not orphaned.
func TestAccounting_FailedDispatchClosesRequestRow(t *testing.T) {
	prov := &fakeInferenceProvider{
		name: "prov",
		generateFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
			return nil, errors.New("upstream exploded with status 500")
		},
	}
	srv, writer := newAccountingServer(t, prov, nil)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"prov/gpt-4o","messages":[]}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code == http.StatusOK {
		t.Fatalf("expected a failure, got 200")
	}

	var reqID int64
	var errorClass string
	waitForTerminalRequest(t, writer, `SELECT id, error_class FROM requests WHERE error_class <> 'in_flight'`, &reqID, &errorClass)
	if errorClass == "" {
		t.Errorf("request row error_class = %q, want the terminal failure class", errorClass)
	}
	var attempts int64
	waitForTerminalRequest(t, writer, `SELECT COUNT(*) FROM request_attempts WHERE request_id = `+int64String(reqID), &attempts)
	if attempts == 0 {
		t.Errorf("no attempt rows linked to request %d", reqID)
	}
}

// 6. Request ids are unique and positive so telemetry rows never collide.
func TestAccounting_RequestIDsUniqueAndPositive(t *testing.T) {
	prov := &fakeInferenceProvider{name: "prov"}
	srv, _ := newAccountingServer(t, prov, nil)

	seen := make(map[int64]bool, 100)
	for i := 0; i < 100; i++ {
		id := srv.nextRequestID()
		if id <= 0 {
			t.Fatalf("request id %d is not positive", id)
		}
		if seen[id] {
			t.Fatalf("duplicate request id %d", id)
		}
		seen[id] = true
	}
}

// waitForRows blocks until the telemetry writer has flushed n requests.
func waitForRows(t *testing.T, w *storage.Writer, n int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		var got int64
		if err := w.DB().QueryRow("SELECT COUNT(*) FROM requests WHERE error_class <> 'in_flight'").Scan(&got); err != nil {
			t.Fatalf("count requests: %v", err)
		}
		if got >= int64(n) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %d flushed requests (got %d)", n, got)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// int64String renders a request id for inline SQL in tests.
func int64String(v int64) string {
	return strings.TrimSpace(strings.Join([]string{jsonInt(v)}, ""))
}

func jsonInt(v int64) string {
	b, _ := json.Marshal(v)
	return string(b)
}
