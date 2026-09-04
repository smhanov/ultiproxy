package opencode

import (
	"context"
	"encoding/json"
	"net/http"
	"path/filepath"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/server"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// newAccountingHarness wires the contract harness (real openaicompat lane, real
// SQLite telemetry) around one priced model alias.
func newAccountingHarness(t *testing.T) (*Harness, *storage.Writer) {
	t.Helper()

	dir := t.TempDir()
	writer, err := storage.NewWriter(filepath.Join(dir, "telemetry.db"), storage.WithBatchSize(1))
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })

	h := NewTestHarness(t,
		WithDataDir(dir),
		WithModelAlias("priced", server.ModelAlias{
			Provider:   "opencode",
			Upstream:   "deepseek-v4-flash",
			InputCost:  3,
			OutputCost: 6,
		}),
		WithServerOptions(server.WithStorageWriter(writer)),
	)
	return h, writer
}

// requestUsage waits for the terminal request row and returns its joined usage.
func requestUsage(t *testing.T, w *storage.Writer) (prompt, completion int64, cost float64) {
	t.Helper()

	deadline := time.Now().Add(3 * time.Second)
	for {
		row := w.DB().QueryRow(`
SELECT COALESCE(u.prompt_tokens, 0), COALESCE(u.completion_tokens, 0), COALESCE(u.cost, 0.0)
FROM requests r
LEFT JOIN usage u ON r.id = u.request_id
WHERE r.error_class <> 'in_flight' AND r.requested_model = 'priced'`)
		var p, c int64
		var cost float64
		err := row.Scan(&p, &c, &cost)
		if err == nil {
			return p, c, cost
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for the priced request's usage row: %v", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A non-streaming request through the real OpenAI-compatible lane records a
// request row and a usage row that share the request id, priced from the alias.
func TestAccounting_NonStreamingThroughOpenAICompatLane(t *testing.T) {
	h, writer := newAccountingHarness(t)
	h.FakeUpstream.QueueChatCompletion("hello there")

	resp, _, err := h.PostChat(context.Background(), map[string]any{
		"model": "priced",
		"messages": []map[string]any{
			{"role": "user", "content": "hi"},
		},
	})
	if err != nil {
		t.Fatalf("PostChat: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	prompt, completion, cost := requestUsage(t, writer)
	if prompt != 10 || completion != 10 {
		t.Fatalf("usage tokens = %d/%d, want the upstream's 10/10", prompt, completion)
	}
	// 10 prompt tokens at $3/1M + 10 completion tokens at $6/1M = $0.00009
	if diff := cost - 0.00009; diff < -1e-12 || diff > 1e-12 {
		t.Errorf("usage cost = %v, want alias-priced 0.00009", cost)
	}
	assertStatsSummary(t, h, 1, 20, 0.00009)
}

// A streaming request with a usage event records the same request/usage pair.
func TestAccounting_StreamingThroughOpenAICompatLane(t *testing.T) {
	h, writer := newAccountingHarness(t)
	h.FakeUpstream.QueueSSEStream(
		NewSSEStream().
			TextDelta("hel").
			TextDelta("lo").
			Usage(40, 60, 100).
			FinishReason("stop"),
	)

	obs, resp, err := h.StreamChat(context.Background(), map[string]any{
		"model": "priced",
		"messages": []map[string]any{
			{"role": "user", "content": "hi"},
		},
	})
	if err != nil {
		t.Fatalf("StreamChat: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if obs.FinishReason != "stop" {
		t.Errorf("finish reason = %q, want stop", obs.FinishReason)
	}

	prompt, completion, cost := requestUsage(t, writer)
	if prompt != 40 || completion != 60 {
		t.Fatalf("usage tokens = %d/%d, want the stream's 40/60", prompt, completion)
	}
	// 40 prompt tokens at $3/1M + 60 completion tokens at $6/1M = $0.00048
	if diff := cost - 0.00048; diff < -1e-12 || diff > 1e-12 {
		t.Errorf("usage cost = %v, want alias-priced 0.00048", cost)
	}
	assertStatsSummary(t, h, 1, 100, 0.00048)
}

// assertStatsSummary checks the public /api/stats/summary aggregation.
func assertStatsSummary(t *testing.T, h *Harness, wantRequests, wantTokens int64, wantCost float64) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	var body []byte
	deadline := time.Now().Add(3 * time.Second)
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, h.URL()+"/api/stats/summary", nil)
		if err != nil {
			t.Fatalf("new request: %v", err)
		}
		resp, err := h.Client().Do(req)
		if err != nil {
			t.Fatalf("GET /api/stats/summary: %v", err)
		}
		buf := make([]byte, 0, 512)
		chunk := make([]byte, 256)
		for {
			n, rerr := resp.Body.Read(chunk)
			buf = append(buf, chunk[:n]...)
			if rerr != nil {
				break
			}
		}
		resp.Body.Close()
		body = buf

		var stats struct {
			TotalRequests int64   `json:"total_requests"`
			TotalTokens   int64   `json:"total_tokens"`
			TotalCost     float64 `json:"total_cost"`
		}
		if json.Unmarshal(body, &stats) == nil && stats.TotalTokens == wantTokens {
			if stats.TotalRequests != wantRequests {
				t.Errorf("stats total_requests = %d, want %d", stats.TotalRequests, wantRequests)
			}
			if diff := stats.TotalCost - wantCost; diff < -1e-12 || diff > 1e-12 {
				t.Errorf("stats total_cost = %v, want %v", stats.TotalCost, wantCost)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("/api/stats/summary never reported %d tokens: %s", wantTokens, string(body))
		}
		time.Sleep(10 * time.Millisecond)
	}
}
