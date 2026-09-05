package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/storage"
)

// storageUsageSource is the test stand-in for the adapter production builds in
// pkg/server: it maps an MCP client_id onto the key hash the telemetry store
// records (here verbatim) and returns the storage totals.
type storageUsageSource struct {
	w *storage.Writer
}

func (s storageUsageSource) GetClientUsage(ctx context.Context, clientID, window string) (any, error) {
	totals, err := s.w.GetClientUsage(ctx, clientID, window)
	if err != nil {
		return nil, err
	}
	totals.ClientID = clientID
	return totals, nil
}

func callToolText(t *testing.T, s *Server, body string) (string, bool) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK from /mcp, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v\n%s", err, rec.Body.String())
	}
	if resp.Error != nil {
		t.Fatalf("JSON-RPC error: %s", resp.Error.Message)
	}
	if len(resp.Result.Content) == 0 {
		t.Fatalf("expected content in result: %s", rec.Body.String())
	}
	return resp.Result.Content[0].Text, resp.Result.IsError
}

// TestGetClientUsageUsesStorageSource proves the MCP layer renders whatever the
// injected UsageSource returns: a SQLite-backed source yields real totals.
func TestGetClientUsageUsesStorageSource(t *testing.T) {
	dir := t.TempDir()
	writer, err := storage.NewWriter(filepath.Join(dir, "telemetry.db"), storage.WithBatchSize(1))
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer writer.Close()

	now := time.Now().UTC().Format(time.RFC3339)
	if err := writer.TrackRequest(storage.RequestRecord{ID: 7, ClientKeyHash: "hash-7", CreatedAt: now}); err != nil {
		t.Fatalf("TrackRequest: %v", err)
	}
	if err := writer.TrackUsage(storage.UsageRecord{RequestID: 7, PromptTokens: 11, CompletionTokens: 4, CachedTokens: 2, Cost: 0.5}); err != nil {
		t.Fatalf("TrackUsage: %v", err)
	}
	time.Sleep(150 * time.Millisecond)

	s := NewServer(nil, nil, WithUsageSource(storageUsageSource{w: writer}))
	text, isErr := callToolText(t, s, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_client_usage","arguments":{}}}`)
	if isErr {
		t.Fatalf("get_client_usage failed: %s", text)
	}

	var payload map[string]any
	if err := json.Unmarshal([]byte(text), &payload); err != nil {
		t.Fatalf("decode payload %q: %v", text, err)
	}
	for key, want := range map[string]float64{
		"total_requests":    1,
		"prompt_tokens":     11,
		"completion_tokens": 4,
		"cached_tokens":     2,
		"total_tokens":      15,
		"cost":              0.5,
	} {
		got, ok := payload[key].(float64)
		if !ok {
			t.Errorf("payload field %q missing/not numeric: %v", key, payload)
			continue
		}
		if got != want {
			t.Errorf("payload %s = %v, want %v (%v)", key, got, want, payload)
		}
	}
}

// TestGetClientUsageWithoutSourceStillZeros documents the fallback for servers
// built without a usage source (no storage writer at all).
func TestGetClientUsageWithoutSourceStillZeros(t *testing.T) {
	s := NewServer(nil, nil)
	text, isErr := callToolText(t, s, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_client_usage","arguments":{"client_id":"x","window":"7d"}}}`)
	if isErr {
		t.Fatalf("get_client_usage failed: %s", text)
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(text), &payload); err != nil {
		t.Fatalf("decode payload %q: %v", text, err)
	}
	if payload["total_requests"].(float64) != 0 || payload["total_tokens"].(float64) != 0 || payload["cost"].(float64) != 0 {
		t.Errorf("expected zero fallback payload, got %v", payload)
	}
}

type failingUsageSource struct{}

func (failingUsageSource) GetClientUsage(ctx context.Context, clientID, window string) (any, error) {
	return nil, context.DeadlineExceeded
}

// TestGetClientUsageSourceErrorSurfaced keeps query failures visible instead of
// silently degrading to zeros.
func TestGetClientUsageSourceErrorSurfaced(t *testing.T) {
	s := NewServer(nil, nil, WithUsageSource(failingUsageSource{}))
	text, isErr := callToolText(t, s, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_client_usage","arguments":{}}}`)
	if !isErr {
		t.Fatalf("expected an error result, got %s", text)
	}
	if !strings.Contains(text, "error retrieving client usage") {
		t.Errorf("expected the storage error to be surfaced, got %s", text)
	}
}
