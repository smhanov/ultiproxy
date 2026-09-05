package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// callMCPGetClientUsage drives the real /mcp JSON-RPC surface of a server and
// returns the decoded get_client_usage payload.
func callMCPGetClientUsage(t *testing.T, srv *Server, args string) map[string]any {
	t.Helper()

	body := `{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"get_client_usage","arguments":` + args + `}}`
	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+adminKey)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK from /mcp, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Result struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode JSON-RPC response: %v\n%s", err, rec.Body.String())
	}
	if resp.Error != nil {
		t.Fatalf("JSON-RPC error: %s", resp.Error.Message)
	}
	if resp.Result.IsError {
		t.Fatalf("get_client_usage returned an error result: %s", rec.Body.String())
	}
	if len(resp.Result.Content) == 0 || resp.Result.Content[0].Type != "text" {
		t.Fatalf("expected text content, got %s", rec.Body.String())
	}

	var payload map[string]any
	if err := json.Unmarshal([]byte(resp.Result.Content[0].Text), &payload); err != nil {
		t.Fatalf("decode get_client_usage payload %q: %v", resp.Result.Content[0].Text, err)
	}
	return payload
}

func usageFloat(t *testing.T, payload map[string]any, key string) float64 {
	t.Helper()
	v, ok := payload[key].(float64)
	if !ok {
		t.Fatalf("payload field %q is not numeric: %v (%T)", key, payload[key], payload[key])
	}
	return v
}

// seedUsageRow writes one request row plus its usage row and waits for the
// asynchronous telemetry writer to flush them.
func seedUsageRow(t *testing.T, w *storage.Writer, id int64, clientKeyHash, createdAt string, prompt, completion, cached int64, cost float64) {
	t.Helper()
	if err := w.TrackRequest(storage.RequestRecord{
		ID:            id,
		ClientKeyHash: clientKeyHash,
		Provider:      "prov",
		CreatedAt:     createdAt,
		CompletedAt:   createdAt,
		FinishReason:  "stop",
	}); err != nil {
		t.Fatalf("TrackRequest: %v", err)
	}
	if err := w.TrackUsage(storage.UsageRecord{
		RequestID:        id,
		PromptTokens:     prompt,
		CompletionTokens: completion,
		CachedTokens:     cached,
		Cost:             cost,
	}); err != nil {
		t.Fatalf("TrackUsage: %v", err)
	}
}

// adminKey authenticates the MCP calls the tests make. Client keys are only
// there so the usage source can resolve names to key hashes.
const adminKey = "test-admin-key"

func newUsageTestServer(t *testing.T, dir string, clientKeys map[string]string) (*Server, *storage.Writer) {
	t.Helper()

	writer, err := storage.NewWriter(filepath.Join(dir, "telemetry.db"), storage.WithBatchSize(1))
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	t.Cleanup(func() { _ = writer.Close() })

	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: &fakeInferenceProvider{name: "prov"}})

	cfg := DefaultConfig()
	cfg.DataDir = dir
	cfg.Server.APIKey = adminKey
	cfg.Server.ClientKeys = clientKeys

	return NewServer(cfg, registry, WithStorageWriter(writer)), writer
}

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// T024 AC1: with request/token/cost rows in the SQLite telemetry store, the MCP
// get_client_usage tool must report those totals instead of the hardcoded
// zeros it used to return.
func TestServer_MCPGetClientUsageReportsStorageTotals(t *testing.T) {
	dir := t.TempDir()
	aliceHash := sha256Hex("alice-secret")
	bobHash := sha256Hex("bob-secret")

	srv, writer := newUsageTestServer(t, dir, map[string]string{
		"alice": "alice-secret",
		"bob":   "bob-secret",
	})

	now := time.Now().UTC()
	seedUsageRow(t, writer, 1001, aliceHash, now.Format(time.RFC3339), 1000, 500, 100, 0.12)
	seedUsageRow(t, writer, 1002, bobHash, now.Add(-2*time.Hour).Format(time.RFC3339), 2000, 1000, 0, 0.34)
	time.Sleep(200 * time.Millisecond)

	overall := callMCPGetClientUsage(t, srv, `{}`)
	if got := usageFloat(t, overall, "total_requests"); got != 2 {
		t.Errorf("overall total_requests = %v, want 2 (payload %v)", got, overall)
	}
	if got := usageFloat(t, overall, "total_tokens"); got != 4500 {
		t.Errorf("overall total_tokens = %v, want 4500 (payload %v)", got, overall)
	}
	if got := usageFloat(t, overall, "prompt_tokens"); got != 3000 {
		t.Errorf("overall prompt_tokens = %v, want 3000 (payload %v)", got, overall)
	}
	if got := usageFloat(t, overall, "completion_tokens"); got != 1500 {
		t.Errorf("overall completion_tokens = %v, want 1500 (payload %v)", got, overall)
	}
	if got := usageFloat(t, overall, "cost"); math.Abs(got-0.46) > 1e-9 {
		t.Errorf("overall cost = %v, want 0.46 (payload %v)", got, overall)
	}

	// Per-client view by configured client key name...
	byName := callMCPGetClientUsage(t, srv, `{"client_id":"alice"}`)
	if got := usageFloat(t, byName, "total_requests"); got != 1 {
		t.Errorf("alice total_requests = %v, want 1 (payload %v)", got, byName)
	}
	if got := usageFloat(t, byName, "total_tokens"); got != 1500 {
		t.Errorf("alice total_tokens = %v, want 1500 (payload %v)", got, byName)
	}
	if got := usageFloat(t, byName, "cost"); math.Abs(got-0.12) > 1e-9 {
		t.Errorf("alice cost = %v, want 0.12 (payload %v)", got, byName)
	}

	// ...and by the raw key hash the requests table stores.
	byHash := callMCPGetClientUsage(t, srv, `{"client_id":"`+bobHash+`"}`)
	if got := usageFloat(t, byHash, "total_requests"); got != 1 {
		t.Errorf("bob total_requests = %v, want 1 (payload %v)", got, byHash)
	}
	if got := usageFloat(t, byHash, "cost"); math.Abs(got-0.34) > 1e-9 {
		t.Errorf("bob cost = %v, want 0.34 (payload %v)", got, byHash)
	}

	// The window filter still applies: a 1h window must exclude the 2h-old row.
	windowed := callMCPGetClientUsage(t, srv, `{"window":"1h"}`)
	if got := usageFloat(t, windowed, "total_requests"); got != 1 {
		t.Errorf("windowed total_requests = %v, want 1 (payload %v)", got, windowed)
	}
	if got := usageFloat(t, windowed, "total_tokens"); got != 1500 {
		t.Errorf("windowed total_tokens = %v, want 1500 (payload %v)", got, windowed)
	}
}

// T024 AC2: an empty telemetry database returns zeros without erroring.
func TestServer_MCPGetClientUsageEmptyDatabase(t *testing.T) {
	srv, _ := newUsageTestServer(t, t.TempDir(), nil)

	payload := callMCPGetClientUsage(t, srv, `{}`)
	for _, key := range []string{"total_requests", "total_tokens", "prompt_tokens", "completion_tokens", "cost"} {
		if got := usageFloat(t, payload, key); got != 0 {
			t.Errorf("empty DB %s = %v, want 0 (payload %v)", key, got, payload)
		}
	}
}

// T024 AC3: production NewServer must hand the MCP server a usage source, so
// get_client_usage reads SQLite instead of answering synthetic zeros. The
// wiring is asserted directly on the constructed server.
func TestServer_NewServerWiresMCPUsageSource(t *testing.T) {
	srv, _ := newUsageTestServer(t, t.TempDir(), nil)

	if srv.mcpServer == nil {
		t.Fatal("NewServer built no MCP server")
	}
	field := reflect.ValueOf(srv.mcpServer).Elem().FieldByName("usageSource")
	if !field.IsValid() {
		t.Fatal("mcp.Server has no usageSource field to assert on")
	}
	if field.IsNil() {
		t.Fatal("NewServer did not call mcp.WithUsageSource: mcp.Server.usageSource is nil")
	}
}
