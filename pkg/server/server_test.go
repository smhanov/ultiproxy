package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

type fakeInferenceProvider struct {
	mu           sync.Mutex
	name         string
	calls        int
	capturedOpts []provider.Option
	streamFn     func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error)
	generateFn   func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error)
}

func (f *fakeInferenceProvider) Name() string { return f.name }

func (f *fakeInferenceProvider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	f.mu.Lock()
	f.calls++
	f.capturedOpts = append(f.capturedOpts, opts...)
	f.mu.Unlock()

	if f.streamFn != nil {
		return f.streamFn(ctx, msgs, opts...)
	}
	ch := make(chan ir.Event, 2)
	ch <- ir.EventMessageStart{ID: "msg-1"}
	ch <- ir.EventMessageStop{FinishReason: "stop"}
	close(ch)
	return ch, nil
}

func (f *fakeInferenceProvider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	f.mu.Lock()
	f.calls++
	f.capturedOpts = append(f.capturedOpts, opts...)
	f.mu.Unlock()

	if f.generateFn != nil {
		return f.generateFn(ctx, msgs, opts...)
	}
	return &ir.Response{
		ID:           "resp-1",
		FinishReason: "stop",
		Usage:        &ir.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}, nil
}

// 1. Honest routing before commit:
// A model routed to ONE lane whose upstream returns a synchronous error BEFORE commit
// must NOT silently walk to another lane.
func TestServer_MappedLaneSyncErrorHonestFailure(t *testing.T) {
	provA := &fakeInferenceProvider{
		name: "prov-a",
		streamFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
			return nil, errors.New("provider A 503 upstream error")
		},
	}

	provB := &fakeInferenceProvider{
		name: "prov-b",
		streamFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
			ch := make(chan ir.Event, 3)
			ch <- ir.EventMessageStart{ID: "msg-b"}
			ch <- ir.EventTextDelta{Index: 0, Text: "Hello from Provider B"}
			ch <- ir.EventMessageStop{FinishReason: "stop"}
			close(ch)
			return ch, nil
		},
	}

	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: provA})
	registry.Register(provider.Provider{Inference: provB})

	srv := NewServer(nil, registry)

	// Stream request routed to prov-a via prefix match
	body := `{"model":"prov-a/gpt-4o","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 Bad Gateway, got %d: %s", rec.Code, rec.Body.String())
	}

	if provA.calls != 1 {
		t.Errorf("expected provA to be called once, got %d", provA.calls)
	}
	if provB.calls != 0 {
		t.Errorf("expected provB calls == 0 (no cross-vendor failover walk), got %d", provB.calls)
	}

	var errResp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("failed to decode JSON error body: %v, body: %s", err, rec.Body.String())
	}
	errObj, ok := errResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected 'error' object in response, got: %v", errResp)
	}
	msg, _ := errObj["message"].(string)
	if !strings.Contains(msg, "provider A 503 upstream error") {
		t.Errorf("expected error message to carry upstream message, got %q", msg)
	}

	// Unknown-model case: model "totally-bogus" -> 404 with error type "unknown_model" and zero provider calls.
	provA.calls = 0
	provB.calls = 0

	bogusBody := `{"model":"totally-bogus","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	bogusReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(bogusBody))
	bogusRec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(bogusRec, bogusReq)

	if bogusRec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 Not Found for totally-bogus, got %d: %s", bogusRec.Code, bogusRec.Body.String())
	}
	if provA.calls != 0 || provB.calls != 0 {
		t.Errorf("expected zero provider calls for unknown model, got provA=%d, provB=%d", provA.calls, provB.calls)
	}

	var bogusErrResp map[string]any
	if err := json.Unmarshal(bogusRec.Body.Bytes(), &bogusErrResp); err != nil {
		t.Fatalf("failed to decode JSON error body for bogus model: %v", err)
	}
	bogusErrObj, ok := bogusErrResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected 'error' object in bogus model response, got: %v", bogusErrResp)
	}
	if bogusErrObj["type"] != "unknown_model" {
		t.Errorf("expected error type 'unknown_model', got %v", bogusErrObj["type"])
	}
}

// TestServer_FailoverBeforeCommit preserves backward compatibility for test runners targeting the old name.
func TestServer_FailoverBeforeCommit(t *testing.T) {
	TestServer_MappedLaneSyncErrorHonestFailure(t)
}

// 2. Failover NEVER after first byte:
// Upstream error mid-stream -> stream terminated without switching provider.
func TestServer_FailoverNeverAfterFirstByte(t *testing.T) {
	provA := &fakeInferenceProvider{
		name: "prov-a",
		streamFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
			ch := make(chan ir.Event, 3)
			ch <- ir.EventMessageStart{ID: "msg-a"}
			ch <- ir.EventTextDelta{Index: 0, Text: "Initial chunk"}
			// Mid-stream upstream error AFTER headers committed:
			ch <- ir.EventUpstreamError{Kind: "provider_overloaded", Message: "server dropped mid-stream"}
			close(ch)
			return ch, nil
		},
	}

	provB := &fakeInferenceProvider{
		name: "prov-b",
	}

	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: provA})
	registry.Register(provider.Provider{Inference: provB})

	srv := NewServer(nil, registry)

	body := `{"model":"prov-a/gpt-4o","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	// Since headers were committed on first chunk, response code is 200
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK headers committed, got %d", rec.Code)
	}

	if provA.calls != 1 {
		t.Errorf("expected provA to be called once, got %d", provA.calls)
	}

	// CRITICAL ASSERTION: Provider B MUST NEVER BE CALLED
	if provB.calls != 0 {
		t.Errorf("FAILOVER AFTER FIRST BYTE OCCURRED: provB called %d times!", provB.calls)
	}

	respBody := rec.Body.String()
	if !strings.Contains(respBody, "Initial chunk") {
		t.Errorf("expected initial chunk in response, got:\n%s", respBody)
	}
	if !strings.Contains(respBody, "server dropped mid-stream") {
		t.Errorf("expected error payload in response stream, got:\n%s", respBody)
	}
}

// 3. Auth 401 paths
func TestServer_Auth401Paths(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			Addr:   "127.0.0.1:9050",
			APIKey: "admin-secret-key-123",
			ClientKeys: map[string]string{
				"client-alpha": "alpha-secret-456",
			},
		},
	}

	prov := &fakeInferenceProvider{name: "prov"}
	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: prov})

	srv := NewServer(cfg, registry)

	// A. Healthz allows unauthenticated
	reqHealth := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	recHealth := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recHealth, reqHealth)
	if recHealth.Code != http.StatusOK {
		t.Errorf("expected /healthz 200 OK without auth, got %d", recHealth.Code)
	}

	// B. /v1/chat/completions without Authorization header -> 401
	reqNoAuth := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"prov-b/gpt-4o"}`))
	recNoAuth := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recNoAuth, reqNoAuth)
	if recNoAuth.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 without auth, got %d", recNoAuth.Code)
	}
	if !strings.Contains(recNoAuth.Body.String(), "invalid_api_key") {
		t.Errorf("expected invalid_api_key error body, got:\n%s", recNoAuth.Body.String())
	}

	// C. /v1/chat/completions with wrong key -> 401
	reqBadAuth := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"prov-a/gpt-4o"}`))
	reqBadAuth.Header.Set("Authorization", "Bearer wrong-key")
	recBadAuth := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recBadAuth, reqBadAuth)
	if recBadAuth.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 with bad key, got %d", recBadAuth.Code)
	}

	// D. Valid admin key -> 200
	reqAdmin := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"prov-a/gpt-4o","messages":[]}`))
	reqAdmin.Header.Set("Authorization", "Bearer admin-secret-key-123")
	recAdmin := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recAdmin, reqAdmin)
	if recAdmin.Code != http.StatusOK {
		t.Errorf("expected 200 with admin key, got %d: %s", recAdmin.Code, recAdmin.Body.String())
	}

	// E. Valid client key -> 200
	reqClient := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"prov-a/gpt-4o","messages":[]}`))
	reqClient.Header.Set("Authorization", "Bearer alpha-secret-456")
	recClient := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recClient, reqClient)
	if recClient.Code != http.StatusOK {
		t.Errorf("expected 200 with client key, got %d: %s", recClient.Code, recClient.Body.String())
	}
}

// 4. Per-client accounting tag
func TestServer_PerClientAccountingTag(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{
			ClientKeys: map[string]string{
				"acct-client": "secret-token-xyz",
			},
		},
	}

	expectedHashBytes := sha256.Sum256([]byte("secret-token-xyz"))
	expectedHash := hex.EncodeToString(expectedHashBytes[:])

	var capturedClientKeyHash string

	prov := &fakeInferenceProvider{
		name: "prov",
		generateFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
			rc := provider.NewRequestConfig(opts...)
			capturedClientKeyHash = rc.ClientKeyHash
			return &ir.Response{FinishReason: "stop"}, nil
		},
	}

	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: prov})
	srv := NewServer(cfg, registry)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"prov-a/gpt-4o","messages":[]}`))
	req.Header.Set("Authorization", "Bearer secret-token-xyz")
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rec.Code, rec.Body.String())
	}

	if capturedClientKeyHash != expectedHash {
		t.Errorf("expected ClientKeyHash %q, got %q", expectedHash, capturedClientKeyHash)
	}
}

// 5. Usage event -> TrackUsage call
func TestServer_UsageEvent_TrackUsageCall(t *testing.T) {
	tmpDB := t.TempDir() + "/test-telemetry.db"
	writer, err := storage.NewWriter(tmpDB, storage.WithBatchSize(1))
	if err != nil {
		t.Fatalf("failed to create storage writer: %v", err)
	}
	defer writer.Close()

	prov := &fakeInferenceProvider{
		name: "prov",
		streamFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
			ch := make(chan ir.Event, 4)
			ch <- ir.EventMessageStart{ID: "msg-test"}
			ch <- ir.EventTextDelta{Index: 0, Text: "Hello"}
			ch <- ir.EventUsageUpdate{PromptTokens: 42, CompletionTokens: 18, TotalTokens: 60, Cost: 0.0035}
			ch <- ir.EventMessageStop{FinishReason: "stop"}
			close(ch)
			return ch, nil
		},
	}

	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: prov})

	srv := NewServer(nil, registry, WithStorageWriter(writer))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"prov-a/gpt-4o","messages":[],"stream":true}`))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rec.Code)
	}

	// Give background worker time to flush batch
	time.Sleep(100 * time.Millisecond)

	var promptTokens, completionTokens int64
	var cost float64
	err = writer.DB().QueryRow("SELECT prompt_tokens, completion_tokens, cost FROM usage LIMIT 1").Scan(&promptTokens, &completionTokens, &cost)
	if err != nil {
		t.Fatalf("failed to query usage table: %v", err)
	}

	if promptTokens != 42 || completionTokens != 18 || cost != 0.0035 {
		t.Errorf("usage mismatch in DB: prompt=%d, completion=%d, cost=%f", promptTokens, completionTokens, cost)
	}
}

// 6. Additional endpoints: models, llms.txt, quota
func TestServer_AdditionalEndpoints(t *testing.T) {
	sm := state.NewStateManager()
	sm.Update(func(snap *state.RuntimeSnapshot) {
		snap.Models["gpt-4o"] = state.ModelRuntime{
			ID:       "gpt-4o",
			Provider: "openai",
			Enabled:  true,
		}
	})

	tmpFile, err := os.CreateTemp("", "llms*.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	_, _ = tmpFile.WriteString("LLMs info test content")
	tmpFile.Close()

	cfg := &Config{
		Server: ServerConfig{
			LLMsTxtPath: tmpFile.Name(),
		},
	}

	srv := NewServer(cfg, nil, WithStateManager(sm))

	// GET /v1/models
	reqModels := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	recModels := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recModels, reqModels)
	if recModels.Code != http.StatusOK {
		t.Errorf("expected 200 from /v1/models, got %d", recModels.Code)
	}
	if !strings.Contains(recModels.Body.String(), "gpt-4o") {
		t.Errorf("expected gpt-4o in models response: %s", recModels.Body.String())
	}

	// GET /llms.txt
	reqLLM := httptest.NewRequest(http.MethodGet, "/llms.txt", nil)
	recLLM := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recLLM, reqLLM)
	if recLLM.Code != http.StatusOK {
		t.Errorf("expected 200 from /llms.txt, got %d", recLLM.Code)
	}
	if !strings.Contains(recLLM.Body.String(), "LLMs info test content") {
		t.Errorf("expected file content in /llms.txt: %s", recLLM.Body.String())
	}

	// GET /api/quota
	reqQuota := httptest.NewRequest(http.MethodGet, "/api/quota", nil)
	recQuota := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recQuota, reqQuota)
	if recQuota.Code != http.StatusOK {
		t.Errorf("expected 200 from /api/quota, got %d", recQuota.Code)
	}
	var qResp QuotaDashboardResponse
	if err := json.Unmarshal(recQuota.Body.Bytes(), &qResp); err != nil {
		t.Errorf("failed to decode quota dashboard response: %v", err)
	}
}

func TestServer_Upstream429ErrorPreserved(t *testing.T) {
	provA := &fakeInferenceProvider{
		name: "augure",
		streamFn: func(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
			return nil, errors.New("upstream error (status 429): Daily token limit exceeded")
		},
	}

	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: provA})

	sm := state.NewStateManager()
	sm.Update(func(snap *state.RuntimeSnapshot) {
		snap.Models["tofino-3"] = state.ModelRuntime{
			ID:       "tofino-3",
			Provider: "augure",
			Enabled:  true,
		}
	})

	srv := NewServer(nil, registry, WithStateManager(sm))

	body := `{"model":"tofino-3","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("expected 429 Too Many Requests, got %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "Daily token limit exceeded") {
		t.Errorf("expected error body to contain upstream error message, got: %s", rec.Body.String())
	}
}
