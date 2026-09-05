package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestOpenAICompat_CodingPlanMaxTokens(t *testing.T) {
	var capturedPayload map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedPayload)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-zai-01",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Coding plan answer",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-zai-key",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			CodingPlanPath: true,
			MaxTokensByModel: map[string]int{
				"air": 98304,
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role:   "user",
			Blocks: []ir.Block{ir.TextBlock{Text: "Write quicksort"}},
		},
	}

	// 1. Request glm-4.5-air -> should carry max_tokens 98304
	_, err = p.Generate(context.Background(), msgs, provider.WithModel("glm-4.5-air"))
	if err != nil {
		t.Fatalf("Generate glm-4.5-air failed: %v", err)
	}
	if mt, ok := capturedPayload["max_tokens"].(float64); !ok || int(mt) != 98304 {
		t.Errorf("expected max_tokens 98304 for glm-4.5-air, got %v", capturedPayload["max_tokens"])
	}

	// 2. Request glm-5.3-flash (or other model) -> should carry 131072 default
	_, err = p.Generate(context.Background(), msgs, provider.WithModel("glm-5.3-flash"))
	if err != nil {
		t.Fatalf("Generate glm-5.3-flash failed: %v", err)
	}
	if mt, ok := capturedPayload["max_tokens"].(float64); !ok || int(mt) != 131072 {
		t.Errorf("expected max_tokens 131072 for other models, got %v", capturedPayload["max_tokens"])
	}
}

func TestOpenAICompat_EchoReasoning(t *testing.T) {
	var capturedPayload map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedPayload)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-ds-reasoning",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":              "assistant",
						"content":           "24",
						"reasoning_content": "12 * 2 = 24.",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-ds-key",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			EchoReasoning: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role:   "user",
			Blocks: []ir.Block{ir.TextBlock{Text: "sqrt(144)?"}},
		},
		{
			Role: "assistant",
			Blocks: []ir.Block{
				ir.ReasoningBlock{Text: "The user asks for sqrt(144). 12*12=144."},
				ir.TextBlock{Text: "12"},
			},
		},
		{
			Role:   "user",
			Blocks: []ir.Block{ir.TextBlock{Text: "double it"}},
		},
	}

	resp, err := p.Generate(context.Background(), msgs, provider.WithModel("deepseek-reasoner"))
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// 1. Assert input reasoning was echoed on upstream wire in reasoning_content
	rawMsgs, ok := capturedPayload["messages"].([]any)
	if !ok || len(rawMsgs) != 3 {
		t.Fatalf("expected 3 messages sent to upstream, got %v", capturedPayload["messages"])
	}
	asstMsg := rawMsgs[1].(map[string]any)
	if rc, ok := asstMsg["reasoning_content"].(string); !ok || rc != "The user asks for sqrt(144). 12*12=144." {
		t.Errorf("expected reasoning_content echoed, got %v", asstMsg["reasoning_content"])
	}
	if asstMsg["content"] != "12" {
		t.Errorf("expected content '12', got %v", asstMsg["content"])
	}

	// 2. Assert output carries reasoning block extracted from reasoning_content
	if resp == nil || resp.Message == nil {
		t.Fatal("expected non-nil response message")
	}
	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("expected 2 blocks in response, got %d", len(resp.Message.Blocks))
	}
	rb, ok := resp.Message.Blocks[0].(ir.ReasoningBlock)
	if !ok || rb.Text != "12 * 2 = 24." {
		t.Errorf("expected reasoning block with text '12 * 2 = 24.', got %+v", resp.Message.Blocks[0])
	}
	tb, ok := resp.Message.Blocks[1].(ir.TextBlock)
	if !ok || tb.Text != "24" {
		t.Errorf("expected text block '24', got %+v", resp.Message.Blocks[1])
	}
}

func TestOpenAICompat_ModelListPassthrough(t *testing.T) {
	var capturedPayload map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models", "/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"id": "meta-llama/Llama-3-8B-Instruct"},
					{"id": "Qwen/Qwen2.5-Coder-32B"},
				},
			})
		case "/v1/chat/completions", "/chat/completions":
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &capturedPayload)

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "cmpl-vllm-01",
				"choices": []map[string]any{
					{
						"index": 0,
						"message": map[string]any{
							"role":    "assistant",
							"content": "vLLM local output",
						},
						"finish_reason": "stop",
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
		Quirks: Quirks{
			ModelListPassthrough: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	models := p.Models()
	if len(models) != 2 {
		t.Fatalf("expected 2 discovered models, got %d: %v", len(models), models)
	}
	if models[0] != "meta-llama/Llama-3-8B-Instruct" {
		t.Errorf("expected first model Llama-3-8B-Instruct, got %s", models[0])
	}

	msgs := []*ir.Message{
		{
			Role:   "user",
			Blocks: []ir.Block{ir.TextBlock{Text: "hello"}},
		},
	}
	_, err = p.Generate(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if capturedPayload["model"] != "meta-llama/Llama-3-8B-Instruct" {
		t.Errorf("expected default model from discovery, got %v", capturedPayload["model"])
	}
}

// TestOpenAICompat_BearerAuth asserts the standard authentication path now
// that the legacy opencode workspace-cookie scheme is gone: lanes authenticate
// with a plain Authorization: Bearer <api_key> header and never emit Cookie or
// X-Workspace-ID headers.
func TestOpenAICompat_BearerAuth(t *testing.T) {
	var capturedAuth string
	var capturedCookie string
	var capturedWorkspace string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		capturedCookie = r.Header.Get("Cookie")
		capturedWorkspace = r.Header.Get("X-Workspace-ID")

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-opencode-01",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "OpenCode response",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	p, err := New(Config{
		Name:       "opencode",
		BaseURL:    server.URL,
		APIKey:     "sk-opencode-bearer",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role:   "user",
			Blocks: []ir.Block{ir.TextBlock{Text: "check auth"}},
		},
	}

	_, err = p.Generate(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if capturedAuth != "Bearer sk-opencode-bearer" {
		t.Errorf("expected Authorization 'Bearer sk-opencode-bearer', got %q", capturedAuth)
	}
	if capturedCookie != "" {
		t.Errorf("expected no Cookie header, got %q", capturedCookie)
	}
	if capturedWorkspace != "" {
		t.Errorf("expected no X-Workspace-ID header, got %q", capturedWorkspace)
	}
}

func TestOpenAICompat_AugureRefreshAndDefaultModel(t *testing.T) {
	var refreshTouched bool
	var capturedModel string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/auth/v1/token":
			refreshTouched = true
			var reqBody map[string]string
			_ = json.NewDecoder(r.Body).Decode(&reqBody)
			if reqBody["refresh_token"] != "rt-init" {
				http.Error(w, "invalid refresh token", http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token":  "token-refreshed",
				"refresh_token": "rt-2",
				"expires_in":    3600,
			})

		case "/v1/chat/completions", "/chat/completions":
			authHeader := r.Header.Get("Authorization")
			if authHeader == "Bearer token-expired" {
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = w.Write([]byte(`{"error":{"message":"token expired"}}`))
				return
			}
			if authHeader != "Bearer token-refreshed" {
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = w.Write([]byte(`{"error":{"message":"unauthorized"}}`))
				return
			}

			body, _ := io.ReadAll(r.Body)
			var payload map[string]any
			_ = json.Unmarshal(body, &payload)
			if m, ok := payload["model"].(string); ok {
				capturedModel = m
			}

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "cmpl-augure-01",
				"choices": []map[string]any{
					{
						"index": 0,
						"message": map[string]any{
							"role":    "assistant",
							"content": "Augure refreshed output",
						},
						"finish_reason": "stop",
					},
				},
			})

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	tokenSrc := NewSupabaseTokenSource(server.Client(), server.URL+"/auth/v1/token?grant_type=refresh_token", "", "token-expired", "rt-init")

	p, err := New(Config{
		BaseURL:     server.URL,
		TokenSource: tokenSrc,
		HTTPClient:  server.Client(),
		Quirks: Quirks{
			AuthViaSupabaseRefresh: true,
			DefaultModel:           "tofino-3",
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role:   "user",
			Blocks: []ir.Block{ir.TextBlock{Text: "hi augure"}},
		},
	}

	// Generate without specifying model -> should apply DefaultModel ("tofino-3")
	// and automatically recover from 401 via Supabase refresh
	resp, err := p.Generate(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}

	if !refreshTouched {
		t.Error("expected Supabase refresh endpoint to be touched on 401")
	}
	if capturedModel != "tofino-3" {
		t.Errorf("expected model 'tofino-3', got %q", capturedModel)
	}
}

type fakeActor struct {
	mu              sync.Mutex
	lock            sync.Mutex
	active          int
	maxActive       int
	acquireCount    int
	instanceID      string
	token           string
	startedAgentIDs []string
}

func (a *fakeActor) Acquire(ctx context.Context) error {
	a.lock.Lock()
	a.mu.Lock()
	a.active++
	if a.active > a.maxActive {
		a.maxActive = a.active
	}
	a.acquireCount++
	a.mu.Unlock()
	return nil
}

func (a *fakeActor) Release() {
	a.mu.Lock()
	a.active--
	a.mu.Unlock()
	a.lock.Unlock()
}

func (a *fakeActor) SetToken(tok string) {
	a.mu.Lock()
	a.token = tok
	a.mu.Unlock()
}

func (a *fakeActor) Token() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.token
}

func (a *fakeActor) InstanceID() string {
	return a.instanceID
}

// FetchUsage returns a canned freebuff usage payload for quota tests.
func (a *fakeActor) FetchUsage(ctx context.Context, fingerprintID string) ([]byte, error) {
	return []byte(`{"dayUsed":10,"dayLimit":100,"weekUsed":30,"weekLimit":300,"monthUsed":50,"monthLimit":500}`), nil
}

// SessionInfo returns a canned session (instance + model) for quota Detail.
func (a *fakeActor) SessionInfo(ctx context.Context) (string, string, error) {
	return a.instanceID, "test-model", nil
}

// startRunAgentIDs records the agentId of every StartRun call.
func (a *fakeActor) startRunAgentIDs() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]string(nil), a.startedAgentIDs...)
}

// fakeAgentRun satisfies the GetRunID() string surface applyRequestTransforms
// accepts for StartRun results.
type fakeAgentRun struct{ id string }

func (r *fakeAgentRun) GetRunID() string { return r.id }

// StartRun records the requested agentId and returns a canned run id.
func (a *fakeActor) StartRun(ctx context.Context, agentID string) (any, error) {
	a.mu.Lock()
	a.startedAgentIDs = append(a.startedAgentIDs, agentID)
	a.mu.Unlock()
	return &fakeAgentRun{id: "run-abc-123"}, nil
}

// fakeSessionManager satisfies the optional freebuff session-lifecycle surface
// (Reconcile/BoundModel/DeleteSession/Bind) so tests can drive each lifecycle
// branch without HTTP. The provider duck-types this surface.
type fakeSessionManager struct {
	reconciled bool
	bound      string // value BoundModel() reports
	bindCalls  []string
	deletes    int
	bindErr    error
}

func (f *fakeSessionManager) Reconcile(...context.Context) error {
	f.reconciled = true
	return nil
}

func (f *fakeSessionManager) BoundModel() string { return f.bound }

func (f *fakeSessionManager) DeleteSession(...context.Context) error {
	f.deletes++
	return nil
}

func (f *fakeSessionManager) Bind(_ any, optionalModel ...string) error {
	if f.bindErr != nil {
		return f.bindErr
	}
	model := ""
	if len(optionalModel) > 0 {
		model = optionalModel[0]
	}
	f.bindCalls = append(f.bindCalls, model)
	return nil
}

// TestOpenAICompat_FreebuffSessionLifecycle: before each chat the lane must
// reconcile the free session and bind it to the canonical upstream model when
// no session is bound (status none) — POST /freebuff/session with the full
// publisher model id, not the bare request alias. This mirrors the old node
// bridge's server.js and fixes "428 waiting_room_required" on a fresh account.
func TestOpenAICompat_FreebuffSessionLifecycle_BindWhenNone(t *testing.T) {
	sm := &fakeSessionManager{} // BoundModel()=="" → session none
	server := freebuffChatStubServer()
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-token",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       sm,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("glm-5.3-flash")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if !sm.reconciled {
		t.Error("expected Reconcile (GET /freebuff/session) before chat")
	}
	if len(sm.bindCalls) != 1 {
		t.Fatalf("expected exactly 1 Bind call, got %d (%v)", len(sm.bindCalls), sm.bindCalls)
	}
	if sm.bindCalls[0] != "z-ai/glm-5.3-flash" {
		t.Errorf("Bind model = %q, want canonical publisher id %q (not the bare alias)", sm.bindCalls[0], "z-ai/glm-5.3-flash")
	}
	if sm.deletes != 0 {
		t.Errorf("expected no DeleteSession on fresh account, got %d", sm.deletes)
	}
}

// TestOpenAICompat_FreebuffSessionLifecycle_ModelSwitch: an active session
// bound to a different model must be deleted and re-bound to the requested one.
func TestOpenAICompat_FreebuffSessionLifecycle_ModelSwitch(t *testing.T) {
	sm := &fakeSessionManager{bound: "deepseek/deepseek-v4-flash"}
	server := freebuffChatStubServer()
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-token",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       sm,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("glm-5.3-flash")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if sm.deletes != 1 {
		t.Errorf("expected 1 DeleteSession on model switch, got %d", sm.deletes)
	}
	if len(sm.bindCalls) != 1 || sm.bindCalls[0] != "z-ai/glm-5.3-flash" {
		t.Errorf("expected re-Bind to z-ai/glm-5.3-flash, got %v", sm.bindCalls)
	}
}

// TestOpenAICompat_FreebuffSessionLifecycle_ActiveMatch: a session already
// bound to the requested model is left alone — no delete, no re-bind.
func TestOpenAICompat_FreebuffSessionLifecycle_ActiveMatch(t *testing.T) {
	sm := &fakeSessionManager{bound: "z-ai/glm-5.3-flash"}
	server := freebuffChatStubServer()
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-token",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       sm,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("glm-5.3-flash")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if sm.deletes != 0 || len(sm.bindCalls) != 0 {
		t.Errorf("expected session reuse (0 deletes, 0 binds), got %d deletes, %d binds", sm.deletes, len(sm.bindCalls))
	}
}

// TestOpenAICompat_FreebuffSessionLifecycle_BindError: a bind failure (e.g.
// 429 rate_limited or 403 banned from POST /freebuff/session) aborts the chat
// with that error instead of sending a doomed upstream request.
func TestOpenAICompat_FreebuffSessionLifecycle_BindError(t *testing.T) {
	sm := &fakeSessionManager{bindErr: fmt.Errorf("failed to bind model, status: 429: rate_limited")}
	server := freebuffChatStubServer()
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-token",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       sm,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	_, err = p.Generate(context.Background(), msgs, provider.WithModel("glm-5.3-flash"))
	if err == nil {
		t.Fatal("expected Generate to fail when Bind fails")
	}
	if !strings.Contains(err.Error(), "rate_limited") {
		t.Errorf("error = %v, want it to carry the bind failure", err)
	}
}

// freebuffChatStubServer answers /agent-runs and /chat/completions like the
// freebuff upstream (session lifecycle happens through the fake actor).
func freebuffChatStubServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/agent-runs") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"runId":"run-abc-123"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"fb-1","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
}

// fakeAdoptingActor mimics the real actor's bind flow: it starts with no
// instance id and adopts the upstream-minted one on Bind (like
// FreebuffAccountActor does). It exposes what the chat headers SHOULD carry.
type fakeAdoptingActor struct {
	fakeActor
	minted string
}

func (a *fakeAdoptingActor) Bind(_ any, optionalModel ...string) error {
	a.mu.Lock()
	a.instanceID = a.minted
	a.mu.Unlock()
	return nil
}

func (a *fakeAdoptingActor) Reconcile(...context.Context) error { return nil }

func (a *fakeAdoptingActor) BoundModel() string { return "" } // fresh account

func (a *fakeAdoptingActor) DeleteSession(...context.Context) error { return nil }

// TestOpenAICompat_FreebuffInstanceHeaderAfterBind: the lifecycle must run
// before header construction — and per the shipped binary, the chat request
// carries NO x-freebuff-instance-id at all (the instance rides session API
// calls only). The observable: the lifecycle ran (Bind adopted the minted id)
// and the chat headers stay binary-identical.
func TestOpenAICompat_FreebuffInstanceHeaderAfterBind(t *testing.T) {
	var mu sync.Mutex
	var chatInstHeader string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/agent-runs"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"runId":"run-abc-123"}`))
		case strings.HasSuffix(r.URL.Path, "/chat/completions"):
			mu.Lock()
			chatInstHeader = r.Header.Get("x-freebuff-instance-id")
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"fb-1","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	actor := &fakeAdoptingActor{minted: "fb-minted-42"}
	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-token",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       actor,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("glm")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	mu.Lock()
	got := chatInstHeader
	mu.Unlock()
	if got != "fb-minted-42" {
		t.Errorf("chat x-freebuff-instance-id = %q, want the post-bind minted id (session keying)", got)
	}
}

// TestOpenAICompat_FreebuffAgentModelMapping: free mode only accepts specific
// agent/model combos, so the lane must translate the request's model into the
// upstream agentId for POST /agent-runs (upstream 403s free_mode_invalid_agent_model
// otherwise) and the returned runId must flow into codebuff_metadata.run_id.
func TestOpenAICompat_FreebuffAgentModelMapping(t *testing.T) {
	var mu sync.Mutex
	var chatMetadata map[string]any
	var chatModel any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var payload map[string]any
		_ = json.Unmarshal(body, &payload)

		switch {
		case strings.HasSuffix(r.URL.Path, "/agent-runs"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"runId":"run-abc-123"}`))
		case strings.HasSuffix(r.URL.Path, "/chat/completions"):
			mu.Lock()
			meta, _ := payload["codebuff_metadata"].(map[string]any)
			chatMetadata = meta
			chatModel = payload["model"]
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"fb-1","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	actor := &fakeActor{instanceID: "fb-inst-007"}
	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-token",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       actor,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}},
	}
	// Bare alias in: upstream must still see the canonical publisher model.
	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("glm")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// The upstream agentId for the glm-5.3-flash free agent — the raw model id
	// is rejected with free_mode_invalid_agent_model.
	ids := actor.startRunAgentIDs()
	if len(ids) != 1 {
		t.Fatalf("expected exactly 1 StartRun call, got %d (%v)", len(ids), ids)
	}
	if ids[0] != "base3-free-glm-5-3-flash" {
		t.Errorf("StartRun agentId = %q, want %q (the base3-free agent, not the raw model id)", ids[0], "base3-free-glm-5-3-flash")
	}

	// codebuff_metadata.run_id must be the runId minted by START, not a placeholder.
	if chatMetadata == nil {
		t.Fatal("chat payload missing codebuff_metadata")
	}
	if got, _ := chatMetadata["run_id"].(string); got != "run-abc-123" {
		t.Errorf("codebuff_metadata.run_id = %q, want the START-minted %q", got, "run-abc-123")
	}

	// The chat body's model must be the canonical publisher-qualified id the
	// upstream knows (both the old bridge and the shipped CLI send it).
	if got, _ := chatModel.(string); got != "z-ai/glm-5.3-flash" {
		t.Errorf("chat body model = %q, want canonical %q", got, "z-ai/glm-5.3-flash")
	}
}

// TestOpenAICompat_FreebuffAgentModelMapping_AliasesAndUnknown: free-tier model
// aliases (bare names like "glm" or "deepseek") map to their agents too, and an
// unknown model passes through unchanged (fail-open, so newly listed upstream
// models still reach the API).
func TestOpenAICompat_FreebuffAgentModelMapping_AliasesAndUnknown(t *testing.T) {
	cases := []struct {
		model string
		want  string
	}{
		{"glm", "base3-free-glm-5-3-flash"},
		{"glm-5.3-flash", "base3-free-glm-5-3-flash"},
		{"deepseek", "base3-free-deepseek-flash"},
		{"deepseek/deepseek-v4-flash", "base3-free-deepseek-flash"},
		{"mimo-v2.5", "base3-free-mimo"},
		{"kimi-k3-eco", "base3-free-kimi-k3-eco"},
		{"openai/gpt-5.6-luna", "base3-free-luna"},
		{"solar", "base3-free-solar-pro4"},
		{"totally-new-model", "totally-new-model"}, // fail-open passthrough
	}

	for _, tc := range cases {
		actor := &fakeActor{instanceID: "fb-inst-007"}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.HasSuffix(r.URL.Path, "/agent-runs") {
				_, _ = w.Write([]byte(`{"runId":"run-x"}`))
				return
			}
			_, _ = w.Write([]byte(`{"id":"fb-1","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]}`))
		}))
		p, err := New(Config{
			BaseURL:    server.URL,
			APIKey:     "test-token",
			HTTPClient: server.Client(),
			Quirks: Quirks{
				FreebuffActor:       actor,
				FreebuffDefaultTool: true,
			},
		})
		if err != nil {
			t.Fatalf("failed to create provider: %v", err)
		}

		msgs := []*ir.Message{
			{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}},
		}
		if _, err := p.Generate(context.Background(), msgs, provider.WithModel(tc.model)); err != nil {
			t.Fatalf("Generate(%q) failed: %v", tc.model, err)
		}
		server.Close()

		ids := actor.startRunAgentIDs()
		if len(ids) != 1 {
			t.Fatalf("model %q: expected 1 StartRun call, got %d", tc.model, len(ids))
		}
		if ids[0] != tc.want {
			t.Errorf("model %q: StartRun agentId = %q, want %q", tc.model, ids[0], tc.want)
		}
	}
}

func TestOpenAICompat_FreebuffActorLock(t *testing.T) {
	var mu sync.Mutex
	var payloads []map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var p map[string]any
		_ = json.Unmarshal(body, &p)

		mu.Lock()
		payloads = append(payloads, p)
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		time.Sleep(20 * time.Millisecond)

		_, _ = w.Write([]byte("data: {\"id\":\"fb-stream-01\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Buffy says hello\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer server.Close()

	actor := &fakeActor{instanceID: "fb-inst-007"}

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-freebuff-token",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			FreebuffActor:       actor,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role:   "user",
			Blocks: []ir.Block{ir.TextBlock{Text: "help me code"}},
		},
	}

	// Run two concurrent streams
	var wg sync.WaitGroup
	errCh := make(chan error, 2)

	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ch, err := p.Stream(context.Background(), msgs)
			if err != nil {
				errCh <- err
				return
			}
			for range ch {
				// drain stream
			}
		}()
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatalf("concurrent stream failed: %v", err)
	}

	// 1. Assert strict FIFO serialization: maxActive must never exceed 1
	if actor.maxActive != 1 {
		t.Errorf("expected maxActive to be 1 (serialized), got %d", actor.maxActive)
	}
	if actor.acquireCount != 2 {
		t.Errorf("expected 2 acquires, got %d", actor.acquireCount)
	}

	// 2. Assert Buffy prompt, default tool, and codebuff_metadata were injected
	mu.Lock()
	defer mu.Unlock()
	if len(payloads) == 0 {
		t.Fatal("no requests received by upstream")
	}

	firstPayload := payloads[0]

	// Check Buffy system prompt injected at message 0
	rawMsgs := firstPayload["messages"].([]any)
	sysMsg := rawMsgs[0].(map[string]any)
	if sysMsg["role"] != "system" || sysMsg["content"] != "You are Buffy, the coding agent behind Codebuff." {
		t.Errorf("expected Buffy system prompt, got %+v", sysMsg)
	}

	// Check default tool injected
	tools, ok := firstPayload["tools"].([]any)
	if !ok || len(tools) == 0 {
		t.Fatalf("expected tools injected, got %v", firstPayload["tools"])
	}
	firstTool := tools[0].(map[string]any)
	fn := firstTool["function"].(map[string]any)
	if fn["name"] != "read_files" {
		t.Errorf("expected tool 'read_files', got %v", fn["name"])
	}

	// Check codebuff_metadata — binary-identical: no freebuff_instance_id.
	meta, ok := firstPayload["codebuff_metadata"].(map[string]any)
	if !ok {
		t.Fatalf("expected codebuff_metadata in payload, got %v", firstPayload)
	}
	if meta["cost_mode"] != "free" {
		t.Errorf("expected cost_mode 'free', got %v", meta["cost_mode"])
	}
	if _, present := meta["freebuff_instance_id"]; present {
		t.Errorf("freebuff_instance_id must be absent from metadata (binary omits it), got %v", meta["freebuff_instance_id"])
	}
	// The instance id rides session calls, not the chat request.
	if inst := actor.InstanceID(); inst != "fb-inst-007" {
		t.Errorf("actor instance id = %q, want fb-inst-007 (retained for session API)", inst)
	}
}

func TestOpenAICompat_CreditsQuotaObserver(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "xai", "credits.grpcweb.bin")
	fixtureData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	var capturedAuth string
	var capturedWebHeader string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		capturedWebHeader = r.Header.Get("X-Grpc-Web")

		w.Header().Set("Content-Type", "application/grpc-web+proto")
		_, _ = w.Write(fixtureData)
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-xai-key",
		HTTPClient: server.Client(),
		Quirks: Quirks{
			CreditsQuotaObserver: server.URL + "/credits",
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	if p.Provider().Quota == nil {
		t.Fatal("expected Quota provider to be attached when CreditsQuotaObserver is set")
	}

	snap, err := p.Quota(context.Background())
	if err != nil {
		t.Fatalf("Quota() failed: %v", err)
	}

	if capturedAuth != "Bearer test-xai-key" {
		t.Errorf("expected Authorization 'Bearer test-xai-key', got %q", capturedAuth)
	}
	if capturedWebHeader != "1" {
		t.Errorf("expected X-Grpc-Web '1', got %q", capturedWebHeader)
	}

	// The billing response is distilled into a single "Grok Build" window: the
	// usage percent is the smallest fixed32 in [0,100] sitting under a field
	// path that ends in field 1 (15 here — the fixture also carries 45 and two
	// 100s), and the reset is the earliest timestamp still in the future.
	if len(snap.Windows) != 1 {
		t.Fatalf("expected exactly 1 window, got %d: %+v", len(snap.Windows), snap.Windows)
	}
	if snap.Windows[0].Label != "Grok Build" {
		t.Errorf("unexpected window label: %+v", snap.Windows[0])
	}
	if snap.Windows[0].UsedPct != 15.0 {
		t.Errorf("UsedPct = %v, want 15 (min percent candidate)", snap.Windows[0].UsedPct)
	}
	if snap.Windows[0].Remaining != 85.0 || snap.Windows[0].Limit != 100 || snap.Windows[0].Unit != "%" {
		t.Errorf("unexpected window scaling: %+v", snap.Windows[0])
	}
	if !strings.Contains(snap.Detail, "Grok Build 15% used") {
		t.Errorf("Detail = %q, want it to report the Grok Build usage", snap.Detail)
	}
}

func TestOpenAICompat_OAuthManagerAuth(t *testing.T) {
	dir := t.TempDir()
	mgr, err := auth.NewManager(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := mgr.Store(context.Background(), defaultXAIClientID, auth.Credential{
		Provider:    "xai",
		AccessToken: "xai-oauth-token-val",
		ExpiresAt:   time.Now().Add(time.Hour),
		ClientID:    defaultXAIClientID,
	}); err != nil {
		t.Fatal(err)
	}

	var capturedAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-xai-01",
			"choices": []map[string]any{
				{
					"index":   0,
					"message": map[string]any{"role": "assistant", "content": "xai answer"},
				},
			},
		})
	}))
	defer server.Close()

	src := NewOAuthManagerTokenSource(mgr, defaultXAIClientID)

	p, err := New(Config{
		BaseURL:     server.URL,
		TokenSource: src,
		HTTPClient:  server.Client(),
		Quirks: Quirks{
			AuthViaOAuthManager: true,
		},
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}}
	_, err = p.Generate(context.Background(), msgs, provider.WithModel("grok-3"))
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if capturedAuth != "Bearer xai-oauth-token-val" {
		t.Errorf("expected Bearer xai-oauth-token-val, got %q", capturedAuth)
	}
}

func TestOpenAICompat_Quirks_UnsetByDefault(t *testing.T) {
	var capturedPayload map[string]any
	var capturedHeaders http.Header

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeaders = r.Header.Clone()
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedPayload)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-plain-01",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Plain openai response",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	// Empty quirks
	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "plain-key-123",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	// Verify bundle has no quota or auth attached
	bundle := p.Provider()
	if bundle.Quota != nil {
		t.Error("expected Quota to be nil when quirks are unset")
	}
	if bundle.Auth != nil {
		t.Error("expected Auth to be nil when quirks are unset")
	}
	if p.Models() != nil {
		t.Error("expected Models() to be nil when ModelListPassthrough is unset")
	}

	msgs := []*ir.Message{
		{
			Role: "assistant",
			Blocks: []ir.Block{
				ir.ReasoningBlock{Text: "prior thoughts"},
				ir.TextBlock{Text: "prior answer"},
			},
		},
		{
			Role:   "user",
			Blocks: []ir.Block{ir.TextBlock{Text: "hello"}},
		},
	}

	_, err = p.Generate(context.Background(), msgs, provider.WithModel("gpt-4o"))
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// 1. No max_tokens override
	if _, ok := capturedPayload["max_tokens"]; ok {
		t.Errorf("max_tokens should not be present, got %v", capturedPayload["max_tokens"])
	}

	// 2. No Buffy prompt or default tool
	if _, ok := capturedPayload["tools"]; ok {
		t.Errorf("tools should not be injected by default, got %v", capturedPayload["tools"])
	}
	if _, ok := capturedPayload["codebuff_metadata"]; ok {
		t.Errorf("codebuff_metadata should not be present, got %v", capturedPayload["codebuff_metadata"])
	}

	// 3. Prior reasoning stripped from assistant message when EchoReasoning is false
	rawMsgs := capturedPayload["messages"].([]any)
	asstMsg := rawMsgs[0].(map[string]any)
	if _, ok := asstMsg["reasoning_content"]; ok {
		t.Errorf("reasoning_content should NOT be echoed when EchoReasoning is false, got %v", asstMsg["reasoning_content"])
	}

	// 4. No special auth headers (Cookie / X-Workspace-ID)
	if capturedHeaders.Get("Cookie") != "" {
		t.Errorf("Cookie header should not be set, got %q", capturedHeaders.Get("Cookie"))
	}
	if capturedHeaders.Get("X-Workspace-ID") != "" {
		t.Errorf("X-Workspace-ID should not be set, got %q", capturedHeaders.Get("X-Workspace-ID"))
	}

	// Also verify New(Config{}) constructs without error
	bare, err := New(Config{HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("New(Config{}) should succeed, got %v", err)
	}
	if bare.Name() != "openaicompat" {
		t.Errorf("expected default name 'openaicompat', got %s", bare.Name())
	}
}

// TestOpenAICompat_FreebuffQuota verifies the freebuff actor quota path
// (ported from the pre-F2 freebuff lane): Quota() calls actor.FetchUsage,
// normalizes via ParseFreebuffUsageSnapshot, and attaches a Detail string
// from the actor session. Regression guard for the F2a rewire which lost
// freebuff's Quota provider.
func TestOpenAICompat_FreebuffQuota(t *testing.T) {
	actor := &fakeActor{instanceID: "fb-inst-007"}

	p, err := New(Config{
		BaseURL:    "https://codebuff.invalid",
		APIKey:     "tok-1",
		HTTPClient: http.DefaultClient,
		Quirks: Quirks{
			FreebuffActor: actor,
		},
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	bundle := p.ProviderBundle()
	if bundle.Quota == nil {
		t.Fatal("expected Quota provider attached when FreebuffActor is set")
	}

	snap, err := p.Quota(context.Background())
	if err != nil {
		t.Fatalf("Quota failed: %v", err)
	}
	if len(snap.Windows) != 3 {
		t.Fatalf("expected 3 quota windows, got %d", len(snap.Windows))
	}
	if snap.Windows[0].Label != "Daily" {
		t.Errorf("expected Daily window first, got %q", snap.Windows[0].Label)
	}
	if snap.Detail != "instance: fb-inst-007, model: test-model" {
		t.Errorf("expected detail with instance+model, got %q", snap.Detail)
	}
}

func TestOpenAICompat_Login_ErrNotImplemented(t *testing.T) {
	p, err := New(Config{
		BaseURL:    "https://api.openai.com/v1",
		APIKey:     "test-key",
		HTTPClient: http.DefaultClient,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if err := p.Login(context.Background()); !errors.Is(err, provider.ErrNotImplemented) {
		t.Errorf("expected ErrNotImplemented, got %v", err)
	}
}

func TestOpenAICompat_Login_Freebuff(t *testing.T) {
	tempDir := t.TempDir()

	actor := &fakeActor{instanceID: "fb-inst-007"}
	p, err := New(Config{
		BaseURL:    "https://codebuff.invalid",
		APIKey:     "secret-freebuff-tok",
		DataDir:    tempDir,
		HTTPClient: http.DefaultClient,
		Quirks: Quirks{
			FreebuffActor:       actor,
			FreebuffDefaultTool: true,
		},
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if err := p.Login(context.Background()); err != nil {
		t.Fatalf("Login failed: %v", err)
	}

	// Verify token was pushed to actor
	if actor.Token() != "secret-freebuff-tok" {
		t.Errorf("expected actor token secret-freebuff-tok, got %q", actor.Token())
	}

	// Verify token was persisted to freebuff_token
	persistedTok, err := os.ReadFile(filepath.Join(tempDir, "freebuff_token"))
	if err != nil {
		t.Fatalf("failed to read persisted token: %v", err)
	}
	if strings.TrimSpace(string(persistedTok)) != "secret-freebuff-tok" {
		t.Errorf("persisted token = %q, want secret-freebuff-tok", strings.TrimSpace(string(persistedTok)))
	}

	// Verify Token() returns the imported token
	tok, err := p.Token(context.Background())
	if err != nil {
		t.Fatalf("Token() failed: %v", err)
	}
	if tok != "secret-freebuff-tok" {
		t.Errorf("Token() = %q, want secret-freebuff-tok", tok)
	}
}

func TestOpenAICompat_Login_OAuthManager(t *testing.T) {
	var deviceForm url.Values
	mux := http.NewServeMux()
	mux.HandleFunc("/device", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		deviceForm, _ = url.ParseQuery(string(body))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"device_code": "dev-test-123",
			"user_code": "TEST-1234",
			"verification_uri": "https://accounts.x.ai/oauth2/device",
			"expires_in": 300,
			"interval": 1
		}`))
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"access_token": "mock-xai-access-tok",
			"refresh_token": "mock-xai-refresh-tok",
			"expires_in": 3600
		}`))
	})

	srv := httptest.NewServer(mux)
	defer srv.Close()

	tempDir := t.TempDir()
	p, err := New(Config{
		Name:          "xai",
		BaseURL:       "https://api.x.ai",
		DataDir:       tempDir,
		HTTPClient:    srv.Client(),
		DeviceAuthURL: srv.URL + "/device",
		TokenURL:      srv.URL + "/token",
		Quirks: Quirks{
			AuthViaOAuthManager: true,
		},
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if err := p.Login(context.Background()); err != nil {
		t.Fatalf("Login failed: %v", err)
	}

	if got := deviceForm.Get("scope"); got != defaultXAIScope {
		t.Errorf("device code scope = %q, want %q", got, defaultXAIScope)
	}

	tok, err := p.Token(context.Background())
	if err != nil {
		t.Fatalf("Token() failed: %v", err)
	}
	if tok != "mock-xai-access-tok" {
		t.Errorf("Token() = %q, want mock-xai-access-tok", tok)
	}
}

func TestOpenAICompat_StartLogin_XAIScope(t *testing.T) {
	var deviceForm url.Values

	mux := http.NewServeMux()
	mux.HandleFunc("/device", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		deviceForm, _ = url.ParseQuery(string(body))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"device_code": "dev-start-456",
			"user_code": "START-4567",
			"verification_uri": "https://accounts.x.ai/oauth2/device",
			"verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=START-4567",
			"expires_in": 600,
			"interval": 5
		}`))
	})

	srv := httptest.NewServer(mux)
	defer srv.Close()

	p, err := New(Config{
		Name:          "xai",
		BaseURL:       "https://api.x.ai",
		DataDir:       t.TempDir(),
		HTTPClient:    srv.Client(),
		DeviceAuthURL: srv.URL + "/device",
		TokenURL:      srv.URL + "/token",
		Quirks: Quirks{
			AuthViaOAuthManager: true,
		},
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	info, err := p.StartLogin(context.Background())
	if err != nil {
		t.Fatalf("StartLogin failed: %v", err)
	}

	// The device-code request must carry the full xAI scope, otherwise the
	// minted token lacks "api:access" and every chat call 403s.
	if got := deviceForm.Get("scope"); got != defaultXAIScope {
		t.Errorf("device code scope = %q, want %q", got, defaultXAIScope)
	}
	for _, want := range []string{"api:access", "grok-cli:access", "offline_access"} {
		if !strings.Contains(deviceForm.Get("scope"), want) {
			t.Errorf("device code scope %q missing %q", deviceForm.Get("scope"), want)
		}
	}
	if got := deviceForm.Get("client_id"); got != defaultXAIClientID {
		t.Errorf("device code client_id = %q, want %q", got, defaultXAIClientID)
	}

	if info.Kind != provider.LoginFlowDevice {
		t.Errorf("LoginStartInfo.Kind = %v, want %v", info.Kind, provider.LoginFlowDevice)
	}
	if info.UserCode != "START-4567" {
		t.Errorf("LoginStartInfo.UserCode = %q, want START-4567", info.UserCode)
	}
	if info.VerificationURI != "https://accounts.x.ai/oauth2/device" {
		t.Errorf("LoginStartInfo.VerificationURI = %q", info.VerificationURI)
	}
	if info.ExpiresIn != 600 {
		t.Errorf("LoginStartInfo.ExpiresIn = %d, want 600", info.ExpiresIn)
	}
}

// TestOpenAICompat_VersionedBaseURLKeepsVersionSegment guards the z.ai lane:
// llmhub's openaichat.EnsureV1Suffix used to append "/v1" to any base URL not
// already ending in "/v1", which turned https://api.z.ai/api/paas/v4 into
// .../v4/v1/chat/completions (upstream 404). A base URL that already pins a
// version segment (/v4) must be used verbatim; a base URL with no version
// segment still gets "/v1" appended.
func TestOpenAICompat_VersionedBaseURLKeepsVersionSegment(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string // path shape handed to Config.BaseURL (host is the fake upstream)
		wantPath string
	}{
		{"zai_v4", "/api/paas/v4", "/api/paas/v4/chat/completions"},
		{"zai_v4_trailing_slash", "/api/paas/v4/", "/api/paas/v4/chat/completions"},
		{"other_version", "/v2", "/v2/chat/completions"},
		{"unversioned_gets_v1", "", "/v1/chat/completions"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotPath string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id": "cmpl-versioned",
					"choices": []map[string]any{
						{
							"index":         0,
							"message":       map[string]any{"role": "assistant", "content": "ok"},
							"finish_reason": "stop",
						},
					},
				})
			}))
			defer server.Close()

			p, err := New(Config{
				Name:       "zai",
				BaseURL:    server.URL + tt.baseURL,
				APIKey:     "test-zai-key",
				HTTPClient: server.Client(),
			})
			if err != nil {
				t.Fatalf("New failed: %v", err)
			}

			if _, err := p.Generate(context.Background(), []*ir.Message{
				{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}},
			}, provider.WithModel("glm-4.6")); err != nil {
				t.Fatalf("Generate failed: %v", err)
			}
			if gotPath != tt.wantPath {
				t.Fatalf("upstream path = %q, want %q (base URL %q must not gain an extra /v1)",
					gotPath, tt.wantPath, tt.baseURL)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Reactive automatic token refresh (T029 AC2)
// ---------------------------------------------------------------------------

// scriptedAuthUpstream is a fake OpenAI-compatible upstream whose chat
// endpoint answers from a script of status codes (consumed in order, the last
// one repeating). It records the Authorization header of every chat request so
// tests can prove both HOW MANY upstream requests a rejected credential costs
// and WHICH credential each one carried.
type scriptedAuthUpstream struct {
	srv *httptest.Server

	mu       sync.Mutex
	statuses []int
	calls    int
	auths    []string
}

// xaiBadCredentialsBody is the live failure that motivated this task: xai
// answers a dead OAuth access token with 403 + this body.
const xaiBadCredentialsBody = `{"code":"unauthenticated:bad-credentials","error":"The OAuth2 access token could not be validated."}`

func newScriptedAuthUpstream(t *testing.T, statuses ...int) *scriptedAuthUpstream {
	t.Helper()
	u := &scriptedAuthUpstream{statuses: statuses}
	u.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			http.NotFound(w, r)
			return
		}
		u.mu.Lock()
		idx := u.calls
		if idx >= len(u.statuses) {
			idx = len(u.statuses) - 1
		}
		status := u.statuses[idx]
		u.calls++
		u.auths = append(u.auths, r.Header.Get("Authorization"))
		u.mu.Unlock()

		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(xaiBadCredentialsBody))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-refresh-1",
			"choices": []map[string]any{
				{
					"index":         0,
					"message":       map[string]any{"role": "assistant", "content": "refreshed answer"},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
		})
	}))
	t.Cleanup(u.srv.Close)
	return u
}

func (u *scriptedAuthUpstream) URL() string { return u.srv.URL }

func (u *scriptedAuthUpstream) setStatuses(statuses ...int) {
	u.mu.Lock()
	u.statuses = statuses
	u.mu.Unlock()
}

func (u *scriptedAuthUpstream) chatCalls() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.calls
}

func (u *scriptedAuthUpstream) authHeader(i int) string {
	u.mu.Lock()
	defer u.mu.Unlock()
	if i < 0 || i >= len(u.auths) {
		return ""
	}
	return u.auths[i]
}

// newRefreshTestProvider builds a lane whose credential lives in a real
// auth.Manager over a temp dir, with a refresher that mints a distinct access
// token per refresh, so a retry with a FRESH credential is distinguishable from
// a retry that replayed the rejected one.
func newRefreshTestProvider(t *testing.T, baseURL string, client *http.Client, refreshes *atomic.Int64) *Provider {
	t.Helper()
	base := time.Now().UTC()
	mgr, err := auth.NewManager(t.TempDir(), func(ctx context.Context, cred auth.Credential) (auth.Credential, error) {
		n := refreshes.Add(1)
		return auth.Credential{
			Provider:     "xai",
			AccessToken:  fmt.Sprintf("token-%d", n),
			RefreshToken: fmt.Sprintf("rt-%d", n),
			ExpiresAt:    base.Add(time.Hour),
			Generation:   cred.Generation + 1,
			ClientID:     defaultXAIClientID,
		}, nil
	})
	if err != nil {
		t.Fatalf("auth.NewManager: %v", err)
	}
	if err := mgr.Store(context.Background(), defaultXAIClientID, auth.Credential{
		Provider:    "xai",
		AccessToken: "token-0",
		ExpiresAt:   base.Add(time.Hour), // still valid: only a 401/403 may force a refresh
		Generation:  1,
		ClientID:    defaultXAIClientID,
	}); err != nil {
		t.Fatalf("seed credential: %v", err)
	}

	p, err := New(Config{
		Name:        "xai",
		BaseURL:     baseURL,
		HTTPClient:  client,
		DataDir:     t.TempDir(),
		TokenSource: NewOAuthManagerTokenSource(mgr, defaultXAIClientID),
		Quirks:      Quirks{AuthViaOAuthManager: true},
	})
	if err != nil {
		t.Fatalf("openaicompat.New: %v", err)
	}
	return p
}

var refreshTestMsgs = []*ir.Message{
	{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hello"}}},
}

// AC2 (happy half): ONE upstream 401 must cost exactly one Invalidate + one
// Token + ONE retry. The client sees the successful response and the upstream
// saw exactly 2 requests, the second carrying the refreshed credential.
func TestOpenAICompat_ReactiveRefreshRetriesOnceOn401(t *testing.T) {
	var refreshes atomic.Int64
	up := newScriptedAuthUpstream(t, http.StatusUnauthorized, http.StatusOK)
	p := newRefreshTestProvider(t, up.URL(), up.srv.Client(), &refreshes)

	resp, err := p.Generate(context.Background(), refreshTestMsgs, provider.WithModel("grok-4.6"))
	if err != nil {
		t.Fatalf("Generate after a single 401: %v", err)
	}
	if got := blockText(t, resp); got != "refreshed answer" {
		t.Errorf("Generate content = %q, want the retried response", got)
	}
	if got := up.chatCalls(); got != 2 {
		t.Fatalf("upstream chat requests = %d, want exactly 2 (first rejected + one retry)", got)
	}
	if got, want := up.authHeader(0), "Bearer token-0"; got != want {
		t.Errorf("first upstream Authorization = %q, want %q", got, want)
	}
	if got, want := up.authHeader(1), "Bearer token-1"; got != want {
		t.Errorf("retry upstream Authorization = %q, want the refreshed %q", got, want)
	}
	if got := refreshes.Load(); got != 1 {
		t.Errorf("credential refreshes = %d, want 1", got)
	}
}

// AC2 (xai shape): the live failure is a 403 unauthenticated:bad-credentials,
// so the retry must cover 403 exactly like 401.
func TestOpenAICompat_ReactiveRefreshRetriesOnceOn403(t *testing.T) {
	var refreshes atomic.Int64
	up := newScriptedAuthUpstream(t, http.StatusForbidden, http.StatusOK)
	p := newRefreshTestProvider(t, up.URL(), up.srv.Client(), &refreshes)

	if _, err := p.Generate(context.Background(), refreshTestMsgs, provider.WithModel("grok-4.6")); err != nil {
		t.Fatalf("Generate after a single 403: %v", err)
	}
	if got := up.chatCalls(); got != 2 {
		t.Fatalf("upstream chat requests = %d, want exactly 2", got)
	}
	if got := refreshes.Load(); got != 1 {
		t.Errorf("credential refreshes = %d, want 1", got)
	}
}

// AC2 (honest failure): a SECOND consecutive auth rejection is returned to the
// caller verbatim. No third upstream request, no retry loop.
func TestOpenAICompat_ReactiveRefreshSecondFailureIsHonest(t *testing.T) {
	var refreshes atomic.Int64
	up := newScriptedAuthUpstream(t, http.StatusUnauthorized)
	p := newRefreshTestProvider(t, up.URL(), up.srv.Client(), &refreshes)

	_, err := p.Generate(context.Background(), refreshTestMsgs, provider.WithModel("grok-4.6"))
	if err == nil {
		t.Fatal("Generate with a persistently rejected credential returned nil error")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("error = %v, want the upstream 401 surfaced honestly", err)
	}
	if got := up.chatCalls(); got != 2 {
		t.Fatalf("upstream chat requests = %d, want exactly 2 (initial + one retry), never a third", got)
	}
	if got := refreshes.Load(); got != 1 {
		t.Errorf("credential refreshes = %d, want 1", got)
	}
}

// AC2 (xai honest failure): the 403 twin of the honest-failure case - two
// consecutive 403s cost exactly two upstream requests and the second one is
// returned to the caller, never retried again.
func TestOpenAICompat_ReactiveRefreshSecond403IsHonest(t *testing.T) {
	var refreshes atomic.Int64
	up := newScriptedAuthUpstream(t, http.StatusForbidden)
	p := newRefreshTestProvider(t, up.URL(), up.srv.Client(), &refreshes)

	_, err := p.Generate(context.Background(), refreshTestMsgs, provider.WithModel("grok-4.6"))
	if err == nil {
		t.Fatal("Generate with a persistently 403-ing credential returned nil error")
	}
	if !strings.Contains(err.Error(), "403") {
		t.Errorf("error = %v, want the upstream 403 surfaced honestly", err)
	}
	if got := up.chatCalls(); got != 2 {
		t.Fatalf("upstream chat requests = %d, want exactly 2 (initial + one retry), never a third", got)
	}
	if got := refreshes.Load(); got != 1 {
		t.Errorf("credential refreshes = %d, want 1", got)
	}
}

// AC2 (stream): the retry also covers streaming, and it sits strictly before
// the first byte - the client sees one clean stream, the upstream saw exactly
// two chat requests.
func TestOpenAICompat_ReactiveRefreshRetriesStreamBeforeFirstByte(t *testing.T) {
	var refreshes atomic.Int64
	up := newSSEAuthUpstream(t, http.StatusUnauthorized, http.StatusOK)
	p := newRefreshTestProvider(t, up.URL(), up.srv.Client(), &refreshes)

	ch, err := p.Stream(context.Background(), refreshTestMsgs, provider.WithModel("grok-4.6"))
	if err != nil {
		t.Fatalf("Stream after a single 401: %v", err)
	}
	var text strings.Builder
	for ev := range ch {
		if d, ok := ev.(ir.EventTextDelta); ok {
			text.WriteString(d.Text)
		}
		if e, ok := ev.(ir.EventUpstreamError); ok {
			t.Fatalf("stream surfaced an upstream error: %s", e.Message)
		}
	}
	if text.String() != "streamed answer" {
		t.Errorf("streamed text = %q, want %q", text.String(), "streamed answer")
	}
	if got := up.chatCalls(); got != 2 {
		t.Fatalf("upstream chat requests = %d, want exactly 2", got)
	}
	if got := refreshes.Load(); got != 1 {
		t.Errorf("credential refreshes = %d, want 1", got)
	}
}

// AC2 (stream boundary): once a stream has produced its first byte there is no
// retry at all - the failure is surfaced, and the upstream never sees a second
// dial for that request.
func TestOpenAICompat_ReactiveRefreshNeverAfterFirstStreamByte(t *testing.T) {
	var refreshes atomic.Int64
	up := newBrokenSSEUpstream(t)
	p := newRefreshTestProvider(t, up.URL(), up.srv.Client(), &refreshes)

	ch, err := p.Stream(context.Background(), refreshTestMsgs, provider.WithModel("grok-4.6"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var (
		sawDelta bool
		sawError bool
	)
	for ev := range ch {
		if _, ok := ev.(ir.EventTextDelta); ok {
			sawDelta = true
		}
		if _, ok := ev.(ir.EventUpstreamError); ok {
			sawError = true
		}
	}
	if !sawDelta {
		t.Error("expected at least one text delta before the stream broke")
	}
	if !sawError {
		t.Error("expected the mid-stream failure to surface as an upstream error event")
	}
	if got := up.chatCalls(); got != 1 {
		t.Fatalf("upstream chat requests = %d, want exactly 1: no retry after the first byte", got)
	}
	if got := refreshes.Load(); got != 0 {
		t.Errorf("credential refreshes = %d, want 0: a mid-stream break is not an auth failure", got)
	}
}

// sseAuthUpstream: the scripted-status fake, but a 200 answers with a complete
// SSE chat-completion stream.
type sseAuthUpstream struct {
	srv *httptest.Server

	mu       sync.Mutex
	statuses []int
	calls    int
	auths    []string
}

func newSSEAuthUpstream(t *testing.T, statuses ...int) *sseAuthUpstream {
	t.Helper()
	u := &sseAuthUpstream{statuses: statuses}
	u.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			http.NotFound(w, r)
			return
		}
		u.mu.Lock()
		idx := u.calls
		if idx >= len(u.statuses) {
			idx = len(u.statuses) - 1
		}
		status := u.statuses[idx]
		u.calls++
		u.auths = append(u.auths, r.Header.Get("Authorization"))
		u.mu.Unlock()

		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(xaiBadCredentialsBody))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: {\"id\":\"cmpl-sse-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"streamed answer\"}}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"cmpl-sse-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(u.srv.Close)
	return u
}

func (u *sseAuthUpstream) URL() string { return u.srv.URL }

func (u *sseAuthUpstream) chatCalls() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.calls
}

// newBrokenSSEUpstream answers 200, emits ONE text delta and then aborts the
// connection: a failure that happens after the first byte of a stream.
func newBrokenSSEUpstream(t *testing.T) *sseAuthUpstream {
	t.Helper()
	u := &sseAuthUpstream{statuses: []int{http.StatusOK}}
	u.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			http.NotFound(w, r)
			return
		}
		u.mu.Lock()
		u.calls++
		u.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: {\"id\":\"cmpl-broken-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		panic(http.ErrAbortHandler)
	}))
	t.Cleanup(u.srv.Close)
	return u
}

func blockText(t *testing.T, resp *ir.Response) string {
	t.Helper()
	if resp == nil || resp.Message == nil {
		return ""
	}
	var sb strings.Builder
	for _, b := range resp.Message.Blocks {
		if tb, ok := b.(ir.TextBlock); ok {
			sb.WriteString(tb.Text)
		}
	}
	return sb.String()
}

// syncLogBuffer is a mutex-guarded log sink, so a test can read what was logged
// while other goroutines may still be writing.
type syncLogBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncLogBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncLogBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// AC3: the reactive refresh log line carries the lane and the new expiry only.
// Everything logged while a rejected credential is refreshed is grepped for
// token-shaped strings - the in-play secrets and generic secret shapes.
func TestOpenAICompat_ReactiveRefreshLogsNoSecrets(t *testing.T) {
	var refreshes atomic.Int64
	up := newScriptedAuthUpstream(t, http.StatusUnauthorized, http.StatusOK)
	p := newRefreshTestProvider(t, up.URL(), up.srv.Client(), &refreshes)

	var buf syncLogBuffer
	log.SetOutput(&buf)
	log.SetFlags(0)
	defer func() {
		log.SetOutput(os.Stderr)
		log.SetFlags(log.LstdFlags)
	}()

	if _, err := p.Generate(context.Background(), refreshTestMsgs, provider.WithModel("grok-4.6")); err != nil {
		t.Fatalf("Generate after a single 401: %v", err)
	}
	if got := refreshes.Load(); got != 1 {
		t.Fatalf("credential refreshes = %d, want 1", got)
	}

	logged := buf.String()
	if !strings.Contains(logged, "credential refreshed") {
		t.Fatalf("no refresh log line was written; log was:\n%s", logged)
	}
	for _, line := range strings.Split(logged, "\n") {
		if !strings.Contains(line, "refresh") {
			continue
		}
		for _, secret := range []string{"token-0", "token-1", "rt-0", "rt-1"} {
			if strings.Contains(line, secret) {
				t.Errorf("reactive refresh leak: line %q contains the in-play secret %q", line, secret)
			}
		}
		for _, shape := range []*regexp.Regexp{
			regexp.MustCompile(`eyJ[A-Za-z0-9_-]{8,}`),
			regexp.MustCompile(`(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}`),
			regexp.MustCompile(`[0-9a-f]{32,}`),
			regexp.MustCompile(`[A-Za-z0-9+/]{40,}={0,2}`),
		} {
			if shape.MatchString(line) {
				t.Errorf("reactive refresh leak: line %q matches token shape %v", line, shape)
			}
		}
	}
}
