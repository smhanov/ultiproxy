package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
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

func TestOpenAICompat_OpenCodeCookieAuth(t *testing.T) {
	var capturedCookie string
	var capturedWorkspace string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		BaseURL:       server.URL,
		WorkspaceID:   "ws-corp-99",
		SessionCookie: "my-session-token-xyz",
		HTTPClient:    server.Client(),
		Quirks: Quirks{
			AuthViaWorkspaceCookie: true,
		},
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

	if capturedCookie != "session=my-session-token-xyz" {
		t.Errorf("expected session cookie header 'session=my-session-token-xyz', got %q", capturedCookie)
	}
	if capturedWorkspace != "ws-corp-99" {
		t.Errorf("expected workspace header 'ws-corp-99', got %q", capturedWorkspace)
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
	mu           sync.Mutex
	lock         sync.Mutex
	active       int
	maxActive    int
	acquireCount int
	instanceID   string
	token        string
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

	// Check codebuff_metadata
	meta, ok := firstPayload["codebuff_metadata"].(map[string]any)
	if !ok {
		t.Fatalf("expected codebuff_metadata in payload, got %v", firstPayload)
	}
	if meta["cost_mode"] != "free" {
		t.Errorf("expected cost_mode 'free', got %v", meta["cost_mode"])
	}
	if meta["freebuff_instance_id"] != "fb-inst-007" {
		t.Errorf("expected instance ID 'fb-inst-007', got %v", meta["freebuff_instance_id"])
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

	if len(snap.Windows) < 2 {
		t.Fatalf("expected at least 2 windows, got %d", len(snap.Windows))
	}
	if snap.Windows[0].Label != "5 hour" || snap.Windows[0].UsedPct != 15.0 {
		t.Errorf("unexpected 5-hour window: %+v", snap.Windows[0])
	}
	if snap.Windows[1].Label != "Weekly" || snap.Windows[1].UsedPct != 45.0 {
		t.Errorf("unexpected Weekly window: %+v", snap.Windows[1])
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
	manicodeFile := filepath.Join(tempDir, "credentials.json")
	credentialsData := `{"default": {"id": "user-1", "email": "test@example.com", "authToken": "secret-freebuff-tok"}}`
	if err := os.WriteFile(manicodeFile, []byte(credentialsData), 0600); err != nil {
		t.Fatalf("write credentials.json: %v", err)
	}

	t.Setenv("ULTIPROXY_MANICODE_CREDENTIALS", manicodeFile)

	actor := &fakeActor{instanceID: "fb-inst-007"}
	p, err := New(Config{
		BaseURL:    "https://codebuff.invalid",
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
	mux := http.NewServeMux()
	mux.HandleFunc("/device", func(w http.ResponseWriter, r *http.Request) {
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

	tok, err := p.Token(context.Background())
	if err != nil {
		t.Fatalf("Token() failed: %v", err)
	}
	if tok != "mock-xai-access-tok" {
		t.Errorf("Token() = %q, want mock-xai-access-tok", tok)
	}
}
