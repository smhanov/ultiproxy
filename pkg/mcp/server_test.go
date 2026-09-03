package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
)

type stubStateSource struct {
	mu       sync.Mutex
	snapshot *state.RuntimeSnapshot
}

func newStubStateSource() *stubStateSource {
	return &stubStateSource{
		snapshot: &state.RuntimeSnapshot{
			Providers: map[string]state.ProviderRuntime{
				"openai": {
					Admin:      state.AdminEnabled,
					Health:     state.HealthHealthy,
					Quota:      state.QuotaHealthy,
					Circuit:    state.CircuitClosed,
					Credential: state.CredentialValid,
				},
			},
			Models: map[string]state.ModelRuntime{
				"gpt-4o": {
					ID:       "gpt-4o",
					Provider: "openai",
					Enabled:  true,
				},
				"claude-3-7-sonnet": {
					ID:       "claude-3-7-sonnet",
					Provider: "anthropic",
					Enabled:  true,
				},
			},
		},
	}
}

func (s *stubStateSource) Snapshot() *state.RuntimeSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.snapshot.Clone()
}

func (s *stubStateSource) ToggleModel(modelID string, enabled bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if m, ok := s.snapshot.Models[modelID]; ok {
		m.Enabled = enabled
		s.snapshot.Models[modelID] = m
	} else {
		s.snapshot.Models[modelID] = state.ModelRuntime{
			ID:      modelID,
			Enabled: enabled,
		}
	}
	return nil
}

type fakeAuthProvider struct {
	name   string
	called bool
}

func (f *fakeAuthProvider) Name() string { return f.name }
func (f *fakeAuthProvider) Login(ctx context.Context) error {
	f.called = true
	return nil
}
func (f *fakeAuthProvider) Token(ctx context.Context) (string, error) { return "token", nil }
func (f *fakeAuthProvider) Refresh(ctx context.Context) error         { return nil }

type fakeQuotaProvider struct {
	name string
}

func (f *fakeQuotaProvider) Name() string { return f.name }
func (f *fakeQuotaProvider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	return &provider.QuotaSnapshot{
		ObservedAt: time.Now(),
		Windows: []provider.QuotaWindow{
			{Label: "Requests", UsedPct: 25.0, Limit: 1000, Remaining: 750, Unit: "requests"},
		},
	}, nil
}

func TestMCPInitialize(t *testing.T) {
	stateSrc := newStubStateSource()
	server := NewServer(nil, stateSrc)

	body := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}`
	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body))
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rec.Code)
	}

	var resp JSONRPCResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Error != nil {
		t.Fatalf("unexpected JSON-RPC error: %+v", resp.Error)
	}

	resMap, ok := resp.Result.(map[string]any)
	if !ok {
		t.Fatalf("expected map result, got %T", resp.Result)
	}
	if resMap["protocolVersion"] != "2024-11-05" {
		t.Errorf("expected protocolVersion 2024-11-05, got %v", resMap["protocolVersion"])
	}
}

func TestMCPToolsList(t *testing.T) {
	stateSrc := newStubStateSource()
	server := NewServer(nil, stateSrc)

	body := `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`
	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body))
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rec.Code)
	}

	var resp JSONRPCResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	resMap, ok := resp.Result.(map[string]any)
	if !ok {
		t.Fatalf("expected map result, got %T", resp.Result)
	}

	tools, ok := resMap["tools"].([]any)
	if !ok {
		t.Fatalf("expected tools list, got %T", resMap["tools"])
	}

	if len(tools) != 11 {
		t.Fatalf("expected 11 tools, got %d", len(tools))
	}

	toolNames := make(map[string]bool)
	for _, item := range tools {
		tMap := item.(map[string]any)
		toolNames[tMap["name"].(string)] = true
	}

	expectedTools := []string{
		"list_models",
		"get_quota_status",
		"toggle_model",
		"get_client_usage",
		"initiate_oauth_login",
		"list_model_aliases",
		"set_model_alias",
		"remove_model_alias",
		"get_provider_timeouts",
		"set_provider_timeout",
		"remove_provider_timeout",
	}

	for _, name := range expectedTools {
		if !toolNames[name] {
			t.Errorf("expected tool %q not found in tools/list", name)
		}
	}
}

func TestMCPToolCall_ListModelsAndToggleModel(t *testing.T) {
	stateSrc := newStubStateSource()
	server := NewServer(nil, stateSrc)

	// 1. Call list_models
	bodyList := `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_models","arguments":{}}}`
	reqList := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(bodyList))
	recList := httptest.NewRecorder()

	server.ServeHTTP(recList, reqList)

	var respList JSONRPCResponse
	if err := json.Unmarshal(recList.Body.Bytes(), &respList); err != nil {
		t.Fatalf("failed to decode list_models response: %v", err)
	}

	resultBytes, _ := json.Marshal(respList.Result)
	var callResult CallToolResult
	if err := json.Unmarshal(resultBytes, &callResult); err != nil {
		t.Fatalf("failed to unmarshal call result: %v", err)
	}

	if len(callResult.Content) == 0 || !strings.Contains(callResult.Content[0].Text, "gpt-4o") {
		t.Errorf("list_models did not return gpt-4o: %+v", callResult)
	}

	// 2. Call toggle_model (disable gpt-4o)
	bodyToggle := `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"toggle_model","arguments":{"model_id":"gpt-4o","enabled":false}}}`
	reqToggle := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(bodyToggle))
	recToggle := httptest.NewRecorder()

	server.ServeHTTP(recToggle, reqToggle)

	var respToggle JSONRPCResponse
	if err := json.Unmarshal(recToggle.Body.Bytes(), &respToggle); err != nil {
		t.Fatalf("failed to decode toggle_model response: %v", err)
	}

	resultToggleBytes, _ := json.Marshal(respToggle.Result)
	var toggleResult CallToolResult
	if err := json.Unmarshal(resultToggleBytes, &toggleResult); err != nil {
		t.Fatalf("failed to unmarshal toggle result: %v", err)
	}

	if toggleResult.IsError || !strings.Contains(toggleResult.Content[0].Text, `"enabled": false`) {
		t.Errorf("toggle_model failed or result unexpected: %+v", toggleResult)
	}

	// Verify state updated
	snap := stateSrc.Snapshot()
	if snap.Models["gpt-4o"].Enabled != false {
		t.Errorf("expected gpt-4o to be disabled in state, got enabled")
	}
}

func TestMCPToolCall_QuotaAndOAuth(t *testing.T) {
	registry := provider.NewRegistry()
	fakeAuth := &fakeAuthProvider{name: "copilot"}
	fakeQuota := &fakeQuotaProvider{name: "copilot"}
	registry.Register(provider.Provider{
		Auth:  fakeAuth,
		Quota: fakeQuota,
	})

	stateSrc := newStubStateSource()
	server := NewServer(registry, stateSrc)

	// Test get_quota_status
	bodyQuota := `{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"get_quota_status","arguments":{"provider":"copilot"}}}`
	reqQuota := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(bodyQuota))
	recQuota := httptest.NewRecorder()
	server.ServeHTTP(recQuota, reqQuota)

	var respQuota JSONRPCResponse
	_ = json.Unmarshal(recQuota.Body.Bytes(), &respQuota)
	b, _ := json.Marshal(respQuota.Result)
	var qResult CallToolResult
	_ = json.Unmarshal(b, &qResult)
	if qResult.IsError || !strings.Contains(qResult.Content[0].Text, "Requests") {
		t.Errorf("get_quota_status failed: %+v", qResult)
	}

	// Test initiate_oauth_login
	bodyAuth := `{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"initiate_oauth_login","arguments":{"provider":"copilot"}}}`
	reqAuth := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(bodyAuth))
	recAuth := httptest.NewRecorder()
	server.ServeHTTP(recAuth, reqAuth)

	var respAuth JSONRPCResponse
	_ = json.Unmarshal(recAuth.Body.Bytes(), &respAuth)
	bAuth, _ := json.Marshal(respAuth.Result)
	var aResult CallToolResult
	_ = json.Unmarshal(bAuth, &aResult)
	if aResult.IsError || !strings.Contains(aResult.Content[0].Text, "initiated") {
		t.Errorf("initiate_oauth_login failed: %+v", aResult)
	}
	if !fakeAuth.called {
		t.Errorf("expected Auth.Login to have been called")
	}
}

func TestMCPStreamableHTTP_GET(t *testing.T) {
	server := NewServer(nil, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req := httptest.NewRequest(http.MethodGet, "/mcp", nil).WithContext(ctx)
	rec := httptest.NewRecorder()

	// Cancel context after short delay to unblock GET handler
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rec.Code)
	}

	ct := rec.Header().Get("Content-Type")
	if !strings.Contains(ct, "text/event-stream") {
		t.Errorf("expected Content-Type text/event-stream, got %s", ct)
	}

	body := rec.Body.String()
	if !strings.Contains(body, "event: endpoint\ndata: /mcp") {
		t.Errorf("expected initial endpoint event, got:\n%s", body)
	}
}
