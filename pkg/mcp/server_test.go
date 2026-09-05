package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
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

	if len(tools) != 16 {
		t.Fatalf("expected 16 tools, got %d", len(tools))
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
		"check_oauth_login",
		"submit_oauth_code",
		"list_model_aliases",
		"set_model_alias",
		"remove_model_alias",
		"get_provider_timeouts",
		"set_provider_timeout",
		"remove_provider_timeout",
		"add_provider",
		"remove_provider",
		"list_providers",
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

// fileProviderStore is a test ProviderStore that persists to a JSON file the
// same way the real RuntimeProviderStore (pkg/server/providers.go) does.
// pkg/server cannot be imported from here (server imports mcp), so the store
// is reimplemented in miniature for these tests.
type fileProviderStore struct {
	mu   sync.Mutex
	path string
	m    map[string]openaicompat.Config
}

func newFileProviderStore(path string) *fileProviderStore {
	return &fileProviderStore{path: path, m: map[string]openaicompat.Config{}}
}

func (s *fileProviderStore) Add(cfg openaicompat.Config) error {
	if cfg.Name == "" {
		return errTest("name is required")
	}
	if cfg.BaseURL == "" {
		return errTest("base_url is required")
	}
	s.mu.Lock()
	s.m[cfg.Name] = cfg
	s.mu.Unlock()
	return s.persist()
}

// AddCustom stores a custom-kind lane in the in-memory test store.
func (s *fileProviderStore) AddCustom(name, kind, apiKey string) error {
	if name == "" {
		return errTest("name is required")
	}
	s.mu.Lock()
	s.m[name] = openaicompat.Config{Name: name, BaseURL: "custom://" + kind, APIKey: apiKey}
	s.mu.Unlock()
	return s.persist()
}

func (s *fileProviderStore) Remove(name string) error {
	s.mu.Lock()
	_, ok := s.m[name]
	delete(s.m, name)
	s.mu.Unlock()
	if !ok {
		return errTest("not stored: " + name)
	}
	return s.persist()
}

func (s *fileProviderStore) List() map[string]openaicompat.Config {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]openaicompat.Config, len(s.m))
	for k, v := range s.m {
		out[k] = v
	}
	return out
}

func (s *fileProviderStore) persist() error {
	s.mu.Lock()
	snapshot := make(map[string]openaicompat.Config, len(s.m))
	for k, v := range s.m {
		snapshot[k] = v
	}
	s.mu.Unlock()
	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

type errTest string

func (e errTest) Error() string { return string(e) }

// callMCPTool posts a tools/call request and decodes the CallToolResult.
func callMCPTool(t *testing.T, srv *Server, id int, name, arguments string) CallToolResult {
	t.Helper()
	body := `{"jsonrpc":"2.0","id":` + strconv.Itoa(id) + `,"method":"tools/call","params":{"name":"` + name + `","arguments":` + arguments + `}}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("%s: HTTP %d: %s", name, rec.Code, rec.Body.String())
	}
	var resp JSONRPCResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("%s: decode: %v", name, err)
	}
	if resp.Error != nil {
		t.Fatalf("%s: json-rpc error: %+v", name, resp.Error)
	}
	raw, _ := json.Marshal(resp.Result)
	var out CallToolResult
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("%s: decode call result: %v", name, err)
	}
	return out
}

func TestMCP_AddRemoveListProviders(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")
	store := newFileProviderStore(path)
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store))

	// add_provider: harmless fake upstream; New() does not dial on construction.
	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"vllm","base_url":"http://127.0.0.1:1/v1","api_key":"sk-supersecret","quirks":{"model_list_passthrough":false}}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}
	if !strings.Contains(res.Content[0].Text, `"registered": true`) ||
		!strings.Contains(res.Content[0].Text, `"lane": "vllm"`) {
		t.Fatalf("add_provider response unexpected: %s", res.Content[0].Text)
	}
	if _, ok := registry.Get("vllm"); !ok {
		t.Fatal("registry does not contain the vllm lane after add_provider")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("providers.json not persisted: %v", err)
	}
	onDisk, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}
	if !strings.Contains(string(onDisk), `"vllm"`) || !strings.Contains(string(onDisk), "127.0.0.1:1") {
		t.Fatalf("providers.json missing the lane: %s", onDisk)
	}

	// list_providers: lane present, api_key redacted.
	list := callMCPTool(t, srv, 2, "list_providers", `{}`)
	if list.IsError {
		t.Fatalf("list_providers failed: %s", list.Content[0].Text)
	}
	if !strings.Contains(list.Content[0].Text, `"vllm"`) {
		t.Fatalf("list_providers missing vllm: %s", list.Content[0].Text)
	}
	if strings.Contains(list.Content[0].Text, "sk-supersecret") {
		t.Fatalf("list_providers leaked the api key: %s", list.Content[0].Text)
	}
	if !strings.Contains(list.Content[0].Text, `"has_api_key": true`) {
		t.Fatalf("list_providers should report key presence: %s", list.Content[0].Text)
	}

	// remove_provider: registry + file updated.
	rem := callMCPTool(t, srv, 3, "remove_provider", `{"name":"vllm"}`)
	if rem.IsError {
		t.Fatalf("remove_provider failed: %s", rem.Content[0].Text)
	}
	if _, ok := registry.Get("vllm"); ok {
		t.Fatal("registry still contains vllm after remove_provider")
	}
	if registry.Len() != 0 {
		t.Fatalf("registry not empty after remove: %v", registry.Names())
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}
	if strings.Contains(string(after), `"vllm"`) {
		t.Fatalf("providers.json still contains vllm: %s", after)
	}

	// list_providers is now empty.
	list2 := callMCPTool(t, srv, 4, "list_providers", `{}`)
	if list2.IsError || strings.Contains(list2.Content[0].Text, `"vllm"`) {
		t.Fatalf("list_providers after remove unexpected: %s", list2.Content[0].Text)
	}

	// add_provider validation: missing base_url is a tool error, nothing registered.
	bad := callMCPTool(t, srv, 5, "add_provider", `{"name":"nope"}`)
	if !bad.IsError {
		t.Fatalf("expected error for missing base_url: %s", bad.Content[0].Text)
	}
	if _, ok := registry.Get("nope"); ok {
		t.Fatal("invalid lane was registered")
	}

	// removing an unknown lane errors.
	unknown := callMCPTool(t, srv, 6, "remove_provider", `{"name":"ghost"}`)
	if !unknown.IsError {
		t.Fatalf("expected error removing unknown lane: %s", unknown.Content[0].Text)
	}
}

// stubCustomLane is a minimal inference provider the custom-lane tests hand
// back from the injected builder.
type stubCustomLane struct {
	name string
}

func (s stubCustomLane) Name() string { return s.name }

func (s stubCustomLane) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	return &ir.Response{}, nil
}

func (s stubCustomLane) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	return nil, nil
}

// TestMCP_AddCustomProviderAnthropic covers kind=anthropic: the API key from
// the add_provider call must reach the injected builder and be persisted, and
// a missing key must be rejected before anything is registered.
func TestMCP_AddCustomProviderAnthropic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "providers.json")
	store := newFileProviderStore(path)
	registry := provider.NewRegistry()

	var gotKind, gotAPIKey string
	srv := NewServer(registry, nil,
		WithProviderStore(store),
		WithCustomLaneBuilder(func(name, kind, apiKey string) (provider.Provider, error) {
			gotKind, gotAPIKey = kind, apiKey
			return provider.Provider{Inference: stubCustomLane{name: name}}, nil
		}),
	)

	// kind=anthropic without an api_key is rejected up front.
	bad := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"anthropic","kind":"anthropic","base_url":"https://api.anthropic.com"}`)
	if !bad.IsError {
		t.Fatalf("expected error for anthropic without api_key: %s", bad.Content[0].Text)
	}
	if _, ok := registry.Get("anthropic"); ok {
		t.Fatal("anthropic lane registered without an api_key")
	}

	// With the key the lane builds, registers and persists.
	res := callMCPTool(t, srv, 2, "add_provider",
		`{"name":"anthropic","kind":"anthropic","base_url":"https://api.anthropic.com","api_key":"sk-ant-test"}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}
	if !strings.Contains(res.Content[0].Text, `"registered": true`) {
		t.Fatalf("add_provider response unexpected: %s", res.Content[0].Text)
	}
	if gotKind != "anthropic" {
		t.Fatalf("builder kind = %q, want anthropic", gotKind)
	}
	if gotAPIKey != "sk-ant-test" {
		t.Fatalf("builder api key = %q, want sk-ant-test", gotAPIKey)
	}
	if _, ok := registry.Get("anthropic"); !ok {
		t.Fatal("registry does not contain the anthropic lane")
	}
	if store.List()["anthropic"].APIKey != "sk-ant-test" {
		t.Fatalf("store did not persist the anthropic api key: %+v", store.List()["anthropic"])
	}
	onDisk, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}
	if !strings.Contains(string(onDisk), "sk-ant-test") {
		t.Fatalf("providers.json missing the anthropic api key: %s", onDisk)
	}

	// The key is never echoed back by list_providers.
	list := callMCPTool(t, srv, 3, "list_providers", `{}`)
	if list.IsError || strings.Contains(list.Content[0].Text, "sk-ant-test") {
		t.Fatalf("list_providers leaked the anthropic api key: %s", list.Content[0].Text)
	}
}

// TestMCP_AddCustomProviderCodex covers kind=codex: no api_key is required (the
// lane authenticates from the ultiproxy-owned credential store), and the lane
// still registers so quota can be read.
func TestMCP_AddCustomProviderCodex(t *testing.T) {
	dir := t.TempDir()
	store := newFileProviderStore(filepath.Join(dir, "providers.json"))
	registry := provider.NewRegistry()

	var gotKind, gotAPIKey string
	srv := NewServer(registry, nil,
		WithProviderStore(store),
		WithCustomLaneBuilder(func(name, kind, apiKey string) (provider.Provider, error) {
			gotKind, gotAPIKey = kind, apiKey
			return provider.Provider{Inference: stubCustomLane{name: name}}, nil
		}),
	)

	res := callMCPTool(t, srv, 1, "add_provider", `{"name":"codex","kind":"codex"}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}
	if !strings.Contains(res.Content[0].Text, `"kind": "codex"`) {
		t.Fatalf("add_provider response unexpected: %s", res.Content[0].Text)
	}
	if gotKind != "codex" {
		t.Fatalf("builder kind = %q, want codex", gotKind)
	}
	if gotAPIKey != "" {
		t.Fatalf("builder api key = %q, want empty", gotAPIKey)
	}
	if _, ok := registry.Get("codex"); !ok {
		t.Fatal("registry does not contain the codex lane")
	}
}

// TestMCP_AddProviderFreebuffQuirks covers the freebuff lane added as an
// OpenAI-compatible lane with quirks.freebuff_actor=true: the lane must carry
// the freebuff quirks (serialized requests, default tool, actor-backed quota)
// even though the actor itself cannot cross the MCP boundary.
func TestMCP_AddProviderFreebuffQuirks(t *testing.T) {
	dir := t.TempDir()
	store := newFileProviderStore(filepath.Join(dir, "providers.json"))
	registry := provider.NewRegistry()
	srv := NewServer(registry, nil, WithProviderStore(store))

	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"freebuff","base_url":"https://www.codebuff.com/api/v1","api_key":"fb-key","quirks":{"freebuff_actor":true,"freebuff_default_tool":true}}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}
	if !strings.Contains(res.Content[0].Text, `"registered": true`) {
		t.Fatalf("add_provider response unexpected: %s", res.Content[0].Text)
	}

	lane, ok := registry.Get("freebuff")
	if !ok {
		t.Fatal("registry does not contain the freebuff lane")
	}
	if lane.Inference == nil || lane.Inference.Name() != "freebuff" {
		t.Fatalf("freebuff lane is not an inference provider: %+v", lane)
	}
	// The actor marker keeps the quota surface attached and serializes requests.
	if lane.Quota == nil {
		t.Fatal("expected a Quota provider on the freebuff lane")
	}
	if lane.Capabilities.MaxConcurrentRequests != 1 {
		t.Errorf("MaxConcurrentRequests = %d, want 1 (serialized freebuff session)", lane.Capabilities.MaxConcurrentRequests)
	}

	stored := store.List()["freebuff"]
	if stored.APIKey != "fb-key" {
		t.Errorf("stored api key = %q, want fb-key", stored.APIKey)
	}
	if stored.Quirks.FreebuffActor == nil {
		t.Error("stored config lost the freebuff actor marker")
	}
	if !stored.Quirks.FreebuffDefaultTool {
		t.Error("stored config lost freebuff_default_tool")
	}

	// The flag is surfaced (as a boolean, never the actor itself) on listing.
	list := callMCPTool(t, srv, 2, "list_providers", `{}`)
	if list.IsError {
		t.Fatalf("list_providers failed: %s", list.Content[0].Text)
	}
	if !strings.Contains(list.Content[0].Text, `"freebuff_actor": true`) {
		t.Errorf("list_providers does not report freebuff_actor: %s", list.Content[0].Text)
	}
	if strings.Contains(list.Content[0].Text, "fb-key") {
		t.Errorf("list_providers leaked the api key: %s", list.Content[0].Text)
	}
}

// TestMCP_AddCustomProviderFreebuff covers kind=freebuff: the add_provider
// call must reach the injected builder with the lane kind and api key so the
// freebuff lane can be registered at runtime without any CLI or config file.
func TestMCP_AddCustomProviderFreebuff(t *testing.T) {
	dir := t.TempDir()
	store := newFileProviderStore(filepath.Join(dir, "providers.json"))
	registry := provider.NewRegistry()

	var gotName, gotKind, gotAPIKey string
	srv := NewServer(registry, nil,
		WithProviderStore(store),
		WithCustomLaneBuilder(func(name, kind, apiKey string) (provider.Provider, error) {
			gotName, gotKind, gotAPIKey = name, kind, apiKey
			return provider.Provider{Inference: stubCustomLane{name: name}}, nil
		}),
	)

	res := callMCPTool(t, srv, 1, "add_provider",
		`{"name":"freebuff","kind":"freebuff","api_key":"fb-kind-key"}`)
	if res.IsError {
		t.Fatalf("add_provider failed: %s", res.Content[0].Text)
	}
	if !strings.Contains(res.Content[0].Text, `"kind": "freebuff"`) {
		t.Fatalf("add_provider response unexpected: %s", res.Content[0].Text)
	}
	if gotName != "freebuff" || gotKind != "freebuff" {
		t.Fatalf("builder got name=%q kind=%q, want freebuff/freebuff", gotName, gotKind)
	}
	if gotAPIKey != "fb-kind-key" {
		t.Fatalf("builder api key = %q, want fb-kind-key", gotAPIKey)
	}
	if _, ok := registry.Get("freebuff"); !ok {
		t.Fatal("registry does not contain the freebuff lane")
	}
	if store.List()["freebuff"].BaseURL != "custom://freebuff" {
		t.Fatalf("custom lane not persisted: %+v", store.List()["freebuff"])
	}
}
