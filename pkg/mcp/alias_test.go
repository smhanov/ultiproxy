package mcp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// stubAliasManager implements AliasManager for tests.
type stubAliasManager struct {
	aliases map[string]ModelAlias
}

func (s *stubAliasManager) List() map[string]ModelAlias {
	out := map[string]ModelAlias{}
	for k, v := range s.aliases {
		out[k] = v
	}
	return out
}
func (s *stubAliasManager) Sorted() []string { return []string{"a"} }
func (s *stubAliasManager) Set(alias string, e ModelAlias) error {
	if s.aliases == nil {
		s.aliases = map[string]ModelAlias{}
	}
	s.aliases[alias] = e
	return nil
}
func (s *stubAliasManager) Remove(alias string) error { delete(s.aliases, alias); return nil }

func TestMCPAliasTools(t *testing.T) {
	am := &stubAliasManager{aliases: map[string]ModelAlias{}}
	srv := NewServer(nil, nil, WithAliasManager(am))

	// set_model_alias
	reqBody := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"set_model_alias","arguments":{"alias":"qwenpoint-3.8","provider":"vllm","upstream":"Qwen/Qwen3.8-Instruct-AWQ"}}}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(reqBody)))
	if rec.Code != http.StatusOK {
		t.Fatalf("set_model_alias: %d", rec.Code)
	}
	var resp JSONRPCResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Error != nil {
		t.Fatalf("set_model_alias error: %+v", resp.Error)
	}
	if _, ok := am.aliases["qwenpoint-3.8"]; !ok {
		t.Fatal("alias not set on manager")
	}

	// list_model_aliases
	reqBody = `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_model_aliases","arguments":{}}}`
	rec = httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(reqBody)))
	var listResp JSONRPCResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &listResp); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	if !strings.Contains(rec.Body.String(), "qwenpoint-3.8") || !strings.Contains(rec.Body.String(), "Qwen/Qwen3.8-Instruct-AWQ") {
		t.Fatalf("list_model_aliases missing alias: %s", rec.Body.String())
	}

	// remove_model_alias
	reqBody = `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"remove_model_alias","arguments":{"alias":"qwenpoint-3.8"}}}`
	rec = httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(reqBody)))
	if _, ok := am.aliases["qwenpoint-3.8"]; ok {
		t.Fatal("alias not removed on manager")
	}
}
