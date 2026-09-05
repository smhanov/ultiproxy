package mcp

import (
	"encoding/json"
	"fmt"
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

// TestMCPAliasLifecycleControlSurfaces pins the two MCP control surfaces the
// router's alias lifecycle depends on, over the wire:
//
//   - toggle_model must land as enabled=false on the state snapshot the router
//     reads (that flag is what makes a disabled alias unroutable), and
//   - remove_model_alias must delete the mapping from the alias manager the
//     catalog is reconciled against, immediately and without a restart.
//
// The pkg/server tests prove the routing consequences; this test proves the
// MCP surface reports and mutates exactly those inputs.
func TestMCPAliasLifecycleControlSurfaces(t *testing.T) {
	am := &stubAliasManager{aliases: map[string]ModelAlias{}}
	src := newStubStateSource()
	srv := NewServer(nil, src, WithAliasManager(am))

	call := func(id int, tool, args string) string {
		t.Helper()
		body := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":"tools/call","params":{"name":%q,"arguments":%s}}`, id, tool, args)
		rec := httptest.NewRecorder()
		srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: HTTP %d: %s", tool, rec.Code, rec.Body.String())
		}
		var resp struct {
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
			Result struct {
				IsError bool `json:"isError"`
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"result"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("%s: decode: %v (%s)", tool, err, rec.Body.String())
		}
		if resp.Error != nil {
			t.Fatalf("%s: JSON-RPC error: %s", tool, resp.Error.Message)
		}
		var text strings.Builder
		for _, c := range resp.Result.Content {
			text.WriteString(c.Text)
		}
		if resp.Result.IsError {
			t.Fatalf("%s: tool reported an error: %s", tool, text.String())
		}
		return text.String()
	}

	call(1, "set_model_alias", `{"alias":"qwenpoint-3.8","provider":"vllm","upstream":"Qwen/Qwen3.8-Instruct-AWQ"}`)

	// toggle_model(false) must be visible in the routing state snapshot.
	call(2, "toggle_model", `{"model_id":"qwenpoint-3.8","enabled":false}`)
	if m, ok := src.Snapshot().Models["qwenpoint-3.8"]; !ok || m.Enabled {
		t.Fatalf("toggle_model(false) did not disable the id in the state snapshot: %+v (present=%v)", m, ok)
	}
	if !strings.Contains(call(3, "list_models", `{}`), `"enabled": false`) {
		t.Fatal("list_models does not report the disabled id")
	}

	// Toggling an unrelated id back on must not touch the first one.
	call(4, "toggle_model", `{"model_id":"gpt-4o","enabled":true}`)
	if m, ok := src.Snapshot().Models["qwenpoint-3.8"]; !ok || m.Enabled {
		t.Fatalf("toggling another id re-enabled the disabled alias: %+v (present=%v)", m, ok)
	}

	// remove_model_alias must drop the mapping immediately.
	call(5, "remove_model_alias", `{"alias":"qwenpoint-3.8"}`)
	if _, ok := am.aliases["qwenpoint-3.8"]; ok {
		t.Fatal("remove_model_alias left the alias on the manager")
	}
	if strings.Contains(call(6, "list_model_aliases", `{}`), "qwenpoint-3.8") {
		t.Fatal("list_model_aliases still reports the removed alias")
	}
}
