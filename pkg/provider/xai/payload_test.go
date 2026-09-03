package xai

import (
	"encoding/json"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestBuildPayloadForwardsToolsAndToolResults(t *testing.T) {
	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "list files"}}},
		{Role: "assistant", Blocks: []ir.Block{
			ir.ToolCallBlock{ID: "call_bash_1", Name: "bash", Arguments: `{"command":"ls"}`},
		}},
		{Role: "tool", Blocks: []ir.Block{
			ir.ToolResultBlock{ToolCallID: "call_bash_1", Name: "bash", Content: "ok"},
		}},
	}
	cfg := provider.NewRequestConfig(
		provider.WithModel("grok-4.6"),
		provider.WithExtraBody(map[string]any{
			"tools": []any{
				map[string]any{"type": "function", "function": map[string]any{"name": "bash"}},
			},
		}),
	)
	payload, err := BuildPayload(msgs, cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := payload["tools"]; !ok {
		t.Fatal("expected tools in payload")
	}
	raw, _ := json.Marshal(payload["messages"])
	var chat []map[string]any
	if err := json.Unmarshal(raw, &chat); err != nil {
		t.Fatal(err)
	}
	if len(chat) != 3 {
		t.Fatalf("messages = %d, want 3: %s", len(chat), raw)
	}
	if chat[1]["role"] != "assistant" {
		t.Errorf("msg[1].role = %v", chat[1]["role"])
	}
	tcs, ok := chat[1]["tool_calls"].([]any)
	if !ok || len(tcs) != 1 {
		t.Fatalf("msg[1].tool_calls = %#v", chat[1]["tool_calls"])
	}
	if chat[2]["role"] != "tool" || chat[2]["tool_call_id"] != "call_bash_1" {
		t.Fatalf("msg[2] = %#v", chat[2])
	}
}
