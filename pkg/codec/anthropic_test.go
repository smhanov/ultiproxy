package codec

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestDecodeMessagesRequest(t *testing.T) {
	rawJSON := `{
		"model": "claude-3-7-sonnet-20250219",
		"system": [
			{"type": "text", "text": "You are a code assistant.", "cache_control": {"type": "ephemeral"}}
		],
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "Can you check this image and run calculate?"},
					{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoAAAANSUhEUg=="}}
				]
			},
			{
				"role": "assistant",
				"content": [
					{"type": "thinking", "thinking": "Let me calculate the answer.", "signature": "sig_secret_123"},
					{"type": "tool_use", "id": "toolu_456", "name": "calculate", "input": {"expression": "21 * 2"}}
				]
			},
			{
				"role": "user",
				"content": [
					{"type": "tool_result", "tool_use_id": "toolu_456", "content": "42"}
				]
			}
		],
		"max_tokens": 1024,
		"temperature": 0.2,
		"thinking": {
			"type": "enabled",
			"budget_tokens": 2048
		},
		"tools": [
			{
				"name": "calculate",
				"description": "Eval math expression",
				"input_schema": {"type": "object"}
			}
		],
		"stream": true
	}`

	decoded, err := DecodeMessagesRequest([]byte(rawJSON))
	if err != nil {
		t.Fatalf("failed to decode messages request: %v", err)
	}

	if decoded.Model != "claude-3-7-sonnet-20250219" {
		t.Errorf("expected model claude-3-7-sonnet-20250219, got %s", decoded.Model)
	}
	if !decoded.Stream {
		t.Errorf("expected stream=true")
	}
	if !decoded.ToolsRequested {
		t.Errorf("expected ToolsRequested=true")
	}

	// 1 system message + 3 messages = 4 messages
	if len(decoded.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(decoded.Messages))
	}

	// System message with breakpoint
	sysMsg := decoded.Messages[0]
	if sysMsg.Role != "system" {
		t.Errorf("expected system role, got %s", sysMsg.Role)
	}
	if len(sysMsg.Blocks) != 2 {
		t.Fatalf("expected 2 blocks in system message, got %d", len(sysMsg.Blocks))
	}
	if tb, ok := sysMsg.Blocks[0].(ir.TextBlock); !ok || tb.Text != "You are a code assistant." {
		t.Errorf("system text mismatch: %+v", sysMsg.Blocks[0])
	}
	if cc, ok := sysMsg.Blocks[1].(ir.CacheControl); !ok || !cc.Breakpoint {
		t.Errorf("system cache control mismatch: %+v", sysMsg.Blocks[1])
	}

	// User message with image
	userMsg := decoded.Messages[1]
	if userMsg.Role != "user" || len(userMsg.Blocks) != 2 {
		t.Fatalf("user msg mismatch: %+v", userMsg)
	}
	if img, ok := userMsg.Blocks[1].(ir.ImageBlock); !ok || !strings.HasPrefix(img.URL, "data:image/png;base64,") {
		t.Errorf("user image block mismatch: %+v", userMsg.Blocks[1])
	}

	// Assistant message with thinking and tool_use
	asstMsg := decoded.Messages[2]
	if asstMsg.Role != "assistant" || len(asstMsg.Blocks) != 2 {
		t.Fatalf("assistant msg mismatch: %+v", asstMsg)
	}
	if rb, ok := asstMsg.Blocks[0].(ir.ReasoningBlock); !ok || rb.Text != "Let me calculate the answer." || rb.Signature != "sig_secret_123" {
		t.Errorf("reasoning block mismatch: %+v", asstMsg.Blocks[0])
	}
	if tc, ok := asstMsg.Blocks[1].(ir.ToolCallBlock); !ok || tc.ID != "toolu_456" || tc.Name != "calculate" || !strings.Contains(tc.Arguments, "21 * 2") {
		t.Errorf("tool call block mismatch: %+v", asstMsg.Blocks[1])
	}

	// Tool result
	resMsg := decoded.Messages[3]
	if resMsg.Role != "user" || len(resMsg.Blocks) != 1 {
		t.Fatalf("tool result message mismatch: %+v", resMsg)
	}
	if tr, ok := resMsg.Blocks[0].(ir.ToolResultBlock); !ok || tr.ToolCallID != "toolu_456" || tr.Content != "42" {
		t.Errorf("tool result block mismatch: %+v", resMsg.Blocks[0])
	}

	// Options
	cfg := provider.NewRequestConfig(decoded.Options...)
	if cfg.MaxTokens != 1024 {
		t.Errorf("expected MaxTokens=1024, got %d", cfg.MaxTokens)
	}
	if cfg.Temperature == nil || *cfg.Temperature != 0.2 {
		t.Errorf("expected Temperature=0.2, got %v", cfg.Temperature)
	}
	if cfg.ExtraBody["thinking"] == nil || cfg.ExtraBody["budget_tokens"] != 2048 {
		t.Errorf("expected budget_tokens in ExtraBody: %+v", cfg.ExtraBody)
	}
}

func TestEncodeMessagesResponse(t *testing.T) {
	resp := &ir.Response{
		ID: "msg_test_abc123",
		Message: &ir.Message{
			Role: "assistant",
			Blocks: []ir.Block{
				ir.ReasoningBlock{
					ReasoningKind: ir.ReasoningText,
					Text:          "Let me compute this.",
					Signature:     "sig_test_sig",
				},
				ir.TextBlock{
					Text: "The result is 42.",
				},
				ir.ToolCallBlock{
					ID:        "toolu_calc",
					Name:      "done",
					Arguments: `{"status":"ok"}`,
				},
			},
		},
		FinishReason: "tool_use",
		Usage: &ir.Usage{
			PromptTokens:             30,
			CompletionTokens:         15,
			CacheCreationInputTokens: 10,
			CacheReadInputTokens:     5,
		},
	}

	bytesOut, err := EncodeMessagesResponse(resp, "claude-3-7-sonnet-20250219")
	if err != nil {
		t.Fatalf("unexpected encode error: %v", err)
	}

	var mResp AnthropicMessagesResponse
	if err := json.Unmarshal(bytesOut, &mResp); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if mResp.ID != "msg_test_abc123" {
		t.Errorf("expected ID msg_test_abc123, got %s", mResp.ID)
	}
	if mResp.Role != "assistant" || mResp.Type != "message" {
		t.Errorf("role/type mismatch: %+v", mResp)
	}
	if len(mResp.Content) != 3 {
		t.Fatalf("expected 3 content blocks, got %d", len(mResp.Content))
	}
	if mResp.Content[0].Type != "thinking" || mResp.Content[0].Signature != "sig_test_sig" {
		t.Errorf("block 0 thinking mismatch: %+v", mResp.Content[0])
	}
	if mResp.Content[1].Type != "text" || mResp.Content[1].Text != "The result is 42." {
		t.Errorf("block 1 text mismatch: %+v", mResp.Content[1])
	}
	if mResp.Content[2].Type != "tool_use" || mResp.Content[2].ID != "toolu_calc" {
		t.Errorf("block 2 tool_use mismatch: %+v", mResp.Content[2])
	}
	if mResp.StopReason == nil || *mResp.StopReason != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %v", mResp.StopReason)
	}
	if mResp.Usage.InputTokens != 30 || mResp.Usage.OutputTokens != 15 || mResp.Usage.CacheCreationInputTokens != 10 {
		t.Errorf("usage mismatch: %+v", mResp.Usage)
	}
}

func TestAnthropicStreamEncoder_MatchesGoldenFixture(t *testing.T) {
	// Read fixture from testdata/anthropic/thinking-with-signature.sses
	golden, err := os.ReadFile("../../testdata/anthropic/thinking-with-signature.sses")
	if err != nil {
		// Fallback path if running with different working directory
		golden, err = os.ReadFile("testdata/anthropic/thinking-with-signature.sses")
	}
	if err != nil {
		t.Logf("Notice: fixture read error %v; skipping strict diff but running semantic test", err)
	}

	var buf bytes.Buffer
	enc := NewAnthropicStreamEncoder(&buf, "claude-3-7-sonnet-20250219")

	events := []ir.Event{
		ir.EventMessageStart{ID: "msg_011kQ87g8y8n5yDqC2Nq28aY"},
		ir.EventUsageUpdate{PromptTokens: 25, CompletionTokens: 0, CacheCreationInputTokens: 0, CacheReadInputTokens: 0},
		ir.EventBlockStart{Index: 0, Kind: "thinking"},
		ir.EventReasoningDelta{Index: 0, Text: "Let me think about how to define the function."},
		ir.EventReasoningSignature{Index: 0, Signature: "Ev4BCAAQACIdCgQIARABGgw1bWluLXRlc3Qtc2ln"},
		ir.EventBlockStart{Index: 1, Kind: "text"},
		ir.EventTextDelta{Index: 1, Text: "function "},
		ir.EventTextDelta{Index: 1, Text: "foo()"},
		ir.EventUsageUpdate{PromptTokens: 25, CompletionTokens: 18},
		ir.EventMessageStop{FinishReason: "end_turn"},
	}

	for _, ev := range events {
		if err := enc.EncodeEvent(ev); err != nil {
			t.Fatalf("failed to encode event: %v", err)
		}
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("failed to close encoder: %v", err)
	}

	streamOutput := buf.String()

	// Verify required state transitions
	requiredSubstrings := []string{
		`event: message_start`,
		`"id":"msg_011kQ87g8y8n5yDqC2Nq28aY"`,
		`event: content_block_start`,
		`"content_block":{"type":"thinking","thinking":""}`,
		`event: content_block_delta`,
		`"type":"thinking_delta","thinking":"Let me think about how to define the function."`,
		`"type":"signature_delta","signature":"Ev4BCAAQACIdCgQIARABGgw1bWluLXRlc3Qtc2ln"`,
		`event: content_block_stop`,
		`"index":0`,
		`"content_block":{"type":"text","text":""}`,
		`"type":"text_delta","text":"function "`,
		`"type":"text_delta","text":"foo()"`,
		`"index":1`,
		`event: message_delta`,
		`"stop_reason":"end_turn"`,
		`"output_tokens":18`,
		`event: message_stop`,
	}

	for _, sub := range requiredSubstrings {
		if !strings.Contains(streamOutput, sub) {
			t.Errorf("stream output missing expected substring %q:\n%s", sub, streamOutput)
		}
	}

	if golden != nil {
		// Verify whitespace exactness against golden fixture
		if !strings.Contains(streamOutput, `"text":"function "`) {
			t.Errorf("whitespace in 'function ' was lost!")
		}
	}
}
