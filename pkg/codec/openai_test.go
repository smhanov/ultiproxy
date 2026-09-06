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

func TestDecodeChatCompletionRequest(t *testing.T) {
	rawJSON := `{
		"model": "gpt-4o",
		"messages": [
			{
				"role": "system",
				"content": "You are a helpful assistant."
			},
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "What is in this picture?"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==", "detail": "high"}},
					{"type": "image_url", "image_url": "https://example.com/photo.jpg"}
				]
			},
			{
				"role": "assistant",
				"content": "Checking...",
				"reasoning_content": "Looking closely at the pixels",
				"tool_calls": [
					{
						"index": 0,
						"id": "call_123",
						"type": "function",
						"function": {
							"name": "analyze_image",
							"arguments": "{\"mode\": \"fast\"}"
						}
					}
				]
			},
			{
				"role": "tool",
				"tool_call_id": "call_123",
				"name": "analyze_image",
				"content": "{\"result\": \"golden retriever\"}"
			}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "analyze_image",
					"description": "Analyze an image",
					"parameters": {"type": "object"}
				}
			}
		],
		"tool_choice": "auto",
		"temperature": 0.7,
		"max_completion_tokens": 512,
		"stream": true,
		"stream_options": {
			"include_usage": true
		},
		"reasoning_effort": "high"
	}`

	decoded, err := DecodeChatCompletionRequest([]byte(rawJSON))
	if err != nil {
		t.Fatalf("unexpected error decoding: %v", err)
	}

	if decoded.Model != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", decoded.Model)
	}
	if !decoded.Stream {
		t.Errorf("expected Stream=true")
	}
	if !decoded.IncludeUsage {
		t.Errorf("expected IncludeUsage=true")
	}
	if !decoded.ToolsRequested {
		t.Errorf("expected ToolsRequested=true")
	}

	if len(decoded.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(decoded.Messages))
	}

	// Message 0: system
	if decoded.Messages[0].Role != "system" {
		t.Errorf("msg[0] role expected system, got %s", decoded.Messages[0].Role)
	}
	if len(decoded.Messages[0].Blocks) != 1 {
		t.Fatalf("msg[0] blocks count expected 1, got %d", len(decoded.Messages[0].Blocks))
	}
	tb, ok := decoded.Messages[0].Blocks[0].(ir.TextBlock)
	if !ok || tb.Text != "You are a helpful assistant." {
		t.Errorf("msg[0] block 0 mismatch: %+v", decoded.Messages[0].Blocks[0])
	}

	// Message 1: user multipart
	if decoded.Messages[1].Role != "user" {
		t.Errorf("msg[1] role expected user, got %s", decoded.Messages[1].Role)
	}
	if len(decoded.Messages[1].Blocks) != 3 {
		t.Fatalf("msg[1] blocks count expected 3, got %d", len(decoded.Messages[1].Blocks))
	}
	if img, ok := decoded.Messages[1].Blocks[1].(ir.ImageBlock); !ok || img.Detail != "high" || !strings.HasPrefix(img.URL, "data:image/png") {
		t.Errorf("msg[1] block 1 image mismatch: %+v", decoded.Messages[1].Blocks[1])
	}
	if img2, ok := decoded.Messages[1].Blocks[2].(ir.ImageBlock); !ok || img2.URL != "https://example.com/photo.jpg" {
		t.Errorf("msg[1] block 2 image mismatch: %+v", decoded.Messages[1].Blocks[2])
	}

	// Message 2: assistant with reasoning and tool_calls
	if decoded.Messages[2].Role != "assistant" {
		t.Errorf("msg[2] role expected assistant, got %s", decoded.Messages[2].Role)
	}
	if len(decoded.Messages[2].Blocks) != 3 {
		t.Fatalf("msg[2] blocks count expected 3, got %d", len(decoded.Messages[2].Blocks))
	}
	if rb, ok := decoded.Messages[2].Blocks[0].(ir.ReasoningBlock); !ok || rb.Text != "Looking closely at the pixels" {
		t.Errorf("msg[2] reasoning block mismatch: %+v", decoded.Messages[2].Blocks[0])
	}
	if tc, ok := decoded.Messages[2].Blocks[2].(ir.ToolCallBlock); !ok || tc.ID != "call_123" || tc.Name != "analyze_image" {
		t.Errorf("msg[2] tool call block mismatch: %+v", decoded.Messages[2].Blocks[2])
	}

	// Message 3: tool result
	if decoded.Messages[3].Role != "tool" {
		t.Errorf("msg[3] role expected tool, got %s", decoded.Messages[3].Role)
	}
	if tr, ok := decoded.Messages[3].Blocks[0].(ir.ToolResultBlock); !ok || tr.ToolCallID != "call_123" || tr.Content != "{\"result\": \"golden retriever\"}" {
		t.Errorf("msg[3] tool result mismatch: %+v", decoded.Messages[3].Blocks[0])
	}

	// Verify options applied to RequestConfig
	cfg := provider.NewRequestConfig(decoded.Options...)
	if cfg.Model != "gpt-4o" {
		t.Errorf("expected cfg.Model=gpt-4o, got %s", cfg.Model)
	}
	if cfg.MaxTokens != 512 {
		t.Errorf("expected cfg.MaxTokens=512, got %d", cfg.MaxTokens)
	}
	if cfg.Temperature == nil || *cfg.Temperature != 0.7 {
		t.Errorf("expected cfg.Temperature=0.7, got %v", cfg.Temperature)
	}
	if cfg.ReasoningEffort != "high" {
		t.Errorf("expected cfg.ReasoningEffort=high, got %s", cfg.ReasoningEffort)
	}
	if cfg.ExtraBody["tools"] == nil {
		t.Errorf("expected tools in ExtraBody")
	}
	if cfg.ExtraBody["tool_choice"] != "auto" {
		t.Errorf("expected tool_choice auto in ExtraBody")
	}
}

func TestEncodeChatCompletionResponse(t *testing.T) {
	resp := &ir.Response{
		ID: "chatcmpl-custom123",
		Message: &ir.Message{
			Role: "assistant",
			Blocks: []ir.Block{
				ir.ReasoningBlock{
					ReasoningKind: ir.ReasoningText,
					Text:          "Thinking through response...",
				},
				ir.TextBlock{
					Text: "Here is your answer.",
				},
				ir.ToolCallBlock{
					Index:     0,
					ID:        "call_abc",
					Name:      "get_weather",
					Arguments: `{"city":"Berlin"}`,
				},
			},
		},
		FinishReason: "tool_calls",
		Usage: &ir.Usage{
			PromptTokens:     20,
			CompletionTokens: 30,
			TotalTokens:      50,
			ReasoningTokens:  10,
		},
	}

	outBytes, err := EncodeChatCompletionResponse(resp, "gpt-4o")
	if err != nil {
		t.Fatalf("unexpected encode error: %v", err)
	}

	var out OpenAIChatCompletionResponse
	if err := json.Unmarshal(outBytes, &out); err != nil {
		t.Fatalf("failed to unmarshal output: %v", err)
	}

	if out.ID != "chatcmpl-custom123" {
		t.Errorf("expected ID chatcmpl-custom123, got %s", out.ID)
	}
	if out.Model != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", out.Model)
	}
	if len(out.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(out.Choices))
	}
	choice := out.Choices[0]
	if choice.FinishReason != "tool_calls" {
		t.Errorf("expected finish_reason tool_calls, got %s", choice.FinishReason)
	}
	if choice.Message.Content == nil || *choice.Message.Content != "Here is your answer." {
		t.Errorf("expected content 'Here is your answer.', got %v", choice.Message.Content)
	}
	if choice.Message.ReasoningContent == nil || *choice.Message.ReasoningContent != "Thinking through response..." {
		t.Errorf("expected reasoning content, got %v", choice.Message.ReasoningContent)
	}
	if len(choice.Message.ToolCalls) != 1 || choice.Message.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("expected tool call get_weather, got %+v", choice.Message.ToolCalls)
	}
	if out.Usage == nil || out.Usage.TotalTokens != 50 || out.Usage.CompletionTokensDetails == nil || out.Usage.CompletionTokensDetails.ReasoningTokens != 10 {
		t.Errorf("usage mismatch: %+v", out.Usage)
	}
}

func TestOpenAIStreamEncoder_WhitespaceExactness(t *testing.T) {
	var buf bytes.Buffer
	enc := NewOpenAIStreamEncoder(&buf, "gpt-4o", true)

	// Stream series of events with leading/trailing whitespaces that MUST NOT be trimmed
	events := []ir.Event{
		ir.EventMessageStart{ID: "chatcmpl-test1"},
		ir.EventReasoningDelta{Index: 0, Text: "  reasoning start\n  second line\n"},
		ir.EventTextDelta{Index: 1, Text: "   def "},
		ir.EventTextDelta{Index: 1, Text: "foo():\n"},
		ir.EventTextDelta{Index: 1, Text: "\treturn 42\n"},
		ir.EventToolCallStart{Index: 2, ID: "call_01", Name: "eval"},
		ir.EventToolArgumentsDelta{Index: 2, Arguments: "{\"code\": "},
		ir.EventToolArgumentsDelta{Index: 2, Arguments: "\"foo()\"}"},
		ir.EventToolCallStop{Index: 2},
		ir.EventUsageUpdate{PromptTokens: 15, CompletionTokens: 25, TotalTokens: 40},
		ir.EventMessageStop{FinishReason: "stop"},
	}

	for _, ev := range events {
		if err := enc.EncodeEvent(ev); err != nil {
			t.Fatalf("failed to encode event %+v: %v", ev, err)
		}
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("failed to close encoder: %v", err)
	}

	output := buf.String()

	// Verify exact whitespace preservation
	if !strings.Contains(output, `"reasoning_content":"  reasoning start\n  second line\n"`) {
		t.Errorf("reasoning content whitespace was modified:\n%s", output)
	}
	if !strings.Contains(output, `"content":"   def "`) {
		t.Errorf("leading/trailing spaces in text delta were trimmed:\n%s", output)
	}
	if !strings.Contains(output, `"content":"\treturn 42\n"`) {
		t.Errorf("tab and newline in text delta were trimmed:\n%s", output)
	}
	if !strings.Contains(output, `"arguments":"{\"code\": "`) {
		t.Errorf("tool argument delta whitespace was modified:\n%s", output)
	}

	// Verify usage chunk and [DONE]
	if !strings.Contains(output, `"usage":{"prompt_tokens":15,"completion_tokens":25,"total_tokens":40`) {
		t.Errorf("expected final usage chunk in output:\n%s", output)
	}
	if !strings.HasSuffix(strings.TrimSpace(output), "data: [DONE]") {
		t.Errorf("expected stream to terminate with data: [DONE], got:\n%s", output)
	}
}

func TestDecodeServerInputFixture(t *testing.T) {
	data, err := os.ReadFile("../../testdata/server-input/openai-chat.json")
	if err != nil {
		t.Fatalf("failed to read test fixture: %v", err)
	}

	decoded, err := DecodeChatCompletionRequest(data)
	if err != nil {
		t.Fatalf("failed to decode fixture: %v", err)
	}

	if decoded.Model != "gpt-4o" {
		t.Errorf("expected gpt-4o, got %s", decoded.Model)
	}
	if len(decoded.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(decoded.Messages))
	}
	if decoded.ToolsRequested {
		t.Errorf("expected ToolsRequested=false")
	}
}

func TestDecodeChatCompletionRequest_DropsAnthropicOnlyExtraFields(t *testing.T) {
	raw := `{
		"model": "glm-5.3",
		"messages": [{"role": "user", "content": "hi"}],
		"metadata": {"user_id": "x"},
		"output_config": {"effort": "max"},
		"context_management": {"edits": []},
		"reasoning_effort": "high"
	}`
	decoded, err := DecodeChatCompletionRequest([]byte(raw))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	cfg := provider.NewRequestConfig(decoded.Options...)
	for _, banned := range []string{"metadata", "output_config", "context_management"} {
		if _, ok := cfg.ExtraBody[banned]; ok {
			t.Errorf("ExtraBody still has %q: %+v", banned, cfg.ExtraBody)
		}
	}
	if cfg.ReasoningEffort != "high" {
		t.Errorf("reasoning_effort = %q, want high", cfg.ReasoningEffort)
	}
}

func TestOpenAIStreamEncoder_SecondStopAndCloseAreNoOp(t *testing.T) {
	var buf bytes.Buffer
	enc := NewOpenAIStreamEncoder(&buf, "glm-5.3", false)
	_ = enc.EncodeEvent(ir.EventMessageStart{})
	_ = enc.EncodeEvent(ir.EventTextDelta{Index: 0, Text: "ok"})
	_ = enc.EncodeEvent(ir.EventMessageStop{FinishReason: "length"})
	_ = enc.EncodeEvent(ir.EventMessageStop{FinishReason: "stop"})
	if err := enc.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	out := buf.String()
	if n := strings.Count(out, `"finish_reason":"length"`); n != 1 {
		t.Fatalf("length finish_reason count = %d, want 1\n%s", n, out)
	}
	if strings.Contains(out, `"finish_reason":"stop"`) {
		t.Fatalf("second stop reason leaked:\n%s", out)
	}
	if n := strings.Count(out, "data: [DONE]"); n != 1 {
		t.Fatalf("[DONE] count = %d, want 1\n%s", n, out)
	}
	before := buf.Len()
	if err := enc.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
	if buf.Len() != before {
		t.Fatalf("second Close emitted extra bytes")
	}
}
