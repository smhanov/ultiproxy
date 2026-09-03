package ir

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestMixedConversationRoundTrip(t *testing.T) {
	conversation := []Message{
		{
			Role: "system",
			Blocks: []Block{
				TextBlock{Text: "You are a helpful and precise assistant."},
				CacheControl{Breakpoint: true},
			},
			Meta: map[string]string{"system_id": "sys_1"},
		},
		{
			Role: "user",
			Blocks: []Block{
				TextBlock{Text: "What is shown in this image?"},
				ImageBlock{URL: "https://example.com/sample.png", Detail: "high"},
			},
		},
		{
			Role: "assistant",
			Blocks: []Block{
				ReasoningBlock{
					ReasoningKind: ReasoningText,
					Text:          "The user is asking about the image. Let's call the image analysis tool.",
					Signature:     "sig_abc123opaque",
					Opaque:        json.RawMessage(`{"thought_seed":42}`),
				},
				ToolCallBlock{
					Index:     0,
					ID:        "tool_call_001",
					Name:      "analyze_image",
					Arguments: `{"url": "https://example.com/sample.png"}`,
				},
			},
			Meta: map[string]string{"model": "claude-3-7-sonnet"},
		},
		{
			Role: "tool",
			Blocks: []Block{
				ToolResultBlock{
					ToolCallID: "tool_call_001",
					Name:       "analyze_image",
					Content:    `{"label": "golden retriever playing in park"}`,
				},
			},
		},
		{
			Role: "assistant",
			Blocks: []Block{
				ReasoningBlock{
					ReasoningKind: ReasoningSummary,
					Text:          "The analysis confirms it's a golden retriever.",
				},
				TextBlock{Text: "The image shows a golden retriever playing happily in a grassy park."},
			},
		},
	}

	data, err := json.Marshal(conversation)
	if err != nil {
		t.Fatalf("failed to marshal conversation: %v", err)
	}

	var decoded []Message
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal conversation: %v", err)
	}

	if len(decoded) != len(conversation) {
		t.Fatalf("expected %d messages, got %d", len(conversation), len(decoded))
	}

	for i := range conversation {
		expectedMsg := conversation[i]
		actualMsg := decoded[i]

		if actualMsg.Role != expectedMsg.Role {
			t.Errorf("msg[%d] role mismatch: expected %s, got %s", i, expectedMsg.Role, actualMsg.Role)
		}

		if len(actualMsg.Blocks) != len(expectedMsg.Blocks) {
			t.Fatalf("msg[%d] block count mismatch: expected %d, got %d", i, len(expectedMsg.Blocks), len(actualMsg.Blocks))
		}

		for j := range expectedMsg.Blocks {
			expBlock := expectedMsg.Blocks[j]
			actBlock := actualMsg.Blocks[j]

			if expBlock.Kind() != actBlock.Kind() {
				t.Errorf("msg[%d].block[%d] kind mismatch: expected %s, got %s", i, j, expBlock.Kind(), actBlock.Kind())
			}

			if !reflect.DeepEqual(expBlock, actBlock) {
				t.Errorf("msg[%d].block[%d] deep equal failed:\nexpected: %+v\ngot: %+v", i, j, expBlock, actBlock)
			}
		}

		if len(expectedMsg.Meta) > 0 {
			if !reflect.DeepEqual(expectedMsg.Meta, actualMsg.Meta) {
				t.Errorf("msg[%d] meta mismatch: expected %+v, got %+v", i, expectedMsg.Meta, actualMsg.Meta)
			}
		}
	}
}

func TestEventAlgebraKinds(t *testing.T) {
	events := []Event{
		EventMessageStart{ID: "msg_123"},
		EventBlockStart{Index: 0, Kind: "text"},
		EventTextDelta{Index: 0, Text: "hello "},
		EventReasoningDelta{Index: 1, Text: "thinking..."},
		EventReasoningSignature{Index: 1, Signature: "sig_999"},
		EventToolCallStart{Index: 2, ID: "call_1", Name: "test_tool"},
		EventToolArgumentsDelta{Index: 2, Arguments: `{"q":`},
		EventToolCallStop{Index: 2},
		EventUsageUpdate{
			PromptTokens:             10,
			CompletionTokens:         20,
			TotalTokens:              30,
			CacheCreationInputTokens: 5,
			CacheReadInputTokens:     2,
			Cost:                     0.0015,
		},
		EventMessageStop{FinishReason: "end_turn", UpstreamID: "up_123"},
		EventUpstreamError{Kind: "rate_limit", Message: "Too Many Requests", RetryAfterSeconds: 5, Permanent: false},
	}

	expectedKinds := []string{
		"message_start",
		"block_start",
		"text_delta",
		"reasoning_delta",
		"reasoning_signature",
		"tool_call_start",
		"tool_arguments_delta",
		"tool_call_stop",
		"usage_update",
		"message_stop",
		"upstream_error",
	}

	for i, evt := range events {
		if evt.EventKind() != expectedKinds[i] {
			t.Errorf("event[%d] expected kind %s, got %s", i, expectedKinds[i], evt.EventKind())
		}
	}
}
