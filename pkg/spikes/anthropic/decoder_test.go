package anthropic

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDecodeThinkingWithSignature(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "anthropic", "thinking-with-signature.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	decoder := NewDecoder()
	state, err := decoder.Decode(f)
	if err != nil {
		t.Fatalf("unexpected error decoding fixture: %v", err)
	}

	if state.MessageID != "msg_011kQ87g8y8n5yDqC2Nq28aY" {
		t.Errorf("expected message id msg_011kQ87g8y8n5yDqC2Nq28aY, got %s", state.MessageID)
	}
	if state.Model != "claude-3-7-sonnet-20250219" {
		t.Errorf("expected model claude-3-7-sonnet-20250219, got %s", state.Model)
	}
	if !state.Completed {
		t.Errorf("expected completed to be true")
	}

	// Block 0: thinking
	blk0 := state.Block(0)
	if blk0 == nil {
		t.Fatalf("expected block 0 to exist")
	}
	if blk0.Type != BlockTypeThinking {
		t.Errorf("expected block 0 type thinking, got %s", blk0.Type)
	}
	if blk0.Thinking != "Let me think about how to define the function." {
		t.Errorf("unexpected thinking content: %q", blk0.Thinking)
	}
	if blk0.Signature != "Ev4BCAAQACIdCgQIARABGgw1bWluLXRlc3Qtc2ln" {
		t.Errorf("unexpected signature: %q", blk0.Signature)
	}

	// Block 1: text with multi-chunk whitespace preservation
	blk1 := state.Block(1)
	if blk1 == nil {
		t.Fatalf("expected block 1 to exist")
	}
	if blk1.Type != BlockTypeText {
		t.Errorf("expected block 1 type text, got %s", blk1.Type)
	}
	expectedText := "function foo()"
	if blk1.Text != expectedText {
		t.Errorf("expected text %q, got %q (whitespace was not preserved!)", expectedText, blk1.Text)
	}

	// Usage accumulation: input_tokens=25 from start, output_tokens=18 from message_delta
	if state.Usage.InputTokens != 25 {
		t.Errorf("expected input tokens 25, got %d", state.Usage.InputTokens)
	}
	if state.Usage.OutputTokens != 18 {
		t.Errorf("expected output tokens 18, got %d", state.Usage.OutputTokens)
	}
	if state.StopReason != "end_turn" {
		t.Errorf("expected stop reason end_turn, got %s", state.StopReason)
	}
}

func TestDecodeToolUseStream(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "anthropic", "tool-use-stream.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	decoder := NewDecoder()
	state, err := decoder.Decode(f)
	if err != nil {
		t.Fatalf("unexpected error decoding fixture: %v", err)
	}

	blk0 := state.Block(0)
	if blk0 == nil {
		t.Fatalf("expected block 0 to exist")
	}
	if blk0.Type != BlockTypeToolUse {
		t.Errorf("expected block 0 type tool_use, got %s", blk0.Type)
	}
	if blk0.ToolUseID != "toolu_01A09q90qw90lq917835lq9" {
		t.Errorf("expected tool id toolu_01A09q90qw90lq917835lq9, got %s", blk0.ToolUseID)
	}
	if blk0.ToolName != "get_weather" {
		t.Errorf("expected tool name get_weather, got %s", blk0.ToolName)
	}
	expectedJSON := `{"location": "San Francisco, CA"}`
	if blk0.ToolInputJSON != expectedJSON {
		t.Errorf("expected tool input json %q, got %q", expectedJSON, blk0.ToolInputJSON)
	}

	if state.Usage.InputTokens != 30 {
		t.Errorf("expected input tokens 30, got %d", state.Usage.InputTokens)
	}
	if state.Usage.OutputTokens != 22 {
		t.Errorf("expected output tokens 22, got %d", state.Usage.OutputTokens)
	}
	if state.StopReason != "tool_use" {
		t.Errorf("expected stop reason tool_use, got %s", state.StopReason)
	}
}

func TestDecodeErrorEvent(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "anthropic", "error-event.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	decoder := NewDecoder()
	state, err := decoder.Decode(f)
	if err != nil {
		t.Fatalf("unexpected error decoding fixture: %v", err)
	}

	if state.Error == nil {
		t.Fatalf("expected error in state, got nil")
	}
	if state.Error.Type != "overloaded_error" {
		t.Errorf("expected overloaded_error, got %s", state.Error.Type)
	}
	if state.Error.Message != "Anthropic is currently overloaded with requests." {
		t.Errorf("unexpected error message: %s", state.Error.Message)
	}
}

func TestDecodeCacheUsage(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "anthropic", "cache-usage.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	decoder := NewDecoder()
	state, err := decoder.Decode(f)
	if err != nil {
		t.Fatalf("unexpected error decoding fixture: %v", err)
	}

	if state.Usage.InputTokens != 100 {
		t.Errorf("expected input tokens 100, got %d", state.Usage.InputTokens)
	}
	if state.Usage.CacheCreationInputTokens != 2000 {
		t.Errorf("expected cache creation 2000, got %d", state.Usage.CacheCreationInputTokens)
	}
	if state.Usage.CacheReadInputTokens != 500 {
		t.Errorf("expected cache read 500, got %d", state.Usage.CacheReadInputTokens)
	}
	if state.Usage.OutputTokens != 12 {
		t.Errorf("expected output tokens 12, got %d", state.Usage.OutputTokens)
	}
	if state.PingCount != 1 {
		t.Errorf("expected 1 ping count, got %d", state.PingCount)
	}
	if state.FullText() != "Cached response text" {
		t.Errorf("unexpected full text: %s", state.FullText())
	}
}

func TestRedactedThinking(t *testing.T) {
	rawSSE := `event: message_start
data: {"type":"message_start","message":{"id":"msg_redacted","type":"message","role":"assistant","content":[],"model":"claude-3-7-sonnet-20250219","usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque_encrypted_payload"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_stop
data: {"type":"message_stop"}
`
	decoder := NewDecoder()
	state, err := decoder.Decode(strings.NewReader(rawSSE))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	blk0 := state.Block(0)
	if blk0 == nil {
		t.Fatalf("expected block 0")
	}
	if blk0.Type != BlockTypeRedactedThinking {
		t.Errorf("expected redacted_thinking, got %s", blk0.Type)
	}
	if blk0.RedactedData != "opaque_encrypted_payload" {
		t.Errorf("expected opaque_encrypted_payload, got %s", blk0.RedactedData)
	}
}
