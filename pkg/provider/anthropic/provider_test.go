package anthropic

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	spikesanthropic "github.com/smhanov/ultiproxy/pkg/spikes/anthropic"
)

func TestAnthropicThinkingWithSignatureFixture(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "anthropic", "thinking-with-signature.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	ch := StreamHandler(context.Background(), f)

	var (
		foundReasoningDelta bool
		foundSignature      bool
		signatureValue      string
		foundText           bool
	)

	for ev := range ch {
		switch e := ev.(type) {
		case ir.EventReasoningDelta:
			foundReasoningDelta = true
		case ir.EventReasoningSignature:
			foundSignature = true
			signatureValue = e.Signature
		case ir.EventTextDelta:
			foundText = true
		}
	}

	if !foundReasoningDelta {
		t.Error("expected EventReasoningDelta")
	}
	if !foundSignature {
		t.Error("expected EventReasoningSignature")
	}
	expectedSig := "Ev4BCAAQACIdCgQIARABGgw1bWluLXRlc3Qtc2ln"
	if signatureValue != expectedSig {
		t.Errorf("expected signature %q, got %q", expectedSig, signatureValue)
	}
	if !foundText {
		t.Error("expected EventTextDelta")
	}

	// Also test ConvertStreamState signature preservation
	f2, _ := os.Open(fixturePath)
	defer f2.Close()
	decoder := spikesanthropic.NewDecoder()
	state, err := decoder.Decode(f2)
	if err != nil {
		t.Fatalf("decoder error: %v", err)
	}

	resp := ConvertStreamState("msg-01", state)
	if len(resp.Message.Blocks) < 2 {
		t.Fatalf("expected at least 2 blocks, got %d", len(resp.Message.Blocks))
	}
	rBlock, ok := resp.Message.Blocks[0].(ir.ReasoningBlock)
	if !ok || rBlock.Signature != expectedSig {
		t.Errorf("expected preserved signature %q in ir.ReasoningBlock, got %+v", expectedSig, resp.Message.Blocks[0])
	}
}

func TestAnthropicCacheUsageFixture(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "anthropic", "cache-usage.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	ch := StreamHandler(context.Background(), f)

	var lastUsage ir.EventUsageUpdate
	for ev := range ch {
		if u, ok := ev.(ir.EventUsageUpdate); ok {
			lastUsage = u
		}
	}

	if lastUsage.CacheCreationInputTokens != 2000 {
		t.Errorf("expected CacheCreationInputTokens 2000, got %d", lastUsage.CacheCreationInputTokens)
	}
	if lastUsage.CacheReadInputTokens != 500 {
		t.Errorf("expected CacheReadInputTokens 500, got %d", lastUsage.CacheReadInputTokens)
	}
	if lastUsage.PromptTokens != 100 {
		t.Errorf("expected PromptTokens 100, got %d", lastUsage.PromptTokens)
	}
	if lastUsage.CompletionTokens != 12 {
		t.Errorf("expected CompletionTokens 12, got %d", lastUsage.CompletionTokens)
	}
	if lastUsage.TotalTokens != 112 {
		t.Errorf("expected TotalTokens 112, got %d", lastUsage.TotalTokens)
	}
}

func TestAnthropicToolUseStreamFixture(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "anthropic", "tool-use-stream.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	ch := StreamHandler(context.Background(), f)

	var (
		startedTools []string
		argDeltas    string
		stoppedTools int
	)

	for ev := range ch {
		switch e := ev.(type) {
		case ir.EventToolCallStart:
			startedTools = append(startedTools, e.Name)
		case ir.EventToolArgumentsDelta:
			argDeltas += e.Arguments
		case ir.EventToolCallStop:
			stoppedTools++
		}
	}

	if len(startedTools) == 0 {
		t.Error("expected tool call started")
	}
	if argDeltas == "" {
		t.Error("expected accumulated argument deltas")
	}
	if stoppedTools == 0 {
		t.Error("expected tool call stop event")
	}
}

func TestAnthropicErrorEventFixture(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "anthropic", "error-event.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	ch := StreamHandler(context.Background(), f)

	var foundError bool
	for ev := range ch {
		if errEv, ok := ev.(ir.EventUpstreamError); ok {
			foundError = true
			if errEv.Message == "" {
				t.Error("expected non-empty error message")
			}
		}
	}

	if !foundError {
		t.Error("expected EventUpstreamError from error-event.sses")
	}
}

func TestAnthropicPayloadExtendedThinkingAndCacheControl(t *testing.T) {
	p, err := New(Config{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role: "system",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "You are a helpful assistant."},
				ir.CacheControl{Breakpoint: true},
			},
		},
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "Solve this problem."},
				ir.CacheControl{Breakpoint: true},
			},
		},
	}

	cfg := provider.NewRequestConfig(
		provider.WithModel("claude-3-7-sonnet-20250219"),
		provider.WithReasoningEffort("high"),
	)

	payload, err := p.buildPayload(msgs, cfg, false)
	if err != nil {
		t.Fatalf("buildPayload failed: %v", err)
	}

	// 1. Check extended thinking budget
	thinking, ok := payload["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" || thinking["budget_tokens"] != 16384 {
		t.Errorf("expected thinking budget 16384, got %v", payload["thinking"])
	}

	// 2. Check system cache_control
	sysRaw, ok := payload["system"].([]map[string]any)
	if !ok || len(sysRaw) != 1 {
		t.Fatalf("expected 1 system block, got %v", payload["system"])
	}
	cc, ok := sysRaw[0]["cache_control"].(map[string]any)
	if !ok || cc["type"] != "ephemeral" {
		t.Errorf("expected system cache_control ephemeral, got %v", sysRaw[0]["cache_control"])
	}

	// 3. Check message cache_control
	msgsRaw, ok := payload["messages"].([]map[string]any)
	if !ok || len(msgsRaw) != 1 {
		t.Fatalf("expected 1 message, got %v", payload["messages"])
	}
	userBlocks := msgsRaw[0]["content"].([]map[string]any)
	userCC, ok := userBlocks[0]["cache_control"].(map[string]any)
	if !ok || userCC["type"] != "ephemeral" {
		t.Errorf("expected user cache_control ephemeral, got %v", userBlocks[0]["cache_control"])
	}
}
