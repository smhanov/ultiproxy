package responses

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParallelToolCalls(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "responses", "parallel-tool-calls.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	translator := NewTranslator()
	events, err := translator.TranslateStream(f)
	if err != nil {
		t.Fatalf("unexpected error translating stream: %v", err)
	}

	if len(events) == 0 {
		t.Fatalf("expected translated events, got 0")
	}

	// Verify mappings
	out0, ok0 := translator.OutputIndex("item_call_1")
	if !ok0 || out0 != 0 {
		t.Errorf("expected item_call_1 to map to output 0, got %d (ok=%v)", out0, ok0)
	}
	out1, ok1 := translator.OutputIndex("item_call_2")
	if !ok1 || out1 != 1 {
		t.Errorf("expected item_call_2 to map to output 1, got %d (ok=%v)", out1, ok1)
	}

	toolIdx0, okT0 := translator.ToolCallIndex("call_apple_01")
	if !okT0 || toolIdx0 != 0 {
		t.Errorf("expected call_apple_01 tool index 0, got %d", toolIdx0)
	}
	toolIdx1, okT1 := translator.ToolCallIndex("call_tesla_02")
	if !okT1 || toolIdx1 != 1 {
		t.Errorf("expected call_tesla_02 tool index 1, got %d", toolIdx1)
	}

	// Verify parallel argument accumulation did not cross-contaminate
	appleArgs := translator.AccumulatedArguments("call_apple_01")
	expectedApple := `{"ticker": "AAPL", "market": "US"}`
	if appleArgs != expectedApple {
		t.Errorf("expected apple args %q, got %q", expectedApple, appleArgs)
	}

	teslaArgs := translator.AccumulatedArguments("call_tesla_02")
	expectedTesla := `{"ticker": "TSLA", "market": "US"}`
	if teslaArgs != expectedTesla {
		t.Errorf("expected tesla args %q, got %q", expectedTesla, teslaArgs)
	}

	// Check final completed event usage
	if translator.FinalUsage.InputTokens != 85 || translator.FinalUsage.OutputTokens != 62 {
		t.Errorf("unexpected final usage: %+v", translator.FinalUsage)
	}
}

func TestTextWithReasoning(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "responses", "text-with-reasoning.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	translator := NewTranslator()
	events, err := translator.TranslateStream(f)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify reasoning deltas emit BEFORE text deltas
	var firstReasoningIdx = -1
	var firstTextIdx = -1

	for i, evt := range events {
		if evt.Kind == EventKindReasoningDelta && firstReasoningIdx == -1 {
			firstReasoningIdx = i
		}
		if evt.Kind == EventKindTextDelta && firstTextIdx == -1 {
			firstTextIdx = i
		}
	}

	if firstReasoningIdx == -1 {
		t.Fatalf("expected at least one reasoning delta event")
	}
	if firstTextIdx == -1 {
		t.Fatalf("expected at least one text delta event")
	}
	if firstReasoningIdx >= firstTextIdx {
		t.Errorf("expected reasoning delta (idx %d) to emit before text delta (idx %d)", firstReasoningIdx, firstTextIdx)
	}

	// Verify final usage
	if translator.FinalUsage.ReasoningTokens != 25 {
		t.Errorf("expected 25 reasoning tokens, got %d", translator.FinalUsage.ReasoningTokens)
	}
	if translator.FinalUsage.TotalTokens != 90 {
		t.Errorf("expected 90 total tokens, got %d", translator.FinalUsage.TotalTokens)
	}
}

func TestCompletedWithUsage(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "responses", "completed-with-usage.sses")
	f, err := os.Open(fixturePath)
	if err != nil {
		t.Fatalf("failed to open fixture: %v", err)
	}
	defer f.Close()

	translator := NewTranslator()
	events, err := translator.TranslateStream(f)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(events) == 0 {
		t.Fatalf("expected events")
	}

	if translator.FinalUsage.InputTokens != 120 {
		t.Errorf("expected 120 input tokens, got %d", translator.FinalUsage.InputTokens)
	}
	if translator.FinalUsage.OutputTokens != 35 {
		t.Errorf("expected 35 output tokens, got %d", translator.FinalUsage.OutputTokens)
	}
	if translator.FinalUsage.TotalTokens != 155 {
		t.Errorf("expected 155 total tokens, got %d", translator.FinalUsage.TotalTokens)
	}
	if translator.FinalUsage.ReasoningTokens != 10 {
		t.Errorf("expected 10 reasoning tokens, got %d", translator.FinalUsage.ReasoningTokens)
	}
}
