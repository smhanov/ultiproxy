package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
)

func TestConvertMessagesTransparency(t *testing.T) {
	// If messages lack a system message, do NOT rewrite user content
	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "Hello there"},
			},
		},
	}
	converted := ConvertMessages(msgs, ConvertOptions{})
	if len(converted) != 1 {
		t.Fatalf("expected 1 message, got %d", len(converted))
	}
	if converted[0].Role != "user" || converted[0].Content != "Hello there" {
		t.Errorf("unexpected conversion: %+v", converted[0])
	}
}

func TestConvertMessagesReasoningEcho(t *testing.T) {
	msgs := []*ir.Message{
		{
			Role: "assistant",
			Blocks: []ir.Block{
				ir.ReasoningBlock{ReasoningKind: ir.ReasoningText, Text: "thought process"},
				ir.TextBlock{Text: "final answer"},
			},
		},
	}
	converted := ConvertMessages(msgs, ConvertOptions{EchoReasoning: true})
	if len(converted) != 1 {
		t.Fatalf("expected 1 message, got %d", len(converted))
	}
	if converted[0].ReasoningContent != "thought process" {
		t.Errorf("expected reasoning content echoed, got %q", converted[0].ReasoningContent)
	}
	if converted[0].Content != "final answer" {
		t.Errorf("expected content 'final answer', got %v", converted[0].Content)
	}
}

func TestExecuteStreamSynchronousError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer server.Close()

	req, _ := http.NewRequest("POST", server.URL, strings.NewReader("{}"))
	ch, err := ExecuteStream(context.Background(), server.Client(), req)
	if err == nil {
		t.Fatal("expected error on non-2xx status")
	}
	if ch != nil {
		t.Fatal("expected nil channel on synchronous error")
	}
}

func TestStreamHandlerOrder(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"reasoning_content":"think first","content":"text second"}}]}

data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

data: [DONE]
`
	ch := StreamHandler(context.Background(), io.NopCloser(strings.NewReader(sseData)))

	var events []ir.Event
	for ev := range ch {
		events = append(events, ev)
	}

	if len(events) < 4 {
		t.Fatalf("expected at least 4 events, got %d", len(events))
	}

	// 1. MessageStart
	if _, ok := events[0].(ir.EventMessageStart); !ok {
		t.Errorf("event 0 want EventMessageStart, got %T", events[0])
	}
	// 2. ReasoningDelta must precede TextDelta
	if r, ok := events[1].(ir.EventReasoningDelta); !ok || r.Text != "think first" {
		t.Errorf("event 1 want EventReasoningDelta, got %v", events[1])
	}
	// 3. TextDelta
	if txt, ok := events[2].(ir.EventTextDelta); !ok || txt.Text != "text second" {
		t.Errorf("event 2 want EventTextDelta, got %v", events[2])
	}
}

func TestExecuteGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := ChatCompletionResponse{
			ID: "gen-1",
			Choices: []ChatChoice{
				{
					Message: ChatMessage{
						Role:             "assistant",
						Content:          "generated response",
						ReasoningContent: "internal thoughts",
					},
					FinishReason: "stop",
				},
			},
			Usage: &ChatUsage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	req, _ := http.NewRequest("POST", server.URL, strings.NewReader("{}"))
	resp, err := ExecuteGenerate(context.Background(), server.Client(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ID != "gen-1" {
		t.Errorf("expected ID gen-1, got %s", resp.ID)
	}
	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(resp.Message.Blocks))
	}
	if rBlock, ok := resp.Message.Blocks[0].(ir.ReasoningBlock); !ok || rBlock.Text != "internal thoughts" {
		t.Errorf("expected reasoning block first, got %v", resp.Message.Blocks[0])
	}
	if tBlock, ok := resp.Message.Blocks[1].(ir.TextBlock); !ok || tBlock.Text != "generated response" {
		t.Errorf("expected text block second, got %v", resp.Message.Blocks[1])
	}
}
