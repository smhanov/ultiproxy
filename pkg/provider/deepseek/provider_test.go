package deepseek

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestDeepSeekMultiTurnReasoningEcho(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "deepseek", "multi-turn-history.json")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	var msgs []*ir.Message
	if err := json.Unmarshal(data, &msgs); err != nil {
		t.Fatalf("failed to unmarshal fixture messages: %v", err)
	}

	var capturedBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &capturedBody)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "chatcmpl-ds-multiturn",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":              "assistant",
						"content":           "24",
						"reasoning_content": "12 * 2 = 24.",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-deepseek-key",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create DeepSeek provider: %v", err)
	}

	resp, err := p.Generate(context.Background(), msgs, provider.WithModel("deepseek-reasoner"))
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// 1. Assert request payload echoed reasoning_content on prior assistant turn
	rawMsgs := capturedBody["messages"].([]any)
	if len(rawMsgs) != 3 {
		t.Fatalf("expected 3 messages sent to DeepSeek, got %d", len(rawMsgs))
	}

	assistantMsg := rawMsgs[1].(map[string]any)
	rc, ok := assistantMsg["reasoning_content"].(string)
	if !ok || rc != "The user is asking for sqrt(144). 12 * 12 = 144." {
		t.Errorf("expected echoed reasoning_content, got %v", assistantMsg["reasoning_content"])
	}
	if assistantMsg["content"] != "12" {
		t.Errorf("expected content '12', got %v", assistantMsg["content"])
	}

	// 2. Assert returned ir.Response extracted reasoning into reasoning block
	if resp == nil || resp.Message == nil {
		t.Fatal("expected non-nil response message")
	}
	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("expected 2 blocks in response, got %d", len(resp.Message.Blocks))
	}
	reasonBlock, ok := resp.Message.Blocks[0].(ir.ReasoningBlock)
	if !ok || reasonBlock.Text != "12 * 2 = 24." {
		t.Errorf("expected reasoning block, got %v", resp.Message.Blocks[0])
	}
	textBlock, ok := resp.Message.Blocks[1].(ir.TextBlock)
	if !ok || textBlock.Text != "24" {
		t.Errorf("expected text block '24', got %v", resp.Message.Blocks[1])
	}
}

func TestDeepSeekStreamingReasoningOrder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Send chunk with both reasoning_content and content
		_, _ = w.Write([]byte("data: {\"id\":\"ds-stream-01\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"I think therefore I am\",\"content\":\"Hello world\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"ds-stream-01\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-key",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "hi"},
			},
		},
	}

	ch, err := p.Stream(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var received []ir.Event
	for ev := range ch {
		received = append(received, ev)
	}

	// Verify order: EventMessageStart, then EventReasoningDelta, then EventTextDelta
	var reasoningIdx, textIdx int = -1, -1
	for i, ev := range received {
		switch ev.(type) {
		case ir.EventReasoningDelta:
			if reasoningIdx == -1 {
				reasoningIdx = i
			}
		case ir.EventTextDelta:
			if textIdx == -1 {
				textIdx = i
			}
		}
	}

	if reasoningIdx == -1 {
		t.Fatal("never received EventReasoningDelta")
	}
	if textIdx == -1 {
		t.Fatal("never received EventTextDelta")
	}
	if reasoningIdx >= textIdx {
		t.Errorf("expected EventReasoningDelta (idx %d) before EventTextDelta (idx %d)", reasoningIdx, textIdx)
	}
}
