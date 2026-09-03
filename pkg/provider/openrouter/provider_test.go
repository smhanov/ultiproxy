package openrouter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestOpenRouterGenerateCostCapture(t *testing.T) {
	var capturedAuth string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "gen-openrouter-01",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Aggregated answer",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     150,
				"completion_tokens": 50,
				"total_tokens":      200,
				"total_cost":        0.00285,
			},
		})
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-openrouter-key",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "Calculate cost"},
			},
		},
	}

	resp, err := p.Generate(context.Background(), msgs, provider.WithModel("anthropic/claude-3.5-sonnet"))
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if capturedAuth != "Bearer test-openrouter-key" {
		t.Errorf("expected bearer token, got %s", capturedAuth)
	}

	if resp.Usage == nil {
		t.Fatal("expected non-nil usage")
	}
	if resp.Usage.Cost != 0.00285 {
		t.Errorf("expected Cost 0.00285, got %v", resp.Usage.Cost)
	}
}

func TestOpenRouterStreamCostAndReasoning(t *testing.T) {
	var capturedPayload map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedPayload)

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"or-str-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"deep thought\",\"content\":\"hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"or-str-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20,\"total_tokens\":120,\"total_cost\":0.0054}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		APIKey:     "test-openrouter-key",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "Think and answer"},
			},
		},
	}

	ch, err := p.Stream(context.Background(), msgs, provider.WithReasoningEffort("high"))
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var foundCost bool
	var capturedCost float64
	var foundReasoning bool

	for ev := range ch {
		switch e := ev.(type) {
		case ir.EventReasoningDelta:
			if e.Text == "deep thought" {
				foundReasoning = true
			}
		case ir.EventUsageUpdate:
			if e.Cost == 0.0054 {
				foundCost = true
				capturedCost = e.Cost
			}
		}
	}

	if !foundReasoning {
		t.Error("expected EventReasoningDelta")
	}
	if !foundCost || capturedCost != 0.0054 {
		t.Errorf("expected usage update cost 0.0054, got %v (found: %v)", capturedCost, foundCost)
	}
	if capturedPayload["reasoning_effort"] != "high" {
		t.Errorf("expected reasoning_effort high, got %v", capturedPayload["reasoning_effort"])
	}
}
