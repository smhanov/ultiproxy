package vllm

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

func TestVLLMStartupModelsFetchAndVision(t *testing.T) {
	var capturedPayload map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models", "/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"id": "meta-llama/Llama-3-8B-Instruct"},
					{"id": "Qwen/Qwen2.5-Coder-32B"},
				},
			})

		case "/v1/chat/completions", "/chat/completions":
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &capturedPayload)

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "vllm-cmpl-01",
				"choices": []map[string]any{
					{
						"index": 0,
						"message": map[string]any{
							"role":    "assistant",
							"content": "Local model output",
						},
						"finish_reason": "stop",
					},
				},
			})

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	// 1. Startup model discovery
	p, err := New(Config{
		BaseURL:    server.URL + "/v1",
		APIKey:     "optional-vllm-key",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create vLLM provider: %v", err)
	}

	models := p.Models()
	if len(models) != 2 {
		t.Fatalf("expected 2 discovered models, got %d: %v", len(models), models)
	}
	if models[0] != "meta-llama/Llama-3-8B-Instruct" {
		t.Errorf("expected first model Llama-3-8B-Instruct, got %s", models[0])
	}

	// 2. Default model resolution and Vision handling
	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "Explain diagram"},
				ir.ImageBlock{URL: "http://example.com/diag.png", Detail: "low"},
			},
		},
	}

	resp, err := p.Generate(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}

	// Used discovered model
	if capturedPayload["model"] != "meta-llama/Llama-3-8B-Instruct" {
		t.Errorf("expected default model from discovery, got %v", capturedPayload["model"])
	}

	// Vision content parts
	rawMsgs := capturedPayload["messages"].([]any)
	parts := rawMsgs[0].(map[string]any)["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts for vision, got %d", len(parts))
	}
	imgPart := parts[1].(map[string]any)
	if imgPart["type"] != "image_url" {
		t.Errorf("expected image_url type, got %v", imgPart["type"])
	}
	imgURL := imgPart["image_url"].(map[string]any)
	if imgURL["url"] != "http://example.com/diag.png" || imgURL["detail"] != "low" {
		t.Errorf("unexpected image_url: %+v", imgURL)
	}
}

func TestVLLMStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"vllm-stream-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"vLLM streamed\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "test"},
			},
		},
	}

	ch, err := p.Stream(context.Background(), msgs, provider.WithModel("custom-model"))
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var hasText bool
	for ev := range ch {
		if txt, ok := ev.(ir.EventTextDelta); ok {
			if txt.Text == "vLLM streamed" {
				hasText = true
			}
		}
	}

	if !hasText {
		t.Error("expected streamed text delta")
	}
}
