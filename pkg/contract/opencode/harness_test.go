package opencode

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

func TestHarness_NonStreamingChatCompletion(t *testing.T) {
	h := NewTestHarness(t)
	h.FakeUpstream.QueueChatCompletion("Hello from fake upstream")

	req := map[string]any{
		"model": "opencode/dummy",
		"messages": []map[string]any{
			{"role": "user", "content": "Hi"},
		},
	}

	resp, body, err := h.PostChat(context.Background(), req)
	if err != nil {
		t.Fatalf("PostChat failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d: %v", resp.StatusCode, body)
	}

	choices, ok := body["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("expected choices in response, got %#v", body)
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		t.Fatalf("expected choice map, got %#v", choices[0])
	}
	msg, ok := choice["message"].(map[string]any)
	if !ok {
		t.Fatalf("expected message map, got %#v", choice["message"])
	}
	content, _ := msg["content"].(string)
	if content != "Hello from fake upstream" {
		t.Errorf("expected content %q, got %q", "Hello from fake upstream", content)
	}

	reqs := h.FakeUpstream.Requests()
	if len(reqs) != 1 {
		t.Fatalf("expected 1 upstream request, got %d", len(reqs))
	}
	rec := reqs[0]
	if rec.Method != http.MethodPost {
		t.Errorf("expected POST, got %s", rec.Method)
	}
	if !strings.HasSuffix(rec.Path, "/chat/completions") {
		t.Errorf("expected path to end with /chat/completions, got %s", rec.Path)
	}
	if rec.UserPrompt() != "Hi" {
		t.Errorf("expected upstream user prompt %q, got %q", "Hi", rec.UserPrompt())
	}
	if rec.Model != "dummy" {
		t.Errorf("expected upstream model %q, got %q", "dummy", rec.Model)
	}
}

func TestHarness_StreamingChatCompletion(t *testing.T) {
	h := NewTestHarness(t)
	h.FakeUpstream.QueueSSE(
		SSEChunk{Content: "Hello "},
		SSEChunk{Content: "from fake upstream"},
		SSEChunk{FinishReason: "stop"},
	)

	req := map[string]any{
		"model": "opencode/dummy",
		"messages": []map[string]any{
			{"role": "user", "content": "Hi"},
		},
	}

	obs, resp, err := h.StreamChat(context.Background(), req)
	if err != nil {
		t.Fatalf("StreamChat failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if obs.Text != "Hello from fake upstream" {
		t.Errorf("expected streamed text %q, got %q", "Hello from fake upstream", obs.Text)
	}
	if obs.FinishReason != "stop" {
		t.Errorf("expected finish_reason %q, got %q", "stop", obs.FinishReason)
	}

	reqs := h.FakeUpstream.Requests()
	if len(reqs) != 1 {
		t.Fatalf("expected 1 upstream request, got %d", len(reqs))
	}
	if !reqs[0].Stream {
		t.Errorf("expected upstream request to be streaming")
	}
}
