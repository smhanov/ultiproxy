package codex

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func codexRecorder(t *testing.T, respond func(w http.ResponseWriter)) (*httptest.Server, func() map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var last map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var p map[string]any
		_ = json.Unmarshal(body, &p)
		mu.Lock()
		last = p
		mu.Unlock()
		respond(w)
	}))
	t.Cleanup(srv.Close)
	return srv, func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return last
	}
}

// TestBuildPayload_ImageBlocksForwarded (AC5): image content blocks in the IR
// must reach the Codex upstream request as multimodal content parts.
// Regression: BuildPayload only collected text/tool blocks, so images were
// dropped and the upstream saw a text-only prompt.
func TestBuildPayload_ImageBlocksForwarded(t *testing.T) {
	dataURL := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
	httpURL := "https://example.com/photo.jpg"

	srv, lastPayload := codexRecorder(t, func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "codex-img",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
			},
		})
	})

	p := New(Config{BaseURL: srv.URL, StaticToken: "tok", HTTPClient: srv.Client()})

	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "Describe these images"},
				ir.ImageBlock{URL: dataURL, Detail: "high"},
				ir.ImageBlock{URL: httpURL},
			},
		},
	}

	if _, err := p.Generate(context.Background(), msgs, provider.WithModel("gpt-5.5")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := lastPayload()
	rawMsgs, ok := payload["messages"].([]any)
	if !ok || len(rawMsgs) != 1 {
		t.Fatalf("expected 1 upstream message, got %v", payload["messages"])
	}
	msg := rawMsgs[0].(map[string]any)
	if msg["role"] != "user" {
		t.Fatalf("expected role user, got %v", msg["role"])
	}

	parts, ok := msg["content"].([]any)
	if !ok {
		t.Fatalf("multimodal message content must be a parts array, got %T: %v", msg["content"], msg["content"])
	}
	if len(parts) != 3 {
		t.Fatalf("expected 3 content parts (text + 2 images), got %d: %v", len(parts), parts)
	}

	p0, _ := parts[0].(map[string]any)
	if p0["type"] != "text" || p0["text"] != "Describe these images" {
		t.Errorf("unexpected text part: %v", parts[0])
	}

	p1, _ := parts[1].(map[string]any)
	if p1["type"] != "image_url" {
		t.Fatalf("expected image_url part, got %v", parts[1])
	}
	iu1, _ := p1["image_url"].(map[string]any)
	if iu1 == nil || iu1["url"] != dataURL {
		t.Errorf("image_url[0] = %v, want url %q", p1["image_url"], dataURL)
	}

	p2, _ := parts[2].(map[string]any)
	if p2["type"] != "image_url" {
		t.Fatalf("expected image_url part, got %v", parts[2])
	}
	iu2, _ := p2["image_url"].(map[string]any)
	if iu2 == nil || iu2["url"] != httpURL {
		t.Errorf("image_url[1] = %v, want url %q", p2["image_url"], httpURL)
	}
}

// TestBuildPayload_TextOnlyStaysAString: non-multimodal messages keep the
// legacy string content shape so existing upstreams see no diff.
func TestBuildPayload_TextOnlyStaysAString(t *testing.T) {
	payload, err := BuildPayload(
		[]*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "plain"}}}},
		provider.NewRequestConfig(provider.WithModel("gpt-5.5")),
		false,
	)
	if err != nil {
		t.Fatalf("BuildPayload failed: %v", err)
	}
	msgs := payload["messages"].([]map[string]any)
	if msgs[0]["content"] != "plain" {
		t.Fatalf("text-only content = %v (%T), want the string \"plain\"", msgs[0]["content"], msgs[0]["content"])
	}
}

// TestStream_ToolCallDeltas (AC7): a Codex stream that carries
// delta.tool_calls must surface tool_call_start / tool_arguments_delta /
// tool_call_stop IR events. Regression: the streaming parser ignored
// delta.tool_calls entirely, so tool-using clients hung with no tool call.
func TestStream_ToolCallDeltas(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"codex-tc-1","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
		`data: {"id":"codex-tc-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`,
		`data: {"id":"codex-tc-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"Oslo\"}"}}]}}]}`,
		`data: {"id":"codex-tc-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{}"}}]}}]}`,
		`data: {"id":"codex-tc-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	}))
	defer srv.Close()

	p := New(Config{BaseURL: srv.URL, StaticToken: "tok", HTTPClient: srv.Client()})

	msgs := []*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "weather?"}}}}
	ch, err := p.Stream(context.Background(), msgs, provider.WithModel("gpt-5.5"))
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var starts []ir.EventToolCallStart
	var argDeltas []ir.EventToolArgumentsDelta
	var stops []ir.EventToolCallStop
	var finish string
	for ev := range ch {
		switch e := ev.(type) {
		case ir.EventToolCallStart:
			starts = append(starts, e)
		case ir.EventToolArgumentsDelta:
			argDeltas = append(argDeltas, e)
		case ir.EventToolCallStop:
			stops = append(stops, e)
		case ir.EventMessageStop:
			finish = e.FinishReason
		}
	}

	if len(starts) != 2 {
		t.Fatalf("expected 2 tool_call_start events, got %d: %+v", len(starts), starts)
	}
	if starts[0].ID != "call_1" || starts[0].Name != "get_weather" || starts[0].Index != 0 {
		t.Errorf("starts[0] = %+v, want {Index:0 ID:call_1 Name:get_weather}", starts[0])
	}
	if starts[1].ID != "call_2" || starts[1].Name != "get_time" || starts[1].Index != 1 {
		t.Errorf("starts[1] = %+v, want {Index:1 ID:call_2 Name:get_time}", starts[1])
	}

	if len(argDeltas) != 2 {
		t.Fatalf("expected 2 tool_arguments_delta events, got %d: %+v", len(argDeltas), argDeltas)
	}
	if argDeltas[0].Arguments != `{"city":"Oslo"}` || argDeltas[0].Index != 0 {
		t.Errorf("argDeltas[0] = %+v, want arguments {\"city\":\"Oslo\"} at index 0", argDeltas[0])
	}
	if argDeltas[1].Arguments != "{}" || argDeltas[1].Index != 1 {
		t.Errorf("argDeltas[1] = %+v, want arguments {} at index 1", argDeltas[1])
	}

	// Every streamed tool call index must be closed. Codex (like the hublane
	// bridge) emits a stop per tool-call chunk, so assert on coverage rather
	// than on an exact count.
	stopped := map[int]bool{}
	for _, st := range stops {
		stopped[st.Index] = true
	}
	for _, idx := range []int{0, 1} {
		if !stopped[idx] {
			t.Fatalf("tool call index %d never received a tool_call_stop event: %+v", idx, stops)
		}
	}
	if finish != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", finish)
	}
}
