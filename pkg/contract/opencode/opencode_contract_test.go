package opencode

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// TestOpenCode_StreamToolCallReassembly verifies that fragmented streaming tool
// calls from upstream are reassembled correctly for the downstream OpenCode
// client.
func TestOpenCode_StreamToolCallReassembly(t *testing.T) {
	h := NewTestHarness(t)

	h.FakeUpstream.QueueSSEStream(
		NewSSEStream().
			ToolCallStart(0, "call_abc", "bash").
			ToolCallArgs(0, `{"com`).
			ToolCallArgs(0, `mand": "ls"}`).
			FinishReason("tool_calls"),
	)

	req := map[string]any{
		"model": "opencode/dummy",
		"messages": []map[string]any{
			{"role": "user", "content": "List files"},
		},
	}

	obs, resp, err := h.StreamChat(context.Background(), req)
	if err != nil {
		t.Fatalf("StreamChat failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	if len(obs.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(obs.ToolCalls))
	}
	tc := obs.ToolCalls[0]
	if tc.Index != 0 {
		t.Errorf("expected index 0, got %d", tc.Index)
	}
	if tc.ID != "call_abc" {
		t.Errorf("expected tool call id %q, got %q", "call_abc", tc.ID)
	}
	if tc.Name != "bash" {
		t.Errorf("expected tool call name %q, got %q", "bash", tc.Name)
	}
	wantArgs := `{"command": "ls"}`
	if tc.Arguments != wantArgs {
		t.Errorf("expected arguments %q, got %q", wantArgs, tc.Arguments)
	}

	if obs.FinishReason != "tool_calls" {
		t.Errorf("expected finish_reason %q, got %q", "tool_calls", obs.FinishReason)
	}
}

// TestOpenCode_StreamEmptyDeltaNoToolCallLie verifies that the proxy does not
// report finish_reason=tool_calls when the upstream never streamed a tool call.
func TestOpenCode_StreamEmptyDeltaNoToolCallLie(t *testing.T) {
	h := NewTestHarness(t)

	h.FakeUpstream.QueueSSEStream(
		NewSSEStream().
			TextDelta("Hello ").
			TextDelta("world").
			FinishReason("stop"),
	)

	req := map[string]any{
		"model": "opencode/dummy",
		"messages": []map[string]any{
			{"role": "user", "content": "Say hello"},
		},
	}

	obs, resp, err := h.StreamChat(context.Background(), req)
	if err != nil {
		t.Fatalf("StreamChat failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	if obs.Text != "Hello world" {
		t.Errorf("expected text %q, got %q", "Hello world", obs.Text)
	}
	if obs.FinishReason != "stop" {
		t.Errorf("expected finish_reason %q, got %q", "stop", obs.FinishReason)
	}
	if len(obs.ToolCalls) != 0 {
		t.Errorf("expected no tool calls, got %d", len(obs.ToolCalls))
	}

	// Also exercise the pure empty-delta + stop path on a fresh harness.
	h2 := NewTestHarness(t)
	h2.FakeUpstream.QueueSSEStream(
		NewSSEStream().
			FinishReason("stop"),
	)

	obs2, resp2, err2 := h2.StreamChat(context.Background(), req)
	if err2 != nil {
		t.Fatalf("StreamChat (empty delta) failed: %v", err2)
	}
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp2.StatusCode)
	}
	if obs2.FinishReason != "stop" {
		t.Errorf("expected empty-delta finish_reason %q, got %q", "stop", obs2.FinishReason)
	}
	if len(obs2.ToolCalls) != 0 {
		t.Errorf("expected no tool calls for empty delta, got %d", len(obs2.ToolCalls))
	}
}

// TestOpenCode_ToolResultReplayCorrelation verifies that a tool result message
// sent by the downstream client is forwarded upstream with the correct
// tool_call_id correlation.
func TestOpenCode_ToolResultReplayCorrelation(t *testing.T) {
	h := NewTestHarness(t)
	h.FakeUpstream.QueueChatCompletion("ok")

	req := map[string]any{
		"model": "opencode/dummy",
		"messages": []map[string]any{
			{"role": "user", "content": "Read a file"},
			{
				"role": "assistant",
				"content": "",
				"tool_calls": []any{
					map[string]any{
						"id":   "call_123",
						"type": "function",
						"function": map[string]any{
							"name":      "read_files",
							"arguments": `{"file":"test.go"}`,
						},
					},
				},
			},
			{
				"role":         "tool",
				"tool_call_id": "call_123",
				"content":      "file contents here",
			},
		},
	}

	resp, body, err := h.PostChat(context.Background(), req)
	if err != nil {
		t.Fatalf("PostChat failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d: %v", resp.StatusCode, body)
	}

	rec := h.FakeUpstream.LastRequest()
	if rec == nil {
		t.Fatal("expected upstream request to be recorded")
	}

	lastMsg := rec.LastMessage()
	if lastMsg == nil {
		t.Fatal("expected upstream messages")
	}
	if lastMsg.Role != "tool" {
		t.Errorf("expected last upstream message role %q, got %q", "tool", lastMsg.Role)
	}
	if lastMsg.ToolCallID != "call_123" {
		t.Errorf("expected tool_call_id %q upstream, got %q", "call_123", lastMsg.ToolCallID)
	}
	if lastMsg.Name != "read_files" {
		t.Errorf("expected tool result name %q upstream, got %q", "read_files", lastMsg.Name)
	}
}

// TestOpenCode_ParallelToolCalls verifies that two interleaved tool calls are
// both delivered downstream with correct indices, names, and arguments.
func TestOpenCode_ParallelToolCalls(t *testing.T) {
	h := NewTestHarness(t)

	h.FakeUpstream.QueueSSEStream(
		NewSSEStream().
			ToolCallStart(0, "call_1", "read_files").
			ToolCallStart(1, "call_2", "bash").
			ToolCallArgs(0, `{"file": "a.txt"}`).
			ToolCallArgs(1, `{"command": "ls"}`).
			FinishReason("tool_calls"),
	)

	req := map[string]any{
		"model": "opencode/dummy",
		"messages": []map[string]any{
			{"role": "user", "content": "Parallel task"},
		},
	}

	obs, resp, err := h.StreamChat(context.Background(), req)
	if err != nil {
		t.Fatalf("StreamChat failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	if len(obs.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(obs.ToolCalls))
	}

	tc0 := obs.GetToolCall(0)
	if tc0 == nil {
		t.Fatal("missing tool call at index 0")
	}
	if tc0.Name != "read_files" {
		t.Errorf("expected index-0 name %q, got %q", "read_files", tc0.Name)
	}
	if tc0.Arguments != `{"file": "a.txt"}` {
		t.Errorf("expected index-0 arguments %q, got %q", `{"file": "a.txt"}`, tc0.Arguments)
	}

	tc1 := obs.GetToolCall(1)
	if tc1 == nil {
		t.Fatal("missing tool call at index 1")
	}
	if tc1.Name != "bash" {
		t.Errorf("expected index-1 name %q, got %q", "bash", tc1.Name)
	}
	if tc1.Arguments != `{"command": "ls"}` {
		t.Errorf("expected index-1 arguments %q, got %q", `{"command": "ls"}`, tc1.Arguments)
	}
}

// TestOpenCode_ToolSchemaPassthrough verifies that tool definitions supplied by
// the OpenCode client are forwarded in the upstream request body.
func TestOpenCode_ToolSchemaPassthrough(t *testing.T) {
	h := NewTestHarness(t)
	h.FakeUpstream.QueueChatCompletion("ok")

	req := map[string]any{
		"model": "opencode/dummy",
		"messages": []map[string]any{
			{"role": "user", "content": "Use tools"},
		},
		"tools": []any{
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        "read_files",
					"description": "Read files from disk",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"file": map[string]any{"type": "string"},
						},
						"required": []string{"file"},
					},
				},
			},
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        "bash",
					"description": "Run shell commands",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"command": map[string]any{"type": "string"},
						},
						"required": []string{"command"},
					},
				},
			},
		},
	}

	resp, body, err := h.PostChat(context.Background(), req)
	if err != nil {
		t.Fatalf("PostChat failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d: %v", resp.StatusCode, body)
	}

	rec := h.FakeUpstream.LastRequest()
	if rec == nil {
		t.Fatal("expected upstream request to be recorded")
	}

	if len(rec.Tools) != 2 {
		t.Fatalf("expected 2 tools upstream, got %d", len(rec.Tools))
	}
	if !rec.HasTool("read_files") {
		t.Errorf("expected upstream to receive read_files tool")
	}
	if !rec.HasTool("bash") {
		t.Errorf("expected upstream to receive bash tool")
	}

	readFiles := rec.GetTool("read_files")
	if readFiles == nil || readFiles.Function.Parameters == nil {
		t.Fatal("read_files tool parameters missing")
	}
	if readFiles.Function.Parameters["type"] != "object" {
		t.Errorf("expected read_files parameters type object, got %v", readFiles.Function.Parameters["type"])
	}
}

// TestOpenCode_UpstreamErrorPropagationWithoutFailover verifies that HTTP 429
// and 401 errors from upstream are surfaced cleanly without being swallowed or
// masked by blind failover to another provider.
func TestOpenCode_UpstreamErrorPropagationWithoutFailover(t *testing.T) {
	tests := []struct {
		status   int
		message  string
		errType  string
	}{
		{http.StatusTooManyRequests, "rate limit hit", "rate_limit_exceeded"},
		{http.StatusUnauthorized, "invalid credentials", "authentication_error"},
	}

	for _, tt := range tests {
		t.Run(tt.errType, func(t *testing.T) {
			h := NewTestHarness(t)
			h.FakeUpstream.QueueError(tt.status, tt.message)

			req := map[string]any{
				"model": "opencode/dummy",
				"messages": []map[string]any{
					{"role": "user", "content": "Trigger error"},
				},
			}

			resp, body, err := h.PostChat(context.Background(), req)
			if err != nil {
				t.Fatalf("PostChat failed: %v", err)
			}
			if resp.StatusCode != tt.status {
				t.Errorf("expected status %d, got %d", tt.status, resp.StatusCode)
			}

			errObj, ok := body["error"].(map[string]any)
			if !ok {
				t.Fatalf("expected error object in body, got %#v", body)
			}
			msg, _ := errObj["message"].(string)
			if !strings.Contains(msg, tt.message) {
				t.Errorf("expected error message to contain %q, got %q", tt.message, msg)
			}
			if errObj["type"] != tt.errType {
				t.Errorf("expected error type %q, got %v", tt.errType, errObj["type"])
			}

			// No blind failover should have occurred; only one upstream request.
			if h.FakeUpstream.RequestCount() != 1 {
				t.Errorf("expected 1 upstream request, got %d", h.FakeUpstream.RequestCount())
			}
		})
	}
}

// TestOpenCode_StreamingUsageTracking verifies that a stream_options
// include_usage request causes the final usage chunk to appear before [DONE].
func TestOpenCode_StreamingUsageTracking(t *testing.T) {
	h := NewTestHarness(t)

	h.FakeUpstream.QueueSSEStream(
		NewSSEStream().
			TextDelta("Sure").
			Usage(12, 5, 17),
	)

	req := map[string]any{
		"model": "opencode/dummy",
		"messages": []map[string]any{
			{"role": "user", "content": "Count tokens"},
		},
		"stream_options": map[string]any{
			"include_usage": true,
		},
	}

	obs, resp, err := h.StreamChat(context.Background(), req)
	if err != nil {
		t.Fatalf("StreamChat failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	// Confirm stream_options were forwarded upstream.
	rec := h.FakeUpstream.LastRequest()
	if rec == nil {
		t.Fatal("expected upstream request to be recorded")
	}
	sopts, ok := rec.JSON["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("expected stream_options upstream, got %#v", rec.JSON["stream_options"])
	}
	if include, _ := sopts["include_usage"].(bool); !include {
		t.Errorf("expected stream_options.include_usage=true upstream")
	}

	if obs.Usage == nil {
		t.Fatal("expected usage chunk downstream, got nil")
	}
	if obs.Usage.PromptTokens != 12 {
		t.Errorf("expected prompt_tokens 12, got %d", obs.Usage.PromptTokens)
	}
	if obs.Usage.CompletionTokens != 5 {
		t.Errorf("expected completion_tokens 5, got %d", obs.Usage.CompletionTokens)
	}
	if obs.Usage.TotalTokens != 17 {
		t.Errorf("expected total_tokens 17, got %d", obs.Usage.TotalTokens)
	}

	// The [DONE] marker must be present.
	foundDone := false
	for _, ev := range obs.RawEvents {
		if ev.Data == "[DONE]" {
			foundDone = true
			break
		}
	}
	if !foundDone {
		t.Errorf("expected [DONE] marker in stream")
	}
}
