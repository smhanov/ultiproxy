package hublane

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/ultiproxy/pkg/codec"
	"github.com/smhanov/ultiproxy/pkg/ir"
)

func TestIRToHubPrompt_AllBlocks(t *testing.T) {
	msgs := []*ir.Message{
		{
			Role: "system",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "sys"},
			},
		},
		{
			Role: "user",
			Blocks: []ir.Block{
				&ir.TextBlock{Text: "hello"},
				ir.ImageBlock{URL: "http://example.com/img.png", Detail: "high"},
				&ir.ImageBlock{URL: "http://example.com/img2.png"},
				ir.ReasoningBlock{ReasoningKind: ir.ReasoningText, Text: "think"},
				&ir.ReasoningBlock{Text: "deep"},
				ir.ToolCallBlock{Index: 0, ID: "tc1", Name: "tool", Arguments: "{}"},
				&ir.ToolCallBlock{Index: 1, ID: "tc2", Name: "tool2", Arguments: "[]"},
			},
		},
		{
			Role: "tool",
			Blocks: []ir.Block{
				ir.ToolResultBlock{ToolCallID: "tc1", Name: "tool", Content: "result"},
				&ir.ToolResultBlock{ToolCallID: "tc2", Name: "tool2", Content: "result2"},
			},
		},
	}

	hubMsgs := IRToHubPrompt(msgs)
	if len(hubMsgs) != 4 {
		t.Fatalf("expected 4 hub messages, got %d", len(hubMsgs))
	}

	if hubMsgs[0].Role != llmhub.RoleSystem {
		t.Fatalf("expected role system, got %s", hubMsgs[0].Role)
	}
	if len(hubMsgs[0].Content) != 1 {
		t.Fatalf("expected 1 system content part, got %d", len(hubMsgs[0].Content))
	}
	if text, ok := hubMsgs[0].Content[0].(*llmhub.TextContent); !ok || text.Text != "sys" {
		t.Fatalf("unexpected system content: %+v", hubMsgs[0].Content[0])
	}

	if hubMsgs[1].Role != llmhub.RoleUser {
		t.Fatalf("expected role user, got %s", hubMsgs[1].Role)
	}
	parts := hubMsgs[1].Content
		if len(parts) != 7 {
			t.Fatalf("expected 7 user content parts, got %d", len(parts))
		}
		assertTextContent(t, parts[0], "hello")
		assertImageContent(t, parts[1], "http://example.com/img.png", "high")
		assertImageContent(t, parts[2], "http://example.com/img2.png", "")
		assertReasoningContent(t, parts[3], "think")
		assertReasoningContent(t, parts[4], "deep")
		assertToolCallContent(t, parts[5], "tc1", "tool", "{}")
		assertToolCallContent(t, parts[6], "tc2", "tool2", "[]")

	// Tool result blocks become separate tool messages.
	for i, wantID := range []string{"tc1", "tc2"} {
		msg := hubMsgs[i+2]
		if msg.Role != llmhub.RoleTool {
			t.Fatalf("expected role tool, got %s", msg.Role)
		}
		if msg.Meta["tool_call_id"] != wantID {
			t.Fatalf("expected tool_call_id %s, got %s", wantID, msg.Meta["tool_call_id"])
		}
		if len(msg.Content) != 1 {
			t.Fatalf("expected 1 tool result part, got %d", len(msg.Content))
		}
	}
}

func TestIRToHubPrompt_NilAndEmpty(t *testing.T) {
	msgs := []*ir.Message{
		nil,
		{Role: "user", Blocks: []ir.Block{nil}},
		{Role: "assistant", Blocks: []ir.Block{}},
	}

	hubMsgs := IRToHubPrompt(msgs)
	if len(hubMsgs) != 0 {
		t.Fatalf("expected 0 hub messages, got %d", len(hubMsgs))
	}
}

func TestHubResponseToIR_AllContentTypes(t *testing.T) {
	resp := &llmhub.Response{
		ID: "resp-123",
		Content: []llmhub.ContentPart{
			llmhub.Text("hello "),
			llmhub.Reasoning("thinking"),
			llmhub.ToolCall("tc1", "weather", `{"city":"nyc"}`),
		},
		Usage: llmhub.UsageMetadata{
			PromptTokens:        10,
			CompletionTokens:    5,
			TotalTokens:         15,
			ReasoningTokens:     1,
			CacheReadTokens:     2,
			CacheCreationTokens: 3,
			Cost:                0.001,
		},
		Raw: map[string]any{"finish_reason": "tool_calls"},
	}

	irResp := HubResponseToIR(resp, "model-x")
	if irResp == nil {
		t.Fatal("expected non-nil response")
	}
	if irResp.ID != "resp-123" || irResp.UpstreamID != "resp-123" {
		t.Fatalf("unexpected ids: %+v", irResp)
	}
	if irResp.FinishReason != "tool_calls" {
		t.Fatalf("expected finish_reason tool_calls, got %q", irResp.FinishReason)
	}
	if irResp.Message == nil || irResp.Message.Role != "assistant" {
		t.Fatalf("unexpected message: %+v", irResp.Message)
	}

	blocks := irResp.Message.Blocks
	if len(blocks) != 3 {
		t.Fatalf("expected 3 blocks, got %d", len(blocks))
	}
	assertIRReasoningBlock(t, blocks[0], ir.ReasoningText, "thinking")
	assertIRTextBlock(t, blocks[1], "hello ")
	assertIRToolCallBlock(t, blocks[2], "tc1", "weather", `{"city":"nyc"}`)

	if irResp.Usage == nil {
		t.Fatal("expected usage")
	}
	wantUsage := &ir.Usage{
		PromptTokens:             10,
		CompletionTokens:         5,
		TotalTokens:              15,
		ReasoningTokens:          1,
		CacheReadInputTokens:     2,
		CacheCreationInputTokens: 3,
		Cost:                     0.001,
	}
	if !reflect.DeepEqual(irResp.Usage, wantUsage) {
		t.Fatalf("usage mismatch: %+v, want %+v", irResp.Usage, wantUsage)
	}
}

func TestHubResponseToIR_Nil(t *testing.T) {
	if HubResponseToIR(nil, "m") != nil {
		t.Fatal("expected nil for nil response")
	}
}

func TestStreamBridge_Events(t *testing.T) {
	ctx := context.Background()
	in := make(chan llmhub.StreamChunk, 8)

	in <- llmhub.StreamChunk{ID: "stream-1"}
	in <- llmhub.StreamChunk{Delta: "Hello", ReasoningDelta: "think"}
	in <- llmhub.StreamChunk{
		ToolCalls: []*llmhub.ToolCallContent{
			llmhub.ToolCallWithIndex(0, "tc1", "weather", `{"ci`),
		},
	}
	in <- llmhub.StreamChunk{
		ToolCalls: []*llmhub.ToolCallContent{
			llmhub.ToolCallWithIndex(0, "", "", `ty":"nyc"}`),
		},
	}
	in <- llmhub.StreamChunk{
		Usage: &llmhub.UsageMetadata{
			PromptTokens:     5,
			CompletionTokens: 3,
			TotalTokens:      8,
		},
	}
	in <- llmhub.StreamChunk{FinishReason: "stop", Done: true}
	close(in)

	out := StreamBridge(ctx, in)
	events := drainEvents(t, out)

	want := []ir.Event{
		ir.EventMessageStart{ID: "stream-1"},
		ir.EventReasoningDelta{Text: "think"},
		ir.EventTextDelta{Text: "Hello"},
		ir.EventToolCallStart{Index: 0, ID: "tc1", Name: "weather"},
		ir.EventToolArgumentsDelta{Index: 0, Arguments: `{"ci`},
		ir.EventToolCallStop{Index: 0},
		ir.EventToolArgumentsDelta{Index: 0, Arguments: `ty":"nyc"}`},
		ir.EventToolCallStop{Index: 0},
		ir.EventUsageUpdate{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
		ir.EventMessageStop{FinishReason: "stop"},
	}

	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events mismatch:\n got: %+v\n want: %+v", events, want)
	}
}

func TestStreamBridge_UpstreamError(t *testing.T) {
	ctx := context.Background()
	in := make(chan llmhub.StreamChunk, 2)

	in <- llmhub.StreamChunk{ID: "stream-err", Delta: "partial"}
	in <- llmhub.StreamChunk{Err: errors.New("boom")}
	close(in)

	out := StreamBridge(ctx, in)
	events := drainEvents(t, out)

	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}
	assertEvent(t, events[0], ir.EventMessageStart{ID: "stream-err"})
	assertEvent(t, events[1], ir.EventTextDelta{Text: "partial"})
	errEvent, ok := events[2].(ir.EventUpstreamError)
	if !ok {
		t.Fatalf("expected upstream error event, got %T", events[2])
	}
	if errEvent.Message != "boom" || errEvent.Kind != "upstream_error" {
		t.Fatalf("unexpected error event: %+v", errEvent)
	}
}

func TestStreamBridge_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	in := make(chan llmhub.StreamChunk)
	out := StreamBridge(ctx, in)

	cancel()

	events := drainEvents(t, out)
	if len(events) != 0 {
		t.Fatalf("expected no events after cancel, got %d", len(events))
	}
}

func drainEvents(t *testing.T, ch <-chan ir.Event) []ir.Event {
	t.Helper()
	var events []ir.Event
	for e := range ch {
		events = append(events, e)
	}
	return events
}

func assertEvent(t *testing.T, got, want ir.Event) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("event mismatch: got %+v, want %+v", got, want)
	}
}

func assertTextContent(t *testing.T, part llmhub.ContentPart, want string) {
	t.Helper()
	text, ok := part.(*llmhub.TextContent)
	if !ok || text.Text != want {
		t.Fatalf("expected text %q, got %+v", want, part)
	}
}

func assertImageContent(t *testing.T, part llmhub.ContentPart, url, detail string) {
	t.Helper()
	img, ok := part.(*llmhub.ImageContent)
	if !ok || img.URL != url || img.Detail != detail {
		t.Fatalf("expected image url=%q detail=%q, got %+v", url, detail, part)
	}
}

func assertReasoningContent(t *testing.T, part llmhub.ContentPart, want string) {
	t.Helper()
	r, ok := part.(*llmhub.ReasoningContent)
	if !ok || r.Text != want {
		t.Fatalf("expected reasoning %q, got %+v", want, part)
	}
}

func assertToolCallContent(t *testing.T, part llmhub.ContentPart, id, name, args string) {
	t.Helper()
	tc, ok := part.(*llmhub.ToolCallContent)
	if !ok || tc.ID != id || tc.Name != name || tc.Arguments != args {
		t.Fatalf("expected tool call id=%q name=%q args=%q, got %+v", id, name, args, part)
	}
}

func assertIRTextBlock(t *testing.T, blk ir.Block, want string) {
	t.Helper()
	b, ok := blk.(ir.TextBlock)
	if !ok || b.Text != want {
		t.Fatalf("expected text block %q, got %+v", want, blk)
	}
}

func assertIRReasoningBlock(t *testing.T, blk ir.Block, kind ir.ReasoningKind, want string) {
	t.Helper()
	b, ok := blk.(ir.ReasoningBlock)
	if !ok || b.ReasoningKind != kind || b.Text != want {
		t.Fatalf("expected reasoning block kind=%q text=%q, got %+v", kind, want, blk)
	}
}

func assertIRToolCallBlock(t *testing.T, blk ir.Block, id, name, args string) {
	t.Helper()
	b, ok := blk.(ir.ToolCallBlock)
	if !ok || b.ID != id || b.Name != name || b.Arguments != args {
		t.Fatalf("expected tool call block id=%q name=%q args=%q, got %+v", id, name, args, blk)
	}
}

// TestIRToHubPrompt_CacheControlPassthrough proves that Anthropic prompt
// caching breakpoints (ir.CacheControl blocks) survive the IR -> llmhub
// conversion instead of being silently dropped.
//
// Regression: IRToHubPrompt had no case for ir.CacheControl, so every
// cache_control marker on an inbound Anthropic request vanished before the
// hub adapter, and the upstream never saw a caching boundary.
func TestIRToHubPrompt_CacheControlPassthrough(t *testing.T) {
	msgs := []*ir.Message{
		{
			Role: "system",
			Meta: map[string]string{"lane": "anthropic"},
			Blocks: []ir.Block{
				ir.TextBlock{Text: "sys"},
				ir.CacheControl{Breakpoint: true},
			},
		},
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "hello"},
				ir.CacheControl{Breakpoint: true},
				ir.ImageBlock{URL: "http://example.com/img.png"},
				&ir.CacheControl{Breakpoint: true},
			},
		},
	}

	hubMsgs := IRToHubPrompt(msgs)
	if len(hubMsgs) != 2 {
		t.Fatalf("expected 2 hub messages, got %d", len(hubMsgs))
	}

	// System message: one text part, breakpoint on part 0, original meta kept.
	if len(hubMsgs[0].Content) != 1 {
		t.Fatalf("expected 1 system content part, got %d", len(hubMsgs[0].Content))
	}
	if hubMsgs[0].Meta["lane"] != "anthropic" {
		t.Errorf("existing message meta must survive, got %+v", hubMsgs[0].Meta)
	}
	sysBp := CacheControlBreakpoints(hubMsgs[0])
	if len(sysBp) != 1 || sysBp[0].Index != 0 || sysBp[0].CacheControl != "ephemeral" {
		t.Fatalf("system breakpoints = %+v, want [{0 ephemeral}]", sysBp)
	}

	// User message: text + image parts, breakpoints on parts 0 and 1.
	if len(hubMsgs[1].Content) != 2 {
		t.Fatalf("expected 2 user content parts (no cache_control part injected), got %d", len(hubMsgs[1].Content))
	}
	userBp := CacheControlBreakpoints(hubMsgs[1])
	if len(userBp) != 2 {
		t.Fatalf("user breakpoints = %+v, want 2 entries", userBp)
	}
	if userBp[0].Index != 0 || userBp[1].Index != 1 {
		t.Fatalf("user breakpoint indexes = [%d %d], want [0 1]", userBp[0].Index, userBp[1].Index)
	}
	for _, bp := range userBp {
		if bp.CacheControl != "ephemeral" {
			t.Errorf("breakpoint %d cache_control = %q, want ephemeral", bp.Index, bp.CacheControl)
		}
	}

	// The marker must be JSON so an upstream hub provider can splice real
	// cache_control fields onto the matching content parts.
	raw := hubMsgs[1].Meta["cache_control"]
	if !json.Valid([]byte(raw)) {
		t.Fatalf("cache_control meta is not valid JSON: %q", raw)
	}
}

// TestIRToHubPrompt_CacheControlWithoutPrecedingBlock: a marker with nothing to
// annotate must not panic and must not emit a negative index.
func TestIRToHubPrompt_CacheControlWithoutPrecedingBlock(t *testing.T) {
	hubMsgs := IRToHubPrompt([]*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.CacheControl{Breakpoint: true}, ir.TextBlock{Text: "hi"}}},
	})
	if len(hubMsgs) != 1 {
		t.Fatalf("expected 1 hub message, got %d", len(hubMsgs))
	}
	for _, bp := range CacheControlBreakpoints(hubMsgs[0]) {
		if bp.Index < 0 {
			t.Fatalf("negative breakpoint index: %+v", bp)
		}
	}
}

// TestAdapter_CacheControlSurvivesToHub: end-to-end from a decoded Anthropic
// request through the hublane adapter. The hub provider must still see the
// caching breakpoint (AC1).
func TestAdapter_CacheControlSurvivesToHub(t *testing.T) {
	body := `{
		"model": "claude-3-7-sonnet",
		"max_tokens": 32,
		"system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
			]}
		]
	}`

	decoded, err := codec.DecodeMessagesRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeMessagesRequest failed: %v", err)
	}

	hub := &fakeHubProvider{
		name:         "fake-hub",
		generateResp: &llmhub.Response{Content: []llmhub.ContentPart{llmhub.Text("ok")}},
	}
	adapter := New(hub)

	if _, err := adapter.Generate(context.Background(), decoded.Messages, decoded.Options...); err != nil {
		t.Fatalf("adapter.Generate failed: %v", err)
	}

	if len(hub.lastPrompt) != 2 {
		t.Fatalf("expected 2 hub messages, got %d", len(hub.lastPrompt))
	}
	if got := CacheControlBreakpoints(hub.lastPrompt[0]); len(got) != 1 || got[0].Index != 0 {
		t.Fatalf("system breakpoint lost on the way to the hub adapter: %+v", got)
	}
	if got := CacheControlBreakpoints(hub.lastPrompt[1]); len(got) != 1 || got[0].Index != 0 {
		t.Fatalf("user breakpoint lost on the way to the hub adapter: %+v", got)
	}
}

// TestIRToHubPrompt_CacheControlOnToolResult: a breakpoint after a tool_result
// block must annotate the standalone tool message the bridge emits, not the
// message the tool result was lifted out of.
func TestIRToHubPrompt_CacheControlOnToolResult(t *testing.T) {
	hubMsgs := IRToHubPrompt([]*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "answer?"},
				ir.ToolResultBlock{ToolCallID: "tc1", Name: "calc", Content: "42"},
				ir.CacheControl{Breakpoint: true},
			},
		},
	})

	if len(hubMsgs) != 2 {
		t.Fatalf("expected 2 hub messages (user + tool), got %d", len(hubMsgs))
	}
	tool := hubMsgs[0]
	if tool.Role != llmhub.RoleTool {
		t.Fatalf("expected the tool message first, got %+v", hubMsgs)
	}
	bp := CacheControlBreakpoints(tool)
	if len(bp) != 1 || bp[0].Index != 0 || bp[0].CacheControl != "ephemeral" {
		t.Fatalf("tool message breakpoints = %+v, want [{0 ephemeral}]", bp)
	}
	if got := CacheControlBreakpoints(hubMsgs[1]); len(got) != 0 {
		t.Fatalf("user message must not inherit the tool breakpoint: %+v", got)
	}
}
