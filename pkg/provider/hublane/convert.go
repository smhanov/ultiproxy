package hublane

import (
	"context"
	"encoding/json"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/ultiproxy/pkg/ir"
)

// CacheControlMetaKey is the llmhub message Meta key that carries Anthropic
// prompt-caching breakpoints across the IR -> llmhub bridge.
//
// llmhub has no cache_control content part, so the breakpoint cannot ride on
// the content itself. Instead IRToHubPrompt records, as JSON, every content
// part index that would carry a `cache_control` field on the Anthropic wire:
//
//	[{"index":0,"cache_control":"ephemeral"},{"index":2,"cache_control":"ephemeral"}]
//
// A hub-side Anthropic serializer can splice the field back onto those parts;
// everything else (openai-shaped lanes) simply ignores the extra meta entry.
const CacheControlMetaKey = "cache_control"

// CacheControlTypeEphemeral is the only cache_control type Anthropic defines
// today ("ephemeral").
const CacheControlTypeEphemeral = "ephemeral"

// CacheBreakpoint names one hub content part that carries a prompt-caching
// breakpoint.
type CacheBreakpoint struct {
	// Index is the position of the marked part inside the hub message content.
	Index int `json:"index"`
	// CacheControl is the Anthropic cache_control type, e.g. "ephemeral".
	CacheControl string `json:"cache_control"`
}

// CacheControlBreakpoints decodes the caching breakpoints carried by a hub
// message. It returns nil when the message carries none.
func CacheControlBreakpoints(msg *llmhub.Message) []CacheBreakpoint {
	if msg == nil {
		return nil
	}
	raw, ok := msg.Meta[CacheControlMetaKey]
	if !ok || raw == "" {
		return nil
	}
	var out []CacheBreakpoint
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil
	}
	return out
}

// setCacheControlMeta attaches the breakpoints collected while converting a
// message, preserving any meta the IR message already carried.
func setCacheControlMeta(msg *llmhub.Message, breakpoints []CacheBreakpoint) {
	if msg == nil || len(breakpoints) == 0 {
		return
	}
	raw, err := json.Marshal(breakpoints)
	if err != nil {
		return
	}
	meta := make(map[string]string, len(msg.Meta)+1)
	for k, v := range msg.Meta {
		meta[k] = v
	}
	meta[CacheControlMetaKey] = string(raw)
	msg.Meta = meta
}

// IRToHubPrompt converts normalized IR messages into llmhub messages.
//
// Supported block mappings:
//   - ir.TextBlock       -> llmhub.TextContent
//   - ir.ImageBlock      -> llmhub.ImageContent
//   - ir.ReasoningBlock  -> llmhub.ReasoningContent
//   - ir.ToolCallBlock   -> llmhub.ToolCallContent
//   - ir.ToolResultBlock -> role:"tool" message with tool_call_id metadata
//     (llmhub does not define a ToolResultContent part, so each result block
//     becomes its own tool message).
//   - ir.CacheControl    -> message Meta["cache_control"] JSON listing the
//     content part indexes that carry the caching breakpoint.
func IRToHubPrompt(msgs []*ir.Message) []*llmhub.Message {
	var out []*llmhub.Message
	for _, m := range msgs {
		if m == nil {
			continue
		}

		// cur accumulates the parts of the message under construction. Tool
		// result blocks instead become standalone tool messages, appended to
		// out as they are met (historical ordering: tool messages first, the
		// remaining parts of the source message last).
		cur := &llmhub.Message{Role: hubRole(m.Role), Meta: m.Meta}
		// target is the hub message a trailing ir.CacheControl marker
		// annotates: cur, or the tool message a tool_result block produced.
		target := cur

		for _, blk := range m.Blocks {
			if blk == nil {
				continue
			}
			// ir.CacheControl markers annotate the part emitted directly
			// before them, mirroring Anthropic's inline cache_control field.
			switch b := blk.(type) {
			case ir.CacheControl:
				if b.Breakpoint {
					markCacheBreakpoint(target)
				}
			case *ir.CacheControl:
				if b != nil && b.Breakpoint {
					markCacheBreakpoint(target)
				}
			case ir.TextBlock:
				cur.Append(llmhub.Text(b.Text))
				target = cur
			case *ir.TextBlock:
				cur.Append(llmhub.Text(b.Text))
				target = cur
			case ir.ImageBlock:
				cur.Append(&llmhub.ImageContent{URL: b.URL, Detail: b.Detail})
				target = cur
			case *ir.ImageBlock:
				cur.Append(&llmhub.ImageContent{URL: b.URL, Detail: b.Detail})
				target = cur
			case ir.ReasoningBlock:
				cur.Append(llmhub.Reasoning(b.Text))
				target = cur
			case *ir.ReasoningBlock:
				cur.Append(llmhub.Reasoning(b.Text))
				target = cur
			case ir.ToolCallBlock:
				cur.Append(&llmhub.ToolCallContent{
					ID:        b.ID,
					Name:      b.Name,
					Arguments: b.Arguments,
				})
				target = cur
			case *ir.ToolCallBlock:
				cur.Append(&llmhub.ToolCallContent{
					ID:        b.ID,
					Name:      b.Name,
					Arguments: b.Arguments,
				})
				target = cur
			case ir.ToolResultBlock:
				toolMsg := llmhub.NewToolResultMessage(
					b.ToolCallID,
					b.Name,
					llmhub.Text(b.Content),
				)
				out = append(out, toolMsg)
				target = toolMsg
			case *ir.ToolResultBlock:
				toolMsg := llmhub.NewToolResultMessage(
					b.ToolCallID,
					b.Name,
					llmhub.Text(b.Content),
				)
				out = append(out, toolMsg)
				target = toolMsg
			}
		}

		if len(cur.Content) > 0 {
			out = append(out, cur)
		}
	}
	return out
}

// markCacheBreakpoint records a prompt-caching breakpoint on the last content
// part of msg. It is a no-op when msg carries no content yet.
func markCacheBreakpoint(msg *llmhub.Message) {
	if msg == nil || len(msg.Content) == 0 {
		return
	}
	bp := append(CacheControlBreakpoints(msg), CacheBreakpoint{
		Index:        len(msg.Content) - 1,
		CacheControl: CacheControlTypeEphemeral,
	})
	setCacheControlMeta(msg, bp)
}

func hubRole(role string) llmhub.Role {
	switch role {
	case "system":
		return llmhub.RoleSystem
	case "user":
		return llmhub.RoleUser
	case "assistant":
		return llmhub.RoleAssistant
	case "tool":
		return llmhub.RoleTool
	default:
		return llmhub.Role(role)
	}
}

// HubResponseToIR converts a llmhub response into the normalized IR response.
// The model parameter is preserved for callers but is not used for content
// conversion (response messages are always assistant-role).
func HubResponseToIR(resp *llmhub.Response, model string) *ir.Response {
	if resp == nil {
		return nil
	}

	irResp := &ir.Response{
		ID:           resp.ID,
		UpstreamID:   resp.ID,
		FinishReason: finishReasonFromRaw(resp.Raw),
	}

	var blocks []ir.Block
	// Reasoning blocks come first (DeepSeek/Anthropic convention: the old
	// internal converters emitted reasoning before text). Text/tool blocks
	// follow in upstream order.
	for _, part := range resp.Content {
		if rc, ok := part.(*llmhub.ReasoningContent); ok && rc != nil {
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningText,
				Text:          rc.Text,
			})
		}
	}
	for _, part := range resp.Content {
		if part == nil {
			continue
		}
		switch p := part.(type) {
		case *llmhub.ReasoningContent:
			// Already emitted above.
		case *llmhub.TextContent:
			blocks = append(blocks, ir.TextBlock{Text: p.Text})
		case *llmhub.ToolCallContent:
			blocks = append(blocks, ir.ToolCallBlock{
				Index:     p.Index,
				ID:        p.ID,
				Name:      p.Name,
				Arguments: p.Arguments,
			})
		}
	}

	if len(blocks) > 0 {
		irResp.Message = &ir.Message{
			Role:   "assistant",
			Blocks: blocks,
		}
	}

	irResp.Usage = &ir.Usage{
		PromptTokens:             resp.Usage.PromptTokens,
		CompletionTokens:         resp.Usage.CompletionTokens,
		TotalTokens:              resp.Usage.TotalTokens,
		ReasoningTokens:          resp.Usage.ReasoningTokens,
		CacheCreationInputTokens: resp.Usage.CacheCreationTokens,
		CacheReadInputTokens:     resp.Usage.CacheReadTokens,
		Cost:                     resp.Usage.Cost,
	}

	return irResp
}

func finishReasonFromRaw(raw interface{}) string {
	if raw == nil {
		return ""
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return ""
	}
	var parsed struct {
		FinishReason string `json:"finish_reason"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return ""
	}
	return parsed.FinishReason
}

// StreamBridge reads llmhub streaming chunks and emits normalized IR events.
//
// Event order per chunk:
//  1. EventMessageStart on the first chunk with a non-empty ID.
//  2. EventTextDelta for chunk.Delta.
//  3. EventReasoningDelta for chunk.ReasoningDelta.
//  4. For each tool call in chunk.ToolCalls: start, arguments delta, stop.
//  5. EventUsageUpdate if chunk.Usage is present.
//  6. EventMessageStop if chunk.FinishReason is set or chunk.Done is true.
//  7. EventUpstreamError if chunk.Err is set.
//
// The returned channel is closed when the input channel closes or the context
// is cancelled.
func StreamBridge(ctx context.Context, hubChan <-chan llmhub.StreamChunk) <-chan ir.Event {
	out := make(chan ir.Event, 64)
	go func() {
		defer close(out)
		started := false
		for {
			select {
			case <-ctx.Done():
				return
			case chunk, ok := <-hubChan:
				if !ok {
					return
				}

				if !started && chunk.ID != "" {
					out <- ir.EventMessageStart{ID: chunk.ID}
					started = true
				}

				// Reasoning delta MUST precede text delta (DeepSeek requirement,
				// preserved from the previous internal/openai stream client).
				if chunk.ReasoningDelta != "" {
					out <- ir.EventReasoningDelta{Text: chunk.ReasoningDelta}
				}
				if chunk.Delta != "" {
					out <- ir.EventTextDelta{Text: chunk.Delta}
				}

				for _, tc := range chunk.ToolCalls {
					if tc == nil {
						continue
					}
					if tc.ID != "" || tc.Name != "" {
						out <- ir.EventToolCallStart{
							Index: tc.Index,
							ID:    tc.ID,
							Name:  tc.Name,
						}
					}
					if tc.Arguments != "" {
						out <- ir.EventToolArgumentsDelta{
							Index:     tc.Index,
							Arguments: tc.Arguments,
						}
					}
					out <- ir.EventToolCallStop{Index: tc.Index}
				}

				if chunk.Usage != nil {
					out <- ir.EventUsageUpdate{
						PromptTokens:             chunk.Usage.PromptTokens,
						CompletionTokens:         chunk.Usage.CompletionTokens,
						TotalTokens:              chunk.Usage.TotalTokens,
						CacheCreationInputTokens: chunk.Usage.CacheCreationTokens,
						CacheReadInputTokens:     chunk.Usage.CacheReadTokens,
						Cost:                     chunk.Usage.Cost,
					}
				}

				if chunk.FinishReason != "" || chunk.Done {
					out <- ir.EventMessageStop{FinishReason: chunk.FinishReason}
				}

				if chunk.Err != nil {
					out <- ir.EventUpstreamError{
						Kind:      "upstream_error",
						Message:   chunk.Err.Error(),
						Permanent: false,
					}
				}
			}
		}
	}()
	return out
}
