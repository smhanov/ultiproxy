package hublane

import (
	"context"
	"encoding/json"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/ultiproxy/pkg/ir"
)

// IRToHubPrompt converts normalized IR messages into llmhub messages.
//
// Supported block mappings:
//   - ir.TextBlock       -> llmhub.TextContent
//   - ir.ReasoningBlock  -> llmhub.ReasoningContent
//   - ir.ToolCallBlock   -> llmhub.ToolCallContent
//   - ir.ToolResultBlock -> role:"tool" message with tool_call_id metadata
//     (llmhub does not define a ToolResultContent part, so each result block
//     becomes its own tool message).
func IRToHubPrompt(msgs []*ir.Message) []*llmhub.Message {
	var out []*llmhub.Message
	for _, m := range msgs {
		if m == nil {
			continue
		}

		var parts []llmhub.ContentPart
		for _, blk := range m.Blocks {
			if blk == nil {
				continue
			}
			switch b := blk.(type) {
			case ir.TextBlock:
				parts = append(parts, llmhub.Text(b.Text))
			case *ir.TextBlock:
				parts = append(parts, llmhub.Text(b.Text))
			case ir.ImageBlock:
				parts = append(parts, &llmhub.ImageContent{URL: b.URL, Detail: b.Detail})
			case *ir.ImageBlock:
				parts = append(parts, &llmhub.ImageContent{URL: b.URL, Detail: b.Detail})
			case ir.ReasoningBlock:
				parts = append(parts, llmhub.Reasoning(b.Text))
			case *ir.ReasoningBlock:
				parts = append(parts, llmhub.Reasoning(b.Text))
			case ir.ToolCallBlock:
				parts = append(parts, &llmhub.ToolCallContent{
					ID:        b.ID,
					Name:      b.Name,
					Arguments: b.Arguments,
				})
			case *ir.ToolCallBlock:
				parts = append(parts, &llmhub.ToolCallContent{
					ID:        b.ID,
					Name:      b.Name,
					Arguments: b.Arguments,
				})
			case ir.ToolResultBlock:
				out = append(out, llmhub.NewToolResultMessage(
					b.ToolCallID,
					b.Name,
					llmhub.Text(b.Content),
				))
			case *ir.ToolResultBlock:
				out = append(out, llmhub.NewToolResultMessage(
					b.ToolCallID,
					b.Name,
					llmhub.Text(b.Content),
				))
			}
		}

		if len(parts) > 0 {
			msg := llmhub.NewMessage(hubRole(m.Role), parts...)
			msg.Meta = m.Meta
			out = append(out, msg)
		}
	}
	return out
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
	for _, part := range resp.Content {
		if part == nil {
			continue
		}
		switch p := part.(type) {
		case *llmhub.TextContent:
			blocks = append(blocks, ir.TextBlock{Text: p.Text})
		case *llmhub.ReasoningContent:
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningText,
				Text:          p.Text,
			})
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
//   1. EventMessageStart on the first chunk with a non-empty ID.
//   2. EventTextDelta for chunk.Delta.
//   3. EventReasoningDelta for chunk.ReasoningDelta.
//   4. For each tool call in chunk.ToolCalls: start, arguments delta, stop.
//   5. EventUsageUpdate if chunk.Usage is present.
//   6. EventMessageStop if chunk.FinishReason is set or chunk.Done is true.
//   7. EventUpstreamError if chunk.Err is set.
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

				if chunk.Delta != "" {
					out <- ir.EventTextDelta{Text: chunk.Delta}
				}
				if chunk.ReasoningDelta != "" {
					out <- ir.EventReasoningDelta{Text: chunk.ReasoningDelta}
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
