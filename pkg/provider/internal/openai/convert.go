package openai

import (
	"strings"

	"github.com/smhanov/ultiproxy/pkg/ir"
)

// ConvertOptions configures IR to OpenAI conversion.
type ConvertOptions struct {
	AllowVision   bool
	EchoReasoning bool
}

// ConvertMessages converts protocol-neutral IR messages to OpenAI ChatMessages.
func ConvertMessages(msgs []*ir.Message, opts ConvertOptions) []ChatMessage {
	var out []ChatMessage

	for _, msg := range msgs {
		if msg == nil {
			continue
		}

		// First, handle any ToolResultBlocks as separate "tool" messages if present
		var toolResults []*ir.ToolResultBlock
		var textBlocks []string
		var imageBlocks []*ir.ImageBlock
		var toolCalls []ChatToolCall
		var reasoningText strings.Builder

		for _, blk := range msg.Blocks {
			switch b := blk.(type) {
			case ir.TextBlock:
				textBlocks = append(textBlocks, b.Text)
			case *ir.TextBlock:
				textBlocks = append(textBlocks, b.Text)
			case ir.ImageBlock:
				imageBlocks = append(imageBlocks, &b)
			case *ir.ImageBlock:
				imageBlocks = append(imageBlocks, b)
			case ir.ToolCallBlock:
				toolCalls = append(toolCalls, ChatToolCall{
					Index: b.Index,
					ID:    b.ID,
					Type:  "function",
					Function: ChatFunctionCall{
						Name:      b.Name,
						Arguments: b.Arguments,
					},
				})
			case *ir.ToolCallBlock:
				toolCalls = append(toolCalls, ChatToolCall{
					Index: b.Index,
					ID:    b.ID,
					Type:  "function",
					Function: ChatFunctionCall{
						Name:      b.Name,
						Arguments: b.Arguments,
					},
				})
			case ir.ToolResultBlock:
				toolResults = append(toolResults, &b)
			case *ir.ToolResultBlock:
				toolResults = append(toolResults, b)
			case ir.ReasoningBlock:
				reasoningText.WriteString(b.Text)
			case *ir.ReasoningBlock:
				reasoningText.WriteString(b.Text)
			}
		}

		// Emit tool results as individual tool role messages
		for _, tr := range toolResults {
			out = append(out, ChatMessage{
				Role:       "tool",
				ToolCallID: tr.ToolCallID,
				Content:    tr.Content,
				Name:       tr.Name,
			})
		}

		// If this was purely a tool result message and had no other content/tool calls, continue
		if len(toolResults) > 0 && len(textBlocks) == 0 && len(imageBlocks) == 0 && len(toolCalls) == 0 && reasoningText.Len() == 0 {
			continue
		}

		role := msg.Role
		if role == "" {
			role = "user"
		}

		chatMsg := ChatMessage{Role: role}

		if len(toolCalls) > 0 {
			chatMsg.ToolCalls = toolCalls
		}

		if opts.EchoReasoning && reasoningText.Len() > 0 && role == "assistant" {
			chatMsg.ReasoningContent = reasoningText.String()
		}

		// Multi-modal content handling
		if opts.AllowVision && len(imageBlocks) > 0 {
			var parts []ContentPart
			for _, txt := range textBlocks {
				if txt != "" {
					parts = append(parts, ContentPart{
						Type: "text",
						Text: txt,
					})
				}
			}
			for _, img := range imageBlocks {
				parts = append(parts, ContentPart{
					Type: "image_url",
					ImageURL: &ImageURLPart{
						URL:    img.URL,
						Detail: img.Detail,
					},
				})
			}
			chatMsg.Content = parts
		} else {
			// Plain text content
			chatMsg.Content = strings.Join(textBlocks, "\n")
		}

		out = append(out, chatMsg)
	}

	return out
}

// ConvertResponse converts an OpenAI ChatCompletionResponse into an ir.Response.
func ConvertResponse(resp *ChatCompletionResponse) *ir.Response {
	if resp == nil {
		return &ir.Response{}
	}

	irResp := &ir.Response{
		ID:         resp.ID,
		UpstreamID: resp.ID,
	}

	if resp.Usage != nil {
		irResp.Usage = &ir.Usage{
			PromptTokens:             resp.Usage.PromptTokens,
			CompletionTokens:         resp.Usage.CompletionTokens,
			TotalTokens:              resp.Usage.TotalTokens,
			ReasoningTokens:          resp.Usage.ReasoningTokens,
			CacheCreationInputTokens: resp.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:     resp.Usage.CacheReadInputTokens,
			Cost:                     resp.Usage.TotalCost,
		}
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		irResp.FinishReason = choice.FinishReason

		var blocks []ir.Block

		// 1. Reasoning block (DeepSeek / thinking models)
		if choice.Message.ReasoningContent != "" {
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningText,
				Text:          choice.Message.ReasoningContent,
			})
		}

		// 2. Text block
		switch content := choice.Message.Content.(type) {
		case string:
			if content != "" {
				blocks = append(blocks, ir.TextBlock{Text: content})
			}
		}

		// 3. Tool calls
		for _, tc := range choice.Message.ToolCalls {
			blocks = append(blocks, ir.ToolCallBlock{
				Index:     tc.Index,
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}

		irResp.Message = &ir.Message{
			Role:   choice.Message.Role,
			Blocks: blocks,
		}
	}

	return irResp
}
