package router

import (
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// EstimateTokens calculates an approximate token count for messages and config.
// Heuristic: ~4 characters per token for prompt text, plus output tokens.
func EstimateTokens(msgs []*ir.Message, opts ...provider.Option) int64 {
	cfg := provider.NewRequestConfig(opts...)

	var charCount int64
	for _, m := range msgs {
		if m == nil {
			continue
		}
		charCount += int64(len(m.Role))
		for _, blk := range m.Blocks {
			if blk == nil {
				continue
			}
			switch b := blk.(type) {
			case ir.TextBlock:
				charCount += int64(len(b.Text))
			case *ir.TextBlock:
				charCount += int64(len(b.Text))
			case ir.ToolCallBlock:
				charCount += int64(len(b.ID) + len(b.Name) + len(b.Arguments))
			case *ir.ToolCallBlock:
				charCount += int64(len(b.ID) + len(b.Name) + len(b.Arguments))
			case ir.ToolResultBlock:
				charCount += int64(len(b.ToolCallID) + len(b.Name) + len(b.Content))
			case *ir.ToolResultBlock:
				charCount += int64(len(b.ToolCallID) + len(b.Name) + len(b.Content))
			case ir.ReasoningBlock:
				charCount += int64(len(b.Text) + len(b.Signature))
			case *ir.ReasoningBlock:
				charCount += int64(len(b.Text) + len(b.Signature))
			}
		}
	}

	promptTokens := (charCount + 3) / 4
	if promptTokens <= 0 {
		promptTokens = 1
	}

	outputTokens := int64(1024)
	if cfg.MaxTokens > 0 {
		outputTokens = int64(cfg.MaxTokens)
	}

	return promptTokens + outputTokens
}
