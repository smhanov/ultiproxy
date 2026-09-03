package anthropic

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"strings"

	"github.com/smhanov/ultiproxy/pkg/ir"
)

// StreamHandler reads Anthropic SSE events and yields normalized ir.Events.
func StreamHandler(ctx context.Context, body io.ReadCloser) <-chan ir.Event {
	ch := make(chan ir.Event, 64)

	go func() {
		defer close(ch)
		defer body.Close()

		scanner := bufio.NewScanner(body)
		buf := make([]byte, 64*1024)
		scanner.Buffer(buf, 1024*1024)

		var (
			currentEvent  string
			currentData   strings.Builder
			upstreamID    string
			stopReason    string
			totalInput    int
			totalOutput   int
			cacheCreation int
			cacheRead     int
			toolIndices   = make(map[int]bool)
		)

		dispatchEvent := func(evType string, dataBytes []byte) bool {
			var raw struct {
				Type    string `json:"type"`
				Index   int    `json:"index"`
				Message struct {
					ID    string `json:"id"`
					Usage struct {
						InputTokens              int `json:"input_tokens"`
						OutputTokens             int `json:"output_tokens"`
						CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
						CacheReadInputTokens     int `json:"cache_read_input_tokens"`
					} `json:"usage"`
				} `json:"message"`
				ContentBlock struct {
					Type string `json:"type"`
					ID   string `json:"id"`
					Name string `json:"name"`
				} `json:"content_block"`
				Delta struct {
					Type         string `json:"type"`
					Text         string `json:"text"`
					Thinking     string `json:"thinking"`
					Signature    string `json:"signature"`
					PartialJSON  string `json:"partial_json"`
					StopReason   string `json:"stop_reason"`
					StopSequence string `json:"stop_sequence"`
				} `json:"delta"`
				Usage struct {
					InputTokens              int `json:"input_tokens"`
					OutputTokens             int `json:"output_tokens"`
					CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
					CacheReadInputTokens     int `json:"cache_read_input_tokens"`
				} `json:"usage"`
				Error struct {
					Type    string `json:"type"`
					Message string `json:"message"`
				} `json:"error"`
			}

			if err := json.Unmarshal(dataBytes, &raw); err != nil {
				return true
			}

			eventType := evType
			if eventType == "" {
				eventType = raw.Type
			}

			switch eventType {
			case "message_start":
				upstreamID = raw.Message.ID
				totalInput = raw.Message.Usage.InputTokens
				cacheCreation = raw.Message.Usage.CacheCreationInputTokens
				cacheRead = raw.Message.Usage.CacheReadInputTokens

				select {
				case ch <- ir.EventMessageStart{ID: upstreamID}:
				case <-ctx.Done():
					return false
				}

				select {
				case ch <- ir.EventUsageUpdate{
					PromptTokens:             totalInput,
					CompletionTokens:         totalOutput,
					TotalTokens:              totalInput + totalOutput,
					CacheCreationInputTokens: cacheCreation,
					CacheReadInputTokens:     cacheRead,
				}:
				case <-ctx.Done():
					return false
				}

			case "content_block_start":
				switch raw.ContentBlock.Type {
				case "thinking", "redacted_thinking":
					select {
					case ch <- ir.EventBlockStart{Index: raw.Index, Kind: "reasoning"}:
					case <-ctx.Done():
						return false
					}
				case "text":
					select {
					case ch <- ir.EventBlockStart{Index: raw.Index, Kind: "text"}:
					case <-ctx.Done():
						return false
					}
				case "tool_use":
					toolIndices[raw.Index] = true
					select {
					case ch <- ir.EventToolCallStart{
						Index: raw.Index,
						ID:    raw.ContentBlock.ID,
						Name:  raw.ContentBlock.Name,
					}:
					case <-ctx.Done():
						return false
					}
				}

			case "content_block_delta":
				switch raw.Delta.Type {
				case "thinking_delta":
					select {
					case ch <- ir.EventReasoningDelta{Index: raw.Index, Text: raw.Delta.Thinking}:
					case <-ctx.Done():
						return false
					}
				case "signature_delta":
					select {
					case ch <- ir.EventReasoningSignature{Index: raw.Index, Signature: raw.Delta.Signature}:
					case <-ctx.Done():
						return false
					}
				case "text_delta":
					select {
					case ch <- ir.EventTextDelta{Index: raw.Index, Text: raw.Delta.Text}:
					case <-ctx.Done():
						return false
					}
				case "input_json_delta":
					select {
					case ch <- ir.EventToolArgumentsDelta{Index: raw.Index, Arguments: raw.Delta.PartialJSON}:
					case <-ctx.Done():
						return false
					}
				}

			case "content_block_stop":
				if toolIndices[raw.Index] {
					select {
					case ch <- ir.EventToolCallStop{Index: raw.Index}:
					case <-ctx.Done():
						return false
					}
					delete(toolIndices, raw.Index)
				}

			case "message_delta":
				if raw.Delta.StopReason != "" {
					stopReason = raw.Delta.StopReason
				}
				totalOutput += raw.Usage.OutputTokens
				if raw.Usage.InputTokens > 0 {
					totalInput += raw.Usage.InputTokens
				}
				if raw.Usage.CacheCreationInputTokens > 0 {
					cacheCreation += raw.Usage.CacheCreationInputTokens
				}
				if raw.Usage.CacheReadInputTokens > 0 {
					cacheRead += raw.Usage.CacheReadInputTokens
				}

				select {
				case ch <- ir.EventUsageUpdate{
					PromptTokens:             totalInput,
					CompletionTokens:         totalOutput,
					TotalTokens:              totalInput + totalOutput,
					CacheCreationInputTokens: cacheCreation,
					CacheReadInputTokens:     cacheRead,
				}:
				case <-ctx.Done():
					return false
				}

			case "message_stop":
				select {
				case ch <- ir.EventMessageStop{FinishReason: stopReason, UpstreamID: upstreamID}:
				case <-ctx.Done():
					return false
				}

			case "error":
				select {
				case ch <- ir.EventUpstreamError{
					Kind:      raw.Error.Type,
					Message:   raw.Error.Message,
					Permanent: true,
				}:
				case <-ctx.Done():
					return false
				}
			}

			return true
		}

		for scanner.Scan() {
			select {
			case <-ctx.Done():
				return
			default:
			}

			line := scanner.Text()

			if line == "" {
				if currentData.Len() > 0 || currentEvent != "" {
					dataBytes := []byte(currentData.String())
					if !dispatchEvent(currentEvent, dataBytes) {
						return
					}
					currentEvent = ""
					currentData.Reset()
				}
				continue
			}

			if strings.HasPrefix(line, ":") {
				continue
			}

			if strings.HasPrefix(line, "event:") {
				val := strings.TrimPrefix(line, "event:")
				currentEvent = strings.TrimSpace(val)
				continue
			}

			if strings.HasPrefix(line, "data:") {
				val := strings.TrimPrefix(line, "data:")
				val = strings.TrimPrefix(val, " ")
				if currentData.Len() > 0 {
					currentData.WriteString("\n")
				}
				currentData.WriteString(val)
				continue
			}
		}

		if currentData.Len() > 0 || currentEvent != "" {
			dataBytes := []byte(currentData.String())
			_ = dispatchEvent(currentEvent, dataBytes)
		}
	}()

	return ch
}
