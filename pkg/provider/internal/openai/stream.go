package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/smhanov/ultiproxy/pkg/ir"
)

// StreamHandler reads an OpenAI-compatible SSE response and translates it into an IR event channel.
func StreamHandler(ctx context.Context, body io.ReadCloser) <-chan ir.Event {
	ch := make(chan ir.Event, 64)

	go func() {
		defer close(ch)
		defer body.Close()

		scanner := bufio.NewScanner(body)
		// Set scanner buffer up to 1MB for large tokens/payloads
		buf := make([]byte, 64*1024)
		scanner.Buffer(buf, 1024*1024)

		var (
			upstreamID      string
			finishReason    string
			activeTools     = make(map[int]bool)
			messageStopSent bool
		)

		for scanner.Scan() {
			select {
			case <-ctx.Done():
				return
			default:
			}

			line := scanner.Text()
			if !strings.HasPrefix(line, "data:") {
				continue
			}

			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "" {
				continue
			}
			if data == "[DONE]" {
				break
			}

			var chunk ChatCompletionChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				// Ignore unparseable non-data ping or partial lines
				continue
			}

			if chunk.ID != "" && upstreamID == "" {
				upstreamID = chunk.ID
				select {
				case ch <- ir.EventMessageStart{ID: upstreamID}:
				case <-ctx.Done():
					return
				}
			}

			// Report usage update if present in chunk
			if chunk.Usage != nil {
				select {
				case ch <- ir.EventUsageUpdate{
					PromptTokens:             chunk.Usage.PromptTokens,
					CompletionTokens:         chunk.Usage.CompletionTokens,
					TotalTokens:              chunk.Usage.TotalTokens,
					CacheCreationInputTokens: chunk.Usage.CacheCreationInputTokens,
					CacheReadInputTokens:     chunk.Usage.CacheReadInputTokens,
					Cost:                     chunk.Usage.TotalCost,
				}:
				case <-ctx.Done():
					return
				}
			}

			for _, choice := range chunk.Choices {
				// 1. Reasoning delta MUST come before text delta (DeepSeek requirement)
				if choice.Delta.ReasoningContent != "" {
					select {
					case ch <- ir.EventReasoningDelta{
						Index: choice.Index,
						Text:  choice.Delta.ReasoningContent,
					}:
					case <-ctx.Done():
						return
					}
				}

				// 2. Text delta
				if choice.Delta.Content != "" {
					select {
					case ch <- ir.EventTextDelta{
						Index: choice.Index,
						Text:  choice.Delta.Content,
					}:
					case <-ctx.Done():
						return
					}
				}

				// 3. Tool call deltas
				for _, tc := range choice.Delta.ToolCalls {
					toolIdx := tc.Index
					if tc.ID != "" || tc.Function.Name != "" {
						activeTools[toolIdx] = true
						select {
						case ch <- ir.EventToolCallStart{
							Index: toolIdx,
							ID:    tc.ID,
							Name:  tc.Function.Name,
						}:
						case <-ctx.Done():
							return
						}
					}
					if tc.Function.Arguments != "" {
						select {
						case ch <- ir.EventToolArgumentsDelta{
							Index:     toolIdx,
							Arguments: tc.Function.Arguments,
						}:
						case <-ctx.Done():
							return
						}
					}
				}

				if choice.FinishReason != nil && *choice.FinishReason != "" {
					finishReason = *choice.FinishReason
					for idx := range activeTools {
						select {
						case ch <- ir.EventToolCallStop{Index: idx}:
						case <-ctx.Done():
							return
						}
					}
					activeTools = make(map[int]bool)
				}
			}
		}

		if err := scanner.Err(); err != nil {
			select {
			case ch <- ir.EventUpstreamError{
				Kind:      "stream_read_error",
				Message:   err.Error(),
				Permanent: false,
			}:
			case <-ctx.Done():
			}
			return
		}

		if !messageStopSent {
			select {
			case ch <- ir.EventMessageStop{
				FinishReason: finishReason,
				UpstreamID:   upstreamID,
			}:
			case <-ctx.Done():
			}
		}
	}()

	return ch
}

// ExecuteStream performs the HTTP request and returns the streaming IR event channel.
// Stream MUST return an error synchronously when the upstream replies with a non-2xx status.
func ExecuteStream(ctx context.Context, httpClient *http.Client, req *http.Request) (<-chan ir.Event, error) {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	resp, err := httpClient.Do(req.WithContext(ctx))
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		errBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("upstream error (status %d): %s", resp.StatusCode, string(errBytes))
	}

	return StreamHandler(ctx, resp.Body), nil
}
