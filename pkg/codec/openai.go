package codec

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// OpenAIChatCompletionRequest represents the standard OpenAI chat completions request body.
type OpenAIChatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []OpenAIMessage `json:"messages"`
	Tools               []OpenAITool    `json:"tools,omitempty"`
	ToolChoice          any             `json:"tool_choice,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
	ReasoningEffort     string          `json:"reasoning_effort,omitempty"`
	ExtraFields         map[string]any  `json:"-"`
}

// StreamOptions contains stream configuration.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// OpenAIMessage represents a single message in an OpenAI chat completion request.
type OpenAIMessage struct {
	Role             string           `json:"role"`
	Content          any              `json:"content,omitempty"` // string or []OpenAIContentPart
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	Name             string           `json:"name,omitempty"`
}

// OpenAIContentPart represents a multipart content item in a message.
type OpenAIContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL any    `json:"image_url,omitempty"` // string or map
}

// OpenAITool describes a function tool definition.
type OpenAITool struct {
	Type     string            `json:"type"`
	Function OpenAIFunctionDef `json:"function"`
}

// OpenAIFunctionDef contains function schema.
type OpenAIFunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// OpenAIToolCall represents a tool call made by the assistant.
type OpenAIToolCall struct {
	Index    *int               `json:"index,omitempty"`
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function OpenAIFunctionCall `json:"function"`
}

// OpenAIFunctionCall contains the name and arguments of a tool call.
type OpenAIFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ChatCompletionDecoded contains the parsed messages, options, and metadata from an OpenAI request.
type ChatCompletionDecoded struct {
	Messages     []*ir.Message
	Options      []provider.Option
	Model        string
	Stream       bool
	IncludeUsage bool
}

// DecodeChatCompletionRequest translates an incoming OpenAI chat completions JSON payload into IR messages and provider options.
func DecodeChatCompletionRequest(body []byte) (*ChatCompletionDecoded, error) {
	var req OpenAIChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("decode chat completion request: %w", err)
	}

	// Also parse arbitrary extra fields for passthrough
	var rawMap map[string]any
	if err := json.Unmarshal(body, &rawMap); err == nil {
		req.ExtraFields = rawMap
	}

	var irMessages []*ir.Message
	for _, msg := range req.Messages {
		role := msg.Role
		if role == "developer" {
			role = "system"
		}

		var blocks []ir.Block

		// 1. Reasoning content if present
		if msg.ReasoningContent != "" {
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningText,
				Text:          msg.ReasoningContent,
			})
		}

		// 2. Main content (string or parts)
		if role != "tool" && msg.Content != nil {
			switch c := msg.Content.(type) {
			case string:
				if c != "" || (len(blocks) == 0 && len(msg.ToolCalls) == 0 && role != "tool") {
					blocks = append(blocks, ir.TextBlock{Text: c})
				}
			case []any:
				for _, partItem := range c {
					partMap, ok := partItem.(map[string]any)
					if !ok {
						continue
					}
					partType, _ := partMap["type"].(string)
					switch partType {
					case "text":
						txt, _ := partMap["text"].(string)
						blocks = append(blocks, ir.TextBlock{Text: txt})
					case "image_url":
						var url, detail string
						switch iu := partMap["image_url"].(type) {
						case string:
							url = iu
						case map[string]any:
							url, _ = iu["url"].(string)
							detail, _ = iu["detail"].(string)
						}
						blocks = append(blocks, ir.ImageBlock{
							URL:    url,
							Detail: detail,
						})
					}
				}
			}
		}

		// 3. Tool calls if assistant
		for i, tc := range msg.ToolCalls {
			idx := i
			if tc.Index != nil {
				idx = *tc.Index
			}
			blocks = append(blocks, ir.ToolCallBlock{
				Index:     idx,
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}

		// 4. Tool result if role == "tool"
		if role == "tool" {
			contentStr := ""
			switch c := msg.Content.(type) {
			case string:
				contentStr = c
			default:
				if c != nil {
					b, _ := json.Marshal(c)
					contentStr = string(b)
				}
			}
			blocks = append(blocks, ir.ToolResultBlock{
				ToolCallID: msg.ToolCallID,
				Name:       msg.Name,
				Content:    contentStr,
			})
		}

		irMessages = append(irMessages, &ir.Message{
			Role:   role,
			Blocks: blocks,
		})
	}

	var opts []provider.Option
	if req.Model != "" {
		opts = append(opts, provider.WithModel(req.Model))
	}
	if req.MaxCompletionTokens != nil && *req.MaxCompletionTokens > 0 {
		opts = append(opts, provider.WithMaxTokens(*req.MaxCompletionTokens))
	} else if req.MaxTokens != nil && *req.MaxTokens > 0 {
		opts = append(opts, provider.WithMaxTokens(*req.MaxTokens))
	}
	if req.Temperature != nil {
		opts = append(opts, provider.WithTemperature(*req.Temperature))
	}
	if req.ReasoningEffort != "" {
		opts = append(opts, provider.WithReasoningEffort(req.ReasoningEffort))
	}

	extra := make(map[string]any)
	if len(req.Tools) > 0 {
		extra["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		extra["tool_choice"] = req.ToolChoice
	}
	if req.ReasoningEffort != "" {
		extra["reasoning_effort"] = req.ReasoningEffort
	}
	if req.StreamOptions != nil {
		extra["stream_options"] = req.StreamOptions
	}
	// Copy other unrecognized body fields into extra
	for k, v := range req.ExtraFields {
		switch k {
		case "model", "messages", "tools", "tool_choice", "temperature",
			"max_tokens", "max_completion_tokens", "stream", "stream_options", "reasoning_effort":
			// already handled
		default:
			extra[k] = v
		}
	}
	if len(extra) > 0 {
		opts = append(opts, provider.WithExtraBody(extra))
	}

	includeUsage := false
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		includeUsage = true
	}

	return &ChatCompletionDecoded{
		Messages:     irMessages,
		Options:      opts,
		Model:        req.Model,
		Stream:       req.Stream,
		IncludeUsage: includeUsage,
	}, nil
}

// OpenAIChatCompletionResponse is the response format for non-streaming completions.
type OpenAIChatCompletionResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
}

// OpenAIChoice represents a completion choice.
type OpenAIChoice struct {
	Index        int                   `json:"index"`
	Message      OpenAIResponseMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

// OpenAIResponseMessage contains the generated message.
type OpenAIResponseMessage struct {
	Role             string           `json:"role"`
	Content          *string          `json:"content"`
	ReasoningContent *string          `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
}

// OpenAIUsage tracks token usage.
type OpenAIUsage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	CompletionTokensDetails *OpenAICompletionDetails `json:"completion_tokens_details,omitempty"`
}

// OpenAICompletionDetails holds detailed token counts.
type OpenAICompletionDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// EncodeChatCompletionResponse converts an IR Response to OpenAI chat completion JSON.
func EncodeChatCompletionResponse(resp *ir.Response, model string) ([]byte, error) {
	id := resp.ID
	if id == "" {
		id = generateID("chatcmpl-")
	}

	var textContent string
	var hasText bool
	var reasoningContent string
	var hasReasoning bool
	var toolCalls []OpenAIToolCall

	if resp.Message != nil {
		for _, b := range resp.Message.Blocks {
			switch block := b.(type) {
			case ir.TextBlock:
				textContent += block.Text
				hasText = true
			case *ir.TextBlock:
				textContent += block.Text
				hasText = true
			case ir.ReasoningBlock:
				reasoningContent += block.Text
				hasReasoning = true
			case *ir.ReasoningBlock:
				reasoningContent += block.Text
				hasReasoning = true
			case ir.ToolCallBlock:
				idx := block.Index
				toolCalls = append(toolCalls, OpenAIToolCall{
					Index: &idx,
					ID:    block.ID,
					Type:  "function",
					Function: OpenAIFunctionCall{
						Name:      block.Name,
						Arguments: block.Arguments,
					},
				})
			case *ir.ToolCallBlock:
				idx := block.Index
				toolCalls = append(toolCalls, OpenAIToolCall{
					Index: &idx,
					ID:    block.ID,
					Type:  "function",
					Function: OpenAIFunctionCall{
						Name:      block.Name,
						Arguments: block.Arguments,
					},
				})
			}
		}
	}

	finishReason := resp.FinishReason
	if finishReason == "" {
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
		} else {
			finishReason = "stop"
		}
	} else {
		switch finishReason {
		case "end_turn":
			finishReason = "stop"
		case "tool_use":
			finishReason = "tool_calls"
		case "max_tokens":
			finishReason = "length"
		}
	}

	msg := OpenAIResponseMessage{
		Role: "assistant",
	}
	if hasText || (len(toolCalls) == 0 && !hasReasoning) {
		msg.Content = &textContent
	}
	if hasReasoning {
		msg.ReasoningContent = &reasoningContent
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	var usage *OpenAIUsage
	if resp.Usage != nil {
		usage = &OpenAIUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
		if resp.Usage.ReasoningTokens > 0 {
			usage.CompletionTokensDetails = &OpenAICompletionDetails{
				ReasoningTokens: resp.Usage.ReasoningTokens,
			}
		}
	}

	out := OpenAIChatCompletionResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []OpenAIChoice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}

	return json.Marshal(out)
}

// OpenAIChatCompletionChunk represents a chunk in an SSE stream.
type OpenAIChatCompletionChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []OpenAIChunkChoice `json:"choices"`
	Usage   *OpenAIUsage        `json:"usage,omitempty"`
}

// OpenAIChunkChoice represents a choice in an OpenAI chunk.
type OpenAIChunkChoice struct {
	Index        int              `json:"index"`
	Delta        OpenAIChunkDelta `json:"delta"`
	FinishReason *string          `json:"finish_reason"`
}

// OpenAIChunkDelta contains the delta fields in a chunk.
type OpenAIChunkDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
}

// OpenAIStreamEncoder converts ir.Event instances into OpenAI SSE chunk lines written to an io.Writer.
type OpenAIStreamEncoder struct {
	w            io.Writer
	model        string
	id           string
	created      int64
	includeUsage bool
	started      bool
	lastUsage    *OpenAIUsage
}

// NewOpenAIStreamEncoder creates an OpenAI SSE stream encoder.
func NewOpenAIStreamEncoder(w io.Writer, model string, includeUsage bool) *OpenAIStreamEncoder {
	return &OpenAIStreamEncoder{
		w:            w,
		model:        model,
		id:           generateID("chatcmpl-"),
		created:      time.Now().Unix(),
		includeUsage: includeUsage,
	}
}

// EncodeEvent encodes an IR event into OpenAI chat.completion.chunk SSE.
func (e *OpenAIStreamEncoder) EncodeEvent(evt ir.Event) error {
	switch ev := evt.(type) {
	case ir.EventMessageStart:
		if ev.ID != "" {
			e.id = ev.ID
		}
		// Emit initial role chunk if not started
		if !e.started {
			e.started = true
			return e.writeChunk(OpenAIChunkDelta{Role: "assistant", Content: ""}, nil)
		}

	case ir.EventTextDelta:
		if !e.started {
			e.started = true
			if err := e.writeChunk(OpenAIChunkDelta{Role: "assistant"}, nil); err != nil {
				return err
			}
		}
		// Preserve whitespace exactly (NO TrimSpace)
		return e.writeChunk(OpenAIChunkDelta{Content: ev.Text}, nil)

	case ir.EventReasoningDelta:
		if !e.started {
			e.started = true
			if err := e.writeChunk(OpenAIChunkDelta{Role: "assistant"}, nil); err != nil {
				return err
			}
		}
		// Preserve whitespace exactly (NO TrimSpace)
		return e.writeChunk(OpenAIChunkDelta{ReasoningContent: ev.Text}, nil)

	case ir.EventToolCallStart:
		if !e.started {
			e.started = true
			if err := e.writeChunk(OpenAIChunkDelta{Role: "assistant"}, nil); err != nil {
				return err
			}
		}
		idx := ev.Index
		delta := OpenAIChunkDelta{
			ToolCalls: []OpenAIToolCall{
				{
					Index: &idx,
					ID:    ev.ID,
					Type:  "function",
					Function: OpenAIFunctionCall{
						Name:      ev.Name,
						Arguments: "",
					},
				},
			},
		}
		return e.writeChunk(delta, nil)

	case ir.EventToolArgumentsDelta:
		idx := ev.Index
		delta := OpenAIChunkDelta{
			ToolCalls: []OpenAIToolCall{
				{
					Index: &idx,
					Function: OpenAIFunctionCall{
						Arguments: ev.Arguments, // Exact whitespace preserved
					},
				},
			},
		}
		return e.writeChunk(delta, nil)

	case ir.EventToolCallStop:
		// No-op in OpenAI SSE format

	case ir.EventUsageUpdate:
		u := &OpenAIUsage{
			PromptTokens:     ev.PromptTokens,
			CompletionTokens: ev.CompletionTokens,
			TotalTokens:      ev.TotalTokens,
		}
		e.lastUsage = u

	case ir.EventMessageStop:
		finish := ev.FinishReason
		if finish == "" {
			finish = "stop"
		} else {
			switch finish {
			case "end_turn":
				finish = "stop"
			case "tool_use":
				finish = "tool_calls"
			case "max_tokens":
				finish = "length"
			}
		}
		return e.writeChunk(OpenAIChunkDelta{}, &finish)

	case ir.EventUpstreamError:
		errJSON, _ := json.Marshal(map[string]any{
			"error": map[string]any{
				"message": ev.Message,
				"type":    ev.Kind,
				"code":    ev.Kind,
			},
		})
		_, err := fmt.Fprintf(e.w, "data: %s\n\n", errJSON)
		return err
	}

	return nil
}

// Close finalizes the stream, writing usage if requested and the final [DONE] line.
func (e *OpenAIStreamEncoder) Close() error {
	if e.includeUsage && e.lastUsage != nil {
		usageChunk := struct {
			ID      string              `json:"id"`
			Object  string              `json:"object"`
			Created int64               `json:"created"`
			Model   string              `json:"model"`
			Choices []OpenAIChunkChoice `json:"choices"`
			Usage   *OpenAIUsage        `json:"usage"`
		}{
			ID:      e.id,
			Object:  "chat.completion.chunk",
			Created: e.created,
			Model:   e.model,
			Choices: []OpenAIChunkChoice{},
			Usage:   e.lastUsage,
		}
		data, err := json.Marshal(usageChunk)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(e.w, "data: %s\n\n", data); err != nil {
			return err
		}
	}

	_, err := fmt.Fprint(e.w, "data: [DONE]\n\n")
	return err
}

func (e *OpenAIStreamEncoder) writeChunk(delta OpenAIChunkDelta, finishReason *string) error {
	chunk := struct {
		ID      string              `json:"id"`
		Object  string              `json:"object"`
		Created int64               `json:"created"`
		Model   string              `json:"model"`
		Choices []OpenAIChunkChoice `json:"choices"`
	}{
		ID:      e.id,
		Object:  "chat.completion.chunk",
		Created: e.created,
		Model:   e.model,
		Choices: []OpenAIChunkChoice{
			{
				Index:        0,
				Delta:        delta,
				FinishReason: finishReason,
			},
		},
	}

	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(e.w, "data: %s\n\n", data)
	return err
}

func generateID(prefix string) string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return prefix + hex.EncodeToString(b)
}
