package codec

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// AnthropicMessagesRequest represents the Anthropic /v1/messages request structure.
type AnthropicMessagesRequest struct {
	Model       string          `json:"model"`
	Messages    []AnthropicMsg  `json:"messages"`
	System      any             `json:"system,omitempty"` // string or []AnthropicContentBlock
	MaxTokens   int             `json:"max_tokens"`
	Temperature *float64        `json:"temperature,omitempty"`
	Thinking    *ThinkingConfig `json:"thinking,omitempty"`
	Tools       []AnthropicTool `json:"tools,omitempty"`
	ToolChoice  any             `json:"tool_choice,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	ExtraFields map[string]any  `json:"-"`
}

// ThinkingConfig specifies Anthropic extended thinking parameters.
type ThinkingConfig struct {
	Type         string `json:"type"` // "enabled"
	BudgetTokens int    `json:"budget_tokens"`
}

// AnthropicTool defines a tool in Anthropic format.
type AnthropicTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema"`
}

// AnthropicMsg represents a single message in Anthropic request.
type AnthropicMsg struct {
	Role    string `json:"role"`    // "user" or "assistant"
	Content any    `json:"content"` // string or []AnthropicContentBlock
}

// AnthropicContentBlock represents a polymorphic content block in Anthropic messages.
type AnthropicContentBlock struct {
	Type         string                 `json:"type"` // text, image, tool_use, tool_result, thinking, redacted_thinking
	Text         string                 `json:"text,omitempty"`
	Source       *AnthropicImageSource  `json:"source,omitempty"`
	ID           string                 `json:"id,omitempty"`
	Name         string                 `json:"name,omitempty"`
	Input        any                    `json:"input,omitempty"`
	ToolUseID    string                 `json:"tool_use_id,omitempty"`
	Content      any                    `json:"content,omitempty"`
	Thinking     string                 `json:"thinking,omitempty"`
	Signature    string                 `json:"signature,omitempty"`
	Data         string                 `json:"data,omitempty"`
	CacheControl *AnthropicCacheControl `json:"cache_control,omitempty"`
}

// AnthropicImageSource describes image payload.
type AnthropicImageSource struct {
	Type      string `json:"type"` // "base64" or "url"
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// AnthropicCacheControl specifies caching breakpoint.
type AnthropicCacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// MessagesDecoded contains translated IR messages and options for Anthropic requests.
type MessagesDecoded struct {
	Messages       []*ir.Message
	Options        []provider.Option
	Model          string
	Stream         bool
	ToolsRequested bool
}

// DecodeMessagesRequest translates Anthropic /v1/messages JSON into IR messages and provider options.
func DecodeMessagesRequest(body []byte) (*MessagesDecoded, error) {
	var req AnthropicMessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("decode messages request: %w", err)
	}

	var rawMap map[string]any
	if err := json.Unmarshal(body, &rawMap); err == nil {
		req.ExtraFields = rawMap
	}

	var irMessages []*ir.Message

	// Handle System prompt (string or blocks)
	if req.System != nil {
		var sysBlocks []ir.Block
		switch sys := req.System.(type) {
		case string:
			if sys != "" {
				sysBlocks = append(sysBlocks, ir.TextBlock{Text: sys})
			}
		case []any:
			for _, item := range sys {
				m, ok := item.(map[string]any)
				if !ok {
					continue
				}
				blockType, _ := m["type"].(string)
				if blockType != "text" {
					// Only text blocks are valid system content; anything else
					// (including an unknown type) cannot carry a breakpoint we
					// could re-attach.
					continue
				}
				txt, _ := m["text"].(string)
				sysBlocks = append(sysBlocks, ir.TextBlock{Text: txt})
				if m["cache_control"] != nil {
					sysBlocks = append(sysBlocks, ir.CacheControl{Breakpoint: true})
				}
			}
		}
		if len(sysBlocks) > 0 {
			irMessages = append(irMessages, &ir.Message{
				Role:   "system",
				Blocks: sysBlocks,
			})
		}
	}

	// Handle conversation messages
	for _, msg := range req.Messages {
		var blocks []ir.Block

		switch c := msg.Content.(type) {
		case string:
			blocks = append(blocks, ir.TextBlock{Text: c})
		case []any:
			for _, bItem := range c {
				bMap, ok := bItem.(map[string]any)
				if !ok {
					continue
				}
				blockType, _ := bMap["type"].(string)
				recognized := true
				switch blockType {
				case "text":
					txt, _ := bMap["text"].(string)
					blocks = append(blocks, ir.TextBlock{Text: txt})

				case "image":
					var url string
					if src, ok := bMap["source"].(map[string]any); ok {
						srcType, _ := src["type"].(string)
						if srcType == "base64" {
							mediaType, _ := src["media_type"].(string)
							data, _ := src["data"].(string)
							url = fmt.Sprintf("data:%s;base64,%s", mediaType, data)
						} else if srcType == "url" {
							url, _ = src["url"].(string)
						}
					}
					blocks = append(blocks, ir.ImageBlock{URL: url})

				case "thinking":
					thinking, _ := bMap["thinking"].(string)
					sig, _ := bMap["signature"].(string)
					blocks = append(blocks, ir.ReasoningBlock{
						ReasoningKind: ir.ReasoningText,
						Text:          thinking,
						Signature:     sig,
					})

				case "redacted_thinking":
					data, _ := bMap["data"].(string)
					blocks = append(blocks, ir.ReasoningBlock{
						ReasoningKind: ir.ReasoningRedacted,
						Text:          data,
					})

				case "tool_use":
					id, _ := bMap["id"].(string)
					name, _ := bMap["name"].(string)
					var argStr string
					if inp, ok := bMap["input"]; ok && inp != nil {
						b, _ := json.Marshal(inp)
						argStr = string(b)
					} else {
						argStr = "{}"
					}
					blocks = append(blocks, ir.ToolCallBlock{
						ID:        id,
						Name:      name,
						Arguments: argStr,
					})

				case "tool_result":
					toolUseID, _ := bMap["tool_use_id"].(string)
					var contentStr string
					switch rc := bMap["content"].(type) {
					case string:
						contentStr = rc
					default:
						if rc != nil {
							b, _ := json.Marshal(rc)
							contentStr = string(b)
						}
					}
					blocks = append(blocks, ir.ToolResultBlock{
						ToolCallID: toolUseID,
						Content:    contentStr,
					})
				default:
					// Unknown block type: nothing was decoded, so there is no
					// block to attach a cache_control marker to.
					recognized = false
				}

				// Anthropic accepts cache_control on any content block. Keep
				// the breakpoint in the IR as a marker placed directly after
				// the block it annotates so the prompt-caching boundary
				// survives the IR round trip.
				if recognized && bMap["cache_control"] != nil {
					blocks = append(blocks, ir.CacheControl{Breakpoint: true})
				}
			}
		}

		irMessages = append(irMessages, &ir.Message{
			Role:   msg.Role,
			Blocks: blocks,
		})
	}

	var opts []provider.Option
	if req.Model != "" {
		opts = append(opts, provider.WithModel(req.Model))
	}
	if req.MaxTokens > 0 {
		opts = append(opts, provider.WithMaxTokens(req.MaxTokens))
	}
	if req.Temperature != nil {
		opts = append(opts, provider.WithTemperature(*req.Temperature))
	}

	extra := make(map[string]any)
	if req.Thinking != nil {
		extra["thinking"] = req.Thinking
		if req.Thinking.BudgetTokens > 0 {
			extra["budget_tokens"] = req.Thinking.BudgetTokens
		}
	}
	if len(req.Tools) > 0 {
		extra["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		extra["tool_choice"] = req.ToolChoice
	}
	for k, v := range req.ExtraFields {
		switch k {
		case "model", "messages", "system", "max_tokens", "temperature", "thinking", "tools", "tool_choice", "stream",
			// Anthropic-only: Claude Code sends these on every /v1/messages
			// request. Strict OpenAI-compatible lanes (z.ai) reject them as
			// unknown keys (400 code 1210). Drop them here so the Anthropic
			// inbound surface can still route to an OpenAI lane.
			"metadata", "output_config", "context_management":
			// handled or Anthropic-only
		default:
			extra[k] = v
		}
	}
	if len(extra) > 0 {
		opts = append(opts, provider.WithExtraBody(extra))
	}

	return &MessagesDecoded{
		Messages:       irMessages,
		Options:        opts,
		Model:          req.Model,
		Stream:         req.Stream,
		ToolsRequested: len(req.Tools) > 0,
	}, nil
}

// AnthropicMessagesResponse represents the non-streaming response body.
type AnthropicMessagesResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"` // "message"
	Role         string                  `json:"role"` // "assistant"
	Model        string                  `json:"model"`
	Content      []AnthropicContentBlock `json:"content"`
	StopReason   *string                 `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        AnthropicUsage          `json:"usage"`
}

// AnthropicUsage represents token usage counts.
type AnthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// EncodeMessagesResponse serializes an ir.Response into Anthropic JSON.
func EncodeMessagesResponse(resp *ir.Response, model string) ([]byte, error) {
	id := resp.ID
	if id == "" {
		id = generateID("msg_")
	}

	var content []AnthropicContentBlock
	var lastIsToolCall bool

	if resp.Message != nil {
		for _, b := range resp.Message.Blocks {
			switch blk := b.(type) {
			case ir.TextBlock:
				content = append(content, AnthropicContentBlock{
					Type: "text",
					Text: blk.Text,
				})
				lastIsToolCall = false
			case *ir.TextBlock:
				content = append(content, AnthropicContentBlock{
					Type: "text",
					Text: blk.Text,
				})
				lastIsToolCall = false
			case ir.ReasoningBlock:
				if blk.ReasoningKind == ir.ReasoningRedacted {
					content = append(content, AnthropicContentBlock{
						Type: "redacted_thinking",
						Data: blk.Text,
					})
				} else {
					content = append(content, AnthropicContentBlock{
						Type:      "thinking",
						Thinking:  blk.Text,
						Signature: blk.Signature,
					})
				}
				lastIsToolCall = false
			case *ir.ReasoningBlock:
				if blk.ReasoningKind == ir.ReasoningRedacted {
					content = append(content, AnthropicContentBlock{
						Type: "redacted_thinking",
						Data: blk.Text,
					})
				} else {
					content = append(content, AnthropicContentBlock{
						Type:      "thinking",
						Thinking:  blk.Text,
						Signature: blk.Signature,
					})
				}
				lastIsToolCall = false
			case ir.ToolCallBlock:
				var inp any
				if err := json.Unmarshal([]byte(blk.Arguments), &inp); err != nil {
					inp = map[string]any{}
				}
				content = append(content, AnthropicContentBlock{
					Type:  "tool_use",
					ID:    blk.ID,
					Name:  blk.Name,
					Input: inp,
				})
				lastIsToolCall = true
			case *ir.ToolCallBlock:
				var inp any
				if err := json.Unmarshal([]byte(blk.Arguments), &inp); err != nil {
					inp = map[string]any{}
				}
				content = append(content, AnthropicContentBlock{
					Type:  "tool_use",
					ID:    blk.ID,
					Name:  blk.Name,
					Input: inp,
				})
				lastIsToolCall = true
			}
		}
	}

	stopReasonStr := resp.FinishReason
	if stopReasonStr == "" {
		if lastIsToolCall {
			stopReasonStr = "tool_use"
		} else {
			stopReasonStr = "end_turn"
		}
	} else {
		switch stopReasonStr {
		case "stop":
			stopReasonStr = "end_turn"
		case "tool_calls":
			stopReasonStr = "tool_use"
		case "length":
			stopReasonStr = "max_tokens"
		}
	}

	var usage AnthropicUsage
	if resp.Usage != nil {
		usage = AnthropicUsage{
			InputTokens:              resp.Usage.PromptTokens,
			OutputTokens:             resp.Usage.CompletionTokens,
			CacheCreationInputTokens: resp.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:     resp.Usage.CacheReadInputTokens,
		}
	}

	out := AnthropicMessagesResponse{
		ID:           id,
		Type:         "message",
		Role:         "assistant",
		Model:        model,
		Content:      content,
		StopReason:   &stopReasonStr,
		StopSequence: nil,
		Usage:        usage,
	}

	return json.Marshal(out)
}

// Concrete SSE payload structs for strict Anthropic schema adherence

type sseMessageStart struct {
	Type    string          `json:"type"`
	Message sseInnerMessage `json:"message"`
}

type sseInnerMessage struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []any          `json:"content"`
	Model        string         `json:"model"`
	StopReason   *string        `json:"stop_reason"`
	StopSequence *string        `json:"stop_sequence"`
	Usage        AnthropicUsage `json:"usage"`
}

type sseBlockStart struct {
	Type         string `json:"type"`
	Index        int    `json:"index"`
	ContentBlock any    `json:"content_block"`
}

type sseBlockDelta struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta any    `json:"delta"`
}

type sseBlockStop struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type sseMessageDelta struct {
	Type  string               `json:"type"`
	Delta sseInnerMessageDelta `json:"delta"`
	Usage sseMessageDeltaUsage `json:"usage"`
}

type sseInnerMessageDelta struct {
	StopReason   string  `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
}

type sseMessageDeltaUsage struct {
	OutputTokens int `json:"output_tokens"`
}

type sseDeltaText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type sseDeltaThinking struct {
	Type     string `json:"type"`
	Thinking string `json:"thinking"`
}

type sseDeltaSignature struct {
	Type      string `json:"type"`
	Signature string `json:"signature"`
}

type sseDeltaInputJSON struct {
	Type        string `json:"type"`
	PartialJSON string `json:"partial_json"`
}

// AnthropicStreamEncoder implements the Anthropic SSE state machine for streaming IR events.
type AnthropicStreamEncoder struct {
	w                io.Writer
	model            string
	id               string
	started          bool
	stopped          bool
	activeBlockIndex int
	startedBlocks    map[int]bool
	usage            AnthropicUsage
}

// NewAnthropicStreamEncoder creates a new Anthropic SSE encoder.
func NewAnthropicStreamEncoder(w io.Writer, model string) *AnthropicStreamEncoder {
	return &AnthropicStreamEncoder{
		w:                w,
		model:            model,
		id:               generateID("msg_"),
		activeBlockIndex: -1,
		startedBlocks:    make(map[int]bool),
	}
}

func (e *AnthropicStreamEncoder) ensureMessageStart() error {
	if e.started {
		return nil
	}
	e.started = true

	payload := sseMessageStart{
		Type: "message_start",
		Message: sseInnerMessage{
			ID:           e.id,
			Type:         "message",
			Role:         "assistant",
			Content:      []any{},
			Model:        e.model,
			StopReason:   nil,
			StopSequence: nil,
			Usage:        e.usage,
		},
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(e.w, "event: message_start\ndata: %s\n\n", data)
	return err
}

func (e *AnthropicStreamEncoder) stopActiveBlockIfNeeded(newIndex int) error {
	if e.activeBlockIndex != -1 && e.activeBlockIndex != newIndex {
		payload := sseBlockStop{
			Type:  "content_block_stop",
			Index: e.activeBlockIndex,
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(e.w, "event: content_block_stop\ndata: %s\n\n", data); err != nil {
			return err
		}
		e.activeBlockIndex = -1
	}
	return nil
}

// EncodeEvent processes an IR streaming event according to Anthropic's SSE protocol.
func (e *AnthropicStreamEncoder) EncodeEvent(evt ir.Event) error {
	switch ev := evt.(type) {
	case ir.EventMessageStart:
		if ev.ID != "" {
			e.id = ev.ID
		}
		return e.ensureMessageStart()

	case ir.EventBlockStart:
		if err := e.ensureMessageStart(); err != nil {
			return err
		}
		if err := e.stopActiveBlockIfNeeded(ev.Index); err != nil {
			return err
		}
		if !e.startedBlocks[ev.Index] {
			e.startedBlocks[ev.Index] = true
			e.activeBlockIndex = ev.Index

			var contentBlock any
			switch ev.Kind {
			case "thinking":
				contentBlock = struct {
					Type     string `json:"type"`
					Thinking string `json:"thinking"`
				}{Type: "thinking", Thinking: ""}
			case "tool_use":
				contentBlock = struct {
					Type  string `json:"type"`
					ID    string `json:"id"`
					Name  string `json:"name"`
					Input any    `json:"input"`
				}{Type: "tool_use", ID: "", Name: "", Input: map[string]any{}}
			default:
				contentBlock = struct {
					Type string `json:"type"`
					Text string `json:"text"`
				}{Type: "text", Text: ""}
			}

			payload := sseBlockStart{
				Type:         "content_block_start",
				Index:        ev.Index,
				ContentBlock: contentBlock,
			}
			data, err := json.Marshal(payload)
			if err != nil {
				return err
			}
			_, err = fmt.Fprintf(e.w, "event: content_block_start\ndata: %s\n\n", data)
			return err
		}

	case ir.EventTextDelta:
		if err := e.ensureMessageStart(); err != nil {
			return err
		}
		if err := e.stopActiveBlockIfNeeded(ev.Index); err != nil {
			return err
		}
		if !e.startedBlocks[ev.Index] {
			e.startedBlocks[ev.Index] = true
			e.activeBlockIndex = ev.Index

			payload := sseBlockStart{
				Type:  "content_block_start",
				Index: ev.Index,
				ContentBlock: struct {
					Type string `json:"type"`
					Text string `json:"text"`
				}{Type: "text", Text: ""},
			}
			data, err := json.Marshal(payload)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintf(e.w, "event: content_block_start\ndata: %s\n\n", data); err != nil {
				return err
			}
		}

		// Emit text_delta with EXACT whitespace preserved (NO TrimSpace)
		payload := sseBlockDelta{
			Type:  "content_block_delta",
			Index: ev.Index,
			Delta: sseDeltaText{
				Type: "text_delta",
				Text: ev.Text,
			},
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(e.w, "event: content_block_delta\ndata: %s\n\n", data)
		return err

	case ir.EventReasoningDelta:
		if err := e.ensureMessageStart(); err != nil {
			return err
		}
		if err := e.stopActiveBlockIfNeeded(ev.Index); err != nil {
			return err
		}
		if !e.startedBlocks[ev.Index] {
			e.startedBlocks[ev.Index] = true
			e.activeBlockIndex = ev.Index

			payload := sseBlockStart{
				Type:  "content_block_start",
				Index: ev.Index,
				ContentBlock: struct {
					Type     string `json:"type"`
					Thinking string `json:"thinking"`
				}{Type: "thinking", Thinking: ""},
			}
			data, err := json.Marshal(payload)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintf(e.w, "event: content_block_start\ndata: %s\n\n", data); err != nil {
				return err
			}
		}

		// Emit thinking_delta with exact whitespace preserved
		payload := sseBlockDelta{
			Type:  "content_block_delta",
			Index: ev.Index,
			Delta: sseDeltaThinking{
				Type:     "thinking_delta",
				Thinking: ev.Text,
			},
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(e.w, "event: content_block_delta\ndata: %s\n\n", data)
		return err

	case ir.EventReasoningSignature:
		if err := e.ensureMessageStart(); err != nil {
			return err
		}
		if err := e.stopActiveBlockIfNeeded(ev.Index); err != nil {
			return err
		}
		if !e.startedBlocks[ev.Index] {
			e.startedBlocks[ev.Index] = true
			e.activeBlockIndex = ev.Index

			payload := sseBlockStart{
				Type:  "content_block_start",
				Index: ev.Index,
				ContentBlock: struct {
					Type     string `json:"type"`
					Thinking string `json:"thinking"`
				}{Type: "thinking", Thinking: ""},
			}
			data, err := json.Marshal(payload)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintf(e.w, "event: content_block_start\ndata: %s\n\n", data); err != nil {
				return err
			}
		}

		// Emit signature_delta preserving opaque signature
		payload := sseBlockDelta{
			Type:  "content_block_delta",
			Index: ev.Index,
			Delta: sseDeltaSignature{
				Type:      "signature_delta",
				Signature: ev.Signature,
			},
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(e.w, "event: content_block_delta\ndata: %s\n\n", data)
		return err

	case ir.EventToolCallStart:
		if err := e.ensureMessageStart(); err != nil {
			return err
		}
		if err := e.stopActiveBlockIfNeeded(ev.Index); err != nil {
			return err
		}
		e.startedBlocks[ev.Index] = true
		e.activeBlockIndex = ev.Index

		payload := sseBlockStart{
			Type:  "content_block_start",
			Index: ev.Index,
			ContentBlock: struct {
				Type  string `json:"type"`
				ID    string `json:"id"`
				Name  string `json:"name"`
				Input any    `json:"input"`
			}{
				Type:  "tool_use",
				ID:    ev.ID,
				Name:  ev.Name,
				Input: map[string]any{},
			},
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(e.w, "event: content_block_start\ndata: %s\n\n", data)
		return err

	case ir.EventToolArgumentsDelta:
		if err := e.ensureMessageStart(); err != nil {
			return err
		}
		if err := e.stopActiveBlockIfNeeded(ev.Index); err != nil {
			return err
		}
		if !e.startedBlocks[ev.Index] {
			e.startedBlocks[ev.Index] = true
			e.activeBlockIndex = ev.Index

			payload := sseBlockStart{
				Type:  "content_block_start",
				Index: ev.Index,
				ContentBlock: struct {
					Type  string `json:"type"`
					ID    string `json:"id"`
					Name  string `json:"name"`
					Input any    `json:"input"`
				}{
					Type:  "tool_use",
					ID:    "",
					Name:  "",
					Input: map[string]any{},
				},
			}
			data, err := json.Marshal(payload)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintf(e.w, "event: content_block_start\ndata: %s\n\n", data); err != nil {
				return err
			}
		}

		payload := sseBlockDelta{
			Type:  "content_block_delta",
			Index: ev.Index,
			Delta: sseDeltaInputJSON{
				Type:        "input_json_delta",
				PartialJSON: ev.Arguments, // Exact whitespace preserved
			},
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(e.w, "event: content_block_delta\ndata: %s\n\n", data)
		return err

	case ir.EventToolCallStop:
		if e.activeBlockIndex == ev.Index {
			payload := sseBlockStop{
				Type:  "content_block_stop",
				Index: ev.Index,
			}
			data, err := json.Marshal(payload)
			if err != nil {
				return err
			}
			e.activeBlockIndex = -1
			_, err = fmt.Fprintf(e.w, "event: content_block_stop\ndata: %s\n\n", data)
			return err
		}

	case ir.EventUsageUpdate:
		e.usage.InputTokens = ev.PromptTokens
		e.usage.OutputTokens = ev.CompletionTokens
		e.usage.CacheCreationInputTokens = ev.CacheCreationInputTokens
		e.usage.CacheReadInputTokens = ev.CacheReadInputTokens

	case ir.EventMessageStop:
		if e.stopped {
			return nil
		}
		if err := e.ensureMessageStart(); err != nil {
			return err
		}
		if e.activeBlockIndex != -1 {
			payload := sseBlockStop{
				Type:  "content_block_stop",
				Index: e.activeBlockIndex,
			}
			data, err := json.Marshal(payload)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintf(e.w, "event: content_block_stop\ndata: %s\n\n", data); err != nil {
				return err
			}
			e.activeBlockIndex = -1
		}

		stopReason := ev.FinishReason
		if stopReason == "" {
			stopReason = "end_turn"
		} else {
			switch stopReason {
			case "stop":
				stopReason = "end_turn"
			case "tool_calls":
				stopReason = "tool_use"
			case "length":
				stopReason = "max_tokens"
			}
		}

		payload := sseMessageDelta{
			Type: "message_delta",
			Delta: sseInnerMessageDelta{
				StopReason:   stopReason,
				StopSequence: nil,
			},
			Usage: sseMessageDeltaUsage{
				OutputTokens: e.usage.OutputTokens,
			},
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(e.w, "event: message_delta\ndata: %s\n\n", data); err != nil {
			return err
		}

		e.stopped = true
		_, err = fmt.Fprint(e.w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
		return err

	case ir.EventUpstreamError:
		payload := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    ev.Kind,
				"message": ev.Message,
			},
		}
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(e.w, "event: error\ndata: %s\n\n", data)
		return err
	}

	return nil
}

// Close finalizes the stream if not already stopped.
func (e *AnthropicStreamEncoder) Close() error {
	if e.stopped {
		return nil
	}
	if e.activeBlockIndex != -1 {
		payload := sseBlockStop{
			Type:  "content_block_stop",
			Index: e.activeBlockIndex,
		}
		data, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(e.w, "event: content_block_stop\ndata: %s\n\n", data)
		e.activeBlockIndex = -1
	}
	if e.started {
		payload := sseMessageDelta{
			Type: "message_delta",
			Delta: sseInnerMessageDelta{
				StopReason:   "end_turn",
				StopSequence: nil,
			},
			Usage: sseMessageDeltaUsage{
				OutputTokens: e.usage.OutputTokens,
			},
		}
		data, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(e.w, "event: message_delta\ndata: %s\n\n", data)
		_, _ = fmt.Fprint(e.w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
		e.stopped = true
	}
	return nil
}
