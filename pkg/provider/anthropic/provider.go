package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	spikesanthropic "github.com/smhanov/ultiproxy/pkg/spikes/anthropic"
)

const (
	ProviderName     = "anthropic"
	DefaultBaseURL   = "https://api.anthropic.com/v1"
	AnthropicVersion = "2023-06-01"
	DefaultModel     = "claude-3-7-sonnet-20250219"
	DefaultMaxTokens = 4096
)

// Config configures the Anthropic provider.
type Config struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

// Provider implements provider.InferenceProvider.
type Provider struct {
	cfg        Config
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

// New creates a new Anthropic Messages provider.
func New(cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")

	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("ANTHROPIC_API_KEY")
	}

	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}

	return &Provider{
		cfg:        cfg,
		httpClient: cfg.HTTPClient,
		baseURL:    cfg.BaseURL,
		apiKey:     cfg.APIKey,
	}, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return ProviderName
}

// Capabilities returns Anthropic capabilities.
func Capabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:          true,
		Messages:      true,
		Reasoning:     true,
		Vision:        true,
		Tools:         true,
		Streaming:     true,
		PromptCaching: true,
	}
}

// Provider returns the provider.Provider bundle.
func (p *Provider) Provider() provider.Provider {
	return provider.Provider{
		Inference:    p,
		Capabilities: Capabilities(),
	}
}

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	reqBody, err := p.buildPayload(msgs, reqConfig, false)
	if err != nil {
		return nil, err
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	p.setHeaders(req, reqConfig)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("anthropic error (status %d): %s", resp.StatusCode, string(respBytes))
	}

	return p.parseResponse(respBytes)
}

// Stream implements provider.InferenceProvider.
// Synchronous error on non-2xx status code.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	reqBody, err := p.buildPayload(msgs, reqConfig, true)
	if err != nil {
		return nil, err
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	p.setHeaders(req, reqConfig)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		errBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("anthropic error (status %d): %s", resp.StatusCode, string(errBytes))
	}

	return StreamHandler(ctx, resp.Body), nil
}

func (p *Provider) setHeaders(req *http.Request, cfg *provider.RequestConfig) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", AnthropicVersion)
	if p.apiKey != "" {
		req.Header.Set("x-api-key", p.apiKey)
	}
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}
}

func (p *Provider) buildPayload(msgs []*ir.Message, cfg *provider.RequestConfig, stream bool) (map[string]any, error) {
	model := cfg.Model
	if model == "" {
		model = DefaultModel
	}

	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}

	payload := map[string]any{
		"model":      model,
		"max_tokens": maxTokens,
		"stream":     stream,
	}

	if cfg.Temperature != nil {
		payload["temperature"] = *cfg.Temperature
	}

	// Extended thinking
	var budgetTokens int
	switch strings.ToLower(cfg.ReasoningEffort) {
	case "low":
		budgetTokens = 2048
	case "medium":
		budgetTokens = 8192
	case "high":
		budgetTokens = 16384
	case "xhigh":
		budgetTokens = 32768
	}

	if budgetTokens > 0 {
		payload["thinking"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": budgetTokens,
		}
		if maxTokens <= budgetTokens {
			payload["max_tokens"] = budgetTokens + 4096
		}
	} else if thinking, ok := cfg.ExtraBody["thinking"]; ok {
		payload["thinking"] = thinking
	}

	// Extract system messages and convert user/assistant messages
	var systemBlocks []map[string]any
	var chatMessages []map[string]any

	for _, msg := range msgs {
		if msg == nil {
			continue
		}

		if msg.Role == "system" {
			for _, blk := range msg.Blocks {
				switch b := blk.(type) {
				case ir.TextBlock:
					systemBlocks = append(systemBlocks, map[string]any{
						"type": "text",
						"text": b.Text,
					})
				case *ir.TextBlock:
					systemBlocks = append(systemBlocks, map[string]any{
						"type": "text",
						"text": b.Text,
					})
				case ir.CacheControl:
					if b.Breakpoint && len(systemBlocks) > 0 {
						systemBlocks[len(systemBlocks)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
					}
				case *ir.CacheControl:
					if b.Breakpoint && len(systemBlocks) > 0 {
						systemBlocks[len(systemBlocks)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
					}
				}
			}
			continue
		}

		// Non-system message
		role := msg.Role
		if role == "tool" {
			role = "user"
		}

		var contentBlocks []map[string]any
		for _, blk := range msg.Blocks {
			switch b := blk.(type) {
			case ir.TextBlock:
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "text",
					"text": b.Text,
				})
			case *ir.TextBlock:
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "text",
					"text": b.Text,
				})
			case ir.ImageBlock:
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "image",
					"source": map[string]any{
						"type": "url",
						"url":  b.URL,
					},
				})
			case *ir.ImageBlock:
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "image",
					"source": map[string]any{
						"type": "url",
						"url":  b.URL,
					},
				})
			case ir.ToolCallBlock:
				var inputObj any
				if err := json.Unmarshal([]byte(b.Arguments), &inputObj); err != nil {
					inputObj = map[string]any{}
				}
				contentBlocks = append(contentBlocks, map[string]any{
					"type":  "tool_use",
					"id":    b.ID,
					"name":  b.Name,
					"input": inputObj,
				})
			case *ir.ToolCallBlock:
				var inputObj any
				if err := json.Unmarshal([]byte(b.Arguments), &inputObj); err != nil {
					inputObj = map[string]any{}
				}
				contentBlocks = append(contentBlocks, map[string]any{
					"type":  "tool_use",
					"id":    b.ID,
					"name":  b.Name,
					"input": inputObj,
				})
			case ir.ToolResultBlock:
				contentBlocks = append(contentBlocks, map[string]any{
					"type":        "tool_result",
					"tool_use_id": b.ToolCallID,
					"content":     b.Content,
				})
			case *ir.ToolResultBlock:
				contentBlocks = append(contentBlocks, map[string]any{
					"type":        "tool_result",
					"tool_use_id": b.ToolCallID,
					"content":     b.Content,
				})
			case ir.ReasoningBlock:
				if b.ReasoningKind == ir.ReasoningRedacted {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "redacted_thinking",
						"data": string(b.Opaque),
					})
				} else {
					th := map[string]any{
						"type":     "thinking",
						"thinking": b.Text,
					}
					if b.Signature != "" {
						th["signature"] = b.Signature
					}
					contentBlocks = append(contentBlocks, th)
				}
			case *ir.ReasoningBlock:
				if b.ReasoningKind == ir.ReasoningRedacted {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "redacted_thinking",
						"data": string(b.Opaque),
					})
				} else {
					th := map[string]any{
						"type":     "thinking",
						"thinking": b.Text,
					}
					if b.Signature != "" {
						th["signature"] = b.Signature
					}
					contentBlocks = append(contentBlocks, th)
				}
			case ir.CacheControl:
				if b.Breakpoint && len(contentBlocks) > 0 {
					contentBlocks[len(contentBlocks)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
				}
			case *ir.CacheControl:
				if b.Breakpoint && len(contentBlocks) > 0 {
					contentBlocks[len(contentBlocks)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
				}
			}
		}

		chatMessages = append(chatMessages, map[string]any{
			"role":    role,
			"content": contentBlocks,
		})
	}

	if len(systemBlocks) > 0 {
		payload["system"] = systemBlocks
	}
	payload["messages"] = chatMessages

	for k, v := range cfg.ExtraBody {
		if k == "thinking" {
			continue
		}
		payload[k] = v
	}

	return payload, nil
}

func (p *Provider) parseResponse(data []byte) (*ir.Response, error) {
	var raw struct {
		ID      string `json:"id"`
		Content []struct {
			Type      string          `json:"type"`
			Text      string          `json:"text,omitempty"`
			Thinking  string          `json:"thinking,omitempty"`
			Signature string          `json:"signature,omitempty"`
			Data      string          `json:"data,omitempty"`
			ID        string          `json:"id,omitempty"`
			Name      string          `json:"name,omitempty"`
			Input     json.RawMessage `json:"input,omitempty"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	var blocks []ir.Block
	for idx, blk := range raw.Content {
		switch blk.Type {
		case "thinking":
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningText,
				Text:          blk.Thinking,
				Signature:     blk.Signature,
			})
		case "redacted_thinking":
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningRedacted,
				Opaque:        []byte(blk.Data),
			})
		case "text":
			blocks = append(blocks, ir.TextBlock{
				Text: blk.Text,
			})
		case "tool_use":
			blocks = append(blocks, ir.ToolCallBlock{
				Index:     idx,
				ID:        blk.ID,
				Name:      blk.Name,
				Arguments: string(blk.Input),
			})
		}
	}

	usage := &ir.Usage{
		PromptTokens:             raw.Usage.InputTokens,
		CompletionTokens:         raw.Usage.OutputTokens,
		TotalTokens:              raw.Usage.InputTokens + raw.Usage.OutputTokens,
		CacheCreationInputTokens: raw.Usage.CacheCreationInputTokens,
		CacheReadInputTokens:     raw.Usage.CacheReadInputTokens,
	}

	return &ir.Response{
		ID:           raw.ID,
		UpstreamID:   raw.ID,
		FinishReason: raw.StopReason,
		Usage:        usage,
		Message: &ir.Message{
			Role:   "assistant",
			Blocks: blocks,
		},
	}, nil
}

// ConvertStreamState converts a decoder StreamState from pkg/spikes/anthropic into ir.Response.
func ConvertStreamState(id string, state *spikesanthropic.StreamState) *ir.Response {
	var blocks []ir.Block

	// Iterate in index order
	maxIdx := -1
	for idx := range state.Blocks {
		if idx > maxIdx {
			maxIdx = idx
		}
	}

	for i := 0; i <= maxIdx; i++ {
		blk, ok := state.Blocks[i]
		if !ok {
			continue
		}
		switch blk.Type {
		case spikesanthropic.BlockTypeThinking:
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningText,
				Text:          blk.Thinking,
				Signature:     blk.Signature,
			})
		case spikesanthropic.BlockTypeRedactedThinking:
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningRedacted,
				Opaque:        []byte(blk.RedactedData),
			})
		case spikesanthropic.BlockTypeText:
			blocks = append(blocks, ir.TextBlock{
				Text: blk.Text,
			})
		case spikesanthropic.BlockTypeToolUse:
			blocks = append(blocks, ir.ToolCallBlock{
				Index:     blk.Index,
				ID:        blk.ToolUseID,
				Name:      blk.ToolName,
				Arguments: blk.ToolInputJSON,
			})
		}
	}

	return &ir.Response{
		ID:           id,
		UpstreamID:   id,
		FinishReason: state.StopReason,
		Usage: &ir.Usage{
			PromptTokens:             state.Usage.InputTokens,
			CompletionTokens:         state.Usage.OutputTokens,
			TotalTokens:              state.Usage.InputTokens + state.Usage.OutputTokens,
			CacheCreationInputTokens: state.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:     state.Usage.CacheReadInputTokens,
		},
		Message: &ir.Message{
			Role:   "assistant",
			Blocks: blocks,
		},
	}
}
