package copilot

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/sse"
	"github.com/smhanov/ultiproxy/pkg/spikes/responses"
)

const (
	DefaultBaseURL      = "https://api.githubcopilot.com"
	DefaultQuotaBaseURL = "https://api.github.com"

	EditorVersion        = "vscode/1.104.1"
	EditorPluginVersion  = "copilot-chat/0.26.7"
	CopilotIntegrationID = "vscode-chat"
	CopilotUserAgent     = "GitHubCopilotChat/0.26.7"
)

var responsesModels = map[string]bool{
	"gpt-5.4":            true,
	"gpt-5.3-codex":      true,
	"gpt-5.4-mini":       true,
	"mai-code-1.1-flash": true,
}

// Config configures the GitHub Copilot provider adapter.
type Config struct {
	Token        string
	BaseURL      string
	QuotaBaseURL string
	HTTPClient   *http.Client
}

// Provider implements InferenceProvider, QuotaProvider, and AuthProvider for GitHub Copilot.
type Provider struct {
	token        string
	baseURL      string
	quotaBaseURL string
	httpClient   *http.Client
}

// New creates a new GitHub Copilot provider adapter.
func New(cfg Config) *Provider {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")

	quotaBaseURL := cfg.QuotaBaseURL
	if quotaBaseURL == "" {
		quotaBaseURL = DefaultQuotaBaseURL
	}
	quotaBaseURL = strings.TrimRight(quotaBaseURL, "/")

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	return &Provider{
		token:        cfg.Token,
		baseURL:      baseURL,
		quotaBaseURL: quotaBaseURL,
		httpClient:   client,
	}
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:      true,
		Tools:     true,
		Reasoning: true,
		Streaming: true,
		Vision:    false,
	}
}

// ProviderBundle returns a provider.Provider bundle for registration.
func (p *Provider) ProviderBundle() provider.Provider {
	return provider.Provider{
		Inference:    p,
		Quota:        p,
		Auth:         p,
		Capabilities: p.Capabilities(),
	}
}

// Register registers this provider with a provider.Registry.
func (p *Provider) Register(r *provider.Registry) {
	r.Register(p.ProviderBundle())
}

// Name implements InferenceProvider, QuotaProvider, and AuthProvider.
func (p *Provider) Name() string {
	return "copilot"
}

// applyEditorHeaders sets mandatory Copilot headers on an outgoing request.
func (p *Provider) applyEditorHeaders(req *http.Request) {
	req.Header.Set("Editor-Version", EditorVersion)
	req.Header.Set("Editor-Plugin-Version", EditorPluginVersion)
	req.Header.Set("Copilot-Integration-Id", CopilotIntegrationID)
	req.Header.Set("User-Agent", CopilotUserAgent)
}

// cleanToken returns the raw token string without "Bearer " or "token " prefix.
func cleanToken(tok string) string {
	tok = strings.TrimSpace(tok)
	if strings.HasPrefix(tok, "Bearer ") {
		return strings.TrimSpace(strings.TrimPrefix(tok, "Bearer "))
	}
	if strings.HasPrefix(tok, "token ") {
		return strings.TrimSpace(strings.TrimPrefix(tok, "token "))
	}
	return tok
}

// DetermineInitiator calculates the X-Initiator header value.
//
// SEMANTIC CAVEAT:
// Inferring X-Initiator from prior transcript history assumes that any request containing
// a prior assistant or tool turn is part of an autonomous agent continuation ("agent"),
// whereas a request with only user/system prompts is an interactive human turn ("user").
// However, in multi-turn conversational chat, a human user frequently responds to a previous
// assistant message, which this heuristic would classify as "agent". Clients should explicitly
// pass the initiator via ir.Message.Meta["x-initiator"] (or WithHeader "X-Ultiproxy-Initiator")
// when a turn is genuinely user-initiated in a multi-turn chat to ensure accurate billing attribution.
func DetermineInitiator(msgs []*ir.Message, headers map[string]string) string {
	// 1. Explicit override via request headers
	if v := headers["X-Ultiproxy-Initiator"]; v != "" {
		return v
	}
	if v := headers["x-ultiproxy-initiator"]; v != "" {
		return v
	}

	// 2. Explicit override via Message metadata (check latest messages first)
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i] != nil && msgs[i].Meta != nil {
			for _, k := range []string{"X-Ultiproxy-Initiator", "x-ultiproxy-initiator", "x-initiator", "initiator"} {
				if val, ok := msgs[i].Meta[k]; ok && val != "" {
					return val
				}
			}
		}
	}

	// 3. Otherwise infer: if any prior assistant/tool message in msgs -> "agent" else "user"
	for _, m := range msgs {
		if m == nil {
			continue
		}
		if m.Role == "assistant" || m.Role == "tool" {
			return "agent"
		}
		for _, b := range m.Blocks {
			if b == nil {
				continue
			}
			switch b.(type) {
			case ir.ToolCallBlock, *ir.ToolCallBlock, ir.ToolResultBlock, *ir.ToolResultBlock:
				return "agent"
			}
		}
	}

	return "user"
}

func isResponsesModel(model string) bool {
	return responsesModels[model]
}

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	cfg := provider.NewRequestConfig(opts...)

	if isResponsesModel(cfg.Model) {
		return p.generateResponses(ctx, msgs, cfg)
	}
	return p.generateChat(ctx, msgs, cfg)
}

// Stream implements provider.InferenceProvider.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	cfg := provider.NewRequestConfig(opts...)

	if isResponsesModel(cfg.Model) {
		return p.streamResponses(ctx, msgs, cfg)
	}
	return p.streamChat(ctx, msgs, cfg)
}

// -----------------------------------------------------------------------------
// OpenAI-Compatible Chat Completions
// -----------------------------------------------------------------------------

type openAIChatMessage struct {
	Role       string           `json:"role"`
	Content    any              `json:"content,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openAIToolCall struct {
	Index    int                `json:"index,omitempty"`
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIFunctionCall `json:"function"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIChatRequest struct {
	Model       string              `json:"model"`
	Messages    []openAIChatMessage `json:"messages"`
	Stream      bool                `json:"stream"`
	MaxTokens   int                 `json:"max_tokens,omitempty"`
	Temperature *float64            `json:"temperature,omitempty"`
}

type openAIChatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Role      string           `json:"role"`
			Content   string           `json:"content"`
			ToolCalls []openAIToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func convertIRToChatMessages(msgs []*ir.Message) []openAIChatMessage {
	var out []openAIChatMessage
	for _, m := range msgs {
		if m == nil {
			continue
		}
		var textParts []string
		var toolCalls []openAIToolCall
		var toolResultID string

		for _, blk := range m.Blocks {
			if blk == nil {
				continue
			}
			switch b := blk.(type) {
			case ir.TextBlock:
				textParts = append(textParts, b.Text)
			case *ir.TextBlock:
				textParts = append(textParts, b.Text)
			case ir.ToolCallBlock:
				toolCalls = append(toolCalls, openAIToolCall{
					Index: b.Index,
					ID:    b.ID,
					Type:  "function",
					Function: openAIFunctionCall{
						Name:      b.Name,
						Arguments: b.Arguments,
					},
				})
			case *ir.ToolCallBlock:
				toolCalls = append(toolCalls, openAIToolCall{
					Index: b.Index,
					ID:    b.ID,
					Type:  "function",
					Function: openAIFunctionCall{
						Name:      b.Name,
						Arguments: b.Arguments,
					},
				})
			case ir.ToolResultBlock:
				toolResultID = b.ToolCallID
				textParts = append(textParts, b.Content)
			case *ir.ToolResultBlock:
				toolResultID = b.ToolCallID
				textParts = append(textParts, b.Content)
			}
		}

		cm := openAIChatMessage{
			Role:      m.Role,
			Content:   strings.Join(textParts, "\n"),
			ToolCalls: toolCalls,
		}
		if m.Role == "tool" || toolResultID != "" {
			cm.Role = "tool"
			cm.ToolCallID = toolResultID
		}
		out = append(out, cm)
	}
	return out
}

func (p *Provider) generateChat(ctx context.Context, msgs []*ir.Message, cfg *provider.RequestConfig) (*ir.Response, error) {
	reqBody := openAIChatRequest{
		Model:       cfg.Model,
		Messages:    convertIRToChatMessages(msgs),
		Stream:      false,
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to marshal chat request: %w", err)
	}

	endpoint := p.baseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create chat request: %w", err)
	}

	p.applyEditorHeaders(req)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cleanToken(p.token))
	req.Header.Set("X-Initiator", DetermineInitiator(msgs, cfg.Headers))
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("copilot: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("copilot: chat completion upstream returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	var chatResp openAIChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("copilot: failed to parse chat response: %w", err)
	}

	irResp := &ir.Response{
		ID:         chatResp.ID,
		UpstreamID: chatResp.ID,
	}

	if len(chatResp.Choices) > 0 {
		choice := chatResp.Choices[0]
		irResp.FinishReason = choice.FinishReason

		var blocks []ir.Block
		if choice.Message.Content != "" {
			blocks = append(blocks, ir.TextBlock{Text: choice.Message.Content})
		}
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

	if chatResp.Usage != nil {
		irResp.Usage = &ir.Usage{
			PromptTokens:     chatResp.Usage.PromptTokens,
			CompletionTokens: chatResp.Usage.CompletionTokens,
			TotalTokens:      chatResp.Usage.TotalTokens,
		}
	}

	return irResp, nil
}

func (p *Provider) streamChat(ctx context.Context, msgs []*ir.Message, cfg *provider.RequestConfig) (<-chan ir.Event, error) {
	reqBody := openAIChatRequest{
		Model:       cfg.Model,
		Messages:    convertIRToChatMessages(msgs),
		Stream:      true,
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to marshal stream request: %w", err)
	}

	endpoint := p.baseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create stream request: %w", err)
	}

	p.applyEditorHeaders(req)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cleanToken(p.token))
	req.Header.Set("X-Initiator", DetermineInitiator(msgs, cfg.Headers))
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("copilot: stream request failed: %w", err)
	}

	// Synchronous error check on non-2xx
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("copilot: stream upstream returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	outCh := make(chan ir.Event, 64)

	go func() {
		defer close(outCh)
		defer resp.Body.Close()

		scanner := sse.NewScanner(resp.Body)
		started := false
		var responseID string

		type streamChunk struct {
			ID      string `json:"id"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Role             string           `json:"role,omitempty"`
					Content          string           `json:"content,omitempty"`
					ReasoningContent string           `json:"reasoning_content,omitempty"`
					ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}

		for scanner.Scan() {
			ev := scanner.Event()
			data := bytes.TrimSpace(ev.Data)
			if len(data) == 0 {
				continue
			}
			if string(data) == "[DONE]" {
				break
			}

			var chunk streamChunk
			if err := json.Unmarshal(data, &chunk); err != nil {
				continue
			}

			if !started && chunk.ID != "" {
				responseID = chunk.ID
				outCh <- ir.EventMessageStart{ID: chunk.ID}
				started = true
			}

			if chunk.Usage != nil {
				outCh <- ir.EventUsageUpdate{
					PromptTokens:     chunk.Usage.PromptTokens,
					CompletionTokens: chunk.Usage.CompletionTokens,
					TotalTokens:      chunk.Usage.TotalTokens,
				}
			}

			for _, c := range chunk.Choices {
				if c.Delta.ReasoningContent != "" {
					outCh <- ir.EventReasoningDelta{Index: c.Index, Text: c.Delta.ReasoningContent}
				}
				if c.Delta.Content != "" {
					outCh <- ir.EventTextDelta{Index: c.Index, Text: c.Delta.Content}
				}
				for _, tc := range c.Delta.ToolCalls {
					if tc.Function.Name != "" {
						outCh <- ir.EventToolCallStart{Index: tc.Index, ID: tc.ID, Name: tc.Function.Name}
					}
					if tc.Function.Arguments != "" {
						outCh <- ir.EventToolArgumentsDelta{Index: tc.Index, Arguments: tc.Function.Arguments}
					}
				}
				if c.FinishReason != nil && *c.FinishReason != "" {
					outCh <- ir.EventMessageStop{FinishReason: *c.FinishReason, UpstreamID: responseID}
				}
			}
		}

		if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
			outCh <- ir.EventUpstreamError{
				Kind:      "stream_error",
				Message:   err.Error(),
				Permanent: false,
			}
		}
	}()

	return outCh, nil
}

// -----------------------------------------------------------------------------
// Responses API Translator & Bridge
// -----------------------------------------------------------------------------

type responsesInputContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type responsesInputItem struct {
	Type      string                  `json:"type"`
	Role      string                  `json:"role,omitempty"`
	Content   []responsesInputContent `json:"content,omitempty"`
	CallID    string                  `json:"call_id,omitempty"`
	Name      string                  `json:"name,omitempty"`
	Arguments string                  `json:"arguments,omitempty"`
	Output    string                  `json:"output,omitempty"`
}

type responsesRequest struct {
	Model  string               `json:"model"`
	Input  []responsesInputItem `json:"input"`
	Stream bool                 `json:"stream"`
}

func convertIRToResponsesInput(msgs []*ir.Message) []responsesInputItem {
	var out []responsesInputItem
	for _, m := range msgs {
		if m == nil {
			continue
		}
		for _, blk := range m.Blocks {
			if blk == nil {
				continue
			}
			switch b := blk.(type) {
			case ir.TextBlock:
				partType := "input_text"
				if m.Role == "assistant" {
					partType = "output_text"
				}
				out = append(out, responsesInputItem{
					Type:    "message",
					Role:    m.Role,
					Content: []responsesInputContent{{Type: partType, Text: b.Text}},
				})
			case *ir.TextBlock:
				partType := "input_text"
				if m.Role == "assistant" {
					partType = "output_text"
				}
				out = append(out, responsesInputItem{
					Type:    "message",
					Role:    m.Role,
					Content: []responsesInputContent{{Type: partType, Text: b.Text}},
				})
			case ir.ToolCallBlock:
				out = append(out, responsesInputItem{
					Type:      "function_call",
					CallID:    b.ID,
					Name:      b.Name,
					Arguments: b.Arguments,
				})
			case *ir.ToolCallBlock:
				out = append(out, responsesInputItem{
					Type:      "function_call",
					CallID:    b.ID,
					Name:      b.Name,
					Arguments: b.Arguments,
				})
			case ir.ToolResultBlock:
				out = append(out, responsesInputItem{
					Type:   "function_call_output",
					CallID: b.ToolCallID,
					Output: b.Content,
				})
			case *ir.ToolResultBlock:
				out = append(out, responsesInputItem{
					Type:   "function_call_output",
					CallID: b.ToolCallID,
					Output: b.Content,
				})
			}
		}
	}
	return out
}

func (p *Provider) generateResponses(ctx context.Context, msgs []*ir.Message, cfg *provider.RequestConfig) (*ir.Response, error) {
	reqBody := responsesRequest{
		Model:  cfg.Model,
		Input:  convertIRToResponsesInput(msgs),
		Stream: false,
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to marshal responses request: %w", err)
	}

	endpoint := p.baseURL + "/responses"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create responses request: %w", err)
	}

	p.applyEditorHeaders(req)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cleanToken(p.token))
	req.Header.Set("X-Initiator", DetermineInitiator(msgs, cfg.Headers))
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("copilot: responses request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to read responses body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("copilot: responses upstream returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	return TranslateResponsesJSONToResponse(body)
}

// TranslateResponsesJSONToResponse parses a completed non-streaming Responses API JSON object.
func TranslateResponsesJSONToResponse(data []byte) (*ir.Response, error) {
	var raw struct {
		ID     string `json:"id"`
		Model  string `json:"model"`
		Status string `json:"status"`
		Output []struct {
			ID        string `json:"id"`
			Type      string `json:"type"`
			Role      string `json:"role"`
			CallID    string `json:"call_id,omitempty"`
			Name      string `json:"name,omitempty"`
			Arguments string `json:"arguments,omitempty"`
			Content   []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content,omitempty"`
			Summary []struct {
				Text string `json:"text"`
			} `json:"summary,omitempty"`
		} `json:"output"`
		Usage struct {
			InputTokens     int `json:"input_tokens"`
			OutputTokens    int `json:"output_tokens"`
			TotalTokens     int `json:"total_tokens"`
			ReasoningTokens int `json:"reasoning_tokens,omitempty"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("copilot: failed to parse responses json: %w", err)
	}

	var blocks []ir.Block
	for _, item := range raw.Output {
		switch item.Type {
		case "reasoning":
			var sb strings.Builder
			for _, sum := range item.Summary {
				sb.WriteString(sum.Text)
			}
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningText,
				Text:          sb.String(),
			})
		case "message":
			for _, part := range item.Content {
				blocks = append(blocks, ir.TextBlock{Text: part.Text})
			}
		case "function_call":
			blocks = append(blocks, ir.ToolCallBlock{
				ID:        item.CallID,
				Name:      item.Name,
				Arguments: item.Arguments,
			})
		}
	}

	finishReason := "stop"
	if raw.Status != "" && raw.Status != "completed" {
		finishReason = raw.Status
	}

	return &ir.Response{
		ID:           raw.ID,
		UpstreamID:   raw.ID,
		FinishReason: finishReason,
		Message: &ir.Message{
			Role:   "assistant",
			Blocks: blocks,
		},
		Usage: &ir.Usage{
			PromptTokens:     raw.Usage.InputTokens,
			CompletionTokens: raw.Usage.OutputTokens,
			TotalTokens:      raw.Usage.TotalTokens,
			ReasoningTokens:  raw.Usage.ReasoningTokens,
		},
	}, nil
}

func (p *Provider) streamResponses(ctx context.Context, msgs []*ir.Message, cfg *provider.RequestConfig) (<-chan ir.Event, error) {
	reqBody := responsesRequest{
		Model:  cfg.Model,
		Input:  convertIRToResponsesInput(msgs),
		Stream: true,
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to marshal responses stream request: %w", err)
	}

	endpoint := p.baseURL + "/responses"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create responses stream request: %w", err)
	}

	p.applyEditorHeaders(req)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cleanToken(p.token))
	req.Header.Set("X-Initiator", DetermineInitiator(msgs, cfg.Headers))
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("copilot: responses stream request failed: %w", err)
	}

	// Synchronous error check on non-2xx
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("copilot: responses upstream returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	outCh := make(chan ir.Event, 64)

	go func() {
		defer close(outCh)
		defer resp.Body.Close()

		translator := responses.NewTranslator()
		scanner := sse.NewScanner(resp.Body)

		for scanner.Scan() {
			ev := scanner.Event()
			normEvents, err := translator.Translate(ev.Type, ev.Data)
			if err != nil {
				outCh <- ir.EventUpstreamError{
					Kind:      "translation_error",
					Message:   err.Error(),
					Permanent: false,
				}
				return
			}

			for _, ne := range normEvents {
				switch ne.Kind {
				case responses.EventKindResponseCreated:
					outCh <- ir.EventMessageStart{ID: ne.ResponseID}
				case responses.EventKindItemAdded:
					outCh <- ir.EventBlockStart{Index: ne.OutputIndex, Kind: "text"}
				case responses.EventKindReasoningDelta:
					outCh <- ir.EventReasoningDelta{Index: ne.OutputIndex, Text: ne.Delta}
				case responses.EventKindTextDelta:
					outCh <- ir.EventTextDelta{Index: ne.OutputIndex, Text: ne.Delta}
				case responses.EventKindToolCallStart:
					outCh <- ir.EventToolCallStart{Index: ne.ToolIndex, ID: ne.CallID, Name: ne.ToolName}
				case responses.EventKindToolCallDelta:
					outCh <- ir.EventToolArgumentsDelta{Index: ne.ToolIndex, Arguments: ne.Delta}
				case responses.EventKindToolCallDone:
					outCh <- ir.EventToolCallStop{Index: ne.ToolIndex}
				case responses.EventKindCompleted:
					outCh <- ir.EventUsageUpdate{
						PromptTokens:     ne.Usage.InputTokens,
						CompletionTokens: ne.Usage.OutputTokens,
						TotalTokens:      ne.Usage.TotalTokens,
					}
					outCh <- ir.EventMessageStop{
						FinishReason: "stop",
						UpstreamID:   ne.ResponseID,
					}
				}
			}
		}

		if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
			outCh <- ir.EventUpstreamError{
				Kind:      "stream_error",
				Message:   err.Error(),
				Permanent: false,
			}
		}
	}()

	return outCh, nil
}

// -----------------------------------------------------------------------------
// QuotaProvider
// -----------------------------------------------------------------------------

type quotaUserResponse struct {
	CopilotPlan       string `json:"copilot_plan"`
	AccessTypeSKU     string `json:"access_type_sku"`
	QuotaResetDate    string `json:"quota_reset_date"`
	QuotaResetDateUTC string `json:"quota_reset_date_utc"`
	QuotaSnapshots    struct {
		Chat struct {
			Unlimited        bool    `json:"unlimited"`
			PercentRemaining float64 `json:"percent_remaining"`
			HasQuota         bool    `json:"has_quota"`
		} `json:"chat"`
		Completions struct {
			Unlimited        bool    `json:"unlimited"`
			PercentRemaining float64 `json:"percent_remaining"`
			HasQuota         bool    `json:"has_quota"`
		} `json:"completions"`
		PremiumInteractions struct {
			Unlimited        bool     `json:"unlimited"`
			PercentRemaining *float64 `json:"percent_remaining"`
			HasQuota         bool     `json:"has_quota"`
			CreditsUsed      *float64 `json:"credits_used"`
			Entitlement      *float64 `json:"entitlement"`
			Remaining        *float64 `json:"remaining"`
		} `json:"premium_interactions"`
	} `json:"quota_snapshots"`
}

// Quota implements provider.QuotaProvider.
func (p *Provider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	tok := cleanToken(p.token)
	if tok == "" {
		return nil, errors.New("copilot: token not configured")
	}

	endpoint := p.quotaBaseURL + "/copilot_internal/user"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create quota request: %w", err)
	}

	p.applyEditorHeaders(req)
	// Mandatory header for internal GitHub quota endpoint
	req.Header.Set("Authorization", "token "+tok)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("copilot: quota request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to read quota response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("copilot: quota endpoint returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	return ParseQuotaUserJSON(body)
}

// ParseQuotaUserJSON parses GitHub /copilot_internal/user payload into QuotaSnapshot.
func ParseQuotaUserJSON(data []byte) (*provider.QuotaSnapshot, error) {
	var resp quotaUserResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("copilot: failed to parse quota json: %w", err)
	}

	now := time.Now().UTC()
	var resetTime time.Time
	if resp.QuotaResetDateUTC != "" {
		t, err := time.Parse(time.RFC3339, resp.QuotaResetDateUTC)
		if err == nil {
			resetTime = t.UTC()
		}
	}
	if resetTime.IsZero() && resp.QuotaResetDate != "" {
		t, err := time.Parse("2006-01-02", resp.QuotaResetDate)
		if err == nil {
			resetTime = t.UTC()
		}
	}

	var secondsRemaining int64
	if !resetTime.IsZero() && resetTime.After(now) {
		secondsRemaining = int64(resetTime.Sub(now).Seconds())
	}

	var windows []provider.QuotaWindow
	prem := resp.QuotaSnapshots.PremiumInteractions

	var usedPct float64
	var limit float64
	var remaining float64

	if prem.Unlimited {
		usedPct = 0
		limit = 0
		remaining = 0
	} else if prem.Entitlement != nil && *prem.Entitlement > 0 {
		limit = *prem.Entitlement
		if prem.Remaining != nil {
			remaining = *prem.Remaining
		} else if prem.CreditsUsed != nil {
			remaining = limit - *prem.CreditsUsed
		}
		if prem.CreditsUsed != nil {
			usedPct = (*prem.CreditsUsed / limit) * 100.0
		} else {
			usedPct = (1.0 - (remaining / limit)) * 100.0
		}
	} else if prem.PercentRemaining != nil {
		limit = 100
		remaining = *prem.PercentRemaining
		usedPct = 100.0 - *prem.PercentRemaining
	}

	windows = append(windows, provider.QuotaWindow{
		Label:            "Premium requests",
		UsedPct:          usedPct,
		Remaining:        remaining,
		Limit:            limit,
		Unit:             "requests",
		ResetAt:          resetTime,
		SecondsRemaining: secondsRemaining,
	})

	chatUsed := 0.0
	if !resp.QuotaSnapshots.Chat.Unlimited && resp.QuotaSnapshots.Chat.PercentRemaining > 0 {
		chatUsed = 100.0 - resp.QuotaSnapshots.Chat.PercentRemaining
	}
	windows = append(windows, provider.QuotaWindow{
		Label:            "Chat (included)",
		UsedPct:          chatUsed,
		Remaining:        100 - chatUsed,
		Limit:            100,
		Unit:             "%",
		ResetAt:          resetTime,
		SecondsRemaining: secondsRemaining,
	})

	compUsed := 0.0
	if !resp.QuotaSnapshots.Completions.Unlimited && resp.QuotaSnapshots.Completions.PercentRemaining > 0 {
		compUsed = 100.0 - resp.QuotaSnapshots.Completions.PercentRemaining
	}
	windows = append(windows, provider.QuotaWindow{
		Label:            "Completions (included)",
		UsedPct:          compUsed,
		Remaining:        100 - compUsed,
		Limit:            100,
		Unit:             "%",
		ResetAt:          resetTime,
		SecondsRemaining: secondsRemaining,
	})

	detail := fmt.Sprintf("Premium %g/%g remaining · resets %s · chat+completions included/unlimited",
		remaining, limit, resp.QuotaResetDate)

	return &provider.QuotaSnapshot{
		ObservedAt: now,
		Windows:    windows,
		Detail:     detail,
	}, nil
}

// -----------------------------------------------------------------------------
// AuthProvider
// -----------------------------------------------------------------------------

func (p *Provider) Login(ctx context.Context) error {
	if p.token == "" {
		return auth.ErrNotFound
	}
	return nil
}

func (p *Provider) Token(ctx context.Context) (string, error) {
	tok := cleanToken(p.token)
	if tok == "" {
		return "", auth.ErrNotFound
	}
	return tok, nil
}

func (p *Provider) Refresh(ctx context.Context) error {
	// Copilot static tokens do not refresh via standard OAuth refresh
	return nil
}
