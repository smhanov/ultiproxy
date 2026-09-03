package openai

import "encoding/json"

// ChatCompletionRequest is the standard OpenAI chat completion request payload.
type ChatCompletionRequest struct {
	Model               string         `json:"model"`
	Messages            []ChatMessage  `json:"messages"`
	Stream              bool           `json:"stream,omitempty"`
	StreamOptions       *StreamOptions `json:"stream_options,omitempty"`
	MaxTokens           int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens int            `json:"max_completion_tokens,omitempty"`
	Temperature         *float64       `json:"temperature,omitempty"`
	ReasoningEffort     string         `json:"reasoning_effort,omitempty"`
	Tools               []ChatTool     `json:"tools,omitempty"`
	ToolChoice          any            `json:"tool_choice,omitempty"`
	CodebuffMetadata    any            `json:"codebuff_metadata,omitempty"`
	Provider            any            `json:"provider,omitempty"`
	Extra               map[string]any `json:"-"`
}

// MarshalJSON merges Extra fields into the top-level payload.
func (r ChatCompletionRequest) MarshalJSON() ([]byte, error) {
	type Alias ChatCompletionRequest
	raw, err := json.Marshal(Alias(r))
	if err != nil {
		return nil, err
	}
	if len(r.Extra) == 0 {
		return raw, nil
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	for k, v := range r.Extra {
		m[k] = v
	}
	return json.Marshal(m)
}

// StreamOptions configures stream behavior.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ChatMessage represents a message in the conversation.
type ChatMessage struct {
	Role             string         `json:"role"`
	Content          any            `json:"content"` // string or []ContentPart
	Name             string         `json:"name,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
	ToolCalls        []ChatToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
}

// ContentPart represents a multimodal part.
type ContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *ImageURLPart `json:"image_url,omitempty"`
}

// ImageURLPart holds image URL details.
type ImageURLPart struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// ChatTool describes a tool available to the model.
type ChatTool struct {
	Type     string       `json:"type"` // "function"
	Function ChatFunction `json:"function"`
}

// ChatFunction describes a function definition.
type ChatFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// ChatToolCall describes a tool call invocation.
type ChatToolCall struct {
	Index    int              `json:"index,omitempty"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"` // "function"
	Function ChatFunctionCall `json:"function"`
}

// ChatFunctionCall describes function call arguments.
type ChatFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ChatCompletionResponse is the non-streaming response format.
type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Choices []ChatChoice `json:"choices"`
	Usage   *ChatUsage   `json:"usage,omitempty"`
}

// ChatChoice represents a choice in non-streaming responses.
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatUsage holds token usage metrics.
type ChatUsage struct {
	PromptTokens             int     `json:"prompt_tokens"`
	CompletionTokens         int     `json:"completion_tokens"`
	TotalTokens              int     `json:"total_tokens"`
	ReasoningTokens          int     `json:"reasoning_tokens,omitempty"`
	CacheCreationInputTokens int     `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int     `json:"cache_read_input_tokens,omitempty"`
	TotalCost                float64 `json:"total_cost,omitempty"` // OpenRouter extension
}

// ChatCompletionChunk represents an SSE chunk during streaming.
type ChatCompletionChunk struct {
	ID      string            `json:"id"`
	Choices []ChatChunkChoice `json:"choices"`
	Usage   *ChatUsage        `json:"usage,omitempty"`
}

// ChatChunkChoice represents one choice in an SSE chunk.
type ChatChunkChoice struct {
	Index        int       `json:"index"`
	Delta        ChatDelta `json:"delta"`
	FinishReason *string   `json:"finish_reason"`
}

// ChatDelta represents streamed increments.
type ChatDelta struct {
	Role             string         `json:"role,omitempty"`
	Content          string         `json:"content,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []ChatToolCall `json:"tool_calls,omitempty"`
}
