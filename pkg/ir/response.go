package ir

// Usage captures token accounting for one request/stream.
type Usage struct {
	PromptTokens             int     `json:"prompt_tokens"`
	CompletionTokens         int     `json:"completion_tokens"`
	TotalTokens              int     `json:"total_tokens"`
	ReasoningTokens          int     `json:"reasoning_tokens,omitempty"`
	CacheCreationInputTokens int     `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int     `json:"cache_read_input_tokens,omitempty"`
	Cost                     float64 `json:"cost,omitempty"`
}

// Response is a normalized non-streaming completion result.
type Response struct {
	ID           string   `json:"id,omitempty"`
	Message      *Message `json:"message,omitempty"`
	FinishReason string   `json:"finish_reason,omitempty"`
	Usage        *Usage   `json:"usage,omitempty"`
	UpstreamID   string   `json:"upstream_id,omitempty"`
}
