package ir

// Event is the interface implemented by all streaming event types.
type Event interface {
	EventKind() string
}

// EventMessageStart indicates the beginning of a message stream.
type EventMessageStart struct {
	ID string `json:"id"`
}

func (e EventMessageStart) EventKind() string { return "message_start" }

// EventBlockStart indicates a new content block has begun.
type EventBlockStart struct {
	Index int    `json:"index"`
	Kind  string `json:"kind"`
}

func (e EventBlockStart) EventKind() string { return "block_start" }

// EventTextDelta represents a chunk of streamed text.
type EventTextDelta struct {
	Index int    `json:"index"`
	Text  string `json:"text"`
}

func (e EventTextDelta) EventKind() string { return "text_delta" }

// EventReasoningDelta represents a chunk of streamed reasoning/thinking text.
type EventReasoningDelta struct {
	Index int    `json:"index"`
	Text  string `json:"text"`
}

func (e EventReasoningDelta) EventKind() string { return "reasoning_delta" }

// EventReasoningSignature provides the cryptographic or opaque signature for reasoning.
type EventReasoningSignature struct {
	Index     int    `json:"index"`
	Signature string `json:"signature"`
}

func (e EventReasoningSignature) EventKind() string { return "reasoning_signature" }

// EventToolCallStart indicates the model is initiating a tool/function call.
type EventToolCallStart struct {
	Index int    `json:"index"`
	ID    string `json:"id"`
	Name  string `json:"name"`
}

func (e EventToolCallStart) EventKind() string { return "tool_call_start" }

// EventToolArgumentsDelta represents an argument fragment for a tool call.
type EventToolArgumentsDelta struct {
	Index     int    `json:"index"`
	Arguments string `json:"arguments"`
}

func (e EventToolArgumentsDelta) EventKind() string { return "tool_arguments_delta" }

// EventToolCallStop marks the end of argument streaming for a tool call.
type EventToolCallStop struct {
	Index int `json:"index"`
}

func (e EventToolCallStop) EventKind() string { return "tool_call_stop" }

// EventUsageUpdate communicates cumulative token counts and estimated cost.
type EventUsageUpdate struct {
	PromptTokens             int     `json:"prompt_tokens"`
	CompletionTokens         int     `json:"completion_tokens"`
	TotalTokens              int     `json:"total_tokens"`
	CacheCreationInputTokens int     `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int     `json:"cache_read_input_tokens"`
	Cost                     float64 `json:"cost"`
}

func (e EventUsageUpdate) EventKind() string { return "usage_update" }

// EventMessageStop indicates message streaming has finished.
type EventMessageStop struct {
	FinishReason string `json:"finish_reason"`
	UpstreamID   string `json:"upstream_id"`
}

func (e EventMessageStop) EventKind() string { return "message_stop" }

// EventUpstreamError communicates an error from the upstream provider.
type EventUpstreamError struct {
	Kind              string `json:"kind"`
	Message           string `json:"message"`
	RetryAfterSeconds int    `json:"retry_after_seconds,omitempty"`
	Permanent         bool   `json:"permanent"`
}

func (e EventUpstreamError) EventKind() string { return "upstream_error" }
