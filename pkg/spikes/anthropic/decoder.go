package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// BlockType represents the type of content block in Anthropic stream.
type BlockType string

const (
	BlockTypeText             BlockType = "text"
	BlockTypeThinking         BlockType = "thinking"
	BlockTypeRedactedThinking BlockType = "redacted_thinking"
	BlockTypeToolUse          BlockType = "tool_use"
)

// Usage tracks cumulative token and cache usage.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// Accumulate merges another usage record into this usage.
func (u *Usage) Accumulate(other Usage) {
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
	u.CacheCreationInputTokens += other.CacheCreationInputTokens
	u.CacheReadInputTokens += other.CacheReadInputTokens
}

// ContentBlock tracks the state of a single content block in the stream.
type ContentBlock struct {
	Index         int       `json:"index"`
	Type          BlockType `json:"type"`
	Text          string    `json:"text,omitempty"`
	Thinking      string    `json:"thinking,omitempty"`
	Signature     string    `json:"signature,omitempty"`
	RedactedData  string    `json:"redacted_data,omitempty"`
	ToolUseID     string    `json:"id,omitempty"`
	ToolName      string    `json:"name,omitempty"`
	ToolInputJSON string    `json:"input_json,omitempty"`
}

// StreamError represents an error event from Anthropic.
type StreamError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func (e *StreamError) Error() string {
	return fmt.Sprintf("anthropic stream error (%s): %s", e.Type, e.Message)
}

// StreamState holds the cumulative state of an Anthropic Messages streaming response.
type StreamState struct {
	MessageID    string                `json:"id"`
	Model        string                `json:"model"`
	Role         string                `json:"role"`
	Blocks       map[int]*ContentBlock `json:"blocks"`
	Usage        Usage                 `json:"usage"`
	StopReason   string                `json:"stop_reason,omitempty"`
	StopSequence string                `json:"stop_sequence,omitempty"`
	Error        *StreamError          `json:"error,omitempty"`
	Completed    bool                  `json:"completed"`
	PingCount    int                   `json:"ping_count"`
}

// Block returns the content block at the given index, or nil if not found.
func (s *StreamState) Block(index int) *ContentBlock {
	return s.Blocks[index]
}

// FullText returns the concatenated text across all text blocks.
func (s *StreamState) FullText() string {
	var sb strings.Builder
	for i := 0; i < len(s.Blocks); i++ {
		if b, ok := s.Blocks[i]; ok && b.Type == BlockTypeText {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

// Decoder decodes Anthropic Messages API SSE events and updates state.
type Decoder struct {
	state *StreamState
}

// NewDecoder creates a new Anthropic streaming decoder.
func NewDecoder() *Decoder {
	return &Decoder{
		state: &StreamState{
			Blocks: make(map[int]*ContentBlock),
		},
	}
}

// State returns the current streaming state.
func (d *Decoder) State() *StreamState {
	return d.state
}

// rawEvent represents common fields across Anthropic SSE data payloads.
type rawEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message struct {
		ID    string `json:"id"`
		Type  string `json:"type"`
		Role  string `json:"role"`
		Model string `json:"model"`
		Usage Usage  `json:"usage"`
	} `json:"message"`
	ContentBlock struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
		Data string `json:"data"`
		Text string `json:"text"`
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
	Usage Usage `json:"usage"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// HandleEvent processes a single SSE event frame.
func (d *Decoder) HandleEvent(eventType string, data []byte) error {
	if len(data) == 0 {
		return nil
	}

	var raw rawEvent
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("failed to unmarshal Anthropic event data: %w", err)
	}

	// Use eventType or payload type
	effectiveType := eventType
	if effectiveType == "" || effectiveType == "message" {
		effectiveType = raw.Type
	}

	switch effectiveType {
	case "message_start":
		d.state.MessageID = raw.Message.ID
		d.state.Model = raw.Message.Model
		d.state.Role = raw.Message.Role
		d.state.Usage.Accumulate(raw.Message.Usage)

	case "content_block_start":
		blk := &ContentBlock{
			Index: raw.Index,
			Type:  BlockType(raw.ContentBlock.Type),
		}
		if blk.Type == BlockTypeToolUse {
			blk.ToolUseID = raw.ContentBlock.ID
			blk.ToolName = raw.ContentBlock.Name
		} else if blk.Type == BlockTypeRedactedThinking {
			blk.RedactedData = raw.ContentBlock.Data
		} else if blk.Type == BlockTypeText && raw.ContentBlock.Text != "" {
			blk.Text = raw.ContentBlock.Text
		}
		d.state.Blocks[raw.Index] = blk

	case "content_block_delta":
		blk, exists := d.state.Blocks[raw.Index]
		if !exists {
			blk = &ContentBlock{Index: raw.Index}
			d.state.Blocks[raw.Index] = blk
		}
		switch raw.Delta.Type {
		case "text_delta":
			// DO NOT TrimSpace streamed text deltas (whitespace preservation matters).
			blk.Type = BlockTypeText
			blk.Text += raw.Delta.Text
		case "thinking_delta":
			blk.Type = BlockTypeThinking
			blk.Thinking += raw.Delta.Thinking
		case "signature_delta":
			blk.Signature += raw.Delta.Signature
		case "input_json_delta":
			blk.Type = BlockTypeToolUse
			blk.ToolInputJSON += raw.Delta.PartialJSON
		}

	case "content_block_stop":
		// block is finalized; already indexed in state.Blocks

	case "message_delta":
		if raw.Delta.StopReason != "" {
			d.state.StopReason = raw.Delta.StopReason
		}
		if raw.Delta.StopSequence != "" {
			d.state.StopSequence = raw.Delta.StopSequence
		}
		d.state.Usage.Accumulate(raw.Usage)

	case "message_stop":
		d.state.Completed = true

	case "ping":
		d.state.PingCount++

	case "error":
		d.state.Error = &StreamError{
			Type:    raw.Error.Type,
			Message: raw.Error.Message,
		}
	}

	return nil
}

// Decode reads an entire SSE stream from r and returns the resulting StreamState.
func (d *Decoder) Decode(r io.Reader) (*StreamState, error) {
	scanner := bufio.NewScanner(r)
	var currentEvent string
	var currentData strings.Builder

	for scanner.Scan() {
		line := scanner.Text()

		// Empty line triggers event dispatch
		if line == "" {
			if currentData.Len() > 0 || currentEvent != "" {
				dataBytes := []byte(currentData.String())
				if err := d.HandleEvent(currentEvent, dataBytes); err != nil {
					return nil, err
				}
				currentEvent = ""
				currentData.Reset()
			}
			continue
		}

		if strings.HasPrefix(line, ":") {
			// Comment line, ignore
			continue
		}

		if strings.HasPrefix(line, "event:") {
			val := strings.TrimPrefix(line, "event:")
			currentEvent = strings.TrimSpace(val)
			continue
		}

		if strings.HasPrefix(line, "data:") {
			val := strings.TrimPrefix(line, "data:")
			// Strip at most one leading space per SSE specification, but DO NOT TrimSpace!
			val = strings.TrimPrefix(val, " ")
			if currentData.Len() > 0 {
				currentData.WriteString("\n")
			}
			currentData.WriteString(val)
			continue
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanner error while reading SSE: %w", err)
	}

	// Dispatch trailing event if present
	if currentData.Len() > 0 || currentEvent != "" {
		dataBytes := []byte(currentData.String())
		if err := d.HandleEvent(currentEvent, dataBytes); err != nil {
			return nil, err
		}
	}

	return d.state, nil
}
