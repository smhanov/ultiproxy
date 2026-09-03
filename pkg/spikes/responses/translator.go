package responses

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// EventKind describes normalized event types.
type EventKind string

const (
	EventKindResponseCreated EventKind = "response_created"
	EventKindItemAdded       EventKind = "item_added"
	EventKindReasoningDelta  EventKind = "reasoning_delta"
	EventKindTextDelta       EventKind = "text_delta"
	EventKindToolCallStart   EventKind = "tool_call_start"
	EventKindToolCallDelta   EventKind = "tool_call_delta"
	EventKindToolCallDone    EventKind = "tool_call_done"
	EventKindItemDone        EventKind = "item_done"
	EventKindCompleted       EventKind = "completed"
)

// Usage contains token counts.
type Usage struct {
	InputTokens     int `json:"input_tokens"`
	OutputTokens    int `json:"output_tokens"`
	TotalTokens     int `json:"total_tokens"`
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// NormalizedEvent is a unified representation of translated events.
type NormalizedEvent struct {
	Kind         EventKind `json:"kind"`
	ResponseID   string    `json:"response_id,omitempty"`
	Model        string    `json:"model,omitempty"`
	ItemID       string    `json:"item_id,omitempty"`
	OutputIndex  int       `json:"output_index"`
	ToolIndex    int       `json:"tool_index,omitempty"`
	CallID       string    `json:"call_id,omitempty"`
	ToolName     string    `json:"tool_name,omitempty"`
	Delta        string    `json:"delta,omitempty"`
	Arguments    string    `json:"arguments,omitempty"`
	Usage        Usage     `json:"usage,omitempty"`
	ContentIndex int       `json:"content_index,omitempty"`
}

// ItemState tracks metadata for a single item in the response.
type ItemState struct {
	ID          string
	OutputIndex int
	Type        string
	Role        string
	PartTypes   map[int]string // content_index -> part type ("output_text", "reasoning", etc.)
	CallID      string
	ToolName    string
}

// Translator translates OpenAI Responses API item events into normalized events.
type Translator struct {
	// Mappings maintained per specification
	itemIDToOutputIndex map[string]int
	callIDToToolIndex   map[string]int
	toolArguments       map[string]*strings.Builder
	items               map[string]*ItemState
	nextToolIndex       int

	// State
	ResponseID string
	Model      string
	FinalUsage Usage

	// Accumulated normalized events
	events []NormalizedEvent
}

// NewTranslator creates a new Responses API Translator.
func NewTranslator() *Translator {
	return &Translator{
		itemIDToOutputIndex: make(map[string]int),
		callIDToToolIndex:   make(map[string]int),
		toolArguments:       make(map[string]*strings.Builder),
		items:               make(map[string]*ItemState),
		events:              make([]NormalizedEvent, 0),
	}
}

// Events returns the slice of accumulated normalized events.
func (t *Translator) Events() []NormalizedEvent {
	return t.events
}

// AccumulatedArguments returns the accumulated argument string for a tool call_id.
func (t *Translator) AccumulatedArguments(callID string) string {
	if sb, ok := t.toolArguments[callID]; ok {
		return sb.String()
	}
	return ""
}

// ToolCallIndex returns the assigned tool call index for a call_id.
func (t *Translator) ToolCallIndex(callID string) (int, bool) {
	idx, ok := t.callIDToToolIndex[callID]
	return idx, ok
}

// OutputIndex returns the output index for an item ID.
func (t *Translator) OutputIndex(itemID string) (int, bool) {
	idx, ok := t.itemIDToOutputIndex[itemID]
	return idx, ok
}

type rawResponsesEvent struct {
	Type         string `json:"type"`
	OutputIndex  int    `json:"output_index"`
	ItemID       string `json:"item_id"`
	ContentIndex int    `json:"content_index"`
	CallID       string `json:"call_id"`
	Delta        string `json:"delta"`

	Response struct {
		ID     string `json:"id"`
		Model  string `json:"model"`
		Status string `json:"status"`
		Usage  Usage  `json:"usage"`
	} `json:"response"`

	Item struct {
		ID        string `json:"id"`
		Type      string `json:"type"`
		Role      string `json:"role"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"item"`

	Part struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"part"`
}

// Translate processes a single OpenAI Responses API event payload.
func (t *Translator) Translate(eventType string, data []byte) ([]NormalizedEvent, error) {
	if len(data) == 0 {
		return nil, nil
	}

	var raw rawResponsesEvent
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to unmarshal responses event: %w", err)
	}

	effectiveType := eventType
	if effectiveType == "" {
		effectiveType = raw.Type
	}

	var emitted []NormalizedEvent

	switch effectiveType {
	case "response.created":
		t.ResponseID = raw.Response.ID
		t.Model = raw.Response.Model
		evt := NormalizedEvent{
			Kind:        EventKindResponseCreated,
			ResponseID:  raw.Response.ID,
			Model:       raw.Response.Model,
			OutputIndex: -1,
		}
		t.events = append(t.events, evt)
		emitted = append(emitted, evt)

	case "response.output_item.added":
		itemID := raw.Item.ID
		t.itemIDToOutputIndex[itemID] = raw.OutputIndex

		itemState := &ItemState{
			ID:          itemID,
			OutputIndex: raw.OutputIndex,
			Type:        raw.Item.Type,
			Role:        raw.Item.Role,
			PartTypes:   make(map[int]string),
			CallID:      raw.Item.CallID,
			ToolName:    raw.Item.Name,
		}
		t.items[itemID] = itemState

		if raw.Item.Type == "function_call" {
			callID := raw.Item.CallID
			toolIdx := t.nextToolIndex
			t.nextToolIndex++
			t.callIDToToolIndex[callID] = toolIdx
			t.toolArguments[callID] = &strings.Builder{}

			evt := NormalizedEvent{
				Kind:        EventKindToolCallStart,
				ItemID:      itemID,
				OutputIndex: raw.OutputIndex,
				ToolIndex:   toolIdx,
				CallID:      callID,
				ToolName:    raw.Item.Name,
			}
			t.events = append(t.events, evt)
			emitted = append(emitted, evt)
		} else {
			evt := NormalizedEvent{
				Kind:        EventKindItemAdded,
				ItemID:      itemID,
				OutputIndex: raw.OutputIndex,
			}
			t.events = append(t.events, evt)
			emitted = append(emitted, evt)
		}

	case "response.content_part.added":
		itemID := raw.ItemID
		if is, ok := t.items[itemID]; ok {
			is.PartTypes[raw.ContentIndex] = raw.Part.Type
		}

	case "response.output_text.delta", "response.reasoning.delta", "response.reasoning_text.delta":
		itemID := raw.ItemID
		outIdx := raw.OutputIndex
		if outIdx == 0 && itemID != "" {
			if idx, ok := t.itemIDToOutputIndex[itemID]; ok {
				outIdx = idx
			}
		}

		isReasoning := effectiveType == "response.reasoning.delta" || effectiveType == "response.reasoning_text.delta"
		if !isReasoning {
			if is, ok := t.items[itemID]; ok {
				if is.Type == "reasoning" || is.PartTypes[raw.ContentIndex] == "reasoning" {
					isReasoning = true
				}
			}
		}

		kind := EventKindTextDelta
		if isReasoning {
			kind = EventKindReasoningDelta
		}

		evt := NormalizedEvent{
			Kind:         kind,
			ItemID:       itemID,
			OutputIndex:  outIdx,
			ContentIndex: raw.ContentIndex,
			Delta:        raw.Delta,
		}
		t.events = append(t.events, evt)
		emitted = append(emitted, evt)

	case "response.function_call_arguments.delta":
		callID := raw.CallID
		toolIdx, exists := t.callIDToToolIndex[callID]
		if !exists {
			toolIdx = t.nextToolIndex
			t.nextToolIndex++
			t.callIDToToolIndex[callID] = toolIdx
		}

		sb, sbExists := t.toolArguments[callID]
		if !sbExists {
			sb = &strings.Builder{}
			t.toolArguments[callID] = sb
		}
		sb.WriteString(raw.Delta)

		outIdx := raw.OutputIndex
		if raw.ItemID != "" {
			if idx, ok := t.itemIDToOutputIndex[raw.ItemID]; ok {
				outIdx = idx
			}
		}

		evt := NormalizedEvent{
			Kind:        EventKindToolCallDelta,
			ItemID:      raw.ItemID,
			OutputIndex: outIdx,
			ToolIndex:   toolIdx,
			CallID:      callID,
			Delta:       raw.Delta,
			Arguments:   sb.String(),
		}
		t.events = append(t.events, evt)
		emitted = append(emitted, evt)

	case "response.output_item.done":
		itemID := raw.Item.ID
		if itemID == "" {
			itemID = raw.ItemID
		}
		outIdx := raw.OutputIndex
		if itemID != "" {
			if idx, ok := t.itemIDToOutputIndex[itemID]; ok {
				outIdx = idx
			}
		}

		if raw.Item.Type == "function_call" || (t.items[itemID] != nil && t.items[itemID].Type == "function_call") {
			callID := raw.Item.CallID
			if callID == "" && t.items[itemID] != nil {
				callID = t.items[itemID].CallID
			}
			toolIdx := t.callIDToToolIndex[callID]
			finalArgs := t.AccumulatedArguments(callID)
			if raw.Item.Arguments != "" {
				finalArgs = raw.Item.Arguments
			}

			evt := NormalizedEvent{
				Kind:        EventKindToolCallDone,
				ItemID:      itemID,
				OutputIndex: outIdx,
				ToolIndex:   toolIdx,
				CallID:      callID,
				ToolName:    raw.Item.Name,
				Arguments:   finalArgs,
			}
			t.events = append(t.events, evt)
			emitted = append(emitted, evt)
		} else {
			evt := NormalizedEvent{
				Kind:        EventKindItemDone,
				ItemID:      itemID,
				OutputIndex: outIdx,
			}
			t.events = append(t.events, evt)
			emitted = append(emitted, evt)
		}

	case "response.completed":
		t.FinalUsage = raw.Response.Usage
		evt := NormalizedEvent{
			Kind:        EventKindCompleted,
			ResponseID:  raw.Response.ID,
			Model:       raw.Response.Model,
			OutputIndex: -1,
			Usage:       raw.Response.Usage,
		}
		t.events = append(t.events, evt)
		emitted = append(emitted, evt)
	}

	return emitted, nil
}

// TranslateStream consumes an entire SSE reader and translates all events.
func (t *Translator) TranslateStream(r io.Reader) ([]NormalizedEvent, error) {
	scanner := bufio.NewScanner(r)
	var currentEvent string
	var currentData strings.Builder
	var allEmitted []NormalizedEvent

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			if currentData.Len() > 0 || currentEvent != "" {
				emitted, err := t.Translate(currentEvent, []byte(currentData.String()))
				if err != nil {
					return nil, err
				}
				allEmitted = append(allEmitted, emitted...)
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

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanner error in responses stream: %w", err)
	}

	if currentData.Len() > 0 || currentEvent != "" {
		emitted, err := t.Translate(currentEvent, []byte(currentData.String()))
		if err != nil {
			return nil, err
		}
		allEmitted = append(allEmitted, emitted...)
	}

	return allEmitted, nil
}
