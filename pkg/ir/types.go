package ir

import (
	"encoding/json"
	"fmt"
)

// BlockKind identifies the variant of a content block.
type BlockKind string

const (
	BlockKindText         BlockKind = "text"
	BlockKindImage        BlockKind = "image"
	BlockKindToolCall     BlockKind = "tool_call"
	BlockKindToolResult   BlockKind = "tool_result"
	BlockKindReasoning    BlockKind = "reasoning"
	BlockKindCacheControl BlockKind = "cache_control"
)

// Block is the common interface implemented by all IR content blocks.
type Block interface {
	Kind() BlockKind
}

// TextBlock represents plain text content.
type TextBlock struct {
	Text string `json:"text"`
}

func (b TextBlock) Kind() BlockKind { return BlockKindText }

// ImageBlock represents an image URL and optional detail level.
type ImageBlock struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

func (b ImageBlock) Kind() BlockKind { return BlockKindImage }

// ToolCallBlock represents a tool/function call requested by the model.
type ToolCallBlock struct {
	Index     int    `json:"index,omitempty"`
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func (b ToolCallBlock) Kind() BlockKind { return BlockKindToolCall }

// ToolResultBlock represents the output returned from executing a tool call.
type ToolResultBlock struct {
	ToolCallID string `json:"tool_call_id"`
	Name       string `json:"name,omitempty"`
	Content    string `json:"content"`
}

func (b ToolResultBlock) Kind() BlockKind { return BlockKindToolResult }

// ReasoningKind indicates whether reasoning is text, summary, opaque, or redacted.
type ReasoningKind string

const (
	ReasoningText     ReasoningKind = "text"
	ReasoningSummary  ReasoningKind = "summary"
	ReasoningOpaque   ReasoningKind = "opaque"
	ReasoningRedacted ReasoningKind = "redacted"
)

// ReasoningBlock represents extended thinking/reasoning data.
type ReasoningBlock struct {
	ReasoningKind ReasoningKind   `json:"reasoning_kind,omitempty"`
	Text          string          `json:"text,omitempty"`
	Signature     string          `json:"signature,omitempty"`
	Opaque        json.RawMessage `json:"opaque,omitempty"`
}

func (b ReasoningBlock) Kind() BlockKind { return BlockKindReasoning }

// CacheControl marks a prompt caching breakpoint.
type CacheControl struct {
	Breakpoint bool `json:"breakpoint"`
}

func (b CacheControl) Kind() BlockKind { return BlockKindCacheControl }

// Message represents a single message in a conversation.
type Message struct {
	Role   string            `json:"role"`
	Blocks []Block           `json:"blocks"`
	Meta   map[string]string `json:"meta,omitempty"`
}

// rawBlockEnvelope wraps a Block for polymorphic JSON serialization.
type rawBlockEnvelope struct {
	BlockKind  BlockKind       `json:"block_kind"`
	Text       string          `json:"text,omitempty"`
	URL        string          `json:"url,omitempty"`
	Detail     string          `json:"detail,omitempty"`
	Index      int             `json:"index,omitempty"`
	ID         string          `json:"id,omitempty"`
	Name       string          `json:"name,omitempty"`
	Arguments  string          `json:"arguments,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Content    string          `json:"content,omitempty"`
	ReasonKind ReasoningKind   `json:"reasoning_kind,omitempty"`
	Signature  string          `json:"signature,omitempty"`
	Opaque     json.RawMessage `json:"opaque,omitempty"`
	Breakpoint bool            `json:"breakpoint,omitempty"`
}

// MarshalJSON implements custom JSON serialization for Message.
func (m Message) MarshalJSON() ([]byte, error) {
	aux := struct {
		Role   string             `json:"role"`
		Blocks []rawBlockEnvelope `json:"blocks"`
		Meta   map[string]string  `json:"meta,omitempty"`
	}{
		Role:   m.Role,
		Blocks: make([]rawBlockEnvelope, 0, len(m.Blocks)),
		Meta:   m.Meta,
	}

	for _, blk := range m.Blocks {
		if blk == nil {
			continue
		}
		env := rawBlockEnvelope{BlockKind: blk.Kind()}
		switch b := blk.(type) {
		case TextBlock:
			env.Text = b.Text
		case *TextBlock:
			env.Text = b.Text
		case ImageBlock:
			env.URL = b.URL
			env.Detail = b.Detail
		case *ImageBlock:
			env.URL = b.URL
			env.Detail = b.Detail
		case ToolCallBlock:
			env.Index = b.Index
			env.ID = b.ID
			env.Name = b.Name
			env.Arguments = b.Arguments
		case *ToolCallBlock:
			env.Index = b.Index
			env.ID = b.ID
			env.Name = b.Name
			env.Arguments = b.Arguments
		case ToolResultBlock:
			env.ToolCallID = b.ToolCallID
			env.Name = b.Name
			env.Content = b.Content
		case *ToolResultBlock:
			env.ToolCallID = b.ToolCallID
			env.Name = b.Name
			env.Content = b.Content
		case ReasoningBlock:
			env.ReasonKind = b.ReasoningKind
			env.Text = b.Text
			env.Signature = b.Signature
			env.Opaque = b.Opaque
		case *ReasoningBlock:
			env.ReasonKind = b.ReasoningKind
			env.Text = b.Text
			env.Signature = b.Signature
			env.Opaque = b.Opaque
		case CacheControl:
			env.Breakpoint = b.Breakpoint
		case *CacheControl:
			env.Breakpoint = b.Breakpoint
		default:
			return nil, fmt.Errorf("unsupported block type: %T", blk)
		}
		aux.Blocks = append(aux.Blocks, env)
	}

	return json.Marshal(aux)
}

// UnmarshalJSON implements custom JSON deserialization for Message.
func (m *Message) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role   string             `json:"role"`
		Blocks []rawBlockEnvelope `json:"blocks"`
		Meta   map[string]string  `json:"meta,omitempty"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	m.Role = raw.Role
	m.Meta = raw.Meta
	m.Blocks = make([]Block, 0, len(raw.Blocks))

	for _, env := range raw.Blocks {
		switch env.BlockKind {
		case BlockKindText:
			m.Blocks = append(m.Blocks, TextBlock{Text: env.Text})
		case BlockKindImage:
			m.Blocks = append(m.Blocks, ImageBlock{URL: env.URL, Detail: env.Detail})
		case BlockKindToolCall:
			m.Blocks = append(m.Blocks, ToolCallBlock{
				Index:     env.Index,
				ID:        env.ID,
				Name:      env.Name,
				Arguments: env.Arguments,
			})
		case BlockKindToolResult:
			m.Blocks = append(m.Blocks, ToolResultBlock{
				ToolCallID: env.ToolCallID,
				Name:       env.Name,
				Content:    env.Content,
			})
		case BlockKindReasoning:
			m.Blocks = append(m.Blocks, ReasoningBlock{
				ReasoningKind: env.ReasonKind,
				Text:          env.Text,
				Signature:     env.Signature,
				Opaque:        env.Opaque,
			})
		case BlockKindCacheControl:
			m.Blocks = append(m.Blocks, CacheControl{Breakpoint: env.Breakpoint})
		default:
			return fmt.Errorf("unknown block kind: %q", env.BlockKind)
		}
	}

	return nil
}
