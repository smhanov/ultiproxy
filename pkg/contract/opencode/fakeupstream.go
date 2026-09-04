package opencode

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"time"
)

// RecordedRequest captures details of an HTTP request received by FakeUpstream.
type RecordedRequest struct {
	Method      string
	Path        string
	URL         string
	Header      http.Header
	Headers     http.Header
	Body        []byte
	JSON        map[string]any
	Model       string
	Stream      bool
	Messages    []RecordedMessage
	Tools       []RecordedTool
	ToolChoice  any
	MaxTokens   int
	Temperature *float64
}

// BodyString returns the raw request body as a string.
func (r *RecordedRequest) BodyString() string {
	return string(r.Body)
}

// GetHeader returns the first value associated with the given key, case-insensitive.
func (r *RecordedRequest) GetHeader(key string) string {
	if r.Header == nil {
		return ""
	}
	return r.Header.Get(key)
}

// HasHeader reports whether the header key exists and is non-empty.
func (r *RecordedRequest) HasHeader(key string) bool {
	return r.GetHeader(key) != ""
}

// HasTool reports whether a tool with the given function name was provided.
func (r *RecordedRequest) HasTool(name string) bool {
	return r.GetTool(name) != nil
}

// GetTool returns the tool with the given function name, or nil if not found.
func (r *RecordedRequest) GetTool(name string) *RecordedTool {
	for i := range r.Tools {
		if r.Tools[i].Function.Name == name {
			return &r.Tools[i]
		}
	}
	return nil
}

// LastMessage returns the last message in the recorded request, or nil if none.
func (r *RecordedRequest) LastMessage() *RecordedMessage {
	if len(r.Messages) == 0 {
		return nil
	}
	return &r.Messages[len(r.Messages)-1]
}

// UserPrompt returns the text content of the last user-role message.
func (r *RecordedRequest) UserPrompt() string {
	for i := len(r.Messages) - 1; i >= 0; i-- {
		if r.Messages[i].Role == "user" {
			return r.Messages[i].ContentString()
		}
	}
	return ""
}

// RecordedMessage represents a message within a recorded request.
type RecordedMessage struct {
	Role             string             `json:"role"`
	Content          any                `json:"content"`
	Name             string             `json:"name,omitempty"`
	ToolCallID       string             `json:"tool_call_id,omitempty"`
	ToolCalls        []RecordedToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string             `json:"reasoning_content,omitempty"`
}

// ContentString returns Content as a plain string. If Content is a slice of
// multipart blocks, it concatenates the text blocks.
func (m RecordedMessage) ContentString() string {
	switch c := m.Content.(type) {
	case string:
		return c
	case []any:
		var sb strings.Builder
		for _, part := range c {
			if partMap, ok := part.(map[string]any); ok {
				if t, ok := partMap["type"].(string); ok && t == "text" {
					if txt, ok := partMap["text"].(string); ok {
						sb.WriteString(txt)
					}
				}
			}
		}
		return sb.String()
	default:
		return ""
	}
}

// RecordedToolCall represents a tool call requested in a message.
type RecordedToolCall struct {
	Index    int                  `json:"index,omitempty"`
	ID       string               `json:"id,omitempty"`
	Type     string               `json:"type,omitempty"`
	Function RecordedFunctionCall `json:"function"`
}

// RecordedFunctionCall holds the function name and raw arguments.
type RecordedFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// RecordedTool describes a tool defined in the request.
type RecordedTool struct {
	Type     string               `json:"type"`
	Function RecordedFunctionDef `json:"function"`
}

// RecordedFunctionDef describes the function schema.
type RecordedFunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ResponseScript defines how a scripted response serves an incoming request.
type ResponseScript interface {
	ServeHTTP(w http.ResponseWriter, r *http.Request)
}

// ResponseScriptFunc is an adapter to allow the use of ordinary functions as ResponseScripts.
type ResponseScriptFunc func(w http.ResponseWriter, r *http.Request)

// ServeHTTP calls f(w, r).
func (f ResponseScriptFunc) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f(w, r)
}

// ToolCallChunk represents a tool call delta within an SSE chunk.
type ToolCallChunk struct {
	Index    int                    `json:"index"`
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function ToolCallFunctionChunk  `json:"function"`
}

// ToolCallFunctionChunk represents the function delta within a tool call chunk.
type ToolCallFunctionChunk struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// UsageChunk represents token usage metrics within an SSE chunk or response.
type UsageChunk struct {
	PromptTokens             int `json:"prompt_tokens"`
	CompletionTokens         int `json:"completion_tokens"`
	TotalTokens              int `json:"total_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// ErrorChunk represents an error payload within an SSE stream or response.
type ErrorChunk struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    any    `json:"code,omitempty"`
}

// SSEChunk represents a single Server-Sent Event chunk scripted for FakeUpstream.
type SSEChunk struct {
	ID               string
	Model            string
	Role             string
	Content          string
	ReasoningContent string
	ToolCalls        []ToolCallChunk
	FinishReason     string
	Usage            *UsageChunk
	Error            *ErrorChunk
	RawData          string
	Event            string
	Delay            time.Duration
}

// SSEStream is a builder for scripting a sequence of SSE chunks.
type SSEStream struct {
	chunks      []SSEChunk
	includeDone bool
	id          string
	model       string
}

// NewSSEStream creates a new SSEStream builder.
func NewSSEStream() *SSEStream {
	return &SSEStream{
		includeDone: true,
		id:          "chatcmpl-test",
		model:       "deepseek-v4-flash",
	}
}

// WithID sets the default completion chunk ID for the stream.
func (s *SSEStream) WithID(id string) *SSEStream {
	s.id = id
	return s
}

// WithModel sets the model name for the stream chunks.
func (s *SSEStream) WithModel(model string) *SSEStream {
	s.model = model
	return s
}

// WithDone configures whether to append `data: [DONE]\n\n` at the end of the stream.
func (s *SSEStream) WithDone(done bool) *SSEStream {
	s.includeDone = done
	return s
}

// TextDelta adds a chunk with text content delta.
func (s *SSEStream) TextDelta(text string) *SSEStream {
	s.chunks = append(s.chunks, SSEChunk{
		ID:      s.id,
		Model:   s.model,
		Content: text,
	})
	return s
}

// ReasoningDelta adds a chunk with reasoning content delta.
func (s *SSEStream) ReasoningDelta(text string) *SSEStream {
	s.chunks = append(s.chunks, SSEChunk{
		ID:               s.id,
		Model:            s.model,
		ReasoningContent: text,
	})
	return s
}

// ToolCall adds a tool call chunk.
func (s *SSEStream) ToolCall(index int, id, name, args string) *SSEStream {
	t := "function"
	if id == "" && name == "" {
		t = ""
	}
	s.chunks = append(s.chunks, SSEChunk{
		ID:    s.id,
		Model: s.model,
		ToolCalls: []ToolCallChunk{
			{
				Index: index,
				ID:    id,
				Type:  t,
				Function: ToolCallFunctionChunk{
					Name:      name,
					Arguments: args,
				},
			},
		},
	})
	return s
}

// ToolCallStart adds a tool call start chunk with index, ID, and function name.
func (s *SSEStream) ToolCallStart(index int, id, name string) *SSEStream {
	return s.ToolCall(index, id, name, "")
}

// ToolCallArgs adds a tool call arguments delta chunk for the given index.
func (s *SSEStream) ToolCallArgs(index int, args string) *SSEStream {
	return s.ToolCall(index, "", "", args)
}

// FinishReason adds a chunk specifying the finish reason (e.g., "stop", "tool_calls").
func (s *SSEStream) FinishReason(reason string) *SSEStream {
	s.chunks = append(s.chunks, SSEChunk{
		ID:           s.id,
		Model:        s.model,
		FinishReason: reason,
	})
	return s
}

// Usage adds a chunk reporting token usage metrics.
func (s *SSEStream) Usage(prompt, completion, total int) *SSEStream {
	s.chunks = append(s.chunks, SSEChunk{
		ID:    s.id,
		Model: s.model,
		Usage: &UsageChunk{
			PromptTokens:     prompt,
			CompletionTokens: completion,
			TotalTokens:      total,
		},
	})
	return s
}

// Error adds a chunk reporting an error.
func (s *SSEStream) Error(msg string) *SSEStream {
	s.chunks = append(s.chunks, SSEChunk{
		Error: &ErrorChunk{Message: msg, Type: "upstream_error"},
	})
	return s
}

// Raw adds a chunk with raw SSE event and data lines.
func (s *SSEStream) Raw(event, data string) *SSEStream {
	s.chunks = append(s.chunks, SSEChunk{
		Event:   event,
		RawData: data,
	})
	return s
}

// Chunks returns the slice of scripted chunks.
func (s *SSEStream) Chunks() []SSEChunk {
	return s.chunks
}

// FakeUpstream is a mock HTTP server that simulates upstream providers
// (OpenCode, OpenAI-compatible, and Anthropic). It records incoming requests
// and serves scripted responses.
type FakeUpstream struct {
	server      *httptest.Server
	mu          sync.Mutex
	requests    []*RecordedRequest
	responses   []ResponseScript
	handler     http.HandlerFunc
	defaultResp ResponseScript
}

// MockUpstream is an alias for FakeUpstream representing the mock HTTP test server.
type MockUpstream = FakeUpstream

// NewMockUpstream creates and starts a new MockUpstream mock HTTP server.
func NewMockUpstream() *MockUpstream {
	return NewFakeUpstream()
}

// NewFakeUpstream creates and starts a new FakeUpstream mock HTTP server.
func NewFakeUpstream() *FakeUpstream {
	f := &FakeUpstream{}
	f.server = httptest.NewServer(http.HandlerFunc(f.serveHTTP))
	return f
}

// URL returns the base URL of the mock HTTP server (e.g., "http://127.0.0.1:45678").
func (f *FakeUpstream) URL() string {
	return f.server.URL
}

// Client returns an http.Client configured to communicate with the mock server.
func (f *FakeUpstream) Client() *http.Client {
	return f.server.Client()
}

// Close stops the mock HTTP server.
func (f *FakeUpstream) Close() {
	if f.server != nil {
		f.server.Close()
	}
}

// Requests returns a copy of all recorded incoming requests.
func (f *FakeUpstream) Requests() []*RecordedRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]*RecordedRequest, len(f.requests))
	copy(out, f.requests)
	return out
}

// LastRequest returns the most recently recorded request, or nil if none.
func (f *FakeUpstream) LastRequest() *RecordedRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) == 0 {
		return nil
	}
	return f.requests[len(f.requests)-1]
}

// RequestCount returns the total number of recorded requests.
func (f *FakeUpstream) RequestCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

// ResetRequests clears all recorded requests.
func (f *FakeUpstream) ResetRequests() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = nil
}

// Reset clears recorded requests, queued responses, and custom handlers.
func (f *FakeUpstream) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = nil
	f.responses = nil
	f.handler = nil
	f.defaultResp = nil
}

// QueueResponse adds a scripted response to the FIFO response queue.
func (f *FakeUpstream) QueueResponse(resp ResponseScript) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.responses = append(f.responses, resp)
}

// QueueResponseFunc adds an http.HandlerFunc as a scripted response to the FIFO queue.
func (f *FakeUpstream) QueueResponseFunc(fn func(w http.ResponseWriter, r *http.Request)) {
	f.QueueResponse(ResponseScriptFunc(fn))
}

// SetHandler sets a fallback handler function when the response queue is empty.
func (f *FakeUpstream) SetHandler(h http.HandlerFunc) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.handler = h
}

// SetDefaultResponse sets a default scripted response when the queue is empty.
func (f *FakeUpstream) SetDefaultResponse(resp ResponseScript) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.defaultResp = resp
}

// QueueJSON queues a non-streaming JSON response with the given status code and payload.
func (f *FakeUpstream) QueueJSON(statusCode int, body any) {
	f.QueueResponseFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		switch b := body.(type) {
		case []byte:
			_, _ = w.Write(b)
		case string:
			_, _ = w.Write([]byte(b))
		default:
			_ = json.NewEncoder(w).Encode(body)
		}
	})
}

// QueueChatCompletion queues a standard 200 OK OpenAI ChatCompletionResponse with the given text content.
func (f *FakeUpstream) QueueChatCompletion(content string) {
	f.QueueResponseFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      fmt.Sprintf("chatcmpl-mock-%d", time.Now().UnixNano()),
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "deepseek-v4-flash",
			"choices": []any{
				map[string]any{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": content,
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 10,
				"total_tokens":      20,
			},
		})
	})
}

// QueueChatCompletionWithTools queues a standard OpenAI ChatCompletionResponse containing tool calls.
func (f *FakeUpstream) QueueChatCompletionWithTools(toolCalls ...RecordedToolCall) {
	f.QueueResponseFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		var tcs []any
		for _, tc := range toolCalls {
			tcs = append(tcs, map[string]any{
				"id":   tc.ID,
				"type": "function",
				"function": map[string]any{
					"name":      tc.Function.Name,
					"arguments": tc.Function.Arguments,
				},
			})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      fmt.Sprintf("chatcmpl-mock-%d", time.Now().UnixNano()),
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "deepseek-v4-flash",
			"choices": []any{
				map[string]any{
					"index": 0,
					"message": map[string]any{
						"role":       "assistant",
						"tool_calls": tcs,
					},
					"finish_reason": "tool_calls",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     15,
				"completion_tokens": 20,
				"total_tokens":      35,
			},
		})
	})
}

// QueueError queues an HTTP error response formatted as an OpenAI error payload.
func (f *FakeUpstream) QueueError(statusCode int, message string) {
	f.QueueResponseFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": message,
				"type":    "upstream_error",
				"code":    statusCode,
			},
		})
	})
}

// QueueWorkspaceHTML queues an OpenCode workspace HTML response with rolling, weekly, and monthly quotas.
func (f *FakeUpstream) QueueWorkspaceHTML(rollingPct, weeklyPct, monthlyPct float64) {
	html := fmt.Sprintf(`<html><body><script>
window.__INITIAL_STATE__ = {
  rollingUsage: { usagePercent: %.1f, resetInSec: 3600 },
  weeklyUsage: { usagePercent: %.1f, resetInSec: 86400 },
  monthlyUsage: { usagePercent: %.1f, resetInSec: 604800 }
};
</script></body></html>`, rollingPct, weeklyPct, monthlyPct)
	f.QueueResponseFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(html))
	})
}

// QueueSSE queues an SSE stream of chunks, terminating with `data: [DONE]\n\n`.
func (f *FakeUpstream) QueueSSE(chunks ...SSEChunk) {
	f.QueueSSEWithDone(true, chunks...)
}

// QueueSSEChunks is an alias for QueueSSE.
func (f *FakeUpstream) QueueSSEChunks(chunks ...SSEChunk) {
	f.QueueSSE(chunks...)
}

// QueueSSEArray queues an array (slice) of SSE chunks for streaming.
func (f *FakeUpstream) QueueSSEArray(chunks []SSEChunk) {
	f.QueueSSEWithDone(true, chunks...)
}

// QueueSSEStream queues the sequence of chunks defined in an SSEStream builder.
func (f *FakeUpstream) QueueSSEStream(stream *SSEStream) {
	f.QueueSSEWithDone(stream.includeDone, stream.chunks...)
}

// QueueSSEWithDone queues an SSE stream of chunks with optional [DONE] trailer.
func (f *FakeUpstream) QueueSSEWithDone(includeDone bool, chunks ...SSEChunk) {
	f.QueueResponseFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		flusher, isFlusher := w.(http.Flusher)
		if isFlusher {
			flusher.Flush()
		}

		defaultID := fmt.Sprintf("chatcmpl-stream-%d", time.Now().UnixNano())
		defaultModel := "deepseek-v4-flash"

		for _, chunk := range chunks {
			if chunk.Delay > 0 {
				time.Sleep(chunk.Delay)
			}

			if chunk.RawData != "" {
				if chunk.Event != "" {
					_, _ = fmt.Fprintf(w, "event: %s\n", chunk.Event)
				}
				_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk.RawData)
			} else if chunk.Error != nil {
				errPayload := map[string]any{"error": chunk.Error}
				data, _ := json.Marshal(errPayload)
				if chunk.Event != "" {
					_, _ = fmt.Fprintf(w, "event: %s\n", chunk.Event)
				}
				_, _ = fmt.Fprintf(w, "data: %s\n\n", string(data))
			} else {
				chunkID := chunk.ID
				if chunkID == "" {
					chunkID = defaultID
				}
				model := chunk.Model
				if model == "" {
					model = defaultModel
				}

				delta := map[string]any{}
				if chunk.Role != "" {
					delta["role"] = chunk.Role
				}
				if chunk.Content != "" {
					delta["content"] = chunk.Content
				}
				if chunk.ReasoningContent != "" {
					delta["reasoning_content"] = chunk.ReasoningContent
				}
				if len(chunk.ToolCalls) > 0 {
					var tcs []any
					for _, tc := range chunk.ToolCalls {
						tcMap := map[string]any{
							"index": tc.Index,
						}
						if tc.ID != "" {
							tcMap["id"] = tc.ID
						}
						if tc.Type != "" {
							tcMap["type"] = tc.Type
						} else if tc.ID != "" || tc.Function.Name != "" {
							tcMap["type"] = "function"
						}
						fnMap := map[string]any{}
						if tc.Function.Name != "" {
							fnMap["name"] = tc.Function.Name
						}
						if tc.Function.Arguments != "" {
							fnMap["arguments"] = tc.Function.Arguments
						}
						tcMap["function"] = fnMap
						tcs = append(tcs, tcMap)
					}
					delta["tool_calls"] = tcs
				}

				var choices []any
				if len(delta) > 0 || chunk.FinishReason != "" || chunk.Usage == nil {
					choice := map[string]any{
						"index": 0,
						"delta": delta,
					}
					if chunk.FinishReason != "" {
						choice["finish_reason"] = chunk.FinishReason
					} else {
						choice["finish_reason"] = nil
					}
					choices = append(choices, choice)
				}

				payload := map[string]any{
					"id":      chunkID,
					"object":  "chat.completion.chunk",
					"created": time.Now().Unix(),
					"model":   model,
					"choices": choices,
				}
				if chunk.Usage != nil {
					payload["usage"] = chunk.Usage
				}

				data, _ := json.Marshal(payload)
				if chunk.Event != "" {
					_, _ = fmt.Fprintf(w, "event: %s\n", chunk.Event)
				}
				_, _ = fmt.Fprintf(w, "data: %s\n\n", string(data))
			}

			if isFlusher {
				flusher.Flush()
			}
		}

		if includeDone {
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			if isFlusher {
				flusher.Flush()
			}
		}
	})
}

func (f *FakeUpstream) serveHTTP(w http.ResponseWriter, r *http.Request) {
	rec, err := parseRecordedRequest(r)
	if err != nil {
		http.Error(w, fmt.Sprintf("fakeupstream: failed to record request: %v", err), http.StatusBadRequest)
		return
	}

	f.mu.Lock()
	f.requests = append(f.requests, rec)

	var resp ResponseScript
	if len(f.responses) > 0 {
		resp = f.responses[0]
		f.responses = f.responses[1:]
	} else if f.handler != nil {
		handler := f.handler
		f.mu.Unlock()
		handler(w, r)
		return
	} else if f.defaultResp != nil {
		resp = f.defaultResp
	}
	f.mu.Unlock()

	if resp != nil {
		resp.ServeHTTP(w, r)
		return
	}

	if rec.Stream {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		chunkID := fmt.Sprintf("chatcmpl-fake-%d", time.Now().UnixNano())
		chunkModel := rec.Model
		if chunkModel == "" {
			chunkModel = "deepseek-v4-flash"
		}
		c1, _ := json.Marshal(map[string]any{
			"id":      chunkID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   chunkModel,
			"choices": []any{
				map[string]any{
					"index": 0,
					"delta": map[string]any{
						"role":    "assistant",
						"content": "default fake response",
					},
					"finish_reason": nil,
				},
			},
		})
		c2, _ := json.Marshal(map[string]any{
			"id":      chunkID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   chunkModel,
			"choices": []any{
				map[string]any{
					"index":         0,
					"delta":         map[string]any{},
					"finish_reason": "stop",
				},
			},
		})
		_, _ = fmt.Fprintf(w, "data: %s\n\n", string(c1))
		_, _ = fmt.Fprintf(w, "data: %s\n\n", string(c2))
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		return
	}

	// Default response when queue is empty: return 200 OK OpenAI completion
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      fmt.Sprintf("chatcmpl-fake-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   rec.Model,
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "default fake response",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     1,
			"completion_tokens": 1,
			"total_tokens":      2,
		},
	})
}

func parseRecordedRequest(r *http.Request) (*RecordedRequest, error) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	rec := &RecordedRequest{
		Method:  r.Method,
		Path:    r.URL.Path,
		URL:     r.URL.String(),
		Header:  r.Header.Clone(),
		Headers: r.Header.Clone(),
		Body:    bodyBytes,
	}

	if len(bodyBytes) == 0 {
		return rec, nil
	}

	var rawMap map[string]any
	if err := json.Unmarshal(bodyBytes, &rawMap); err != nil {
		return rec, nil
	}
	rec.JSON = rawMap

	if m, ok := rawMap["model"].(string); ok {
		rec.Model = m
	}
	if s, ok := rawMap["stream"].(bool); ok {
		rec.Stream = s
	}
	if mt, ok := rawMap["max_tokens"].(float64); ok {
		rec.MaxTokens = int(mt)
	} else if mt, ok := rawMap["max_tokens"].(int); ok {
		rec.MaxTokens = mt
	} else if mt, ok := rawMap["max_completion_tokens"].(float64); ok {
		rec.MaxTokens = int(mt)
	}
	if t, ok := rawMap["temperature"].(float64); ok {
		rec.Temperature = &t
	}
	if tc, ok := rawMap["tool_choice"]; ok {
		rec.ToolChoice = tc
	}

	// Parse tools (OpenAI or Anthropic style)
	if toolsRaw, ok := rawMap["tools"].([]any); ok {
		for _, item := range toolsRaw {
			itemMap, ok := item.(map[string]any)
			if !ok {
				continue
			}
			var tool RecordedTool
			if fnMap, ok := itemMap["function"].(map[string]any); ok {
				// OpenAI format
				tool.Type, _ = itemMap["type"].(string)
				tool.Function.Name, _ = fnMap["name"].(string)
				tool.Function.Description, _ = fnMap["description"].(string)
				if params, ok := fnMap["parameters"].(map[string]any); ok {
					tool.Function.Parameters = params
				}
			} else if name, ok := itemMap["name"].(string); ok {
				// Anthropic format
				tool.Type = "function"
				tool.Function.Name = name
				tool.Function.Description, _ = itemMap["description"].(string)
				if schema, ok := itemMap["input_schema"].(map[string]any); ok {
					tool.Function.Parameters = schema
				}
			}
			rec.Tools = append(rec.Tools, tool)
		}
	}

	// Parse messages
	if msgsRaw, ok := rawMap["messages"].([]any); ok {
		for _, item := range msgsRaw {
			itemMap, ok := item.(map[string]any)
			if !ok {
				continue
			}
			msg := RecordedMessage{
				Role:             stringVal(itemMap["role"]),
				Content:          itemMap["content"],
				Name:             stringVal(itemMap["name"]),
				ToolCallID:       stringVal(itemMap["tool_call_id"]),
				ReasoningContent: stringVal(itemMap["reasoning_content"]),
			}

			if tcs, ok := itemMap["tool_calls"].([]any); ok {
				for _, tcItem := range tcs {
					tcMap, ok := tcItem.(map[string]any)
					if !ok {
						continue
					}
					var tc RecordedToolCall
					if idx, ok := tcMap["index"].(float64); ok {
						tc.Index = int(idx)
					}
					tc.ID = stringVal(tcMap["id"])
					tc.Type = stringVal(tcMap["type"])
					if fn, ok := tcMap["function"].(map[string]any); ok {
						tc.Function.Name = stringVal(fn["name"])
						tc.Function.Arguments = stringVal(fn["arguments"])
					}
					msg.ToolCalls = append(msg.ToolCalls, tc)
				}
			}
			rec.Messages = append(rec.Messages, msg)
		}
	}

	return rec, nil
}

func stringVal(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}
