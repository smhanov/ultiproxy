package opencode

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/server"
)

// Harness wires up an ultiproxy Server with a registered fake or OpenCode provider
// pointing to a FakeUpstream mock server.
type Harness struct {
	Server           *server.Server
	TestServer       *httptest.Server
	FakeUpstream     *FakeUpstream
	Registry         *provider.Registry
	Provider         provider.Provider
	OpenCodeProvider *openaicompat.Provider
	tempDir          string
	client           *http.Client
}

// URL returns the base URL of the downstream ultiproxy HTTP test server.
func (h *Harness) URL() string {
	return h.TestServer.URL
}

// Client returns an http.Client configured to speak to the downstream server.
func (h *Harness) Client() *http.Client {
	return h.client
}

// Upstream returns the underlying FakeUpstream mock HTTP server.
func (h *Harness) Upstream() *FakeUpstream {
	return h.FakeUpstream
}

// Close gracefully terminates both downstream and upstream test servers and cleans up temp data.
func (h *Harness) Close() {
	if h.TestServer != nil {
		h.TestServer.Close()
	}
	if h.FakeUpstream != nil {
		h.FakeUpstream.Close()
	}
	if h.tempDir != "" {
		_ = os.RemoveAll(h.tempDir)
	}
}

// FakeProvider implements provider.InferenceProvider pointing to a FakeUpstream mock server.
type FakeProvider struct {
	name     string
	upstream *FakeUpstream
	inner    *openaicompat.Provider
}

// Name returns the provider identifier.
func (f *FakeProvider) Name() string {
	return f.name
}

// Generate implements provider.InferenceProvider.
func (f *FakeProvider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	return f.inner.Generate(ctx, msgs, opts...)
}

// Stream implements provider.InferenceProvider.
func (f *FakeProvider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	return f.inner.Stream(ctx, msgs, opts...)
}

// Capabilities returns standard capabilities.
func (f *FakeProvider) Capabilities() provider.Capabilities {
	return f.inner.Capabilities()
}

// Provider returns the provider.Provider bundle.
func (f *FakeProvider) Provider() provider.Provider {
	return provider.Provider{
		Inference:    f,
		Capabilities: f.Capabilities(),
	}
}

// Bundle returns the provider.Provider bundle.
func (f *FakeProvider) Bundle() provider.Provider {
	return f.Provider()
}

// optOutOfModelDiscovery keeps a harness lane on the explicit opt-in model
// discovery semantics: FakeUpstream serves the chat wire only and has no
// /v1/models catalog, while discovery now defaults to ON for OpenAI-compatible
// lanes - without this, every wire test would also record a model probe next to
// its chat request. Tests that DO want discovery set
// quirks.model_list_passthrough explicitly (see TestWire_ModelListPassthrough);
// the default itself is covered by pkg/provider/openaicompat and pkg/server.
func optOutOfModelDiscovery(cfg *openaicompat.Config) {
	if !cfg.Quirks.ModelListPassthrough {
		cfg.OptOutModelListPassthrough = true
	}
}

// NewFakeProvider constructs a FakeProvider with the given name pointing to the fake upstream.
func NewFakeProvider(name string, fake *FakeUpstream, opts ...func(*openaicompat.Config)) (*FakeProvider, error) {
	cfg := openaicompat.Config{
		Name:       name,
		BaseURL:    fake.URL(),
		APIKey:     "test-" + name + "-key",
		HTTPClient: fake.Client(),
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	optOutOfModelDiscovery(&cfg)
	inner, err := openaicompat.New(cfg)
	if err != nil {
		return nil, err
	}
	return &FakeProvider{
		name:     name,
		upstream: fake,
		inner:    inner,
	}, nil
}

// HarnessOption configures Harness construction.
type HarnessOption func(*harnessConfig)

type harnessConfig struct {
	fakeUpstream      *FakeUpstream
	serverConfig      *server.Config
	serverOpts        []server.Option
	apiKey            string
	clientKeys        map[string]string
	modelAliases      map[string]server.ModelAlias
	providers         []provider.Provider
	providerName      string
	customOpenCodeCfg func(*openaicompat.Config)
	dataDir           string
}

// WithFakeUpstream uses an existing FakeUpstream instance.
func WithFakeUpstream(fake *FakeUpstream) HarnessOption {
	return func(c *harnessConfig) {
		c.fakeUpstream = fake
	}
}

// WithServerConfig overrides the default server.Config.
func WithServerConfig(cfg *server.Config) HarnessOption {
	return func(c *harnessConfig) {
		c.serverConfig = cfg
	}
}

// WithServerOptions adds server.Option instances passed to server.NewServer.
func WithServerOptions(opts ...server.Option) HarnessOption {
	return func(c *harnessConfig) {
		c.serverOpts = append(c.serverOpts, opts...)
	}
}

// WithAPIKey sets the admin APIKey on the server config.
func WithAPIKey(key string) HarnessOption {
	return func(c *harnessConfig) {
		c.apiKey = key
	}
}

// WithClientKey registers a client virtual key on the server config.
func WithClientKey(name, key string) HarnessOption {
	return func(c *harnessConfig) {
		if c.clientKeys == nil {
			c.clientKeys = make(map[string]string)
		}
		c.clientKeys[name] = key
	}
}

// WithModelAlias adds a model alias to the server config.
func WithModelAlias(alias string, target server.ModelAlias) HarnessOption {
	return func(c *harnessConfig) {
		if c.modelAliases == nil {
			c.modelAliases = make(map[string]server.ModelAlias)
		}
		c.modelAliases[alias] = target
	}
}

// WithProvider registers an additional or replacement provider bundle.
func WithProvider(p provider.Provider) HarnessOption {
	return func(c *harnessConfig) {
		c.providers = append(c.providers, p)
	}
}

// WithProviderName configures a custom provider name for the fake provider.
func WithProviderName(name string) HarnessOption {
	return func(c *harnessConfig) {
		c.providerName = name
	}
}

// WithOpenCodeConfig allows custom modification of the OpenAI-compatible provider configuration.
func WithOpenCodeConfig(fn func(*openaicompat.Config)) HarnessOption {
	return func(c *harnessConfig) {
		c.customOpenCodeCfg = fn
	}
}

// WithDataDir specifies an explicit directory for server SQLite and JSON caches.
func WithDataDir(dir string) HarnessOption {
	return func(c *harnessConfig) {
		c.dataDir = dir
	}
}

// NewHarness constructs and starts a test harness with ultiproxy Server wired to FakeUpstream.
func NewHarness(opts ...HarnessOption) (*Harness, error) {
	hcfg := &harnessConfig{}
	for _, opt := range opts {
		opt(hcfg)
	}

	fake := hcfg.fakeUpstream
	if fake == nil {
		fake = NewFakeUpstream()
	}

	tempDir := hcfg.dataDir
	if tempDir == "" {
		d, err := os.MkdirTemp("", "ultiproxy-harness-*")
		if err != nil {
			return nil, fmt.Errorf("create temp data dir: %w", err)
		}
		tempDir = d
	}

	registry := provider.NewRegistry()

	var activeProvBundle provider.Provider
	var ocProv *openaicompat.Provider

	if hcfg.providerName != "" && hcfg.providerName != "opencode" {
		var optFn func(*openaicompat.Config)
		if hcfg.customOpenCodeCfg != nil {
			optFn = hcfg.customOpenCodeCfg
		}
		fp, err := NewFakeProvider(hcfg.providerName, fake, optFn)
		if err != nil {
			return nil, fmt.Errorf("create fake provider: %w", err)
		}
		activeProvBundle = fp.Provider()
		registry.Register(activeProvBundle)
	} else {
		ocCfg := openaicompat.Config{
			Name:       "opencode",
			BaseURL:    fake.URL(),
			APIKey:     "test-key",
			HTTPClient: fake.Client(),
		}
		if hcfg.customOpenCodeCfg != nil {
			hcfg.customOpenCodeCfg(&ocCfg)
		}
		optOutOfModelDiscovery(&ocCfg)
		p, err := openaicompat.New(ocCfg)
		if err != nil {
			return nil, fmt.Errorf("create opencode provider: %w", err)
		}
		ocProv = p
		activeProvBundle = p.Provider()
		registry.Register(activeProvBundle)
	}

	for _, p := range hcfg.providers {
		registry.Register(p)
	}

	srvCfg := hcfg.serverConfig
	if srvCfg == nil {
		srvCfg = server.DefaultConfig()
	}
	if srvCfg.DataDir == "" || srvCfg.DataDir == "." {
		srvCfg.DataDir = tempDir
	}
	if srvCfg.Storage.DBPath == "" || srvCfg.Storage.DBPath == "ultiproxy.db" {
		srvCfg.Storage.DBPath = filepath.Join(tempDir, "ultiproxy.db")
	}
	if hcfg.apiKey != "" {
		srvCfg.Server.APIKey = hcfg.apiKey
	}
	for k, v := range hcfg.clientKeys {
		if srvCfg.Server.ClientKeys == nil {
			srvCfg.Server.ClientKeys = make(map[string]string)
		}
		srvCfg.Server.ClientKeys[k] = v
	}
	for alias, target := range hcfg.modelAliases {
		if srvCfg.Server.Models == nil {
			srvCfg.Server.Models = make(map[string]server.ModelAlias)
		}
		srvCfg.Server.Models[alias] = target
	}

	srv := server.NewServer(srvCfg, registry, hcfg.serverOpts...)
	testServer := httptest.NewServer(srv.Handler())

	client := testServer.Client()
	client.Timeout = 10 * time.Second

	return &Harness{
		Server:           srv,
		TestServer:       testServer,
		FakeUpstream:     fake,
		Registry:         registry,
		Provider:         activeProvBundle,
		OpenCodeProvider: ocProv,
		tempDir:          tempDir,
		client:           client,
	}, nil
}

// NewTestHarness creates a Harness and registers cleanup with t.Cleanup.
func NewTestHarness(t testing.TB, opts ...HarnessOption) *Harness {
	h, err := NewHarness(opts...)
	if err != nil {
		t.Fatalf("failed to create test harness: %v", err)
	}
	t.Cleanup(h.Close)
	return h
}

// PostChat sends a non-streaming POST request to /v1/chat/completions.
func (h *Harness) PostChat(ctx context.Context, req any) (*http.Response, map[string]any, error) {
	reqBytes, err := marshalRequest(req)
	if err != nil {
		return nil, nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, h.URL()+"/v1/chat/completions", bytes.NewReader(reqBytes))
	if err != nil {
		return nil, nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := h.Client().Do(httpReq)
	if err != nil {
		return nil, nil, err
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return resp, nil, err
	}
	resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var jsonMap map[string]any
	if len(bodyBytes) > 0 {
		_ = json.Unmarshal(bodyBytes, &jsonMap)
	}

	return resp, jsonMap, nil
}

// StreamChat sends a streaming POST request to /v1/chat/completions and collects the observed stream events.
func (h *Harness) StreamChat(ctx context.Context, req any) (*StreamObservation, *http.Response, error) {
	reqBytes, err := marshalRequestWithStream(req, true)
	if err != nil {
		return nil, nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, h.URL()+"/v1/chat/completions", bytes.NewReader(reqBytes))
	if err != nil {
		return nil, nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := h.Client().Do(httpReq)
	if err != nil {
		return nil, nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, resp, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(body))
	}

	obs, err := ObserveStream(resp)
	return obs, resp, err
}

// PostMessages sends a non-streaming POST request to /v1/messages (Anthropic surface).
func (h *Harness) PostMessages(ctx context.Context, req any) (*http.Response, map[string]any, error) {
	reqBytes, err := marshalRequest(req)
	if err != nil {
		return nil, nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, h.URL()+"/v1/messages", bytes.NewReader(reqBytes))
	if err != nil {
		return nil, nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := h.Client().Do(httpReq)
	if err != nil {
		return nil, nil, err
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return resp, nil, err
	}
	resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var jsonMap map[string]any
	if len(bodyBytes) > 0 {
		_ = json.Unmarshal(bodyBytes, &jsonMap)
	}

	return resp, jsonMap, nil
}

// StreamMessages sends a streaming POST request to /v1/messages and collects the observed stream events.
func (h *Harness) StreamMessages(ctx context.Context, req any) (*StreamObservation, *http.Response, error) {
	reqBytes, err := marshalRequestWithStream(req, true)
	if err != nil {
		return nil, nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, h.URL()+"/v1/messages", bytes.NewReader(reqBytes))
	if err != nil {
		return nil, nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := h.Client().Do(httpReq)
	if err != nil {
		return nil, nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, resp, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(body))
	}

	obs, err := ObserveStream(resp)
	return obs, resp, err
}

// StreamObservation represents the accumulated downstream SSE stream received from ultiproxy.
type StreamObservation struct {
	RawEvents    []DownstreamEvent
	Text         string
	Reasoning    string
	ToolCalls    []DownstreamToolCall
	FinishReason string
	Usage        *DownstreamUsage
	RawLines     []string
}

// CombinedText returns the accumulated text content.
func (o *StreamObservation) CombinedText() string {
	return o.Text
}

// CombinedReasoning returns the accumulated reasoning content.
func (o *StreamObservation) CombinedReasoning() string {
	return o.Reasoning
}

// GetToolCall returns the tool call at the specified index, or nil.
func (o *StreamObservation) GetToolCall(index int) *DownstreamToolCall {
	for i := range o.ToolCalls {
		if o.ToolCalls[i].Index == index {
			return &o.ToolCalls[i]
		}
	}
	return nil
}

// GetToolCallByName returns the first tool call matching the given function name.
func (o *StreamObservation) GetToolCallByName(name string) *DownstreamToolCall {
	for i := range o.ToolCalls {
		if o.ToolCalls[i].Name == name {
			return &o.ToolCalls[i]
		}
	}
	return nil
}

// HasToolCall reports whether a tool call with the given function name was observed.
func (o *StreamObservation) HasToolCall(name string) bool {
	return o.GetToolCallByName(name) != nil
}

// DownstreamEvent represents a single Server-Sent Event read from the downstream stream.
type DownstreamEvent struct {
	Event string
	Data  string
	JSON  map[string]any
}

// DownstreamToolCall represents a tool call invocation observed downstream.
type DownstreamToolCall struct {
	Index     int
	ID        string
	Name      string
	Arguments string
}

// DownstreamUsage represents token usage metrics reported downstream.
type DownstreamUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// ObserveStream reads an SSE HTTP response body and collects it into a StreamObservation.
func ObserveStream(resp *http.Response) (*StreamObservation, error) {
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 64*1024)
	scanner.Buffer(buf, 10*1024*1024)

	obs := &StreamObservation{}
	var curEvent string
	var curDataLines []string

	for scanner.Scan() {
		line := scanner.Text()
		obs.RawLines = append(obs.RawLines, line)

		if line == "" {
			if len(curDataLines) > 0 || curEvent != "" {
				data := strings.Join(curDataLines, "\n")
				ev := DownstreamEvent{
					Event: curEvent,
					Data:  data,
				}

				if data != "" && data != "[DONE]" {
					var jsonMap map[string]any
					if err := json.Unmarshal([]byte(data), &jsonMap); err == nil {
						ev.JSON = jsonMap
						processDownstreamJSON(obs, curEvent, jsonMap)
					}
				}

				obs.RawEvents = append(obs.RawEvents, ev)
				curEvent = ""
				curDataLines = nil
			}
			continue
		}

		if strings.HasPrefix(line, ":") {
			continue
		}

		if strings.HasPrefix(line, "event:") {
			curEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			dataLine := strings.TrimPrefix(line, "data:")
			dataLine = strings.TrimPrefix(dataLine, " ")
			curDataLines = append(curDataLines, dataLine)
		}
	}

	if len(curDataLines) > 0 || curEvent != "" {
		data := strings.Join(curDataLines, "\n")
		ev := DownstreamEvent{
			Event: curEvent,
			Data:  data,
		}
		if data != "" && data != "[DONE]" {
			var jsonMap map[string]any
			if err := json.Unmarshal([]byte(data), &jsonMap); err == nil {
				ev.JSON = jsonMap
				processDownstreamJSON(obs, curEvent, jsonMap)
			}
		}
		obs.RawEvents = append(obs.RawEvents, ev)
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return obs, err
	}

	return obs, nil
}

func processDownstreamJSON(obs *StreamObservation, eventType string, m map[string]any) {
	// 1. OpenAI format
	if choices, ok := m["choices"].([]any); ok {
		for _, c := range choices {
			choiceMap, ok := c.(map[string]any)
			if !ok {
				continue
			}

			if delta, ok := choiceMap["delta"].(map[string]any); ok {
				if content, ok := delta["content"].(string); ok {
					obs.Text += content
				}
				if reasoning, ok := delta["reasoning_content"].(string); ok {
					obs.Reasoning += reasoning
				}
				if tcs, ok := delta["tool_calls"].([]any); ok {
					for _, tcItem := range tcs {
						tcMap, ok := tcItem.(map[string]any)
						if !ok {
							continue
						}
						var idx int
						if iVal, ok := tcMap["index"].(float64); ok {
							idx = int(iVal)
						}
						id, _ := tcMap["id"].(string)

						var name, args string
						if fn, ok := tcMap["function"].(map[string]any); ok {
							name, _ = fn["name"].(string)
							args, _ = fn["arguments"].(string)
						}

						// Find or create in obs.ToolCalls
						found := false
						for i := range obs.ToolCalls {
							if obs.ToolCalls[i].Index == idx {
								if id != "" {
									obs.ToolCalls[i].ID = id
								}
								if name != "" {
									obs.ToolCalls[i].Name = name
								}
								if args != "" {
									obs.ToolCalls[i].Arguments += args
								}
								found = true
								break
							}
						}
						if !found {
							obs.ToolCalls = append(obs.ToolCalls, DownstreamToolCall{
								Index:     idx,
								ID:        id,
								Name:      name,
								Arguments: args,
							})
						}
					}
				}
			}

			if fr, ok := choiceMap["finish_reason"].(string); ok && fr != "" {
				obs.FinishReason = fr
			}
		}
	}

	if usage, ok := m["usage"].(map[string]any); ok {
		if obs.Usage == nil {
			obs.Usage = &DownstreamUsage{}
		}
		if p, ok := usage["prompt_tokens"].(float64); ok {
			obs.Usage.PromptTokens = int(p)
		}
		if c, ok := usage["completion_tokens"].(float64); ok {
			obs.Usage.CompletionTokens = int(c)
		}
		if t, ok := usage["total_tokens"].(float64); ok {
			obs.Usage.TotalTokens = int(t)
		}
	}

	// 2. Anthropic format
	switch eventType {
	case "content_block_start":
		var idx int
		if iVal, ok := m["index"].(float64); ok {
			idx = int(iVal)
		}
		if cb, ok := m["content_block"].(map[string]any); ok {
			if cbType, ok := cb["type"].(string); ok && cbType == "tool_use" {
				id, _ := cb["id"].(string)
				name, _ := cb["name"].(string)
				obs.ToolCalls = append(obs.ToolCalls, DownstreamToolCall{
					Index: idx,
					ID:    id,
					Name:  name,
				})
			}
		}

	case "content_block_delta":
		var idx int
		if iVal, ok := m["index"].(float64); ok {
			idx = int(iVal)
		}
		if delta, ok := m["delta"].(map[string]any); ok {
			dType, _ := delta["type"].(string)
			switch dType {
			case "text_delta":
				if txt, ok := delta["text"].(string); ok {
					obs.Text += txt
				}
			case "thinking_delta":
				if thk, ok := delta["thinking"].(string); ok {
					obs.Reasoning += thk
				}
			case "input_json_delta":
				if pjson, ok := delta["partial_json"].(string); ok {
					for i := range obs.ToolCalls {
						if obs.ToolCalls[i].Index == idx {
							obs.ToolCalls[i].Arguments += pjson
							break
						}
					}
				}
			}
		}

	case "message_delta":
		if delta, ok := m["delta"].(map[string]any); ok {
			if sr, ok := delta["stop_reason"].(string); ok && sr != "" {
				obs.FinishReason = sr
			}
		}
		if usage, ok := m["usage"].(map[string]any); ok {
			if obs.Usage == nil {
				obs.Usage = &DownstreamUsage{}
			}
			if out, ok := usage["output_tokens"].(float64); ok {
				obs.Usage.CompletionTokens = int(out)
				obs.Usage.TotalTokens = obs.Usage.PromptTokens + obs.Usage.CompletionTokens
			}
		}

	case "message_start":
		if msg, ok := m["message"].(map[string]any); ok {
			if usage, ok := msg["usage"].(map[string]any); ok {
				if obs.Usage == nil {
					obs.Usage = &DownstreamUsage{}
				}
				if in, ok := usage["input_tokens"].(float64); ok {
					obs.Usage.PromptTokens = int(in)
					obs.Usage.TotalTokens = obs.Usage.PromptTokens + obs.Usage.CompletionTokens
				}
			}
		}
	}
}

func marshalRequest(req any) ([]byte, error) {
	switch v := req.(type) {
	case []byte:
		return v, nil
	case string:
		return []byte(v), nil
	default:
		return json.Marshal(v)
	}
}

func marshalRequestWithStream(req any, stream bool) ([]byte, error) {
	switch v := req.(type) {
	case []byte:
		var m map[string]any
		if err := json.Unmarshal(v, &m); err == nil {
			m["stream"] = stream
			return json.Marshal(m)
		}
		return v, nil
	case string:
		var m map[string]any
		if err := json.Unmarshal([]byte(v), &m); err == nil {
			m["stream"] = stream
			return json.Marshal(m)
		}
		return []byte(v), nil
	case map[string]any:
		v["stream"] = stream
		return json.Marshal(v)
	default:
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err == nil {
			m["stream"] = stream
			return json.Marshal(m)
		}
		return raw, nil
	}
}
