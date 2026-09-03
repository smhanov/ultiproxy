package xai

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/oauth"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/sse"
)

const (
	DefaultBaseURL         = "https://api.x.ai"
	DefaultBillingURL      = "https://grok.com/grok_api_v2.GrokBuildBilling/GetGrokCreditsConfig"
	DefaultClientID        = "b1a00492-073a-47ea-816f-4c329264a828"
	DefaultIssuer          = "https://auth.x.ai"
	DefaultDeviceURL       = "https://auth.x.ai/oauth2/device/code"
	DefaultTokenURL        = "https://auth.x.ai/oauth2/token"
	DefaultVerificationURL = "https://accounts.x.ai/oauth2/device"
)

// Config configures the xAI Grok provider adapter.
type Config struct {
	BaseURL         string
	BillingURL      string
	ClientID        string
	ClientSecret    string
	Issuer          string
	DeviceAuthURL   string
	TokenURL        string
	VerificationURL string
	AuthManager     *auth.Manager
	Refresher       auth.Refresher
	StaticToken     string
	HTTPClient      *http.Client
}

// Provider implements InferenceProvider, QuotaProvider, and AuthProvider for xAI.
type Provider struct {
	baseURL         string
	billingURL      string
	clientID        string
	clientSecret    string
	issuer          string
	deviceAuthURL   string
	tokenURL        string
	verificationURL string
	authManager     *auth.Manager
	refresher       auth.Refresher
	staticToken     string
	httpClient      *http.Client

	mu        sync.RWMutex
	liveToken string
}

// New creates a new xAI Grok provider adapter.
func New(cfg Config) *Provider {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")

	billingURL := cfg.BillingURL
	if billingURL == "" {
		billingURL = DefaultBillingURL
	}

	clientID := cfg.ClientID
	if clientID == "" {
		clientID = DefaultClientID
	}

	issuer := cfg.Issuer
	if issuer == "" {
		issuer = DefaultIssuer
	}

	deviceURL := cfg.DeviceAuthURL
	if deviceURL == "" {
		deviceURL = DefaultDeviceURL
	}

	tokenURL := cfg.TokenURL
	if tokenURL == "" {
		tokenURL = DefaultTokenURL
	}

	verificationURL := cfg.VerificationURL
	if verificationURL == "" {
		verificationURL = DefaultVerificationURL
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	return &Provider{
		baseURL:         baseURL,
		billingURL:      billingURL,
		clientID:        clientID,
		clientSecret:    cfg.ClientSecret,
		issuer:          issuer,
		deviceAuthURL:   deviceURL,
		tokenURL:        tokenURL,
		verificationURL: verificationURL,
		authManager:     cfg.AuthManager,
		refresher:       cfg.Refresher,
		staticToken:     cfg.StaticToken,
		httpClient:      client,
	}
}

// Name implements InferenceProvider, QuotaProvider, and AuthProvider.
func (p *Provider) Name() string {
	return "xai"
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:      true,
		Tools:     true,
		Reasoning: true,
		Streaming: true,
		Vision:    true,
	}
}

// ProviderBundle returns a provider.Provider bundle.
func (p *Provider) ProviderBundle() provider.Provider {
	return provider.Provider{
		Inference:    p,
		Quota:        p,
		Auth:         p,
		Capabilities: p.Capabilities(),
	}
}

// Register registers this provider in a registry.
func (p *Provider) Register(r *provider.Registry) {
	r.Register(p.ProviderBundle())
}

func (p *Provider) getToken(ctx context.Context) (string, error) {
	if p.staticToken != "" {
		return p.staticToken, nil
	}
	p.mu.RLock()
	if p.liveToken != "" {
		t := p.liveToken
		p.mu.RUnlock()
		return t, nil
	}
	p.mu.RUnlock()

	if p.authManager != nil {
		cred, err := p.authManager.Get(ctx, p.clientID)
		if err == nil && cred.AccessToken != "" {
			return cred.AccessToken, nil
		}
	}
	return "", errors.New("xai: no access token available")
}

// -----------------------------------------------------------------------------
// Completions & Inference
// -----------------------------------------------------------------------------

// BuildPayload constructs the JSON payload for xAI chat completions,
// including reasoning_effort passthrough.
func BuildPayload(msgs []*ir.Message, cfg *provider.RequestConfig, stream bool) (map[string]any, error) {
	payload := make(map[string]any)
	payload["model"] = cfg.Model
	payload["stream"] = stream

	if cfg.MaxTokens > 0 {
		payload["max_tokens"] = cfg.MaxTokens
	}
	if cfg.Temperature != nil {
		payload["temperature"] = *cfg.Temperature
	}

	// reasoning_effort passthrough (Default/low/medium/high/xhigh)
	if cfg.ReasoningEffort != "" {
		payload["reasoning_effort"] = cfg.ReasoningEffort
	}

	var chatMsgs []map[string]any
	for _, m := range msgs {
		if m == nil {
			continue
		}
		var textParts []string
		var toolCalls []map[string]any
		var toolCallID string

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
				toolCalls = append(toolCalls, map[string]any{
					"id":   b.ID,
					"type": "function",
					"function": map[string]any{
						"name":      b.Name,
						"arguments": b.Arguments,
					},
				})
			case *ir.ToolCallBlock:
				toolCalls = append(toolCalls, map[string]any{
					"id":   b.ID,
					"type": "function",
					"function": map[string]any{
						"name":      b.Name,
						"arguments": b.Arguments,
					},
				})
			case ir.ToolResultBlock:
				toolCallID = b.ToolCallID
				textParts = append(textParts, b.Content)
			case *ir.ToolResultBlock:
				toolCallID = b.ToolCallID
				textParts = append(textParts, b.Content)
			}
		}

		entry := map[string]any{
			"role":    m.Role,
			"content": strings.Join(textParts, "\n"),
		}
		if len(toolCalls) > 0 {
			entry["tool_calls"] = toolCalls
		}
		if toolCallID != "" {
			entry["role"] = "tool"
			entry["tool_call_id"] = toolCallID
		}
		chatMsgs = append(chatMsgs, entry)
	}

	payload["messages"] = chatMsgs

	for k, v := range cfg.ExtraBody {
		payload[k] = v
	}

	return payload, nil
}

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	cfg := provider.NewRequestConfig(opts...)

	tok, err := p.getToken(ctx)
	if err != nil {
		return nil, err
	}

	payloadMap, err := BuildPayload(msgs, cfg, false)
	if err != nil {
		return nil, fmt.Errorf("xai: failed to build payload: %w", err)
	}

	bodyBytes, err := json.Marshal(payloadMap)
	if err != nil {
		return nil, fmt.Errorf("xai: failed to marshal payload: %w", err)
	}

	endpoint := p.baseURL + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("xai: failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("xai: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("xai: failed to read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("xai: upstream returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	var chatResp struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
				ToolCalls        []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("xai: failed to parse response: %w", err)
	}

	irResp := &ir.Response{
		ID:         chatResp.ID,
		UpstreamID: chatResp.ID,
	}

	if len(chatResp.Choices) > 0 {
		choice := chatResp.Choices[0]
		irResp.FinishReason = choice.FinishReason

		var blocks []ir.Block
		if choice.Message.ReasoningContent != "" {
			blocks = append(blocks, ir.ReasoningBlock{
				ReasoningKind: ir.ReasoningText,
				Text:          choice.Message.ReasoningContent,
			})
		}
		if choice.Message.Content != "" {
			blocks = append(blocks, ir.TextBlock{Text: choice.Message.Content})
		}
		for _, tc := range choice.Message.ToolCalls {
			blocks = append(blocks, ir.ToolCallBlock{
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

// Stream implements provider.InferenceProvider.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	cfg := provider.NewRequestConfig(opts...)

	tok, err := p.getToken(ctx)
	if err != nil {
		return nil, err
	}

	payloadMap, err := BuildPayload(msgs, cfg, true)
	if err != nil {
		return nil, fmt.Errorf("xai: failed to build payload: %w", err)
	}

	bodyBytes, err := json.Marshal(payloadMap)
	if err != nil {
		return nil, fmt.Errorf("xai: failed to marshal payload: %w", err)
	}

	endpoint := p.baseURL + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("xai: failed to create stream request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("xai: stream request failed: %w", err)
	}

	// Synchronous check: return error synchronously on non-2xx
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("xai: upstream returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	outCh := make(chan ir.Event, 64)

	go func() {
		defer close(outCh)
		defer resp.Body.Close()

		scanner := sse.NewScanner(resp.Body)
		started := false
		var responseID string

		type streamToolCall struct {
			Index    int    `json:"index"`
			ID       string `json:"id"`
			Type     string `json:"type"`
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		}

		type streamChunk struct {
			ID      string `json:"id"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Content          string           `json:"content,omitempty"`
					ReasoningContent string           `json:"reasoning_content,omitempty"`
					ToolCalls        []streamToolCall `json:"tool_calls,omitempty"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}

		activeTools := make(map[int]bool)

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

			// Deliver reasoning_content BEFORE text deltas, then tool_calls.
			// Dropping tool_calls while still forwarding finish_reason=tool_calls
			// is what made OpenCode retry the same turn forever.
			for _, c := range chunk.Choices {
				if c.Delta.ReasoningContent != "" {
					outCh <- ir.EventReasoningDelta{Index: c.Index, Text: c.Delta.ReasoningContent}
				}
				if c.Delta.Content != "" {
					outCh <- ir.EventTextDelta{Index: c.Index, Text: c.Delta.Content}
				}
				for _, tc := range c.Delta.ToolCalls {
					toolIdx := tc.Index
					if !activeTools[toolIdx] {
						activeTools[toolIdx] = true
						outCh <- ir.EventToolCallStart{Index: toolIdx, ID: tc.ID, Name: tc.Function.Name}
					}
					if tc.Function.Arguments != "" {
						outCh <- ir.EventToolArgumentsDelta{Index: toolIdx, Arguments: tc.Function.Arguments}
					}
				}
				if c.FinishReason != nil && *c.FinishReason != "" {
					for idx := range activeTools {
						outCh <- ir.EventToolCallStop{Index: idx}
					}
					activeTools = make(map[int]bool)
					outCh <- ir.EventMessageStop{FinishReason: *c.FinishReason, UpstreamID: responseID}
				}
			}

			// Usage in final chunk
			if chunk.Usage != nil {
				outCh <- ir.EventUsageUpdate{
					PromptTokens:     chunk.Usage.PromptTokens,
					CompletionTokens: chunk.Usage.CompletionTokens,
					TotalTokens:      chunk.Usage.TotalTokens,
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
// QuotaProvider: gRPC-Web Credit Scraping
// -----------------------------------------------------------------------------

// Quota implements provider.QuotaProvider.
func (p *Provider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	tok, err := p.getToken(ctx)
	if err != nil {
		return nil, err
	}

	// gRPC-web framed empty request: 5 zero bytes
	emptyFrame := []byte{0x00, 0x00, 0x00, 0x00, 0x00}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.billingURL, bytes.NewReader(emptyFrame))
	if err != nil {
		return nil, fmt.Errorf("xai: failed to create billing request: %w", err)
	}

	req.Header.Set("Content-Type", "application/grpc-web+proto")
	req.Header.Set("X-Grpc-Web", "1")
	req.Header.Set("X-User-Agent", "grpc-web-javascript/0.1")
	req.Header.Set("Authorization", "Bearer "+tok)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("xai: billing request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("xai: failed to read billing response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return &provider.QuotaSnapshot{
			ObservedAt: time.Now().UTC(),
			Detail:     fmt.Sprintf("xAI billing HTTP %d (status unknown)", resp.StatusCode),
		}, nil
	}

	return ParseGrokCreditsResponse(body, time.Now().UTC())
}

// -----------------------------------------------------------------------------
// gRPC-Web Frame & Protobuf Parsing
// -----------------------------------------------------------------------------

type scannedField struct {
	Path   []int
	Wire   int
	Varint uint64
	Float  float32
	Double float64
	String string
	Bytes  []byte
}

type protobufScanner struct {
	fields []scannedField
}

func readVarint(data []byte, idx int) (uint64, int, bool) {
	var val uint64
	var shift uint
	for idx < len(data) && shift < 64 {
		b := data[idx]
		idx++
		val |= uint64(b&0x7f) << shift
		if b < 0x80 {
			return val, idx, true
		}
		shift += 7
	}
	return 0, idx, false
}

func (s *protobufScanner) scan(data []byte, path []int, depth int) {
	if depth > 10 {
		return
	}
	i := 0
	for i < len(data) {
		start := i
		key, nextIdx, ok := readVarint(data, i)
		if !ok || key == 0 {
			i = start + 1
			continue
		}
		fieldNum := int(key >> 3)
		wireType := int(key & 7)
		currPath := append(append([]int(nil), path...), fieldNum)
		i = nextIdx

		switch wireType {
		case 0: // Varint
			val, endIdx, ok := readVarint(data, i)
			if !ok {
				i = start + 1
				continue
			}
			s.fields = append(s.fields, scannedField{
				Path:   currPath,
				Wire:   0,
				Varint: val,
			})
			i = endIdx

		case 1: // 64-bit
			if i+8 > len(data) {
				i = start + 1
				continue
			}
			bits := binary.LittleEndian.Uint64(data[i : i+8])
			valFloat := math.Float64frombits(bits)
			s.fields = append(s.fields, scannedField{
				Path:   currPath,
				Wire:   1,
				Double: valFloat,
			})
			i += 8

		case 2: // Length-delimited
			size, endIdx, ok := readVarint(data, i)
			if !ok || int(size) > len(data)-endIdx {
				i = start + 1
				continue
			}
			subBytes := data[endIdx : endIdx+int(size)]
			s.fields = append(s.fields, scannedField{
				Path:   currPath,
				Wire:   2,
				String: string(subBytes),
				Bytes:  subBytes,
			})
			// Recurse into embedded message
			s.scan(subBytes, currPath, depth+1)
			i = endIdx + int(size)

		case 5: // 32-bit
			if i+4 > len(data) {
				i = start + 1
				continue
			}
			bits := binary.LittleEndian.Uint32(data[i : i+4])
			valFloat := math.Float32frombits(bits)
			s.fields = append(s.fields, scannedField{
				Path:  currPath,
				Wire:  5,
				Float: valFloat,
			})
			i += 4

		default:
			i = start + 1
		}
	}
}

// UnframeGRPCWeb extracts all data payloads from a gRPC-web stream.
func UnframeGRPCWeb(data []byte) ([][]byte, error) {
	var payloads [][]byte
	i := 0
	for i+5 <= len(data) {
		flags := data[i]
		size := binary.BigEndian.Uint32(data[i+1 : i+5])
		start := i + 5
		end := start + int(size)
		if end > len(data) {
			break
		}
		// flags & 0x80 == 0 indicates a data frame
		if (flags & 0x80) == 0 {
			payloads = append(payloads, data[start:end])
		}
		i = end
	}
	if len(payloads) == 0 && len(data) > 0 {
		// Fall back to entire buffer if not framed
		payloads = append(payloads, data)
	}
	return payloads, nil
}

// ParseGrokCreditsResponse parses gRPC-web binary data into QuotaSnapshot.
func ParseGrokCreditsResponse(data []byte, now time.Time) (*provider.QuotaSnapshot, error) {
	payloads, err := UnframeGRPCWeb(data)
	if err != nil {
		return &provider.QuotaSnapshot{
			ObservedAt: now,
			Detail:     "Failed to unframe gRPC-web response (status unknown)",
		}, nil
	}

	scanner := &protobufScanner{}
	for _, p := range payloads {
		scanner.scan(p, nil, 0)
	}

	// Look for pools: 5-hour and Weekly
	type poolData struct {
		Name      string
		UsedPct   *float64
		PoolUsed  *float64
		PoolTotal *float64
		ResetAt   time.Time
	}

	pools := make(map[string]*poolData)

	getOrCreatePool := func(name string) *poolData {
		if p, ok := pools[name]; ok {
			return p
		}
		p := &poolData{Name: name}
		pools[name] = p
		return p
	}

	// 1. Check for named pools or pool submessages
	for _, f := range scanner.fields {
		if f.Wire == 2 && (strings.Contains(strings.ToLower(f.String), "5 hour") || strings.Contains(strings.ToLower(f.String), "weekly")) {
			label := "5 hour"
			if strings.Contains(strings.ToLower(f.String), "weekly") {
				label = "Weekly"
			}
			getOrCreatePool(label)
		}
	}

	// If named pools found, match fields inside their subtrees
	for _, f := range scanner.fields {
		for name, p := range pools {
			isMatch := false
			if strings.EqualFold(name, "5 hour") && len(f.Path) > 0 && f.Path[0] == 1 {
				isMatch = true
			} else if strings.EqualFold(name, "Weekly") && len(f.Path) > 0 && f.Path[0] == 2 {
				isMatch = true
			}
			if isMatch {
				if f.Wire == 5 && f.Float >= 0 && f.Float <= 100 && p.UsedPct == nil {
					pct := float64(f.Float)
					p.UsedPct = &pct
				}
				if f.Wire == 0 && f.Varint >= 1700000000 && f.Varint <= 2500000000 {
					p.ResetAt = time.Unix(int64(f.Varint), 0).UTC()
				}
			}
		}
	}

	var windows []provider.QuotaWindow

	// If 5 hour and/or Weekly pools were found:
	order := []string{"5 hour", "Weekly"}
	for _, name := range order {
		if p, ok := pools[name]; ok && p.UsedPct != nil {
			var secRem int64
			if !p.ResetAt.IsZero() && p.ResetAt.After(now) {
				secRem = int64(p.ResetAt.Sub(now).Seconds())
			}
			used := *p.UsedPct
			rem := math.Max(0, 100.0-used)
			windows = append(windows, provider.QuotaWindow{
				Label:            name,
				UsedPct:          used,
				Remaining:        rem,
				Limit:            100,
				Unit:             "%",
				ResetAt:          p.ResetAt,
				SecondsRemaining: secRem,
			})
		}
	}

	// 2. Fallback to generic credit percentage and reset scan if no specific pools
	if len(windows) == 0 {
		var candidates []float64
		for _, f := range scanner.fields {
			if f.Wire == 5 && f.Float >= 0 && f.Float <= 100 {
				candidates = append(candidates, float64(f.Float))
			}
		}

		var resets []time.Time
		for _, f := range scanner.fields {
			if f.Wire == 0 && f.Varint >= 1700000000 && f.Varint <= 2500000000 {
				resets = append(resets, time.Unix(int64(f.Varint), 0).UTC())
			}
		}

		if len(candidates) > 0 {
			usedPct := candidates[0]
			var reset time.Time
			if len(resets) > 0 {
				reset = resets[0]
			}
			var secRem int64
			if !reset.IsZero() && reset.After(now) {
				secRem = int64(reset.Sub(now).Seconds())
			}

			windows = append(windows, provider.QuotaWindow{
				Label:            "Grok Build credits",
				UsedPct:          usedPct,
				Remaining:        math.Max(0, 100.0-usedPct),
				Limit:            100,
				Unit:             "%",
				ResetAt:          reset,
				SecondsRemaining: secRem,
			})
		}
	}

	if len(windows) == 0 {
		return &provider.QuotaSnapshot{
			ObservedAt: now,
			Detail:     "Grok billing response contained no recognizable credit pools (status unknown)",
		}, nil
	}

	var detailParts []string
	for _, w := range windows {
		detailParts = append(detailParts, fmt.Sprintf("%s: %.1f%% used", w.Label, w.UsedPct))
	}

	return &provider.QuotaSnapshot{
		ObservedAt: now,
		Windows:    windows,
		Detail:     strings.Join(detailParts, " · "),
	}, nil
}

// -----------------------------------------------------------------------------
// AuthProvider
// -----------------------------------------------------------------------------

func (p *Provider) Login(ctx context.Context) error {
	cfg := oauth.DeviceFlowConfig{
		ClientID:      p.clientID,
		ClientSecret:  p.clientSecret,
		DeviceAuthURL: p.deviceAuthURL,
		TokenURL:      p.tokenURL,
		HTTPClient:    p.httpClient,
	}

	dcr, err := oauth.RequestDeviceCode(ctx, cfg)
	if err != nil {
		return fmt.Errorf("xai: login device code failed: %w", err)
	}

	tokResp, err := oauth.PollToken(ctx, cfg, dcr.DeviceCode, dcr.Interval)
	if err != nil {
		return fmt.Errorf("xai: login token poll failed: %w", err)
	}

	p.mu.Lock()
	p.liveToken = tokResp.AccessToken
	p.mu.Unlock()

	if p.authManager != nil {
		expiresIn := tokResp.ExpiresIn
		if expiresIn <= 0 {
			expiresIn = 3600
		}
		cred := auth.Credential{
			Provider:     "xai",
			AccessToken:  tokResp.AccessToken,
			RefreshToken: tokResp.RefreshToken,
			ExpiresAt:    time.Now().Add(time.Duration(expiresIn) * time.Second),
			ClientID:     p.clientID,
		}
		_ = p.authManager.Store(ctx, p.clientID, cred)
	}

	return nil
}

func (p *Provider) Token(ctx context.Context) (string, error) {
	return p.getToken(ctx)
}

func (p *Provider) Refresh(ctx context.Context) error {
	var cred auth.Credential
	if p.authManager != nil {
		c, err := p.authManager.LoadFromDisk(p.clientID)
		if err == nil {
			cred = c
		}
	}

	ref := p.refresher
	if ref == nil {
		ref = oauth.MakeRefresher(p.httpClient, p.tokenURL, p.clientID, p.clientSecret)
	}

	newCred, err := ref(ctx, cred)
	if err != nil {
		return err
	}

	p.mu.Lock()
	p.liveToken = newCred.AccessToken
	p.mu.Unlock()

	if p.authManager != nil {
		_ = p.authManager.Store(ctx, p.clientID, newCred)
	}
	return nil
}
