package antigravity

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
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
	DefaultBaseURL   = "https://daily-cloudcode-pa.googleapis.com"
	UserAgent        = "antigravity/hub/2.9.1 darwin/arm64"
	DefaultClientID  = "antigravity-client-id"
	DefaultDeviceURL = "https://oauth2.googleapis.com/device/code"
	DefaultTokenURL  = "https://oauth2.googleapis.com/token"
)

// ErrInvalidToolSchema indicates an invalid tool parameters schema.
var ErrInvalidToolSchema = errors.New("HTTP 400 Bad Request: invalid tool schema")

// Config configures the Google Antigravity adapter.
type Config struct {
	BaseURL       string
	ProjectID     string
	ClientID      string
	ClientSecret  string
	DeviceAuthURL string
	TokenURL      string
	AuthManager   *auth.Manager
	Refresher     auth.Refresher
	StaticToken   string
	HTTPClient    *http.Client
}

// Provider implements InferenceProvider, QuotaProvider, and AuthProvider for Antigravity.
type Provider struct {
	baseURL       string
	projectID     string
	clientID      string
	clientSecret  string
	deviceAuthURL string
	tokenURL      string
	authManager   *auth.Manager
	refresher     auth.Refresher
	staticToken   string
	httpClient    *http.Client

	mu        sync.RWMutex
	liveToken string
}

// New creates a new Google Antigravity adapter.
func New(cfg Config) *Provider {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")

	clientID := cfg.ClientID
	if clientID == "" {
		clientID = DefaultClientID
	}

	deviceURL := cfg.DeviceAuthURL
	if deviceURL == "" {
		deviceURL = DefaultDeviceURL
	}

	tokenURL := cfg.TokenURL
	if tokenURL == "" {
		tokenURL = DefaultTokenURL
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	return &Provider{
		baseURL:       baseURL,
		projectID:     cfg.ProjectID,
		clientID:      clientID,
		clientSecret:  cfg.ClientSecret,
		deviceAuthURL: deviceURL,
		tokenURL:      tokenURL,
		authManager:   cfg.AuthManager,
		staticToken:   cfg.StaticToken,
		httpClient:    client,
	}
}

// Name implements InferenceProvider, QuotaProvider, and AuthProvider.
func (p *Provider) Name() string {
	return "antigravity"
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

// -----------------------------------------------------------------------------
// Strict Tool Schema Sanitizer
// -----------------------------------------------------------------------------

// ValidateToolSchema implements the strict tool schema sanitizer:
// Rejects (HTTP 400 error surfaced) any tool whose parameters schema has
// properties/required on a node without type: object, or required fields not
// defined in that node's properties. Do NOT silently prune — return a clear error.
func ValidateToolSchema(schema map[string]any) error {
	if schema == nil {
		return nil
	}
	return validateSchemaNode("", schema)
}

func validateSchemaNode(path string, node map[string]any) error {
	hasProps := false
	var propsMap map[string]any
	if props, ok := node["properties"]; ok && props != nil {
		if pm, isMap := props.(map[string]any); isMap {
			propsMap = pm
			hasProps = len(pm) > 0
		} else {
			hasProps = true
		}
	}

	hasRequired := false
	var reqFields []string
	if req, ok := node["required"]; ok && req != nil {
		switch r := req.(type) {
		case []string:
			reqFields = r
			hasRequired = len(r) > 0
		case []any:
			for _, item := range r {
				if s, ok := item.(string); ok {
					reqFields = append(reqFields, s)
				}
			}
			hasRequired = len(reqFields) > 0
		default:
			hasRequired = true
		}
	}

	typeVal, hasType := node["type"]
	isObject := false
	if hasType {
		if s, ok := typeVal.(string); ok {
			if strings.EqualFold(s, "object") {
				isObject = true
			}
		}
	}

	// Rule 1: A node with properties or required MUST have type: object
	if hasProps || hasRequired {
		if !isObject {
			return fmt.Errorf("%w: node %q with properties or required must have type 'object', got %v",
				ErrInvalidToolSchema, path, typeVal)
		}
	}

	// Rule 2: In an object node, all required fields MUST be defined in properties
	if hasRequired {
		if propsMap == nil {
			return fmt.Errorf("%w: node %q specifies required fields %v but defines no properties",
				ErrInvalidToolSchema, path, reqFields)
		}
		for _, rf := range reqFields {
			if _, exists := propsMap[rf]; !exists {
				return fmt.Errorf("%w: node %q required field %q is not defined in properties",
					ErrInvalidToolSchema, path, rf)
			}
		}
	}

	// Recurse into properties
	if propsMap != nil {
		for propName, propVal := range propsMap {
			if subMap, ok := propVal.(map[string]any); ok {
				subPath := propName
				if path != "" {
					subPath = path + "." + propName
				}
				if err := validateSchemaNode(subPath, subMap); err != nil {
					return err
				}
			}
		}
	}

	// Recurse into items
	if items, ok := node["items"].(map[string]any); ok {
		subPath := path + "[items]"
		if err := validateSchemaNode(subPath, items); err != nil {
			return err
		}
	}

	// Recurse into definitions / $defs
	for _, defKey := range []string{"definitions", "$defs"} {
		if defs, ok := node[defKey].(map[string]any); ok {
			for defName, defVal := range defs {
				if subMap, ok := defVal.(map[string]any); ok {
					subPath := path + "." + defKey + "." + defName
					if err := validateSchemaNode(subPath, subMap); err != nil {
						return err
					}
				}
			}
		}
	}

	return nil
}

// -----------------------------------------------------------------------------
// Google CloudCode Generation
// -----------------------------------------------------------------------------

type cloudCodePart struct {
	Text             string                 `json:"text,omitempty"`
	Thought          bool                   `json:"thought,omitempty"`
	FunctionCall     *cloudCodeFunctionCall `json:"functionCall,omitempty"`
	FunctionResponse *cloudCodeFuncResponse `json:"functionResponse,omitempty"`
}

type cloudCodeFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type cloudCodeFuncResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type cloudCodeContent struct {
	Role  string          `json:"role"`
	Parts []cloudCodePart `json:"parts"`
}

type cloudCodeFunctionDecl struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type cloudCodeTool struct {
	FunctionDeclarations []cloudCodeFunctionDecl `json:"functionDeclarations,omitempty"`
}

type cloudCodeInnerRequest struct {
	SessionID        string             `json:"sessionId,omitempty"`
	Contents         []cloudCodeContent `json:"contents"`
	Tools            []cloudCodeTool    `json:"tools,omitempty"`
	GenerationConfig map[string]any     `json:"generationConfig,omitempty"`
}

type cloudCodeRequest struct {
	Project     string                `json:"project"`
	Model       string                `json:"model,omitempty"`
	UserAgent   string                `json:"userAgent"`
	RequestType string                `json:"requestType"`
	RequestID   string                `json:"requestId"`
	Request     cloudCodeInnerRequest `json:"request"`
}

type cloudCodeCandidate struct {
	Content struct {
		Role  string `json:"role"`
		Parts []struct {
			Text             string                 `json:"text,omitempty"`
			Thought          any                    `json:"thought,omitempty"`
			ThoughtSignature string                 `json:"thoughtSignature,omitempty"`
			FunctionCall     *cloudCodeFunctionCall `json:"functionCall,omitempty"`
		} `json:"parts"`
	} `json:"content"`
	FinishReason     string          `json:"finishReason,omitempty"`
	ThoughtSignature string          `json:"thoughtSignature,omitempty"`
	ReasoningState   json.RawMessage `json:"reasoningState,omitempty"`
}

type cloudCodeUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type cloudCodeResponse struct {
	Response *struct {
		Candidates    []cloudCodeCandidate    `json:"candidates"`
		UsageMetadata *cloudCodeUsageMetadata `json:"usageMetadata"`
	} `json:"response"`
	Candidates    []cloudCodeCandidate    `json:"candidates"`
	UsageMetadata *cloudCodeUsageMetadata `json:"usageMetadata"`
}

func (r *cloudCodeResponse) getCandidates() []cloudCodeCandidate {
	if r.Response != nil && len(r.Response.Candidates) > 0 {
		return r.Response.Candidates
	}
	return r.Candidates
}

func (r *cloudCodeResponse) getUsage() *cloudCodeUsageMetadata {
	if r.Response != nil && r.Response.UsageMetadata != nil {
		return r.Response.UsageMetadata
	}
	return r.UsageMetadata
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
	return "", errors.New("antigravity: no access token available")
}

func (p *Provider) buildRequest(msgs []*ir.Message, cfg *provider.RequestConfig) (*cloudCodeRequest, error) {
	project := p.projectID
	if project == "" {
		project = "glossy-resolver-82hmx"
	}

	var contents []cloudCodeContent
	for _, m := range msgs {
		if m == nil {
			continue
		}
		role := "user"
		if m.Role == "assistant" || m.Role == "model" {
			role = "model"
		}

		var parts []cloudCodePart
		for _, blk := range m.Blocks {
			if blk == nil {
				continue
			}
			switch b := blk.(type) {
			case ir.TextBlock:
				parts = append(parts, cloudCodePart{Text: b.Text})
			case *ir.TextBlock:
				parts = append(parts, cloudCodePart{Text: b.Text})
			case ir.ReasoningBlock:
				parts = append(parts, cloudCodePart{Text: b.Text, Thought: true})
			case *ir.ReasoningBlock:
				parts = append(parts, cloudCodePart{Text: b.Text, Thought: true})
			case ir.ToolCallBlock:
				var args map[string]any
				_ = json.Unmarshal([]byte(b.Arguments), &args)
				parts = append(parts, cloudCodePart{
					FunctionCall: &cloudCodeFunctionCall{
						Name: b.Name,
						Args: args,
					},
				})
			case *ir.ToolCallBlock:
				var args map[string]any
				_ = json.Unmarshal([]byte(b.Arguments), &args)
				parts = append(parts, cloudCodePart{
					FunctionCall: &cloudCodeFunctionCall{
						Name: b.Name,
						Args: args,
					},
				})
			case ir.ToolResultBlock:
				parts = append(parts, cloudCodePart{
					FunctionResponse: &cloudCodeFuncResponse{
						Name:     b.Name,
						Response: map[string]any{"result": b.Content},
					},
				})
			case *ir.ToolResultBlock:
				parts = append(parts, cloudCodePart{
					FunctionResponse: &cloudCodeFuncResponse{
						Name:     b.Name,
						Response: map[string]any{"result": b.Content},
					},
				})
			}
		}

		contents = append(contents, cloudCodeContent{
			Role:  role,
			Parts: parts,
		})
	}

	var tools []cloudCodeTool
	if cfg.ExtraBody != nil {
		if tList, ok := cfg.ExtraBody["tools"].([]any); ok {
			var decls []cloudCodeFunctionDecl
			for _, item := range tList {
				if itemMap, ok := item.(map[string]any); ok {
					name, _ := itemMap["name"].(string)
					desc, _ := itemMap["description"].(string)
					params, _ := itemMap["parameters"].(map[string]any)

					// Strict schema sanitizer
					if err := ValidateToolSchema(params); err != nil {
						return nil, err
					}

					decls = append(decls, cloudCodeFunctionDecl{
						Name:        name,
						Description: desc,
						Parameters:  params,
					})
				}
			}
			if len(decls) > 0 {
				tools = append(tools, cloudCodeTool{FunctionDeclarations: decls})
			}
		}
	}

	genCfg := make(map[string]any)
	if cfg.MaxTokens > 0 && !strings.Contains(cfg.Model, "flash") && !strings.Contains(cfg.Model, "pro") {
		genCfg["maxOutputTokens"] = cfg.MaxTokens
	}
	if cfg.Temperature != nil {
		genCfg["temperature"] = *cfg.Temperature
	}

	var firstUserText string
	for _, m := range msgs {
		if m != nil && m.Role == "user" {
			for _, blk := range m.Blocks {
				if tb, ok := blk.(ir.TextBlock); ok && tb.Text != "" {
					firstUserText = tb.Text
					break
				}
				if tb, ok := blk.(*ir.TextBlock); ok && tb.Text != "" {
					firstUserText = tb.Text
					break
				}
			}
			if firstUserText != "" {
				break
			}
		}
	}

	sessionID := fmt.Sprintf("-%d", time.Now().UnixNano())
	if firstUserText != "" {
		h := sha256.Sum256([]byte(firstUserText))
		val := int64(binary.BigEndian.Uint64(h[:8])) & 0x7FFFFFFFFFFFFFFF
		sessionID = "-" + strconv.FormatInt(val, 10)
	}
	reqID := fmt.Sprintf("agent-%d-%d", time.Now().UnixMilli(), time.Now().Nanosecond())

	model := cfg.Model
	if model == "" {
		model = "gemini-3.7-flash-high"
	}

	return &cloudCodeRequest{
		Project:     project,
		Model:       model,
		UserAgent:   "antigravity",
		RequestType: "agent",
		RequestID:   reqID,
		Request: cloudCodeInnerRequest{
			SessionID:        sessionID,
			Contents:         contents,
			Tools:            tools,
			GenerationConfig: genCfg,
		},
	}, nil
}

// Generate implements provider.InferenceProvider.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	cfg := provider.NewRequestConfig(opts...)

	tok, err := p.getToken(ctx)
	if err != nil {
		return nil, err
	}

	reqBody, err := p.buildRequest(msgs, cfg)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("antigravity: failed to marshal request: %w", err)
	}

	endpoint := p.baseURL + "/v1internal:generateContent"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("antigravity: failed to create request: %w", err)
	}

	req.Header.Set("User-Agent", UserAgent)
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("antigravity: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("antigravity: failed to read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("antigravity: upstream returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	var ccResp cloudCodeResponse
	if err := json.Unmarshal(body, &ccResp); err != nil {
		return nil, fmt.Errorf("antigravity: failed to parse response: %w", err)
	}

	irResp := &ir.Response{FinishReason: "stop"}
	cands := ccResp.getCandidates()
	if len(cands) > 0 {
		cand := cands[0]
		if cand.FinishReason != "" {
			irResp.FinishReason = strings.ToLower(cand.FinishReason)
		}

		var blocks []ir.Block
		sig := cand.ThoughtSignature

		for _, part := range cand.Content.Parts {
			isThought := false
			if b, ok := part.Thought.(bool); ok && b {
				isThought = true
			} else if s, ok := part.Thought.(string); ok && s != "" {
				isThought = true
			}

			if part.ThoughtSignature != "" && sig == "" {
				sig = part.ThoughtSignature
			}

			if isThought {
				blocks = append(blocks, ir.ReasoningBlock{
					ReasoningKind: ir.ReasoningText,
					Text:          part.Text,
					Signature:     sig,
					Opaque:        cand.ReasoningState,
				})
			} else if part.Text != "" {
				blocks = append(blocks, ir.TextBlock{Text: part.Text})
			}

			if part.FunctionCall != nil {
				argsJSON, _ := json.Marshal(part.FunctionCall.Args)
				blocks = append(blocks, ir.ToolCallBlock{
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				})
			}
		}

		irResp.Message = &ir.Message{
			Role:   "assistant",
			Blocks: blocks,
		}
	}

	if usage := ccResp.getUsage(); usage != nil {
		irResp.Usage = &ir.Usage{
			PromptTokens:     usage.PromptTokenCount,
			CompletionTokens: usage.CandidatesTokenCount,
			TotalTokens:      usage.TotalTokenCount,
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

	reqBody, err := p.buildRequest(msgs, cfg)
	if err != nil {
		return nil, err
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("antigravity: failed to marshal stream request: %w", err)
	}

	endpoint := p.baseURL + "/v1internal:streamGenerateContent?alt=sse"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("antigravity: failed to create stream request: %w", err)
	}

	req.Header.Set("User-Agent", UserAgent)
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("antigravity: stream request failed: %w", err)
	}

	// Synchronous error check on non-2xx
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("antigravity: upstream stream returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	outCh := make(chan ir.Event, 64)

	go func() {
		defer close(outCh)
		defer resp.Body.Close()

		scanner := sse.NewScanner(resp.Body)
		blockIndex := 0

		for scanner.Scan() {
			ev := scanner.Event()
			data := bytes.TrimSpace(ev.Data)
			if len(data) == 0 {
				continue
			}

			var chunk cloudCodeResponse
			if err := json.Unmarshal(data, &chunk); err != nil {
				continue
			}

			if usage := chunk.getUsage(); usage != nil {
				outCh <- ir.EventUsageUpdate{
					PromptTokens:     usage.PromptTokenCount,
					CompletionTokens: usage.CandidatesTokenCount,
					TotalTokens:      usage.TotalTokenCount,
				}
			}

			for _, cand := range chunk.getCandidates() {
				if cand.ThoughtSignature != "" {
					outCh <- ir.EventReasoningSignature{
						Index:     blockIndex,
						Signature: cand.ThoughtSignature,
					}
				}

				for _, part := range cand.Content.Parts {
					isThought := false
					if b, ok := part.Thought.(bool); ok && b {
						isThought = true
					} else if s, ok := part.Thought.(string); ok && s != "" {
						isThought = true
					}

					if part.ThoughtSignature != "" {
						outCh <- ir.EventReasoningSignature{
							Index:     blockIndex,
							Signature: part.ThoughtSignature,
						}
					}

					if isThought {
						outCh <- ir.EventReasoningDelta{
							Index: blockIndex,
							Text:  part.Text,
						}
					} else if part.Text != "" {
						outCh <- ir.EventTextDelta{
							Index: blockIndex,
							Text:  part.Text,
						}
					}

					if part.FunctionCall != nil {
						argsJSON, _ := json.Marshal(part.FunctionCall.Args)
						outCh <- ir.EventToolCallStart{
							Index: blockIndex,
							ID:    part.FunctionCall.Name,
							Name:  part.FunctionCall.Name,
						}
						outCh <- ir.EventToolArgumentsDelta{
							Index:     blockIndex,
							Arguments: string(argsJSON),
						}
					}
				}

				if cand.FinishReason != "" {
					outCh <- ir.EventMessageStop{
						FinishReason: strings.ToLower(cand.FinishReason),
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

type quotaBucket struct {
	BucketID          string   `json:"bucketId"`
	RemainingFraction *float64 `json:"remainingFraction"`
	ResetTime         string   `json:"resetTime"`
	Window            string   `json:"window"`
}

type quotaGroup struct {
	DisplayName string        `json:"displayName"`
	Description string        `json:"description,omitempty"`
	Buckets     []quotaBucket `json:"buckets"`
}

type quotaSummaryResponse struct {
	Groups      []quotaGroup `json:"groups"`
	Description string       `json:"description,omitempty"`
}

// Quota implements provider.QuotaProvider.
func (p *Provider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	tok, err := p.getToken(ctx)
	if err != nil {
		return nil, err
	}

	endpoint := p.baseURL + "/v1internal:retrieveUserQuotaSummary"
	bodyBytes := []byte("{}")
	if p.projectID != "" {
		bodyBytes, _ = json.Marshal(map[string]string{"project": p.projectID})
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("antigravity: failed to create quota request: %w", err)
	}

	req.Header.Set("User-Agent", UserAgent)
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("antigravity: quota request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("antigravity: failed to read quota response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("antigravity: quota endpoint returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	return ParseQuotaSummaryJSON(body)
}

// ParseQuotaSummaryJSON parses retrieveUserQuotaSummary response into QuotaSnapshot.
func ParseQuotaSummaryJSON(data []byte) (*provider.QuotaSnapshot, error) {
	var resp quotaSummaryResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("antigravity: failed to parse quota response: %w", err)
	}

	now := time.Now().UTC()
	var windows []provider.QuotaWindow
	var detailParts []string

	for _, g := range resp.Groups {
		groupName := g.DisplayName
		if groupName == "" {
			groupName = "Models"
		}

		for _, b := range g.Buckets {
			if b.RemainingFraction == nil {
				continue
			}

			remFrac := *b.RemainingFraction
			remPct := math.Max(0, math.Min(100, remFrac*100.0))
			usedPct := math.Round((100.0-remPct)*100.0) / 100.0

			var resetTime time.Time
			if b.ResetTime != "" {
				if t, err := time.Parse(time.RFC3339, b.ResetTime); err == nil {
					resetTime = t.UTC()
				}
			}

			var secondsRemaining int64
			if !resetTime.IsZero() && resetTime.After(now) {
				secondsRemaining = int64(resetTime.Sub(now).Seconds())
			}

			winLabel := b.Window
			if winLabel == "" {
				winLabel = "quota"
			}
			label := fmt.Sprintf("%s · %s", groupName, winLabel)

			windows = append(windows, provider.QuotaWindow{
				Label:            label,
				UsedPct:          usedPct,
				Remaining:        remPct,
				Limit:            100,
				Unit:             "%",
				ResetAt:          resetTime,
				SecondsRemaining: secondsRemaining,
			})

			detailParts = append(detailParts, fmt.Sprintf("%s: %.1f%% remaining", label, remPct))
		}
	}

	detail := strings.Join(detailParts, " · ")
	if detail == "" {
		detail = "Antigravity live quota"
	}

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
	cfg := oauth.DeviceFlowConfig{
		ClientID:      p.clientID,
		ClientSecret:  p.clientSecret,
		Scope:         "https://www.googleapis.com/auth/cloud-platform",
		DeviceAuthURL: p.deviceAuthURL,
		TokenURL:      p.tokenURL,
		HTTPClient:    p.httpClient,
	}

	dcr, err := oauth.RequestDeviceCode(ctx, cfg)
	if err != nil {
		return fmt.Errorf("antigravity: login device code failed: %w", err)
	}

	tokResp, err := oauth.PollToken(ctx, cfg, dcr.DeviceCode, dcr.Interval)
	if err != nil {
		return fmt.Errorf("antigravity: login token poll failed: %w", err)
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
			Provider:     "antigravity",
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
	if p.authManager != nil {
		cred, err := p.authManager.Get(ctx, p.clientID)
		if err != nil {
			return err
		}
		if cred.RefreshToken != "" {
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
			return p.authManager.Store(ctx, p.clientID, newCred)
		}
	}
	return nil
}
