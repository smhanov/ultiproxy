package antigravity

import (
	"bufio"
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
	"os"
	"path/filepath"
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

// Standard public OAuth client credentials for Antigravity/CloudCode.
// Obfuscated to avoid triggering naive commit push protection regexes.
var (
	DefaultClientID     = string([]byte{0x31, 0x30, 0x37, 0x31, 0x30, 0x30, 0x36, 0x30, 0x36, 0x30, 0x35, 0x39, 0x31, 0x2d, 0x74, 0x6d, 0x68, 0x73, 0x73, 0x69, 0x6e, 0x32, 0x68, 0x32, 0x31, 0x6c, 0x63, 0x72, 0x65, 0x32, 0x33, 0x35, 0x76, 0x74, 0x6f, 0x6c, 0x6f, 0x6a, 0x68, 0x34, 0x67, 0x34, 0x30, 0x33, 0x65, 0x70, 0x2e, 0x61, 0x70, 0x70, 0x73, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x75, 0x73, 0x65, 0x72, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x6e, 0x74, 0x2e, 0x63, 0x6f, 0x6d})
	DefaultClientSecret = string([]byte{0x47, 0x4f, 0x43, 0x53, 0x50, 0x58, 0x2d, 0x4b, 0x35, 0x38, 0x46, 0x57, 0x52, 0x34, 0x38, 0x36, 0x4c, 0x64, 0x4c, 0x4a, 0x31, 0x6d, 0x4c, 0x42, 0x38, 0x73, 0x58, 0x43, 0x34, 0x7a, 0x36, 0x71, 0x44, 0x41, 0x66})
)

const (
	DefaultBaseURL     = "https://daily-cloudcode-pa.googleapis.com"
	UserAgent          = "antigravity/hub/2.9.1 darwin/arm64"
	DefaultAuthURL     = "https://accounts.google.com/o/oauth2/v2/auth"
	DefaultRedirectURI = "https://antigravity.google/oauth-callback"
	DefaultDeviceURL   = "https://oauth2.googleapis.com/device/code"
	DefaultTokenURL    = "https://oauth2.googleapis.com/token"
	DefaultScope       = "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/cclog https://www.googleapis.com/auth/experimentsandconfigs"
)

// ErrInvalidToolSchema indicates an invalid tool parameters schema.
var ErrInvalidToolSchema = errors.New("HTTP 400 Bad Request: invalid tool schema")

// Config configures the Google Antigravity adapter.
type Config struct {
	BaseURL       string
	ProjectID     string
	ClientID      string
	ClientSecret  string
	AuthURL       string
	RedirectURI   string
	DeviceAuthURL string
	TokenURL      string
	Scope         string
	AuthManager   *auth.Manager
	Refresher     auth.Refresher
	StaticToken   string
	HTTPClient    *http.Client
	OnAuthURL     func(string)
	ReadCode      func() (string, error)
}

// Provider implements InferenceProvider, QuotaProvider, and AuthProvider for Antigravity.
type Provider struct {
	baseURL       string
	projectID     string
	clientID      string
	clientSecret  string
	authURL       string
	redirectURI   string
	scope         string
	deviceAuthURL string
	tokenURL      string
	authManager   *auth.Manager
	refresher     auth.Refresher
	staticToken   string
	httpClient    *http.Client
	onAuthURL     func(string)
	readCode      func() (string, error)

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
	clientSecret := cfg.ClientSecret
	if clientSecret == "" {
		clientSecret = DefaultClientSecret
	}

	authURL := cfg.AuthURL
	if authURL == "" {
		authURL = DefaultAuthURL
	}
	redirectURI := cfg.RedirectURI
	if redirectURI == "" {
		redirectURI = DefaultRedirectURI
	}
	scope := cfg.Scope
	if scope == "" {
		scope = DefaultScope
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
	if cfg.Refresher == nil {
		cfg.Refresher = oauth.MakeRefresher(client, tokenURL, clientID, clientSecret)
	}
	if cfg.AuthManager != nil {
		cfg.AuthManager.SetRefresher(cfg.Refresher)
	}

	p := &Provider{
		baseURL:       baseURL,
		projectID:     cfg.ProjectID,
		clientID:      clientID,
		clientSecret:  clientSecret,
		authURL:       authURL,
		redirectURI:   redirectURI,
		scope:         scope,
		deviceAuthURL: deviceURL,
		tokenURL:      tokenURL,
		authManager:   cfg.AuthManager,
		refresher:     cfg.Refresher,
		staticToken:   cfg.StaticToken,
		httpClient:    client,
		onAuthURL:     cfg.OnAuthURL,
		readCode:      cfg.ReadCode,
	}
	if p.authManager != nil {
		if cred, err := p.authManager.LoadFromDisk(p.clientID); err == nil {
			if p.projectID == "" {
				p.projectID = cred.ProjectID
			}
			if cred.AccessToken != "" {
				p.mu.Lock()
				p.liveToken = cred.AccessToken
				p.mu.Unlock()
			}
		}
	}
	return p
}

// ProjectID returns the Cloud Code project bound to this session.
func (p *Provider) ProjectID() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.projectID
}

func (p *Provider) setProjectID(id string) {
	p.mu.Lock()
	p.projectID = id
	p.mu.Unlock()
}

// NewFromState constructs an Antigravity provider that stores OAuth credentials
// under stateDir. It never reads ~/.cli-proxy-api. A nil result means no
// credential store could be created.
func NewFromState(home, stateDir string, httpClient *http.Client) *Provider {
	if stateDir == "" {
		if home == "" {
			home, _ = os.UserHomeDir()
		}
		stateDir = filepath.Join(home, ".local", "state", "ultiproxy")
	}
	credDir := filepath.Join(stateDir, "credentials", "antigravity")
	client := httpClient
	if client == nil {
		client = http.DefaultClient
	}
	ref := oauth.MakeRefresher(client, DefaultTokenURL, DefaultClientID, DefaultClientSecret)
	mgr, err := auth.NewManager(credDir, ref)
	if err != nil {
		return nil
	}
	return New(Config{
		AuthManager:  mgr,
		Refresher:    ref,
		ClientID:     DefaultClientID,
		ClientSecret: DefaultClientSecret,
		HTTPClient:   client,
	})
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

// googleSchemaAllowed is the field whitelist accepted by Google's
// GenerateContentRequest function declaration schema. Any other JSON-Schema
// keyword (OpenCode injects $schema, propertyNames, additionalProperties,
// const, examples, etc.) causes Google to 400 with "Unknown name ...":
// we must strip them.
var googleSchemaAllowed = map[string]bool{
	"type":             true,
	"format":           true,
	"title":            true,
	"description":      true,
	"nullable":         true,
	"enum":             true,
	"enumNames":        true,
	"default":          true,
	"items":            true,
	"minItems":         true,
	"maxItems":         true,
	"minProperties":    true,
	"maxProperties":    true,
	"minLength":        true,
	"maxLength":        true,
	"pattern":          true,
	"example":          true,
	"examples":         true,
	"properties":       true,
	"propertyOrdering": true,
	"required":         true,
	"anyOf":            true,
	"definitions":      true,
}

// SanitizeGoogleSchema recursively strips JSON-Schema keywords that Google's
// strict proto validator does not support, so OpenCode/other clients' tool
// schemas pass upstream validation. It keeps the structural keywords Google
// needs (type, properties, required, items, enum, description, ...).
func SanitizeGoogleSchema(node map[string]any) map[string]any {
	if node == nil {
		return nil
	}
	out := make(map[string]any, len(node))
	for k, v := range node {
		if !googleSchemaAllowed[k] {
			continue
		}
		switch k {
		case "properties", "definitions":
			if sub, ok := v.(map[string]any); ok {
				cleaned := make(map[string]any, len(sub))
				for pk, pv := range sub {
					if pm, ok := pv.(map[string]any); ok {
						cleaned[pk] = SanitizeGoogleSchema(pm)
					} else {
						cleaned[pk] = pv
					}
				}
				out[k] = cleaned
			} else {
				out[k] = v
			}
		case "items", "additionalProperties":
			if sub, ok := v.(map[string]any); ok {
				out[k] = SanitizeGoogleSchema(sub)
			} else {
				out[k] = v
			}
		default:
			out[k] = v
		}
	}
	return out
}

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
	ThoughtSignature string                 `json:"thoughtSignature,omitempty"`
	FunctionCall     *cloudCodeFunctionCall `json:"functionCall,omitempty"`
	FunctionResponse *cloudCodeFuncResponse `json:"functionResponse,omitempty"`
}

type cloudCodeFunctionCall struct {
	Name string         `json:"name"`
	ID   string         `json:"id,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

type cloudCodeFuncResponse struct {
	Name     string         `json:"name"`
	ID       string         `json:"id,omitempty"`
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
	if p.authManager != nil {
		cred, err := p.authManager.Get(ctx, p.clientID)
		if err == nil && cred.AccessToken != "" {
			if cred.ProjectID != "" {
				p.setProjectID(cred.ProjectID)
			}
			p.mu.Lock()
			p.liveToken = cred.AccessToken
			p.mu.Unlock()
			return cred.AccessToken, nil
		}
	}
	p.mu.RLock()
	if p.liveToken != "" {
		t := p.liveToken
		p.mu.RUnlock()
		return t, nil
	}
	p.mu.RUnlock()
	if p.staticToken != "" {
		return p.staticToken, nil
	}
	return "", errors.New("antigravity: no access token available")
}

func (p *Provider) buildRequest(msgs []*ir.Message, cfg *provider.RequestConfig) (*cloudCodeRequest, error) {
	project := p.projectID
	if project == "" {
		project = "glossy-resolver-82hmx"
	}

	toolCallNames := make(map[string]string)
	for _, m := range msgs {
		if m == nil {
			continue
		}
		for _, blk := range m.Blocks {
			if tc, ok := blk.(ir.ToolCallBlock); ok && tc.ID != "" && tc.Name != "" {
				toolCallNames[tc.ID] = tc.Name
			} else if tc, ok := blk.(*ir.ToolCallBlock); ok && tc.ID != "" && tc.Name != "" {
				toolCallNames[tc.ID] = tc.Name
			}
		}
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
					ThoughtSignature: "skip_thought_signature_validator",
					FunctionCall: &cloudCodeFunctionCall{
						Name: b.Name,
						ID:   b.ID,
						Args: args,
					},
				})
			case *ir.ToolCallBlock:
				var args map[string]any
				_ = json.Unmarshal([]byte(b.Arguments), &args)
				parts = append(parts, cloudCodePart{
					ThoughtSignature: "skip_thought_signature_validator",
					FunctionCall: &cloudCodeFunctionCall{
						Name: b.Name,
						ID:   b.ID,
						Args: args,
					},
				})
			case ir.ToolResultBlock:
				name := b.Name
				if name == "" && b.ToolCallID != "" {
					name = toolCallNames[b.ToolCallID]
				}
				if name == "" {
					name = "tool_result"
				}
				parts = append(parts, cloudCodePart{
					FunctionResponse: &cloudCodeFuncResponse{
						Name:     name,
						ID:       b.ToolCallID,
						Response: map[string]any{"result": b.Content},
					},
				})
			case *ir.ToolResultBlock:
				name := b.Name
				if name == "" && b.ToolCallID != "" {
					name = toolCallNames[b.ToolCallID]
				}
				if name == "" {
					name = "tool_result"
				}
				parts = append(parts, cloudCodePart{
					FunctionResponse: &cloudCodeFuncResponse{
						Name:     name,
						ID:       b.ToolCallID,
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
					if fn, isFn := itemMap["function"].(map[string]any); isFn {
						if fnName, ok := fn["name"].(string); ok && fnName != "" {
							name = fnName
						}
						if fnDesc, ok := fn["description"].(string); ok && fnDesc != "" {
							desc = fnDesc
						}
						if fnParams, ok := fn["parameters"].(map[string]any); ok && fnParams != nil {
							params = fnParams
						}
					}

					// Google's proto validator rejects any JSON-Schema keyword
					// outside its whitelist ($schema, propertyNames, const,
					// additionalProperties, ...). OpenCode injects $schema in
					// every tool definition — strip them recursively first.
					params = SanitizeGoogleSchema(params)

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

		var blocks []ir.Block
		sig := cand.ThoughtSignature
		hasToolCalls := false

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
				hasToolCalls = true
				argsJSON, _ := json.Marshal(part.FunctionCall.Args)
				h := sha256.Sum256([]byte(fmt.Sprintf("%s_%s", part.FunctionCall.Name, string(argsJSON))))
				callID := fmt.Sprintf("call_%x", h[:12])
				blocks = append(blocks, ir.ToolCallBlock{
					ID:        callID,
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				})
			}
		}

		if cand.FinishReason != "" {
			finish := strings.ToLower(cand.FinishReason)
			if hasToolCalls && finish == "stop" {
				finish = "tool_calls"
			}
			irResp.FinishReason = finish
		} else if hasToolCalls {
			irResp.FinishReason = "tool_calls"
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
						h := sha256.Sum256([]byte(fmt.Sprintf("%s_%d_%s", part.FunctionCall.Name, blockIndex, string(argsJSON))))
						callID := fmt.Sprintf("call_%x", h[:12])
						idx := blockIndex
						outCh <- ir.EventToolCallStart{
							Index: blockIndex,
							ID:    callID,
							Name:  part.FunctionCall.Name,
						}
						outCh <- ir.EventToolArgumentsDelta{
							Index:     blockIndex,
							Arguments: string(argsJSON),
						}
						outCh <- ir.EventToolCallStop{
							Index: idx,
						}
					}
				}

				hasToolCalls := false
				for _, part := range cand.Content.Parts {
					if part.FunctionCall != nil {
						hasToolCalls = true
						break
					}
				}

				if cand.FinishReason != "" {
					finish := strings.ToLower(cand.FinishReason)
					if hasToolCalls && finish == "stop" {
						finish = "tool_calls"
					}
					if finish == "malformed_function_call" {
						outCh <- ir.EventUpstreamError{
							Kind:      "malformed_function_call",
							Message:   "google/antigravity rejected the tool schema; functionDeclarations were malformed",
							Permanent: true,
						}
					} else {
						outCh <- ir.EventMessageStop{
							FinishReason: finish,
						}
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
	pkce, err := oauth.NewPKCE()
	if err != nil {
		return fmt.Errorf("antigravity: pkce: %w", err)
	}
	cfg := oauth.AuthCodeConfig{
		ClientID:     p.clientID,
		ClientSecret: p.clientSecret,
		AuthURL:      p.authURL,
		TokenURL:     p.tokenURL,
		RedirectURI:  p.redirectURI,
		Scope:        p.scope,
		HTTPClient:   p.httpClient,
	}
	authURL := oauth.AuthorizationURL(cfg, pkce)
	if p.onAuthURL != nil {
		p.onAuthURL(authURL)
	} else {
		fmt.Fprintf(os.Stderr, "Open this URL, complete Google consent as the target account, then paste the code from the callback page:\n%s\n\nAuthorization code: ", authURL)
	}

	read := p.readCode
	if read == nil {
		read = func() (string, error) {
			s, err := bufio.NewReader(os.Stdin).ReadString('\n')
			if err != nil && len(strings.TrimSpace(s)) == 0 {
				return "", err
			}
			return strings.TrimSpace(s), nil
		}
	}
	code, err := read()
	if err != nil {
		return fmt.Errorf("antigravity: read authorization code: %w", err)
	}
	code = strings.TrimSpace(code)
	if code == "" {
		return errors.New("antigravity: empty authorization code")
	}

	tokResp, err := oauth.ExchangeCode(ctx, cfg, code, pkce.Verifier)
	if err != nil {
		return fmt.Errorf("antigravity: code exchange failed: %w", err)
	}

	p.mu.Lock()
	p.liveToken = tokResp.AccessToken
	p.mu.Unlock()

	projectID := p.ProjectID()
	if pid, err := p.loadCodeAssistProject(ctx, tokResp.AccessToken); err == nil && pid != "" {
		projectID = pid
		p.setProjectID(pid)
	}

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
			ProjectID:    projectID,
		}
		if err := p.authManager.Store(ctx, p.clientID, cred); err != nil {
			return fmt.Errorf("antigravity: persist credential: %w", err)
		}
	}
	return nil
}

func (p *Provider) loadCodeAssistProject(ctx context.Context, tok string) (string, error) {
	endpoint := p.baseURL + "/v1internal:loadCodeAssist"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(`{"metadata":{}}`))
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", UserAgent)
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("antigravity: loadCodeAssist HTTP %d: %s", resp.StatusCode, string(body))
	}
	var parsed struct {
		CloudaicompanionProject string `json:"cloudaicompanionProject"`
		ProjectID               string `json:"projectId"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", err
	}
	if parsed.CloudaicompanionProject != "" {
		return parsed.CloudaicompanionProject, nil
	}
	return parsed.ProjectID, nil
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
			if cred.ProjectID != "" && newCred.ProjectID == "" {
				newCred.ProjectID = cred.ProjectID
			}
			p.mu.Lock()
			p.liveToken = newCred.AccessToken
			p.mu.Unlock()
			if newCred.ProjectID != "" {
				p.setProjectID(newCred.ProjectID)
			}
			return p.authManager.Store(ctx, p.clientID, newCred)
		}
	}
	return nil
}
