package freebuff

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/internal/openai"
	spikesfreebuff "github.com/smhanov/ultiproxy/pkg/spikes/freebuff"
)

const (
	DefaultBaseURL = "https://www.codebuff.com/api/v1"
	ProviderName   = "freebuff"
	UserAgentValue = "ai-sdk/openai-compatible/0.0.0-test/codebuff ai-sdk/provider-utils/3.0.25 runtime/node.js/v22.23.2"
)

// DefaultModels maps model names to agent IDs.
var DefaultModels = map[string]string{
	"deepseek/deepseek-v4-flash": "base3-free-deepseek-flash",
	"z-ai/glm-5.3-flash":         "base3-free-glm-5-3-flash",
	"openai/gpt-5.6-luna":        "base3-free-luna",
	"upstage/solar-pro4":         "upstage/solar-pro4",
	"minimax/minimax-m3":         "minimax/minimax-m3",
	"crof/kimi-k3-eco":           "crof/kimi-k3-eco",
	"mimo/mimo-v2.5":             "mimo/mimo-v2.5",
}

// DefaultAliases maps model aliases to canonical models.
var DefaultAliases = map[string]string{
	"deepseek-v4-flash": "deepseek/deepseek-v4-flash",
	"deepseek-flash":    "deepseek/deepseek-v4-flash",
	"glm-5.3-flash":     "z-ai/glm-5.3-flash",
	"gpt-5.6-luna":      "openai/gpt-5.6-luna",
	"luna":              "openai/gpt-5.6-luna",
	"solar-pro4":        "upstage/solar-pro4",
	"minimax-m3":        "minimax/minimax-m3",
	"kimi-k3-eco":       "crof/kimi-k3-eco",
	"mimo-v2.5":         "mimo/mimo-v2.5",
}

// Config holds configuration for the Freebuff adapter.
type Config struct {
	BaseURL    string
	Token      string
	DataDir    string
	LockPath   string
	HTTPClient *http.Client
	Models     map[string]string
	Aliases    map[string]string
	Actor      *spikesfreebuff.FreebuffAccountActor
}

// Provider implements provider.InferenceProvider and provider.QuotaProvider.
type Provider struct {
	cfg            Config
	actor          *spikesfreebuff.FreebuffAccountActor
	models         map[string]string
	aliases        map[string]string
	instanceID     string
	instanceIDFile string
}

// New creates a new Freebuff provider.
func New(cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")

	if cfg.Token == "" {
		cfg.Token = os.Getenv("FREEBUFF_TOKEN")
	}

	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}

	models := make(map[string]string)
	for k, v := range DefaultModels {
		models[k] = v
	}
	for k, v := range cfg.Models {
		models[k] = v
	}

	aliases := make(map[string]string)
	for k, v := range DefaultAliases {
		aliases[k] = v
	}
	for k, v := range cfg.Aliases {
		aliases[k] = v
	}

	// Persist/load instance ID under data dir
	dataDir := cfg.DataDir
	if dataDir == "" {
		home, err := os.UserHomeDir()
		if err == nil {
			dataDir = filepath.Join(home, ".local", "state", "ultiproxy")
		} else {
			dataDir = filepath.Join(os.TempDir(), "ultiproxy")
		}
	}
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data dir: %w", err)
	}

	instanceIDFile := filepath.Join(dataDir, "freebuff_instance_id")
	instanceID := ""
	if data, err := os.ReadFile(instanceIDFile); err == nil {
		instanceID = strings.TrimSpace(string(data))
	}
	// Homemade fb-inst-* ids are rejected by Codebuff (HTTP 409). Let
	// POST /freebuff/session mint a real UUID, then persist it.
	if strings.HasPrefix(instanceID, "fb-inst-") {
		instanceID = ""
	}

	actor := cfg.Actor
	if actor == nil {
		var err error
		actor, err = spikesfreebuff.NewFreebuffAccountActor(
			cfg.LockPath,
			cfg.HTTPClient,
			cfg.Token,
			spikesfreebuff.WithBaseURL(cfg.BaseURL),
			spikesfreebuff.WithInstanceID(instanceID),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize FreebuffAccountActor: %w", err)
		}
	} else {
		actor.SetInstanceID(instanceID)
	}

	return &Provider{
		cfg:            cfg,
		actor:          actor,
		models:         models,
		aliases:        aliases,
		instanceID:     instanceID,
		instanceIDFile: instanceIDFile,
	}, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return ProviderName
}

// Actor returns the underlying FreebuffAccountActor.
func (p *Provider) Actor() *spikesfreebuff.FreebuffAccountActor {
	return p.actor
}

// InstanceID returns the persisted instance ID.
func (p *Provider) InstanceID() string {
	return p.instanceID
}

// ResolveModel maps a requested model or alias to canonical model and agent ID.
func (p *Provider) ResolveModel(requested string) (canonicalModel string, agentID string, err error) {
	if requested == "" {
		requested = "deepseek/deepseek-v4-flash"
	}
	if target, ok := p.aliases[requested]; ok {
		requested = target
	}
	agentID, ok := p.models[requested]
	if !ok {
		// Default to deepseek if unknown
		return requested, "base3-free-deepseek-flash", nil
	}
	return requested, agentID, nil
}

// Capabilities returns Freebuff capabilities.
func Capabilities() provider.Capabilities {
	return provider.Capabilities{
		Chat:                  true,
		Streaming:             true,
		Tools:                 true,
		MaxConcurrentRequests: 1,
		SessionAffinity:       true,
		Queueing:              true,
	}
}

// Provider returns the provider.Provider bundle.
func (p *Provider) Provider() provider.Provider {
	return provider.Provider{
		Inference:    p,
		Quota:        p,
		Auth:         p,
		Capabilities: Capabilities(),
	}
}

// Generate executes a non-streaming chat completion.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	model, agentID, err := p.ResolveModel(reqConfig.Model)
	if err != nil {
		return nil, err
	}

	if err := p.actor.Acquire(ctx); err != nil {
		return nil, err
	}
	defer p.actor.Release()

	if p.actor.BoundModel() != model {
		if err := p.actor.Bind(ctx, model); err != nil {
			return nil, fmt.Errorf("failed to bind model: %w", err)
		}
		p.persistInstanceID()
	}

	run, err := p.actor.StartRun(ctx, agentID)
	if err != nil {
		return nil, fmt.Errorf("failed to start agent run: %w", err)
	}

	reqBody, err := p.buildPayload(msgs, model, run.GetRunID(), false, reqConfig)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/chat/completions", reqBody)
	if err != nil {
		return nil, err
	}
	p.setHeaders(req, reqConfig)

	return openai.ExecuteGenerate(ctx, p.cfg.HTTPClient, req)
}

// Stream executes a streaming chat completion.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	reqConfig := provider.NewRequestConfig(opts...)
	model, agentID, err := p.ResolveModel(reqConfig.Model)
	if err != nil {
		return nil, err
	}

	if err := p.actor.Acquire(ctx); err != nil {
		return nil, err
	}

	if p.actor.BoundModel() != model {
		if err := p.actor.Bind(ctx, model); err != nil {
			_ = p.actor.Release()
			return nil, fmt.Errorf("failed to bind model: %w", err)
		}
		p.persistInstanceID()
	}

	run, err := p.actor.StartRun(ctx, agentID)
	if err != nil {
		_ = p.actor.Release()
		return nil, fmt.Errorf("failed to start agent run: %w", err)
	}

	reqBody, err := p.buildPayload(msgs, model, run.GetRunID(), true, reqConfig)
	if err != nil {
		_ = p.actor.Release()
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/chat/completions", reqBody)
	if err != nil {
		_ = p.actor.Release()
		return nil, err
	}
	p.setHeaders(req, reqConfig)

	// Stream through actor's serialized queue
	body, err := p.actor.DoStream(ctx, req)
	if err != nil {
		_ = p.actor.Release()
		return nil, err
	}

	rawCh := openai.StreamHandler(ctx, body)
	outCh := make(chan ir.Event, 64)

	go func() {
		defer close(outCh)
		defer p.actor.Release()

		for ev := range rawCh {
			outCh <- ev
		}
	}()

	return outCh, nil
}

func (p *Provider) buildPayload(msgs []*ir.Message, model, runID string, stream bool, cfg *provider.RequestConfig) (io.Reader, error) {
	chatMsgs := openai.ConvertMessages(msgs, openai.ConvertOptions{})
	hasSystem := false
	for _, m := range chatMsgs {
		if m.Role == "system" {
			hasSystem = true
			break
		}
	}
	if !hasSystem {
		chatMsgs = append([]openai.ChatMessage{{Role: "system", Content: "You are Buffy, the coding agent behind Codebuff."}}, chatMsgs...)
	}

	inst := p.actor.InstanceID()
	if inst == "" {
		inst = p.instanceID
	}

	defaultTool := map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "read_files",
			"description": "Read files from project",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"paths": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				},
			},
		},
	}

	var finalTools []any
	if cfg.ExtraBody != nil {
		if tools, ok := cfg.ExtraBody["tools"].([]any); ok && len(tools) > 0 {
			finalTools = append(finalTools, defaultTool)
			finalTools = append(finalTools, tools...)
		}
	}
	if len(finalTools) == 0 {
		finalTools = []any{defaultTool}
	}

	payload := map[string]any{
		"model":    model,
		"messages": chatMsgs,
		"stream":   stream,
		"tools":    finalTools,
		"codebuff_metadata": map[string]any{
			"run_id":               runID,
			"freebuff_instance_id": inst,
			"cost_mode":            "free",
			"client_id":            "cli-" + inst,
			"llm_step_number":      "1",
		},
		"provider": map[string]any{
			"allow_fallbacks": true,
		},
	}

	if cfg.MaxTokens > 0 {
		payload["max_tokens"] = cfg.MaxTokens
	}
	if cfg.Temperature != nil {
		payload["temperature"] = *cfg.Temperature
	}
	for k, v := range cfg.ExtraBody {
		if k == "tools" {
			continue
		}
		payload[k] = v
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(data), nil
}

func (p *Provider) persistInstanceID() {
	id := p.actor.InstanceID()
	if id == "" {
		return
	}
	p.instanceID = id
	if p.instanceIDFile == "" {
		return
	}
	_ = os.WriteFile(p.instanceIDFile, []byte(id+"\n"), 0o600)
}

func (p *Provider) setHeaders(req *http.Request, cfg *provider.RequestConfig) {
	req.Header.Set("User-Agent", UserAgentValue)
	req.Header.Set("Content-Type", "application/json")
	inst := p.actor.InstanceID()
	if inst == "" {
		inst = p.instanceID
	}
	if inst != "" {
		req.Header.Set("x-freebuff-instance-id", inst)
	}
	req.Header.Set("x-freebuff-acting-user-id", "adcc6f59-fffd-4735-8c09-703eb3158941")
	tok := p.cfg.Token
	if tok == "" {
		tok = p.actor.Token()
	}
	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}
}

type usagePayload struct {
	DayUsed    float64 `json:"dayUsed"`
	DayLimit   float64 `json:"dayLimit"`
	WeekUsed   float64 `json:"weekUsed"`
	WeekLimit  float64 `json:"weekLimit"`
	MonthUsed  float64 `json:"monthUsed"`
	MonthLimit float64 `json:"monthLimit"`
	ResetAt    string  `json:"resetAt"`

	// snake_case fallback
	DayUsedSnake    float64 `json:"day_used"`
	DayLimitSnake   float64 `json:"day_limit"`
	WeekUsedSnake   float64 `json:"week_used"`
	WeekLimitSnake  float64 `json:"week_limit"`
	MonthUsedSnake  float64 `json:"month_used"`
	MonthLimitSnake float64 `json:"month_limit"`
	ResetAtSnake    string  `json:"reset_at"`
}

// Quota queries usage via the actor and returns normalized QuotaSnapshot.
func (p *Provider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	// Call POST /usage {fingerprintId: cli-usage} and GET /freebuff/session via actor
	rawUsage, err := p.actor.FetchUsage(ctx, "cli-usage")
	if err != nil {
		return nil, fmt.Errorf("failed to fetch usage from actor: %w", err)
	}

	sess, err := p.actor.GetSession(ctx)
	if err != nil {
		// Session get is optional/non-fatal if usage succeeded
		sess = &spikesfreebuff.Session{}
	}

	snapshot, err := ParseUsageSnapshot(rawUsage, time.Now())
	if err != nil {
		return nil, err
	}

	snapshot.Detail = fmt.Sprintf("instance: %s, model: %s", sess.InstanceID, sess.Model)
	return snapshot, nil
}

// ParseUsageSnapshot parses Freebuff usage response bytes into a QuotaSnapshot.
func ParseUsageSnapshot(data []byte, now time.Time) (*provider.QuotaSnapshot, error) {
	var u usagePayload
	if err := json.Unmarshal(data, &u); err != nil {
		return nil, fmt.Errorf("failed to parse freebuff usage json: %w", err)
	}

	dayUsed := u.DayUsed
	if dayUsed == 0 && u.DayUsedSnake != 0 {
		dayUsed = u.DayUsedSnake
	}
	dayLimit := u.DayLimit
	if dayLimit == 0 && u.DayLimitSnake != 0 {
		dayLimit = u.DayLimitSnake
	}

	weekUsed := u.WeekUsed
	if weekUsed == 0 && u.WeekUsedSnake != 0 {
		weekUsed = u.WeekUsedSnake
	}
	weekLimit := u.WeekLimit
	if weekLimit == 0 && u.WeekLimitSnake != 0 {
		weekLimit = u.WeekLimitSnake
	}

	monthUsed := u.MonthUsed
	if monthUsed == 0 && u.MonthUsedSnake != 0 {
		monthUsed = u.MonthUsedSnake
	}
	monthLimit := u.MonthLimit
	if monthLimit == 0 && u.MonthLimitSnake != 0 {
		monthLimit = u.MonthLimitSnake
	}

	resetStr := u.ResetAt
	if resetStr == "" {
		resetStr = u.ResetAtSnake
	}
	var resetAt time.Time
	var secRem int64
	if resetStr != "" {
		if t, err := time.Parse(time.RFC3339, resetStr); err == nil {
			resetAt = t
			secRem = int64(t.Sub(now).Seconds())
			if secRem < 0 {
				secRem = 0
			}
		}
	}

	calcPct := func(used, limit float64) float64 {
		if limit <= 0 {
			return 0
		}
		return (used / limit) * 100.0
	}

	windows := []provider.QuotaWindow{
		{
			Label:            "Daily",
			UsedPct:          calcPct(dayUsed, dayLimit),
			Remaining:        dayLimit - dayUsed,
			Limit:            dayLimit,
			Unit:             "requests",
			ResetAt:          resetAt,
			SecondsRemaining: secRem,
		},
		{
			Label:     "Weekly",
			UsedPct:   calcPct(weekUsed, weekLimit),
			Remaining: weekLimit - weekUsed,
			Limit:     weekLimit,
			Unit:      "requests",
		},
		{
			Label:     "Monthly",
			UsedPct:   calcPct(monthUsed, monthLimit),
			Remaining: monthLimit - monthUsed,
			Limit:     monthLimit,
			Unit:      "requests",
		},
	}

	return &provider.QuotaSnapshot{
		ObservedAt: now,
		Windows:    windows,
	}, nil
}

// Close closes the underlying actor.
func (p *Provider) Close() error {
	if p.actor != nil {
		return p.actor.Close()
	}
	return nil
}

// ----------------------------------------------------------------------------
// AuthProvider — imports the Freebuff CLI token; there is no OAuth dance.
// ----------------------------------------------------------------------------

const freebuffAuthKey = "freebuff"

func manicodeCredentialsPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "manicode", "credentials.json")
}

// ReadCLIToken loads the Codebuff / Freebuff CLI auth token from
// ~/.config/manicode/credentials.json. Never prints the token.
func ReadCLIToken() (token, email, userID string, err error) {
	data, err := os.ReadFile(manicodeCredentialsPath())
	if err != nil {
		return "", "", "", fmt.Errorf("freebuff: CLI credentials not found (%s): %w — run `freebuff` and sign in first", manicodeCredentialsPath(), err)
	}
	var parsed map[string]struct {
		ID        string `json:"id"`
		Email     string `json:"email"`
		AuthToken string `json:"authToken"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return "", "", "", fmt.Errorf("freebuff: parse CLI credentials: %w", err)
	}
	entry, ok := parsed["default"]
	if !ok {
		for _, v := range parsed {
			entry = v
			ok = true
			break
		}
	}
	if !ok || entry.AuthToken == "" {
		return "", "", "", fmt.Errorf("freebuff: no authToken in CLI credentials")
	}
	return entry.AuthToken, entry.Email, entry.ID, nil
}

func (p *Provider) Login(ctx context.Context) error {
	tok, email, userID, err := ReadCLIToken()
	if err != nil {
		return err
	}
	p.cfg.Token = tok
	if p.actor != nil {
		p.actor.SetToken(tok)
	}
	_ = email
	_ = userID
	return nil
}

func (p *Provider) Token(ctx context.Context) (string, error) {
	if p.cfg.Token != "" {
		return p.cfg.Token, nil
	}
	if p.actor != nil {
		if t := p.actor.Token(); t != "" {
			return t, nil
		}
	}
	tok, _, _, err := ReadCLIToken()
	if err != nil {
		return "", err
	}
	p.cfg.Token = tok
	if p.actor != nil {
		p.actor.SetToken(tok)
	}
	return tok, nil
}

func (p *Provider) Refresh(ctx context.Context) error {
	_, err := p.Token(ctx)
	return err
}
