package anthropichub

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/smhanov/llmhub"
	hubanthropic "github.com/smhanov/llmhub/providers/anthropic"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/modelmeta"
	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/hublane"
)

// anthropicVersion is the Messages-API version header the first-party model
// list requires (it is the same surface /v1/messages uses).
const anthropicVersion = "2023-06-01"

// modelsDiscoveryBudget bounds one FetchModels call, matching the budget the
// server's discovery passes use.
const modelsDiscoveryBudget = 5 * time.Second

// Provider implements provider.InferenceProvider, backed by llmhub.
//
// It also implements the model-discovery surfaces (FetchModels,
// CachedModels, CachedModelInfo, ModelDiscoveryEnabled) so the lane's cached
// model list - ids with the windows and modalities Anthropic reports - feeds
// the aggregated GET /v1/models exactly like an OpenAI-compatible lane. The
// cache is filled by discovery only; listing never dials the upstream.
type Provider struct {
	adapter    *hublane.Adapter
	baseURL    string
	apiKey     string
	httpClient *http.Client

	mu        sync.RWMutex
	models    []string
	modelInfo []provider.ModelInfo
}

// New creates the Anthropic lane via llmhub + hublane. Returns an error when
// the API key is empty (registration is opt-in upstream).
func New(cfg Config) (*Provider, error) {
	opts := []llmhub.Option{}
	if cfg.BaseURL != "" {
		opts = append(opts, llmhub.WithBaseURL(cfg.BaseURL))
	}
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	opts = append(opts, llmhub.WithHTTPClient(client))
	hubProv, err := hubanthropic.New(cfg.APIKey, opts...)
	if err != nil {
		return nil, err
	}
	caps := provider.Capabilities{
		Chat:      true,
		Reasoning: true,
		Vision:    true,
		Tools:     true,
		Streaming: true,
	}
	adapter := hublane.New(hubProv, hublane.WithCapabilities(caps))
	baseURL := strings.TrimSuffix(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Provider{
		adapter:    adapter,
		baseURL:    baseURL,
		apiKey:     cfg.APIKey,
		httpClient: client,
	}, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string { return ProviderName }

// Generate forwards to the hublane adapter.
func (p *Provider) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	return p.adapter.Generate(ctx, msgs, opts...)
}

// Stream forwards to the hublane adapter.
func (p *Provider) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (
	<-chan ir.Event, error,
) {
	return p.adapter.Stream(ctx, msgs, opts...)
}

// Provider returns the registry bundle. The lane itself is the bundle's
// inference provider (not the bare hublane adapter) so the discovery surfaces
// below are visible to the registry.
func (p *Provider) Provider() provider.Provider {
	bundle := p.adapter.ProviderBundle()
	bundle.Inference = p
	return bundle
}

// ModelDiscoveryEnabled reports whether this lane can discover its model
// list: an Anthropic lane always needs an API key to call it.
func (p *Provider) ModelDiscoveryEnabled() bool {
	return p != nil && p.apiKey != ""
}

// FetchModels queries GET /v1/models with the Anthropic headers and replaces
// the cached model list. A failed fetch leaves the previous cache untouched
// and invents no ids.
func (p *Provider) FetchModels(ctx context.Context) ([]string, error) {
	endpoint := p.baseURL + "/v1/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("models endpoint returned status %d", resp.StatusCode)
	}

	var raw struct {
		Data []struct {
			ID             string `json:"id"`
			MaxInputTokens int    `json:"max_input_tokens"`
			MaxTokens      int    `json:"max_tokens"`
			Capabilities   struct {
				ImageInput struct {
					Supported bool `json:"supported"`
				} `json:"image_input"`
			} `json:"capabilities"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}

	var models []string
	var info []provider.ModelInfo
	for _, m := range raw.Data {
		if m.ID == "" {
			continue
		}
		models = append(models, m.ID)
		input := []string{modelmeta.ModalityText}
		if m.Capabilities.ImageInput.Supported {
			input = append(input, modelmeta.ModalityImage)
		}
		info = append(info, provider.ModelInfo{
			ID:               m.ID,
			ContextLength:    m.MaxInputTokens,
			MaxOutput:        m.MaxTokens,
			InputModalities:  input,
			OutputModalities: []string{modelmeta.ModalityText},
		})
	}
	sort.Strings(models)
	sort.Slice(info, func(i, j int) bool { return info[i].ID < info[j].ID })

	p.mu.Lock()
	p.models = models
	p.modelInfo = info
	p.mu.Unlock()
	return models, nil
}

// CachedModels returns the discovered model ids without contacting the
// upstream.
func (p *Provider) CachedModels() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.models) == 0 {
		return nil
	}
	out := make([]string, len(p.models))
	copy(out, p.models)
	return out
}

// CachedModelInfo returns the discovered ids with the windows and modalities
// Anthropic reported. Zero values mean the upstream did not say.
func (p *Provider) CachedModelInfo() []provider.ModelInfo {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.modelInfo) == 0 {
		return nil
	}
	out := make([]provider.ModelInfo, len(p.modelInfo))
	copy(out, p.modelInfo)
	return out
}

var (
	_ provider.ModelInfoCache = (*Provider)(nil)
	_ interface {
		FetchModels(ctx context.Context) ([]string, error)
	} = (*Provider)(nil)
)
