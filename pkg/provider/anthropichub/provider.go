// Package anthropichub adapts llmhub's Anthropic provider to the ultiproxy
// provider registry via the shared hublane bridge.
//
// The lane constructor, the inference delegation and the model-discovery
// cache (GET /v1/models with the Anthropic headers) live in models.go.
package anthropichub

import "net/http"

const (
	// ProviderName is the lane name the registry and routing use.
	ProviderName = "anthropic"
	// DefaultBaseURL is the first-party Anthropic API root.
	DefaultBaseURL = "https://api.anthropic.com"
)

// Config configures the Anthropic (first-party API) lane.
type Config struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}
