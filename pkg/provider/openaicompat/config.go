package openaicompat

import (
	"context"
	"net/http"

	"github.com/smhanov/llmhub/auth"
)

// Config configures the OpenAI-compatible provider adapter.
type Config struct {
	Name        string           // registry lane name ("openai-zai", "openai-vllm", ...)
	BaseURL     string           // vendor default OR configured
	APIKey      string           // static key
	TokenSource auth.TokenSource // opencode cookie, xai OAuth, augure refresh
	HTTPClient  *http.Client
	DataDir     string // for xai OAuth cred dir, augure token file
	Quirks      Quirks

	// Optional quirk-specific credentials / overrides
	WorkspaceID   string // for AuthViaWorkspaceCookie (opencode)
	SessionCookie string // for AuthViaWorkspaceCookie (opencode)
	RefreshURL    string // for AuthViaSupabaseRefresh (augure)
	TokenFile     string // for AuthViaSupabaseRefresh (augure)
	DeviceAuthURL string // for AuthViaOAuthManager (xai)
	TokenURL      string // for AuthViaOAuthManager (xai)
}

// Quirks contains vendor-specific behavioral tweaks for OpenAI-compatible wire endpoints.
type Quirks struct {
	CodingPlanPath         bool           // zai: base URL contains "coding" -> coding-plan variant + max-tokens defaults
	MaxTokensByModel       map[string]int // zai: glm-4.5-air -> 98304, else 131072 (resolveMaxTokens)
	EchoReasoning          bool           // deepseek: re-emit reasoning_content on input + parse on output
	ModelListPassthrough   bool           // vllm: /v1/models from upstream, no auth required
	AuthViaWorkspaceCookie bool           // opencode: workspace id + session cookie headers
	AuthViaOAuthManager    bool           // xai: auth.Manager creds + refresh
	CreditsQuotaObserver   string         // xai: credits endpoint id for the quota observer ("" = none)
	AuthViaSupabaseRefresh bool           // augure: token file + Supabase refresh
	FreebuffActor          any            // injected *spikesfreebuff.FreebuffAccountActor (avoid import cycle: use an interface or any with a small interface type asserted at runtime; document it)
	FreebuffDefaultTool    bool           // freebuff: prepend default tool + Buffy system prompt + codebuff_metadata
	DefaultModel           string         // augure: "tofino-3"; empty otherwise
}

// FreebuffActor defines the minimal lock interface needed for serialized requests.
// Injected *spikesfreebuff.FreebuffAccountActor or a test fake satisfies this interface.
type FreebuffActor interface {
	Acquire(ctx context.Context) error
	Release()
}

// freebuffQuotaSource is the actor subset needed to report freebuff quota.
// Satisfied by *freebuffActorAdapter in cmd (wrapping *FreebuffAccountActor).
type freebuffQuotaSource interface {
	FetchUsage(ctx context.Context, fingerprintID string) ([]byte, error)
	SessionInfo(ctx context.Context) (instanceID, model string, err error)
}

// freebuffInstanceIDer is implemented by freebuff actors that expose their
// instance id (used for the x-freebuff-instance-id header).
type freebuffInstanceIDer interface {
	InstanceID() string
}

// freebuffTokenSetter is implemented by freebuff actors that can accept an
// imported CLI token during login.
type freebuffTokenSetter interface {
	SetToken(tok string)
}

// freebuffInstanceIDSetter is implemented by freebuff actors that can accept an
// imported instance ID during login.
type freebuffInstanceIDSetter interface {
	SetInstanceID(id string)
}
