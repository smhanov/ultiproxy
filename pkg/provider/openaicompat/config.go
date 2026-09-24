package openaicompat

import (
	"context"
	"net/http"

	llmauth "github.com/smhanov/llmhub/auth"
	"github.com/smhanov/ultiproxy/pkg/auth"
)

// Config configures the OpenAI-compatible provider adapter.
type Config struct {
	Name        string               // registry lane name ("openai-zai", "openai-vllm", ...)
	BaseURL     string               // vendor default OR configured
	APIKey      string               // static key
	TokenSource llmauth.TokenSource  // xai OAuth, augure refresh (set by the daemon, never by MCP)
	Creds       auth.CredentialStore // xai OAuth credential store (injected by the daemon, never by MCP)
	HTTPClient  *http.Client
	// DataDir is deprecated daemon plumbing (not client-facing): only the
	// augure TokenFile derivation still reads it (see New). xai OAuth lanes
	// must use Creds instead — a missing store is a hard error, never a
	// TempDir fallback. Runtime lanes never carry their own data dir.
	DataDir string // deprecated: augure token file only; xai uses Creds
	Quirks  Quirks

	// OptOutModelListPassthrough explicitly disables upstream model discovery
	// for this lane. Discovery is ON by default for OpenAI-compatible lanes
	// (see ModelListPassthroughEnabled): one GET <base>/models, cached
	// afterwards, is what makes /v1/models honest instead of a stub. The MCP
	// quirks.model_list_passthrough:false switch and the persisted
	// quirks.model_list_passthrough=false both map here, so an explicit opt-out
	// survives a restart while an absent field keeps the default.
	OptOutModelListPassthrough bool

	// Optional quirk-specific overrides
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
	ModelListPassthrough   bool           // discovery: slurp GET <base>/v1/models (default ON, see ModelListPassthroughEnabled)
	AuthViaOAuthManager    bool           // xai: auth.Manager creds + refresh
	CreditsQuotaObserver   string         // xai: credits endpoint id for the quota observer ("" = none)
	AuthViaSupabaseRefresh bool           // augure: token file + Supabase refresh
	FreebuffActor          any            // injected *spikesfreebuff.FreebuffAccountActor (avoid import cycle: use an interface or any with a small interface type asserted at runtime; document it)
	FreebuffDefaultTool    bool           // freebuff: prepend default tool + Buffy system prompt + codebuff_metadata
	FreebuffValidate       bool           // freebuff: fire the CLI's best-effort POST /api/agents/validate pre-chat (off in unit tests; on in wired lanes)
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

// freebuffRunFinisher is implemented by freebuff actors that can finish an agent run.
type freebuffRunFinisher interface {
	FinishRun(ctx context.Context, runID, status string, err error) error
}

// freebuffHeartbeatStarter is implemented by freebuff actors with an active heartbeat loop.
type freebuffHeartbeatStarter interface {
	StartHeartbeat()
}
