package openaicompat

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

// CLI wire ground truth, captured from the official freebuff CLI 0.0.167
// (mitmdump reverse capture, 2026-09-06; see
// plans/freebuff-protocol-and-proxy.md). Upstream gates /chat/completions on
// this exact shape: bodies that deviate (missing metadata fields, wrong
// provider object, toy toolset, temperature, non-stream) get 428
// waiting_room_required even with a freshly-bound active session.
var (
	//go:embed assets/cli_tools.json
	cliToolsJSON []byte

	//go:embed assets/cli_system_prompt.txt
	cliSystemPrompt string

	//go:embed assets/cli_validate_template.json
	cliValidateTemplate []byte

	cliToolsOnce sync.Once
	cliTools     []any
	cliToolsErr  error
)

// freebuffCLITools returns the CLI's 15-tool function array (decoded once).
func freebuffCLITools() ([]any, error) {
	cliToolsOnce.Do(func() {
		cliToolsErr = json.Unmarshal(cliToolsJSON, &cliTools)
	})
	return cliTools, cliToolsErr
}

// freebuffToolsForRequest builds the chat body's tools array: the CLI's
// 15-tool set is always the base (upstream's free-mode router matches
// endpoints against it — replacing it wholesale 404s "No endpoints found");
// the caller's tools are appended after the base so arbitrary tools survive
// the lane. Live-verified 2026-09-06: base+append -> 200, replace -> 404.
func freebuffToolsForRequest(reqConfig *provider.RequestConfig) ([]any, error) {
	base, err := freebuffCLITools()
	if err != nil {
		return nil, err
	}
	out := make([]any, len(base), len(base)+8)
	copy(out, base)
	if reqConfig != nil && reqConfig.ExtraBody != nil {
		if tools, ok := reqConfig.ExtraBody["tools"].([]any); ok {
			out = append(out, tools...)
		}
	}
	return out, nil
}

// freebuffValidatePreChat fires the CLI's pre-chat agent validation call
// (POST /api/agents/validate) with the captured template, substituting the
// requested model. Best-effort: failures are logged and do not block chat —
// upstream answers 200 configs on success; a hard failure means the account
// or model is rejected and chat will surface that on its own.
func freebuffValidatePreChat(ctx context.Context, httpClient *http.Client, baseURL, apiKey, model string) error {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	// Bounded: validation must never wedge a chat behind a stuck upstream.
	vctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	ctx = vctx
	payload := string(cliValidateTemplate)
	payload = strings.ReplaceAll(payload, "{MODEL}", model)
	// The validate endpoint lives OUTSIDE /api/v1: the CLI calls
	// https://www.codebuff.com/api/agents/validate (captured 2026-09-06).
	validateURL := strings.Replace(baseURL+"/agents/validate", "/api/v1/agents/", "/api/agents/", 1)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, validateURL, strings.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("agents/validate status %d: %s", resp.StatusCode, string(body))
	}
	return nil
}

// freebuffCLIUserAgent is the User-Agent the current CLI's ai-sdk factory sends.
const freebuffCLIUserAgent = "ai-sdk/openai-compatible/0.0.0-test/codebuff ai-sdk/provider-utils/3.0.25 runtime/browser"

// freebuffCLISTOP is the stop-sequence array the CLI sends verbatim.
var freebuffCLISTOP = []any{"\"cb_easp\""}

// actorInstanceID extracts the instance id from the lane's freebuff actor for
// codebuff_metadata.freebuff_instance_id (an empty string if unknown).
func actorInstanceID(p *Provider) string {
	if p == nil || p.cfg.Quirks.FreebuffActor == nil {
		return ""
	}
	if inst, ok := p.cfg.Quirks.FreebuffActor.(freebuffInstanceIDer); ok {
		return inst.InstanceID()
	}
	return ""
}

// freebuffRepoSnapshot is the repo_snapshot metadata string for an empty
// non-git workspace (what a bare proxy workspace looks like to the CLI).
const freebuffRepoSnapshot = `{"gitAvailable":false,"repositoryVisibility":"unknown","fileCount":0,"fileCountIsLowerBound":false,"testFileCount":0,"changedFileCount":0,"changedFileScanTruncated":false}`

// freebuffToolsSig fingerprints a toolset (names only, order-sensitive) so the
// lane can detect toolset changes between requests and re-bind the session.
func freebuffToolsSig(tools []any) string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		if tm, ok := t.(map[string]any); ok {
			if fn, ok := tm["function"].(map[string]any); ok {
				if n, ok := fn["name"].(string); ok {
					names = append(names, n)
				}
			}
		}
	}
	return strings.Join(names, ",")
}
