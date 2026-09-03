package opencode

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

const (
	ProviderName   = "opencode"
	DefaultBaseURL = "https://opencode.ai"
)

// Config configures the OpenCode scraper.
type Config struct {
	BaseURL       string
	WorkspaceID   string
	SessionCookie string
	HTTPClient    *http.Client
}

// Provider implements provider.QuotaProvider.
type Provider struct {
	cfg           Config
	httpClient    *http.Client
	baseURL       string
	workspaceID   string
	sessionCookie string
}

// New creates a new OpenCode quota provider.
func New(cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")

	if cfg.WorkspaceID == "" {
		cfg.WorkspaceID = os.Getenv("OPENCODE_WORKSPACE_ID")
	}

	if cfg.SessionCookie == "" {
		cfg.SessionCookie = os.Getenv("OPENCODE_SESSION_COOKIE")
	}

	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}

	return &Provider{
		cfg:           cfg,
		httpClient:    cfg.HTTPClient,
		baseURL:       cfg.BaseURL,
		workspaceID:   cfg.WorkspaceID,
		sessionCookie: cfg.SessionCookie,
	}, nil
}

// Name returns the provider identifier.
func (p *Provider) Name() string {
	return ProviderName
}

// Provider returns the provider.Provider bundle.
func (p *Provider) Provider() provider.Provider {
	return provider.Provider{
		Quota: p,
	}
}

// Quota fetches workspace HTML and parses quota usage into QuotaSnapshot.
func (p *Provider) Quota(ctx context.Context) (*provider.QuotaSnapshot, error) {
	if p.workspaceID == "" {
		return nil, fmt.Errorf("opencode workspace ID is required")
	}

	url := fmt.Sprintf("%s/workspace/%s", p.baseURL, p.workspaceID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	if p.sessionCookie != "" {
		cookieHeader := p.sessionCookie
		if !strings.Contains(cookieHeader, "=") {
			cookieHeader = "session=" + cookieHeader
		}
		req.Header.Set("Cookie", cookieHeader)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch workspace HTML: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("opencode workspace request failed with status %d", resp.StatusCode)
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return ParseWorkspaceHTML(string(bodyBytes), time.Now())
}

// Regex patterns to parse JS object literals:
// rollingUsage: { usagePercent: 42.5, resetInSec: 3600 }
var (
	rollingBlockRe = regexp.MustCompile(`(?:rollingUsage|["']rollingUsage["'])\s*:\s*\{([^}]+)\}`)
	weeklyBlockRe  = regexp.MustCompile(`(?:weeklyUsage|["']weeklyUsage["'])\s*:\s*\{([^}]+)\}`)
	monthlyBlockRe = regexp.MustCompile(`(?:monthlyUsage|["']monthlyUsage["'])\s*:\s*\{([^}]+)\}`)

	usagePercentRe = regexp.MustCompile(`(?:usagePercent|["']usagePercent["'])\s*:\s*([0-9.]+)`)
	resetInSecRe   = regexp.MustCompile(`(?:resetInSec|["']resetInSec["'])\s*:\s*([0-9]+)`)
)

func parseUsageBlock(blockContent string) (float64, int64, bool) {
	pctMatch := usagePercentRe.FindStringSubmatch(blockContent)
	secMatch := resetInSecRe.FindStringSubmatch(blockContent)

	if len(pctMatch) < 2 || len(secMatch) < 2 {
		return 0, 0, false
	}

	pct, err := strconv.ParseFloat(pctMatch[1], 64)
	if err != nil {
		return 0, 0, false
	}

	sec, err := strconv.ParseInt(secMatch[1], 10, 64)
	if err != nil {
		return 0, 0, false
	}

	return pct, sec, true
}

// ParseWorkspaceHTML regex-parses workspace HTML into QuotaSnapshot.
func ParseWorkspaceHTML(html string, now time.Time) (*provider.QuotaSnapshot, error) {
	var windows []provider.QuotaWindow

	// 1. Rolling
	if m := rollingBlockRe.FindStringSubmatch(html); len(m) >= 2 {
		if pct, sec, ok := parseUsageBlock(m[1]); ok {
			windows = append(windows, provider.QuotaWindow{
				Label:            "Rolling",
				UsedPct:          pct,
				Unit:             "%",
				SecondsRemaining: sec,
				ResetAt:          now.Add(time.Duration(sec) * time.Second),
			})
		}
	}

	// 2. Weekly
	if m := weeklyBlockRe.FindStringSubmatch(html); len(m) >= 2 {
		if pct, sec, ok := parseUsageBlock(m[1]); ok {
			windows = append(windows, provider.QuotaWindow{
				Label:            "Weekly",
				UsedPct:          pct,
				Unit:             "%",
				SecondsRemaining: sec,
				ResetAt:          now.Add(time.Duration(sec) * time.Second),
			})
		}
	}

	// 3. Monthly
	if m := monthlyBlockRe.FindStringSubmatch(html); len(m) >= 2 {
		if pct, sec, ok := parseUsageBlock(m[1]); ok {
			windows = append(windows, provider.QuotaWindow{
				Label:            "Monthly",
				UsedPct:          pct,
				Unit:             "%",
				SecondsRemaining: sec,
				ResetAt:          now.Add(time.Duration(sec) * time.Second),
			})
		}
	}

	if len(windows) == 0 {
		return nil, fmt.Errorf("failed to parse any usage blocks from workspace HTML")
	}

	return &provider.QuotaSnapshot{
		ObservedAt: now,
		Windows:    windows,
	}, nil
}
