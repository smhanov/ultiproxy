package quota

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// ProviderMetadata provides display names and subscription plans for a provider.
type ProviderMetadata struct {
	Name string `json:"name"`
	Plan string `json:"plan"`
}

// HandlerConfig configures the dashboard HTTP handler.
type HandlerConfig struct {
	StateManager *state.StateManager
	Registry     *provider.Registry
	Store        *QuotaStore
	Storage      *storage.Writer
	Metadata     map[string]ProviderMetadata
	NowFn        func() time.Time
}

// WindowSummary represents one rate limit window in the quota.fjkl.cc contract.
type WindowSummary struct {
	Label            string  `json:"label"`
	UsedPct          float64 `json:"used_pct"`
	ResetAt          string  `json:"reset_at,omitempty"`
	SecondsRemaining int64   `json:"seconds_remaining"`
}

// BarSummary represents a usage bar in the quota.fjkl.cc contract.
type BarSummary struct {
	Label     string  `json:"label"`
	Used      float64 `json:"used"`
	Limit     float64 `json:"limit"`
	Remaining float64 `json:"remaining"`
	Unit      string  `json:"unit"`
	Pct       float64 `json:"pct"`
}

// ProviderSummary represents one provider in the quota.fjkl.cc contract.
type ProviderSummary struct {
	ID       string                 `json:"id"`
	Name     string                 `json:"name"`
	Plan     string                 `json:"plan"`
	Status   string                 `json:"status"`
	GaugePct float64                `json:"gauge_pct"`
	Windows  []WindowSummary        `json:"windows"`
	Bars     []BarSummary           `json:"bars"`
	Updated  string                 `json:"updated"`
	Detail   string                 `json:"detail"`
	Extra    map[string]interface{} `json:"extra"`
}

// NextReset describes the soonest reset event.
type NextReset struct {
	Seconds  int64  `json:"seconds"`
	Provider string `json:"provider"`
	ResetAt  string `json:"reset_at"`
}

// DashboardSummary represents the top-level summary in the quota.fjkl.cc contract.
type DashboardSummary struct {
	Total              int        `json:"total"`
	Ok                 int        `json:"ok"`
	Unavailable        int        `json:"unavailable"`
	NextReset          *NextReset `json:"next_reset"`
	FetchedAt          string     `json:"fetched_at"`
	FetchedAtEpoch     int64      `json:"fetched_at_epoch"`
	Stale              bool       `json:"stale"`
	AgeSeconds         int64      `json:"age_seconds"`
	Refreshing         bool       `json:"refreshing"`
	RefreshStartedAt   *string    `json:"refresh_started_at"`
	RefreshMinInterval int        `json:"refresh_min_interval"`
	LastRefreshError   *string    `json:"last_refresh_error"`
}

// DashboardResponse represents the complete quota.fjkl.cc schema.
type DashboardResponse struct {
	Providers []ProviderSummary `json:"providers"`
	Summary   DashboardSummary  `json:"summary"`
}

// DashboardHandler serves dashboard and telemetry endpoints.
type DashboardHandler struct {
	cfg HandlerConfig
	mux *http.ServeMux
}

// NewHandler constructs a new dashboard HTTP handler implementing the contract.
func NewHandler(cfg HandlerConfig) http.Handler {
	if cfg.Metadata == nil {
		cfg.Metadata = defaultMetadata()
	}
	h := &DashboardHandler{
		cfg: cfg,
		mux: http.NewServeMux(),
	}
	h.registerRoutes()
	return h
}

// RegisterRoutes mounts all quota endpoints directly onto the given ServeMux.
func RegisterRoutes(mux *http.ServeMux, cfg HandlerConfig) {
	h := NewHandler(cfg)
	mux.Handle("/api/quota", h)
	mux.Handle("/quota.txt", h)
	mux.Handle("/quota.md", h)
	mux.Handle("/llms.txt", h)
	mux.Handle("/healthz", h)
	mux.Handle("/api/stats/summary", h)
	mux.Handle("/api/stats/by-client", h)
}

func (h *DashboardHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mux.ServeHTTP(w, r)
}

func (h *DashboardHandler) registerRoutes() {
	h.mux.HandleFunc("/api/quota", h.handleAPIQuota)
	h.mux.HandleFunc("/quota.txt", h.handleQuotaTxt)
	h.mux.HandleFunc("/quota.md", h.handleQuotaMd)
	h.mux.HandleFunc("/llms.txt", h.handleLLMsTxt)
	h.mux.HandleFunc("/healthz", h.handleHealthz)
	h.mux.HandleFunc("/api/stats/summary", h.handleStatsSummary)
	h.mux.HandleFunc("/api/stats/by-client", h.handleStatsByClient)
}

func (h *DashboardHandler) now() time.Time {
	if h.cfg.NowFn != nil {
		return h.cfg.NowFn()
	}
	if h.cfg.StateManager != nil {
		return h.cfg.StateManager.Now()
	}
	return time.Now().UTC()
}

func (h *DashboardHandler) BuildDashboardResponse() DashboardResponse {
	now := h.now()

	var providerNames []string
	if h.cfg.Registry != nil {
		providerNames = h.cfg.Registry.Names()
	} else if h.cfg.StateManager != nil {
		snap := h.cfg.StateManager.Snapshot()
		for id := range snap.Providers {
			providerNames = append(providerNames, id)
		}
		sort.Strings(providerNames)
	}

	var providersOut []ProviderSummary
	var okCount, unavailCount int
	var earliestReset *NextReset

	refreshing, refreshStartedAt, minInterval, lastErr, lastFetchedAt := false, (*time.Time)(nil), 30*time.Second, (*string)(nil), time.Time{}
	if h.cfg.Store != nil {
		refreshing, refreshStartedAt, minInterval, lastErr, lastFetchedAt = h.cfg.Store.Metadata()
	}
	if lastFetchedAt.IsZero() {
		lastFetchedAt = now
	}

	for _, id := range providerNames {
		meta := h.lookupMetadata(id)
		pSummary := ProviderSummary{
			ID:      id,
			Name:    meta.Name,
			Plan:    meta.Plan,
			Windows: []WindowSummary{},
			Bars:    []BarSummary{},
			Extra:   make(map[string]interface{}),
		}

		var pr state.ProviderRuntime
		hasPR := false
		if h.cfg.StateManager != nil {
			snap := h.cfg.StateManager.Snapshot()
			if p, exists := snap.Providers[id]; exists {
				pr = p
				hasPR = true
			}
		}

		// Status calculation
		if hasPR {
			if pr.Admin == state.AdminDisabled || pr.Circuit == state.CircuitOpen || pr.Quota == state.QuotaExhausted {
				pSummary.Status = "unavailable"
			} else if pr.Health == state.HealthUnavailable {
				pSummary.Status = "error"
			} else {
				pSummary.Status = "ok"
			}
		} else {
			pSummary.Status = "ok"
		}

		if pSummary.Status == "ok" {
			okCount++
		} else {
			unavailCount++
		}

		// Windows & Bars
		var qSnap *provider.QuotaSnapshot
		if h.cfg.Store != nil {
			qSnap, _ = h.cfg.Store.Get(id)
		}

		var worstGauge float64
		if qSnap != nil {
			if !qSnap.ObservedAt.IsZero() {
				pSummary.Updated = qSnap.ObservedAt.Format(time.RFC3339)
			}
			pSummary.Detail = qSnap.Detail

			for _, w := range qSnap.Windows {
				if w.UsedPct > worstGauge {
					worstGauge = w.UsedPct
				}

				secsRem := w.SecondsRemaining
				var resetStr string
				if !w.ResetAt.IsZero() {
					resetStr = w.ResetAt.Format(time.RFC3339)
					if secsRem <= 0 {
						diff := int64(w.ResetAt.Sub(now).Seconds())
						if diff > 0 {
							secsRem = diff
						}
					}
				}

				pSummary.Windows = append(pSummary.Windows, WindowSummary{
					Label:            w.Label,
					UsedPct:          w.UsedPct,
					ResetAt:          resetStr,
					SecondsRemaining: secsRem,
				})

				unit := w.Unit
				if unit == "" {
					unit = "requests"
				}

				usedVal := w.Limit - w.Remaining
				if w.Limit > 0 && w.Remaining >= 0 {
					usedVal = w.Limit - w.Remaining
				} else if w.Limit > 0 {
					usedVal = w.Limit * (w.UsedPct / 100.0)
				} else {
					usedVal = w.UsedPct
				}

				pSummary.Bars = append(pSummary.Bars, BarSummary{
					Label:     w.Label,
					Used:      usedVal,
					Limit:     w.Limit,
					Remaining: w.Remaining,
					Unit:      unit,
					Pct:       w.UsedPct,
				})

				// Check earliest next_reset across ok providers
				if pSummary.Status == "ok" && secsRem > 0 && resetStr != "" {
					if earliestReset == nil || secsRem < earliestReset.Seconds {
						earliestReset = &NextReset{
							Seconds:  secsRem,
							Provider: meta.Name,
							ResetAt:  resetStr,
						}
					}
				}
			}
		}

		if pSummary.Updated == "" {
			if hasPR && !pr.ObservedAt.IsZero() {
				pSummary.Updated = pr.ObservedAt.Format(time.RFC3339)
			} else {
				pSummary.Updated = now.Format(time.RFC3339)
			}
		}

		pSummary.GaugePct = worstGauge
		providersOut = append(providersOut, pSummary)
	}

	ageSeconds := int64(now.Sub(lastFetchedAt).Seconds())
	if ageSeconds < 0 {
		ageSeconds = 0
	}
	stale := ageSeconds >= 60

	var refreshStartedStr *string
	if refreshStartedAt != nil {
		s := refreshStartedAt.Format(time.RFC3339)
		refreshStartedStr = &s
	}

	minIntervalSecs := int(minInterval.Seconds())
	if minIntervalSecs <= 0 {
		minIntervalSecs = 30
	}

	summary := DashboardSummary{
		Total:              len(providersOut),
		Ok:                 okCount,
		Unavailable:        unavailCount,
		NextReset:          earliestReset,
		FetchedAt:          lastFetchedAt.Format("15:04:05"),
		FetchedAtEpoch:     lastFetchedAt.Unix(),
		Stale:              stale,
		AgeSeconds:         ageSeconds,
		Refreshing:         refreshing,
		RefreshStartedAt:   refreshStartedStr,
		RefreshMinInterval: minIntervalSecs,
		LastRefreshError:   lastErr,
	}

	return DashboardResponse{
		Providers: providersOut,
		Summary:   summary,
	}
}

func (h *DashboardHandler) handleAPIQuota(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	resp := h.BuildDashboardResponse()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *DashboardHandler) handleQuotaTxt(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	resp := h.BuildDashboardResponse()
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")

	var b strings.Builder
	for _, p := range resp.Providers {
		b.WriteString(fmt.Sprintf("%s (%s): %s - %.1f%% used", p.Name, p.ID, p.Status, p.GaugePct))
		if len(p.Bars) > 0 {
			bar := p.Bars[0]
			b.WriteString(fmt.Sprintf(" [%.0f/%.0f %s remaining]", bar.Remaining, bar.Limit, bar.Unit))
		}
		if p.Detail != "" {
			b.WriteString(fmt.Sprintf(" (%s)", p.Detail))
		}
		b.WriteString("\n")
	}
	_, _ = w.Write([]byte(b.String()))
}

func (h *DashboardHandler) handleQuotaMd(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	resp := h.BuildDashboardResponse()
	w.Header().Set("Content-Type", "text/markdown; charset=utf-8")

	var b strings.Builder
	b.WriteString("# Quota Status\n\n")
	for _, p := range resp.Providers {
		b.WriteString(fmt.Sprintf("- **%s**: `%s` (%.1f%% used)\n", p.Name, p.Status, p.GaugePct))
		for _, bar := range p.Bars {
			b.WriteString(fmt.Sprintf("  - %s: %.0f / %.0f %s remaining\n", bar.Label, bar.Remaining, bar.Limit, bar.Unit))
		}
		if p.Detail != "" {
			b.WriteString(fmt.Sprintf("  - *Detail*: %s\n", p.Detail))
		}
	}
	_, _ = w.Write([]byte(b.String()))
}

func (h *DashboardHandler) handleLLMsTxt(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write([]byte("# Ultiproxy\nUnified AI routing and quota management gateway.\n"))
}

func (h *DashboardHandler) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func (h *DashboardHandler) handleStatsSummary(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if h.cfg.Storage == nil {
		http.Error(w, "storage writer unavailable", http.StatusServiceUnavailable)
		return
	}

	ctx := r.Context()
	summaries, err := h.cfg.Storage.ListUsageSummary(ctx)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to query usage summary: %v", err), http.StatusInternalServerError)
		return
	}

	clients, err := h.cfg.Storage.ListUsageByClient(ctx, "30d")
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to query usage by client: %v", err), http.StatusInternalServerError)
		return
	}

	resp := map[string]any{
		"summary": summaries,
		"clients": clients,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *DashboardHandler) handleStatsByClient(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if h.cfg.Storage == nil {
		http.Error(w, "storage writer unavailable", http.StatusServiceUnavailable)
		return
	}

	window := r.URL.Query().Get("window")
	if window == "" {
		window = "7d"
	}

	ctx := r.Context()
	clients, err := h.cfg.Storage.ListUsageByClient(ctx, window)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to query usage by client: %v", err), http.StatusInternalServerError)
		return
	}

	resp := map[string]any{
		"window":  window,
		"clients": clients,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *DashboardHandler) lookupMetadata(providerID string) ProviderMetadata {
	if h.cfg.Metadata != nil {
		if m, ok := h.cfg.Metadata[providerID]; ok {
			return m
		}
	}
	return ProviderMetadata{
		Name: formatDefaultName(providerID),
		Plan: "Standard",
	}
}

func defaultMetadata() map[string]ProviderMetadata {
	return map[string]ProviderMetadata{
		"copilot": {
			Name: "GitHub Copilot",
			Plan: "Pro (annual)",
		},
		"openai": {
			Name: "OpenAI Codex",
			Plan: "Usage-based",
		},
		"anthropic": {
			Name: "Anthropic Claude",
			Plan: "Team Tier 4",
		},
		"antigravity": {
			Name: "Antigravity AI",
			Plan: "Enterprise",
		},
	}
}

func formatDefaultName(id string) string {
	parts := strings.Split(id, "-")
	for i, p := range parts {
		if len(p) > 0 {
			parts[i] = strings.ToUpper(p[:1]) + p[1:]
		}
	}
	return strings.Join(parts, " ")
}
