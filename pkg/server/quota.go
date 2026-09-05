package server

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

// QuotaDashboardResponse matches quota dashboard JSON structure.
type QuotaDashboardResponse struct {
	Providers []ProviderSummary `json:"providers"`
	Summary   DashboardSummary  `json:"summary"`
}

type ProviderSummary struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Plan     string            `json:"plan"`
	Status   string            `json:"status"` // ok | unavailable | error
	GaugePct float64           `json:"gauge_pct"`
	Windows  []DashboardWindow `json:"windows"`
	Bars     []DashboardBar    `json:"bars"`
	Updated  string            `json:"updated"`
	Detail   string            `json:"detail"`
	Extra    map[string]any    `json:"extra"`
}

type DashboardWindow struct {
	Label            string  `json:"label"`
	UsedPct          float64 `json:"used_pct"`
	ResetAt          string  `json:"reset_at,omitempty"`
	SecondsRemaining int64   `json:"seconds_remaining,omitempty"`
}

type DashboardBar struct {
	Label     string  `json:"label"`
	Used      float64 `json:"used"`
	Limit     float64 `json:"limit"`
	Remaining float64 `json:"remaining"`
	Unit      string  `json:"unit"`
	Pct       float64 `json:"pct"`
}

type DashboardSummary struct {
	Total              int        `json:"total"`
	OK                 int        `json:"ok"`
	Unavailable        int        `json:"unavailable"`
	NextReset          *NextReset `json:"next_reset"`
	FetchedAt          string     `json:"fetched_at"`
	FetchedAtEpoch     int64      `json:"fetched_at_epoch"`
	Stale              bool       `json:"stale"`
	AgeSeconds         int        `json:"age_seconds"`
	Refreshing         bool       `json:"refreshing"`
	RefreshStartedAt   *string    `json:"refresh_started_at"`
	RefreshMinInterval int        `json:"refresh_min_interval"`
	LastRefreshError   *string    `json:"last_refresh_error"`
}

type NextReset struct {
	Seconds  int64  `json:"seconds"`
	Provider string `json:"provider"`
	ResetAt  string `json:"reset_at"`
}

func (s *Server) handleQuotaDashboard(w http.ResponseWriter, r *http.Request) {
	resp := s.buildQuotaResponse(r.Context())
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleQuotaMarkdown(w http.ResponseWriter, r *http.Request) {
	resp := s.buildQuotaResponse(r.Context())
	w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
	fmt.Fprintf(w, "# Quota Status\n\nTotal: %d | OK: %d | Unavailable: %d\n\n", resp.Summary.Total, resp.Summary.OK, resp.Summary.Unavailable)
	for _, p := range resp.Providers {
		fmt.Fprintf(w, "## %s (%s)\n- Status: %s (%.1f%% used)\n- Detail: %s\n\n", p.Name, p.ID, p.Status, p.GaugePct, p.Detail)
	}
}

func (s *Server) handleQuotaText(w http.ResponseWriter, r *http.Request) {
	resp := s.buildQuotaResponse(r.Context())
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	fmt.Fprintf(w, "QUOTA STATUS: Total=%d OK=%d Unavailable=%d\n", resp.Summary.Total, resp.Summary.OK, resp.Summary.Unavailable)
	for _, p := range resp.Providers {
		fmt.Fprintf(w, "%-15s %-12s gauge=%.1f%% %s\n", p.ID, p.Status, p.GaugePct, p.Detail)
	}
}

func (s *Server) buildQuotaResponse(ctx context.Context) QuotaDashboardResponse {
	now := time.Now().UTC()
	var summaries []ProviderSummary
	var okCount, unavailCount int
	var earliestReset *NextReset

	if s.registry != nil {
		for _, name := range s.registry.Names() {
			prov, _ := s.registry.Get(name)
			status := "ok"
			gaugePct := 0.0
			detail := "Operational"
			var windows []DashboardWindow
			var bars []DashboardBar

			if prov.Quota != nil {
				qSnap, err := prov.Quota.Quota(ctx)
				if err == nil && qSnap != nil {
					for _, w := range qSnap.Windows {
						resetStr := ""
						if !w.ResetAt.IsZero() {
							resetStr = w.ResetAt.UTC().Format(time.RFC3339)
						}
						windows = append(windows, DashboardWindow{
							Label:            w.Label,
							UsedPct:          w.UsedPct,
							ResetAt:          resetStr,
							SecondsRemaining: w.SecondsRemaining,
						})
						bars = append(bars, DashboardBar{
							Label:     w.Label,
							Used:      w.Limit - w.Remaining,
							Limit:     w.Limit,
							Remaining: w.Remaining,
							Unit:      w.Unit,
							Pct:       w.UsedPct,
						})
						if w.UsedPct > gaugePct {
							gaugePct = w.UsedPct
						}
						if w.SecondsRemaining > 0 {
							if earliestReset == nil || w.SecondsRemaining < earliestReset.Seconds {
								earliestReset = &NextReset{
									Seconds:  w.SecondsRemaining,
									Provider: name,
									ResetAt:  resetStr,
								}
							}
						}
					}
					if qSnap.Detail != "" {
						detail = qSnap.Detail
					}
				} else {
					status = "error"
				}
			}

			// Cross check state manager
			if s.sm != nil {
				snap := s.sm.Snapshot()
				if snap != nil {
					if pr, exists := snap.Providers[name]; exists {
						if !pr.IsAvailable() {
							status = "unavailable"
						}
						if pr.Quota == state.QuotaExhausted {
							gaugePct = 100.0
						}
					}
				}
			}

			if status == "ok" {
				okCount++
			} else {
				unavailCount++
			}

			summaries = append(summaries, ProviderSummary{
				ID:       name,
				Name:     name,
				Plan:     "standard",
				Status:   status,
				GaugePct: gaugePct,
				Windows:  windows,
				Bars:     bars,
				Updated:  now.Format(time.RFC3339),
				Detail:   detail,
				Extra:    make(map[string]any),
			})
		}
	}

	return QuotaDashboardResponse{
		Providers: summaries,
		Summary: DashboardSummary{
			Total:              len(summaries),
			OK:                 okCount,
			Unavailable:        unavailCount,
			NextReset:          earliestReset,
			FetchedAt:          now.Format("15:04:05"),
			FetchedAtEpoch:     now.Unix(),
			Stale:              false,
			AgeSeconds:         0,
			Refreshing:         false,
			RefreshMinInterval: 30,
		},
	}
}

// StatsSummaryResponse is the GET /api/stats/summary contract published in
// docs/openapi.yaml (components/schemas/StatsSummaryResponse). Every field the
// spec marks required is always present, including for an empty store.
//
// estimated_cost_saved_usd is the pay-as-you-go equivalent of the recorded
// traffic (the sum of usage.cost, priced from each model's alias pricing): the
// pooled subscriptions are flat fee, so that amount is what the same tokens
// would have cost from a metered API.
//
// total_tokens and total_cost predate the published schema and stay in the
// payload for existing clients; the spec documents them as deprecated.
type StatsSummaryResponse struct {
	TotalRequests         int64                              `json:"total_requests"`
	TotalPromptTokens     int64                              `json:"total_prompt_tokens"`
	TotalCompletionTokens int64                              `json:"total_completion_tokens"`
	TotalCachedTokens     int64                              `json:"total_cached_tokens"`
	ActiveClients         int64                              `json:"active_clients"`
	EstimatedCostSavedUSD float64                            `json:"estimated_cost_saved_usd"`
	ProviderBreakdown     map[string]*ProviderStatsBreakdown `json:"provider_breakdown"`

	// Deprecated: kept for clients written against the pre-contract payload.
	TotalTokens int64   `json:"total_tokens"`
	TotalCost   float64 `json:"total_cost"`
}

// ProviderStatsBreakdown is one entry of the /api/stats/summary
// provider_breakdown map.
type ProviderStatsBreakdown struct {
	Requests int64 `json:"requests"`
	Tokens   int64 `json:"tokens"`
	Errors   int64 `json:"errors"`
}

// handleStatsSummary serves GET /api/stats/summary: the aggregate usage and
// accounting numbers behind the MCP get_client_usage tool, straight from the
// telemetry store. A daemon built without storage answers the contract with
// zeros rather than a 503, so monitoring setups keep working.
func (s *Server) handleStatsSummary(w http.ResponseWriter, r *http.Request) {
	resp := StatsSummaryResponse{ProviderBreakdown: map[string]*ProviderStatsBreakdown{}}

	if s.writer != nil && s.writer.DB() != nil {
		var err error
		resp, err = statsSummaryFromStore(r.Context(), s.writer.DB())
		if err != nil {
			s.writeStatsError(w, http.StatusInternalServerError, "failed to query stats summary: "+err.Error())
			return
		}
	}

	respondJSON(w, http.StatusOK, resp)
}

// statsSummaryFromStore aggregates the requests/usage tables into the published
// summary shape: totals, distinct client keys, and a per-provider breakdown. A
// request row with no usage still counts as a request; a usage row only exists
// for a request that produced tokens.
func statsSummaryFromStore(ctx context.Context, db *sql.DB) (StatsSummaryResponse, error) {
	resp := StatsSummaryResponse{ProviderBreakdown: map[string]*ProviderStatsBreakdown{}}

	rows, err := db.QueryContext(ctx, `
SELECT
    COALESCE(r.provider, '') AS provider,
    COUNT(r.id) AS requests,
    COALESCE(SUM(u.prompt_tokens), 0) AS prompt_tokens,
    COALESCE(SUM(u.completion_tokens), 0) AS completion_tokens,
    COALESCE(SUM(u.cached_tokens), 0) AS cached_tokens,
    COALESCE(SUM(u.prompt_tokens + u.completion_tokens), 0) AS total_tokens,
    COALESCE(SUM(u.cost), 0.0) AS cost,
    SUM(CASE WHEN COALESCE(r.error_class, '') NOT IN ('', 'in_flight') THEN 1 ELSE 0 END) AS errors
FROM requests r
LEFT JOIN usage u ON r.id = u.request_id
GROUP BY provider
ORDER BY provider;
`)
	if err != nil {
		return resp, fmt.Errorf("query provider breakdown: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var (
			provider                                            string
			requests, prompt, completion, cached, total, errors int64
			cost                                                float64
		)
		if err := rows.Scan(&provider, &requests, &prompt, &completion, &cached, &total, &cost, &errors); err != nil {
			return resp, fmt.Errorf("scan provider breakdown row: %w", err)
		}

		resp.TotalRequests += requests
		resp.TotalPromptTokens += prompt
		resp.TotalCompletionTokens += completion
		resp.TotalCachedTokens += cached
		resp.TotalTokens += total
		resp.TotalCost += cost

		if provider == "" {
			provider = "unknown"
		}
		resp.ProviderBreakdown[provider] = &ProviderStatsBreakdown{
			Requests: requests,
			Tokens:   total,
			Errors:   errors,
		}
	}
	if err := rows.Err(); err != nil {
		return resp, fmt.Errorf("iterate provider breakdown rows: %w", err)
	}

	// Cost savings are the metered-API equivalent of the recorded traffic.
	resp.EstimatedCostSavedUSD = resp.TotalCost

	if err := db.QueryRowContext(ctx, `
SELECT COUNT(DISTINCT r.client_key_hash) FROM requests r WHERE COALESCE(r.client_key_hash, '') <> '';
`).Scan(&resp.ActiveClients); err != nil {
		return resp, fmt.Errorf("query active clients: %w", err)
	}

	return resp, nil
}

// ClientUsageRecord is one entry of the GET /api/stats/by-client array, as
// published in docs/openapi.yaml (components/schemas/ClientUsageRecord).
// client_id is the configured client key name when the digest is known, and the
// digest itself otherwise, so anonymous rows stay distinguishable.
type ClientUsageRecord struct {
	ClientID         string  `json:"client_id"`
	ClientName       string  `json:"client_name,omitempty"`
	ClientKeyHash    string  `json:"client_key_hash,omitempty"`
	RequestCount     int64   `json:"request_count"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	CachedTokens     int64   `json:"cached_tokens"`
	TotalTokens      int64   `json:"total_tokens"`
	Cost             float64 `json:"cost"`
	ErrorCount       int64   `json:"error_count"`
	RateLimitedCount int64   `json:"rate_limited_count"`
	LastActive       string  `json:"last_active"`
}

// handleStatsByClient serves GET /api/stats/by-client: per-client-key accounting
// over a lookback window (the `window` query parameter, e.g. 1h/24h/7d/30d,
// defaulting to 7d). The response is the array of records the OpenAPI contract
// promises, ordered by request count descending.
func (s *Server) handleStatsByClient(w http.ResponseWriter, r *http.Request) {
	window := r.URL.Query().Get("window")
	if window == "" {
		window = "7d"
	}
	dur, err := storage.ParseWindowDuration(window)
	if err != nil {
		s.writeStatsError(w, http.StatusBadRequest, err.Error())
		return
	}
	since := time.Now().UTC().Add(-dur).Format(time.RFC3339)

	if s.writer == nil || s.writer.DB() == nil {
		respondJSON(w, http.StatusOK, []ClientUsageRecord{})
		return
	}

	records, err := s.clientUsageFromStore(r.Context(), s.writer.DB(), since)
	if err != nil {
		s.writeStatsError(w, http.StatusInternalServerError, "failed to query stats by client: "+err.Error())
		return
	}
	if records == nil {
		records = []ClientUsageRecord{}
	}
	respondJSON(w, http.StatusOK, records)
}

// clientUsageFromStore aggregates the requests/usage tables per client key hash.
func (s *Server) clientUsageFromStore(ctx context.Context, db *sql.DB, since string) ([]ClientUsageRecord, error) {
	rows, err := db.QueryContext(ctx, `
SELECT
    COALESCE(r.client_key_hash, '') AS client_key_hash,
    COUNT(r.id) AS requests,
    COALESCE(SUM(u.prompt_tokens), 0) AS prompt_tokens,
    COALESCE(SUM(u.completion_tokens), 0) AS completion_tokens,
    COALESCE(SUM(u.cached_tokens), 0) AS cached_tokens,
    COALESCE(SUM(u.prompt_tokens + u.completion_tokens), 0) AS total_tokens,
    COALESCE(SUM(u.cost), 0.0) AS cost,
    SUM(CASE WHEN COALESCE(r.error_class, '') NOT IN ('', 'in_flight') THEN 1 ELSE 0 END) AS errors,
    SUM(CASE WHEN LOWER(COALESCE(r.error_class, '')) LIKE 'rate_limit%' THEN 1 ELSE 0 END) AS rate_limited,
    COALESCE(MAX(r.created_at), '') AS last_active
FROM requests r
LEFT JOIN usage u ON r.id = u.request_id
WHERE r.created_at >= ?
GROUP BY client_key_hash
ORDER BY requests DESC, client_key_hash ASC;
`, since)
	if err != nil {
		return nil, fmt.Errorf("query client usage: %w", err)
	}
	defer rows.Close()

	// Configured client names resolve the stored digest to something an
	// operator recognises; unknown digests are reported as-is.
	names := s.clientNameByHash()

	var records []ClientUsageRecord
	for rows.Next() {
		var rec ClientUsageRecord
		if err := rows.Scan(
			&rec.ClientKeyHash,
			&rec.RequestCount,
			&rec.PromptTokens,
			&rec.CompletionTokens,
			&rec.CachedTokens,
			&rec.TotalTokens,
			&rec.Cost,
			&rec.ErrorCount,
			&rec.RateLimitedCount,
			&rec.LastActive,
		); err != nil {
			return nil, fmt.Errorf("scan client usage row: %w", err)
		}

		if name, ok := names[rec.ClientKeyHash]; ok {
			rec.ClientID = name
			rec.ClientName = name
		} else {
			rec.ClientID = rec.ClientKeyHash
		}
		if rec.ClientID == "" {
			rec.ClientID = "anonymous"
		}
		records = append(records, rec)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate client usage rows: %w", err)
	}

	return records, nil
}

// clientNameByHash maps the sha256 hex digest stored in requests.client_key_hash
// to the configured client key name.
func (s *Server) clientNameByHash() map[string]string {
	out := make(map[string]string)
	if s == nil || s.cfg == nil {
		return out
	}
	for name, key := range s.cfg.Server.ClientKeys {
		if key == "" {
			continue
		}
		sum := sha256.Sum256([]byte(key))
		out[hex.EncodeToString(sum[:])] = name
	}
	return out
}

func respondJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func (s *Server) writeStatsError(w http.ResponseWriter, status int, msg string) {
	respondJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "invalid_request_error",
			"param":   nil,
			"code":    "stats_query_failed",
		},
	})
}
