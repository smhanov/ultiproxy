package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/smhanov/ultiproxy/pkg/state"
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

func (s *Server) handleStatsSummary(w http.ResponseWriter, r *http.Request) {
	// Provide the stats summary served by GET /api/stats/summary (the
	// numbers behind the MCP get_client_usage tool).
	type StatsSummary struct {
		TotalRequests int64   `json:"total_requests"`
		TotalTokens   int64   `json:"total_tokens"`
		TotalCost     float64 `json:"total_cost"`
	}

	res := StatsSummary{}
	if s.writer != nil && s.writer.DB() != nil {
		_ = s.writer.DB().QueryRowContext(r.Context(), "SELECT COUNT(*) FROM requests").Scan(&res.TotalRequests)
		_ = s.writer.DB().QueryRowContext(r.Context(), "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0), COALESCE(SUM(cost), 0.0) FROM usage").Scan(&res.TotalTokens, &res.TotalCost)
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(res)
}
