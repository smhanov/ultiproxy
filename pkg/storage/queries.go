package storage

import (
	"context"
	"database/sql"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// DailySummaryRow represents aggregated usage by day and client.
type DailySummaryRow struct {
	Date             string  `json:"date"`
	ClientKeyHash    string  `json:"client_key_hash"`
	Requests         int64   `json:"requests"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	ReasoningTokens  int64   `json:"reasoning_tokens"`
	CachedTokens     int64   `json:"cached_tokens"`
	TotalTokens      int64   `json:"total_tokens"`
	Cost             float64 `json:"cost"`
}

// ClientUsageRow represents aggregated usage per client within a time window.
type ClientUsageRow struct {
	ClientKeyHash    string  `json:"client_key_hash"`
	Requests         int64   `json:"requests"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	ReasoningTokens  int64   `json:"reasoning_tokens"`
	CachedTokens     int64   `json:"cached_tokens"`
	TotalTokens      int64   `json:"total_tokens"`
	Cost             float64 `json:"cost"`
	FirstSeen        string  `json:"first_seen"`
	LastSeen         string  `json:"last_seen"`
}

// ParseWindowDuration parses duration strings like "7d", "24h", "30d", "1h".
func ParseWindowDuration(window string) (time.Duration, error) {
	window = strings.TrimSpace(strings.ToLower(window))
	if window == "" {
		return 7 * 24 * time.Hour, nil
	}
	if strings.HasSuffix(window, "d") {
		daysStr := strings.TrimSuffix(window, "d")
		days, err := strconv.Atoi(daysStr)
		if err != nil {
			return 0, fmt.Errorf("invalid day duration %q: %w", window, err)
		}
		return time.Duration(days) * 24 * time.Hour, nil
	}
	dur, err := time.ParseDuration(window)
	if err != nil {
		return 0, fmt.Errorf("invalid window duration %q: %w", window, err)
	}
	return dur, nil
}

// ListUsageSummary queries daily totals per day and client key hash.
func ListUsageSummary(ctx context.Context, db *sql.DB) ([]DailySummaryRow, error) {
	const query = `
SELECT 
    substr(r.created_at, 1, 10) as day,
    COALESCE(r.client_key_hash, '') as client_key_hash,
    COUNT(r.id) as requests,
    COALESCE(SUM(u.prompt_tokens), 0) as prompt_tokens,
    COALESCE(SUM(u.completion_tokens), 0) as completion_tokens,
    COALESCE(SUM(u.reasoning_tokens), 0) as reasoning_tokens,
    COALESCE(SUM(u.cached_tokens), 0) as cached_tokens,
    COALESCE(SUM(u.prompt_tokens + u.completion_tokens), 0) as total_tokens,
    COALESCE(SUM(u.cost), 0.0) as cost
FROM requests r
LEFT JOIN usage u ON r.id = u.request_id
GROUP BY day, client_key_hash
ORDER BY day DESC, requests DESC;
`
	rows, err := db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("query usage summary: %w", err)
	}
	defer rows.Close()

	var results []DailySummaryRow
	for rows.Next() {
		var row DailySummaryRow
		if err := rows.Scan(
			&row.Date,
			&row.ClientKeyHash,
			&row.Requests,
			&row.PromptTokens,
			&row.CompletionTokens,
			&row.ReasoningTokens,
			&row.CachedTokens,
			&row.TotalTokens,
			&row.Cost,
		); err != nil {
			return nil, fmt.Errorf("scan usage summary row: %w", err)
		}
		results = append(results, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("rows usage summary: %w", err)
	}
	return results, nil
}

// ListUsageByClient queries client totals within the given window (e.g. "7d").
func ListUsageByClient(ctx context.Context, db *sql.DB, window string) ([]ClientUsageRow, error) {
	dur, err := ParseWindowDuration(window)
	if err != nil {
		return nil, err
	}

	since := time.Now().UTC().Add(-dur).Format(time.RFC3339)

	const query = `
SELECT 
    COALESCE(r.client_key_hash, '') as client_key_hash,
    COUNT(r.id) as requests,
    COALESCE(SUM(u.prompt_tokens), 0) as prompt_tokens,
    COALESCE(SUM(u.completion_tokens), 0) as completion_tokens,
    COALESCE(SUM(u.reasoning_tokens), 0) as reasoning_tokens,
    COALESCE(SUM(u.cached_tokens), 0) as cached_tokens,
    COALESCE(SUM(u.prompt_tokens + u.completion_tokens), 0) as total_tokens,
    COALESCE(SUM(u.cost), 0.0) as cost,
    COALESCE(MIN(r.created_at), '') as first_seen,
    COALESCE(MAX(r.created_at), '') as last_seen
FROM requests r
LEFT JOIN usage u ON r.id = u.request_id
WHERE r.created_at >= ?
GROUP BY client_key_hash
ORDER BY requests DESC;
`
	rows, err := db.QueryContext(ctx, query, since)
	if err != nil {
		return nil, fmt.Errorf("query usage by client: %w", err)
	}
	defer rows.Close()

	var results []ClientUsageRow
	for rows.Next() {
		var row ClientUsageRow
		if err := rows.Scan(
			&row.ClientKeyHash,
			&row.Requests,
			&row.PromptTokens,
			&row.CompletionTokens,
			&row.ReasoningTokens,
			&row.CachedTokens,
			&row.TotalTokens,
			&row.Cost,
			&row.FirstSeen,
			&row.LastSeen,
		); err != nil {
			return nil, fmt.Errorf("scan client usage row: %w", err)
		}
		results = append(results, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("rows client usage: %w", err)
	}
	return results, nil
}

// ListUsageSummary queries daily usage summaries using the writer's underlying DB.
func (w *Writer) ListUsageSummary(ctx context.Context) ([]DailySummaryRow, error) {
	return ListUsageSummary(ctx, w.db)
}

// ListUsageByClient queries usage per client using the writer's underlying DB.
func (w *Writer) ListUsageByClient(ctx context.Context, window string) ([]ClientUsageRow, error) {
	return ListUsageByClient(ctx, w.db, window)
}
