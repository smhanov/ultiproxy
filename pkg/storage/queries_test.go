package storage

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestUsageQueries(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "telemetry.db")

	w, err := NewWriter(dbPath, WithBatchSize(1))
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer w.Close()

	now := time.Now().UTC()
	today := now.Format(time.RFC3339)
	yesterday := now.Add(-24 * time.Hour).Format(time.RFC3339)

	// Insert requests and usage
	if err := w.TrackRequest(RequestRecord{
		ID:            1,
		ClientKeyHash: "client-a",
		CreatedAt:     today,
	}); err != nil {
		t.Fatalf("TrackRequest: %v", err)
	}
	if err := w.TrackUsage(UsageRecord{
		RequestID:        1,
		PromptTokens:     100,
		CompletionTokens: 50,
		Cost:             0.02,
	}); err != nil {
		t.Fatalf("TrackUsage: %v", err)
	}

	if err := w.TrackRequest(RequestRecord{
		ID:            2,
		ClientKeyHash: "client-b",
		CreatedAt:     yesterday,
	}); err != nil {
		t.Fatalf("TrackRequest: %v", err)
	}
	if err := w.TrackUsage(UsageRecord{
		RequestID:        2,
		PromptTokens:     200,
		CompletionTokens: 100,
		Cost:             0.05,
	}); err != nil {
		t.Fatalf("TrackUsage: %v", err)
	}

	// Wait for async flush
	time.Sleep(50 * time.Millisecond)

	ctx := context.Background()

	// Test ListUsageSummary
	summaries, err := w.ListUsageSummary(ctx)
	if err != nil {
		t.Fatalf("ListUsageSummary: %v", err)
	}
	if len(summaries) != 2 {
		t.Fatalf("expected 2 summary rows, got %d", len(summaries))
	}

	// Test ListUsageByClient
	clients7d, err := w.ListUsageByClient(ctx, "7d")
	if err != nil {
		t.Fatalf("ListUsageByClient: %v", err)
	}
	if len(clients7d) != 2 {
		t.Fatalf("expected 2 clients in 7d, got %d", len(clients7d))
	}

	clients1h, err := w.ListUsageByClient(ctx, "1h")
	if err != nil {
		t.Fatalf("ListUsageByClient(1h): %v", err)
	}
	if len(clients1h) != 1 {
		t.Fatalf("expected 1 client in 1h, got %d", len(clients1h))
	}
	if clients1h[0].ClientKeyHash != "client-a" {
		t.Errorf("expected client-a, got %s", clients1h[0].ClientKeyHash)
	}
}

func TestGetClientUsage(t *testing.T) {
	dir := t.TempDir()
	w, err := NewWriter(filepath.Join(dir, "usage.db"), WithBatchSize(1))
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer w.Close()

	now := time.Now().UTC()
	seed := func(id int64, client string, createdAt string, prompt, completion, cached int64, cost float64) {
		t.Helper()
		if err := w.TrackRequest(RequestRecord{ID: id, ClientKeyHash: client, CreatedAt: createdAt}); err != nil {
			t.Fatalf("TrackRequest: %v", err)
		}
		if err := w.TrackUsage(UsageRecord{RequestID: id, PromptTokens: prompt, CompletionTokens: completion, CachedTokens: cached, Cost: cost}); err != nil {
			t.Fatalf("TrackUsage: %v", err)
		}
	}
	seed(1, "client-a", now.Format(time.RFC3339), 100, 50, 10, 0.02)
	seed(2, "client-b", now.Add(-2*time.Hour).Format(time.RFC3339), 200, 100, 0, 0.05)
	time.Sleep(100 * time.Millisecond)

	ctx := context.Background()

	// Aggregate over every client.
	overall, err := w.GetClientUsage(ctx, "", "7d")
	if err != nil {
		t.Fatalf("GetClientUsage: %v", err)
	}
	if overall.Requests != 2 || overall.PromptTokens != 300 || overall.CompletionTokens != 150 ||
		overall.CachedTokens != 10 || overall.TotalTokens != 450 || overall.Cost < 0.069 || overall.Cost > 0.071 {
		t.Errorf("unexpected aggregate totals: %+v", overall)
	}
	if overall.FirstSeen == "" || overall.LastSeen == "" {
		t.Errorf("expected first_seen/last_seen to be populated: %+v", overall)
	}

	// Single client filter.
	a, err := w.GetClientUsage(ctx, "client-a", "7d")
	if err != nil {
		t.Fatalf("GetClientUsage(client-a): %v", err)
	}
	if a.Requests != 1 || a.TotalTokens != 150 || a.Cost < 0.019 || a.Cost > 0.021 || a.ClientKeyHash != "client-a" {
		t.Errorf("unexpected client-a totals: %+v", a)
	}

	// Unknown client hash reports zeros, not an error.
	unknown, err := w.GetClientUsage(ctx, "nobody", "7d")
	if err != nil {
		t.Fatalf("GetClientUsage(unknown): %v", err)
	}
	if unknown.Requests != 0 || unknown.TotalTokens != 0 || unknown.Cost != 0 {
		t.Errorf("expected zero totals for unknown client: %+v", unknown)
	}

	// Window filtering excludes the 2h-old row.
	windowed, err := w.GetClientUsage(ctx, "", "1h")
	if err != nil {
		t.Fatalf("GetClientUsage(1h): %v", err)
	}
	if windowed.Requests != 1 || windowed.TotalTokens != 150 {
		t.Errorf("unexpected 1h window totals: %+v", windowed)
	}

	// Invalid window is an error, not silently ignored.
	if _, err := w.GetClientUsage(ctx, "", "not-a-window"); err == nil {
		t.Error("expected an error for an invalid window string")
	}

	// Rows without a client key hash (admin/unauthenticated lanes) only count
	// toward the aggregate view.
	if err := w.TrackRequest(RequestRecord{ID: 3, CreatedAt: now.Format(time.RFC3339)}); err != nil {
		t.Fatalf("TrackRequest: %v", err)
	}
	if err := w.TrackUsage(UsageRecord{RequestID: 3, PromptTokens: 7, CompletionTokens: 3, Cost: 0.001}); err != nil {
		t.Fatalf("TrackUsage: %v", err)
	}
	time.Sleep(100 * time.Millisecond)
	anon, err := w.GetClientUsage(ctx, "", "7d")
	if err != nil {
		t.Fatalf("GetClientUsage(with anonymous row): %v", err)
	}
	if anon.Requests != 3 || anon.TotalTokens != 460 {
		t.Errorf("anonymous rows must count toward the aggregate: %+v", anon)
	}
	if empty, err := w.GetClientUsage(ctx, "", "1h"); err != nil || empty.Requests != 2 || empty.TotalTokens != 160 {
		t.Errorf("unexpected 1h totals with anonymous row: %+v err=%v", empty, err)
	}
}

func TestGetClientUsageEmptyDatabase(t *testing.T) {
	dir := t.TempDir()
	w, err := NewWriter(filepath.Join(dir, "empty.db"), WithBatchSize(1))
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer w.Close()

	totals, err := w.GetClientUsage(context.Background(), "", "")
	if err != nil {
		t.Fatalf("GetClientUsage on an empty DB: %v", err)
	}
	if totals.Requests != 0 || totals.TotalTokens != 0 || totals.Cost != 0 || totals.FirstSeen != "" {
		t.Errorf("expected zero totals on an empty DB, got %+v", totals)
	}
}
