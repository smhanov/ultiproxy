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
