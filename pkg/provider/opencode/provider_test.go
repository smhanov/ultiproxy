package opencode

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestOpenCodeParseWorkspaceHTMLFixture(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "opencode", "workspace.html")
	htmlBytes, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	refTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	snap, err := ParseWorkspaceHTML(string(htmlBytes), refTime)
	if err != nil {
		t.Fatalf("ParseWorkspaceHTML failed: %v", err)
	}

	if len(snap.Windows) != 3 {
		t.Fatalf("expected 3 windows, got %d", len(snap.Windows))
	}

	// 1. Rolling
	rolling := snap.Windows[0]
	if rolling.Label != "Rolling" || rolling.UsedPct != 42.5 || rolling.SecondsRemaining != 3600 || rolling.Unit != "%" {
		t.Errorf("unexpected Rolling window: %+v", rolling)
	}
	expectedRollingReset := refTime.Add(3600 * time.Second)
	if !rolling.ResetAt.Equal(expectedRollingReset) {
		t.Errorf("expected ResetAt %v, got %v", expectedRollingReset, rolling.ResetAt)
	}

	// 2. Weekly
	weekly := snap.Windows[1]
	if weekly.Label != "Weekly" || weekly.UsedPct != 65.0 || weekly.SecondsRemaining != 86400 || weekly.Unit != "%" {
		t.Errorf("unexpected Weekly window: %+v", weekly)
	}

	// 3. Monthly
	monthly := snap.Windows[2]
	if monthly.Label != "Monthly" || monthly.UsedPct != 88.2 || monthly.SecondsRemaining != 604800 || monthly.Unit != "%" {
		t.Errorf("unexpected Monthly window: %+v", monthly)
	}
}

func TestOpenCodeHTTPAuthAndScrape(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "opencode", "workspace.html")
	htmlBytes, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	var capturedCookie string
	var capturedPath string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedCookie = r.Header.Get("Cookie")
		capturedPath = r.URL.Path

		if capturedCookie != "session=my-secret-session-cookie" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write(htmlBytes)
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:       server.URL,
		WorkspaceID:   "ws-abc-123",
		SessionCookie: "my-secret-session-cookie",
		HTTPClient:    server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	snap, err := p.Quota(context.Background())
	if err != nil {
		t.Fatalf("Quota failed: %v", err)
	}

	if capturedPath != "/workspace/ws-abc-123" {
		t.Errorf("expected path /workspace/ws-abc-123, got %s", capturedPath)
	}
	if capturedCookie != "session=my-secret-session-cookie" {
		t.Errorf("expected session cookie header, got %s", capturedCookie)
	}
	if len(snap.Windows) != 3 {
		t.Errorf("expected 3 windows, got %d", len(snap.Windows))
	}
}
