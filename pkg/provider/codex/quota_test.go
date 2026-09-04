package codex

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// TestQuotaNotLoggedIn: with no credential store entry and no static token the
// lane cannot query wham/usage. That must surface as a clean snapshot telling
// the operator to log in \u2014 not as an error, which MCP get_quota_status renders
// as "error getting quota: codex: wham usage request failed: codex: no access
// token available".
func TestQuotaNotLoggedIn(t *testing.T) {
	var requests int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requests, 1)
		_ = json.NewEncoder(w).Encode(map[string]any{})
	}))
	defer srv.Close()

	p := New(Config{BaseURL: srv.URL})

	snap, err := p.Quota(context.Background())
	if err != nil {
		t.Fatalf("Quota returned an error for a logged-out lane: %v", err)
	}
	if snap == nil {
		t.Fatal("Quota returned a nil snapshot")
	}
	if snap.ObservedAt.IsZero() {
		t.Error("ObservedAt not set")
	}
	if len(snap.Windows) != 0 {
		t.Errorf("expected empty windows, got %+v", snap.Windows)
	}
	if !strings.Contains(snap.Detail, "not logged in") || !strings.Contains(snap.Detail, "ultiproxy login codex") {
		t.Errorf("Detail = %q, want guidance to run 'ultiproxy login codex'", snap.Detail)
	}
	if got := atomic.LoadInt32(&requests); got != 0 {
		t.Errorf("logged-out lane still made %d upstream request(s)", got)
	}
}
