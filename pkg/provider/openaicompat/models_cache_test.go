package openaicompat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestCachedModelsDoesNotProbeUpstream: CachedModels must return whatever the
// startup discovery (or a later FetchModels) cached and never contact the
// upstream \u2014 the aggregated /v1/models handler calls it on every request.
func TestCachedModelsDoesNotProbeUpstream(t *testing.T) {
	var models int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&models, 1)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data":   []map[string]any{{"id": "Qwen/Qwen3.8-27B-Instruct"}},
		})
	}))
	defer srv.Close()

	p, err := New(Config{
		BaseURL:    srv.URL,
		HTTPClient: srv.Client(),
		Quirks:     Quirks{ModelListPassthrough: true},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if got := atomic.LoadInt32(&models); got != 1 {
		t.Fatalf("expected exactly one startup discovery request, got %d", got)
	}

	for i := 0; i < 5; i++ {
		got := p.CachedModels()
		if len(got) != 1 || got[0] != "Qwen/Qwen3.8-27B-Instruct" {
			t.Fatalf("CachedModels = %v, want [Qwen/Qwen3.8-27B-Instruct]", got)
		}
	}
	if got := atomic.LoadInt32(&models); got != 1 {
		t.Errorf("CachedModels probed the upstream: %d requests", got)
	}

	// Even with the upstream gone, the cache still answers.
	srv.Close()
	got := p.CachedModels()
	if len(got) != 1 {
		t.Errorf("CachedModels after upstream shutdown = %v", got)
	}
}

// TestCachedModelsEmptyWhenNothingDiscovered: a non-passthrough lane (or a
// passthrough lane whose discovery failed) reports no cached models rather
// than a placeholder id.
func TestCachedModelsEmptyWhenNothingDiscovered(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()

	p, err := New(Config{BaseURL: srv.URL, HTTPClient: srv.Client()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := p.CachedModels(); len(got) != 0 {
		t.Errorf("CachedModels = %v, want empty", got)
	}
}

// grokFrame wraps a protobuf payload in a single gRPC-web data frame.
func grokFrame(payload []byte) []byte {
	out := make([]byte, 5+len(payload))
	out[0] = 0x00
	out[1] = byte(len(payload) >> 24)
	out[2] = byte(len(payload) >> 16)
	out[3] = byte(len(payload) >> 8)
	out[4] = byte(len(payload))
	copy(out[5:], payload)
	return out
}

// TestParseGrokCreditsResponse_NoCreditPools: a billing payload that carries no
// recognizable pools (free / no-credit account, or a spending limit already
// reached) must come back as a valid snapshot with empty windows and an
// explanatory Detail \u2014 not as "no recognizable credit pools (status unknown)"
// and not as an error.
func TestParseGrokCreditsResponse_NoCreditPools(t *testing.T) {
	// Field 1 varint (42) + field 2 string ("ok") \u2014 no percentages, no resets,
	// no pool names, so nothing can be turned into a quota window.
	payload := []byte{0x08, 0x2a, 0x12, 0x02, 'o', 'k'}
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)

	snap, err := ParseGrokCreditsResponse(grokFrame(payload), now)
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse returned an error: %v", err)
	}
	if snap == nil {
		t.Fatal("nil snapshot")
	}
	if !snap.ObservedAt.Equal(now) {
		t.Errorf("ObservedAt = %v, want %v", snap.ObservedAt, now)
	}
	if len(snap.Windows) != 0 {
		t.Errorf("expected empty windows, got %+v", snap.Windows)
	}
	for _, bad := range []string{"status unknown", "no recognizable credit pools"} {
		if strings.Contains(snap.Detail, bad) {
			t.Errorf("Detail = %q, must not say %q", snap.Detail, bad)
		}
	}
	if !strings.Contains(snap.Detail, "no credit pools") {
		t.Errorf("Detail = %q, want it to explain the account has no credit pools", snap.Detail)
	}
	if !strings.Contains(snap.Detail, "spending limit") {
		t.Errorf("Detail = %q, want it to mention the spending limit", snap.Detail)
	}
}
