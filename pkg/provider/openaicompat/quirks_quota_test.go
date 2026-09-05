package openaicompat

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// goldenGrokCreditsHex is a real 73-byte GetGrokCreditsConfig response captured
// live from grok.com (HTTP 200, gRPC-web framing): one 48-byte data frame
// followed by a 15-byte trailer frame carrying "grpc-status:0\r\n".
const goldenGrokCreditsHex = "00000000300a2e12001a0022060880c2c8d4062a060880b7edd4064212080212060880c2c8d4061a060880b7edd406580162006801800000000f677270632d7374617475733a300d0a"

// goldenGrokCreditsPayloadHex is the protobuf message inside the data frame:
//
//	field 1 message {
//	  2: "", 3: "",
//	  4: {1: 1787961600}       // window start  2026-08-29T00:00:00Z
//	  5: {1: 1788566400}       // window reset  2026-09-05T00:00:00Z
//	  8: {1: 2, 2: {1: start}, 3: {1: end}}   // state 2 == active
//	  11: 1, 12: "", 13: 1,    // flags
//	}
const goldenGrokCreditsPayloadHex = "0a2e12001a0022060880c2c8d4062a060880b7edd4064212080212060880c2c8d4061a060880b7edd406580162006801"

func mustHex(t *testing.T, s string) []byte {
	t.Helper()
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatalf("bad hex fixture: %v", err)
	}
	return b
}

// grokGoldenTimes are the timestamps encoded in the golden capture.
var (
	goldenStart = time.Date(2026, 8, 29, 0, 0, 0, 0, time.UTC)
	goldenReset = time.Date(2026, 9, 5, 0, 0, 0, 0, time.UTC)
)

// ----------------------------------------------------------------------------
// protobuf test builders
// ----------------------------------------------------------------------------

func appendProtoVarint(b []byte, v uint64) []byte {
	for v >= 0x80 {
		b = append(b, byte(v)|0x80)
		v >>= 7
	}
	return append(b, byte(v))
}

func appendProtoKey(b []byte, field, wire int) []byte {
	return appendProtoVarint(b, uint64(field)<<3|uint64(wire))
}

func appendVarintField(b []byte, field int, v uint64) []byte {
	b = appendProtoKey(b, field, 0)
	return appendProtoVarint(b, v)
}

func appendFixed32Field(b []byte, field int, f float32) []byte {
	b = appendProtoKey(b, field, 5)
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], math.Float32bits(f))
	return append(b, buf[:]...)
}

func appendMessageField(b []byte, field int, sub []byte) []byte {
	b = appendProtoKey(b, field, 2)
	b = appendProtoVarint(b, uint64(len(sub)))
	return append(b, sub...)
}

// grokDataFrame wraps payload in a gRPC-web data frame (flag byte 0).
func grokDataFrame(payload []byte) []byte {
	out := make([]byte, 5+len(payload))
	binary.BigEndian.PutUint32(out[1:5], uint32(len(payload)))
	copy(out[5:], payload)
	return out
}

// grokTrailerFrame wraps payload in a gRPC-web trailer frame (flag byte 0x80).
func grokTrailerFrame(payload []byte) []byte {
	out := make([]byte, 5+len(payload))
	out[0] = 0x80
	binary.BigEndian.PutUint32(out[1:5], uint32(len(payload)))
	copy(out[5:], payload)
	return out
}

// ----------------------------------------------------------------------------
// 1. golden live capture
// ----------------------------------------------------------------------------

// TestParseGrokCreditsResponse_GoldenLiveCapture feeds the real captured bytes
// through the frame parser and the window heuristics.
func TestParseGrokCreditsResponse_GoldenLiveCapture(t *testing.T) {
	body := mustHex(t, goldenGrokCreditsHex)
	payload := mustHex(t, goldenGrokCreditsPayloadHex)
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC) // inside the window

	// Framing: exactly one data frame; the trailer frame is skipped.
	frames, err := UnframeGRPCWeb(body)
	if err != nil {
		t.Fatalf("UnframeGRPCWeb: %v", err)
	}
	if len(frames) != 1 || string(frames[0]) != string(payload) {
		t.Fatalf("UnframeGRPCWeb = %d frames %x, want 1 frame %x", len(frames), frames, payload)
	}

	// Window heuristics.
	w := parseGrokCredits(payload, now)
	if !w.Start.Equal(goldenStart) {
		t.Errorf("Start = %v, want %v", w.Start, goldenStart)
	}
	if !w.Reset.Equal(goldenReset) {
		t.Errorf("Reset = %v, want %v", w.Reset, goldenReset)
	}
	if !w.Active {
		t.Error("Active = false, want true (field 8 state 2)")
	}
	if w.HasPct || w.Pct != 0 {
		t.Errorf("Pct = %v (HasPct %v), want 0 with HasPct false (no usage recorded yet)", w.Pct, w.HasPct)
	}

	// Snapshot mapping.
	snap, err := ParseGrokCreditsResponse(body, now)
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse: %v", err)
	}
	if !snap.ObservedAt.Equal(now) {
		t.Errorf("ObservedAt = %v, want %v", snap.ObservedAt, now)
	}
	if len(snap.Windows) != 1 {
		t.Fatalf("Windows = %+v, want exactly one", snap.Windows)
	}
	win := snap.Windows[0]
	if win.Label != "Grok Build" {
		t.Errorf("Label = %q, want %q", win.Label, "Grok Build")
	}
	if win.UsedPct != 0 {
		t.Errorf("UsedPct = %v, want 0", win.UsedPct)
	}
	if win.Remaining != 100 || win.Limit != 100 || win.Unit != "%" {
		t.Errorf("Remaining/Limit/Unit = %v/%v/%q, want 100/100/%%", win.Remaining, win.Limit, win.Unit)
	}
	if !win.ResetAt.Equal(goldenReset) {
		t.Errorf("ResetAt = %v, want %v", win.ResetAt, goldenReset)
	}
	if win.SecondsRemaining <= 0 {
		t.Errorf("SecondsRemaining = %d, want > 0 (now %v is before the reset %v)",
			win.SecondsRemaining, now, goldenReset)
	}
	wantDetail := "Grok Build window active (Aug 29 - Sep 5 2026), no usage recorded yet"
	if snap.Detail != wantDetail {
		t.Errorf("Detail = %q, want %q", snap.Detail, wantDetail)
	}
}

// ----------------------------------------------------------------------------
// 2. usage percent carried by a fixed32 under a field ending in 1
// ----------------------------------------------------------------------------

func TestParseGrokCreditsResponse_PercentPath(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)

	// field 1 { field 1: fixed32 0.68 }  -> 68% used.
	sub := appendFixed32Field(nil, 1, 0.68)
	payload := appendMessageField(nil, 1, sub)

	w := parseGrokCredits(payload, now)
	if !w.HasPct || w.Pct != 68 {
		t.Fatalf("Pct = %v (HasPct %v), want 68 (fraction 0.68 scaled to percent)", w.Pct, w.HasPct)
	}

	snap, err := ParseGrokCreditsResponse(grokDataFrame(payload), now)
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse: %v", err)
	}
	if len(snap.Windows) != 1 {
		t.Fatalf("Windows = %+v, want exactly one", snap.Windows)
	}
	if snap.Windows[0].UsedPct != 68 {
		t.Errorf("UsedPct = %v, want 68", snap.Windows[0].UsedPct)
	}
	if snap.Windows[0].Remaining != 32 {
		t.Errorf("Remaining = %v, want 32", snap.Windows[0].Remaining)
	}

	// With a reset timestamp the detail names it.
	payload = append(payload[:0:0], payload...)
	payload = appendMessageField(payload, 5, appendVarintField(nil, 1, uint64(goldenReset.Unix())))
	snap, err = ParseGrokCreditsResponse(grokDataFrame(payload), now)
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse: %v", err)
	}
	if !snap.Windows[0].ResetAt.Equal(goldenReset) {
		t.Errorf("ResetAt = %v, want %v", snap.Windows[0].ResetAt, goldenReset)
	}
	want := "Grok Build 68% used \u00b7 resets Sep 5 2026"
	if snap.Detail != want {
		t.Errorf("Detail = %q, want %q", snap.Detail, want)
	}
}

// TestGrokUsagePercent_Heuristic: candidates are
// fixed32 values in [0,100] whose path ends with field 1, pct is the smallest,
// and values <= 1 are fractions.
func TestGrokUsagePercent_Heuristic(t *testing.T) {
	fields := []scannedField{
		{Path: []int{1, 1}, Wire: 5, Float: 0.68},
		{Path: []int{1, 2}, Wire: 5, Float: 100}, // wrong path: not a candidate
	}
	if pct, ok := grokUsagePercent(fields); !ok || pct != 68 {
		t.Errorf("grokUsagePercent = %v (%v), want 68 (true)", pct, ok)
	}

	fields = []scannedField{
		{Path: []int{1, 1}, Wire: 5, Float: 15},
		{Path: []int{2, 1}, Wire: 5, Float: 45},
	}
	if pct, ok := grokUsagePercent(fields); !ok || pct != 15 {
		t.Errorf("grokUsagePercent = %v (%v), want 15 (min candidate)", pct, ok)
	}

	if pct, ok := grokUsagePercent(nil); ok || pct != 0 {
		t.Errorf("grokUsagePercent(no candidates) = %v (%v), want 0 (false)", pct, ok)
	}
}

// ----------------------------------------------------------------------------
// 3. trailer frames are skipped
// ----------------------------------------------------------------------------

func TestParseGrokCreditsResponse_TrailerFrameIgnored(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	trailer := grokTrailerFrame([]byte("grpc-status:0\r\n"))

	frames, err := UnframeGRPCWeb(trailer)
	if err != nil {
		t.Fatalf("UnframeGRPCWeb: %v", err)
	}
	if len(frames) != 0 {
		t.Fatalf("UnframeGRPCWeb(trailer) = %d frames (%x), want 0", len(frames), frames)
	}

	snap, err := ParseGrokCreditsResponse(trailer, now)
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse: %v", err)
	}
	if len(snap.Windows) != 0 {
		t.Errorf("trailer-only body produced windows: %+v", snap.Windows)
	}
	if !strings.Contains(snap.Detail, "no credit pools") {
		t.Errorf("Detail = %q, want the no-credit-pools explanation", snap.Detail)
	}

	// A data frame followed by a trailer parses exactly like the data frame
	// alone.
	golden := mustHex(t, goldenGrokCreditsHex)
	withTrailer := append(append([]byte{}, golden[:5+48]...), trailer...)

	plain, err := ParseGrokCreditsResponse(golden, now)
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse: %v", err)
	}
	combined, err := ParseGrokCreditsResponse(withTrailer, now)
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse: %v", err)
	}
	if combined.Detail != plain.Detail || len(combined.Windows) != len(plain.Windows) {
		t.Errorf("trailer changed the result: %+v vs %+v", combined, plain)
	}
}

// ----------------------------------------------------------------------------
// 4. empty body keeps the no-credit-pools behaviour
// ----------------------------------------------------------------------------

func TestParseGrokCreditsResponse_EmptyBody(t *testing.T) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)

	snap, err := ParseGrokCreditsResponse(nil, now)
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse: %v", err)
	}
	if len(snap.Windows) != 0 {
		t.Errorf("expected empty windows, got %+v", snap.Windows)
	}
	if !strings.Contains(snap.Detail, "no credit pools") {
		t.Errorf("Detail = %q, want the no-credit-pools explanation", snap.Detail)
	}
	if !strings.Contains(snap.Detail, "spending limit") {
		t.Errorf("Detail = %q, want it to mention the spending limit", snap.Detail)
	}

	// Same for a bare empty data frame.
	snap, err = ParseGrokCreditsResponse(grokDataFrame(nil), now)
	if err != nil {
		t.Fatalf("ParseGrokCreditsResponse: %v", err)
	}
	if len(snap.Windows) != 0 {
		t.Errorf("expected empty windows, got %+v", snap.Windows)
	}
}

// ----------------------------------------------------------------------------
// 5. the HTTP request uses the gRPC-web wire format
// ----------------------------------------------------------------------------

func TestFetchCreditsQuota_GRPCWebRequest(t *testing.T) {
	golden := mustHex(t, goldenGrokCreditsHex)

	var (
		gotMethod string
		gotPath   string
		gotBody   []byte
		gotHeader http.Header
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotHeader = r.Header.Clone()
		var err error
		gotBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		w.Header().Set("Content-Type", grokGRPCWebContentType)
		_, _ = w.Write(golden)
	}))
	defer srv.Close()

	snap, err := fetchCreditsQuota(context.Background(), srv.Client(), srv.URL, "tok-123")
	if err != nil {
		t.Fatalf("fetchCreditsQuota: %v", err)
	}

	if gotMethod != http.MethodPost {
		t.Errorf("method = %q, want POST", gotMethod)
	}
	if gotPath != "/" {
		t.Errorf("path = %q, want the configured URL path", gotPath)
	}

	// The body must be the 5-byte gRPC-web frame for an empty protobuf message.
	if len(gotBody) != 5 || gotBody[0] != 0 || gotBody[1] != 0 || gotBody[2] != 0 || gotBody[3] != 0 || gotBody[4] != 0 {
		t.Errorf("request body = %x, want 5 zero bytes (empty protobuf request message)", gotBody)
	}

	wantHeaders := map[string]string{
		"Content-Type": grokGRPCWebContentType,
		"X-Grpc-Web":   "1",
		"X-User-Agent": "connect-es/2.1.1",
		"Origin":       "https://grok.com",
		"Referer":      "https://grok.com/?_s=usage",
		"Accept":       "*/*",
	}
	for name, want := range wantHeaders {
		if got := gotHeader.Get(name); got != want {
			t.Errorf("header %q = %q, want %q", name, got, want)
		}
	}
	if got := gotHeader.Get("Authorization"); got != "Bearer tok-123" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer tok-123")
	}

	// And the golden response parses into a single Grok Build window.
	if len(snap.Windows) != 1 {
		t.Fatalf("Windows = %+v, want exactly one", snap.Windows)
	}
	win := snap.Windows[0]
	if win.Label != "Grok Build" || win.Unit != "%" || win.Limit != 100 {
		t.Errorf("window = %+v, want a 100%% Grok Build window", win)
	}
	if !win.ResetAt.IsZero() && win.ResetAt.After(snap.ObservedAt) && win.SecondsRemaining <= 0 {
		t.Errorf("SecondsRemaining = %d with reset %v in the future", win.SecondsRemaining, win.ResetAt)
	}
	if snap.Detail == "" {
		t.Error("Detail is empty")
	}
}

// TestFetchCreditsQuota_NoTokenOmitsAuthorization: an empty token must not send
// a bearer header at all.
func TestFetchCreditsQuota_NoTokenOmitsAuthorization(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", grokGRPCWebContentType)
		_, _ = w.Write(grokDataFrame(nil))
	}))
	defer srv.Close()

	if _, err := fetchCreditsQuota(context.Background(), srv.Client(), srv.URL, ""); err != nil {
		t.Fatalf("fetchCreditsQuota: %v", err)
	}
	if gotAuth != "" {
		t.Errorf("Authorization = %q, want empty", gotAuth)
	}
}

// TestFetchCreditsQuota_EmptyUpstreamResponse: HTTP 200 with zero bytes (what
// the old JSON request used to get) still reports the no-credit-pools snapshot
// instead of inventing numbers.
func TestFetchCreditsQuota_EmptyUpstreamResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", grokGRPCWebContentType)
	}))
	defer srv.Close()

	snap, err := fetchCreditsQuota(context.Background(), srv.Client(), srv.URL, "tok")
	if err != nil {
		t.Fatalf("fetchCreditsQuota: %v", err)
	}
	if len(snap.Windows) != 0 {
		t.Errorf("Windows = %+v, want empty", snap.Windows)
	}
	if !strings.Contains(snap.Detail, "no credit pools") {
		t.Errorf("Detail = %q, want the no-credit-pools explanation", snap.Detail)
	}
}

// TestGrokBillingURL_Defaults: an empty or non-absolute observer falls back to
// the real Grok billing endpoint; an absolute URL is passed through untouched.
func TestGrokBillingURL_Defaults(t *testing.T) {
	if got := grokBillingURL(""); got != defaultGrokBillingURL {
		t.Errorf("grokBillingURL(empty) = %q, want %q", got, defaultGrokBillingURL)
	}
	if got := grokBillingURL("grok.com/grok_api_v2.GrokBuildBilling/GetGrokCreditsConfig"); got != defaultGrokBillingURL {
		t.Errorf("grokBillingURL(relative) = %q, want %q", got, defaultGrokBillingURL)
	}
	const custom = "http://127.0.0.1:9/credits"
	if got := grokBillingURL(custom); got != custom {
		t.Errorf("grokBillingURL(absolute) = %q, want %q", got, custom)
	}
}

// TestParseGrokCreditsResponse_MalformedInputNeverPanics: the parser walks
// untrusted bytes; random garbage and truncations must never panic.
func TestParseGrokCreditsResponse_MalformedInputNeverPanics(t *testing.T) {
	now := time.Date(2026, 9, 4, 22, 30, 0, 0, time.UTC)
	golden := mustHex(t, goldenGrokCreditsHex)

	seed := uint64(20260904)
	next := func() uint64 { seed = seed*6364136223846793005 + 1442695040888963407; return seed >> 33 }

	for i := 0; i < 500; i++ {
		buf := make([]byte, next()%200)
		for j := range buf {
			buf[j] = byte(next())
		}
		if _, err := ParseGrokCreditsResponse(buf, now); err != nil {
			t.Fatalf("ParseGrokCreditsResponse(random %x) error: %v", buf, err)
		}
	}
	for n := 0; n <= len(golden); n++ {
		if _, err := ParseGrokCreditsResponse(golden[:n], now); err != nil {
			t.Fatalf("ParseGrokCreditsResponse(truncated to %d) error: %v", n, err)
		}
	}
}

// ----------------------------------------------------------------------------
// Freebuff usage parsing
// ----------------------------------------------------------------------------

// modernFreebuffUsageJSON is the shape the current Codebuff usage endpoint
// (POST https://www.codebuff.com/api/v1/usage) answers with: a single credit
// balance plus the next quota reset, not the legacy day/week/month counters.
const modernFreebuffUsageJSON = `{
	"type": "usage-response",
	"usage": 12.5,
	"remainingBalance": 37.5,
	"balanceBreakdown": {"free": 5, "referral": 2.5, "subscription": 30, "purchase": 0},
	"next_quota_reset": "2026-10-03T12:42:08.035Z",
	"autoTopupEnabled": false
}`

// TestParseFreebuffUsageSnapshot_ModernFormat verifies the modern
// usage-response payload: one credits window carrying the remaining balance,
// the used/total percentage, the limit and the next_quota_reset countdown.
func TestParseFreebuffUsageSnapshot_ModernFormat(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 42, 8, 0, time.UTC)
	wantReset := time.Date(2026, 10, 3, 12, 42, 8, 35000000, time.UTC)

	snap, err := ParseFreebuffUsageSnapshot([]byte(modernFreebuffUsageJSON), now)
	if err != nil {
		t.Fatalf("ParseFreebuffUsageSnapshot: %v", err)
	}
	if !snap.ObservedAt.Equal(now) {
		t.Errorf("ObservedAt = %v, want %v", snap.ObservedAt, now)
	}
	if len(snap.Windows) != 1 {
		t.Fatalf("Windows = %+v, want exactly one credits window", snap.Windows)
	}
	win := snap.Windows[0]
	if win.Label != "Credits" && win.Label != "Monthly" {
		t.Errorf("Label = %q, want %q or %q", win.Label, "Credits", "Monthly")
	}
	if win.Unit != "credits" {
		t.Errorf("Unit = %q, want %q", win.Unit, "credits")
	}
	if win.Remaining != 37.5 {
		t.Errorf("Remaining = %v, want 37.5 (remainingBalance)", win.Remaining)
	}
	if win.Limit != 50 {
		t.Errorf("Limit = %v, want 50 (usage + remainingBalance)", win.Limit)
	}
	if win.UsedPct != 25 {
		t.Errorf("UsedPct = %v, want 25 (12.5 of 50)", win.UsedPct)
	}
	if !win.ResetAt.Equal(wantReset) {
		t.Errorf("ResetAt = %v, want %v (next_quota_reset)", win.ResetAt, wantReset)
	}
	// 2026-09-03 -> 2026-10-03 is exactly 30 days.
	if win.SecondsRemaining != 30*24*60*60 {
		t.Errorf("SecondsRemaining = %d, want %d", win.SecondsRemaining, 30*24*60*60)
	}
}

// TestParseFreebuffUsageSnapshot_ModernFormatZeroBalance: an exhausted
// balance must not divide by zero: the window stays honest at 0%.
func TestParseFreebuffUsageSnapshot_ModernFormatZeroBalance(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 42, 8, 0, time.UTC)
	body := `{"type":"usage-response","usage":0,"remainingBalance":0,"next_quota_reset":"2026-10-03T12:42:08.035Z"}`

	snap, err := ParseFreebuffUsageSnapshot([]byte(body), now)
	if err != nil {
		t.Fatalf("ParseFreebuffUsageSnapshot: %v", err)
	}
	if len(snap.Windows) != 1 {
		t.Fatalf("Windows = %+v, want one window", snap.Windows)
	}
	win := snap.Windows[0]
	if win.UsedPct != 0 || win.Limit != 0 || win.Remaining != 0 {
		t.Errorf("window = %+v, want all-zero usage and balance", win)
	}
	if win.Unit != "credits" {
		t.Errorf("Unit = %q, want credits", win.Unit)
	}
}

// TestParseFreebuffUsageSnapshot_LegacyFormatStillParses: the legacy
// day/week/month counters keep their three request windows when the modern
// fields are absent.
func TestParseFreebuffUsageSnapshot_LegacyFormatStillParses(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 42, 8, 0, time.UTC)
	body := `{"dayUsed":10,"dayLimit":100,"weekUsed":30,"weekLimit":300,"monthUsed":50,"monthLimit":500,"resetAt":"2026-09-04T00:00:00Z"}`

	snap, err := ParseFreebuffUsageSnapshot([]byte(body), now)
	if err != nil {
		t.Fatalf("ParseFreebuffUsageSnapshot: %v", err)
	}
	if len(snap.Windows) != 3 {
		t.Fatalf("Windows = %+v, want the legacy daily/weekly/monthly trio", snap.Windows)
	}
	want := []struct {
		label     string
		remaining float64
		limit     float64
	}{
		{"Daily", 90, 100},
		{"Weekly", 270, 300},
		{"Monthly", 450, 500},
	}
	for i, w := range want {
		win := snap.Windows[i]
		if win.Label != w.label || win.Remaining != w.remaining || win.Limit != w.limit || win.Unit != "requests" {
			t.Errorf("window %d = %+v, want %s remaining %v limit %v requests", i, win, w.label, w.remaining, w.limit)
		}
	}
	if snap.Windows[0].UsedPct != 10 {
		t.Errorf("Daily UsedPct = %v, want 10", snap.Windows[0].UsedPct)
	}
}

// modernUsageActor is a freebuff actor whose /usage endpoint answers with the
// modern Codebuff usage-response payload.
type modernUsageActor struct{}

func (a *modernUsageActor) Acquire(ctx context.Context) error { return nil }
func (a *modernUsageActor) Release()                          {}
func (a *modernUsageActor) InstanceID() string                { return "fb-inst-mid" }
func (a *modernUsageActor) FetchUsage(ctx context.Context, fingerprintID string) ([]byte, error) {
	return []byte(modernFreebuffUsageJSON), nil
}
func (a *modernUsageActor) SessionInfo(ctx context.Context) (string, string, error) {
	return "fb-inst-mid", "claude-opus-4.6", nil
}

// TestFreebuffQuota_ModernUsageResponse wires the modern payload through the
// actor-backed quota path the freebuff lane actually uses.
func TestFreebuffQuota_ModernUsageResponse(t *testing.T) {
	snap, err := freebuffQuota(context.Background(), &modernUsageActor{})
	if err != nil {
		t.Fatalf("freebuffQuota: %v", err)
	}
	if len(snap.Windows) != 1 {
		t.Fatalf("Windows = %+v, want one credits window", snap.Windows)
	}
	win := snap.Windows[0]
	if win.Unit != "credits" || win.Remaining != 37.5 || win.Limit != 50 || win.UsedPct != 25 {
		t.Errorf("window = %+v, want 37.5 of 50 credits (25%% used)", win)
	}
	if !win.ResetAt.Equal(time.Date(2026, 10, 3, 12, 42, 8, 35000000, time.UTC)) {
		t.Errorf("ResetAt = %v, want the next_quota_reset instant", win.ResetAt)
	}
	if snap.Detail != "instance: fb-inst-mid, model: claude-opus-4.6" {
		t.Errorf("Detail = %q, want the actor session detail", snap.Detail)
	}
}
