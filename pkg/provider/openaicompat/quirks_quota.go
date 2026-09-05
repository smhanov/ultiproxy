package openaicompat

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

// ----------------------------------------------------------------------------
// Grok Build credits quota (xAI) — gRPC-web
//
// The Grok web client fetches its credits config from
// grok_api_v2.GrokBuildBilling/GetGrokCreditsConfig as a gRPC-web call, not as
// a REST/JSON one: POST an empty protobuf request message (a 5-byte gRPC-web
// frame: flag byte 0 + big-endian uint32 length 0) with Content-Type
// application/grpc-web+proto, and read back a gRPC-web framed protobuf
// response. A JSON body (or none at all) is answered with HTTP 200 and zero
// bytes, which used to be misread as "no credit pools".
// ----------------------------------------------------------------------------

// defaultGrokBillingURL is the gRPC-web endpoint the Grok web client calls.
// The URL itself was always right; only the request wire format was wrong.
const defaultGrokBillingURL = "https://grok.com/grok_api_v2.GrokBuildBilling/GetGrokCreditsConfig"

// Request shape captured from the Grok web client (connect-es runtime).
const (
	grokGRPCWebContentType = "application/grpc-web+proto"
	grokGRPCWebFlagHeader  = "x-grpc-web"
	grokGRPCWebFlagValue   = "1"
	grokUserAgentHeader    = "x-user-agent"
	grokUserAgentValue     = "connect-es/2.1.1"
	grokOrigin             = "https://grok.com"
	grokReferer            = "https://grok.com/?_s=usage"

	// grokGRPCWebEmptyRequest is the gRPC-web frame carrying an empty protobuf
	// request message: data-frame flag byte 0x00 plus a zero big-endian uint32
	// payload length — exactly five zero bytes.
	grokGRPCWebEmptyRequest = "\x00\x00\x00\x00\x00"

	// grokNoCreditPoolsDetail is what a well-formed billing payload that
	// carries no usable window means: the account has no credits to report.
	grokNoCreditPoolsDetail = "Grok billing reports no credit pools: the account has no credits (free/no-credit plan) or its spending limit has been reached"
)

// setGrokGRPCWebHeaders applies the exact header set the Grok web client sends
// to the billing endpoint. Missing Origin/Referer gets the request silently
// downgraded to an empty response.
func setGrokGRPCWebHeaders(req *http.Request, token string) {
	req.Header.Set("Accept", "*/*")
	req.Header.Set("Content-Type", grokGRPCWebContentType)
	req.Header.Set("Origin", grokOrigin)
	req.Header.Set("Referer", grokReferer)
	req.Header.Set(grokGRPCWebFlagHeader, grokGRPCWebFlagValue)
	req.Header.Set(grokUserAgentHeader, grokUserAgentValue)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}

// grokBillingURL resolves the configured credits observer to a usable URL,
// falling back to the real Grok billing endpoint when it is empty or not an
// absolute http(s) URL.
func grokBillingURL(observer string) string {
	if observer == "" || (!strings.HasPrefix(observer, "http://") && !strings.HasPrefix(observer, "https://")) {
		return defaultGrokBillingURL
	}
	return observer
}

// fetchCreditsQuota queries the gRPC-web Grok credits endpoint and parses the
// response.
func fetchCreditsQuota(ctx context.Context, client *http.Client, billingURL, token string) (*provider.QuotaSnapshot, error) {
	billingURL = grokBillingURL(billingURL)
	if client == nil {
		client = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, billingURL,
		strings.NewReader(grokGRPCWebEmptyRequest))
	if err != nil {
		return nil, fmt.Errorf("openaicompat: create billing request: %w", err)
	}
	setGrokGRPCWebHeaders(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: billing request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: read billing response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return &provider.QuotaSnapshot{
			ObservedAt: time.Now().UTC(),
			Detail:     fmt.Sprintf("xAI billing HTTP %d (status unknown)", resp.StatusCode),
		}, nil
	}

	return ParseGrokCreditsResponse(body, time.Now().UTC())
}

// ----------------------------------------------------------------------------
// protobuf scanning (wire-format walk, no schema needed)
// ----------------------------------------------------------------------------

// scannedField is one scalar leaf found while walking a protobuf message. Only
// the shapes the quota heuristics need are recorded: varints and 32-bit floats,
// each with its full field path so "field 8 sub 1" can be recognised.
type scannedField struct {
	Path   []int   // field numbers from the message root, e.g. [1, 8, 1]
	Wire   int     // 0 = varint, 5 = fixed32
	Varint uint64  // valid when Wire == 0
	Float  float32 // valid when Wire == 5
}

// protobufScanner walks arbitrary protobuf bytes and collects every varint and
// fixed32 value together with its field path.
type protobufScanner struct {
	fields []scannedField
}

// readVarint decodes one base-128 varint starting at idx.
func readVarint(data []byte, idx int) (uint64, int, bool) {
	var val uint64
	var shift uint
	for idx < len(data) && shift < 64 {
		b := data[idx]
		idx++
		val |= uint64(b&0x7f) << shift
		if b < 0x80 {
			return val, idx, true
		}
		shift += 7
	}
	return 0, idx, false
}

const maxProtoDepth = 10

// scan recursively walks data, appending every varint/fixed32 leaf it finds.
// Malformed input never panics: an unreadable field just stops the walk at that
// offset and scanning resumes one byte later.
func (s *protobufScanner) scan(data []byte, path []int, depth int) {
	if depth > maxProtoDepth {
		return
	}
	i := 0
	for i < len(data) {
		start := i
		key, nextIdx, ok := readVarint(data, i)
		if !ok || key == 0 {
			i = start + 1
			continue
		}
		fieldNum := int(key >> 3)
		if fieldNum <= 0 {
			i = start + 1
			continue
		}
		wireType := int(key & 7)
		currPath := append(append([]int(nil), path...), fieldNum)
		i = nextIdx

		switch wireType {
		case 0: // varint
			val, endIdx, ok := readVarint(data, i)
			if !ok {
				i = start + 1
				continue
			}
			s.fields = append(s.fields, scannedField{Path: currPath, Wire: 0, Varint: val})
			i = endIdx

		case 1: // 64-bit — not needed by any heuristic, skip the payload
			if i+8 > len(data) {
				i = start + 1
				continue
			}
			i += 8

		case 2: // length-delimited: recurse (nested messages carry the windows)
			size, endIdx, ok := readVarint(data, i)
			if !ok || size > uint64(len(data)-endIdx) {
				i = start + 1
				continue
			}
			s.scan(data[endIdx:endIdx+int(size)], currPath, depth+1)
			i = endIdx + int(size)

		case 5: // fixed32
			if i+4 > len(data) {
				i = start + 1
				continue
			}
			s.fields = append(s.fields, scannedField{
				Path:  currPath,
				Wire:  5,
				Float: math.Float32frombits(binary.LittleEndian.Uint32(data[i : i+4])),
			})
			i += 4

		default: // groups (3/4) and anything unknown: skip a byte
			i = start + 1
		}
	}
}

// pathEndsWith reports whether path ends with the given field-number suffix.
func pathEndsWith(path []int, suffix ...int) bool {
	if len(path) < len(suffix) {
		return false
	}
	return equalInts(path[len(path)-len(suffix):], suffix)
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ----------------------------------------------------------------------------
// gRPC-web framing
// ----------------------------------------------------------------------------

// UnframeGRPCWeb extracts every data-frame payload from a gRPC-web stream. A
// frame is a flag byte followed by a big-endian uint32 length and that many
// payload bytes; frames with the 0x80 trailer bit set carry gRPC metadata
// ("grpc-status:0\r\n") and are skipped, never handed to the protobuf walker.
func UnframeGRPCWeb(data []byte) ([][]byte, error) {
	var payloads [][]byte
	i := 0
	for i+5 <= len(data) {
		flags := data[i]
		size := binary.BigEndian.Uint32(data[i+1 : i+5])
		start := i + 5
		end := start + int(size)
		if end > len(data) {
			// truncated frame — keep whatever data frames we already have
			break
		}
		if flags&0x80 == 0 {
			payloads = append(payloads, data[start:end])
		}
		i = end
	}
	return payloads, nil
}

// ----------------------------------------------------------------------------
// Grok credits protobuf — window extraction
// ----------------------------------------------------------------------------

// Verified shape of GetGrokCreditsConfig (from a live response):
//
//	field 4  message {1: varint seconds} — window start (unix seconds, UTC)
//	field 5  message {1: varint seconds} — window end / reset
//	field 8  message {1: varint state}   — 2 == active
//	          nested {2: {1: start_secs}}, {3: {1: end_secs}}
//	field 11, field 13 varint             — flags (1 == true)
//	optional: a usage percent as a fixed32 float 0..100 nested under a field
//	          whose last path element is 1 (the ai-quota-dashboard heuristic).
const (
	// Plausible unix-seconds range for a billing window timestamp: 2023-11
	// through 2036. Anything outside is a flag, a counter or a state enum.
	grokUnixSecondsMin int64 = 1700000000
	grokUnixSecondsMax int64 = 2100000000

	// Field 8 sub-field 1 value that marks a window as active.
	grokWindowStateActive uint64 = 2

	// The usage percent is carried by a field whose last path element is 1.
	grokUsagePercentField = 1

	grokCreditsLabel      = "Grok Build"
	grokCreditsDateLayout = "Jan 2 2006"
	grokCreditsDayLayout  = "Jan 2"
)

// grokCreditsWindow is what the billing protobuf boils down to for the quota
// dashboard: one rate-limit window plus how much of it has been consumed.
type grokCreditsWindow struct {
	Start  time.Time // window start; zero when unknown
	Reset  time.Time // window end / reset; zero when unknown
	Active bool      // field 8 reported state == 2 (active)
	Pct    float64   // usage percent 0..100
	HasPct bool      // a usage percent was actually present in the payload
}

// parseGrokCredits walks one GetGrokCreditsConfig protobuf message and distils
// it into a grokCreditsWindow:
//
//  1. starts  = varints in the unix-seconds range that are <= now
//     resets  = varints in the unix-seconds range that are >  now
//     window start = earliest plausible start, reset = earliest plausible reset
//  2. pct = smallest fixed32 in [0,100] whose field path ends in field 1
//     (values <= 1 are fractions and are scaled to percent)
//  3. active = any field-8 sub-message reporting state 2
func parseGrokCredits(payload []byte, now time.Time) grokCreditsWindow {
	scanner := &protobufScanner{}
	scanner.scan(payload, nil, 0)

	var starts, resets []time.Time
	for _, f := range scanner.fields {
		// Only varints can be timestamps here; the fixed32 usage floats are
		// handled by grokUsagePercent below.
		if f.Wire != 0 {
			continue
		}
		if f.Varint >= uint64(grokUnixSecondsMin) && f.Varint <= uint64(grokUnixSecondsMax) {
			ts := time.Unix(int64(f.Varint), 0).UTC()
			if ts.After(now) {
				resets = append(resets, ts)
			} else {
				starts = append(starts, ts)
			}
		}
	}

	w := grokCreditsWindow{
		Start: earliestTime(starts),
		Reset: earliestTime(resets),
	}
	w.Active = grokWindowActive(scanner.fields)
	w.Pct, w.HasPct = grokUsagePercent(scanner.fields)
	return w
}

// grokWindowActive reports whether any field-8 sub-message reports the active
// state (2).
func grokWindowActive(fields []scannedField) bool {
	for _, f := range fields {
		if f.Wire == 0 && pathEndsWith(f.Path, 8, 1) && f.Varint == grokWindowStateActive {
			return true
		}
	}
	return false
}

// grokUsagePercent implements the ai-quota-dashboard heuristic: candidates are
// fixed32 values in [0,100] whose field path ends with field 1, and the usage
// percent is the smallest of them.
func grokUsagePercent(fields []scannedField) (float64, bool) {
	var (
		best  float64
		found bool
	)
	for _, f := range fields {
		if f.Wire != 5 || !pathEndsWith(f.Path, grokUsagePercentField) {
			continue
		}
		if math.IsNaN(float64(f.Float)) || math.IsInf(float64(f.Float), 0) {
			continue
		}
		pct := grokPercentFromFloat(f.Float)
		if pct < 0 || pct > 100 {
			continue
		}
		if !found || pct < best {
			best, found = pct, true
		}
	}
	return best, found
}

// grokPercentFromFloat normalises a raw usage float: values <= 1 are fractions
// (0.68 -> 68%), anything larger is already a percent. The result is rounded to
// two decimals so float32 representation noise never leaks into a report.
func grokPercentFromFloat(v float32) float64 {
	pct := float64(v)
	if v <= 1 {
		pct *= 100
	}
	return math.Round(pct*100) / 100
}

func earliestTime(times []time.Time) time.Time {
	var earliest time.Time
	for _, t := range times {
		if earliest.IsZero() || t.Before(earliest) {
			earliest = t
		}
	}
	return earliest
}

// ParseGrokCreditsResponse parses a gRPC-web Grok credits response into a
// QuotaSnapshot with a single "Grok Build" window.
func ParseGrokCreditsResponse(data []byte, now time.Time) (*provider.QuotaSnapshot, error) {
	payloads, err := UnframeGRPCWeb(data)
	if err != nil {
		return &provider.QuotaSnapshot{
			ObservedAt: now,
			Detail:     "Failed to unframe gRPC-web response (status unknown)",
		}, nil
	}

	// Concatenate the data frames: a message may be split across frames.
	var message []byte
	for _, p := range payloads {
		message = append(message, p...)
	}

	window := parseGrokCredits(message, now)

	// No reset, no usage percent and no active window: a well-formed billing
	// payload that reports nothing usable. That means the account simply has
	// no credits to report (free / no-credit plan, or a spending limit already
	// reached) — a valid observation, not an error.
	if window.Reset.IsZero() && !window.HasPct && !window.Active {
		return &provider.QuotaSnapshot{
			ObservedAt: now,
			Windows:    []provider.QuotaWindow{},
			Detail:     grokNoCreditPoolsDetail,
		}, nil
	}

	pct := window.Pct
	remaining := math.Max(0, 100-pct)
	var secondsRemaining int64
	if !window.Reset.IsZero() {
		secondsRemaining = int64(window.Reset.Sub(now).Seconds())
		if secondsRemaining < 0 {
			secondsRemaining = 0
		}
	}

	w := provider.QuotaWindow{
		Label:            grokCreditsLabel,
		UsedPct:          pct,
		Remaining:        remaining,
		Limit:            100,
		Unit:             "%",
		ResetAt:          window.Reset,
		SecondsRemaining: secondsRemaining,
	}

	return &provider.QuotaSnapshot{
		ObservedAt: now,
		Windows:    []provider.QuotaWindow{w},
		Detail:     grokCreditsDetail(window),
	}, nil
}

// grokCreditsDetail renders the human-facing one-liner for a Grok Build window.
func grokCreditsDetail(w grokCreditsWindow) string {
	if w.HasPct {
		if w.Reset.IsZero() {
			return fmt.Sprintf("Grok Build %v%% used", w.Pct)
		}
		return fmt.Sprintf("Grok Build %v%% used · resets %s", w.Pct, w.Reset.Format(grokCreditsDateLayout))
	}
	if w.Reset.IsZero() {
		return "Grok Build window active, no usage recorded yet"
	}
	if w.Start.IsZero() {
		return fmt.Sprintf("Grok Build window active, no usage recorded yet (resets %s)",
			w.Reset.Format(grokCreditsDateLayout))
	}
	return fmt.Sprintf("Grok Build window active (%s - %s), no usage recorded yet",
		w.Start.Format(grokCreditsDayLayout), w.Reset.Format(grokCreditsDateLayout))
}

// ----------------------------------------------------------------------------
// Freebuff quota (ported verbatim from freebuff — that package is
// deleted in Phase F2b; the account actor stays in pkg/spikes/freebuff).
// ----------------------------------------------------------------------------

type freebuffUsagePayload struct {
	DayUsed    float64 `json:"dayUsed"`
	DayLimit   float64 `json:"dayLimit"`
	WeekUsed   float64 `json:"weekUsed"`
	WeekLimit  float64 `json:"weekLimit"`
	MonthUsed  float64 `json:"monthUsed"`
	MonthLimit float64 `json:"monthLimit"`
	ResetAt    string  `json:"resetAt"`

	// Modern Codebuff usage response (POST https://www.codebuff.com/api/v1/usage):
	// a single credit balance plus the next quota reset, not the legacy
	// day/week/month request counters.
	Type             string             `json:"type"`
	Usage            float64            `json:"usage"`
	RemainingBalance float64            `json:"remainingBalance"`
	NextQuotaReset   string             `json:"next_quota_reset"`
	BalanceBreakdown map[string]float64 `json:"balanceBreakdown"`

	// snake_case fallback
	DayUsedSnake    float64 `json:"day_used"`
	DayLimitSnake   float64 `json:"day_limit"`
	WeekUsedSnake   float64 `json:"week_used"`
	WeekLimitSnake  float64 `json:"week_limit"`
	MonthUsedSnake  float64 `json:"month_used"`
	MonthLimitSnake float64 `json:"month_limit"`
	ResetAtSnake    string  `json:"reset_at"`
}

// parseReset parses an RFC3339 reset timestamp into the instant plus the
// seconds remaining until then (clamped at zero for past resets).
func parseReset(resetStr string, now time.Time) (time.Time, int64) {
	if resetStr == "" {
		return time.Time{}, 0
	}
	t, err := time.Parse(time.RFC3339, resetStr)
	if err != nil {
		return time.Time{}, 0
	}
	secRem := int64(t.Sub(now).Seconds())
	if secRem < 0 {
		secRem = 0
	}
	return t, secRem
}

// ParseFreebuffUsageSnapshot parses freebuff usage response bytes into a
// normalized QuotaSnapshot (ported verbatim from freebuff, extended for the
// modern Codebuff usage-response payload).
func ParseFreebuffUsageSnapshot(data []byte, now time.Time) (*provider.QuotaSnapshot, error) {
	var u freebuffUsagePayload
	if err := json.Unmarshal(data, &u); err != nil {
		return nil, fmt.Errorf("failed to parse freebuff usage json: %w", err)
	}

	calcPct := func(used, limit float64) float64 {
		if limit <= 0 {
			return 0
		}
		return (used / limit) * 100.0
	}

	// Modern Codebuff usage-response: one credit-balance window that resets at
	// next_quota_reset. The legacy day/week/month counters are absent there,
	// and vice versa, so the two shapes never mix.
	if u.Type == "usage-response" || u.NextQuotaReset != "" {
		resetAt, secRem := parseReset(u.NextQuotaReset, now)
		limit := u.Usage + u.RemainingBalance
		return &provider.QuotaSnapshot{
			ObservedAt: now,
			Windows: []provider.QuotaWindow{{
				Label:            "Credits",
				Remaining:        u.RemainingBalance,
				UsedPct:          calcPct(u.Usage, limit),
				Limit:            limit,
				Unit:             "credits",
				ResetAt:          resetAt,
				SecondsRemaining: secRem,
			}},
		}, nil
	}

	dayUsed := u.DayUsed
	if dayUsed == 0 && u.DayUsedSnake != 0 {
		dayUsed = u.DayUsedSnake
	}
	dayLimit := u.DayLimit
	if dayLimit == 0 && u.DayLimitSnake != 0 {
		dayLimit = u.DayLimitSnake
	}

	weekUsed := u.WeekUsed
	if weekUsed == 0 && u.WeekUsedSnake != 0 {
		weekUsed = u.WeekUsedSnake
	}
	weekLimit := u.WeekLimit
	if weekLimit == 0 && u.WeekLimitSnake != 0 {
		weekLimit = u.WeekLimitSnake
	}

	monthUsed := u.MonthUsed
	if monthUsed == 0 && u.MonthUsedSnake != 0 {
		monthUsed = u.MonthUsedSnake
	}
	monthLimit := u.MonthLimit
	if monthLimit == 0 && u.MonthLimitSnake != 0 {
		monthLimit = u.MonthLimitSnake
	}

	resetStr := u.ResetAt
	if resetStr == "" {
		resetStr = u.ResetAtSnake
	}
	resetAt, secRem := parseReset(resetStr, now)

	windows := []provider.QuotaWindow{
		{
			Label:            "Daily",
			UsedPct:          calcPct(dayUsed, dayLimit),
			Remaining:        dayLimit - dayUsed,
			Limit:            dayLimit,
			Unit:             "requests",
			ResetAt:          resetAt,
			SecondsRemaining: secRem,
		},
		{
			Label:     "Weekly",
			UsedPct:   calcPct(weekUsed, weekLimit),
			Remaining: weekLimit - weekUsed,
			Limit:     weekLimit,
			Unit:      "requests",
		},
		{
			Label:     "Monthly",
			UsedPct:   calcPct(monthUsed, monthLimit),
			Remaining: monthLimit - monthUsed,
			Limit:     monthLimit,
			Unit:      "requests",
		},
	}

	return &provider.QuotaSnapshot{
		ObservedAt: now,
		Windows:    windows,
	}, nil
}

// freebuffQuota queries usage via the injected actor and returns a normalized
// QuotaSnapshot (ported from freebuff.Provider.Quota).
func freebuffQuota(ctx context.Context, actor any) (*provider.QuotaSnapshot, error) {
	src, ok := actor.(freebuffQuotaSource)
	if !ok {
		return nil, fmt.Errorf("openaicompat: freebuff actor does not implement quota source (got %T)", actor)
	}

	rawUsage, err := src.FetchUsage(ctx, "cli-usage")
	if err != nil {
		return nil, fmt.Errorf("failed to fetch usage from actor: %w", err)
	}

	snapshot, err := ParseFreebuffUsageSnapshot(rawUsage, time.Now())
	if err != nil {
		return nil, err
	}

	instanceID, model, _ := src.SessionInfo(ctx)
	snapshot.Detail = fmt.Sprintf("instance: %s, model: %s", instanceID, model)
	return snapshot, nil
}
