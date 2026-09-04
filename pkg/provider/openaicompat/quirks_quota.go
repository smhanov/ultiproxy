package openaicompat

import (
	"bytes"
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

const defaultGrokBillingURL = "https://grok.com/grok_api_v2.GrokBuildBilling/GetGrokCreditsConfig"

// fetchCreditsQuota queries the gRPC-web Grok credits endpoint and parses the response.
func fetchCreditsQuota(ctx context.Context, client *http.Client, billingURL, token string) (*provider.QuotaSnapshot, error) {
	if billingURL == "" || (!strings.HasPrefix(billingURL, "http://") && !strings.HasPrefix(billingURL, "https://")) {
		billingURL = defaultGrokBillingURL
	}
	if client == nil {
		client = http.DefaultClient
	}

	emptyFrame := []byte{0x00, 0x00, 0x00, 0x00, 0x00}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, billingURL, bytes.NewReader(emptyFrame))
	if err != nil {
		return nil, fmt.Errorf("openaicompat: create billing request: %w", err)
	}

	req.Header.Set("Content-Type", "application/grpc-web+proto")
	req.Header.Set("X-Grpc-Web", "1")
	req.Header.Set("X-User-Agent", "grpc-web-javascript/0.1")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

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

type scannedField struct {
	Path   []int
	Wire   int
	Varint uint64
	Float  float32
	Double float64
	String string
	Bytes  []byte
}

type protobufScanner struct {
	fields []scannedField
}

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

func (s *protobufScanner) scan(data []byte, path []int, depth int) {
	if depth > 10 {
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
		wireType := int(key & 7)
		currPath := append(append([]int(nil), path...), fieldNum)
		i = nextIdx

		switch wireType {
		case 0: // Varint
			val, endIdx, ok := readVarint(data, i)
			if !ok {
				i = start + 1
				continue
			}
			s.fields = append(s.fields, scannedField{
				Path:   currPath,
				Wire:   0,
				Varint: val,
			})
			i = endIdx

		case 1: // 64-bit
			if i+8 > len(data) {
				i = start + 1
				continue
			}
			bits := binary.LittleEndian.Uint64(data[i : i+8])
			valFloat := math.Float64frombits(bits)
			s.fields = append(s.fields, scannedField{
				Path:   currPath,
				Wire:   1,
				Double: valFloat,
			})
			i += 8

		case 2: // Length-delimited
			size, endIdx, ok := readVarint(data, i)
			if !ok || int(size) > len(data)-endIdx {
				i = start + 1
				continue
			}
			subBytes := data[endIdx : endIdx+int(size)]
			s.fields = append(s.fields, scannedField{
				Path:   currPath,
				Wire:   2,
				String: string(subBytes),
				Bytes:  subBytes,
			})
			s.scan(subBytes, currPath, depth+1)
			i = endIdx + int(size)

		case 5: // 32-bit
			if i+4 > len(data) {
				i = start + 1
				continue
			}
			bits := binary.LittleEndian.Uint32(data[i : i+4])
			valFloat := math.Float32frombits(bits)
			s.fields = append(s.fields, scannedField{
				Path:  currPath,
				Wire:  5,
				Float: valFloat,
			})
			i += 4

		default:
			i = start + 1
		}
	}
}

// UnframeGRPCWeb extracts all data payloads from a gRPC-web stream.
func UnframeGRPCWeb(data []byte) ([][]byte, error) {
	var payloads [][]byte
	i := 0
	for i+5 <= len(data) {
		flags := data[i]
		size := binary.BigEndian.Uint32(data[i+1 : i+5])
		start := i + 5
		end := start + int(size)
		if end > len(data) {
			break
		}
		if (flags & 0x80) == 0 {
			payloads = append(payloads, data[start:end])
		}
		i = end
	}
	if len(payloads) == 0 && len(data) > 0 {
		payloads = append(payloads, data)
	}
	return payloads, nil
}

// ParseGrokCreditsResponse parses gRPC-web binary data into QuotaSnapshot.
func ParseGrokCreditsResponse(data []byte, now time.Time) (*provider.QuotaSnapshot, error) {
	payloads, err := UnframeGRPCWeb(data)
	if err != nil {
		return &provider.QuotaSnapshot{
			ObservedAt: now,
			Detail:     "Failed to unframe gRPC-web response (status unknown)",
		}, nil
	}

	scanner := &protobufScanner{}
	for _, p := range payloads {
		scanner.scan(p, nil, 0)
	}

	type poolData struct {
		Name    string
		UsedPct *float64
		ResetAt time.Time
	}

	pools := make(map[string]*poolData)
	getOrCreatePool := func(name string) *poolData {
		if p, ok := pools[name]; ok {
			return p
		}
		p := &poolData{Name: name}
		pools[name] = p
		return p
	}

	for _, f := range scanner.fields {
		if f.Wire == 2 && (strings.Contains(strings.ToLower(f.String), "5 hour") || strings.Contains(strings.ToLower(f.String), "weekly")) {
			label := "5 hour"
			if strings.Contains(strings.ToLower(f.String), "weekly") {
				label = "Weekly"
			}
			getOrCreatePool(label)
		}
	}

	for _, f := range scanner.fields {
		for name, p := range pools {
			isMatch := false
			if strings.EqualFold(name, "5 hour") && len(f.Path) > 0 && f.Path[0] == 1 {
				isMatch = true
			} else if strings.EqualFold(name, "Weekly") && len(f.Path) > 0 && f.Path[0] == 2 {
				isMatch = true
			}
			if isMatch {
				if f.Wire == 5 && f.Float >= 0 && f.Float <= 100 && p.UsedPct == nil {
					pct := float64(f.Float)
					p.UsedPct = &pct
				}
				if f.Wire == 0 && f.Varint >= 1700000000 && f.Varint <= 2500000000 {
					p.ResetAt = time.Unix(int64(f.Varint), 0).UTC()
				}
			}
		}
	}

	var windows []provider.QuotaWindow
	order := []string{"5 hour", "Weekly"}
	for _, name := range order {
		if p, ok := pools[name]; ok && p.UsedPct != nil {
			var secRem int64
			if !p.ResetAt.IsZero() && p.ResetAt.After(now) {
				secRem = int64(p.ResetAt.Sub(now).Seconds())
			}
			used := *p.UsedPct
			rem := math.Max(0, 100.0-used)
			windows = append(windows, provider.QuotaWindow{
				Label:            name,
				UsedPct:          used,
				Remaining:        rem,
				Limit:            100,
				Unit:             "%",
				ResetAt:          p.ResetAt,
				SecondsRemaining: secRem,
			})
		}
	}

	if len(windows) == 0 {
		var candidates []float64
		for _, f := range scanner.fields {
			if f.Wire == 5 && f.Float >= 0 && f.Float <= 100 {
				candidates = append(candidates, float64(f.Float))
			}
		}

		var resets []time.Time
		for _, f := range scanner.fields {
			if f.Wire == 0 && f.Varint >= 1700000000 && f.Varint <= 2500000000 {
				resets = append(resets, time.Unix(int64(f.Varint), 0).UTC())
			}
		}

		if len(candidates) > 0 {
			usedPct := candidates[0]
			var reset time.Time
			if len(resets) > 0 {
				reset = resets[0]
			}
			var secRem int64
			if !reset.IsZero() && reset.After(now) {
				secRem = int64(reset.Sub(now).Seconds())
			}

			windows = append(windows, provider.QuotaWindow{
				Label:            "Grok Build credits",
				UsedPct:          usedPct,
				Remaining:        math.Max(0, 100.0-usedPct),
				Limit:            100,
				Unit:             "%",
				ResetAt:          reset,
				SecondsRemaining: secRem,
			})
		}
	}

	if len(windows) == 0 {
		return &provider.QuotaSnapshot{
			ObservedAt: now,
			Detail:     "Grok billing response contained no recognizable credit pools (status unknown)",
		}, nil
	}

	var detailParts []string
	for _, w := range windows {
		detailParts = append(detailParts, fmt.Sprintf("%s: %.1f%% used", w.Label, w.UsedPct))
	}

	return &provider.QuotaSnapshot{
		ObservedAt: now,
		Windows:    windows,
		Detail:     strings.Join(detailParts, " · "),
	}, nil
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

	// snake_case fallback
	DayUsedSnake    float64 `json:"day_used"`
	DayLimitSnake   float64 `json:"day_limit"`
	WeekUsedSnake   float64 `json:"week_used"`
	WeekLimitSnake  float64 `json:"week_limit"`
	MonthUsedSnake  float64 `json:"month_used"`
	MonthLimitSnake float64 `json:"month_limit"`
	ResetAtSnake    string  `json:"reset_at"`
}

// ParseFreebuffUsageSnapshot parses freebuff usage response bytes into a
// normalized QuotaSnapshot (ported verbatim from freebuff).
func ParseFreebuffUsageSnapshot(data []byte, now time.Time) (*provider.QuotaSnapshot, error) {
	var u freebuffUsagePayload
	if err := json.Unmarshal(data, &u); err != nil {
		return nil, fmt.Errorf("failed to parse freebuff usage json: %w", err)
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
	var resetAt time.Time
	var secRem int64
	if resetStr != "" {
		if t, err := time.Parse(time.RFC3339, resetStr); err == nil {
			resetAt = t
			secRem = int64(t.Sub(now).Seconds())
			if secRem < 0 {
				secRem = 0
			}
		}
	}

	calcPct := func(used, limit float64) float64 {
		if limit <= 0 {
			return 0
		}
		return (used / limit) * 100.0
	}

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
