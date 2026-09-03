package quota

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// LimitErrorKind identifies the category of rate or quota limitation.
type LimitErrorKind string

const (
	LimitKindBurstRate       LimitErrorKind = "burst_rate"
	LimitKindTokenRate       LimitErrorKind = "token_rate"
	LimitKindQuotaWindow     LimitErrorKind = "quota_window"
	LimitKindSpendLimit      LimitErrorKind = "spend_limit"
	LimitKindAccountLimit    LimitErrorKind = "account_limit"
	LimitKindCapacity        LimitErrorKind = "capacity"
	LimitKindSessionConflict LimitErrorKind = "session_conflict"
	LimitKindWaitingRoom     LimitErrorKind = "waiting_room"
	LimitKindAbuseGuard      LimitErrorKind = "abuse_guard"
	LimitKindUnknown         LimitErrorKind = "unknown"
)

// LimitError represents a classified rate limit or quota exhaustion event.
type LimitError struct {
	Scope      string         `json:"scope,omitempty"`
	Kind       LimitErrorKind `json:"kind"`
	RetryAt    *time.Time     `json:"retry_at,omitempty"`
	RetryAfter time.Duration  `json:"retry_after,omitempty"`
	Permanent  bool           `json:"permanent"`
	Provider   string         `json:"provider,omitempty"`
	Model      string         `json:"model,omitempty"`
	Credential string         `json:"credential,omitempty"`
	Cause      error          `json:"cause,omitempty"`
}

func (e LimitError) Error() string {
	var parts []string
	parts = append(parts, fmt.Sprintf("limit error [%s]", e.Kind))
	if e.Scope != "" {
		parts = append(parts, fmt.Sprintf("scope=%s", e.Scope))
	}
	if e.Provider != "" {
		parts = append(parts, fmt.Sprintf("provider=%s", e.Provider))
	}
	if e.Model != "" {
		parts = append(parts, fmt.Sprintf("model=%s", e.Model))
	}
	if e.Permanent {
		parts = append(parts, "permanent")
	} else if e.RetryAfter > 0 {
		parts = append(parts, fmt.Sprintf("retry_after=%s", e.RetryAfter))
	}
	if e.Cause != nil {
		parts = append(parts, fmt.Sprintf("cause=%v", e.Cause))
	}
	return strings.Join(parts, " ")
}

// Classify parses an HTTP status, headers, and body error JSON to produce a typed LimitError.
func Classify(httpStatus int, headers http.Header, body []byte, providerHint string) LimitError {
	now := time.Now().UTC()
	err := LimitError{
		Provider: providerHint,
		Kind:     LimitKindUnknown,
	}

	// 1. Parse retry timing from headers
	var retryAfterDur time.Duration
	var retryAtTime *time.Time

	if headers != nil {
		// Retry-After can be seconds or HTTP Date
		if ra := getHeader(headers, "Retry-After"); ra != "" {
			if secs, parseErr := strconv.ParseFloat(ra, 64); parseErr == nil && secs >= 0 {
				retryAfterDur = time.Duration(secs * float64(time.Second))
			} else if t, dateErr := http.ParseTime(ra); dateErr == nil {
				retryAtTime = &t
				diff := t.Sub(now)
				if diff > 0 {
					retryAfterDur = diff
				}
			}
		}

		// retry-after-ms
		if retryAfterDur == 0 {
			if rams := getHeader(headers, "retry-after-ms"); rams != "" {
				if ms, parseErr := strconv.ParseInt(rams, 10, 64); parseErr == nil && ms >= 0 {
					retryAfterDur = time.Duration(ms) * time.Millisecond
				}
			}
		}

		// x-ratelimit-reset / ratelimit-reset
		if retryAfterDur == 0 {
			for _, hdr := range []string{"x-ratelimit-reset", "ratelimit-reset", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"} {
				if val := getHeader(headers, hdr); val != "" {
					if resetVal, parseErr := strconv.ParseInt(val, 10, 64); parseErr == nil {
						if resetVal > 1_000_000_000 {
							// Epoch seconds
							t := time.Unix(resetVal, 0).UTC()
							retryAtTime = &t
							diff := t.Sub(now)
							if diff > 0 {
								retryAfterDur = diff
							}
							break
						} else if resetVal > 0 {
							// Seconds duration
							retryAfterDur = time.Duration(resetVal) * time.Second
							break
						}
					}
				}
			}
		}
	}

	if retryAfterDur > 0 && retryAtTime == nil {
		t := now.Add(retryAfterDur)
		retryAtTime = &t
	}
	err.RetryAfter = retryAfterDur
	err.RetryAt = retryAtTime

	// 2. Parse body error text and JSON
	var parsedBody map[string]any
	bodyStr := strings.TrimSpace(string(body))
	if len(bodyStr) > 0 {
		_ = json.Unmarshal(body, &parsedBody)
	}

	// Extract message, code, type, param
	var rawMsg, errType, errCode string
	if parsedBody != nil {
		if innerErr, ok := parsedBody["error"].(map[string]any); ok {
			if m, ok := innerErr["message"].(string); ok {
				rawMsg = m
			}
			if t, ok := innerErr["type"].(string); ok {
				errType = t
			}
			if c, ok := innerErr["code"].(string); ok {
				errCode = c
			} else if numCode, ok := innerErr["code"].(float64); ok {
				errCode = fmt.Sprintf("%.0f", numCode)
			}
			if model, ok := innerErr["model"].(string); ok && model != "" {
				err.Model = model
			}
		} else if m, ok := parsedBody["message"].(string); ok {
			rawMsg = m
		}
		if code, ok := parsedBody["code"].(string); ok && errCode == "" {
			errCode = code
		}
		if detail, ok := parsedBody["detail"].(string); ok && rawMsg == "" {
			rawMsg = detail
		}
	}

	text := strings.ToLower(fmt.Sprintf("%s %s %s %s", rawMsg, errType, errCode, bodyStr))

	// 3. Classify according to status and body patterns
	switch {
	// Spend limit / billing / credits
	case httpStatus == http.StatusPaymentRequired ||
		strings.Contains(text, "insufficient_quota") ||
		strings.Contains(text, "spend limit") ||
		strings.Contains(text, "credit balance is too low") ||
		strings.Contains(text, "exceeded your current quota") ||
		strings.Contains(text, "billing details") ||
		strings.Contains(text, "purchase credits") ||
		strings.Contains(text, "payment required"):
		err.Kind = LimitKindSpendLimit
		err.Scope = "account"
		err.Permanent = true

	// Abuse guard / policy / banned
	case httpStatus == http.StatusForbidden && (strings.Contains(text, "abuse") || strings.Contains(text, "policy") || strings.Contains(text, "suspended")) ||
		strings.Contains(text, "abuse guard") ||
		strings.Contains(text, "account suspended") ||
		strings.Contains(text, "policy violation") ||
		strings.Contains(text, "flagged for abuse"):
		err.Kind = LimitKindAbuseGuard
		err.Scope = "account"
		err.Permanent = true

	// Session conflict (e.g. Freebuff single-session, concurrent session)
	case strings.Contains(text, "session conflict") ||
		strings.Contains(text, "session in use") ||
		strings.Contains(text, "concurrent session") ||
		strings.Contains(text, "another request is in progress") ||
		strings.Contains(text, "session active"):
		err.Kind = LimitKindSessionConflict
		err.Scope = "session"

	// Waiting room / queued
	case strings.Contains(text, "waiting room") ||
		strings.Contains(text, "request queued") ||
		strings.Contains(text, "queue full") ||
		strings.Contains(text, "in queue"):
		err.Kind = LimitKindWaitingRoom
		err.Scope = "queue"

	// Capacity / Overload (e.g. status 529, 503 capacity)
	case httpStatus == 529 ||
		strings.Contains(text, "overloaded") ||
		strings.Contains(text, "overloaded_error") ||
		strings.Contains(text, "capacity") ||
		strings.Contains(text, "server is temporarily overloaded") ||
		strings.Contains(text, "high demand") ||
		strings.Contains(text, "too busy"):
		err.Kind = LimitKindCapacity
		err.Scope = "server"

	// Token rate (TPM)
	case strings.Contains(text, "token_rate") ||
		strings.Contains(text, "tokens per minute") ||
		strings.Contains(text, "tpm") ||
		strings.Contains(text, "rate limit reached for tokens") ||
		errType == "tokens":
		err.Kind = LimitKindTokenRate
		err.Scope = "tokens"

	// Quota window (RPD, daily, weekly, monthly, 5-hour)
	case strings.Contains(text, "quota_window") ||
		strings.Contains(text, "requests per day") ||
		strings.Contains(text, "rpd") ||
		strings.Contains(text, "daily limit") ||
		strings.Contains(text, "weekly limit") ||
		strings.Contains(text, "monthly limit") ||
		strings.Contains(text, "5-hour") ||
		strings.Contains(text, "plan limit"):
		err.Kind = LimitKindQuotaWindow
		err.Scope = "window"

	// Burst rate (RPM, concurrent requests, generic request rate limit)
	case strings.Contains(text, "burst_rate") ||
		strings.Contains(text, "requests per minute") ||
		strings.Contains(text, "rpm") ||
		strings.Contains(text, "rate limit reached for requests") ||
		strings.Contains(text, "rate_limit_exceeded") ||
		strings.Contains(text, "rate_limit_error") ||
		errType == "requests" ||
		strings.Contains(text, "too many requests"):
		err.Kind = LimitKindBurstRate
		err.Scope = "requests"

	// Account limit
	case strings.Contains(text, "account limit") ||
		strings.Contains(text, "org limit") ||
		strings.Contains(text, "organization limit"):
		err.Kind = LimitKindAccountLimit
		err.Scope = "account"

	// Fallback
	default:
		err.Kind = LimitKindUnknown
		err.Scope = "unknown"
	}

	if rawMsg != "" {
		err.Cause = fmt.Errorf("%s", rawMsg)
	} else if len(bodyStr) > 0 {
		// Truncate long bodies
		if len(bodyStr) > 200 {
			bodyStr = bodyStr[:200]
		}
		err.Cause = fmt.Errorf("%s", bodyStr)
	}

	return err
}

// Ensure LimitError satisfies error interface.
var _ error = (*LimitError)(nil)

// CompactJSON returns compact string of JSON if possible.
func compactJSON(b []byte) string {
	var buf bytes.Buffer
	if err := json.Compact(&buf, b); err == nil {
		return buf.String()
	}
	return string(b)
}
