package quota

import (
	"net/http"
	"testing"
	"time"
)

func TestClassifyTableDriven(t *testing.T) {
	tests := []struct {
		name         string
		httpStatus   int
		headers      http.Header
		body         []byte
		providerHint string
		wantKind     LimitErrorKind
		wantScope    string
		wantPerm     bool
		minRetry     time.Duration
	}{
		{
			name:         "Overload - 529 Status Code",
			httpStatus:   529,
			headers:      http.Header{"Retry-After": []string{"15"}},
			body:         []byte(`{"error": {"type": "overloaded_error", "message": "Server is temporarily overloaded"}}`),
			providerHint: "anthropic",
			wantKind:     LimitKindCapacity,
			wantScope:    "server",
			wantPerm:     false,
			minRetry:     15 * time.Second,
		},
		{
			name:         "Overload - 503 Capacity message",
			httpStatus:   503,
			headers:      http.Header{},
			body:         []byte(`{"message": "Capacity exceeded due to high demand"}`),
			providerHint: "openai",
			wantKind:     LimitKindCapacity,
			wantScope:    "server",
			wantPerm:     false,
		},
		{
			name:         "Spend - Insufficient Quota 429",
			httpStatus:   429,
			headers:      http.Header{},
			body:         []byte(`{"error": {"message": "You exceeded your current quota, please check your plan and billing details.", "type": "insufficient_quota", "code": "insufficient_quota"}}`),
			providerHint: "openai",
			wantKind:     LimitKindSpendLimit,
			wantScope:    "account",
			wantPerm:     true,
		},
		{
			name:         "Spend - 402 Payment Required",
			httpStatus:   402,
			headers:      http.Header{},
			body:         []byte(`{"error": "Credit balance is too low to fulfill request"}`),
			providerHint: "openrouter",
			wantKind:     LimitKindSpendLimit,
			wantScope:    "account",
			wantPerm:     true,
		},
		{
			name:         "Session Conflict - Concurrent session active",
			httpStatus:   409,
			headers:      http.Header{},
			body:         []byte(`{"detail": "A session is already in progress, concurrent session conflict"}`),
			providerHint: "freebuff",
			wantKind:     LimitKindSessionConflict,
			wantScope:    "session",
			wantPerm:     false,
		},
		{
			name:         "Token Rate - TPM limit",
			httpStatus:   429,
			headers:      http.Header{"retry-after-ms": []string{"2500"}},
			body:         []byte(`{"error": {"type": "tokens", "message": "Rate limit reached for tokens per minute"}}`),
			providerHint: "openai",
			wantKind:     LimitKindTokenRate,
			wantScope:    "tokens",
			wantPerm:     false,
			minRetry:     2500 * time.Millisecond,
		},
		{
			name:         "Quota Window - Requests per day",
			httpStatus:   429,
			headers:      http.Header{},
			body:         []byte(`{"message": "Daily limit reached for 5-hour window"}`),
			providerHint: "copilot",
			wantKind:     LimitKindQuotaWindow,
			wantScope:    "window",
			wantPerm:     false,
		},
		{
			name:         "Burst Rate - RPM limit",
			httpStatus:   429,
			headers:      http.Header{"x-ratelimit-reset": []string{"10"}},
			body:         []byte(`{"error": {"message": "Rate limit reached for requests per minute (RPM)", "code": "rate_limit_exceeded"}}`),
			providerHint: "openai",
			wantKind:     LimitKindBurstRate,
			wantScope:    "requests",
			wantPerm:     false,
			minRetry:     10 * time.Second,
		},
		{
			name:         "Abuse Guard - 403 Suspended",
			httpStatus:   403,
			headers:      http.Header{},
			body:         []byte(`{"message": "Account suspended for policy violation"}`),
			providerHint: "anthropic",
			wantKind:     LimitKindAbuseGuard,
			wantScope:    "account",
			wantPerm:     true,
		},
		{
			name:         "Waiting Room",
			httpStatus:   429,
			headers:      http.Header{},
			body:         []byte(`{"message": "Request placed in waiting room queue"}`),
			providerHint: "codex",
			wantKind:     LimitKindWaitingRoom,
			wantScope:    "queue",
			wantPerm:     false,
		},
		{
			name:         "Unknown - Unrecognized 429",
			httpStatus:   429,
			headers:      http.Header{},
			body:         []byte(`{"random": "payload with no known keywords"}`),
			providerHint: "custom",
			wantKind:     LimitKindUnknown,
			wantScope:    "unknown",
			wantPerm:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Classify(tt.httpStatus, tt.headers, tt.body, tt.providerHint)

			if got.Kind != tt.wantKind {
				t.Errorf("Kind = %v, want %v", got.Kind, tt.wantKind)
			}
			if got.Scope != tt.wantScope {
				t.Errorf("Scope = %v, want %v", got.Scope, tt.wantScope)
			}
			if got.Permanent != tt.wantPerm {
				t.Errorf("Permanent = %v, want %v", got.Permanent, tt.wantPerm)
			}
			if got.Provider != tt.providerHint {
				t.Errorf("Provider = %v, want %v", got.Provider, tt.providerHint)
			}
			if tt.minRetry > 0 && got.RetryAfter < tt.minRetry {
				t.Errorf("RetryAfter = %v, want at least %v", got.RetryAfter, tt.minRetry)
			}
			if tt.minRetry > 0 && got.RetryAt == nil {
				t.Errorf("RetryAt is nil, expected timestamp")
			}
			if got.Error() == "" {
				t.Errorf("Error() returned empty string")
			}
		})
	}
}

func TestClassifyHTTPDateRetryAfter(t *testing.T) {
	future := time.Now().UTC().Add(30 * time.Second).Truncate(time.Second)
	dateStr := future.Format(http.TimeFormat)

	headers := http.Header{
		"Retry-After": []string{dateStr},
	}
	got := Classify(429, headers, []byte(`{"message": "too many requests"}`), "copilot")
	if got.Kind != LimitKindBurstRate {
		t.Errorf("expected BurstRate, got %v", got.Kind)
	}
	if got.RetryAfter <= 0 || got.RetryAfter > 35*time.Second {
		t.Errorf("expected RetryAfter ~30s, got %v", got.RetryAfter)
	}
	if got.RetryAt == nil {
		t.Fatalf("expected non-nil RetryAt")
	}
}
