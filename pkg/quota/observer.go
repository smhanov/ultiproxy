package quota

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/state"
	"github.com/smhanov/ultiproxy/pkg/storage"
)

type contextKey string

const (
	ProviderContextKey contextKey = "quota_provider"
	ModelContextKey    contextKey = "quota_model"
)

// WithProviderContext injects provider into request context.
func WithProviderContext(ctx context.Context, providerName string) context.Context {
	return context.WithValue(ctx, ProviderContextKey, providerName)
}

// WithModelContext injects model into request context.
func WithModelContext(ctx context.Context, modelName string) context.Context {
	return context.WithValue(ctx, ModelContextKey, modelName)
}

// Observer coordinates passive telemetry from HTTP responses.
type Observer struct {
	sm      *state.StateManager
	breaker *CircuitBreaker
	writer  *storage.Writer
	store   *QuotaStore
}

// NewObserver creates an Observer.
func NewObserver(sm *state.StateManager, breaker *CircuitBreaker, writer *storage.Writer, store *QuotaStore) *Observer {
	return &Observer{
		sm:      sm,
		breaker: breaker,
		writer:  writer,
		store:   store,
	}
}

// RecordClassifierResult processes a LimitError, updating circuit breaker, state snapshot, and storage.
func (o *Observer) RecordClassifierResult(limitErr LimitError) {
	if o == nil {
		return
	}

	if o.breaker != nil {
		o.breaker.RecordLimitError(limitErr)
	} else if o.sm != nil {
		now := time.Now().UTC()
		o.sm.Update(func(snap *state.RuntimeSnapshot) {
			p, ok := snap.Providers[limitErr.Provider]
			if !ok {
				return
			}
			p.ObservedAt = now
			p.Error = limitErr.Error()
			if limitErr.Permanent {
				p.Quota = state.QuotaExhausted
				p.Health = state.HealthUnavailable
			} else if limitErr.Kind == LimitKindQuotaWindow {
				p.Quota = state.QuotaExhausted
				if limitErr.RetryAt != nil {
					p.ValidUntil = *limitErr.RetryAt
				}
			} else {
				p.Circuit = state.CircuitOpen
				p.CircuitOpenedAt = now
				p.Health = state.HealthDegraded
			}
			snap.Providers[limitErr.Provider] = p
		})
	}

	if o.writer != nil && limitErr.Provider != "" {
		var retrySecs int
		if limitErr.RetryAfter > 0 {
			retrySecs = int(limitErr.RetryAfter.Seconds())
		}
		var resetAtStr string
		if limitErr.RetryAt != nil {
			resetAtStr = limitErr.RetryAt.Format(time.RFC3339)
		}
		_ = o.writer.TrackAttempt(storage.AttemptRecord{
			Provider:          limitErr.Provider,
			Model:             limitErr.Model,
			StatusCode:        http.StatusTooManyRequests,
			ErrorClass:        string(limitErr.Kind),
			RetryAfterSeconds: retrySecs,
			ResetAt:           resetAtStr,
		})
	}
}

// WrapRoundTripper wraps an http.RoundTripper with passive observation.
func (o *Observer) WrapRoundTripper(next http.RoundTripper) http.RoundTripper {
	if next == nil {
		next = http.DefaultTransport
	}
	return &observerRoundTripper{
		next:     next,
		observer: o,
	}
}

type observerRoundTripper struct {
	next     http.RoundTripper
	observer *Observer
}

func (ort *observerRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	providerHint := req.Header.Get("X-Provider")
	if providerHint == "" {
		if val, ok := req.Context().Value(ProviderContextKey).(string); ok {
			providerHint = val
		}
	}
	modelHint := req.Header.Get("X-Model")
	if modelHint == "" {
		if val, ok := req.Context().Value(ModelContextKey).(string); ok {
			modelHint = val
		}
	}

	resp, err := ort.next.RoundTrip(req)
	if err != nil {
		if providerHint != "" && ort.observer != nil && ort.observer.breaker != nil {
			ort.observer.breaker.RecordFailure(providerHint, err)
		}
		return nil, err
	}

	if ort.observer == nil {
		return resp, nil
	}

	// 1. Check for rate limit / overload status
	isRateLimit := resp.StatusCode == http.StatusTooManyRequests ||
		resp.StatusCode == 529 ||
		resp.StatusCode == http.StatusPaymentRequired ||
		(resp.StatusCode == http.StatusServiceUnavailable && getHeader(resp.Header, "Retry-After") != "")

	if isRateLimit {
		var bodyBytes []byte
		if resp.Body != nil {
			bodyBytes, _ = io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		}

		limitErr := Classify(resp.StatusCode, resp.Header, bodyBytes, providerHint)
		if modelHint != "" && limitErr.Model == "" {
			limitErr.Model = modelHint
		}
		ort.observer.RecordClassifierResult(limitErr)
		return resp, nil
	}

	// 2. For successful 2xx responses, record success and inspect passive rate-limit headers
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		if providerHint != "" && ort.observer.breaker != nil {
			ort.observer.breaker.RecordSuccess(providerHint)
		}

		ort.extractPassiveQuota(providerHint, resp.Header)
	}

	return resp, nil
}

func (ort *observerRoundTripper) extractPassiveQuota(providerHint string, headers http.Header) {
	if providerHint == "" || headers == nil {
		return
	}

	// Common rate-limit headers:
	// x-ratelimit-remaining-requests, x-ratelimit-remaining-tokens, x-ratelimit-remaining, ratelimit-remaining
	remReqStr := getHeader(headers, "x-ratelimit-remaining-requests")
	if remReqStr == "" {
		remReqStr = getHeader(headers, "x-ratelimit-remaining")
	}
	if remReqStr == "" {
		remReqStr = getHeader(headers, "ratelimit-remaining")
	}

	limitReqStr := getHeader(headers, "x-ratelimit-limit-requests")
	if limitReqStr == "" {
		limitReqStr = getHeader(headers, "x-ratelimit-limit")
	}
	if limitReqStr == "" {
		limitReqStr = getHeader(headers, "ratelimit-limit")
	}

	if remReqStr != "" {
		rem, err := strconv.ParseFloat(remReqStr, 64)
		if err == nil {
			limit, _ := strconv.ParseFloat(limitReqStr, 64)
			var usedPct float64
			if limit > 0 {
				usedPct = ((limit - rem) / limit) * 100.0
				if usedPct < 0 {
					usedPct = 0
				}
			}

			now := time.Now().UTC()
			obs := storage.QuotaObservationRecord{
				Provider:   providerHint,
				Label:      "requests",
				Remaining:  rem,
				Limit:      limit,
				UsedPct:    usedPct,
				Unit:       "requests",
				ObservedAt: now.Format(time.RFC3339),
				Source:     "header",
			}

			if ort.observer.writer != nil {
				_ = ort.observer.writer.TrackQuotaObservation(obs)
			}

			if ort.observer.store != nil {
				ort.observer.store.Set(providerHint, &provider.QuotaSnapshot{
					ObservedAt: now,
					Windows: []provider.QuotaWindow{
						{
							Label:     "requests",
							UsedPct:   usedPct,
							Remaining: rem,
							Limit:     limit,
							Unit:      "requests",
						},
					},
					Detail: "Observed from upstream response headers",
				})
			}
		}
	}
}
