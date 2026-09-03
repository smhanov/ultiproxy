package quota

import (
	"bytes"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/state"
)

type mockTransport struct {
	roundTripFn func(req *http.Request) (*http.Response, error)
}

func (m *mockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.roundTripFn(req)
}

func TestObserverRoundTripper429(t *testing.T) {
	sm := state.NewStateManager()
	sm.SetProvider("copilot", state.ProviderRuntime{
		Admin:      state.AdminEnabled,
		Health:     state.HealthHealthy,
		Quota:      state.QuotaHealthy,
		Circuit:    state.CircuitClosed,
		Credential: state.CredentialValid,
	})

	cb := NewCircuitBreaker(sm, CircuitBreakerConfig{
		DefaultOpenDuration: 30 * time.Second,
	})
	store := NewQuotaStore()
	obs := NewObserver(sm, cb, nil, store)

	// Mock upstream returning 429 burst rate
	rt := obs.WrapRoundTripper(&mockTransport{
		roundTripFn: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Header: http.Header{
					"Retry-After": []string{"10"},
				},
				Body: io.NopCloser(bytes.NewReader([]byte(`{"error":{"type":"requests","message":"Rate limit reached for requests per minute"}}`))),
			}, nil
		},
	})

	req, _ := http.NewRequest(http.MethodPost, "https://api.githubcopilot.com/chat", nil)
	req.Header.Set("X-Provider", "copilot")

	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip err: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if len(body) == 0 {
		t.Errorf("expected response body to be readable after observer inspection")
	}

	// Provider circuit should now be Open
	snap := sm.Snapshot()
	p := snap.Providers["copilot"]
	if p.Circuit != state.CircuitOpen {
		t.Errorf("expected CircuitOpen after 429, got %v", p.Circuit)
	}
	if p.Health != state.HealthDegraded {
		t.Errorf("expected HealthDegraded, got %v", p.Health)
	}
}

func TestObserverPassiveHeaders(t *testing.T) {
	store := NewQuotaStore()
	obs := NewObserver(nil, nil, nil, store)

	rt := obs.WrapRoundTripper(&mockTransport{
		roundTripFn: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header: http.Header{
					"X-Ratelimit-Remaining-Requests": []string{"450"},
					"X-Ratelimit-Limit-Requests":     []string{"500"},
				},
				Body: io.NopCloser(bytes.NewReader([]byte(`{"choices":[]}`))),
			}, nil
		},
	})

	req, _ := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat", nil)
	req.Header.Set("X-Provider", "openai")

	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	defer resp.Body.Close()

	snap, ok := store.Get("openai")
	if !ok {
		t.Fatalf("expected quota store to have snapshot for openai")
	}
	if len(snap.Windows) == 0 {
		t.Fatalf("expected windows in quota snapshot")
	}
	if snap.Windows[0].Remaining != 450 {
		t.Errorf("expected remaining=450, got %v", snap.Windows[0].Remaining)
	}
	if snap.Windows[0].Limit != 500 {
		t.Errorf("expected limit=500, got %v", snap.Windows[0].Limit)
	}
}
