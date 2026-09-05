package freebuff

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The acting-user-id must be resolved live from GET /api/v1/me?fields=id (the
// CLI does exactly this at run start) and cached per actor. Sending a stale or
// hardcoded user id with a valid token is itself a fingerprint mismatch.

func TestActor_ActingUserID_FetchedFromMe(t *testing.T) {
	var meCalls int
	var sawFields string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/me":
			meCalls++
			sawFields = r.URL.Query().Get("fields")
			if got := r.Header.Get("Authorization"); got != "Bearer tok-1" {
				t.Errorf("me Authorization = %q", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"usr-live-1","email":"a@b.c"}`))
		case r.URL.Path == "/freebuff/session":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":"none"}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	a, err := NewFreebuffAccountActor("", server.Client(), "tok-1", WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("NewFreebuffAccountActor: %v", err)
	}

	got := a.ActingUserID(context.Background())
	if got != "usr-live-1" {
		t.Fatalf("ActingUserID = %q, want usr-live-1 fetched from /me", got)
	}
	if sawFields != "id" {
		t.Errorf("me fields query = %q, want exactly \"id\" (binary sends fields=id)", sawFields)
	}

	// Cached: a second call must not re-hit /me.
	if got := a.ActingUserID(context.Background()); got != "usr-live-1" {
		t.Fatalf("second ActingUserID = %q", got)
	}
	if meCalls != 1 {
		t.Errorf("/me called %d times, want 1 (cached)", meCalls)
	}
}

// Bind must POST with NO body and NO instance header — the server mints the
// instanceId. Sending a JSON body was the historical bind bug.
func TestActor_Bind_NoBody_NoInstanceHeader(t *testing.T) {
	var bindBody []byte
	var bindInstHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/freebuff/session":
			if r.Method == "POST" {
				bindBody, _ = readAll(r)
				bindInstHeader = r.Header.Get("x-freebuff-instance-id")
				if got := r.Header.Get("x-freebuff-model"); got != "z-ai/glm-5.3-flash" {
					t.Errorf("bind x-freebuff-model = %q", got)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"status":"active","instanceId":"minted-1","model":"z-ai/glm-5.3-flash"}`))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":"none"}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	a, err := NewFreebuffAccountActor("", server.Client(), "tok-1", WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("NewFreebuffAccountActor: %v", err)
	}
	if err := a.Bind(context.Background(), "z-ai/glm-5.3-flash"); err != nil {
		t.Fatalf("Bind: %v", err)
	}
	if len(bindBody) != 0 {
		t.Errorf("bind body = %q, want none (server mints the id)", bindBody)
	}
	if bindInstHeader != "" {
		t.Errorf("bind x-freebuff-instance-id = %q, want absent", bindInstHeader)
	}
	if got := a.InstanceID(); got != "minted-1" {
		t.Errorf("InstanceID after bind = %q, want adopted minted-1", got)
	}
	var zero json.RawMessage
	_ = zero
}

func readAll(r *http.Request) ([]byte, error) {
	defer r.Body.Close()
	buf := make([]byte, 0, 64)
	tmp := make([]byte, 64)
	for {
		n, err := r.Body.Read(tmp)
		buf = append(buf, tmp[:n]...)
		if err != nil {
			break
		}
	}
	return buf, nil
}
