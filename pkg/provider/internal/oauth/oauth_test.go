package oauth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
)

func TestRequestDeviceCodeAndRefresh(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/device", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"device_code": "dev123",
			"user_code": "ABCD-1234",
			"verification_uri": "https://example.com/verify",
			"expires_in": 300,
			"interval": 1
		}`))
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		r.ParseForm()
		if r.Form.Get("grant_type") == "refresh_token" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{
				"access_token": "new_acc",
				"refresh_token": "new_ref",
				"expires_in": 7200
			}`))
			return
		}
		http.Error(w, "bad grant", http.StatusBadRequest)
	})

	srv := httptest.NewServer(mux)
	defer srv.Close()

	ctx := context.Background()
	cfg := DeviceFlowConfig{
		ClientID:      "test-client",
		DeviceAuthURL: srv.URL + "/device",
		TokenURL:      srv.URL + "/token",
		HTTPClient:    srv.Client(),
	}

	dcr, err := RequestDeviceCode(ctx, cfg)
	if err != nil {
		t.Fatalf("RequestDeviceCode failed: %v", err)
	}
	if dcr.UserCode != "ABCD-1234" || dcr.DeviceCode != "dev123" {
		t.Fatalf("unexpected dcr: %+v", dcr)
	}

	ref := MakeRefresher(srv.Client(), srv.URL+"/token", "test-client", "")
	oldCred := auth.Credential{
		RefreshToken: "old_ref",
		Generation:   1,
	}
	newCred, err := ref(ctx, oldCred)
	if err != nil {
		t.Fatalf("refresher failed: %v", err)
	}
	if newCred.AccessToken != "new_acc" || newCred.RefreshToken != "new_ref" {
		t.Fatalf("unexpected new cred: %+v", newCred)
	}
	if newCred.Generation != 2 {
		t.Fatalf("generation = %d, want 2", newCred.Generation)
	}
	if newCred.ExpiresAt.Before(time.Now().Add(7000 * time.Second)) {
		t.Fatalf("unexpected expiry: %v", newCred.ExpiresAt)
	}
}
