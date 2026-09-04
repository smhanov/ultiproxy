package antigravity

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/auth"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// loopbackGet performs a callback request on a dedicated client so the shared
// http.DefaultClient keep-alive pool cannot hand a request to a listener left
// over from an earlier test.
func loopbackGet(rawURL string) (*http.Response, error) {
	client := &http.Client{Transport: &http.Transport{DisableKeepAlives: true}, Timeout: 5 * time.Second}
	return client.Get(rawURL)
}

// waitForPortFree polls until 127.0.0.1:51121 can be bound again.
func waitForPortFree(t *testing.T) bool {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		ln, err := net.Listen("tcp", LoopbackListenAddr)
		if err == nil {
			_ = ln.Close()
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

func newLoopbackTestProvider(t *testing.T, tokenURL, baseURL string, client *http.Client, mgr *auth.Manager) *Provider {
	t.Helper()
	p := New(Config{
		BaseURL:      baseURL,
		TokenURL:     tokenURL,
		ClientID:     DefaultClientID,
		ClientSecret: DefaultClientSecret,
		AuthManager:  mgr,
		HTTPClient:   client,
	})
	t.Cleanup(func() { p.stopLoopback() })
	return p
}

func TestStartLogin_BuildsLoopbackURL(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p := newLoopbackTestProvider(t, "http://127.0.0.1:1/token", "http://127.0.0.1:1", nil, nil)

	info, err := p.StartLogin(ctx)
	if err != nil {
		t.Fatalf("StartLogin: %v", err)
	}
	if info.Kind != provider.LoginFlowAuthCode {
		t.Fatalf("kind = %q, want %q", info.Kind, provider.LoginFlowAuthCode)
	}

	raw := info.VerificationURI
	if !strings.HasPrefix(raw, DefaultAuthURL+"?") {
		t.Fatalf("url %q must start with %s?", raw, DefaultAuthURL)
	}
	// Deterministic url.Values encoding of the loopback redirect.
	if !strings.Contains(raw, "redirect_uri=http%3A%2F%2Flocalhost%3A51121%2Foauth-callback") {
		t.Fatalf("url %q missing encoded loopback redirect_uri", raw)
	}
	if strings.Contains(raw, "code_challenge") {
		t.Fatalf("url %q must not contain PKCE code_challenge", raw)
	}

	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	q := u.Query()
	want := map[string]string{
		"access_type":   "offline",
		"client_id":     DefaultClientID,
		"prompt":        "consent",
		"redirect_uri":  DefaultLoopbackRedirectURI,
		"response_type": "code",
		"scope":         DefaultScope,
	}
	for k, v := range want {
		if got := q.Get(k); got != v {
			t.Errorf("param %s = %q, want %q", k, got, v)
		}
	}
	if q.Get("code_challenge_method") != "" {
		t.Errorf("code_challenge_method must be absent, got %q", q.Get("code_challenge_method"))
	}
	state := q.Get("state")
	if len(state) < 16 {
		t.Fatalf("state %q too short / not random per attempt", state)
	}

	p.mu.RLock()
	pending := p.pendingAuth
	p.mu.RUnlock()
	if pending == nil {
		t.Fatal("pendingAuth not stored")
	}
	if pending.state != state {
		t.Fatalf("pending state %q != url state %q", pending.state, state)
	}
	if cap(pending.codeCh) != 1 {
		t.Fatalf("codeCh must be buffered(1), cap=%d", cap(pending.codeCh))
	}

	// Cancelling StartLogin's context must release the loopback port.
	cancel()
	if !waitForPortFree(t) {
		t.Fatal("loopback port still bound after StartLogin ctx cancel")
	}
}

func TestStartLogin_StateIsRandomPerAttempt(t *testing.T) {
	p := newLoopbackTestProvider(t, "http://127.0.0.1:1/token", "http://127.0.0.1:1", nil, nil)
	first, err := p.StartLogin(context.Background())
	if err != nil {
		t.Fatalf("StartLogin: %v", err)
	}
	second, err := p.StartLogin(context.Background())
	if err != nil {
		t.Fatalf("StartLogin (2nd): %v", err)
	}
	s1, _ := url.Parse(first.VerificationURI)
	s2, _ := url.Parse(second.VerificationURI)
	if s1.Query().Get("state") == s2.Query().Get("state") {
		t.Fatal("state must be random per attempt")
	}
}

func TestStartLogin_PortBusyIsAClearError(t *testing.T) {
	other, err := net.Listen("tcp", LoopbackListenAddr)
	if err != nil {
		t.Skipf("cannot pre-bind loopback port: %v", err)
	}
	defer other.Close()

	p := newLoopbackTestProvider(t, "http://127.0.0.1:1/token", "http://127.0.0.1:1", nil, nil)
	_, err = p.StartLogin(context.Background())
	if err == nil {
		t.Fatal("StartLogin must fail when the loopback port is taken")
	}
	if !strings.Contains(err.Error(), "port 51121 busy") {
		t.Fatalf("error should mention the busy port, got: %v", err)
	}
}

func TestCompleteLogin_LoopbackCapture(t *testing.T) {
	type tokenRequest struct {
		form url.Values
		path string
	}
	tokenCh := make(chan tokenRequest, 1)
	var sawAssist bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/token"):
			_ = r.ParseForm()
			tokenCh <- tokenRequest{form: r.PostForm, path: r.URL.Path}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token":  "loop-access",
				"refresh_token": "loop-refresh",
				"expires_in":    3600,
				"token_type":    "Bearer",
			})
		case strings.Contains(r.URL.Path, "loadCodeAssist"):
			sawAssist = true
			if got := r.Header.Get("Authorization"); got != "Bearer loop-access" {
				t.Errorf("assist auth = %q", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"cloudaicompanionProject": "loop-project"})
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	mgr, err := auth.NewManager(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	p := newLoopbackTestProvider(t, srv.URL+"/token", srv.URL, srv.Client(), mgr)

	info, err := p.StartLogin(context.Background())
	if err != nil {
		t.Fatalf("StartLogin: %v", err)
	}
	authURL, err := url.Parse(info.VerificationURI)
	if err != nil {
		t.Fatal(err)
	}
	state := authURL.Query().Get("state")

	// The browser would land here after Google's consent screen.
	callbackDone := make(chan *http.Response, 1)
	go func() {
		resp, err := loopbackGet(fmt.Sprintf("http://127.0.0.1:%d/oauth-callback?code=TESTCODE&state=%s", LoopbackCallbackPort, state))
		if err != nil {
			t.Errorf("callback request: %v", err)
			callbackDone <- nil
			return
		}
		callbackDone <- resp
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := p.CompleteLogin(ctx, ""); err != nil {
		t.Fatalf("CompleteLogin: %v", err)
	}

	resp := <-callbackDone
	if resp == nil {
		t.Fatal("no callback response")
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("callback status = %d body=%s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "Sign-in complete") {
		t.Fatalf("callback page = %q", body)
	}

	select {
	case req := <-tokenCh:
		for k, v := range map[string]string{
			"code":          "TESTCODE",
			"client_id":     DefaultClientID,
			"client_secret": DefaultClientSecret,
			"redirect_uri":  DefaultLoopbackRedirectURI,
			"grant_type":    "authorization_code",
		} {
			if got := req.form.Get(k); got != v {
				t.Errorf("token form %s = %q, want %q", k, got, v)
			}
		}
		if req.form.Get("code_verifier") != "" {
			t.Errorf("token form must not carry code_verifier, got %q", req.form.Get("code_verifier"))
		}
		if req.form.Encode() != "client_id="+url.QueryEscape(DefaultClientID)+
			"&client_secret="+url.QueryEscape(DefaultClientSecret)+
			"&code=TESTCODE&grant_type=authorization_code"+
			"&redirect_uri=http%3A%2F%2Flocalhost%3A51121%2Foauth-callback" {
			t.Errorf("unexpected token form: %s", req.form.Encode())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("token endpoint never saw the exchange")
	}

	if !sawAssist {
		t.Error("loadCodeAssist not called")
	}
	if p.ProjectID() != "loop-project" {
		t.Errorf("project = %q, want loop-project", p.ProjectID())
	}
	tok, err := p.Token(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tok != "loop-access" {
		t.Errorf("token = %q, want loop-access", tok)
	}
	disk, err := mgr.LoadFromDisk(DefaultClientID)
	if err != nil {
		t.Fatal(err)
	}
	if disk.RefreshToken != "loop-refresh" || disk.ProjectID != "loop-project" {
		t.Errorf("disk cred: %+v", disk)
	}

	p.mu.RLock()
	pending := p.pendingAuth
	p.mu.RUnlock()
	if pending != nil {
		t.Error("pendingAuth must be cleared after success")
	}
	if !waitForPortFree(t) {
		t.Error("loopback port still bound after CompleteLogin")
	}
}

func TestCompleteLogin_AcceptsPastedCode(t *testing.T) {
	sawCode := ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/token") {
			http.NotFound(w, r)
			return
		}
		_ = r.ParseForm()
		sawCode = r.PostForm.Get("code")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"pasted-access","refresh_token":"pasted-refresh","expires_in":3600,"token_type":"Bearer"}`))
	}))
	defer srv.Close()

	mgr, err := auth.NewManager(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	p := newLoopbackTestProvider(t, srv.URL+"/token", srv.URL, srv.Client(), mgr)
	if _, err := p.StartLogin(context.Background()); err != nil {
		t.Fatalf("StartLogin: %v", err)
	}
	if err := p.CompleteLogin(context.Background(), "PASTEDCODE"); err != nil {
		t.Fatalf("CompleteLogin: %v", err)
	}
	if sawCode != "PASTEDCODE" {
		t.Fatalf("exchanged code = %q, want PASTEDCODE", sawCode)
	}
	if tok, _ := p.Token(context.Background()); tok != "pasted-access" {
		t.Fatalf("token = %q", tok)
	}
}

func TestCompleteLogin_StateMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("token endpoint must not be reached on a state mismatch: %s", r.URL.Path)
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	mgr, err := auth.NewManager(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	p := newLoopbackTestProvider(t, srv.URL+"/token", srv.URL, srv.Client(), mgr)

	info, err := p.StartLogin(context.Background())
	if err != nil {
		t.Fatalf("StartLogin: %v", err)
	}
	u, _ := url.Parse(info.VerificationURI)
	state := u.Query().Get("state")
	if state == "wrong-state" {
		t.Fatal("unlucky state collision with the test fixture")
	}

	resp, err := loopbackGet(fmt.Sprintf("http://127.0.0.1:%d/oauth-callback?code=EVILCODE&state=wrong-state", LoopbackCallbackPort))
	if err != nil {
		t.Fatalf("callback request: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("callback status = %d, want 400", resp.StatusCode)
	}
	if !strings.Contains(string(body), "state mismatch") {
		t.Fatalf("callback body = %q", body)
	}

	// The code must NOT have been delivered, so the poll times out.
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	err = p.CompleteLogin(ctx, "")
	if err == nil {
		t.Fatal("CompleteLogin must not complete on a state mismatch")
	}
	p.mu.RLock()
	pending := p.pendingAuth
	p.mu.RUnlock()
	if pending == nil {
		t.Fatal("pendingAuth must survive a rejected callback")
	}
	select {
	case code := <-pending.codeCh:
		t.Fatalf("channel delivered %q on state mismatch", code)
	default:
	}
	if tok, _ := p.Token(context.Background()); tok == "EVILCODE" || tok != "" {
		t.Fatalf("token = %q, want empty", tok)
	}
}

func TestCompleteLogin_EmptyCodeAndErrorCodeParam(t *testing.T) {
	p := newLoopbackTestProvider(t, "http://127.0.0.1:1/token", "http://127.0.0.1:1", nil, nil)
	info, err := p.StartLogin(context.Background())
	if err != nil {
		t.Fatalf("StartLogin: %v", err)
	}
	u, _ := url.Parse(info.VerificationURI)
	state := u.Query().Get("state")

	resp, err := loopbackGet(fmt.Sprintf("http://127.0.0.1:%d/oauth-callback?error=access_denied&state=%s", LoopbackCallbackPort, state))
	if err != nil {
		t.Fatalf("callback request: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("callback status = %d, want 400", resp.StatusCode)
	}
	if !strings.Contains(string(body), "access_denied") {
		t.Fatalf("callback body = %q", body)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if err := p.CompleteLogin(ctx, ""); err == nil {
		t.Fatal("CompleteLogin must not complete without a code")
	}
}

func TestCompleteLogin_RearmsListenerAfterStartLoginCtxEnded(t *testing.T) {
	sawCode := ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/token") {
			http.NotFound(w, r)
			return
		}
		_ = r.ParseForm()
		sawCode = r.PostForm.Get("code")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"rearm-access","refresh_token":"rearm-refresh","expires_in":3600,"token_type":"Bearer"}`))
	}))
	defer srv.Close()

	mgr, err := auth.NewManager(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	p := newLoopbackTestProvider(t, srv.URL+"/token", srv.URL, srv.Client(), mgr)

	// initiate_oauth_login is served with the HTTP request's context, which is
	// cancelled as soon as the sign-in URL has been returned.
	startCtx, cancelStart := context.WithCancel(context.Background())
	info, err := p.StartLogin(startCtx)
	if err != nil {
		t.Fatalf("StartLogin: %v", err)
	}
	cancelStart()
	if !waitForPortFree(t) {
		t.Fatal("loopback port still bound after the initiate request ended")
	}
	u, _ := url.Parse(info.VerificationURI)
	state := u.Query().Get("state")

	// check_oauth_login then polls with its own (shorter) context.
	pollCtx, cancelPoll := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelPoll()
	done := make(chan error, 1)
	go func() { done <- p.CompleteLogin(pollCtx, "") }()

	// The browser lands on the loopback port once the poll re-armed it.
	callbackURL := fmt.Sprintf("http://127.0.0.1:%d/oauth-callback?code=REARMCODE&state=%s", LoopbackCallbackPort, state)
	status := 0
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := loopbackGet(callbackURL)
		if err == nil {
			_, _ = io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			status = resp.StatusCode
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if status != http.StatusOK {
		t.Fatalf("callback status = %d, want 200 (listener not re-armed)", status)
	}
	if err := <-done; err != nil {
		t.Fatalf("CompleteLogin: %v", err)
	}
	if sawCode != "REARMCODE" {
		t.Fatalf("exchanged code = %q, want REARMCODE", sawCode)
	}
	if tok, _ := p.Token(context.Background()); tok != "rearm-access" {
		t.Fatalf("token = %q", tok)
	}
}
