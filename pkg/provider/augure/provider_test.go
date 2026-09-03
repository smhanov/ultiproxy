package augure

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestAugureRefreshFlow(t *testing.T) {
	var refreshCalls atomic.Int32
	var receivedRefreshTokens []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/auth/v1/token" {
			refreshCalls.Add(1)
			if r.Header.Get("apikey") != "test-anon-key" {
				http.Error(w, "missing or invalid apikey header", http.StatusUnauthorized)
				return
			}
			var req map[string]string
			_ = json.NewDecoder(r.Body).Decode(&req)
			receivedRefreshTokens = append(receivedRefreshTokens, req["refresh_token"])

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token":  "new-access-token-123",
				"refresh_token": "rotated-refresh-token-456",
				"expires_in":    3600,
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	tempDir := t.TempDir()
	tokenFile := filepath.Join(tempDir, "augure-auth.json")

	// Write initial token (expired)
	initialTok := TokenData{
		AccessToken:  "old-access-token",
		RefreshToken: "initial-refresh-token",
		ExpiresAt:    time.Now().Unix() - 100, // expired
	}
	tokData, _ := json.Marshal(initialTok)
	_ = os.WriteFile(tokenFile, tokData, 0600)

	p, err := New(Config{
		RefreshURL: server.URL + "/auth/v1/token",
		TokenFile:  tokenFile,
		AnonKey:    "test-anon-key",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create Augure provider: %v", err)
	}

	// 1. Token() should trigger refresh because initial token is expired
	tok, err := p.Token(context.Background())
	if err != nil {
		t.Fatalf("Token failed: %v", err)
	}
	if tok != "new-access-token-123" {
		t.Errorf("expected new access token, got %s", tok)
	}
	if refreshCalls.Load() != 1 {
		t.Errorf("expected 1 refresh call, got %d", refreshCalls.Load())
	}
	if len(receivedRefreshTokens) != 1 || receivedRefreshTokens[0] != "initial-refresh-token" {
		t.Errorf("expected initial-refresh-token sent, got %v", receivedRefreshTokens)
	}

	// 2. Verify disk state updated
	diskTok, err := p.readTokenFile()
	if err != nil {
		t.Fatalf("failed to re-read token file: %v", err)
	}
	if diskTok.AccessToken != "new-access-token-123" {
		t.Errorf("expected persisted access token, got %s", diskTok.AccessToken)
	}
	if diskTok.RefreshToken != "rotated-refresh-token-456" {
		t.Errorf("expected persisted rotated refresh token, got %s", diskTok.RefreshToken)
	}
}

func TestAugureModelPassthroughAndVisionFalse(t *testing.T) {
	var receivedPayload map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/chat/completions" {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &receivedPayload)

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-augure",
				"choices": []map[string]any{
					{
						"index": 0,
						"message": map[string]any{
							"role":    "assistant",
							"content": "Canadian sovereign response",
						},
						"finish_reason": "stop",
					},
				},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	tempDir := t.TempDir()
	tokenFile := filepath.Join(tempDir, "augure-auth.json")
	initialTok := TokenData{
		AccessToken:  "valid-token",
		RefreshToken: "ref-tok",
		ExpiresAt:    time.Now().Unix() + 7200,
	}
	tokData, _ := json.Marshal(initialTok)
	_ = os.WriteFile(tokenFile, tokData, 0600)

	p, err := New(Config{
		BaseURL:    server.URL,
		TokenFile:  tokenFile,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "Analyze this text"},
				ir.ImageBlock{URL: "https://example.com/photo.png"},
			},
		},
	}

	// Request with ossington-5 model
	resp, err := p.Generate(context.Background(), msgs, provider.WithModel("ossington-5"))
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp == nil || len(resp.Message.Blocks) == 0 {
		t.Fatal("empty response")
	}

	// Verify model passthrough
	if receivedPayload["model"] != "ossington-5" {
		t.Errorf("expected model ossington-5, got %v", receivedPayload["model"])
	}

	// Verify vision false: messages content should be plain string, not multimodal parts
	rawMsgs := receivedPayload["messages"].([]any)
	firstMsg := rawMsgs[0].(map[string]any)
	contentStr, isString := firstMsg["content"].(string)
	if !isString {
		t.Errorf("expected text-only content string for Vision: false, got %T: %v", firstMsg["content"], firstMsg["content"])
	}
	if contentStr != "Analyze this text" {
		t.Errorf("unexpected content string: %s", contentStr)
	}
}

func TestAugureStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/chat/completions" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"id\":\"stream-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Tofino-3 output\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	tempDir := t.TempDir()
	tokenFile := filepath.Join(tempDir, "augure-auth.json")
	initialTok := TokenData{
		AccessToken:  "valid-token",
		RefreshToken: "ref-tok",
		ExpiresAt:    time.Now().Unix() + 7200,
	}
	tokData, _ := json.Marshal(initialTok)
	_ = os.WriteFile(tokenFile, tokData, 0600)

	p, err := New(Config{
		BaseURL:    server.URL,
		TokenFile:  tokenFile,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "hello"},
			},
		},
	}

	ch, err := p.Stream(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var hasText bool
	for ev := range ch {
		if txt, ok := ev.(ir.EventTextDelta); ok {
			if txt.Text == "Tofino-3 output" {
				hasText = true
			}
		}
	}

	if !hasText {
		t.Error("expected Tofino-3 output in stream text deltas")
	}
}
