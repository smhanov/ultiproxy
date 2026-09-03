package zai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestZaiRequestShapeReasoningAndVision(t *testing.T) {
	var capturedPayload map[string]any
	var capturedAuth string
	var capturedPath string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		capturedPath = r.URL.Path

		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &capturedPayload)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "chatcmpl-zai-001",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Zai coding plan response",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	p, err := New(Config{
		BaseURL:    server.URL + "/api/coding/paas/v4",
		APIKey:     "test-zai-key",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create Z.ai provider: %v", err)
	}

	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "Look at this screenshot"},
				ir.ImageBlock{URL: "https://example.com/screenshot.png", Detail: "high"},
			},
		},
	}

	// 1. Test standard model (glm-5.3-flash) with reasoning
	resp, err := p.Generate(context.Background(), msgs,
		provider.WithModel("glm-5.3-flash"),
		provider.WithReasoningEffort("high"),
	)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}

	// Check endpoint path and auth
	if capturedPath != "/api/coding/paas/v4/chat/completions" {
		t.Errorf("expected coding paas endpoint, got %s", capturedPath)
	}
	if capturedAuth != "Bearer test-zai-key" {
		t.Errorf("expected bearer auth, got %s", capturedAuth)
	}

	// Check max tokens default for standard model
	if maxTok, ok := capturedPayload["max_tokens"].(float64); !ok || int(maxTok) != MaxOutputTokensStandard {
		t.Errorf("expected max_tokens %d, got %v", MaxOutputTokensStandard, capturedPayload["max_tokens"])
	}

	// Check reasoning effort
	if re, ok := capturedPayload["reasoning_effort"].(string); !ok || re != "high" {
		t.Errorf("expected reasoning_effort high, got %v", capturedPayload["reasoning_effort"])
	}

	// Check vision parts
	rawMsgs := capturedPayload["messages"].([]any)
	firstMsg := rawMsgs[0].(map[string]any)
	parts, ok := firstMsg["content"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("expected 2 content parts for vision, got %v", firstMsg["content"])
	}
	p1 := parts[0].(map[string]any)
	if p1["type"] != "text" || p1["text"] != "Look at this screenshot" {
		t.Errorf("unexpected part 1: %+v", p1)
	}
	p2 := parts[1].(map[string]any)
	if p2["type"] != "image_url" {
		t.Errorf("unexpected part 2 type: %+v", p2)
	}
	imgURL := p2["image_url"].(map[string]any)
	if imgURL["url"] != "https://example.com/screenshot.png" || imgURL["detail"] != "high" {
		t.Errorf("unexpected image_url payload: %+v", imgURL)
	}

	// 2. Test glm-4.5-air max_tokens default (98k)
	_, err = p.Generate(context.Background(), msgs, provider.WithModel("glm-4.5-air"))
	if err != nil {
		t.Fatalf("Generate air failed: %v", err)
	}
	if maxTok, ok := capturedPayload["max_tokens"].(float64); !ok || int(maxTok) != MaxOutputTokensAir {
		t.Errorf("expected max_tokens %d for glm-4.5-air, got %v", MaxOutputTokensAir, capturedPayload["max_tokens"])
	}
}

func TestZaiQuotaParse(t *testing.T) {
	fixtureData, err := os.ReadFile(filepath.Join("..", "..", "..", "testdata", "zai", "quota-limit.json"))
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	refTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	snap, err := ParseQuotaLimits(fixtureData, refTime)
	if err != nil {
		t.Fatalf("failed to parse quota limits: %v", err)
	}

	if len(snap.Windows) != 2 {
		t.Fatalf("expected 2 windows, got %d", len(snap.Windows))
	}

	w5h := snap.Windows[0]
	if w5h.Label != "5-hour" || w5h.Limit != 100.0 || w5h.Remaining != 65.0 || w5h.UsedPct != 35.0 || w5h.Unit != "credits" {
		t.Errorf("unexpected 5-hour window: %+v", w5h)
	}

	wWeek := snap.Windows[1]
	if wWeek.Label != "Weekly" || wWeek.Limit != 1000.0 || wWeek.Remaining != 760.0 || wWeek.UsedPct != 24.0 || wWeek.Unit != "credits" {
		t.Errorf("unexpected Weekly window: %+v", wWeek)
	}
}
