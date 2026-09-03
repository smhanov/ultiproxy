package freebuff

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
	spikesfreebuff "github.com/smhanov/ultiproxy/pkg/spikes/freebuff"
)

func TestFreebuffActorIntegrationAndModelSwitch(t *testing.T) {
	var (
		sessionGets  atomic.Int32
		sessionPosts atomic.Int32
		agentRuns    atomic.Int32
		completions  atomic.Int32
		lastBound    string
		lastAgentID  string
		lastPayload  map[string]any
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/freebuff/session":
			sessionGets.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(spikesfreebuff.Session{
				InstanceID: "inst-test",
				Model:      "deepseek/deepseek-v4-flash",
			})

		case r.Method == http.MethodPost && r.URL.Path == "/freebuff/session":
			sessionPosts.Add(1)
			modelHeader := r.Header.Get("x-freebuff-model")
			var body map[string]string
			_ = json.NewDecoder(r.Body).Decode(&body)
			lastBound = modelHeader
			if lastBound == "" {
				lastBound = body["model"]
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(spikesfreebuff.Session{
				InstanceID: "inst-test",
				Model:      lastBound,
			})

		case r.Method == http.MethodPost && r.URL.Path == "/agent-runs":
			agentRuns.Add(1)
			var body map[string]string
			_ = json.NewDecoder(r.Body).Decode(&body)
			lastAgentID = body["agentId"]
			if lastAgentID == "" {
				lastAgentID = body["agent_id"]
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(spikesfreebuff.AgentRun{
				RunID:      "run-456",
				AgentID:    lastAgentID,
				InstanceID: "inst-test",
				Status:     "running",
			})

		case r.Method == http.MethodPost && r.URL.Path == "/chat/completions":
			completions.Add(1)
			bodyBytes, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(bodyBytes, &lastPayload)

			// Check user-agent and instance headers
			if ua := r.Header.Get("User-Agent"); ua != UserAgentValue {
				t.Errorf("expected User-Agent %q, got %q", UserAgentValue, ua)
			}
			if inst := r.Header.Get("x-freebuff-instance-id"); inst == "" {
				t.Errorf("missing x-freebuff-instance-id header")
			}

			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-001\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello from Freebuff!\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	tempDir := t.TempDir()
	lockPath := filepath.Join(tempDir, "freebuff.lock")

	p, err := New(Config{
		BaseURL:    server.URL,
		Token:      "mock-token",
		DataDir:    tempDir,
		LockPath:   lockPath,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create Freebuff provider: %v", err)
	}
	defer p.Close()

	// Initial model switch test: request z-ai/glm-5.3-flash
	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "Test message"},
			},
		},
	}

	ch, err := p.Stream(context.Background(), msgs, provider.WithModel("z-ai/glm-5.3-flash"))
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var events []ir.Event
	for ev := range ch {
		events = append(events, ev)
	}

	if len(events) == 0 {
		t.Fatal("expected streamed events, got 0")
	}

	if lastBound != "z-ai/glm-5.3-flash" {
		t.Errorf("expected bound model z-ai/glm-5.3-flash, got %s", lastBound)
	}
	if lastAgentID != "base3-free-glm-5-3-flash" {
		t.Errorf("expected agent ID base3-free-glm-5-3-flash, got %s", lastAgentID)
	}
	if completions.Load() != 1 {
		t.Errorf("expected 1 completion call, got %d", completions.Load())
	}
}

func TestFreebuffNoFakeToolsAndNoSystemMessageRewrite(t *testing.T) {
	var capturedPayload map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/agent-runs":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(spikesfreebuff.AgentRun{
				RunID:  "run-789",
				Status: "running",
			})
		case r.Method == http.MethodPost && r.URL.Path == "/chat/completions":
			bodyBytes, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(bodyBytes, &capturedPayload)

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "gen-1",
				"choices": []map[string]any{
					{
						"index": 0,
						"message": map[string]any{
							"role":    "assistant",
							"content": "ok",
						},
						"finish_reason": "stop",
					},
				},
			})
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()

	tempDir := t.TempDir()
	p, err := New(Config{
		BaseURL:    server.URL,
		Token:      "mock-token",
		DataDir:    tempDir,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}
	defer p.Close()

	userMsgText := "Just a raw user question without any system message"
	msgs := []*ir.Message{
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: userMsgText},
			},
		},
	}

	_, err = p.Generate(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// 1. NEVER inject fake tools
	if tools, exists := capturedPayload["tools"]; exists && tools != nil {
		t.Errorf("expected no tools in payload when none requested, got %v", tools)
	}

	// 2. Transparency over rewriting: check user content passed through as-is
	msgsRaw, ok := capturedPayload["messages"].([]any)
	if !ok || len(msgsRaw) != 1 {
		t.Fatalf("expected 1 message in payload, got %v", msgsRaw)
	}
	firstMsg := msgsRaw[0].(map[string]any)
	if firstMsg["role"] != "user" || firstMsg["content"] != userMsgText {
		t.Errorf("user message was rewritten: %+v", firstMsg)
	}
}

func TestFreebuffQuotaParseFixture(t *testing.T) {
	fixtureData, err := os.ReadFile(filepath.Join("..", "..", "..", "testdata", "freebuff", "usage.json"))
	if err != nil {
		t.Fatalf("failed to read fixture: %v", err)
	}

	refTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	snap, err := ParseUsageSnapshot(fixtureData, refTime)
	if err != nil {
		t.Fatalf("failed to parse usage snapshot: %v", err)
	}

	if len(snap.Windows) != 3 {
		t.Fatalf("expected 3 windows, got %d", len(snap.Windows))
	}

	// Daily
	daily := snap.Windows[0]
	if daily.Label != "Daily" || daily.Limit != 100 || daily.Remaining != 75 || daily.UsedPct != 25.0 {
		t.Errorf("unexpected Daily window: %+v", daily)
	}
	if daily.SecondsRemaining <= 0 {
		t.Errorf("expected positive seconds remaining, got %d", daily.SecondsRemaining)
	}

	// Weekly
	weekly := snap.Windows[1]
	if weekly.Label != "Weekly" || weekly.Limit != 500 || weekly.Remaining != 350 || weekly.UsedPct != 30.0 {
		t.Errorf("unexpected Weekly window: %+v", weekly)
	}

	// Monthly
	monthly := snap.Windows[2]
	if monthly.Label != "Monthly" || monthly.Limit != 2000 || monthly.Remaining != 1550 || monthly.UsedPct != 22.5 {
		t.Errorf("unexpected Monthly window: %+v", monthly)
	}
}
