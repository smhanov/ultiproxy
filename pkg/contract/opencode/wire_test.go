package opencode

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
)

// wireActor is a minimal local fake for the freebuff account actor injected
// through Config.Quirks.FreebuffActor. It counts Acquire/Release so the wire
// test can assert the serialized account lock is balanced end-to-end.
type wireActor struct {
	mu            sync.Mutex
	instanceID    string
	acquireCount  int
	releaseCount  int
	active        int
	maxActive     int
	startRunCalls int
}

// Acquire implements openaicompat.FreebuffActor.
func (a *wireActor) Acquire(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.acquireCount++
	a.active++
	if a.active > a.maxActive {
		a.maxActive = a.active
	}
	return nil
}

// Release implements openaicompat.FreebuffActor.
func (a *wireActor) Release() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.releaseCount++
	a.active--
}

// InstanceID feeds the x-freebuff-instance-id header and codebuff_metadata.
func (a *wireActor) InstanceID() string { return a.instanceID }

// StartRun feeds codebuff_metadata.run_id.
func (a *wireActor) StartRun(ctx context.Context, model string) (any, error) {
	a.mu.Lock()
	a.startRunCalls++
	a.mu.Unlock()
	return "run-wire-42", nil
}

func (a *wireActor) counts() (acquired, released, active, maxActive int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.acquireCount, a.releaseCount, a.active, a.maxActive
}

// newWireHarness builds a harness whose quirk-bearing lane is a FakeProvider
// named name, wired through the real stack: downstream HTTP -> server ->
// router -> hublane -> llmhub openai client -> FakeUpstream.
func newWireHarness(t *testing.T, name string, opts ...func(*openaicompat.Config)) (*Harness, *FakeProvider) {
	t.Helper()

	fake := NewFakeUpstream()
	fp, err := NewFakeProvider(name, fake, opts...)
	if err != nil {
		t.Fatalf("NewFakeProvider(%q) failed: %v", name, err)
	}
	h := NewTestHarness(t, WithFakeUpstream(fake), WithProvider(fp.Provider()))
	return h, fp
}

// postChatAndRecord posts exactly one non-streaming chat completion through the
// server and returns the single request recorded by the fake upstream.
func postChatAndRecord(t *testing.T, h *Harness, req map[string]any) *RecordedRequest {
	t.Helper()

	h.FakeUpstream.QueueChatCompletion("wire-ok")

	resp, body, err := h.PostChat(context.Background(), req)
	if err != nil {
		t.Fatalf("PostChat failed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 from server, got %d: %v", resp.StatusCode, body)
	}
	if got := h.FakeUpstream.RequestCount(); got != 1 {
		t.Fatalf("expected exactly 1 upstream request, got %d", got)
	}
	return h.FakeUpstream.Requests()[0]
}

// TestWire_CodingPlanMaxTokens verifies the zai coding-plan quirk on the wire:
// max_tokens is derived per model (glm-4.5-air -> 98304, everything else ->
// 131072 default) and lands in the upstream request body.
func TestWire_CodingPlanMaxTokens(t *testing.T) {
	quirk := func(c *openaicompat.Config) {
		c.Quirks.CodingPlanPath = true
		c.Quirks.MaxTokensByModel = map[string]int{"glm-4.5-air": 98304}
	}

	t.Run("air model gets 98304", func(t *testing.T) {
		h, _ := newWireHarness(t, "codplan", quirk)

		rec := postChatAndRecord(t, h, map[string]any{
			"model": "codplan/glm-4.5-air",
			"messages": []map[string]any{
				{"role": "user", "content": "Write quicksort"},
			},
		})

		if rec.Model != "glm-4.5-air" {
			t.Errorf("expected upstream model %q, got %q", "glm-4.5-air", rec.Model)
		}
		if rec.MaxTokens != 98304 {
			t.Errorf("expected max_tokens 98304 for glm-4.5-air, got %d", rec.MaxTokens)
		}
		if raw, ok := rec.JSON["max_tokens"].(float64); !ok || int(raw) != 98304 {
			t.Errorf("expected max_tokens 98304 in upstream body, got %v", rec.JSON["max_tokens"])
		}
	})

	t.Run("other model gets 131072 default", func(t *testing.T) {
		h, _ := newWireHarness(t, "codplan", quirk)

		rec := postChatAndRecord(t, h, map[string]any{
			"model": "codplan/glm-5.3",
			"messages": []map[string]any{
				{"role": "user", "content": "Write mergesort"},
			},
		})

		if rec.MaxTokens != 131072 {
			t.Errorf("expected max_tokens 131072 for glm-5.3, got %d", rec.MaxTokens)
		}
		if raw, ok := rec.JSON["max_tokens"].(float64); !ok || int(raw) != 131072 {
			t.Errorf("expected max_tokens 131072 in upstream body, got %v", rec.JSON["max_tokens"])
		}
	})
}

// TestWire_EchoReasoning verifies the deepseek quirk on the wire: an assistant
// turn carrying reasoning is replayed upstream in the dedicated
// reasoning_content field and never merged into content.
func TestWire_EchoReasoning(t *testing.T) {
	h, _ := newWireHarness(t, "deepecho", func(c *openaicompat.Config) {
		c.Quirks.EchoReasoning = true
	})

	rec := postChatAndRecord(t, h, map[string]any{
		"model": "deepecho/deepseek-reasoner",
		"messages": []map[string]any{
			{"role": "user", "content": "sqrt(144)?"},
			{"role": "assistant", "reasoning_content": "prior reasoning", "content": "12"},
			{"role": "user", "content": "double it"},
		},
	})

	if len(rec.Messages) != 3 {
		t.Fatalf("expected 3 upstream messages, got %d: %+v", len(rec.Messages), rec.Messages)
	}

	asst := rec.Messages[1]
	if asst.Role != "assistant" {
		t.Errorf("expected message 1 to keep the assistant role, got %q", asst.Role)
	}
	if asst.ReasoningContent != "prior reasoning" {
		t.Errorf("expected reasoning_content %q on the wire, got %q", "prior reasoning", asst.ReasoningContent)
	}
	if asst.ContentString() != "12" {
		t.Errorf("expected content %q on the wire, got %q", "12", asst.ContentString())
	}

	// The reasoning must never be folded into any message's content field.
	for i, m := range rec.Messages {
		if strings.Contains(m.ContentString(), "prior reasoning") {
			t.Errorf("message %d merged reasoning into content: %q", i, m.ContentString())
		}
	}
	if strings.Contains(rec.BodyString(), `"content":"prior reasoning"`) {
		t.Errorf("reasoning leaked into a content field: %s", rec.BodyString())
	}
}

// TestWire_WorkspaceCookieAuth verifies the opencode quirk on the wire: the
// workspace id and session cookie reach the upstream as headers.
func TestWire_WorkspaceCookieAuth(t *testing.T) {
	h, _ := newWireHarness(t, "wslane", func(c *openaicompat.Config) {
		c.Quirks.AuthViaWorkspaceCookie = true
		c.WorkspaceID = "ws-test"
		c.SessionCookie = "sess-test"
	})

	rec := postChatAndRecord(t, h, map[string]any{
		"model": "wslane/gpt-5",
		"messages": []map[string]any{
			{"role": "user", "content": "check auth"},
		},
	})

	if cookie := rec.GetHeader("Cookie"); !strings.Contains(cookie, "session=sess-test") {
		t.Errorf("expected Cookie header containing session=sess-test, got %q", cookie)
	}
	if ws := rec.GetHeader("X-Workspace-ID"); ws != "ws-test" {
		t.Errorf("expected X-Workspace-ID %q, got %q", "ws-test", ws)
	}
}

// TestWire_ModelListPassthrough verifies the vllm quirk on the wire: the model
// catalog is discovered from the upstream /v1/models endpoint. GET /v1/models on
// the ultiproxy server is served from the local catalog, so the passthrough is
// asserted on the provider's FetchModels/Models call against the fake upstream.
func TestWire_ModelListPassthrough(t *testing.T) {
	modelsPayload := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"id": "meta-llama/Llama-3-8B-Instruct"},
			{"id": "Qwen/Qwen2.5-Coder-32B"},
		},
	}

	// Consumed by the startup discovery inside openaicompat.New.
	fake := NewFakeUpstream()
	fake.QueueJSON(http.StatusOK, modelsPayload)

	fp, err := NewFakeProvider("vllmlane", fake, func(c *openaicompat.Config) {
		c.Quirks.ModelListPassthrough = true
	})
	if err != nil {
		t.Fatalf("NewFakeProvider failed: %v", err)
	}
	h := NewTestHarness(t, WithFakeUpstream(fake), WithProvider(fp.Provider()))

	discovered := fp.inner.Models()
	if len(discovered) != 2 {
		t.Fatalf("expected 2 models discovered at construction, got %d: %v", len(discovered), discovered)
	}
	if discovered[0] != "meta-llama/Llama-3-8B-Instruct" {
		t.Errorf("expected first discovered model meta-llama/Llama-3-8B-Instruct, got %q", discovered[0])
	}

	// Re-fetch explicitly so the recorded upstream call is deterministic.
	fake.ResetRequests()
	fake.QueueJSON(http.StatusOK, modelsPayload)

	models, err := fp.inner.FetchModels(context.Background())
	if err != nil {
		t.Fatalf("FetchModels failed: %v", err)
	}
	if len(models) != 2 || models[0] != "meta-llama/Llama-3-8B-Instruct" || models[1] != "Qwen/Qwen2.5-Coder-32B" {
		t.Errorf("unexpected model list from upstream: %v", models)
	}

	reqs := fake.Requests()
	if len(reqs) != 1 {
		t.Fatalf("expected exactly 1 upstream models request, got %d", len(reqs))
	}
	if reqs[0].Method != http.MethodGet {
		t.Errorf("expected GET models request, got %s", reqs[0].Method)
	}
	if !strings.HasSuffix(reqs[0].Path, "/v1/models") {
		t.Errorf("expected upstream path /v1/models, got %q", reqs[0].Path)
	}

	// The server's own /v1/models is catalog-backed and must not fan out to
	// the lane upstream.
	before := fake.RequestCount()
	srvResp, err := h.Client().Get(h.URL() + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models failed: %v", err)
	}
	defer func() { _ = srvResp.Body.Close() }()
	if srvResp.StatusCode != http.StatusOK {
		t.Errorf("expected 200 from server /v1/models, got %d", srvResp.StatusCode)
	}
	if after := fake.RequestCount(); after != before {
		t.Errorf("server /v1/models should not hit the upstream, requests went %d -> %d", before, after)
	}
}

// TestWire_FreebuffQuirk verifies the freebuff quirks on the wire: the Buffy
// system prompt, the read_files default tool, codebuff_metadata, the instance id
// header, and a balanced account lock.
func TestWire_FreebuffQuirk(t *testing.T) {
	actor := &wireActor{instanceID: "fb-inst-wire"}

	h, _ := newWireHarness(t, "freebufflane", func(c *openaicompat.Config) {
		c.Quirks.FreebuffActor = actor
		c.Quirks.FreebuffDefaultTool = true
	})

	rec := postChatAndRecord(t, h, map[string]any{
		"model": "freebufflane/claude-sonnet-4",
		"messages": []map[string]any{
			{"role": "user", "content": "help me code"},
		},
	})

	// Buffy system prompt prepended.
	if len(rec.Messages) == 0 {
		t.Fatal("expected upstream messages, got none")
	}
	first := rec.Messages[0]
	if first.Role != "system" {
		t.Fatalf("expected a system message first, got role %q", first.Role)
	}
	if !strings.Contains(first.ContentString(), "You are Buffy") {
		t.Errorf("expected Buffy system prompt, got %q", first.ContentString())
	}

	// read_files default tool injected (request carried no tools).
	if !rec.HasTool("read_files") {
		t.Errorf("expected default read_files tool on the wire, got %+v", rec.Tools)
	}
	if len(rec.Tools) != 1 {
		t.Errorf("expected exactly 1 injected tool, got %d: %+v", len(rec.Tools), rec.Tools)
	}

	// codebuff_metadata injected.
	meta, ok := rec.JSON["codebuff_metadata"].(map[string]any)
	if !ok {
		t.Fatalf("expected codebuff_metadata in upstream body, got %v", rec.JSON["codebuff_metadata"])
	}
	if meta["run_id"] != "run-wire-42" {
		t.Errorf("expected run_id run-wire-42, got %v", meta["run_id"])
	}
	if meta["freebuff_instance_id"] != "fb-inst-wire" {
		t.Errorf("expected freebuff_instance_id fb-inst-wire, got %v", meta["freebuff_instance_id"])
	}
	if meta["cost_mode"] != "free" {
		t.Errorf("expected cost_mode free, got %v", meta["cost_mode"])
	}

	// Actor instance id header.
	if got := rec.GetHeader("x-freebuff-instance-id"); got != "fb-inst-wire" {
		t.Errorf("expected x-freebuff-instance-id fb-inst-wire, got %q", got)
	}
	if rec.GetHeader("x-freebuff-acting-user-id") == "" {
		t.Error("expected x-freebuff-acting-user-id header to be set")
	}

	// The account lock must be acquired and released exactly once.
	acquired, released, active, maxActive := actor.counts()
	if acquired != 1 || released != 1 {
		t.Errorf("expected balanced actor lock (1 acquire / 1 release), got %d/%d", acquired, released)
	}
	if active != 0 {
		t.Errorf("expected actor lock fully released, %d still held", active)
	}
	if maxActive != 1 {
		t.Errorf("expected max 1 concurrent actor hold, got %d", maxActive)
	}
}

// TestWire_QuirksUnsetByDefault is the negative guard: a plain openaicompat
// lane rewrites nothing - no injected tools or codebuff_metadata, no reasoning
// echo, no cookie/workspace headers, and max_tokens passes through untouched.
func TestWire_QuirksUnsetByDefault(t *testing.T) {
	h, _ := newWireHarness(t, "plain")

	rec := postChatAndRecord(t, h, map[string]any{
		"model":      "plain/gpt-4o",
		"max_tokens": 512,
		"messages": []map[string]any{
			{"role": "assistant", "reasoning_content": "prior thoughts", "content": "prior answer"},
			{"role": "user", "content": "hello"},
		},
	})

	if _, ok := rec.JSON["tools"]; ok {
		t.Errorf("tools should not be injected by default, got %v", rec.JSON["tools"])
	}
	if _, ok := rec.JSON["codebuff_metadata"]; ok {
		t.Errorf("codebuff_metadata should not be present, got %v", rec.JSON["codebuff_metadata"])
	}

	if len(rec.Messages) != 2 {
		t.Fatalf("expected 2 upstream messages (no Buffy prompt), got %d: %+v", len(rec.Messages), rec.Messages)
	}
	for i, m := range rec.Messages {
		if m.ReasoningContent != "" {
			t.Errorf("message %d should not echo reasoning_content, got %q", i, m.ReasoningContent)
		}
	}
	if got := rec.Messages[0].ContentString(); got != "prior answer" {
		t.Errorf("expected prior assistant content %q, got %q", "prior answer", got)
	}

	for _, hdr := range []string{"Cookie", "X-Workspace-ID", "x-freebuff-instance-id", "x-freebuff-acting-user-id"} {
		if rec.HasHeader(hdr) {
			t.Errorf("header %q should not be set by default, got %q", hdr, rec.GetHeader(hdr))
		}
	}

	if rec.MaxTokens != 512 {
		t.Errorf("expected max_tokens to pass through as 512, got %d", rec.MaxTokens)
	}
}

// TestWire_ImageInputPassthrough verifies that multipart content containing
// image_url parts (data URLs and http URLs) translates cleanly through the IR
// and hublane bridge to the upstream request.
func TestWire_ImageInputPassthrough(t *testing.T) {
	h, _ := newWireHarness(t, "vision")

	dataURL := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	httpURL := "https://example.com/photo.jpg"

	rec := postChatAndRecord(t, h, map[string]any{
		"model": "vision/qwen-vl",
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "Describe these images"},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURL, "detail": "high"}},
					map[string]any{"type": "image_url", "image_url": httpURL},
				},
			},
		},
	})

	if len(rec.Messages) != 1 {
		t.Fatalf("expected 1 upstream message, got %d", len(rec.Messages))
	}

	msg := rec.Messages[0]
	if msg.Role != "user" {
		t.Errorf("expected role 'user', got %q", msg.Role)
	}

	// In upstream JSON, content should be an array of parts
	parts, ok := msg.Content.([]any)
	if !ok {
		t.Fatalf("expected []any content for multimodal message, got %T: %v", msg.Content, msg.Content)
	}
	if len(parts) != 3 {
		t.Fatalf("expected 3 content parts, got %d: %v", len(parts), parts)
	}

	// Part 0: text
	p0, ok := parts[0].(map[string]any)
	if !ok || p0["type"] != "text" || p0["text"] != "Describe these images" {
		t.Errorf("unexpected part 0: %v", parts[0])
	}

	// Part 1: image_url (data URL, detail high)
	p1, ok := parts[1].(map[string]any)
	if !ok || p1["type"] != "image_url" {
		t.Fatalf("unexpected part 1 type: %v", parts[1])
	}
	iu1, ok := p1["image_url"].(map[string]any)
	if !ok || iu1["url"] != dataURL || iu1["detail"] != "high" {
		t.Errorf("unexpected part 1 image_url: %v", p1["image_url"])
	}

	// Part 2: image_url (http URL)
	p2, ok := parts[2].(map[string]any)
	if !ok || p2["type"] != "image_url" {
		t.Fatalf("unexpected part 2 type: %v", parts[2])
	}
	iu2, ok := p2["image_url"].(map[string]any)
	if !ok || iu2["url"] != httpURL {
		t.Errorf("unexpected part 2 image_url: %v", p2["image_url"])
	}
}

