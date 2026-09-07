package opencode

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/server"
)

func encodeJSON(w http.ResponseWriter, v any) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(v)
}

func decodeJSONBody(resp *http.Response, v any) error {
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// containsUnknownModel reports whether an error body carries the 404
// unknown_model contract type.
func containsUnknownModel(body map[string]any) bool {
	errObj, ok := body["error"].(map[string]any)
	if !ok {
		return false
	}
	return errObj["type"] == "unknown_model"
}

// T041: /v1/models advertises only canonical "<lane>/<model>" ids and the
// request path accepts exactly (a) canonical ids and (b) user aliases.
//
// The fixture wires two real OpenAI-compatible lanes (zai -> upstream A,
// vllm -> upstream B) that each discover their own model list, plus two user
// aliases. Each lane has its own FakeUpstream, so a chat request reaching a
// lane is observable as a /v1/chat/completions hit on that lane's upstream.

// modelListUpstream serves /v1/models with the given ids and /v1/chat/completions
// with a standard completion, so an openaicompat lane can discover and be chatted
// against.
func modelListUpstream(t *testing.T, modelIDs ...string) *FakeUpstream {
	t.Helper()
	fake := NewFakeUpstream()
	fake.SetDefaultResponse(ResponseScriptFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			rows := make([]map[string]any, 0, len(modelIDs))
			for _, id := range modelIDs {
				rows = append(rows, map[string]any{"id": id})
			}
			_ = encodeJSON(w, map[string]any{"object": "list", "data": rows})
		default:
			_ = encodeJSON(w, map[string]any{
				"id": "chatcmpl-t041", "object": "chat.completion",
				"choices": []map[string]any{{
					"index":         0,
					"message":       map[string]any{"role": "assistant", "content": "ok"},
					"finish_reason": "stop",
				}},
			})
		}
	}))
	return fake
}

// chatRequests returns the recorded requests whose path is the chat endpoint.
func chatRequests(fake *FakeUpstream) []*RecordedRequest {
	out := []*RecordedRequest{}
	for _, r := range fake.Requests() {
		if r.Path == "/v1/chat/completions" {
			out = append(out, r)
		}
	}
	return out
}

// newPrefixedIDsHarness builds the two-lane + alias fixture. Both lanes are
// real openaicompat providers (so their discovery cache is visible to
// /v1/models); each has its own FakeUpstream so a chat reaching a lane is
// observable as a chat request on that lane's upstream. The harness's default
// "opencode" lane points at an empty fake and advertises nothing.
func newPrefixedIDsHarness(t *testing.T) (*Harness, *FakeUpstream, *FakeUpstream) {
	t.Helper()

	fakeZai := modelListUpstream(t, "glm-5.3-flash", "shared-model")
	fakeVLLM := modelListUpstream(t, "shared-model")
	emptyFake := NewFakeUpstream() // default "opencode" lane: no discovery, no ads

	pZai, err := openaicompat.New(openaicompat.Config{
		Name:       "zai",
		BaseURL:    fakeZai.URL(),
		APIKey:     "test-zai-key",
		HTTPClient: fakeZai.Client(),
		Quirks:     openaicompat.Quirks{ModelListPassthrough: true},
	})
	if err != nil {
		t.Fatalf("openaicompat.New(zai): %v", err)
	}
	pVLLM, err := openaicompat.New(openaicompat.Config{
		Name:       "vllm",
		BaseURL:    fakeVLLM.URL(),
		APIKey:     "test-vllm-key",
		HTTPClient: fakeVLLM.Client(),
		Quirks:     openaicompat.Quirks{ModelListPassthrough: true},
	})
	if err != nil {
		t.Fatalf("openaicompat.New(vllm): %v", err)
	}

	h := NewTestHarness(t,
		WithFakeUpstream(emptyFake),
		WithProvider(pZai.Provider()),
		WithProvider(pVLLM.Provider()),
		WithModelAlias("qwenpoint-3.8", server.ModelAlias{Provider: "zai", Upstream: "glm-5.3-flash"}),
		WithModelAlias("secret-thing", server.ModelAlias{Provider: "zai", Upstream: "secret-model"}),
	)
	return h, fakeZai, fakeVLLM
}

// getModels fetches GET /v1/models through the harness and returns the id set.
func getModelsIDSet(t *testing.T, h *Harness) map[string]bool {
	t.Helper()
	resp, err := h.Client().Get(h.URL() + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/models: %d", resp.StatusCode)
	}
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := decodeJSONBody(resp, &body); err != nil {
		t.Fatalf("decode /v1/models: %v", err)
	}
	set := make(map[string]bool, len(body.Data))
	for _, e := range body.Data {
		set[e.ID] = true
	}
	return set
}

// AC1: every advertised id is canonical "<lane>/<model>" whose lane segment
// names a registered lane; no bare alias or bare lane ids appear.
func TestPrefixedIDs_AdvrtisesOnlyCanonicalIDs(t *testing.T) {
	t.Parallel()
	h, _, _ := newPrefixedIDsHarness(t)
	defer h.Close()

	ids := getModelsIDSet(t, h)

	// The advertised set is exactly the canonical "<lane>/<model>" ids: the
	// lanes' discovered ids plus each user alias's canonical target
	// (qwenpoint-3.8 -> zai/glm-5.3-flash, already discovered; secret-thing ->
	// zai/secret-model). Bare alias names and bare lane names are never
	// advertised; every id's lane segment names a registered lane.
	want := map[string]bool{
		"zai/glm-5.3-flash": true, // discovered + alias qwenpoint-3.8 target
		"zai/shared-model":  true, // discovered (cross-lane duplicate)
		"vllm/shared-model": true, // discovered (cross-lane duplicate)
		"zai/secret-model":  true, // alias secret-thing canonical target
	}
	if len(ids) != len(want) {
		t.Errorf("advertised id set = %v, want exactly %v", ids, want)
	}
	for id := range want {
		if !ids[id] {
			t.Errorf("canonical id %q missing from /v1/models: %v", id, ids)
		}
	}
	for id := range ids {
		if !want[id] {
			t.Errorf("unexpected advertised id %q (want only %v): %v", id, want, ids)
		}
	}

	// Bare alias and bare lane ids must be gone.
	for _, bare := range []string{"qwenpoint-3.8", "secret-thing", "zai", "vllm"} {
		if ids[bare] {
			t.Errorf("bare id %q still advertised: %v", bare, ids)
		}
	}

	// Every id is canonical: "<registered-lane>/<...>".
	registry := map[string]bool{"zai": true, "vllm": true}
	for id := range ids {
		idx := strings.Index(id, "/")
		if idx <= 0 {
			t.Errorf("id %q is not canonical (no lane segment): %v", id, ids)
			continue
		}
		if !registry[id[:idx]] {
			t.Errorf("id %q names unregistered lane %q: %v", id, id[:idx], ids)
		}
	}
}

// AC2: a canonical id routes to its own lane's upstream with the prefix
// stripped, and the other lane's upstream never sees it.
func TestPrefixedIDs_PrefixedIDRoutesToItsLane(t *testing.T) {
	t.Parallel()
	h, fakeZai, fakeVLLM := newPrefixedIDsHarness(t)
	defer h.Close()

	resp, body, err := h.PostChat(context.Background(), map[string]any{
		"model":    "zai/glm-5.3-flash",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if err != nil {
		t.Fatalf("PostChat: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d: %v", resp.StatusCode, body)
	}

	zaiChats := chatRequests(fakeZai)
	if len(zaiChats) != 1 {
		t.Fatalf("zai upstream saw %d chat requests, want 1", len(zaiChats))
	}
	if got := zaiChats[0].Model; got != "glm-5.3-flash" {
		t.Errorf("zai upstream model = %q, want prefix-stripped %q", got, "glm-5.3-flash")
	}
	if got := len(chatRequests(fakeVLLM)); got != 0 {
		t.Errorf("vllm upstream saw %d chat requests, want 0 (no cross-lane routing)", got)
	}
}

// AC3: the same upstream model name on two lanes yields two distinct
// advertised ids, and each routes to its own lane.
func TestPrefixedIDs_CrossLaneDuplicatesStayDistinct(t *testing.T) {
	t.Parallel()
	h, fakeZai, fakeVLLM := newPrefixedIDsHarness(t)
	defer h.Close()

	ids := getModelsIDSet(t, h)
	if !ids["zai/shared-model"] || !ids["vllm/shared-model"] {
		t.Fatalf("both cross-lane duplicates must be advertised distinctly, got %v", ids)
	}

	// zai/shared-model -> zai upstream only.
	if _, _, err := h.PostChat(context.Background(), map[string]any{
		"model": "zai/shared-model", "messages": []map[string]any{{"role": "user", "content": "hi"}},
	}); err != nil {
		t.Fatalf("PostChat zai/shared-model: %v", err)
	} else if len(chatRequests(fakeZai)) != 1 || len(chatRequests(fakeVLLM)) != 0 {
		t.Errorf("zai/shared-model misrouted: zai=%d vllm=%d", len(chatRequests(fakeZai)), len(chatRequests(fakeVLLM)))
	} else if got := chatRequests(fakeZai)[0].Model; got != "shared-model" {
		t.Errorf("zai upstream model = %q, want shared-model", got)
	}

	fakeZai.ResetRequests()
	fakeVLLM.ResetRequests()

	// vllm/shared-model -> vllm upstream only.
	if _, _, err := h.PostChat(context.Background(), map[string]any{
		"model": "vllm/shared-model", "messages": []map[string]any{{"role": "user", "content": "hi"}},
	}); err != nil {
		t.Fatalf("PostChat vllm/shared-model: %v", err)
	} else if len(chatRequests(fakeVLLM)) != 1 || len(chatRequests(fakeZai)) != 0 {
		t.Errorf("vllm/shared-model misrouted: vllm=%d zai=%d", len(chatRequests(fakeVLLM)), len(chatRequests(fakeZai)))
	} else if got := chatRequests(fakeVLLM)[0].Model; got != "shared-model" {
		t.Errorf("vllm upstream model = %q, want shared-model", got)
	}
}

// AC4: a user alias completes a chat and routes to the aliased lane/model with
// the alias upstream id; an unprefixed id that is not a user alias answers
// 404 unknown_model and reaches no lane.
func TestPrefixedIDs_AliasRoutesAndBareNonAlias404(t *testing.T) {
	t.Parallel()
	h, fakeZai, fakeVLLM := newPrefixedIDsHarness(t)
	defer h.Close()

	// Alias routes to the aliased lane/model.
	resp, body, err := h.PostChat(context.Background(), map[string]any{
		"model":    "qwenpoint-3.8",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if err != nil {
		t.Fatalf("PostChat alias: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("alias chat: expected 200, got %d: %v", resp.StatusCode, body)
	}
	zaiChats := chatRequests(fakeZai)
	if len(zaiChats) != 1 {
		t.Fatalf("alias: zai upstream saw %d chats, want 1", len(zaiChats))
	}
	if got := zaiChats[0].Model; got != "glm-5.3-flash" {
		t.Errorf("alias upstream model = %q, want the alias target glm-5.3-flash", got)
	}
	if got := len(chatRequests(fakeVLLM)); got != 0 {
		t.Errorf("alias: vllm upstream saw %d chats, want 0", got)
	}

	fakeZai.ResetRequests()
	fakeVLLM.ResetRequests()

	// Unprefixed id that is not a user alias -> 404 unknown_model, no lane hit.
	resp, body, err = h.PostChat(context.Background(), map[string]any{
		"model":    "not-an-alias",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if err != nil {
		t.Fatalf("PostChat bare non-alias: %v", err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("bare non-alias: expected 404, got %d: %v", resp.StatusCode, body)
	}
	if !containsUnknownModel(body) {
		t.Errorf("bare non-alias: expected unknown_model error, got %v", body)
	}
	if got := len(chatRequests(fakeZai)) + len(chatRequests(fakeVLLM)); got != 0 {
		t.Errorf("bare non-alias reached a lane: %d chats", got)
	}
}

// A bare lane name (no slash, not an alias) is a routing prefix, not a model:
// it must answer 404 unknown_model and reach no lane.
func TestPrefixedIDs_BareLaneNameIsNotAModel(t *testing.T) {
	t.Parallel()
	h, fakeZai, fakeVLLM := newPrefixedIDsHarness(t)
	defer h.Close()

	resp, body, err := h.PostChat(context.Background(), map[string]any{
		"model":    "zai",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if err != nil {
		t.Fatalf("PostChat bare lane name: %v", err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("bare lane name: expected 404, got %d: %v", resp.StatusCode, body)
	}
	if !containsUnknownModel(body) {
		t.Errorf("bare lane name: expected unknown_model error, got %v", body)
	}
	if got := len(chatRequests(fakeZai)) + len(chatRequests(fakeVLLM)); got != 0 {
		t.Errorf("bare lane name reached a lane: %d chats", got)
	}
}
