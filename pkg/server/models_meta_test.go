package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
	"github.com/smhanov/ultiproxy/pkg/provider/anthropichub"
	"github.com/smhanov/ultiproxy/pkg/provider/openaicompat"
	"github.com/smhanov/ultiproxy/pkg/state"
)

// rawByID returns the raw /v1/models object for one id, so tests can assert
// which keys are absent, not just which values are present.
func rawByID(t *testing.T, rows []map[string]any, id string) map[string]any {
	t.Helper()
	for _, m := range rows {
		if m["id"] == id {
			return m
		}
	}
	t.Fatalf("missing %q in %v", id, rows)
	return nil
}

func wantInt(t *testing.T, row map[string]any, key string, want float64) {
	t.Helper()
	got, ok := row[key]
	if !ok {
		t.Errorf("%s: %s missing (row = %v)", row["id"], key, row)
		return
	}
	if got != want {
		t.Errorf("%s: %s = %v, want %v", row["id"], key, got, want)
	}
}

func wantAbsent(t *testing.T, row map[string]any, keys ...string) {
	t.Helper()
	for _, k := range keys {
		if _, ok := row[k]; ok {
			t.Errorf("%s: %q must be omitted when unknown, got %v", row["id"], k, row[k])
		}
	}
}

func wantStrings(t *testing.T, row map[string]any, path string, want []string) {
	t.Helper()
	arch, _ := row["architecture"].(map[string]any)
	if arch == nil {
		t.Fatalf("%s: no architecture object (row = %v)", row["id"], row)
	}
	raw, ok := arch[path].([]any)
	if !ok {
		t.Fatalf("%s: architecture.%s missing (row = %v)", row["id"], path, row)
	}
	got := make([]string, 0, len(raw))
	for _, v := range raw {
		s, _ := v.(string)
		got = append(got, s)
	}
	if len(got) != len(want) {
		t.Fatalf("%s: architecture.%s = %v, want %v", row["id"], path, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("%s: architecture.%s = %v, want %v", row["id"], path, got, want)
		}
	}
}

// AC1: an OpenRouter-shaped row lists with both window names, the output cap,
// the modality arrays and supports_vision, and never a legacy max_tokens key.
func TestHandleModels_TopProviderWindowOutputCapAndModalities(t *testing.T) {
	dir := t.TempDir()
	upstream := newFakeModelUpstreamRows(t, []map[string]any{{
		"id": "gpt-4o",
		"top_provider": map[string]any{
			"context_length":        128000,
			"max_completion_tokens": 16384,
		},
		"architecture": map[string]any{
			"input_modalities":  []string{"text", "image"},
			"output_modalities": []string{"text"},
		},
	}})
	srv := registerDiscoveredLane(t, dir, "openai", upstream)

	row := rawByID(t, getModelsRaw(t, srv), "openai/gpt-4o")
	wantInt(t, row, "context_length", 128000)
	wantInt(t, row, "max_model_len", 128000)
	wantInt(t, row, "max_output_tokens", 16384)
	wantStrings(t, row, "input_modalities", []string{"text", "image"})
	wantStrings(t, row, "output_modalities", []string{"text"})
	if row["supports_vision"] != true {
		t.Errorf("supports_vision = %v, want true", row["supports_vision"])
	}
	wantAbsent(t, row, "max_tokens")
}

// AC2: the served llama.cpp window (meta.n_ctx) beats the trained one, and a
// row without window or modality fields omits every one of those keys.
func TestHandleModels_ServedWindowWinsAndUnknownStaysOmitted(t *testing.T) {
	dir := t.TempDir()
	upstream := newFakeModelUpstreamRows(t, []map[string]any{
		{"id": "local", "meta": map[string]any{"n_ctx": 8192, "n_ctx_train": 131072}, "context_window": 0},
		{"id": "bare"},
	})
	srv := registerDiscoveredLane(t, dir, "lane", upstream)
	raw := getModelsRaw(t, srv)

	local := rawByID(t, raw, "lane/local")
	wantInt(t, local, "context_length", 8192)
	wantInt(t, local, "max_model_len", 8192)

	bare := rawByID(t, raw, "lane/bare")
	wantAbsent(t, bare, "context_length", "max_model_len", "max_output_tokens", "architecture", "supports_vision", "max_tokens")
}

// AC3: a Groq-style context_window fills both window names, and an HF-style
// architecture.modality string becomes normalized modality arrays.
func TestHandleModels_GroqWindowAndModalityString(t *testing.T) {
	dir := t.TempDir()
	upstream := newFakeModelUpstreamRows(t, []map[string]any{
		{"id": "llama", "context_window": 131072},
		{"id": "x", "architecture": map[string]any{"modality": "text+image+file->text"}},
	})
	srv := registerDiscoveredLane(t, dir, "groq", upstream)
	raw := getModelsRaw(t, srv)

	llama := rawByID(t, raw, "groq/llama")
	wantInt(t, llama, "context_length", 131072)
	wantInt(t, llama, "max_model_len", 131072)

	x := rawByID(t, raw, "groq/x")
	wantStrings(t, x, "input_modalities", []string{"text", "image", "file"})
	wantStrings(t, x, "output_modalities", []string{"text"})
	if x["supports_vision"] != true {
		t.Errorf("supports_vision = %v, want true", x["supports_vision"])
	}
}

// newFakeAnthropicUpstream serves the Anthropic /v1/models wire and counts the
// requests, so a test can prove listing does not dial it.
type fakeAnthropicUpstream struct {
	srv  *httptest.Server
	hits *int32
}

func (f *fakeAnthropicUpstream) modelRequests() int32 { return atomic.LoadInt32(f.hits) }

func newFakeAnthropicUpstream(t *testing.T, rows []map[string]any) *fakeAnthropicUpstream {
	t.Helper()
	hits := new(int32)
	counter := hits
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		atomic.AddInt32(counter, 1)
		_ = json.NewEncoder(w).Encode(map[string]any{"data": rows})
	}))
	t.Cleanup(srv.Close)
	return &fakeAnthropicUpstream{srv: srv, hits: hits}
}

// AC4: Anthropic discovery fills the listed anthropic/<id> rows, and listing
// reads the cache only (the fake's request count does not move).
func TestHandleModels_AnthropicWindowsAndModalitiesWithoutFanout(t *testing.T) {
	upstream := newFakeAnthropicUpstream(t, []map[string]any{{
		"id":               "claude-sonnet-4-6",
		"max_input_tokens": 1000000,
		"max_tokens":       64000,
		"capabilities": map[string]any{
			"image_input": map[string]any{"supported": true},
		},
	}})

	lane, err := anthropichub.New(anthropichub.Config{BaseURL: upstream.srv.URL, APIKey: "sk-ant-test"})
	if err != nil {
		t.Fatalf("anthropic lane: %v", err)
	}
	if _, err := lane.FetchModels(context.Background()); err != nil {
		t.Fatalf("anthropic discovery: %v", err)
	}

	registry := provider.NewRegistry()
	registry.Register(lane.Provider())

	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()
	srv := NewServer(cfg, registry, WithStateManager(state.NewStateManager()))

	before := upstream.modelRequests()
	raw := getModelsRaw(t, srv)
	if got := upstream.modelRequests(); got != before {
		t.Errorf("GET /v1/models dialed the anthropic upstream: %d -> %d", before, got)
	}

	row := rawByID(t, raw, "anthropic/claude-sonnet-4-6")
	wantInt(t, row, "context_length", 1000000)
	wantInt(t, row, "max_model_len", 1000000)
	wantInt(t, row, "max_output_tokens", 64000)
	wantStrings(t, row, "input_modalities", []string{"text", "image"})
	if row["supports_vision"] != true {
		t.Errorf("supports_vision = %v, want true", row["supports_vision"])
	}
}

// AC5: the cited static catalog fills ids whose discovery reports no window,
// never replaces a served window, and operator alias modalities win over a
// catalog that claims image input.
func TestHandleModels_CatalogAndAliasPrecedence(t *testing.T) {
	dir := t.TempDir()
	overlay := `[
	  {"id":"glm-5.3","context_length":1000000,"input_modalities":["text","image"],"source":"https://docs.z.ai/guides/coding-plan"},
	  {"id":"Qwen/Qwen3","context_length":999999,"source":"https://operator.example/models"}
	]`
	if err := os.WriteFile(filepath.Join(dir, "windows.json"), []byte(overlay), 0o600); err != nil {
		t.Fatalf("write overlay: %v", err)
	}

	upstream := newFakeModelUpstreamRows(t, []map[string]any{
		{"id": "glm-5.3"},
		{"id": "Qwen/Qwen3", "max_model_len": 262144},
	})
	cfg := DefaultConfig()
	cfg.DataDir = dir
	cfg.Server.Models = map[string]ModelAlias{
		// The alias asserts text-only input: that must beat the catalog row
		// above, which claims image.
		"glm-5.3": {Provider: "zai", Upstream: "glm-5.3", InputModalities: []string{"text"}},
	}
	store := NewRuntimeProviderStore(filepath.Join(dir, "providers.json"))
	store.DefaultDataDir = dir
	if err := store.Add(openaicompat.Config{
		Name:    "zai",
		BaseURL: upstream.srv.URL,
		Quirks:  openaicompat.Quirks{ModelListPassthrough: true},
	}); err != nil {
		t.Fatalf("add lane: %v", err)
	}
	srv := NewServer(cfg, provider.NewRegistry(),
		WithStateManager(state.NewStateManager()),
		WithRuntimeProviderStore(store),
	)

	raw := getModelsRaw(t, srv)

	alias := rawByID(t, raw, "glm-5.3")
	wantInt(t, alias, "context_length", 1000000)
	wantInt(t, alias, "max_model_len", 1000000)
	wantStrings(t, alias, "input_modalities", []string{"text"})
	wantAbsent(t, alias, "supports_vision")

	discovered := rawByID(t, raw, "zai/glm-5.3")
	wantInt(t, discovered, "context_length", 1000000)

	// A served vLLM window is never replaced by a catalog row.
	vllm := rawByID(t, raw, "zai/Qwen/Qwen3")
	wantInt(t, vllm, "context_length", 262144)
	wantInt(t, vllm, "max_model_len", 262144)
}

// The compiled catalog (no overlay file) already fills the z.ai coding-plan
// ids, whose own model list reports no window.
func TestHandleModels_CompiledCatalogFillsWindow(t *testing.T) {
	dir := t.TempDir()
	upstream := newFakeModelUpstreamRows(t, []map[string]any{{"id": "glm-5.3"}})
	srv := registerDiscoveredLane(t, dir, "zai", upstream)
	row := rawByID(t, getModelsRaw(t, srv), "zai/glm-5.3")
	wantInt(t, row, "context_length", 1000000)
	wantInt(t, row, "max_model_len", 1000000)
}

// An operator overlay that does not validate (uncited numbers) must not take
// the whole listing down: the compiled catalog is served instead.
func TestHandleModels_InvalidOverlayFallsBackToCompiledCatalog(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "windows.json"), []byte(`[{"id":"guessed","context_length":999999}]`), 0o600); err != nil {
		t.Fatalf("write overlay: %v", err)
	}
	upstream := newFakeModelUpstreamRows(t, []map[string]any{{"id": "glm-5.3"}})
	srv := registerDiscoveredLane(t, dir, "zai", upstream)
	row := rawByID(t, getModelsRaw(t, srv), "zai/glm-5.3")
	wantInt(t, row, "context_length", 1000000)
	guessed := getModelsRaw(t, srv)
	for _, m := range guessed {
		if m["id"] == "zai/guessed" {
			t.Error("uncited overlay row was served")
		}
	}
}
