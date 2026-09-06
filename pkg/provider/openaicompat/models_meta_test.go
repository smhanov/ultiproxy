package openaicompat

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

// newMetaUpstream serves one /v1/models document and counts the requests, so
// tests can both drive discovery and prove the cache is what is read.
func newMetaUpstream(t *testing.T, rows []map[string]any) (*httptest.Server, *int32) {
	t.Helper()
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		hits++
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": rows})
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func mustProvider(t *testing.T, rows []map[string]any) (*Provider, *int32) {
	t.Helper()
	srv, hits := newMetaUpstream(t, rows)
	p, err := New(Config{Name: "lane", BaseURL: srv.URL, APIKey: "sk-test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p, hits
}

func infoFor(t *testing.T, p *Provider, id string) (ctxLen, maxOut int, in, out []string) {
	t.Helper()
	for _, m := range p.CachedModelInfo() {
		if m.ID == id {
			return m.ContextLength, m.MaxOutput, m.InputModalities, m.OutputModalities
		}
	}
	t.Fatalf("no cached model info for %q (have %+v)", id, p.CachedModelInfo())
	return
}

// AC1 (provider half): OpenRouter-style top_provider.context_length +
// max_completion_tokens and architecture modality arrays all land in the cache.
func TestFetchModels_TopProviderWindowOutputCapAndModalities(t *testing.T) {
	p, _ := mustProvider(t, []map[string]any{{
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
	ctxLen, maxOut, in, out := infoFor(t, p, "gpt-4o")
	if ctxLen != 128000 {
		t.Errorf("context length = %d, want 128000", ctxLen)
	}
	if maxOut != 16384 {
		t.Errorf("max output = %d, want 16384", maxOut)
	}
	if !reflect.DeepEqual(in, []string{"text", "image"}) {
		t.Errorf("input modalities = %v, want [text image]", in)
	}
	if !reflect.DeepEqual(out, []string{"text"}) {
		t.Errorf("output modalities = %v, want [text]", out)
	}
}

// AC2 (provider half): llama.cpp served window (meta.n_ctx) beats the trained
// window (meta.n_ctx_train); a zero context_window is not a window.
func TestFetchModels_ServedNCtxBeatsTrainedAndZero(t *testing.T) {
	p, _ := mustProvider(t, []map[string]any{
		{"id": "local", "meta": map[string]any{"n_ctx": 8192, "n_ctx_train": 131072}, "context_window": 0},
	})
	if ctxLen, _, _, _ := infoFor(t, p, "local"); ctxLen != 8192 {
		t.Errorf("context length = %d, want 8192 (served n_ctx wins)", ctxLen)
	}
}

// AC3 (provider half): Groq-style context_window, and HF-style
// architecture.modality "text+image+file->text".
func TestFetchModels_GroqContextWindow(t *testing.T) {
	p, _ := mustProvider(t, []map[string]any{{"id": "llama", "context_window": 131072}})
	if ctxLen, _, _, _ := infoFor(t, p, "llama"); ctxLen != 131072 {
		t.Errorf("context length = %d, want 131072", ctxLen)
	}
}

func TestFetchModels_ArchitectureModalityString(t *testing.T) {
	p, _ := mustProvider(t, []map[string]any{{
		"id":           "x",
		"architecture": map[string]any{"modality": "text+image+file->text"},
	}})
	_, _, in, out := infoFor(t, p, "x")
	if !reflect.DeepEqual(in, []string{"text", "image", "file"}) {
		t.Errorf("input modalities = %v, want [text image file]", in)
	}
	if !reflect.DeepEqual(out, []string{"text"}) {
		t.Errorf("output modalities = %v, want [text]", out)
	}
}

// FR-02: supports_vision without arrays still advertises image input.
func TestFetchModels_SupportsVisionWithoutArrays(t *testing.T) {
	p, _ := mustProvider(t, []map[string]any{{"id": "visionary", "supports_vision": true}})
	_, _, in, _ := infoFor(t, p, "visionary")
	if !reflect.DeepEqual(in, []string{"image"}) {
		t.Errorf("input modalities = %v, want [image]", in)
	}
}

// FR-02: pdf normalizes to file; unknown tokens are dropped, not invented.
func TestFetchModels_PDFNormalizesToFile(t *testing.T) {
	p, _ := mustProvider(t, []map[string]any{{
		"id": "reader",
		"architecture": map[string]any{
			"input_modalities":  []string{"Text", "PDF", "image", "", "telepathy"},
			"output_modalities": []string{"text"},
		},
	}})
	_, _, in, _ := infoFor(t, p, "reader")
	if !reflect.DeepEqual(in, []string{"text", "file", "image"}) {
		t.Errorf("input modalities = %v, want [text file image]", in)
	}
}

// FR-01: output cap shapes other than top_provider.max_completion_tokens.
func TestFetchModels_OutputCapShapes(t *testing.T) {
	p, _ := mustProvider(t, []map[string]any{
		{"id": "a", "max_output_tokens": 4096},
		{"id": "b", "max_completion_tokens": 8192},
		{"id": "c", "max_output_tokens": 1000, "top_provider": map[string]any{"max_completion_tokens": 2000}},
	})
	for id, want := range map[string]int{"a": 4096, "b": 8192, "c": 2000} {
		if _, got, _, _ := infoFor(t, p, id); got != want {
			t.Errorf("%s max output = %d, want %d", id, got, want)
		}
	}
}

// AC2 / T032 honesty: a row with no window or modality fields caches unknowns,
// never 0-as-a-window or empty arrays standing in for "unknown".
func TestFetchModels_RowWithoutMetadataStaysUnknown(t *testing.T) {
	p, _ := mustProvider(t, []map[string]any{{"id": "bare"}})
	info := p.CachedModelInfo()
	if len(info) != 1 || info[0].ID != "bare" {
		t.Fatalf("cached info = %+v", info)
	}
	if info[0].ContextLength != 0 || info[0].MaxOutput != 0 {
		t.Errorf("windows = %d/%d, want 0/0 (unknown)", info[0].ContextLength, info[0].MaxOutput)
	}
	if info[0].InputModalities != nil || info[0].OutputModalities != nil {
		t.Errorf("modalities = %v/%v, want nil/nil (unknown)", info[0].InputModalities, info[0].OutputModalities)
	}
}

// Window precedence across every shape FR-01 names.
func TestFetchModels_WindowPrecedence(t *testing.T) {
	p, _ := mustProvider(t, []map[string]any{
		{"id": "maxmodel", "max_model_len": 100, "context_length": 200, "context_window": 300},
		{"id": "ctxlen", "context_length": 200, "context_window": 300, "meta": map[string]any{"n_ctx": 400}},
		{"id": "topprov", "context_window": 300, "top_provider": map[string]any{"context_length": 500}},
		{"id": "metactx", "context_window": 300, "meta": map[string]any{"context_length": 600}},
		{"id": "nctx", "context_window": 300, "meta": map[string]any{"n_ctx": 700}},
		{"id": "ctxwin", "context_window": 300, "meta": map[string]any{"n_ctx_train": 800}},
		{"id": "nctxtrain", "meta": map[string]any{"n_ctx_train": 900}},
	})
	for id, want := range map[string]int{
		"maxmodel": 100, "ctxlen": 200, "topprov": 500, "metactx": 600,
		"nctx": 700, "ctxwin": 300, "nctxtrain": 900,
	} {
		if got, _, _, _ := infoFor(t, p, id); got != want {
			t.Errorf("%s context length = %d, want %d", id, got, want)
		}
	}
}

// Explicit FetchModels refreshes the cache (refresh_models path).
func TestFetchModels_RefreshesCache(t *testing.T) {
	p, hits := mustProvider(t, []map[string]any{{"id": "one"}})
	if _, err := p.FetchModels(context.Background()); err != nil {
		t.Fatalf("FetchModels: %v", err)
	}
	if *hits != 2 {
		t.Errorf("upstream hits = %d, want 2 (construction + explicit fetch)", *hits)
	}
}
