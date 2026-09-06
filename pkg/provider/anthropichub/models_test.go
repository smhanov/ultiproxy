package anthropichub

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/provider"
)

// newFakeAnthropic serves the Anthropic /v1/models wire and records requests.
type fakeAnthropic struct {
	srv    *httptest.Server
	models int32
	// lastDump is the raw request dump of the most recent /v1/models call, so
	// tests can assert the anthropic-version header travelled with it.
	lastDump string
}

func newFakeAnthropic(t *testing.T, rows []map[string]any) *fakeAnthropic {
	t.Helper()
	f := &fakeAnthropic{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			dump, _ := httputil.DumpRequest(r, false)
			f.lastDump = string(dump)
			atomic.AddInt32(&f.models, 1)
			_ = json.NewEncoder(w).Encode(map[string]any{"data": rows, "has_more": false})
		case "/v1/messages":
			_ = json.NewEncoder(w).Encode(map[string]any{"id": "msg_1", "content": []map[string]any{{"type": "text", "text": "ok"}}})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeAnthropic) requests() int32 { return atomic.LoadInt32(&f.models) }

func TestFetchModels_AnthropicModelList(t *testing.T) {
	f := newFakeAnthropic(t, []map[string]any{
		{
			"id":               "claude-sonnet-4-6",
			"max_input_tokens": 1000000,
			"max_tokens":       64000,
			"capabilities": map[string]any{
				"image_input": map[string]any{"supported": true},
			},
		},
		{"id": "claude-haiku-4-5", "max_input_tokens": 200000, "max_tokens": 32000},
	})
	p, err := New(Config{BaseURL: f.srv.URL, APIKey: "sk-ant-test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := p.FetchModels(context.Background()); err != nil {
		t.Fatalf("FetchModels: %v", err)
	}
	if !strings.Contains(strings.ToLower(f.lastDump), "anthropic-version: 2023-06-01") {
		t.Errorf("request missing the anthropic-version header:\n%s", f.lastDump)
	}
	if !strings.Contains(strings.ToLower(f.lastDump), "x-api-key: sk-ant-test") {
		t.Errorf("request missing the x-api-key header:\n%s", f.lastDump)
	}

	info := p.CachedModelInfo()
	if len(info) != 2 {
		t.Fatalf("cached model info = %+v, want 2 rows", info)
	}
	var sonnet, haiku provider.ModelInfo
	for _, m := range info {
		switch m.ID {
		case "claude-sonnet-4-6":
			sonnet = m
		case "claude-haiku-4-5":
			haiku = m
		}
	}
	if sonnet.ID == "" || haiku.ID == "" {
		t.Fatalf("cached rows = %+v", info)
	}
	if sonnet.ContextLength != 1000000 {
		t.Errorf("context length = %d, want 1000000 (max_input_tokens)", sonnet.ContextLength)
	}
	if sonnet.MaxOutput != 64000 {
		t.Errorf("max output = %d, want 64000 (max_tokens)", sonnet.MaxOutput)
	}
	if !sonnet.SupportsImageInput() {
		t.Errorf("input modalities = %v, want image input", sonnet.InputModalities)
	}
	if haiku.SupportsImageInput() {
		t.Errorf("haiku advertised image input it did not report: %v", haiku.InputModalities)
	}
	if got := p.CachedModels(); len(got) != 2 {
		t.Errorf("cached models = %v", got)
	}
}

// A failed fetch invents nothing and keeps whatever was cached before.
func TestFetchModels_UpstreamErrorInventsNothing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)
	p, err := New(Config{BaseURL: srv.URL, APIKey: "sk-ant-test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := p.FetchModels(context.Background()); err == nil {
		t.Fatal("FetchModels on a 500 upstream returned no error")
	}
	if got := p.CachedModelInfo(); len(got) != 0 {
		t.Errorf("failed fetch left model info: %+v", got)
	}
	if got := p.CachedModels(); len(got) != 0 {
		t.Errorf("failed fetch left models: %v", got)
	}
}

// The registry bundle exposes the discovery surfaces the server's listing and
// discovery loop type-assert: the lane itself, not just the hublane adapter.
func TestProviderBundleExposesModelDiscovery(t *testing.T) {
	f := newFakeAnthropic(t, []map[string]any{{"id": "claude-sonnet-4-6", "max_input_tokens": 1000}})
	p, err := New(Config{BaseURL: f.srv.URL, APIKey: "sk-ant-test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	bundle := p.Provider()
	if bundle.Inference == nil {
		t.Fatal("bundle has no inference provider")
	}
	if _, ok := bundle.Inference.(interface {
		FetchModels(ctx context.Context) ([]string, error)
	}); !ok {
		t.Fatal("registered lane does not implement FetchModels")
	}
	if _, ok := bundle.Inference.(interface{ CachedModelInfo() []provider.ModelInfo }); !ok {
		t.Fatal("registered lane does not implement CachedModelInfo")
	}
	if !p.ModelDiscoveryEnabled() {
		t.Error("keyed anthropic lane is not discoverable")
	}
	if bundle.Capabilities.Chat != true {
		t.Errorf("bundle capabilities = %+v", bundle.Capabilities)
	}
}
