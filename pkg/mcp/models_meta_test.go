package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// metaLane is a registered lane with a discovery cache carrying the new
// metadata fields.
type metaLane struct {
	name string
	info []provider.ModelInfo
}

func (l *metaLane) Name() string { return l.name }
func (l *metaLane) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	return &ir.Response{}, nil
}
func (l *metaLane) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	ch := make(chan ir.Event, 1)
	close(ch)
	return ch, nil
}
func (l *metaLane) CachedModels() []string {
	out := make([]string, 0, len(l.info))
	for _, m := range l.info {
		out = append(out, m.ID)
	}
	return out
}
func (l *metaLane) CachedModelInfo() []provider.ModelInfo { return l.info }

// metaSourceAliasManager is an alias manager that also resolves cited static
// catalog rows, exactly like the http server's catalog bridge.
type metaSourceAliasManager struct {
	stubAliasManager
	meta map[string]ModelMeta
}

func (m *metaSourceAliasManager) Sorted() []string {
	out := make([]string, 0, len(m.aliases))
	for k := range m.aliases {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func (m *metaSourceAliasManager) ModelMetaEntry(listedID, lane, upstream string) (ModelMeta, bool) {
	for _, key := range []string{listedID, lane + "/" + upstream, upstream} {
		if e, ok := m.meta[key]; ok {
			return e, true
		}
	}
	return ModelMeta{}, false
}

func callTool(t *testing.T, srv *Server, id int, name, args string) string {
	t.Helper()
	body := `{"jsonrpc":"2.0","id":` + strconv.Itoa(id) + `,"method":"tools/call","params":{"name":"` + name + `","arguments":` + args + `}}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("%s: %d %s", name, rec.Code, rec.Body.String())
	}
	var rpc struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &rpc); err != nil {
		t.Fatalf("decode %s reply: %v (%s)", name, err, rec.Body.String())
	}
	if rpc.Error != nil {
		t.Fatalf("%s rpc error: %s", name, rpc.Error.Message)
	}
	if len(rpc.Result.Content) == 0 {
		t.Fatalf("%s returned no content", name)
	}
	if rpc.Result.IsError {
		t.Fatalf("%s reported an error: %s", name, rpc.Result.Content[0].Text)
	}
	return rpc.Result.Content[0].Text
}

// listedEntry decodes one list_models row.
func listedEntry(t *testing.T, payload, id string) map[string]any {
	t.Helper()
	var rows map[string]map[string]any
	if err := json.Unmarshal([]byte(payload), &rows); err != nil {
		t.Fatalf("decode list_models payload: %v (%s)", err, payload)
	}
	row, ok := rows[id]
	if !ok {
		t.Fatalf("list_models has no %q (payload %s)", id, payload)
	}
	return row
}

// FR-03: list_models advertises the discovered output cap and modalities, and
// omits every one of those keys for a row without them.
func TestListModels_WindowOutputCapAndModalities(t *testing.T) {
	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: &metaLane{name: "openai", info: []provider.ModelInfo{
		{
			ID:               "gpt-4o",
			ContextLength:    128000,
			MaxOutput:        16384,
			InputModalities:  []string{"text", "image"},
			OutputModalities: []string{"text"},
		},
		{ID: "bare"},
	}}})
	srv := NewServer(registry, nil)
	payload := callTool(t, srv, 1, "list_models", `{}`)

	row := listedEntry(t, payload, "openai/gpt-4o")
	if row["context_length"] != float64(128000) || row["max_model_len"] != float64(128000) {
		t.Errorf("windows = %v / %v", row["context_length"], row["max_model_len"])
	}
	if row["max_output_tokens"] != float64(16384) {
		t.Errorf("max_output_tokens = %v, want 16384", row["max_output_tokens"])
	}
	if row["max_output"] != float64(16384) {
		t.Errorf("max_output = %v, want 16384 (kept for agents that read it)", row["max_output"])
	}
	arch, _ := row["architecture"].(map[string]any)
	if arch == nil {
		t.Fatalf("no architecture block: %v", row)
	}
	if in, _ := arch["input_modalities"].([]any); len(in) != 2 || in[0] != "text" || in[1] != "image" {
		t.Errorf("architecture.input_modalities = %v", arch["input_modalities"])
	}
	if row["supports_vision"] != true {
		t.Errorf("supports_vision = %v, want true", row["supports_vision"])
	}

	bare := listedEntry(t, payload, "openai/bare")
	for _, key := range []string{"context_length", "max_model_len", "max_output_tokens", "max_output", "architecture", "supports_vision"} {
		if _, ok := bare[key]; ok {
			t.Errorf("unknown metadata %q emitted as %v", key, bare[key])
		}
	}
}

// FR-05 on the MCP surface: the cited static catalog fills what discovery and
// the alias table leave out, and alias modalities win over the catalog.
func TestListModels_StaticCatalogAndAliasPrecedence(t *testing.T) {
	am := &metaSourceAliasManager{meta: map[string]ModelMeta{
		"glm-5.3": {ID: "glm-5.3", ContextLength: 1000000, InputModalities: []string{"text", "image"}, Source: "https://docs.z.ai/guides/coding-plan"},
	}}
	am.aliases = map[string]ModelAlias{
		"glm-5.3": {Provider: "zai", Upstream: "glm-5.3", InputModalities: []string{"text"}},
	}
	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: &metaLane{name: "zai", info: []provider.ModelInfo{{ID: "glm-5.3"}}}})
	srv := NewServer(registry, newStubStateSource(), WithAliasManager(am))
	payload := callTool(t, srv, 1, "list_models", `{}`)

	alias := listedEntry(t, payload, "glm-5.3")
	if alias["context_length"] != float64(1000000) {
		t.Errorf("alias context_length = %v, want 1000000 from the catalog", alias["context_length"])
	}
	if arch, _ := alias["architecture"].(map[string]any); arch != nil {
		if in, _ := arch["input_modalities"].([]any); len(in) != 1 || in[0] != "text" {
			t.Errorf("alias input_modalities = %v, want [text] (alias wins)", arch["input_modalities"])
		}
	}
	if _, ok := alias["supports_vision"]; ok {
		t.Errorf("supports_vision emitted despite a text-only alias: %v", alias["supports_vision"])
	}

	discovered := listedEntry(t, payload, "zai/glm-5.3")
	if discovered["context_length"] != float64(1000000) {
		t.Errorf("discovered context_length = %v, want 1000000 from the catalog", discovered["context_length"])
	}
	if discovered["supports_vision"] != true {
		t.Errorf("discovered supports_vision = %v, want true (catalog claims image)", discovered["supports_vision"])
	}
}

// FR-05: set_model_alias persists the modality arrays and rejects tokens that
// do not normalize.
func TestSetModelAliasModalities(t *testing.T) {
	am := &stubAliasManager{aliases: map[string]ModelAlias{}}
	srv := NewServer(nil, nil, WithAliasManager(am))
	callTool(t, srv, 1, "set_model_alias",
		`{"alias":"qwen","provider":"vllm","upstream":"Qwen/Qwen3","context_limit":131072,"max_output":8192,"input_modalities":["text","image"],"output_modalities":["text"]}`)

	got := am.aliases["qwen"]
	if len(got.InputModalities) != 2 || got.InputModalities[0] != "text" || got.InputModalities[1] != "image" {
		t.Errorf("input_modalities = %v, want [text image]", got.InputModalities)
	}
	if len(got.OutputModalities) != 1 || got.OutputModalities[0] != "text" {
		t.Errorf("output_modalities = %v, want [text]", got.OutputModalities)
	}
	if got.ContextLimit != 131072 || got.MaxOutput != 8192 {
		t.Errorf("limits = %d/%d", got.ContextLimit, got.MaxOutput)
	}

	// Unknown tokens are rejected rather than silently dropped.
	body := `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"set_model_alias","arguments":{"alias":"bad","provider":"vllm","upstream":"Qwen/Qwen3","input_modalities":["vibes"]}}}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body)))
	if !strings.Contains(rec.Body.String(), "vibes") || !strings.Contains(rec.Body.String(), "isError") {
		t.Fatalf("unknown modality not rejected: %s", rec.Body.String())
	}
	if _, ok := am.aliases["bad"]; ok {
		t.Error("rejected call still created the alias")
	}
}
