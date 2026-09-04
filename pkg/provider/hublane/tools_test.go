package hublane

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestToolsFromExtra_NormalizedSlice(t *testing.T) {
	want := []llmhub.Tool{
		llmhub.NewTool("weather", "Get weather", map[string]any{"type": "object"}),
		llmhub.NewTool("calculator", "Do math", nil),
	}
	cfg := provider.NewRequestConfig(provider.WithExtraBody(map[string]any{
		"tools": want,
	}))

	got := ToolsFromExtra(cfg)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ToolsFromExtra returned %+v, want %+v", got, want)
	}
}

func TestToolsFromExtra_OpenAIAnySlice(t *testing.T) {
	cfg := provider.NewRequestConfig(provider.WithExtraBody(map[string]any{
		"tools": []any{
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        "weather",
					"description": "Get weather",
					"parameters":  map[string]any{"type": "object"},
				},
			},
		},
	}))

	got := ToolsFromExtra(cfg)
	want := []llmhub.Tool{
		llmhub.NewTool("weather", "Get weather", map[string]any{"type": "object"}),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ToolsFromExtra returned %+v, want %+v", got, want)
	}
}

func TestToolsFromExtra_FlatAnySlice(t *testing.T) {
	cfg := provider.NewRequestConfig(provider.WithExtraBody(map[string]any{
		"tools": []any{
			map[string]any{
				"name":        "calculator",
				"description": "Do math",
				"parameters":  map[string]any{"type": "object"},
			},
		},
	}))

	got := ToolsFromExtra(cfg)
	want := []llmhub.Tool{
		llmhub.NewTool("calculator", "Do math", map[string]any{"type": "object"}),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ToolsFromExtra returned %+v, want %+v", got, want)
	}
}

func TestToolsFromExtra_MapSlice(t *testing.T) {
	cfg := provider.NewRequestConfig(provider.WithExtraBody(map[string]any{
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "weather",
					"description": "Get weather",
					"parameters":  map[string]any{"type": "object"},
				},
			},
			{
				"name":        "calculator",
				"description": "Do math",
			},
		},
	}))

	got := ToolsFromExtra(cfg)
	want := []llmhub.Tool{
		llmhub.NewTool("weather", "Get weather", map[string]any{"type": "object"}),
		llmhub.NewTool("calculator", "Do math", nil),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ToolsFromExtra returned %+v, want %+v", got, want)
	}
}

func TestToolsFromExtra_RawMessage(t *testing.T) {
	data := json.RawMessage(`[
		{"type":"function","function":{"name":"weather","description":"Get weather","parameters":{"type":"object"}}},
		{"name":"calculator","description":"Do math"}
	]`)
	cfg := provider.NewRequestConfig(provider.WithExtraBody(map[string]any{
		"tools": data,
	}))

	got := ToolsFromExtra(cfg)
	want := []llmhub.Tool{
		llmhub.NewTool("weather", "Get weather", map[string]any{"type": "object"}),
		llmhub.NewTool("calculator", "Do math", nil),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ToolsFromExtra returned %+v, want %+v", got, want)
	}
}

func TestToolsFromExtra_RawMessageAsAnySlice(t *testing.T) {
	data := json.RawMessage(`[
		{"name":"flat","description":"flat tool","parameters":{"type":"object"}}
	]`)
	cfg := provider.NewRequestConfig(provider.WithExtraBody(map[string]any{
		"tools": data,
	}))

	got := ToolsFromExtra(cfg)
	want := []llmhub.Tool{
		llmhub.NewTool("flat", "flat tool", map[string]any{"type": "object"}),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ToolsFromExtra returned %+v, want %+v", got, want)
	}
}

func TestToolsFromExtra_EmptyAndNil(t *testing.T) {
	cases := []struct {
		name string
		cfg  *provider.RequestConfig
	}{
		{"nil config", nil},
		{"empty extra body", provider.NewRequestConfig()},
		{"nil tools value", provider.NewRequestConfig(provider.WithExtraBody(map[string]any{"tools": nil}))},
		{"empty any slice", provider.NewRequestConfig(provider.WithExtraBody(map[string]any{"tools": []any{}}))},
		{"empty map slice", provider.NewRequestConfig(provider.WithExtraBody(map[string]any{"tools": []map[string]any{}}))},
		{"empty raw message", provider.NewRequestConfig(provider.WithExtraBody(map[string]any{"tools": json.RawMessage{}}))},
		{"missing tools key", provider.NewRequestConfig(provider.WithExtraBody(map[string]any{"other": []llmhub.Tool{}}))},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ToolsFromExtra(tc.cfg)
			if len(got) != 0 {
				t.Fatalf("expected empty tools, got %+v", got)
			}
		})
	}
}

func TestToolsFromExtra_SkipsInvalidEntries(t *testing.T) {
	cfg := provider.NewRequestConfig(provider.WithExtraBody(map[string]any{
		"tools": []any{
			map[string]any{"name": "valid", "description": "ok"},
			"not a map",
			map[string]any{"description": "missing name"},
			42,
		},
	}))

	got := ToolsFromExtra(cfg)
	want := []llmhub.Tool{
		llmhub.NewTool("valid", "ok", nil),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ToolsFromExtra returned %+v, want %+v", got, want)
	}
}
