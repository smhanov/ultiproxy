package hublane

import (
	"encoding/json"
	"fmt"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// ToolsFromExtra robustly extracts a slice of llmhub.Tool values from
// cfg.ExtraBody["tools"]. It tolerates the normalized llmhub shape, the
// OpenAI tool shape, a flat tool shape, and raw JSON.
func ToolsFromExtra(cfg *provider.RequestConfig) []llmhub.Tool {
	if cfg == nil || len(cfg.ExtraBody) == 0 {
		return nil
	}
	raw, ok := cfg.ExtraBody["tools"]
	if !ok || raw == nil {
		return nil
	}

	// 1. Already normalized.
	if tools, ok := raw.([]llmhub.Tool); ok {
		return tools
	}

	// 2. Slice of string-keyed maps.
	if maps, ok := raw.([]map[string]any); ok {
		return toolsFromMaps(maps)
	}

	// 3. Slice of any values (common after JSON unmarshalling into any).
	if items, ok := raw.([]any); ok {
		return toolsFromAnySlice(items)
	}

	// 4. json.RawMessage.
	if data, ok := raw.(json.RawMessage); ok && len(data) > 0 {
		var tools []llmhub.Tool
		if err := json.Unmarshal(data, &tools); err == nil && allToolsNamed(tools) {
			return tools
		}
		var maps []map[string]any
		if err := json.Unmarshal(data, &maps); err == nil {
			return toolsFromMaps(maps)
		}
		var items []any
		if err := json.Unmarshal(data, &items); err == nil {
			return toolsFromAnySlice(items)
		}
	}

	return nil
}

func allToolsNamed(tools []llmhub.Tool) bool {
	for _, t := range tools {
		if t.Name == "" {
			return false
		}
	}
	return true
}

func toolsFromAnySlice(items []any) []llmhub.Tool {
	maps := make([]map[string]any, 0, len(items))
	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		maps = append(maps, m)
	}
	return toolsFromMaps(maps)
}

func toolsFromMaps(maps []map[string]any) []llmhub.Tool {
	tools := make([]llmhub.Tool, 0, len(maps))
	for _, m := range maps {
		tool, err := toolFromMap(m)
		if err != nil {
			continue
		}
		tools = append(tools, tool)
	}
	return tools
}

func toolFromMap(m map[string]any) (llmhub.Tool, error) {
	// OpenAI shape: {"type":"function","function":{...}}
	if fnRaw, ok := m["function"]; ok {
		if fnMap, ok := fnRaw.(map[string]any); ok {
			return buildTool(fnMap)
		}
	}
	// Flat shape: {name,description,parameters}
	return buildTool(m)
}

func buildTool(m map[string]any) (llmhub.Tool, error) {
	name, _ := m["name"].(string)
	description, _ := m["description"].(string)
	params, _ := m["parameters"].(map[string]any)
	if name == "" {
		return llmhub.Tool{}, fmt.Errorf("tool missing name")
	}
	return llmhub.NewTool(name, description, params), nil
}
