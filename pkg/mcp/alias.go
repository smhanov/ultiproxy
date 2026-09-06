package mcp

import "github.com/smhanov/ultiproxy/pkg/modelmeta"

// ModelAlias is the mcp-local shape for a client-visible model alias mapping.
// The http Server adapts its own catalog to this interface to avoid an import
// cycle (mcp cannot import server).
type ModelAlias struct {
	Provider        string             `json:"provider"`
	Upstream        string             `json:"upstream"`
	ContextLimit    int                `json:"context_limit,omitempty"`
	MaxOutput       int                `json:"max_output,omitempty"`
	PricingTag      string             `json:"pricing_tag,omitempty"`
	InputCost       float64            `json:"input_cost,omitempty"`
	OutputCost      float64            `json:"output_cost,omitempty"`
	BenchmarkScores map[string]float64 `json:"benchmarks,omitempty"`
	// InputModalities / OutputModalities are optional operator-asserted
	// modality arrays (text, image, file, audio, video). Advisory metadata:
	// surfaced on GET /v1/models and list_models, and preferred over live
	// discovery and the cited static catalog. Empty means "not asserted".
	InputModalities  []string `json:"input_modalities,omitempty"`
	OutputModalities []string `json:"output_modalities,omitempty"`
}

// AliasManager is implemented by the http Server's model alias catalog.
type AliasManager interface {
	List() map[string]ModelAlias
	Sorted() []string
	Set(alias string, entry ModelAlias) error
	Remove(alias string) error
}

// TimeoutManager is implemented by the http Server's per-provider timeout
// store. Durations are exchange as Go duration strings ("10m", "3m30s").
type TimeoutManager interface {
	Timeout(provider string) string
	Set(provider string, timeout string) error
	Remove(provider string) error
	List() map[string]string
}

// ModelMeta is one cited static-catalog row, shared with pkg/modelmeta so the
// http listing and list_models can never drift apart.
type ModelMeta = modelmeta.Entry

// ModelMetaSource resolves cited static-catalog metadata for a listed id. The
// http server's alias bridge implements it, so list_models applies exactly the
// precedence GET /v1/models applies: alias fields, then live discovery, then
// this catalog, then omit.
type ModelMetaSource interface {
	// ModelMetaEntry resolves a row by listed id, "<lane>/<upstream>" and
	// bare upstream id, in that order.
	ModelMetaEntry(listedID, lane, upstream string) (ModelMeta, bool)
}

// ModelInfoStore is the MCP set_model_info overlay. The http server's catalog
// bridge implements it so listing and the MCP tools share one persist-before-publish map.
type ModelInfoStore interface {
	ModelInfoEntry(listedID string) (ModelMeta, bool)
	MergeModelInfo(id string, patch ModelMeta) error
	ClearModelInfo(id string, fields []string) error
	ListedIDs() []string
}
