package mcp

// ModelAlias is the mcp-local shape for a client-visible model alias mapping.
// The http Server adapts its own catalog to this interface to avoid an import
// cycle (mcp cannot import server).
type ModelAlias struct {
	Provider        string             `json:"provider"`
	Upstream        string             `json:"upstream"`
	ContextLimit    int                `json:"context_limit,omitempty"`
	MaxOutput       int                `json:"max_output,omitempty"`
	PricingTag      string             `json:"pricing_tag,omitempty"`
	BenchmarkScores map[string]float64 `json:"benchmarks,omitempty"`
}

// AliasManager is implemented by the http Server's model alias catalog.
type AliasManager interface {
	List() map[string]ModelAlias
	Sorted() []string
	Set(alias string, entry ModelAlias) error
	Remove(alias string) error
}
