// Package server: model alias catalog.
//
// Ultiproxy supports arbitrary client-visible model names mapped to any
// registered provider lane + upstream identifier. A user can expose
// "qwenpoint-3.8" for the weird upstream id "Qwen/Qwen3.8-Instruct-AWQ" on
// their vLLM lane. Aliases are configured in config.yaml (models:) and can
// be created/removed at runtime via the MCP tools (set_model_alias,
// remove_model_alias, list_model_aliases). Runtime changes persist to a JSON
// file under data_dir so they survive restarts.
package server

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// ModelAlias maps a client-visible alias to a provider lane + upstream id.
//
// MaxOutput is enforced on the request path: max_tokens is clamped to it in
// pkg/server/handlers.go (upstreamOptions). ContextLimit is advisory metadata
// only — it is surfaced to clients through GET /v1/models (context_length) and
// is not enforced, because estimating prompt tokens would require a
// provider-specific tokenizer.
type ModelAlias struct {
	Provider        string             `yaml:"provider" json:"provider"`
	Upstream        string             `yaml:"upstream" json:"upstream"`
	ContextLimit    int                `yaml:"context_limit,omitempty" json:"context_limit,omitempty"`
	MaxOutput       int                `yaml:"max_output,omitempty" json:"max_output,omitempty"`
	PricingTag      string             `yaml:"pricing_tag,omitempty" json:"pricing_tag,omitempty"`
	BenchmarkScores map[string]float64 `yaml:"benchmarks,omitempty" json:"benchmarks,omitempty"`
}

// ModelCatalog is a thread-safe alias registry with JSON persistence.
type ModelCatalog struct {
	mu          sync.RWMutex
	aliases     map[string]ModelAlias
	persistPath string
}

// NewModelCatalog builds a catalog from config models, then overlays any
// runtime-persisted aliases from persistPath (runtime changes win).
func NewModelCatalog(configModels map[string]ModelAlias, persistPath string) (*ModelCatalog, error) {
	c := &ModelCatalog{
		aliases:     make(map[string]ModelAlias, len(configModels)+4),
		persistPath: persistPath,
	}
	for alias, entry := range configModels {
		c.aliases[alias] = entry
	}
	if persistPath != "" {
		if data, err := os.ReadFile(persistPath); err == nil {
			var extra map[string]ModelAlias
			if json.Unmarshal(data, &extra) == nil {
				for alias, entry := range extra {
					c.aliases[alias] = entry
				}
			}
		}
	}
	return c, nil
}

// List returns all aliases sorted by name with their entries.
func (c *ModelCatalog) List() map[string]ModelAlias {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make(map[string]ModelAlias, len(c.aliases))
	for k, v := range c.aliases {
		out[k] = v
	}
	return out
}

// Sorted returns alias names in stable order (for MCP listing).
func (c *ModelCatalog) Sorted() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	names := make([]string, 0, len(c.aliases))
	for k := range c.aliases {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// UpstreamName resolves an alias to its upstream model id.
func (c *ModelCatalog) UpstreamName(alias string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	e, ok := c.aliases[alias]
	if !ok {
		return "", false
	}
	return e.Upstream, true
}

// Get returns the full alias entry.
func (c *ModelCatalog) Get(alias string) (ModelAlias, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	e, ok := c.aliases[alias]
	return e, ok
}

// Set adds or replaces an alias and persists it.
func (c *ModelCatalog) Set(alias string, entry ModelAlias) error {
	if alias == "" || entry.Provider == "" || entry.Upstream == "" {
		return errors.New("alias, provider and upstream are required")
	}
	c.mu.Lock()
	c.aliases[alias] = entry
	c.mu.Unlock()
	return c.persist()
}

// Remove deletes an alias and persists the change.
func (c *ModelCatalog) Remove(alias string) error {
	c.mu.Lock()
	delete(c.aliases, alias)
	c.mu.Unlock()
	return c.persist()
}

func (c *ModelCatalog) persist() error {
	if c.persistPath == "" {
		return nil
	}
	c.mu.RLock()
	snapshot := make(map[string]ModelAlias, len(c.aliases))
	for k, v := range c.aliases {
		snapshot[k] = v
	}
	c.mu.RUnlock()

	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(c.persistPath); dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	tmp := c.persistPath + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, c.persistPath)
}
