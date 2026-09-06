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
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/smhanov/ultiproxy/pkg/mcp"
	"github.com/smhanov/ultiproxy/pkg/modelmeta"
)

// ModelAlias maps a client-visible alias to a provider lane + upstream id.
//
// MaxOutput is enforced on the request path: max_tokens is clamped to it in
// pkg/server/handlers.go (upstreamOptions). ContextLimit is advisory metadata
// only — it is surfaced to clients through GET /v1/models (context_length) and
// is not enforced, because estimating prompt tokens would require a
// provider-specific tokenizer.
type ModelAlias struct {
	Provider     string `yaml:"provider" json:"provider"`
	Upstream     string `yaml:"upstream" json:"upstream"`
	ContextLimit int    `yaml:"context_limit,omitempty" json:"context_limit,omitempty"`
	MaxOutput    int    `yaml:"max_output,omitempty" json:"max_output,omitempty"`
	PricingTag   string `yaml:"pricing_tag,omitempty" json:"pricing_tag,omitempty"`
	// InputCost / OutputCost price the alias in US dollars per 1M prompt /
	// completion tokens. They ride along on the request as provider.WithCost
	// (llmhub lanes) and back-fill the recorded usage cost when the upstream
	// reports none, so accounting works on lanes that never price themselves.
	InputCost       float64            `yaml:"input_cost,omitempty" json:"input_cost,omitempty"`
	OutputCost      float64            `yaml:"output_cost,omitempty" json:"output_cost,omitempty"`
	BenchmarkScores map[string]float64 `yaml:"benchmarks,omitempty" json:"benchmarks,omitempty"`
	// InputModalities / OutputModalities are optional operator-asserted
	// modality arrays (text, image, file, audio, video). They are advisory
	// metadata surfaced on GET /v1/models as architecture.input_modalities /
	// output_modalities (plus supports_vision when image is an input), and
	// they win over both live discovery and the cited static catalog. Empty
	// means "not asserted": nothing is emitted for it.
	InputModalities  []string `yaml:"input_modalities,omitempty" json:"input_modalities,omitempty"`
	OutputModalities []string `yaml:"output_modalities,omitempty" json:"output_modalities,omitempty"`
}

// ModelCatalog is a thread-safe alias registry with JSON persistence. It also
// owns the cited static model catalog (modelmeta) plus its operator overlay,
// because both describe the same listed ids.
type ModelCatalog struct {
	mu          sync.RWMutex
	aliases     map[string]ModelAlias
	persistPath string
	// meta is the cited static catalog (compiled seed merged with the
	// data_dir/windows.json overlay). Immutable after construction, so reads
	// need no lock.
	meta *modelmeta.Catalog
}

// NewModelCatalog builds a catalog from config models, then overlays any
// runtime-persisted aliases from persistPath (runtime changes win). The cited
// static catalog is loaded from the same directory (windows.json over the
// compiled seed); a missing overlay file means the compiled seed only, and a
// present but invalid overlay is logged and ignored rather than dropping the
// daemon's model listing.
func NewModelCatalog(configModels map[string]ModelAlias, persistPath string) (*ModelCatalog, error) {
	c := &ModelCatalog{
		aliases:     make(map[string]ModelAlias, len(configModels)+4),
		persistPath: persistPath,
	}
	for alias, entry := range configModels {
		c.aliases[alias] = entry
	}
	overlayPath := ""
	if dir := filepath.Dir(persistPath); dir != "" && dir != "." {
		overlayPath = filepath.Join(dir, modelmeta.OverlayFileName)
	}
	meta, metaErr := modelmeta.LoadOverlay(overlayPath)
	switch {
	case metaErr != nil:
		log.Printf("[server] static model catalog overlay ignored: %v (serving the compiled catalog)", metaErr)
		c.meta = modelmeta.Default()
	default:
		c.meta = meta
	}
	if persistPath == "" {
		return c, nil
	}
	data, err := os.ReadFile(persistPath)
	switch {
	case err == nil:
		var extra map[string]ModelAlias
		if err := json.Unmarshal(data, &extra); err != nil {
			// A truncated/garbage aliases.json is control-plane corruption, not
			// "no runtime state": report it so startup logs it instead of
			// silently coming up with an empty runtime overlay.
			return nil, fmt.Errorf("model catalog: corrupt persistence file %s: %w (fix or remove the file and restart)", persistPath, err)
		}
		for alias, entry := range extra {
			c.aliases[alias] = entry
		}
	case errors.Is(err, os.ErrNotExist):
		// No persisted runtime state yet.
	default:
		return nil, fmt.Errorf("model catalog: read persistence file %s: %w", persistPath, err)
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
//
// The mutation, its snapshot and the disk write all happen under the write
// lock, so two concurrent MCP set_model_alias calls cannot interleave their
// snapshots or their temporary files, and the new alias only becomes live once
// the write succeeded. A failed persist therefore leaves the catalog exactly
// as it was while still returning the error to the caller.
func (c *ModelCatalog) Set(alias string, entry ModelAlias) error {
	if alias == "" || entry.Provider == "" || entry.Upstream == "" {
		return errors.New("alias, provider and upstream are required")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	next := cloneAliases(c.aliases)
	next[alias] = entry
	if err := c.persistAliases(next); err != nil {
		return err
	}
	c.aliases = next
	return nil
}

// Remove deletes an alias and persists the change, with the same
// persist-before-publish ordering as Set.
func (c *ModelCatalog) Remove(alias string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	next := cloneAliases(c.aliases)
	delete(next, alias)
	if err := c.persistAliases(next); err != nil {
		return err
	}
	c.aliases = next
	return nil
}

// persistAliases writes one complete alias state atomically. Callers must hold
// c.mu so the state they pass is the state that gets published.
func (c *ModelCatalog) persistAliases(state map[string]ModelAlias) error {
	if c.persistPath == "" {
		return nil
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(c.persistPath, data)
}

func cloneAliases(in map[string]ModelAlias) map[string]ModelAlias {
	out := make(map[string]ModelAlias, len(in)+1)
	for k, v := range in {
		out[k] = v
	}
	return out
}

// writeFileAtomic persists data to path through a uniquely named temporary
// file in the same directory followed by a rename:
//
//   - the temp name is unique per writer (os.CreateTemp), so concurrent
//     mutations never write or rename each other's half-written file;
//   - the rename is atomic, so a crash can never leave a truncated path
//     behind for the next startup to misread;
//   - the temp file is removed again if anything fails before the rename.
//
// catalog.go, timeouts.go and providers.go all persist through this helper.
func writeFileAtomic(path string, data []byte) error {
	dir := filepath.Dir(path)
	if dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return fmt.Errorf("create directory %s: %w", dir, err)
		}
	}
	f, err := os.CreateTemp(dir, "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("create temp file in %s: %w", dir, err)
	}
	tmp := f.Name()
	defer os.Remove(tmp) // no-op once the rename below succeeded
	if _, err := f.Write(data); err != nil {
		f.Close()
		return fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("sync %s: %w", tmp, err)
	}
	if err := f.Chmod(0o600); err != nil {
		f.Close()
		return fmt.Errorf("chmod %s: %w", tmp, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("rename %s to %s: %w", tmp, path, err)
	}
	return nil
}

// MetaEntry resolves the cited static catalog entry for one listed id. Keys
// are tried in order - the listed id ("zai/glm-5.3" or an alias), then
// "<lane>/<upstream>", then the bare upstream id - so both a lane-prefixed
// discovery row and an alias resolve the same catalog row. Callers only use it
// to fill gaps: live discovery and operator aliases win over it.
func (c *ModelCatalog) MetaEntry(listedID, lane, upstream string) (modelmeta.Entry, bool) {
	if c == nil || c.meta == nil {
		return modelmeta.Entry{}, false
	}
	return c.meta.Lookup(listedID, lane+"/"+upstream, upstream)
}

// ModelMetaEntry adapts ModelCatalog.MetaEntry to the MCP surface
// (mcp.ModelMetaSource), so list_models applies the same precedence chain as
// GET /v1/models.
func (b *catalogBridge) ModelMetaEntry(listedID, lane, upstream string) (mcp.ModelMeta, bool) {
	return b.catalog.MetaEntry(listedID, lane, upstream)
}
