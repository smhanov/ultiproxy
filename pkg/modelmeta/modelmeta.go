// Package modelmeta is ultiproxy's cited static catalog of model context
// windows, output caps and input/output modalities.
//
// Live discovery (GET /v1/models on a lane) is always preferred, but many
// upstreams list ids with no metadata at all. This catalog is the last-resort
// source: rows that carry the vendor documentation URL they were taken from
// (Entry.Source). Nothing here is inferred from a model name.
//
// Operators extend it with an overlay file, data_dir/windows.json, whose
// entries merge over the compiled catalog at startup; a missing file means the
// compiled catalog only.
package modelmeta

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
)

// Modality tokens listing normalizes to. Anything a caller spells differently
// is either mapped (pdf -> file) or dropped, never invented.
const (
	ModalityText  = "text"
	ModalityImage = "image"
	ModalityFile  = "file"
	ModalityAudio = "audio"
	ModalityVideo = "video"
)

// Entry is one cited model row.
type Entry struct {
	// ID is the key the entry is looked up by: a listed id ("<lane>/<model>"
	// or an alias), or a bare upstream model id.
	ID string `json:"id"`
	// ContextLength is the context window in tokens (0 = unknown).
	ContextLength int `json:"context_length,omitempty"`
	// MaxOutput is the output cap in tokens (0 = unknown). It is never the
	// legacy max_tokens field.
	MaxOutput int `json:"max_output_tokens,omitempty"`
	// InputModalities / OutputModalities are normalized modality tokens.
	InputModalities  []string `json:"input_modalities,omitempty"`
	OutputModalities []string `json:"output_modalities,omitempty"`
	// Source is the vendor documentation URL the numbers came from. Every
	// entry must carry one: uncited numbers are indistinguishable from
	// guesses.
	Source string `json:"source"`
}

// empty reports whether the entry carries no metadata at all.
func (e Entry) empty() bool {
	return e.ContextLength <= 0 && e.MaxOutput <= 0 &&
		len(e.InputModalities) == 0 && len(e.OutputModalities) == 0
}

// Catalog is a read-only, thread-safe set of cited entries.
type Catalog struct {
	mu      sync.RWMutex
	entries map[string]Entry
}

// New validates and builds a catalog. An entry is rejected when it has no id,
// no source URL, no metadata at all, a non-positive window/output, or a
// modality token that does not normalize.
func New(entries []Entry) (*Catalog, error) {
	out := make(map[string]Entry, len(entries))
	for _, e := range entries {
		e.ID = strings.TrimSpace(e.ID)
		if e.ID == "" {
			return nil, errors.New("modelmeta: entry with no id")
		}
		if strings.TrimSpace(e.Source) == "" {
			return nil, fmt.Errorf("modelmeta: entry %q carries no source URL (uncited numbers are not advertised)", e.ID)
		}
		if e.empty() {
			return nil, fmt.Errorf("modelmeta: entry %q carries no window, output cap or modality", e.ID)
		}
		if e.ContextLength < 0 || e.MaxOutput < 0 {
			return nil, fmt.Errorf("modelmeta: entry %q carries a negative limit", e.ID)
		}
		if len(e.InputModalities) > 0 {
			if err := ValidateModalities(e.InputModalities); err != nil {
				return nil, fmt.Errorf("modelmeta: entry %q input modalities: %w", e.ID, err)
			}
		}
		if len(e.OutputModalities) > 0 {
			if err := ValidateModalities(e.OutputModalities); err != nil {
				return nil, fmt.Errorf("modelmeta: entry %q output modalities: %w", e.ID, err)
			}
		}
		e.InputModalities = NormalizeModalities(e.InputModalities)
		e.OutputModalities = NormalizeModalities(e.OutputModalities)
		if e.empty() {
			return nil, fmt.Errorf("modelmeta: entry %q normalized to no metadata", e.ID)
		}
		out[e.ID] = e
	}
	return &Catalog{entries: out}, nil
}

// Lookup returns the first entry found among ids, in the order given. Empty
// ids are skipped so callers can pass optional lookups (listed id, lane/id,
// bare upstream) without branching.
func (c *Catalog) Lookup(ids ...string) (Entry, bool) {
	if c == nil {
		return Entry{}, false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, id := range ids {
		if id == "" {
			continue
		}
		if e, ok := c.entries[id]; ok {
			return e, true
		}
	}
	return Entry{}, false
}

// IDs lists the catalog keys in sorted order (diagnostics and tests).
func (c *Catalog) IDs() []string {
	if c == nil {
		return nil
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]string, 0, len(c.entries))
	for id := range c.entries {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// OverlayFileName is the operator overlay, read from the server data dir.
const OverlayFileName = "windows.json"

// LoadOverlay returns the compiled catalog with the overlay file at path
// merged over it (overlay rows win, per field group). A missing file yields
// the compiled catalog and no error; a present but invalid file is an error,
// because silently ignoring operator-supplied numbers is worse than failing
// loud.
func LoadOverlay(path string) (*Catalog, error) {
	compiled := Default()
	if path == "" {
		return compiled, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return compiled, nil
		}
		return nil, fmt.Errorf("modelmeta: read overlay %s: %w", path, err)
	}
	var overlay []Entry
	if err := json.Unmarshal(data, &overlay); err != nil {
		return nil, fmt.Errorf("modelmeta: overlay %s is not a JSON array of entries: %w", path, err)
	}
	extra, err := New(overlay)
	if err != nil {
		return nil, fmt.Errorf("modelmeta: overlay %s: %w", path, err)
	}
	merged := make(map[string]Entry, len(compiled.entries)+len(extra.entries))
	for id, e := range compiled.entries {
		merged[id] = e
	}
	for id, e := range extra.entries {
		merged[id] = e
	}
	return &Catalog{entries: merged}, nil
}
