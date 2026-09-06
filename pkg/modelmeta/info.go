package modelmeta

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// InfoFileName is the operator overlay for listed-id metadata, persisted
// under the server data dir. It is distinct from windows.json (the cited
// static catalog overlay): this file is written by MCP set_model_info and
// keyed by already-listed ids.
const InfoFileName = "model_info.json"

// InfoOverlay is a persist-before-publish map of listed-id metadata patches.
type InfoOverlay struct {
	mu      sync.Mutex
	path    string
	entries map[string]Entry
}

// NewInfoOverlay loads path if it exists. A missing file is empty, not an error.
func NewInfoOverlay(path string) (*InfoOverlay, error) {
	o := &InfoOverlay{path: path, entries: map[string]Entry{}}
	if path == "" {
		return o, nil
	}
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		var extra map[string]Entry
		if err := json.Unmarshal(data, &extra); err != nil {
			return nil, fmt.Errorf("modelmeta: corrupt info overlay %s: %w", path, err)
		}
		for id, e := range extra {
			e.ID = id
			o.entries[id] = e
		}
	case errors.Is(err, os.ErrNotExist):
	default:
		return nil, fmt.Errorf("modelmeta: read info overlay %s: %w", path, err)
	}
	return o, nil
}

// Get returns the overlay row for a listed id.
func (o *InfoOverlay) Get(id string) (Entry, bool) {
	if o == nil {
		return Entry{}, false
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	e, ok := o.entries[id]
	return e, ok
}

// Merge applies a partial patch onto an existing overlay row. Zero/empty
// fields in patch are ignored (not stored). The caller must already have
// validated the id is listed and rejected 0/[] values.
func (o *InfoOverlay) Merge(id string, patch Entry) error {
	if o == nil {
		return errors.New("model info overlay not configured")
	}
	if id == "" {
		return errors.New("id is required")
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	next := cloneEntries(o.entries)
	cur := next[id]
	cur.ID = id
	if patch.ContextLength > 0 {
		cur.ContextLength = patch.ContextLength
	}
	if patch.MaxOutput > 0 {
		cur.MaxOutput = patch.MaxOutput
	}
	if len(patch.InputModalities) > 0 {
		cur.InputModalities = append([]string(nil), patch.InputModalities...)
	}
	if len(patch.OutputModalities) > 0 {
		cur.OutputModalities = append([]string(nil), patch.OutputModalities...)
	}
	if patch.Source != "" {
		cur.Source = patch.Source
	}
	if cur.Source == "" {
		cur.Source = "mcp:set_model_info"
	}
	if cur.empty() {
		return errors.New("overlay patch carries no window, output cap or modality")
	}
	next[id] = cur
	if err := o.persist(next); err != nil {
		return err
	}
	o.entries = next
	return nil
}

// Clear removes the overlay for id, or only the named fields when fields is non-empty.
func (o *InfoOverlay) Clear(id string, fields []string) error {
	if o == nil {
		return errors.New("model info overlay not configured")
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	next := cloneEntries(o.entries)
	cur, ok := next[id]
	if !ok {
		return nil
	}
	if len(fields) == 0 {
		delete(next, id)
	} else {
		for _, f := range fields {
			switch f {
			case "context_length", "context_limit", "max_model_len":
				cur.ContextLength = 0
			case "max_output", "max_output_tokens":
				cur.MaxOutput = 0
			case "input_modalities":
				cur.InputModalities = nil
			case "output_modalities":
				cur.OutputModalities = nil
			}
		}
		if cur.empty() {
			delete(next, id)
		} else {
			next[id] = cur
		}
	}
	if err := o.persist(next); err != nil {
		return err
	}
	o.entries = next
	return nil
}

func (o *InfoOverlay) persist(state map[string]Entry) error {
	if o.path == "" {
		return nil
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(o.path)
	if dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	f, err := os.CreateTemp(dir, "."+filepath.Base(o.path)+".*.tmp")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Chmod(0o600); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, o.path)
}

func cloneEntries(in map[string]Entry) map[string]Entry {
	out := make(map[string]Entry, len(in)+1)
	for k, v := range in {
		out[k] = v
	}
	return out
}
