package modelmeta

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestNormalizeModalities(t *testing.T) {
	got := NormalizeModalities([]string{"Text", "PDF", "image", "", "telepathy", "image"})
	if !reflect.DeepEqual(got, []string{"text", "file", "image"}) {
		t.Errorf("normalized = %v, want [text file image]", got)
	}
	if got := NormalizeModalities(nil); got != nil {
		t.Errorf("nil input normalized to %v, want nil (unknown)", got)
	}
}

func TestValidateModalitiesRejectsUnknownToken(t *testing.T) {
	if err := ValidateModalities([]string{"text", "vibes"}); err == nil || !strings.Contains(err.Error(), "vibes") {
		t.Fatalf("err = %v, want a message naming the bad token", err)
	}
	if err := ValidateModalities([]string{"text", "pdf", "image", "audio", "video", "file"}); err != nil {
		t.Fatalf("valid tokens rejected: %v", err)
	}
}

func TestParseModalityString(t *testing.T) {
	cases := []struct {
		in              string
		wantIn, wantOut []string
	}{
		{"text+image->text", []string{"text", "image"}, []string{"text"}},
		{"text+image+file->text", []string{"text", "image", "file"}, []string{"text"}},
		{"text->text+audio", []string{"text"}, []string{"text", "audio"}},
		{"text", []string{"text"}, nil},
		{"", nil, nil},
	}
	for _, c := range cases {
		in, out := ParseModalityString(c.in)
		if !reflect.DeepEqual(in, c.wantIn) || !reflect.DeepEqual(out, c.wantOut) {
			t.Errorf("ParseModalityString(%q) = %v / %v, want %v / %v", c.in, in, out, c.wantIn, c.wantOut)
		}
	}
}

func TestHasImage(t *testing.T) {
	if !HasImage([]string{"text", "image"}) {
		t.Error("HasImage([text image]) = false")
	}
	if HasImage([]string{"text"}) || HasImage(nil) {
		t.Error("HasImage reported vision without image input")
	}
}

// Every catalog entry must carry a source URL and at least one datum.
func TestNewRejectsUncitedAndEmptyEntries(t *testing.T) {
	cases := []struct {
		name string
		in   []Entry
		want string
	}{
		{"no source", []Entry{{ID: "m", ContextLength: 1000}}, "source"},
		{"no data", []Entry{{ID: "m", Source: "https://vendor.example/models"}}, "no window"},
		{"no id", []Entry{{Source: "https://vendor.example/models", ContextLength: 1}}, "no id"},
	}
	for _, c := range cases {
		_, err := New(c.in)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want it to mention %q", c.name, err, c.want)
		}
	}
	if _, err := New([]Entry{{ID: "m", Source: "https://vendor.example/models", ContextLength: 4096, InputModalities: []string{"text"}}}); err != nil {
		t.Fatalf("valid entry rejected: %v", err)
	}
}

func TestLookupOrder(t *testing.T) {
	c, err := New([]Entry{
		{ID: "glm-5.3", Source: "https://docs.z.ai/guides/coding-plan", ContextLength: 1000000},
		{ID: "zai/glm-5.3", Source: "https://docs.z.ai/guides/coding-plan", ContextLength: 200000},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if e, ok := c.Lookup("glm-5.3"); !ok || e.ContextLength != 1000000 {
		t.Errorf("Lookup(glm-5.3) = %+v ok=%v", e, ok)
	}
	// First non-empty id wins, so a listed id beats the lane-prefixed form.
	if e, _ := c.Lookup("glm-5.3", "zai/glm-5.3"); e.ContextLength != 1000000 {
		t.Errorf("listed id did not win: %+v", e)
	}
	if e, _ := c.Lookup("", "zai/glm-5.3"); e.ContextLength != 200000 {
		t.Errorf("empty id was not skipped: %+v", e)
	}
	if _, ok := c.Lookup("nope", ""); ok {
		t.Error("Lookup invented an entry")
	}
	var nilCatalog *Catalog
	if _, ok := nilCatalog.Lookup("glm-5.3"); ok {
		t.Error("nil catalog invented an entry")
	}
}

// The compiled seed is cited and covers the operator's documented aliases.
func TestDefaultSeedIsCitedAndUseful(t *testing.T) {
	c := Default()
	if len(c.IDs()) == 0 {
		t.Fatal("compiled seed catalog is empty")
	}
	for _, e := range c.entries {
		if !strings.HasPrefix(e.Source, "https://") {
			t.Errorf("%s: source %q is not a vendor documentation URL", e.ID, e.Source)
		}
		if e.ContextLength <= 0 && e.MaxOutput <= 0 && len(e.InputModalities) == 0 {
			t.Errorf("%s: no metadata", e.ID)
		}
		for _, mod := range append(append([]string{}, e.InputModalities...), e.OutputModalities...) {
			if _, ok := normalizeToken(mod); !ok {
				t.Errorf("%s: unnormalized modality %q", e.ID, mod)
			}
		}
	}
	// The z.ai coding-plan ids have no live window: the catalog is what fills
	// them (T034 AC5).
	for _, id := range []string{"glm-5.3", "glm-5.3-flash"} {
		e, ok := c.Lookup(id)
		if !ok {
			t.Errorf("seed missing %s", id)
			continue
		}
		if e.ContextLength != 1000000 {
			t.Errorf("%s context length = %d, want 1000000", id, e.ContextLength)
		}
	}
	// Ids that live discovery owns (vLLM, OpenRouter fakes, Anthropic) must
	// not be seeded: the catalog must never shadow a served window.
	for _, id := range []string{"Qwen/Qwen3", "gpt-4o", "bare-id", "m1", "tofino-3"} {
		if _, ok := c.Lookup(id); ok {
			t.Errorf("seed shadows live discovery for %s", id)
		}
	}
}

func TestLoadOverlayMergesOverCompiled(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, OverlayFileName)
	overlay := `[{"id":"glm-5.3","context_length":1000000,"input_modalities":["text","image"],"source":"https://docs.z.ai/guides/coding-plan"},
	            {"id":"operator-only","context_length":8192,"source":"https://operator.example/models"}]`
	if err := os.WriteFile(path, []byte(overlay), 0o600); err != nil {
		t.Fatalf("write overlay: %v", err)
	}
	c, err := LoadOverlay(path)
	if err != nil {
		t.Fatalf("LoadOverlay: %v", err)
	}
	if e, _ := c.Lookup("glm-5.3"); e.ContextLength != 1000000 || !HasImage(e.InputModalities) {
		t.Errorf("overlay row did not win: %+v", e)
	}
	if e, ok := c.Lookup("operator-only"); !ok || e.ContextLength != 8192 {
		t.Errorf("overlay-only entry missing: %+v ok=%v", e, ok)
	}
	// Compiled rows survive the merge.
	if _, ok := c.Lookup("grok-4.6"); !ok {
		t.Error("compiled seed lost by overlay merge")
	}
}

func TestLoadOverlayMissingFileMeansCompiledOnly(t *testing.T) {
	c, err := LoadOverlay(filepath.Join(t.TempDir(), OverlayFileName))
	if err != nil {
		t.Fatalf("LoadOverlay on missing file: %v", err)
	}
	if len(c.IDs()) == 0 {
		t.Error("missing overlay emptied the compiled catalog")
	}
	if _, err := LoadOverlay(""); err != nil {
		t.Errorf("empty path: %v", err)
	}
}

func TestLoadOverlayRejectsUncitedRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), OverlayFileName)
	if err := os.WriteFile(path, []byte(`[{"id":"guessed","context_length":999999}]`), 0o600); err != nil {
		t.Fatalf("write overlay: %v", err)
	}
	if _, err := LoadOverlay(path); err == nil || !strings.Contains(err.Error(), "source") {
		t.Fatalf("err = %v, want uncited row rejected", err)
	}
	if err := os.WriteFile(path, []byte(`{"not":"an array"}`), 0o600); err != nil {
		t.Fatalf("rewrite overlay: %v", err)
	}
	if _, err := LoadOverlay(path); err == nil {
		t.Fatal("non-array overlay accepted")
	}
}
