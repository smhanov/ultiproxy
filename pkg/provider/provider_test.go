package provider

import (
	"bytes"
	"context"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
)

type fakeInference struct{}

func (f fakeInference) Name() string { return "fake" }
func (f fakeInference) Generate(ctx context.Context, msgs []*ir.Message, opts ...Option) (*ir.Response, error) {
	return &ir.Response{FinishReason: "stop"}, nil
}
func (f fakeInference) Stream(ctx context.Context, msgs []*ir.Message, opts ...Option) (<-chan ir.Event, error) {
	return nil, ErrNotImplemented
}

type fakeQuota struct{}

func (f fakeQuota) Name() string { return "q" }
func (f fakeQuota) Quota(ctx context.Context) (*QuotaSnapshot, error) {
	return &QuotaSnapshot{Windows: []QuotaWindow{{Label: "w", UsedPct: 10}}}, nil
}

func TestRegistryRegisterGet(t *testing.T) {
	r := NewRegistry()
	p := Provider{Inference: fakeInference{}, Capabilities: Capabilities{Chat: true}}
	r.Register(p)

	got, ok := r.Get("fake")
	if !ok {
		t.Fatal("expected provider registered")
	}
	if got.Inference.Name() != "fake" {
		t.Fatalf("name mismatch: %s", got.Inference.Name())
	}
	if r.Len() != 1 {
		t.Fatalf("len = %d, want 1", r.Len())
	}
	if _, ok := r.Get("nope"); ok {
		t.Fatal("unexpected provider")
	}
}

func TestRegistryReplacesInPlace(t *testing.T) {
	r := NewRegistry()
	r.Register(Provider{Inference: fakeInference{}})
	r.Register(Provider{Quota: fakeQuota{}})
	if r.Len() != 2 {
		t.Fatalf("len = %d, want 2", r.Len())
	}
	names := r.Names()
	if len(names) != 2 {
		t.Fatalf("names = %v", names)
	}
}

// taggedInference is a same-named fake distinguishable by marker, so a
// replace test can prove lookups return the NEW provider.
type taggedInference struct {
	name   string
	marker string
}

func (f taggedInference) Name() string { return f.name }
func (f taggedInference) Generate(ctx context.Context, msgs []*ir.Message, opts ...Option) (*ir.Response, error) {
	return &ir.Response{FinishReason: "stop"}, nil
}
func (f taggedInference) Stream(ctx context.Context, msgs []*ir.Message, opts ...Option) (<-chan ir.Event, error) {
	return nil, ErrNotImplemented
}

// TestRegistryReplaceSameName (T007 AC1): registering a lane with an existing
// name replaces the entry; lookups return the new provider and order holds
// the name exactly once, in its original position.
func TestRegistryReplaceSameName(t *testing.T) {
	r := NewRegistry()
	r.Register(Provider{Inference: taggedInference{name: "lane", marker: "old"}})
	r.Register(Provider{Inference: taggedInference{name: "other", marker: "x"}})
	r.RegisterWithSource(Provider{Inference: taggedInference{name: "lane", marker: "new"}}, "restore")

	if r.Len() != 2 {
		t.Fatalf("len = %d, want 2 (no duplicate order entry)", r.Len())
	}
	names := r.Names()
	if len(names) != 2 || names[0] != "lane" || names[1] != "other" {
		t.Fatalf("names = %v, want [lane other] (position preserved, exactly once)", names)
	}
	got, ok := r.Get("lane")
	if !ok {
		t.Fatal("lane missing after replace")
	}
	inf, ok := got.Inference.(taggedInference)
	if !ok {
		t.Fatalf("lane inference = %T, want taggedInference", got.Inference)
	}
	if inf.marker != "new" {
		t.Fatalf("lane marker = %q, want %q (newest build wins)", inf.marker, "new")
	}
}

// TestRegistryReplaceLogsSource (T007 AC2): every replacement emits a log line
// naming the lane and its source; a first registration is silent.
func TestRegistryReplaceLogsSource(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)

	r := NewRegistry()
	r.RegisterWithSource(Provider{Inference: taggedInference{name: "xai", marker: "a"}}, "startup-scan:credential-scan")
	if out := buf.String(); out != "" {
		t.Fatalf("first registration logged %q, want silence", out)
	}

	r.RegisterWithSource(Provider{Inference: taggedInference{name: "xai", marker: "b"}}, "restore")
	out := buf.String()
	if !strings.Contains(out, `"xai"`) {
		t.Errorf("replacement log %q does not name the lane", out)
	}
	if !strings.Contains(out, "restore") {
		t.Errorf("replacement log %q does not name the source", out)
	}

	// Empty source defaults to runtime.
	buf.Reset()
	r.Register(Provider{Inference: taggedInference{name: "xai", marker: "c"}})
	if out := buf.String(); !strings.Contains(out, "runtime") {
		t.Errorf("default-source log %q does not name %q", out, "runtime")
	}
}

func TestOptionsBuildConfig(t *testing.T) {
	cfg := NewRequestConfig(
		WithModel("m"),
		WithMaxTokens(100),
		WithReasoningEffort("high"),
		WithHeader("X-Test", "1"),
		WithExtraBody(map[string]any{"reasoning_effort": "high"}),
		WithClientKeyHash("abc"),
	)
	if cfg.Model != "m" || cfg.MaxTokens != 100 || cfg.ReasoningEffort != "high" {
		t.Fatalf("options not applied: %+v", cfg)
	}
	if cfg.Headers["X-Test"] != "1" {
		t.Fatalf("header missing")
	}
	if cfg.ExtraBody["reasoning_effort"] != "high" {
		t.Fatalf("extra body missing")
	}
	if cfg.ClientKeyHash != "abc" {
		t.Fatalf("client key hash missing")
	}
}

func TestProviderMethodSelection(t *testing.T) {
	p := Provider{Inference: fakeInference{}}
	if p.Inference == nil || p.Quota != nil || p.Auth != nil {
		t.Fatal("bundle fields wrong")
	}
}
