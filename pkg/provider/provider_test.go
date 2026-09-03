package provider

import (
	"context"
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
