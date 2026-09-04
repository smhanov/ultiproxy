package mcp

import (
	"context"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// TestGetQuotaStatus_NoQuotaMechanism: a registered lane that implements no
// QuotaProvider (plain openaicompat lanes: zai, opencode, vllm, ...) must be
// reported as "no quota mechanism", not as "not found or quota not available",
// which reads like a routing bug.
func TestGetQuotaStatus_NoQuotaMechanism(t *testing.T) {
	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: laneWithoutQuota{name: "zai"}})

	srv := NewServer(registry, newStubStateSource())

	res := callMCPTool(t, srv, 1, "get_quota_status", `{"provider":"zai"}`)
	if !res.IsError {
		t.Fatalf("expected an error result for a lane with no quota surface, got %+v", res)
	}
	text := res.Content[0].Text
	if !strings.Contains(text, "no quota mechanism") {
		t.Errorf("message = %q, want it to say the lane has no quota mechanism", text)
	}
	if strings.Contains(text, "not found") {
		t.Errorf("message = %q, must not claim the provider is missing", text)
	}
}

// laneWithoutQuota is an inference-only lane (no QuotaProvider).
type laneWithoutQuota struct {
	name string
}

func (l laneWithoutQuota) Name() string { return l.name }

func (l laneWithoutQuota) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	return &ir.Response{}, nil
}

func (l laneWithoutQuota) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	return nil, nil
}
