package hublane

import (
	"context"
	"testing"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// 1. provider.WithCost on a request is forwarded to llmhub as llmhub.WithCost,
// so a priced model alias (input_cost / output_cost) reaches the lane's
// per-million-token cost accounting.
func TestHubOptsForwardsCostRates(t *testing.T) {
	a := New(nil)
	cfg := provider.NewRequestConfig(
		provider.WithModel("Qwen/Qwen3.8-Instruct-AWQ"),
		provider.WithCost(2.5, 10),
	)

	hubCfg := llmhub.NewConfig(a.hubOpts(cfg)...)
	if hubCfg.InputCostPerMillionTokens != 2.5 || hubCfg.OutputCostPerMillionTokens != 10 {
		t.Errorf("llmhub cost rates = %v/%v, want 2.5/10",
			hubCfg.InputCostPerMillionTokens, hubCfg.OutputCostPerMillionTokens)
	}
	if hubCfg.Model != "Qwen/Qwen3.8-Instruct-AWQ" {
		t.Errorf("llmhub model = %q, want the upstream alias id", hubCfg.Model)
	}

	// Unpriced requests must not set a zero cost rate that could clobber a
	// provider-reported one.
	empty := llmhub.NewConfig(a.hubOpts(provider.NewRequestConfig(provider.WithModel("m")))...)
	if empty.InputCostPerMillionTokens != 0 || empty.OutputCostPerMillionTokens != 0 {
		t.Errorf("unpriced request got cost rates %v/%v, want 0/0",
			empty.InputCostPerMillionTokens, empty.OutputCostPerMillionTokens)
	}
}

// 2. End to end through the adapter: a cost an upstream reports itself (e.g.
// OpenRouter's usage.cost) is carried into the IR response verbatim, and a
// stub that prices nothing yields a zero cost. The proxy's accounting layer
// back-fills that zero from the alias rates (see pkg/server trackUsage), which
// is what makes cost accounting work on lanes that never price themselves.
type costStubProvider struct{}

func (costStubProvider) Name() string { return "cost-stub" }

func (costStubProvider) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	return &llmhub.Response{
		ID:    "resp-1",
		Usage: llmhub.UsageMetadata{PromptTokens: 1000, CompletionTokens: 2000, TotalTokens: 3000, Cost: 0.0042},
	}, nil
}

func (costStubProvider) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	return nil, nil
}

func TestAdapterCarriesUsageCostThrough(t *testing.T) {
	resp, err := New(costStubProvider{}).Generate(context.Background(),
		[]*ir.Message{{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hi"}}}},
		provider.WithModel("m"),
		provider.WithCost(2, 5),
	)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Usage == nil {
		t.Fatalf("no usage on the IR response")
	}
	if resp.Usage.PromptTokens != 1000 || resp.Usage.CompletionTokens != 2000 {
		t.Errorf("usage tokens = %d/%d, want 1000/2000", resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}
	if resp.Usage.Cost != 0.0042 {
		t.Errorf("usage cost = %v, want the upstream-reported 0.0042", resp.Usage.Cost)
	}
}
