package router

import (
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

func TestEstimateTokens(t *testing.T) {
	msgs := []*ir.Message{
		{
			Role: "system",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "You are a helpful assistant."},
			},
		},
		{
			Role: "user",
			Blocks: []ir.Block{
				ir.TextBlock{Text: "What is 2+2?"},
			},
		},
	}

	// Default output tokens is 1024
	estDefault := EstimateTokens(msgs)
	if estDefault <= 1024 {
		t.Errorf("expected est > 1024, got %d", estDefault)
	}

	// Override max tokens to 50
	estCustom := EstimateTokens(msgs, provider.WithMaxTokens(50))
	if estCustom >= 100 {
		t.Errorf("expected est < 100 with MaxTokens=50, got %d", estCustom)
	}
}
