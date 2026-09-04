package hublane

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/smhanov/llmhub"
	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

type fakeHubProvider struct {
	name        string
	lastPrompt  []*llmhub.Message
	lastOpts    []llmhub.Option
	generateResp *llmhub.Response
	generateErr  error
	streamCh     chan llmhub.StreamChunk
	streamErr    error
}

func (f *fakeHubProvider) Name() string { return f.name }

func (f *fakeHubProvider) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	f.lastPrompt = prompt
	f.lastOpts = opts
	return f.generateResp, f.generateErr
}

func (f *fakeHubProvider) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	f.lastPrompt = prompt
	f.lastOpts = opts
	return f.streamCh, f.streamErr
}

func TestAdapter_Generate(t *testing.T) {
	hub := &fakeHubProvider{
		name: "fake-hub",
		generateResp: &llmhub.Response{
			ID:      "resp-1",
			Content: []llmhub.ContentPart{llmhub.Text("hi")},
			Usage:   llmhub.UsageMetadata{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3},
			Raw:     map[string]any{"finish_reason": "stop"},
		},
	}

	adapter := New(hub)
	if adapter.Name() != "fake-hub" {
		t.Fatalf("expected name fake-hub, got %s", adapter.Name())
	}

	msgs := []*ir.Message{
		{Role: "system", Blocks: []ir.Block{ir.TextBlock{Text: "sys"}}},
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hello"}}},
	}

	temp := 0.5
	tools := []llmhub.Tool{llmhub.NewTool("weather", "Get weather", nil)}
	resp, err := adapter.Generate(context.Background(), msgs,
		provider.WithModel("model-x"),
		provider.WithMaxTokens(100),
		provider.WithTemperature(temp),
		provider.WithHeader("X-Custom", "value"),
		provider.WithExtraBody(map[string]any{"tools": tools}),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify prompt translation.
	if len(hub.lastPrompt) != 2 {
		t.Fatalf("expected 2 hub messages, got %d", len(hub.lastPrompt))
	}
	if hub.lastPrompt[0].Role != llmhub.RoleSystem {
		t.Fatalf("expected system role, got %s", hub.lastPrompt[0].Role)
	}
	if hub.lastPrompt[1].Role != llmhub.RoleUser {
		t.Fatalf("expected user role, got %s", hub.lastPrompt[1].Role)
	}

	// Verify options are mapped correctly.
	gotCfg := llmhub.NewConfig(hub.lastOpts...)
	wantCfg := llmhub.NewConfig(
		llmhub.WithModel("model-x"),
		llmhub.WithMaxTokens(100),
		llmhub.WithTemperature(temp),
		llmhub.WithHeader("X-Custom", "value"),
		llmhub.WithTools(tools...),
	)
	if !reflect.DeepEqual(gotCfg, wantCfg) {
		t.Fatalf("config mismatch:\n got: %+v\n want: %+v", gotCfg, wantCfg)
	}

	// Verify response translation.
	if resp == nil || resp.ID != "resp-1" {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if resp.FinishReason != "stop" {
		t.Fatalf("expected finish_reason stop, got %q", resp.FinishReason)
	}
	if resp.Message == nil || len(resp.Message.Blocks) != 1 {
		t.Fatalf("unexpected message blocks: %+v", resp.Message)
	}
	if text, ok := resp.Message.Blocks[0].(ir.TextBlock); !ok || text.Text != "hi" {
		t.Fatalf("unexpected text block: %+v", resp.Message.Blocks[0])
	}
}

func TestAdapter_Generate_Timeout(t *testing.T) {
	hub := &fakeHubProvider{
		generateResp: &llmhub.Response{Content: []llmhub.ContentPart{llmhub.Text("ok")}},
	}
	adapter := New(hub)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	// The fake returns immediately, so the timeout should not fire.
	_, err := adapter.Generate(ctx, []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hello"}}},
	}, provider.WithTimeout(5*time.Second))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAdapter_Stream(t *testing.T) {
	streamCh := make(chan llmhub.StreamChunk, 4)
	streamCh <- llmhub.StreamChunk{ID: "s1", Delta: "Hello"}
	streamCh <- llmhub.StreamChunk{Delta: " world", FinishReason: "stop"}
	close(streamCh)

	hub := &fakeHubProvider{
		name:     "fake-hub",
		streamCh: streamCh,
	}
	adapter := New(hub)

	msgs := []*ir.Message{
		{Role: "user", Blocks: []ir.Block{ir.TextBlock{Text: "hello"}}},
	}

	temp := 0.3
	tools := []llmhub.Tool{llmhub.NewTool("calculator", "Do math", nil)}
	eventCh, err := adapter.Stream(context.Background(), msgs,
		provider.WithModel("model-y"),
		provider.WithTemperature(temp),
		provider.WithExtraBody(map[string]any{"tools": tools}),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify prompt translation.
	if len(hub.lastPrompt) != 1 {
		t.Fatalf("expected 1 hub message, got %d", len(hub.lastPrompt))
	}

	// Verify options.
	gotCfg := llmhub.NewConfig(hub.lastOpts...)
	wantCfg := llmhub.NewConfig(
		llmhub.WithModel("model-y"),
		llmhub.WithTemperature(temp),
		llmhub.WithTools(tools...),
	)
	if !reflect.DeepEqual(gotCfg, wantCfg) {
		t.Fatalf("config mismatch:\n got: %+v\n want: %+v", gotCfg, wantCfg)
	}

	// Verify streamed events.
	events := drainEvents(t, eventCh)
	want := []ir.Event{
		ir.EventMessageStart{ID: "s1"},
		ir.EventTextDelta{Text: "Hello"},
		ir.EventTextDelta{Text: " world"},
		ir.EventMessageStop{FinishReason: "stop"},
	}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events mismatch:\n got: %+v\n want: %+v", events, want)
	}
}

func TestAdapter_NilHub(t *testing.T) {
	var adapter *Adapter
	if _, err := adapter.Generate(context.Background(), nil); err != provider.ErrProviderNotFound {
		t.Fatalf("expected ErrProviderNotFound, got %v", err)
	}
	if _, err := adapter.Stream(context.Background(), nil); err != provider.ErrProviderNotFound {
		t.Fatalf("expected ErrProviderNotFound, got %v", err)
	}
	if adapter.Name() != "hublane" {
		t.Fatalf("expected default name hublane, got %s", adapter.Name())
	}

	empty := New(nil)
	if _, err := empty.Generate(context.Background(), nil); err != provider.ErrProviderNotFound {
		t.Fatalf("expected ErrProviderNotFound, got %v", err)
	}
	if _, err := empty.Stream(context.Background(), nil); err != provider.ErrProviderNotFound {
		t.Fatalf("expected ErrProviderNotFound, got %v", err)
	}
}

func TestAdapter_ProviderBundle(t *testing.T) {
	adapter := New(&fakeHubProvider{name: "hub"},
		WithCapabilities(provider.Capabilities{Chat: true, Tools: true}),
	)

	bundle := adapter.ProviderBundle()
	if bundle.Inference != adapter {
		t.Fatal("bundle inference mismatch")
	}
	if !bundle.Capabilities.Chat || !bundle.Capabilities.Tools {
		t.Fatalf("unexpected capabilities: %+v", bundle.Capabilities)
	}
}
