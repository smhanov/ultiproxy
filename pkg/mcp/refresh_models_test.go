package mcp

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/smhanov/ultiproxy/pkg/ir"
	"github.com/smhanov/ultiproxy/pkg/provider"
)

// fakeModelFetchLane is an InferenceProvider that also implements the
// optional FetchModels capability (as openaicompat lanes do).
type fakeModelFetchLane struct {
	name   string
	models []string
	err    error
	called bool
}

func (f *fakeModelFetchLane) Name() string { return f.name }

func (f *fakeModelFetchLane) Generate(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (*ir.Response, error) {
	return &ir.Response{}, nil
}

func (f *fakeModelFetchLane) Stream(ctx context.Context, msgs []*ir.Message, opts ...provider.Option) (<-chan ir.Event, error) {
	return nil, nil
}

func (f *fakeModelFetchLane) FetchModels(ctx context.Context) ([]string, error) {
	f.called = true
	return f.models, f.err
}

// TestMCPRefreshModels: the happy path caches the upstream model ids and
// reports them back, so a lane registered before startup discovery existed
// gets <lane>/<model> ids into /v1/models.
func TestMCPRefreshModels(t *testing.T) {
	registry := provider.NewRegistry()
	lane := &fakeModelFetchLane{name: "opencode", models: []string{"nemotron-3.5-lightning-free", "mimo-v2.5-free"}}
	registry.Register(provider.Provider{Inference: lane})

	srv := NewServer(registry, newStubStateSource())
	res := callMCPTool(t, srv, 1, "refresh_models", `{"name":"opencode"}`)
	if res.IsError {
		t.Fatalf("refresh_models failed: %s", res.Content[0].Text)
	}
	if !lane.called {
		t.Fatal("FetchModels was not called on the lane")
	}
	text := res.Content[0].Text
	if !strings.Contains(text, "2 models cached for lane opencode") {
		t.Errorf("result text = %q, want the cached model count", text)
	}
	for _, id := range lane.models {
		if !strings.Contains(text, id) {
			t.Errorf("result text = %q, want model id %q listed", text, id)
		}
	}
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) != 3 {
		t.Errorf("result text = %q, want 1 summary line + 2 model lines", text)
	}
}

// TestMCPRefreshModels_UnknownLane: an unregistered lane name is a tool error,
// not a JSON-RPC error or a panic.
func TestMCPRefreshModels_UnknownLane(t *testing.T) {
	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: &fakeModelFetchLane{name: "vllm"}})

	srv := NewServer(registry, newStubStateSource())
	res := callMCPTool(t, srv, 2, "refresh_models", `{"name":"ghost"}`)
	if !res.IsError {
		t.Fatalf("expected an error result, got %+v", res)
	}
	if want := "error: lane ghost not found"; res.Content[0].Text != want {
		t.Errorf("message = %q, want %q", res.Content[0].Text, want)
	}
}

// TestMCPRefreshModels_NoModelDiscovery: a lane whose Inference surface does
// not implement FetchModels (antigravity, codex, anthropic, ...) says so
// instead of pretending or 500ing.
func TestMCPRefreshModels_NoModelDiscovery(t *testing.T) {
	registry := provider.NewRegistry()
	registry.Register(provider.Provider{Inference: laneWithoutQuota{name: "codex"}})

	srv := NewServer(registry, newStubStateSource())
	res := callMCPTool(t, srv, 3, "refresh_models", `{"name":"codex"}`)
	if !res.IsError {
		t.Fatalf("expected an error result, got %+v", res)
	}
	if want := "error: lane codex does not support model discovery"; res.Content[0].Text != want {
		t.Errorf("message = %q, want %q", res.Content[0].Text, want)
	}
}

// TestMCPRefreshModels_UpstreamError: an upstream failure surfaces as a tool
// error containing the cause; the daemon is not wrapped in a 500.
func TestMCPRefreshModels_UpstreamError(t *testing.T) {
	registry := provider.NewRegistry()
	lane := &fakeModelFetchLane{name: "opencode", err: errors.New("models endpoint returned status 503")}
	registry.Register(provider.Provider{Inference: lane})

	srv := NewServer(registry, newStubStateSource())
	res := callMCPTool(t, srv, 4, "refresh_models", `{"name":"opencode"}`)
	if !res.IsError {
		t.Fatalf("expected an error result, got %+v", res)
	}
	if !strings.Contains(res.Content[0].Text, "error") {
		t.Errorf("message = %q, want it to be an error", res.Content[0].Text)
	}
	if !strings.Contains(res.Content[0].Text, "status 503") {
		t.Errorf("message = %q, want the upstream error text", res.Content[0].Text)
	}
}

// TestMCPRefreshModels_MissingArgument: no lane name is a tool error.
func TestMCPRefreshModels_MissingArgument(t *testing.T) {
	srv := NewServer(provider.NewRegistry(), newStubStateSource())
	res := callMCPTool(t, srv, 5, "refresh_models", `{}`)
	if !res.IsError {
		t.Fatalf("expected an error result, got %+v", res)
	}
	if !strings.Contains(res.Content[0].Text, "name argument is required") {
		t.Errorf("message = %q, want a missing-argument error", res.Content[0].Text)
	}
}
