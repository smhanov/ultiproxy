package mcp

import (
	"strings"
	"testing"
)

type overlayAliasManager struct {
	stubAliasManager
	overlay map[string]ModelMeta
	listed  []string
}

func (m *overlayAliasManager) ModelInfoEntry(id string) (ModelMeta, bool) {
	e, ok := m.overlay[id]
	return e, ok
}

func (m *overlayAliasManager) MergeModelInfo(id string, patch ModelMeta) error {
	if m.overlay == nil {
		m.overlay = map[string]ModelMeta{}
	}
	cur := m.overlay[id]
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
	m.overlay[id] = cur
	return nil
}

func (m *overlayAliasManager) ClearModelInfo(id string, fields []string) error {
	if len(fields) == 0 {
		delete(m.overlay, id)
		return nil
	}
	cur, ok := m.overlay[id]
	if !ok {
		return nil
	}
	for _, f := range fields {
		switch f {
		case "context_length", "max_model_len":
			cur.ContextLength = 0
		case "max_output_tokens", "max_output":
			cur.MaxOutput = 0
		case "input_modalities":
			cur.InputModalities = nil
		case "output_modalities":
			cur.OutputModalities = nil
		}
	}
	m.overlay[id] = cur
	return nil
}

func (m *overlayAliasManager) ListedIDs() []string {
	return m.listed
}

func TestSetModelInfo_HappyPathAndPartial(t *testing.T) {
	am := &overlayAliasManager{
		listed: []string{"lane/bare"},
		overlay: map[string]ModelMeta{},
	}
	srv := NewServer(nil, nil, WithAliasManager(am))

	// 1. First set: context_length + text modality
	res := callTool(t, srv, 1, "set_model_info", `{"id":"lane/bare","context_length":8192,"input_modalities":["text"]}`)
	if !strings.Contains(res, `"ok": true`) {
		t.Fatalf("first set failed: %s", res)
	}
	e := am.overlay["lane/bare"]
	if e.ContextLength != 8192 || len(e.InputModalities) != 1 || e.InputModalities[0] != "text" {
		t.Fatalf("overlay = %+v", e)
	}

	// 2. Partial update: only max_output_tokens, context_length stays
	res2 := callTool(t, srv, 2, "set_model_info", `{"id":"lane/bare","max_output_tokens":2048}`)
	if !strings.Contains(res2, `"ok": true`) {
		t.Fatalf("partial set failed: %s", res2)
	}
	e2 := am.overlay["lane/bare"]
	if e2.ContextLength != 8192 || e2.MaxOutput != 2048 {
		t.Fatalf("partial update lost context_length: %+v", e2)
	}

	// 3. Reject unlisted id
	errRes := callToolExpectingError(t, srv, 3, "set_model_info", `{"id":"not-listed","context_length":1000}`)
	if !strings.Contains(errRes, "not a listed model id") {
		t.Fatalf("expected not-listed error, got %s", errRes)
	}

	// 4. Reject zero context
	errZero := callToolExpectingError(t, srv, 4, "set_model_info", `{"id":"lane/bare","context_length":0}`)
	if !strings.Contains(errZero, "context_length 0 is rejected") {
		t.Fatalf("expected zero rejection, got %s", errZero)
	}

	// 5. Clear specific field
	resClear := callTool(t, srv, 5, "clear_model_info", `{"id":"lane/bare","fields":["context_length"]}`)
	if !strings.Contains(resClear, `"ok": true`) {
		t.Fatalf("clear failed: %s", resClear)
	}
	if am.overlay["lane/bare"].ContextLength != 0 || am.overlay["lane/bare"].MaxOutput != 2048 {
		t.Fatalf("clear context_length failed: %+v", am.overlay["lane/bare"])
	}

	// 6. Clear all
	_ = callTool(t, srv, 6, "clear_model_info", `{"id":"lane/bare"}`)
	if _, ok := am.overlay["lane/bare"]; ok {
		t.Fatalf("overlay not cleared")
	}
}

func callToolExpectingError(t *testing.T, srv *Server, id int, name, args string) string {
	t.Helper()
	res := callMCPTool(t, srv, id, name, args)
	if !res.IsError {
		t.Fatalf("%s expected error, got success: %v", name, res)
	}
	if len(res.Content) == 0 {
		t.Fatalf("%s returned no error content", name)
	}
	return res.Content[0].Text
}
