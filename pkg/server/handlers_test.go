package server

import "testing"

func TestStripProviderPrefix(t *testing.T) {
	cases := []struct {
		model, provider, want string
	}{
		{"zai/glm-5.3-flash", "zai", "glm-5.3-flash"},
		{"ZAI/glm-5.3", "zai", "glm-5.3"},                                                 // case-insensitive prefix
		{"openrouter/meta-llama/llama-3.3-70b", "openrouter", "meta-llama/llama-3.3-70b"}, // one prefix, keep nested slash
		{"meta-llama/llama-3.3-70b", "openrouter", "meta-llama/llama-3.3-70b"},            // wrong prefix: keep as-is
		{"glm-5.3-flash", "zai", "glm-5.3-flash"},                                         // no slash
		{"", "zai", ""},
		{"claude-sonnet-4-6", "copilot", "claude-sonnet-4-6"},
	}
	for _, c := range cases {
		if got := stripProviderPrefix(c.model, c.provider); got != c.want {
			t.Errorf("stripProviderPrefix(%q, %q) = %q, want %q", c.model, c.provider, got, c.want)
		}
	}
}
