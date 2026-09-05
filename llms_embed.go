// Package ultiproxy is the module root package. It exists so the canonical
// llms.txt can be compiled into the binary: go:embed cannot reference paths
// outside the directory that contains the embedding Go file, so the embed
// directive has to live beside llms.txt at the repository root.
package ultiproxy

import _ "embed"

// llms.txt is the agent-facing documentation served at GET /llms.txt. The
// server prefers an llms.txt found on disk (operator override) and falls back
// to these bytes when the file is absent, which is the normal case for an
// installed binary that does not run from a source checkout.
//
//go:embed llms.txt
var llmsTxt []byte

// LLMsTxt returns the llms.txt bytes embedded at build time.
func LLMsTxt() []byte {
	if len(llmsTxt) == 0 {
		return nil
	}
	out := make([]byte, len(llmsTxt))
	copy(out, llmsTxt)
	return out
}
