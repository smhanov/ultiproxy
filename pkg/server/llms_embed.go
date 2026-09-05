package server

import embeddocs "github.com/smhanov/ultiproxy"

// embeddedLLMsTxt returns the llms.txt documentation that is compiled into
// the binary. It is the fallback for GET /llms.txt when the configured
// filesystem path is missing, so a packaged/systemd install that does not run
// from a source checkout still serves the documented discovery endpoint
// instead of a 404.
func embeddedLLMsTxt() []byte {
	return embeddocs.LLMsTxt()
}
