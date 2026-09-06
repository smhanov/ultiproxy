package modelmeta

// compiled is the cited seed catalog. Sources are primary vendor
// documentation; numbers are never derived from a model name, and ids whose
// upstream reports its own window on GET /v1/models (vLLM, OpenRouter,
// Anthropic, ...) deliberately have no row: live discovery wins over this
// catalog, so a row here could only shadow a served window.
//
// The seed covers the ids the operator's published alias table
// (cmd/ultiproxy/config.example.yaml) documents for lanes whose model list
// carries no metadata at all: the z.ai coding plan, the antigravity consumer
// subscription, GitHub Copilot and xAI. Everything else stays omitted until an
// operator adds data_dir/windows.json or a later cited seed lands.
var compiled = []Entry{
	// Z.ai coding plan (GLM). The coding-plan models endpoint lists ids only.
	{
		ID:               "glm-5.3",
		ContextLength:    1000000,
		MaxOutput:        131072,
		InputModalities:  []string{ModalityText},
		OutputModalities: []string{ModalityText},
		Source:           "https://docs.z.ai/guides/coding-plan",
	},
	{
		ID:               "glm-5.3-flash",
		ContextLength:    1000000,
		MaxOutput:        131072,
		InputModalities:  []string{ModalityText},
		OutputModalities: []string{ModalityText},
		Source:           "https://docs.z.ai/guides/coding-plan",
	},
	// xAI Grok Build / subscription models.
	{
		ID:               "grok-4.6",
		ContextLength:    131072,
		MaxOutput:        8192,
		InputModalities:  []string{ModalityText},
		OutputModalities: []string{ModalityText},
		Source:           "https://docs.x.ai/docs/models",
	},
	// Google consumer subscription (antigravity lane).
	{
		ID:               "gemini-3.8-flash-high",
		ContextLength:    1048576,
		MaxOutput:        65536,
		InputModalities:  []string{ModalityText, ModalityImage},
		OutputModalities: []string{ModalityText},
		Source:           "https://ai.google.dev/gemini-api/docs/models",
	},
	{
		ID:               "gemini-3.7-flash-high",
		ContextLength:    1048576,
		MaxOutput:        65536,
		InputModalities:  []string{ModalityText, ModalityImage},
		OutputModalities: []string{ModalityText},
		Source:           "https://ai.google.dev/gemini-api/docs/models",
	},
	// Claude models reached through non-Anthropic lanes (antigravity,
	// copilot). The anthropic lane discovers its own numbers live.
	{
		ID:               "claude-sonnet-4-6",
		ContextLength:    200000,
		MaxOutput:        65536,
		InputModalities:  []string{ModalityText, ModalityImage},
		OutputModalities: []string{ModalityText},
		Source:           "https://docs.anthropic.com/en/docs/about-claude/models/overview",
	},
	{
		ID:               "claude-haiku-4.5",
		ContextLength:    200000,
		MaxOutput:        32768,
		InputModalities:  []string{ModalityText, ModalityImage},
		OutputModalities: []string{ModalityText},
		Source:           "https://docs.anthropic.com/en/docs/about-claude/models/overview",
	},
}

// Default returns the compiled seed catalog.
func Default() *Catalog {
	c, err := New(compiled)
	if err != nil {
		// The seed is a compile-time constant table; a bad row is a bug, not
		// a runtime condition.
		panic(err)
	}
	return c
}
