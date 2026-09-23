# Agent notes — ultiproxy

## llmhub is a published module dependency

- ultiproxy consumes `github.com/smhanov/llmhub` as a normal published Go module pinned in `go.mod` (pseudo-version or tag). No extra checkouts, no `replace` directive.
- **Never push llmhub commits to llmhub's `main` directly.** Anything needed upstream (options, fixes, provider tweaks) goes through a GitHub PR: branch off `origin/main`, `gh pr create` with the rationale, then after the maintainer merges/reviews, bump the pin here (`go get github.com/smhanov/llmhub@<version>` + `go mod tidy`) and **adapt ultiproxy to whatever API shape actually landed** — do not assume your proposed API survived.
- Integration notes: `pkg/provider/openaicompat/openaicompat.go` builds its llmhub client with `llmhub.WithRetryOnStatus(http.StatusTooManyRequests, false)` so upstream 429s surface to clients honestly instead of being retried inside the HTTP layer (freebuff lanes additionally opt into 428 waiting-room retries). Preserve this honest-429 posture when bumping the module.

## Architecture summary (post-migration)

- **One OpenAI-compatible provider:** `pkg/provider/openaicompat` replaces all vendor openai-shaped lanes (zai, vllm, openrouter, deepseek, opencode, xai, augure, freebuff). Vendor difference = `Quirks` config, never a package. Custom wires stay separate: antigravity (CCPA), copilot (editor headers + `/responses`), codex (backend-api), anthropichub (Messages API).
- Routing: `pkg/server/router.go` `RegistryRouter` — unknown model → HTTP 404 `unknown_model` (no cross-vendor failover). Tools request to a lane with `Capabilities.Tools == false` → 409 `model_does_not_support_tools`. Failover happens only before the first byte; nothing after.
- Contract suite: `pkg/contract/opencode/` — end-to-end wire tests through the real stack; `wire_test.go` carries the openaicompat quirk matrix.

## Build & test

```bash
go build ./...
go test ./...          # full suite
go vet ./...           # clean (except pre-existing gofmt debt in some files)
```

Pre-existing gofmt violations that predate the migration (do not "fix" them as part of other tasks): `pkg/codec/openai.go`, `pkg/contract/opencode/fakeupstream.go`, `pkg/provider/hublane/adapter_test.go`, `pkg/provider/hublane/convert_test.go`.