# Agent notes — ultiproxy

## Task tracking: plans/todo.md

Use the untracked `plans/todo.md` as the running list of future/outstanding tasks (status column: `todo` → `in-progress` → `done`, with a Done table carrying commit hashes). When you finish work or discover a new follow-up, update it — don't leave task state only in conversation.

## llmhub changes MUST go through PRs

- ultiproxy depends on `github.com/smhanov/llmhub` via a local `replace` (`go.mod`: `replace github.com/smhanov/llmhub => ../llmhub`).
- **Never push llmhub commits to llmhub's `main` directly.** Anything needed upstream (options, fixes, provider tweaks) goes through a GitHub PR: branch off `origin/main`, `gh pr create` with the rationale, then after the maintainer merges/reviews, **pull the merged result and adapt ultiproxy to whatever API shape actually landed** — do not assume your proposed API survived.
- llmhub's local `main` should stay in sync with `origin/main`; work on a PR branch so this `replace` keeps resolving.
- Current integration notes:
  - `openaicompat.New()` passes `llmhub.WithRetryOnStatus(http.StatusTooManyRequests, false)` for honest-429 semantics (maintainer's API; the old `WithNoRetryOn429` name no longer exists).
  - Migration plan: `plans/2026-09-03_212346-ultiproxy-llmhub-migration.md`.

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