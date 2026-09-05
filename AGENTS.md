# Agent notes — ultiproxy

## Task tracking: plans/todo.md

`plans/todo.md` is the canonical local queue for future/outstanding work. It is intentionally compact; implementation detail belongs in the linked plan file.

**Before starting work:**
- Read `plans/todo.md` and the linked plan. If the user did not name a task, take the first `ready` row in queue order.
- Change that row to `in-progress` before editing code. Never work a `blocked` task until its dependencies are complete.
- If the requested work is not already listed, create a detailed `plans/<date>-<slug>.md` plan, assign the next unused permanent task ID, and add one concise Active row.
- Every implementation plan must contain an **Acceptance Criteria** section before coding starts. Acceptance criteria define the required outcome, not the implementation: each criterion must be binary/falsifiable, externally observable where possible, and state how the agent will verify it (test, contract check, command/result, or live receipt).
- Acceptance criteria must cover the main success path plus relevant regression, error/negative, compatibility, persistence/restart, or concurrency cases for the change. For bug fixes, include a criterion that reproduces the old failure and proves the regression is fixed.
- `go test ./...` / build / vet passing may be required verification gates, but **"tests pass" alone is never sufficient acceptance criteria**. Do not use vague criteria such as "works correctly", "is robust", or "handles edge cases" without naming the observable behavior and evidence.
- Prefer behavior/invariant criteria over implementation-detail criteria. Do not force a specific internal design unless that design constraint is itself part of the requirement.
- A task is not complete until every acceptance criterion has concrete evidence. If a criterion cannot be verified, leave the task `in-progress` or `blocked` and record the reason in the plan instead of claiming completion.

**Test-first development for behavior changes (RED → GREEN → REFACTOR → VERIFY):**
- For bug fixes, protocol/behavior changes, persistence/lifecycle changes, concurrency changes, and nontrivial new functionality, write the smallest meaningful test for the acceptance criterion **before editing production code**.
- Run that focused test and observe it fail for the expected behavioral reason. A compile error, bad fixture, unrelated failure, or test that already passes does not count as RED. Never claim test-first development unless the failing result was actually observed before the production-code change.
- Only after RED is established, make the minimum production change needed to satisfy the behavior. Do not delete, weaken, skip, or rewrite the test merely to obtain GREEN; if the requirement/test is wrong, update the plan/acceptance criterion explicitly first.
- Refactor only after GREEN, keeping the focused tests green. Then VERIFY from narrow to broad: focused regression test → affected package/contract tests → repository-wide gates where appropriate. Use `go test -race` for concurrency/lifecycle work when practical.
- Tests verify behavior/invariants, not private implementation details. Prefer the highest useful observable boundary: wire/contract tests for protocol behavior, restart/readback tests for persistence, and bounded cancellation/concurrency tests for lifecycle bugs.
- Test-first is not required for documentation/comments, purely mechanical renames, deletions better proven by search/build, or genuinely trivial low-risk changes. Still provide the acceptance evidence appropriate to those changes.

**Before finishing work:**
- If verified complete, move the task from Active to Completed with the date and commit hash; use `working-tree` only when no commit was requested.
- If unfinished, leave it `in-progress`; if blocked, mark it `blocked` and name the blocking task ID(s).
- Never put long notes, logs, receipts, audit prose, or completed tasks in Active.
- Run `python3 scripts/validate_todo.py`. A task-tracker validation failure must be fixed before declaring the task complete.

Do not leave task state only in conversation. IDs are permanent and must never be reused.

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