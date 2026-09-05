<p align="center">
  <img src="assets/ultiproxy-banner.svg" alt="Ultiproxy - One endpoint to rule your LLM subscriptions" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
  <a href="https://go.dev"><img src="https://img.shields.io/badge/Go-1.25-00ADD8.svg?logo=go" alt="Go 1.25"></a>
  <a href="#contributing"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  <img src="https://img.shields.io/badge/Port-:9050-22d3ee.svg" alt="Port 9050">
  <img src="https://img.shields.io/badge/MCP-Native-e879f9.svg" alt="MCP Native">
  <img src="https://img.shields.io/badge/Zero%20Config-22c55e.svg" alt="Zero Config">
  <img src="https://img.shields.io/badge/Architecture-Linux%20%7C%20macOS-a3e635.svg" alt="Linux and macOS">
</p>

---

## Why Ultiproxy?

**The problem:** You pay for GitHub Copilot ($10/mo), ChatGPT Plus/Codex ($20/mo), Claude Pro ($20/mo), Google Antigravity, xAI Grok, DeepSeek, and Z.ai. That's $100+/month in flat-rate AI subscriptions.

Yet every time you spin up an autonomous coding agent--[OpenCode](https://opencode.ai), [Hermes](https://hermes-agent.dev), [Claude Code](https://github.com/anthropics/claude-code), [Cursor](https://cursor.com), or [Aider](https://aider.chat)--you are forced to supply raw, pay-as-you-go API keys that burn money per token. Meanwhile, your monthly subscriptions sit idle 85% of the day.

**The solution:** **Ultiproxy** pools all your subscriptions into **ONE local endpoint at `:9050`** that natively speaks both OpenAI (`POST /v1/chat/completions`) and Anthropic (`POST /v1/messages`) -- **and exposes its entire control plane over MCP**, so an agent can wire it up itself.

- **Zero-Config Startup**: `ultiproxy serve` is the whole installation step. Nothing to author, nothing to edit, nothing to reload: lanes, aliases and timeouts are added at runtime over MCP and persist in the daemon's data dir (`providers.json`, `aliases.json`, `timeouts.json`).
- **MCP-Native Control Plane**: every knob an operator would touch is an MCP tool at `http://localhost:9050/mcp` -- `add_provider`, `set_model_alias`, `set_provider_timeout`, `get_quota_status`, `initiate_oauth_login`, ... An agent can bootstrap the whole gateway on its own. There is no separate admin CLI to install or remember: the daemon binary only serves, and MCP is the sole control plane.
- **Zero-Marginal-Cost Pooling**: Run agents 24/7 on your flat subscriptions. Stop paying per-token API bills for workloads you've already paid for.
- **Quota Monitoring, Not Quota Routing**: live per-provider quota windows and credit state via the MCP `get_quota_status` tool, `/api/quota`, and `quota.txt`. Operators and agents decide routing; the proxy never reroutes on quota by itself.
- **Per-API-Key Accounting**: scoped client keys (`sk-up-agent-a`, `sk-up-ci-worker`) with granular tracking of prompt tokens, completion tokens, cached tokens, requests and cost.
- **Reasoning & Tool Passthrough**: flawless streaming for thinking tokens (`reasoning_content` in o1/DeepSeek-R1), tool calls, and multi-image vision inputs.
- **Agent-First, Zero GUI Bloat**: a single high-performance Go daemon. No Electron runtime, no dashboard to click through, sub-millisecond routing latency.

---

## Agent-First Quickstart

Ultiproxy starts with **zero configuration**: nothing to author, nothing to reload, and nothing that has to be in place before the first request. You boot the daemon, then you (or your agent) configure everything else over MCP while it runs.

### 0. Install (once)

```bash
# Precompiled binary into ~/.local/bin (no root required):
curl -fsSL https://ultiproxy.dev/install.sh | sh

# ...or build from source (see the sibling-checkout note below):
git clone https://github.com/smhanov/ultiproxy
cd ultiproxy
go build -o ultiproxy ./cmd/ultiproxy
```

The installer also installs a hardened systemd user unit on Linux. Review it first with `bash dist/install.sh --dry-run` if you prefer.

<details>
<summary><strong>Building from source: the <code>../llmhub</code> sibling checkout</strong></summary>

`go.mod` resolves the provider library through a local replace:

```
replace github.com/smhanov/llmhub => ../llmhub
```

A source build therefore **requires a sibling `llmhub` checkout next to this repository** -- `go build ./cmd/ultiproxy` fails with a module-resolution error if `../llmhub` is missing:

```bash
git clone https://github.com/smhanov/ultiproxy
git clone https://github.com/smhanov/llmhub      # must sit beside ultiproxy
cd ultiproxy
go build -o ultiproxy ./cmd/ultiproxy
```

Keep the sibling `llmhub` on its `origin/main`; upstream changes go through PRs (see `AGENTS.md`). Building Go **1.25** or newer is required (`go.mod` declares `go 1.25.0`). Release binaries have no such dependency.

</details>

### 1. Start the daemon

```bash
# Directly:
ultiproxy serve

# Or as a hardened systemd user service (what the installer sets up on Linux):
systemctl --user enable --now ultiproxy

# Or bound to all interfaces for remote agents, still with no configuration:
ULTIPROXY_ADDR=0.0.0.0:9050 ultiproxy serve
```

Verify it is alive:

```bash
curl http://localhost:9050/healthz     # {"status":"ok"}
```

By default the daemon listens on `127.0.0.1:9050` and accepts every request, which is the right posture for a localhost-only install. Declaring client keys turns on Bearer authentication and gives you per-key attribution; upstream credentials are never exposed to clients either way.

### 2. Connect your agent to the embedded MCP server

Streamable HTTP at `http://localhost:9050/mcp` (legacy SSE at `http://localhost:9050/mcp/sse`):

```json
{
  "mcpServers": {
    "ultiproxy": {
      "url": "http://localhost:9050/mcp",
      "headers": { "Authorization": "Bearer sk-up-local-agent-key" }
    }
  }
}
```

Raw JSON-RPC works too, which is handy for scripts:

```bash
curl -s http://localhost:9050/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### 3. Register upstream providers

Subscription lanes log in over OAuth; API lanes take a key:

```bash
curl -s http://localhost:9050/mcp -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","id":2,"method":"tools/call",
  "params":{"name":"add_provider","arguments":{
    "name":"deepseek","base_url":"https://api.deepseek.com/v1","api_key":"sk-..."
  }}}'

curl -s http://localhost:9050/mcp -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","id":3,"method":"tools/call",
  "params":{"name":"add_provider","arguments":{"name":"antigravity","kind":"antigravity"}}}'
```

For OAuth lanes, drive the flow instead of pasting keys:

```bash
# 1) get the sign-in URL (and user_code for device flows such as xai)
{"name":"initiate_oauth_login","arguments":{"provider":"antigravity"}}
# 2a) auth-code flow: copy the code=... value out of the browser callback URL
{"name":"submit_oauth_code","arguments":{"provider":"antigravity","code":"4/0A..."}}
# 2b) device flow: poll until it answers "completed"
{"name":"check_oauth_login","arguments":{"provider":"xai"}}
```

### 4. Map client-visible model names

```bash
curl -s http://localhost:9050/mcp -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","id":4,"method":"tools/call",
  "params":{"name":"set_model_alias","arguments":{
    "alias":"qwenpoint-3.8",
    "provider":"vllm",
    "upstream":"Qwen/Qwen3.8-Instruct-AWQ",
    "context_limit":131072,
    "max_output":8192,
    "pricing_tag":"local-gpu"
  }}}'
```

Aliases, lanes and timeouts all survive a restart: the daemon re-reads its data dir and rebuilds the registry, so nothing has to be re-applied.

### 5. Point your agents at the endpoint

- **OpenAI clients**: `http://localhost:9050/v1`
- **Anthropic clients**: `http://localhost:9050`

---

## MCP Tools

Everything below is available on `tools/list` at `http://localhost:9050/mcp`, with full self-describing docstrings -- an agent never needs this README to operate the proxy.

| Tool | What it does |
| :--- | :--- |
| `list_models` | Client-visible model ids with their lane, enabled state, context limit, max output, pricing tag, capabilities and source (`alias` / `discovery` / `default`). Same id set as `GET /v1/models`, so the two surfaces cannot drift; ids disabled with `toggle_model` stay listed with `enabled: false`. |
| `get_quota_status` | Real-time quota/credit status for one lane (`antigravity`, `xai`, `copilot`, `codex`, `freebuff`, `zai`, ...): used %, remaining, units, reset times. Monitoring only -- never auto-reroutes. |
| `toggle_model` | Enable/disable a model id at runtime without deleting its mapping. |
| `get_client_usage` | Token (prompt, completion, cached) and request usage per client key or overall, over a window such as `1h`, `24h`, `7d`. |
| `initiate_oauth_login` | Non-blocking OAuth start for subscription lanes; returns the sign-in URL and the device `user_code`. |
| `check_oauth_login` | Poll a pending device-flow login (e.g. `xai`) or complete a server-side token exchange. |
| `submit_oauth_code` | Submit the authorization code from the redirect callback URL (e.g. `antigravity`). |
| `list_model_aliases` | Alias -> provider lane + upstream model id, with limits and pricing. |
| `set_model_alias` | Create/update an alias mapping (provider, upstream, optional `context_limit`, `max_output`, `pricing_tag`, `input_cost`, `output_cost`); persists in `aliases.json`. |
| `remove_model_alias` | Delete an alias mapping by name. |
| `get_provider_timeouts` | Per-lane timeout durations in force. |
| `set_provider_timeout` | Configure a lane timeout (Go duration string: `10m`, `3m30s`, ...); persists in `timeouts.json`. |
| `remove_provider_timeout` | Reset a lane to the server default timeout (120s). |
| `add_provider` | Register a lane at runtime -- OpenAI-compatible (`name`, `base_url`, optional `api_key`, quirks) or custom-wire kinds (`antigravity`, `anthropic`, `codex`, `freebuff`); persists in `providers.json`. The reply reports how many upstream models the lane serves ("discovered N models") -- discovery already ran. |
| `remove_provider` | Unregister a lane from the registry and from `providers.json`. |
| `list_providers` | Runtime-registered and compiled/in-memory lanes, secrets redacted. |
| `refresh_models` | Re-fetch `GET <base>/v1/models` and cache it so `<lane>/<model>` ids appear in `/v1/models`. Manual override -- discovery also runs at registration, at startup for lanes whose cache is empty, and every 6h afterwards. |

---

## Point Your Clients at Ultiproxy

### OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "providers": {
    "ultiproxy": {
      "name": "Ultiproxy Universal Gateway",
      "type": "openai",
      "baseURL": "http://localhost:9050/v1",
      "apiKey": "sk-up-local-agent-key",
      "models": [
        "gpt-4o",
        "claude-3-5-sonnet",
        "deepseek-r1",
        "vllm/qwenpoint-3.8"
      ]
    }
  }
}
```

### Hermes Agent

Give Hermes the OpenAI-compatible base URL and a model id that exists on the proxy:

```json
{
  "llm": {
    "provider": "custom",
    "base_url": "http://localhost:9050/v1",
    "api_key": "sk-up-local-agent-key",
    "model": "claude-3-5-sonnet",
    "max_tokens": 8192,
    "temperature": 0.2,
    "extra_headers": { "X-Client-Id": "hermes-agent-primary" }
  }
}
```

### Claude Code

Run Claude Code against your pooled subscriptions using the native Anthropic endpoint:

```bash
export ANTHROPIC_BASE_URL="http://localhost:9050"
export ANTHROPIC_AUTH_TOKEN="sk-up-local-agent-key"

# Launch Claude Code
claude
```

### Cursor

Settings -> Models -> add an OpenAI-compatible provider:

- **Base URL**: `http://localhost:9050/v1`
- **API Key**: `sk-up-local-agent-key`

### OpenAI SDKs (and anything else OpenAI-shaped)

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://localhost:9050/v1",
  apiKey: "sk-up-local-agent-key",
});

const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Implement quota-aware load balancer in Go." }],
  stream: true,
});
```

```bash
curl -s http://localhost:9050/v1/chat/completions \
  -H "Authorization: Bearer sk-up-local-agent-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Ping"}]}'
```

---

## HTTP Surface

| Route | Purpose |
| :--- | :--- |
| `POST /v1/chat/completions` | OpenAI chat completions (text, tools, images, SSE streaming). |
| `POST /v1/messages` | Anthropic Messages API (Claude Code and friends). |
| `GET /v1/models` | Only routable model ids: aliases (with `context_length` / `max_model_len` / `max_output_tokens`), `<lane>/<model>` per discovered upstream model, and `<lane>/<default>` for lanes with a default model (e.g. `antigravity/gemini-3.7-flash-high`). `max_model_len` is the vLLM-compatible name for the same window as `context_length` (from alias `context_limit` or upstream `max_model_len`). No bare lane names -- a lane name is a routing prefix, not a model (`"model": "<lane>"` still routes, as a legacy form). A lane with an empty discovery cache and no default lists nothing. Served from the discovery cache only -- listing never fans out to an upstream; set `ULTIPROXY_HIDE_TEST_LANES=1` to keep test lanes (`probe`, `fake`) out. |
| `POST /mcp`, `GET /mcp` | Streamable HTTP MCP server (JSON-RPC 2.0). |
| `GET /mcp/sse` | Legacy SSE MCP transport. |
| `GET /api/quota`, `/quota.txt`, `/quota.md` | Live quota dashboard, plain-text and Markdown views. |
| `GET /api/stats/summary` | Aggregated tokens, requests and estimated cost savings. |
| `GET /healthz`, `GET /llms.txt` | Health probe and machine-readable overview for LLMs. |

---

## Supported Providers (11 + Local)

Ultiproxy fronts 11 cloud subscription providers and local self-hosted runtimes. Lanes are added at runtime over MCP -- OAuth subscription lanes through `initiate_oauth_login`, keyed API lanes through `add_provider`:

| Provider | Subscription Needed | Notes |
| :--- | :--- | :--- |
| **GitHub Copilot** | Pro / Business / Enterprise | GPT-4o, Claude 3.5 Sonnet, o1; automatic GitHub token refresh |
| **Google Antigravity** | Antigravity / Gemini Advanced | Gemini 1.5 Pro, Flash; per-model-group quota buckets |
| **OpenAI Codex / Plus** | ChatGPT Plus / Team / API | GPT-4o, o1-preview, o1-mini; 5-hour & weekly sliding windows |
| **xAI Grok** | X Premium+ / xAI Console | Grok 2, Grok Beta; ultra-low latency; device-flow login |
| **Freebuff** | Freebuff Account | Multi-provider mirror inference backends |
| **Z.ai Coding Plan** | Z.ai Coding Subscription | GLM-4, DeepSeek coder; tracks dominant 5-hour quota |
| **Augure AI** | Augure Subscription | Specialized coding agent models |
| **DeepSeek** | DeepSeek Balance / VIP | DeepSeek-V3, DeepSeek-R1; full reasoning chain passthrough |
| **OpenRouter** | OpenRouter Credits / Tier | Fallback gateway routing across 200+ models |
| **Anthropic** | Claude Pro / Team Console | Claude 3.5 Sonnet, Claude 3 Opus, Haiku |
| **OpenCode Go** | OpenCode Membership | Optimized coding pipeline backend |
| **Local vLLM** | None (Local GPU / Hardware) | Llama 3, Qwen 2.5 Coder, Mistral; zero latency & zero marginal cost |

Any other OpenAI-compatible upstream (OpenRouter-style gateways, llama.cpp, TGI, another proxy) is one `add_provider` call away.

---

## Security & Architecture

- **Single Key Surface**: Clients talk to Ultiproxy with one client key (`sk-up-...`). Upstream subscription credentials and OAuth refresh tokens stay inside the daemon's credential store and never reach a client or a network trace.
- **Localhost by Default**: the daemon binds `127.0.0.1:9050`; `ULTIPROXY_ADDR` is the only switch needed to expose it to remote agents.
- **Per-API-Key Accounting**: track token usage by agent, workload or team member via `get_client_usage` and `GET /api/stats/summary`.
- **State Belongs to the Daemon**: runtime changes made over MCP are persisted atomically under the data dir (`providers.json`, `aliases.json`, `timeouts.json`) and restored on the next start -- no restart ritual, no drift between what you asked for and what runs.
- **System Hardening**: the systemd user service (`dist/ultiproxy.service`) enforces strict process boundaries (`ProtectSystem=strict`, `NoNewPrivileges=true`, isolated `StateDirectory=ultiproxy`).
- **Honest Routing**: an unknown model is a 404 `unknown_model`, never a silent detour to another vendor; tools against a lane without tool support is a 409 `model_does_not_support_tools`. Failover happens only before the first byte.

---

## Contributing

Pull requests are welcome! Ensure you adhere to the project standards:
- API changes must update `docs/openapi.yaml` and `llms.txt`.
- MCP tool docstrings in `pkg/mcp/tools.go` are the agent-facing documentation -- keep them complete and self-contained.
- Verification gates must pass before submitting PRs (`go build ./...`, `go test ./...`, `go vet ./...`).

## License

MIT (c) 2026 Ultiproxy Contributors.
