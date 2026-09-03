<p align="center">
  <img src="assets/ultiproxy-banner.svg" alt="Ultiproxy - One endpoint to rule your LLM subscriptions" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
  <a href="https://go.dev"><img src="https://img.shields.io/badge/Go-1.23-00ADD8.svg?logo=go" alt="Go 1.23"></a>
  <a href="#contributing"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  <img src="https://img.shields.io/badge/Port-:8317-22d3ee.svg" alt="Port 8317">
  <img src="https://img.shields.io/badge/MCP-Native-e879f9.svg" alt="MCP Native">
  <img src="https://img.shields.io/badge/Architecture-Linux%20%7C%20macOS-a3e635.svg" alt="Linux and macOS">
</p>

---

## Why Ultiproxy?

**The problem:** You pay for GitHub Copilot ($10/mo), ChatGPT Plus/Codex ($20/mo), Claude Pro ($20/mo), Google Antigravity, xAI Grok, DeepSeek, and Z.ai. That's $100+/month in flat-rate AI subscriptions.

Yet every time you spin up an autonomous coding agent—[OpenCode](https://opencode.ai), [Hermes](https://hermes-agent.dev), [Claude Code](https://github.com/anthropics/claude-code), [Cursor](https://cursor.com), or [Aider](https://aider.chat)—you are forced to supply raw, pay-as-you-go API keys that burn money per token. Meanwhile, your monthly subscriptions sit idle 85% of the day.

**The solution:** **Ultiproxy** pools all your subscriptions into **ONE local endpoint at `:8317`** that natively speaks both OpenAI (`POST /v1/chat/completions`) and Anthropic (`POST /v1/messages`).

- **Zero-Marginal-Cost Pooling**: Run agents 24/7 on your flat subscriptions. Stop paying per-token API bills for workloads you've already paid for.
- **Quota-Aware Intelligent Routing**: When Copilot exhausts its 5-hour rate window, Ultiproxy instantly cascades traffic to Google Antigravity, Codex, or DeepSeek without dropping agent context.
- **Native Embedded MCP Server**: AI agents can inspect `/mcp` directly to query live quotas, list model readiness, and toggle routing providers dynamically.
- **Per-API-Key Accounting**: Issue scoped keys (`sk-up-agent-a`, `sk-up-ci-worker`) with granular tracking of prompt tokens, completion tokens, cached tokens, and estimated cost savings.
- **Reasoning & Tool Passthrough**: Flawless streaming support for thinking tokens (`reasoning_content` in o1/DeepSeek-R1), tool calls, and multi-image vision inputs.
- **Agent-First, Zero GUI Bloat**: High-performance Go daemon. No Electron runtime, no web server bloat, sub-millisecond routing latency.

---

## Quickstart

### 1. One-Line Install

Installs `ultiproxy` to `~/.local/bin` (no root required) and sets up a hardened systemd user unit on Linux:

```bash
curl -fsSL https://ultiproxy.dev/install.sh | sh
```

*(You can inspect the installation script or run a preview using `bash dist/install.sh --dry-run`)*

### 2. 12-Line Configuration

Edit `~/.config/ultiproxy/config.yaml`:

```yaml
server:
  listen: "127.0.0.1:8317"
  api_keys: ["sk-up-local-agent-key"]
routing:
  strategy: "quota-priority"
providers:
  copilot: { enabled: true, token: "ghu_..." }
  openai: { enabled: true, api_key: "sk-proj-..." }
  anthropic: { enabled: true, api_key: "sk-ant-..." }
  vllm: { enabled: true, base_url: "http://127.0.0.1:8000/v1" }
accounting:
  enabled: true
```

Start the service:

```bash
# If using systemd (Linux):
systemctl --user enable --now ultiproxy

# Or run directly:
ultiproxy serve
```

Verify connectivity:
```bash
curl http://localhost:8317/healthz
```

---

## Copy-Paste Client Configurations

Point your favorite agents and editors directly at Ultiproxy:

### OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "providers": {
    "ultiproxy": {
      "name": "Ultiproxy Universal Gateway",
      "type": "openai",
      "baseURL": "http://localhost:8317/v1",
      "apiKey": "sk-up-local-agent-key",
      "models": [
        "gpt-4o",
        "claude-3-5-sonnet",
        "deepseek-r1"
      ]
    }
  }
}
```

### Hermes Agent

Add to your Hermes agent YAML configuration:

```yaml
llm:
  provider: "custom"
  base_url: "http://localhost:8317/v1"
  api_key: "sk-up-local-agent-key"
  model: "claude-3-5-sonnet"
  max_tokens: 8192
  temperature: 0.2
```

### Claude Code

Run Claude Code against your pooled subscriptions using the native Anthropic endpoint:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8317"
export ANTHROPIC_AUTH_TOKEN="sk-up-local-agent-key"

# Launch Claude Code
claude
```

### Cursor & OpenAI SDKs

In Cursor or standard OpenAI SDKs:

- **Base URL**: `http://localhost:8317/v1`
- **API Key**: `sk-up-local-agent-key`

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://localhost:8317/v1",
  apiKey: "sk-up-local-agent-key",
});

const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Implement quota-aware load balancer in Go." }],
  stream: true,
});
```

---

## Supported Providers (11 + Local)

Ultiproxy fronts 11 cloud subscription providers and local self-hosted runtimes:

| Provider | Subscription Needed | Notes |
| :--- | :--- | :--- |
| **GitHub Copilot** | Pro / Business / Enterprise | GPT-4o, Claude 3.5 Sonnet, o1; automatic GitHub token refresh |
| **Google Antigravity** | Antigravity / Gemini Advanced | Gemini 1.5 Pro, Flash; per-model-group quota buckets |
| **OpenAI Codex / Plus** | ChatGPT Plus / Team / API | GPT-4o, o1-preview, o1-mini; 5-hour & weekly sliding windows |
| **xAI Grok** | X Premium+ / xAI Console | Grok 2, Grok Beta; ultra-low latency |
| **Freebuff** | Freebuff Account | Multi-provider mirror inference backends |
| **Z.ai Coding Plan** | Z.ai Coding Subscription | GLM-4, DeepSeek coder; tracks dominant 5-hour quota |
| **Augure AI** | Augure Subscription | Specialized coding agent models |
| **DeepSeek** | DeepSeek Balance / VIP | DeepSeek-V3, DeepSeek-R1; full reasoning chain passthrough |
| **OpenRouter** | OpenRouter Credits / Tier | Fallback gateway routing across 200+ models |
| **Anthropic** | Claude Pro / Team Console | Claude 3.5 Sonnet, Claude 3 Opus, Haiku |
| **OpenCode Go** | OpenCode Membership | Optimized coding pipeline backend |
| **Local vLLM** | None (Local GPU / Hardware) | Llama 3, Qwen 2.5 Coder, Mistral; zero latency & zero marginal cost |

---

## Embedded MCP Server

Ultiproxy embeds a full Model Context Protocol (MCP) server over **Streamable HTTP** (`/mcp`) and **SSE** (`/mcp/sse`), enabling autonomous AI agents to introspect and manage their own model access:

### Exposed Tools

1. **`list_models`**: Returns all available models, context lengths, and active upstream health.
2. **`get_quota_status`**: Queries remaining request windows and time-to-reset across pooled subscriptions.
3. **`toggle_model`**: Dynamically enables or disables specific upstream providers at runtime.
4. **`get_client_usage`**: Inspects client token consumption and remaining quota allowances.
5. **`initiate_oauth_login`**: Triggers OAuth login / device token refresh without touching config files.

Configure MCP in Claude Desktop or your agent's MCP config:

```json
{
  "mcpServers": {
    "ultiproxy": {
      "url": "http://localhost:8317/mcp",
      "headers": {
        "Authorization": "Bearer sk-up-local-agent-key"
      }
    }
  }
}
```

---

## Replacing `quota.fjkl.cc`

Ultiproxy is a 100% drop-in replacement for the legacy `quota.fjkl.cc` dashboard and `dashboard-quota-probe.py` probe scripts.

- **`GET /api/quota`**: Serves the exact JSON shape required by existing web dashboards, router probes, and monitoring jobs.
- **`GET /quota.txt` & `GET /quota.md`**: Text and Markdown formatted quota reports for terminal inspection and LLM context loading.
- **`GET /llms.txt`**: Machine-readable specification adhering to the llms.txt standard for automated agent discovery.

---

## Security & Architecture

- **Single Key Surface**: Clients communicate with Ultiproxy via a single client key (`sk-up-...`). Upstream subscription credentials never leak to clients or network traces.
- **Per-API-Key Accounting**: Track token usage by agent, workload, or team member via `GET /api/stats/by-client`.
- **System Hardening**: The systemd user service (`dist/ultiproxy.service`) enforces strict process boundaries (`ProtectSystem=strict`, `NoNewPrivileges=true`, isolated `StateDirectory=ultiproxy`).

---

## Contributing

Pull requests are welcome! Ensure you adhere to the project standards:
- API changes must update `docs/openapi.yaml` and `llms.txt`.
- Verification gates must pass before submitting PRs.

## License

MIT © 2026 Ultiproxy Contributors.
