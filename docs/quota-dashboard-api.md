# Quota Dashboard API Contract (quota.fjkl.cc compatibility)

Ultiproxy's `/api/quota` endpoint MUST return the exact JSON shape served by the legacy
dashboard at https://quota.fjkl.cc (source: ~/ai-quota-dashboard/app.py) so the existing
frontend, MCP resource, and coding-router probes (`dashboard-quota-probe.py`) keep working
unchanged.

## Top-level shape

```json
{
  "providers": [ /* ProviderSummary ... */ ],
  "summary": {
    "total": 14,
    "ok": 10,
    "unavailable": 4,
    "next_reset": { "seconds": 1800, "provider": "OpenAI Codex", "reset_at": "2026-09-03T14:00:00+00:00" },
    "fetched_at": "12:00:00",
    "fetched_at_epoch": 1788440000,
    "stale": false,
    "age_seconds": 5,
    "refreshing": false,
    "refresh_started_at": null,
    "refresh_min_interval": 30,
    "last_refresh_error": null
  }
}
```

## Provider summary object

```json
{
  "id": "copilot",
  "name": "GitHub Copilot",
  "plan": "Pro (annual)",
  "status": "ok",
  "gauge_pct": 42.5,
  "windows": [
    {
      "label": "Premium requests",
      "used_pct": 42.5,
      "reset_at": "2026-09-03T14:00:00+00:00",
      "seconds_remaining": 3600
    }
  ],
  "bars": [
    {
      "label": "Premium requests",
      "used": 425,
      "limit": 1000,
      "remaining": 575,
      "unit": "requests",
      "pct": 42.5
    }
  ],
  "updated": "2026-09-03T16:00:00+00:00",
  "detail": "Premium 57.5% remaining · resets 2026-09-03 · chat+completions included/unlimited",
  "extra": {}
}
```

## Semantics

- `status`: one of `ok` | `unavailable` | `error`.
- `gauge_pct`: 0-100 (worst/most-used window; for Z.ai the DOMINANT window, not the
  freshest).
- `windows[].label`: human label; for OpenAI "5 hour" vs "Weekly"; Z.ai
  "5-hour"/"Weekly"; Copilot "Premium requests"; Antigravity is per-model-group buckets.
- `bars[].unit`: `%` or `credits` or `requests`.
- `summary.next_reset`: earliest `seconds_remaining` across all ok providers.
- `summary.stale` true when cache age >= CACHE_TTL (60s for quota.fjkl.cc).
- `summary.refreshing` true while a refresh is in-flight; `refresh_min_interval` 30s.

## Consumer notes (what reads /api/quota)

- `dashboard-quota-probe.py <provider>` — reads `providers[].status` and windows for
  ROUTER_*_QUOTA_CMD.
- Quota dashboard frontend (`static/index.html`) — renders cards from `providers`.
- `getLlmAgentProviders` MCP resource — `mcp_resource_quotas` returns the markdown-formatted
  `/quota.md` view.

Ultiproxy must serve: `/api/quota`, `/quota.txt`, `/quota.md`, `/llms.txt`, `/healthz`.