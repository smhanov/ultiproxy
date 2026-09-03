CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    client_key_hash TEXT,
    logical_id TEXT,
    requested_model TEXT,
    resolved_model TEXT,
    provider TEXT,
    created_at TEXT,
    completed_at TEXT,
    finish_reason TEXT,
    stream_complete INTEGER,
    error_class TEXT,
    ttft_ms INTEGER,
    total_ms INTEGER
);

CREATE TABLE IF NOT EXISTS request_attempts (
    id INTEGER PRIMARY KEY,
    request_id INTEGER REFERENCES requests(id),
    attempt INTEGER,
    provider TEXT,
    model TEXT,
    upstream_request_id TEXT,
    status_code INTEGER,
    error_class TEXT,
    retry_after_seconds INTEGER,
    reset_at TEXT
);

CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY,
    request_id INTEGER REFERENCES requests(id),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens INTEGER,
    cached_tokens INTEGER,
    cost REAL
);

CREATE TABLE IF NOT EXISTS quota_observations (
    id INTEGER PRIMARY KEY,
    provider TEXT,
    label TEXT,
    used_pct REAL,
    remaining REAL,
    "limit" REAL,
    unit TEXT,
    reset_at TEXT,
    observed_at TEXT,
    source TEXT
);
