#!/usr/bin/env bash
set -euo pipefail

# Ultiproxy Installer
# Installs Ultiproxy to ~/.local/bin without root privileges.

REPO="smhanov/ultiproxy"
DEFAULT_RELEASE_URL="https://github.com/${REPO}/releases/latest/download"

DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      echo "Usage: install.sh [--dry-run]"
      echo "  --dry-run: Preview installation steps without executing changes"
      exit 0
      ;;
    *)
      ;;
  esac
done

# Detect OS
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "${OS}" in
  linux)
    TARGET_OS="linux"
    ;;
  darwin)
    TARGET_OS="darwin"
    ;;
  *)
    echo "Error: Unsupported operating system '${OS}'. Ultiproxy supports Linux and macOS (Darwin)." >&2
    exit 1
    ;;
esac

# Detect Architecture
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64|amd64)
    TARGET_ARCH="x86_64"
    ;;
  arm64|aarch64)
    TARGET_ARCH="arm64"
    ;;
  *)
    echo "Error: Unsupported CPU architecture '${ARCH}'. Ultiproxy supports x86_64 and arm64." >&2
    exit 1
    ;;
esac

TARBALL="ultiproxy-${TARGET_OS}-${TARGET_ARCH}.tar.gz"
DOWNLOAD_URL="${DEFAULT_RELEASE_URL}/${TARBALL}"
SHA256_URL="${DEFAULT_RELEASE_URL}/${TARBALL}.sha256"

INSTALL_DIR="${HOME}/.local/bin"
BINARY_PATH="${INSTALL_DIR}/ultiproxy"
CONFIG_DIR="${HOME}/.config/ultiproxy"
CONFIG_FILE="${CONFIG_DIR}/config.yaml"
STATE_DIR="${HOME}/.local/state/ultiproxy"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
SYSTEMD_SERVICE_FILE="${SYSTEMD_USER_DIR}/ultiproxy.service"

echo "=== Ultiproxy Installer ==="
echo "Target Platform : ${TARGET_OS}-${TARGET_ARCH}"
echo "Install Path    : ${BINARY_PATH}"
echo "Config Path     : ${CONFIG_FILE}"
echo "State Path      : ${STATE_DIR}"

if [ "${DRY_RUN}" -eq 1 ]; then
  echo ""
  echo "[DRY RUN] Would execute the following actions:"
  echo "1. Verify curl and sha256sum/shasum tools"
  echo "2. Download ${DOWNLOAD_URL} and ${SHA256_URL}"
  echo "3. Verify SHA256 checksum of release archive"
  echo "4. Create directory ${INSTALL_DIR} and extract binary to ${BINARY_PATH}"
  echo "5. Create a current-schema configuration (server.addr, server.api_key, data_dir, storage.db_path) at ${CONFIG_FILE}"
  echo "6. Create state directory at ${STATE_DIR}"
  if [ "${TARGET_OS}" = "linux" ] && command -v systemctl >/dev/null 2>&1; then
    echo "7. Install systemd user service to ${SYSTEMD_SERVICE_FILE}"
    echo "8. Run systemctl --user daemon-reload"
  fi
  echo "9. Check PATH for ${INSTALL_DIR} and display startup instructions"
  echo ""
  echo "[DRY RUN] Complete. No changes made."
  exit 0
fi

# Ensure curl is available
if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required to download Ultiproxy." >&2
  exit 1
fi

# Choose SHA verification utility
if command -v sha256sum >/dev/null 2>&1; then
  SHA_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  SHA_CMD="shasum -a 256"
else
  echo "Error: Neither sha256sum nor shasum was found for checksum verification." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "Downloading ${TARBALL}..."
curl -fsSL "${DOWNLOAD_URL}" -o "${TMP_DIR}/${TARBALL}"
curl -fsSL "${SHA256_URL}" -o "${TMP_DIR}/${TARBALL}.sha256"

echo "Verifying SHA256 checksum..."
cd "${TMP_DIR}"
# Handle formats: "hash  filename" or "hash"
EXPECTED_HASH="$(awk '{print $1}' "${TARBALL}.sha256")"
ACTUAL_HASH="$(${SHA_CMD} "${TARBALL}" | awk '{print $1}')"

if [ "${EXPECTED_HASH}" != "${ACTUAL_HASH}" ]; then
  echo "Error: SHA256 mismatch!" >&2
  echo "  Expected: ${EXPECTED_HASH}" >&2
  echo "  Actual:   ${ACTUAL_HASH}" >&2
  exit 1
fi
echo "Checksum OK."

echo "Extracting release..."
tar -xzf "${TARBALL}" -C "${TMP_DIR}"

mkdir -p "${INSTALL_DIR}"
if [ -f "${TMP_DIR}/ultiproxy" ]; then
  mv "${TMP_DIR}/ultiproxy" "${BINARY_PATH}"
else
  # Find executable named ultiproxy in extracted contents
  FOUND_BIN="$(find "${TMP_DIR}" -type f -name "ultiproxy" | head -n 1)"
  if [ -n "${FOUND_BIN}" ]; then
    mv "${FOUND_BIN}" "${BINARY_PATH}"
  else
    echo "Error: 'ultiproxy' binary not found in downloaded archive." >&2
    exit 1
  fi
fi
chmod 755 "${BINARY_PATH}"
echo "Installed binary to ${BINARY_PATH}"

# Ensure state directory exists
mkdir -p "${STATE_DIR}"

# generate_api_key mints the single admin bearer token written to the config.
generate_api_key() {
  if command -v openssl >/dev/null 2>&1; then
    printf 'sk-up-%s' "$(openssl rand -hex 24)"
    return 0
  fi
  printf 'sk-up-%s' "$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
}

# installed_api_key reports the admin key already present in an existing config
# (empty when there is none), so the post-install hints never show a key that
# was not actually written.
installed_api_key() {
  [ -f "${CONFIG_FILE}" ] || return 0
  sed -n 's/^[[:space:]]*api_key:[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' "${CONFIG_FILE}" | head -n 1
}

# Create config template if missing. The template only uses the current config
# schema (server.addr, server.api_key, data_dir, storage.db_path); lanes, model
# aliases and timeouts are added at runtime over MCP and persist in the data
# dir, so there is deliberately nothing else to write here.
API_KEY="$(installed_api_key)"
if [ ! -f "${CONFIG_FILE}" ]; then
  API_KEY="$(generate_api_key)"
  echo "Generating initial configuration at ${CONFIG_FILE}..."
  mkdir -p "${CONFIG_DIR}"
  cat <<CONFIG_EOF > "${CONFIG_FILE}"
# Ultiproxy Configuration
# Agent-first LLM subscription proxy on :9050.
#
# This file only carries the keys the current schema understands:
#   server.addr, server.api_key, data_dir, storage.db_path
# Anything else makes the daemon refuse to start (unknown keys are rejected).
#
# Providers (lanes), model aliases and per-lane timeouts are NOT configured
# here: add them over MCP while the daemon runs and they persist in the data
# dir (providers.json, aliases.json, timeouts.json).

server:
  # Bind address. Override per-run with ULTIPROXY_ADDR=0.0.0.0:9050 to serve
  # remote agents without editing this file.
  addr: "127.0.0.1:9050"

  # Single admin bearer token for clients. Remove (or leave empty) for an
  # open-access localhost install; declare scoped client keys over MCP for
  # per-key attribution instead.
  api_key: "${API_KEY}"

# Runtime state (providers, aliases, timeouts, credentials) lives here.
data_dir: "${STATE_DIR}"

# SQLite telemetry database.
storage:
  db_path: "${STATE_DIR}/ultiproxy.db"
CONFIG_EOF
  chmod 600 "${CONFIG_FILE}"
fi

# Set up systemd user service if available on Linux
if [ "${TARGET_OS}" = "linux" ] && command -v systemctl >/dev/null 2>&1; then
  echo "Configuring systemd user service..."
  mkdir -p "${SYSTEMD_USER_DIR}"
  cat <<'SERVICE_EOF' > "${SYSTEMD_SERVICE_FILE}"
[Unit]
Description=Ultiproxy - Universal LLM Subscription Proxy
Documentation=https://github.com/smhanov/ultiproxy
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/ultiproxy serve --config %h/.config/ultiproxy/config.yaml --data-dir %h/.local/state/ultiproxy
WorkingDirectory=%h/.local/state/ultiproxy
Restart=on-failure
RestartSec=5
EnvironmentFile=-%h/.config/ultiproxy/env

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
StateDirectory=ultiproxy

[Install]
WantedBy=default.target
SERVICE_EOF
  chmod 644 "${SYSTEMD_SERVICE_FILE}"
  systemctl --user daemon-reload || true
  echo "Systemd service installed to ${SYSTEMD_SERVICE_FILE}"
fi

echo ""
echo "=== Ultiproxy Installed Successfully! ==="
echo ""
if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
  echo "Notice: ${INSTALL_DIR} is not in your current PATH."
  echo "Add it by running:"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo "or add that line to ~/.bashrc or ~/.zshrc."
  echo ""
fi

echo "Next steps:"
echo "1. Add your subscriptions with the agent-facing MCP tools at"
echo "   http://localhost:9050/mcp (add_provider, set_model_alias, ...)."
echo "   No credentials are needed in ${CONFIG_FILE}."
echo ""
echo "2. Start Ultiproxy:"
if [ "${TARGET_OS}" = "linux" ] && command -v systemctl >/dev/null 2>&1; then
  echo "   systemctl --user enable --now ultiproxy"
  echo "   (or run manually: ultiproxy serve --config ${CONFIG_FILE})"
else
  echo "   ${BINARY_PATH} serve --config ${CONFIG_FILE}"
fi
echo ""
echo "3. Verify health & endpoints:"
echo "   curl http://localhost:9050/healthz"
echo "   curl http://localhost:9050/llms.txt"
echo "   curl http://localhost:9050/api/quota"
echo ""
if [ -f "${CONFIG_FILE}" ] && [ -z "${API_KEY}" ]; then
  echo "Notice: existing ${CONFIG_FILE} declares no server.api_key, so the daemon"
  echo "accepts every request until one is added. Unknown keys from an older"
  echo "template now abort startup; rewrite that file with the current schema."
  echo ""
fi
if [ -n "${API_KEY}" ]; then
  echo "Clients send the admin bearer key below (minted on first install) as"
  echo "\"Authorization: Bearer <key>\"). It is stored in ${CONFIG_FILE}:"
  echo "   ${API_KEY}"
  echo "   curl http://localhost:9050/v1/models -H 'Authorization: Bearer ${API_KEY}'"
  echo "Delete server.api_key in ${CONFIG_FILE} for an open-access localhost install."
  echo ""
fi
