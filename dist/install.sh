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
  echo "5. Create configuration template at ${CONFIG_FILE} (if not already present)"
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

# Create config template if missing
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Generating initial configuration at ${CONFIG_FILE}..."
  mkdir -p "${CONFIG_DIR}"
  cat <<'CONFIG_EOF' > "${CONFIG_FILE}"
# Ultiproxy Configuration
# Universal LLM Subscription Proxy (:8317)

server:
  listen: "127.0.0.1:8317"
  # Client bearer tokens accepted by Ultiproxy
  api_keys:
    - "sk-up-local-agent-key"

routing:
  strategy: "quota-priority" # quota-priority | round-robin | latency
  fallback_to_openrouter: true

providers:
  # GitHub Copilot (uses github token or oauth)
  copilot:
    enabled: true
    # token: "ghu_..."

  # OpenAI Codex / Plus
  openai:
    enabled: true
    # api_key: "sk-proj-..."

  # Anthropic Claude
  anthropic:
    enabled: true
    # api_key: "sk-ant-..."

  # DeepSeek
  deepseek:
    enabled: false
    # api_key: "sk-..."

  # Local vLLM (zero marginal cost)
  vllm:
    enabled: false
    base_url: "http://127.0.0.1:8000/v1"

accounting:
  enabled: true
  db_path: "~/.local/state/ultiproxy/accounting.db"
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
echo "1. Configure your upstream subscription credentials:"
echo "   nano ${CONFIG_FILE}"
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
echo "   curl http://localhost:8317/healthz"
echo "   curl http://localhost:8317/v1/models -H 'Authorization: Bearer sk-up-local-agent-key'"
echo "   curl http://localhost:8317/api/quota"
echo ""
