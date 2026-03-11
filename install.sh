#!/bin/sh
# Install zerostart from GitHub releases.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/gpu-cli/zerostart/main/install.sh | sh
#
# Options (via env vars):
#   INSTALL_DIR=/usr/local/bin   Where to put the binary (default: /usr/local/bin)
#   VERSION=v0.1.0               Specific version (default: latest)

set -eu

REPO="gpu-cli/zerostart"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux) ;;
  *) echo "Error: zerostart only supports Linux (got $OS)." >&2; exit 1 ;;
esac

case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) echo "Error: unsupported architecture $ARCH." >&2; exit 1 ;;
esac

BINARY="zerostart-linux-${ARCH}"

# Resolve version
if [ -z "${VERSION:-}" ]; then
  VERSION="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"//;s/".*//')"
  if [ -z "$VERSION" ]; then
    echo "Error: could not determine latest version." >&2
    exit 1
  fi
fi

URL="https://github.com/${REPO}/releases/download/${VERSION}/${BINARY}"

echo "Installing zerostart ${VERSION} (${ARCH})..."

# Download
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o "/tmp/${BINARY}"
elif command -v wget >/dev/null 2>&1; then
  wget -q "$URL" -O "/tmp/${BINARY}"
else
  echo "Error: curl or wget required." >&2
  exit 1
fi

# Install
chmod +x "/tmp/${BINARY}"
if [ -w "$INSTALL_DIR" ]; then
  mv "/tmp/${BINARY}" "${INSTALL_DIR}/zerostart"
else
  sudo mv "/tmp/${BINARY}" "${INSTALL_DIR}/zerostart"
fi

echo "Installed zerostart to ${INSTALL_DIR}/zerostart"
echo ""
echo "  zerostart run -p torch serve.py"
echo ""

# Optional: install Python SDK
if command -v pip >/dev/null 2>&1; then
  echo "To install the Python SDK (accelerate(), vLLM integration):"
  echo "  pip install git+https://github.com/${REPO}.git#subdirectory=python"
fi
