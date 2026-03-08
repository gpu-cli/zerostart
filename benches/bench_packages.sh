#!/usr/bin/env bash
# Benchmark: uvx vs zerostart cold/warm boot for popular packages
#
# Usage: ./benches/bench_packages.sh [package...]
# Default packages if none specified.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS="$SCRIPT_DIR/results/packages.csv"
ZS_CACHE="$REPO_DIR/.zerostart"

mkdir -p "$SCRIPT_DIR/results"

# Packages and their --version/--help flag + expected-success exit codes
# Format: "package|entry_cmd|args|ok_exit"
PACKAGES=(
    "black|black|--version|0"
    "ruff|ruff|--version|0"
    "httpie|httpie|--version|0"
    "mkdocs|mkdocs|--version|0"
    "datasette|datasette|--version|0"
    "celery|celery|--version|0"
    "fastapi[standard]|fastapi|--version|0"
)

# Allow overriding from CLI
if [ $# -gt 0 ]; then
    PACKAGES=()
    for pkg in "$@"; do
        PACKAGES+=("$pkg|$pkg|--version|0")
    done
fi

echo "package,tool,run,elapsed_s,exit_code" > "$RESULTS"

time_cmd() {
    local start end elapsed
    start=$(python3 -c "import time; print(f'{time.monotonic():.4f}')")
    "$@" > /dev/null 2>&1
    local exit_code=$?
    end=$(python3 -c "import time; print(f'{time.monotonic():.4f}')")
    elapsed=$(python3 -c "print(f'{$end - $start:.3f}')")
    echo "$elapsed $exit_code"
}

for entry in "${PACKAGES[@]}"; do
    IFS='|' read -r pkg cmd args ok_exit <<< "$entry"
    echo ""
    echo "=== $pkg ==="

    # --- uvx cold (clear cache) ---
    echo "  uvx cold..."
    uv cache clean "$cmd" > /dev/null 2>&1
    rm -rf "$HOME/.cache/uv/environments/$cmd"* 2>/dev/null
    result=$(time_cmd uv tool run --no-cache "$pkg" $args)
    elapsed=$(echo "$result" | awk '{print $1}')
    ec=$(echo "$result" | awk '{print $2}')
    echo "  uvx cold: ${elapsed}s (exit=$ec)"
    echo "$pkg,uvx,cold,$elapsed,$ec" >> "$RESULTS"

    # --- uvx warm ---
    echo "  uvx warm..."
    result=$(time_cmd uv tool run "$pkg" $args)
    elapsed=$(echo "$result" | awk '{print $1}')
    ec=$(echo "$result" | awk '{print $2}')
    echo "  uvx warm: ${elapsed}s (exit=$ec)"
    echo "$pkg,uvx,warm,$elapsed,$ec" >> "$RESULTS"

    # --- zerostart cold ---
    echo "  zerostart cold..."
    rm -rf "$ZS_CACHE"
    result=$(time_cmd uv run --no-project --with-editable "$REPO_DIR/python" -- zerostart "$cmd" -- $args)
    elapsed=$(echo "$result" | awk '{print $1}')
    ec=$(echo "$result" | awk '{print $2}')
    echo "  zerostart cold: ${elapsed}s (exit=$ec)"
    echo "$pkg,zerostart,cold,$elapsed,$ec" >> "$RESULTS"

    # --- zerostart warm ---
    echo "  zerostart warm..."
    result=$(time_cmd uv run --no-project --with-editable "$REPO_DIR/python" -- zerostart "$cmd" -- $args)
    elapsed=$(echo "$result" | awk '{print $1}')
    ec=$(echo "$result" | awk '{print $2}')
    echo "  zerostart warm: ${elapsed}s (exit=$ec)"
    echo "$pkg,zerostart,warm,$elapsed,$ec" >> "$RESULTS"
done

echo ""
echo "=== Results ==="
column -t -s',' "$RESULTS"
