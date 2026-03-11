#!/usr/bin/env bash
# Benchmark 50 packages: cold-vs-cold and warm-vs-warm
# Measures: uvx cold, uvx warm, zerostart cold, zerostart warm

set -o pipefail

PYTHON="${PYTHON:-python3}"
RESULTS_FILE="$(cd "$(dirname "$0")/.." && pwd)/benches/results/bench_50.csv"
mkdir -p "$(dirname "$RESULTS_FILE")"

PACKAGES=(
  black ruff isort autopep8 yapf pyflakes flake8 pylint mypy bandit pydocstyle
  pyright vulture radon
  twine build flit hatch maturin pdm
  flask gunicorn uvicorn httpie datasette
  celery invoke nox tox
  mkdocs pelican
  litecli
  pipx cookiecutter pre-commit glances yt-dlp rich-cli
  docutils ipython bpython ptpython pytest
)

# Binary name overrides for uvx
declare -A UVX_BIN
UVX_BIN[docutils]="rst2html"

# Install name overrides for zerostart
declare -A ZS_PKG
ZS_PKG[docutils]="docutils"

measure() {
  local start end
  start=$(python3 -c "import time; print(time.monotonic())")
  timeout 60 "$@" >/dev/null 2>&1
  local rc=$?
  end=$(python3 -c "import time; print(time.monotonic())")
  if [ $rc -eq 0 ]; then
    python3 -c "print(f'{$end - $start:.3f}')"
  else
    echo "FAIL"
  fi
}

# Header
echo "package,uvx_cold,uvx_warm,zs_cold,zs_warm" > "$RESULTS_FILE"

printf "%-20s %10s %10s %10s %10s\n" "package" "uvx_cold" "uvx_warm" "zs_cold" "zs_warm"
printf "%-20s %10s %10s %10s %10s\n" "-------" "--------" "--------" "-------" "-------"

for pkg in "${PACKAGES[@]}"; do
  uvx_bin="${UVX_BIN[$pkg]:-$pkg}"
  zs_pkg="${ZS_PKG[$pkg]:-$pkg}"

  # uvx cold: clear environment cache
  rm -rf ~/.cache/uv/environments-v2/ 2>/dev/null
  uvx_cold=$(measure uvx "$uvx_bin" --version)

  # uvx warm: environments-v2 cache is now populated
  uvx_warm=$(measure uvx "$uvx_bin" --version)

  # zerostart cold: clear env cache
  rm -rf ~/.cache/zerostart/envs/
  zs_cold=$(measure "$PYTHON" -m zerostart.run "$zs_pkg" -- --version)

  # zerostart warm: env cache populated
  zs_warm=$(measure "$PYTHON" -m zerostart.run "$zs_pkg" -- --version)

  printf "%-20s %10s %10s %10s %10s\n" "$pkg" "$uvx_cold" "$uvx_warm" "$zs_cold" "$zs_warm"
  echo "$pkg,$uvx_cold,$uvx_warm,$zs_cold,$zs_warm" >> "$RESULTS_FILE"
done

echo ""
echo "Results saved to $RESULTS_FILE"
