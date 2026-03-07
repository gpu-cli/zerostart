#!/usr/bin/env bash
#
# Test: Can we inject an extracted wheel into uv's cache and have uv use it?
#
# This proves the core thesis: zs-fast-wheel downloads fast, writes to uv cache,
# then uv install/sync links from cache to venv instantly.
#
set -euo pipefail

UV="$HOME/.local/bin/uv"

echo "=== Phase 1: Setup ==="

if ! [ -x "$UV" ]; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

UV_CACHE=$($UV cache dir)
echo "uv cache dir: $UV_CACHE"
echo "uv version:   $($UV --version)"

WORKDIR=$(mktemp -d)
echo "workdir:      $WORKDIR"

PKG_NAME="six"
CACHE_KEY="1.17.0-py2.py3-none-any"

echo ""
echo "=== Phase 2: Clean slate ==="

$UV cache clean "$PKG_NAME" 2>/dev/null || true
$UV venv "$WORKDIR/venv" 2>/dev/null
echo "Created venv"

# Verify not installed
if "$WORKDIR/venv/bin/python" -c "import six" 2>/dev/null; then
    echo "ERROR: six already in fresh venv"; exit 1
fi
echo "Confirmed: six is NOT installed"

echo ""
echo "=== Phase 3: Download wheel via pip (simulating our fast download) ==="

# Download via pip (just getting the .whl file — in prod zs-fast-wheel does this)
mkdir -p "$WORKDIR/wheels"
pip3 download "six==1.17.0" --no-deps -d "$WORKDIR/wheels" --no-cache-dir -q 2>&1
WHEEL_FILE=$(ls "$WORKDIR/wheels/"*.whl | head -1)
echo "Downloaded: $WHEEL_FILE ($(wc -c < "$WHEEL_FILE" | tr -d ' ') bytes)"

echo ""
echo "=== Phase 4: Extract into uv archive bucket ==="

ARCHIVE_ID="zs-test-$(date +%s)"
ARCHIVE_DIR="$UV_CACHE/archive-v0/$ARCHIVE_ID"
mkdir -p "$ARCHIVE_DIR"

python3 -c "
import zipfile, sys
with zipfile.ZipFile('$WHEEL_FILE') as zf:
    zf.extractall('$ARCHIVE_DIR')
    print(f'Extracted {len(zf.namelist())} files')
"

echo "Archive contents:"
ls "$ARCHIVE_DIR/"

echo ""
echo "=== Phase 5: Create symlink in wheels bucket ==="

WHEELS_DIR="$UV_CACHE/wheels-v6/pypi/$PKG_NAME"
mkdir -p "$WHEELS_DIR"

LINK_PATH="$WHEELS_DIR/$CACHE_KEY"
rm -rf "$LINK_PATH" 2>/dev/null || true
ln -s "../../../archive-v0/$ARCHIVE_ID" "$LINK_PATH"

echo "Symlink: $LINK_PATH"
echo "  -> $(readlink "$LINK_PATH")"
echo "  valid: $([ -d "$(cd "$WHEELS_DIR" && readlink -f "$CACHE_KEY")" ] && echo YES || echo NO)"

echo ""
echo "=== Phase 6: Can uv install from our injected cache? ==="

echo ""
echo "--- Attempt: uv pip install six==1.17.0 ---"
$UV pip install \
    --python "$WORKDIR/venv/bin/python" \
    --no-build \
    "six==1.17.0" \
    -v 2>&1 | tee "$WORKDIR/install.log"

echo ""
if "$WORKDIR/venv/bin/python" -c "import six; print(f'six {six.__version__} at {six.__file__}')" 2>&1; then
    echo "IMPORT: SUCCESS"
else
    echo "IMPORT: FAILED"
fi

echo ""
echo "--- What uv did ---"
grep -iE "six|cache|download|install|link|skip|cached|built" "$WORKDIR/install.log" | head -20

echo ""
echo "=== Phase 7: Normal uv install for comparison ==="

# Clean and redo
$UV cache clean "$PKG_NAME" 2>/dev/null || true
rm -rf "$WORKDIR/venv"
$UV venv "$WORKDIR/venv" 2>/dev/null

echo "--- Normal install ---"
$UV pip install \
    --python "$WORKDIR/venv/bin/python" \
    --no-build \
    "six==1.17.0" \
    -v 2>&1 | tee "$WORKDIR/normal.log"

echo ""
echo "--- Normal install log ---"
grep -iE "six|cache|download|install|link|skip|cached|built" "$WORKDIR/normal.log" | head -20

echo ""
echo "=== Phase 8: Compare cache layouts ==="

echo ""
echo "--- wheels-v6/pypi/six/ ---"
ls -la "$UV_CACHE/wheels-v6/pypi/$PKG_NAME/" 2>/dev/null || echo "(empty)"

echo ""
echo "--- .http files ---"
for f in "$UV_CACHE/wheels-v6/pypi/$PKG_NAME/"*.http; do
    [ -f "$f" ] || continue
    echo "  $(basename "$f") ($(wc -c < "$f" | tr -d ' ') bytes)"
    file "$f" | sed 's/^/  /'
    xxd -l 80 "$f" | sed 's/^/    /'
done

echo ""
echo "--- symlinks ---"
for f in "$UV_CACHE/wheels-v6/pypi/$PKG_NAME/"*; do
    [ -L "$f" ] || continue
    echo "  $(basename "$f") -> $(readlink "$f")"
done

echo ""
echo "--- archive dirs ---"
for d in "$UV_CACHE/archive-v0/"*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    count=$(ls "$d" | wc -l | tr -d ' ')
    echo "  $name ($count items): $(ls "$d" | head -3 | tr '\n' ' ')"
done

echo ""
echo "=== RESULT ==="
echo ""
if grep -qi "cached" "$WORKDIR/install.log"; then
    echo "CACHE INJECTION WORKED — uv used our pre-populated cache entry"
elif grep -qi "download" "$WORKDIR/install.log"; then
    echo "CACHE INJECTION FAILED — uv downloaded fresh"
    echo ""
    echo "This means we likely need to also write the .http cache policy file."
    echo "Check the comparison above to see what uv's normal cache looks like."
else
    echo "UNCLEAR — check logs above"
fi

echo ""
rm -rf "$WORKDIR"
echo "Cleaned up."
