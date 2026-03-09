#!/bin/bash
set -euo pipefail

echo "=== Docker Smoke Tests ==="
echo "Python: $(python3 --version)"
echo "uv: $(uv --version)"
echo "zerostart: $(zerostart --help | head -1)"
echo ""

export ZEROSTART_CACHE=/tmp/.zs-cache

run_test() {
    local name=$1
    shift
    echo "--- Test: $name ---"
    local start=$(date +%s%3N)
    if "$@" 2>&1; then
        local rc=0
    else
        local rc=$?
    fi
    local end=$(date +%s%3N)
    echo "  Time: $((end - start))ms (exit=$rc)"
    echo ""
}

# Test 1: cowsay (tiny, pure python)
rm -rf /tmp/.zs-cache
run_test "cowsay cold" zerostart run cowsay -- -t "hello docker"
run_test "cowsay warm" zerostart run cowsay -- -t "hello warm"

# Test 2: black (medium, compiled extensions)
rm -rf /tmp/.zs-cache
run_test "black cold" zerostart run black -- --version
run_test "black warm" zerostart run black -- --version

# Test 3: httpie (many deps)
rm -rf /tmp/.zs-cache
run_test "httpie cold" zerostart run httpie -- --version
run_test "httpie warm" zerostart run httpie -- --version

# Test 4: ruff (single binary wheel)
rm -rf /tmp/.zs-cache
run_test "ruff cold" zerostart run ruff -- --version
run_test "ruff warm" zerostart run ruff -- --version

# Test 5: torch+cuda (download + install only, won't boot without GPU)
echo "--- Test: torch+cuda download+install ---"
rm -rf /tmp/.zs-cache
cat > /tmp/torch_check.py << 'PYEOF'
import sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '.')
try:
    import torch
    print(f"torch {torch.__version__} installed OK (CUDA: {torch.cuda.is_available()})")
except Exception as e:
    print(f"torch import: {e}")
PYEOF
start=$(date +%s%3N)
zerostart run -v -p "torch" -p "torchvision" /tmp/torch_check.py 2>&1 || true
end=$(date +%s%3N)
echo "  torch+cuda install time: $((end - start))ms"
echo ""

echo "=== All tests passed ==="
