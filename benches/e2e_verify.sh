#!/usr/bin/env bash
# End-to-end verification: both uv and zs-fast-wheel produce working installs
#
# Runs the SAME Python script under both install methods and verifies outputs match.
#
# Usage: gpu run "bash benches/e2e_verify.sh"

set -euo pipefail

WARM_BIN="./bin/zs-fast-wheel-linux-x86_64"
chmod +x "$WARM_BIN"
UV="$HOME/.local/bin/uv"
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "========================================"
echo "  E2E Verification: uv vs zs-fast-wheel"
echo "  Python: $PY_VERSION"
echo "  $(date)"
echo "========================================"

DEPS="numpy
requests
pyyaml
tqdm
packaging
jinja2
safetensors"

# The test script that exercises each package
TEST_SCRIPT=$(cat <<'PYEOF'
import sys, os, json

results = {}

# numpy
try:
    import numpy as np
    a = np.array([1, 2, 3])
    results["numpy"] = {"version": np.__version__, "test": (a * 2).tolist()}
except Exception as e:
    results["numpy"] = {"error": str(e)}

# requests
try:
    import requests
    results["requests"] = {"version": requests.__version__, "test": "ok"}
except Exception as e:
    results["requests"] = {"error": str(e)}

# pyyaml
try:
    import yaml
    data = yaml.safe_load("key: value\nlist:\n  - 1\n  - 2")
    results["pyyaml"] = {"version": yaml.__version__, "test": data}
except Exception as e:
    results["pyyaml"] = {"error": str(e)}

# tqdm
try:
    import tqdm
    results["tqdm"] = {"version": tqdm.__version__, "test": "ok"}
except Exception as e:
    results["tqdm"] = {"error": str(e)}

# packaging
try:
    import packaging.version
    v = packaging.version.Version("1.2.3")
    results["packaging"] = {"version": packaging.__version__, "test": str(v)}
except Exception as e:
    results["packaging"] = {"error": str(e)}

# jinja2
try:
    import jinja2
    t = jinja2.Template("Hello {{ name }}")
    results["jinja2"] = {"version": jinja2.__version__, "test": t.render(name="world")}
except Exception as e:
    results["jinja2"] = {"error": str(e)}

# safetensors
try:
    import safetensors
    results["safetensors"] = {"version": safetensors.__version__, "test": "ok"}
except Exception as e:
    results["safetensors"] = {"error": str(e)}

# Summary
passed = sum(1 for v in results.values() if "error" not in v)
total = len(results)
print(json.dumps(results, indent=2))
print(f"\n{passed}/{total} packages working")
sys.exit(0 if passed == total else 1)
PYEOF
)

# ─── Test 1: uv pip install ─────────────────────────────────────────────

echo ""
echo "=== Test 1: uv pip install ==="
UV_VENV="/tmp/e2e-uv-$$"
rm -rf "$UV_VENV"
$UV venv "$UV_VENV" --python "$PY_VERSION" -q 2>/dev/null || $UV venv "$UV_VENV" -q

echo "$DEPS" > /tmp/e2e-reqs-$$.txt
$UV pip install -r /tmp/e2e-reqs-$$.txt --python "$UV_VENV/bin/python" -q 2>/dev/null

echo "--- uv install output ---"
"$UV_VENV/bin/python" -c "$TEST_SCRIPT"
UV_EXIT=$?
echo "--- uv exit: $UV_EXIT ---"

rm -rf "$UV_VENV" /tmp/e2e-reqs-$$.txt

# ─── Test 2: zs-fast-wheel warm ─────────────────────────────────────────

echo ""
echo "=== Test 2: zs-fast-wheel warm ==="
ZS_SP="/tmp/e2e-zs-$$"
rm -rf "$ZS_SP"
mkdir -p "$ZS_SP"

$WARM_BIN warm --requirements "$DEPS" --site-packages "$ZS_SP" --python-version "$PY_VERSION" -j 8 2>&1

echo "--- zs-warm output ---"
PYTHONPATH="$ZS_SP" python3 -c "$TEST_SCRIPT"
ZS_EXIT=$?
echo "--- zs-warm exit: $ZS_EXIT ---"

rm -rf "$ZS_SP"

# ─── Test 3: torch e2e (the big one) ────────────────────────────────────

TORCH_DEPS="torch
numpy"

TORCH_SCRIPT=$(cat <<'PYEOF'
import sys, json

results = {}

try:
    import torch
    results["torch_version"] = torch.__version__
    results["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        results["gpu_memory_gb"] = round(mem / 1e9, 1)

    # Basic tensor ops
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    results["cpu_dot"] = torch.dot(a, b).item()

    # GPU tensor ops if available
    if torch.cuda.is_available():
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        results["gpu_dot"] = torch.dot(a_gpu, b_gpu).item()

        # Matrix multiply on GPU
        m = torch.randn(1000, 1000, device="cuda")
        result = torch.mm(m, m)
        results["gpu_matmul_shape"] = list(result.shape)

    results["status"] = "ok"
except Exception as e:
    results["error"] = str(e)
    import traceback
    results["traceback"] = traceback.format_exc()

print(json.dumps(results, indent=2))
sys.exit(0 if results.get("status") == "ok" else 1)
PYEOF
)

echo ""
echo "=== Test 3a: torch via uv pip install ==="
UV_TORCH="/tmp/e2e-uv-torch-$$"
rm -rf "$UV_TORCH"
$UV venv "$UV_TORCH" --python "$PY_VERSION" -q 2>/dev/null || $UV venv "$UV_TORCH" -q

echo "$TORCH_DEPS" > /tmp/e2e-torch-reqs-$$.txt
$UV pip install -r /tmp/e2e-torch-reqs-$$.txt --python "$UV_TORCH/bin/python" -q 2>/dev/null

echo "--- uv torch output ---"
"$UV_TORCH/bin/python" -c "$TORCH_SCRIPT"
UV_TORCH_EXIT=$?
echo "--- uv torch exit: $UV_TORCH_EXIT ---"

rm -rf "$UV_TORCH" /tmp/e2e-torch-reqs-$$.txt

echo ""
echo "=== Test 3b: torch via zs-fast-wheel warm ==="
ZS_TORCH="/tmp/e2e-zs-torch-$$"
rm -rf "$ZS_TORCH"
mkdir -p "$ZS_TORCH"

$WARM_BIN warm --requirements "$TORCH_DEPS" --site-packages "$ZS_TORCH" --python-version "$PY_VERSION" -j 8 2>&1

echo "--- zs-warm torch output ---"
PYTHONPATH="$ZS_TORCH" python3 -c "$TORCH_SCRIPT"
ZS_TORCH_EXIT=$?
echo "--- zs-warm torch exit: $ZS_TORCH_EXIT ---"

rm -rf "$ZS_TORCH"

# ─── Summary ────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo "  uv small packages:     $([ $UV_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "  zs-warm small packages: $([ $ZS_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "  uv torch+GPU:          $([ $UV_TORCH_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "  zs-warm torch+GPU:     $([ $ZS_TORCH_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "========================================"

# Exit non-zero if any test failed
[ $UV_EXIT -eq 0 ] && [ $ZS_EXIT -eq 0 ] && [ $UV_TORCH_EXIT -eq 0 ] && [ $ZS_TORCH_EXIT -eq 0 ]
