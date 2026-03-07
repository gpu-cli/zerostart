#!/usr/bin/env bash
#
# Probe driver versions across all available GPU types on RunPod
#
# This launches probe-gpu-env.sh on each GPU type sequentially
# to build a matrix of driver versions and CRIU compatibility.
#
# Usage: bash tools/probe-all-gpus.sh
#
set -euo pipefail

RESULTS_DIR="benches/results/gpu-probe"
mkdir -p "$RESULTS_DIR"

# GPU types to probe (most common RunPod types)
GPU_TYPES=(
    "RTX 4090"
    "RTX A6000"
    "A100 80GB"
    "H100 PCIe"
    "H100 SXM"
    "L40S"
    "L4"
    "RTX 3090"
    "RTX A5000"
    "A40"
)

echo "========================================"
echo "  Multi-GPU Type Driver Version Probe"
echo "========================================"
echo ""
echo "Will probe ${#GPU_TYPES[@]} GPU types on RunPod"
echo "Results saved to: $RESULTS_DIR/"
echo ""

for GPU in "${GPU_TYPES[@]}"; do
    SAFE_NAME=$(echo "$GPU" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
    OUTFILE="$RESULTS_DIR/${SAFE_NAME}.txt"

    echo "--- Probing: $GPU ---"

    # Create a temporary gpu.jsonc for this specific GPU type
    TEMP_CONFIG=$(mktemp /tmp/gpu-probe-XXXXXX.jsonc)
    cat > "$TEMP_CONFIG" << EOJSON
{
  "\$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "zs-probe-${SAFE_NAME}",
  "gpu_types": [{ "type": "${GPU}" }],
  "min_vram": 8,
  "keep_alive_minutes": 2,
  "volume_mode": "global"
}
EOJSON

    # Run the probe script on this GPU type
    # Use timeout to avoid hanging on unavailable GPUs
    if timeout 300 gpu run --config "$TEMP_CONFIG" "bash tools/probe-gpu-env.sh" > "$OUTFILE" 2>&1; then
        # Extract key info for summary
        DRIVER=$(grep 'version:' "$OUTFILE" | head -1 | awk '{print $NF}')
        MAJOR=$(grep 'major:' "$OUTFILE" | head -1 | awk '{print $NF}')
        echo "  driver: $DRIVER (r${MAJOR})"
    else
        echo "  FAILED or UNAVAILABLE"
        echo "UNAVAILABLE" > "$OUTFILE"
    fi

    # Clean up the pod
    timeout 30 gpu stop --config "$TEMP_CONFIG" 2>/dev/null || true
    rm -f "$TEMP_CONFIG"
    echo ""
done

# Build summary table
echo "========================================"
echo "  SUMMARY: Driver Versions by GPU Type"
echo "========================================"
echo ""
printf "%-20s %-15s %-8s %-10s %s\n" "GPU Type" "Driver" "Major" "CRIU OK?" "Notes"
printf "%-20s %-15s %-8s %-10s %s\n" "--------------------" "---------------" "--------" "----------" "-----"

for GPU in "${GPU_TYPES[@]}"; do
    SAFE_NAME=$(echo "$GPU" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
    OUTFILE="$RESULTS_DIR/${SAFE_NAME}.txt"

    if [ -f "$OUTFILE" ] && ! grep -q "UNAVAILABLE" "$OUTFILE"; then
        DRIVER=$(grep '  version:' "$OUTFILE" | head -1 | awk '{print $NF}' 2>/dev/null || echo "?")
        MAJOR=$(grep '  major:' "$OUTFILE" | head -1 | awk '{print $NF}' 2>/dev/null || echo "?")

        if [ "${MAJOR:-0}" -ge 555 ]; then
            CRIU_OK="YES"
        else
            CRIU_OK="NO (r${MAJOR})"
        fi

        NOTES=""
        if [ "${MAJOR:-0}" -ge 580 ]; then
            NOTES="GPU migration"
        elif [ "${MAJOR:-0}" -ge 570 ]; then
            NOTES="full features"
        elif [ "${MAJOR:-0}" -ge 555 ]; then
            NOTES="basic only"
        fi

        printf "%-20s %-15s %-8s %-10s %s\n" "$GPU" "$DRIVER" "r$MAJOR" "$CRIU_OK" "$NOTES"
    else
        printf "%-20s %-15s %-8s %-10s %s\n" "$GPU" "-" "-" "-" "unavailable"
    fi
done

echo ""
echo "Results saved to $RESULTS_DIR/"
