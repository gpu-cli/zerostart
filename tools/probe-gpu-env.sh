#!/usr/bin/env bash
#
# Probe GPU environment for CRIU + cuda-checkpoint compatibility
#
# Reports: driver version, CUDA version, GPU type/count/UUIDs, compute cap,
#          CRIU availability, cuda-checkpoint availability, kernel config
#
# Run: gpu run "bash tools/probe-gpu-env.sh"
#
set -euo pipefail

echo "========================================"
echo "  GPU Environment Probe for zerostart"
echo "========================================"
echo ""

# --- Driver ---
echo "=== NVIDIA Driver ==="
if [ -f /proc/driver/nvidia/version ]; then
    DRIVER_LINE=$(head -1 /proc/driver/nvidia/version)
    echo "  raw: $DRIVER_LINE"
    DRIVER_VER=$(echo "$DRIVER_LINE" | grep -oP '\d+\.\d+\.\d+' | head -1)
    DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
    echo "  version: $DRIVER_VER"
    echo "  major: $DRIVER_MAJOR"

    if [ "$DRIVER_MAJOR" -ge 580 ]; then
        echo "  features: lock, checkpoint, restore, unlock, timeout, process-tree, NVML, GPU migration"
    elif [ "$DRIVER_MAJOR" -ge 570 ]; then
        echo "  features: lock, checkpoint, restore, unlock, timeout, process-tree, NVML"
    elif [ "$DRIVER_MAJOR" -ge 555 ]; then
        echo "  features: lock, checkpoint, restore, unlock (CRIU plugin compatible)"
    elif [ "$DRIVER_MAJOR" -ge 550 ]; then
        echo "  features: toggle only (no --action flag, no CRIU plugin)"
    else
        echo "  features: NO checkpoint support (need r550+)"
    fi
else
    echo "  NOT FOUND - no NVIDIA driver loaded"
fi
echo ""

# --- CUDA ---
echo "=== CUDA ==="
if command -v nvcc &>/dev/null; then
    echo "  nvcc: $(nvcc --version 2>&1 | grep 'release' | sed 's/.*release //' | sed 's/,.*//')"
fi
if [ -f /usr/local/cuda/version.txt ]; then
    echo "  toolkit: $(cat /usr/local/cuda/version.txt)"
elif [ -d /usr/local/cuda ]; then
    echo "  toolkit dir: $(ls -d /usr/local/cuda-* 2>/dev/null | tr '\n' ' ')"
fi
echo ""

# --- GPUs ---
echo "=== GPU Topology ==="
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
    echo "  count: $GPU_COUNT"
    echo ""
    nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap,pci.bus_id --format=csv,noheader | while IFS= read -r line; do
        echo "  $line"
    done
    echo ""
    echo "  --- nvidia-smi summary ---"
    nvidia-smi --query-gpu=name,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
else
    echo "  nvidia-smi NOT FOUND"
fi
echo ""

# --- Kernel C/R support ---
echo "=== Kernel ==="
echo "  kernel: $(uname -r)"
if [ -f /proc/config.gz ]; then
    echo "  CHECKPOINT_RESTORE: $(zcat /proc/config.gz 2>/dev/null | grep CONFIG_CHECKPOINT_RESTORE || echo 'not found')"
    echo "  USERFAULTFD: $(zcat /proc/config.gz 2>/dev/null | grep CONFIG_USERFAULTFD || echo 'not found')"
elif [ -f "/boot/config-$(uname -r)" ]; then
    echo "  CHECKPOINT_RESTORE: $(grep CONFIG_CHECKPOINT_RESTORE "/boot/config-$(uname -r)" || echo 'not found')"
    echo "  USERFAULTFD: $(grep CONFIG_USERFAULTFD "/boot/config-$(uname -r)" || echo 'not found')"
else
    echo "  config: not available (checking /proc/sys/kernel/)"
    echo "  ns_last_pid: $(cat /proc/sys/kernel/ns_last_pid 2>/dev/null && echo ' (C/R likely supported)' || echo 'not found (C/R may not be enabled)')"
fi
echo ""

# --- CRIU ---
echo "=== CRIU ==="
if command -v criu &>/dev/null; then
    echo "  version: $(criu --version 2>&1 | head -1)"
    echo "  check: $(criu check 2>&1 | tail -1 || echo 'check failed')"
    # Check for CUDA plugin
    CRIU_LIBDIR=$(criu --help 2>&1 | grep -oP 'default: \K[^ ]+' | head -1 || echo "")
    if [ -n "$CRIU_LIBDIR" ] && [ -f "$CRIU_LIBDIR/cuda_plugin.so" ]; then
        echo "  cuda_plugin.so: $CRIU_LIBDIR/cuda_plugin.so"
    else
        for dir in /usr/lib/criu /usr/lib64/criu /usr/local/lib/criu; do
            if [ -f "$dir/cuda_plugin.so" ]; then
                echo "  cuda_plugin.so: $dir/cuda_plugin.so"
                break
            fi
        done
        echo "  cuda_plugin.so: NOT FOUND (need CRIU 4.0+ with CUDA plugin)"
    fi
else
    echo "  NOT INSTALLED"
    echo ""
    echo "  To install CRIU 4.0+:"
    echo "    apt-get install -y criu  # if distro has 4.0+"
    echo "    # or build from source:"
    echo "    git clone https://github.com/checkpoint-restore/criu.git"
    echo "    cd criu && git checkout v4.0 && make -j\$(nproc) && make install"
fi
echo ""

# --- cuda-checkpoint ---
echo "=== cuda-checkpoint ==="
if command -v cuda-checkpoint &>/dev/null; then
    echo "  path: $(which cuda-checkpoint)"
    echo "  help: $(cuda-checkpoint --help 2>&1 | head -2)"
    # Check if --action flag is supported (r555+)
    if cuda-checkpoint --help 2>&1 | grep -q -- '--action'; then
        echo "  --action: SUPPORTED"
    else
        echo "  --action: NOT SUPPORTED (need r555+ driver)"
    fi
else
    echo "  NOT INSTALLED"
    echo ""
    echo "  To install:"
    echo "    # Download pre-built binary from NVIDIA"
    echo "    wget https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint"
    echo "    chmod +x cuda-checkpoint && mv cuda-checkpoint /usr/local/bin/"
fi
echo ""

# --- GPU persistence mode ---
echo "=== GPU Persistence ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | while IFS= read -r line; do
        echo "  persistence_mode: $line"
    done
    echo "  (persistence mode or cuInit() required before CRIU restore)"
fi
echo ""

# --- Summary ---
echo "========================================"
echo "  COMPATIBILITY SUMMARY"
echo "========================================"
READY=true

if [ -z "${DRIVER_MAJOR:-}" ]; then
    echo "  [FAIL] No NVIDIA driver detected"
    READY=false
elif [ "$DRIVER_MAJOR" -lt 555 ]; then
    echo "  [FAIL] Driver r${DRIVER_MAJOR} too old for CRIU CUDA plugin (need r555+)"
    READY=false
elif [ "$DRIVER_MAJOR" -lt 570 ]; then
    echo "  [WARN] Driver r${DRIVER_MAJOR} - basic support only (no lock timeout, no process tree)"
else
    echo "  [OK]   Driver r${DRIVER_MAJOR} - full CRIU + cuda-checkpoint support"
fi

if command -v criu &>/dev/null; then
    CRIU_VER=$(criu --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    CRIU_MAJOR=$(echo "$CRIU_VER" | cut -d. -f1)
    if [ "$CRIU_MAJOR" -ge 4 ]; then
        echo "  [OK]   CRIU $CRIU_VER (4.0+ has CUDA plugin)"
    else
        echo "  [WARN] CRIU $CRIU_VER (need 4.0+ for built-in CUDA plugin)"
    fi
else
    echo "  [MISS] CRIU not installed"
fi

if command -v cuda-checkpoint &>/dev/null; then
    echo "  [OK]   cuda-checkpoint installed"
else
    echo "  [MISS] cuda-checkpoint not installed"
fi

echo ""
if [ "$READY" = true ]; then
    echo "  GPU snapshots: LIKELY SUPPORTED (install CRIU + cuda-checkpoint to confirm)"
else
    echo "  GPU snapshots: NOT SUPPORTED on this configuration"
fi
echo ""
