# /// script
# dependencies = ["torch>=2.5"]
# ///
import time, os

t_wall_start = float(os.environ.get("BENCH_T_START", "0"))

t0 = time.time() * 1000
import torch
t_torch = time.time() * 1000
print(f"  [time] torch import: {(t_torch - t0)/1000:.1f}s  (v{torch.__version__})")

# Verify CUDA works
props = torch.cuda.get_device_properties(0)
total_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
print(f"  [gpu] {torch.cuda.get_device_name(0)}, {total_gb:.1f}GB")
x = torch.randn(1000, 1000, device="cuda")
y = x @ x
print(f"  [gpu] matmul ok: {y.shape}")

t_end = time.time() * 1000
if t_wall_start > 0:
    total = (t_end - t_wall_start) / 1000
    print(f"RESULT: {total:.1f}s total")
