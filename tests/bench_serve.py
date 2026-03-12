# /// script
# dependencies = ["torch>=2.5", "transformers", "safetensors", "accelerate"]
# ///
import time, os, gc

t_wall_start = float(os.environ.get("BENCH_T_START", "0"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_id = os.environ.get("BENCH_MODEL", "Qwen/Qwen3.5-35B-A3B")

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16)

t_done = time.time() * 1000
param_b = sum(p.numel() for p in model.parameters()) / 1e9
peak_mb = torch.cuda.max_memory_allocated() / 1e6

if t_wall_start > 0:
    total = (t_done - t_wall_start) / 1000
    print(f"RESULT: {total:.1f}s  (params={param_b:.1f}B, peak_gpu={peak_mb:.0f}MB)")
else:
    print(f"RESULT: unknown (no BENCH_T_START)")

del model; gc.collect(); torch.cuda.empty_cache()
