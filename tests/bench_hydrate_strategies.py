"""Benchmark different hydrate strategies to find the fastest cold+warm path.

Strategies tested:
  A. Current: cloudpickle everything (baseline)
  B. save_pretrained tokenizer + cloudpickle model placeholder only
  C. Tokenizer JSON + model config JSON (no cloudpickle, no transformers import)
  D. torch.jit.trace (no transformers at all during hydrate)

Each strategy is tested both cold (fresh process via subprocess) and warm
(in-process, modules already imported).
"""
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
log = logging.getLogger("bench")

MODEL_ID = os.environ.get("SNAP_MODEL", "Qwen/Qwen2.5-7B")
BENCH_DIR = "/gpu-cli-workspaces/.bench-strategies"
PROMPT = "The quick brown fox"

shutil.rmtree(BENCH_DIR, ignore_errors=True)
os.makedirs(BENCH_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model once for all snapshot strategies
# ---------------------------------------------------------------------------
log.info("Loading model %s for snapshot creation...", MODEL_ID)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cpu")
model.eval()

# Get reference output
inputs = tokenizer(PROMPT, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
REFERENCE_OUTPUT = tokenizer.decode(out[0], skip_special_tokens=True)
log.info("Reference output: %s", REFERENCE_OUTPUT[:80])

# ---------------------------------------------------------------------------
# Strategy A: Current approach (cloudpickle everything)
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
log.info("STRATEGY A: cloudpickle (current baseline)")
log.info("=" * 60)

snap_a = os.path.join(BENCH_DIR, "snap_a")
from zerostart.snapshot import snapshot, hydrate
t0 = time.monotonic()
snapshot(state={"model": model, "tokenizer": tokenizer}, path=snap_a)
log.info("  Snapshot A: %.2fs", time.monotonic() - t0)

# Warm hydrate
t0 = time.monotonic()
restored = hydrate(snap_a)
t_a_warm = time.monotonic() - t0
m = restored["model"]; m.eval()
tok = restored["tokenizer"]
inp = tok(PROMPT, return_tensors="pt")
with torch.no_grad():
    o = m.generate(**inp, max_new_tokens=10, do_sample=False)
result_a = tok.decode(o[0], skip_special_tokens=True)
log.info("  Warm hydrate A: %.3fs  output_match=%s", t_a_warm, result_a == REFERENCE_OUTPUT)
del restored, m, tok; gc.collect()

# ---------------------------------------------------------------------------
# Strategy B: save_pretrained tokenizer, cloudpickle model placeholder only
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
log.info("STRATEGY B: save_pretrained tokenizer + cloudpickle model only")
log.info("=" * 60)

snap_b = os.path.join(BENCH_DIR, "snap_b")
os.makedirs(snap_b, exist_ok=True)
t0 = time.monotonic()

# Save tokenizer via save_pretrained (JSON, no pickle)
tok_dir = os.path.join(snap_b, "tokenizer")
tokenizer.save_pretrained(tok_dir)

# Save model snapshot (without tokenizer — so cloudpickle is smaller)
snapshot(state={"model": model}, path=snap_b)
t_snap_b = time.monotonic() - t0
log.info("  Snapshot B: %.2fs", t_snap_b)

# Warm hydrate
t0 = time.monotonic()
restored = hydrate(snap_b)
tok_b = AutoTokenizer.from_pretrained(tok_dir)
t_b_warm = time.monotonic() - t0
m = restored["model"]; m.eval()
inp = tok_b(PROMPT, return_tensors="pt")
with torch.no_grad():
    o = m.generate(**inp, max_new_tokens=10, do_sample=False)
result_b = tok_b.decode(o[0], skip_special_tokens=True)
log.info("  Warm hydrate B: %.3fs  output_match=%s", t_b_warm, result_b == REFERENCE_OUTPUT)
del restored, m, tok_b; gc.collect()

# ---------------------------------------------------------------------------
# Strategy C: No cloudpickle — tokenizer JSON + model config JSON
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
log.info("STRATEGY C: JSON-only (no cloudpickle at all)")
log.info("=" * 60)

snap_c = os.path.join(BENCH_DIR, "snap_c")
os.makedirs(snap_c, exist_ok=True)
t0 = time.monotonic()

# Save tokenizer via save_pretrained
tok_dir_c = os.path.join(snap_c, "tokenizer")
tokenizer.save_pretrained(tok_dir_c)

# Save model config as JSON + safetensors refs (reuse snapshot's manifest logic)
from zerostart.snapshot import (
    _find_safetensors_for_model, _build_tensor_to_file_map,
    _extract_model_config, _environment_fingerprint,
)

sf_files = _find_safetensors_for_model(model)
tensor_to_file = _build_tensor_to_file_map(sf_files)

# Build tensor refs
sd = model.state_dict()
tensor_refs = {}
for param_name, tensor in sd.items():
    ref_key = f"model.{param_name}"
    ref_info = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    # Match to safetensors
    candidates = [ref_key]
    remaining = ref_key
    while "." in remaining:
        remaining = remaining.split(".", 1)[1]
        candidates.append(remaining)
    for candidate in candidates:
        if candidate in tensor_to_file:
            sf_path, sf_tensor_name = tensor_to_file[candidate]
            ref_info["safetensors_file"] = str(sf_path)
            ref_info["safetensors_tensor"] = sf_tensor_name
            break
    tensor_refs[ref_key] = ref_info

model_config_data = _extract_model_config(model)
manifest_c = {
    "version": 2,
    "strategy": "json_only",
    "fingerprint": _environment_fingerprint(),
    "model_config": model_config_data,
    "tensor_refs": tensor_refs,
    "safetensors_files": [str(f) for f in sf_files],
    "tensor_keys": [f"model.{k}" for k in sd.keys()],
}
with open(os.path.join(snap_c, "manifest.json"), "w") as f:
    json.dump(manifest_c, f)
t_snap_c = time.monotonic() - t0
log.info("  Snapshot C: %.2fs", t_snap_c)

# Hydrate C (warm)
def hydrate_c(snap_dir, _model_class=None, _config_class=None):
    """JSON-only hydrate — no cloudpickle."""
    t0 = time.monotonic()
    with open(os.path.join(snap_dir, "manifest.json")) as f:
        manifest = json.load(f)

    # Load tokenizer from saved files (uses tokenizers Rust lib internally)
    tok_path = os.path.join(snap_dir, "tokenizer")

    # Try fast tokenizers Rust library first (no transformers import)
    try:
        from tokenizers import Tokenizer as RustTokenizer
        # Check if tokenizer.json exists (fast tokenizer format)
        tj = os.path.join(tok_path, "tokenizer.json")
        if os.path.exists(tj):
            _tok = RustTokenizer.from_file(tj)
            # Wrap in a minimal class that has __call__ and decode
            tok = _FastTokenizerWrapper(_tok, tok_path)
        else:
            raise FileNotFoundError("No tokenizer.json")
    except (ImportError, FileNotFoundError):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_path)
    t_tok = time.monotonic()

    # Load tensors via mmap
    from safetensors.torch import load_file
    tensor_refs = manifest["tensor_refs"]
    file_to_tensors = {}
    for ref_key, ref_info in tensor_refs.items():
        sf_file = ref_info.get("safetensors_file")
        sf_tensor = ref_info.get("safetensors_tensor")
        if sf_file and sf_tensor:
            file_to_tensors.setdefault(sf_file, []).append((ref_key, sf_tensor))

    loaded_tensors = {}
    for sf_file, tensor_pairs in file_to_tensors.items():
        all_tensors = load_file(sf_file, device="cpu")
        for ref_key, sf_tensor_name in tensor_pairs:
            if sf_tensor_name in all_tensors:
                loaded_tensors[ref_key] = all_tensors[sf_tensor_name]
    t_mmap = time.monotonic()

    # Reconstruct model from config JSON
    mc = manifest["model_config"]
    import importlib
    if _model_class is None:
        model_module = importlib.import_module(mc["_module"])
        _model_class = getattr(model_module, mc["_class"])
    if _config_class is None:
        config_module = importlib.import_module(mc["config_module"])
        _config_class = getattr(config_module, mc["config_class"])

    model_config = _config_class.from_dict(mc["config_dict"])

    from zerostart.snapshot import _no_init_weights, _materialize_meta_tensors
    with _no_init_weights():
        with torch.device("meta"):
            mdl = _model_class(model_config)

    # Build state dict
    state_dict = {}
    for ref_key, tensor in loaded_tensors.items():
        # Strip "model." prefix
        param_name = ref_key.split(".", 1)[1] if "." in ref_key else ref_key
        state_dict[param_name] = tensor

    mdl.load_state_dict(state_dict, strict=False, assign=True)
    if hasattr(mdl, "tie_weights"):
        mdl.tie_weights()
    _materialize_meta_tensors(mdl)
    t_model = time.monotonic()

    log.info("  hydrate_c: tok=%.3fs mmap=%.3fs model=%.3fs total=%.3fs",
             t_tok - t0, t_mmap - t_tok, t_model - t_mmap, t_model - t0)
    return {"model": mdl, "tokenizer": tok}


class _FastTokenizerWrapper:
    """Minimal wrapper around tokenizers.Tokenizer for generate() compat."""

    def __init__(self, rust_tokenizer, tok_dir):
        self._tok = rust_tokenizer
        self._tok_dir = tok_dir
        # Load special tokens from tokenizer_config.json
        config_path = os.path.join(tok_dir, "tokenizer_config.json")
        self._config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                self._config = json.load(f)
        self.eos_token_id = self._config.get("eos_token_id")
        self.pad_token_id = self._config.get("pad_token_id", self.eos_token_id)

    def __call__(self, text, return_tensors=None, **kwargs):
        encoded = self._tok.encode(text)
        ids = encoded.ids
        mask = encoded.attention_mask
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)


t0 = time.monotonic()
restored_c = hydrate_c(snap_c)
t_c_warm = time.monotonic() - t0
m = restored_c["model"]; m.eval()
tok_c = restored_c["tokenizer"]
inp = tok_c(PROMPT, return_tensors="pt")
with torch.no_grad():
    o = m.generate(**inp, max_new_tokens=10, do_sample=False)
result_c = tok_c.decode(o[0], skip_special_tokens=True)
log.info("  Warm hydrate C: %.3fs  output_match=%s", t_c_warm, result_c == REFERENCE_OUTPUT)
del restored_c, m; gc.collect()

# ---------------------------------------------------------------------------
# Strategy D: torch.jit.trace (no transformers at hydrate time at all)
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
log.info("STRATEGY D: torch.jit.trace (no transformers import on hydrate)")
log.info("=" * 60)

snap_d = os.path.join(BENCH_DIR, "snap_d")
os.makedirs(snap_d, exist_ok=True)
t0 = time.monotonic()

# Save tokenizer
tok_dir_d = os.path.join(snap_d, "tokenizer")
tokenizer.save_pretrained(tok_dir_d)

# Trace the model
example_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
try:
    traced = torch.jit.trace(model, example_ids)
    torch.jit.save(traced, os.path.join(snap_d, "model.pt"))
    t_snap_d = time.monotonic() - t0
    log.info("  Snapshot D (jit.trace): %.2fs", t_snap_d)
    jit_ok = True
except Exception as e:
    log.warning("  torch.jit.trace FAILED: %s", e)
    jit_ok = False

if jit_ok:
    # Warm hydrate D
    t0 = time.monotonic()
    jit_model = torch.jit.load(os.path.join(snap_d, "model.pt"))
    from tokenizers import Tokenizer as RustTokenizer
    tj = os.path.join(tok_dir_d, "tokenizer.json")
    rust_tok = RustTokenizer.from_file(tj)
    tok_d = _FastTokenizerWrapper(rust_tok, tok_dir_d)
    t_d_warm = time.monotonic() - t0

    inp = tok_d(PROMPT, return_tensors="pt")
    with torch.no_grad():
        o = jit_model(inp["input_ids"])
    # jit trace gives logits, need to do generate manually or check output
    log.info("  Warm hydrate D: %.3fs (jit.load + tokenizer)", t_d_warm)
    log.info("  NOTE: jit model returns logits, not generate() — skipping output match")
    del jit_model; gc.collect()

# ---------------------------------------------------------------------------
# Strategy E: torch.export (newer alternative to jit)
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
log.info("STRATEGY E: torch.export (no transformers import on hydrate)")
log.info("=" * 60)

snap_e = os.path.join(BENCH_DIR, "snap_e")
os.makedirs(snap_e, exist_ok=True)
t0 = time.monotonic()

try:
    exported = torch.export.export(model, (example_ids,))
    torch.export.save(exported, os.path.join(snap_e, "model.pt2"))
    t_snap_e = time.monotonic() - t0
    log.info("  Snapshot E (export): %.2fs", t_snap_e)
    export_ok = True
except Exception as e:
    log.warning("  torch.export FAILED: %s", e)
    export_ok = False
    t_snap_e = 0

if export_ok:
    t0 = time.monotonic()
    loaded = torch.export.load(os.path.join(snap_e, "model.pt2"))
    t_e_warm = time.monotonic() - t0
    log.info("  Warm hydrate E: %.3fs (export.load)", t_e_warm)

# ---------------------------------------------------------------------------
# Cold benchmarks: run each hydrate in a subprocess (fresh Python)
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
log.info("COLD HYDRATE BENCHMARKS (subprocess, fresh Python)")
log.info("=" * 60)

def cold_bench(name, script):
    """Run a hydrate script in a fresh Python process and measure wall clock."""
    script_path = os.path.join(BENCH_DIR, f"cold_{name}.py")
    with open(script_path, "w") as f:
        f.write(script)

    # Drop page cache if possible (need root)
    try:
        subprocess.run("sync && echo 3 > /proc/sys/vm/drop_caches",
                       shell=True, capture_output=True, timeout=5)
    except Exception:
        pass

    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    elapsed = time.monotonic() - t0
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    # Extract timing lines
    for line in (stdout + "\n" + stderr).split("\n"):
        if "COLD_TIME" in line or "RESULT" in line or "hydrate" in line or "zerostart" in line:
            log.info("    %s", line.strip())
    if result.returncode != 0:
        log.warning("    FAILED (exit %d): %s", result.returncode, stderr[-500:])
    return elapsed, result.returncode == 0


# Cold A: cloudpickle (current)
log.info("")
log.info("--- Cold A: cloudpickle (current) ---")
t_cold_a, ok_a = cold_bench("a", f"""
import time, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()
import torch
t_torch = time.monotonic()
from zerostart.snapshot import hydrate
t_import = time.monotonic()
restored = hydrate("{snap_a}")
model = restored["model"]; model.eval()
tokenizer = restored["tokenizer"]
t_hydrate = time.monotonic()
inputs = tokenizer("{PROMPT}", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
result = tokenizer.decode(out[0], skip_special_tokens=True)
t_done = time.monotonic()
print(f"RESULT: {{result}}")
print(f"COLD_TIME torch={{t_torch-t0:.2f}}s import={{t_import-t_torch:.2f}}s hydrate={{t_hydrate-t_import:.2f}}s inference={{t_done-t_hydrate:.2f}}s total={{t_done-t0:.2f}}s")
""")
log.info("  Cold A wall clock: %.2fs", t_cold_a)

# Cold B: save_pretrained tokenizer
log.info("")
log.info("--- Cold B: save_pretrained tokenizer ---")
t_cold_b, ok_b = cold_bench("b", f"""
import time, logging
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()
import torch
t_torch = time.monotonic()
from zerostart.snapshot import hydrate
t_import = time.monotonic()
restored = hydrate("{snap_b}")
t_hydrate = time.monotonic()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("{os.path.join(snap_b, 'tokenizer')}")
model = restored["model"]; model.eval()
t_ready = time.monotonic()
inputs = tokenizer("{PROMPT}", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
result = tokenizer.decode(out[0], skip_special_tokens=True)
t_done = time.monotonic()
print(f"RESULT: {{result}}")
print(f"COLD_TIME torch={{t_torch-t0:.2f}}s import={{t_import-t_torch:.2f}}s hydrate={{t_hydrate-t_import:.2f}}s tokenizer={{t_ready-t_hydrate:.2f}}s inference={{t_done-t_ready:.2f}}s total={{t_done-t0:.2f}}s")
""")
log.info("  Cold B wall clock: %.2fs", t_cold_b)

# Cold C: JSON-only (no cloudpickle, try Rust tokenizer first)
log.info("")
log.info("--- Cold C: JSON-only (Rust tokenizer, no cloudpickle) ---")
# Write the hydrate_c + wrapper code as a standalone script
t_cold_c, ok_c = cold_bench("c", f"""
import time, logging, json, os, sys
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
log = logging.getLogger("cold_c")
t0 = time.monotonic()
import torch
t_torch = time.monotonic()

snap_dir = "{snap_c}"

# 1. Load manifest
with open(os.path.join(snap_dir, "manifest.json")) as f:
    manifest = json.load(f)
t_manifest = time.monotonic()

# 2. Load tokenizer (Rust, no transformers)
from tokenizers import Tokenizer as RustTokenizer
tj = os.path.join(snap_dir, "tokenizer", "tokenizer.json")
rust_tok = RustTokenizer.from_file(tj)
# Load config for special tokens
with open(os.path.join(snap_dir, "tokenizer", "tokenizer_config.json")) as f:
    tok_config = json.load(f)
t_tok = time.monotonic()

# 3. mmap tensors
from safetensors.torch import load_file
tensor_refs = manifest["tensor_refs"]
file_to_tensors = {{}}
for ref_key, ref_info in tensor_refs.items():
    sf_file = ref_info.get("safetensors_file")
    sf_tensor = ref_info.get("safetensors_tensor")
    if sf_file and sf_tensor:
        file_to_tensors.setdefault(sf_file, []).append((ref_key, sf_tensor))

loaded_tensors = {{}}
for sf_file, tensor_pairs in file_to_tensors.items():
    all_tensors = load_file(sf_file, device="cpu")
    for ref_key, sf_tensor_name in tensor_pairs:
        if sf_tensor_name in all_tensors:
            loaded_tensors[ref_key] = all_tensors[sf_tensor_name]
t_mmap = time.monotonic()

# 4. Reconstruct model (this is where transformers import happens)
mc = manifest["model_config"]
import importlib
model_module = importlib.import_module(mc["_module"])
model_class = getattr(model_module, mc["_class"])
config_module = importlib.import_module(mc["config_module"])
config_class = getattr(config_module, mc["config_class"])
t_import_model = time.monotonic()

model_config = config_class.from_dict(mc["config_dict"])

from zerostart.snapshot import _no_init_weights, _materialize_meta_tensors
with _no_init_weights():
    with torch.device("meta"):
        mdl = model_class(model_config)

state_dict = {{}}
for ref_key, tensor in loaded_tensors.items():
    param_name = ref_key.split(".", 1)[1] if "." in ref_key else ref_key
    state_dict[param_name] = tensor

mdl.load_state_dict(state_dict, strict=False, assign=True)
if hasattr(mdl, "tie_weights"):
    mdl.tie_weights()
_materialize_meta_tensors(mdl)
mdl.eval()
t_model = time.monotonic()

# 5. Inference
encoded = rust_tok.encode("{PROMPT}")
input_ids = torch.tensor([encoded.ids], dtype=torch.long)
attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.long)
with torch.no_grad():
    out = mdl.generate(input_ids=input_ids, attention_mask=attention_mask,
                       max_new_tokens=10, do_sample=False)
result = rust_tok.decode(out[0].tolist(), skip_special_tokens=True)
t_done = time.monotonic()

print(f"RESULT: {{result}}")
print(f"COLD_TIME torch={{t_torch-t0:.2f}}s manifest={{t_manifest-t_torch:.2f}}s tok={{t_tok-t_manifest:.2f}}s mmap={{t_mmap-t_tok:.2f}}s import_model={{t_import_model-t_mmap:.2f}}s reconstruct={{t_model-t_import_model:.2f}}s inference={{t_done-t_model:.2f}}s total={{t_done-t0:.2f}}s")
""")
log.info("  Cold C wall clock: %.2fs", t_cold_c)

# Cold D: torch.jit (if it worked)
if jit_ok:
    log.info("")
    log.info("--- Cold D: torch.jit.load (no transformers) ---")
    t_cold_d, ok_d = cold_bench("d", f"""
import time, logging, json, os
logging.basicConfig(level=logging.INFO, format="%(name)-20s %(message)s")
t0 = time.monotonic()
import torch
t_torch = time.monotonic()

# Load tokenizer (Rust)
from tokenizers import Tokenizer as RustTokenizer
rust_tok = RustTokenizer.from_file("{os.path.join(snap_d, 'tokenizer', 'tokenizer.json')}")
t_tok = time.monotonic()

# Load traced model
jit_model = torch.jit.load("{os.path.join(snap_d, 'model.pt')}")
t_load = time.monotonic()

# Inference (jit model takes input_ids, returns logits)
encoded = rust_tok.encode("{PROMPT}")
input_ids = torch.tensor([encoded.ids], dtype=torch.long)
with torch.no_grad():
    logits = jit_model(input_ids)
# Greedy decode from logits
next_tokens = []
for _ in range(10):
    next_id = logits[0, -1].argmax().item()
    next_tokens.append(next_id)
    input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
    with torch.no_grad():
        logits = jit_model(input_ids)
result = rust_tok.decode(encoded.ids + next_tokens, skip_special_tokens=True)
t_done = time.monotonic()

print(f"RESULT: {{result}}")
print(f"COLD_TIME torch={{t_torch-t0:.2f}}s tok={{t_tok-t_torch:.2f}}s jit_load={{t_load-t_tok:.2f}}s inference={{t_done-t_load:.2f}}s total={{t_done-t0:.2f}}s")
""")
    log.info("  Cold D wall clock: %.2fs", t_cold_d)
else:
    t_cold_d = 0
    ok_d = False

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print(f"HYDRATE STRATEGY BENCHMARK — {MODEL_ID}")
print(f"{'Strategy':<50} {'Warm':>8} {'Cold':>8}")
print("-" * 70)
print(f"{'A: cloudpickle (current)':<50} {t_a_warm:>7.2f}s {t_cold_a:>7.2f}s")
print(f"{'B: save_pretrained tok + cloudpickle model':<50} {t_b_warm:>7.2f}s {t_cold_b:>7.2f}s")
print(f"{'C: JSON-only (Rust tok, no cloudpickle)':<50} {t_c_warm:>7.2f}s {t_cold_c:>7.2f}s")
if jit_ok:
    print(f"{'D: torch.jit.trace (no transformers)':<50} {t_d_warm:>7.2f}s {t_cold_d:>7.2f}s")
else:
    print(f"{'D: torch.jit.trace':<50} {'FAILED':>8} {'N/A':>8}")
if export_ok:
    print(f"{'E: torch.export':<50} {t_e_warm:>7.2f}s {'N/A':>8}")
else:
    print(f"{'E: torch.export':<50} {'FAILED':>8} {'N/A':>8}")
print("=" * 70)
print()
print("Cold = fresh subprocess (no modules pre-imported)")
print("Warm = in-process (torch + transformers already imported)")

# Cleanup
shutil.rmtree(BENCH_DIR)
