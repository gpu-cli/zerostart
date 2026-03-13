# /// script
# dependencies = ["torch>=2.5", "transformers", "fastapi", "uvicorn", "accelerate"]
# ///
"""Inference server: measures time-to-health-check and time-to-first-inference."""
import time, os, sys, threading

T_START = float(os.environ.get("BENCH_T_START", "0"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def elapsed():
    return (time.time() * 1000 - T_START) / 1000

# Phase 1: import web framework (small packages)
t0 = time.time()
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
t_web = time.time()
print(f"  [phase1] fastapi+uvicorn imported: {t_web - t0:.1f}s (wall: {elapsed():.1f}s)")

app = FastAPI()
model_obj = None
model_ready = threading.Event()

@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "model_ready": model_ready.is_set(), "uptime_s": round(elapsed(), 1)})

@app.get("/predict")
def predict():
    if not model_ready.is_set():
        return JSONResponse({"error": "model not loaded"}, status_code=503)
    import torch
    x = torch.randn(1, 10, device="cuda")
    with torch.no_grad():
        out = model_obj(x)
    return JSONResponse({"result": out.tolist(), "time_s": round(elapsed(), 1)})

# Start uvicorn in background thread
server_config = uvicorn.Config(app, host="127.0.0.1", port=8199, log_level="error")
server = uvicorn.Server(server_config)
server_thread = threading.Thread(target=server.run, daemon=True)
server_thread.start()

# Wait for server to bind
import urllib.request
for _ in range(50):
    try:
        urllib.request.urlopen("http://127.0.0.1:8199/health", timeout=1)
        break
    except Exception:
        time.sleep(0.1)

t_health = time.time()
print(f"  [phase1] /health responding (wall: {elapsed():.1f}s)")
print(f"RESULT_HEALTH: {elapsed():.1f}s")

# Phase 2: heavy ML imports
t0 = time.time()
import torch
t_torch = time.time()
print(f"  [phase2] torch imported: {t_torch - t0:.1f}s (wall: {elapsed():.1f}s, v{torch.__version__})")

from transformers import AutoModelForCausalLM, AutoTokenizer
t_tf = time.time()
print(f"  [phase2] transformers imported: {t_tf - t_torch:.1f}s (wall: {elapsed():.1f}s)")

# Simple model (no HF download needed for benchmark)
model_obj = torch.nn.Linear(10, 5).cuda()
model_obj.eval()
model_ready.set()

# Warmup inference
with torch.no_grad():
    _ = model_obj(torch.randn(1, 10, device="cuda"))
t_infer = time.time()
print(f"  [phase2] first CUDA inference (wall: {elapsed():.1f}s)")
print(f"RESULT_INFER: {elapsed():.1f}s")

# Verify /predict works
resp = urllib.request.urlopen("http://127.0.0.1:8199/predict", timeout=5)
print(f"  [phase2] /predict verified: {resp.read().decode()[:80]}")

print(f"")
print(f"=== SUMMARY ===")
print(f"  Health check ready: {(t_health * 1000 - T_START)/1000:.1f}s")
print(f"  First inference:    {(t_infer * 1000 - T_START)/1000:.1f}s")

server.should_exit = True
time.sleep(0.5)
