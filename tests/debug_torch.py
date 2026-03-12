# /// script
# dependencies = ["torch>=2.5"]
# ///
import sys
print("sys.path:")
for p in sys.path:
    print(f"  {p}")
import torch
print(f"torch: {torch.__version__} at {torch.__file__}")
try:
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
    print("flex_attention: OK")
except ImportError as e:
    print(f"flex_attention: FAILED - {e}")
