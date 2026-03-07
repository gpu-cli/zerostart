"""Demo app that imports packages progressively as they install."""
import time

t0 = time.monotonic()

print(f"[{time.monotonic()-t0:.2f}s] App starting...")

import requests
print(f"[{time.monotonic()-t0:.2f}s] requests {requests.__version__}")

import yaml
print(f"[{time.monotonic()-t0:.2f}s] yaml {yaml.__version__}")

import six
print(f"[{time.monotonic()-t0:.2f}s] six {six.__version__}")

# Actually use the packages
resp_data = {"status": "ok", "yaml_version": yaml.__version__}
print(f"[{time.monotonic()-t0:.2f}s] App ready: {resp_data}")
