"""Environment cache: simplified — just tracks venv directories.

The heavy lifting (resolution, download caching, hardlinking) is done by uv.
We just manage which venvs exist and whether they're complete.

Layout:
    ~/.cache/zerostart/
      envs/{key}/          # venv for a set of requirements
      wheels/              # daemon-downloaded .whl files (fed to uv for caching)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("zerostart.cache")
