"""Environment cache: plan-keyed reuse of completed environments.

Layout:
    .zerostart/
      inputs/{input_hash}      # maps raw requirements → env_key (fast warm lookup)
      plans/{env_key}.json     # cached artifact plan
      envs/{env_key}/          # completed environment (venv)
      envs/{env_key}.tmp/      # in-progress environment build
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

from zerostart.resolver import ArtifactPlan

log = logging.getLogger("zerostart.cache")

DEFAULT_CACHE_DIR = Path(".zerostart")


def _input_hash(
    requirements: list[str],
    python_version: str = "3.11",
    platform: str = "linux",
) -> str:
    """Hash raw input requirements for fast cache lookup."""
    payload = json.dumps({
        "requirements": sorted(requirements),
        "python_version": python_version,
        "platform": platform,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class EnvironmentCache:
    """Manages cached environments keyed by artifact plan."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.plans_dir = self.cache_dir / "plans"
        self.envs_dir = self.cache_dir / "envs"
        self.inputs_dir = self.cache_dir / "inputs"
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)

    def lookup_by_input(
        self,
        requirements: list[str],
        python_version: str = "3.11",
        platform: str = "linux",
    ) -> CachedEnv | None:
        """Fast warm-path lookup: skip resolution entirely.

        Checks if we've seen these exact input requirements before and
        have a completed environment for them.
        """
        ih = _input_hash(requirements, python_version, platform)
        input_file = self.inputs_dir / ih

        if not input_file.exists():
            return None

        env_key = input_file.read_text().strip()
        env_dir = self.envs_dir / env_key

        if not env_dir.exists() or not (env_dir / ".complete").exists():
            return None

        sp = _find_site_packages(env_dir)
        if not sp:
            return None

        return CachedEnv(
            env_dir=env_dir,
            site_packages=sp,
            is_complete=True,
            key=env_key,
        )

    def save_input_mapping(
        self,
        requirements: list[str],
        plan: ArtifactPlan,
        python_version: str = "3.11",
        platform: str = "linux",
    ) -> None:
        """Save mapping from raw requirements → env cache key."""
        ih = _input_hash(requirements, python_version, platform)
        input_file = self.inputs_dir / ih
        input_file.write_text(plan.cache_key)

    def lookup(self, plan: ArtifactPlan) -> CachedEnv | None:
        """Check if a completed environment exists for this plan."""
        key = plan.cache_key
        env_dir = self.envs_dir / key

        if env_dir.exists() and (env_dir / ".complete").exists():
            sp = _find_site_packages(env_dir)
            if sp:
                return CachedEnv(
                    env_dir=env_dir,
                    site_packages=sp,
                    is_complete=True,
                    key=key,
                )

        # Plan exists but env incomplete — can resume
        plan_path = self.plans_dir / f"{key}.json"
        if plan_path.exists():
            return CachedEnv(
                env_dir=env_dir,
                site_packages=_find_site_packages(env_dir) if env_dir.exists() else None,
                is_complete=False,
                key=key,
            )

        return None

    def create_env(self, plan: ArtifactPlan) -> CachedEnv:
        """Create a new environment for this plan.

        Builds in a .tmp directory, then renames to final location.
        """
        key = plan.cache_key
        env_dir = self.envs_dir / key
        tmp_dir = self.envs_dir / f"{key}.tmp"

        # Save plan
        plan_path = self.plans_dir / f"{key}.json"
        plan_path.write_text(json.dumps(plan.cache_key_payload, indent=2))

        # Clean up stale tmp if exists
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        # Create venv in tmp dir
        _create_venv(tmp_dir)

        # Move to final location
        if env_dir.exists():
            shutil.rmtree(env_dir)
        os.rename(str(tmp_dir), str(env_dir))

        sp = _find_site_packages(env_dir)
        return CachedEnv(
            env_dir=env_dir,
            site_packages=sp,
            is_complete=False,
            key=key,
        )

    def mark_complete(self, env: CachedEnv) -> None:
        """Mark an environment as fully installed."""
        (env.env_dir / ".complete").touch()
        env.is_complete = True


class CachedEnv:
    """A cached environment, possibly still being built."""

    def __init__(
        self,
        env_dir: Path,
        site_packages: Path | None,
        is_complete: bool,
        key: str,
    ):
        self.env_dir = env_dir
        self.site_packages = site_packages
        self.is_complete = is_complete
        self.key = key


def _find_site_packages(env_dir: Path) -> Path | None:
    """Find site-packages directory in a venv."""
    candidates = list(env_dir.glob("lib/python*/site-packages"))
    if candidates:
        return candidates[0]
    return None


def _create_venv(path: Path) -> None:
    """Create a venv at the given path."""
    # Try uv first (faster)
    uv = shutil.which("uv")
    if uv:
        subprocess.run(
            [uv, "venv", str(path)],
            check=True,
            capture_output=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "venv", str(path)],
            check=True,
            capture_output=True,
        )
