# Copyright 2026 Arkhip A. Dmitriev
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The environment record: machine, OS, Python and library versions, Lean and Mathlib, and the
git commit with whether the tree is dirty."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LEAN_DIR = ROOT / "archytaszx" / "lean"
_PACKAGES = ("numpy", "sympy", "pyzx", "qiskit", "matplotlib")


def _out(cmd: list[str], cwd: Path = ROOT) -> str:
    try:
        done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip()


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _memory_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        pass
    return None


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _mathlib() -> str | None:
    try:
        manifest = json.loads((LEAN_DIR / "lake-manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for package in manifest.get("packages", []):
        if package.get("name") == "mathlib":
            return f"{package.get('inputRev')} ({package.get('rev')})"
    return None


def record() -> dict[str, Any]:
    """The environment record."""
    from benchmarks.lean import lake

    toolchain = LEAN_DIR / "lean-toolchain"
    lean_version = _out([lake(), "env", "lean", "--version"], LEAN_DIR)
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in _PACKAGES},
        "archytaszx": _version("archytaszx"),
        "lean_toolchain": toolchain.read_text(encoding="utf-8").strip()
        if toolchain.exists()
        else None,
        "lean": lean_version or None,
        "mathlib": _mathlib(),
        "git_commit": _out(["git", "rev-parse", "HEAD"]) or None,
        "git_dirty": bool(_out(["git", "status", "--porcelain", "--untracked-files=no"])),
    }


def write(path: Path, **run: object) -> None:
    """Writes the environment record and ``run`` (the run's options) to ``path``."""
    path.write_text(json.dumps({**record(), "run": run}, indent=1), encoding="utf-8")
