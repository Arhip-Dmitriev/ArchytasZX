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

"""The cluster profile's Slurm job array: ``python -m benchmarks slurm [--set k=v ...]`` writes
benchmarks/slurm/run.sbatch (submitted from the repository root; scavenger partition, requeue,
whole cores, one worker per core, one shard of the units per array task) and prints the sbatch
command. The checkout must be at origin/dev's latest green commit (or slurm.commit): a
clean checkout elsewhere is moved there, a dirty one is refused, and every array task checks
the commit again before it runs."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return r.stdout.strip()


def green_commit(cfg: dict[str, Any]) -> str | None:
    """slurm.commit when set, else the head of origin/dev's latest successful CI run (gh)."""
    if cfg["slurm"].get("commit"):
        return str(cfg["slurm"]["commit"])
    cmd = ["gh", "run", "list", "--branch", "dev", "--status", "success", "--workflow"]
    cmd += ["ci.yml", "--limit", "1", "--json", "headSha", "-q", ".[0].headSha"]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None


def checkout(cfg: dict[str, Any]) -> str:
    """The green commit, the checkout moved onto it when clean; exits when it cannot be."""
    target = green_commit(cfg)
    if not target:
        sys.exit("slurm: no green dev commit found (gh unavailable?); set slurm.commit=SHA")
    _git("fetch", "-q", "origin", "dev")
    sha = _git("rev-parse", "--verify", f"{target}^{{commit}}")
    if not sha:
        sys.exit(f"slurm: commit {target} not in this checkout")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    if dirty:
        sys.exit("slurm: checkout has local changes")
    if _git("rev-parse", "HEAD") != sha:
        _git("checkout", "-q", "--detach", sha)
        if _git("rev-parse", "HEAD") != sha:
            sys.exit(f"slurm: could not check out {sha}")
    return sha


def script(cfg: dict[str, Any], sets: list[str], commit: str) -> str:
    """The sbatch file of the config, run only at ``commit`` with a clean checkout."""
    sc = cfg["slurm"]
    if sc["partition"] != "scavenger":
        raise ValueError("Zaratan jobs run on the scavenger partition only")
    extra = " ".join(f"--set {shlex.quote(s)}" for s in sets)
    account = f"#SBATCH --account={sc['account']}\n" if sc["account"] else ""
    return f"""#!/bin/bash
#SBATCH --job-name=archytaszx-bench
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --array=0-{int(sc["array"]) - 1}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={int(sc["cpus_per_task"])}
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu={sc["mem_per_cpu"]}
#SBATCH --time={sc["time"]}
#SBATCH --output=benchmarks/slurm/logs/%A_%a.out
{account}
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ "$(git rev-parse HEAD)" = "{commit}" ] || {{ echo "checkout is not at {commit}" >&2; exit 1; }}
[ -z "$(git status --porcelain --untracked-files=no)" ] || {{ echo "dirty checkout" >&2; exit 1; }}
source {sc["venv"]}/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"
python -m benchmarks run --profile cluster --shard ${{SLURM_ARRAY_TASK_ID}}/{int(sc["array"])} \\
    --set platform.jobs={int(sc["cpus_per_task"])} {extra}
"""


def main(cfg: dict[str, Any], args: Any) -> None:
    """Checks out the green commit, writes the sbatch file and prints how to submit it."""
    cfg["platform"]["profile"] = "cluster"
    out = ROOT / "benchmarks" / "slurm"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    commit = checkout(cfg)
    path = out / "run.sbatch"
    path.write_text(script(cfg, list(args.sets), commit), encoding="utf-8")
    print(f"wrote {path} at {commit}; submit with: sbatch {path.relative_to(ROOT)}")
