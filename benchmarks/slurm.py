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
command."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def script(cfg: dict[str, Any], sets: list[str]) -> str:
    """The sbatch file of the config."""
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
source {sc["venv"]}/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"
python -m benchmarks run --profile cluster --shard ${{SLURM_ARRAY_TASK_ID}}/{int(sc["array"])} \\
    --set platform.jobs={int(sc["cpus_per_task"])} {extra}
"""


def main(cfg: dict[str, Any], args: Any) -> None:
    """Writes the sbatch file and prints how to submit it."""
    cfg["platform"]["profile"] = "cluster"
    out = ROOT / "benchmarks" / "slurm"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    path = out / "run.sbatch"
    path.write_text(script(cfg, list(args.sets)), encoding="utf-8")
    print(f"wrote {path}; submit with: sbatch {path.relative_to(ROOT)}")
