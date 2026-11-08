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

"""The ancillary-file export: ``python -m benchmarks anc OUT_DIR`` copies the benchmark
scripts, the subject catalogue, the config, the logs, the aggregates, the plots and
docs/BENCHMARKS.txt into OUT_DIR."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from benchmarks import config

ROOT = Path(__file__).resolve().parent.parent
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".venvs", "results", "plots", "slurm")


def main(cfg: dict[str, Any], out: Path) -> None:
    """Copies everything listed in the module docstring into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "benchmarks", out / "benchmarks", ignore=IGNORE, dirs_exist_ok=True)
    results = config.path(cfg, "results")
    for sub in ("raw", "aggregate", "environment", "configs"):
        if (results / sub).exists():
            shutil.copytree(results / sub, out / "results" / sub, dirs_exist_ok=True)
    if (results / "false_set.json").exists():
        shutil.copy2(results / "false_set.json", out / "results" / "false_set.json")
    plots = config.path(cfg, "plots")
    if plots.exists():
        shutil.copytree(plots, out / "plots", dirs_exist_ok=True)
    (out / "docs").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "docs" / "BENCHMARKS.txt", out / "docs" / "BENCHMARKS.txt")
    print(f"anc -> {out}")
