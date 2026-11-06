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

"""``python -m benchmarks anc OUT_DIR``: the EPTCS ancillary files. Copies the benchmark
scripts, the case files (benchmarks/workloads/, examples/ and the lemma corpus), every results
file of the results directory and every plot of the plots directory into OUT_DIR, with a plain
text README. OUT_DIR must not lie under publications/ or public_publications/."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from benchmarks import harness, plots

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = ("publications", "public_publications")
README = """ARCHYTASZX BENCHMARKS: ANCILLARY FILES
=====================================

Generated {date} by python -m benchmarks anc.

  scripts/     the benchmark suites (python -m benchmarks run SUITE; see BENCHMARKS.txt)
  workloads/   the command files the suites run (benchmarks/workloads/, examples/, and the
               lemma lists archytaszx/papers/*/LEMMAS.txt)
  results/     every suite's rows (SUITE.json), runs (SUITE.runs.jsonl) and environment
               record (SUITE.env.json)
  plots/       every plot (PDF and 300 dpi PNG) and every summary table
  BENCHMARKS.txt  the benchmark documentation

Reproduce: install ArchytasZX (docs/INSTALL.txt), copy scripts/ to benchmarks/ in a checkout of
the commit named in results/*.env.json, and run the commands of BENCHMARKS.txt, section 5.
"""


def export(out: Path) -> None:
    """Writes the ancillary files to ``out``."""
    out = out.resolve()
    for name in FORBIDDEN:
        banned = (ROOT / name).resolve()
        if out == banned or banned in out.parents:
            raise SystemExit(f"refusing to write under {banned}")
    out.mkdir(parents=True, exist_ok=True)
    bench = ROOT / "benchmarks"
    scripts = out / "scripts"
    scripts.mkdir(exist_ok=True)
    for path in sorted(bench.glob("*.py")):
        shutil.copy2(path, scripts / path.name)
    work = out / "workloads"
    shutil.copytree(bench / "workloads", work / "benchmarks", dirs_exist_ok=True)
    shutil.copytree(
        ROOT / "examples",
        work / "examples",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.lean", "*.png"),
    )
    for key in ("poor2023zxw", "wang2023qufinitezxw"):
        target = work / "papers" / key
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "archytaszx" / "papers" / key / "LEMMAS.txt", target / "LEMMAS.txt")
    for source, name in ((harness.results_dir(), "results"), (plots.plots_dir(), "plots")):
        if source.exists():
            shutil.copytree(source, out / name, dirs_exist_ok=True)
    doc = ROOT / "docs" / "BENCHMARKS.txt"
    if doc.exists():
        shutil.copy2(doc, out / "BENCHMARKS.txt")
    (out / "README").write_text(README.format(date=time.strftime("%Y-%m-%d")), encoding="utf-8")


def main(argv: list[str]) -> None:
    """Command-line entry: ``anc OUT_DIR``."""
    if len(argv) != 1:
        raise SystemExit("usage: python -m benchmarks anc OUT_DIR")
    export(Path(argv[0]))


if __name__ == "__main__":
    main(sys.argv[1:])
