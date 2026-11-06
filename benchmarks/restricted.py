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


"""Benchmark: restricted claims (PROOFS.txt, section 8). Rows: GHZ-state preparation under
each clause of :data:`CLAUSES`, and every verdict unit of ``examples/restricted/`` except
``errors.azx``; each decided as written, then ``certificate kernel``, ``certificate lean`` and
the Lean runner.

Fields: ``workload``, ``clause``, ``verdict``, ``method``, ``time`` (Lean CPU seconds),
``script_cpu``, ``lean_status``, ``axiom_class``, ``statement_level``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from benchmarks import lean, workloads

SIZE = "size"
REPEAT = 3
RECORD_ENV = True
CLAUSES = (
    "n = 0",
    "n = 3",
    "n >= 1",
    "n >= 2",
    "n >= 3",
    "n = 2*m",
    "n = 2*m + 1",
    "n = 3*m",
    "n = 2*m + 2",
    "d = 3",
    "d >= 2",
    "n >= 2, d = 3",
    "n = 3, d = 5",
)
QUICK_CLAUSES = ("n >= 2", "d = 3")
EXAMPLES = workloads.ROOT / "examples" / "restricted"


def ghz_script(clause: str) -> str:
    """GHZ preparation decided against the GHZ state under ``clause``."""
    text = workloads.family_script("GHZ")
    head = text.split("\ndecide", 1)[0]
    return f"{head}\ndecide against @rhs for {clause}\n"


def example_units() -> dict[str, str]:
    """``FILE #i`` to the i-th verdict unit of each example file."""
    out: dict[str, str] = {}
    for path in sorted(EXAMPLES.glob("*.azx")):
        if path.stem == "errors":
            continue
        for i, unit in enumerate(workloads.units(path.read_text(encoding="utf-8")), 1):
            out[f"{path.stem} #{i}"] = unit
    return out


def cases(quick: bool) -> list[dict[str, Any]]:
    """GHZ under every clause, then every example unit."""
    clauses = QUICK_CLAUSES if quick else CLAUSES
    rows = [{"suite": "restricted", "workload": "GHZ", "clause": c, "size": 1} for c in clauses]
    units = list(example_units())
    rows += [
        {"suite": "restricted", "workload": u, "clause": "as written", "size": 1}
        for u in (units[:1] if quick else units)
    ]
    return rows


def run(case: dict[str, Any]) -> dict[str, Any]:
    """Decide, export and check one restricted claim."""
    if case["workload"] == "GHZ":
        script, base = ghz_script(case["clause"]), str(workloads.ROOT / "examples")
    else:
        script, base = example_units()[case["workload"]], str(EXAMPLES)
    with tempfile.TemporaryDirectory(prefix="bench-restricted-") as tmp:
        out = lean.export_and_check(script, base, Path(tmp) / "restricted.lean")
    if out.get("verdict") == "unequal" and case["workload"] == "GHZ":
        return out | {"status": "wrong", "detail": "a true restricted GHZ claim refuted"}
    return out
