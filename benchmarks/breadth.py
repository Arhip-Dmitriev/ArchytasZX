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

"""Benchmark: which textbook algorithms and protocols the engine states and proves for every n
and d. Each workload is a command file under workloads/breadth/ whose header gives the target
from the algorithm's definition and its source by reference key; a candidate the syntax cannot
state is a row with status ``unsupported`` and the missing construct.

Tools: ``azx-free`` (the file as written; ``certificate kernel`` and ``certificate lean``
after the verdict, Lean-checked) and ``azx-bound`` (``dim d = 2`` added where the file uses the
default dimension, the count ``n`` bound to each size of :data:`NS`; files without ``n`` have
no bound rows).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from benchmarks import engine, lean, workloads
from benchmarks.harness import Inconclusive, Wrong

SIZE = "n"
RECORD_ENV = True
NS = [2**k for k in range(21)]
QUICK_NS = [1, 4]
QUICK_WORKLOADS = ("teleportation", "bernstein_vazirani_secret", "nested_depth2")
UNSUPPORTED = {
    "grover_every_n": "one Grover iteration for every n: the reflection 2|s><s| - I on n "
    "registers is a multi-controlled phase, not a repeated local gadget, so no bang box states it",
    "hidden_subgroup_secret_vector": "a secret vector s_1, ..., s_n varying per copy: a bang "
    "box repeats one subdiagram, so every copy carries the same s (fourier_sampling states "
    "s = (t, ..., t))",
    "phase_estimation_n_controls": "phase estimation with n control qudits: controlled-U^(d^j) "
    "differs per copy j (an exponent indexed by the copy), which a bang box cannot state",
}


def files() -> dict[str, Path]:
    """The breadth workloads by name."""
    return {p.stem: p for p in workloads.breadth_files()}


def _has_n(text: str) -> bool:
    return re.search(r"^\s*count\s+(?:[a-z, ]*\b)?n\b", text, re.MULTILINE) is not None


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every workload free, the bound sizes where it has a count n, every unsupported one."""
    out: list[dict[str, Any]] = []
    for name, path in files().items():
        if quick and name not in QUICK_WORKLOADS:
            continue
        out.append({"suite": "breadth", "tool": "azx-free", "workload": name, "n": "free"})
        if _has_n(path.read_text(encoding="utf-8")):
            out += [
                {"suite": "breadth", "tool": "azx-bound", "workload": name, "n": n}
                for n in (QUICK_NS if quick else NS)
            ]
    if not quick:
        out += [
            {"suite": "breadth", "tool": "azx-free", "workload": w, "n": "free"}
            for w in UNSUPPORTED
        ]
    return out


def bound(text: str, n: int) -> str:
    """``text`` with ``count n`` bound to ``n`` and, where the default dimension is used,
    ``dim d = 2`` declared in every block."""
    text = re.sub(r"^(\s*count\s+(?:[a-z]+, )*)n\b", rf"\g<1>n = {n}", text, flags=re.MULTILINE)
    uses_d = re.search(r"\bdim\s", text) is None and "literal" not in text
    if uses_d:
        text = re.sub(r"^(@\w+ =|load)\n(\s+)", r"\1\n\2dim d = 2\n\2", text, flags=re.MULTILINE)
    return text


def run(case: dict[str, Any]) -> dict[str, Any]:
    """One workload: the verdict, and for ``azx-free`` its kernel and Lean check."""
    name = case["workload"]
    if name in UNSUPPORTED:
        return {
            "status": "unsupported",
            "detail": UNSUPPORTED[name],
            "cpu": 0.0,
            "wall": 0.0,
            "expressible": False,
        }
    path = files()[name]
    text = path.read_text(encoding="utf-8")
    script = workloads.units(text)[0]
    out: dict[str, Any] = {"expressible": True}
    if case["tool"] == "azx-bound":
        run_ = engine.run_script(bound(script, case["n"]), base=str(path.parent))
        v = engine.verdict(run_)
        cpu, wall = engine.until_verdict(run_)
        out |= {k: v[k] for k in ("verdict", "method", "nodes", "wires")}
        if v["verdict"] == "unequal":
            raise Wrong(f"refuted by {v['method']}: {v['reason']}"[:400])
        if v["verdict"] != "equal":
            raise Inconclusive(f"{v['method']}: {v['reason']}"[:400])
        return out | {"cpu": cpu, "wall": wall}
    with tempfile.TemporaryDirectory(prefix="bench-breadth-") as tmp:
        out |= lean.export_and_check(script, str(path.parent), Path(tmp) / "breadth.lean")
    out["cpu"], out["wall"] = out.pop("script_cpu"), out.pop("script_wall")
    if out.get("verdict") == "unequal":
        return out | {"status": "wrong", "detail": "refuted"}
    return out
