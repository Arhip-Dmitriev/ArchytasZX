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

"""Benchmark: certificate and kernel-proof size against diagram size. Workloads: every family
free, every breadth workload and every encoded lemma (proved cases settle; others are
``inconclusive``), and size.py's ArchytasZX cases rebuilt here at their sizes (its results
file is neither read nor written). Each run: the verdict, ``certificate kernel``,
``certificate lean`` (written, not checked) and ``certificate show``.

Fields: ``nodes``, ``wires``, ``derivation_length`` (rewrite steps), ``inferences``,
``certificate_bytes`` (the output of ``certificate show``), ``lean_lines``
and ``lean_bytes``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from benchmarks import engine, size, workloads
from benchmarks.harness import Inconclusive, Wrong

SIZE = "s"
RECORD_ENV = True
SIZE_TOOLS = ("azx-family", "azx-qubit")
QUICK_WORKLOADS = ("spider fusion", "[P1] Lemma 11 #1")


def _named() -> dict[str, tuple[str, str]]:
    return {
        name: (script, base)
        for group, name, script, base in workloads.unit_scripts()
        if group != "examples"
    }


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every named workload once; every size.py ArchytasZX case."""
    out: list[dict[str, Any]] = []
    for group, name, _, _ in workloads.unit_scripts():
        if group == "examples" or (quick and name not in QUICK_WORKLOADS):
            continue
        out.append(
            {"suite": "certificates", "group": group, "workload": name, "w": 0, "seed": 0, "s": 0}
        )
    for case in size.cases(quick):
        if case["tool"] not in SIZE_TOOLS or (quick and case["s"] > 2):
            continue
        out.append(
            {
                "suite": "certificates",
                "group": "size",
                "workload": case["tool"],
                "w": case["w"],
                "seed": case["seed"],
                "s": case["s"],
            }
        )
    return out


def run(case: dict[str, Any]) -> dict[str, Any]:
    """The verdict, the kernel check and the Lean export of one workload."""
    from archytaszx.repl.commands import format_outcome

    if case["group"] == "size":
        gates = size.circuit(case["w"], case["s"], case["seed"])
        script, base = (
            size.azx_script(gates, case["w"], case["workload"] == "azx-family"),
            str(workloads.ROOT),
        )
    else:
        script, base = _named()[case["workload"]]
    with tempfile.TemporaryDirectory(prefix="bench-cert-") as tmp:
        path = Path(tmp) / "certificate.lean"
        run_ = engine.run_script(
            workloads.with_lean(script, path) + "certificate show\n", base=base
        )
        v = engine.verdict(run_)
        if v["verdict"] == "unequal":
            raise Wrong(f"refuted by {v['method']}: {v['reason']}"[:400])
        if v["verdict"] != "equal":
            raise Inconclusive(f"{v['method']}: {v['reason']}"[:400])
        cpu, wall = engine.until_verdict(run_)
        out: dict[str, Any] = {
            "cpu": cpu,
            "wall": wall,
            **{k: v[k] for k in ("method", "nodes", "wires", "derivation_length")},
            **engine.kernel_fields(run_),
            **engine.lean_export_fields(run_),
        }
        cert = run_.session.certificate
        shown = run_.last("certificate", "certificate show")
        out["certificate_kind"] = type(cert).__name__ if cert is not None else None
        out["certificate_bytes"] = (
            len(format_outcome(shown.outcome).encode("utf-8"))
            if shown is not None and shown.outcome.ok
            else None
        )
        if path.exists():
            text = path.read_text(encoding="utf-8")
            out["lean_lines"] = text.count("\n") + 1
            out["lean_bytes"] = len(text.encode("utf-8"))
        else:
            out["lean_lines"] = out["lean_bytes"] = None
    return out
