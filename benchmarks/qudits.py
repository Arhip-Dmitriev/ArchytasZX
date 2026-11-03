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

"""Benchmark: odd prime dimensions against DiZX, the qudit port of PyZX (Clifford only).

Workloads: ``C C^-1`` (a seeded random circuit of 4 n gates on n qudits from size.py's
generator, phases restricted to Paulis, then its inverse), ``Fourier order 4`` and the
families ``GHZ`` and ``Fourier superposition``; d = 3, 5, 7, 11. Tools:

- ``dizx``: DiZX in its own environment (benchmarks/.venvs/dizx, run with ``python -I``),
  ``clifford_simp`` to the identity; state equations are ``unsupported`` (DiZX reduces
  circuits to the identity and has no tensor or scalar comparison).
- ``azx-bound``: ArchytasZX at ``dim d = D`` with n bound (``C C^-1`` with n explicit wires).
- ``azx-free``: ArchytasZX with n and d free, one run (``C C^-1``: two wires repeated n times).

Without the DiZX environment every dizx row is ``unsupported`` with the reason.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from benchmarks import size, workloads
from benchmarks.harness import Inconclusive, Wrong

SIZE = "n"
RECORD_ENV = True
DS = [3, 5, 7, 11]
NS = [2**k for k in range(11)]
QUICK_DS = [3]
QUICK_NS = [1, 2]
WORKLOADS = ("C C^-1", "Fourier order 4", "GHZ", "Fourier superposition")
TOOLS = ("dizx", "azx-bound", "azx-free")
DIZX_PYTHON = Path(__file__).resolve().parent / ".venvs" / "dizx" / "bin" / "python"
DRIVER = Path(__file__).resolve().parent / "dizx_driver.py"
DIZX_STATES = (
    "DiZX reduces circuits to the identity; it has no tensor or scalar comparison for a state "
    "equation"
)


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every workload and tool at every (d, n); ``azx-free`` once per workload."""
    ds, ns = (QUICK_DS, QUICK_NS) if quick else (DS, NS)
    out: list[dict[str, Any]] = []
    for w in WORKLOADS:
        out.append({"suite": "qudits", "tool": "azx-free", "workload": w, "d": "free", "n": "free"})
        out += [
            {"suite": "qudits", "tool": t, "workload": w, "d": d, "n": n}
            for t in ("dizx", "azx-bound")
            for d in ds
            for n in ns
        ]
    return out


def clifford(n: int, seed: int = 0) -> list[tuple[Any, ...]]:
    """size.py's circuit C C^-1 of 4 n gates (2 n each way) on n qudits, phases S k/d."""
    gates = size.circuit(n, 2 * n, seed)
    return [("P", g[1], g[2], 1) if g[0] == "P" else g for g in gates]


def azx_script(case: dict[str, Any]) -> str:
    """The ArchytasZX command file of the case."""
    w, d, n = case["workload"], case["d"], case["n"]
    if w == "C C^-1":
        if case["tool"] == "azx-free":
            return size.azx_script(clifford(2), 2, True)
        return size.azx_script(clifford(n), n, False).replace("dim d = 2", f"dim d = {d}")
    return workloads.family_script(w, n, d)


def dizx(case: dict[str, Any]) -> dict[str, Any]:
    """DiZX on the case's circuit."""
    if case["workload"] in ("GHZ", "Fourier superposition"):
        return {"status": "unsupported", "detail": DIZX_STATES, "cpu": 0.0, "wall": 0.0}
    if not DIZX_PYTHON.exists():
        detail = "DiZX is not installed (benchmarks/.venvs/dizx missing; docs/BENCHMARKS.txt, 4.8)"
        return {"status": "unsupported", "detail": detail, "cpu": 0.0, "wall": 0.0}
    n = case["n"]
    gates = (
        clifford(n)
        if case["workload"] == "C C^-1"
        else [("F", q, 1) for q in range(n) for _ in range(4)]
    )
    spec: dict[str, Any] = {"d": case["d"], "n": n, "gates": gates}
    done = subprocess.run(
        [str(DIZX_PYTHON), "-I", str(DRIVER), json.dumps(spec)],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"DiZX failed: {done.stderr.strip()[-300:]}")
    out: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    if not out.pop("identity"):
        raise Inconclusive("clifford_simp did not reach the identity")
    return out | {"method": "clifford_simp"}


def run(case: dict[str, Any]) -> dict[str, Any]:
    """One check of the case by its tool."""
    if case["tool"] == "dizx":
        return dizx(case)
    from benchmarks import engine

    run_ = engine.run_script(azx_script(case))
    v = engine.verdict(run_)
    cpu, wall = engine.until_verdict(run_)
    if v["verdict"] == "unequal":
        raise Wrong(f"refuted by {v['method']}: {v['reason']}"[:400])
    if v["verdict"] != "equal":
        raise Inconclusive(f"{v['method']}: {v['reason']}"[:400])
    return {"cpu": cpu, "wall": wall, **{k: v[k] for k in ("method", "nodes", "wires")}}
