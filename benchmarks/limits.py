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

"""Benchmark: the limits of docs/OVERVIEW.txt section 6, measured.

Kinds (``param`` is the swept parameter):

- ``contraction``: ``contract_symbolic`` on a chain of ``param`` Z spiders with symbolic
  phases a1, a2, ... joined by F boxes (1 .. 12).
- ``nesting``: nested bang boxes of depth ``param`` (1 .. 5), the Fourier superposition with
  one register per level, by ``decide`` (normal form) and by ``prove`` (search, induction).
- ``symbols``: ``param`` independent (count, dimension) pairs (1 .. 6) in one diagram.
- ``deferred``: matches that need d1 d2 = 6 or d^p = d^q; ``assumptions`` counts the
  conditions a verdict rests on.
- ``prime``: every family and breadth workload with d free, bound to the prime 3, and bound to
  the composite 4 (``param`` 0, 3, 4), counting unsettled verdicts.

Fields: ``verdict``, ``method``, ``time``, ``peak_mb``, ``assumptions``.
"""

from __future__ import annotations

import re
import resource
import time
from typing import Any

from benchmarks import engine, workloads
from benchmarks.harness import Inconclusive, Wrong

SIZE = "param"
RECORD_ENV = True
DEFERRED = {
    "fusion d1 d2 = 6": """
@rhs =
    z = Z(0 -> 1, literal 6)
load
    a = Z(0 -> 1, d1*d2) -> b = Z(1 -> 1, literal 6)
decide against @rhs
""",
    "fusion d^p = d^q": """
@rhs =
    z = Z(0 -> 1, d^q)
load
    a = Z(0 -> 1, d^p) -> b = Z(1 -> 1, d^q)
decide against @rhs
""",
    "copy d1 d2 = 6": """
@rhs =
    a = X(0 -> 1, literal 6)
    b = X(0 -> 1, literal 6)
    scalar 6^(-1/2)
load
    X(0 -> 1, d1*d2) -> z = Z(1 -> 2, literal 6)
decide against @rhs
""",
    "Fourier d^p = d^q": """
@rhs =
    z = Z(1 -> 1, d^q)
load
    F(d^p) -> F(d^q) -> F(d^q) -> F(d^q)
decide against @rhs
""",
}


def _nested(depth: int) -> str:
    names = [f"n{i}" for i in range(depth)]
    total = " + ".join("*".join(names[: i + 1]) for i in range(depth))

    def side(state: bool) -> str:
        lines = []
        for i in range(depth):
            lines.append(f"    z{i} = Z(0 -> 1)" if state else f"    x{i} = X(0 -> 1) -> f{i} = F")
        for i in range(depth):
            parent = f" in B{i - 1}" if i else ""
            nodes = ", ".join((f"z{j}" if state else f"x{j}, f{j}") for j in range(i, depth))
            lines.append(f"    repeat {names[i]} as B{i}{parent}: {nodes}")
        return "\n".join(lines)

    head = f"    count {', '.join(names)}\n"
    return (
        f"@rhs =\n{head}{side(True)}\n    scalar d^(-({total})/2)\n\n"
        f"load\n{head}{side(False)}\n    scalar d^(-({total})/2)\n\n"
    )


def _symbols(k: int) -> str:
    rhs = "\n".join(f"    repeat n{i}: Z(0 -> 1, d{i})" for i in range(k))
    lhs = "\n".join(f"    repeat n{i}: X(0 -> 1, d{i}) -> F(d{i})" for i in range(k))
    scalar = " * ".join(f"d{i}^(-n{i}/2)" for i in range(k))
    head = f"    count {', '.join(f'n{i}' for i in range(k))}\n"
    return (
        f"@rhs =\n{head}{rhs}\n    scalar {scalar}\n\nload\n{head}{lhs}\n    scalar {scalar}\n\n"
        "decide against @rhs\n"
    )


def _chain(k: int) -> str:
    steps = " -> F -> ".join(f"z{i} = Z(1 -> 1) phase{{1: a{i}}}" for i in range(1, k + 1))
    return f"load\n    {steps}\n"


def _prime_workloads() -> dict[str, tuple[str, str]]:
    out = {
        f"family {f}": (workloads.family_script(f), str(workloads.ROOT)) for f in workloads.FAMILIES
    }
    for path in workloads.breadth_files():
        out[f"breadth {path.stem}"] = (
            workloads.units(path.read_text(encoding="utf-8"))[0],
            str(path.parent),
        )
    return out


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every kind at every parameter value; ``quick`` keeps the smallest ones."""
    rows: list[tuple[str, str, int]] = []
    rows += [("contraction", "chain", k) for k in range(1, 13 if not quick else 3)]
    for how in ("decide", "prove"):
        rows += [("nesting", how, k) for k in range(1, 6 if not quick else 3)]
    rows += [("symbols", "Fourier superposition", k) for k in range(1, 7 if not quick else 3)]
    rows += [("deferred", name, 0) for name in DEFERRED]
    for name in _prime_workloads():
        if quick and name != "family spider fusion":
            continue
        rows += [("prime", name, d) for d in (0, 3, 4)]
    return [{"suite": "limits", "kind": k, "workload": w, "param": p} for k, w, p in rows]


def _bind_d(script: str, d: int) -> str:
    if re.search(r"^\s*dim d\{?", script, re.MULTILINE):
        return re.sub(r"^(\s*)dim d\b(?: = \d+)?", rf"\1dim d = {d}", script, flags=re.MULTILINE)
    return re.sub(r"^(@\w+ =|load)\n(\s+)", rf"\1\n\2dim d = {d}\n\2", script, flags=re.MULTILINE)


def _peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def run(case: dict[str, Any]) -> dict[str, Any]:
    """One limit measurement."""
    kind, name, param = case["kind"], case["workload"], case["param"]
    if kind == "contraction":
        from archytaszx.semantics.contract_symbolic import contract_symbolic

        session = engine.run_script(_chain(param)).session
        assert session.current is not None
        c0, w0 = time.process_time(), time.perf_counter()
        contract_symbolic(session.current)
        cpu, wall = time.process_time() - c0, time.perf_counter() - w0
        return {"cpu": cpu, "wall": wall, "method": "contract_symbolic", "peak_mb": _peak()}
    if kind == "nesting":
        script = _nested(param) + (
            "decide against @rhs\n" if name == "decide" else "prove against @rhs\n"
        )
        base = str(workloads.ROOT)
    elif kind == "symbols":
        script, base = _symbols(param), str(workloads.ROOT)
    elif kind == "deferred":
        script, base = DEFERRED[name], str(workloads.ROOT)
    else:
        script, base = _prime_workloads()[name]
        if param:
            script = _bind_d(script, param)
    run_ = engine.run_script(script, base=base)
    v = engine.verdict(run_)
    cpu, wall = engine.until_verdict(run_)
    out = {
        "cpu": cpu,
        "wall": wall,
        "peak_mb": _peak(),
        **{k: v[k] for k in ("verdict", "method", "assumptions", "nodes", "wires")},
    }
    if kind == "deferred":
        return out | {"reason": str(v["reason"])[:300]}
    if v["verdict"] == "unequal":
        raise Wrong(f"refuted by {v['method']}: {v['reason']}"[:400])
    if v["verdict"] != "equal":
        raise Inconclusive(f"{v['method']}: {v['reason']}"[:400])
    return out
