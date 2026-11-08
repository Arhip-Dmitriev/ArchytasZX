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

"""DiZX adapter: Clifford circuits at odd prime d, concrete n; the circuit composed with the
inverse of the equation's own reference circuit, reduced by clifford_simp and checked for the
identity (drivers/dizx.py). States, post-selection, non-Clifford gates and scalars are
unsupported."""

from __future__ import annotations

import time
from typing import Any

import sympy as sp

from benchmarks import ir, subject
from benchmarks.tools import common, rival

ENV = "dizx"


def supports(inst: Any, point: Any) -> str | None:
    """Equations with a reference circuit and odd prime d, the environment installed."""
    why = rival.installed(ENV)
    if why:
        return why
    if inst.equation.reference is None:
        return "no gate-level reference circuit for this equation"
    return None


def clifford_ops(c: ir.Circuit, d: int) -> list[list[Any]]:
    """[name, wires, power] per gate: Z (powers), HAD (power p), CX and CZ (s), SWAP; X^a as
    HAD, Z^a, HAD^3 (DiZX's X gate drops its phase in the graph)."""
    idx = {q: i for i, q in enumerate(c.wires)}
    out: list[list[Any]] = []
    for g in c.gates:
        if isinstance(g, (ir.X, ir.Z)):
            a = int(sp.sympify(g.a).subs(sp.Symbol("d"), d)) % d
            if isinstance(g, ir.X):
                out += [["HAD", [idx[g.q]], 1], ["Z", [idx[g.q]], a], ["HAD", [idx[g.q]], 3]]
            else:
                out.append(["Z", [idx[g.q]], a])
        elif isinstance(g, ir.F):
            out.append(["HAD", [idx[g.q]], g.p % 4])
        elif isinstance(g, (ir.CX, ir.CZ)):
            out.append([type(g).__name__, [idx[g.c], idx[g.t]], g.s])
        elif isinstance(g, ir.Swap):
            out.append(["SWAP", [idx[g.a], idx[g.b]], 1])
        else:
            raise rival.Unsupported(f"gate {type(g).__name__} is not a DiZX Clifford gate")
    return out


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One clifford_simp identity check."""
    n, d = common.point(job)
    if job.get("point") is None:
        return common.unsupported("no symbolic n or d")
    why = supports(inst, None)
    if why or d < 3 or d % 2 == 0 or not subject.is_prime(d):
        return common.unsupported(why or "odd prime d only")
    c0 = time.process_time()
    try:
        a = common.concrete_circuit(inst, n, d)
        b = rival.explicit_reference(inst, n, d)
        if len(a.wires) != len(b.wires):
            raise rival.Unsupported("circuit and reference differ in width")
        ops_a, ops_b = clifford_ops(a, d), clifford_ops(b, d)
        if str(a.scalar) != "1" or str(b.scalar) != "1":
            raise rival.Unsupported("a circuit scalar")
    except rival.Unsupported as exc:
        return common.unsupported(str(exc))
    prep = time.process_time() - c0
    payload = {"d": d, "n": len(a.wires), "a": ops_a, "b": ops_b}
    res = rival.run_driver(ENV, "dizx", payload, float(job["config"]["stages"]["timeout_seconds"]))
    return rival.finish(inst, res, prep, {"dizx": "benchmarks/.venvs/src/dizx"})
