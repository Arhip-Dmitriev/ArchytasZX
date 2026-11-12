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

"""Shared adapter pieces: expected verdicts, outcomes, the point of a job, and the expanded
qubit circuit rivals translate from."""

from __future__ import annotations

from typing import Any

from benchmarks import ir, subject


def expected(inst: Any) -> str:
    """equal for an S equation, unequal for an F equation."""
    return "equal" if inst.truth else "unequal"


def outcome(inst: Any, status: str, verdict: str | None) -> str:
    """ok (the expected verdict, certified), uncertified (the expected verdict, uncertified),
    wrong (the opposite verdict, certified), wrong (uncertified), unknown, or the failing
    stage's status."""
    return verdict_outcome(bool(inst.truth), status, verdict)


def verdict_outcome(truth: bool, status: str, verdict: str | None) -> str:
    """The :func:`outcome` of a verdict on an equation whose truth is ``truth``."""
    if status != "ok":
        return status
    want = "equal" if truth else "unequal"
    if verdict == want:
        return "ok"
    if verdict == f"{want}_uncertified":
        return "uncertified"
    if verdict in ("equal", "unequal"):
        return "wrong"
    if verdict in ("equal_uncertified", "unequal_uncertified"):
        return "wrong (uncertified)"
    return "unknown"


def point(job: dict[str, Any]) -> tuple[int | None, int | None]:
    """(n, d) of the job, None where symbolic."""
    p = job.get("point")
    if p is None:
        return None, None
    return p[0], p[1]


def unsupported(reason: str) -> dict[str, Any]:
    """The record of an instance the tool cannot run."""
    return {"outcome": "unsupported", "detail": reason}


def expand(c: ir.Circuit, n: int) -> ir.Circuit:
    """c with every register replaced by its copies, gates on registers repeated per copy."""
    import sympy as sp

    copies: dict[str, list[str]] = {}
    wires: list[str] = []
    group: list[str] = []
    gkey: str | None = None

    def count(q: str) -> int:
        return int(sp.sympify(c.count(q)).subs(sp.Symbol("n"), n))

    def flush() -> None:
        if group:
            k = count(group[0])
            for i in range(k):
                for q in group:
                    wires.append(f"{q}_{i}")
            for q in group:
                copies[q] = [f"{q}_{i}" for i in range(k)]
            group.clear()

    for q in c.wires:
        key = str(c.count(q)) if c.count(q) is not None else None
        if key is None:
            flush()
            gkey = None
            wires.append(q)
            copies[q] = [q]
        else:
            if gkey != key:
                flush()
                gkey = key
            group.append(q)
    flush()
    gates: list[ir.Gate] = []
    for g in c.gates:
        qs = ir.gate_wires(g)
        reg = [q for q in qs if c.count(q) is not None]
        if not reg:
            gates.append(g)
            continue
        for i in range(len(copies[reg[0]])):
            sub = {q: (copies[q][i] if c.count(q) is not None else q) for q in qs}
            gates.append(_rename(g, sub))
    dims = {e: c.dim(q) for q in c.wires for e in copies[q]}
    return ir.Circuit(tuple(wires), tuple(gates), {}, dims, c.scalar)


def _rename(g: ir.Gate, sub: dict[str, str]) -> ir.Gate:
    from dataclasses import fields, replace

    changes = {}
    for f in fields(g):
        v = getattr(g, f.name)
        if f.name in ("q", "c", "t", "a", "b", "y") and isinstance(v, str) and v in sub:
            changes[f.name] = sub[v]
        elif f.name in ("qs", "cs", "xs"):
            changes[f.name] = tuple(sub.get(x, x) for x in v)
    return replace(g, **changes)


def concrete_circuit(inst: Any, n: int | None, d: int) -> ir.Circuit:
    """The instance's circuit at (n, d) with registers expanded and amounts evaluated, the
    scalar (an F break's included) evaluated at n and d."""
    from dataclasses import replace

    import sympy as sp

    c = subject.circuit_at(inst, n, d, expand=True)
    vals = {sp.Symbol(k): v for k, v in (("n", n), ("d", d)) if v is not None}
    c = replace(c, scalar=str(sp.sympify(c.scalar).subs(vals)))
    return expand(c, n if n is not None else 0)
