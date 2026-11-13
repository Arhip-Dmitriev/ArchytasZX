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
    """The record of an instance the tool cannot run, with its outcomes.UNSUPPORTED code."""
    from benchmarks.outcomes import unsupported_code

    return {
        "outcome": "unsupported",
        "detail": reason,
        "unsupported_code": unsupported_code(reason),
    }


def expand(c: ir.Circuit, n: int) -> ir.Circuit:
    """c with every register replaced by its copies, gates on registers repeated per copy; the
    output order of ``c`` realised by trailing Swaps onto the input order."""
    import sympy as sp

    def count(q: str) -> int:
        return int(sp.sympify(c.count(q)).subs(sp.Symbol("n"), n))

    def inner(q: str) -> int:
        return int(sp.sympify(c.inner(q)).subs(sp.Symbol("n"), n))

    rows: dict[str, list[list[str]]] = {
        q: [[f"{q}_{m}_{j}" for j in range(inner(q))] for m in range(count(q))]
        for q in c.wires
        if c.inner(q) is not None
    }
    copies: dict[str, list[str]] = {
        q: [q]
        if c.count(q) is None
        else (
            [x for row in rows[q] for x in row]
            if q in rows
            else [f"{q}_{i}" for i in range(count(q))]
        )
        for q in c.wires
    }

    def neighbours(g: ir.Neighbours, own: list[str]) -> list[ir.Gate]:
        return [ir.CZ(own[j], own[j + 1], g.s) for j in range(len(own) - 1)]

    wires = _layout(c, c.wires, copies)
    outs = _layout(c, c.output_order(), copies)
    gates: list[ir.Gate] = []
    leaves: dict[str, list[str]] = {}
    alias: dict[str, str] = {}
    tree_dims: dict[str, Any] = {}
    for g in c.gates:
        if isinstance(g, ir.Chain):
            span = range(int(sp.sympify(g.count).subs(sp.Symbol("n"), n)))
            row_carry = {carry: reg for carry, reg in g.carries if c.count(carry) is not None}
            for m in reversed(span) if g.reverse else span:
                values = {g.index: m} if g.index is not None else {}
                ended: set[str] = set()

                def row(q: str, m: int = m) -> list[str]:
                    return rows[q][m] if q in rows else copies[q]

                for h in g.body:
                    qs = ir.gate_wires(h)
                    at = ir.at_index(h, values)
                    if isinstance(h, ir.Neighbours):
                        gates += neighbours(h, row(h.over))
                        continue
                    wide = [q for q in qs if q in rows or q in row_carry]
                    plain = {
                        q: copies[q][m] for q in qs if c.count(q) is not None and q not in wide
                    }
                    for j in range(len(row(wide[0]))) if wide else [None]:
                        sub = dict(plain)
                        if j is not None:
                            sub.update({q: row(q)[j] for q in wide})
                        sub.update({q: alias[q] for q in qs if q in alias})
                        gates.append(_rename(at, sub))
                    if isinstance(h, (ir.Post, ir.Discard)):
                        ended.add(h.q)
                for carry, reg in g.carries:
                    if carry in row_carry:
                        gates += [
                            ir.Swap(a, b) for a, b in zip(copies[carry], row(reg), strict=True)
                        ]
                    elif carry in ended:
                        alias[carry] = copies[reg][m]
                    else:
                        gates.append(ir.Swap(alias.get(carry, carry), copies[reg][m]))
            continue
        if isinstance(g, ir.Tree):
            depth = int(sp.sympify(g.count).subs(sp.Symbol("n"), n))
            width = len(g.children)
            feed = {(1, 0): g.parent}
            local = {w for h in g.body for w in ir.gate_wires(h)} - {g.parent}
            for level in range(1, depth + 1):
                for pos in range(width ** (level - 1)):
                    sub = {g.parent: feed[(level, pos)]}
                    sub.update({w: f"{g.parent}_{w}_{level}_{pos}" for w in local})
                    for w in local:
                        tree_dims[sub[w]] = c.dim(g.parent)
                    values = {
                        k: v for k, v in ((g.level, level), (g.position, pos)) if k is not None
                    }
                    for h in g.body:
                        gates.append(_rename(ir.at_index(h, values), sub))
                    for j, w in enumerate(g.children):
                        feed[(level + 1, width * pos + j)] = sub[w]
            leaves[g.parent] = [feed[(depth + 1, pos)] for pos in range(width**depth)]
            continue
        if isinstance(g, ir.Pairs):
            own = copies[g.over]
            for j in range(len(own)):
                for k in range(j + 1, len(own)):
                    values = {g.index: j, g.pair_index: k}
                    for h in g.body:
                        sub = {"left": own[j], "right": own[k]}
                        gates.append(_rename(ir.at_index(h, values), sub))
            continue
        if isinstance(g, ir.Neighbours):
            gates += neighbours(g, copies[g.over])
            continue
        if isinstance(g, ir.LowerMCX):
            own = copies[g.over]
            order = range(len(own)) if g.reverse else reversed(range(len(own)))
            for k in order:
                cs = (*g.controls, *own[:k])
                gates.append(ir.MCX(cs, own[k], (*g.values, *(1,) * k)))
            continue
        if alias and not isinstance(g, (ir.Chain, ir.Pairs, ir.Tree)):
            g = _rename(g, {q: alias[q] for q in ir.gate_wires(g) if q in alias})
        qs = ir.gate_wires(g)
        reg = [q for q in qs if c.count(q) is not None]
        if not reg:
            gates.append(g)
            continue
        for i in range(len(copies[reg[0]])):
            sub = {q: (copies[q][i] if c.count(q) is not None else q) for q in qs}
            gates.append(_rename(g, sub))
    dims = {e: c.dim(q) for q in c.wires for e in copies[q]}
    dims.update(tree_dims)
    moved = set(alias.values())

    def placed(order: list[str]) -> list[str]:
        order = [e for q in order for e in leaves.get(q, [q])]
        return [
            x for q in order if q not in moved for x in (q, *([alias[q]] if q in alias else []))
        ]

    wires, outs = placed(wires), placed(outs)
    if outs != wires:
        closed = {g.q for g in gates if isinstance(g, (ir.Post, ir.Discard))}
        gates += _permute(
            [w for w in wires if w not in closed], [w for w in outs if w not in closed], dims
        )
    return ir.Circuit(tuple(wires), tuple(gates), {}, dims, c.scalar)


def _layout(c: ir.Circuit, order: tuple[str, ...], copies: dict[str, list[str]]) -> list[str]:
    """``order`` with each run of registers of one count expanded copy-major."""
    out: list[str] = []
    group: list[str] = []
    gkey: str | None = None
    for q in (*order, None):
        key = str(c.registers[q]) if q is not None and c.count(q) is not None else None
        if group and key != gkey:
            out += [copies[r][i] for i in range(len(copies[group[0]])) for r in group]
            group = []
        gkey = key
        if q is not None and key is None:
            out.append(q)
        elif q is not None:
            group.append(q)
    return out


def _permute(slots: list[str], want: list[str], dims: dict[str, Any]) -> list[ir.Gate]:
    """Swaps after which slot ``slots[i]`` holds the value of wire ``want[i]``."""
    at = {w: w for w in slots}
    holder = {w: w for w in slots}
    swaps: list[ir.Gate] = []
    for slot, value in zip(slots, want, strict=True):
        src = holder[value]
        if src == slot:
            continue
        if str(dims[src]) != str(dims[slot]):
            raise ValueError(f"output order moves {value} onto a wire of another dimension")
        swaps.append(ir.Swap(slot, src))
        moved = at[slot]
        at[slot], at[src] = value, moved
        holder[value], holder[moved] = slot, src
    return swaps


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
    """The instance's circuit at (n, d) with registers expanded and amounts evaluated."""
    c = subject.circuit_at(inst, n, d, expand=True)
    return expand(c, n if n is not None else 0)


def matches_unrolled(inst: Any, unrolled: ir.Circuit, n: int, d: int) -> bool:
    """Whether the instance's boxed translation instantiated at (n, d) and the translation of
    ``unrolled`` are isomorphic once spider fusion and identity removal reduce both."""
    from archytaszx.repl.parser import parse_dsl
    from archytaszx.rewrite.normal_form import fused_isomorphic
    from archytaszx.semantics.check import instantiate
    from benchmarks import translate

    def body(c: ir.Circuit, values: dict[str, int]) -> str:
        ctx = translate.Context(values=values, literal=frozenset())
        return translate.translate(c, ctx).body

    boxed = parse_dsl(
        f"dim d = {d}\ncount n = {n}\n" + body(subject.circuit_at(inst, n, d), {"n": n, "d": d})
    )
    flat = parse_dsl(f"dim d = {d}\n" + body(unrolled, {"d": d}))
    return fused_isomorphic(instantiate(boxed, {"n": n}), instantiate(flat, {}))
