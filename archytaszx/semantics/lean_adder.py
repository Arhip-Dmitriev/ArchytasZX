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

"""Lean output for the Fourier constant adder and the Draper adder families. export_family
instantiates both sides of a theorem in its one multiplicity at ``0 … CHECKED`` and compares each
flat export (networks, dimension and phase tables, listed labels, scalars) with the networks
``engL k`` and ``engR k`` over ``engE d k`` of ArchytasZX/AdderFamFcaEng.lean (``FcaEng``) or
ArchytasZX/AdderFamDraperEng.lean (``DraperEng``); on a match it writes those exports as
literals, ``decide`` checks of ``engL k`` and ``engR k`` against them, and the theorem by
``FcaEng.engFam`` or ``DraperEng.engFamS``, then both boxed sides as ``FamDiagram`` literals
(fam_literal) checked against ``FamFca.D0`` / ``D1`` or ``FamDraper.D0`` / ``D1`` by ``rfl`` and
the diagram-level theorem by ``FamFca.famFca`` or ``FamDraper.famDraper``, and None otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import sympy  # type: ignore[import-untyped]

from archytaszx.diagram.graph import Diagram

CHECKED = 5
"""The largest multiplicity at which export_family compares the exports."""


def _tri(j: int) -> int:
    return j * (j - 1) // 2


def _off(n: int, m: int) -> int:
    return sum(n - 1 - k for k in range(m))


def _node(kind: str, outs: list[int], ins: list[int], dim: int = 0, ph: int = 0) -> str:
    return f"⟨.{kind}, [{', '.join(map(str, outs))}], [{', '.join(map(str, ins))}], {dim}, {ph}⟩"


def _net(nodes: list[str], summed: range, bnd: range) -> str:
    return (
        f"⟨([{', '.join(nodes)}] : List Node), [{', '.join(map(str, summed))}], "
        f"[{', '.join(map(str, bnd))}]⟩"
    )


def eng_l(n: int) -> str:
    """``FcaEng.engL n`` as the exporter prints a network."""
    p = _tri(n)

    def cb(j: int) -> int:
        return 2 * n + 8 * j

    def a0(j: int, m: int) -> int:
        return 10 * n + _tri(j) + m

    def a2(m: int, j: int) -> int:
        return 10 * n + p + _off(n, m) + j - m - 1

    def a5(m: int, j: int) -> int:
        return 10 * n + 2 * p + _off(n, m) + j - m - 1

    def a9(j: int, m: int) -> int:
        return 10 * n + 3 * p + _tri(j) + m

    nodes: list[str] = []
    for j in range(n):
        b = cb(j)
        nodes += [
            _node("F", [b + 1], [b]),
            _node("Z", [b + 3], [b + 2], 0, j + 1),
            _node("F", [b + 5], [b + 4]),
            _node("F", [b + 6], [b + 5]),
            _node("F", [b + 7], [b + 6]),
        ]
    later = [list(range(m + 1, n)) for m in range(n)]
    nodes += [_node("Z", [cb(j)], [n + j] + [a0(j, m) for m in range(j)]) for j in range(n)]
    nodes += [_node("Z", [cb(j) + 2] + [a2(j, t) for t in later[j]], [cb(j) + 1]) for j in range(n)]
    nodes += [_node("Z", [cb(j) + 4] + [a5(j, t) for t in later[j]], [cb(j) + 3]) for j in range(n)]
    nodes += [_node("Z", [j], [cb(j) + 7] + [a9(j, m) for m in range(j)]) for j in range(n)]
    nodes += [
        _node("ZP 0 0 (1)", [a0(j, m)], [a2(m, j)], j - m) for m in range(n) for j in later[m]
    ]
    nodes += [
        _node("ZP 0 0 (-1)", [a9(j, m)], [a5(m, j)], j - m) for m in range(n) for j in later[m]
    ]
    return _net(nodes, range(2 * n, 10 * n + 4 * p), range(2 * n))


def eng_r(n: int) -> str:
    """``FcaEng.engR n`` as the exporter prints a network."""
    if n == 0:
        nodes = [_node("Z", [0], []), _node("X", [1], [0]), _node("Z", [], [1])]
        return _net(nodes, range(2), range(0))
    b = 2 * n * n + 8 * n

    def d_i(t: int) -> int:
        return n if t == 0 else t - 1

    def bout(t: int) -> int:
        return b + 1 if t + 1 == n else b + 4 + t

    def bin_(t: int) -> int:
        return b if t == 0 else b + 3 + t

    def sv(t: int) -> int:
        return b + 3 if t == 0 else (b + 2 if t == n else b + n + 2 + t)

    h, x, e = (
        _node("Z", [b], [], n),
        _node("X", [b + 2], [b + 1], n - 1, n + 1),
        _node("Z", [], [b + 3], n),
    )
    bs = [_node("B", [bout(t)], [bin_(t), n + t], d_i(t)) for t in range(n)]
    ss = [_node("S", [sv(t), t], [sv(t + 1)], d_i(t)) for t in range(n)]
    nodes = [h, *bs, x, *ss, e] if n == 1 else [h, x, e, *bs, *ss]
    return _net(nodes, range(b, b + 2 * n + 2), range(2 * n))


def eng_labs(n: int) -> list[tuple[int, int]]:
    """The labels of ``FcaEng.engLabs n`` with a dimension index other than ``0``."""
    if n == 0:
        return []
    b = 2 * n * n + 8 * n
    labs = [(b, n), (b + 1, n - 1), (b + 2, n - 1), (b + 3, n)]
    labs += [(b + 4 + t, t) for t in range(n - 1)] + [(b + n + 3 + t, t) for t in range(n - 1)]
    return sorted(p for p in labs if p[1])


def eng_dims(n: int, d: str) -> list[str]:
    """The dimension table of ``FcaEng.engE d n`` and ``DraperEng.engE d n`` as the exporter
    prints it."""
    return [d, *(f"({d} ^ {i + 1})" for i in range(1, n)), "1"]


def eng_phases(n: int, d: str) -> list[str]:
    """The phase table of ``FcaEng.engE d n`` as the exporter prints it."""

    def turn(c: int, e: int) -> str:
        return f"fun k => (0) + (if k = 0 then 0 else (({c} : ℝ) * (k : ℝ) * (({d} : ℝ) ^ {e})⁻¹))"

    if n == 0:
        return ["fun _ => 0"]
    return ["fun _ => 0", *(turn(2, n - p + 1) for p in range(1, n + 1)), turn(-2, n)]


def drp_l(n: int) -> str:
    """``DraperEng.engL n`` as the exporter prints a network."""
    p = _tri(n)

    def cb(j: int) -> int:
        return 4 * n + 11 * j

    def i2(m: int, j: int) -> int:
        return _off(n, m) + j - m - 1

    def a0(j: int, m: int) -> int:
        return 15 * n + _tri(j) + m

    def a2(m: int, j: int) -> int:
        return 15 * n + p + i2(m, j)

    def a7(j: int, m: int) -> int:
        return 15 * n + 2 * p + _tri(j) + m

    def g8o(m: int) -> int:
        return 15 * n + 3 * p + _off(n, m) + m

    def a8(m: int, j: int) -> int:
        return 15 * n + 3 * p + i2(m, j) + m + 1

    def a10(m: int, j: int) -> int:
        return 15 * n + 4 * p + n + i2(m, j)

    def a14(j: int, m: int) -> int:
        return 15 * n + 5 * p + n + _tri(j) + m

    nodes: list[str] = []
    for j in range(n):
        b = cb(j)
        nodes += [
            _node("F", [b + 1], [b]),
            _node("Z", [b + 2, b + 3], [2 * n + 2 * j]),
            _node("Z", [b + 6], [b + 4, b + 5]),
            _node("F", [b + 5], [b + 3]),
            _node("F", [b + 8], [b + 7]),
            _node("F", [b + 9], [b + 8]),
            _node("F", [b + 10], [b + 9]),
        ]
    later = [list(range(m + 1, n)) for m in range(n)]
    nodes += [
        _node("Z", [cb(j)], [2 * n + 2 * j + 1] + [a0(j, m) for m in range(j)]) for j in range(n)
    ]
    nodes += [_node("Z", [cb(j) + 4] + [a2(j, t) for t in later[j]], [cb(j) + 1]) for j in range(n)]
    nodes += [_node("Z", [2 * j], [cb(j) + 2] + [a7(j, m) for m in range(j)]) for j in range(n)]
    nodes += [_node("Z", [g8o(j)] + [a8(j, t) for t in later[j]], [cb(j) + 6]) for j in range(n)]
    nodes += [_node("Z", [cb(j) + 7] + [a10(j, t) for t in later[j]], [g8o(j)]) for j in range(n)]
    nodes += [
        _node("Z", [2 * j + 1], [cb(j) + 10] + [a14(j, m) for m in range(j)]) for j in range(n)
    ]
    pairs = [(m, j) for m in range(n) for j in later[m]]
    nodes += [_node("ZP 0 0 (1)", [a0(j, m)], [a2(m, j)], j - m) for m, j in pairs]
    nodes += [_node("ZP 0 0 (1)", [a7(j, m)], [a8(m, j)], j - m) for m, j in pairs]
    nodes += [_node("ZP 0 0 (-1)", [a14(j, m)], [a10(m, j)], j - m) for m, j in pairs]
    return _net(nodes, range(4 * n, 16 * n + 6 * p), range(4 * n))


_POFS = (0, 3, 4, 7, 1, 2, 8, 5, 6)
"""The spec's label offsets of the roles ``0 … 8`` at ``n = 1``."""


def drp_r(n: int) -> str:
    """``DraperEng.engR n`` as the exporter prints a network."""
    if n == 0:
        nodes = [
            _node("Z", [0], []),
            _node("Z", [], [1]),
            _node("Z", [2], []),
            _node("Z", [], [3]),
            _node("Z", [1, 4], [0]),
            _node("X", [3], [2, 4]),
        ]
        return _net(nodes, range(5), range(0))
    b = 16 * n + 6 * _tri(n)

    def rl(k: int) -> int:
        return b + (_POFS[k] if n == 1 else k)

    def d_i(t: int) -> int:
        return n if t == 0 else t - 1

    def bv(c: int, t: int) -> int:
        return rl(4 + 3 * c) if t + 1 == n else rl(9 + 2 * t + c)

    def bw(c: int, t: int) -> int:
        return rl(2 * c) if t == 0 else bv(c, t - 1)

    def sv(c: int, t: int) -> int:
        if t == 0:
            return rl(2 * c + 1)
        return rl(5 + 3 * c) if t == n else rl(9 + 2 * (n - 1) + 2 * (t - 1) + c)

    def bn(c: int, t: int) -> str:
        return _node("B", [bv(c, t)], [bw(c, t), 2 * n + 2 * t + c], d_i(t))

    def sn(c: int, t: int) -> str:
        return _node("S", [sv(c, t), 2 * t + c], [sv(c, t + 1)], d_i(t))

    h = [_node("Z", [rl(2 * c)], [], n) for c in range(2)]
    e = [_node("Z", [], [rl(2 * c + 1)], n) for c in range(2)]
    z = _node("Z", [rl(5), rl(6)], [rl(4)], n - 1)
    x = _node("X", [rl(8)], [rl(7), rl(6)], n - 1)
    if n == 1:
        nodes = [h[0], bn(0, 0), sn(0, 0), e[0], h[1], bn(1, 0), sn(1, 0), e[1], z, x]
    else:
        nodes = [h[0], e[0], h[1], e[1], z, x]
        nodes += [bn(c, t) for t in range(n) for c in range(2)]
        nodes += [sn(c, t) for t in range(n) for c in range(2)]
    return _net(nodes, range(b, b + 4 * n + 5), range(4 * n))


def drp_labs(n: int) -> list[tuple[int, int]]:
    """The labels of ``DraperEng.engLabs n`` with a dimension index other than ``0``."""
    b = 16 * n + 6 * _tri(n)
    role = {o: k for k, o in enumerate(_POFS)} if n == 1 else {}

    def edim(k: int) -> int:
        if k < 4:
            return 0
        if k < 9:
            return n
        if k < 9 + 2 * (n - 1):
            return (k - 9) // 2 + 1
        return (k - 9 - 2 * (n - 1)) // 2 + 1

    labs = []
    for o in range(4 * n + 5):
        t = edim(role.get(o, o))
        labs.append((b + o, n if t == 0 else t - 1))
    return sorted(p for p in labs if p[1])


def drp_phases(n: int, d: str) -> list[str]:
    """The phase table of ``DraperEng.engE d n`` as the exporter prints it."""
    return ["fun _ => 0"]


def _fam_ix(e: object, base: str) -> tuple[int, str | None]:
    """The static index and the index function, if any, of the dimension ``e`` as a power of
    ``base``."""
    b = sympy.Symbol(base, positive=True) if not base.isdigit() else sympy.Integer(int(base))
    ex = (
        sympy.sympify(str(e)).subs(sympy.Symbol(base), b)
        if not base.isdigit()
        else sympy.sympify(str(e))
    )
    if ex == 1:
        a = sympy.Integer(0)
    elif ex == b:
        a = sympy.Integer(1)
    else:
        a = sympy.simplify(sympy.expand_log(sympy.log(ex, b), force=True))
    if a.is_Integer:
        return (0, "fun n _ => n") if a == 0 else (int(a) - 1, None)
    s = str(a).replace("**", " ^ ")
    return 0, f"fun n j => if {s} = 0 then n else {s} - 1"


def _fam_port(r: object) -> str:
    from archytaszx.diagram.graph import Direction

    out = "true" if r.direction is Direction.OUTPUT else "false"  # type: ignore[attr-defined]
    return f"⟨{int(r.node_id)}, {out}, {r.index}⟩"  # type: ignore[attr-defined]


def fam_literal(dg: Diagram, base: str) -> str:
    """The boxed diagram ``dg`` as a ``FamDiagram`` term: node boxes, chain links, pair boxes and
    the phase, dimension and leg indices, dimensions read as powers of ``base``."""
    pair_nodes = {
        int(x): b for b in dg.bang_boxes.values() if b.pair_of is not None for x in b.node_scope
    }
    nodes: list[str] = []
    phs: list[str] = []
    dms: list[str] = []
    legs: list[str] = []
    pairs: list[str] = []
    for x in sorted(dg.nodes):
        nd = dg.nodes[x]
        g = nd.generator_type.name
        outs: list[int] = []
        ins: list[int] = []
        for out, ps in ((True, nd.outputs), (False, nd.inputs)):
            for i, p in enumerate(ps):
                s, f = _fam_ix(p.dim, base)
                (outs if out else ins).append(s)
                if f:
                    legs.append(f"(⟨{int(x)}, {'true' if out else 'false'}, {i}⟩, {f})")
        dim, kind = 0, g
        if g in ("B", "S"):
            dim, f = _fam_ix(nd.inputs[0].dim if g == "B" else nd.outputs[0].dim, base)
            if f:
                dms.append(f"({int(x)}, {f})")
        elif g == "Zp":
            pair = nd.phase.pair  # type: ignore[union-attr]
            sgn = "-1" if str(pair.to_sympy_turns()).startswith("-") else "1"
            kind = f"ZP 0 0 ({sgn})"
            dms.append(f"({int(x)}, fun _ c => c)")
        else:
            dim, f = _fam_ix(nd.outputs[0].dim if nd.outputs else nd.inputs[0].dim, base)
            if f:
                dms.append(f"({int(x)}, {f})")
            rule = getattr(nd.phase, "rule", None) if nd.phase is not None else None
            if rule is not None:
                t = str(rule.to_sympy_turns())
                phs.append(
                    f"({int(x)}, " + ("fun _ j => j + 1" if "j" in t else "fun n _ => n + 1") + ")"
                )
        lit = (
            f"⟨{int(x)}, .{kind}, {dim}, 0, [{', '.join(map(str, outs))}], "
            f"[{', '.join(map(str, ins))}]⟩"
        )
        if int(x) in pair_nodes:
            b = pair_nodes[int(x)]
            left = _fam_port(next(iter(b.left_ports)))
            right = _fam_port(next(iter(b.right_ports)))
            pairs.append(f"⟨{b.id}, {b.pair_of}, {left}, {right}, {lit}⟩")
        else:
            nodes.append(lit)
    partner = {}
    for w in dg.wires:
        partner[w.a] = w.b
        partner[w.b] = w.a
    links: list[str] = []
    boxes: list[str] = []
    drop: set[frozenset[object]] = set()
    for b in sorted(dg.bang_boxes.values(), key=lambda b: b.id):
        if b.pair_of is not None:
            continue
        boxes.append(
            f"⟨{b.id}, 0, none, [{', '.join(str(int(x)) for x in sorted(b.node_scope))}], []⟩"
        )
        for l1, l2 in b.links:
            e1, e2 = partner[l1], partner[l2]
            drop |= {frozenset((l1, e1)), frozenset((l2, e2))}
            links.append(f"⟨{_fam_port(l1)}, {_fam_port(l2)}, {_fam_port(e1)}, {_fam_port(e2)}⟩")
    wires = [
        f"({_fam_port(w.a)}, {_fam_port(w.b)})"
        for w in sorted(dg.wires, key=lambda w: w.sort_key())
        if int(w.a.node_id) not in pair_nodes
        and int(w.b.node_id) not in pair_nodes
        and frozenset((w.a, w.b)) not in drop
    ]
    outs_l = ", ".join(map(_fam_port, dg.boundary_outputs))
    ins_l = ", ".join(map(_fam_port, dg.boundary_inputs))
    return (
        f"{{ base := {{ nodes := [{', '.join(nodes)}], wires := [{', '.join(wires)}], "
        f"outs := [{outs_l}], ins := [{ins_l}], boxes := [{', '.join(boxes)}] }}, "
        f"links := [{', '.join(links)}], pairs := [{', '.join(pairs)}], ph := [{', '.join(phs)}], "
        f"dim := [{', '.join(dms)}], legs := [{', '.join(legs)}] }}"
    )


@dataclass(frozen=True)
class Family:
    """An adder family: its Lean namespace, export tag, mirrors of ``engL``, ``engR``,
    ``engLabs`` and the phase table, whether its scalars are ``1`` (else equal), and the
    namespace and theorem of its diagram-level family."""

    ns: str
    tag: str
    left: Callable[[int], str]
    right: Callable[[int], str]
    labs: Callable[[int], list[tuple[int, int]]]
    phases: Callable[[int, str], list[str]]
    unit_scalar: bool
    fam_ns: str
    fam_thm: str


FAMILIES = (
    Family("FcaEng", "Fca", eng_l, eng_r, eng_labs, eng_phases, True, "FamFca", "famFca"),
    Family(
        "DraperEng", "Draper", drp_l, drp_r, drp_labs, drp_phases, False, "FamDraper", "famDraper"
    ),
)
"""The families export_family tries, in order."""


def _matches(lhs: Diagram, rhs: Diagram, symbol: str, fam: Family) -> str | None:
    """The dimension symbol when both sides' exports at ``0 … CHECKED`` are the family's
    networks, None otherwise."""
    from archytaszx.diagram.bangbox import instantiate_symbol
    from archytaszx.semantics import lean_net

    d: str | None = None
    for k in (1, *range(2, CHECKED + 1), 0):
        left, right = instantiate_symbol(lhs, symbol, k), instantiate_symbol(rhs, symbol, k)
        sl, sr = left.scalar.to_sympy(), right.scalar.to_sympy()
        if fam.unit_scalar and (sl != 1 or sr != 1):
            return None
        if not fam.unit_scalar and sympy.simplify(sl - sr) != 0:
            return None
        ex = lean_net._Export(lambda _: "")
        a, b = ex.labelled(left), ex.labelled(right)
        if a.net.lean() != fam.left(k) or b.net.lean() != fam.right(k):
            return None
        if k > 0:
            d = d or ex.dims[0]
            if ex.dims != eng_dims(k, d) or ex.phases != fam.phases(k, d):
                return None
        elif ex.dims != ["1"] or ex.phases != fam.phases(0, ""):
            return None
        if sorted((x, i) for x, i in ex.labs.items() if i) != fam.labs(k):
            return None
    return d


def export_family(
    lhs: Diagram,
    rhs: Diagram,
    symbols: tuple[str, ...],
    name: str = "main",
) -> str | None:
    """Lean source proving ``lhs = rhs`` at every value of the one multiplicity in ``symbols``
    by the first family in FAMILIES whose ``engL`` and ``engR`` match the exports at
    ``0 … CHECKED``; None unless one matches."""
    from archytaszx.diagram.bangbox import instantiate_symbol
    from archytaszx.semantics import lean_net

    if len(symbols) != 1:
        return None
    found: tuple[Family, str] | None = None
    for fam in FAMILIES:
        try:
            d = _matches(lhs, rhs, symbols[0], fam)
        except (lean_net.NetUnsupported, ValueError, KeyError, TypeError, AttributeError):
            d = None
        if d is not None:
            found = (fam, d)
            break
    if found is None:
        return None
    fam, d = found
    ns = fam.ns
    hd = f"(by norm_num : 0 < {d})" if d.isdigit() else "hd"
    lits: list[str] = []
    for k in range(CHECKED + 1):
        ex = lean_net._Export(lambda _: "")
        left = ex.labelled(instantiate_symbol(lhs, symbols[0], k)).net.lean()
        right = ex.labelled(instantiate_symbol(rhs, symbols[0], k)).net.lean()
        lits += [
            f"/-- The left side's export at multiplicity {k}. -/",
            f"def litL{k} : Net := {left}",
            "",
            f"/-- The right side's export at multiplicity {k}. -/",
            f"def litR{k} : Net := {right}",
            "",
            f"theorem engL_lit{k} : {ns}.engL {k} = litL{k} := by decide +kernel",
            "",
            f"theorem engR_lit{k} : {ns}.engR {k} = litR{k} := by decide +kernel",
            "",
        ]
    if fam.unit_scalar:
        thm, binder, eq = "engFam", "", f"Equal ({ns}.engE {d} n) 1 ({ns}.engL n) 1 ({ns}.engR n)"
        note: list[str] = []
    else:
        thm, binder = "engFamS", " (s : ℂ)"
        eq = f"Equal ({ns}.engE {d} n) s ({ns}.engL n) s ({ns}.engR n)"
        note = [
            "  Both sides carry the same scalar at every n; the theorem holds for every",
            "  common scalar s.",
        ]
    if d.isdigit():
        stmt = f"∀ (n : ℕ){binder}, {eq}"
        proof = f"{ns}.{thm} {d} {hd}"
    else:
        stmt = f"∀ ({d} : ℕ), 0 < {d} → ∀ (n : ℕ){binder}, {eq}"
        proof = f"fun {d} hd => {ns}.{thm} {d} hd"
    deq = f"DEqual ({ns}.engE {d} n) 1 (famL.famExpand n) 1 (famR.famExpand n)"
    fproof = f"{fam.fam_ns}.{fam.fam_thm} {d} {hd} n"
    if d.isdigit():
        fstmt = f"∀ (n : ℕ), {deq}"
        fbody = [f"  fun n => by rw [famL_eq, famR_eq]; exact {fproof}"]
    else:
        fstmt = f"∀ ({d} : ℕ), 0 < {d} → ∀ (n : ℕ), {deq}"
        fbody = [f"  fun {d} hd n => by rw [famL_eq, famR_eq]; exact {fproof}"]
    fam_lines = [
        "/-- The left side as a boxed diagram. -/",
        f"def famL : FamDiagram := {fam_literal(lhs, d)}",
        "",
        "/-- The right side as a boxed diagram. -/",
        f"def famR : FamDiagram := {fam_literal(rhs, d)}",
        "",
        f"theorem famL_eq : famL = {fam.fam_ns}.D0 := rfl",
        "",
        f"theorem famR_eq : famR = {fam.fam_ns}.D1 := rfl",
        "",
        f"theorem {name}_diagram : {fstmt} :=",
        *fbody,
        "",
    ]
    lines = [
        "/-",
        "  ArchytasZX proof export. Both sides at multiplicity n are the networks",
        f"  {ns}.engL n and {ns}.engR n over {ns}.engE d n (their exports at",
        f"  n = 0 … {CHECKED} below, checked by decide); {ns}.{thm} proves them equal.",
        *note,
        "  Both boxed sides as FamDiagram literals (famL, famR), checked equal to",
        f"  {fam.fam_ns}.D0 and {fam.fam_ns}.D1 by rfl; {fam.fam_ns}.{fam.fam_thm} proves their",
        "  expansions at every n the same tensor.",
        "-/",
        "",
        "import ArchytasZX",
        "",
        f"namespace ArchytasZX.Export.{fam.tag}",
        "",
        "open ArchytasZX",
        "",
        *lits,
        f"theorem {name} : {stmt} :=",
        f"  {proof}",
        "",
        *fam_lines,
        f"end ArchytasZX.Export.{fam.tag}",
        "",
    ]
    return "\n".join(lines)
