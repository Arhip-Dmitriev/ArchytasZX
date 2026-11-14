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

"""A library of stated lemmas: named identities between two diagrams over one symbol.

Each lemma is proved once, with the symbol formal, by
:func:`~archytaszx.semantics.decide.decide_equal` with no oracle sample, and kept with its
kernel theorem (:func:`proved`). An instance at a value of the symbol (:func:`instance`) is a
kernel theorem by symbolic contraction of the instantiated sides; :func:`rewrite` finds an
instance's left side inside a larger diagram through cut wires and replaces it by the right
side under :func:`~archytaszx.semantics.kernel.congr`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim, DimensionError
from archytaszx.algebra.phase import INDEX, Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    X_SPIDER,
    Z_SPIDER,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef


class LemmaError(Exception):
    """A lemma is unknown, unproved, or has no instance where one was asked for."""


@dataclass(frozen=True, slots=True)
class Lemma:
    """A named identity ``build(e)[0] = build(e)[1]`` for every value ``e`` of ``symbol``."""

    name: str
    symbol: str
    statement: str
    build: Callable[[sp.Expr], tuple[Diagram, Diagram]]


def _out(node: NodeId, index: int = 0) -> PortRef:
    return PortRef(node, Direction.OUTPUT, index)


def _in(node: NodeId, index: int = 0) -> PortRef:
    return PortRef(node, Direction.INPUT, index)


def _shift(d: Diagram, dim: Dim, sign: int) -> NodeId:
    """An X(1 -> 1) over ``dim`` with phase ``-sign k / dim``: the shift by ``sign``."""
    turns = -sign * INDEX / dim.to_sympy()
    return d.add_node(X_SPIDER, [dim], [dim], phase=PhaseVector(dim, rule=Phase(turns)))


def _carry_sides(p: sp.Expr) -> tuple[Diagram, Diagram]:
    """Left: (top, low) bound by B(2, 2^p), shifted by one over 2^(p + 1), split by
    S(2, 2^p). Right: F on top; the bound register shifted by one, weight -1 off index 0,
    shifted back, scalar -1 (a sign at the all-ones index); F^3 on top; low shifted by one
    over 2^p."""
    two = Dim(2)
    low = Dim.from_sympy(2**p)
    reg = Dim.from_sympy(2 ** (p + 1))

    left = Diagram()
    b = left.add_node(DIM_BINDER, [two, low], [reg])
    x = _shift(left, reg, 1)
    s = left.add_node(DIM_SPLITTER, [reg], [two, low])
    left.add_wire(_out(b), _in(x))
    left.add_wire(_out(x), _in(s))
    left.set_boundary_inputs([_in(b, 0), _in(b, 1)])
    left.set_boundary_outputs([_out(s, 0), _out(s, 1)])

    right = Diagram()
    f = right.add_node(FOURIER_BOX, [two], [two])
    rb = right.add_node(DIM_BINDER, [two, low], [reg])
    u = _shift(right, reg, 1)
    g = right.add_node(
        Z_SPIDER, [reg], [reg], phase=PhaseVector(reg, weight_rule=Scalar(sp.Integer(-1)))
    )
    v = _shift(right, reg, -1)
    rs = right.add_node(DIM_SPLITTER, [reg], [two, low])
    f1, f2, f3 = (right.add_node(FOURIER_BOX, [two], [two]) for _ in range(3))
    lx = _shift(right, low, 1)
    for a, z in ((f, rb), (rb, u), (u, g), (g, v), (v, rs), (f1, f2), (f2, f3)):
        right.add_wire(_out(a), _in(z))
    right.add_wire(_out(rs, 0), _in(f1))
    right.add_wire(_out(rs, 1), _in(lx))
    right.set_boundary_inputs([_in(f), _in(rb, 1)])
    right.set_boundary_outputs([_out(f3), _out(lx)])
    right.set_scalar(Scalar(sp.Integer(-1)))
    return left, right


CARRY_SHIFT = Lemma(
    name="carry_shift",
    symbol="p",
    statement=(
        "the shift by one over 2^(p + 1), with the top bit and the low 2^p register bound by "
        "B(2, 2^p), is the top bit flipped when the low register is all ones, then the low "
        "register shifted by one over 2^p"
    ),
    build=_carry_sides,
)

_AND_LEFT = """
count p
c = Z(1 -> 2, literal 2)
t = T(literal 2)
z = Z(1 -> 1, literal 2)
h = Z(0 -> 1, literal 1)
b = B(literal 2, literal 2^m)
wire c.out1 -- t.in0, t.out0 -- z.in0, c.out0 -- b.in0, h.out0 -- b.in1
repeat p as V index m: c, t, b
chain V: b.out0 -> b.in1
inputs c.in0
outputs b.out0, z.out0
"""

_AND_RIGHT = """
count p
h = Z(0 -> 1, literal 1)
b = B(literal 2, literal 2^m)
zt = Z(0 -> 1, literal 2)
bb = B(literal 2, literal 2^p)
u = X(1 -> 1, literal 2^(p + 1)) phase{k: -k/2^(p + 1)}
g = Z(1 -> 1, literal 2^(p + 1)) weight{k: -1}
v = X(1 -> 1, literal 2^(p + 1)) phase{k: k/2^(p + 1)}
ss = S(literal 2, literal 2^p)
f = F(literal 2)
tr = T(literal 2)
wire h.out0 -- b.in1, b.out0 -- bb.in1, zt.out0 -- bb.in0, bb.out0 -- u.in0, u.out0 -- g.in0
wire g.out0 -- v.in0, v.out0 -- ss.in0, ss.out0 -- f.in0, f.out0 -- tr.in0
repeat p as V index m: b
chain V: b.out0 -> b.in1
inputs b.in0
outputs ss.out1, tr.out0
scalar -1/sqrt(2)
"""


def _and_sides(p: sp.Expr) -> tuple[Diagram, Diagram]:
    """Left: p bits, each copied into a triangle gathered by one Z spider (the AND wire), and
    bound by a B chain into a 2^p register. Right: the bits bound into the register, then
    (|0> + [register all ones] |1>) on the AND wire: the register bound under |+>, shifted by
    one over 2^(p + 1), sign -1 off index 0, shifted back, split, F then T on the top wire,
    scalar -1/sqrt(2)."""
    from archytaszx.diagram.bangbox import Mult
    from archytaszx.repl.parser import parse_dsl
    from archytaszx.semantics.induction import rename_multiplicity

    out: list[Diagram] = []
    for text in (_AND_LEFT, _AND_RIGHT):
        diagram = parse_dsl(text)
        value = sp.sympify(p).xreplace(
            {s: Mult.symbol(s.name).to_sympy() for s in sp.sympify(p).free_symbols}
        )
        if value != Mult.symbol("p").to_sympy():
            diagram = rename_multiplicity(diagram, "p", Mult._from_expr(value))
        out.append(diagram)
    return out[0], out[1]


AND_REGISTER = Lemma(
    name="and_register",
    symbol="p",
    statement=(
        "p bits copied into triangles gathered by one Z spider give |0> + AND |1>, which is "
        "|0> + [register all ones] |1> on the bits bound into a 2^p register"
    ),
    build=_and_sides,
)

LIBRARY: tuple[Lemma, ...] = (CARRY_SHIFT, AND_REGISTER)


def lemma(name: str) -> Lemma:
    """The library lemma called ``name``."""
    for entry in LIBRARY:
        if entry.name == name:
            return entry
    raise LemmaError(f"no lemma named {name!r}")


def sides(entry: Lemma, value: sp.Expr | None = None) -> tuple[Diagram, Diagram]:
    """Both sides of ``entry`` at ``value``, or with the symbol formal."""
    symbol = Dim.symbol(entry.symbol).to_sympy()
    return entry.build(symbol if value is None else sp.sympify(value))


_PROVED: dict[str, object] = {}


def proved(entry: Lemma) -> object:
    """The kernel theorem of ``entry`` with its symbol formal, proved on first use by
    :func:`~archytaszx.semantics.decide.decide_equal` with no oracle sample."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.decide import EqualityVerdict, decide_equal

    if entry.name in _PROVING:
        raise LemmaError(f"lemma {entry.name} is being proved")
    if entry.name not in _PROVED:
        _PROVING.add(entry.name)
        try:
            left, right = sides(entry)
            decision = decide_equal(left, right, samples=[])
            if decision.verdict is not EqualityVerdict.EQUAL:
                raise LemmaError(f"lemma {entry.name} is not proved: {decision.verdict.value}")
            _PROVED[entry.name] = kernel.from_decision(decision, left, right)
        finally:
            _PROVING.discard(entry.name)
    return _PROVED[entry.name]


_PROVING: set[str] = set()


def _substituted(
    entry: Lemma, theorem: object, value: sp.Expr, left: Diagram, right: Diagram
) -> object | None:
    """``left = right`` by :func:`~archytaszx.semantics.kernel.subst` of ``theorem`` at
    ``symbol = s + c`` (``value`` one symbol plus an integer, the symbol a box multiplicity of
    ``theorem``) between isomorphisms; else None."""
    from archytaszx.diagram.bangbox import free_mult_symbols
    from archytaszx.semantics import kernel
    from archytaszx.semantics.decide import DecideGrammarError
    from archytaszx.semantics.restrict import MULTIPLICITY, SHIFT, Clause

    t = cast("kernel.Theorem", theorem)
    symbols = sorted(value.free_symbols, key=lambda s: str(s.name))
    if len(symbols) != 1 or entry.symbol not in free_mult_symbols(t.lhs):
        return None
    offset = sp.expand(value - symbols[0])
    if not offset.is_Integer or int(offset) < 0:
        return None
    clause = Clause(entry.symbol, MULTIPLICITY, SHIFT, a=1, c=int(offset), fresh=symbols[0].name)
    try:
        moved = kernel.subst(t, clause)
        return kernel.trans(
            kernel.iso(left, moved.lhs), kernel.trans(moved, kernel.iso(moved.rhs, right))
        )
    except (kernel.KernelError, DecideGrammarError):
        return None


def instance(entry: Lemma, value: sp.Expr) -> object:
    """The kernel theorem of ``entry`` at ``value``, by substitution into its proved theorem, by
    symbolic contraction or path sums of both sides, else by a decision with no oracle sample."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.decide import EqualityVerdict, decide_equal

    theorem = proved(entry)
    left, right = sides(entry, value)
    shifted = _substituted(entry, theorem, sp.sympify(value), left, right)
    if shifted is not None:
        return shifted
    try:
        return kernel.contraction(left, right)
    except kernel.KernelError as exc:
        try:
            return kernel.path_sum(left, right)
        except kernel.KernelError:
            pass
        decision = decide_equal(left, right, samples=[])
        if decision.verdict is not EqualityVerdict.EQUAL:
            raise LemmaError(f"lemma {entry.name} at {value}: {exc}") from exc
        try:
            return kernel.from_decision(decision, left, right)
        except kernel.KernelError as again:
            raise LemmaError(f"lemma {entry.name} at {value}: {again}") from again


def _values(diagram: Diagram, entry: Lemma) -> list[sp.Expr]:
    """The symbolic values ``e`` and ``e - 1`` of ``entry``'s symbol for each dimension ``2^e``
    of ``diagram``, in order of first appearance."""
    found: list[sp.Expr] = []
    for _node_id, node in sorted(diagram.nodes.items()):
        for port in (*node.inputs, *node.outputs):
            expr = port.dim.to_sympy()
            if expr.is_Pow and expr.base == 2 and expr.exp.free_symbols:
                for value in (expr.exp, sp.expand(expr.exp - 1)):
                    if value not in found:
                        found.append(value)
    return found


def _regions(diagram: Diagram, pattern: Diagram) -> Iterator[frozenset[NodeId]]:
    """Node sets of ``diagram`` holding ``pattern``'s generator types in its counts and joined
    by wires as ``pattern``'s nodes are."""
    nodes = sorted(pattern.nodes)
    candidates = [
        [
            n
            for n in sorted(diagram.nodes)
            if diagram.nodes[n].generator_type is pattern.nodes[p].generator_type
        ]
        for p in nodes
    ]
    adjacency = {(w.a.node_id, w.b.node_id) for w in diagram.wires}
    adjacency |= {(b, a) for a, b in adjacency}
    pattern_edges = {(w.a.node_id, w.b.node_id) for w in pattern.wires}
    seen: set[frozenset[NodeId]] = set()

    def extend(chosen: list[NodeId]) -> Iterator[list[NodeId]]:
        if len(chosen) == len(nodes):
            yield chosen
            return
        position = len(chosen)
        for n in candidates[position]:
            if n in chosen:
                continue
            ok = all(
                (chosen[nodes.index(a)], n) in adjacency
                for a, b in pattern_edges
                if b == nodes[position] and a in nodes[:position]
            ) and all(
                (n, chosen[nodes.index(b)]) in adjacency
                for a, b in pattern_edges
                if a == nodes[position] and b in nodes[:position]
            )
            if ok:
                yield from extend([*chosen, n])

    for chosen in extend([]):
        region = frozenset(chosen)
        if region not in seen:
            seen.add(region)
            yield region


def rewrite(diagram: Diagram, entry: Lemma) -> object | None:
    """``diagram = diagram'`` by :func:`~archytaszx.semantics.kernel.congr` of an instance of
    ``entry`` whose left side ``diagram`` holds through cut wires, ``diagram'`` holding its
    right side there; None when no instance is found."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.induction import find_cut_region

    for value in _values(diagram, entry):
        try:
            left, _right = sides(entry, value)
        except DimensionError:
            continue
        for region in _regions(diagram, left):
            cut = find_cut_region(diagram, region, left)
            if cut is None:
                continue
            theorem = instance(entry, value)
            try:
                return kernel.congr(theorem, diagram, cut)  # type: ignore[arg-type]
            except kernel.KernelError:
                continue
    return None


def rewrite_all(diagram: Diagram) -> object:
    """``diagram = diagram'`` by :func:`rewrite` with each library lemma in turn, once each
    (reflexivity when none applies)."""
    from archytaszx.semantics import kernel

    theorem = kernel.refl(diagram)
    for entry in LIBRARY:
        if entry.name in _PROVING:
            continue
        try:
            step = cast("kernel.Theorem | None", rewrite(theorem.rhs, entry))
        except LemmaError:
            continue
        if step is not None:
            theorem = kernel.trans(theorem, step)
    return theorem


def _owner_cores(diagram: Diagram, index: str) -> frozenset[NodeId] | None:
    """The union of the chain cores (scope, and the nodes off it reached from its link-in ends
    without passing a scope node) of every chain box of ``diagram`` carrying ``index``; None
    when there is none or a core's head meets its tail."""
    from archytaszx.diagram.bangbox import index_owners
    from archytaszx.semantics.induction import box_core

    owners = [b for b in index_owners(diagram, index) if diagram.bang_boxes[b].links]
    if not owners:
        return None
    region: frozenset[NodeId] = frozenset()
    for owner in owners:
        core = box_core(diagram, owner)
        if core is None:
            return None
        region |= core
    return region


def close_by_cores(a: Diagram, b: Diagram, index: str) -> object:
    """``a = b`` by an isomorphism, else by
    :func:`~archytaszx.semantics.kernel.local_contraction` over the chain cores of the boxes
    carrying ``index`` (:func:`_owner_cores`), read on ``a`` as an open diagram."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.induction import _open_region

    try:
        return kernel.iso(a, b)
    except kernel.KernelError:
        pass
    region_a, region_b = _owner_cores(a, index), _owner_cores(b, index)
    if region_a is None or region_b is None:
        raise kernel.KernelError("no chain cores carrying the index on both sides")
    return kernel.local_contraction(a, region_a, b, region_b, _open_region(a, region_a))


def close_with_lemmas(replaced: Diagram, target: Diagram, source: Diagram, index: str) -> object:
    """``replaced = target`` (``target`` the joint peel of ``source``'s boxes carrying
    ``index``): library lemmas rewritten on ``replaced`` (:func:`rewrite_all`), and on
    ``target`` as it is or re-peeled with every first-peeling chain box peeling its last copy
    (:func:`~archytaszx.semantics.kernel.repeel`), the two results closed by
    :func:`close_by_cores`."""
    from archytaszx.diagram.bangbox import index_owners, peels_first
    from archytaszx.semantics import kernel

    there = cast("kernel.Theorem", rewrite_all(replaced))
    firsts = frozenset(b for b in index_owners(source, index) if peels_first(source.bang_boxes[b]))
    starts = [kernel.refl(target)]
    if firsts:
        try:
            moved = kernel.repeel(source, index, frozenset(), firsts)
            if kernel._same(moved.lhs, target):
                starts.insert(0, moved)
        except kernel.KernelError:
            pass
    reasons: list[str] = []
    for start in starts:
        back = kernel.trans(start, cast("kernel.Theorem", rewrite_all(start.rhs)))
        try:
            middle = cast("kernel.Theorem", close_by_cores(there.rhs, back.rhs, index))
        except kernel.KernelError as exc:
            reasons.append(str(exc))
            continue
        return kernel.trans(there, kernel.trans(middle, kernel.symm(back)))
    raise kernel.KernelError("lemma window: " + ("; ".join(reasons) or "no lemma applies"))


__all__ = [
    "AND_REGISTER",
    "CARRY_SHIFT",
    "LIBRARY",
    "Lemma",
    "LemmaError",
    "close_by_cores",
    "close_with_lemmas",
    "instance",
    "lemma",
    "proved",
    "rewrite",
    "rewrite_all",
    "sides",
]
