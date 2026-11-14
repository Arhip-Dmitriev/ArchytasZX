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

Each lemma is proved once, with the symbol formal, by its own prover or by
:func:`~archytaszx.semantics.decide.decide_equal` with no oracle sample, and kept with its
kernel theorem (:func:`proved`). An instance at a value of the symbol (:func:`instance`) is a
kernel theorem by symbolic contraction of the instantiated sides; :func:`rewrite` finds an
instance's left side inside a larger diagram through cut wires and replaces it by the right
side under :func:`~archytaszx.semantics.kernel.congr`.

:func:`expose_by_insertion` splits spiders off a peeled copy and inserts identities
(:data:`SPLIT_BIND`, four F boxes, a triangle pair) until the hypothesis reads through cut
wires; :func:`induct_by_insertion` proves a lemma by induction over such exposures, and
:func:`induct_by_copies` a value-count family whose peeled copies meet by a :data:`COPIES`
lemma (:func:`close_rests`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from archytaszx.semantics.induction import CutRegion
    from archytaszx.semantics.kernel import Theorem


class LemmaError(Exception):
    """A lemma is unknown, unproved, or has no instance where one was asked for."""


@dataclass(frozen=True, slots=True)
class Lemma:
    """A named identity ``build(e)[0] = build(e)[1]`` for every value ``e`` of ``symbol``,
    proved by ``prove`` (both sides formal to a kernel theorem) when given."""

    name: str
    symbol: str
    statement: str
    build: Callable[[sp.Expr], tuple[Diagram, Diagram]]
    prove: Callable[[Diagram, Diagram, str], object] | None = None


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
    return _dsl_sides(_AND_LEFT, _AND_RIGHT, p)


def _dsl_sides(left: str, right: str, p: sp.Expr) -> tuple[Diagram, Diagram]:
    """The DSL texts ``left`` and ``right`` parsed, the count ``p`` renamed to ``p``'s value."""
    from archytaszx.diagram.bangbox import Mult
    from archytaszx.repl.parser import parse_dsl
    from archytaszx.semantics.induction import rename_multiplicity

    out: list[Diagram] = []
    for text in (left, right):
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

_GATED_LEFT = _AND_LEFT.replace("z = Z(1 -> 1, literal 2)", "z = Z(2 -> 1, literal 2)").replace(
    "inputs c.in0", "inputs c.in0, z.in1"
)

_GATED_RIGHT = (
    _AND_RIGHT.replace("tr = T(literal 2)", "tr = T(literal 2)\nzz = Z(2 -> 1, literal 2)")
    .replace("f.out0 -- tr.in0", "f.out0 -- tr.in0, tr.out0 -- zz.in0")
    .replace("inputs b.in0", "inputs b.in0, zz.in1")
    .replace("outputs ss.out1, tr.out0", "outputs ss.out1, zz.out0")
)

AND_GATED = Lemma(
    name="and_gated",
    symbol="p",
    statement=("and_register with the gathering Z spider taking one more input, on both sides"),
    build=lambda p: _dsl_sides(_GATED_LEFT, _GATED_RIGHT, p),
)

_SPLIT_LEFT = """
count p
s = S(literal 2, literal 2^m)
e = Z(1 -> 0, literal 1)
c = Z(1 -> 2, literal 2)
t = T(literal 2)
z = Z(2 -> 1, literal 2)
wire s.out0 -- c.in0, c.out1 -- t.in0, t.out0 -- z.in0, s.out1 -- e.in0
repeat p as U index m: s, c, t
chain U: s.in0 -> s.out1
inputs s.in0, z.in1
outputs c.out0, z.out0
"""

_SPLIT_RIGHT = """
count p
zt = Z(0 -> 1, literal 2)
bb = B(literal 2, literal 2^p)
u = X(1 -> 1, literal 2^(p + 1)) phase{k: -k/2^(p + 1)}
g = Z(1 -> 1, literal 2^(p + 1)) weight{k: -1}
v = X(1 -> 1, literal 2^(p + 1)) phase{k: k/2^(p + 1)}
ss = S(literal 2, literal 2^p)
f = F(literal 2)
tr = T(literal 2)
zz = Z(2 -> 1, literal 2)
s = S(literal 2, literal 2^m)
e = Z(1 -> 0, literal 1)
wire zt.out0 -- bb.in0, bb.out0 -- u.in0, u.out0 -- g.in0, g.out0 -- v.in0, v.out0 -- ss.in0
wire ss.out0 -- f.in0, f.out0 -- tr.in0, tr.out0 -- zz.in0, ss.out1 -- s.in0, s.out1 -- e.in0
repeat p as U index m: s
chain U: s.in0 -> s.out1
inputs bb.in1, zz.in1
outputs s.out0, zz.out0
scalar -1/sqrt(2)
"""

AND_SPLIT = Lemma(
    name="and_split",
    symbol="p",
    statement=(
        "a 2^p register split by an S chain into p bits, each copied into a triangle gathered "
        "with one more input by one Z spider, is the register tested for all ones as in "
        "and_register, then split"
    ),
    build=lambda p: _dsl_sides(_SPLIT_LEFT, _SPLIT_RIGHT, p),
)

_ZERO_LEFT = """
count p
r = B(literal 2, literal 2^(m + 1))
zr = X(0 -> 1, literal 2)
wire zr.out0 -- r.in0
repeat p as R index m: r, zr
chain R: r.out0 -> r.in1
inputs r.in1
outputs r.out0
"""

_ZERO_RIGHT = """
count p
zr = X(0 -> 1, literal 2^p)
r = B(literal 2^p, literal 2)
wire zr.out0 -- r.in0
inputs r.in1
outputs r.out0
"""

ZERO_REGISTER = Lemma(
    name="zero_register",
    symbol="p",
    statement=(
        "a bit bound under p zero bits by a B chain is the bit bound under one zero 2^p register"
    ),
    build=lambda p: _dsl_sides(_ZERO_LEFT, _ZERO_RIGHT, p),
)

LIBRARY: tuple[Lemma, ...] = (CARRY_SHIFT, AND_REGISTER, AND_GATED, AND_SPLIT, ZERO_REGISTER)

_SPLIT_BIND_LEFT = """
count p
s = S(literal 2, literal 2^m)
e = Z(1 -> 0, literal 1)
h = Z(0 -> 1, literal 1)
b = B(literal 2, literal 2^m)
wire s.out1 -- e.in0, h.out0 -- b.in1, s.out0 -- b.in0
repeat p as U index m: s
chain U: s.in0 -> s.out1
repeat p as V index m: b
chain V: b.out0 -> b.in1
align U with V
inputs s.in0
outputs b.out0
"""

_SPLIT_BIND_RIGHT = """
count p
z = Z(1 -> 1, literal 2^p)
inputs z.in0
outputs z.out0
"""

SPLIT_BIND = Lemma(
    name="split_bind",
    symbol="p",
    statement=(
        "a 2^p register split by an S chain into p bits and bound back by an aligned B chain is "
        "the identity"
    ),
    build=lambda p: _dsl_sides(_SPLIT_BIND_LEFT, _SPLIT_BIND_RIGHT, p),
)

INSERTIONS: tuple[Lemma, ...] = (SPLIT_BIND,)


def lemma(name: str) -> Lemma:
    """The library, insertion or per-copy lemma called ``name``."""
    for entry in (*LIBRARY, *INSERTIONS, *COPIES):
        if entry.name == name:
            return entry
    raise LemmaError(f"no lemma named {name!r}")


def sides(entry: Lemma, value: sp.Expr | None = None) -> tuple[Diagram, Diagram]:
    """Both sides of ``entry`` at ``value``, or with the symbol formal."""
    symbol = Dim.symbol(entry.symbol).to_sympy()
    return entry.build(symbol if value is None else sp.sympify(value))


_PROVED: dict[str, object] = {}


def proved(entry: Lemma) -> object:
    """The kernel theorem of ``entry`` with its symbol formal, proved on first use by its
    ``prove``, else by :func:`~archytaszx.semantics.decide.decide_equal` with no oracle
    sample."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.decide import EqualityVerdict, decide_equal

    if entry.name in _PROVING:
        raise LemmaError(f"lemma {entry.name} is being proved")
    if entry.name not in _PROVED:
        _PROVING.add(entry.name)
        try:
            left, right = sides(entry)
            if entry.prove is not None:
                _PROVED[entry.name] = entry.prove(left, right, entry.symbol)
            else:
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
    ``theorem``; a negative ``c`` holds for ``s >= -c``) between isomorphisms; else None."""
    from archytaszx.diagram.bangbox import free_mult_symbols
    from archytaszx.semantics import kernel
    from archytaszx.semantics.decide import DecideGrammarError
    from archytaszx.semantics.restrict import MULTIPLICITY, SHIFT, Clause

    t = cast("kernel.Theorem", theorem)
    symbols = sorted(value.free_symbols, key=lambda s: str(s.name))
    if len(symbols) != 1 or entry.symbol not in free_mult_symbols(t.lhs):
        return None
    offset = sp.expand(value - symbols[0])
    if not offset.is_Integer:
        return None
    clause = Clause(entry.symbol, MULTIPLICITY, SHIFT, a=1, c=int(offset), fresh=symbols[0].name)
    try:
        moved = kernel.subst(t, clause)
        return kernel.trans(
            kernel.iso(left, moved.lhs), kernel.trans(moved, kernel.iso(moved.rhs, right))
        )
    except (kernel.KernelError, DecideGrammarError):
        return None


def _renamed(
    entry: Lemma, theorem: object, value: sp.Expr, left: Diagram, right: Diagram
) -> object | None:
    """``left = right`` by :func:`~archytaszx.semantics.kernel.instance` of ``theorem`` at
    ``symbol = value`` between isomorphisms; else None."""
    from archytaszx.diagram.bangbox import Mult
    from archytaszx.semantics import kernel

    t = cast("kernel.Theorem", theorem)
    try:
        moved = kernel.instance(t, entry.symbol, Mult._from_expr(value))
        return kernel.trans(
            kernel.iso(left, moved.lhs), kernel.trans(moved, kernel.iso(moved.rhs, right))
        )
    except kernel.KernelError:
        return None


def instance(entry: Lemma, value: sp.Expr) -> object:
    """The kernel theorem of ``entry`` at ``value``, by substitution into its proved theorem, by
    symbolic contraction or path sums of both sides, for a lemma with its own prover by
    :func:`~archytaszx.semantics.kernel.instance` of its proved theorem, else by a decision with
    no oracle sample; at ``value = s - c`` it holds for ``s >= c``."""
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
        renamed = (
            None
            if entry.prove is None
            else _renamed(entry, theorem, sp.sympify(value), left, right)
        )
        if renamed is not None:
            return renamed
        decision = decide_equal(left, right, samples=[])
        if decision.verdict is not EqualityVerdict.EQUAL:
            raise LemmaError(f"lemma {entry.name} at {value}: {exc}") from exc
        try:
            return kernel.from_decision(decision, left, right)
        except kernel.KernelError as again:
            raise LemmaError(f"lemma {entry.name} at {value}: {again}") from again


def _values(diagram: Diagram, entry: Lemma) -> list[sp.Expr]:
    """The symbolic values ``e``, ``e - 1`` and ``e - 2`` of ``entry``'s symbol for each
    dimension ``2^e`` of ``diagram``, in order of first appearance."""
    found: list[sp.Expr] = []
    for _node_id, node in sorted(diagram.nodes.items()):
        for port in (*node.inputs, *node.outputs):
            expr = port.dim.to_sympy()
            if expr.is_Pow and expr.base == 2 and expr.exp.free_symbols:
                for value in (expr.exp, sp.expand(expr.exp - 1), sp.expand(expr.exp - 2)):
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
    """``diagram = diagram'`` by :func:`rewrite` with each library lemma in turn, each as long
    as it applies, at most ``_MAX_REWRITES`` times (reflexivity when none applies)."""
    from archytaszx.semantics import kernel

    theorem = kernel.refl(diagram)
    for entry in LIBRARY:
        for _ in range(_MAX_REWRITES):
            if entry.name in _PROVING:
                break
            try:
                step = cast("kernel.Theorem | None", rewrite(theorem.rhs, entry))
            except LemmaError:
                break
            if step is None:
                break
            theorem = kernel.trans(theorem, step)
    return theorem


_MAX_REWRITES = 4


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


def _partners(diagram: Diagram) -> dict[PortRef, PortRef]:
    """Each wired port of ``diagram`` mapped to the port at the wire's other end."""
    out: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        out[wire.a], out[wire.b] = wire.b, wire.a
    return out


def _wire(diagram: Diagram, a: PortRef, b: PortRef) -> None:
    """``a`` and ``b`` wired, the output end first."""
    if a.direction is Direction.INPUT:
        a, b = b, a
    diagram.add_wire(a, b)


def _split_off(
    diagram: Diagram, v: NodeId, moved: frozenset[PortRef], via: PortRef
) -> tuple[Diagram, NodeId, NodeId, NodeId]:
    """``diagram`` with spider ``v`` as three spiders of its type in series: a phaseless leaf
    holding ``via``'s wire and the legs ``moved``, a phaseless 1 -> 1 middle, and a hub with
    ``v``'s phase and other legs, ``via``'s place wired to the middle; and leaf, middle, hub."""
    if any(v in box.node_scope for box in diagram.bang_boxes.values()):
        raise LemmaError("a boxed spider")
    node = diagram.nodes[v]
    partner = _partners(diagram)
    out = diagram.copy()
    dirs = (Direction.INPUT, Direction.OUTPUT)
    keep = {d: [i for i in range(len(node.legs(d))) if PortRef(v, d, i) not in moved] for d in dirs}
    hub = out.add_node(
        node.generator_type,
        [node.inputs[i].dim for i in keep[Direction.INPUT]],
        [node.outputs[i].dim for i in keep[Direction.OUTPUT]],
        phase=node.phase,
    )
    hub_of = {PortRef(v, d, i): PortRef(hub, d, j) for d in dirs for j, i in enumerate(keep[d])}
    legs = {
        d: [r for r in (via, *sorted(moved, key=PortRef.sort_key)) if r.direction is d]
        for d in dirs
    }
    dim = node.legs(via.direction)[via.index].dim
    toward = Direction.OUTPUT if via.direction is Direction.INPUT else Direction.INPUT
    leaf = out.add_node(
        node.generator_type,
        [node.inputs[r.index].dim for r in legs[Direction.INPUT]]
        + [dim] * (toward is Direction.INPUT),
        [node.outputs[r.index].dim for r in legs[Direction.OUTPUT]]
        + [dim] * (toward is Direction.OUTPUT),
    )
    leaf_of = {r: PortRef(leaf, d, j) for d in dirs for j, r in enumerate(legs[d])}
    mid = out.add_node(node.generator_type, [dim], [dim])
    for wire in list(out.wires):
        if v in (wire.a.node_id, wire.b.node_id):
            out.remove_wire(wire.a, wire.b)
    for leg, far in sorted(partner.items(), key=lambda item: item[0].sort_key()):
        if leg.node_id == v:
            if far.node_id == v:
                raise LemmaError("a spider with a self-loop")
            _wire(out, leaf_of.get(leg) or hub_of[leg], far)
    _wire(out, PortRef(leaf, toward, len(legs[toward])), PortRef(mid, via.direction, 0))
    _wire(out, PortRef(mid, toward, 0), hub_of[via])
    out.set_boundary_inputs([leaf_of.get(r) or hub_of.get(r, r) for r in diagram.boundary_inputs])
    out.set_boundary_outputs([leaf_of.get(r) or hub_of.get(r, r) for r in diagram.boundary_outputs])
    out.remove_node(v)
    return out, leaf, mid, hub


def _fusion(diagram: Diagram, a: NodeId, b: NodeId) -> tuple[object, NodeId]:
    """``diagram = diagram'`` by spider fusion of ``a`` and ``b``, and the fused node."""
    from archytaszx.rewrite.engine import apply
    from archytaszx.rewrite.match import FusionMatch
    from archytaszx.rewrite.rules_library import SPIDER_FUSION
    from archytaszx.semantics import kernel

    for match in SPIDER_FUSION.pattern.find_matches(diagram):
        if isinstance(match, FusionMatch) and {match.a_id, match.b_id} == {a, b}:
            step = apply(diagram, SPIDER_FUSION, match).step
            (fused,) = step.new_node_ids
            return kernel.rule(diagram, step), fused
    raise LemmaError("the spiders do not fuse")


def _split_theorem(
    diagram: Diagram, split: Diagram, leaf: NodeId, mid: NodeId, hub: NodeId
) -> object:
    """``diagram = split`` for a :func:`_split_off` result: two spider fusions, reversed."""
    from archytaszx.semantics import kernel

    first, fused = _fusion(split, mid, hub)
    t = cast("kernel.Theorem", first)
    second, _ = _fusion(t.rhs, leaf, fused)
    whole = kernel.trans(t, cast("kernel.Theorem", second))
    return kernel.symm(kernel.trans(whole, kernel.iso(whole.rhs, diagram)))


def _fourier_identity(dim: Dim) -> object:
    """Four F boxes over ``dim`` in series = a phaseless Z(1 -> 1), by Fourier cancellation."""
    from archytaszx.rewrite.engine import apply
    from archytaszx.rewrite.rules_library import FOURIER_CANCELLATION
    from archytaszx.semantics import kernel

    chain = Diagram()
    boxes = [chain.add_node(FOURIER_BOX, [dim], [dim]) for _ in range(4)]
    for a, b in pairwise(boxes):
        chain.add_wire(_out(a), _in(b))
    chain.set_boundary_inputs([_in(boxes[0])])
    chain.set_boundary_outputs([_out(boxes[-1])])
    for match in FOURIER_CANCELLATION.pattern.find_matches(chain):
        return kernel.rule(chain, apply(chain, FOURIER_CANCELLATION, match).step)
    raise LemmaError("no Fourier cancellation")


def _triangle_identity(dim: Dim) -> object:
    """A phaseless Z(1 -> 1) over ``dim`` = Ti, then a Z(1 -> 2) with one leg on a phaseless
    Z effect, then T, by symbolic contraction."""
    from archytaszx.diagram.generators import TRIANGLE, TRIANGLE_INVERSE
    from archytaszx.semantics import kernel

    plain = Diagram()
    z = plain.add_node(Z_SPIDER, [dim], [dim])
    plain.set_boundary_inputs([_in(z)])
    plain.set_boundary_outputs([_out(z)])
    pair = Diagram()
    ti = pair.add_node(TRIANGLE_INVERSE, [dim], [dim])
    c = pair.add_node(Z_SPIDER, [dim], [dim, dim])
    e = pair.add_node(Z_SPIDER, [dim], [])
    t = pair.add_node(TRIANGLE, [dim], [dim])
    for a, b in ((_out(ti), _in(c)), (_out(c, 0), _in(e)), (_out(c, 1), _in(t))):
        pair.add_wire(a, b)
    pair.set_boundary_inputs([_in(ti)])
    pair.set_boundary_outputs([_out(t)])
    return kernel.contraction(plain, pair)


def _inserted(
    diagram: Diagram, mid: NodeId, hub: NodeId, kind: str
) -> tuple[object, frozenset[NodeId], frozenset[NodeId]]:
    """``diagram = diagram'`` with the Z(1 -> 1) ``mid`` replaced by an identity of ``kind``
    (``split_bind``: :data:`SPLIT_BIND`; ``fourier``: four F boxes; ``triangle``:
    :func:`_triangle_identity`), the inserted nodes on ``hub``'s side (the B chain; the last F
    box; T and the Z(1 -> 2)) and the other inserted nodes."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.induction import find_cut_region

    dim = diagram.nodes[mid].inputs[0].dim
    if kind == "split_bind":
        expr = dim.to_sympy()
        if not (expr.is_Pow and expr.base == 2 and expr.exp.free_symbols):
            raise LemmaError("not a register of symbolic width")
        identity = kernel.symm(cast("kernel.Theorem", instance(SPLIT_BIND, expr.exp)))
    elif kind == "triangle":
        identity = cast("kernel.Theorem", _triangle_identity(dim))
    else:
        identity = kernel.symm(cast("kernel.Theorem", _fourier_identity(dim)))
    cut = find_cut_region(diagram, frozenset({mid}), identity.lhs)
    if cut is None:
        raise LemmaError("the middle spider is not the identity")
    t = kernel.congr(identity, diagram, cut)
    after = t.rhs
    new = frozenset(after.nodes) - frozenset(diagram.nodes)
    neighbours: dict[NodeId, set[NodeId]] = {n: set() for n in after.nodes}
    for wire in after.wires:
        neighbours[wire.a.node_id].add(wire.b.node_id)
        neighbours[wire.b.node_id].add(wire.a.node_id)
    inner = {n for n in neighbours[hub] if n in new}
    if kind == "split_bind":
        todo = list(inner)
        while todo:
            n = todo.pop()
            for m in neighbours[n]:
                if (
                    m in new
                    and m not in inner
                    and after.nodes[m].generator_type is not DIM_SPLITTER
                ):
                    inner.add(m)
                    todo.append(m)
    elif kind == "triangle":
        inner |= {
            m
            for n in inner
            for m in neighbours[n]
            if m in new and after.nodes[m].generator_type is Z_SPIDER
        }
    return t, frozenset(inner), new - frozenset(inner)


def _insertion_options(
    diagram: Diagram, v: NodeId, copy: frozenset[NodeId]
) -> list[tuple[PortRef, str]]:
    """Per leg of ``v`` wired off ``copy`` to an unboxed node or a chain end: that leg with
    each identity kind fitting its dimension (:func:`_inserted`)."""
    boxed = frozenset(n for box in diagram.bang_boxes.values() for n in box.node_scope)
    ends = frozenset(p for box in diagram.bang_boxes.values() for link in box.links for p in link)
    out: list[tuple[PortRef, str]] = []
    for leg, far in sorted(_partners(diagram).items(), key=lambda item: item[0].sort_key()):
        if leg.node_id != v or far.node_id in copy or (far.node_id in boxed and far not in ends):
            continue
        expr = diagram.nodes[v].legs(leg.direction)[leg.index].dim.to_sympy()
        if expr.is_Pow and expr.base == 2 and expr.exp.free_symbols:
            out.append((leg, "split_bind"))
        if expr == 2:
            out.append((leg, "triangle"))
        out.append((leg, "fourier"))
    return out


def expose_by_insertion(
    peeled: Diagram, copies: Sequence[frozenset[NodeId]], hypothesis: Diagram
) -> tuple[Theorem, CutRegion, frozenset[NodeId]] | None:
    """``peeled = exposed``, the cut reading ``hypothesis`` in ``exposed`` and the copy, for the
    first of ``copies`` that gives one: every unboxed Z spider off the copy wired into it and
    holding no boundary slot split (:func:`_split_off`) on a wire of
    :func:`_insertion_options`, its middle replaced by an identity (:func:`_inserted`), the
    region every node off the copy not reached from the leaves and other inserted nodes
    without passing the copy, a hub or an inner inserted node; None when no choice reads as
    ``hypothesis``."""
    from itertools import product

    from archytaszx.semantics import kernel

    boxed = frozenset(n for box in peeled.bang_boxes.values() for n in box.node_scope)
    slots = frozenset(r.node_id for r in (*peeled.boundary_inputs, *peeled.boundary_outputs))
    partner = _partners(peeled)
    for copy in copies:
        spiders = [
            v
            for v in sorted(peeled.nodes)
            if v not in copy
            and v not in boxed
            and v not in slots
            and peeled.nodes[v].generator_type is Z_SPIDER
            and any(leg.node_id == v and far.node_id in copy for leg, far in partner.items())
        ]
        if not spiders or len(spiders) > _MAX_INSERTED:
            continue
        choices = [_insertion_options(peeled, v, copy) for v in spiders]
        for combo in product(*choices):
            try:
                found = _expose_with(peeled, copy, hypothesis, spiders, combo)
            except (kernel.KernelError, LemmaError, DimensionError):
                continue
            if found is not None:
                return (*found, copy)
    return None


_MAX_INSERTED = 3


def _expose_with(
    peeled: Diagram,
    copy: frozenset[NodeId],
    hypothesis: Diagram,
    spiders: list[NodeId],
    combo: tuple[tuple[PortRef, str], ...],
) -> tuple[Theorem, CutRegion] | None:
    """:func:`expose_by_insertion` at one choice of wire and identity per spider."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.induction import find_cut_region

    theorem = kernel.refl(peeled)
    leaves: set[NodeId] = set()
    blocked: set[NodeId] = set()
    for v, (via, kind) in zip(spiders, combo, strict=True):
        current = theorem.rhs
        moved = frozenset(
            leg
            for leg, far in _partners(current).items()
            if leg.node_id == v and far.node_id in copy
        )
        split, leaf, mid, hub = _split_off(current, v, moved, via)
        step = cast("kernel.Theorem", _split_theorem(current, split, leaf, mid, hub))
        inserted, inner, outer = _inserted(split, mid, hub, kind)
        theorem = kernel.trans(theorem, kernel.trans(step, cast("kernel.Theorem", inserted)))
        leaves |= {leaf, *outer}
        blocked |= {hub, *inner}
    exposed = theorem.rhs
    neighbours: dict[NodeId, set[NodeId]] = {n: set() for n in exposed.nodes}
    for wire in exposed.wires:
        neighbours[wire.a.node_id].add(wire.b.node_id)
        neighbours[wire.b.node_id].add(wire.a.node_id)
    reached = set(copy) | leaves
    todo = list(leaves)
    while todo:
        n = todo.pop()
        for m in neighbours[n]:
            if m not in reached and m not in blocked:
                reached.add(m)
                todo.append(m)
    cut = find_cut_region(exposed, frozenset(exposed.nodes) - frozenset(reached), hypothesis)
    return None if cut is None else (theorem, cut)


def induct_by_insertion(left: Diagram, right: Diagram, index: str) -> object:
    """``left = right`` for every value of ``index`` by
    :func:`~archytaszx.semantics.kernel.induction`: the base by symbolic contraction, one
    side's peeled successor (as it is or re-peeled at its chains' last copies,
    :func:`_repeels`) exposed by :func:`expose_by_insertion` and its hypothesis replaced by the
    other side's, the step closed by :func:`close_by_cores` on the other peeled successor as it
    is or re-peeled; each side tried in turn."""
    from archytaszx.diagram.bangbox import BangBoxError
    from archytaszx.diagram.validate import ValidateError
    from archytaszx.semantics import kernel
    from archytaszx.semantics.induction import (
        InductionError,
        _fresh_step_symbol,
        copy_candidates,
        replace_cut_region,
    )
    from archytaszx.semantics.prove import (
        InductionSide,
        induction_base,
        induction_hypothesis,
        induction_renamed,
        induction_successor,
    )

    k = _fresh_step_symbol(left, right, index, "k")
    base = kernel.contraction(induction_base(left, index), induction_base(right, index))
    reasons: list[str] = []
    for side, own, other in ((InductionSide.START, left, right), (InductionSide.GOAL, right, left)):
        peeled = induction_successor(own, index, k)
        other_peeled = induction_successor(other, index, k)
        if peeled is None or other_peeled is None:
            reasons.append(f"{side.value}: no single box to peel")
            continue
        own_k, other_k = (induction_hypothesis(d, index, k) for d in (own, other))
        found = None
        for start in _repeels(induction_renamed(own, index, k), peeled, k):
            copy = frozenset(start.rhs.nodes) - frozenset(own_k.nodes)
            candidates = copy_candidates(start.rhs, frozenset(own_k.nodes), copy)
            found = expose_by_insertion(start.rhs, candidates, own_k)
            if found is not None:
                found = (kernel.trans(start, found[0]), found[1], found[2])
                break
        if found is None:
            reasons.append(f"{side.value}: no insertion exposes the hypothesis")
            continue
        expose, cut, _copy = found
        errors = (kernel.KernelError, InductionError, BangBoxError, ValidateError)
        try:
            replaced = replace_cut_region(expose.rhs, cut, own_k, other_k)
        except errors as exc:
            reasons.append(f"{side.value}: {exc}")
            continue
        for back in _repeels(induction_renamed(other, index, k), other_peeled, k):
            try:
                there = cast("kernel.Theorem", close_by_cores(replaced, back.rhs, k))
                step = kernel.trans(there, kernel.symm(back))
                return kernel.induction(
                    left, right, index, k, base, side, expose, cut.region, step, cut=cut
                )
            except errors as exc:
                reasons.append(f"{side.value}: {exc}")
    raise LemmaError("; ".join(reasons))


def _repeels(source: Diagram, peeled: Diagram, index: str) -> list[Theorem]:
    """``peeled = peeled`` and, when a chain box of ``source`` carrying ``index`` peels its
    first copy, ``peeled = p'`` with every such box peeling its last copy
    (:func:`~archytaszx.semantics.kernel.repeel`)."""
    from archytaszx.diagram.bangbox import index_owners, peels_first
    from archytaszx.semantics import kernel

    out: list[Theorem] = [kernel.refl(peeled)]
    firsts = frozenset(b for b in index_owners(source, index) if peels_first(source.bang_boxes[b]))
    if firsts:
        try:
            moved = kernel.repeel(source, index, frozenset(), firsts)
        except kernel.KernelError:
            return out
        if kernel._same(moved.lhs, peeled):
            out.append(kernel.trans(kernel.iso(peeled, moved.lhs), moved))
    return out


_COPY_LEFT = """
count p
g0 = Z(1 -> 2, literal 2)
g1 = Z(1 -> 2, literal 2)
g3 = S(literal 2, literal 2^j)
g4 = Z(1 -> 0, literal 1)
g5 = X(2 -> 1, literal 2)
g6 = X(1 -> 1, literal 2) phase{1: 1/2}
g7 = T(literal 2)
g8 = Z(2 -> 1, literal 2)
g9 = T(literal 2)
g10 = Ti(literal 2)
g11 = X(2 -> 1, literal 2)
wire g3.out1 -- g4.in0, g3.out0 -- g5.in1, g0.out1 -- g5.in0, g5.out0 -- g6.in0
wire g6.out0 -- g7.in0, g7.out0 -- g8.in1, g1.out1 -- g9.in0, g9.out0 -- g8.in0
wire g8.out0 -- g10.in0, g10.out0 -- g11.in1
repeat p as C index j: g0, g3, g5, g6, g7
chain C: g3.in0 -> g3.out1
inputs g0.in0, g1.in0, g11.in0, g3.in0
outputs g1.out0, g0.out0, g11.out0
scalar sqrt(2)
"""

_COPY_RIGHT = """
count p
h = Z(0 -> 1, literal 1)
b = B(literal 2, literal 2^m)
z = Z(1 -> 2, literal 2^p)
xs = X(1 -> 2, literal 2^p)
cap = Z(2 -> 0, literal 2^p)
sr = S(literal 2, literal 2^m)
er = Z(1 -> 0, literal 1)
hr = Z(0 -> 1, literal 1)
br = B(literal 2, literal 2^m)
c = Z(1 -> 2, literal 2)
pb = B(literal 2^p, literal 2)
qb = B(literal 2^(p + 1), literal 2)
g = Z(1 -> 0, literal 2^(p + 2)) phase{3: 1/2}
f0 = F(literal 2)
y = Z(1 -> 2, literal 2)
f1 = F(literal 2)
f2 = F(literal 2)
f3 = F(literal 2)
s = S(literal 2, literal 2^m)
e = Z(1 -> 0, literal 1)
wire h.out0 -- b.in1, b.out0 -- z.in0, z.out0 -- s.in0, s.out1 -- e.in0
wire sr.out1 -- er.in0, sr.out0 -- br.in0, hr.out0 -- br.in1, br.out0 -- cap.in1
wire z.out1 -- xs.in0, xs.out0 -- pb.in0, xs.out1 -- cap.in0, c.out1 -- pb.in1, pb.out0 -- qb.in0
wire y.out1 -- qb.in1, qb.out0 -- g.in0
wire f0.out0 -- y.in0, y.out0 -- f1.in0, f1.out0 -- f2.in0, f2.out0 -- f3.in0
repeat p as V index m: b
chain V: b.out0 -> b.in1
repeat p as UR index m: sr
chain UR: sr.in0 -> sr.out1
repeat p as Q index m: br
chain Q: br.out0 -> br.in1
align UR with Q
repeat p as U index m: s
chain U: s.in0 -> s.out1
inputs b.in0, c.in0, f0.in0, sr.in0
outputs c.out0, s.out0, f3.out0
"""

QRAM_COPY = Lemma(
    name="qram_copy",
    symbol="p",
    statement=(
        "p address bits, each flipped unless it matches the bit of a 2^p register R split by an "
        "S chain, gathered with a data bit through triangles into a controlled flip of the "
        "target, is the address bound into a 2^p register less R (split and bound back), "
        "tested for 0 with the data bit and the Fourier-side target"
    ),
    build=lambda p: _dsl_sides(_COPY_LEFT, _COPY_RIGHT, p),
    prove=induct_by_insertion,
)

COPIES: tuple[Lemma, ...] = (QRAM_COPY,)


def close_rests(a: Diagram, b: Diagram) -> object:
    """``a = b`` by an isomorphism, else by :func:`~archytaszx.semantics.kernel.congr` of a
    :data:`COPIES` instance whose left side ``a`` holds through cut wires, :data:`SPLIT_BIND`
    rewritten on the result as long as it applies, and :func:`close_by_cores` over a free
    multiplicity's chain cores."""
    from archytaszx.diagram.bangbox import free_mult_symbols
    from archytaszx.semantics import kernel
    from archytaszx.semantics.induction import find_cut_region

    try:
        return kernel.iso(a, b)
    except kernel.KernelError:
        pass
    reasons: list[str] = []
    for entry in COPIES:
        for value in _values(a, entry):
            try:
                lhs, _rhs = sides(entry, value)
            except DimensionError:
                continue
            for region in _regions(a, lhs):
                cut = find_cut_region(a, region, lhs)
                if cut is None:
                    continue
                try:
                    theorem = kernel.congr(cast("kernel.Theorem", instance(entry, value)), a, cut)
                except (kernel.KernelError, LemmaError) as exc:
                    reasons.append(str(exc))
                    continue
                for _ in range(_MAX_REWRITES):
                    step = cast("kernel.Theorem | None", rewrite(theorem.rhs, SPLIT_BIND))
                    if step is None:
                        break
                    theorem = kernel.trans(theorem, step)
                for symbol in sorted(free_mult_symbols(theorem.rhs)):
                    try:
                        closed = cast("kernel.Theorem", close_by_cores(theorem.rhs, b, symbol))
                    except kernel.KernelError as exc:
                        reasons.append(str(exc))
                        continue
                    return kernel.trans(theorem, closed)
    raise LemmaError("rests: " + ("; ".join(reasons) or "no copy lemma applies"))


def induct_by_copies(left: Diagram, right: Diagram, index: str) -> Theorem:
    """``left = right`` for every value of ``index`` by
    :func:`~archytaszx.semantics.kernel.induction`: the base by a decision with no oracle
    sample, one side's peeled successor exposed by :func:`expose_by_insertion` and its
    hypothesis replaced by the other side's, the other side's peeled successor exposed by
    :func:`~archytaszx.semantics.induction.expose_cut`, and the parts off the two hypothesis
    regions met by :func:`close_rests`; each side tried in turn."""
    from archytaszx.diagram.bangbox import BangBoxError
    from archytaszx.diagram.validate import ValidateError
    from archytaszx.semantics import kernel
    from archytaszx.semantics.decide import EqualityVerdict, decide_equal
    from archytaszx.semantics.induction import (
        InductionError,
        _fresh_step_symbol,
        _open_region,
        copy_candidates,
        expose_cut,
        find_cut_region,
        replace_cut_region,
    )
    from archytaszx.semantics.induction_kernel import NoInductionTheorem, _steps_theorem
    from archytaszx.semantics.prove import (
        InductionSide,
        induction_base,
        induction_hypothesis,
        induction_successor,
    )

    k = _fresh_step_symbol(left, right, index, "k")
    errors = (kernel.KernelError, LemmaError, InductionError, BangBoxError, ValidateError)
    reasons: list[str] = []
    base: kernel.Theorem | None = None
    for side, own, other in ((InductionSide.START, left, right), (InductionSide.GOAL, right, left)):
        peeled = induction_successor(own, index, k)
        other_peeled = induction_successor(other, index, k)
        if peeled is None or other_peeled is None:
            reasons.append(f"{side.value}: no single box to peel")
            continue
        own_k, other_k = (induction_hypothesis(d, index, k) for d in (own, other))
        copy = frozenset(peeled.nodes) - frozenset(own_k.nodes)
        found = expose_by_insertion(
            peeled, copy_candidates(peeled, frozenset(own_k.nodes), copy), own_k
        )
        held = frozenset(other_peeled.nodes) - frozenset(other_k.nodes)
        there = expose_cut(
            other_peeled, copy_candidates(other_peeled, frozenset(other_k.nodes), held), other_k
        )
        if found is None or there is None:
            reasons.append(f"{side.value}: a hypothesis is not exposed")
            continue
        expose, cut, _copy = found
        exposed, other_cut, steps = there
        try:
            replaced = replace_cut_region(expose.rhs, cut, own_k, other_k)
            rest = frozenset(expose.rhs.nodes) - cut.region
            other_rest = frozenset(exposed.nodes) - other_cut.region
            rests = cast(
                "kernel.Theorem",
                close_rests(_open_region(exposed, other_rest), _open_region(replaced, rest)),
            )
            placed = find_cut_region(exposed, other_rest, rests.lhs)
            if placed is None:
                raise LemmaError("the other rest is not its open diagram")
            moved = kernel.trans(
                _steps_theorem(other_peeled, steps), kernel.congr(rests, exposed, placed)
            )
            step = kernel.trans(kernel.iso(replaced, moved.rhs), kernel.symm(moved))
            if base is None:
                at_left, at_right = induction_base(left, index), induction_base(right, index)
                decision = decide_equal(at_left, at_right, samples=[])
                if decision.verdict is not EqualityVerdict.EQUAL:
                    raise LemmaError(f"base: {decision.verdict.value}")
                base = kernel.from_decision(decision, at_left, at_right)
            return kernel.induction(
                left, right, index, k, base, side, expose, cut.region, step, cut=cut
            )
        except (*errors, NoInductionTheorem) as exc:
            reasons.append(f"{side.value}: {exc}")
    raise LemmaError("; ".join(reasons))


__all__ = [
    "AND_REGISTER",
    "CARRY_SHIFT",
    "COPIES",
    "INSERTIONS",
    "LIBRARY",
    "QRAM_COPY",
    "SPLIT_BIND",
    "Lemma",
    "LemmaError",
    "close_by_cores",
    "close_rests",
    "close_with_lemmas",
    "expose_by_insertion",
    "induct_by_copies",
    "induct_by_insertion",
    "instance",
    "lemma",
    "proved",
    "rewrite",
    "rewrite_all",
    "sides",
]
