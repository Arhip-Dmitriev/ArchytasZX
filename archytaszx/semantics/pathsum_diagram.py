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

"""The path sum of a diagram whose every dimension is a concrete power of two.

Each leg of dimension 2^k is k Boolean digits, value ``Sum_j 2^j x_j``. A Z spider identifies
the digits of all its legs and adds the multilinear interpolation of its phase table; an X
spider adds a fresh digit vector y with the phase ``y * (signed leg total) / 2^k`` and its
table over y; F adds ``x * z / 2^k``; B and S regroup digit vectors. Boundary digits are
variables 0, 1, ... in boundary order (outputs, then inputs), low digit first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, islice, product

from archytaszx.algebra.pathsum import (
    SQRT2,
    Cyclo,
    F2Poly,
    ModPoly,
    PathSum,
    PathSumUnsupported,
    PhasePoly,
    QuditPathSum,
    canonical,
    canonical_qudit,
    constant,
    entry_enumerated,
    entry_enumerated_qudit,
    reduce,
    reduce_qudit,
    residual_form,
    sqrt_prime,
    with_sqrt2,
    witness_points,
    witness_points_qudit,
)
from archytaszx.diagram.bangbox import expand_concrete_boxes
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    X_SPIDER,
    Z_SPIDER,
)
from archytaszx.diagram.graph import Diagram, Direction, Node, NodeId, PortRef

_Item = tuple[object, ...]


class _Union:
    """Union-find over hashable items."""

    def __init__(self) -> None:
        self.parent: dict[_Item, _Item] = {}

    def find(self, x: _Item) -> _Item:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: _Item, b: _Item) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if repr(rb) < repr(ra):
                ra, rb = rb, ra
            self.parent[rb] = ra


def _power_of(d: int, base: int) -> int | None:
    """The k with ``d = base^k``, or None."""
    k = 0
    while d > 1 and d % base == 0:
        d //= base
        k += 1
    return k if d == 1 else None


def _bits(diagram: Diagram, ref: PortRef, base: int = 2) -> int:
    """The number of base-``base`` digits of one port; raises unless its dimension is a
    concrete power of ``base``."""
    node = diagram.nodes[ref.node_id]
    dim = node.legs(ref.direction)[ref.index].dim
    if not dim.is_concrete:
        raise PathSumUnsupported(f"node {ref.node_id!r} has a symbolic dimension {dim}")
    k = _power_of(dim.to_int(), base)
    if k is None:
        raise PathSumUnsupported(
            f"node {ref.node_id!r} has dimension {dim.to_int()}, not a power of {base}"
        )
    return k


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % q for q in range(2, int(n**0.5) + 1))


def diagram_base(diagram: Diagram) -> int:
    """2 when every leg dimension is a power of two, else the odd prime p every leg dimension
    is a power of; raises :class:`PathSumUnsupported` otherwise."""
    dims = set()
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        for ref, _sign in _legs(node_id, node):
            dim = node.legs(ref.direction)[ref.index].dim
            if not dim.is_concrete:
                raise PathSumUnsupported(f"node {node_id!r} has a symbolic dimension {dim}")
            dims.add(dim.to_int())
        if not _legs(node_id, node) and node.phase is not None and node.phase.dim.is_concrete:
            dims.add(node.phase.dim.to_int())
    big = sorted(d for d in dims if d > 1)
    if not big or all(_power_of(d, 2) is not None for d in big):
        return 2
    p = next(q for q in range(2, big[0] + 1) if big[0] % q == 0)
    if p == 2 or not _is_prime(p) or any(_power_of(d, p) is None for d in big):
        raise PathSumUnsupported(
            f"leg dimensions {big} are neither powers of two nor powers of one odd prime"
        )
    return p


def _legs(node_id: NodeId, node: Node) -> list[tuple[PortRef, int]]:
    """Every leg of a node with its sign: outputs +1, inputs -1."""
    return [
        (PortRef(node_id, direction, position), sign)
        for direction, sign in ((Direction.OUTPUT, 1), (Direction.INPUT, -1))
        for position in range(len(node.legs(direction)))
    ]


def _turns_table(node_id: NodeId, node: Node, d: int) -> list[Fraction]:
    """The phase of each value 0 .. d-1 in turns; raises on a weighted or irrational phase."""
    phase = node.phase
    if phase is None:
        return [Fraction(0)] * d
    if phase.is_weighted:
        raise PathSumUnsupported(f"node {node_id!r} carries weights")
    out = [Fraction(0)]
    for v in range(1, d):
        t = phase.get(v).to_sympy_turns()
        if not t.is_Rational:
            raise PathSumUnsupported(f"node {node_id!r} has a phase {t} that is not rational")
        out.append(Fraction(int(t.p), int(t.q)))
    return out


def _interpolate(
    phase: PhasePoly, digits: list[int], table: list[Fraction], times: int | None = None
) -> None:
    """Add the multilinear interpolation of ``table`` over ``digits`` (Moebius inversion),
    each term multiplied by the variable ``times`` when given."""
    k = len(digits)
    if not any(table):
        return
    extra = () if times is None else (times,)
    for size in range(k + 1):
        for subset in combinations(range(k), size):
            coeff = Fraction(0)
            for inner in range(size + 1):
                for sub in combinations(subset, inner):
                    coeff += (-1) ** (size - inner) * table[sum(1 << j for j in sub)]
            if coeff % 1:
                phase.add((*(digits[j] for j in subset), *extra), coeff)


def _f2_indicator(digits: list[int], table: list[int]) -> F2Poly:
    """The F_2 polynomial over ``digits`` with the 0/1 values ``table`` (Moebius mod 2)."""
    k = len(digits)
    out: set[frozenset[int]] = set()
    for size in range(k + 1):
        for subset in combinations(range(k), size):
            coeff = 0
            for inner in range(size + 1):
                for sub in combinations(subset, inner):
                    coeff += table[sum(1 << j for j in sub)]
            if coeff % 2:
                out.add(frozenset(digits[j] for j in subset))
    return frozenset(out)


def _weights(
    node_id: NodeId, node: Node, d: int
) -> tuple[list[Fraction], list[int], list[int], Fraction, int]:
    """A weighted table as turns, zero indicator, sqrt-2 indicator and base magnitude (a
    rational times sqrt(2)^power): each nonzero coefficient is a root of unity times the base
    or sqrt(2) times it; raises otherwise."""
    assert node.phase is not None
    turns: list[Fraction] = []
    zero: list[int] = []
    mags: list[tuple[Fraction, Fraction, int] | None] = []
    for v in range(d):
        c, k = constant(node.phase.coefficient(v).to_sympy())
        if c.is_zero:
            turns.append(Fraction(0))
            zero.append(1)
            mags.append(None)
            continue
        terms = [(e, q) for e, q in c.c.items() if q]
        if len(terms) != 1:
            raise PathSumUnsupported(
                f"node {node_id!r} has a weight {c} that is not a root times a real"
            )
        ((e, q),) = terms
        turns.append((Fraction(e, c.n) + (Fraction(1, 2) if q < 0 else 0)) % 1)
        zero.append(0)
        mags.append((abs(q) ** 2 * Fraction(2) ** k, abs(q), k))
    present = [m for m in mags if m is not None]
    if not present:
        raise PathSumUnsupported(f"node {node_id!r} has every weight zero")
    base = min(present, key=lambda m: m[0])
    root2 = []
    for m in mags:
        ratio = Fraction(1) if m is None else m[0] / base[0]
        if ratio not in (1, 2):
            raise PathSumUnsupported(f"node {node_id!r} has weights of magnitude ratio {ratio}")
        root2.append(1 if ratio == 2 else 0)
    return turns, zero, root2, base[1], base[2]


def port_bits(diagram: Diagram, base: int = 2) -> list[int]:
    """The digit count of each boundary port, outputs then inputs."""
    return [
        _bits(diagram, ref, base) for ref in (*diagram.boundary_outputs, *diagram.boundary_inputs)
    ]


def from_diagram(
    diagram: Diagram, layout: dict[str, object] | None = None
) -> PathSum | QuditPathSum:
    """The path sum of ``diagram`` (a :class:`QuditPathSum` at an odd prime base); raises
    :class:`PathSumUnsupported` naming the first node outside the class: a generator other than
    Z, X, F, B and S, a dimension that is symbolic or not a power of two or of one odd prime, a
    weighted or irrational phase, or a symbolic bang box. ``layout``, when given, receives the
    diagram read (``diagram``, concrete boxes expanded), each port's digit variables
    (``digit``), each X spider's fresh digits (``fresh``) and each legless Z spider's digits
    (``zdigits``)."""
    if diagram.bang_boxes:
        diagram = expand_concrete_boxes(diagram)
        if diagram.bang_boxes:
            raise PathSumUnsupported("a bang box with a symbolic multiplicity")
    allowed = {Z_SPIDER.name, X_SPIDER.name, FOURIER_BOX.name, DIM_BINDER.name, DIM_SPLITTER.name}
    for node_id in sorted(diagram.nodes):
        name = diagram.nodes[node_id].generator_type.name
        if name not in allowed:
            raise PathSumUnsupported(f"node {node_id!r} is a {name}")
    base = diagram_base(diagram)
    uf = _Union()
    bits: dict[PortRef, int] = {}
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        for ref, _sign in _legs(node_id, node):
            bits[ref] = _bits(diagram, ref, base)
            for j in range(bits[ref]):
                uf.find(("p", ref.sort_key(), j))
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if bits[wire.a] != bits[wire.b]:
            raise PathSumUnsupported(
                f"a wire joins dimensions {base}^{bits[wire.a]}, {base}^{bits[wire.b]}"
            )
        for j in range(bits[wire.a]):
            uf.union(("p", wire.a.sort_key(), j), ("p", wire.b.sort_key(), j))
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        name = node.generator_type.name
        if name == Z_SPIDER.name:
            legs = _legs(node_id, node)
            k = bits[legs[0][0]] if legs else _phase_bits(node_id, node, base)
            for j in range(k):
                uf.find(("z", node_id, j))
            for ref, _sign in legs:
                for j in range(k):
                    uf.union(("p", ref.sort_key(), j), ("z", node_id, j))
        elif name in (DIM_BINDER.name, DIM_SPLITTER.name):
            whole_dir, part_dir = (
                (Direction.OUTPUT, Direction.INPUT)
                if name == DIM_BINDER.name
                else (Direction.INPUT, Direction.OUTPUT)
            )
            whole = PortRef(node_id, whole_dir, 0)
            high, low = PortRef(node_id, part_dir, 0), PortRef(node_id, part_dir, 1)
            q = bits[low]
            for j in range(q):
                uf.union(("p", whole.sort_key(), j), ("p", low.sort_key(), j))
            for j in range(bits[high]):
                uf.union(("p", whole.sort_key(), q + j), ("p", high.sort_key(), j))
    var_of: dict[_Item, int] = {}
    constraints = []
    same: list[tuple[int, int]] = []
    boundary: list[int] = []
    for ref in (*diagram.boundary_outputs, *diagram.boundary_inputs):
        for j in range(bits[ref]):
            v = len(boundary)
            boundary.append(v)
            root = uf.find(("p", ref.sort_key(), j))
            if root in var_of:
                constraints.append(frozenset({frozenset({var_of[root]}), frozenset({v})}))
                same.append((var_of[root], v))
            else:
                var_of[root] = v
    path: set[int] = set()
    roots = sorted({uf.find(x) for x in list(uf.parent)}, key=repr)
    for root in roots:
        if root not in var_of:
            var_of[root] = len(boundary) + len(path)
            path.add(var_of[root])
    counter = [len(boundary) + len(path)]

    def fresh() -> int:
        counter[0] += 1
        path.add(counter[0] - 1)
        return counter[0] - 1

    def digit(ref: PortRef, j: int) -> int:
        return var_of[uf.find(("p", ref.sort_key(), j))]

    fresh_of: dict[NodeId, list[int]] = {}
    if layout is not None:
        layout["diagram"] = diagram
        layout["digit"] = {ref: [digit(ref, j) for j in range(k)] for ref, k in bits.items()}
        layout["fresh"] = fresh_of
        layout["zdigits"] = {
            nid: [var_of[uf.find(("z", nid, j))] for j in range(_phase_bits(nid, n, base))]
            for nid, n in diagram.nodes.items()
            if n.generator_type.name == Z_SPIDER.name and not _legs(nid, n)
        }
    if base != 2:
        return _qudit_sum(diagram, base, bits, uf, var_of, boundary, path, same, fresh, digit)
    phase = PhasePoly()
    amps: list[tuple[Cyclo, F2Poly]] = []
    c, s2 = constant(diagram.scalar.to_sympy())
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        name = node.generator_type.name
        legs = _legs(node_id, node)
        if name == Z_SPIDER.name:
            k = bits[legs[0][0]] if legs else _phase_bits(node_id, node)
            digits = [var_of[uf.find(("z", node_id, j))] for j in range(k)]
            if node.phase is None or not node.phase.is_weighted:
                _interpolate(phase, digits, _turns_table(node_id, node, 1 << k))
                continue
            turns, zero, root2, mag, mag_k = _weights(node_id, node, 1 << k)
            _interpolate(phase, digits, turns)
            c, s2 = c.scale(mag), s2 + mag_k
            if any(zero):
                amps.append((Cyclo.rational(0), _f2_indicator(digits, zero)))
            if any(root2):
                amps.append((SQRT2, _f2_indicator(digits, root2)))
        elif name == X_SPIDER.name:
            k = bits[legs[0][0]] if legs else _phase_bits(node_id, node)
            ys = [fresh() for _ in range(k)]
            fresh_of[node_id] = ys
            for ref, sign in legs:
                for a in range(k):
                    for b in range(k - a):
                        phase.add((ys[a], digit(ref, b)), Fraction(sign * (1 << (a + b)), 1 << k))
            _interpolate(phase, ys, _turns_table(node_id, node, 1 << k))
            s2 -= k * len(legs)
        elif name == FOURIER_BOX.name:
            out, inp = PortRef(node_id, Direction.OUTPUT, 0), PortRef(node_id, Direction.INPUT, 0)
            k = bits[out]
            for a in range(k):
                for b in range(k - a):
                    phase.add((digit(out, a), digit(inp, b)), Fraction(1 << (a + b), 1 << k))
            s2 -= k
    return PathSum(tuple(boundary), path, phase, c, s2, constraints, [], amps)


def _phase_bits(node_id: NodeId, node: Node, base: int = 2) -> int:
    """The digit count of a legless spider, from its phase vector's dimension."""
    if node.phase is None:
        raise PathSumUnsupported(f"node {node_id!r} has no legs and no phase vector")
    dim = node.phase.dim
    k = _power_of(dim.to_int(), base) if dim.is_concrete else None
    if k is None:
        raise PathSumUnsupported(f"node {node_id!r} has dimension {dim}, not a power of {base}")
    return k


def _qudit_table(node_id: NodeId, node: Node, digits: list[int], p: int) -> ModPoly:
    """The F_p polynomial over base-p ``digits`` whose value is p times the phase table in
    turns (tensor Lagrange interpolation); raises when an entry is not a multiple of 1/p."""
    k = len(digits)
    table = _turns_table(node_id, node, p**k)
    out = ModPoly(p)
    for value, t in enumerate(table):
        a = t * p
        if a.denominator != 1:
            raise PathSumUnsupported(
                f"node {node_id!r} has a phase {t} that is not a multiple of 1/{p}"
            )
        if not int(a) % p:
            continue
        term = ModPoly.const(p, int(a))
        for j, x in enumerate(digits):
            c = value // p**j % p
            shifted = ModPoly(p, {((x, 1),): 1, (): -c % p} if c else {((x, 1),): 1})
            term = term * (ModPoly.const(p, 1) + shifted.power(p - 1).scale(-1))
        out = out + term
    return out


def _qudit_sum(
    diagram: Diagram,
    p: int,
    bits: dict[PortRef, int],
    uf: _Union,
    var_of: dict[_Item, int],
    boundary: list[int],
    path: set[int],
    same: list[tuple[int, int]],
    fresh: Callable[[], int],
    digit: Callable[[PortRef, int], int],
) -> QuditPathSum:
    """The phase, constant and constraints of an odd prime base: Z adds its interpolated
    table, X on dimension p adds ``y * (signed leg total) + table(y)`` and p^(-legs/2), F on
    dimension p adds ``x * z`` and p^(-1/2); X and F on p^k, k > 1, raise."""
    phase = ModPoly(p)
    atoms: list[tuple[ModPoly, tuple[Cyclo, ...]]] = []
    katoms: list[tuple[tuple[ModPoly, ...], tuple[Cyclo, ...]]] = []
    c, s2 = constant(diagram.scalar.to_sympy())
    if s2:
        c = with_sqrt2(c, s2)
    inv_root = sqrt_prime(p).scale(Fraction(1, p))
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        name = node.generator_type.name
        legs = _legs(node_id, node)
        if name == Z_SPIDER.name:
            k = bits[legs[0][0]] if legs else _phase_bits(node_id, node, p)
            digits = [var_of[uf.find(("z", node_id, j))] for j in range(k)]
            phase = phase + _qudit_part(node_id, node, digits, p, atoms, katoms)
        elif name in (X_SPIDER.name, FOURIER_BOX.name):
            k = bits[legs[0][0]] if legs else _phase_bits(node_id, node, p)
            if k != 1:
                raise PathSumUnsupported(f"node {node_id!r} is a {name} on dimension {p}^{k}")
            if name == FOURIER_BOX.name:
                out, inp = (
                    PortRef(node_id, Direction.OUTPUT, 0),
                    PortRef(node_id, Direction.INPUT, 0),
                )
                if digit(out, 0) == digit(inp, 0):
                    raise PathSumUnsupported(f"node {node_id!r} is an F on a self-loop")
                phase.add(tuple(sorted(((digit(out, 0), 1), (digit(inp, 0), 1)))), 1)
                c = c * inv_root
                continue
            y = fresh()
            for ref, sign in legs:
                phase.add(
                    ((y, 1), (digit(ref, 0), 1))
                    if y < digit(ref, 0)
                    else (
                        (digit(ref, 0), 1),
                        (y, 1),
                    ),
                    sign,
                )
            phase = phase + _qudit_part(node_id, node, [y], p, atoms, katoms)
            for _leg in legs:
                c = c * inv_root
    constraints = [ModPoly(p, {((a, 1),): 1, ((b, 1),): p - 1}) for a, b in same]
    return QuditPathSum(p, tuple(boundary), path, phase, c, constraints, [], atoms, katoms)


def _qudit_part(
    node_id: NodeId,
    node: Node,
    digits: list[int],
    p: int,
    atoms: list[tuple[ModPoly, tuple[Cyclo, ...]]],
    katoms: list[tuple[tuple[ModPoly, ...], tuple[Cyclo, ...]]],
) -> ModPoly:
    """A spider table as an F_p phase polynomial, or, on dimension p when its entries are not
    multiples of 1/p or carry weights, as an atom on its variable (its p coefficients)."""
    try:
        return _qudit_table(node_id, node, digits, p)
    except PathSumUnsupported:
        if node.phase is None:
            raise
    table = []
    for v in range(p ** len(digits)):
        cv, kv = constant(node.phase.coefficient(v).to_sympy())
        table.append(with_sqrt2(cv, kv))
    if len(digits) == 1:
        atoms.append((ModPoly(p, {((digits[0], 1),): 1}), tuple(table)))
    else:
        katoms.append((tuple(ModPoly(p, {((x, 1),): 1}) for x in digits), tuple(table)))
    return ModPoly(p)


def index_tuple(bits: Iterable[int], values: dict[int, int], base: int = 2) -> tuple[int, ...]:
    """Port values from a boundary digit assignment, ports in boundary order."""
    out: list[int] = []
    v = 0
    for k in bits:
        out.append(sum(values.get(v + j, 0) * base**j for j in range(k)))
        v += k
    return tuple(out)


@dataclass(frozen=True)
class PathSumComparison:
    """The path-sum rung's outcome: ``equal``, ``unequal`` with the differing boundary index
    and both entries, or ``unknown``, with a reason, and the parameter point it holds at."""

    status: str
    reason: str
    index: tuple[int, ...] | None = None
    left: Cyclo | None = None
    right: Cyclo | None = None
    point: tuple[tuple[str, int], ...] = ()
    residual_enum: bool = False


def compare_path_sums(
    left: Diagram, right: Diagram, max_residual_paths: int | None = None
) -> PathSumComparison:
    """Reduce both path sums and compare their canonical forms; with ``max_residual_paths``, two
    sums left with path variables are evaluated exactly at boundary points by
    :func:`_enumerate_residual` (binary) or :func:`_enumerate_residual_qudit` (Z_p), which only
    find an UNEQUAL; raises
    :class:`PathSumUnsupported` when either diagram is outside the class."""
    base = diagram_base(left)
    if diagram_base(right) != base:
        raise PathSumUnsupported("the two diagrams have different dimension bases")
    bits_l, bits_r = port_bits(left, base), port_bits(right, base)
    if bits_l != bits_r:
        raise PathSumUnsupported("the boundaries carry different layouts")
    if base != 2:
        return _compare_qudit(left, right, base, bits_l, max_residual_paths)
    raw_l, raw_r = from_diagram(left), from_diagram(right)
    assert isinstance(raw_l, PathSum) and isinstance(raw_r, PathSum)
    ps_l, ps_r = reduce(raw_l), reduce(raw_r)
    canon_l, canon_r = canonical(ps_l), canonical(ps_r)
    if canon_l is None or canon_r is None:
        left_n = len(ps_l.path)
        right_n = len(ps_r.path)
        if not ps_l.is_zero and not ps_r.is_zero:
            res_l, res_r = residual_form(ps_l), residual_form(ps_r)
            if res_l is not None and res_r is not None and res_l.same(res_r):
                return PathSumComparison(
                    "equal",
                    f"path sums reduce to one residual form up to renaming of {left_n} path "
                    f"variable(s) ({len(ps_l.trace)} and {len(ps_r.trace)} rule steps)",
                )
        stuck = (
            f"path_sum: {left_n} and {right_n} path variable(s) left after reduction"
            if left_n or right_n
            else "path_sum: a support constraint is not affine"
        )
        if max_residual_paths is not None:
            return _enumerate_residual(ps_l, ps_r, bits_l, max_residual_paths, stuck)
        return PathSumComparison("unknown", stuck)
    if canon_l.same(canon_r):
        return PathSumComparison(
            "equal",
            f"path sums reduce to one canonical form ({len(ps_l.trace)} and "
            f"{len(ps_r.trace)} rule steps)",
        )
    for point in witness_points(canon_l, canon_r):
        a, b = canon_l.value(point), canon_r.value(point)
        if a != b:
            return PathSumComparison(
                "unequal",
                "path sums reduce to different canonical forms",
                index_tuple(bits_l, point),
                a,
                b,
            )
    return PathSumComparison("unknown", "path_sum: the canonical forms differ at no tried point")


_RESIDUAL_POINTS = 64
"""The most boundary points :func:`_enumerate_residual` evaluates."""


def _enumerate_residual(
    ps_l: PathSum, ps_r: PathSum, bits: list[int], max_paths: int, stuck: str
) -> PathSumComparison:
    """Both entries evaluated exactly (each residual sum over its 2^k path assignments, k at
    most ``max_paths`` after the boundary is substituted) at boundary points in increasing
    Hamming weight, at most ``_RESIDUAL_POINTS``: ``unequal`` at the first that differ, else
    ``unknown``."""
    k = max(len(ps_l.path), len(ps_r.path))
    if k > max_paths:
        return PathSumComparison(
            "unknown",
            f"{stuck}; path_sum.residual_enum refused: {k} path variables over "
            f"max_residual_paths {max_paths}",
        )
    boundary = list(ps_l.boundary)
    by_weight = (
        ones for weight in range(len(boundary) + 1) for ones in combinations(boundary, weight)
    )
    order = list(islice(by_weight, _RESIDUAL_POINTS))
    for ones in order:
        point = {v: int(v in ones) for v in boundary}
        a = entry_enumerated(ps_l, point, max_paths)
        b = entry_enumerated(ps_r, point, max_paths)
        if a is None or b is None:
            continue
        if a != b:
            return PathSumComparison(
                "unequal",
                f"path sums differ at a boundary point, the residual sums evaluated exactly "
                f"(path_sum.residual_enum, {k} path variable(s) at most)",
                index_tuple(bits, point),
                a,
                b,
                residual_enum=True,
            )
    return PathSumComparison(
        "unknown",
        f"{stuck}; path_sum.residual_enum: equal at the {len(order)} boundary point(s) tried",
    )


def _enumerate_residual_qudit(
    ps_l: QuditPathSum, ps_r: QuditPathSum, bits: list[int], max_paths: int, stuck: str
) -> PathSumComparison:
    """:func:`_enumerate_residual` over Z_p: each residual sum over its p^k assignments, at
    boundary points in increasing Hamming weight (nonzero values 1..p-1), at most
    ``_RESIDUAL_POINTS``."""
    p = ps_l.p
    k = max(len(ps_l.path), len(ps_r.path))
    if k > max_paths:
        return PathSumComparison(
            "unknown",
            f"{stuck}; path_sum.residual_enum refused: {k} path variables over "
            f"max_residual_paths {max_paths}",
        )
    boundary = list(ps_l.boundary)
    points = (
        dict(zip(support, values, strict=True))
        for weight in range(len(boundary) + 1)
        for support in combinations(boundary, weight)
        for values in product(range(1, p), repeat=weight)
    )
    order = list(islice(points, _RESIDUAL_POINTS))
    for nonzero in order:
        point = {v: nonzero.get(v, 0) for v in boundary}
        a = entry_enumerated_qudit(ps_l, point, max_paths)
        b = entry_enumerated_qudit(ps_r, point, max_paths)
        if a is None or b is None:
            continue
        if a != b:
            return PathSumComparison(
                "unequal",
                f"path sums over Z_{p} differ at a boundary point, the residual sums evaluated "
                f"exactly (path_sum.residual_enum, {k} path variable(s) at most)",
                index_tuple(bits, point, p),
                a,
                b,
                residual_enum=True,
            )
    return PathSumComparison(
        "unknown",
        f"{stuck}; path_sum.residual_enum: equal at the {len(order)} boundary point(s) tried",
    )


def _compare_qudit(
    left: Diagram, right: Diagram, p: int, bits: list[int], max_residual_paths: int | None = None
) -> PathSumComparison:
    """compare_path_sums over Z_p: reduce, compare canonical forms, evaluate witness points;
    with ``max_residual_paths``, sums left with path variables go to
    :func:`_enumerate_residual_qudit`."""
    raw_l, raw_r = from_diagram(left), from_diagram(right)
    assert isinstance(raw_l, QuditPathSum) and isinstance(raw_r, QuditPathSum)
    ps_l, ps_r = reduce_qudit(raw_l), reduce_qudit(raw_r)
    canon_l, canon_r = canonical_qudit(ps_l), canonical_qudit(ps_r)
    if canon_l is None or canon_r is None:
        left_n, right_n = len(ps_l.path), len(ps_r.path)
        stuck = (
            f"path_sum: {left_n} and {right_n} path variable(s) left after reduction over Z_{p}"
            if left_n or right_n
            else f"path_sum: a support constraint over Z_{p} has no pivot"
        )
        if max_residual_paths is not None:
            return _enumerate_residual_qudit(ps_l, ps_r, bits, max_residual_paths, stuck)
        return PathSumComparison("unknown", stuck)
    if canon_l.same(canon_r):
        return PathSumComparison(
            "equal",
            f"path sums over Z_{p} reduce to one canonical form ({len(ps_l.trace)} and "
            f"{len(ps_r.trace)} rule steps)",
        )
    for point in witness_points_qudit(canon_l, canon_r):
        a, b = canon_l.value(point), canon_r.value(point)
        if a != b:
            return PathSumComparison(
                "unequal",
                f"path sums over Z_{p} reduce to different canonical forms",
                index_tuple(bits, point, p),
                a,
                b,
            )
    return PathSumComparison("unknown", "path_sum: the canonical forms differ at no tried point")


__all__ = [
    "PathSumComparison",
    "compare_path_sums",
    "constant",
    "diagram_base",
    "from_diagram",
    "index_tuple",
    "port_bits",
]
