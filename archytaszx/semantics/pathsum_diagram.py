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

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from archytaszx.algebra.pathsum import (
    Cyclo,
    PathSum,
    PathSumUnsupported,
    PhasePoly,
    canonical,
    constant,
    reduce,
    witness_points,
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


def _bits(diagram: Diagram, ref: PortRef) -> int:
    """The number of digits of one port; raises unless its dimension is a concrete power of 2."""
    node = diagram.nodes[ref.node_id]
    dim = node.legs(ref.direction)[ref.index].dim
    if not dim.is_concrete:
        raise PathSumUnsupported(f"node {ref.node_id!r} has a symbolic dimension {dim}")
    d = dim.to_int()
    if d & (d - 1):
        raise PathSumUnsupported(f"node {ref.node_id!r} has dimension {d}, not a power of two")
    return d.bit_length() - 1


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


def _interpolate(phase: PhasePoly, digits: list[int], table: list[Fraction]) -> None:
    """Add the multilinear interpolation of ``table`` over ``digits`` (Moebius inversion)."""
    k = len(digits)
    if not any(table):
        return
    for size in range(k + 1):
        for subset in combinations(range(k), size):
            coeff = Fraction(0)
            for inner in range(size + 1):
                for sub in combinations(subset, inner):
                    coeff += (-1) ** (size - inner) * table[sum(1 << j for j in sub)]
            if coeff % 1:
                phase.add((digits[j] for j in subset), coeff)


def port_bits(diagram: Diagram) -> list[int]:
    """The digit count of each boundary port, outputs then inputs."""
    return [_bits(diagram, ref) for ref in (*diagram.boundary_outputs, *diagram.boundary_inputs)]


def from_diagram(diagram: Diagram) -> PathSum:
    """The path sum of ``diagram``; raises :class:`PathSumUnsupported` naming the first node
    outside the class: a generator other than Z, X, F, B and S, a dimension that is symbolic or
    not a power of two, a weighted or irrational phase, or a symbolic bang box."""
    if diagram.bang_boxes:
        diagram = expand_concrete_boxes(diagram)
        if diagram.bang_boxes:
            raise PathSumUnsupported("a bang box with a symbolic multiplicity")
    allowed = {Z_SPIDER.name, X_SPIDER.name, FOURIER_BOX.name, DIM_BINDER.name, DIM_SPLITTER.name}
    for node_id in sorted(diagram.nodes):
        name = diagram.nodes[node_id].generator_type.name
        if name not in allowed:
            raise PathSumUnsupported(f"node {node_id!r} is a {name}")
    uf = _Union()
    bits: dict[PortRef, int] = {}
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        for ref, _sign in _legs(node_id, node):
            bits[ref] = _bits(diagram, ref)
            for j in range(bits[ref]):
                uf.find(("p", ref.sort_key(), j))
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if bits[wire.a] != bits[wire.b]:
            raise PathSumUnsupported(f"a wire joins dimensions 2^{bits[wire.a]}, 2^{bits[wire.b]}")
        for j in range(bits[wire.a]):
            uf.union(("p", wire.a.sort_key(), j), ("p", wire.b.sort_key(), j))
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        name = node.generator_type.name
        if name == Z_SPIDER.name:
            legs = _legs(node_id, node)
            k = bits[legs[0][0]] if legs else _phase_bits(node_id, node)
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
    boundary: list[int] = []
    for ref in (*diagram.boundary_outputs, *diagram.boundary_inputs):
        for j in range(bits[ref]):
            v = len(boundary)
            boundary.append(v)
            root = uf.find(("p", ref.sort_key(), j))
            if root in var_of:
                constraints.append(frozenset({frozenset({var_of[root]}), frozenset({v})}))
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

    phase = PhasePoly()
    c, s2 = constant(diagram.scalar.to_sympy())
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        name = node.generator_type.name
        legs = _legs(node_id, node)
        if name == Z_SPIDER.name:
            k = bits[legs[0][0]] if legs else _phase_bits(node_id, node)
            digits = [var_of[uf.find(("z", node_id, j))] for j in range(k)]
            _interpolate(phase, digits, _turns_table(node_id, node, 1 << k))
        elif name == X_SPIDER.name:
            k = bits[legs[0][0]] if legs else _phase_bits(node_id, node)
            ys = [fresh() for _ in range(k)]
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
    return PathSum(tuple(boundary), path, phase, c, s2, constraints)


def _phase_bits(node_id: NodeId, node: Node) -> int:
    """The digit count of a legless spider, from its phase vector's dimension."""
    if node.phase is None:
        raise PathSumUnsupported(f"node {node_id!r} has no legs and no phase vector")
    dim = node.phase.dim
    if not dim.is_concrete or dim.to_int() & (dim.to_int() - 1):
        raise PathSumUnsupported(f"node {node_id!r} has dimension {dim}, not a power of two")
    return dim.to_int().bit_length() - 1


def index_tuple(bits: Iterable[int], values: dict[int, int]) -> tuple[int, ...]:
    """Port values from a boundary digit assignment, ports in boundary order."""
    out: list[int] = []
    v = 0
    for k in bits:
        out.append(sum(values.get(v + j, 0) << j for j in range(k)))
        v += k
    return tuple(out)


@dataclass(frozen=True)
class PathSumComparison:
    """The path-sum rung's outcome: ``equal``, ``unequal`` with the differing boundary index
    and both entries, or ``unknown``, with a reason."""

    status: str
    reason: str
    index: tuple[int, ...] | None = None
    left: Cyclo | None = None
    right: Cyclo | None = None


def compare_path_sums(left: Diagram, right: Diagram) -> PathSumComparison:
    """Reduce both path sums and compare their canonical forms; raises
    :class:`PathSumUnsupported` when either diagram is outside the class."""
    bits_l, bits_r = port_bits(left), port_bits(right)
    if bits_l != bits_r:
        raise PathSumUnsupported("the boundaries carry different layouts")
    ps_l, ps_r = reduce(from_diagram(left)), reduce(from_diagram(right))
    canon_l, canon_r = canonical(ps_l), canonical(ps_r)
    if canon_l is None or canon_r is None:
        left_n = len(ps_l.path)
        right_n = len(ps_r.path)
        return PathSumComparison(
            "unknown",
            f"path_sum: {left_n} and {right_n} path variable(s) left after reduction"
            if left_n or right_n
            else "path_sum: a support constraint is not affine",
        )
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


__all__ = [
    "PathSumComparison",
    "compare_path_sums",
    "constant",
    "from_diagram",
    "index_tuple",
    "port_bits",
]
