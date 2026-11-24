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

"""Path sums of diagrams on legs of dimension 2, 3 and 6 = 3 x 2.

A leg of dimension 6 carries the digit pair (s, f) of its value ``2 s + f``, s a Z_3 digit and
f an F_2 digit; every binder and splitter on dimension 6 takes the 3-leg first, the 2-leg
second. The sum is built over F_2 digits and Z_3 digits: the F_3 phase polynomial holds the Z_3
spiders, X and F on dimension 3; a Z spider on dimension 6 adds to the F_2 phase its table read
as ``Sum_k I_k(s) (t(2k) + (t(2k+1) - t(2k)) f)``, each indicator ``I_k(s) = [s = k]`` an atom
variable ``u(s, k)`` for k = 1, 2 and ``I_0 = 1 - u(s, 1) - u(s, 2)``.

The Z_3 digits are then removed by two rules:

  [Elim3]  s linear in the F_3 phase, ``s L``, and in no indicator: the sum over s is
           ``3 [L = 0]``; a constant L is a factor 3 or 0, else one variable x of L with a unit
           coefficient is solved and substituted, its indicators read through the solution
           (``I_k(a z + b) = I_{(k - b)/a}(z)``, ``I_k(b) = [b = k]``).
  [Count3] s outside the F_3 phase: with no indicator the sum is a factor 3, else the sum over
           s of a function of ``(I_1(s), I_2(s))`` is ``Sum_{u, v in F_2} 0^(u v)`` of it, the
           pair ``(u, v)`` ranging over the three values ``(I_1, I_2)`` takes; ``0^(u v)`` is
           a zero amplitude atom.

What remains is a qubit :class:`PathSum` with the same boundary."""

from __future__ import annotations

from fractions import Fraction

from archytaszx.algebra.pathsum import (
    Cyclo,
    F2Poly,
    ModPoly,
    Mono,
    PathSum,
    PathSumUnsupported,
    PhasePoly,
    constant,
    sqrt_prime,
)
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    X_SPIDER,
    Z_SPIDER,
)
from archytaszx.diagram.graph import Diagram, Direction, PortRef

_KINDS = {2: ("2",), 3: ("3",), 6: ("3", "2")}


def _dims(diagram: Diagram) -> set[int] | None:
    """Every leg dimension, or None when one is symbolic."""
    out: set[int] = set()
    for node in diagram.nodes.values():
        for direction in (Direction.OUTPUT, Direction.INPUT):
            for leg in node.legs(direction):
                if not leg.dim.is_concrete:
                    return None
                out.add(leg.dim.to_int())
    return out


def is_mixed(diagram: Diagram) -> bool:
    """Whether every leg has dimension 1, 2, 3 or 6, with both a 2 and a 3 factor present."""
    if diagram.bang_boxes:
        return False
    dims = _dims(diagram)
    if dims is None or not dims <= {1, 2, 3, 6}:
        return False
    return bool(dims & {2, 6}) and bool(dims & {3, 6})


def is_mixed_pair(left: Diagram, right: Diagram) -> bool:
    """Whether one side is :func:`is_mixed` and both are box-free with every leg of dimension
    1, 2, 3 or 6."""
    if not (is_mixed(left) or is_mixed(right)):
        return False
    for diagram in (left, right):
        dims = _dims(diagram)
        if diagram.bang_boxes or dims is None or not dims <= {1, 2, 3, 6}:
            return False
    return True


def _dim(diagram: Diagram, ref: PortRef) -> int:
    return int(diagram.nodes[ref.node_id].legs(ref.direction)[ref.index].dim.to_int())


def _turns(node_id: object, node: object, d: int) -> list[Fraction]:
    from archytaszx.semantics.pathsum_diagram import _turns_table

    return _turns_table(node_id, node, d)  # type: ignore[arg-type]


def _expand(
    terms: dict[Mono, Fraction], var: int, value: dict[Mono, int], pairs: dict[int, int]
) -> dict[Mono, Fraction]:
    """``terms`` with ``var`` replaced by the integer combination ``value``; a monomial holding
    both indicators of one Z_3 digit (``pairs`` maps each to the other) is dropped."""
    out: dict[Mono, Fraction] = {}
    for mono, c in terms.items():
        if var not in mono:
            out[mono] = out.get(mono, Fraction(0)) + c
            continue
        rest = mono - {var}
        for vm, k in value.items():
            m = rest | vm
            if any(pairs.get(x) in m for x in m):
                continue
            out[m] = out.get(m, Fraction(0)) + c * k
    return out


class _Mixed:
    """The digits, phases and constant of one mixed diagram during the Z_3 elimination."""

    def __init__(self) -> None:
        self.next = 0
        self.ph2: dict[Mono, Fraction] = {}
        self.ph3 = ModPoly(3)
        self.const = Cyclo.rational(1)
        self.sqrt2 = 0
        self.path2: set[int] = set()
        self.path3: set[int] = set()
        self.ind: dict[tuple[int, int], int] = {}
        self.pairs: dict[int, int] = {}
        self.zeros: list[F2Poly] = []

    def fresh(self) -> int:
        self.next += 1
        return self.next - 1

    def add2(self, mono: Mono, c: Fraction) -> None:
        self.ph2[mono] = self.ph2.get(mono, Fraction(0)) + c

    def u(self, s: int, k: int) -> dict[Mono, int]:
        """``I_k(s)`` as an integer combination of the atom variables."""
        if (s, 1) not in self.ind:
            a, b = self.fresh(), self.fresh()
            self.ind[(s, 1)], self.ind[(s, 2)] = a, b
            self.pairs[a], self.pairs[b] = b, a
        if k:
            return {frozenset({self.ind[(s, k)]}): 1}
        return {
            frozenset(): 1,
            frozenset({self.ind[(s, 1)]}): -1,
            frozenset({self.ind[(s, 2)]}): -1,
        }

    def uses(self, s: int) -> bool:
        """Whether an F_2 phase term carries an indicator of ``s``."""
        if (s, 1) not in self.ind:
            return False
        a, b = self.ind[(s, 1)], self.ind[(s, 2)]
        return any(c % 1 and (a in m or b in m) for m, c in self.ph2.items())

    def read_through(self, x: int, value: ModPoly) -> None:
        """Replace the indicators of ``x`` by those of ``value`` (a constant or ``a z + b``)."""
        if (x, 1) not in self.ind:
            return
        b = value.terms.get((), 0)
        lin = [(m, c) for m, c in value.terms.items() if m]
        for k in (1, 2):
            if not lin:
                repl: dict[Mono, int] = {frozenset(): 1} if b == k else {}
            else:
                ((((z, _e),), a),) = lin
                repl = self.u(z, (k - b) * pow(a, -1, 3) % 3)
            self.ph2 = _expand(self.ph2, self.ind[(x, k)], repl, self.pairs)

    def elim3(self, s: int) -> bool:
        """[Elim3] on ``s``; False when it does not apply."""
        ph = self.ph3
        if ph.degree_in(s) != 1 or self.uses(s):
            return False
        lin = ph.coefficient(s, 1)
        if any(e > 1 or len(m) > 1 for m in lin.terms for _v, e in m):
            return False
        rest = ph.without(s)
        if not lin.variables():
            if lin.terms:
                self.const = Cyclo.rational(0)
            self.const = self.const.scale(3)
            self.ph3 = rest
            self.path3.discard(s)
            return True
        for x in sorted(lin.variables()):
            c = lin.coefficient(x, 1).terms.get((), 0)
            m = lin.copy()
            m.terms.pop(((x, 1),), None)
            if self.uses(x) and len(m.variables()) > 1:
                continue
            value = m.scale(-pow(c, -1, 3))
            self.read_through(x, value)
            self.ph3 = rest.substitute(x, value)
            self.const = self.const.scale(3)
            self.path3 -= {s, x}
            return True
        return False

    def count3(self, s: int) -> bool:
        """[Count3] on ``s``; False when it does not apply."""
        if s in self.ph3.variables():
            return False
        self.path3.discard(s)
        if not self.uses(s):
            self.const = self.const.scale(3)
            return True
        u, v = self.ind[(s, 1)], self.ind[(s, 2)]
        self.path2 |= {u, v}
        self.zeros.append(frozenset({frozenset({u, v})}))
        return True


def mixed_sum(diagram: Diagram) -> PathSum:
    """The qubit path sum of a mixed diagram (module docstring); raises
    :class:`PathSumUnsupported` outside it: a generator other than Z, X, F, B, S, a boundary leg
    not of dimension 2, X or F on dimension 6, a binder on dimension 6 other than (3, 2), a
    weighted phase, a Z_3 phase not a multiple of 1/3, or a Z_3 digit neither rule removes."""
    from archytaszx.semantics.pathsum_diagram import _legs, _phase_terms, _qudit_table, _Union

    allowed = {Z_SPIDER.name, X_SPIDER.name, FOURIER_BOX.name, DIM_BINDER.name, DIM_SPLITTER.name}
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        if node.generator_type.name not in allowed:
            raise PathSumUnsupported(f"node {node_id!r} is a {node.generator_type.name}")
        if not _legs(node_id, node):
            raise PathSumUnsupported(f"node {node_id!r} has no legs")
        if node.phase is not None and node.phase.is_weighted:
            raise PathSumUnsupported(f"node {node_id!r} carries weights")
    for ref in (*diagram.boundary_outputs, *diagram.boundary_inputs):
        if _dim(diagram, ref) != 2:
            raise PathSumUnsupported("a boundary leg of the mixed diagram is not of dimension 2")
    uf = _Union()
    for node_id in sorted(diagram.nodes):
        for ref, _sign in _legs(node_id, diagram.nodes[node_id]):
            for kind in _KINDS.get(_dim(diagram, ref), ()):
                uf.find(("p", ref.sort_key(), kind))
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if _dim(diagram, wire.a) != _dim(diagram, wire.b):
            raise PathSumUnsupported("a wire joins two dimensions")
        for kind in _KINDS.get(_dim(diagram, wire.a), ()):
            uf.union(("p", wire.a.sort_key(), kind), ("p", wire.b.sort_key(), kind))
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        name = node.generator_type.name
        legs = _legs(node_id, node)
        if name == Z_SPIDER.name:
            for ref, _sign in legs:
                for kind in _KINDS.get(_dim(diagram, ref), ()):
                    uf.union(("p", ref.sort_key(), kind), ("z", node_id, kind))
        elif name in (DIM_BINDER.name, DIM_SPLITTER.name):
            whole_dir, part_dir = (
                (Direction.OUTPUT, Direction.INPUT)
                if name == DIM_BINDER.name
                else (Direction.INPUT, Direction.OUTPUT)
            )
            whole = PortRef(node_id, whole_dir, 0)
            high, low = PortRef(node_id, part_dir, 0), PortRef(node_id, part_dir, 1)
            if (_dim(diagram, whole), _dim(diagram, high), _dim(diagram, low)) != (6, 3, 2):
                raise PathSumUnsupported(f"node {node_id!r} is not a (3, 2) {name} on 6")
            uf.union(("p", whole.sort_key(), "3"), ("p", high.sort_key(), "3"))
            uf.union(("p", whole.sort_key(), "2"), ("p", low.sort_key(), "2"))
        elif any(_dim(diagram, ref) == 6 for ref, _sign in legs):
            raise PathSumUnsupported(f"node {node_id!r} is a {name} on dimension 6")
    st = _Mixed()
    var_of: dict[object, int] = {}
    boundary: list[int] = []
    constraints: list[F2Poly] = []
    for ref in (*diagram.boundary_outputs, *diagram.boundary_inputs):
        v = st.fresh()
        boundary.append(v)
        root = uf.find(("p", ref.sort_key(), "2"))
        if root in var_of:
            constraints.append(frozenset({frozenset({var_of[root]}), frozenset({v})}))
        else:
            var_of[root] = v
    for root in sorted({uf.find(x) for x in list(uf.parent)}, key=repr):
        if root not in var_of:
            var_of[root] = st.fresh()
            (st.path3 if root[-1] == "3" else st.path2).add(var_of[root])

    def digit(ref: PortRef, kind: str) -> int:
        return var_of[uf.find(("p", ref.sort_key(), kind))]

    c, s2 = constant(diagram.scalar.to_sympy())
    st.const, st.sqrt2 = c, s2
    roots = 0
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        name = node.generator_type.name
        legs = _legs(node_id, node)
        d = _dim(diagram, legs[0][0])
        if name == Z_SPIDER.name:
            zs = {kind: var_of[uf.find(("z", node_id, kind))] for kind in _KINDS.get(d, ())}
            if d == 2:
                ph = PhasePoly()
                _phase_terms(ph, [zs["2"]], node_id, node)
                for m, t in ph.terms.items():
                    st.add2(m, t)
            elif d == 3:
                st.ph3 = st.ph3 + _qudit_table(node_id, node, [zs["3"]], 3)
            elif d == 6:
                table = _turns(node_id, node, 6)
                s, f = zs["3"], zs["2"]
                for k in range(3):
                    a, b = table[2 * k], table[2 * k + 1] - table[2 * k]
                    for m, cm in st.u(s, k).items():
                        if a:
                            st.add2(m, a * cm)
                        if b:
                            st.add2(m | {f}, b * cm)
        elif name in (X_SPIDER.name, FOURIER_BOX.name):
            if name == FOURIER_BOX.name:
                out = PortRef(node_id, Direction.OUTPUT, 0)
                inp = PortRef(node_id, Direction.INPUT, 0)
                if d == 2:
                    st.add2(frozenset({digit(out, "2"), digit(inp, "2")}), Fraction(1, 2))
                    st.sqrt2 -= 1
                else:
                    xo, xi = digit(out, "3"), digit(inp, "3")
                    if xo == xi:
                        raise PathSumUnsupported(f"node {node_id!r} is an F on a self-loop")
                    st.ph3.add(tuple(sorted(((xo, 1), (xi, 1)))), 1)
                    roots += 1
                continue
            y = st.fresh()
            if d == 2:
                st.path2.add(y)
                for ref, sign in legs:
                    st.add2(frozenset({y, digit(ref, "2")}), Fraction(sign, 2))
                st.sqrt2 -= len(legs)
                ph = PhasePoly()
                _phase_terms(ph, [y], node_id, node)
                for m, t in ph.terms.items():
                    st.add2(m, t)
            else:
                st.path3.add(y)
                for ref, sign in legs:
                    st.ph3.add(tuple(sorted(((y, 1), (digit(ref, "3"), 1)))), sign)
                st.ph3 = st.ph3 + _qudit_table(node_id, node, [y], 3)
                roots += len(legs)
    st.const = st.const.scale(Fraction(1, 3 ** (roots // 2)))
    if roots % 2:
        st.const = st.const * sqrt_prime(3).scale(Fraction(1, 3))
    while st.path3 and not st.const.is_zero:
        if not any(st.elim3(s) or st.count3(s) for s in sorted(st.path3)):
            raise PathSumUnsupported(
                f"Z_3 digit(s) {sorted(st.path3)} neither linear nor only in indicators"
            )
    if st.const.is_zero:
        return PathSum(tuple(boundary), set(), PhasePoly(), Cyclo.rational(0), 0, [], [], [])
    if st.ph3.variables():
        raise PathSumUnsupported("a Z_3 phase term outlives its digits")
    st.const = st.const * Cyclo.root(Fraction(st.ph3.terms.get((), 0), 3))
    turn = st.ph2.pop(frozenset(), Fraction(0)) % 1
    if turn:
        st.const = st.const * Cyclo.root(turn)
    phase = PhasePoly()
    for m, t in sorted(st.ph2.items(), key=lambda kv: (len(kv[0]), sorted(kv[0]))):
        phase.add(m, t)
    live = {v for m in phase.terms for v in m}
    stray = live - st.path2 - set(boundary)
    if stray:
        raise PathSumUnsupported(f"indicator variable(s) {sorted(stray)} outlive their digit")
    amps = [(Cyclo.rational(0), z) for z in st.zeros]
    return PathSum(tuple(boundary), st.path2, phase, st.const, st.sqrt2, constraints, [], amps)
