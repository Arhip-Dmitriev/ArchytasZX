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

"""The axioms of [P1] and [P2] as template rules, with the definitional and structural rules
that translate between their generators and the engine's.

Conventions (docs/COMPLETENESS.txt, section 2): H is F, H^dagger is F^3, D is F^2, a pink
K_j spider is an X spider with phase vector K_j in [P1]'s normalisation (``pink=True``), a
Z box is a Z spider whose parameter is a weighted :class:`~archytaszx.algebra.phase.PhaseVector`.
A W transpose is a W node whose legs are bent; a cap or cup is a wire.

Rules named ``zxw_*`` state an axiom of [P1], ``qf_*`` an axiom of [P2] that is not one of
[P1]'s, ``*_converse`` the same equation read right to left. :data:`AXIOM_RULES` maps every
name to its :class:`~archytaszx.rewrite.rule.Rule`; every one is ``automatic=False``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import replace

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import INDEX, Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.rewrite.match_template import (
    PHASELESS,
    Arities,
    Bindings,
    Condition,
    Derived,
    Fixed,
    Kj,
    Param,
    Side,
    Template,
    TemplateAxiom,
    TNode,
    TPort,
    Var,
    k_vector,
)
from archytaszx.rewrite.rule import RewriteDomainError

# -- vectors ------------------------------------------------------------------------------


def coefficients(pv: PhaseVector | None, dim: Dim) -> list[Scalar]:
    """``[a_0, ..., a_{d-1}]`` at a concrete dimension."""
    r = dim.to_int()
    if pv is None:
        return [Scalar.one()] * r
    return [pv.coefficient(k) for k in range(r)]


def from_coefficients(values: Sequence[Scalar], dim: Dim) -> PhaseVector | None:
    """The weight vector with coefficients ``values[1:]`` (``values[0]`` must be 1)."""
    weights = {k: values[k] for k in range(1, len(values))}
    vector = PhaseVector.from_weights(dim, weights)
    return None if vector.is_zero else vector


def zero_vector(dim: Dim) -> PhaseVector:
    """``(0, ..., 0)``: the Z box ``|0><0|``."""
    return PhaseVector.from_weights(dim, rule=Scalar.zero())


def minus_ones(dim: Dim) -> PhaseVector:
    """``(-1, ..., -1)``."""
    return PhaseVector.from_weights(dim, rule=Scalar.rational(-1))


def last_vector(x: Scalar, dim: Dim) -> PhaseVector:
    """``(0, ..., 0, x)``: the box labelled x."""
    if dim.is_concrete:
        r = dim.to_int()
        weights = {k: Scalar.zero() for k in range(1, r - 1)}
        weights[r - 1] = x
        return PhaseVector.from_weights(dim, weights)
    rule = x.to_sympy() * sp.KroneckerDelta(INDEX, dim.to_sympy() - 1)
    return PhaseVector.from_weights(dim, rule=Scalar(rule))


def unit_vector(index: sp.Expr | int, dim: Dim) -> PhaseVector:
    """The vector with a single 1 at ``index`` (mod d) and 0 elsewhere; zero at index 0."""
    if dim.is_concrete:
        r = dim.to_int()
        i = int(sp.Integer(index)) % r
        weights = {k: Scalar.zero() for k in range(1, r)}
        if i:
            weights[i] = Scalar.one()
        return PhaseVector.from_weights(dim, weights)
    rule = sp.KroneckerDelta(INDEX, sp.Integer(0) + index)
    return PhaseVector.from_weights(dim, rule=Scalar(rule))


def reversed_vector(pv: PhaseVector | None, dim: Dim) -> PhaseVector | None:
    """``<-a = (a_{d-1}, ..., a_1)``; concrete dimension."""
    if pv is None or pv.is_zero:
        return pv
    r = dim.to_int()
    if not pv.is_weighted:
        return PhaseVector(dim, {k: pv.get(r - k) for k in range(1, r)})
    values = coefficients(pv, dim)
    return from_coefficients([values[0]] + [values[r - k] for k in range(1, r)], dim)


def sum_vector(a: PhaseVector | None, b: PhaseVector | None, dim: Dim) -> PhaseVector:
    """``a + b`` entrywise on indices 1..d-1."""
    if dim.is_concrete:
        ca, cb = coefficients(a, dim), coefficients(b, dim)
        weights = {k: ca[k] + cb[k] for k in range(1, dim.to_int())}
        return PhaseVector.from_weights(dim, weights)
    va = a if a is not None else PhaseVector(dim)
    vb = b if b is not None else PhaseVector(dim)
    if va.entries() or vb.entries() or va.weights() or vb.weights():
        raise RewriteDomainError("a + b at a symbolic dimension needs rule-only vectors")
    rule = _rule_coefficient(va) + _rule_coefficient(vb)
    return PhaseVector.from_weights(dim, rule=Scalar(rule))


def _rule_coefficient(pv: PhaseVector) -> sp.Expr:
    weight = sp.Integer(1) if pv.weight_rule is None else pv.weight_rule.to_sympy()
    if pv.rule is None:
        return weight
    return weight * sp.exp(2 * sp.pi * sp.I * pv.rule.to_sympy_turns())


def shifted_vector(pv: PhaseVector | None, j: int, dim: Dim) -> PhaseVector | None:
    """``k_j(a) = (a_{1-j}/a_{-j}, ..., a_{d-1-j}/a_{-j})``; concrete dimension."""
    r = dim.to_int()
    values = coefficients(pv, dim)
    pivot = values[(-j) % r]
    if pivot.is_zero or sp.simplify(pivot.to_sympy()) == 0:
        raise RewriteDomainError("k_j(a) needs a_{-j} != 0")
    out = [
        Scalar(sp.nsimplify(values[(k - j) % r].to_sympy() / pivot.to_sympy(), rational=False))
        for k in range(r)
    ]
    return from_coefficients(out, dim)


def constant_vector(x: Scalar, dim: Dim) -> PhaseVector:
    """``(x, ..., x)``."""
    return PhaseVector.from_weights(dim, rule=x)


def vector_of(values: Sequence[Scalar], dim: Dim) -> PhaseVector | None:
    """``(values[0], ..., values[d-2])`` as a weight vector."""
    return from_coefficients([Scalar.one(), *values], dim)


def _coefficient(pv: object, index: int, dim: Dim) -> Scalar:
    if pv is None:
        return Scalar.one()
    if not isinstance(pv, PhaseVector):
        raise RewriteDomainError(f"expected a vector, got {pv!r}")
    return pv.coefficient(index % dim.to_int())


def _nonzero(value: Scalar) -> bool:
    return not value.is_zero and sp.simplify(value.to_sympy()) != 0


# -- parameters ----------------------------------------------------------------------------


class LastEntry(Param):
    """The box labelled x, ``(0, ..., 0, x)``; binds the Scalar x."""

    def __init__(self, name: str) -> None:
        self.name = name

    def match(self, phase: PhaseVector | None, dim: Dim, bindings: Bindings) -> Bindings | None:
        if not dim.is_concrete:
            return None
        r = dim.to_int()
        values = coefficients(phase, dim)
        if any(
            not values[k].is_zero and sp.simplify(values[k].to_sympy()) != 0
            for k in range(1, r - 1)
        ):
            return None
        x = values[r - 1]
        if self.name in bindings:
            known = bindings[self.name]
            assert isinstance(known, Scalar)
            return bindings if sp.simplify(known.to_sympy() - x.to_sympy()) == 0 else None
        return {**bindings, self.name: x}

    def build(self, dim: Dim, bindings: Bindings) -> PhaseVector | None:
        x = bindings[self.name]
        assert isinstance(x, Scalar)
        return last_vector(x, dim)

    def variables(self) -> frozenset[str]:
        return frozenset({self.name})

    def binds(self) -> frozenset[str]:
        return frozenset({self.name})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LastEntry) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("LastEntry", self.name))


def fixed(build: Callable[[Dim], PhaseVector | None]) -> Fixed:
    """A fixed parameter."""
    return Fixed(build)


ZERO = Fixed(zero_vector)
MINUS_ONES = Fixed(minus_ones)
ONE_LAST = Fixed(lambda dim: last_vector(Scalar.one(), dim))
"""The box labelled 1, ``(0, ..., 0, 1)``."""


def k_fixed(j: int) -> Fixed:
    """The fixed phase vector K_j."""
    return Fixed(lambda dim: k_vector(j, dim))


# -- template builder ----------------------------------------------------------------------


class TB:
    """Builds a :class:`Template` node by node."""

    def __init__(self) -> None:
        self.nodes: list[TNode] = []
        self.wires: list[tuple[TPort, TPort]] = []
        self.terminals: dict[str, TPort] = {}
        self.bare: list[tuple[str, str]] = []
        self.rest: dict[str, str] = {}
        self._count = 0

    def _key(self, key: str | None) -> str:
        if key is None:
            self._count += 1
            key = f"_{self._count}"
        return key

    def node(
        self,
        gen: GeneratorType,
        ins: int,
        outs: int,
        param: Param = PHASELESS,
        key: str | None = None,
        *,
        open: bool = False,
        pink: bool = False,
        dim: Dim | None = None,
    ) -> str:
        key = self._key(key)
        self.nodes.append(TNode(key, gen, ins, outs, param, open, pink, dim))
        return key

    def z(
        self,
        ins: int,
        outs: int,
        param: Param = PHASELESS,
        key: str | None = None,
        *,
        open: bool = False,
    ) -> str:
        return self.node(Z_SPIDER, ins, outs, param, key, open=open)

    def pink(self, ins: int, outs: int, param: Param = PHASELESS, key: str | None = None) -> str:
        return self.node(X_SPIDER, ins, outs, param, key, pink=True)

    def w(self, outs: int, key: str | None = None) -> str:
        return self.node(W_NODE, 1, outs, PHASELESS, key)

    def f(self, key: str | None = None) -> str:
        return self.node(FOURIER_BOX, 1, 1, PHASELESS, key)

    def conn(
        self,
        gen: GeneratorType,
        dims: tuple[tuple[int, ...], tuple[int, ...]],
        key: str | None = None,
    ) -> str:
        """A connective (B or S) with the given input and output dimensions."""
        key = self._key(key)
        port_dims = (tuple(_cdim(v) for v in dims[0]), tuple(_cdim(v) for v in dims[1]))
        self.nodes.append(
            TNode(key, gen, len(dims[0]), len(dims[1]), PHASELESS, port_dims=port_dims)
        )
        return key

    def wire(self, a: TPort, b: TPort) -> None:
        self.wires.append((a, b))

    def term(self, name: str, port: TPort) -> None:
        self.terminals[name] = port

    def chain(self, keys: Sequence[str]) -> None:
        """Wires each node's output 0 to the next node's input 0."""
        for a, b in itertools.pairwise(keys):
            self.wire(o(a), i(b))

    def fs(self, count: int) -> list[str]:
        """``count`` Fourier boxes in series."""
        keys = [self.f() for _ in range(count)]
        self.chain(keys)
        return keys

    def multiplier(self, m: int, *, up: bool = False) -> tuple[TPort, TPort]:
        """A multiplier labelled m (m >= 0 wires): (its input, its output).

        Down: plain green dot (1 -> m) into pink dot (m -> 1). Up (transpose): pink dot
        (1 -> m) into plain green dot (m -> 1).
        """
        if up:
            top = self.pink(1, m)
            bottom = self.z(m, 1)
        else:
            top = self.z(1, m)
            bottom = self.pink(m, 1)
        for k in range(m):
            self.wire(o(top, k), i(bottom, k))
        return i(top), o(bottom)

    def done(self, scalar: Callable[[Dim, Bindings], Scalar] | None = None) -> Template:
        return Template(
            tuple(self.nodes),
            tuple(self.wires),
            dict(self.terminals),
            tuple(self.bare),
            dict(self.rest),
            scalar,
        )


def _cdim(value: int) -> Dim:
    return Dim.concrete(value)


def i(key: str, index: int = 0) -> TPort:
    """Input ``index`` of template node ``key``."""
    return (key, Direction.INPUT, index)


def o(key: str, index: int = 0) -> TPort:
    """Output ``index`` of template node ``key``."""
    return (key, Direction.OUTPUT, index)


def fixed_side(build: Callable[[], Template]) -> Side:
    """A side whose template does not depend on the dimension or arities."""
    return Side(lambda _dim, _a: build())


def concrete_side(build: Callable[[int], Template]) -> Side:
    """A side defined only at a concrete dimension."""
    return Side(lambda dim, _a: build(dim.to_int()) if dim.is_concrete else None)


# -- arity enumerators ---------------------------------------------------------------------


def _partners(diagram: Diagram) -> dict[PortRef, PortRef]:
    out: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        out[wire.a] = wire.b
        out[wire.b] = wire.a
    return out


def _legs(diagram: Diagram, node_id: NodeId) -> list[PortRef]:
    node = diagram.nodes[node_id]
    return [PortRef(node_id, Direction.INPUT, k) for k in range(node.num_inputs)] + [
        PortRef(node_id, Direction.OUTPUT, k) for k in range(node.num_outputs)
    ]


def _neighbour_ids(diagram: Diagram, node_id: NodeId) -> list[NodeId]:
    partners = _partners(diagram)
    return [partners[ref].node_id for ref in _legs(diagram, node_id) if ref in partners]


def _count(diagram: Diagram, node_id: NodeId) -> int:
    return len(_legs(diagram, node_id))


def arity_from_anchor(
    build: Callable[[Diagram, NodeId], Iterable[Arities]],
) -> Callable[[Diagram, NodeId, Dim], Iterable[Arities]]:
    """An arity enumerator that ignores the dimension."""
    return lambda diagram, node_id, _dim: build(diagram, node_id)


# -- the axioms of [P1] --------------------------------------------------------------------


def _s1_left(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.z(0, a["k"], Var("a"), "a", open=True)
    t.z(a["k"], 0, Var("b"), "b", open=True)
    for k in range(a["k"]):
        t.wire(o("a", k), i("b", k))
    return t.done()


def _s1_right(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(0, 0, Derived(("a", "b"), lambda dim, b: _product(dim, b["a"], b["b"])), "c")
    t.rest = {"a": "c", "b": "c"}
    return t.done()


def _product(dim: Dim, a: object, b: object) -> PhaseVector:
    assert isinstance(a, PhaseVector) and isinstance(b, PhaseVector)
    return a + b


def _s1_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    if diagram.nodes[anchor].generator_type.name != Z_SPIDER.name:
        return
    seen: dict[NodeId, int] = {}
    for other in _neighbour_ids(diagram, anchor):
        if other != anchor and diagram.nodes[other].generator_type.name == Z_SPIDER.name:
            seen[other] = seen.get(other, 0) + 1
    for k in sorted(set(seen.values())):
        yield {"k": k}


ZXW_S1 = TemplateAxiom(
    "zxw_s1",
    Side(_s1_left, arity_from_anchor(_s1_arities)),
    Side(_s1_right),
    description="[P1] S1: Z boxes a and b joined by k >= 1 wires fuse into the Z box ab.",
    quantifiers=("k", "legs", "d"),
)


def _s2_left() -> Template:
    t = TB()
    t.z(1, 1, key="z")
    t.term("in", i("z"))
    t.term("out", o("z"))
    return t.done()


def _bare() -> Template:
    t = TB()
    t.bare = [("in", "out")]
    return t.done()


ZXW_S2 = TemplateAxiom(
    "zxw_s2",
    fixed_side(_s2_left),
    fixed_side(_bare),
    description="[P1] S2: the plain green dot with two legs is the identity.",
)


def _s2_pink_left() -> Template:
    t = TB()
    t.pink(1, 1, key="p")
    t.term("in", i("p"))
    t.term("out", o("p"))
    return t.done()


ZXW_S2_PINK = TemplateAxiom(
    "zxw_s2_pink",
    fixed_side(_s2_pink_left),
    fixed_side(_bare),
    description="[P1] S2 (second part), [P2] S2: the pink dot (1 -> 1) is the identity.",
)


def _d1_left(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(1, 1, Var("a"), "z")
    f = t.fs(2)
    t.wire(o("z"), i(f[0]))
    t.term("in", i("z"))
    t.term("out", o(f[1]))
    return t.done()


def _d1_right(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    f = t.fs(2)
    t.z(1, 1, Derived(("a",), lambda dim, b: reversed_vector(_vec(b["a"]), dim)), "z")
    t.wire(o(f[1]), i("z"))
    t.term("in", i(f[0]))
    t.term("out", o("z"))
    return t.done()


def _vec(value: object) -> PhaseVector | None:
    if value is None or isinstance(value, PhaseVector):
        return value
    raise RewriteDomainError(f"expected a vector, got {value!r}")


ZXW_D1 = TemplateAxiom(
    "zxw_d1",
    Side(_d1_left),
    Side(_d1_right),
    concrete_only=True,
    description="[P1] D1: Z box a followed by D is D followed by Z box <-a.",
)


def _ept_left() -> Template:
    t = TB()
    t.pink(0, 1, key="p")
    t.z(1, 0, Var("a"), "z")
    t.wire(o("p"), i("z"))
    return t.done()


def _empty() -> Template:
    return TB().done()


ZXW_EPT = TemplateAxiom(
    "zxw_ept",
    fixed_side(_ept_left),
    fixed_side(_empty),
    converse_free={"a": lambda dim, _b: PhaseVector(dim)},
    description="[P1] Ept: the pink dot state into a Z box effect is the empty diagram.",
)


def _b2_left() -> Template:
    t = TB()
    t.z(1, 2, key="g1")
    t.z(1, 2, key="g2")
    t.pink(2, 1, key="p1")
    t.pink(2, 1, key="p2")
    t.wire(o("g1", 0), i("p1", 0))
    t.wire(o("g1", 1), i("p2", 0))
    t.wire(o("g2", 0), i("p1", 1))
    t.wire(o("g2", 1), i("p2", 1))
    for name, port in (("a", i("g1")), ("b", i("g2")), ("c", o("p1")), ("e", o("p2"))):
        t.term(name, port)
    return t.done()


def _b2_right() -> Template:
    t = TB()
    t.pink(2, 1, key="p")
    t.z(1, 2, key="g")
    t.wire(o("p"), i("g"))
    for name, port in (("a", i("p", 0)), ("b", i("p", 1)), ("c", o("g", 0)), ("e", o("g", 1))):
        t.term(name, port)
    return t.done()


ZXW_B2 = TemplateAxiom(
    "zxw_b2",
    fixed_side(_b2_left),
    fixed_side(_b2_right),
    description="[P1] B2, [P2] B2: the bialgebra of the plain green and pink dots.",
)


def _k0_left() -> Template:
    t = TB()
    t.pink(0, 1, Kj("j"), "p")
    t.z(1, 2, key="g")
    t.wire(o("p"), i("g"))
    t.term("a", o("g", 0))
    t.term("b", o("g", 1))
    return t.done()


def _k0_right() -> Template:
    t = TB()
    t.pink(0, 1, Kj("j"), "p1")
    t.pink(0, 1, Kj("j"), "p2")
    t.term("a", o("p1"))
    t.term("b", o("p2"))
    return t.done()


ZXW_K0 = TemplateAxiom(
    "zxw_k0",
    fixed_side(_k0_left),
    fixed_side(_k0_right),
    description="[P1] K0: a pink K_j state copied by a plain green dot (1 -> 2).",
    quantifiers=("j", "d"),
)


def _k1_left(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.pink(1, 1, Kj("j"), "p")
    t.z(1, a["n"], key="g")
    t.wire(o("p"), i("g"))
    t.term("in", i("p"))
    for k in range(a["n"]):
        t.term(f"o{k}", o("g", k))
    return t.done()


def _k1_right(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.z(1, a["n"], key="g")
    t.term("in", i("g"))
    for k in range(a["n"]):
        q = t.pink(1, 1, Kj("j"), f"q{k}")
        t.wire(o("g", k), i(q))
        t.term(f"o{k}", o(q))
    return t.done()


def _k1_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != X_SPIDER.name or node.num_inputs != 1 or node.num_outputs != 1:
        return
    for n in sorted(
        {
            _count(diagram, other) - 1
            for other in _neighbour_ids(diagram, anchor)
            if diagram.nodes[other].generator_type.name == Z_SPIDER.name
        }
    ):
        yield {"n": n}


def _k1_right_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    if diagram.nodes[anchor].generator_type.name == Z_SPIDER.name:
        yield {"n": _count(diagram, anchor) - 1}


ZXW_K1 = TemplateAxiom(
    "zxw_k1",
    Side(_k1_left, arity_from_anchor(_k1_arities)),
    Side(_k1_right, arity_from_anchor(_k1_right_arities)),
    description="[P1] K1: pink K_j (1 -> 1) commutes through a plain green dot (1 -> n).",
    quantifiers=("n", "j", "d"),
)


def _k2_left(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.pink(1, 1, Kj("j"), "p")
    t.z(1, 1, Var("a"), "z")
    t.wire(o("p"), i("z"))
    t.term("in", i("p"))
    t.term("out", o("z"))
    return t.done()


def _k2_right(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(
        1,
        1,
        Derived(("a", "j"), lambda dim, b: shifted_vector(_vec(b["a"]), _int(b["j"]), dim)),
        "z",
    )
    t.pink(1, 1, Kj("j"), "p")
    t.wire(o("z"), i("p"))
    t.term("in", i("z"))
    t.term("out", o("p"))
    return t.done(lambda dim, b: _coefficient(b["a"], -_int(b["j"]), dim))


def _int(value: object) -> int:
    expr = sp.Integer(value) if isinstance(value, int) else value
    if not isinstance(expr, sp.Expr) or not expr.is_Integer:
        raise RewriteDomainError(f"expected a concrete integer, got {value!r}")
    return int(expr)


ZXW_K2 = TemplateAxiom(
    "zxw_k2",
    Side(_k2_left),
    Side(_k2_right),
    conditions=(
        Condition(
            "a_minus_j_nonzero",
            "a_{d-j} != 0, so k_j(a) is defined",
            lambda dim, b: _nonzero(_coefficient(b["a"], -_int(b["j"]), dim)),
        ),
    ),
    concrete_only=True,
    description="[P1] K2, [P2] K2: pink K_j through a Z box a, times the scalar a_{-j}.",
    quantifiers=("j", "d"),
)


def _zer_left() -> Template:
    t = TB()
    t.z(0, 1, ZERO, "z")
    t.term("out", o("z"))
    return t.done()


def _zer_right() -> Template:
    t = TB()
    t.pink(0, 1, key="p")
    t.term("out", o("p"))
    return t.done()


ZXW_ZER = TemplateAxiom(
    "zxw_zer",
    fixed_side(_zer_left),
    fixed_side(_zer_right),
    description="[P1] Zer: the Z box 0 state is the pink dot state.",
)


def _hh_left() -> Template:
    t = TB()
    f = t.fs(2)
    t.term("in", i(f[0]))
    t.term("out", o(f[1]))
    return t.done()


def _bent_pink() -> Template:
    t = TB()
    t.pink(0, 2, key="p")
    t.term("in", o("p", 0))
    t.term("out", o("p", 1))
    return t.done()


ZXW_P1 = TemplateAxiom(
    "zxw_p1",
    fixed_side(_hh_left),
    fixed_side(_bent_pink),
    description="[P1] P1 with (Du): H ; H is D, the pink dot (0 -> 2) with one leg bent.",
)


def _multiplier_side(dim: Dim, _a: Arities) -> Template | None:
    if not dim.is_concrete:
        return None
    t = TB()
    top, bottom = t.multiplier(dim.to_int() - 1)
    t.term("in", top)
    t.term("out", bottom)
    return t.done()


ZXW_P1_MULTIPLIER = TemplateAxiom(
    "zxw_p1_multiplier",
    Side(lambda _dim, _a: _hh_left()),
    Side(_multiplier_side),
    concrete_only=True,
    description="[P1] P1, [P2] P1: H ; H = D is the multiplier -1.",
)


def _h1_left(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.z(a["n"], a["m"], Kj("j"), "p")
    for k in range(a["n"]):
        f = t.fs(3)
        t.wire(o(f[2]), i("p", k))
        t.term(f"i{k}", i(f[0]))
    for k in range(a["m"]):
        f = t.fs(1)
        t.wire(o("p", k), i(f[0]))
        t.term(f"o{k}", o(f[0]))
    return t.done()


def _h1_right(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.z(a["n"], a["m"], Kj("j", sign=-1), "g")
    for k in range(a["n"]):
        f = t.fs(1)
        t.wire(o(f[0]), i("g", k))
        t.term(f"i{k}", i(f[0]))
    for k in range(a["m"]):
        f = t.fs(3)
        t.wire(o("g", k), i(f[0]))
        t.term(f"o{k}", o(f[2]))
    return t.done()


def _h1_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name == Z_SPIDER.name:
        legs = node.num_inputs + node.num_outputs
        for n in range(legs + 1):
            yield {"n": n, "m": legs - n}


def _h1_right_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name == Z_SPIDER.name:
        legs = node.num_inputs + node.num_outputs
        for n in range(legs + 1):
            yield {"n": n, "m": legs - n}


ZXW_H1 = TemplateAxiom(
    "zxw_h1",
    Side(_h1_left, arity_from_anchor(_h1_arities)),
    Side(_h1_right, arity_from_anchor(_h1_right_arities)),
    description=(
        "[P1] H1: the green circle spider K_j with H^dagger on its inputs and H on its "
        "outputs is the green circle spider K_{-j} with H on its inputs and H^dagger on its "
        "outputs."
    ),
    quantifiers=("n", "m", "j", "d"),
)


def _pcy_left() -> Template:
    t = TB()
    t.z(1, 1, Var("a"), "z")
    t.w(2, "w")
    t.wire(o("z"), i("w"))
    t.term("in", i("z"))
    t.term("a", o("w", 0))
    t.term("b", o("w", 1))
    return t.done()


def _pcy_right() -> Template:
    t = TB()
    t.w(2, "w")
    t.z(1, 1, Var("a"), "z1")
    t.z(1, 1, Var("a"), "z2")
    t.wire(o("w", 0), i("z1"))
    t.wire(o("w", 1), i("z2"))
    t.term("in", i("w"))
    t.term("a", o("z1"))
    t.term("b", o("z2"))
    return t.done()


ZXW_PCY = TemplateAxiom(
    "zxw_pcy",
    fixed_side(_pcy_left),
    fixed_side(_pcy_right),
    description="[P1] Pcy: a Z box a on the input of a W node is copied onto both outputs.",
)


def _ad_left() -> Template:
    t = TB()
    t.w(2, "w")
    t.z(1, 0, Var("a"), "za")
    t.z(1, 0, Var("b"), "zb")
    t.wire(o("w", 0), i("za"))
    t.wire(o("w", 1), i("zb"))
    t.term("in", i("w"))
    return t.done()


def _ad_right() -> Template:
    t = TB()
    t.z(1, 0, Derived(("a", "b"), lambda dim, b: sum_vector(_vec(b["a"]), _vec(b["b"]), dim)), "z")
    t.term("in", i("z"))
    return t.done()


ZXW_AD = TemplateAxiom(
    "zxw_ad",
    fixed_side(_ad_left),
    fixed_side(_ad_right),
    description="[P1] AD, [P2] AD: Z box effects a and b on a W node add.",
)


def _sym_left() -> Template:
    t = TB()
    t.w(2, "w")
    t.term("in", i("w"))
    t.term("a", o("w", 0))
    t.term("b", o("w", 1))
    return t.done()


def _sym_right() -> Template:
    t = TB()
    t.w(2, "w")
    t.term("in", i("w"))
    t.term("a", o("w", 1))
    t.term("b", o("w", 0))
    return t.done()


ZXW_SYM = TemplateAxiom(
    "zxw_sym",
    fixed_side(_sym_left),
    fixed_side(_sym_right),
    description="[P1] Sym: the two outputs of a W node may be swapped.",
)


def _aso_left() -> Template:
    t = TB()
    t.w(2, "w1")
    t.w(2, "w2")
    t.wire(o("w1", 0), i("w2"))
    t.term("in", i("w1"))
    t.term("a", o("w2", 0))
    t.term("b", o("w2", 1))
    t.term("c", o("w1", 1))
    return t.done()


def _aso_right() -> Template:
    t = TB()
    t.w(2, "w1")
    t.w(2, "w2")
    t.wire(o("w1", 1), i("w2"))
    t.term("in", i("w1"))
    t.term("a", o("w1", 0))
    t.term("b", o("w2", 0))
    t.term("c", o("w2", 1))
    return t.done()


ZXW_ASO = TemplateAxiom(
    "zxw_aso",
    fixed_side(_aso_left),
    fixed_side(_aso_right),
    description="[P1] Aso: the W node is associative.",
)


def _bzw_left() -> Template:
    t = TB()
    t.w(2, "w1")
    t.w(2, "w2")
    t.z(2, 1, key="g1")
    t.z(2, 1, key="g2")
    t.wire(o("w1", 0), i("g1", 0))
    t.wire(o("w1", 1), i("g2", 0))
    t.wire(o("w2", 0), i("g1", 1))
    t.wire(o("w2", 1), i("g2", 1))
    for name, port in (("a", i("w1")), ("b", i("w2")), ("c", o("g1")), ("e", o("g2"))):
        t.term(name, port)
    return t.done()


def _bzw_right() -> Template:
    t = TB()
    t.z(2, 1, key="g")
    t.w(2, "w")
    t.wire(o("g"), i("w"))
    for name, port in (("a", i("g", 0)), ("b", i("g", 1)), ("c", o("w", 0)), ("e", o("w", 1))):
        t.term(name, port)
    return t.done()


ZXW_BZW = TemplateAxiom(
    "zxw_bzw",
    fixed_side(_bzw_left),
    fixed_side(_bzw_right),
    description="[P1] BZW: the bialgebra of the W node and the plain green dot.",
)


def _ww_left(_dim: Dim, a: Arities) -> Template:
    n, m = a["n"], a["m"]
    t = TB()
    t.w(n, "t")
    t.w(m, "w")
    t.w(m, "p")
    t.wire(i("t"), i("w"))
    t.term("u", i("p"))
    for k in range(n):
        t.term(f"x{k}", o("t", k))
    for k in range(m):
        g = t.z(2, 1, key=f"g{k}")
        t.wire(o("p", k), i(g, 0))
        t.wire(o("w", k), i(g, 1))
        t.term(f"y{k}", o(g))
    return t.done()


def _ww_right(_dim: Dim, a: Arities) -> Template:
    n, m = a["n"], a["m"]
    t = TB()
    t.w(m, "p")
    t.term("u", i("p"))
    for k in range(n):
        t.w(m, f"w{k}")
        t.term(f"x{k}", i(f"w{k}"))
    for c in range(m):
        t.w(n, f"t{c}")
        for k in range(n):
            t.wire(o(f"t{c}", k), o(f"w{k}", c))
        g = t.z(2, 1, key=f"g{c}")
        t.wire(o("p", c), i(g, 0))
        t.wire(i(f"t{c}"), i(g, 1))
        t.term(f"y{c}", o(g))
    return t.done()


def _ww_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != W_NODE.name:
        return
    partner = _partners(diagram).get(PortRef(anchor, Direction.INPUT, 0))
    if partner is None or partner.direction is not Direction.INPUT:
        return
    other = diagram.nodes[partner.node_id]
    if other.generator_type.name == W_NODE.name:
        yield {"n": node.num_outputs, "m": other.num_outputs}


def _ww_right_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != W_NODE.name:
        return
    m = node.num_outputs
    for n in range(1, len(diagram.nodes) + 1):
        yield {"n": n, "m": m}


ZXW_WW = TemplateAxiom(
    "zxw_ww",
    Side(_ww_left, arity_from_anchor(_ww_arities)),
    Side(_ww_right, arity_from_anchor(_ww_right_arities)),
    description=(
        "[P1] WW: a W transpose (n inputs) into a W node (1 -> m), projected by a W node on "
        "plain green dots, is the crossed bialgebra of W nodes and W transposes."
    ),
    quantifiers=("n", "m", "d"),
)


def _bs0_left() -> Template:
    t = TB()
    t.pink(0, 1, key="p")
    t.w(2, "w")
    t.wire(o("p"), i("w"))
    t.term("a", o("w", 0))
    t.term("b", o("w", 1))
    return t.done()


def _bs0_right() -> Template:
    t = TB()
    t.pink(0, 1, key="p1")
    t.pink(0, 1, key="p2")
    t.term("a", o("p1"))
    t.term("b", o("p2"))
    return t.done()


ZXW_BS0 = TemplateAxiom(
    "zxw_bs0",
    fixed_side(_bs0_left),
    fixed_side(_bs0_right),
    description="[P1] Bs0: the pink dot state into a W node is two pink dot states.",
)


def _bsj_left() -> Template:
    t = TB()
    t.pink(0, 1, Kj("j"), "p")
    t.w(2, "w")
    t.z(1, 0, key="e")
    t.wire(o("p"), i("w"))
    t.wire(o("w", 0), i("e"))
    t.term("out", o("w", 1))
    return t.done()


def _bsj_right() -> Template:
    t = TB()
    t.z(0, 1, Derived(("j",), lambda dim, b: unit_vector(-_int_expr(b["j"]), dim)), "z")
    t.term("out", o("z"))
    return t.done()


def _int_expr(value: object) -> sp.Expr:
    if isinstance(value, int):
        return sp.Integer(value)
    assert isinstance(value, sp.Expr)
    return value


ZXW_BSJ = TemplateAxiom(
    "zxw_bsj",
    fixed_side(_bsj_left),
    fixed_side(_bsj_right),
    description=(
        "[P1] Bsj, [P2] Bsj: the pink K_j state into a W node with one output on a plain green "
        "effect is the Z box T_j state."
    ),
    quantifiers=("j", "d"),
)


def _ta_left(second: bool) -> Template:
    t = TB()
    t.w(2, "w")
    if second:
        t.w(2, "p")
    else:
        t.pink(1, 2, key="p")
    t.z(2, 1, key="g1")
    t.z(2, 1, key="g2")
    t.wire(o("w", 0), i("g1", 0))
    t.wire(o("w", 1), i("g2", 0))
    t.wire(o("p", 0), i("g1", 1))
    t.wire(o("p", 1), i("g2", 1))
    for name, port in (("a", i("w")), ("b", i("p")), ("c", o("g1")), ("e", o("g2"))):
        t.term(name, port)
    return t.done()


ZXW_TA = TemplateAxiom(
    "zxw_ta",
    fixed_side(lambda: _ta_left(False)),
    fixed_side(lambda: _ta_left(True)),
    description="[P1] TA: the trialgebra rule, the pink dot (1 -> 2) becomes a W node.",
)


def _v_box(t: TB, r: int) -> tuple[TPort, TPort]:
    """The V box: (its input, its output)."""
    box = t.z(1, 1, ONE_LAST)
    w = t.w(r - 1)
    t.wire(o(box), i(w))
    pink = t.pink(r - 1, 1)
    for k in range(1, r):
        top, bottom = t.multiplier((-k) % r)
        t.wire(o(w, k - 1), top)
        t.wire(bottom, i(pink, k - 1))
    return i(box), o(pink)


def _m_box(t: TB, r: int) -> tuple[TPort, TPort]:
    """The M box: (its input, its output)."""
    box = t.z(1, 1, ONE_LAST)
    w = t.w(r)
    t.wire(o(box), i(w))
    pink = t.pink(r, 1)
    labels = [(-k) % r for k in range(1, r)] + [0]
    for position, label in enumerate(labels):
        top, bottom = t.multiplier(label)
        t.wire(o(w, position), top)
        t.wire(bottom, i(pink, position))
    return i(box), o(pink)


def _zv_left(r: int) -> Template:
    t = TB()
    z = t.z(0, 1, Var("a"), "z")
    top, bottom = _v_box(t, r)
    t.wire(o(z), top)
    t.term("out", bottom)
    return t.done()


def _zv_right(r: int) -> Template:
    t = TB()
    t.z(
        0,
        1,
        Derived(("a",), lambda dim, b: constant_vector(_coefficient(b["a"], -1, dim), dim)),
        "z",
    )
    t.term("out", o("z"))
    return t.done()


ZXW_ZV = TemplateAxiom(
    "zxw_zv",
    concrete_side(_zv_left),
    concrete_side(_zv_right),
    concrete_only=True,
    description="[P1] ZV: a Z box a state into V is the Z box state (a_{d-1}, ..., a_{d-1}).",
)


def _va_left(r: int) -> Template:
    t = TB()
    t.w(r - 1, "w")
    t.pink(r - 1, 1, key="q")
    for k in range(1, r):
        box = t.z(1, 1, LastEntry(f"a{k}"), f"b{k}")
        t.wire(o("w", k - 1), i(box))
        top, bottom = t.multiplier((-k) % r)
        t.wire(o(box), top)
        t.wire(bottom, i("q", k - 1))
    t.term("in", i("w"))
    t.term("out", o("q"))
    return t.done()


def _va_right(r: int) -> Template:
    t = TB()
    top, bottom = _v_box(t, r)
    reads = tuple(f"a{k}" for k in range(1, r))
    z = t.z(
        1,
        1,
        Derived(reads, lambda dim, b: vector_of([_scalar(b[n]) for n in reads], dim)),
        "z",
    )
    t.wire(bottom, i(z))
    t.term("in", top)
    t.term("out", o(z))
    return t.done()


def _scalar(value: object) -> Scalar:
    assert isinstance(value, Scalar)
    return value


ZXW_VA = TemplateAxiom(
    "zxw_va",
    concrete_side(_va_left),
    concrete_side(_va_right),
    concrete_only=True,
    description=(
        "[P1] VA: d-1 boxes a_i on a W node through multipliers -i into a pink dot are V "
        "followed by the Z box a."
    ),
)


def _vw_left(r: int) -> Template:
    t = TB()
    top, bottom = _v_box(t, r)
    w = t.w(2, "w")
    t.wire(bottom, i(w))
    t.term("in", top)
    t.term("a", o(w, 0))
    t.term("b", o(w, 1))
    return t.done()


def _vw_right(r: int) -> Template:
    t = TB()
    w = t.w(2, "w")
    t.term("in", i(w))
    for name, k in (("a", 0), ("b", 1)):
        top, bottom = _v_box(t, r)
        t.wire(o(w, k), top)
        t.term(name, bottom)
    return t.done()


def _first_kind(gen: GeneratorType) -> Callable[[Diagram, NodeId], Iterator[Arities]]:
    def enumerate_(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
        if diagram.nodes[anchor].generator_type.name == gen.name:
            yield {}

    return enumerate_


ZXW_VW = TemplateAxiom(
    "zxw_vw",
    concrete_side(_vw_left),
    concrete_side(_vw_right),
    concrete_only=True,
    description="[P1] VW: V into a W node is the W node with V on both outputs.",
)


def _kz_left(dim: Dim, a: Arities) -> Template:
    es = [a[f"e{k}"] for k in range(a["n"])]
    n = len(es)
    t = TB()
    t.w(n + 1, "w")
    t.pink(0, 1, k_fixed(1), "p")
    t.pink(0, n, key="q")
    t.wire(o("p"), i("w"))
    t.term("o0", o("w", 0))
    for k, e in enumerate(es):
        g = t.z(2, 1, key=f"g{k}")
        t.wire(o("w", k + 1), i(g, 0))
        top, bottom = t.multiplier(e)
        t.wire(o("q", k), top)
        t.wire(bottom, i(g, 1))
        t.term(f"o{k + 1}", o(g))
    return t.done()


def _kz_right(dim: Dim, a: Arities) -> Template:
    t = TB()
    t.pink(0, 1, k_fixed(1), "p")
    t.term("o0", o("p"))
    for k in range(a["n"]):
        q = t.pink(0, 1, key=f"q{k}")
        t.term(f"o{k + 1}", o(q))
    return t.done()


def _kz_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != W_NODE.name or node.num_outputs < 2:
        return
    n = node.num_outputs - 1
    labels: set[int] = set()
    for green in _neighbour_ids(diagram, anchor):
        if diagram.nodes[green].generator_type.name != Z_SPIDER.name:
            continue
        for pink in _neighbour_ids(diagram, green):
            if diagram.nodes[pink].generator_type.name == X_SPIDER.name:
                labels.add(diagram.nodes[pink].num_inputs)
    bound = dim.to_int() - 1 if dim.is_concrete else None
    usable = sorted(e for e in labels if e >= 1 and (bound is None or e <= bound))
    for es in itertools.combinations_with_replacement(usable, n):
        yield {"n": n, **{f"e{k}": e for k, e in enumerate(es)}}


def _kz_right_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if not dim.is_concrete or node.generator_type.name != X_SPIDER.name or node.num_inputs:
        return
    states = sum(
        1
        for other in diagram.nodes.values()
        if other.generator_type.name == X_SPIDER.name
        and not other.num_inputs
        and other.num_outputs == 1
        and (other.phase is None or other.phase.is_zero)
    )
    for n in range(1, states + 1):
        for es in itertools.combinations_with_replacement(range(1, dim.to_int()), n):
            yield {"n": n, **{f"e{k}": e for k, e in enumerate(es)}}


ZXW_KZ = TemplateAxiom(
    "zxw_kz",
    Side(_kz_left, _kz_arities),
    Side(_kz_right, _kz_right_arities),
    concrete_only=True,
    description=(
        "[P1] KZ: a pink K_1 state on a W node whose other outputs meet a pink dot through "
        "multipliers e_1..e_n is a pink K_1 state and n pink dot states."
    ),
    quantifiers=("n", "e_1..e_n", "d"),
)


def _hd_left(r: int) -> Template:
    t = TB()
    t.pink(1, r, key="top")
    t.term("in", i("top"))
    t.pink(0, 1, k_fixed(1), "k1")
    t.w(r, "w")
    t.wire(o("k1"), i("w"))
    t.pink(r, 1, key="bottom")
    t.term("out", o("bottom"))
    for c in range(r):
        top, bottom = t.multiplier((-c) % r, up=True)
        t.wire(o("top", c), top)
        g = t.z(3, 0, key=f"c{c}")
        t.wire(bottom, i(g, 0))
        t.wire(o("w", c), i(g, 1))
        m_in, m_out = _m_box(t, r)
        t.wire(i(g, 2), m_in)
        kr = t.z(1, 1, k_fixed(c), f"K{c}")
        t.wire(m_out, i(kr))
        t.wire(o(kr), i("bottom", c))
    return t.done(lambda dim, _b: Scalar.dim_power(dim, -1, 2))


def _hd_right(_r: int) -> Template:
    t = TB()
    f = t.f("h")
    t.term("in", i(f))
    t.term("out", o(f))
    return t.done()


ZXW_HD = TemplateAxiom(
    "zxw_hd",
    concrete_side(_hd_left),
    concrete_side(_hd_right),
    concrete_only=True,
    description="[P1] HD: the decomposition of the Hadamard box.",
)


# -- [P2]: rules not among [P1]'s ------------------------------------------------------------


def _wf_left(_dim: Dim, a: Arities) -> Template:
    p, q, r = a["p"], a["q"], a["r"]
    t = TB()
    t.w(p + 1 + q, "w1")
    t.w(r, "w2")
    t.wire(o("w1", p), i("w2"))
    t.term("in", i("w1"))
    names = [f"o{k}" for k in range(p + r + q)]
    ports = [o("w1", k) for k in range(p)] + [o("w2", k) for k in range(r)]
    ports += [o("w1", p + 1 + k) for k in range(q)]
    for name, port in zip(names, ports, strict=True):
        t.term(name, port)
    return t.done()


def _wf_right(_dim: Dim, a: Arities) -> Template:
    total = a["p"] + a["q"] + a["r"]
    t = TB()
    t.w(total, "w")
    t.term("in", i("w"))
    for k in range(total):
        t.term(f"o{k}", o("w", k))
    return t.done()


def _wf_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != W_NODE.name or node.num_outputs < 1:
        return
    partners = _partners(diagram)
    rs = set()
    for k in range(node.num_outputs):
        far = partners.get(PortRef(anchor, Direction.OUTPUT, k))
        if far is not None and far.direction is Direction.INPUT:
            other = diagram.nodes[far.node_id]
            if other.generator_type.name == W_NODE.name and far.node_id != anchor:
                rs.add(other.num_outputs)
    for r in sorted(rs):
        yield {"p": 0, "q": node.num_outputs - 1, "r": r}


def _wf_right_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != W_NODE.name:
        return
    total = node.num_outputs
    for r in range(total + 1):
        yield {"p": 0, "q": total - r, "r": r}


QF_WF = TemplateAxiom(
    "qf_wf",
    Side(_wf_left, arity_from_anchor(_wf_arities)),
    Side(_wf_right, arity_from_anchor(_wf_right_arities)),
    description="[P2] WF: a W node fed by an output of another is one W node.",
    quantifiers=("p", "q", "r", "d"),
)


def _w_swap_side(swapped: bool) -> Callable[[Dim, Arities], Template]:
    def build(_dim: Dim, a: Arities) -> Template:
        n, k = a["n"], a["k"]
        t = TB()
        t.w(n, "w")
        t.term("in", i("w"))
        order = list(range(n))
        if swapped:
            order[k], order[k + 1] = order[k + 1], order[k]
        for position, index in enumerate(order):
            t.term(f"o{position}", o("w", index))
        return t.done()

    return build


def _w_swap_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name == W_NODE.name:
        for k in range(node.num_outputs - 1):
            yield {"n": node.num_outputs, "k": k}


QF_W_SYM = TemplateAxiom(
    "qf_w_sym",
    Side(_w_swap_side(False), arity_from_anchor(_w_swap_arities)),
    Side(_w_swap_side(True), arity_from_anchor(_w_swap_arities)),
    description="[P2] (2.10): outputs k and k+1 of a W node with n outputs may be swapped.",
    quantifiers=("n", "k", "d"),
)


def _flex_side(bent: bool) -> Callable[[Dim, Arities], Template]:
    def build(_dim: Dim, a: Arities) -> Template:
        m, n = a["m"], a["n"]
        t = TB()
        if bent:
            t.z(m - 1, n + 1, Var("a"), "z")
            for k in range(m - 1):
                t.term(f"i{k}", i("z", k))
            t.term(f"i{m - 1}", o("z", n))
        else:
            t.z(m, n, Var("a"), "z")
            for k in range(m):
                t.term(f"i{k}", i("z", k))
        for k in range(n):
            t.term(f"o{k}", o("z", k))
        return t.done()

    return build


def _flex_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name == Z_SPIDER.name and node.num_inputs >= 1:
        yield {"m": node.num_inputs, "n": node.num_outputs}


def _flex_right_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name == Z_SPIDER.name and node.num_outputs >= 1:
        yield {"m": node.num_inputs + 1, "n": node.num_outputs - 1}


QF_Z_FLEX = TemplateAxiom(
    "qf_z_flex",
    Side(_flex_side(False), arity_from_anchor(_flex_arities)),
    Side(_flex_side(True), arity_from_anchor(_flex_right_arities)),
    description="[P2] (2.9): a Z box's input leg bends to an output leg.",
    quantifiers=("m", "n", "d"),
)


def _mw_left(_dim: Dim, a: Arities) -> Template:
    t = TB()
    box = t.z(1, 1, ONE_LAST, "box")
    top, bottom = t.multiplier(a["e"])
    w = t.w(2, "w")
    t.wire(o(box), top)
    t.wire(bottom, i(w))
    t.term("in", i(box))
    t.term("a", o(w, 0))
    t.term("b", o(w, 1))
    return t.done()


def _mw_right(_dim: Dim, a: Arities) -> Template:
    t = TB()
    box = t.z(1, 1, ONE_LAST, "box")
    w = t.w(2, "w")
    t.wire(o(box), i(w))
    t.term("in", i(box))
    for name, k in (("a", 0), ("b", 1)):
        top, bottom = t.multiplier(a["e"])
        t.wire(o(w, k), top)
        t.term(name, bottom)
    return t.done()


def _mw_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    if diagram.nodes[anchor].generator_type.name != Z_SPIDER.name:
        return
    labels = {
        diagram.nodes[x].num_outputs
        for x in _neighbour_ids(diagram, anchor)
        if diagram.nodes[x].generator_type.name == Z_SPIDER.name
    }
    for e in sorted(labels):
        if not dim.is_concrete or e % dim.to_int() != 0:
            yield {"e": e}


def _mw_right_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    if diagram.nodes[anchor].generator_type.name != Z_SPIDER.name:
        return
    labels: set[int] = set()
    for w in _neighbour_ids(diagram, anchor):
        if diagram.nodes[w].generator_type.name != W_NODE.name:
            continue
        for green in _neighbour_ids(diagram, w):
            if diagram.nodes[green].generator_type.name == Z_SPIDER.name and green != anchor:
                labels.add(diagram.nodes[green].num_outputs)
    for e in sorted(labels):
        if not dim.is_concrete or e % dim.to_int() != 0:
            yield {"e": e}


QF_MW = TemplateAxiom(
    "qf_mw",
    Side(_mw_left, _mw_arities),
    Side(_mw_right, _mw_right_arities),
    description="[P2] MW: a multiplier e (e != 0 mod d) through the box 1 and a W node.",
    quantifiers=("e", "d"),
)


def _d1g_left(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.z(a["n"], a["m"], Var("a"), "z")
    for k in range(a["n"]):
        f = t.fs(2)
        t.wire(o(f[1]), i("z", k))
        t.term(f"i{k}", i(f[0]))
    for k in range(a["m"]):
        t.term(f"o{k}", o("z", k))
    return t.done()


def _d1g_right(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.z(a["n"], a["m"], Derived(("a",), lambda dim, b: reversed_vector(_vec(b["a"]), dim)), "z")
    for k in range(a["n"]):
        t.term(f"i{k}", i("z", k))
    for k in range(a["m"]):
        f = t.fs(2)
        t.wire(o("z", k), i(f[0]))
        t.term(f"o{k}", o(f[1]))
    return t.done()


def _z_split_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name == Z_SPIDER.name:
        legs = node.num_inputs + node.num_outputs
        for n in range(1, legs + 1):
            yield {"n": n, "m": legs - n}


QF_D1 = TemplateAxiom(
    "qf_d1",
    Side(_d1g_left, arity_from_anchor(_z_split_arities)),
    Side(_d1g_right, arity_from_anchor(_z_split_arities)),
    concrete_only=True,
    description="[P2] D1: D on every input of a Z box a is H ; H on every output of <-a.",
    quantifiers=("n", "m", "d"),
)


def _ww2_left(_dim: Dim, a: Arities) -> Template:
    m, n = a["m"], a["n"]
    t = TB()
    t.w(m, "t")
    t.w(n, "w")
    t.wire(i("t"), i("w"))
    for k in range(m):
        t.term(f"x{k}", o("t", k))
    for k in range(n):
        t.term(f"y{k}", o("w", k))
    return t.done()


def _ww2_right(_dim: Dim, a: Arities) -> Template:
    m, n = a["m"], a["n"]
    t = TB()
    for k in range(m):
        t.w(n, f"u{k}")
        t.term(f"x{k}", i(f"u{k}"))
    t.w(n, "s")
    e = t.z(1, 0, key="e")
    t.wire(i("s"), i(e))
    for c in range(n):
        t.w(m, f"v{c}")
        for k in range(m):
            t.wire(o(f"v{c}", k), o(f"u{k}", c))
        g = t.z(1, 2, key=f"g{c}")
        t.wire(i(f"v{c}"), i(g))
        t.wire(o(g, 1), o("s", c))
        t.term(f"y{c}", o(g, 0))
    return t.done()


def _ww2_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != W_NODE.name:
        return
    partner = _partners(diagram).get(PortRef(anchor, Direction.INPUT, 0))
    if partner is None or partner.direction is not Direction.INPUT:
        return
    other = diagram.nodes[partner.node_id]
    if other.generator_type.name == W_NODE.name and partner.node_id != anchor:
        yield {"m": node.num_outputs, "n": other.num_outputs}


def _ww2_right_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != W_NODE.name:
        return
    partners = _partners(diagram)
    ms = set()
    for k in range(node.num_outputs):
        far = partners.get(PortRef(anchor, Direction.OUTPUT, k))
        if far is not None and far.direction is Direction.OUTPUT:
            other = diagram.nodes[far.node_id]
            if other.generator_type.name == W_NODE.name:
                ms.add(other.num_outputs)
    for m in sorted(ms):
        yield {"m": m, "n": node.num_outputs}


QF_WW = TemplateAxiom(
    "qf_ww",
    Side(_ww2_left, arity_from_anchor(_ww2_arities)),
    Side(_ww2_right, arity_from_anchor(_ww2_right_arities)),
    description=(
        "[P2] WW: a W transpose (m inputs) into a W node (1 -> n) is m W nodes crossed into n "
        "W transposes, projected through plain green dots and a W transpose on a green effect."
    ),
    quantifiers=("m", "n", "d"),
)


def _va2_left(r: int) -> Template:
    t = TB()
    t.w(r, "w")
    t.pink(r, 1, key="q")
    box = t.z(1, 1, ONE_LAST, "b0")
    t.wire(o("w", 0), i(box))
    top, bottom = t.multiplier(0)
    t.wire(o(box), top)
    t.wire(bottom, i("q", 0))
    for k in range(1, r):
        box = t.z(1, 1, LastEntry(f"a{k}"), f"b{k}")
        t.wire(o("w", k), i(box))
        top, bottom = t.multiplier((-k) % r)
        t.wire(o(box), top)
        t.wire(bottom, i("q", k))
    t.term("in", i("w"))
    t.term("out", o("q"))
    return t.done()


def _va2_right(r: int) -> Template:
    t = TB()
    box = t.z(1, 1, ONE_LAST, "b0")
    km = t.pink(1, 1, k_fixed(-1), "km")
    wt = t.w(2, "wt")
    green = t.z(0, 1, key="g")
    kp = t.pink(1, 1, k_fixed(1), "kp")
    reads = tuple(f"a{k}" for k in range(1, r))
    z = t.z(
        1,
        1,
        Derived(reads, lambda dim, b: vector_of([_scalar(b[n]) for n in reads], dim)),
        "z",
    )
    t.wire(o(box), i(km))
    t.wire(o(km), o(wt, 0))
    t.wire(o(green), o(wt, 1))
    t.wire(i(wt), i(kp))
    t.wire(o(kp), i(z))
    t.term("in", i(box))
    t.term("out", o(z))
    return t.done()


QF_VA = TemplateAxiom(
    "qf_va",
    concrete_side(_va2_left),
    concrete_side(_va2_right),
    concrete_only=True,
    description=(
        "[P2] VA: boxes 1, a_1..a_{d-1} on a W node through multipliers 0, -1..-(d-1) into "
        "a pink dot are the box 1, pink K_{-1}, a W transpose with a plain green state, "
        "pink K_1 and the Z box a in series."
    ),
)


def _hd2_left(r: int) -> Template:
    t = TB()
    t.pink(1, r, key="top")
    t.term("in", i("top"))
    t.pink(0, 1, k_fixed(1), "k1")
    t.w(r, "w")
    t.wire(o("k1"), i("w"))
    t.pink(r, 1, key="bottom")
    t.term("out", o("bottom"))
    for c in range(r):
        top, bottom = t.multiplier((-c) % r, up=True)
        t.wire(o("top", c), top)
        g = t.z(2, 1, key=f"c{c}")
        t.wire(bottom, i(g, 0))
        t.wire(o("w", c), i(g, 1))
        km = t.pink(1, 1, k_fixed(-1), f"km{c}")
        t.wire(o(g), i(km))
        wt = t.w(2, f"wt{c}")
        green = t.z(0, 1, key=f"s{c}")
        t.wire(o(km), o(wt, 1))
        t.wire(o(green), o(wt, 0))
        kr = t.z(1, 1, k_fixed(c), f"K{c}")
        t.wire(i(wt), i(kr))
        t.wire(o(kr), i("bottom", c))
    return t.done(
        lambda dim, _b: Scalar(sp.Integer(-1) ** (dim.to_sympy() + 1) / sp.sqrt(dim.to_sympy()))
    )


QF_HD = TemplateAxiom(
    "qf_hd",
    concrete_side(_hd2_left),
    concrete_side(_hd_right),
    concrete_only=True,
    description="[P2] HD: the decomposition of the Hadamard box.",
)


# -- definitional and structural rules -----------------------------------------------------


def _x_def_left(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.node(X_SPIDER, a["n"], a["m"], Var("a"), "x")
    for k in range(a["n"]):
        t.term(f"i{k}", i("x", k))
    for k in range(a["m"]):
        t.term(f"o{k}", o("x", k))
    return t.done()


def _x_def_right(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.z(a["n"], a["m"], Var("a"), "z")
    for k in range(a["n"]):
        f = t.fs(3)
        t.wire(o(f[2]), i("z", k))
        t.term(f"i{k}", i(f[0]))
    for k in range(a["m"]):
        f = t.fs(1)
        t.wire(o("z", k), i(f[0]))
        t.term(f"o{k}", o(f[0]))
    return t.done()


def _x_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name == X_SPIDER.name:
        yield {"n": node.num_inputs, "m": node.num_outputs}


X_SPIDER_DEFINITION = TemplateAxiom(
    "x_spider_definition",
    Side(_x_def_left, arity_from_anchor(_x_arities)),
    Side(_x_def_right, arity_from_anchor(_h1_right_arities)),
    description="The X spider is the Z spider with F on each output and F^3 on each input.",
    quantifiers=("n", "m", "d"),
)


def _triangle_left(gen: GeneratorType) -> Template:
    t = TB()
    t.node(gen, 1, 1, PHASELESS, "t")
    t.term("in", i("t"))
    t.term("out", o("t"))
    return t.done()


def _triangle_right(param: Param) -> Template:
    t = TB()
    t.w(2, "w")
    t.z(1, 0, param, "e")
    t.wire(o("w", 0), i("e"))
    t.term("in", i("w"))
    t.term("out", o("w", 1))
    return t.done()


TRIANGLE_DEFINITION = TemplateAxiom(
    "triangle_definition",
    fixed_side(lambda: _triangle_left(TRIANGLE)),
    fixed_side(lambda: _triangle_right(PHASELESS)),
    description="[P1] (YT): the triangle is a W node with one output on a plain green effect.",
)

TRIANGLE_INVERSE_DEFINITION = TemplateAxiom(
    "triangle_inverse_definition",
    fixed_side(lambda: _triangle_left(TRIANGLE_INVERSE)),
    fixed_side(lambda: _triangle_right(MINUS_ONES)),
    description="[P1] (YT): the inverse triangle is a W node with one output on the box -1.",
)


def _phase_weight_left(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(0, 0, Var("a"), "z", open=True)
    return t.done()


def _phase_weight_right(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(0, 0, Derived(("a",), lambda dim, b: as_weights(_vec(b["a"]), dim)), "z")
    t.rest = {"z": "z"}
    return t.done()


def as_weights(pv: PhaseVector | None, dim: Dim) -> PhaseVector:
    """``pv`` with every phase moved into the weights: the same coefficients."""
    if pv is None:
        raise RewriteDomainError("a phaseless spider has no phase to move")
    if not pv.entries() and pv.rule is None:
        raise RewriteDomainError("the vector carries no phase")
    if dim.is_concrete:
        return PhaseVector.from_weights(dim, {k: pv.coefficient(k) for k in range(1, dim.to_int())})
    if pv.entries() or pv.weights():
        raise RewriteDomainError("moving explicit entries needs a concrete dimension")
    return PhaseVector.from_weights(dim, rule=Scalar(_rule_coefficient(pv)))


def _has_phase(dim: Dim, b: Bindings) -> bool:
    pv = b["a"]
    return isinstance(pv, PhaseVector) and (bool(pv.entries()) or pv.rule is not None)


PHASE_TO_WEIGHT = TemplateAxiom(
    "phase_to_weight",
    Side(_phase_weight_left, arity_from_anchor(_first_kind(Z_SPIDER))),
    Side(_phase_weight_right),
    conditions=(Condition("carries_a_phase", "the vector has a phase entry or rule", _has_phase),),
    description="[P1] green circle spider: the phase alpha is the weight e^{i alpha}.",
)


def _scalar_box_left(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(0, 0, Var("a"), "z")
    return t.done()


def _box_value(dim: Dim, b: Bindings) -> Scalar:
    pv = _vec(b["a"])
    if dim.is_concrete:
        total = Scalar.zero()
        for value in coefficients(pv, dim):
            total = total + value
        return total
    if pv is None:
        return Scalar.from_dim(dim)
    if pv.entries() or pv.weights():
        raise RewriteDomainError("a symbolic sum over explicit entries")
    expr = _rule_coefficient(pv)
    return (
        Scalar.one()
        + Scalar.index_sum(dim, lambda k: Scalar(expr.xreplace({INDEX: k.to_sympy()})))
        - Scalar(expr.xreplace({INDEX: sp.Integer(0)}))
    )


SCALAR_BOX = TemplateAxiom(
    "scalar_box",
    Side(_scalar_box_left, arity_from_anchor(_first_kind(Z_SPIDER))),
    Side(lambda _dim, _a: _empty()),
    scalar=lambda dim, b: _box_value(dim, b),
    converse_free={"a": lambda dim, _b: PhaseVector(dim)},
    description="A Z box with no legs is the scalar 1 + a_1 + ... + a_{d-1}.",
)


# -- [P2] Figure 2: mixed dimensions (concrete dimensions) -----------------------------------


def _d(value: int) -> Dim:
    return Dim.concrete(value)


class MixedVar(Param):
    """The parameter of a mixed-dimensional Z box over ``D`` with least leg ``low``: zero at
    every index >= ``low``; binds the vector over ``low``."""

    def __init__(self, name: str, low: int) -> None:
        self.name = name
        self.low = low

    def match(self, phase: PhaseVector | None, dim: Dim, bindings: Bindings) -> Bindings | None:
        values = coefficients(phase, dim)
        if any(not values[k].is_zero for k in range(self.low, dim.to_int())):
            return None
        vector = from_coefficients(values[: self.low], _d(self.low)) or PhaseVector(_d(self.low))
        if self.name in bindings:
            known = bindings[self.name]
            ok = isinstance(known, PhaseVector) and _same_coefficients(known, vector, _d(self.low))
            return bindings if ok else None
        return {**bindings, self.name: vector}

    def build(self, dim: Dim, bindings: Bindings) -> PhaseVector | None:
        return _spread(_vec(bindings[self.name]), self.low, dim)

    def variables(self) -> frozenset[str]:
        return frozenset({self.name})

    def binds(self) -> frozenset[str]:
        return frozenset({self.name})


def _spread(a: PhaseVector | None, low: int, dim: Dim) -> PhaseVector:
    """``(a_1, ..., a_{low-1}, 0, ..., 0)`` over ``dim``."""
    values = coefficients(a, _d(low))
    weights = {k: values[k] if k < low else Scalar.zero() for k in range(1, dim.to_int())}
    return PhaseVector.from_weights(dim, weights)


def _ones_below(low: int) -> Callable[[Dim], PhaseVector | None]:
    def build(dim: Dim) -> PhaseVector | None:
        return _spread(None, low, dim)

    return build


def mixed_z(
    t: TB, legs: Sequence[tuple[Direction, int]], param: Param | None = None
) -> list[TPort]:
    """[P2]'s mixed-dimensional Z box on ``legs`` (direction, dimension), as its definition
    (2.3.3): a Z box over the product D, each leg through a merger or splitter whose other
    leg carries a pink dot of dimension D/d_i. Returns the external ports in order."""
    total = 1
    for _direction, value in legs:
        total *= value
    low = min(value for _direction, value in legs)
    ins = sum(1 for direction, _v in legs if direction is Direction.INPUT)
    z = t.node(
        Z_SPIDER,
        ins,
        len(legs) - ins,
        param if param is not None else Fixed(_ones_below(low)),
        dim=_d(total),
    )
    ports: list[TPort] = []
    k_in = k_out = 0
    for direction, value in legs:
        rest = total // value
        if direction is Direction.INPUT:
            b = t.conn(DIM_BINDER, ((rest, value), (total,)))
            pink = t.node(X_SPIDER, 0, 1, PHASELESS, pink=True, dim=_d(rest))
            t.wire(o(pink), i(b, 0))
            t.wire(o(b), i(z, k_in))
            k_in += 1
            ports.append(i(b, 1))
        else:
            sp_ = t.conn(DIM_SPLITTER, ((total,), (rest, value)))
            pink = t.node(X_SPIDER, 1, 0, PHASELESS, pink=True, dim=_d(rest))
            t.wire(o(z, k_out), i(sp_))
            t.wire(o(sp_, 0), i(pink))
            k_out += 1
            ports.append(o(sp_, 1))
    return ports


def _mult_at(t: TB, m: int, dim: int, *, up: bool = False) -> tuple[TPort, TPort]:
    """A multiplier labelled m on wires of dimension ``dim``."""
    if up:
        top = t.node(X_SPIDER, 1, m, PHASELESS, pink=True, dim=_d(dim))
        bottom = t.node(Z_SPIDER, m, 1, PHASELESS, dim=_d(dim))
    else:
        top = t.node(Z_SPIDER, 1, m, PHASELESS, dim=_d(dim))
        bottom = t.node(X_SPIDER, m, 1, PHASELESS, pink=True, dim=_d(dim))
    for k in range(m):
        t.wire(o(top, k), i(bottom, k))
    return i(top), o(bottom)


def _host_values(diagram: Diagram) -> list[int]:
    """Every concrete leg dimension of ``diagram`` read through its parameter environment."""
    values: set[int] = set()
    for node in diagram.nodes.values():
        for port in (*node.inputs, *node.outputs):
            dim = diagram.resolve_dim(port.dim)
            if dim.is_concrete:
                values.add(dim.to_int())
    return sorted(values)


def _divisor_pairs(values: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Every (m, n), m, n >= 2, with m and m * n among ``values``."""
    for m in values:
        for total in values:
            if m >= 2 and total % m == 0 and total // m >= 2:
                yield m, total // m


def _binder_left(_dim: Dim, a: Arities) -> Template:
    s_, t_ = a["s"], a["t"]
    t = TB()
    b = t.conn(DIM_BINDER, ((s_, t_), (s_ * t_,)), key="b")
    t.term("a", i(b, 0))
    t.term("b", i(b, 1))
    t.term("c", o(b))
    return t.done()


def _binder_right(_dim: Dim, a: Arities) -> Template:
    s_, t_ = a["s"], a["t"]
    t = TB()
    sp_ = t.conn(DIM_SPLITTER, ((s_ * t_,), (s_, t_)), key="s")
    t.term("a", o(sp_, 0))
    t.term("b", o(sp_, 1))
    t.term("c", i(sp_))
    return t.done()


def _connective_arities(gen: GeneratorType) -> Callable[[Diagram, NodeId, Dim], Iterator[Arities]]:
    def enumerate_(diagram: Diagram, anchor: NodeId, _dim: Dim) -> Iterator[Arities]:
        node = diagram.nodes[anchor]
        if node.generator_type.name != gen.name:
            return
        side = node.inputs if gen.name == DIM_BINDER.name else node.outputs
        s_, t_ = (diagram.resolve_dim(p.dim) for p in side)
        if s_.is_concrete and t_.is_concrete:
            yield {"s": s_.to_int(), "t": t_.to_int()}

    return enumerate_


BINDER_DEFINITION = TemplateAxiom(
    "binder_definition",
    Side(_binder_left, _connective_arities(DIM_BINDER)),
    Side(_binder_right, _connective_arities(DIM_SPLITTER)),
    concrete_only=True,
    description="[P2] (2.8): the dimension merger B is the splitter S with its legs bent.",
    quantifiers=("s", "t"),
)


def _dd_left(_dim: Dim, a: Arities) -> Template:
    m, n = a["m"], a["n"]
    t = TB()
    sp_ = t.conn(DIM_SPLITTER, ((m * n,), (m, n)), key="s")
    t.term("in", i(sp_))
    t.term("o0", o(sp_, 0))
    t.term("o1", o(sp_, 1))
    return t.done()


def _dd_right(_dim: Dim, a: Arities) -> Template:
    m, n = a["m"], a["n"]
    t = TB()
    p = t.node(X_SPIDER, 1, 2, PHASELESS, "p", pink=True, dim=_d(m * n))
    t.term("in", i(p))
    top, bottom = _mult_at(t, n, m * n, up=True)
    t.wire(o(p, 0), top)
    left = mixed_z(t, [(Direction.INPUT, m * n), (Direction.OUTPUT, m)])
    t.wire(bottom, left[0])
    t.term("o0", left[1])
    right = mixed_z(t, [(Direction.INPUT, m * n), (Direction.OUTPUT, n)])
    t.wire(o(p, 1), right[0])
    t.term("o1", right[1])
    return t.done()


def _dd_arities(diagram: Diagram, anchor: NodeId, _dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != DIM_SPLITTER.name:
        return
    m, n = (diagram.resolve_dim(p.dim) for p in node.outputs)
    if m.is_concrete and n.is_concrete:
        yield {"m": m.to_int(), "n": n.to_int()}


def _dd_right_arities(diagram: Diagram, anchor: NodeId, _dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != X_SPIDER.name:
        return
    for m, n in _divisor_pairs(_host_values(diagram)):
        yield {"m": m, "n": n}


QF_DD = TemplateAxiom(
    "qf_dd",
    Side(_dd_left, _dd_arities),
    Side(_dd_right, _dd_right_arities),
    concrete_only=True,
    description=(
        "[P2] DD: the splitter mn -> (m, n) is the pink dot of dimension mn whose outputs reach "
        "two-legged plain green dots, the first through the multiplier n."
    ),
    quantifiers=("m", "n"),
)


def _e_left(_dim: Dim, a: Arities) -> Template:
    n, e = a["n"], a["e"]
    t = TB()
    p = t.node(X_SPIDER, 0, 1, PHASELESS, "p", pink=True, dim=_d(n))
    top, bottom = _mult_at(t, e, n)
    t.wire(o(p), top)
    legs = mixed_z(t, [(Direction.INPUT, n), (Direction.OUTPUT, 2)])
    t.wire(bottom, legs[0])
    t.term("out", legs[1])
    return t.done()


def _e_right(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    p = t.node(X_SPIDER, 0, 1, PHASELESS, "p", pink=True, dim=_d(2))
    t.term("out", o(p))
    return t.done()


def _e_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != X_SPIDER.name or node.num_inputs or not dim.is_concrete:
        return
    n = dim.to_int()
    for e in range(1, n):
        yield {"n": n, "e": e}


def _e_right_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != X_SPIDER.name or dim != _d(2):
        return
    for n in _host_values(diagram):
        for e in range(1, n):
            yield {"n": n, "e": e}


QF_E = TemplateAxiom(
    "qf_e",
    Side(_e_left, _e_arities),
    Side(_e_right, _e_right_arities),
    concrete_only=True,
    description=(
        "[P2] E: the pink dot state of dimension n through a multiplier e != 0 mod n and a "
        "two-legged plain green dot (n -> 2) is the pink dot state of dimension 2."
    ),
    quantifiers=("n", "e"),
)


def _dzx_left(_dim: Dim, a: Arities) -> Template:
    m, n, p, q = a["m"], a["n"], a["p"], a["q"]
    t = TB()
    c = t.node(X_SPIDER, p, q, PHASELESS, "c", pink=True, dim=_d(m * n))
    for k in range(p):
        legs = mixed_z(t, [(Direction.INPUT, m), (Direction.OUTPUT, m * n)])
        top, bottom = _mult_at(t, n, m * n)
        t.wire(legs[1], top)
        t.wire(bottom, i(c, k))
        t.term(f"i{k}", legs[0])
    for k in range(q):
        top, bottom = _mult_at(t, n, m * n, up=True)
        t.wire(o(c, k), top)
        legs = mixed_z(t, [(Direction.INPUT, m * n), (Direction.OUTPUT, m)])
        t.wire(bottom, legs[0])
        t.term(f"o{k}", legs[1])
    return t.done()


def _dzx_right(_dim: Dim, a: Arities) -> Template:
    m, p, q = a["m"], a["p"], a["q"]
    t = TB()
    c = t.node(X_SPIDER, p, q, PHASELESS, "c", pink=True, dim=_d(m))
    for k in range(p):
        t.term(f"i{k}", i(c, k))
    for k in range(q):
        t.term(f"o{k}", o(c, k))
    return t.done()


def _dzx_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != X_SPIDER.name or not dim.is_concrete:
        return
    total = dim.to_int()
    for m, n in _divisor_pairs(_host_values(diagram)):
        if m * n == total:
            yield {"m": m, "n": n, "p": node.num_inputs, "q": node.num_outputs}


def _dzx_right_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != X_SPIDER.name or not dim.is_concrete:
        return
    m = dim.to_int()
    for n in _host_values(diagram):
        if n >= 2:
            yield {"m": m, "n": n, "p": node.num_inputs, "q": node.num_outputs}


QF_DZX = TemplateAxiom(
    "qf_dzx",
    Side(_dzx_left, _dzx_arities),
    Side(_dzx_right, _dzx_right_arities),
    concrete_only=True,
    description=(
        "[P2] DZX: a pink dot of dimension mn whose legs pass through multipliers n and "
        "two-legged plain green dots to dimension m is the pink dot of dimension m."
    ),
    quantifiers=("m", "n", "p", "q"),
)


def _k0m_left(_dim: Dim, a: Arities) -> Template:
    m, n1, n2 = a["m"], a["n1"], a["n2"]
    t = TB()
    p = t.node(X_SPIDER, 0, 1, Kj("j", sign=-1), "p", pink=True, dim=_d(m))
    legs = mixed_z(t, [(Direction.INPUT, m), (Direction.OUTPUT, n1), (Direction.OUTPUT, n2)])
    t.wire(o(p), legs[0])
    t.term("a", legs[1])
    t.term("b", legs[2])
    return t.done()


def _k0m_right(_dim: Dim, a: Arities) -> Template:
    m, n1, n2 = a["m"], a["n1"], a["n2"]
    low = min(m, n1, n2)
    t = TB()
    p1 = t.node(X_SPIDER, 0, 1, Kj("j", sign=-1), "p1", pink=True, dim=_d(n1))
    p2 = t.node(X_SPIDER, 0, 1, Kj("j", sign=-1), "p2", pink=True, dim=_d(n2))
    t.term("a", o(p1))
    t.term("b", o(p2))

    def value(_d0: Dim, b: Bindings) -> Scalar:
        j = _int(b["j"]) % m
        return Scalar.one() if j < low else Scalar.zero()

    return t.done(value)


def _k0m_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != X_SPIDER.name or node.num_inputs or not dim.is_concrete:
        return
    values = _host_values(diagram)
    for n1 in values:
        for n2 in values:
            if n1 <= n2:
                yield {"m": dim.to_int(), "n1": n1, "n2": n2}


def _k0m_right_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != X_SPIDER.name or node.num_inputs or not dim.is_concrete:
        return
    values = _host_values(diagram)
    for m in values:
        for n2 in values:
            if dim.to_int() <= n2:
                yield {"m": m, "n1": dim.to_int(), "n2": n2}


QF_K0 = TemplateAxiom(
    "qf_k0",
    Side(_k0m_left, _k0m_arities),
    Side(_k0m_right, _k0m_right_arities),
    concrete_only=True,
    description=(
        "[P2] K0: the pink K_{-j} state of dimension m into a plain green dot with legs m, n1, "
        "n2 is pink K_{-j} states of dimensions n1 and n2 times [j < min(m, n1, n2)]."
    ),
    quantifiers=("m", "n1", "n2", "j"),
)


def _bzwm_left(_dim: Dim, a: Arities) -> Template:
    m, n, k = a["m"], a["n"], a["k"]
    low = min(m, n, k)
    t = TB()
    wm = t.node(W_NODE, 1, 2, PHASELESS, "wm", dim=_d(m))
    wn = t.node(W_NODE, 1, 2, PHASELESS, "wn", dim=_d(n))
    t.term("u", i(wm))
    t.term("v", i(wn))
    for c in range(2):
        legs = mixed_z(
            t,
            [(Direction.INPUT, m), (Direction.INPUT, n), (Direction.OUTPUT, k)],
            MixedVar("a", low),
        )
        t.wire(o(wm, c), legs[0])
        t.wire(o(wn, c), legs[1])
        t.term(f"o{c}", legs[2])
    return t.done()


def _bzwm_right(_dim: Dim, a: Arities) -> Template:
    m, n, k = a["m"], a["n"], a["k"]
    low = min(m, n, k)
    t = TB()
    legs = mixed_z(
        t, [(Direction.INPUT, m), (Direction.INPUT, n), (Direction.OUTPUT, k)], MixedVar("a", low)
    )
    w = t.node(W_NODE, 1, 2, PHASELESS, "w", dim=_d(k))
    t.wire(legs[2], i(w))
    t.term("u", legs[0])
    t.term("v", legs[1])
    t.term("o0", o(w, 0))
    t.term("o1", o(w, 1))
    return t.done()


def _three_values(
    diagram: Diagram, anchor: NodeId, dim: Dim, gen: GeneratorType
) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != gen.name or not dim.is_concrete:
        return
    values = _host_values(diagram)
    for n in values:
        for k in values:
            yield {"m": dim.to_int(), "n": n, "k": k}


def _product_triples(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != Z_SPIDER.name or not dim.is_concrete:
        return
    values = _host_values(diagram)
    for m in values:
        for n in values:
            for k in values:
                if m * n * k == dim.to_int():
                    yield {"m": m, "n": n, "k": k}


def _bzwm_right_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    yield from _product_triples(diagram, anchor, dim)


QF_BZW = TemplateAxiom(
    "qf_bzw",
    Side(_bzwm_left, lambda d, a, dim: _three_values(d, a, dim, W_NODE)),
    Side(_bzwm_right, _bzwm_right_arities),
    concrete_only=True,
    description=(
        "[P2] BZW: W nodes of dimensions m and n crossed into mixed-dimensional Z boxes a "
        "(legs m, n, k) are the Z box a into a W node of dimension k."
    ),
    quantifiers=("m", "n", "k"),
)


def _tam_side(second: bool) -> Callable[[Dim, Arities], Template]:
    def build(_dim: Dim, a: Arities) -> Template:
        m, n, k, count = a["m"], a["n"], a["k"], a["i"]
        es = [a[f"e{c}"] for c in range(count)]
        t = TB()
        w = t.node(W_NODE, 1, count, PHASELESS, "w", dim=_d(m))
        if second:
            p = t.node(W_NODE, 1, count, PHASELESS, "p", dim=_d(n))
        else:
            p = t.node(X_SPIDER, 1, count, PHASELESS, "p", pink=True, dim=_d(n))
        t.term("u", i(w))
        t.term("v", i(p))
        for c, e in enumerate(es):
            top, bottom = _mult_at(t, e, n, up=True)
            t.wire(o(p, c), top)
            legs = mixed_z(t, [(Direction.INPUT, m), (Direction.INPUT, n), (Direction.OUTPUT, k)])
            t.wire(o(w, c), legs[0])
            t.wire(bottom, legs[1])
            t.term(f"o{c}", legs[2])
        return t.done()

    return build


def _tam_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    node = diagram.nodes[anchor]
    if node.generator_type.name != W_NODE.name or not dim.is_concrete:
        return
    values = _host_values(diagram)
    count = node.num_outputs
    for n in values:
        for k in values:
            for es in itertools.combinations_with_replacement(range(1, n), count):
                yield {
                    "m": dim.to_int(),
                    "n": n,
                    "k": k,
                    "i": count,
                    **{f"e{c}": e for c, e in enumerate(es)},
                }


QF_TA = TemplateAxiom(
    "qf_ta",
    Side(_tam_side(False), _tam_arities),
    Side(_tam_side(True), _tam_arities),
    concrete_only=True,
    description=(
        "[P2] TA: a W node of dimension m and a pink dot of dimension n whose outputs pass "
        "through multipliers e_t != 0 mod n meet in mixed plain green dots with outputs of "
        "dimension k; the pink dot may be replaced by a W node of dimension n."
    ),
    quantifiers=("m", "n", "k", "i", "e_1..e_i"),
)


def mixed_legs(diagram: Diagram, z: NodeId) -> list[tuple[Direction, int, PortRef]] | None:
    """The legs of the mixed-dimensional Z box whose Z spider is ``z`` (direction, dimension,
    external port), read through its mergers and splitters; None when ``z`` is not one."""
    node = diagram.nodes[z]
    if node.generator_type.name != Z_SPIDER.name:
        return None
    partners = _partners(diagram)
    legs: list[tuple[Direction, int, PortRef]] = []
    total = 1
    for ref in _legs(diagram, z):
        far = partners.get(ref)
        if far is None:
            return None
        conn = diagram.nodes[far.node_id]
        if ref.direction is Direction.INPUT:
            if conn.generator_type.name != DIM_BINDER.name or far.direction is not Direction.OUTPUT:
                return None
            pink_ref = partners.get(PortRef(far.node_id, Direction.INPUT, 0))
            external = PortRef(far.node_id, Direction.INPUT, 1)
            side = conn.inputs[1].dim
        else:
            if (
                conn.generator_type.name != DIM_SPLITTER.name
                or far.direction is not Direction.INPUT
            ):
                return None
            pink_ref = partners.get(PortRef(far.node_id, Direction.OUTPUT, 0))
            external = PortRef(far.node_id, Direction.OUTPUT, 1)
            side = conn.outputs[1].dim
        if pink_ref is None or not side.is_concrete:
            return None
        pink = diagram.nodes[pink_ref.node_id]
        if pink.generator_type.name != X_SPIDER.name or pink.num_inputs + pink.num_outputs != 1:
            return None
        if pink.phase is not None and not pink.phase.is_zero:
            return None
        legs.append((ref.direction, side.to_int(), external))
        total *= side.to_int()
    dim = _node_dim_of(diagram, z)
    if dim is None or not dim.is_concrete or dim.to_int() != total:
        return None
    return legs


def _node_dim_of(diagram: Diagram, node_id: NodeId) -> Dim | None:
    node = diagram.nodes[node_id]
    for port in (*node.inputs, *node.outputs):
        return port.dim
    return None


def _s1m_arities(diagram: Diagram, anchor: NodeId, dim: Dim) -> Iterator[Arities]:
    legs_a = mixed_legs(diagram, anchor)
    if legs_a is None:
        return
    partners = _partners(diagram)
    owners: dict[PortRef, tuple[NodeId, int]] = {}
    candidates: set[NodeId] = set()
    for node_id in diagram.nodes:
        if node_id == anchor:
            continue
        legs = mixed_legs(diagram, node_id)
        if legs is None:
            continue
        for index, (_d0, _v, external) in enumerate(legs):
            owners[external] = (node_id, index)
        candidates.add(node_id)
    by_b: dict[NodeId, list[tuple[int, int]]] = {}
    for index, (_d0, _v, external) in enumerate(legs_a):
        far = partners.get(external)
        if far is not None and far in owners:
            other, b_index = owners[far]
            by_b.setdefault(other, []).append((index, b_index))
    for other, connections in sorted(by_b.items(), key=lambda item: int(item[0])):
        legs_b = mixed_legs(diagram, other)
        assert legs_b is not None
        arities: dict[str, int] = {"na": len(legs_a), "nb": len(legs_b), "nc": len(connections)}
        for k, (direction, value, _e) in enumerate(legs_a):
            arities[f"a{k}d"] = value
            arities[f"a{k}o"] = int(direction is Direction.OUTPUT)
        for k, (direction, value, _e) in enumerate(legs_b):
            arities[f"b{k}d"] = value
            arities[f"b{k}o"] = int(direction is Direction.OUTPUT)
        for k, (ia, ib) in enumerate(connections):
            arities[f"c{k}a"] = ia
            arities[f"c{k}b"] = ib
        yield arities


def _s1m_legs(a: Arities, side: str) -> list[tuple[Direction, int]]:
    return [
        (Direction.OUTPUT if a[f"{side}{k}o"] else Direction.INPUT, a[f"{side}{k}d"])
        for k in range(a[f"n{side}"])
    ]


def _s1m_lows(a: Arities) -> tuple[int, int, int]:
    legs_a, legs_b = _s1m_legs(a, "a"), _s1m_legs(a, "b")
    low_a = min(v for _d0, v in legs_a)
    low_b = min(v for _d0, v in legs_b)
    return low_a, low_b, min(low_a, low_b)


def _s1m_left(_dim: Dim, a: Arities) -> Template:
    low_a, low_b, _m = _s1m_lows(a)
    t = TB()
    ports_a = mixed_z(t, _s1m_legs(a, "a"), MixedVar("a", low_a))
    ports_b = mixed_z(t, _s1m_legs(a, "b"), MixedVar("b", low_b))
    joined_a = {a[f"c{k}a"] for k in range(a["nc"])}
    joined_b = {a[f"c{k}b"] for k in range(a["nc"])}
    for k in range(a["nc"]):
        t.wire(ports_a[a[f"c{k}a"]], ports_b[a[f"c{k}b"]])
    for k, port in enumerate(ports_a):
        if k not in joined_a:
            t.term(f"a{k}", port)
    for k, port in enumerate(ports_b):
        if k not in joined_b:
            t.term(f"b{k}", port)
    return t.done()


def _s1m_right(_dim: Dim, a: Arities) -> Template:
    low_a, low_b, low = _s1m_lows(a)
    joined_a = {a[f"c{k}a"] for k in range(a["nc"])}
    joined_b = {a[f"c{k}b"] for k in range(a["nc"])}
    legs_a, legs_b = _s1m_legs(a, "a"), _s1m_legs(a, "b")
    free = [("a", k, leg) for k, leg in enumerate(legs_a) if k not in joined_a]
    free += [("b", k, leg) for k, leg in enumerate(legs_b) if k not in joined_b]
    t = TB()
    if not free:
        return t.done(lambda dim, b: _closed_product(b, low_a, low_b, low))
    low_out = min(v for _s, _k, (_d0, v) in free)

    def product(dim: Dim, b: Bindings) -> PhaseVector | None:
        va = coefficients(_vec(b["a"]), _d(low_a))
        vb = coefficients(_vec(b["b"]), _d(low_b))
        values = [va[k] * vb[k] if k < low else Scalar.zero() for k in range(low_out)]
        return _spread(from_coefficients(values, _d(low_out)), low_out, dim)

    ports = mixed_z(t, [leg for _s, _k, leg in free], Derived(("a", "b"), product))
    for (side, k, _leg), port in zip(free, ports, strict=True):
        t.term(f"{side}{k}", port)
    return t.done()


def _closed_product(b: Bindings, low_a: int, low_b: int, low: int) -> Scalar:
    va = coefficients(_vec(b["a"]), _d(low_a))
    vb = coefficients(_vec(b["b"]), _d(low_b))
    total = Scalar.zero()
    for k in range(low):
        total = total + va[k] * vb[k]
    return total


QF_S1 = TemplateAxiom(
    "qf_s1",
    Side(_s1m_left, _s1m_arities),
    Side(_s1m_right),
    concrete_only=True,
    description=(
        "[P2] S1: mixed-dimensional Z boxes a and b joined by wires of any dimensions fuse into "
        "the Z box ab' = (a_1 b_1, ..., a_{M-1} b_{M-1}, 0, ...), M the least leg dimension."
    ),
    quantifiers=("legs", "dimensions"),
)


# -- explicit converses ---------------------------------------------------------------------


def _d1_converse_left(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    f = t.fs(2)
    t.z(1, 1, Var("b"), "z")
    t.wire(o(f[1]), i("z"))
    t.term("in", i(f[0]))
    t.term("out", o("z"))
    return t.done()


def _d1_converse_right(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(1, 1, Derived(("b",), lambda dim, b: reversed_vector(_vec(b["b"]), dim)), "z")
    f = t.fs(2)
    t.wire(o("z"), i(f[0]))
    t.term("in", i("z"))
    t.term("out", o(f[1]))
    return t.done()


ZXW_D1_CONVERSE = TemplateAxiom(
    "zxw_d1_converse",
    Side(_d1_converse_left),
    Side(_d1_converse_right),
    concrete_only=True,
    description="[P1] D1 read right to left: D followed by Z box b is Z box <-b followed by D.",
)


def _k2_converse_left(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(1, 1, Var("c"), "z")
    t.pink(1, 1, Kj("j"), "p")
    t.wire(o("z"), i("p"))
    t.term("in", i("z"))
    t.term("out", o("p"))
    return t.done()


def _unshift(dim: Dim, b: Bindings) -> PhaseVector | None:
    """The a with ``k_j(a) = c``: ``a_k = c_{k+j} / c_j``."""
    r = dim.to_int()
    j = _int(b["j"])
    values = coefficients(_vec(b["c"]), dim)
    pivot = values[j % r]
    if not _nonzero(pivot):
        raise RewriteDomainError("k_j(a) = c needs c_j != 0")
    out = [Scalar(values[(k + j) % r].to_sympy() / pivot.to_sympy()) for k in range(r)]
    return from_coefficients(out, dim)


def _k2_converse_right(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.pink(1, 1, Kj("j"), "p")
    t.z(1, 1, Derived(("c", "j"), _unshift), "z")
    t.wire(o("p"), i("z"))
    t.term("in", i("p"))
    t.term("out", o("z"))
    return t.done(lambda dim, b: _coefficient(b["c"], _int(b["j"]), dim))


ZXW_K2_CONVERSE = TemplateAxiom(
    "zxw_k2_converse",
    Side(_k2_converse_left),
    Side(_k2_converse_right),
    conditions=(
        Condition(
            "c_j_nonzero",
            "c_j != 0, so c = k_j(a) for a_k = c_{k+j}/c_j",
            lambda dim, b: _nonzero(_coefficient(b["c"], _int(b["j"]), dim)),
        ),
    ),
    concrete_only=True,
    description="[P1] K2 read right to left.",
    quantifiers=("j", "d"),
)


def _unfuse_left(_dim: Dim, a: Arities) -> Template:
    t = TB()
    n = a["n"]
    t.z(0, n, Var("c"), "z")
    for k in range(n):
        t.term(f"l{k}", o("z", k))
    return t.done()


def _unfuse_right(_dim: Dim, a: Arities) -> Template:
    t = TB()
    n, mask, k = a["n"], a["mask"], a["k"]
    first = [index for index in range(n) if mask >> index & 1]
    second = [index for index in range(n) if not mask >> index & 1]
    t.z(0, len(first) + k, Var("a"), "a")
    t.z(k, len(second), Var("b"), "b")
    for wire in range(k):
        t.wire(o("a", len(first) + wire), i("b", wire))
    for position, index in enumerate(first):
        t.term(f"l{index}", o("a", position))
    for position, index in enumerate(second):
        t.term(f"l{index}", o("b", position))
    return t.done()


def _unfuse_arities(diagram: Diagram, anchor: NodeId) -> Iterator[Arities]:
    if diagram.nodes[anchor].generator_type.name != Z_SPIDER.name:
        return
    n = _count(diagram, anchor)
    for mask in range(2**n):
        for k in (1, 2):
            yield {"n": n, "mask": mask, "k": k}


def _fuses_to(dim: Dim, b: Bindings) -> bool:
    a, c, d = _vec(b["a"]), _vec(b["b"]), _vec(b["c"])
    product = (a or PhaseVector(dim)) + (c or PhaseVector(dim))
    return _same_coefficients(product, d, dim)


def _same_coefficients(x: PhaseVector | None, y: PhaseVector | None, dim: Dim) -> bool:
    left = x if x is not None else PhaseVector(dim)
    right = y if y is not None else PhaseVector(dim)
    if left == right:
        return True
    if not dim.is_concrete:
        return False
    return all(
        sp.simplify(left.coefficient(k).to_sympy() - right.coefficient(k).to_sympy()) == 0
        for k in range(1, dim.to_int())
    )


ZXW_S1_CONVERSE = TemplateAxiom(
    "zxw_s1_converse",
    Side(_unfuse_left, arity_from_anchor(_unfuse_arities)),
    Side(_unfuse_right),
    conditions=(Condition("fuses_back", "ab = c", _fuses_to),),
    free={"a": lambda dim, b: b["c"], "b": lambda dim, _b: PhaseVector(dim)},
    description=(
        "[P1] S1 read right to left: the Z box c is Z boxes a and b (ab = c) joined by k "
        "wires, the legs split by a mask; by default a = c and b is the plain green dot."
    ),
    quantifiers=("n", "mask", "k", "d"),
)


def _ad_converse_left() -> Template:
    t = TB()
    t.z(1, 0, Var("c"), "z")
    t.term("in", i("z"))
    return t.done()


def _ad_converse_right() -> Template:
    t = TB()
    t.w(2, "w")
    t.z(1, 0, Var("a"), "za")
    t.z(1, 0, Var("b"), "zb")
    t.wire(o("w", 0), i("za"))
    t.wire(o("w", 1), i("zb"))
    t.term("in", i("w"))
    return t.done()


def _adds_to(dim: Dim, b: Bindings) -> bool:
    return _same_coefficients(sum_vector(_vec(b["a"]), _vec(b["b"]), dim), _vec(b["c"]), dim)


ZXW_AD_CONVERSE = TemplateAxiom(
    "zxw_ad_converse",
    fixed_side(_ad_converse_left),
    fixed_side(_ad_converse_right),
    conditions=(Condition("adds_back", "a + b = c", _adds_to),),
    free={"a": lambda dim, b: b["c"], "b": lambda dim, _b: zero_vector(dim)},
    concrete_only=True,
    description=(
        "[P1] AD read right to left: the Z box effect c is a W node with effects a and b, "
        "a + b = c; by default a = c and b = 0."
    ),
)


class UnitIndex(Param):
    """The vector with a single 1 at index ``sign * j`` and 0 elsewhere; binds j."""

    def __init__(self, name: str, sign: int = 1) -> None:
        self.name = name
        self.sign = sign

    def match(self, phase: PhaseVector | None, dim: Dim, bindings: Bindings) -> Bindings | None:
        if not dim.is_concrete:
            return None
        r = dim.to_int()
        values = coefficients(phase, dim)
        ones = [k for k in range(1, r) if sp.simplify(values[k].to_sympy() - 1) == 0]
        zeros = [k for k in range(1, r) if values[k].is_zero]
        if len(zeros) + len(ones) != r - 1 or len(ones) > 1:
            return None
        index = ones[0] if ones else 0
        return {**bindings, self.name: sp.Integer((self.sign * index) % r)}

    def build(self, dim: Dim, bindings: Bindings) -> PhaseVector | None:
        return unit_vector(self.sign * _int_expr(bindings[self.name]), dim)

    def variables(self) -> frozenset[str]:
        return frozenset({self.name})

    def binds(self) -> frozenset[str]:
        return frozenset({self.name})


def _bsj_converse_left() -> Template:
    t = TB()
    t.z(0, 1, UnitIndex("j", sign=-1), "z")
    t.term("out", o("z"))
    return t.done()


ZXW_BSJ_CONVERSE = TemplateAxiom(
    "zxw_bsj_converse",
    fixed_side(_bsj_converse_left),
    fixed_side(_bsj_left),
    concrete_only=True,
    description="[P1] Bsj read right to left.",
    quantifiers=("j", "d"),
)


def _zv_converse_left(r: int) -> Template:
    t = TB()
    t.z(0, 1, Var("c"), "z")
    t.term("out", o("z"))
    return t.done()


def _constant(dim: Dim, b: Bindings) -> bool:
    values = coefficients(_vec(b["c"]), dim)
    return all(
        sp.simplify(values[k].to_sympy() - values[1].to_sympy()) == 0
        for k in range(1, dim.to_int())
    )


def _zv_converse_right(r: int) -> Template:
    t = TB()
    z = t.z(0, 1, Var("a"), "z")
    top, bottom = _v_box(t, r)
    t.wire(o(z), top)
    t.term("out", bottom)
    return t.done()


def _zv_matches(dim: Dim, b: Bindings) -> bool:
    return _constant(dim, b) and _same_coefficients(
        constant_vector(_coefficient(b["a"], -1, dim), dim), _vec(b["c"]), dim
    )


ZXW_ZV_CONVERSE = TemplateAxiom(
    "zxw_zv_converse",
    concrete_side(_zv_converse_left),
    concrete_side(_zv_converse_right),
    conditions=(Condition("constant_vector", "c = (x, ..., x) with x = a_{d-1}", _zv_matches),),
    free={"a": lambda dim, b: last_vector(_coefficient(b["c"], 1, dim), dim)},
    concrete_only=True,
    description="[P1] ZV read right to left: by default a = (0, ..., 0, x).",
)


def _box_entry(k: int) -> Callable[[Dim, Bindings], PhaseVector | None]:
    """The box labelled a_k, read off the bound vector a."""

    def compute(dim: Dim, b: Bindings) -> PhaseVector | None:
        return last_vector(_coefficient(b["a"], k, dim), dim)

    return compute


def _va_converse_left(r: int) -> Template:
    t = TB()
    top, bottom = _v_box(t, r)
    z = t.z(1, 1, Var("a"), "z")
    t.wire(bottom, i(z))
    t.term("in", top)
    t.term("out", o(z))
    return t.done()


def _va_converse_right(r: int) -> Template:
    t = TB()
    t.w(r - 1, "w")
    t.pink(r - 1, 1, key="q")
    for k in range(1, r):
        box = t.z(
            1,
            1,
            Derived(("a",), _box_entry(k)),
            f"b{k}",
        )
        t.wire(o("w", k - 1), i(box))
        top, bottom = t.multiplier((-k) % r)
        t.wire(o(box), top)
        t.wire(bottom, i("q", k - 1))
    t.term("in", i("w"))
    t.term("out", o("q"))
    return t.done()


ZXW_VA_CONVERSE = TemplateAxiom(
    "zxw_va_converse",
    concrete_side(_va_converse_left),
    concrete_side(_va_converse_right),
    concrete_only=True,
    description="[P1] VA read right to left.",
)


def _va2_converse_left(r: int) -> Template:
    t = TB()
    box = t.z(1, 1, ONE_LAST, "b0")
    km = t.pink(1, 1, k_fixed(-1), "km")
    wt = t.w(2, "wt")
    green = t.z(0, 1, key="g")
    kp = t.pink(1, 1, k_fixed(1), "kp")
    z = t.z(1, 1, Var("a"), "z")
    t.wire(o(box), i(km))
    t.wire(o(km), o(wt, 0))
    t.wire(o(green), o(wt, 1))
    t.wire(i(wt), i(kp))
    t.wire(o(kp), i(z))
    t.term("in", i(box))
    t.term("out", o(z))
    return t.done()


def _va2_converse_right(r: int) -> Template:
    t = TB()
    t.w(r, "w")
    t.pink(r, 1, key="q")
    box = t.z(1, 1, ONE_LAST, "b0")
    t.wire(o("w", 0), i(box))
    top, bottom = t.multiplier(0)
    t.wire(o(box), top)
    t.wire(bottom, i("q", 0))
    for k in range(1, r):
        box = t.z(
            1,
            1,
            Derived(("a",), _box_entry(k)),
            f"b{k}",
        )
        t.wire(o("w", k), i(box))
        top, bottom = t.multiplier((-k) % r)
        t.wire(o(box), top)
        t.wire(bottom, i("q", k))
    t.term("in", i("w"))
    t.term("out", o("q"))
    return t.done()


QF_VA_CONVERSE = TemplateAxiom(
    "qf_va_converse",
    concrete_side(_va2_converse_left),
    concrete_side(_va2_converse_right),
    concrete_only=True,
    description="[P2] VA read right to left.",
)


def _d1g_converse_left(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.z(a["n"], a["m"], Var("b"), "z")
    for k in range(a["n"]):
        t.term(f"i{k}", i("z", k))
    for k in range(a["m"]):
        f = t.fs(2)
        t.wire(o("z", k), i(f[0]))
        t.term(f"o{k}", o(f[1]))
    return t.done()


def _d1g_converse_right(_dim: Dim, a: Arities) -> Template:
    t = TB()
    t.z(a["n"], a["m"], Derived(("b",), lambda dim, b: reversed_vector(_vec(b["b"]), dim)), "z")
    for k in range(a["n"]):
        f = t.fs(2)
        t.wire(o(f[1]), i("z", k))
        t.term(f"i{k}", i(f[0]))
    for k in range(a["m"]):
        t.term(f"o{k}", o("z", k))
    return t.done()


QF_D1_CONVERSE = TemplateAxiom(
    "qf_d1_converse",
    Side(_d1g_converse_left, arity_from_anchor(_z_split_arities)),
    Side(_d1g_converse_right),
    concrete_only=True,
    description="[P2] D1 read right to left.",
    quantifiers=("n", "m", "d"),
)


def _phase_of_weight(value: Scalar) -> Phase:
    """The phase t with ``value = e^{2 pi i t}``: read off ``exp(2 pi i t)``, else t rational."""
    expr = value.to_sympy()
    if isinstance(expr, sp.exp):
        turns = sp.simplify(expr.args[0] / (2 * sp.pi * sp.I))
        if turns.is_real:
            return Phase(turns)
    if sp.simplify(sp.Abs(expr) - 1) != 0:
        raise RewriteDomainError(f"{value} is not a unit complex number")
    turns = sp.nsimplify(sp.arg(expr) / (2 * sp.pi))
    if not turns.is_Rational:
        raise RewriteDomainError(f"{value} is not a rational root of unity")
    if sp.simplify(sp.exp(2 * sp.pi * sp.I * turns) - expr) != 0:
        raise RewriteDomainError(f"{value} is not e^(2 pi i {turns})")
    return Phase(turns)


def _as_phases(dim: Dim, b: Bindings) -> PhaseVector:
    pv = _vec(b["w"])
    if pv is None or pv.entries() or pv.rule is not None or not pv.is_weighted:
        raise RewriteDomainError("weight_to_phase reads a vector of weights only")
    return PhaseVector(dim, {k: _phase_of_weight(pv.weight(k)) for k in range(1, dim.to_int())})


def _weight_phase_left(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(0, 0, Var("w"), "z", open=True)
    return t.done()


def _weight_phase_right(_dim: Dim, _a: Arities) -> Template:
    t = TB()
    t.z(0, 0, Derived(("w",), _as_phases), "z")
    t.rest = {"z": "z"}
    return t.done()


WEIGHT_TO_PHASE = TemplateAxiom(
    "weight_to_phase",
    Side(_weight_phase_left, arity_from_anchor(_first_kind(Z_SPIDER))),
    Side(_weight_phase_right),
    concrete_only=True,
    description="phase_to_weight read right to left: weights that are roots of unity.",
)


AXIOMS: tuple[TemplateAxiom, ...] = (
    ZXW_S1,
    ZXW_S2,
    ZXW_S2_PINK,
    ZXW_D1,
    ZXW_EPT,
    ZXW_B2,
    ZXW_K0,
    ZXW_K1,
    ZXW_K2,
    ZXW_ZER,
    ZXW_P1,
    ZXW_P1_MULTIPLIER,
    ZXW_H1,
    ZXW_PCY,
    ZXW_AD,
    ZXW_SYM,
    ZXW_ASO,
    ZXW_BZW,
    ZXW_WW,
    ZXW_BS0,
    ZXW_BSJ,
    ZXW_TA,
    ZXW_HD,
    ZXW_ZV,
    ZXW_VA,
    ZXW_VW,
    ZXW_KZ,
    QF_WF,
    QF_W_SYM,
    QF_Z_FLEX,
    QF_MW,
    QF_D1,
    QF_WW,
    QF_VA,
    QF_HD,
    BINDER_DEFINITION,
    QF_DD,
    QF_E,
    QF_DZX,
    QF_K0,
    QF_BZW,
    QF_TA,
    QF_S1,
    X_SPIDER_DEFINITION,
    TRIANGLE_DEFINITION,
    TRIANGLE_INVERSE_DEFINITION,
    PHASE_TO_WEIGHT,
    SCALAR_BOX,
)
"""Every axiom and definitional rule, read left to right."""


def _determined(axiom: TemplateAxiom) -> bool:
    """Whether the right side's parameters, with the converse defaults, determine every
    variable the left side builds."""
    dim = Dim.concrete(3)
    for sample in _SAMPLES:
        try:
            right = axiom.right.template(dim, sample)
            left = axiom.left.template(dim, sample)
        except (KeyError, IndexError, ValueError, RewriteDomainError):
            continue
        if right is None or left is None:
            return False
        if not right.nodes and not right.bare and not axiom.converse_free:
            return False
        bound: set[str] = set(axiom.converse_free)
        for node in right.nodes:
            bound |= node.param.binds()
        needed: set[str] = set()
        for node in left.nodes:
            needed |= node.param.variables()
        return needed <= bound
    return False


_SAMPLES: tuple[dict[str, int], ...] = (
    {"k": 1, "n": 2, "m": 2, "p": 0, "q": 1, "r": 1, "e": 1, "e0": 1},
    {"k": 0, "n": 2, "m": 2},
    {"n": 1, "e0": 1},
    {"s": 2, "t": 3},
    {"m": 2, "n": 3, "k": 2, "i": 1, "e0": 1},
    {"m": 2, "n1": 2, "n2": 3},
)
"""Arity samples :func:`_determined` reads the templates at."""


SCALAR_BOX_CONVERSE = replace(
    SCALAR_BOX,
    conditions=(
        Condition(
            "box_value_nonzero",
            "the box's value 1 + a_1 + ... + a_{d-1} is not 0",
            lambda dim, b: _nonzero(_box_value(dim, b)),
        ),
    ),
).converse("scalar_box_converse")
"""A Z box of nonzero value v with no legs inserted, times 1/v."""

EXPLICIT_CONVERSES: tuple[TemplateAxiom, ...] = (
    SCALAR_BOX_CONVERSE,
    ZXW_D1_CONVERSE,
    ZXW_K2_CONVERSE,
    ZXW_S1_CONVERSE,
    ZXW_AD_CONVERSE,
    ZXW_BSJ_CONVERSE,
    ZXW_ZV_CONVERSE,
    ZXW_VA_CONVERSE,
    QF_VA_CONVERSE,
    QF_D1_CONVERSE,
    WEIGHT_TO_PHASE,
)
"""Converses written out: their matched side carries a derived parameter."""

CONVERSES: tuple[TemplateAxiom, ...] = (
    *(
        axiom.converse(f"{axiom.name}_converse")
        for axiom in AXIOMS
        if _determined(axiom)
        and f"{axiom.name}_converse" not in {c.name for c in EXPLICIT_CONVERSES}
        and axiom is not PHASE_TO_WEIGHT
    ),
    *EXPLICIT_CONVERSES,
)
"""The converse of every axiom: generated where the right side determines the left side's
parameters, written out otherwise. ``weight_to_phase`` is the converse of
``phase_to_weight``."""


def by_name() -> Mapping[str, TemplateAxiom]:
    """Every template axiom and converse by rule name."""
    return {axiom.name: axiom for axiom in (*AXIOMS, *CONVERSES)}


__all__ = ["AXIOMS", "CONVERSES", "by_name"]
