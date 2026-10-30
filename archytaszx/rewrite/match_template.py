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

"""Template rules: an equation between two small diagrams, matched and rebuilt generically.

A :class:`Template` is a small diagram of :class:`TNode`\\ s, wires between their ports, and
named terminals (the ports the equation's interface passes through). A :class:`TemplateAxiom`
pairs two :class:`Side`\\ s, each a function from a dimension and arity parameters to a
template, with an enumerator of the arity parameters a host node can anchor. Applying the
axiom forwards matches the left side and builds the right; its converse matches the right
side and builds the left.

Matching is subgraph matching up to the symmetry of each generator's legs: every leg of a Z
spider is interchangeable, in either direction; an X spider's inputs among themselves and its
outputs among themselves; a W node's outputs. Port-ordered generators (F, T, Ti, B, S) match
port for port. An open node may carry further legs (its rest), which the built side hands to
the node its side names. Every matched node lies outside every bang box and every matched leg
carries the anchor's dimension exactly.

A node's parameter is a :class:`Param`: :class:`Var` binds the host's phase vector, a fixed
vector must equal it, :class:`Kj` recognises the phase vector K_j = (j k / d turns)_k and
binds j, and a derived parameter is computed from the bindings. A ``pink`` node is an X spider
in [P1]'s normalisation, ``d^((m + n)/2 - 1)`` times the engine's X spider; the scalar a
template rule introduces includes the ratio of the two sides' pink normalisations.

A right-hand bare wire (two terminals joined directly) splices through the far port of one
terminal; when both terminals are boundary entries, a phaseless Z spider with one leg per
entry stands in for the wire.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import INDEX, Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import (
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef, Wire
from archytaszx.rewrite.rule import (
    BuildResult,
    DimensionConstraint,
    Match,
    Pattern,
    RewriteDomainError,
    RewriteGrammarError,
    SideCondition,
    SideConditionOutcome,
    check_side_condition_coverage,
)

TPort = tuple[str, Direction, int]
"""A template port: node key, direction, index."""

Bindings = Mapping[str, object]
"""Variable name -> bound value: a PhaseVector, an int or sympy expression, or a Scalar."""


class Param:
    """A node parameter of a template: how to match a host phase vector and how to build one."""

    def match(self, phase: PhaseVector | None, dim: Dim, bindings: Bindings) -> Bindings | None:
        """The bindings extended by this parameter's match against ``phase``, else None."""
        built = self.build(dim, bindings)
        return bindings if _same_vector(phase, built, dim) else None

    def build(self, dim: Dim, bindings: Bindings) -> PhaseVector | None:
        """The phase vector this parameter denotes under ``bindings``."""
        raise NotImplementedError

    def variables(self) -> frozenset[str]:
        """The variables :meth:`build` reads."""
        return frozenset()

    def binds(self) -> frozenset[str]:
        """The variables :meth:`match` binds."""
        return frozenset()


@dataclass(frozen=True)
class Fixed(Param):
    """A parameter that is a fixed function of the dimension; None is phaseless."""

    vector: Callable[[Dim], PhaseVector | None]

    def build(self, dim: Dim, bindings: Bindings) -> PhaseVector | None:
        return self.vector(dim)


PHASELESS = Fixed(lambda _dim: None)
"""The phaseless parameter."""


@dataclass(frozen=True)
class Var(Param):
    """A free vector variable: binds the host's vector (or reads the bound one)."""

    name: str

    def match(self, phase: PhaseVector | None, dim: Dim, bindings: Bindings) -> Bindings | None:
        vector = phase if phase is not None else PhaseVector(dim)
        if self.name in bindings:
            return bindings if _same_vector(vector, _vector(bindings[self.name]), dim) else None
        return {**bindings, self.name: vector}

    def build(self, dim: Dim, bindings: Bindings) -> PhaseVector | None:
        return _vector(bindings[self.name])

    def variables(self) -> frozenset[str]:
        return frozenset({self.name})

    def binds(self) -> frozenset[str]:
        return frozenset({self.name})


@dataclass(frozen=True)
class Derived(Param):
    """A vector computed from bound variables; matched by recomputing and comparing."""

    reads: tuple[str, ...]
    compute: Callable[[Dim, Bindings], PhaseVector | None]

    def match(self, phase: PhaseVector | None, dim: Dim, bindings: Bindings) -> Bindings | None:
        if any(name not in bindings for name in self.reads):
            return None
        try:
            built = self.compute(dim, bindings)
        except (ArithmeticError, ValueError, KeyError):
            return None
        return bindings if _same_vector(phase, built, dim) else None

    def build(self, dim: Dim, bindings: Bindings) -> PhaseVector | None:
        return self.compute(dim, bindings)

    def variables(self) -> frozenset[str]:
        return frozenset(self.reads)


@dataclass(frozen=True)
class Kj(Param):
    """The phase vector K_{sign * j + offset}: ``(sign j + offset) k / d`` turns at index k."""

    name: str
    sign: int = 1
    offset: int = 0

    def match(self, phase: PhaseVector | None, dim: Dim, bindings: Bindings) -> Bindings | None:
        found = kj_index(phase, dim)
        if found is None:
            return None
        if self.name in bindings:
            want = self.sign * _int_expr(bindings[self.name]) + self.offset
            return bindings if _congruent(found, want, dim) else None
        value = self.sign * (found - self.offset)
        if dim.is_concrete:
            value = sp.Integer(int(value) % dim.to_int())
        return {**bindings, self.name: value}

    def build(self, dim: Dim, bindings: Bindings) -> PhaseVector | None:
        return k_vector(self.sign * _int_expr(bindings[self.name]) + self.offset, dim)

    def variables(self) -> frozenset[str]:
        return frozenset({self.name})

    def binds(self) -> frozenset[str]:
        return frozenset({self.name})


def _int_expr(value: object) -> sp.Expr:
    if isinstance(value, int) and not isinstance(value, bool):
        return sp.Integer(value)
    if isinstance(value, sp.Expr):
        return value
    raise RewriteGrammarError(f"expected an integer expression, got {value!r}")


def _vector(value: object) -> PhaseVector:
    if not isinstance(value, PhaseVector):
        raise RewriteGrammarError(f"expected a PhaseVector binding, got {value!r}")
    return value


def _congruent(a: sp.Expr, b: sp.Expr, dim: Dim) -> bool:
    if dim.is_concrete:
        diff = sp.simplify(a - b)
        return bool(diff.is_Integer and int(diff) % dim.to_int() == 0)
    return bool(sp.simplify(a - b) == 0)


def k_vector(j: sp.Expr | int, dim: Dim) -> PhaseVector | None:
    """The phase vector K_j over ``dim``: ``j k / d`` turns at index k; None when j == 0."""
    j_expr = _int_expr(j)
    if dim.is_concrete:
        r = dim.to_int()
        if not j_expr.is_Integer:
            raise RewriteDomainError(f"K_j at concrete d needs an integer j, got {j_expr}")
        jr = int(j_expr) % r
        if jr == 0:
            return None
        return PhaseVector(dim, {k: Phase.turns(sp.Rational(jr * k % r, r)) for k in range(1, r)})
    if j_expr == 0:
        return None
    return PhaseVector(dim, rule=Phase(j_expr * INDEX / dim.to_sympy()))


def kj_index(phase: PhaseVector | None, dim: Dim) -> sp.Expr | None:
    """The j with ``phase == K_j``, or None. Phaseless is j = 0."""
    if phase is None or phase.is_zero:
        return sp.Integer(0)
    if phase.is_weighted:
        return None
    if dim.is_concrete:
        r = dim.to_int()
        first = phase.get(1).to_sympy_turns()
        if not first.is_Rational:
            return None
        j = first * r
        if not j.is_Integer:
            return None
        built = k_vector(j, dim)
        return sp.Integer(int(j) % r) if built == phase else None
    if phase.entries() or phase.rule is None:
        return None
    turns = phase.rule.to_sympy_turns()
    j = sp.simplify(turns * dim.to_sympy() / INDEX)
    if INDEX in j.free_symbols or not (j.is_integer or j.is_Integer):
        return None
    return j if k_vector(j, dim) == phase else None


def _same_vector(a: PhaseVector | None, b: PhaseVector | None, dim: Dim) -> bool:
    """Whether two parameters are provably the same vector; None is the zero vector."""
    left = a if a is not None else PhaseVector(dim)
    right = b if b is not None else PhaseVector(dim)
    if left == right:
        return True
    if not dim.is_concrete:
        return False
    r = dim.to_int()
    return all(
        sp.simplify(left.coefficient(k).to_sympy() - right.coefficient(k).to_sympy()) == 0
        for k in range(1, r)
    )


@dataclass(frozen=True)
class TNode:
    """One node of a template.

    ``ins``/``outs`` are its legs in the template. ``open`` lets a host node carry further
    legs in its symmetric groups (an open node has no terminal port). ``pink`` marks an X
    spider in [P1]'s normalisation.
    """

    key: str
    gen: GeneratorType
    ins: int
    outs: int
    param: Param = PHASELESS
    open: bool = False
    pink: bool = False
    dim: Dim | None = None
    """The dimension of every leg, when it is not the template's dimension."""
    port_dims: tuple[tuple[Dim, ...], tuple[Dim, ...]] | None = None
    """Input and output dimensions of a connective (B or S), in port order."""

    def leg_dims(self, default: Dim) -> tuple[tuple[Dim, ...], tuple[Dim, ...]]:
        """The input and output leg dimensions of this node."""
        if self.port_dims is not None:
            return self.port_dims
        dim = self.dim if self.dim is not None else default
        return (dim,) * self.ins, (dim,) * self.outs


@dataclass(frozen=True)
class Template:
    """A small diagram: nodes, wires between template ports, and named terminals.

    ``bare`` joins two terminals directly by a wire (built side only). ``rest`` names, per
    open node of the other side, the node of this side that receives its rest legs.
    ``scalar`` is a factor this side carries (a scalar diagram of the paper, evaluated).
    """

    nodes: tuple[TNode, ...]
    wires: tuple[tuple[TPort, TPort], ...] = ()
    terminals: Mapping[str, TPort] = field(default_factory=dict)
    bare: tuple[tuple[str, str], ...] = ()
    rest: Mapping[str, str] = field(default_factory=dict)
    scalar: Callable[[Dim, Bindings], Scalar] | None = None

    def node(self, key: str) -> TNode:
        for node in self.nodes:
            if node.key == key:
                return node
        raise RewriteGrammarError(f"template has no node {key!r}")


Arities = Mapping[str, int]
"""Arity parameters of a family of templates, e.g. ``{"n": 3}``."""


@dataclass(frozen=True)
class Side:
    """One side of an axiom: a template per dimension and arities, and the arities a host
    node can anchor (the template's first node is the anchor)."""

    template: Callable[[Dim, Arities], Template | None]
    arities: Callable[[Diagram, NodeId, Dim], Iterable[Arities]] = lambda _d, _n, _dim: ({},)


@dataclass(frozen=True)
class Condition:
    """A named side condition on the bindings, beyond the template match."""

    name: str
    description: str
    holds: Callable[[Dim, Bindings], bool]


TEMPLATE_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("template_matches", "the side's template matches up to leg symmetry"),
    SideCondition("parameters_match", "every node's parameter matches its template"),
    SideCondition("same_dimension", "every matched leg carries the anchor's dimension"),
    SideCondition("outside_every_bang_box", "no matched node or leg lies in a bang box"),
    SideCondition("right_side_defined", "every parameter of the built side is defined"),
)
"""The side conditions every template match reports."""


@dataclass(frozen=True)
class TemplateMatch:
    """One occurrence of a template: node, port and rest assignments and the bindings."""

    rule_name: str
    arities: tuple[tuple[str, int], ...]
    nodes: tuple[tuple[str, NodeId], ...]
    terminals: tuple[tuple[str, PortRef], ...]
    rest: tuple[tuple[str, tuple[PortRef, ...]], ...]
    wires: tuple[Wire, ...]
    bindings: tuple[tuple[str, object], ...]
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    value_dim: Dim | None = None
    """The dimension the template was read at: ``shared_dim`` with the diagram's bound
    parameters substituted; None when they are the same."""
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The matched nodes; for an insertion on a wire, the wire's two end nodes."""
        nodes = {node_id for _key, node_id in self.nodes}
        if not nodes:
            nodes = {ref.node_id for _name, ref in self.terminals}
        return tuple(sorted(nodes, key=int))

    @property
    def template_dim(self) -> Dim:
        """The dimension the template is read at."""
        return self.value_dim if self.value_dim is not None else self.shared_dim

    def binding_map(self) -> dict[str, object]:
        return dict(self.bindings)


def _groups(gen: GeneratorType, ins: int, outs: int) -> dict[str, list[tuple[Direction, int]]]:
    """The symmetric leg groups of a node: legs within one group are interchangeable."""
    inputs = [(Direction.INPUT, i) for i in range(ins)]
    outputs = [(Direction.OUTPUT, i) for i in range(outs)]
    name = gen.name
    if name == Z_SPIDER.name:
        return {"legs": inputs + outputs}
    if name in (X_SPIDER.name, W_NODE.name):
        return {"in": inputs, "out": outputs}
    groups: dict[str, list[tuple[Direction, int]]] = {}
    for direction, index in inputs + outputs:
        groups[f"{direction.value}{index}"] = [(direction, index)]
    return groups


_OPEN_GROUPS = {Z_SPIDER.name: ("legs",), X_SPIDER.name: ("in", "out"), W_NODE.name: ("out",)}


def _group_of(gen: GeneratorType, direction: Direction, index: int) -> str:
    name = gen.name
    if name == Z_SPIDER.name:
        return "legs"
    if name in (X_SPIDER.name, W_NODE.name):
        return "in" if direction is Direction.INPUT else "out"
    return f"{direction.value}{index}"


def _boxed_nodes(diagram: Diagram) -> frozenset[NodeId]:
    boxed: set[NodeId] = set()
    for box in diagram.bang_boxes.values():
        boxed |= set(box.node_scope)
        boxed |= {ref.node_id for ref in box.port_scope}
    return frozenset(boxed)


def _partner_map(diagram: Diagram) -> dict[PortRef, PortRef]:
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a] = wire.b
        partner[wire.b] = wire.a
    return partner


def _node_dim(diagram: Diagram, node_id: NodeId) -> Dim | None:
    node = diagram.nodes[node_id]
    for port in (*node.inputs, *node.outputs):
        return port.dim
    return node.phase.dim if node.phase is not None else None


def _legs_have_dim(diagram: Diagram, node_id: NodeId, dim: Dim) -> bool:
    node = diagram.nodes[node_id]
    legs = (*node.inputs, *node.outputs)
    if not legs:
        return node.phase is not None and node.phase.dim == dim
    return all(port.dim == dim for port in legs) and (node.phase is None or node.phase.dim == dim)


def _compatible(diagram: Diagram, node_id: NodeId, tnode: TNode) -> bool:
    node = diagram.nodes[node_id]
    if node.generator_type.name != tnode.gen.name:
        return False
    host = _groups(node.generator_type, node.num_inputs, node.num_outputs)
    want = _groups(tnode.gen, tnode.ins, tnode.outs)
    for group, ports in want.items():
        have = len(host.get(group, []))
        if tnode.open and group in _OPEN_GROUPS.get(tnode.gen.name, ()):
            if have < len(ports):
                return False
        elif have != len(ports):
            return False
    return set(host) <= set(want) or tnode.open


@dataclass
class _State:
    nodes: dict[str, NodeId]
    used: set[NodeId]
    ports: dict[TPort, PortRef]
    used_ports: set[PortRef]
    wires: list[Wire]
    bindings: dict[str, object]


def match_template(
    diagram: Diagram, template: Template, anchor: NodeId, dim: Dim
) -> Iterator[_State]:
    """Every complete assignment of ``template`` anchored at ``anchor`` (its first node)."""
    if not template.nodes:
        return
    partner = _partner_map(diagram)
    boxed = _boxed_nodes(diagram)
    first = template.nodes[0]
    by_key = {node.key: node for node in template.nodes}
    adjacency: dict[str, list[tuple[TPort, TPort]]] = {node.key: [] for node in template.nodes}
    for a, b in template.wires:
        adjacency[a[0]].append((a, b))
        adjacency[b[0]].append((b, a))

    def place(state: _State, key: str, node_id: NodeId) -> _State | None:
        tnode = by_key[key]
        if node_id in state.used or node_id in boxed:
            return None
        if not _compatible(diagram, node_id, tnode):
            return None
        if tnode.port_dims is not None:
            node = diagram.nodes[node_id]
            ins_d, outs_d = tnode.port_dims
            if (
                tuple(p.dim for p in node.inputs) != ins_d
                or tuple(p.dim for p in node.outputs) != outs_d
            ):
                return None
        elif not _legs_have_dim(diagram, node_id, tnode.dim if tnode.dim is not None else dim):
            return None
        own = tnode.dim if tnode.dim is not None else dim
        bound = tnode.param.match(diagram.nodes[node_id].phase, own, state.bindings)
        if bound is None:
            return None
        return _State(
            {**state.nodes, key: node_id},
            state.used | {node_id},
            dict(state.ports),
            set(state.used_ports),
            list(state.wires),
            dict(bound),
        )

    def host_candidates(state: _State, tport: TPort) -> list[PortRef]:
        key, direction, index = tport
        tnode = by_key[key]
        node_id = state.nodes[key]
        node = diagram.nodes[node_id]
        group = _group_of(tnode.gen, direction, index)
        host = _groups(node.generator_type, node.num_inputs, node.num_outputs)[group]
        return [
            PortRef(node_id, d, i)
            for d, i in host
            if PortRef(node_id, d, i) not in state.used_ports
        ]

    pending = [w for w in template.wires]

    seen: set[tuple[object, ...]] = set()

    def extend(state: _State, remaining: list[tuple[TPort, TPort]]) -> Iterator[_State]:
        key = (
            tuple(sorted(state.nodes.items())),
            frozenset(state.used_ports),
            frozenset(state.ports),
        )
        if key in seen:
            return
        seen.add(key)
        if not remaining:
            yield from finish(state)
            return
        # Take a wire with at least one placed endpoint, else place a node of a new component.
        for position, (a, b) in enumerate(remaining):
            if a[0] in state.nodes or b[0] in state.nodes:
                break
        else:
            yield from component(state, remaining)
            return
        a, b = remaining[position]
        if a[0] not in state.nodes:
            a, b = b, a
        rest = remaining[:position] + remaining[position + 1 :]
        a_refs = [state.ports[a]] if a in state.ports else host_candidates(state, a)
        for a_ref in a_refs:
            far = partner.get(a_ref)
            if far is None or far in state.used_ports or far == a_ref:
                continue
            if b[0] in state.nodes:
                if far.node_id != state.nodes[b[0]]:
                    continue
                if b in state.ports:
                    if state.ports[b] != far:
                        continue
                elif far not in host_candidates(state, b):
                    continue
                nxt = _State(
                    dict(state.nodes),
                    set(state.used),
                    {**state.ports, a: a_ref, b: far},
                    state.used_ports | {a_ref, far},
                    state.wires + [Wire(a_ref, far)],
                    dict(state.bindings),
                )
                yield from extend(nxt, rest)
                continue
            placed = place(state, b[0], far.node_id)
            if placed is None or far not in host_candidates(placed, b):
                continue
            placed.ports.update({a: a_ref, b: far})
            placed.used_ports |= {a_ref, far}
            placed.wires.append(Wire(a_ref, far))
            yield from extend(placed, rest)

    def component(state: _State, remaining: list[tuple[TPort, TPort]]) -> Iterator[_State]:
        unplaced = [node for node in template.nodes if node.key not in state.nodes]
        if not unplaced:
            return
        key = unplaced[0].key
        for node_id in sorted(diagram.nodes, key=int):
            placed = place(state, key, node_id)
            if placed is not None:
                yield from extend(placed, remaining)

    def finish(state: _State) -> Iterator[_State]:
        if len(state.nodes) != len(template.nodes):
            yield from component(state, [])
            return
        for tnode in template.nodes:
            if tnode.open:
                continue
            node_id = state.nodes[tnode.key]
            node = diagram.nodes[node_id]
            host = _groups(node.generator_type, node.num_inputs, node.num_outputs)
            want = _groups(tnode.gen, tnode.ins, tnode.outs)
            for group, tports in want.items():
                free_t = [
                    (tnode.key, d, i) for d, i in tports if (tnode.key, d, i) not in state.ports
                ]
                free_h = [
                    PortRef(node_id, d, i)
                    for d, i in host[group]
                    if PortRef(node_id, d, i) not in state.used_ports
                ]
                if len(free_t) != len(free_h):
                    return
                for tport, href in zip(free_t, sorted(free_h, key=_ref_key), strict=True):
                    state.ports[tport] = href
                    state.used_ports.add(href)
        yield state

    start = _State({}, set(), {}, set(), [], {})
    placed = place(start, first.key, anchor)
    if placed is None:
        return
    yield from extend(placed, pending)


def _has_arities(axiom: TemplateAxiom) -> bool:
    """Whether the built side of an insertion depends on arity parameters."""
    try:
        axiom.right.template(Dim.symbol("d"), {})
    except (KeyError, IndexError, ValueError, RewriteDomainError):
        return True
    return False


def _ref_key(ref: PortRef) -> tuple[int, str, int]:
    return (int(ref.node_id), ref.direction.value, ref.index)


@dataclass(frozen=True)
class TemplateAxiom:
    """An equation ``left = right * scalar``, both sides templates; applied in one direction.

    ``scalar`` is the factor the equation states beyond the sides' own scalars and pink
    normalisations. ``conditions`` are checked on the bindings of the matched side.
    ``concrete_only`` restricts matching to a concrete dimension.
    """

    name: str
    left: Side
    right: Side
    scalar: Callable[[Dim, Bindings], Scalar] = lambda _dim, _b: Scalar.one()
    conditions: tuple[Condition, ...] = ()
    concrete_only: bool = False
    description: str = ""
    quantifiers: tuple[str, ...] = ()
    free: Mapping[str, Callable[[Dim, Bindings], object]] = field(default_factory=dict)
    """Variables the built side reads that the matched side does not bind, each with the
    default value a match carries; ``apply ... with NAME=VALUE`` supplies another value,
    which the conditions then check."""
    converse_free: Mapping[str, Callable[[Dim, Bindings], object]] = field(default_factory=dict)
    """The :attr:`free` variables of :meth:`converse`."""

    def converse(self, name: str) -> TemplateAxiom:
        """The same equation read right to left."""
        scalar = self.scalar
        return TemplateAxiom(
            name,
            self.right,
            self.left,
            lambda dim, b: _inverse(scalar(dim, b)),
            self.conditions,
            self.concrete_only,
            f"{self.description} Read right to left.",
            self.quantifiers,
            self.converse_free,
        )

    @property
    def side_conditions(self) -> tuple[SideCondition, ...]:
        return TEMPLATE_SIDE_CONDITIONS + tuple(
            SideCondition(c.name, c.description) for c in self.conditions
        )


def _inverse(value: Scalar) -> Scalar:
    if value.is_zero:
        raise RewriteDomainError("a template rule scalar of 0 has no converse")
    return value**-1


def _pink_norm(template: Template, dim: Dim, legs: Mapping[str, int]) -> Scalar:
    """The product of ``d^((m + n)/2 - 1)`` over the pink nodes, ``legs`` their leg counts."""
    value = sp.Integer(1)
    for node in template.nodes:
        if node.pink:
            base = (node.dim if node.dim is not None else dim).to_sympy()
            value *= sp.Pow(base, sp.Rational(legs[node.key], 2) - 1)
    return Scalar(value)


class TemplatePattern(Pattern):
    """Finds every match of an axiom's left side."""

    locality_radius = 8

    def __init__(self, axiom: TemplateAxiom) -> None:
        self.axiom = axiom

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        if self.axiom.concrete_only and not CONCRETE_AXIOMS:
            return ()
        found: dict[tuple[object, ...], TemplateMatch] = {}
        host = diagram
        diagram = resolved_view(host)
        try:
            insertion = self.axiom.left.template(Dim.symbol("d"), {})
        except (KeyError, IndexError, ValueError, RewriteDomainError):
            insertion = None
        if insertion is not None and not insertion.nodes:
            return self._insertions(host, diagram, insertion)
        for anchor in sorted(diagram.nodes, key=int):
            dim = _node_dim(diagram, anchor)
            host_dim = _node_dim(host, anchor)
            if dim is None or host_dim is None:
                continue
            if self.axiom.concrete_only and not dim.is_concrete:
                continue
            for arities in self.axiom.left.arities(diagram, anchor, dim):
                template = self.axiom.left.template(dim, arities)
                if template is None or not template.nodes:
                    continue
                if diagram.nodes[anchor].generator_type.name != template.nodes[0].gen.name:
                    continue
                for state in match_template(diagram, template, anchor, dim):
                    match = self._to_match(diagram, template, state, dim, arities, host_dim)
                    if match is None:
                        continue
                    key = (frozenset(match.support_node_ids), frozenset(match.wires), match.arities)
                    if key not in found:
                        found[key] = match
        return tuple(sorted(found.values(), key=self.order_key))

    def _insertions(self, host: Diagram, diagram: Diagram, template: Template) -> tuple[Match, ...]:
        """Matches of an empty side: one per wire for a bare wire, else one per dimension."""
        out: list[TemplateMatch] = []
        boxed = _boxed_nodes(diagram)
        if template.bare:
            ((first, second),) = template.bare
            for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
                if wire.a.node_id in boxed or wire.b.node_id in boxed:
                    continue
                node = diagram.nodes[wire.a.node_id]
                dim = node.legs(wire.a.direction)[wire.a.index].dim
                host_node = host.nodes[wire.a.node_id]
                host_dim = host_node.legs(wire.a.direction)[wire.a.index].dim
                if self.axiom.concrete_only and not dim.is_concrete:
                    continue
                for arities in self.axiom.left.arities(diagram, wire.a.node_id, dim):
                    state = _State({}, set(), {}, {wire.a, wire.b}, [wire], {})
                    match = self._to_match(diagram, template, state, dim, arities, host_dim)
                    if match is not None:
                        out.append(replace(match, terminals=((first, wire.a), (second, wire.b))))
            boundary = [*diagram.boundary_inputs, *diagram.boundary_outputs]
            for ref in sorted(boundary, key=_ref_key):
                if ref.node_id in boxed or any(template.nodes) or _has_arities(self.axiom):
                    continue
                node = diagram.nodes[ref.node_id]
                dim = node.legs(ref.direction)[ref.index].dim
                host_dim = host.nodes[ref.node_id].legs(ref.direction)[ref.index].dim
                if self.axiom.concrete_only and not dim.is_concrete:
                    continue
                state = _State({}, set(), {}, {ref}, [], {})
                match = self._to_match(diagram, template, state, dim, {}, host_dim)
                if match is not None:
                    out.append(
                        replace(
                            match,
                            nodes=((_COPY, ref.node_id),),
                            terminals=((first, ref), (second, ref)),
                        )
                    )
            return tuple(out)
        seen: set[Dim] = set()
        for node_id in sorted(diagram.nodes, key=int):
            value = _node_dim(diagram, node_id)
            outer = _node_dim(host, node_id)
            if value is None or outer is None or outer in seen:
                continue
            if self.axiom.concrete_only and not value.is_concrete:
                continue
            seen.add(outer)
            state = _State({}, set(), {}, set(), [], {})
            match = self._to_match(diagram, template, state, value, {}, outer)
            if match is not None:
                out.append(match)
        return tuple(out)

    def verify(self, diagram: Diagram, match: Match) -> bool:
        """Whether ``match`` is a match: its default form is rediscovered and, with its own
        free values, every condition holds and the built side is defined."""
        if not isinstance(match, TemplateMatch):
            return False
        found = self.find_matches(diagram)
        if match in found:
            return True
        if not self.axiom.free:
            return False
        bindings = match.binding_map()
        base = [
            m
            for m in found
            if isinstance(m, TemplateMatch)
            and (m.nodes, m.terminals, m.rest, m.wires, m.arities)
            == (match.nodes, match.terminals, match.rest, match.wires, match.arities)
        ]
        for candidate in base:
            fixed = {k: v for k, v in candidate.bindings if k not in self.axiom.free}
            if any(bindings.get(k) != v for k, v in fixed.items()):
                continue
            if replace(candidate, bindings=match.bindings) != match:
                continue
            return self._holds(match.template_dim, bindings, dict(match.arities))
        return False

    def _holds(self, dim: Dim, bindings: Bindings, arities: Arities) -> bool:
        built = self.axiom.right.template(dim, arities)
        if built is None:
            return False
        try:
            for condition in self.axiom.conditions:
                if not condition.holds(dim, bindings):
                    return False
            for tnode in built.nodes:
                tnode.param.build(dim, bindings)
            if built.scalar is not None:
                built.scalar(dim, bindings)
        except (ArithmeticError, ValueError, KeyError, RewriteDomainError):
            return False
        return True

    def with_values(self, diagram: Diagram, match: Match, values: Bindings) -> TemplateMatch:
        """``match`` with its free variables set to ``values``; raises unless it verifies."""
        if not isinstance(match, TemplateMatch):
            raise RewriteGrammarError(f"{self.axiom.name}: not a template match")
        unknown = sorted(set(values) - set(self.axiom.free))
        if unknown:
            raise RewriteGrammarError(
                f"{self.axiom.name}: {unknown} are not free variables of this rule "
                f"(free: {sorted(self.axiom.free) or 'none'})"
            )
        bindings = {**match.binding_map(), **values}
        changed = replace(match, bindings=tuple(sorted(bindings.items(), key=lambda item: item[0])))
        if not self.verify(diagram, changed):
            raise RewriteDomainError(
                f"{self.axiom.name}: the given values do not satisfy the rule's conditions"
            )
        return changed

    def order_key(self, match: Match) -> tuple[object, ...]:
        assert isinstance(match, TemplateMatch)
        return (
            tuple(int(n) for n in match.support_node_ids),
            match.arities,
            tuple(sorted(w.sort_key() for w in match.wires)),
        )

    def _to_match(
        self,
        diagram: Diagram,
        template: Template,
        state: _State,
        dim: Dim,
        arities: Arities,
        host_dim: Dim,
    ) -> TemplateMatch | None:
        bindings = dict(state.bindings)
        outcomes = [
            SideConditionOutcome("template_matches", True, "every template wire is a host wire"),
            SideConditionOutcome("parameters_match", True, "every parameter matched"),
            SideConditionOutcome("same_dimension", True, f"every matched leg has dimension {dim}"),
            SideConditionOutcome("outside_every_bang_box", True, "no matched node is boxed"),
        ]
        for name, default in self.axiom.free.items():
            try:
                bindings[name] = default(dim, bindings)
            except (ArithmeticError, ValueError, KeyError, RewriteDomainError):
                return None
        built = self.axiom.right.template(dim, arities)
        if built is None:
            return None
        try:
            for tnode in built.nodes:
                tnode.param.build(dim, bindings)
            if built.scalar is not None:
                built.scalar(dim, bindings)
            if template.scalar is not None and template.scalar(dim, bindings).is_zero:
                return None
        except (ArithmeticError, ValueError, KeyError, RewriteDomainError):
            return None
        outcomes.append(
            SideConditionOutcome("right_side_defined", True, "every built parameter is defined")
        )
        rest: list[tuple[str, tuple[PortRef, ...]]] = []
        for tnode in template.nodes:
            if not tnode.open:
                continue
            node_id = state.nodes[tnode.key]
            node = diagram.nodes[node_id]
            refs = [
                PortRef(node_id, d, i)
                for d, count in (
                    (Direction.INPUT, node.num_inputs),
                    (Direction.OUTPUT, node.num_outputs),
                )
                for i in range(count)
                if PortRef(node_id, d, i) not in state.used_ports
            ]
            rest.append((tnode.key, tuple(refs)))
        for condition in self.axiom.conditions:
            try:
                held = condition.holds(dim, bindings)
            except (ArithmeticError, ValueError, KeyError, RewriteDomainError):
                held = False
            if not held:
                return None
            outcomes.append(SideConditionOutcome(condition.name, True, condition.description))
        terminals = tuple(
            sorted(
                ((name, state.ports[tport]) for name, tport in template.terminals.items()),
                key=lambda item: item[0],
            )
        )
        return TemplateMatch(
            rule_name=self.axiom.name,
            arities=tuple(sorted(arities.items())),
            nodes=tuple(sorted(state.nodes.items())),
            terminals=terminals,
            rest=tuple(rest),
            wires=tuple(sorted(state.wires, key=lambda w: w.sort_key())),
            bindings=tuple(sorted(bindings.items(), key=lambda item: item[0])),
            shared_dim=host_dim,
            side_condition_outcomes=tuple(outcomes),
            value_dim=None if host_dim == dim else dim,
        )


def template_scalar(axiom: TemplateAxiom, match: Match, diagram: Diagram | None = None) -> Scalar:
    """The scalar applying ``axiom`` at ``match`` introduces."""
    if not isinstance(match, TemplateMatch):
        raise RewriteGrammarError(
            f"{axiom.name} requires a TemplateMatch, got {type(match).__name__}"
        )
    dim = match.template_dim
    host_dim = match.shared_dim
    bindings = match.binding_map()
    arities = dict(match.arities)
    left = axiom.left.template(dim, arities)
    right = axiom.right.template(dim, arities)
    assert left is not None and right is not None
    rest = dict(match.rest)
    left_legs = {
        node.key: node.ins + node.outs + len(rest.get(node.key, ())) for node in left.nodes
    }
    right_legs = _built_legs(left, right, rest)
    value = axiom.scalar(dim, bindings)
    value = value * _pink_norm(right, host_dim, right_legs)
    value = value * _inverse(_pink_norm(left, host_dim, left_legs))
    if right.scalar is not None:
        value = value * right.scalar(dim, bindings)
    if left.scalar is not None:
        value = value * _inverse(left.scalar(dim, bindings))
    return value


def _built_legs(
    left: Template, right: Template, rest: Mapping[str, tuple[PortRef, ...]]
) -> dict[str, int]:
    counts = {node.key: node.ins + node.outs for node in right.nodes}
    for left_key, right_key in right.rest.items():
        counts[right_key] += len(rest.get(left_key, ()))
    del left
    return counts


def template_builder(axiom: TemplateAxiom) -> Callable[[Diagram, Match], BuildResult]:
    """The right-hand builder of ``axiom``."""

    def build(diagram: Diagram, match: Match) -> BuildResult:
        if not isinstance(match, TemplateMatch) or match.rule_name != axiom.name:
            raise RewriteGrammarError(f"{axiom.name}: builder requires its own TemplateMatch")
        check_side_condition_coverage(match, axiom.side_conditions, f"{axiom.name}_builder")
        if not TemplatePattern(axiom).verify(diagram, match):
            raise RewriteDomainError(
                f"{axiom.name}_builder: the match is not among those rediscovered in this diagram"
            )
        return _build(diagram, axiom, match)

    build.side_conditions = axiom.side_conditions  # type: ignore[attr-defined]
    return build


_COPY = "_copy"
"""The node key of a boundary insertion: the node owning the boundary port, rebuilt as is."""


def _build_boundary_insertion(
    diagram: Diagram, axiom: TemplateAxiom, match: TemplateMatch
) -> BuildResult:
    """A right side inserted between a boundary port and the boundary: the port's node is
    rebuilt unchanged, the port wired to one terminal and the other put on the boundary."""
    ((_key, owner_id),) = match.nodes
    (first, port), (second, _same) = match.terminals
    right = axiom.right.template(match.template_dim, dict(match.arities))
    if right is None:
        raise RewriteDomainError(f"{axiom.name}: no right-hand template")
    owner = diagram.nodes[owner_id]
    copy = diagram.add_node(
        owner.generator_type,
        [p.dim for p in owner.inputs],
        [p.dim for p in owner.outputs],
        phase=owner.phase,
    )
    port_mapping: dict[PortRef, PortRef] = {}
    for d, count in ((Direction.INPUT, owner.num_inputs), (Direction.OUTPUT, owner.num_outputs)):
        for k in range(count):
            old = PortRef(owner_id, d, k)
            if old != port:
                port_mapping[old] = PortRef(copy, d, k)
    dim = owner.legs(port.direction)[port.index].dim
    new_ids = [copy]
    port_of: dict[TPort, PortRef] = {}
    for tnode in right.nodes:
        node_id = diagram.add_node(
            tnode.gen,
            [dim] * tnode.ins,
            [dim] * tnode.outs,
            phase=_redim(tnode.param.build(match.template_dim, match.binding_map()), dim),
        )
        new_ids.append(node_id)
        for d, count in ((Direction.INPUT, tnode.ins), (Direction.OUTPUT, tnode.outs)):
            for k in range(count):
                port_of[(tnode.key, d, k)] = PortRef(node_id, d, k)
    outer = port_of[right.terminals[second]]
    inner = port_of[right.terminals[first]]
    if outer.direction is not port.direction:
        outer, inner = inner, outer
    if outer.direction is not port.direction:
        raise RewriteDomainError(f"{axiom.name}: no port of the inserted side faces the boundary")
    port_mapping[port] = outer
    new_wires = [Wire(PortRef(copy, port.direction, port.index), inner)]
    new_wires += [Wire(port_of[a], port_of[b]) for a, b in right.wires]
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=tuple(new_ids),
        consumed_node_ids=(owner_id,),
        consumed_wires=(),
        port_mapping=MappingProxyType(port_mapping),
        scalar_introduced=template_scalar(axiom, match, diagram),
        new_wires=tuple(new_wires),
    )


def _build(diagram: Diagram, axiom: TemplateAxiom, match: TemplateMatch) -> BuildResult:
    if match.nodes and match.nodes[0][0] == _COPY:
        return _build_boundary_insertion(diagram, axiom, match)
    value_dim = match.template_dim
    dim = match.shared_dim
    bindings = match.binding_map()
    arities = dict(match.arities)
    right = axiom.right.template(value_dim, arities)
    if right is None:
        raise RewriteDomainError(f"{axiom.name}: no right-hand template at {dim}")
    terminals = dict(match.terminals)
    rest = dict(match.rest)
    boundary_dir: dict[PortRef, Direction] = {}
    for ref in diagram.boundary_inputs:
        boundary_dir[ref] = Direction.INPUT
    for ref in diagram.boundary_outputs:
        boundary_dir[ref] = Direction.OUTPUT
    partner = _partner_map(diagram)

    # Which right-hand terminal ports face a host boundary entry, and with what direction.
    wanted: dict[TPort, Direction] = {}
    for name, tport in right.terminals.items():
        host = terminals.get(name)
        if host is not None and host in boundary_dir:
            wanted[tport] = boundary_dir[host]

    receivers: dict[str, list[str]] = {}
    for left_key, right_key in right.rest.items():
        receivers.setdefault(right_key, []).append(left_key)
    new_ids: list[NodeId] = []
    port_of: dict[TPort, PortRef] = {}
    port_mapping: dict[PortRef, PortRef] = {}
    adapters: list[tuple[PortRef, PortRef]] = []
    for tnode in right.nodes:
        legs: list[tuple[Direction, TPort | PortRef]] = []
        for d, count in ((Direction.INPUT, tnode.ins), (Direction.OUTPUT, tnode.outs)):
            for i in range(count):
                tport = (tnode.key, d, i)
                direction = d
                if tnode.gen.name == Z_SPIDER.name and tport in wanted:
                    direction = wanted[tport]
                legs.append((direction, tport))
        received = [ref for key in receivers.get(tnode.key, ()) for ref in rest.get(key, ())]
        for ref in received:
            direction = ref.direction
            if tnode.gen.name == W_NODE.name:
                direction = Direction.OUTPUT
            legs.append((direction, ref))
        ins = [leg for leg in legs if leg[0] is Direction.INPUT]
        outs = [leg for leg in legs if leg[0] is Direction.OUTPUT]
        own = tnode.dim if tnode.dim is not None else value_dim
        phase = tnode.param.build(own, bindings)
        if tnode.dim is None:
            phase = _redim(phase, dim)
        if (
            phase is None
            and not ins
            and not outs
            and tnode.gen.name
            in (
                Z_SPIDER.name,
                X_SPIDER.name,
            )
        ):
            phase = PhaseVector(dim)
        if tnode.port_dims is not None:
            in_dims, out_dims = list(tnode.port_dims[0]), list(tnode.port_dims[1])
        else:
            leg = tnode.dim if tnode.dim is not None else dim
            in_dims, out_dims = [leg] * len(ins), [leg] * len(outs)
        node_id = diagram.add_node(tnode.gen, in_dims, out_dims, phase=phase)
        new_ids.append(node_id)
        for d, group in ((Direction.INPUT, ins), (Direction.OUTPUT, outs)):
            for index, (_direction, item) in enumerate(group):
                ref = PortRef(node_id, d, index)
                if isinstance(item, PortRef):
                    port_mapping[item] = ref
                else:
                    port_of[item] = ref

    consumed_ids = tuple(node_id for _key, node_id in match.nodes)
    consumed = set(consumed_ids)
    for name, tport in right.terminals.items():
        host = terminals[name]
        target = port_of[tport]
        if host.node_id not in consumed:
            adapters.append((host, target))
            continue
        if host in boundary_dir and boundary_dir[host] is not target.direction:
            own = _port_dim(diagram, host)
            adapter = diagram.add_node(Z_SPIDER, [own], [own])
            new_ids.append(adapter)
            outer = PortRef(adapter, boundary_dir[host], 0)
            inner_direction = (
                Direction.OUTPUT if boundary_dir[host] is Direction.INPUT else Direction.INPUT
            )
            port_mapping[host] = outer
            adapters.append((PortRef(adapter, inner_direction, 0), target))
        else:
            port_mapping[host] = target

    consumed_wires = list(match.wires)
    for first, second in right.bare:
        h1, h2 = terminals[first], terminals[second]
        far1, far2 = partner.get(h1), partner.get(h2)
        if far1 is not None and far1.node_id not in consumed and far1 != h2:
            consumed_wires.append(Wire(h1, far1))
            port_mapping[h2] = far1
        elif far2 is not None and far2.node_id not in consumed and far2 != h1:
            consumed_wires.append(Wire(h2, far2))
            port_mapping[h1] = far2
        elif h1 in boundary_dir and h2 in boundary_dir:
            dirs = [boundary_dir[h1], boundary_dir[h2]]
            ins_n = sum(1 for d in dirs if d is Direction.INPUT)
            own = _port_dim(diagram, h1)
            node_id = diagram.add_node(Z_SPIDER, [own] * ins_n, [own] * (2 - ins_n))
            new_ids.append(node_id)
            counters = {Direction.INPUT: 0, Direction.OUTPUT: 0}
            for host, d in ((h1, dirs[0]), (h2, dirs[1])):
                port_mapping[host] = PortRef(node_id, d, counters[d])
                counters[d] += 1
        else:
            raise RewriteDomainError(
                f"{axiom.name}: a bare wire of the right-hand side closes on the match itself"
            )

    new_wires = [Wire(port_of[a], port_of[b]) for a, b in right.wires]
    new_wires += [Wire(a, b) for a, b in adapters]
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=tuple(new_ids),
        consumed_node_ids=tuple(sorted(consumed_ids, key=int)),
        consumed_wires=tuple(consumed_wires),
        port_mapping=MappingProxyType(port_mapping),
        scalar_introduced=template_scalar(axiom, match, diagram),
        new_wires=tuple(new_wires),
    )


def _port_dim(diagram: Diagram, ref: PortRef) -> Dim:
    node = diagram.nodes[ref.node_id]
    return node.legs(ref.direction)[ref.index].dim


def _redim(phase: PhaseVector | None, dim: Dim) -> PhaseVector | None:
    """``phase`` read over the host dimension ``dim``."""
    if phase is None or phase.dim == dim:
        return phase
    return PhaseVector(
        dim,
        phase.entries(),
        phase.rule,
        weights=phase.weights(),
        weight_rule=phase.weight_rule,
    )


CONCRETE_AXIOMS = True
"""Whether template matching substitutes bound dimensions and matches the concrete-only
axioms; when False, neither."""


def resolved_view(diagram: Diagram) -> Diagram:
    """``diagram`` with every dimension symbol its parameter environment binds substituted,
    or ``diagram`` itself when :data:`CONCRETE_AXIOMS` is False; node ids preserved."""
    if not CONCRETE_AXIOMS:
        return diagram
    names: set[str] = set()
    for node in diagram.nodes.values():
        for port in (*node.inputs, *node.outputs):
            names |= port.dim.free_symbols
        if node.phase is not None:
            names |= node.phase.dim.free_symbols
    mapping = {name: value for name, value in diagram.parameters.items() if name in names}
    return diagram.substitute(mapping) if mapping else diagram


def _follow_port_scopes(diagram: Diagram, port_mapping: Mapping[PortRef, PortRef]) -> None:
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if not any(ref in port_mapping for ref in box.port_scope):
            continue
        diagram.set_bang_box_port_scope(
            box_id, frozenset(port_mapping.get(ref, ref) for ref in box.port_scope)
        )


# -- template construction helpers ------------------------------------------------------


def legs_of(diagram: Diagram, node_id: NodeId) -> tuple[int, int]:
    """The input and output counts of a host node."""
    node = diagram.nodes[node_id]
    return node.num_inputs, node.num_outputs


def neighbours(diagram: Diagram, node_id: NodeId) -> list[NodeId]:
    """The nodes wired to ``node_id``, in port order, with repetition."""
    partner = _partner_map(diagram)
    node = diagram.nodes[node_id]
    out: list[NodeId] = []
    for d, count in ((Direction.INPUT, node.num_inputs), (Direction.OUTPUT, node.num_outputs)):
        for i in range(count):
            far = partner.get(PortRef(node_id, d, i))
            if far is not None:
                out.append(far.node_id)
    return out


def ranges(*bounds: Sequence[int]) -> Iterator[tuple[int, ...]]:
    """The cartesian product of the given integer sequences."""
    return itertools.product(*bounds)


__all__ = [
    "PHASELESS",
    "TEMPLATE_SIDE_CONDITIONS",
    "Arities",
    "Bindings",
    "Condition",
    "Derived",
    "Fixed",
    "Kj",
    "Param",
    "Side",
    "TNode",
    "Template",
    "TemplateAxiom",
    "TemplateMatch",
    "TemplatePattern",
    "Var",
    "k_vector",
    "kj_index",
    "legs_of",
    "match_template",
    "neighbours",
    "template_builder",
    "template_scalar",
]
