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

"""Mixed diagrams: conjugation, doubling, purification, and the standard CPM constructions.

A mixed diagram is a :class:`~archytaszx.diagram.graph.Diagram` that may contain discard nodes
``G``. Its meaning is the pure diagram :func:`double` builds: ``D`` beside ``conj(D)``, the two
copies of every ``G`` leg joined by one wire, boundary ``D0`` outputs, ``D1`` outputs, ``D0``
inputs, ``D1`` inputs, scalar ``c * conj(c)``. With ``T`` that diagram's tensor of axes
``(o, o', i, i')``, ``D`` is the CPM ``rho -> sum_{i, i'} T[o, o', i, i'] rho[i, i']``.

Conjugation per generator: Z phase ``alpha`` -> ``-alpha``, Z weights ``a`` -> ``conj(a)``; X
phase ``alpha_k`` -> ``-alpha_{-k mod d}``, X weights ``a_k`` -> ``conj(a_{-k mod d})``; ``F`` ->
``F^3``; ``T``, ``Ti``, ``W``, ``B``, ``S``, ``G`` unchanged.

A ``G`` leg wired to another node doubles to a direct wire between the two copies of that
node's port, or, for a ``G`` in a bang box, to a phase-free two-leg Z in that box. A ``G`` leg
on the boundary doubles to a phase-free two-leg Z (the cup or cap); a ``G`` wired to a ``G``
doubles to a phase-free zero-leg Z over the leg dimension (the scalar ``d``). Every doubled or
purified node keeps a map back to the node of ``D`` it came from.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import INDEX, Phase, PhaseVector
from archytaszx.algebra.scalar import ModDelta, Scalar
from archytaszx.diagram.bangbox import BangBox
from archytaszx.diagram.generators import (
    DISCARD,
    FOURIER_BOX,
    MIXED_ONLY,
    X_SPIDER,
    Z_SPIDER,
)
from archytaszx.diagram.graph import (
    BangBoxId,
    Diagram,
    Direction,
    Node,
    NodeId,
    PortRef,
)


class MixedError(Exception):
    """Base class for all errors raised by this module."""


class MixedDomainError(MixedError):
    """A diagram is outside the domain an operation here requires.

    Raised for a ``G`` leg that is neither wired nor on the boundary, for a leg index out of
    range, and for a mixture whose states and probabilities disagree in number or shape.
    """


class Sheet(enum.Enum):
    """Which copy of a doubled diagram a node belongs to."""

    PLAIN = "plain"
    CONJUGATE = "conjugate"
    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class Origin:
    """Where a node of a doubled or purified diagram came from: a sheet and a node of ``D``."""

    sheet: Sheet
    node: NodeId


@dataclass(frozen=True, slots=True)
class Doubled:
    """``double(D)`` with the origin of each of its nodes."""

    diagram: Diagram
    origin: Mapping[NodeId, Origin] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Purified:
    """``purify(D)``: the pure diagram, the boundary-output positions of the environment, and
    the origin of each node."""

    diagram: Diagram
    env: tuple[int, ...]
    origin: Mapping[NodeId, NodeId] = field(default_factory=dict)

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Unpack as ``(diagram, env)``."""
        return iter((self.diagram, self.env))


def is_pure(diagram: Diagram) -> bool:
    """True iff ``diagram`` has no node of a mixed-only generator."""
    return not any(n.generator_type.name in MIXED_ONLY for n in diagram.nodes.values())


def discards(diagram: Diagram) -> tuple[NodeId, ...]:
    """The ``G`` nodes of ``diagram``, by ascending id."""
    return tuple(
        sorted(i for i, n in diagram.nodes.items() if n.generator_type.name == DISCARD.name)
    )


# -- conjugation ----------------------------------------------------------------------------


def _reflect(expr: sp.Expr, dim: Dim) -> sp.Expr:
    """``expr`` with ``INDEX`` replaced by ``d - INDEX``, zero at ``INDEX = 0``."""
    reflected = expr.xreplace({INDEX: dim.to_sympy() - INDEX})
    return sp.expand((1 - ModDelta(INDEX, dim.to_sympy())) * reflected)


def conj_phase_vector(phase: PhaseVector, *, reflected: bool) -> PhaseVector:
    """The phase vector of the conjugate node: negated phases and conjugated weights, at
    index ``-k mod d`` when ``reflected`` (the X spider)."""
    dim = phase.dim
    if not reflected:
        return PhaseVector(
            dim,
            entries={k: -p for k, p in phase.entries().items()},
            rule=None if phase.rule is None else -phase.rule,
            expand=False,
            weights={k: w.conjugate() for k, w in phase.weights().items()},
            weight_rule=None if phase.weight_rule is None else phase.weight_rule.conjugate(),
        )
    if dim.is_concrete:
        d = dim.to_int()
        entries = {(d - k) % d: -p for k, p in phase.entries().items()}
        weights = {(d - k) % d: w.conjugate() for k, w in phase.weights().items()}
        rule = None if phase.rule is None else Phase(-_reflect(phase.rule.to_sympy_turns(), dim))
        weight_rule = (
            None
            if phase.weight_rule is None
            else Scalar(sp.conjugate(_reflect(phase.weight_rule.to_sympy(), dim)))
        )
        return PhaseVector(
            dim, entries=entries, rule=rule, expand=False, weights=weights, weight_rule=weight_rule
        )
    gauge = 1 - ModDelta(INDEX, dim.to_sympy())
    turns_r = sp.Integer(0)
    if phase.rule is not None:
        turns_r -= phase.rule.to_sympy_turns().xreplace({INDEX: dim.to_sympy() - INDEX})
    for k, p in phase.entries().items():
        turns_r -= ModDelta(INDEX + k, dim.to_sympy()) * p.to_sympy_turns()
    rule_out = None if turns_r == 0 else Phase(sp.expand(gauge * turns_r))
    weight_rule_out: Scalar | None = None
    if phase.is_weighted:
        base = sp.Integer(1) if phase.weight_rule is None else phase.weight_rule.to_sympy()
        expr = base.xreplace({INDEX: dim.to_sympy() - INDEX})
        for k, w in phase.weights().items():
            base_k = base.xreplace({INDEX: sp.Integer(k)})
            expr += ModDelta(INDEX + k, dim.to_sympy()) * (w.to_sympy() - base_k)
        expr = gauge * expr + ModDelta(INDEX, dim.to_sympy())
        weight_rule_out = Scalar(sp.conjugate(sp.expand(expr)))
    return PhaseVector(dim, rule=rule_out, expand=False, weight_rule=weight_rule_out)


def conj_node_phase(node: Node) -> PhaseVector | None:
    """The phase slot of ``conj(node)`` for a Z or X spider; ``node.phase`` otherwise."""
    if node.phase is None:
        return None
    name = node.generator_type.name
    if name == Z_SPIDER.name:
        return conj_phase_vector(node.phase, reflected=False)
    if name == X_SPIDER.name:
        return conj_phase_vector(node.phase, reflected=True)
    return node.phase


def _replace_ref(diagram: Diagram, old: PortRef, new: PortRef) -> None:
    """Move whatever ``old`` is attached to (a wire end or a boundary slot) onto ``new``."""
    for wire in diagram.wires:
        if old in (wire.a, wire.b):
            other = wire.b if wire.a == old else wire.a
            diagram.remove_wire(wire.a, wire.b)
            diagram.add_wire(new, other)
    diagram.set_boundary_inputs([new if r == old else r for r in diagram.boundary_inputs])
    diagram.set_boundary_outputs([new if r == old else r for r in diagram.boundary_outputs])


def _box_of(diagram: Diagram, node_id: NodeId) -> BangBoxId | None:
    """The innermost node-scope bang box holding ``node_id``, or None."""
    holding = [b for b in diagram.bang_boxes.values() if node_id in b.node_scope]
    if not holding:
        return None
    depth = {b.id: _depth(diagram, b) for b in holding}
    return max(holding, key=lambda b: depth[b.id]).id


def _depth(diagram: Diagram, box: BangBox) -> int:
    count = 0
    parent = box.parent
    while parent is not None:
        count += 1
        parent = diagram.bang_boxes[parent].parent
    return count


def _add_to_box(diagram: Diagram, box: BangBoxId | None, node_ids: Iterable[NodeId]) -> None:
    """Add ``node_ids`` to ``box`` and to every box enclosing it."""
    while box is not None:
        b = diagram.bang_boxes[box]
        diagram.set_bang_box_node_scope(box, b.node_scope | frozenset(node_ids))
        box = b.parent


def conj_with_origin(diagram: Diagram) -> tuple[Diagram, dict[NodeId, NodeId]]:
    """``conj(diagram)`` and, for each of its nodes, the node of ``diagram`` it came from."""
    out = diagram.copy()
    origin = {i: i for i in diagram.nodes}
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        name = node.generator_type.name
        if name in (Z_SPIDER.name, X_SPIDER.name):
            out.set_phase(node_id, conj_node_phase(node))
        elif name == FOURIER_BOX.name:
            dim = node.inputs[0].dim
            second = out.add_node(FOURIER_BOX, [dim], [dim])
            third = out.add_node(FOURIER_BOX, [dim], [dim])
            _replace_ref(
                out,
                PortRef(node_id, Direction.OUTPUT, 0),
                PortRef(third, Direction.OUTPUT, 0),
            )
            out.add_wire(PortRef(node_id, Direction.OUTPUT, 0), PortRef(second, Direction.INPUT, 0))
            out.add_wire(PortRef(second, Direction.OUTPUT, 0), PortRef(third, Direction.INPUT, 0))
            origin[second] = origin[third] = node_id
            _add_to_box(out, _box_of(diagram, node_id), (second, third))
    out.set_scalar(diagram.scalar.conjugate())
    return out, origin


def conj(diagram: Diagram) -> Diagram:
    """The entrywise complex conjugate of ``diagram``. See the module docstring."""
    return conj_with_origin(diagram)[0]


# -- assembling ------------------------------------------------------------------------------


class _Builder:
    """A fresh diagram assembled from copies of other diagrams' nodes."""

    def __init__(self) -> None:
        self.diagram = Diagram()
        self.origin: dict[NodeId, Origin] = {}

    def add_copy(
        self,
        source: Diagram,
        sheet: Sheet,
        *,
        skip: frozenset[NodeId] = frozenset(),
        origin: Mapping[NodeId, NodeId] | None = None,
    ) -> dict[NodeId, NodeId]:
        """Copy every node of ``source`` outside ``skip`` and the wires among them."""
        ids: dict[NodeId, NodeId] = {}
        for node_id in sorted(source.nodes):
            if node_id in skip:
                continue
            node = source.nodes[node_id]
            new = self.diagram.add_node(
                node.generator_type,
                [p.dim for p in node.inputs],
                [p.dim for p in node.outputs],
                node.phase,
            )
            ids[node_id] = new
            back = node_id if origin is None else origin[node_id]
            self.origin[new] = Origin(sheet, back)
        for wire in sorted(source.wires, key=lambda w: w.sort_key()):
            if wire.a.node_id in ids and wire.b.node_id in ids:
                self.diagram.add_wire(_moved(wire.a, ids), _moved(wire.b, ids))
        return ids

    def cup(
        self, dim: Dim, direction: Direction, legs: int, back: NodeId, sheet: Sheet = Sheet.DISCARD
    ) -> NodeId:
        """A phase-free Z with ``legs`` legs in ``direction`` (zero legs: phase over ``dim``)."""
        inputs = [dim] * legs if direction is Direction.INPUT else []
        outputs = [dim] * legs if direction is Direction.OUTPUT else []
        phase = PhaseVector(dim) if legs == 0 else None
        new = self.diagram.add_node(Z_SPIDER, inputs, outputs, phase)
        self.origin[new] = Origin(sheet, back)
        return new


def _moved(ref: PortRef, ids: Mapping[NodeId, NodeId]) -> PortRef:
    return PortRef(ids[ref.node_id], ref.direction, ref.index)


def _partner(diagram: Diagram, ref: PortRef) -> PortRef | None:
    """The port wired to ``ref``, or None."""
    for wire in diagram.wires:
        if wire.a == ref:
            return wire.b
        if wire.b == ref:
            return wire.a
    return None


def _leg(diagram: Diagram, g: NodeId) -> PortRef:
    node = diagram.nodes[g]
    if node.inputs:
        return PortRef(g, Direction.INPUT, 0)
    return PortRef(g, Direction.OUTPUT, 0)


def _leg_dim(diagram: Diagram, g: NodeId) -> Dim:
    node = diagram.nodes[g]
    return (node.inputs or node.outputs)[0].dim


def _boundary_side(diagram: Diagram, ref: PortRef) -> tuple[str, int] | None:
    if ref in diagram.boundary_outputs:
        return ("out", diagram.boundary_outputs.index(ref))
    if ref in diagram.boundary_inputs:
        return ("in", diagram.boundary_inputs.index(ref))
    return None


def _copy_boxes(
    source: Diagram,
    target: Diagram,
    maps: Sequence[Mapping[NodeId, NodeId]],
    extra: Mapping[NodeId, Sequence[NodeId]],
) -> None:
    """Each node-scope box of ``source`` as one box over its images under every map, each
    port-scope box as one box per map, all sharing the source box's multiplicity."""
    order = sorted(source.bang_boxes.values(), key=lambda b: (_depth(source, b), b.id))
    new_id: dict[BangBoxId, BangBoxId] = {}
    port_ids: dict[BangBoxId, list[BangBoxId]] = {}
    for box in order:
        parent = None if box.parent is None else new_id.get(box.parent)
        if box.is_node_scope:
            scope: set[NodeId] = set()
            for m in maps:
                scope |= {m[n] for n in box.node_scope if n in m}
            for n in box.node_scope:
                scope |= set(extra.get(n, ()))
            new_id[box.id] = target.add_bang_box(
                box.multiplicity, node_scope=frozenset(scope), parent=parent
            )
        else:
            port_ids[box.id] = []
            for m in maps:
                ports = frozenset(
                    PortRef(m[r.node_id], r.direction, r.index)
                    for r in box.port_scope
                    if r.node_id in m
                )
                if ports:
                    port_ids[box.id].append(
                        target.add_bang_box(box.multiplicity, port_scope=ports, parent=parent)
                    )


def double_with_origin(diagram: Diagram) -> Doubled:
    """``double(diagram)`` with the origin of each node. See the module docstring."""
    conjugate, conj_origin = conj_with_origin(diagram)
    gs = frozenset(discards(diagram))
    b = _Builder()
    ids0 = b.add_copy(diagram, Sheet.PLAIN, skip=gs)
    ids1 = b.add_copy(conjugate, Sheet.CONJUGATE, skip=gs, origin=conj_origin)
    out = b.diagram
    bnd: dict[PortRef, tuple[PortRef, PortRef]] = {}
    extra: dict[NodeId, list[NodeId]] = {}
    done: set[NodeId] = set()
    for g in sorted(gs):
        if g in done:
            continue
        done.add(g)
        leg = _leg(diagram, g)
        dim = _leg_dim(diagram, g)
        partner = _partner(diagram, leg)
        if partner is not None and partner.node_id not in gs:
            twin = _partner(conjugate, leg)
            assert twin is not None
            if _box_of(diagram, g) is None:
                out.add_wire(_moved(partner, ids0), _moved(twin, ids1))
                continue
            cap = b.cup(dim, Direction.INPUT, 2, g)
            out.add_wire(_moved(partner, ids0), PortRef(cap, Direction.INPUT, 0))
            out.add_wire(_moved(twin, ids1), PortRef(cap, Direction.INPUT, 1))
            extra[g] = [cap]
            continue
        if partner is not None:
            done.add(partner.node_id)
            loop = b.cup(dim, Direction.INPUT, 0, g)
            extra[g] = [loop]
            continue
        if _boundary_side(diagram, leg) is None:
            raise MixedDomainError(f"discard {g!r} has a leg that is neither wired nor open")
        cup = b.cup(dim, leg.direction, 2, g)
        extra[g] = [cup]
        bnd[leg] = (PortRef(cup, leg.direction, 0), PortRef(cup, leg.direction, 1))

    def side(refs: Sequence[PortRef], sheet: int) -> list[PortRef]:
        ids = ids0 if sheet == 0 else ids1
        return [bnd[r][sheet] if r in bnd else _moved(r, ids) for r in refs]

    out.set_boundary_outputs(
        side(diagram.boundary_outputs, 0) + side(conjugate.boundary_outputs, 1)
    )
    out.set_boundary_inputs(side(diagram.boundary_inputs, 0) + side(conjugate.boundary_inputs, 1))
    out.set_scalar(diagram.scalar * diagram.scalar.conjugate())
    out.set_parameters(dict(diagram.parameters))
    _copy_boxes(conjugate, out, (ids0, ids1), extra)
    return Doubled(out, dict(b.origin))


def double(diagram: Diagram) -> Diagram:
    """The doubled pure diagram of ``diagram``. See the module docstring."""
    return double_with_origin(diagram).diagram


# -- purification ----------------------------------------------------------------------------


def purify(diagram: Diagram) -> Purified:
    """The Stinespring form: ``diagram`` with every ``G`` removed and its leg moved to the end of
    the boundary outputs (the environment), so that ``diagram == channel(*purify(diagram))``.

    A ``G`` wired to an input port, or on a boundary output, becomes a cup with one leg there
    and one in the environment; a ``G`` on a boundary input becomes an identity Z from there
    to the environment; a ``G-G`` pair becomes a cup with both legs in the environment.
    """
    gs = frozenset(discards(diagram))
    b = _Builder()
    ids = b.add_copy(diagram, Sheet.PLAIN, skip=gs)
    out = b.diagram
    origin = {n: o.node for n, o in b.origin.items()}
    env: list[PortRef] = []
    bnd: dict[PortRef, PortRef] = {}
    extra: dict[NodeId, list[NodeId]] = {}
    done: set[NodeId] = set()
    for g in sorted(gs):
        if g in done:
            continue
        done.add(g)
        leg = _leg(diagram, g)
        dim = _leg_dim(diagram, g)
        partner = _partner(diagram, leg)
        if partner is not None and partner.node_id not in gs:
            boxed = _box_of(diagram, g) is not None
            if partner.direction is Direction.OUTPUT and not boxed:
                env.append(_moved(partner, ids))
                continue
            if partner.direction is Direction.OUTPUT:
                cup = out.add_node(Z_SPIDER, [dim], [dim])
                out.add_wire(_moved(partner, ids), PortRef(cup, Direction.INPUT, 0))
                env.append(PortRef(cup, Direction.OUTPUT, 0))
                origin[cup] = g
                extra[g] = [cup]
                continue
            cup = b.cup(dim, Direction.OUTPUT, 2, g, Sheet.PLAIN)
            out.add_wire(PortRef(cup, Direction.OUTPUT, 0), _moved(partner, ids))
            env.append(PortRef(cup, Direction.OUTPUT, 1))
        elif partner is not None:
            done.add(partner.node_id)
            cup = b.cup(dim, Direction.OUTPUT, 2, g, Sheet.PLAIN)
            env += [PortRef(cup, Direction.OUTPUT, 0), PortRef(cup, Direction.OUTPUT, 1)]
        else:
            where = _boundary_side(diagram, leg)
            if where is None:
                raise MixedDomainError(f"discard {g!r} has a leg that is neither wired nor open")
            if where[0] == "out":
                cup = b.cup(dim, Direction.OUTPUT, 2, g, Sheet.PLAIN)
                bnd[leg] = PortRef(cup, Direction.OUTPUT, 0)
                env.append(PortRef(cup, Direction.OUTPUT, 1))
            else:
                cup = out.add_node(Z_SPIDER, [dim], [dim])
                bnd[leg] = PortRef(cup, Direction.INPUT, 0)
                env.append(PortRef(cup, Direction.OUTPUT, 0))
        origin[cup] = g
        extra[g] = [cup]

    def side(refs: Sequence[PortRef]) -> list[PortRef]:
        return [bnd[r] if r in bnd else _moved(r, ids) for r in refs]

    outputs = side(diagram.boundary_outputs)
    out.set_boundary_outputs(outputs + env)
    out.set_boundary_inputs(side(diagram.boundary_inputs))
    out.set_scalar(diagram.scalar)
    out.set_parameters(dict(diagram.parameters))
    _copy_boxes(diagram, out, (ids,), extra)
    return Purified(out, tuple(range(len(outputs), len(outputs) + len(env))), origin)


def channel(diagram: Diagram, env: Iterable[int]) -> Diagram:
    """``diagram`` with the boundary outputs at positions ``env`` discarded (removed from the
    boundary and capped by ``G``)."""
    out = diagram.copy()
    outputs = list(diagram.boundary_outputs)
    positions = sorted(set(env))
    for p in positions:
        if not 0 <= p < len(outputs):
            raise MixedDomainError(f"output position {p} out of range 0..{len(outputs) - 1}")
    for p in positions:
        ref = outputs[p]
        node = diagram.nodes[ref.node_id]
        dim = node.legs(ref.direction)[ref.index].dim
        g = out.add_node(DISCARD, [dim], [])
        out.add_wire(ref, PortRef(g, Direction.INPUT, 0))
        _add_to_box(out, _box_of(diagram, ref.node_id), (g,))
    out.set_boundary_outputs([r for i, r in enumerate(outputs) if i not in set(positions)])
    return out


def partial_trace(diagram: Diagram, legs: Iterable[int]) -> Diagram:
    """``diagram`` with ``G`` on the boundary outputs at positions ``legs``."""
    return channel(diagram, legs)


# -- standard constructions ------------------------------------------------------------------


def _as_dim(d: Dim | int | str) -> Dim:
    if isinstance(d, Dim):
        return d
    if isinstance(d, int):
        return Dim.concrete(d)
    return Dim.symbol(d)


def discard(d: Dim | int | str) -> Diagram:
    """The discard effect ``G`` on one input of dimension ``d``: the trace."""
    dim = _as_dim(d)
    out = Diagram()
    g = out.add_node(DISCARD, [dim], [])
    out.set_boundary_inputs([PortRef(g, Direction.INPUT, 0)])
    return out


def maximally_mixed(d: Dim | int | str) -> Diagram:
    """The state ``G`` with pure scalar ``1/sqrt(d)``: the CPM ``id / d``."""
    dim = _as_dim(d)
    out = Diagram()
    g = out.add_node(DISCARD, [], [dim])
    out.set_boundary_outputs([PortRef(g, Direction.OUTPUT, 0)])
    out.set_scalar(Scalar.dim_power(dim, -1, 2))
    return out


def _copy_spider(name: str, dim: Dim) -> Diagram:
    out = Diagram()
    spider = Z_SPIDER if name == "Z" else X_SPIDER
    s = out.add_node(spider, [dim], [dim, dim])
    g = out.add_node(DISCARD, [dim], [])
    out.add_wire(PortRef(s, Direction.OUTPUT, 1), PortRef(g, Direction.INPUT, 0))
    out.set_boundary_inputs([PortRef(s, Direction.INPUT, 0)])
    out.set_boundary_outputs([PortRef(s, Direction.OUTPUT, 0)])
    return out


def dephase(d: Dim | int | str) -> Diagram:
    """The dephasing channel: ``Z(1 -> 2)`` with its second output discarded."""
    return _copy_spider("Z", _as_dim(d))


def measure_z(d: Dim | int | str) -> Diagram:
    """The nondestructive computational-basis measurement, outcome forgotten: :func:`dephase`."""
    return dephase(d)


def measure_x(d: Dim | int | str) -> Diagram:
    """The Fourier-basis measurement ``F . dephase . F^dagger``, as ``X(1 -> 2)`` with its second
    output discarded."""
    return _copy_spider("X", _as_dim(d))


def outcome(k: int | str, d: Dim | int | str) -> Diagram:
    """The pure effect ``<k|``: ``d^(-1/2)`` times the X effect with phase ``j k / d`` at index
    ``j``. Closed against a state through :func:`double`, the Born probability of ``k``."""
    dim = _as_dim(d)
    index = sp.Integer(k) if isinstance(k, int) else sp.Symbol(k, integer=True)
    out = Diagram()
    phase = PhaseVector(dim, rule=Phase(INDEX * index / dim.to_sympy()), expand=True)
    x = out.add_node(X_SPIDER, [dim], [], phase)
    out.set_boundary_inputs([PortRef(x, Direction.INPUT, 0)])
    out.set_scalar(Scalar.dim_power(dim, -1, 2))
    return out


def compose(first: Diagram, second: Diagram) -> Diagram:
    """``second . first``: ``first``'s boundary outputs wired, in order, to ``second``'s inputs."""
    if len(first.boundary_outputs) != len(second.boundary_inputs):
        raise MixedDomainError(
            f"cannot compose: {len(first.boundary_outputs)} output(s) against "
            f"{len(second.boundary_inputs)} input(s)"
        )
    b = _Builder()
    ids1 = b.add_copy(first, Sheet.PLAIN)
    ids2 = b.add_copy(second, Sheet.PLAIN)
    out = b.diagram
    for a, c in zip(first.boundary_outputs, second.boundary_inputs, strict=True):
        out.add_wire(_moved(a, ids1), _moved(c, ids2))
    out.set_boundary_inputs([_moved(r, ids1) for r in first.boundary_inputs])
    out.set_boundary_outputs([_moved(r, ids2) for r in second.boundary_outputs])
    out.set_scalar(first.scalar * second.scalar)
    out.set_parameters({**first.parameters, **second.parameters})
    _copy_boxes(first, out, (ids1,), {})
    _copy_boxes(second, out, (ids2,), {})
    return out


def tensor(*parts: Diagram) -> Diagram:
    """The parallel composition of ``parts``, boundaries concatenated in order."""
    b = _Builder()
    inputs: list[PortRef] = []
    outputs: list[PortRef] = []
    scalar = Scalar.one()
    params: dict[str, int] = {}
    for part in parts:
        ids = b.add_copy(part, Sheet.PLAIN)
        inputs += [_moved(r, ids) for r in part.boundary_inputs]
        outputs += [_moved(r, ids) for r in part.boundary_outputs]
        scalar = scalar * part.scalar
        params.update(part.parameters)
        _copy_boxes(part, b.diagram, (ids,), {})
    b.diagram.set_boundary_inputs(inputs)
    b.diagram.set_boundary_outputs(outputs)
    b.diagram.set_scalar(scalar)
    b.diagram.set_parameters(params)
    return b.diagram


__all__ = [
    "DISCARD",
    "Doubled",
    "MixedDomainError",
    "MixedError",
    "Origin",
    "Purified",
    "Sheet",
    "channel",
    "compose",
    "conj",
    "conj_node_phase",
    "conj_phase_vector",
    "conj_with_origin",
    "dephase",
    "discard",
    "discards",
    "double",
    "double_with_origin",
    "is_pure",
    "maximally_mixed",
    "measure_x",
    "measure_z",
    "outcome",
    "partial_trace",
    "purify",
    "tensor",
]
