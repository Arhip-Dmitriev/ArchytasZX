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

"""Scalable sheet-wire notation and its lossless translation to and from bang boxes.

A :class:`ScalableDiagram` is the SZX reading (Carette & Lemonnier, arXiv:2204.11702) of a
bang-boxed diagram. Each :class:`Scale` is one scaling index:

* :attr:`ScaleKind.COPIES` -- a node-scope bang box: every node in it is a scaled generator,
  ``k`` parallel copies. :attr:`ScaledNode.scale` is the innermost COPIES scale holding a node.
* :attr:`ScaleKind.LEGS` -- a port-scope bang box: a fanned leg gathered into one sheet of width
  ``k``. :attr:`SheetPort.fan` is the innermost LEGS scale covering a port.

A :class:`Bundle` is a gathered sheet wire on the boundary. Its legs expand copy-major: copy 0
of every item in order, then copy 1, and so on, the layout
:func:`~archytaszx.diagram.bangbox.instantiate_symbol` produces. A COPIES scale has at most one
bundle per side, and the bundles enclosing a boundary ref, outer to inner, are exactly the
COPIES chain of its node. :meth:`ScalableDiagram.input_type` and
:meth:`ScalableDiagram.output_type` give the SZX wire sizes of the top-level boundary items.

:func:`to_scalable` and :func:`from_scalable` translate both ways, preserving node ids, scale
ids (bang-box ids), phases, dimensions, wires, scalar and parameters. A bang-box boundary list
regroups into bundles at the position of each scope's first ref; :func:`is_bundle_normal`
reports whether that regrouping leaves the list unchanged. :func:`strip` is the wire-stripping
functor: the plain diagram a ScalableDiagram denotes at concrete multiplicities.
:func:`validate_scalable` lists every structural problem, plus every error
:func:`~archytaszx.diagram.validate.validate` reports on the bang-box form.
"""

from __future__ import annotations

import enum
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import NewType, TypeAlias

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import BangBoxDomainError, BangBoxError, Mult
from archytaszx.diagram.generators import GeneratorType
from archytaszx.diagram.graph import (
    BangBoxId,
    Diagram,
    Direction,
    GraphError,
    NodeId,
    PortRef,
    Wire,
)
from archytaszx.diagram.validate import IssueKind, validate


class ScalableError(Exception):
    """Base class for all errors raised by this module."""


class ScalableGrammarError(ScalableError):
    """A request or structure is malformed: wrong type, unknown id, or invalid nesting."""


class ScalableDomainError(ScalableError):
    """A value is outside the accepted domain, such as an unresolved multiplicity in strip."""


ScaleId = NewType("ScaleId", int)
"""A scale identifier; :func:`to_scalable` reuses the bang-box id value."""


class ScaleKind(enum.Enum):
    """Which bang-box scope a :class:`Scale` stands for."""

    COPIES = "copies"
    LEGS = "legs"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class Scale:
    """One scaling index: its kind, multiplicity, and enclosing scale."""

    id: ScaleId
    kind: ScaleKind
    multiplicity: Mult
    parent: ScaleId | None = None

    def __post_init__(self) -> None:
        """Check field types."""
        if not _is_int(self.id) or not isinstance(self.kind, ScaleKind):
            raise ScalableGrammarError(f"Scale needs an int id and a ScaleKind, got {self!r}")
        if not isinstance(self.multiplicity, Mult):
            raise ScalableGrammarError(f"Scale multiplicity must be a Mult, got {self!r}")
        if self.parent is not None and not _is_int(self.parent):
            raise ScalableGrammarError(f"Scale parent must be an int or None, got {self!r}")


@dataclass(frozen=True, slots=True)
class SheetPort:
    """One port: its dimension and the innermost LEGS scale fanning it, if any."""

    dim: Dim
    fan: ScaleId | None = None

    def __post_init__(self) -> None:
        """Check field types."""
        if not isinstance(self.dim, Dim):
            raise ScalableGrammarError(f"SheetPort dim must be a Dim, got {self.dim!r}")
        if self.fan is not None and not _is_int(self.fan):
            raise ScalableGrammarError(f"SheetPort fan must be an int or None, got {self.fan!r}")


@dataclass(frozen=True, slots=True)
class ScaledNode:
    """One scaled generator: ports, phase, and the innermost COPIES scale holding it."""

    id: NodeId
    generator_type: GeneratorType
    inputs: tuple[SheetPort, ...]
    outputs: tuple[SheetPort, ...]
    phase: PhaseVector | None = None
    scale: ScaleId | None = None

    def __post_init__(self) -> None:
        """Check field types."""
        if not _is_int(self.id) or not isinstance(self.generator_type, GeneratorType):
            raise ScalableGrammarError(f"ScaledNode needs an int id and a GeneratorType: {self!r}")
        if not isinstance(self.inputs, tuple) or not isinstance(self.outputs, tuple):
            raise ScalableGrammarError("ScaledNode inputs and outputs must be tuples")
        if not all(isinstance(p, SheetPort) for p in (*self.inputs, *self.outputs)):
            raise ScalableGrammarError("ScaledNode ports must be SheetPort instances")
        if self.phase is not None and not isinstance(self.phase, PhaseVector):
            raise ScalableGrammarError(f"ScaledNode phase must be a PhaseVector, got {self.phase}")
        if self.scale is not None and not _is_int(self.scale):
            raise ScalableGrammarError(f"ScaledNode scale must be an int or None: {self.scale}")

    def legs(self, direction: Direction) -> tuple[SheetPort, ...]:
        """The ordered port tuple for ``direction``."""
        return self.inputs if direction is Direction.INPUT else self.outputs


@dataclass(frozen=True, slots=True)
class Bundle:
    """A gathered sheet wire: the boundary items of one COPIES scale, expanded copy-major."""

    scale: ScaleId
    items: tuple[BoundaryItem, ...]

    def __post_init__(self) -> None:
        """Check field types."""
        if not _is_int(self.scale) or not isinstance(self.items, tuple):
            raise ScalableGrammarError(f"Bundle needs an int scale and a tuple of items: {self!r}")
        for item in self.items:
            if not isinstance(item, (PortRef, Bundle)):
                raise ScalableGrammarError(f"Bundle items must be PortRef or Bundle: {item!r}")


BoundaryItem: TypeAlias = PortRef | Bundle
"""One top-level or nested boundary entry."""


def _flatten(items: Iterable[BoundaryItem]) -> list[PortRef]:
    """The refs of ``items`` in depth-first order."""
    refs: list[PortRef] = []
    for item in items:
        if isinstance(item, Bundle):
            refs.extend(_flatten(item.items))
        else:
            refs.append(item)
    return refs


def _check_items(items: object, what: str) -> tuple[BoundaryItem, ...]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        raise ScalableGrammarError(f"{what} must be a sequence of boundary items, got {items!r}")
    result = tuple(items)
    for item in result:
        if not isinstance(item, (PortRef, Bundle)):
            raise ScalableGrammarError(f"{what} entries must be PortRef or Bundle, got {item!r}")
    return result


def _mult_product(values: Iterable[Mult]) -> Mult:
    total = Mult(1)
    for value in values:
        total = total * value
    return total


@dataclass(frozen=True, slots=True, eq=False)
class ScalableDiagram:
    """An immutable diagram in scalable notation; equality and hashing are structural."""

    nodes: tuple[ScaledNode, ...]
    wires: frozenset[Wire]
    scales: tuple[Scale, ...]
    inputs: tuple[BoundaryItem, ...]
    outputs: tuple[BoundaryItem, ...]
    scalar: Scalar = field(default_factory=Scalar.one)
    parameters: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    _node_index: Mapping[NodeId, ScaledNode] = field(init=False, repr=False)
    _scale_index: Mapping[ScaleId, Scale] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Sort nodes and scales by id, freeze every container, and check element types."""
        if not all(isinstance(n, ScaledNode) for n in self.nodes):
            raise ScalableGrammarError("ScalableDiagram nodes must be ScaledNode instances")
        if not all(isinstance(s, Scale) for s in self.scales):
            raise ScalableGrammarError("ScalableDiagram scales must be Scale instances")
        nodes = tuple(sorted(self.nodes, key=lambda n: int(n.id)))
        scales = tuple(sorted(self.scales, key=lambda s: int(s.id)))
        wires = frozenset(self.wires)
        if not all(isinstance(w, Wire) for w in wires):
            raise ScalableGrammarError("ScalableDiagram wires must be Wire instances")
        if not isinstance(self.scalar, Scalar):
            raise ScalableGrammarError(f"ScalableDiagram scalar must be a Scalar: {self.scalar!r}")
        raw = dict(self.parameters)
        for name, value in raw.items():
            if not isinstance(name, str) or not name.isidentifier() or not _is_int(value):
                raise ScalableGrammarError(f"bad parameter binding {name!r} -> {value!r}")
        parameters = dict(sorted(raw.items()))
        node_index = {n.id: n for n in nodes}
        scale_index = {s.id: s for s in scales}
        if len(node_index) != len(nodes) or len(scale_index) != len(scales):
            raise ScalableGrammarError("ScalableDiagram node and scale ids must be unique")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "wires", wires)
        object.__setattr__(self, "inputs", _check_items(self.inputs, "inputs"))
        object.__setattr__(self, "outputs", _check_items(self.outputs, "outputs"))
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "_node_index", MappingProxyType(node_index))
        object.__setattr__(self, "_scale_index", MappingProxyType(scale_index))

    def _key(self) -> tuple[object, ...]:
        return (
            self.nodes,
            self.wires,
            self.scales,
            self.inputs,
            self.outputs,
            self.scalar,
            tuple(self.parameters.items()),
        )

    def __eq__(self, other: object) -> bool:
        """Equal iff every field agrees."""
        if not isinstance(other, ScalableDiagram):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hash of every field."""
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            f"ScalableDiagram(nodes={len(self.nodes)}, wires={len(self.wires)}, "
            f"scales={len(self.scales)}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"
        )

    def node(self, node_id: NodeId) -> ScaledNode:
        """The node with id ``node_id``."""
        node = self._node_index.get(node_id)
        if node is None:
            raise ScalableGrammarError(f"no such node: {node_id!r}")
        return node

    def scale(self, scale_id: ScaleId) -> Scale:
        """The scale with id ``scale_id``."""
        scale = self._scale_index.get(scale_id)
        if scale is None:
            raise ScalableGrammarError(f"no such scale: {scale_id!r}")
        return scale

    def chain(self, scale_id: ScaleId | None) -> tuple[Scale, ...]:
        """``scale_id`` and its ancestors, innermost first (empty for None)."""
        found: list[Scale] = []
        current = scale_id
        while current is not None:
            scale = self.scale(current)
            if len(found) > len(self.scales) or scale in found:
                raise ScalableGrammarError(f"scale {scale_id!r} has a cyclic parent chain")
            found.append(scale)
            current = scale.parent
        return tuple(found)

    def copies_chain(self, scale_id: ScaleId | None) -> tuple[ScaleId, ...]:
        """The COPIES scales on the chain of ``scale_id``, outermost first."""
        return tuple(s.id for s in reversed(self.chain(scale_id)) if s.kind is ScaleKind.COPIES)

    def fan_chain(self, fan: ScaleId | None) -> tuple[ScaleId, ...]:
        """The LEGS scales from ``fan`` up to the first non-LEGS ancestor, outermost first."""
        found: list[ScaleId] = []
        for scale in self.chain(fan):
            if scale.kind is not ScaleKind.LEGS:
                break
            found.append(scale.id)
        return tuple(reversed(found))

    def size(self, scale_id: ScaleId | None) -> Mult:
        """The product of the COPIES multiplicities on the chain of ``scale_id``."""
        return _mult_product(self.scale(s).multiplicity for s in self.copies_chain(scale_id))

    def fan_width(self, port_fan: ScaleId | None) -> Mult:
        """The product of the LEGS multiplicities on :meth:`fan_chain`."""
        return _mult_product(self.scale(s).multiplicity for s in self.fan_chain(port_fan))

    def port(self, ref: PortRef) -> SheetPort:
        """The port ``ref`` addresses."""
        legs = self.node(ref.node_id).legs(ref.direction)
        if ref.index >= len(legs):
            raise ScalableGrammarError(f"port {ref!r} is out of range")
        return legs[ref.index]

    def width(self, item: BoundaryItem) -> Mult:
        """The number of plain wires ``item`` stands for."""
        if isinstance(item, PortRef):
            return self.fan_width(self.port(item).fan)
        total = Mult(0)
        for inner in item.items:
            total = total + self.width(inner)
        return self.scale(item.scale).multiplicity * total

    def wire_width(self, wire: Wire) -> Mult:
        """The number of plain wires ``wire`` stands for: the size of its endpoints' scale."""
        return self.size(self.node(wire.a.node_id).scale)

    def input_type(self) -> tuple[Mult, ...]:
        """The SZX type of the input boundary: one width per top-level item."""
        return tuple(self.width(item) for item in self.inputs)

    def output_type(self) -> tuple[Mult, ...]:
        """The SZX type of the output boundary: one width per top-level item."""
        return tuple(self.width(item) for item in self.outputs)

    def free_mult_symbols(self) -> frozenset[str]:
        """Every symbol in any scale multiplicity."""
        return frozenset(name for s in self.scales for name in s.multiplicity.free_symbols)

    def renumbered(self) -> ScalableDiagram:
        """A copy with node ids and scale ids replaced by their ranks in ascending order."""
        node_map = {n.id: NodeId(rank) for rank, n in enumerate(self.nodes)}
        scale_map = {s.id: ScaleId(rank) for rank, s in enumerate(self.scales)}

        def known(value: ScaleId) -> ScaleId:
            if value not in scale_map:
                raise ScalableGrammarError(f"no such scale: {value!r}")
            return scale_map[value]

        def sid(value: ScaleId | None) -> ScaleId | None:
            return None if value is None else known(value)

        def ref(old: PortRef) -> PortRef:
            if old.node_id not in node_map:
                raise ScalableGrammarError(f"no such node: {old.node_id!r}")
            return PortRef(node_map[old.node_id], old.direction, old.index)

        def item(old: BoundaryItem) -> BoundaryItem:
            if isinstance(old, PortRef):
                return ref(old)
            return Bundle(known(old.scale), tuple(item(i) for i in old.items))

        def ports(old: tuple[SheetPort, ...]) -> tuple[SheetPort, ...]:
            return tuple(SheetPort(p.dim, sid(p.fan)) for p in old)

        return ScalableDiagram(
            nodes=tuple(
                ScaledNode(
                    node_map[n.id],
                    n.generator_type,
                    ports(n.inputs),
                    ports(n.outputs),
                    n.phase,
                    sid(n.scale),
                )
                for n in self.nodes
            ),
            wires=frozenset(Wire(ref(w.a), ref(w.b)) for w in self.wires),
            scales=tuple(
                Scale(scale_map[s.id], s.kind, s.multiplicity, sid(s.parent)) for s in self.scales
            ),
            inputs=tuple(item(i) for i in self.inputs),
            outputs=tuple(item(i) for i in self.outputs),
            scalar=self.scalar,
            parameters=self.parameters,
        )


def _to_mult(multiplicity: Mult | int | str) -> Mult:
    try:
        return Mult(multiplicity)
    except BangBoxDomainError as exc:
        raise ScalableDomainError(str(exc)) from exc
    except (BangBoxError, TypeError) as exc:
        raise ScalableGrammarError(str(exc)) from exc


class ScalableBuilder:
    """A mutable assembler for a :class:`ScalableDiagram`; ids are allocated 0, 1, 2, ..."""

    __slots__ = ("_inputs", "_nodes", "_outputs", "_parameters", "_scalar", "_scales", "_wires")

    def __init__(self) -> None:
        """Start from an empty diagram."""
        self._nodes: dict[NodeId, ScaledNode] = {}
        self._scales: dict[ScaleId, Scale] = {}
        self._wires: set[Wire] = set()
        self._inputs: tuple[BoundaryItem, ...] = ()
        self._outputs: tuple[BoundaryItem, ...] = ()
        self._scalar = Scalar.one()
        self._parameters: dict[str, int] = {}

    def add_scale(
        self, kind: ScaleKind, multiplicity: Mult | int | str, parent: ScaleId | None = None
    ) -> ScaleId:
        """Add a scale and return its id."""
        scale_id = ScaleId(len(self._scales))
        self._scales[scale_id] = Scale(scale_id, kind, _to_mult(multiplicity), parent)
        return scale_id

    def add_node(
        self,
        generator_type: GeneratorType,
        input_dims: Sequence[Dim],
        output_dims: Sequence[Dim],
        phase: PhaseVector | None = None,
        scale: ScaleId | None = None,
    ) -> NodeId:
        """Add an unfanned node and return its id."""
        node_id = NodeId(len(self._nodes))
        self._nodes[node_id] = ScaledNode(
            node_id,
            generator_type,
            tuple(SheetPort(d) for d in input_dims),
            tuple(SheetPort(d) for d in output_dims),
            phase,
            scale,
        )
        return node_id

    def set_fan(self, ref: PortRef, fan: ScaleId) -> None:
        """Mark the port ``ref`` as fanned by the LEGS scale ``fan``."""
        node = self._nodes.get(ref.node_id)
        if node is None or ref.index >= len(node.legs(ref.direction)):
            raise ScalableGrammarError(f"set_fan: no such port {ref!r}")
        legs = list(node.legs(ref.direction))
        legs[ref.index] = SheetPort(legs[ref.index].dim, fan)
        if ref.direction is Direction.INPUT:
            self._nodes[ref.node_id] = replace(node, inputs=tuple(legs))
        else:
            self._nodes[ref.node_id] = replace(node, outputs=tuple(legs))

    def add_wire(self, a: PortRef, b: PortRef) -> None:
        """Join two ports."""
        try:
            self._wires.add(Wire(a, b))
        except GraphError as exc:
            raise ScalableGrammarError(str(exc)) from exc

    def set_inputs(self, items: Sequence[BoundaryItem]) -> None:
        """Replace the input boundary."""
        self._inputs = _check_items(items, "inputs")

    def set_outputs(self, items: Sequence[BoundaryItem]) -> None:
        """Replace the output boundary."""
        self._outputs = _check_items(items, "outputs")

    def multiply_scalar(self, s: Scalar) -> None:
        """Multiply the scalar accumulator by ``s``."""
        if not isinstance(s, Scalar):
            raise ScalableGrammarError(f"multiply_scalar requires a Scalar, got {s!r}")
        self._scalar = self._scalar * s

    def bind_parameter(self, name: str, value: int) -> None:
        """Record ``value`` for the symbol ``name``."""
        if not isinstance(name, str) or not name.isidentifier() or not _is_int(value):
            raise ScalableGrammarError(f"bad parameter binding {name!r} -> {value!r}")
        self._parameters[name] = value

    def build(self, *, check: bool = True) -> ScalableDiagram:
        """The assembled diagram; with ``check``, refused unless :func:`validate_scalable`
        passes."""
        result = ScalableDiagram(
            nodes=tuple(self._nodes.values()),
            wires=frozenset(self._wires),
            scales=tuple(self._scales.values()),
            inputs=self._inputs,
            outputs=self._outputs,
            scalar=self._scalar,
            parameters=dict(self._parameters),
        )
        if check:
            issues = validate_scalable(result)
            if issues:
                raise ScalableGrammarError("; ".join(issues))
        return result


# -- validation --------------------------------------------------------------------------


def _forest_issues(s: ScalableDiagram) -> list[str]:
    """Unknown parents, cycles, and kind/parent mismatches."""
    issues: list[str] = []
    for scale in s.scales:
        if scale.parent is not None and scale.parent not in s._scale_index:
            issues.append(f"scale {scale.id} declares unknown parent {scale.parent}")
    if issues:
        return issues
    for scale in s.scales:
        try:
            s.chain(scale.id)
        except ScalableGrammarError as exc:
            issues.append(str(exc))
    if issues:
        return issues
    for scale in s.scales:
        parent = None if scale.parent is None else s.scale(scale.parent)
        if scale.kind is ScaleKind.COPIES and parent is not None and parent.kind is ScaleKind.LEGS:
            issues.append(f"COPIES scale {scale.id} has LEGS parent {scale.parent}")
    return issues


def _anchor(s: ScalableDiagram, fan: ScaleId) -> ScaleId | None:
    """The nearest non-LEGS ancestor of the LEGS scale ``fan``."""
    for scale in s.chain(fan):
        if scale.kind is not ScaleKind.LEGS:
            return scale.id
    return None


def _all_refs(node: ScaledNode) -> Iterable[tuple[PortRef, SheetPort]]:
    for direction in (Direction.INPUT, Direction.OUTPUT):
        for index, port in enumerate(node.legs(direction)):
            yield PortRef(node.id, direction, index), port


def _membership_issues(s: ScalableDiagram) -> list[str]:
    """Node scales and port fans must name scales of the right kind and region."""
    issues: list[str] = []
    for node in s.nodes:
        if node.scale is not None:
            scale = s._scale_index.get(node.scale)
            if scale is None or scale.kind is not ScaleKind.COPIES:
                issues.append(f"node {node.id} scale {node.scale} is not a COPIES scale")
                continue
        for ref, port in _all_refs(node):
            if port.fan is None:
                continue
            fan = s._scale_index.get(port.fan)
            if fan is None or fan.kind is not ScaleKind.LEGS:
                issues.append(f"port {ref} fan {port.fan} is not a LEGS scale")
            elif _anchor(s, fan.id) != node.scale:
                issues.append(
                    f"port {ref} fan {fan.id} lies outside the scale {node.scale} of its node"
                )
    return issues


def _usage_issues(s: ScalableDiagram) -> list[str]:
    """Each port used exactly once; fanned ports only on the boundary; flat wires."""
    issues: list[str] = []
    wires = sorted(s.wires, key=lambda w: w.sort_key())
    used = Counter([end for w in wires for end in (w.a, w.b)])
    used.update(_flatten(s.inputs))
    used.update(_flatten(s.outputs))
    resolvable: set[PortRef] = set()
    for ref in sorted(used, key=lambda r: r.sort_key()):
        node = s._node_index.get(ref.node_id)
        if node is None or ref.index >= len(node.legs(ref.direction)):
            issues.append(f"{ref} does not resolve to a port")
        else:
            resolvable.add(ref)
    for node in s.nodes:
        for ref, _port in _all_refs(node):
            if used[ref] != 1:
                issues.append(f"{ref} is used {used[ref]} times, not once")
    for wire in wires:
        ends = [end for end in (wire.a, wire.b) if end in resolvable]
        for end in ends:
            if s.port(end).fan is not None:
                issues.append(f"fanned port {end} is wired; fanned ports lie on the boundary")
        if len(ends) == 2:
            scales = (s.node(wire.a.node_id).scale, s.node(wire.b.node_id).scale)
            if scales[0] != scales[1]:
                issues.append(f"{wire} joins nodes of different scales {scales[0]}, {scales[1]}")
    return issues


def _coverage_issues(s: ScalableDiagram) -> list[str]:
    """Every scale is inhabited, and a LEGS child covers strictly fewer ports than its parent."""
    issues: list[str] = []
    covered: dict[ScaleId, set[PortRef]] = {scale.id: set() for scale in s.scales}
    inhabited: set[ScaleId] = set()
    for node in s.nodes:
        inhabited.update(sc.id for sc in s.chain(node.scale))
        for ref, port in _all_refs(node):
            for scale in s.chain(port.fan):
                if scale.kind is ScaleKind.LEGS:
                    covered[scale.id].add(ref)
    for scale in s.scales:
        if scale.kind is ScaleKind.COPIES and scale.id not in inhabited:
            issues.append(f"COPIES scale {scale.id} contains no node")
        if scale.kind is ScaleKind.LEGS and not covered[scale.id]:
            issues.append(f"LEGS scale {scale.id} covers no port")
        parent = s.scale(scale.parent) if scale.parent is not None else None
        if (
            scale.kind is ScaleKind.LEGS
            and parent is not None
            and parent.kind is ScaleKind.LEGS
            and not covered[scale.id] < covered[parent.id]
        ):
            issues.append(
                f"LEGS scale {scale.id} does not cover strictly fewer ports than "
                f"its parent {parent.id}"
            )
    return issues


def _walk_bundles(
    s: ScalableDiagram,
    side: str,
    items: tuple[BoundaryItem, ...],
    path: tuple[ScaleId, ...],
    seen: Counter[ScaleId],
    issues: list[str],
) -> None:
    for item in items:
        if isinstance(item, Bundle):
            scale = s._scale_index.get(item.scale)
            if scale is None or scale.kind is not ScaleKind.COPIES:
                issues.append(f"{side}: bundle scale {item.scale} is not a COPIES scale")
                continue
            seen[item.scale] += 1
            if not item.items:
                issues.append(f"{side}: bundle of scale {item.scale} is empty")
            _walk_bundles(s, side, item.items, (*path, item.scale), seen, issues)
            continue
        node = s._node_index.get(item.node_id)
        if node is None:
            continue
        expected = s.copies_chain(node.scale)
        if path != expected:
            issues.append(
                f"{side}: {item} sits in bundles {list(path)}, not its node's "
                f"scale chain {list(expected)}"
            )


def _placement_issues(s: ScalableDiagram) -> list[str]:
    """Bundles nest along COPIES chains, once per scale per side, never empty."""
    issues: list[str] = []
    for side, items in (("inputs", s.inputs), ("outputs", s.outputs)):
        seen: Counter[ScaleId] = Counter()
        _walk_bundles(s, side, items, (), seen, issues)
        for scale_id, count in sorted(seen.items()):
            if count > 1:
                issues.append(f"{side}: scale {scale_id} has {count} bundles, not at most one")
    return issues


def _structural_issues(s: ScalableDiagram) -> list[str]:
    for check in (_forest_issues, _membership_issues):
        issues = check(s)
        if issues:
            return issues
    return [*_usage_issues(s), *_coverage_issues(s), *_placement_issues(s)]


def validate_scalable(s: ScalableDiagram) -> tuple[str, ...]:
    """Every structural problem of ``s``, then every validate error of its bang-box form.

    Empty when ``s`` is valid; deferred validate issues are not reported.
    """
    issues = _structural_issues(s)
    if issues:
        return tuple(issues)
    report = validate(from_scalable(s))
    return tuple(f"bang-box form: {issue.message}" for issue in report.errors)


# -- translation -------------------------------------------------------------------------

_UNRESOLVABLE = frozenset({IssueKind.UNKNOWN_NODE, IssueKind.PORT_INDEX_OUT_OF_RANGE})


def _depth(diagram: Diagram, box_id: BangBoxId) -> int:
    depth = 0
    current = diagram.bang_boxes[box_id].parent
    while current is not None:
        depth += 1
        current = diagram.bang_boxes[current].parent
    return depth


@dataclass(frozen=True, slots=True)
class _Open:
    """A bundle under construction."""

    scale: ScaleId
    items: list[PortRef | _Open]


def group_boundary(
    refs: Sequence[PortRef], chains: Mapping[NodeId, tuple[ScaleId, ...]]
) -> tuple[BoundaryItem, ...]:
    """``refs`` regrouped into nested bundles, each placed where its first ref is met."""
    root: list[PortRef | _Open] = []
    opened: dict[ScaleId, _Open] = {}
    for ref in refs:
        target = root
        for scale_id in chains[ref.node_id]:
            if scale_id not in opened:
                opened[scale_id] = _Open(scale_id, [])
                target.append(opened[scale_id])
            target = opened[scale_id].items
        target.append(ref)
    return _freeze(root)


def _freeze(items: list[PortRef | _Open]) -> tuple[BoundaryItem, ...]:
    return tuple(
        item if isinstance(item, PortRef) else Bundle(item.scale, _freeze(item.items))
        for item in items
    )


def to_scalable(diagram: Diagram) -> ScalableDiagram:
    """The scalable form of ``diagram``: node-scope boxes become COPIES scales, port-scope
    boxes LEGS scales, with ids, phases, dims, wires, scalar and parameters kept.

    Refuses a diagram with a bang-box validate error, an unresolvable ref, a node-scope box
    under a port-scope box, or a wire between nodes of different innermost node-scope boxes.
    """
    if not isinstance(diagram, Diagram):
        raise ScalableGrammarError(f"to_scalable requires a Diagram, got {diagram!r}")
    blocking = [
        issue.message
        for issue in validate(diagram).errors
        if issue.kind.name.startswith("BANGBOX_") or issue.kind in _UNRESOLVABLE
    ]
    if blocking:
        raise ScalableGrammarError("cannot translate: " + "; ".join(blocking))
    boxes = diagram.bang_boxes
    scales: list[Scale] = []
    for box_id in sorted(boxes):
        box = boxes[box_id]
        kind = ScaleKind.COPIES if box.is_node_scope else ScaleKind.LEGS
        parent = None if box.parent is None else ScaleId(int(box.parent))
        scales.append(Scale(ScaleId(int(box_id)), kind, box.multiplicity, parent))
    by_depth = sorted(boxes, key=lambda b: (-_depth(diagram, b), int(b)))
    node_scale: dict[NodeId, ScaleId] = {}
    port_fan: dict[PortRef, ScaleId] = {}
    for box_id in by_depth:
        for node_id in boxes[box_id].node_scope:
            node_scale.setdefault(node_id, ScaleId(int(box_id)))
        for ref in boxes[box_id].port_scope:
            port_fan.setdefault(ref, ScaleId(int(box_id)))
    nodes = tuple(
        ScaledNode(
            node.id,
            node.generator_type,
            tuple(
                SheetPort(p.dim, port_fan.get(PortRef(nid, Direction.INPUT, i)))
                for i, p in enumerate(node.inputs)
            ),
            tuple(
                SheetPort(p.dim, port_fan.get(PortRef(nid, Direction.OUTPUT, i)))
                for i, p in enumerate(node.outputs)
            ),
            node.phase,
            node_scale.get(nid),
        )
        for nid, node in sorted(diagram.nodes.items())
    )
    draft = ScalableDiagram(nodes, diagram.wires, tuple(scales), (), (), diagram.scalar, {})
    issues = _forest_issues(draft) or _membership_issues(draft)
    if issues:
        raise ScalableGrammarError("cannot translate: " + "; ".join(issues))
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if draft.node(wire.a.node_id).scale != draft.node(wire.b.node_id).scale:
            raise ScalableGrammarError(f"cannot translate: {wire} crosses a node-scope box")
    chains = {n.id: draft.copies_chain(n.scale) for n in nodes}
    return ScalableDiagram(
        nodes=nodes,
        wires=diagram.wires,
        scales=tuple(scales),
        inputs=group_boundary(diagram.boundary_inputs, chains),
        outputs=group_boundary(diagram.boundary_outputs, chains),
        scalar=diagram.scalar,
        parameters=diagram.parameters,
    )


def from_scalable(s: ScalableDiagram) -> Diagram:
    """The bang-box form of ``s``: nodes added in ascending id order, one box per scale in
    ascending id order, boundary lists the depth-first flattening of the bundles."""
    if not isinstance(s, ScalableDiagram):
        raise ScalableGrammarError(f"from_scalable requires a ScalableDiagram, got {s!r}")
    diagram = Diagram()
    node_map: dict[NodeId, NodeId] = {}
    for node in s.nodes:
        node_map[node.id] = diagram.add_node(
            node.generator_type,
            [p.dim for p in node.inputs],
            [p.dim for p in node.outputs],
            phase=node.phase,
        )
    box_map = {scale.id: BangBoxId(rank) for rank, scale in enumerate(s.scales)}

    def ref(old: PortRef) -> PortRef:
        if old.node_id not in node_map:
            raise ScalableGrammarError(f"from_scalable: no such node {old.node_id!r}")
        return PortRef(node_map[old.node_id], old.direction, old.index)

    members: dict[ScaleId, set[NodeId]] = {scale.id: set() for scale in s.scales}
    ports: dict[ScaleId, set[PortRef]] = {scale.id: set() for scale in s.scales}
    for node in s.nodes:
        for scale in s.chain(node.scale):
            members[scale.id].add(node_map[node.id])
        for old, port in _all_refs(node):
            for scale in s.chain(port.fan):
                if scale.kind is ScaleKind.LEGS:
                    ports[scale.id].add(ref(old))
    try:
        for scale in s.scales:
            parent = None if scale.parent is None else box_map[scale.parent]
            if scale.kind is ScaleKind.COPIES:
                diagram.add_bang_box(
                    scale.multiplicity, node_scope=frozenset(members[scale.id]), parent=parent
                )
            else:
                diagram.add_bang_box(
                    scale.multiplicity, port_scope=frozenset(ports[scale.id]), parent=parent
                )
    except (BangBoxError, KeyError) as exc:
        raise ScalableGrammarError(f"from_scalable: malformed scale structure: {exc}") from exc
    for wire in sorted(s.wires, key=lambda w: w.sort_key()):
        diagram.add_wire(ref(wire.a), ref(wire.b))
    diagram.set_boundary_inputs([ref(r) for r in _flatten(s.inputs)])
    diagram.set_boundary_outputs([ref(r) for r in _flatten(s.outputs)])
    diagram.set_scalar(s.scalar)
    diagram.set_parameters(s.parameters)
    return diagram


def is_bundle_normal(diagram: Diagram) -> bool:
    """Whether bundle regrouping leaves both boundary lists of ``diagram`` unchanged."""
    s = to_scalable(diagram)
    return (
        tuple(_flatten(s.inputs)) == diagram.boundary_inputs
        and tuple(_flatten(s.outputs)) == diagram.boundary_outputs
    )


# -- wire stripping ----------------------------------------------------------------------


class _Evaluator:
    """Concrete scale values under an environment, resolved on first use."""

    __slots__ = ("_cache", "_env", "_s")

    def __init__(self, s: ScalableDiagram, env: Mapping[str, int]) -> None:
        self._s = s
        self._env = env
        self._cache: dict[ScaleId, int] = {}

    def value(self, scale_id: ScaleId) -> int:
        if scale_id not in self._cache:
            mult = self._s.scale(scale_id).multiplicity
            names = sorted(mult.free_symbols)
            try:
                resolved = mult.substitute({n: self._env[n] for n in names if n in self._env})
            except (BangBoxError, TypeError) as exc:
                raise ScalableDomainError(f"scale {scale_id}: {exc}") from exc
            if not resolved.is_concrete:
                missing = sorted(resolved.free_symbols)
                raise ScalableDomainError(
                    f"scale {scale_id} multiplicity {mult} has unresolved symbol(s) {missing}"
                )
            self._cache[scale_id] = resolved.to_int()
        return self._cache[scale_id]

    def vectors(self, chain: tuple[ScaleId, ...]) -> list[tuple[int, ...]]:
        """Every copy-index vector along ``chain``, lexicographically."""
        if not chain:
            return [()]
        count = self.value(chain[0])
        if count == 0:
            return []
        rest = self.vectors(chain[1:])
        return [(i, *tail) for i in range(count) for tail in rest]

    def width(self, fan_chain: tuple[ScaleId, ...]) -> int:
        total = 1
        for scale_id in fan_chain:
            total *= self.value(scale_id)
            if total == 0:
                return 0
        return total


def strip(s: ScalableDiagram, assignment: Mapping[str, int] | None = None) -> Diagram:
    """The plain diagram ``s`` denotes once every multiplicity is concrete.

    Multiplicities are evaluated under ``assignment`` over ``s.parameters``. A COPIES scale of
    value k yields k copies of its contents, a LEGS fan of width k yields k consecutive legs
    in place of its port, and each bundle expands copy-major. The result has no bang boxes,
    the scalar of ``s``, and its parameters minus every multiplicity symbol.
    """
    if not isinstance(s, ScalableDiagram):
        raise ScalableGrammarError(f"strip requires a ScalableDiagram, got {s!r}")
    issues = _structural_issues(s)
    if issues:
        raise ScalableGrammarError("strip: " + "; ".join(issues))
    env = dict(s.parameters)
    for name, value in (assignment or {}).items():
        if not isinstance(name, str) or not _is_int(value) or value < 0:
            raise ScalableDomainError(f"strip: bad assignment {name!r} -> {value!r}")
        env[name] = value
    ev = _Evaluator(s, env)
    out = Diagram()
    instances: dict[tuple[NodeId, tuple[int, ...]], NodeId] = {}
    offsets: dict[PortRef, tuple[int, int]] = {}
    node_vectors: dict[NodeId, list[tuple[int, ...]]] = {}
    for node in s.nodes:
        vectors = ev.vectors(s.copies_chain(node.scale))
        node_vectors[node.id] = vectors
        if not vectors:
            continue
        dims: dict[Direction, list[Dim]] = {Direction.INPUT: [], Direction.OUTPUT: []}
        for ref, port in _all_refs(node):
            width = ev.width(s.fan_chain(port.fan))
            offsets[ref] = (len(dims[ref.direction]), width)
            dims[ref.direction].extend([port.dim] * width)
        for vector in vectors:
            instances[(node.id, vector)] = out.add_node(
                node.generator_type,
                dims[Direction.INPUT],
                dims[Direction.OUTPUT],
                phase=node.phase,
            )

    def legs(ref: PortRef, vector: tuple[int, ...]) -> list[PortRef]:
        start, width = offsets[ref]
        new_id = instances[(ref.node_id, vector)]
        return [PortRef(new_id, ref.direction, start + j) for j in range(width)]

    for wire in sorted(s.wires, key=lambda w: w.sort_key()):
        for vector in node_vectors[wire.a.node_id]:
            (a,) = legs(wire.a, vector)
            (b,) = legs(wire.b, vector)
            out.add_wire(a, b)

    def expand(items: tuple[BoundaryItem, ...], vector: tuple[int, ...]) -> list[PortRef]:
        refs: list[PortRef] = []
        for item in items:
            if isinstance(item, PortRef):
                refs.extend(legs(item, vector))
            else:
                for copy in range(ev.value(item.scale)):
                    refs.extend(expand(item.items, (*vector, copy)))
        return refs

    out.set_boundary_inputs(expand(s.inputs, ()))
    out.set_boundary_outputs(expand(s.outputs, ()))
    out.set_scalar(s.scalar)
    dead = s.free_mult_symbols()
    out.set_parameters({k: v for k, v in s.parameters.items() if k not in dead})
    return out
