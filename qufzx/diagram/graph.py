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

"""Core graph data model: Port, Node, and Diagram, with deep copy and controlled mutation.

A :class:`Diagram` is an open graph: :class:`Node` objects joined by :class:`Wire`\\ s, two
ordered boundary lists of unwired ports, and an exact
:class:`~qufzx.algebra.scalar.Scalar` accumulator. Dimension lives on each :class:`Port`
individually -- there is no global or per-diagram dimension field here, not even as a
default, per the spec invariant that dimension is stored per port.

What this representation cannot express. A boundary entry is a :class:`PortRef` into some
node, so there is no way to express a bare identity wire running from a boundary input
straight to a boundary output. Representing it would need either a synthetic boundary-node
type (not a real generator; it would need its own policy, denotation, and validation
carve-outs) or a boundary-to-boundary edge alongside node-to-node wires (breaking the
invariant that every wire has two ports drawn from actual nodes). Out of scope until an
explicit identity generator is registered or the Phase 18 parser must round-trip a bare
wire.

Validation ownership. Construction and mutation here are deliberately permissive: methods
check only properties of the data structure itself (an index is a non-negative int, a wire
cannot join a port to itself, you cannot mutate a node that does not exist). Cross-cutting
well-formedness -- dimension agreement, double-wired ports, boundary/wire conflicts,
out-of-range indices, generator policy -- is entirely :mod:`qufzx.diagram.validate`'s
responsibility, so its report can carry every problem in one pass, including problems
spanning several mutation calls, rather than surfacing as scattered exceptions from
whichever mutator introduced them.

Node removal. :meth:`Diagram.remove_node` cascades to every incident wire and boundary
entry. This is the one invariant enforced eagerly: leaving a :class:`PortRef` that resolves
to nothing would corrupt every other reference to the removed node, with no later pass able
to tell "dangling because stale" from "never wired". Rejecting the removal instead was the
alternative; cascading was chosen so deleting a node always succeeds and always leaves the
diagram referentially consistent (though not necessarily *valid* -- the boundary may now
have a different arity than a caller expected).

Symbol substitution. :meth:`Diagram.substitute` is node-id-preserving: identical
:class:`NodeId`\\ s, wire set, boundary lists, and ``_next_id``, with only each port's
``Dim``, each node's ``PhaseVector``, and the diagram's ``Scalar`` substituted. Rebuilding
through :meth:`add_node` could not express this -- that allocates fresh ids, and every wire
and boundary ref addresses a node by id, so a rebuild would detach them all. Like every
method here it validates nothing and never mutates the receiver.

Diagram equality. :class:`Diagram` defines no ``__eq__``. Diagram equality -- "do these
denote the same map, possibly after rewriting" -- is Phase 13's normal form, checked by
Phase 4's oracle; a naive structural ``__eq__`` would be a weaker notion later code could
come to rely on by accident. :class:`PortRef`, :class:`Port`, :class:`Wire`, and
:class:`~qufzx.diagram.generators.GeneratorType` are value objects and stay value-equal and
hashable, since they are looked up and de-duplicated by value throughout.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import NewType, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim, DimSubstituteValue, DimSymbolKey
from qufzx.algebra.phase import Phase, PhaseSubstituteValue, PhaseSymbolKey, PhaseVector
from qufzx.algebra.scalar import Scalar, ScalarSubstituteValue, ScalarSymbolKey
from qufzx.diagram.generators import GeneratorType


class GraphError(Exception):
    """Base class for all errors raised by this module."""


class GraphDomainError(GraphError):
    """A value or operation is outside the mathematical domain this module accepts.

    Raised for a negative port index and for a wire that would connect a port to
    itself.
    """


class GraphGrammarError(GraphError):
    """A request is malformed: wrong type, or a reference to something absent.

    Raised for non-Port/non-PhaseVector arguments, and for operating on a node id or
    wire that the diagram does not currently hold.
    """


class Direction(enum.Enum):
    """Which ordered leg list of a node a :class:`PortRef` or :class:`Port` belongs to."""

    INPUT = "input"
    OUTPUT = "output"


NodeId = NewType("NodeId", int)
"""An opaque node identifier, allocated by :meth:`Diagram.add_node`.

A distinct type over bare ``int`` so that a node id is never accidentally used where a
port index or a leg count is expected, and vice versa.
"""


@dataclass(frozen=True, slots=True)
class PortRef:
    """An address for one port: which node, which side, and which position in that side's list."""

    node_id: NodeId
    direction: Direction
    index: int

    def __post_init__(self) -> None:
        """Validate that ``direction`` is a Direction and ``index`` a non-negative int."""
        if not isinstance(self.direction, Direction):
            raise GraphGrammarError(
                f"PortRef direction must be a Direction, got {self.direction!r}"
            )
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise GraphGrammarError(f"PortRef index must be an int, got {self.index!r}")
        if self.index < 0:
            raise GraphDomainError(f"PortRef index must be >= 0, got {self.index}")

    def __repr__(self) -> str:
        return (
            f"PortRef(node_id={self.node_id!r}, direction={self.direction.value}, "
            f"index={self.index})"
        )

    def sort_key(self) -> tuple[int, str, int]:
        """A canonical, hash-independent ordering key: ``(node_id, direction, index)``.

        :class:`PortRef`'s own ``__hash__`` folds in :class:`Direction`, an ``enum.Enum``
        hashed by member name and so ``PYTHONHASHSEED``-dependent. Any caller that iterates
        a set of ``PortRef`` (or of :class:`Wire`) and lets that order reach a returned
        value, a certificate field, or an exception message would produce a result varying
        by process. This key uses ``direction.value``, not the enum member, so it contains
        no seed-dependent hash anywhere in its comparison path. Pinned end to end by
        ``tests/test_engine.py::TestCrossProcessDeterminism``.
        """
        return (int(self.node_id), self.direction.value, self.index)


@dataclass(frozen=True, slots=True)
class Port:
    """A single port, carrying its own dimension. See the module docstring for why."""

    dim: Dim

    def __post_init__(self) -> None:
        """Validate that ``dim`` is a Dim."""
        if not isinstance(self.dim, Dim):
            raise GraphGrammarError(f"Port dim must be a Dim, got {self.dim!r}")


@dataclass(frozen=True, slots=True, eq=False)
class Wire:
    """An unordered pair of ports: ``Wire(a, b) == Wire(b, a)``.

    A self-loop through two *distinct* ports of the same node is legal ZX and is not
    rejected here; only a wire naming the exact same port twice is malformed, since it
    cannot be given any meaning as a connection between two things.
    """

    a: PortRef
    b: PortRef

    def __post_init__(self) -> None:
        """Reject a wire that names the same port at both ends."""
        if self.a == self.b:
            raise GraphDomainError(f"a wire cannot connect port {self.a} to itself")

    def endpoints(self) -> frozenset[PortRef]:
        """The two ports this wire joins, as an unordered pair."""
        return frozenset((self.a, self.b))

    def __eq__(self, other: object) -> bool:
        """Equal iff the same two ports, regardless of stored order."""
        if not isinstance(other, Wire):
            return NotImplemented
        return self.endpoints() == other.endpoints()

    def __hash__(self) -> int:
        """Hash agrees with __eq__: order-independent."""
        return hash(self.endpoints())

    def __repr__(self) -> str:
        return f"Wire({self.a!r}, {self.b!r})"

    def sort_key(self) -> tuple[tuple[int, str, int], tuple[int, str, int]]:
        """A canonical, hash-independent ordering key: the sorted pair of endpoint keys.

        See :meth:`PortRef.sort_key` for why this is needed. ``Wire`` is unordered, so
        this key must not depend on which endpoint is stored as ``a`` versus ``b``: it is
        the smaller of ``(a, b)`` and ``(b, a)``, so two equal wires always produce the
        identical key.
        """
        a_key = self.a.sort_key()
        b_key = self.b.sort_key()
        return (a_key, b_key) if a_key <= b_key else (b_key, a_key)


@dataclass(frozen=True, slots=True)
class Node:
    """One diagram node: a generator type, ordered input/output ports, and a phase slot.

    Immutable: :meth:`Diagram.set_phase` builds a replacement ``Node`` via
    :meth:`with_phase` rather than mutating one in place, so a ``Node`` handed out by a
    read-only view can never be corrupted by a later mutation elsewhere.
    """

    id: NodeId
    generator_type: GeneratorType
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    phase: PhaseVector | None = None

    def __post_init__(self) -> None:
        """Validate field types: GeneratorType, tuples of Port, and an optional PhaseVector."""
        if not isinstance(self.generator_type, GeneratorType):
            raise GraphGrammarError(
                f"Node generator_type must be a GeneratorType, got {self.generator_type!r}"
            )
        if not isinstance(self.inputs, tuple) or not isinstance(self.outputs, tuple):
            raise GraphGrammarError("Node inputs and outputs must be tuples of Port")
        for port in (*self.inputs, *self.outputs):
            if not isinstance(port, Port):
                raise GraphGrammarError(f"Node ports must be Port instances, got {port!r}")
        if self.phase is not None and not isinstance(self.phase, PhaseVector):
            raise GraphGrammarError(f"Node phase must be a PhaseVector or None, got {self.phase!r}")

    @property
    def num_inputs(self) -> int:
        """The number of input ports."""
        return len(self.inputs)

    @property
    def num_outputs(self) -> int:
        """The number of output ports."""
        return len(self.outputs)

    def legs(self, direction: Direction) -> tuple[Port, ...]:
        """The ordered port tuple for the given direction."""
        return self.inputs if direction is Direction.INPUT else self.outputs

    def with_phase(self, phase: PhaseVector | None) -> Node:
        """Return a new Node identical to this one but with its phase slot replaced."""
        return Node(
            id=self.id,
            generator_type=self.generator_type,
            inputs=self.inputs,
            outputs=self.outputs,
            phase=phase,
        )


class Diagram:
    """An open ZX diagram: nodes, wires, ordered boundaries, and an exact scalar factor.

    All state is private. Callers observe it only through read-only views (tuples,
    ``MappingProxyType``) exposed as properties, and mutate it only through the
    explicit methods below -- never by reaching into a container a getter returned.
    See the module docstring for what construction does and does not validate, for the
    node-removal cascade, and for why this class has no ``__eq__``.
    """

    __slots__ = ("_boundary_inputs", "_boundary_outputs", "_next_id", "_nodes", "_scalar", "_wires")

    def __init__(self) -> None:
        """Build an empty diagram: no nodes, no wires, empty boundaries, scalar 1."""
        self._nodes: dict[NodeId, Node] = {}
        self._wires: set[Wire] = set()
        self._boundary_inputs: list[PortRef] = []
        self._boundary_outputs: list[PortRef] = []
        self._scalar: Scalar = Scalar.one()
        self._next_id: int = 0

    # -- read-only views -----------------------------------------------------------

    @property
    def nodes(self) -> MappingProxyType[NodeId, Node]:
        """A read-only view of every node, keyed by id. Node values are themselves immutable."""
        return MappingProxyType(self._nodes)

    @property
    def wires(self) -> frozenset[Wire]:
        """A read-only snapshot of every wire."""
        return frozenset(self._wires)

    @property
    def boundary_inputs(self) -> tuple[PortRef, ...]:
        """A read-only, ordered snapshot of the boundary-input port references."""
        return tuple(self._boundary_inputs)

    @property
    def boundary_outputs(self) -> tuple[PortRef, ...]:
        """A read-only, ordered snapshot of the boundary-output port references."""
        return tuple(self._boundary_outputs)

    @property
    def scalar(self) -> Scalar:
        """The exact scalar accumulator (Scalar is itself immutable)."""
        return self._scalar

    def __iter__(self) -> Iterator[NodeId]:
        """Iterate over node ids, mirroring dict-like iteration over the node keys."""
        return iter(self._nodes)

    # -- node mutation --------------------------------------------------------------

    def add_node(
        self,
        generator_type: GeneratorType,
        input_dims: Sequence[Dim],
        output_dims: Sequence[Dim],
        phase: PhaseVector | None = None,
    ) -> NodeId:
        """Allocate a fresh NodeId and add a node with the given ports and phase.

        Does not check the generator type's leg policy, dimension policy, or phase
        schema -- that conformance check is :mod:`qufzx.diagram.validate`'s job, so
        that a caller assembling a diagram step by step is never blocked mid-
        construction (e.g. before a phase is attached).
        """
        node_id = NodeId(self._next_id)
        self._next_id += 1
        node = Node(
            id=node_id,
            generator_type=generator_type,
            inputs=tuple(Port(d) for d in input_dims),
            outputs=tuple(Port(d) for d in output_dims),
            phase=phase,
        )
        self._nodes[node_id] = node
        return node_id

    def remove_node(self, node_id: NodeId) -> None:
        """Remove a node, cascading to drop every incident wire and boundary entry.

        Raises GraphGrammarError if ``node_id`` is not present. See the module
        docstring for why removal cascades rather than rejecting nodes with incident
        wires.
        """
        if node_id not in self._nodes:
            raise GraphGrammarError(f"no such node: {node_id!r}")
        del self._nodes[node_id]
        self._wires = {
            wire
            for wire in self._wires
            if wire.a.node_id != node_id and wire.b.node_id != node_id
        }
        self._boundary_inputs = [ref for ref in self._boundary_inputs if ref.node_id != node_id]
        self._boundary_outputs = [ref for ref in self._boundary_outputs if ref.node_id != node_id]

    def set_phase(self, node_id: NodeId, phase: PhaseVector | None) -> None:
        """Replace a node's phase slot. Raises GraphGrammarError if node_id is absent."""
        if node_id not in self._nodes:
            raise GraphGrammarError(f"no such node: {node_id!r}")
        self._nodes[node_id] = self._nodes[node_id].with_phase(phase)

    # -- wire mutation ----------------------------------------------------------------

    def add_wire(self, a: PortRef, b: PortRef) -> None:
        """Add a wire between two ports. Order does not matter.

        Raises GraphDomainError if ``a == b``. Does not check that the referenced
        nodes or port indices exist, that dimensions agree, or that either port is
        already wired or on the boundary -- see the module docstring on validation
        ownership.
        """
        self._wires.add(Wire(a, b))

    def remove_wire(self, a: PortRef, b: PortRef) -> None:
        """Remove the wire between two ports. Raises GraphGrammarError if it is absent."""
        wire = Wire(a, b)
        if wire not in self._wires:
            raise GraphGrammarError(f"no such wire: {wire!r}")
        self._wires.discard(wire)

    # -- boundary mutation ------------------------------------------------------------

    def set_boundary_inputs(self, refs: Sequence[PortRef]) -> None:
        """Replace the ordered boundary-input list wholesale.

        Does not check for duplicates, wire conflicts, or direction consistency --
        see :mod:`qufzx.diagram.validate`.
        """
        self._boundary_inputs = list(refs)

    def set_boundary_outputs(self, refs: Sequence[PortRef]) -> None:
        """Replace the ordered boundary-output list wholesale. See :meth:`set_boundary_inputs`."""
        self._boundary_outputs = list(refs)

    # -- scalar mutation ----------------------------------------------------------------

    def multiply_scalar(self, factor: Scalar) -> None:
        """Multiply the scalar accumulator by ``factor`` in place (Scalar itself is immutable)."""
        if not isinstance(factor, Scalar):
            raise GraphGrammarError(f"multiply_scalar requires a Scalar, got {factor!r}")
        self._scalar = self._scalar * factor

    # -- substitution -----------------------------------------------------------------

    def substitute(
        self,
        mapping: Mapping[str, DimSubstituteValue | PhaseSubstituteValue | ScalarSubstituteValue],
    ) -> Diagram:
        """Return a new Diagram with symbols substituted, preserving every NodeId.

        ``mapping`` is a single symbol-name -> value mapping shared across every port's
        ``Dim``, every node's ``PhaseVector``, and the diagram's ``Scalar``. Unlike a
        single call to one of those three ``substitute()`` methods, this method must
        first split ``mapping`` by each value's *type* before dispatching: each of
        ``Dim.substitute``, ``PhaseVector.substitute``, and ``Scalar.substitute``
        validates every entry of the mapping it is given up front (not only the entries
        that name one of its own free symbols), so a mapping entry meant only for a
        phase symbol -- e.g. an ``sp.Rational`` turns value -- would make
        ``Dim.substitute`` raise even though no ``Dim`` here ever mentions that symbol.
        Splitting by value type (``int`` goes to all three; ``sp.Rational`` goes to the
        phase and scalar mappings; ``Dim``, ``Phase``, and ``Scalar`` values go only to
        their own matching mapping) avoids that false conflict. See the module docstring
        for why this preserves node ids (and thus every ``Wire`` and boundary
        ``PortRef``) rather than being expressible via :meth:`add_node`. This Diagram is
        never mutated.
        """
        dim_mapping: dict[str, DimSubstituteValue] = {}
        phase_mapping: dict[str, PhaseSubstituteValue] = {}
        scalar_mapping: dict[str, ScalarSubstituteValue] = {}
        for name, value in mapping.items():
            if isinstance(value, Dim):
                dim_mapping[name] = value
            elif isinstance(value, Phase):
                phase_mapping[name] = value
            elif isinstance(value, Scalar):
                scalar_mapping[name] = value
            elif isinstance(value, (int, sp.Rational)):
                # int is valid for all three targets; a non-integral sympy Rational is
                # valid only for phase/scalar targets (each downstream substitute()
                # still re-validates and raises its own typed error if inappropriate,
                # e.g. a non-integral Rational reaching a dimension symbol).
                if isinstance(value, int):
                    dim_mapping[name] = value
                phase_mapping[name] = value
                scalar_mapping[name] = value
            else:
                raise GraphGrammarError(
                    f"substitution value for {name!r} must be int, sympy Rational, Dim, "
                    f"Phase, or Scalar, got {type(value).__name__}"
                )

        # Each dict above has str-only keys, a subtype of the corresponding
        # str-or-value-object SymbolKey union every substitute() accepts; Mapping's key
        # type is invariant in the typing sense, so this narrowing needs an explicit
        # (and sound) cast rather than passing the dicts directly.
        typed_dim_mapping = cast(Mapping[DimSymbolKey, DimSubstituteValue], dim_mapping)
        typed_phase_mapping = cast(Mapping[PhaseSymbolKey, PhaseSubstituteValue], phase_mapping)
        typed_scalar_mapping = cast(Mapping[ScalarSymbolKey, ScalarSubstituteValue], scalar_mapping)

        clone = Diagram()
        clone._next_id = self._next_id
        clone._nodes = {
            node_id: Node(
                id=node.id,
                generator_type=node.generator_type,
                inputs=tuple(Port(port.dim.substitute(typed_dim_mapping)) for port in node.inputs),
                outputs=tuple(
                    Port(port.dim.substitute(typed_dim_mapping)) for port in node.outputs
                ),
                phase=(
                    node.phase.substitute(typed_phase_mapping) if node.phase is not None else None
                ),
            )
            for node_id, node in self._nodes.items()
        }
        clone._wires = set(self._wires)
        clone._boundary_inputs = list(self._boundary_inputs)
        clone._boundary_outputs = list(self._boundary_outputs)
        clone._scalar = self._scalar.substitute(typed_scalar_mapping)
        return clone

    # -- copying --------------------------------------------------------------------------

    def copy(self) -> Diagram:
        """Return a fully independent deep copy, preserving node ids.

        Every value stored (Node, Wire, PortRef, PhaseVector, Dim, Scalar) is itself
        immutable, so independence only requires that the *containers* -- the node
        dict, the wire set, and the two boundary lists -- are fresh objects, not that
        their elements be recursively cloned.
        """
        clone = Diagram()
        clone._nodes = dict(self._nodes)
        clone._wires = set(self._wires)
        clone._boundary_inputs = list(self._boundary_inputs)
        clone._boundary_outputs = list(self._boundary_outputs)
        clone._scalar = self._scalar
        clone._next_id = self._next_id
        return clone
