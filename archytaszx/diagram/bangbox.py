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

"""Bang boxes: scoped subgraph multiplicities with instantiate, copy, kill, merge, and nesting.

Two scopes, never mixed in one box, since Phase 7's two worked examples need genuinely
different mechanics: a spider's own leg count growing (the GHZ family) is not the same
operation as a fused pair being replicated wholesale (fusion under a box).

* :attr:`BangBox.port_scope` -- ports on an otherwise-untouched node; instantiating at k
  grows each scoped port to k legs of the node it is on, in place. No node is replicated,
  so this is the mechanism that makes one spider denote the whole GHZ family.
* :attr:`BangBox.node_scope` -- whole nodes; instantiating at k builds k independent
  fresh copies of the scope and its internal wiring.

Exactly one is non-empty per box (:meth:`BangBox.__post_init__`). A crossing attachment
(a wire leaving scope, or a scoped port) is only supported when it lands on the diagram's
own boundary lists -- both Phase 7 worked examples only ever cross there; wiring a box's
interior to an untouched sibling node is future work.

Nesting. A child's ``parent`` names its enclosing box. Instantiating a node-scope box at
k >= 2 also replicates every child once per new copy, *keeping its multiplicity symbol
unrenamed* -- the copies still denote one shared, not-yet-supplied count.
:func:`instantiate_symbol`, not a single box id, is therefore Phase 7's unit of
instantiation: it acts on every live box currently carrying one bare symbol together, so
a nested two-index family instantiates correctly regardless of which index goes first.

Multiplicities: :class:`Mult`, not a :class:`~archytaszx.algebra.dimension.Dim`. ``Dim``'s
grammar deliberately rejects sums and floors at 1, while a multiplicity needs sums and
admits 0 (``kill``) -- reusing ``Dim`` would weaken it for every dimension in the system.
``Mult`` mirrors ``Dim``'s shape (concrete int, symbol, sum, product; canonical sympy
expression) but is never interchangeable with it. Its symbols carry an inert
``multiplicity=True`` marker so :mod:`archytaszx.diagram.validate` never confuses one with a
dimension's exponent, which shares the same two real assumptions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef, Wire


class BangBoxError(Exception):
    """Base class for all errors raised by this module."""


class BangBoxDomainError(BangBoxError):
    """A value is outside the mathematical domain this module accepts.

    Raised for a negative multiplicity, and for :func:`instantiate_symbol` or
    :func:`kill` being asked to resolve a non-concrete or negative count.
    """


class BangBoxGrammarError(BangBoxError):
    """A request is malformed: wrong type, an empty or mixed scope, an unknown box or
    symbol, boxes that are not siblings, or a crossing this phase does not support.
    """


def _check_symbol_name(name: str) -> None:
    if not isinstance(name, str) or not name.isidentifier():
        raise BangBoxGrammarError(f"symbol name must be a bare identifier, got {name!r}")


def _mult_symbol(name: str) -> sp.Symbol:
    _check_symbol_name(name)
    return sp.Symbol(name, integer=True, nonnegative=True, multiplicity=True)


def _check_mult_domain(expr: sp.Expr) -> None:
    if expr.is_Integer:
        if int(expr) < 0:
            raise BangBoxDomainError(f"concrete multiplicity must be >= 0, got {expr}")
        return
    if expr.is_Symbol:
        assumptions = expr.assumptions0
        if not (assumptions.get("integer") and assumptions.get("nonnegative")):
            raise BangBoxGrammarError(
                f"symbol {expr} is not a valid multiplicity symbol "
                "(multiplicity symbols must be non-negative integers)"
            )
        _check_symbol_name(str(expr.name))
        return
    if expr.is_Add or expr.is_Mul:
        for arg in expr.args:
            _check_mult_domain(arg)
        return
    raise BangBoxGrammarError(
        f"expression {expr} is outside the multiplicity grammar "
        "(only concrete non-negative integers, symbols, sums, and products are allowed)"
    )


MultSubstituteValue = int


class Mult:
    """An immutable, hashable multiplicity expression: a non-negative integer, possibly
    symbolic, built from concrete counts, symbols, sums, and products. See the module
    docstring for why this is not a :class:`~archytaszx.algebra.dimension.Dim`.
    """

    __slots__ = ("_expr",)
    _expr: sp.Expr

    def __init__(self, value: int | str | Mult) -> None:
        """Build a Mult from a concrete non-negative int, a symbol name, or a copy of
        another Mult."""
        if isinstance(value, bool):
            raise BangBoxGrammarError(f"Mult does not accept bool, got {value!r}")
        if isinstance(value, Mult):
            self._expr = value._expr
        elif isinstance(value, int):
            if value < 0:
                raise BangBoxDomainError(f"concrete multiplicity must be >= 0, got {value}")
            self._expr = sp.Integer(value)
        elif isinstance(value, str):
            self._expr = _mult_symbol(value)
        else:
            raise TypeError(f"Mult() accepts int, str, or Mult, got {type(value).__name__}")

    @classmethod
    def concrete(cls, value: int) -> Mult:
        """Build a Mult from a concrete non-negative integer. Alias for Mult(value)."""
        return cls(value)

    @classmethod
    def symbol(cls, name: str) -> Mult:
        """Build a Mult from a fresh non-negative-integer symbol name. Alias for Mult(name)."""
        return cls(name)

    @classmethod
    def _from_expr(cls, expr: sp.Expr) -> Mult:
        normalized = sp.sympify(expr)
        _check_mult_domain(normalized)
        obj = cls.__new__(cls)
        obj._expr = normalized
        return obj

    def to_sympy(self) -> sp.Expr:
        """Escape hatch returning the underlying canonical sympy expression."""
        return self._expr

    @property
    def is_concrete(self) -> bool:
        """True iff this expression has no free symbols."""
        return not self._expr.free_symbols

    @property
    def is_bare_symbol(self) -> bool:
        """True iff this expression is a single symbol, not a sum/product/concrete value.

        Distinguishes a box that *owns* a name (this) from one that merely *uses* it as
        a subterm of a compound expression.
        """
        return bool(self._expr.is_Symbol)

    def bare_symbol_name(self) -> str:
        """The name of this Mult's symbol. Raises BangBoxGrammarError if not a bare symbol."""
        if not self.is_bare_symbol:
            raise BangBoxGrammarError(f"{self} is not a bare symbol")
        return str(self._expr.name)

    @property
    def free_symbols(self) -> frozenset[str]:
        """The names of all symbols appearing in this expression."""
        return frozenset(str(s.name) for s in self._expr.free_symbols)

    def to_int(self) -> int:
        """Return the concrete int value of this multiplicity. Raises BangBoxDomainError
        if symbolic."""
        if not self.is_concrete:
            raise BangBoxDomainError(
                f"cannot convert symbolic multiplicity {self} to int; "
                f"free symbols: {sorted(self.free_symbols)}"
            )
        return int(self._expr)

    def substitute(self, mapping: Mapping[str, MultSubstituteValue]) -> Mult:
        """Return a new Mult with symbols replaced by concrete non-negative integers.

        The mapping need not be total: symbols not mentioned are left symbolic. This
        Mult is never mutated.
        """
        subs_dict: dict[sp.Symbol, sp.Integer] = {}
        for sym in sorted(self._expr.free_symbols, key=lambda s: str(s.name)):
            name = str(sym.name)
            if name not in mapping:
                continue
            value = mapping[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"substitution value for {name!r} must be int, got {value!r}")
            if value < 0:
                raise BangBoxDomainError(
                    f"multiplicity symbol {name!r} requires a non-negative integer, got {value}"
                )
            subs_dict[sym] = sp.Integer(value)
        new_expr = self._expr.subs(subs_dict)
        return Mult._from_expr(new_expr)

    def abstract(
        self, *, avoid: Iterable[str] = (), stem: str = "n"
    ) -> tuple[Mult, Mapping[str, int]]:
        """Return a fresh symbol standing for this concrete multiplicity, and its binding.

        Mirrors :meth:`Dim.abstract` exactly. Raises BangBoxDomainError on a symbolic Mult.
        """
        if not self.is_concrete:
            raise BangBoxDomainError(f"cannot abstract symbolic multiplicity {self}")
        _check_symbol_name(stem)
        taken = set(avoid)
        name = stem
        suffix = 0
        while name in taken:
            suffix += 1
            name = f"{stem}{suffix}"
        return Mult.symbol(name), MappingProxyType({name: self.to_int()})

    def __add__(self, other: Mult | int) -> Mult:
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            other = Mult(other)
        if not isinstance(other, Mult):
            return NotImplemented
        return Mult._from_expr(self._expr + other._expr)

    def __radd__(self, other: Mult | int) -> Mult:
        return self.__add__(other)

    def __mul__(self, other: Mult | int) -> Mult:
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            other = Mult(other)
        if not isinstance(other, Mult):
            return NotImplemented
        return Mult._from_expr(self._expr * other._expr)

    def __rmul__(self, other: Mult | int) -> Mult:
        return self.__mul__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mult):
            return NotImplemented
        return bool(self._expr == other._expr)

    def __hash__(self) -> int:
        return hash(self._expr)

    def __repr__(self) -> str:
        return f"Mult({self._expr})"

    def __str__(self) -> str:
        return str(self._expr)


@dataclass(frozen=True, slots=True)
class BangBox:
    """One bang box: a scope (node-mode xor port-mode), a multiplicity, and a parent link.

    See the module docstring for the two scope modes and why they are not unified.
    """

    id: BangBoxId
    multiplicity: Mult
    node_scope: frozenset[NodeId] = frozenset()
    port_scope: frozenset[PortRef] = frozenset()
    parent: BangBoxId | None = None

    def __post_init__(self) -> None:
        """Validate field types and that exactly one of the two scopes is non-empty."""
        if not isinstance(self.multiplicity, Mult):
            raise BangBoxGrammarError(
                f"BangBox multiplicity must be a Mult, got {self.multiplicity!r}"
            )
        if not isinstance(self.node_scope, frozenset) or not isinstance(self.port_scope, frozenset):
            raise BangBoxGrammarError("BangBox node_scope and port_scope must be frozensets")
        for node_id in self.node_scope:
            if not isinstance(node_id, int):
                raise BangBoxGrammarError(
                    f"BangBox node_scope entries must be NodeId, got {node_id!r}"
                )
        for ref in self.port_scope:
            if not isinstance(ref, PortRef):
                raise BangBoxGrammarError(
                    f"BangBox port_scope entries must be PortRef, got {ref!r}"
                )
        if bool(self.node_scope) == bool(self.port_scope):
            raise BangBoxGrammarError(
                "BangBox must have exactly one of node_scope or port_scope non-empty, got "
                f"node_scope={self.node_scope!r}, port_scope={self.port_scope!r}"
            )

    @property
    def is_node_scope(self) -> bool:
        """True iff this box replicates whole nodes (as opposed to growing scoped ports)."""
        return bool(self.node_scope)

    def with_node_scope(self, node_scope: frozenset[NodeId]) -> BangBox:
        """Return a copy of this (node-scope) box with its scope replaced."""
        return replace(self, node_scope=frozenset(node_scope))

    def with_port_scope(self, port_scope: frozenset[PortRef]) -> BangBox:
        """Return a copy of this (port-scope) box with its scope replaced."""
        return replace(self, port_scope=frozenset(port_scope))

    def with_multiplicity(self, multiplicity: Mult) -> BangBox:
        """A copy of this box carrying ``multiplicity``."""
        return replace(self, multiplicity=multiplicity)


# -- structural helpers, shared with archytaszx.diagram.validate ---------------------------


def crossing_wires(diagram: Diagram, node_scope: frozenset[NodeId]) -> frozenset[Wire]:
    """Every wire with exactly one endpoint's node inside ``node_scope``."""
    return frozenset(
        w for w in diagram.wires if (w.a.node_id in node_scope) != (w.b.node_id in node_scope)
    )


def internal_wires(diagram: Diagram, node_scope: frozenset[NodeId]) -> frozenset[Wire]:
    """Every wire with both endpoints' nodes inside ``node_scope``."""
    return frozenset(
        w for w in diagram.wires if w.a.node_id in node_scope and w.b.node_id in node_scope
    )


def boundary_refs_in_scope(diagram: Diagram, node_scope: frozenset[NodeId]) -> frozenset[PortRef]:
    """Every diagram boundary port (input or output) whose node lies inside ``node_scope``."""
    return frozenset(
        ref
        for ref in (*diagram.boundary_inputs, *diagram.boundary_outputs)
        if ref.node_id in node_scope
    )


def _non_boundary_crossing(diagram: Diagram, node_scope: frozenset[NodeId]) -> Wire | None:
    """The first crossing wire (if any) that does not land on the diagram boundary.

    Every crossing wire's outside endpoint is on some other live node; a crossing wire is
    only supported here when that outside endpoint is a diagram boundary port -- see the
    module docstring's Phase 7 scope restriction. Returns ``None`` when every crossing is
    boundary-only.
    """
    boundary = set(diagram.boundary_inputs) | set(diagram.boundary_outputs)
    for wire in crossing_wires(diagram, node_scope):
        outside_ref = wire.a if wire.a.node_id not in node_scope else wire.b
        if outside_ref not in boundary:
            return wire
    return None


# -- abstraction ------------------------------------------------------------------------


def _existing_symbol_names(diagram: Diagram) -> set[str]:
    """Every dimension/phase/scalar symbol name and every multiplicity symbol name
    already present in ``diagram``, plus every key already in its parameter environment.
    """
    names: set[str] = set(diagram.parameters)
    for node in diagram.nodes.values():
        for port in (*node.inputs, *node.outputs):
            names |= port.dim.free_symbols
        if node.phase is not None:
            names |= node.phase.free_symbols
    names |= diagram.scalar.free_symbols
    for box in diagram.bang_boxes.values():
        names |= box.multiplicity.free_symbols
    return names


def abstract_port_count(
    diagram: Diagram, ref: PortRef, concrete_k: int, *, stem: str = "n"
) -> tuple[Diagram, BangBoxId, Mult]:
    """Abstract a concrete leg count at ``ref`` into a fresh port-scope bang box.

    Mirrors :meth:`~archytaszx.algebra.dimension.Dim.abstract`: mints a symbol absent from
    every symbol name (dimension, phase, scalar, or multiplicity) and every parameter-
    environment key already in ``diagram``, builds a port-scope box over ``{ref}`` bound
    to that symbol, binds the symbol to ``concrete_k`` in the parameter environment, and
    returns the mutated (copied) diagram, the new box's id, and the symbol as a ``Mult``.
    """
    working = diagram.copy()
    avoid = _existing_symbol_names(working)
    symbol, binding = Mult(concrete_k).abstract(avoid=avoid, stem=stem)
    box_id = working.add_bang_box(symbol, port_scope=frozenset({ref}), parent=None)
    for name, value in binding.items():
        working.bind_parameter(name, value)
    return working, box_id, symbol


def abstract_subgraph_count(
    diagram: Diagram,
    node_scope: frozenset[NodeId],
    concrete_k: int,
    *,
    parent: BangBoxId | None = None,
    stem: str = "n",
) -> tuple[Diagram, BangBoxId, Mult]:
    """Abstract a concrete copy count over ``node_scope`` into a fresh node-scope bang box.

    See :func:`abstract_port_count`; identical contract, node-scope instead of port-scope.
    """
    working = diagram.copy()
    avoid = _existing_symbol_names(working)
    symbol, binding = Mult(concrete_k).abstract(avoid=avoid, stem=stem)
    box_id = working.add_bang_box(symbol, node_scope=frozenset(node_scope), parent=parent)
    for name, value in binding.items():
        working.bind_parameter(name, value)
    return working, box_id, symbol


# -- queries -----------------------------------------------------------------------------


def free_mult_symbols(diagram: Diagram) -> frozenset[str]:
    """Every free symbol appearing in any live bang box's multiplicity expression."""
    symbols: set[str] = set()
    for box in diagram.bang_boxes.values():
        symbols |= box.multiplicity.free_symbols
    return frozenset(symbols)


def _purge_symbol_if_dead(diagram: Diagram, name: str | None) -> None:
    """Drop ``name`` from the parameter environment if no live box still carries it.

    A box's own instantiation always consumes its *own* target symbol's binding (see
    :func:`instantiate_symbol`), but killing a node-scope box also kills every child --
    each carrying its own, generally *different*, symbol -- as a side effect the caller
    never named. Without this, a killed child's leftover parameter binding is a pending
    substitution against nothing, which :mod:`archytaszx.semantics.contract_numeric` refuses
    outright even though :mod:`archytaszx.diagram.validate` only defers it.
    """
    if name is not None and name not in free_mult_symbols(diagram) and name in diagram.parameters:
        diagram.set_parameters({k: v for k, v in diagram.parameters.items() if k != name})


# -- port-scope instantiation (the GHZ-family mechanism) -------------------------------


def _remap_grown_port(
    old: PortRef, node_id: NodeId, new_node_id: NodeId, direction: Direction, index: int, k: int
) -> list[PortRef]:
    """Where one old port on ``node_id`` lands on ``new_node_id`` once the port at
    ``(direction, index)`` has grown from one leg to ``k`` legs (or shrunk to zero).

    Every other port on this node keeps its direction and shifts index only if it comes
    after the grown slot on the *same* direction; a port on the other direction, or
    before the grown slot, keeps its index unchanged. The grown slot itself expands to
    ``k`` consecutive new refs (an empty list when ``k == 0``).
    """
    if old.node_id != node_id:
        return [old]
    if old.direction is not direction:
        return [PortRef(new_node_id, old.direction, old.index)]
    if old.index < index:
        return [PortRef(new_node_id, old.direction, old.index)]
    if old.index == index:
        return [PortRef(new_node_id, old.direction, index + i) for i in range(k)]
    return [PortRef(new_node_id, old.direction, old.index + (k - 1))]


def _grow_port(
    diagram: Diagram, ref: PortRef, k: int, *, owner_box_id: BangBoxId | None = None
) -> None:
    """Grow (or, at ``k == 0``, remove) the single leg at ``ref`` to ``k`` legs, in place.

    Requires ``ref`` to currently be a diagram boundary slot -- see the module
    docstring's Phase 7 scope restriction; a leg wired to another node cannot grow
    without that node growing too, which this phase does not attempt.

    ``owner_box_id``, when given, is excluded from the "repoint every other box that
    named this node/port" fixup below: it is the box currently being instantiated (the
    caller deletes it immediately afterwards regardless of what its scope says), and at
    ``k == 0`` remapping its own about-to-vanish scoped port would otherwise leave it
    with neither scope populated, tripping :class:`BangBox`'s own invariant before the
    caller gets a chance to delete it.
    """
    node = diagram.nodes.get(ref.node_id)
    if node is None:
        raise BangBoxGrammarError(f"port-scope bang box references unknown node {ref.node_id!r}")
    legs = node.legs(ref.direction)
    if ref.index >= len(legs):
        raise BangBoxGrammarError(f"port-scope bang box port {ref!r} is out of range")
    if ref not in diagram.boundary_inputs and ref not in diagram.boundary_outputs:
        raise BangBoxGrammarError(
            f"port-scope instantiate only supports a scoped port that is a diagram "
            f"boundary slot; {ref!r} is wired internally, which this phase does not "
            "support (see archytaszx.diagram.bangbox's module docstring)"
        )

    target_dim = legs[ref.index].dim
    old_input_dims = [p.dim for p in node.inputs]
    old_output_dims = [p.dim for p in node.outputs]
    if ref.direction is Direction.INPUT:
        new_input_dims = (
            old_input_dims[: ref.index] + [target_dim] * k + old_input_dims[ref.index + 1 :]
        )
        new_output_dims = old_output_dims
    else:
        new_input_dims = old_input_dims
        new_output_dims = (
            old_output_dims[: ref.index] + [target_dim] * k + old_output_dims[ref.index + 1 :]
        )

    old_wires = [
        w for w in diagram.wires if w.a.node_id == ref.node_id or w.b.node_id == ref.node_id
    ]
    old_boundary_inputs = list(diagram.boundary_inputs)
    old_boundary_outputs = list(diagram.boundary_outputs)

    new_node_id = diagram.add_node(
        node.generator_type, new_input_dims, new_output_dims, phase=node.phase
    )

    def remap(old_ref: PortRef) -> list[PortRef]:
        return _remap_grown_port(old_ref, ref.node_id, new_node_id, ref.direction, ref.index, k)

    for wire in old_wires:
        this_end = wire.a if wire.a.node_id == ref.node_id else wire.b
        other_end = wire.b if wire.a.node_id == ref.node_id else wire.a
        # this_end is never the grown port itself: that would require it to be both a
        # wire endpoint and a boundary slot, which _check_port_usage already treats as
        # ill-formed input, and the guard above already requires the grown port to be a
        # boundary slot, so a wired grown port never reaches here in a well-formed diagram.
        (remapped,) = remap(this_end)
        diagram.add_wire(remapped, other_end)

    new_boundary_inputs = [
        new_ref
        for old_ref in old_boundary_inputs
        for new_ref in (remap(old_ref) if old_ref.node_id == ref.node_id else [old_ref])
    ]
    new_boundary_outputs = [
        new_ref
        for old_ref in old_boundary_outputs
        for new_ref in (remap(old_ref) if old_ref.node_id == ref.node_id else [old_ref])
    ]

    # Any *other* bang box (an outer node-scope box that has not yet been told this leg
    # grew, or another port-scope box naming a different leg of this same node) must be
    # repointed at new_node_id too, or it goes stale the moment ref.node_id is removed
    # below -- this is exactly Phase 7's nested-family case: an outer node-scope box's
    # node_scope names this very node directly, not through a child-box declaration.
    for other_id, other in list(diagram.bang_boxes.items()):
        if other_id == owner_box_id:
            continue
        if ref.node_id in other.node_scope:
            diagram.set_bang_box_node_scope(
                other_id,
                frozenset(new_node_id if n == ref.node_id else n for n in other.node_scope),
            )
        if any(p.node_id == ref.node_id for p in other.port_scope):
            diagram.set_bang_box_port_scope(
                other_id,
                frozenset(
                    new_ref
                    for p in other.port_scope
                    for new_ref in (remap(p) if p.node_id == ref.node_id else [p])
                ),
            )

    diagram.remove_node(ref.node_id)
    diagram.set_boundary_inputs(new_boundary_inputs)
    diagram.set_boundary_outputs(new_boundary_outputs)


def _instantiate_port_scope(diagram: Diagram, box: BangBox, k: int) -> None:
    for ref in sorted(box.port_scope, key=lambda r: r.sort_key()):
        _grow_port(diagram, ref, k, owner_box_id=box.id)
    diagram.remove_bang_box(box.id)


# -- node-scope instantiation (the repeated-subgraph mechanism) ------------------------


def _children_of(diagram: Diagram, box_id: BangBoxId) -> tuple[BangBox, ...]:
    return tuple(box for _, box in sorted(diagram.bang_boxes.items()) if box.parent == box_id)


def _splice_boundary_block(
    refs: list[PortRef], all_old_refs: list[PortRef], id_maps: list[dict[NodeId, NodeId]]
) -> list[PortRef]:
    """Replace every ref in ``old_refs`` (in list order) with one contiguous, copy-major
    block at the position of ``old_refs[0]``: copy 0's remap of every old ref, then copy
    1's, and so on.

    Copy-major, not one independent per-ref splice, and done as a single batch: growing
    a node's boundary legs one ref at a time (each spliced into its own position
    independently) interleaves copies whenever the scope crosses the boundary at more
    than one ref, which silently depends on the order those refs happen to be visited in
    -- exactly the axis order Phase 7's order-independence test would otherwise see
    differ between "instantiate the inner index first" (which grows a single ref to many
    before any node-copying happens, and so is trivially copy-major already) and
    "instantiate the outer index first" (which, without this block-splice, would
    interleave several already-existing refs per copy). One batch call fixes both orders
    to the same canonical layout.
    """
    old_set = set(refs) & set(all_old_refs)
    if not old_set:
        return refs
    # Preserve the original list's relative order among the crossing refs that belong to
    # *this* list (inputs or outputs), not all_old_refs' own (arbitrary, sort_key) order.
    old_refs_here = [r for r in refs if r in old_set]
    insert_at = next(i for i, r in enumerate(refs) if r in old_set)
    block = [
        PortRef(id_map[old.node_id], old.direction, old.index)
        for id_map in id_maps
        for old in old_refs_here
    ]
    before = [r for r in refs[:insert_at] if r not in old_set]
    after = [r for r in refs[insert_at:] if r not in old_set]
    return before + block + after


def _instantiate_node_scope(diagram: Diagram, box: BangBox, k: int) -> None:
    scope = box.node_scope
    bad_wire = _non_boundary_crossing(diagram, scope)
    if bad_wire is not None:
        raise BangBoxGrammarError(
            f"node-scope instantiate only supports a crossing that lands on the diagram "
            f"boundary; {bad_wire!r} crosses to another live node, which this phase does "
            "not support (see archytaszx.diagram.bangbox's module docstring)"
        )

    children = _children_of(diagram, box.id)

    if k == 0:
        for child in children:
            child_symbol = (
                child.multiplicity.bare_symbol_name() if child.multiplicity.is_bare_symbol else None
            )
            _kill_one(diagram, child)
            _purge_symbol_if_dead(diagram, child_symbol)
        # Re-read: killing a port-scope child regrows (replaces) the node it names, and
        # that node may be exactly one of this box's own scope nodes -- box.node_scope,
        # captured above as `scope`, would otherwise still name the now-removed old id.
        # _grow_port's fixup (triggered by _kill_one -> _instantiate_port_scope ->
        # _grow_port) already repointed diagram.bang_boxes[box.id] itself; only this
        # function's own local snapshot goes stale.
        scope = diagram.bang_boxes[box.id].node_scope
        for node_id in scope:
            diagram.remove_node(node_id)
        diagram.remove_bang_box(box.id)
        return

    if k == 1:
        # The one copy keeps the nodes it already has, but the boundary still goes through
        # the same block splice k >= 2 performs: without it a scope whose crossings are not
        # already contiguous would order its boundary differently at 1 than at 2.
        identity = {node_id: node_id for node_id in scope}
        sorted_crossings = sorted(
            boundary_refs_in_scope(diagram, scope), key=lambda r: r.sort_key()
        )
        diagram.set_boundary_inputs(
            _splice_boundary_block(list(diagram.boundary_inputs), sorted_crossings, [identity])
        )
        diagram.set_boundary_outputs(
            _splice_boundary_block(list(diagram.boundary_outputs), sorted_crossings, [identity])
        )
        for child in children:
            diagram.remove_bang_box(child.id)
            diagram.add_bang_box(
                child.multiplicity,
                node_scope=child.node_scope,
                port_scope=child.port_scope,
                parent=box.parent,
            )
        diagram.remove_bang_box(box.id)
        return

    internal = internal_wires(diagram, scope)
    boundary_crossings = boundary_refs_in_scope(diagram, scope)

    id_maps: list[dict[NodeId, NodeId]] = []
    for _ in range(k):
        id_map: dict[NodeId, NodeId] = {}
        for old_id in sorted(scope):
            node = diagram.nodes[old_id]
            new_id = diagram.add_node(
                node.generator_type,
                [p.dim for p in node.inputs],
                [p.dim for p in node.outputs],
                phase=node.phase,
            )
            id_map[old_id] = new_id
        id_maps.append(id_map)

    def remap_ref(ref: PortRef, id_map: dict[NodeId, NodeId]) -> PortRef:
        return PortRef(id_map[ref.node_id], ref.direction, ref.index)

    for id_map in id_maps:
        for wire in sorted(internal, key=lambda w: w.sort_key()):
            diagram.add_wire(remap_ref(wire.a, id_map), remap_ref(wire.b, id_map))

    sorted_crossings = sorted(boundary_crossings, key=lambda r: r.sort_key())
    new_boundary_inputs = _splice_boundary_block(
        list(diagram.boundary_inputs), sorted_crossings, id_maps
    )
    new_boundary_outputs = _splice_boundary_block(
        list(diagram.boundary_outputs), sorted_crossings, id_maps
    )

    for child in children:
        for id_map in id_maps:
            if child.is_node_scope:
                diagram.add_bang_box(
                    child.multiplicity,
                    node_scope=frozenset(id_map[n] for n in child.node_scope),
                    parent=box.parent,
                )
            else:
                diagram.add_bang_box(
                    child.multiplicity,
                    port_scope=frozenset(
                        PortRef(id_map[r.node_id], r.direction, r.index) for r in child.port_scope
                    ),
                    parent=box.parent,
                )
        diagram.remove_bang_box(child.id)

    for node_id in scope:
        diagram.remove_node(node_id)
    diagram.set_boundary_inputs(new_boundary_inputs)
    diagram.set_boundary_outputs(new_boundary_outputs)
    diagram.remove_bang_box(box.id)


def _kill_one(diagram: Diagram, box: BangBox) -> None:
    if box.is_node_scope:
        _instantiate_node_scope(diagram, box, 0)
    else:
        _instantiate_port_scope(diagram, box, 0)


def _instantiate_one(diagram: Diagram, box_id: BangBoxId, k: int) -> None:
    box = diagram.bang_boxes[box_id]
    if box.is_node_scope:
        _instantiate_node_scope(diagram, box, k)
    else:
        _instantiate_port_scope(diagram, box, k)


# -- public entry points -----------------------------------------------------------------


_MAX_INSTANTIATE_ROUNDS = 10_000


def instantiate_symbol(diagram: Diagram, name: str, value: int) -> Diagram:
    """Substitute ``value`` for ``name`` in every live bang box's multiplicity, expanding
    each box whose multiplicity thereby becomes concrete.

    This -- not a single box id -- is Phase 7's unit of instantiation: nesting can
    multiply how many boxes carry one shared, not-yet-supplied count (see the module
    docstring), and instantiating only one of them would leave the others still
    denoting the same free variable at a different value. A compound multiplicity
    (``2*k``, ``k + 1``, ``k1 + k2``) is substituted the same way; one still carrying
    another symbol afterwards keeps its box until that symbol is supplied too. Boxes are
    taken one at a time and the diagram rescanned, so copies a parent's expansion makes
    of a child that also carries ``name`` are reached in turn. Raises BangBoxGrammarError
    if no live box carries this symbol, and BangBoxDomainError if ``value`` is not a
    non-negative int.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BangBoxDomainError(f"instantiate_symbol requires a non-negative int, got {value!r}")
    working = diagram.copy()
    if not any(name in box.multiplicity.free_symbols for box in working.bang_boxes.values()):
        raise BangBoxGrammarError(f"no live bang box has multiplicity symbol {name!r}")
    for _round in range(_MAX_INSTANTIATE_ROUNDS):
        targets = [
            box_id
            for box_id, box in sorted(working.bang_boxes.items())
            if name in box.multiplicity.free_symbols
        ]
        if not targets:
            break
        box_id = targets[0]
        substituted = working.bang_boxes[box_id].multiplicity.substitute({name: value})
        working.set_bang_box_multiplicity(box_id, substituted)
        if substituted.is_concrete:
            _instantiate_one(working, box_id, substituted.to_int())
    else:
        raise BangBoxGrammarError(
            f"instantiating {name!r} did not settle within {_MAX_INSTANTIATE_ROUNDS} rounds"
        )
    if name in working.parameters:
        # Mirrors Diagram.substitute's "entries the mapping names are consumed from
        # parameters" contract: the symbol is now gone from every box, so a lingering
        # binding for it is a pending substitution against nothing.
        working.set_parameters({k: v for k, v in working.parameters.items() if k != name})
    return working


def _predecessor(multiplicity: Mult) -> Mult:
    """``multiplicity`` less one, for an expression whose constant term is at least one."""
    expr = sp.expand(multiplicity.to_sympy())
    constant = expr.as_coeff_Add()[0]
    if not (constant.is_Integer and constant >= 1):
        raise BangBoxDomainError(
            f"cannot peel a copy off multiplicity {multiplicity}: its constant term is not "
            "a positive integer, so it is not known to be at least one"
        )
    return Mult._from_expr(sp.expand(expr - 1))


def _peel_port_scope(diagram: Diagram, box: BangBox) -> None:
    """Split the scoped leg into the box's own leg plus one bare leg after it."""
    (ref,) = sorted(box.port_scope, key=lambda r: r.sort_key())
    _grow_port(diagram, ref, 2)
    grown = sorted(diagram.bang_boxes[box.id].port_scope, key=lambda r: r.sort_key())
    if len(grown) != 2:
        raise BangBoxGrammarError(
            f"peeling port-scope box {box.id!r} grew its scope to {len(grown)} legs, not two"
        )
    diagram.set_bang_box_port_scope(box.id, frozenset({grown[0]}))


def _peel_node_scope(diagram: Diagram, box: BangBox) -> None:
    """Add one concrete copy of the scope, after the box's own boundary block."""
    scope = box.node_scope
    bad_wire = _non_boundary_crossing(diagram, scope)
    if bad_wire is not None:
        raise BangBoxGrammarError(
            f"node-scope peel only supports a crossing that lands on the diagram boundary; "
            f"{bad_wire!r} crosses to another live node"
        )

    internal = internal_wires(diagram, scope)
    boundary_crossings = boundary_refs_in_scope(diagram, scope)
    children = _children_of(diagram, box.id)

    id_map: dict[NodeId, NodeId] = {}
    for old_id in sorted(scope):
        node = diagram.nodes[old_id]
        id_map[old_id] = diagram.add_node(
            node.generator_type,
            [p.dim for p in node.inputs],
            [p.dim for p in node.outputs],
            phase=node.phase,
        )

    for wire in sorted(internal, key=lambda w: w.sort_key()):
        diagram.add_wire(
            PortRef(id_map[wire.a.node_id], wire.a.direction, wire.a.index),
            PortRef(id_map[wire.b.node_id], wire.b.direction, wire.b.index),
        )

    identity = {old_id: old_id for old_id in scope}
    sorted_crossings = sorted(boundary_crossings, key=lambda r: r.sort_key())
    id_maps = [identity, id_map]
    diagram.set_boundary_inputs(
        _splice_boundary_block(list(diagram.boundary_inputs), sorted_crossings, id_maps)
    )
    diagram.set_boundary_outputs(
        _splice_boundary_block(list(diagram.boundary_outputs), sorted_crossings, id_maps)
    )

    for child in children:
        if child.is_node_scope:
            diagram.add_bang_box(
                child.multiplicity,
                node_scope=frozenset(id_map[n] for n in child.node_scope),
                parent=box.parent,
            )
        else:
            diagram.add_bang_box(
                child.multiplicity,
                port_scope=frozenset(
                    PortRef(id_map[r.node_id], r.direction, r.index) for r in child.port_scope
                ),
                parent=box.parent,
            )


@dataclass(frozen=True, slots=True)
class PeelResult:
    """The outcome of :func:`peel_one`: the split diagram and the copy it split off.

    ``copy_node_ids`` is empty for a port-scope peel, where the peeled leg grows on the
    box's own node rather than onto a separate copy; ``separable`` records that distinction.
    """

    diagram: Diagram
    copy_node_ids: frozenset[NodeId]
    separable: bool


def peel_one(diagram: Diagram, box_id: BangBoxId) -> PeelResult:
    """Split one copy off a bang box: multiplicity ``m`` becomes ``m - 1`` beside one bare copy.

    The peeled copy is laid out after the box's own boundary block, so instantiating the
    residual box at ``k`` reproduces exactly what instantiating the original at ``k + 1``
    would: :func:`instantiate_symbol` builds its copies copy-major from the same splice.
    Requires a multiplicity whose constant term is at least one.
    """
    working = diagram.copy()
    box = working.bang_boxes.get(box_id)
    if box is None:
        raise BangBoxGrammarError(f"no such bang box: {box_id!r}")
    predecessor = _predecessor(box.multiplicity)
    before = frozenset(working.nodes)
    if box.is_node_scope:
        _peel_node_scope(working, box)
    else:
        _peel_port_scope(working, box)
    working.set_bang_box_multiplicity(box_id, predecessor)
    separable = box.is_node_scope
    copy_node_ids = frozenset(working.nodes) - before if separable else frozenset()
    return PeelResult(working, copy_node_ids, separable)


def expand_concrete_boxes(diagram: Diagram) -> Diagram:
    """Expand every bang box whose multiplicity is already a concrete number.

    Boxes are taken one at a time and the diagram rescanned, so a box a parent's expansion
    copies is reached in turn. Returns a new Diagram.
    """
    working = diagram.copy()
    for _round in range(_MAX_INSTANTIATE_ROUNDS):
        targets = [
            box_id
            for box_id, box in sorted(working.bang_boxes.items())
            if box.multiplicity.is_concrete
        ]
        if not targets:
            return working
        box_id = targets[0]
        _instantiate_one(working, box_id, working.bang_boxes[box_id].multiplicity.to_int())
    raise BangBoxGrammarError(
        f"expanding concrete bang boxes did not settle within {_MAX_INSTANTIATE_ROUNDS} rounds"
    )


def scope_is_closed(diagram: Diagram, node_scope: frozenset[NodeId]) -> bool:
    """Whether ``node_scope`` meets the rest of the diagram at no wire and no boundary slot."""
    for wire in diagram.wires:
        if (wire.a.node_id in node_scope) != (wire.b.node_id in node_scope):
            return False
    return not any(
        ref.node_id in node_scope for ref in (*diagram.boundary_inputs, *diagram.boundary_outputs)
    )


def kill(diagram: Diagram, box_id: BangBoxId) -> Diagram:
    """Instantiate one box at multiplicity 0: its scope (and, for a node-scope box, every
    nested child) vanishes entirely. Returns a new Diagram."""
    working = diagram.copy()
    box = working.bang_boxes.get(box_id)
    if box is None:
        raise BangBoxGrammarError(f"no such bang box: {box_id!r}")
    _kill_one(working, box)
    return working


def copy_box(diagram: Diagram, box_id: BangBoxId) -> tuple[Diagram, BangBoxId]:
    """Structurally duplicate a node-scope bang box: a fresh, independent copy of its
    scope and internal wiring, under a fresh multiplicity symbol never unified with the
    original's.

    Node-scope only -- a port-scope box has no subgraph to duplicate independently of
    the single node it grows (see the module docstring); calling this on a port-scope
    box raises BangBoxGrammarError. Boundary crossings of the duplicated scope are left
    dangling on fresh diagram boundary slots appended to the appropriate list, so the
    result is always well formed without guessing where the caller wants them wired.
    """
    working = diagram.copy()
    box = working.bang_boxes.get(box_id)
    if box is None:
        raise BangBoxGrammarError(f"no such bang box: {box_id!r}")
    if not box.is_node_scope:
        raise BangBoxGrammarError("copy_box only supports a node-scope bang box")
    bad_wire = _non_boundary_crossing(working, box.node_scope)
    if bad_wire is not None:
        raise BangBoxGrammarError(
            f"copy_box only supports a crossing that lands on the diagram boundary; "
            f"{bad_wire!r} crosses to another live node"
        )

    id_map: dict[NodeId, NodeId] = {}
    for old_id in sorted(box.node_scope):
        node = working.nodes[old_id]
        new_id = working.add_node(
            node.generator_type,
            [p.dim for p in node.inputs],
            [p.dim for p in node.outputs],
            phase=node.phase,
        )
        id_map[old_id] = new_id

    for wire in sorted(internal_wires(working, box.node_scope), key=lambda w: w.sort_key()):
        working.add_wire(
            PortRef(id_map[wire.a.node_id], wire.a.direction, wire.a.index),
            PortRef(id_map[wire.b.node_id], wire.b.direction, wire.b.index),
        )

    scope_boundary_refs = boundary_refs_in_scope(working, box.node_scope)
    for old_ref in sorted(scope_boundary_refs, key=lambda r: r.sort_key()):
        new_ref = PortRef(id_map[old_ref.node_id], old_ref.direction, old_ref.index)
        if old_ref.direction is Direction.INPUT:
            working.set_boundary_inputs([*working.boundary_inputs, new_ref])
        else:
            working.set_boundary_outputs([*working.boundary_outputs, new_ref])

    avoid = _existing_symbol_names(working)
    stem = box.multiplicity.bare_symbol_name() if box.multiplicity.is_bare_symbol else "n"
    fresh_symbol = Mult.symbol(_fresh_name(stem, avoid))
    new_box_id = working.add_bang_box(
        fresh_symbol, node_scope=frozenset(id_map.values()), parent=None
    )
    return working, new_box_id


def _fresh_name(stem: str, avoid: set[str]) -> str:
    name = stem
    suffix = 0
    while name in avoid:
        suffix += 1
        name = f"{stem}{suffix}"
    return name


def merge(
    diagram: Diagram, box_id_1: BangBoxId, box_id_2: BangBoxId, *, assume_equal: bool = False
) -> Diagram:
    """Merge two sibling, disjoint-scope node-scope bang boxes into one.

    Requires ``box_id_1`` and ``box_id_2`` to share a ``parent`` and have disjoint
    ``node_scope``\\ s (a non-disjoint, non-nested pair is already malformed --
    :mod:`archytaszx.diagram.validate`'s job to catch, never this function's to repair).
    Their multiplicities must be the identical symbol, or ``assume_equal=True`` must be
    passed to proceed anyway, asserting the two counts equal -- never inferred silently.
    The result's node_scope is the union; any wire that crossed from one into the
    other's scope is now interior and reclassified as an ordinary internal wire
    automatically, since crossing/internal status is derived from the current scope,
    never cached (see :func:`crossing_wires`/:func:`internal_wires`).
    """
    working = diagram.copy()
    box_1 = working.bang_boxes.get(box_id_1)
    box_2 = working.bang_boxes.get(box_id_2)
    if box_1 is None or box_2 is None:
        raise BangBoxGrammarError(f"no such bang box(es): {box_id_1!r}, {box_id_2!r}")
    if not box_1.is_node_scope or not box_2.is_node_scope:
        raise BangBoxGrammarError("merge only supports node-scope bang boxes")
    if box_1.parent != box_2.parent:
        raise BangBoxGrammarError(
            f"merge requires sibling boxes (same parent); {box_id_1!r} has parent "
            f"{box_1.parent!r}, {box_id_2!r} has parent {box_2.parent!r}"
        )
    if box_1.node_scope & box_2.node_scope:
        raise BangBoxGrammarError(
            f"merge requires disjoint scopes; {box_id_1!r} and {box_id_2!r} share node(s) "
            f"{sorted(box_1.node_scope & box_2.node_scope)!r}"
        )
    if box_1.multiplicity != box_2.multiplicity and not assume_equal:
        raise BangBoxGrammarError(
            f"merge requires identical multiplicities ({box_1.multiplicity!r} != "
            f"{box_2.multiplicity!r}); pass assume_equal=True to merge anyway, asserting "
            "the two counts equal"
        )
    working.remove_bang_box(box_id_1)
    working.remove_bang_box(box_id_2)
    working.add_bang_box(
        box_1.multiplicity,
        node_scope=box_1.node_scope | box_2.node_scope,
        parent=box_1.parent,
    )
    return working
