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

Exactly one is non-empty per box (:meth:`BangBox.__post_init__`). A scoped port must be a
diagram boundary slot. A node-scope box may also be wired to a node outside it: each such
outside port fans out to one leg per copy, so a hub spider wired to a boxed satellite
denotes the hub wired to k satellites.

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

from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import PhaseSchema
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
    if expr.is_Add:
        for arg in expr.args:
            if not arg.is_Integer:
                _check_mult_domain(arg)
        return
    if expr.is_Mul:
        for arg in expr.args:
            _check_mult_domain(arg)
        return
    if expr.is_Pow and expr.base.is_Integer and int(expr.base) >= 1:
        _check_mult_domain(expr.exp)
        return
    raise BangBoxGrammarError(
        f"expression {expr} is outside the multiplicity grammar "
        "(only concrete non-negative integers, symbols, sums with integer offsets, products "
        "and powers of a positive integer are allowed)"
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
    sheets: bool = False
    """A node-scope box of a doubled diagram: its boundary slots in the first half of each
    boundary list are the plain sheet's, those in the second half the conjugate sheet's."""
    links: tuple[tuple[PortRef, PortRef], ...] = ()
    """Chain links ``(p, q)``: copy m's ``p`` is wired to copy m + 1's ``q``."""
    cyclic: bool = False
    """A chain whose last copy's ``p`` ports are wired to the first copy's ``q`` ports."""
    index: str | None = None
    """The copy index symbol: ``m`` in copy m (m = 0 .. n - 1)."""
    copy_scalar: Scalar | None = None
    """A scalar ``c(index)`` multiplied in once per copy."""
    pair_of: BangBoxId | None = None
    """For a pair box: the indexed box whose ordered copy pairs ``(j, k)``, ``j < k``, it is
    instantiated over."""
    pair_index: str | None = None
    """For a pair box: the symbol read as ``k``; the parent's index reads ``j``."""
    left_ports: frozenset[PortRef] = frozenset()
    """For a pair box: ports of the parent's scope resolved in copy ``j``."""
    right_ports: frozenset[PortRef] = frozenset()
    """For a pair box: ports of the parent's scope resolved in copy ``k``."""
    tree: tuple[PortRef, ...] = ()
    """For a tree box: the child ports ``p_1 .. p_b`` of one copy (branching ``b``)."""
    tree_parent: PortRef | None = None
    """For a tree box: the copy's parent port ``q``."""
    level: str | None = None
    """For a tree box: the depth index ``L`` (1 .. k)."""
    position: str | None = None
    """For a tree box: the position index ``P`` (0 .. b^(L-1) - 1)."""

    def __post_init__(self) -> None:
        """Validate field types, that exactly one of the two scopes is non-empty, and the
        chain, index and copy-scalar fields."""
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
        if not isinstance(self.links, tuple):
            raise BangBoxGrammarError("BangBox links must be a tuple of PortRef pairs")
        seen: set[PortRef] = set()
        for link in self.links:
            if (
                not isinstance(link, tuple)
                or len(link) != 2
                or not all(isinstance(r, PortRef) for r in link)
            ):
                raise BangBoxGrammarError(f"BangBox link must be a PortRef pair, got {link!r}")
            for ref in link:
                if ref in seen:
                    raise BangBoxGrammarError(f"BangBox port {ref!r} is in more than one link")
                seen.add(ref)
        if self.cyclic and not self.links:
            raise BangBoxGrammarError("a cyclic BangBox needs at least one link")
        if (self.links or self.index is not None or self.copy_scalar is not None) and not (
            self.node_scope
        ):
            raise BangBoxGrammarError("links, index and copy scalar need a node-scope BangBox")
        if self.index is not None:
            _check_symbol_name(self.index)
        if self.copy_scalar is not None and not isinstance(self.copy_scalar, Scalar):
            raise BangBoxGrammarError(
                f"BangBox copy_scalar must be a Scalar, got {self.copy_scalar!r}"
            )
        if self.pair_of is not None:
            if not self.node_scope or self.links or self.pair_index is None:
                raise BangBoxGrammarError(
                    "a pair box is a node-scope box without links and with a pair index"
                )
            _check_symbol_name(self.pair_index)
        elif self.pair_index is not None or self.left_ports or self.right_ports:
            raise BangBoxGrammarError("pair index, left and right ports need a pair box")
        if self.tree or self.tree_parent is not None:
            if not self.node_scope or self.links or self.pair_of is not None:
                raise BangBoxGrammarError("a tree box is a node-scope box without links or pairs")
            if self.tree_parent is None or not self.tree:
                raise BangBoxGrammarError("a tree box needs a parent port and child ports")
            ports = (self.tree_parent, *self.tree)
            if not all(isinstance(r, PortRef) for r in ports) or len(set(ports)) != len(ports):
                raise BangBoxGrammarError("tree box ports must be distinct PortRefs")
        elif self.level is not None or self.position is not None:
            raise BangBoxGrammarError("level and position indices need a tree box")
        for name in (self.level, self.position):
            if name is not None:
                _check_symbol_name(name)

    @property
    def is_pair(self) -> bool:
        """True iff this box is a pair box."""
        return self.pair_of is not None

    @property
    def is_tree(self) -> bool:
        """True iff this box is a tree box."""
        return bool(self.tree)

    @property
    def is_extended(self) -> bool:
        """True iff this box carries links, an index, a copy scalar or is a pair or tree box."""
        return (
            bool(self.links)
            or self.index is not None
            or self.copy_scalar is not None
            or self.pair_of is not None
            or bool(self.tree)
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


def outside_crossings(
    diagram: Diagram, node_scope: frozenset[NodeId]
) -> tuple[tuple[PortRef, PortRef], ...]:
    """Every crossing wire as an ``(inside, outside)`` port pair, sorted by the outside port."""
    pairs = (
        (w.a, w.b) if w.a.node_id in node_scope else (w.b, w.a)
        for w in crossing_wires(diagram, node_scope)
    )
    return tuple(sorted(pairs, key=lambda pair: pair[1].sort_key()))


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


def _port_multiplicity(diagram: Diagram, ref: PortRef) -> sp.Expr | None:
    """The product of the multiplicities of every box covering ``ref``, or None when those
    boxes are not one parent chain of covering boxes with no node scope under a port scope."""
    boxes = diagram.bang_boxes
    covering = {
        box_id
        for box_id, box in boxes.items()
        if ref.node_id in box.node_scope or ref in box.port_scope
    }
    if not covering:
        return sp.Integer(1)
    chains: dict[BangBoxId, list[BangBoxId]] = {}
    for box_id in covering:
        chain = [box_id]
        current = boxes[box_id].parent
        while current is not None:
            if current not in boxes or current in chain:
                return None
            chain.append(current)
            current = boxes[current].parent
        chains[box_id] = chain
    deepest = max(chains.values(), key=len)
    if set(deepest) != covering:
        return None
    seen_port_scope = False
    for box_id in reversed(deepest):
        if boxes[box_id].is_node_scope and seen_port_scope:
            return None
        seen_port_scope = seen_port_scope or not boxes[box_id].is_node_scope
    product = sp.Integer(1)
    for box_id in deepest:
        product = product * _slot_count(boxes[box_id], ref)
    return product


def _slot_count(box: BangBox, ref: PortRef) -> sp.Expr:
    """How many slots ``box`` makes of boundary slot ``ref``: one for a chain link port and a
    tree's parent port, b^k / b for a tree's child port, the copy count otherwise."""
    count = box.multiplicity.to_sympy()
    if any(ref in link for link in box.links) or ref == box.tree_parent:
        return sp.Integer(1)
    if not box.tree:
        return count
    b = len(box.tree)
    if ref in box.tree:
        return sp.Integer(b) ** count / b
    return count if b == 1 else (sp.Integer(b) ** count - 1) / (b - 1)


def boundary_arity(diagram: Diagram) -> tuple[sp.Expr, sp.Expr] | None:
    """The boundary input and output counts as expanded polynomials in the free
    multiplicities, or None when a boundary port's covering boxes are not one nesting chain."""
    counts: list[sp.Expr] = []
    for refs in (diagram.boundary_inputs, diagram.boundary_outputs):
        total = sp.Integer(0)
        for ref in refs:
            mult = _port_multiplicity(diagram, ref)
            if mult is None:
                return None
            total = total + mult
        counts.append(sp.expand(total))
    return counts[0], counts[1]


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
) -> NodeId:
    """Grow (or, at ``k == 0``, remove) the single boundary leg at ``ref`` to ``k`` legs, in place.

    A phaseless spider left with no legs takes the zero phase vector at the removed leg's
    dimension.

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
    if ref.index >= len(node.legs(ref.direction)):
        raise BangBoxGrammarError(f"port-scope bang box port {ref!r} is out of range")
    if ref not in diagram.boundary_inputs and ref not in diagram.boundary_outputs:
        raise BangBoxGrammarError(
            f"port-scope instantiate only supports a scoped port that is a diagram "
            f"boundary slot; {ref!r} is wired internally"
        )
    return _regrow_port(diagram, ref, k, owner_box_id=owner_box_id)


def _regrow_port(
    diagram: Diagram, ref: PortRef, k: int, *, owner_box_id: BangBoxId | None = None
) -> NodeId:
    """Replace ``ref``'s node by one whose leg at ``ref`` is ``k`` legs; ``ref`` must be unwired."""
    node = diagram.nodes[ref.node_id]
    legs = node.legs(ref.direction)
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

    phase = node.phase
    if (
        phase is None
        and not new_input_dims
        and not new_output_dims
        and node.generator_type.phase_schema is PhaseSchema.TIED_TO_LEG_DIM
    ):
        phase = PhaseVector(target_dim)
    new_node_id = diagram.add_node(
        node.generator_type, new_input_dims, new_output_dims, phase=phase
    )

    def remap(old_ref: PortRef) -> list[PortRef]:
        return _remap_grown_port(old_ref, ref.node_id, new_node_id, ref.direction, ref.index, k)

    for wire in old_wires:
        # Both ends are remapped, so a wire looping back onto this node survives.
        (end_a,) = remap(wire.a)
        (end_b,) = remap(wire.b)
        diagram.add_wire(end_a, end_b)

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
    for other_id, other in sorted(diagram.bang_boxes.items()):
        if other_id == owner_box_id or other_id not in diagram.bang_boxes:
            continue
        if ref.node_id in other.node_scope:
            if any(r.node_id == ref.node_id for link in other.links for r in link):
                moved_links = []
                for link in other.links:
                    ends = []
                    for r in link:
                        image = remap(r) if r.node_id == ref.node_id else [r]
                        if len(image) != 1:
                            raise BangBoxGrammarError(
                                f"chain box {other_id!r}: link port {r!r} grew to {len(image)} legs"
                            )
                        ends.append(image[0])
                    moved_links.append((ends[0], ends[1]))
                diagram.set_bang_box_extension(
                    other_id,
                    node_scope=frozenset(
                        new_node_id if n == ref.node_id else n for n in other.node_scope
                    ),
                    links=tuple(moved_links),
                )
            else:
                diagram.set_bang_box_node_scope(
                    other_id,
                    frozenset(new_node_id if n == ref.node_id else n for n in other.node_scope),
                )
        if any(p.node_id == ref.node_id for p in other.port_scope):
            new_port_scope = frozenset(
                new_ref
                for p in other.port_scope
                for new_ref in (remap(p) if p.node_id == ref.node_id else [p])
            )
            if new_port_scope:
                diagram.set_bang_box_port_scope(other_id, new_port_scope)
            else:
                # A nested port-scope box whose every leg was removed at k == 0.
                _drop_box(diagram, other)

    diagram.remove_node(ref.node_id)
    diagram.set_boundary_inputs(new_boundary_inputs)
    diagram.set_boundary_outputs(new_boundary_outputs)
    return new_node_id


def _instantiate_port_scope(diagram: Diagram, box: BangBox, k: int) -> None:
    if k == 0:
        _gather_ancestors(diagram, box)
    pending = sorted(box.port_scope, key=lambda r: r.sort_key())
    while pending:
        ref = pending.pop(0)
        new_node_id = _grow_port(diagram, ref, k, owner_box_id=box.id)
        # Later refs on the same node follow it to its replacement id and shifted index.
        pending = [
            moved
            for old in pending
            for moved in _remap_grown_port(
                old, ref.node_id, new_node_id, ref.direction, ref.index, k
            )
        ]
    # Surviving nested boxes move up to this box's parent, under fresh ids.
    identity = {node_id: node_id for node_id in diagram.nodes}
    _copy_subtree(diagram, box.id, identity, box.parent)
    _remove_subtree(diagram, box.id)
    diagram.remove_bang_box(box.id)


def _drop_box(diagram: Diagram, box: BangBox) -> None:
    """Remove ``box`` and every box below it, purging their dead symbols."""
    doomed = (*_descendants_of(diagram, box.id), box)
    _remove_subtree(diagram, box.id)
    diagram.remove_bang_box(box.id)
    for name in sorted({name for gone in doomed for name in gone.multiplicity.free_symbols}):
        _purge_symbol_if_dead(diagram, name)


# -- node-scope instantiation (the repeated-subgraph mechanism) ------------------------


def _children_of(diagram: Diagram, box_id: BangBoxId) -> tuple[BangBox, ...]:
    return tuple(box for _, box in sorted(diagram.bang_boxes.items()) if box.parent == box_id)


def _descendants_of(diagram: Diagram, box_id: BangBoxId) -> tuple[BangBox, ...]:
    """Every box below ``box_id`` in the parent forest, parents before children."""
    found: list[BangBox] = []
    frontier = [box_id]
    while frontier:
        children = _children_of(diagram, frontier.pop(0))
        found.extend(children)
        frontier.extend(child.id for child in children)
    return tuple(found)


def _copy_subtree(
    diagram: Diagram, box_id: BangBoxId, id_map: Mapping[NodeId, NodeId], parent: BangBoxId | None
) -> None:
    """Add a remapped copy of every child of ``box_id`` (recursively) under ``parent``."""
    for child in _children_of(diagram, box_id):
        if child.is_node_scope:
            new_id = diagram.add_bang_box(
                child.multiplicity,
                node_scope=frozenset(id_map[n] for n in child.node_scope),
                parent=parent,
                sheets=child.sheets,
                links=tuple((_moved(a, id_map), _moved(b, id_map)) for a, b in child.links),
                cyclic=child.cyclic,
                index=child.index,
                copy_scalar=child.copy_scalar,
            )
        else:
            new_id = diagram.add_bang_box(
                child.multiplicity,
                port_scope=frozenset(
                    PortRef(id_map[r.node_id], r.direction, r.index) for r in child.port_scope
                ),
                parent=parent,
            )
        _copy_subtree(diagram, child.id, id_map, new_id)


def index_symbols(diagram: Diagram) -> frozenset[str]:
    """The copy and pair index symbols of ``diagram``'s bang boxes."""
    names: set[str] = set()
    for box in diagram.bang_boxes.values():
        names |= {x for x in (box.index, box.pair_index, box.level, box.position) if x is not None}
    return frozenset(names)


def add_box_image(
    target: Diagram,
    box: BangBox,
    id_map: Mapping[NodeId, NodeId],
    parent: BangBoxId | None,
    pair_of: BangBoxId | None = None,
) -> BangBoxId:
    """Add ``box`` to ``target`` over its nodes' images under ``id_map``, every field kept."""
    return target.add_bang_box(
        box.multiplicity,
        node_scope=frozenset(id_map[n] for n in box.node_scope),
        port_scope=frozenset(_moved(r, id_map) for r in box.port_scope),
        parent=parent,
        sheets=box.sheets,
        links=tuple((_moved(a, id_map), _moved(b, id_map)) for a, b in box.links),
        cyclic=box.cyclic,
        index=box.index,
        copy_scalar=box.copy_scalar,
        pair_of=pair_of,
        pair_index=box.pair_index,
        left_ports=frozenset(_moved(r, id_map) for r in box.left_ports),
        right_ports=frozenset(_moved(r, id_map) for r in box.right_ports),
        tree=tuple(_moved(r, id_map) for r in box.tree),
        tree_parent=None if box.tree_parent is None else _moved(box.tree_parent, id_map),
        level=box.level,
        position=box.position,
    )


def _moved(ref: PortRef, id_map: Mapping[NodeId, NodeId]) -> PortRef:
    """``ref`` on its node's image under ``id_map``."""
    return PortRef(id_map[ref.node_id], ref.direction, ref.index)


def _remove_subtree(diagram: Diagram, box_id: BangBoxId) -> None:
    """Remove every box below ``box_id``, leaving ``box_id`` itself."""
    for box in reversed(_descendants_of(diagram, box_id)):
        diagram.remove_bang_box(box.id)


def _fan_out(
    diagram: Diagram,
    crossings: Iterable[tuple[PortRef, PortRef]],
    id_maps: list[dict[NodeId, NodeId]],
) -> None:
    """Rewire each ``(inside, outside)`` crossing as one outside leg per copy in ``id_maps``."""
    k = len(id_maps)
    pending = list(crossings)
    while pending:
        inside, outside = pending.pop(0)
        diagram.remove_wire(inside, outside)
        new_node_id = _regrow_port(diagram, outside, k)
        for j, id_map in enumerate(id_maps):
            diagram.add_wire(
                PortRef(id_map[inside.node_id], inside.direction, inside.index),
                PortRef(new_node_id, outside.direction, outside.index + j),
            )
        pending = [
            (other_inside, moved)
            for other_inside, other_outside in pending
            for moved in _remap_grown_port(
                other_outside, outside.node_id, new_node_id, outside.direction, outside.index, k
            )
        ]


def _rescope_others(diagram: Diagram, removed: frozenset[NodeId], added: frozenset[NodeId]) -> None:
    """Replace ``removed`` by ``added`` in every live node scope; drop emptied boxes."""
    for other_id, other in sorted(diagram.bang_boxes.items()):
        if other_id not in diagram.bang_boxes or not (other.node_scope & removed):
            continue
        if any(r.node_id in removed for link in other.links for r in link):
            raise BangBoxGrammarError(
                f"chain box {other_id!r}: a link port lies in a replicated child box"
            )
        new_scope = (other.node_scope - removed) | added
        if new_scope:
            diagram.set_bang_box_node_scope(other_id, new_scope)
        else:
            _remove_subtree(diagram, other_id)
            diagram.remove_bang_box(other_id)


def _splice_boundary_block(
    refs: list[PortRef],
    all_old_refs: list[PortRef],
    id_maps: list[dict[NodeId, NodeId]],
    sheets: bool = False,
) -> list[PortRef]:
    """With ``sheets``, one block per half of ``refs`` (plain, then conjugate), each at its
    half's first old ref; otherwise :func:`_splice_one_block`."""
    if not sheets:
        return _splice_one_block(refs, all_old_refs, id_maps)
    half = len(refs) // 2
    old = set(all_old_refs)
    first = [r for r in refs[:half] if r in old]
    second = [r for r in refs[half:] if r in old]
    out = _splice_one_block(refs, first, id_maps)
    return _splice_one_block(out, second, id_maps)


def _splice_one_block(
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


def _gather_ancestors(diagram: Diagram, box: BangBox) -> None:
    """Splice the boundary refs of each node-scope ancestor of ``box`` into one block at
    its first ref, innermost ancestor first, as an expansion at one copy does."""
    ancestor = box.parent
    while ancestor is not None:
        outer = diagram.bang_boxes[ancestor]
        if outer.is_node_scope:
            identity = {node_id: node_id for node_id in outer.node_scope}
            crossings = sorted(
                boundary_refs_in_scope(diagram, outer.node_scope), key=lambda r: r.sort_key()
            )
            diagram.set_boundary_inputs(
                _splice_boundary_block(
                    list(diagram.boundary_inputs), crossings, [identity], outer.sheets
                )
            )
            diagram.set_boundary_outputs(
                _splice_boundary_block(
                    list(diagram.boundary_outputs), crossings, [identity], outer.sheets
                )
            )
        ancestor = outer.parent


def _instantiate_node_scope(diagram: Diagram, box: BangBox, k: int) -> list[dict[NodeId, NodeId]]:
    """Replace the box by ``k`` copies of its scope; the node map of each copy."""
    scope = box.node_scope
    children = _children_of(diagram, box.id)

    if k == 0:
        _gather_ancestors(diagram, box)
        nested_symbols = sorted(
            {
                name
                for child in _descendants_of(diagram, box.id)
                for name in child.multiplicity.free_symbols
            }
        )
        for child in children:
            if child.id not in diagram.bang_boxes:
                continue
            _kill_one(diagram, diagram.bang_boxes[child.id])
        for name in nested_symbols:
            _purge_symbol_if_dead(diagram, name)
        if box.id not in diagram.bang_boxes:
            # Killing the children emptied this box, and _rescope_others already dropped it.
            return []
        # Re-read: killing a port-scope child regrows (replaces) the node it names, and
        # that node may be exactly one of this box's own scope nodes -- box.node_scope,
        # captured above as `scope`, would otherwise still name the now-removed old id.
        # _grow_port's fixup (triggered by _kill_one -> _instantiate_port_scope ->
        # _grow_port) already repointed diagram.bang_boxes[box.id] itself; only this
        # function's own local snapshot goes stale.
        scope = diagram.bang_boxes[box.id].node_scope
        _fan_out(diagram, outside_crossings(diagram, scope), [])
        for node_id in scope:
            diagram.remove_node(node_id)
        diagram.remove_bang_box(box.id)
        _rescope_others(diagram, scope, frozenset())
        return []

    if k == 1:
        # The one copy keeps the nodes it already has, but the boundary still goes through
        # the same block splice k >= 2 performs: without it a scope whose crossings are not
        # already contiguous would order its boundary differently at 1 than at 2.
        identity = {node_id: node_id for node_id in scope}
        sorted_crossings = sorted(
            boundary_refs_in_scope(diagram, scope), key=lambda r: r.sort_key()
        )
        diagram.set_boundary_inputs(
            _splice_boundary_block(
                list(diagram.boundary_inputs), sorted_crossings, [identity], box.sheets
            )
        )
        diagram.set_boundary_outputs(
            _splice_boundary_block(
                list(diagram.boundary_outputs), sorted_crossings, [identity], box.sheets
            )
        )
        _copy_subtree(diagram, box.id, identity, box.parent)
        _remove_subtree(diagram, box.id)
        diagram.remove_bang_box(box.id)
        return [identity]

    internal = internal_wires(diagram, scope)

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
    _fan_out(diagram, outside_crossings(diagram, scope), id_maps)

    sorted_crossings = sorted(boundary_refs_in_scope(diagram, scope), key=lambda r: r.sort_key())
    new_boundary_inputs = _splice_boundary_block(
        list(diagram.boundary_inputs), sorted_crossings, id_maps, box.sheets
    )
    new_boundary_outputs = _splice_boundary_block(
        list(diagram.boundary_outputs), sorted_crossings, id_maps, box.sheets
    )

    for id_map in id_maps:
        _copy_subtree(diagram, box.id, id_map, box.parent)
    _remove_subtree(diagram, box.id)
    diagram.remove_bang_box(box.id)
    copies = frozenset(new_id for id_map in id_maps for new_id in id_map.values())
    _rescope_others(diagram, scope, copies)

    for node_id in scope:
        diagram.remove_node(node_id)
    diagram.set_boundary_inputs(new_boundary_inputs)
    diagram.set_boundary_outputs(new_boundary_outputs)
    return id_maps


def _tree_leaf_block(diagram: Diagram, box: BangBox) -> tuple[bool, list[PortRef], list[PortRef]]:
    """Whether a tree box's child ports are boundary outputs, and the boundary list before and
    after their contiguous run; raise when they are not one run in ``tree`` order."""
    for outputs, refs in (
        (True, list(diagram.boundary_outputs)),
        (False, list(diagram.boundary_inputs)),
    ):
        if box.tree[0] not in refs:
            continue
        first = refs.index(box.tree[0])
        if refs[first : first + len(box.tree)] != list(box.tree):
            break
        return outputs, refs[:first], refs[first + len(box.tree) :]
    raise BangBoxGrammarError(
        f"tree box {box.id!r}: its child ports are not one run of boundary slots in order"
    )


def _tree_copy(
    diagram: Diagram,
    box: BangBox,
    id_map: Mapping[NodeId, NodeId],
    values: Mapping[str, int],
    new_boxes: Iterable[BangBoxId],
) -> None:
    """Copy ``id_map``'s nodes and child boxes with each index of ``values`` set, and the copy
    scalar at those values multiplied into the diagram scalar."""
    nodes = frozenset(id_map.values())
    for node_id in sorted(nodes):
        phase = diagram.nodes[node_id].phase
        hit = {k: v for k, v in values.items() if phase is not None and k in phase.free_symbols}
        if phase is not None and hit:
            diagram.set_phase(node_id, phase.substitute(hit))  # type: ignore[arg-type]
    for child_id in new_boxes:
        child = diagram.bang_boxes.get(child_id)
        if child is None:
            continue
        covered = child.node_scope or frozenset(r.node_id for r in child.port_scope)
        if not covered <= nodes:
            continue
        mult = child.multiplicity.to_sympy()
        for name, value in values.items():
            if name in child.multiplicity.free_symbols:
                mult = mult.subs(_mult_symbol(name), value)
        if mult != child.multiplicity.to_sympy():
            diagram.set_bang_box_multiplicity(child_id, Mult._from_expr(sp.expand(mult)))
    if box.copy_scalar is not None:
        factor = box.copy_scalar
        for name, value in values.items():
            factor = _scalar_at(factor, name, value)
        diagram.multiply_scalar(factor)


def _instantiate_tree(diagram: Diagram, box: BangBox, k: int) -> None:
    """Instantiate a tree box at depth ``k``: copies c(L, P) for L = 1 .. k, P < b^(L-1), the
    head into c(1, 0)'s parent port, c(L, P)'s i-th child port into c(L + 1, b P + i)'s parent
    port, the leaves' child ports as the b^k slots of the stored child ports' run, leaf-major;
    at ``k == 0`` the head alone in that run."""
    assert box.tree_parent is not None
    b = len(box.tree)
    temps: list[NodeId] = []
    head = _detach(diagram, box.tree_parent, temps)
    outputs, before, after = _tree_leaf_block(diagram, box)
    setter = diagram.set_boundary_outputs if outputs else diagram.set_boundary_inputs
    setter([*before, *after])
    coords = [(level, pos) for level in range(1, k + 1) for pos in range(b ** (level - 1))]
    where = {coord: i for i, coord in enumerate(coords)}
    diagram.set_bang_box_extension(
        box.id, tree=(), tree_parent=None, level=None, position=None, index=None, copy_scalar=None
    )
    plain = diagram.bang_boxes[box.id]
    known = set(diagram.bang_boxes)
    id_maps = _instantiate_node_scope(diagram, plain, len(coords))
    new_boxes = sorted(set(diagram.bang_boxes) - known)
    if k == 0:
        leaves = [head]
    else:
        diagram.add_wire(head, _moved(box.tree_parent, id_maps[0]))
        for (level, pos), i in where.items():
            if level == k:
                continue
            for j, p_ref in enumerate(box.tree):
                child = id_maps[where[(level + 1, b * pos + j)]]
                diagram.add_wire(_moved(p_ref, id_maps[i]), _moved(box.tree_parent, child))
        leaves = [
            _moved(p_ref, id_maps[where[(k, pos)]])
            for pos in range(b ** (k - 1))
            for p_ref in box.tree
        ]
    setter([*before, *leaves, *after])
    for (level, pos), i in where.items():
        values = {
            name: value
            for name, value in ((box.level, level), (box.position, pos), (box.index, i))
            if name is not None
        }
        _tree_copy(diagram, box, id_maps[i], values, new_boxes)
    _drop_identities(diagram, temps)


def _kill_one(diagram: Diagram, box: BangBox) -> None:
    if box.is_tree:
        _instantiate_tree(diagram, box, 0)
    elif box.is_extended:
        _instantiate_extended(diagram, box, 0)
    elif box.is_node_scope:
        _instantiate_node_scope(diagram, box, 0)
    else:
        _instantiate_port_scope(diagram, box, 0)


def _instantiate_one(diagram: Diagram, box_id: BangBoxId, k: int) -> None:
    box = diagram.bang_boxes[box_id]
    if box.is_pair:
        raise BangBoxGrammarError(f"pair box {box_id!r} is instantiated with its parent only")
    if box.is_tree:
        _instantiate_tree(diagram, box, k)
    elif box.is_extended:
        _instantiate_extended(diagram, box, k)
    elif box.is_node_scope:
        _instantiate_node_scope(diagram, box, k)
    else:
        _instantiate_port_scope(diagram, box, k)


# -- chain, index and copy-scalar boxes ---------------------------------------------------


def _identity_node(diagram: Diagram, ref: PortRef) -> NodeId:
    """A phaseless Z spider 1 -> 1 at ``ref``'s leg dimension."""
    from archytaszx.diagram.generators import Z_SPIDER

    dim = diagram.nodes[ref.node_id].legs(ref.direction)[ref.index].dim
    return diagram.add_node(Z_SPIDER, [dim], [dim], phase=PhaseVector(dim))


def _partner(diagram: Diagram, ref: PortRef) -> PortRef | None:
    """The port wired to ``ref``, or None."""
    for wire in diagram.wires:
        if wire.a == ref:
            return wire.b
        if wire.b == ref:
            return wire.a
    return None


def _detach(diagram: Diagram, ref: PortRef, temps: list[NodeId]) -> PortRef:
    """Put an identity node between ``ref`` and its wire or boundary slot, unwire ``ref`` from
    it, and return the identity's port that faced ``ref``."""
    node = _identity_node(diagram, ref)
    temps.append(node)
    facing_out = ref.direction is Direction.INPUT
    near = PortRef(node, Direction.OUTPUT if facing_out else Direction.INPUT, 0)
    far = PortRef(node, Direction.INPUT if facing_out else Direction.OUTPUT, 0)
    partner = _partner(diagram, ref)
    if partner is not None:
        diagram.remove_wire(ref, partner)
        diagram.add_wire(far, partner)
    elif ref in diagram.boundary_inputs:
        diagram.set_boundary_inputs([far if r == ref else r for r in diagram.boundary_inputs])
    elif ref in diagram.boundary_outputs:
        diagram.set_boundary_outputs([far if r == ref else r for r in diagram.boundary_outputs])
    else:
        raise BangBoxGrammarError(f"chain link port {ref!r} is neither wired nor a boundary slot")
    return near


def _drop_identities(diagram: Diagram, temps: list[NodeId]) -> None:
    """Remove each identity node of ``temps`` that is not wired boundary to boundary, joining
    what its two legs met."""
    for node_id in temps:
        legs = [PortRef(node_id, Direction.INPUT, 0), PortRef(node_id, Direction.OUTPUT, 0)]
        ends = [_partner(diagram, leg) for leg in legs]
        if ends[0] is None and ends[1] is None:
            continue
        for leg, end in zip(legs, ends, strict=True):
            if end is not None:
                diagram.remove_wire(leg, end)
        if ends[0] is not None and ends[1] is not None:
            diagram.add_wire(ends[0], ends[1])
        else:
            slot, other = (legs[0], ends[1]) if ends[0] is None else (legs[1], ends[0])
            assert other is not None
            inputs = [other if r == slot else r for r in diagram.boundary_inputs]
            diagram.set_boundary_inputs(inputs)
            outputs = [other if r == slot else r for r in diagram.boundary_outputs]
            diagram.set_boundary_outputs(outputs)
        diagram.remove_node(node_id)


def _detach_links(
    diagram: Diagram, box: BangBox, temps: list[NodeId]
) -> list[tuple[PortRef | None, PortRef | None]]:
    """Unwire every link port of ``box``: a cyclic link's own wire is removed, an open link's
    ``q`` and ``p`` each get an identity in front; per link, the identity ports facing
    ``q`` and ``p`` (None for a cyclic link)."""
    ends: list[tuple[PortRef | None, PortRef | None]] = []
    for p_ref, q_ref in box.links:
        if box.cyclic:
            if _partner(diagram, p_ref) != q_ref:
                raise BangBoxGrammarError(
                    f"cyclic chain box {box.id!r}: link port {p_ref!r} is not wired to {q_ref!r}"
                )
            diagram.remove_wire(p_ref, q_ref)
            ends.append((None, None))
        else:
            ends.append((_detach(diagram, q_ref, temps), _detach(diagram, p_ref, temps)))
    return ends


def _index_copy(
    diagram: Diagram,
    box: BangBox,
    id_map: Mapping[NodeId, NodeId],
    value: int | sp.Expr,
    new_boxes: Iterable[BangBoxId],
) -> None:
    """Copy ``id_map``'s nodes and the child boxes over them with ``box.index`` := ``value``,
    and the copy scalar at that value multiplied into the diagram scalar."""
    nodes = frozenset(id_map.values())
    if box.index is not None:
        sub = value if isinstance(value, int) else Phase(value)
        for node_id in sorted(nodes):
            phase = diagram.nodes[node_id].phase
            if phase is not None and box.index in phase.free_symbols:
                diagram.set_phase(node_id, phase.substitute({box.index: sub}))
        for child_id in new_boxes:
            child = diagram.bang_boxes.get(child_id)
            if child is None:
                continue
            covered = child.node_scope or frozenset(r.node_id for r in child.port_scope)
            if not covered <= nodes:
                continue
            if box.index in child.multiplicity.free_symbols:
                mult = child.multiplicity.to_sympy().subs(_mult_symbol(box.index), value)
                diagram.set_bang_box_multiplicity(child_id, Mult._from_expr(sp.expand(mult)))
            if child.copy_scalar is not None and box.index in child.copy_scalar.free_symbols:
                diagram.set_bang_box_extension(
                    child_id, copy_scalar=_scalar_at(child.copy_scalar, box.index, value)
                )
    if box.copy_scalar is not None:
        factor = box.copy_scalar
        if box.index is not None:
            factor = _scalar_at(factor, box.index, value)
        diagram.multiply_scalar(factor)


def _scalar_at(scalar: Scalar, name: str, value: int | sp.Expr) -> Scalar:
    """``scalar`` with the symbol ``name`` replaced by ``value``."""
    symbols = [x for x in scalar.to_sympy().free_symbols if str(x) == name]
    if not symbols:
        return scalar
    return Scalar(scalar.to_sympy().xreplace({symbols[0]: sp.sympify(value)}))


@dataclass(frozen=True, slots=True)
class _PairTemplate:
    """A detached pair box: its nodes (copied out), internal wires, and its wires to the
    parent's left and right ports as ``(template port, parent port, side)``."""

    box: BangBox
    nodes: tuple[tuple[NodeId, object, tuple[object, ...], tuple[object, ...], object], ...]
    internal: tuple[tuple[PortRef, PortRef], ...]
    attached: tuple[tuple[PortRef, PortRef, str], ...]


def _pairs_of(diagram: Diagram, box_id: BangBoxId) -> list[BangBox]:
    """The pair boxes over ``box_id``, by id."""
    return [b for _, b in sorted(diagram.bang_boxes.items()) if b.pair_of == box_id]


def _detach_pair(diagram: Diagram, pair: BangBox) -> _PairTemplate:
    """Remove ``pair``'s nodes and box, recording them; each left or right port it met is left
    unwired."""
    scope = pair.node_scope
    nodes = tuple(
        (
            nid,
            diagram.nodes[nid].generator_type,
            tuple(p.dim for p in diagram.nodes[nid].inputs),
            tuple(p.dim for p in diagram.nodes[nid].outputs),
            diagram.nodes[nid].phase,
        )
        for nid in sorted(scope)
    )
    internal = tuple((w.a, w.b) for w in sorted(internal_wires(diagram, scope), key=Wire.sort_key))
    attached: list[tuple[PortRef, PortRef, str]] = []
    for inside, outside in outside_crossings(diagram, scope):
        if outside in pair.left_ports:
            side = "left"
        elif outside in pair.right_ports:
            side = "right"
        else:
            raise BangBoxGrammarError(
                f"pair box {pair.id!r} is wired to {outside!r}, not a left or right port"
            )
        attached.append((inside, outside, side))
    if any(r.node_id in scope for r in (*diagram.boundary_inputs, *diagram.boundary_outputs)):
        raise BangBoxGrammarError(f"pair box {pair.id!r} holds a boundary slot")
    for nid in sorted(scope):
        diagram.remove_node(nid)
    diagram.remove_bang_box(pair.id)
    return _PairTemplate(pair, nodes, internal, tuple(attached))


def _place_pairs(
    diagram: Diagram, parent: BangBox, template: _PairTemplate, id_maps: list[dict[NodeId, NodeId]]
) -> None:
    """One copy of ``template`` per pair of copies ``j < k`` of ``parent`` (``index`` := j,
    ``pair_index`` := k), each left port of copy j and right port of copy k grown to one leg
    per pair it meets."""
    k_total = len(id_maps)
    pairs = [(j, k) for j in range(k_total) for k in range(j + 1, k_total)]
    legs: dict[tuple[PortRef, int], list[PortRef]] = {}
    ports = {a[1] for a in template.attached}
    for parent_port in sorted(ports, key=lambda r: (r.node_id, r.direction.value, -r.index)):
        side = next(a[2] for a in template.attached if a[1] == parent_port)
        for m, id_map in enumerate(id_maps):
            count = sum(1 for j, k in pairs if (j if side == "left" else k) == m)
            ref = _moved(parent_port, id_map)
            new_node = _regrow_port(diagram, ref, count)
            id_map[parent_port.node_id] = new_node
            for key, grown in list(legs.items()):
                if key[1] == m and grown and grown[0].node_id == ref.node_id:
                    legs[key] = [
                        moved
                        for old in grown
                        for moved in _remap_grown_port(
                            old, ref.node_id, new_node, ref.direction, ref.index, count
                        )
                    ]
            legs[(parent_port, m)] = [
                PortRef(new_node, ref.direction, ref.index + i) for i in range(count)
            ]
    used: dict[tuple[PortRef, int], int] = {}
    box = template.box
    for j, k in pairs:
        node_map: dict[NodeId, NodeId] = {}
        for nid, generator, ins, outs, phase in template.nodes:
            node_map[nid] = diagram.add_node(generator, list(ins), list(outs), phase=phase)  # type: ignore[arg-type]
        for a, b in template.internal:
            diagram.add_wire(_moved(a, node_map), _moved(b, node_map))
        for inside, parent_port, side in template.attached:
            key = (parent_port, j if side == "left" else k)
            slot = used.get(key, 0)
            used[key] = slot + 1
            diagram.add_wire(_moved(inside, node_map), legs[key][slot])
        for new_id in node_map.values():
            phase = diagram.nodes[new_id].phase
            if phase is None:
                continue
            subs: dict[str, int] = {}
            if parent.index is not None and parent.index in phase.free_symbols:
                subs[parent.index] = j
            if box.pair_index is not None and box.pair_index in phase.free_symbols:
                subs[box.pair_index] = k
            if subs:
                diagram.set_phase(new_id, phase.substitute(subs))  # type: ignore[arg-type]
        if box.copy_scalar is not None:
            factor = box.copy_scalar
            if parent.index is not None:
                factor = _scalar_at(factor, parent.index, j)
            if box.pair_index is not None:
                factor = _scalar_at(factor, box.pair_index, k)
            diagram.multiply_scalar(factor)


def _plain(diagram: Diagram, box: BangBox) -> BangBox:
    """``box`` stripped of its links, index and copy scalar in ``diagram``."""
    diagram.set_bang_box_extension(box.id, links=(), cyclic=False, index=None, copy_scalar=None)
    return diagram.bang_boxes[box.id]


def _instantiate_extended(diagram: Diagram, box: BangBox, k: int) -> None:
    """Instantiate a chain, indexed or copy-scalar box at ``k``: plain copies, each link wired
    copy to copy (head to the first copy and last copy to tail, or around for a cyclic chain;
    head to tail at ``k == 0``), the index set to the copy number, the copy scalars
    multiplied in."""
    temps: list[NodeId] = []
    ends = _detach_links(diagram, box, temps)
    pairs = [_detach_pair(diagram, pair) for pair in _pairs_of(diagram, box.id)]
    before = set(diagram.bang_boxes)
    id_maps = _instantiate_node_scope(diagram, _plain(diagram, box), k)
    new_boxes = sorted(set(diagram.bang_boxes) - before)
    for template in pairs:
        _place_pairs(diagram, box, template, id_maps)
    for (p_ref, q_ref), (head, tail) in zip(box.links, ends, strict=True):
        if box.cyclic:
            for m in range(k):
                diagram.add_wire(_moved(p_ref, id_maps[m]), _moved(q_ref, id_maps[(m + 1) % k]))
            continue
        assert head is not None and tail is not None
        if k == 0:
            diagram.add_wire(head, tail)
            continue
        diagram.add_wire(head, _moved(q_ref, id_maps[0]))
        for m in range(k - 1):
            diagram.add_wire(_moved(p_ref, id_maps[m]), _moved(q_ref, id_maps[m + 1]))
        diagram.add_wire(_moved(p_ref, id_maps[-1]), tail)
    for m, id_map in enumerate(id_maps):
        _index_copy(diagram, box, id_map, m, new_boxes)
    _drop_identities(diagram, temps)


# -- public entry points -----------------------------------------------------------------


_MAX_INSTANTIATE_ROUNDS = 10_000


def substitute_index_phases(diagram: Diagram, name: str, value: int | Mult) -> Diagram:
    """``diagram`` with ``name`` replaced by ``value`` in every node phase vector."""
    from archytaszx.algebra.phase import Phase

    sub: int | Phase = value if isinstance(value, int) else Phase(value.to_sympy())
    working = diagram.copy()
    for node_id, node in sorted(diagram.nodes.items()):
        if node.phase is not None and name in node.phase.free_symbols:
            working.set_phase(node_id, node.phase.substitute({name: sub}))
    return working


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


def _peel_node_scope(diagram: Diagram, box: BangBox) -> dict[NodeId, NodeId]:
    """Add one concrete copy of the scope, after the box's own boundary block; its node map."""
    scope = box.node_scope
    internal = internal_wires(diagram, scope)

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
    id_maps = [identity, id_map]
    _fan_out(diagram, outside_crossings(diagram, scope), id_maps)
    sorted_crossings = sorted(boundary_refs_in_scope(diagram, scope), key=lambda r: r.sort_key())
    diagram.set_boundary_inputs(
        _splice_boundary_block(list(diagram.boundary_inputs), sorted_crossings, id_maps, box.sheets)
    )
    diagram.set_boundary_outputs(
        _splice_boundary_block(
            list(diagram.boundary_outputs), sorted_crossings, id_maps, box.sheets
        )
    )

    _copy_subtree(diagram, box.id, id_map, box.parent)
    ancestor = box.parent
    while ancestor is not None:
        outer = diagram.bang_boxes[ancestor]
        if outer.is_node_scope:
            diagram.set_bang_box_node_scope(ancestor, outer.node_scope | frozenset(id_map.values()))
        ancestor = outer.parent
    return id_map


def _peel_extended(diagram: Diagram, box: BangBox, last: sp.Expr) -> None:
    """Peel the last copy (index ``last``) off a chain, indexed or copy-scalar box: the residual
    box's ``p`` ports wired to the copy's ``q`` ports and the copy's ``p`` ports to the tail."""
    if box.cyclic or box.is_tree:
        raise BangBoxGrammarError(f"cyclic chain or tree box {box.id!r} has no peel_one")
    if box.is_pair or _pairs_of(diagram, box.id):
        raise BangBoxGrammarError(f"box {box.id!r} with pair boxes has no peel_one")
    temps: list[NodeId] = []
    ends = _detach_links(diagram, box, temps)
    before = set(diagram.bang_boxes)
    id_map = _peel_node_scope(diagram, _plain(diagram, box))
    new_boxes = sorted(set(diagram.bang_boxes) - before)
    for (p_ref, q_ref), (head, tail) in zip(box.links, ends, strict=True):
        assert head is not None and tail is not None
        diagram.add_wire(head, q_ref)
        diagram.add_wire(p_ref, _moved(q_ref, id_map))
        diagram.add_wire(_moved(p_ref, id_map), tail)
    _index_copy(diagram, box, id_map, last, new_boxes)
    diagram.set_bang_box_extension(
        box.id,
        links=box.links,
        cyclic=box.cyclic,
        index=box.index,
        copy_scalar=box.copy_scalar,
    )
    _drop_identities(diagram, temps)


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
    if box.is_extended:
        _peel_extended(working, box, predecessor.to_sympy())
    elif box.is_node_scope:
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
            if box.multiplicity.is_concrete and not box.is_pair
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
    dangling on fresh diagram boundary slots appended to the appropriate list; each wire
    to an outside node fans out to one outside leg per copy.
    """
    working = diagram.copy()
    box = working.bang_boxes.get(box_id)
    if box is None:
        raise BangBoxGrammarError(f"no such bang box: {box_id!r}")
    if not box.is_node_scope:
        raise BangBoxGrammarError("copy_box only supports a node-scope bang box")
    if box.is_extended:
        raise BangBoxGrammarError(f"copy_box does not support chain or indexed box {box_id!r}")
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
    identity = {old_id: old_id for old_id in box.node_scope}
    _fan_out(working, outside_crossings(working, box.node_scope), [identity, id_map])

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
    if box_1.is_extended or box_2.is_extended:
        raise BangBoxGrammarError(
            f"merge does not support chain or indexed boxes ({box_id_1!r}, {box_id_2!r})"
        )
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
        sheets=box_1.sheets and box_2.sheets,
    )
    return working
