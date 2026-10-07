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

"""Rewrites performed directly on a :class:`~archytaszx.diagram.scalable.ScalableDiagram`.

Six pure operations, each returning a :class:`SheetStep` whose ``after`` passes
:func:`~archytaszx.diagram.scalable.validate_scalable`; an invalid input is refused.

* :func:`split_scale` -- the divider: a COPIES scale of multiplicity ``m`` becomes two sibling
  scales of ``first`` and ``m - first``, the second holding a duplicate of the whole subtree,
  its bundle right after the first's. A part of multiplicity exactly 1 is then dissolved.
* :func:`join_scales` -- the gatherer: two corresponding sibling scales become one carrying
  the sum of their multiplicities.
* :func:`enclose` -- wraps nodes of one region in a new COPIES scale of multiplicity 1.
* :func:`dissolve` -- removes a scale of multiplicity 1, keeping its contents in place.
* :func:`kill_scale` -- removes a scale of multiplicity 0 and everything it scales.
* :func:`fuse_sheet` -- spider fusion across one wire of any size, with the leg order, phase
  sum, parallel-wire self-loops and scalar of
  :data:`~archytaszx.rewrite.rules_library.SPIDER_FUSION`; every leg and phase must carry the
  connecting pair's exact dimension.

New node and scale ids are allocated from the largest existing id plus one, ascending. A
parameter binding for a multiplicity symbol that an operation removes from every scale is
dropped, as :func:`~archytaszx.diagram.bangbox.instantiate_symbol` consumes its own.
:data:`SHEET_OPERATIONS` maps each operation name to its function, and
:func:`replay_sheet_step` re-runs a step to confirm its ``after``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.phase import PhaseVector
from archytaszx.diagram.bangbox import BangBoxError, Mult
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Direction, NodeId, PortRef, Wire
from archytaszx.diagram.scalable import (
    BoundaryItem,
    Bundle,
    ScalableDiagram,
    Scale,
    ScaledNode,
    ScaleId,
    ScaleKind,
    SheetPort,
    validate_scalable,
)
from archytaszx.rewrite.rule import RewriteDomainError, RewriteError, RewriteGrammarError
from archytaszx.rewrite.rules_library import SPIDER_FUSION


class SheetError(RewriteError):
    """Base class for all errors raised by this module."""


class SheetGrammarError(SheetError, RewriteGrammarError):
    """A request is malformed: wrong type, unknown id, wrong scale kind, or no correspondence."""


class SheetDomainError(SheetError, RewriteDomainError):
    """A value is outside an operation's domain, such as a multiplicity that is not 1."""


@dataclass(frozen=True, slots=True)
class SheetStep:
    """One applied operation: its name, normalized arguments sorted by name, and both states."""

    operation: str
    arguments: tuple[tuple[str, object], ...]
    before: ScalableDiagram
    after: ScalableDiagram


# -- shared helpers ------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_valid(s: object, operation: str) -> ScalableDiagram:
    if not isinstance(s, ScalableDiagram):
        raise SheetGrammarError(f"{operation} requires a ScalableDiagram, got {s!r}")
    issues = validate_scalable(s)
    if issues:
        raise SheetGrammarError(f"{operation}: input is not valid: " + "; ".join(issues))
    return s


def _scale_arg(s: ScalableDiagram, value: object, operation: str) -> Scale:
    if not _is_int(value):
        raise SheetGrammarError(f"{operation}: a scale id must be an int, got {value!r}")
    for scale in s.scales:
        if scale.id == value:
            return scale
    raise SheetGrammarError(f"{operation}: no such scale {value!r}")


def _mult_arg(value: object, operation: str) -> Mult:
    if isinstance(value, Mult):
        return value
    if not _is_int(value):
        raise SheetGrammarError(f"{operation}: a multiplicity must be a Mult or int: {value!r}")
    try:
        return Mult(value)  # type: ignore[arg-type]
    except BangBoxError as exc:
        raise SheetDomainError(f"{operation}: {exc}") from exc


def _finish(
    operation: str, arguments: Mapping[str, object], before: ScalableDiagram, after: ScalableDiagram
) -> SheetStep:
    """The step, with bindings of multiplicity symbols ``after`` no longer carries dropped."""
    dead = before.free_mult_symbols() - after.free_mult_symbols()
    if dead & set(after.parameters):
        kept = {k: v for k, v in after.parameters.items() if k not in dead}
        after = replace(after, parameters=kept)
    issues = validate_scalable(after)
    if issues:
        raise SheetGrammarError(f"{operation}: result is not valid: " + "; ".join(issues))
    return SheetStep(operation, tuple(sorted(arguments.items())), before, after)


def _next_node_id(s: ScalableDiagram) -> int:
    return max((int(n.id) for n in s.nodes), default=-1) + 1


def _next_scale_id(s: ScalableDiagram) -> int:
    return max((int(sc.id) for sc in s.scales), default=-1) + 1


def _subtree_scales(s: ScalableDiagram, root: ScaleId) -> tuple[Scale, ...]:
    """``root`` and every scale below it, in ascending id order."""
    return tuple(sc for sc in s.scales if root in {c.id for c in s.chain(sc.id)})


def _subtree_nodes(s: ScalableDiagram, root: ScaleId) -> tuple[ScaledNode, ...]:
    """Every node whose scale chain contains ``root``, in ascending id order."""
    return tuple(n for n in s.nodes if root in s.copies_chain(n.scale))


def _transform(
    items: tuple[BoundaryItem, ...],
    ref: Callable[[PortRef], Iterable[BoundaryItem]],
    bundle: Callable[[Bundle], Iterable[BoundaryItem]],
) -> tuple[BoundaryItem, ...]:
    """``items`` rebuilt bottom-up: refs through ``ref``, rebuilt bundles through ``bundle``."""
    out: list[BoundaryItem] = []
    for item in items:
        if isinstance(item, PortRef):
            out.extend(ref(item))
        else:
            out.extend(bundle(Bundle(item.scale, _transform(item.items, ref, bundle))))
    return tuple(out)


def _keep_ref(ref: PortRef) -> tuple[BoundaryItem, ...]:
    return (ref,)


def _keep_nonempty(bundle: Bundle) -> tuple[BoundaryItem, ...]:
    return (bundle,) if bundle.items else ()


def _relabel(
    items: tuple[BoundaryItem, ...],
    node_map: Mapping[NodeId, NodeId],
    scale_map: Mapping[ScaleId, ScaleId],
) -> tuple[BoundaryItem, ...]:
    """``items`` with node ids and bundle scales replaced through the maps (identity outside)."""
    return _transform(
        items,
        lambda r: (PortRef(node_map.get(r.node_id, r.node_id), r.direction, r.index),),
        lambda b: (Bundle(scale_map.get(b.scale, b.scale), b.items),),
    )


def _find_bundle(items: tuple[BoundaryItem, ...], scale: ScaleId) -> Bundle | None:
    for item in items:
        if isinstance(item, Bundle):
            if item.scale == scale:
                return item
            found = _find_bundle(item.items, scale)
            if found is not None:
                return found
    return None


def _region_items(
    s: ScalableDiagram, items: tuple[BoundaryItem, ...], scale: ScaleId | None
) -> tuple[BoundaryItem, ...] | None:
    """The direct item list of ``scale`` on one side: the top level for None."""
    if scale is None:
        return items
    found = _find_bundle(items, scale)
    return None if found is None else found.items


def _copy_scales(
    s: ScalableDiagram, root: Scale, new_root: Scale, first_free: int
) -> tuple[dict[ScaleId, ScaleId], list[Scale]]:
    """Duplicates of every scale strictly below ``root``, hung under ``new_root``."""
    scale_map: dict[ScaleId, ScaleId] = {root.id: new_root.id}
    added: list[Scale] = [new_root]
    next_id = first_free
    for scale in _subtree_scales(s, root.id):
        if scale.id != root.id:
            scale_map[scale.id] = ScaleId(next_id)
            next_id += 1
    for scale in _subtree_scales(s, root.id):
        if scale.id != root.id:
            assert scale.parent is not None
            added.append(
                Scale(scale_map[scale.id], scale.kind, scale.multiplicity, scale_map[scale.parent])
            )
    return scale_map, added


def _map_ports(
    ports: tuple[SheetPort, ...], scale_map: Mapping[ScaleId, ScaleId]
) -> tuple[SheetPort, ...]:
    return tuple(
        SheetPort(p.dim, None if p.fan is None else scale_map.get(p.fan, p.fan)) for p in ports
    )


# -- dissolve --------------------------------------------------------------------------------


def _dissolve(s: ScalableDiagram, scale: Scale) -> ScalableDiagram:
    parent = scale.parent
    scales = tuple(
        replace(sc, parent=parent) if sc.parent == scale.id else sc
        for sc in s.scales
        if sc.id != scale.id
    )
    if scale.kind is ScaleKind.COPIES:
        nodes = tuple(replace(n, scale=parent) if n.scale == scale.id else n for n in s.nodes)

        def splice(b: Bundle) -> tuple[BoundaryItem, ...]:
            return b.items if b.scale == scale.id else (b,)

        return replace(
            s,
            nodes=nodes,
            scales=scales,
            inputs=_transform(s.inputs, _keep_ref, splice),
            outputs=_transform(s.outputs, _keep_ref, splice),
        )
    parent_scale = None if parent is None else s.scale(parent)
    new_fan = parent if parent_scale is not None and parent_scale.kind is ScaleKind.LEGS else None

    def refan(ports: tuple[SheetPort, ...]) -> tuple[SheetPort, ...]:
        return tuple(SheetPort(p.dim, new_fan) if p.fan == scale.id else p for p in ports)

    nodes = tuple(replace(n, inputs=refan(n.inputs), outputs=refan(n.outputs)) for n in s.nodes)
    return replace(s, nodes=nodes, scales=scales)


def dissolve(s: ScalableDiagram, scale: ScaleId) -> SheetStep:
    """Remove a scale of multiplicity exactly 1, keeping its contents.

    COPIES: its nodes and child scales move to its parent and each of its bundles is spliced
    in place into the enclosing item list. LEGS: its ports' fans and its children move to its
    parent, a port's fan becoming None when that parent is not a LEGS scale.
    """
    s = _require_valid(s, "dissolve")
    target = _scale_arg(s, scale, "dissolve")
    if target.multiplicity != Mult(1):
        raise SheetDomainError(
            f"dissolve: scale {target.id} has multiplicity {target.multiplicity}"
        )
    return _finish("dissolve", {"scale": target.id}, s, _dissolve(s, target))


# -- split and join --------------------------------------------------------------------------


def _mult_from_expr(expr: sp.Expr, operation: str) -> Mult:
    """The Mult of an expanded polynomial whose coefficients are non-negative integers."""
    total = Mult(0)
    for term in sp.Add.make_args(sp.expand(expr)):
        coeff, rest = term.as_coeff_Mul()
        if not (coeff.is_Integer and coeff >= 0):
            raise SheetDomainError(f"{operation}: {expr} has a negative or non-integer term")
        product = Mult(int(coeff))
        for factor in sp.Mul.make_args(rest):
            if factor == 1:
                continue
            base, exponent = factor.as_base_exp()
            if not (base.is_Symbol and exponent.is_Integer and exponent >= 1):
                raise SheetDomainError(f"{operation}: {expr} is not a polynomial multiplicity")
            for _ in range(int(exponent)):
                product = product * Mult(str(base.name))
        total = total + product
    return total


def split_scale(
    s: ScalableDiagram, scale: ScaleId, first: Mult | int, *, dissolve_unit: bool = True
) -> SheetStep:
    """Split a COPIES scale of multiplicity ``m`` into ``first`` and ``m - first`` copies.

    ``scale`` keeps ``first``; a new sibling scale of ``m - first`` holds a duplicate of every
    node, descendant scale, fan and wire of the subtree, and its bundle follows the original's
    on each side. ``m - first`` must expand to non-negative coefficients and neither part may
    be zero. With ``dissolve_unit``, a part of multiplicity exactly 1 is then dissolved, the
    new part first.
    """
    s = _require_valid(s, "split_scale")
    target = _scale_arg(s, scale, "split_scale")
    if target.kind is not ScaleKind.COPIES:
        raise SheetGrammarError(f"split_scale: scale {target.id} is not a COPIES scale")
    if not isinstance(dissolve_unit, bool):
        raise SheetGrammarError(f"split_scale: dissolve_unit must be a bool: {dissolve_unit!r}")
    first_mult = _mult_arg(first, "split_scale")
    second = _mult_from_expr(target.multiplicity.to_sympy() - first_mult.to_sympy(), "split_scale")
    if first_mult == Mult(0) or second == Mult(0):
        raise SheetDomainError(
            f"split_scale: {target.multiplicity} = {first_mult} + {second} has a zero part"
        )
    new_root = Scale(ScaleId(_next_scale_id(s)), ScaleKind.COPIES, second, target.parent)
    scale_map, added = _copy_scales(s, target, new_root, _next_scale_id(s) + 1)
    originals = _subtree_nodes(s, target.id)
    node_map = {n.id: NodeId(_next_node_id(s) + rank) for rank, n in enumerate(originals)}
    duplicates = tuple(
        ScaledNode(
            node_map[n.id],
            n.generator_type,
            _map_ports(n.inputs, scale_map),
            _map_ports(n.outputs, scale_map),
            n.phase,
            None if n.scale is None else scale_map[n.scale],
        )
        for n in originals
    )
    wires = frozenset(
        Wire(
            PortRef(node_map[w.a.node_id], w.a.direction, w.a.index),
            PortRef(node_map[w.b.node_id], w.b.direction, w.b.index),
        )
        for w in s.wires
        if w.a.node_id in node_map
    )

    def pair(b: Bundle) -> tuple[BoundaryItem, ...]:
        if b.scale != target.id:
            return (b,)
        return (b, *_relabel((b,), node_map, scale_map))

    result = replace(
        s,
        nodes=(*s.nodes, *duplicates),
        wires=s.wires | wires,
        scales=(
            *(
                replace(sc, multiplicity=first_mult) if sc.id == target.id else sc
                for sc in s.scales
            ),
            *added,
        ),
        inputs=_transform(s.inputs, _keep_ref, pair),
        outputs=_transform(s.outputs, _keep_ref, pair),
    )
    if dissolve_unit:
        for part in (new_root.id, target.id):
            current = result.scale(part)
            if current.multiplicity == Mult(1):
                result = _dissolve(result, current)
    arguments = {"dissolve_unit": dissolve_unit, "first": first_mult, "scale": target.id}
    return _finish("split_scale", arguments, s, result)


class _Correspondence:
    """The scale and node bijections between two sibling subtrees."""

    __slots__ = ("nodes", "scales")

    def __init__(self, first: ScaleId, second: ScaleId) -> None:
        self.scales: dict[ScaleId, ScaleId] = {first: second}
        self.nodes: dict[NodeId, NodeId] = {}

    def bind(self, a: ScaleId, b: ScaleId, what: str) -> None:
        """Record ``a`` -> ``b``, refusing a conflicting earlier pairing."""
        if self.scales.setdefault(a, b) != b:
            raise SheetGrammarError(
                f"join_scales: {what}: scale {a} pairs with both {self.scales[a]} and {b}"
            )


def _upto(s: ScalableDiagram, scale_id: ScaleId | None, root: ScaleId) -> list[ScaleId]:
    """The chain of ``scale_id``, innermost first, through ``root`` inclusive."""
    found: list[ScaleId] = []
    for scale in s.chain(scale_id):
        found.append(scale.id)
        if scale.id == root:
            break
    return found


def _pair_node(
    s: ScalableDiagram,
    corr: _Correspondence,
    a: ScaledNode,
    b: ScaledNode,
    roots: tuple[ScaleId, ScaleId],
) -> None:
    what = f"nodes {a.id} and {b.id}"
    if a.generator_type != b.generator_type or a.phase != b.phase:
        raise SheetGrammarError(f"join_scales: {what} differ in generator or phase")
    if [p.dim for p in a.inputs] != [p.dim for p in b.inputs] or [p.dim for p in a.outputs] != [
        p.dim for p in b.outputs
    ]:
        raise SheetGrammarError(f"join_scales: {what} differ in ports")
    chains = [
        (_upto(s, a.scale, roots[0]), _upto(s, b.scale, roots[1])),
        *(
            (list(s.fan_chain(pa.fan)), list(s.fan_chain(pb.fan)))
            for pa, pb in zip((*a.inputs, *a.outputs), (*b.inputs, *b.outputs), strict=True)
        ),
    ]
    for chain_a, chain_b in chains:
        if len(chain_a) != len(chain_b):
            raise SheetGrammarError(f"join_scales: {what} differ in scale nesting")
        for x, y in zip(chain_a, chain_b, strict=True):
            corr.bind(x, y, what)
    corr.nodes[a.id] = b.id


def _check_scale_pairing(s: ScalableDiagram, corr: _Correspondence, one: Scale, two: Scale) -> None:
    left = {sc.id for sc in _subtree_scales(s, one.id)}
    right = {sc.id for sc in _subtree_scales(s, two.id)}
    if set(corr.scales) != left or sorted(corr.scales.values()) != sorted(right):
        raise SheetGrammarError(
            f"join_scales: the scales below {one.id} and {two.id} do not correspond"
        )
    for x, y in sorted(corr.scales.items()):
        if x == one.id:
            continue
        sx, sy = s.scale(x), s.scale(y)
        if sx.kind is not sy.kind or sx.multiplicity != sy.multiplicity:
            raise SheetGrammarError(f"join_scales: scales {x} and {y} differ in kind or size")
        if sx.parent is None or corr.scales.get(sx.parent) != sy.parent:
            raise SheetGrammarError(f"join_scales: scales {x} and {y} differ in parent")


def _drop_second(
    s: ScalableDiagram,
    items: tuple[BoundaryItem, ...],
    one: Scale,
    two: Scale,
    corr: _Correspondence,
    side: str,
) -> tuple[BoundaryItem, ...]:
    region = _region_items(s, items, one.parent) or ()
    where = {
        item.scale: index
        for index, item in enumerate(region)
        if isinstance(item, Bundle) and item.scale in (one.id, two.id)
    }
    if not where:
        return items
    if len(where) != 2 or where[two.id] != where[one.id] + 1:
        raise SheetGrammarError(
            f"join_scales: {side}: the bundle of {two.id} does not directly follow that of {one.id}"
        )
    expected = _relabel((region[where[one.id]],), corr.nodes, corr.scales)
    if expected != (region[where[two.id]],):
        raise SheetGrammarError(f"join_scales: {side}: the two bundles do not correspond")
    return _transform(items, _keep_ref, lambda b: () if b.scale == two.id else (b,))


def join_scales(s: ScalableDiagram, first: ScaleId, second: ScaleId) -> SheetStep:
    """Merge two sibling COPIES scales into ``first``, carrying the sum of both multiplicities.

    The subtrees must correspond: nodes paired in ascending id order agree on generator,
    dimensions, phase, scale nesting and fans; descendant scales pair with equal kinds,
    multiplicities and parents; wires map onto wires; on each side the bundle of ``second``
    immediately follows the bundle of ``first`` with corresponding items, or neither appears.
    The ``second`` subtree is removed.
    """
    s = _require_valid(s, "join_scales")
    one = _scale_arg(s, first, "join_scales")
    two = _scale_arg(s, second, "join_scales")
    if one.id == two.id:
        raise SheetGrammarError("join_scales: first and second are the same scale")
    if one.kind is not ScaleKind.COPIES or two.kind is not ScaleKind.COPIES:
        raise SheetGrammarError("join_scales: both scales must be COPIES scales")
    if one.parent != two.parent:
        raise SheetGrammarError(f"join_scales: scales {one.id} and {two.id} are not siblings")
    corr = _Correspondence(one.id, two.id)
    nodes_one = _subtree_nodes(s, one.id)
    nodes_two = _subtree_nodes(s, two.id)
    if len(nodes_one) != len(nodes_two):
        raise SheetGrammarError(
            f"join_scales: node counts differ ({len(nodes_one)} and {len(nodes_two)})"
        )
    for a, b in zip(nodes_one, nodes_two, strict=True):
        _pair_node(s, corr, a, b, (one.id, two.id))
    _check_scale_pairing(s, corr, one, two)
    removed = frozenset(corr.nodes.values())
    mapped = frozenset(
        Wire(
            PortRef(corr.nodes[w.a.node_id], w.a.direction, w.a.index),
            PortRef(corr.nodes[w.b.node_id], w.b.direction, w.b.index),
        )
        for w in s.wires
        if w.a.node_id in corr.nodes
    )
    own = frozenset(w for w in s.wires if w.a.node_id in removed)
    if mapped != own:
        raise SheetGrammarError("join_scales: the subtrees' wires do not correspond")
    gone = frozenset(corr.scales.values())
    result = replace(
        s,
        nodes=tuple(n for n in s.nodes if n.id not in removed),
        wires=s.wires - own,
        scales=tuple(
            replace(sc, multiplicity=one.multiplicity + two.multiplicity) if sc.id == one.id else sc
            for sc in s.scales
            if sc.id not in gone
        ),
        inputs=_drop_second(s, s.inputs, one, two, corr, "inputs"),
        outputs=_drop_second(s, s.outputs, one, two, corr, "outputs"),
    )
    return _finish("join_scales", {"first": one.id, "second": two.id}, s, result)


# -- enclose ---------------------------------------------------------------------------------


def _node_ids_arg(s: ScalableDiagram, nodes: object) -> tuple[NodeId, ...]:
    if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Iterable):
        raise SheetGrammarError(f"enclose: nodes must be an iterable of node ids, got {nodes!r}")
    chosen = tuple(nodes)
    if not chosen:
        raise SheetGrammarError("enclose: the node set is empty")
    known = {n.id for n in s.nodes}
    for node_id in chosen:
        if not _is_int(node_id) or node_id not in known:
            raise SheetGrammarError(f"enclose: no such node {node_id!r}")
    return tuple(sorted({NodeId(int(n)) for n in chosen}))


def _covered_nodes(s: ScalableDiagram, legs: ScaleId) -> frozenset[NodeId]:
    return frozenset(
        n.id
        for n in s.nodes
        for p in (*n.inputs, *n.outputs)
        if legs in {c.id for c in s.chain(p.fan)}
    )


def enclose(
    s: ScalableDiagram, nodes: Iterable[NodeId], *, multiplicity: Mult | int = 1
) -> SheetStep:
    """Wrap ``nodes`` in a new COPIES scale of multiplicity 1 under their common scale ``P``.

    ``P`` is the scale of the shallowest chosen nodes, which must all share it; every deeper
    chosen node lies in a child COPIES scale of ``P`` whose nodes are all chosen, and that
    child is reparented. LEGS scales under ``P`` covering only chosen nodes are reparented.
    No wire may join a chosen node to another node, and on each side the chosen nodes' items
    must form one contiguous run of ``P``'s direct items, which becomes the new bundle.
    """
    s = _require_valid(s, "enclose")
    if _mult_arg(multiplicity, "enclose") != Mult(1):
        raise SheetDomainError(f"enclose: multiplicity must be 1, got {multiplicity}")
    chosen = _node_ids_arg(s, nodes)
    members = frozenset(chosen)
    depth = {n: len(s.copies_chain(s.node(n).scale)) for n in chosen}
    shallowest = min(depth.values())
    tops = {s.node(n).scale for n in chosen if depth[n] == shallowest}
    if len(tops) != 1:
        raise SheetGrammarError(f"enclose: nodes lie in different scales {sorted(map(str, tops))}")
    (region,) = tops
    direct = frozenset(n for n in chosen if depth[n] == shallowest)
    children: set[ScaleId] = set()
    for n in chosen:
        if n in direct:
            continue
        chain = s.copies_chain(s.node(n).scale)
        if region is not None and region not in chain:
            raise SheetGrammarError(f"enclose: node {n} lies outside scale {region}")
        children.add(chain[shallowest])
    for child in sorted(children):
        if not {m.id for m in _subtree_nodes(s, child)} <= members:
            raise SheetGrammarError(f"enclose: scale {child} is only partly enclosed")
    for wire in sorted(s.wires, key=lambda w: w.sort_key()):
        if (wire.a.node_id in members) != (wire.b.node_id in members):
            raise SheetGrammarError(f"enclose: {wire} joins an enclosed node to another node")
    legs: set[ScaleId] = set()
    for scale in s.scales:
        if scale.kind is ScaleKind.LEGS and scale.parent == region:
            covered = _covered_nodes(s, scale.id)
            if covered <= direct:
                legs.add(scale.id)
            elif covered & direct:
                raise SheetGrammarError(f"enclose: LEGS scale {scale.id} is only partly enclosed")
    new = Scale(ScaleId(_next_scale_id(s)), ScaleKind.COPIES, Mult(1), region)
    moved = children | legs

    def wrap(items: tuple[BoundaryItem, ...], side: str) -> tuple[BoundaryItem, ...]:
        hits = [
            i
            for i, item in enumerate(items)
            if (isinstance(item, PortRef) and item.node_id in direct)
            or (isinstance(item, Bundle) and item.scale in children)
        ]
        if not hits:
            return items
        if hits != list(range(hits[0], hits[-1] + 1)):
            raise SheetGrammarError(f"enclose: {side}: the enclosed items are not contiguous")
        return (
            *items[: hits[0]],
            Bundle(new.id, items[hits[0] : hits[-1] + 1]),
            *items[hits[-1] + 1 :],
        )

    def side(items: tuple[BoundaryItem, ...], name: str) -> tuple[BoundaryItem, ...]:
        if region is None:
            return wrap(items, name)
        return _transform(
            items,
            _keep_ref,
            lambda b: (Bundle(b.scale, wrap(b.items, name)),) if b.scale == region else (b,),
        )

    result = replace(
        s,
        nodes=tuple(replace(n, scale=new.id) if n.id in direct else n for n in s.nodes),
        scales=(
            *(replace(sc, parent=new.id) if sc.id in moved else sc for sc in s.scales),
            new,
        ),
        inputs=side(s.inputs, "inputs"),
        outputs=side(s.outputs, "outputs"),
    )
    arguments = {"multiplicity": Mult(1), "nodes": chosen}
    return _finish("enclose", arguments, s, result)


# -- kill ------------------------------------------------------------------------------------


def _kill_copies(s: ScalableDiagram, target: Scale) -> ScalableDiagram:
    dead_nodes = {n.id for n in _subtree_nodes(s, target.id)}
    dead_scales = {sc.id for sc in _subtree_scales(s, target.id)}

    def ref(r: PortRef) -> tuple[BoundaryItem, ...]:
        return () if r.node_id in dead_nodes else (r,)

    def bundle(b: Bundle) -> tuple[BoundaryItem, ...]:
        return (b,) if b.items and b.scale not in dead_scales else ()

    return replace(
        s,
        nodes=tuple(n for n in s.nodes if n.id not in dead_nodes),
        wires=frozenset(w for w in s.wires if w.a.node_id not in dead_nodes),
        scales=tuple(sc for sc in s.scales if sc.id not in dead_scales),
        inputs=_transform(s.inputs, ref, bundle),
        outputs=_transform(s.outputs, ref, bundle),
    )


def _kill_legs(s: ScalableDiagram, target: Scale) -> ScalableDiagram:
    dead_scales = {sc.id for sc in _subtree_scales(s, target.id)}
    index_map: dict[PortRef, PortRef] = {}
    nodes: list[ScaledNode] = []
    for node in s.nodes:
        kept: dict[Direction, list[SheetPort]] = {Direction.INPUT: [], Direction.OUTPUT: []}
        for direction in (Direction.INPUT, Direction.OUTPUT):
            for index, port in enumerate(node.legs(direction)):
                if target.id in {c.id for c in s.chain(port.fan)}:
                    continue
                index_map[PortRef(node.id, direction, index)] = PortRef(
                    node.id, direction, len(kept[direction])
                )
                kept[direction].append(port)
        nodes.append(
            replace(
                node, inputs=tuple(kept[Direction.INPUT]), outputs=tuple(kept[Direction.OUTPUT])
            )
        )

    def ref(r: PortRef) -> tuple[BoundaryItem, ...]:
        return (index_map[r],) if r in index_map else ()

    return replace(
        s,
        nodes=tuple(nodes),
        wires=frozenset(Wire(index_map[w.a], index_map[w.b]) for w in s.wires),
        scales=tuple(sc for sc in s.scales if sc.id not in dead_scales),
        inputs=_transform(s.inputs, ref, _keep_nonempty),
        outputs=_transform(s.outputs, ref, _keep_nonempty),
    )


def kill_scale(s: ScalableDiagram, scale: ScaleId) -> SheetStep:
    """Remove a scale of multiplicity exactly 0 together with everything it scales.

    COPIES: the subtree's nodes, wires, scales and bundles vanish. LEGS: every port it covers
    leaves its node, later indices shifting down, and its subtree's scales vanish. Emptied
    bundles are removed.
    """
    s = _require_valid(s, "kill_scale")
    target = _scale_arg(s, scale, "kill_scale")
    if target.multiplicity != Mult(0):
        raise SheetDomainError(
            f"kill_scale: scale {target.id} has multiplicity {target.multiplicity}"
        )
    if target.kind is ScaleKind.COPIES:
        result = _kill_copies(s, target)
    else:
        result = _kill_legs(s, target)
    return _finish("kill_scale", {"scale": target.id}, s, result)


# -- fusion ----------------------------------------------------------------------------------

_FUSABLE = (Z_SPIDER, X_SPIDER)


def fuse_sheet(s: ScalableDiagram, wire: Wire) -> SheetStep:
    """Fuse the two same-colour spiders ``wire`` joins, once per copy of their shared scale.

    Mirrors :data:`~archytaszx.rewrite.rules_library.SPIDER_FUSION`: the lower node id is A;
    the merged node, with a fresh id and the same scale, has A's surviving inputs then B's,
    and outputs likewise, each keeping its fan; its phase is the sum of both (None when both
    are None and a leg survives); every other wire between the pair becomes a self-loop. X
    spiders need an output-to-input wire. Every leg and phase must carry the connecting
    pair's exact dimension. A rule scalar other than one is raised to the concrete scale size.
    """
    s = _require_valid(s, "fuse_sheet")
    if not isinstance(wire, Wire) or wire not in s.wires:
        raise SheetGrammarError(f"fuse_sheet: {wire!r} is not a wire of the diagram")
    if wire.a.node_id == wire.b.node_id:
        raise SheetGrammarError(f"fuse_sheet: {wire!r} is a self-loop")
    ref_a, ref_b = sorted((wire.a, wire.b), key=lambda r: int(r.node_id))
    node_a, node_b = s.node(ref_a.node_id), s.node(ref_b.node_id)
    if node_a.generator_type != node_b.generator_type or node_a.generator_type not in _FUSABLE:
        raise SheetDomainError("fuse_sheet: the nodes are not two spiders of the same colour")
    if node_a.generator_type == X_SPIDER and ref_a.direction is ref_b.direction:
        raise SheetDomainError("fuse_sheet: an X fusion wire must run from output to input")
    if node_a.scale != node_b.scale:
        raise SheetDomainError("fuse_sheet: the nodes lie in different scales")
    shared = s.port(ref_a).dim
    legs: list[tuple[Direction, PortRef, SheetPort]] = []
    for direction in (Direction.INPUT, Direction.OUTPUT):
        for node, consumed in ((node_a, ref_a), (node_b, ref_b)):
            for index, port in enumerate(node.legs(direction)):
                ref = PortRef(node.id, direction, index)
                if ref != consumed:
                    legs.append((direction, ref, port))
    dims = [s.port(ref_b).dim, *(port.dim for _, _, port in legs)]
    dims += [n.phase.dim for n in (node_a, node_b) if n.phase is not None]
    if any(dim != shared for dim in dims):
        raise SheetDomainError(f"fuse_sheet: a leg or phase is not over dimension {shared}")
    factor = SPIDER_FUSION.scalar_introduced
    scalar = s.scalar
    if not factor.is_one:
        size = s.size(node_a.scale)
        if not size.is_concrete:
            raise SheetDomainError(f"fuse_sheet: scalar {factor} at symbolic size {size}")
        scalar = scalar * factor ** size.to_int()
    if node_a.phase is None and node_b.phase is None:
        phase = None if legs else PhaseVector(shared)
    else:
        phase = (node_a.phase or PhaseVector(shared)) + (node_b.phase or PhaseVector(shared))
    new_id = NodeId(_next_node_id(s))
    mapping: dict[PortRef, PortRef] = {}
    ports: dict[Direction, list[SheetPort]] = {Direction.INPUT: [], Direction.OUTPUT: []}
    for direction, ref, port in legs:
        mapping[ref] = PortRef(new_id, direction, len(ports[direction]))
        ports[direction].append(port)
    merged = ScaledNode(
        new_id,
        node_a.generator_type,
        tuple(ports[Direction.INPUT]),
        tuple(ports[Direction.OUTPUT]),
        phase,
        node_a.scale,
    )
    result = replace(
        s,
        nodes=(*(n for n in s.nodes if n.id not in (node_a.id, node_b.id)), merged),
        wires=frozenset(
            Wire(mapping.get(w.a, w.a), mapping.get(w.b, w.b)) for w in s.wires if w != wire
        ),
        inputs=_transform(s.inputs, lambda r: (mapping.get(r, r),), _keep_nonempty),
        outputs=_transform(s.outputs, lambda r: (mapping.get(r, r),), _keep_nonempty),
        scalar=scalar,
    )
    return _finish("fuse_sheet", {"wire": wire}, s, result)


# -- registry and replay ---------------------------------------------------------------------

SHEET_OPERATIONS: Mapping[str, Callable[..., SheetStep]] = MappingProxyType(
    {
        "dissolve": dissolve,
        "enclose": enclose,
        "fuse_sheet": fuse_sheet,
        "join_scales": join_scales,
        "kill_scale": kill_scale,
        "split_scale": split_scale,
    }
)
"""Every sheet operation by name."""


def replay_sheet_step(step: SheetStep) -> SheetStep:
    """Re-run ``step``'s operation on ``step.before`` and return the fresh step.

    Raises :class:`SheetGrammarError` when the operation is unknown or the fresh step's
    arguments or ``after`` differ from ``step``'s.
    """
    if not isinstance(step, SheetStep):
        raise SheetGrammarError(f"replay_sheet_step requires a SheetStep, got {step!r}")
    operation = SHEET_OPERATIONS.get(step.operation)
    if operation is None:
        raise SheetGrammarError(f"replay_sheet_step: unknown operation {step.operation!r}")
    try:
        fresh = operation(step.before, **dict(step.arguments))
    except TypeError as exc:
        raise SheetGrammarError(f"replay_sheet_step: bad arguments: {exc}") from exc
    if fresh.arguments != step.arguments:
        raise SheetGrammarError("replay_sheet_step: the step's arguments are not normalized")
    if fresh.after != step.after:
        raise SheetGrammarError(
            f"replay_sheet_step: {step.operation} does not reproduce the recorded result"
        )
    return fresh
