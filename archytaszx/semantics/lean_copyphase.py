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

"""Lean family exports of boxed diagrams whose one node box reads its copy index in one node
phase per side.

:func:`gadget` makes the node ``x`` whose phase is ``s j k / d`` (``j`` the box index, ``s = 1``
for a Z spider, ``-1`` for an X spider) phaseless with one more input: a Z spider's wired from
a new Fourier box in the box whose input is a new last boundary input ``K``, an X spider's the
boundary input ``K`` itself; the box loses its index. The normaliser proves the two gadget
diagrams equal with ``K`` open.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.phase import INDEX, PhaseVector
from archytaszx.diagram.generators import FOURIER_BOX
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.semantics.lean_net import NetUnsupported


@dataclass(frozen=True, slots=True)
class Gadget:
    """A gadget diagram: the diagram, the node ``x``, its kind, its sign and the Fourier box
    (``None`` for an X spider)."""

    diagram: Diagram
    node: NodeId
    kind: str
    sign: int
    box: NodeId | None


def _reads(expr: sp.Expr, name: str) -> bool:
    return name in {str(s) for s in expr.free_symbols}


def gadget(d: Diagram) -> Gadget:
    """The gadget of ``d`` (see the module docstring). Raises :class:`NetUnsupported` unless ``d``
    has one node box, with an index, no parent, links, pairs or copy scalar, read only by one Z or
    X spider's phase ``s j k / d`` over its legs' dimension ``d``, a dimension symbol."""
    if len(d.bang_boxes) != 1:
        raise NetUnsupported("copy-index phases need one bang box")
    ((bid, box),) = d.bang_boxes.items()
    if (
        box.index is None
        or box.parent is not None
        or box.links
        or box.cyclic
        or box.pair_of is not None
        or box.copy_scalar is not None
        or box.port_scope
        or box.tree
        or box.align is not None
    ):
        raise NetUnsupported("copy-index phases need a plain indexed node box")
    j = str(box.index)
    readers = []
    for nid, node in d.nodes.items():
        if any(_reads(p.dim.to_sympy(), j) for p in (*node.inputs, *node.outputs)):
            raise NetUnsupported("a leg dimension reads the box index")
        ph = node.phase
        if ph is not None and j in {str(s) for s in ph.free_symbols}:
            readers.append(nid)
    if len(readers) != 1:
        raise NetUnsupported("copy-index phases need exactly one node reading the index")
    (x,) = readers
    node = d.nodes[x]
    kind = node.generator_type.name
    if kind not in ("Z", "X") or x not in box.node_scope:
        raise NetUnsupported("the node reading the index is not a Z or X spider in the box")
    legs = [*node.inputs, *node.outputs]
    phase = node.phase
    if not legs or not isinstance(phase, PhaseVector) or phase.entries() or phase.rule is None:
        raise NetUnsupported("the index phase is not a rule")
    dim = legs[0].dim
    dsym = dim.to_sympy()
    if any(p.dim != dim for p in legs) or phase.dim != dim or not isinstance(dsym, sp.Symbol):
        raise NetUnsupported("the index phase's legs are not of one dimension symbol")
    sign = 1 if kind == "Z" else -1
    jsym = next(s for s in phase.rule.to_sympy_turns().free_symbols if str(s) == j)
    if sp.simplify(phase.rule.to_sympy_turns() - sign * jsym * INDEX / dsym) != 0:
        raise NetUnsupported("the index phase is not s j k / d")
    g = d.copy()
    wires = [w for w in g.wires if x in (w.a.node_id, w.b.node_id)]
    ins, outs = list(g.boundary_inputs), list(g.boundary_outputs)
    n_in = node.num_inputs
    g.remove_node(x)
    y = g.add_node(
        node.generator_type,
        [*(p.dim for p in node.inputs), dim],
        [p.dim for p in node.outputs],
        None,
    )

    def moved(r: PortRef) -> PortRef:
        return PortRef(y, r.direction, r.index) if r.node_id == x else r

    for w in wires:
        g.add_wire(moved(w.a), moved(w.b))
    leg = PortRef(y, Direction.INPUT, n_in)
    scope = (box.node_scope - {x}) | {y}
    f: NodeId | None = None
    if kind == "Z":
        f = g.add_node(FOURIER_BOX, [dim], [dim])
        g.add_wire(PortRef(f, Direction.OUTPUT, 0), leg)
        k_port = PortRef(f, Direction.INPUT, 0)
        scope |= {f}
    else:
        k_port = leg
    g.set_boundary_inputs([*map(moved, ins), k_port])
    g.set_boundary_outputs(list(map(moved, outs)))
    g.set_bang_box_node_scope(bid, frozenset(scope))
    g.set_bang_box_extension(bid, index=None)
    return Gadget(g, y, kind, sign, f)
