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

"""Lean family proofs of a chain box carrying one wire against a node box holding it once.

:func:`export_chain_node` writes a family theorem whose start is a chain box of two phaseless
Z(1 -> 1) spiders, one on a wire of each copy's own and one on a wire carried from copy to copy
(its output linked to the next copy's input, the carried wire's ends on the boundary), and
whose goal is a node box over the same own-wire spider beside one phaseless Z(1 -> 1) on the
carried wire, both of scalar 1. The engine's diagrams are written as Lean ``Diagram`` literals
(the start's carried ends behind the identities ``2`` and ``3``, as ``ArchytasZX/ChainBox.lean``
holds a link port on the boundary), checked equal by ``decide`` to ``ChainNode.DL``, ``cL`` and
``DR`` of ``ArchytasZX/ChainNode.lean``, and ``main_diagrams`` is ``ChainNode.cyclic_every`` at
every dimension value and copy count.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef

_DL = (
    "{ nodes := [⟨0, .Z, 0, 0, [0], [0]⟩, ⟨1, .Z, 0, 0, [0], [0]⟩, ⟨2, .I, 0, 0, [0], [0]⟩,\n"
    "      ⟨3, .I, 0, 0, [0], [0]⟩],\n"
    "    wires := [(⟨2, true, 0⟩, ⟨1, false, 0⟩), (⟨1, true, 0⟩, ⟨3, false, 0⟩)],\n"
    "    outs := [⟨0, true, 0⟩, ⟨3, true, 0⟩],\n"
    "    ins := [⟨2, false, 0⟩, ⟨0, false, 0⟩],\n"
    "    boxes := [⟨0, 0, none, [0, 1], []⟩] }"
)
"""The start with the carried wire's boundary ends behind identities."""

_CL = "⟨⟨0, 0, none, [0, 1], []⟩, [(⟨1, true, 0⟩, ⟨1, false, 0⟩)]⟩"
"""The start's chain box."""

_DR = (
    "{ nodes := [⟨0, .Z, 0, 0, [0], [0]⟩, ⟨1, .Z, 0, 0, [0], [0]⟩],\n"
    "    wires := [],\n"
    "    outs := [⟨1, true, 0⟩, ⟨0, true, 0⟩],\n"
    "    ins := [⟨0, false, 0⟩, ⟨1, false, 0⟩],\n"
    "    boxes := [⟨0, 0, none, [1], []⟩] }"
)
"""The goal."""


def _plain_id(d: Diagram) -> sp.Expr | None:
    """The common leg dimension when every node is a phaseless Z(1 -> 1), else None."""
    dims = set()
    for n in d.nodes.values():
        if n.generator_type.name != "Z" or len(n.inputs) != 1 or len(n.outputs) != 1:
            return None
        if n.phase is not None and not n.phase.is_zero:
            return None
        dims |= {p.dim.to_sympy() for p in (*n.inputs, *n.outputs)}
    return dims.pop() if len(dims) == 1 else None


def _ref(node: int, out: bool) -> PortRef:
    return PortRef(NodeId(node), Direction.OUTPUT if out else Direction.INPUT, 0)


def _is_start(d: Diagram) -> str | None:
    """The box multiplicity's symbol when ``d`` is the start, else None."""
    if set(d.nodes) != {0, 1} or d.wires or len(d.bang_boxes) != 1:
        return None
    (box,) = d.bang_boxes.values()
    if sorted(box.node_scope) != [0, 1] or box.port_scope or box.parent is not None:
        return None
    if tuple(box.links) != ((_ref(1, True), _ref(1, False)),):
        return None
    if tuple(d.boundary_inputs) != (_ref(1, False), _ref(0, False)):
        return None
    if tuple(d.boundary_outputs) != (_ref(0, True), _ref(1, True)):
        return None
    mult = box.multiplicity.to_sympy()
    return str(mult) if mult.is_Symbol else None


def _is_goal(d: Diagram) -> str | None:
    """The box multiplicity's symbol when ``d`` is the goal, else None."""
    if set(d.nodes) != {0, 1} or d.wires or len(d.bang_boxes) != 1:
        return None
    (box,) = d.bang_boxes.values()
    if sorted(box.node_scope) != [1] or box.port_scope or box.parent is not None or box.links:
        return None
    if tuple(d.boundary_inputs) != (_ref(0, False), _ref(1, False)):
        return None
    if tuple(d.boundary_outputs) != (_ref(1, True), _ref(0, True)):
        return None
    mult = box.multiplicity.to_sympy()
    return str(mult) if mult.is_Symbol else None


def export_chain_node(
    lhs: Diagram, rhs: Diagram, name: str = "main", var: Callable[[str], str] = lambda s: s
) -> str | None:
    """The Lean family theorem of ``lhs = rhs`` through ``ChainNode.cyclic_every``, or None
    when the pair is not the chain box carrying one wire against the node box holding it once."""
    n0, n1 = _is_start(lhs), _is_goal(rhs)
    if n0 is None or n0 != n1:
        return None
    dim0, dim1 = _plain_id(lhs), _plain_id(rhs)
    if dim0 is None or dim0 != dim1:
        return None
    if sp.simplify(lhs.scalar.to_sympy() - 1) != 0 or sp.simplify(rhs.scalar.to_sympy() - 1) != 0:
        return None
    if not (dim0.is_Symbol or (dim0.is_Integer and int(dim0) > 0)):
        return None
    mult = var(n0)
    binders = [f"({mult} : ℕ)"]
    if dim0.is_Symbol:
        dv = var(str(dim0))
        binders.insert(0, f"({dv} : ℕ)")
        env, args = f"(E {dv})", f"{dv} {mult}"
        env_def = (
            f"noncomputable def E ({dv} : ℕ) : Env := {{ dimv := fun _ => {dv}, "
            "phase := fun _ _ => 0, labs := [] }"
        )
    else:
        env, args = "E", mult
        env_def = (
            f"noncomputable def E : Env := {{ dimv := fun _ => {int(dim0)}, "
            "phase := fun _ _ => 0, labs := [] }"
        )
    off = f"(2 * {mult} + 2)"
    lines = [
        "/-",
        "  ArchytasZX proof export over the tensor-network semantics of the ArchytasZX Lean",
        "  library: every diagram is a Net, every step proved there.",
        "  a chain box carrying one wire against a node box holding it once",
        f"  (ArchytasZX/ChainNode.lean, cyclic_every), for every {mult}",
        "-/",
        "",
        "import ArchytasZX",
        "",
        "namespace ArchytasZX.Export",
        "",
        "open ArchytasZX",
        "",
        "/-- The start: its carried wire's boundary ends behind the identities `2` and `3`. -/",
        f"def D0 : Diagram :=\n  {_DL}",
        "",
        "/-- The start's chain box: copy `j`'s output `1.out0` feeds copy `j + 1`'s `1.in0`. -/",
        f"def C0 : DChain := {_CL}",
        "",
        "/-- The goal. -/",
        f"def D1 : Diagram :=\n  {_DR}",
        "",
        "theorem D0_eq : D0 = ChainNode.DL := by decide",
        "",
        "theorem C0_eq : C0 = ChainNode.cL := by decide",
        "",
        "theorem D1_eq : D1 = ChainNode.DR := by decide",
        "",
        "/-- The environment: every leg of one dimension, phase row `0` zero. -/",
        env_def,
        "",
        "/-- The start's expansion against the goal's at every copy count. -/",
        f"theorem {name}_diagrams :",
        f"    ∀ {' '.join(binders)},",
        (
            f"      Equal {env} 1 ((D0.expandChain C0 {mult}).toNet {off}) 1 "
            f"((D1.expand (ksOf [{mult}])).toNet {off}) := by"
        ),
        f"  intro {args}",
        "  rw [D0_eq, C0_eq, D1_eq]",
        f"  exact ChainNode.cyclic_every {env} (fun _ => rfl) (fun _ => rfl) {mult} _ le_rfl",
        "",
        "end ArchytasZX.Export",
        "",
    ]
    return "\n".join(lines)
