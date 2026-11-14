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

"""Lean output for the slot layouts of a kernel regroup step (ArchytasZX/LinkedRegroup.lean).
Both diagrams are one linked diagram ``L`` (:func:`lean_linked.linked`, which keeps no box
groups); each diagram's box groups are a list of slot blocks ``(chain, slots)`` over the Lean
boundary ``outs ++ ins``, one block per box group, per other nodes of a box and per unboxed slot,
in the order of their first slots (:func:`blocks`). ``export_regroup`` writes ``L``, both layouts,
and ``main`` through ``regroup_layout``; None outside that form."""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.lean_linked import (
    LDiag,
    LinkedUnsupported,
    Port,
    _lnat,
    _Tables,
    linked,
    literal,
)

Block = tuple[tuple[int, ...], tuple[int, ...]]
"""A slot block: a chain of multiplicity indices and its slots."""


def blocks(d: Diagram, tables: _Tables) -> list[Block] | None:
    """``d``'s slot blocks over ``outs ++ ins``; None when a slot lies in more than one box, in a
    box with a parent, or the diagram has a port box."""
    if any(box.port_scope for box in d.bang_boxes.values()):
        return None
    out: list[Block] = []
    offset = 0
    for refs in (d.boundary_outputs, d.boundary_inputs):
        at: dict[object, int] = {}
        for i, ref in enumerate(refs):
            owners = [b for b, box in d.bang_boxes.items() if ref.node_id in box.node_scope]
            if not owners:
                out.append(((), (offset + i,)))
                continue
            if len(owners) > 1 or d.bang_boxes[owners[0]].parent is not None:
                return None
            box = d.bang_boxes[owners[0]]
            group = next((j for j, g in enumerate(box.groups) if ref.node_id in g), -1)
            key = (box.id, group)
            if key not in at:
                m = _Tables.index(tables.mults, str(sp.expand(box.multiplicity.to_sympy())))
                at[key] = len(out)
                out.append(((m,), ()))
            chain, slots = out[at[key]]
            out[at[key]] = (chain, (*slots, offset + i))
        offset += len(refs)
    return out


def _leg_chain(L: LDiag, p: Port) -> list[int]:
    """The Lean ``legChain`` of ``p`` (no port boxes)."""
    a = L.chain(p[0])
    partner = next((w[1] if w[0] == p else w[0] for w in L.wires if p in w), None)
    if partner is None:
        return a
    b = L.chain(partner[0])
    return a + b[len(a) :] if len(a) < len(b) else a


def covers(L: LDiag, G: list[Block]) -> bool:
    """The Lean ``Covers L G``."""
    bnd = L.bnd()
    slots = [i for _, s in G for i in s]
    if len(set(slots)) != len(slots) or set(slots) != set(range(len(bnd))):
        return False
    return all(_leg_chain(L, bnd[i]) == list(c) for c, s in G for i in s)


def _blocks_lean(G: list[Block]) -> str:
    return "[" + ", ".join(f"({_lnat(c)}, {_lnat(s)})" for c, s in G) + "]"


def export_regroup(new: Diagram, old: Diagram) -> str | None:
    """Lean source whose ``main`` proves the layout of ``new``'s slots a permutation of
    ``old``'s, ``lperm`` one to one and onto, and the one linked digit sum read through the
    first equal to it read through the second after ``lperm``; None when the two differ beyond
    box groups or outside :func:`blocks` and :func:`covers`."""
    tables = _Tables()
    try:
        L = linked(old, tables)
        R = linked(new, tables)
    except LinkedUnsupported:
        return None
    if L != R or old.scalar != new.scalar:
        return None
    G, G2 = blocks(old, tables), blocks(new, tables)
    if G is None or G2 is None or not covers(L, G) or not covers(L, G2):
        return None
    lay, lay2 = "(layout ks G)", "(layout ks G')"
    return (
        "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
        "open ArchytasZX\n\n"
        "/-- Both diagrams, as one linked diagram. -/\n"
        f"def L : LDiagram :=\n  {literal(L)}\n\n"
        "/-- The slot blocks of the premise's side. -/\n"
        f"def G : List Blk := {_blocks_lean(G)}\n\n"
        "/-- The slot blocks of the regrouped side. -/\n"
        f"def G' : List Blk := {_blocks_lean(G2)}\n\n"
        "/-- The regrouped layout permutes the slots; the digit sums read alike through it. -/\n"
        "theorem main : ∀ (E : Env) (ks : ℕ → ℕ) (x : ℕ → ℕ),\n"
        f"    {lay2}.Perm {lay} ∧ Function.Bijective (lperm {lay} {lay2}) ∧\n"
        f"      L.evalL E ks (readAt {lay2} x) =\n"
        f"        L.evalL E ks (readAt {lay} (x ∘ lperm {lay} {lay2})) :=\n"
        "  regroup_layout L G G' (by decide +kernel) (by decide +kernel)\n\n"
        "end ArchytasZX.Export\n"
    )
