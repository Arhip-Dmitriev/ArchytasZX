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

"""Lean output for an induction over a chain box settled by chain_window. export_chain writes
the two families as ``chainNet`` networks (ArchytasZX/ChainInduct.lean): each side's network at
``k = 0`` and its peeled copy, read off the exposed successor instantiated at ``k = 0``, the
base pair and the copy pair proved equal by their path sums (lean_pathsum.net_body), the copy
pair relabelled onto the chain's label scheme, and ``chain_equal`` giving both families equal at
every ``k``. Qubit networks of Z, X and F only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from archytaszx.algebra.pathsum import PathSumUnsupported
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.semantics.lean_pathsum import NET_OPTIONS, net_body, net_env

W = 1 << 20
"""The label stride between chain levels."""

_FREE = 1 << 19
"""The offset of a copy's own boundary labels within its level."""


@dataclass
class _Split:
    """One side at ``k = 0`` and its copy as open diagrams, the cut ports' boundary positions
    in each and the copy's link-out positions."""

    base: Diagram
    copy: Diagram
    base_cut: list[int]
    copy_in: list[int]
    copy_out: list[int]


def _sub(
    diagram: Diagram, nodes: frozenset[NodeId], outs: Sequence[PortRef], ins: Sequence[PortRef]
) -> Diagram:
    """The nodes ``nodes`` of ``diagram`` with their internal wires and the boundary lists
    ``outs``, ``ins``."""
    out = Diagram()
    id_map: dict[NodeId, NodeId] = {}
    for nid in sorted(nodes):
        node = diagram.nodes[nid]
        id_map[nid] = out.add_node(
            node.generator_type,
            [p.dim for p in node.inputs],
            [p.dim for p in node.outputs],
            phase=node.phase,
        )
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if wire.a.node_id in nodes and wire.b.node_id in nodes:
            out.add_wire(_moved(wire.a, id_map), _moved(wire.b, id_map))
    out.set_boundary_outputs([_moved(r, id_map) for r in outs])
    out.set_boundary_inputs([_moved(r, id_map) for r in ins])
    return out


def _moved(ref: PortRef, id_map: Mapping[NodeId, NodeId]) -> PortRef:
    """``ref`` on its node's image."""
    return PortRef(id_map[ref.node_id], ref.direction, ref.index)


def _split(
    exposed: Diagram,
    copy_ids: frozenset[NodeId],
    values: Mapping[str, int],
    step: str,
    base_scalar: Scalar,
) -> _Split:
    """The exposed successor at ``k = 0`` split into the side at ``k = 0`` and the copy; raises
    :class:`PathSumUnsupported` when the copy meets the rest off its link-in ports."""
    from archytaszx.semantics.check import instantiate

    owners = [b for b, box in exposed.bang_boxes.items() if step in box.multiplicity.free_symbols]
    if len(owners) != 1:
        raise PathSumUnsupported("no single chain box carries the step index")
    box = exposed.bang_boxes[owners[0]]
    if not box.links or box.cyclic or box.index is not None or box.copy_scalar is not None:
        raise PathSumUnsupported("the chain box is cyclic, indexed or carries a copy scalar")
    scope_map = dict(zip(sorted(box.node_scope), sorted(copy_ids), strict=True))
    q_ports = [_moved(q, scope_map) for _p, q in box.links]
    p_ports = [_moved(p, scope_map) for p, _q in box.links]
    x0 = instantiate(exposed, {step: 0, **values})
    if not copy_ids <= frozenset(x0.nodes):
        raise PathSumUnsupported("the copy does not survive instantiation at k = 0")
    base_ids = frozenset(x0.nodes) - copy_ids
    cut: dict[PortRef, PortRef] = {}
    for wire in x0.wires:
        a_in, b_in = wire.a.node_id in copy_ids, wire.b.node_id in copy_ids
        if a_in != b_in:
            inside, end = (wire.a, wire.b) if a_in else (wire.b, wire.a)
            cut[inside] = end
    if sorted(cut, key=PortRef.sort_key) != sorted(q_ports, key=PortRef.sort_key):
        raise PathSumUnsupported("the copy meets the rest off its link-in ports")
    far = [cut[q] for q in q_ports]
    b_outs = [r for r in x0.boundary_outputs if r.node_id in base_ids]
    b_ins = [r for r in x0.boundary_inputs if r.node_id in base_ids]
    c_outs = [r for r in x0.boundary_outputs if r.node_id in copy_ids]
    c_ins = [r for r in x0.boundary_inputs if r.node_id in copy_ids]
    b_outs += [r for r in far if r.direction is Direction.OUTPUT]
    b_ins += [r for r in far if r.direction is Direction.INPUT]
    c_outs += [q for q in q_ports if q.direction is Direction.OUTPUT]
    c_ins += [q for q in q_ports if q.direction is Direction.INPUT]
    base = _sub(x0, base_ids, b_outs, b_ins)
    base.multiply_scalar(base_scalar)
    copy = _sub(x0, copy_ids, c_outs, c_ins)
    if x0.scalar.to_sympy() != base_scalar.to_sympy():
        raise PathSumUnsupported("the successor at k = 0 carries a scalar beside the base's")
    b_bnd, c_bnd = [*b_outs, *b_ins], [*c_outs, *c_ins]
    if any(p not in c_bnd for p in p_ports):
        raise PathSumUnsupported("a link-out port is not on the boundary")
    return _Split(
        base,
        copy,
        [b_bnd.index(r) for r in far],
        [c_bnd.index(q) for q in q_ports],
        [c_bnd.index(p) for p in p_ports],
    )


def _relabel(labels: set[int], target: Mapping[int, int]) -> list[tuple[int, int]]:
    """A permutation list ``π`` with ``ren π`` equal to ``target`` on ``labels``."""
    image = {target.get(x, x) for x in labels}
    if len(image) != len(labels):
        raise PathSumUnsupported("the copy relabelling is not injective")
    pairs = [(x, target.get(x, x)) for x in sorted(labels) if target.get(x, x) != x]
    moved_to = sorted(image - labels)
    freed = sorted(labels - image)
    pairs += list(zip(moved_to, freed, strict=True))
    return pairs


def export_chain(
    theorem: object, values: Mapping[str, int] | None = None, name: str = "main"
) -> str:
    """Lean 4 source proving, for every ``k``, that the two sides of a chain_window induction are
    equal as ``chainNet`` families (see the module docstring); raises
    :class:`PathSumUnsupported` outside that class."""
    from archytaszx.diagram.bangbox import Mult
    from archytaszx.semantics.check import instantiate
    from archytaszx.semantics.induction import _peel_step_index, rename_multiplicity
    from archytaszx.semantics.kernel import Theorem
    from archytaszx.semantics.prove import InductionSide

    assert isinstance(theorem, Theorem)
    inf = theorem.inference
    if inf.kind != "induction" or inf.data.get("window") != "cut":
        raise PathSumUnsupported("not an induction settled through a chain cut region")
    data = inf.data
    step = str(data["step"])
    vals = dict(values or {})
    exposed = cast(Diagram, data["exposed"])
    own_k, other_k = cast(Diagram, data["own_k"]), cast(Diagram, data["other_k"])
    region = cast("frozenset[NodeId]", data["region"])
    copy_own = frozenset(exposed.nodes) - region
    succ_x = rename_multiplicity(other_k, step, Mult(step) + 1)
    peeled = _peel_step_index(succ_x, step)
    if peeled is None:
        raise PathSumUnsupported("the other side's chain box does not peel")
    exposed_x, copy_oth = peeled
    s_own = instantiate(own_k, {step: 0, **vals}).scalar
    s_oth = instantiate(other_k, {step: 0, **vals}).scalar
    sp_o = _split(exposed, copy_own, vals, step, s_own)
    sp_x = _split(exposed_x, copy_oth, vals, step, s_oth)
    if (sp_o.base_cut, sp_o.copy_in, sp_o.copy_out) != (sp_x.base_cut, sp_x.copy_in, sp_x.copy_out):
        raise PathSumUnsupported("the two sides' cut and link positions differ")
    phases: list[tuple[str, int]] = []
    base = net_body(sp_o.base, sp_x.base, "baseEq", phases)
    copy = net_body(sp_o.copy, sp_x.copy, "copyEq", phases)
    if base.m != copy.m:
        raise PathSumUnsupported("the base and copy proofs read phases over different moduli")
    labs = {**base.labs, **copy.labs}
    if any(k != 1 for k in labs.values()):
        raise PathSumUnsupported("the chain export covers qubit wires only")
    cut = list(sp_o.base_cut)
    n_copy = len(sp_o.copy.boundary_outputs) + len(sp_o.copy.boundary_inputs)
    target: dict[int, int] = {}
    for t, j in enumerate(sp_o.copy_in):
        target[j] = cut[t]
    for t, j in enumerate(sp_o.copy_out):
        target[j] = cut[t] + W
    for j in range(n_copy):
        target.setdefault(j, W + _FREE + j)
    internal = [x for x in copy.labs if x >= n_copy]
    for x in internal:
        target[x] = W + x
    pi = _relabel(set(range(n_copy)) | set(internal), target)
    side = "start" if data["side"] is InductionSide.START else "goal"
    other = "goal" if side == "start" else "start"
    ns = f"ArchytasZX.ChainProof.{name}"
    lines = [
        "import ArchytasZX.PathSumDigits",
        "import ArchytasZX.ChainInduct",
        "",
        "/-!",
        f"# Chain induction `{name}`",
        "",
        f"The {side} side (peeled) and the {other} side as chain families: their networks at",
        "`k = 0` and their copies proved equal by path sums, the copies relabelled onto the chain",
        "labels, and `chain_equal` for every `k`.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        "open ArchytasZX ArchytasZX.PSem",
        "",
        *NET_OPTIONS,
        "",
        *net_env(phases, labs, "2 ^ max k 1", base.m),
        "namespace Base",
        "",
        *base.lines,
        "end Base",
        "",
        "namespace Copy",
        "",
        *copy.lines,
        "end Copy",
        "",
        f"def W : ℕ := {W}",
        f"def cut : List ℕ := {cut}",
        f"def πC : List (ℕ × ℕ) := {pi}",
        "",
        "def cO : Net := Copy.NL.mapL (ren πC)",
        "def cX : Net := Copy.NR.mapL (ren πC)",
        "",
        "theorem hD : ∀ x y, labDim E x = labDim E y := labDim_two E (fun _ => rfl) (by decide)",
        "",
        f"theorem copyEq : Equal E ({copy.s_l}) cO ({copy.s_r}) cX :=",
        "  Equal.mapL E _ (ren_injective πC (by decide) (by decide)) (fun _ => hD _ _) Copy.copyEq",
        "",
        f"/-- The {side} and {other} families agree at every `k`. -/",
        "theorem main : ∀ k, Equal E",
        f"    (({base.s_l}) * ({copy.s_l}) ^ k) (chainNet W cut Base.NL cO k)",
        f"    (({base.s_r}) * ({copy.s_r}) ^ k) (chainNet W cut Base.NR cX k) :=",
        "  chain_equal E W cut _ _ _ _ hD (by decide) (by decide) (by decide) (by decide) _ _ _ _",
        "    Base.baseEq copyEq",
        "",
        f"end {ns}",
        "",
    ]
    return "\n".join(lines)


__all__ = ["export_chain"]
