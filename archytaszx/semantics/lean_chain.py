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
``k = 0`` read off the exposed successor instantiated at ``k = 0``, its copy ``i`` read off the
exposed successor at ``k = i`` with the scalar ratio of the families at ``i + 1`` and ``i``, one
period of copies kept as a table, the base pair and each copy pair proved equal by their path
sums (lean_pathsum.net_body on dimension 2^K, lean_pathsum.qnet_body on an odd prime), the copy
pairs relabelled onto the chain's label scheme, and ``chain_equal`` giving both families equal
at every ``k``. At each checked ``k`` the networks of both sides instantiated at ``k`` are
written out, matched to the chain networks at ``k`` by a label map found by search, and proved
equal (``At{k}.inst``) from ``main k`` by ``Equal.mapL2`` and permutations checked by
``decide``, with the chain scalars proved equal to the instances'.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.pathsum import PathSumUnsupported, constant
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.semantics.lean_pathsum import (
    NET_OPTIONS,
    QNET_OPTIONS,
    NetBody,
    _const_model,
    _scalars,
    net_body,
    net_env,
    qnet_body,
)

W = 1 << 20
"""The label stride between chain levels."""

_FREE = 1 << 19
"""The offset of a copy's own boundary labels within its level."""


@dataclass
class _Split:
    """One side at ``k = 0`` as an open diagram, its copies at ``k = 0 .. P - 1`` (period P),
    the cut ports' boundary positions in the base and in each copy and the copies' link-out
    positions."""

    base: Diagram
    copies: list[Diagram]
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


def _key(diagram: Diagram) -> tuple[object, ...]:
    """The nodes, phases, wires, boundary and scalar of ``diagram`` as a comparable value."""
    nodes = tuple(
        (
            nid,
            n.generator_type.name,
            tuple(map(str, n.inputs)),
            tuple(map(str, n.outputs)),
            repr(n.phase),
        )
        for nid, n in sorted(diagram.nodes.items())
    )
    wires = tuple(sorted(w.sort_key() for w in diagram.wires))
    bnd = (tuple(diagram.boundary_outputs), tuple(diagram.boundary_inputs))
    return nodes, wires, bnd, str(diagram.scalar.to_sympy())


MAX_PERIOD = 64
"""The longest copy period export_chain looks for."""


def _split(
    exposed: Diagram,
    family: Diagram,
    copy_ids: frozenset[NodeId],
    values: Mapping[str, int],
    step: str,
) -> _Split:
    """The exposed successor at ``k = 0`` split into the side at ``k = 0`` and the copy, and
    the copies at ``k = 0 .. P - 1``, P the least period of the copies; raises
    :class:`PathSumUnsupported` when the copy meets the rest off its link-in ports or has no
    period up to :data:`MAX_PERIOD`."""

    from archytaszx.semantics.check import instantiate

    owners = [b for b, box in exposed.bang_boxes.items() if step in box.multiplicity.free_symbols]
    if len(owners) != 1:
        raise PathSumUnsupported("no single chain box carries the step index")
    box = exposed.bang_boxes[owners[0]]
    if not box.links or box.cyclic:
        raise PathSumUnsupported("the chain box is cyclic or has no links")
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
    b_outs += [r for r in far if r.direction is Direction.OUTPUT]
    b_ins += [r for r in far if r.direction is Direction.INPUT]
    base = _sub(x0, base_ids, b_outs, b_ins)
    base.set_scalar(instantiate(family, {step: 0, **values}).scalar)

    def copy_at(j: int) -> tuple[Diagram, list[PortRef]]:
        xj = instantiate(exposed, {step: j, **values})
        if not copy_ids <= frozenset(xj.nodes):
            raise PathSumUnsupported(f"the copy does not survive instantiation at k = {j}")
        c_outs = [r for r in xj.boundary_outputs if r.node_id in copy_ids]
        c_ins = [r for r in xj.boundary_inputs if r.node_id in copy_ids]
        c_outs += [q for q in q_ports if q.direction is Direction.OUTPUT]
        c_ins += [q for q in q_ports if q.direction is Direction.INPUT]
        copy = _sub(xj, copy_ids, c_outs, c_ins)
        here = instantiate(family, {step: j, **values}).scalar.to_sympy()
        if here == 0:
            raise PathSumUnsupported(f"the family at k = {j} has scalar zero")
        copy.set_scalar(Scalar(sp.nsimplify(sp.simplify(xj.scalar.to_sympy() / here))))
        return copy, [*c_outs, *c_ins]

    first, c_bnd = copy_at(0)
    keys = [_key(first)]
    copies = [first]
    period = 0
    for j in range(1, 2 * MAX_PERIOD + 1):
        cj, bj = copy_at(j)
        if bj != c_bnd:
            raise PathSumUnsupported(f"the copy at k = {j} has another boundary")
        keys.append(_key(cj))
        copies.append(cj)
        half = (j + 1) // 2
        if (j + 1) % 2 == 0 and keys[:half] == keys[half:]:
            period = next(
                q
                for q in range(1, half + 1)
                if half % q == 0 and all(keys[t] == keys[t % q] for t in range(j + 1))
            )
            break
    if not period:
        raise PathSumUnsupported(f"the copies have no period up to {MAX_PERIOD}")
    b_bnd = [*b_outs, *b_ins]
    if any(p not in c_bnd for p in p_ports):
        raise PathSumUnsupported("a link-out port is not on the boundary")
    return _Split(
        base,
        copies[:period],
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


NetT = tuple[list[tuple[str, tuple[int, ...], tuple[int, ...], int, int]], list[int], list[int]]
"""A network as nodes (kind, outputs, inputs, dimension index, phase index), summed and boundary
labels."""

_NODE_RE = re.compile(r"⟨(\.\w+), \[([0-9, ]*)\], \[([0-9, ]*)\], (\d+), (\d+)⟩")
_TAIL_RE = re.compile(r"\], \[([0-9, ]*)\], \[([0-9, ]*)\]⟩$")


def _ints(text: str) -> list[int]:
    """The naturals of a comma-separated list."""
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_net(lines: Sequence[str], tag: str) -> NetT:
    """The network ``N{tag}`` of a body's Lean lines."""
    line = next(ln for ln in lines if ln.startswith(f"def N{tag} : Net := "))
    nodes = [
        (m[1], tuple(_ints(m[2])), tuple(_ints(m[3])), int(m[4]), int(m[5]))
        for m in _NODE_RE.finditer(line)
    ]
    tail = _TAIL_RE.search(line)
    assert tail is not None
    return nodes, _ints(tail[1]), _ints(tail[2])


def _ren(pi: Sequence[tuple[int, int]], x: int) -> int:
    """Lean's ``ren pi x``."""
    return next((b for a, b in pi if a == x), x)


def _map_net(net: NetT, f: Callable[[int], int]) -> NetT:
    """Lean's ``Net.mapL f``."""
    nodes, int_, bnd = net
    return (
        [(k, tuple(map(f, o)), tuple(map(f, i)), d, ph) for k, o, i, d, ph in nodes],
        [f(x) for x in int_],
        [f(x) for x in bnd],
    )


def _chain_net(cut: Sequence[int], base: NetT, copies: Sequence[NetT], k: int) -> NetT:
    """Lean's ``chainNet W cut base (perIdx copies) k``."""
    net = base
    for level in range(k):
        c_nodes, c_int, c_bnd = _map_net(
            copies[level % len(copies)], functools.partial(int.__add__, level * W)
        )
        cs = [c + level * W for c in cut]
        net = (
            c_nodes + net[0],
            c_int + cs + net[1],
            [x for x in net[2] if x not in cs] + [x for x in c_bnd if x not in cs],
        )
    return net


def _instance_net(
    diagram: Diagram, phases: Sequence[tuple[str, int]], qudit: bool, m: int
) -> tuple[list[tuple[str, tuple[int, ...], tuple[int, ...], int, str | None]], list[int], int]:
    """The nodes of ``diagram`` (kind, output and input labels, dimension index, phase vector
    body or None), boundary port i labelled i and wire w labelled by the boundary size plus w,
    its summed labels and its boundary size."""
    from archytaszx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
    from archytaszx.semantics.pathsum_diagram import _bits, _turns_table, diagram_base

    if diagram.bang_boxes:
        raise PathSumUnsupported("the instance keeps a bang box")
    names = {Z_SPIDER.name: ".Z", X_SPIDER.name: ".X", FOURIER_BOX.name: ".F"}
    base_d = diagram_base(diagram)
    bnd = [*diagram.boundary_outputs, *diagram.boundary_inputs]
    label = {ref: i for i, ref in enumerate(bnd)}
    summed = []
    for w, wire in enumerate(sorted(diagram.wires, key=lambda w: w.sort_key())):
        label[wire.a] = label[wire.b] = len(bnd) + w
        summed.append(len(bnd) + w)
    out = []
    for nid in sorted(diagram.nodes):
        node = diagram.nodes[nid]
        kind = names.get(node.generator_type.name)
        if kind is None:
            raise PathSumUnsupported(f"node {nid!r} is a {node.generator_type.name}")
        os_ = tuple(label[PortRef(nid, Direction.OUTPUT, j)] for j in range(len(node.outputs)))
        is_ = tuple(label[PortRef(nid, Direction.INPUT, j)] for j in range(len(node.inputs)))
        ref = (
            PortRef(nid, Direction.OUTPUT, 0) if node.outputs else PortRef(nid, Direction.INPUT, 0)
        )
        k = _bits(diagram, ref, base_d)
        body = None
        if kind != ".F":
            size = base_d if qudit else 1 << k
            ts = [int((t % 1) * m) for t in _turns_table(nid, node, size)]
            body = f"(({ts} : List ℕ).getD v 0 : ℕ)" if any(ts) else None
        out.append((kind, os_, is_, 0 if qudit else k, body))
    return out, summed, len(bnd)


def _iso(
    chain: NetT,
    inst: list[tuple[str, tuple[int, ...], tuple[int, ...], int, str | None]],
    body_of: Callable[[int], str | None],
) -> tuple[dict[int, int], list[int]]:
    """A label map from ``chain`` onto the instance and the chain node matched to each instance
    node, nodes matched by kind, dimension, phase vector and legs in order; raises
    :class:`PathSumUnsupported` when there is none."""
    nodes = chain[0]
    if len(nodes) != len(inst):
        raise PathSumUnsupported("the chain and the instance have different node counts")
    fwd: dict[int, int] = {}
    back: dict[int, int] = {}
    used = [False] * len(nodes)
    match: list[int] = []

    def fits(j: int, i: int) -> list[tuple[int, int]] | None:
        k, o, ins, d, ph = nodes[j]
        ik, io, iins, idim, ibody = inst[i]
        if (k, d, body_of(ph)) != (ik, idim, ibody) or len(o) != len(io) or len(ins) != len(iins):
            return None
        new: list[tuple[int, int]] = []
        for a, b in zip((*o, *ins), (*io, *iins), strict=True):
            if fwd.get(a, b) != b or back.get(b, a) != a:
                undo(new)
                return None
            if a not in fwd:
                fwd[a], back[b] = b, a
                new.append((a, b))
        return new

    def undo(new: list[tuple[int, int]]) -> None:
        for a, b in new:
            del fwd[a], back[b]

    order: list[int] = []
    for start in range(len(inst)):
        if start in order:
            continue
        queue = [start]
        order.append(start)
        while queue:
            a = queue.pop(0)
            legs = {*inst[a][1], *inst[a][2]}
            for b in range(len(inst)):
                if b not in order and legs & {*inst[b][1], *inst[b][2]}:
                    order.append(b)
                    queue.append(b)
    found: dict[int, int] = {}

    def go(t: int) -> bool:
        if t == len(order):
            return True
        i = order[t]
        for j in range(len(nodes)):
            if used[j]:
                continue
            new = fits(j, i)
            if new is None:
                continue
            used[j] = True
            found[i] = j
            if go(t + 1):
                return True
            del found[i]
            used[j] = False
            undo(new)
        return False

    if not go(0):
        raise PathSumUnsupported("the chain at k is not the instance's network")
    match[:] = [found[i] for i in range(len(inst))]
    return fwd, match


def _bodies(
    pairs: Sequence[tuple[Diagram, Diagram]], names: Sequence[str]
) -> tuple[list[NetBody], list[tuple[str, int]], bool]:
    """The network proofs of ``pairs`` over one environment: qudit bodies on an odd prime, else
    qubit-digit bodies over one phase modulus; whether they are qudit bodies."""
    from archytaszx.algebra.pathsum import QuditPathSum
    from archytaszx.semantics.pathsum_diagram import from_diagram

    qudit = isinstance(from_diagram(pairs[0][0]), QuditPathSum)
    phases: list[tuple[str, int]] = []
    if qudit:
        return (
            [qnet_body(a, b, n, phases) for (a, b), n in zip(pairs, names, strict=True)],
            (phases),
            True,
        )
    bodies = [net_body(a, b, n, phases) for (a, b), n in zip(pairs, names, strict=True)]
    m = math.lcm(*(body.m for body in bodies))
    if any(body.m != m for body in bodies):
        phases = []
        bodies = [net_body(a, b, n, phases, m) for (a, b), n in zip(pairs, names, strict=True)]
    return bodies, phases, False


def export_chain(
    theorem: object,
    values: Mapping[str, int] | None = None,
    name: str = "main",
    checks: Sequence[int] = (2,),
) -> str:
    """Lean 4 source proving, for every ``k``, that the two sides of a chain_window induction are
    equal as ``chainNet`` families (see the module docstring), and at each ``k`` of ``checks``
    that each family's network relabelled is a permutation of the network of the side
    instantiated at ``k`` (instK), the two instances equal; raises :class:`PathSumUnsupported`
    outside that class."""
    from archytaszx.diagram.bangbox import Mult
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
    sp_o = _split(exposed, own_k, copy_own, vals, step)
    sp_x = _split(exposed_x, other_k, copy_oth, vals, step)
    if (sp_o.base_cut, sp_o.copy_in, sp_o.copy_out) != (sp_x.base_cut, sp_x.copy_in, sp_x.copy_out):
        raise PathSumUnsupported("the two sides' cut and link positions differ")
    period = math.lcm(len(sp_o.copies), len(sp_x.copies))
    pairs = [(sp_o.base, sp_x.base)] + [
        (sp_o.copies[j % len(sp_o.copies)], sp_x.copies[j % len(sp_x.copies)])
        for j in range(period)
    ]
    names = ["baseEq", *("copyEq" for _ in range(period))]
    bodies, phases, qudit = _bodies(pairs, names)
    base, copies = bodies[0], bodies[1:]
    if qudit:
        dims = str(base.m)
        labs: dict[int, int] = {}
    else:
        labs = {k: v for body in bodies for k, v in body.labs.items()}
        ks = set(labs.values())
        if len(ks) > 1:
            raise PathSumUnsupported("the chain export needs one wire dimension")
        dims = f"2 ^ {ks.pop() if ks else 1}"
    cut = list(sp_o.base_cut)
    n_copy = len(sp_o.copies[0].boundary_outputs) + len(sp_o.copies[0].boundary_inputs)
    target: dict[int, int] = {}
    for t, j in enumerate(sp_o.copy_in):
        target[j] = cut[t]
    for t, j in enumerate(sp_o.copy_out):
        target[j] = cut[t] + W
    for j in range(n_copy):
        target.setdefault(j, W + _FREE + j)
    pis = []
    for body in copies:
        internal = [x for x in body.labs if x >= n_copy]
        tgt = dict(target)
        for x in internal:
            tgt[x] = W + x
        pis.append(_relabel(set(range(n_copy)) | set(internal), tgt))
    side = "start" if data["side"] is InductionSide.START else "goal"
    other = "goal" if side == "start" else "start"
    ns = f"ArchytasZX.ChainProof.{name}"
    sem = "QSem" if qudit else "PSem"
    lines = [
        "import ArchytasZX.PathSumQuditNet" if qudit else "import ArchytasZX.PathSumDigits",
        "import ArchytasZX.ChainInduct",
        "",
        "/-!",
        f"# Chain induction `{name}`",
        "",
        f"The {side} side (peeled) and the {other} side as chain families: their networks at",
        "`k = 0` and one period of copies proved equal by path sums, the copies relabelled onto",
        "the chain labels, and `chain_equal` for every `k`.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        f"open ArchytasZX ArchytasZX.{sem}",
        "",
        *(QNET_OPTIONS if qudit else NET_OPTIONS),
        "",
        *([f"instance : Fact (Nat.Prime {base.m}) := ⟨by norm_num⟩", ""] if qudit else []),
        *net_env(phases, labs, dims, base.m),
        "theorem hD : ∀ x y, labDim E x = labDim E y := fun _ _ => rfl",
        "",
        "namespace Base",
        "",
        *base.lines,
        "end Base",
        "",
    ]
    for j, body in enumerate(copies):
        lines += [f"namespace Copy{j}", "", *body.lines, f"end Copy{j}", ""]
    lines += [
        f"def W : ℕ := {W}",
        f"def cut : List ℕ := {cut}",
        *(f"def πC{j} : List (ℕ × ℕ) := {pi}" for j, pi in enumerate(pis)),
        "",
        "def cOt : List Net := ["
        + ", ".join(f"Copy{j}.NL.mapL (ren πC{j})" for j in range(period))
        + "]",
        "def cXt : List Net := ["
        + ", ".join(f"Copy{j}.NR.mapL (ren πC{j})" for j in range(period))
        + "]",
        "noncomputable def aOt : List ℂ := [" + ", ".join(f"({b.s_l})" for b in copies) + "]",
        "noncomputable def aXt : List ℂ := [" + ", ".join(f"({b.s_r})" for b in copies) + "]",
        "",
        "def cO (i : ℕ) : Net := perIdx cOt ⟨[], [], []⟩ i",
        "def cX (i : ℕ) : Net := perIdx cXt ⟨[], [], []⟩ i",
        "noncomputable def aO (i : ℕ) : ℂ := perIdx aOt 1 i",
        "noncomputable def aX (i : ℕ) : ℂ := perIdx aXt 1 i",
        "",
    ]
    for j, body in enumerate(copies):
        lines += [
            (
                f"theorem copyEq{j} : Equal E ({body.s_l}) (Copy{j}.NL.mapL (ren πC{j})) "
                f"({body.s_r}) (Copy{j}.NR.mapL (ren πC{j})) :="
            ),
            (
                f"  Equal.mapL E _ (ren_injective πC{j} (by decide) (by decide)) "
                f"(fun _ => hD _ _) Copy{j}.copyEq"
            ),
            "",
        ]
    lines += [
        "theorem copyEq : ∀ i, Equal E (aO i) (cO i) (aX i) (cX i) := by",
        "  intro i",
        f"  have h := Nat.mod_lt i (show 0 < {period} by norm_num)",
        "  simp only [aO, aX, cO, cX, perIdx, aOt, aXt, cOt, cXt, List.length_cons,",
        "    List.length_nil]",
        f"  generalize i % {period} = j at h ⊢",
        "  interval_cases j",
        *(f"  · exact copyEq{j}" for j in range(period)),
        "",
        "theorem hcO : ∀ i, copyOK W cut (cO i) = true :=",
        "  perIdx_all cOt _ (fun c => copyOK W cut c = true) (by decide) (by decide)",
        "theorem hcX : ∀ i, copyOK W cut (cX i) = true :=",
        "  perIdx_all cXt _ (fun c => copyOK W cut c = true) (by decide) (by decide)",
        "",
        f"/-- The {side} and {other} families agree at every `k`. -/",
        "theorem main : ∀ k, Equal E",
        f"    (({base.s_l}) * ∏ i ∈ Finset.range k, aO i) (chainNet W cut Base.NL cO k)",
        f"    (({base.s_r}) * ∏ i ∈ Finset.range k, aX i) (chainNet W cut Base.NR cX k) :=",
        "  chain_equal E W cut _ _ _ _ hD (by decide) hcO (by decide) hcX _ _ _ _",
        "    Base.baseEq copyEq",
        "",
    ]
    sc = _scalars()

    def model(diagram: Diagram) -> sp.Expr:
        if qudit:
            out = sc.model(diagram.scalar.to_sympy())
            if sc.opaques:
                raise PathSumUnsupported(f"the scalar {diagram.scalar} has no exact Lean form")
            return out
        return _const_model(*constant(diagram.scalar.to_sympy()))

    models = tuple((model(sp_.base), [model(c) for c in sp_.copies]) for sp_ in (sp_o, sp_x))
    lines += _link_lines(
        cut, bodies, pis, phases, qudit, base.m, (own_k, other_k), step, vals, checks, models, model
    )
    lines += [f"end {ns}", ""]
    return "\n".join(lines)


def _link_lines(
    cut: Sequence[int],
    bodies: Sequence[NetBody],
    pis: Sequence[Sequence[tuple[int, int]]],
    phases: Sequence[tuple[str, int]],
    qudit: bool,
    m: int,
    families: tuple[Diagram, Diagram],
    step: str,
    values: Mapping[str, int],
    checks: Sequence[int],
    models: tuple[tuple[sp.Expr, list[sp.Expr]], ...],
    model: Callable[[Diagram], sp.Expr],
) -> list[str]:
    """For each ``k`` of ``checks``: both sides' instances at ``k`` as networks DL, DR, the
    relabellings of the chain networks onto them, their node, summed and boundary permutations
    by ``decide``, instK (the two instances equal under the chain scalars), each side's chain
    scalar at ``k`` equal to its instance's (scalL, scalR) and inst (the two instances equal
    under their own scalars)."""
    from archytaszx.semantics.check import instantiate

    sc = _scalars()

    base, copies = bodies[0], bodies[1:]

    def body_of(ph: int) -> str | None:
        return phases[ph - 1][0] if ph else None

    out: list[str] = []
    for k in checks:
        texts = []
        scal: list[tuple[str, str, str, sp.Expr]] = []
        for (tag, fam), (m_base, m_copies) in zip(
            zip(("L", "R"), families, strict=True), models, strict=True
        ):
            chain = _chain_net(
                cut,
                _parse_net(base.lines, tag),
                [
                    _map_net(_parse_net(body.lines, tag), functools.partial(_ren, pi))
                    for body, pi in zip(copies, pis, strict=True)
                ],
                k,
            )
            inst_d = instantiate(fam, {step: k, **values})
            inst, summed, nb = _instance_net(inst_d, phases, qudit, m)
            lhs = m_base * sp.Mul(*(m_copies[i % len(m_copies)] for i in range(k)))
            rhs = model(inst_d)
            scal.append((tag, sc.text(rhs), sc.prove(lhs, rhs), lhs))
            fwd, match = _iso(chain, inst, body_of)
            labels = {x for n in chain[0] for x in (*n[1], *n[2])} | set(chain[1]) | set(chain[2])
            if sorted(fwd[x] for x in chain[2]) != list(range(nb)):
                raise PathSumUnsupported("the chain boundary is not the instance boundary")
            pi = _relabel(labels, fwd)
            nodes = [
                f"⟨{kd}, {list(o)}, {list(i)}, {d}, {chain[0][j][4]}⟩"
                for (kd, o, i, d, _b), j in zip(inst, match, strict=True)
            ]
            texts.append((tag, pi, f"⟨[{', '.join(nodes)}], {summed}, {list(range(nb))}⟩"))
        out += [f"namespace At{k}", ""]
        for tag, pi, net in texts:
            out += [f"def π{tag} : List (ℕ × ℕ) := {pi}", f"def D{tag} : Net := {net}", ""]
        for tag, fam_net in (("L", "Base.NL cO"), ("R", "Base.NR cX")):
            chain_k = f"(chainNet W cut {fam_net} {k}).mapL (ren π{tag})"
            out += [
                f"theorem link{tag} : ({chain_k}).nodes.Perm D{tag}.nodes ∧",
                f"    ({chain_k}).int.Perm D{tag}.int ∧ ({chain_k}).bnd.Perm D{tag}.bnd := by",
                "  decide",
                "",
            ]
        out += [
            f"/-- Both sides instantiated at `k = {k}` are equal. -/",
            "theorem instK : Equal E",
            f"    (({base.s_l}) * ∏ i ∈ Finset.range {k}, aO i) DL",
            f"    (({base.s_r}) * ∏ i ∈ Finset.range {k}, aX i) DR := by",
            "  have h := Equal.mapL2 E (ren πL) (ren πR)",
            "    (ren_injective πL (by decide) (by decide))",
            "    (ren_injective πR (by decide) (by decide)) (fun _ => hD _ _) (fun _ => hD _ _)",
            f"    (main {k}) (by decide) (by decide)",
            "  exact (h.reorderL E linkL.1 linkL.2.1 linkL.2.2).reorder E linkR.1 linkR.2.1",
            "    linkR.2.2",
            "",
        ]
        for (tag, rhs_text, proof, lhs), (a_tab, b_text) in zip(
            scal, (("aO", base.s_l), ("aX", base.s_r)), strict=True
        ):
            out += [
                f"theorem scal{tag} : ({b_text}) * ∏ i ∈ Finset.range {k}, {a_tab} i =",
                f"    {rhs_text} := by",
                (
                    f"  simp only [Finset.prod_range_succ, Finset.prod_range_zero, {a_tab}, "
                    f"perIdx, {a_tab}t, List.length_cons, List.length_nil, Nat.reduceAdd, "
                    "Nat.reduceMod, List.getD_cons_zero, List.getD_cons_succ, one_mul]"
                ),
                f"  have key : ({sc.text(lhs)} : ℂ) = {rhs_text} :=",
                *("    " + ln for ln in proof.splitlines()),
                "  all_goals linear_combination key",
                "",
            ]
        out += [
            f"/-- Both sides instantiated at `k = {k}` are equal, under their own scalars. -/",
            f"theorem inst : Equal E ({scal[0][1]}) DL ({scal[1][1]}) DR :=",
            "  instK.congr_scalar scalL scalR",
            "",
            f"end At{k}",
            "",
        ]
    return out


__all__ = ["export_chain"]
