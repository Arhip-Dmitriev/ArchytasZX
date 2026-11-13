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
``decide``, with the chain scalars proved equal to the instances'. An offset induction
(induction_offset) takes its copies at ``k = 0`` and offset ``0 .. P - 1``, and ``main s k``
gives level ``j`` of the chain at ``k`` the copy ``s + k - 1 - j``, for every ``s`` and ``k``;
the instances are linked at ``s = 0``. Over mixed qubit wire dimensions (_export_mixed) each
body keeps its own labels' digit counts and ``main`` holds over ``lvEnv E W C B k``
(ArchytasZX/ChainMixed.lean), the instances linked under ``EK``, their labels' digit counts
before ``lvEnv E W C B k``'s.
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

_BASE = 1 << 18
"""The shift of the base labels in a chain over mixed wire dimensions."""


@dataclass
class _Split:
    """One side at ``k = 0`` as an open diagram, its copies at ``k = 0 .. P - 1`` (period P),
    the cut ports' boundary positions in the base and in each copy and the copies' link-out
    positions (the link-in ports of a copy meeting the rest at its link-out ports)."""

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
    offset: str | None = None,
) -> _Split:
    """The exposed successor at ``k = 0`` split into the side at ``k = 0`` and the copy, and
    the copies at ``k = 0 .. P - 1`` (with ``offset``, at ``k = 0`` and offset ``0 .. P - 1``),
    P the least period of the copies; raises
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

    def at(j: int) -> dict[str, int]:
        if offset is None:
            return {step: j, **values}
        return {step: 0, offset: j, **values}

    x0 = instantiate(exposed, at(0))
    if not copy_ids <= frozenset(x0.nodes):
        raise PathSumUnsupported("the copy does not survive instantiation at k = 0")
    base_ids = frozenset(x0.nodes) - copy_ids
    cut: dict[PortRef, PortRef] = {}
    for wire in x0.wires:
        a_in, b_in = wire.a.node_id in copy_ids, wire.b.node_id in copy_ids
        if a_in != b_in:
            inside, end = (wire.a, wire.b) if a_in else (wire.b, wire.a)
            cut[inside] = end
    met = sorted(cut, key=PortRef.sort_key)
    if met == sorted(p_ports, key=PortRef.sort_key) and met != sorted(
        q_ports, key=PortRef.sort_key
    ):
        q_ports, p_ports = p_ports, q_ports
    if met != sorted(q_ports, key=PortRef.sort_key):
        raise PathSumUnsupported("the copy meets the rest off its link ports")
    far = [cut[q] for q in q_ports]
    b_outs = [r for r in x0.boundary_outputs if r.node_id in base_ids]
    b_ins = [r for r in x0.boundary_inputs if r.node_id in base_ids]
    b_outs += [r for r in far if r.direction is Direction.OUTPUT]
    b_ins += [r for r in far if r.direction is Direction.INPUT]
    base = _sub(x0, base_ids, b_outs, b_ins)
    base.set_scalar(instantiate(family, at(0)).scalar)

    def copy_at(j: int, env: dict[str, int] | None = None) -> tuple[Diagram, list[PortRef]]:
        xj = instantiate(exposed, at(j) if env is None else env)
        if not copy_ids <= frozenset(xj.nodes):
            raise PathSumUnsupported(f"the copy does not survive instantiation at k = {j}")
        c_outs = [r for r in xj.boundary_outputs if r.node_id in copy_ids]
        c_ins = [r for r in xj.boundary_inputs if r.node_id in copy_ids]
        c_outs += [q for q in q_ports if q.direction is Direction.OUTPUT]
        c_ins += [q for q in q_ports if q.direction is Direction.INPUT]
        copy = _sub(xj, copy_ids, c_outs, c_ins)
        here = instantiate(family, at(j) if env is None else env).scalar.to_sympy()
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
    if offset is not None:
        moved, m_bnd = copy_at(0, {step: 1, offset: 0, **values})
        if m_bnd != c_bnd or _key(moved) != keys[0]:
            raise PathSumUnsupported("the offset copy depends on k")
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


def _chain_net(
    cut: Sequence[int], base: NetT, copies: Sequence[NetT], k: int, offset: bool = False
) -> NetT:
    """Lean's ``chainNet W cut base (perIdx copies) k``; with ``offset``, level ``j`` taking
    copy ``k - 1 - j``."""
    net = base
    for level in range(k):
        c_nodes, c_int, c_bnd = _map_net(
            copies[(k - 1 - level if offset else level) % len(copies)],
            functools.partial(int.__add__, level * W),
        )
        cs = [c + level * W for c in cut]
        net = (
            c_nodes + net[0],
            c_int + cs + net[1],
            [x for x in net[2] if x not in cs] + [x for x in c_bnd if x not in cs],
        )
    return net


def _instance_net(
    diagram: Diagram,
    phases: Sequence[tuple[str, int]],
    qudit: bool,
    m: int,
    wire_base: int | None = None,
) -> tuple[list[tuple[str, tuple[int, ...], tuple[int, ...], int, str | None]], list[int], int]:
    """The nodes of ``diagram`` (kind, output and input labels, dimension index, phase vector
    body or None), boundary port i labelled i and wire w labelled by ``wire_base`` (the
    boundary size when None) plus w, its summed labels and its boundary size."""
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
        first = len(bnd) if wire_base is None else wire_base
        label[wire.a] = label[wire.b] = first + w
        summed.append(first + w)
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
    seed: Mapping[int, int] | None = None,
) -> tuple[dict[int, int], list[int]]:
    """A label map from ``chain`` onto the instance, extending ``seed``, and the chain node
    matched to each instance node, nodes matched by kind, dimension, phase vector and legs in
    order; raises :class:`PathSumUnsupported` when there is none."""
    nodes = chain[0]
    if len(nodes) != len(inst):
        raise PathSumUnsupported("the chain and the instance have different node counts")
    fwd: dict[int, int] = dict(seed or {})
    back: dict[int, int] = {b: a for a, b in fwd.items()}
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
) -> tuple[list[NetBody], list[tuple[str, int]], bool, dict[int, str]]:
    """The network proofs of ``pairs`` over one environment: qudit bodies on an odd prime, else
    qubit-digit bodies over one phase modulus; whether they are qudit bodies, and the weighted
    spiders' coefficients by phase index."""
    from archytaszx.algebra.pathsum import QuditPathSum
    from archytaszx.semantics.pathsum_diagram import from_diagram

    qudit = isinstance(from_diagram(pairs[0][0]), QuditPathSum)
    phases: list[tuple[str, int]] = []
    coefs: dict[int, str] = {}
    if qudit:
        bodies = [qnet_body(a, b, n, phases) for (a, b), n in zip(pairs, names, strict=True)]
        return bodies, phases, True, coefs
    bodies = [
        net_body(a, b, n, phases, coefs=coefs) for (a, b), n in zip(pairs, names, strict=True)
    ]
    m = math.lcm(*(body.m for body in bodies))
    if any(body.m != m for body in bodies):
        phases, coefs = [], {}
        bodies = [
            net_body(a, b, n, phases, m, coefs) for (a, b), n in zip(pairs, names, strict=True)
        ]
    return bodies, phases, False, coefs


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
    from archytaszx.semantics.prove import InductionSide, induction_hypothesis, induction_offset

    assert isinstance(theorem, Theorem)
    inf = theorem.inference
    if inf.kind not in ("induction", "induction_offset") or inf.data.get("window") != "cut":
        raise PathSumUnsupported("not an induction settled through a chain cut region")
    data = inf.data
    step = str(data["step"])
    offset = cast("str | None", data.get("offset"))
    vals = dict(values or {})
    exposed = cast(Diagram, data["exposed"])
    own_k, other_k = cast(Diagram, data["own_k"]), cast(Diagram, data["other_k"])
    region = cast("frozenset[NodeId]", data["region"])
    copy_own = frozenset(exposed.nodes) - region
    own_d, other_d = (
        (theorem.lhs, theorem.rhs)
        if data["side"] is InductionSide.START
        else (theorem.rhs, theorem.lhs)
    )
    index = str(data["index"])
    families = (own_k, other_k)
    other_s = other_k
    if offset is not None:
        sym = sp.Symbol(offset, integer=True, nonnegative=True)
        families = (
            induction_hypothesis(own_d, index, step),
            induction_hypothesis(other_d, index, step),
        )
        other_s = induction_offset(families[1], step, sym)
    succ_x = rename_multiplicity(other_s, step, Mult(step) + 1)
    peeled = _peel_step_index(succ_x, step)
    if peeled is None:
        raise PathSumUnsupported("the other side's chain box does not peel")
    exposed_x, copy_oth = peeled
    sp_o = _split(exposed, own_k, copy_own, vals, step, offset)
    sp_x = _split(exposed_x, other_k, copy_oth, vals, step, offset)
    if (sp_o.base_cut, sp_o.copy_in, sp_o.copy_out) != (sp_x.base_cut, sp_x.copy_in, sp_x.copy_out):
        raise PathSumUnsupported("the two sides' cut and link positions differ")
    period = math.lcm(len(sp_o.copies), len(sp_x.copies))
    pairs = [(sp_o.base, sp_x.base)] + [
        (sp_o.copies[j % len(sp_o.copies)], sp_x.copies[j % len(sp_x.copies)])
        for j in range(period)
    ]
    names = ["baseEq", *("copyEq" for _ in range(period))]
    bodies, phases, qudit, coefs = _bodies(pairs, names)
    base, copies = bodies[0], bodies[1:]
    mixed = False
    if qudit:
        dims = str(base.m)
        labs: dict[int, int] = {}
    else:
        labs = {k: v for body in bodies for k, v in body.labs.items()}
        ks = set(labs.values())
        mixed = len(ks) > 1
        dims = "2 ^ k" if mixed else f"2 ^ {ks.pop() if ks else 1}"
    cut = [c + (_BASE if mixed else 0) for c in sp_o.base_cut]
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
    sc = _scalars()

    def model(diagram: Diagram) -> sp.Expr:
        if qudit:
            out = sc.model(diagram.scalar.to_sympy())
            if sc.opaques:
                raise PathSumUnsupported(f"the scalar {diagram.scalar} has no exact Lean form")
            return out
        return _const_model(*constant(diagram.scalar.to_sympy()))

    def links() -> list[str]:
        models = tuple((model(sp_.base), [model(c) for c in sp_.copies]) for sp_ in (sp_o, sp_x))
        return _link_lines(
            cut,
            bodies,
            pis,
            phases,
            qudit,
            base.m,
            families,
            step,
            vals,
            checks,
            models,
            model,
            offset is not None,
            mixed,
        )

    if mixed:
        return _export_mixed(
            name, side, other, bodies, phases, coefs, cut, pis, offset is not None, links()
        )
    ns = f"ArchytasZX.ChainProof.{name}"
    sem = "QSem" if qudit else "PSem"
    lines = [
        "import ArchytasZX.PathSumQuditNet" if qudit else "import ArchytasZX.PathSumNetW",
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
        *net_env(phases, labs, dims, base.m, coefs),
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
    ]
    if offset is None:
        lines += [
            f"/-- The {side} and {other} families agree at every `k`. -/",
            "theorem main : ∀ k, Equal E",
            f"    (({base.s_l}) * ∏ i ∈ Finset.range k, aO i) (chainNet W cut Base.NL cO k)",
            f"    (({base.s_r}) * ∏ i ∈ Finset.range k, aX i) (chainNet W cut Base.NR cX k) :=",
            "  chain_equal E W cut _ _ _ _ hD (by decide) hcO (by decide) hcX _ _ _ _",
            "    Base.baseEq copyEq",
            "",
        ]
    else:
        lines += [
            f"/-- The {side} and {other} families agree at every `k` with copy index read",
            "as `index + s` for every `s`: level `j` of `k` carries copy `s + k - 1 - j`. -/",
            "theorem main : ∀ s k, Equal E",
            f"    (({base.s_l}) * ∏ i ∈ Finset.range k, aO (s + k - 1 - i))",
            "    (chainNet W cut Base.NL (fun j => cO (s + k - 1 - j)) k)",
            f"    (({base.s_r}) * ∏ i ∈ Finset.range k, aX (s + k - 1 - i))",
            "    (chainNet W cut Base.NR (fun j => cX (s + k - 1 - j)) k) :=",
            "  fun s k => chain_equal E W cut _ (fun j => cO (s + k - 1 - j)) _",
            "    (fun j => cX (s + k - 1 - j)) hD (by decide) (fun _ => hcO _) (by decide)",
            "    (fun _ => hcX _) _ _ (fun j => aO (s + k - 1 - j)) (fun j => aX (s + k - 1 - j))",
            "    Base.baseEq (fun _ => copyEq _) k",
            "",
        ]
    lines += [*links(), f"end {ns}", ""]
    return "\n".join(lines)


def _labels(net: NetT) -> set[int]:
    """Every label of ``net``."""
    return {x for nd in net[0] for x in (*nd[1], *nd[2])} | set(net[1]) | set(net[2])


def _export_mixed(
    name: str,
    side: str,
    other: str,
    bodies: Sequence[NetBody],
    phases: Sequence[tuple[str, int]],
    coefs: Mapping[int, str],
    cut: Sequence[int],
    pis: Sequence[Sequence[tuple[int, int]]],
    offset: bool,
    links: Sequence[str],
) -> str:
    """:func:`export_chain`'s Lean source over mixed qubit wire dimensions
    (ArchytasZX/ChainMixed.lean): each body proved under its own labels' digit counts, the
    base networks shifted by ``_BASE``, the copies relabelled onto the chain labels,
    ``chain_equal_lv`` over ``lvEnv E W C B k`` (C the copy labels in ``[W, 2 W)``, B the base
    labels) for every ``k``, then ``links``."""
    base, copies = bodies[0], bodies[1:]
    period = len(copies)
    ns = f"ArchytasZX.ChainProof.{name}"
    for body in bodies:
        if any(x >= _BASE for x in body.labs):
            raise PathSumUnsupported("a body label reaches the base shift")
        for tag in ("L", "R"):
            if not _labels(_parse_net(body.lines, tag)) <= set(body.labs):
                raise PathSumUnsupported("a network label has no digit count")
    b_labs = sorted((x + _BASE, d) for x, d in base.labs.items())
    c_labs: dict[int, int] = {}
    c_pis = []
    for body, pi in zip(copies, pis, strict=True):
        moved = sorted((_ren(pi, x), d) for x, d in body.labs.items())
        c_pis.append(moved)
        for x, d in moved:
            if x >= W and c_labs.setdefault(x, d) != d:
                raise PathSumUnsupported("the copies give one chain label two dimensions")
    lines = [
        "import ArchytasZX.PathSumNetW",
        "import ArchytasZX.ChainMixed",
        "",
        "/-!",
        f"# Chain induction `{name}` over mixed wire dimensions",
        "",
        f"The {side} side (peeled) and the {other} side as chain families: each network pair",
        "proved equal by path sums under its own labels, the base shifted and the copies",
        "relabelled onto the chain labels, and `chain_equal_lv` for every `k`.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        "open ArchytasZX ArchytasZX.PSem",
        "",
        *NET_OPTIONS,
        "",
        *net_env(phases, {}, "2 ^ k", base.m, coefs),
    ]
    for tag, body in [("Base", base), *((f"Copy{j}", b) for j, b in enumerate(copies))]:
        lines += [
            f"namespace {tag}",
            "",
            f"def labsN : List (ℕ × ℕ) := {sorted(body.labs.items())}",
            f"noncomputable def E : Env := {{ {ns}.E with labs := labsN }}",
            "",
            *body.lines,
            f"end {tag}",
            "",
        ]
    lines += [
        f"def W : ℕ := {W}",
        f"def OB : ℕ := {_BASE}",
        f"def cut : List ℕ := {list(cut)}",
        f"def C : List (ℕ × ℕ) := {sorted(c_labs.items())}",
        f"def B : List (ℕ × ℕ) := {b_labs}",
        *(f"def πC{j} : List (ℕ × ℕ) := {pi}" for j, pi in enumerate(pis)),
        *(f"def Cπ{j} : List (ℕ × ℕ) := {m}" for j, m in enumerate(c_pis)),
        "",
        "def bO : Net := Base.NL.mapL (· + OB)",
        "def bX : Net := Base.NR.mapL (· + OB)",
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
        "theorem hC : ∀ p ∈ C, W ≤ p.1 ∧ p.1 < 2 * W := by decide",
        "theorem hB : ∀ p ∈ B, p.1 < W := by decide",
        "theorem hcut : ∀ x ∈ cut, x < W ∧",
        "    labDim (lvEnv E W C B 1) (x + W) = labDim (lvEnv E W C B 1) x := by decide",
        "",
        f"theorem baseEq : Equal (lvEnv E W C B 1) ({base.s_l}) bO ({base.s_r}) bX :=",
        "  ((Base.baseEq.relabs (B ++ Base.labsN) (by decide) (by decide)).mapL_on _ (· + OB)",
        "    (fun x y h => by simpa using h) (by decide)).relabs (lvls W C 1 ++ B) (by decide)",
        "    (by decide)",
        "",
    ]
    for j, body in enumerate(copies):
        lines += [
            (
                f"theorem copyEq{j} : Equal (lvEnv E W C B 1) ({body.s_l}) "
                f"(Copy{j}.NL.mapL (ren πC{j})) ({body.s_r}) (Copy{j}.NR.mapL (ren πC{j})) :="
            ),
            f"  ((Copy{j}.copyEq.relabs (Cπ{j} ++ Copy{j}.labsN) (by decide) (by decide)).mapL_on",
            f"    _ (ren πC{j}) (ren_injective πC{j} (by decide) (by decide)) (by decide)).relabs",
            "    (lvls W C 1 ++ B) (by decide) (by decide)",
            "",
        ]
    lines += [
        "theorem copyEq : ∀ i, Equal (lvEnv E W C B 1) (aO i) (cO i) (aX i) (cX i) := by",
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
    ]
    if not offset:
        lines += [
            f"/-- The {side} and {other} families agree at every `k`. -/",
            "theorem main : ∀ k, Equal (lvEnv E W C B k)",
            f"    (({base.s_l}) * ∏ i ∈ Finset.range k, aO i) (chainNet W cut bO cO k)",
            f"    (({base.s_r}) * ∏ i ∈ Finset.range k, aX i) (chainNet W cut bX cX k) :=",
            "  chain_equal_lv E W cut bO cO bX cX C B hC hB hcut (by decide) hcO (by decide) hcX",
            "    _ _ _ _ baseEq copyEq",
            "",
        ]
    else:
        lines += [
            f"/-- The {side} and {other} families agree at every `k` with copy index read",
            "as `index + s` for every `s`: level `j` of `k` carries copy `s + k - 1 - j`. -/",
            "theorem main : ∀ s k, Equal (lvEnv E W C B k)",
            f"    (({base.s_l}) * ∏ i ∈ Finset.range k, aO (s + k - 1 - i))",
            "    (chainNet W cut bO (fun j => cO (s + k - 1 - j)) k)",
            f"    (({base.s_r}) * ∏ i ∈ Finset.range k, aX (s + k - 1 - i))",
            "    (chainNet W cut bX (fun j => cX (s + k - 1 - j)) k) :=",
            "  fun s k => chain_equal_lv E W cut bO (fun j => cO (s + k - 1 - j)) bX",
            "    (fun j => cX (s + k - 1 - j)) C B hC hB hcut (by decide) (fun _ => hcO _)",
            "    (by decide) (fun _ => hcX _) _ _ (fun j => aO (s + k - 1 - j))",
            "    (fun j => aX (s + k - 1 - j)) baseEq (fun _ => copyEq _) k",
            "",
        ]
    lines += [*links, f"end {ns}", ""]
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
    offset: bool = False,
    mixed: bool = False,
) -> list[str]:
    """For each ``k`` of ``checks``: both sides' instances at ``k`` as networks DL, DR, the
    relabellings of the chain networks onto them, their node, summed and boundary permutations
    by ``decide``, instK (the two instances equal under the chain scalars), each side's chain
    scalar at ``k`` equal to its instance's (scalL, scalR) and inst (the two instances equal
    under their own scalars); with ``offset``, the chain networks at offset 0; with ``mixed``,
    the base shifted by ``_BASE``, DR's wires labelled from ``1 << 17`` and both instances
    over ``EK`` (their labels' digit counts, then ``lvEnv E W C B k``'s)."""
    from archytaszx.semantics.check import instantiate

    sc = _scalars()

    base, copies = bodies[0], bodies[1:]

    def body_of(ph: int) -> str | None:
        return phases[ph - 1][0] if ph else None

    out: list[str] = []
    for k in checks:
        texts: list[tuple[str, list[tuple[int, int]], str]] = []
        scal: list[tuple[str, str, str, sp.Expr]] = []
        digits: dict[int, int] = {}
        for (tag, fam), (m_base, m_copies) in zip(
            zip(("L", "R"), families, strict=True), models, strict=True
        ):
            base_net = _parse_net(base.lines, tag)
            if mixed:
                base_net = _map_net(base_net, functools.partial(int.__add__, _BASE))
            chain = _chain_net(
                cut,
                base_net,
                [
                    _map_net(_parse_net(body.lines, tag), functools.partial(_ren, pi))
                    for body, pi in zip(copies, pis, strict=True)
                ],
                k,
                offset,
            )
            inst_d = instantiate(fam, {step: k, **values})
            wire_base = 1 << 17 if mixed and tag == "R" else None
            inst, summed, nb = _instance_net(inst_d, phases, qudit, m, wire_base)
            for _kd, o, i, d, _b in inst:
                for x in (*o, *i):
                    if digits.setdefault(x, d) != d:
                        raise PathSumUnsupported("an instance label has two dimensions")
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

        def family_of(c: str, k: int = k) -> str:
            return f"(fun j => {c} (0 + {k} - 1 - j))" if offset else c

        def arg(a: str, k: int = k) -> str:
            return f"{a} (0 + {k} - 1 - i)" if offset else f"{a} i"

        main_k = f"main 0 {k}" if offset else f"main {k}"
        bases = ("bO", "bX") if mixed else ("Base.NL", "Base.NR")
        for tag, fam_net in (
            ("L", f"{bases[0]} {family_of('cO')}"),
            ("R", f"{bases[1]} {family_of('cX')}"),
        ):
            chain_k = f"(chainNet W cut {fam_net} {k}).mapL (ren π{tag})"
            out += [
                f"theorem link{tag} : ({chain_k}).nodes.Perm D{tag}.nodes ∧",
                f"    ({chain_k}).int.Perm D{tag}.int ∧ ({chain_k}).bnd.Perm D{tag}.bnd := by",
                "  decide",
                "",
            ]
        env = "EK" if mixed else "E"
        if mixed:
            out += [
                f"def instLabs : List (ℕ × ℕ) := {sorted(digits.items())}",
                f"noncomputable def EK : Env := {{ E with labs := instLabs ++ lvls W C {k} ++ B }}",
                "",
            ]
            mapped = [
                f"  have m := ({main_k}).relabs (instLabs ++ lvls W C {k} ++ B) (by decide)",
                "    (by decide)",
                "  have h := Equal.mapL2_on EK (ren πL) (ren πR)",
                "    (ren_injective πL (by decide) (by decide))",
                "    (ren_injective πR (by decide) (by decide)) (by decide) (by decide)",
                "    m (by decide) (by decide)",
            ]
        else:
            mapped = [
                "  have h := Equal.mapL2 E (ren πL) (ren πR)",
                "    (ren_injective πL (by decide) (by decide))",
                "    (ren_injective πR (by decide) (by decide)) (fun _ => hD _ _)",
                f"    (fun _ => hD _ _) ({main_k}) (by decide) (by decide)",
            ]
        out += [
            f"/-- Both sides instantiated at `k = {k}` are equal. -/",
            f"theorem instK : Equal {env}",
            f"    (({base.s_l}) * ∏ i ∈ Finset.range {k}, {arg('aO')}) DL",
            f"    (({base.s_r}) * ∏ i ∈ Finset.range {k}, {arg('aX')}) DR := by",
            *mapped,
            f"  exact (h.reorderL {env} linkL.1 linkL.2.1 linkL.2.2).reorder {env} linkR.1",
            "    linkR.2.1 linkR.2.2",
            "",
        ]
        for (tag, rhs_text, proof, lhs), (a_tab, b_text) in zip(
            scal, (("aO", base.s_l), ("aX", base.s_r)), strict=True
        ):
            out += [
                f"theorem scal{tag} : ({b_text}) * ∏ i ∈ Finset.range {k}, {arg(a_tab)} =",
                f"    {rhs_text} := by",
                (
                    f"  simp only [Finset.prod_range_succ, Finset.prod_range_zero, {a_tab}, "
                    f"perIdx, {a_tab}t, List.length_cons, List.length_nil, Nat.reduceAdd, "
                    "Nat.reduceMod, List.getD_cons_zero, List.getD_cons_succ, one_mul"
                    + (", Nat.reduceSub, zero_add]" if offset else "]")
                ),
                f"  have key : ({sc.text(lhs)} : ℂ) = {rhs_text} :=",
                *("    " + ln for ln in proof.splitlines()),
                "  all_goals linear_combination key",
                "",
            ]
        out += [
            f"/-- Both sides instantiated at `k = {k}` are equal, under their own scalars. -/",
            f"theorem inst : Equal {env} ({scal[0][1]}) DL ({scal[1][1]}) DR :=",
            "  instK.congr_scalar scalL scalR",
            "",
            f"end At{k}",
            "",
        ]
    return out


def _tree_net(r: int, p: Callable[[int], int], base: NetT, copies: Sequence[NetT], n: int) -> NetT:
    """Lean's ``treeNet W r p base (perIdx copies) n``."""
    net = base
    for t in range(n):

        def plug_map(x: int, t: int = t) -> int:
            y = x + t * W
            return p(t) if y == r + t * W else r + t * W if y == p(t) else y

        c_nodes, c_int, c_bnd = _map_net(copies[t % len(copies)], plug_map)
        net = (
            c_nodes + net[0],
            c_int + [p(t)] + net[1],
            [x for x in net[2] if x != p(t)] + [x for x in c_bnd if x != p(t)],
        )
    return net


def _heap_plug(leaf: int, kids: Sequence[int]) -> Callable[[int], int]:
    """Lean's ``heapPlug W leaf kids``."""

    def p(t: int) -> int:
        return leaf if t == 0 else kids[(t - 1) % len(kids)] + (t - 1) // len(kids) * W

    return p


def _grow_tree(
    base: Diagram, copies: Sequence[Diagram], leaf: int, root: int, kids: Sequence[int], n: int
) -> Diagram:
    """``base`` with ``n`` copies plugged in heap order: copy 0's boundary port ``root`` wired to
    the base's boundary port ``leaf``, copy ``t + 1``'s to child ``kids[t mod b]`` of copy
    ``t / b``; the boundary the unplugged ports in order (outputs, then inputs), the scalar the
    product."""
    out = base.copy()
    bnd = [*base.boundary_outputs, *base.boundary_inputs]
    plugged: set[PortRef] = set()
    placed: list[list[PortRef]] = []
    open_child: list[PortRef] = []
    scalar = base.scalar.to_sympy()
    for t in range(n):
        piece = copies[t % len(copies)]
        id_map: dict[NodeId, NodeId] = {}
        for nid in sorted(piece.nodes):
            node = piece.nodes[nid]
            id_map[nid] = out.add_node(
                node.generator_type,
                [q.dim for q in node.inputs],
                [q.dim for q in node.outputs],
                phase=node.phase,
            )
        for wire in sorted(piece.wires, key=lambda w: w.sort_key()):
            out.add_wire(_moved(wire.a, id_map), _moved(wire.b, id_map))
        p_bnd = [_moved(q, id_map) for q in (*piece.boundary_outputs, *piece.boundary_inputs)]
        target = bnd[leaf] if t == 0 else placed[(t - 1) // len(kids)][(t - 1) % len(kids)]
        out.add_wire(target, p_bnd[root])
        plugged |= {target, p_bnd[root]}
        placed.append([p_bnd[j] for j in kids])
        open_child += [q for j, q in enumerate(p_bnd) if j != root]
        scalar *= piece.scalar.to_sympy()
    ports = [q for q in (*bnd, *open_child) if q not in plugged]
    out.set_boundary_outputs([q for q in ports if q.direction is Direction.OUTPUT])
    out.set_boundary_inputs([q for q in ports if q.direction is Direction.INPUT])
    out.set_scalar(Scalar(scalar))
    return out


def _reorder(diagram: Diagram, order: Sequence[int]) -> Diagram:
    """``diagram`` with boundary position ``i`` moved to ``order[i]``; raises
    :class:`PathSumUnsupported` when an output would land among the inputs or back."""
    outs = list(diagram.boundary_outputs)
    bnd = [*outs, *diagram.boundary_inputs]
    new: list[PortRef | None] = [None] * len(bnd)
    for i, j in enumerate(order):
        new[j] = bnd[i]
    if any(
        (ref is None) or ((j < len(outs)) != (ref.direction is Direction.OUTPUT))
        for j, ref in enumerate(new)
    ):
        raise PathSumUnsupported("the fused tree's boundary does not match the instance's")
    out = diagram.copy()
    refs = cast("list[PortRef]", new)
    out.set_boundary_outputs(refs[: len(outs)])
    out.set_boundary_inputs(refs[len(outs) :])
    return out


def _tree_pis(
    cbodies: Sequence[NetBody], copies: Sequence[tuple[Diagram, Diagram]], root: int
) -> list[list[tuple[int, int]]]:
    """Each copy body's relabelling onto the tree labels: its root port to 0, boundary port
    ``j`` to ``W + _FREE + j``, internal label ``x`` to ``W + x``."""
    first = copies[0][0]
    n_copy = len(first.boundary_outputs) + len(first.boundary_inputs)
    target = {j: W + _FREE + j for j in range(n_copy)}
    target[root] = 0
    pis = []
    for body in cbodies:
        tgt = dict(target)
        internal = [x for x in body.labs if x >= n_copy]
        for x in internal:
            tgt[x] = W + x
        pis.append(_relabel(set(range(n_copy)) | set(internal), tgt))
    return pis


def _tree_nets(
    bodies: Sequence[NetBody],
    copies: Sequence[tuple[Diagram, Diagram]],
    leaf: int,
    root: int,
    kids: Sequence[int],
    n: int,
) -> dict[str, NetT]:
    """Both families' ``treeNet`` after ``n`` copies, by tag L, R."""
    base, cbodies = bodies[0], bodies[1 : 1 + len(copies)]
    pis = _tree_pis(cbodies, copies, root)
    p = _heap_plug(leaf, [W + _FREE + j for j in kids])
    return {
        tag: _tree_net(
            0,
            p,
            _parse_net(base.lines, tag),
            [
                _map_net(_parse_net(b.lines, tag), functools.partial(_ren, pi))
                for b, pi in zip(cbodies, pis, strict=True)
            ],
            n,
        )
        for tag in ("L", "R")
    }


def _tree_count(b: int, depth: int) -> int:
    """The copies of a tree of branching ``b`` and depth ``depth``."""
    return sum(b**j for j in range(depth))


def _fused_tree(
    bodies: Sequence[NetBody],
    phases: Sequence[tuple[str, int]],
    qudit: bool,
    bases: tuple[Diagram, Diagram],
    copies: Sequence[tuple[Diagram, Diagram]],
    leaf: int,
    root: int,
    kids: Sequence[int],
    chk: TreeCheck,
) -> Diagram:
    """The fan side's base with its copies grown to ``chk.depth``, its boundary ordered as the
    tree side's instance orders the family's boundary labels."""

    def body_of(ph: int) -> str | None:
        return phases[ph - 1][0] if ph else None

    n = _tree_count(len(kids), chk.depth)
    nets = _tree_nets(bodies, copies, leaf, root, kids, n)
    tree_tag = "L" if chk.fan == "R" else "R"
    tree_inst = chk.left if tree_tag == "L" else chk.right
    inst, _summed, nb = _instance_net(tree_inst, phases, qudit, bodies[0].m)
    fwd_t, _ = _iso(nets[tree_tag], inst, body_of)
    side = 0 if chk.fan == "L" else 1
    grown = _grow_tree(bases[side], [c[side] for c in copies], leaf, root, kids, n)
    g_inst, _gs, g_nb = _instance_net(grown, phases, qudit, bodies[0].m)
    fwd_g, _ = _iso(nets[chk.fan], g_inst, body_of)
    if g_nb != nb or sorted(fwd_t[x] for x in nets[tree_tag][2]) != list(range(nb)):
        raise PathSumUnsupported("the tree family's boundary is not the instance's")
    back_g = {fwd_g[x]: x for x in nets[chk.fan][2]}
    return _reorder(grown, [fwd_t[back_g[q]] for q in range(nb)])


@dataclass
class TreeCheck:
    """A tree family pair linked at ``depth``: both sides instantiated there, ``fan`` (L or R)
    the side whose instance is the fused tree, the other's being the tree itself."""

    depth: int
    left: Diagram
    right: Diagram
    fan: str


def export_tree(
    bases: tuple[Diagram, Diagram],
    copies: Sequence[tuple[Diagram, Diagram]],
    leaf: int,
    root: int,
    kids: Sequence[int],
    name: str = "main",
    checks: Sequence[TreeCheck] = (),
) -> str:
    """Lean 4 source proving two tree families equal after every number of copies
    (ArchytasZX/TreeInduct.lean): ``bases`` the two networks before any copy, boundary position
    ``leaf`` the first plug; ``copies`` one period of copy pairs, boundary position ``root`` the
    copy's parent port and ``kids`` its child ports; copy ``n`` plugged in heap order (onto
    child ``(n - 1) mod b`` of copy ``(n - 1) / b``), so depth ``k`` is ``(b^k - 1) / (b - 1)``
    copies; each pair proved equal by its path sums and ``tree_equal`` for every count. At each
    of ``checks``, ``At{k}.inst`` proves the two instances equal: the tree side's instance a
    relabelling and permutation of its family at depth ``k``, the fan side's family relabelled
    onto the grown tree, which equals the fan instance by its path sums (fusion); raises
    :class:`PathSumUnsupported` outside that class."""
    names = ["baseEq", *("copyEq" for _ in copies)]
    pairs = [(bases[0], bases[1]), *copies]
    bodies, phases, qudit, coefs = _bodies(pairs, names)
    base, cbodies = bodies[0], bodies[1:]
    grown = [
        (chk, _fused_tree(bodies, phases, qudit, bases, copies, leaf, root, kids, chk))
        for chk in checks
    ]
    if grown:
        pairs += [(g, chk.right if chk.fan == "R" else chk.left) for chk, g in grown]
        names += ["fuseEq" for _ in grown]
        bodies, phases, qudit, coefs = _bodies(pairs, names)
        base, cbodies = bodies[0], bodies[1 : 1 + len(copies)]
    fusions = bodies[1 + len(copies) :]
    if qudit:
        dims, labs = str(base.m), {}
    else:
        labs = {k: v for body in bodies for k, v in body.labs.items()}
        ks = set(labs.values())
        if len(ks) > 1:
            raise PathSumUnsupported("the tree export needs one wire dimension")
        dims = f"2 ^ {ks.pop() if ks else 1}"
    r = 0
    pis = _tree_pis(cbodies, copies, root)
    period = len(copies)
    ns = f"ArchytasZX.TreeProof.{name}"
    sem = "QSem" if qudit else "PSem"
    lines = [
        "import ArchytasZX.PathSumQuditNet" if qudit else "import ArchytasZX.PathSumNetW",
        "import ArchytasZX.TreeInduct",
        "",
        "/-!",
        f"# Tree families `{name}`",
        "",
        "Both sides' networks before any copy and one period of copies proved equal by path sums,",
        "the copies relabelled onto the tree labels, and `tree_equal` for every number of copies",
        "in heap order.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        f"open ArchytasZX ArchytasZX.{sem}",
        "",
        *(QNET_OPTIONS if qudit else NET_OPTIONS),
        "",
        *([f"instance : Fact (Nat.Prime {base.m}) := ⟨by norm_num⟩", ""] if qudit else []),
        *net_env(phases, labs, dims, base.m, coefs),
        "theorem hD : ∀ x y, labDim E x = labDim E y := fun _ _ => rfl",
        "",
        "namespace Base",
        "",
        *base.lines,
        "end Base",
        "",
    ]
    for j, body in enumerate(cbodies):
        lines += [f"namespace Copy{j}", "", *body.lines, f"end Copy{j}", ""]
    lines += [
        f"def W : ℕ := {W}",
        f"def r : ℕ := {r}",
        f"def leaf0 : ℕ := {leaf}",
        f"def kids : List ℕ := {[W + _FREE + j for j in kids]}",
        "def p : ℕ → ℕ := heapPlug W leaf0 kids",
        *(f"def πC{j} : List (ℕ × ℕ) := {pi}" for j, pi in enumerate(pis)),
        "",
        "def cOt : List Net := ["
        + ", ".join(f"Copy{j}.NL.mapL (ren πC{j})" for j in range(period))
        + "]",
        "def cXt : List Net := ["
        + ", ".join(f"Copy{j}.NR.mapL (ren πC{j})" for j in range(period))
        + "]",
        "noncomputable def aOt : List ℂ := [" + ", ".join(f"({b.s_l})" for b in cbodies) + "]",
        "noncomputable def aXt : List ℂ := [" + ", ".join(f"({b.s_r})" for b in cbodies) + "]",
        "",
        "def cO (i : ℕ) : Net := perIdx cOt ⟨[], [], []⟩ i",
        "def cX (i : ℕ) : Net := perIdx cXt ⟨[], [], []⟩ i",
        "noncomputable def aO (i : ℕ) : ℂ := perIdx aOt 1 i",
        "noncomputable def aX (i : ℕ) : ℂ := perIdx aXt 1 i",
        "",
    ]
    for j, body in enumerate(cbodies):
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
        "theorem hcO : ∀ i, tcopyOK W r (cO i) = true :=",
        "  perIdx_all cOt _ (fun c => tcopyOK W r c = true) (by decide) (by decide)",
        "theorem hcX : ∀ i, tcopyOK W r (cX i) = true :=",
        "  perIdx_all cXt _ (fun c => tcopyOK W r c = true) (by decide) (by decide)",
        "theorem hpO : PlugOK W p Base.NL cO :=",
        "  heap_plugOK W r leaf0 kids Base.NL cOt _ (by decide) hcO (by decide)",
        "theorem hpX : PlugOK W p Base.NR cX :=",
        "  heap_plugOK W r leaf0 kids Base.NR cXt _ (by decide) hcX (by decide)",
        "",
        "/-- Both tree families agree after every number of copies. -/",
        "theorem main : ∀ n, Equal E",
        f"    (({base.s_l}) * ∏ i ∈ Finset.range n, aO i) (treeNet W r p Base.NL cO n)",
        f"    (({base.s_r}) * ∏ i ∈ Finset.range n, aX i) (treeNet W r p Base.NR cX n) :=",
        "  tree_equal E W r p _ _ _ _ hD (by decide) hcO (by decide) hcX hpO hpX _ _ _ _",
        "    Base.baseEq copyEq",
        "",
    ]
    for (chk, grown_d), fusion in zip(grown, fusions, strict=True):
        lines += _tree_link(
            chk, grown_d, fusion, bodies, bases, copies, phases, qudit, leaf, root, kids
        )
    lines += [f"end {ns}", ""]
    return "\n".join(lines)


def _scalar_model(qudit: bool) -> Callable[[Diagram], sp.Expr]:
    """A diagram's scalar as a Lean scalar model."""
    sc = _scalars()

    def model(diagram: Diagram) -> sp.Expr:
        if qudit:
            out = sc.model(diagram.scalar.to_sympy())
            if sc.opaques:
                raise PathSumUnsupported(f"the scalar {diagram.scalar} has no exact Lean form")
            return out
        return _const_model(*constant(diagram.scalar.to_sympy()))

    return model


def _tree_link(
    chk: TreeCheck,
    grown: Diagram,
    fusion: NetBody,
    bodies: Sequence[NetBody],
    bases: tuple[Diagram, Diagram],
    copies: Sequence[tuple[Diagram, Diagram]],
    phases: Sequence[tuple[str, int]],
    qudit: bool,
    leaf: int,
    root: int,
    kids: Sequence[int],
) -> list[str]:
    """``At{depth}``: the tree side's instance D, the fusion body (the grown fan tree NL equal
    to the fan instance NR), each family at depth relabelled onto D or NL with its node, summed
    and boundary permutations by ``decide``, instK (D and NL under the family scalars), the
    family scalars equal to D's and the grown tree's, and inst (the two instances equal under
    their own scalars)."""
    sc = _scalars()
    model = _scalar_model(qudit)
    base = bodies[0]

    def body_of(ph: int) -> str | None:
        return phases[ph - 1][0] if ph else None

    n = _tree_count(len(kids), chk.depth)
    nets = _tree_nets(bodies, copies, leaf, root, kids, n)
    tree_tag = "L" if chk.fan == "R" else "R"
    tree_inst = chk.left if tree_tag == "L" else chk.right
    inst, summed, nb = _instance_net(tree_inst, phases, qudit, base.m)
    fwd_t, match_t = _iso(nets[tree_tag], inst, body_of)
    g_nodes, _g_int, g_bnd = _parse_net(fusion.lines, "L")
    if g_bnd != list(range(nb)):
        raise PathSumUnsupported("the grown tree's boundary is not in order")
    g_inst = [(kd, o, i, d, body_of(ph)) for kd, o, i, d, ph in g_nodes]
    seed = {x: fwd_t[x] for x in nets[chk.fan][2]}
    fwd_g, _ = _iso(nets[chk.fan], g_inst, body_of, seed)
    k = chk.depth
    out = [f"namespace At{k}", "", "namespace Fuse", "", *fusion.lines, "end Fuse", ""]
    chain = nets[tree_tag]
    labels = {x for nd in chain[0] for x in (*nd[1], *nd[2])} | set(chain[1]) | set(chain[2])
    nodes = [
        f"⟨{kd}, {list(o)}, {list(i)}, {d}, {chain[0][j][4]}⟩"
        for (kd, o, i, d, _b), j in zip(inst, match_t, strict=True)
    ]
    fan_net = nets[chk.fan]
    fan_labels = (
        {x for nd in fan_net[0] for x in (*nd[1], *nd[2])} | set(fan_net[1]) | set(fan_net[2])
    )
    pi = {tree_tag: _relabel(labels, fwd_t), chk.fan: _relabel(fan_labels, fwd_g)}
    target = {tree_tag: "D", chk.fan: "Fuse.NL"}
    out += [
        f"def π{tree_tag} : List (ℕ × ℕ) := {pi[tree_tag]}",
        f"def D : Net := ⟨[{', '.join(nodes)}], {summed}, {list(range(nb))}⟩",
        f"def π{chk.fan} : List (ℕ × ℕ) := {pi[chk.fan]}",
        "",
    ]
    for tag, fam_net in (("L", "Base.NL cO"), ("R", "Base.NR cX")):
        tree_k = f"(treeNet W r p {fam_net} {n}).mapL (ren π{tag})"
        out += [
            f"theorem link{tag} : ({tree_k}).nodes.Perm {target[tag]}.nodes ∧",
            f"    ({tree_k}).int.Perm {target[tag]}.int ∧",
            f"    ({tree_k}).bnd.Perm {target[tag]}.bnd := by",
            "  decide",
            "",
        ]
    out += [
        f"/-- The two families at depth {k} ({n} copies) on the instance and the grown tree. -/",
        f"theorem instK : Equal E (({base.s_l}) * ∏ i ∈ Finset.range {n}, aO i) {target['L']}",
        f"    (({base.s_r}) * ∏ i ∈ Finset.range {n}, aX i) {target['R']} := by",
        "  have h := Equal.mapL2 E (ren πL) (ren πR)",
        "    (ren_injective πL (by decide) (by decide))",
        "    (ren_injective πR (by decide) (by decide)) (fun _ => hD _ _) (fun _ => hD _ _)",
        f"    (main {n}) (by decide) (by decide)",
        "  exact (h.reorderL E linkL.1 linkL.2.1 linkL.2.2).reorder E linkR.1 linkR.2.1",
        "    linkR.2.2",
        "",
    ]
    side = {"L": 0, "R": 1}
    rhs_of = {tree_tag: sc.text(model(tree_inst)), chk.fan: fusion.s_l}
    rhs_model = {tree_tag: model(tree_inst), chk.fan: model(grown)}
    for tag, a_tab, b_text in (("L", "aO", base.s_l), ("R", "aX", base.s_r)):
        lhs_m = model(bases[side[tag]]) * sp.Mul(
            *(model(copies[t % len(copies)][side[tag]]) for t in range(n))
        )
        proof = sc.prove(lhs_m, rhs_model[tag])
        out += [
            f"theorem scal{tag} : ({b_text}) * ∏ i ∈ Finset.range {n}, {a_tab} i =",
            f"    {rhs_of[tag]} := by",
            (
                f"  simp only [Finset.prod_range_succ, Finset.prod_range_zero, {a_tab}, "
                f"perIdx, {a_tab}t, List.length_cons, List.length_nil, Nat.reduceAdd, "
                "Nat.reduceMod, List.getD_cons_zero, List.getD_cons_succ, one_mul]"
            ),
            f"  have key : ({sc.text(lhs_m)} : ℂ) = {sc.text(rhs_model[tag])} :=",
            *("    " + ln for ln in proof.splitlines()),
            "  all_goals linear_combination key",
            "",
        ]
    s_t, s_f = sc.text(model(tree_inst)), fusion.s_r
    linked = "instK.congr_scalar scalL scalR"
    if chk.fan == "R":
        stmt, proof_text = f"({s_t}) D ({s_f}) Fuse.NR", f"({linked}).trans Fuse.fuseEq"
    else:
        stmt, proof_text = f"({s_f}) Fuse.NR ({s_t}) D", f"Fuse.fuseEq.symm.trans ({linked})"
    out += [
        f"/-- Both sides instantiated at depth {k} are equal, under their own scalars. -/",
        f"theorem inst : Equal E {stmt} :=",
        f"  {proof_text}",
        "",
        f"end At{k}",
        "",
    ]
    return out


def _tree_side(d: Diagram, index: str, values: Mapping[str, int]) -> tuple[Diagram, Diagram, int]:
    """A tree-box diagram's network before any copy, its copy (boundary: the child ports, then
    the parent port) and the plug's boundary position in the first."""
    from archytaszx.semantics.check import instantiate

    boxes = list(d.bang_boxes.values())
    if len(boxes) != 1 or not boxes[0].is_tree or boxes[0].parent is not None:
        raise PathSumUnsupported("the tree side is not one top-level tree box")
    box = boxes[0]
    if box.multiplicity.free_symbols != {index} or box.copy_scalar is not None:
        raise PathSumUnsupported("the tree box has another multiplicity or a copy scalar")
    if box.level is not None or box.position is not None or box.index is not None:
        raise PathSumUnsupported("the tree box reads an index")
    if box.tree_parent is None or box.tree_parent.direction is not Direction.INPUT:
        raise PathSumUnsupported("the tree's parent port is no input")
    if any(r.direction is not Direction.OUTPUT for r in box.tree):
        raise PathSumUnsupported("a tree child port is no output")
    scope = box.node_scope
    if any(
        r.node_id in scope and r not in box.tree for r in (*d.boundary_outputs, *d.boundary_inputs)
    ):
        raise PathSumUnsupported("the copy has boundary ports off its child ports")
    base = instantiate(d, {index: 0, **values})
    one = instantiate(d, {index: 1, **values})
    if instantiate(d, {index: 2, **values}).scalar.to_sympy() != base.scalar.to_sympy():
        raise PathSumUnsupported("the tree family's scalar depends on the depth")
    copy_ids = frozenset(one.nodes) - frozenset(base.nodes)
    plug = [
        w.b if w.a.node_id in copy_ids else w.a
        for w in one.wires
        if (w.a.node_id in copy_ids) != (w.b.node_id in copy_ids)
    ]
    if len(plug) != 1:
        raise PathSumUnsupported("the copy meets the base off its parent port")
    bnd = [*base.boundary_outputs, *base.boundary_inputs]
    if plug[0] not in bnd:
        raise PathSumUnsupported("the plug is not on the base's boundary")
    root = next(
        r
        for w in one.wires
        for r in (w.a, w.b)
        if r.node_id in copy_ids and (w.a == plug[0] or w.b == plug[0])
    )
    kids = [r for r in (*one.boundary_outputs, *one.boundary_inputs) if r.node_id in copy_ids]
    if any(r.direction is not Direction.OUTPUT for r in kids) or root.direction is not (
        Direction.INPUT
    ):
        raise PathSumUnsupported("the copy's child ports are not its outputs")
    copy = _sub(one, copy_ids, kids, [root])
    return base, copy, bnd.index(plug[0])


def _fan_side(
    d: Diagram, index: str, values: Mapping[str, int], b: int
) -> tuple[Diagram, Diagram, int]:
    """A spider with one leg port-boxed ``b^index`` times as a tree: the spider with one copy of
    the leg, its copy a phase-free spider of the same kind with ``b`` output legs and one input
    leg, and the leg's boundary position."""
    from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
    from archytaszx.semantics.check import instantiate

    boxes = list(d.bang_boxes.values())
    if len(d.nodes) != 1 or len(boxes) != 1 or boxes[0].node_scope or d.wires:
        raise PathSumUnsupported("the other side is not one spider with one boxed leg")
    box = boxes[0]
    k = next((s for s in box.multiplicity.to_sympy().free_symbols if str(s) == index), None)
    if k is None or sp.simplify(box.multiplicity.to_sympy() - b**k) != 0:
        raise PathSumUnsupported(f"the boxed leg's multiplicity is not {b}^{index}")
    ((_, node),) = d.nodes.items()
    if node.generator_type.name not in (Z_SPIDER.name, X_SPIDER.name):
        raise PathSumUnsupported("the fanned node is not a Z or X spider")
    if len(box.port_scope) != 1:
        raise PathSumUnsupported("more than one boxed leg")
    (port,) = tuple(box.port_scope)
    base = instantiate(d, {index: 0, **values})
    ((bid, inst),) = base.nodes.items()
    bnd = [*base.boundary_outputs, *base.boundary_inputs]
    legs = [i for i, r in enumerate(bnd) if r.direction is port.direction and r.node_id == bid]
    if port.direction is not Direction.OUTPUT or len(legs) != 1:
        raise PathSumUnsupported("the spider has outputs besides the boxed leg")
    dim = inst.outputs[0].dim
    copy = Diagram()
    cid = copy.add_node(node.generator_type, [dim], [dim] * b)
    copy.set_boundary_outputs([PortRef(cid, Direction.OUTPUT, j) for j in range(b)])
    copy.set_boundary_inputs([PortRef(cid, Direction.INPUT, 0)])
    return base, copy, legs[0]


def export_tree_theorem(
    theorem: object,
    values: Mapping[str, int] | None = None,
    name: str = "main",
    checks: Sequence[int] = (2,),
) -> str:
    """:func:`export_tree` for a kernel induction through a tree window whose one side is a
    tree box and whose other side is one Z or X spider with a leg port-boxed ``b^n`` times
    (``b`` the tree's branching): the tree side split into its network before any copy and its
    copy, the other side into the spider with one leg and a phase-free fan spider with ``b``
    legs out, both sides instantiated at each depth of ``checks`` and linked; raises
    :class:`PathSumUnsupported` outside that class."""
    from archytaszx.semantics.check import instantiate
    from archytaszx.semantics.kernel import Theorem

    assert isinstance(theorem, Theorem)
    inf = theorem.inference
    if inf.kind != "induction" or inf.data.get("window") != "tree":
        raise PathSumUnsupported("not an induction settled through a tree window")
    index = str(inf.data["index"])
    vals = dict(values or {})
    sides = [theorem.lhs, theorem.rhs]
    trees = [i for i, s in enumerate(sides) if any(b.is_tree for b in s.bang_boxes.values())]
    if len(trees) != 1:
        raise PathSumUnsupported("not exactly one side is a tree box")
    t = trees[0]
    base_t, copy_t, leaf_t = _tree_side(sides[t], index, vals)
    b = len(copy_t.boundary_outputs) + len(copy_t.boundary_inputs) - 1
    base_f, copy_f, leaf_f = _fan_side(sides[1 - t], index, vals, b)
    if leaf_t != leaf_f:
        raise PathSumUnsupported("the two sides plug at different boundary positions")
    pairs = [(base_t, base_f), (copy_t, copy_f)] if t == 0 else [(base_f, base_t), (copy_f, copy_t)]
    links = [
        TreeCheck(
            k,
            instantiate(theorem.lhs, {index: k, **vals}),
            instantiate(theorem.rhs, {index: k, **vals}),
            "R" if t == 0 else "L",
        )
        for k in checks
    ]
    return export_tree(pairs[0], [pairs[1]], leaf_t, b, list(range(b)), name, links)


__all__ = ["export_chain", "export_tree", "export_tree_theorem"]
