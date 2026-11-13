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

"""Lean output for a periodic theorem (kernel.periodic). export_periodic writes the side holding
the chain box as ``famNet W u tl cp hd bd k`` (ArchytasZX/PeriodicChain.lean): ``tl`` the nodes
on the head side of the chain (and every node on neither side) with the boundary slots at
links whose head end is the boundary, ``cp`` the box's nodes, ``hd`` the nodes on the tail side,
link ``t`` labelled ``W + t``. The pairs at ``index = 0 .. p - 1`` and the pair (``p`` copies then
the tail side, the tail side times ``c``) are proved by their path sums (lean_pathsum.net_body)
and relabelled onto the templates; ``fam_periodic`` gives the two sides equal at every index,
the chain side's scalar the table of its scalars at ``0 .. p - 1`` times ``A`` per period (``A``
the scalar's own ratio), the other side's times ``A c``. At each index of ``checks`` both
sides instantiated are written out, matched to ``famNet`` and ``G`` by a label map found by
search, and proved equal under their own scalars (``At{k}.inst``)."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import cast

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.pathsum import PathSumUnsupported, constant
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.semantics.lean_chain import (
    _NODE_RE,
    NetT,
    _bodies,
    _instance_net,
    _iso,
    _map_net,
    _parse_net,
    _relabel,
    _ren,
    _sub,
)
from archytaszx.semantics.lean_pathsum import (
    NET_OPTIONS,
    NetBody,
    _const_model,
    _scalars,
    net_env,
)

W = 1 << 20
"""The label stride between chain levels."""

NodeT = tuple[str, tuple[int, ...], tuple[int, ...], int, int]
"""A Lean node: kind, outputs, inputs, dimension index, phase index."""


def _partner(diagram: Diagram, ref: PortRef) -> PortRef | None:
    """The port wired to ``ref``, or None."""
    for wire in diagram.wires:
        if wire.a == ref:
            return wire.b
        if wire.b == ref:
            return wire.a
    return None


def _reach(diagram: Diagram, seeds: set[NodeId], box: frozenset[NodeId]) -> set[NodeId]:
    """The nodes outside ``box`` connected to ``seeds`` by wires off ``box``."""
    seen = set(seeds)
    todo = list(seeds)
    while todo:
        nid = todo.pop()
        for wire in diagram.wires:
            for a, b in ((wire.a, wire.b), (wire.b, wire.a)):
                if a.node_id == nid and b.node_id not in box and b.node_id not in seen:
                    seen.add(b.node_id)
                    todo.append(b.node_id)
    return seen


def _node(
    diagram: Diagram, nid: NodeId, label: Mapping[PortRef, int], m: int, phase_of: Mapping[str, int]
) -> NodeT:
    """The Lean node of ``nid`` with its legs labelled by ``label``, its phase vector's index
    read from ``phase_of``."""
    from archytaszx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
    from archytaszx.semantics.pathsum_diagram import _bits, _turns_table

    names = {Z_SPIDER.name: ".Z", X_SPIDER.name: ".X", FOURIER_BOX.name: ".F"}
    node = diagram.nodes[nid]
    kind = names.get(node.generator_type.name)
    if kind is None:
        raise PathSumUnsupported(f"node {nid!r} is a {node.generator_type.name}")
    outs = tuple(label[PortRef(nid, Direction.OUTPUT, j)] for j in range(len(node.outputs)))
    ins = tuple(label[PortRef(nid, Direction.INPUT, j)] for j in range(len(node.inputs)))
    ref = PortRef(nid, Direction.OUTPUT, 0) if node.outputs else PortRef(nid, Direction.INPUT, 0)
    k = _bits(diagram, ref, 2)
    ph = 0
    if kind != ".F":
        ts = [int((t % 1) * m) for t in _turns_table(nid, node, 1 << k)]
        if any(ts):
            body = f"(({ts} : List ℕ).getD v 0 : ℕ)"
            if body not in phase_of:
                raise PathSumUnsupported(f"node {nid!r} has a phase vector no proof registers")
            ph = phase_of[body]
    return kind, outs, ins, k, ph


def _ports(diagram: Diagram, nid: NodeId) -> Iterator[PortRef]:
    """Every leg of ``nid``."""
    node = diagram.nodes[nid]
    for j in range(len(node.outputs)):
        yield PortRef(nid, Direction.OUTPUT, j)
    for j in range(len(node.inputs)):
        yield PortRef(nid, Direction.INPUT, j)


def _canon_phases(
    line: str, phases: Sequence[tuple[str, int]], first: Mapping[tuple[str, int], int]
) -> str:
    """``line`` with each node's phase index replaced by the first index of the same phase
    vector."""

    def sub(mt: re.Match[str]) -> str:
        ph = int(mt[5])
        if ph:
            ph = first[phases[ph - 1]]
        return f"⟨{mt[1]}, [{mt[2]}], [{mt[3]}], {mt[4]}, {ph}⟩"

    return _NODE_RE.sub(sub, line)


def _net_text(net: NetT) -> str:
    """Lean's ``Net`` literal of ``net``."""
    nodes = ", ".join(f"⟨{k}, {list(o)}, {list(i)}, {d}, {ph}⟩" for k, o, i, d, ph in net[0])
    return f"⟨[{nodes}], {net[1]}, {net[2]}⟩"


def _sh(k: int, x: int) -> int:
    """Lean's ``sh W k x``."""
    return x if x < W else x + k * W


def _fam(u: Sequence[int], tl: NetT, cp: NetT, hd: NetT, bd: Sequence[int], k: int) -> NetT:
    """Lean's ``famNet W u tl cp hd bd k``."""
    nodes = list(tl[0])
    summed = [*tl[1], *(x for x in u if x not in bd)]
    for j in range(k):
        nodes += _map_net(cp, functools.partial(_sh, j))[0]
        summed += [_sh(j, x) for x in cp[1]] + [_sh(j + 1, x) for x in u]
    nodes += _map_net(hd, functools.partial(_sh, k))[0]
    return nodes, [*summed, *hd[1]], list(bd)


def _blk(u: Sequence[int], cp: NetT, hd: NetT, p: int) -> NetT:
    """Lean's ``blkNet W u cp hd p``."""
    nodes: list[NodeT] = []
    summed: list[int] = []
    for j in range(p):
        nodes += _map_net(cp, functools.partial(_sh, j))[0]
        summed += [_sh(j, x) for x in cp[1]] + [_sh(j + 1, x) for x in u]
    nodes += _map_net(hd, functools.partial(_sh, p))[0]
    return nodes, [*summed, *hd[1]], [*u, *hd[2]]


def _labels(net: NetT) -> set[int]:
    """Every label of ``net``."""
    return {x for n in net[0] for x in (*n[1], *n[2])} | set(net[1]) | set(net[2])


def _pull(
    template: NetT, inst: NetT, seed: Mapping[int, int], body_of: Mapping[int, str | None]
) -> list[tuple[int, int]]:
    """A permutation list relabelling ``inst`` onto ``template``, the template's labels in
    ``seed`` sent to the given instance labels."""
    rows = [(k, o, i, d, body_of.get(ph)) for k, o, i, d, ph in inst[0]]
    fwd, _match = _iso(template, rows, lambda ph: body_of.get(ph), seed)
    back = {b: a for a, b in fwd.items()}
    return _relabel(_labels(inst), back)


def export_periodic(
    theorem: object,
    values: Mapping[str, int] | None = None,
    name: str = "main",
    checks: Sequence[int] = (2, 3),
) -> str:
    """Lean 4 source proving, for every value of the index, the two sides of a periodic theorem
    equal as ``famNet`` and a fixed network ``G`` (see the module docstring), and at each index
    of ``checks`` the two sides instantiated equal under their own scalars; raises
    :class:`PathSumUnsupported` outside that class."""
    from archytaszx.semantics.check import instantiate
    from archytaszx.semantics.decide import _instantiate_index
    from archytaszx.semantics.kernel import Theorem
    from archytaszx.semantics.periodic import chain_period, shift_ratio

    assert isinstance(theorem, Theorem)
    inf = theorem.inference
    if inf.kind != "periodic":
        raise PathSumUnsupported("not a periodic theorem")
    index = str(inf.data["index"])
    p = int(cast(int, inf.data["step"]))
    vals = dict(values or {})
    sides = [instantiate(d, vals) if vals else d for d in (theorem.lhs, theorem.rhs)]
    periods = [chain_period(d, index) for d in sides]
    if periods[0] is not None and periods[1] is None:
        flipped = False
    elif periods[0] is None and periods[1] is not None:
        flipped = True
    else:
        raise PathSumUnsupported("the export needs a chain on exactly one side")
    chain_d, goal_d = (sides[1], sides[0]) if flipped else (sides[0], sides[1])
    period, c_chain = cast("tuple[int, sp.Expr]", periods[1 if flipped else 0])
    box = next(b for b in chain_d.bang_boxes.values() if index in b.multiplicity.free_symbols)
    if box.copy_scalar is not None and sp.simplify(box.copy_scalar.to_sympy() - 1) != 0:
        raise PathSumUnsupported("the chain box carries a copy scalar")
    scope = frozenset(box.node_scope)
    links = list(box.links)
    outs_q = {q: t for t, (_p, q) in enumerate(links)}
    outs_p = {pp: t for t, (pp, _q) in enumerate(links)}
    bnd = [*chain_d.boundary_outputs, *chain_d.boundary_inputs]
    slot = {ref: i for i, ref in enumerate(bnd)}
    for nid in scope:
        for ref in _ports(chain_d, nid):
            other = _partner(chain_d, ref)
            if ref in outs_p and (other is None or other.node_id in scope):
                raise PathSumUnsupported("a link's tail end is not a node outside the box")
            if ref in outs_q or ref in outs_p:
                continue
            if ref in slot or (other is not None and other.node_id not in scope):
                raise PathSumUnsupported("a box node meets the outside off its links")
    head_seed = {o.node_id for q in outs_q if (o := _partner(chain_d, q)) is not None}
    tail_seed = {o.node_id for pp in outs_p if (o := _partner(chain_d, pp)) is not None}
    head, tail = _reach(chain_d, head_seed, scope), _reach(chain_d, tail_seed, scope)
    if head & tail:
        raise PathSumUnsupported("the two ends of the chain meet outside the box")
    head |= set(chain_d.nodes) - scope - tail

    def cut_ports(inst: Diagram) -> list[tuple[int, PortRef]]:
        """The ports of ``inst`` off ``head`` wired to ``head``, with their links."""
        found: list[tuple[int, PortRef]] = []
        for t, (_pp, q) in enumerate(links):
            h = _partner(chain_d, q)
            if h is None:
                continue
            o = _partner(inst, h)
            if o is None or o.node_id in head:
                raise PathSumUnsupported("a head end is not wired off the head side")
            found.append((t, o))
        return found

    def open_order(inst: Diagram) -> tuple[list[PortRef], list[PortRef], dict[int, int]]:
        """The outputs and inputs of ``inst`` without the head side (its own boundary ports,
        then its cut ports) and each one's template label's slot."""
        cuts = cut_ports(inst)
        full = [*inst.boundary_outputs, *inst.boundary_inputs]
        outs = [r for r in inst.boundary_outputs if r.node_id not in head]
        ins = [r for r in inst.boundary_inputs if r.node_id not in head]
        outs += [r for _t, r in cuts if r.direction is Direction.OUTPUT]
        ins += [r for _t, r in cuts if r.direction is Direction.INPUT]
        cut_of = {r: t for t, r in cuts}
        seed: dict[int, int] = {}
        for i, r in enumerate([*outs, *ins]):
            seed[W + cut_of[r] if r in cut_of else bd[full.index(r)]] = i
        return outs, ins, seed

    def open_part(inst: Diagram, scalar: sp.Expr) -> Diagram:
        """``inst`` without the head side, its cut ports on the boundary after its own."""
        outs, ins, _seed = open_order(inst)
        out = _sub(inst, frozenset(inst.nodes) - frozenset(head), outs, ins)
        out.set_scalar(Scalar(scalar))
        return out

    n_slots = len(bnd)
    u = [W + t for t in range(len(links))]
    bd = [W + outs_q[r] if r in outs_q else i for i, r in enumerate(bnd)]
    c_val = sp.nsimplify(c_chain ** (p // period))
    a_val = shift_ratio(chain_d, index, p)
    insts = [_instantiate_index(chain_d, index, r) for r in range(p + 1)]
    goals = [_instantiate_index(goal_d, index, r) for r in range(p)]
    if any(n not in d.nodes for d in insts for n in head | tail):
        raise PathSumUnsupported("instantiation renamed a node outside the box")
    blk_d, hd_d = open_part(insts[p], sp.Integer(1)), open_part(insts[0], c_val)
    if [r.direction for r in open_order(insts[p])[1]] != [
        r.direction for r in open_order(insts[0])[1]
    ]:
        raise PathSumUnsupported("the open parts' boundaries differ in direction")
    pairs = [*zip(insts[:p], goals, strict=True), (blk_d, hd_d)]
    names = [*("baseEq" for _ in range(p)), "blkEq"]
    bodies, phases, qudit, coefs = _bodies(pairs, names)
    if qudit or coefs:
        raise PathSumUnsupported("the periodic export covers unweighted qubit networks")
    labs = {k: v for body in bodies for k, v in body.labs.items()}
    ks = set(labs.values())
    if len(ks) > 1:
        raise PathSumUnsupported("the periodic export covers one wire dimension")
    m = bodies[0].m
    first: dict[tuple[str, int], int] = {}
    for i, entry in enumerate(phases):
        first.setdefault(entry, i + 1)
    for body in bodies:
        body.lines = [_canon_phases(ln, phases, first) for ln in body.lines]
    phase_of = {body: i for (body, _m), i in first.items()}
    body_of: dict[int, str | None] = {i + 1: body for i, (body, _m) in enumerate(phases)}
    fresh = iter(range(n_slots, W))

    def piece(nodes: set[NodeId], end: dict[PortRef, int]) -> NetT:
        """The nodes ``nodes`` with legs at ``end`` labelled by it, boundary legs by their slot
        and wires inside ``nodes`` by fresh labels."""
        label: dict[PortRef, int] = {}
        summed: list[int] = []
        own: list[int] = []
        for nid in sorted(nodes):
            for ref in _ports(chain_d, nid):
                if ref in label:
                    continue
                if ref in end:
                    label[ref] = end[ref]
                    continue
                if ref in slot:
                    label[ref] = bd[slot[ref]]
                    own.append(bd[slot[ref]])
                    continue
                other = _partner(chain_d, ref)
                if other is None or other.node_id not in nodes:
                    raise PathSumUnsupported(f"leg {ref} of {nid!r} leaves its piece")
                x = next(fresh)
                label[ref] = label[other] = x
                summed.append(x)
        return [_node(chain_d, nid, label, m, phase_of) for nid in sorted(nodes)], summed, own

    head_end: dict[PortRef, int] = {}
    tail_end: dict[PortRef, int] = {}
    for t, (pp, q) in enumerate(links):
        if (h := _partner(chain_d, q)) is not None:
            head_end[h] = W + t
        tail_end[cast(PortRef, _partner(chain_d, pp))] = W + t
    copy_end = {q: W + t for t, (_pp, q) in enumerate(links)}
    copy_end |= {pp: 2 * W + t for t, (pp, _q) in enumerate(links)}
    tl = piece(head, head_end)
    hd = piece(tail, tail_end)
    fresh = iter(range(W + len(links), 2 * W))
    cp = piece(set(scope), copy_end)

    bd_seed = {x: i for i, x in enumerate(bd)}
    pis = [
        _pull(_fam(u, tl, cp, hd, bd, r), _parse_net(bodies[r].lines, "L"), bd_seed, body_of)
        for r in range(p)
    ]
    g_net = _parse_net(bodies[0].lines, "R")
    rho = _relabel(_labels(g_net), dict(enumerate(bd)))
    g_tmpl = _map_net(g_net, lambda x: _ren(rho, x))
    rhos = [_pull(g_tmpl, _parse_net(bodies[r].lines, "R"), bd_seed, body_of) for r in range(p)]
    pi_b = _pull(
        _blk(u, cp, hd, p), _parse_net(bodies[p].lines, "L"), open_order(insts[p])[2], body_of
    )
    pi_h = _pull(
        (hd[0], hd[1], [*u, *hd[2]]),
        _parse_net(bodies[p].lines, "R"),
        open_order(insts[0])[2],
        body_of,
    )
    sc = _scalars()

    def model(diagram: Diagram) -> sp.Expr:
        return _const_model(*constant(diagram.scalar.to_sympy()))

    one = Diagram()
    one.set_scalar(Scalar(a_val))
    a_text = sc.text(model(one))
    blk = bodies[p]
    side = "goal" if flipped else "start"
    ns = f"ArchytasZX.PeriodicProof.{name}"
    lines = [
        "import ArchytasZX.PathSumNetW",
        "import ArchytasZX.PeriodicChain",
        "",
        "/-!",
        f"# Periodic theorem `{name}`",
        "",
        f"The {side} side as a periodic chain family and the other side as a fixed network: the",
        f"pairs at `{index} = 0 .. {p - 1}` and {p} copies then the tail side against the tail",
        "side proved by path sums and relabelled onto the templates, and `fam_periodic` for",
        "every index.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        "open ArchytasZX ArchytasZX.PSem",
        "",
        *NET_OPTIONS,
        "",
        *net_env(phases, labs, f"2 ^ {ks.pop() if ks else 1}", m, coefs),
        "theorem hD : ∀ x y, labDim E x = labDim E y := fun _ _ => rfl",
        "",
    ]
    for r in range(p):
        lines += [f"namespace Base{r}", "", *bodies[r].lines, f"end Base{r}", ""]
    lines += ["namespace Blk", "", *blk.lines, "end Blk", ""]
    lines += [
        f"def W : ℕ := {W}",
        f"def u : List ℕ := {u}",
        f"def tl : Net := {_net_text(tl)}",
        f"def cp : Net := {_net_text(cp)}",
        f"def hd : Net := {_net_text(hd)}",
        f"def bd : List ℕ := {bd}",
        f"def ρ : List (ℕ × ℕ) := {rho}",
        "def G : Net := Base0.NR.mapL (ren ρ)",
        *(f"def π{r} : List (ℕ × ℕ) := {pi}" for r, pi in enumerate(pis)),
        *(f"def ρ{r} : List (ℕ × ℕ) := {pi}" for r, pi in enumerate(rhos)),
        f"def πB : List (ℕ × ℕ) := {pi_b}",
        f"def πH : List (ℕ × ℕ) := {pi_h}",
        "noncomputable def sT : List ℂ := [" + ", ".join(f"({b.s_l})" for b in bodies[:p]) + "]",
        "noncomputable def gT : List ℂ := [" + ", ".join(f"({b.s_r})" for b in bodies[:p]) + "]",
        f"noncomputable def A : ℂ := {a_text}",
        f"noncomputable def c : ℂ := {blk.s_r}",
        "",
        "theorem ok : famOK W u tl cp hd bd = true := by decide",
        "",
        f"/-- {p} copies then the tail side are `c` times the tail side. -/",
        f"theorem h0 : Equal E 1 (blkNet W u cp hd {p}) c (headNet u hd) := by",
        "  have h := Equal.mapL2 E (ren πB) (ren πH) (ren_injective πB (by decide) (by decide))",
        "    (ren_injective πH (by decide) (by decide)) (fun _ => hD _ _) (fun _ => hD _ _)",
        "    Blk.blkEq (by decide) (by decide)",
        "  refine ((h.reorderL E (A' := blkNet W u cp hd "
        + f"{p}) (by decide) (by decide) (by decide)).reorder E",
        "    (B' := headNet u hd) (by decide) (by decide) (by decide)).congr_scalar ?_ rfl",
        "  norm_num",
        "",
    ]
    for r in range(p):
        lines += [
            f"theorem base{r} : Equal E ({bodies[r].s_l}) (famNet W u tl cp hd bd {r})",
            f"    ({bodies[r].s_r}) G := by",
            f"  have h := Equal.mapL2 E (ren π{r}) (ren ρ{r}) (ren_injective π{r} (by decide)",
            f"    (by decide)) (ren_injective ρ{r} (by decide) (by decide)) (fun _ => hD _ _)",
            f"    (fun _ => hD _ _) Base{r}.baseEq (by decide) (by decide)",
            f"  exact (h.reorderL E (A' := famNet W u tl cp hd bd {r}) (by decide) (by decide)",
            "    (by decide)).reorder E (B' := G) (by decide) (by decide) (by decide)",
            "",
        ]
    lines += [
        f"theorem base : ∀ r < {p}, Equal E (perS sT A {p} r) (famNet W u tl cp hd bd r)",
        f"    (perS gT (A * c) {p} r) G := by",
        "  intro r hr",
        "  rw [perS_lt _ _ _ _ hr, perS_lt _ _ _ _ hr]",
        "  interval_cases r",
        *(f"  · exact base{r}" for r in range(p)),
        "",
        f"/-- The {side} side's family and `G` agree at every index. -/",
        f"theorem main : ∀ k, Equal E (perS sT A {p} k) (famNet W u tl cp hd bd k)",
        f"    (perS gT (A * c) {p} k) G :=",
        f"  fam_periodic E hD ok {p} (by norm_num) c A h0 G _ _ (perS_step _ _ _ (by norm_num))",
        "    (perS_step _ _ _ (by norm_num)) base",
        "",
    ]
    for k in checks:
        lines += _check_lines(
            k,
            chain_d,
            goal_d,
            index,
            (u, tl, cp, hd, bd),
            g_tmpl,
            bodies[:p],
            a_val,
            c_val,
            p,
            m,
            phases,
            body_of,
            model,
        )
    lines += [f"end {ns}", ""]
    return "\n".join(lines)


def _check_lines(
    k: int,
    chain_d: Diagram,
    goal_d: Diagram,
    index: str,
    tmpl: tuple[list[int], NetT, NetT, NetT, list[int]],
    g_tmpl: NetT,
    bases: Sequence[NetBody],
    a_val: sp.Expr,
    c_val: sp.Expr,
    p: int,
    m: int,
    phases: Sequence[tuple[str, int]],
    body_of: Mapping[int, str | None],
    model: Callable[[Diagram], sp.Expr],
) -> list[str]:
    """``At{k}``: both sides instantiated at ``k`` as networks DL, DR, the relabellings of
    ``famNet`` at ``k`` and ``G`` onto them, their permutations by ``decide``, and inst (the
    two instances equal under their own scalars)."""
    from archytaszx.semantics.decide import _instantiate_index

    sc = _scalars()
    mdl = model
    u, tl, cp, hd, bd = tmpl
    out = [f"namespace At{k}", ""]
    proofs = []
    for tag, d, net, s_tab, ratio in (
        ("L", chain_d, _fam(u, tl, cp, hd, bd, k), [b.s_l for b in bases], a_val),
        ("R", goal_d, g_tmpl, [b.s_r for b in bases], a_val * c_val),
    ):
        inst_d = _instantiate_index(d, index, k)
        inst, summed, nb = _instance_net(inst_d, phases, False, m)
        fwd, match = _iso(net, inst, lambda ph: body_of.get(ph), {x: i for i, x in enumerate(bd)})
        pi = _relabel(_labels(net), fwd)
        nodes = [
            f"⟨{kd}, {list(o)}, {list(i)}, {dd}, {net[0][j][4]}⟩"
            for (kd, o, i, dd, _b), j in zip(inst, match, strict=True)
        ]
        out += [
            f"def π{tag} : List (ℕ × ℕ) := {pi}",
            f"def D{tag} : Net := ⟨[{', '.join(nodes)}], {summed}, {list(range(nb))}⟩",
            "",
        ]
        base_d = _instantiate_index(d, index, k % p)
        one = Diagram()
        one.set_scalar(Scalar(ratio))
        lhs = mdl(base_d) * mdl(one) ** (k // p)
        rhs = mdl(inst_d)
        proofs.append((tag, sc.text(rhs), sc.prove(lhs, rhs), sc.text(lhs)))
    fam = f"famNet W u tl cp hd bd {k}"
    out += [
        f"theorem linkL : (({fam}).mapL (ren πL)).nodes.Perm DL.nodes ∧",
        f"    (({fam}).mapL (ren πL)).int.Perm DL.int ∧ (({fam}).mapL (ren πL)).bnd.Perm DL.bnd :=",
        "  by decide",
        "theorem linkR : (G.mapL (ren πR)).nodes.Perm DR.nodes ∧",
        "    (G.mapL (ren πR)).int.Perm DR.int ∧ (G.mapL (ren πR)).bnd.Perm DR.bnd := by decide",
        "",
        f"theorem instK : Equal E (perS sT A {p} {k}) DL (perS gT (A * c) {p} {k}) DR := by",
        "  have h := Equal.mapL2 E (ren πL) (ren πR) (ren_injective πL (by decide) (by decide))",
        "    (ren_injective πR (by decide) (by decide)) (fun _ => hD _ _) (fun _ => hD _ _)",
        f"    (main {k}) (by decide) (by decide)",
        "  exact (h.reorderL E linkL.1 linkL.2.1 linkL.2.2).reorder E linkR.1 linkR.2.1 linkR.2.2",
        "",
    ]
    for (tag, rhs_text, proof, lhs_text), (tab, mult) in zip(
        proofs, (("sT", "A"), ("gT", "(A * c)")), strict=True
    ):
        out += [
            f"theorem scal{tag} : perS {tab} {mult} {p} {k} = {rhs_text} := by",
            f"  simp only [perS, {tab}, A, c, Nat.reduceMod, Nat.reduceDiv, List.getD_cons_zero,",
            "    List.getD_cons_succ]",
            f"  have key : ({lhs_text} : ℂ) = {rhs_text} :=",
            *("    " + ln for ln in proof.splitlines()),
            "  all_goals linear_combination key",
            "",
        ]
    out += [
        f"/-- Both sides instantiated at `{k}` are equal under their own scalars. -/",
        f"theorem inst : Equal E ({proofs[0][1]}) DL ({proofs[1][1]}) DR :=",
        "  instK.congr_scalar scalL scalR",
        "",
        f"end At{k}",
        "",
    ]
    return out


__all__ = ["export_periodic"]
