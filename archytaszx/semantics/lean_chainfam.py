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

"""Lean output for two closed chain boxes equal at every copy count. export_chain_family writes
each side as a Lean ``Diagram`` with a ``DChain`` (ArchytasZX/ChainBridge.lean), each side's
block of one copy against its head and the two sides at ``0`` copies proved by the verified
normaliser (lean_sym.export_symbolic) and moved onto the chain family's labels, and
``chain_pair`` (ArchytasZX/ChainPair.lean) giving both expansions equal at every ``k``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.pathsum import PathSumUnsupported
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.semantics.lean_chain import _ints, _relabel

LPort = tuple[int, bool, int]
"""A Lean port: node id, output side or not, position."""

LNet = tuple[list[tuple[str, tuple[int, ...], tuple[int, ...]]], list[int], list[int]]
"""A network of dimension and phase index 0: nodes (kind, outputs, inputs), summed and boundary
labels."""

_FAM_KINDS = {"Z": "Z", "X": "X", "F": "F"}


@dataclass
class _ChainSide:
    """One side as a Lean ``Diagram`` with a ``DChain``: the engine's nodes and wires, each
    boundary link port behind a phaseless Z(1 -> 1), the box's nodes and links; the engine
    nodes' leg dimension, the box multiplicity and the scalar."""

    nodes: list[tuple[int, str, int, int]]
    wires: list[tuple[LPort, LPort]]
    outs: list[LPort]
    ins: list[LPort]
    box: list[int]
    links: list[tuple[LPort, LPort]]
    dim: Dim
    mult: sp.Expr
    scalar: sp.Expr


def _lport(ref: PortRef) -> LPort:
    return (int(ref.node_id), ref.direction is Direction.OUTPUT, ref.index)


def _chain_side(d: Diagram) -> _ChainSide:
    """``d`` as a :class:`_ChainSide`; :class:`PathSumUnsupported` unless ``d`` has one plain
    node box with links, Z, X and F nodes of one leg dimension and no phase."""
    from archytaszx.algebra.phase import PhaseVector

    if len(d.bang_boxes) != 1:
        raise PathSumUnsupported("not one chain box")
    box = next(iter(d.bang_boxes.values()))
    if (
        not box.links
        or box.cyclic
        or box.port_scope
        or box.index is not None
        or box.parent is not None
        or box.copy_scalar is not None
        or box.pair_of is not None
        or box.tree
        or box.feed is not None
        or box.align is not None
    ):
        raise PathSumUnsupported("the chain box is not a plain linked node box")
    dims = {str(p.dim) for n in d.nodes.values() for p in (*n.inputs, *n.outputs)}
    if len(dims) != 1:
        raise PathSumUnsupported("the legs differ in dimension")
    dim = next(p.dim for n in d.nodes.values() for p in (*n.inputs, *n.outputs))
    nodes: list[tuple[int, str, int, int]] = []
    for x in sorted(d.nodes):
        n = d.nodes[x]
        kind = _FAM_KINDS.get(n.generator_type.name)
        if kind is None:
            raise PathSumUnsupported(f"generator {n.generator_type.name}")
        if n.phase is not None and not (isinstance(n.phase, PhaseVector) and n.phase.is_zero):
            raise PathSumUnsupported("a phased node")
        nodes.append((int(x), kind, len(n.outputs), len(n.inputs)))
    wires = [(_lport(w.a), _lport(w.b)) for w in sorted(d.wires, key=lambda w: w.sort_key())]
    links = [(_lport(p), _lport(q)) for p, q in box.links]
    ends = {r for pq in links for r in pq}
    nid = 1 + max((x for x, *_ in nodes), default=0)

    def held(refs: Sequence[PortRef]) -> list[LPort]:
        nonlocal nid
        out = []
        for ref in refs:
            p = _lport(ref)
            if p not in ends:
                out.append(p)
                continue
            nodes.append((nid, "Z", 1, 1))
            if p[1]:
                wires.append((p, (nid, False, 0)))
                out.append((nid, True, 0))
            else:
                wires.append(((nid, True, 0), p))
                out.append((nid, False, 0))
            nid += 1
        return out

    outs = held(d.boundary_outputs)
    ins = held(d.boundary_inputs)
    return _ChainSide(
        nodes,
        wires,
        outs,
        ins,
        sorted(int(x) for x in box.node_scope),
        links,
        dim,
        sp.sympify(box.multiplicity.to_sympy()),
        sp.sympify(d.scalar.to_sympy()),
    )


class _ChainLabels:
    """The Lean ``chLab0``, ``chLabT`` and the nets ``chTl``, ``chCp``, ``chHdN`` of a side at
    the width ``W``."""

    def __init__(self, s: _ChainSide, w: int) -> None:
        self.s, self.w = s, w
        self.bnd = [*s.outs, *s.ins]
        box = set(s.box)
        self.box = box

        def on_link(e: tuple[LPort, LPort]) -> bool:
            return any(e[0] in lk or e[1] in lk for lk in s.links)

        base = [e for e in s.wires if not on_link(e)]
        self.w_in = [e for e in base if e[0][0] in box]
        self.w_out = [e for e in base if e[0][0] not in box]
        if any((e[0][0] in box) != (e[1][0] in box) for e in base):
            raise PathSumUnsupported("a wire crosses the chain box off its links")
        self.partner = {}
        for p in {r for lk in s.links for r in lk}:
            found = next((e for e in s.wires if p in e), None)
            if found is None:
                raise PathSumUnsupported("a link port without a wire")
            self.partner[p] = found[1] if found[0] == p else found[0]
        self.hd = {self.partner[lk[0]][0] for lk in s.links}

    def lab0(self, p: LPort) -> int:
        if p in self.bnd:
            return self.bnd.index(p)
        for i, e in enumerate(self.w_out):
            if p in e:
                return len(self.bnd) + i
        for t, lk in enumerate(self.s.links):
            if self.partner[lk[1]] == p or self.partner[lk[0]] == p:
                return self.w + t
        return 0

    def lab_t(self, q: LPort) -> int:
        for i, e in enumerate(self.w_in):
            if q in e:
                return self.w + len(self.s.links) + i
        for t, lk in enumerate(self.s.links):
            if lk[1] == q:
                return self.w + t
        for t, lk in enumerate(self.s.links):
            if lk[0] == q:
                return 2 * self.w + t
        return 0

    def nodes(
        self, keep: Callable[[int], bool], lab: Callable[[LPort], int]
    ) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
        """The nodes ``keep`` holds, in order, their legs labelled by ``lab``."""
        return [
            (
                kind,
                tuple(lab((x, True, j)) for j in range(no)),
                tuple(lab((x, False, j)) for j in range(ni)),
            )
            for x, kind, no, ni in self.s.nodes
            if keep(x)
        ]

    def nets(self) -> tuple[LNet, LNet, LNet]:
        """``chTl``, ``chCp`` and ``chHdN``."""
        m = len(self.bnd)
        tl = self.nodes(lambda x: x not in self.box and x not in self.hd, self.lab0)
        cp = self.nodes(lambda x: x in self.box, self.lab_t)
        hd = self.nodes(lambda x: x not in self.box and x in self.hd, self.lab0)
        hd_bnd = [x for _, o, i in hd for x in (*o, *i) if x < m]
        nl = len(self.s.links)
        return (
            (tl, [m + i for i in range(len(self.w_out))], []),
            (cp, [self.w + nl + i for i in range(len(self.w_in))], []),
            (hd, [], hd_bnd),
        )

    def sizes(self) -> int:
        """The labels below the width: slots, outside wires, links and inside wires."""
        return len(self.bnd) + len(self.w_out) + len(self.s.links) + len(self.w_in)

    def check(self) -> None:
        """:class:`PathSumUnsupported` unless the side meets Lean's ``ChainFamOK``."""
        w, box = self.w, self.box
        if not box or any(p[0] in box for p in self.bnd) or self.sizes() >= w:
            raise PathSumUnsupported("the boundary meets the chain box")
        for t, (p1, p2) in enumerate(self.s.links):
            pp, qq = self.partner[p1], self.partner[p2]
            if (
                p1[0] not in box
                or p2[0] not in box
                or pp[0] in box
                or qq[0] in box
                or qq[0] in self.hd
                or self.lab0(pp) != w + t
                or self.lab0(qq) != w + t
                or self.lab_t(p2) != w + t
                or self.lab_t(p1) != 2 * w + t
            ):
                raise PathSumUnsupported("a link does not run from a head node to a tail node")
        if any(self.lab0(b) != self.lab0(a) for a, b in self.w_out) or any(
            self.lab_t(b) != self.lab_t(a) for a, b in self.w_in
        ):
            raise PathSumUnsupported("a wire's ends read different labels")
        ends = {r for lk in self.s.links for r in lk}
        far = set(self.partner.values())
        for x, _, no, ni in self.s.nodes:
            for q in [(x, True, j) for j in range(no)] + [(x, False, j) for j in range(ni)]:
                inside = any(q in e for e in self.w_in) or q in ends
                outside = q in self.bnd or any(q in e for e in self.w_out) or q in far
                if not (inside if x in box else outside):
                    raise PathSumUnsupported("a leg off every wire, link and slot")


def _sh(w: int, k: int, x: int) -> int:
    return x if x < w else x + k * w


def _sh_net(net: LNet, w: int, k: int) -> LNet:
    nodes, int_, bnd = net
    return (
        [
            (kd, tuple(_sh(w, k, x) for x in o), tuple(_sh(w, k, x) for x in i))
            for kd, o, i in nodes
        ],
        [_sh(w, k, x) for x in int_],
        [_sh(w, k, x) for x in bnd],
    )


def _fam_pieces(w: int, u: list[int], m: int, tl: LNet, cp: LNet, hd: LNet) -> tuple[LNet, ...]:
    """Lean's ``blkNet w u cp hd 1``, ``headNet u hd`` and ``famNet w u tl cp hd (range m) 0``."""
    hd1 = _sh_net(hd, w, 1)
    blk = (cp[0] + hd1[0], cp[1] + [_sh(w, 1, x) for x in u] + hd[1], u + hd[2])
    head = (hd[0], hd[1], u + hd[2])
    base = (tl[0] + hd[0], tl[1] + [x for x in u if x >= m] + hd[1], list(range(m)))
    return blk, head, base


def _net_diagram(net: LNet, dim: Dim, scalar: sp.Expr) -> Diagram:
    """``net`` times ``scalar`` as an engine diagram: each summed label a wire, each boundary
    label a boundary leg in ``net``'s order, outputs first."""
    from archytaszx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER

    gens = {"Z": Z_SPIDER, "X": X_SPIDER, "F": FOURIER_BOX}
    d = Diagram()
    legs: dict[int, list[PortRef]] = {}
    for kind, o, i in net[0]:
        nid = d.add_node(gens[kind], [dim] * len(i), [dim] * len(o))
        for j, x in enumerate(o):
            legs.setdefault(x, []).append(PortRef(nid, Direction.OUTPUT, j))
        for j, x in enumerate(i):
            legs.setdefault(x, []).append(PortRef(nid, Direction.INPUT, j))
    for x in net[1]:
        if len(legs.get(x, [])) != 2:
            raise PathSumUnsupported("a summed label not on two legs")
        d.add_wire(legs[x][0], legs[x][1])
    outs = [x for x in net[2] if legs[x][0].direction is Direction.OUTPUT]
    ins = [x for x in net[2] if legs[x][0].direction is Direction.INPUT]
    if any(len(legs[x]) != 1 for x in net[2]):
        raise PathSumUnsupported("a boundary label not on one leg")
    d.set_boundary_outputs([legs[x][0] for x in outs])
    d.set_boundary_inputs([legs[x][0] for x in ins])
    d.set_scalar(Scalar(scalar))
    return d


_PIECE_NET = re.compile(r"^def (N[01]) : Net := (.*)$", re.MULTILINE)
_LNODE = re.compile(r"⟨\.(\w+), \[([0-9, ]*)\], \[([0-9, ]*)\], (\d+), (\d+)⟩")
_LTAIL = re.compile(r"\[([0-9, ]*)\], \[([0-9, ]*)\]⟩$")

PieceNet = tuple[list[tuple[str, list[int], list[int], int, int]], list[int], list[int]]
"""A normaliser network: nodes (kind, outputs, inputs, dimension index, phase index), summed and
boundary labels."""


def _piece_net(source: str, tag: str) -> PieceNet:
    """The network ``tag`` of a normaliser export."""
    line = next(m[2] for m in _PIECE_NET.finditer(source) if m[1] == tag)
    nodes = [(m[1], _ints(m[2]), _ints(m[3]), int(m[4]), int(m[5])) for m in _LNODE.finditer(line)]
    tail = _LTAIL.search(line)
    if tail is None:
        raise PathSumUnsupported("a normaliser network this reader does not parse")
    return nodes, _ints(tail[1]), _ints(tail[2])


def _piece_map(piece: PieceNet, net: LNet) -> dict[int, int]:
    """The label map from a normaliser network onto ``net``, node by node in order."""
    nodes = piece[0]
    if len(nodes) != len(net[0]):
        raise PathSumUnsupported("the normaliser network has other nodes")
    out: dict[int, int] = {}
    for (kind, o, i, dm, ph), (k2, o2, i2) in zip(nodes, net[0], strict=True):
        if kind != k2 or len(o) != len(o2) or len(i) != len(i2) or dm or ph:
            raise PathSumUnsupported("the normaliser network's nodes differ")
        for a, b in zip((*o, *i), (*o2, *i2), strict=True):
            if out.setdefault(a, b) != b:
                raise PathSumUnsupported("the normaliser network's labels differ")
    if len(set(out.values())) != len(out):
        raise PathSumUnsupported("the label map is not one to one")
    return out


_MAIN = re.compile(
    r"^theorem main ((?:\([^()]*\) ?)*):\n\s*Equal \(E d\) (.*) N0 (.*) (N[01]) :=", re.MULTILINE
)


def _lnat(xs: Sequence[int]) -> str:
    return "[" + ", ".join(map(str, xs)) + "]"


def _lp(p: LPort) -> str:
    return f"⟨{p[0]}, {'true' if p[1] else 'false'}, {p[2]}⟩"


def _lean_side(s: _ChainSide) -> tuple[str, str]:
    """The Lean ``Diagram`` and ``DChain`` literals of a side."""
    nodes = ", ".join(
        f"⟨{x}, .{k}, 0, 0, {_lnat([0] * no)}, {_lnat([0] * ni)}⟩" for x, k, no, ni in s.nodes
    )
    wires = ", ".join(f"({_lp(a)}, {_lp(b)})" for a, b in s.wires)
    box = f"⟨0, 0, none, {_lnat(s.box)}, []⟩"
    outs, ins = ", ".join(map(_lp, s.outs)), ", ".join(map(_lp, s.ins))
    diagram = (
        f"{{ nodes := [{nodes}],\n    wires := [{wires}],\n    outs := [{outs}],"
        f"\n    ins := [{ins}],\n    boxes := [{box}] }}"
    )
    links = ", ".join(f"({_lp(a)}, {_lp(b)})" for a, b in s.links)
    return diagram, f"⟨{box}, [{links}]⟩"


def _mult_nat(mult: sp.Expr) -> tuple[str, str]:
    """The box multiplicity as a Lean natural and the Lean name of its one symbol."""
    from archytaszx.semantics.kernel import _nat_expr, _var

    syms = sorted(mult.free_symbols, key=str)
    if len(syms) != 1:
        raise PathSumUnsupported("the box multiplicity has no single symbol")
    return _nat_expr(mult), _var(str(syms[0]))


def _per_copy(s: _ChainSide, k: sp.Symbol) -> tuple[sp.Expr, sp.Expr]:
    """The scalar at `0` copies and the ratio per copy, the scalar read at the box's copy count
    ``k``."""
    syms = sorted(s.mult.free_symbols, key=str)
    if len(syms) != 1:
        raise PathSumUnsupported("the box multiplicity has no single symbol")
    n = syms[0]
    sol = sp.solve(sp.Eq(s.mult, k), n)
    if len(sol) != 1:
        raise PathSumUnsupported("the box multiplicity is not linear")
    at = sp.simplify(s.scalar.subs(n, sol[0]))
    ratio = sp.simplify(at.subs(k, k + 1) / at)
    if k in ratio.free_symbols:
        raise PathSumUnsupported("the scalar does not grow by a fixed ratio per copy")
    return sp.simplify(at.subs(k, 0)), ratio


def _piece_source(source: str, ns: str, macros: bool) -> str:
    """A normaliser export as the namespace ``ns``: its import, header and diagram restatement
    dropped, its macros kept only with ``macros``."""
    body = source.split("namespace ArchytasZX.Export", 1)[1]
    body = body.rsplit("end ArchytasZX.Export", 1)[0]
    body = re.sub(r"\n/--[^\n]*-/\s*$", "\n", body.split("\ndef D0 : Diagram", 1)[0]) + "\n"
    if not macros:
        body = re.sub(r"set_option hygiene false in\n/--[^\n]*-/\nmacro [^\n]*\n", "", body)
    return f"namespace {ns}\n{body}end {ns}\n"


def export_chain_family(left: Diagram, right: Diagram, name: str = "main") -> str:
    """Lean 4 source proving ``left = right`` at every count of their chain boxes, for every
    value of the dimension: each side a closed chain box (ArchytasZX/ChainBridge.lean), its
    expansion at ``k`` the chain family ``chFam`` (``expand_chFam``); each side's block of one
    copy and its head, and the two sides at ``0`` copies, proved by the verified normaliser
    (``export_symbolic``) and moved to one environment (``Equal.env0``, ``Equal.mapL2``,
    ``Equal.reorder``); ``chain_pair`` for every ``k``, and ``main`` at the box multiplicity in
    the engine's symbol; :class:`PathSumUnsupported` outside that form."""
    from archytaszx.semantics.lean_net import NetUnsupported
    from archytaszx.semantics.lean_sym import export_symbolic

    sa, sb = _chain_side(left), _chain_side(right)
    if sp.simplify(sa.mult - sb.mult) != 0 or len(sa.links) != len(sb.links):
        raise PathSumUnsupported("the two chain boxes differ in count or links")
    if len(sa.outs) + len(sa.ins) != len(sb.outs) + len(sb.ins) or str(sa.dim) != str(sb.dim):
        raise PathSumUnsupported("the two sides differ in boundary")
    probe = [_ChainLabels(s, 1) for s in (sa, sb)]
    w = 1 + max(p.sizes() for p in probe)
    la, lb = _ChainLabels(sa, w), _ChainLabels(sb, w)
    la.check()
    lb.check()
    m = len(la.bnd)
    u = [w + t for t in range(len(sa.links))]
    k = sp.Symbol("k_chain", integer=True, nonnegative=True)
    a0, ra = _per_copy(sa, k)
    b0, rb = _per_copy(sb, k)
    blk_a, head_a, base_a = _fam_pieces(w, u, m, *la.nets())
    blk_b, head_b, base_b = _fam_pieces(w, u, m, *lb.nets())
    pieces = [
        ("PA", (blk_a, ra), (head_a, sp.Integer(1))),
        ("PB", (blk_b, rb), (head_b, sp.Integer(1))),
        ("P0", (base_a, a0), (base_b, b0)),
    ]
    blocks: list[str] = []
    stmts: dict[str, tuple[str, str, str]] = {}
    for j, (tag, (n0, s0), (n1, s1)) in enumerate(pieces):
        try:
            src = export_symbolic(_net_diagram(n0, sa.dim, s0), _net_diagram(n1, sa.dim, s1))
        except NetUnsupported as exc:
            raise PathSumUnsupported(f"the normaliser does not prove {tag}: {exc}") from exc
        found = _MAIN.search(src)
        if found is None:
            raise PathSumUnsupported(f"{tag}: a normaliser theorem this reader does not parse")
        if "| 0 => d\n" not in src.split("def Δ", 1)[1].split("\n\n", 1)[0] + "\n":
            raise PathSumUnsupported(f"{tag}: the dimension table does not start at d")
        stmts[tag] = (found[1].strip(), found[2], found[3])
        f0 = _piece_map(_piece_net(src, "N0"), n0)
        f1 = _piece_map(_piece_net(src, found[4]), n1)
        pi0 = _relabel(set(f0), f0)
        pi1 = _relabel(set(f1), f1)
        blocks.append(_piece_source(src, tag, j == 0))
        blocks.append(
            f"def π{tag}0 : List (ℕ × ℕ) := {pi0}\n\ndef π{tag}1 : List (ℕ × ℕ) := {pi1}\n"
        )
    binders = {b for b, _, _ in stmts.values()}
    if len(binders) != 1:
        raise PathSumUnsupported("the normaliser pieces take different hypotheses")
    bind = binders.pop()
    args = " ".join(re.findall(r"\((\w+) :", bind))
    da, ca = _lean_side(sa)
    db, cb = _lean_side(sb)
    chain_ok = (
        "⟨by decide +kernel, ⟨by decide +kernel, by decide +kernel, by decide +kernel, "
        "by decide +kernel⟩, " + ", ".join(["by decide +kernel"] * 10) + "⟩"
    )

    def moved(tag: str, lhs: str, rhs: str) -> str:
        return (
            f"  have h := ({tag}.main {args}).env0 (E' := Ec d)\n"
            f"    (fun x => by simp only [labDim, Ec, {tag}.E, lookup_getD_all {tag}.labs 0 x "
            "(by decide), " + f"{tag}.Δ])\n"
            "    (by decide) rfl rfl\n"
            f"  have h2 := Equal.mapL2 (Ec d) (ren π{tag}0) (ren π{tag}1)\n"
            f"    (ren_injective π{tag}0 (by decide) (by decide))\n"
            f"    (ren_injective π{tag}1 (by decide) (by decide)) (fun _ => rfl) (fun _ => rfl) h\n"
            "    (by decide) (by decide)\n"
            f"  exact (h2.reorderL (Ec d) (A' := {lhs}) (by decide +kernel) (by decide +kernel)\n"
            f"    (by decide +kernel)).reorder (Ec d) (B' := {rhs}) (by decide +kernel)\n"
            "    (by decide +kernel) (by decide +kernel)\n"
        )

    _, ra_t, _ = stmts["PA"]
    _, rb_t, _ = stmts["PB"]
    _, a_t, b_t = stmts["P0"]
    kk, nv = _mult_nat(sa.mult)
    u_t = "(chU cA W)"

    def step(side: str, c: str, r: str) -> tuple[str, str, str]:
        blk = f"blkNet W {u_t} (D{side}.chCp {c} W) (D{side}.chHdN {c} W) 1"
        head = f"headNet {u_t} (D{side}.chHdN {c} W)"
        return f"Equal (Ec d) {r} ({blk}) 1\n      ({head})", blk, head

    def fam_ok(side: str, c: str) -> str:
        nets = f"(D{side}.chTl {c} W) (D{side}.chCp {c} W) (D{side}.chHdN {c} W)"
        return f"famOK W {u_t} {nets}\n    (List.range DA.bnd.length) = true := by decide +kernel"

    def at(k: str) -> str:
        return (
            f"Equal (Ec d) ({a_t} * ({ra_t}) ^ {k}) ((DA.expandChain cA {k}).toNet {m})\n"
            f"      ({b_t} * ({rb_t}) ^ {k}) ((DB.expandChain cB {k}).toNet {m})"
        )

    st_a, blk_a_t, head_a_t = step("A", "cA", ra_t)
    st_b, blk_b_t, head_b_t = step("B", "cB", rb_t)
    ec = "{ dimv := fun _ => d, phase := fun _ _ => 0, labs := [] }"
    s0 = f"Equal (Ec d) {a_t} (DA.chFam cA W 0) {b_t} (DB.chFam cB W 0)"
    lines = [
        "import ArchytasZX",
        "",
        "/-!",
        "# A closed chain family",
        "",
        "Both sides are closed chain boxes; `family` gives them equal at every copy count `k`",
        "(`chain_pair`): each side's expansion is its chain family (`expand_chFam`), each side's",
        "block of one copy and its head, and the two sides at `0` copies, are proved by the",
        "normaliser (`PA`, `PB`, `P0`) and moved to the environment `Ec`; `main` states it at the",
        "box multiplicity.",
        "-/",
        "",
        "namespace ArchytasZX.Export",
        "",
        "open ArchytasZX ArchytasZX.Diagram",
        "",
        "set_option maxRecDepth 100000",
        "set_option maxHeartbeats 2000000",
        "",
        *blocks,
        "/-- The left side with its boundary link ports behind identities. -/",
        f"def DA : Diagram :=\n  {da}",
        "",
        f"def cA : DChain := {ca}",
        "",
        "/-- The right side with its boundary link ports behind identities. -/",
        f"def DB : Diagram :=\n  {db}",
        "",
        f"def cB : DChain := {cb}",
        "",
        f"def W : ℕ := {w}",
        "",
        "/-- Every label of dimension `d`, every phase zero. -/",
        f"noncomputable def Ec (d : ℕ) : Env := {ec}",
        "",
        "theorem hDc (d : ℕ) : ∀ x y, labDim (Ec d) x = labDim (Ec d) y := fun _ _ => rfl",
        "",
        f"theorem okA : ChainFamOK DA cA W :=\n  {chain_ok}",
        "",
        f"theorem okB : ChainFamOK DB cB W :=\n  {chain_ok}",
        "",
        f"theorem fA : {fam_ok('A', 'cA')}",
        "",
        f"theorem fB : {fam_ok('B', 'cB')}",
        "",
        f"theorem sA {bind} :\n    {st_a} := by",
        moved("PA", blk_a_t, head_a_t),
        f"theorem sB {bind} :\n    {st_b} := by",
        moved("PB", blk_b_t, head_b_t),
        f"theorem s0 {bind} :\n    {s0} := by",
        moved("P0", "DA.chFam cA W 0", "DB.chFam cB W 0"),
        "/-- The two sides equal at every copy count. -/",
        f"theorem family {bind} (k : ℕ) :\n    {at('k')} :=",
        f"  chain_pair okA okB (by decide) (by decide) (Ec d) (hDc d) {m} (by decide) fA fB",
        f"    (sA {args}) (sB {args}) (s0 {args}) k",
        "",
        "/-- The two sides equal at the box multiplicity. -/",
        f"theorem {name} {bind} ({nv} : ℕ) :\n    {at(kk)} :=",
        f"  family {args} {kk}",
        "",
        "end ArchytasZX.Export",
        "",
    ]
    return "\n".join(lines)
