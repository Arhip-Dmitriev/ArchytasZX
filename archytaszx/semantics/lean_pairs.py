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

"""Lean output for two diagrams with a pair box over the node box holding every node, equal at
every copy count. export_pair_family writes each side as a Lean ``PDiagram`` with its ``DPair``
(ArchytasZX/PairBox.lean), the two templates without their pair legs proved equal by the verified
normaliser (lean_sym.export_symbolic) and moved onto ``tmpl``, the pair legs' pin certificates,
and ``pair_family`` (ArchytasZX/PairFam.lean) giving both expansions equal at every ``k``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.pathsum import PathSumUnsupported
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.semantics.lean_chain import _relabel
from archytaszx.semantics.lean_chainfam import (
    _MAIN,
    LNet,
    _lnat,
    _lp,
    _mult_nat,
    _net_diagram,
    _piece_map,
    _piece_net,
    _piece_source,
)

LPort = tuple[int, bool, int]
"""A Lean port: node id, output side or not, position."""

_KINDS = {"Z": "Z", "X": "X", "F": "F"}


@dataclass
class _PairSide:
    """One side as a Lean ``Diagram`` with its ``DPair``: the engine's nodes and wires off the pair
    node, the node box's nodes, the pair ports and node; the leg dimension, the node box's
    multiplicity, the scalar and the two copy scalars."""

    nodes: list[tuple[int, str, int, int]]
    wires: list[tuple[LPort, LPort]]
    outs: list[LPort]
    box: list[int]
    left: LPort
    right: LPort
    fid: int
    dim: Dim
    mult: sp.Expr
    scalar: sp.Expr
    per_copy: sp.Expr
    per_pair: sp.Expr


def _lport(ref: PortRef) -> LPort:
    return (int(ref.node_id), ref.direction is Direction.OUTPUT, ref.index)


def has_pairs(left: Diagram, right: Diagram) -> bool:
    """Whether either side has a pair box."""
    return any(b.pair_of is not None for d in (left, right) for b in d.bang_boxes.values())


def _pair_side(d: Diagram) -> _PairSide:
    """``d`` as a :class:`_PairSide`; :class:`PathSumUnsupported` unless ``d`` has one plain node
    box holding every node off one pair box over it, that pair box one ``F`` from its left port
    to its right port, Z, X and F nodes of one leg dimension and no phase, and no inputs."""
    from archytaszx.algebra.phase import PhaseVector

    boxes = list(d.bang_boxes.values())
    pairs = [b for b in boxes if b.pair_of is not None]
    if len(boxes) != 2 or len(pairs) != 1:
        raise PathSumUnsupported("not one node box with one pair box")
    pb = pairs[0]
    vb = next(b for b in boxes if b is not pb)
    if pb.pair_of != vb.id:
        raise PathSumUnsupported("the pair box is not over the node box")
    if (
        vb.links
        or vb.cyclic
        or vb.port_scope
        or vb.parent is not None
        or vb.tree
        or vb.feed is not None
        or vb.align is not None
        or vb.groups
        or pb.port_scope
        or pb.links
        or pb.tree
    ):
        raise PathSumUnsupported("the boxes are not a plain node box and its pair box")
    if len(pb.node_scope) != 1 or len(pb.left_ports) != 1 or len(pb.right_ports) != 1:
        raise PathSumUnsupported("the pair box is not one node between two ports")
    fid = int(next(iter(pb.node_scope)))
    left = _lport(next(iter(pb.left_ports)))
    right = _lport(next(iter(pb.right_ports)))
    if d.boundary_inputs:
        raise PathSumUnsupported("a boundary input")
    dims = {str(p.dim) for n in d.nodes.values() for p in (*n.inputs, *n.outputs)}
    if len(dims) != 1:
        raise PathSumUnsupported("the legs differ in dimension")
    dim = next(p.dim for n in d.nodes.values() for p in (*n.inputs, *n.outputs))
    nodes: list[tuple[int, str, int, int]] = []
    for x in sorted(d.nodes):
        n = d.nodes[x]
        kind = _KINDS.get(n.generator_type.name)
        if kind is None:
            raise PathSumUnsupported(f"generator {n.generator_type.name}")
        if n.phase is not None and not (isinstance(n.phase, PhaseVector) and n.phase.is_zero):
            raise PathSumUnsupported("a phased node")
        if int(x) != fid:
            nodes.append((int(x), kind, len(n.outputs), len(n.inputs)))
    fnode = d.nodes[next(iter(pb.node_scope))]
    if fnode.generator_type.name != "F":
        raise PathSumUnsupported("the pair node is not a Fourier box")
    ids = {x for x, *_ in nodes}
    if {int(x) for x in vb.node_scope} - {fid} != ids:
        raise PathSumUnsupported("the node box does not hold every node")
    wires: list[tuple[LPort, LPort]] = []
    attach: set[tuple[LPort, LPort]] = set()
    for w in sorted(d.wires, key=lambda w: w.sort_key()):
        a, b = _lport(w.a), _lport(w.b)
        if a[0] == fid or b[0] == fid:
            inner, outer = (a, b) if a[0] == fid else (b, a)
            attach.add((inner, outer))
        else:
            wires.append((a, b))
    if attach != {((fid, False, 0), left), ((fid, True, 0), right)}:
        raise PathSumUnsupported(
            "the pair node's input is not at the left port, its output at the right"
        )
    one = sp.Integer(1)
    return _PairSide(
        nodes,
        wires,
        [_lport(r) for r in d.boundary_outputs],
        sorted(ids),
        left,
        right,
        fid,
        dim,
        sp.sympify(vb.multiplicity.to_sympy()),
        sp.sympify(d.scalar.to_sympy()),
        sp.sympify(vb.copy_scalar.to_sympy()) if vb.copy_scalar is not None else one,
        sp.sympify(pb.copy_scalar.to_sympy()) if pb.copy_scalar is not None else one,
    )


class _Template:
    """Python mirror of Lean's ``D.label m``, ``tmpl`` and ``pinOK`` for a side."""

    def __init__(self, s: _PairSide) -> None:
        self.s = s
        self.m = len(s.outs)
        self.pair = {s.left, s.right}

    def label(self, q: LPort) -> int:
        if q in self.s.outs:
            return self.s.outs.index(q)
        for i, w in enumerate(self.s.wires):
            if q in w:
                return self.m + i
        return 0

    def net(self) -> LNet:
        """``tmpl``: each node without its pair legs, wire ``i`` at ``m + i``, the slots."""
        nodes = []
        for x, kind, no, ni in self.s.nodes:
            outs = tuple(
                self.label((x, True, k)) for k in range(no) if (x, True, k) not in self.pair
            )
            ins = tuple(
                self.label((x, False, k)) for k in range(ni) if (x, False, k) not in self.pair
            )
            nodes.append((kind, outs, ins))
        return nodes, [self.m + i for i in range(len(self.s.wires))], list(range(self.m))

    def check(self) -> None:
        """:class:`PathSumUnsupported` unless the side meets Lean's ``PairOK``."""
        s, legs = self.s, {x: (no, ni) for x, _, no, ni in self.s.nodes}

        def side_len(q: LPort) -> int:
            no, ni = legs.get(q[0], (0, 0))
            return no if q[1] else ni

        l, r = s.left, s.right
        if l == r or l[0] not in legs or r[0] not in legs:
            raise PathSumUnsupported("the pair ports are not two ports of the box")
        if r[2] + 1 != side_len(r):
            raise PathSumUnsupported("the right pair port is not the last leg of its side")
        same = (r[0], r[1]) == (l[0], l[1])
        if not (l[2] + 1 == side_len(l) or (same and r[2] == l[2] + 1 and l[2] + 2 == side_len(l))):
            raise PathSumUnsupported("the left pair port is not last on its side")
        if l in s.outs or r in s.outs or any(q in w for w in s.wires for q in (l, r)):
            raise PathSumUnsupported("a pair port on a wire or the boundary")
        ends = [q for w in s.wires for q in w]
        if len(set(ends)) != len(ends) or any(q in s.outs for q in ends):
            raise PathSumUnsupported("a leg on two wires")
        if len(set(s.outs)) != len(s.outs):
            raise PathSumUnsupported("a leg on two slots")
        for q in [*ends, *s.outs]:
            if q[0] not in legs or q[2] >= side_len(q):
                raise PathSumUnsupported("a wire or slot off a leg")
        for x, _, no, ni in s.nodes:
            for q in [(x, True, k) for k in range(no)] + [(x, False, k) for k in range(ni)]:
                if q not in self.pair and q not in s.outs and q not in ends:
                    raise PathSumUnsupported("a leg off every wire and slot")

    def pin(self, q: LPort) -> tuple[list[tuple[int, int]], int]:
        """A chain certificate of Z spiders from the node of ``q`` to a slot, and the slot."""
        nodes = self.net()[0]
        ids = [x for x, *_ in self.s.nodes]
        start = ids.index(q[0])
        labs = [set(o) | set(i) for _, o, i in nodes]
        z = [k == "Z" for k, _, _ in nodes]
        if not z[start]:
            raise PathSumUnsupported("a pair port off a Z spider")
        prev: dict[int, tuple[int, int] | None] = {start: None}
        todo = [start]
        while todo:
            i = todo.pop(0)
            slots = sorted(x for x in labs[i] if x < self.m)
            if slots:
                path = [(i, slots[0])]
                while prev[path[0][0]] is not None:
                    j, x = prev[path[0][0]]  # type: ignore[misc]
                    path.insert(0, (j, x))
                return path, slots[0]
            for j in range(len(nodes)):
                if z[j] and j not in prev:
                    shared = sorted(labs[i] & labs[j])
                    if shared:
                        prev[j] = (i, shared[0])
                        todo.append(j)
        raise PathSumUnsupported("a pair port not tied to a slot by Z spiders")


def _lean_side(s: _PairSide) -> tuple[str, str]:
    """The Lean ``PDiagram`` and ``DPair`` literals of a side."""
    nodes = ", ".join(
        f"⟨{x}, .{k}, 0, 0, {_lnat([0] * no)}, {_lnat([0] * ni)}⟩" for x, k, no, ni in s.nodes
    )
    wires = ", ".join(f"({_lp(a)}, {_lp(b)})" for a, b in s.wires)
    outs = ", ".join(map(_lp, s.outs))
    pair = f"⟨1, 0, {_lp(s.left)}, {_lp(s.right)}, ⟨{s.fid}, .F, 0, 0, [0], [0]⟩⟩"
    diagram = (
        f"{{ nodes := [{nodes}],\n    wires := [{wires}],\n    outs := [{outs}],"
        f"\n    ins := [],\n    boxes := [⟨0, 0, none, {_lnat(s.box)}, []⟩],"
        f"\n    pairs := [{pair}] }}"
    )
    return diagram, f"⟨0, 0, none, {_lnat(s.box)}, []⟩"


def _per_copy(s: _PairSide, k: sp.Symbol) -> tuple[sp.Expr, sp.Expr]:
    """The scalar at `0` copies and the ratio per copy, its node box's copy scalar included."""
    syms = sorted(s.mult.free_symbols, key=str)
    if len(syms) != 1:
        raise PathSumUnsupported("the box multiplicity has no single symbol")
    n = syms[0]
    sol = sp.solve(sp.Eq(s.mult, k), n)
    if len(sol) != 1:
        raise PathSumUnsupported("the box multiplicity is not linear")
    at = sp.simplify(s.scalar.subs(n, sol[0]))
    ratio = sp.simplify(at.subs(k, k + 1) / at * s.per_copy)
    if k in ratio.free_symbols:
        raise PathSumUnsupported("the scalar does not grow by a fixed ratio per copy")
    return sp.simplify(at.subs(k, 0)), ratio


def _pair_text(q: sp.Expr, dsym: sp.Symbol, dname: str) -> str:
    """The pair copy scalar ``d ^ r`` as a Lean ℂ term."""
    if sp.simplify(q - 1) == 0:
        return "1"
    base, ex = sp.Pow(q, 1).as_base_exp()
    if base != dsym or not ex.is_Rational or ex.q not in (1, 2):
        raise PathSumUnsupported("a pair copy scalar other than a power of the dimension")
    r = rf"(rt {dname})" if ex.q == 2 else f"(({dname} : ℕ) : ℂ)"
    p = int(ex.p)
    if p > 0:
        return f"({r} ^ {p})"
    return f"(({r})⁻¹ ^ {-p})"


def export_pair_family(left: Diagram, right: Diagram, name: str = "main") -> str:
    """Lean 4 source proving ``left = right`` at every count of their node boxes, for every value
    of the dimension: each side a pair box over the node box holding every node
    (ArchytasZX/PairBox.lean), its pair legs pinned to the same slots on both sides; the two
    templates proved by the verified normaliser (``export_symbolic``) and moved to one
    environment (``Equal.env0``, ``Equal.mapL2``, ``Equal.reorder``); ``pair_family`` for every
    ``k`` and ``main`` at the box multiplicity in the engine's symbol;
    :class:`PathSumUnsupported` outside that form."""
    from archytaszx.semantics.lean_net import NetUnsupported
    from archytaszx.semantics.lean_sym import export_symbolic

    sa, sb = _pair_side(left), _pair_side(right)
    if sp.simplify(sa.mult - sb.mult) != 0 or len(sa.outs) != len(sb.outs):
        raise PathSumUnsupported("the two sides differ in count or boundary")
    if str(sa.dim) != str(sb.dim):
        raise PathSumUnsupported("the two sides differ in dimension")
    ta, tb = _Template(sa), _Template(sb)
    ta.check()
    tb.check()
    cl_a, sl_a = ta.pin(sa.left)
    cr_a, sr_a = ta.pin(sa.right)
    cl_b, sl_b = tb.pin(sb.left)
    cr_b, sr_b = tb.pin(sb.right)
    if (sl_a, sr_a) != (sl_b, sr_b):
        raise PathSumUnsupported("the pair legs are pinned to different slots")
    k = sp.Symbol("k_pair", integer=True, nonnegative=True)
    c_a, u_a = _per_copy(sa, k)
    c_b, u_b = _per_copy(sb, k)
    if sp.simplify(c_a - 1) != 0 or sp.simplify(c_b - 1) != 0:
        raise PathSumUnsupported("a scalar other than 1 at 0 copies")
    if sp.simplify(sa.per_pair - sb.per_pair) != 0:
        raise PathSumUnsupported("the pair copy scalars differ")
    net_a, net_b = ta.net(), tb.net()
    try:
        src = export_symbolic(_net_diagram(net_a, sa.dim, u_a), _net_diagram(net_b, sa.dim, u_b))
    except NetUnsupported as exc:
        raise PathSumUnsupported(f"the normaliser does not prove the templates: {exc}") from exc
    found = _MAIN.search(src)
    if found is None:
        raise PathSumUnsupported("a normaliser theorem this reader does not parse")
    if "| 0 => d\n" not in src.split("def Δ", 1)[1].split("\n\n", 1)[0] + "\n":
        raise PathSumUnsupported("the dimension table does not start at d")
    bind, a_t, b_t = found[1].strip(), found[2], found[3]
    args = " ".join(re.findall(r"\((\w+) :", bind))
    f0 = _piece_map(_piece_net(src, "N0"), net_a)
    f1 = _piece_map(_piece_net(src, found[4]), net_b)
    pi0, pi1 = _relabel(set(f0), f0), _relabel(set(f1), f1)
    dsyms = sorted(sp.sympify(sa.dim.to_sympy()).free_symbols, key=str)
    if len(dsyms) != 1 or sp.sympify(sa.dim.to_sympy()) != dsyms[0]:
        raise PathSumUnsupported("the dimension is not one symbol")
    q_t = _pair_text(sa.per_pair, dsyms[0], "d")
    kk, nv = _mult_nat(sa.mult)
    m = len(sa.outs)
    pa, ba = _lean_side(sa)
    pb, bb = _lean_side(sb)

    def at(kx: str) -> str:
        sc = f"({kx} * ({kx} - 1) / 2)"
        return (
            f"Equal (Ec d) (1 * ({a_t}) ^ {kx} * {q_t} ^ {sc})"
            f" ((PA.expandP {kx}).toNet ({kx} * {m}))\n"
            f"      (1 * ({b_t}) ^ {kx} * {q_t} ^ {sc}) ((PB.expandP {kx}).toNet ({kx} * {m}))"
        )

    def cert(c: list[tuple[int, int]]) -> str:
        return "[" + ", ".join(f"({i}, {x})" for i, x in c) + "]"

    pin = (
        f"(by decide +kernel : PA.toDiagram.pinOK pA pA.left {cert(cl_a)} {sl_a} = true)\n"
        f"    (by decide +kernel : PA.toDiagram.pinOK pA pA.right {cert(cr_a)} {sr_a} = true)\n"
        f"    (by decide +kernel : PB.toDiagram.pinOK pB pB.left {cert(cl_b)} {sl_b} = true)\n"
        f"    (by decide +kernel : PB.toDiagram.pinOK pB pB.right {cert(cr_b)} {sr_b} = true)"
    )
    ec = "{ dimv := fun _ => d, phase := fun _ _ => 0, labs := [] }"
    moved = (
        f"  have h := (PT.main {args}).env0 (E' := Ec d)\n"
        "    (fun x => by\n"
        "      simp only [labDim, Ec, PT.E, lookup_getD_all PT.labs 0 x (by decide), PT.Δ])\n"
        "    (by decide) rfl rfl\n"
        "  have h2 := Equal.mapL2 (Ec d) (ren πT0) (ren πT1)\n"
        "    (ren_injective πT0 (by decide) (by decide))\n"
        "    (ren_injective πT1 (by decide) (by decide)) (fun _ => rfl) (fun _ => rfl) h\n"
        "    (by decide) (by decide)\n"
        "  exact (h2.reorderL (Ec d) (A' := PA.toDiagram.tmpl pA) (by decide +kernel)\n"
        "    (by decide +kernel) (by decide +kernel)).reorder (Ec d) (B' := PB.toDiagram.tmpl pB)\n"
        "    (by decide +kernel) (by decide +kernel) (by decide +kernel)\n"
    )
    lines = [
        "import ArchytasZX",
        "",
        "/-!",
        "# Two pair boxes over their node boxes",
        "",
        "Both sides are a pair box over the node box holding every node (`PDiagram`, `DPair`);",
        "`family` gives them equal at every copy count `k` (`pair_family`): each expansion is its",
        "template at each copy's slots times the pair nodes' entries at the slots the pin",
        "certificates tie the pair legs to; the templates are proved by the normaliser (`PT`) and",
        "moved to the environment `Ec`; `main` states it at the box multiplicity.",
        "-/",
        "",
        "namespace ArchytasZX.Export",
        "",
        "open ArchytasZX ArchytasZX.Diagram",
        "",
        "set_option maxRecDepth 100000",
        "set_option maxHeartbeats 2000000",
        "",
        _piece_source(src, "PT", True),
        f"def πT0 : List (ℕ × ℕ) := {pi0}\n\ndef πT1 : List (ℕ × ℕ) := {pi1}\n",
        "/-- The left side's pair box. -/",
        f"def pA : DPair := {_pair_lit(sa)}",
        "",
        "/-- The left side, its pair node off the diagram. -/",
        f"def PA : PDiagram :=\n  {pa.replace(_pair_lit(sa), 'pA')}",
        "",
        f"def bA : DBox := {ba}",
        "",
        "/-- The right side's pair box. -/",
        f"def pB : DPair := {_pair_lit(sb)}",
        "",
        "/-- The right side, its pair node off the diagram. -/",
        f"def PB : PDiagram :=\n  {pb.replace(_pair_lit(sb), 'pB')}",
        "",
        f"def bB : DBox := {bb}",
        "",
        "/-- Every label of dimension `d`, every phase zero. -/",
        f"noncomputable def Ec (d : ℕ) : Env := {ec}",
        "",
        "theorem okA : PA.toDiagram.PairOK pA bA = true := by decide +kernel",
        "",
        "theorem okB : PB.toDiagram.PairOK pB bB = true := by decide +kernel",
        "",
        "/-- The two templates. -/",
        (
            f"theorem sT {bind} :\n    Equal (Ec d) {a_t} (PA.toDiagram.tmpl pA) {b_t}"
            " (PB.toDiagram.tmpl pB) := by"
        ),
        moved,
        "/-- The two sides equal at every copy count. -/",
        f"theorem family {bind} (k : ℕ) :\n    {at('k')} :=",
        (
            "  (Diagram.pair_family okA okB (by decide) "
            f"{cert(cl_a)} {cert(cr_a)} {cert(cl_b)} {cert(cr_b)} {sl_a} {sr_a}\n    {pin}\n"
            f"    (Ec d) (fun _ => rfl) (sT {args}) 1 k).mul_trans (Equal.refl (Ec d) _ _)"
        ),
        "",
        "/-- The two sides equal at the box multiplicity. -/",
        f"theorem {name} {bind} ({nv} : ℕ) :\n    {at(kk)} :=",
        f"  family {args} {kk}",
        "",
        "end ArchytasZX.Export",
        "",
    ]
    return "\n".join(lines)


def _pair_lit(s: _PairSide) -> str:
    """The Lean ``DPair`` literal of a side."""
    return f"⟨1, 0, {_lp(s.left)}, {_lp(s.right)}, ⟨{s.fid}, .F, 0, 0, [0], [0]⟩⟩"
