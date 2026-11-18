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

:func:`export_copy_phase` writes that family theorem and restates it as
``main_diagrams_original`` over the engine's diagrams as Lean linked diagrams ``L0``, ``L1`` with
the gadget undone, ``x`` at copy ``t`` reading phase index ``Nat.pair P t`` of the table ``Ψ``:
the family's phases below ``P``, ``s t k / d`` at ``Nat.pair P t``
(``LDiagram.evalL_eq_of_gad``). The text holds no copy count.
"""

from __future__ import annotations

import re
from collections.abc import Callable
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


_PORT = re.compile(r"⟨(\d+), (true|false), (\d+)⟩")
_NODE = re.compile(r"⟨(\d+), \.(\w+), (\d+), (\d+), \[([\d, ]*)\], \[([\d, ]*)\]⟩")
_BOX = re.compile(r"⟨(\d+), (\d+), (none|some \d+), \[([\d, ]*)\], \[\]⟩")

LPort = tuple[int, bool, int]


@dataclass
class _LDiag:
    """A boxed diagram literal: nodes ``(id, kind, dim, ph, outs, ins)``, wires, outputs, inputs
    and boxes ``(id, mult, parent, nodes)``."""

    nodes: list[tuple[int, str, int, int, list[int], list[int]]]
    wires: list[tuple[LPort, LPort]]
    outs: list[LPort]
    ins: list[LPort]
    boxes: list[tuple[int, int, str, list[int]]]


def _nats(text: str) -> list[int]:
    return [int(v) for v in text.replace(" ", "").split(",") if v]


def _ports(text: str) -> list[LPort]:
    return [(int(a), b == "true", int(c)) for a, b, c in _PORT.findall(text)]


def _field(block: str, name: str, nxt: str) -> str:
    start = block.index(f"{name} := [") + len(name) + 5
    return block[start : block.index(nxt, start)]


def _parse(source: str, name: str) -> _LDiag:
    """The literal ``def name : Diagram`` of ``source``."""
    m = re.search(rf"^def {name} : Diagram :=\n(.*?)\}}\n", source, re.MULTILINE | re.DOTALL)
    if m is None:
        raise NetUnsupported(f"no boxed diagram {name} in the export")
    block = m[1]
    nodes = [
        (int(a), k, int(b), int(c), _nats(o), _nats(i))
        for a, k, b, c, o, i in _NODE.findall(_field(block, "nodes", "],\n"))
    ]
    wp = _ports(_field(block, "wires", "],\n"))
    wires = list(zip(wp[0::2], wp[1::2], strict=True))
    outs = _ports(_field(block, "outs", "],\n"))
    ins = _ports(_field(block, "ins", "],\n"))
    boxes = [
        (int(a), int(b), p, _nats(ns))
        for a, b, p, ns in _BOX.findall(block[block.index("boxes") :])
    ]
    return _LDiag(nodes, wires, outs, ins, boxes)


def _lport(p: LPort) -> str:
    return f"⟨{p[0]}, {'true' if p[1] else 'false'}, {p[2]}⟩"


def _lnat(xs: list[int]) -> str:
    return "[" + ", ".join(map(str, xs)) + "]"


@dataclass(frozen=True, slots=True)
class _Side:
    """One side undone: its linked diagram text, the node ``x`` as a literal, the Fourier box
    and the gadget wire's place (``0`` for an X spider), and the zero phase index."""

    literal: str
    xnode: str
    f: int
    i: int
    z: int
    x_dim: int


def _undo(D: _LDiag, g: Gadget, ph: int) -> _Side:
    """``D``, the gadget diagram's literal, with the gadget undone and ``x`` at phase index
    ``ph`` read at its box's copy index."""
    x = int(g.node)
    k_port = D.ins[-1]
    nodes = {n[0]: n for n in D.nodes}
    xn = nodes[x]
    f, i = 0, 0
    wires = list(D.wires)
    if g.box is not None:
        f = int(g.box)
        fo = (f, True, 0)
        i = next(j for j, w in enumerate(wires) if w == (fo, (x, False, len(xn[5]) - 1)))
        del wires[i]
        if k_port != (f, False, 0) or nodes[f][3] != xn[3]:
            raise NetUnsupported("the gadget's Fourier box is not read as written")
    elif k_port != (x, False, len(xn[5]) - 1):
        raise NetUnsupported("the gadget's boundary input is not read as written")
    z = xn[3]
    x0 = (x, xn[1], xn[2], ph, xn[4], xn[5][:-1])
    lit_nodes = []
    for n in D.nodes:
        if g.box is not None and n[0] == f:
            continue
        m = x0 if n[0] == x else n
        lit_nodes.append(f"⟨{m[0]}, .{m[1]}, {m[2]}, {m[3]}, {_lnat(m[4])}, {_lnat(m[5])}⟩")
    boxes = []
    bid = -1
    for b, mult, par, ns in D.boxes:
        kept = [v for v in ns if g.box is None or v != f]
        boxes.append(f"⟨{b}, {mult}, {par}, {_lnat(kept)}, []⟩")
        if x in ns:
            bid = b
    if bid < 0:
        raise NetUnsupported("the gadget node lies in no box")
    literal = (
        "{ base :=\n      { nodes := [" + ", ".join(lit_nodes) + "],\n"
        "        wires := [" + ", ".join(f"({_lport(a)}, {_lport(b)})" for a, b in wires) + "],\n"
        "        outs := [" + ", ".join(map(_lport, D.outs)) + "],\n"
        "        ins := [" + ", ".join(map(_lport, D.ins[:-1])) + "],\n"
        "        boxes := [" + ", ".join(boxes) + "] },\n"
        f"    cph := [({x}, {bid}, 0, 0)] }}"
    )
    xnode = f"⟨{x0[0]}, .{x0[1]}, {x0[2]}, {x0[3]}, {_lnat(x0[4])}, {_lnat(x0[5])}⟩"
    return _Side(literal, xnode, f, i, z, xn[2])


def _groups(text: str) -> list[str]:
    """The top-level bracket groups of ``text``."""
    out, depth, start = [], 0, 0
    for k, ch in enumerate(text):
        if ch in "([{⟨":
            if depth == 0:
                start = k
            depth += 1
        elif ch in ")]}⟩":
            depth -= 1
            if depth == 0:
                out.append(text[start : k + 1])
    return out


_MAIN = re.compile(
    r"^theorem main_diagrams :\n    ∀ (?P<binders>[^\n]*),\n      DEqual (?P<rest>[^\n]*) := by\n"
    r"  intro (?P<intro>[^\n]*)\n"
    r"  exact Diagram\.DEqual_expandK_of_checks (?P<ks>\([^\n]*?\)) _ _\n"
    r"    (?P<ok0>\(Diagram\.FamOK_congr [^\n]* D0_ok\))\n"
    r"    (?P<ok1>\(Diagram\.FamOK_congr [^\n]* D1_ok\))\n"
    r"    \(by decide \+kernel\) \(by decide \+kernel\) (?P<pos>\(Fpos [^\n]*\))\n"
    r"    (?P<eq>\(D0_net ▸ D1_net ▸ [^\n]*\))\n",
    re.MULTILINE,
)


def export_copy_phase(
    left: Diagram, right: Diagram, describe: Callable[[Diagram], str] | None = None
) -> str:
    """The normaliser's family theorem of the gadget pair (:func:`gadget`), restated as
    ``main_diagrams_original`` over the engine's diagrams as linked diagrams: ``x`` at copy
    ``t`` reads phase index ``Nat.pair P t``, the phase table ``Ψ`` reading it as ``s t k / d``
    above the family's phases (``LDiagram.evalL_eq_of_gad``). Raises :class:`NetUnsupported`
    where :func:`gadget` or the normaliser declines."""
    from archytaszx.semantics import lean_sym

    ga, gb = gadget(left), gadget(right)
    source = lean_sym.export_symbolic(ga.diagram, gb.diagram, "main", describe)
    m = _MAIN.search(source)
    phi = re.search(r"^noncomputable def Φ ([^\n]*?) ?: ℕ → ℕ → ℝ$", source, re.MULTILINE)
    if m is None or phi is None:
        raise NetUnsupported("the gadget pair's export has no family theorem on its diagrams")
    rest = _groups(m["rest"])
    if len(rest) != 5:
        raise NetUnsupported("the gadget pair's family theorem does not read as written")
    env, s0, _, s1, _ = rest
    ks = m["ks"]
    params = phi[1].strip()
    args = " ".join(re.findall(r"\(([^\s:]+) : ℕ\)", params))
    dname = str(ga.diagram.nodes[ga.node].inputs[0].dim.to_sympy())
    if dname not in args.split():
        raise NetUnsupported("the index phase's dimension is not a family parameter")
    D0, D1 = _parse(source, "D0"), _parse(source, "D1")
    top = max(n[3] for n in [*D0.nodes, *D1.nodes]) + 1
    a, b = _undo(D0, ga, top), _undo(D1, gb, top + 1)
    if a.z != b.z:
        raise NetUnsupported("the gadget pair's phaseless indices differ")
    sg = {top: ga.sign, top + 1: gb.sign}
    phase = f"Ψ {args}".strip()
    er = f"{{ {env} with phase := {phase} }}"
    intro = m["intro"]
    sig = params + " (ks : ℕ → ℕ)"
    fcall = f"(F {args})" if args else "F"

    def sgn(v: int) -> str:
        return "(1 : ℝ)" if v > 0 else "(-1 : ℝ)"

    body = (
        f"(∀ i (hi : i < L0.base.bnd.length), ∀ t ∈ cidx {ks} (L0.base.legChain L0.base.bnd[i]),\n"
        f"        σ i t < {env}.dimv (L0.base.portDim L0.base.bnd[i])) →\n"
        f"      {s0} * L0.evalL {er} {ks} σ =\n"
        f"        {s1} * L1.evalL {er} {ks} σ"
    )
    desc = (lambda d: "") if describe is None else describe
    extra = (
        "\n/-- The phase table of the engine's diagrams: the family's below "
        f"{top}, a phase index `Nat.pair p t` at or above it read as `s t k / {dname}`. -/\n"
        f"noncomputable def Ψ {params} (p : ℕ) : ℕ → ℝ :=\n"
        f"  if p < {top} then Φ {args} p else fun k =>\n"
        f"    (if (Nat.unpair p).1 = {top} then {sgn(sg[top])} else {sgn(sg[top + 1])}) *\n"
        f"      (((Nat.unpair p).2 : ℝ) * k / {dname})\n\n"
        f"theorem dimx {sig} : ({fcall}.env ks).dimv {a.x_dim} = {dname} := by\n"
        "  simp [Fam.env, F, Fam.dimv, bundL, List.lookup, Δ]\n\n"
    )
    for tag, side, d0, fam in (("0", a, left, "D0"), ("1", b, right, "D1")):
        note = desc(d0)
        extra += (
            f"/-- {note + ', as ' if note else ''}the engine's boxed diagram with its copy-indexed "
            f"phase. -/\ndef L{tag} : LDiagram :=\n  {side.literal}\n\n"
            f"/-- The node of L{tag} reading its copy index. -/\n"
            f"def X{tag} : DNode := {side.xnode}\n\n"
            f"theorem G{tag} : LDiagram.Gad L{tag} {fam} X{tag} {side.f} {side.z} {side.i} := by "
            "decide +kernel\n\n"
            f"theorem W{tag} : WF L{tag}.base := Diagram.WF_of_WFb (by decide +kernel)\n\n"
        )
    offs = re.findall(r"^theorem D([01])_ok : D[01]\.FamOK FC (\d+) :=", source, re.MULTILINE)
    if [t for t, _ in offs] != ["0", "1"]:
        raise NetUnsupported("the gadget pair's export has no family checks")
    o0, o1 = offs[0][1], offs[1][1]
    for tag in ("0", "1"):
        extra += (
            f"theorem CPh{tag} {params} : ∀ m ∈ L{tag}.base.nodes, m.id ≠ X{tag}.id →\n"
            f"    {phase} m.ph = {fcall}.phase m.ph := by\n"
            "  intro m hm hne\n"
            f"  have hp := (by decide : ∀ m ∈ L{tag}.base.nodes, m.id ≠ X{tag}.id →\n"
            f"    m.ph < {top}) m hm hne\n"
            "  unfold Ψ\n  rw [if_pos hp]\n  rfl\n\n"
            f"theorem CPx{tag} {sig} : ∀ u k, ex ({phase} (Nat.pair X{tag}.ph u) k) =\n"
            f"    ex (LDiagram.gsg X{tag}.kind *\n"
            f"      ((u : ℝ) * k / ({fcall}.env ks).dimv X{tag}.dim)) := by\n"
            f"  intro u k\n  rw [show X{tag}.dim = {a.x_dim} from rfl, dimx {args} ks]\n"
            "  unfold Ψ\n"
            f"  rw [if_neg (not_lt.2 (le_trans (by decide : {top} ≤ X{tag}.ph)\n"
            "    (Nat.left_le_pair _ _))), Nat.unpair_pair]\n"
            f"  simp [LDiagram.gsg, X{tag}]\n\n"
        )
    extra += (
        f"theorem CPz {params} : ∀ k, {fcall}.phase {a.z} k = 0 := fun k => by simp [F, Φ]\n\n"
        "theorem CPb : D0.bndData = D1.bndData := by decide +kernel\n\n"
        "theorem CPs : L0.base.nodeChain X0.id = L1.base.nodeChain X1.id ∧ X0.dim = X1.dim ∧\n"
        "    L0.rdOf = L1.rdOf ∧ L0.base.bndData = L1.base.bndData := by decide +kernel\n\n"
        "/-- The family theorem on the engine's diagrams with their copy-indexed phases, through "
        "`LDiagram.evalL_eq_of_gad`. -/\n"
        "theorem main_diagrams_original :\n"
        f"    ∀ {m['binders']} (σ : ℕ → List ℕ → ℕ),\n      {body} := by\n"
        f"  intro {intro} σ hσ\n"
        f"  have hE : Equal {env} {s0} (D0.toNetB {o0}) {s1} (D1.toNetB {o1}) :=\n"
        f"    {m['eq']}\n"
        f"  exact LDiagram.evalL_eq_of_gad (F := {fcall}) {ks} {s0} {s1} G0 G1 W0 W1\n"
        f"    {m['ok0']}\n    {m['ok1']}\n"
        f"    CPb {m['pos']} CPs {er} rfl rfl rfl (CPh0 {args}) (CPh1 {args}) (CPz {args})\n"
        f"    (CPx0 {args} {ks}) (CPx1 {args} {ks}) hE σ hσ\n"
    )
    end = source.rindex("\nend ")
    return source[:end].rstrip() + "\n" + extra + source[end:]
