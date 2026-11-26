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

"""Lean 4 network proofs of qudit path sums whose legs have dimension p^k
(ArchytasZX/PathSumQuditDigits.lean).

:func:`export_qudit_digits_net` writes both diagrams as networks over digit variables in Z_p: a
label of dimension p^k is k digit variables, low digit first; a label of dimension p is its own
digit variable. Z spiders on p^k, binders and splitters become digit constraints, Z, X and F on
dimension p the fragments of ArchytasZX/PathSumQuditNet.lean. Pin steps set the summed digits
that are not path variables to their anchors, each side's reduction trace is replayed and the
final comparison gives ``Equal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.pathsum import (
    ModPoly,
    PathSumUnsupported,
    QuditPathSum,
    canonical_qudit,
    reduce_qudit,
)
from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.lean_pathsum import (
    QNET_OPTIONS,
    QPIN_OFFSET,
    _legendre_model,
    _qpoly,
    _qudit_const_turn,
    _qudit_order,
    _qudit_steps,
    _qvar,
    _scalars,
    net_env,
)
from archytaszx.semantics.pathsum_diagram import from_diagram

LABEL_OFFSET = 1 << 19
"""The first label of a summed wire of more than one digit."""


@dataclass
class _Side:
    """One diagram's network over digit variables: nodes, fragments, fragment proofs, summed and
    boundary labels, each label's digits, the power of ``1 / sqrt p`` in its fragments, its steps,
    the reduced path sum, its canonical pivots and its scalar."""

    nodes: list[str]
    frags: list[str]
    proofs: list[str]
    summed: list[int]
    bnd: list[int]
    dig: dict[int, list[int]]
    inv_root: int
    steps: list[str]
    red: QuditPathSum
    order: list[int]
    scalar: sp.Expr


def _side(diagram: Diagram, phases: list[tuple[str, int]], sc: Any, label0: int) -> _Side:
    """The network of ``diagram`` over digit variables numbered as from_diagram's: boundary port
    i is the label of its first digit variable with its boundary digits; a summed wire of one
    digit is labelled by that digit, its digit the class's variable when that is a path variable
    not yet taken, else a variable from QPIN_OFFSET; a summed wire of k > 1 digits is label
    ``label0 + w``. Each Z spider's digit j is anchored at the leg through which a search from
    the class's variable first reaches it; the pin steps set the other summed digits to their
    anchors deepest first, then the trace of the pinned sum's reduction follows."""
    from archytaszx.diagram.generators import (
        DIM_BINDER,
        DIM_SPLITTER,
        FOURIER_BOX,
        X_SPIDER,
        Z_SPIDER,
    )
    from archytaszx.diagram.graph import Direction, NodeId, PortRef
    from archytaszx.semantics.pathsum_diagram import _qudit_table, _turns_table

    lay: dict[str, Any] = {}
    raw = from_diagram(diagram, lay)
    if not isinstance(raw, QuditPathSum):
        raise PathSumUnsupported("the qudit digit network export covers odd prime path sums only")
    if raw.atoms or raw.katoms:
        raise PathSumUnsupported("the qudit digit network export has no atoms")
    p = raw.p
    d: Diagram = lay["diagram"]
    digit: dict[PortRef, list[int]] = lay["digit"]
    names = {
        Z_SPIDER.name: ".Z",
        X_SPIDER.name: ".X",
        FOURIER_BOX.name: ".F",
        DIM_BINDER.name: ".B",
        DIM_SPLITTER.name: ".S",
    }
    label: dict[PortRef, int] = {}
    dig: dict[int, list[int]] = {}
    cls: dict[int, int] = {}
    bnd: list[int] = []
    v = 0
    for ref in (*d.boundary_outputs, *d.boundary_inputs):
        k = len(digit[ref])
        label[ref] = v
        dig[v] = list(range(v, v + k))
        for j, x in enumerate(dig[v]):
            cls[x] = digit[ref][j]
        bnd.append(v)
        v += k
    nb_vars = v
    taken: set[int] = set()
    summed: list[int] = []
    nxt = QPIN_OFFSET
    for w, wire in enumerate(sorted(d.wires, key=lambda w: w.sort_key())):
        if wire.a in label or wire.b in label:
            raise PathSumUnsupported("a wire on a boundary port")
        ds = []
        for cv in digit[wire.a]:
            if cv >= nb_vars and cv not in taken:
                x = cv
                taken.add(cv)
            else:
                x, nxt = nxt, nxt + 1
            ds.append(x)
            cls[x] = cv
        lab = ds[0] if len(ds) == 1 else label0 + w
        label[wire.a] = label[wire.b] = lab
        dig[lab] = ds
        summed.append(lab)
    owner = {x: lab for lab, ds in dig.items() for x in ds}
    k_of = {lab: len(ds) for lab, ds in dig.items()}
    outs: dict[NodeId, list[int]] = {}
    ins: dict[NodeId, list[int]] = {}
    for nid in sorted(d.nodes):
        node = d.nodes[nid]
        if node.generator_type.name not in names:
            raise PathSumUnsupported(f"node {nid!r} is a {node.generator_type.name}")
        if node.phase is not None and node.phase.is_weighted:
            raise PathSumUnsupported(f"node {nid!r} carries weights")
        n_o, n_i = len(node.legs(Direction.OUTPUT)), len(node.legs(Direction.INPUT))
        outs[nid] = [label[PortRef(nid, Direction.OUTPUT, j)] for j in range(n_o)]
        ins[nid] = [label[PortRef(nid, Direction.INPUT, j)] for j in range(n_i)]
        if not outs[nid] and not ins[nid]:
            raise PathSumUnsupported(f"node {nid!r} has no legs")
    edges: list[tuple[object, list[int]]] = []
    pair_of: dict[NodeId, tuple[int, int, int]] = {}
    for nid in sorted(d.nodes):
        kind = names[d.nodes[nid].generator_type.name]
        legs = outs[nid] + ins[nid]
        if kind == ".Z":
            ks = {k_of[lab] for lab in legs}
            if len(ks) != 1:
                raise PathSumUnsupported(f"node {nid!r} joins legs of different dimensions")
            for j in range(ks.pop()):
                edges.append((("z", nid, j), [dig[lab][j] for lab in legs]))
        elif kind in (".B", ".S"):
            whole, (a, b) = (outs[nid][0], ins[nid]) if kind == ".B" else (ins[nid][0], outs[nid])
            pair_of[nid] = (whole, a, b)
            for x, y in zip(dig[whole], dig[b] + dig[a], strict=True):
                edges.append((("b", nid), [x, y]))
        elif any(k_of[lab] != 1 for lab in legs):
            raise PathSumUnsupported(f"node {nid!r} is a {kind[1:]} on dimension {p}^k, k > 1")
    on: dict[int, list[int]] = {}
    for e, (_key, vs) in enumerate(edges):
        for x in vs:
            on.setdefault(x, []).append(e)
    entry: dict[int, int] = {}
    parent: dict[int, int] = {}
    reached: list[int] = []
    for root in sorted({x for x, c in cls.items() if x == c}):
        seen, queue = {root}, [root]
        while queue:
            x = queue.pop(0)
            for e in on.get(x, []):
                if e in entry:
                    continue
                entry[e] = x
                for y in edges[e][1]:
                    if y not in seen:
                        seen.add(y)
                        parent[y] = x
                        reached.append(y)
                        queue.append(y)
    if len(entry) != len(edges):
        raise PathSumUnsupported("a digit constraint not reached from its class's variable")
    anchor = {key: owner[entry[e]] for e, (key, _vs) in enumerate(edges)}
    xs = [nid for nid in sorted(d.nodes) if names[d.nodes[nid].generator_type.name] == ".X"]
    fresh = dict(zip(xs, sorted(raw.path)[len(raw.path) - len(xs) :], strict=True))
    common = "E σ own{T} labs{T} dig{T} hown hσ"
    nodes, frags, proofs = [], [], []
    phase = ModPoly(p)
    cons: list[ModPoly] = []
    inv_root = 0
    for nid in sorted(d.nodes):
        node = d.nodes[nid]
        kind = names[node.generator_type.name]
        os_, is_ = outs[nid], ins[nid]
        legs = os_ + is_
        if kind in (".B", ".S"):
            whole, a, b = pair_of[nid]
            r, q = k_of[a], k_of[b]
            nodes.append(f"⟨{kind}, {os_}, {is_}, {r}, {q}⟩")
            frags.append(f"bFragQ {p} dig{{T}} {whole} {a} {b}")
            lemma = "b_frag_q" if kind == ".B" else "s_frag_q"
            proofs.append(
                f"{lemma} {common} _ rfl {r} {q} rfl rfl {whole} {a} {b} rfl rfl (by decide) "
                "(by decide) (by decide) rfl rfl rfl"
            )
            cons += [
                _qvar(p, x) + _qvar(p, y).scale(-1)
                for x, y in zip(dig[whole], dig[b] + dig[a], strict=True)
            ]
            continue
        lift = f"lift_q {common} _ _ (by decide) (by decide) (fun σ' hσ' => @)"
        if kind == ".F":
            if os_[0] == is_[0]:
                raise PathSumUnsupported(f"node {nid!r} is an F on a self-loop")
            nodes.append(f"⟨.F, {os_}, {is_}, 1, 0⟩")
            frags.append(f"fFragQ {p} {os_[0]} {is_[0]}")
            proofs.append(
                lift.replace("@", f"f_frag_q (p := {p}) E σ' _ rfl rfl {os_[0]} {is_[0]} rfl rfl")
            )
            phase = phase + _qvar(p, os_[0]) * _qvar(p, is_[0])
            inv_root += 1
            continue
        k = k_of[legs[0]]
        if kind == ".Z":
            as_ = [anchor[("z", nid, j)] for j in range(k)]
            anchor_vars = [dig[a][j] for j, a in enumerate(as_)]
        else:
            anchor_vars = [fresh[nid]]
        table = _qudit_table(nid, node, anchor_vars, p)
        ts = []
        for t in _turns_table(nid, node, p**k):
            u = (t % 1) * p
            if u.denominator != 1:
                raise PathSumUnsupported(f"node {nid!r} has a phase {t} off the 1/{p} grid")
            ts.append(int(u))
        ph = 0
        if any(ts):
            phases.append((f"(({ts} : List ℕ).getD v 0 : ℕ)", p))
            ph = len(phases)
        ts_text = f"({ts if ph else []} : List ℕ)"
        nodes.append(f"⟨{kind}, {os_}, {is_}, {k}, {ph}⟩")
        poly = _qpoly(table)
        phase = phase + table
        if kind == ".Z" and k > 1:
            frags.append(f"zFragQD {p} dig{{T}} {as_} {legs} {poly}")
            proofs.append(
                f"z_frag_qd {common} _ rfl {k} rfl (by decide) (by decide) {as_} rfl "
                f"(by decide) {legs[0]} (by decide) {poly} {ts_text} (fun v => rfl) "
                "(by decide +kernel) (by decide) (by decide)"
            )
            for leg in legs:
                cons += [
                    _qvar(p, x) + _qvar(p, y).scale(-1)
                    for x, y in zip(anchor_vars, dig[leg], strict=True)
                ]
        elif kind == ".Z":
            var = anchor_vars[0]
            frags.append(f"zFragQ {p} {var} {legs} {poly}")
            proofs.append(
                lift.replace(
                    "@",
                    f"z_frag_q (p := {p}) E σ' hσ' _ rfl rfl {var} (by decide) {poly} "
                    f"{ts_text} (fun v => rfl) (by decide)",
                )
            )
            cons += [_qvar(p, var) + _qvar(p, leg).scale(-1) for leg in legs]
        else:
            y = anchor_vars[0]
            frags.append(f"xFragQ {p} {y} {poly} {os_} {is_}")
            proofs.append(
                lift.replace(
                    "@",
                    f"x_frag_q (p := {p}) E σ' hσ' _ rfl rfl {y} (by decide) {poly} {ts_text} "
                    "(fun v => rfl) (by decide)",
                )
            )
            for leg in os_:
                phase = phase + _qvar(p, y) * _qvar(p, leg)
            for leg in is_:
                phase = phase + (_qvar(p, y) * _qvar(p, leg)).scale(-1)
            inv_root += len(legs)
    path = [x for lab in summed for x in dig[lab]] + [fresh[nid] for nid in xs]
    steps: list[str] = []
    for x in reversed(reached):
        if x < QPIN_OFFSET:
            continue
        a = parent[x]
        hit = next(
            (
                q
                for q in cons
                if (c := q.terms.get(((x, 1),), 0))
                and x not in (q + ModPoly(p, {((x, 1),): -c % p})).variables()
                and (q + ModPoly(p, {((x, 1),): -c % p})).scale(-pow(c, -1, p)).terms
                == {((a, 1),): 1}
            ),
            None,
        )
        if hit is None:
            raise PathSumUnsupported(f"no constraint sets the digit {x} to {a}")
        cons.remove(hit)
        path.remove(x)
        steps += [f".perm {[x, *path]}", f".pin {a}"]
        value = _qvar(p, a)
        phase = phase.substitute(x, value)
        cons = [q.substitute(x, value) for q in cons]
    steps.append(".norm")
    if sorted(path) != sorted(raw.path):
        raise PathSumUnsupported("the pinned network keeps other path variables")
    sim = QuditPathSum(p, raw.boundary, set(path), phase, raw.const, [q for q in cons if q.terms])
    red = reduce_qudit(sim)
    order = _qudit_order(red.constraints, red.boundary)
    steps += _qudit_steps(sim, red, order)
    scalar = sc.model(d.scalar.to_sympy())
    if sc.opaques:
        raise PathSumUnsupported(f"the scalar {d.scalar} has no exact Lean form")
    return _Side(nodes, frags, proofs, summed, bnd, dig, inv_root, steps, red, order, scalar)


def _side_lines(tag: str, side: _Side, p: int) -> list[str]:
    """The digit tables, the network, its fragments and the QNetPSD facts of one side."""
    labels = sorted(side.dig)
    own = [(x, (lab, j)) for lab in labels for j, x in enumerate(side.dig[lab])]
    dig_tbl = [(lab, side.dig[lab]) for lab in labels]
    pairs = ", ".join(f"({n}, {f})" for n, f in zip(side.nodes, side.frags, strict=True))
    cases = " | ".join("rfl" for _ in side.nodes)
    return [
        f"def own{tag} : PSem.Own := {own}",
        f"def labs{tag} : List ℕ := {labels}",
        f"def dig{tag} (l : ℕ) : List ℕ := ((({dig_tbl} : List (ℕ × List ℕ))).lookup l).getD []",
        "",
        f"theorem own{tag}OK : PSem.ownOK own{tag} labs{tag} dig{tag} = true := by decide",
        "",
        f"def N{tag} : Net := ⟨[{', '.join(side.nodes)}], {side.summed}, {side.bnd}⟩",
        "",
        f"noncomputable def ps{tag} : List (Node × QFrag) := [{pairs.replace('{T}', tag)}]",
        "",
        f"theorem net{tag} : QNetPSD E N{tag} {p} own{tag} labs{tag} dig{tag} ps{tag} where",
        f"  ownok := own{tag}OK",
        "  nodes := rfl",
        "  node := by",
        "    intro q hq σ hσ",
        f"    have hown := own{tag}OK",
        f"    simp only [ps{tag}, List.mem_cons, List.not_mem_nil, or_false] at hq",
        f"    rcases hq with {cases}",
        *(f"    · exact {pr.replace('{T}', tag)}" for pr in side.proofs),
        "  disj := by decide",
        "  dims := by decide",
        "  cover := by decide",
        "  sub := by decide",
        "",
        (
            f"theorem c{tag} : (qjoinAll (ps{tag}.map Prod.snd)).c = "
            f"(rt {p})⁻¹ ^ {side.inv_root} := by"
        ),
        (
            f"  simp only [ps{tag}, List.map_cons, List.map_nil, qjoinAll, QFrag.join, QFrag.unit, "
            "zFragQ, xFragQ, fFragQ, zFragQD, bFragQ, List.length_cons, List.length_nil]"
        ),
        "  ring",
        "",
    ]


QDIGIT_DIAGRAM_LABEL0 = 1 << 26
"""The first wire label of the left diagram literal's network; the right one's is twice it."""


def _diagram_lines(
    left: Diagram,
    right: Diagram,
    sides: tuple[_Side, _Side],
    scalars: tuple[str, str],
    labs: dict[int, int],
    name: str,
) -> list[str]:
    """The theorem ``name_diagrams``: ``DEqual`` of the engine's diagrams at ``E`` with the
    label table ``labsD``, through ``Equal.relabel`` of ``name`` by the permutation ``πB`` of the
    boundary labels onto the slots, then each literal's network renamed onto the relabelled
    export's (``iso``) and ``DEqual_iff``; empty when either diagram has no literal."""
    from archytaszx.semantics.lean_pathsum import _swap, literal

    side_l, side_r = sides
    bnd = side_l.bnd
    pi_b = _swap(bnd, range(len(bnd)))
    ren = dict(pi_b)
    lits = [
        literal(tag, d, side.nodes, side.summed, labs, label0, "labsD", bnd)
        for tag, d, side, label0 in (
            ("L", left, side_l, QDIGIT_DIAGRAM_LABEL0),
            ("R", right, side_r, QDIGIT_DIAGRAM_LABEL0 << 1),
        )
    ]
    lit_l, lit_r = lits
    if lit_l is None or lit_r is None:
        return []
    if any(x in ren for side in sides for x in side.summed):
        return []
    table = {ren.get(x, x): labs[x] for side in sides for x in (*side.bnd, *side.summed)}
    for side, lit in ((side_l, lit_l), (side_r, lit_r)):
        for j, x in enumerate(side.summed):
            table[lit.off + j] = labs[x]
    eb = "{ E with labs := labsD }"

    def iso(tag: str, s: str, lit: Any, side: _Side) -> str:
        order = list(range(len(side.nodes)))
        return (
            f"(iso {eb} ({s}) (D{tag}.toNet {lit.off}) (N{tag}.relabel (ren πB)) {lit.perm} "
            f"{order} (by decide +kernel))"
        )

    s_l, s_r = scalars
    return [
        f"def πB : List (ℕ × ℕ) := {pi_b}",
        "",
        f"def labsD : List (ℕ × ℕ) := {sorted(table.items())}",
        "",
        *lit_l.lines,
        *lit_r.lines,
        "/-- The theorem on the engine's diagrams, through `Equal.relabel`, `iso` and",
        "`DEqual_iff`. -/",
        f"theorem {name}_diagrams : DEqual {eb} ({s_l}) DL ({s_r}) DR :=",
        f"  (DEqual_iff (o₁ := {lit_l.off}) (o₂ := {lit_r.off}) (L := labsD) rfl DL_chk DR_chk).2",
        f"    (({iso('L', s_l, lit_l, side_l)}.trans",
        "      (Equal.relabel E labsD (ren πB) (ren_injective πB (by decide) (by decide))",
        f"        (by decide +kernel) (by decide +kernel) {name})).trans",
        f"      {iso('R', s_r, lit_r, side_r)}.symm)",
        "",
    ]


def export_qudit_digits_net(
    left: Diagram, right: Diagram, name: str = "main", diagrams: bool = False
) -> str:
    """Lean 4 source proving ``Equal E sL NL sR NR`` for the networks of two diagrams of Z, X,
    F, binder and splitter nodes on the powers of one odd prime p, some leg of dimension p^k with
    k > 1 (ArchytasZX/PathSumQuditDigits.lean): each network's value is its fragments' path sum
    over digit variables (QNetPSD.eval_eq), pin steps set the summed digits of one class to one
    variable, each side's reduction trace is replayed, the Gauss sums take their Legendre form
    and the final comparison gives ``Equal``; raises :class:`PathSumUnsupported` outside that
    class or when the sums do not reduce to one canonical form, :class:`ScalarUnsupported` when
    the constants' identity has no proof; with ``diagrams``, then ``name_diagrams`` over the
    engine's diagrams (:func:`_diagram_lines`)."""
    from archytaszx.algebra.pathsum import legendre
    from archytaszx.semantics.lean_scalar import Inv, Om, Rt

    sc = _scalars()
    phases: list[tuple[str, int]] = []
    side_l = _side(left, phases, sc, LABEL_OFFSET)
    side_r = _side(right, phases, sc, LABEL_OFFSET + (1 << 17))
    if all(len(ds) == 1 for side in (side_l, side_r) for ds in side.dig.values()):
        raise PathSumUnsupported("no leg of dimension p^k, k > 1")
    p = side_l.red.p
    if side_r.red.p != p or side_l.red.boundary != side_r.red.boundary:
        raise PathSumUnsupported("the two path sums have different bases or boundaries")
    if side_l.bnd != side_r.bnd:
        raise PathSumUnsupported("the two networks have different boundary labels")
    labs: dict[int, int] = {}
    for side in (side_l, side_r):
        for lab, ds in side.dig.items():
            if labs.setdefault(lab, len(ds)) != len(ds):
                raise PathSumUnsupported(f"the label {lab} has two dimensions")
    bvars = [x for lab in side_l.bnd for x in side_l.dig[lab]]
    if any(side_r.dig[lab] != side_l.dig[lab] for lab in side_l.bnd):
        raise PathSumUnsupported("the two networks have different boundary digits")
    can_l, can_r = canonical_qudit(side_l.red), canonical_qudit(side_r.red)
    if can_l is None or can_r is None or not can_l.same(can_r) or can_l.zero:
        raise PathSumUnsupported("the qudit path sums do not reduce to one canonical form")
    if side_l.order != side_r.order:
        raise PathSumUnsupported("the qudit elimination orders differ")
    k_l = _qudit_const_turn(side_l.red, side_l.order)
    k_r = _qudit_const_turn(side_r.red, side_r.order)
    n = 12 if p == 3 else p
    sub = f"rw [om_sub {n} {p} {n // p} (by norm_num) (by norm_num)]" if n != p else "skip"

    def fac(steps: list[str]) -> tuple[str, sp.Expr]:
        texts, model = [], sp.Integer(1)
        for st in steps:
            if st in (".elim", ".hhCons", ".hhPivot"):
                texts.append(f"({p} : ℂ)")
                model *= p
            elif st.startswith(".gauss "):
                a = int(st.split()[1])
                texts.append(f"qg {p} {a}")
                model *= _legendre_model(a, p, n)
        return (" * ".join(texts) or "1"), model

    (ft_l, fm_l), (ft_r, fm_r) = fac(side_l.steps), fac(side_r.steps)
    s_l, s_r = sc.text(side_l.scalar), sc.text(side_r.scalar)
    root = Inv(Rt(sp.Integer(p)))
    lhs = fm_l * side_l.scalar * root**side_l.inv_root * Om(n) ** (k_l * n // p)
    rhs = fm_r * side_r.scalar * root**side_r.inv_root * Om(n) ** (k_r * n // p)
    proof = sc.prove(lhs, rhs)
    steps_all = (*side_l.steps, *side_r.steps)
    gauss = sorted({int(st.split()[1]) for st in steps_all if st.startswith(".gauss")})
    lg_vals = {k: legendre(k, p) if k % p else 0 for k in range(p)}
    body = [
        *_side_lines("L", side_l, p),
        *_side_lines("R", side_r, p),
        f"def stepsL : List QSem.Step := [{', '.join(side_l.steps)}]",
        f"def stepsR : List QSem.Step := [{', '.join(side_r.steps)}]",
        "",
        *(f"theorem lg{k} : lg {p} {k} = {v} := by decide" for k, v in sorted(lg_vals.items())),
        "",
    ]
    for a in gauss:
        body += [
            f"theorem qg{a} : qg {p} {a} = {sc.text(_legendre_model(a, p, n))} := by",
            f"  rw [qg_legendre (by norm_num) {a} (by decide)]",
            "  simp only [Finset.sum_range_succ, Finset.sum_range_zero, "
            + ", ".join(f"lg{k}" for k in sorted(lg_vals))
            + "]",
            f"  {sub}",
            "  push_cast",
            "  ring",
            "",
        ]
    for tag, text in (("L", ft_l), ("R", ft_r)):
        body += [
            f"theorem fac{tag} : QSem.facs {p} steps{tag} = {text} := by",
            f"  simp only [steps{tag}, QSem.facs, QSem.fac]",
            "  push_cast",
            "  ring",
            "",
        ]
    body += [
        (
            f"theorem consts : ({ft_l}) * (({s_l}) * (rt {p})⁻¹ ^ {side_l.inv_root}) * "
            f"om {p} ^ {k_l} ="
        ),
        f"    ({ft_r}) * (({s_r}) * (rt {p})⁻¹ ^ {side_r.inv_root}) * om {p} ^ {k_r} := by",
        f"  have key : ({sc.text(lhs)} : ℂ) = {sc.text(rhs)} :=",
        *("    " + ln for ln in proof.splitlines()),
        *(f"  all_goals rw [qg{a}]" for a in gauss),
        f"  all_goals {sub}",
        "  all_goals linear_combination key",
        "",
        "/-- The two networks are equal. -/",
        f"theorem {name} : Equal E ({s_l}) NL ({s_r}) NR := by",
        f"  apply equal_qpsd2 _ _ netL netR {bvars} (by decide) (by decide) rfl",
        "  intro τ",
        "  rw [← QSem.val_smul, ← QSem.val_smul,",
        "    QSem.run_sound stepsL (qnetDataD digL NL (psL.map Prod.snd)) _ τ (by decide),",
        "    QSem.run_sound stepsR (qnetDataD digR NR (psR.map Prod.snd)) _ τ (by decide)]",
        f"  apply QSem.final_sound _ _ _ _ τ {k_l} {k_r} (by decide) (by decide) (by decide)",
        "  rw [facL, facR, cL, cR]",
        "  all_goals exact consts",
        "",
    ]
    extra = (
        _diagram_lines(left, right, (side_l, side_r), (s_l, s_r), labs, name) if diagrams else []
    )
    ns = "ArchytasZX.PathSumQuditDigitsProof"
    lines = [
        "import ArchytasZX.PathSumQuditDigits",
        *(["import ArchytasZX.NetRelabel"] if extra else []),
        "",
        "/-!",
        f"# Qudit digit network proof `{name}` over Z_{p}",
        "",
        "Each network's value is its fragments' path sum over digit variables in Z_p; pin steps",
        "set the summed digits of one class to one variable, the reduction steps follow, and the",
        "final comparison gives `Equal`.",
        "-/",
        "",
        f"namespace {ns}",
        "",
        "open ArchytasZX ArchytasZX.QSem",
        "",
        *QNET_OPTIONS,
        "",
        f"instance : Fact (Nat.Prime {p}) := ⟨by norm_num⟩",
        "",
        *net_env(phases, labs, f"{p} ^ k", p),
        *body,
        *extra,
        f"end {ns}",
        "",
    ]
    return "\n".join(lines)


__all__ = ["export_qudit_digits_net"]
