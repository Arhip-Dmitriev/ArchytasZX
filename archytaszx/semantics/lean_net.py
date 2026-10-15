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

"""Lean 4 export of kernel theorems over the tensor-network semantics of the bundled Lean
library ``ArchytasZX`` (``archytaszx/lean``).

Every diagram becomes a ``Net``: one ``Node`` per diagram node, legs labelled by wire, boundary
slot ``i`` labelled ``i`` (outputs first, then inputs), wires labelled from a counter shared by
the whole export. Dimensions, phase vectors and the label dimensions go into one ``Env``. The
exported theorem states ``Equal E s N s' N'``: the two scaled networks are the same tensor.

Proof pieces:

* ``iso`` -- relabelling, node reordering and spider leg order, decided by ``isoOK``;
* a rule step -- ``iso`` into ``C ++ L``, ``rewrite`` with the rule's lemma, ``wire_elim`` per
  bare wire the rule leaves, ``iso`` onto the next diagram's network;
* ``Equal.symm`` and ``Equal.trans``.

A rule without a lemma in ``ArchytasZX/Rules.lean``, or an isomorphism the exporter cannot
match, becomes an axiom stating that one step over the same semantics, named ``unproved_*``.
:class:`NetUnsupported` is raised for a theorem outside the semantics: free multiplicities,
bang boxes, induction, pending parameters, dangling ports, disagreeing wire dimensions, or a
scalar, dimension or phase this module cannot write in Lean.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import INDEX, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.compare import isomorphism
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef, Wire
from archytaszx.rewrite.engine import RewriteStep
from archytaszx.semantics.denote import resolve_dim

__all__ = ["NetUnsupported", "check", "export", "lean_project"]


class NetUnsupported(Exception):
    """A theorem or diagram outside the tensor-network export."""


_KINDS = {"Z": "Z", "X": "X", "F": "F", "T": "T", "Ti": "Ti", "W": "W", "B": "B", "S": "S"}
_RESERVED = frozenset(
    {"E", "Φ", "Δ", "labs", "k", "at", "by", "do", "fun", "in", "let", "if", "then", "else"}
    | {"end", "from", "have", "show", "with", "match", "theorem", "def", "axiom", "open"}
    | {"ex", "om", "fou", "Net", "Node", "Env", "Equal", "I", "π", "pi", "exp", "sqrt"}
)


@dataclass(frozen=True)
class LNode:
    """One network node: kind, output and input labels, dimension and phase index."""

    kind: str
    outs: tuple[int, ...]
    ins: tuple[int, ...]
    dim: int
    ph: int

    def lean(self) -> str:
        """The Lean ``Node`` literal."""
        return f"⟨.{self.kind}, {_list(self.outs)}, {_list(self.ins)}, {self.dim}, {self.ph}⟩"

    def labels(self) -> tuple[int, ...]:
        """Outputs then inputs."""
        return self.outs + self.ins

    def rename(self, f: Callable[[int], int]) -> LNode:
        """Every label through ``f``."""
        return replace(self, outs=tuple(map(f, self.outs)), ins=tuple(map(f, self.ins)))


@dataclass(frozen=True)
class LNet:
    """Nodes, summed labels and boundary labels."""

    nodes: tuple[LNode, ...]
    summed: tuple[int, ...]
    bnd: tuple[int, ...]

    def lean(self) -> str:
        """The Lean ``Net`` literal."""
        return f"⟨{_nodes(self.nodes)}, {_list(self.summed)}, {_list(self.bnd)}⟩"


@dataclass
class Labelled:
    """A diagram's network, with the node order and every port's label."""

    net: LNet
    order: tuple[NodeId, ...]
    port: dict[PortRef, int]
    name: str
    scalar_note: str


@dataclass(frozen=True)
class Proof:
    """A Lean term proving ``Equal E left lhs right rhs``."""

    term: str
    left: object
    right: object


def _list(xs: Sequence[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def _nodes(ns: Sequence[LNode]) -> str:
    return "[" + ", ".join(n.lean() for n in ns) + "]"


def _pairs(ps: Sequence[tuple[int, int]]) -> str:
    return "[" + ", ".join(f"({a}, {b})" for a, b in ps) + "]"


def _comment(text: str) -> str:
    return text.replace("-/", "- /").replace("/-", "/ -")


def _erase(xs: Sequence[int], x: int) -> tuple[int, ...]:
    out = list(xs)
    out.remove(x)
    return tuple(out)


def _ren(pairs: Sequence[tuple[int, int]]) -> Callable[[int], int]:
    table: dict[int, int] = {}
    for a, b in pairs:
        table.setdefault(a, b)
    return lambda x: table.get(x, x)


def _perm_to_bijection(mapping: dict[int, int]) -> list[tuple[int, int]]:
    """``mapping`` extended to a bijection of its keys and values, as pairs."""
    keys = list(mapping)
    values = set(mapping.values())
    if len(values) != len(keys):
        raise NetUnsupported("the label renaming is not injective")
    spare_from = [v for v in sorted(values) if v not in mapping]
    spare_to = [k for k in keys if k not in values]
    pairs = [(k, mapping[k]) for k in keys]
    pairs += list(zip(spare_from, spare_to, strict=True))
    return [(a, b) for a, b in pairs if a != b]


class _Sym:
    """Sympy expressions written as Lean terms over the export's parameters."""

    def __init__(self) -> None:
        self.params: dict[str, tuple[str, str]] = {}

    def name(self, sym: sp.Symbol, kind: str) -> str:
        raw = str(sym.name)
        if raw in self.params:
            return self.params[raw][0]
        plain = re.sub(r"[^A-Za-z0-9_]", "_", raw)
        if not re.match(r"[A-Za-z]", plain) or plain in _RESERVED:
            plain = "p_" + plain
        taken = {v[0] for v in self.params.values()}
        while plain in taken:
            plain += "'"
        self.params[raw] = (plain, kind)
        return plain

    def kind_of(self, sym: sp.Symbol, default: str) -> str:
        raw = str(sym.name)
        return self.params[raw][1] if raw in self.params else default

    def render(self, expr: sp.Expr, ty: str, sym_kind: str) -> str:
        """``expr`` as a Lean term of type ``ty`` (ℕ, ℝ or ℂ)."""
        expr = sp.sympify(expr)
        if expr == 1:
            return "1"
        if expr.is_Integer:
            v = int(expr)
            return f"({v} : {ty})" if v >= 0 else f"(-{-v} : {ty})"
        if expr.is_Rational:
            if ty == "ℕ":
                raise NetUnsupported(f"a rational {expr} where a natural number is required")
            return f"(({expr.p} : {ty}) / {expr.q})"
        if expr is sp.I:
            if ty != "ℂ":
                raise NetUnsupported("the imaginary unit outside ℂ")
            return "Complex.I"
        if expr is sp.pi:
            if ty == "ℕ":
                raise NetUnsupported("π where a natural number is required")
            return "(Real.pi : ℂ)" if ty == "ℂ" else "Real.pi"
        if expr.is_Symbol:
            kind = self.kind_of(expr, sym_kind)
            name = self.name(expr, kind)
            if kind == ty:
                return name
            if (kind, ty) in {("ℕ", "ℝ"), ("ℕ", "ℂ"), ("ℝ", "ℂ")}:
                return f"({name} : {ty})"
            raise NetUnsupported(f"symbol {expr} of type {kind} used as {ty}")
        if expr.is_Add:
            return "(" + " + ".join(self.render(a, ty, sym_kind) for a in expr.args) + ")"
        if expr.is_Mul:
            return "(" + " * ".join(self.render(a, ty, sym_kind) for a in expr.args) + ")"
        if expr.is_Pow:
            base, exp = expr.args
            if exp.is_Integer and int(exp) >= 0:
                return f"({self.render(base, ty, sym_kind)} ^ {int(exp)})"
            if ty == "ℕ":
                raise NetUnsupported(f"{expr} is not a natural number")
            if exp.is_Integer:
                return f"({self.render(base, ty, sym_kind)} ^ {-int(exp)})⁻¹"
            if exp.is_Rational and exp.q == 2:
                root = f"(Real.sqrt {self.render(base, 'ℝ', sym_kind)} : {ty})"
                if ty == "ℝ":
                    root = f"Real.sqrt {self.render(base, 'ℝ', sym_kind)}"
                power = int(exp.p)
                if power == 1:
                    return f"({root})"
                if power > 0:
                    return f"(({root}) ^ {power})"
                return f"(({root}) ^ {-power})⁻¹"
            raise NetUnsupported(f"the power {expr}")
        if isinstance(expr, sp.exp):
            if ty != "ℂ":
                raise NetUnsupported("an exponential outside ℂ")
            return f"Complex.exp {self.render(expr.args[0], 'ℂ', sym_kind)}"
        raise NetUnsupported(f"cannot write {expr} ({type(expr).__name__}) in Lean")


class _Export:
    """Accumulates the tables, networks and proofs of one export."""

    def __init__(self, describe: Callable[[Diagram], str]) -> None:
        self.describe = describe
        self.sym = _Sym()
        self.dims: list[str] = []
        self.dim_key: dict[str, int] = {}
        self.phases: list[str] = ["fun _ => 0"]
        self.phase_key: dict[str, int] = {"": 0}
        self.labs: dict[int, int] = {}
        self.nb: int | None = None
        self.next_label = 0
        self.nets: dict[object, Labelled] = {}
        self.decls: list[str] = []
        self.count = 0
        self.unproved: list[str] = []

    # tables

    def dim(self, d: Dim) -> int:
        key = str(d.to_sympy())
        if key not in self.dim_key:
            self.dim_key[key] = len(self.dims)
            self.dims.append(self.sym.render(d.to_sympy(), "ℕ", "ℕ"))
        return self.dim_key[key]

    def phase(self, pv: PhaseVector | None) -> int:
        if pv is None or pv.is_zero:
            return 0
        parts = []
        for index, ph in sorted(pv.entries().items()):
            parts.append((index, self.sym.render(ph.to_sympy_turns(), "ℝ", "ℝ")))
        body = "0"
        for index, text in reversed(parts):
            body = f"if k = {index} then {text} else {body}"
        if pv.rule is not None:
            k = sp.Symbol("k_lean_index", integer=True, nonnegative=True)
            turns = pv.rule.to_sympy_turns().xreplace({INDEX: k})
            self.sym.params.setdefault(str(k.name), ("(k : ℝ)", "ℝ"))
            rule = self.sym.render(turns, "ℝ", "ℝ")
            body = f"({body}) + (if k = 0 then 0 else {rule})"
        key = body
        if key not in self.phase_key:
            self.phase_key[key] = len(self.phases)
            self.phases.append(f"fun k => {body}")
        return self.phase_key[key]

    def label(self, dim: int) -> int:
        lab = self.next_label
        self.next_label += 1
        self.labs[lab] = dim
        return lab

    def scalar(self, s: Scalar) -> str:
        return self.sym.render(s.to_sympy(), "ℂ", "ℂ")

    # networks

    def node_dims(self, d: Diagram, nid: NodeId) -> tuple[int, int]:
        node = d.nodes[nid]
        name = node.generator_type.name
        if name == "B":
            return self.dim(node.inputs[0].dim), self.dim(node.inputs[1].dim)
        if name == "S":
            return self.dim(node.outputs[0].dim), self.dim(node.outputs[1].dim)
        return self.dim(resolve_dim(node)), 0

    def labelled(self, d: Diagram) -> Labelled:
        if d.bang_boxes or d.parameters:
            raise NetUnsupported("bang boxes and pending parameters have no network")
        key = self.structure_key(d)
        if key in self.nets:
            return self.nets[key]
        outs, ins = d.boundary_outputs, d.boundary_inputs
        nb = len(outs) + len(ins)
        if self.nb is None:
            self.nb = nb
            self.next_label = nb
        elif self.nb != nb:
            raise NetUnsupported("the diagrams have different boundaries")
        port: dict[PortRef, int] = {}
        for slot, ref in enumerate((*outs, *ins)):
            dim = self.dim(_port_dim(d, ref))
            if self.labs.setdefault(slot, dim) != dim:
                raise NetUnsupported("a boundary slot changes dimension")
            port[ref] = slot
        internal = []
        for wire in sorted(d.wires, key=lambda w: w.sort_key()):
            da, db = _port_dim(d, wire.a), _port_dim(d, wire.b)
            if da != db:
                raise NetUnsupported(f"a wire joins dimensions {da} and {db}")
            lab = self.label(self.dim(da))
            port[wire.a] = port[wire.b] = lab
            internal.append(lab)
        order = tuple(sorted(d.nodes))
        nodes = []
        for nid in order:
            node = d.nodes[nid]
            kind = _KINDS.get(node.generator_type.name)
            if kind is None:
                raise NetUnsupported(f"generator {node.generator_type.name} has no Lean kind")
            legs = []
            for direction in (Direction.OUTPUT, Direction.INPUT):
                row = []
                for i in range(len(node.legs(direction))):
                    ref = PortRef(nid, direction, i)
                    if ref not in port:
                        raise NetUnsupported(f"port {ref} is neither wired nor on the boundary")
                    row.append(port[ref])
                legs.append(tuple(row))
            dim, aux = self.node_dims(d, nid)
            ph = self.phase(node.phase) if kind in ("Z", "X") else aux
            nodes.append(LNode(kind, legs[0], legs[1], dim, ph))
        net = LNet(tuple(nodes), tuple(internal), tuple(range(nb)))
        name = f"N{len(self.nets)}"
        labelled = Labelled(net, order, port, name, str(d.scalar.to_sympy()))
        self.nets[key] = labelled
        self.decls.append(f"/-- {_comment(self.describe(d))} -/\ndef {name} : Net := {net.lean()}")
        return labelled

    @staticmethod
    def structure_key(d: Diagram) -> object:
        nodes = tuple(
            (
                int(nid),
                n.generator_type.name,
                str(n.phase),
                tuple(str(p.dim) for p in n.inputs),
                tuple(str(p.dim) for p in n.outputs),
            )
            for nid, n in sorted(d.nodes.items())
        )
        wires = tuple(sorted(w.sort_key() for w in d.wires))
        return (nodes, wires, d.boundary_inputs, d.boundary_outputs, str(d.scalar.to_sympy()))

    # proofs

    def fresh(self, stem: str) -> str:
        self.count += 1
        return f"{stem}{self.count}"

    def statement(self, sl: str, a: Labelled, sr: str, b: Labelled) -> str:
        return f"Equal {self.env} {_paren(sl)} {a.name} {_paren(sr)} {b.name}"

    def axiom(self, stem: str, note: str, a: Labelled, b: Labelled, k: str) -> Proof:
        name = self.fresh(f"unproved_{re.sub(r'[^A-Za-z0-9_]', '_', stem)}_")
        self.unproved.append(name)
        self.decls.append(
            f"/-- {_comment(note)} -/\naxiom {_join(name, self.binders)} : "
            f"{self.statement('1', a, k, b)}"
        )
        return Proof(f"({_join(name, self.args)})", "1", k)

    def theorem(self, a: Labelled, b: Labelled, k: str, tactic: list[str]) -> Proof:
        name = self.fresh("step")
        body = "\n".join("  " + line for line in tactic)
        self.decls.append(
            f"theorem {_join(name, self.binders, self.hyps)} :\n    "
            f"{self.statement('1', a, k, b)} := by\n{body}"
        )
        return Proof(f"({_join(name, self.args, self.hyp_args)})", "1", k)

    def iso_term(self, s: str, a: LNet, b: LNet, pairs: Sequence[tuple[int, int]]) -> str:
        f = _ren(pairs)
        renamed = [n.rename(f) for n in a.nodes]
        used: set[int] = set()
        p = []
        for target in b.nodes:
            for j, n in enumerate(renamed):
                if j not in used and _equiv(n, target):
                    used.add(j)
                    p.append(j)
                    break
            else:
                raise NetUnsupported("the networks do not match node for node")
        return f"iso {self.env} {_paren(s)} _ {b.lean()} {_pairs(pairs)} {_list(p)} (by chk)"

    def term(self, t: object) -> Proof:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(t, Theorem)
        inf = t.inference
        if inf.kind == "refl":
            lab = self.labelled(t.lhs)
            return Proof(f"(Equal.refl {self.env} 1 {lab.name})", "1", "1")
        if inf.kind == "symm":
            p = self.term(inf.premises[0])
            return Proof(f"({p.term}).symm", p.right, p.left)
        if inf.kind == "trans":
            a, b = (self.term(p) for p in inf.premises)
            return Proof(
                f"(Equal.mul_trans {a.term} {b.term})",
                ("*", a.left, b.left),
                ("*", a.right, b.right),
            )
        if inf.kind == "iso":
            return self.iso(t.lhs, t.rhs)
        if inf.kind == "rule":
            return self.rule(t)
        raise NetUnsupported(f"a {inf.kind} inference has no network proof")

    def iso(self, x: Diagram, y: Diagram) -> Proof:
        a, b = self.labelled(x), self.labelled(y)
        if a.name == b.name:
            return Proof(f"(Equal.refl {self.env} 1 {a.name})", "1", "1")
        try:
            pairs = _iso_pairs(x, y, a, b)
            tactic = self.iso_term("1", a.net, b.net, pairs)
        except NetUnsupported as exc:
            return self.axiom("iso", f"isomorphic comparison views ({exc})", a, b, "1")
        return self.theorem(a, b, "1", [f"exact {tactic}"])

    def rule(self, t: object) -> Proof:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(t, Theorem)
        step = cast(RewriteStep | None, t.inference.data.get("step"))
        a, b = self.labelled(t.lhs), self.labelled(t.rhs)
        builder = _RULES.get(t.inference.rule)
        engine_k = self.scalar(step.scalar_introduced) if step is not None else "1"
        if step is None or builder is None:
            return self.axiom(t.inference.rule, f"rule {t.inference.rule}", a, b, engine_k)
        try:
            k, tactic = self.rule_tactic(t.lhs, t.rhs, step, a, b, builder)
        except NetUnsupported as exc:
            note = f"rule {t.inference.rule} ({exc})"
            return self.axiom(t.inference.rule, note, a, b, engine_k)
        return self.theorem(a, b, k, tactic)

    def rule_tactic(
        self,
        before: Diagram,
        after: Diagram,
        step: RewriteStep,
        a: Labelled,
        b: Labelled,
        builder: _RuleBuilder,
    ) -> tuple[str, list[str]]:
        new_ids = tuple(step.new_node_ids)
        mapping: dict[PortRef, PortRef] = dict(step.port_mapping)
        consumed_wires: tuple[Wire, ...] = tuple(step.consumed_wires)
        net = a.net
        index = {nid: i for i, nid in enumerate(a.order)}
        inverse = {new: old for old, new in mapping.items()}
        fresh: dict[Wire, int] = {}
        new_nodes: dict[NodeId, LNode] = {}
        for nid in new_ids:
            node = after.nodes[nid]
            legs = []
            for direction in (Direction.OUTPUT, Direction.INPUT):
                row = []
                for i in range(len(node.legs(direction))):
                    ref = PortRef(nid, direction, i)
                    if ref in inverse:
                        row.append(a.port[inverse[ref]])
                        continue
                    wire = _wire_at(after, ref)
                    if wire is None:
                        raise NetUnsupported(f"new port {ref} has no wire")
                    if wire not in fresh:
                        fresh[wire] = self.label(self.dim(_port_dim(after, ref)))
                    row.append(fresh[wire])
                legs.append(tuple(row))
            dim, aux = self.node_dims(after, nid)
            kind = _KINDS[node.generator_type.name]
            ph = self.phase(node.phase) if kind in ("Z", "X") else aux
            new_nodes[nid] = LNode(kind, legs[0], legs[1], dim, ph)
        deltas = [
            LNode("I", (a.port[old],), (a.port[new],), self.labs[a.port[new]], 0)
            for old, new in mapping.items()
            if new.node_id not in new_nodes
        ]
        spec = builder(_Ctx(self, before, after, step, a, new_nodes, deltas))
        inner = set(spec.l_ids)
        if spec.i_l is not None:
            i_l = tuple(spec.i_l)
        else:
            i_l = tuple(
                a.port[w.a] for w in consumed_wires if w.a.node_id in inner and w.b.node_id in inner
            )
        i_r = tuple(spec.i_r) if spec.i_r is not None else tuple(sorted(set(fresh.values())))
        c_ids = [nid for nid in a.order if nid not in inner]
        c_nodes = tuple(net.nodes[index[nid]] for nid in c_ids)
        l_nodes = tuple(net.nodes[index[nid]] for nid in spec.l_ids)
        r_nodes = tuple(n for _, n in spec.r)
        i_c = tuple(x for x in net.summed if x not in i_l)
        if len(i_c) + len(i_l) != len(net.summed) or len(set(i_l)) != len(i_l):
            raise NetUnsupported("a consumed wire label is not summed")
        bnd = net.bnd
        iface: list[int] = []
        for n in l_nodes:
            for x in n.labels():
                if x not in i_l and x not in iface:
                    iface.append(x)
        lemma = spec.lemma.replace("IFACE", _list(iface))
        k = spec.k
        a1 = LNet(c_nodes + l_nodes, i_c + i_l, bnd)
        lines = [f"refine Equal.trans ({self.iso_term('1', net, a1, [])}) ?_"]
        lines.append(
            f"refine Equal.trans (rewrite {self.env} {_nodes(c_nodes)} {_nodes(l_nodes)} "
            f"{_nodes(r_nodes)} {_list(i_c)} {_list(i_l)} {_list(i_r)} {_list(bnd)} "
            f"{_list(iface)} 1 {_paren(k)} {_paren(k)} ({lemma}) (by chk) (by chk) (by chk) "
            f"(one_mul _).symm) ?_"
        )
        current = LNet(c_nodes + r_nodes, i_c + i_r, bnd)
        ids: list[NodeId | None] = [*c_ids, *(nid for nid, _ in spec.r)]
        while any(n.kind == "I" for n in current.nodes):
            pos = next(i for i, n in enumerate(current.nodes) if n.kind == "I")
            bare = current.nodes[pos]
            x, w = bare.outs[0], bare.ins[0]
            if w not in current.summed or x == w:
                raise NetUnsupported("a bare wire joins two boundary slots")
            pre, post = current.nodes[:pos], current.nodes[pos + 1 :]
            target = LNet(pre + (bare,) + post, current.summed, bnd)
            lines.append(f"refine Equal.trans ({self.iso_term(k, current, target, [])}) ?_")
            lines.append(
                f"refine Equal.trans (wire_elim {self.env} {_paren(k)} {_nodes(pre)} "
                f"{_nodes(post)} "
                f"{_list(current.summed)} {_list(bnd)} {x} {w} {bare.dim} (by chk) (by chk) "
                f"(by chk) (by chk) (by chk)) ?_"
            )
            f = _ren([(w, x)])
            current = LNet(tuple(n.rename(f) for n in pre + post), _erase(current.summed, w), bnd)
            del ids[pos]
        pairs = _match_labels(current, ids, b)
        lines.append(f"exact {self.iso_term(k, current, b.net, pairs)}")
        return k, lines

    # whole export

    def finish(self, theorem: object, name: str) -> str:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(theorem, Theorem)
        self.env = "E"
        self.binders = self.args = self.hyps = self.hyp_args = ""
        start, goal = self.labelled(theorem.lhs), self.labelled(theorem.rhs)
        self.collect(theorem)
        self.set_params()
        proof = self.term(theorem)
        left, right = _drop_ones(proof.left), _drop_ones(proof.right)
        self.decls.append(
            f"/-- {_comment(f'start = {start.scalar_note}, goal = {goal.scalar_note}')} -/\n"
            f"theorem {_join(name, self.binders, self.hyps)} :\n    "
            f"{self.statement(left, start, right, goal)} := by\n"
            f"  simpa only [one_mul, mul_one] using {proof.term}"
        )
        return self.render(theorem)

    def collect(self, t: object) -> None:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(t, Theorem)
        self.labelled(t.lhs)
        self.labelled(t.rhs)
        for p in t.inference.premises:
            self.collect(p)
        step = cast(RewriteStep | None, t.inference.data.get("step"))
        if step is not None:
            self.scalar(step.scalar_introduced)

    def set_params(self) -> None:
        params = [(n, ty) for raw, (n, ty) in self.sym.params.items() if raw != "k_lean_index"]
        nats = [n for n, ty in params if ty == "ℕ"]
        self.binders = " ".join(f"({n} : {ty})" for n, ty in params)
        self.args = " ".join(n for n, _ in params)
        self.hyps = " ".join(f"(h_{n} : 0 < {n})" for n in nats)
        self.hyp_args = " ".join(f"h_{n}" for n in nats)
        self.env = f"(E {self.args})" if params else "E"

    def render(self, theorem: object) -> str:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(theorem, Theorem)
        params = [(n, ty) for raw, (n, ty) in self.sym.params.items() if raw != "k_lean_index"]
        binders = " ".join(f"({n} : {ty})" for n, ty in params)
        dims = "\n".join(f"  | {i} => {text}" for i, text in enumerate(self.dims))
        phases = "\n".join(f"  | {i} => {text}" for i, text in enumerate(self.phases))
        labs = _pairs(sorted(self.labs.items()))
        tables = [
            f"/-- Dimension values. -/\ndef Δ {binders} : ℕ → ℕ\n{dims}\n  | _ => 0",
            (
                f"/-- Phase vectors in turns. -/\nnoncomputable def Φ {binders} : ℕ → ℕ → ℝ\n"
                f"{phases}\n  | _ => fun _ => 0"
            ),
            f"/-- Each label's dimension index. -/\ndef labs : List (ℕ × ℕ) := {labs}",
            f"/-- The environment. -/\nnoncomputable def E {binders} : Env := "
            f"⟨Δ {self.args}, Φ {self.args}, labs⟩".replace("  ", " "),
        ]
        notes = [f"  assumes {a}" for a in theorem.assumptions]
        if self.unproved:
            notes.append("  unproved steps: " + ", ".join(self.unproved))
        header = [
            "/-",
            "  ArchytasZX proof export over the tensor-network semantics of the ArchytasZX Lean",
            "  library: every diagram is a Net, every rule step an instance of a lemma proved",
            "  there. `#print axioms` lists any step stated without proof.",
            *notes,
            "-/",
            "",
            "import ArchytasZX",
            "",
            "namespace ArchytasZX.Export",
            "",
            "open ArchytasZX",
            "",
            "set_option linter.unusedVariables false",
            "set_option linter.unusedTactic false",
            "set_option linter.unreachableTactic false",
            "",
            "set_option hygiene false in",
            "/-- Decides a structural side condition, unfolding the environment if needed. -/",
            'macro "chk" : tactic => `(tactic| first | decide | (simp only [E]; decide))',
        ]
        return "\n".join(
            [*header, "", "\n\n".join([*tables, *self.decls]), "", "end ArchytasZX.Export", ""]
        )


def _port_dim(d: Diagram, ref: PortRef) -> Dim:
    node = d.nodes[ref.node_id]
    return node.legs(ref.direction)[ref.index].dim


def _wire_at(d: Diagram, ref: PortRef) -> Wire | None:
    for wire in d.wires:
        if ref in (wire.a, wire.b):
            return wire
    return None


def _equiv(n: LNode, m: LNode) -> bool:
    """Python mirror of the Lean ``nodeEquiv``."""
    if (n.kind, n.dim, n.ph) != (m.kind, m.dim, m.ph):
        return False
    if n.kind == "Z":
        return sorted(n.labels()) == sorted(m.labels())
    if n.kind == "X":
        return sorted(n.outs) == sorted(m.outs) and sorted(n.ins) == sorted(m.ins)
    return n.outs == m.outs and n.ins == m.ins


def _match_labels(
    current: LNet, ids: Sequence[NodeId | None], b: Labelled
) -> list[tuple[int, int]]:
    """The renaming carrying ``current``'s summed labels onto ``b``'s, node by node."""
    mapping: dict[int, int] = {}
    target_index = {nid: i for i, nid in enumerate(b.order)}
    if len(ids) != len(b.order) or None in ids:
        raise NetUnsupported("the rewritten network has the wrong nodes")
    for nid, node in zip(ids, current.nodes, strict=True):
        if nid not in target_index:
            raise NetUnsupported(f"node {nid} is missing after the rewrite")
        target = b.net.nodes[target_index[nid]]
        if len(node.labels()) != len(target.labels()):
            raise NetUnsupported(f"node {nid} changed arity")
        for x, y in zip(node.labels(), target.labels(), strict=True):
            if mapping.setdefault(x, y) != y:
                raise NetUnsupported("inconsistent labels after the rewrite")
    bnd = set(current.bnd)
    for x, y in mapping.items():
        if (x in bnd or y in bnd) and x != y:
            raise NetUnsupported("a boundary label moves")
    inner_map = {x: y for x, y in mapping.items() if x not in bnd}
    if sorted(inner_map) != sorted(current.summed) or sorted(inner_map.values()) != sorted(
        b.net.summed
    ):
        raise NetUnsupported("the summed labels do not correspond")
    return _perm_to_bijection(inner_map)


def _iso_pairs(x: Diagram, y: Diagram, a: Labelled, b: Labelled) -> list[tuple[int, int]]:
    """The summed-label renaming from ``x``'s network to ``y``'s, through a graph isomorphism."""
    gx, gy = x.copy(), y.copy()
    gx.set_scalar(Scalar.one())
    gy.set_scalar(Scalar.one())
    gx.set_parameters({})
    gy.set_parameters({})
    nodes = isomorphism(gx, gy, symmetric_legs=True)
    if nodes is None:
        raise NetUnsupported("no node isomorphism")
    used: set[Wire] = set()
    mapping: dict[int, int] = {}
    symmetric = {"Z", "X"}

    def fits(ref: PortRef, target: PortRef) -> bool:
        if nodes[ref.node_id] != target.node_id or ref.direction != target.direction:
            return False
        kind = x.nodes[ref.node_id].generator_type.name
        return kind in symmetric or ref.index == target.index

    for wire in sorted(x.wires, key=lambda w: w.sort_key()):
        for other in sorted(y.wires, key=lambda w: w.sort_key()):
            if other in used:
                continue
            if (fits(wire.a, other.a) and fits(wire.b, other.b)) or (
                fits(wire.a, other.b) and fits(wire.b, other.a)
            ):
                used.add(other)
                mapping[a.port[wire.a]] = b.port[other.a]
                break
        else:
            raise NetUnsupported("no wire correspondence")
    if sorted(mapping) != sorted(a.net.summed):
        raise NetUnsupported("the summed labels do not correspond")
    return _perm_to_bijection(mapping)


@dataclass
class _Ctx:
    """What a rule hook sees of one step."""

    ex: _Export
    before: Diagram
    after: Diagram
    step: RewriteStep
    a: Labelled
    new_nodes: dict[NodeId, LNode]
    deltas: list[LNode]

    @property
    def match(self) -> Any:
        return self.step.match

    def node(self, nid: NodeId) -> LNode:
        return self.a.net.nodes[self.a.order.index(nid)]

    def wire(self, w: Wire) -> int:
        return self.a.port[w.a]

    @property
    def env(self) -> str:
        return self.ex.env


@dataclass
class _Spec:
    """A rule step's local equation: L's node ids, R's nodes, the lemma and its scalar."""

    l_ids: tuple[NodeId, ...]
    r: Sequence[tuple[NodeId | None, LNode]]
    lemma: str
    k: str = "1"
    i_l: tuple[int, ...] | None = None
    i_r: tuple[int, ...] | None = None


_RuleBuilder = Callable[[_Ctx], _Spec]

_CHK = "(by chk)"


def _phaseless(c: _Ctx, n: LNode) -> str:
    if n.ph != 0:
        raise NetUnsupported("a phase where the rule needs none")
    return "(by intro k _; simp [E, Φ, ex_zero])"


def _phase_sum() -> str:
    return (
        "(by intro k _; simp only [E, Φ]; rw [← ex_add]; exact ex_congr_int (by "
        "(try split_ifs) <;> first | exact ⟨0, by push_cast; ring⟩ "
        "| exact ⟨-1, by push_cast; ring⟩ "
        "| exact ⟨1, by push_cast; ring⟩))"
    )


def _positive(c: _Ctx, dz: int) -> str:
    return (
        "(by simp only [E, Δ] <;> first | decide | omega | positivity | "
        "exact Nat.mul_pos (by omega) (by omega) | exact Nat.pos_pow_of_pos _ (by omega))"
    )


def _chks(n: int) -> str:
    return " ".join([_CHK] * n)


def _need(cond: bool, what: str) -> None:
    if not cond:
        raise NetUnsupported(what)


def _fusion(c: _Ctx) -> _Spec:
    m = c.match
    _need(not c.deltas and len(c.new_nodes) == 1, "fusion leaves one merged node")
    na, nb = c.node(m.a_id), c.node(m.b_id)
    ((rid, r),) = c.new_nodes.items()
    w = c.wire(m.wire)
    _need(na.dim == nb.dim == r.dim and na.kind == nb.kind == r.kind, "fusion across dims")
    order = (m.a_id, m.b_id)
    if na.kind == "X":
        first, second = na, nb
        if w not in na.outs:
            order = (m.b_id, m.a_id)
            first, second = nb, na
        lemma = (
            f"fusionX {c.env} {first.ph} {second.ph} {r.ph} {r.dim} {w} {_list(first.outs)} "
            f"{_list(first.ins)} {_list(second.outs)} {_list(second.ins)} {_list(r.outs)} "
            f"{_list(r.ins)} IFACE {_positive(c, r.dim)} {_chks(9)} {_phase_sum()}"
        )
    elif na.kind == "Z":
        lemma = (
            f"fusionZ {c.env} {na.ph} {nb.ph} {r.ph} {r.dim} {w} {_list(na.outs)} "
            f"{_list(na.ins)} {_list(nb.outs)} {_list(nb.ins)} {_list(r.outs)} {_list(r.ins)} "
            f"IFACE {_chks(4)} {_phase_sum()}"
        )
    else:
        raise NetUnsupported(f"fusion of {na.kind}")
    return _Spec(order, [(rid, r)], lemma)


def _identity(c: _Ctx) -> _Spec:
    m = c.match
    _need(not c.new_nodes and len(c.deltas) == 1, "identity removal leaves one bare wire")
    n = c.node(m.node_id)
    delta = c.deltas[0]
    x, w = delta.outs[0], delta.ins[0]
    p, q = n.outs[0], n.ins[0]
    wire = replace(delta, dim=n.dim)
    if n.kind == "Z":
        lemma = f"identityZ {c.env} {n.ph} {n.dim} {p} {q} {w} {x} IFACE {_CHK} {_phaseless(c, n)}"
    elif n.kind == "X":
        lemma = (
            f"identityX {c.env} {n.ph} {n.dim} {p} {q} {w} {x} IFACE {_positive(c, n.dim)} "
            f"{_chks(4)} {_phaseless(c, n)}"
        )
    else:
        raise NetUnsupported(f"identity removal of {n.kind}")
    return _Spec((m.node_id,), [(None, wire)], lemma)


def _cap(c: _Ctx) -> _Spec:
    m = c.match
    st, ef = c.node(m.state_id), c.node(m.effect_id)
    w = c.wire(m.wire)
    _need(st.outs == (w,) and not st.ins and ef.ins == (w,) and not ef.outs, "cap shape")
    name = "capSwapped" if m.swapped else "cap"
    _need((st.kind, ef.kind) == (("X", "Z") if m.swapped else ("Z", "X")), "cap colours")
    dz = st.dim
    lemma = (
        f"{name} {c.env} {st.ph} {ef.ph} {dz} {w} IFACE {_positive(c, dz)} {_CHK} "
        f"{_phaseless(c, st)} {_phaseless(c, ef)}"
    )
    return _Spec((m.state_id, m.effect_id), [], lemma, f"rt ({c.env}.dimv {dz})")


def _color(c: _Ctx) -> _Spec:
    m = c.match
    sp_, fo = c.node(m.spider_id), c.node(m.fourier_id)
    ((rid, r),) = c.new_nodes.items()
    w = c.wire(m.wire)
    dz = sp_.dim
    if m.is_state:
        _need(sp_.outs == (w,) and not sp_.ins and fo.ins == (w,), "colour change shape")
        x = fo.outs[0]
        name = "colorStateX" if m.swapped else "colorState"
    else:
        _need(sp_.ins == (w,) and not sp_.outs and fo.outs == (w,), "colour change shape")
        x = fo.ins[0]
        name = "colorEffectX" if m.swapped else "colorEffect"
    extra = f"{_chks(2)} " if m.swapped else ""
    lemma = (
        f"{name} {c.env} {sp_.ph} {r.ph} {dz} {w} {x} IFACE {_positive(c, dz)} {_chks(2)} "
        f"{extra}{_phaseless(c, sp_)} {_phaseless(c, r)}"
    )
    return _Spec((m.spider_id, m.fourier_id), [(rid, r)], lemma)


def _state_copy(c: _Ctx) -> _Spec:
    m = c.match
    st, sp_ = c.node(m.state_id), c.node(m.spider_id)
    w = c.wire(m.wire)
    _need(st.outs == (w,) and not st.ins and sp_.ins == (w,), "state copy shape")
    rs = [(nid, c.new_nodes[nid]) for nid in c.step.new_node_ids]
    phs = {n.ph for _, n in rs}
    _need(len(phs) <= 1, "new states differ")
    cph = phs.pop() if phs else 0
    dz, n = st.dim, len(sp_.outs)
    rt = f"rt ({c.env}.dimv {dz})"
    k = rt if n == 0 else "1" if n == 1 else f"({rt} * ({rt})⁻¹ ^ {n})"
    name = "stateCopySwapped" if m.swapped else "stateCopy"
    lemma = (
        f"{name} {c.env} {st.ph} {sp_.ph} {cph} {dz} {w} {_list(sp_.outs)} IFACE {_paren(k)} "
        f"{_positive(c, dz)} {_chks(3)} {_phaseless(c, st)} {_phaseless(c, sp_)} "
        f"(by intro k _; simp [E, Φ, ex_zero]) "
        f"(by first | (simp; done) | exact rt_cancel _ {n} {_positive(c, dz)})"
    )
    return _Spec((m.state_id, m.spider_id), rs, lemma, k)


def _triangle(c: _Ctx) -> _Spec:
    m = c.match
    first, second = c.node(m.first_id), c.node(m.second_id)
    _need(len(c.deltas) == 1 and not c.new_nodes, "triangle cancellation leaves one wire")
    w = c.wire(m.wire)
    _need(first.outs == (w,) and second.ins == (w,), "triangle shape")
    o, i = second.outs[0], first.ins[0]
    delta = replace(c.deltas[0], dim=first.dim)
    x, y = delta.outs[0], delta.ins[0]
    if (first.kind, second.kind) == ("T", "Ti"):
        name = "triangle"
    elif (first.kind, second.kind) == ("Ti", "T"):
        name = "triangle'"
    else:
        raise NetUnsupported("triangle pair kinds")
    lemma = f"{name} {c.env} {first.dim} {w} {o} {i} {x} {y} IFACE {_chks(6)}"
    return _Spec((m.first_id, m.second_id), [(None, delta)], lemma)


def _fourier4(c: _Ctx) -> _Spec:
    m = c.match
    ids = tuple(m.node_ids)
    fs = [c.node(nid) for nid in ids]
    ((rid, r),) = c.new_nodes.items()
    w1, w2, w3 = fs[0].outs[0], fs[1].outs[0], fs[2].outs[0]
    a_, b_ = fs[0].ins[0], fs[3].outs[0]
    _need(fs[1].ins == (w1,) and fs[2].ins == (w2,) and fs[3].ins == (w3,), "F chain shape")
    _need(r.outs == (b_,) and r.ins == (a_,), "F chain replacement")
    dz = fs[0].dim
    lemma = (
        f"fourier4 {c.env} {a_} {b_} {r.ph} {dz} {w1} {w2} {w3} IFACE {_positive(c, dz)} "
        f"{_chks(14)} {_phaseless(c, r)}"
    )
    return _Spec(ids, [(rid, r)], lemma, "1", (w1, w2, w3))


def _bialgebra(c: _Ctx) -> _Spec:
    m = c.match
    x, z = c.node(m.x_id), c.node(m.z_id)
    w = c.wire(m.wire)
    _need(x.outs == (w,) and len(x.ins) == 2 and z.ins == (w,) and len(z.outs) == 2, "shape")
    a_, b_ = x.ins
    cc, e = z.outs
    rs = [(nid, c.new_nodes[nid]) for nid in c.step.new_node_ids]
    z1, z2, x1, x2 = (n for _, n in rs)
    n00, n01 = z1.outs
    n10, n11 = z2.outs
    _need(z1.ins == (a_,) and z2.ins == (b_,), "bialgebra fan-out")
    _need(x1.outs == (cc,) and x1.ins == (n00, n10), "bialgebra fan-in")
    _need(x2.outs == (e,) and x2.ins == (n01, n11), "bialgebra fan-in")
    name = "bialgebraSwapped" if m.swapped else "bialgebra"
    dz = x.dim
    lemma = (
        f"{name} {c.env} {x.ph} {z.ph} {z1.ph} {z2.ph} {x1.ph} {x2.ph} {dz} {w} {a_} {b_} {cc} "
        f"{e} {n00} {n01} {n10} {n11} IFACE {_positive(c, dz)} {_chks(5)} "
        + " ".join(_phaseless(c, n) for n in (x, z, z1, z2, x1, x2))
    )
    return _Spec(
        (m.x_id, m.z_id),
        rs,
        lemma,
        f"rt ({c.env}.dimv {dz})",
        None,
        (n00, n01, n10, n11),
    )


def _hopf(c: _Ctx) -> _Spec:
    m = c.match
    up, lo = c.node(m.z_id), c.node(m.x_id)
    f1, f2 = (c.node(nid) for nid in m.fourier_ids)
    u, v1, v2, v3 = (c.wire(w) for w in m.wires)
    _need(f1.outs == (v2,) and f1.ins == (v1,) and f2.outs == (v3,) and f2.ins == (v2,), "F path")
    rs = [(nid, c.new_nodes[nid]) for nid in c.step.new_node_ids]
    nu, nl = (n for _, n in rs)
    dz = up.dim
    k = f"((({c.env}.dimv {dz} : ℕ) : ℂ))⁻¹"
    tail = f"{_positive(c, dz)} {_chks(6)} (nat_inv_mul _ {_positive(c, dz)})"
    if not m.swapped:
        lemma = (
            f"hopf {c.env} {up.ph} {lo.ph} {dz} {u} {v1} {v2} {v3} {_list(up.outs)} "
            f"{_list(up.ins)} {_list(lo.outs)} {_list(lo.ins)} {_list(nu.outs)} {_list(nu.ins)} "
            f"{_list(nl.outs)} {_list(nl.ins)} IFACE {k} {tail}"
        )
    else:
        lemma = (
            f"hopfSwapped {c.env} {up.ph} {lo.ph} {dz} {u} {v1} {v2} {v3} {_list(up.outs)} "
            f"{_list(up.ins)} {_list(lo.outs)} {_list(lo.ins)} {_list(nu.outs)} "
            f"{_list(nu.ins)} {_list(nl.outs)} {_list(nl.ins)} IFACE {k} {tail}"
        )
    ids = (m.z_id, m.x_id, *m.fourier_ids)
    return _Spec(ids, rs, lemma, k, (u, v1, v2, v3))


def _dims_mul() -> str:
    return "(by simp only [E, Δ] <;> first | rfl | norm_num | ring)"


def _w_fusion(c: _Ctx) -> _Spec:
    m = c.match
    first, second = c.node(m.first_id), c.node(m.second_id)
    ((rid, r),) = c.new_nodes.items()
    w = c.wire(m.wire)
    _need(len(first.ins) == 1 and second.ins == (w,) and r.ins == first.ins, "W fusion shape")
    dz = first.dim
    lemma = (
        f"wFusion {c.env} {dz} {w} {first.ins[0]} {_list(first.outs)} {_list(second.outs)} "
        f"{_list(r.outs)} IFACE {_chks(6)} {_positive(c, dz)} {_CHK}"
    )
    return _Spec((m.first_id, m.second_id), [(rid, r)], lemma)


def _w_identity(c: _Ctx) -> _Spec:
    m = c.match
    n = c.node(m.node_id)
    _need(n.kind == "W" and len(c.deltas) == 1 and not c.new_nodes, "W identity shape")
    delta = replace(c.deltas[0], dim=n.dim)
    x, w = delta.outs[0], delta.ins[0]
    lemma = f"wIdentity {c.env} {n.dim} {n.outs[0]} {n.ins[0]} {w} {x} IFACE {_chks(4)}"
    return _Spec((m.node_id,), [(None, delta)], lemma)


def _fed_state(c: _Ctx) -> _Spec:
    m = c.match
    st, tg = c.node(m.state_id), c.node(m.target_id)
    w = c.wire(m.wire)
    _need(st.kind == "X" and st.outs == (w,) and not st.ins and tg.ins == (w,), "fed state shape")
    dz = st.dim
    if tg.kind == "W":
        rs = [(nid, c.new_nodes[nid]) for nid in c.step.new_node_ids]
        phs = {n.ph for _, n in rs}
        _need(len(phs) <= 1, "new states differ")
        cph = phs.pop() if phs else 0
        n = len(tg.outs)
        rt = f"rt ({c.env}.dimv {dz})"
        k = rt if n == 0 else "1" if n == 1 else f"({rt} * ({rt})⁻¹ ^ {n})"
        lemma = (
            f"wZeroCopy {c.env} {st.ph} {cph} {dz} {w} {_list(tg.outs)} IFACE {_paren(k)} "
            f"{_positive(c, dz)} {_chks(3)} {_phaseless(c, st)} "
            f"(by intro k _; simp [E, Φ, ex_zero]) "
            f"(by first | (simp; done) | exact rt_cancel _ {n} {_positive(c, dz)})"
        )
        return _Spec((m.state_id, m.target_id), rs, lemma, k)
    _need(tg.kind in ("T", "Ti") and len(tg.outs) == 1, "fed triangle shape")
    o = tg.outs[0]
    name = "triZeroState" if tg.kind == "T" else "triInvZeroState"
    r = LNode("X", (o,), (), dz, st.ph)
    lemma = (
        f"{name} {c.env} {st.ph} {dz} {w} {o} IFACE {_positive(c, dz)} {_chks(3)} "
        f"{_phaseless(c, st)}"
    )
    return _Spec((m.state_id, m.target_id), [(m.state_id, r)], lemma)


def _tri_effect(c: _Ctx) -> _Spec:
    m = c.match
    tr, ef = c.node(m.triangle_id), c.node(m.effect_id)
    ((rid, r),) = c.new_nodes.items()
    w = c.wire(m.wire)
    _need(tr.kind == "T" and tr.outs == (w,) and ef.ins == (w,) and not ef.outs, "shape")
    dz = tr.dim
    lemma = (
        f"triZeroEffect {c.env} {ef.ph} {r.ph} {dz} {w} {tr.ins[0]} IFACE {_positive(c, dz)} "
        f"{_chks(3)} {_phaseless(c, ef)} {_phaseless(c, r)}"
    )
    return _Spec((m.triangle_id, m.effect_id), [(rid, r)], lemma, f"rt ({c.env}.dimv {dz})")


def _w_effect(c: _Ctx) -> _Spec:
    m = c.match
    wn, ef = c.node(m.w_id), c.node(m.effect_id)
    w = c.wire(m.wire)
    _need(wn.kind == "W" and len(wn.ins) == 1 and ef.ins == (w,) and not ef.outs, "shape")
    rs = [(nid, c.new_nodes[nid]) for nid in c.step.new_node_ids]
    dz, i = wn.dim, wn.ins[0]
    if m.zero:
        ((_, r),) = rs
        lemma = (
            f"wZeroEffect {c.env} {ef.ph} {dz} {w} {i} {_list(wn.outs)} {_list(r.outs)} IFACE "
            f"{_positive(c, dz)} {_chks(6)} {_phaseless(c, ef)}"
        )
        return _Spec((m.w_id, m.effect_id), rs, lemma, f"rt ({c.env}.dimv {dz})")
    (_, tri), (_, rest) = rs
    v = tri.outs[0]
    _need(tri.ins == (i,) and rest.ins == (v,), "W into Z effect replacement")
    lemma = (
        f"wZEffect {c.env} {ef.ph} {dz} {w} {i} {v} {_list(wn.outs)} {_list(rest.outs)} IFACE "
        f"{_positive(c, dz)} {_chks(10)} {_phaseless(c, ef)}"
    )
    return _Spec((m.w_id, m.effect_id), rs, lemma, "1", None, (v,))


def _conn_states(c: _Ctx) -> _Spec:
    m = c.match
    cn = c.node(m.connective_id)
    ends = [c.node(nid) for nid in m.end_ids]
    ((rid, r),) = c.new_nodes.items()
    if m.effects:
        _need(cn.kind == "S" and len(cn.outs) == 2, "connective effects shape")
        a_, b_ = cn.outs
        k = cn.ins[0]
        side = "ins"
    else:
        _need(cn.kind == "B" and len(cn.ins) == 2, "connective states shape")
        a_, b_ = cn.ins
        k = cn.outs[0]
        side = "outs"
    pairs = {getattr(n, side)[0]: (nid, n) for nid, n in zip(m.end_ids, ends, strict=True)}
    _need(set(pairs) == {a_, b_}, "connective ends")
    (ia, na), (ib, nb) = pairs[a_], pairs[b_]
    name = {
        (False, False): "connStates",
        (True, False): "connEffects",
        (False, True): "connStatesX",
        (True, True): "connEffectsX",
    }[(m.effects, m.swapped)]
    ds, dt = cn.dim, cn.ph
    dk = r.dim
    lemma = (
        f"{name} {c.env} {ds} {dt} {dk} {na.ph} {nb.ph} {r.ph} {a_} {b_} {k} IFACE "
        f"{_positive(c, ds)} {_positive(c, dt)} {_dims_mul()} {_chks(6)} "
        f"{_phaseless(c, na)} {_phaseless(c, nb)} {_phaseless(c, r)}"
    )
    return _Spec((m.connective_id, ia, ib), [(rid, r)], lemma, "1", (a_, b_))


def _conn_inverse(c: _Ctx) -> _Spec:
    m = c.match
    first, second = c.node(m.first_id), c.node(m.second_id)
    ds, dt = first.dim, first.ph
    if not m.bind_first:
        _need(first.kind == "S" and second.kind == "B" and first.outs == second.ins, "shape")
        _need(len(c.deltas) == 1, "one identity wire")
        a_, b_ = first.outs
        k1, k2 = first.ins[0], second.outs[0]
        delta = c.deltas[0]
        dk = c.ex.labs[k1]
        delta = replace(delta, dim=dk)
        lemma = (
            f"connInverse {c.env} {ds} {dt} {dk} {a_} {b_} {k1} {k2} {delta.outs[0]} "
            f"{delta.ins[0]} IFACE {_positive(c, dt)} {_dims_mul()} {_chks(6)}"
        )
        return _Spec((m.first_id, m.second_id), [(None, delta)], lemma, "1", (a_, b_))
    _need(first.kind == "B" and second.kind == "S" and first.outs == second.ins, "shape")
    _need(len(c.deltas) == 2, "two identity wires")
    k = first.outs[0]
    a1, b1 = first.ins
    a2, b2 = second.outs
    da = [d for d in c.deltas if {d.outs[0], d.ins[0]} == {a1, a2}]
    db = [d for d in c.deltas if {d.outs[0], d.ins[0]} == {b1, b2}]
    _need(len(da) == 1 and len(db) == 1, "identity wires")
    d1, d2 = replace(da[0], dim=ds), replace(db[0], dim=dt)
    dk = c.ex.labs[k]
    lemma = (
        f"connInverseBind {c.env} {ds} {dt} {dk} {k} {a1} {b1} {a2} {b2} {d1.outs[0]} "
        f"{d1.ins[0]} {d2.outs[0]} {d2.ins[0]} IFACE {_dims_mul()} {_chks(8)}"
    )
    return _Spec((m.first_id, m.second_id), [(None, d1), (None, d2)], lemma, "1", (k,))


_RULES: dict[str, _RuleBuilder] = {
    "spider_fusion": _fusion,
    "z_fusion_prime_d": _fusion,
    "identity_removal": _identity,
    "zx_cap": _cap,
    "zx_cap_swapped": _cap,
    "fourier_state_color_change": _color,
    "fourier_state_color_change_swapped": _color,
    "state_copy": _state_copy,
    "state_copy_swapped": _state_copy,
    "triangle_inverse_cancellation": _triangle,
    "fourier_cancellation": _fourier4,
    "bialgebra": _bialgebra,
    "bialgebra_swapped": _bialgebra,
    "hopf": _hopf,
    "hopf_swapped": _hopf,
    "w_fusion": _w_fusion,
    "w_identity": _w_identity,
    "w_zero_copy": _fed_state,
    "triangle_zero_state": _fed_state,
    "triangle_zero_effect": _tri_effect,
    "w_z_effect": _w_effect,
    "w_zero_effect": _w_effect,
    "connective_states": _conn_states,
    "connective_states_swapped": _conn_states,
    "connective_inverse": _conn_inverse,
    "connective_inverse_bind_first": _conn_inverse,
}


def _join(*parts: str) -> str:
    """The non-empty parts, space-separated."""
    return " ".join(p for p in parts if p)


def _paren(text: str) -> str:
    """``text`` as one Lean argument."""
    if re.fullmatch(r"[A-Za-z0-9_.']+", text) or (text.startswith("(") and _closes_at_end(text)):
        return text
    return f"({text})"


def _closes_at_end(text: str) -> bool:
    depth = 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(text) - 1:
            return False
    return depth == 0


def _drop_ones(tree: object) -> str:
    """A product tree as Lean, without the factors ``simp only [one_mul, mul_one]`` removes."""
    if isinstance(tree, tuple):
        left, right = _drop_ones(tree[1]), _drop_ones(tree[2])
        if left == "1":
            return right
        if right == "1":
            return left
        return f"({left} * {right})"
    return str(tree)


def export(
    theorem: object, name: str = "main", describe: Callable[[Diagram], str] | None = None
) -> str:
    """Lean 4 source proving ``theorem`` over the tensor-network semantics."""
    from archytaszx.semantics.kernel import Theorem

    if not isinstance(theorem, Theorem):
        raise NetUnsupported(f"export takes a Theorem, got {type(theorem).__name__}")
    if theorem.variables:
        raise NetUnsupported("free multiplicities have no network")
    out = _Export(describe or (lambda d: f"a diagram of {len(d.nodes)} nodes"))
    return out.finish(theorem, name)


def lean_project() -> str:
    """The bundled Lean project's directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "lean")


def _lake() -> str | None:
    found = shutil.which("lake")
    home = os.path.expanduser("~/.elan/bin/lake")
    return found or (home if os.path.exists(home) else None)


def check(source: str, path: str, timeout: int = 1800) -> tuple[bool, str]:
    """Write ``source`` to ``path`` and check it with Lean in the bundled project, fetching
    Mathlib and building the library first; ``(accepted, output)``."""
    lake = _lake()
    if lake is None:
        raise NetUnsupported("Lean (lake) is not installed")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    project = lean_project()
    if not os.path.isdir(os.path.join(project, ".lake", "packages", "mathlib")):
        fetch = subprocess.run(
            [lake, "exe", "cache", "get"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if fetch.returncode != 0:
            return False, fetch.stdout + fetch.stderr
    build = subprocess.run(
        [lake, "build"], cwd=project, capture_output=True, text=True, timeout=timeout, check=False
    )
    if build.returncode != 0:
        return False, build.stdout + build.stderr
    done = subprocess.run(
        [lake, "env", "lean", os.path.abspath(path)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return done.returncode == 0, done.stdout + done.stderr
