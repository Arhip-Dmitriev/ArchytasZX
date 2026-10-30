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
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import INDEX, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import _mult_symbol
from archytaszx.diagram.compare import isomorphism
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef, Wire
from archytaszx.rewrite.engine import RewriteStep
from archytaszx.semantics.denote import DenoteDomainError, resolve_dim
from archytaszx.semantics.lean_scalar import NC, Inv, Om, Rt, Scalars, ScalarUnsupported

__all__ = ["NetUnsupported", "check", "export", "lean_project"]


class NetUnsupported(Exception):
    """A theorem or diagram outside the tensor-network export."""


ONE = sp.Integer(1)
H_OFFSET = 1 << 16
_POSITIVE = (
    "(by first | omega | norm_num | positivity | (apply Nat.mul_pos <;> omega) "
    "| exact Nat.pos_pow_of_pos _ (by omega))"
)


_KINDS = {"Z": "Z", "X": "X", "F": "F", "T": "T", "Ti": "Ti", "W": "W", "B": "B", "S": "S"}
_RESERVED = frozenset(
    {"E", "Φ", "Ψ", "Δ", "labs", "k", "at", "by", "do", "fun", "in", "let", "if", "then", "else"}
    | {"end", "from", "have", "show", "with", "match", "theorem", "def", "axiom", "open"}
    | {"ex", "om", "fou", "Net", "Node", "Env", "Equal", "I", "π", "pi", "exp", "sqrt"}
    | {"F", "FC", "SpC", "ksOf", "bundL", "sS", "sG", "Δpos", "j"}
)


@dataclass(frozen=True)
class LNode:
    """One network node: kind, output and input labels, dimension and phase index."""

    kind: Any
    outs: tuple[int, ...]
    ins: tuple[int, ...]
    dim: int
    ph: int

    def lean(self) -> str:
        """The Lean ``Node`` literal."""
        return (
            f"⟨{_klean(self.kind)}, {_list(self.outs)}, {_list(self.ins)}, {self.dim}, {self.ph}⟩"
        )

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
class _Named:
    """A scalar written as a Lean definition applied to arguments."""

    text: str
    model: sp.Expr


@dataclass(frozen=True)
class Proof:
    """A Lean term proving ``Equal E left lhs right rhs``."""

    term: str
    left: object
    right: object


def _list(xs: Sequence[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def _nodes(ns: Sequence[LNode]) -> str:
    return "([" + ", ".join(n.lean() for n in ns) + "] : List Node)"


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
            if (kind, ty) in {("ℕ", "ℝ"), ("ℕ", "ℂ"), ("ℝ", "ℂ"), ("ℤ", "ℝ"), ("ℤ", "ℂ")}:
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
            if not exp.is_number:
                try:
                    power = self.render(exp, "ℕ", sym_kind)
                except NetUnsupported:
                    power = None
                if power is not None:
                    return f"({self.render(base, ty, sym_kind)} ^ {power})"
            if ty == "ℕ":
                raise NetUnsupported(f"{expr} is not a natural number")
            if exp.is_Integer:
                return f"({self.render(base, ty, sym_kind)} ^ {-int(exp)})⁻¹"
            if exp.is_Rational and exp.q == 2:
                root = f"(Real.sqrt {self.render(base, 'ℝ', sym_kind)} : {ty})"
                if ty == "ℝ":
                    root = f"Real.sqrt {self.render(base, 'ℝ', sym_kind)}"
                pw = int(exp.p)
                if pw == 1:
                    return f"({root})"
                if pw > 0:
                    return f"(({root}) ^ {pw})"
                return f"(({root}) ^ {-pw})⁻¹"
            raise NetUnsupported(f"the power {expr}")
        if isinstance(expr, sp.exp):
            if ty != "ℂ":
                raise NetUnsupported("an exponential outside ℂ")
            return f"Complex.exp {self.render(expr.args[0], 'ℂ', sym_kind)}"
        raise NetUnsupported(f"cannot write {expr} ({type(expr).__name__}) in Lean")


class _Export:
    """Accumulates the tables, networks and proofs of one export."""

    mixed_ok: bool
    mixed: set[tuple[int, int]]
    env: str
    binders: str
    args: str
    hyps: str
    hyp_args: str

    def __init__(self, describe: Callable[[Diagram], str]) -> None:
        self.describe = describe
        self.sym = _Sym()
        self.dims: list[str] = []
        self.dim_key: dict[str, int] = {}
        self.phases: list[str] = ["fun _ => 0"]
        self.phase_specs: list[tuple[dict[int, sp.Expr], sp.Expr | None]] = [({}, None)]
        self.phase_key: dict[str, int] = {"": 0}
        self.weight_tab: dict[int, str] = {}
        self.coef_models: dict[str, dict[int, sp.Expr]] = {}
        self.labs: dict[int, int] = {}
        self.nb: int | None = None
        self.next_label = 0
        self.nets: dict[object, Labelled] = {}
        self.decls: list[str] = []
        self.count = 0
        self.unproved: list[str] = []
        self.scalar_gap = ""
        self.dim_exprs: list[sp.Expr] = []
        self.bund: dict[tuple[int, int], int] = {}
        self.bund_of: dict[int, tuple[int, int]] = {}
        self.bundles: dict[int, int] = {}
        self.mult_ix: dict[str, int] = {}
        self.mult_names: list[str] = []
        self.mult_exprs: list[sp.Expr] = []
        self.mult_bare: list[bool] = []
        self.ks_vals: list[str] = []
        self.ks_syms: list[sp.Expr] = []
        self.slots: tuple[int, ...] | None = None
        self.alias: dict[str, sp.Expr] = {}
        self.shadow: dict[str, int] = {}
        self.shadow_pairs: list[tuple[int, int]] = []
        self.mults: set[str] = set()
        self.sp_binders = self.sp_args = ""
        self.sp_hyps = self.sp_hyp_args = ""
        self.scalar_defs: list[str] = []
        self.unfold = "E"
        self.family = False
        self.sc = Scalars(
            nat=lambda sym: self.sym.name(sym, "ℕ"),
            other=lambda sym: (
                self.sym.name(sym, self.sym.kind_of(sym, "ℂ")),
                self.sym.kind_of(sym, "ℂ"),
            ),
            positive=lambda text: _POSITIVE,
        )

    # tables

    def canon(self, e: sp.Expr) -> sp.Expr:
        """The representative of ``e`` among the dimensions assumed equal."""
        while str(e) in self.alias:
            e = self.alias[str(e)]
        return e

    def raw_dim(self, d: Dim) -> int:
        """The index of ``d`` itself, apart from the dimensions assumed equal to it."""
        i = self.dim(d)
        e0 = d.to_sympy()
        if str(self.canon(e0)) == str(e0):
            return i
        key = str(e0)
        if key not in self.shadow:
            self.shadow[key] = len(self.dims)
            self.dims.append(self.sym.render(e0, "ℕ", "ℕ"))
            self.dim_exprs.append(e0)
            self.sc.nats |= {str(x) for x in e0.free_symbols}
            self.shadow_pairs.append((self.shadow[key], i))
        return self.shadow[key]

    def faithful(self, d: Diagram, lab: Labelled) -> tuple[LNet, dict[int, int]]:
        """``lab``'s network with each node and label at its own dimension's index."""
        labs: dict[int, int] = {}
        for ref, x in lab.port.items():
            j = self.raw_dim(_port_dim(d, ref))
            if labs.setdefault(x, j) != j:
                raise NetUnsupported("a wire joins dimensions assumed equal")
        nodes = []
        for nid, n in zip(lab.order, lab.net.nodes, strict=True):
            node = d.nodes[nid]
            legs = (*node.inputs, *node.outputs)
            if node.generator_type.name == "B":
                n = replace(
                    n, dim=self.raw_dim(node.inputs[0].dim), ph=self.raw_dim(node.inputs[1].dim)
                )
            elif node.generator_type.name == "S":
                n = replace(
                    n, dim=self.raw_dim(node.outputs[0].dim), ph=self.raw_dim(node.outputs[1].dim)
                )
            elif legs:
                n = replace(n, dim=self.raw_dim(legs[0].dim))
            nodes.append(n)
        return LNet(tuple(nodes), lab.net.summed, lab.net.bnd), labs

    def assume(self, a: sp.Expr, b: sp.Expr) -> None:
        """Identify the dimensions ``a`` and ``b``."""
        ra, rb = self.canon(a), self.canon(b)
        if ra != rb:
            lo, hi = sorted((ra, rb), key=str)
            self.alias[str(hi)] = lo

    def dim(self, d: Dim) -> int:
        e = self.canon(d.to_sympy())
        key = str(e)
        if key not in self.dim_key:
            self.dim_key[key] = len(self.dims)
            self.dims.append(self.sym.render(e, "ℕ", "ℕ"))
            self.dim_exprs.append(e)
            self.sc.nats |= {str(x) for x in e.free_symbols}
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
        weights = self.weights(pv) if pv.is_weighted else None
        key = body if weights is None else f"{body}|{weights}"
        if key not in self.phase_key:
            self.phase_key[key] = len(self.phases)
            if weights is not None:
                self.weight_tab[len(self.phases)] = weights
            self.phases.append(f"fun k => {body}")
            self.phase_specs.append(
                (
                    {i: ph.to_sympy_turns() for i, ph in pv.entries().items()},
                    None if pv.rule is None else pv.rule.to_sympy_turns(),
                )
            )
        return self.phase_key[key]

    def weights(self, pv: PhaseVector) -> str:
        """The coefficient vector of ``pv`` as the body of a Lean ``fun k``; its models kept in
        ``coef_models`` under the body."""
        _need(pv.dim.is_concrete, "complex weights at a symbolic dimension")
        models = {k: self.scalar(pv.coefficient(k)) for k in range(1, pv.dim.to_int())}
        body = "1"
        for index, model in sorted(models.items(), reverse=True):
            body = f"if k = {index} then {self.sc.text(model)} else {body}"
        self.coef_models[body] = models
        return body

    def kind_ph(self, kind: str, pv: PhaseVector | None, aux: int) -> tuple[str, int]:
        """A node's Lean kind and phase index: a weighted Z spider is ``ZW``."""
        if kind not in ("Z", "X"):
            return kind, aux
        ph = self.phase(pv)
        if pv is not None and pv.is_weighted:
            if kind != "Z":
                raise NetUnsupported("an X spider with complex weights")
            return "ZW", ph
        return kind, ph

    def label(self, dim: int) -> int:
        lab = self.next_label
        self.next_label += 1
        self.labs[lab] = dim
        self.peel_reg(lab, H_OFFSET)
        return lab

    def peel_reg(self, lab: int, off: int) -> None:
        """Declare ``lab + off`` with the dimension ``lab``'s bundle bundles."""
        dim = self.labs.get(lab)
        if dim in self.bund_of:
            self.labs.setdefault(lab + off, self.bund_of[dim][0])

    def scalar(self, s: Scalar) -> sp.Expr:
        try:
            return self.sc.model(s.to_sympy())
        except ScalarUnsupported as exc:
            raise NetUnsupported(str(exc)) from exc

    def text(self, k: object) -> str:
        """A scalar model or product tree as Lean."""
        if isinstance(k, _Named):
            return k.text
        if isinstance(k, tuple):
            return f"({self.text(k[1])} * {self.text(k[2])})"
        return self.sc.text(k)

    def drop_ones(self, k: object) -> str:
        """A product tree as Lean without the factors ``simp only [one_mul, mul_one]`` removes."""
        if isinstance(k, tuple):
            left, right = self.drop_ones(k[1]), self.drop_ones(k[2])
            if left == "1":
                return right
            if right == "1":
                return left
            return f"({left} * {right})"
        return self.sc.text(k)

    def nonzero(self, k: object) -> str:
        """A Lean proof that a product tree is nonzero."""
        if isinstance(k, tuple):
            return f"(mul_ne_zero {self.nonzero(k[1])} {self.nonzero(k[2])})"
        return self.sc.nonzero_expr(k)

    def flat(self, k: object) -> sp.Expr:
        """A product tree as one model."""
        if isinstance(k, _Named):
            return k.model
        if isinstance(k, tuple):
            return self.flat(k[1]) * self.flat(k[2])
        return k

    # networks

    def node_dims(self, d: Diagram, nid: NodeId) -> tuple[int, int]:
        node = d.nodes[nid]
        name = node.generator_type.name
        if name == "B":
            return self.dim(node.inputs[0].dim), self.dim(node.inputs[1].dim)
        if name == "S":
            return self.dim(node.outputs[0].dim), self.dim(node.outputs[1].dim)
        try:
            return self.dim(resolve_dim(node)), 0
        except DenoteDomainError:
            legs = {self.dim(p.dim) for p in (*node.inputs, *node.outputs)}
            if node.phase is not None:
                legs.add(self.dim(node.phase.dim))
            if len(legs) != 1:
                if not getattr(self, "mixed_ok", False):
                    raise NetUnsupported(f"node {nid} has legs of different dimensions") from None
                first = self.dim(next(iter((*node.inputs, *node.outputs))).dim)
                self.mixed |= {(i, first) for i in legs if i != first}
                return first, 0
            return legs.pop(), 0

    def mult_index(self, name: str) -> int:
        """The index of the multiplicity symbol ``name``."""
        return self.mult_of(_mult_symbol(name))

    def mult_of(self, expr: sp.Expr) -> int:
        """The index of the multiplicity ``expr``: a symbol, a number or a compound."""
        key = str(expr)
        if key not in self.mult_ix:
            if expr.is_Symbol:
                self.mults.add(key)
                self.sc.nats.add(key)
                text = self.sym.name(_mult_symbol(key), "ℕ")
            else:
                for x in sorted(expr.free_symbols, key=str):
                    self.mult_index(str(x))
                text = self.sym.render(expr, "ℕ", "ℕ")
            self.mult_ix[key] = len(self.mult_names)
            self.mult_names.append(text)
            self.mult_exprs.append(_mult_symbol(key) if expr.is_Symbol else expr)
            self.mult_bare.append(bool(expr.is_Symbol))
            self.family = True
            self.unfold = _FAMILY_UNFOLD
        return self.mult_ix[key]

    def node_chain(self, d: Diagram) -> dict[NodeId, tuple[tuple[int, ...], Chain]]:
        """Each boxed node's node boxes outermost first, with their multiplicity indices."""
        boxes = {bid: box for bid, box in d.bang_boxes.items() if box.node_scope}

        def depth(bid: BangBoxId) -> int:
            n, b = 0, boxes[bid].parent
            while b is not None:
                n += 1
                b = d.bang_boxes[b].parent if b in d.bang_boxes else None
            return n

        out: dict[NodeId, tuple[tuple[int, ...], Chain]] = {}
        for nid in d.nodes:
            mine = sorted((bid for bid, box in boxes.items() if nid in box.node_scope), key=depth)
            if not mine:
                continue
            if boxes[mine[0]].parent is not None or any(
                boxes[b].parent != a for a, b in pairwise(mine)
            ):
                raise NetUnsupported("node boxes that do not nest")
            out[nid] = (
                tuple(mine),
                tuple(self.mult_of(boxes[b].multiplicity.to_sympy()) for b in mine),
            )
        return out

    def bundle_dim(self, base: int, m: int) -> int:
        """The dimension index of a bundle along multiplicity ``m`` of the index ``base``."""
        if (base, m) not in self.bund:
            i = len(self.dims)
            self.bund[(base, m)] = i
            self.bund_of[i] = (base, m)
            self.dims.append(self.dims[base])
            self.dim_exprs.append(self.dim_exprs[base])
        return self.bund[(base, m)]

    def dim_for(self, base: int, chain: Chain) -> int:
        """The dimension index bundling ``base`` along ``chain``, outermost first."""
        idx = base
        for m in reversed(chain):
            idx = self.bundle_dim(idx, m)
        return idx

    def bundle_label(self, base: int, m: int) -> int:
        """A fresh bundle label along ``m`` of the index ``base``."""
        return self.label(self.bundle_dim(base, m))

    def chain(self, i: int) -> Chain:
        out: list[int] = []
        while i in self.bund_of:
            i, m = self.bund_of[i]
            out.append(m)
        return tuple(out)

    def root(self, i: int) -> int:
        while i in self.bund_of:
            i = self.bund_of[i][0]
        return i

    def label_chain(self, x: int) -> Chain:
        return self.chain(self.labs.get(x, 0))

    def isb(self, m: int) -> Callable[[int], bool]:
        """Whether a label is a bundle along ``m`` outermost."""
        return lambda x: self.label_chain(x)[:1] == (m,)

    def leg_tails(self, n: LNode, skip: int = 0) -> dict[tuple[str, int], Chain]:
        """``n``'s legs whose label bundles beyond its own ``skip`` copies, with the rest of
        their chain."""
        out: dict[tuple[str, int], Chain] = {}
        for side, legs in (("o", n.outs), ("i", n.ins)):
            for j, x in enumerate(legs):
                c = self.label_chain(x)[skip:]
                if c:
                    out[(side, j)] = c
        return out

    def env_text(self, vals: Sequence[str]) -> str:
        """The family's environment at the multiplicity values ``vals``."""
        fam = f"(F {self.sp_args})" if self.sp_args else "F"
        return f"({fam}.env (ksOf [{', '.join(vals)}]))"

    def fam_text(self) -> str:
        return f"(F {self.sp_args})" if self.sp_args else "F"

    def sdefs(self) -> str:
        """The scalar definitions to unfold in a scalar proof."""
        return ", ".join(self.scalar_defs)

    def bund_pos(self, vals: Sequence[str]) -> str:
        """A proof that every bundle base is positive."""
        return f"(fun _ _ _ => Fam.dimv_pos _ _ _ (Fpos {_join(self.sp_args, self.sp_hyp_args)} _))"

    def translate(
        self,
        d: Diagram,
        slots: Sequence[int] | None,
        alloc: Callable[[int, Chain], int],
        register: bool = True,
        dims_out: dict[int, int] | None = None,
    ) -> tuple[LNet, tuple[NodeId, ...], dict[PortRef, int]]:
        """``d`` as a network with boundary slot ``i`` labelled ``slots[i]``, internal labels
        from ``alloc(dim, chain)``."""
        nodes_of = self.node_chain(d)
        port_box: dict[PortRef, Chain] = {}
        holders: dict[PortRef, list[BangBoxId]] = {}
        for bid, box in sorted(d.bang_boxes.items()):
            for ref in box.port_scope:
                holders.setdefault(ref, []).append(bid)
        for ref, bids in holders.items():
            inner = nodes_of[ref.node_id][0][-1] if ref.node_id in nodes_of else None
            byid = {b: d.bang_boxes[b] for b in bids}
            top = [b for b in bids if byid[b].parent not in byid]
            if len(top) != 1 or byid[top[0]].parent != inner:
                raise NetUnsupported("a port box outside its node's innermost box")
            order = [top[0]]
            while len(order) < len(bids):
                nxt = [b for b in bids if byid[b].parent == order[-1]]
                if len(nxt) != 1:
                    raise NetUnsupported("port boxes that do not nest")
                order.append(nxt[0])
            port_box[ref] = tuple(self.mult_of(byid[b].multiplicity.to_sympy()) for b in order)
        outs, ins = d.boundary_outputs, d.boundary_inputs
        refs = (*outs, *ins)
        if slots is not None and len(slots) != len(refs):
            raise NetUnsupported("the boundary does not match")
        port: dict[PortRef, int] = {}
        tails: dict[PortRef, Chain] = {}
        for i, ref in enumerate(refs):
            base = self.dim(_port_dim(d, ref))
            own = nodes_of.get(ref.node_id, ((), ()))[1]
            extra = port_box.get(ref, ())
            want = self.dim_for(base, own + extra)
            if extra:
                tails[ref] = extra
            lab = slots[i] if slots is not None else i
            if register:
                if self.labs.setdefault(lab, want) != want:
                    raise NetUnsupported("a boundary slot changes dimension")
                self.peel_reg(lab, H_OFFSET)
            if dims_out is not None:
                dims_out[lab] = want
            port[ref] = lab
        internal = []
        for wire in sorted(d.wires, key=lambda w: w.sort_key()):
            da, db = _port_dim(d, wire.a), _port_dim(d, wire.b)
            if self.dim(da) != self.dim(db):
                raise NetUnsupported(f"a wire joins dimensions {da} and {db}")
            if wire.a in port_box or wire.b in port_box:
                raise NetUnsupported("a port box on a wire")
            ba = nodes_of.get(wire.a.node_id, ((), ()))
            bb = nodes_of.get(wire.b.node_id, ((), ()))
            if ba[0][: len(bb[0])] == bb[0]:
                chain, short, n_short = ba[1], wire.b, len(bb[1])
            elif bb[0][: len(ba[0])] == ba[0]:
                chain, short, n_short = bb[1], wire.a, len(ba[1])
            else:
                raise NetUnsupported("a wire between two node boxes")
            if len(chain) > n_short:
                tails[short] = chain[n_short:]
            lab = alloc(self.dim(da), chain)
            if dims_out is not None:
                dims_out[lab] = self.dim_for(self.dim(da), chain)
            port[wire.a] = port[wire.b] = lab
            internal.append(lab)
        node_order = tuple(sorted(d.nodes))
        nodes = []
        for nid in node_order:
            node = d.nodes[nid]
            kind = _KINDS.get(node.generator_type.name)
            if kind is None:
                raise NetUnsupported(f"generator {node.generator_type.name} has no Lean kind")
            legs = []
            pos: dict[tuple[str, int], Chain] = {}
            for direction, side in ((Direction.OUTPUT, "o"), (Direction.INPUT, "i")):
                row = []
                for i in range(len(node.legs(direction))):
                    ref = PortRef(nid, direction, i)
                    if ref not in port:
                        raise NetUnsupported(f"port {ref} is neither wired nor on the boundary")
                    row.append(port[ref])
                    if ref in tails:
                        pos[(side, i)] = tails[ref]
                legs.append(tuple(row))
            dim, aux = self.node_dims(d, nid)
            kind, ph = self.kind_ph(kind, node.phase, aux)
            o, i_ = legs
            bnode = _bundle_node(LNode(kind, o, i_, dim, ph), pos)
            k = bnode.kind
            if nid in nodes_of:
                if not _copyable(k):
                    raise NetUnsupported(f"a boxed {kind} node")
                for m in reversed(nodes_of[nid][1]):
                    k = ("Bun", m, k)
            nodes.append(LNode(k, bnode.outs, bnode.ins, dim, ph))
        bnd = tuple(slots) if slots is not None else tuple(range(len(refs)))
        return LNet(tuple(nodes), tuple(internal), bnd), node_order, port

    def labelled(self, d: Diagram, slots: Sequence[int] | None = None) -> Labelled:
        key = (self.structure_key(d), tuple(slots) if slots is not None else None)
        if key in self.nets:
            return self.nets[key]
        if slots is None and not d.bang_boxes:
            nb = len(d.boundary_outputs) + len(d.boundary_inputs)
            if self.nb is None:
                self.nb = nb
                self.next_label = max(self.next_label, nb)
            elif self.nb != nb:
                raise NetUnsupported("the diagrams have different boundaries")
        if slots is None:
            self.next_label = max(self.next_label, len(d.boundary_outputs) + len(d.boundary_inputs))

        def alloc(dim: int, chain: Chain) -> int:
            return self.label(self.dim_for(dim, chain))

        net, order, port = self.translate(d, slots, alloc)
        name = f"N{len(self.nets)}"
        labelled = Labelled(net, order, port, name, str(d.scalar.to_sympy()))
        self.nets[key] = labelled
        self.decls.append(f"/-- {_comment(self.describe(d))} -/\ndef {name} : Net := {net.lean()}")
        return labelled

    @staticmethod
    def structure_key(d: Diagram) -> object:
        boxes = tuple(
            (
                tuple(sorted(map(int, b.node_scope))),
                tuple(sorted(r.sort_key() for r in b.port_scope)),
            )
            for _, b in sorted(d.bang_boxes.items())
        )
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
        return (nodes, wires, boxes, d.boundary_inputs, d.boundary_outputs)

    # proofs

    def fresh(self, stem: str) -> str:
        self.count += 1
        return f"{stem}{self.count}"

    def statement(self, sl: str, a: Labelled, sr: str, b: Labelled) -> str:
        return f"Equal {self.env} {_paren(sl)} {a.name} {_paren(sr)} {b.name}"

    def axiom(self, stem: str, note: str, a: Labelled, b: Labelled, k: sp.Expr) -> Proof:
        name = self.fresh(f"unproved_{re.sub(r'[^A-Za-z0-9_]', '_', stem)}_")
        self.unproved.append(name)
        self.decls.append(
            f"/-- {_comment(note)} -/\naxiom {_join(name, self.binders)} : "
            f"{self.statement('1', a, self.text(k), b)}"
        )
        return Proof(f"({_join(name, self.args)})", ONE, k)

    def theorem(self, a: Labelled, b: Labelled, k: sp.Expr, tactic: list[str]) -> Proof:
        name = self.fresh("step")
        body = "\n".join("  " + line for line in tactic)
        self.decls.append(
            f"theorem {_join(name, self.binders, self.hyps)} :\n    "
            f"{self.statement('1', a, self.text(k), b)} := by\n{body}"
        )
        return Proof(f"({_join(name, self.args, self.hyp_args)})", ONE, k)

    def iso_term(self, s: str, a: LNet, b: LNet, pairs: Sequence[tuple[int, int]]) -> str:
        f = _ren(pairs)
        renamed = [n.rename(f) for n in a.nodes]
        used: set[int] = set()
        p = []
        for target in b.nodes:
            for j, n in enumerate(renamed):
                if j not in used and _equiv_b(n, target):
                    used.add(j)
                    p.append(j)
                    break
            else:
                raise NetUnsupported("the networks do not match node for node")
        return f"isoB {self.env} {_paren(s)} _ {b.lean()} {_pairs(pairs)} {_list(p)} (by chk)"

    def term(self, t: object) -> Proof:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(t, Theorem)
        inf = t.inference
        if inf.kind == "refl":
            lab = self.labelled(t.lhs, self.slots)
            return Proof(f"(Equal.refl {self.env} 1 {lab.name})", ONE, ONE)
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
        if inf.kind == "induction" and self.family:
            return self.nested_induction(t)
        raise NetUnsupported(f"a {inf.kind} inference has no network proof")

    def nested_induction(self, t: object) -> Proof:
        """An induction inside a proof, as its own theorem applied at the current values."""
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(t, Theorem)
        saved = (
            self.env,
            list(self.ks_vals),
            list(self.ks_syms),
            self.slots,
            self.binders,
            self.args,
            self.hyps,
            self.hyp_args,
        )
        idx = str(t.inference.data["index"])
        ks = str(t.inference.data["step"])
        alias = self.mult_ix.get(ks)
        self.prepare_induction(t, top=False)
        self.collect(t)
        name = self.fresh("ind")
        try:
            self.induction_lemma(t, name, f"{name}_")
        finally:
            (
                self.env,
                self.ks_vals,
                self.ks_syms,
                self.slots,
                self.binders,
                self.args,
                self.hyps,
                self.hyp_args,
            ) = saved
            if alias is None:
                self.mult_ix.pop(ks, None)
            else:
                self.mult_ix[ks] = alias
        mi = self.mult_ix[idx]
        oth = [j for j in range(len(self.mult_names)) if j != mi and self.mult_bare[j]]
        lead = _join(self.sp_args, *(_paren(self.ks_vals[j]) for j in oth))
        cur = _paren(self.ks_vals[mi])
        sub = {
            sym: self.ks_syms[j]
            for j in range(len(self.mult_names))
            for sym in [self.mult_exprs[j]]
            if self.mult_bare[j]
        }

        def leaf(stem: str, model: sp.Expr) -> _Named:
            return _Named(f"({_join(f'{name}_{stem}', lead, cur)})", model.xreplace(sub))

        left = leaf("sS", self.scalar(t.lhs.scalar))
        right = leaf("sG", self.scalar(t.rhs.scalar))
        return Proof(f"({_join(name, lead, self.sp_hyp_args, cur)})", left, right)

    def iso(self, x: Diagram, y: Diagram) -> Proof:
        a, b = self.labelled(x, self.slots), self.labelled(y, self.slots)
        if a.name == b.name:
            return Proof(f"(Equal.refl {self.env} 1 {a.name})", ONE, ONE)
        try:
            pairs = _iso_pairs(x, y, a, b)
            tactic = self.iso_term("1", a.net, b.net, pairs)
        except NetUnsupported as exc:
            return self.axiom("iso", f"isomorphic comparison views ({exc})", a, b, ONE)
        return self.theorem(a, b, ONE, [f"exact {tactic}"])

    def rule(self, t: object) -> Proof:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(t, Theorem)
        step = cast(RewriteStep | None, t.inference.data.get("step"))
        a, b = self.labelled(t.lhs, self.slots), self.labelled(t.rhs, self.slots)
        builder = _RULES.get(t.inference.rule)
        engine_k = self.scalar(step.scalar_introduced) if step is not None else ONE
        if step is not None and builder is None:
            reverse = _through_forward(t)
            if reverse is not None:
                return self.term(reverse)
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
    ) -> tuple[object, list[str]]:
        new_ids = tuple(step.new_node_ids)
        removed = [nid for nid in before.nodes if nid not in after.nodes]
        nb_after, nb_before = self.node_chain(after), self.node_chain(before)
        boxed = {nb_after[nid] for nid in new_ids if nid in nb_after}
        boxed |= {nb_before[nid] for nid in removed if nid in nb_before}
        if boxed:
            return self.bundled_tactic(before, after, step, a, b, builder, boxed)
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
            kind, ph = self.kind_ph(_KINDS[node.generator_type.name], node.phase, aux)
            plain = LNode(kind, legs[0], legs[1], dim, ph)
            new_nodes[nid] = _bundle_node(plain, self.leg_tails(plain))
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
        l_nodes = (
            spec.l if spec.l is not None else tuple(net.nodes[index[nid]] for nid in spec.l_ids)
        )
        r_nodes = tuple(n for _, n in spec.r)
        lemma = spec.lemma
        if "IFACE" in lemma:
            lemma = lemma.replace("IFACE", _list(_iface(l_nodes, i_l)))
        return spec.k, self.region_lines(
            a, b, spec.l_ids, l_nodes, r_nodes, [nid for nid, _ in spec.r], i_l, i_r, spec.k, lemma
        )

    def region_lines(
        self,
        a: Labelled,
        b: Labelled,
        l_ids: Sequence[NodeId],
        l_nodes: tuple[LNode, ...],
        r_nodes: tuple[LNode, ...],
        r_ids: Sequence[NodeId | None],
        i_l: tuple[int, ...],
        i_r: tuple[int, ...],
        k_model: sp.Expr,
        lemma: str,
    ) -> list[str]:
        """The step's tactic: reorder, rewrite the region with ``lemma``, drop bare wires,
        reorder onto ``b``."""
        net = a.net
        index = {nid: i for i, nid in enumerate(a.order)}
        inner = set(l_ids)
        c_ids = [nid for nid in a.order if nid not in inner]
        c_nodes = tuple(net.nodes[index[nid]] for nid in c_ids)
        i_c = tuple(x for x in net.summed if x not in i_l)
        if len(i_c) + len(i_l) != len(net.summed) or len(set(i_l)) != len(i_l):
            raise NetUnsupported("a consumed wire label is not summed")
        bnd = net.bnd
        iface = _iface(l_nodes, i_l)
        k = self.text(k_model)
        a1 = LNet(c_nodes + l_nodes, i_c + i_l, bnd)
        lines = [f"refine Equal.trans ({self.iso_term('1', net, a1, [])}) ?_"]
        lines.append(
            f"refine Equal.trans (rewrite {self.env} {_nodes(c_nodes)} {_nodes(l_nodes)} "
            f"{_nodes(r_nodes)} {_list(i_c)} {_list(i_l)} {_list(i_r)} {_list(bnd)} "
            f"{_list(iface)} 1 {_paren(k)} {_paren(k)} ({lemma}) (by chk) (by chk) (by chk) "
            f"(one_mul _).symm) ?_"
        )
        current = LNet(c_nodes + r_nodes, i_c + i_r, bnd)
        ids: list[NodeId | None] = [*c_ids, *r_ids]
        while any(_tower_ms(n.kind) for n in current.nodes):
            pos = next(j for j, n in enumerate(current.nodes) if _tower_ms(n.kind))
            n = current.nodes[pos]
            ms = _tower_ms(n.kind)
            assert ms is not None
            x, w = n.outs[0], n.ins[0]
            rest = current.nodes[:pos] + current.nodes[pos + 1 :]
            bz = self.labs[x]
            plain = LNode("I", (x,), (w,), bz, 0)
            target = LNet(rest + (n,), current.summed, bnd)
            lines.append(f"refine Equal.trans ({self.iso_term(k, current, target, [])}) ?_")
            lemma = (
                f"bunI {self.env} {_list(ms)} {n.dim} {bz} {x} {w} {_list([x, w])} "
                f"(Fam.env_dimv_of _ _ _ _ _ rfl rfl ⟨rfl, rfl⟩) (by pos) {_chks(4)}"
            )
            lines.append(
                f"refine Equal.trans (rewrite {self.env} {_nodes(rest)} {_nodes((n,))} "
                f"{_nodes((plain,))} {_list(current.summed)} [] [] {_list(bnd)} {_list([x, w])} "
                f"{_paren(k)} 1 {_paren(k)} ({lemma}) {_chks(3)} (mul_one _).symm) ?_"
            )
            current = LNet(rest + (plain,), current.summed, bnd)
            ids = ids[:pos] + ids[pos + 1 :] + [ids[pos]]
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
        try:
            pairs = _match_labels(current, ids, b)
            final = self.iso_term(k, current, b.net, pairs)
        except NetUnsupported:
            m = _match(current, b.net, self.dims_of, self.dims_of, fix_bnd=True)  # type: ignore[arg-type]
            if m is None:
                raise
            pairs = _perm_to_bijection({x: y for x, y in m.items() if x in set(current.summed)})
            final = self.iso_term(k, current, b.net, pairs)
        lines.append(f"exact {final}")
        return lines

    # whole export

    def finish(self, theorem: object, name: str) -> str:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(theorem, Theorem)
        self.env = "E"
        self.binders = self.args = self.hyps = self.hyp_args = ""
        if theorem.inference.kind == "induction":
            return self.finish_induction(theorem, name)
        for v in theorem.variables:
            self.mult_index(str(v))
        start, goal = self.labelled(theorem.lhs), self.labelled(theorem.rhs)
        if self.alias:
            self.faithful(theorem.lhs, start)
            self.faithful(theorem.rhs, goal)
        self.collect(theorem)
        try:
            self.engine_scalars: tuple[sp.Expr, sp.Expr] | None = (
                self.scalar(theorem.lhs.scalar),
                self.scalar(theorem.rhs.scalar),
            )
        except NetUnsupported:
            self.engine_scalars = None
        self.set_params()
        if self.family:
            self.sp_binders, self.sp_args = self.binders, self.args
            self.sp_hyps, self.sp_hyp_args = self.hyps, self.hyp_args
            bare = [n for n, b in zip(self.mult_names, self.mult_bare, strict=True) if b]
            self.ks_vals = list(self.mult_names)
            self.ks_syms = list(self.mult_exprs)
            self.binders = _join(self.binders, *(f"({n} : ℕ)" for n in bare))
            self.args = _join(self.args, *bare)
            self.env = self.env_text(self.ks_vals)
        if self.alias and self.family:
            raise NetUnsupported("a family whose proof assumes dimensions equal")
        outer = name
        if self.alias:
            name = f"{name}M"
        proof = self.term(theorem)
        head = f"theorem {_join(name, self.binders, self.hyps)} :\n    "
        try:
            if self.engine_scalars is None:
                raise NetUnsupported("a diagram scalar has no Lean form")
            s_start, s_goal = self.engine_scalars
            hst = self.sc.prove(s_start * self.flat(proof.right), s_goal * self.flat(proof.left))
            self.decls.append(
                f"/-- The start and goal diagrams with their own scalars. -/\n{head}"
                f"{self.statement(self.text(s_start), start, self.text(s_goal), goal)} :=\n"
                f"  Equal.rescale {proof.term}\n    "
                + (
                    f"(by (try simp only [{self.sdefs()}]); nzt)\n    "
                    f"(by simp only [{self.sdefs()}]; {hst[3:]})"
                    if self.scalar_defs
                    else f"{self.nonzero(proof.left)}\n    ({hst})"
                )
            )
        except (ScalarUnsupported, NetUnsupported) as exc:
            self.scalar_gap = str(exc)
            left, right = self.drop_ones(proof.left), self.drop_ones(proof.right)
            self.decls.append(
                f"/-- {_comment(f'start = {start.scalar_note}, goal = {goal.scalar_note}')} -/\n"
                f"{head}{self.statement(left, start, right, goal)} := by\n"
                f"  simpa only [one_mul, mul_one] using {proof.term}"
            )
        if self.alias:
            self.unmerge(theorem, outer, name, start, goal)
        return self.render(theorem)

    def unmerge(
        self, theorem: object, outer: str, inner: str, start: Labelled, goal: Labelled
    ) -> None:
        """The theorem ``outer`` over the networks at their own dimensions, the assumed
        equalities as hypotheses, from ``inner`` over the merged networks."""
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(theorem, Theorem)
        fa, la = self.faithful(theorem.lhs, start)
        fb, lb = self.faithful(theorem.rhs, goal)
        for x, j in lb.items():
            la.setdefault(x, j)
        if not self.shadow_pairs:
            raise NetUnsupported("no dimension of the start or goal is assumed equal to another")
        labs_f = dict(self.labs)
        labs_f.update(la)
        pairs = self.shadow_pairs
        self.constraints = [f"{_paren(self.dims[i])} = {_paren(self.dims[j])}" for j, i in pairs]
        hyps = " ".join(f"(hc{k} : {c})" for k, c in enumerate(self.constraints))
        hcs = ", ".join(f"hc{k}" for k in range(len(pairs)))
        envf = f"(EF {self.args})" if self.args else "EF"
        envm = f"(E {self.args})" if self.args else "E"
        ps = _pairs(pairs)
        alts = " | ".join(["rfl"] * len(pairs))
        sl = self.sc.text(self.engine_scalars[0]) if self.engine_scalars else "1"
        sr = self.sc.text(self.engine_scalars[1]) if self.engine_scalars else "1"
        if self.scalar_gap:
            raise NetUnsupported("the diagrams' scalars are not compared")
        self.decls += [
            (
                f"/-- Each label's own dimension index. -/\ndef labsF : List (ℕ × ℕ) := "
                f"{_pairs(sorted(labs_f.items()))}"
            ),
            (
                f"/-- The environment at the diagrams' own dimensions. -/\nnoncomputable def EF "
                f"{self.binders} : Env := {{ dimv := Δ {self.args}, phase := Φ {self.args}, "
                f"labs := labsF"
                f"{f', coef := Ψ {self.args}'.rstrip() if self.weight_tab else ''} }}"
            ).replace("  ", " "),
            *([f"/-- The start at its own dimensions. -/\ndef {start.name}F : Net := {fa.lean()}"]),
            *(
                [f"/-- The goal at its own dimensions. -/\ndef {goal.name}F : Net := {fb.lean()}"]
                if goal.name != start.name
                else []
            ),
            (
                f"/-- The diagrams are equal wherever the assumed dimension equalities hold. -/\n"
                f"theorem {_join(outer, self.binders, self.hyps, hyps)} :\n"
                f"    Equal {envf} {_paren(sl)} {start.name}F {_paren(sr)} {goal.name}F :=\n"
                f"  Equal.of_mapDim' {envf} {envm} {ps} (by\n"
                f"    intro p hp\n    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp\n"
                f"    rcases hp with {alts} <;> simp [EF, Δ, {hcs}])\n"
                f"    {start.name}F {goal.name}F {start.name} {goal.name} (by decide) (by decide)\n"
                f"    (by simp only [EF, E]; congr 1)\n"
                f"    {_paren(_join(inner, self.args, self.hyp_args))}"
            ),
        ]

    # families

    def dims_of(self, x: int) -> int | None:
        return self.labs.get(x)

    def slots_for(self, d: Diagram, target: LNet) -> tuple[int, ...]:
        """Boundary labels for ``d`` matching it onto ``target`` node for node."""
        prov: dict[int, int] = {}
        counter = [10**9]

        def alloc(dim: int, chain: Chain) -> int:
            counter[0] += 1
            return counter[0]

        nb = len(d.boundary_outputs) + len(d.boundary_inputs)
        slots = tuple(2 * 10**9 + i for i in range(nb))
        net, _, _ = self.translate(d, slots, alloc, register=False, dims_out=prov)
        m = _match(net, target, prov.get, self.dims_of, fix_bnd=False)  # type: ignore[arg-type]
        if m is None:
            raise NetUnsupported("a diagram does not match its family form")
        return tuple(m[x] for x in slots)

    def iso_b(self, env: str, a_text: str, a: LNet, b_text: str, b: LNet) -> str:
        """``isoB`` from ``a`` to ``b`` relabelling their summed labels."""
        m = _match(a, b, self.dims_of, self.dims_of, fix_bnd=True)  # type: ignore[arg-type]
        if m is None:
            raise NetUnsupported("two networks of the proof do not match")
        pairs = _perm_to_bijection({x: y for x, y in m.items() if x in set(a.summed)})
        f = _ren(pairs)
        renamed = [n.rename(f) for n in a.nodes]
        used: set[int] = set()
        p = []
        for target in b.nodes:
            for j, n in enumerate(renamed):
                if j not in used and _equiv_b(n, target):
                    used.add(j)
                    p.append(j)
                    break
            else:
                raise NetUnsupported("the networks do not match node for node")
        return f"(isoB {env} 1 {a_text} {b_text} {_pairs(pairs)} {_list(p)} (by chk))"

    def fresh_like(self, x: int) -> int:
        """A fresh label of ``x``'s dimension, a bundle when ``x`` is one."""
        return self.label(self.labs[x])

    def finish_induction(self, t: object, name: str) -> str:
        self.prepare_induction(t, top=True)
        self.collect(t)
        self.set_params()
        self.sp_binders, self.sp_args = self.binders, self.args
        self.sp_hyps, self.sp_hyp_args = self.hyps, self.hyp_args
        self.induction_lemma(t, name, "")
        return self.render(t)

    def prepare_induction(self, t: object, top: bool) -> tuple[str, str]:
        """Register an induction's index and step symbols; their Lean names."""
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(t, Theorem)
        data = t.inference.data
        idx, ks = str(data["index"]), str(data["step"])
        if idx not in map(str, t.variables):
            raise NetUnsupported("an induction over a fixed multiplicity")
        if top:
            self.sym.params[idx] = ("n", "ℕ")
            self.sym.params[ks] = ("k", "ℕ")
        else:
            mi0 = self.mult_ix.get(idx)
            nn = self.mult_names[mi0] if mi0 is not None else self.sym.name(_mult_symbol(idx), "ℕ")
            taken = set(self.mult_names) | {n for n, _ in self.sym.params.values()}
            kk = f"{nn}k"
            while kk in taken:
                kk += "'"
            self.sym.params.setdefault(ks, (kk, "ℕ"))
        mi = self.mult_index(idx)
        self.mult_ix[ks] = mi
        self.mults.add(ks)
        self.sc.nats.add(ks)
        for v in t.variables:
            self.mult_index(str(v))
        return self.mult_names[mi], self.sym.params[ks][0]

    def induction_lemma(self, t: object, name: str, pre: str) -> None:
        """``t`` as the Lean theorem ``name`` by induction, its helpers prefixed ``pre``."""
        from archytaszx.semantics.kernel import Theorem
        from archytaszx.semantics.prove import InductionSide

        assert isinstance(t, Theorem)
        data = t.inference.data
        base_t, expose_t, step_t = t.inference.premises
        idx, ks = str(data["index"]), str(data["step"])
        nn = self.mult_names[self.mult_ix[idx]]
        kk = self.sym.params[ks][0]
        sS, sG, zc, sc_ = f"{pre}sS", f"{pre}sG", f"{pre}zcase", f"{pre}scase"
        mi = self.mult_ix[idx]
        start, goal = self.labelled(t.lhs, self.slots), self.labelled(t.rhs, self.slots)
        names = list(self.mult_names)
        syms = list(self.mult_exprs)
        if any(
            not self.mult_bare[j] and idx in map(str, syms[j].free_symbols)
            for j in range(len(names))
        ):
            raise NetUnsupported("a box whose multiplicity depends on the induction index")
        oth = [j for j in range(len(names)) if j != mi and self.mult_bare[j]]
        binders = _join(self.sp_binders, *(f"({names[j]} : ℕ)" for j in oth))
        args = _join(self.sp_args, *(names[j] for j in oth))
        self.binders, self.args = binders, args
        self.hyps, self.hyp_args = self.sp_hyps, self.sp_hyp_args
        hyps_args = _join(args, self.hyp_args)
        fam = self.fam_text()

        def vals(v: str) -> list[str]:
            return [v if j == mi else names[j] for j in range(len(names))]

        def lst(v: str) -> str:
            return "[" + ", ".join(vals(v)) + "]"

        def ret(x: str, *more: str) -> str:
            return "(" + _join(x, args, *more) + ")"

        def enter(v: str, sym: sp.Expr) -> None:
            self.ks_vals = vals(v)
            self.ks_syms = [sym if j == mi else syms[j] for j in range(len(names))]
            self.env = self.env_text(self.ks_vals)

        n_sym = syms[mi]
        k_sym = _mult_symbol(ks)
        s_s, s_g = self.scalar(t.lhs.scalar), self.scalar(t.rhs.scalar)

        def at(model: sp.Expr, value: object) -> sp.Expr:
            return model.xreplace({sym: value for sym in model.free_symbols if str(sym) == idx})

        bind = _join(binders, f"({nn} : ℕ)")
        self.scalar_defs += [sS, sG]
        self.decls.append(
            f"/-- The start's scalar. -/\nnoncomputable def {sS} {bind} : ℂ := {self.sc.text(s_s)}"
        )
        self.decls.append(
            f"/-- The goal's scalar. -/\nnoncomputable def {sG} {bind} : ℂ := {self.sc.text(s_g)}"
        )
        isb = self.isb(mi)
        ns, ng = start.name, goal.name
        H = H_OFFSET
        spc = f"(SpC {mi})"

        # n = 0
        z0, g0 = _zero_net(start.net, mi, isb), _zero_net(goal.net, mi, isb)
        self.slots = self.slots_for(base_t.lhs, z0)
        enter("0", sp.Integer(0))
        p0 = self.term(base_t)
        lb, rb = self.labelled(base_t.lhs, self.slots), self.labelled(base_t.rhs, self.slots)
        i1 = self.iso_b(self.env, f"(Net.zero {spc} {ns})", z0, lb.name, lb.net)
        i2 = self.iso_b(self.env, rb.name, rb.net, f"(Net.zero {spc} {ng})", g0)
        left = ("*", ONE, ("*", p0.left, ONE))
        right = ("*", ONE, ("*", p0.right, ONE))
        hst = self.sc.prove(at(s_s, 0) * self.flat(right), at(s_g, 0) * self.flat(left))
        self.decls.append(
            f"theorem {zc} {_join(binders, self.hyps)} :\n    Equal {self.env} "
            f"{ret(sS, '0')} (Net.zero {spc} {ns}) {ret(sG, '0')} (Net.zero {spc} {ng}) :=\n"
            f"  Equal.rescale (Equal.mul_trans {i1} (Equal.mul_trans {p0.term} {i2}))\n"
            f"    (by (try simp only [{self.sdefs()}]); nzt)\n"
            f"    (by simp only [{self.sdefs()}]; {hst[3:]})"
        )

        # k + 1
        start_side = data["side"] is InductionSide.START
        own, other = (start, goal) if start_side else (goal, start)
        s_own, s_oth = (sS, sG) if start_side else (sG, sS)
        split_own = _split_net(own.net, mi, isb)
        split_oth = _split_net(other.net, mi, isb)
        peeled = cast(Diagram, data["peeled"])
        self.slots = self.slots_for(peeled, split_own)
        enter(kk, k_sym)
        self.binders = _join(binders, f"({kk} : ℕ)")
        self.args = _join(args, kk)
        lp = self.labelled(peeled, self.slots)
        ia = self.iso_b(self.env, f"(Net.split {spc} {H} {own.name})", split_own, lp.name, lp.net)
        pe = self.term(expose_t)
        lx = self.labelled(cast(Diagram, data["exposed"]), self.slots)
        region = set(cast("frozenset[NodeId]", data["region"]))
        ctx = tuple(n for nid, n in zip(lx.order, lx.net.nodes, strict=True) if nid not in region)
        reg = tuple(n for nid, n in zip(lx.order, lx.net.nodes, strict=True) if nid in region)
        ctx_labels = {x for n in ctx for x in n.labels()}
        reg_labels = [x for n in reg for x in n.labels()]
        inner = [x for x in lx.net.summed if x in set(reg_labels) and x not in ctx_labels]
        iface = tuple(dict.fromkeys(x for x in reg_labels if x not in set(inner)))
        sub = LNet(reg, tuple(inner), iface)
        m = _match(own.net, sub, self.dims_of, self.dims_of, fix_bnd=False)  # type: ignore[arg-type]
        if m is None:
            raise NetUnsupported("the hypothesis region does not match")
        for x in other.net.summed:
            m.setdefault(x, self.fresh_like(x))
        pairs = _perm_to_bijection(m)
        pt = _pairs(pairs)
        ro, rx = _rename_net(own.net, pairs), _rename_net(other.net, pairs)
        i_c = tuple(x for x in lx.net.summed if x not in set(inner))
        bnd = lx.net.bnd
        plug_a = LNet(ctx + ro.nodes, i_c + ro.summed, bnd)
        plug_b = LNet(ctx + rx.nodes, i_c + rx.summed, bnd)

        def plug_text(o: str) -> str:
            return (
                f"⟨{_nodes(ctx)} ++ (Net.rename {pt} {o}).nodes, "
                f"{_list(i_c)} ++ (Net.rename {pt} {o}).int, {_list(bnd)}⟩"
            )

        ib = self.iso_b(self.env, lx.name, lx.net, plug_text(own.name), plug_a)
        ih = "ih" if start_side else "ih.symm"
        hr = f"(Equal.rename {self.env} {pt} {own.name} {other.name} (by chk) (by chk) {ih})"
        plug = (
            f"(plug {self.env} {_nodes(ctx)} (Net.rename {pt} {own.name}).nodes "
            f"(Net.rename {pt} {other.name}).nodes {_list(i_c)} "
            f"(Net.rename {pt} {own.name}).int (Net.rename {pt} {other.name}).int {_list(bnd)} "
            f"(Net.rename {pt} {own.name}).bnd {ret(s_own, kk)} {ret(s_oth, kk)} "
            f"(Equal.toLocal _ _ _ {hr}) (by chk) (by chk) (by chk))"
        )
        lr = self.labelled(cast(Diagram, data["replaced"]), self.slots)
        ic = self.iso_b(self.env, plug_text(other.name), plug_b, lr.name, lr.net)
        ps = self.term(step_t)
        lo = self.labelled(cast(Diagram, data["other_peeled"]), self.slots)
        idd = self.iso_b(
            self.env, lo.name, lo.net, f"(Net.split {spc} {H} {other.name})", split_oth
        )
        own_k = at(s_s if start_side else s_g, k_sym)
        oth_k = at(s_g if start_side else s_s, k_sym)
        own_leaf = _Named(ret(s_own, kk), own_k)
        oth_leaf = _Named(ret(s_oth, kk), oth_k)
        chain = (
            f"(Equal.mul_trans {ia} (Equal.mul_trans {pe.term} (Equal.mul_trans {ib} "
            f"(Equal.mul_trans {plug} (Equal.mul_trans {ic} (Equal.mul_trans {ps.term} "
            f"{idd}))))))"
        )
        left = (
            "*",
            ONE,
            ("*", pe.left, ("*", ONE, ("*", own_leaf, ("*", ONE, ("*", ps.left, ONE))))),
        )
        right = (
            "*",
            ONE,
            ("*", pe.right, ("*", ONE, ("*", oth_leaf, ("*", ONE, ("*", ps.right, ONE))))),
        )
        if not start_side:
            chain = f"({chain}).symm"
            left, right = right, left
        k1 = k_sym + 1
        hst = self.sc.prove(at(s_s, k1) * self.flat(right), at(s_g, k1) * self.flat(left))
        env_k = self.env
        self.binders, self.args = binders, args
        self.decls.append(
            f"theorem {sc_} {_join(binders, self.hyps)} ({kk} : ℕ)\n"
            f"    (ih : Equal {env_k} {ret(sS, kk)} {ns} {ret(sG, kk)} {ng}) :\n"
            f"    Equal {env_k} {ret(sS, f'({kk} + 1)')} (Net.split {spc} {H} {ns}) "
            f"{ret(sG, f'({kk} + 1)')} (Net.split {spc} {H} {ng}) :=\n"
            f"  Equal.rescale {chain}\n"
            f"    (by (try simp only [{self.sdefs()}]); nzt)\n"
            f"    (by simp only [{self.sdefs()}]; {hst[3:]})"
        )

        # the theorem
        ok_s = f"(by decide : Spec.splitOK {spc} {H} {ns} = true)"
        ok_g = f"(by decide : Spec.splitOK {spc} {H} {ng} = true)"
        enter(nn, n_sym)
        self.decls.append(
            f"/-- The start and goal families with their own scalars, for every `{nn}`. -/\n"
            f"theorem {name} {_join(binders, self.hyps)} :\n"
            f"    ∀ {nn} : ℕ, Equal {self.env} {ret(sS, nn)} {ns} {ret(sG, nn)} {ng} := by\n"
            f"  intro {nn}\n  induction {nn} with\n"
            f"  | zero => exact Equal.of_zero_at {fam} {mi} {H} {lst('0')} (by simp) {ns} {ng} "
            f"{ok_s} {ok_g} (by decide) ({zc} {hyps_args})\n"
            f"  | succ {kk} ih => exact Equal.of_split_at {fam} {mi} {H} {kk} {lst(kk)} (by simp) "
            f"{ns} {ng} {ok_s} {ok_g} {self.bund_pos(vals(kk))} "
            f"({sc_} {_join(hyps_args, kk, 'ih')})"
        )

    def bundled_tactic(
        self,
        before: Diagram,
        after: Diagram,
        step: RewriteStep,
        a: Labelled,
        b: Labelled,
        builder: _RuleBuilder,
        boxed: set[tuple[tuple[int, ...], Chain]],
    ) -> tuple[object, list[str]]:
        """A rule inside node boxes: the base rule on the peeled copies, lifted to every
        multiplicity of each box by induction."""
        if not self.family or not self.ks_vals or len(boxed) != 1:
            raise NetUnsupported("a boxed rewrite across boxes")
        ((boxes, chain),) = boxed
        nb_after = self.node_chain(after)
        mapping: dict[PortRef, PortRef] = dict(step.port_mapping)
        inverse = {new: old for old, new in mapping.items()}
        fresh: dict[Wire, int] = {}
        new_top: dict[NodeId, LNode] = {}
        for nid in step.new_node_ids:
            if nb_after.get(nid) != (boxes, chain):
                raise NetUnsupported("a boxed rewrite reaching outside its box")
            node = after.nodes[nid]
            kind = _KINDS[node.generator_type.name]
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
                        fresh[wire] = self.label(
                            self.dim_for(self.dim(_port_dim(after, ref)), chain)
                        )
                    row.append(fresh[wire])
                legs.append(tuple(row))
            dim, aux = self.node_dims(after, nid)
            kind, ph = self.kind_ph(kind, node.phase, aux)
            plain = LNode(kind, legs[0], legs[1], dim, ph)
            base = _bundle_node(plain, self.leg_tails(plain, len(chain)))
            k: Any = base.kind
            for m in reversed(chain):
                k = ("Bun", m, k)
            new_top[nid] = LNode(k, base.outs, base.ins, dim, ph)
        deltas: list[LNode] = []
        for old, new in mapping.items():
            if new.node_id in new_top:
                continue
            x, w = a.port[old], a.port[new]
            if self.label_chain(w) != chain:
                raise NetUnsupported("a boxed rewrite rewiring a bundled leg")
            deltas.append(LNode(_tower(chain), (x,), (w,), self.root(self.labs[w]), 0))
        removed = [nid for nid in before.nodes if nid not in after.nodes]
        spec = self.lift(
            chain,
            0,
            a,
            new_top,
            deltas,
            sorted(fresh.values()),
            removed,
            before,
            after,
            step,
            builder,
            list(self.ks_vals),
            list(self.ks_syms),
        )
        l_nodes = (
            spec.l
            if spec.l is not None
            else tuple(a.net.nodes[a.order.index(nid)] for nid in spec.l_ids)
        )
        r_nodes = tuple(n for _, n in spec.r)
        assert spec.i_l is not None and spec.i_r is not None
        return spec.k, self.region_lines(
            a,
            b,
            spec.l_ids,
            l_nodes,
            r_nodes,
            [nid for nid, _ in spec.r],
            spec.i_l,
            spec.i_r,
            spec.k,
            spec.lemma,
        )

    def lift(
        self,
        chain: Chain,
        depth: int,
        view: Labelled,
        new_view: dict[NodeId, LNode],
        deltas: list[LNode],
        fresh: list[int],
        removed: list[NodeId],
        before: Diagram,
        after: Diagram,
        step: RewriteStep,
        builder: _RuleBuilder,
        vals: list[str],
        syms: list[Any],
    ) -> _Spec:
        """The rule's local equation in ``view``, whose region copies along ``chain``."""
        if not chain:
            spec = builder(_Ctx(self, before, after, step, view, new_view, deltas))
            inner_ids = set(spec.l_ids)
            i_l = (
                tuple(spec.i_l)
                if spec.i_l is not None
                else tuple(
                    view.port[w.a]
                    for w in step.consumed_wires
                    if w.a.node_id in inner_ids and w.b.node_id in inner_ids
                )
            )
            i_r = tuple(spec.i_r) if spec.i_r is not None else tuple(sorted(set(fresh)))
            lemma = spec.lemma
            if "IFACE" in lemma:
                index = {nid: i for i, nid in enumerate(view.order)}
                l_nodes = (
                    spec.l
                    if spec.l is not None
                    else tuple(view.net.nodes[index[nid]] for nid in spec.l_ids)
                )
                lemma = lemma.replace("IFACE", _list(_iface(l_nodes, i_l)))
            return replace(spec, i_l=i_l, i_r=i_r, lemma=lemma)
        mb = chain[0]
        off = H_OFFSET << depth
        isb = self.isb(mb)

        def up(x: int) -> int:
            if isb(x):
                self.peel_reg(x, off)
                return x + off
            return x

        def base_of(n: LNode) -> LNode:
            if _head(n.kind) != "Bun" or n.kind[1] != mb:
                return n
            return LNode(n.kind[2], tuple(map(up, n.outs)), tuple(map(up, n.ins)), n.dim, n.ph)

        vb = Labelled(
            LNet(tuple(map(base_of, view.net.nodes)), view.net.summed, view.net.bnd),
            view.order,
            {r: up(x) for r, x in view.port.items()},
            view.name,
            view.scalar_note,
        )
        taken = set(self.mult_names) | {n for n, _ in self.sym.params.values()}
        names = list(self.mult_names)
        for k in range(len(names)):
            if not self.mult_bare[k]:
                c = f"c{k}"
                while c in taken:
                    c += "'"
                names[k] = c
                taken.add(c)
        oth = [k for k in range(len(names)) if k != mb]
        jv = "j"
        while jv in taken:
            jv += "'"
        jsym = sp.Symbol(f"lift_{jv}", integer=True, nonnegative=True)
        vals_j = [jv if k == mb else names[k] for k in range(len(names))]
        syms_j = [jsym if k == mb else self.mult_exprs[k] for k in range(len(names))]
        env_j = self.env_text(vals_j)
        saved_env = self.env
        self.env = env_j
        try:
            inner = self.lift(
                chain[1:],
                depth + 1,
                vb,
                {n: base_of(x) for n, x in new_view.items()},
                [base_of(x) for x in deltas],
                [up(x) for x in fresh],
                removed,
                before,
                after,
                step,
                builder,
                vals_j,
                syms_j,
            )
        finally:
            self.env = saved_env
        index = {nid: i for i, nid in enumerate(view.order)}
        lb = (
            inner.l
            if inner.l is not None
            else tuple(vb.net.nodes[index[nid]] for nid in inner.l_ids)
        )
        rb = tuple(n for _, n in inner.r)
        assert inner.i_l is not None and inner.i_r is not None
        i_lb, i_rb = inner.i_l, inner.i_r
        if_b = list(_iface(lb, i_lb))
        labels_b = {x for n in lb + rb for x in n.labels()} | set(i_lb) | set(i_rb)
        if any(x < off for x in labels_b):
            raise NetUnsupported("a boxed rewrite on unbundled legs")
        if any(not _copyable(n.kind) for n in rb):
            raise NetUnsupported("a boxed rule producing an uncopyable node")

        def down_l(n: LNode) -> LNode:
            return LNode(
                ("Bun", mb, n.kind),
                tuple(x - off for x in n.outs),
                tuple(x - off for x in n.ins),
                n.dim,
                n.ph,
            )

        l_nodes = tuple(down_l(n) for n in lb)
        r_nodes = tuple(map(down_l, rb))
        i_l = tuple(x - off for x in i_lb)
        i_r = tuple(x - off for x in i_rb)
        iface = tuple(x - off for x in if_b)
        ln, rn = LNet(l_nodes, i_l, iface), LNet(r_nodes, i_r, iface)
        name = self.fresh("bun")
        lname, rname = f"{name}L", f"{name}R"
        self.decls.append(
            f"/-- The boxed region before `{name}`. -/\ndef {lname} : Net := {ln.lean()}"
        )
        self.decls.append(
            f"/-- The boxed region after `{name}`. -/\ndef {rname} : Net := {rn.lean()}"
        )
        k1 = self.text(inner.k)
        kj = f"({k1}) ^ {jv}"
        lemma = inner.lemma
        both = tuple(y for x in iface for y in (x, x + off))
        p1 = LNet(lb + l_nodes, i_lb + i_l, both)
        p2 = LNet(lb + r_nodes, i_lb + i_r, both)
        p3 = LNet(r_nodes + lb, i_r + i_lb, both)
        p4 = LNet(r_nodes + rb, i_r + i_rb, both)
        t1 = f"⟨{_nodes(lb)} ++ {lname}.nodes, {_list(i_lb)} ++ {lname}.int, {_list(both)}⟩"
        t2 = f"⟨{_nodes(lb)} ++ {rname}.nodes, {_list(i_lb)} ++ {rname}.int, {_list(both)}⟩"
        t3 = f"⟨{rname}.nodes ++ {_nodes(lb)}, {rname}.int ++ {_list(i_lb)}, {_list(both)}⟩"
        t4 = f"⟨{rname}.nodes ++ {_nodes(rb)}, {rname}.int ++ {_list(i_rb)}, {_list(both)}⟩"
        sl, sr = _split_net(ln, mb, isb, off), _split_net(rn, mb, isb, off)
        spc = f"(SpC {mb})"
        self.env = env_j
        try:
            i1 = self.iso_b(env_j, f"(Net.split {spc} {off} {lname})", sl, t1, p1)
            i2 = self.iso_b(env_j, t2, p2, t3, p3)
            i3 = self.iso_b(env_j, t4, p4, f"(Net.split {spc} {off} {rname})", sr)
        finally:
            self.env = saved_env
        plug1 = (
            f"(plug {env_j} {_nodes(lb)} {lname}.nodes {rname}.nodes {_list(i_lb)} "
            f"{lname}.int {rname}.int {_list(both)} {lname}.bnd 1 ({kj}) "
            f"(Equal.toLocal _ _ _ ih) {_chks(3)})"
        )
        plug2 = (
            f"(plug {env_j} {rname}.nodes {_nodes(lb)} {_nodes(rb)} {rname}.int "
            f"{_list(i_lb)} {_list(i_rb)} {_list(both)} {_list(if_b)} 1 ({k1}) "
            f"({lemma}) {_chks(3)})"
        )
        chain_t = (
            f"(Equal.mul_trans {i1} (Equal.mul_trans {plug1} (Equal.mul_trans {i2} "
            f"(Equal.mul_trans {plug2} {i3}))))"
        )
        ok_l = f"(by decide : Spec.splitOK {spc} {off} {lname} = true)"
        ok_r = f"(by decide : Spec.splitOK {spc} {off} {rname} = true)"
        fam = self.fam_text()

        def lst(v: str) -> str:
            return "[" + ", ".join(v if k == mb else names[k] for k in range(len(names))) + "]"

        env_0 = self.env_text([("0" if k == mb else names[k]) for k in range(len(names))])
        z = (
            f"(Equal.rescale (isoB {env_0} 1 (Net.zero {spc} {lname}) "
            f"(Net.zero {spc} {rname}) [] [] (by chk)) one_ne_zero (by simp))"
        )
        binders = _join(
            self.sp_binders, *(f"({names[k]} : ℕ)" for k in oth), f"({jv} : ℕ)", self.sp_hyps
        )
        self.decls.append(
            f"/-- The boxed rewrite at every multiplicity. -/\n"
            f"theorem {_join(name, binders)} :\n"
            f"    Equal {env_j} 1 {lname} ({kj}) {rname} := by\n"
            f"  induction {jv} with\n"
            f"  | zero => exact Equal.of_zero_at {fam} {mb} {off} {lst('0')} (by simp) {lname} "
            f"{rname} {ok_l} {ok_r} (by decide) {z}\n"
            f"  | succ {jv} ih => exact Equal.of_split_at {fam} {mb} {off} {jv} {lst(jv)} "
            f"(by simp) "
            f"{lname} {rname} {ok_l} {ok_r} {self.bund_pos(vals_j)} "
            f"(Equal.rescale {chain_t} (by norm_num) (by ring))"
        )
        cur = vals[mb]
        k_model = _Named(f"({k1}) ^ {_paren(cur)}", self.flat(inner.k) ** syms[mb])
        applied = _join(
            name, self.sp_args, *(_paren(vals[k]) for k in oth), _paren(cur), self.sp_hyp_args
        )
        return _Spec(
            tuple(inner.l_ids),
            [(nid, down_l(n)) for nid, n in inner.r],
            f"Equal.toLocal _ _ _ ({applied})",
            k_model,
            i_l,
            i_r,
            l_nodes,
        )

    def register(self, d: Diagram) -> None:
        """Every dimension, phase and scalar symbol of ``d``."""
        for node in d.nodes.values():
            for port in (*node.inputs, *node.outputs):
                self.dim(port.dim)
            if node.phase is not None:
                self.phase(node.phase)
        self.scalar(d.scalar)

    def collect(self, t: object) -> None:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(t, Theorem)
        self.register(t.lhs)
        self.register(t.rhs)
        for p in t.inference.premises:
            self.collect(p)
        for key in ("peeled", "other_peeled", "own_k", "other_k", "replaced", "exposed"):
            d = t.inference.data.get(key)
            if isinstance(d, Diagram):
                self.register(d)
        step = t.inference.data.get("step")
        if isinstance(step, RewriteStep):
            self.scalar(step.scalar_introduced)

    def set_params(self) -> None:
        params = [
            (n, ty)
            for raw, (n, ty) in self.sym.params.items()
            if raw != "k_lean_index" and raw not in self.mults
        ]
        nats = [n for n, ty in params if ty == "ℕ"]
        self.binders = " ".join(f"({n} : {ty})" for n, ty in params)
        self.args = " ".join(n for n, _ in params)
        self.hyps = " ".join(f"(h_{n} : 0 < {n})" for n in nats)
        self.hyp_args = " ".join(f"h_{n}" for n in nats)
        self.env = f"(E {self.args})" if params else "E"

    def render(self, theorem: object) -> str:
        from archytaszx.semantics.kernel import Theorem

        assert isinstance(theorem, Theorem)
        return self.source(theorem.assumptions)

    def source(self, assumptions: Sequence[object] = (), lead: str = "") -> str:
        """The file: header with ``lead`` and notes, tables and declarations."""
        params = [
            (n, ty)
            for raw, (n, ty) in self.sym.params.items()
            if raw != "k_lean_index" and raw not in self.mults
        ]
        binders = " ".join(f"({n} : {ty})" for n, ty in params)
        args = " ".join(n for n, _ in params)
        dims = "\n".join(f"  | {i} => {text}" for i, text in enumerate(self.dims))
        phases = "\n".join(f"  | {i} => {text}" for i, text in enumerate(self.phases))
        labs = _pairs(sorted(self.labs.items()))
        coef = f", coef := Ψ {args}".rstrip() if self.weight_tab else ""
        tables = [
            f"/-- Dimension values. -/\ndef Δ {binders} : ℕ → ℕ\n{dims}\n  | _ => 1",
            (
                f"/-- Phase vectors in turns. -/\nnoncomputable def Φ {binders} : ℕ → ℕ → ℝ\n"
                f"{phases}\n  | _ => fun _ => 0"
            ),
            f"/-- Each label's dimension index. -/\ndef labs : List (ℕ × ℕ) := {labs}",
            f"/-- The environment. -/\nnoncomputable def E {binders} : Env := "
            f"{{ dimv := Δ {args}, phase := Φ {args}, labs := labs{coef} }}".replace("  ", " "),
        ]
        if self.weight_tab:
            if self.family:
                raise NetUnsupported("a family with complex weights")
            weights = "\n".join(
                f"  | {i} => fun k => {t}" for i, t in sorted(self.weight_tab.items())
            )
            tables.insert(
                2,
                f"/-- Weight vectors. -/\nnoncomputable def Ψ {binders} : ℕ → ℕ → ℂ\n"
                f"{weights}\n  | _ => fun _ => 1",
            )
        if self.family:
            nats = [n for n, ty in params if ty == "ℕ"]
            hyps = " ".join(f"(h_{n} : 0 < {n})" for n in nats)
            tables.insert(
                1,
                (
                    f"/-- Every dimension value is positive. -/\n"
                    f"theorem Δpos {_join(binders, hyps)} :"
                    f" ∀ i, 0 < Δ {args} i := by\n  intro i\n  unfold Δ\n  split <;> "
                    f"first | omega | "
                    "positivity | (apply pow_pos; omega) | norm_num"
                ),
            )
            bund = ", ".join(f"({i}, {b}, {m})" for i, (b, m) in sorted(self.bund_of.items()))
            names = ", ".join(
                f"{k} = {self.mult_names[i]}"
                for k, i in sorted(self.mult_ix.items(), key=lambda kv: kv[1])
            )
            tables += [
                (
                    f"/-- Each bundle dimension index with its base index and multiplicity. -/\n"
                    f"def bundL : List (ℕ × ℕ × ℕ) := [{bund}]"
                ),
                f"/-- The family, its multiplicities in order: {names}. -/\n"
                f"noncomputable def F {binders} : Fam := "
                f"{{ dims := Δ {args}, bund := bundL, phase := Φ {args}, labs := labs }}".replace(
                    "  ", " "
                ),
                f"/-- Every dimension value of the family is positive. -/\ntheorem Fpos "
                f"{_join(binders, hyps)} : ∀ i, 0 < (F {args}).dims i := fun i => Δpos "
                f"{_join(args, ' '.join('h_' + n for n in nats))} i".replace("(F )", "F"),
                (
                    "/-- The family's structure alone. -/\nnoncomputable def FC : Fam := "
                    "{ dims := fun _ => 1, bund := bundL, phase := fun _ _ => 0, labs := labs }"
                ),
                (
                    "/-- The family's structure along multiplicity `m`. -/\n"
                    "noncomputable def SpC (m : ℕ) : Spec := ⟨FC, m, ksOf []⟩"
                ),
            ]
        notes = [f"  assumes {a}" for a in assumptions]
        if self.alias:
            notes.append(
                "  dimensions assumed equal are one dimension here: "
                + ", ".join(f"{k} = {self.canon(v)}" for k, v in sorted(self.alias.items()))
            )
        if getattr(self, "constraints", None):
            notes.append(
                "  assumes the dimension constraints "
                + ", ".join(self.constraints)
                + " (hypotheses of the theorem)"
            )
        for sym, expr in self.sc.opaques.values():
            notes.append(f"  {self.sym.name(sym, 'ℂ')} stands for the scalar {expr}")
        if self.unproved:
            notes.append("  unproved steps: " + ", ".join(self.unproved))
        if self.scalar_gap:
            notes.append(f"  the diagrams' own scalars are not compared: {self.scalar_gap}")
        header = [
            "/-",
            "  ArchytasZX proof export over the tensor-network semantics of the ArchytasZX Lean",
            "  library: every diagram is a Net, every rule step an instance of a lemma proved",
            "  there. `#print axioms` lists any step stated without proof.",
            *([lead] if lead else []),
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
            *_macros(
                self.unfold,
                f"exact Fam.env_dimv_pos _ _ _ (Fpos {_join(self.sp_args, self.sp_hyp_args)} _)"
                if self.family
                else "",
            ),
        ]
        return "\n".join(
            [*header, "", "\n\n".join([*tables, *self.decls]), "", "end ArchytasZX.Export", ""]
        )


def _norm_tail(ls: Sequence[int], cs: Sequence[Chain]) -> tuple[tuple[int, ...], list[Chain]]:
    """Python mirror of ``normTail``."""
    ps = list(zip(_tail(ls, cs), cs, strict=False))
    head = tuple(ls[: len(ls) - len(cs)])
    return (
        head + tuple(x for x, c in ps if not c) + tuple(x for x, c in ps if c),
        [c for _, c in ps if c],
    )


def _norm(n: LNode) -> LNode:
    """Python mirror of ``Node.norm``."""
    h = _head(n.kind)
    if h == "ZB":
        ins, cs = _norm_tail(n.ins, n.kind[1])
        return replace(n, kind=_zk(cs), ins=ins)
    if h == "XB":
        outs, co = _norm_tail(n.outs, n.kind[1])
        ins, ci = _norm_tail(n.ins, n.kind[2])
        return replace(n, kind=_xk(co, ci), outs=outs, ins=ins)
    if h == "WB":
        outs, cs = _norm_tail(n.outs, n.kind[1])
        return replace(n, kind=_wk(cs), outs=outs)
    return n


def _segments(n: LNode) -> list[tuple[tuple[int, ...], bool]]:
    """``n``'s legs in the groups the Lean ``nodeEquivB`` lets reorder (True) or not."""
    n = _norm(n)
    o, i = n.outs, n.ins
    k = n.kind
    h = _head(k)

    def groups(ls: Sequence[int], cs: Sequence[Chain]) -> list[tuple[tuple[int, ...], bool]]:
        tl = _tail(ls, cs)
        return [
            (tuple(x for x, c2 in zip(tl, cs, strict=True) if c2 == c), True)
            for c in sorted(set(cs))
        ]

    if h == "Bun":
        if k[2] == "Z":
            return [(o + i, True)]
        if k[2] == "X":
            return [(o, True), (i, True)]
        return [(o, False), (i, False)]
    if h in ("Z", "ZW"):
        return [(o + i, True)]
    if h == "X":
        return [(o, True), (i, True)]
    if h == "W":
        return [(o, True), (i, False)]
    if h == "ZB":
        cs = k[1]
        return [(o + i[: len(i) - len(cs)], True), *groups(i, cs)]
    if h == "XB":
        co, ci = k[1], k[2]
        return [
            (o[: len(o) - len(co)], True),
            *groups(o, co),
            (i[: len(i) - len(ci)], True),
            *groups(i, ci),
        ]
    if h == "WB":
        cs = k[1]
        return [(o[: len(o) - len(cs)], True), *groups(o, cs), (i, False)]
    return [(o, False), (i, False)]


def _match(
    a: LNet, b: LNet, dims_a: Callable[[int], int], dims_b: Callable[[int], int], fix_bnd: bool
) -> dict[int, int] | None:
    """A label bijection carrying ``a`` onto ``b`` node for node, boundary to boundary (each
    boundary label to itself when ``fix_bnd``), or None."""
    if len(a.nodes) != len(b.nodes) or len(a.bnd) != len(b.bnd) or len(a.summed) != len(b.summed):
        return None
    bnd_a, bnd_b = set(a.bnd), set(b.bnd)

    def sig(n: LNode) -> tuple[object, ...]:
        return (
            _canon(_norm(n).kind),
            n.dim,
            n.ph,
            tuple((len(seg), sym) for seg, sym in _segments(n)),
        )

    order = sorted(
        range(len(a.nodes)), key=lambda j: sum(1 for m in b.nodes if sig(m) == sig(a.nodes[j]))
    )
    used: set[int] = set()
    fwd: dict[int, int] = {}
    back: dict[int, int] = {}
    if fix_bnd:
        for x in a.bnd:
            fwd[x] = x
            back[x] = x

    def ok(x: int, y: int) -> bool:
        if x in fwd:
            return fwd[x] == y
        if y in back:
            return False
        return (x in bnd_a) == (y in bnd_b) and dims_a(x) == dims_b(y)

    def bind(pairs: list[tuple[int, int]]) -> list[int] | None:
        added: list[int] = []
        for x, y in pairs:
            if not ok(x, y):
                for z in added:
                    del back[fwd.pop(z)]
                return None
            if x not in fwd:
                fwd[x], back[y] = y, x
                added.append(x)
        return added

    def unbind(added: list[int]) -> None:
        for z in added:
            del back[fwd.pop(z)]

    def legs(xs: tuple[int, ...], ys: tuple[int, ...], sym: bool):  # type: ignore[no-untyped-def]
        if len(xs) != len(ys):
            return
        if not sym:
            added = bind(list(zip(xs, ys, strict=True)))
            if added is not None:
                yield
                unbind(added)
            return
        if not xs:
            yield
            return
        x, rest = xs[0], xs[1:]
        tried: set[int] = set()
        for j, y in enumerate(ys):
            if y in tried:
                continue
            tried.add(y)
            added = bind([(x, y)])
            if added is None:
                continue
            yield from legs(rest, ys[:j] + ys[j + 1 :], True)
            unbind(added)

    def node_align(n: LNode, m: LNode):  # type: ignore[no-untyped-def]
        segs_n, segs_m = _segments(n), _segments(m)
        if len(segs_n) != len(segs_m):
            return

        def go(t: int):  # type: ignore[no-untyped-def]
            if t == len(segs_n):
                yield
                return
            (xs, sym), (ys, _) = segs_n[t], segs_m[t]
            for _ in legs(xs, ys, sym):
                yield from go(t + 1)

        yield from go(0)

    def search(t: int) -> bool:
        if t == len(order):
            return True
        n = a.nodes[order[t]]
        for j, m in enumerate(b.nodes):
            if j in used or sig(m) != sig(n):
                continue
            used.add(j)
            for _added in node_align(n, m):
                if search(t + 1):
                    return True
            used.discard(j)
        return False

    if not search(0):
        return None
    return dict(fwd)


def _iface(l_nodes: Sequence[LNode], i_l: Sequence[int]) -> tuple[int, ...]:
    """The region's labels that are not summed inside it, in order of appearance."""
    out: list[int] = []
    for n in l_nodes:
        for x in n.labels():
            if x not in i_l and x not in out:
                out.append(x)
    return tuple(out)


Chain = tuple[int, ...]


def _head(kind: Any) -> str:
    """A kind's constructor."""
    return kind if isinstance(kind, str) else kind[0]


def _klean(kind: Any) -> str:
    """A kind as a Lean term."""
    if isinstance(kind, str):
        return f".{kind}"

    def cs(x: Sequence[Chain]) -> str:
        return "[" + ", ".join(_list(c) for c in x) + "]"

    h = kind[0]
    if h in ("ZB", "WB"):
        return f".{h} {cs(kind[1])}"
    if h == "XB":
        return f".XB {cs(kind[1])} {cs(kind[2])}"
    base = _klean(kind[2])
    return f".Bun {kind[1]} {base if isinstance(kind[2], str) else '(' + base + ')'}"


def _canon(kind: Any) -> Any:
    """A kind with its bundled chains in a fixed order."""
    if isinstance(kind, str):
        return kind
    h = kind[0]
    if h in ("ZB", "WB"):
        return (h, tuple(sorted(kind[1])))
    if h == "XB":
        return (h, tuple(sorted(kind[1])), tuple(sorted(kind[2])))
    return (h, kind[1], _canon(kind[2]))


def _outcs(kind: Any, n: int) -> list[Chain]:
    """Python mirror of ``Kind.outCs``."""
    h = _head(kind)
    if h == "XB":
        return [()] * (n - len(kind[1])) + list(kind[1])
    if h == "WB":
        return [()] * (n - len(kind[1])) + list(kind[1])
    if h == "Bun":
        return [(kind[1],) + c for c in _outcs(kind[2], n)]
    return [()] * n


def _incs(kind: Any, n: int) -> list[Chain]:
    """Python mirror of ``Kind.inCs``."""
    h = _head(kind)
    if h == "ZB":
        return [()] * (n - len(kind[1])) + list(kind[1])
    if h == "XB":
        return [()] * (n - len(kind[2])) + list(kind[2])
    if h == "Bun":
        return [(kind[1],) + c for c in _incs(kind[2], n)]
    return [()] * n


def _kms(kind: Any) -> set[int]:
    """Python mirror of ``Kind.ms``."""
    h = _head(kind)
    if h in ("ZB", "WB"):
        return {x for c in kind[1] for x in c}
    if h == "XB":
        return {x for c in (*kind[1], *kind[2]) for x in c}
    if h == "Bun":
        return {kind[1]} | _kms(kind[2])
    return set()


def _copyable(kind: Any) -> bool:
    """Python mirror of ``copyKind``."""
    h = _head(kind)
    if h in ("B", "S"):
        return False
    return _copyable(kind[2]) if h == "Bun" else True


def _zk(cs: Sequence[Chain]) -> Any:
    return ("ZB", tuple(cs)) if cs else "Z"


def _xk(co: Sequence[Chain], ci: Sequence[Chain]) -> Any:
    return ("XB", tuple(co), tuple(ci)) if co or ci else "X"


def _wk(cs: Sequence[Chain]) -> Any:
    return ("WB", tuple(cs)) if cs else "W"


def _tower(ms: Sequence[int]) -> Any:
    """Python mirror of ``Kind.tower``."""
    return "I" if not ms else ("Bun", ms[0], _tower(ms[1:]))


def _tower_ms(kind: Any) -> tuple[int, ...] | None:
    """The multiplicities of a tower of copied bare wires."""
    if kind == "I":
        return ()
    if _head(kind) == "Bun":
        rest = _tower_ms(kind[2])
        return None if rest is None else (kind[1],) + rest
    return None


def _peel(
    m: int, off: int, tl: Sequence[int], cs: Sequence[Chain]
) -> tuple[list[int], list[Chain]]:
    ps = [(b, c) for b, c in zip(tl, cs, strict=False) if c[:1] == (m,)]
    return [b + off for b, _ in ps], [c[1:] for _, c in ps]


def _tail(ls: Sequence[int], cs: Sequence[Chain]) -> tuple[int, ...]:
    return tuple(ls[len(ls) - len(cs) :])


def _split_node(n: LNode, m: int, off: int) -> list[LNode]:
    """Python mirror of ``Node.split``."""
    h = _head(n.kind)
    if h == "ZB":
        cs = n.kind[1]
        pl, pc = _peel(m, off, _tail(n.ins, cs), cs)
        return [replace(n, kind=("ZB", tuple(cs) + tuple(pc)), ins=n.ins + tuple(pl))]
    if h == "XB":
        co, ci = n.kind[1], n.kind[2]
        ol, oc = _peel(m, off, _tail(n.outs, co), co)
        il, ic = _peel(m, off, _tail(n.ins, ci), ci)
        return [
            replace(
                n,
                kind=("XB", tuple(co) + tuple(oc), tuple(ci) + tuple(ic)),
                outs=n.outs + tuple(ol),
                ins=n.ins + tuple(il),
            )
        ]
    if h == "WB":
        cs = n.kind[1]
        pl, pc = _peel(m, off, _tail(n.outs, cs), cs)
        return [replace(n, kind=("WB", tuple(cs) + tuple(pc)), outs=n.outs + tuple(pl))]
    if h == "Bun" and n.kind[1] == m:
        return [
            n,
            LNode(
                n.kind[2],
                tuple(x + off for x in n.outs),
                tuple(x + off for x in n.ins),
                n.dim,
                n.ph,
            ),
        ]
    return [n]


def _split_net(net: LNet, m: int, isb: Callable[[int], bool], off: int = H_OFFSET) -> LNet:
    """Python mirror of ``Net.split`` along ``m``."""
    nodes = [x for n in net.nodes for x in _split_node(n, m, off)]

    def expand(ls: Sequence[int]) -> tuple[int, ...]:
        return tuple(y for x in ls for y in ((x, x + off) if isb(x) else (x,)))

    return LNet(tuple(nodes), expand(net.summed), expand(net.bnd))


def _drop(m: int, ls: Sequence[int], cs: Sequence[Chain]) -> tuple[tuple[int, ...], list[Chain]]:
    ps = [(b, c) for b, c in zip(_tail(ls, cs), cs, strict=False) if c[:1] != (m,)]
    return tuple(ls[: len(ls) - len(cs)]) + tuple(b for b, _ in ps), [c for _, c in ps]


def _zero_net(net: LNet, m: int, isb: Callable[[int], bool]) -> LNet:
    """Python mirror of ``Net.zero`` along ``m``."""
    nodes: list[LNode] = []
    for n in net.nodes:
        h = _head(n.kind)
        if h == "ZB":
            ins, cs = _drop(m, n.ins, n.kind[1])
            nodes.append(replace(n, kind=_zk(cs), ins=ins))
        elif h == "XB":
            outs, co = _drop(m, n.outs, n.kind[1])
            ins, ci = _drop(m, n.ins, n.kind[2])
            nodes.append(replace(n, kind=_xk(co, ci), outs=outs, ins=ins))
        elif h == "WB":
            outs, cs = _drop(m, n.outs, n.kind[1])
            nodes.append(replace(n, kind=_wk(cs), outs=outs))
        elif not (h == "Bun" and n.kind[1] == m):
            nodes.append(n)
    return LNet(
        tuple(nodes),
        tuple(x for x in net.summed if not isb(x)),
        tuple(x for x in net.bnd if not isb(x)),
    )


def _rename_net(net: LNet, pairs: Sequence[tuple[int, int]]) -> LNet:
    """Python mirror of ``Net.rename``."""
    f = _ren(pairs)
    return LNet(
        tuple(n.rename(f) for n in net.nodes), tuple(map(f, net.summed)), tuple(map(f, net.bnd))
    )


def _bundle_node(n: LNode, tails: dict[tuple[str, int], Chain]) -> LNode:
    """``n`` with its bundled legs (side and index to chain) moved last in a fixed order."""
    if not tails:
        return n
    if n.kind == "ZW":
        raise NetUnsupported("a weighted spider with bundled legs")
    outs = [x for j, x in enumerate(n.outs) if ("o", j) not in tails]
    ins = [x for j, x in enumerate(n.ins) if ("i", j) not in tails]
    to = sorted(((c, j) for (side, j), c in tails.items() if side == "o"))
    ti = sorted(((c, j) for (side, j), c in tails.items() if side == "i"))
    if n.kind == "Z":
        both = sorted([(c, 0, j) for c, j in to] + [(c, 1, j) for c, j in ti])
        tl = [n.outs[j] if s == 0 else n.ins[j] for c, s, j in both]
        return replace(
            n,
            kind=("ZB", tuple(c for c, _, _ in both)),
            outs=tuple(outs),
            ins=tuple(ins) + tuple(tl),
        )
    if n.kind == "X":
        return replace(
            n,
            kind=("XB", tuple(c for c, _ in to), tuple(c for c, _ in ti)),
            outs=tuple(outs) + tuple(n.outs[j] for _, j in to),
            ins=tuple(ins) + tuple(n.ins[j] for _, j in ti),
        )
    if n.kind == "W" and not ti:
        return replace(
            n,
            kind=("WB", tuple(c for c, _ in to)),
            outs=tuple(outs) + tuple(n.outs[j] for _, j in to),
        )
    raise NetUnsupported(f"a bundled leg on a {n.kind} node")


def _equiv_b(n: LNode, m: LNode) -> bool:
    """Python mirror of the Lean ``nodeEquivB``."""
    if (_canon(_norm(n).kind), n.dim, n.ph) != (_canon(_norm(m).kind), m.dim, m.ph):
        return False
    sn, sm = _segments(n), _segments(m)
    if [len(x) for x, _ in sn] != [len(x) for x, _ in sm]:
        return False
    return all(
        sorted(x) == sorted(y) if sym else x == y for (x, sym), (y, _) in zip(sn, sm, strict=True)
    )


_FAMILY_UNFOLD = "F, Fam.env, bundL, List.lookup, ksOf, List.getD"


def _macros(unfold: str, pos_first: str = "") -> list[str]:
    """The export's tactics, unfolding the environment through ``unfold``."""
    pre = f"first | {pos_first} | " if pos_first else ""
    defs = [
        (
            "chk",
            "Decides a structural side condition.",
            f"first | decide | (simp only [{unfold}]; decide)",
        ),
        (
            "pos",
            "A dimension is positive.",
            (
                f"{pre}(simp [{unfold}, Δ] <;> first | decide | omega | positivity | "
                "(apply Nat.mul_pos <;> omega) | (apply pow_pos; omega))"
            ),
        ),
        ("phz", "A phase vector is zero.", f"(intro k _; simp [{unfold}, Φ, ex_zero])"),
        (
            "nzt",
            "A product of nonzero scalars is nonzero.",
            (
                "repeat (first | exact one_ne_zero | apply mul_ne_zero | apply pow_ne_zero | "
                "apply inv_ne_zero | exact Complex.I_ne_zero | exact Complex.exp_ne_zero _ | "
                "(apply rt_ne; first | omega | norm_num | positivity | (apply pow_pos; omega)) | "
                "(apply om_ne_zero; first | omega | norm_num) | "
                "(apply Nat.cast_ne_zero.2; first | omega | positivity | norm_num) | norm_num)"
            ),
        ),
        (
            "phs",
            "A phase vector is the sum of two.",
            f"(intro k _; simp only [{unfold}, Φ]; rw [← ex_add]; apply ex_congr_int; "
            "(try split_ifs) <;> first "
            + " ".join(f"| (refine ⟨{z}, ?_⟩; push_cast; linarith)" for z in (0, -1, 1, -2, 2))
            + ")",
        ),
        (
            "zcz",
            "A coefficient vector is `[k = 0]`.",
            (
                f"(intro k hk; simp only [{unfold}, Δ] at hk; simp only [zc, {unfold}, Φ]; "
                "(try simp only [Ψ]); (try split_ifs) <;> first | (exfalso; omega) | "
                "simp [ex_zero])"
            ),
        ),
    ]
    out: list[str] = []
    for name, doc, body in defs:
        out += [
            "",
            "set_option hygiene false in",
            f"/-- {doc} -/",
            f'macro "{name}" : tactic => `(tactic| {body})',
        ]
    return out


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
    if n.kind in ("Z", "ZW"):
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

    def rt(self, dz: int) -> sp.Expr:
        """``√d`` for the dimension index ``dz``."""
        return Rt(self.ex.dim_exprs[dz])

    def copy_scalar(self, dz: int, n: int) -> tuple[sp.Expr, str, str]:
        """The copy scalar ``√d^{1-n}``, its Lean text, and the proof of ``k √d^n = √d``."""
        r = self.rt(dz)
        k = r if n == 0 else ONE if n == 1 else Inv(r) ** (n - 1)
        sc = self.ex.sc
        try:
            body = sc.prove(k * r**n, r)
        except ScalarUnsupported as exc:
            raise NetUnsupported(str(exc)) from exc
        proof = f"(show {sc.text(k)} * {sc.text(r)} ^ {n} = {sc.text(r)} from ({body}))"
        return k, _paren(sc.text(k)), proof


@dataclass
class _Spec:
    """A rule step's local equation: L's node ids, R's nodes, the lemma and its scalar."""

    l_ids: tuple[NodeId, ...]
    r: Sequence[tuple[NodeId | None, LNode]]
    lemma: str
    k: Any = field(default_factory=lambda: sp.Integer(1))
    i_l: tuple[int, ...] | None = None
    i_r: tuple[int, ...] | None = None
    l: tuple[LNode, ...] | None = None


_RuleBuilder = Callable[[_Ctx], _Spec]

_CHK = "(by chk)"


def _phaseless(c: _Ctx, n: LNode) -> str:
    if n.ph != 0:
        raise NetUnsupported("a phase where the rule needs none")
    return "(by phz)"


def _phase_sum() -> str:
    return "(by phs)"


def _positive(c: _Ctx, dz: int) -> str:
    return "(by pos)"


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
    _need(na.dim == nb.dim == r.dim, "fusion across dims")
    if any(_head(x.kind) in ("ZB", "XB") for x in (na, nb, r)):
        return _fusion_bundled(c, na, nb, rid, r, w)
    if "ZW" in (na.kind, nb.kind, r.kind):
        _need(all(x.kind in _ZC for x in (na, nb, r)), "fusion across kinds")
        lemma = (
            f"zcFusion {c.env} {_zc(na)} {_zc(nb)} {_zc(r)} {r.dim} {w} IFACE "
            f"(by decide) (by decide) (by decide) rfl rfl rfl (by decide) (by decide) "
            f"(by decide) {_CHK} {_coef_hyp(c, r, [na, nb])}"
        )
        return _Spec((m.a_id, m.b_id), [(rid, r)], lemma)
    _need(na.kind == nb.kind == r.kind, "fusion across kinds")
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


def _fusion_bundled(c: _Ctx, na: LNode, nb: LNode, rid: NodeId, r: LNode, w: int) -> _Spec:
    """Fusion of spiders some of whose legs bundle."""
    m = c.match
    ids = {id(na): m.a_id, id(nb): m.b_id}
    hr = _head(r.kind)

    def zparts(n: LNode) -> tuple[Sequence[Chain], tuple[int, ...], tuple[int, ...]]:
        cs = n.kind[1] if _head(n.kind) == "ZB" else ()
        return cs, n.ins[: len(n.ins) - len(cs)], _tail(n.ins, cs)

    if hr in ("Z", "ZB"):
        _need(
            _head(na.kind) in ("Z", "ZB") and _head(nb.kind) in ("Z", "ZB"), "fusion across kinds"
        )
        c1, i1, t1 = zparts(na)
        c2, i2, t2 = zparts(nb)
        _, i3, _ = zparts(r)
        rn = LNode(_zk(tuple(c1) + tuple(c2)), r.outs, i3 + t1 + t2, r.dim, r.ph)

        def fmtz(x: Any) -> str:
            return "[" + ", ".join(_list(y) for y in x) + "]"

        lemma = (
            f"fusionZG {c.env} {fmtz(c1)} {fmtz(c2)} {na.ph} {nb.ph} {r.ph} {r.dim} {w} "
            f"{_list(na.outs)} {_list(i1)} {_list(t1)} {_list(nb.outs)} {_list(i2)} {_list(t2)} "
            f"{_list(r.outs)} {_list(i3)} IFACE {_chks(9)} {_phase_sum()}"
        )
        return _Spec((m.a_id, m.b_id), [(rid, r)], _as_r(lemma, [rn], [r]))
    _need(_head(na.kind) in ("X", "XB") and _head(nb.kind) in ("X", "XB"), "fusion across kinds")

    def xparts(
        n: LNode,
    ) -> tuple[
        Sequence[Chain],
        Sequence[Chain],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        co, ci = (n.kind[1], n.kind[2]) if _head(n.kind) == "XB" else ((), ())
        return (
            co,
            ci,
            n.outs[: len(n.outs) - len(co)],
            _tail(n.outs, co),
            n.ins[: len(n.ins) - len(ci)],
            _tail(n.ins, ci),
        )

    pa, pb = xparts(na), xparts(nb)
    if w in pa[2]:
        first, second, f, g = na, nb, pa, pb
    else:
        first, second, f, g = nb, na, pb, pa
    _need(w in f[2] and w in g[4], "an X fusion along one output")
    pr = xparts(r)
    rn = LNode(
        _xk(tuple(f[0]) + tuple(g[0]), tuple(f[1]) + tuple(g[1])),
        pr[2] + f[3] + g[3],
        pr[4] + f[5] + g[5],
        r.dim,
        r.ph,
    )

    def fmtx(x: Any) -> str:
        return "[" + ", ".join(_list(y) for y in x) + "]"

    lemma = (
        f"fusionXG {c.env} {fmtx(f[0])} {fmtx(f[1])} {fmtx(g[0])} {fmtx(g[1])} "
        f"{first.ph} {second.ph} "
        f"{r.ph} {r.dim} {w} {_list(f[2])} {_list(f[3])} {_list(f[4])} {_list(f[5])} "
        f"{_list(g[2])} {_list(g[3])} {_list(g[4])} {_list(g[5])} {_list(pr[2])} {_list(pr[4])} "
        f"IFACE {_chks(4)} {_positive(c, r.dim)} {_chks(10)} {_phase_sum()}"
    )
    return _Spec((ids[id(first)], ids[id(second)]), [(rid, r)], _as_r(lemma, [rn], [r]))


def _as_r(lemma: str, made: Sequence[LNode], want: Sequence[LNode]) -> str:
    """``lemma`` with its right side ``made`` restated as the equivalent nodes ``want``."""
    if list(made) == list(want):
        return lemma
    _need(all(_equiv_b(x, y) for x, y in zip(made, want, strict=True)), "rewritten nodes differ")
    return f"LocalEq.equivR ({lemma}) rfl (by decide)"


def _as_l(lemma: str, made: Sequence[LNode], want: Sequence[LNode]) -> str:
    """``lemma`` with its left side ``made`` restated as the equivalent nodes ``want``."""
    if list(made) == list(want):
        return lemma
    _need(all(_equiv_b(x, y) for x, y in zip(made, want, strict=True)), "matched nodes differ")
    return f"LocalEq.equivL ({lemma}) rfl (by decide)"


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
    return _Spec((m.state_id, m.effect_id), [], lemma, c.rt(dz))


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
    k, kt, hk = c.copy_scalar(dz, n)
    name = "stateCopySwapped" if m.swapped else "stateCopy"
    lemma = (
        f"{name} {c.env} {st.ph} {sp_.ph} {cph} {dz} {w} {_list(sp_.outs)} IFACE {kt} "
        f"{_positive(c, dz)} {_chks(3)} {_phaseless(c, st)} {_phaseless(c, sp_)} "
        f"(by phz) {hk}"
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
    return _Spec(ids, [(rid, r)], lemma, ONE, (w1, w2, w3))


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
        c.rt(dz),
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
    x = NC(c.ex.dim_exprs[dz])
    k = Inv(x)
    sc = c.ex.sc
    try:
        body = sc.prove(k * x, ONE)
    except ScalarUnsupported as exc:
        raise NetUnsupported(str(exc)) from exc
    hk = f"(show {sc.text(k)} * {sc.text(x)} = 1 from ({body}))"
    kt = _paren(sc.text(k))
    tail = f"{_positive(c, dz)} {_chks(6)} {hk}"
    if not m.swapped:
        lemma = (
            f"hopf {c.env} {up.ph} {lo.ph} {dz} {u} {v1} {v2} {v3} {_list(up.outs)} "
            f"{_list(up.ins)} {_list(lo.outs)} {_list(lo.ins)} {_list(nu.outs)} {_list(nu.ins)} "
            f"{_list(nl.outs)} {_list(nl.ins)} IFACE {kt} {tail}"
        )
    else:
        lemma = (
            f"hopfSwapped {c.env} {up.ph} {lo.ph} {dz} {u} {v1} {v2} {v3} {_list(up.outs)} "
            f"{_list(up.ins)} {_list(lo.outs)} {_list(lo.ins)} {_list(nu.outs)} "
            f"{_list(nu.ins)} {_list(nl.outs)} {_list(nl.ins)} IFACE {kt} {tail}"
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
        k, kt, hk = c.copy_scalar(dz, n)
        lemma = (
            f"wZeroCopy {c.env} {st.ph} {cph} {dz} {w} {_list(tg.outs)} IFACE {kt} "
            f"{_positive(c, dz)} {_chks(3)} {_phaseless(c, st)} "
            f"(by phz) {hk}"
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
    return _Spec((m.triangle_id, m.effect_id), [(rid, r)], lemma, c.rt(dz))


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
        return _Spec((m.w_id, m.effect_id), rs, lemma, c.rt(dz))
    (_, tri), (_, rest) = rs
    v = tri.outs[0]
    _need(tri.ins == (i,) and rest.ins == (v,), "W into Z effect replacement")
    lemma = (
        f"wZEffect {c.env} {ef.ph} {dz} {w} {i} {v} {_list(wn.outs)} {_list(rest.outs)} IFACE "
        f"{_positive(c, dz)} {_chks(10)} {_phaseless(c, ef)}"
    )
    return _Spec((m.w_id, m.effect_id), rs, lemma, ONE, None, (v,))


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
    return _Spec((m.connective_id, ia, ib), [(rid, r)], lemma, ONE, (a_, b_))


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
        return _Spec((m.first_id, m.second_id), [(None, delta)], lemma, ONE, (a_, b_))
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
    return _Spec((m.first_id, m.second_id), [(None, d1), (None, d2)], lemma, ONE, (k,))


def _wz(c: _Ctx) -> _Spec:
    m = c.match
    z, wn = c.node(m.z_id), c.node(m.w_id)
    w = c.wire(m.wire)
    _need(z.kind == "Z" and z.outs == (w,) and wn.kind == "W" and wn.ins == (w,), "shape")
    zi, wo = z.ins, wn.outs
    _need(len(zi) >= 1, "a Z with no inputs")
    rs = [(nid, c.new_nodes[nid]) for nid in c.step.new_node_ids]
    ws, zs = [n for _, n in rs[: len(zi)]], [n for _, n in rs[len(zi) :]]
    _need(len(zs) == len(wo), "one Z per W output")
    rows = [n.outs for n in ws]
    cols = [n.ins for n in zs]
    _need(all(n.ins == (i,) for n, i in zip(ws, zi, strict=True)), "W inputs")
    _need(all(n.outs == (o,) for n, o in zip(zs, wo, strict=True)), "Z outputs")
    qs = {n.ph for n in zs}
    _need(len(qs) == 1, "Z phases differ")
    q = qs.pop()
    pin = [(e, wo[b]) for row in rows for b, e in enumerate(row)]
    flat = [e for row in rows for e in row]
    dz = z.dim

    def lists(xs: Sequence[Sequence[int]]) -> str:
        return "[" + ", ".join(_list(x) for x in xs) + "]"

    lemma = (
        f"wzBialgebra {c.env} {z.ph} {q} {dz} {w} {_list(zi)} {_list(wo)} IFACE {lists(rows)} "
        f"{lists(cols)} {_pairs(pin)} {_positive(c, dz)} {_chks(14)} {_phaseless(c, z)} "
        f"{_phaseless(c, zs[0])}"
    )
    return _Spec((m.z_id, m.w_id), rs, lemma, ONE, None, tuple(flat))


def _unfuse(c: _Ctx) -> _Spec:
    m = c.match
    n = c.node(m.node_id)
    rs = [(nid, c.new_nodes[nid]) for nid in c.step.new_node_ids]
    (main_id, main), (side_id, side) = rs
    b = c.a.port[m.boxed]
    dz = n.dim
    h = _head(n.kind)

    def cs(x: Any) -> str:
        return "[" + ", ".join(_list(y) for y in x) + "]"

    def move_last(
        ls: tuple[int, ...], chains: Sequence[Chain]
    ) -> tuple[tuple[int, ...], tuple[int, ...], list[Chain], Chain]:
        hd = ls[: len(ls) - len(chains)]
        tl = list(zip(_tail(ls, chains), chains, strict=True))
        j = next(j for j, (x, _) in enumerate(tl) if x == b)
        (_bb, cb) = tl.pop(j)
        return hd, tuple(x for x, _ in tl), [c2 for _, c2 in tl], cb

    if h == "ZB":
        _need(b in _tail(n.ins, n.kind[1]), "unfusion of an unbundled leg")
        hd, tl, rest, cb = move_last(n.ins, n.kind[1])
        legs = side.outs + side.ins
        _need(
            _head(side.kind) == "ZB" and len(legs) == 2 and b in _tail(side.ins, side.kind[1]),
            "unfusion side spider",
        )
        y = next(x for x in legs if x != b)
        lhs = LNode(("ZB", tuple(rest) + (cb,)), n.outs, hd + tl + (b,), dz, n.ph)
        mh = main.ins[: len(main.ins) - len(rest)] if _head(main.kind) == "ZB" else main.ins
        rmain = LNode(_zk(rest), main.outs, tuple(mh) + tl, dz, main.ph)
        rside = LNode(("ZB", (cb,)), (), (y, b), dz, side.ph)
        lemma = (
            f"unfuseZB {c.env} {cs(rest)} {_list(cb)} {n.ph} {side.ph} {dz} {y} {b} "
            f"{_list(n.outs)} {_list(hd)} {_list(tl)} {_list(main.outs)} {_list(mh)} IFACE "
            f"{_chks(4)} {_phaseless(c, side)}"
        )
    elif h == "XB" and b in _tail(n.outs, n.kind[1]):
        hd, tl, rest, cb = move_last(n.outs, n.kind[1])
        ci = n.kind[2]
        _need(side.outs == (b,) and len(side.ins) == 1, "unfusion side")
        y = side.ins[0]
        lhs = LNode(("XB", tuple(rest) + (cb,), ci), hd + tl + (b,), n.ins, dz, n.ph)
        mo = main.outs[: len(main.outs) - len(rest)] if _head(main.kind) == "XB" else main.outs
        rmain = LNode(_xk(rest, ci), tuple(mo) + tl, n.ins, dz, main.ph)
        rside = LNode(("XB", (cb,), ()), (b,), (y,), dz, side.ph)
        lemma = (
            f"unfuseXB {c.env} {cs(rest)} {cs(ci)} {_list(cb)} {n.ph} {side.ph} {dz} {y} {b} "
            f"{_list(hd)} {_list(tl)} {_list(n.ins)} {_list(mo)} IFACE {_chks(1)} "
            f"{_positive(c, dz)} {_chks(3)} {_phaseless(c, side)}"
        )
    elif h == "XB" and b in _tail(n.ins, n.kind[2]):
        co = n.kind[1]
        hd, tl, rest, cb = move_last(n.ins, n.kind[2])
        _need(side.ins == (b,) and len(side.outs) == 1, "unfusion side")
        y = side.outs[0]
        lhs = LNode(("XB", co, tuple(rest) + (cb,)), n.outs, hd + tl + (b,), dz, n.ph)
        mi = main.ins[: len(main.ins) - len(rest)] if _head(main.kind) == "XB" else main.ins
        rmain = LNode(_xk(co, rest), n.outs, tuple(mi) + tl, dz, main.ph)
        rside = LNode(("XB", (), (cb,)), (y,), (b,), dz, side.ph)
        lemma = (
            f"unfuseXBi {c.env} {cs(co)} {cs(rest)} {_list(cb)} {n.ph} {side.ph} {dz} {y} {b} "
            f"{_list(n.outs)} {_list(hd)} {_list(tl)} {_list(mi)} IFACE {_chks(1)} "
            f"{_positive(c, dz)} {_chks(3)} {_phaseless(c, side)}"
        )
    else:
        raise NetUnsupported(f"unfusion of a {h} node")
    _need(main.ph == n.ph and _head(main.kind) in ("Z", "X", "ZB", "XB"), "unfusion main spider")
    lemma = _as_l(_as_r(lemma, [rmain, rside], [main, side]), [lhs], [n])
    return _Spec((m.node_id,), [(main_id, main), (side_id, side)], lemma, ONE, (), (y,))


def _w_perm(c: _Ctx) -> _Spec:
    m = c.match
    nodes = dict(m.nodes)
    n = c.node(nodes["w"])
    ((rid, r),) = c.new_nodes.items()
    _need(n.kind == "W" and r.kind == "W" and r.ins == n.ins and len(n.ins) == 1, "W shape")
    _need(sorted(n.outs) == sorted(r.outs), "W outputs permuted")
    lemma = (
        f"wPerm {c.env} {n.ph} {r.ph} {n.dim} {n.ins[0]} {_list(n.outs)} {_list(r.outs)} IFACE "
        "(by decide)"
    )
    return _Spec((nodes["w"],), [(rid, r)], lemma)


_ZC = ("Z", "ZW")


def _zc(n: LNode) -> str:
    """A node literal as one Lean argument."""
    return f"({n.lean()} : Node)"


def _zc_terms(c: _Ctx, n: LNode, k: int) -> tuple[sp.Expr, str]:
    """The model of ``n``'s coefficient at ``k`` and the rewrite of its phase to that model."""
    if n.kind == "ZW":
        body = c.ex.weight_tab[n.ph]
        return (ONE if k == 0 else c.ex.coef_models[body][k]), ""
    entries, rule = c.ex.phase_specs[n.ph]
    _need(rule is None, "phases written entry by entry")
    turns = sp.nsimplify(entries.get(k, sp.Integer(0))) if k else sp.Integer(0)
    _need(bool(turns.is_Rational), "a rational phase")
    m, z = int(turns.q), int(sp.floor(turns))
    p = int(turns.p) - z * m
    rw = (
        f"rw [show ex ({c.env}.phase {n.ph} {k}) = om {m} ^ {p} from ex_of {m} {p} ({z}) "
        f"(by norm_num) (by norm_num [{c.ex.unfold}, Φ])]"
    )
    return Om(m) ** p, rw


def _coef_rewrites(c: _Ctx, nodes: Sequence[LNode], ks: Sequence[int]) -> list[str]:
    """The phase rewrites of ``nodes`` at the indices ``ks``, each once."""
    return _pair_rewrites(c, [(n, k) for k in ks for n in nodes])


def _pair_rewrites(c: _Ctx, pairs: Sequence[tuple[LNode, int]]) -> list[str]:
    """The phase rewrites of each node at its index, each once."""
    out: list[str] = []
    for n, k in pairs:
        rw = _zc_terms(c, n, k)[1]
        if rw and rw not in out:
            out.append(rw)
    if any(n.kind == "ZW" for n, _ in pairs):
        out.append(f"norm_num only [{c.ex.unfold}, Ψ, ite_true, ite_false]")
    return out


def _coef_case(c: _Ctx, lhs: sp.Expr, rhs: sp.Expr, rewrites: Sequence[str]) -> str:
    """Tactics proving one coefficient identity once ``zc`` is unfolded: ``rewrites``, every
    root of unity as a power of one, then ``linear_combination``."""
    steps = list(rewrites)
    orders = {int(a.args[0]) for a in (lhs - rhs).atoms(Om)}
    if orders:
        big = (
            int(sp.ilcm(*orders, 4)) if (lhs - rhs).has(sp.Symbol("ii")) else int(sp.ilcm(*orders))
        )
        subs: dict[sp.Expr, sp.Expr] = {}
        rws = []
        for m in sorted(orders):
            if m != big:
                subs[Om(m)] = Om(big) ** (big // m)
                rws.append(
                    f"show om {m} = om {big} ^ {big // m} from "
                    f"om_sub {big} {m} {big // m} (by norm_num) (by norm_num)"
                )
        if (lhs - rhs).has(sp.Symbol("ii")):
            subs[sp.Symbol("ii")] = Om(big) ** (big // 4)
            rws.append(
                f"show Complex.I = om {big} ^ {big // 4} from "
                f"I_sub {big} {big // 4} (by norm_num) (by norm_num)"
            )
        if rws:
            steps.append(f"simp only [{', '.join(rws)}]")
        lhs, rhs = lhs.xreplace(subs), rhs.xreplace(subs)
    try:
        proof = c.ex.sc.prove(lhs, rhs)
    except ScalarUnsupported as exc:
        raise NetUnsupported(str(exc)) from exc
    steps.append(f"exact ({proof})")
    return "; ".join(f"all_goals {step}" for step in steps)


def _coef_hyp(c: _Ctx, left: LNode, right: Sequence[LNode]) -> str:
    """A proof of ``∀ k < E.dimv dz, zc E left k = ∏ zc E right k`` at a concrete dimension."""
    d_expr = c.ex.dim_exprs[left.dim]
    _need(d_expr.is_Integer, "coefficients at a concrete dimension")
    cases = []
    for k in range(int(d_expr)):
        lhs = _zc_terms(c, left, k)[0]
        rhs = sp.Mul(*(_zc_terms(c, r, k)[0] for r in right))
        rewrites = _coef_rewrites(c, [left, *right], [k])
        body = _coef_case(c, lhs, rhs, rewrites).replace("\n", "\n      ")
        cases.append(f"· simp only [zc]; {body}")
    lines = "\n    ".join(cases)
    return (
        f"(by\n    intro k hk\n    simp only [{c.ex.unfold}, Δ] at hk\n    interval_cases k\n"
        f"    {lines})"
    )


def _coef_at(c: _Ctx, k: int, lhs: sp.Expr, rhs: sp.Expr, nodes: Sequence[LNode]) -> str:
    """Tactics proving one coefficient identity at the literal index ``k``."""
    body = _coef_case(c, lhs, rhs, _coef_rewrites(c, nodes, [k]))
    return f"simp only [zc]; {body}"


def _zc_ad(c: _Ctx) -> _Spec:
    nodes = dict(c.match.nodes)
    w, na, nb = c.node(nodes["w"]), c.node(nodes["za"]), c.node(nodes["zb"])
    ((rid, r),) = c.new_nodes.items()
    _need(w.kind == "W" and len(w.outs) == 2 and len(w.ins) == 1, "W shape")
    x, y = w.outs
    (i,) = w.ins
    _need(na.labels() == (x,) and nb.labels() == (y,) and r.labels() == (i,), "effect shape")
    _need(all(n.kind in _ZC for n in (na, nb, r)), "spider effects")
    d_expr = c.ex.dim_exprs[w.dim]
    _need(d_expr.is_Integer, "coefficients at a concrete dimension")
    nodes3 = [r, na, nb]

    def model(n: LNode, k: int) -> sp.Expr:
        return _zc_terms(c, n, k)[0]

    h0 = _coef_at(c, 0, model(r, 0), model(na, 0) * model(nb, 0), nodes3)
    cases = ["· exact absurd rfl hk0"]
    for k in range(1, int(d_expr)):
        rhs = model(na, k) * model(nb, 0) + model(na, 0) * model(nb, k)
        pairs = [(n, q) for q in (k, 0) for n in nodes3]
        body = _coef_case(c, model(r, k), rhs, _pair_rewrites(c, pairs))
        cases.append(f"· simp only [zc]; {body}")
    body = "\n    ".join(cases)
    hc = (
        f"(by\n    intro k hk hk0\n    simp only [{c.ex.unfold}, Δ] at hk\n    interval_cases k\n"
        f"    {body})"
    )
    lemma = (
        f"zcAd {c.env} {w.dim} {x} {y} {i} {_zc(na)} {_zc(nb)} {_zc(r)} IFACE (by decide) "
        f"(by decide) (by decide) rfl rfl rfl rfl rfl rfl (by decide) (by decide) (by decide) "
        f"{_CHK} {_CHK} {_CHK} (by {h0}) {hc}"
    )
    return _Spec((nodes["w"], nodes["za"], nodes["zb"]), [(rid, r)], lemma)


def _zc_k2(c: _Ctx) -> _Spec:
    m = c.match
    nodes = dict(m.nodes)
    px, za = c.node(nodes["p"]), c.node(nodes["z"])
    _need(px.kind == "X" and za.kind in _ZC and len(c.new_nodes) == 2, "K2 shape")
    ((w,), (i,)) = px.outs, px.ins
    _need(za.ins == (w,) and len(za.outs) == 1, "K2 spider shape")
    (o,) = za.outs
    zs = [kv for kv in c.new_nodes.items() if kv[1].kind in _ZC]
    xs = [kv for kv in c.new_nodes.items() if kv[1].kind == "X"]
    _need(len(zs) == 1 and len(xs) == 1, "K2 right nodes")
    ((zid, nb),), ((pid, py),) = zs, xs
    _need(py.ph == px.ph, "K2 right shape")
    _need(nb.ins == (i,) and py.outs == (o,) and nb.outs == py.ins, "K2 right wiring")
    (w2,) = nb.outs
    d_expr = c.ex.dim_exprs[px.dim]
    _need(d_expr.is_Integer, "coefficients at a concrete dimension")
    d = int(d_expr)
    j = int(sp.Integer(dict(m.bindings)["j"])) % d
    k = c.ex.scalar(c.step.scalar_introduced)
    cases = []
    for q in range(d):
        r = (q + j) % d
        lhs = _zc_terms(c, za, q)[0]
        rhs = k * _zc_terms(c, nb, r)[0]
        body = _coef_case(c, lhs, rhs, _pair_rewrites(c, [(za, q), (nb, r)]))
        cases.append(
            f"· simp only [show {c.env}.dimv {px.dim} = {d} from rfl, Nat.reduceAdd, "
            f"Nat.reduceMod, zc]; {body}"
        )
    lines = "\n    ".join(cases)
    hc = (
        f"(by\n    intro k hk\n    simp only [{c.ex.unfold}, Δ] at hk\n    interval_cases k\n"
        f"    {lines})"
    )
    lemma = (
        f"zcK2 {c.env} {px.dim} {px.ph} {j} {w} {w2} {o} {i} {_zc(za)} {_zc(nb)} _ IFACE "
        f"(by decide) (by decide) rfl rfl rfl rfl {_CHK} {_CHK} (by decide) (by decide) "
        f"(by decide) (by decide) (by decide) (by decide) {_positive(c, px.dim)} "
        f"{_kj_proof(c, px.ph, px.dim, j)} {hc}"
    )
    return _Spec((nodes["p"], nodes["z"]), [(zid, nb), (pid, py)], lemma, k)


def _zc_ti(c: _Ctx) -> _Spec:
    nodes = dict(c.match.nodes)
    w, ef = c.node(nodes["w"]), c.node(nodes["e"])
    ((rid, r),) = c.new_nodes.items()
    _need(w.kind == "W" and len(w.outs) == 2 and len(w.ins) == 1, "W shape")
    (e, o), (i,) = w.outs, w.ins
    _need(ef.kind in _ZC and ef.labels() == (e,), "effect shape")
    _need(r.kind == "Ti" and r.outs == (o,) and r.ins == (i,), "inverse triangle shape")
    d_expr = c.ex.dim_exprs[w.dim]
    _need(d_expr.is_Integer, "coefficients at a concrete dimension")
    h0 = _coef_at(c, 0, _zc_terms(c, ef, 0)[0], ONE, [ef])
    cases = ["· exact absurd rfl hk0"]
    for k in range(1, int(d_expr)):
        cases.append(f"· {_coef_at(c, k, _zc_terms(c, ef, k)[0], -ONE, [ef])}")
    body = "\n    ".join(cases)
    hc = (
        f"(by\n    intro k hk hk0\n    simp only [{c.ex.unfold}, Δ] at hk\n    interval_cases k\n"
        f"    {body})"
    )
    lemma = (
        f"zcTi {c.env} {w.dim} {e} {o} {i} {_zc(ef)} IFACE (by decide) rfl rfl {_CHK} "
        f"(by decide) (by decide) (by decide) (by decide) (by {h0}) {hc}"
    )
    return _Spec((nodes["w"], nodes["e"]), [(rid, r)], lemma)


def _zc_congr(c: _Ctx) -> _Spec:
    nodes = dict(c.match.nodes)
    n = c.node(nodes["z"])
    ((rid, r),) = c.new_nodes.items()
    _need(n.kind in _ZC and r.kind in _ZC and n.dim == r.dim, "spider shape")
    lemma = (
        f"zcCongr {c.env} {_zc(n)} {_zc(r)} IFACE (by decide) (by decide) rfl (by decide) "
        f"{_coef_hyp(c, r, [n])}"
    )
    return _Spec((nodes["z"],), [(rid, r)], lemma)


def _zc_scalar(c: _Ctx) -> _Spec:
    nodes = dict(c.match.nodes)
    n = c.node(nodes["z"])
    _need(n.kind in _ZC and not n.outs and not n.ins and not c.new_nodes, "scalar box shape")
    k = c.ex.scalar(c.step.scalar_introduced)
    d_expr = c.ex.dim_exprs[n.dim]
    _need(d_expr.is_Integer, "a scalar box at a concrete dimension")
    d = int(d_expr)
    total = sp.Add(*(_zc_terms(c, n, j)[0] for j in range(d)))
    body = _coef_case(c, k, total, _coef_rewrites(c, [n], range(d)))
    hs = (
        f"(by dsimp only; rw [show {c.env}.dimv {n.dim} = {d} from rfl]; simp only "
        f"[Finset.sum_range_succ, Finset.sum_range_zero, zero_add, zc]; {body})"
    )
    lemma = f"zcScalar {c.env} {_zc(n)} IFACE _ (by decide) rfl rfl {hs}"
    return _Spec((nodes["z"],), [], lemma, k)


def _zc_ept(c: _Ctx) -> _Spec:
    nodes = dict(c.match.nodes)
    st, ef = c.node(nodes["p"]), c.node(nodes["z"])
    w = c.wire(c.match.wires[0])
    _need(st.kind == "X" and st.outs == (w,) and not st.ins, "pink state shape")
    _need(ef.kind in _ZC and ef.labels() == (w,), "spider effect shape")
    lemma = (
        f"zcEpt {c.env} {st.ph} {st.dim} {w} {_zc(ef)} IFACE (by decide) rfl (by decide) {_CHK} "
        f"{_positive(c, st.dim)} (by phz) (by simp only [zc, {c.ex.unfold}, Φ]; "
        "(try simp only [Ψ]); simp [ex_zero])"
    )
    return _Spec((nodes["p"], nodes["z"]), [], lemma, c.rt(st.dim))


def _zc_zer(c: _Ctx) -> _Spec:
    nodes = dict(c.match.nodes)
    n = c.node(nodes["z"])
    ((rid, r),) = c.new_nodes.items()
    _need(n.kind in _ZC and len(n.labels()) == 1 and r.kind == "X", "zero state shape")
    (o,) = n.labels()
    _need(r.outs == (o,) and not r.ins, "pink state shape")
    lemma = (
        f"zcZer {c.env} {r.ph} {r.dim} {o} {_zc(n)} IFACE (by decide) rfl (by decide) (by decide) "
        f"{_CHK} {_positive(c, r.dim)} (by phz) (by zcz)"
    )
    return _Spec((nodes["z"],), [(rid, r)], lemma, Inv(c.rt(r.dim)))


def _z_perm(c: _Ctx) -> _Spec:
    m = c.match
    nodes = dict(m.nodes)
    n = c.node(nodes["z"])
    ((rid, r),) = c.new_nodes.items()
    if "ZW" in (n.kind, r.kind):
        return _zc_congr(c)
    _need(n.kind == "Z" and r.kind == "Z" and n.ph == r.ph, "Z shape")
    _need(sorted(n.outs + n.ins) == sorted(r.outs + r.ins), "Z legs permuted")
    lemma = (
        f"zPerm {c.env} {n.ph} {n.dim} {_list(n.outs)} {_list(n.ins)} {_list(r.outs)} "
        f"{_list(r.ins)} IFACE (by decide)"
    )
    return _Spec((nodes["z"],), [(rid, r)], lemma)


def _template_fusion(c: _Ctx) -> _Spec:
    _need(len(c.match.wires) == 1, "fusion along one wire")
    return _template(_fusion, a_id="a", b_id="b")(c)


def _w_aso(c: _Ctx) -> _Spec:
    m = c.match
    nodes = dict(m.nodes)
    first, second = c.node(nodes["w1"]), c.node(nodes["w2"])
    w = c.wire(m.wires[0])
    rs = [(nid, c.new_nodes[nid]) for nid in c.step.new_node_ids]
    _need(len(rs) == 2 and len(first.ins) == 1 and second.ins == (w,), "Aso shape")
    i = first.ins[0]
    outer = [x for x in rs if x[1].ins == (i,)]
    inner = [x for x in rs if x[1].ins != (i,)]
    _need(len(outer) == 1 and len(inner) == 1, "Aso right side")
    p1, p2 = outer[0][1], inner[0][1]
    w2 = p2.ins[0]
    o3 = [x for x in first.outs if x != w] + list(second.outs)
    dz = first.dim
    lemma = (
        f"wAso {c.env} {dz} {w} {w2} {i} {_list(first.outs)} {_list(second.outs)} "
        f"{_list(p1.outs)} {_list(p2.outs)} {_list(o3)} IFACE {_chks(6)} {_positive(c, dz)} "
        f"{_chks(7)}"
    )
    return _Spec((nodes["w1"], nodes["w2"]), [outer[0], inner[0]], lemma)


def _kj_proof(c: _Ctx, ph: int, dz: int, j: int) -> str:
    """A proof of ``∀ q < E.dimv dz, ex (E.phase ph q) = om (E.dimv dz) ^ (q * j)`` at a
    concrete dimension: one ``ex_of`` per index."""
    d_expr = c.ex.dim_exprs[dz]
    _need(d_expr.is_Integer, "K_j phases at a concrete dimension")
    d = int(d_expr)
    entries, rule = c.ex.phase_specs[ph]
    _need(rule is None, "K_j phases written entry by entry")
    cases = ["· simp [E, Φ, ex_zero]"]
    for q in range(1, d):
        turns = sp.nsimplify(entries.get(q, sp.Integer(0)))
        z = turns - sp.Rational(q * j, d)
        _need(bool(z.is_Integer), f"phase at {q} is not {q * j}/{d} turns")
        cases.append(f"· exact ex_of {d} {q * j} ({int(z)}) (by norm_num) (by norm_num [E, Φ])")
    body = "\n    ".join(cases)
    return f"(by\n    intro q hq\n    simp only [E, Δ] at hq\n    interval_cases q\n    {body})"


def _k_copy(c: _Ctx) -> _Spec:
    m = c.match
    nodes = dict(m.nodes)
    st, sp_ = c.node(nodes["p"]), c.node(nodes["g"])
    w = c.wire(m.wires[0])
    _need(st.kind == "X" and st.outs == (w,) and not st.ins and sp_.ins == (w,), "K0 shape")
    rs = [(nid, c.new_nodes[nid]) for nid in c.step.new_node_ids]
    phs = {n.ph for _, n in rs}
    _need(len(phs) == 1, "new states differ")
    cph = phs.pop()
    j = int(sp.Integer(dict(m.bindings)["j"]))
    dz, n = st.dim, len(sp_.outs)
    k, kt, hk = c.copy_scalar(dz, n)
    lemma = (
        f"kCopy {c.env} {st.ph} {sp_.ph} {cph} {dz} {w} {j} {_list(sp_.outs)} IFACE {kt} "
        f"{_positive(c, dz)} {_chks(3)} {_kj_proof(c, st.ph, dz, j)} {_phaseless(c, sp_)} "
        f"{_kj_proof(c, cph, dz, j)} {hk}"
    )
    return _Spec((nodes["p"], nodes["g"]), rs, lemma, k)


_FORWARD: dict[str, str] = {
    "x_fusion_converse": "spider_fusion",
    "triangle_inverse_cancellation_ti_t_converse": "triangle_inverse_cancellation",
    "fourier_effect_color_change_converse": "fourier_state_color_change",
    "fourier_effect_color_change_swapped_converse": "fourier_state_color_change_swapped",
    "triangle_zero_state_t_converse": "triangle_zero_state",
    "triangle_zero_state_ti_converse": "triangle_zero_state",
    "connective_states_states_converse": "connective_states",
    "connective_states_effects_converse": "connective_states",
    "connective_states_swapped_states_converse": "connective_states_swapped",
    "connective_states_swapped_effects_converse": "connective_states_swapped",
}
"""Converse rules whose forward rule has another name."""


def _through_forward(t: object) -> object | None:
    """A step's theorem rebuilt as the symmetric of a step of its reverse rule (the forward
    rule of a converse, the converse of a rule) applied to the step's result and closed by an
    isomorphism; None when no such step exists."""
    from archytaszx.diagram.compare import isomorphic
    from archytaszx.rewrite.engine import apply as apply_rule
    from archytaszx.rewrite.normal_form import comparison_view, views_isomorphic
    from archytaszx.rewrite.rule import RewriteError
    from archytaszx.rewrite.rules_library import RULES
    from archytaszx.semantics import kernel

    assert isinstance(t, kernel.Theorem)
    name = t.inference.rule
    if name.endswith("_converse"):
        forward = _FORWARD.get(name, name.removesuffix("_converse"))
    else:
        forward = f"{name}_converse"
    if forward == name or forward not in RULES or forward not in _RULES:
        return None
    rule = RULES[forward]
    target = comparison_view(t.lhs)
    try:
        matches = rule.pattern.find_matches(t.rhs)
    except RewriteError:
        return None
    for match in matches:
        try:
            result = apply_rule(t.rhs, rule, match)
        except RewriteError:
            continue
        view = comparison_view(result.diagram)
        if not (isomorphic(result.diagram, t.lhs) or views_isomorphic(view, target)):
            continue
        try:
            step = kernel.rule(t.rhs, result.step)
            closed = kernel.trans(step, kernel.iso(result.diagram, t.lhs))
        except kernel.KernelError:
            continue
        return kernel.symm(closed)
    return None


@dataclass(frozen=True)
class _Shim:
    """A template match read through the fields an existing rule's hook expects."""

    fields: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name) from None


def _template(builder: _RuleBuilder, **keys: str | bool) -> _RuleBuilder:
    """``builder`` applied to a template match: each field a template node key, ``wire`` the
    match's internal wire, a bool field given as is."""

    def hook(c: _Ctx) -> _Spec:
        m = c.match
        nodes = dict(m.nodes)
        fields: dict[str, Any] = {}
        for field_name, key in keys.items():
            fields[field_name] = key if isinstance(key, bool) else nodes[key]
        if m.wires:
            fields["wire"] = m.wires[0]
        return builder(replace(c, step=replace(c.step, match=_Shim(fields))))

    return hook


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
    "wz_bialgebra": _wz,
    "port_box_unfusion": _unfuse,
    "zxw_s2": _template(_identity, node_id="z"),
    "zxw_s2_pink": _template(_identity, node_id="p"),
    "zxw_ept": _zc_ept,
    "zxw_zer": _zc_zer,
    "phase_to_weight": _zc_congr,
    "weight_to_phase": _zc_congr,
    "scalar_box": _zc_scalar,
    "zxw_k0": _k_copy,
    "zxw_bs0": _template(_fed_state, state_id="p", target_id="w"),
    "zxw_b2_converse": _template(_bialgebra, x_id="p", z_id="g", swapped=False),
    "zxw_bzw_converse": _template(_wz, z_id="g", w_id="w"),
    "qf_wf": _template(_w_fusion, first_id="w1", second_id="w2"),
    "zxw_sym": _w_perm,
    "zxw_sym_converse": _w_perm,
    "qf_w_sym": _w_perm,
    "qf_w_sym_converse": _w_perm,
    "zxw_s1": _template_fusion,
    "zxw_aso": _w_aso,
    "zxw_aso_converse": _w_aso,
    "qf_z_flex": _z_perm,
    "qf_z_flex_converse": _z_perm,
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


def export(
    theorem: object, name: str = "main", describe: Callable[[Diagram], str] | None = None
) -> str:
    """Lean 4 source proving ``theorem`` over the tensor-network semantics."""
    from archytaszx.semantics.kernel import Theorem

    if not isinstance(theorem, Theorem):
        raise NetUnsupported(f"export takes a Theorem, got {type(theorem).__name__}")
    out = _Export(describe or (lambda d: f"a diagram of {len(d.nodes)} nodes"))
    for a, b in _deferred(theorem):
        out.assume(a, b)
    try:
        return out.finish(theorem, name)
    except ScalarUnsupported as exc:
        raise NetUnsupported(str(exc)) from exc


def _deferred(t: object) -> list[tuple[sp.Expr, sp.Expr]]:
    """Every dimension equality a step of ``t`` assumed."""
    from archytaszx.rewrite.rule import ConstraintOutcome

    found: list[tuple[sp.Expr, sp.Expr]] = []
    stack = [t]
    while stack:
        u = stack.pop()
        stack.extend(u.inference.premises)  # type: ignore[attr-defined]
        step = u.inference.data.get("step")  # type: ignore[attr-defined]
        if isinstance(step, RewriteStep):
            for c in step.dimension_constraints:
                if c.outcome is ConstraintOutcome.DEFERRED:
                    found.append((c.assumed.to_sympy(), c.equal_to.to_sympy()))
    return found


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
