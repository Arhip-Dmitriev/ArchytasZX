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

"""Textual output for engine states: Dirac rendering, graph listings, and notation modes.

Dirac rendering. :func:`dirac_term` reads a :class:`~archytaszx.diagram.graph.Diagram` into a
:class:`DiracTerm`, the structural index-sum form with one bound index per spider; it never
contracts, evaluates or rewrites. :func:`render_term` prints a term as text, and
:func:`recognize` names a diagram from a small catalog (``GHZ_{n}``, ``copy_{n}``, ``W_{n}``,
...) when every connected component is one bare generator.

Semantics of a term, matching :mod:`archytaszx.semantics.denote` axis for axis (outputs, then
inputs):

* ``scalar * sum_{indices} prod_{factors} |kets><bras|``. An :class:`IndexVar` is a family:
  one index in ``0..dim-1`` per value of its copy variables.
* A :class:`CopyVar` ``i{b}`` ranges over the copies of bang box ``b``. A node in node-scope
  boxes has its indices and its factor taken once per copy vector (a :class:`Product`).
* A :class:`Leg` with a fan is one leg per fan value; a :class:`Fan` ket item likewise, and a
  :class:`Block` lists its items once per value of its copy variable, copy-major, exactly
  as :func:`~archytaszx.diagram.bangbox.instantiate_symbol` lays out a boundary.
* ``Z``: every leg carries the spider index ``k``, weighted ``e^{2 pi i alpha(k)}``.
* ``X``: ``d^{-(m+n)/2} sum_q e^{2 pi i alpha(q)} w_d^{q (sum outs - sum ins)}``.
* ``F``: ``d^{-1/2} w_d^{o i}``; ``T``: ``[o = 0 or o = i]``; ``Ti``:
  ``[o = i] - [o = 0][i >= 1]``; ``W``: ``[in = sum outs][nnz(outs) <= 1]``; ``B`` and ``S``:
  ``[ab = a*t + b]``.

Text is ASCII by default; ``unicode=True`` swaps ``sum``, ``prod``, ``(x)``, ``w``, ``pi`` and
ket/bra brackets for their symbols. :func:`to_dirac_source` emits the Phase 5 parser's own
grammar for a single phase-free Z state.

Errors: :class:`PrinterGrammarError` for a wrong argument type, :class:`PrinterDomainError`
for a diagram with no Dirac form (a validate error, or a fan on a fixed-arity leg).
"""

from __future__ import annotations

import enum
import itertools
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim, DimensionError
from archytaszx.algebra.phase import PhaseError, PhaseVector
from archytaszx.algebra.scalar import Scalar, ScalarError
from archytaszx.diagram.bangbox import BangBox, BangBoxError, Mult
from archytaszx.diagram.compare import compare_structure
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorError,
)
from archytaszx.diagram.graph import (
    BangBoxId,
    Diagram,
    Direction,
    GraphError,
    Node,
    NodeId,
    PortRef,
    Wire,
)
from archytaszx.diagram.scalable import (
    BoundaryItem,
    Bundle,
    ScalableDiagram,
    ScalableError,
    Scale,
    ScaleId,
    SheetPort,
    from_scalable,
    group_boundary,
)
from archytaszx.diagram.validate import ValidateError, validate
from archytaszx.rewrite.engine import RewriteResult, RewriteStep, apply
from archytaszx.rewrite.normal_form import NormalForm
from archytaszx.rewrite.rules_library import lookup_rule
from archytaszx.rewrite.sheet import SheetStep
from archytaszx.semantics.certificate import Certificate, Derivation, DerivationKind
from archytaszx.semantics.denote import DenoteError, resolve_dim
from archytaszx.semantics.prove import InductionProof, ProofCertificate


class PrinterError(Exception):
    """Base class for all errors raised by this module."""


class PrinterDomainError(PrinterError):
    """A diagram whose Dirac form is undefined: a validate error or a fanned fixed-arity leg."""


class PrinterGrammarError(PrinterError):
    """An argument of the wrong type."""


_FOREIGN = (
    BangBoxError,
    DenoteError,
    DimensionError,
    GeneratorError,
    GraphError,
    PhaseError,
    ScalableError,
    ScalarError,
    ValidateError,
)


class NotationMode(enum.Enum):
    """The session notation mode: which renderings :func:`render` emits."""

    DIRAC = "dirac"
    ZX = "zx"


DEFAULT_MODE = NotationMode.DIRAC
"""The notation mode a session starts in."""


# -- Dirac AST -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CopyVar:
    """The copy index of bang box ``box``, ranging over ``1..count``."""

    name: str
    box: BangBoxId
    count: Mult


@dataclass(frozen=True, slots=True)
class IndexVar:
    """A family of bound indices in ``0..dim-1``, one per value of ``copies``."""

    name: str
    dim: Dim
    copies: tuple[CopyVar, ...]


@dataclass(frozen=True, slots=True)
class IndexAt:
    """The member of ``var`` addressed by the copy variables ``at``."""

    var: IndexVar
    at: tuple[CopyVar, ...]


@dataclass(frozen=True, slots=True)
class Leg:
    """One leg, or one leg per value of the ``fan`` copy variables."""

    index: IndexAt
    fan: tuple[CopyVar, ...] = ()


@dataclass(frozen=True, slots=True)
class Delta:
    """``[left = right]``."""

    left: IndexAt
    right: IndexAt


@dataclass(frozen=True, slots=True)
class ZFactor:
    """``e^{2 pi i phase(index)}``."""

    index: IndexAt
    phase: PhaseVector


@dataclass(frozen=True, slots=True)
class XFactor:
    """One X spider: normalisation, phase on ``kappa``, and the Fourier character."""

    kappa: IndexAt
    phase: PhaseVector | None
    outputs: tuple[Leg, ...]
    inputs: tuple[Leg, ...]
    dim: Dim


@dataclass(frozen=True, slots=True)
class FourierFactor:
    """``d^{-1/2} w_d^{out inp}``."""

    out: IndexAt
    inp: IndexAt
    dim: Dim


@dataclass(frozen=True, slots=True)
class TriangleFactor:
    """``[out = 0 or out = inp]``."""

    out: IndexAt
    inp: IndexAt


@dataclass(frozen=True, slots=True)
class TriangleInverseFactor:
    """``[out = inp] - [out = 0][inp >= 1]``."""

    out: IndexAt
    inp: IndexAt


@dataclass(frozen=True, slots=True)
class WFactor:
    """``[inp = sum outputs][nnz(outputs) <= 1]``."""

    inp: IndexAt
    outputs: tuple[Leg, ...]


@dataclass(frozen=True, slots=True)
class BinderFactor:
    """``[out = a*t + b]``."""

    out: IndexAt
    a: IndexAt
    b: IndexAt
    t: Dim


@dataclass(frozen=True, slots=True)
class SplitterFactor:
    """``[inp = a*t + b]``."""

    inp: IndexAt
    a: IndexAt
    b: IndexAt
    t: Dim


@dataclass(frozen=True, slots=True)
class Product:
    """The product of ``factors`` over every value of the ``over`` copy variables."""

    over: tuple[CopyVar, ...]
    factors: tuple[Factor, ...]


Factor: TypeAlias = (
    Delta
    | ZFactor
    | XFactor
    | FourierFactor
    | TriangleFactor
    | TriangleInverseFactor
    | WFactor
    | BinderFactor
    | SplitterFactor
    | Product
)
"""Any factor of a :class:`DiracTerm`."""


@dataclass(frozen=True, slots=True)
class Slot:
    """One boundary leg carrying ``index``."""

    index: IndexAt


@dataclass(frozen=True, slots=True)
class Fan:
    """One boundary leg per value of ``over``, each carrying ``index`` at that value."""

    index: IndexAt
    over: tuple[CopyVar, ...]


@dataclass(frozen=True, slots=True)
class Block:
    """``items`` once per value of ``over``, copy-major."""

    over: CopyVar
    items: tuple[KetItem, ...]


KetItem: TypeAlias = Slot | Fan | Block
"""Any boundary item of a :class:`DiracTerm`."""


@dataclass(frozen=True, slots=True)
class DiracTerm:
    """``scalar * sum_{indices} prod(factors) |kets><bras|``."""

    scalar: Scalar
    indices: tuple[IndexVar, ...]
    factors: tuple[Factor, ...]
    kets: tuple[KetItem, ...]
    bras: tuple[KetItem, ...]


@dataclass(frozen=True, slots=True)
class DiracRendering:
    """A diagram's Dirac form: term, catalog name, text, instance text and parser source."""

    term: DiracTerm
    name: str | None
    text: str
    instance_text: str | None
    source: str | None


# -- diagram to term -----------------------------------------------------------------------


def _require_diagram(diagram: object) -> Diagram:
    if not isinstance(diagram, Diagram):
        raise PrinterGrammarError(f"expected a Diagram, got {type(diagram).__name__}")
    return diagram


def _require_valid(diagram: Diagram) -> None:
    try:
        errors = validate(diagram).errors
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise PrinterDomainError(f"diagram has no Dirac form: malformed ({exc})") from exc
    if errors:
        raise PrinterDomainError(
            "diagram has no Dirac form: " + "; ".join(issue.message for issue in errors)
        )


def _depth(diagram: Diagram, box_id: BangBoxId) -> int:
    depth = 0
    current = diagram.bang_boxes[box_id].parent
    while current is not None:
        depth += 1
        current = diagram.bang_boxes[current].parent
    return depth


def _symbols(diagram: Diagram) -> frozenset[str]:
    """Every symbol name ``diagram`` carries or binds."""
    names: set[str] = set(diagram.scalar.free_symbols) | set(diagram.parameters)
    for node in diagram.nodes.values():
        for port in (*node.inputs, *node.outputs):
            names |= port.dim.free_symbols
        if node.phase is not None:
            names |= node.phase.free_symbols
    for box in diagram.bang_boxes.values():
        names |= box.multiplicity.free_symbols
    return frozenset(names)


def _fresh(name: str, taken: frozenset[str]) -> str:
    """``name`` primed until it is not in ``taken``."""
    while name in taken:
        name += "'"
    return name


class _Structure:
    """Node-scope chains, port fans and copy variables of a valid diagram."""

    def __init__(self, diagram: Diagram) -> None:
        self.diagram = diagram
        self.taken = _symbols(diagram)
        boxes = diagram.bang_boxes
        self.copy_vars = {
            box_id: CopyVar(_fresh(f"i{int(box_id)}", self.taken), box_id, box.multiplicity)
            for box_id, box in sorted(boxes.items())
        }
        self.chains: dict[NodeId, tuple[BangBoxId, ...]] = {}
        for node_id in sorted(diagram.nodes):
            holders = [b for b, box in boxes.items() if node_id in box.node_scope]
            self.chains[node_id] = self._walk(holders, node_scope=True)
        self.fans: dict[PortRef, tuple[BangBoxId, ...]] = {}
        for box in boxes.values():
            for ref in box.port_scope:
                if ref not in self.fans:
                    holders = [b for b, other in boxes.items() if ref in other.port_scope]
                    self.fans[ref] = self._walk(holders, node_scope=False)

    def _walk(self, holders: list[BangBoxId], *, node_scope: bool) -> tuple[BangBoxId, ...]:
        """The parent walk from the deepest holder, outer to inner, kept to one scope kind."""
        if not holders:
            return ()
        boxes = self.diagram.bang_boxes
        current: BangBoxId | None = max(holders, key=lambda b: (_depth(self.diagram, b), b))
        chain: list[BangBoxId] = []
        while current is not None and boxes[current].is_node_scope == node_scope:
            chain.append(current)
            current = boxes[current].parent
        if node_scope and current is not None:
            raise PrinterDomainError(f"node-scope bang box {chain[-1]!r} has a port-scope parent")
        if set(chain) != set(holders):
            raise PrinterDomainError(f"bang boxes {sorted(holders)} do not form one nested chain")
        return tuple(reversed(chain))

    def chain_vars(self, node_id: NodeId) -> tuple[CopyVar, ...]:
        return tuple(self.copy_vars[b] for b in self.chains[node_id])

    def fan_vars(self, ref: PortRef) -> tuple[CopyVar, ...]:
        return tuple(self.copy_vars[b] for b in self.fans.get(ref, ()))


def _is(node: Node, generator_name: str) -> bool:
    return node.generator_type.name == generator_name


def _all_refs(node: Node) -> list[PortRef]:
    return [PortRef(node.id, Direction.OUTPUT, i) for i in range(node.num_outputs)] + [
        PortRef(node.id, Direction.INPUT, i) for i in range(node.num_inputs)
    ]


def _leg_name(ref: PortRef, taken: frozenset[str]) -> str:
    side = "o" if ref.direction is Direction.OUTPUT else "i"
    return _fresh(f"j{int(ref.node_id)}{side}{ref.index}", taken)


def _port_dim(diagram: Diagram, ref: PortRef) -> Dim:
    return diagram.nodes[ref.node_id].legs(ref.direction)[ref.index].dim


def _at(var: IndexVar) -> IndexAt:
    return IndexAt(var, var.copies)


_KNOWN = frozenset(
    g.name
    for g in (
        Z_SPIDER,
        X_SPIDER,
        FOURIER_BOX,
        TRIANGLE,
        TRIANGLE_INVERSE,
        W_NODE,
        DIM_BINDER,
        DIM_SPLITTER,
    )
)


class _TermBuilder:
    """Builds the :class:`DiracTerm` of one valid diagram."""

    def __init__(self, diagram: Diagram) -> None:
        self.d = diagram
        self.s = _Structure(diagram)
        self.partner: dict[PortRef, PortRef] = {}
        for wire in diagram.wires:
            self.partner[wire.a] = wire.b
            self.partner[wire.b] = wire.a
        for node in diagram.nodes.values():
            if node.generator_type.name not in _KNOWN:
                raise PrinterDomainError(
                    f"node {node.id!r} has generator {node.generator_type.name!r}, "
                    "which has no Dirac form"
                )
        z_nodes = [v for v in sorted(diagram.nodes) if _is(diagram.nodes[v], Z_SPIDER.name)]
        lone = len(diagram.nodes) == 1 and len(z_nodes) == 1 and not self.s.chains[z_nodes[0]]
        self.z_index = {
            v: IndexVar(
                _fresh("k" if lone else f"k{int(v)}", self.s.taken),
                resolve_dim(diagram.nodes[v]),
                self.s.chain_vars(v),
            )
            for v in z_nodes
        }
        self.x_index = {
            v: IndexVar(
                _fresh(f"q{int(v)}", self.s.taken),
                resolve_dim(diagram.nodes[v]),
                self.s.chain_vars(v),
            )
            for v in sorted(diagram.nodes)
            if _is(diagram.nodes[v], X_SPIDER.name)
        }
        self.leg_vars: dict[PortRef, IndexVar] = {}
        self.legs: dict[PortRef, Leg] = {}
        for node_id in sorted(diagram.nodes):
            if node_id in self.z_index:
                continue
            for ref in _all_refs(diagram.nodes[node_id]):
                self.legs[ref] = self._leg(ref)

    def _shared(self, ref: PortRef, other: PortRef) -> IndexVar:
        first, second = sorted((ref, other), key=lambda r: r.sort_key())
        if first not in self.leg_vars:
            own = self.s.chain_vars(first.node_id)
            extra = tuple(c for c in self.s.chain_vars(second.node_id) if c not in own)
            name = _leg_name(first, self.s.taken)
            self.leg_vars[first] = IndexVar(name, _port_dim(self.d, first), own + extra)
        return self.leg_vars[first]

    def _leg(self, ref: PortRef) -> Leg:
        own = self.s.chain_vars(ref.node_id)
        other = self.partner.get(ref)
        if other is None:
            fan = self.s.fan_vars(ref)
            var = IndexVar(_leg_name(ref, self.s.taken), _port_dim(self.d, ref), own + fan)
            self.leg_vars[ref] = var
            return Leg(_at(var), fan)
        if other.node_id in self.z_index:
            var = self.z_index[other.node_id]
        else:
            var = self._shared(ref, other)
        return Leg(_at(var), tuple(c for c in var.copies if c not in own))

    def _fixed(self, ref: PortRef) -> IndexAt:
        leg = self.legs[ref]
        if leg.fan:
            raise PrinterDomainError(f"fixed-arity leg {ref!r} is fanned over {leg.fan}")
        return leg.index

    def _node_factor(self, node: Node) -> Factor | None:
        v = node.id
        outs = [PortRef(v, Direction.OUTPUT, i) for i in range(node.num_outputs)]
        ins = [PortRef(v, Direction.INPUT, i) for i in range(node.num_inputs)]
        name = node.generator_type.name
        if name == Z_SPIDER.name:
            if node.phase is None or node.phase.is_zero:
                return None
            return ZFactor(_at(self.z_index[v]), node.phase)
        if name == X_SPIDER.name:
            phase = None if node.phase is None or node.phase.is_zero else node.phase
            return XFactor(
                _at(self.x_index[v]),
                phase,
                tuple(self.legs[r] for r in outs),
                tuple(self.legs[r] for r in ins),
                self.x_index[v].dim,
            )
        if name == W_NODE.name:
            return WFactor(self._fixed(ins[0]), tuple(self.legs[r] for r in outs))
        if name == DIM_BINDER.name:
            return BinderFactor(
                self._fixed(outs[0]),
                self._fixed(ins[0]),
                self._fixed(ins[1]),
                _port_dim(self.d, ins[1]),
            )
        if name == DIM_SPLITTER.name:
            return SplitterFactor(
                self._fixed(ins[0]),
                self._fixed(outs[0]),
                self._fixed(outs[1]),
                _port_dim(self.d, outs[1]),
            )
        out, inp = self._fixed(outs[0]), self._fixed(ins[0])
        if name == FOURIER_BOX.name:
            return FourierFactor(out, inp, resolve_dim(node))
        if name == TRIANGLE.name:
            return TriangleFactor(out, inp)
        return TriangleInverseFactor(out, inp)

    def _factors(self) -> tuple[Factor, ...]:
        raw: list[tuple[tuple[CopyVar, ...], Factor]] = []
        for node_id in sorted(self.d.nodes):
            factor = self._node_factor(self.d.nodes[node_id])
            if factor is not None:
                raw.append((self.s.chain_vars(node_id), factor))
        for wire in sorted(self.d.wires, key=lambda w: w.sort_key()):
            a, b = sorted((wire.a, wire.b), key=lambda r: r.sort_key())
            if a.node_id == b.node_id or a.node_id not in self.z_index:
                continue
            if b.node_id not in self.z_index:
                continue
            own = self.s.chain_vars(a.node_id)
            over = own + tuple(c for c in self.s.chain_vars(b.node_id) if c not in own)
            raw.append((over, Delta(_at(self.z_index[a.node_id]), _at(self.z_index[b.node_id]))))
        merged: list[Factor] = []
        for over, factor in raw:
            if not over:
                merged.append(factor)
            elif merged and isinstance(merged[-1], Product) and merged[-1].over == over:
                merged[-1] = Product(over, (*merged[-1].factors, factor))
            else:
                merged.append(Product(over, (factor,)))
        return tuple(merged)

    def _item(self, item: BoundaryItem) -> KetItem:
        if isinstance(item, Bundle):
            return Block(
                self.s.copy_vars[BangBoxId(int(item.scale))],
                tuple(self._item(inner) for inner in item.items),
            )
        if item.node_id in self.z_index:
            index = _at(self.z_index[item.node_id])
            fan = self.s.fan_vars(item)
        else:
            leg = self.legs[item]
            index, fan = leg.index, leg.fan
        return Fan(index, fan) if fan else Slot(index)

    def _boundary(self, refs: Sequence[PortRef]) -> tuple[KetItem, ...]:
        chains = {
            node_id: tuple(ScaleId(int(b)) for b in chain)
            for node_id, chain in self.s.chains.items()
        }
        return tuple(self._item(item) for item in group_boundary(refs, chains))

    def build(self) -> DiracTerm:
        legs = sorted(self.leg_vars.items(), key=lambda item: item[0].sort_key())
        indices = (
            *(self.z_index[v] for v in sorted(self.z_index)),
            *(self.x_index[v] for v in sorted(self.x_index)),
            *(var for _ref, var in legs),
        )
        return DiracTerm(
            scalar=self.d.scalar,
            indices=indices,
            factors=self._factors(),
            kets=self._boundary(self.d.boundary_outputs),
            bras=self._boundary(self.d.boundary_inputs),
        )


def dirac_term(diagram: Diagram) -> DiracTerm:
    """The structural index-sum form of ``diagram``, one bound index per spider."""
    checked = _require_diagram(diagram)
    try:
        _require_valid(checked)
        return _TermBuilder(checked).build()
    except _FOREIGN as exc:
        raise PrinterDomainError(f"diagram has no Dirac form: {exc}") from exc


# -- term to text --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Glyphs:
    sum: str
    prod: str
    otimes: str
    omega: str
    pi: str
    ket: str
    bra: str


_ASCII = _Glyphs("sum", "prod", "(x)", "w", "pi", ">", "<")
_UNICODE = _Glyphs("Σ", "∏", "⊗", "ω", "π", "⟩", "⟨")


def _subs(expr: object, env: Mapping[str, int]) -> sp.Expr:
    value = sp.sympify(expr)
    mapping = {s: env[s.name] for s in value.free_symbols if s.name in env}
    return value.subs(mapping) if mapping else value


def _plain(expr: sp.Expr) -> str:
    return str(sp.sstr(expr)).replace("**", "^").replace(" ", "")


class _Renderer:
    """Prints one :class:`DiracTerm` under an environment and a glyph set."""

    def __init__(self, env: Mapping[str, int], glyphs: _Glyphs) -> None:
        self.env = env
        self.g = glyphs

    def expr(self, value: object) -> str:
        return _plain(_subs(value, self.env))

    def atom(self, value: object) -> str:
        expr = _subs(value, self.env)
        text = _plain(expr)
        return text if expr.is_Atom and not text.startswith("-") else f"({text})"

    def dim(self, dim: Dim) -> str:
        return self.atom(dim.to_sympy())

    def count(self, var: CopyVar) -> str:
        return self.expr(var.count.to_sympy())

    def bound(self, dim: Dim) -> str:
        return f"{self.dim(dim)}-1"

    def ranges(self, copies: Sequence[CopyVar]) -> str:
        return ", ".join(f"{c.name}=1..{self.count(c)}" for c in copies)

    def big(self, op: str, copies: Sequence[CopyVar]) -> str:
        return " ".join(f"{op}_{{{c.name}=1}}^{{{self.count(c)}}}" for c in copies)

    def index(self, at: IndexAt) -> str:
        if not at.at:
            return at.var.name
        return f"{at.var.name}[{','.join(c.name for c in at.at)}]"

    def sums(self, indices: Sequence[IndexVar]) -> list[str]:
        parts: list[str] = []
        for (dim, copies), group in itertools.groupby(indices, key=lambda v: (v.dim, v.copies)):
            names = ",".join(self.index(_at(var)) for var in group)
            scope = f"{names}=0, {self.ranges(copies)}" if copies else f"{names}=0"
            parts.append(f"{self.g.sum}_{{{scope}}}^{{{self.bound(dim)}}}")
        return parts

    def leg_sum(self, leg: Leg) -> str:
        text = self.index(leg.index)
        return f"{self.big(self.g.sum, leg.fan)} {text}" if leg.fan else text

    def phase(self, index: IndexAt, phase: PhaseVector) -> str:
        terms = [
            f"{self.atom(entry.to_sympy_turns())} [{self.index(index)}={k}]"
            for k, entry in sorted(phase.entries().items())
        ]
        return f"e^{{2 {self.g.pi} i ({' + '.join(terms)})}}"

    def x_factor(self, f: XFactor) -> str:
        parts: list[str] = []
        width = sp.Integer(0)
        for leg in (*f.outputs, *f.inputs):
            term = sp.Integer(1)
            for c in leg.fan:
                term = term * c.count.to_sympy()
            width = width + term
        width = _subs(width, self.env)
        if width != 0:
            exponent = -width / 2
            body = _plain(width) if width.is_Atom else f"({_plain(width)})"
            shown = _plain(exponent) if exponent.is_Rational else f"-{body}/2"
            parts.append(f"{self.dim(f.dim)}^{{{shown}}}")
        if f.phase is not None:
            parts.append(self.phase(f.kappa, f.phase))
        signed = [("+", self.leg_sum(leg)) for leg in f.outputs]
        signed += [("-", self.leg_sum(leg)) for leg in f.inputs]
        if signed:
            body = ("-" if signed[0][0] == "-" else "") + signed[0][1]
            body += "".join(f" {sign} {text}" for sign, text in signed[1:])
            omega = f"{self.g.omega}_{{{self.expr(f.dim.to_sympy())}}}"
            parts.append(f"{omega}^{{{self.index(f.kappa)} ({body})}}")
        return " ".join(parts) if parts else "1"

    def w_factor(self, f: WFactor) -> str:
        inp = self.index(f.inp)
        if not f.outputs:
            return f"[{inp} = 0]"
        total = " + ".join(self.leg_sum(leg) for leg in f.outputs)
        listed = []
        for leg in f.outputs:
            text = self.index(leg.index)
            for c in reversed(leg.fan):
                text = f"({text})_{{{c.name}=1}}^{{{self.count(c)}}}"
            listed.append(text)
        return f"[{inp} = {total}][nnz({', '.join(listed)}) <= 1]"

    def factor(self, f: Factor) -> str:
        if isinstance(f, Product):
            inner = " ".join(self.factor(g) for g in f.factors)
            return f"{self.big(self.g.prod, f.over)}( {inner} )"
        if isinstance(f, Delta):
            return f"[{self.index(f.left)} = {self.index(f.right)}]"
        if isinstance(f, ZFactor):
            return self.phase(f.index, f.phase)
        if isinstance(f, XFactor):
            return self.x_factor(f)
        if isinstance(f, WFactor):
            return self.w_factor(f)
        if isinstance(f, FourierFactor):
            omega = f"{self.g.omega}_{{{self.expr(f.dim.to_sympy())}}}"
            return f"{self.dim(f.dim)}^{{-1/2}} {omega}^{{{self.index(f.out)} {self.index(f.inp)}}}"
        if isinstance(f, (BinderFactor, SplitterFactor)):
            whole = self.index(f.out if isinstance(f, BinderFactor) else f.inp)
            return f"[{whole} = {self.index(f.a)}*{self.dim(f.t)} + {self.index(f.b)}]"
        out, inp = self.index(f.out), self.index(f.inp)
        if isinstance(f, TriangleFactor):
            return f"[{out} = 0 or {out} = {inp}]"
        return f"([{out} = {inp}] - [{out} = 0][{inp} >= 1])"

    def items(self, items: Sequence[KetItem], *, ket: bool) -> str:
        parts: list[str] = []
        slots: list[str] = []

        def wrap(text: str) -> str:
            return f"|{text}{self.g.ket}" if ket else f"{self.g.bra}{text}|"

        def flush() -> None:
            if slots:
                parts.append(wrap(", ".join(slots)))
                slots.clear()

        for item in items:
            if isinstance(item, Slot):
                slots.append(self.index(item.index))
                continue
            flush()
            if isinstance(item, Block):
                inner = self.items(item.items, ket=ket)
                parts.append(f"{self.big(self.g.otimes, (item.over,))}( {inner} )")
                continue
            looped = [c for c in item.over if c in item.index.at]
            power = sp.Integer(1)
            for c in item.over:
                if c not in looped:
                    power = power * c.count.to_sympy()
            text = wrap(self.index(item.index))
            if power != 1:
                text = f"{text}^{{{self.expr(power)}}}"
            parts.append(f"{self.big(self.g.otimes, looped)} {text}" if looped else text)
        flush()
        return " ".join(parts)

    def term(self, term: DiracTerm) -> str:
        body = [*self.sums(term.indices), *(self.factor(f) for f in term.factors)]
        boundary = self.items(term.kets, ket=True) + self.items(term.bras, ket=False)
        if boundary:
            body.append(boundary)
        scalar = _subs(term.scalar.to_sympy(), self.env)
        if not body:
            return _plain(scalar)
        if term.indices and not term.factors and not boundary:
            body.append("1")
        if scalar != 1:
            body.insert(0, f"({_plain(scalar)})")
        return " ".join(body)


def render_term(
    term: DiracTerm, *, env: Mapping[str, int] | None = None, unicode: bool = False
) -> str:
    """``term`` as one line of text, with ``env`` substituted into every symbol it binds."""
    if not isinstance(term, DiracTerm):
        raise PrinterGrammarError(f"render_term expects a DiracTerm, got {type(term).__name__}")
    if env is None:
        env = {}
    if not isinstance(env, Mapping) or not all(
        isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
        for k, v in env.items()
    ):
        raise PrinterGrammarError("render_term expects env to map symbol names to ints")
    renderer = _Renderer(dict(env), _UNICODE if unicode else _ASCII)
    try:
        return renderer.term(term)
    except (*_FOREIGN, sp.SympifyError, TypeError, ValueError) as exc:
        raise PrinterDomainError(f"cannot render term: {exc}") from exc


# -- catalog names -------------------------------------------------------------------------


def _width(structure: _Structure, ref: PortRef) -> sp.Expr:
    total = sp.Integer(1)
    for var in structure.fan_vars(ref):
        total = total * var.count.to_sympy()
    return total


def _at_least_two(value: sp.Expr) -> bool:
    return bool((value - 2).is_nonnegative is True)


def _phase_label(phase: PhaseVector) -> str:
    return ", ".join(
        f"{k}: {_plain(p.to_sympy_turns())}" for k, p in sorted(phase.entries().items())
    )


def _z_name(m: sp.Expr, n: sp.Expr, phase: PhaseVector | None) -> str:
    generic = f"Z_{{{_plain(m)}->{_plain(n)}}}"
    if phase is not None:
        return "Z_phase" if m == 1 and n == 1 else f"{generic}({_phase_label(phase)})"
    for count, other, tail in ((n, m, ""), (m, n, "^dagger")):
        if other == 0 and count == 1:
            return f"plus{tail}"
        if other == 0 and _at_least_two(count):
            return f"GHZ_{{{_plain(count)}}}{tail}"
        if other == 1 and _at_least_two(count):
            return f"copy_{{{_plain(count)}}}{tail}"
    return "id" if m == 1 and n == 1 else generic


def _node_name(diagram: Diagram, structure: _Structure, node: Node) -> str | None:
    refs = _all_refs(node)
    widths = {ref: _width(structure, ref) for ref in refs}
    n = sum((widths[r] for r in refs if r.direction is Direction.OUTPUT), sp.Integer(0))
    m = sum((widths[r] for r in refs if r.direction is Direction.INPUT), sp.Integer(0))
    phase = None if node.phase is None or node.phase.is_zero else node.phase
    name = node.generator_type.name
    if name == Z_SPIDER.name:
        return _z_name(m, n, phase)
    if name == X_SPIDER.name:
        generic = f"X_{{{_plain(m)}->{_plain(n)}}}"
        if phase is not None:
            return f"{generic}({_phase_label(phase)})"
        if n == 1 and m in (0, 1):
            return "id" if m == 1 else "zero"
        return generic
    if name == W_NODE.name:
        return f"W_{{{_plain(n)}}}"
    order = {
        DIM_BINDER.name: ("bind", diagram.boundary_inputs, node.num_inputs, Direction.INPUT),
        DIM_SPLITTER.name: ("split", diagram.boundary_outputs, node.num_outputs, Direction.OUTPUT),
    }
    if name in order:
        label, side, count, direction = order[name]
        own = [r for r in side if r.node_id == node.id]
        expected = [PortRef(node.id, direction, i) for i in range(count)]
        return label if own == expected else None
    return {FOURIER_BOX.name: "F", TRIANGLE.name: "triangle", TRIANGLE_INVERSE.name: "triangle^-1"}[
        name
    ]


def _components(diagram: Diagram) -> list[list[NodeId]]:
    parent = {v: v for v in diagram.nodes}

    def find(v: NodeId) -> NodeId:
        while parent[v] != v:
            v = parent[v]
        return v

    def join(a: NodeId, b: NodeId) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for wire in diagram.wires:
        join(wire.a.node_id, wire.b.node_id)
    for box in diagram.bang_boxes.values():
        scope = sorted(box.node_scope)
        for v in scope[1:]:
            join(scope[0], v)
    groups: dict[NodeId, list[NodeId]] = {}
    for v in sorted(diagram.nodes):
        groups.setdefault(find(v), []).append(v)
    return [groups[root] for root in sorted(groups)]


def _ordered(refs: Sequence[PortRef], component_of: Mapping[NodeId, int]) -> bool:
    seq = [component_of[ref.node_id] for ref in refs]
    return seq == sorted(seq)


def _recognize(diagram: Diagram) -> str | None:
    components = _components(diagram)
    if not components or not diagram.scalar.is_one:
        return None
    structure = _Structure(diagram)
    component_of = {v: i for i, group in enumerate(components) for v in group}
    if not _ordered(diagram.boundary_outputs, component_of):
        return None
    if not _ordered(diagram.boundary_inputs, component_of):
        return None
    names: list[str] = []
    for group in components:
        v = group[0]
        if len(group) > 1 or structure.chains[v] or any(w.a.node_id == v for w in diagram.wires):
            return None
        name = _node_name(diagram, structure, diagram.nodes[v])
        if name is None:
            return None
        names.append(name)
    return " (x) ".join(names)


def recognize(diagram: Diagram) -> str | None:
    """The catalog name of ``diagram``, or None when some component is not one bare generator."""
    checked = _require_diagram(diagram)
    try:
        _require_valid(checked)
        return _recognize(checked)
    except _FOREIGN as exc:
        raise PrinterDomainError(f"cannot recognize diagram: {exc}") from exc


# -- renderings ----------------------------------------------------------------------------


def _source(diagram: Diagram) -> str | None:
    if len(diagram.nodes) != 1 or diagram.wires or not diagram.scalar.is_one:
        return None
    (node,) = diagram.nodes.values()
    if not _is(node, Z_SPIDER.name) or node.num_inputs or not node.num_outputs:
        return None
    if node.phase is not None and not node.phase.is_zero:
        return None
    if any(box.is_node_scope for box in diagram.bang_boxes.values()):
        return None
    dim = diagram.resolve_dim(node.outputs[0].dim)
    if dim.is_concrete:
        dim_text = str(dim.to_int())
    elif dim.to_sympy().is_Symbol and str(dim) != "k":
        dim_text = str(dim)
    else:
        return None
    if not diagram.bang_boxes:
        return f"sum_{{k=0}}^{{{dim_text}-1}} |{', '.join('k' * node.num_outputs)}>"
    structure = _Structure(diagram)
    total = sp.Integer(0)
    for ref in _all_refs(node):
        total = total + _subs(_width(structure, ref), diagram.parameters)
    if total.free_symbols or total < 1:
        return None
    return f"sum_{{k=0}}^{{{dim_text}-1}} |k>^{{{int(total)}}}"


def to_dirac_source(diagram: Diagram) -> str | None:
    """Phase 5 parser source for a single phase-free Z state, else None."""
    checked = _require_diagram(diagram)
    try:
        _require_valid(checked)
        return _source(checked)
    except _FOREIGN as exc:
        raise PrinterDomainError(f"cannot emit parser source: {exc}") from exc


def dirac(diagram: Diagram, *, unicode: bool = False) -> DiracRendering:
    """Every Dirac view of ``diagram``: term, name, text, instance text, parser source."""
    term = dirac_term(diagram)
    name = recognize(diagram)
    formula = render_term(term, unicode=unicode)
    if name is not None and unicode:
        name = name.replace(" (x) ", " ⊗ ")
    text = formula if name is None else f"{name} = {formula}"
    params = dict(diagram.parameters)
    instance = render_term(term, env=params, unicode=unicode) if params else None
    return DiracRendering(term, name, text, instance, to_dirac_source(diagram))


def render_dirac(diagram: Diagram, *, unicode: bool = False, instance: bool = False) -> str:
    """The Dirac text of ``diagram``; with ``instance``, the ``at d = 3: ...`` line follows."""
    rendering = dirac(diagram, unicode=unicode)
    if not instance or rendering.instance_text is None:
        return rendering.text
    env = ", ".join(f"{k} = {v}" for k, v in sorted(diagram.parameters.items()))
    return f"{rendering.text}\nat {env}: {rendering.instance_text}"


# -- graph listing and notation modes --


# -- graph listing -------------------------------------------------------------------------

_LISTING_INDENT = "  "


def _listing_count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _listing_expr(value: object) -> str:
    """``sstr`` of a sympy expression (or ``str`` of anything else), ``**`` shown as ``^``."""
    to_sympy = getattr(value, "to_sympy", None)
    text = sp.sstr(to_sympy()) if callable(to_sympy) else str(value)
    return str(text).replace("**", "^")


def _listing_node(node_id: object) -> str:
    """``n3`` for an int id, ``n'x'`` (the repr) for anything else."""
    if isinstance(node_id, int) and not isinstance(node_id, bool):
        return f"n{int(node_id)}"
    return f"n{node_id!r}"


def _listing_ref(ref: PortRef) -> str:
    side = "out" if ref.direction is Direction.OUTPUT else "in"
    return f"{_listing_node(ref.node_id)}.{side}{ref.index}"


def _listing_ref_key(ref: PortRef) -> tuple[int, int, str, str, int]:
    """``sort_key`` order for int node ids, then any other id by its repr."""
    node_id = ref.node_id
    if isinstance(node_id, int) and not isinstance(node_id, bool):
        return (0, int(node_id), "", ref.direction.value, ref.index)
    return (1, 0, repr(node_id), ref.direction.value, ref.index)


def _listing_wire_key(
    wire: Wire,
) -> tuple[tuple[int, int, str, str, int], tuple[int, int, str, str, int]]:
    a, b = sorted((_listing_ref_key(wire.a), _listing_ref_key(wire.b)))
    return (a, b)


def _listing_wire(wire: Wire) -> str:
    a, b = sorted((wire.a, wire.b), key=_listing_ref_key)
    return f"{_listing_ref(a)} -- {_listing_ref(b)}"


def _listing_refs(refs: Sequence[PortRef]) -> str:
    return "[" + ", ".join(_listing_ref(r) for r in refs) + "]"


def _listing_phase(phase: PhaseVector | None) -> str:
    """``  phase[d]: {1: 1/3} turns``, ``  phase[d]: 0`` for a zero vector, empty for None."""
    if phase is None:
        return ""
    if not phase.entries():
        return f"  phase[{_listing_expr(phase.dim)}]: 0"
    entries = ", ".join(
        f"{index}: {_listing_expr(sp.sstr(p.to_sympy_turns()))}"
        for index, p in sorted(phase.entries().items())
    )
    return f"  phase[{_listing_expr(phase.dim)}]: {{{entries}}} turns"


def _listing_header(kind: str, nodes: int, wires: int, inputs: int, outputs: int) -> str:
    return (
        f"{kind}: {_listing_count(nodes, 'node')}, {_listing_count(wires, 'wire')}, "
        f"{_listing_count(inputs, 'input')} -> {_listing_count(outputs, 'output')}"
    )


def _listing_parameters(parameters: Mapping[str, int]) -> str:
    if not parameters:
        return "parameters: none"
    return "parameters: " + ", ".join(f"{k} = {v}" for k, v in sorted(parameters.items()))


_ForestEntry: TypeAlias = tuple[int, int, object, str | None]


def _listing_forest(ids: Sequence[int], parent_of: Mapping[int, object]) -> list[_ForestEntry]:
    """Depth-first ``(id, depth, parent, flag)`` order, children ascending by id. Roots: no
    parent, or a missing one (flag ``missing``); each parent cycle is cut at its smallest id
    (flag ``cycle``). Iterative."""
    known = set(ids)
    children: dict[int, list[int]] = {i: [] for i in ids}
    for i in sorted(ids):
        parent = parent_of.get(i)
        if parent is not None and parent in known:
            children[parent].append(i)
    order: list[_ForestEntry] = []
    seen: set[int] = set()

    def walk(root: int, flag: str | None) -> None:
        stack = [(root, 0)]
        while stack:
            i, depth = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            order.append((i, depth, parent_of.get(i), flag if i == root else None))
            stack.extend((c, depth + 1) for c in reversed(children[i]) if c not in seen)

    for i in sorted(ids):
        parent = parent_of.get(i)
        if parent is None:
            walk(i, None)
        elif parent not in known:
            walk(i, "missing")
    for i in sorted(ids):
        if i in seen:
            continue
        position: dict[int, int] = {}
        path: list[int] = []
        current = i
        while current not in position:
            position[current] = len(path)
            path.append(current)
            current = parent_of[current]  # type: ignore[assignment]
        walk(min(path[position[current] :]), "cycle")
    return order


def _listing_depths(order: Sequence[_ForestEntry]) -> dict[int, int]:
    return {i: depth for i, depth, _, _ in order}


def render_graph(diagram: Diagram) -> str:
    """The graph listing of ``diagram``: header, scalar, parameters, nodes, wires, boundary,
    bang-box forest. Never fails on a :class:`Diagram`, valid or not."""
    if not isinstance(diagram, Diagram):
        raise PrinterGrammarError(f"render_graph requires a Diagram, got {type(diagram).__name__}")
    boxes: dict[int, BangBox] = {int(b): box for b, box in diagram.bang_boxes.items()}
    order = _listing_forest(sorted(boxes), {i: box.parent for i, box in boxes.items()})
    depth = _listing_depths(order)
    port_boxes: dict[PortRef, list[int]] = {}
    node_boxes: dict[int, list[int]] = {}
    for i, box in boxes.items():
        for ref in box.port_scope:
            port_boxes.setdefault(ref, []).append(i)
        for nid in box.node_scope:
            node_boxes.setdefault(int(nid), []).append(i)

    def box_order(ids: list[int]) -> list[int]:
        return sorted(ids, key=lambda i: (depth.get(i, 0), i))

    lines = [
        _listing_header(
            "graph",
            len(diagram.nodes),
            len(diagram.wires),
            len(diagram.boundary_inputs),
            len(diagram.boundary_outputs),
        ),
        f"scalar: {_listing_expr(diagram.scalar)}",
        _listing_parameters(diagram.parameters),
    ]
    lines.append("nodes:" if diagram.nodes else "nodes: none")
    for nid in sorted(diagram.nodes, key=int):
        node = diagram.nodes[nid]
        sides = []
        for label, ports, direction in (
            ("in", node.inputs, Direction.INPUT),
            ("out", node.outputs, Direction.OUTPUT),
        ):
            texts = []
            for index, port in enumerate(ports):
                ref = PortRef(nid, direction, index)
                marks = "".join(f"!{b}" for b in box_order(port_boxes.get(ref, [])))
                texts.append(_listing_expr(port.dim) + marks)
            sides.append(f"{label}: [{', '.join(texts)}]")
        line = f"{_LISTING_INDENT}n{int(nid)}  {node.generator_type.name}  " + "  ".join(sides)
        line += _listing_phase(node.phase)
        chain = box_order(node_boxes.get(int(nid), []))
        if chain:
            line += "  boxes: " + " > ".join(f"!{b}" for b in chain)
        lines.append(line)
    lines.append("wires:" if diagram.wires else "wires: none")
    for wire in sorted(diagram.wires, key=_listing_wire_key):
        lines.append(_LISTING_INDENT + _listing_wire(wire))
    lines.append(f"inputs: {_listing_refs(diagram.boundary_inputs)}")
    lines.append(f"outputs: {_listing_refs(diagram.boundary_outputs)}")
    lines.append("bang boxes:" if boxes else "bang boxes: none")
    for i, level, parent, flag in order:
        box = boxes[i]
        if box.node_scope:
            scope = "nodes: " + ", ".join(_listing_node(n) for n in sorted(box.node_scope))
        else:
            scope = "ports: " + ", ".join(
                _listing_ref(r) for r in sorted(box.port_scope, key=_listing_ref_key)
            )
        line = f"{_LISTING_INDENT * (level + 1)}!{i}  x {_listing_expr(box.multiplicity)}  {scope}"
        if flag is not None:
            line += f"  parent: !{parent} ({flag})"
        lines.append(line)
    return _listing_join([lines[0], *(_LISTING_INDENT + line for line in lines[1:])])


def _listing_join(lines: Sequence[str]) -> str:
    return "\n".join(line.rstrip() for line in lines)


def _listing_item(item: PortRef | Bundle) -> str:
    if isinstance(item, Bundle):
        inner = ", ".join(_listing_item(i) for i in item.items)
        return f"{{s{int(item.scale)}: {inner}}}"
    return _listing_ref(item)


def _listing_items(items: Sequence[PortRef | Bundle]) -> str:
    return "[" + ", ".join(_listing_item(i) for i in items) + "]"


def _listing_sheet_port(port: SheetPort) -> str:
    text = _listing_expr(port.dim)
    return text if port.fan is None else f"{text}~s{int(port.fan)}"


def render_scalable_graph(s: ScalableDiagram) -> str:
    """The scalable listing of ``s``: header, scalar, parameters, scale forest, nodes, wires,
    boundary with bundles."""
    if not isinstance(s, ScalableDiagram):
        raise PrinterGrammarError(
            f"render_scalable_graph requires a ScalableDiagram, got {type(s).__name__}"
        )
    scales: dict[int, Scale] = {int(scale.id): scale for scale in s.scales}
    order = _listing_forest(
        sorted(scales),
        {i: (None if sc.parent is None else int(sc.parent)) for i, sc in scales.items()},
    )
    lines = [
        _listing_header("scalable", len(s.nodes), len(s.wires), len(s.inputs), len(s.outputs)),
        f"scalar: {_listing_expr(s.scalar)}",
        _listing_parameters(s.parameters),
    ]
    lines.append("scales:" if scales else "scales: none")
    for i, level, parent, flag in order:
        scale = scales[i]
        line = (
            f"{_LISTING_INDENT * (level + 1)}s{i}  {scale.kind.name} x "
            f"{_listing_expr(scale.multiplicity)}"
        )
        if flag is not None:
            line += f"  parent: s{parent} ({flag})"
        lines.append(line)
    lines.append("nodes:" if s.nodes else "nodes: none")
    for node in sorted(s.nodes, key=lambda n: int(n.id)):
        ins = ", ".join(_listing_sheet_port(p) for p in node.inputs)
        outs = ", ".join(_listing_sheet_port(p) for p in node.outputs)
        line = (
            f"{_LISTING_INDENT}n{int(node.id)}  {node.generator_type.name}  "
            f"in: [{ins}]  out: [{outs}]"
        )
        line += _listing_phase(node.phase)
        if node.scale is not None:
            line += f"  scale: s{int(node.scale)}"
        lines.append(line)
    lines.append("wires:" if s.wires else "wires: none")
    for wire in sorted(s.wires, key=_listing_wire_key):
        lines.append(_LISTING_INDENT + _listing_wire(wire))
    lines.append(f"inputs: {_listing_items(s.inputs)}")
    lines.append(f"outputs: {_listing_items(s.outputs)}")
    return _listing_join([lines[0], *(_LISTING_INDENT + line for line in lines[1:])])


# -- notation modes and dispatch -----------------------------------------------------------


def _mode_dirac_lines(diagram: Diagram, unicode: bool) -> list[str]:
    """The Dirac block: the rendering line(s), then ``at d = 3: ...`` when parameters are
    bound; ``dirac: unavailable (<reason>)`` on a domain error."""
    try:
        text = render_dirac(diagram, unicode=unicode, instance=True)
    except PrinterDomainError as exc:
        return [f"dirac: unavailable ({_mode_one_line(exc)})"]
    return text.splitlines() or [""]


def _mode_scalable_dirac_lines(s: ScalableDiagram, unicode: bool) -> list[str]:
    try:
        diagram = from_scalable(s)
    except _FOREIGN as exc:
        return [f"dirac: unavailable ({_mode_one_line(exc)})"]
    return _mode_dirac_lines(diagram, unicode)


def _mode_diagram(diagram: Diagram, mode: NotationMode, unicode: bool) -> str:
    dirac_lines = _mode_dirac_lines(diagram, unicode)
    if mode is NotationMode.ZX:
        return _listing_join([render_graph(diagram), *dirac_lines])
    return _listing_join(dirac_lines)


def _mode_scalable(s: ScalableDiagram, mode: NotationMode, unicode: bool) -> str:
    dirac_lines = _mode_scalable_dirac_lines(s, unicode)
    if mode is NotationMode.ZX:
        return _listing_join([render_scalable_graph(s), *dirac_lines])
    return _listing_join(dirac_lines)


def _mode_one_line(exc: BaseException) -> str:
    """``exc``'s message with every whitespace run collapsed to one space."""
    return " ".join(str(exc).split())


def _mode_indent(text: str, levels: int = 1) -> str:
    pad = _LISTING_INDENT * levels
    return _listing_join([pad + line if line else line for line in text.split("\n")])


def _mode_check(mode: object, unicode: object) -> NotationMode:
    if not isinstance(mode, NotationMode):
        raise PrinterGrammarError(f"mode must be a NotationMode, got {mode!r}")
    if not isinstance(unicode, bool):
        raise PrinterGrammarError(f"unicode must be a bool, got {unicode!r}")
    return mode


def _mode_step_line(index: int | None, step: RewriteStep) -> str:
    """``step i: <rule> (consumed n.., new n.., scalar s)``."""
    consumed = ", ".join(f"n{int(n)}" for n in sorted(step.consumed_node_ids, key=int)) or "none"
    new = ", ".join(f"n{int(n)}" for n in sorted(step.new_node_ids, key=int)) or "none"
    label = "step" if index is None else f"step {index}"
    return (
        f"{label}: {step.rule_name} (consumed {consumed}, new {new}, "
        f"scalar {_listing_expr(step.scalar_introduced)})"
    )


def _mode_state(
    label: str, state: Diagram | str, mode: NotationMode, unicode: bool, note: str = ""
) -> str:
    """``<label>:`` with the rendered diagram indented, or ``<label>: unavailable (<reason>)``
    for a reason string; ``note`` follows the colon."""
    if isinstance(state, str):
        return f"{label}: unavailable ({state})"
    head = f"{label}: {note}" if note else f"{label}:"
    return head + "\n" + _mode_indent(_mode_diagram(state, mode, unicode))


def _mode_replay(derivation: Derivation) -> tuple[list[Diagram | str], str | None]:
    """Every state of ``derivation`` by re-applying its steps from ``initial`` (each replayed
    step must equal its record), a reason string for each intermediate state not reached,
    and the last state the recorded ``final``; with the first failure's reason, or None."""
    working = derivation.initial.copy()
    states: list[Diagram | str] = [working]
    failure: str | None = None
    for index, step in enumerate(derivation.steps, 1):
        if failure is not None:
            states.append(f"not reached: {failure}")
            continue
        prefix = f"step {index} ({step.rule_name}) failed"
        try:
            result = apply(working, lookup_rule(step.rule_name), step.match)
        except Exception as exc:  # noqa: BLE001
            failure = f"{prefix}: {type(exc).__name__}: {_mode_one_line(exc)}"
        else:
            if result.step == step:
                working = result.diagram
                states.append(working)
                continue
            failure = f"{prefix}: the replayed step differs from the record"
        states.append(failure)
    if derivation.steps:
        states[-1] = derivation.final
    return states, failure


def _mode_derivation(derivation: Derivation, mode: NotationMode, unicode: bool) -> str:
    """Header, induction claim, every state and step, then the children, nested."""
    header = f"derivation: {derivation.kind.value}, {_listing_count(len(derivation.steps), 'step')}"
    if derivation.label:
        header += f", label {derivation.label!r}"
    blocks: list[str] = []
    claim = derivation.induction
    if claim is not None:
        held = ", ".join(claim.held_symbolic) or "none"
        text = f"induction: {claim.symbol}, base {claim.base_value}, held symbolic: {held}"
        if claim.step_discharge:
            text += f", step discharge: {claim.step_discharge}"
        blocks.append(text)
    states, failure = _mode_replay(derivation)
    blocks.append(_mode_state("state 0", states[0], mode, unicode))
    last = len(derivation.steps)
    for index, step in enumerate(derivation.steps, 1):
        blocks.append(_mode_step_line(index, step))
        note = f"recorded final (replay {failure})" if index == last and failure else ""
        blocks.append(_mode_state(f"state {index}", states[index], mode, unicode, note))
    if (
        not derivation.steps
        and not compare_structure(derivation.initial, derivation.final).identical
    ):
        blocks.append(_mode_state("final", derivation.final, mode, unicode))
    if derivation.children:
        names = (
            ("base", "step")
            if derivation.kind is DerivationKind.INDUCTION and len(derivation.children) == 2
            else tuple(f"child {i}" for i in range(len(derivation.children)))
        )
        blocks.append("children:")
        for name, child in zip(names, derivation.children, strict=True):
            body = _mode_indent(_mode_derivation(child, mode, unicode))
            blocks.append(_mode_indent(f"{name}:\n{body}"))
    return header + "\n" + _mode_indent("\n".join(blocks))


def _mode_certificate(cert: Certificate, mode: NotationMode, unicode: bool) -> str:
    body = _mode_indent(_mode_derivation(cert.derivation, mode, unicode))
    return f"certificate: {cert.check_method.value}\n{body}"


def _mode_proof(proof: ProofCertificate, mode: NotationMode, unicode: bool) -> str:
    lines = [
        (
            f"proof: {_listing_count(proof.length, 'step')} "
            f"(forward {len(proof.forward.steps)}, backward {len(proof.backward.steps)})"
        ),
        f"{_LISTING_INDENT}moves: {', '.join(proof.moves) or 'none'}",
        f"{_LISTING_INDENT}backward moves: {', '.join(proof.backward_moves) or 'none'}",
        f"{_LISTING_INDENT}forward:",
        _mode_indent(_mode_certificate(proof.forward, mode, unicode), 2),
        f"{_LISTING_INDENT}backward:",
        _mode_indent(_mode_certificate(proof.backward, mode, unicode), 2),
    ]
    return "\n".join(lines)


def _mode_induction(proof: InductionProof, mode: NotationMode, unicode: bool) -> str:
    """Header, then the base proof, the expose certificate and the step proof, nested."""
    region = ", ".join(_listing_node(n) for n in sorted(proof.region)) or "none"
    lines = [
        (
            f"induction proof: over {proof.index}, step symbol {proof.step_symbol}, "
            f"side {proof.side.value}, region {region}, {_listing_count(proof.length, 'step')}"
        ),
        f"{_LISTING_INDENT}start:",
        _mode_indent(_mode_diagram(proof.start, mode, unicode), 2),
        f"{_LISTING_INDENT}goal:",
        _mode_indent(_mode_diagram(proof.goal, mode, unicode), 2),
    ]
    for name, piece in (("base", proof.base), ("expose", proof.expose), ("step", proof.step)):
        lines.append(f"{_LISTING_INDENT}{name}:")
        lines.append(_mode_indent(_mode_proof_like(piece, mode, unicode), 2))
    return "\n".join(lines)


def _mode_proof_like(
    piece: ProofCertificate | InductionProof | Certificate, mode: NotationMode, unicode: bool
) -> str:
    if isinstance(piece, InductionProof):
        return _mode_induction(piece, mode, unicode)
    if isinstance(piece, ProofCertificate):
        return _mode_proof(piece, mode, unicode)
    return _mode_certificate(piece, mode, unicode)


def _mode_results(
    initial: Diagram, results: Sequence[RewriteResult], mode: NotationMode, unicode: bool
) -> list[str]:
    blocks = [_mode_state("state 0", initial, mode, unicode)]
    for index, result in enumerate(results, 1):
        blocks.append(_mode_step_line(index, result.step))
        blocks.append(_mode_state(f"state {index}", result.diagram, mode, unicode))
    return blocks


def _mode_normal_form(nf: NormalForm, mode: NotationMode, unicode: bool) -> str:
    lines = [
        f"normal form: {_listing_count(len(nf.results), 'step')}, {nf.outcome.stop_reason.value}",
        f"{_LISTING_INDENT}source:",
        _mode_indent(_mode_diagram(nf.source, mode, unicode), 2),
        f"{_LISTING_INDENT}derivation:" if nf.results else f"{_LISTING_INDENT}derivation: none",
        *(_mode_indent(b, 2) for b in _mode_results(nf.source, nf.results, mode, unicode)[1:]),
        f"{_LISTING_INDENT}reduced:",
        _mode_indent(_mode_diagram(nf.diagram, mode, unicode), 2),
    ]
    return "\n".join(lines)


def _mode_argument(value: object) -> str:
    if isinstance(value, Wire):
        return _listing_wire(value)
    if isinstance(value, PortRef):
        return _listing_ref(value)
    if isinstance(value, (tuple, list, frozenset, set)):
        texts = [_mode_argument(v) for v in value]
        if isinstance(value, (frozenset, set)):
            texts.sort()
        return "(" + ", ".join(texts) + ")"
    return _listing_expr(value)


def _mode_sheet_step(step: SheetStep, mode: NotationMode, unicode: bool) -> str:
    arguments = ", ".join(f"{k}={_mode_argument(v)}" for k, v in step.arguments)
    lines = [
        f"sheet step: {step.operation}({arguments})",
        f"{_LISTING_INDENT}before:",
        _mode_indent(_mode_scalable(step.before, mode, unicode), 2),
        f"{_LISTING_INDENT}after:",
        _mode_indent(_mode_scalable(step.after, mode, unicode), 2),
    ]
    return "\n".join(lines)


def render(state: object, mode: NotationMode = DEFAULT_MODE, *, unicode: bool = False) -> str:
    """``state`` in ``mode``: DIRAC emits Dirac blocks only, ZX the graph listing then the Dirac
    block. Accepts Diagram, ScalableDiagram, NormalForm, RewriteResult, Derivation,
    Certificate, ProofCertificate, InductionProof and SheetStep."""
    mode = _mode_check(mode, unicode)
    if isinstance(state, Diagram):
        return _mode_diagram(state, mode, unicode)
    if isinstance(state, ScalableDiagram):
        return _mode_scalable(state, mode, unicode)
    if isinstance(state, NormalForm):
        return _mode_normal_form(state, mode, unicode)
    if isinstance(state, RewriteResult):
        body = _mode_indent(_mode_diagram(state.diagram, mode, unicode))
        return f"{_mode_step_line(None, state.step)}\nresult:\n{body}"
    if isinstance(state, Derivation):
        return _mode_derivation(state, mode, unicode)
    if isinstance(state, Certificate):
        return _mode_certificate(state, mode, unicode)
    if isinstance(state, ProofCertificate):
        return _mode_proof(state, mode, unicode)
    if isinstance(state, InductionProof):
        return _mode_induction(state, mode, unicode)
    if isinstance(state, SheetStep):
        return _mode_sheet_step(state, mode, unicode)
    raise PrinterGrammarError(f"render cannot print a {type(state).__name__}")


def render_derivation(
    initial: Diagram,
    results: Sequence[RewriteResult],
    mode: NotationMode = DEFAULT_MODE,
    *,
    unicode: bool = False,
) -> str:
    """``state 0:``, then ``step i: ...`` and ``state i:`` per result, each state in ``mode``."""
    mode = _mode_check(mode, unicode)
    if not isinstance(initial, Diagram):
        raise PrinterGrammarError(f"initial must be a Diagram, got {type(initial).__name__}")
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise PrinterGrammarError(f"results must be a Sequence, got {type(results).__name__}")
    if not all(isinstance(r, RewriteResult) for r in results):
        raise PrinterGrammarError("every element of results must be a RewriteResult")
    return "\n".join(_mode_results(initial, results, mode, unicode))


def render_certificate(
    cert: Certificate, mode: NotationMode = DEFAULT_MODE, *, unicode: bool = False
) -> str:
    """The certificate's derivation tree with every state recovered by re-applying its steps;
    a failed re-application renders ``state i: unavailable (<reason>)``."""
    mode = _mode_check(mode, unicode)
    if not isinstance(cert, Certificate):
        raise PrinterGrammarError(f"cert must be a Certificate, got {type(cert).__name__}")
    return _mode_certificate(cert, mode, unicode)


def side_by_side(left: str, right: str, *, gap: int = 4) -> str:
    """``left`` and ``right`` as two columns: left padded to its widest line in terminal
    columns, then ``gap`` spaces; trailing blanks stripped."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise PrinterGrammarError("side_by_side requires two str")
    if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
        raise PrinterGrammarError(f"gap must be a non-negative int, got {gap!r}")
    left_lines = left.splitlines() or [""]
    right_lines = right.splitlines() or [""]
    widths = [_mode_width(line) for line in left_lines]
    width = max(widths)
    rows = max(len(left_lines), len(right_lines))
    left_lines += [""] * (rows - len(left_lines))
    widths += [0] * (rows - len(widths))
    right_lines += [""] * (rows - len(right_lines))
    return _listing_join(
        [
            f"{a}{' ' * (width - w + gap)}{b}"
            for a, w, b in zip(left_lines, widths, right_lines, strict=True)
        ]
    )


def _mode_width(line: str) -> int:
    """Terminal columns of ``line``: 2 per wide or fullwidth char, 0 per combining or format
    char, 1 otherwise."""
    width = 0
    for char in line:
        if unicodedata.combining(char) or unicodedata.category(char) in ("Mn", "Me", "Cf"):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _mode_detail_target(state: object) -> Diagram | ScalableDiagram:
    """The diagram ``render_detail`` shows: the state itself, or a composite's resulting one."""
    if isinstance(state, (Diagram, ScalableDiagram)):
        return state
    if isinstance(state, (RewriteResult, NormalForm)):
        return state.diagram
    if isinstance(state, (Derivation, Certificate)):
        return state.final
    if isinstance(state, (ProofCertificate, InductionProof)):
        return state.goal
    if isinstance(state, SheetStep):
        return state.after
    raise PrinterGrammarError(f"render_detail cannot print a {type(state).__name__}")


def render_detail(state: object, *, unicode: bool = False) -> str:
    """The graph listing and the Dirac block side by side, regardless of mode."""
    _mode_check(DEFAULT_MODE, unicode)
    target = _mode_detail_target(state)
    if isinstance(target, ScalableDiagram):
        listing = render_scalable_graph(target)
        dirac_lines = _mode_scalable_dirac_lines(target, unicode)
    else:
        listing = render_graph(target)
        dirac_lines = _mode_dirac_lines(target, unicode)
    return side_by_side(listing, "\n".join(dirac_lines))
