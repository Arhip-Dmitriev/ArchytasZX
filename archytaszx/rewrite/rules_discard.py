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

"""The discard rules of [CJPV] for the engine's generators, as template rules.

[CJPV] (archytaszx/papers/carette2019mixed/AXIOMS.txt) adds to a complete pure language one
equation ``G . V = G`` per isometry ``V`` of a spanning set, and for the qubit ZX-calculus
takes the set (e^{i alpha}, |0>, H, R_Z(alpha), CNot). The rules here are those equations for
the qudit analogues under the mapping of COMPLETENESS.txt, section 2 (H is F, |0> the pink dot
state, R_Z(alpha) the Z spider (1 -> 1) with phase alpha, CNot the Z (1 -> 2) and pink (2 -> 1)
pair), with the Z spider (1 -> n) for n >= 1, the [P2] connectives B and S, the closed
discard pair, and the derived merge of two discarded legs of one Z spider. The global phase
equation e^{i alpha} = 1 is the mixed scalar convention: a mixed diagram's scalar is compared
by modulus.

A rule's scalar is a pure scalar ``c``: it multiplies the CPM by ``|c|^2``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    DISCARD,
    Z_SPIDER,
)
from archytaszx.diagram.graph import Diagram, NodeId
from archytaszx.rewrite.match_template import (
    Arities,
    Bindings,
    Condition,
    Derived,
    Side,
    Template,
    TemplateAxiom,
    Var,
)
from archytaszx.rewrite.rule import Rule
from archytaszx.rewrite.rules_axioms import TB, _legs, _partners, fixed_side, i, o

# -- arity enumerators --------------------------------------------------------------------------


def _neighbour(diagram: Diagram, anchor: NodeId) -> NodeId | None:
    """The node the anchor's one leg is wired to, or None."""
    partners = _partners(diagram)
    for ref in _legs(diagram, anchor):
        far = partners.get(ref)
        if far is not None:
            return far.node_id
    return None


def _spider_arities(diagram: Diagram, anchor: NodeId, _dim: Dim) -> Iterator[Arities]:
    """``n`` = the leg count less one of the Z spider the anchor discard is wired to."""
    if diagram.nodes[anchor].generator_type.name != DISCARD.name:
        return
    far = _neighbour(diagram, anchor)
    if far is None or diagram.nodes[far].generator_type.name != Z_SPIDER.name:
        return
    node = diagram.nodes[far]
    legs = node.num_inputs + node.num_outputs
    if legs >= 2:
        yield {"n": legs - 1}


def _connective_arities(name: str):  # type: ignore[no-untyped-def]
    """``s`` and ``t`` of the B or S node the anchor discard is wired to, when concrete."""

    def enumerate_(diagram: Diagram, anchor: NodeId, _dim: Dim) -> Iterator[Arities]:
        if diagram.nodes[anchor].generator_type.name != DISCARD.name:
            return
        far = _neighbour(diagram, anchor)
        if far is None or diagram.nodes[far].generator_type.name != name:
            return
        node = diagram.nodes[far]
        pair = node.inputs if name == DIM_BINDER.name else node.outputs
        s, t = (diagram.resolve_dim(p.dim) for p in pair)
        if s.is_concrete and t.is_concrete:
            yield {"s": s.to_int(), "t": t.to_int()}

    return enumerate_


def _ground_dims(diagram: Diagram) -> list[int]:
    out: list[int] = []
    for node in diagram.nodes.values():
        if node.generator_type.name == DISCARD.name:
            dim = diagram.resolve_dim((node.inputs or node.outputs)[0].dim)
            if dim.is_concrete:
                out.append(dim.to_int())
    return sorted(set(out))


def _split_arities(diagram: Diagram, anchor: NodeId, _dim: Dim) -> Iterator[Arities]:
    """Every ``s, t >= 2`` with ``s * t`` the concrete dimension of the anchor discard."""
    node = diagram.nodes[anchor]
    if node.generator_type.name != DISCARD.name:
        return
    dim = diagram.resolve_dim((node.inputs or node.outputs)[0].dim)
    if not dim.is_concrete:
        return
    total = dim.to_int()
    for s in range(2, total):
        if total % s == 0 and total // s >= 2:
            yield {"s": s, "t": total // s}


def _pair_arities(diagram: Diagram, anchor: NodeId, _dim: Dim) -> Iterator[Arities]:
    """``s`` the anchor discard's concrete dimension, ``t`` that of any discard."""
    node = diagram.nodes[anchor]
    if node.generator_type.name != DISCARD.name:
        return
    dim = diagram.resolve_dim((node.inputs or node.outputs)[0].dim)
    if not dim.is_concrete:
        return
    for t in _ground_dims(diagram):
        yield {"s": dim.to_int(), "t": t}


# -- templates ----------------------------------------------------------------------------------


def _ground(t: TB, key: str | None = None, dim: Dim | None = None) -> str:
    return t.node(DISCARD, 1, 0, key=key, dim=dim)


def _ground_term(name: str) -> Template:
    t = TB()
    g = _ground(t, "g")
    t.term(name, i(g))
    return t.done()


def _zero_left() -> Template:
    t = TB()
    g = _ground(t, "g")
    p = t.pink(0, 1, key="p")
    t.wire(o(p), i(g))
    return t.done()


def _empty() -> Template:
    return TB().done()


def _fourier_left() -> Template:
    t = TB()
    g = _ground(t, "g")
    f = t.f("f")
    t.wire(o(f), i(g))
    t.term("in", i(f))
    return t.done()


def _spider_left(_dim: Dim, a: Arities) -> Template:
    n = a["n"]
    t = TB()
    grounds = [_ground(t, f"g{k}") for k in range(n)]
    z = t.z(1, n, Var("a"), "z")
    for k, g in enumerate(grounds):
        t.wire(o(z, k), i(g))
    t.term("in", i(z))
    return t.done()


def _cx_left() -> Template:
    t = TB()
    g1 = _ground(t, "g1")
    c = t.z(1, 2, key="c")
    x = t.pink(2, 1, key="x")
    g2 = _ground(t, "g2")
    t.wire(o(c, 0), i(g1))
    t.wire(o(c, 1), i(x, 0))
    t.wire(o(x), i(g2))
    t.term("ctl", i(c))
    t.term("tgt", i(x, 1))
    return t.done()


def _cx_right() -> Template:
    t = TB()
    a = _ground(t, "a")
    b = _ground(t, "b")
    t.term("ctl", i(a))
    t.term("tgt", i(b))
    return t.done()


def _kept(_dim: Dim, bindings: Bindings) -> PhaseVector | None:
    vector = bindings["a"]
    assert isinstance(vector, PhaseVector)
    return None if vector.is_zero else vector


def _merge_left() -> Template:
    t = TB()
    g1 = _ground(t, "g1")
    z = t.z(0, 2, Var("a"), "z", open=True)
    g2 = _ground(t, "g2")
    t.wire(o(z, 0), i(g1))
    t.wire(o(z, 1), i(g2))
    return t.done()


def _merge_right() -> Template:
    t = TB()
    z = t.z(0, 1, Derived(("a",), _kept), "z")
    g = _ground(t, "g")
    t.wire(o(z), i(g))
    t.rest = {"z": "z"}
    return t.done()


def _loop_left() -> Template:
    t = TB()
    a = _ground(t, "a")
    b = _ground(t, "b")
    t.wire(i(a), i(b))
    return t.done()


def _binder_left(_dim: Dim, a: Arities) -> Template:
    s, u = a["s"], a["t"]
    t = TB()
    g = _ground(t, "g", Dim.concrete(s * u))
    b = t.conn(DIM_BINDER, ((s, u), (s * u,)), key="b")
    t.wire(o(b), i(g))
    t.term("a", i(b, 0))
    t.term("b", i(b, 1))
    return t.done()


def _binder_right(_dim: Dim, a: Arities) -> Template:
    s, u = a["s"], a["t"]
    t = TB()
    ga = _ground(t, "ga", Dim.concrete(s))
    gb = _ground(t, "gb", Dim.concrete(u))
    t.term("a", i(ga))
    t.term("b", i(gb))
    return t.done()


def _splitter_left(_dim: Dim, a: Arities) -> Template:
    s, u = a["s"], a["t"]
    t = TB()
    ga = _ground(t, "ga", Dim.concrete(s))
    sp_ = t.conn(DIM_SPLITTER, ((s * u,), (s, u)), key="s")
    gb = _ground(t, "gb", Dim.concrete(u))
    t.wire(o(sp_, 0), i(ga))
    t.wire(o(sp_, 1), i(gb))
    t.term("in", i(sp_))
    return t.done()


def _splitter_right(_dim: Dim, a: Arities) -> Template:
    s, u = a["s"], a["t"]
    t = TB()
    g = _ground(t, "g", Dim.concrete(s * u))
    t.term("in", i(g))
    return t.done()


def _unweighted(_dim: Dim, bindings: Bindings) -> bool:
    vector = bindings.get("a")
    return not getattr(vector, "is_weighted", False)


def _sqrt_dim(dim: Dim, _b: Bindings) -> Scalar:
    return Scalar.dim_power(dim, 1, 2)


DISCARD_ZERO = TemplateAxiom(
    "discard_zero",
    fixed_side(_zero_left),
    fixed_side(_empty),
    description="[CJPV] ket-0: the discarded pink dot state |0> is the empty diagram.",
)

DISCARD_FOURIER = TemplateAxiom(
    "discard_fourier",
    fixed_side(_fourier_left),
    fixed_side(lambda: _ground_term("in")),
    description="[CJPV] H: a discard after F is a discard.",
)

DISCARD_SPIDER = TemplateAxiom(
    "discard_spider",
    Side(_spider_left, _spider_arities),
    fixed_side(lambda: _ground_term("in")),
    conditions=(
        Condition(
            "unit_modulus",
            "the spider carries phases only, no weights",
            _unweighted,
        ),
    ),
    description=(
        "[CJPV] R_Z(alpha), for the isometry Z (1 -> n), n >= 1: its n outputs discarded is a "
        "discard on its input."
    ),
    quantifiers=("n",),
)


def _phase_gate_left() -> Template:
    t = TB()
    g = _ground(t, "g")
    z = t.z(1, 1, Var("a"), "z")
    t.wire(o(z), i(g))
    t.term("in", i(z))
    return t.done()


DISCARD_PHASE = TemplateAxiom(
    "discard_phase",
    fixed_side(_phase_gate_left),
    fixed_side(lambda: _ground_term("in")),
    conditions=(
        Condition("unit_modulus", "the spider carries phases only, no weights", _unweighted),
    ),
    description="[CJPV] R_Z(alpha): a discard after the phase gate Z (1 -> 1) is a discard.",
    converse_free={"a": lambda dim, _b: PhaseVector(dim)},
)

DISCARD_CX = TemplateAxiom(
    "discard_cx",
    fixed_side(_cx_left),
    fixed_side(_cx_right),
    description="[CJPV] CNot: both outputs of the qudit CX (Z (1 -> 2), pink (2 -> 1)) discarded.",
)

DISCARD_MERGE = TemplateAxiom(
    "discard_merge",
    fixed_side(_merge_left),
    fixed_side(_merge_right),
    description=(
        "Derived: two discarded legs of one Z spider are one discarded leg (unfuse a "
        "phaseless Z (1 -> 2) onto them, then discard_spider)."
    ),
)

DISCARD_LOOP = TemplateAxiom(
    "discard_loop",
    fixed_side(_loop_left),
    fixed_side(_empty),
    scalar=_sqrt_dim,
    description="Two discards joined by a wire: the trace of the identity, CPM scalar d.",
)

DISCARD_BINDER = TemplateAxiom(
    "discard_binder",
    Side(_binder_left, _connective_arities(DIM_BINDER.name)),
    Side(_binder_right),
    concrete_only=True,
    description="[CJPV] for the [P2] unitary B (s, t): a discard after it discards both inputs.",
    quantifiers=("s", "t"),
)

DISCARD_SPLITTER = TemplateAxiom(
    "discard_splitter",
    Side(_splitter_left, _connective_arities(DIM_SPLITTER.name)),
    Side(_splitter_right),
    concrete_only=True,
    description="[CJPV] for the [P2] unitary S (s, t): both outputs discarded is one discard.",
    quantifiers=("s", "t"),
)

DISCARD_AXIOMS: tuple[TemplateAxiom, ...] = (
    DISCARD_ZERO,
    DISCARD_FOURIER,
    DISCARD_PHASE,
    DISCARD_SPIDER,
    DISCARD_CX,
    DISCARD_MERGE,
    DISCARD_LOOP,
    DISCARD_BINDER,
    DISCARD_SPLITTER,
)
"""Every discard rule, left to right."""

DISCARD_CONVERSES: tuple[TemplateAxiom, ...] = (
    DISCARD_ZERO.converse("discard_zero_converse"),
    DISCARD_FOURIER.converse("discard_fourier_converse"),
    DISCARD_PHASE.converse("discard_phase_converse"),
    DISCARD_CX.converse("discard_cx_converse"),
    TemplateAxiom(
        "discard_binder_converse",
        Side(_binder_right, _pair_arities),
        Side(_binder_left),
        concrete_only=True,
        description=f"{DISCARD_BINDER.description} Read right to left.",
        quantifiers=("s", "t"),
    ),
    TemplateAxiom(
        "discard_splitter_converse",
        Side(_splitter_right, _split_arities),
        Side(_splitter_left),
        concrete_only=True,
        description=f"{DISCARD_SPLITTER.description} Read right to left.",
        quantifiers=("s", "t"),
    ),
)
"""The converses of the spanning-set rules: each inserts an isometry before a discard."""


def _rules() -> Mapping[str, Rule]:
    from archytaszx.rewrite.rules_library import template_rule

    return {axiom.name: template_rule(axiom) for axiom in (*DISCARD_AXIOMS, *DISCARD_CONVERSES)}


MIXED_RULES: Mapping[str, Rule] = _rules()
"""Every discard rule by name; they match only diagrams with a discard."""
