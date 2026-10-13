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

"""Phase 18 Form 3: Dirac expressions through ``parse_dirac`` and ``parse``.

Legacy subset corpus against ``parse_dirac_source``, ket sums, catalog atoms, feed and
tensor composition, numeric-oracle semantics, and the ParseError-only error sweep.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import BangBox, Mult
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import validate
from archytaszx.repl import printer as P
from archytaszx.repl.parser import (
    DiracDomainError,
    DiracError,
    DiracGrammarError,
    InputForm,
    ParsedDiagram,
    ParseDomainError,
    ParseError,
    ParseGrammarError,
    parse,
    parse_dirac,
    parse_dirac_source,
)
from archytaszx.semantics.check import compare

from .helpers import bounded, build_ghz_with_copy

D = Dim("d")
GRAMMAR = ParseGrammarError
DOMAIN = ParseDomainError
DS = (3, 4)
THETA = sp.Rational(1, 5)


def out(node: int, i: int) -> PortRef:
    return PortRef(NodeId(node), Direction.OUTPUT, i)


def inp(node: int, i: int) -> PortRef:
    return PortRef(NodeId(node), Direction.INPUT, i)


def fields(d: Diagram) -> tuple[object, ...]:
    """Every observable field of a diagram, for identity comparison."""
    return (
        dict(d.nodes),
        d.wires,
        d.boundary_inputs,
        d.boundary_outputs,
        dict(d.bang_boxes),
        dict(d.parameters),
        d.scalar,
    )


def bound(d: Diagram, box: BangBox) -> int:
    """A box multiplicity evaluated in the diagram's parameter environment."""
    expr = box.multiplicity.to_sympy()
    return int(expr.subs({s: d.parameters[s.name] for s in expr.free_symbols}))


def dim_value(d: Diagram, dim: Dim) -> int:
    """A port dimension evaluated in the diagram's parameter environment."""
    return d.resolve_dim(dim).to_int()


def all_dims(d: Diagram) -> set[Dim]:
    """Every port dimension in the diagram."""
    return {p.dim for n in d.nodes.values() for p in (*n.inputs, *n.outputs)}


def ok(source: str) -> Diagram:
    """``parse_dirac`` of ``source``, checked valid."""
    diagram = parse_dirac(source)
    assert isinstance(diagram, Diagram)
    assert validate(diagram).is_valid, validate(diagram)
    return diagram


def assert_oracle(a: Diagram, b: Diagram, envs: Iterable[Mapping[str, object]]) -> None:
    """``a`` and ``b`` are exactly equal at every env."""
    for env in envs:
        result = compare(a, b, env)  # type: ignore[arg-type]
        assert result.matched, (env, result.reason)


def at_d(**extra: object) -> list[dict[str, object]]:
    """One env per concrete ``d`` in ``DS``, with ``extra`` bindings."""
    return [{"d": v, **extra} for v in DS]


# -- errors --------------------------------------------------------------------------------


def test_error_classes_are_aliases() -> None:
    assert DiracError is ParseError
    assert DiracGrammarError is ParseGrammarError
    assert DiracDomainError is ParseDomainError
    assert issubclass(ParseGrammarError, ParseError)
    assert issubclass(ParseDomainError, ParseError)


# -- legacy subset -------------------------------------------------------------------------

_DIMS = (
    "d",
    "D",
    "dim",
    "_x",
    "k2",
    "kd",
    "literal",
    "n",
    "dé",
    "3",
    "07",
    "1",
    "100",
    "literal 3",
    "literal 07",
    "literal  2",
)
_BODIES = ("k", "k,k", "kk", "k,,k", " k , k ", "k, k, k", "kkk", ",k", "k,", "k k")
_POWERS = (
    *(f"^{{{i}}}" for i in range(1, 10)),
    "^{\\otimes 3}",
    "^{⊗3}",
    "^{\\otimes  2}",
    "^{literal 4}",
    "^{ 2 }",
    "^{\\otimes literal 2}",
    "^{⊗ literal 1}",
    "^{03}",
    "^{12}",
)
_TAILS = ("", "; copy", ";copy", " ;  copy ")
_POWER_DIMS = ("d", "3", "literal 3", "n", "literal", "k2")


def _legacy_candidates() -> list[str]:
    plain = [
        f"sum_{{k=0}}^{{{dim}-1}} |{body}>{tail}"
        for dim, body, tail in itertools.product(_DIMS, _BODIES, _TAILS)
    ]
    powered = [
        f"sum_{{k=0}}^{{{dim}-1}} |k>{power}{tail}"
        for dim, power, tail in itertools.product(_POWER_DIMS, _POWERS, _TAILS)
    ]
    spaced = [
        "  sum_{k=0}^{d-1}|k,k>  ",
        "sum_{k=0}^{d-1}\t|k>^{2}; copy",
        "sum_{k=0}^{d-1}\n|k, k>",
        "\tsum_{k=0}^{3-1}   |k>^{\\otimes 5} ;copy\n",
        "sum_{k=0}^{n-1} |k>^{3}",
        "sum_{k=0}^{n-1} |k>^{3}; copy",
    ]
    return plain + powered + spaced


def _legacy_accepts(source: str) -> bool:
    try:
        parse_dirac_source(source)
    except DiracError:
        return False
    return True


LEGACY_CORPUS = [s for s in _legacy_candidates() if _legacy_accepts(s)]


def test_legacy_corpus_is_broad() -> None:
    assert len(LEGACY_CORPUS) > 700
    for must in (
        "sum_{k=0}^{literal-1} |k>",
        "sum_{k=0}^{kd-1} |kk>; copy",
        "sum_{k=0}^{k2-1} |k,,k>",
        "sum_{k=0}^{07-1} |k k>",
        "sum_{k=0}^{d-1} |k>^{⊗3}; copy",
        "sum_{k=0}^{literal 3-1} |k>^{literal 4}",
        "sum_{k=0}^{n-1} |k>^{3}",
    ):
        assert must in LEGACY_CORPUS, must


@pytest.mark.parametrize("source", LEGACY_CORPUS)
def test_legacy_source_is_identical(source: str) -> None:
    want = fields(parse_dirac_source(source))
    assert fields(parse_dirac(source)) == want
    parsed = parse(source)
    assert isinstance(parsed, ParsedDiagram)
    assert parsed.form is InputForm.DIRAC
    assert fields(parsed.diagram) == want


def test_legacy_rejection_of_double_copy_is_kept() -> None:
    with pytest.raises(DiracGrammarError):
        parse_dirac_source("sum_{k=0}^{d-1} |k,k>; copy; copy")


# -- ket sums ------------------------------------------------------------------------------

FUSION_RENDER = "sum_{k0,k1=0}^{d-1} [k0 = k1] |k0, k1, k1>"


def test_fusion_example_render_structure() -> None:
    d = ok(FUSION_RENDER)
    assert [(n.num_inputs, n.num_outputs) for n in d.nodes.values()] == [(0, 2), (1, 2)]
    assert all(n.generator_type.name == "Z" for n in d.nodes.values())
    assert all(n.phase is None or n.phase.is_zero for n in d.nodes.values())
    assert all_dims(d) == {D}
    assert {w.endpoints() for w in d.wires} == {frozenset({out(0, 1), inp(1, 0)})}
    assert d.boundary_outputs == (out(0, 0), out(1, 0), out(1, 1))
    assert d.boundary_inputs == ()
    assert not d.bang_boxes
    reference, _a, _b = build_ghz_with_copy(D)
    assert isomorphic(d, reference, symmetric_legs=True)
    assert_oracle(d, reference, at_d())


def test_unicode_glyphs_match_ascii() -> None:
    assert fields(ok("Σ_{k0,k1=0}^{d-1} [k0 = k1] |k0, k1, k1⟩")) == fields(ok(FUSION_RENDER))
    assert fields(ok("Σ_{k=0}^{d-1} |k⟩⟨k|")) == fields(ok("sum_{k=0}^{d-1} |k><k|"))
    assert fields(ok("Σ_{k=0}^{d-1} |k⟩^{⊗3}")) == fields(ok("sum_{k=0}^{d-1} |k>^{3}"))
    ascii_phase = "sum_{k=0}^{d-1} e^{2 pi i (theta [k=1])} |k>"
    assert fields(ok("Σ_{k=0}^{d-1} e^{2 π i (theta [k=1])} |k⟩")) == fields(ok(ascii_phase))


def test_separate_groups_with_different_dims() -> None:
    d = ok("sum_{a=0}^{d-1} sum_{b=0}^{e-1} |a, b, a>")
    n0, n1 = d.nodes[NodeId(0)], d.nodes[NodeId(1)]
    assert [p.dim for p in n0.outputs] == [D, D]
    assert [p.dim for p in n1.outputs] == [Dim("e")]
    assert d.boundary_outputs == (out(0, 0), out(1, 0), out(0, 1))
    assert not d.wires


def test_multi_index_group_shares_dim() -> None:
    d = ok("sum_{a,b,c=0}^{3-1} |a, b, c>")
    assert len(d.nodes) == 3
    assert len(all_dims(d)) == 1
    assert sorted(d.parameters.values()) == [3]


def test_bras() -> None:
    d = ok("sum_{k=0}^{d-1} <k, k|")
    (node,) = d.nodes.values()
    assert (node.num_inputs, node.num_outputs) == (2, 0)
    assert d.boundary_inputs == (inp(0, 0), inp(0, 1))
    assert d.boundary_outputs == ()
    assert_oracle(d, ok("GHZ_{literal 2}^dagger[d]"), at_d())


def test_ket_and_bra() -> None:
    d = ok("sum_{k=0}^{d-1} |k><k|")
    (node,) = d.nodes.values()
    assert (node.num_inputs, node.num_outputs) == (1, 1)
    assert d.boundary_inputs == (inp(0, 0),)
    assert d.boundary_outputs == (out(0, 0),)
    assert_oracle(d, ok("id[d]"), at_d())


def test_symbolic_ket_fan() -> None:
    d = ok("sum_{k=0}^{d-1} |k>^{n}")
    (node,) = d.nodes.values()
    assert node.num_outputs == 1
    (box,) = d.bang_boxes.values()
    assert box.port_scope == frozenset({out(0, 0)})
    assert not box.node_scope
    assert str(box.multiplicity) == "n"
    assert dict(d.parameters) == {}


def test_symbolic_bra_fan() -> None:
    d = ok("sum_{k=0}^{d-1} <k|^{m}")
    (node,) = d.nodes.values()
    assert (node.num_inputs, node.num_outputs) == (1, 0)
    (box,) = d.bang_boxes.values()
    assert box.port_scope == frozenset({inp(0, 0)})
    assert str(box.multiplicity) == "m"


def test_symbolic_mult_expression_fan() -> None:
    d = ok("sum_{k=0}^{d-1} |k>^{n+1}")
    (box,) = d.bang_boxes.values()
    assert {str(s) for s in box.multiplicity.to_sympy().free_symbols} == {"n"}
    assert_oracle(d, ok("sum_{k=0}^{d-1} |k, k, k>"), at_d(n=2))


def test_concrete_fan_is_abstracted_and_literal_expands() -> None:
    d = ok("sum_{k=0}^{d-1} |k>^{3}")
    (box,) = d.bang_boxes.values()
    assert bound(d, box) == 3
    e = ok("sum_{k=0}^{d-1} |k>^{literal 3}")
    assert not e.bang_boxes
    assert e.nodes[NodeId(0)].num_outputs == 3


def test_fan_then_plain_slot_order() -> None:
    d = ok("sum_{k=0}^{d-1} |k>^{2} |k>")
    (node,) = d.nodes.values()
    assert node.num_outputs == 2
    (box,) = d.bang_boxes.values()
    assert box.port_scope == frozenset({out(0, 0)})
    assert d.boundary_outputs == (out(0, 0), out(0, 1))
    assert_oracle(d, ok("sum_{k=0}^{d-1} |k, k, k>"), at_d())


def _phased_z() -> Diagram:
    d = Diagram()
    pv = PhaseVector(D, {1: Phase.symbol("theta"), 2: Phase.turns(sp.Rational(1, 3))})
    n = d.add_node(Z_SPIDER, [D], [D, D], phase=pv)
    d.set_boundary_inputs([inp(n, 0)])
    d.set_boundary_outputs([out(n, 0), out(n, 1)])
    return d


def test_phase_factor() -> None:
    source = "sum_{k=0}^{d-1} e^{2 pi i (theta [k=1] + 1/3 [k=2])} |k, k><k|"
    d = ok(source)
    (node,) = d.nodes.values()
    entries = node.phase.entries()
    assert set(entries) == {1, 2}
    assert {str(s) for s in entries[1].to_sympy_turns().free_symbols} == {"theta"}
    assert entries[2].to_sympy_turns() == sp.Rational(1, 3)
    assert_oracle(d, _phased_z(), at_d(theta=THETA))
    assert_oracle(d, ok("Z_{1->2}(1: theta, 2: 1/3)[d]"), at_d(theta=THETA))


def test_phase_terms_on_one_value_sum() -> None:
    d = ok("sum_{k=0}^{d-1} e^{2 pi i ((1/3) [k=1] + (1/6) [k=1])} |k>")
    (node,) = d.nodes.values()
    assert node.phase.entries()[1].to_sympy_turns() == sp.Rational(1, 2)


def test_phase_on_legless_index() -> None:
    d = ok("sum_{k,j=0}^{d-1} e^{2 pi i (1/3 [j=1])} |k>")
    legless = d.nodes[NodeId(1)]
    assert (legless.num_inputs, legless.num_outputs) == (0, 0)
    assert legless.phase.entries()[1].to_sympy_turns() == sp.Rational(1, 3)


def test_scalar_prefix() -> None:
    d = ok("(1/2) sum_{k=0}^{d-1} |k, k>")
    assert d.scalar.to_sympy() == sp.Rational(1, 2)
    plain = ok("sum_{k=0}^{d-1} |k, k>")
    assert dict(d.nodes) == dict(plain.nodes)
    assert d.boundary_outputs == plain.boundary_outputs


@pytest.mark.parametrize(
    ("source", "value"),
    [("1/2", sp.Rational(1, 2)), ("sqrt(2)", sp.sqrt(2)), ("2*3", sp.Integer(6))],
)
def test_scalar_only_input(source: str, value: sp.Expr) -> None:
    d = ok(source)
    assert not d.nodes
    assert d.boundary_inputs == d.boundary_outputs == ()
    assert sp.simplify(d.scalar.to_sympy() - value) == 0


def test_symbolic_scalar_only_input() -> None:
    d = ok("2*x")
    assert not d.nodes
    assert d.scalar.free_symbols == frozenset({"x"})


def test_legless_index_gets_zero_phase_vector() -> None:
    d = ok("sum_{k,j=0}^{d-1} |k>")
    assert len(d.nodes) == 2
    legless = d.nodes[NodeId(1)]
    assert (legless.num_inputs, legless.num_outputs) == (0, 0)
    assert legless.phase is not None
    assert legless.phase.is_zero
    assert legless.phase.dim == D


def test_leg_order_kets_then_delta_outs() -> None:
    d = ok("sum_{a,b,c=0}^{d-1} [a = b] [a = c] |a, b, c>")
    assert d.nodes[NodeId(0)].num_outputs == 3
    assert {w.endpoints() for w in d.wires} == {
        frozenset({out(0, 1), inp(1, 0)}),
        frozenset({out(0, 2), inp(2, 0)}),
    }
    assert d.boundary_outputs == (out(0, 0), out(1, 0), out(2, 0))
    assert_oracle(d, ok("GHZ_{literal 3}[d]"), at_d())


def test_leg_order_bras_then_delta_ins() -> None:
    d = ok("sum_{a,b=0}^{d-1} [a = b] [b = a] |a> <b|")
    a, b = d.nodes[NodeId(0)], d.nodes[NodeId(1)]
    assert (a.num_inputs, a.num_outputs) == (1, 2)
    assert (b.num_inputs, b.num_outputs) == (2, 1)
    assert {w.endpoints() for w in d.wires} == {
        frozenset({out(0, 1), inp(1, 1)}),
        frozenset({out(1, 0), inp(0, 0)}),
    }
    assert d.boundary_outputs == (out(0, 0),)
    assert d.boundary_inputs == (inp(1, 0),)


@pytest.mark.parametrize(
    "source",
    ["sum_{k=0}^{k-1} |k>", "sum_{j=0}^{j-1} |j>", "sum_{a,b=0}^{b-1} |a, b>"],
)
def test_dim_equal_to_index_is_domain_error(source: str) -> None:
    with pytest.raises(ParseDomainError):
        parse_dirac(source)


@pytest.mark.parametrize(
    "source",
    ["sum_{k=0}^{d-1} |j>", "sum_{k=0}^{d-1} |k, j>", "sum_{k=0}^{d-1} <q|"],
)
def test_unbound_index_is_error(source: str) -> None:
    with pytest.raises(ParseError):
        parse_dirac(source)


def test_copy_family_is_grammar_error() -> None:
    with pytest.raises(ParseGrammarError):
        parse_dirac("sum_{k=0}^{d-1} |k[i1]>")


@pytest.mark.parametrize("power", ["^{0}", "^{literal 0}", "^{\\otimes 0}"])
def test_zero_power_is_domain_error(power: str) -> None:
    with pytest.raises(ParseDomainError):
        parse_dirac(f"sum_{{k=0}}^{{d-1}} |k>{power}")


# -- catalog atoms -------------------------------------------------------------------------

CATALOG = [
    ("plus[d]", "Z", 0, 1),
    ("plus^dagger[d]", "Z", 1, 0),
    ("id[d]", "Z", 1, 1),
    ("copy[d]", "Z", 1, 2),
    ("zero[d]", "X", 0, 1),
    ("F[d]", "F", 1, 1),
    ("triangle[d]", "T", 1, 1),
    ("triangle^-1[d]", "Ti", 1, 1),
    ("GHZ_{literal 3}[d]", "Z", 0, 3),
    ("GHZ_{literal 3}^dagger[d]", "Z", 3, 0),
    ("copy_{literal 3}[d]", "Z", 1, 3),
    ("copy_{literal 3}^dagger[d]", "Z", 3, 1),
    ("Z_{1->1}[d]", "Z", 1, 1),
    ("Z_{0->1}[d]", "Z", 0, 1),
    ("X_{1->1}[d]", "X", 1, 1),
    ("X_{1->0}[d]", "X", 1, 0),
    ("W_{literal 3}[d]", "W", 1, 3),
    ("W_{1}[d]", "W", 1, 1),
]


@pytest.mark.parametrize(("source", "gen", "m", "n"), CATALOG, ids=[c[0] for c in CATALOG])
def test_catalog_atom(source: str, gen: str, m: int, n: int) -> None:
    d = ok(source)
    (node,) = d.nodes.values()
    assert node.generator_type.name == gen
    assert (node.num_inputs, node.num_outputs) == (m, n)
    assert all_dims(d) == {D}
    assert not d.bang_boxes
    assert dict(d.parameters) == {}
    assert d.boundary_inputs == tuple(inp(0, i) for i in range(m))
    assert d.boundary_outputs == tuple(out(0, i) for i in range(n))


def test_bind_and_split_atoms() -> None:
    s, t = Dim("s"), Dim("t")
    b = ok("bind[s, t]")
    (node,) = b.nodes.values()
    assert node.generator_type.name == DIM_BINDER.name
    assert [p.dim for p in node.inputs] == [s, t]
    assert [p.dim for p in node.outputs] == [s * t]
    split = ok("split[s, t]")
    (node,) = split.nodes.values()
    assert node.generator_type.name == DIM_SPLITTER.name
    assert [p.dim for p in node.inputs] == [s * t]
    assert [p.dim for p in node.outputs] == [s, t]


def test_subscript_zero_gives_no_legs() -> None:
    d = ok("GHZ_{0}[d]")
    (node,) = d.nodes.values()
    assert (node.num_inputs, node.num_outputs) == (0, 0)
    assert not d.bang_boxes


def test_subscript_one_gives_plain_leg() -> None:
    d = ok("GHZ_{1}[d]")
    assert fields(d) == fields(ok("plus[d]"))


@pytest.mark.parametrize("c", [2, 3, 7])
def test_subscript_two_or_more_gives_abstracted_fan(c: int) -> None:
    d = ok(f"GHZ_{{{c}}}[d]")
    (node,) = d.nodes.values()
    assert (node.num_inputs, node.num_outputs) == (0, 1)
    (box,) = d.bang_boxes.values()
    assert box.port_scope == frozenset({out(0, 0)})
    assert bound(d, box) == c
    assert len(d.parameters) == 1


def test_symbolic_subscript_with_plain_part() -> None:
    d = ok("GHZ_{n+2}[d]")
    (node,) = d.nodes.values()
    assert node.num_outputs == 3
    (box,) = d.bang_boxes.values()
    assert box.port_scope == frozenset({out(0, 2)})
    assert str(box.multiplicity) == "n"
    assert dict(d.parameters) == {}
    assert_oracle(d, ok("GHZ_{literal 4}[d]"), at_d(n=2))


def test_symbolic_subscript_bare() -> None:
    d = ok("copy_{n}[d]")
    (node,) = d.nodes.values()
    assert (node.num_inputs, node.num_outputs) == (1, 1)
    (box,) = d.bang_boxes.values()
    assert box.port_scope == frozenset({out(0, 0)})
    assert str(box.multiplicity) == "n"


def test_two_concrete_subscripts_get_distinct_symbols() -> None:
    d = ok("Z_{2->3}[d]")
    (node,) = d.nodes.values()
    assert (node.num_inputs, node.num_outputs) == (1, 1)
    boxes = [d.bang_boxes[b] for b in sorted(d.bang_boxes)]
    assert [box.port_scope for box in boxes] == [frozenset({inp(0, 0)}), frozenset({out(0, 0)})]
    assert [bound(d, box) for box in boxes] == [2, 3]
    assert boxes[0].multiplicity != boxes[1].multiplicity
    assert_oracle(d, ok("Z_{literal 2->literal 3}[d]"), at_d())


def test_dagger_atoms_are_transposes() -> None:
    assert_oracle(ok("GHZ_{3}^dagger[d]"), ok("sum_{k=0}^{d-1} <k, k, k|"), at_d())
    assert_oracle(ok("copy_{3}^dagger[d]"), ok("sum_{k=0}^{d-1} |k><k, k, k|"), at_d())
    assert_oracle(ok("copy_{3}[d]"), ok("sum_{k=0}^{d-1} |k, k, k><k|"), at_d())


def test_triangle_inverse_composes_to_identity() -> None:
    assert_oracle(ok("triangle[d] ; triangle^-1"), ok("id[d]"), at_d())


def test_z_and_x_phase_entries() -> None:
    d = ok("Z_{1->1}(1: theta, 2: 1/3)[d]")
    (node,) = d.nodes.values()
    assert set(node.phase.entries()) == {1, 2}
    assert node.phase.entries()[2].to_sympy_turns() == sp.Rational(1, 3)
    x = ok("X_{1->1}(1: 1/2)[d]")
    (node,) = x.nodes.values()
    assert node.generator_type.name == "X"
    assert node.phase.entries()[1].to_sympy_turns() == sp.Rational(1, 2)


def test_z_phase_name_is_rejected_with_hint() -> None:
    with pytest.raises(ParseGrammarError) as info:
        parse_dirac("Z_phase(1: 1/3)[d]")
    assert "Z_{1->1}" in str(info.value)


def test_numeral_dim_is_abstracted_and_literal_is_not() -> None:
    d = ok("plus[3]")
    (dim,) = all_dims(d)
    assert not dim.is_concrete
    assert dict(d.parameters) == {str(dim): 3}
    e = ok("plus[literal 3]")
    assert all_dims(e) == {Dim(3)}
    assert dict(e.parameters) == {}


def test_dims_shared_per_numeral_value() -> None:
    d = ok("GHZ_{literal 2}[3] (x) plus[3]")
    assert len(all_dims(d)) == 1
    assert sorted(d.parameters.values()) == [3]
    e = ok("plus[3] (x) plus[5]")
    assert len(all_dims(e)) == 2
    assert sorted(e.parameters.values()) == [3, 5]
    assert [dim_value(e, n.outputs[0].dim) for n in e.nodes.values()] == [3, 5]


def test_numeral_dim_and_count_get_separate_symbols() -> None:
    d = ok("GHZ_{3}[3]")
    assert sorted(d.parameters.values()) == [3, 3]
    (box,) = d.bang_boxes.values()
    (dim,) = all_dims(d)
    assert str(box.multiplicity) != str(dim)


@pytest.mark.parametrize("source", ["plus[d] ; copy", "plus[d] ; id ; copy", "id[d] ; copy"])
def test_dims_inferred_through_feed(source: str) -> None:
    assert all_dims(ok(source)) == {D}


def test_abstracted_dim_inferred_through_feed() -> None:
    d = ok("plus[3] ; copy")
    assert len(all_dims(d)) == 1
    assert sorted(d.parameters.values()) == [3]


def test_bind_infers_both_dims() -> None:
    d = ok("plus[2] (x) plus[3] ; bind")
    p0, p1, b = (d.nodes[NodeId(i)] for i in range(3))
    assert b.generator_type.name == DIM_BINDER.name
    assert [p.dim for p in b.inputs] == [p0.outputs[0].dim, p1.outputs[0].dim]
    assert b.outputs[0].dim == p0.outputs[0].dim * p1.outputs[0].dim
    assert_oracle(d, ok("plus[6]"), [{}])
    s = ok("plus[s] (x) plus[t] ; bind")
    assert s.nodes[NodeId(2)].outputs[0].dim == Dim("s") * Dim("t")


def test_bind_then_split_is_identity() -> None:
    assert_oracle(ok("bind[2, 3] ; split[2, 3]"), ok("id[2] (x) id[3]"), [{}])
    assert_oracle(ok("bind[s, t] ; split[s, t]"), ok("id[s] (x) id[t]"), [{"s": 2, "t": 3}])


@pytest.mark.parametrize(
    ("source", "atom"),
    [("copy", "copy"), ("plus ; copy", "plus"), ("id", "id"), ("bind", "bind"), ("F", "F")],
)
def test_unbound_inferred_dim_is_domain_error(source: str, atom: str) -> None:
    with pytest.raises(ParseDomainError) as info:
        parse_dirac(source)
    assert atom in str(info.value)


@pytest.mark.parametrize("source", ["plus[6] ; split", "split", "bind[2, 3] ; split"])
def test_split_needs_explicit_dims(source: str) -> None:
    with pytest.raises(ParseError):
        parse_dirac(source)


# -- composition ---------------------------------------------------------------------------


def test_tensor_concatenates_outputs() -> None:
    d = ok("plus[d] (x) GHZ_{literal 2}[d]")
    assert d.boundary_outputs == (out(0, 0), out(1, 0), out(1, 1))
    assert not d.wires
    assert fields(ok("plus[d] ⊗ GHZ_{literal 2}[d]")) == fields(d)


def test_tensor_concatenates_inputs() -> None:
    d = ok("id[d] (x) copy[d]")
    assert d.boundary_inputs == (inp(0, 0), inp(1, 0))
    assert d.boundary_outputs == (out(0, 0), out(1, 0), out(1, 1))


def test_feed_boundary_order() -> None:
    d = ok("GHZ_{literal 3}[d] ; copy")
    assert {w.endpoints() for w in d.wires} == {frozenset({out(0, 0), inp(1, 0)})}
    assert d.boundary_outputs == (out(0, 1), out(0, 2), out(1, 0), out(1, 1))
    assert d.boundary_inputs == ()


def test_feed_into_tensor() -> None:
    d = ok("copy[d] ; id (x) id")
    assert {w.endpoints() for w in d.wires} == {
        frozenset({out(0, 0), inp(1, 0)}),
        frozenset({out(0, 1), inp(2, 0)}),
    }
    assert d.boundary_inputs == (inp(0, 0),)
    assert d.boundary_outputs == (out(1, 0), out(2, 0))
    assert_oracle(d, ok("copy[d]"), at_d())


def test_feed_keeps_a_inputs() -> None:
    d = ok("id[d] ; copy")
    assert d.boundary_inputs == (inp(0, 0),)
    assert d.boundary_outputs == (out(1, 0), out(1, 1))


def test_peeling_pending_fan_matches_legacy() -> None:
    d = ok("GHZ_{5}[d] ; copy")
    assert fields(d) == fields(parse_dirac_source("sum_{k=0}^{d-1} |k>^{5}; copy"))
    (box,) = d.bang_boxes.values()
    assert box.port_scope == frozenset({out(0, 1)})
    assert bound(d, box) == 4


def test_peeling_to_zero_keeps_fan() -> None:
    d = ok("GHZ_{2}[d] ; copy ; copy")
    node = d.nodes[NodeId(0)]
    assert node.num_outputs == 3
    (box,) = d.bang_boxes.values()
    assert box.port_scope == frozenset({out(0, 2)})
    assert bound(d, box) == 0
    assert_oracle(d, ok("sum_{k=0}^{d-1} |k, k, k, k>"), at_d())


@pytest.mark.parametrize(
    "source",
    [
        "GHZ_{n}[d] ; copy",
        "sum_{k=0}^{d-1} |k>^{n} ; copy",
        "plus[d] ; copy_{n}^dagger[d]",
    ],
)
def test_symbolic_fan_in_feed_is_domain_error(source: str) -> None:
    with pytest.raises(ParseDomainError):
        parse_dirac(source)


@pytest.mark.parametrize(
    "source",
    ["plus[d] ; copy_{literal 2}^dagger[d]", "plus[d] ; copy_{2}^dagger[d]", "plus[d] ; id (x) id"],
)
def test_too_few_outputs_is_domain_error(source: str) -> None:
    with pytest.raises(ParseDomainError):
        parse_dirac(source)


def test_parenthesized_groups() -> None:
    d = ok("(plus[d] ; copy) (x) plus[d]")
    assert d.boundary_outputs == (out(1, 0), out(1, 1), out(2, 0))
    e = ok("plus[d] (x) (plus[d] ; copy)")
    assert e.boundary_outputs == (out(0, 0), out(2, 0), out(2, 1))
    assert fields(ok("(plus[d])")) == fields(ok("plus[d]"))


def test_scalar_symbol_x_at_term_start() -> None:
    d = ok("(x) plus[d]")
    assert len(d.nodes) == 1
    assert d.scalar.free_symbols == frozenset({"x"})
    e = ok("(x) plus[d] (x) plus[d]")
    assert len(e.nodes) == 2
    assert e.scalar.free_symbols == frozenset({"x"})


def test_scalars_multiply() -> None:
    assert ok("(2) plus[d] (x) (3) plus[d]").scalar.to_sympy() == 6
    assert ok("(2) plus[d] ; (3) copy").scalar.to_sympy() == 6


# -- oracle semantics ----------------------------------------------------------------------


def test_ghz_atom_equals_ket_sum() -> None:
    assert_oracle(ok("GHZ_{3}[d]"), ok("sum_{k=0}^{d-1} |k,k,k>"), at_d())


def test_plus_into_copy_equals_ghz2() -> None:
    assert_oracle(ok("plus[d] ; copy"), ok("GHZ_{2}[d]"), at_d())


def _with_scalar_chain() -> Diagram:
    d = parse_dirac_source("sum_{k=0}^{d-1} |k,k,k>")
    n = d.add_node(Z_SPIDER, [D], [D])
    d.add_wire(out(0, 0), inp(n, 0))
    d.set_boundary_outputs([out(n, 0), out(0, 1), out(0, 2)])
    d.set_boundary_inputs([])
    d.multiply_scalar(Scalar.rational(1, 2))
    return d


def _input_fan() -> Diagram:
    return parse_dirac_source("sum_{k=0}^{d-1} |k>^{3}; copy")


RENDER_CASES = {
    "ghz_with_copy": lambda: build_ghz_with_copy(D)[0],
    "fan_into_copy": _input_fan,
    "phased": _phased_z,
    "concrete": lambda: parse_dirac_source("sum_{k=0}^{3-1} |k>^{4}"),
    "scalar_chain": _with_scalar_chain,
}


@pytest.mark.parametrize("unicode", [False, True])
@pytest.mark.parametrize("case", sorted(RENDER_CASES))
def test_render_reparses_oracle_equal(case: str, unicode: bool) -> None:
    diagram = RENDER_CASES[case]()
    text = P.render_term(P.dirac_term(diagram), unicode=unicode)
    reparsed = ok(text)
    assert_oracle(diagram, reparsed, at_d(n=3, theta=THETA))


def _single(gen: GeneratorType, ins: list[Dim], outs: list[Dim], phase: object = None) -> Diagram:
    d = Diagram()
    if phase is None and gen in (Z_SPIDER, X_SPIDER):
        phase = PhaseVector(ins[0] if ins else outs[0])
    n = d.add_node(gen, ins, outs, phase=phase)  # type: ignore[arg-type]
    d.set_boundary_inputs([inp(n, i) for i in range(len(ins))])
    d.set_boundary_outputs([out(n, i) for i in range(len(outs))])
    return d


S_, T_ = Dim("s"), Dim("t")
RECOGNIZE_CASES = {
    "ghz3": lambda: _single(Z_SPIDER, [], [D] * 3),
    "ghz2_dagger": lambda: _single(Z_SPIDER, [D] * 2, []),
    "copy3": lambda: _single(Z_SPIDER, [D], [D] * 3),
    "copy2_dagger": lambda: _single(Z_SPIDER, [D] * 2, [D]),
    "plus": lambda: _single(Z_SPIDER, [], [D]),
    "plus_dagger": lambda: _single(Z_SPIDER, [D], []),
    "id": lambda: _single(Z_SPIDER, [D], [D]),
    "z23": lambda: _single(Z_SPIDER, [D] * 2, [D] * 3),
    "z_phased": lambda: _single(Z_SPIDER, [D], [D, D], PhaseVector(D, {1: Phase.symbol("theta")})),
    "zero": lambda: _single(X_SPIDER, [], [D]),
    "x12": lambda: _single(X_SPIDER, [D], [D, D]),
    "x_phased": lambda: _single(
        X_SPIDER, [D, D], [D], PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3))})
    ),
    "fourier": lambda: _single(FOURIER_BOX, [D], [D]),
    "triangle": lambda: _single(TRIANGLE, [D], [D]),
    "triangle_inv": lambda: _single(TRIANGLE_INVERSE, [D], [D]),
    "w3": lambda: _single(W_NODE, [D], [D] * 3),
    "bind": lambda: _single(DIM_BINDER, [S_, T_], [S_ * T_]),
    "split": lambda: _single(DIM_SPLITTER, [S_ * T_], [S_, T_]),
}


@pytest.mark.parametrize("case", sorted(RECOGNIZE_CASES))
def test_recognized_name_reparses_oracle_equal(case: str) -> None:
    diagram = RECOGNIZE_CASES[case]()
    name = P.recognize(diagram)
    assert name is not None
    dims = "[s, t]" if name in ("bind", "split") else "[d]"
    reparsed = ok(name + dims)
    envs = [{"d": v, "s": 2, "t": 3, "theta": THETA} for v in DS]
    assert_oracle(diagram, reparsed, envs)


# -- error sweep ---------------------------------------------------------------------------

BAD = [
    ("plus[d] ;", GRAMMAR),
    ("; plus[d]", GRAMMAR),
    ("plus[d] (x)", GRAMMAR),
    ("plus[d] ⊗", GRAMMAR),
    ("(plus[d]", GRAMMAR),
    ("plus[d])", GRAMMAR),
    ("plus[d] (x) (x)", GRAMMAR),
    ("plus[d][d]", GRAMMAR),
    ("plus[d, e]", GRAMMAR),
    ("plus[1.5]", GRAMMAR),
    ("plus[0]", DOMAIN),
    ("plus[-1]", GRAMMAR),
    ("bind[d]", GRAMMAR),
    ("split[d]", GRAMMAR),
    ("GHZ_{}[d]", GRAMMAR),
    ("GHZ_{-1}[d]", GRAMMAR),
    ("GHZ_{3}", DOMAIN),
    ("Z_{1->}[d]", GRAMMAR),
    ("Z_{1->1}(1: )[d]", GRAMMAR),
    ("Z_{1->1}(x)[d]", GRAMMAR),
    ("Z_phase(1: 1/3)[d]", GRAMMAR),
    ("frob[d]", GRAMMAR),
    ("copy", DOMAIN),
    ("plus ; copy", DOMAIN),
    ("split", GRAMMAR),
    ("plus[2] ; copy[3]", DOMAIN),
    ("plus[d] ; copy_{n}^dagger[d]", DOMAIN),
    ("GHZ_{n}[d] ; copy", DOMAIN),
    ("sum_{k=0}^{d-1} |j>", GRAMMAR),
    ("sum_{k=0}^{k-1} |k>", DOMAIN),
    ("sum_{d=0}^{d-1} |d>", DOMAIN),
    ("sum_{k=0}^{d-1} |k[i1]>", GRAMMAR),
    ("sum_{k=0}^{d-1} |k>^{0}", DOMAIN),
    ("sum_{k=0}^{d-1} |k>^{literal 0}", DOMAIN),
    ("sum_{k=0}^{d-1} |k,k>^{2}", GRAMMAR),
    ("sum_{k=0}^{d-1} |k>^{literal 100000}", DOMAIN),
    ("sum_{k=1}^{d-1} |k>", GRAMMAR),
    ("sum_{k=0}^{d} |k>", GRAMMAR),
    ("sum_{k=0}^{d-1} |k", GRAMMAR),
    ("sum_{k=0}^{d-1} k>", GRAMMAR),
    ("sum_{k=0}^{d-1} [k = q] |k>", GRAMMAR),
    ("sum_{k=0}^{d-1} e^{2 pi i (1/3 [k=1])", GRAMMAR),
    ("sum_{k=0}^{d-1} e^{2 pi i (1/3 [q=1])} |k>", GRAMMAR),
    ("sum_{k=0}^{d-1} |k>⟩", GRAMMAR),
    ("sum_{k=0}^{0-1} |k>", DOMAIN),
    ("sum_{k=0}^{d-1} |kk>^{2}", GRAMMAR),
    ("(" * 5000 + "plus[d]" + ")" * 5000, GRAMMAR),
    ("(" * 5000 + "1" + ")" * 5000, GRAMMAR),
    ("|k>", GRAMMAR),
    ("<k|", GRAMMAR),
    ("@", GRAMMAR),
    ("\x00", GRAMMAR),
    ("plus[d] ;; copy", GRAMMAR),
]


@pytest.mark.parametrize(("source", "cls"), BAD, ids=range(len(BAD)))
def test_bad_input_raises_its_error_class(source: str, cls: type[ParseError]) -> None:
    with pytest.raises(cls):
        parse_dirac(source)
    with pytest.raises(cls):
        parse(source)


BIG_NUMERALS = [
    "GHZ_{9999999999999999999}[d]",
    "sum_{k=0}^{d-1} |k>^{" + "9" * 30 + "}",
    "sum_{k=0}^{" + "9" * 30 + "-1} |k>",
]


@pytest.mark.parametrize("source", BIG_NUMERALS, ids=range(len(BIG_NUMERALS)))
def test_big_numerals_parse_or_raise_parse_error(source: str) -> None:
    try:
        diagram = parse_dirac(source)
    except ParseError:
        return
    assert validate(diagram).is_valid


@pytest.mark.parametrize("depth", [5000, 100000])
def test_nesting_cap_is_grammar_error(depth: int) -> None:
    with pytest.raises(ParseGrammarError):
        parse_dirac("(" * depth + "plus[d]" + ")" * depth)


GOOD = [
    FUSION_RENDER,
    "sum_{k=0}^{d-1} e^{2 pi i (theta [k=1] + 1/3 [k=2])} |k, k><k|",
    "(1/2) GHZ_{3}[d] ; copy (x) Z_{1->1}(1: 1/3)[d]",
    "bind[2, 3] ; split[2, 3]",
    "Σ_{k=0}^{literal 3-1} |k⟩^{⊗ n+1} ⟨k|",
]


@pytest.mark.parametrize("source", GOOD, ids=range(len(GOOD)))
def test_every_prefix_parses_or_raises_parse_error(source: str) -> None:
    for end in range(len(source) + 1):
        try:
            diagram = parse_dirac(source[:end])
        except ParseError:
            continue
        assert validate(diagram).is_valid, source[:end]


# -- bound values of a concrete render -------------------------------------------------------


@pytest.mark.parametrize("unicode", [False, True])
def test_concrete_render_reparses_at_its_bound_values(unicode: bool) -> None:
    diagram = RENDER_CASES["concrete"]()
    assert diagram.parameters
    reparsed = ok(P.render_term(P.dirac_term(diagram), unicode=unicode))
    assert_oracle(diagram, reparsed, [dict(diagram.parameters)])


# -- subscripts, dispatch and atoms (fix list D) ---------------------------------------------


def test_constant_subscript_sum_is_one_count() -> None:
    d = ok("GHZ_{2 + 3*1}[d]")
    (node,) = d.nodes.values()
    assert (node.num_inputs, node.num_outputs) == (0, 1)
    (box,) = d.bang_boxes.values()
    assert bound(d, box) == 5
    assert_oracle(d, ok("GHZ_{literal 5}[d]"), at_d())


def test_product_subscript_is_one_symbolic_fan() -> None:
    d = ok("GHZ_{2*n}[d]")
    (node,) = d.nodes.values()
    assert (node.num_inputs, node.num_outputs) == (0, 1)
    (box,) = d.bang_boxes.values()
    assert box.port_scope == frozenset({out(0, 0)})
    assert box.multiplicity == Mult("n") * 2
    assert dict(d.parameters) == {}
    assert_oracle(d, ok("GHZ_{literal 4}[d]"), at_d(n=2))


def test_non_generator_call_is_not_dsl() -> None:
    with pytest.raises(ParseGrammarError) as info:
        parse("psi = id (x) id ; bind[2, 3]")
    assert "generator" not in str(info.value)


def test_catalog_name_line_is_dirac() -> None:
    parsed = parse("id (x) id = id (x) id ; bind[2, 3]")
    assert isinstance(parsed, ParsedDiagram)
    assert parsed.form is InputForm.DIRAC
    assert_oracle(parsed.diagram, ok("id[2] (x) id[3] ; bind"), [{}])


@pytest.mark.parametrize(
    ("source", "form"),
    [
        ("# note\nsum_{k=0}^{d-1} |k, k>", InputForm.DIRAC),
        ("# a = Z(0 -> 1, d)\nplus[d]", InputForm.DIRAC),
        ("# plus[d]\na = Z(0 -> 1, d)", InputForm.DSL),
    ],
)
def test_comments_are_stripped_before_dispatch(source: str, form: InputForm) -> None:
    parsed = parse(source)
    assert isinstance(parsed, ParsedDiagram)
    assert parsed.form is form


@pytest.mark.parametrize("source", ["x = 2", "psi = plus[d]", "a b = plus[d]"])
def test_name_line_with_non_catalog_left_side(source: str) -> None:
    with pytest.raises(ParseGrammarError):
        parse_dirac(source)


def test_name_line_with_catalog_left_side() -> None:
    assert fields(ok("GHZ_{2}^dagger = sum_{k=0}^{3-1} <k, k|")) == fields(
        ok("sum_{k=0}^{3-1} <k, k|")
    )


def test_tensor_right_after_a_spider_atom() -> None:
    d = ok("plus[d] (x) plus[d] ; Z_{1->1}(x)Z_{1->1}")
    assert len(d.nodes) == 4
    assert all(n.generator_type.name == "Z" for n in d.nodes.values())
    assert_oracle(d, ok("plus[d] (x) plus[d]"), at_d())
    e = ok("X_{0->1}[d](x)plus[d]")
    assert [n.generator_type.name for n in e.nodes.values()] == ["X", "Z"]


@pytest.mark.parametrize("source", ["plus[] ; id[d]", "plus[]", "id[d] ; copy[]"])
def test_empty_dims_are_grammar_errors(source: str) -> None:
    with pytest.raises(ParseGrammarError):
        parse_dirac(source)


@pytest.mark.parametrize("source", ["bind[2]", "split[d]"])
def test_bind_split_hint(source: str) -> None:
    with pytest.raises(ParseGrammarError) as info:
        parse_dirac(source)
    assert "[s, t]" in str(info.value)


@pytest.mark.parametrize("source", ["cpy", "spider", "x", "theta"])
def test_lone_identifier_is_not_a_scalar(source: str) -> None:
    with pytest.raises(ParseGrammarError) as info:
        parse(source)
    assert "unknown catalog atom" in str(info.value)


@pytest.mark.parametrize("source", ["2", "pi", "E", "I", "2*x", "x + 1", "sqrt(2)", "-x"])
def test_numbers_constants_and_compounds_are_scalars(source: str) -> None:
    d = ok(source)
    assert not d.nodes


@pytest.mark.parametrize(
    ("source", "reference"),
    [
        ("bind[2, 3] ; id[6]", "bind[2, 3]"),
        ("plus[2] ; id[literal 2]", "plus[literal 2]"),
        ("plus[literal 2] ; id[2]", "plus[literal 2]"),
        ("plus[6] ; split[2, 3]", "plus[2] (x) plus[3]"),
    ],
)
def test_feed_unifies_dims_under_bindings(source: str, reference: str) -> None:
    d = ok(source)
    assert_oracle(d, ok(reference), [{}])


@pytest.mark.parametrize(
    "source",
    [
        "plus[d] ; id[e]",
        "plus[2] ; id[3]",
        "bind[2, 3] ; id[5]",
        "sum_{a=0}^{d-1} sum_{b=0}^{e-1} [a = b] |a, b>",
        "sum_{a=0}^{2-1} sum_{b=0}^{3-1} [a = b] |a> <b|",
    ],
)
def test_dim_mismatch_is_domain_error(source: str) -> None:
    with pytest.raises(ParseDomainError):
        parse_dirac(source)


def test_delta_unifies_dims_under_bindings() -> None:
    d = ok("sum_{a=0}^{2-1} sum_{b=0}^{literal 2-1} [a = b] |a> <b|")
    assert_oracle(d, ok("id[literal 2]"), [{}])


def test_feed_into_pending_fan_on_b_side() -> None:
    d = ok("GHZ_{literal 3}[d] ; copy_{3}^dagger")
    assert not d.bang_boxes
    assert d.nodes[NodeId(1)].num_inputs == 3
    assert_oracle(d, ok("plus[d]"), at_d())
    e = ok("GHZ_{3}[d] ; copy_{3}^dagger")
    assert_oracle(e, ok("plus[d]"), at_d())
    with pytest.raises(ParseDomainError):
        parse_dirac("GHZ_{2}[d] ; copy_{3}^dagger")


def test_total_feed_expansion_is_capped() -> None:
    source = "plus[d] ; sum_{k=0}^{d-1} " + "<k|^{1024}" * 600
    outcome, _ = bounded(lambda: parse_dirac(source), 2.0)
    assert isinstance(outcome, ParseDomainError)


def test_phase_coefficient_naming_an_index_is_domain_error() -> None:
    with pytest.raises(ParseDomainError):
        parse_dirac("sum_{k=0}^{d-1} e^{2 pi i (k [k=1])} |k>")
    with pytest.raises(ParseDomainError):
        parse_dirac("sum_{k,j=0}^{d-1} e^{2 pi i (j [k=1])} |k, j>")


@pytest.mark.parametrize(
    "source",
    [
        "Z_{1->1}(5: 1/2)[3]",
        "Z_{1->1}(3: 1/2)[3]",
        "X_{1->1}(3: 1/2)[literal 3]",
        "sum_{k=0}^{3-1} e^{2 pi i (1/2 [k=3])} |k>",
    ],
)
def test_phase_index_beyond_concrete_dim_is_domain_error(source: str) -> None:
    with pytest.raises(ParseDomainError):
        parse_dirac(source)


def test_phase_index_within_concrete_dim() -> None:
    ok("Z_{1->1}(2: 1/2)[3]")
    ok("Z_{1->1}(5: 1/2)[d]")


@pytest.mark.parametrize(
    "template",
    ["GHZ_{literal N}[d]", "copy_{literal N}[d]", "sum_{k=0}^{d-1} |k>^{literal N}"],
)
def test_leg_bound(template: str) -> None:
    d = ok(template.replace("N", "1024"))
    assert sum(n.num_outputs for n in d.nodes.values()) >= 1024
    with pytest.raises(ParseDomainError):
        parse_dirac(template.replace("N", "1025"))


def test_minting_order_across_tensor_atoms() -> None:
    d = ok("Z_{2->3}[d] (x) GHZ_{4}[7] (x) plus[5] (x) plus[7]")
    boxes = [d.bang_boxes[b] for b in sorted(d.bang_boxes)]
    assert [str(b.multiplicity) for b in boxes] == ["n", "n1", "n2"]
    assert [b.port_scope for b in boxes] == [
        frozenset({inp(0, 0)}),
        frozenset({out(0, 0)}),
        frozenset({out(1, 0)}),
    ]
    assert dict(d.parameters) == {"n": 2, "n1": 3, "n2": 4, "d1": 7, "d2": 5}


def test_minted_names_avoid_source_identifiers() -> None:
    d = ok("plus[d] (x) sum_{d0,d1,d2,d3=0}^{5-1} |d0, d1, d2, d3>")
    assert dict(d.parameters) == {"d4": 5}
    e = ok("plus[n] (x) plus[n1] (x) GHZ_{3}[d] (x) GHZ_{2}[d]")
    assert dict(e.parameters) == {"n2": 3, "n3": 2}
