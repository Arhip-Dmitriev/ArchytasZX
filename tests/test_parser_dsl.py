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

"""Phase 18 parser: the spider/wire DSL, the expression syntax, the graph-listing form,
rule steps, dispatch, and the error contract."""

from __future__ import annotations

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.compare import compare_structure
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import validate
from archytaszx.repl import parser as PR
from archytaszx.repl.printer import render_graph
from archytaszx.rewrite.engine import RewriteResult, apply
from archytaszx.rewrite.rules_library import RULES, SPIDER_FUSION

from .helpers import bounded, build_ghz_with_copy

D = Dim.symbol("d")
E = Dim.symbol("e")
GRAMMAR = PR.ParseGrammarError
DOMAIN = PR.ParseDomainError


def i(node: int, index: int) -> PortRef:
    return PortRef(NodeId(node), Direction.INPUT, index)


def o(node: int, index: int) -> PortRef:
    return PortRef(NodeId(node), Direction.OUTPUT, index)


def dsl(source: str) -> Diagram:
    """parse_dsl, asserting the result validates."""
    diagram = PR.parse_dsl(source)
    assert isinstance(diagram, Diagram)
    assert validate(diagram).is_valid, validate(diagram)
    return diagram


def dims(ports: tuple[object, ...]) -> list[Dim]:
    return [p.dim for p in ports]  # type: ignore[attr-defined]


def scalar_of(expr: str, prelude: str = "a = Z(0 -> 1, d)") -> Scalar:
    """The scalar a DSL source with ``scalar expr`` produces."""
    return dsl(f"{prelude}; scalar {expr}").scalar


# -- error classes ----------------------------------------------------------------------------


class TestErrorClasses:
    def test_hierarchy(self) -> None:
        assert issubclass(PR.ParseGrammarError, PR.ParseError)
        assert issubclass(PR.ParseDomainError, PR.ParseError)
        assert issubclass(PR.ParseError, Exception)

    def test_dirac_aliases(self) -> None:
        assert PR.DiracError is PR.ParseError
        assert PR.DiracGrammarError is PR.ParseGrammarError
        assert PR.DiracDomainError is PR.ParseDomainError

    def test_legacy_parser_raises_parse_errors(self) -> None:
        with pytest.raises(PR.ParseError):
            PR.parse_dirac_source("sum_{k=0}^{d-1} |k,k>; copy; copy")

    def test_z_phase_is_a_grammar_error(self) -> None:
        with pytest.raises(PR.ParseGrammarError, match=r"Z_\{1->1\}"):
            PR.parse_dirac("Z_phase")


# -- DSL node forms ---------------------------------------------------------------------------


class TestNodeForms:
    def test_shorthand_legs(self) -> None:
        d = dsl("a = Z(1 -> 2, d)")
        (n,) = d.nodes.values()
        assert n.generator_type.name == "Z"
        assert dims(n.inputs) == [D]
        assert dims(n.outputs) == [D, D]
        assert n.phase is None

    @pytest.mark.parametrize("gen", ["Z", "X"])
    def test_shorthand_spiders(self, gen: str) -> None:
        d = dsl(f"a = {gen}(2 -> 0, d)")
        assert d.nodes[NodeId(0)].generator_type.name == gen
        assert d.boundary_inputs == (i(0, 0), i(0, 1))
        assert d.boundary_outputs == ()

    def test_w_shorthand(self) -> None:
        d = dsl("w = W(1 -> 3, d)")
        n = d.nodes[NodeId(0)]
        assert n.generator_type.name == "W"
        assert dims(n.outputs) == [D, D, D]

    @pytest.mark.parametrize("gen", ["F", "T", "Ti"])
    def test_one_to_one_shorthands(self, gen: str) -> None:
        d = dsl(f"a = {gen}(d)")
        n = d.nodes[NodeId(0)]
        assert n.generator_type.name == gen
        assert dims(n.inputs) == [D]
        assert dims(n.outputs) == [D]

    def test_binder_and_splitter_shorthands(self) -> None:
        d = dsl("b = B(d, e); s = S(d, e); wire b.out0 -- s.in0")
        b, s = d.nodes[NodeId(0)], d.nodes[NodeId(1)]
        assert (b.generator_type.name, s.generator_type.name) == ("B", "S")
        assert dims(b.inputs) == [D, E]
        assert dims(b.outputs) == [D * E]
        assert dims(s.inputs) == [D * E]
        assert dims(s.outputs) == [D, E]

    def test_binder_numerals_are_abstracted(self) -> None:
        d = dsl("b = B(2, 3)")
        b = d.nodes[NodeId(0)]
        s, t = dims(b.inputs)
        assert not s.is_concrete and not t.is_concrete
        assert s != t
        assert dims(b.outputs) == [s * t]
        assert sorted(d.parameters.values()) == [2, 3]

    def test_port_list_form(self) -> None:
        d = dsl("x = X(in: [d, d], out: [d])")
        n = d.nodes[NodeId(0)]
        assert n.generator_type.name == "X"
        assert dims(n.inputs) == [D, D]
        assert dims(n.outputs) == [D]

    def test_port_list_form_any_generator(self) -> None:
        d = dsl("b = B(in: [d, e], out: [d*e]); w = W(in: [e], out: [])")
        assert dims(d.nodes[NodeId(0)].outputs) == [D * E]
        assert d.nodes[NodeId(1)].generator_type.name == "W"
        assert dims(d.nodes[NodeId(1)].outputs) == []

    def test_port_list_empty_sides(self) -> None:
        d = dsl("a = Z(in: [], out: [d])")
        assert d.boundary_outputs == (o(0, 0),)

    def test_ids_follow_declaration_order(self) -> None:
        d = dsl("c = X(0 -> 1, d); a = Z(1 -> 1, d); b = W(1 -> 0, d); wire c.out0 -- a.in0")
        assert [d.nodes[NodeId(k)].generator_type.name for k in range(3)] == ["X", "Z", "W"]
        assert i(1, 0) not in d.boundary_inputs
        assert d.boundary_inputs == (i(2, 0),)


class TestPhases:
    def test_entries_default_to_node_dim(self) -> None:
        d = dsl("a = Z(1 -> 1, d) phase {1: 1/3, 2: theta}")
        want = PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3)), 2: Phase.symbol("theta")})
        assert d.nodes[NodeId(0)].phase == want

    def test_explicit_phase_dim(self) -> None:
        d = dsl("a = X(1 -> 1, d) phase[d] {1: 1/4}")
        assert d.nodes[NodeId(0)].phase == PhaseVector(D, {1: Phase.turns(sp.Rational(1, 4))})

    def test_phase_zero(self) -> None:
        d = dsl("a = Z(0 -> 2, d) phase 0")
        phase = d.nodes[NodeId(0)].phase
        assert phase is not None
        assert phase == PhaseVector(D)
        assert not phase.entries()

    def test_phase_dim_from_first_port(self) -> None:
        d = dsl("a = Z(in: [e], out: [e, e]) phase {1: 1/2}")
        assert d.nodes[NodeId(0)].phase == PhaseVector(E, {1: Phase.turns(sp.Rational(1, 2))})

    def test_legless_shorthand_takes_its_dim(self) -> None:
        d = dsl("a = Z(0 -> 0, d) phase {1: 1/2}")
        n = d.nodes[NodeId(0)]
        assert n.phase == PhaseVector(D, {1: Phase.turns(sp.Rational(1, 2))})
        assert not n.inputs and not n.outputs

    def test_legless_needs_explicit_dim(self) -> None:
        with pytest.raises(PR.ParseDomainError):
            PR.parse_dsl("a = Z(in: [], out: []) phase {1: 1/2}")
        d = dsl("a = Z(in: [], out: []) phase[d] {1: 1/2}")
        assert d.nodes[NodeId(0)].phase == PhaseVector(D, {1: Phase.turns(sp.Rational(1, 2))})

    def test_root_of_unity_entry(self) -> None:
        d = dsl("a = Z(1 -> 1, d) phase {1: 1/d}")
        assert d.nodes[NodeId(0)].phase == PhaseVector(D, {1: Phase.root_of_unity(1, D)})

    def test_symbolic_expression_entry(self) -> None:
        d = dsl("a = Z(1 -> 1, d) phase {1: theta + 1/2}")
        want = Phase.symbol("theta") + Phase.turns(sp.Rational(1, 2))
        assert d.nodes[NodeId(0)].phase == PhaseVector(D, {1: want})

    def test_numeral_phase_dim_shares_the_port_symbol(self) -> None:
        d = dsl("a = Z(1 -> 1, 3) phase[3] {1: 1/3}")
        n = d.nodes[NodeId(0)]
        assert n.phase is not None
        assert n.phase.dim == n.inputs[0].dim
        (name,) = n.inputs[0].dim.free_symbols
        assert dict(d.parameters) == {name: 3}


class TestWiresAndBoundary:
    def test_ghz_matches_helper(self) -> None:
        src = (
            "a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0; "
            "outputs a.out1, b.out0, b.out1"
        )
        want = build_ghz_with_copy(D)[0]
        assert compare_structure(dsl(src), want).identical

    def test_several_wires_in_one_statement(self) -> None:
        d = dsl(
            "a = Z(0 -> 1, d); b = Z(1 -> 1, d); c = Z(1 -> 0, d); "
            "wire a.out0 -- b.in0, b.out0 -- c.in0"
        )
        assert len(d.wires) == 2
        assert d.boundary_inputs == () and d.boundary_outputs == ()

    def test_self_loop_and_cap(self) -> None:
        d = dsl("a = Z(1 -> 2, d); wire a.out1 -- a.in0; x = X(0 -> 3, d); wire x.out0 -- x.out2")
        assert len(d.wires) == 2
        assert d.boundary_outputs == (o(0, 0), o(1, 1))

    def test_explicit_boundary_order(self) -> None:
        d = dsl("a = Z(1 -> 1, d); b = Z(1 -> 1, d); inputs b.in0, a.in0; outputs b.out0, a.out0")
        assert d.boundary_inputs == (i(1, 0), i(0, 0))
        assert d.boundary_outputs == (o(1, 0), o(0, 0))

    def test_auto_boundary_in_node_then_port_order(self) -> None:
        d = dsl("a = Z(0 -> 2, d); b = Z(2 -> 1, d); wire a.out1 -- b.in1")
        assert d.boundary_inputs == (i(1, 0),)
        assert d.boundary_outputs == (o(0, 0), o(1, 0))

    def test_one_side_explicit_other_auto(self) -> None:
        d = dsl("a = Z(2 -> 2, d); inputs a.in1, a.in0")
        assert d.boundary_inputs == (i(0, 1), i(0, 0))
        assert d.boundary_outputs == (o(0, 0), o(0, 1))

    def test_empty_boundary_statement(self) -> None:
        d = dsl("a = Z(1 -> 0, d); outputs")
        assert d.boundary_outputs == ()
        assert d.boundary_inputs == (i(0, 0),)

    def test_boundary_statement_twice(self) -> None:
        with pytest.raises(PR.ParseError):
            PR.parse_dsl("a = Z(0 -> 1, d); outputs a.out0; outputs a.out0")

    def test_explicit_boundary_missing_a_port(self) -> None:
        with pytest.raises(PR.ParseDomainError):
            PR.parse_dsl("a = Z(0 -> 2, d); outputs a.out0")


class TestScalarStatement:
    def test_scalar(self) -> None:
        assert scalar_of("1/sqrt(d)") == Scalar.dim_power(D, -1, 2)

    def test_scalar_twice(self) -> None:
        with pytest.raises(PR.ParseError):
            PR.parse_dsl("a = Z(0 -> 1, d); scalar 2; scalar 3")

    def test_default_scalar_is_one(self) -> None:
        assert dsl("a = Z(0 -> 1, d)").scalar == Scalar.one()


class TestBindings:
    def test_dim_binding(self) -> None:
        assert dict(dsl("dim d = 3; a = Z(0 -> 1, d)").parameters) == {"d": 3}

    def test_dim_declaration_without_value(self) -> None:
        d = dsl("dim d, e; a = Z(0 -> 1, d); b = Z(0 -> 1, e)")
        assert dict(d.parameters) == {}

    def test_count_binding(self) -> None:
        d = dsl("count n = 4; a = Z(0 -> 1, d); box p x n ports a.out0")
        assert dict(d.parameters) == {"n": 4}
        d = dsl("count n; a = Z(0 -> 1, d); box p x n ports a.out0")
        assert dict(d.parameters) == {}

    def test_let_binding(self) -> None:
        assert dict(dsl("let d = 5; a = Z(0 -> 1, d)").parameters) == {"d": 5}

    @pytest.mark.parametrize(
        "src",
        [
            "dim e = 3; a = Z(0 -> 1, d)",
            "count m = 2; a = Z(0 -> 1, d)",
            "let q = 1; a = Z(0 -> 1, d)",
        ],
    )
    def test_unused_binding_is_a_domain_error(self, src: str) -> None:
        with pytest.raises(PR.ParseDomainError):
            PR.parse_dsl(src)


class TestAbstraction:
    def test_numeral_dims_share_one_symbol(self) -> None:
        d = dsl("a = Z(0 -> 2, 3); b = Z(1 -> 2, 3); wire a.out0 -- b.in0")
        want = build_ghz_with_copy(D)[0]
        want.set_parameters({"d": 3})
        assert compare_structure(d, want).identical

    def test_distinct_numerals_get_distinct_symbols(self) -> None:
        d = dsl("a = Z(0 -> 1, 2); b = Z(0 -> 1, 3)")
        da = d.nodes[NodeId(0)].outputs[0].dim
        db = d.nodes[NodeId(1)].outputs[0].dim
        assert da != db
        (na,) = da.free_symbols
        (nb,) = db.free_symbols
        assert dict(d.parameters) == {na: 2, nb: 3}
        assert na.startswith("d") and nb.startswith("d")

    def test_reuses_a_user_symbol_bound_to_the_value(self) -> None:
        d = dsl("dim e = 3; a = Z(0 -> 1, 3)")
        assert d.nodes[NodeId(0)].outputs[0].dim == E
        assert dict(d.parameters) == {"e": 3}

    def test_minted_name_avoids_source_identifiers(self) -> None:
        d = dsl("d = Z(0 -> 1, 3)")
        dim = d.nodes[NodeId(0)].outputs[0].dim
        assert not dim.is_concrete
        (name,) = dim.free_symbols
        assert name != "d"
        assert dict(d.parameters) == {name: 3}

    def test_literal_dim_stays_concrete(self) -> None:
        d = dsl("a = Z(0 -> 1, literal 3)")
        assert d.nodes[NodeId(0)].outputs[0].dim == Dim(3)
        assert dict(d.parameters) == {}

    def test_counts_are_fresh_per_occurrence(self) -> None:
        d = dsl("a = Z(0 -> 1, d); b = Z(0 -> 1, d); box p x 4 nodes a; box q x 4 nodes b")
        m0 = d.bang_boxes[0].multiplicity  # type: ignore[index]
        m1 = d.bang_boxes[1].multiplicity  # type: ignore[index]
        assert m0.is_bare_symbol and m1.is_bare_symbol
        assert m0 != m1
        assert m0.bare_symbol_name().startswith("n") and m1.bare_symbol_name().startswith("n")
        assert dict(d.parameters) == {m0.bare_symbol_name(): 4, m1.bare_symbol_name(): 4}

    def test_literal_count_stays_concrete(self) -> None:
        d = dsl("a = Z(0 -> 1, d); box p x literal 4 nodes a")
        assert d.bang_boxes[0].multiplicity == Mult(4)  # type: ignore[index]
        assert dict(d.parameters) == {}


class TestBoxes:
    def test_node_box(self) -> None:
        d = dsl("a = Z(0 -> 1, d); b = X(0 -> 1, d); box p x n nodes a, b")
        (box,) = d.bang_boxes.values()
        assert int(box.id) == 0
        assert box.multiplicity == Mult("n")
        assert box.node_scope == frozenset({NodeId(0), NodeId(1)})
        assert box.parent is None

    def test_port_box(self) -> None:
        d = dsl("a = Z(1 -> 2, d); box p x n ports a.out1, a.in0")
        (box,) = d.bang_boxes.values()
        assert box.port_scope == frozenset({o(0, 1), i(0, 0)})
        assert not box.node_scope

    def test_nested_boxes(self) -> None:
        d = dsl(
            "a = Z(0 -> 2, d); b = Z(1 -> 1, d); wire a.out0 -- b.in0; "
            "box outer x n nodes a, b; box inner x m nodes b in outer; "
            "box legs x k ports a.out1 in outer"
        )
        boxes = d.bang_boxes
        assert [int(b) for b in sorted(boxes)] == [0, 1, 2]
        assert boxes[1].parent == 0  # type: ignore[index]
        assert boxes[2].parent == 0  # type: ignore[index]
        assert boxes[1].node_scope == frozenset({NodeId(1)})  # type: ignore[index]
        assert boxes[2].port_scope == frozenset({o(0, 1)})  # type: ignore[index]

    def test_parent_declared_later(self) -> None:
        d = dsl(
            "a = Z(0 -> 2, d); b = Z(1 -> 1, d); wire a.out0 -- b.in0; "
            "box inner x m nodes b in outer; box outer x n nodes a, b"
        )
        inner, outer = d.bang_boxes[0], d.bang_boxes[1]  # type: ignore[index]
        assert inner.multiplicity == Mult("m")
        assert inner.parent == 1
        assert outer.parent is None
        assert outer.node_scope == frozenset({NodeId(0), NodeId(1)})

    def test_mult_expressions(self) -> None:
        d = dsl("a = Z(0 -> 2, d); box p x 2*k nodes a; box q x k + 1 ports a.out0 in p")
        assert d.bang_boxes[0].multiplicity == Mult("k") * 2  # type: ignore[index]
        assert d.bang_boxes[1].multiplicity == Mult("k") + 1  # type: ignore[index]

    def test_mult_product(self) -> None:
        d = dsl("a = Z(0 -> 1, d); box p x n*m nodes a")
        assert d.bang_boxes[0].multiplicity == Mult("n") * Mult("m")  # type: ignore[index]

    @pytest.mark.parametrize(
        "src",
        [
            "a = Z(0 -> 1, d); box p x n nodes a in nowhere",
            "a = Z(0 -> 1, d); box p x n nodes zz",
            "a = Z(0 -> 1, d); box p x n ports a.out5",
            "a = Z(0 -> 1, d); box p x n nodes a; box p x m nodes a",
        ],
    )
    def test_bad_box_references(self, src: str) -> None:
        with pytest.raises(PR.ParseError):
            PR.parse_dsl(src)


# -- expressions ------------------------------------------------------------------------------


class TestExpressions:
    @pytest.mark.parametrize(
        ("expr", "want"),
        [
            ("2^3", 8),
            ("2**3", 8),
            ("-2^2", -4),
            ("(-2)^2", 4),
            ("2^3^2", 512),
            ("1 - 2 - 3", -4),
            ("12/2/3", 2),
            ("2*3 + 4", 10),
            ("2*(3 + 4)", 14),
            ("-(3)", -3),
            ("sqrt(4)", 2),
            ("exp(I*pi)", -1),
            ("exp(I*π)", -1),
            ("I^2", -1),
        ],
    )
    def test_arithmetic(self, expr: str, want: int) -> None:
        assert scalar_of(expr) == Scalar.rational(want)

    def test_gaussian(self) -> None:
        assert scalar_of("(1 + 2*I)/2") == Scalar.gaussian_rational(sp.Rational(1, 2), 1)

    def test_e_constant(self) -> None:
        assert scalar_of("E") == Scalar(sp.E)

    def test_omega(self) -> None:
        assert scalar_of("exp(2*I*pi/d)") == Scalar.omega(D)

    def test_identifier_roles_in_scalar(self) -> None:
        assert scalar_of("c") == Scalar.symbol("c")
        assert scalar_of("d") == Scalar.from_dim(D)
        boxed = "a = Z(0 -> 1, d); box p x n nodes a"
        assert scalar_of("n", boxed) == Scalar(Mult("n").to_sympy())

    def test_exponent_symbol_in_dim(self) -> None:
        d = dsl("a = Z(1 -> 1, d^m)")
        assert d.nodes[NodeId(0)].inputs[0].dim == D ** Dim.symbol("m")
        exp_m = sp.Symbol("m", integer=True, nonnegative=True)
        assert scalar_of("2^m", "a = Z(1 -> 1, d^m)") == Scalar(sp.Integer(2) ** exp_m)

    def test_dim_product(self) -> None:
        d = dsl("a = Z(1 -> 1, d*e)")
        assert d.nodes[NodeId(0)].inputs[0].dim == D * E

    def test_phase_symbol_is_real(self) -> None:
        d = dsl("a = Z(1 -> 1, d) phase {1: theta}")
        assert d.nodes[NodeId(0)].phase == PhaseVector(D, {1: Phase.symbol("theta")})


# -- graph listing ----------------------------------------------------------------------------


LISTING = """\
graph: 3 nodes, 1 wire, 0 inputs -> 3 outputs
  scalar: 1/sqrt(d)
  parameters: d = 3
  nodes:
    n0  Z  in: []  out: [d, d!0]  phase[d]: 0
    n1  X  in: [d]  out: [d]  phase[d]: {1: 1/3, 2: theta} turns  boxes: !1
    n2  Z  in: []  out: [d]  phase[d]: 0  boxes: !1 > !2
  wires:
    n0.out0 -- n1.in0
  inputs: []
  outputs: [n0.out1, n1.out0, n2.out0]
  bang boxes:
    !0  x n  ports: n0.out1
    !1  x m  nodes: n1, n2
      !2  x k  nodes: n2
"""


class TestListing:
    def test_hand_written_listing(self) -> None:
        d = PR.parse_listing(LISTING)
        assert validate(d).is_valid
        assert [d.nodes[NodeId(k)].generator_type.name for k in range(3)] == ["Z", "X", "Z"]
        assert d.nodes[NodeId(0)].phase == PhaseVector(D)
        assert d.nodes[NodeId(1)].phase == PhaseVector(
            D, {1: Phase.turns(sp.Rational(1, 3)), 2: Phase.symbol("theta")}
        )
        assert d.scalar == Scalar.dim_power(D, -1, 2)
        assert dict(d.parameters) == {"d": 3}
        assert d.boundary_outputs == (o(0, 1), o(1, 0), o(2, 0))
        boxes = d.bang_boxes
        assert boxes[0].port_scope == frozenset({o(0, 1)})  # type: ignore[index]
        assert boxes[1].node_scope == frozenset({NodeId(1), NodeId(2)})  # type: ignore[index]
        assert boxes[2].parent == 1  # type: ignore[index]
        assert render_graph(d) == LISTING.rstrip("\n")

    def test_ids_with_gaps_are_preserved(self) -> None:
        text = (
            "graph: 2 nodes, 1 wire, 0 inputs -> 2 outputs\n"
            "  scalar: 1\n"
            "  parameters: none\n"
            "  nodes:\n"
            "    n2  Z  in: []  out: [d, d]\n"
            "    n5  Z  in: [d]  out: [d!3]\n"
            "  wires:\n"
            "    n2.out0 -- n5.in0\n"
            "  inputs: []\n"
            "  outputs: [n2.out1, n5.out0]\n"
            "  bang boxes:\n"
            "    !3  x n  ports: n5.out0\n"
        )
        d = PR.parse_listing(text)
        assert sorted(int(k) for k in d.nodes) == [2, 5]
        assert sorted(int(k) for k in d.bang_boxes) == [3]
        assert d.nodes[NodeId(2)].phase is None
        assert render_graph(d) == text.rstrip("\n")

    def test_listing_matches_dsl(self) -> None:
        src = "a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0"
        d = dsl(src)
        assert compare_structure(PR.parse_listing(render_graph(d)), d).identical

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("graph: 3 nodes", "graph: 4 nodes"),
            ("1 wire,", "2 wires,"),
            ("0 inputs ->", "1 input ->"),
            ("-> 3 outputs", "-> 2 outputs"),
        ],
    )
    def test_header_count_mismatch(self, old: str, new: str) -> None:
        with pytest.raises(PR.ParseGrammarError):
            PR.parse_listing(LISTING.replace(old, new))

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("out: [d, d!0]", "out: [d, d]"),
            ("out: [d, d!0]", "out: [d!0, d!0]"),
            ("out: [d]  phase[d]: {1", "out: [d!0]  phase[d]: {1"),
            ("boxes: !1 > !2", "boxes: !1"),
            ("turns  boxes: !1", "turns"),
            ("out: [d, d!0]", "out: [d, d!1]"),
        ],
    )
    def test_inconsistent_marks(self, old: str, new: str) -> None:
        assert old in LISTING
        with pytest.raises(PR.ParseGrammarError):
            PR.parse_listing(LISTING.replace(old, new, 1))

    @pytest.mark.parametrize(
        "text",
        [
            "graph:",
            "graph: 1 node, 0 wires, 0 inputs -> 0 outputs",
            LISTING.replace("    n0.out0 -- n1.in0", "    n0.out0 -> n1.in0"),
            LISTING.replace("  Z  in: []  out: [d, d!0]", "  Q  in: []  out: [d, d!0]"),
            LISTING.replace("n0.out0 -- n1.in0", "n0.out0 -- n9.in0"),
            LISTING.replace("!2  x k  nodes: n2", "!2  x k  nodes: n7"),
        ],
    )
    def test_malformed_listing(self, text: str) -> None:
        with pytest.raises(PR.ParseError):
            PR.parse_listing(text)


# -- steps ------------------------------------------------------------------------------------


def ghz() -> Diagram:
    return build_ghz_with_copy(D)[0]


class TestSteps:
    def test_default(self) -> None:
        step = PR.parse_step("apply spider_fusion")
        assert isinstance(step, PR.RuleStep)
        assert (step.rule, step.index, step.nodes) == ("spider_fusion", None, None)
        assert "apply" in step.source

    def test_index(self) -> None:
        step = PR.parse_step("apply spider_fusion at 1")
        assert (step.index, step.nodes) == (1, None)

    def test_nodes(self) -> None:
        step = PR.parse_step("apply spider_fusion at n0, n1")
        assert step.index is None
        assert step.nodes == (0, 1)
        assert PR.parse_step("apply spider_fusion at n3").nodes == (3,)

    @pytest.mark.parametrize("rule", sorted(RULES))
    def test_every_library_rule_parses(self, rule: str) -> None:
        assert PR.parse_step(f"apply {rule}").rule == rule

    def test_unknown_rule(self) -> None:
        with pytest.raises(PR.ParseDomainError):
            PR.parse_step("apply no_such_rule")

    @pytest.mark.parametrize(
        "src",
        [
            "apply",
            "apply spider_fusion at",
            "apply spider_fusion at x",
            "apply spider_fusion at n0,",
            "apply spider_fusion at -1",
            "apply spider_fusion at 0 1",
            "apply spider_fusion now",
        ],
    )
    def test_malformed_step(self, src: str) -> None:
        with pytest.raises(PR.ParseError):
            PR.parse_step(src)

    def test_apply_default(self) -> None:
        pre = ghz()
        result = PR.apply_step(pre, PR.parse_step("apply spider_fusion"))
        assert isinstance(result, RewriteResult)
        assert result.step.rule_name == SPIDER_FUSION.name
        assert len(result.diagram.nodes) == 1

    def test_apply_index_and_nodes(self) -> None:
        pre = ghz()
        by_index = PR.apply_step(pre, PR.parse_step("apply spider_fusion at 0"))
        by_nodes = PR.apply_step(pre, PR.parse_step("apply spider_fusion at n1, n0"))
        assert compare_structure(by_index.diagram, by_nodes.diagram).identical

    def test_apply_leaves_input_untouched(self) -> None:
        pre = ghz()
        PR.apply_step(pre, PR.parse_step("apply spider_fusion"))
        assert compare_structure(pre, ghz()).identical

    @pytest.mark.parametrize(
        "src",
        [
            "apply spider_fusion at 1",
            "apply spider_fusion at n0",
            "apply spider_fusion at n0, n7",
            "apply hopf",
        ],
    )
    def test_no_match(self, src: str) -> None:
        with pytest.raises(PR.ParseDomainError):
            PR.apply_step(ghz(), PR.parse_step(src))

    def test_no_match_on_single_node(self) -> None:
        with pytest.raises(PR.ParseDomainError):
            PR.apply_step(dsl("a = Z(0 -> 1, d)"), PR.parse_step("apply spider_fusion"))


# -- dispatch ---------------------------------------------------------------------------------


class TestDispatch:
    @pytest.mark.parametrize(
        ("src", "form"),
        [
            ("a = Z(0 -> 1, d)", "DSL"),
            ("  a = Z(0 -> 1, d)", "DSL"),
            ("dim d; a = Z(0 -> 1, d)", "DSL"),
            ("count n; a = Z(0 -> 1, d); box p x n nodes a", "DSL"),
            ("let d = 2; a = Z(0 -> 1, d)", "DSL"),
            ("scalar 2; a = Z(0 -> 1, d)", "DSL"),
            ("sum_{k=0}^{d-1} |k,k>", "DIRAC"),
            ("sum_{k=0}^{d-1} |k,k>; copy", "DIRAC"),
            ("1/2", "DIRAC"),
            ("\n" + render_graph(build_ghz_with_copy(D)[0]), "LISTING"),
        ],
    )
    def test_forms(self, src: str, form: str) -> None:
        parsed = PR.parse(src)
        assert isinstance(parsed, PR.ParsedDiagram)
        assert parsed.form is PR.InputForm[form]
        assert validate(parsed.diagram).is_valid

    def test_step(self) -> None:
        step = PR.parse("apply spider_fusion at 0")
        assert isinstance(step, PR.RuleStep)
        assert step == PR.parse_step("apply spider_fusion at 0")

    def test_scalar_only_dirac(self) -> None:
        parsed = PR.parse("1/2")
        assert isinstance(parsed, PR.ParsedDiagram)
        assert not parsed.diagram.nodes
        assert parsed.diagram.scalar == Scalar.rational(1, 2)

    def test_parse_diagram_rejects_a_step(self) -> None:
        with pytest.raises(PR.ParseGrammarError):
            PR.parse_diagram("apply spider_fusion")

    def test_parse_diagram_agrees_with_parse(self) -> None:
        src = "a = Z(0 -> 1, d) phase {1: theta}"
        parsed = PR.parse(src)
        assert isinstance(parsed, PR.ParsedDiagram)
        assert compare_structure(PR.parse_diagram(src), parsed.diagram).identical

    def test_input_forms(self) -> None:
        assert {f.name for f in PR.InputForm} == {"DIRAC", "DSL", "LISTING", "STEP"}


# -- errors -----------------------------------------------------------------------------------


BAD = [
    ("", GRAMMAR),
    ("   ", GRAMMAR),
    ("a = ", GRAMMAR),
    ("a = Z(", GRAMMAR),
    ("a = Z()", GRAMMAR),
    ("a = Z(0 -> 1)", GRAMMAR),
    ("a = Z(0 -> 1, d", GRAMMAR),
    ("a = Z(0 -> 1, d))", GRAMMAR),
    ("a = Q(0 -> 1, d)", GRAMMAR),
    ("a = Z(0 => 1, d)", GRAMMAR),
    ("a = Z(-1 -> 1, d)", GRAMMAR),
    ("a = Z(0 -> 99999, d)", DOMAIN),
    ("a = Z(0 -> 1, d); a = Z(0 -> 1, d)", GRAMMAR),
    ("wire = Z(0 -> 1, d)", GRAMMAR),
    ("dim = Z(0 -> 1, d)", GRAMMAR),
    ("a = Z(0 -> 1, d); wire a.out0 -- b.in0", GRAMMAR),
    ("a = Z(0 -> 1, d); wire a.out0 --", GRAMMAR),
    ("a = Z(0 -> 1, d); wire a.out0 -- a.out0", DOMAIN),
    ("a = Z(0 -> 1, d); wire a.out3 -- a.in0", DOMAIN),
    ("a = Z(0 -> 1, d); wire a.side0 -- a.in0", GRAMMAR),
    ("a = Z(0 -> 1, d); outputs a.out0, a.out0", DOMAIN),
    ("a = Z(1 -> 1, d); b = Z(0 -> 1, d); wire b.out0 -- a.in0; inputs a.in0", DOMAIN),
    ("a = Z(0 -> 1, d) phase {0: 1/2}", DOMAIN),
    ("a = Z(0 -> 1, d) phase {1 1/2}", GRAMMAR),
    ("a = Z(0 -> 1, d) phase {1: }", GRAMMAR),
    ("a = Z(0 -> 1, d) phase {1: 1/2, 1: 1/3}", GRAMMAR),
    ("a = Z(0 -> 1, d) phase[d {1: 1/2}", GRAMMAR),
    ("a = F(d) phase {1: 1/2}", DOMAIN),
    ("a = Z(0 -> 1, d); scalar", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar 1/0", DOMAIN),
    ("a = Z(0 -> 1, d); scalar 2 +", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar (2", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar 2)", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar __import__('os')", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar lambda: 1", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar 1e5", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar 0x10", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar 2 @ 3", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar a.b", GRAMMAR),
    ("a = Z(0 -> 1, 0)", DOMAIN),
    ("a = Z(0 -> 1, literal 0)", DOMAIN),
    ("a = Z(0 -> 1, -3)", GRAMMAR),
    ("a = Z(0 -> 1, d/2)", GRAMMAR),
    ("a = Z(0 -> 1, ٣)", GRAMMAR),
    ("a = Z(0 -> 1, ³)", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar ١", GRAMMAR),
    ("a = Z(0 -> 1, d); box p x -1 nodes a", GRAMMAR),
    ("a = Z(0 -> 1, d); box p x n/2 nodes a", GRAMMAR),
    ("a = Z(0 -> 1, d); box p x n", GRAMMAR),
    ("a = Z(0 -> 1, d); box p nodes a", GRAMMAR),
    ("a = Z(0 -> 1, d); box p x n nodes", GRAMMAR),
    ("a = Z(0 -> 1, d); box p x n nodes a in p", DOMAIN),
    ("a = Z(0 -> 1, d); box p x n nodes a in q; box q x m nodes a in p", DOMAIN),
    ("dim d = 0; a = Z(0 -> 1, d)", DOMAIN),
    ("dim d = x; a = Z(0 -> 1, d)", GRAMMAR),
    ("count n = -1; a = Z(0 -> 1, d); box p x n nodes a", DOMAIN),
    ("dim d = 3; dim d = 4; a = Z(0 -> 1, d)", GRAMMAR),
    ("a = Z(0 -> 1, d) phase {1: 1/2} extra", GRAMMAR),
    ("sum_{k=0}^{k-1} |k>", DOMAIN),
    ("sum_{k=0}^{d-1} |k>^{0}", DOMAIN),
    ("Z_phase", GRAMMAR),
    ("sum_{k=0}^{d-1} |k[i1]>", GRAMMAR),
    ("apply", GRAMMAR),
    ("graph: 1 node", GRAMMAR),
    ("(" * 5000 + "1" + ")" * 5000, GRAMMAR),
    ("a = Z(0 -> 1, d); scalar " + "(" * 5000 + "1" + ")" * 5000, GRAMMAR),
    ("a = Z(0 -> 1, d); scalar " + "c^" * 5000 + "c", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar " + "exp(" * 5000 + "0" + ")" * 5000, GRAMMAR),
    ("a = Z(0 -> 1, " + "(" * 5000 + "d" + ")" * 5000 + ")", GRAMMAR),
    ("\x00", GRAMMAR),
    ("a = Z(0 -> 1, d)\x00", GRAMMAR),
    ("a = Z(0 -> 1, d); scalar ∞", GRAMMAR),
]

FUZZ = [
    ";;;",
    "# only a comment",
    "a = Z(0 -> 1, d); scalar " + "-" * 5000 + "1",
    "a = Z(0 -> 1, d) # " + "(" * 5000,
    "sum_{k=0}^{d-1} " + "|k>" * 2000,
    "a = Z(0 -> 1, d); " * 300,
]

PARSERS = ("parse", "parse_diagram", "parse_dirac", "parse_dsl", "parse_listing", "parse_step")


@pytest.mark.parametrize(("src", "cls"), BAD, ids=range(len(BAD)))
def test_bad_input_raises_its_error_class(src: str, cls: type[PR.ParseError]) -> None:
    with pytest.raises(cls):
        PR.parse(src)


@pytest.mark.parametrize("src", [s for s, _ in BAD] + FUZZ, ids=range(len(BAD) + len(FUZZ)))
def test_bad_input_raises_only_parse_errors_everywhere(src: str) -> None:
    for name in PARSERS:
        try:
            getattr(PR, name)(src)
        except PR.ParseError:
            pass
        except RecursionError:  # pragma: no cover
            pytest.fail(f"{name} hit the recursion limit on {src[:40]!r}")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{name} raised {type(exc).__name__} on {src[:40]!r}: {exc}")


@pytest.mark.parametrize("value", [None, 3, b"a = Z(0 -> 1, d)", ["a"]])
def test_non_string_input(value: object) -> None:
    for name in PARSERS:
        with pytest.raises(PR.ParseError):
            getattr(PR, name)(value)


@pytest.mark.parametrize("value", [None, 3])
def test_apply_step_bad_arguments(value: object) -> None:
    with pytest.raises(PR.ParseError):
        PR.apply_step(ghz(), value)  # type: ignore[arg-type]
    with pytest.raises(PR.ParseError):
        PR.apply_step(value, PR.parse_step("apply spider_fusion"))  # type: ignore[arg-type]


def test_deep_nesting_is_a_grammar_error() -> None:
    with pytest.raises(PR.ParseGrammarError):
        PR.parse("a = Z(0 -> 1, d); scalar " + "(" * 5000 + "1" + ")" * 5000)
    with pytest.raises(PR.ParseGrammarError):
        PR.parse_dirac("(" * 5000 + "1" + ")" * 5000)


class TestValidationFailures:
    def test_dimension_mismatch_wire(self) -> None:
        with pytest.raises(PR.ParseDomainError):
            PR.parse_dsl("a = Z(0 -> 1, literal 2); b = Z(1 -> 0, literal 3); wire a.out0 -- b.in0")

    def test_generator_leg_policy(self) -> None:
        with pytest.raises(PR.ParseDomainError):
            PR.parse_dsl("a = F(in: [d, d], out: [d])")

    def test_unused_port_left_off_boundary(self) -> None:
        with pytest.raises(PR.ParseDomainError):
            PR.parse_dsl("a = Z(1 -> 2, d); inputs")

    def test_parameter_for_unknown_symbol_in_listing(self) -> None:
        text = render_graph(ghz()).replace("parameters: none", "parameters: zz = 3")
        with pytest.raises(PR.ParseDomainError):
            PR.parse_listing(text)


# -- dispatch and statements (fix list C) ----------------------------------------------------


def test_self_wire_is_a_dsl_domain_error() -> None:
    src = "a = Z(0 -> 1, d); wire a.out0 -- a.out0"
    with pytest.raises(PR.ParseDomainError):
        PR.parse(src)
    with pytest.raises(PR.ParseDomainError):
        PR.parse_dsl(src)


GHZ_DSL = "a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0; outputs a.out1, b.out0, b.out1"


@pytest.mark.parametrize(
    "src",
    [
        "wire a.out0 -- b.in0; a = Z(0 -> 2, d); b = Z(1 -> 2, d)",
        "outputs a.out1, b.out0, b.out1; a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0",
        "inputs; a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0",
        "box p x literal 1 nodes a; a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0",
    ],
)
def test_first_keyword_dispatches_to_dsl(src: str) -> None:
    parsed = PR.parse(src)
    assert isinstance(parsed, PR.ParsedDiagram)
    assert parsed.form is PR.InputForm.DSL
    assert compare_structure(parsed.diagram.copy(), dsl(src)).identical


def test_statement_order_is_free() -> None:
    want = ghz()
    for src in (
        "outputs a.out1, b.out0, b.out1; wire a.out0 -- b.in0; a = Z(0 -> 2, d); b = Z(1 -> 2, d)",
        "a = Z(0 -> 2, d); wire a.out0 -- b.in0; b = Z(1 -> 2, d)",
        "scalar 1; wire a.out0 -- b.in0; dim d; a = Z(0 -> 2, d); b = Z(1 -> 2, d)",
    ):
        assert compare_structure(dsl(src), want).identical, src
    a = dsl("box q x m nodes b in p; a = Z(0 -> 2, d); box p x n nodes a, b; b = Z(1 -> 1, d)")
    b = dsl("a = Z(0 -> 2, d); b = Z(1 -> 1, d); box q x m nodes b in p; box p x n nodes a, b")
    assert compare_structure(a, b).identical
    assert dict(a.bang_boxes) == dict(b.bang_boxes)


@pytest.mark.parametrize("src", ["", "   ", ";;;", "# only", " ; # x\n;\n", "\t\n"])
def test_empty_program_is_a_grammar_error(src: str) -> None:
    with pytest.raises(PR.ParseGrammarError):
        PR.parse_dsl(src)
    with pytest.raises(PR.ParseGrammarError):
        PR.parse(src)


def test_let_binding_is_reused_like_dim() -> None:
    d = dsl("let e = 3; a = Z(0 -> 1, 3)")
    assert d.nodes[NodeId(0)].outputs[0].dim == E
    assert dict(d.parameters) == {"e": 3}


@pytest.mark.parametrize(
    "src",
    [
        "a = Z(0 -> 1, d);outputs\ta.out0",
        "a = Z(0 -> 1, d); b = Z(1 -> 0, d); wire\ta.out0 -- b.in0",
        "dim\td = 3; a = Z(0 -> 1, d)",
        "a = Z(0 -> 1, d); scalar\t2",
        "a = Z(0 -> 1, d); box\tp x n nodes a",
    ],
)
def test_keywords_split_on_any_whitespace(src: str) -> None:
    assert dsl(src).nodes


RESERVED = [
    "dim", "count", "let", "wire", "inputs", "outputs", "scalar", "box", "literal", "apply",
    "I", "E", "pi", "exp", "sqrt",
]  # fmt: skip


@pytest.mark.parametrize("name", RESERVED)
def test_reserved_node_names(name: str) -> None:
    with pytest.raises(PR.ParseGrammarError):
        PR.parse_dsl(f"x = Z(0 -> 1, d); {name} = Z(0 -> 1, d)")


@pytest.mark.parametrize("name", RESERVED)
def test_reserved_box_names(name: str) -> None:
    with pytest.raises(PR.ParseGrammarError):
        PR.parse_dsl(f"a = Z(0 -> 1, d); box {name} x n nodes a")


def test_count_binding_does_not_capture_a_numeral_count() -> None:
    with pytest.raises(PR.ParseDomainError):
        PR.parse_dsl("count n = 4; a = Z(0 -> 1, d); box p x 4 nodes a")


@pytest.mark.parametrize(
    "src",
    [
        "a = Z(0 -> 1, 3) phase {3: 1/2}",
        "a = Z(0 -> 1, 3) phase {5: 1/2}",
        "dim d = 3; a = Z(0 -> 1, d) phase {3: 1/2}",
        "let d = 2; a = Z(0 -> 1, d) phase {2: 1/2}",
        "a = Z(0 -> 1, literal 3) phase {3: 1/2}",
        "a = Z(0 -> 1, d) phase[3] {4: 1/2}",
    ],
)
def test_phase_index_beyond_concrete_dim(src: str) -> None:
    with pytest.raises(PR.ParseDomainError):
        PR.parse_dsl(src)


def test_phase_index_within_dim() -> None:
    dsl("a = Z(0 -> 1, 3) phase {2: 1/2}")
    dsl("a = Z(0 -> 1, d) phase {9: 1/2}")


def test_leg_bound() -> None:
    d = dsl("a = Z(0 -> 1024, d)")
    assert d.nodes[NodeId(0)].num_outputs == 1024
    with pytest.raises(PR.ParseDomainError):
        PR.parse_dsl("a = Z(0 -> 1025, d)")
    with pytest.raises(PR.ParseDomainError):
        PR.parse_dsl("a = X(1025 -> 0, d)")


def test_minted_dim_avoids_every_source_identifier() -> None:
    d = dsl("; ".join(f"{n} = Z(0 -> 1, 3)" for n in ("d", "d0", "d1", "d2", "d3")))
    assert {p.dim for n in d.nodes.values() for p in n.outputs} == {Dim.symbol("d4")}
    assert dict(d.parameters) == {"d4": 3}


def test_minted_counts_avoid_source_identifiers() -> None:
    d = dsl(
        "n = Z(0 -> 1, d); n0 = Z(0 -> 1, d); n1 = Z(0 -> 1, d); "
        "box p x 4 nodes n; box q x 5 nodes n0; box r x 4 nodes n1"
    )
    assert [str(d.bang_boxes[b].multiplicity) for b in sorted(d.bang_boxes)] == ["n2", "n3", "n4"]
    assert dict(d.parameters) == {"n2": 4, "n3": 5, "n4": 4}


def test_phases_and_scalars_resolve_after_the_whole_input() -> None:
    d = dsl("scalar d*n; a = Z(0 -> 1, e) phase {1: n}; b = Z(0 -> 1, d); box p x n nodes b")
    n = Mult("n").to_sympy()
    assert d.scalar == Scalar(D.to_sympy() * n)
    phase = d.nodes[NodeId(0)].phase
    assert phase is not None
    assert phase.entries()[1].to_sympy_turns() == n


# -- expressions (fix list A) -----------------------------------------------------------------


def test_nineteen_digit_numerals_are_accepted() -> None:
    big = 1234567890123456789
    d = dsl(f"a = Z(0 -> 1, {big})")
    assert sorted(d.parameters.values()) == [big]
    assert scalar_of(str(big)) == Scalar.rational(big)


def test_expression_numeral_cap() -> None:
    assert scalar_of("1" + "0" * 999) == Scalar.rational(10**999)
    with pytest.raises(PR.ParseDomainError):
        PR.parse_dsl("a = Z(0 -> 1, d); scalar 1" + "0" * 1000)


@pytest.mark.parametrize("name", ["I", "E", "pi", "exp", "sqrt", "δ", "θ_1"])
def test_names_in_dims_and_mults_are_symbols(name: str) -> None:
    d = dsl(f"a = Z(0 -> 1, {name})")
    assert d.nodes[NodeId(0)].outputs[0].dim == Dim.symbol(name)
    m = dsl(f"a = Z(0 -> 1, d); box p x {name} nodes a")
    assert m.bang_boxes[0].multiplicity == Mult(name)  # type: ignore[index]
    e = dsl(f"a = Z(1 -> 1, d^{name})")
    assert e.nodes[NodeId(0)].inputs[0].dim == D ** Dim.symbol(name)


def test_unicode_scalar_and_phase_symbols() -> None:
    assert scalar_of("α") == Scalar.symbol("α")
    d = dsl("a = Z(0 -> 1, d) phase {1: θ}")
    assert d.nodes[NodeId(0)].phase == PhaseVector(D, {1: Phase.symbol("θ")})


def test_polynomial_exponent_dim() -> None:
    d = dsl("a = Z(1 -> 1, d^(n^2))")
    exp_n = sp.Symbol("n", integer=True, nonnegative=True)
    assert d.nodes[NodeId(0)].inputs[0].dim == Dim.from_sympy(D.to_sympy() ** exp_n**2)


def test_exponent_naming_a_mult_uses_the_mult_symbol() -> None:
    d = dsl("a = Z(1 -> 1, d^n); box p x n nodes a")
    assert Mult("n").to_sympy() in d.nodes[NodeId(0)].inputs[0].dim.to_sympy().free_symbols


DOS = [
    "a = Z(0 -> 1, d); scalar (2*I)^(10^17)",
    "a = Z(0 -> 1, d); scalar sqrt(2)^(10^17)",
    "a = Z(0 -> 1, d); scalar (a+b+c+d+e+f)^31",
    "a = Z(0 -> 1, d); scalar " + "*".join(f"(a{k}+b{k})" for k in range(14)),
    "a = Z(0 -> 1, d); scalar 2^8000*2^8000",
    "a = Z(0 -> 1, d); scalar (10^300)^(1/2)",
    "a = Z(0 -> 1, d) phase {1: (w+x+y+z)^40}",
    "a = Z(0 -> 1, (2*d)^(10^17))",
    "(2*I)^(10^17)",
    "sqrt(2)^(10^17)",
    "(a+b+c+d+e+f)^31",
    "2^8000*2^8000",
]


@pytest.mark.parametrize("src", DOS, ids=range(len(DOS)))
def test_expression_blowup_is_a_fast_domain_error(src: str) -> None:
    outcome, _ = bounded(lambda: PR.parse(src), 2.0)
    assert isinstance(outcome, PR.ParseDomainError), outcome


@pytest.mark.parametrize("terms", [1000, 5000])
def test_long_sums_finish_fast(terms: int) -> None:
    src = "a = Z(0 -> 1, d); scalar " + " + ".join(f"x{k}" for k in range(terms))
    outcome, _ = bounded(lambda: PR.parse_dsl(src), 4.0)
    if terms == 1000:
        assert isinstance(outcome, Diagram), outcome
    else:
        assert isinstance(outcome, (Diagram, PR.ParseError)), outcome


# -- listing (fix list B) ---------------------------------------------------------------------


HUGE_ID_LISTING = """\
graph: 1 node, 0 wires, 0 inputs -> 1 output
  scalar: 1
  parameters: none
  nodes:
    n999999999999  Z  in: []  out: [d!999999999998]  boxes: !999999999999
  wires: none
  inputs: []
  outputs: [n999999999999.out0]
  bang boxes:
    !999999999999  x m  nodes: n999999999999
      !999999999998  x n  ports: n999999999999.out0"""


def test_huge_listing_ids_parse_fast() -> None:
    outcome, _ = bounded(lambda: PR.parse_listing(HUGE_ID_LISTING), 2.0)
    assert isinstance(outcome, Diagram), outcome
    assert [int(k) for k in outcome.nodes] == [999999999999]
    assert sorted(int(k) for k in outcome.bang_boxes) == [999999999998, 999999999999]
    assert render_graph(outcome) == HUGE_ID_LISTING


@pytest.mark.parametrize("tail", ["garbage", "  extra: 1", "nodes:", "    !9  x n  nodes: n0"])
def test_trailing_listing_text(tail: str) -> None:
    text = render_graph(ghz())
    with pytest.raises(PR.ParseGrammarError):
        PR.parse_listing(text + "\n" + tail)


def test_trailing_blank_lines_are_accepted() -> None:
    assert compare_structure(PR.parse_listing(render_graph(ghz()) + "\n\n"), ghz()).identical


def test_unicode_listing_parameters() -> None:
    d = dsl("dim δ = 3; a = Z(0 -> 1, δ)")
    assert dict(PR.parse_listing(render_graph(d)).parameters) == {"δ": 3}


def test_listing_parameter_numeral_cap() -> None:
    big = 10**999
    d = dsl(f"a = Z(0 -> 1, {big})")
    back = PR.parse_listing(render_graph(d))
    assert sorted(back.parameters.values()) == [big]


@pytest.mark.parametrize("source", [LISTING, render_graph(ghz())], ids=["hand", "ghz"])
def test_every_listing_prefix_parses_or_raises_parse_error(source: str) -> None:
    for end in range(len(source) + 1):
        try:
            diagram = PR.parse_listing(source[:end])
        except PR.ParseError:
            continue
        assert validate(diagram).is_valid, source[:end]


DSL_PREFIX_SOURCES = [
    GHZ_DSL + "; box p x 2*k nodes a; scalar 1/sqrt(d)",
    "dim e = 3; a = X(in: [d, e], out: [d*e]) phase[d] {1: theta}; b = B(d, e); count n",
]


@pytest.mark.parametrize("source", DSL_PREFIX_SOURCES, ids=range(len(DSL_PREFIX_SOURCES)))
def test_every_dsl_prefix_parses_or_raises_parse_error(source: str) -> None:
    for end in range(len(source) + 1):
        try:
            diagram = PR.parse_dsl(source[:end])
        except PR.ParseError:
            continue
        assert validate(diagram).is_valid, source[:end]


# -- steps (fix list E) -----------------------------------------------------------------------


CHAIN = (
    "a = Z(0 -> 1, d); b = Z(1 -> 1, d); c = Z(1 -> 0, d); wire a.out0 -- b.in0, b.out0 -- c.in0"
)


def test_match_selection_on_a_chain() -> None:
    pre = dsl(CHAIN)
    matches = SPIDER_FUSION.pattern.find_matches(pre)
    assert len(matches) == 2
    default = PR.apply_step(pre, PR.parse_step("apply spider_fusion")).diagram
    second = PR.apply_step(pre, PR.parse_step("apply spider_fusion at 1")).diagram
    by_nodes = PR.apply_step(pre, PR.parse_step("apply spider_fusion at n2, n1")).diagram
    want = apply(pre.copy(), SPIDER_FUSION, matches[1]).diagram
    assert set(matches[1].support_node_ids) == {NodeId(1), NodeId(2)}
    assert not compare_structure(second, default).identical
    assert compare_structure(second, by_nodes).identical
    assert compare_structure(second, want).identical
    with pytest.raises(PR.ParseDomainError):
        PR.apply_step(pre, PR.parse_step("apply spider_fusion at 2"))


@pytest.mark.parametrize(
    ("args", "label"),
    [
        (("spider_fusion", -1, None), "negative index"),
        (("spider_fusion", True, None), "bool index"),
        (("spider_fusion", 0, (0, 1)), "both selectors"),
        (("spider_fusion", None, (0, 0)), "duplicate nodes"),
        (("spider_fusion", None, (0, -1)), "negative node"),
        (("spider_fusion", None, [0, 1]), "list nodes"),
        (("spider_fusion", None, (0, True)), "bool node"),
        ((3, None, None), "rule not str"),
    ],
)
def test_rule_step_validates(args: tuple[object, object, object], label: str) -> None:
    with pytest.raises(PR.ParseGrammarError):
        PR.RuleStep(*args, source="")  # type: ignore[arg-type]


def test_duplicate_step_nodes() -> None:
    with pytest.raises(PR.ParseGrammarError):
        PR.parse_step("apply spider_fusion at n0, n0")


# -- the parser never contracts ---------------------------------------------------------------


NO_CONTRACT_SOURCES = [
    GHZ_DSL,
    LISTING,
    "sum_{k=0}^{d-1} |k,k>; copy",
    "(1/2) GHZ_{3}[d] ; copy (x) Z_{1->1}(1: 1/3)[d]",
    "bind[2, 3] ; split[2, 3]",
    "a = Z(0 -> 2, 3) phase {1: 1/3}; box p x 4 nodes a; scalar 1/sqrt(3)",
    "sqrt(2)/2",
]


def test_parser_never_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    import archytaszx.semantics.contract_numeric as numeric
    import archytaszx.semantics.contract_symbolic as symbolic
    from archytaszx.semantics import check

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the parser contracted")

    for module, name in (
        (numeric, "contract"),
        (symbolic, "contract_symbolic"),
        (check, "compare"),
        (check, "compare_symbolic"),
        (check, "compare_tensors"),
        (check, "instantiate"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    for source in NO_CONTRACT_SOURCES:
        parsed = PR.parse(source)
        assert isinstance(parsed, PR.ParsedDiagram)
    PR.apply_step(PR.parse_diagram(GHZ_DSL), PR.parse_step("apply spider_fusion"))
