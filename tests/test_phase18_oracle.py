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

"""Phase 18 done-whens: GHZ-with-copy from text, listing and Dirac round trips, mode
independence of every input form, and parse-apply-check end to end."""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, free_mult_symbols
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
)
from archytaszx.diagram.graph import Diagram
from archytaszx.diagram.validate import validate
from archytaszx.repl import parser as PR
from archytaszx.repl import printer as P
from archytaszx.repl.printer import NotationMode, render, render_graph
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.rules_library import RULES, SPIDER_FUSION
from archytaszx.semantics.check import compare

from .helpers import build_ghz_with_copy
from .test_phase17_oracle import BOXED, PHASES, SYMBOLS, WIRED, close, inp, node, out, spider
from .test_printer import chain_results, listing_example
from .test_scalable import BANG_BOX_FAMILIES

D = Dim.symbol("d")
E = Dim.symbol("e")
DS = (3, 4, 5)
Builder = Callable[[], Diagram]

GHZ_DSL = "a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0; outputs a.out1, b.out0, b.out1"
GHZ_DIRAC = "sum_{k=0}^{d-1} |k,k>; copy"


def ghz() -> Diagram:
    return build_ghz_with_copy(D)[0]


def ghz_bound(value: int = 3) -> Diagram:
    d = ghz()
    d.set_parameters({"d": value})
    return d


def assert_identical(got: Diagram, want: Diagram) -> None:
    """Field-for-field equality of two diagrams, ids included."""
    result = compare_structure(got, want)
    assert result.identical, result.reason
    assert dict(got.bang_boxes) == dict(want.bang_boxes)


# -- GHZ-with-copy done-when ----------------------------------------------------------------


class TestGhzWithCopy:
    def test_dsl_gives_the_hand_built_graph(self) -> None:
        assert_identical(PR.parse_dsl(GHZ_DSL), ghz())

    def test_dsl_through_parse_and_parse_diagram(self) -> None:
        parsed = PR.parse(GHZ_DSL)
        assert isinstance(parsed, PR.ParsedDiagram)
        assert parsed.form is PR.InputForm.DSL
        assert_identical(parsed.diagram, ghz())
        assert_identical(PR.parse_diagram(GHZ_DSL), ghz())

    def test_dsl_with_auto_outputs(self) -> None:
        src = "a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0"
        assert_identical(PR.parse_dsl(src), ghz())

    def test_dsl_multiline_with_comments(self) -> None:
        src = (
            "# GHZ with copy\n"
            "a = Z(0 -> 2, d)   # state\n"
            "b = Z(1 -> 2, d)\n"
            "\n"
            "wire a.out0 -- b.in0\n"
            "outputs a.out1, b.out0, b.out1\n"
        )
        assert_identical(PR.parse_dsl(src), ghz())

    def test_dirac_gives_the_hand_built_graph(self) -> None:
        assert_identical(PR.parse_dirac(GHZ_DIRAC), ghz())
        parsed = PR.parse(GHZ_DIRAC)
        assert isinstance(parsed, PR.ParsedDiagram)
        assert parsed.form is PR.InputForm.DIRAC
        assert_identical(parsed.diagram, ghz())

    def test_dirac_agrees_with_the_legacy_parser(self) -> None:
        assert_identical(PR.parse_dirac(GHZ_DIRAC), PR.parse_dirac_source(GHZ_DIRAC))

    def test_dsl_with_numeral_dim_binds_d(self) -> None:
        src = GHZ_DSL.replace(", d)", ", 3)")
        assert_identical(PR.parse_dsl(src), ghz_bound(3))

    def test_dirac_with_numeral_dim_binds_d(self) -> None:
        assert_identical(PR.parse_dirac("sum_{k=0}^{3-1} |k,k>; copy"), ghz_bound(3))

    def test_dsl_and_dirac_agree(self) -> None:
        assert_identical(PR.parse_dsl(GHZ_DSL), PR.parse_dirac(GHZ_DIRAC))

    def test_listing_of_the_graph_parses_back(self) -> None:
        assert_identical(PR.parse_listing(render_graph(ghz())), ghz())

    def test_parsed_graph_is_valid(self) -> None:
        for d in (PR.parse_dsl(GHZ_DSL), PR.parse_dirac(GHZ_DIRAC)):
            assert validate(d).is_valid


# -- listing round trip -----------------------------------------------------------------------


def phased_scalar() -> Diagram:
    d = Diagram()
    a = node(
        d,
        Z_SPIDER,
        1,
        2,
        phase=PhaseVector(D, {1: Phase.symbol("theta"), 2: Phase.turns(sp.Rational(1, 3))}),
    )
    b = node(d, Z_SPIDER, 1, 1)
    d.add_wire(out(a, 0), inp(b, 0))
    close(d)
    d.multiply_scalar(Scalar.dim_power(D, -1, 2))
    return d


def root_phase_omega() -> Diagram:
    d = Diagram()
    node(d, X_SPIDER, 0, 2, phase=PhaseVector(D, {1: Phase.root_of_unity(1, D)}))
    close(d)
    d.multiply_scalar(Scalar.omega(D))
    return d


def omega_power_times_dim() -> Diagram:
    d = Diagram()
    node(d, Z_SPIDER, 1, 1)
    close(d)
    d.multiply_scalar(Scalar.dim_power(D, 1) * Scalar.omega(D, 2) * Scalar.rational(1, 2))
    return d


def connectives() -> Diagram:
    d = Diagram()
    b = d.add_node(DIM_BINDER, [D, E], [D * E])
    s = d.add_node(DIM_SPLITTER, [D * E], [D, E])
    z = node(d, Z_SPIDER, 1, 1, E)
    d.add_wire(out(b, 0), inp(s, 0))
    d.add_wire(out(s, 1), inp(z, 0))
    return close(d)


def connectives_concrete() -> Diagram:
    d = Diagram()
    b = d.add_node(DIM_BINDER, [Dim(2), D], [Dim(2) * D])
    s = d.add_node(DIM_SPLITTER, [Dim(2) * D], [D, Dim(2)])
    d.add_wire(out(b, 0), inp(s, 0))
    close(d)
    d.set_parameters({"d": 3})
    return d


def box_chain() -> Diagram:
    d = Diagram()
    f = node(d, FOURIER_BOX, 1, 1)
    t = node(d, TRIANGLE, 1, 1)
    ti = node(d, TRIANGLE_INVERSE, 1, 1)
    w = node(d, W_NODE, 1, 3)
    x = node(d, X_SPIDER, 1, 1, phase=PhaseVector(D, {1: Phase.turns(sp.Rational(-1, 4))}))
    d.add_wire(out(f, 0), inp(t, 0))
    d.add_wire(out(t, 0), inp(ti, 0))
    d.add_wire(out(ti, 0), inp(w, 0))
    d.add_wire(out(w, 1), inp(x, 0))
    close(d)
    d.multiply_scalar(Scalar.gaussian_rational(1, -2))
    return d


def concrete_dims() -> Diagram:
    dim = Dim(3)
    d = Diagram()
    a = node(
        d,
        Z_SPIDER,
        0,
        2,
        dim,
        PhaseVector(dim, {1: Phase.turns(sp.Rational(1, 3)), 2: Phase.turns(sp.Rational(2, 3))}),
    )
    b = node(d, X_SPIDER, 1, 1, dim)
    d.add_wire(out(a, 0), inp(b, 0))
    close(d)
    d.multiply_scalar(Scalar.rational(3, 2))
    return d


def exponent_dim() -> Diagram:
    dim = D ** Dim.symbol("m")
    d = Diagram()
    node(d, Z_SPIDER, 1, 1, dim)
    return close(d)


def symbol_scalar() -> Diagram:
    d = Diagram()
    node(d, Z_SPIDER, 0, 1, phase=PhaseVector(D, {1: Phase.symbol("phi")}))
    close(d)
    d.multiply_scalar(Scalar.symbol("c") * Scalar.from_dim(D))
    return d


def zero_scalar() -> Diagram:
    d = Diagram()
    node(d, Z_SPIDER, 1, 1)
    close(d)
    d.set_scalar(Scalar.zero())
    return d


def id_gaps() -> Diagram:
    d = Diagram()
    gone = node(d, Z_SPIDER, 0, 1)
    a = node(d, Z_SPIDER, 0, 2)
    gone2 = node(d, X_SPIDER, 1, 1)
    b = node(d, X_SPIDER, 1, 1)
    d.remove_node(gone)
    d.remove_node(gone2)
    d.add_wire(out(a, 0), inp(b, 0))
    close(d)
    first = d.add_bang_box(Mult("k"), node_scope=frozenset({a}))
    d.add_bang_box(Mult("n") * 2, port_scope=frozenset({out(b, 0)}))
    d.remove_bang_box(first)
    return d


def multi_index_nested() -> Diagram:
    d = Diagram()
    a = node(d, Z_SPIDER, 0, 2, phase=PhaseVector(D, {1: Phase.symbol("theta")}))
    b = node(d, X_SPIDER, 1, 2)
    c = node(d, Z_SPIDER, 1, 1, E)
    w = node(d, W_NODE, 1, 1, E)
    d.add_wire(out(a, 0), inp(b, 0))
    d.add_wire(out(c, 0), inp(w, 0))
    close(d)
    outer = d.add_bang_box(Mult("n"), node_scope=frozenset({a, b}))
    mid = d.add_bang_box(Mult("m") + 1, node_scope=frozenset({b}), parent=outer)
    d.add_bang_box(Mult("k"), port_scope=frozenset({out(b, 1)}), parent=mid)
    d.add_bang_box(Mult("j"), node_scope=frozenset({c, w}))
    d.add_bang_box(Mult("j") * Mult("m"), port_scope=frozenset({out(a, 1)}), parent=outer)
    return d


def sqrt_d_scalar_only() -> Diagram:
    d = Diagram()
    a = node(d, Z_SPIDER, 0, 1, phase=PhaseVector(D))
    b = node(d, X_SPIDER, 1, 0, phase=PhaseVector(D))
    d.add_wire(out(a, 0), inp(b, 0))
    rule = RULES["state_copy_swapped"]
    (match,) = rule.pattern.find_matches(d)
    return apply(d, rule, match).diagram


def unicode_names() -> Diagram:
    delta = Dim.symbol("δ")
    d = Diagram()
    node(d, Z_SPIDER, 1, 1, delta, PhaseVector(delta, {1: Phase.symbol("θ")}))
    close(d)
    d.multiply_scalar(Scalar.symbol("α") * Scalar.from_dim(delta))
    d.set_parameters({"δ": 3})
    return d


def constant_named_dims() -> Diagram:
    d = Diagram()
    for name in ("I", "E", "pi", "exp", "sqrt"):
        node(d, Z_SPIDER, 1, 1, Dim.symbol(name))
    return close(d)


def polynomial_exponent_dim() -> Diagram:
    exp_n = sp.Symbol("n", integer=True, nonnegative=True)
    d = Diagram()
    node(d, Z_SPIDER, 1, 1, Dim.from_sympy(D.to_sympy() ** exp_n**2))
    return close(d)


def huge_dim() -> Diagram:
    d = Diagram()
    node(d, Z_SPIDER, 1, 1, Dim(2**64))
    return close(d)


def huge_parameter() -> Diagram:
    d = ghz()
    d.set_parameters({"d": 10**999})
    return d


def parameters_bound() -> Diagram:
    d = multi_index_nested()
    d.set_parameters({"d": 3, "e": 2, "n": 2, "m": 1})
    return d


HAND: dict[str, Builder] = {
    "empty": Diagram,
    "ghz": ghz,
    "ghz_bound": ghz_bound,
    "phased_scalar": phased_scalar,
    "root_phase_omega": root_phase_omega,
    "omega_power_times_dim": omega_power_times_dim,
    "connectives": connectives,
    "connectives_concrete": connectives_concrete,
    "box_chain": box_chain,
    "concrete_dims": concrete_dims,
    "exponent_dim": exponent_dim,
    "symbol_scalar": symbol_scalar,
    "zero_scalar": zero_scalar,
    "id_gaps": id_gaps,
    "multi_index_nested": multi_index_nested,
    "parameters_bound": parameters_bound,
    "listing_example": listing_example,
    "sqrt_d_scalar_only": sqrt_d_scalar_only,
    "unicode_names": unicode_names,
    "constant_named_dims": constant_named_dims,
    "polynomial_exponent_dim": polynomial_exponent_dim,
    "huge_dim": huge_dim,
    "huge_parameter": huge_parameter,
    **{f"wired_{k}": v for k, v in WIRED.items()},
    **{f"boxed_{k}": v for k, v in BOXED.items()},
    **{f"family_{k}": v for k, v in BANG_BOX_FAMILIES.items()},
    **{
        f"spider_{g.name}{m}{n}_{ph}": spider(g, m, n, PHASES[ph])
        for g in (Z_SPIDER, X_SPIDER)
        for m, n in ((0, 1), (1, 2), (2, 0))
        for ph in PHASES
    },
}


def assert_listing_round_trip(diagram: Diagram) -> None:
    """parse_listing inverts render_graph field for field and text for text."""
    text = render_graph(diagram)
    back = PR.parse_listing(text)
    assert_identical(back, diagram)
    assert render_graph(back) == text


@pytest.mark.parametrize("name", sorted(HAND))
def test_listing_round_trip(name: str) -> None:
    diagram = HAND[name]()
    assert validate(diagram).is_valid, validate(diagram)
    assert_listing_round_trip(diagram)


@pytest.mark.parametrize("name", sorted(HAND))
def test_listing_round_trip_through_parse(name: str) -> None:
    diagram = HAND[name]()
    parsed = PR.parse(render_graph(diagram))
    assert isinstance(parsed, PR.ParsedDiagram)
    assert parsed.form is PR.InputForm.LISTING
    assert_identical(parsed.diagram, diagram)


REWRITE_BASES = sorted(
    name for name in HAND if not name.startswith("spider_") and name not in ("empty",)
)


def rewrite_pairs() -> list[tuple[str, str]]:
    """Every (base, rule) pair whose rule matches the base."""
    return [
        (base, rule)
        for base in REWRITE_BASES
        for rule in sorted(RULES)
        if RULES[rule].pattern.find_matches(HAND[base]())
    ]


REWRITE_PAIRS = rewrite_pairs()


def test_rewrite_pairs_are_plentiful() -> None:
    assert len(REWRITE_PAIRS) > 20
    assert len({base for base, _ in REWRITE_PAIRS}) > 10


@pytest.mark.parametrize(
    ("base", "rule"), REWRITE_PAIRS, ids=[f"{b}-{r}" for b, r in REWRITE_PAIRS]
)
def test_listing_round_trip_after_rewrites(base: str, rule: str) -> None:
    diagram = HAND[base]()
    matches = RULES[rule].pattern.find_matches(diagram)
    assert matches
    result = apply(diagram, RULES[rule], matches[0])
    assert validate(result.diagram).is_valid, validate(result.diagram)
    assert_listing_round_trip(result.diagram)


def test_listing_round_trip_on_fusion_and_chain_states() -> None:
    pre = ghz()
    fused = apply(pre, SPIDER_FUSION, SPIDER_FUSION.pattern.find_matches(pre)[0]).diagram
    assert_listing_round_trip(fused)
    initial, results = chain_results()
    assert results
    for state in (initial, *(r.diagram for r in results)):
        assert_listing_round_trip(state)
    boxed = BANG_BOX_FAMILIES["fusion_pair"]()
    matches = SPIDER_FUSION.pattern.find_matches(boxed)
    assert matches
    assert_listing_round_trip(apply(boxed, SPIDER_FUSION, matches[0]).diagram)


def test_listing_never_abstracts() -> None:
    diagram = concrete_dims()
    back = PR.parse_listing(render_graph(diagram))
    assert dict(back.parameters) == {}
    assert all(p.dim == Dim(3) for n in back.nodes.values() for p in (*n.inputs, *n.outputs))


# -- Dirac round trip -----------------------------------------------------------------------


def phased_two_dims() -> Diagram:
    d = Diagram()
    node(d, Z_SPIDER, 1, 1)
    node(d, Z_SPIDER, 1, 1, E)
    close(d)
    d.multiply_scalar(Scalar.omega(D))
    return d


def ghz_effect_concrete() -> Diagram:
    d = Diagram()
    node(d, Z_SPIDER, 2, 0, Dim(3))
    return close(d)


DIRAC_CASES: dict[str, tuple[Builder, bool]] = {
    "ghz": (ghz, True),
    "phased_scalar": (phased_scalar, True),
    "phased_two_dims": (phased_two_dims, True),
    "symbol_scalar": (symbol_scalar, True),
    "ghz_effect_concrete": (ghz_effect_concrete, False),
    "wired_Z-Z": (WIRED["Z-Z"], True),
    "wired_z_self_loop": (WIRED["z_self_loop"], True),
    "boxed_ghz_z": (BOXED["ghz_z"], True),
    "boxed_ghz_z_in_out": (BOXED["ghz_z_in_out"], True),
    "family_ghz": (BANG_BOX_FAMILIES["ghz"], False),
    "family_ghz_abstracted": (BANG_BOX_FAMILIES["ghz_abstracted"], False),
    "family_multi_port": (BANG_BOX_FAMILIES["multi_port"], False),
    **{
        f"spider_{m}{n}_{ph}": (spider(Z_SPIDER, m, n, PHASES[ph]), True)
        for m, n in ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 0))
        for ph in PHASES
    },
}


def assignments(diagram: Diagram, symbolic_d: bool) -> list[dict[str, object]]:
    """Concrete d values (when d is symbolic) crossed with small counts per box symbol."""
    names = sorted(free_mult_symbols(diagram))
    counts = [
        dict(zip(names, c, strict=True)) for c in itertools.product(range(3), repeat=len(names))
    ]
    ds: list[dict[str, object]] = [{"d": v, "e": v + 1} for v in DS] if symbolic_d else [{}]
    return [{**SYMBOLS, **dv, **cv} for dv in ds for cv in counts]


def dirac_texts(diagram: Diagram) -> list[str]:
    term = P.dirac_term(diagram)
    return [P.render_term(term), P.render_term(term, unicode=True)]


@pytest.mark.parametrize("name", sorted(DIRAC_CASES))
def test_dirac_round_trip_is_oracle_equal(name: str) -> None:
    build, symbolic_d = DIRAC_CASES[name]
    diagram = build()
    assert all(n.generator_type is Z_SPIDER for n in diagram.nodes.values())
    assert all(not b.node_scope for b in diagram.bang_boxes.values())
    for text in dirac_texts(diagram):
        back = PR.parse_dirac(text)
        assert validate(back).is_valid, (text, validate(back))
        for env in assignments(diagram, symbolic_d):
            result = compare(diagram, back, env)
            assert result.matched, (text, env, result.reason)


@pytest.mark.parametrize("name", sorted(DIRAC_CASES))
@pytest.mark.parametrize("unicode", [False, True])
def test_render_dirac_formula_round_trip(name: str, unicode: bool) -> None:
    build, symbolic_d = DIRAC_CASES[name]
    diagram = build()
    text = P.render_dirac(diagram, unicode=unicode)
    formula = text.split(" = ", 1)[1] if P.recognize(diagram) is not None else text
    parsed = PR.parse(text)
    assert isinstance(parsed, PR.ParsedDiagram)
    assert parsed.form is PR.InputForm.DIRAC
    for back in (PR.parse_dirac(formula), PR.parse_dirac(text), parsed.diagram):
        assert validate(back).is_valid
        for env in assignments(diagram, symbolic_d):
            assert compare(diagram, back, env).matched, (text, env)


# -- mode independence ------------------------------------------------------------------------


FORM_SOURCES: dict[str, str] = {
    "dsl": GHZ_DSL,
    "dirac": GHZ_DIRAC,
    "listing": render_graph(listing_example()),
    "dirac_scalar_prefix": "(1/sqrt(d)) sum_{k=0}^{d-1} |k, k>",
}

FORM_KIND = {
    "dsl": "DSL",
    "dirac": "DIRAC",
    "listing": "LISTING",
    "dirac_scalar_prefix": "DIRAC",
}


@pytest.mark.parametrize("name", sorted(FORM_SOURCES))
def test_every_form_is_stable_and_renders_in_both_modes(name: str) -> None:
    source = FORM_SOURCES[name]
    first = PR.parse(source)
    second = PR.parse(source)
    assert isinstance(first, PR.ParsedDiagram)
    assert isinstance(second, PR.ParsedDiagram)
    assert first.form is PR.InputForm[FORM_KIND[name]]
    assert first.form is second.form
    assert_identical(first.diagram, second.diagram)
    assert validate(first.diagram).is_valid
    dirac_text = render(first.diagram, NotationMode.DIRAC)
    zx_text = render(first.diagram, NotationMode.BOTH)
    assert "graph:" not in dirac_text
    assert zx_text.startswith("graph:")
    assert dirac_text.strip() and dirac_text.strip() in zx_text


def test_step_form_is_stable_and_its_result_renders_in_both_modes() -> None:
    source = "apply spider_fusion"
    first = PR.parse(source)
    second = PR.parse(source)
    assert isinstance(first, PR.RuleStep)
    assert first == second
    result = PR.apply_step(PR.parse_diagram(GHZ_DSL), first)
    dirac_text = render(result, NotationMode.DIRAC)
    zx_text = render(result, NotationMode.ZX)
    assert "graph:" not in dirac_text
    assert "graph:" in zx_text


PUBLIC_FUNCTIONS = (
    PR.parse,
    PR.parse_diagram,
    PR.parse_dirac,
    PR.parse_dsl,
    PR.parse_listing,
    PR.parse_step,
    PR.apply_step,
)


@pytest.mark.parametrize("fn", PUBLIC_FUNCTIONS, ids=[f.__name__ for f in PUBLIC_FUNCTIONS])
def test_parser_takes_no_mode_argument(fn: Callable[..., object]) -> None:
    signature = inspect.signature(fn)
    for name, param in signature.parameters.items():
        assert "mode" not in name.lower(), (fn, name)
        assert "NotationMode" not in str(param.annotation), (fn, name)
    assert "NotationMode" not in str(signature.return_annotation)


# -- end to end -----------------------------------------------------------------------------


def test_dsl_apply_fusion_is_oracle_exact() -> None:
    pre = PR.parse_diagram(GHZ_DSL)
    step = PR.parse("apply spider_fusion")
    assert isinstance(step, PR.RuleStep)
    result = PR.apply_step(pre, step)
    post = result.diagram
    assert validate(post).is_valid
    assert len(post.nodes) == 1
    (merged,) = post.nodes.values()
    assert (merged.num_inputs, merged.num_outputs) == (0, 3)
    assert_identical(pre, ghz())
    for dv in DS:
        r = compare(pre, post, {"d": dv})
        assert r.matched, r.reason


def test_dsl_apply_fusion_at_node_set_matches_default() -> None:
    pre = PR.parse_diagram(GHZ_DSL)
    by_default = PR.apply_step(pre, PR.parse_step("apply spider_fusion"))
    by_nodes = PR.apply_step(pre, PR.parse_step("apply spider_fusion at n0, n1"))
    by_index = PR.apply_step(pre, PR.parse_step("apply spider_fusion at 0"))
    assert compare_structure(by_default.diagram, by_nodes.diagram).identical
    assert compare_structure(by_default.diagram, by_index.diagram).identical


def test_bound_dsl_apply_fusion_checks_with_parameters_only() -> None:
    pre = PR.parse_diagram(GHZ_DSL.replace(", d)", ", 3)"))
    post = PR.apply_step(pre, PR.parse_step("apply spider_fusion")).diagram
    assert dict(post.parameters) == {"d": 3}
    assert compare(pre, post, {}).matched


def test_rewritten_listing_reparses_and_rechecks() -> None:
    pre = PR.parse_diagram(GHZ_DSL)
    post = PR.apply_step(pre, PR.parse_step("apply spider_fusion")).diagram
    back = PR.parse_diagram(render_graph(post))
    assert_identical(back, post)
    for dv in DS:
        assert compare(pre, back, {"d": dv}).matched


def test_box_scoped_dsl_fusion_is_oracle_exact() -> None:
    src = GHZ_DSL + "; box p x m nodes a, b"
    pre = PR.parse_diagram(src)
    assert [b.multiplicity for b in pre.bang_boxes.values()] == [Mult("m")]
    post = PR.apply_step(pre, PR.parse_step("apply spider_fusion")).diagram
    for dv, mv in itertools.product((2, 3), (0, 1, 2)):
        assert compare(pre, post, {"d": dv, "m": mv}).matched
