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

"""The literal chain ``FULL_PLAN.md`` names for Phase 5's completion condition: Dirac source
string to :class:`~qufzx.diagram.graph.Diagram` to fusion match to post-diagram, oracle
confirmed -- run end to end for the first time (Phase 5 post-closing audit round 19, Task 3).

``tests/test_phase5_oracle.py`` already covers the graph-to-fuse-to-graph half, starting
from ``tests/helpers.py::build_ghz_with_copy``'s hand-built diagram, and states explicitly
that the Dirac half was deferred. This module is that deferral closed, not a duplicate of
that file's oracle coverage: it starts from a Dirac *source string*, parses it with
:func:`qufzx.repl.parser.parse_dirac_source`, and pins the parsed diagram to
``build_ghz_with_copy``'s own, already-oracle-checked construction before running the same
fuse-and-compare chain that file does -- so this path is verified against the existing
ground truth, not standing alone as a second, independently-trusted diagram builder.
"""

from __future__ import annotations

import re

import pytest

import qufzx.repl.parser as parser_module
from qufzx.algebra.dimension import Dim
from qufzx.diagram.validate import validate
from qufzx.repl.parser import DiracDomainError, DiracError, DiracGrammarError, parse_dirac_source
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import EqualityMode, compare

from .helpers import build_ghz_with_copy

_SOURCE = "sum_{k=0}^{d-1} |k,k>; copy"
_CONCRETE_DS = (2, 3, 5)


class TestDiracToGraphToFuseToGraph:
    def test_parsed_diagram_is_structurally_identical_to_the_hand_built_fixture(self) -> None:
        d = Dim.symbol("d")
        parsed = parse_dirac_source(_SOURCE)
        reference, _a_id, _b_id = build_ghz_with_copy(d)
        assert parsed.nodes == reference.nodes
        assert parsed.wires == reference.wires
        assert parsed.boundary_inputs == reference.boundary_inputs
        assert parsed.boundary_outputs == reference.boundary_outputs

    def test_full_chain_fuses_to_a_single_spider_with_expected_leg_count(self) -> None:
        pre = parse_dirac_source(_SOURCE)
        matches = find_matches(pre)
        assert len(matches) == 1, "expected exactly one fusion match on the parsed diagram"
        post = apply(pre, SPIDER_FUSION, matches[0]).diagram
        assert len(post.nodes) == 1
        (merged,) = post.nodes.values()
        assert merged.num_inputs == 0
        assert merged.num_outputs == 3

    def test_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        pre = parse_dirac_source("sum_{k=0}^{3-1} |k,k>; copy")
        matches = find_matches(pre)
        post = apply(pre, SPIDER_FUSION, matches[0]).diagram
        report = validate(post)
        assert report.is_valid
        assert not report.deferred

    def test_full_chain_is_oracle_exact_at_several_concrete_d(self) -> None:
        pre = parse_dirac_source(_SOURCE)
        matches = find_matches(pre)
        post = apply(pre, SPIDER_FUSION, matches[0]).diagram
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason

    def test_full_chain_is_oracle_exact_at_a_concrete_source_dimension(self) -> None:
        # d itself given concretely in the source string, not bound after the fact --
        # exercises _parse_dim's concrete-integer branch through the whole chain.
        pre = parse_dirac_source("sum_{k=0}^{5-1} |k,k>; copy")
        matches = find_matches(pre)
        post = apply(pre, SPIDER_FUSION, matches[0]).diagram
        result = compare(pre, post, {})
        assert result.mode is EqualityMode.EXACT
        assert result.matched, result.reason

    def test_tensor_power_shorthand_parses_to_the_same_diagram_as_repeated_indices(
        self,
    ) -> None:
        repeated = parse_dirac_source("sum_{k=0}^{d-1} |k,k>")
        shorthand = parse_dirac_source("sum_{k=0}^{d-1} |k>^{2}")
        assert repeated.nodes == shorthand.nodes
        assert repeated.boundary_outputs == shorthand.boundary_outputs

    def test_state_only_source_without_copy_parses_to_a_single_state_spider(self) -> None:
        diagram = parse_dirac_source("sum_{k=0}^{d-1} |k,k,k>")
        assert len(diagram.nodes) == 1
        (node,) = diagram.nodes.values()
        assert node.num_inputs == 0
        assert node.num_outputs == 3
        assert not diagram.wires


class TestDiracParserGrammar:
    def test_malformed_source_raises_grammar_error(self) -> None:
        for bad in ("not dirac at all", "sum_{k=0}^{d-1} |k,j>", "sum_{k=0}^{d-1} |k,k>; frob"):
            try:
                parse_dirac_source(bad)
            except DiracGrammarError:
                continue
            raise AssertionError(f"{bad!r} should have raised DiracGrammarError")

    def test_zero_leg_count_raises_domain_error(self) -> None:
        try:
            parse_dirac_source("sum_{k=0}^{d-1} |k>^{0}")
        except DiracDomainError:
            return
        raise AssertionError("a zero-leg ket family should have raised DiracDomainError")

    def test_body_and_power_together_is_a_grammar_error(self) -> None:
        try:
            parse_dirac_source("sum_{k=0}^{d-1} |k,k>^{2}")
        except DiracGrammarError:
            return
        raise AssertionError("ambiguous body+power should have raised DiracGrammarError")

    def test_every_malformed_or_out_of_domain_source_raises_only_dirac_error(self) -> None:
        """Round 20, Task 1: no foreign exception (e.g. DimensionDomainError) may escape
        parse_dirac_source. Swept over a table of distinct out-of-domain/malformed sources,
        not just the one zero-dimension case that originally exposed the leak."""
        bad_sources = (
            "sum_{k=0}^{0-1} |k,k>",  # concrete dimension 0
            "sum_{k=0}^{k-1} |k,k>",  # bound index used as a dimension symbol
            f"sum_{{k=0}}^{{d-1}} |k>^{{{parser_module._MAX_KET_LEG_COUNT + 1}}}",  # too many legs
            "sum_{k=0}^{d-1} |k>^{0}",  # zero legs
            "sum_{k=0}^{d-1} |k,j>",  # grammar error, not domain
            "not dirac at all",  # grammar error, not domain
        )
        for source in bad_sources:
            try:
                parse_dirac_source(source)
            except DiracError as exc:
                assert isinstance(exc, DiracError), (source, exc)
                continue
            raise AssertionError(f"{source!r} should have raised a DiracError")

    def test_dimension_zero_is_a_dirac_domain_error_not_a_foreign_one(self) -> None:
        try:
            parse_dirac_source("sum_{k=0}^{0-1} |k,k>")
        except DiracDomainError:
            return
        raise AssertionError("dimension 0 should have raised DiracDomainError")


class TestDiracParserAsciiNumericTokens:
    """Round 24: the numeric guard is the predicate its contract is written in terms of.

    :func:`~qufzx.repl.parser._parse_dim` used to gate ``int(token)`` on
    ``token.isdigit()`` while :class:`~qufzx.repl.parser.DiracError`'s own safety audit
    justified the call as safe because "the token matches ``\\d+``". Those are three
    different predicates, widening in this order: ``[0-9]+`` (ASCII), ``\\d+`` (Unicode
    category ``Nd``), ``str.isdigit()`` (``Nd`` *and* ``No``). ``int()`` accepts the first
    two and raises ``ValueError`` on the third, so ``_parse_dim`` called with a ``No``
    character leaked a bare ``ValueError`` through this module's ``DiracError`` boundary --
    the Task 1 defect class round 20 closed everywhere else here. Nothing reachable through
    :func:`~qufzx.repl.parser.parse_dirac_source` did that, because ``_KET_SUM_RE`` gated
    every real call site, so the audit's *conclusion* held while its stated *reason* did
    not; the module docstring says Phase 18 replaces that grammar.

    These tests pin the fixed contract at both levels -- the private helper standing alone,
    and the public entry point -- so a future caller that does not go through the current
    regex cannot silently reopen it.
    """

    def test_helper_rejects_non_ascii_digits_as_dirac_errors_not_value_errors(self) -> None:
        """The regression proper: category ``No`` (isdigit, but int() raises)."""
        for token in ("\u00b2", "\u2075"):  # superscript two, superscript five
            assert token.isdigit(), f"{token!r} must be isdigit() or this pins nothing"
            with pytest.raises(DiracError):
                parser_module._parse_dim(token)

    def test_helper_rejects_non_ascii_decimal_digits(self) -> None:
        """Category ``Nd``: matched ``\\d+`` and parsed as a concrete dimension before."""
        token = "\u0663"  # Arabic-Indic digit three
        assert re.fullmatch(r"\d+", token), f"{token!r} must match \\d+ or this pins nothing"
        with pytest.raises(DiracError):
            parser_module._parse_dim(token)

    def test_helper_is_total_rejecting_tokens_matching_neither_branch(self) -> None:
        """A token that is neither a decimal literal nor an identifier must raise, not
        silently become a symbol named after itself -- which is what dropping the
        ``ValueError`` alone would have left behind."""
        for token in ("", "1abc", "-3", "a b", "\u00b2"):
            with pytest.raises(DiracGrammarError):
                parser_module._parse_dim(token)

    def test_helper_still_accepts_ascii_digits_and_identifiers(self) -> None:
        assert parser_module._parse_dim("12") == Dim.concrete(12)
        assert parser_module._parse_dim("d") == Dim.symbol("d")
        assert parser_module._parse_dim("_d2") == Dim.symbol("_d2")

    def test_source_with_non_ascii_dimension_digits_is_a_grammar_error(self) -> None:
        """End to end: the public entry point rejects it as malformed source."""
        with pytest.raises(DiracGrammarError):
            parse_dirac_source("sum_{k=0}^{\u0663-1} |k,k>")

    def test_source_with_non_ascii_power_digits_is_a_grammar_error(self) -> None:
        """The tensor-power group is narrowed by the same shared constant."""
        with pytest.raises(DiracGrammarError):
            parse_dirac_source("sum_{k=0}^{d-1} |k>^{\u0663}")

    def test_ascii_sources_are_unaffected(self) -> None:
        """The narrowing must not have moved any legitimate source."""
        assert len(parse_dirac_source("sum_{k=0}^{3-1} |k,k>").nodes) == 1
        assert len(parse_dirac_source("sum_{k=0}^{d-1} |k,k>; copy").nodes) == 2
        assert len(parse_dirac_source("sum_{k=0}^{d-1} |k>^{4}").nodes) == 1

    def test_grammar_and_guard_share_one_source_of_truth(self) -> None:
        """The guard cannot drift from the grammar again: both are built from the same
        constants, which is what makes the docstring's claim structural rather than a
        coincidence two separately-maintained predicates happen to agree on."""
        assert parser_module._ASCII_DIGITS in parser_module._KET_SUM_RE.pattern
        assert parser_module._IDENTIFIER in parser_module._KET_SUM_RE.pattern
        assert parser_module._ASCII_DIGITS_RE.pattern == f"^{parser_module._ASCII_DIGITS}$"
        assert parser_module._IDENTIFIER_RE.pattern == f"^{parser_module._IDENTIFIER}$"
        # Both numeric groups in the grammar, not just the dimension one.
        assert parser_module._KET_SUM_RE.pattern.count(parser_module._ASCII_DIGITS) == 2


class TestDiracParserSummationIndexCapture:
    def test_bound_index_as_dimension_is_rejected(self) -> None:
        try:
            parse_dirac_source("sum_{k=0}^{k-1} |k,k>")
        except DiracDomainError:
            return
        raise AssertionError("'k' as a dimension symbol should have raised DiracDomainError")

    def test_nearby_identifiers_are_still_accepted(self) -> None:
        for name in ("k2", "kd"):
            diagram = parse_dirac_source(f"sum_{{k=0}}^{{{name}-1}} |k,k>")
            (node,) = diagram.nodes.values()
            assert node.outputs[0].dim == Dim.symbol(name)


class TestDiracParserLegCountBound:
    def test_leg_count_above_bound_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(parser_module, "_MAX_KET_LEG_COUNT", 4)
        try:
            parse_dirac_source("sum_{k=0}^{d-1} |k>^{5}")
        except DiracDomainError:
            pass
        else:
            raise AssertionError("leg count above the patched bound should have been rejected")

    def test_leg_count_at_bound_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(parser_module, "_MAX_KET_LEG_COUNT", 4)
        diagram = parse_dirac_source("sum_{k=0}^{d-1} |k>^{4}")
        (node,) = diagram.nodes.values()
        assert node.num_outputs == 4


class TestDiracParserErrorMessageRendering:
    def test_body_and_power_message_renders_braces_not_doubled_literals(self) -> None:
        try:
            parse_dirac_source("sum_{k=0}^{d-1} |k,k>^{2}")
        except DiracGrammarError as exc:
            message = str(exc)
            assert "{{" not in message and "}}" not in message
            assert "'|k>^{n}'" in message
        else:
            raise AssertionError("expected DiracGrammarError")
