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

from qufzx.algebra.dimension import Dim
from qufzx.diagram.validate import validate
from qufzx.repl.parser import DiracDomainError, DiracGrammarError, parse_dirac_source
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
