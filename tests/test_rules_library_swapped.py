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

"""The colour-swapped rule twins: each matches the Z/X exchange of its original's left-hand
side, never the original's, and preserves the denotation at every dimension sampled and with
``d`` formal.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.rule import RewriteGrammarError, Rule
from archytaszx.rewrite.rules_library import (
    BIALGEBRA,
    BIALGEBRA_SWAPPED,
    FOURIER_STATE_COLOR_CHANGE,
    FOURIER_STATE_COLOR_CHANGE_SWAPPED,
    HOPF,
    HOPF_SWAPPED,
    RULES,
    STATE_COPY,
    STATE_COPY_SWAPPED,
    ZX_CAP,
    ZX_CAP_SWAPPED,
    state_copy_swapped_builder,
)
from archytaszx.semantics.check import compare, compare_symbolic
from archytaszx.semantics.contract_symbolic import contract_symbolic
from archytaszx.semantics.decide import DecisionMethod, EqualityVerdict, decide_equal
from archytaszx.semantics.prove import ProofStatus, prove

from .test_rules_library_phase11 import (
    DIMENSIONS,
    bialgebra_diagram,
    fourier_state_diagram,
    hopf_diagram,
    state_copy_diagram,
)


def swap_colours(diagram: Diagram, dim: Dim | None = None) -> Diagram:
    """``diagram`` with Z and X spiders exchanged, every leg at ``dim`` when one is given."""
    swapped = Diagram()
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        generator = {Z_SPIDER.name: X_SPIDER, X_SPIDER.name: Z_SPIDER}.get(
            node.generator_type.name, node.generator_type
        )
        added = swapped.add_node(
            generator,
            [dim or port.dim for port in node.inputs],
            [dim or port.dim for port in node.outputs],
            phase=node.phase,
        )
        assert added == node_id
    for wire in diagram.wires:
        swapped.add_wire(wire.a, wire.b)
    swapped.set_boundary_inputs(list(diagram.boundary_inputs))
    swapped.set_boundary_outputs(list(diagram.boundary_outputs))
    return swapped


def cap_diagram(d: int) -> Diagram:
    """A phaseless Z state wired into a phaseless X effect."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    effect = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
    diagram.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
    return diagram


CASES: dict[str, tuple[Rule, Rule, Callable[[int], Diagram]]] = {
    "cap": (ZX_CAP, ZX_CAP_SWAPPED, cap_diagram),
    "state_copy_0": (STATE_COPY, STATE_COPY_SWAPPED, lambda d: state_copy_diagram(d, 0)),
    "state_copy_3": (STATE_COPY, STATE_COPY_SWAPPED, lambda d: state_copy_diagram(d, 3)),
    "hopf": (HOPF, HOPF_SWAPPED, lambda d: hopf_diagram(d, 1, 1, 1, 1)),
    "hopf_bare": (HOPF, HOPF_SWAPPED, lambda d: hopf_diagram(d, 0, 0, 0, 0)),
    "hopf_phased": (HOPF, HOPF_SWAPPED, lambda d: hopf_diagram(d, 1, 1, 1, 1, phases=True)),
    "bialgebra": (BIALGEBRA, BIALGEBRA_SWAPPED, bialgebra_diagram),
    "fourier_state": (
        FOURIER_STATE_COLOR_CHANGE,
        FOURIER_STATE_COLOR_CHANGE_SWAPPED,
        lambda d: fourier_state_diagram(d, is_state=True),
    ),
    "fourier_effect": (
        FOURIER_STATE_COLOR_CHANGE,
        FOURIER_STATE_COLOR_CHANGE_SWAPPED,
        lambda d: fourier_state_diagram(d, is_state=False),
    ),
}


class TestSwappedTwins:
    def test_every_twin_is_registered(self) -> None:
        for _original, twin, _build in CASES.values():
            assert RULES[twin.name] is twin
            assert twin.name.endswith("_swapped")

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_each_matches_only_its_own_colours(self, case: str) -> None:
        original, twin, build = CASES[case]
        left = build(3)
        assert original.pattern.find_matches(left)
        assert not twin.pattern.find_matches(left)
        swapped = swap_colours(left)
        assert twin.pattern.find_matches(swapped)
        assert not original.pattern.find_matches(swapped)

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("case", sorted(CASES))
    def test_applying_preserves_the_denotation(self, case: str, d: int) -> None:
        original, twin, build = CASES[case]
        left = swap_colours(build(d))
        match = twin.pattern.find_matches(left)[0]
        result = apply(left, twin, match)
        assert result.step.scalar_introduced == original.scalar_for_match(
            original.pattern.find_matches(build(d))[0]
        )
        assert compare(left, result.diagram, {}).matched

    @pytest.mark.parametrize("case", sorted(set(CASES) - {"hopf_phased"}))
    def test_applying_preserves_the_denotation_with_d_formal(self, case: str) -> None:
        _original, twin, build = CASES[case]
        left = swap_colours(build(3), Dim.symbol("d"))
        result = apply(left, twin, twin.pattern.find_matches(left)[0])
        assert compare_symbolic(contract_symbolic(left), contract_symbolic(result.diagram)).matched

    def test_a_swapped_builder_refuses_an_unswapped_match(self) -> None:
        left = state_copy_diagram(3, 2)
        match = STATE_COPY.pattern.find_matches(left)[0]
        with pytest.raises(RewriteGrammarError):
            state_copy_swapped_builder(left.copy(), match)


class TestEntryPointsAgreeOnSwappedStateCopy:
    """A Z state copied through an X spider is settled by rewriting in both entry points."""

    def test_decide_and_prove_both_rewrite_it(self) -> None:
        dim = Dim.symbol("d")
        left = Diagram()
        state = left.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        spider = left.add_node(X_SPIDER, input_dims=[dim], output_dims=[dim, dim])
        left.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(spider, Direction.INPUT, 0))
        left.set_boundary_outputs([PortRef(spider, Direction.OUTPUT, i) for i in range(2)])
        right = Diagram()
        copies = [right.add_node(Z_SPIDER, input_dims=[], output_dims=[dim]) for _ in range(2)]
        right.set_boundary_outputs([PortRef(copy, Direction.OUTPUT, 0) for copy in copies])
        match = STATE_COPY_SWAPPED.pattern.find_matches(left)[0]
        right.multiply_scalar(STATE_COPY_SWAPPED.scalar_for_match(match))
        decision = decide_equal(left, right)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.NORMAL_FORM
        outcome = prove(left, right)
        assert outcome.status is ProofStatus.PROVED, outcome.reason
