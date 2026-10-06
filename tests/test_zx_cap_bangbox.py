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

"""Checks :data:`ZX_CAP` against node-scope bang boxes, at several multiplicities and dimensions."""

from __future__ import annotations

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.bangbox import abstract_subgraph_count, instantiate_symbol
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.rewrite.engine import apply, toward_normal_form
from archytaszx.rewrite.match import find_cap_matches
from archytaszx.rewrite.rules_library import RULES, ZX_CAP
from archytaszx.semantics.check import compare


def cap_diagram(d: int, *, passthrough: bool) -> tuple[Diagram, NodeId, NodeId, NodeId | None]:
    """A Z state wired into an X effect, optionally beside a one-in-one-out Z spider."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    effect = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
    diagram.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
    extra = None
    if passthrough:
        extra = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        diagram.set_boundary_inputs([PortRef(extra, Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(extra, Direction.OUTPUT, 0)])
    return diagram, state, effect, extra


def boxed(d: int, scope: str) -> Diagram:
    """:func:`cap_diagram` with a node-scope box ``m`` over the cap, or the cap and the spider."""
    diagram, state, effect, extra = cap_diagram(d, passthrough=scope != "cap")
    nodes = {state, effect} if scope != "all" else {state, effect, extra}
    pre, _box_id, _m = abstract_subgraph_count(
        diagram, frozenset(n for n in nodes if n is not None), 1, stem="m"
    )
    return pre


def assert_equal_at_every_multiplicity(left: Diagram, right: Diagram) -> None:
    """Assert ``left`` and ``right`` contract equal at ``m`` in 0..3."""
    for m_value in (0, 1, 2, 3):
        result = compare(
            instantiate_symbol(left, "m", m_value), instantiate_symbol(right, "m", m_value), {}
        )
        assert result.matched, (m_value, result.reason)


@pytest.mark.parametrize("scope", ["cap", "cap_beside_spider", "all"])
@pytest.mark.parametrize("d", [2, 3])
class TestCapInsideABox:
    def test_no_match(self, d: int, scope: str) -> None:
        assert find_cap_matches(boxed(d, scope)) == ()

    def test_toward_normal_form_does_not_raise_and_preserves_semantics(
        self, d: int, scope: str
    ) -> None:
        pre = boxed(d, scope)
        outcome = toward_normal_form(pre)
        assert all(step.rule_name != ZX_CAP.name for step in outcome.steps)
        assert_equal_at_every_multiplicity(pre, outcome.diagram)

    def test_every_registered_rule_preserves_semantics(self, d: int, scope: str) -> None:
        pre = boxed(d, scope)
        for rule in RULES.values():
            for match in rule.pattern.find_matches(pre):
                assert_equal_at_every_multiplicity(pre, apply(pre, rule, match).diagram)


@pytest.mark.parametrize("d", [2, 3])
def test_cap_outside_a_box_still_fires(d: int) -> None:
    diagram, state, effect, extra = cap_diagram(d, passthrough=True)
    assert extra is not None
    pre, _box_id, _m = abstract_subgraph_count(diagram, frozenset({extra}), 1, stem="m")
    (match,) = find_cap_matches(pre)
    assert (match.state_id, match.effect_id) == (state, effect)
    assert_equal_at_every_multiplicity(pre, apply(pre, ZX_CAP, match).diagram)
