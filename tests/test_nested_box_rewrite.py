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

"""Rewrites inside nested node-scope bang boxes rescope every enclosing box, checked by the
validator and by the numeric oracle at several multiplicities and dimensions."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.generators import (
    FOURIER_BOX,
    TRIANGLE,
    TRIANGLE_INVERSE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import validate
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.rules_library import RULES
from archytaszx.semantics.check import compare

_MULTIPLICITIES = (0, 1, 2)


def _phase(d: int, turns: int) -> PhaseVector:
    return PhaseVector(Dim(d), {1: Phase.turns(sp.Rational(turns, 7))})


def _chain(
    g: Diagram, d: int, middle: tuple[GeneratorType, ...], *, effect: GeneratorType
) -> list[NodeId]:
    """A phased Z state through ``middle`` into a one-leg ``effect``; the node ids in order."""
    dim = Dim(d)
    ids = [g.add_node(Z_SPIDER, [], [dim], phase=_phase(d, 1))]
    ids += [g.add_node(generator, [dim], [dim]) for generator in middle]
    effect_phase = _phase(d, 3) if effect is Z_SPIDER else None
    ids.append(g.add_node(effect, [dim], [], phase=effect_phase))
    for left, right in itertools.pairwise(ids):
        g.add_wire(PortRef(left, Direction.OUTPUT, 0), PortRef(right, Direction.INPUT, 0))
    return ids


def _nested(d: int, middle: tuple[GeneratorType, ...], effect: GeneratorType) -> Diagram:
    """``!k(!j(chain) ⊗ chain')``, ``chain'`` a phased Z state into a Z effect."""
    g = Diagram()
    inner = frozenset(_chain(g, d, middle, effect=effect))
    sibling = frozenset(_chain(g, d, (), effect=Z_SPIDER))
    outer = g.add_bang_box(Mult("k"), node_scope=inner | sibling)
    g.add_bang_box(Mult("j"), node_scope=inner, parent=outer)
    return g


def _three_level(d: int) -> Diagram:
    """``!k(!j(!i(chain) ⊗ chain') ⊗ chain'')``, every chain a Z state into a Z effect."""
    g = Diagram()
    x = frozenset(_chain(g, d, (), effect=Z_SPIDER))
    y = frozenset(_chain(g, d, (), effect=Z_SPIDER))
    z = frozenset(_chain(g, d, (), effect=Z_SPIDER))
    k = g.add_bang_box(Mult("k"), node_scope=x | y | z)
    j = g.add_bang_box(Mult("j"), node_scope=x | y, parent=k)
    g.add_bang_box(Mult("i"), node_scope=x, parent=j)
    return g


def _port_scope_inside_node_scope(d: int) -> Diagram:
    """``!k(Z state -> Z with a !j output on the boundary)``."""
    dim = Dim(d)
    g = Diagram()
    a = g.add_node(Z_SPIDER, [], [dim], phase=_phase(d, 2))
    b = g.add_node(Z_SPIDER, [dim], [dim])
    g.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
    out = PortRef(b, Direction.OUTPUT, 0)
    g.set_boundary_outputs([out])
    outer = g.add_bang_box(Mult("k"), node_scope=frozenset({a, b}))
    g.add_bang_box(Mult("j"), port_scope=frozenset({out}), parent=outer)
    return g


def _fourier_state(d: int) -> Diagram:
    """``!k(!j(Z state -> F -> X effect) ⊗ chain')`` with a phaseless Z state."""
    g = _nested(d, (FOURIER_BOX,), X_SPIDER)
    state = min(min(box.node_scope) for box in g.bang_boxes.values())
    g.set_phase(state, None)
    return g


_FIXTURES: dict[str, tuple[str, Callable[[int], Diagram]]] = {
    "fusion": ("spider_fusion", lambda d: _nested(d, (), Z_SPIDER)),
    "fusion_three_level": ("spider_fusion", _three_level),
    "fusion_port_scope": ("spider_fusion", _port_scope_inside_node_scope),
    "identity": ("identity_removal", lambda d: _nested(d, (Z_SPIDER,), X_SPIDER)),
    "triangle": (
        "triangle_inverse_cancellation",
        lambda d: _nested(d, (TRIANGLE, TRIANGLE_INVERSE), X_SPIDER),
    ),
    "fourier": ("fourier_cancellation", lambda d: _nested(d, (FOURIER_BOX,) * 4, X_SPIDER)),
    "fourier_state": ("fourier_state_color_change", _fourier_state),
}


def _assignments(diagram: Diagram) -> list[dict[str, int]]:
    names = sorted({str(box.multiplicity) for box in diagram.bang_boxes.values()})
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(_MULTIPLICITIES, repeat=len(names))
    ]


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("fixture", sorted(_FIXTURES))
class TestEveryRuleInsideNestedBoxes:
    def test_target_rule_rescopes_every_enclosing_box(self, fixture: str, d: int) -> None:
        rule_name, build = _FIXTURES[fixture]
        pre = build(d)
        rule = RULES[rule_name]
        matches = rule.pattern.find_matches(pre)
        assert matches
        for match in matches:
            result = apply(pre, rule, match)
            post = result.diagram
            consumed = frozenset(result.step.consumed_node_ids)
            new = frozenset(result.new_node_ids)
            for box_id, box in pre.bang_boxes.items():
                after = post.bang_boxes[box_id]
                if box.is_node_scope and consumed <= box.node_scope:
                    assert after.node_scope == (box.node_scope - consumed) | new
                else:
                    assert after.node_scope == box.node_scope
                assert after.multiplicity == box.multiplicity
                assert after.parent == box.parent

    def test_every_match_of_every_rule_is_valid_and_exact(self, fixture: str, d: int) -> None:
        pre = _FIXTURES[fixture][1](d)
        assert validate(pre).errors == ()
        for rule in sorted(RULES.values(), key=lambda r: r.name):
            for match in rule.pattern.find_matches(pre):
                post = apply(pre, rule, match).diagram
                assert validate(post).errors == (), rule.name
                for assignment in _assignments(pre):
                    outcome = compare(pre, post, assignment)
                    assert outcome.matched, (rule.name, assignment, outcome.reason)


def test_nested_fusion_is_deterministic() -> None:
    pre = _nested(3, (), Z_SPIDER)
    rule = RULES["spider_fusion"]
    (first, *_rest) = rule.pattern.find_matches(pre)
    a = apply(pre, rule, first).diagram
    b = apply(pre, rule, first).diagram
    assert a.bang_boxes == b.bang_boxes
    assert a.nodes == b.nodes
