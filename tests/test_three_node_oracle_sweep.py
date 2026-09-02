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

"""Permanent regression guard: the 3-node oracle sweep.

Two same-colour fusable spiders (A, B) joined by a wire, plus a third node C wired to a
surviving leg of either A or B, boundaries on everything else. This covers a region the
rest of the suite under-samples: :func:`~qufzx.rewrite.match.find_matches` and
:func:`~qufzx.rewrite.engine.apply` with a third node whose own wiring must survive a
fusion's port remapping untouched, across a spread of leg-count shapes. ``d = 2``
throughout.

Deliberate subsampling:

* The seven leg shapes (0,1),(1,0),(1,1),(2,1),(1,2),(2,0),(0,2) are cross producted over A
  and B independently (49 shape pairs), stressing ``spider_fusion_builder``'s leg-ordering
  convention across genuinely different shapes on each side.
* The connecting leg is chosen deterministically (A's last output if it has one, else its
  only input; B's opposite-direction leg if available, else -- for Z only -- a
  same-direction one), not by trying every pairing within a shape. A shape combination with
  no valid connecting pair under a colour's direction rule is skipped, not force-matched.
* C is always the same colour as A and B and attaches to the first surviving leg found,
  checking A's before B's. This still reaches both "C attaches to A" and "C attaches to B"
  without multiplying the sweep by C's own colour and leg choice.
* Phase presence is swept for A and B (4 combinations); C never carries a phase.

The sweep is therefore 2 colour pairs x 49 shape pairs x 4 phase combinations = 392
diagrams, most of which produce a real, cleanly contractible fusion candidate.
"""

from __future__ import annotations

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER, GeneratorType
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef
from qufzx.diagram.validate import validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import compare

pytestmark = pytest.mark.slow
"""Every test in this module is a multi-thousand-seed sweep."""

_D = Dim.concrete(2)
_LEG_SHAPES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 0),
    (0, 2),
)
_COLORS = (Z_SPIDER, X_SPIDER)
_PHASE_CHOICES = (None, PhaseVector(_D, {1: Phase.turns(sp.Rational(1, 3))}))


def _pick_a_connector(shape: tuple[int, int]) -> tuple[Direction, int]:
    """A's own connecting leg: its last output if it has one, else its only input."""
    inputs, outputs = shape
    if outputs > 0:
        return (Direction.OUTPUT, outputs - 1)
    return (Direction.INPUT, inputs - 1)


def _pick_b_connector(
    shape: tuple[int, int], a_direction: Direction, color: GeneratorType
) -> tuple[Direction, int] | None:
    """B's connecting leg: alternating with A's if possible, else same-direction (Z only).

    Returns ``None`` if ``shape`` has no leg B can offer under either rule -- the caller
    skips that combination rather than force a nonsensical pairing.
    """
    inputs, outputs = shape
    opposite = Direction.INPUT if a_direction is Direction.OUTPUT else Direction.OUTPUT
    if opposite is Direction.INPUT and inputs > 0:
        return (Direction.INPUT, inputs - 1)
    if opposite is Direction.OUTPUT and outputs > 0:
        return (Direction.OUTPUT, outputs - 1)
    if color is Z_SPIDER:
        if a_direction is Direction.INPUT and inputs > 0:
            return (Direction.INPUT, inputs - 1)
        if a_direction is Direction.OUTPUT and outputs > 0:
            return (Direction.OUTPUT, outputs - 1)
    return None


def _surviving_legs(
    node_id: NodeId, shape: tuple[int, int], consumed: tuple[Direction, int]
) -> list[PortRef]:
    inputs, outputs = shape
    legs = [PortRef(node_id, Direction.INPUT, i) for i in range(inputs)]
    legs += [PortRef(node_id, Direction.OUTPUT, i) for i in range(outputs)]
    return [ref for ref in legs if (ref.direction, ref.index) != consumed]


def _build_diagram(
    color: GeneratorType,
    shape_a: tuple[int, int],
    shape_b: tuple[int, int],
    phase_a: PhaseVector | None,
    phase_b: PhaseVector | None,
) -> Diagram | None:
    a_connector = _pick_a_connector(shape_a)
    b_connector = _pick_b_connector(shape_b, a_connector[0], color)
    if b_connector is None:
        return None

    diagram = Diagram()
    a_id = diagram.add_node(
        color, input_dims=[_D] * shape_a[0], output_dims=[_D] * shape_a[1], phase=phase_a
    )
    b_id = diagram.add_node(
        color, input_dims=[_D] * shape_b[0], output_dims=[_D] * shape_b[1], phase=phase_b
    )
    ref_a = PortRef(a_id, *a_connector)
    ref_b = PortRef(b_id, *b_connector)
    diagram.add_wire(ref_a, ref_b)

    surviving_a = _surviving_legs(a_id, shape_a, a_connector)
    surviving_b = _surviving_legs(b_id, shape_b, b_connector)

    # C attaches to the first surviving leg found, preferring A's over B's -- see the
    # module docstring for why this, rather than an exhaustive attachment sweep.
    c_host_ref = surviving_a[0] if surviving_a else (surviving_b[0] if surviving_b else None)
    used: set[PortRef] = set()
    if c_host_ref is not None:
        used.add(c_host_ref)
        if c_host_ref.direction is Direction.OUTPUT:
            c_id = diagram.add_node(color, input_dims=[_D], output_dims=[])
            diagram.add_wire(c_host_ref, PortRef(c_id, Direction.INPUT, 0))
        else:
            c_id = diagram.add_node(color, input_dims=[], output_dims=[_D])
            diagram.add_wire(PortRef(c_id, Direction.OUTPUT, 0), c_host_ref)

    boundary_inputs = [
        ref
        for ref in (*surviving_a, *surviving_b)
        if ref not in used and ref.direction is Direction.INPUT
    ]
    boundary_outputs = [
        ref
        for ref in (*surviving_a, *surviving_b)
        if ref not in used and ref.direction is Direction.OUTPUT
    ]
    diagram.set_boundary_inputs(boundary_inputs)
    diagram.set_boundary_outputs(boundary_outputs)
    return diagram


class TestThreeNodeOracleSweep:
    """Permanent regression guard for audit probe 1. See the module docstring."""

    def test_every_combination_fuses_soundly_or_is_skipped_as_documented(self) -> None:
        checked = 0
        skipped_no_connector = 0
        skipped_not_clean = 0
        for color in _COLORS:
            for shape_a in _LEG_SHAPES:
                for shape_b in _LEG_SHAPES:
                    for phase_a in _PHASE_CHOICES:
                        for phase_b in _PHASE_CHOICES:
                            diagram = _build_diagram(color, shape_a, shape_b, phase_a, phase_b)
                            if diagram is None:
                                skipped_no_connector += 1
                                continue
                            report = validate(diagram)
                            if not report.is_valid or report.deferred:
                                skipped_not_clean += 1
                                continue

                            matches = find_matches(diagram)
                            for match in matches:
                                result = apply(diagram, SPIDER_FUSION, match)
                                post_report = validate(result.diagram)
                                assert post_report.is_valid, (
                                    f"color={color.name} shape_a={shape_a} shape_b={shape_b}: "
                                    f"post-fusion diagram is invalid: {post_report.errors}"
                                )
                                comparison = compare(diagram, result.diagram, {})
                                assert comparison.matched, (
                                    f"color={color.name} shape_a={shape_a} shape_b={shape_b} "
                                    f"phase_a={phase_a} phase_b={phase_b}: {comparison.reason}"
                                )
                                checked += 1

        assert checked > 0, "the sweep never produced a single comparison"
        # Sanity bound: with 2 colors x 49 shape pairs x 4 phase combos = 392 diagrams, and
        # every valid combination yielding exactly one match (a single connecting wire),
        # checked should land close to 392 minus whatever was skipped -- catches a
        # generator regression that silently stopped building most diagrams.
        assert checked >= 100, (
            f"only {checked} comparisons ran (skipped_no_connector={skipped_no_connector}, "
            f"skipped_not_clean={skipped_not_clean}) -- suspiciously low for this sweep"
        )
