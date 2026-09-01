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

"""Permanent regression guard: the 3-parallel-wire arm.

``tests/test_match.py``'s ``TestParallelWiresYieldOneCandidatePerWire`` covers a pair
joined by two parallel wires; this arm goes to three, then performs a second fusion round
on the result. The diagram: A and B, same colour, joined by three parallel wires
(``a.out0-b.in0``, ``a.out1-b.in1``, ``a.out2-b.in2``); B also carries one further leg
wired to a third node C. Round 1 fuses A and B across whichever of the three wires
:func:`~qufzx.rewrite.match.find_matches` picks first (deterministically -- see that
module's own ordering guarantee), leaving the other two as self-loops on the merged node
and carrying B's wire to C over, remapped. The merged node and C are now themselves a
fresh fusion candidate (same colour, one connecting wire) -- round 2 fuses them, and the
final, once-more-merged node carries the two self-loops through a second remap. Both
rounds are oracle-compared against the ORIGINAL three-node diagram, not against each
other's intermediate result, so a defect that only cancels out between two remaps (and
would otherwise look locally correct at each step) cannot hide.

Both colours are covered; every wire here is an alternating OUTPUT-to-INPUT wire, valid
fusion for both, so X's direction restriction (condition 4 in :mod:`qufzx.rewrite.match`)
is respected throughout.
"""

from __future__ import annotations

from qufzx.algebra.dimension import Dim
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER, GeneratorType
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.diagram.validate import validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import compare

_D = Dim.concrete(2)


def _build_diagram(color: GeneratorType) -> Diagram:
    diagram = Diagram()
    a_id = diagram.add_node(color, input_dims=[], output_dims=[_D, _D, _D])
    b_id = diagram.add_node(color, input_dims=[_D, _D, _D], output_dims=[_D])
    for index in range(3):
        diagram.add_wire(
            PortRef(a_id, Direction.OUTPUT, index), PortRef(b_id, Direction.INPUT, index)
        )
    c_id = diagram.add_node(color, input_dims=[_D], output_dims=[])
    diagram.add_wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))
    # Fully wired, no boundary -- contracts to a bare scalar, the simplest possible shape
    # that still exercises both self-loop remaps.
    return diagram


class TestThreeParallelWireArm:
    """Permanent regression guard for audit probe 3. See the module docstring."""

    def test_three_parallel_wires_then_a_second_fusion_round(self) -> None:
        for color in (Z_SPIDER, X_SPIDER):
            original = _build_diagram(color)
            assert validate(original).is_valid

            round1_matches = find_matches(original)
            # One candidate per parallel A-B wire (see match.py's condition 3) -- 3 of
            # them -- plus B-C is already its own, independent one-wire candidate on the
            # *original* diagram (find_matches does not require A and B to have merged
            # first; B and C are already directly wired). Matches are sorted by (a_id,
            # b_id) then wire (see find_matches's own ordering guarantee), so the 3 A-B
            # matches sort before the single B-C one -- round1_matches[0] below is
            # guaranteed to be an A-B match, not B-C.
            assert len(round1_matches) == 4, f"{color.name}: expected 3 A-B matches + 1 B-C match"
            assert {match.a_id for match in round1_matches[:3]} == {round1_matches[0].a_id}
            assert {match.b_id for match in round1_matches[:3]} == {round1_matches[0].b_id}

            round1 = apply(original, SPIDER_FUSION, round1_matches[0])
            assert validate(round1.diagram).is_valid
            merged_id = round1.new_node_ids[0]
            # Two self-loops (4 ports) survived onto the merged node, plus the wire to C.
            self_loop_wires = [
                wire
                for wire in round1.diagram.wires
                if wire.a.node_id == merged_id and wire.b.node_id == merged_id
            ]
            assert len(self_loop_wires) == 2, f"{color.name}: expected 2 self-loops after round 1"

            comparison_round1 = compare(original, round1.diagram, {})
            assert comparison_round1.matched, f"{color.name} round 1: {comparison_round1.reason}"

            round2_matches = find_matches(round1.diagram)
            assert len(round2_matches) == 1, (
                f"{color.name}: expected exactly one round-2 match (the merged node and C) "
                f"-- self-loops must not themselves look like a fusion candidate"
            )
            round2 = apply(round1.diagram, SPIDER_FUSION, round2_matches[0])
            assert validate(round2.diagram).is_valid

            final_id = round2.new_node_ids[0]
            final_self_loops = [
                wire
                for wire in round2.diagram.wires
                if wire.a.node_id == final_id and wire.b.node_id == final_id
            ]
            assert len(final_self_loops) == 2, (
                f"{color.name}: the two self-loops from round 1 must survive round 2's own "
                f"remap onto the final node"
            )
            assert len(round2.diagram.nodes) == 1, f"{color.name}: expected a single final node"

            # Oracle-compared against the ORIGINAL diagram, not against round 1's
            # intermediate result -- see the module docstring for why.
            comparison_round2 = compare(original, round2.diagram, {})
            assert comparison_round2.matched, f"{color.name} round 2: {comparison_round2.reason}"
