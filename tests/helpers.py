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

"""Shared test fixtures: the "A into B" GHZ-with-copy worked example from the build plan.

Not a test module itself (no ``Test*`` classes). Imported across the Phase 3, 4 and 5
suites -- :mod:`tests.test_diagram_graph`, :mod:`tests.test_validate`,
:mod:`tests.test_phase3_oracle`, :mod:`tests.test_check`,
:mod:`tests.test_contract_numeric`, :mod:`tests.test_phase4_oracle`,
:mod:`tests.test_match`, :mod:`tests.test_rules_library`, :mod:`tests.test_engine`,
:mod:`tests.test_phase5_oracle` and :mod:`tests.test_phase5_dirac_oracle` -- so every one
builds the worked example the same way.
"""

from __future__ import annotations

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseVector
from qufzx.diagram.generators import Z_SPIDER, GeneratorType
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef


def build_ghz_with_copy(
    dim: Dim,
    *,
    phase_on_a: PhaseVector | None = None,
    phase_on_b: PhaseVector | None = None,
    generator_type: GeneratorType = Z_SPIDER,
) -> tuple[Diagram, NodeId, NodeId]:
    """Build the "A into B" example: a state-prep spider feeding a copy spider.

    A is a spider with 0 inputs and 2 outputs (the state ``sum_k |kk>``). B is a spider
    with 1 input and 2 outputs (the copy/Delta). One output of A is wired into the input
    of B, leaving 3 free outputs -- A's remaining output and both of B's outputs -- as
    the ordered boundary (the GHZ-with-copy graph). ``generator_type`` defaults to
    ``Z_SPIDER`` (the original worked example); Phase 5's oracle suite passes
    ``X_SPIDER`` additively to exercise the same shape through the X color.

    Returns the diagram together with A's and B's node ids, since callers frequently
    need to address specific ports (e.g. to attach a phase or to build a deliberately
    mismatched wire).
    """
    diagram = Diagram()
    a_id = diagram.add_node(
        generator_type, input_dims=[], output_dims=[dim, dim], phase=phase_on_a
    )
    b_id = diagram.add_node(
        generator_type, input_dims=[dim], output_dims=[dim, dim], phase=phase_on_b
    )
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [
            PortRef(a_id, Direction.OUTPUT, 1),
            PortRef(b_id, Direction.OUTPUT, 0),
            PortRef(b_id, Direction.OUTPUT, 1),
        ]
    )
    return diagram, a_id, b_id
