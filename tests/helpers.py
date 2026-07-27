"""Shared test fixtures: the "A into B" GHZ-with-copy worked example from the build plan.

Not a test module itself (no ``Test*`` classes); imported by :mod:`test_diagram_graph`,
:mod:`test_validate`, and :mod:`test_phase3_oracle`, and intended for reuse by the
Phase 4/5/6 test suites that build on the same example.
"""

from __future__ import annotations

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseVector
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef


def build_ghz_with_copy(
    dim: Dim,
    *,
    phase_on_a: PhaseVector | None = None,
    phase_on_b: PhaseVector | None = None,
) -> tuple[Diagram, NodeId, NodeId]:
    """Build the "A into B" example: a state-prep spider feeding a copy spider.

    A is a Z spider with 0 inputs and 2 outputs (the state ``sum_k |kk>``). B is a Z
    spider with 1 input and 2 outputs (the copy/Delta). One output of A is wired into
    the input of B, leaving 3 free outputs -- A's remaining output and both of B's
    outputs -- as the ordered boundary (the GHZ-with-copy graph).

    Returns the diagram together with A's and B's node ids, since callers frequently
    need to address specific ports (e.g. to attach a phase or to build a deliberately
    mismatched wire).
    """
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim], phase=phase_on_a)
    b_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim], phase=phase_on_b)
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [
            PortRef(a_id, Direction.OUTPUT, 1),
            PortRef(b_id, Direction.OUTPUT, 0),
            PortRef(b_id, Direction.OUTPUT, 1),
        ]
    )
    return diagram, a_id, b_id
