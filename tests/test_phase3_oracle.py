"""Phase 3 numeric-oracle stand-in for the diagram data model.

Per CLAUDE.md, every build phase ends with a numeric oracle check. The real semantics
oracle arrives in Phase 4; here, exactly as tests/test_phase2_oracle.py does with cmath,
we build the GHZ-with-copy worked example with a symbolic dimension and a symbolic
phase, instantiate those symbols to concrete values, and verify -- by direct inspection
of the diagram's own structure, with no contractor and no numpy -- the structural facts
that a denotation function will later have to depend on: every wire's two ports agree on
a concrete Dim, boundary arity is exactly 3, every leg of each spider carries the same
concrete dimension, and the scalar accumulator is exactly Scalar.one() (nothing in
diagram construction quietly drops or introduces a global factor).
"""

import sympy as sp  # type: ignore[import-untyped]

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.validate import validate

from .helpers import build_ghz_with_copy


class TestGhzWithCopyOracle:
    def test_instantiated_diagram_is_structurally_consistent(self) -> None:
        d_symbolic = Dim.symbol("d")
        symbolic_phase = PhaseVector(d_symbolic, {1: Phase.symbol("alpha")})

        for d_value in (2, 3, 5):
            concrete_dim = d_symbolic.substitute({"d": d_value})
            concrete_phase = symbolic_phase.substitute({"d": d_value, "alpha": sp.Rational(1, 6)})

            diagram, a_id, b_id = build_ghz_with_copy(concrete_dim, phase_on_b=concrete_phase)
            report = validate(diagram)
            assert report.is_valid, report.errors

            # Every wire's two ports agree on a concrete Dim.
            assert len(diagram.wires) == 1
            for wire in diagram.wires:
                node_a = diagram.nodes[wire.a.node_id]
                node_b = diagram.nodes[wire.b.node_id]
                port_a = node_a.legs(wire.a.direction)[wire.a.index]
                port_b = node_b.legs(wire.b.direction)[wire.b.index]
                assert port_a.dim.is_concrete
                assert port_b.dim.is_concrete
                assert port_a.dim == port_b.dim
                assert port_a.dim == Dim.concrete(d_value)

            # Boundary arity is exactly 3.
            assert len(diagram.boundary_outputs) == 3
            assert diagram.boundary_inputs == ()

            # Every leg of each spider carries the same concrete dimension.
            for node_id in (a_id, b_id):
                node = diagram.nodes[node_id]
                all_ports = (*node.inputs, *node.outputs)
                assert all(port.dim == Dim.concrete(d_value) for port in all_ports)

            # The phase attached to B is concrete after substitution and carries the
            # instantiated value, not a dropped or default one.
            b_phase = diagram.nodes[b_id].phase
            assert b_phase is not None
            assert b_phase.is_concrete
            assert b_phase.get(1) == Phase.turns(sp.Rational(1, 6))

            # The scalar accumulator is exactly Scalar.one(): construction introduced
            # no factor of its own.
            assert diagram.scalar == Scalar.one()
            assert diagram.scalar.is_one
