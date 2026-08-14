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

"""Tests for qufzx.diagram.validate: the Phase 3 well-formedness checker."""

import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef
from qufzx.diagram.validate import (
    IssueKind,
    ValidationFailedError,
    validate,
    validate_or_raise,
)

from .helpers import build_ghz_with_copy


class TestGhzWithCopyValidates:
    @pytest.mark.parametrize("d_value", [2, 3])
    def test_concrete_dims_pass(self, d_value: int) -> None:
        diagram, _a, _b = build_ghz_with_copy(Dim.concrete(d_value))
        report = validate(diagram)
        assert report.is_valid, report.errors

    def test_symbolic_dim_with_symbolic_phase_passes(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.symbol("alpha")})
        diagram, _a, _b = build_ghz_with_copy(d, phase_on_a=phase)
        report = validate(diagram)
        assert report.is_valid, report.errors

    def test_validate_or_raise_does_not_raise_on_valid_diagram(self) -> None:
        diagram, _a, _b = build_ghz_with_copy(Dim.concrete(2))
        validate_or_raise(diagram)


class TestDimensionMismatch:
    def test_mismatched_concrete_dims_fail_with_specific_kind(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.DIMENSION_MISMATCH for issue in report.errors)

    def test_validate_or_raise_raises_on_mismatch(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        with pytest.raises(ValidationFailedError):
            validate_or_raise(diagram)


class TestDeferredDimensionConstraint:
    def test_symbol_against_a_product_containing_it_is_deferred_not_error(self) -> None:
        # Dim.unify's occurs check defers exactly this shape (see its docstring): "d"
        # occurs as a proper subterm of "d * e", so it is neither bound (as bare
        # symbol-vs-symbol would be) nor rejected -- it is a residual constraint.
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b = diagram.add_node(Z_SPIDER, input_dims=[d * e], output_dims=[])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        report = validate(diagram)
        assert report.is_valid
        assert any(issue.kind is IssueKind.DIMENSION_DEFERRED for issue in report.deferred)
        assert not any(issue.kind is IssueKind.DIMENSION_MISMATCH for issue in report.errors)


class TestBoundaryViolations:
    def test_duplicate_boundary_entry(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        ref = PortRef(a, Direction.OUTPUT, 0)
        diagram.set_boundary_outputs([ref, ref])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.DUPLICATE_BOUNDARY_ENTRY for issue in report.errors)

    def test_boundary_port_also_wired(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        ref = PortRef(a, Direction.OUTPUT, 0)
        diagram.add_wire(ref, PortRef(b, Direction.INPUT, 0))
        diagram.set_boundary_outputs([ref])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PORT_WIRED_AND_BOUNDARY for issue in report.errors)

    def test_out_of_range_index(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 5)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PORT_INDEX_OUT_OF_RANGE for issue in report.errors)

    def test_unknown_node_id(self) -> None:
        diagram = Diagram()
        diagram.set_boundary_outputs([PortRef(NodeId(999), Direction.OUTPUT, 0)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.UNKNOWN_NODE for issue in report.errors)

    def test_malformed_ref_on_both_boundary_lists_reported_once(self) -> None:
        """A ref listed on both boundary lists resolves once, not twice (Phase 5 post-closing
        audit round 23, Task 6): resolving it on each list separately used to append two
        identical PORT_INDEX_OUT_OF_RANGE issues for one malformed reference."""
        diagram = Diagram()
        n = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        bad = PortRef(n, Direction.INPUT, 7)
        diagram.set_boundary_inputs([bad])
        diagram.set_boundary_outputs([bad])
        report = validate(diagram)
        out_of_range = [
            issue for issue in report.errors if issue.kind is IssueKind.PORT_INDEX_OUT_OF_RANGE
        ]
        assert len(out_of_range) == 1
        # The separate, legitimate DUPLICATE_BOUNDARY_ENTRY finding must still fire.
        assert any(issue.kind is IssueKind.DUPLICATE_BOUNDARY_ENTRY for issue in report.errors)

    def test_wrong_direction_in_boundary_inputs(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        diagram.set_boundary_inputs([PortRef(a, Direction.OUTPUT, 0)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.BOUNDARY_DIRECTION_MISMATCH for issue in report.errors)

    def test_port_wired_twice(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        c = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        ref = PortRef(a, Direction.OUTPUT, 0)
        diagram.add_wire(ref, PortRef(b, Direction.INPUT, 0))
        diagram.add_wire(ref, PortRef(c, Direction.INPUT, 0))
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PORT_WIRED_TWICE for issue in report.errors)


class TestGeneratorPolicyConformance:
    def test_all_legs_equal_violation(self) -> None:
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[Dim.concrete(3)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION for issue in report.errors)

    def test_phase_dimension_mismatch(self) -> None:
        diagram = Diagram()
        mismatched_phase = PhaseVector(Dim.concrete(3), {1: Phase.turns(1)})
        diagram.add_node(
            Z_SPIDER,
            input_dims=[Dim.concrete(2)],
            output_dims=[Dim.concrete(2)],
            phase=mismatched_phase,
        )
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH for issue in report.errors)

    def test_symbolic_leg_dims_deferred_not_hard_error(self) -> None:
        # d occurs as a proper subterm of d * e, so Dim.unify defers this pair rather
        # than binding or failing it (see its docstring) -- this must land as
        # DIMENSION_DEFERRED, not DIMENSION_POLICY_VIOLATION, mirroring the wire-level
        # case in TestDeferredDimensionConstraint.
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d * e])
        report = validate(diagram)
        assert not any(
            issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION for issue in report.errors
        )
        assert any(
            issue.kind is IssueKind.DIMENSION_DEFERRED and issue.node_id is not None
            for issue in report.deferred
        )

    def test_concrete_leg_dims_still_hard_error(self) -> None:
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[Dim.concrete(3)])
        report = validate(diagram)
        assert any(
            issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION and not issue.deferred
            for issue in report.errors
        )

    def test_symbolic_phase_dim_deferred_not_hard_error(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        phase = PhaseVector(d * e, {1: Phase.symbol("alpha")})
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d], phase=phase)
        report = validate(diagram)
        assert not any(issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH for issue in report.errors)
        assert any(
            issue.kind is IssueKind.DIMENSION_DEFERRED and issue.node_id is not None
            for issue in report.deferred
        )

    def test_concrete_phase_dim_still_hard_error(self) -> None:
        diagram = Diagram()
        mismatched_phase = PhaseVector(Dim.concrete(3), {1: Phase.turns(1)})
        diagram.add_node(
            Z_SPIDER,
            input_dims=[Dim.concrete(2)],
            output_dims=[Dim.concrete(2)],
            phase=mismatched_phase,
        )
        report = validate(diagram)
        assert any(
            issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH and not issue.deferred
            for issue in report.errors
        )


class TestPhaseDimensionResolvedThroughLegBindings:
    """D1, Phase 5 post-closing audit round 22: the TIED_TO_LEG_DIM phase check must compare
    against the leg set's *jointly resolved* dimension (via unify_all), not the raw,
    unresolved first leg -- see qufzx.diagram.validate._check_generator_policy. Before this
    fix, the verdict depended on which leg the diagram happened to list first, and a
    leg/phase disagreement masked by a leg/leg binding computed independently (and
    discarded) could pass as valid even though no single substitution satisfies both.
    """

    def test_first_leg_binding_masking_a_phase_disagreement_is_rejected(self) -> None:
        # legs [d, 2] bind d := 2 (independently satisfiable); phase dim 3 unifies against
        # the raw first leg d, binding d := 3 -- discarded pre-fix, so nothing caught that
        # d cannot be both 2 and 3 at once. This diagram is unsatisfiable at every
        # substitution (apply d := 2 and phase dim 3 disagrees with leg dim 2).
        d = Dim.symbol("d")
        phase = PhaseVector(Dim.concrete(3), {})
        diagram = Diagram()
        diagram.add_node(
            Z_SPIDER, input_dims=[d, Dim.concrete(2)], output_dims=[], phase=phase
        )
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH for issue in report.errors)

    def test_verdict_is_order_independent_across_leg_permutation(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(Dim.concrete(3), {})
        for legs in ([d, Dim.concrete(2)], [Dim.concrete(2), d]):
            diagram = Diagram()
            diagram.add_node(Z_SPIDER, input_dims=list(legs), output_dims=[], phase=phase)
            report = validate(diagram)
            assert not report.is_valid
            assert any(
                issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH for issue in report.errors
            ), f"legs={legs}: verdict must not depend on leg order"

    def test_phase_agreeing_with_a_bound_leg_symbol_is_accepted(self) -> None:
        # legs [d, 2] bind d := 2; a phase over d (not yet resolved) or over the concrete
        # value 2 must both be accepted -- the resolved leg dimension is 2 either way.
        d = Dim.symbol("d")
        for phase_dim in (d, Dim.concrete(2)):
            diagram = Diagram()
            diagram.add_node(
                Z_SPIDER,
                input_dims=[d, Dim.concrete(2)],
                output_dims=[],
                phase=PhaseVector(phase_dim, {}),
            )
            report = validate(diagram)
            assert not any(
                issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH for issue in report.errors
            ), f"phase_dim={phase_dim}: {report.errors}"

    def test_failed_leg_set_does_not_also_report_a_phase_mismatch(self) -> None:
        # Legs [2, 3] are already a DIMENSION_POLICY_VIOLATION on their own; the phase
        # check must not additionally fire an arbitrary PHASE_DIMENSION_MISMATCH against
        # whichever leg happens to be first -- the two are distinct findings, and a leg
        # set with no coherent shared dimension has nothing well-defined for the phase to
        # be checked against.
        diagram = Diagram()
        diagram.add_node(
            Z_SPIDER,
            input_dims=[Dim.concrete(2), Dim.concrete(3)],
            output_dims=[],
            phase=PhaseVector(Dim.concrete(5), {}),
        )
        report = validate(diagram)
        assert not report.is_valid
        kinds = {issue.kind for issue in report.errors}
        assert IssueKind.DIMENSION_POLICY_VIOLATION in kinds
        assert IssueKind.PHASE_DIMENSION_MISMATCH not in kinds

    @pytest.mark.parametrize("seed", range(60))
    def test_sweep_over_leg_palettes_permutations_and_phase_dims_is_order_independent(
        self, seed: int
    ) -> None:
        # Domain sweep (not one hand-picked shape, per round 19's own meta-lesson): random
        # leg-dimension palettes, every permutation of them, and a random phase dimension --
        # asserting the verdict (is_valid, and specifically whether a
        # PHASE_DIMENSION_MISMATCH fires) never depends on leg order.
        import itertools
        import random

        rng = random.Random(seed)
        d, e = Dim.symbol("d"), Dim.symbol("e")
        palette = (Dim.concrete(2), Dim.concrete(3), Dim.concrete(5), d, e)
        legs = [rng.choice(palette) for _ in range(rng.randint(1, 3))]
        phase_dim = rng.choice(palette)
        phase = PhaseVector(phase_dim, {})

        verdicts: set[tuple[bool, bool]] = set()
        for perm in itertools.permutations(legs):
            diagram = Diagram()
            diagram.add_node(Z_SPIDER, input_dims=list(perm), output_dims=[], phase=phase)
            report = validate(diagram)
            has_phase_mismatch = any(
                issue.kind is IssueKind.PHASE_DIMENSION_MISMATCH for issue in report.errors
            )
            verdicts.add((report.is_valid, has_phase_mismatch))

        assert len(verdicts) == 1, (
            f"seed {seed}: legs={legs}, phase_dim={phase_dim}: verdict depends on leg "
            f"order across permutations: {verdicts}"
        )


class TestNonConcreteLegBindingIsSilentlyAccepted:
    """Phase 5 post-closing audit round 23, Task 7b: a node whose legs unify only by binding
    one bare symbol to another (e.g. legs ``d`` and ``e``, binding ``d := e``) reports no
    issue at all -- a deliberate Phase 5 decision (see ``_check_generator_policy``'s own
    inline comment), not an oversight, and asymmetric with the structurally identical
    situation in :mod:`qufzx.rewrite.match`, which records it as a ``BOUND`` certificate
    entry. ``UnifyAllResult.declined_bindings`` is populated regardless (so the assumption
    is not lost, only not yet reported) -- pinned directly in ``test_dimension.py``. This
    class pins today's silent behavior at the ``validate()`` level, so a future change to
    surface ``declined_bindings`` here (left to Phase 10, per that comment) is a deliberate,
    visible change to this test, not a silent behavior drift.
    """

    def test_two_bare_symbol_legs_report_no_issue(self) -> None:
        d = Dim.symbol("d")
        e = Dim.symbol("e")
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d, e], output_dims=[])
        diagram.set_boundary_inputs(
            [PortRef(node_id, Direction.INPUT, 0), PortRef(node_id, Direction.INPUT, 1)]
        )
        report = validate(diagram)
        assert report.is_valid
        assert not report.deferred
        assert not report.errors


class TestAllLegsEqualJointSatisfiability:
    """F1, Phase 5 post-closing audit round 21: leg dims must be *jointly* unifiable, not
    merely pairwise-unifiable against the first leg -- see qufzx.algebra.dimension.unify_all.
    """

    def test_jointly_unsatisfiable_bindings_are_rejected(self) -> None:
        # d unifies with 2 (binds d := 2) and with 3 (binds d := 3) independently, but the
        # two bindings contradict -- validate() must reject this, not silently discard both.
        d = Dim.symbol("d")
        diagram = Diagram()
        diagram.add_node(
            Z_SPIDER, input_dims=[d, Dim.concrete(2), Dim.concrete(3)], output_dims=[]
        )
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION for issue in report.errors)

    def test_verdict_is_order_independent(self) -> None:
        d = Dim.symbol("d")
        for dims in ([d, Dim.concrete(2), Dim.concrete(3)], [Dim.concrete(2), d, Dim.concrete(3)]):
            diagram = Diagram()
            diagram.add_node(Z_SPIDER, input_dims=list(dims), output_dims=[])
            report = validate(diagram)
            assert not report.is_valid
            assert any(
                issue.kind is IssueKind.DIMENSION_POLICY_VIOLATION for issue in report.errors
            )

    def test_multiple_residual_deferred_pairs_are_all_reported(self) -> None:
        # R3: three legs, each pair deferred against the others -- one DIMENSION_DEFERRED
        # issue per residual pair, not one collapsed "strongest" issue for the whole node.
        d, e, f = Dim.symbol("d"), Dim.symbol("e"), Dim.symbol("f")
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[d * e, d * f, e * f], output_dims=[])
        diagram.set_boundary_inputs(
            [PortRef(node, Direction.INPUT, i) for i in range(3)]
        )
        report = validate(diagram)
        assert report.is_valid
        deferred_on_node = [
            issue for issue in report.deferred if issue.kind is IssueKind.DIMENSION_DEFERRED
        ]
        assert len(deferred_on_node) >= 2


class TestSymbolRoleCollision:
    """F2, Phase 5 post-closing audit round 21: a name cannot legally serve as both a
    dimension symbol and a phase parameter within one diagram (see
    qufzx.diagram.validate._classify_symbol_role)."""

    def test_same_name_dimension_and_phase_symbol_is_rejected(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.symbol("d")})
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[], phase=phase)
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.SYMBOL_ROLE_COLLISION for issue in report.errors)

    def test_dimension_symbol_embedded_in_root_of_unity_phase_is_not_a_collision(self) -> None:
        # A root-of-unity phase entry over the node's own symbolic dim legitimately embeds
        # that same dimension symbol (not a distinct phase-role symbol of the same name) --
        # this must not be flagged.
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.root_of_unity(1, d)})
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[], phase=phase)
        report = validate(diagram)
        assert not any(
            issue.kind is IssueKind.SYMBOL_ROLE_COLLISION for issue in report.errors
        )

    def test_distinct_symbols_of_the_same_name_are_never_equal(self) -> None:
        # The discriminator itself: Dim.symbol/Phase.symbol/Scalar.symbol build distinct
        # sympy Symbol objects for the same name string, via different assumptions.
        from qufzx.algebra.scalar import Scalar

        dim_symbol = Dim.symbol("d").to_sympy()
        phase_symbol = Phase.symbol("d").to_sympy_turns()
        scalar_symbol = Scalar.symbol("d").to_sympy()
        assert dim_symbol != phase_symbol
        assert dim_symbol != scalar_symbol
        assert phase_symbol != scalar_symbol

    def test_exponent_symbol_dimension_collision_is_flagged_and_named_correctly(self) -> None:
        # D2: a dimension symbol 'n' and an exponent symbol also named 'n' (d ** n) is a
        # genuine collision -- differing domains under one name (positive vs. merely
        # nonnegative integers) -- and must be reported as {'dimension', 'exponent'}, not
        # mislabelled {'dimension', 'phase'} the way the pre-fix three-role classifier did
        # (an exponent symbol, lacking 'positive', fell through into the "phase" branch).
        from qufzx.diagram.validate import _classify_symbol_role

        n_dim = Dim.symbol("n")
        d = Dim.symbol("d")
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[n_dim], output_dims=[d**n_dim])
        report = validate(diagram)
        assert not report.is_valid
        collision_issues = [
            issue for issue in report.errors if issue.kind is IssueKind.SYMBOL_ROLE_COLLISION
        ]
        assert collision_issues
        assert "dimension" in collision_issues[0].message
        assert "exponent" in collision_issues[0].message
        assert "phase" not in collision_issues[0].message

        # The classifier itself, directly: an exponent symbol built via Dim.__pow__ must
        # read as "exponent", not "phase".
        exponent_symbol = (d**n_dim).to_sympy().free_symbols - d.to_sympy().free_symbols
        (exponent_sym,) = exponent_symbol
        assert _classify_symbol_role(exponent_sym) == "exponent"

    def test_exponent_and_phase_parameter_sharing_a_name_is_flagged(self) -> None:
        # D2's false-negative case: pre-fix, both an exponent and a phase parameter
        # (Phase.symbol) landed in the same "phase" bucket, so a name shared between them
        # went completely unflagged.
        d = Dim.symbol("d")
        n = Dim.symbol("n")
        phase = PhaseVector(Dim.concrete(2), {1: Phase.symbol("n")})
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d**n], phase=phase)
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.SYMBOL_ROLE_COLLISION for issue in report.errors)


class TestSymbolConstructorRolesRoundTrip:
    """D2 hardening: every symbol constructor in qufzx.algebra must round-trip to its own,
    distinct role under _classify_symbol_role -- a test that fails loudly the day a fifth
    constructor is added without updating the classifier to give it a role of its own
    (silently aliasing into an existing one would otherwise only surface as a missed or
    mislabelled SYMBOL_ROLE_COLLISION much later, the way the exponent gap did).
    """

    def test_every_constructor_round_trips_to_a_distinct_role(self) -> None:
        from qufzx.algebra.scalar import Scalar
        from qufzx.diagram.validate import _classify_symbol_role

        d = Dim.symbol("d")
        n = (d**Dim.symbol("n")).to_sympy().free_symbols - d.to_sympy().free_symbols
        (exponent_symbol,) = n

        constructors: dict[str, object] = {
            "dimension": Dim.symbol("s").to_sympy(),
            "exponent": exponent_symbol,
            "phase": Phase.symbol("s").to_sympy_turns(),
            "scalar": Scalar.symbol("s").to_sympy(),
        }

        roles = {name: _classify_symbol_role(symbol) for name, symbol in constructors.items()}
        for name, role in roles.items():
            assert role == name, f"{name} constructor classified as {role!r}, expected {name!r}"
        assert len(set(roles.values())) == len(roles), (
            f"two constructors mapped to the same role: {roles}"
        )


class TestNodeDimensionUndetermined:
    """Round 20, Task 9: a node with zero legs and no phase carries its dimension nowhere at
    all (per CLAUDE.md, "dimension is stored per port, not as one global parameter"), so it
    is not well-formed -- yet this module used to accept it as valid, while
    :mod:`qufzx.semantics.denote` correctly refused it. The invariant this closes:
    ``validate(d).is_valid`` implies every node in ``d`` is denotable (see
    ``tests/test_phase5_exhaustive_oracle.py``'s exhaustive sweep, which now checks this
    over its whole space, and ``denote.py``'s own "has no legs and no phase vector" message,
    which this issue's message deliberately echoes).
    """

    def test_legless_phaseless_node_is_a_hard_error(self) -> None:
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[])
        report = validate(diagram)
        assert not report.is_valid
        assert any(
            issue.kind is IssueKind.NODE_DIMENSION_UNDETERMINED
            and issue.node_id == node_id
            and not issue.deferred
            for issue in report.errors
        )

    def test_legless_node_with_a_phase_is_not_flagged(self) -> None:
        diagram = Diagram()
        diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[], phase=PhaseVector(Dim.concrete(2), {})
        )
        report = validate(diagram)
        assert not any(
            issue.kind is IssueKind.NODE_DIMENSION_UNDETERMINED for issue in report.issues
        )

    def test_node_with_a_leg_and_no_phase_is_not_flagged(self) -> None:
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        report = validate(diagram)
        assert not any(
            issue.kind is IssueKind.NODE_DIMENSION_UNDETERMINED for issue in report.issues
        )


class TestDanglingPorts:
    def test_dangling_output_port(self) -> None:
        diagram, a_id, _b = build_ghz_with_copy(Dim.concrete(2))
        # A's output 1 is on the boundary by default; strip it so it dangles.
        diagram.set_boundary_outputs(
            [ref for ref in diagram.boundary_outputs if ref != PortRef(a_id, Direction.OUTPUT, 1)]
        )
        report = validate(diagram)
        assert not report.is_valid
        assert any(
            issue.kind is IssueKind.PORT_UNUSED
            and issue.port_ref == PortRef(a_id, Direction.OUTPUT, 1)
            for issue in report.errors
        )

    def test_dangling_input_port(self) -> None:
        diagram = Diagram()
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        report = validate(diagram)
        assert not report.is_valid
        assert any(
            issue.kind is IssueKind.PORT_UNUSED and issue.port_ref == PortRef(b, Direction.INPUT, 0)
            for issue in report.errors
        )

    def test_dangling_port_suppressed_when_node_reference_already_broken(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 5)])
        report = validate(diagram)
        assert not report.is_valid
        assert any(issue.kind is IssueKind.PORT_INDEX_OUT_OF_RANGE for issue in report.errors)
        assert not any(issue.kind is IssueKind.PORT_UNUSED for issue in report.errors)


class TestValidationIsPure:
    def test_validate_does_not_mutate_diagram(self) -> None:
        diagram, _a, _b = build_ghz_with_copy(Dim.concrete(2))
        nodes_before = dict(diagram.nodes)
        wires_before = diagram.wires
        boundary_before = diagram.boundary_outputs
        scalar_before = diagram.scalar
        validate(diagram)
        assert dict(diagram.nodes) == nodes_before
        assert diagram.wires == wires_before
        assert diagram.boundary_outputs == boundary_before
        assert diagram.scalar == scalar_before
