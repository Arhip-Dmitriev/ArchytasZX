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

"""Permanent regression guard: the symbolic-dimension sweep.

1-in/1-out spider pairs (A, B), each leg dim drawn from ``{2, 3, d, e}``, built and fused
once symbolically, then oracle-compared at several concrete ``(d, e)`` assignments via
:func:`~qufzx.semantics.check.compare`'s ``assignment`` parameter. This is the arm that
exercises :func:`~qufzx.rewrite.match._resolve_with_bindings`,
:func:`~qufzx.rewrite.match._unify_surviving_legs`,
:func:`~qufzx.rewrite.match._unify_phase_dims`, and
:func:`~qufzx.rewrite.match.reattach_phase` together: a phase legally stated over a symbolic
dimension that a fusion's own unify resolves through a binding -- the ``_over_shared_dim``
family described in :mod:`qufzx.rewrite.rules_library`'s module docstring.

Phases are drawn independently of the leg dims: besides a phase over the node's own leg
dim, root-of-unity phases over the fixed concrete dims ``2`` and ``3`` are offered whatever
the connecting leg carries, making reachable a diagram in which A's and B's phases would
bind the same shared symbol to two different concrete values.

Anti-laundering invariant. For every fused diagram, at every concrete assignment, this arm
asserts that if the post-fusion diagram is cleanly contractible then the pre-fusion diagram
is too -- the case where it is not is exactly what a laundered assumption produces, so it
is asserted rather than skipped.

Deliberate subsampling:

* The full ``{2, 3, d, e}`` palette is applied to the connecting pair (A's output, B's
  input), where the 16 combinations exercise every
  :meth:`~qufzx.algebra.dimension.Dim.unify` outcome and drive ``shared_dim``. Each node's
  surviving leg is fixed at one already-mixed pair (concrete ``2`` and symbolic ``d``).
* Only the ``(2, 3)`` and ``(3, 2)`` oracle substitutions are checked; the same-value pairs
  are covered by ``test_phase5_oracle.py``'s ``_CONCRETE_DS`` sweep.
* ``DEFERRED`` never arises from ``{2, 3, d, e}`` alone -- it needs a symbol occurring as a
  proper subterm of the other side -- so "clean" here means ``validate(diagram).is_valid``
  alone. That shape is covered by ``tests/test_fusion_properties.py``'s ``d*e``/``d**2``
  palette entries.
"""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.diagram.validate import validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import compare, instantiate

_PALETTE: dict[str, Dim] = {
    "2": Dim.concrete(2),
    "3": Dim.concrete(3),
    "d": Dim.symbol("d"),
    "e": Dim.symbol("e"),
}
_D_E_PAIRS: tuple[tuple[int, int], ...] = ((2, 3), (3, 2))
_UNRELATED_PHASE_DIMS: tuple[Dim, Dim] = (Dim.concrete(2), Dim.concrete(3))


def _diagram_free_symbols(diagram: Diagram) -> frozenset[str]:
    """Every dimension/phase symbol name still free anywhere in ``diagram``.

    Used to scope the anti-laundering invariant below to assignments the post-fusion
    diagram could actually still depend on -- see that invariant's own comment.
    """
    symbols: set[str] = set()
    for node in diagram.nodes.values():
        for port in (*node.inputs, *node.outputs):
            symbols |= port.dim.free_symbols
        if node.phase is not None:
            symbols |= node.phase.free_symbols
    return frozenset(symbols)


def _phase_choices(dim: Dim) -> tuple[PhaseVector | None, ...]:
    # The first three choices are tied to the node's own connecting leg dim (``dim``); the
    # last two are fixed concrete dims unrelated to it -- see the module docstring,
    # "Independent phase dimensions", for why both shapes need to be present.
    return (
        None,
        PhaseVector(dim, {1: Phase.turns(sp.Rational(1, 3))}),
        PhaseVector(dim, {1: Phase.root_of_unity(1, dim)}),
        PhaseVector(
            _UNRELATED_PHASE_DIMS[0], {1: Phase.root_of_unity(1, _UNRELATED_PHASE_DIMS[0])}
        ),
        PhaseVector(
            _UNRELATED_PHASE_DIMS[1], {1: Phase.root_of_unity(1, _UNRELATED_PHASE_DIMS[1])}
        ),
    )


def _build_diagram(
    a_out_label: str, b_in_label: str, phase_a: PhaseVector | None, phase_b: PhaseVector | None
) -> Diagram:
    # A's surviving input and B's surviving output are fixed, not swept -- see the module
    # docstring for why the connecting pair is where this arm's combinatorics live.
    a_in = Dim.concrete(2)
    b_out = Dim.symbol("d")
    a_out = _PALETTE[a_out_label]
    b_in = _PALETTE[b_in_label]

    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[a_in], output_dims=[a_out], phase=phase_a)
    b_id = diagram.add_node(Z_SPIDER, input_dims=[b_in], output_dims=[b_out], phase=phase_b)
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])
    diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])
    return diagram


class TestSymbolicDimensionSweep:
    """Permanent regression guard for audit probe 2. See the module docstring."""

    def test_every_combination_fuses_soundly_or_is_skipped_as_documented(self) -> None:
        checked = 0
        diagrams_fused = 0
        skipped_not_clean = 0
        skipped_no_match = 0
        for a_out_label, phase_dim_a in _PALETTE.items():
            for b_in_label, phase_dim_b in _PALETTE.items():
                for phase_a in _phase_choices(phase_dim_a):
                    for phase_b in _phase_choices(phase_dim_b):
                        diagram = _build_diagram(a_out_label, b_in_label, phase_a, phase_b)
                        report = validate(diagram)
                        if not report.is_valid:
                            skipped_not_clean += 1
                            continue

                        matches = find_matches(diagram)
                        if not matches:
                            skipped_no_match += 1
                            continue
                        assert len(matches) == 1, (
                            f"a_out={a_out_label} b_in={b_in_label}: expected at most one "
                            f"match on a 1-in/1-out pair, got {len(matches)}"
                        )
                        result = apply(diagram, SPIDER_FUSION, matches[0])
                        diagrams_fused += 1

                        # Every fused diagram in this sweep resolves shared_dim to a fully
                        # concrete value (A's boundary input leg is fixed at concrete 2 in
                        # every diagram this sweep builds -- see _build_diagram -- and the
                        # fixpoint always forces shared_dim to unify with it), so
                        # result.diagram genuinely contains neither `d` nor `e` as a free
                        # symbol whenever a match actually fused. The anti-laundering
                        # invariant below is scoped to exactly the cases where that is not
                        # true -- see its own comment for why an unconditional version would
                        # be unsound, not merely broad.
                        post_free_symbols = _diagram_free_symbols(result.diagram)

                        for d_val, e_val in _D_E_PAIRS:
                            assignment = {"d": d_val, "e": e_val}
                            instantiated_pre = instantiate(diagram, assignment)
                            pre_report = validate(instantiated_pre)
                            pre_clean = pre_report.is_valid and not pre_report.deferred

                            instantiated_post = instantiate(result.diagram, assignment)
                            post_report = validate(instantiated_post)
                            post_clean = post_report.is_valid and not post_report.deferred

                            # The anti-laundering invariant: a rewrite must never leave a
                            # dimension symbol dangling in the post-fusion diagram in a way
                            # that lets it validate at an assignment the pre-fusion diagram
                            # rejects -- e.g. a merged node keeping `Dim(d)` on its legs and
                            # phase container while the frozen numeric phase entry is only
                            # correct at the one value the binding assumed.
                            #
                            # Scoped to assignment keys the post-fusion diagram still
                            # mentions: once a fusion fully resolves a symbol, it is gone
                            # from result.diagram and re-instantiating it is a no-op, so
                            # "validates regardless of what d is assigned" is then simply
                            # correct. Checking unconditionally would fail on correct code
                            # whenever a leg is forced concrete by an unrelated surviving
                            # leg. See TestPinnedRegressionReproductions for a shape where a
                            # genuinely dangling symbol is exercised end to end.
                            symbols_still_live = assignment.keys() & post_free_symbols
                            if symbols_still_live and post_clean and not pre_clean:
                                raise AssertionError(
                                    f"a_out={a_out_label} b_in={b_in_label} phase_a={phase_a} "
                                    f"phase_b={phase_b} d={d_val} e={e_val}: anti-laundering "
                                    "invariant violated -- the post-fusion diagram still "
                                    f"mentions {sorted(symbols_still_live)} "
                                    "and validates at this assignment, but the pre-fusion "
                                    f"diagram does not (pre issues: {pre_report.issues})"
                                )

                            if not pre_clean:
                                # "Cleanly contractible" is checked per (d, e) assignment,
                                # not only on the pre-substitution symbolic diagram: a
                                # diagram whose unbound symbols unify freely (e.g. a leg
                                # over bare `e` against one over bare `d`) is clean
                                # symbolically, but a concrete assignment that sets d != e
                                # can still make two now-concrete legs disagree outright --
                                # exactly the audit brief's own "where the input is cleanly
                                # contractible" qualifier. The anti-laundering invariant
                                # above has already been checked for this (pre, post) pair
                                # regardless; only the oracle comparison below needs a clean
                                # pre-fusion diagram to be meaningful.
                                skipped_not_clean += 1
                                continue

                            comparison = compare(diagram, result.diagram, assignment)
                            assert comparison.matched, (
                                f"a_out={a_out_label} b_in={b_in_label} phase_a={phase_a} "
                                f"phase_b={phase_b} d={d_val} e={e_val}: {comparison.reason}"
                            )
                            assert post_clean, (
                                f"a_out={a_out_label} b_in={b_in_label} d={d_val} e={e_val}: "
                                f"post-fusion diagram is not cleanly contractible at this "
                                f"assignment despite a clean pre-fusion diagram: "
                                f"{post_report.issues}"
                            )
                            checked += 1

        assert diagrams_fused > 0, "the sweep never produced a single fused diagram"
        assert checked >= 30, (
            f"only {checked} comparisons ran over {diagrams_fused} fused diagrams "
            f"(skipped_not_clean={skipped_not_clean}, skipped_no_match={skipped_no_match}) "
            "-- suspiciously low for this sweep"
        )


class TestPinnedRegressionReproductions:
    """Two literal witness diagrams, pinned as permanent regressions.

    Unlike :class:`TestSymbolicDimensionSweep`, which sweeps a combinatorial space and could
    stop generating either shape if the palette above ever changed, these two tests build
    the witness diagrams directly, so a regression is caught even if the sweep's own
    coverage shifts.
    """

    def test_defect_1_contradictory_phase_bindings_are_refused_not_accepted(self) -> None:
        """Two present phases must not each bind the same symbol against a stale, unrefined
        ``shared_dim`` (last-write-wins). A Z spider pair, every leg over the symbol ``d``,
        joined by one OUTPUT->INPUT wire; A's phase is over the concrete ``2``, B's over the
        concrete ``3`` -- mutually unsatisfiable, and already refused when the identical
        contradiction is expressed through *legs* instead of phases (see
        :mod:`tests.test_match`'s ``TestSharedDimResolvesThroughBinding`` and
        ``TestSurvivingLegDimensionUnification`` for that leg-side asymmetry check)."""
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[],
            output_dims=[d],
            phase=PhaseVector(Dim.concrete(2), {1: Phase.root_of_unity(1, Dim.concrete(2))}),
        )
        b_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[d],
            output_dims=[],
            phase=PhaseVector(Dim.concrete(3), {1: Phase.root_of_unity(1, Dim.concrete(3))}),
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([])

        report = validate(diagram)
        assert report.is_valid, f"input diagram must validate cleanly: {report.issues}"
        assert find_matches(diagram) == (), (
            "a Z spider pair whose two present phases bind the shared leg symbol to two "
            "different concrete values must not be reported as a fusion match"
        )

    def test_defect_2_a_phase_only_binding_refines_the_merged_legs_not_just_the_phase(
        self,
    ) -> None:
        """A binding a present phase alone produces must refine ``shared_dim`` before the
        merged node's legs are built from it, not only the merged phase's entries --
        otherwise the merged node's legs keep the pre-binding symbol while its phase is
        frozen at the bound value, and the two disagree on the dimension the binding
        assumed. A Z spider pair, every leg over the symbol ``d``; A has one extra surviving
        output leg (so the merged node keeps a leg to disagree with its phase over, exactly
        the audit's own "legs = [Dim(d)]" witness), B carries an empty phase over the
        concrete ``2`` -- B's mere presence binds ``d := 2``.


        Before the fix, the merged node came out as ``legs = [Dim(d)]``,
        ``phase = PhaseVector[d]({1: 1/2 turns})`` -- a container still claiming the
        unresolved symbol ``d`` while its one entry was already the numeric value only
        correct at ``d = 2``. Both fields must now agree on the same concrete value the
        binding actually produced.
        """
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[],
            output_dims=[d, d],
            phase=PhaseVector(d, {1: Phase.root_of_unity(1, d)}),
        )
        b_id = diagram.add_node(
            Z_SPIDER,
            input_dims=[d],
            output_dims=[],
            phase=PhaseVector(Dim.concrete(2), {}),
        )
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(a_id, Direction.OUTPUT, 1)])

        report = validate(diagram)
        assert report.is_valid, f"input diagram must validate cleanly: {report.issues}"
        matches = find_matches(diagram)
        assert len(matches) == 1
        match = matches[0]

        two = Dim.concrete(2)
        assert match.shared_dim == two, (
            "the phase-only binding d := 2 must refine shared_dim itself, not merely be "
            "recorded as a binding the merged phase substitutes through"
        )

        result = apply(diagram, SPIDER_FUSION, match)
        merged_node_id = result.new_node_ids[0]
        merged = result.diagram.nodes[merged_node_id]
        assert len(merged.outputs) == 1, "A's non-consumed output leg must survive"
        assert merged.outputs[0].dim == two, (
            "the surviving leg must be built from the same concrete shared_dim the "
            "phase-only binding produced, not the still-unresolved symbol d"
        )
        assert merged.phase is not None
        assert merged.phase.dim == two, (
            "the merged node's phase must be reattached to the same concrete shared_dim "
            "its own binding produced -- legs and phase must agree, not merely each look "
            "individually plausible"
        )

        # The pre-fusion diagram is contractible only at d = 2 (B's phase-vs-leg dimension
        # disagrees at any other value). The post-fusion diagram no longer mentions `d` at
        # all -- fully resolved to shared_dim=2 on both the surviving leg and the phase --
        # so it validates regardless of what `d` is assigned, which is correct once the
        # binding is baked in. The oracle comparison below is what distinguishes correct
        # from wrong: it catches a merged phase whose frozen entry is right only by
        # coincidence at the assumed value while its container still claims `d`.
        comparison = compare(diagram, result.diagram, {"d": 2})
        assert comparison.matched, comparison.reason
