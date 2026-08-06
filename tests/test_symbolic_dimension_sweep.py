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

"""Permanent regression guard: the symbolic-dimension sweep (Phase 5 post-closing audit
probe 2), extended (round-12 audit) to close two blind spots that let defects 1 and 2 (see
:mod:`qufzx.rewrite.match`'s module docstring, condition 6) through undetected.

1-in/1-out spider pairs (A, B), each leg dim drawn from ``{2, 3, d, e}`` (genuinely
symbolic where ``d``/``e`` appear -- these diagrams are built once, symbolically, matched
and fused once, then oracle-compared at several concrete ``(d, e)`` assignments via
:func:`~qufzx.semantics.check.compare`'s own ``assignment`` parameter), and phases drawn
independently of the leg dims (see "Independent phase dimensions" below). This is the arm
that actually exercises :func:`~qufzx.rewrite.match._resolve_with_bindings`,
:func:`~qufzx.rewrite.match._unify_surviving_legs`,
:func:`~qufzx.rewrite.match._unify_phase_dims`, and :func:`~qufzx.rewrite.match.reattach_phase`
together -- a phase legally stated over a symbolic dimension that a fusion's own unify
resolves through a binding, exactly the ``_over_shared_dim`` defect family
:mod:`qufzx.rewrite.rules_library`'s module docstring describes at length.

Independent phase dimensions (round-12 audit, closing blind spot 2): a node's phase is no
longer only ever offered over its *own* leg's dim -- two additional choices, root-of-unity
phases over the fixed concrete dims ``2`` and ``3``, are offered regardless of what the
connecting leg's own dim is for that diagram. This is what lets the sweep generate defect
1's shape at all: a phase over a concrete dim that has nothing to do with the (possibly
symbolic) leg dim it will end up sharing a node with, on *both* A and B independently, so a
diagram where A's phase and B's phase would bind the same shared symbol to two different
concrete values (defect 1's exact reproduction) is reachable by this sweep, not merely
possible in principle.

Anti-laundering invariant, asserted not skipped (round-12 audit, closing blind spot 1): for
every fused diagram, at every concrete ``(d, e)`` assignment, this arm now asserts the exact
contrapositive of defect 2 -- if the post-fusion diagram is cleanly contractible at that
assignment, the pre-fusion diagram must be too. The prior version silently ``continue``d
whenever the pre-fusion diagram was not cleanly contractible at a given assignment, which is
exactly the condition defect 2's own witness diagrams satisfy (the pre-fusion diagram is
only contractible at the one dimension value the laundered assumption assumed), so that arm
was structurally blind to the defect it was meant to guard against.

Deliberate subsampling (stated, per ``claude.md``'s standing instruction), smaller than the
audit brief's ~25,800 comparisons:

* The audit brief does not say whether ``{2, 3, d, e}`` applies to all four legs (A's
  input and output, B's input and output) independently, or only to the connecting pair.
  This sweep applies the full 4-symbol palette to the *connecting* pair (A's output, B's
  input) -- the 16 combinations there exercise every :meth:`~qufzx.algebra.dimension.Dim.unify`
  outcome type (syntactic identity, binding, and outright failure -- ``DEFERRED`` does not
  arise from this simple, product/power-free palette, see below) on the pair that actually
  drives ``shared_dim`` -- and fixes each node's *surviving* leg (A's input, B's output) at
  one representative, already-mixed pair (a concrete ``2`` and the symbolic ``d``), rather
  than sweeping all 16 combinations there too:
  :func:`~qufzx.rewrite.match._unify_surviving_legs` is already exercised meaningfully by a
  single symbolic surviving leg (see ``tests/test_match.py``'s
  ``TestSurvivingLegDimensionUnification``, which uses exactly this shape), and the
  connecting pair is where the combinatorics actually matter for this arm's stated purpose.
* Only two ``(d, e)`` oracle-substitution pairs are checked (``(2, 3)`` and ``(3, 2)``), not
  all four the brief lists -- the same-value pairs ``(2, 2)``/``(3, 3)`` are already covered
  elsewhere (``tests/test_phase5_oracle.py``'s ``_CONCRETE_DS`` sweep); this arm's own
  purpose is specifically the ``d != e`` case a same-value substitution cannot exercise.
* ``DEFERRED`` genuinely never arises from ``{2, 3, d, e}`` alone (it needs a symbol
  occurring as a proper subterm of the other side, e.g. ``d`` against ``d*e`` -- see
  :mod:`qufzx.rewrite.match`'s module docstring, condition 5), so "clean" here means
  ``validate(diagram).is_valid`` alone; every match this arm finds is a genuine
  syntactic-identity or single-binding case, never a deferred one (that shape is covered by
  the existing fuzz harness's ``d*e``/``d**2`` palette entries in
  ``tests/test_fusion_properties.py``).
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

                            # The anti-laundering invariant (round-12 audit, closing blind
                            # spot 1): a rewrite must never leave a dimension symbol dangling
                            # in the post-fusion diagram in a way that lets it validate at an
                            # assignment the pre-fusion diagram genuinely rejects -- exactly
                            # defect 2's failure mode (the merged node kept `Dim(d)` on its
                            # legs and phase container alike, so instantiating `d` at any
                            # value looked self-consistent to `validate()`, even though the
                            # frozen numeric phase entry was only actually correct at the one
                            # value the binding assumed). This is deliberately scoped to
                            # ``assignment`` keys the post-fusion diagram still mentions: once
                            # a fusion fully resolves a symbol to a concrete value (as every
                            # diagram in this sweep does), the symbol is gone from
                            # ``result.diagram`` entirely, and re-``instantiate``-ing at a
                            # different value for it is a no-op -- so "the post-fusion
                            # diagram validates regardless of what `d` is assigned" is then
                            # simply true and correct (the certificate's own `bindings`
                            # records the assumption; see rules_library.py's "Phase 5
                            # judgement call 1"), not a laundering bug. Checking it
                            # unconditionally would make this assertion fail on correct code
                            # for the ordinary, ever-present case of a leg forced concrete by
                            # an unrelated surviving leg (e.g. a_out=b_in="2"), which has
                            # nothing to do with defect 2. See
                            # ``TestPinnedRegressionReproductions`` below for a diagram shape
                            # where a genuinely dangling symbol -- the actual defect 2 shape
                            # -- is exercised end to end instead.
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
    """Permanent regressions pinned to the round-12 audit's two literal reproductions.

    Unlike :class:`TestSymbolicDimensionSweep`, which sweeps a combinatorial space and could
    in principle stop generating either shape if the palette above ever changed, these two
    tests build the audit's exact witness diagrams directly, so a regression on either
    defect is caught even if the sweep's own coverage were to shift.
    """

    def test_defect_1_contradictory_phase_bindings_are_refused_not_accepted(self) -> None:
        """Defect 1: two present phases must not each bind the same symbol against a stale,
        unrefined ``shared_dim`` (last-write-wins). A Z spider pair, every leg over the
        symbol ``d``, joined by one OUTPUT->INPUT wire; A's phase is over the concrete ``2``,
        B's over the concrete ``3`` -- mutually unsatisfiable, and already refused when the
        identical contradiction is expressed through *legs* instead of phases (see
        :mod:`tests.test_match`'s ``TestSharedDimResolvesThroughBinding`` and
        ``TestSurvivingLegDimensionUnification`` for that leg-side asymmetry check).
        """
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
        """Defect 2: a binding a present phase alone produces must refine ``shared_dim``
        before the merged node's legs are built from it, not only the merged phase's
        entries -- otherwise the merged node's legs keep the pre-binding symbol while its
        phase is frozen at the bound value, and the two disagree on the dimension the
        binding assumed. A Z spider pair, every leg over the symbol ``d``; A has one extra
        surviving output leg (so the merged node keeps a leg to disagree with its phase over,
        exactly the audit's own "legs = [Dim(d)]" witness), B carries an empty phase over the
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

        # The pre-fusion diagram is contractible only at d = 2 (B's own phase-vs-leg
        # dimension disagrees at any other value). The post-fusion diagram, since it no
        # longer mentions `d` at all -- fully resolved to shared_dim=2 on both the
        # surviving leg and the phase, per the assertions above -- validates and
        # instantiates identically regardless of what `d` is assigned; that is the correct,
        # intended behavior once the binding is baked in (see rules_library.py's "Phase 5
        # judgement call 1"), not a laundering bug -- the assumption itself is what the
        # shared_dim/phase.dim assertions above already pin down. The oracle comparison
        # below is the check that actually distinguishes correct from wrong: before the fix,
        # the merged phase's frozen entry (1/2 turns) was correct only by coincidence at
        # d = 2 while the container still claimed `d`; comparing against the pre-fusion
        # diagram at the assumed value catches any future regression in that substitution.
        comparison = compare(diagram, result.diagram, {"d": 2})
        assert comparison.matched, comparison.reason
