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

"""Tests for archytaszx.semantics.interop: sampling, round-trip checks, sheet-step checks,
and mixed bang-box / scalable workflows."""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, peel_one
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import (
    BangBoxId,
    Diagram,
    Direction,
    GraphGrammarError,
    NodeId,
    PortRef,
)
from archytaszx.diagram.scalable import (
    ScalableDiagram,
    ScalableGrammarError,
    ScaleId,
    from_scalable,
    to_scalable,
)
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.match import find_matches
from archytaszx.rewrite.normal_form import comparison_view
from archytaszx.rewrite.rules_library import SPIDER_FUSION
from archytaszx.rewrite.sheet import (
    SheetDomainError,
    SheetStep,
    dissolve,
    enclose,
    fuse_sheet,
    join_scales,
    kill_scale,
    split_scale,
)
from archytaszx.semantics import interop
from archytaszx.semantics.check import CheckAssignmentValue, ComparisonResult, compare, score
from archytaszx.semantics.contract_numeric import ContractSizeError
from archytaszx.semantics.interop import (
    InteropGrammarError,
    check_round_trip,
    check_sheet_step,
    default_samples,
    via_scalable,
)

from .helpers import build_ghz_with_copy
from .test_scalable import BANG_BOX_FAMILIES, NOT_BUNDLE_NORMAL, SCALABLE_FAMILIES

D2 = Dim(2)


def _out(node: NodeId, index: int) -> PortRef:
    return PortRef(node, Direction.OUTPUT, index)


def _bang_box_form(build: Callable[[], ScalableDiagram]) -> Diagram:
    return from_scalable(build())


def _all_families() -> dict[str, Callable[[], Diagram]]:
    families = dict(BANG_BOX_FAMILIES)
    for name, build in SCALABLE_FAMILIES.items():
        families[name] = functools.partial(_bang_box_form, build)
    return families


FAMILIES = _all_families()


def phased_ghz(turns: sp.Expr) -> Diagram:
    """A GHZ family over a Z spider whose leg-1 phase is ``turns`` turns."""
    d = Diagram()
    z = d.add_node(Z_SPIDER, [], [D2], phase=PhaseVector(D2, {1: Phase(turns)}))
    d.set_boundary_outputs([_out(z, 0)])
    d.add_bang_box(Mult("n"), port_scope=frozenset({_out(z, 0)}))
    return d


def boxed_pair(multiplicity: Mult) -> Diagram:
    """The GHZ-with-copy fusion pair under one node-scope box."""
    d, a, b = build_ghz_with_copy(D2)
    d.add_bang_box(multiplicity, node_scope=frozenset({a, b}))
    return d


def zero_fan() -> Diagram:
    """A Z state whose second output is fanned zero times."""
    d = Diagram()
    z = d.add_node(Z_SPIDER, [], [D2, D2], phase=PhaseVector(D2))
    d.set_boundary_outputs([_out(z, 0), _out(z, 1)])
    d.add_bang_box(Mult(0), port_scope=frozenset({_out(z, 1)}))
    return d


def engine_fusion(diagram: Diagram) -> Diagram:
    """``diagram`` after spider fusion at its single match, by the bang-box engine."""
    (match,) = find_matches(diagram)
    return apply(diagram, SPIDER_FUSION, match).diagram


def assert_oracle_equal(left: Diagram, right: Diagram) -> None:
    for sample in default_samples(left, right):
        result = compare(left, right, sample)
        assert result.matched, (dict(sample), result.reason)


# -- sampling -------------------------------------------------------------------------------


class TestDefaultSamples:
    def test_a_diagram_without_symbols_gets_one_empty_sample(self) -> None:
        assert [dict(s) for s in default_samples(FAMILIES["plain"]())] == [{}]

    def test_multiplicities_take_0_1_2_and_dimensions_2_3(self) -> None:
        samples = default_samples(FAMILIES["qufinite_mixed_dims"](), limit=100)
        assert len(samples) == 3 * 3 * 2
        assert {s["k"] for s in samples} == {0, 1, 2}
        assert {s["n"] for s in samples} == {0, 1, 2}
        assert {s["d"] for s in samples} == {2, 3}

    def test_diagonals_come_first_and_the_limit_caps(self) -> None:
        samples = default_samples(FAMILIES["nested_copies"](), limit=4)
        assert [dict(s) for s in samples] == [
            {"k1": 0, "k2": 0},
            {"k1": 1, "k2": 1},
            {"k1": 2, "k2": 2},
            {"k1": 0, "k2": 1},
        ]

    def test_samples_are_deterministic_and_read_only(self) -> None:
        assert [dict(s) for s in default_samples(Diagram())] == [{}]
        samples = default_samples(FAMILIES["symbolic_phase"]())
        assert samples == default_samples(FAMILIES["symbolic_phase"]())
        with pytest.raises(TypeError):
            samples[0]["k"] = 5  # type: ignore[index]

    def test_phase_symbols_respect_assumptions(self) -> None:
        j = sp.Symbol("j", integer=True, positive=True)
        samples = default_samples(phased_ghz(j), limit=100)
        assert {s["j"] for s in samples} == {1, 2}
        r = sp.Symbol("r")
        assert {s["r"] for s in default_samples(phased_ghz(r), limit=100)} == {
            sp.Rational(1, 3),
            sp.Rational(1, 4),
        }

    def test_union_over_diagrams_and_scalable_input(self) -> None:
        s = SCALABLE_FAMILIES["sheet_family"]()
        names = set(default_samples(s, FAMILIES["ghz"]())[0])
        assert names == {"k", "m", "n"}

    def test_a_bound_symbol_gets_a_final_sample_without_it(self) -> None:
        samples = [dict(s) for s in default_samples(FAMILIES["ghz_abstracted"]())]
        assert samples == [{"n": 0}, {"n": 1}, {"n": 2}, {}]
        assert [dict(s) for s in default_samples(FAMILIES["ghz_abstracted"](), limit=2)] == [
            {"n": 0},
            {},
        ]
        assert [dict(s) for s in default_samples(FAMILIES["ghz_abstracted"](), limit=1)] == [
            {"n": 0}
        ]

    def test_bad_arguments(self) -> None:
        with pytest.raises(InteropGrammarError):
            default_samples(FAMILIES["ghz"](), limit=0)
        with pytest.raises(InteropGrammarError):
            default_samples(FAMILIES["ghz"](), limit=True)
        with pytest.raises(InteropGrammarError):
            default_samples("ghz")  # type: ignore[arg-type]


# -- round trips ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_round_trip_is_ok_on_every_family(name: str) -> None:
    diagram = FAMILIES[name]()
    report = check_round_trip(diagram)
    assert report.ok, report.reason
    assert report.stripped_ok and report.oracle_ok
    assert report.samples_evaluated == len(default_samples(diagram)) > 0
    assert report.counterexample is None
    assert report.identical is (name not in NOT_BUNDLE_NORMAL), report.reason
    assert isinstance(report.scalable, ScalableDiagram)
    assert isomorphic(report.back, from_scalable(report.scalable))


class TestCorruptedRoundTrip:
    def test_a_dropped_scalar_is_caught_with_a_counterexample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = from_scalable

        def lossy(s: ScalableDiagram) -> Diagram:
            back = real(s)
            back.set_scalar(Scalar.one())
            return back

        monkeypatch.setattr(interop, "from_scalable", lossy)
        report = check_round_trip(FAMILIES["symbolic_phase"]())
        assert not report.ok and not report.oracle_ok
        assert report.stripped_ok
        assert report.counterexample is not None
        assert "oracle fails" in report.reason
        assert not report.identical

    def test_a_wrong_strip_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(interop, "strip", lambda s, a: Diagram())
        report = check_round_trip(FAMILIES["ghz"]())
        assert not report.ok and not report.stripped_ok and report.oracle_ok
        assert dict(report.counterexample or {}) == {"n": 0}
        assert "strip fails" in report.reason

    def test_an_unevaluable_back_side_is_a_counterexample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(s: ScalableDiagram) -> Diagram:
            d = Diagram()
            d.add_node(Z_SPIDER, [], [Dim("zz")])
            return d

        monkeypatch.setattr(interop, "from_scalable", broken)
        report = check_round_trip(FAMILIES["ghz"]())
        assert not report.oracle_ok
        assert "cannot be evaluated" in report.reason


def _corrupting(edit: Callable[[Diagram], None]) -> Callable[[ScalableDiagram], Diagram]:
    def corrupted(s: ScalableDiagram) -> Diagram:
        back = from_scalable(s)
        edit(back)
        return back

    return corrupted


def _drop_first_wire(d: Diagram) -> None:
    wire = min(d.wires, key=lambda w: w.sort_key())
    d.remove_wire(wire.a, wire.b)


def _swap_first_outputs(d: Diagram) -> None:
    first, second, *rest = d.boundary_outputs
    d.set_boundary_outputs([second, first, *rest])


class TestSubtleRoundTripCorruptions:
    @pytest.mark.parametrize(
        ("name", "edit"),
        [
            ("fusion_pair", _drop_first_wire),
            ("nested_sheets", _drop_first_wire),
            ("compound", _swap_first_outputs),
            ("fusion_pair", _swap_first_outputs),
            ("sheet_family", _swap_first_outputs),
            ("ghz_abstracted", lambda d: d.set_parameters({})),
            ("nested_sheets", lambda d: d.set_parameters({"a": 3})),
        ],
    )
    def test_each_is_caught_with_default_samples(
        self, monkeypatch: pytest.MonkeyPatch, name: str, edit: Callable[[Diagram], None]
    ) -> None:
        monkeypatch.setattr(interop, "from_scalable", _corrupting(edit))
        report = check_round_trip(FAMILIES[name]())
        assert not report.ok and not report.family_equal, report.reason
        assert report.counterexample is not None
        assert "family check fails" in report.reason

    def test_a_dropped_binding_fails_the_oracle_at_the_binding_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(interop, "from_scalable", _corrupting(lambda d: d.set_parameters({})))
        report = check_round_trip(FAMILIES["ghz_abstracted"]())
        assert not report.oracle_ok
        assert "oracle fails at {}" in report.reason


class TestOneSidedErrors:
    def _patched_compare(self, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
        real = compare

        def flaky(
            a: Diagram, b: Diagram, sample: Mapping[str, object], **kwargs: object
        ) -> ComparisonResult:
            if sample.get("n") == 2:
                raise error
            return real(a, b, sample, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(interop, "compare", flaky)

    def test_a_domain_error_on_the_other_side_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patched_compare(monkeypatch, ContractSizeError("too big"))
        report = check_round_trip(FAMILIES["ghz"]())
        assert report.ok and report.samples_evaluated == 2, report.reason
        assert "1 sample(s) skipped: only the reference evaluates" in report.reason

    def test_a_non_domain_error_of_the_reference_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(diagram: Diagram, sample: Mapping[str, CheckAssignmentValue]) -> object:
            if sample.get("n") == 1:
                raise GraphGrammarError("stale ref")
            return score(diagram, sample)

        monkeypatch.setattr(interop, "score", broken)
        report = check_round_trip(FAMILIES["ghz"]())
        assert not report.ok and not report.oracle_ok and report.family_equal
        assert dict(report.counterexample or {}) == {"n": 1}
        assert "the reference cannot be evaluated: GraphGrammarError" in report.reason

    def test_any_other_error_on_the_other_side_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patched_compare(monkeypatch, GraphGrammarError("dangling ref"))
        report = check_round_trip(FAMILIES["ghz"]())
        assert not report.ok and not report.oracle_ok
        assert dict(report.counterexample or {}) == {"n": 2}
        assert "GraphGrammarError: dangling ref" in report.reason


class TestRoundTripSamples:
    def test_no_samples_is_not_ok_and_says_so(self) -> None:
        report = check_round_trip(FAMILIES["ghz"](), samples=[])
        assert not report.ok and report.samples_evaluated == 0
        assert report.family_equal and report.stripped_ok and report.oracle_ok
        assert report.counterexample is None
        assert "no sample evaluated" in report.reason

    def test_impossible_and_inadmissible_samples_are_skipped(self) -> None:
        j = sp.Symbol("j", integer=True)
        report = check_round_trip(
            phased_ghz(j), samples=[{"n": -1, "j": 1}, {"n": 1, "j": sp.Rational(1, 3)}]
        )
        assert not report.ok and report.samples_evaluated == 0
        assert report.counterexample is None
        assert "2 sample(s) skipped: outside the reference's domain" in report.reason
        report = check_round_trip(phased_ghz(j), samples=[{"n": -1, "j": 1}, {"n": 2, "j": 1}])
        assert report.ok and report.samples_evaluated == 1
        assert "1 sample(s) skipped: outside the reference's domain" in report.reason

    def test_a_sample_too_large_to_contract_still_gets_structural_checks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = from_scalable

        def swapped(s: ScalableDiagram) -> Diagram:
            back = real(s)
            first, second, *rest = back.boundary_outputs
            back.set_boundary_outputs([second, first, *rest])
            return back

        monkeypatch.setattr(interop, "from_scalable", swapped)
        report = check_round_trip(FAMILIES["multi_port"](), samples=[{"n": 40}])
        assert not report.family_equal and not report.ok and report.oracle_ok
        assert report.samples_evaluated == 0
        assert dict(report.counterexample or {}) == {"n": 40}
        assert "family check fails" in report.reason
        assert "skipped: outside the reference's domain" in report.reason

    def test_explicit_samples_are_used(self) -> None:
        report = check_round_trip(FAMILIES["compound"](), samples=[{"k": 3}])
        assert report.ok and report.samples_evaluated == 1

    def test_bad_requests(self) -> None:
        with pytest.raises(InteropGrammarError):
            check_round_trip("ghz")  # type: ignore[arg-type]
        with pytest.raises(InteropGrammarError):
            check_round_trip(FAMILIES["ghz"](), samples="n")  # type: ignore[arg-type]
        with pytest.raises(InteropGrammarError):
            check_round_trip(FAMILIES["ghz"](), samples=[1])  # type: ignore[list-item]
        bad = FAMILIES["ghz"]()
        bad.add_bang_box(Mult("m"), port_scope=frozenset({_out(NodeId(7), 0)}))
        with pytest.raises(ScalableGrammarError):
            check_round_trip(bad)


# -- sheet steps ----------------------------------------------------------------------------


def _steps() -> dict[str, SheetStep]:
    pair = to_scalable(boxed_pair(Mult("m")))
    (wire,) = pair.wires
    halves = split_scale(
        to_scalable(boxed_pair(Mult("k") * 2)), ScaleId(0), Mult("k"), dissolve_unit=False
    )
    d, a, b = build_ghz_with_copy(D2)
    enclosed = enclose(to_scalable(d), [a, b])
    return {
        "fuse_sheet": fuse_sheet(pair, wire),
        "split_scale": split_scale(to_scalable(boxed_pair(Mult("k") + 1)), ScaleId(0), Mult("k")),
        "split_scale_keep_unit": halves,
        "join_scales": join_scales(halves.after, ScaleId(0), ScaleId(1)),
        "enclose": enclosed,
        "dissolve": dissolve(enclosed.after, ScaleId(0)),
        "kill_scale": kill_scale(to_scalable(boxed_pair(Mult(0))), ScaleId(0)),
        "kill_legs": kill_scale(to_scalable(zero_fan()), ScaleId(0)),
    }


STEPS = _steps()


@pytest.mark.parametrize("name", sorted(STEPS))
def test_every_operation_checks(name: str) -> None:
    check = check_sheet_step(STEPS[name])
    assert check.ok, check.reason
    assert check.replayed and check.valid and check.oracle_ok
    assert check.samples_evaluated > 0 and check.counterexample is None


class TestTamperedSteps:
    def test_a_wrong_after_fails_replay_and_the_oracle(self) -> None:
        step = STEPS["fuse_sheet"]
        wrong = replace(
            step, after=replace(step.after, scalar=step.after.scalar * Scalar.rational(2))
        )
        check = check_sheet_step(wrong)
        assert not check.ok and not check.replayed and check.valid and not check.oracle_ok
        assert check.counterexample is not None
        assert "replay failed" in check.reason and "oracle fails" in check.reason

    def test_tampered_arguments_fail_replay_only(self) -> None:
        step = STEPS["split_scale"]
        arguments = tuple((k, Mult(1) if k == "first" else v) for k, v in step.arguments)
        check = check_sheet_step(replace(step, arguments=arguments))
        assert not check.ok and not check.replayed and check.valid and check.oracle_ok

    def test_an_unknown_operation_fails_replay(self) -> None:
        check = check_sheet_step(replace(STEPS["enclose"], operation="teleport"))
        assert not check.replayed and check.oracle_ok

    def test_an_invalid_after_is_reported(self) -> None:
        step = STEPS["fuse_sheet"]
        broken = replace(step.after, outputs=())
        check = check_sheet_step(replace(step, after=broken))
        assert not check.valid and not check.ok
        assert "after is invalid" in check.reason

    def test_a_denotationally_wrong_after_is_caught_with_a_counterexample(self) -> None:
        step = STEPS["split_scale"]
        other = STEPS["split_scale_keep_unit"]
        check = check_sheet_step(replace(step, after=other.before), samples=[{"k": 1}, {"k": 2}])
        assert not check.oracle_ok and dict(check.counterexample or {}) == {"k": 2}
        assert check.samples_evaluated == 2

    def test_a_sample_only_before_evaluates_at_is_skipped_and_noted(self) -> None:
        d = Diagram()
        a = d.add_node(Z_SPIDER, [], [D2])
        b = d.add_node(Z_SPIDER, [D2], [D2])
        d.add_wire(_out(a, 0), PortRef(b, Direction.INPUT, 0))
        d.set_boundary_outputs([_out(b, 0)])
        d.add_bang_box(Mult("n"), port_scope=frozenset({_out(b, 0)}))
        s = to_scalable(d)
        (wire,) = s.wires
        check = check_sheet_step(fuse_sheet(s, wire))
        assert check.ok and check.samples_evaluated == 2, check.reason
        assert "1 sample(s) skipped" in check.reason
        assert not check_sheet_step(fuse_sheet(s, wire), samples=[{"n": 0}]).ok

    def test_no_samples_is_not_ok_and_says_so(self) -> None:
        check = check_sheet_step(STEPS["fuse_sheet"], samples=[])
        assert not check.ok and check.samples_evaluated == 0
        assert check.replayed and check.valid and check.oracle_ok
        assert "no sample evaluated" in check.reason

    @pytest.mark.parametrize("side", ["before", "after"])
    def test_a_side_without_a_bang_box_form_is_reported_not_raised(self, side: str) -> None:
        step = STEPS["fuse_sheet"]
        s = getattr(step, side)
        orphaned = replace(s, nodes=tuple(replace(n, scale=None) for n in s.nodes))
        check = check_sheet_step(replace(step, **{side: orphaned}))
        assert not check.valid and not check.ok and check.samples_evaluated == 0
        assert f"{side} is invalid" in check.reason

    def test_bad_requests(self) -> None:
        with pytest.raises(InteropGrammarError):
            check_sheet_step("step")  # type: ignore[arg-type]
        with pytest.raises(InteropGrammarError):
            check_sheet_step(STEPS["fuse_sheet"], samples=({"m": 1}, 2))  # type: ignore[arg-type]


# -- mixed workflows ------------------------------------------------------------------------


class TestViaScalable:
    def test_fusion_under_a_box_matches_engine_fusion(self) -> None:
        start = boxed_pair(Mult("m"))
        (wire,) = start.wires
        outcome = via_scalable(start, [("fuse_sheet", {"wire": wire})])
        assert outcome.ok, [c.reason for c in outcome.checks]
        assert outcome.start is start and len(outcome.steps) == len(outcome.checks) == 1
        engine = engine_fusion(start)
        assert isomorphic(outcome.result, engine)
        assert isomorphic(comparison_view(outcome.result), comparison_view(engine))
        assert_oracle_equal(outcome.result, engine)
        assert_oracle_equal(outcome.result, start)

    def test_peel_via_split(self) -> None:
        start = boxed_pair(Mult("k") + 1)
        outcome = via_scalable(start, [("split_scale", {"scale": 0, "first": Mult("k")})])
        assert outcome.ok
        assert isomorphic(outcome.result, peel_one(start, BangBoxId(0)).diagram)

    def test_users_may_work_in_either_notation_losslessly(self) -> None:
        start = boxed_pair(Mult("k") + 1)
        (wire,) = start.wires
        bang_box_only = peel_one(engine_fusion(start), BangBoxId(0)).diagram
        scalable_only = via_scalable(
            start,
            [("fuse_sheet", {"wire": wire}), ("split_scale", {"scale": 0, "first": Mult("k")})],
        )
        mixed = via_scalable(
            engine_fusion(start),
            [
                ("split_scale", {"scale": 0, "first": Mult("k"), "dissolve_unit": False}),
                ("join_scales", {"first": 0, "second": 1}),
                ("split_scale", {"scale": 0, "first": Mult("k")}),
            ],
        )
        assert scalable_only.round_trip.identical
        for outcome in (scalable_only, mixed):
            assert outcome.ok, outcome.round_trip.reason
            assert isomorphic(outcome.result, bang_box_only)
            assert_oracle_equal(outcome.result, start)
            back = via_scalable(outcome.result, [])
            assert back.ok and back.round_trip.identical and not back.steps
            assert isomorphic(back.result, outcome.result)

    def test_arguments_name_diagram_ids_then_current_state_ids(self) -> None:
        fused = engine_fusion(FAMILIES["plain"]())
        (merged,) = fused.nodes
        assert merged == NodeId(2)
        outcome = via_scalable(
            fused, [("enclose", {"nodes": [merged]}), ("dissolve", {"scale": 0})]
        )
        assert outcome.ok
        assert [step.operation for step in outcome.steps] == ["enclose", "dissolve"]
        assert outcome.steps[0].after.node(merged).scale == ScaleId(0)
        assert outcome.steps[1].after == outcome.steps[0].before
        assert isomorphic(outcome.result, fused)

    def test_refusals_and_bad_requests(self) -> None:
        start = boxed_pair(Mult("m"))
        with pytest.raises(SheetDomainError):
            via_scalable(start, [("split_scale", {"scale": 0, "first": 1})])
        with pytest.raises(InteropGrammarError):
            via_scalable(start, [("teleport", {})])
        with pytest.raises(InteropGrammarError):
            via_scalable(start, [("dissolve",)])  # type: ignore[list-item]
        with pytest.raises(InteropGrammarError):
            via_scalable(start, [("dissolve", {1: 0})])  # type: ignore[dict-item]
        with pytest.raises(InteropGrammarError):
            via_scalable(start, "fuse_sheet")  # type: ignore[arg-type]
        with pytest.raises(InteropGrammarError):
            via_scalable(to_scalable(start), [])  # type: ignore[arg-type]


# -- determinism ----------------------------------------------------------------------------

_DETERMINISM_SCRIPT = """
from archytaszx.diagram.bangbox import Mult
from archytaszx.semantics.interop import check_round_trip, default_samples, via_scalable
from tests.test_interop import FAMILIES, boxed_pair
for name in sorted(FAMILIES):
    report = check_round_trip(FAMILIES[name]())
    print(name, report.ok, report.identical, report.samples_evaluated, report.reason)
    print([sorted(s.items()) for s in default_samples(FAMILIES[name]())])
start = boxed_pair(Mult("k") + 1)
(wire,) = start.wires
outcome = via_scalable(
    start, [("fuse_sheet", {"wire": wire}), ("split_scale", {"scale": 0, "first": Mult("k")})]
)
print(outcome.ok, [c.reason for c in outcome.checks])
print(sorted(w.sort_key() for w in outcome.result.wires), outcome.result.boundary_outputs)
"""


def _run_with_seed(seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    result = subprocess.run(
        [sys.executable, "-c", _DETERMINISM_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_output_is_identical_across_hash_seeds() -> None:
    first = _run_with_seed("0")
    assert first
    assert _run_with_seed("2147483647") == first
