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

"""Phase 19 commands: every command's success path, the derivation, oracle, purity and undo
invariants, mode independence, the error contract and the detail payloads."""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.graph import Diagram
from archytaszx.diagram.scalable import ScalableDiagram
from archytaszx.repl import commands as C
from archytaszx.repl.commands import (
    ALIASES,
    COMMANDS,
    HELP,
    CommandDomainError,
    CommandError,
    CommandGrammarError,
    ErrorDetail,
    Outcome,
    Proof,
    Session,
    State,
    Text,
    command_names,
    detail_text,
    execute,
    format_outcome,
    run,
)
from archytaszx.repl.parser import ParseError, parse_diagram
from archytaszx.repl.printer import NotationMode, render, render_dirac, render_graph
from archytaszx.rewrite.engine import RewriteResult
from archytaszx.rewrite.normal_form import NormalForm
from archytaszx.rewrite.rules_library import RULES
from archytaszx.semantics import kernel
from archytaszx.semantics.certificate import (
    Certificate,
    Derivation,
    VerificationReport,
    certify,
    compare_structure,
    verify,
)
from archytaszx.semantics.check import compare
from archytaszx.semantics.decide import decide_equal, sample_grid
from archytaszx.semantics.interop import RoundTripReport, SheetCheck
from archytaszx.semantics.lean_net import check, lean_project
from archytaszx.semantics.prove import InductionProof, ProofCertificate, ProofCheck

from .helpers import build_ghz_with_copy
from .test_lean_sym import BUILT
from .test_phase15_oracle import PROVED, boxed_state_copy

WORKED = "sum_{k=0}^{3-1} |k>^{30}; copy"
DSL_GHZ = "a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0; outputs a.out1, b.out0, b.out1"
BOXED = "count n; a = Z(0 -> 2, d); b = Z(1 -> 1, d); wire a.out0 -- b.in0; box p x n ports a.out1"
ZERO_PHASE = "a = Z(0 -> 1, d) phase 0"
PLUS = "a = Z(0 -> 1, d)"
COPIES4 = "a = Z(0 -> 1, d); box p x literal 4 nodes a"
TWO_PLUS = "a = Z(0 -> 1, d); b = Z(0 -> 1, d)"
KILLABLE = "a = Z(0 -> 1, d); b = Z(0 -> 1, d); box p x literal 0 nodes a"
UNEQUAL = "sum_{k=0}^{3-1} |k>^{30}"
DIRAC = NotationMode.DIRAC
ZX = NotationMode.ZX
BOTH = NotationMode.BOTH
LISTING_HEADS = ("graph:", "scalable:")
HINT = "(detail shows the engine detail)"
EXPECTED_COMMANDS = {
    "load",
    "show",
    "rules",
    "match",
    "apply",
    "check",
    "normalize",
    "decide",
    "saturate",
    "prove",
    "induct",
    "certificate",
    "translate",
    "name",
    "names",
    "undo",
    "reset",
    "detail",
    "help",
}


# -- helpers -------------------------------------------------------------------------------


def ok(session: Session, line: str) -> Outcome:
    """run ``line``, asserting a successful outcome."""
    outcome = run(session, line)
    assert outcome.ok, f"{line!r}: {format_outcome(outcome)}"
    return outcome


def failed(session: Session, line: str) -> Outcome:
    """run ``line``, asserting a failure report (not an error)."""
    outcome = run(session, line)
    assert not outcome.ok, f"{line!r}: {format_outcome(outcome)}"
    assert not isinstance(outcome.detail, ErrorDetail), format_outcome(outcome)
    return outcome


def errored(session: Session, line: object) -> Outcome:
    """run ``line``, asserting an error outcome with its ErrorDetail stored."""
    outcome = run(session, line)  # type: ignore[arg-type]
    assert not outcome.ok
    assert not outcome.changed
    assert isinstance(outcome.detail, ErrorDetail)
    assert session.last_detail is outcome.detail
    lines = format_outcome(outcome).splitlines()
    assert lines[0].startswith("error: ")
    assert lines[1] == HINT
    return outcome


def session(*lines: str) -> Session:
    """A fresh session with each line run successfully."""
    s = Session()
    for line in lines:
        ok(s, line)
    return s


def texts(outcome: Outcome) -> list[str]:
    """The lines of every Text block."""
    return [line for b in outcome.blocks if isinstance(b, Text) for line in b.text.splitlines()]


def listing_lines(text: str) -> list[str]:
    """The lines opening a graph or scalable listing."""
    return [line for line in text.splitlines() if line.lstrip().startswith(LISTING_HEADS)]


def same(a: Diagram, b: Diagram) -> bool:
    """Node-for-node structural identity."""
    return compare_structure(a, b).identical


def points(left: Diagram, right: Diagram, limit: int = 2) -> list[dict[str, object]]:
    """The first ``limit`` sample-grid points of the pair."""
    return [dict(p) for p in list(sample_grid(left, right))[:limit]]


def assert_replays(s: Session) -> None:
    """certify(start, results) verifies at small samples and ends at the current diagram."""
    if s.start is None:
        assert s.current is None and not s.results
        return
    assert s.current is not None
    certificate = certify(s.start, s.results)
    assert same(certificate.final, s.current), compare_structure(certificate.final, s.current)
    for sample in points(s.start, s.current):
        report = verify(certificate, sample)
        assert report.verified, (sample, report.reason)


def assert_oracle(s: Session) -> None:
    """The start and current diagrams agree numerically at small samples."""
    if s.start is None or s.current is None:
        return
    for sample in points(s.start, s.current):
        result = compare(s.start, s.current, sample)
        assert result.matched, (sample, result.reason)


def assignment_of(line: str) -> dict[str, int]:
    """The ``NAME = INT`` pairs of a counterexample line."""
    head = line.split(":", 2)[1]
    return {k: int(v) for k, v in re.findall(r"(\w+) = (-?\d+)", head)}


def signature(outcome: Outcome) -> tuple[object, ...]:
    """Command, ok, changed and the blocks, State blocks by type and graph listing."""
    blocks: list[object] = []
    for block in outcome.blocks:
        if isinstance(block, Text):
            blocks.append(block.text)
        elif isinstance(block, Proof):
            blocks.append(("proof", type(block.proof).__name__))
        elif isinstance(block.state, Diagram):
            blocks.append(("state", render_graph(block.state)))
        else:
            blocks.append(("state", type(block.state).__name__))
    return (outcome.command, outcome.ok, outcome.changed, tuple(blocks))


def diagram_states(outcome: Outcome) -> Iterator[Diagram]:
    """The Diagram held by each State block, a RewriteResult's diagram included."""
    for block in outcome.blocks:
        if isinstance(block, State):
            if isinstance(block.state, Diagram):
                yield block.state
            elif isinstance(block.state, RewriteResult):
                yield block.state.diagram


# -- shared sessions -----------------------------------------------------------------------

MODE_SCRIPT: tuple[str, ...] = (
    WORKED,
    "show",
    "match",
    "apply spider_fusion",
    "check",
    "check n=2",
    "normalize",
    "decide",
    f"decide against {UNEQUAL}",
    "prove",
    "induct",
    "certificate emit",
    "certificate verify",
    "show derivation",
    "show certificate",
    "name fused",
    "names",
    "translate",
    "show scalable",
    "translate back",
    "saturate",
    "rules spider_fusion",
    "apply nosuch",
    "check d=0",
    "undo",
    "reset",
    "help apply",
)


@pytest.fixture(scope="module")
def mode_runs() -> tuple[list[Outcome], list[Outcome]]:
    """MODE_SCRIPT run in two independent sessions."""
    first, second = Session(), Session()
    return [run(first, line) for line in MODE_SCRIPT], [run(second, line) for line in MODE_SCRIPT]


def _payload_script() -> list[tuple[Session, list[str]]]:
    left, right = boxed_state_copy()
    induction = Session()
    induction.names.update(l=left, r=right)
    return [
        (
            Session(),
            [
                WORKED,
                "show",
                "rules",
                "match",
                "apply spider_fusion",
                "check",
                f"check against {UNEQUAL}",
                "normalize",
                "decide",
                f"decide against {UNEQUAL}",
                "induct",
                "prove",
                "certificate emit",
                "certificate emit proof",
                "certificate verify",
                "certificate emit derivation",
                "certificate verify",
                "show derivation",
                "name fused",
                "names",
                "translate",
                "show scalable",
                "translate back",
                "undo",
                "apply nosuch",
                "help",
            ],
        ),
        (Session(), [WORKED, "saturate", "reset", "undo"]),
        (Session(), [ZERO_PHASE, f"prove against {PLUS}"]),
        (Session(), [BOXED, "induct d=2", "certificate emit", "certificate verify"]),
        (Session(), [DSL_GHZ, "translate fuse n0.out0 -- n1.in0", "translate back"]),
        (induction, ["load @l", "prove against @r", "certificate emit", "certificate verify"]),
        (induction, ["load @l", "decide against @r"]),
    ]


@pytest.fixture(scope="module")
def payloads() -> list[tuple[str, Outcome]]:
    """Every outcome of a broad script set, paired with its line."""
    collected: list[tuple[str, Outcome]] = []
    for s, lines in _payload_script():
        for line in lines:
            collected.append((line, run(s, line)))
    return collected


# -- registry and help ---------------------------------------------------------------------


class TestRegistry:
    """COMMANDS, ALIASES, HELP and command_names."""

    def test_commands_registry(self) -> None:
        assert set(COMMANDS) == EXPECTED_COMMANDS
        assert all(callable(h) for h in COMMANDS.values())

    def test_registries_are_read_only(self) -> None:
        with pytest.raises(TypeError):
            COMMANDS["x"] = COMMANDS["load"]  # type: ignore[index]
        with pytest.raises(TypeError):
            ALIASES["x"] = "load"  # type: ignore[index]

    def test_aliases(self) -> None:
        assert dict(ALIASES) == {"list-rules": "rules"}

    def test_help_covers_commands_and_shell_commands(self) -> None:
        assert EXPECTED_COMMANDS | {
            "dirac",
            "zx",
            "zxmatplot",
            "zxunicode",
            "zxascii",
            "both",
            "listing",
            "raw",
            "verbose_on",
            "verbose_off",
            "config",
            "history",
            "quit",
            "exit",
        } == set(HELP)
        for name, usage in HELP.items():
            assert usage.startswith(name), usage

    def test_command_names(self) -> None:
        names = command_names()
        assert names == tuple(sorted(names))
        assert set(names) == EXPECTED_COMMANDS | {"list-rules"}

    def test_help_lists_every_usage(self) -> None:
        outcome = ok(Session(), "help")
        lines = texts(outcome)
        assert lines[: len(HELP)] == [HELP[n] for n in sorted(HELP)]
        assert lines[-2] == "aliases: list-rules = rules"
        assert lines[-1] == "any other line is load <line>"
        assert not outcome.changed

    @pytest.mark.parametrize("word", ["apply", "zx", "config", "quit"])
    def test_help_command(self, word: str) -> None:
        assert texts(ok(Session(), f"help {word}")) == [HELP[word]]

    def test_help_alias(self) -> None:
        assert texts(ok(Session(), "help list-rules")) == [HELP["rules"]]

    def test_help_unknown(self) -> None:
        errored(Session(), "help nosuch")


# -- load ------------------------------------------------------------------------------------


class TestLoad:
    """load: Dirac, DSL, listing, @NAME, bare input."""

    def test_dirac(self) -> None:
        s = Session()
        outcome = ok(s, f"load {WORKED}")
        assert outcome.command == "load" and outcome.changed
        assert texts(outcome) == ["loaded dirac input"]
        assert s.current is s.start and isinstance(s.current, Diagram)
        assert s.results == [] and outcome.detail is s.current
        assert same(s.current, parse_diagram(WORKED))

    def test_bare_line_is_load(self) -> None:
        a, b = Session(), Session()
        bare, explicit = run(a, WORKED), run(b, f"load {WORKED}")
        assert signature(bare) == signature(explicit)
        assert a.current is not None and b.current is not None
        assert same(a.current, b.current)

    def test_dsl_matches_the_hand_built_example(self) -> None:
        s = Session()
        outcome = ok(s, DSL_GHZ)
        assert texts(outcome) == ["loaded dsl input"]
        assert s.current is not None
        assert same(s.current, build_ghz_with_copy(Dim.symbol("d"))[0])

    @pytest.mark.parametrize("prefix", ["", "load "])
    def test_listing_round_trip(self, prefix: str) -> None:
        source = session(WORKED, "apply spider_fusion")
        assert source.current is not None
        listing = render_graph(source.current)
        s = Session()
        outcome = ok(s, prefix + listing)
        assert texts(outcome) == ["loaded listing input"]
        assert s.current is not None and same(s.current, source.current)
        assert dict(s.current.parameters) == dict(source.current.parameters)

    def test_named(self) -> None:
        s = session(WORKED, "apply spider_fusion", "name fused", WORKED)
        held = s.names["fused"]
        outcome = ok(s, "load @fused")
        assert texts(outcome) == ["loaded @fused"]
        assert s.current is not None and s.current is not held
        assert same(s.current, held)

    def test_builtin_names(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        start, current = s.start, s.current
        assert start is not None and current is not None
        ok(s, "load @start")
        assert s.current is not None and same(s.current, start) and s.current is not start
        ok(s, "load @current")
        assert s.current is not None and same(s.current, start)

    def test_load_clears_the_derivation_and_proof(self) -> None:
        s = session(WORKED, "apply spider_fusion", "prove", "certificate emit")
        assert s.last_proof is not None and s.certificate is not None
        ok(s, DSL_GHZ)
        assert s.results == [] and s.last_proof is None
        assert s.certificate is None and s.certificate_witness is None

    def test_load_drops_the_scalable_form(self) -> None:
        s = session(WORKED, "translate")
        assert s.scalable is not None
        ok(s, DSL_GHZ)
        assert s.scalable is None and s.sheet_steps == []

    def test_dirac_output_lines(self) -> None:
        text = format_outcome(ok(Session(), WORKED), DIRAC)
        assert text.splitlines() == [
            "loaded dirac input",
            "sum_{k0,k1=0}^{d-1} [k0 = k1] |k0>^{n} |k1, k1>",
            "at d = 3, n = 29: sum_{k0,k1=0}^{3-1} [k0 = k1] |k0>^{29} |k1, k1>",
        ]


# -- show ------------------------------------------------------------------------------------


class TestShow:
    """show and its variants."""

    def test_current(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        outcome = ok(s, "show")
        assert outcome.detail is s.current and not outcome.changed
        assert format_outcome(outcome) == render(s.current, DIRAC)

    def test_start(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        outcome = ok(s, "show start")
        assert outcome.detail is s.start

    def test_derivation(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        outcome = ok(s, "show derivation")
        assert isinstance(outcome.detail, Derivation)
        assert len(outcome.detail.steps) == 1
        assert texts(outcome)[0] == "derivation: 1 step from the start diagram"

    def test_scalable(self) -> None:
        s = session(WORKED, "translate")
        outcome = ok(s, "show scalable")
        assert outcome.detail is s.scalable
        assert isinstance(outcome.detail, ScalableDiagram)

    def test_certificate(self) -> None:
        s = session(WORKED, "apply spider_fusion", "certificate emit")
        outcome = ok(s, "show certificate")
        assert outcome.detail is s.certificate

    def test_named(self) -> None:
        s = session(WORKED, "name w", "apply spider_fusion")
        outcome = ok(s, "show @w")
        assert isinstance(outcome.detail, Diagram) and same(outcome.detail, s.names["w"])

    def test_source_leaves_the_session(self) -> None:
        s = session(WORKED)
        current = s.current
        outcome = ok(s, f"show {DSL_GHZ}")
        assert s.current is current and not outcome.changed
        assert isinstance(outcome.detail, Diagram)
        assert same(outcome.detail, parse_diagram(DSL_GHZ))

    @pytest.mark.parametrize("what", ["scalable", "certificate", "@nope"])
    def test_missing(self, what: str) -> None:
        errored(session(WORKED), f"show {what}")


# -- rules and match -------------------------------------------------------------------------


class TestRules:
    """rules, list-rules, rules RULE."""

    def test_names(self) -> None:
        outcome = ok(Session(), "rules")
        assert texts(outcome) == list(RULES)
        assert outcome.command == "rules"

    def test_alias(self) -> None:
        assert signature(ok(Session(), "list-rules")) == signature(ok(Session(), "rules"))

    def test_one_rule(self) -> None:
        lines = texts(ok(Session(), "rules spider_fusion"))
        assert lines[0] == "rule: spider_fusion"
        assert lines[1].startswith("  quantifiers: leg counts ")
        assert lines[2].startswith("  side conditions: distinct_nodes")
        assert lines[3].startswith("  dimension guards: ")
        assert lines[4] == "  scalar: 1"

    @pytest.mark.parametrize("name", sorted(RULES))
    def test_every_rule_describes(self, name: str) -> None:
        lines = texts(ok(Session(), f"list-rules {name}"))
        assert lines[0] == f"rule: {name}" and len(lines) == 5

    def test_unknown(self) -> None:
        errored(Session(), "rules nosuch")


class TestMatch:
    """match and match RULE."""

    def test_all_rules(self) -> None:
        outcome = ok(session(WORKED), "match")
        assert texts(outcome) == ["#0 spider_fusion at n0, n1"]
        assert not outcome.changed

    def test_one_rule(self) -> None:
        assert texts(ok(session(WORKED), "match spider_fusion")) == ["#0 spider_fusion at n0, n1"]

    def test_no_matches(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        assert texts(ok(s, "match spider_fusion")) == ["no matches"]

    def test_requires_a_diagram(self) -> None:
        errored(Session(), "match")


# -- apply -----------------------------------------------------------------------------------


class TestApply:
    """apply RULE, at INT, at nodes."""

    @pytest.mark.parametrize(
        "line", ["apply spider_fusion", "apply spider_fusion at 0", "apply spider_fusion at n0, n1"]
    )
    def test_forms_agree(self, line: str) -> None:
        reference = session(WORKED, "apply spider_fusion")
        s = session(WORKED)
        outcome = ok(s, line)
        assert outcome.changed and outcome.command == "apply"
        assert isinstance(outcome.detail, RewriteResult)
        assert outcome.detail.step.rule_name == "spider_fusion"
        assert s.current is outcome.detail.diagram and s.results == [outcome.detail]
        assert s.current is not None and reference.current is not None
        assert same(s.current, reference.current)
        assert_replays(s)
        assert_oracle(s)

    def test_dirac_output(self) -> None:
        text = format_outcome(ok(session(WORKED), "apply spider_fusion"), DIRAC)
        assert text.splitlines()[0] == "step: spider_fusion (consumed n0, n1, new n2, scalar 1)"
        assert "GHZ_{n+2} = sum_{k=0}^{d-1} |k>^{n} |k, k>" in text

    @pytest.mark.parametrize(
        "line", ["apply nosuch", "apply spider_fusion at 5", "apply spider_fusion at n0, n7"]
    )
    def test_failure_leaves_the_session(self, line: str) -> None:
        s = session(WORKED)
        current, stack = s.current, list(s.undo_stack)
        errored(s, line)
        assert s.current is current and s.results == [] and s.undo_stack == stack


# -- check -----------------------------------------------------------------------------------


class TestCheck:
    """check with no, partial and full assignments, and against a target."""

    def test_no_args(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        outcome = ok(s, "check")
        assert texts(outcome) == ["check: no mismatch in 16 samples over d, n"]
        assert not outcome.changed

    def test_partial(self) -> None:
        lines = texts(ok(session(WORKED, "apply spider_fusion"), "check n=2"))
        assert re.fullmatch(r"check: no mismatch in \d+ samples over d, n", lines[0])
        assert int(lines[0].split()[4]) < 16

    @pytest.mark.parametrize("line", ["check d=2 n=3", "check d = 2, n = 3", "check n=3,d=2"])
    def test_full(self, line: str) -> None:
        lines = texts(ok(session(WORKED, "apply spider_fusion"), line))
        assert lines == ["check: no mismatch in 1 sample over d, n"]

    def test_against_equal(self) -> None:
        ok(session(WORKED, "apply spider_fusion"), "check against @start")

    def test_against_unequal(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        outcome = failed(s, f"check against {UNEQUAL}")
        lines = texts(outcome)
        assert lines[0].startswith("check: mismatch after ")
        assert lines[1] == "dirac outcome: left != right"
        assert lines[2] == "  left:  GHZ_{n+2} = sum_{k=0}^{d-1} |k>^{n} |k, k>"
        assert lines[3] == "  right: Z_{0->n} = sum_{k=0}^{d-1} |k>^{n}"
        assert lines[4] == "disproof: re-checkable; certificate lean"
        assert lines[5].startswith("counterexample: ")
        sample = assignment_of(lines[5])
        assert s.current is not None
        assert not compare(s.current, parse_diagram(UNEQUAL), sample).matched

    def test_rational_phase_value(self) -> None:
        s = session("a = Z(1 -> 1, d) phase {1: theta}")
        ok(s, "check theta=1/3 d=2")

    @pytest.mark.parametrize(
        "line",
        [
            "check d=0",
            "check n=-1",
            "check n=1/2",
            "check q=2",
            "check d=x",
            "check d",
            "check d=1/0",
            "check d=2 d=3",
            "check against",
        ],
    )
    def test_bad_assignment(self, line: str) -> None:
        errored(session(WORKED, "apply spider_fusion"), line)


# -- normalize, decide, saturate -------------------------------------------------------------


class TestNormalize:
    """normalize."""

    def test_reduces_and_extends_the_derivation(self) -> None:
        s = session(WORKED)
        outcome = ok(s, "normalize")
        assert outcome.changed and isinstance(outcome.detail, NormalForm)
        assert s.current is outcome.detail.diagram
        assert len(s.results) == len(outcome.detail.results) >= 1
        assert re.fullmatch(r"normalized in \d+ steps?: fixpoint", texts(outcome)[0])
        assert_replays(s)
        assert_oracle(s)

    def test_fixpoint_is_unchanged(self) -> None:
        s = session(WORKED, "normalize")
        depth = len(s.undo_stack)
        outcome = ok(s, "normalize")
        assert not outcome.changed
        assert texts(outcome) == ["normalized in 0 steps: fixpoint"]
        assert len(s.undo_stack) == depth

    def test_arguments(self) -> None:
        errored(session(WORKED), "normalize now")


class TestDecide:
    """decide: default pair, equal and unequal targets, an unsettled pair."""

    def test_default_pair(self) -> None:
        outcome = ok(session(WORKED, "apply spider_fusion"), "decide")
        assert texts(outcome)[0].startswith("decide: equal by ")
        assert not outcome.changed

    @pytest.mark.parametrize("line", ["decide against @start", "decide @start"])
    def test_equal_target(self, line: str) -> None:
        ok(session(WORKED, "apply spider_fusion"), line)

    def test_unequal_target(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        lines = texts(failed(s, f"decide against {UNEQUAL}"))
        assert lines[0].startswith("decide: unequal by ")
        assert lines[1] == "dirac outcome: left != right"
        counter = [line for line in lines if line.startswith("counterexample: ")]
        assert len(counter) == 1
        assert s.current is not None
        assert not compare(s.current, parse_diagram(UNEQUAL), assignment_of(counter[0])).matched

    def test_unsettled_reports_furthest_simplified_and_bound(self) -> None:
        left, right = boxed_state_copy()
        s = Session()
        s.names.update(l=left, r=right)
        ok(s, "load @l")
        outcome = failed(s, "decide against @r")
        lines = texts(outcome)
        assert lines[0].startswith("decide: unknown by ")
        assert lines[1] == "dirac outcome: left ?= right"
        assert any(line.startswith("furthest-simplified left (normal form") for line in lines)
        assert any(line.startswith("furthest-simplified right (normal form") for line in lines)
        assert lines[-1].startswith("bound: ")
        assert not any(line.startswith("counterexample") for line in lines)
        assert sum(isinstance(b, State) for b in outcome.blocks) == 2


class TestSaturate:
    """saturate."""

    def test_extracts_the_fused_diagram(self) -> None:
        s = session(WORKED)
        outcome = ok(s, "saturate")
        assert outcome.changed
        extraction, report = outcome.detail  # type: ignore[misc]
        assert report.stop_reason.value == "saturated"
        assert s.current is extraction.diagram and len(s.results) == len(extraction.results)
        assert s.current is not None
        assert same(s.current, session(WORKED, "apply spider_fusion").current)  # type: ignore[arg-type]
        assert texts(outcome)[0].startswith("saturation saturated: ")
        assert_replays(s)
        assert_oracle(s)

    def test_fixpoint_is_unchanged(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        current = s.current
        outcome = ok(s, "saturate")
        assert not outcome.changed and s.current is current and len(s.results) == 1

    def test_arguments(self) -> None:
        errored(session(WORKED), "saturate 3")


# -- prove and induct ------------------------------------------------------------------------


class TestProve:
    """prove: search, refutation, not found, induction."""

    def test_default_pair(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        outcome = ok(s, "prove")
        lines = texts(outcome)
        assert lines[0].startswith("prove: proved: ")
        assert lines[1] == "proof by search; certificate emits it"
        assert s.last_proof is not None and s.last_proof.proved
        assert not outcome.changed

    def test_refuted(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        lines = texts(failed(s, f"prove against {UNEQUAL}"))
        assert lines[0].startswith("prove: refuted: ")
        assert lines[1] == "dirac outcome: left != right"
        assert lines[-1].startswith("counterexample: ")
        assert s.current is not None
        assert not compare(s.current, parse_diagram(UNEQUAL), assignment_of(lines[-1])).matched

    def test_not_found_reports_furthest_simplified_and_bound(self) -> None:
        lines = texts(failed(session(ZERO_PHASE), f"prove against {PLUS}"))
        assert lines[0].startswith("prove: not_found: ")
        assert lines[1] == "dirac outcome: left ?= right"
        assert any(line.startswith("furthest-simplified left") for line in lines)
        assert lines[-1].startswith("bound: search exhausted (limits depth ")

    def test_induction(self) -> None:
        left, right = boxed_state_copy()
        s = Session()
        s.names.update(l=left, r=right)
        ok(s, "load @l")
        lines = texts(ok(s, "prove against @r"))
        assert lines[1] == "proof by induction; certificate emits it"
        emitted = ok(s, "certificate emit")
        assert isinstance(emitted.detail, InductionProof)
        assert texts(emitted)[0] == "certificate: induction proof over n, 3 steps"
        again = ok(s, "certificate emit induction")
        assert again.detail is emitted.detail
        verified = ok(s, "certificate verify")
        assert isinstance(verified.detail, ProofCheck) and verified.detail.verified
        assert texts(verified)[-1].startswith("verify: verified: ")
        ok(s, "certificate verify d=2")


class TestInduct:
    """induct: witness, on SYMBOL, missing witness, no certificate."""

    @pytest.mark.parametrize("line", ["induct d=2", "induct on n d=2", "induct on n, d = 2"])
    def test_proves(self, line: str) -> None:
        s = session(BOXED)
        lines = texts(ok(s, line))
        assert lines[0].startswith("induct: proved")
        assert " over n from 0 " in lines[0]
        assert lines[1] == "witness: d = 2"
        assert lines[2] == "induction certificate available; certificate emits it"

    def test_against_target(self) -> None:
        s = session(BOXED, "name b")
        ok(s, "induct d=2 against @b")

    def test_witness_from_the_parameter_environment(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        lines = texts(ok(s, "induct"))
        assert lines[1] == "witness: d = 3"
        assert lines[2] == "no induction certificate was built"
        errored(s, "certificate emit induction")

    @pytest.mark.parametrize(
        ("source", "line", "message"),
        [
            (BOXED, "induct", "induct holds d at a value; supply NAME=VALUE for each"),
            (BOXED, "induct on n", "induct holds d at a value; supply NAME=VALUE for each"),
            (BOXED, "induct on n n=2 d=2", "n is the induction index; it takes no value"),
            (BOXED, "induct n=2 d=2", "n is the induction index; it takes no value"),
            (BOXED, "induct on q d=2", "q is no free multiplicity symbol (n)"),
            (BOXED, "induct q=2", "q not free in the compared diagrams (free: d, n)"),
            (BOXED, "induct on d", "d is no free multiplicity symbol (n)"),
            (DSL_GHZ, "induct", "no bang box: induct needs a free multiplicity symbol"),
        ],
    )
    def test_errors(self, source: str, line: str, message: str) -> None:
        outcome = errored(session(source), line)
        assert texts(outcome)[0] == f"error: {message}"


# -- certificate -----------------------------------------------------------------------------


class TestDecisionProofs:
    """decide and induct verdicts hand the kernel a re-checkable theorem."""

    @pytest.mark.parametrize("name", ["state_copy", "bialgebra_once"])
    def test_decide_equal_is_kernel_checked(self, name: str) -> None:
        left, right = PROVED[name]()
        s = Session()
        s.names.update(l=left, r=right)
        ok(s, "load @l")
        lines = texts(ok(s, "decide against @r"))
        assert "proof: re-checkable; certificate kernel / certificate lean" in lines
        kernel_lines = texts(ok(s, "certificate kernel"))
        assert any(line.startswith("kernel: theorem ") for line in kernel_lines)

    def test_symbolic_contraction_verdict_is_a_normal_form_proof(self, tmp_path: Path) -> None:
        left, right = PROVED["bialgebra_4"]()
        s = Session()
        s.names.update(l=left, r=right)
        ok(s, "load @l")
        lines = texts(ok(s, "decide against @r"))
        assert lines[0].startswith("decide: equal by symbolic_contraction")
        assert "proof: re-checkable; certificate kernel / certificate lean" in lines
        assert isinstance(s.last_proof, kernel.Symbolic)
        kernel_lines = texts(ok(s, "certificate kernel"))
        assert kernel_lines[-1].endswith("by the normal form of symbolic sums")
        target = tmp_path / "sym.lean"
        lean_lines = texts(ok(s, f"certificate lean {target}"))
        assert lean_lines[-1] == f"lean: wrote {target}"
        assert "equal_sym" in target.read_text(encoding="utf-8")
        verified = texts(ok(s, "certificate verify"))[-1]
        assert verified.startswith("verify: verified: certificate lean")

    @staticmethod
    def _boxed_symbolic(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        """The Lean file of a boxed family decided by symbolic contraction."""
        monkeypatch.setattr(C, "decide_equal",
                            functools.partial(decide_equal, rules=(), use_saturation=False))
        left, right = BUILT[name]()
        s = Session()
        s.names.update(l=left, r=right)
        ok(s, "load @l")
        lines = texts(ok(s, "decide against @r"))
        assert lines[0].startswith("decide: equal by symbolic_contraction")
        assert "proof: re-checkable; certificate kernel / certificate lean" in lines
        assert isinstance(s.last_proof, kernel.Symbolic)
        target = tmp_path / "family.lean"
        assert texts(ok(s, f"certificate lean {target}"))[-1] == f"lean: wrote {target}"
        return target.read_text(encoding="utf-8")

    @pytest.mark.parametrize("name", ["nested_boxes", "x_port_box"])
    def test_boxed_symbolic_verdict_is_an_induction(self, name: str, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        source = self._boxed_symbolic(name, tmp_path, monkeypatch)
        assert "induction " in source and "symCheck" in source

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
        reason="the ArchytasZX Lean project has no Mathlib",
    )
    @pytest.mark.parametrize("name", ["nested_boxes", "x_port_box"])
    def test_boxed_symbolic_verdict_lean_checks(self, name: str, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
        source = self._boxed_symbolic(name, tmp_path, monkeypatch)
        passed, message = check(source, str(tmp_path / "check.lean"))
        assert passed, message

    def test_a_conditional_proof_says_so(self, tmp_path: Path) -> None:
        left, right = BUILT["mixed_legs"]()
        s = Session()
        s.names.update(l=left, r=right)
        ok(s, "load @l")
        ok(s, "decide against @r")
        lines = texts(ok(s, f"certificate lean {tmp_path / 'mixed.lean'}"))
        assert "lean: conditional on (d ^ p) = (d ^ q)" in lines

    def test_induct_rederives_a_theorem(self) -> None:
        s = session(BOXED, "name b", "induct d=2 against @b")
        lines = texts(ok(s, "certificate kernel"))
        assert any("re-derived a re-checkable proof by search" in line for line in lines)
        theorems = [line for line in lines if line.startswith("kernel: theorem ")]
        assert len(theorems) == 1 and theorems[0].endswith(" for all n")

    @pytest.mark.parametrize("verb", ["decide against", "prove"])
    def test_unequal_verdict_is_a_refutation(self, verb: str) -> None:
        s = session("a = Z(0 -> 1, d)", "name l", "a = X(0 -> 1, d)", "name r", "load @l")
        lines = texts(failed(s, f"{verb} @r"))
        assert "disproof: re-checkable; certificate lean" in lines
        assert isinstance(s.last_proof, kernel.Refutation)
        kernel_lines = texts(ok(s, "certificate kernel"))
        assert kernel_lines[0] == "emitted certificate: refutation at d = 2"
        assert kernel_lines[1].startswith("kernel: refutation of ")
        assert texts(ok(s, "certificate verify"))[-1].startswith("verify: verified: the diagrams")
        lean_lines = texts(ok(s, "certificate lean"))
        assert "lean: tensor-network semantics, refuted exactly at d = 2" in lean_lines

    def test_boundary_sizes_refute(self) -> None:
        s = session("a = Z(0 -> 1, d)", "name l", "a = Z(0 -> 2, d)", "name r", "load @l")
        assert "disproof: re-checkable; certificate lean" in texts(failed(s, "decide against @r"))
        lines = texts(ok(s, "certificate lean"))
        assert "lean: tensor-network semantics, the boundaries differ in size" in lines

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
        reason="the ArchytasZX Lean project has no Mathlib",
    )
    def test_unequal_verdict_lean_checks(self, tmp_path: Path) -> None:
        s = session("a = Z(0 -> 1, d) phase {1: theta}", "name l", "a = Z(0 -> 1, d)",
                    "name r", "load @l")
        failed(s, "decide against @r")
        target = tmp_path / "refute.lean"
        ok(s, f"certificate lean {target}")
        passed, message = check(target.read_text(encoding="utf-8"), str(target))
        assert passed, message

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
        reason="the ArchytasZX Lean project has no Mathlib",
    )
    @pytest.mark.parametrize("name", ["state_copy", "bialgebra_once", "bialgebra_4"])
    def test_decide_equal_lean_checks(self, name: str, tmp_path: Path) -> None:
        left, right = PROVED[name]()
        s = Session()
        s.names.update(l=left, r=right)
        ok(s, "load @l")
        ok(s, "decide against @r")
        target = tmp_path / "decide.lean"
        lines = texts(ok(s, f"certificate lean {target}"))
        assert "lean: tensor-network semantics, every step proved" in lines
        passed, message = check(target.read_text(encoding="utf-8"), str(target))
        assert passed, message


class TestCertificate:
    """certificate emit, emit KIND, verify, show."""

    def test_emit_session_derivation(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        outcome = ok(s, "certificate")
        assert isinstance(outcome.detail, Certificate) and s.certificate is outcome.detail
        assert texts(outcome)[0] == "certificate: numeric_oracle, step_sequence, 1 step"
        assert outcome.detail.derivation.label == "session derivation"
        assert same(outcome.detail.final, s.current)  # type: ignore[arg-type]

    def test_emit_proof_and_derivation(self) -> None:
        s = session(WORKED, "apply spider_fusion", "prove")
        assert isinstance(ok(s, "certificate emit").detail, ProofCertificate)
        assert isinstance(ok(s, "certificate emit proof").detail, ProofCertificate)
        assert isinstance(ok(s, "certificate emit derivation").detail, Certificate)

    def test_verify_numeric(self) -> None:
        s = session(WORKED, "apply spider_fusion", "certificate emit")
        outcome = ok(s, "certificate verify")
        assert isinstance(outcome.detail, VerificationReport) and outcome.detail.verified
        assert texts(outcome)[-1].startswith("verify: verified at 16 points: ")
        lines = texts(ok(s, "certificate verify d=2 n=1"))
        assert lines[-1].startswith("verify: verified at 1 point: ")

    def test_verify_emits_first(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        lines = texts(ok(s, "certificate verify"))
        assert lines[0] == "emitted certificate: numeric_oracle, step_sequence, 1 step"
        assert s.certificate is not None

    def test_verify_proof(self) -> None:
        s = session(WORKED, "apply spider_fusion", "prove", "certificate emit")
        outcome = ok(s, "certificate verify")
        assert isinstance(outcome.detail, ProofCheck) and outcome.detail.verified

    def test_induction_certificate(self) -> None:
        s = session(BOXED, "induct d=2")
        outcome = ok(s, "certificate emit induction")
        assert isinstance(outcome.detail, Certificate)
        assert outcome.detail.check_method.value == "induction"
        assert s.certificate_witness == {"d": 2}
        assert isinstance(ok(s, "certificate verify").detail, VerificationReport)
        ok(s, "certificate verify d=2")
        message = texts(errored(s, "certificate verify d=3"))[0]
        assert message.startswith("error: an induction certificate verifies only at its witness")

    def test_show(self) -> None:
        s = session(WORKED, "apply spider_fusion", "certificate emit")
        outcome = ok(s, "certificate show")
        assert outcome.detail is s.certificate

    @pytest.mark.parametrize(
        "line",
        [
            "certificate show",
            "certificate emit proof",
            "certificate emit induction",
            "certificate emit everything",
            "certificate frob",
            "certificate show now",
        ],
    )
    def test_errors(self, line: str) -> None:
        errored(session(WORKED, "apply spider_fusion"), line)


# -- translate -------------------------------------------------------------------------------


class TestTranslate:
    """translate, sheet operations, translate back."""

    def test_round_trip(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        outcome = ok(s, "translate")
        assert isinstance(outcome.detail, RoundTripReport) and outcome.detail.ok
        assert s.scalable is outcome.detail.scalable and s.sheet_steps == []
        assert texts(outcome)[0].startswith("round trip ok: ")

    def test_back_without_steps_keeps_the_current(self) -> None:
        s = session(WORKED, "apply spider_fusion", "translate")
        current, results = s.current, list(s.results)
        outcome = ok(s, "translate back")
        assert texts(outcome)[0] == "translated back; no sheet step, the current diagram stands"
        assert s.current is current and s.results == results and s.scalable is None

    def test_fuse_then_back_restarts(self) -> None:
        s = session(DSL_GHZ)
        original = s.current
        assert original is not None
        outcome = ok(s, "translate fuse n0.out0 -- n1.in0")
        assert isinstance(outcome.detail, SheetCheck) and outcome.detail.ok
        assert texts(outcome)[0] == "translated the current diagram to scalable notation"
        assert len(s.sheet_steps) == 1
        back = ok(s, "translate back")
        assert texts(back)[0] == "translated back; derivation restarted at the translated diagram"
        assert s.current is not None and s.start is s.current and s.results == []
        assert len(s.current.nodes) == 1
        for sample in points(original, s.current):
            assert compare(original, s.current, sample).matched
        assert_replays(s)

    def test_split_join_back_is_identical(self) -> None:
        s = session(COPIES4)
        current = s.current
        assert texts(ok(s, "translate split s0 2"))[-1].startswith("sheet step ok: ")
        assert texts(ok(s, "translate join s0 s1"))[-1].startswith("sheet step ok: ")
        back = ok(s, "translate back")
        assert texts(back)[0] == "translated back; identical to the current diagram"
        assert s.current is current

    def test_symbolic_split(self) -> None:
        s = session("count m, k; a = Z(0 -> 1, d); box p x m + k nodes a")
        ok(s, "translate split s0 m")
        ok(s, "translate join 0 1")

    def test_enclose_dissolve(self) -> None:
        s = session(TWO_PLUS)
        ok(s, "translate enclose n0, n1")
        ok(s, "translate dissolve s0")
        assert texts(ok(s, "translate back"))[0].endswith("identical to the current diagram")

    def test_kill(self) -> None:
        s = session(KILLABLE)
        original = s.current
        ok(s, "translate kill s0")
        ok(s, "translate back")
        assert s.current is not None and original is not None and len(s.current.nodes) == 1
        for sample in points(original, s.current):
            assert compare(original, s.current, sample).matched

    @pytest.mark.parametrize(
        ("source", "line"),
        [
            (TWO_PLUS, "translate enclose n0, n1 x 2"),
            (TWO_PLUS, "translate dissolve s0"),
            (COPIES4, "translate split s0"),
            (COPIES4, "translate join s0 s1"),
            (COPIES4, "translate fuse n0.out0"),
            (COPIES4, "translate kill s0 s1"),
            (COPIES4, "translate frob"),
            (COPIES4, "translate back"),
            (COPIES4, "translate split x 2"),
        ],
    )
    def test_errors_leave_no_scalable(self, source: str, line: str) -> None:
        s = session(source)
        errored(s, line)
        assert s.scalable is None and s.sheet_steps == []

    def test_undo_sheet_step(self) -> None:
        s = session(COPIES4, "translate split s0 2")
        before = s.scalable
        ok(s, "translate join s0 s1")
        ok(s, "undo")
        assert s.scalable is before and len(s.sheet_steps) == 1


# -- names, undo, reset, detail --------------------------------------------------------------


class TestNames:
    """name and names."""

    def test_name_and_list(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        outcome = ok(s, "name fused")
        assert texts(outcome) == ["named @fused"] and not outcome.changed
        assert s.names["fused"] is not s.current
        assert texts(ok(s, "names")) == ["@fused: GHZ_{n+2} = sum_{k=0}^{d-1} |k>^{n} |k, k>"]

    def test_no_names(self) -> None:
        assert texts(ok(Session(), "names")) == ["no names"]

    @pytest.mark.parametrize("line", ["name 9x", "name start", "name current", "name", "names x"])
    def test_errors(self, line: str) -> None:
        errored(session(WORKED), line)

    def test_name_requires_a_diagram(self) -> None:
        errored(Session(), "name x")


class TestUndoReset:
    """undo and reset."""

    def test_undo_apply(self) -> None:
        s = session(WORKED)
        current = s.current
        ok(s, "apply spider_fusion")
        outcome = ok(s, "undo")
        assert texts(outcome)[0] == "undid apply spider_fusion"
        assert s.current is current and s.results == []

    def test_undo_load(self) -> None:
        s = session(WORKED)
        ok(s, "undo")
        assert s.current is None and s.start is None

    def test_undo_label_is_cut(self) -> None:
        s = session(DSL_GHZ)
        label = texts(ok(s, "undo"))[0]
        assert label.startswith("undid a = Z(") and label.endswith("...")
        assert len(label) == len("undid ") + 60

    def test_nothing_to_undo(self) -> None:
        errored(Session(), "undo")

    def test_reset(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        start, results = s.start, list(s.results)
        outcome = ok(s, "reset")
        assert outcome.changed
        assert s.start is s.current and s.results == []
        assert_replays(s)
        ok(s, "undo")
        assert s.start is start and s.results == results

    @pytest.mark.parametrize(
        "line",
        ["show", "match", "check", "decide", "prove", "name x", "names", "rules", "help", "detail"],
    )
    def test_reading_commands_push_nothing(self, line: str) -> None:
        s = session(WORKED, "apply spider_fusion")
        depth = len(s.undo_stack)
        outcome = ok(s, line)
        assert not outcome.changed and len(s.undo_stack) == depth


class TestDetail:
    """detail and last_detail."""

    def test_every_success_sets_last_detail(self) -> None:
        s = Session()
        for line in (WORKED, "apply spider_fusion", "check", "decide", "rules", "names"):
            outcome = ok(s, line)
            assert s.last_detail is outcome.detail

    def test_detail_renders_the_last_payload(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        payload = s.last_detail
        outcome = ok(s, "detail")
        assert texts(outcome) == detail_text(payload).splitlines()
        assert outcome.detail is payload

    def test_nothing_to_detail(self) -> None:
        errored(Session(), "detail")

    def test_detail_arguments(self) -> None:
        errored(session(WORKED), "detail now")

    def test_error_detail_side_by_side(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        errored(s, "apply nosuch")
        text = format_outcome(ok(s, "detail"), DIRAC)
        lines = text.splitlines()
        assert lines[0] == "error in apply: apply nosuch"
        assert lines[1].startswith("  CommandDomainError: no rule 'nosuch'")
        assert lines[2].startswith("  ParseDomainError: no rule 'nosuch'")
        assert lines[3] == "  current:"
        graph = next(line for line in lines if line.lstrip().startswith("graph:"))
        assert "GHZ_{n+2} = sum_{k=0}^{d-1} |k>^{n} |k, k>" in graph


# -- invariants ------------------------------------------------------------------------------

REPLAY_SCRIPTS: dict[str, tuple[str, ...]] = {
    "worked": (
        WORKED,
        "apply spider_fusion",
        "normalize",
        "check",
        "undo",
        "undo",
        "saturate",
        "reset",
        "normalize",
    ),
    "dsl": (
        DSL_GHZ,
        "normalize",
        "translate",
        "translate back",
        "undo",
        "translate fuse n0.out0 -- n1.in0",
        "translate back",
        "normalize",
    ),
    "boxed": (BOXED, "apply spider_fusion", "saturate", "normalize", "undo", "reset"),
    "with_errors": (
        WORKED,
        "apply nosuch",
        "apply spider_fusion",
        "apply spider_fusion",
        "translate kill s0",
        f"decide against {UNEQUAL}",
        "normalize",
    ),
}


class TestInvariants:
    """Replay, oracle, purity and exact undo along scripted sessions."""

    @pytest.mark.parametrize("name", sorted(REPLAY_SCRIPTS))
    def test_derivation_always_replays_and_oracle_agrees(self, name: str) -> None:
        s = Session()
        for line in REPLAY_SCRIPTS[name]:
            run(s, line)
            assert_replays(s)
            assert_oracle(s)

    @pytest.mark.parametrize("name", sorted(REPLAY_SCRIPTS))
    def test_names_and_snapshots_are_never_mutated(self, name: str) -> None:
        s = Session()
        seen: dict[int, tuple[Diagram, str]] = {}

        def record() -> None:
            held: list[Diagram | None] = [*s.names.values()]
            for snap in s.undo_stack:
                held += [snap.current, snap.start]
                held += [r.diagram for r in snap.results]
            held += [r.diagram for r in s.results]
            for diagram in held:
                if diagram is not None and id(diagram) not in seen:
                    seen[id(diagram)] = (diagram, render_graph(diagram))

        for index, line in enumerate(REPLAY_SCRIPTS[name]):
            run(s, line)
            if index % 2 == 0 and s.current is not None:
                run(s, f"name k{index}")
            record()
        for diagram, listing in seen.values():
            assert render_graph(diagram) == listing

    @pytest.mark.parametrize("name", sorted(REPLAY_SCRIPTS))
    def test_undo_restores_exactly(self, name: str) -> None:
        s = Session()
        for line in REPLAY_SCRIPTS[name]:
            before = (
                s.current,
                s.start,
                list(s.results),
                s.scalable,
                list(s.sheet_steps),
                len(s.undo_stack),
            )
            listing = render_graph(s.current) if s.current is not None else None
            outcome = run(s, line)
            if not outcome.changed:
                continue
            ok(s, "undo")
            after = (
                s.current,
                s.start,
                list(s.results),
                s.scalable,
                list(s.sheet_steps),
                len(s.undo_stack),
            )
            assert after == before, line
            if listing is not None:
                assert render_graph(s.current) == listing  # type: ignore[arg-type]
            run(s, line)

    def test_failed_state_change_restores_the_session(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        before = (s.current, s.start, list(s.results), s.scalable, len(s.undo_stack))
        errored(s, "translate split s0 1")
        assert (s.current, s.start, list(s.results), s.scalable, len(s.undo_stack)) == before


class TestModeIndependence:
    """Outcomes never depend on the mode; only format_outcome reads it."""

    def test_outcomes_agree_across_sessions(
        self, mode_runs: tuple[list[Outcome], list[Outcome]]
    ) -> None:
        first, second = mode_runs
        assert [signature(o) for o in first] == [signature(o) for o in second]

    def test_dirac_output_has_no_listing(
        self, mode_runs: tuple[list[Outcome], list[Outcome]]
    ) -> None:
        for line, outcome in zip(MODE_SCRIPT, mode_runs[0], strict=True):
            assert listing_lines(format_outcome(outcome, DIRAC)) == [], line
            assert listing_lines(format_outcome(outcome, DIRAC, unicode=True)) == [], line

    def test_both_output_has_listing_and_dirac(
        self, mode_runs: tuple[list[Outcome], list[Outcome]]
    ) -> None:
        with_states = 0
        for line, outcome in zip(MODE_SCRIPT, mode_runs[0], strict=True):
            zx = format_outcome(outcome, BOTH, listing=True)
            for diagram in diagram_states(outcome):
                with_states += 1
                assert listing_lines(zx), line
                for dirac_line in render_dirac(diagram).splitlines():
                    assert dirac_line in zx, (line, dirac_line)
                assert render_graph(diagram).splitlines()[0] in zx
        assert with_states >= 6

    def test_texts_identical_in_both_modes(
        self, mode_runs: tuple[list[Outcome], list[Outcome]]
    ) -> None:
        for outcome in mode_runs[0]:
            dirac, zx = format_outcome(outcome, DIRAC), format_outcome(outcome, ZX)
            for line in texts(outcome):
                assert line in dirac and line in zx

    def test_detail_shows_both_notations(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        text = format_outcome(ok(s, "detail"), DIRAC)
        assert listing_lines(text)
        assert "GHZ_{n+2} = sum_{k=0}^{d-1} |k>^{n} |k, k>" in text


# -- error contract --------------------------------------------------------------------------


class TestErrors:
    """run never raises an Exception; execute raises CommandError; interrupts propagate."""

    @pytest.mark.parametrize(
        "line",
        [
            "sum_{k=0",
            ")))",
            "xyz",
            "load",
            "load sum_{k=0",
            "",
            "   ",
            "show @nope",
            "load @nope",
            "load @start",
            "apply spider_fusion",
            "rules nosuch",
            "match nosuch",
            "check d=2",
        ],
    )
    def test_run_on_an_empty_session(self, line: str) -> None:
        s = Session()
        errored(s, line)
        assert s.current is None and s.undo_stack == []

    @pytest.mark.parametrize("line", [42, None, b"show", ["show"]])
    def test_non_str_line(self, line: object) -> None:
        outcome = errored(session(WORKED), line)
        assert outcome.command == ""
        assert outcome.detail.error_types == ("CommandGrammarError",)  # type: ignore[union-attr]
        assert outcome.detail.line == repr(line)  # type: ignore[union-attr]

    def test_non_session(self) -> None:
        outcome = run(None, "show")  # type: ignore[arg-type]
        assert not outcome.ok and isinstance(outcome.detail, ErrorDetail)
        with pytest.raises(CommandGrammarError):
            execute(None, "show")  # type: ignore[arg-type]

    def test_error_detail_fields(self) -> None:
        s = session(WORKED)
        outcome = errored(s, "apply nosuch")
        detail = outcome.detail
        assert isinstance(detail, ErrorDetail)
        assert detail.line == "apply nosuch" and detail.command == "apply"
        assert detail.error_types == ("CommandDomainError", "ParseDomainError")
        assert len(detail.messages) == 2 and "nosuch" in detail.messages[1]
        assert detail.state is not None and detail.state is not s.current
        assert same(detail.state, s.current)  # type: ignore[arg-type]
        assert outcome.command == "apply"

    def test_error_detail_without_a_diagram(self) -> None:
        detail = errored(Session(), "apply spider_fusion").detail
        assert detail.state is None  # type: ignore[union-attr]
        assert detail.error_types == ("CommandDomainError",)  # type: ignore[union-attr]
        assert texts(errored(Session(), "match"))[0] == "error: no diagram loaded; use load"

    @pytest.mark.parametrize(
        ("line", "kind", "cause"),
        [
            ("sum_{k=0", CommandGrammarError, ParseError),
            ("load )))", CommandGrammarError, ParseError),
            ("apply nosuch", CommandDomainError, ParseError),
            ("apply spider_fusion at 5", CommandDomainError, ParseError),
            ("translate kill s0", CommandDomainError, Exception),
        ],
    )
    def test_execute_chains_the_engine_error(
        self, line: str, kind: type[CommandError], cause: type[Exception]
    ) -> None:
        with pytest.raises(kind) as caught:
            execute(session(WORKED), line)
        assert isinstance(caught.value.__cause__, cause)
        assert not isinstance(caught.value.__cause__, CommandError)

    @pytest.mark.parametrize(
        ("line", "kind"),
        [
            ("check d", CommandGrammarError),
            ("rules nosuch", CommandDomainError),
            ("undo now", CommandGrammarError),
            ("", CommandGrammarError),
        ],
    )
    def test_execute_raises_command_errors(self, line: str, kind: type[CommandError]) -> None:
        with pytest.raises(kind):
            execute(session(WORKED), line)

    def test_execute_rejects_non_str(self) -> None:
        with pytest.raises(CommandGrammarError):
            execute(Session(), 3)  # type: ignore[arg-type]

    def test_keyboard_interrupt_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = session(WORKED)
        before = (s.current, s.start, list(s.results), len(s.undo_stack), s.last_detail)

        def interrupted(*args: object, **kwargs: object) -> object:
            raise KeyboardInterrupt

        monkeypatch.setattr(C, "normal_form", interrupted)
        with pytest.raises(KeyboardInterrupt):
            run(s, "normalize")
        assert (s.current, s.start, list(s.results), len(s.undo_stack), s.last_detail) == before

    def test_foreign_exception_becomes_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = session(WORKED)

        def broken(*args: object, **kwargs: object) -> object:
            raise ZeroDivisionError("boom")

        monkeypatch.setattr(C, "normal_form", broken)
        outcome = errored(s, "normalize")
        assert texts(outcome)[0] == "error: boom"
        assert outcome.detail.error_types == (  # type: ignore[union-attr]
            "CommandDomainError",
            "ZeroDivisionError",
        )
        with pytest.raises(CommandDomainError) as caught:
            execute(s, "normalize")
        assert isinstance(caught.value.__cause__, ZeroDivisionError)

    def test_comment_is_stripped_from_commands(self) -> None:
        s = session(WORKED)
        assert ok(s, "apply spider_fusion # fuse").changed

    def test_equals_after_a_command_word_is_load(self) -> None:
        s = Session()
        outcome = ok(s, "match = Z(0 -> 1, d)")
        assert outcome.command == "load"


class TestBlocks:
    """Text, Outcome, ErrorDetail and format_outcome type checks."""

    def test_text_rejects_non_str(self) -> None:
        with pytest.raises(CommandGrammarError):
            Text(3)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "fields",
        [
            (1, True, (), None, False),
            ("x", 1, (), None, False),
            ("x", True, [], None, False),
            ("x", True, ("text",), None, False),
            ("x", True, (), None, "no"),
        ],
    )
    def test_outcome_rejects_bad_fields(self, fields: tuple[object, ...]) -> None:
        with pytest.raises(CommandGrammarError):
            Outcome(*fields)  # type: ignore[arg-type]

    def test_error_detail_rejects_bad_fields(self) -> None:
        with pytest.raises(CommandGrammarError):
            ErrorDetail("x", "y", ["T"], (), None)  # type: ignore[arg-type]
        with pytest.raises(CommandGrammarError):
            ErrorDetail("x", "y", (), (), "state")  # type: ignore[arg-type]

    def test_format_outcome_rejects_bad_arguments(self) -> None:
        outcome = Outcome("x", True, (Text("a"),), None, False)
        with pytest.raises(CommandGrammarError):
            format_outcome("x")  # type: ignore[arg-type]
        with pytest.raises(CommandGrammarError):
            format_outcome(outcome, "zx")  # type: ignore[arg-type]

    def test_format_outcome_unrenderable_state(self) -> None:
        outcome = Outcome("x", True, (Text("a"), State(object())), None, False)
        lines = format_outcome(outcome).splitlines()
        assert lines[0] == "a" and lines[1].startswith("render: unavailable (")

    def test_detail_text_rejects_non_bool_unicode(self) -> None:
        with pytest.raises(CommandGrammarError):
            detail_text(None, unicode=1)  # type: ignore[arg-type]


# -- detail payloads -------------------------------------------------------------------------


class _Unprintable:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


DIAGRAM_PAYLOADS = (
    "Diagram",
    "RewriteResult",
    "_Pair",
    "NormalForm",
    "Derivation",
    "Certificate",
    "ProofCertificate",
    "InductionProof",
    "VerificationReport",
    "SheetCheck",
    "RoundTripReport",
    "tuple",
    "ScalableDiagram",
    "ErrorDetail",
)


class TestDetailText:
    """detail_text over every payload the commands produce."""

    def test_covers_every_payload_type(self, payloads: Sequence[tuple[str, Outcome]]) -> None:
        kinds = {type(o.detail).__name__ for _line, o in payloads}
        assert set(DIAGRAM_PAYLOADS) | {"Text", "ProofCheck"} <= kinds

    @pytest.mark.parametrize("unicode", [False, True])
    def test_never_raises_and_shows_both_notations(
        self, payloads: Sequence[tuple[str, Outcome]], unicode: bool
    ) -> None:
        for line, outcome in payloads:
            text = detail_text(outcome.detail, unicode=unicode)
            assert isinstance(text, str) and text, line
            assert not text.startswith("detail: unavailable"), (line, text)
            assert "unavailable (" not in text, (line, text)
            kind = type(outcome.detail).__name__
            if kind in DIAGRAM_PAYLOADS and not (
                isinstance(outcome.detail, ErrorDetail) and outcome.detail.state is None
            ):
                assert listing_lines(text), (line, kind)
                assert ("sum_" in text) or ("Σ" in text) or ("GHZ" in text), (line, kind)

    def test_states_render_side_by_side(self, payloads: Sequence[tuple[str, Outcome]]) -> None:
        for line, outcome in payloads:
            if isinstance(outcome.detail, Diagram):
                text = detail_text(outcome.detail)
                head = render_dirac(outcome.detail).splitlines()[0]
                graph = next(t for t in text.splitlines() if t.lstrip().startswith("graph:"))
                assert graph.rstrip().endswith(head), line

    def test_decision_proof_and_induction_sections(
        self, payloads: Sequence[tuple[str, Outcome]]
    ) -> None:
        joined = "\n".join(detail_text(o.detail) for _line, o in payloads)
        for header in (
            "decision: equal by",
            "decision: unequal by",
            "decision: unknown by",
            "proof outcome: proved",
            "proof outcome: not_found",
            "induction: proved",
            "saturation: saturated",
            "extraction: e-class",
            "verification: verified",
            "proof check: verified",
            "sheet check: ok",
            "round trip: ok",
            "error in apply: apply nosuch",
        ):
            assert header in joined, header

    def test_derivation_detail_replays_every_state(self) -> None:
        s = session(WORKED, "normalize")
        text = detail_text(ok(s, "show derivation").detail)
        for index in range(len(s.results) + 1):
            assert f"state {index}:" in text
        assert "recorded final" not in text

    @pytest.mark.parametrize(
        "payload", [None, 7, "text", object(), (1, 2), (), Text("a\nb"), State(3)]
    )
    def test_odd_payloads(self, payload: object) -> None:
        assert isinstance(detail_text(payload), str)

    def test_unprintable_payload(self) -> None:
        text = detail_text(_Unprintable())
        assert text == "detail: unavailable (RuntimeError: no repr)"

    def test_unrenderable_state(self) -> None:
        text = detail_text(State(_Unprintable()))
        assert text.startswith("detail: unavailable")

    def test_detail_command_matches_detail_text(self) -> None:
        s = session(WORKED, "apply spider_fusion")
        for line in ("check", "decide", "prove", "certificate emit", "translate"):
            outcome = ok(s, line)
            assert texts(ok(s, "detail")) == detail_text(outcome.detail).splitlines()


class TestProofText:
    def test_derivation_reads_as_a_proof(self) -> None:
        session = Session()
        run(
            session,
            "load a = Z(1 -> 1, d); b = Z(1 -> 1, d); wire a.out0 -- b.in0; "
            "inputs a.in0; outputs b.out0",
        )
        run(session, "apply spider_fusion")
        out = format_outcome(run(session, "certificate emit derivation"), unicode=True)
        lines = out.splitlines()
        assert lines[0] == "certificate: numeric_oracle, step_sequence, 1 step"
        assert lines[1] == "Theorem." and "Proof." in lines
        assert lines[-2].rstrip().endswith("(by spider_fusion at n0, n1)")
        assert lines[-1] == "∎  (checked by numeric oracle)"
        shown = format_outcome(run(session, "show certificate"), unicode=True)
        assert shown == "\n".join(lines[1:])

    def test_induction_proof(self) -> None:
        from archytaszx.repl.commands import proof_text
        from archytaszx.semantics.prove import prove
        from tests.test_phase15_oracle import boxed_state_copy

        outcome = prove(*boxed_state_copy())
        assert outcome.induction is not None
        text = proof_text(outcome.induction)
        assert "Proof by induction on n." in text
        assert "Base case (n = 0)." in text and "Inductive step." in text
        assert text.endswith("QED  (checked by replay)")
