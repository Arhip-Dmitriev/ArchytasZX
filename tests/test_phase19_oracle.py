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

"""Phase 19 test-and-debug items and done-when: the scripted worked example end to end, clean
errors, mode independence of the session, Dirac output in Dirac mode, detail payloads in both
notations, failure reports, and every command accepted in both modes."""

from __future__ import annotations

import functools
import io
import re
from dataclasses import dataclass

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import PhaseVector
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.repl import commands as C
from archytaszx.repl.printer import NotationMode, recognize, render_graph
from archytaszx.repl.shell import Shell
from archytaszx.semantics.check import compare
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.induction import InductionResult
from archytaszx.semantics.prove import ProofCertificate, prove

from .helpers import build_ghz_with_copy

D = Dim.symbol("d")
EXAMPLE = "sum_{k=0}^{3-1} |k>^{30}; copy"
GHZ_DSL = "a = Z(0 -> 2, d); b = Z(1 -> 2, d); wire a.out0 -- b.in0; outputs a.out1, b.out0, b.out1"
UNEQUAL = "sum_{k=0}^{d-1} |k>^{n} |k, k, k>"
MODES = (NotationMode.DIRAC, NotationMode.BOTH)
SMALL = [{"d": d, "n": n} for d in (2, 3) for n in (0, 1, 2, 3)]

WORKED_SCRIPT = f"""\
# worked example: build, fuse, check, induct, certify
{EXAMPLE}
show
match spider_fusion
apply spider_fusion
check
check d=2 n=1
induct
certificate
certificate verify
detail
prove
certificate emit proof
certificate verify
show derivation
detail
show
"""

DSL_SCRIPT = f"""\
load {GHZ_DSL}
apply spider_fusion
check
check d=2
prove
certificate emit proof
certificate verify d=2
certificate emit derivation
certificate verify
"""


@dataclass(frozen=True)
class Record:
    line: str
    output: str
    outcome: C.Outcome | None


def run_session(mode: NotationMode, script: str) -> tuple[Shell, list[Record]]:
    """A fresh shell in ``mode`` and one record per executed logical line."""
    shell = Shell(commands=True, listing=True, mode=mode)
    records: list[Record] = []
    for line, output in shell.iter_script(script):
        records.append(Record(line, output, shell.last_outcome))
    return shell, records


def lines_of(text: str) -> list[str]:
    return text.splitlines()


def listing_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.lstrip().startswith(("graph:", "scalable:"))]


def is_subsequence(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(any(item == other for other in it) for item in small)


def hand_ghz() -> Diagram:
    """One Z spider with outputs ``d!n, d, d``, the bang box over output 0."""
    diagram = Diagram()
    nid = diagram.add_node(Z_SPIDER, [], [D, D, D], phase=PhaseVector(D))
    ports = [PortRef(nid, Direction.OUTPUT, i) for i in range(3)]
    diagram.set_boundary_outputs(ports)
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({ports[0]}))
    return diagram


def hand_ghz3() -> Diagram:
    diagram = Diagram()
    nid = diagram.add_node(Z_SPIDER, [], [D, D, D], phase=PhaseVector(D))
    diagram.set_boundary_outputs([PortRef(nid, Direction.OUTPUT, i) for i in range(3)])
    return diagram


@pytest.fixture(scope="module")
def worked() -> dict[NotationMode, tuple[Shell, list[Record]]]:
    return {mode: run_session(mode, WORKED_SCRIPT) for mode in MODES}


@pytest.fixture(scope="module")
def dsl_worked() -> dict[NotationMode, tuple[Shell, list[Record]]]:
    return {mode: run_session(mode, DSL_SCRIPT) for mode in MODES}


def record(records: list[Record], line: str, occurrence: int = 0) -> Record:
    found = [r for r in records if r.line == line]
    return found[occurrence]


# -- the worked example --------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
class TestWorkedExample:
    def test_every_line_runs_and_succeeds(self, worked, mode) -> None:
        shell, records = worked[mode]
        assert records[0].line == EXAMPLE
        assert len(records) == 16
        for r in records:
            assert r.outcome is not None and r.outcome.ok, (r.line, r.output)
            assert "error" not in r.output.split(":", 1)[0]
            assert "Traceback" not in r.output
        assert shell.errors == 0

    def test_load_reads_the_example(self, worked, mode) -> None:
        _, records = worked[mode]
        start = record(records, EXAMPLE).outcome.detail
        assert isinstance(start, Diagram)
        assert len(start.nodes) == 2 and len(start.wires) == 1
        assert dict(start.parameters) == {"d": 3, "n": 29}

    def test_match_lists_the_fusion_site(self, worked, mode) -> None:
        _, records = worked[mode]
        assert record(records, "match spider_fusion").output == "#0 spider_fusion at n0, n1"

    def test_fusion_gives_ghz(self, worked, mode) -> None:
        _, records = worked[mode]
        out = record(records, "apply spider_fusion").output
        assert "step: spider_fusion (consumed n0, n1, new n2, scalar 1)" in out
        assert "GHZ_{n+2} = sum_{k=0}^{d-1} |k>^{n} |k, k>" in out

    def test_checks_find_no_mismatch(self, worked, mode) -> None:
        _, records = worked[mode]
        assert record(records, "check").output.startswith("check: no mismatch in ")
        assert record(records, "check").output.endswith("over d, n")
        assert record(records, "check d=2 n=1").output == "check: no mismatch in 1 sample over d, n"

    def test_induct_proves_for_all_n(self, worked, mode) -> None:
        _, records = worked[mode]
        r = record(records, "induct")
        assert r.output.startswith("induct: proved")
        assert " over n from 0 " in r.output
        result = r.outcome.detail.payload
        assert isinstance(result, InductionResult) and result.proved
        assert result.index == "n"

    def test_certificate_emits_and_verifies(self, worked, mode) -> None:
        _, records = worked[mode]
        assert record(records, "certificate").output.startswith(
            "certificate: numeric_oracle, step_sequence, 1 step"
        )
        verify = record(records, "certificate verify").output
        assert "verify: verified at " in verify

    def test_prove_and_proof_certificate_verify(self, worked, mode) -> None:
        shell, records = worked[mode]
        assert record(records, "prove").output.startswith("prove: proved")
        emitted = record(records, "certificate emit proof")
        assert emitted.output.startswith("certificate: proof, 1 step")
        assert isinstance(emitted.outcome.detail, ProofCertificate)
        assert "verify: verified" in record(records, "certificate verify", 1).output
        assert isinstance(shell.session.certificate, ProofCertificate)

    def test_final_state_is_ghz(self, worked, mode) -> None:
        shell, records = worked[mode]
        current = shell.session.current
        assert len(current.nodes) == 1 and not current.wires
        (node,) = current.nodes.values()
        assert node.generator_type is Z_SPIDER
        assert recognize(current) == "GHZ_{n+2}"
        assert [r.step.rule_name for r in shell.session.results] == ["spider_fusion"]
        assert "GHZ_{n+2} = sum_{k=0}^{d-1} |k>^{n} |k, k>" in record(records, "show", 1).output

    @pytest.mark.parametrize("assignment", SMALL, ids=str)
    def test_final_state_oracle_equals_start(self, worked, mode, assignment) -> None:
        shell, _ = worked[mode]
        result = compare(shell.session.start, shell.session.current, assignment)
        assert result.matched, result.reason

    @pytest.mark.parametrize("assignment", SMALL, ids=str)
    def test_final_state_oracle_equals_hand_built_ghz(self, worked, mode, assignment) -> None:
        shell, _ = worked[mode]
        result = compare(shell.session.current, hand_ghz(), assignment)
        assert result.matched, result.reason

    def test_hand_built_ghz_differs_from_a_wrong_target(self, worked, mode) -> None:
        shell, _ = worked[mode]
        unequal = C.Session()
        C.run(unequal, UNEQUAL)
        assert not compare(shell.session.current, unequal.current, {"d": 2, "n": 1}).matched


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
class TestDslWorkedExample:
    def test_every_line_succeeds(self, dsl_worked, mode) -> None:
        shell, records = dsl_worked[mode]
        for r in records:
            assert r.outcome is not None and r.outcome.ok, (r.line, r.output)
        assert shell.errors == 0

    def test_dsl_load_is_the_hand_built_graph(self, dsl_worked, mode) -> None:
        shell, _ = dsl_worked[mode]
        start = shell.session.start
        want = build_ghz_with_copy(D)[0]
        assert render_graph(start) == render_graph(want)

    def test_final_state(self, dsl_worked, mode) -> None:
        shell, records = dsl_worked[mode]
        current = shell.session.current
        assert recognize(current) == "GHZ_{3}"
        assert "verify: verified" in record(records, "certificate verify d=2").output
        assert "verify: verified at " in record(records, "certificate verify").output
        for d in (2, 3, 4):
            assert compare(shell.session.start, current, {"d": d}).matched
            assert compare(current, hand_ghz3(), {"d": d}).matched

    def test_induct_without_a_bang_box_is_a_clean_error(self, dsl_worked, mode) -> None:
        shell, _ = dsl_worked[mode]
        probe = Shell(commands=True, listing=True, mode=mode, session=C.Session())
        probe.run_line(f"load {GHZ_DSL}")
        out = probe.run_line("induct")
        assert out.startswith("error: no bang box")
        assert shell.errors == 0


def test_worked_example_runs_interactively() -> None:
    stdin = io.StringIO(WORKED_SCRIPT + "quit\nshow\n")
    stdout = io.StringIO()
    shell = Shell(commands=True)
    assert shell.loop(stdin, stdout) == 0
    text = stdout.getvalue()
    assert "induct: proved" in text
    assert "verify: verified" in text
    assert text.count("GHZ_{n+2}") >= 2
    assert shell.done and shell.errors == 0
    assert recognize(shell.session.current) == "GHZ_{n+2}"
    assert shell.history[-1] == "quit"


# -- malformed input -----------------------------------------------------------------------

MALFORMED = [
    "load |0> +* (",
    "sum_{k=0}^{d-1 |k>",
    "frobnicate (",
    "apply nosuchrule",
    "apply spider_fusion at 99",
    "apply spider_fusion at",
    "check d=x",
    "check d=1/0",
    "check q=2",
    "check d=2 d=3",
    "check against",
    "induct on q",
    "induct n=2",
    "certificate frob",
    "certificate emit nonsense",
    "translate frob",
    "translate split",
    "show @nosuch",
    "rules nosuchrule",
    "normalize now",
    "name 9x",
    "name start",
    "undo extra",
    "help nosuchcommand",
    "decide against @nosuch",
]


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("line", MALFORMED)
def test_malformed_input_gives_a_clean_error(mode: NotationMode, line: str) -> None:
    shell = Shell(commands=True, listing=True, mode=mode)
    shell.run_script([EXAMPLE, "apply spider_fusion"])
    session = shell.session
    before = (
        render_graph(session.current),
        render_graph(session.start),
        len(session.results),
        len(session.undo_stack),
        dict(session.names),
    )
    out = shell.run_line(line)
    assert out.splitlines()[0].startswith("error: "), out
    assert out.splitlines()[1:] == ["(detail shows the engine detail)"]
    assert "Traceback" not in out
    assert shell.errors == 1
    assert shell.last_outcome is not None and not shell.last_outcome.ok
    assert isinstance(session.last_detail, C.ErrorDetail)
    after = (
        render_graph(session.current),
        render_graph(session.start),
        len(session.results),
        len(session.undo_stack),
        dict(session.names),
    )
    assert after == before
    detail = shell.run_line("detail")
    assert detail.startswith("error in ")
    assert "graph:" in detail


def test_errors_before_any_load_are_clean() -> None:
    shell = Shell(commands=True)
    for line in ("show", "apply spider_fusion", "check", "induct", "certificate", "undo"):
        out = shell.run_line(line)
        assert out.startswith("error: "), (line, out)
    assert shell.errors == 6
    assert shell.session.current is None


# -- mode independence ---------------------------------------------------------------------


class TestModeIndependence:
    def test_same_lines_in_both_modes(self, worked) -> None:
        a = [r.line for r in worked[NotationMode.DIRAC][1]]
        b = [r.line for r in worked[NotationMode.BOTH][1]]
        assert a == b

    def test_final_session_state_identical(self, worked) -> None:
        sa, sb = worked[NotationMode.DIRAC][0].session, worked[NotationMode.BOTH][0].session
        assert render_graph(sa.current) == render_graph(sb.current)
        assert render_graph(sa.start) == render_graph(sb.start)
        assert [r.step for r in sa.results] == [r.step for r in sb.results]
        assert type(sa.certificate) is type(sb.certificate)
        assert C.detail_text(sa.certificate) == C.detail_text(sb.certificate)
        assert len(sa.undo_stack) == len(sb.undo_stack)

    def test_engine_outcomes_identical(self, worked) -> None:
        ra, rb = worked[NotationMode.DIRAC][1], worked[NotationMode.BOTH][1]
        for a, b in zip(ra, rb, strict=True):
            assert a.outcome is not None and b.outcome is not None
            assert (a.outcome.command, a.outcome.ok, a.outcome.changed) == (
                b.outcome.command,
                b.outcome.ok,
                b.outcome.changed,
            )
            for mode in MODES:
                assert C.format_outcome(a.outcome, mode) == C.format_outcome(b.outcome, mode)
            assert C.detail_text(a.outcome.detail) == C.detail_text(b.outcome.detail)

    def test_outputs_differ_only_in_rendering(self, worked) -> None:
        ra, rb = worked[NotationMode.DIRAC][1], worked[NotationMode.BOTH][1]
        for a, b in zip(ra, rb, strict=True):
            if a.line == "detail":
                assert a.output == b.output
                continue
            assert is_subsequence(lines_of(a.output), lines_of(b.output)), a.line
            assert not listing_lines(a.output), a.line
            if a.output != b.output:
                assert listing_lines(b.output), a.line

    def test_dirac_text_appears_in_both_modes(self, worked) -> None:
        for mode in MODES:
            out = record(worked[mode][1], "apply spider_fusion").output
            assert "GHZ_{n+2} = sum_{k=0}^{d-1} |k>^{n} |k, k>" in out
        assert listing_lines(record(worked[NotationMode.BOTH][1], "apply spider_fusion").output)

    def test_switching_mode_mid_session_changes_rendering_only(self) -> None:
        shell = Shell(commands=True)
        shell.run_line(EXAMPLE)
        dirac = shell.run_line("show")
        assert shell.run_line("zx") == "mode: zx"
        zx = shell.run_line("show")
        assert zx.startswith("scalar:") and not listing_lines(zx)
        assert not is_subsequence(lines_of(dirac), lines_of(zx))
        assert shell.run_line("raw") == "mode: raw"
        raw = shell.run_line("show")
        assert listing_lines(raw) and not is_subsequence(lines_of(dirac), lines_of(raw))
        assert shell.run_line("both") == "mode: both"
        both = shell.run_line("show")
        assert not listing_lines(dirac) and both.startswith("scalar:")
        assert is_subsequence(lines_of(dirac), lines_of(both))
        assert shell.run_line("dirac") == "mode: dirac"
        assert shell.run_line("show") == dirac


# -- ZX commands in Dirac mode -------------------------------------------------------------

ZX_COMMANDS = [
    f"load {GHZ_DSL}",
    f"{GHZ_DSL}",
    "show",
    "apply spider_fusion",
    "show start",
    "show derivation",
    "undo",
    "normalize",
    "saturate",
    "translate",
    "show scalable",
    "translate back",
    "certificate",
    "certificate show",
]


@pytest.fixture(scope="module")
def zx_commands() -> dict[NotationMode, tuple[Shell, list[Record]]]:
    return {mode: run_session(mode, "\n".join(ZX_COMMANDS)) for mode in MODES}


class TestZxCommandsInDiracMode:
    def test_all_succeed(self, zx_commands) -> None:
        for mode in MODES:
            for r in zx_commands[mode][1]:
                assert r.outcome is not None and r.outcome.ok, (mode, r.line, r.output)

    @pytest.mark.parametrize("index", range(len(ZX_COMMANDS)))
    def test_dirac_mode_emits_dirac_only(self, zx_commands, index) -> None:
        r = zx_commands[NotationMode.DIRAC][1][index]
        assert not listing_lines(r.output), r.output
        assert re.search(r"sum_|GHZ_|Z_\{", r.output), r.output

    @pytest.mark.parametrize("index", range(len(ZX_COMMANDS)))
    def test_zx_mode_adds_the_listing(self, zx_commands, index) -> None:
        r = zx_commands[NotationMode.BOTH][1][index]
        assert re.search(r"sum_|GHZ_|Z_\{", r.output), r.output
        if r.line.startswith("certificate"):
            assert (
                "Proof." in r.output
                and r.output == zx_commands[NotationMode.DIRAC][1][index].output
            )
            return
        assert listing_lines(r.output), r.output

    def test_listing_input_in_dirac_mode_answers_in_dirac(self) -> None:
        listing = render_graph(build_ghz_with_copy(D)[0])
        shell = Shell(commands=True)
        session = shell.session
        outcome = C.run(session, f"load {listing}")
        assert outcome.ok, C.format_outcome(outcome)
        text = C.format_outcome(outcome, shell.mode)
        assert not listing_lines(text) and "sum_" in text


# -- detail payloads -----------------------------------------------------------------------

DETAIL_LINES = {
    "apply": "apply spider_fusion",
    "error": "apply nosuchrule",
    "induction": "induct",
    "proof": "prove",
    "derivation": "show derivation",
    "certificate": "certificate emit proof",
    "verification": "certificate verify",
    "decision": "decide",
}


def side_by_side(text: str) -> bool:
    """Some line carries both a graph listing header and Dirac text."""
    return any(
        "graph:" in line and re.search(r"sum_|GHZ_|Z_\{", line.split("graph:", 1)[1])
        for line in text.splitlines()
    )


@pytest.fixture(scope="module")
def details() -> dict[NotationMode, dict[str, str]]:
    found: dict[NotationMode, dict[str, str]] = {}
    for mode in MODES:
        shell = Shell(commands=True, listing=True, mode=mode)
        shell.run_line(EXAMPLE)
        found[mode] = {}
        for label, line in DETAIL_LINES.items():
            if label == "verification":
                shell.run_line("certificate emit derivation")
            shell.run_line(line)
            found[mode][label] = shell.run_line("detail")
    return found


@pytest.mark.parametrize("label", sorted(DETAIL_LINES))
class TestDetailPayloads:
    def test_identical_in_both_modes(self, details, label) -> None:
        assert details[NotationMode.DIRAC][label] == details[NotationMode.BOTH][label]

    def test_side_by_side_listing_and_dirac(self, details, label) -> None:
        text = details[NotationMode.DIRAC][label]
        assert listing_lines(text), text
        assert side_by_side(text), text

    def test_detail_headers(self, details, label) -> None:
        text = details[NotationMode.DIRAC][label]
        head = {
            "error": "error in apply: apply nosuchrule",
            "proof": "prove:",
            "induction": "induct:",
            "derivation": "derivation: step_sequence",
            "certificate": "proof: 1 step",
            "verification": "verification: verified",
            "decision": "decide:",
            "apply": "step: spider_fusion",
        }[label]
        assert text.startswith(head), text


def test_detail_is_not_shown_by_default() -> None:
    shell = Shell(commands=True)
    shell.run_line(EXAMPLE)
    out = shell.run_line("prove")
    assert "graph:" not in out and "proof:" not in out
    assert shell.run_line("detail").startswith("prove:")


# -- failure reports -----------------------------------------------------------------------


def counterexample(text: str) -> dict[str, int]:
    (line,) = [ln for ln in text.splitlines() if ln.startswith("counterexample: ")]
    body = line[len("counterexample: ") :].split(":", 1)[0]
    return {k.strip(): int(v) for k, v in (p.split("=") for p in body.split(","))}


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
class TestFailureReports:
    @pytest.mark.parametrize("command", ["decide", "check", "prove"])
    def test_unequal_target_reports_dirac_and_counterexample(self, mode, command) -> None:
        shell = Shell(commands=True, listing=True, mode=mode)
        shell.run_script([EXAMPLE, "apply spider_fusion"])
        out = shell.run_line(f"{command} against {UNEQUAL}")
        assert not shell.last_outcome.ok
        lines = out.splitlines()
        assert "dirac outcome: left != right" in lines
        assert "  left:  GHZ_{n+2} = sum_{k=0}^{d-1} |k>^{n} |k, k>" in lines
        assert "  right: GHZ_{n+3} = sum_{k=0}^{d-1} |k>^{n} |k, k, k>" in lines
        target = C.Session()
        C.run(target, UNEQUAL)
        point = counterexample(out)
        assert not compare(shell.session.current, target.current, point).matched
        assert recognize(shell.session.current) == "GHZ_{n+2}"

    def test_value_counterexample_is_an_oracle_mismatch(self, mode) -> None:
        shell = Shell(commands=True, listing=True, mode=mode)
        shell.run_line("Z_{0->3}[d]")
        for command in ("check", "decide"):
            out = shell.run_line(f"{command} against X_{{0->3}}[d]")
            assert "dirac outcome: left != right" in out
            point = counterexample(out)
            target = C.Session()
            C.run(target, "X_{0->3}[d]")
            result = compare(shell.session.current, target.current, point)
            assert not result.matched and result.max_abs_deviation > 0.5

    def test_induct_refutation_has_a_counterexample(self, mode) -> None:
        shell = Shell(commands=True, listing=True, mode=mode)
        shell.run_script([EXAMPLE, "apply spider_fusion"])
        out = shell.run_line("induct against sum_{k=0}^{d-1} |k>^{n} |k>")
        assert out.startswith("induct: refuted")
        assert "dirac outcome: left != right" in out
        assert "counterexample: n = 0 with d = 3" in out

    def test_unknown_decide_shows_simplified_state_and_bound(self, mode, monkeypatch) -> None:
        patched = functools.partial(
            decide_equal, samples=[], use_symbolic=False, use_induction=False
        )
        monkeypatch.setattr(C, "decide_equal", patched)
        shell = Shell(commands=True, listing=True, mode=mode)
        shell.run_line("Z_{0->3}[d]")
        out = shell.run_line("decide against X_{0->3}[d]")
        assert out.startswith("decide: unknown")
        assert "dirac outcome: left ?= right" in out
        assert "furthest-simplified left (normal form" in out
        assert "furthest-simplified right (normal form" in out
        assert "bound: saturation stop" in out
        assert "counterexample" not in out
        assert bool(listing_lines(out)) is (mode is NotationMode.BOTH)

    def test_not_found_prove_shows_simplified_state_and_bound(self, mode, monkeypatch) -> None:
        monkeypatch.setattr(C, "prove", functools.partial(prove, refute=False, moves=()))
        shell = Shell(commands=True, listing=True, mode=mode)
        shell.run_line(EXAMPLE)
        out = shell.run_line("prove against sum_{k=0}^{d-1} |k>^{n} |k> |k>")
        assert out.startswith("prove: not_found")
        assert "dirac outcome: left ?= right" in out
        assert "furthest-simplified left (normal form, 1 step, stop fixpoint):" in out
        assert "GHZ_{n+2}" in out.split("furthest-simplified left", 1)[1]
        assert re.search(r"bound: search exhausted \(limits depth \d+", out)
        assert bool(listing_lines(out)) is (mode is NotationMode.BOTH)


# -- the mode gates nothing ----------------------------------------------------------------

COMMAND_LINES = {
    "load": f"load {EXAMPLE}",
    "show": "show",
    "rules": "rules",
    "match": "match",
    "apply": "apply spider_fusion",
    "check": "check d=2 n=1",
    "normalize": "normalize",
    "decide": "decide",
    "saturate": "saturate",
    "prove": "prove",
    "induct": "induct",
    "certificate": "certificate",
    "translate": "translate",
    "name": "name g",
    "names": "names",
    "undo": "undo",
    "reset": "reset",
    "detail": "detail",
    "help": "help",
}


def test_command_table_is_covered() -> None:
    assert set(COMMAND_LINES) == set(C.COMMANDS)


@pytest.mark.parametrize("name", sorted(C.COMMANDS))
def test_every_command_accepted_in_both_modes(name: str) -> None:
    outputs: dict[NotationMode, str] = {}
    outcomes: dict[NotationMode, C.Outcome] = {}
    for mode in MODES:
        shell = Shell(commands=True, listing=True, mode=mode)
        shell.run_script(
            [EXAMPLE] if name in ("apply", "undo") else [EXAMPLE, "apply spider_fusion"]
        )
        outputs[mode] = shell.run_line(COMMAND_LINES[name])
        assert shell.last_outcome is not None
        outcomes[mode] = shell.last_outcome
        assert shell.last_outcome.command == name
        assert shell.last_outcome.ok, (mode, outputs[mode])
    a, b = outcomes[NotationMode.DIRAC], outcomes[NotationMode.BOTH]
    assert C.format_outcome(a, NotationMode.BOTH, listing=True) == outputs[NotationMode.BOTH]
    assert C.format_outcome(b, NotationMode.DIRAC) == outputs[NotationMode.DIRAC]
    assert not listing_lines(outputs[NotationMode.DIRAC]) or name == "detail"


@pytest.mark.parametrize("name", sorted(C.COMMANDS))
def test_bare_command_name_dispatches_in_both_modes(name: str) -> None:
    results = []
    for mode in MODES:
        shell = Shell(commands=True, listing=True, mode=mode)
        shell.run_line(EXAMPLE)
        out = shell.run_line(name)
        assert shell.last_outcome is not None and shell.last_outcome.command == name
        assert "Traceback" not in out
        results.append((shell.last_outcome.ok, out.splitlines()[0] if out else ""))
    assert results[0][0] == results[1][0]
    if not results[0][0]:
        assert results[0][1] == results[1][1]


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_shell_commands_in_both_modes(mode: NotationMode) -> None:
    shell = Shell(commands=True, listing=True, mode=mode)
    assert shell.run_line(mode.value) == f"mode: {mode.value}"
    assert shell.run_line("history") == f"1  {mode.value}\n2  history"
    assert shell.run_line("list-rules spider_fusion").startswith("rule: spider_fusion")
    assert shell.run_line("quit") == ""
    assert shell.done and shell.errors == 0
