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

"""ArchytasZX runner: one ``.azx`` script ending in ``decide`` or ``prove``, in one fresh
session, as ``archytaszx --no-concrete-axioms`` runs it."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from archytaszx.repl.commands import Outcome, Session, Text, _Pair, run
from archytaszx.repl.shell import _logical_lines
from archytaszx.rewrite import match_template
from archytaszx.semantics import kernel
from archytaszx.semantics.decide import Decision, EqualityVerdict
from archytaszx.semantics.prove import ProofStatus
from benchmarks.harness import Inconclusive, Wrong

_VERDICTS = ("decide", "prove")


def decide_script(script: str) -> dict:  # type: ignore[type-arg]
    """The script's CPU and wall seconds, its last ``decide``/``prove`` line's CPU seconds and
    rung, and both sides' node and wire counts; raises :class:`Wrong` on a refutation and
    :class:`Inconclusive` on an unsettled verdict."""
    lines = [ln for ln in _logical_lines(script.splitlines()) if ln.split("#", 1)[0].strip()]
    match_template.CONCRETE_AXIOMS = False
    session = Session()
    outcome = None
    verdict_cpu = 0.0
    c0, w0 = time.process_time(), time.perf_counter()
    for line in lines:
        c = time.process_time()
        outcome = run(session, line)
        if outcome.command in _VERDICTS:
            verdict_cpu = time.process_time() - c
        elif not outcome.ok:
            raise RuntimeError(f"{line.splitlines()[0]}: {outcome.blocks}")
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    if outcome is None or outcome.command not in _VERDICTS:
        raise RuntimeError("the script does not end in decide or prove")
    pair: Any = outcome.detail
    result = pair.payload
    if outcome.command == "decide":
        method = result.method.value
        refuted = result.verdict is EqualityVerdict.UNEQUAL
        reason = result.reason
    else:
        method = "induction" if result.induction is not None else "search"
        refuted = result.status is ProofStatus.REFUTED
        reason = result.reason
    if refuted:
        raise Wrong(f"refuted by {method}: {reason}")
    if not outcome.ok:
        raise Inconclusive(f"{outcome.command}: {reason}")
    return {
        "cpu": cpu,
        "wall": wall,
        "verdict_cpu": verdict_cpu,
        "command": outcome.command,
        "method": method,
        "nodes": len(pair.left.nodes) + len(pair.right.nodes),
        "wires": len(pair.left.wires) + len(pair.right.wires),
    }


@dataclass
class Line:
    """One executed command line: its text, outcome, CPU and wall seconds."""

    text: str
    outcome: Outcome
    cpu: float
    wall: float


@dataclass
class ScriptRun:
    """A script run in one fresh session: the session and every executed line."""

    session: Session
    lines: list[Line] = field(default_factory=list)

    def last(self, command: str, prefix: str = "") -> Line | None:
        """The last line running ``command`` whose text starts with ``prefix``."""
        for line in reversed(self.lines):
            if line.outcome.command == command and line.text.startswith(prefix):
                return line
        return None


def logical_lines(script: str) -> list[str]:
    """The script's command lines, as the shell joins them."""
    return [ln for ln in _logical_lines(script.splitlines()) if ln.split("#", 1)[0].strip()]


def run_script(script: str, *, base: str = "", concrete_axioms: bool = True) -> ScriptRun:
    """Every line of ``script`` in one fresh session, as ``archytaszx FILE`` runs it (concrete
    axioms on unless ``concrete_axioms`` is False); a failing line does not stop the run."""
    match_template.CONCRETE_AXIOMS = concrete_axioms
    out = ScriptRun(Session(base=base))
    for text in logical_lines(script):
        c, w = time.process_time(), time.perf_counter()
        outcome = run(out.session, text)
        out.lines.append(Line(text, outcome, time.process_time() - c, time.perf_counter() - w))
    return out


def until_verdict(run_: ScriptRun) -> tuple[float, float]:
    """CPU and wall seconds of every line up to and including the last ``decide``/``prove``."""
    k = max((i for i, ln in enumerate(run_.lines) if ln.outcome.command in _VERDICTS), default=-1)
    lines = run_.lines[: k + 1]
    return sum(ln.cpu for ln in lines), sum(ln.wall for ln in lines)


def text_of(outcome: Outcome) -> str:
    """The outcome's text blocks, one per line."""
    return "\n".join(b.text for b in outcome.blocks if isinstance(b, Text))


def verdict(run_: ScriptRun) -> dict[str, Any]:
    """The last ``decide``/``prove`` line: ``verdict`` (equal, unequal, unknown), ``method``,
    ``reason``, ``verdict_cpu``, ``verdict_wall``, both sides' ``nodes`` and ``wires``, and
    for a proof its ``derivation_length``."""
    line = next((ln for ln in reversed(run_.lines) if ln.outcome.command in _VERDICTS), None)
    if line is None or not isinstance(line.outcome.detail, _Pair):
        failed = next((ln for ln in run_.lines if not ln.outcome.ok), None)
        reason = text_of(failed.outcome) if failed is not None else "no decide or prove line"
        raise RuntimeError(reason.splitlines()[0] if reason else "no verdict")
    pair = line.outcome.detail
    result: Any = pair.payload
    out: dict[str, Any] = {
        "verdict_cpu": line.cpu,
        "verdict_wall": line.wall,
        "command": line.text.split("\n", 1)[0],
        "nodes": len(pair.left.nodes) + len(pair.right.nodes),
        "wires": len(pair.left.wires) + len(pair.right.wires),
        "derivation_length": None,
    }
    if isinstance(result, Decision):
        out["verdict"] = result.verdict.value
        out["method"] = result.method.value
        out["reason"] = result.reason
        out["assumptions"] = len(result.assumptions)
        if result.certificates:
            out["derivation_length"] = sum(len(c.steps) for c in result.certificates)
    else:
        status = result.status
        out["verdict"] = (
            "equal"
            if status is ProofStatus.PROVED
            else "unequal"
            if status is ProofStatus.REFUTED
            else "unknown"
        )
        if status is ProofStatus.REFUTED:
            out["method"] = (
                "interface" if result.counterexample is None else "oracle_counterexample"
            )
        elif result.induction is not None:
            out["method"] = "induction"
        elif status is ProofStatus.PROVED and line.text.startswith("prove exhaustive"):
            out["method"] = "exhaustive"
        else:
            out["method"] = "search"
        out["reason"] = result.reason
        out["assumptions"] = len(result.assumptions)
        if result.certificate is not None:
            out["derivation_length"] = result.certificate.length
        elif result.induction is not None:
            out["derivation_length"] = result.induction.length
    return out


def kernel_fields(run_: ScriptRun) -> dict[str, Any]:
    """The last ``certificate kernel`` line: ``kernel`` (theorem, symbolic, refutation,
    rejected, or None when absent), ``kernel_cpu``, ``inferences`` and ``kernel_rules``."""
    line = run_.last("certificate", "certificate kernel")
    if line is None:
        return {"kernel": None, "kernel_cpu": None, "inferences": None, "kernel_rules": None}
    detail = line.outcome.detail
    out: dict[str, Any] = {"kernel_cpu": line.cpu, "inferences": None, "kernel_rules": None}
    if not line.outcome.ok:
        out["kernel"] = "rejected"
        out["kernel_detail"] = text_of(line.outcome)[-300:]
    elif isinstance(detail, kernel.Theorem):
        out["kernel"] = "theorem"
        out["inferences"] = detail.size()
        out["kernel_rules"] = len(set(detail.axioms()))
    elif isinstance(detail, kernel.Symbolic):
        out["kernel"] = "symbolic"
    elif isinstance(detail, kernel.Refutation):
        out["kernel"] = "refutation"
    else:
        out["kernel"] = type(detail).__name__
    return out


def lean_export_fields(run_: ScriptRun) -> dict[str, Any]:
    """The last ``certificate lean`` line: ``export_ok``, ``export_cpu``, ``export_wall`` and
    the shell's ``lean_semantics`` line."""
    line = run_.last("certificate", "certificate lean")
    if line is None:
        return {"export_ok": None, "export_cpu": None, "export_wall": None, "lean_semantics": None}
    lean_lines = [t for t in text_of(line.outcome).splitlines() if t.startswith("lean: ")]
    semantics = next((t for t in lean_lines if not t.startswith("lean: wrote")), "")
    return {
        "export_ok": line.outcome.ok,
        "export_cpu": line.cpu,
        "export_wall": line.wall,
        "lean_semantics": semantics.removeprefix("lean: ") or text_of(line.outcome)[-300:],
    }
