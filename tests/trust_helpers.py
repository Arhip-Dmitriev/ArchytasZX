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

"""Script helpers of the trust-mutation test: running a command file in one fresh session and
reading its verdict and Lean export lines, and splitting command files into export units."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from archytaszx.repl.commands import Outcome, Session, Text, _Pair, run
from archytaszx.repl.shell import _logical_lines
from archytaszx.rewrite import match_template
from archytaszx.semantics.decide import Decision
from archytaszx.semantics.prove import ProofStatus

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
WORKLOADS = ROOT / "tests" / "data"
_VERDICTS = ("decide", "prove")
_VERDICT = re.compile(r"^(decide|prove)\b")


@dataclass
class Line:
    """One executed command line: its text, outcome, CPU and wall seconds."""

    text: str
    outcome: Outcome
    cpu: float
    wall: float


@dataclass
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


def example_files() -> list[Path]:
    """Every command file under examples/, sorted."""
    return sorted(EXAMPLES.rglob("*.azx"))


def rel(path: Path) -> str:
    """``path`` relative to the repository root."""
    return str(path.resolve().relative_to(ROOT))


def _blocks(script: str) -> list[list[str]]:
    """The script's physical lines grouped into logical commands (a command and its indented
    continuation lines); comment and blank lines are dropped."""
    out: list[list[str]] = []
    for raw in script.splitlines():
        code = raw.split("#", 1)[0].rstrip()
        if not code.strip():
            continue
        if raw[:1] in (" ", "\t") and out:
            out[-1].append(raw)
        else:
            out.append([raw])
    return out


def units(script: str) -> list[str]:
    """One script per ``decide``/``prove`` line: every non-verdict, non-certificate line before
    it, the verdict, and the ``certificate emit``/``kernel`` lines right after it."""
    blocks = _blocks(script)
    out: list[str] = []
    for k, block in enumerate(blocks):
        head = block[0].strip()
        if not _VERDICT.match(head):
            continue
        before = [
            b
            for b in blocks[:k]
            if not _VERDICT.match(b[0].strip()) and not b[0].strip().startswith("certificate")
        ]
        after: list[list[str]] = []
        for b in blocks[k + 1 :]:
            word = b[0].strip()
            if not word.startswith("certificate"):
                break
            if word.startswith(("certificate emit", "certificate kernel")):
                after.append(b)
        out.append("\n".join(line for b in [*before, block, *after] for line in b) + "\n")
    return out


def with_lean(script: str, path: Path, kernel: bool = True) -> str:
    """``script`` followed by ``certificate kernel`` (unless present or ``kernel`` is False)
    and ``certificate lean PATH``."""
    lines = [script.rstrip("\n")]
    if kernel and "certificate kernel" not in script:
        lines.append("certificate kernel")
    lines.append(f"certificate lean {path}")
    return "\n".join(lines) + "\n"
