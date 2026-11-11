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

"""ArchytasZX stages of one run (BENCHMARKS.txt, section 8), in the calling process.

decide runs the equation script line by line in one fresh session, as ``archytaszx FILE``
runs it (concrete axioms on); certificate runs ``certificate kernel``; lean_export runs
``certificate lean FILE``; lean_check runs the Lean runner on FILE. Each stage is timed and
held to its own CPU limit by a profiling timer.
"""

from __future__ import annotations

import resource
import signal
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STAGES = ("decide", "certificate", "lean_export", "lean_check")


class StageTimeout(Exception):
    """A stage used up its CPU limit."""


_expired = [False]


def _alarm(signum: int, frame: Any) -> None:
    _expired[0] = True
    signal.setitimer(signal.ITIMER_PROF, 0.5)
    raise StageTimeout()


def peak_mb() -> float:
    """The process's peak resident memory in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def timed(fn: Callable[[], dict[str, Any]], limit: float | None) -> dict[str, Any]:
    """``fn()`` under a CPU limit of ``limit`` seconds: its fields plus ``status`` (ok,
    timeout, oom, error), ``cpu``, ``wall`` and ``peak_mb``."""
    from archytaszx import trace

    old = signal.signal(signal.SIGPROF, _alarm)
    _expired[0] = False
    if limit:
        signal.setitimer(signal.ITIMER_PROF, limit)
    trace.reset()
    c0, w0 = time.process_time(), time.perf_counter()
    out: dict[str, Any]
    try:
        out = {"status": "ok", **fn()}
    except StageTimeout:
        out = {"status": "timeout"}
    except MemoryError:
        out = {"status": "oom", "detail": "MemoryError"}
    except Exception as exc:  # noqa: BLE001
        out = {"status": "error", "detail": traceback.format_exception_only(exc)[-1].strip()[:500]}
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, old)
    if _expired[0]:
        out = {"status": "timeout", "detail": out.get("detail", "")}
    out["cpu"] = time.process_time() - c0
    out["wall"] = time.perf_counter() - w0
    out["peak_mb"] = round(peak_mb(), 1)
    if trace.ON:
        out["spans"] = trace.tree()
    return out


def logical_lines(script: str) -> list[str]:
    """The script's command lines, as the shell joins them."""
    from archytaszx.repl.shell import _logical_lines

    return [ln for ln in _logical_lines(script.splitlines()) if ln.split("#", 1)[0].strip()]


@dataclass
class Run:
    """One fresh session and every executed line with its outcome."""

    session: Any
    outcomes: list[Any] = field(default_factory=list)


def new_run(base: str = "") -> Run:
    """A fresh session with concrete axioms on, as the shell starts."""
    from archytaszx.repl.commands import Session
    from archytaszx.rewrite import match_template

    match_template.CONCRETE_AXIOMS = True
    return Run(Session(base=base))


def execute(run_: Run, text: str) -> Any:
    """One command line in the run's session."""
    from archytaszx.repl.commands import run

    out = run(run_.session, text)
    run_.outcomes.append(out)
    return out


def text_of(outcome: Any) -> str:
    """The outcome's text blocks."""
    from archytaszx.repl.commands import Text

    return "\n".join(b.text for b in outcome.blocks if isinstance(b, Text))


def diagrams(script: str) -> tuple[Any, Any]:
    """The loaded circuit diagram and ``@rhs`` of an equation script (lines before decide)."""
    run_ = new_run()
    for line in logical_lines(script):
        if line.startswith(("decide", "prove")):
            break
        out = execute(run_, line)
        if not out.ok:
            raise RuntimeError(f"{line.splitlines()[0]}: {text_of(out)[:300]}")
    return run_.session.current, run_.session.names["rhs"]


def verdict_fields(outcome: Any) -> dict[str, Any]:
    """``verdict`` (equal, unequal, equal_uncertified, unequal_uncertified, unknown),
    ``certified``, ``method``, ``inner_method``, ``reason``, ``derivation_steps``, ``samples``
    and ``assumptions`` of a decide outcome."""
    from archytaszx.repl.commands import _Pair
    from archytaszx.semantics.decide import Decision

    detail = outcome.detail
    if not isinstance(detail, _Pair) or not isinstance(detail.payload, Decision):
        raise RuntimeError(text_of(outcome)[:300] or "no decision")  # noqa: TRY004
    dec = detail.payload
    steps = sum(len(c.steps) for c in dec.certificates) if dec.certificates else None
    if steps is None and dec.induction is not None:
        steps = getattr(dec.induction, "length", None)
    return {
        "verdict": dec.verdict.value,
        "certified": dec.decided,
        "method": dec.method.value,
        "inner_method": dec.inner_method.value if dec.inner_method is not None else None,
        "reason": str(dec.reason)[:400],
        "derivation_steps": steps,
        "samples": dec.samples_checked,
        "assumptions": len(dec.assumptions),
        "family_holds": dec.family_holds,
    }


def decide_stage(run_: Run, script: str) -> dict[str, Any]:
    """Every line of ``script`` in the run's session; a failing load line is an error."""
    fields: dict[str, Any] = {}
    for line in logical_lines(script):
        out = execute(run_, line)
        if out.command == "decide":
            fields = verdict_fields(out)
        elif not out.ok:
            raise RuntimeError(f"{line.splitlines()[0]}: {text_of(out)[:300]}")
    if not fields:
        raise RuntimeError("the script has no decide line")
    return fields


def certificate_stage(run_: Run) -> dict[str, Any]:
    """``certificate kernel``: ``kernel`` (theorem, symbolic, refutation, rejected), the number
    of inferences, the inference kinds and the certificate's pickled bytes."""
    from archytaszx.semantics import kernel

    out = execute(run_, "certificate kernel")
    detail = out.detail
    fields: dict[str, Any] = {"inferences": None, "inference_kinds": None, "rules": None}
    if not out.ok:
        fields["kernel"] = "rejected"
        fields["detail"] = text_of(out)[-300:]
    elif isinstance(detail, kernel.Theorem):
        fields["kernel"] = "theorem"
        fields["inferences"] = detail.size()
        fields["inference_kinds"] = _kinds(detail)
        fields["rules"] = len(detail.axioms())
    elif isinstance(detail, kernel.Symbolic):
        fields["kernel"] = "symbolic"
    elif isinstance(detail, kernel.Refutation):
        fields["kernel"] = "refutation"
    else:
        fields["kernel"] = type(detail).__name__
    return fields


def certificate_bytes(run_: Run) -> int | None:
    """The bytes of the stored certificate as ``certificate show`` prints it."""
    out = execute(run_, "certificate show")
    return len(text_of(out).encode()) if out.ok else None


def _kinds(t: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    stack = [t]
    while stack:
        x = stack.pop()
        k = x.inference.kind
        out[k] = out.get(k, 0) + 1
        stack.extend(x.inference.premises)
    return out


def lean_export_stage(run_: Run, path: Path) -> dict[str, Any]:
    """``certificate lean PATH``: the file's bytes and lines and the shell's semantics line."""
    out = execute(run_, f"certificate lean {path}")
    if not out.ok:
        raise RuntimeError(text_of(out)[-300:])
    lines = [t for t in text_of(out).splitlines() if t.startswith("lean: ")]
    semantics = next((t for t in lines if not t.startswith("lean: wrote")), "")
    if not path.exists():
        raise RuntimeError("no Lean file written")
    data = path.read_bytes()
    return {
        "lean_semantics": semantics.removeprefix("lean: ")[:200],
        "lean_bytes": len(data),
        "lean_lines": data.count(b"\n") + 1,
    }


def lean_check_stage(path: Path, limit: float | None, memory_fraction: float) -> dict[str, Any]:
    """The Lean runner on ``path``; a file Lean does not accept is an error."""
    from benchmarks import lean

    res = lean.check(path, timeout=limit, memory_fraction=memory_fraction)
    if res["lean_status"] == "timeout":
        raise StageTimeout()
    if res["lean_status"] != "accepted":
        res["status"] = "oom" if res["lean_status"] == "oom" else "error"
        res["detail"] = res.get("lean_detail")
    return res


def not_run(reason: str) -> dict[str, Any]:
    """The record of a stage that could not run."""
    return {"status": "not_run", "detail": reason}
