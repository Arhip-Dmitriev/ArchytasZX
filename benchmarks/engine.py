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

from archytaszx.repl.commands import Session, run
from archytaszx.repl.shell import _logical_lines
from archytaszx.rewrite import match_template
from archytaszx.semantics.decide import EqualityVerdict
from archytaszx.semantics.prove import ProofStatus
from benchmarks.harness import Inconclusive, Wrong

_VERDICTS = ("decide", "prove")


def decide_script(script: str) -> dict:
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
    pair = outcome.detail
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
