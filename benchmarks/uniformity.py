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


"""Benchmark: cost of a derivation against the bound values of n and d. Every family of
:data:`benchmarks.workloads.FAMILIES` is proved by ``prove`` with n and d free and with n and
d bound to each pair of :data:`NS` and :data:`DS`, concrete axioms on (as a user runs it) and
off. A family ``prove`` does not settle is decided instead.

Fields: ``family``, ``axioms`` (on, off), ``n``, ``d``, ``command``, ``steps`` (rewrite steps
of the derivation), ``time`` (the verdict line's CPU seconds), ``script_cpu`` (the whole
script), ``binding_cpu`` (script CPU outside the verdict line: parsing and the binding).
"""

from __future__ import annotations

from typing import Any

from benchmarks import engine, workloads
from benchmarks.harness import Inconclusive

SIZE = "size"
REPEAT = 3
RECORD_ENV = True
NS: tuple[Any, ...] = ("free", 1, 16, 2**20)
DS: tuple[Any, ...] = ("free", 2, 3, 8, 1024)
AXIOMS = ("on", "off")
QUICK_FAMILIES = ("GHZ", "spider fusion")
QUICK_NS: tuple[Any, ...] = ("free", 16)
QUICK_DS: tuple[Any, ...] = ("free", 3)


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every family, axiom setting, n and d."""
    families = QUICK_FAMILIES if quick else tuple(workloads.FAMILIES)
    ns, ds = (QUICK_NS, QUICK_DS) if quick else (NS, DS)
    return [
        {"suite": "uniformity", "family": f, "axioms": a, "n": n, "d": d, "size": 1}
        for f in families
        for a in AXIOMS
        for n in ns
        for d in ds
    ]


def _prove(family: str, n: Any, d: Any, axioms: str) -> tuple[list[str], float, float, str]:
    """The derivation's steps (forward, a separator, backward), the verdict line's and the
    script's CPU seconds, and the command used: ``prove``, else ``decide`` (its rung and its
    normal-form steps)."""
    for command in ("prove", "decide"):
        script = workloads.family_script(family, n=n, d=d)
        script = script.replace("decide against", f"{command} against")
        run_ = engine.run_script(script, concrete_axioms=axioms == "on")
        line = run_.lines[-1]
        payload: Any = getattr(line.outcome.detail, "payload", None)
        certificate = getattr(payload, "certificate", None)
        if certificate is not None:
            steps = [*_steps(certificate.forward), "|", *_steps(certificate.backward)]
        elif getattr(payload, "induction", None) is not None and getattr(payload, "proved", False):
            steps = ["induction", "|", str(payload.reason)]
        elif command == "decide" and getattr(payload, "verdict", None) is not None:
            if payload.verdict.value != "equal":
                break
            steps = [payload.method.value, "|"]
            for part in payload.certificates:
                steps += _steps(part)
        else:
            continue
        return steps, line.cpu, sum(ln.cpu for ln in run_.lines), command
    reason = getattr(payload, "reason", None) or engine.text_of(line.outcome)
    raise Inconclusive(f"not settled: {reason}"[:400])


def _steps(part: Any) -> list[str]:
    """Each step's rule name and the node ids it consumes."""
    steps = getattr(part, "steps", ()) if part is not None else ()
    return [f"{s.rule_name}{tuple(s.consumed_node_ids)}" for s in steps]


def run(case: dict[str, Any]) -> dict[str, Any]:
    """Settle the bound family; its step count and CPU seconds."""
    steps, cpu, script_cpu, command = _prove(case["family"], case["n"], case["d"], case["axioms"])
    return {
        "cpu": cpu,
        "wall": cpu,
        "script_cpu": script_cpu,
        "binding_cpu": max(0.0, script_cpu - cpu),
        "steps": sum(1 for s in steps if s != "|") if steps[:1] != ["induction"] else 0,
        "command": command,
    }
