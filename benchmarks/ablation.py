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

"""Benchmark (ablation, not a user run): what each rung of decide_equal contributes. Every row
is labelled ``ablation``. The pair a workload's verdict line names is decided by
``decide_equal`` under each configuration, through its documented switches only.

Configurations: ``full``; ``no saturation`` (use_saturation=False); ``no symbolic``
(use_symbolic=False); ``no induction`` (use_induction=False); ``no concrete axioms``
(match_template.CONCRETE_AXIOMS = False); ``no normal form`` (no switch: ``unsupported``).
Workloads: every family free, every breadth workload, and the lemmas of :data:`LEMMAS`.
"""

from __future__ import annotations

import time
from typing import Any

from benchmarks import engine, workloads

SIZE = "size"
RECORD_ENV = True
CONFIGS: dict[str, dict[str, Any] | None] = {
    "full": {},
    "no normal form": None,
    "no saturation": {"use_saturation": False},
    "no symbolic": {"use_symbolic": False},
    "no induction": {"use_induction": False},
    "no concrete axioms": {},
}
LEMMAS = ("8", "11", "12", "16", "38", "42")
QUICK = ("spider fusion", "teleportation")


def _workloads() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for group, name, script, base in workloads.unit_scripts():
        lemma = any(name.startswith(f"[P1] Lemma {n} #") for n in LEMMAS)
        if group in ("families", "breadth") or (group == "completeness" and lemma):
            out[name] = (script, base)
    return out


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every workload under every configuration."""
    return [
        {"suite": "ablation", "label": "ablation", "configuration": c, "workload": w, "size": 1}
        for w in _workloads()
        if not quick or w.split(" #")[0] in QUICK
        for c in CONFIGS
    ]


def run(case: dict[str, Any]) -> dict[str, Any]:
    """``decide_equal`` on the workload's pair under the configuration."""
    from archytaszx.repl.commands import _pair_args
    from archytaszx.rewrite import match_template
    from archytaszx.semantics.decide import decide_equal

    switches = CONFIGS[case["configuration"]]
    if switches is None:
        return {
            "status": "unsupported",
            "detail": "decide_equal has no switch for the normal-form rung",
            "cpu": 0.0,
            "wall": 0.0,
        }
    script, base = _workloads()[case["workload"]]
    text, line = workloads.split_verdict(script)
    concrete = case["configuration"] != "no concrete axioms"
    prefix = engine.run_script(text, base=base, concrete_axioms=concrete)
    verb = line.split(None, 1)
    args = verb[1] if len(verb) > 1 else ""
    if args.startswith("exhaustive"):
        args = args.removeprefix("exhaustive").strip()
    left, right = _pair_args(prefix.session, args)
    match_template.CONCRETE_AXIOMS = concrete
    c0, w0 = time.process_time(), time.perf_counter()
    decision = decide_equal(left, right, **switches)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    out = {
        "cpu": cpu,
        "wall": wall,
        "verdict": decision.verdict.value,
        "method": decision.method.value,
        "settled": decision.verdict.value != "unknown",
    }
    if decision.verdict.value == "unknown":
        return out | {"status": "inconclusive", "detail": decision.reason[:400]}
    if decision.verdict.value == "unequal":
        return out | {"status": "wrong", "detail": decision.reason[:400]}
    return out
