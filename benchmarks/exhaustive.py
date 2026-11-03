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

"""Benchmark: how far exhaustive mode reaches. From a seed diagram (an encoded lemma's left
side at its literal dimension, or a family's left side at n = 1, d = 2), a target is made by
``k`` registered rule steps chosen by a seeded random generator (every match of every rule of
RULES, in name order, one picked per step); ``prove_exhaustive`` then searches from the seed
to the target. Default timeout 3600 CPU s, three runs; layers, diagrams visited and peak
memory are reported, also on timeout.

Fields: ``seed``, ``k``, ``found``, ``layers_start``, ``layers_goal``, ``visited``, ``time``,
``peak_mb``, ``derivation_length`` (may be below ``k``), ``path`` (the generating rules).
"""

from __future__ import annotations

import random
import resource
import time
from typing import Any, cast

from benchmarks import engine, harness, workloads
from benchmarks.harness import Inconclusive, Wrong

SIZE = "k"
REPEAT = 3
TIMEOUT = 3600.0
RECORD_ENV = True
KS = list(range(1, 13))
QUICK_KS = [1, 2]
LEMMA_SEEDS = ("7", "8", "11", "12", "15", "16", "17", "38", "39", "42")
FAMILY_SEEDS = ("spider fusion", "Fourier order 4", "Fourier superposition", "GHZ")
QUICK_SEEDS = ("[P1] Lemma 11",)


def seeds() -> list[str]:
    """The seed names."""
    return [f"[P1] Lemma {n}" for n in LEMMA_SEEDS] + [f"family {f}" for f in FAMILY_SEEDS]


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every seed at every k."""
    return [
        {"suite": "exhaustive", "seed": s, "k": k}
        for s in (QUICK_SEEDS if quick else seeds())
        for k in (QUICK_KS if quick else KS)
    ]


def seed_diagram(name: str) -> Any:
    """The seed diagram."""
    from archytaszx.semantics.check import instantiate

    if name.startswith("family "):
        family = name.removeprefix("family ")
        run_ = engine.run_script(workloads.family_script(family).split("decide")[0])
        symbol = workloads.FAMILIES[family][0]
        current = run_.session.current
        assert current is not None
        return instantiate(current, {symbol: 1, "d": 2})
    lemma = name.rsplit(" ", 1)[1]
    entry = next(e for e in workloads.lemma_entries() if e["paper"] == "P1" and e["lemma"] == lemma)
    path = workloads.ROOT / str(entry["file"])
    script = workloads.units(path.read_text(encoding="utf-8"))[0]
    run_ = engine.run_script(workloads.split_verdict(script)[0], base=str(path.parent))
    return run_.session.current


def target(seed: Any, k: int, name: str) -> tuple[Any, list[str]]:
    """``seed`` after ``k`` seeded random rule steps, and the rules applied."""
    from archytaszx.rewrite.engine import apply
    from archytaszx.rewrite.rules_library import RULES

    rng = random.Random(f"{name}-{k}")
    current, path = seed, cast(list[str], [])
    rules = [RULES[n] for n in sorted(RULES)]
    while len(path) < k:
        options = []
        for rule in rules:
            try:
                options += [(rule, m) for m in rule.pattern.find_matches(current)]
            except Exception:  # noqa: BLE001, S112
                continue
        rng.shuffle(options)
        for rule, match in options:
            try:
                current = apply(current, rule, match).diagram
            except Exception:  # noqa: BLE001, S112
                continue
            path.append(rule.name)
            break
        else:
            raise Inconclusive(f"no rule applies after {len(path)} step(s)")
    return current, path


def run(case: dict[str, Any]) -> dict[str, Any]:
    """Generate the target, then search for it exhaustively."""
    from archytaszx.semantics.prove import ProofStatus, prove_exhaustive

    seed = seed_diagram(case["seed"])
    goal, path = target(seed, case["k"], case["seed"])
    seen: dict[str, Any] = {"layers_start": 0, "layers_goal": 0, "visited": 0, "path": path}
    reached: dict[str, int] = {}

    def on_layer(report: Any) -> None:
        seen[f"layers_{report.side}"] = report.depth
        reached[report.side] = report.reached
        seen["visited"] = sum(reached.values())
        seen["peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        harness.progress(seen)

    c0, w0 = time.process_time(), time.perf_counter()
    outcome = prove_exhaustive(seed, goal, on_layer=on_layer)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    seen["peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    if outcome.status is ProofStatus.REFUTED:
        raise Wrong(f"a target made by rule steps refuted: {outcome.reason}"[:400])
    found = outcome.status is ProofStatus.PROVED and outcome.certificate is not None
    out = {"cpu": cpu, "wall": wall, "found": found, **seen}
    if not found:
        return out | {"status": "inconclusive", "detail": outcome.reason[:400]}
    assert outcome.certificate is not None
    return out | {"derivation_length": outcome.certificate.length}
