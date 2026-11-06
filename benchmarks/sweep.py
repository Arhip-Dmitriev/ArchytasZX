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


"""Benchmark: soundness and settle rate on a seeded random corpus of family pairs (n and d
free). A seed diagram is drawn by :func:`generate` (one to three Z, X or F nodes, wires
between them, a port box, a node box or both on ``n``); the pair is then one of:

- ``rewrite``: the seed against itself after ``k`` seeded random rule steps (true by
  construction);
- ``scaled``: the seed against itself times a factor of :data:`FACTORS` (false where the
  factor is not 1);
- ``restricted``: the ``scaled`` pair under a clause of :data:`CLAUSES`.

Each pair is decided by ``decide_equal``. Every verdict is checked against the numeric oracle
at up to :data:`CHECK_POINTS` points (n <= 5, d <= 6, inside the clause): an EQUAL with a
differing point, or an UNEQUAL whose counterexample agrees or lies outside the clause, is
``wrong``. Fields: ``corpus``, ``seed``, ``k``, ``truth`` (true, false, untested), ``verdict``,
``method``, ``time``, ``boxes``, ``nodes``, ``checked``.
"""

from __future__ import annotations

import itertools
import random
import time
from typing import Any

from benchmarks.harness import Inconclusive, Wrong

SIZE = "k"
REPEAT = 1
TIMEOUT = 300.0
RECORD_ENV = True
CORPORA = ("rewrite", "scaled", "restricted")
SEEDS = range(200)
QUICK_SEEDS = range(3)
KS = (1, 4, 8, 16)
FACTORS = (
    "1",
    "(-1)^n",
    "d^(n-2)",
    "d^((n-1)*(n-2))",
    "(d-1)",
    "d^(n*(d-2))",
    "2^(n-1)",
    "d^(n-n)",
)
CLAUSES = (
    "n = 0",
    "n = 2",
    "n >= 1",
    "n >= 2",
    "n >= 3",
    "n = 2*m",
    "n = 2*m + 1",
    "n = 3*m + 1",
    "d = 2",
    "d = 3",
    "d >= 2",
    "d >= 3",
    "n >= 1, d = 2",
    "n = 2, d >= 2",
    "n = 2*m, d = 3",
)
CHECK_POINTS = 40
GENERATOR_TRIES = 20
MULT_RANGE = range(6)
DIM_RANGE = range(1, 7)


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every corpus at every seed; ``rewrite`` at every k, the others at k = 0."""
    seeds = QUICK_SEEDS if quick else SEEDS
    out = []
    for corpus in CORPORA:
        for seed in seeds:
            for k in KS if corpus == "rewrite" else (0,):
                out.append({"suite": "sweep", "corpus": corpus, "seed": seed, "k": k})
    return out


def _kind(lines: list[str], name: str) -> str:
    return next(ln for ln in lines if ln.startswith(name + " =")).split("=", 1)[1].strip()[0]


def generate(rng: random.Random) -> str:
    """A random boxed diagram in the engine's command syntax, n and d free."""
    lines = ["count n"]
    names: list[tuple[str, int, int]] = []
    for i in range(rng.randint(1, 3)):
        g = rng.choice(["Z", "X", "Z", "X", "F"])
        if g == "F":
            lines.append(f"a{i} = F")
            names.append((f"a{i}", 1, 1))
            continue
        ins, outs = rng.randint(0, 2), rng.randint(1, 2)
        phase = rng.choice(["", "", " phase{1: 1/2}", " phase{k: k/d}"])
        lines.append(f"a{i} = {g}({ins} -> {outs}){phase}")
        names.append((f"a{i}", ins, outs))
    used_out: set[tuple[str, int]] = set()
    used_in: set[tuple[str, int]] = set()
    for _ in range(rng.randint(0, 2)):
        if len(names) < 2:
            break
        a, b = rng.sample(names, 2)
        oi = [j for j in range(a[2]) if (a[0], j) not in used_out]
        ii = [j for j in range(b[1]) if (b[0], j) not in used_in]
        if oi and ii:
            o, i = rng.choice(oi), rng.choice(ii)
            used_out.add((a[0], o))
            used_in.add((b[0], i))
            lines.append(f"wire {a[0]}.out{o} -- {b[0]}.in{i}")
    mode = rng.choice(["port", "node", "both"])
    spiders = [x for x in names if _kind(lines, x[0]) in "ZX"]
    if mode in ("port", "both"):
        ports = [(nm, j) for nm, _, outs in spiders for j in range(outs) if (nm, j) not in used_out]
        if ports:
            nm, j = rng.choice(ports)
            used_out.add((nm, j))
            lines.append(f"repeat n: {nm}.out{j}")
    if mode in ("node", "both"):
        cross = []
        for nm, ins, outs in spiders:
            for j in range(ins):
                if (nm, j) not in used_in:
                    ph = rng.choice(["", " phase{1: 1/3}"])
                    cross.append(f"repeat n: {rng.choice('XZ')}(0 -> 1){ph} -> {nm}.in{j}")
            for j in range(outs):
                if (nm, j) not in used_out:
                    cross.append(f"repeat n: {nm}.out{j} -> {rng.choice('XZ')}(1 -> 0)")
        lines.append(
            rng.choice(
                [
                    "repeat n: X(0 -> 1)",
                    "repeat n: Z(0 -> 1)",
                    "repeat n: z = Z(1 -> 1); wire z.out0 -- z.in0",
                    "repeat n: Z(0 -> 0) phase{1: 1/3}",
                    *cross,
                    *cross,
                ]
            )
        )
    return "\n".join(lines)


def _rewrite(diagram: Any, k: int, rng: random.Random) -> Any:
    from archytaszx.rewrite.engine import apply
    from archytaszx.rewrite.rules_library import RULES

    current, done = diagram, 0
    rules = [RULES[n] for n in sorted(RULES)]
    while done < k:
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
            done += 1
            break
        else:
            raise Inconclusive(f"no rule applies after {done} step(s)")
    return current


def _points(left: Any, right: Any, restriction: Any) -> Any:
    from archytaszx.semantics.decide import _free_symbols

    dl, ml, _ = _free_symbols(left)
    dr, mr, _ = _free_symbols(right)
    mults = ml | mr
    names = sorted(dl | dr | mults)
    ranges = [MULT_RANGE if n in mults else DIM_RANGE for n in names]
    clauses = restriction.clauses if restriction is not None else ()
    for values in itertools.product(*ranges):
        point = dict(zip(names, values, strict=True))
        if all(c.symbol not in point or c.contains(point[c.symbol]) for c in clauses):
            yield point


def _check(left: Any, right: Any, restriction: Any) -> tuple[int, dict[str, int] | None]:
    """Points compared and the first differing point."""
    from archytaszx.semantics.check import compare

    checked = 0
    for point in _points(left, right, restriction):
        if checked >= CHECK_POINTS:
            break
        try:
            result = compare(left, right, point, max_elements=200_000)
        except Exception:  # noqa: BLE001, S112
            continue
        checked += 1
        if not result.matched:
            return checked, point
    return checked, None


def run(case: dict[str, Any]) -> dict[str, Any]:
    """Draw, decide and check one pair."""
    from archytaszx.repl.parser import parse
    from archytaszx.semantics.check import compare
    from archytaszx.semantics.decide import EqualityVerdict, decide_equal
    from archytaszx.semantics.restrict import in_domain

    corpus = case["corpus"]
    rng = random.Random(f"sweep-{corpus}-{case['seed']}")
    for _ in range(GENERATOR_TRIES):
        source = generate(rng)
        try:
            left = parse(source).diagram  # type: ignore[union-attr]
            break
        except Exception:  # noqa: BLE001, S112
            continue
    else:
        raise RuntimeError(f"no valid diagram in {GENERATOR_TRIES} draws")
    clause = None
    if corpus == "rewrite":
        right = _rewrite(left, case["k"], rng)
        factor = "1"
    else:
        factor = rng.choice(FACTORS)
        right = parse(f"{source}\nscalar {factor}").diagram  # type: ignore[union-attr]
        if corpus == "restricted":
            clause = rng.choice(CLAUSES)
    c0, w0 = time.process_time(), time.perf_counter()
    try:
        decision = decide_equal(left, right, restrict=clause)
    except Exception as exc:
        if clause is not None and "free in neither" in str(exc):
            return {"status": "unsupported", "detail": str(exc)[:300], "cpu": 0.0, "wall": 0.0}
        raise
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    restriction = decision.restriction
    out: dict[str, Any] = {
        "cpu": cpu,
        "wall": wall,
        "clause": clause,
        "factor": factor,
        "verdict": decision.verdict.value,
        "method": decision.method.value,
        "boxes": len(left.bang_boxes),
        "nodes": len(left.nodes) + len(right.nodes),
        "source": source,
    }
    checked, differing = _check(left, right, restriction)
    out["checked"] = checked
    out["truth"] = "untested" if checked == 0 else ("false" if differing else "true")
    if corpus == "rewrite" and differing is not None:
        raise Wrong(f"rule steps changed the map at {differing}: {source!r}"[:400])
    if decision.verdict is EqualityVerdict.EQUAL and differing is not None:
        raise Wrong(f"EQUAL, but the oracle differs at {differing}"[:400])
    if decision.verdict is EqualityVerdict.UNEQUAL:
        point = decision.counterexample_original or decision.counterexample
        if point is not None:
            point = dict(point)
            if restriction is not None and not in_domain(restriction, point):
                raise Wrong(f"counterexample {point} outside {clause}")
            try:
                if compare(left, right, point, max_elements=200_000).matched:
                    raise Wrong(f"UNEQUAL, but the oracle agrees at {point}")
            except Wrong:
                raise
            except Exception:  # noqa: BLE001, S110
                pass
    if decision.verdict is EqualityVerdict.UNKNOWN:
        return out | {"status": "inconclusive", "detail": decision.reason[:300]}
    return out
