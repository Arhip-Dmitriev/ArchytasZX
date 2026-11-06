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

"""Benchmark: ArchytasZX ``decide`` on a mixed family, the partial trace of the (n + 1)-qudit
GHZ state over n qudits against the maximally mixed state, n written out (no bang box), the
dimension bound to each ``d`` and once free; and the same family with the n discarded legs in
a node box, decided for n >= 1 (``count`` free) and for n = v (``count`` bound)."""

from __future__ import annotations

from typing import Any

SIZE = "n"
NS = [2**k for k in range(6)]
DS = [2, 3, 5, 8, 64, 1024]
QUICK_NS = [1, 4]
QUICK_DS = [2, "free"]
FAMILIES = ("partial trace of GHZ", "partial trace of GHZ, boxed")
BOXED_NS = [1, 4, 64, 1024, 2**20]
QUICK_BOXED_NS = [4]


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every (n, d), d bound or free."""
    ns, ds = (QUICK_NS, QUICK_DS) if quick else (NS, [*DS, "free"])
    boxed = QUICK_BOXED_NS if quick else BOXED_NS
    out = [{"suite": "mixed", "family": FAMILIES[0], "d": d, "n": n} for d in ds for n in ns]
    for d in ds:
        out.append({"suite": "mixed", "family": FAMILIES[1], "count": "free", "d": d, "n": "free"})
        out += [
            {"suite": "mixed", "family": FAMILIES[1], "count": "bound", "d": d, "n": n}
            for n in boxed
        ]
    return out


def script(case: dict[str, Any]) -> str:
    """The GHZ state with legs 1..n discarded, decided against ``maxmixed[d]``."""
    n, d = case["n"], case["d"]
    bound = "" if d == "free" else f" = {d}"
    if case["family"] == FAMILIES[1]:
        clause = "n >= 1" if n == "free" else f"n = {n}"
        body = [f"    dim d{bound}", "    count n", "    g = Z(0 -> 2)"]
        body += ["    repeat n: g.out1 -> G", "    scalar d^(-1/2)"]
        return "load\n" + "\n".join(body) + f"\n\ndecide against maxmixed[d] for {clause}\n"
    lines = [f"    dim d{bound}", f"    g = Z(0 -> {n + 1})"]
    lines += [f"    g.out{k} -> G" for k in range(1, n + 1)]
    lines.append("    scalar d^(-1/2)")
    return "load\n" + "\n".join(lines) + "\n\ndecide against maxmixed[d]\n"


def run(case: dict[str, Any]) -> dict[str, Any]:
    """One ``decide`` of the script, as a user runs it."""
    from benchmarks.engine import decide_script

    return decide_script(script(case))
