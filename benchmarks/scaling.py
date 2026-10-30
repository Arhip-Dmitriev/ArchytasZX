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

"""Benchmark 1: ArchytasZX ``decide`` (``prove`` for state copy, as in its example) on true
family equations, with the qudit count bound to each ``n`` and the dimension to each ``d``, and
once with both free."""

from __future__ import annotations

SIZE = "n"
NS = [4**k for k in range(11)]
DS = [2, 3, 5, 8, 64, 1024, 2**20]
QUICK_NS = [1, 16]
QUICK_DS = [2, 3]

# {N} is the bound count ("" or " = VALUE"), {D} the bound dimension; COUNT names the count.
FAMILIES: dict[str, tuple[str, str]] = {
    "state copy": (
        "n",
        """
@rhs =
    dim d{D}
    count n{N}
    repeat n: X(0 -> 1)
    scalar d^((1-n)/2)

load
    dim d{D}
    count n{N}
    X(0 -> 1) -> z = Z(1 -> 1)
    repeat n: z.out0

prove against @rhs
""",
    ),
    "spider fusion": (
        "n",
        """
@rhs =
    dim d{D}
    count n{N}
    z = Z(0 -> 1)
    repeat n: z.out0

load
    dim d{D}
    count n{N}
    a = Z(0 -> 1) -> b = Z(1 -> 1)
    repeat n: b.out0

decide against @rhs
""",
    ),
    "Fourier order 4": (
        "n",
        """
@rhs =
    dim d{D}
    count n{N}
    repeat n: Z(1 -> 1)

load
    dim d{D}
    count n{N}
    repeat n: F -> F -> F -> F

decide against @rhs
""",
    ),
    "Fourier superposition": (
        "n",
        """
@rhs =
    dim d{D}
    count n{N}
    repeat n: Z(0 -> 1)
    scalar d^(-n/2)

load
    dim d{D}
    count n{N}
    repeat n: X(0 -> 1) -> F
    scalar d^(-n/2)

decide against @rhs
""",
    ),
    "Bernstein-Vazirani": (
        "n",
        """
@rhs =
    dim d{D}
    count n{N}
    repeat n: X(0 -> 1) phase{{k: -k/d}}
    scalar d^(-n/2)

load
    dim d{D}
    count n{N}
    repeat n: X(0 -> 1) -> F -> Z(1 -> 1) phase{{k: k/d}} -> F^3
    scalar d^(-n/2)

decide against @rhs
""",
    ),
    "Deutsch-Jozsa constant": (
        "n",
        """
@rhs =
    dim d{D}
    count n{N}
    repeat n: X(0 -> 1)
    t = Z(0 -> 1) phase{{k: -k/d}}
    e = Z(1 -> 0) phase{{k: k/d}}
    scalar d^(-(n+1)/2)

load
    dim d{D}
    count n{N}
    repeat n: X(0 -> 1) -> F^4
    X(0 -> 1) phase{{k: -k/d}} -> F^3 -> u = X(2 -> 1)
    inputs u.in1
    scalar d^(-n/2)

decide against @rhs
""",
    ),
    "Deutsch-Jozsa balanced": (
        "m",
        """
@rhs =
    dim d{D}
    count m{N}
    repeat m: X(0 -> 1)
    o = X(0 -> 1) phase{{k: -k/d}}
    t = Z(0 -> 1) phase{{k: -k/d}}
    scalar d^(-(m+2)/2)

load
    dim d{D}
    count m{N}
    repeat m: X(0 -> 1) -> F^4
    X(0 -> 1) -> F -> c = Z(1 -> 2) -> F^3
    X(0 -> 1) phase{{k: -k/d}} -> F^3 -> u = X(2 -> 1)
    c -> u
    scalar d^(-(m+1)/2)

decide against @rhs
""",
    ),
}


def cases(quick: bool) -> list[dict]:
    """Every family at every (n, d), then free; ``quick`` keeps two of each."""
    ns, ds = (QUICK_NS, QUICK_DS) if quick else (NS, DS)
    out = []
    for family in FAMILIES:
        out += [{"suite": "scaling", "family": family, "d": d, "n": n} for d in ds for n in ns]
        out.append({"suite": "scaling", "family": family, "d": "free", "n": "free"})
    return out


def script(case: dict) -> str:
    """The family's script with ``n`` and ``d`` bound, or free when ``"free"``."""
    _, text = FAMILIES[case["family"]]
    bind = lambda v: "" if v == "free" else f" = {v}"
    return text.format(N=bind(case["n"]), D=bind(case["d"]))


def run(case: dict) -> dict:
    """One ``decide`` of the family script, as a user runs it."""
    from benchmarks.engine import decide_script

    return decide_script(script(case))
