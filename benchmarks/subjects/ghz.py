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

"""GHZ state on n + 1 qudits: F on the first, CX from it onto each of the others."""

from benchmarks.subjects._common import CX, Domain, Equation, F, Init, Subject, circuit, unrolled


def _circuit(n, d):
    """F then a CX fan from c onto the register."""
    return circuit(["c", "r"], [Init("c"), F("c"), Init("r"), CX("c", "r")], registers={"r": n})


def _spec(c):
    """sum_k |k>^(n+1) / sqrt(d)."""
    return f"g = Z(0 -> 2, {c.D})\nrepeat {c.N}: g.out1\nscalar {c.d}^(-1/2)"


def _reference(n, d):
    """GHZ by a CX ladder: F on c, CX c -> r0 -> r1 -> ..."""
    r = unrolled("r", n)
    gates = [Init("c"), F("c")] + [Init(q) for q in r]
    prev = "c"
    for q in r:
        gates.append(CX(prev, q))
        prev = q
    return circuit(["c", *r], gates)


SUBJECT = Subject(
    id="ghz",
    name="GHZ state",
    category="circuit",
    source="greenberger1989ghz",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
    restricted=(("n >= 2",),),
)
