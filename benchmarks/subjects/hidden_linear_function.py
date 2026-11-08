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

"""Hidden linear function problem (Bravyi-Gosset-Konig) on the path 0 - 1 - 2 with S on qubits 0
and 2: H, CZ on the edges, S, H, measure. Here q(x) = 2(x0 x1 + x1 x2) + x0 + x2 mod 4 is linear
on L_q = {000, 111} with q(111) = 2, and the solutions z with 2 z.x = q(x) on L_q are the odd-
parity strings; the outcome is uniform on them."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, S

_Q = ["q0", "q1", "q2"]


def _circuit(n, d):
    """The shallow HLF circuit."""
    gates = (
        [g for q in _Q for g in (Init(q), H(q))]
        + [CZ("q0", "q1"), CZ("q1", "q2"), S("q0"), S("q2")]
        + [H(q) for q in _Q]
        + [Measure(q) for q in _Q]
    )
    return circuit(_Q, gates)


def _spec(c):
    """The odd-parity X spider, every leg dephased."""
    lines = [f"x = X(0 -> 3, {c.D}) phase{{1: 1/2}}"]
    for i in range(3):
        lines += [
            f"m{i} = Z(1 -> 2, {c.D})",
            f"g{i} = G(1 -> 0, {c.D})",
            f"wire x.out{i} -- m{i}.in0, m{i}.out1 -- g{i}.in0",
        ]
    lines += ["outputs m0.out0, m1.out0, m2.out0", "scalar 2^(-1/2)"]
    return "\n".join(lines)


SUBJECT = Subject(
    id="hidden_linear_function",
    name="Hidden linear function",
    category="algorithm",
    source="bravyi2018quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
