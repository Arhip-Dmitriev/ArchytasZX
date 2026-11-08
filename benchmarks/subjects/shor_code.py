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

"""Qudit Shor code encoding on 9 qudits: CX from the data onto the heads of the other two blocks, F
on the three heads, a CX fan inside each block; the encoder is F, the X spider onto the three
heads and a Z copy spider per block."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Outer phase-flip repetition, inner bit-flip repetition."""
    q = unrolled("q", 9)
    gates = [Init(x) for x in q[1:]] + [CX(q[0], q[3]), CX(q[0], q[6]), F(q[0]), F(q[3]), F(q[6])]
    for h in (0, 3, 6):
        gates += [CX(q[h], q[h + 1]), CX(q[h], q[h + 2])]
    return circuit(q, gates)


def _spec(c):
    """F, X(1 -> 3), Z(1 -> 3) per block."""
    lines = [f"f = F({c.D})", f"x = X(1 -> 3, {c.D})", "wire f.out0 -- x.in0"]
    lines += [f"z{i} = Z(1 -> 3, {c.D})" for i in range(3)]
    lines.append("wire " + ", ".join(f"x.out{i} -- z{i}.in0" for i in range(3)))
    lines.append("inputs f.in0")
    lines.append("outputs " + ", ".join(f"z{i}.out{j}" for i in range(3) for j in range(3)))
    return "\n".join(lines)


SUBJECT = Subject(
    id="shor_code",
    name="Qudit Shor code",
    category="code",
    source="shor1995scheme",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("encoding", _circuit, _spec, form="fixed"),),
)
