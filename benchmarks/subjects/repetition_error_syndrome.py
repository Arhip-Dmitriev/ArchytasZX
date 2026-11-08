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

"""Repetition code syndrome of a shift error: after encoding on n + 2 qudits, X on the middle qudit
x_1; the neighbouring differences read -1 and +1, the others 0."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Encode, X on x_1, syndromes s_i = x_i - x_(i+1)."""
    x, s = unrolled("x", n + 2), unrolled("s", n + 1)
    gates = [Init(q) for q in x[1:]] + [CX(x[0], q) for q in x[1:]] + [X(x[1])]
    for i in range(n + 1):
        gates += [Init(s[i]), CX(x[i], s[i]), CX(x[i + 1], s[i], -1)]
    return circuit([*x, *s], gates)


def _spec(c):
    """Copies of the data with x_1 shifted by 1; syndromes |d-1>, |1>, |0>, ..."""
    n = c.nv
    lines = [
        f"z = Z(1 -> {n + 2}, {c.D})",
        f"x = X(1 -> 1, {c.D}) phase{{k: -k/{c.d}}}",
        "wire z.out1 -- x.in0",
    ]
    outs = ["z.out0", "x.out0"] + [f"z.out{i}" for i in range(2, n + 2)]
    for i in range(n + 1):
        v = {0: -1, 1: 1}.get(i, 0)
        ph = f" phase{{k: {-v}*k/{c.d}}}" if v else ""
        lines.append(f"a{i} = X(0 -> 1, {c.D}){ph}")
        outs.append(f"a{i}.out0")
    lines += ["inputs z.in0", "outputs " + ", ".join(outs), f"scalar {c.d}^(-{n + 1}/2)"]
    return "\n".join(lines)


SUBJECT = Subject(
    id="repetition_error_syndrome",
    name="Repetition code error syndrome",
    category="code",
    source="nielsen2010quantum",
    size_parameter="qudits beyond two",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
