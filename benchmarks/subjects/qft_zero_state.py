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

"""The quantum Fourier transform of |0...0> is the uniform superposition: the QFT circuit on n zero
qudits gives |+>^n."""

from benchmarks.subjects._common import *
from benchmarks.subjects.qft import _circuit as _qft


def _circuit(n, d):
    """|0>^n, then the QFT circuit."""
    c = _qft(n, d)
    return circuit(c.wires, [*(Init(q) for q in c.wires), *c.gates])


def _spec(c):
    """sum_k |k> / sqrt(d) per qudit."""
    return "\n".join(
        [f"p{i} = Z(0 -> 1, {c.D})" for i in range(c.nv)] + [f"scalar {c.d}^(-{c.nv}/2)"]
    )


def _reference(n, d):
    """F on every |0>."""
    q = unrolled("q", n)
    return circuit(q, [g for x in q for g in (Init(x), F(x))])


SUBJECT = Subject(
    id="qft_zero_state",
    name="QFT of the zero state",
    category="algorithm",
    source="nielsen2010quantum",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=1, d_min=2),
    equations=(
        Equation("main", _circuit, _spec, form="unrolled", literal=True, reference=_reference),
    ),
)
