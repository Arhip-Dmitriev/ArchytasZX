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

"""The quantum Fourier transform followed by its inverse is the identity on n qudits."""

from benchmarks.subjects._common import *
from benchmarks.subjects.inverse_qft import _circuit as _iqft
from benchmarks.subjects.qft import _circuit as _qft


def _circuit(n, d):
    """QFT then QFT^-1."""
    a, b = _qft(n, d), _iqft(n, d)
    return circuit(a.wires, [*a.gates, *b.gates])


def _spec(c):
    """Identity wires."""
    return "\n".join(f"w{i} = Z(1 -> 1, {c.D})" for i in range(c.nv))


def _reference(n, d):
    """The empty circuit."""
    return circuit(unrolled("q", n), [])


SUBJECT = Subject(
    id="qft_roundtrip",
    name="QFT round trip",
    category="algorithm",
    source="nielsen2010quantum",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=1, d_min=2),
    equations=(
        Equation("main", _circuit, _spec, form="unrolled", literal=True, reference=_reference),
    ),
)
