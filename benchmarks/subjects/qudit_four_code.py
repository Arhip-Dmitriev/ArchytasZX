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

"""Qudit [[4,2,2]]_d code: X stabilizer (1, -1, 1, -1), Z stabilizer (1, 1, 1, 1), logical X
rows (1, -1, 0, 0) and (0, 1, -1, 0); the carrier-fan encoder against the integer CSS normal
form."""

from benchmarks.ir import circuit
from benchmarks.subject import Domain, Equation, Subject
from benchmarks.subjects._qudit_codes import css_encoder, css_spec

_STABS = [[1, -1, 1, -1]]
_LOG = [[1, -1, 0, 0], [0, 1, -1, 0]]
_CARRIERS = [0, 1]


def _circuit(n, d):
    """The carrier-fan encoder."""
    return circuit([f"q{j}" for j in range(4)], css_encoder(4, _STABS, _LOG, _CARRIERS))


def _spec(c):
    """The integer CSS normal form."""
    return css_spec(c, 4, _STABS, _LOG, _CARRIERS)


SUBJECT = Subject(
    id="qudit_four_code",
    name="Qudit [[4,2,2]] code",
    category="code",
    source="grassl1997codes",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("encoding", _circuit, _spec, form="fixed"),),
    clifford=True,
)
