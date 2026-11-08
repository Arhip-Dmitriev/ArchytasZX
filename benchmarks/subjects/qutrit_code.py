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

"""Qutrit [[3,1,2]]_3 code (the threshold-(2, 3) secret-sharing code of Cleve, Gottesman and
Lo): |s> -> sum_y |y, y + s, y + 2s>; the carrier-fan encoder against the integer CSS normal
form."""

from benchmarks.ir import circuit
from benchmarks.subject import Domain, Equation, Subject
from benchmarks.subjects._qudit_codes import css_encoder, css_spec

STABS = [[1, 1, 1]]
LOG = [[0, 1, 2]]
CARRIERS = [1]


def _circuit(n, d):
    """The carrier-fan encoder."""
    return circuit([f"q{j}" for j in range(3)], css_encoder(3, STABS, LOG, CARRIERS))


def _spec(c):
    """The integer CSS normal form."""
    return css_spec(c, 3, STABS, LOG, CARRIERS)


SUBJECT = Subject(
    id="qutrit_code",
    name="Qutrit [[3,1,2]] code",
    category="code",
    source="cleve1999share",
    size_parameter=None,
    dimension=3,
    domain=Domain(d_min=3, d_max=3),
    equations=(Equation("encoding", _circuit, _spec, form="fixed"),),
    clifford=True,
)
