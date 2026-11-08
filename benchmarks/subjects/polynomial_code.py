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

"""Quantum polynomial (Reed-Solomon) code [[5,1,3]]_5 of Aharonov and Ben-Or: |s> -> sum over
f(x) = s + a x + b x^2 of |f(1), ..., f(5)>; stabilizer rows (x) and (x^2), the logical row all
ones; the carrier-fan encoder (on a stabilizer-equivalent logical row) against the integer CSS
normal form."""

from benchmarks.ir import circuit
from benchmarks.subject import Domain, Equation, Subject
from benchmarks.subjects._qudit_codes import css_encoder, css_spec

_STABS = [[x % 5 for x in range(1, 6)], [x * x % 5 for x in range(1, 6)]]
_LOG = [[1, 1, 1, 1, 1]]
_LOG_ENC = [[0, 0, 1, 3, 1]]
_CARRIERS = [4]


def _circuit(n, d):
    """The carrier-fan encoder."""
    return circuit([f"q{j}" for j in range(5)], css_encoder(5, _STABS, _LOG_ENC, _CARRIERS))


def _spec(c):
    """The integer CSS normal form of the definition."""
    return css_spec(c, 5, _STABS, _LOG, _CARRIERS)


SUBJECT = Subject(
    id="polynomial_code",
    name="Quantum polynomial code",
    category="code",
    source="aharonov1997fault",
    size_parameter=None,
    dimension=5,
    domain=Domain(d_min=5, d_max=5),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
