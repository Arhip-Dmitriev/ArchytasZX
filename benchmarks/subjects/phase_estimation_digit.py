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

"""Phase estimation of one digit: U = Z on the eigenstate |1> (eigenphase 1/d); F, controlled-U,
F^3 on the control give |1>."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Control in F|0>, CZ onto the target |1>, F^3 on the control."""
    return circuit(["c", "t"], [Init("c"), F("c"), Init("t", 1), CZ("c", "t"), F("c", 3)])


def _spec(c):
    """|1> (x) |1>."""
    return (
        f"a = X(0 -> 1, {c.D}) phase{{k: -k/{c.d}}}\nb = X(0 -> 1, {c.D}) phase{{k: -k/{c.d}}}\n"
        f"scalar {c.d}^(-1)"
    )


def _reference(n, d):
    """|1, 1> by X on both."""
    return circuit(["c", "t"], [Init("c"), Init("t"), X("c"), X("t")])


SUBJECT = Subject(
    id="phase_estimation_digit",
    name="Phase estimation, one digit",
    category="algorithm",
    source="kitaev1995quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
