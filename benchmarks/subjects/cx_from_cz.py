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

"""Controlled add from controlled phase: F on the target, CZ, F^3; equals CX."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F(t), CZ(c, t), F^3(t)."""
    return circuit(["c", "t"], [F("t"), CZ("c", "t"), F("t", 3)])


def _spec(c):
    """|c, t> -> |c, t + c>: copy on c, sum on t."""
    return (
        f"z = Z(1 -> 2, {c.D})\nx = X(2 -> 1, {c.D})\nwire z.out1 -- x.in1\n"
        f"inputs z.in0, x.in0\noutputs z.out0, x.out0\nscalar {c.d}^(1/2)"
    )


def _reference(n, d):
    """The CX gate."""
    return circuit(["c", "t"], [CX("c", "t")])


SUBJECT = Subject(
    id="cx_from_cz",
    name="CX from CZ and Fourier",
    category="circuit",
    source="gottesman1999fault",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
