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

"""Bridge gate: a controlled add from a to c through an intermediate qudit b by four CX; equals
CX(a, c) with b unchanged."""

from benchmarks.subjects._common import *


def _cx_spec(c):
    """|c, t> -> |c, t + c>: copy on c, sum on t."""
    return (
        f"z = Z(1 -> 2, {c.D})\nx = X(2 -> 1, {c.D})\nwire z.out1 -- x.in1\n"
        f"inputs z.in0, x.in0\noutputs z.out0, x.out0\nscalar {c.d}^(1/2)"
    )


def _circuit(n, d):
    """CX(a, b), CX(b, c), CX^-1(a, b), CX^-1(b, c)."""
    return circuit(
        ["a", "b", "c"], [CX("a", "b"), CX("b", "c"), CX("a", "b", -1), CX("b", "c", -1)]
    )


def _spec(c):
    """CX(a, c) beside the identity on b."""
    return (
        f"z = Z(1 -> 2, {c.D})\nx = X(2 -> 1, {c.D})\ni = Z(1 -> 1, {c.D})\nwire z.out1 -- x.in1\n"
        f"inputs z.in0, i.in0, x.in0\noutputs z.out0, i.out0, x.out0\nscalar {c.d}^(1/2)"
    )


SUBJECT = Subject(
    id="bridge_gate",
    name="Bridge gate",
    category="circuit",
    source="itoko2020optimization",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
