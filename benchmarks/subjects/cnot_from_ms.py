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

"""CNOT from the Molmer-Sorensen gate XX(pi/4) = e^(-i pi/4 X X) (a native trapped-ion gate,
here H H, the ZZ phase table, H H) and single-qubit rotations: Ry(pi/2) on the control, XX,
Rx(-pi/2) on both, Ry(-pi/2) on the control."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, P, ry


def _cx_spec(c):
    """|c, t> -> |c, t + c>: copy on c, sum on t."""
    return (
        f"z = Z(1 -> 2, {c.D})\nx = X(2 -> 1, {c.D})\nwire z.out1 -- x.in1\n"
        f"inputs z.in0, x.in0\noutputs z.out0, x.out0\nscalar {c.d}^(1/2)"
    )


def _ms(a, b):
    return [
        H(a),
        H(b),
        PhaseFn((a, b), lambda x: Fraction(-1, 8) if x[0] == x[1] else Fraction(1, 8)),
        H(a),
        H(b),
    ]


def _rx(q, t):
    """e^(i pi t) Rx(2 pi t)."""
    return [H(q), P(q, t), H(q)]


def _circuit(n, d):
    """Ry, XX, Rx Rx, Ry; the scalar removes the rotations' phases."""
    q = Fraction(1, 4)
    gates = [*ry("c", q), *_ms("c", "t"), *_rx("c", -q), *_rx("t", -q), *ry("c", -q)]
    return circuit(["c", "t"], gates, scalar="exp(I*pi/4)")


_spec = _cx_spec


def _reference(n, d):
    """CX."""
    return circuit(["c", "t"], [CX("c", "t")])


SUBJECT = Subject(
    id="cnot_from_ms",
    name="CNOT from the Molmer-Sorensen gate",
    category="circuit",
    source="maslov2017basic",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
