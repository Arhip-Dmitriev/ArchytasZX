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

"""Shifted Legendre symbol problem (van Dam-Hallgren-Ip) over p = 5 with hidden shift s = 2: F on
the register x, a flag f = [x + s = 0] post-selected on 0, the phase chi(x + s), F, the phase
chi(y), F^dagger; by the Gauss sum sum_z chi(z) w^z = sqrt(5) the register ends in
(1/5) sum_x (5 [x = -s] - 1) |x>."""

from fractions import Fraction

from benchmarks.subjects._common import *

_P, _S = 5, 2


def _chi(v):
    """The Legendre symbol mod 5 as a turn: 0 for a nonzero square, 1/2 for a non-square."""
    v %= _P
    return Fraction(0) if v == 0 or pow(v, (_P - 1) // 2, _P) == 1 else Fraction(1, 2)


def _circuit(n, d):
    """The vDHI steps on x with the flag f."""
    gates = [
        Init("x"),
        F("x"),
        Init("f"),
        Oracle(("x",), "f", lambda v: int((v[0] + _S) % _P == 0)),
        Post("f"),
        PhaseFn(("x",), lambda v: _chi(v[0] + _S)),
        F("x"),
        PhaseFn(("x",), lambda v: _chi(v[0])),
        F("x", 3),
    ]
    return circuit(["x", "f"], gates, dims={"x": _P})


def _spec(c):
    """Weight -4 at -s, 1 elsewhere, times -1/5."""
    return f"x = Z(0 -> 1, literal {_P}) weight{{{(-_S) % _P}: -4}}\nscalar -1/5"


SUBJECT = Subject(
    id="legendre_hidden_shift",
    name="Shifted Legendre symbol (van Dam-Hallgren-Ip)",
    category="algorithm",
    source="vandam2006quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
