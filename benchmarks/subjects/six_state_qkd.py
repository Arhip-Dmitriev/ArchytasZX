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

"""Six-state key distribution, matching bases: Alice's uniform key bit a sets X^a, her uniform basis
trit b (a qutrit) the basis Z, X or Y: H when b != 0 through a flag qubit f = [b != 0] computed
and uncomputed by an oracle, then S when b = 2; Bob undoes the basis with the same b and
measures; the key pair is perfectly correlated and uniform."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ch


def _flag():
    return Oracle(("b",), "f", lambda x: int(x[0] != 0))


def _s_if_y(sign):
    return PhaseFn(("b", "q"), lambda x: Fraction(sign * x[1] * (x[0] == 2), 4))


def _circuit(n, d):
    """Key bit and basis trit, encoding, decoding in the same basis, measurement."""
    gates = [Init("a"), F("a"), Measure("a"), Init("b"), F("b"), Measure("b")]
    gates += [Init("q"), CX("a", "q"), Init("f"), _flag(), *ch("f", "q"), _s_if_y(1)]
    gates += [_s_if_y(-1), *ch("f", "q"), _flag(), Measure("q"), Discard("b"), Discard("f")]
    return circuit(["a", "b", "f", "q"], gates, dims={"b": 3})


def _spec(c):
    """sum_a (1/2) |a, a><a, a|: Z(0 -> 3) with one leg discarded."""
    return (
        f"z = Z(0 -> 3, {c.D})\ng = G(1 -> 0, {c.D})\nwire z.out2 -- g.in0\n"
        "outputs z.out0, z.out1\nscalar 2^(-1/2)"
    )


SUBJECT = Subject(
    id="six_state_qkd",
    name="Six-state key distribution",
    category="protocol",
    source="bruss1998optimal",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
