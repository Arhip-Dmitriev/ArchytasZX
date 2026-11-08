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

"""iSWAP from S (x) S, H, two CX and H; equals |x, y> -> i^(x + y) |y, x> on the odd-parity
states, the identity on |00> and |11>."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, S
from benchmarks.subjects._spec import defining


def _circuit(n, d):
    """S S, H(a), CX(a, b), CX(b, a), H(b)."""
    return circuit(["a", "b"], [S("a"), S("b"), H("a"), CX("a", "b"), CX("b", "a"), H("b")])


def _defining():
    return [Swap("a", "b"), PhaseFn(("a", "b"), lambda x: Fraction(1, 4) if x[0] != x[1] else 0)]


def _spec(c):
    """SWAP and the phase i on |01>, |10>."""
    return defining(c, ["a", "b"], _defining())


def _reference(n, d):
    """SWAP and the phase table."""
    return circuit(["a", "b"], _defining())


SUBJECT = Subject(
    id="iswap_decomposition",
    name="iSWAP decomposition",
    category="circuit",
    source="schuch2003natural",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
