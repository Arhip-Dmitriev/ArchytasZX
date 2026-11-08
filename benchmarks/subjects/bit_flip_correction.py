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

"""Three-qubit bit-flip code with correction: encode, an X error on qubit e (or none), syndromes q0
+ q1 and q1 + q2 into two ancillas, Toffoli corrections controlled on the syndrome, decode; the
data is restored and the ancillas hold the syndrome of e."""

from benchmarks.subjects._common import *

SYNDROME = {None: (0, 0), 0: (1, 0), 1: (1, 1), 2: (0, 1)}


def _make(e):
    """The equation for the error on qubit e (None: no error)."""

    def _circuit(n, d):
        gates = [Init("b"), Init("c"), CX("a", "b"), CX("a", "c")]
        if e is not None:
            gates.append(X("abc"[e]))
        gates += [Init("s"), Init("t"), CX("a", "s"), CX("b", "s"), CX("b", "t"), CX("c", "t")]
        gates += [
            MCX(("s", "t"), "a", (1, 0)),
            MCX(("s", "t"), "b", (1, 1)),
            MCX(("s", "t"), "c", (0, 1)),
        ]
        gates += [CX("a", "c"), CX("a", "b"), Post("b"), Post("c")]
        return circuit(["a", "b", "c", "s", "t"], gates)

    def _spec(c):
        s, t = SYNDROME[e]
        ph = lambda v: " phase{1: 1/2}" if v else ""
        return (
            f"i = Z(1 -> 1, {c.D})\ns = X(0 -> 1, {c.D}){ph(s)}\nt = X(0 -> 1, {c.D}){ph(t)}\n"
            "scalar 1/2"
        )

    return Equation(f"error{'none' if e is None else e}", _circuit, _spec, form="fixed")


SUBJECT = Subject(
    id="bit_flip_correction",
    name="Bit-flip code correction",
    category="code",
    source="nielsen2010quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=tuple(_make(e) for e in (None, 0, 1, 2)),
)
