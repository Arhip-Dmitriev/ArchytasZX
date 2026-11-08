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

"""Square root of SWAP from CX(a, b), a controlled sqrt(X) from b onto a, CX(a, b); equals e^(i
pi/8) e^(-i pi/8 XX) e^(-i pi/8 YY) e^(-i pi/8 ZZ), three commuting Pauli gadgets."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, cphase
from benchmarks.subjects._spec import Dsl, pauli_gadget


def _circuit(n, d):
    """CX, H cS H, CX."""
    gates = [CX("a", "b"), H("a"), *cphase("b", "a", Fraction(1, 4)), H("a"), CX("a", "b")]
    return circuit(["a", "b"], gates)


def _spec(c):
    """The XX, YY and ZZ gadgets and the phase e^(i pi/8)."""
    b = Dsl()
    ia, ib = b.node(f"Z(1 -> 1, {c.D})"), b.node(f"Z(1 -> 1, {c.D})")
    b.ins += [f"{ia}.in0", f"{ib}.in0"]
    ports = {"a": f"{ia}.out0", "b": f"{ib}.out0"}
    for letter in "XYZ":
        pauli_gadget(b, ports, {"a": letter, "b": letter}, "1/16", c.D)
    b.outs += [ports["a"], ports["b"]]
    b.scalar.append("exp(I*pi/8)")
    return b.text()


SUBJECT = Subject(
    id="sqrt_swap",
    name="Square root of SWAP",
    category="circuit",
    source="loss1998quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
