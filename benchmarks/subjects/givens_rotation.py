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

"""Givens rotation by pi/4 in the span of |01> and |10>: CX(b, a), a controlled Ry(pi/2) from a
onto b, CX(b, a); equals e^(-i pi/8 (XY - YX)) = e^(-i pi/8 XY) e^(i pi/8 YX), two commuting
Pauli gadgets."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cry
from benchmarks.subjects._spec import Dsl, pauli_gadget


def _circuit(n, d):
    """CX, controlled Ry, CX."""
    return circuit(["a", "b"], [CX("b", "a"), *cry("a", "b", Fraction(1, 4)), CX("b", "a")])


def _spec(c):
    """The XY gadget then the YX gadget."""
    b = Dsl()
    ia, ib = b.node(f"Z(1 -> 1, {c.D})"), b.node(f"Z(1 -> 1, {c.D})")
    b.ins += [f"{ia}.in0", f"{ib}.in0"]
    ports = {"a": f"{ia}.out0", "b": f"{ib}.out0"}
    pauli_gadget(b, ports, {"a": "X", "b": "Y"}, "1/16", c.D)
    pauli_gadget(b, ports, {"a": "Y", "b": "X"}, "-1/16", c.D)
    b.outs += [ports["a"], ports["b"]]
    return b.text()


SUBJECT = Subject(
    id="givens_rotation",
    name="Givens rotation",
    category="circuit",
    source="kivlichan2018quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
