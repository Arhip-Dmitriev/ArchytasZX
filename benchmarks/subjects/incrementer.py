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

"""Quantum incrementer on n qubits: from the most significant bit down, each bit flipped by a
multi-controlled X on all lower bits; equals |x> -> |x + 1 mod 2^n>."""

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import Dsl, bind, split


def _circuit(n, d):
    """MCX(lower bits, x_i) from the top, X on the lowest."""
    x = unrolled("x", n)
    gates = [MCX(tuple(x[i + 1 :]), x[i]) for i in range(n - 1)] + [X(x[n - 1])]
    return circuit(x, gates)


def _spec(c):
    """The shift by one on the bound register."""
    n = c.nv
    D = 2**n
    b = Dsl()
    ins, port = bind(b, n, 2)
    s = b.node(f"X(1 -> 1, literal {D}) phase{{k: -k/{D}}}")
    b.wire(port, f"{s}.in0")
    b.ins += ins
    b.outs += split(b, f"{s}.out0", n, 2)
    return b.text()


SUBJECT = Subject(
    id="incrementer",
    name="Quantum incrementer",
    category="circuit",
    source="barenco1995elementary",
    size_parameter="bits",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
