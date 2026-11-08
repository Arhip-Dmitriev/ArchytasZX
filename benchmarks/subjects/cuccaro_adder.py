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

"""Cuccaro ripple-carry adder on two n-bit registers with one clean ancilla: MAJ blocks up the
carry chain, UMA blocks down, Clifford+T Toffolis; equals |0, a, b> -> |0, a, a + b mod 2^n>."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import toffoli
from benchmarks.subjects._spec import Dsl, register_adder


def _maj(x, y, w):
    return [CX(w, y), CX(w, x), *toffoli(x, y, w)]


def _uma(x, y, w):
    return [*toffoli(x, y, w), CX(w, x), CX(x, y)]


def _circuit(n, d):
    """Wires z, a (most significant first), b (most significant first)."""
    a = [f"a{i}" for i in range(n)]
    b = [f"b{i}" for i in range(n)]
    gates = [Init("z"), *_maj("z", b[0], a[0])]
    for i in range(1, n):
        gates += _maj(a[i - 1], b[i], a[i])
    for i in range(n - 1, 0, -1):
        gates += _uma(a[i - 1], b[i], a[i])
    gates += _uma("z", b[0], a[0])
    return circuit(["z", *reversed(a), *reversed(b)], gates)


def _spec(c):
    """|0> beside addition on the bound registers."""
    b = Dsl()
    z = b.node(f"X(0 -> 1, {c.D})")
    b.outs.append(f"{z}.out0")
    b.scalar.append("2^(-1/2)")
    return register_adder(c, c.nv, 2, b).text()


SUBJECT = Subject(
    id="cuccaro_adder",
    name="Cuccaro ripple-carry adder",
    category="circuit",
    source="cuccaro2004new",
    size_parameter="bits per register",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
