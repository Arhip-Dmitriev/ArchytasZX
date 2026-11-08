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

"""Quantum Fourier transform over the dihedral group D4 = <r, s | r^4 = s^2 = 1, s r s = r^-1>:
|g> -> sum_(rho, j, k) sqrt(d_rho/8) rho(g)_jk |rho, j, k> on g = r^a s^b, wires (a1, a0, b). The
irreps are chi_(e1, e2)(g) = (-1)^(e1 a + e2 b) and rho(g) = diag(i^a, (-i)^a) X^b; the output
|m1, m0, k> labels chi_(m/2, k) for even m and rho row j = (m - 1)/2, column k for odd m. The
circuit is QFT_4 on a, then H on b when m is even and X on b when m = 3; the specification is
the matrix from the irreps on one eight-level leg per side."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ch, qft


def _circuit(n, d):
    """QFT_4, the controlled H and the doubly controlled X."""
    gates = [*qft(["a1", "a0"], 2), X("a0"), *ch("a0", "b"), X("a0"), MCX(("a1", "a0"), "b")]
    return circuit(["a1", "a0", "b"], gates)


def _weight(out, g):
    """sqrt(8) <out| F |g> as text."""
    m, k = out >> 1, out & 1
    a, b = g >> 1, g & 1
    if m % 2 == 0:
        return "-1" if ((m // 2) * a + k * b) % 2 else "1"
    j = (m - 1) // 2
    if k != j ^ b:
        return "0"
    return ["sqrt(2)", "sqrt(2)*I", "-sqrt(2)", "-sqrt(2)*I"][(a * (1 + 2 * j)) % 4]


def _spec(c):
    """sum F_(out, g) |out><g| as a weighted 64-level state, split and capped."""
    table = ", ".join(f"{8 * o + g}: {_weight(o, g)}" for o in range(8) for g in range(8))
    return (
        f"m = Z(0 -> 1, literal 64) weight{{{table}}}\n"
        "s = S(literal 8, literal 8)\n"
        "o = S(literal 2, literal 4)\n"
        "p = S(literal 2, literal 2)\n"
        "i = B(literal 2, literal 2)\n"
        "j = B(literal 4, literal 2)\n"
        "cap = Z(2 -> 0, literal 8)\n"
        "wire m.out0 -- s.in0\n"
        "wire s.out0 -- o.in0\n"
        "wire o.out1 -- p.in0\n"
        "wire i.out0 -- j.in0\n"
        "wire s.out1 -- cap.in0\n"
        "wire j.out0 -- cap.in1\n"
        "inputs i.in0, i.in1, j.in1\n"
        "outputs o.out0, p.out0, p.out1\n"
        "scalar 8^(-1/2)"
    )


SUBJECT = Subject(
    id="dihedral_qft",
    name="Quantum Fourier transform over the dihedral group D4",
    category="algorithm",
    source="puschel1999fast",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
