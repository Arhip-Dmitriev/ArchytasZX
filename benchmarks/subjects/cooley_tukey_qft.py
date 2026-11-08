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

"""Cooley-Tukey (mixed-radix) QFT of dimension 6 on a qubit and a qutrit: F_2 on the qubit, the
twiddle w_6^(j1 k2), F_3 on the qutrit; equals the DFT of dimension 6 on the bound pair, the
output digits in the transposed order."""

from fractions import Fraction

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F_2, twiddle, F_3."""
    gates = [F("a"), PhaseFn(("a", "b"), lambda x: Fraction(x[0] * x[1], 6)), F("b")]
    return circuit(["a", "b"], gates, dims={"a": 2, "b": 3})


def _spec(c):
    """Binder (2, 3), F of dimension 6, splitter (3, 2), outputs crossed."""
    return (
        "b = B(literal 2, literal 3)\nf = F(literal 6)\ns = S(literal 3, literal 2)\n"
        "wire b.out0 -- f.in0, f.out0 -- s.in0\ninputs b.in0, b.in1\noutputs s.out1, s.out0"
    )


SUBJECT = Subject(
    id="cooley_tukey_qft",
    name="Cooley-Tukey mixed-radix QFT",
    category="algorithm",
    source="cleve2000fast",
    size_parameter=None,
    dimension=6,
    domain=Domain(d_min=6, d_max=6),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
