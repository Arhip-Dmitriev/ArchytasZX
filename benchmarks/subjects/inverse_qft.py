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

"""Inverse quantum Fourier transform: the QFT circuit reversed with conjugate phases; the inverse
DFT of dimension d^n."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects.qft import _circuit as _qft
from benchmarks.subjects.qft import bound


def _circuit(n, d):
    """The QFT circuit's gates reversed, each inverted."""
    c = _qft(n, d)
    inv = []
    for g in reversed(c.gates):
        if isinstance(g, F):
            inv.append(F(g.q, 3))
        elif isinstance(g, PhaseFn):
            inv.append(PhaseFn(g.qs, lambda x, f=g.f: -Fraction(f(x))))
        else:
            inv.append(g)
    return circuit(c.wires, inv)


def _spec(c):
    """F^3 of dimension d^n between binders and splitters."""
    n = c.nv
    if n == 1:
        return f"t = F(literal {c.dv})\nu = F(literal {c.dv})\nv = F(literal {c.dv})\nt -> u -> v"
    lines, _, _, ins, outs = bound(c, n, "F")
    lines = [ln.replace("t = F(", "t = F(") for ln in lines]
    extra = [f"u = F(literal {c.dv**n})", f"v = F(literal {c.dv**n})"]
    lines = [ln for ln in lines if not ln.startswith("wire t.out0")]
    lines += extra + ["wire t.out0 -- u.in0, u.out0 -- v.in0", f"wire v.out0 -- s{n - 1}.in0"]
    return "\n".join([*lines, "inputs " + ", ".join(ins), "outputs " + ", ".join(outs)])


SUBJECT = Subject(
    id="inverse_qft",
    name="Inverse quantum Fourier transform",
    category="algorithm",
    source="nielsen2010quantum",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=1, d_min=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled", literal=True),),
)
