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

"""Quantum Fourier transform on n qudits: F and controlled phases w_{d^(k-j+1)}^(x_j x_k), then
the swaps reversing the register; it equals the DFT of dimension d^n on the bound register
(first qudit most significant)."""

from fractions import Fraction

from benchmarks.subjects._common import (
    Domain,
    Equation,
    F,
    PhaseFn,
    Subject,
    Swap,
    circuit,
    unrolled,
)


def _circuit(n, d):
    """The textbook QFT circuit."""
    q = unrolled("q", n)
    gates = []
    for j in range(n):
        gates.append(F(q[j]))
        for k in range(j + 1, n):
            m = k - j + 1
            gates.append(PhaseFn((q[k], q[j]), lambda x, m=m: Fraction(x[0] * x[1], d**m)))
    for j in range(n // 2):
        gates.append(Swap(q[j], q[n - 1 - j]))
    return circuit(q, gates)


def bound(c, n, inner):
    """Binders joining n legs of dimension d into one, inner on it, splitters back."""
    d = c.dv
    lines = []
    if n == 1:
        return [f"t = {inner}(literal {d})"], "t.in0", "t.out0", [], []
    acc = d
    for i in range(1, n):
        lines.append(f"b{i} = B(literal {acc}, literal {d})")
        acc *= d
    for i in range(2, n):
        lines.append(f"wire b{i - 1}.out0 -- b{i}.in0")
    lines.append(f"t = {inner}(literal {acc})")
    lines.append(f"wire b{n - 1}.out0 -- t.in0")
    for i in range(n - 1, 0, -1):
        lines.append(f"s{i} = S(literal {d**i}, literal {d})")
    lines.append(f"wire t.out0 -- s{n - 1}.in0")
    for i in range(n - 1, 1, -1):
        lines.append(f"wire s{i}.out0 -- s{i - 1}.in0")
    ins = ["b1.in0", "b1.in1"] + [f"b{i}.in1" for i in range(2, n)]
    outs = ["s1.out0", "s1.out1"] + [f"s{i}.out1" for i in range(2, n)]
    return lines, None, None, ins, outs


def _spec(c):
    """The DFT F of dimension d^n between binders and splitters."""
    n = c.nv
    lines, _, _, ins, outs = bound(c, n, "F")
    if n == 1:
        return lines[0]
    return "\n".join([*lines, "inputs " + ", ".join(ins), "outputs " + ", ".join(outs)])


SUBJECT = Subject(
    id="qft",
    name="Quantum Fourier transform",
    category="algorithm",
    source="nielsen2010quantum",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=1, d_min=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled", literal=True),),
)
