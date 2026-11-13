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

"""Quantum incrementer on n qubits (copy m of register x holds the bit of value 2^m): from the
most significant bit down, each bit flipped by a multi-controlled X on all lower bits; equals
|x> -> |x + 1 mod 2^n>."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """MCX(lower bits, x_m) from the top copy down; the lowest copy's MCX has no control."""
    return circuit(["x"], [LowerMCX("x")], registers={"x": n}, dims={"x": 2})


def _spec(c):
    """The shift by one on the register bound copy by copy: a binder chain B(2, 2^m) from a
    dimension-1 head, X(1 -> 1, 2^n) phase -k/2^n, a splitter chain S(2, 2^m) back."""
    return "\n".join(
        [
            "h = Z(0 -> 1, literal 1)",
            "b = B(literal 2, literal 2^m)",
            f"s = X(1 -> 1, literal 2^{c.N}) phase{{k: -k/2^{c.N}}}",
            "t = S(literal 2, literal 2^m)",
            "e = Z(1 -> 0, literal 1)",
            "wire h.out0 -- b.in1, b.out0 -- s.in0, s.out0 -- t.in0, t.out1 -- e.in0",
            f"repeat {c.N} as V index m: b",
            "chain V: b.out0 -> b.in1",
            f"repeat {c.N} as U index m: t",
            "chain U: t.in0 -> t.out1",
            "inputs b.in0",
            "outputs t.out0",
        ]
    )


SUBJECT = Subject(
    id="incrementer",
    name="Quantum incrementer",
    category="circuit",
    source="barenco1995elementary",
    size_parameter="bits",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
