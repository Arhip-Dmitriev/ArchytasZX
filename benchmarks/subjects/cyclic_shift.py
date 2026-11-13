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

"""Cyclic shift of n + 1 wires by a chain of SWAP gates; the wire permutation moving every qudit
one place down."""

from benchmarks.subjects._common import *


def _unrolled(n, d):
    """SWAP(q_i, q_(i+1)) for i = 0 .. n - 1."""
    q = unrolled("q", n + 1)
    return circuit(q, [Swap(q[i], q[i + 1]) for i in range(n)])


def _circuit(n, d):
    """Per copy SWAP(c, r), c carried on to r: q_0 is c, q_(m+1) copy m of r; the outputs list
    r before c."""
    gates = [Chain(n, (Swap("c", "r"),), carries=(("c", "r"),))]
    return circuit(["c", "r"], gates, registers={"r": n}, outputs=["r", "c"])


def _spec(c):
    """Identity wires, input i to output i - 1 (cyclically): the head wire's output after the
    box's."""
    return "\n".join(
        [
            f"h = Z(1 -> 1, {c.D})",
            f"repeat {c.N} as V: w = Z(1 -> 1, {c.D})",
            "inputs h.in0, w.in0",
            "outputs w.out0, h.out0",
        ]
    )


SUBJECT = Subject(
    id="cyclic_shift",
    name="Cyclic shift by SWAP chain",
    category="circuit",
    source="nielsen2010quantum",
    size_parameter="wires after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_unrolled),),
)
