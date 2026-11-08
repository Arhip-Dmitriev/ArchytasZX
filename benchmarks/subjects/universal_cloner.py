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

"""Optimal universal 1 -> 2 qubit cloner in Werner's symmetric-projection form: the input a beside
half b of a Bell pair (b, r), projected by (I + SWAP)/2 on (a, b) through a controlled SWAP from
a |+> ancilla post-selected on <+|. On |x>: 2^(-1/2) (|x x x> + (|x x' x'> + |x' x x'>)/2),
x' = 1 - x, r the purifying qubit; each copy is the depolarised input (2/3) rho + (1/3) I/2. The
specification copies x onto three X adders and flips (a, r) or (b, r) by a W spider on
|0> + |1>/2."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cswap


def _circuit(n, d):
    """Bell pair, |+> ancilla, controlled SWAP, ancilla post-selected."""
    gates = [Init("b"), Init("r"), F("b"), CX("b", "r"), Init("c"), F("c")]
    gates += [*cswap("c", "a", "b"), F("c"), Post("c")]
    return circuit(["c", "a", "b", "r"], gates)


def _spec(c):
    """Z copies of x; W on |0> + |1>/2 choosing the flipped copy; X adders."""
    D = c.D
    return "\n".join(
        [
            f"z = Z(1 -> 3, {D})",
            f"wt = Z(0 -> 1, {D}) weight{{1: 1/2}}",
            f"w = W(1 -> 2, {D})",
            f"fa = Z(1 -> 2, {D})",
            f"fb = Z(1 -> 2, {D})",
            f"xa = X(2 -> 1, {D})",
            f"xb = X(2 -> 1, {D})",
            f"xr = X(3 -> 1, {D})",
            (
                "wire wt.out0 -- w.in0, w.out0 -- fa.in0, w.out1 -- fb.in0, z.out0 -- xa.in0, "
                "fa.out0 -- xa.in1, z.out1 -- xb.in0, fb.out0 -- xb.in1, z.out2 -- xr.in0, "
                "fa.out1 -- xr.in1, fb.out1 -- xr.in2"
            ),
            "inputs z.in0",
            "outputs xa.out0, xb.out0, xr.out0",
            "scalar 2^(3/2)",
        ]
    )


SUBJECT = Subject(
    id="universal_cloner",
    name="Universal quantum cloner",
    category="protocol",
    source="werner1998optimal",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
