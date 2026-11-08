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

"""CCZ with T-depth one (Selinger): four clean ancillas take the parities a+b, a+c, b+c, a+b+c
by CX, T on a, b, c and a+b+c and T^dagger on the pairwise parities in one layer, the parities
uncomputed; equals CCZ with the ancillas returned to |0>."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import T, Tdg
from benchmarks.subjects._spec import defining

_PAR = {"p0": ("a", "b"), "p1": ("a", "c"), "p2": ("b", "c"), "p3": ("a", "b", "c")}


def _circuit(n, d):
    """Parities, one T layer, parities undone."""
    comp = [CX(x, p) for p, xs in _PAR.items() for x in xs]
    layer = [T("a"), T("b"), T("c"), Tdg("p0"), Tdg("p1"), Tdg("p2"), T("p3")]
    gates = [Init(p) for p in _PAR] + comp + layer + list(reversed(comp))
    return circuit(["a", "b", "c", *_PAR], gates)


def _spec(c):
    """CCZ beside |0> ancillas."""
    return defining(
        c,
        ["a", "b", "c", *_PAR],
        [
            *(Init(p) for p in _PAR),
            PhaseFn(("a", "b", "c"), lambda x: Fraction(x[0] * x[1] * x[2], 2)),
        ],
    )


SUBJECT = Subject(
    id="ccz_t_depth_one",
    name="CCZ with T-depth one",
    category="circuit",
    source="selinger2013quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
