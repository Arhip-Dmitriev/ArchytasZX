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

"""Qudit SWAP from controlled phases: each of the three controlled adds of the SWAP written as
F, CZ^(+-1), F^-1 on its target, then the antipode F^2 on a; equals the crossing of the two
wires."""

from benchmarks.subjects._common import *


def _cx(c, t, s):
    return [F(t), CZ(c, t, s), F(t, 3)]


def _circuit(n, d):
    """CX(a, b), CX^-1(b, a), CX(a, b) through CZ, F^2 on a."""
    gates = [*_cx("a", "b", 1), *_cx("b", "a", -1), *_cx("a", "b", 1), F("a", 2)]
    return circuit(["a", "b"], gates)


def _spec(c):
    """Two crossed identity wires."""
    return (
        f"i = Z(1 -> 1, {c.D})\nj = Z(1 -> 1, {c.D})\ninputs i.in0, j.in0\noutputs j.out0, i.out0"
    )


SUBJECT = Subject(
    id="swap_from_cz",
    name="SWAP from controlled phases",
    category="circuit",
    source="nielsen2010quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
