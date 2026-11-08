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

"""Dephasing by uniformly random clock gates: Z^k with k a uniform classical dit, the dit
discarded, is the completely dephasing channel."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """A uniform key dit k, CZ from it onto q, the key discarded."""
    gates = [Init("k"), F("k"), Measure("k"), CZ("k", "q"), Discard("k")]
    return circuit(["k", "q"], gates)


def _spec(c):
    """The dephasing channel: Z(1 -> 2) with one output discarded."""
    return (
        f"z = Z(1 -> 2, {c.D})\ng = G(1 -> 0, {c.D})\n"
        "wire z.out1 -- g.in0\ninputs z.in0\noutputs z.out0"
    )


SUBJECT = Subject(
    id="dephasing_twirl",
    name="Dephasing twirl",
    category="protocol",
    source="nielsen2010quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
