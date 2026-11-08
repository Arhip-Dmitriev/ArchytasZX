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

"""GHZ conference key agreement: a GHZ state over one party and n more, each measured in the
computational basis; all n + 1 keys agree and are uniform."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """GHZ by a CX fan, every share measured."""
    gates = [Init("a"), F("a"), Init("r"), CX("a", "r"), Measure("a"), Measure("r")]
    return circuit(["a", "r"], gates, registers={"r": n})


def _spec(c):
    """sum_k (1/d) |k ... k><k ... k|."""
    return (
        f"z = Z(0 -> 3, {c.D})\ng = G(1 -> 0, {c.D})\nwire z.out1 -- g.in0\n"
        f"repeat {c.N}: z.out2\noutputs z.out0, z.out2\nscalar {c.d}^(-1/2)"
    )


SUBJECT = Subject(
    id="ghz_key_agreement",
    name="GHZ conference key agreement",
    category="protocol",
    source="epping2017multi",
    size_parameter="parties after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
