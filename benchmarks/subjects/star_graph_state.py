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

"""Star graph state on n + 1 qudits: |+> everywhere, CZ from the centre to each leaf; the centre Z
spider joined to each leaf Z spider by a Fourier edge."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Centre and leaves in |+>, CZ fan from the centre."""
    gates = [Init("c"), F("c"), Init("r"), F("r"), CZ("c", "r")]
    return circuit(["c", "r"], gates, registers={"r": n})


def _spec(c):
    """Centre spider, one Fourier edge per leaf."""
    return (
        f"c = Z(0 -> 2, {c.D})\nf = F({c.D})\nl = Z(1 -> 1, {c.D})\nf -> l\n"
        f"repeat {c.N}: f, l\nwire c.out1 -- f.in0\noutputs c.out0, l.out0\nscalar {c.d}^(-1/2)"
    )


SUBJECT = Subject(
    id="star_graph_state",
    name="Star graph state",
    category="circuit",
    source="hein2004multiparty",
    size_parameter="leaves",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
