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

"""GHZ state from the star graph state: the centre and n leaves in F|0>, CZ from the centre to
each leaf, F^-1 on the leaves; equals the Z spider with n + 1 outputs."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Star graph state, F^3 on every leaf."""
    gates = [Init("c"), F("c"), Init("l"), F("l"), CZ("c", "l"), F("l", 3)]
    return circuit(["c", "l"], gates, registers={"l": n})


def _spec(c):
    """Z(0 -> n + 1)."""
    return f"z = Z(0 -> 2, {c.D})\nrepeat {c.N}: z.out1\nscalar {c.d}^(-1/2)"


SUBJECT = Subject(
    id="star_graph_ghz",
    name="GHZ from the star graph",
    category="circuit",
    source="hein2004multiparty",
    size_parameter="leaves",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
