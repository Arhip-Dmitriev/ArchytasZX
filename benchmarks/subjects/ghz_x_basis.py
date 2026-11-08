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

"""GHZ state in the Fourier basis: F on every qudit of a GHZ state gives the X spider state of n +
1 legs."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """GHZ, then F everywhere."""
    gates = [Init("c"), F("c"), Init("r"), CX("c", "r"), F("c"), F("r")]
    return circuit(["c", "r"], gates, registers={"r": n})


def _spec(c):
    """X(0 -> n + 1) / sqrt(d)."""
    return f"g = X(0 -> 2, {c.D})\nrepeat {c.N}: g.out1\nscalar {c.d}^(-1/2)"


SUBJECT = Subject(
    id="ghz_x_basis",
    name="GHZ state in the Fourier basis",
    category="circuit",
    source="greenberger1989ghz",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
