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

"""Fourier basis states: F on |1> for every qudit of a register gives sum_k w^k |k> / sqrt(d) per
qudit."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """|1> then F per qudit."""
    return circuit(["r"], [Init("r", 1), F("r")], registers={"r": n})


def _spec(c):
    """The phase state Z(0 -> 1) with phases k/d per qudit."""
    return f"repeat {c.N}: Z(0 -> 1, {c.D}) phase{{k: k/{c.d}}}\nscalar {c.d}^(-{c.n}/2)"


SUBJECT = Subject(
    id="fourier_basis_state",
    name="Fourier basis state",
    category="circuit",
    source="nielsen2010quantum",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
