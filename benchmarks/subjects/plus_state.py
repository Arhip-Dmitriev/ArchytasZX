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

"""Uniform superposition on n qudits: F on every |0>; |+>^n."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F on each register qudit."""
    return circuit(["r"], [Init("r"), F("r")], registers={"r": n})


def _spec(c):
    """sum_k |k> / sqrt(d) per qudit."""
    return f"repeat {c.N}: Z(0 -> 1, {c.D})\nscalar {c.d}^(-{c.n}/2)"


SUBJECT = Subject(
    id="plus_state",
    name="Uniform superposition",
    category="circuit",
    source="nielsen2010quantum",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
