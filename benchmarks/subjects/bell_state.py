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

"""Bell state: F on a, CX onto b; sum_k |k, k> / sqrt(d)."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F then CX."""
    return circuit(["a", "b"], [Init("a"), F("a"), Init("b"), CX("a", "b")])


def _spec(c):
    """The cup Z(0 -> 2), normalised."""
    return f"e = Z(0 -> 2, {c.D})\nscalar {c.d}^(-1/2)"


def _reference(n, d):
    """The Bell pair from |0, 0> by F and CX."""
    return circuit(["a", "b"], [Init("a"), Init("b"), F("a"), CX("a", "b")])


SUBJECT = Subject(
    id="bell_state",
    name="Bell state",
    category="circuit",
    source="nielsen2010quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
