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

"""Weyl commutation: the shift then the clock equals w times the clock then the shift."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """X then Z."""
    return circuit(["q"], [X("q"), Z("q")])


def _spec(c):
    """w X Z in time order Z, X: Z then X, scalar w."""
    return (
        f"z = Z(1 -> 1, {c.D}) phase{{k: k/{c.d}}}\nx = X(1 -> 1, {c.D}) phase{{k: -k/{c.d}}}\n"
        f"z -> x\nscalar exp(2*pi*I/{c.d})"
    )


SUBJECT = Subject(
    id="weyl_commutation",
    name="Weyl commutation relation",
    category="circuit",
    source="weyl1927quantenmechanik",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
