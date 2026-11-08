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

"""Disentangling a GHZ state: the CX fan applied twice with opposite signs leaves the first qudit
in |+> and the others in |0>."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """GHZ by F and the CX fan, then the inverse fan."""
    gates = [Init("c"), F("c"), Init("r"), CX("c", "r"), CX("c", "r", -1)]
    return circuit(["c", "r"], gates, registers={"r": n})


def _spec(c):
    """|+> (x) |0>^n."""
    return f"p = Z(0 -> 1, {c.D})\nrepeat {c.N}: X(0 -> 1, {c.D})\nscalar {c.d}^(-({c.n}+1)/2)"


def _reference(n, d):
    """F on the first qudit, nothing else."""
    r = unrolled("r", n)
    return circuit(["c", *r], [Init("c"), F("c"), *(Init(q) for q in r)])


SUBJECT = Subject(
    id="ghz_disentangle",
    name="GHZ disentangling",
    category="circuit",
    source="greenberger1989ghz",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
)
