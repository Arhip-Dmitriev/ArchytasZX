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

"""The Fourier gate has order four: F^4 on every qudit of a register is the identity."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F four times per qudit."""
    return circuit(["r"], [F("r", 1), F("r", 1), F("r", 1), F("r", 1)], registers={"r": n})


def _spec(c):
    """The identity on n qudits."""
    return f"repeat {c.N}: Z(1 -> 1, {c.D})"


def _reference(n, d):
    """The empty circuit on n qudits."""
    return circuit(unrolled("r", n), [])


SUBJECT = Subject(
    id="fourier_order_four",
    name="Fourier gate of order four",
    category="circuit",
    source="wang2023qufinitezxw",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
)
