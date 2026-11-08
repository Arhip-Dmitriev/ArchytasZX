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

"""Parity check uncomputed: CX from every data qudit onto the parity ancilla and back returns the
data and the ancilla |0>."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """CX fan into p, then the inverse fan."""
    return circuit(["x", "p"], [Init("p"), CX("x", "p"), CX("x", "p", -1)], registers={"x": n})


def _spec(c):
    """The identity on the data beside |0>."""
    return f"repeat {c.N}: Z(1 -> 1, {c.D})\np = X(0 -> 1, {c.D})\nscalar {c.d}^(-1/2)"


def _reference(n, d):
    """The identity beside a fresh zero."""
    x = unrolled("x", n)
    return circuit([*x, "p"], [Init("p")])


SUBJECT = Subject(
    id="parity_decoding",
    name="Parity check uncomputation",
    category="code",
    source="macwilliams1977theory",
    size_parameter="data qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("decoding", _circuit, _spec, form="boxed", reference=_reference),),
)
