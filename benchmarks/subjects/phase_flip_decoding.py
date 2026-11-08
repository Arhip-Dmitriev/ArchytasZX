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

"""Qudit phase-flip code decoding: F^dagger on every qudit and the inverse CX fan after the
encoding return the data and n zeros."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Encode (CX fan, F everywhere), decode (F^3 everywhere, inverse fan)."""
    gates = [Init("r"), CX("q", "r"), F("q"), F("r"), F("q", 3), F("r", 3), CX("q", "r", -1)]
    return circuit(["q", "r"], gates, registers={"r": n})


def _spec(c):
    """The identity beside |0>^n."""
    return f"i = Z(1 -> 1, {c.D})\nrepeat {c.N}: X(0 -> 1, {c.D})\nscalar {c.d}^(-{c.n}/2)"


def _reference(n, d):
    """The identity beside n fresh zeros."""
    r = unrolled("r", n)
    return circuit(["q", *r], [Init(x) for x in r])


SUBJECT = Subject(
    id="phase_flip_decoding",
    name="Qudit phase-flip code decoding",
    category="code",
    source="nielsen2010quantum",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("decoding", _circuit, _spec, form="boxed", reference=_reference),),
)
