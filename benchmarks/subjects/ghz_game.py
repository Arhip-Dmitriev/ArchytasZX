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

"""Mermin's GHZ game, quantum strategy: the GHZ state measured in X or Y per player; the parity
of the outcomes is computed into a fresh bit; XXX gives parity 0 and XYY parity 1, as the
winning condition x or y or z requires."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, Sdg


def _game(bases):
    def build(n, d):
        """GHZ, the per-player basis change, measurement, parity into p."""
        q = ["a", "b", "c"]
        gates = [Init("a"), Init("b"), Init("c"), F("a"), CX("a", "b"), CX("a", "c")]
        for x, base in zip(q, bases):
            gates += [Sdg(x), H(x)] if base == "Y" else [H(x)]
        gates += [Measure(x) for x in q] + [Init("p")] + [CX(x, "p") for x in q]
        gates += [Measure("p")] + [Discard(x) for x in q]
        return circuit([*q, "p"], gates)

    return build


def _parity(v):
    def spec(c):
        """The parity bit."""
        ph = " phase{1: 1/2}" if v else ""
        return f"p = X(0 -> 1, {c.D}){ph}\nscalar 2^(-1/2)"

    return spec


SUBJECT = Subject(
    id="ghz_game",
    name="GHZ game",
    category="protocol",
    source="mermin1990quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(
        Equation("xxx", _game("XXX"), _parity(0), form="fixed"),
        Equation("xyy", _game("XYY"), _parity(1), form="fixed"),
    ),
)
