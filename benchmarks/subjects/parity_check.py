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

"""Parity check: n data qudits and an ancilla receiving their sum mod d; the data copied, the
ancilla the X spider of the copies."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """CX from every data qudit onto the ancilla |0>."""
    return circuit(["x", "p"], [Init("p"), CX("x", "p")], registers={"x": n})


def _spec(c):
    """Each data qudit copied, one copy into the sum."""
    return (
        f"z = Z(1 -> 2, {c.D})\ns = X(1 -> 1, {c.D})\nrepeat {c.N}: z\nwire z.out1 -- s.in0\n"
        f"outputs z.out0, s.out0\nscalar {c.d}^({c.n}/2 - 1/2)"
    )


SUBJECT = Subject(
    id="parity_check",
    name="Parity check",
    category="code",
    source="macwilliams1977theory",
    size_parameter="data qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("syndrome", _circuit, _spec, form="boxed"),),
)
