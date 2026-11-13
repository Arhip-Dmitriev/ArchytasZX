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

"""GHZ state on 2^m qudits in logarithmic depth: F on qudit 0, then per level every qudit
holding the value copies it by CX to a fresh partner; equals the Z spider with 2^m outputs."""

from benchmarks.subjects._common import *


def _unrolled(m, d):
    """The doubling tree, qudits in leaf-major order (each holder followed by its partner)."""
    n = 2**m
    q = unrolled("q", n)
    gates = [Init(x) for x in q] + [F(q[0])]
    leaves = [0]
    for level in range(m):
        for i in range(2**level):
            gates.append(CX(q[i], q[i + 2**level]))
        leaves = [x for i in leaves for x in (i, i + 2**level)]
    return circuit([q[i] for i in leaves], gates)


def _circuit(m, d):
    """F on q, then a binary tree of depth m whose copy copies its value by CX to a fresh r."""
    body = (Init("r"), CX("q", "r"))
    return circuit(["q"], [Init("q"), F("q"), Tree(m, body, "q", ("q", "r"))])


def _spec(c):
    """Z(0 -> 2^m)."""
    return f"z = Z(0 -> 1, {c.D})\nrepeat 2^{c.N}: z.out0\nscalar {c.d}^(-1/2)"


SUBJECT = Subject(
    id="ghz_log_depth",
    name="GHZ state, logarithmic depth",
    category="circuit",
    source="cruz2019efficient",
    size_parameter="tree levels (2^m qudits)",
    dimension="general",
    domain=Domain(n_min=0, n_max=3, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
