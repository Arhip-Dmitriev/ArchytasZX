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

"""Randomized compiling (Pauli frame randomisation) of CX over Z_d: uniform key dits pick X^a
Z^b on the control and target before CX, the propagated frame is undone after it, the keys
discarded; the channel is CX."""

from benchmarks.subjects._common import *


def _cx_spec(c):
    """|c, t> -> |c, t + c>: copy on c, sum on t."""
    return (
        f"z = Z(1 -> 2, {c.D})\nx = X(2 -> 1, {c.D})\nwire z.out1 -- x.in1\n"
        f"inputs z.in0, x.in0\noutputs z.out0, x.out0\nscalar {c.d}^(1/2)"
    )


def _key(q):
    return [Init(q), F(q), Measure(q)]


def _circuit(n, d):
    """Random frame, CX, inverse propagated frame."""
    k = ["k1", "k2", "k3", "k4"]
    gates = [g for q in k for g in _key(q)]
    gates += [
        CZ("k2", "c"),
        CX("k1", "c"),
        CZ("k4", "t"),
        CX("k3", "t"),
        CX("c", "t"),
        CX("k1", "c", -1),
        CX("k1", "t", -1),
        CX("k3", "t", -1),
        CZ("k2", "c", -1),
        CZ("k4", "c"),
        CZ("k4", "t", -1),
    ]
    gates += [Discard(q) for q in k]
    return circuit([*k, "c", "t"], gates)


_spec = _cx_spec


SUBJECT = Subject(
    id="randomized_compiling",
    name="Randomized compiling",
    category="protocol",
    source="wallman2016noise",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
