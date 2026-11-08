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

"""Measurement-based CNOT on a four-qudit cluster: the target teleported through two ancillas in
|+> by CZ links, the control joined to the middle ancilla; the target and the middle ancilla
measured in the Fourier basis, the byproducts X^-a, Z^-t on the output and Z^-t on the control
corrected, F^2 on the output; the channel is CX."""

from benchmarks.subjects._common import *


def _cx_spec(c):
    """|c, t> -> |c, t + c>: copy on c, sum on t."""
    return (
        f"z = Z(1 -> 2, {c.D})\nx = X(2 -> 1, {c.D})\nwire z.out1 -- x.in1\n"
        f"inputs z.in0, x.in0\noutputs z.out0, x.out0\nscalar {c.d}^(1/2)"
    )


def _circuit(n, d):
    """Cluster links, two measurements, corrections."""
    gates = [
        Init("a"),
        F("a"),
        Init("b"),
        F("b"),
        CZ("t", "a"),
        CZ("a", "b"),
        CZ("c", "a"),
        F("t", 3),
        Measure("t"),
        F("a", 3),
        Measure("a"),
        CX("a", "b", -1),
        CZ("t", "b", -1),
        CZ("t", "c", -1),
        F("b", 2),
        Discard("t"),
        Discard("a"),
    ]
    return circuit(["c", "t", "a", "b"], gates)


_spec = _cx_spec


SUBJECT = Subject(
    id="mbqc_cnot",
    name="Measurement-based CNOT",
    category="protocol",
    source="raussendorf2003measurement",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
