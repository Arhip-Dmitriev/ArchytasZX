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

"""Superdense coding of 2n dits over n Bell pairs in parallel; the channel is the dephasing of
every dit."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """The superdense circuit on registers a, b, p, q."""
    gates = [
        Measure("a"),
        Measure("b"),
        Init("p"),
        F("p"),
        Init("q"),
        CX("p", "q"),
        CX("a", "p"),
        CZ("b", "p"),
        Discard("a"),
        Discard("b"),
        CX("p", "q", -1),
        F("p", 3),
        F("q", 2),
        Swap("p", "q"),
    ]
    return circuit(["a", "b", "p", "q"], gates, registers={"a": n, "b": n, "p": n, "q": n})


def _spec(c):
    """Two dephasing gadgets per copy."""
    return (
        f"za = Z(1 -> 2, {c.D}); ga = G(1 -> 0, {c.D}); wire za.out1 -- ga.in0\n"
        f"zb = Z(1 -> 2, {c.D}); gb = G(1 -> 0, {c.D}); wire zb.out1 -- gb.in0\n"
        f"repeat {c.N}: za, ga, zb, gb\ninputs za.in0, zb.in0\noutputs za.out0, zb.out0"
    )


SUBJECT = Subject(
    id="superdense_register",
    name="Register superdense coding",
    category="protocol",
    source="bennett1992communication",
    size_parameter="dit pairs",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
