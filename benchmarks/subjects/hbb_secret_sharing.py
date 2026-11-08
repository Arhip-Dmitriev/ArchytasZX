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

"""Hillery-Buzek-Berthiaume secret sharing over a GHZ state: Alice Bell-measures the secret with
her share, Bob measures his in the Fourier basis, Charlie corrects X^-a Z^-(s + b); the channel
to Charlie is the identity."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """GHZ, Alice's Bell measurement, Bob's Fourier measurement, Charlie's corrections."""
    gates = [
        Init("a"),
        F("a"),
        Init("b"),
        Init("c"),
        CX("a", "b"),
        CX("a", "c"),
        CX("s", "a", -1),
        F("s"),
        F("b"),
        Measure("s"),
        Measure("a"),
        Measure("b"),
        CX("a", "c", -1),
        CZ("s", "c", -1),
        CZ("b", "c", -1),
        Discard("s"),
        Discard("a"),
        Discard("b"),
    ]
    return circuit(["s", "a", "b", "c"], gates)


def _spec(c):
    """The identity channel."""
    return f"i = Z(1 -> 1, {c.D})"


SUBJECT = Subject(
    id="hbb_secret_sharing",
    name="HBB quantum secret sharing",
    category="protocol",
    source="hillery1999quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
