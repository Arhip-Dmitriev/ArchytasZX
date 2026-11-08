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

"""Quantum one-time pad: two uniform key dits a, b; X^a Z^b then Z^-b X^-a is the identity channel
(correctness); without the decryption, the key discarded, the channel is the completely
depolarising one (secrecy)."""

from benchmarks.subjects._common import *


def _key():
    return [Init("a"), F("a"), Measure("a"), Init("b"), F("b"), Measure("b")]


def _correct(n, d):
    """Encrypt and decrypt with a shared uniform key."""
    gates = [
        *_key(),
        CX("a", "q"),
        CZ("b", "q"),
        CZ("b", "q", -1),
        CX("a", "q", -1),
        Discard("a"),
        Discard("b"),
    ]
    return circuit(["a", "b", "q"], gates)


def _correct_spec(c):
    """The identity channel."""
    return f"i = Z(1 -> 1, {c.D})"


def _secret(n, d):
    """Encrypt, then discard the key."""
    return circuit(
        ["a", "b", "q"], [*_key(), CX("a", "q"), CZ("b", "q"), Discard("a"), Discard("b")]
    )


def _secret_spec(c):
    """Trace the input, output the maximally mixed state."""
    return (
        f"g = G(1 -> 0, {c.D})\nm = G(0 -> 1, {c.D})\n"
        f"inputs g.in0\noutputs m.out0\nscalar {c.d}^(-1/2)"
    )


SUBJECT = Subject(
    id="one_time_pad",
    name="Quantum one-time pad",
    category="protocol",
    source="ambainis2000private",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(
        Equation("correctness", _correct, _correct_spec, form="fixed"),
        Equation("secrecy", _secret, _secret_spec, form="fixed"),
    ),
)
