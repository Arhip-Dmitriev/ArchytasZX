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

"""Qudit repetition code on n + 1 qudits: encoding |k> -> |k>^(n+1) by a CX fan; syndrome
measurement of neighbouring differences x_i - x_(i+1) into n ancillas, which on the code space
leaves the codeword and zero syndromes."""

from benchmarks.subjects._common import CX, Domain, Equation, Init, Subject, circuit, unrolled


def _encode(n, d):
    """CX from the data qudit onto each of n fresh qudits."""
    return circuit(["q", "r"], [Init("r"), CX("q", "r")], registers={"r": n})


def _encode_spec(c):
    """The copy spider Z(1 -> n + 1)."""
    return f"z = Z(1 -> 2, {c.D})\nrepeat {c.N}: z.out1"


def _syndrome(n, d):
    """Encoding, then ancilla s_i = x_i - x_(i+1) for i < n."""
    x = unrolled("x", n + 1)
    s = unrolled("s", n)
    gates = [Init(q) for q in x[1:]] + [CX(x[0], q) for q in x[1:]]
    for i in range(n):
        gates += [Init(s[i]), CX(x[i], s[i]), CX(x[i + 1], s[i], -1)]
    return circuit([*x, *s], gates)


def _syndrome_spec(c):
    """The copy spider onto n + 1 outputs beside n zero ancillas."""
    n = c.nv
    lines = [f"z = Z(1 -> {n + 1}, {c.D})"]
    lines += [f"a{i} = X(0 -> 1, {c.D})" for i in range(n)]
    if n:
        lines.append(f"scalar {c.d}^(-{n}/2)")
    return "\n".join(lines)


def _encode_reference(n, d):
    """Encoding by a CX ladder q -> r0 -> r1 -> ..."""
    r = unrolled("r", n)
    gates = [Init(q) for q in r]
    prev = "q"
    for q in r:
        gates.append(CX(prev, q))
        prev = q
    return circuit(["q", *r], gates)


SUBJECT = Subject(
    id="repetition_code",
    name="Qudit repetition code",
    category="code",
    source="nielsen2010quantum",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(
        Equation("encoding", _encode, _encode_spec, form="boxed", reference=_encode_reference),
        Equation("syndrome", _syndrome, _syndrome_spec, form="unrolled"),
    ),
)
