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

"""Variational quantum eigensolver ansatz, unitary coupled cluster for H2 in a minimal basis: the
Hartree-Fock state |1100> (Jordan-Wigner, four spin orbitals) and the double excitation
e^(theta (a3^dag a2^dag a1 a0 - h.c.)) at theta = pi/4 as its eight commuting Pauli exponentials,
each by basis change, a CX ladder and one phase; the state is (|0011> + |1100>)/sqrt(2)."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, P, S, Sdg

_TERMS = {
    "XXXY": -1,
    "XXYX": -1,
    "XYXX": 1,
    "XYYY": -1,
    "YXXX": 1,
    "YXYY": -1,
    "YYXY": 1,
    "YYYX": 1,
}
_Q = ["q0", "q1", "q2", "q3"]


def _gadget(word, s):
    """e^(i pi/32 s word) = e^(-i 2 pi t word), t = -s/64, up to the phase e^(-2 pi i t)."""
    t = Fraction(-s, 64)
    pre, post = [], []
    for q, letter in zip(_Q, word):
        if letter == "X":
            pre.append(H(q))
            post.append(H(q))
        elif letter == "Y":
            pre += [Sdg(q), H(q)]
            post += [H(q), S(q)]
    ladder = [CX(_Q[j], _Q[j + 1]) for j in range(3)]
    return [*pre, *ladder, P(_Q[3], 2 * t), *reversed(ladder), *post], t


def _circuit(n, d):
    """Hartree-Fock, the eight exponentials."""
    gates = [Init("q0", 1), Init("q1", 1), Init("q2"), Init("q3")]
    total = Fraction(0)
    for word, s in _TERMS.items():
        g, t = _gadget(word, s)
        gates += g
        total += t
    return circuit(_Q, gates, scalar=f"exp(-2*pi*I*({total}))")


def _spec(c):
    """(|0011> + |1100>)/sqrt(2) as GHZ with two flips."""
    from benchmarks.subjects._spec import defining

    gates = [Init(q) for q in _Q] + [F("q0")] + [CX("q0", q) for q in _Q[1:]]
    return defining(c, _Q, [*gates, X("q2"), X("q3")])


SUBJECT = Subject(
    id="vqe_ucc",
    name="VQE unitary coupled-cluster ansatz (H2)",
    category="algorithm",
    source="peruzzo2014variational",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
