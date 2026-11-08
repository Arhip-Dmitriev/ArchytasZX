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

"""Holographic pentagon code of Pastawski, Yoshida, Harlow and Preskill, first layer: a central
six-leg perfect tensor (the Choi state of the five-qubit encoder, one bulk leg) contracted on each
of its five boundary legs with one leg of a further perfect tensor. The code is the subgroup of
the network's stabilizer group trivial on the six bulk legs, on the twenty open legs, [[20,6]]."""

from benchmarks.subjects._clifford import contract, trivial_on
from benchmarks.subjects._stab import Pauli, code_subject, independent, pauli, standard_form

_FIVE = standard_form(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])


def _tile():
    """The six-qubit perfect state, bulk leg first: the five-qubit code with Xbar X and Zbar Z."""
    gens = [pauli("I" + str(g)[1:]) for g in _FIVE.defining]
    for lo, letter in ((_FIVE.xbar[0], "X"), (_FIVE.zbar[0], "Z")):
        gens.append(
            pauli(letter + str(lo)[1:]) if str(lo)[0] == "+" else pauli("-" + letter + str(lo)[1:])
        )
    return gens


def _network():
    """The network's generators and the indices of its bulk and open legs."""
    tile = _tile()
    n = 36
    gens = []
    for t in range(6):
        for g in tile:
            x = [0] * n
            z = [0] * n
            x[6 * t : 6 * t + 6] = g.x
            z[6 * t : 6 * t + 6] = g.z
            gens.append(Pauli(tuple(x), tuple(z), g.p))
    labels = [(t, j) for t in range(6) for j in range(6)]
    for i in range(5):
        u, v = labels.index((0, 1 + i)), labels.index((1 + i, 1))
        gens = contract(gens, u, v)
        labels = [lab for k, lab in enumerate(labels) if k not in (u, v)]
    bulk = [k for k, lab in enumerate(labels) if lab[1] == 0]
    return gens, bulk, [k for k in range(len(labels)) if k not in bulk]


def _generators():
    gens, bulk, open_ = _network()
    code = trivial_on(gens, bulk)
    return independent(
        [str(Pauli(tuple(g.x[j] for j in open_), tuple(g.z[j] for j in open_), g.p)) for g in code]
    )


SUBJECT = code_subject(
    id="happy_code",
    name="Holographic pentagon (HaPPY) code",
    source="pastawski2015holographic",
    generators=_generators(),
)
