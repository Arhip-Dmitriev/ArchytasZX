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

"""15-to-1 magic state distillation: fifteen copies of T|+>, the [[15,1,3]] Reed-Muller decoder
(the inverse standard-form encoder), the fourteen check qubits post-selected on |0>; the carrier
holds 2^(-11/2) (|0> + e^(-i pi/4) |1>), the state T^dagger|+> with its success amplitude."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import T, inverse
from benchmarks.subjects._stab import encoder_gates
from benchmarks.subjects.reed_muller_code import SUBJECT as _RM

_CODE = _RM.equations[0].specification.__closure__[0].cell_contents


def _circuit(n, d):
    """T|+> on every qubit, the decoder, checks post-selected."""
    q = [f"q{j}" for j in range(15)]
    _, enc = encoder_gates(_CODE)
    gates = [g for x in q for g in (Init(x), F(x), T(x))] + inverse(enc)
    gates += [Post(q[j]) for j in range(15) if j not in _CODE.carriers]
    return circuit(q, gates)


def _spec(c):
    """2^(-11/2) (|0> + e^(-i pi/4) |1>)."""
    return f"z = Z(0 -> 1, {c.D}) phase{{1: -1/8}}\nscalar 2^(-11/2)"


SUBJECT = Subject(
    id="magic_state_distillation",
    name="15-to-1 magic state distillation",
    category="protocol",
    source="bravyi2005universal",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
