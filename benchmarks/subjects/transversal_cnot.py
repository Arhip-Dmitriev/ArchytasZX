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

"""Transversal CNOT on two Steane [[7,1,3]] blocks: both standard-form encoders, then CX from
every qubit of the first block onto its partner; equals the logical CX on the inputs followed by
both encoders."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import rename
from benchmarks.subjects._spec import Dsl
from benchmarks.subjects._stab import css_into, encoder_gates, standard_form
from benchmarks.subjects.steane_code import GENERATORS

_CODE = standard_form(GENERATORS)


def _block(prefix):
    inits, gates = encoder_gates(_CODE)
    return rename([*inits, *gates], {f"q{j}": f"{prefix}{j}" for j in range(7)})


def _circuit(n, d):
    """Encode both blocks, CX qubit by qubit."""
    a, b = unrolled("a", 7), unrolled("b", 7)
    return circuit([*a, *b], [*_block("a"), *_block("b"), *(CX(x, y) for x, y in zip(a, b))])


def _spec(c):
    """CX on the logical inputs, then the CSS normal form of each block."""
    b = Dsl()
    css_into(_CODE, c, b)
    css_into(_CODE, c, b)
    za = b.node(f"Z(1 -> 2, {c.D})")
    xb = b.node(f"X(2 -> 1, {c.D})")
    b.wire(f"{za}.out1", f"{xb}.in1")
    b.wire(f"{za}.out0", b.ins[0])
    b.wire(f"{xb}.out0", b.ins[1])
    b.ins = [f"{za}.in0", f"{xb}.in0"]
    b.scalar.append(f"{c.d}^(1/2)")
    return b.text()


SUBJECT = Subject(
    id="transversal_cnot",
    name="Transversal CNOT, Steane code",
    category="code",
    source="steane1996multiple",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
