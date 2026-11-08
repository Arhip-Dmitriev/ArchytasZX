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

"""Griffiths-Niu semiclassical QFT: the inverse QFT with each qudit measured right after its
F^-1, the later controlled phases classically controlled; equals the inverse QFT followed by
measurement of every qudit."""

from benchmarks.ir import F as _F
from benchmarks.subjects._common import *
from benchmarks.subjects._gates import inverse, qft
from benchmarks.subjects._spec import Dsl, bind, split


def _circuit(n, d):
    """The inverse QFT, a measurement after each F^-1."""
    q = unrolled("q", n)
    gates = []
    for g in inverse(qft(q, d)):
        gates.append(g)
        if isinstance(g, _F):
            gates.append(Measure(g.q))
    return circuit(q, gates)


def _spec(c):
    """The bound register through F^3 (the inverse DFT), split, every leg dephased."""
    n, d = c.nv, c.dv
    b = Dsl()
    ins, port = bind(b, n, d)
    f = b.node(f"F(literal {d**n})")
    f2 = b.node(f"F(literal {d**n})")
    f3 = b.node(f"F(literal {d**n})")
    b.wire(port, f"{f}.in0")
    b.wire(f"{f}.out0", f"{f2}.in0")
    b.wire(f"{f2}.out0", f"{f3}.in0")
    b.ins += ins
    for i, o in enumerate(split(b, f"{f3}.out0", n, d)):
        m = b.node(f"Z(1 -> 2, literal {d})")
        g = b.node(f"G(1 -> 0, literal {d})")
        b.wire(o, f"{m}.in0")
        b.wire(f"{m}.out1", f"{g}.in0")
        b.outs.append(f"{m}.out0")
    return b.text()


SUBJECT = Subject(
    id="semiclassical_qft",
    name="Semiclassical QFT",
    category="algorithm",
    source="griffiths1996semiclassical",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=1, d_min=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled", literal=True),),
)
