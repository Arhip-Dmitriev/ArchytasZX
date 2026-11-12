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

"""Coined quantum walk on the cycle of 2^n sites, one step: the Hadamard coin, then the shift x
+ 1 when the coin is 1 and x - 1 when it is 0, each a multi-controlled incrementer; equals F on
the coin followed by the conditional shift x -> x + 2 coin - 1 on the bound position register."""

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import Dsl, bind, split


def _inc(x, coin, value):
    """Increment of x controlled on coin = value: MCX from the top bit down."""
    n = len(x)
    return [MCX((coin, *x[i + 1 :]), x[i], (value,) + (1,) * (n - 1 - i)) for i in range(n)]


def _circuit(n, d):
    """Coin, controlled increment, controlled decrement."""
    x = unrolled("x", n)
    gates = [F("coin"), *_inc(x, "coin", 1), *reversed(_inc(x, "coin", 0))]
    return circuit(["coin", *x], gates)


def _spec(c):
    """F on the coin; x bound, shifted by -1, the copied coin added at bit value 2, split."""
    n = c.nv
    N = 2**n
    b = Dsl()
    f = b.node(f"F({c.D})")
    b.ins.append(f"{f}.in0")
    ins, port = bind(b, n, 2)
    b.ins += ins
    dec = b.node(f"X(1 -> 1, literal {N}) phase{{k: k/{N}}}")
    b.wire(port, f"{dec}.in0")
    port, coin = f"{dec}.out0", f"{f}.out0"
    if n >= 2:
        z = b.node(f"Z(1 -> 2, {c.D})")
        b.wire(coin, f"{z}.in0")
        coin = f"{z}.out0"
        bits, two_c = bind(b, n, 2)
        for i, p in enumerate(bits):
            leg = f"{z}.out1" if i == n - 2 else f"{b.node(f'X(0 -> 1, {c.D})')}.out0"
            b.wire(leg, p)
        add = b.node(f"X(2 -> 1, literal {N})")
        b.wire(port, f"{add}.in0")
        b.wire(two_c, f"{add}.in1")
        port = f"{add}.out0"
        b.scalar.append(f"2^(-{n - 1}/2) * {N}^(1/2)")
    b.outs += [coin, *split(b, port, n, 2)]
    return b.text()


SUBJECT = Subject(
    id="coined_quantum_walk",
    name="Coined quantum walk on a cycle",
    category="algorithm",
    source="aharonov1993quantum",
    size_parameter="position bits",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
