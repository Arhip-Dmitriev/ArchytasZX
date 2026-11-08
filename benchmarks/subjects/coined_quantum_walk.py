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
the coin followed by the shift permutation."""

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import defining


def _inc(x, coin, value):
    """Increment of x controlled on coin = value: MCX from the top bit down."""
    n = len(x)
    return [MCX((coin, *x[i + 1 :]), x[i], (value,) + (1,) * (n - 1 - i)) for i in range(n)]


def _circuit(n, d):
    """Coin, controlled increment, controlled decrement."""
    x = unrolled("x", n)
    gates = [F("coin"), *_inc(x, "coin", 1), *reversed(_inc(x, "coin", 0))]
    return circuit(["coin", *x], gates)


def _shift(n):
    N = 2**n

    def p(v):
        c, x = divmod(v, N)
        return c * N + ((x + 1) % N if c else (x - 1) % N)

    return p


def _spec(c):
    """F on the coin, then the shift."""
    x = unrolled("x", c.nv)
    return defining(c, ["coin", *x], [F("coin"), Perm(("coin", *x), _shift(c.nv))])


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
