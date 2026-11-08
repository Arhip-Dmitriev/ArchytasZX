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

"""Multi-controlled Toffoli C^n X by the V-chain: n - 2 clean ancillas, Clifford+T Toffolis
computing the AND ladder, one onto the target, the ladder uncomputed; equals C^n X with the
ancillas returned to |0>."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import toffoli
from benchmarks.subjects._spec import defining


def _wires(n):
    return unrolled("c", n), "t", unrolled("a", max(n - 2, 0))


def _circuit(n, d):
    """The compute ladder, the target Toffoli, the uncompute ladder."""
    c, t, a = _wires(n)
    ladder = []
    if n > 2:
        ladder += toffoli(c[0], c[1], a[0])
        for i in range(1, n - 2):
            ladder += toffoli(c[i + 1], a[i - 1], a[i])
    last = toffoli(c[0], c[1], t) if n == 2 else toffoli(c[n - 1], a[n - 3], t)
    gates = [Init(q) for q in a] + ladder + last + [g for g in reversed(ladder)]
    return circuit([*c, t, *a], gates)


def _defining(n):
    c, t, a = _wires(n)
    return [*c, t, *a], [*(Init(q) for q in a), MCX(tuple(c), t)]


def _spec(c):
    """C^n X beside |0> ancillas."""
    w, g = _defining(c.nv)
    return defining(c, w, g)


def _reference(n, d):
    """C^n X as one gate."""
    w, g = _defining(n)
    return circuit(w, g)


SUBJECT = Subject(
    id="multi_controlled_toffoli",
    name="Multi-controlled Toffoli, V-chain",
    category="circuit",
    source="barenco1995elementary",
    size_parameter="controls",
    dimension=2,
    domain=Domain(n_min=2, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled", reference=_reference),),
)
