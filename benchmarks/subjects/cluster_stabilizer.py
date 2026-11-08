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

"""Stabilizer of the linear cluster state: X on vertex 0 and Z on its neighbour leave the state
unchanged."""

from benchmarks.subjects._common import *
from benchmarks.subjects.cluster_state import _circuit as _cluster
from benchmarks.subjects.cluster_state import _spec as _cluster_spec


def _circuit(n, d):
    """The cluster state, then X(q_0) Z(q_1)."""
    c = _cluster(n, d)
    extra = [X("q0")] + ([Z("q1")] if n >= 1 else [])
    return circuit(c.wires, [*c.gates, *extra])


def _spec(c):
    """The cluster state."""
    return _cluster_spec(c)


def _reference(n, d):
    """The cluster state circuit."""
    return _cluster(n, d)


SUBJECT = Subject(
    id="cluster_stabilizer",
    name="Cluster state stabilizer",
    category="circuit",
    source="raussendorf2001one",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=1, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="unrolled", reference=_reference),),
)
