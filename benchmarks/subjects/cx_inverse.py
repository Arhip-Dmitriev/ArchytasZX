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

"""CX followed by CX^-1 on every pair of two registers is the identity."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Pairwise CX then CX^-1 between registers a and b."""
    return circuit(["a", "b"], [CX("a", "b"), CX("a", "b", -1)], registers={"a": n, "b": n})


def _spec(c):
    """The identity on 2n qudits, copy-major."""
    return f"repeat {c.N}: Z(1 -> 1, {c.D}), Z(1 -> 1, {c.D})"


def _reference(n, d):
    """The empty circuit on 2n qudits."""
    w = [x for i in range(n) for x in (f"a{i}", f"b{i}")]
    return circuit(w, [])


SUBJECT = Subject(
    id="cx_inverse",
    name="Controlled add and its inverse",
    category="circuit",
    source="gottesman1999fault",
    size_parameter="qudit pairs",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
)
