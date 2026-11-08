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

"""n Bell pairs in parallel: F on each a_i, CX onto b_i; every pair sum_k |k, k> / sqrt(d), copy-
major."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F and CX per pair."""
    return circuit(
        ["a", "b"], [Init("a"), F("a"), Init("b"), CX("a", "b")], registers={"a": n, "b": n}
    )


def _spec(c):
    """A cup per pair."""
    return f"repeat {c.N}: Z(0 -> 2, {c.D})\nscalar {c.d}^(-{c.n}/2)"


SUBJECT = Subject(
    id="bell_register",
    name="Register of Bell pairs",
    category="circuit",
    source="nielsen2010quantum",
    size_parameter="pairs",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
