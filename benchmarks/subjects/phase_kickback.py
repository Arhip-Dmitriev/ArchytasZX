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

"""Phase kickback: CX from a control onto F|1> multiplies |c> by w^(-c) and leaves the target."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Target F|1>, CX from c."""
    return circuit(["c", "y"], [Init("y", 1), F("y"), CX("c", "y")])


def _spec(c):
    """Z^-1 on c beside the state F|1>."""
    return (
        f"z = Z(1 -> 1, {c.D}) phase{{k: -k/{c.d}}}\ny = Z(0 -> 1, {c.D}) phase{{k: k/{c.d}}}\n"
        f"inputs z.in0\noutputs z.out0, y.out0\nscalar {c.d}^(-1/2)"
    )


SUBJECT = Subject(
    id="phase_kickback",
    name="Phase kickback",
    category="circuit",
    source="cleve1998quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
