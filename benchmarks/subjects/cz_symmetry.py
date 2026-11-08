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

"""The controlled phase is symmetric: CZ(a, b) equals the diagonal w^(ab), its two copies joined by
one Fourier edge."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """CZ with b as control, a as target."""
    return circuit(["a", "b"], [CZ("b", "a")])


def _spec(c):
    """Copy spiders on a and b joined by F, scalar sqrt(d)."""
    return (
        f"x = Z(1 -> 2, {c.D})\ny = Z(1 -> 2, {c.D})\nf = F({c.D})\n"
        "wire x.out1 -- f.in0, f.out0 -- y.out1\n"
        f"inputs x.in0, y.in0\noutputs x.out0, y.out0\nscalar {c.d}^(1/2)"
    )


def _reference(n, d):
    """CZ with a as control."""
    return circuit(["a", "b"], [CZ("a", "b")])


SUBJECT = Subject(
    id="cz_symmetry",
    name="Controlled-phase symmetry",
    category="circuit",
    source="gottesman1999fault",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
