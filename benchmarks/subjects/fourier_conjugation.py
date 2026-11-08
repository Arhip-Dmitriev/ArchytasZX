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

"""Fourier conjugation turns the shift into the clock: F^dagger, X, F equals Z, on every qudit of a
register."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F^3, X, F per qudit."""
    return circuit(["r"], [F("r", 3), X("r"), F("r")], registers={"r": n})


def _spec(c):
    """Z per qudit."""
    return f"repeat {c.N}: Z(1 -> 1, {c.D}) phase{{k: k/{c.d}}}"


def _reference(n, d):
    """Z per qudit."""
    r = unrolled("r", n)
    return circuit(r, [Z(q) for q in r])


SUBJECT = Subject(
    id="fourier_conjugation",
    name="Fourier conjugation of the shift",
    category="circuit",
    source="gottesman1999fault",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
)
