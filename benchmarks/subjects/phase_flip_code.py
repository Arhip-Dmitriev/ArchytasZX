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

"""Qudit phase-flip repetition code on n + 1 qudits: the CX fan then F on every qudit; the encoder
is F then the X spider."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Repetition encoding, then F everywhere."""
    return circuit(["q", "r"], [Init("r"), CX("q", "r"), F("q"), F("r")], registers={"r": n})


def _spec(c):
    """F, then X(1 -> n + 1)."""
    return f"f = F({c.D})\nx = X(1 -> 2, {c.D})\nf -> x\nrepeat {c.N}: x.out1\ninputs f.in0"


SUBJECT = Subject(
    id="phase_flip_code",
    name="Qudit phase-flip code",
    category="code",
    source="nielsen2010quantum",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("encoding", _circuit, _spec, form="boxed"),),
)
