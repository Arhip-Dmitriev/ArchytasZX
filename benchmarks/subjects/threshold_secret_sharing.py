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

"""Threshold (2, 3) quantum secret sharing (Cleve-Gottesman-Lo) on qutrits: the secret encoded
into three shares by the [[3,1,2]]_3 code, share 0 lost, shares 1 and 2 decoded by two CX^-1;
the secret comes out on share 2, share 1 maximally mixed."""

from benchmarks.subjects._common import *
from benchmarks.subjects._qudit_codes import css_encoder
from benchmarks.subjects.qutrit_code import CARRIERS, LOG, STABS


def _circuit(n, d):
    """Encode, discard share 0, reconstruct from shares 1 and 2."""
    gates = [
        *css_encoder(3, STABS, LOG, CARRIERS),
        Discard("q0"),
        CX("q1", "q2", -1),
        CX("q2", "q1", -1),
    ]
    return circuit(["q0", "q1", "q2"], gates)


def _spec(c):
    """I/3 on share 1, the identity onto share 2."""
    return (
        f"m = G(0 -> 1, {c.D})\ni = Z(1 -> 1, {c.D})\ninputs i.in0\noutputs m.out0, i.out0\n"
        f"scalar {c.d}^(-1/2)"
    )


SUBJECT = Subject(
    id="threshold_secret_sharing",
    name="Threshold quantum secret sharing",
    category="protocol",
    source="cleve1999share",
    size_parameter=None,
    dimension=3,
    domain=Domain(d_min=3, d_max=3),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
