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

"""Toffoli with 5 controls from Toffolis on 3 dirty (borrowed) ancillas, Barenco et al.
(Amy-Maslov-Mosca benchmark barenco_tof_5): the target flips when every control is 1, the
ancillas end in their input values."""

from benchmarks.ir import MCX
from benchmarks.subjects._qc import amm_subject, read

_Q = read("barenco_tof_5")
_CONTROLS, _TARGET = _Q.wires[:5], _Q.wires[-1]


SUBJECT = amm_subject(
    "barenco_tof_5",
    "5-controlled Toffoli, dirty ancillas",
    None,
    "",
    gates=(MCX(_CONTROLS, _TARGET),),
)
