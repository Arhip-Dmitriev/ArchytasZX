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

"""Toffoli with 4 controls from 3 Toffolis and 2 clean ancillas (Amy-Maslov-Mosca
benchmark tof_4): the target flips when every control is 1, the ancillas return to |0>."""

from benchmarks.ir import MCX
from benchmarks.subjects._qc import amm_subject, read

_Q = read("tof_4")
_CONTROLS, _TARGET = _Q.inputs[:-1], _Q.inputs[-1]


SUBJECT = amm_subject(
    "tof_4", "4-controlled Toffoli, clean ancillas", None, "", gates=(MCX(_CONTROLS, _TARGET),)
)
