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

"""Steane [[7,1,3]] code, the CSS code of the [7,4,3] Hamming code: the standard-form encoder
against the CSS normal form; one Y error corrected by syndrome lookup; transversal H after the
encoder against H before it."""

from benchmarks.ir import F
from benchmarks.subjects._stab import Logical, code_subject, pre_unary, wires

GENERATORS = [
    "IIIXXXX",
    "IXXIIXX",
    "XIXIXIX",
    "IIIZZZZ",
    "IZZIIZZ",
    "ZIZIZIZ",
]

SUBJECT = code_subject(
    id="steane_code",
    name="Steane [[7,1,3]] code",
    source="steane1996multiple",
    generators=GENERATORS,
    correct=(4, "Y"),
    logical=[
        Logical(
            "transversal_h",
            lambda code: [F(q) for q in wires(code.n)],
            lambda code: pre_unary("F({D})"),
        )
    ],
)
