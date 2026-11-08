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

"""Vedral-Barenco-Ekert 3-bit ripple-carry adder (Amy-Maslov-Mosca benchmark vbe_adder_3): wires
(c_i, a_i, b_i) = (w(3i), w(3i+1), w(3i+2)) and the overflow w9; b <- a + b + c_0 mod 8, w9 ^= the
carry out, with the internal carries c_1, c_2 initialised to |0>."""

from benchmarks.subjects._qc import amm_subject


def _fn(v):
    """The sum into b, the carry out into w9."""
    a = sum(v[f"w{3 * i + 1}"] << i for i in range(3))
    b = sum(v[f"w{3 * i + 2}"] << i for i in range(3))
    s = a + b + v["w0"]
    out = {**v, "w3": 0, "w6": 0, "w9": v["w9"] ^ (s >> 3)}
    out.update({f"w{3 * i + 2}": (s >> i) & 1 for i in range(3)})
    return out


SUBJECT = amm_subject("vbe_adder_3", "VBE ripple-carry adder", _fn, "", clean=("w3", "w6"))
