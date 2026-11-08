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

"""Divisibility by five (Amy-Maslov-Mosca benchmark mod5_4): a ^= [bcde = 0 mod 5], b the most
significant input bit."""

from benchmarks.subjects._qc import amm_subject, number


def _fn(v):
    """a set by the divisibility of the 4-bit input bcde by 5."""
    return {**v, "a": int(number([v[c] for c in "bcde"]) % 5 == 0)}


SUBJECT = amm_subject("mod5_4", "Divisibility by five", _fn, "")
