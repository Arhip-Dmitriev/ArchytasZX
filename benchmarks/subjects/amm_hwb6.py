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

"""Hidden weighted bit on 6 bits (Amy-Maslov-Mosca benchmark hwb6): the input abcdef rotated right
by its Hamming weight, one clean ancilla."""

from benchmarks.subjects._qc import amm_subject


def _fn(v):
    """abcdef rotated right by its weight."""
    x = [v[c] for c in "abcdef"]
    k = sum(x) % 6
    y = x[-k:] + x[:-k] if k else x
    return {**v, **dict(zip("abcdef", y)), "w0": 0}


SUBJECT = amm_subject("hwb6", "Hidden weighted bit", _fn, "")
