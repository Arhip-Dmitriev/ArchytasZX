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

"""Gottesman [[8,3,3]] code, the first of the non-CSS quantum Hamming family: the standard-form
encoder against the projector form; one Y error corrected by syndrome lookup."""

from benchmarks.subjects._stab import code_subject

SUBJECT = code_subject(
    id="gottesman_code",
    name="Gottesman [[8,3,3]] code",
    source="gottesman1996class",
    generators=["XXXXXXXX", "ZZZZZZZZ", "IXIXYZYZ", "IXZYIXZY", "IYXZXZIY"],
    correct=(5, "Y"),
)
