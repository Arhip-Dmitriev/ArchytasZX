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

"""Five-qubit perfect [[5,1,3]] code, stabilizers the cyclic shifts of XZZXI: the standard-form
encoder against the projector form; one X error corrected by syndrome lookup."""

from benchmarks.subjects._stab import code_subject

SUBJECT = code_subject(
    id="five_qubit_code",
    name="Five-qubit perfect code",
    source="laflamme1996perfect",
    generators=["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
    correct=(2, "X"),
)
