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

"""The [[4,2,2]] error-detecting code, stabilizers XXXX and ZZZZ: the standard-form encoder
against the CSS normal form."""

from benchmarks.subjects._stab import code_subject

SUBJECT = code_subject(
    id="four_qubit_code",
    name="[[4,2,2]] error-detecting code",
    source="vaidman1996error",
    generators=["XXXX", "ZZZZ"],
)
