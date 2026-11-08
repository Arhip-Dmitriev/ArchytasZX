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

"""Hypergraph product of two [3,1,3] repetition codes, [[13,1,3]] (the unrotated distance-3
surface code): H_X = (H (x) I, I (x) H^T), H_Z = (I (x) H, H^T (x) I); the standard-form encoder
against the CSS normal form."""

import numpy as np

from benchmarks.subjects._stab import code_subject, css_generators

_H = np.array([[1, 1, 0], [0, 1, 1]])
_HX = np.hstack([np.kron(_H, np.eye(3, dtype=int)), np.kron(np.eye(2, dtype=int), _H.T)]) % 2
_HZ = np.hstack([np.kron(np.eye(3, dtype=int), _H), np.kron(_H.T, np.eye(2, dtype=int))]) % 2

SUBJECT = code_subject(
    id="hypergraph_product_code",
    name="Hypergraph product code [[13,1,3]]",
    source="tillich2014quantum",
    generators=css_generators(_HX.tolist(), _HZ.tolist()),
)
