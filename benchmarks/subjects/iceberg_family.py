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

"""Iceberg code family [[2m, 2m - 2, 2]]: the stabilizers X^(2m) and Z^(2m); the standard-form
encoder against the CSS normal form."""

from benchmarks.subject import Domain
from benchmarks.subjects._stab import code_family


def generators(m):
    """X and Z on all 2m qubits."""
    return ("X" * (2 * m), "Z" * (2 * m))


SUBJECT = code_family(
    id="iceberg_family",
    name="Iceberg code family",
    source="self2024protecting",
    make=generators,
    size_parameter="qubit pairs",
    domain=Domain(n_min=2, d_min=2, d_max=2),
)
