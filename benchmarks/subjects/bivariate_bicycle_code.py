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

"""Bivariate bicycle code [[72,12,6]] of Bravyi et al.: l = m = 6, A = x^3 + y + y^2,
B = y^3 + x + x^2 over the cyclic shifts x, y of Z_6 x Z_6; H_X = [A | B], H_Z = [B^T | A^T]."""

from benchmarks.subjects._stab import code_subject, css_generators, independent

_L = _M = 6


def _mono(a, b):
    """The permutation matrix of x^a y^b as a function (i, j) -> 0/1 on Z_6 x Z_6 indices."""
    return lambda r, c: int((c // _M - r // _M) % _L == a % _L and (c % _M - r % _M) % _M == b % _M)


def _poly(terms):
    """The 36 x 36 0/1 matrix of a sum of monomials x^a y^b."""
    n = _L * _M
    ms = [_mono(a, b) for a, b in terms]
    return [[sum(f(r, c) for f in ms) % 2 for c in range(n)] for r in range(n)]


_A = _poly([(3, 0), (0, 1), (0, 2)])
_B = _poly([(0, 3), (1, 0), (2, 0)])
_T = lambda m: [list(r) for r in zip(*m)]
_HX = [ra + rb for ra, rb in zip(_A, _B)]
_HZ = [ra + rb for ra, rb in zip(_T(_B), _T(_A))]

SUBJECT = code_subject(
    id="bivariate_bicycle_code",
    name="Bivariate bicycle code",
    source="bravyi2024high",
    generators=independent(css_generators(_HX, _HZ)),
)
