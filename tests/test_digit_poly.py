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

"""Digit polynomials of phase tables (digit_poly): fits, refusals, values and Lean text."""

from __future__ import annotations

from archytaszx.semantics.digit_poly import digit_poly


def test_product_of_two_digits() -> None:
    b = 40
    nums = [(k // b) * (k % b) for k in range(b * b)]
    dp = digit_poly(nums, b * b, b * b)
    assert dp is not None
    assert dp.bases == (40, 40) and dp.terms == ((1, (1, 1)),) and dp.den == 1600
    assert all(dp.value(k) == nums[k] for k in range(b * b))
    assert dp.lean() == "((1 * ((k / 40 : ℕ) : ℤ) * ((k % 40 : ℕ) : ℤ)) % 1600)"


def test_one_digit_quadratic() -> None:
    nums = [(3 * k * k + 7 * k) % 999 for k in range(1000)]
    dp = digit_poly(nums, 999, 1000)
    assert dp is not None and dp.bases == (1000,)
    assert all(dp.value(k) == nums[k] for k in range(1000))


def test_halves_need_a_larger_denominator() -> None:
    nums = [(k * (k - 1) // 2) % 7 for k in range(200)]
    dp = digit_poly(nums, 7, 200)
    assert dp is not None and dp.den == 14
    assert all(dp.value(k) == 2 * nums[k] for k in range(200))


def test_refusals() -> None:
    assert digit_poly([(k % 7) * (k % 5) % 64 for k in range(256)], 64, 256) is None
    assert digit_poly([-1, 0, 1], 4, 3) is None
    assert digit_poly([0, 1, 4, 3], 4, 4) is None


if __name__ == "__main__":
    test_product_of_two_digits()
    test_one_digit_quadratic()
    test_halves_need_a_larger_denominator()
    test_refusals()
    print("ok")
