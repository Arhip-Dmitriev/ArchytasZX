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

"""Phase 4 numeric-oracle check: the completion condition for this phase.

Per the spec, every build phase ends with a numeric oracle check, and per the build
plan Phase 4 is done when "the oracle can score any concrete diagram and compare any two
diagrams, exactly by default." This module exercises exactly that sentence against the
GHZ-with-copy worked example shared with tests/test_phase2_oracle.py and
tests/test_phase3_oracle.py: build it with a symbolic dimension, instantiate at several
concrete d, score it, and confirm the contracted tensor is the GHZ vector; then build a
second copy with a scalar-shifted (but otherwise identical) diagram and confirm the
oracle's default (EXACT) comparison correctly tells them apart while UP_TO_GLOBAL_PHASE,
opted into explicitly, correctly does not.
"""

from __future__ import annotations

import numpy as np

from qufzx.algebra.dimension import Dim
from qufzx.algebra.scalar import Scalar
from qufzx.semantics.check import EqualityMode, compare, score

from .helpers import build_ghz_with_copy


class TestScoreAnyConcreteDiagram:
    def test_ghz_with_copy_scores_correctly_at_several_d(self) -> None:
        d = Dim.symbol("d")
        diagram, _a, _b = build_ghz_with_copy(d)
        for d_value in (2, 3, 5):
            result = score(diagram, {"d": d_value})
            expected = np.zeros((d_value,) * 3, dtype=complex)
            for k in range(d_value):
                expected[k, k, k] = 1
            np.testing.assert_allclose(result.tensor, expected)


class TestCompareAnyTwoDiagramsExactlyByDefault:
    def test_identical_ghz_diagrams_match_by_default(self) -> None:
        d = Dim.symbol("d")
        diagram_a, _a1, _b1 = build_ghz_with_copy(d)
        diagram_b, _a2, _b2 = build_ghz_with_copy(d)
        for d_value in (2, 3, 5):
            result = compare(diagram_a, diagram_b, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched

    def test_unit_modulus_scalar_shift_fails_exact_passes_global_phase(self) -> None:
        d = Dim.concrete(2)
        diagram_a, _a1, _b1 = build_ghz_with_copy(d)
        diagram_b, _a2, _b2 = build_ghz_with_copy(d)
        diagram_b.multiply_scalar(Scalar.omega(Dim.concrete(4), 1))

        default_result = compare(diagram_a, diagram_b, {})
        assert default_result.mode is EqualityMode.EXACT
        assert not default_result.matched

        opted_in_result = compare(diagram_a, diagram_b, {}, mode=EqualityMode.UP_TO_GLOBAL_PHASE)
        assert opted_in_result.matched
