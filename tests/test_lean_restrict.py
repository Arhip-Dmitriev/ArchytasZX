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

"""Corollaries of restricted Lean exports."""

from __future__ import annotations

from archytaszx.semantics.lean_restrict import corollary
from archytaszx.semantics.restrict import parse_restriction

_OPAQUE = """namespace ArchytasZX.Export.C1

theorem main (d : ℕ) (h_d : 1 ≤ d) : Equal (E d) 1 N0 1 N1 := sorry

theorem main_diagrams :
    ∀ (d : ℕ) (h_d : 1 ≤ d), DEqual (E d) 1 D0 1 D1 := sorry

end ArchytasZX.Export.C1

namespace ArchytasZX.Opaque

theorem main : ∀ (n : Nat), (d0 n) = (d2 n) :=
  fun n => (fam11 n)

end ArchytasZX.Opaque
"""


def test_the_corollary_states_the_last_final_theorem() -> None:
    clauses = parse_restriction("n = 3").clauses
    out = corollary(_OPAQUE, clauses)
    tail = out.split("namespace ArchytasZX.Opaque")[1]
    assert "theorem main_at :\n    (d0 3) = (d2 3) :=\n  main 3" in tail
    assert "main_diagrams" not in tail
