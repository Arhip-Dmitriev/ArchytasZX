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

"""_Refuter.hinted: each hint entry range-checked against the label it lands on."""

from __future__ import annotations

from types import SimpleNamespace

from archytaszx.semantics.lean_refute import _Refuter


def _stub(hint: tuple[int, ...], dims: dict[int, int]) -> SimpleNamespace:
    return SimpleNamespace(hint=hint, labdim=dims.__getitem__, bundled_hint=lambda *_: None)


def test_a_hint_is_checked_against_its_own_label() -> None:
    stub = _stub((4, 1), {0: 5, 1: 2})
    assert _Refuter.hinted(stub, [1, 0]) == [(1, 4)]  # type: ignore[arg-type]
    assert _Refuter.hinted(_stub((1, 4), {0: 5, 1: 2}), [1, 0]) == []  # type: ignore[arg-type]
