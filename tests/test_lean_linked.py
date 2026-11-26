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

"""Tests for the linked-diagram reading of lean_linked."""

from __future__ import annotations

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.semantics.lean_linked import LBox, LinkedUnsupported, _Tables, linked


def test_an_empty_held_box_is_unsupported_not_an_index_error() -> None:
    d = Diagram()
    node = d.add_node(Z_SPIDER, input_dims=[Dim(2)], output_dims=[Dim(2)])
    out, inp = PortRef(node, Direction.OUTPUT, 0), PortRef(node, Direction.INPUT, 0)
    d.set_boundary_outputs([out])
    d.set_boundary_inputs([inp])
    d.add_bang_box(Mult.concrete(2), node_scope=frozenset([node]), links=((out, inp),))
    held = {(int(node), True, 0): LBox(9, 0, None, ())}
    with pytest.raises(LinkedUnsupported):
        linked(d, _Tables(), ghost=7, held=held)
