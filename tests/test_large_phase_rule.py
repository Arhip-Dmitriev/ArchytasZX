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

"""A phase rule over a large concrete dimension parses as a rule, unexpanded."""

from __future__ import annotations

import time

import pytest

from archytaszx.algebra.phase import PhaseVector
from archytaszx.diagram.graph import Diagram
from archytaszx.repl import parser
from archytaszx.repl.parser import ParseDomainError, parse_dsl

BIG = 1168**3


def _shift(dim: str) -> str:
    return f"""s = X(1 -> 1, {dim}) phase{{k: -2*k/{dim.split()[-1]}}}
inputs s.in0
outputs s.out0"""


def _phase(diagram: Diagram) -> PhaseVector:
    (phase,) = (n.phase for n in diagram.nodes.values() if n.phase is not None)
    return phase


def test_a_rule_above_the_bound_is_kept_as_a_rule() -> None:
    start = time.monotonic()
    diagram = parse_dsl(_shift(f"literal {BIG}"))
    assert time.monotonic() - start < 10
    phase = _phase(diagram)
    assert phase.rule is not None
    assert not phase.entries()


def test_a_rule_at_the_bound_is_expanded() -> None:
    diagram = parse_dsl(_shift("literal 64"))
    phase = _phase(diagram)
    assert phase.rule is None
    assert len(phase.entries()) == 62
    assert parser._EXPAND_MAX >= 64


def test_an_index_past_a_parameter_dimension_is_still_refused() -> None:
    with pytest.raises(ParseDomainError):
        parse_dsl("dim d = 3\ns = Z(1 -> 1, d) phase{5: 1/2}\ninputs s.in0\noutputs s.out0")
