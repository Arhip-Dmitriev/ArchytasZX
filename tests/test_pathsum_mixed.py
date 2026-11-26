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

"""Mixed path sums on dimensions 2, 3, 6: entries against the dense contraction."""

from __future__ import annotations

from itertools import product

import pytest

from archytaszx.algebra.pathsum import entry_enumerated, reduce, zero_indicators
from archytaszx.diagram.graph import Diagram
from archytaszx.repl.commands import Session, run
from archytaszx.semantics.contract_numeric import contract
from archytaszx.semantics.pathsum_mixed import is_mixed, mixed_sum

_SCRIPT = """load
    a = X(0 -> 1, literal 3)
    fa = F(literal 3)
    za = Z(1 -> 2, literal 3)
    q = X(0 -> 1, literal 2)
    fq = F(literal 2)
    b = B(literal 3, literal 2)
    z = Z(1 -> 1, literal 6) phase{{{p1}}}
    s = S(literal 3, literal 2)
    b2 = B(literal 3, literal 2)
    z2 = Z(1 -> 1, literal 6) phase{{{p2}}}
    s2 = S(literal 3, literal 2)
    f3 = F(literal 3)
    e = Z(1 -> 0, literal 3)
    e2 = X(1 -> 0, literal 3)
    wire a.out0 -- fa.in0, fa.out0 -- za.in0, za.out0 -- b.in0, q.out0 -- fq.in0
    wire fq.out0 -- b.in1, b.out0 -- z.in0, z.out0 -- s.in0, s.out0 -- b2.in0
    wire s.out1 -- b2.in1, b2.out0 -- z2.in0, z2.out0 -- s2.in0, s2.out0 -- f3.in0
    wire f3.out0 -- e.in0, za.out1 -- e2.in0
    outputs s2.out1
"""


def _diagram(p1: str, p2: str) -> Diagram:
    session = Session()
    out = run(session, _SCRIPT.format(p1=p1, p2=p2))
    assert out.ok
    assert session.current is not None
    return session.current


@pytest.mark.parametrize(
    ("p1", "p2"),
    [("3: 1/2", "5: 1/2"), ("1: 1/4, 3: 3/8", "2: 5/8, 5: 1/8"), ("4: 7/8", "3: 1/2, 5: 1/2")],
)
def test_mixed_entries_match_contraction(p1: str, p2: str) -> None:
    dg = _diagram(p1, p2)
    assert is_mixed(dg)
    want = contract(dg).tensor
    ps = mixed_sum(dg)
    with zero_indicators():
        reduced = reduce(ps)
    for idx in product(range(2), repeat=len(ps.boundary)):
        values = dict(zip(ps.boundary, idx, strict=True))
        for sum_ in (ps, reduced):
            got = entry_enumerated(sum_, values, 30)
            assert got is not None
            assert abs(got.to_complex() - complex(want[idx])) < 1e-9


def test_mixed_label_ranges_past_their_offsets_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from archytaszx.algebra.pathsum import PathSumUnsupported
    from archytaszx.semantics import lean_pathsum as P

    dg = _diagram("3: 1/2", "5: 1/2")
    side = P._mixed_net_side(dg, "L", P.MIX_LABEL0, 24, [])
    assert len(side.ents) == 4
    monkeypatch.setattr(P, "MIX_ENT", P.MIX_FRESH + 1)
    with pytest.raises(PathSumUnsupported, match="X spiders"):
        P._mixed_net_side(dg, "L", P.MIX_LABEL0, 24, [])
    monkeypatch.setattr(P, "MIX_ENT", 1 << 13)
    monkeypatch.setattr(P, "MIX_SPAN", P.MIX_ENT + 3)
    with pytest.raises(PathSumUnsupported, match="indicator entries"):
        P._mixed_net_side(dg, "L", P.MIX_LABEL0, 24, [])
