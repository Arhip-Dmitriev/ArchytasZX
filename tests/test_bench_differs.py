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

"""F pairs of benchmarks.breaks against benchmarks.differs.equal_at, by dense tensors of both
sides at small points."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

TOL = 1e-8


@functools.cache
def _insts() -> dict[str, Any]:
    from benchmarks import subject

    return {i.id: i for i in subject.instances(list(subject.catalogue()))}


def _record(eq: str, kind: str, label: str | None = None, **over: Any) -> dict[str, Any]:
    """The F record of today's first variant of ``kind`` on ``eq`` (with label ``label``), its
    params updated by ``over``, without a differs_at table."""
    from benchmarks import breaks

    inst = _insts()[eq]
    for v in breaks.variants(kind, inst, {"false_set": {}}):
        if label is None or v.label == label:
            return {
                "equation": eq,
                "kind": kind,
                "label": v.label,
                "params": {**v.params, **over},
                "witness": {},
                "equal_where": v.equal_where,
            }
    raise LookupError(f"no {kind} variant {label!r} on {eq}")


def _dense(rec: dict[str, Any], n: int | None, d: int) -> str:
    """equal, differ or interface for the F pair of ``rec`` at (n, d), both sides doubled when
    either is mixed."""
    from archytaszx.diagram.graph import Direction
    from archytaszx.diagram.mixed import double
    from archytaszx.semantics import check
    from archytaszx.semantics.contract_numeric import contract
    from benchmarks import breaks, stages, subject

    fi = breaks.false_instance(rec, _insts())
    left, right = stages.diagrams(subject.script(fi, n, d).text)
    mixed = not (left.is_pure() and right.is_pure())
    sides = []
    for raw in (left, right):
        x = check.instantiate(raw, {})
        x = double(x) if mixed else x
        dims = []
        for refs in (x.boundary_outputs, x.boundary_inputs):
            for ref in refs:
                nd = x.nodes[ref.node_id]
                legs = nd.outputs if ref.direction is Direction.OUTPUT else nd.inputs
                dims.append(int(x.resolve_dim(legs[ref.index].dim).to_int()))
        sides.append((x, dims))
    (lx, ld), (rx, rd) = sides
    if [v for v in ld if v != 1] != [v for v in rd if v != 1]:
        return "interface"
    assert math.prod(ld) <= 1 << 14
    a = np.asarray(contract(lx).tensor).reshape(-1)
    b = np.asarray(contract(rx).tensor).reshape(-1)
    return "differ" if float(np.max(np.abs(a - b), initial=0.0)) > TOL else "equal"


def _disagreements(
    rec: dict[str, Any], points: Sequence[tuple[int | None, int]]
) -> list[tuple[Any, ...]]:
    """The points where differs.equal_at and the dense check disagree."""
    from benchmarks import differs

    out = []
    for n, d in points:
        claim = differs.equal_at(rec, n, d)
        dense = _dense(rec, n, d)
        if (claim is True and dense != "equal") or (claim is False and dense == "equal"):
            out.append((n, d, claim, dense))
    return out


def test_untabled_record_refused() -> None:
    from benchmarks import differs

    rec = _record("bridge_gate/main", "swapped_control_target")
    with pytest.raises(ValueError, match="differs_at"):
        differs.equal_at(rec, None, 2)
    assert differs.equal_at(None, 3, 2) is False


def test_tabled_fills_tables() -> None:
    from benchmarks import breaks

    recs = [{"equation": "x", "kind": None}, {**_record("deutsch/f11", "leg_order")}]
    out = breaks.tabled(recs, 10.0, 1)
    assert out[0] == recs[0] and "differs_at" in out[1]
    assert breaks.tabled(out, 10.0, 1) is out


def test_annotated_at_d1_and_past_table() -> None:
    from benchmarks import breaks

    for eq, kind in (("bridge_gate/main", "swapped_control_target"), ("deutsch/f11", "leg_order")):
        rec = breaks.annotate(_record(eq, kind), _insts(), 10.0)
        assert _disagreements(rec, [(None, 1), (None, 2), (None, 5)]) == []


def test_off_by_one_boxed_chain_and_tree() -> None:
    for eq, pts in (
        ("bv_secret_pattern/main", [(1, 2), (2, 2), (1, 3)]),
        ("w_state/main", [(1, 2)]),
    ):
        rec = {**_record(eq, "off_by_one_n"), "differs_at": []}
        assert _disagreements(rec, pts) == []
        assert {_dense(rec, n, d) for n, d in pts} == {"interface"}


def test_beyond_n_unrolled_small_n() -> None:
    from benchmarks import breaks

    inst = _insts()["repetition_error_syndrome/main"]
    n0 = breaks._ns(inst, 0, 1)[0]
    assert n0 is not None
    for label in ("X on output -1 beyond N", "X on output -2 beyond N"):
        rec = {**_record(inst.id, "beyond_N", label, N=n0), "differs_at": []}
        pts = [(n, d) for d in (1, 2) for n in (n0, n0 + 1, n0 + 2)]
        assert _disagreements(rec, pts) == []


def test_stale_positional_leg_order_is_refused() -> None:
    from benchmarks import breaks, outcomes, subject

    inst = _insts()["qram/main"]
    rec = {
        "equation": "qram/main",
        "kind": "leg_order",
        "label": "swap last two outputs",
        "params": {"i": 2, "j": 3},
        "witness": {},
        "equal_where": {},
        "differs_at": [],
    }
    assert breaks.mismatch(rec, inst) is not None
    with pytest.raises(breaks.NotApplicable):
        subject.circuit_at(breaks.false_instance(rec, _insts()), 1, 2)
    run = {"tool": "qcec", "outcome": "ok", "verdict": "equal", "n": 3, "d": 2, "stages": {}}
    out = outcomes.classify(run, false=rec, stale=True)
    assert out["outcome"] == "invalid_point" and out["cause"] == "false_record_mismatch"
