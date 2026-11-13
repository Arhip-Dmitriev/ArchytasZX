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

"""Boundary order of a chain box with inputs: the copies' carry ends listed before the chain's
tail give the one-place shift of the unrolled ladder."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics.check import compare
from archytaszx.semantics.decide import EqualityVerdict, decide_equal

if TYPE_CHECKING:
    from archytaszx.diagram.graph import NodeId
    from archytaszx.semantics.prove import InductionSide

_SWAPS = (
    "dim d\ncount n\ng0 = Z(1 -> 1, d)\ng1 = Z(1 -> 1, d)\nrepeat n as C0: g0, g1\n"
    "chain C0: g1.out0 -> g1.in0\ninputs g1.in0, g0.in0\noutputs {outs}"
)
"""A SWAP chain: the head carried through every copy, each copy's register wire straight."""

SHIFT = _SWAPS.format(outs="g0.out0, g1.out0")
"""Copies first, tail last: input i + 1 to output i, input 0 to the last output."""

CARRY_FIRST = _SWAPS.format(outs="g1.out0, g0.out0")
"""Tail first, copies after: every input to its own output."""


def _wires(outs: str) -> str:
    return (
        "dim d\ncount n\nh = Z(1 -> 1, d)\nrepeat n as V: w = Z(1 -> 1, d)\n"
        f"inputs h.in0, w.in0\noutputs {outs}"
    )


CYCLIC = _wires("w.out0, h.out0")
"""The cyclic shift down by one place, boxed."""

IDENTITY = _wires("h.out0, w.out0")
"""The identity on n + 1 wires, boxed."""


def _unrolled_shift(n: int) -> str:
    lines = ["dim d"] + [f"w{i} = Z(1 -> 1, d)" for i in range(n + 1)]
    lines.append("inputs " + ", ".join(f"w{i}.in0" for i in range(n + 1)))
    lines.append("outputs " + ", ".join(f"w{(i + 1) % (n + 1)}.out0" for i in range(n + 1)))
    return "\n".join(lines)


def test_copies_first_is_the_cyclic_shift() -> None:
    assert decide_equal(parse_dsl(SHIFT), parse_dsl(CYCLIC)).verdict is EqualityVerdict.EQUAL
    assert decide_equal(parse_dsl(CARRY_FIRST), parse_dsl(IDENTITY)).verdict in (
        EqualityVerdict.EQUAL,
        EqualityVerdict.EQUAL_UNCERTIFIED,
    )
    assert decide_equal(parse_dsl(SHIFT), parse_dsl(IDENTITY)).verdict is EqualityVerdict.UNEQUAL


def _reversed() -> tuple[str, str]:
    from .test_lean_chain import CIRCUIT, SPEC

    return (
        CIRCUIT.replace("chain C0: g5.out0 -> g4.in0", "chain C0: g4.in0 -> g5.out0"),
        SPEC.replace("chain V: w.out0 -> v.in0", "chain V: v.in0 -> w.out0"),
    )


def test_a_reversed_chain_peels_its_first_copy() -> None:
    """Links from an input to an output run copy m + 1 into copy m; the successor peeled at
    copy 0, instantiated at k, is the chain at k + 1."""
    from archytaszx.diagram.bangbox import instantiate_symbol, peels_first
    from archytaszx.rewrite.normal_form import comparison_view, views_isomorphic
    from archytaszx.semantics.prove import induction_successor

    for text in _reversed():
        d = parse_dsl(text)
        assert all(peels_first(box) for box in d.bang_boxes.values())
        succ = induction_successor(d, "n", "k")
        assert succ is not None
        for k in (0, 1, 2):
            at = instantiate_symbol(succ, "k", k)
            whole = instantiate_symbol(d, "n", k + 1)
            assert views_isomorphic(comparison_view(at), comparison_view(whole))


def _reversed_indexed() -> tuple[str, str]:
    from .test_lean_chain import INDEXED

    return (
        INDEXED[0].replace("chain C0: g5.out0 -> g4.in0", "chain C0: g4.in0 -> g5.out0"),
        INDEXED[1].replace("chain V: w.out0 -> v.in0", "chain V: v.in0 -> w.out0"),
    )


def _unrolled_reversed_spec(n: int) -> str:
    """The reversed indexed spec at n: copy m's w with phase m/4, copy m + 1's w into copy m."""
    lines = ["dim d = 2", "h = Z(0 -> 1, d)"]
    for m in range(n):
        lines += [
            f"v{m} = Z(1 -> 2, d)",
            f"f{m} = F(d)",
            f"w{m} = Z(1 -> 1, d) phase {{1: {m}/4}}",
            f"wire v{m}.out1 -- f{m}.in0, f{m}.out0 -- w{m}.in0",
        ]
        if m < n - 1:
            lines.append(f"wire v{m}.in0 -- w{m + 1}.out0")
    if n:
        lines.append(f"wire h.out0 -- v{n - 1}.in0")
    outs = [f"v{m}.out0" for m in range(n)] + (["w0.out0"] if n else ["h.out0"])
    lines += ["outputs " + ", ".join(outs), "scalar 1/sqrt(d)"]
    return "\n".join(lines)


def test_a_reversed_indexed_chain_shifts_its_residual_index() -> None:
    """The first copy reads index 0 and the residual index + 1; at k = 0 .. 3 the peeled
    successor is the chain at k + 1, and the chain matches its unrolled form."""
    from archytaszx.diagram.bangbox import instantiate_symbol
    from archytaszx.rewrite.normal_form import comparison_view, views_isomorphic
    from archytaszx.semantics.prove import induction_successor

    circuit, spec = _reversed_indexed()
    for text in (circuit, spec, circuit + "\ncopy scalar C0: 2^i"):
        d = parse_dsl(text)
        succ = induction_successor(d, "n", "k")
        assert succ is not None
        for k in range(4):
            at = instantiate_symbol(succ, "k", k)
            whole = instantiate_symbol(d, "n", k + 1)
            assert views_isomorphic(comparison_view(at), comparison_view(whole))
    for n in range(4):
        assert compare(parse_dsl(spec), parse_dsl(_unrolled_reversed_spec(n)), {"n": n}).matched
    broken = spec.replace("phase {1: i/4}", "phase {1: (i + 1)/4}")
    verdict = decide_equal(parse_dsl(circuit), parse_dsl(broken)).verdict
    assert verdict is EqualityVerdict.UNEQUAL


def test_a_reversed_indexed_chain_is_proved_with_an_index_offset() -> None:
    """Equal with a kernel induction over ``index + s``; its pieces refused without the offset
    or with an offset that is not fresh; phase, copy scalar and order variants not proved."""
    from archytaszx.semantics import kernel as K

    from .test_lean_chain import INDEXED

    circuit, spec = _reversed_indexed()
    theorems: list[K.Theorem] = []
    for a, b in (
        (circuit, spec),
        (circuit + "\ncopy scalar C0: 2^i", spec + "\ncopy scalar V: 2^i"),
    ):
        decision = decide_equal(parse_dsl(a), parse_dsl(b))
        assert decision.verdict is EqualityVerdict.EQUAL
        assert isinstance(decision.induction_proof, K.Theorem)
        theorems.append(decision.induction_proof)
        assert theorems[-1].inference.kind == "induction_offset"
    theorem = theorems[0]
    data = theorem.inference.data
    base, expose, step = theorem.inference.premises
    for offset in (None, "i"):
        with pytest.raises(K.KernelError):
            K.induction(
                theorem.lhs,
                theorem.rhs,
                "n",
                str(data["step"]),
                base,
                cast("InductionSide", data["side"]),
                expose,
                cast("frozenset[NodeId]", data["region"]),
                step,
                offset=offset,
            )
    with pytest.raises(K.KernelError):
        K.lean(theorem, semantic=False)
    broken = (
        (circuit, spec.replace("phase {1: i/4}", "phase {1: (i + 1)/4}")),
        (circuit, spec.replace("phase {1: i/4}", "phase {1: i/2}")),
        (circuit + "\ncopy scalar C0: 2^i", spec + "\ncopy scalar V: 2^(i + 1)"),
        (circuit, INDEXED[1]),
    )
    for a, b in broken:
        verdict = decide_equal(parse_dsl(a), parse_dsl(b)).verdict
        assert verdict is not EqualityVerdict.EQUAL


def test_a_reversed_chain_is_proved_by_induction() -> None:
    circuit, spec = _reversed()
    decision = decide_equal(parse_dsl(circuit), parse_dsl(spec))
    assert decision.verdict is EqualityVerdict.EQUAL and decision.induction_proof is not None
    broken = spec.replace("f = F(d)", "f = F(d), e = F(d)").replace(
        "f.out0 -- w.in0", "f.out0 -- e.in0, e.out0 -- w.in0"
    )
    assert decide_equal(parse_dsl(circuit), parse_dsl(broken)).verdict is EqualityVerdict.UNEQUAL


@pytest.mark.parametrize("n", [2, 3])
def test_the_boxed_shift_is_isomorphic_to_the_unrolled_one_after_fusion(n: int) -> None:
    """The SWAP chain's identity spiders keep it from the unrolled shift's isomorphism class
    until fusion and identity removal reduce both."""
    from archytaszx.rewrite.normal_form import comparison_view, fused_isomorphic, views_isomorphic
    from archytaszx.semantics.check import instantiate

    boxed = instantiate(parse_dsl(SHIFT.replace("dim d", "dim d = 2")), {"n": n})
    unrolled = instantiate(parse_dsl(_unrolled_shift(n).replace("dim d", "dim d = 2")), {})
    assert not views_isomorphic(comparison_view(boxed), comparison_view(unrolled))
    assert fused_isomorphic(boxed, unrolled)
    carry_first = instantiate(parse_dsl(CARRY_FIRST.replace("dim d", "dim d = 2")), {"n": n})
    assert not fused_isomorphic(carry_first, unrolled)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_copies_first_matches_the_unrolled_shift(n: int) -> None:
    unrolled = parse_dsl(_unrolled_shift(n))
    assert compare(parse_dsl(SHIFT), unrolled, {"n": n, "d": 2}).matched
    assert not compare(parse_dsl(CARRY_FIRST), unrolled, {"n": n, "d": 2}).matched


_STABILIZER_SPEC = """dim d
count n
h = Z(0 -> 1, d)
repeat n as V: v = Z(1 -> 2, d), f = F(d), w = Z(1 -> 1, d)
t = Z(1 -> 2, d)
g = F(d)
a = Z(0 -> 2, d)
wire h.out0 -- v.in0, v.out1 -- f.in0, f.out0 -- w.in0, w.out0 -- t.in0, t.out1 -- g.in0
wire g.out0 -- a.out1
chain V: v.in0 -> w.out0
outputs a.out0, t.out0, v.out0
scalar d^(-1/2)"""
"""cluster_stabilizer's spec: the reversed path chain with vertex 0 joined to the tail."""

_STABILIZER_CIRCUIT = """dim d
count n
g0 = X(0 -> 1, d); g1 = F(d); g2 = X(0 -> 1, d); g3 = F(d); g4 = Z(1 -> 2, d)
g5 = Z(2 -> 1, d); g6 = F(d); g7 = X(0 -> 1, d); g8 = F(d); g9 = Z(1 -> 2, d)
g10 = Z(2 -> 1, d); g11 = F(d); g12 = X(1 -> 1, d) phase{k: -k/d}; g13 = Z(1 -> 1, d) phase{k: k/d}
wire g0.out0 -- g1.in0, g2.out0 -- g3.in0, g3.out0 -- g4.in0, g1.out0 -- g5.in0
wire g4.out1 -- g6.in0, g6.out0 -- g5.in1, g7.out0 -- g8.in0, g8.out0 -- g9.in0
wire g4.out0 -- g10.in0, g9.out1 -- g11.in0, g11.out0 -- g10.in1, g9.out0 -- g12.in0
wire g10.out0 -- g13.in0
repeat n as C0: g2, g3, g4, g5, g6
chain C0: g5.in0 -> g4.out0
outputs g12.out0, g13.out0, g5.out0
scalar 1/sqrt(d)"""
"""cluster_stabilizer's translated circuit: the CZ path from the far end, X and Z on the tail."""


def test_chains_with_head_and_tail_are_proved_with_their_ends_cut() -> None:
    """Both chain-end wires cut on each side: the open pair by chain-window induction, the
    context and closed pairs by normal form; a phase on either side is not proved."""
    decision = decide_equal(parse_dsl(_STABILIZER_CIRCUIT), parse_dsl(_STABILIZER_SPEC))
    assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
    assert "chain ends (both) cut on both sides" in decision.reason
    assert decision.induction_proof is not None
    broken = (
        (
            _STABILIZER_CIRCUIT,
            _STABILIZER_SPEC.replace("w = Z(1 -> 1, d)", "w = Z(1 -> 1, d) phase{1: 1/3}"),
        ),
        (
            _STABILIZER_CIRCUIT.replace("g9 = Z(1 -> 2, d)", "g9 = Z(1 -> 2, d) phase{1: 1/3}"),
            _STABILIZER_SPEC,
        ),
    )
    for left, right in broken:
        verdict = decide_equal(parse_dsl(left), parse_dsl(right), max_samples=0).verdict
        assert verdict is not EqualityVerdict.EQUAL


_SYNDROME_SPEC = """dim d
count n
z = Z(1 -> 2, d)
repeat n as V: e = Z(1 -> 1, d), a = X(0 -> 1, d)
wire z.out0 -- e.in0
outputs e.out0, a.out0, z.out1
scalar d^(-n/2)"""
"""repetition_code/syndrome's spec: the copy spider fanned into the box beside zero ancillas."""

_SYNDROME_CIRCUIT = """dim d
count n
g0 = X(0 -> 1, d); g1 = Z(1 -> 2, d); g2 = X(2 -> 1, d); g3 = X(0 -> 1, d); g4 = Z(1 -> 2, d)
g5 = X(2 -> 1, d); g6 = Z(1 -> 2, d); g7 = X(2 -> 1, d); g8 = F(d); g9 = F(d)
wire g0.out0 -- g2.in0, g1.out1 -- g2.in1, g1.out0 -- g4.in0, g3.out0 -- g5.in0
wire g4.out1 -- g5.in1, g2.out0 -- g6.in0, g5.out0 -- g7.in0, g6.out1 -- g8.in0
wire g8.out0 -- g9.in0, g9.out0 -- g7.in1
repeat n as C0: g0, g2, g3, g4, g5, g6, g7, g8, g9
chain C0: g6.out0 -> g4.in0
inputs g1.in0
outputs g4.out0, g7.out0, g6.out0
scalar d^(n/2)"""
"""repetition_code/syndrome's translated circuit: the CX fan from q, then the ancilla chain."""


def test_a_fan_spider_is_split_off_the_peeled_copy() -> None:
    """The peeled copy's leg of the fan spider is split off with its boundary slot, so the
    residual is the hypothesis; a phase on the fan spider or on a copy is not proved."""
    decision = decide_equal(parse_dsl(_SYNDROME_CIRCUIT), parse_dsl(_SYNDROME_SPEC))
    assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
    assert decision.induction_proof is not None
    broken = (
        (
            _SYNDROME_CIRCUIT,
            _SYNDROME_SPEC.replace("e = Z(1 -> 1, d)", "e = Z(1 -> 1, d) phase{1: 1/3}"),
        ),
        (
            _SYNDROME_CIRCUIT.replace("g1 = Z(1 -> 2, d)", "g1 = Z(1 -> 2, d) phase{1: 1/3}"),
            _SYNDROME_SPEC,
        ),
    )
    for left, right in broken:
        verdict = decide_equal(parse_dsl(left), parse_dsl(right), max_samples=0).verdict
        assert verdict is not EqualityVerdict.EQUAL


_FLIPS = """count n
a = X(0 -> 1, literal 2)
repeat n as V: x = X(1 -> 1, literal 2) phase{1: 1/2}
b = X(1 -> 0, literal 2)
wire a.out0 -- x.in0, x.out0 -- b.in0
chain V: x.out0 -> x.in0"""
"""<0| X^n |0> by a chain of n X(pi) spiders."""


def _parity(scalar: str) -> str:
    return (
        f"count n\na = X(0 -> 1, literal 2)\nb = X(1 -> 0, literal 2)\nwire a.out0 -- b.in0\n"
        f"scalar {scalar}"
    )


def test_a_chain_with_a_scalar_power_of_its_transfer_is_decided_periodically() -> None:
    """X(pi) squares to the identity: both sides scale by 1 from n to n + 2 and agree at
    n = 0, 1; a scalar that grows from n = 4 on, or the odd parity, is not proved."""
    decision = decide_equal(parse_dsl(_FLIPS), parse_dsl(_parity("(1 + (-1)^n)/2")))
    assert decision.verdict is EqualityVerdict.EQUAL_UNCERTIFIED, decision.reason
    assert "periodic: each side scales by 1 from n to n + 2" in decision.reason
    for scalar in ("(1 - (-1)^n)/2", "(1 + (-1)^n)/2 * 2^(n*(n - 1)*(n - 2))"):
        verdict = decide_equal(parse_dsl(_FLIPS), parse_dsl(_parity(scalar)), max_samples=0).verdict
        assert verdict not in (EqualityVerdict.EQUAL, EqualityVerdict.EQUAL_UNCERTIFIED)
