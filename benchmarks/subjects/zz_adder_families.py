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

"""Boxed Draper and Fourier constant adders: copy-major QFT core with pair phases on n-digit
registers, copy j holding a_j, b_j (draper) or x_j (fca)."""

from benchmarks.subjects._common import *

_C = 2


def _pow(c, e):
    """The dimension token c.D to the power ``e``."""
    if c.D.startswith("literal "):
        return f"literal {c.D[len('literal ') :]}^{e}"
    return f"{c.D}^{e}"


def _draper(n, d):
    """Copy-major QFT core on b, w_d^(a_j y_j), w_{d^(m-j+1)}^(y_j a_m), inverse core on b;
    copy j holds a_j, b_j."""
    gates = [
        Pairs("b", (PairPhase("left", "right", "x*y/d^(m-j+1)"),), between=(F("b"),)),
        CZ("a", "b"),
        Pairs("b", (PairPhase("left", "right", "x*y/d^(m-j+1)"),), right_over="a"),
        Pairs(
            "b",
            (PairPhase("left", "right", "-x*y/d^(m-j+1)"),),
            between=(F("b", 3),),
            left_first=True,
        ),
    ]
    return circuit(["a", "b"], gates, registers={"a": n, "b": n})


def _draper_spec(c):
    """Addition on the registers bound copy by copy (copy 0 most significant), copy j
    holding a_j, b_j."""
    lines = []
    for r in ("a", "b"):
        lines += [
            f"h{r} = Z(0 -> 1, literal 1)",
            f"b{r} = B({_pow(c, 'j')}, {c.D})",
            f"t{r} = S({_pow(c, 'j')}, {c.D})",
            f"e{r} = Z(1 -> 0, literal 1)",
            f"wire h{r}.out0 -- b{r}.in0, t{r}.out0 -- e{r}.in0",
        ]
    return "\n".join(
        [
            *lines,
            f"z = Z(1 -> 2, {_pow(c, c.N)})",
            f"x = X(2 -> 1, {_pow(c, c.N)})",
            "wire ba.out0 -- z.in0, bb.out0 -- x.in0, z.out1 -- x.in1",
            "wire z.out0 -- ta.in0, x.out0 -- tb.in0",
            f"repeat {c.N} as V index j: ba, bb",
            "chain V: ba.out0 -> ba.in0, bb.out0 -> bb.in0",
            f"repeat {c.N} as U index j: ta, tb",
            "chain U: ta.in0 -> ta.out0, tb.in0 -> tb.out0",
            "inputs ba.in1, bb.in1",
            "outputs ta.out1, tb.out1",
            f"scalar {c.d}^({c.N}/2)",
        ]
    )


def _fca(n, d):
    """Copy-major QFT core, digit phase 2 k/d^(n-j) on copy j, inverse core."""
    gates = [
        Pairs("x", (PairPhase("left", "right", "x*y/d^(m-j+1)"),), between=(F("x"),)),
        ZPhase("x", rule=f"{_C}*k/d^({n}-j)"),
        Pairs(
            "x",
            (PairPhase("left", "right", "-x*y/d^(m-j+1)"),),
            between=(F("x", 3),),
            left_first=True,
        ),
    ]
    return circuit(["x"], gates, registers={"x": n})


def _fca_spec(c):
    """The shift by c on the register bound copy by copy (copy 0 most significant)."""
    return "\n".join(
        [
            "h = Z(0 -> 1, literal 1)",
            f"b = B({_pow(c, 'j')}, {c.D})",
            f"s = X(1 -> 1, {_pow(c, c.N)}) phase{{k: -{_C}*k/{c.d}^{c.N}}}",
            f"t = S({_pow(c, 'j')}, {c.D})",
            "e = Z(1 -> 0, literal 1)",
            "wire h.out0 -- b.in0, b.out0 -- s.in0, s.out0 -- t.in0, t.out0 -- e.in0",
            f"repeat {c.N} as V index j: b",
            "chain V: b.out0 -> b.in0",
            f"repeat {c.N} as U index j: t",
            "chain U: t.in0 -> t.out0",
            "inputs b.in1",
            "outputs t.out1",
        ]
    )


SUBJECT = Subject(
    id="zz_adder_families",
    name="Boxed QFT adders",
    category="circuit",
    source="draper2000addition",
    size_parameter="digits per register",
    dimension="general",
    domain=Domain(n_min=1, d_min=2),
    equations=(
        Equation("draper_boxed", _draper, _draper_spec, form="boxed"),
        Equation("fca_boxed", _fca, _fca_spec, form="boxed"),
    ),
)
