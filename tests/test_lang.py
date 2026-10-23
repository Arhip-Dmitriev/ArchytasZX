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

"""Tests for :mod:`archytaszx.repl.lang`: the Dirac language, the Shor worked example in
symbolic and numeric form, typing errors, and the shell's language mode."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from archytaszx.repl.lang import (
    Interpreter,
    LangError,
    LangSyntaxError,
    LangTypeError,
    tokenize,
)
from archytaszx.repl.shell import Shell, main
from archytaszx.semantics import pathsum as P

HORRIBLE = str(Path(__file__).resolve().parent.parent / "examples" / "horrible_state.azx")

SHOR = """\
var n, m, N, a
var q = 2^n
func f(x) = a^x % N

gate U_f = |x>|y> -> |x>|y xor f(x)>
gate QFT  = |x> -> 1/sqrt(q) sum(y in 0..q){ e^(2*pi*i*x*y/q) |y> }

|psi_0> = |0>^@n @ |0>^@m
|psi_1> = (H^@n @ I)|psi_0>
|psi_2> = U_f|psi_1>
var z = measure|psi_2>
|psi_3> = (QFT @ I) |psi_2>
list states = {|psi_0>, |psi_1>, |psi_2>, |psi_2>, |psi_3>}
show(states)
var y = measure|psi_2>
"""

SHOR_ASCII = """\
|psi_0> = |0>^@n @ |0>^@m
|psi_1> = 1/sqrt(q) sum(x in 0..q){ |x> @ |0>^@m }
|psi_2><psi_2| = 1/q sum(k0 in 0..q){ |k0><k0| @ |f(k0)><f(k0)| }
|psi_2><psi_2| = 1/q sum(k0 in 0..q){ |k0><k0| @ |f(k0)><f(k0)| }
|psi_3><psi_3| = q^(-2) sum(k0 in 0..q, y in 0..q, y' in 0..q){ e^(2*pi*i*k0*(y - y')/q) \
|y><y'| @ |f(k0)><f(k0)| }"""

SHOR_UNICODE = """\
|ψ_0⟩ = |0⟩^⊗n ⊗ |0⟩^⊗m
|ψ_1⟩ = 1/√q Σ_{x=0}^{q - 1} |x⟩ ⊗ |0⟩^⊗m
|ψ_2⟩⟨ψ_2| = 1/q Σ_{k0=0}^{q - 1} |k0⟩⟨k0| ⊗ |f(k0)⟩⟨f(k0)|
|ψ_2⟩⟨ψ_2| = 1/q Σ_{k0=0}^{q - 1} |k0⟩⟨k0| ⊗ |f(k0)⟩⟨f(k0)|
|ψ_3⟩⟨ψ_3| = q^(-2) Σ_{k0=0}^{q - 1} Σ_{y=0}^{q - 1} Σ_{y'=0}^{q - 1} e^{2πi·k0·(y - y')/q} \
|y⟩⟨y'| ⊗ |f(k0)⟩⟨f(k0)|"""


def numeric_shor(n: int = 3, m: int = 4, big_n: int = 15, a: int = 7) -> str:
    head = f"var n = {n}\nvar m = {m}\nvar N = {big_n}\nvar a = {a}\n"
    return head + SHOR.split("\n", 1)[1]


def run(source: str, style: str = "ascii", seed: int | None = 0) -> tuple[Interpreter, str]:
    interp = Interpreter(style=style, seed=seed)
    return interp, interp.execute(source)


# -- the worked example --------------------------------------------------------------------


class TestShorSymbolic:
    def test_ascii(self) -> None:
        interp, out = run(SHOR)
        assert out == SHOR_ASCII
        assert interp.execute("show(y)") == "y = [(2^m*k0 + f(k0), 1/q) : k0 in 0..q]"

    def test_unicode(self) -> None:
        assert run(SHOR, "unicode")[1] == SHOR_UNICODE

    def test_plot_falls_back_to_unicode_without_a_backend(self) -> None:
        interp = Interpreter(style="plot", plotter=lambda lines: None)
        assert interp.execute(SHOR) == SHOR_UNICODE

    def test_plot_prints_only_the_image(self) -> None:
        drawn: list[list[str]] = []
        interp = Interpreter(style="plot", plotter=lambda lines: drawn.append(lines) or "IMG")
        assert interp.execute(SHOR) == "IMG"
        assert len(drawn) == 1 and len(drawn[0]) == 5
        assert drawn[0][1].startswith(r"|\psi_{1}\rangle = \frac{1}{\sqrt{q}} \sum_{x=0}^{q - 1}")

    def test_no_output_without_show(self) -> None:
        assert run(SHOR.replace("show(states)\n", ""))[1] == ""


class TestShorNumeric:
    def test_states_and_outcomes(self) -> None:
        interp, _ = run(numeric_shor())
        assert interp.execute("show(z, y)") == (
            "z = [(16*k0 + f(k0), 0.125) : k0 in 0..8]\ny = [(16*k0 + f(k0), 0.125) : k0 in 0..8]"
        )
        (branch,) = interp.values["z"].branches  # type: ignore[union-attr]
        (k,) = branch.free
        for x in range(8):
            z = int(branch.value.xreplace({k.symbol: x}))
            assert z >> 4 == x and z & 15 == pow(7, x, 15)
        assert interp.execute("show(|psi_2>, <psi_1|psi_1>)") == (
            "|psi_2><psi_2| = 1/8 sum(k0 in 0..8){ |k0><k0| @ |f(k0)><f(k0)| }\n<psi_1|psi_1> = 1"
        )

    def test_display_is_numeric(self) -> None:
        out = run(numeric_shor())[1].splitlines()
        assert out[0] == "|psi_0> = |0>^@3 @ |0>^@4"
        assert out[1] == "|psi_1> = sqrt(2)/4 sum(x in 0..8){ |x> @ |0>^@4 }"
        assert not any(line.lstrip().startswith("where") for line in out)

    def test_measurement_is_the_same_for_every_seed(self) -> None:
        a = run(numeric_shor(), seed=5)[0].execute("show(z)")
        b = run(numeric_shor(), seed=9)[0].execute("show(z)")
        assert a == b == "z = [(16*k0 + f(k0), 0.125) : k0 in 0..8]"

    def test_symbolic_and_concrete_sizes_agree(self) -> None:
        symbolic = run(SHOR)[0].execute("show(z)")
        assert symbolic == "z = [(2^m*k0 + f(k0), 1/q) : k0 in 0..q]"


# -- language ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("|a> = H|0>", "|a> = sqrt(2)/2 sum(y in 0..2){ |y> }"),
        ("|a> = H |+>", "|a> = |0>"),
        ("|a> = H|->", "|a> = |1>"),
        ("|a> = X|0> @ |1>", "|a> = |1> @ |1>"),
        ("|a> = CNOT (H|0> @ |0>)", "|a> = sqrt(2)/2 sum(y in 0..2){ |y> @ |y> }"),
        ("|a> = sum(y in 0..2){ |y> } + |1>", "|a> = sum(y in 0..2){ |y> } + |1>"),
        ("|a> = H|0> + H|1>", "|a> = sqrt(2) |0>"),
        ("|a> = |010`>", "|a> = |0> @ |1> @ |0>"),
        ("|a> = |000>", "|a> = |0>"),
        ("|a> = normalise(3 |0> + 4 |1>)", "|a> = 3/5 |0> + 4/5 |1>"),
        ("|a> = 1/sqrt(2) (|0> - |1>)", "|a> = sqrt(2)/2 |0> - sqrt(2)/2 |1>"),
        ("|a> = |1>^@3", "|a> = |1>^@3"),
        ("|a> = |5>", "|a>_6 = |5>_6"),
        ("|a> = |0> + |2>", "|a>_3 = |0>_3 + |2>_3"),
        ("|a> = |0> + |1>", "|a> = |0> + |1>"),
        ("|a>_3 = |0> + |1>", "|a>_3 = |0>_3 + |1>_3"),
        ("|a> = |0>_3 + |1>", "|a>_3 = |0>_3 + |1>_3"),
        ("|a> = |1>|0>|1>", "|a> = |1> @ |0> @ |1>"),
        ("|a> = |0> + |0>|0>", "|a> = 2 |0>^@2"),
        ("|a> = |0>|2> - |1>|0>", "|a> = |0> @ |2>_3 - |1> @ |0>_3"),
        ("|a> = |i>", "|a> = sqrt(2)/2 |0> + sqrt(2)*i/2 |1>"),
        ("|a> = |-i>", "|a> = sqrt(2)/2 |0> - sqrt(2)*i/2 |1>"),
        ("|a> = H*X|0>", "|a> = sqrt(2)/2 sum(y in 0..2){ (-1)^y |y> }"),
        ("|a> = sum(i in 0..2){ |i> }", "|a> = sum(i in 0..2){ |i> }"),
        ("|a> = |0>|1>", "|a> = |0> @ |1>"),
        ("|a> = (|0><1|) |1>", "|a> = |0>"),
        ("|a> = (X @ I) |0>|0>|0>", "|a> = |1> @ |0> @ |0>"),
        ("|a> = SWAP |0>|1>", "|a> = |1> @ |0>"),
        ("|a> = (X*X) |0>", "|a> = |0>"),
        ("|a> = X^3 |0>", "|a> = |1>"),
        ("|a> = (X + Z) |0>", "|a> = |0> + |1>"),
        ("|a> = Y|0>", "|a> = i |1>"),
        ("|a> = T|1>", "|a> = e^(2*pi*i/8) |1>"),
    ],
)
def test_concrete_states(source: str, expected: str) -> None:
    _, out = run(source + "\nshow(|a>)")
    assert out == expected


class TestSymbolic:
    def test_hadamard_power_twice_is_identity(self) -> None:
        assert run("var n\n|a> = H^@n H^@n |0>^@n\nshow(|a>)")[1] == "|a> = |0>^@n"

    def test_hadamard_power_without_sugar(self) -> None:
        out = run("var n\n|a> = H^@n |0>^@n\nshow(|a>)")[1]
        assert out == "|a> = 2^(-n/2) sum(x in 0..2^n){ |x> }"

    def test_norm(self) -> None:
        assert run("var n\n|a> = H^@n |0>^@n\nshow(<a|a>)")[1] == "<a|a> = 1"

    def test_ket_power_of_plus(self) -> None:
        out = run("var n\nvar q = 2^n\n|a> = |+>^@n\nshow(|a>)")[1]
        assert out.startswith("|a> = 1/sqrt(q) sum(x in 0..q){ |x> }")

    def test_real_parameter(self) -> None:
        source = "var t: real\ngate R = |x> -> e^(i*t*x) |x>\n|r> = R|1>\nshow(|r>, <r|r>)"
        assert run(source)[1] == "|r> = e^(i*t) |1>\n<r|r> = 1"

    def test_bra_definition_and_inner(self) -> None:
        source = "<b| = <0| + <1|\n|k> = |0>\nshow(<b|, |b>, <b|k>)"
        assert run(source)[1] == "<b| = <0| + <1|\n|b> = |0> + |1>\n<b|k> = 1"

    def test_qft_inverse_contracts(self) -> None:
        source = (
            "var n\nvar q = 2^n\n"
            "gate F = |x> -> 1/sqrt(q) sum(y in 0..q){ e^(2*pi*i*x*y/q) |y> }\n"
            "gate G = |x> -> 1/sqrt(q) sum(y in 0..q){ e^(-2*pi*i*x*y/q) |y> }\n"
            "|a> = G F |0>^@n\n|b> = F G |1>^@n\nshow(|a>, |b>)"
        )
        assert run(source)[1] == "|a> = |0>^@n\n|b> = |1>^@n"

    def test_measure_of_a_basis_state_is_certain(self) -> None:
        interp, _ = run("|a> = |1>\nvar b = measure|a>")
        assert interp.execute("show(b)") == "b = [(1, 1)]"
        assert interp.execute("var c = b + 1\nshow(c)") == "c = 2"

    def test_certain_outcomes_act_as_numbers(self) -> None:
        interp, _ = run("|a> = |1>|0>|1>\nvar b = measure|a>\nvar c = b * 2\nshow(b, c)")
        assert interp.execute("show(b, c)") == "b = [(5, 1)]\nc = 10"

    def test_symbolic_measure_of_basis_state(self) -> None:
        interp, _ = run("var n, k\n|o> = |0>^@n\n|a> = (|o> - |o> + |k>)|0>\nvar b = measure|a>")
        assert interp.execute("show(b, |a>)") == "b = [(2*k, 1)]\n|a><a| = |k><k| @ |0><0|"

    def test_measure_leaves_the_mixture(self) -> None:
        interp, _ = run("|a> = H|0>\nvar b = measure|a>\nvar c = measure|a>")
        assert interp.execute("show(b, c, |a>)") == (
            "b = [(k, 0.5) : k in 0..2]\nc = [(k, 0.5) : k in 0..2]\n"
            "|a><a| = 1/2 sum(k in 0..2){ |k><k| }"
        )

    def test_measure_is_an_expression(self) -> None:
        source = (
            "var n\n|p> = |0>^@n @ |1>\n(H^@n @ H)|p>\ngate U = |x>|y> -> |x>|y xor bit(x, 0)>\n"
            "U|p>\n(H^@n @ I)|p>\nvar x = floor(measure|p> / 2)\nshow(x)"
        )
        assert run(source)[1] == "x = [(1, 1)]"

    def test_measure_is_the_same_for_every_size(self) -> None:
        source = (
            "func f(x) = bit(x, 0)\ngate U = |x>|y> -> |x>|y xor f(x)>\n"
            "|p> = (H^@n @ I) U (H^@n @ H) (|0>^@n @ |1>)\nvar z = measure|p>\nshow(z)"
        )
        for head in ("var n\n", "var n = 3\n", "var n = 40\n"):
            assert run(head + source)[1] == "z = [(k1 + 2, 0.5) : k1 in 0..2]", head

    def test_operators_act_on_the_mixture(self) -> None:
        interp, _ = run("var n\n|p> = H^@n |0>^@n @ |->\nvar z = measure|p>\n(H^@n @ H)|p>")
        assert interp.execute("show(|p>)") == (
            "|p><p| = 2^(-n - 1) sum(y in 0..2, x in 0..2^n){ |x><x| @ |y><y| }"
        )

    def test_density_matrix_from_outer_products(self) -> None:
        interp, out = run("|r> = 1/2|0><0| + 1/2|1><1|\nshow(|r>)")
        assert out == "|r><r| = 1/2 |0><0| + 1/2 |1><1|"
        assert interp.execute("show(measure|r>)") == "measure|r> = [(k, 0.5) : k in 0..2]"

    def test_apply_in_place(self) -> None:
        interp, _ = run("|a> = |0>\nX|a>\nH X|a>\nshow(|a>)")
        assert interp.execute("show(|a>)") == "|a> = sqrt(2)/2 sum(y in 0..2){ |y> }"


def dense_qudit(s: P.State) -> np.ndarray:
    keys, amps = P.enumerate_state(s)
    out = np.zeros(int(np.prod(P.concrete_levels(s))), dtype=complex)
    np.add.at(out, keys.astype(np.int64), amps)
    return out


FOURIER = "gate F = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(2*pi*i*x*y/d) |y>_d }\n"


class TestQudits:
    def test_symbolic_fourier_power(self) -> None:
        source = "var d, n\nvar q = d^n\n" + FOURIER + "|chi>_d = F^@n |0>_d^@n\n"
        out = run(source + "show(|chi>_d, <chi|chi>_d)")[1]
        assert out == "|chi>_d = 1/sqrt(q) sum(x in 0..q){ |x>_d }\n<chi|chi>_d = 1"

    def test_fourier_twice_reflects_zero(self) -> None:
        source = "var d, n\n" + FOURIER + "|b>_d = F^@n F^@n |0>_d^@n\nshow(|b>_d)"
        assert run(source)[1] == "|b>_d = |0>_d^@n"

    def test_symbolic_measurement_domain(self) -> None:
        source = "var d, n\n" + FOURIER + "|chi>_d = F^@n |0>_d^@n\nvar z = measure|chi>_d\n"
        out = run(source + "show(|chi>_d)", "unicode")[1]
        assert out == "|χ⟩⟨χ|_d = d^(-n) Σ_{k=0}^{d^n - 1} |k⟩_d⟨k|_d"
        z = run(source + "show(z)", "unicode")[1]
        assert z == "z = [(k, d^(-n)) : k ∈ {0,…,d^n - 1}]"

    def test_free_parameters_bind_to_concrete_registers(self) -> None:
        source = (
            "var d, n\n" + FOURIER + "|p> = (1 + i)|1>_3|2>_3|0>_3|1>_3 - 2|2>_3|2>_3|0>_3|0>_3"
            " + sum(k in 0..3){ |k>_3^@4 / (k + 1) }\n|q> = F^@n |p>\n"
        )
        interp, _ = run(source)
        p, q = interp.states["p"], interp.states["q"]
        assert q.widths == (4,) and q.dims == (3,)
        f3 = np.exp(2j * np.pi * np.outer(range(3), range(3)) / 3) / np.sqrt(3)
        full = f3
        for _ in range(3):
            full = np.kron(full, f3)
        assert np.allclose(dense_qudit(q), full @ dense_qudit(p))

    def test_bound_parameters_keep_the_closed_form(self) -> None:
        kets = "".join(f"|{k % 7}>_7" for k in range(22))
        for head, power in (("var d, n\n", "n"), ("var d = 7\n", "22")):
            source = head + FOURIER + f"|p> = |0>_7^@22 + {kets}\n|q> = F^@{power} |p>\n"
            interp, _ = run(source + "show(|q>)")
            q = interp.states["q"]
            assert len(q.terms) == 2 and all(t.bound for t in q.terms)
            interp.execute("var m = measure|q>")
            m = interp.values["m"]
            assert isinstance(m, P.Distribution) and P.certain(m) is None

    def test_horrible_state_runs(self) -> None:
        interp, out = run(Path(HORRIBLE).read_text())
        psi = interp.states["psi"]
        assert psi.widths == (22,) and psi.dims == (7,)
        assert all(t.bound for t in psi.terms)
        assert out.splitlines()[1].count("sum(") == len(psi.terms) == 38
        interp.execute("|phi> = F^@n |psi>")
        phi = interp.states["phi"]
        assert sum(not t.bound for t in phi.terms) == 37

    def test_digit_shift_power(self) -> None:
        source = "var d, n\ngate Xd = |x>_d -> |(x + 1) % d>_d\n|v>_d = Xd^@n |0>_d^@n\n"
        assert run(source + "show(|v>_d)")[1] == "|v>_d = |1>_d^@n"

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("|a>_3 = |2>_3", "|a>_3 = |2>_3"),
            ("|a>_3 = |2>_3 @ |1>_3", "|a>_3 = |2>_3 @ |1>_3"),
            ("|a>_3 = |1>_3|2>_3", "|a>_3 = |1>_3 @ |2>_3"),
            ("gate F3 = |x>_3 -> |x>_3\n|a> = F3|1>", "|a>_3 = |1>_3"),
            ("|a>_3 = |0>_3^@2", "|a>_3 = |0>_3^@2"),
            ("gate X3 = |x>_3 -> |(x + 1) % 3>_3\n|a>_3 = X3^@4 |0>_3^@4", "|a>_3 = |1>_3^@4"),
            (
                "gate X3 = |x>_3 -> |(x + 1) % 3>_3\n|a>_3 = (X3 @ I) |2>_3|1>_3",
                "|a>_3 = |0>_3 @ |1>_3",
            ),
            ("|a> = |1>_3 @ |0>", "|a> = |1>_3 @ |0>"),
            ("|a>_3 = sum(y in 0..3){ |y>_3 }", "|a>_3 = sum(y in 0..3){ |y>_3 }"),
            (
                "|a>_3 = sum(y in 0..3){ |y>_3 } + |1>_3",
                "|a>_3 = sum(y in 0..3){ |y>_3 } + |1>_3",
            ),
        ],
    )
    def test_concrete(self, source: str, expected: str) -> None:
        name = expected.split(" =")[0]
        assert run(source + f"\nshow({name})")[1] == expected

    def test_qutrit_fourier_numeric(self) -> None:
        source = (
            "gate F3 = |x>_3 -> 1/sqrt(3) sum(y in 0..3){ e^(2*pi*i*x*y/3) |y>_3 }\n"
            "|g>_3 = F3^@2 |0>_3^@2\nshow(|g>_3, <g|g>_3)"
        )
        assert run(source)[1] == "|g>_3 = 1/3 sum(x in 0..9){ |x>_3 }\n<g|g>_3 = 1"

    def test_qutrit_measurement(self) -> None:
        interp, _ = run("|a>_3 = sum(y in 0..3){ |y>_3 }\nvar r = measure|a>_3")
        assert interp.execute("show(r)") == "r = [(k, 1/3) : k in 0..3]"

    def test_mixed_radix_outcome(self) -> None:
        interp, _ = run("|m> = |2>_3 @ |1>\nvar k = measure|m>")
        (branch,) = interp.values["k"].branches  # type: ignore[union-attr]
        assert branch.value == 5 and branch.registers == (2, 1) and branch.prob == 1

    def test_inner_shorthand_suffix(self) -> None:
        assert run("|a>_5 = |3>_5\nshow(<a|a>_5)")[1] == "<a|a>_5 = 1"

    @pytest.mark.parametrize(
        ("source", "match"),
        [
            ("|a>_2 = |2>", "does not fit in dimension 2"),
            ("|a> = |0>_3 + |5>", "does not fit in dimension 3"),
            ("|a> = |0> + |2>\n|b> = |a> + |5>", "does not fit in dimension 3"),
            ("|a> = |5>_3", "does not fit in one digit"),
            ("|a>_4 = |2>_3", "dimension 3"),
            ("|a>_3 = |2>_3\nshow(|a>_5)", "does not match"),
            ("|a>_3 = H|0>_3", "single qubits"),
            ("|a>_3 = |3>_3", "does not fit"),
            ("|a> = |3 : 1>", "not a ket"),
            ("|a>_1 = |0>_1", "at least 2"),
            ("|a> = |0>_3 + |0>_4", "dimensions"),
            ("gate F3 = |x>_3 -> |x>_3\n|a>_7 = F3|0>_7", "dimension-7"),
            ("var d\ngate G = |x>_d|y>_d -> |x>_d|y>_d\n|a> = G|0>_3|0>_5", "dimension-5"),
            ("|a>_3 = |+>_3", "qubit state"),
        ],
    )
    def test_errors(self, source: str, match: str) -> None:
        with pytest.raises(LangError, match=match):
            run(source)


class TestErrors:
    @pytest.mark.parametrize(
        ("source", "error", "match"),
        [
            ("|a> = 3", LangTypeError, "must be assigned a state"),
            ("var x = |0>", LangTypeError, "must be a scalar"),
            ("|a> = <0|", LangTypeError, "cannot assign a bra"),
            ("|a> = H|0> @ H", LangTypeError, "cannot tensor"),
            ("|a> = |0>_2 + |0>|0>", LangError, "different widths"),
            ("var n\n|a> = H|0>^@n", LangError, "single qubits"),
            ("|a> = QQ|0>", LangError, "not defined"),
            ("var H", LangTypeError, "already a op"),
            ("var pi", LangSyntaxError, "reserved"),
            ("gate H = |x> -> |x>", LangTypeError, "built-in"),
            ("oracle B = |x> -> |0>", LangSyntaxError, "oracle is gone"),
            ("|a> = |0, 1>", LangSyntaxError, "write \\|0>\\|1>"),
            ('show("x")', LangTypeError, "print"),
            ("var n\nprint(n)", LangTypeError, "symbolic"),
            ("print(|0>)", LangTypeError, "not a concrete value"),
            ("|a> = |0\nshow(|a>)", LangSyntaxError, "unclosed ket"),
            ("hello", LangError, "not defined"),
            ("var x = 1\nx x x =", LangSyntaxError, "a statement starts"),
            ("var x: float", LangTypeError, "unknown type"),
            ("|a> = |0>\nvar z = measure|b>", LangTypeError, "not defined"),
            ("|a> = |0>\nshow(a)", LangTypeError, "is a state"),
            ("var k\n|a> = |k>", LangTypeError, "register width"),
            ("|a> = (H @ H) |0>", LangError, "registers"),
            ("func f(x) = |x>", LangTypeError, "scalar"),
            ("|a> = |4>_2", LangTypeError, "does not fit"),
            ("|a> = |4 : 2>", LangSyntaxError, "not a ket"),
        ],
    )
    def test_error(self, source: str, error: type[Exception], match: str) -> None:
        with pytest.raises(error, match=match):
            run(source)


def test_tokenize_braket() -> None:
    kinds = [(t.kind, t.text) for t in tokenize("<a|b> |x>|y> -> <c|")]
    assert kinds == [
        ("bra", "a"),
        ("ket", "b"),
        ("ket", "x"),
        ("ket", "y"),
        ("op", "->"),
        ("bra", "c"),
    ]


# -- shell ---------------------------------------------------------------------------------


class TestShellLanguage:
    def test_language_is_the_default(self) -> None:
        shell = Shell(display="ascii")
        assert shell.run_line("|a> = H|0>") == ""
        assert shell.run_line("show(|a>)") == "|a> = sqrt(2)/2 sum(y in 0..2){ |y> }"

    def test_engine_commands_take_a_colon(self) -> None:
        shell = Shell(display="ascii")
        assert shell.run_line(":zx") == "mode: zx"
        assert "spider_fusion" in shell.run_line(":rules")

    def test_errors_are_counted(self) -> None:
        shell = Shell(display="ascii")
        assert shell.run_line("|a> = 3").startswith("error: ")
        assert shell.errors == 1

    def test_script(self, capsys: pytest.CaptureFixture[str], tmp_path: object) -> None:
        path = tmp_path / "shor.azx"  # type: ignore[operator]
        path.write_text(SHOR, encoding="utf-8")
        assert main(["--ascii", str(path)]) == 0
        assert capsys.readouterr().out.rstrip("\n") == SHOR_ASCII

    def test_unicode_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--unicode", "-c", "|a> = |0>^@2", "-c", "show(|a>)"]) == 0
        assert capsys.readouterr().out == "|a⟩ = |0⟩^⊗2\n"

    def test_loop(self) -> None:
        out = io.StringIO()
        Shell(display="ascii").loop(io.StringIO("|a> = |1>\nshow(|a>)\n"), out)
        assert out.getvalue() == "|a> = |1>\n"


SHOR_SOURCE = """\
var N, a
var m = bitlen(N)
var n = 2*m
var Q = 2^n

gate U_f = |x>|y> -> |x>|y xor powmod(a, x, N)>
gate IQFT = |x> -> 1/sqrt(Q) sum(k in 0..Q){ e^(-2*pi*i*x*k/Q) |k> }

|psi> = |0>^@n @ |0>^@m
|psi> = (H^@n @ I) |psi>
|psi> = U_f |psi>
|psi> = (IQFT @ I) |psi>
show(|psi>)

var z = measure|psi>
var r = denom(floor(z / 2^m) / Q, N)
var h = powmod(a, r/2, N)
var p = gcd(h - 1, N)
var q = N / p
show(r, p, q)
"""


class TestShorsScript:
    def test_symbolic(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        script = tmp_path / "shor.azx"
        script.write_text(SHOR_SOURCE)
        assert main([str(script), "--ascii"]) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == (
            "|psi> = 1/Q sum(x in 0..Q, k in 0..Q){ e^(-2*pi*i*k*x/Q) |k> @ |a^x % N> }"
        )
        assert out[1].startswith("r = [(denom(k0/Q + floor(k1/2^m)/Q, N), ")
        assert out[2].startswith("p = [(gcd(a^(denom(") and out[3].startswith("q = [(N/gcd(")

    def test_numeric(self) -> None:
        source = SHOR_SOURCE.replace("var N, a\n", "var N = 15, a = 7\n", 1)
        interp, out = run(source, seed=0)
        lines = out.splitlines()
        assert lines[1].startswith("r = [(denom(k0/256, 15), ")
        (p,) = interp.values["p"].branches  # type: ignore[union-attr]
        (q,) = interp.values["q"].branches  # type: ignore[union-attr]
        assert p.free == q.free and P.same(p.value * q.value, 15)


class TestControlFlow:
    def test_for_while_if(self) -> None:
        source = (
            "var s = 0\nfor j in 1..=4 {\n  var s = s + j\n}\nvar k = 0\n"
            "while k < 3 { var k = k + 1 }\nif s == 10 { var t = 1 } else { var t = 2 }\n"
            "show(s, k, t, s > 9 and not k == 2)"
        )
        assert run(source)[1] == "s = 10\nk = 3\nt = 1\ns > 9 and not k == 2: true"

    def test_symbolic_loop_folds(self) -> None:
        source = (
            "var n, a, N\n"
            "gate M = |x>|y> -> |x>|(y * powmod(a, 2^j, N)^bit(x, j)) % N>\n"
            "|o> = |0>^@n\n|s> = H^@n |0>^@n @ (|o> - |o> + |1>)\n"
            "for j in 0..n { |s> = M |s> }\nshow(|s>)"
        )
        assert run(source)[1] == "|s> = 2^(-n/2) sum(x in 0..2^n){ |x> @ |a^x % N> }"

    def test_guarded_rotation_loop_folds(self) -> None:
        source = (
            "var n, d, x\nvar M = d^n - 1\nfunc rot(y) = ifelse(y < M, (y*d) % M, y)\n"
            "gate ROT = |y> -> |rot(y)>\n|z>_d = |0>_d^@n\n|s>_d = 0 |z> + |x>\n"
            "for j in 0..n { |s> = ROT |s> }\nshow(|s>)"
        )
        assert run(source)[1] == "|s>_d = |x>_d"

    def test_xor_loop_folds(self) -> None:
        source = (
            "var n, t\ngate CX = |c>|y> -> |c>|y xor c*2^j>\n|z> = |0>^@n\n"
            "|s> = H|0> @ (0 |z> + |t>)\nfor j in 0..n { |s> = CX |s> }\nshow(|s>)"
        )
        assert run(source)[1] == "|s> = sqrt(2)/2 sum(y in 0..2){ |y> @ |(t xor y*(2^n - 1))> }"

    def test_digit_phase_loop_folds(self) -> None:
        source = (
            "var n, d, x\nvar Q = d^n\n"
            "gate QFT = |x>_d -> 1/sqrt(Q) sum(k in 0..Q){ e^(2*pi*i*x*k/Q) |k>_d }\n"
            "gate IQFT = |x>_d -> 1/sqrt(Q) sum(k in 0..Q){ e^(-2*pi*i*x*k/Q) |k>_d }\n"
            "gate P = |k> -> e^(2*pi*i*d^j*(floor(k/d^j) % d)/Q) |k>\n"
            "|z>_d = |0>_d^@n\n|a>_d = 0 |z> + |x>\n|a> = QFT |a>\n"
            "for j in 0..n { |a> = P |a> }\n|a> = IQFT |a>\nshow(|a>)"
        )
        assert run(source)[1] == "|a>_d = |(x + 1) % Q>_d"

    def test_qudit_phase_kickback(self) -> None:
        source = (
            "var n, d, c\n"
            + FOURIER
            + "gate Fd = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(-2*pi*i*x*y/d) |y>_d }\n"
            "gate U_c = |x>|y>_d -> |x>|(y + c) % d>_d\n"
            "gate U_b = |x>|y>_d -> |x>|(y + x) % d>_d\n|p>_d = |0>_d^@n @ |1>_d\n"
            "|t>_d = |0>_d^@n @ Fd |1>_d\n"
            "|q> = (Fd^@n @ I) U_c (F^@n @ Fd) |p>\n|r> = (Fd^@n @ I) U_b (F^@n @ Fd) |p>\n"
            "show(<t|q>, <t|r>)"
        )
        assert run(source)[1] == "<t|q> = e^(2*pi*i*c/d)\n<t|r> = 0"

    def test_bit_dot(self) -> None:
        source = (
            "var n, x, a, b\ngate XA = |y> -> |y xor a>\ngate ZB = |y> -> (-1)^dot(y, b) |y>\n"
            "gate XB = |y> -> |y xor b>\ngate ZA = |y> -> (-1)^dot(y, a) |y>\n"
            "|z> = |0>^@n\n|w> = 0 |z> + |x>\n|l> = H^@n XA ZB H^@n |w>\n|r> = ZA XB |w>\n"
            "show(<r|l>, dot(6, 3), dot(5, 7, 3))"
        )
        assert run(source)[1] == "<r|l> = 1\ndot(6, 3) = 1\ndot(5, 7, 3) = 4"

    def test_teleport_loop_folds(self) -> None:
        source = (
            "var n, d, x\n|B>_d = 1/sqrt(d) sum(k in 0..d){ |k>_d @ |k>_d }\n"
            "gate HOP = |a>_d -> d sum(c in 0..d){ ((<B| @ <c|_d)(|a>_d @ |B>)) |c>_d }\n"
            "|l> = |x>_d\nfor j in 0..n { |l> = HOP |l> }\nshow(|l>)"
        )
        assert run(source)[1] == "|l>_d = |x>_d"

    def test_gate_body_bound_index_not_captured(self) -> None:
        source = (
            "var d\n|B>_d = 1/sqrt(d) sum(k in 0..d){ |k>_d @ |k>_d }\n"
            "gate HOP = |a>_d -> d sum(c in 0..d){ ((<B| @ <c|_d)(|a>_d @ |B>)) |c>_d }\n"
            "show((I @ HOP)|B>)"
        )
        assert run(source)[1] == "(I @ HOP)|B> = 1/sqrt(d) sum(k in 0..d){ |k>_d @ |k>_d }"

    def test_assume_and_assert(self) -> None:
        interp, _ = run("var r\nassume r > 2\nshow(r > 2, r > 3)")
        assert interp.execute("show(r > 2, r > 3)") == "r > 2: assumed\nr > 3: open"
        with pytest.raises(LangError, match="assertion failed"):
            interp.execute("assert 1 == 2")

    def test_arguments(self) -> None:
        interp = Interpreter(style="ascii", args=("21", "x"))
        assert interp.execute(
            "var a = arg(1)\nvar b = arg(2)\nvar c = arg(3, 7)\nshow(a, b, c)"
        ) == ("a = 21\nb = x\nc = 7")


class TestPrintAndEcho:
    def test_print_strings_and_concrete_values(self) -> None:
        out = run('var t = 3/4\nprint("t is", t, 2 + 3*i, sqrt(4), 1 == 1)')[1]
        assert out == "t is 3/4 2 + 3i 2 true"

    def test_print_measured_distribution(self) -> None:
        out = run("|a> = |1>|0>|1>\nvar z = measure|a>\nprint(z)")[1]
        assert out == "[(5, 1)]"

    def test_echo_of_each_statement_kind(self) -> None:
        interp = Interpreter(style="ascii")
        cases = [
            ("var n", ("n", "nat", None)),
            ("var q = 2^n", ("q", "nat", "2^n")),
            ("func f(x) = x + 1", ("f(x)", "func", "x + 1")),
            ("gate U = |x> -> |x xor 1>", ("U", "gate", "|x> -> |x xor 1>")),
            ("|a> = X|0>", ("|a>", "ket", "|1>")),
            ("<b| = <0|", ("<b|", "bra", "<0|")),
            ("<a|a>", ("-", "nat", "1")),
            ("H", ("-", "op", "H")),
            ("list L = {|a>}", ("L", "list", "{|a>}")),
            ("var z = measure|a>", ("z", "dist", "[(1, 1)]")),
        ]
        for source, expected in cases:
            interp.execute(source)
            assert interp.last is not None, source
            assert interp.echo_parts(interp.last) == expected, source
        interp.execute("seed 3")
        assert interp.last is None

    def test_only_the_last_statement_echoes(self) -> None:
        interp = Interpreter(style="ascii")
        interp.execute("var a = 1; var b = 2")
        assert interp.echo_parts(interp.last) == ("b", "nat", "2")  # type: ignore[arg-type]


class TestOracleCheck:
    def test_off_by_default(self) -> None:
        assert run("gate B = |x> -> |0>\n|a> = B|1>\nshow(|a>)")[1] == "|a> = |0>"

    def test_irreversible_classical_gate_is_refused(self) -> None:
        interp = Interpreter(style="ascii", check_oracles=True)
        with pytest.raises(LangError, match="not reversible"):
            interp.execute("gate B = |x> -> |0>\n|a> = B|1>")

    def test_quantum_gates_pass(self) -> None:
        interp = Interpreter(style="ascii", check_oracles=True)
        interp.execute("gate G = |x> -> H|x>\ngate N = |x> -> |1 - x>\n|a> = G N|0>")
        assert interp.execute("show(|a>)") == "|a> = sqrt(2)/2 sum(y in 0..2){ (-1)^y |y> }"


class TestCompose:
    def test_star_composes(self) -> None:
        assert run("|a> = (X*X)|0>\nshow(|a>)")[1] == "|a> = |0>"

    def test_concatenated_names_do_not_compose(self) -> None:
        with pytest.raises(LangError, match="'XX' is not defined"):
            run("|a> = XX|0>")


class TestFormat:
    @pytest.mark.parametrize(
        ("source", "printed"),
        [
            ('print(f"x = {1/3:.3f}")', "x = 0.333"),
            ('var z = 5\nprint(f"{z} and {z:03d}, {{z}}")', "5 and 005, {z}"),
            ('|a> = |1>\nvar b = measure|a>\nprint(f"b={b}")', "b=[(1, 1)]"),
            ('|a> = |1>\nvar b = measure|a>\nprint(f"{b:d}")', "1"),
            ('var n\nprint(f"n+1 is {n + 1}")', "n+1 is n + 1"),
            ('|a> = |0> + |2>\nprint(f"{|a>}")', "|0>_3 + |2>_3"),
            ('print(f"{2:>4}|")', "   2|"),
            ('print("# and ;", f"{1}") # comment', "# and ; 1"),
        ],
    )
    def test_print(self, source: str, printed: str) -> None:
        assert run(source)[1] == printed

    @pytest.mark.parametrize(
        ("source", "match"),
        [
            ('print(f"{1")', "unclosed"),
            ('print(f"1}")', "single }"),
            ('print(f"{}")', "empty"),
            ('var n\nprint(f"{n:.2f}")', "concrete"),
            ('show(f"x")', "show takes values"),
        ],
    )
    def test_errors(self, source: str, match: str) -> None:
        with pytest.raises(LangError, match=match):
            run(source)


class TestRanges:
    def test_exclusive_and_inclusive(self) -> None:
        assert run("for j in 0..3 { print(j) }")[1] == "0\n1\n2"
        assert run("for j in 0..=3 { print(j) }")[1] == "0\n1\n2\n3"
        assert run("show(sum(k in 1..4){ k })")[1] == "sum(k in 1..4){ k } = 6"
        assert run("show(sum(k in 1..=4){ k })")[1] == "sum(k in 1..=4){ k } = 10"

    def test_else_on_a_later_line(self) -> None:
        source = (
            "var t = 2\nif t == 1 {\n  print(1)\n}\n"
            "else if t == 2 { print(2) }\nelse {\n  print(3)\n}"
        )
        assert run(source)[1] == "2"

    def test_lone_else(self) -> None:
        with pytest.raises(LangError, match="else without an if"):
            run("else { print(1) }")


class TestShowModes:
    def test_zx_and_both(self) -> None:
        from archytaszx.repl.printer import NotationMode

        interp, _ = run("|a> = |1>")
        zx = interp.render([(None, interp.states["a"])], "ascii", NotationMode.ZX)
        assert zx.splitlines()[0] == "=" and zx.splitlines()[1].startswith("  scalar:")
        assert "(X 1)" in zx and "out 0" in zx
        interp.mode = NotationMode.BOTH
        both = interp.execute("show(|a>)").splitlines()
        assert both[0] == "|a> = |1>" and both[1] == "|a> =" and both[2].startswith("  scalar:")
        interp.listing = True
        listed = interp.execute("show(|a>)")
        assert "  graph:" in listed

    def test_any_state_has_a_drawing(self) -> None:
        from archytaszx.repl.printer import NotationMode

        source = "|a> = sum(x in 0..128){ 1/(x + 1) |x> }\nshow(|a>)"
        zx = Interpreter(style="ascii", mode=NotationMode.ZX).execute(source)
        assert "no ZX form" not in zx and "out 0" in zx
        raw = Interpreter(style="ascii", mode=NotationMode.RAW).execute(source)
        assert raw.splitlines()[1].startswith("  graph:") and "|a> = " not in raw

    def test_drawing_is_the_same_for_every_size(self) -> None:
        from archytaszx.repl.printer import NotationMode

        source = (
            "func f(x) = bit(x, 0)\ngate U = |x>|y> -> |x>|y xor f(x)>\n"
            "|p> = (H^@n @ I) U (H^@n @ H) (|0>^@n @ |1>)\nshow(|p>)"
        )
        drawings = []
        for head, box in (("var n\n", "n - 1"), ("var n = 6\n", "5"), ("var n = 50\n", "49")):
            out = Interpreter(style="ascii", mode=NotationMode.ZX).execute(head + source)
            drawings.append(out.split("\n", 2)[2].replace(f"- {box} -", "- B -"))
        assert (
            drawings[0].replace("-", "")
            == drawings[1].replace("-", "")
            == (drawings[2].replace("-", ""))
        )

    def test_zx_drawing_styles(self) -> None:
        from archytaszx.repl.printer import NotationMode

        for zx_style, mark in (("unicode", "Ⓧ1"), ("ascii", "(X 1)"), ("matplot", "Ⓧ1")):
            interp = Interpreter(style="unicode", mode=NotationMode.ZX, zx_style=zx_style)
            out = interp.execute("|a> = |1>\nshow(|a>)")
            assert mark in out

    def test_symbolic_state_draws_a_bang_box(self) -> None:
        from archytaszx.repl.printer import NotationMode

        interp = Interpreter(style="ascii", mode=NotationMode.ZX)
        out = interp.execute("var n\n|a> = |0>^@n\nshow(|a>)")
        assert "+-- n --+" in out and "(X)" in out


CYCLE_SUM = "gate G = [[0,0,1],[1,0,0],[0,1,0]]\n|s>_3 = sum(x in 0..3){ x |x>_3 }\n|a> = G |s>"


class TestMatrixGates:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("gate U = [[0, 1], [1, 0]]\n|a> = U |0>", "|a> = |1>"),
            ("gate C = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]\n|a> = C |1>|0>", "|a> = |1>^@2"),
            (
                "gate C = |x>|y> -> [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]\n|a> = C |1>|0>",
                "|a> = |1> @ |1>",
            ),
            ("gate G = [[0,0,1],[1,0,0],[0,1,0]]\n|a> = G |0>", "|a>_3 = |1>_3"),
            (
                CYCLE_SUM,
                "|a>_3 = 2 |0>_3 + |2>_3",
            ),
            ("|a> = [[0, 1], [1, 0]] |0>", "|a> = |1>"),
            ("|a> = [1, i]", "|a> = |0> + i |1>"),
            (
                "gate U = [[1, 2], [3, 4]]\nvar n\n|a> = U^@n |0>^@n",
                "|a> = sum(x in 0..2^n){ 3^(popcount(x)) |x> }",
            ),
            (
                "var d\ngate F = matrix(r, c in 0..d){ e^(2*pi*i*r*c/d)/sqrt(d) }\n|a>_d = F |0>_d",
                "|a>_d = 1/sqrt(d) sum(r in 0..d){ |r>_d }",
            ),
            (
                "var t: real\ngate R = [[cos(t), -sin(t)], [sin(t), cos(t)]]\n|a> = R |0>",
                "|a> = cos(t) |0> + sin(t) |1>",
            ),
        ],
    )
    def test_matrix(self, source: str, expected: str) -> None:
        assert run(source + "\nshow(|a>)")[1] == expected

    def test_entries_read_values_at_application(self) -> None:
        source = (
            "|m> = |1>\nvar z = measure|m>\nfunc f(x) = x + z\n"
            "gate P = [[f(1), 0], [0, f(2)]]\n|a> = P |1>\nshow(|a>)"
        )
        assert run(source)[1] == "|a> = 3 |1>"

    @pytest.mark.parametrize(
        ("source", "match"),
        [
            ("gate U = [[0, 1], [1, 0]]\n|a> = U |0>_3", "2 levels applied to 3"),
            ("gate U = [[0, 1], [1]]", "entries"),
            ("|a> = [1]", "at least 2"),
            ("var d\ngate F = matrix(r, c in 1..d){ r }", "starts at 0"),
        ],
    )
    def test_errors(self, source: str, match: str) -> None:
        with pytest.raises(LangError, match=match):
            run(source)


class TestFlexibleWidth:
    def test_grover_reflection(self) -> None:
        source = "var n, w\n|s> = H^@n |0>^@n\n|p> = |s> - 2 |w><w|s>\nshow(|p>)"
        assert run(source)[1] == "|p> = 2^(-n/2) sum(x in 0..2^n){ |x> } - 2^(1 - n/2) |w>"

    def test_literal_joins_a_wider_register(self) -> None:
        source = "|s> = H^@3 |0>^@3\n|k> = (|5><5|) |s>\nshow(|k>)"
        assert run(source)[1] == "|k> = sqrt(2)/4 |5>"

    def test_unresolved_width_is_an_error(self) -> None:
        with pytest.raises(LangTypeError, match="width"):
            run("var w\n|a> = |w>")
