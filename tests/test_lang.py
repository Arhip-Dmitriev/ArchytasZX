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
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from archytaszx.repl.lang import (
    Bits,
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
|psi_2> = |z[:n]> @ |f(z[:n])>
|psi_2> = |z[:n]> @ |f(z[:n])>
|psi_3> = 1/sqrt(q) sum(y in 0..q){ e^(2*pi*i*y*z[:n]/q) |y> @ |f(z[:n])> }"""

SHOR_UNICODE = """\
|ψ_0⟩ = |0⟩^⊗n ⊗ |0⟩^⊗m
|ψ_1⟩ = 1/√q Σ_{x=0}^{q - 1} |x⟩ ⊗ |0⟩^⊗m
|ψ_2⟩ = |z[:n]⟩ ⊗ |f(z[:n])⟩
|ψ_2⟩ = |z[:n]⟩ ⊗ |f(z[:n])⟩
|ψ_3⟩ = 1/√q Σ_{y=0}^{q - 1} e^{2πi·y·z[:n]/q} |y⟩ ⊗ |f(z[:n])⟩"""


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
        assert interp.execute("show(y)").splitlines()[0] == "y = z"

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
        z, y = interp.values["z"], interp.values["y"]
        assert isinstance(z, Bits) and z.width == 7 and y == z
        x, fx = int(z) >> 4, int(z) & 15
        assert fx == pow(7, x, 15)

        def reg(v: int, w: int) -> str:
            return {0: f"|0>^@{w}", 2**w - 1: f"|1>^@{w}"}.get(v, f"|{v} : {w}>")

        assert interp.execute("show(|psi_2>)") == f"|psi_2> = {reg(x, 3)} @ {reg(fx, 4)}"
        assert interp.execute("show(<psi_3|psi_3>, <psi_1|psi_1>)") == (
            "<psi_3|psi_3> = 1\n<psi_1|psi_1> = 1"
        )

    def test_display_is_numeric(self) -> None:
        out = run(numeric_shor())[1].splitlines()
        assert out[0] == "|psi_0> = |0>^@3 @ |0>^@4"
        assert out[1] == "|psi_1> = sqrt(2)/4 sum(x in 0..8){ |x> @ |0>^@4 }"
        assert not any(line.lstrip().startswith("where") for line in out)

    def test_measurement_statistics(self) -> None:
        counts: Counter[int] = Counter()
        for seed in range(400):
            interp, _ = run(numeric_shor(), seed=seed)
            z = int(interp.values["z"])  # type: ignore[call-overload]
            assert z & 15 == pow(7, z >> 4, 15)
            counts[z >> 4] += 1
        assert set(counts) == set(range(8))
        assert all(25 <= c <= 75 for c in counts.values())

    def test_seed_reproduces(self) -> None:
        a = run(numeric_shor(), seed=5)[0].values["z"]
        b = run(numeric_shor(), seed=5)[0].values["z"]
        assert a == b


# -- language ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("|a> = H|0>", "|a> = sqrt(2)/2 sum(y in 0..2){ |y> }"),
        ("|a> = H |+>", "|a> = |0>"),
        ("|a> = H|->", "|a> = |1>"),
        ("|a> = X|0> @ |1>", "|a> = |1> @ |1>"),
        ("|a> = CNOT (H|0> @ |0>)", "|a> = sqrt(2)/2 sum(y in 0..2){ |y> @ |y> }"),
        ("|a> = sum(y in 0..2){ |y> } + |1>", "|a> = |0> + 2 |1>"),
        ("|a> = 1/sqrt(2) (|0> - |1>)", "|a> = sqrt(2)/2 |0> - sqrt(2)/2 |1>"),
        ("|a> = |1>^@3", "|a> = |1>^@3"),
        ("|a> = |5>", "|a>_6 = |5>_6"),
        ("|a> = |0> + |2>", "|a>_3 = |0>_3 + |2>_3"),
        ("|a> = |0> + |1>", "|a> = |0> + |1>"),
        ("|a>_3 = |0> + |1>", "|a>_3 = |0>_3 + |1>_3"),
        ("|a> = |0>_3 + |1>", "|a>_3 = |0>_3 + |1>_3"),
        ("|a> = |5 : 3>", "|a> = |5 : 3>"),
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

    def test_measure_one_qubit_is_bool(self) -> None:
        interp, _ = run("|a> = |1>\nvar b = measure|a>")
        assert interp.values["b"] is True
        assert interp.execute("var c = b + 1\nshow(c)") == "c = 2"

    def test_bits_act_as_numbers(self) -> None:
        interp, _ = run("|a> = |5 : 3>\nvar b = measure|a>\nvar c = b * 2\nshow(b, c)")
        assert interp.execute("show(b, c)") == "b = 101 (5)\nc = 10"

    def test_symbolic_measure_of_basis_state(self) -> None:
        interp, _ = run("var n, k\n|a> = |k : n>|0>\nvar b = measure|a>")
        assert interp.execute("show(b, |a>)") == "b = 2*k\n|a> = |k> @ |0>"

    def test_measure_collapses(self) -> None:
        interp, _ = run("|a> = H|0>\nvar b = measure|a>\nvar c = measure|a>")
        assert interp.values["b"] == interp.values["c"]
        assert interp.execute("show(|a>)") in ("|a> = |0>", "|a> = |1>")


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
        assert out == "|χ⟩_d = |z⟩_d"
        z = run(source + "show(z)", "unicode")[1]
        assert z == "z ∈ {0,…,d - 1}^n\n  where z ∈ {0,…,d - 1}^n measured from |χ⟩_d"

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
            assert not interp.values["m"].is_Integer

    def test_horrible_state_runs(self) -> None:
        source = Path(HORRIBLE).read_text() + "|phi> = F^@n |psi>\nshow(|phi>)\n"
        interp, out = run(source)
        phi = interp.states["phi"]
        assert phi.widths == (22,) and phi.dims == (7,)
        assert all(t.bound for t in phi.terms)
        assert out.count("sum(x in 0..3909821048582988049){") == len(phi.terms)

    def test_digit_shift_power(self) -> None:
        source = "var d, n\ngate Xd = |x>_d -> |(x + 1) % d>_d\n|v>_d = Xd^@n |0>_d^@n\n"
        assert run(source + "show(|v>_d)")[1] == "|v>_d = |1>_d^@n"

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("|a>_3 = |2>_3", "|a>_3 = |2>_3"),
            ("|a>_3 = |2>_3 @ |1>_3", "|a>_3 = |2>_3 @ |1>_3"),
            ("|a>_3 = |5 : 2>_3", "|a>_3 = |5 : 2>_3"),
            ("gate F3 = |x>_3 -> |x>_3\n|a> = F3|1>", "|a>_3 = |1>_3"),
            ("|a>_3 = |0>_3^@2", "|a>_3 = |0>_3^@2"),
            ("gate X3 = |x>_3 -> |(x + 1) % 3>_3\n|a>_3 = X3^@4 |0 : 4>_3", "|a>_3 = |1>_3^@4"),
            (
                "gate X3 = |x>_3 -> |(x + 1) % 3>_3\n|a>_3 = (X3 @ I) |2>_3|1>_3",
                "|a>_3 = |0>_3 @ |1>_3",
            ),
            ("|a> = |1>_3 @ |0>", "|a> = |1>_3 @ |0>"),
            ("|a>_3 = sum(y in 0..3){ |y>_3 }", "|a>_3 = sum(y in 0..3){ |y>_3 }"),
            ("|a>_3 = sum(y in 0..3){ |y>_3 } + |1>_3", "|a>_3 = |0>_3 + 2 |1>_3 + |2>_3"),
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

    def test_numeric_measurement_digits(self) -> None:
        seen = set()
        for seed in range(60):
            interp, _ = run("|a>_3 = sum(y in 0..3){ |y>_3 }\nvar r = measure|a>_3", seed=seed)
            r = interp.values["r"]
            assert isinstance(r, Bits) and r.radix == (3,)
            seen.add(str(r))
        assert seen == {"0", "1", "2"}

    def test_mixed_radix_outcome(self) -> None:
        interp, _ = run("|m> = |2>_3 @ |1>\nvar k = measure|m>")
        k = interp.values["k"]
        assert isinstance(k, Bits) and k.radix == (3, 2) and str(k) == "21" and int(k) == 5

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
            ("|a>_3 = |3 : 1>_3", "does not fit"),
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
            ("|a> = |0> + |0>|0>", LangError, "different widths"),
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
            ("func f(x) = |x>", LangTypeError, "width"),
            ("|a> = |4 : 2>_2", LangTypeError, "does not fit"),
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


SHOR_SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "shor.azx"


class TestShorsScript:
    def test_symbolic(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([str(SHOR_SCRIPT), "--ascii"]) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == (
            "|psi> = 1/Q sum(x in 0..Q, k in 0..Q){ e^(-2*pi*i*k*x/Q) |k> @ |a^x % N> }"
        )
        assert "p = gcd(h - 1, N)" in out and "q = N/p" in out

    def test_numeric(self) -> None:
        source = SHOR_SCRIPT.read_text().replace("var N, a ", "var N = 15, a = 7 ", 1)
        interp, out = run(source, seed=0)
        assert interp.values["p"] * interp.values["q"] == 15
        assert out.endswith("p = 3\nq = 5")


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
            "|s> = H^@n |0>^@n @ |1 : n>\nfor j in 0..n { |s> = M |s> }\nshow(|s>)"
        )
        assert run(source)[1] == "|s> = 2^(-n/2) sum(x in 0..2^n){ |x> @ |a^x % N> }"

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

    def test_print_measured_bits_as_integer(self) -> None:
        out = run("|a> = |1>|0>|1>\nvar z = measure|a>\nprint(z)")[1]
        assert out == "5"

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
            ("var z = measure|a>", ("z", "bool", "true")),
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
            ('|a> = |1>\nvar b = measure|a>\nprint(f"b={b}")', "b=true"),
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
        assert zx.splitlines()[0] == "=" and zx.splitlines()[1].startswith("  graph:")
        interp.mode = NotationMode.BOTH
        both = interp.execute("show(|a>)").splitlines()
        assert both[0] == "|a> = |1>" and both[1] == "|a> =" and both[2].startswith("  graph:")

    def test_symbolic_state_has_no_zx_form(self) -> None:
        from archytaszx.repl.printer import NotationMode

        interp = Interpreter(style="ascii", mode=NotationMode.ZX)
        out = interp.execute("var n\n|a> = |0>^@n\nshow(|a>)")
        assert out == "|a> =\n  no ZX form (symbolic widths, dimensions or ranges)"
