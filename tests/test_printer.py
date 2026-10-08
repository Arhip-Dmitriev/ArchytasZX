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

"""Unit tests for repl/printer.py: graph listing, scalable listing, Dirac text, catalog names,
parser source, render dispatch, side-by-side layout, errors and purity."""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.compare import compare_structure
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.scalable import ScaleId, to_scalable
from archytaszx.repl.parser import parse_dirac_source
from archytaszx.repl.printer import (
    DEFAULT_MODE,
    DiracRendering,
    DiracTerm,
    NotationMode,
    PrinterDomainError,
    PrinterError,
    PrinterGrammarError,
    dirac,
    dirac_term,
    recognize,
    render,
    render_certificate,
    render_derivation,
    render_detail,
    render_dirac,
    render_graph,
    render_scalable_graph,
    render_term,
    side_by_side,
    to_dirac_source,
)
from archytaszx.rewrite.engine import RewriteResult, apply_until_fixpoint
from archytaszx.rewrite.normal_form import normal_form
from archytaszx.rewrite.rules_library import IDENTITY_REMOVAL, SPIDER_FUSION
from archytaszx.rewrite.sheet import dissolve
from archytaszx.semantics.certificate import Certificate, Derivation, DerivationKind, certify
from archytaszx.semantics.prove import InductionProof, InductionSide, ProofCertificate

from .test_phase17_oracle import BOXED, WIRED, close, inp, node, out, spider
from .test_scalable import BANG_BOX_FAMILIES, ghz, sheet_family, symbolic_phase

D = Dim("d")


def nospace(text: str | None) -> str | None:
    return None if text is None else text.replace(" ", "")


def lines_of(text: str) -> list[str]:
    return text.splitlines()


def single(gen: GeneratorType, ins: int, outs: int, phase: PhaseVector | None = None) -> Diagram:
    d = Diagram()
    node(d, gen, ins, outs, D, phase)
    return close(d)


def listing_example() -> Diagram:
    """The spec's graph-listing example shape."""
    d = Diagram()
    n0 = node(d, Z_SPIDER, 0, 2)
    n1 = node(
        d,
        X_SPIDER,
        1,
        1,
        phase=PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3)), 2: Phase.symbol("theta")}),
    )
    n2 = node(d, Z_SPIDER, 0, 1)
    d.add_wire(out(n0, 0), inp(n1, 0))
    close(d)
    d.add_bang_box(Mult("n"), port_scope=frozenset({out(n0, 1)}))
    outer = d.add_bang_box(Mult("m"), node_scope=frozenset({n1, n2}))
    d.add_bang_box(Mult("k"), node_scope=frozenset({n2}), parent=outer)
    d.set_parameters({"d": 3})
    return d


def chain_results() -> tuple[Diagram, list[RewriteResult]]:
    d = Diagram()
    a = node(d, Z_SPIDER, 0, 2)
    b = node(d, Z_SPIDER, 1, 1)
    c = node(d, Z_SPIDER, 1, 2)
    d.add_wire(out(a, 0), inp(b, 0))
    d.add_wire(out(b, 0), inp(c, 0))
    close(d)
    results: list[RewriteResult] = []
    apply_until_fixpoint(d.copy(), [IDENTITY_REMOVAL, SPIDER_FUSION], on_result=results.append)
    return d, results


# -- graph listing ------------------------------------------------------------------------


class TestGraphListing:
    def test_spec_example(self) -> None:
        text = render_graph(listing_example())
        ls = lines_of(text)
        assert ls[0] == "graph: 3 nodes, 1 wire, 0 inputs -> 3 outputs"
        assert "  scalar: 1" in ls
        assert "  parameters: d = 3" in ls
        assert "  nodes:" in ls
        assert "    n0  Z  in: []  out: [d, d!0]  phase[d]: 0" in ls
        n1 = next(line for line in ls if line.startswith("    n1  X"))
        assert "in: [d]  out: [d]  phase[d]: {1: 1/3, 2: theta} turns" in n1
        assert n1.endswith("boxes: !1")
        n2 = next(line for line in ls if line.startswith("    n2  Z"))
        assert n2.endswith("boxes: !1 > !2")
        assert "  wires:" in ls
        assert "    n0.out0 -- n1.in0" in ls
        assert "  inputs: []" in ls
        assert "  outputs: [n0.out1, n1.out0, n2.out0]" in ls
        boxes = ls[ls.index("  bang boxes:") :]
        assert boxes[1:4] == [
            "    !0  x n  ports: n0.out1",
            "    !1  x m  nodes: n1, n2",
            "      !2  x k  nodes: n2",
        ]

    def test_plain_diagram(self) -> None:
        ls = lines_of(render_graph(single(Z_SPIDER, 1, 2)))
        assert ls[0] == "graph: 1 node, 0 wires, 1 input -> 2 outputs"
        assert "  parameters: none" in ls
        assert "  bang boxes: none" in ls
        assert "  inputs: [n0.in0]" in ls
        assert "  outputs: [n0.out0, n0.out1]" in ls

    def test_scalar_and_mixed_dims(self) -> None:
        d = Diagram()
        d.add_node(DIM_BINDER, [Dim(2), Dim(3)], [Dim(6)])
        close(d)
        d.multiply_scalar(Scalar.rational(3, 2))
        ls = lines_of(render_graph(d))
        scalar = next(line for line in ls if line.startswith("  scalar:"))
        assert "3/2" in scalar
        b = next(line for line in ls if line.startswith("    n0  B"))
        assert "in: [2, 3]  out: [6]" in b

    def test_multiple_parameters_sorted(self) -> None:
        d = parse_dirac_source("sum_{k=0}^{3-1} |k>^{2}")
        line = next(line for line in lines_of(render_graph(d)) if "parameters:" in line)
        assert line.index("d = 3") < line.index("n = 2")

    def test_multiple_symbols_and_nested_forest(self) -> None:
        text = render_graph(BANG_BOX_FAMILIES["nested_copies"]())
        ls = lines_of(text)
        boxes = ls[ls.index("  bang boxes:") + 1 :]
        assert boxes[0].startswith("    !0  x k1  nodes:")
        assert boxes[1].startswith("      !1  x k2  nodes:")
        compound = lines_of(render_graph(BANG_BOX_FAMILIES["compound"]()))
        assert "    !0  x 2*k  nodes: n0" in compound
        assert "    !1  x k + 1  ports: n1.out0" in compound

    def test_never_raises_on_invalid(self) -> None:
        d = Diagram()
        node(d, Z_SPIDER, 1, 1)
        node(d, X_SPIDER, 0, 0, Dim(2), None)
        d.add_bang_box(Mult("n"), port_scope=frozenset({out(NodeId(0), 0)}))
        text = render_graph(d)
        assert text.startswith("graph:")

    def test_deterministic_across_insertion_order(self) -> None:
        def build(flip: bool) -> Diagram:
            d = Diagram()
            a = node(d, Z_SPIDER, 0, 3)
            b = node(d, X_SPIDER, 2, 1)
            c = node(d, Z_SPIDER, 1, 1)
            wires = [(out(a, 0), inp(b, 0)), (out(a, 1), inp(b, 1)), (out(b, 0), inp(c, 0))]
            for w in reversed(wires) if flip else wires:
                d.add_wire(*(reversed(w) if flip else w))
            close(d)
            scope = [b, c]
            d.add_bang_box(Mult("m"), node_scope=frozenset(reversed(scope) if flip else scope))
            return d

        first, second = build(False), build(True)
        assert render_graph(first) == render_graph(second)
        assert render(first, NotationMode.ZX) == render(second, NotationMode.ZX)
        assert dirac_term(first) == dirac_term(second)


_SCRIPT = """
from tests.test_scalable import BANG_BOX_FAMILIES
from archytaszx.diagram.scalable import to_scalable
from archytaszx.repl.printer import NotationMode, render, render_scalable_graph
for name in sorted(BANG_BOX_FAMILIES):
    d = BANG_BOX_FAMILIES[name]()
    print(render(d, NotationMode.ZX))
    print(render_scalable_graph(to_scalable(d)))
"""


def _run(seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_cross_process_determinism() -> None:
    first = _run("0")
    assert "graph:" in first
    assert _run("4242") == first


# -- scalable listing ---------------------------------------------------------------------


class TestScalableListing:
    def test_sheet_family(self) -> None:
        ls = lines_of(render_scalable_graph(sheet_family()))
        assert ls[0].startswith("scalable:")
        assert any(line.startswith("  scalar:") for line in ls)
        assert any(line.startswith("  parameters:") for line in ls)
        assert "  scales:" in ls
        s0 = next(line for line in ls if line.strip().startswith("s0  COPIES x k"))
        s1 = next(line for line in ls if line.strip().startswith("s1  LEGS x m"))
        s2 = next(line for line in ls if line.strip().startswith("s2  LEGS x n"))
        assert _indent(s1) > _indent(s0) == _indent(s2)
        text = "\n".join(ls)
        assert "2~s1" in text and "2~s2" in text
        assert "scale: s0" in text
        assert "{s0: " in text

    def test_listing_never_contains_graph_prefix(self) -> None:
        assert not render_scalable_graph(to_scalable(ghz())).startswith("graph:")


# -- Dirac text -----------------------------------------------------------------------------


class TestDiracText:
    def test_rendering_fields(self) -> None:
        r = dirac(single(Z_SPIDER, 0, 2))
        assert isinstance(r, DiracRendering)
        assert isinstance(r.term, DiracTerm)
        assert r.name == "GHZ_{2}"
        assert r.text == "GHZ_{2} = sum_{k=0}^{d-1} |k, k>"
        assert r.source is not None and r.source.replace(" ", "") == "sum_{k=0}^{d-1}|k,k>"
        assert r.instance_text is None

    def test_unnamed_has_no_equals(self) -> None:
        d = Diagram()
        a = node(d, Z_SPIDER, 0, 2)
        b = node(d, X_SPIDER, 1, 1)
        d.add_wire(out(a, 0), inp(b, 0))
        close(d)
        r = dirac(d)
        assert r.name is None
        assert r.text == render_term(r.term)
        assert " = " not in r.text.split("sum", 1)[0]

    def test_scalar_prefix(self) -> None:
        d = single(Z_SPIDER, 1, 1)
        d.multiply_scalar(Scalar.rational(1, 2))
        assert render_term(dirac_term(d)).startswith("(1/2) ")

    def test_empty_diagram(self) -> None:
        assert render_term(dirac_term(Diagram())) == "1"
        assert recognize(Diagram()) is None

    def test_x_spider_text(self) -> None:
        text = render_term(dirac_term(single(X_SPIDER, 1, 1)))
        assert "w_{d}^{q" in text
        assert "d^{-" in text
        assert re.search(r"\|j\w*><j\w*\|$", text), text

    def test_fourier_and_triangles(self) -> None:
        assert "d^{-1/2} w_{d}^{" in render_term(dirac_term(single(FOURIER_BOX, 1, 1)))
        assert " or " in render_term(dirac_term(single(TRIANGLE, 1, 1)))
        assert ">= 1]" in render_term(dirac_term(single(TRIANGLE_INVERSE, 1, 1)))
        assert "nnz(" in render_term(dirac_term(single(W_NODE, 1, 2)))
        d = Diagram()
        d.add_node(DIM_SPLITTER, [Dim(6)], [Dim(2), Dim(3)])
        assert "*3 + " in render_term(dirac_term(close(d))).replace(" * ", "*")

    def test_family_and_product_syntax(self) -> None:
        text = render_term(dirac_term(symbolic_phase()))
        assert "prod_{" in text
        assert re.search(r"\[i\d+\]", text)
        assert re.search(r"i\d+=1\.\.k", text)
        assert text.startswith("(3) sum_{k0[i0]=0, i0=1..k}^{3-1} ")

    def test_z_fan_and_non_z_fan(self) -> None:
        assert render_term(dirac_term(ghz())).endswith("|k>^{n}")
        d = single(X_SPIDER, 0, 1)
        d.add_bang_box(Mult("n"), port_scope=frozenset({out(NodeId(0), 0)}))
        assert re.search(r"\(x\)_\{i\d+=1\}\^\{n\} \|j\w*\[i\d+\]>", render_term(dirac_term(d)))

    def test_unicode_tokens(self) -> None:
        d = single(X_SPIDER, 1, 1, PhaseVector(D, {1: Phase.symbol("theta")}))
        d.add_bang_box(Mult("n"), port_scope=frozenset({out(NodeId(0), 0)}))
        ascii_text = render_term(dirac_term(d))
        uni = render_term(dirac_term(d), unicode=True)
        assert "Σ" in uni and "⊗" in uni and "ω" in uni and "π" in uni
        assert "⟩" in uni and "⟨" in uni
        assert "sum_" not in uni and "(x)" not in uni and " pi " not in uni
        assert "sum_" in ascii_text and "Σ" not in ascii_text
        prod = render_term(dirac_term(symbolic_phase()), unicode=True)
        assert "∏" in prod and "prod_" not in prod
        assert "Σ" in render_dirac(single(Z_SPIDER, 0, 2), unicode=True)

    def test_instance_line(self) -> None:
        d = parse_dirac_source("sum_{k=0}^{3-1} |k>^{2}")
        ls = lines_of(render_dirac(d, instance=True))
        assert len(ls) == 2
        assert ls[1].startswith("at d = 3, n = 2: ")
        assert "n" not in ls[1].split(": ", 1)[1].replace("nnz", "")
        r = dirac(d)
        assert r.instance_text is not None and "{2}" in r.instance_text
        assert len(lines_of(render_dirac(d))) == 1

    def test_render_term_env(self) -> None:
        term = dirac_term(ghz())
        assert render_term(term, env={"n": 3}).endswith("|k>^{3}")


# -- catalog names --------------------------------------------------------------------------

PHASED = PhaseVector(D, {1: Phase.symbol("theta")})
CATALOG = [
    (Z_SPIDER, 0, 2, None, "GHZ_{2}"),
    (Z_SPIDER, 0, 3, None, "GHZ_{3}"),
    (Z_SPIDER, 0, 1, None, "plus"),
    (Z_SPIDER, 2, 0, None, "GHZ_{2}^dagger"),
    (Z_SPIDER, 1, 0, None, "plus^dagger"),
    (Z_SPIDER, 1, 1, None, "id"),
    (Z_SPIDER, 1, 2, None, "copy_{2}"),
    (Z_SPIDER, 1, 3, None, "copy_{3}"),
    (Z_SPIDER, 2, 1, None, "copy_{2}^dagger"),
    (Z_SPIDER, 2, 2, None, "Z_{2->2}"),
    (Z_SPIDER, 0, 0, None, "Z_{0->0}"),
    (Z_SPIDER, 1, 1, PHASED, "Z_phase"),
    (X_SPIDER, 1, 1, None, "id"),
    (X_SPIDER, 0, 1, None, "zero"),
    (X_SPIDER, 1, 2, None, "X_{1->2}"),
    (X_SPIDER, 2, 0, None, "X_{2->0}"),
    (FOURIER_BOX, 1, 1, None, "F"),
    (TRIANGLE, 1, 1, None, "triangle"),
    (TRIANGLE_INVERSE, 1, 1, None, "triangle^-1"),
    (W_NODE, 1, 3, None, "W_{3}"),
    (W_NODE, 1, 0, None, "W_{0}"),
]


class TestRecognize:
    @pytest.mark.parametrize(("gen", "m", "n", "phase", "name"), CATALOG)
    def test_catalog(
        self, gen: GeneratorType, m: int, n: int, phase: PhaseVector | None, name: str
    ) -> None:
        assert recognize(single(gen, m, n, phase)) == name

    @pytest.mark.parametrize(("gen", "prefix"), [(Z_SPIDER, "Z_{0->2}("), (X_SPIDER, "X_{1->1}(")])
    def test_phased(self, gen: GeneratorType, prefix: str) -> None:
        shape = (0, 2) if gen is Z_SPIDER else (1, 1)
        name = recognize(single(gen, *shape, PHASED))
        assert name is not None and name.startswith(prefix) and name.endswith(")")

    def test_bind_split(self) -> None:
        d = Diagram()
        d.add_node(DIM_BINDER, [Dim(2), Dim(3)], [Dim(6)])
        assert recognize(close(d)) == "bind"
        d = Diagram()
        d.add_node(DIM_SPLITTER, [Dim(6)], [Dim(2), Dim(3)])
        assert recognize(close(d)) == "split"

    def test_symbolic_count(self) -> None:
        d = single(Z_SPIDER, 0, 3)
        d.add_bang_box(Mult("n"), port_scope=frozenset({out(NodeId(0), 1)}))
        name = recognize(d)
        assert name is not None and name.replace(" ", "") == "GHZ_{n+2}"

    def test_tensor_product(self) -> None:
        d = Diagram()
        node(d, Z_SPIDER, 0, 2)
        node(d, X_SPIDER, 1, 1)
        close(d)
        assert recognize(d) == "GHZ_{2} (x) id"
        assert render_dirac(d).startswith("GHZ_{2} (x) id = ")
        assert "⊗" in render_dirac(d, unicode=True).split("=")[0]

    def test_non_contiguous_boundary(self) -> None:
        d = Diagram()
        a = node(d, Z_SPIDER, 0, 2)
        b = node(d, Z_SPIDER, 0, 2)
        d.set_boundary_outputs([out(a, 0), out(b, 0), out(a, 1), out(b, 1)])
        assert recognize(d) is None

    def test_unnameable(self) -> None:
        loop = Diagram()
        z = node(loop, Z_SPIDER, 1, 2)
        loop.add_wire(out(z, 1), inp(z, 0))
        close(loop)
        assert recognize(loop) is None
        assert recognize(BANG_BOX_FAMILIES["fusion_pair"]()) is None
        assert recognize(symbolic_phase()) is None
        assert recognize(BANG_BOX_FAMILIES["plain"]()) is None


# -- parser source ----------------------------------------------------------------------------


class TestToDiracSource:
    def test_states(self) -> None:
        assert nospace(to_dirac_source(single(Z_SPIDER, 0, 2))) == "sum_{k=0}^{d-1}|k,k>"
        assert nospace(to_dirac_source(single(Z_SPIDER, 0, 1))) == "sum_{k=0}^{d-1}|k>"
        d = Diagram()
        node(d, Z_SPIDER, 0, 2, Dim(3))
        assert nospace(to_dirac_source(close(d))) == "sum_{k=0}^{3-1}|k,k>"

    def test_boxed(self) -> None:
        d = parse_dirac_source("sum_{k=0}^{d-1} |k>^{4}")
        assert to_dirac_source(d) == "sum_{k=0}^{d-1} |k>^{4}"
        assert to_dirac_source(ghz()) is None

    @pytest.mark.parametrize(
        "build",
        [
            lambda: single(X_SPIDER, 0, 2),
            lambda: single(Z_SPIDER, 1, 2),
            lambda: single(Z_SPIDER, 0, 2, PHASED),
            lambda: single(Z_SPIDER, 0, 0),
            lambda: BANG_BOX_FAMILIES["fusion_pair"](),
            lambda: BANG_BOX_FAMILIES["plain"](),
            lambda: BANG_BOX_FAMILIES["nested_port_in_copies"](),
            Diagram,
        ],
    )
    def test_none(self, build: object) -> None:
        assert to_dirac_source(build()) is None  # type: ignore[operator]

    def test_scalar_blocks_source(self) -> None:
        d = single(Z_SPIDER, 0, 2)
        d.multiply_scalar(Scalar.rational(2))
        assert to_dirac_source(d) is None


# -- render dispatch ------------------------------------------------------------------------


def _states() -> dict[str, object]:
    initial, results = chain_results()
    cert = certify(initial, results)
    proof = ProofCertificate(
        start=initial,
        goal=results[-1].diagram,
        forward=cert,
        backward=certify(results[-1].diagram, []),
        moves=tuple(r.step.rule_name for r in results),
        backward_moves=(),
    )
    unit_ghz = ghz()
    unit_ghz.set_bang_box_multiplicity(BangBoxId(0), Mult(1))
    return {
        "diagram": initial,
        "scalable": sheet_family(),
        "normal_form": normal_form(initial),
        "rewrite_result": results[0],
        "derivation": cert.derivation,
        "certificate": cert,
        "proof": proof,
        "sheet_step": dissolve(to_scalable(unit_ghz), ScaleId(0)),
    }


@pytest.fixture(scope="module")
def states() -> dict[str, object]:
    return _states()


GRAPH_PREFIXES = ("graph:", "scalable:")


class TestRender:
    def test_default_mode(self) -> None:
        assert DEFAULT_MODE is NotationMode.DIRAC
        assert NotationMode.DIRAC.value == "dirac" and NotationMode.ZX.value == "zx"

    @pytest.mark.parametrize(
        "name",
        [
            "diagram",
            "scalable",
            "normal_form",
            "rewrite_result",
            "derivation",
            "certificate",
            "proof",
            "sheet_step",
        ],
    )
    def test_both_modes(self, states: dict[str, object], name: str) -> None:
        state = states[name]
        dirac_text = render(state)
        assert dirac_text.strip()
        assert not any(line.lstrip().startswith(GRAPH_PREFIXES) for line in lines_of(dirac_text))
        assert "sum_{" in dirac_text
        zx_text = render(state, NotationMode.ZX)
        assert any(line.lstrip().startswith(GRAPH_PREFIXES) for line in lines_of(zx_text))
        assert "sum_{" in zx_text
        uni = render(state, unicode=True)
        assert "Σ" in uni

    def test_derivation_lines(self) -> None:
        initial, results = chain_results()
        text = render_derivation(initial, results)
        assert lines_of(text)[0].startswith("state 0:")
        assert re.search(r"step 1: \S+ \(consumed .*new .*scalar .*\)", text), text
        assert f"state {len(results)}:" in text

    def test_certificate_reapplies(self) -> None:
        initial, results = chain_results()
        cert = certify(initial, results, label="chain")
        text = render_certificate(cert)
        assert "chain" in text
        for i in range(len(results) + 1):
            assert f"state {i}:" in text
        assert "unavailable" not in text
        zx = render_certificate(cert, NotationMode.ZX)
        assert "graph:" in zx

    def test_certificate_failed_reapplication(self) -> None:
        _initial, results = chain_results()
        other = single(X_SPIDER, 1, 1)
        bad = Certificate(
            Derivation(
                kind=DerivationKind.STEP_SEQUENCE,
                initial=other,
                final=results[-1].diagram,
                steps=tuple(r.step for r in results),
            )
        )
        text = render_certificate(bad)
        assert re.search(r"state \d+: unavailable \(.+\)", text), text
        assert "state 0:" in text

    @pytest.mark.parametrize("bad", [42, "diagram", None, object()])
    def test_wrong_type(self, bad: object) -> None:
        with pytest.raises(PrinterGrammarError):
            render(bad)

    def test_bad_mode(self) -> None:
        with pytest.raises(PrinterGrammarError):
            render(single(Z_SPIDER, 0, 1), "zx")  # type: ignore[arg-type]


# -- layout -----------------------------------------------------------------------------------


class TestLayout:
    def test_side_by_side(self) -> None:
        text = side_by_side("a\nbbb", "x\ny\nz", gap=4)
        assert [line.rstrip() for line in lines_of(text)] == ["a      x", "bbb    y", "       z"]

    def test_side_by_side_left_longer(self) -> None:
        text = side_by_side("aa\nb\nc", "xyz", gap=2)
        assert [line.rstrip() for line in lines_of(text)] == ["aa  xyz", "b", "c"]

    def test_side_by_side_column_alignment(self) -> None:
        left = render_graph(listing_example())
        right = "R1\nR2\nR3"
        ls = lines_of(side_by_side(left, right))
        width = max(len(line) for line in lines_of(left))
        for i, label in enumerate(("R1", "R2", "R3")):
            assert ls[i][width + 4 :] == label
            assert ls[i][:width].rstrip() == lines_of(left)[i]

    def test_render_detail(self) -> None:
        d = single(Z_SPIDER, 0, 3)
        ls = lines_of(render_detail(d))
        assert ls[0].startswith("graph:")
        assert ls[0].endswith(render_dirac(d))
        assert render_detail(d) == side_by_side(render_graph(d), render_dirac(d))
        assert "Σ" in render_detail(d, unicode=True)


# -- errors and purity ------------------------------------------------------------------------


def invalid_diagram() -> Diagram:
    d = Diagram()
    node(d, Z_SPIDER, 1, 1)
    return d


def fixed_arity_fan() -> Diagram:
    d = single(FOURIER_BOX, 1, 1)
    d.add_bang_box(Mult("n"), port_scope=frozenset({out(NodeId(0), 0)}))
    return d


class TestErrors:
    def test_hierarchy(self) -> None:
        assert issubclass(PrinterDomainError, PrinterError)
        assert issubclass(PrinterGrammarError, PrinterError)

    @pytest.mark.parametrize(
        "call",
        [dirac_term, recognize, render_graph, to_dirac_source, render_dirac, dirac, render_detail],
        ids=lambda f: f.__name__,
    )
    @pytest.mark.parametrize("bad", [None, 3, "x"])
    def test_grammar(self, call: object, bad: object) -> None:
        with pytest.raises(PrinterGrammarError):
            call(bad)  # type: ignore[operator]

    def test_grammar_other(self) -> None:
        with pytest.raises(PrinterGrammarError):
            render_term("x")  # type: ignore[arg-type]
        with pytest.raises(PrinterGrammarError):
            render_scalable_graph(Diagram())  # type: ignore[arg-type]
        with pytest.raises(PrinterGrammarError):
            side_by_side(1, "x")  # type: ignore[arg-type]
        with pytest.raises(PrinterGrammarError):
            render_certificate(Diagram())  # type: ignore[arg-type]
        with pytest.raises(PrinterGrammarError):
            render_derivation(Diagram(), [3])  # type: ignore[list-item]

    @pytest.mark.parametrize("build", [invalid_diagram, fixed_arity_fan])
    def test_domain(self, build: object) -> None:
        d = build()  # type: ignore[operator]
        for call in (dirac_term, render_dirac, dirac):
            with pytest.raises(PrinterDomainError):
                call(d)
        assert render_graph(d).startswith("graph:")

    def test_no_foreign_exception_on_invalid(self) -> None:
        d = invalid_diagram()
        for call in (recognize, to_dirac_source):
            with pytest.raises(PrinterDomainError):
                call(d)


class TestPurity:
    @pytest.mark.parametrize("name", sorted(BANG_BOX_FAMILIES))
    def test_no_mutation(self, name: str) -> None:
        d = BANG_BOX_FAMILIES[name]()
        before = d.copy()
        boxes = dict(d.bang_boxes)
        params = dict(d.parameters)
        dirac_term(d)
        recognize(d)
        to_dirac_source(d)
        render_graph(d)
        render_dirac(d, instance=True)
        render(d, NotationMode.ZX)
        render_detail(d)
        assert compare_structure(before, d).identical
        assert dict(d.bang_boxes) == boxes
        assert dict(d.parameters) == params
        assert d.scalar == before.scalar


class TestCoreNamesAndEdges:
    def test_lone_z_index_avoids_dim_named_k(self) -> None:
        d = Diagram()
        node(d, Z_SPIDER, 0, 2, Dim("k"))
        assert render_term(dirac_term(close(d))) == "sum_{k'=0}^{k-1} |k', k'>"

    def test_generated_names_avoid_diagram_symbols(self) -> None:
        d = Diagram()
        z = node(d, Z_SPIDER, 0, 1, Dim("k0"))
        x = node(d, X_SPIDER, 1, 1, Dim("k0"), PhaseVector(Dim("k0"), {1: Phase.symbol("q1")}))
        d.add_wire(out(z, 0), inp(x, 0))
        close(d)
        d.add_bang_box(Mult("i0"), node_scope=frozenset({x}))
        term = dirac_term(d)
        taken = {"k0", "q1", "i0"}
        names = {v.name for v in term.indices} | {c.name for v in term.indices for c in v.copies}
        assert not names & taken
        assert "i0'=1..i0" in render_term(term)

    def test_legless_phase_free_x_renders_one(self) -> None:
        d = Diagram()
        node(d, X_SPIDER, 0, 0)
        assert render_term(dirac_term(d)) == "sum_{q0=0}^{d-1} 1"

    @pytest.mark.parametrize("env", ["x", {"d": "abc"}, {"d": 2.5}, {1: 2}, {"d": True}])
    def test_render_term_env_type(self, env: object) -> None:
        with pytest.raises(PrinterGrammarError):
            render_term(dirac_term(ghz()), env=env)  # type: ignore[arg-type]


class TestListingModesRegressions:
    def test_parent_cycles_and_self_parent_kept(self) -> None:
        d = single(Z_SPIDER, 0, 1)
        n0 = NodeId(0)
        d.add_bang_box(Mult("a"), node_scope=frozenset({n0}), parent=BangBoxId(1))
        d.add_bang_box(Mult("b"), node_scope=frozenset({n0}), parent=BangBoxId(0))
        d.add_bang_box(Mult("c"), node_scope=frozenset({n0}), parent=BangBoxId(2))
        d.add_bang_box(Mult("e"), node_scope=frozenset({n0}), parent=BangBoxId(9))
        ls = lines_of(render_graph(d))
        boxes = ls[ls.index("  bang boxes:") + 1 :]
        assert boxes == [
            "    !3  x e  nodes: n0  parent: !9 (missing)",
            "    !0  x a  nodes: n0  parent: !1 (cycle)",
            "      !1  x b  nodes: n0",
            "    !2  x c  nodes: n0  parent: !2 (cycle)",
        ]

    def test_deep_nesting_no_recursion_error(self) -> None:
        d = single(Z_SPIDER, 0, 1)
        parent = None
        for _ in range(3000):
            parent = d.add_bang_box(Mult(1), node_scope=frozenset({NodeId(0)}), parent=parent)
        ls = lines_of(render_graph(d))
        assert ls[-1].startswith("  " * 3001 + "!2999  x 1")

    def test_zero_phase_vector_distinct_from_none(self) -> None:
        texts = []
        for phase in (None, PhaseVector(D), PhaseVector(Dim(3))):
            d = Diagram()
            d.add_node(Z_SPIDER, [], [], phase)
            texts.append(next(line for line in lines_of(render_graph(d)) if "n0  Z" in line))
        assert texts == [
            "    n0  Z  in: []  out: []",
            "    n0  Z  in: []  out: []  phase[d]: 0",
            "    n0  Z  in: []  out: []  phase[3]: 0",
        ]

    def test_non_int_node_ids_listed(self) -> None:
        d = Diagram()
        d.add_wire(PortRef("x", Direction.OUTPUT, 0), PortRef(NodeId(1), Direction.INPUT, 0))  # type: ignore[arg-type]
        assert "    n1.in0 -- n'x'.out0" in lines_of(render_graph(d))

    def _chain(self) -> tuple[Diagram, list[RewriteResult]]:
        d = Diagram()
        ns = [node(d, Z_SPIDER, 0, 2)] + [node(d, Z_SPIDER, 1, 2) for _ in range(3)]
        for a, b in itertools.pairwise(ns):
            d.add_wire(out(a, 0), inp(b, 0))
        close(d)
        results: list[RewriteResult] = []
        apply_until_fixpoint(d.copy(), [SPIDER_FUSION], on_result=results.append)
        assert len(results) == 3
        return d, results

    def test_replay_divergence_and_last_step_failure_reported(self) -> None:
        d, results = self._chain()
        steps = tuple(r.step for r in results)
        skipped = Certificate(
            Derivation(DerivationKind.STEP_SEQUENCE, d, results[-1].diagram, steps[1:])
        )
        text = render_certificate(skipped)
        assert "state 1: unavailable (step 1 (spider_fusion) failed: the replayed step" in text
        assert "state 2: recorded final (replay step 1" in text
        last = Certificate(
            Derivation(DerivationKind.STEP_SEQUENCE, d, results[-1].diagram, steps[::2])
        )
        text = render_certificate(last)
        assert re.search(r"state 1:\n", text)
        assert "state 2: recorded final (replay step 2 (spider_fusion) failed: " in text

    def test_zero_step_derivation_shows_differing_final(self) -> None:
        d, results = self._chain()
        cert = Certificate(Derivation(DerivationKind.STEP_SEQUENCE, d, results[-1].diagram))
        text = render_certificate(cert)
        assert "    final:\n      GHZ_{5} = " in text
        assert "final:" not in render_certificate(certify(d, []))

    def test_normal_form_renders_every_state(self) -> None:
        d, _results = self._chain()
        nf = normal_form(d)
        text = render(nf)
        assert len(nf.results) == 3
        for i in range(1, len(nf.results) + 1):
            assert f"\n    state {i}:\n      " in text

    def test_induction_proof_rendered(self) -> None:
        d, results = self._chain()
        cert = certify(d, results)
        piece = ProofCertificate(
            start=d,
            goal=results[-1].diagram,
            forward=cert,
            backward=certify(results[-1].diagram, []),
            moves=("normalize",),
            backward_moves=(),
        )
        proof = InductionProof(
            start=d,
            goal=results[-1].diagram,
            index="n",
            step_symbol="k",
            base=piece,
            side=InductionSide.START,
            expose=cert,
            region=frozenset({NodeId(1)}),
            step=piece,
        )
        text = render(proof)
        assert text.startswith("induction proof: over n, step symbol k, side start, region n1")
        assert "  base:\n    proof:" in text and "  expose:\n    certificate:" in text
        assert "graph:" not in text and "graph:" in render(proof, NotationMode.ZX)
        assert render_detail(proof) == render_detail(results[-1].diagram)

    def test_side_by_side_wide_chars_and_line_breaks(self) -> None:
        assert side_by_side("日本\nab", "X\nY", gap=2) == "日本  X\nab    Y"
        assert side_by_side("e\u0301\nab", "X\nY", gap=1) == "e\u0301  X\nab Y"
        assert side_by_side("a\r\nb", "x", gap=1) == "a x\nb"
        assert side_by_side("", "") == ""


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


GOLDEN_TEXT: dict[str, tuple[Callable[[], Diagram], str]] = {
    "T-Ti": (
        lambda: WIRED["T-Ti"](),
        (
            "sum_{j0i0,j0o0,j1o0=0}^{d-1} [j0o0 = 0 or j0o0 = j0i0] "
            "([j1o0 = j0o0] - [j1o0 = 0][j0o0 >= 1]) |j1o0><j0i0|"
        ),
    ),
    "F": (
        lambda: spider(FOURIER_BOX, 1, 1)(),
        "sum_{j0i0,j0o0=0}^{d-1} d^{-1/2} w_{d}^{j0o0 j0i0} |j0o0><j0i0|",
    ),
    "x_self_loop": (
        lambda: WIRED["x_self_loop"](),
        (
            "sum_{q0,j0i0,j0i1,j0o1=0}^{d-1} d^{-2} e^{2 pi i (phi [q0=1])} "
            "w_{d}^{q0 (j0i1 + j0o1 - j0i0 - j0i1)} |j0o1><j0i0|"
        ),
    ),
    "double_wire": (
        lambda: WIRED["double_wire"](),
        "sum_{k0,q1,j1o0=0}^{d-1} d^{-3/2} w_{d}^{q1 (j1o0 - k0 - k0)} |k0, j1o0>",
    ),
    "sb_chain": (
        lambda: WIRED["sb_chain"](),
        (
            "sum_{k2=0}^{2-1} sum_{j0i0=0}^{6-1} sum_{j0o1=0}^{3-1} sum_{j1o0=0}^{6-1} "
            "[j0i0 = k2*3 + j0o1] [j1o0 = j0o1*2 + k2] |j1o0><j0i0|"
        ),
    ),
    "W-Z": (
        lambda: WIRED["W-Z"](),
        "sum_{k1,j0i0,j0o1=0}^{d-1} [j0i0 = k1 + j0o1][nnz(k1, j0o1) <= 1] |j0o1, k1><j0i0|",
    ),
    "Z-Z": (
        lambda: WIRED["Z-Z"](),
        "sum_{k0,k1=0}^{d-1} e^{2 pi i (theta [k0=1])} [k0 = k1] |k0, k1, k1>",
    ),
    "disconnected": (
        lambda: WIRED["disconnected"](),
        (
            "sum_{k1,q0,j0i0,j0o0,j2i0,j2o0,j2o1=0}^{d-1} d^{-1} w_{d}^{q0 (j0o0 - j0i0)} "
            "[j2i0 = j2o0 + j2o1][nnz(j2o0, j2o1) <= 1] |j2o1, j2o0, k1, k1, j0o0><j0i0, j2i0|"
        ),
    ),
    "port_x": (
        lambda: BOXED["port_x"](),
        (
            "sum_{q0,j0i0=0}^{d-1} sum_{j0o0[i0]=0, i0=1..n}^{d-1} sum_{j0o1=0}^{d-1} "
            "d^{-(n+2)/2} e^{2 pi i (theta [q0=1])} "
            "w_{d}^{q0 (sum_{i0=1}^{n} j0o0[i0] + j0o1 - j0i0)} "
            "(x)_{i0=1}^{n} |j0o0[i0]> |j0o1><j0i0|"
        ),
    ),
    "port_x_in": (
        lambda: BOXED["port_x_in"](),
        (
            "sum_{q0=0}^{d-1} sum_{j0i0[i0]=0, i0=1..n}^{d-1} sum_{j0o0=0}^{d-1} d^{-(n+1)/2} "
            "e^{2 pi i (theta [q0=1])} w_{d}^{q0 (j0o0 - sum_{i0=1}^{n} j0i0[i0])} "
            "|j0o0>(x)_{i0=1}^{n} <j0i0[i0]|"
        ),
    ),
    "port_w": (
        lambda: BOXED["port_w"](),
        (
            "sum_{j0i0,j0o0=0}^{d-1} sum_{j0o1[i0]=0, i0=1..n}^{d-1} "
            "[j0i0 = j0o0 + sum_{i0=1}^{n} j0o1[i0]][nnz(j0o0, (j0o1[i0])_{i0=1}^{n}) <= 1] "
            "|j0o0> (x)_{i0=1}^{n} |j0o1[i0]><j0i0|"
        ),
    ),
    "nested_two_index": (
        lambda: BOXED["nested_two_index"](),
        (
            "sum_{k0[i0]=0, i0=1..n}^{d-1} sum_{k1[i0,i1]=0, i0=1..n, i1=1..m}^{d-1} "
            "prod_{i0=1}^{n} prod_{i1=1}^{m}( "
            "e^{2 pi i (phi [k1[i0,i1]=1])} [k0[i0] = k1[i0,i1]] ) "
            "(x)_{i0=1}^{n}( |k0[i0]> (x)_{i1=1}^{m}( |k1[i0,i1]> ) )"
        ),
    ),
    "hub_x_to_x": (
        lambda: BOXED["hub_x_to_x"](),
        (
            "sum_{q0=0}^{d-1} sum_{q1[i0]=0, i0=1..n}^{d-1} sum_{j0i0=0}^{d-1} "
            "sum_{j0o0[i0],j1o0[i0]=0, i0=1..n}^{d-1} d^{-(n+1)/2} "
            "w_{d}^{q0 (sum_{i0=1}^{n} j0o0[i0] - j0i0)} "
            "prod_{i0=1}^{n}( d^{-1} w_{d}^{q1[i0] (j1o0[i0] - j0o0[i0])} ) "
            "(x)_{i0=1}^{n}( |j1o0[i0]> )<j0i0|"
        ),
    ),
    "hub_z_to_z": (
        lambda: BOXED["hub_z_to_z"](),
        (
            "sum_{k0=0}^{d-1} sum_{k1[i0]=0, i0=1..n}^{d-1} prod_{i0=1}^{n}( [k0 = k1[i0]] ) "
            "|k0> (x)_{i0=1}^{n}( |k1[i0]> )"
        ),
    ),
    "x_port_in_node_box": (
        lambda: BOXED["x_port_in_node_box"](),
        (
            "sum_{q0[i0]=0, i0=1..n}^{d-1} sum_{j0i0[i0,i1]=0, i0=1..n, i1=1..m}^{d-1} "
            "sum_{j0o0[i0]=0, i0=1..n}^{d-1} "
            "prod_{i0=1}^{n}( d^{-(m+1)/2} "
            "w_{d}^{q0[i0] (j0o0[i0] - sum_{i1=1}^{m} j0i0[i0,i1])} ) "
            "(x)_{i0=1}^{n}( |j0o0[i0]> )(x)_{i0=1}^{n}( (x)_{i1=1}^{m} <j0i0[i0,i1]| )"
        ),
    ),
    "two_entry_phase": (
        listing_example,
        (
            "sum_{k0=0}^{d-1} sum_{k2[i1,i2]=0, i1=1..m, i2=1..k}^{d-1} "
            "sum_{q1[i1],j1o0[i1]=0, i1=1..m}^{d-1} "
            "prod_{i1=1}^{m}( d^{-1} e^{2 pi i (1/3 [q1[i1]=1] + theta [q1[i1]=2])} "
            "w_{d}^{q1[i1] (j1o0[i1] - k0)} ) "
            "|k0>^{n} (x)_{i1=1}^{m}( |j1o0[i1]> (x)_{i2=1}^{k}( |k2[i1,i2]> ) )"
        ),
    ),
}


class TestGoldenText:
    """Exact text per factor and ket kind; the oracle tests only see the AST."""

    @pytest.mark.parametrize("name", sorted(GOLDEN_TEXT))
    def test_text(self, name: str) -> None:
        build, want = GOLDEN_TEXT[name]
        assert render_term(dirac_term(build())) == want

    def test_certificate_states_are_the_intermediate_diagrams(self) -> None:
        initial, results = chain_results()
        text = render_certificate(certify(initial, results))
        states = [initial, *(r.diagram for r in results)]
        assert len({render_term(dirac_term(s)) for s in states}) == len(states)
        for i, state in enumerate(states):
            block = text.split(f"state {i}:", 1)[1].split(f"state {i + 1}:", 1)[0]
            assert render_term(dirac_term(state)) in block, (i, block)


class TestMalformedNodeIds:
    @pytest.mark.parametrize("where", ["port_box", "wire"])
    def test_non_int_node_id_is_domain_error(self, where: str) -> None:
        d = Diagram()
        z = node(d, Z_SPIDER, 0, 2)
        bad = out(NodeId("a"), 0)  # type: ignore[arg-type]
        if where == "wire":
            d.add_wire(out(z, 0), bad)
        else:
            d.set_boundary_outputs([out(z, 0), out(z, 1)])
            d.add_bang_box(Mult("n"), port_scope=frozenset({bad, out(z, 0)}))
        for call in (dirac_term, recognize, to_dirac_source, dirac, render_dirac):
            with pytest.raises(PrinterDomainError):
                call(d)
