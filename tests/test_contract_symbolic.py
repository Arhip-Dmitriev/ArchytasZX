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


"""Phase 9: symbolic contraction with the dimension formal, cross-checked against the oracle."""

from __future__ import annotations

import cmath
import itertools

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.bangbox import Mult, expand_concrete_boxes
from qufzx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER, GeneratorType
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.semantics.check import score
from qufzx.semantics.contract_numeric import ContractSizeError, contract
from qufzx.semantics.contract_symbolic import (
    SymbolicContractionDomainError,
    SymbolicContractionUnsupportedError,
    SymbolicContractionValidationError,
    SymbolicTensor,
    contract_symbolic,
)

D = Dim.symbol("d")


def _single(gen: GeneratorType, num_inputs: int, num_outputs: int, dim: Dim) -> Diagram:
    """One node of ``gen`` with every leg on the boundary."""
    diagram = Diagram()
    node_id = diagram.add_node(gen, input_dims=[dim] * num_inputs, output_dims=[dim] * num_outputs)
    diagram.set_boundary_outputs(
        [PortRef(node_id, Direction.OUTPUT, i) for i in range(num_outputs)]
    )
    diagram.set_boundary_inputs([PortRef(node_id, Direction.INPUT, i) for i in range(num_inputs)])
    return diagram


def _ghz_with_copy(dim: Dim) -> Diagram:
    """The Phase 3 worked example: a two-leg Z state feeding a copy spider."""
    diagram = Diagram()
    a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
    b = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    diagram.add_wire(PortRef(a, Direction.OUTPUT, 1), PortRef(b, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [
            PortRef(a, Direction.OUTPUT, 0),
            PortRef(b, Direction.OUTPUT, 0),
            PortRef(b, Direction.OUTPUT, 1),
        ]
    )
    diagram.set_boundary_inputs([])
    return diagram


def _fourier_chain(dim: Dim, count: int) -> Diagram:
    """``count`` Fourier boxes wired output-to-input in series."""
    diagram = Diagram()
    ids = [diagram.add_node(FOURIER_BOX, input_dims=[dim], output_dims=[dim]) for _ in range(count)]
    for left, right in itertools.pairwise(ids):
        diagram.add_wire(PortRef(left, Direction.OUTPUT, 0), PortRef(right, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(ids[0], Direction.INPUT, 0)])
    diagram.set_boundary_outputs([PortRef(ids[-1], Direction.OUTPUT, 0)])
    return diagram


def _identity_wire(dim: Dim) -> Diagram:
    """A phaseless one-in-one-out Z spider, which denotes the identity."""
    diagram = Diagram()
    node_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
    diagram.set_boundary_inputs([PortRef(node_id, Direction.INPUT, 0)])
    diagram.set_boundary_outputs([PortRef(node_id, Direction.OUTPUT, 0)])
    return diagram


class TestAgainstTheNumericOracle:
    @pytest.mark.parametrize(
        "gen,num_inputs,num_outputs",
        [
            (Z_SPIDER, 0, 2),
            (Z_SPIDER, 2, 1),
            (Z_SPIDER, 1, 1),
            (X_SPIDER, 0, 1),
            (X_SPIDER, 1, 2),
            (X_SPIDER, 2, 2),
            (FOURIER_BOX, 1, 1),
        ],
    )
    @pytest.mark.parametrize("value", [2, 3, 4, 5])
    def test_single_generator_matches_numeric(
        self, gen: GeneratorType, num_inputs: int, num_outputs: int, value: int
    ) -> None:
        symbolic = contract_symbolic(_single(gen, num_inputs, num_outputs, D))
        dense = np.asarray(symbolic.substitute({"d": value}).to_dense())
        numeric = contract(_single(gen, num_inputs, num_outputs, Dim.concrete(value))).tensor
        assert np.allclose(dense, numeric)

    @pytest.mark.parametrize("value", [2, 3, 4])
    def test_ghz_with_copy_matches_numeric(self, value: int) -> None:
        symbolic = contract_symbolic(_ghz_with_copy(D))
        dense = np.asarray(symbolic.substitute({"d": value}).to_dense())
        assert np.allclose(dense, contract(_ghz_with_copy(Dim.concrete(value))).tensor)

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("value", [2, 3, 5])
    def test_fourier_chain_matches_numeric(self, count: int, value: int) -> None:
        symbolic = contract_symbolic(_fourier_chain(D, count))
        dense = np.asarray(symbolic.substitute({"d": value}).to_dense())
        assert np.allclose(dense, contract(_fourier_chain(Dim.concrete(value), count)).tensor)

    def test_axis_order_is_outputs_then_inputs(self) -> None:
        symbolic = contract_symbolic(_single(Z_SPIDER, 1, 2, D))
        axes = [(axis.port.direction, axis.port.index) for axis in symbolic.axes]
        assert axes == [
            (Direction.OUTPUT, 0),
            (Direction.OUTPUT, 1),
            (Direction.INPUT, 0),
        ]


class TestSymbolicIdentities:
    def test_fourth_power_of_fourier_is_the_identity_with_d_formal(self) -> None:
        assert (
            contract_symbolic(_fourier_chain(D, 4)).entry
            == contract_symbolic(_identity_wire(D)).entry
        )

    @pytest.mark.parametrize("count", [8, 12])
    def test_every_fourth_power_of_fourier_is_the_identity(self, count: int) -> None:
        assert (
            contract_symbolic(_fourier_chain(D, count)).entry
            == contract_symbolic(_identity_wire(D)).entry
        )

    @pytest.mark.parametrize("count", [1, 2, 3, 5, 6, 7])
    def test_a_non_multiple_of_four_is_not_the_identity(self, count: int) -> None:
        assert (
            contract_symbolic(_fourier_chain(D, count)).entry
            != contract_symbolic(_identity_wire(D)).entry
        )

    def test_cube_of_fourier_is_its_adjoint(self) -> None:
        cube = contract_symbolic(_fourier_chain(D, 3)).entry
        single = contract_symbolic(_fourier_chain(D, 1)).entry
        assert cube == single.conjugate().simplify()


class TestTheSubstitutionPath:
    def test_a_large_concrete_dimension_answers_through_substitution(self) -> None:
        symbolic = contract_symbolic(_single(Z_SPIDER, 1, 1, D))
        answered = symbolic.substitute({"d": 4096})
        assert answered.dims()[0] == Dim.concrete(4096)
        assert "4096" in str(answered.entry)

    def test_the_instantiate_then_contract_path_refuses_the_same_size(self) -> None:
        with pytest.raises(ContractSizeError):
            contract(_single(Z_SPIDER, 1, 1, Dim.concrete(4096)), max_elements=1000)

    def test_the_contraction_itself_never_reaches_for_a_number(self) -> None:
        """to_dense is the one declared bridge to numbers; the contraction is not."""
        import ast
        import pathlib

        source = pathlib.Path("qufzx/semantics/contract_symbolic.py").read_text()
        tree = ast.parse(source)
        contraction = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "contract_symbolic"
        )
        text = ast.unparse(contraction)
        assert "to_int" not in text
        assert "to_complex" not in text
        assert "numpy" not in text and "np." not in text

    def test_numpy_is_reached_only_from_the_dense_bridge(self) -> None:
        import ast
        import pathlib

        source = pathlib.Path("qufzx/semantics/contract_symbolic.py").read_text()
        tree = ast.parse(source)
        importers = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            for statement in ast.walk(node)
            if isinstance(statement, ast.Import)
            and any(alias.name == "numpy" for alias in statement.names)
        }
        module_level = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom)) and "numpy" in ast.unparse(node)
        ]
        assert importers == {"to_dense"}
        assert module_level == []


class TestRefusals:
    def test_a_hard_validation_failure_is_refused(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        with pytest.raises(SymbolicContractionValidationError):
            contract_symbolic(diagram)

    def test_going_dense_refuses_a_symbolic_dimension(self) -> None:
        symbolic = contract_symbolic(_single(Z_SPIDER, 1, 1, D))
        with pytest.raises(SymbolicContractionDomainError):
            symbolic.to_dense()

    def test_going_dense_respects_its_element_cap(self) -> None:
        symbolic = contract_symbolic(_single(Z_SPIDER, 2, 2, D)).substitute({"d": 10})
        with pytest.raises(SymbolicContractionDomainError):
            symbolic.to_dense(max_elements=100)


class TestUnsupportedShapes:
    def test_a_symbolic_multiplicity_with_an_open_boundary_is_unsupported(self) -> None:
        from qufzx.diagram.bangbox import Mult

        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[D])
        diagram.set_boundary_outputs([PortRef(node_id, Direction.OUTPUT, 0)])
        diagram.add_bang_box(node_scope=frozenset({node_id}), multiplicity=Mult("n"))
        with pytest.raises(SymbolicContractionUnsupportedError, match="variable rank"):
            contract_symbolic(diagram)


class TestPhasedSpiders:
    """A phase vector is sparse over concrete indices, so it survives a symbolic dimension."""

    @staticmethod
    def _phased(
        gen: GeneratorType, num_inputs: int, num_outputs: int, dim: Dim, entries: dict[int, Phase]
    ) -> Diagram:
        diagram = Diagram()
        node_id = diagram.add_node(
            gen, input_dims=[dim] * num_inputs, output_dims=[dim] * num_outputs
        )
        diagram.set_phase(node_id, PhaseVector(dim, entries))
        diagram.set_boundary_outputs(
            [PortRef(node_id, Direction.OUTPUT, i) for i in range(num_outputs)]
        )
        diagram.set_boundary_inputs(
            [PortRef(node_id, Direction.INPUT, i) for i in range(num_inputs)]
        )
        return diagram

    @pytest.mark.parametrize("gen", [Z_SPIDER, X_SPIDER])
    @pytest.mark.parametrize("num_inputs,num_outputs", [(1, 1), (0, 2), (2, 1), (1, 2), (0, 1)])
    @pytest.mark.parametrize("value", [3, 4, 5])
    def test_a_phased_spider_matches_numeric_at_symbolic_d(
        self, gen: GeneratorType, num_inputs: int, num_outputs: int, value: int
    ) -> None:
        entries = {1: Phase.turns(sp.Rational(1, 2)), 2: Phase.turns(sp.Rational(1, 3))}
        symbolic = contract_symbolic(self._phased(gen, num_inputs, num_outputs, D, entries))
        dense = np.asarray(symbolic.substitute({"d": value}).to_dense())
        numeric = contract(
            self._phased(gen, num_inputs, num_outputs, Dim.concrete(value), entries)
        ).tensor
        assert np.allclose(dense, numeric)

    @pytest.mark.parametrize("gen", [Z_SPIDER, X_SPIDER])
    def test_a_rank_four_phased_spider_matches_numeric(self, gen: GeneratorType) -> None:
        entries = {1: Phase.turns(sp.Rational(1, 2)), 2: Phase.turns(sp.Rational(1, 3))}
        symbolic = contract_symbolic(self._phased(gen, 2, 2, D, entries))
        dense = np.asarray(symbolic.substitute({"d": 3}).to_dense())
        assert np.allclose(
            dense, contract(self._phased(gen, 2, 2, Dim.concrete(3), entries)).tensor
        )

    @pytest.mark.parametrize("gen", [Z_SPIDER, X_SPIDER])
    def test_a_free_phase_parameter_stays_formal(self, gen: GeneratorType) -> None:
        symbolic = contract_symbolic(self._phased(gen, 1, 1, D, {1: Phase.symbol("theta")}))
        assert "theta" in symbolic.entry.free_symbols
        assert "d" in symbolic.entry.free_symbols

    @pytest.mark.parametrize("gen", [Z_SPIDER, X_SPIDER])
    @pytest.mark.parametrize("value", [2, 3, 5])
    def test_a_free_phase_parameter_matches_numeric_once_bound(
        self, gen: GeneratorType, value: int
    ) -> None:
        turns = sp.Rational(1, 4)
        symbolic = contract_symbolic(self._phased(gen, 1, 1, D, {1: Phase.symbol("theta")}))
        at_d = symbolic.substitute({"d": value})
        bound = SymbolicTensor(at_d.axes, at_d.entry.substitute({"theta": turns}))
        numeric = contract(
            self._phased(gen, 1, 1, Dim.concrete(value), {1: Phase.turns(turns)})
        ).tensor
        assert np.allclose(np.asarray(bound.to_dense()), numeric)

    def test_a_phaseless_spider_is_unchanged_by_the_correction_path(self) -> None:
        entries: dict[int, Phase] = {}
        with_vector = contract_symbolic(self._phased(Z_SPIDER, 1, 1, D, entries))
        without = contract_symbolic(_single(Z_SPIDER, 1, 1, D))
        assert with_vector.entry == without.entry

    @pytest.mark.parametrize("value", [3, 4])
    def test_two_phased_spiders_wired_together_match_numeric(self, value: int) -> None:
        entries = {1: Phase.turns(sp.Rational(1, 2))}

        def build(dim: Dim) -> Diagram:
            diagram = Diagram()
            a = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
            b = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[dim])
            diagram.set_phase(a, PhaseVector(dim, entries))
            diagram.set_phase(b, PhaseVector(dim, entries))
            diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
            diagram.set_boundary_inputs([PortRef(a, Direction.INPUT, 0)])
            diagram.set_boundary_outputs([PortRef(b, Direction.OUTPUT, 0)])
            return diagram

        symbolic = contract_symbolic(build(D))
        dense = np.asarray(symbolic.substitute({"d": value}).to_dense())
        assert np.allclose(dense, contract(build(Dim.concrete(value))).tensor)


class TestCrossProcessDeterminism:
    """Byte-identical output across processes and PYTHONHASHSEED values."""

    @staticmethod
    def _run(seed: str) -> str:
        import os
        import subprocess
        import sys

        environment = dict(os.environ, PYTHONHASHSEED=seed)
        completed = subprocess.run(
            [sys.executable, "tests/_symbolic_determinism_script.py"],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        return completed.stdout

    def test_output_is_identical_across_hash_seeds(self) -> None:
        outputs = {self._run(seed) for seed in ("0", "1", "17", "424242")}
        assert len(outputs) == 1
        assert "Sum" in next(iter(outputs))

    def test_no_dummy_symbol_is_ever_constructed(self) -> None:
        import pathlib

        for name in ("qufzx/algebra/scalar.py", "qufzx/semantics/contract_symbolic.py"):
            assert "Dummy(" not in pathlib.Path(name).read_text()


class TestBangBoxesInSymbolicContraction:
    """A bang box multiplies what its scope contributes; ignoring it drops that factor."""

    @staticmethod
    def _closed_cap_family(multiplicity: Mult | None) -> Diagram:
        d = Dim.symbol("d")
        diagram = Diagram()
        state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        effect = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
        if multiplicity is not None:
            diagram.add_bang_box(multiplicity, node_scope=frozenset({state, effect}))
        return diagram

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("d_value", [2, 3, 5])
    def test_a_concrete_multiplicity_is_honoured(self, k: int, d_value: int) -> None:
        boxed = self._closed_cap_family(Mult.concrete(k))
        entry = contract_symbolic(boxed).entry.substitute({"d": d_value})
        got = complex(sp.N(entry.to_sympy().doit()))
        expected = complex(sp.N(sp.sqrt(d_value) ** k))
        assert cmath.isclose(got, expected, abs_tol=1e-9)

    def test_a_symbolic_multiplicity_over_a_closed_scope_stays_closed_form(self) -> None:
        """Phase 9 ii: substituting the environment answers a count at any size."""
        boxed = self._closed_cap_family(Mult.symbol("m"))
        entry = contract_symbolic(boxed).entry
        assert not entry.to_sympy().has(sp.Sum)
        for m in (0, 1, 2, 5, 30):
            for d_value in (2, 3, 7):
                got = complex(sp.N(entry.substitute({"m": m, "d": d_value}).to_sympy().doit()))
                expected = complex(sp.N(sp.sqrt(d_value) ** m))
                assert cmath.isclose(got, expected, rel_tol=1e-9)

    def test_a_thousand_copies_answer_without_building_a_tensor(self) -> None:
        entry = contract_symbolic(self._closed_cap_family(Mult.symbol("m"))).entry
        value = entry.substitute({"m": 1000, "d": 7}).to_sympy()
        assert sp.simplify(value - sp.sqrt(7) ** 1000) == 0

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("d_value", [2, 3])
    def test_a_concrete_box_meeting_the_boundary_is_expanded(self, k: int, d_value: int) -> None:
        """A concrete count is expanded whatever its scope touches, and must match the oracle."""
        d = Dim.symbol("d")
        diagram = Diagram()
        state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        diagram.set_boundary_outputs([PortRef(state, Direction.OUTPUT, 0)])
        diagram.add_bang_box(Mult.concrete(k), node_scope=frozenset({state}))

        contracted = contract_symbolic(diagram)
        assert len(contracted.axes) == k
        expanded = expand_concrete_boxes(diagram)
        truth = score(expanded, {"d": d_value}).tensor
        entry = contracted.entry.substitute({"d": d_value}).to_sympy()
        for index in itertools.product(range(d_value), repeat=k):
            substituted = entry
            for position, value in enumerate(index):
                substituted = substituted.subs(
                    sp.Symbol(f"_i{position}", integer=True, nonnegative=True), value
                )
            got = complex(sp.N(substituted.doit()))
            assert cmath.isclose(got, complex(truth[index]) if k else complex(truth), abs_tol=1e-9)

    def test_a_box_meeting_the_boundary_is_refused_not_ignored(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        diagram.set_boundary_outputs([PortRef(state, Direction.OUTPUT, 0)])
        diagram.add_bang_box(Mult.symbol("m"), node_scope=frozenset({state}))
        with pytest.raises(SymbolicContractionUnsupportedError):
            contract_symbolic(diagram)
