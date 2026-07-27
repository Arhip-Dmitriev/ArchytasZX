"""Tests for qufzx.diagram.graph: Port, Node, and Diagram."""

import pytest
import sympy as sp  # type: ignore[import-untyped]

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import (
    Diagram,
    Direction,
    GraphDomainError,
    GraphGrammarError,
    Node,
    NodeId,
    Port,
    PortRef,
    Wire,
)

from .helpers import build_ghz_with_copy


class TestPortRef:
    def test_equal_by_value(self) -> None:
        a = PortRef(node_id=0, direction=Direction.INPUT, index=1)  # type: ignore[arg-type]
        b = PortRef(node_id=0, direction=Direction.INPUT, index=1)  # type: ignore[arg-type]
        assert a == b
        assert hash(a) == hash(b)

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(GraphDomainError):
            PortRef(node_id=0, direction=Direction.INPUT, index=-1)  # type: ignore[arg-type]

    def test_bool_index_rejected(self) -> None:
        with pytest.raises(GraphGrammarError):
            PortRef(node_id=0, direction=Direction.INPUT, index=True)  # type: ignore[arg-type]


class TestWire:
    def test_unordered_equality(self) -> None:
        p0 = PortRef(node_id=0, direction=Direction.OUTPUT, index=0)  # type: ignore[arg-type]
        p1 = PortRef(node_id=1, direction=Direction.INPUT, index=0)  # type: ignore[arg-type]
        assert Wire(p0, p1) == Wire(p1, p0)
        assert hash(Wire(p0, p1)) == hash(Wire(p1, p0))

    def test_self_loop_same_port_rejected(self) -> None:
        p0 = PortRef(node_id=0, direction=Direction.OUTPUT, index=0)  # type: ignore[arg-type]
        with pytest.raises(GraphDomainError):
            Wire(p0, p0)


class TestTwoLegZSpider:
    def test_concrete_dimension(self) -> None:
        d = Dim.concrete(3)
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        node = diagram.nodes[node_id]
        assert node.num_inputs == 1
        assert node.num_outputs == 1
        assert node.inputs[0].dim == d
        assert node.outputs[0].dim == d
        assert node.phase is None

    def test_symbolic_dimension(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        node = diagram.nodes[node_id]
        assert not node.inputs[0].dim.is_concrete
        assert node.outputs[0].dim == d


class TestSymbolicPhaseSlot:
    def test_symbol_and_root_of_unity_entries(self) -> None:
        d = Dim.symbol("d")
        phase_vector = PhaseVector(
            d,
            {
                1: Phase.symbol("alpha"),
                2: Phase.root_of_unity(3, d),
            },
        )
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d], phase=phase_vector)
        node = diagram.nodes[node_id]
        assert node.phase is not None
        assert node.phase.get(1) == Phase.symbol("alpha")
        assert node.phase.get(2) == Phase.root_of_unity(3, d)


class TestGhzWithCopy:
    @pytest.mark.parametrize("d_value", [2, 3])
    def test_concrete_dims(self, d_value: int) -> None:
        d = Dim.concrete(d_value)
        diagram, a_id, b_id = build_ghz_with_copy(d)
        assert diagram.nodes[a_id].num_outputs == 2
        assert diagram.nodes[b_id].num_inputs == 1
        assert diagram.nodes[b_id].num_outputs == 2
        assert len(diagram.wires) == 1
        assert len(diagram.boundary_outputs) == 3

    def test_symbolic_dim_with_symbolic_phase(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.symbol("alpha")})
        diagram, _a_id, b_id = build_ghz_with_copy(d, phase_on_b=phase)
        assert diagram.nodes[b_id].phase is not None
        assert not diagram.nodes[b_id].phase.is_concrete  # type: ignore[union-attr]
        assert len(diagram.boundary_outputs) == 3


class TestReadOnlyViews:
    def test_nodes_view_is_read_only(self) -> None:
        diagram = Diagram()
        diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        with pytest.raises(TypeError):
            diagram.nodes[999] = None  # type: ignore[index]

    def test_mutating_returned_wires_does_not_affect_diagram(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        wires = diagram.wires
        wires_mutable: set[Wire] = set(wires)
        wires_mutable.clear()
        assert len(diagram.wires) == 1

    def test_mutating_returned_boundary_tuple_does_not_affect_diagram(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        ref = PortRef(a, Direction.OUTPUT, 0)
        diagram.set_boundary_outputs([ref])
        boundary = diagram.boundary_outputs
        as_list = list(boundary)
        as_list.clear()
        assert diagram.boundary_outputs == (ref,)


class TestDeepCopyIndependence:
    def _base_diagram(self) -> tuple[Diagram, NodeId, NodeId]:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2), Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[Dim.concrete(2)])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        diagram.set_boundary_outputs(
            [PortRef(a, Direction.OUTPUT, 1), PortRef(b, Direction.OUTPUT, 0)]
        )
        return diagram, a, b

    def test_adding_node_to_copy_leaves_original_unchanged(self) -> None:
        diagram, _a, _b = self._base_diagram()
        original_node_count = len(diagram.nodes)
        copy = diagram.copy()
        copy.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        assert len(diagram.nodes) == original_node_count
        assert len(copy.nodes) == original_node_count + 1

    def test_adding_wire_to_copy_leaves_original_unchanged(self) -> None:
        diagram, a, _b = self._base_diagram()
        original_wire_count = len(diagram.wires)
        copy = diagram.copy()
        c = copy.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        copy.add_wire(PortRef(a, Direction.OUTPUT, 1), PortRef(c, Direction.INPUT, 0))
        assert len(diagram.wires) == original_wire_count
        assert len(copy.wires) == original_wire_count + 1

    def test_removing_wire_from_copy_leaves_original_unchanged(self) -> None:
        diagram, a, b = self._base_diagram()
        original_wire_count = len(diagram.wires)
        copy = diagram.copy()
        copy.remove_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        assert len(diagram.wires) == original_wire_count
        assert len(copy.wires) == original_wire_count - 1

    def test_setting_phase_on_copy_leaves_original_unchanged(self) -> None:
        diagram, a, _b = self._base_diagram()
        copy = diagram.copy()
        phase = PhaseVector(Dim.concrete(2), {1: Phase.turns(1)})
        copy.set_phase(a, phase)
        assert diagram.nodes[a].phase is None
        assert copy.nodes[a].phase == phase

    def test_multiplying_scalar_on_copy_leaves_original_unchanged(self) -> None:
        diagram, _a, _b = self._base_diagram()
        copy = diagram.copy()
        copy.multiply_scalar(Scalar.omega(Dim.concrete(4), 1))
        assert diagram.scalar == Scalar.one()
        assert copy.scalar != Scalar.one()

    def test_copy_preserves_node_ids(self) -> None:
        diagram, a, b = self._base_diagram()
        copy = diagram.copy()
        assert set(copy.nodes.keys()) == {a, b}


class TestNodeRemoval:
    def test_removing_node_drops_incident_wires(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(2)], output_dims=[])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        diagram.remove_node(b)
        assert len(diagram.wires) == 0
        assert b not in diagram.nodes

    def test_removing_node_drops_boundary_entries(self) -> None:
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 0)])
        diagram.remove_node(a)
        assert diagram.boundary_outputs == ()

    def test_removing_unknown_node_raises(self) -> None:
        diagram = Diagram()
        with pytest.raises(GraphGrammarError):
            diagram.remove_node(12345)  # type: ignore[arg-type]


class TestNodeValueChecks:
    def test_node_rejects_non_port_entries(self) -> None:
        with pytest.raises(GraphGrammarError):
            Node(id=0, generator_type=Z_SPIDER, inputs=(1,), outputs=())  # type: ignore[arg-type]

    def test_node_rejects_non_phase_vector_phase(self) -> None:
        with pytest.raises(GraphGrammarError):
            Node(id=0, generator_type=Z_SPIDER, inputs=(), outputs=(), phase="nope")  # type: ignore[arg-type]

    def test_port_rejects_non_dim(self) -> None:
        with pytest.raises(GraphGrammarError):
            Port(dim=2)  # type: ignore[arg-type]


class TestSubstitute:
    def test_preserves_node_ids_wires_and_boundary(self) -> None:
        d = Dim.symbol("d")
        diagram, a_id, b_id = build_ghz_with_copy(d)
        substituted = diagram.substitute({"d": 3})
        assert set(substituted.nodes.keys()) == {a_id, b_id}
        assert substituted.wires == diagram.wires
        assert substituted.boundary_outputs == diagram.boundary_outputs
        assert substituted.boundary_inputs == diagram.boundary_inputs

    def test_substitutes_port_dims_phase_and_scalar(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.symbol("alpha")})
        diagram, a_id, b_id = build_ghz_with_copy(d, phase_on_b=phase)
        diagram.multiply_scalar(Scalar.symbol("s"))

        substituted = diagram.substitute({"d": 3, "alpha": sp.Rational(1, 6), "s": 2})

        for node_id in (a_id, b_id):
            node = substituted.nodes[node_id]
            for port in (*node.inputs, *node.outputs):
                assert port.dim == Dim.concrete(3)
        b_phase = substituted.nodes[b_id].phase
        assert b_phase is not None
        assert b_phase.get(1) == Phase.turns(sp.Rational(1, 6))
        assert substituted.scalar == Scalar.rational(2)

    def test_does_not_mutate_original(self) -> None:
        d = Dim.symbol("d")
        diagram, a_id, _b_id = build_ghz_with_copy(d)
        diagram.substitute({"d": 3})
        assert not diagram.nodes[a_id].outputs[0].dim.is_concrete

    def test_partial_substitution_leaves_other_symbols(self) -> None:
        d = Dim.symbol("d")
        phase = PhaseVector(d, {1: Phase.symbol("alpha")})
        diagram, _a_id, b_id = build_ghz_with_copy(d, phase_on_b=phase)
        substituted = diagram.substitute({"d": 3})
        b_phase = substituted.nodes[b_id].phase
        assert b_phase is not None
        assert not b_phase.is_concrete
        assert b_phase.dim == Dim.concrete(3)
