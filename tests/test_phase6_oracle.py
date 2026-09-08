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

"""The literal chain ``FULL_PLAN.md`` names for Phase 6's completion condition: emit a
certificate for the Phase 5 fusion, replay it on a fresh copy of the input, and confirm the
replay reproduces the output and passes the oracle.
"""

from __future__ import annotations

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.repl.parser import parse_dirac_source
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.match import find_matches
from archytaszx.rewrite.rules_library import SPIDER_FUSION
from archytaszx.semantics.certificate import (
    Certificate,
    Derivation,
    DerivationKind,
    certify,
    compare_structure,
    replay,
    verify,
)

from .helpers import build_ghz_with_copy


def test_the_ghz_fusion_certificate_replays_and_passes_the_oracle() -> None:
    for d_value in (2, 3, 4):
        diagram, _a_id, _b_id = build_ghz_with_copy(Dim.concrete(d_value))
        match = find_matches(diagram)[0]
        result = apply(diagram, SPIDER_FUSION, match)
        certificate = certify(diagram, [result])
        replayed = replay(certificate, source=diagram.copy())
        assert replayed.reproduced, replayed.reason
        comparison = compare_structure(replayed.diagram, certificate.final)
        assert comparison.identical, comparison.reason
        report = verify(certificate, {})
        assert report.verified, report.reason


def test_a_two_step_derivation_certifies_as_a_sequence() -> None:
    d = Dim.concrete(2)
    diagram = Diagram()
    s1 = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
    s2 = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
    s3 = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
    diagram.add_wire(PortRef(s1, Direction.OUTPUT, 0), PortRef(s2, Direction.INPUT, 0))
    diagram.add_wire(PortRef(s2, Direction.OUTPUT, 0), PortRef(s3, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(s1, Direction.INPUT, 0)])
    diagram.set_boundary_outputs(
        [PortRef(s1, Direction.OUTPUT, 1), PortRef(s3, Direction.OUTPUT, 0)]
    )

    match_one = find_matches(diagram)[0]
    result_one = apply(diagram, SPIDER_FUSION, match_one)
    match_two = find_matches(result_one.diagram)[0]
    result_two = apply(result_one.diagram, SPIDER_FUSION, match_two)

    certificate = certify(diagram, [result_one, result_two])
    assert len(certificate.steps) == 2

    replayed = replay(certificate)
    assert replayed.reproduced, replayed.reason
    assert len(replayed.steps) == 2
    assert all(step.reproduced for step in replayed.steps)

    report = verify(certificate, {})
    assert report.verified, report.reason


def test_a_symbolic_dimension_bound_in_the_environment_verifies() -> None:
    diagram, _a_id, _b_id = build_ghz_with_copy(Dim.symbol("d"))
    diagram.bind_parameter("d", 3)
    match = find_matches(diagram)[0]
    result = apply(diagram, SPIDER_FUSION, match)
    certificate = certify(diagram, [result])
    report = verify(certificate, {})
    assert report.verified, report.reason


def test_the_dirac_path_end_to_end_carries_a_certificate() -> None:
    for d_value in (2, 3, 5):
        pre = parse_dirac_source("sum_{k=0}^{d-1} |k,k>; copy")
        match = find_matches(pre)[0]
        result = apply(pre, SPIDER_FUSION, match)
        certificate = certify(pre, [result])
        report = verify(certificate, {"d": d_value})
        assert report.verified, report.reason

    pre_abstracted = parse_dirac_source("sum_{k=0}^{3-1} |k,k>; copy")
    match_abstracted = find_matches(pre_abstracted)[0]
    result_abstracted = apply(pre_abstracted, SPIDER_FUSION, match_abstracted)
    certificate_abstracted = certify(pre_abstracted, [result_abstracted])
    report_abstracted = verify(certificate_abstracted, {})
    assert report_abstracted.verified, report_abstracted.reason


def test_a_deliberately_broken_derivation_does_not_verify() -> None:
    diagram, _a_id, _b_id = build_ghz_with_copy(Dim.concrete(2))
    match = find_matches(diagram)[0]
    result = apply(diagram, SPIDER_FUSION, match)
    certificate = certify(diagram, [result])

    tampered_final = certificate.final.copy()
    (merged_id,) = tampered_final.nodes
    tampered_final.add_wire(
        PortRef(merged_id, Direction.OUTPUT, 0), PortRef(merged_id, Direction.OUTPUT, 1)
    )
    tampered_derivation = Derivation(
        kind=DerivationKind.STEP_SEQUENCE,
        initial=certificate.initial.copy(),
        final=tampered_final,
        steps=certificate.steps,
    )
    tampered_certificate = Certificate(derivation=tampered_derivation)

    report = verify(tampered_certificate, {})
    assert report.verified is False
    assert report.comparison is None
    assert report.reason.startswith("replay failed: final diagram differs")
