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

"""The proof of an EQUAL decision: its own certificates, else the normaliser of symbolic sums,
else a search; restricted decisions through the family's proof or the restricted pair's."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from archytaszx.diagram.graph import Diagram
from archytaszx.semantics import kernel, lean_net, lean_sym
from archytaszx.semantics.decide import Decision, DecisionMethod, dimension_floors
from archytaszx.semantics.lean_scalar import ScalarUnsupported
from archytaszx.semantics.prove import ProofCertificate, ProofOutcome, prove
from archytaszx.semantics.restrict import Restriction

EqualProof = (
    ProofCertificate | ProofOutcome | kernel.Theorem | kernel.Symbolic | kernel.RestrictedProof
)


def decision_proof(
    decision: Decision,
    left: Diagram,
    right: Diagram,
    describe: Callable[[Diagram], str] | None = None,
) -> tuple[EqualProof | None, str]:
    """The proof of an EQUAL ``decision`` of ``left = right`` over the pair it is about, or
    None and the reasons each builder gave."""
    if decision.restriction is not None:
        return restricted_proof(decision, left, right, describe)
    pair = (left, right) if decision.doubled is None else decision.doubled
    inner = decision
    if decision.doubled is not None and decision.inner_method is not None:
        inner = replace(decision, method=decision.inner_method)
    return pair_proof(inner, *pair, describe=describe)


def pair_proof(
    decision: Decision,
    left: Diagram,
    right: Diagram,
    describe: Callable[[Diagram], str] | None = None,
) -> tuple[ProofCertificate | ProofOutcome | kernel.Theorem | kernel.Symbolic | None, str]:
    """A Lean-checkable proof of ``left = right`` from ``decision``: its own certificates, else
    the normaliser of symbolic sums, else a search; None and the three reasons when all fail."""
    if decision.method is DecisionMethod.NORMAL_FORM and len(decision.certificates) == 2:
        forward, backward = decision.certificates
        if forward.initial is left and backward.initial is right:
            return (
                ProofCertificate(
                    left,
                    right,
                    forward,
                    backward,
                    tuple(s.rule_name for s in forward.steps),
                    tuple(s.rule_name for s in backward.steps),
                ),
                "",
            )
    try:
        return kernel.from_decision(decision, left, right), ""
    except kernel.KernelError as exc:
        why_kernel = str(exc)
    try:
        return kernel.symbolic(left, right, describe=describe), ""
    except kernel.KernelError as exc:
        why_symbolic = str(exc)
    outcome = prove(left, right)
    if outcome.proved:
        return outcome, ""
    return None, f"kernel: {why_kernel}; normaliser: {why_symbolic}; search: {outcome.reason}"


def restricted_proof(
    decision: Decision,
    left: Diagram,
    right: Diagram,
    describe: Callable[[Diagram], str] | None = None,
) -> tuple[kernel.RestrictedProof | None, str]:
    """The proof of a restricted EQUAL: the family's proof, or the restricted pair's."""
    assert decision.restriction is not None
    family = decision.family_holds
    source = decision.family if family and decision.family is not None else decision
    pair = (left, right) if family else decision.restricted_pair
    if pair is None or source is None:
        return None, "the decision names no pair to prove"
    if source.doubled is not None:
        pair = source.doubled
    inner = source
    if source.doubled is not None and source.inner_method is not None:
        inner = replace(source, method=source.inner_method)
    floors = tuple(decision.floors_used or ())
    evidence: object | None = None
    why = ""
    if floors and not family:
        try:
            evidence = kernel.Symbolic(
                pair[0],
                pair[1],
                lean_sym.export_symbolic(
                    pair[0],
                    pair[1],
                    describe=describe,
                    floors=floors_of(pair, decision.restriction),
                ),
            )
        except (lean_net.NetUnsupported, ScalarUnsupported) as exc:
            evidence, why = None, f"normaliser under floors: {exc}; "
    if evidence is None:
        evidence, rest = pair_proof(inner, *pair, describe=describe)
        why += rest
    if evidence is None:
        return None, why
    if isinstance(evidence, ProofOutcome):
        artefact = evidence.certificate if evidence.certificate is not None else evidence.induction
        if artefact is None:
            return None, f"search: {evidence.reason} (no certificate or induction proof)"
        evidence = artefact
    return (
        kernel.RestrictedProof(evidence, decision.restriction, left, right, family, floors),
        "",
    )


def floors_of(pair: tuple[Diagram, Diagram], restriction: Restriction) -> dict[str, int]:
    """The pair's dimension floors raised to the restriction's."""
    merged = dimension_floors(*pair)
    for name, low in restriction.floors.items():
        merged[name] = max(merged.get(name, 1), low)
    return merged
