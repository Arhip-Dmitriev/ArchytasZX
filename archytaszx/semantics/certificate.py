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

"""Proof certificates: machine-checkable records of rewrite steps and replayable derivations.

A :class:`Derivation` is one node of a proof tree, carrying an initial diagram, a final
diagram, and the evidence between them. A ``STEP_SEQUENCE`` kind carries a tuple of
:class:`~qufzx.rewrite.engine.RewriteStep`; an ``INDUCTION`` kind carries those over a free
multiplicity symbol together with a base child and a step child. A :class:`Certificate`
pairs a derivation with the :class:`CheckMethod` its claim is discharged by.

:func:`replay` re-applies each recorded step to a fresh copy of the initial diagram, checking
per step that the rule still resolves by name, that the consumed nodes and wires are present,
that the matcher rediscovers the recorded match, and that the step ``apply`` produces equals
the step on record; then that the diagram reached is identical, id for id, to the recorded
final. :func:`verify` runs that replay and contracts both ends at a supplied assignment
through :func:`~qufzx.semantics.check.compare`; for an ``INDUCTION`` derivation it contracts
each child's own ends at that assignment, never the symbolic node's.

A verified certificate is evidence of five things: the recorded steps re-derive the recorded
final diagram exactly; every rule named still exists; every match is still discoverable in
the diagram it was recorded against; every recorded side-condition outcome and dimension
constraint is what the matcher derives fresh; and, for a ``NUMERIC_ORACLE`` derivation, the
two ends denote the same tensor at that assignment.

An ``INDUCTION`` derivation's own ends are symbolic in the induction index and are never
contracted. What is checked there is that each child is the parent's ends instantiated at the
claimed base value and one above it, and that both children's ends agree at the assignment.
That is the base case and one concrete instance of the step, not the step schema itself,
which :mod:`qufzx.semantics.induction` discharges and which no field of the certificate
records in re-checkable form.

It is not evidence about other assignments, about the satisfiability of a ``DEFERRED``
dimension constraint, or about a rule whose meaning changed under a name it kept.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from qufzx.diagram.bangbox import BangBoxError, free_mult_symbols
from qufzx.diagram.graph import Diagram, PortRef, Wire
from qufzx.rewrite.engine import RewriteResult, RewriteStep, apply
from qufzx.rewrite.rule import RewriteError
from qufzx.rewrite.rules_library import lookup_rule
from qufzx.semantics.check import (
    DEFAULT_TOLERANCE,
    CheckAssignmentValue,
    CheckError,
    ComparisonResult,
    EqualityMode,
    compare,
    instantiate,
)
from qufzx.semantics.contract_numeric import DEFAULT_MAX_ELEMENTS, ContractError
from qufzx.semantics.denote import DenoteError


class CertificateError(Exception):
    """Base class for all errors raised by this module."""


class CertificateDomainError(CertificateError):
    """Raised for a value outside the domain a certificate operation is defined on."""


class CertificateGrammarError(CertificateError):
    """Raised for a malformed certificate request, independent of any diagram's concreteness."""


class CheckMethod(enum.Enum):
    """How a certificate's claimed equality is discharged."""

    NUMERIC_ORACLE = "numeric_oracle"
    INDUCTION = "induction"


class DerivationKind(enum.Enum):
    """Which shape of evidence a derivation node carries."""

    STEP_SEQUENCE = "step_sequence"
    INDUCTION = "induction"


@dataclass(frozen=True, slots=True)
class InductionClaim:
    """The multiplicity symbol an induction derivation quantifies over, and how each half of it
    was discharged."""

    symbol: str
    base_value: int
    held_symbolic: tuple[str, ...] = ()
    step_discharge: str = ""

    def __post_init__(self) -> None:
        """Validate every field's type and shape, in declaration order."""
        if not isinstance(self.symbol, str):
            raise CertificateGrammarError(f"symbol must be a str, got {type(self.symbol).__name__}")
        if isinstance(self.base_value, bool) or not isinstance(self.base_value, int):
            raise CertificateGrammarError(
                f"base_value must be an int, got {type(self.base_value).__name__}"
            )
        if not isinstance(self.held_symbolic, tuple) or not all(
            isinstance(name, str) for name in self.held_symbolic
        ):
            raise CertificateGrammarError(
                f"held_symbolic must be a tuple of str, got {self.held_symbolic!r}"
            )
        if not isinstance(self.step_discharge, str):
            raise CertificateGrammarError(
                f"step_discharge must be a str, got {type(self.step_discharge).__name__}"
            )


@dataclass(frozen=True, slots=True)
class StructuralComparison:
    """The outcome of comparing two diagrams id for id: whether they agree, and where they
    first differ."""

    identical: bool
    reason: str


def compare_structure(a: Diagram, b: Diagram) -> StructuralComparison:
    """Compare two diagrams node id for node id, reporting the first difference in a fixed
    check order."""
    if not isinstance(a, Diagram) or not isinstance(b, Diagram):
        raise CertificateGrammarError(
            f"compare_structure requires two Diagram instances, got {type(a).__name__!r} "
            f"and {type(b).__name__!r}"
        )

    if sorted(a.nodes) != sorted(b.nodes):
        only_a = sorted(set(a.nodes) - set(b.nodes))
        only_b = sorted(set(b.nodes) - set(a.nodes))
        return StructuralComparison(
            False, f"node ids differ: only in a {only_a!r}, only in b {only_b!r}"
        )

    for nid in sorted(a.nodes):
        node_a = a.nodes[nid]
        node_b = b.nodes[nid]
        if node_a.generator_type != node_b.generator_type:
            return StructuralComparison(
                False,
                f"node {nid}: generator_type differs: "
                f"{node_a.generator_type.name} vs {node_b.generator_type.name}",
            )
        if node_a.inputs != node_b.inputs:
            left = tuple(p.dim for p in node_a.inputs)
            right = tuple(p.dim for p in node_b.inputs)
            return StructuralComparison(False, f"node {nid}: inputs differs: {left!r} vs {right!r}")
        if node_a.outputs != node_b.outputs:
            left = tuple(p.dim for p in node_a.outputs)
            right = tuple(p.dim for p in node_b.outputs)
            return StructuralComparison(
                False, f"node {nid}: outputs differs: {left!r} vs {right!r}"
            )
        if node_a.phase != node_b.phase:
            left_phase = str(node_a.phase) if node_a.phase is not None else "None"
            right_phase = str(node_b.phase) if node_b.phase is not None else "None"
            return StructuralComparison(
                False, f"node {nid}: phase differs: {left_phase} vs {right_phase}"
            )

    wires_a = sorted(a.wires, key=Wire.sort_key)
    wires_b = sorted(b.wires, key=Wire.sort_key)
    if wires_a != wires_b:
        wire_only_a = sorted(a.wires - b.wires, key=Wire.sort_key)
        wire_only_b = sorted(b.wires - a.wires, key=Wire.sort_key)
        return StructuralComparison(
            False, f"wires differ: only in a {wire_only_a!r}, only in b {wire_only_b!r}"
        )

    if sorted(a.bang_boxes) != sorted(b.bang_boxes):
        box_only_a = sorted(set(a.bang_boxes) - set(b.bang_boxes))
        box_only_b = sorted(set(b.bang_boxes) - set(a.bang_boxes))
        return StructuralComparison(
            False, f"bang box ids differ: only in a {box_only_a!r}, only in b {box_only_b!r}"
        )

    for box_id in sorted(a.bang_boxes):
        box_a = a.bang_boxes[box_id]
        box_b = b.bang_boxes[box_id]
        if box_a.multiplicity != box_b.multiplicity:
            return StructuralComparison(
                False,
                f"bang box {box_id}: multiplicity differs: "
                f"{box_a.multiplicity!r} vs {box_b.multiplicity!r}",
            )
        if sorted(box_a.node_scope) != sorted(box_b.node_scope):
            return StructuralComparison(
                False,
                f"bang box {box_id}: node scope differs: "
                f"{sorted(box_a.node_scope)!r} vs {sorted(box_b.node_scope)!r}",
            )
        scope_a = sorted(box_a.port_scope, key=PortRef.sort_key)
        scope_b = sorted(box_b.port_scope, key=PortRef.sort_key)
        if scope_a != scope_b:
            return StructuralComparison(
                False, f"bang box {box_id}: port scope differs: {scope_a!r} vs {scope_b!r}"
            )
        if box_a.parent != box_b.parent:
            return StructuralComparison(
                False,
                f"bang box {box_id}: parent differs: {box_a.parent!r} vs {box_b.parent!r}",
            )

    if a.boundary_inputs != b.boundary_inputs:
        return StructuralComparison(
            False,
            f"boundary inputs differ: {a.boundary_inputs!r} vs {b.boundary_inputs!r}",
        )

    if a.boundary_outputs != b.boundary_outputs:
        return StructuralComparison(
            False,
            f"boundary outputs differ: {a.boundary_outputs!r} vs {b.boundary_outputs!r}",
        )

    if a.scalar != b.scalar:
        return StructuralComparison(False, f"scalar differs: {a.scalar!r} vs {b.scalar!r}")

    items_a = sorted(a.parameters.items(), key=lambda kv: kv[0])
    items_b = sorted(b.parameters.items(), key=lambda kv: kv[0])
    if items_a != items_b:
        return StructuralComparison(
            False, f"parameter environments differ: {items_a!r} vs {items_b!r}"
        )

    return StructuralComparison(True, "identical")


@dataclass(frozen=True, slots=True, eq=False)
class Derivation:
    """One node of a proof tree: an initial diagram, a final diagram, and the evidence
    between them."""

    kind: DerivationKind
    initial: Diagram
    final: Diagram
    steps: tuple[RewriteStep, ...] = ()
    children: tuple[Derivation, ...] = ()
    label: str = ""
    induction: InductionClaim | None = None

    def __post_init__(self) -> None:
        """Validate every field's type and shape, in declaration order."""
        if not isinstance(self.kind, DerivationKind):
            raise CertificateGrammarError(f"kind must be a DerivationKind, got {self.kind!r}")
        if not isinstance(self.initial, Diagram):
            raise CertificateGrammarError(
                f"initial must be a Diagram, got {type(self.initial).__name__}"
            )
        if not isinstance(self.final, Diagram):
            raise CertificateGrammarError(
                f"final must be a Diagram, got {type(self.final).__name__}"
            )
        if not isinstance(self.steps, tuple) or not all(
            isinstance(step, RewriteStep) for step in self.steps
        ):
            raise CertificateGrammarError(
                f"steps must be a tuple of RewriteStep, got {self.steps!r}"
            )
        if not isinstance(self.children, tuple) or not all(
            isinstance(child, Derivation) for child in self.children
        ):
            raise CertificateGrammarError(
                f"children must be a tuple of Derivation, got {self.children!r}"
            )
        if not isinstance(self.label, str):
            raise CertificateGrammarError(f"label must be a str, got {self.label!r}")
        if self.induction is not None and not isinstance(self.induction, InductionClaim):
            raise CertificateGrammarError(
                f"induction must be an InductionClaim or None, got {type(self.induction).__name__}"
            )
        if self.kind is DerivationKind.STEP_SEQUENCE and self.children:
            raise CertificateGrammarError("a step_sequence derivation carries no children")
        if self.induction is not None and self.kind is not DerivationKind.INDUCTION:
            raise CertificateGrammarError(
                "induction is set but kind is not DerivationKind.INDUCTION"
            )
        if self.induction is None and self.kind is DerivationKind.INDUCTION:
            raise CertificateGrammarError(
                "kind is DerivationKind.INDUCTION but induction is not set"
            )
        if self.induction is not None and self.induction.base_value not in (0, 1):
            raise CertificateGrammarError(
                f"induction base_value must be 0 or 1, got {self.induction.base_value!r}"
            )
        if self.kind is DerivationKind.INDUCTION:
            if len(self.children) != 2:
                raise CertificateGrammarError(
                    f"an induction derivation must have exactly 2 children, got "
                    f"{len(self.children)}"
                )
            base, step = self.children
            if base.kind is not DerivationKind.STEP_SEQUENCE:
                raise CertificateGrammarError(
                    f"induction base child must be step_sequence, got {base.kind!r}"
                )
            if step.kind is not DerivationKind.STEP_SEQUENCE:
                raise CertificateGrammarError(
                    f"induction step child must be step_sequence, got {step.kind!r}"
                )

    def __eq__(self, other: object) -> bool:
        """Compare by content, with the two diagrams compared through :func:`compare_structure`."""
        if not isinstance(other, Derivation):
            return NotImplemented
        return (
            self.kind is other.kind
            and self.label == other.label
            and self.steps == other.steps
            and self.children == other.children
            and self.induction == other.induction
            and compare_structure(self.initial, other.initial).identical
            and compare_structure(self.final, other.final).identical
        )


@dataclass(frozen=True, slots=True, eq=False)
class Certificate:
    """A derivation together with the named method by which its claimed equality is discharged."""

    derivation: Derivation
    check_method: CheckMethod = CheckMethod.NUMERIC_ORACLE

    def __post_init__(self) -> None:
        """Validate that ``derivation`` is a Derivation and ``check_method`` a CheckMethod."""
        if not isinstance(self.derivation, Derivation):
            raise CertificateGrammarError(
                f"derivation must be a Derivation, got {type(self.derivation).__name__}"
            )
        if not isinstance(self.check_method, CheckMethod):
            raise CertificateGrammarError(
                f"check_method must be a CheckMethod, got {self.check_method!r}"
            )

    def __eq__(self, other: object) -> bool:
        """Compare by content: the same check method and an equal derivation."""
        if not isinstance(other, Certificate):
            return NotImplemented
        return self.check_method is other.check_method and self.derivation == other.derivation

    @property
    def initial(self) -> Diagram:
        """The derivation's initial diagram."""
        return self.derivation.initial

    @property
    def final(self) -> Diagram:
        """The derivation's final diagram."""
        return self.derivation.final

    @property
    def steps(self) -> tuple[RewriteStep, ...]:
        """The derivation's recorded steps."""
        return self.derivation.steps


def certify(
    initial: Diagram,
    results: Sequence[RewriteResult],
    *,
    label: str = "",
    check_method: CheckMethod = CheckMethod.NUMERIC_ORACLE,
) -> Certificate:
    """Assemble a certificate from a starting diagram and the results of applying rules to
    it in order."""
    if not isinstance(initial, Diagram):
        raise CertificateGrammarError(f"initial must be a Diagram, got {type(initial).__name__}")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise CertificateGrammarError(
            f"results must be a Sequence of RewriteResult, got {type(results).__name__}"
        )
    if not all(isinstance(result, RewriteResult) for result in results):
        raise CertificateGrammarError(
            f"every element of results must be a RewriteResult, got {results!r}"
        )

    final_source = results[-1].diagram if results else initial
    derivation = Derivation(
        kind=DerivationKind.STEP_SEQUENCE,
        initial=initial.copy(),
        final=final_source.copy(),
        steps=tuple(r.step for r in results),
        label=label,
    )
    return Certificate(derivation=derivation, check_method=check_method)


def certify_induction(
    initial: Diagram,
    final: Diagram,
    results: Sequence[RewriteResult],
    base: Derivation,
    step: Derivation,
    claim: InductionClaim,
    *,
    label: str = "",
    check_method: CheckMethod = CheckMethod.INDUCTION,
) -> Certificate:
    """Assemble an induction certificate from the symbolic pre and post diagrams, the symbolic
    rewrite results between them, the base and step child derivations, and the induction claim."""
    if not isinstance(initial, Diagram):
        raise CertificateGrammarError(f"initial must be a Diagram, got {type(initial).__name__}")
    if not isinstance(final, Diagram):
        raise CertificateGrammarError(f"final must be a Diagram, got {type(final).__name__}")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise CertificateGrammarError(
            f"results must be a Sequence of RewriteResult, got {type(results).__name__}"
        )
    if not all(isinstance(result, RewriteResult) for result in results):
        raise CertificateGrammarError(
            f"every element of results must be a RewriteResult, got {results!r}"
        )
    if not isinstance(base, Derivation):
        raise CertificateGrammarError(f"base must be a Derivation, got {type(base).__name__}")
    if not isinstance(step, Derivation):
        raise CertificateGrammarError(f"step must be a Derivation, got {type(step).__name__}")
    if not isinstance(claim, InductionClaim):
        raise CertificateGrammarError(
            f"claim must be an InductionClaim, got {type(claim).__name__}"
        )

    derivation = Derivation(
        kind=DerivationKind.INDUCTION,
        initial=initial.copy(),
        final=final.copy(),
        steps=tuple(r.step for r in results),
        children=(base, step),
        label=label,
        induction=claim,
    )
    return Certificate(derivation=derivation, check_method=check_method)


@dataclass(frozen=True, slots=True)
class StepReplay:
    """The outcome of re-applying one recorded step: which rule, whether it reproduced, and
    why not."""

    index: int
    rule_name: str
    reproduced: bool
    reason: str


@dataclass(frozen=True, slots=True, eq=False)
class ReplayResult:
    """The outcome of a whole replay: the re-derived diagram and the per-step record."""

    reproduced: bool
    reason: str
    diagram: Diagram
    steps: tuple[StepReplay, ...]
    children: tuple[ReplayResult, ...] = ()


def _first_step_difference(recorded: RewriteStep, produced: RewriteStep) -> str:
    """Name the first of ``recorded``'s thirteen fields that differs from ``produced``'s."""
    for field_name in (
        "rule_name",
        "match",
        "consumed_node_ids",
        "consumed_wires",
        "side_condition_outcomes",
        "dimension_constraints",
        "scalar_introduced",
        "port_mapping",
        "new_node_ids",
        "removed_deferred_issues",
        "introduced_deferred_issues",
        "phase_substitutions",
        "deferred_issue_identity_ambiguous",
    ):
        recorded_value = getattr(recorded, field_name)
        produced_value = getattr(produced, field_name)
        if recorded_value != produced_value:
            if field_name in ("port_mapping", "phase_substitutions"):
                recorded_value = sorted(recorded_value.items(), key=lambda kv: repr(kv[0]))
                produced_value = sorted(produced_value.items(), key=lambda kv: repr(kv[0]))
            return f"{field_name}: {recorded_value!r} vs {produced_value!r}"
    return "no field differs"


def _replay_one(
    working: Diagram, index: int, step: RewriteStep, rediscover: bool
) -> tuple[str | None, Diagram]:
    """Re-apply one recorded step, returning a failure reason or None, and the resulting diagram."""
    prefix = f"step {index} ({step.rule_name}): "
    try:
        rule = lookup_rule(step.rule_name)
    except RewriteError as exc:
        return f"{prefix}rule lookup failed: {exc}", working

    for nid in step.consumed_node_ids:
        if nid not in working.nodes:
            return (
                f"{prefix}consumed node {nid} is absent from the diagram being replayed",
                working,
            )

    for wire in step.consumed_wires:
        if wire not in working.wires:
            return (
                f"{prefix}consumed wire {wire!r} is absent from the diagram being replayed",
                working,
            )

    if rediscover and step.match not in rule.pattern.find_matches(working):
        reason = (
            f"{prefix}the recorded match is not among the matches rediscovered in the "
            "diagram being replayed"
        )
        return reason, working

    try:
        result = apply(working, rule, step.match)
    except RewriteError as exc:
        return f"{prefix}apply raised {type(exc).__name__}: {exc}", working

    if result.step != step:
        difference = _first_step_difference(step, result.step)
        return f"{prefix}the replayed step differs from the record: {difference}", working

    return None, result.diagram


def _replay_derivation(
    derivation: Derivation, source: Diagram | None, rediscover: bool
) -> ReplayResult:
    """Re-apply one derivation's recorded steps, then recurse into its children."""
    working = (source if source is not None else derivation.initial).copy()
    records: list[StepReplay] = []
    for index, step in enumerate(derivation.steps):
        failure, next_working = _replay_one(working, index, step, rediscover)
        if failure is not None:
            records.append(StepReplay(index, step.rule_name, False, failure))
            return ReplayResult(False, failure, working, tuple(records))
        records.append(StepReplay(index, step.rule_name, True, "reproduced"))
        working = next_working

    comparison = compare_structure(working, derivation.final)
    if not comparison.identical:
        reason = f"final diagram differs: {comparison.reason}"
        return ReplayResult(False, reason, working, tuple(records))

    child_results: list[ReplayResult] = []
    for child in derivation.children:
        child_result = _replay_derivation(child, None, rediscover)
        child_results.append(child_result)
        if not child_result.reproduced:
            reason = f"child derivation did not reproduce: {child_result.reason}"
            return ReplayResult(False, reason, working, tuple(records), tuple(child_results))

    return ReplayResult(True, "reproduced", working, tuple(records), tuple(child_results))


def replay(
    certificate: Certificate,
    source: Diagram | None = None,
    *,
    rediscover: bool = True,
) -> ReplayResult:
    """Re-apply every recorded step to a fresh copy of ``source``, defaulting to the
    recorded initial diagram."""
    if not isinstance(certificate, Certificate):
        raise CertificateGrammarError(
            f"certificate must be a Certificate, got {type(certificate).__name__}"
        )
    if source is not None and not isinstance(source, Diagram):
        raise CertificateGrammarError(f"source must be a Diagram or None, got {source!r}")
    if not isinstance(rediscover, bool):
        raise CertificateGrammarError(f"rediscover must be a bool, got {rediscover!r}")

    return _replay_derivation(certificate.derivation, source, rediscover)


@dataclass(frozen=True, slots=True, eq=False)
class VerificationReport:
    """The outcome of checking a certificate: the replay, the oracle comparison, and what
    they were run at."""

    verified: bool
    reason: str
    replay: ReplayResult
    comparison: ComparisonResult | None
    assignment: Mapping[str, CheckAssignmentValue]
    mode: EqualityMode
    check_method: CheckMethod
    child_comparisons: tuple[ComparisonResult, ...] = ()


def _induction_binding_failure(
    derivation: Derivation, assignment: Mapping[str, CheckAssignmentValue]
) -> str | None:
    """Why ``derivation``'s children and claim are not tied to its own ends, or None.

    A base child and a successor child, each the parent's ends instantiated at the claimed
    base value and one above it, with the parent's own multiplicity symbol supplying the
    index. Anything looser leaves the claim asserting nothing about the parent.
    """
    claim = derivation.induction
    if claim is None:
        return "the derivation carries no InductionClaim"
    if not claim.symbol or not claim.symbol.isidentifier():
        return f"claim symbol {claim.symbol!r} is not an identifier"
    if len(derivation.children) != 2:
        return f"expected a base child and a successor child, got {len(derivation.children)}"

    live = free_mult_symbols(derivation.initial)
    if len(live) != 1:
        return (
            f"the parent's initial diagram carries {sorted(live)!r} as free multiplicity "
            "symbol(s); induction needs exactly one to instantiate the children at"
        )
    (index,) = sorted(live)

    for offset, (child, label) in enumerate(zip(derivation.children, ("base", "successor"))):
        value = claim.base_value + offset
        binding = {**dict(assignment), index: value}
        for end, parent_end in (("initial", derivation.initial), ("final", derivation.final)):
            try:
                expected = instantiate(parent_end, binding)
            except (BangBoxError, CheckError, DenoteError, ContractError) as exc:
                return (
                    f"the parent's {end} diagram cannot be instantiated at {index}={value} "
                    f"to check the {label} child against: {exc}"
                )
            found = child.initial if end == "initial" else child.final
            structural = compare_structure(found, expected)
            if not structural.identical:
                return (
                    f"the {label} child's {end} diagram is not the parent's {end} diagram at "
                    f"{index}={value}: {structural.reason}"
                )
    return None


def verify(
    certificate: Certificate,
    assignment: Mapping[str, CheckAssignmentValue],
    *,
    mode: EqualityMode = EqualityMode.EXACT,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    rediscover: bool = True,
) -> VerificationReport:
    """Replay a certificate, then check its initial and re-derived final diagrams against
    the oracle."""
    if not isinstance(certificate, Certificate):
        raise CertificateGrammarError(
            f"certificate must be a Certificate, got {type(certificate).__name__}"
        )
    if not isinstance(assignment, Mapping):
        raise CertificateGrammarError(
            f"assignment must be a Mapping, got {type(assignment).__name__}"
        )
    if not isinstance(mode, EqualityMode):
        raise CertificateGrammarError(f"mode must be an EqualityMode, got {mode!r}")

    frozen_assignment = MappingProxyType(dict(assignment))

    if certificate.check_method is CheckMethod.NUMERIC_ORACLE:
        replay_result = replay(certificate, rediscover=rediscover)
        if not replay_result.reproduced:
            return VerificationReport(
                verified=False,
                reason=f"replay failed: {replay_result.reason}",
                replay=replay_result,
                comparison=None,
                assignment=frozen_assignment,
                mode=mode,
                check_method=certificate.check_method,
            )

        comparison = compare(
            certificate.initial,
            replay_result.diagram,
            frozen_assignment,
            mode=mode,
            tolerance=tolerance,
            max_elements=max_elements,
        )
        return VerificationReport(
            verified=comparison.matched,
            reason=comparison.reason
            if comparison.matched
            else f"oracle disagreed: {comparison.reason}",
            replay=replay_result,
            comparison=comparison,
            assignment=frozen_assignment,
            mode=mode,
            check_method=certificate.check_method,
        )

    if certificate.check_method is CheckMethod.INDUCTION:
        if certificate.derivation.kind is not DerivationKind.INDUCTION:
            raise CertificateGrammarError(
                f"check_method is INDUCTION but derivation.kind is {certificate.derivation.kind!r}"
            )

        replay_result = replay(certificate, rediscover=rediscover)
        if not replay_result.reproduced:
            return VerificationReport(
                verified=False,
                reason=f"replay failed: {replay_result.reason}",
                replay=replay_result,
                comparison=None,
                assignment=frozen_assignment,
                mode=mode,
                check_method=certificate.check_method,
            )

        binding_failure = _induction_binding_failure(certificate.derivation, frozen_assignment)
        if binding_failure is not None:
            return VerificationReport(
                verified=False,
                reason=f"induction claim unbound: {binding_failure}",
                replay=replay_result,
                comparison=None,
                assignment=frozen_assignment,
                mode=mode,
                check_method=certificate.check_method,
            )

        child_comparisons = tuple(
            compare(
                child.initial,
                child_replay.diagram,
                frozen_assignment,
                mode=mode,
                tolerance=tolerance,
                max_elements=max_elements,
            )
            for child, child_replay in zip(
                certificate.derivation.children, replay_result.children, strict=True
            )
        )
        verified = all(comparison.matched for comparison in child_comparisons)
        reason = (
            "reproduced and every child comparison matched"
            if verified
            else "; ".join(
                f"child {index}: {child_comparison.reason}"
                for index, child_comparison in enumerate(child_comparisons)
                if not child_comparison.matched
            )
        )
        return VerificationReport(
            verified=verified,
            reason=reason,
            replay=replay_result,
            comparison=None,
            assignment=frozen_assignment,
            mode=mode,
            check_method=certificate.check_method,
            child_comparisons=child_comparisons,
        )

    raise CertificateGrammarError(f"unknown check method: {certificate.check_method!r}")
