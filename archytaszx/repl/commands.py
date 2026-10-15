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

"""REPL commands over a :class:`Session`: load, show, rules, match, apply, check, normalize,
decide, saturate, prove, induct, certificate, translate, name, names, undo, reset, detail and
help.

A command line is ``WORD args``; a line whose first word is not a command name or alias is
``load <line>``. ``@NAME`` stands for a named diagram wherever a diagram source is expected
(``@start`` and ``@current`` are built in).

No command sees the notation mode. :func:`execute` returns an :class:`Outcome` of blocks:
:class:`Text` (mode-independent lines) and :class:`State` (an engine object). Only
:func:`format_outcome` reads the mode, rendering each State with
:func:`~archytaszx.repl.printer.render`; DIRAC output therefore holds no graph listing.
:func:`detail_text` shows a command's full payload with every diagram state as
:func:`~archytaszx.repl.printer.render_detail` (listing and Dirac side by side).

:func:`execute` raises only :class:`CommandError` subclasses, the engine error chained as
``__cause__``; :func:`run` never raises for an ``Exception``, turning it into an ``ok=False``
outcome and storing an :class:`ErrorDetail` as ``session.last_detail``.

Numeric checks run only at user-supplied assignments or
:func:`~archytaszx.semantics.decide.sample_grid` points, overridden by supplied values; the
parameter environment is used only as the induction witness for non-multiplicity symbols.

Failure reports state the Dirac-level outcome of both sides, an oracle counterexample where
one exists, otherwise the furthest-simplified states and the bound that was hit.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import DimensionError
from archytaszx.algebra.phase import PhaseError
from archytaszx.algebra.scalar import ScalarError
from archytaszx.diagram.bangbox import BangBoxError, Mult, free_mult_symbols
from archytaszx.diagram.generators import GeneratorError
from archytaszx.diagram.graph import Diagram, Direction, GraphError, NodeId, PortRef, Wire
from archytaszx.diagram.scalable import (
    ScalableDiagram,
    ScalableError,
    ScaleId,
    from_scalable,
    to_scalable,
)
from archytaszx.diagram.validate import ValidateError
from archytaszx.repl import zxdraw, zxview
from archytaszx.repl.parser import (
    ParseError,
    ParseGrammarError,
    RuleStep,
    apply_step,
    parse,
    parse_step,
)
from archytaszx.repl.printer import (
    DEFAULT_MODE,
    NotationMode,
    PrinterError,
    drawing,
    render,
    render_detail,
    render_dirac,
    render_graph,
)
from archytaszx.rewrite.cache import CacheError
from archytaszx.rewrite.egraph import (
    DEFAULT_LIMITS,
    EGraph,
    Extraction,
    SaturationReport,
)
from archytaszx.rewrite.engine import DEFAULT_GUARD, RewriteResult, RewriteStep, apply
from archytaszx.rewrite.normal_form import NormalForm, normal_form
from archytaszx.rewrite.rule import RewriteError
from archytaszx.rewrite.rules_library import RULES, lookup_rule
from archytaszx.rewrite.sheet import (
    SheetStep,
    dissolve,
    enclose,
    fuse_sheet,
    join_scales,
    kill_scale,
    split_scale,
)
from archytaszx.rewrite.tactics import DEFAULT_SEARCH_LIMITS
from archytaszx.semantics import kernel
from archytaszx.semantics.certificate import (
    Certificate,
    CertificateError,
    CheckMethod,
    Derivation,
    DerivationKind,
    VerificationReport,
    certify,
    compare_structure,
    verify,
)
from archytaszx.semantics.check import CheckAssignmentValue, CheckError, ComparisonResult, compare
from archytaszx.semantics.contract_numeric import ContractError
from archytaszx.semantics.contract_symbolic import SymbolicContractionError
from archytaszx.semantics.decide import (
    DECIDE_SATURATION_LIMITS,
    DecideError,
    Decision,
    DecisionMethod,
    EqualityVerdict,
    decide_equal,
    sample_grid,
)
from archytaszx.semantics.denote import DenoteError
from archytaszx.semantics.induction import (
    DEFAULT_MAX_STEPS,
    InductionError,
    InductionResult,
    prove_by_induction,
)
from archytaszx.semantics.interop import (
    InteropError,
    RoundTripReport,
    SheetCheck,
    check_round_trip,
    check_sheet_step,
)
from archytaszx.semantics.prove import (
    InductionProof,
    ProofCertificate,
    ProofCheck,
    ProofOutcome,
    ProofStatus,
    ProveError,
    check_proof,
    prove,
)


class CommandError(Exception):
    """Base class for all errors raised by this module."""


class CommandGrammarError(CommandError):
    """A malformed command line: unknown option, bad assignment, missing argument."""


class CommandDomainError(CommandError):
    """A well-formed command the session or the engine cannot carry out."""


_ENGINE_ERRORS: tuple[type[Exception], ...] = (
    BangBoxError,
    CacheError,
    CertificateError,
    CheckError,
    ContractError,
    DecideError,
    DenoteError,
    DimensionError,
    GeneratorError,
    GraphError,
    InductionError,
    InteropError,
    ParseError,
    PhaseError,
    PrinterError,
    ProveError,
    RewriteError,
    ScalableError,
    ScalarError,
    SymbolicContractionError,
    ValidateError,
)
_ORACLE_ERRORS: tuple[type[Exception], ...] = (
    BangBoxError,
    CheckError,
    ContractError,
    DenoteError,
    DimensionError,
    GeneratorError,
    GraphError,
    PhaseError,
    ScalarError,
    ValidateError,
    MemoryError,
)
_NO_DIAGRAM = "no diagram loaded; use load"
_INDENT = "  "


# -- blocks, outcomes, session -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Text:
    """Mode-independent output text."""

    text: str

    def __post_init__(self) -> None:
        """Type check: CommandGrammarError on a non-str."""
        if not isinstance(self.text, str):
            raise CommandGrammarError(f"Text.text must be a str, got {type(self.text).__name__}")


@dataclass(frozen=True, slots=True)
class State:
    """An engine state that :func:`format_outcome` renders with ``printer.render``."""

    state: object


@dataclass(frozen=True, slots=True)
class Proof:
    """A proof artefact that :func:`format_outcome` writes out with :func:`proof_text`."""

    proof: object


Block: TypeAlias = "Text | State | Proof"


@dataclass(frozen=True, slots=True)
class Outcome:
    """One command's result: its canonical name, success, output blocks, full payload, and
    whether it changed the session state."""

    command: str
    ok: bool
    blocks: tuple[Block, ...]
    detail: object | None
    changed: bool

    def __post_init__(self) -> None:
        """Type checks: CommandGrammarError on a bad field."""
        if not isinstance(self.command, str):
            raise CommandGrammarError("Outcome.command must be a str")
        if not isinstance(self.ok, bool) or not isinstance(self.changed, bool):
            raise CommandGrammarError("Outcome.ok and Outcome.changed must be bool")
        if not isinstance(self.blocks, tuple) or not all(
            isinstance(b, (Text, State, Proof)) for b in self.blocks
        ):
            raise CommandGrammarError("Outcome.blocks must be a tuple of Text, State or Proof")


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    """A failed line: the line, the command, the exception class chain (``__cause__`` order),
    its messages, and the current diagram at the time."""

    line: str
    command: str
    error_types: tuple[str, ...]
    messages: tuple[str, ...]
    state: Diagram | None

    def __post_init__(self) -> None:
        """Type checks: CommandGrammarError on a bad field."""
        if not isinstance(self.line, str) or not isinstance(self.command, str):
            raise CommandGrammarError("ErrorDetail.line and command must be str")
        for name in ("error_types", "messages"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(isinstance(v, str) for v in value):
                raise CommandGrammarError(f"ErrorDetail.{name} must be a tuple of str")
        if self.state is not None and not isinstance(self.state, Diagram):
            raise CommandGrammarError("ErrorDetail.state must be a Diagram or None")


ProofArtefact: TypeAlias = "Certificate | ProofCertificate | InductionProof"


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """The undoable part of a session, taken before a state-changing command."""

    command: str
    current: Diagram | None
    start: Diagram | None
    results: tuple[RewriteResult, ...]
    scalable: ScalableDiagram | None
    sheet_steps: tuple[SheetStep, ...]


@dataclass(slots=True)
class Session:
    """REPL state: the current diagram, the derivation start and its replayable results, named
    diagrams, the scalable form and its sheet steps, the stored certificate (and the witness an
    induction certificate verifies at), the latest proof, the latest detail payload, and the
    undo stack."""

    current: Diagram | None = None
    start: Diagram | None = None
    results: list[RewriteResult] = field(default_factory=list)
    names: dict[str, Diagram] = field(default_factory=dict)
    scalable: ScalableDiagram | None = None
    sheet_steps: list[SheetStep] = field(default_factory=list)
    certificate: ProofArtefact | None = None
    certificate_witness: dict[str, CheckAssignmentValue] | None = None
    last_proof: ProofOutcome | InductionResult | None = None
    last_detail: object | None = None
    undo_stack: list[_Snapshot] = field(default_factory=list)

    def _snapshot(self, command: str) -> _Snapshot:
        return _Snapshot(
            command,
            self.current,
            self.start,
            tuple(self.results),
            self.scalable,
            tuple(self.sheet_steps),
        )

    def _restore(self, snap: _Snapshot) -> None:
        self.current = snap.current
        self.start = snap.start
        self.results = list(snap.results)
        self.scalable = snap.scalable
        self.sheet_steps = list(snap.sheet_steps)


@dataclass(frozen=True, slots=True)
class _Pair:
    """A comparison payload: its title, both sides, and the engine result."""

    title: str
    left: Diagram
    right: Diagram
    payload: object


# -- small helpers -------------------------------------------------------------------------


def _one_line(text: object) -> str:
    """``str(text)`` with whitespace runs collapsed; the type name when ``str`` fails."""
    try:
        shown = str(text)
    except Exception:  # noqa: BLE001
        shown = type(text).__name__
    return " ".join(shown.split())


def _expr(value: object) -> str:
    """``sstr`` of a value's sympy form (or ``str``), ``**`` shown as ``^``."""
    to_sympy = getattr(value, "to_sympy", None)
    text = sp.sstr(to_sympy()) if callable(to_sympy) else str(value)
    return str(text).replace("**", "^")


def _dirac(diagram: Diagram) -> str:
    try:
        return " ".join(render_dirac(diagram).splitlines())
    except PrinterError as exc:
        return f"unavailable ({_one_line(exc)})"


def _assignment_text(assignment: Mapping[str, object]) -> str:
    if not assignment:
        return "(no symbols)"
    return ", ".join(f"{k} = {_expr(assignment[k])}" for k in sorted(assignment))


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _free_symbols(diagram: Diagram) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """``diagram``'s free symbols as (all, dimension, multiplicity); a multiplicity symbol is
    never also a dimension."""
    dims: set[str] = set()
    others: set[str] = set(diagram.scalar.free_symbols)
    for node in diagram.nodes.values():
        for port in (*node.inputs, *node.outputs):
            dims |= port.dim.free_symbols
        if node.phase is not None:
            dims |= node.phase.dim.free_symbols
            others |= node.phase.free_symbols
    mults = set(free_mult_symbols(diagram))
    dims -= mults
    return frozenset(dims | others | mults), frozenset(dims), frozenset(mults)


def _require_current(session: Session) -> Diagram:
    if session.current is None:
        raise CommandDomainError(_NO_DIAGRAM)
    return session.current


def _start(session: Session) -> Diagram:
    _require_current(session)
    assert session.start is not None
    return session.start


def _ok(command: str, blocks: Iterable[Block], detail: object, changed: bool = False) -> Outcome:
    return Outcome(command, True, tuple(blocks), detail, changed)


def _fail(command: str, blocks: Iterable[Block], detail: object) -> Outcome:
    return Outcome(command, False, tuple(blocks), detail, False)


_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_NAME_RE = re.compile(r"@([A-Za-z_]\w*)")
_VALUE_RE = re.compile(r"(-?[0-9]{1,18})(?:/([0-9]{1,18}))?")
_AGAINST_RE = re.compile(r"(?:^|\s)against(?:\s|$)")


def _resolve(session: Session, text: str) -> tuple[Diagram, str]:
    """The diagram ``text`` names (``@NAME`` or a source) and a description of its origin."""
    source = text.strip()
    if not source:
        raise CommandGrammarError("expected a diagram source or @NAME")
    named = _NAME_RE.fullmatch(source)
    if named is not None:
        name = named[1]
        if name in ("start", "current"):
            diagram = session.start if name == "start" else session.current
            if diagram is None:
                raise CommandDomainError(_NO_DIAGRAM)
            return diagram.copy(), f"@{name}"
        if name not in session.names:
            raise CommandDomainError(f"no diagram named {name!r}; names lists them")
        return session.names[name].copy(), f"@{name}"
    parsed = parse(source)
    if isinstance(parsed, RuleStep):
        raise CommandGrammarError("a rule step names no diagram; use apply")
    return parsed.diagram, f"{parsed.form.value} input"


def _value(text: str) -> CheckAssignmentValue:
    match = _VALUE_RE.fullmatch(text)
    if match is None:
        raise CommandGrammarError(f"{text!r} is not an int or a p/q rational")
    if match[2] is None:
        return int(match[1])
    if int(match[2]) == 0:
        raise CommandGrammarError(f"{text!r} divides by zero")
    value = sp.Rational(int(match[1]), int(match[2]))
    return int(value) if value.is_Integer else value


def _split_against(text: str) -> tuple[str, str | None]:
    found = _AGAINST_RE.search(text)
    if found is None:
        return text, None
    target = text[found.end() :].strip()
    if not target:
        raise CommandGrammarError("against needs a diagram source or @NAME")
    return text[: found.start()], target


def _options(text: str, *, allow_on: bool) -> tuple[str | None, dict[str, CheckAssignmentValue]]:
    """``[on SYMBOL] [NAME=VALUE ...]`` (comma or space separated) as (symbol, assignment)."""
    tokens = [t for t in re.split(r"[\s,]+", re.sub(r"\s*=\s*", "=", text.strip())) if t]
    on: str | None = None
    assignment: dict[str, CheckAssignmentValue] = {}
    position = 0
    while position < len(tokens):
        token = tokens[position]
        position += 1
        if allow_on and token == "on":
            if on is not None or position >= len(tokens):
                raise CommandGrammarError("on takes exactly one multiplicity symbol")
            on = tokens[position]
            position += 1
            if not _IDENT_RE.fullmatch(on):
                raise CommandGrammarError(f"{on!r} is not a symbol name")
            continue
        name, sep, value = token.partition("=")
        if not sep or not _IDENT_RE.fullmatch(name):
            raise CommandGrammarError(f"expected NAME=VALUE, got {token!r}")
        if name in assignment:
            raise CommandGrammarError(f"{name} is assigned twice")
        assignment[name] = _value(value)
    return on, assignment


def _check_supplied(
    assignment: Mapping[str, CheckAssignmentValue], diagrams: Sequence[Diagram]
) -> None:
    """Refuse a supplied name free in no diagram, or a value outside its symbol's role."""
    free: set[str] = set()
    dims: set[str] = set()
    mults: set[str] = set()
    for diagram in diagrams:
        a, d, m = _free_symbols(diagram)
        free |= a
        dims |= d
        mults |= m
    unknown = sorted(set(assignment) - free)
    if unknown:
        raise CommandDomainError(
            f"{', '.join(unknown)} not free in the compared diagrams (free: "
            f"{', '.join(sorted(free)) or 'none'})"
        )
    for name, value in assignment.items():
        if name in mults and not (isinstance(value, int) and value >= 0):
            raise CommandDomainError(f"multiplicity {name} needs an int >= 0, got {value}")
        if name in dims and not (isinstance(value, int) and value >= 1):
            raise CommandDomainError(f"dimension {name} needs an int >= 1, got {value}")


def _samples(
    left: Diagram, right: Diagram, supplied: Mapping[str, CheckAssignmentValue]
) -> tuple[dict[str, CheckAssignmentValue], ...]:
    """``sample_grid(left, right)`` with ``supplied`` overriding, deduplicated in order."""
    seen: set[tuple[tuple[str, CheckAssignmentValue], ...]] = set()
    chosen: list[dict[str, CheckAssignmentValue]] = []
    for point in sample_grid(left, right):
        sample = {**dict(point), **supplied}
        key = tuple(sorted(sample.items()))
        if key not in seen:
            seen.add(key)
            chosen.append(sample)
    return tuple(chosen)


def _pair(session: Session, target: str | None) -> tuple[Diagram, Diagram]:
    """(start, current), or (current, TARGET) when a target is given."""
    current = _require_current(session)
    if target is None:
        return _start(session), current
    return current, _resolve(session, target)[0]


def _pair_args(session: Session, args: str) -> tuple[Diagram, Diagram]:
    """``[against] [TARGET]``."""
    text = args.strip()
    if text == "against" or text.startswith(("against ", "against\t")):
        text = text[len("against") :].strip()
        if not text:
            raise CommandGrammarError("against needs a diagram source or @NAME")
    return _pair(session, text or None)


def _dirac_outcome(left: Diagram, right: Diagram, relation: str) -> list[Block]:
    return [
        Text(f"dirac outcome: left {relation} right"),
        Text(f"{_INDENT}left:  {_dirac(left)}"),
        Text(f"{_INDENT}right: {_dirac(right)}"),
    ]


def _simplified(
    left: Diagram,
    right: Diagram,
    left_nf: NormalForm | None = None,
    right_nf: NormalForm | None = None,
) -> list[Block]:
    """Both sides' normal forms as State blocks with the guard bound, computing missing ones."""
    blocks: list[Block] = []
    for side, diagram, nf in (("left", left, left_nf), ("right", right, right_nf)):
        if nf is None:
            try:
                nf = normal_form(diagram)
            except _ENGINE_ERRORS as exc:
                blocks.append(Text(f"furthest-simplified {side}: unavailable ({_one_line(exc)})"))
                continue
        blocks.append(
            Text(
                f"furthest-simplified {side} (normal form, {_count(len(nf.results), 'step')}, "
                f"stop {nf.outcome.stop_reason.value}"
                f"{_guard_text(nf.outcome.stop_reason.value)}):"
            )
        )
        blocks.append(State(nf.diagram))
    return blocks


def _counterexample_line(assignment: Mapping[str, object], reason: str | None) -> Text:
    tail = f": {_one_line(reason)}" if reason else ""
    return Text(f"counterexample: {_assignment_text(assignment)}{tail}")


# -- commands ------------------------------------------------------------------------------

Handler = Callable[[Session, str], Outcome]


def _cmd_load(session: Session, args: str) -> Outcome:
    diagram, origin = _resolve(session, args)
    session.current = diagram
    session.start = diagram
    session.results = []
    session.last_proof = None
    session.certificate, session.certificate_witness = None, None
    return _ok("load", [Text(f"loaded {origin}"), State(diagram)], diagram, True)


def _cmd_show(session: Session, args: str) -> Outcome:
    what = args.strip()
    if not what:
        current = _require_current(session)
        return _ok("show", [State(current)], current)
    if what == "derivation":
        derivation = certify(
            _start(session), session.results, label="session derivation"
        ).derivation
        line = Text(f"derivation: {_count(len(session.results), 'step')} from the start diagram")
        return _ok("show", [line, State(derivation)], derivation)
    if what == "start":
        start = _start(session)
        return _ok("show", [State(start)], start)
    if what == "scalable":
        if session.scalable is None:
            raise CommandDomainError("no scalable diagram; use translate")
        return _ok("show", [State(session.scalable)], session.scalable)
    if what == "certificate":
        if session.certificate is None:
            raise CommandDomainError("no certificate; use certificate emit")
        return _ok("show", [Proof(session.certificate)], session.certificate)
    diagram, _origin = _resolve(session, what)
    return _ok("show", [State(diagram)], diagram)


def _rule_lines(name: str) -> list[str]:
    rule = RULES[name]
    guards = ", ".join(
        g.kind.value if g.value is None else f"{g.kind.value} {g.value}"
        for g in rule.dimension_guards
    )
    return [
        f"rule: {name}",
        (
            f"{_INDENT}quantifiers: leg counts {', '.join(rule.quantifiers.leg_counts) or 'none'}; "
            f"dimensions {', '.join(rule.quantifiers.dimensions) or 'none'}"
        ),
        f"{_INDENT}side conditions: {', '.join(c.name for c in rule.side_conditions) or 'none'}",
        f"{_INDENT}dimension guards: {guards or 'none'}",
        f"{_INDENT}scalar: {_expr(rule.scalar_introduced)}",
    ]


def _rule_name(text: str) -> str:
    name = text.strip()
    if name not in RULES:
        raise CommandDomainError(f"no rule {name!r}; rules lists them")
    return name


def _cmd_rules(session: Session, args: str) -> Outcome:
    lines = _rule_lines(_rule_name(args)) if args.strip() else list(RULES)
    return _ok("rules", [Text(line) for line in lines], Text("\n".join(lines)))


def _cmd_match(session: Session, args: str) -> Outcome:
    current = _require_current(session)
    names = [_rule_name(args)] if args.strip() else list(RULES)
    lines: list[str] = []
    for name in names:
        try:
            matches = RULES[name].pattern.find_matches(current)
        except _ENGINE_ERRORS as exc:
            lines.append(f"{name}: matcher refused ({_one_line(exc)})")
            continue
        for index, found in enumerate(matches):
            nodes = ", ".join(f"n{int(n)}" for n in sorted(found.support_node_ids, key=int))
            lines.append(f"#{index} {name} at {nodes}")
    if not any(line.startswith("#") for line in lines):
        lines.append("no matches")
    return _ok("match", [Text(line) for line in lines], Text("\n".join(lines)))


def _cmd_apply(session: Session, args: str) -> Outcome:
    current = _require_current(session)
    step = parse_step(f"apply {args.strip()}")
    result = apply_step(current, step)
    session.results.append(result)
    session.current = result.diagram
    return _ok("apply", [State(result)], result, True)


def _cmd_check(session: Session, args: str) -> Outcome:
    head, target = _split_against(args)
    _on, supplied = _options(head, allow_on=False)
    left, right = _pair(session, target)
    _check_supplied(supplied, (left, right))
    samples = _samples(left, right, supplied)
    refusals: list[str] = []
    evaluated = 0
    for sample in samples:
        missing = _uncovered(sample, (left, right))
        if missing:
            refusals.append(f"{_assignment_text(sample)} (no sample value for {missing})")
            continue
        try:
            result = compare(left, right, sample)
        except _ORACLE_ERRORS as exc:
            refusals.append(f"{_assignment_text(sample)} ({_one_line(exc)})")
            continue
        evaluated += 1
        if not result.matched:
            payload = _Pair("check", left, right, Text(_comparison_text(sample, result)))
            blocks = [
                Text(f"check: mismatch after {_count(evaluated, 'sample')}"),
                *_dirac_outcome(left, right, "!="),
                _counterexample_line(sample, result.reason),
            ]
            return _fail("check", blocks, payload)
    names = sorted(set().union(*(set(s) for s in samples)))
    over = f" over {', '.join(names)}" if names else ""
    lines = [f"check: no mismatch in {_count(evaluated, 'sample')}{over}"]
    if refusals:
        lines.append(f"{_count(len(refusals), 'sample')} refused, first at {refusals[0]}")
    payload = _Pair("check", left, right, Text("\n".join(lines)))
    if evaluated == 0:
        return _fail(
            "check", [Text("check: no sample could be evaluated"), *map(Text, lines)], payload
        )
    return _ok("check", [Text(line) for line in lines], payload)


def _comparison_text(sample: Mapping[str, object], result: ComparisonResult) -> str:
    return (
        f"mismatch at {_assignment_text(sample)}: {result.reason} "
        f"(max deviation {result.max_abs_deviation:g}, mode {result.mode.value})"
    )


def _uncovered(sample: Mapping[str, object], diagrams: Sequence[Diagram]) -> str:
    """The free symbols of ``diagrams`` that ``sample`` leaves to the parameter environment,
    comma separated, or ``""``."""
    free: set[str] = set()
    for diagram in diagrams:
        free |= _free_symbols(diagram)[0]
    return ", ".join(sorted(free - set(sample)))


def _guard_text(stop: str) -> str:
    """The guard bound behind a normal-form stop reason value, or ``""`` at a fixpoint."""
    bounds = {
        "step_limit": f"guard max_steps {DEFAULT_GUARD.max_steps}",
        "node_limit": f"guard max_nodes {DEFAULT_GUARD.max_nodes}",
        "loop_detected": "guard loop detection",
    }
    return f" ({bounds[stop]})" if stop in bounds else ""


def _cmd_normalize(session: Session, args: str) -> Outcome:
    if args.strip():
        raise CommandGrammarError("normalize takes no arguments")
    nf = normal_form(_require_current(session))
    session.current = nf.diagram
    session.results.extend(nf.results)
    stop = nf.outcome.stop_reason.value
    line = Text(f"normalized in {_count(len(nf.results), 'step')}: {stop}{_guard_text(stop)}")
    return _ok("normalize", [State(nf.diagram), line], nf, bool(nf.results))


def _cmd_decide(session: Session, args: str) -> Outcome:
    left, right = _pair_args(session, args)
    decision = decide_equal(left, right)
    payload = _Pair("decide", left, right, decision)
    head = f"decide: {decision.verdict.value} by {decision.method.value}: {decision.reason}"
    if decision.verdict is EqualityVerdict.EQUAL:
        blocks: list[Block] = [Text(head)]
        if decision.assumptions:
            blocks.append(
                Text(
                    "conditional on "
                    f"{_count(len(decision.assumptions), 'deferred dimension constraint')}"
                )
            )
        return _ok("decide", blocks, payload)
    relation = "!=" if decision.verdict is EqualityVerdict.UNEQUAL else "?="
    blocks = [Text(head), *_dirac_outcome(left, right, relation)]
    if decision.counterexample is not None:
        reason = decision.comparison.reason if decision.comparison is not None else None
        blocks.append(_counterexample_line(decision.counterexample, reason))
    elif decision.induction is not None and decision.induction.counterexample is not None:
        blocks.append(_induction_counterexample(decision.induction))
    elif decision.method is not DecisionMethod.INTERFACE:
        blocks += _simplified(left, right, decision.left_nf, decision.right_nf)
        if decision.saturation is not None:
            limits = DECIDE_SATURATION_LIMITS
            blocks.append(
                Text(
                    f"bound: saturation stop {decision.saturation.stop_reason.value} (limits "
                    f"{limits.max_iterations} iterations, {limits.max_enodes} e-nodes, "
                    f"{limits.max_applications} applications)"
                )
            )
    return _fail("decide", blocks, payload)


def _induction_counterexample(result: InductionResult) -> Text:
    witness = f" with {_assignment_text(result.witness)}" if result.witness else ""
    reason = result.comparison.reason if result.comparison is not None else result.reason
    return Text(f"counterexample: {result.index} = {result.counterexample}{witness}: {reason}")


def _cmd_saturate(session: Session, args: str) -> Outcome:
    if args.strip():
        raise CommandGrammarError("saturate takes no arguments")
    current = _require_current(session)
    graph = EGraph()
    root = graph.add(current)
    report = graph.saturate(None, limits=DEFAULT_LIMITS)
    extraction = graph.extract(graph.class_of(root))
    chained = extraction.root == root
    if chained:
        session.results.extend(extraction.results)
        session.current = extraction.diagram
        if not extraction.results:
            session.current = current
    else:
        session.current = session.start = extraction.diagram
        session.results = []
    blocks: list[Block] = [State(session.current), Text(_saturation_line(report, extraction))]
    if not chained:
        blocks.append(Text("derivation restarted at the extracted diagram"))
    changed = bool(extraction.results) or not chained
    return _ok("saturate", blocks, (extraction, report), changed)


def _saturation_line(report: SaturationReport, extraction: Extraction) -> str:
    bound = ""
    if report.stop_reason.value != "saturated":
        limits = DEFAULT_LIMITS
        bound = (
            f" (limits {limits.max_iterations} iterations, {limits.max_enodes} e-nodes, "
            f"{limits.max_applications} applications)"
        )
    return (
        f"saturation {report.stop_reason.value}{bound}: "
        f"{_count(report.iterations, 'iteration')}, "
        f"{report.enodes_added} e-node(s) added, {_count(report.merges, 'merge')}, "
        f"{_count(report.applications, 'application')}; extracted cost {extraction.cost} "
        f"in {_count(len(extraction.results), 'step')}"
    )


def _cmd_prove(session: Session, args: str) -> Outcome:
    left, right = _pair_args(session, args)
    outcome = prove(left, right)
    session.last_proof = outcome
    payload = _Pair("prove", left, right, outcome)
    head = f"prove: {outcome.status.value}: {outcome.reason}"
    if outcome.proved:
        kind = "induction" if outcome.induction is not None else "search"
        return _ok("prove", [Text(head), Text(f"proof by {kind}; certificate emits it")], payload)
    relation = "!=" if outcome.status is ProofStatus.REFUTED else "?="
    blocks: list[Block] = [Text(head), *_dirac_outcome(left, right, relation)]
    if outcome.counterexample is not None:
        reason = None
        if outcome.check is not None and outcome.check.comparison is not None:
            reason = outcome.check.comparison.reason
        blocks.append(_counterexample_line(outcome.counterexample, reason))
    elif outcome.status is ProofStatus.CHECK_FAILED and outcome.check is not None:
        blocks.append(Text(f"check failed: {outcome.check.reason}"))
    elif outcome.status is not ProofStatus.REFUTED:
        blocks += _simplified(left, right)
        limits = DEFAULT_SEARCH_LIMITS
        status = outcome.search.status.value if outcome.search is not None else "none"
        blocks.append(
            Text(
                f"bound: search {status} (limits depth {limits.max_depth}, states "
                f"{limits.max_states}, applications {limits.max_applications})"
            )
        )
    return _fail("prove", blocks, payload)


def _cmd_induct(session: Session, args: str) -> Outcome:
    head, target = _split_against(args)
    index, supplied = _options(head, allow_on=True)
    left, right = _pair(session, target)
    _check_supplied(supplied, (left, right))
    if index is not None and index in supplied:
        raise CommandDomainError(f"{index} is the induction index; it takes no value")
    all_l, _d, mults_l = _free_symbols(left)
    all_r, _d, mults_r = _free_symbols(right)
    live = sorted(mults_l | mults_r)
    if not live:
        raise CommandDomainError("no bang box: induct needs a free multiplicity symbol")
    if index is not None and index not in live:
        raise CommandDomainError(f"{index} is no free multiplicity symbol ({', '.join(live)})")
    if index is None and live[0] in supplied:
        raise CommandDomainError(f"{live[0]} is the induction index; it takes no value")
    chosen = index if index is not None else live[0]
    env = {**dict(right.parameters), **dict(left.parameters)}
    required = (all_l | all_r) - mults_l - mults_r
    witness: dict[str, CheckAssignmentValue] = {
        name: env[name] for name in sorted(required) if name in env
    }
    witness.update(supplied)
    held = {name for name in live if name != chosen}
    missing = sorted((required | held) - set(witness))
    if missing:
        raise CommandDomainError(
            f"induct holds {', '.join(missing)} at a value; supply NAME=VALUE for each"
        )
    result = prove_by_induction(left, right, index=index, witness=witness)
    session.last_proof = result
    payload = _Pair("induct", left, right, result)
    discharge = result.discharge.value if result.discharge is not None else "none"
    line = (
        f"induct: {result.verdict.value} over {result.index} from {result.base_value} "
        f"via {discharge}: {result.reason}"
    )
    blocks: list[Block] = [Text(line), Text(f"witness: {_assignment_text(result.witness)}")]
    if result.proved:
        blocks.append(
            Text(
                "induction certificate available; certificate emits it"
                if result.derivation is not None
                else "no induction certificate was built"
            )
        )
        return _ok("induct", blocks, payload)
    relation = "!=" if result.counterexample is not None else "?="
    blocks += _dirac_outcome(left, right, relation)
    if result.counterexample is not None:
        blocks.append(_induction_counterexample(result))
    elif result.discharge is not None:
        blocks += [Text(f"tier {t.discharge.value}: {_one_line(t.reason)}") for t in result.tiers]
        blocks.append(
            Text(f"bound: only the {result.discharge.value} tier settled the step; not a proof")
        )
    else:
        blocks += _simplified(left, right)
        for tier in result.tiers:
            blocks.append(Text(f"tier {tier.discharge.value}: {_one_line(tier.reason)}"))
        blocks.append(
            Text(f"bound: no discharge tier settled the step (max_steps {DEFAULT_MAX_STEPS})")
        )
    return _fail("induct", blocks, payload)


def _session_certificate(session: Session) -> Certificate:
    return certify(_start(session), session.results, label="session derivation")


def _emit(
    session: Session, which: str
) -> tuple[ProofArtefact, dict[str, CheckAssignmentValue] | None]:
    """The certificate ``which`` selects: ``derivation``, ``induction``, ``proof``, or ``""``
    for the latest proof artefact falling back to the session derivation."""
    proof = session.last_proof
    if which in ("", "proof") and isinstance(proof, ProofOutcome) and proof.proved:
        artefact = proof.certificate if proof.certificate is not None else proof.induction
        if artefact is not None:
            return artefact, None
    if (
        which in ("", "induction")
        and isinstance(proof, InductionResult)
        and proof.proved
        and proof.derivation is not None
    ):
        certificate = Certificate(proof.derivation, CheckMethod.INDUCTION)
        return certificate, dict(proof.witness)
    if which == "induction" and isinstance(proof, ProofOutcome) and proof.induction is not None:
        return proof.induction, None
    if which in ("", "derivation"):
        return _session_certificate(session), None
    if which == "induction" and isinstance(proof, InductionResult) and proof.proved:
        raise CommandDomainError(f"the last induct built no certificate: {proof.reason}")
    raise CommandDomainError(f"no {which} certificate; prove or induct first")


def _artefact_line(artefact: ProofArtefact) -> str:
    if isinstance(artefact, Certificate):
        return (
            f"certificate: {artefact.check_method.value}, {artefact.derivation.kind.value}, "
            f"{_count(len(artefact.steps), 'step')}"
        )
    if isinstance(artefact, ProofCertificate):
        return f"certificate: proof, {_count(artefact.length, 'step')}"
    return f"certificate: induction proof over {artefact.index}, {_count(artefact.length, 'step')}"


def _math(diagram: Diagram, u: bool) -> str:
    """The Dirac line of ``diagram`` without a leading catalog name."""
    try:
        text = " ".join(render_dirac(diagram, unicode=u).splitlines())
    except PrinterError as exc:
        return f"[no Dirac form: {_one_line(exc)}]"
    named = re.match(r"[^\s|<>⟨⟩=]+ = (.+)", text)
    return named.group(1) if named else text


def _because(step: RewriteStep, backwards: bool) -> str:
    nodes = ", ".join(f"n{int(n)}" for n in sorted(step.consumed_node_ids, key=int))
    where = f" at {nodes}" if nodes else ""
    return f"by {step.rule_name}{where}{', read backwards' if backwards else ''}"


def _chain(derivation: Derivation, u: bool, backwards: bool = False) -> list[tuple[str, str]]:
    """``(diagram, justification)`` rows of a step sequence, reversed when ``backwards``."""
    states, _failure = _replayed_states(derivation)
    texts = [_math(x, u) if isinstance(x, Diagram) else f"[{x}]" for x in states]
    rows = [(texts[0], "")]
    rows += [(texts[k], _because(step, False)) for k, step in enumerate(derivation.steps, 1)]
    if not backwards:
        return rows
    flipped = [(rows[-1][0], "")]
    for k in range(len(rows) - 1, 0, -1):
        flipped.append((rows[k - 1][0], _because(derivation.steps[k - 1], True)))
    return flipped


def _display(rows: list[tuple[str, str]], u: bool) -> list[str]:
    """Rows as ``  A`` then ``= B    (by rule)`` lines, justifications aligned."""
    width = max(len(text) for text, _ in rows)
    out = []
    for k, (text, why) in enumerate(rows):
        head = f"{'  ' if k == 0 else '= '}{text}"
        out.append(f"{head.ljust(width + 2)}    ({why})" if why else head)
    return out


def _merge(rows: list[tuple[str, str]], more: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """``rows`` then ``more``, joined at a shared diagram or by the meeting of the halves."""
    if rows and more and rows[-1][0] == more[0][0]:
        return rows + more[1:]
    return rows + [(more[0][0], "the same diagram"), *more[1:]] if more else rows


def _proof_rows(proof: ProofArtefact, u: bool) -> list[tuple[str, str]] | None:
    """The equality chain of a step-sequence proof, None for an induction."""
    if isinstance(proof, Certificate):
        if proof.derivation.kind is DerivationKind.STEP_SEQUENCE:
            return _chain(proof.derivation, u)
        return None
    if isinstance(proof, ProofCertificate):
        forward = _chain(proof.forward.derivation, u)
        return _merge(forward, _chain(proof.backward.derivation, u, backwards=True))
    return None


def proof_text(proof: ProofArtefact, *, unicode: bool = False) -> str:
    """``proof`` written as a theorem and its proof: a chain of equalities, each justified by
    the rule applied, or a base case and an inductive step; closed by an end mark."""
    u = unicode
    end = "∎" if u else "QED"
    if isinstance(proof, Certificate):
        left, right = proof.initial, proof.final
    else:
        left, right = proof.start, proof.goal
    lines = [
        "Theorem.",
        f"  {_math(left, u)}",
        f"    = {_math(right, u)}",
        "",
        *_proof_body(proof, u),
    ]
    method = proof.check_method.value if isinstance(proof, Certificate) else "replay"
    lines.append(f"{end}  (checked by {method.replace('_', ' ')})")
    return "\n".join(lines)


def _proof_body(proof: ProofArtefact, u: bool, depth: int = 0) -> list[str]:
    pad = _INDENT * depth
    rows = _proof_rows(proof, u)
    if rows is not None:
        if len(rows) == 1:
            return [f"{pad}Proof. Both sides are the same diagram."]
        return [f"{pad}Proof.", *(pad + _INDENT + line for line in _display(rows, u))]
    if isinstance(proof, Certificate):
        d = proof.derivation
        claim = d.induction
        index = claim.symbol if claim else "n"
        base = claim.base_value if claim else 0
        lines = [f"{pad}Proof by induction on {index}."]
        held = ", ".join(claim.held_symbolic) if claim and claim.held_symbolic else ""
        if held:
            lines.append(f"{pad}{_INDENT}({held} held symbolic.)")
        names = ("Base case", "Inductive step") if len(d.children) == 2 else None
        for k, child in enumerate(d.children):
            title = names[k] if names else f"Part {k + 1}"
            if names and k == 0:
                title += f" ({index} = {base})"
            if names and k == 1:
                title += f" (assume the claim at {index}, show it at {index} + 1)"
            how = claim.step_discharge if (names and k == 1 and claim) else ""
            lines.append(f"{pad}{_INDENT}{title}{', by ' + how if how else ''}.")
            lines += _proof_body(Certificate(child), u, depth + 2)[1:] or [
                f"{pad}{_INDENT * 2}(no rewrite steps)"
            ]
        return lines
    assert isinstance(proof, InductionProof)
    region = ", ".join(f"n{int(n)}" for n in sorted(proof.region, key=int))
    side = "left" if proof.side.value == "start" else "right"
    idx, sym = proof.index, proof.step_symbol
    return [
        f"{pad}Proof by induction on {idx}.",
        f"{pad}{_INDENT}Base case ({idx} = 0).",
        *_proof_body(proof.base, u, depth + 2)[1:],
        f"{pad}{_INDENT}Inductive step. Assume the claim at {idx} = {sym}; at {idx} = {sym} + 1:",
        f"{pad}{_INDENT * 2}the {side} side contains its own instance at {sym} (nodes {region}):",
        *(pad + _INDENT * 3 + line for line in _display(_chain(proof.expose.derivation, u), u)),
        f"{pad}{_INDENT * 2}replacing it by the induction hypothesis, the other side follows:",
        *_proof_body(proof.step, u, depth + 3)[1:],
    ]


def _cmd_certificate(session: Session, args: str) -> Outcome:
    words = args.split(None, 1)
    action = words[0] if words else "emit"
    rest = words[1] if len(words) > 1 else ""
    if action == "emit":
        which = rest.strip()
        if which not in ("", "derivation", "induction", "proof"):
            raise CommandGrammarError("certificate emit [derivation|induction|proof]")
        artefact, witness = _emit(session, which)
        session.certificate, session.certificate_witness = artefact, witness
        return _ok("certificate", _proof_blocks(artefact), artefact)
    if action == "show":
        if rest.strip():
            raise CommandGrammarError("certificate show takes no arguments")
        if session.certificate is None:
            raise CommandDomainError("no certificate; use certificate emit")
        artefact = session.certificate
        return _ok("certificate", _proof_blocks(artefact), artefact)
    if action == "verify":
        return _verify(session, rest)
    if action in ("kernel", "lean"):
        return _kernel(session, action, rest.strip())
    raise CommandGrammarError(
        f"unknown certificate action {action!r}; use emit, verify, show, kernel or lean"
    )


def _kernel(session: Session, action: str, path: str) -> Outcome:
    """``certificate kernel`` checks the certificate in the proof kernel; ``certificate lean
    [FILE]`` also prints its Lean 4 export, or writes it to FILE."""
    if action == "kernel" and path:
        raise CommandGrammarError("certificate kernel takes no arguments")
    blocks: list[Block] = []
    if session.certificate is None:
        artefact, witness = _emit(session, "")
        session.certificate, session.certificate_witness = artefact, witness
        blocks.append(Text(f"emitted {_artefact_line(session.certificate)}"))
    try:
        theorem = kernel.check(session.certificate)
    except (kernel.KernelError, *_ORACLE_ERRORS) as exc:  # type: ignore[misc]
        text = f"kernel: rejected: {_one_line(exc)}"
        return _fail("certificate", [*blocks, Text(text)], Text(text))
    left, right = _math(theorem.lhs, False), _math(theorem.rhs, False)
    names = ", ".join(dict.fromkeys(theorem.axioms())) or "none"
    over = f" for all {', '.join(theorem.variables)}" if theorem.variables else ""
    blocks.append(Text(f"kernel: theorem {left} = {right}{over}"))
    blocks.append(Text(f"  {_count(theorem.size(), 'inference')}; rules: {names}"))
    blocks += [Text(f"  assumes {a}") for a in theorem.assumptions]
    if action == "lean":
        source = kernel.lean(theorem, describe=lambda d: _math(d, True))
        if "import ArchytasZX" in source:
            unproved = source.count("\naxiom unproved_")
            note = f"{unproved} unproved step(s)" if unproved else "every step proved"
            blocks.append(Text(f"lean: tensor-network semantics, {note}"))
        else:
            blocks.append(Text("lean: opaque diagrams, every step an axiom"))
        if path:
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(source)
            except OSError as exc:
                raise CommandDomainError(f"cannot write {path}: {exc}") from exc
            blocks.append(Text(f"lean: wrote {path}"))
        else:
            blocks.append(Text(source.rstrip("\n")))
    return _ok("certificate", blocks, theorem)


def _proof_blocks(artefact: ProofArtefact) -> list[Block]:
    return [Text(_artefact_line(artefact)), Proof(artefact)]


def _verify(session: Session, args: str) -> Outcome:
    _on, supplied = _options(args, allow_on=False)
    blocks: list[Block] = []
    if session.certificate is None:
        artefact, witness = _emit(session, "")
        session.certificate, session.certificate_witness = artefact, witness
        blocks.append(Text(f"emitted {_artefact_line(artefact)}"))
    artefact = session.certificate
    blocks.append(Text(f"verifying {_artefact_line(artefact)}"))
    if isinstance(artefact, Certificate):
        left, right = artefact.initial, artefact.final
        _check_supplied(supplied, (left, right))
        if artefact.check_method is CheckMethod.INDUCTION:
            witness = session.certificate_witness or {}
            clash = sorted(k for k, v in supplied.items() if k not in witness or witness[k] != v)
            if clash:
                raise CommandDomainError(
                    f"an induction certificate verifies only at its witness "
                    f"{_assignment_text(witness)}; {', '.join(clash)} differs"
                )
            points: tuple[dict[str, CheckAssignmentValue], ...] = (dict(witness),)
        else:
            points = _samples(left, right, supplied)
        report: VerificationReport | None = None
        refusals: list[str] = []
        evaluated = 0
        for point in points:
            induction = artefact.check_method is CheckMethod.INDUCTION
            missing = "" if induction else _uncovered(point, (left, right))
            if missing:
                refusals.append(f"{_assignment_text(point)} (no sample value for {missing})")
                continue
            try:
                report = verify(artefact, point)
            except _ORACLE_ERRORS as exc:
                refusals.append(f"{_assignment_text(point)} ({_one_line(exc)})")
                continue
            evaluated += 1
            if not report.verified:
                blocks += [
                    Text(f"verify: failed at {_assignment_text(point)}: {report.reason}"),
                    *_dirac_outcome(left, right, "?="),
                ]
                if report.comparison is not None and not report.comparison.matched:
                    blocks.append(_counterexample_line(point, report.comparison.reason))
                return _fail("certificate", blocks, report)
        note = ""
        if refusals:
            note = f"; {_count(len(refusals), 'point')} refused, first at {refusals[0]}"
        if report is None:
            text = f"verify: no point evaluated{note}"
            return _fail("certificate", [*blocks, Text(text)], Text(text))
        blocks.append(
            Text(f"verify: verified at {_count(evaluated, 'point')}: {report.reason}{note}")
        )
        return _ok("certificate", blocks, report)
    left, right = artefact.start, artefact.goal
    _check_supplied(supplied, (left, right))
    samples = _samples(left, right, supplied)
    piecewise = isinstance(artefact, InductionProof) and not supplied
    check = check_proof(artefact, samples=None if piecewise else samples, max_samples=len(samples))
    if not check.verified:
        blocks += [Text(f"verify: failed: {check.reason}"), *_dirac_outcome(left, right, "?=")]
        if check.counterexample is not None:
            reason = check.comparison.reason if check.comparison is not None else None
            blocks.append(_counterexample_line(check.counterexample, reason))
        return _fail("certificate", blocks, check)
    blocks.append(Text(f"verify: verified: {check.reason}"))
    return _ok("certificate", blocks, check)


_SCALE_RE = r"s?([0-9]{1,9})"
_REF_RE = r"n([0-9]{1,9})\.(out|in)([0-9]{1,9})"


def _scale(text: str) -> ScaleId:
    match = re.fullmatch(_SCALE_RE, text.strip())
    if match is None:
        raise CommandGrammarError(f"{text!r} is not a scale id like s0")
    return ScaleId(int(match[1]))


def _mult(text: str) -> Mult | int:
    """A multiplicity: ``+`` of ``*`` products of ints and symbols."""
    total: Mult | None = None
    terms = text.split("+")
    for term in terms:
        product: Mult | None = None
        for factor in term.split("*"):
            word = factor.strip()
            if re.fullmatch(r"[0-9]{1,18}", word):
                piece = Mult(int(word))
            elif _IDENT_RE.fullmatch(word):
                piece = Mult(word)
            else:
                raise CommandGrammarError(f"{text!r} is not a multiplicity (ints, symbols, +, *)")
            product = piece if product is None else product * piece
        assert product is not None
        total = product if total is None else total + product
    assert total is not None
    if len(terms) == 1 and re.fullmatch(r"\s*[0-9]{1,18}\s*", text):
        return int(text)
    return total


def _port(node: str, side: str, index: str) -> PortRef:
    direction = Direction.OUTPUT if side == "out" else Direction.INPUT
    return PortRef(NodeId(int(node)), direction, int(index))


def _sheet_step(s: ScalableDiagram, op: str, rest: str) -> SheetStep:
    words = rest.split()
    if op in ("dissolve", "kill"):
        if len(words) != 1:
            raise CommandGrammarError(f"translate {op} S")
        return (dissolve if op == "dissolve" else kill_scale)(s, _scale(words[0]))
    if op == "split":
        if len(words) < 2:
            raise CommandGrammarError("translate split S FIRST")
        return split_scale(s, _scale(words[0]), _mult(" ".join(words[1:])))
    if op == "join":
        if len(words) != 2:
            raise CommandGrammarError("translate join S S")
        return join_scales(s, _scale(words[0]), _scale(words[1]))
    if op == "enclose":
        found = re.fullmatch(
            r"(n?[0-9]{1,9}(?:\s*,\s*n?[0-9]{1,9})*)(?:\s+x\s+(.+))?", rest.strip()
        )
        if found is None:
            raise CommandGrammarError("translate enclose nA, nB [x MULT]")
        nodes = [NodeId(int(t.strip().lstrip("n"))) for t in found[1].split(",")]
        multiplicity = 1 if found[2] is None else _mult(found[2])
        return enclose(s, nodes, multiplicity=multiplicity)
    if op == "fuse":
        found = re.fullmatch(rf"{_REF_RE}\s*--\s*{_REF_RE}", rest.strip())
        if found is None:
            raise CommandGrammarError("translate fuse nA.outI -- nB.inJ")
        wire = Wire(_port(found[1], found[2], found[3]), _port(found[4], found[5], found[6]))
        return fuse_sheet(s, wire)
    raise CommandGrammarError(
        f"unknown translate operation {op!r}; use back, dissolve, split, join, enclose, kill, fuse"
    )


def _cmd_translate(session: Session, args: str) -> Outcome:
    words = args.split(None, 1)
    if not words:
        current = _require_current(session)
        report = check_round_trip(current, samples=sample_grid(current, current))
        session.scalable = report.scalable
        session.sheet_steps = []
        status = "ok" if report.ok else "failed"
        blocks: list[Block] = [
            Text(f"round trip {status}: {report.reason}"),
            State(report.scalable),
        ]
        if not report.ok:
            blocks += _dirac_outcome(current, report.back, "?=")
            if report.counterexample is not None:
                blocks.append(_counterexample_line(report.counterexample, None))
        return Outcome("translate", report.ok, tuple(blocks), report, True)
    op, rest = words[0], (words[1] if len(words) > 1 else "")
    if op == "back":
        if rest.strip():
            raise CommandGrammarError("translate back takes no arguments")
        if session.scalable is None:
            raise CommandDomainError("no scalable diagram; use translate")
        back = from_scalable(session.scalable)
        blocks = [State(back)]
        existing = session.current
        if existing is not None and not session.sheet_steps:
            back = existing
            blocks = [Text("translated back; no sheet step, the current diagram stands")]
            blocks.append(State(existing))
        elif existing is not None and compare_structure(back, existing).identical:
            blocks.insert(0, Text("translated back; identical to the current diagram"))
        else:
            session.current = session.start = back
            session.results = []
            blocks.insert(
                0, Text("translated back; derivation restarted at the translated diagram")
            )
        session.scalable = None
        session.sheet_steps = []
        return _ok("translate", blocks, back, True)
    blocks = []
    if session.scalable is None:
        session.scalable = to_scalable(_require_current(session))
        blocks.append(Text("translated the current diagram to scalable notation"))
    step = _sheet_step(session.scalable, op, rest)
    session.scalable = step.after
    session.sheet_steps.append(step)
    before: Diagram | None = None
    after: Diagram | None = None
    samples: tuple[dict[str, CheckAssignmentValue], ...] = ()
    try:
        before, after = from_scalable(step.before), from_scalable(step.after)
        samples = _samples(before, after, {})
    except _ENGINE_ERRORS:
        before = after = None
    check = check_sheet_step(step, samples=samples)
    blocks += [State(step), Text(f"sheet step {'ok' if check.ok else 'failed'}: {check.reason}")]
    if not check.ok and before is not None and after is not None:
        blocks += _dirac_outcome(before, after, "?=")
        if check.counterexample is not None:
            blocks.append(_counterexample_line(check.counterexample, None))
    return Outcome("translate", check.ok, tuple(blocks), check, True)


def _cmd_name(session: Session, args: str) -> Outcome:
    name = args.strip()
    if not _IDENT_RE.fullmatch(name):
        raise CommandGrammarError(f"name NAME: {name!r} is not an identifier")
    if name in ("start", "current"):
        raise CommandDomainError(f"{name} is built in")
    current = _require_current(session)
    session.names[name] = current.copy()
    return _ok("name", [Text(f"named @{name}")], current)


def _cmd_names(session: Session, args: str) -> Outcome:
    if args.strip():
        raise CommandGrammarError("names takes no arguments")
    lines = [f"@{n}: {_dirac(session.names[n])}" for n in sorted(session.names)] or ["no names"]
    return _ok("names", [Text(line) for line in lines], Text("\n".join(lines)))


def _cmd_undo(session: Session, args: str) -> Outcome:
    if args.strip():
        raise CommandGrammarError("undo takes no arguments")
    if not session.undo_stack:
        raise CommandDomainError("nothing to undo")
    snap = session.undo_stack.pop()
    session._restore(snap)
    blocks: list[Block] = [Text(f"undid {snap.command}")]
    if session.current is not None:
        blocks.append(State(session.current))
    return _ok("undo", blocks, session.current)


def _cmd_reset(session: Session, args: str) -> Outcome:
    if args.strip():
        raise CommandGrammarError("reset takes no arguments")
    current = _require_current(session)
    session.start = current
    session.results = []
    return _ok("reset", [Text("derivation restarted at the current diagram")], current, True)


def _cmd_detail(session: Session, args: str) -> Outcome:
    if args.strip():
        raise CommandGrammarError("detail takes no arguments")
    if session.last_detail is None:
        raise CommandDomainError("nothing to detail yet")
    return _ok("detail", [Text(detail_text(session.last_detail))], session.last_detail)


def _cmd_help(session: Session, args: str) -> Outcome:
    word = args.strip()
    if not word:
        lines = [HELP[name] for name in sorted(HELP)]
        lines.append(f"aliases: {', '.join(f'{a} = {c}' for a, c in sorted(ALIASES.items()))}")
        lines.append("any other line is load <line>")
    else:
        name = ALIASES.get(word, word)
        if name not in HELP:
            raise CommandDomainError(f"no command {word!r}; help lists them")
        lines = [HELP[name]]
    return _ok("help", [Text(line) for line in lines], Text("\n".join(lines)))


COMMANDS: Mapping[str, Handler] = MappingProxyType(
    {
        "load": _cmd_load,
        "show": _cmd_show,
        "rules": _cmd_rules,
        "match": _cmd_match,
        "apply": _cmd_apply,
        "check": _cmd_check,
        "normalize": _cmd_normalize,
        "decide": _cmd_decide,
        "saturate": _cmd_saturate,
        "prove": _cmd_prove,
        "induct": _cmd_induct,
        "certificate": _cmd_certificate,
        "translate": _cmd_translate,
        "name": _cmd_name,
        "names": _cmd_names,
        "undo": _cmd_undo,
        "reset": _cmd_reset,
        "detail": _cmd_detail,
        "help": _cmd_help,
    }
)
"""Every command by canonical name."""

ALIASES: Mapping[str, str] = MappingProxyType({"list-rules": "rules"})
"""Alternative command names."""

HELP: Mapping[str, str] = MappingProxyType(
    {
        "load": "load SOURCE|@NAME -- read a diagram (Dirac, DSL or listing); any non-command "
        "line is load",
        "show": "show [derivation|start|scalable|certificate|@NAME|SOURCE] -- show a state",
        "rules": "rules [RULE] -- rule names, or one rule's quantifiers, side conditions, "
        "dimension guards and scalar (alias list-rules)",
        "match": "match [RULE] -- matches in the current diagram, all rules when none",
        "apply": "apply RULE [at INT | at nI, nJ] -- rewrite the current diagram at a match",
        "check": "check [NAME=VALUE ...] [against TARGET] -- oracle check of start vs current "
        "(or current vs TARGET) at the sample grid, supplied values overriding",
        "normalize": "normalize -- reduce the current diagram to normal form",
        "decide": "decide [against] [TARGET] -- decide equality of start vs current (or "
        "current vs TARGET)",
        "saturate": "saturate -- equality saturation, then extract the cheapest diagram",
        "prove": "prove [against] [TARGET] -- tactic search for a proof",
        "induct": "induct [on SYMBOL] [NAME=VALUE ...] [against TARGET] -- symbolic-n proof by "
        "induction",
        "certificate": "certificate [emit [derivation|induction|proof] | verify [NAME=VALUE "
        "...] | show | kernel | lean [FILE]] -- emit, verify, show, kernel-check or export a "
        "certificate to Lean 4",
        "translate": "translate [back | dissolve S | split S FIRST | join S S | enclose nA, nB "
        "[x MULT] | kill S | fuse nA.outI -- nB.inJ] -- scalable notation and sheet steps",
        "name": "name NAME -- store the current diagram as @NAME",
        "names": "names -- list named diagrams",
        "undo": "undo -- revert the last state-changing command",
        "reset": "reset -- restart the derivation at the current diagram",
        "detail": "detail -- the last command's full payload, listing and Dirac side by side",
        "help": "help [CMD] -- this text, or one command's usage",
        "dirac": "dirac -- Dirac notation",
        "zx": "zx -- ZX diagrams drawn with matplotlib (alias of zxmatplot)",
        "zxmatplot": "zxmatplot -- ZX diagrams drawn with matplotlib",
        "zxunicode": "zxunicode -- ZX diagrams drawn on a Unicode grid",
        "zxascii": "zxascii -- ZX diagrams drawn on an ASCII grid",
        "both": "both -- Dirac notation and ZX drawings",
        "raw": "raw -- full ZX graph listings, nothing summarised or capped",
        "listing": "listing -- toggle the ZX graph listing under each drawing",
        "verbose_on": "verbose_on -- echo each line's value, utop style",
        "verbose_off": "verbose_off -- stop echoing values",
        "config": "config [KEY VALUE | save | reset] -- image settings: size, color",
        "history": "history -- list the lines entered this session",
        "quit": "quit -- leave the shell",
        "exit": "exit -- leave the shell",
    }
)
"""Usage line per command, shell-only commands included."""


def command_names() -> tuple[str, ...]:
    """Every command name and alias, sorted."""
    return tuple(sorted({*COMMANDS, *ALIASES}))


# -- dispatch ------------------------------------------------------------------------------


def _split(line: str) -> tuple[str, str]:
    """(canonical command, arguments); a non-command line is ``load`` of the whole line."""
    text = line.strip()
    parts = text.split(None, 1)
    word, rest = parts[0], (parts[1] if len(parts) > 1 else "")
    name = ALIASES.get(word, word)
    if name not in COMMANDS or (rest.lstrip().startswith("=") and name != "apply"):
        return "load", text
    return name, rest.split("#", 1)[0].strip()


_STATE_CHANGING = frozenset({"load", "apply", "normalize", "saturate", "translate", "reset"})
_LABEL_WIDTH = 60


def _label(line: str) -> str:
    """``line`` on one line, cut to ``_LABEL_WIDTH`` characters with ``...``."""
    text = _one_line(line)
    return text if len(text) <= _LABEL_WIDTH else text[: _LABEL_WIDTH - 3] + "..."


def execute(session: Session, line: str) -> Outcome:
    """Run one command line on ``session``; raises only :class:`CommandError`. A failed
    command leaves the undoable state as it was; a changing load, apply, normalize or saturate
    drops the scalable form and its sheet steps."""
    if not isinstance(session, Session):
        raise CommandGrammarError(f"session must be a Session, got {type(session).__name__}")
    if not isinstance(line, str):
        raise CommandGrammarError(f"line must be a str, got {type(line).__name__}")
    if not line.strip():
        raise CommandGrammarError("empty command")
    name, args = _split(line)
    snap = session._snapshot(_label(line)) if name in _STATE_CHANGING else None
    try:
        outcome = COMMANDS[name](session, args)
    except CommandError:
        if snap is not None:
            session._restore(snap)
        raise
    except Exception as exc:
        if snap is not None:
            session._restore(snap)
        kind = CommandGrammarError if isinstance(exc, ParseGrammarError) else CommandDomainError
        raise kind(_one_line(exc) or type(exc).__name__) from exc
    except BaseException:
        if snap is not None:
            session._restore(snap)
        raise
    if snap is not None and outcome.changed:
        session.undo_stack.append(snap)
        if name in ("load", "apply", "normalize", "saturate"):
            session.scalable = None
            session.sheet_steps = []
    session.last_detail = outcome.detail
    return outcome


def run(session: Session, line: str) -> Outcome:
    """:func:`execute`, with every ``Exception`` turned into an ``ok=False`` outcome whose
    :class:`ErrorDetail` becomes ``session.last_detail``."""
    try:
        return execute(session, line)
    except Exception as exc:  # noqa: BLE001
        text = line if isinstance(line, str) else repr(line)
        name = _split(text)[0] if isinstance(line, str) and text.strip() else ""
        chain: list[BaseException] = []
        cursor: BaseException | None = exc
        while cursor is not None and cursor not in chain:
            chain.append(cursor)
            cursor = cursor.__cause__
        state = session.current if isinstance(session, Session) else None
        detail = ErrorDetail(
            text,
            name,
            tuple(type(e).__name__ for e in chain),
            tuple(_one_line(e) for e in chain),
            state.copy() if state is not None else None,
        )
        if isinstance(session, Session):
            session.last_detail = detail
        message = _one_line(exc) or type(exc).__name__
        blocks = (Text(f"error: {message}"), Text("(detail shows the engine detail)"))
        return Outcome(name, False, blocks, detail, False)


def format_outcome(
    outcome: Outcome,
    mode: NotationMode = DEFAULT_MODE,
    *,
    unicode: bool = False,
    zx_style: str = "matplot",
    listing: bool = False,
) -> str:
    """``outcome``'s blocks joined by newlines: Text as is, State via ``printer.render`` in
    ``mode``, Proof via :func:`proof_text`; ZX and BOTH draw each diagram in ``zx_style``, its
    graph listing under it when ``listing``."""
    if not isinstance(outcome, Outcome):
        raise CommandGrammarError(f"outcome must be an Outcome, got {type(outcome).__name__}")
    if not isinstance(mode, NotationMode):
        raise CommandGrammarError(f"mode must be a NotationMode, got {mode!r}")
    if zx_style not in zxdraw.STYLES:
        raise CommandGrammarError(f"unknown ZX drawing style {zx_style!r}")

    def draw(diagram: Diagram) -> str:
        text = zxdraw.draw(zxview.View(diagram), zx_style)
        return f"{text}\n{render_graph(diagram)}" if listing else text

    parts: list[str] = []
    with drawing(draw if mode in (NotationMode.ZX, NotationMode.BOTH) else None):
        for block in outcome.blocks:
            if isinstance(block, Text):
                parts.append(block.text)
                continue
            try:
                if isinstance(block, Proof):
                    parts.append(proof_text(block.proof, unicode=unicode))  # type: ignore[arg-type]
                    continue
                parts.append(render(block.state, mode, unicode=unicode))
            except Exception as exc:  # noqa: BLE001
                parts.append(f"render: unavailable ({type(exc).__name__}: {_one_line(exc)})")
    return "\n".join(parts)


# -- detail --------------------------------------------------------------------------------


def _indent(lines: Iterable[str], depth: int = 1) -> list[str]:
    pad = _INDENT * depth
    return [pad + line if line else line for line in lines]


def _failure_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {_one_line(exc)}"


def _state_lines(label: str, state: object, unicode: bool, note: str = "") -> list[str]:
    """``<label>:`` (then ``note``) and the state's ``render_detail`` block indented, or
    ``<label>: unavailable (<reason>)``."""
    try:
        body = render_detail(state, unicode=unicode)
    except Exception as exc:  # noqa: BLE001
        return [f"{label}: unavailable ({_failure_text(exc)})"]
    head = f"{label}: {note}" if note else f"{label}:"
    return [head, *_indent(body.splitlines())]


def _step_line(index: int | None, step: RewriteStep) -> str:
    consumed = ", ".join(f"n{int(n)}" for n in sorted(step.consumed_node_ids, key=int)) or "none"
    new = ", ".join(f"n{int(n)}" for n in sorted(step.new_node_ids, key=int)) or "none"
    label = "step" if index is None else f"step {index}"
    return (
        f"{label}: {step.rule_name} (consumed {consumed}, new {new}, "
        f"scalar {_expr(step.scalar_introduced)})"
    )


def _results_lines(initial: Diagram, results: Sequence[RewriteResult], u: bool) -> list[str]:
    lines = _state_lines("state 0", initial, u)
    for index, result in enumerate(results, 1):
        lines.append(_step_line(index, result.step))
        lines += _state_lines(f"state {index}", result.diagram, u)
    return lines


def _replayed_states(derivation: Derivation) -> tuple[list[Diagram | str], str | None]:
    """Each state from re-applying the recorded steps to ``initial`` (a replayed step must equal
    its record), a reason string for each state not reached, the last state replaced by the
    recorded ``final``; with the first failure, or None."""
    working = derivation.initial.copy()
    states: list[Diagram | str] = [working]
    failure: str | None = None
    for index, step in enumerate(derivation.steps, 1):
        if failure is not None:
            states.append(f"not reached: {failure}")
            continue
        prefix = f"step {index} ({step.rule_name}) failed"
        try:
            result = apply(working, lookup_rule(step.rule_name), step.match)
        except Exception as exc:  # noqa: BLE001
            failure = f"{prefix}: {_failure_text(exc)}"
        else:
            if result.step == step:
                working = result.diagram
                states.append(working)
                continue
            failure = f"{prefix}: the replayed step differs from the record"
        states.append(failure)
    if derivation.steps:
        states[-1] = derivation.final
    return states, failure


def _derivation_lines(derivation: Derivation, u: bool) -> list[str]:
    """Header, induction claim, every state recovered by re-applying the recorded steps, then
    the children, nested."""
    head = f"derivation: {derivation.kind.value}, {_count(len(derivation.steps), 'step')}"
    if derivation.label:
        head += f", label {derivation.label!r}"
    body: list[str] = []
    claim = derivation.induction
    if claim is not None:
        body.append(
            f"induction over {claim.symbol} from {claim.base_value}, step discharge "
            f"{claim.step_discharge or 'unstated'}, held symbolic "
            f"{', '.join(claim.held_symbolic) or 'none'}"
        )
    states, failure = _replayed_states(derivation)
    last = len(derivation.steps)
    for index, state in enumerate(states):
        if index:
            body.append(_step_line(index, derivation.steps[index - 1]))
        if isinstance(state, str):
            body.append(f"state {index}: unavailable ({state})")
            continue
        note = f"recorded final (replay {failure})" if index == last and failure else ""
        body += _state_lines(f"state {index}", state, u, note)
    if not derivation.steps:
        try:
            same = compare_structure(derivation.initial, derivation.final).identical
        except Exception:  # noqa: BLE001
            same = False
        if not same:
            body += _state_lines("final", derivation.final, u)
    two = derivation.induction is not None and len(derivation.children) == 2
    for position, child in enumerate(derivation.children):
        label = ("base", "step")[position] if two else f"child {position}"
        body += [f"{label}:", *_indent(_derivation_lines(child, u))]
    return [head, *_indent(body)]


def _proof_lines(proof: ProofCertificate | InductionProof | Certificate, u: bool) -> list[str]:
    if isinstance(proof, Certificate):
        return [
            f"certificate: {proof.check_method.value}",
            *_indent(_derivation_lines(proof.derivation, u)),
        ]
    if isinstance(proof, ProofCertificate):
        return [
            (
                f"proof: {_count(proof.length, 'step')} (forward {len(proof.forward.steps)}, "
                f"backward {len(proof.backward.steps)})"
            ),
            f"{_INDENT}moves: {', '.join(proof.moves) or 'none'}",
            f"{_INDENT}backward moves: {', '.join(proof.backward_moves) or 'none'}",
            *_indent(_state_lines("start", proof.start, u)),
            *_indent(_state_lines("goal", proof.goal, u)),
            f"{_INDENT}forward:",
            *_indent(_proof_lines(proof.forward, u), 2),
            f"{_INDENT}backward:",
            *_indent(_proof_lines(proof.backward, u), 2),
        ]
    region = ", ".join(f"n{int(n)}" for n in sorted(proof.region, key=int)) or "none"
    lines = [
        (
            f"induction proof: over {proof.index}, step symbol {proof.step_symbol}, side "
            f"{proof.side.value}, region {region}, {_count(proof.length, 'step')}"
        ),
        *_indent(_state_lines("start", proof.start, u)),
        *_indent(_state_lines("goal", proof.goal, u)),
    ]
    for name, piece in (("base", proof.base), ("expose", proof.expose), ("step", proof.step)):
        lines += [f"{_INDENT}{name}:", *_indent(_proof_lines(piece, u), 2)]
    return lines


def _comparison_line(label: str, comparison: ComparisonResult) -> str:
    text = (
        f"{label}: {'matched' if comparison.matched else 'mismatch'} "
        f"({comparison.mode.value}, max deviation {comparison.max_abs_deviation:g}): "
        f"{_one_line(comparison.reason)}"
    )
    if comparison.recovered_lambda is not None:
        text += f", lambda {comparison.recovered_lambda}"
    return text


def _assumption_lines(decision: Decision | ProofOutcome) -> list[str]:
    lines = [f"assumptions: {len(decision.assumptions) or 'none'}"]
    for c in decision.assumptions:
        source = c.source.kind.value
        if c.source.port_ref is not None:
            source += f" {c.source.port_ref}"
        if c.source.node_id is not None:
            source += f" n{c.source.node_id}"
        lines.append(f"{_INDENT}{c.assumed} = {c.equal_to} ({source}, {c.outcome.value})")
    return lines


def _nf_lines(label: str, nf: NormalForm, u: bool) -> list[str]:
    outcome = nf.outcome
    return [
        f"{label}: normal form, {_count(len(nf.results), 'step')}, {outcome.stop_reason.value}",
        (
            f"{_INDENT}steps attempted {outcome.steps_attempted}, scalar accumulated "
            f"{_expr(outcome.scalar_accumulated)}, key {nf.key}"
        ),
        *_indent(_results_lines(nf.source, nf.results, u)),
        *_indent(_state_lines("reduced", nf.diagram, u)),
    ]


def _decision_lines(decision: Decision, u: bool) -> list[str]:
    lines = [
        f"decision: {decision.verdict.value} by {decision.method.value}",
        f"{_INDENT}reason: {decision.reason}",
        f"{_INDENT}samples checked: {decision.samples_checked}",
        f"{_INDENT}parameters agree: {decision.parameters_agree}",
        *_indent(_assumption_lines(decision)),
    ]
    if decision.counterexample is not None:
        lines.append(f"{_INDENT}counterexample: {_assignment_text(decision.counterexample)}")
    if decision.witness is not None:
        w = decision.witness
        lines.append(
            f"{_INDENT}witness: {_assignment_text(w.assignment)} at index {w.index}: "
            f"left {w.left}, right {w.right}"
        )
    if decision.comparison is not None:
        lines.append(_INDENT + _comparison_line("comparison", decision.comparison))
    if decision.saturation is not None:
        lines += _indent(_report_lines(decision.saturation))
    for label, nf in (("left", decision.left_nf), ("right", decision.right_nf)):
        if nf is not None:
            lines += _indent(_nf_lines(label, nf, u))
    for index, certificate in enumerate(decision.certificates):
        lines += _indent([f"certificate {index}:", *_indent(_proof_lines(certificate, u))])
    if decision.induction is not None:
        if decision.induction_reversed:
            lines.append(f"{_INDENT}induction ran with the right side as its left side")
        lines += _indent(_induction_lines(decision.induction, u))
    return lines


def _report_lines(report: SaturationReport) -> list[str]:
    lines = [
        (
            f"saturation: {report.stop_reason.value}, {_count(report.iterations, 'iteration')}, "
            f"{report.enodes_added} e-node(s) added, {_count(report.merges, 'merge')}, "
            f"{_count(report.applications, 'application')}, {report.redundant} redundant, "
            f"{report.pruned} pruned, {len(report.failed)} failed"
        )
    ]
    for failed in report.failed:
        lines.append(
            f"{_INDENT}failed: {failed.rule_name} at e-node {failed.enode}: "
            f"{_one_line(failed.message)}"
        )
    return lines


def _induction_lines(result: InductionResult, u: bool) -> list[str]:
    ob = result.obligation
    lines = [
        f"induction: {result.verdict.value} over {result.index} from {result.base_value}",
        f"{_INDENT}reason: {result.reason}",
        f"{_INDENT}witness: {_assignment_text(result.witness)}",
        f"{_INDENT}step symbol: {ob.step_symbol}",
        f"{_INDENT}held symbolic: {', '.join(sorted(ob.held_symbolic)) or 'none'}",
    ]
    if result.counterexample is not None:
        lines.append(f"{_INDENT}counterexample: {result.index} = {result.counterexample}")
    if result.base is not None:
        lines.append(_INDENT + _comparison_line("base comparison", result.base))
    if result.comparison is not None:
        lines.append(_INDENT + _comparison_line("comparison", result.comparison))
    for tier in result.tiers:
        settled = "settled" if tier.settled else "declined"
        lines.append(f"{_INDENT}tier {tier.discharge.value}: {settled}: {_one_line(tier.reason)}")
        if tier.counterexample is not None:
            lines.append(f"{_INDENT * 2}counterexample: {result.index} = {tier.counterexample}")
        for index, comparison in enumerate(tier.comparisons):
            lines.append(_INDENT * 2 + _comparison_line(f"comparison {index}", comparison))
        for index, step in enumerate(tier.steps, 1):
            lines.append(_INDENT * 2 + _step_line(index, step))
    for label, diagram in (
        ("left at base", ob.left_at_base),
        ("right at base", ob.right_at_base),
        ("left at k", ob.left_at_k),
        ("right at k", ob.right_at_k),
        ("left at successor", ob.left_at_successor),
        ("right at successor", ob.right_at_successor),
    ):
        lines += _indent(_state_lines(label, diagram, u))
    if result.derivation is not None:
        lines += _indent(_derivation_lines(result.derivation, u))
    return lines


def _outcome_lines(outcome: ProofOutcome, u: bool) -> list[str]:
    lines = [
        f"proof outcome: {outcome.status.value}",
        f"{_INDENT}reason: {outcome.reason}",
        *_indent(_assumption_lines(outcome)),
    ]
    if outcome.search is not None:
        s = outcome.search
        lines.append(
            f"{_INDENT}search: {s.status.value}, depth {s.depth}, {s.states} state(s), "
            f"{s.expanded} expanded, {s.applications} application(s), {s.pruned} pruned"
        )
    if outcome.counterexample is not None:
        lines.append(f"{_INDENT}counterexample: {_assignment_text(outcome.counterexample)}")
    if outcome.check is not None:
        lines += _indent(_check_lines(outcome.check, u))
    for piece in (outcome.certificate, outcome.induction):
        if piece is not None:
            lines += _indent(_proof_lines(piece, u))
    return lines


def _replay_lines(label: str, replay: object, u: bool) -> list[str]:
    """``<label>: reproduced|not reproduced: <reason>``, each step record, the replayed
    diagram, then each child replay, nested."""
    from archytaszx.semantics.certificate import ReplayResult

    if not isinstance(replay, ReplayResult):
        return [f"{label}: {replay!r}"]
    done = "reproduced" if replay.reproduced else "not reproduced"
    lines = [f"{label}: {done}: {_one_line(replay.reason)}"]
    for step in replay.steps:
        done = "reproduced" if step.reproduced else "not reproduced"
        lines.append(
            f"{_INDENT}step {step.index + 1} {step.rule_name}: {done}: {_one_line(step.reason)}"
        )
    lines += _indent(_state_lines("replayed", replay.diagram, u))
    for index, child in enumerate(replay.children):
        lines += _indent(_replay_lines(f"child {index}", child, u))
    return lines


def _check_lines(check: ProofCheck, u: bool) -> list[str]:
    lines = [
        f"proof check: {'verified' if check.verified else 'failed'}",
        f"{_INDENT}reason: {check.reason}",
        f"{_INDENT}meet isomorphic: {check.meet_isomorphic}",
        f"{_INDENT}samples checked: {check.samples_checked}",
    ]
    if check.counterexample is not None:
        lines.append(f"{_INDENT}counterexample: {_assignment_text(check.counterexample)}")
    if check.comparison is not None:
        lines.append(_INDENT + _comparison_line("comparison", check.comparison))
    for label, replay in (
        ("forward replay", check.forward_replay),
        ("backward replay", check.backward_replay),
    ):
        if replay is not None:
            lines += _indent(_replay_lines(label, replay, u))
    return lines


def _verification_lines(report: VerificationReport, u: bool) -> list[str]:
    lines = [
        f"verification: {'verified' if report.verified else 'failed'}, {report.check_method.value}",
        f"{_INDENT}reason: {report.reason}",
        f"{_INDENT}assignment: {_assignment_text(report.assignment)}",
        f"{_INDENT}mode: {report.mode.value}",
    ]
    if report.comparison is not None:
        lines.append(_INDENT + _comparison_line("comparison", report.comparison))
    for index, comparison in enumerate(report.child_comparisons):
        lines.append(_INDENT + _comparison_line(f"child {index}", comparison))
    return lines + _indent(_replay_lines("replay", report.replay, u))


def _argument_text(value: object) -> str:
    """A sheet-step argument: ports and wires as in the graph listing, containers in
    parentheses, anything else via :func:`_expr`."""
    if isinstance(value, PortRef):
        side = "out" if value.direction is Direction.OUTPUT else "in"
        return f"n{value.node_id}.{side}{value.index}"
    if isinstance(value, Wire):
        return f"{_argument_text(value.a)} -- {_argument_text(value.b)}"
    if isinstance(value, (tuple, list, frozenset, set)):
        texts = [_argument_text(v) for v in value]
        if isinstance(value, (frozenset, set)):
            texts.sort()
        return "(" + ", ".join(texts) + ")"
    return _expr(value)


def _sheet_lines(step: SheetStep, u: bool) -> list[str]:
    arguments = ", ".join(f"{k}={_argument_text(v)}" for k, v in step.arguments)
    return [
        f"sheet step: {step.operation}({arguments})",
        *_indent(_state_lines("before", step.before, u)),
        *_indent(_state_lines("after", step.after, u)),
    ]


def _sheet_check_lines(check: SheetCheck, u: bool) -> list[str]:
    lines = [
        f"sheet check: {'ok' if check.ok else 'failed'}: {check.reason}",
        (
            f"{_INDENT}replayed {check.replayed}, valid {check.valid}, oracle ok "
            f"{check.oracle_ok}, {check.samples_evaluated} sample(s) evaluated"
        ),
    ]
    if check.counterexample is not None:
        lines.append(f"{_INDENT}counterexample: {_assignment_text(check.counterexample)}")
    return lines + _indent(_sheet_lines(check.step, u))


def _round_trip_lines(report: RoundTripReport, u: bool) -> list[str]:
    lines = [
        f"round trip: {'ok' if report.ok else 'failed'}",
        f"{_INDENT}reason: {report.reason}",
        (
            f"{_INDENT}identical {report.identical}, family equal {report.family_equal}, "
            f"stripped ok {report.stripped_ok}, oracle ok {report.oracle_ok}, "
            f"{report.samples_evaluated} sample(s) evaluated"
        ),
    ]
    if report.counterexample is not None:
        lines.append(f"{_INDENT}counterexample: {_assignment_text(report.counterexample)}")
    return [
        *lines,
        *_indent(_state_lines("scalable", report.scalable, u)),
        *_indent(_state_lines("back", report.back, u)),
    ]


def _extraction_lines(extraction: Extraction, report: SaturationReport, u: bool) -> list[str]:
    lines = [
        *_report_lines(report),
        (
            f"extraction: e-class {extraction.eclass}, e-node {extraction.enode}, root e-node "
            f"{extraction.root}, cost {extraction.cost}, "
            f"{_count(len(extraction.results), 'step')}"
        ),
    ]
    for index, result in enumerate(extraction.results, 1):
        lines.append(f"{_INDENT}{_step_line(index, result.step)}")
        lines += _indent(_state_lines(f"state {index}", result.diagram, u))
    return lines + _state_lines("extracted", extraction.diagram, u)


def _error_lines(error: ErrorDetail, u: bool) -> list[str]:
    lines = [f"error in {error.command or 'input'}: {error.line}"]
    lines += [
        f"{_INDENT}{kind}: {message}"
        for kind, message in zip(error.error_types, error.messages, strict=False)
    ]
    if error.state is not None:
        lines += _indent(_state_lines("current", error.state, u))
    return lines


def _detail_lines(payload: object, u: bool) -> list[str]:
    """The detail of one payload, as lines."""
    if isinstance(payload, (Diagram, ScalableDiagram)):
        return _state_lines("state", payload, u)
    if isinstance(payload, Text):
        return payload.text.splitlines()
    if isinstance(payload, State):
        return _detail_lines(payload.state, u)
    if isinstance(payload, RewriteResult):
        return [_step_line(None, payload.step), *_state_lines("result", payload.diagram, u)]
    if isinstance(payload, NormalForm):
        return _nf_lines("normalize", payload, u)
    if isinstance(payload, Derivation):
        return _derivation_lines(payload, u)
    if isinstance(payload, (Certificate, ProofCertificate, InductionProof)):
        return _proof_lines(payload, u)
    if isinstance(payload, Decision):
        return _decision_lines(payload, u)
    if isinstance(payload, ProofOutcome):
        return _outcome_lines(payload, u)
    if isinstance(payload, InductionResult):
        return _induction_lines(payload, u)
    if isinstance(payload, ProofCheck):
        return _check_lines(payload, u)
    if isinstance(payload, VerificationReport):
        return _verification_lines(payload, u)
    if isinstance(payload, SheetStep):
        return _sheet_lines(payload, u)
    if isinstance(payload, SheetCheck):
        return _sheet_check_lines(payload, u)
    if isinstance(payload, RoundTripReport):
        return _round_trip_lines(payload, u)
    if isinstance(payload, SaturationReport):
        return _report_lines(payload)
    if isinstance(payload, ComparisonResult):
        return [_comparison_line("comparison", payload)]
    if isinstance(payload, _Pair):
        return [
            f"{payload.title}:",
            *_indent(_state_lines("left", payload.left, u)),
            *_indent(_state_lines("right", payload.right, u)),
            *_indent(_detail_lines(payload.payload, u)),
        ]
    if isinstance(payload, ErrorDetail):
        return _error_lines(payload, u)
    if isinstance(payload, tuple):
        if (
            len(payload) == 2
            and isinstance(payload[0], Extraction)
            and isinstance(payload[1], SaturationReport)
        ):
            return _extraction_lines(payload[0], payload[1], u)
        return [line for item in payload for line in _detail_lines(item, u)]
    return [repr(payload)]


def detail_text(payload: object, *, unicode: bool = False) -> str:
    """The full payload with every diagram state as listing and Dirac side by side, each
    preceded by its header lines; ``detail: unavailable (<reason>)`` when it cannot be
    printed."""
    if not isinstance(unicode, bool):
        raise CommandGrammarError(f"unicode must be a bool, got {unicode!r}")
    try:
        return "\n".join(_detail_lines(payload, unicode))
    except Exception as exc:  # noqa: BLE001
        return f"detail: unavailable ({_failure_text(exc)})"
