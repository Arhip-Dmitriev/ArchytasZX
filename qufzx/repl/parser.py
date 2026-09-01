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

"""Input DSL parser: spiders, wires, symbolic phases, bang boxes, dimensions, and Dirac kets.

Phase 5 slice only. FULL_PLAN.md's Phase 5 completion condition names a specific chain:
"the full path Dirac to graph to fuse to graph runs and the oracle confirms exact
equality". Before this module that chain did not exist -- ``tests/test_phase5_oracle.py``
built its worked example by hand (``tests/helpers.py::build_ghz_with_copy``). This module
closes exactly that gap, and no more:

* :func:`parse_dirac_source` accepts one restricted form: a summed ket family
  ``sum_{k=0}^{D-1} |k,k,...>`` (or the ``|k>^{n}`` tensor-power shorthand), optionally
  followed by ``; copy`` to feed the state into a fixed two-output copy spider. That is
  exactly the shape the worked example needs: a state-prep spider with 0 inputs and ``n``
  outputs, optionally wired into a 1-input/2-output copy spider.
* ``D`` may be a concrete positive integer or a bare identifier (a symbolic
  :class:`~qufzx.algebra.dimension.Dim`); ``n`` must be concrete, since a symbolic leg
  count is a :class:`~qufzx.diagram.generators.LegPolicy` question this slice does not
  touch. The bound summation index is rejected in a dimension slot.
* The emitted diagram never builds a matrix or dense tensor -- the ket-sum is never
  evaluated numerically. It only allocates nodes, wires, and a boundary order.
* The tensor-power leg count is bounded by :data:`_MAX_KET_LEG_COUNT`, since it comes from
  user input and drives eager allocation.
* Every outbound call is wrapped so no foreign exception hierarchy escapes this module's
  boundary; see :class:`DiracError`.

What this module does not do, because it belongs to a later phase: a general spider/wire/
bang-box declaration syntax (Phase 18's DSL), bang boxes, families indexed by more than the
one implicit ``k``, or a printer (the diagram-to-Dirac direction, Phase 17). ``copy`` is a
single keyword standing in for the one copy spider the worked example needs, not a general
generator-declaration mechanism; Phase 18 is expected to replace it with real declaration
syntax, at which point this grammar is a strict subset of that one.
"""

from __future__ import annotations

import re

from qufzx.algebra.dimension import Dim, DimensionDomainError
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef


class DiracError(Exception):
    """Base of every error :func:`parse_dirac_source` can raise.

    The same containment discipline :mod:`qufzx.rewrite` keeps for
    :class:`~qufzx.rewrite.rule.RewriteError`: every exception reaching a caller is a
    :class:`DiracError`, never a foreign class from a package this module calls into. The
    question for a future reader touching this file is whether every outbound call still
    has its foreign hierarchy contained here. The current calls:

    * ``Dim.concrete`` -- not safe on its own; raises
      :class:`~qufzx.algebra.dimension.DimensionDomainError` for a non-positive integer.
      :func:`_parse_dim` range-checks the token and additionally wraps the call, so a future
      change to ``Dim.concrete``'s domain cannot reopen the leak.
    * ``Dim.symbol`` -- safe. Every name passed has matched ``_KET_SUM_RE``'s
      ``[A-Za-z_]\\w*``, exactly the identifier shape ``sympy.Symbol`` accepts, and is built
      with the ``positive=True, integer=True`` pair ``_check_dimension_domain`` requires.
    * ``Diagram.add_node`` -- safe; per its own docstring it never raises.
    * ``Diagram.add_wire`` -- safe as used here; it raises ``GraphDomainError`` only for
      ``a == b``, and every wire built here joins two distinct fresh ports on different nodes.
    * ``Diagram.set_boundary_outputs`` -- safe; it never raises.
    * ``int(token)`` -- safe as used here; every call site first confirms the token is a
      non-empty run of ASCII ``0-9`` (:data:`_ASCII_DIGITS_RE`, or the regex's own
      ``(?P<power>[0-9]+)`` group). Note that ``str.isdigit()`` is *not* this predicate: it
      is true for Unicode category ``No`` as well as ``Nd``, so ``'²'`` is ``isdigit()``,
      matches neither ``\\d+`` nor ``[0-9]+``, and makes ``int()`` raise ``ValueError``.
    """


class DiracGrammarError(DiracError):
    """The source text does not match this module's restricted Dirac grammar at all."""


class DiracDomainError(DiracError):
    """The source text parses, but names a value outside what this slice accepts (e.g. a
    zero leg count, a dimension of 0, the bound summation index used as a dimension symbol,
    or a leg count above :data:`_MAX_KET_LEG_COUNT`)."""


_SUMMATION_INDEX = "k"
"""The (only) bound summation variable this slice's grammar recognizes. A single module-level
constant so the regex (:data:`_KET_SUM_RE`) and the dimension-symbol exclusion in
:func:`_parse_dim` are derived from the same source and cannot drift apart (Task 2, round 20)."""

_ASCII_DIGITS = "[0-9]+"
"""The one decimal-literal shape this module's grammar admits, as a regex fragment.

A single module-level constant so :data:`_KET_SUM_RE`'s two numeric groups and
:func:`_parse_dim`'s own guard (:data:`_ASCII_DIGITS_RE`) are derived from the same source
and cannot drift apart -- the same discipline :data:`_SUMMATION_INDEX` already applies to the
bound index (Task 2, round 20).

Deliberately ``[0-9]+``, not ``\\d+`` and emphatically not ``str.isdigit()`` (round 24).
Python's ``\\d`` is Unicode-aware (category ``Nd``), so ``\\d+`` admits e.g. ``'\u0663'``
and ``int()`` happily returns 3 for it -- a non-ASCII decimal digit silently accepted as a
concrete dimension in a DSL whose every other token is ASCII. ``str.isdigit()`` is broader
still, additionally admitting category ``No`` (``'\u00b2'``, ``'\u2075'``), for which
``int()`` raises ``ValueError`` outright. See :class:`DiracError`'s docstring, ``int(token)``
bullet, for the full account of what that divergence cost."""

_ASCII_DIGITS_RE = re.compile(rf"^{_ASCII_DIGITS}$")
"""Whole-token form of :data:`_ASCII_DIGITS`, for :func:`_parse_dim`'s guard."""

_IDENTIFIER = r"[A-Za-z_]\w*"
"""The one bare-identifier shape this module's grammar admits, as a regex fragment.

Shared by :data:`_KET_SUM_RE`'s ``dim`` group and :func:`_parse_dim`'s own symbol branch
(via :data:`_IDENTIFIER_RE`), for the same single-source-of-truth reason as
:data:`_ASCII_DIGITS`. ``\\w`` is left Unicode-aware deliberately: ``sympy.Symbol`` accepts
such a name, :meth:`~qufzx.algebra.dimension.Dim.symbol` builds it without complaint, and
nothing downstream cares -- unlike the numeric branch, where Unicode-awareness was a real
divergence between the guard and the conversion it guards."""

_IDENTIFIER_RE = re.compile(rf"^{_IDENTIFIER}$")
"""Whole-token form of :data:`_IDENTIFIER`, for :func:`_parse_dim`'s guard."""

_MAX_KET_LEG_COUNT = 1024
"""Parser sanity bound on the ``^{n}`` tensor-power leg count, not a semantic limit -- nothing
about ZX-calculus or this project caps a spider's leg count, and a large or symbolic ``n`` is
Phase 7's bang boxes' answer, not this slice's. This bound exists only so a single malformed or
adversarial source string cannot force this parser to eagerly allocate an unbounded number of
ports before any later phase gets a chance to reject the diagram. Same role and shape as
``_MAX_FIXPOINT_PASSES`` in :mod:`qufzx.rewrite.match` (Task 3, round 20)."""

_KET_SUM_RE = re.compile(
    rf"^sum_\{{{_SUMMATION_INDEX}=0\}}\^\{{(?P<dim>{_IDENTIFIER}|{_ASCII_DIGITS})-1\}}\s*"
    rf"\|(?P<body>[^>]*)>"
    rf"(?:\^\{{(?:\\otimes|⊗)?\s*(?P<power>{_ASCII_DIGITS})\s*\}})?$"
)
"""Matches ``sum_{k=0}^{D-1} |BODY>`` with an optional ``^{n}`` (or ``^{\\otimes n}``/
``^{⊗n}``) tensor-power suffix on the ket. ``BODY`` is read by :func:`_leg_count_from_body`."""


def _leg_count_from_body(body: str, power: str | None) -> int:
    """How many output legs the ket family declares.

    ``body`` must, once commas and whitespace are stripped, consist only of repeated ``k``
    characters (``"k"``, ``"kk"``, ``"k,k"``, ``"k, k, k"``, ...) -- one leg per ``k``. A
    ``power`` suffix (``|k>^{n}``) is the tensor-power shorthand instead: it requires
    ``body`` to be the single index ``"k"`` and supplies the leg count directly.
    """
    stripped = body.replace(",", "").replace(" ", "")
    if not stripped or any(ch != "k" for ch in stripped):
        raise DiracGrammarError(
            f"unsupported ket body {body!r}: this slice only accepts repeated 'k' indices, "
            "e.g. '|k,k>' or '|kk>'"
        )
    if power is not None:
        if stripped != "k":
            raise DiracGrammarError(
                f"ket body {body!r} combined with a tensor-power suffix ^{{{power}}} is "
                f"ambiguous: use either '|k>^{{n}}' or '|k,k,...>', not both"
            )
        leg_count = int(power)
    else:
        leg_count = len(stripped)
    if leg_count < 1:
        raise DiracDomainError(
            f"ket family declares {leg_count} output legs; at least 1 is required"
        )
    if leg_count > _MAX_KET_LEG_COUNT:
        raise DiracDomainError(
            f"ket family declares {leg_count} output legs, above this parser's sanity bound "
            f"of {_MAX_KET_LEG_COUNT} (see _MAX_KET_LEG_COUNT's docstring)"
        )
    return leg_count


def _parse_dim(token: str) -> Dim:
    """A concrete positive integer, or a bare identifier naming a symbolic ``Dim``.

    "Concrete positive integer" means a non-empty run of ASCII ``0-9`` and nothing else
    (:data:`_ASCII_DIGITS_RE`, the same constant :data:`_KET_SUM_RE`'s own numeric groups are
    built from). Round 24: this guard used to be ``token.isdigit()``, which is a strictly
    broader predicate than the ``\\d+`` this function's contract is stated in terms of --
    see :data:`_ASCII_DIGITS` and :class:`DiracError`'s ``int(token)`` bullet for what that
    divergence cost and why the guard is now literally the predicate it is documented as.
    A token matching neither branch never reaches this function through
    :func:`parse_dirac_source` at all: :data:`_KET_SUM_RE` rejects it as a grammar error
    first.

    Raises :class:`DiracDomainError` for a concrete value below 1, and for the bound
    summation index (:data:`_SUMMATION_INDEX`) used as a dimension symbol -- the index is
    bound by the enclosing sum, so a dimension named after it is always a capture error: a
    later ``substitute`` on the returned ``Dim`` would bind that symbol independently of the
    sum it lexically came from, silently detaching the dimension from its own index.
    """
    if _ASCII_DIGITS_RE.match(token):
        value = int(token)
        if value < 1:
            raise DiracDomainError(
                f"dimension token {token!r} names {value}, but a concrete dimension must "
                "be >= 1"
            )
        try:
            return Dim.concrete(value)
        except DimensionDomainError as exc:
            raise DiracDomainError(
                f"dimension token {token!r} is outside Dim's domain: {exc}"
            ) from exc
    if token == _SUMMATION_INDEX:
        raise DiracDomainError(
            f"dimension token {token!r} is the bound summation index; a dimension symbol "
            "must be free, not the variable the enclosing sum binds"
        )
    # Round 24: this function is total on its own terms, not merely on the tokens
    # _KET_SUM_RE currently hands it. Without this branch a token matching neither shape
    # fell through to Dim.symbol() and became a symbol *named* after it -- trading the bare
    # ValueError this round removed for a silent wrong parse, which is strictly worse. The
    # same "unreachable from every current caller is not the same claim as correct"
    # principle _assign_labels' own fix is recorded under (see qufzx.rewrite.match's module
    # docstring, round 23 summary).
    if not _IDENTIFIER_RE.match(token):
        raise DiracGrammarError(
            f"dimension token {token!r} is neither a decimal literal ({_ASCII_DIGITS}) nor "
            f"a bare identifier ({_IDENTIFIER})"
        )
    return Dim.symbol(token)


def _parse_ket_sum(text: str) -> tuple[Dim, int]:
    """Parse ``sum_{k=0}^{D-1} |...>`` into ``(dim, leg_count)``."""
    match = _KET_SUM_RE.match(text.strip())
    if match is None:
        raise DiracGrammarError(
            f"{text!r} does not match this slice's grammar: expected "
            "'sum_{k=0}^{D-1} |k,k,...>' (or the '|k>^{n}' shorthand)"
        )
    dim = _parse_dim(match.group("dim"))
    leg_count = _leg_count_from_body(match.group("body"), match.group("power"))
    return dim, leg_count


def parse_dirac_source(source: str) -> Diagram:
    """Parse a restricted Dirac-ket source string into a :class:`~qufzx.diagram.graph.Diagram`.

    See the module docstring for exactly what grammar this accepts. Two forms:

    * ``"sum_{k=0}^{D-1} |k,k,...>"`` alone: one state-prep spider, 0 inputs, one output
      per ket index, boundary outputs in ket order. This is ``A`` in
      :func:`tests.helpers.build_ghz_with_copy`.
    * ``"sum_{k=0}^{D-1} |k,k,...>; copy"``: the same state, with its first output wired
      into a fixed 1-input/2-output copy spider (also over ``D``); the boundary is the
      state's own remaining outputs (in order) followed by the copy spider's two outputs.
      This is the full "A into B" shape ``build_ghz_with_copy`` builds by hand, structurally
      identical port-for-port (see ``tests/test_phase5_dirac_oracle.py``).

    Raises :class:`DiracGrammarError` for source text outside this slice's restricted
    grammar, and :class:`DiracDomainError` for grammatically valid source naming a value
    (e.g. a zero leg count) this slice does not accept.
    """
    terms = [term.strip() for term in source.split(";")]
    if len(terms) not in (1, 2) or any(not term for term in terms):
        raise DiracGrammarError(
            f"{source!r}: expected a single ket-sum term, optionally followed by one "
            "';' and the keyword 'copy'"
        )
    dim, leg_count = _parse_ket_sum(terms[0])

    diagram = Diagram()
    state_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim] * leg_count)

    if len(terms) == 1:
        diagram.set_boundary_outputs(
            [PortRef(state_id, Direction.OUTPUT, i) for i in range(leg_count)]
        )
        return diagram

    if terms[1] != "copy":
        raise DiracGrammarError(
            f"{terms[1]!r}: this slice recognizes only the 'copy' keyword after ';' "
            "(a general spider-declaration syntax is Phase 18's, not this slice's)"
        )
    copy_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    diagram.add_wire(
        PortRef(state_id, Direction.OUTPUT, 0), PortRef(copy_id, Direction.INPUT, 0)
    )
    boundary_outputs = [
        PortRef(state_id, Direction.OUTPUT, i) for i in range(1, leg_count)
    ] + [PortRef(copy_id, Direction.OUTPUT, 0), PortRef(copy_id, Direction.OUTPUT, 1)]
    diagram.set_boundary_outputs(boundary_outputs)
    return diagram
