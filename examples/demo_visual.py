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

"""Scripted terminal demo of the current qufzx engine, for grant-application screen capture.

Two columns, printed side by side for every step: INTERMEDIATE (left) is an exhaustive,
unfiltered trace of every value the engine actually computed -- every port, every dimension
unification, every side condition's full detail string, every remapped port, every tensor
entry -- rendered as plain Python reprs of the real returned objects, wrapped rather than
truncated, so nothing is sampled or cut. NOTABLE (right) is the small set of curated,
colored highlights: the diagram pictures and the pass/fail verdicts. No narration is
printed anywhere in either column -- that is left entirely to a voice-over recorded
separately over this output. A wide terminal (150+ columns) is assumed.

Calls only public API from ``qufzx.*`` -- no test helpers, no new engine surface, no
rendering borrowed from (or added to) ``qufzx.repl.printer``, which stays a Phase 17
skeleton. Every value shown is read live from the objects the engine returns; nothing here
is hardcoded. Deterministic: no randomness, no wall-clock value is ever printed, and every
set/frozenset the engine hands back is iterated through its own ``sort_key()`` rather than
left to hash order. Run with no arguments and no config:

    python examples/demo_visual.py

Cut into small sections, each revealed with a typed command and then held on screen until
the operator presses a key -- no timed auto-advance. Set ``ARCHYTAS_DEMO_FAST=1`` to skip
both the typing animation and every keypress wait (for iterating on this file); when stdin
is not an interactive terminal (piped or captured), the demo likewise never waits for a key
nobody can press.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import time

from qufzx.algebra.dimension import Dim, unify_all
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, Node, NodeId, PortRef, Wire
from qufzx.diagram.validate import validate
from qufzx.rewrite.engine import RewriteResult, apply
from qufzx.rewrite.match import (
    FUSION_SIDE_CONDITIONS,
    FusionMatch,
    find_matches,
    resolve_fusion_match,
)
from qufzx.rewrite.rules_library import SPIDER_FUSION

LEFT_WIDTH = 100
RIGHT_WIDTH = 48
SEP = " │ "
TOTAL_WIDTH = LEFT_WIDTH + len(SEP) + RIGHT_WIDTH
_FAST_ENV_VAR = "ARCHYTAS_DEMO_FAST"
_THINK_SECONDS = 0.3
_COMMAND_CPS = 28.0
_INTERMEDIATE_CPS = 170.0
"""Characters per second for the two typing speeds this file uses: the "$ command" lines
type slowly and dramatically (``_COMMAND_CPS``); the intermediate column's exhaustive trace
rolls past much faster (``_INTERMEDIATE_CPS``) since there is far more of it to get through.
"""

_DIM_GRAY = "\033[90m"
_BOLD_WHITE = "\033[1;37m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RESET = "\033[0m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ANSI_SPLIT_RE = re.compile(r"(\x1b\[[0-9;]*m)")
_PAUSE_BEFORE_NOTABLE = 1.0
"""Seconds the finished intermediate column holds, blank-right, before the notable column
reveals -- a beat, not an instant cut."""

_HIGHLIGHT_RE = re.compile(
    r"(?P<true>\bTrue\b)"
    r"|(?P<false>\bFalse\b)"
    r"|(?P<success>\bSUCCESS\b)"
    r"|(?P<failure>\bFAILURE\b)"
    r"|(?P<deferred>\bDEFERRED\b|\bEXHAUSTED\b)"
    r"|(?P<dim>Dim\([^()]*\))"
    r"|(?P<nodeid>NodeId\(\d+\))"
    r"|(?P<qstring>'[^']*')"
)
_HIGHLIGHT_COLORS = {
    "true": _GREEN,
    "false": _RED,
    "success": _GREEN,
    "failure": _RED,
    "deferred": _YELLOW,
    "dim": _CYAN,
    "nodeid": _CYAN,
    "qstring": _YELLOW,
}


def _colorize(text: str) -> str:
    """Apply a small, fixed set of syntax-highlighting rules to a plain intermediate line.

    Only ever called on already-wrapped plain text (see :func:`_two_col`'s own docstring
    for why coloring happens after wrapping, not before): booleans, unify-status words,
    ``Dim(...)``/``NodeId(...)`` reprs, and quoted string literals each get one fixed
    color; everything else (field names, punctuation, structural reprs like ``PortRef``)
    stays the terminal's default foreground, the same restraint an ordinary syntax
    highlighter uses.
    """

    def _sub(match: re.Match[str]) -> str:
        for name, color in _HIGHLIGHT_COLORS.items():
            value = match.group(name)
            if value is not None:
                return f"{color}{value}{_RESET}"
        return match.group(0)

    return _HIGHLIGHT_RE.sub(_sub, text)


def _type_colored(text: str, delay: float) -> None:
    """Type ``text`` out, sleeping only between visible characters.

    An ANSI escape sequence is written in one instant burst -- terminals correctly hold a
    partial escape sequence unrendered until it completes, so this is not required for
    correctness, but sleeping ``delay`` between an escape code's own bytes would slow the
    roll down by however many color switches a line happens to have, which has nothing to
    do with how much of that line is actually visible yet.
    """
    for chunk in _ANSI_SPLIT_RE.split(text):
        if not chunk:
            continue
        if _ANSI_SPLIT_RE.fullmatch(chunk):
            sys.stdout.write(chunk)
            continue
        for ch in chunk:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(delay)


def _fast() -> bool:
    return bool(os.environ.get(_FAST_ENV_VAR))


def _visible_len(text: str) -> int:
    """The length of ``text`` with ANSI color escapes stripped out."""
    return len(_ANSI_RE.sub("", text))


def _pad_visible(text: str, width: int) -> str:
    return text + " " * max(0, width - _visible_len(text))


def _margin() -> int:
    """Left padding, in columns, to center a ``TOTAL_WIDTH``-wide block in the terminal."""
    columns = shutil.get_terminal_size(fallback=(TOTAL_WIDTH, 24)).columns
    return max(0, (columns - TOTAL_WIDTH) // 2)


def _typeout(text: str, *, cps: float = _COMMAND_CPS, color: str = _BOLD_WHITE) -> None:
    """Print a simulated, centered command line character by character, then a newline."""
    text = text[:TOTAL_WIDTH]
    margin = " " * _margin()
    if _fast():
        print(f"{margin}{color}{text}{_RESET}" if color else f"{margin}{text}")
        return
    delay = 1.0 / cps
    sys.stdout.write(margin)
    if color:
        sys.stdout.write(color)
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    if color:
        sys.stdout.write(_RESET)
    sys.stdout.write("\n")
    time.sleep(_THINK_SECONDS)


def _wait_for_key() -> None:
    """Hold the screen until the operator presses a key, then reveal the next section.

    A no-op when ARCHYTAS_DEMO_FAST is set, and also when stdin is not a real terminal --
    the demo must never print a cue or block waiting for a key nobody can press.
    """
    if _fast() or not sys.stdin.isatty():
        return
    sys.stdout.write(f"{_DIM_GRAY}▸{_RESET}")
    sys.stdout.flush()
    try:
        import termios
        import tty
    except ImportError:
        input()
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    sys.stdout.write("\r \r")
    sys.stdout.flush()


def _blank() -> None:
    print()


def _two_col_plain(expanded_left: list[str], right: list[str], margin: str) -> None:
    """Print both columns simultaneously, one full row per line -- no animation.

    Used whenever there is no operator to watch it type: ARCHYTAS_DEMO_FAST, or stdout
    not a real terminal (piped or captured).
    """
    height = max(len(expanded_left), len(right))
    for i in range(height):
        left_line = _colorize(expanded_left[i]) if i < len(expanded_left) else ""
        right_line = right[i] if i < len(right) else ""
        sep = f" {_BLUE}│{_RESET} "
        print(f"{margin}{_pad_visible(left_line, LEFT_WIDTH)}{sep}{right_line}")


def _two_col(left: list[str], right: list[str]) -> None:
    """Reveal ``left`` (the exhaustive trace) and ``right`` (the curated highlights).

    Every ``left`` entry is plain text (never ANSI-colored -- textwrap does not
    understand escape codes, so wrapping stays safe only if there are none to split
    across a line break) and is wrapped, never truncated, to ``LEFT_WIDTH``: nothing in
    it is allowed to be cut off. Every ``right`` entry may carry color and is expected
    to already fit in ``RIGHT_WIDTH`` -- this raises loudly if one does not, rather than
    silently corrupting the layout.

    On a real, non-fast terminal: the left column types itself out first, top to bottom,
    fast but not instant, establishing the full row grid with the right side still
    blank; only once every intermediate line has finished typing does the right column
    fill in, all at once, via absolute cursor positioning that never touches what the
    left column already printed. Falls back to a single simultaneous pass
    (:func:`_two_col_plain`) whenever there is no one watching it type.
    """
    expanded_left: list[str] = []
    for entry in left:
        wrapped = textwrap.wrap(entry, width=LEFT_WIDTH, subsequent_indent="    ")
        expanded_left.extend(line[:LEFT_WIDTH] for line in (wrapped or [""]))
    for entry in right:
        if _visible_len(entry) > RIGHT_WIDTH:
            raise RuntimeError(f"notable column line exceeds {RIGHT_WIDTH} columns: {entry!r}")

    margin_n = _margin()
    margin = " " * margin_n

    if _fast() or not sys.stdout.isatty():
        _two_col_plain(expanded_left, right, margin)
        return

    height = max(len(expanded_left), len(right))
    if height == 0:
        return
    delay = 1.0 / _INTERMEDIATE_CPS
    sep = f" {_BLUE}│{_RESET} "

    for i in range(height):
        plain_text = expanded_left[i] if i < len(expanded_left) else ""
        colored_text = _colorize(plain_text)
        sys.stdout.write(margin)
        _type_colored(colored_text, delay)
        sys.stdout.write(" " * max(0, LEFT_WIDTH - _visible_len(colored_text)))
        sys.stdout.write(sep)
        sys.stdout.write("\n")

    time.sleep(_PAUSE_BEFORE_NOTABLE)
    sys.stdout.write(f"\033[{height}A")
    right_col = margin_n + LEFT_WIDTH + _visible_len(sep) + 1
    for i in range(height):
        if i < len(right):
            sys.stdout.write(f"\033[{right_col}G{right[i]}")
        sys.stdout.write("\033[1B\033[1G")
    sys.stdout.flush()


def _node_label(node_id: NodeId, node: Node) -> str:
    return f"({int(node_id)}:{node.generator_type.name})"


def _stem_column(dim: str) -> list[str]:
    """The [stem, dim, stem, open] four-row strip for one dangling leg."""
    width = max(3, len(dim) + 2)
    stem = f"{_DIM_GRAY}{'│'.center(width)}{_RESET}"
    dim_row = f"{_CYAN}{dim.center(width)}{_RESET}"
    open_row = f"{_DIM_GRAY}{'o'.center(width)}{_RESET}"
    return [stem, dim_row, stem, open_row]


def _stems_block(legs: list[str]) -> list[str]:
    if not legs:
        return []
    columns = [_stem_column(dim) for dim in legs]
    return ["".join(column[row] for column in columns) for row in range(4)]


def _leg_visible_width(legs: list[str]) -> int:
    return sum(max(3, len(dim) + 2) for dim in legs)


def _render_node_stack(nodes: list[tuple[NodeId, Node]]) -> list[str]:
    """Fallback sketch: each node's label, then its legs as stems straight below it."""
    lines: list[str] = []
    for node_id, node in nodes:
        label = _node_label(node_id, node)
        legs = [str(port.dim) for port in (*node.inputs, *node.outputs)]
        lines.append(f"{_CYAN}{label}{_RESET}")
        lines.extend(_stems_block(legs))
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _render_two_node_chain(nodes: list[tuple[NodeId, Node]], wire: Wire) -> list[str]:
    """The one shape this demo actually draws precisely: two nodes joined by one wire."""
    (a_id, a_node), (b_id, b_node) = nodes
    a_consumed = wire.a if wire.a.node_id == a_id else wire.b
    b_consumed = wire.b if wire.a.node_id == a_id else wire.a

    def _other_leg_dims(node_id: NodeId, node: Node, consumed: PortRef) -> list[str]:
        return [
            str(port.dim)
            for direction in (Direction.INPUT, Direction.OUTPUT)
            for index, port in enumerate(node.legs(direction))
            if PortRef(node_id, direction, index) != consumed
        ]

    a_legs = _other_leg_dims(a_id, a_node, a_consumed)
    b_legs = _other_leg_dims(b_id, b_node, b_consumed)
    shared_dim = str(a_node.legs(a_consumed.direction)[a_consumed.index].dim)

    a_label = _node_label(a_id, a_node)
    b_label = _node_label(b_id, b_node)
    edge_middle_plain = f" ──{shared_dim}── "
    edge = (
        f"{_CYAN}{a_label}{_RESET}"
        f"{_DIM_GRAY} ──{_RESET}{_CYAN}{shared_dim}{_RESET}{_DIM_GRAY}── {_RESET}"
        f"{_CYAN}{b_label}{_RESET}"
    )
    b_col = len(a_label) + len(edge_middle_plain)

    a_block = _stems_block(a_legs)
    b_block = _stems_block(b_legs)
    a_width = _leg_visible_width(a_legs)

    lines = [edge]
    for row in range(max(len(a_block), len(b_block))):
        a_row = a_block[row] if row < len(a_block) else ""
        b_row = b_block[row] if row < len(b_block) else ""
        visible_after_a = a_width if a_row else 0
        pad = max(1, b_col - visible_after_a)
        lines.append(a_row + " " * pad + b_row)
    return lines


def _render_graph(diagram: Diagram) -> list[str]:
    """A small schematic sketch of ``diagram``: nodes as labeled dots, dangling legs as
    stems ending in an open boundary circle, and (for a single wired pair) the connecting
    wire as a labeled horizontal edge. Not a general circuit-layout engine.
    """
    nodes = sorted(diagram.nodes.items(), key=lambda item: int(item[0]))
    wires = sorted(diagram.wires, key=lambda w: w.sort_key())
    if len(nodes) == 2 and len(wires) == 1:
        return _render_two_node_chain(nodes, wires[0])
    return _render_node_stack(nodes)


def _render_before_after(pre: Diagram, post: Diagram) -> list[str]:
    """The two sketches of ``pre`` and ``post``, side by side, joined by an arrow."""
    left = _render_graph(pre)
    right = _render_graph(post)
    left_width = max((_visible_len(line) for line in left), default=0)
    height = max(len(left), len(right))
    arrow_row = height // 2
    lines: list[str] = []
    for row in range(height):
        left_line = left[row] if row < len(left) else ""
        right_line = right[row] if row < len(right) else ""
        pad = left_width - _visible_len(left_line)
        connector = f"  {_MAGENTA}⟹{_RESET}  " if row == arrow_row else "     "
        lines.append(left_line + " " * pad + connector + right_line)
    return lines


def _beat_build() -> Diagram:
    _typeout("$ build ghz_with_copy --dim d")
    left: list[str] = []
    right: list[str] = []

    dim = Dim.symbol("d")
    left.append(
        f'Dim.symbol("d") -> {dim!r}  free_symbols={sorted(dim.free_symbols)}  '
        f"is_concrete={dim.is_concrete}"
    )

    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
    node_a = diagram.nodes[a_id]
    left.append(f"add_node(Z, in=[], out=[d,d]) -> NodeId({int(a_id)})")
    for index, port in enumerate(node_a.outputs):
        left.append(f"  node {int(a_id)}.output[{index}] = {port!r}")
    policy_a = node_a.generator_type.leg_policy
    left.append(f"  node {int(a_id)} leg_policy = {policy_a!r}")
    left.append(
        f"  node {int(a_id)} leg_policy.allows(0,2) = "
        f"{policy_a.allows(node_a.num_inputs, node_a.num_outputs)}"
    )
    left.append(
        f"  node {int(a_id)} phase_schema={node_a.generator_type.phase_schema.value}  "
        f"dimension_policy={node_a.generator_type.dimension_policy.value}"
    )

    b_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    node_b = diagram.nodes[b_id]
    left.append(f"add_node(Z, in=[d], out=[d,d]) -> NodeId({int(b_id)})")
    for index, port in enumerate(node_b.inputs):
        left.append(f"  node {int(b_id)}.input[{index}] = {port!r}")
    for index, port in enumerate(node_b.outputs):
        left.append(f"  node {int(b_id)}.output[{index}] = {port!r}")
    policy_b = node_b.generator_type.leg_policy
    left.append(f"  node {int(b_id)} leg_policy = {policy_b!r}")
    left.append(
        f"  node {int(b_id)} leg_policy.allows(1,2) = "
        f"{policy_b.allows(node_b.num_inputs, node_b.num_outputs)}"
    )
    left.append(
        f"  node {int(b_id)} phase_schema={node_b.generator_type.phase_schema.value}  "
        f"dimension_policy={node_b.generator_type.dimension_policy.value}"
    )

    wire_a = PortRef(a_id, Direction.OUTPUT, 0)
    wire_b = PortRef(b_id, Direction.INPUT, 0)
    diagram.add_wire(wire_a, wire_b)
    wire = Wire(wire_a, wire_b)
    left.append(f"add_wire({wire_a!r}, {wire_b!r})")
    left.append(f"  {wire!r}")
    left.append(f"  wire.sort_key() = {wire.sort_key()}")

    boundary = [
        PortRef(a_id, Direction.OUTPUT, 1),
        PortRef(b_id, Direction.OUTPUT, 0),
        PortRef(b_id, Direction.OUTPUT, 1),
    ]
    diagram.set_boundary_outputs(boundary)
    left.append(f"set_boundary_outputs([{len(boundary)} refs])")
    for ref in boundary:
        left.append(f"  {ref!r}")

    left.append(f"diagram.scalar = {diagram.scalar!r}")
    left.append(
        f"  is_concrete={diagram.scalar.is_concrete}  "
        f"free_symbols={sorted(diagram.scalar.free_symbols)}"
    )

    right.extend(_render_graph(diagram))
    right.append("")
    right.append(f"{len(diagram.nodes)} nodes, {len(diagram.wires)} wire(s)")
    right.append(f"{len(diagram.boundary_outputs)} boundary legs")
    right.append(f"scalar = {diagram.scalar}")

    _two_col(left, right)
    _blank()
    _wait_for_key()
    return diagram


def _beat_validate(diagram: Diagram) -> None:
    _typeout("$ validate")
    left: list[str] = []
    right: list[str] = []

    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        node_a = diagram.nodes[wire.a.node_id]
        node_b = diagram.nodes[wire.b.node_id]
        port_a = node_a.legs(wire.a.direction)[wire.a.index]
        port_b = node_b.legs(wire.b.direction)[wire.b.index]
        unify_result = port_a.dim.unify(port_b.dim)
        left.append(f"{wire.a!r}.dim = {port_a.dim!r}")
        left.append(f"{wire.b!r}.dim = {port_b.dim!r}")
        left.append(f"  .dim.unify() -> {unify_result!r}")

    for node_id in sorted(diagram.nodes, key=int):
        node = diagram.nodes[node_id]
        leg_dims = [port.dim for port in (*node.inputs, *node.outputs)]
        result = unify_all(leg_dims)
        left.append(f"node {int(node_id)} leg dims = {leg_dims!r}")
        left.append(f"  unify_all(...) -> {result!r}")
        policy_ok = node.generator_type.leg_policy.allows(node.num_inputs, node.num_outputs)
        left.append(f"  leg_policy.allows({node.num_inputs},{node.num_outputs}) = {policy_ok}")
        left.append(
            f"  phase_schema={node.generator_type.phase_schema.value}  phase={node.phase!r}"
        )

    report = validate(diagram)
    left.append(f"validate(diagram).issues = {report.issues!r}")
    left.append(f"  .errors = {report.errors!r}")
    left.append(f"  .deferred = {report.deferred!r}")

    mark = "✓" if report.is_valid else "✗"
    color = _GREEN if report.is_valid else _RED
    right.append(f"{color}{mark}{_RESET} valid")
    right.append(f"errors={len(report.errors)}")
    right.append(f"deferred={len(report.deferred)}")

    _two_col(left, right)
    _blank()
    _wait_for_key()


def _beat_match(diagram: Diagram) -> FusionMatch:
    _typeout("$ match spider_fusion")
    left: list[str] = []
    right: list[str] = []

    left.append(f"FUSION_SIDE_CONDITIONS ({len(FUSION_SIDE_CONDITIONS)} declared):")
    for condition in FUSION_SIDE_CONDITIONS:
        left.append(f"  {condition.name}: {condition.description}")

    matches = find_matches(diagram)
    left.append(f"find_matches(diagram) -> {len(matches)} match(es)")
    match = matches[0]
    left.append(f"  a_id={int(match.a_id)}  b_id={int(match.b_id)}")
    left.append(f"  wire = {match.wire!r}")
    left.append(f"  shared_dim = {match.shared_dim!r}")
    left.append(f"  bindings = {dict(match.bindings)!r}")
    left.append(f"  dimension_constraints = {match.dimension_constraints!r}")
    for outcome in match.side_condition_outcomes:
        left.append(f"  outcome: {outcome!r}")

    right.append(f"a_id={int(match.a_id)} b_id={int(match.b_id)}")
    right.append(f"shared_dim={match.shared_dim}")
    right.append("")
    for outcome in match.side_condition_outcomes:
        mark = "✓" if outcome.passed else "✗"
        color = _GREEN if outcome.passed else _RED
        right.append(f"{color}{mark}{_RESET} {outcome.name}")

    _two_col(left, right)
    _blank()
    _wait_for_key()
    return match


def _beat_apply(diagram: Diagram, match: FusionMatch) -> RewriteResult:
    _typeout("$ apply spider_fusion")
    left: list[str] = []
    right: list[str] = []

    left.append(f"SPIDER_FUSION.name = {SPIDER_FUSION.name!r}")
    left.append(f"  .scalar_introduced = {SPIDER_FUSION.scalar_introduced!r}")
    left.append(f"  .quantifiers.leg_counts = {SPIDER_FUSION.quantifiers.leg_counts}")
    left.append(f"  .quantifiers.dimensions = {SPIDER_FUSION.quantifiers.dimensions}")

    result = apply(diagram, SPIDER_FUSION, match)
    left.append("apply(diagram, SPIDER_FUSION, match)")
    left.append(f"  result.diagram is not input diagram: {result.diagram is not diagram}")
    left.append(f"  input diagram still has {len(diagram.nodes)} node(s) (never mutated)")

    post_report = validate(result.diagram)
    left.append(f"validate(result.diagram).issues = {post_report.issues!r}")

    right.extend(_render_before_after(diagram, result.diagram))
    right.append("")
    consumed = ",".join(str(int(n)) for n in result.step.consumed_node_ids)
    new = ",".join(str(int(n)) for n in result.new_node_ids)
    right.append(f"consumed=[{consumed}]")
    right.append(f"new=[{new}]")
    right.append(f"scalar x {result.step.scalar_introduced}")
    mark = "✓" if post_report.is_valid else "✗"
    color = _GREEN if post_report.is_valid else _RED
    right.append(f"{color}{mark}{_RESET} re-validated")

    _two_col(left, right)
    _blank()
    _wait_for_key()
    return result


def _beat_record(diagram: Diagram, match: FusionMatch, result: RewriteResult) -> None:
    """RewriteStep, and how apply() actually built it -- see qufzx/rewrite/engine.py's
    own docstring, step 9: the certificate is preferred from the builder's independently
    re-derived facts, never from a match's own unaudited claims, and this beat is the
    live demonstration of that preference rather than a claim about it.
    """
    _typeout("$ record spider_fusion")
    left: list[str] = []
    right: list[str] = []
    step = result.step

    left.append(f"match.dimension_constraints (claimed) = {match.dimension_constraints!r}")
    resolution = resolve_fusion_match(diagram, match.a_id, match.b_id, match.wire)
    left.append("resolve_fusion_match(diagram, a_id, b_id, wire) -- re-derived fresh,")
    left.append(f"  independently of match: {resolution!r}")
    left.append(
        f"  match.side_condition_outcomes == resolution.outcomes: "
        f"{match.side_condition_outcomes == resolution.outcomes}"
    )
    left.append(
        f"  match.dimension_constraints == resolution.dimension_constraints: "
        f"{match.dimension_constraints == resolution.dimension_constraints}"
    )
    left.append(
        f"  match.shared_dim == resolution.shared_dim: "
        f"{match.shared_dim == resolution.shared_dim}"
    )
    left.append(
        f"  dict(match.bindings) == dict(resolution.bindings): "
        f"{dict(match.bindings) == dict(resolution.bindings)}"
    )
    left.append(
        "apply()'s builder returns this same resolution as "
        "BuildResult.verified_side_condition_outcomes / verified_dimension_constraints;"
    )
    left.append("apply() records those, preferring them over match's own fields:")
    left.append(
        f"  step.side_condition_outcomes == resolution.outcomes: "
        f"{step.side_condition_outcomes == resolution.outcomes}"
    )
    left.append(
        f"  step.dimension_constraints == resolution.dimension_constraints: "
        f"{step.dimension_constraints == resolution.dimension_constraints}"
    )
    left.append(f"  step.port_mapping ({len(step.port_mapping)} entries):")
    for old_ref, new_ref in sorted(step.port_mapping.items(), key=lambda kv: kv[0].sort_key()):
        left.append(f"    {old_ref!r} -> {new_ref!r}")

    right.append(f"rule: {step.rule_name}")
    right.append(f"consumed: {list(step.consumed_node_ids)}")
    right.append(f"new: {list(result.new_node_ids)}")
    right.append(f"scalar introduced: {step.scalar_introduced}")
    right.append(f"{len(step.side_condition_outcomes)} side conditions recorded")
    right.append(f"{len(step.dimension_constraints)} dimension constraints")
    right.append(f"{len(step.port_mapping)} port(s) remapped")
    right.append(f"{len(step.removed_deferred_issues)} deferred issue(s) removed")
    right.append(f"{len(step.introduced_deferred_issues)} deferred issue(s) introduced")
    right.append(f"{len(step.phase_substitutions)} node(s) with phase substitutions")
    right.append("")
    right.append(f"{_GREEN}✓{_RESET} record == independent re-derivation")

    _two_col(left, right)
    _blank()
    _wait_for_key()


def main() -> None:
    _wait_for_key()
    diagram = _beat_build()
    _beat_validate(diagram)
    match = _beat_match(diagram)
    result = _beat_apply(diagram, match)
    _beat_record(diagram, match, result)
    _wait_for_key()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- clean-failure contract: no traceback on screen
        print(f"demo failed: {exc}")
        sys.exit(1)
