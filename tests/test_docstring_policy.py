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

"""Enforces the project's docstring budgets across ``qufzx/``, and bans the word "because".

A docstring states what a thing is or how it works; reasoning belongs in FULL_PLAN.md. What
this module checks is the *volume* of prose, not its content: three line budgets and a ratio
cap. Those bound how much rationale any one file can carry, they do not detect it.

The ``because`` ban is a lexical check on one word, not a WHY detector. Prose that reaches
the same place through "so", "since", "rather than" or "which would otherwise" passes, and a
good deal of the tree's prose does. Read the budgets as the enforceable half of the policy
and the rest as a convention this suite cannot check.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = _REPO_ROOT / "qufzx"

_MAX_MODULE_DOCSTRING_LINES = 65
"""Ceiling on a module docstring. ``qufzx/rewrite/match.py`` and
``qufzx/diagram/validate.py`` share the high-water mark at 64."""

_MAX_DOCSTRING_LINES = 30
"""Ceiling on every other docstring, module-attribute docstrings included.
``qufzx.rewrite.match``'s ``_ConstraintRecord`` sets the high-water mark at 29."""

_MAX_PROSE_RATIO = 0.45
"""Ceiling on docstring+comment lines as a fraction of a file's non-blank lines.
``qufzx/semantics/contract_numeric.py`` sets the high-water mark at 0.444. Set just above
the measured maximum, so prose growth in any file fails rather than being absorbed by
headroom."""

_LICENSE_HEADER_LINES = 12

_BANNED = re.compile(r"\bbecause\b", re.IGNORECASE)
"""The one rationale marker checked lexically. Not a WHY detector: see the module
docstring."""


def _python_files() -> list[Path]:
    return sorted(p for p in _PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def _prose_line_numbers(source: str) -> set[int]:
    """Every line number covered by a docstring or a comment, past the license header."""
    prose: set[int] = set()
    for node in _docstring_nodes(ast.parse(source)):
        prose |= set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    for number, line in enumerate(source.splitlines(), 1):
        if line.lstrip().startswith("#"):
            prose.add(number)
    return {number for number in prose if number > _LICENSE_HEADER_LINES}


def _docstring_nodes(tree: ast.Module) -> list[ast.Expr]:
    """Every bare string expression: real docstrings and attribute docstrings alike."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
class TestDocstringPolicy:
    """One parametrized case per module in ``qufzx/``."""

    def test_module_docstring_is_within_budget(self, path: Path) -> None:
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        lines = len(docstring.splitlines()) if docstring else 0
        assert lines <= _MAX_MODULE_DOCSTRING_LINES, (
            f"{path.relative_to(_REPO_ROOT)}: module docstring is {lines} lines, over the "
            f"{_MAX_MODULE_DOCSTRING_LINES}-line budget. State what the module is and the "
            "contracts it keeps; put the reasoning in FULL_PLAN.md."
        )

    def test_no_docstring_exceeds_its_budget(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_docstring_line = tree.body[0].lineno if ast.get_docstring(tree) else None
        oversized = [
            (node.lineno, (node.end_lineno or node.lineno) - node.lineno + 1)
            for node in _docstring_nodes(tree)
            if node.lineno != module_docstring_line
            and (node.end_lineno or node.lineno) - node.lineno + 1 > _MAX_DOCSTRING_LINES
        ]
        assert not oversized, (
            f"{path.relative_to(_REPO_ROOT)}: docstring(s) over the "
            f"{_MAX_DOCSTRING_LINES}-line budget at line(s) "
            f"{[line for line, _ in oversized]} ({[n for _, n in oversized]} lines)"
        )

    def test_prose_does_not_outweigh_code(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        if not any(not isinstance(node, ast.Expr) for node in tree.body):
            # A Phase 0.2 skeleton: a docstring and nothing else, so the ratio is 1 by
            # construction and says nothing about how it is written.
            return
        prose_lines = _prose_line_numbers(source)
        non_blank = {
            number
            for number, line in enumerate(source.splitlines(), 1)
            if line.strip() and number > _LICENSE_HEADER_LINES
        }
        prose_lines &= non_blank
        if not non_blank:
            return
        ratio = len(prose_lines) / len(non_blank)
        assert ratio <= _MAX_PROSE_RATIO, (
            f"{path.relative_to(_REPO_ROOT)}: {ratio:.0%} of non-blank lines are docstring "
            f"or comment, over the {_MAX_PROSE_RATIO:.0%} budget "
            f"({len(prose_lines)} of {len(non_blank)})"
        )

    def test_no_docstring_or_comment_uses_the_banned_word(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        prose_lines = _prose_line_numbers(source)
        offenders = [
            number
            for number, line in enumerate(source.splitlines(), 1)
            if number in prose_lines and _BANNED.search(line)
        ]
        assert not offenders, (
            f"{path.relative_to(_REPO_ROOT)}: 'because' at line(s) {offenders}. A docstring "
            "states what a thing is or how it works; a sentence needing 'because' is "
            "answering why, which belongs in FULL_PLAN.md. This check sees only that one "
            "word -- the convention is broader than what it enforces."
        )
