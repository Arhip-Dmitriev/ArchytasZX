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

"""Interactive shell over :mod:`archytaszx.repl.lang` and :mod:`archytaszx.repl.commands`:
history, scripts and the console entry point.

A :class:`Shell` owns one :class:`~archytaszx.repl.lang.Interpreter`, one
:class:`~archytaszx.repl.commands.Session` and a notation mode. Each input line goes through
:meth:`Shell.run_line`:

* blank lines and ``#`` comment lines produce nothing and are not recorded;
* ``history`` and ``quit``/``exit`` are handled here;
* ``:dirac``, ``:zx`` (``:zxrender``), ``:zxunicode``, ``:zxascii``, ``:both`` and ``:raw``
  pick what ``show``, verbose mode and engine commands print and how ZX diagrams are drawn,
  ``:listing`` toggles the graph listing under each drawing, ``:verbose_on``/``:verbose_off``
  switch verbose mode, ``:style`` names the display styles (render, unicode, ascii, latex;
  several print every output in each, in order) and ``:config`` shows or sets the image size
  and colour;
* any other line starting with ``:`` is an engine command, the rest of the line going to
  :func:`~archytaszx.repl.commands.run` and its outcome printed by
  :func:`~archytaszx.repl.commands.format_outcome` in the current mode;
* every other line is a :mod:`~archytaszx.repl.lang` statement; ``show`` and ``print`` print.

With ``commands=True`` every line is an engine command, without the ``:``. In verbose mode the
last statement of each line is echoed utop style (``val x : nat = 3``, ``- : ket = ...``) and,
interactively, a ``─( time )─< command n >──{ dir }─`` bar precedes each prompt.

With ``order`` output is held until an error or the end, then printed one section per style.
``error:`` lines are red when colour is on. A line ending in ``\\`` is joined to the next one
with a single space; a closed ``if`` waits
for the next line in case it starts with ``else``. :func:`main` is the
``archytaszx`` console script: ``-c`` commands and script files run non-interactively,
otherwise :meth:`Shell.loop` reads from standard input (prompt ``dir(n): ``, readline history
on a terminal).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from typing import TextIO

from archytaszx.repl import display as D
from archytaszx.repl.commands import Outcome, Session, detail_text, format_outcome, run
from archytaszx.repl.lang import Interpreter, LangError, incomplete, is_else, may_continue
from archytaszx.repl.printer import DEFAULT_MODE, NotationMode

HISTORY_FILE = "~/.archytaszx_history"
"""The readline history file, user-expanded."""

ERROR = "error"
"""The piece kind of an error line."""

_STYLE_WORDS = ("render", "unicode", "ascii", "latex")
_HISTORY_LENGTH = 1000
_QUIT_WORDS = frozenset({"quit", "exit"})
VIEWS: dict[str, tuple[NotationMode, str | None]] = {
    "dirac": (NotationMode.DIRAC, None),
    "zx": (NotationMode.ZX, "matplot"),
    "zxrender": (NotationMode.ZX, "matplot"),
    "zxunicode": (NotationMode.ZX, "unicode"),
    "zxascii": (NotationMode.ZX, "ascii"),
    "both": (NotationMode.BOTH, "matplot"),
    "raw": (NotationMode.RAW, None),
}
"""Each view word: the notation mode and the ZX drawing style it selects."""

_VIEW_HELP = {
    "dirac": "show Dirac notation",
    "zx": "draw ZX diagrams with matplotlib (alias of --zxrender)",
    "zxrender": "draw ZX diagrams with matplotlib, Unicode without an image protocol",
    "zxunicode": "draw ZX diagrams on a Unicode character grid",
    "zxascii": "draw ZX diagrams on an ASCII character grid",
    "both": "show Dirac notation and the matplotlib ZX drawing",
    "raw": "print full ZX graph listings, nothing summarised or capped",
}


class Shell:
    """A REPL session: engine :class:`Session`, notation mode, line history and error count."""

    def __init__(
        self,
        *,
        mode: NotationMode = DEFAULT_MODE,
        zx_style: str = "matplot",
        unicode: bool = False,
        session: Session | None = None,
        commands: bool = False,
        display: str | None = None,
        seed: int | None = None,
        plotter: D.Plotter | None = None,
        args: Sequence[str] = (),
        check_oracles: bool = False,
        verbose: bool = False,
        color: bool = False,
        listing: bool = False,
        styles: Sequence[str] = (),
        order: bool = False,
    ) -> None:
        """Type checks: TypeError on a mode that is not a NotationMode, a non-bool unicode or
        a session that is not a Session; ValueError on an unknown display style.

        ``display`` defaults to ``unicode`` when ``unicode`` is set, else ``plot``; ``styles``
        (several display styles, each output rendered in all of them in order) overrides it;
        ``order`` holds output until an error or :meth:`finish`, then emits it per style.
        ``args`` are the script arguments read by ``arg(k)``; ``check_oracles`` checks classical
        gates for reversibility; ``verbose`` starts in verbose mode; ``color`` colours echoes and
        errors; ``listing`` starts with graph listings under ZX drawings.
        """
        if not isinstance(mode, NotationMode):
            raise TypeError(f"mode must be a NotationMode, got {mode!r}")
        if not isinstance(unicode, bool):
            raise TypeError(f"unicode must be a bool, got {unicode!r}")
        if session is not None and not isinstance(session, Session):
            raise TypeError(f"session must be a Session, got {type(session).__name__}")
        self.mode = mode
        self.unicode = unicode
        self.session = session if session is not None else Session()
        self.commands = commands
        style = display if display is not None else ("unicode" if unicode else "plot")
        chosen = tuple(dict.fromkeys(styles)) or (style,)
        for each in chosen:
            if each not in D.STYLES:
                raise ValueError(f"unknown display style {each!r}")
        self.interpreter = Interpreter(
            style=chosen[0],
            seed=seed,
            plotter=plotter,
            args=tuple(args),
            check_oracles=check_oracles,
            mode=mode,
            zx_style=zx_style,
        )
        self.interpreter.listing = listing
        self.interpreter.styles = chosen
        self.order = order
        self.held: list[tuple[str | None, str]] = []
        """Output held under ``order``: ``(style, text)``, style None for shared text."""
        self.verbose = verbose
        self.color = color
        self.pending: list[str] = []
        """Lines of a language block whose ``{`` is not yet closed."""
        self.history: list[str] = []
        self.errors = 0
        """Count of lines whose outcome had ``ok=False``, shell errors included."""
        self.last_outcome: Outcome | None = None
        """The outcome of the latest line sent to :func:`commands.run`."""
        self.done = False
        """Set by ``quit``/``exit``."""

    def run_line(self, line: str) -> str:
        """Runs one logical line and returns its printed output (``""`` for none, and for
        held output under :attr:`order`)."""
        return self._emit(self.line_parts(line))

    def line_parts(self, line: str) -> list[tuple[str | None, str]]:
        """Runs one logical line; ``(kind, text)`` per printed piece, kind a display style,
        None for text shared by every style, or ``"error"``."""
        if not isinstance(line, str):
            self.errors += 1
            return [(ERROR, f"error: a line must be a str, got {type(line).__name__}")]
        text = line.strip()
        if self.pending and self.awaiting():
            if is_else(text):
                self.pending.append(text)
                return self._block()
            held = self.flush_parts()
            if not text:
                return held
            return held + self.line_parts(text)
        if self.pending:
            self.pending.append(text)
            return self._block()
        if not text or text.startswith("#"):
            return []
        language = not self.commands and not text.startswith(":")
        if language and (incomplete(text) or may_continue(text)):
            self.pending = [text]
            return self._block()
        self.history.append(text)
        word, rest = (text.split(maxsplit=1) + [""])[:2]
        word = word.lower()
        rest = rest.strip()
        if word in _QUIT_WORDS and not rest:
            self.done = True
            return []
        if word == "history" and not rest:
            return _shared(self._history())
        if not self.commands:
            if not text.startswith(":"):
                return self._statement(text)
            text = text[1:].strip()
            if not text:
                return []
            word, rest = (text.split(maxsplit=1) + [""])[:2]
            word = word.lower()
            rest = rest.strip()
        if word in VIEWS and not rest:
            return _shared(self._mode(word))
        if word == "listing" and not rest:
            self.interpreter.listing = not self.interpreter.listing
            return _shared(f"listing: {'on' if self.interpreter.listing else 'off'}")
        if word in ("verbose_on", "verbose_off") and not rest:
            self.verbose = word == "verbose_on"
            return _shared(f"verbose: {'on' if self.verbose else 'off'}")
        if word == "config":
            text = self._config(rest)
            return [(ERROR if text.startswith("error:") else None, text)]
        if word == "style":
            return self._style(rest)
        try:
            outcome = run(self.session, text)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._internal(exc)
        self.last_outcome = outcome
        if not outcome.ok:
            self.errors += 1
        try:
            if not outcome.ok:
                return [(ERROR, self._outcome_text(outcome, self.interpreter.styles[0]))]
            return [(st, self._outcome_text(outcome, st)) for st in self.interpreter.styles]
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._internal(exc)

    def _outcome_text(self, outcome: Outcome, style: str) -> str:
        """An engine outcome in a display style: Unicode text in ``unicode`` style or under
        ``unicode``, except in ``ascii`` style."""
        unicode = style == "unicode" or (self.unicode and style != "ascii")
        if outcome.ok and outcome.command == "detail" and unicode:
            return detail_text(outcome.detail, unicode=True)
        interp = self.interpreter
        zx = "ascii" if style == "ascii" else interp.zx_style
        return format_outcome(
            outcome, self.mode, unicode=unicode, zx_style=zx, listing=interp.listing
        )

    def _emit(self, parts: list[tuple[str | None, str]]) -> str:
        """Printable text of ``parts``: in order, each repeated style text once; under
        :attr:`order` held until an error, then every held piece by style and the errors."""
        if self.order:
            self.held += [p for p in parts if p[0] != ERROR]
            errors = [text for kind, text in parts if kind == ERROR]
            if not errors:
                return ""
            return "\n".join(x for x in (self.finish(), *map(self._red, errors)) if x)
        out: list[str] = []
        last: str | None = None
        for kind, text in parts:
            if not text or (kind not in (None, ERROR) and text == last):
                continue
            last = text if kind not in (None, ERROR) else None
            out.append(self._red(text) if kind == ERROR else text)
        return "\n".join(out)

    def finish(self) -> str:
        """The held output, one section per display style (shared text in each), emptied."""
        held, self.held = self.held, []
        styles = self.interpreter.styles
        sections = []
        for style in styles:
            texts = [text for kind, text in held if text and kind in (None, style)]
            if len(styles) > 1:
                texts.insert(0, f"── {style} ──")
            if texts:
                sections.append("\n".join(texts))
        return "\n".join(sections) if any(text for _, text in held) else ""

    def _red(self, text: str) -> str:
        """``text`` with its ``error:`` lines in red when :attr:`color` is set."""
        if not self.color:
            return text
        return "\n".join(
            self._paint(line, "31") if line.startswith("error:") else line
            for line in text.split("\n")
        )

    def awaiting(self) -> bool:
        """Whether the pending lines form a closed if that an ``else`` line may extend."""
        return bool(self.pending) and not incomplete("\n".join(self.pending))

    def _block(self) -> list[tuple[str | None, str]]:
        """Runs the pending lines once they are complete and no ``else`` can follow."""
        joined = "\n".join(self.pending)
        if incomplete(joined) or may_continue(joined):
            return []
        return self.flush_parts()

    def flush(self) -> str:
        """Runs the pending lines now; their printable text."""
        return self._emit(self.flush_parts())

    def flush_parts(self) -> list[tuple[str | None, str]]:
        """Runs the pending lines now; their pieces as :meth:`line_parts` gives them."""
        if not self.pending:
            return []
        block, self.pending = "\n".join(self.pending), []
        self.history.append(block)
        return self._statement(block)

    def _statement(self, text: str) -> list[tuple[str | None, str]]:
        """Runs a language statement; errors become ``error: ...`` pieces after what printed
        before them."""
        interp = self.interpreter
        try:
            parts = interp.execute_parts(text)
            if self.verbose:
                parts += [(st, self._echo(st)) for st in interp.styles]
            return parts
        except KeyboardInterrupt:
            raise
        except LangError as exc:
            self.errors += 1
            return [*interp.parts, (ERROR, f"error: {exc}")]
        except Exception as exc:  # noqa: BLE001
            return [*interp.parts, *self._internal(exc)]

    def run_script(self, lines: Iterable[str]) -> list[str]:
        """Outputs of each logical line that is not blank or a comment, up to ``quit``; a str
        is split into lines."""
        return [output for _, output in self.iter_script(lines)]

    def iter_script(self, lines: Iterable[str]) -> Iterator[tuple[str, str]]:
        """Yields ``(logical line, output)`` per non-blank, non-comment line, up to ``quit``; a
        str is split into lines."""
        if isinstance(lines, str):
            lines = lines.splitlines()
        for logical in _logical_lines(lines):
            if self.done:
                return
            text = logical.strip()
            if (not text or text.startswith("#")) and (not self.pending or self.awaiting()):
                continue
            yield text, self.run_line(text)
        if self.pending and not self.done:
            if self.awaiting():
                yield "", self.flush()
                return
            self.errors += 1
            self.pending = []
            yield "", self._emit([(ERROR, "error: unclosed { at the end of the input")])

    def loop(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
        """Read-eval-print until EOF or ``quit`` (returns 0), or until input cannot be read or
        output cannot be written (returns 1).

        Prompts, a banner and readline history appear only when ``stdin`` is a terminal.
        Ctrl-C discards the pending line or interrupts the running command and prints
        ``interrupted``.
        """
        source = sys.stdin if stdin is None else stdin
        sink = sys.stdout if stdout is None else stdout
        interactive = _isatty(source)
        if interactive and "plot" in self.interpreter.styles:
            D.prepare()
        use_input = interactive and source is sys.stdin and sink is sys.stdout
        readline_module = _start_readline() if use_input else None
        try:
            return self._repl(source, sink, interactive, use_input)
        except (OSError, ValueError):
            return 1
        finally:
            if readline_module is not None:
                _save_readline(readline_module)

    def _repl(self, source: TextIO, sink: TextIO, interactive: bool, use_input: bool) -> int:
        """The body of :meth:`loop`; output errors propagate."""
        if interactive and self.commands:
            _write(sink, f"ArchytasZX shell, mode {self.mode.value}; type help for commands.\n")
        while not self.done:
            try:
                try:
                    line = self._read_logical(source, sink, interactive, use_input)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.errors += 1
                    _write(sink, f"error: cannot read input: {_failure_text(exc)}\n")
                    return 1
                if line is None:
                    if interactive:
                        _write(sink, "\n")
                    held = self.flush() if self.awaiting() else ""
                    held = "\n".join(x for x in (held, self.finish()) if x)
                    if held:
                        _write(sink, held + "\n")
                    return 0
                output = self.run_line(line)
                if output:
                    _write(sink, output + "\n")
            except KeyboardInterrupt:
                _write(sink, "\ninterrupted\n" if interactive else "interrupted\n")
        held = self.finish()
        if held:
            _write(sink, held + "\n")
        return 0

    def _read_logical(
        self, source: TextIO, sink: TextIO, interactive: bool, use_input: bool
    ) -> str | None:
        """Reads physical lines up to one without a trailing ``\\`` (a comment line never
        continues); ``None`` at EOF."""
        parts: list[str] = []
        if self.verbose and interactive and not self.pending:
            _write(sink, self.bar() + "\n")
        prompt = self.prompt(bool(self.pending))
        while True:
            raw = _read_physical(source, sink, prompt if interactive else "", use_input)
            if raw is None:
                return " ".join(parts) if parts else None
            stripped = raw.rstrip("\r\n").rstrip()
            if stripped.endswith("\\") and (parts or not stripped.lstrip().startswith("#")):
                parts.append(stripped[:-1].strip())
                prompt = self.prompt(True)
                continue
            parts.append(stripped.strip())
            return " ".join(part for part in parts if part)

    def _mode(self, word: str) -> str:
        """Sets the notation mode and the ZX drawing style of a view word."""
        self.mode, style = VIEWS[word]
        self.interpreter.mode = self.mode
        if style is not None:
            self.interpreter.zx_style = style
        return f"mode: {word}"

    def _style(self, rest: str) -> list[tuple[str | None, str]]:
        """``style`` names the display styles; ``style WORD ...`` sets them, in order."""
        words = rest.lower().split()
        bad = [w for w in words if w not in D.STYLES and w != "render"]
        if bad:
            self.errors += 1
            return [(ERROR, f"error: unknown style {bad[0]!r}; use {', '.join(_STYLE_WORDS)}")]
        if words:
            chosen = tuple(dict.fromkeys("plot" if w == "render" else w for w in words))
            self.interpreter.styles = chosen
            self.interpreter.style = chosen[0]
        names = ("render" if st == "plot" else st for st in self.interpreter.styles)
        return _shared(f"style: {' '.join(names)}")

    def _config(self, rest: str) -> str:
        """``config`` lists the image settings; ``config KEY VALUE`` sets one, ``config save``
        writes them to :func:`~archytaszx.repl.display.config_path`, ``config reset`` restores
        the defaults."""
        words = rest.split(maxsplit=1)
        try:
            if words == ["save"]:
                return f"saved {D.save_config(D.CONFIG)}"
            if words == ["reset"]:
                D.CONFIG.size, D.CONFIG.color = D.Config().size, D.Config().color
            elif len(words) == 2:
                D.set_config(D.CONFIG, words[0].lower(), words[1])
            elif words:
                raise ValueError("usage: config [KEY VALUE | save | reset]")
        except (ValueError, OSError) as exc:
            self.errors += 1
            return f"error: {exc}"
        return f"size = {D.CONFIG.size:g}\ncolor = {D.CONFIG.color}\nfile: {D.config_path()}"

    def _paint(self, text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else text

    def _echo(self, style: str | None = None) -> str:
        """``val name : type = value`` or ``- : type = value`` for the last statement; a
        one-line value in ``latex`` style as ``$...$``."""
        last = self.interpreter.last
        if last is None:
            return ""
        name, kind, value = self.interpreter.echo_parts(last, style)
        if style == "latex" and value is not None and "\n" not in value:
            value = f"${value}$"
        head = "-" if name == "-" else f"{self._paint('val', '1')} {name}"
        line = f"{head} : {self._paint(kind, '36')}"
        if value is None:
            return line
        if "\n" in value:
            return f"{line} =\n{value}"
        return f"{line} = {value}"

    def prompt(self, continuation: bool = False) -> str:
        """``dir(n): ``, the working directory's name and the next line number; dots replace
        the name on a continuation line."""
        number = len(self.history) + 1
        name = _place()
        if continuation:
            name = "." * len(name)
        return f"{name}({number}): "

    def bar(self, width: int | None = None) -> str:
        """The utop-style rule shown above each prompt in verbose mode."""
        width = width or shutil.get_terminal_size((80, 24)).columns
        left = f"─( {time.strftime('%H:%M:%S')} )─< command {len(self.history) + 1} >"
        right = f"{{ {_place()} }}─"
        fill = "─" * max(1, width - len(left) - len(right))
        if not self.color:
            return left + fill + right
        return self._paint(left, "33") + self._paint(fill, "2") + self._paint(right, "32")

    def _history(self) -> str:
        """Numbered list of recorded lines."""
        width = len(str(len(self.history)))
        return "\n".join(f"{i:>{width}}  {line}" for i, line in enumerate(self.history, 1))

    def _internal(self, exc: Exception) -> list[tuple[str | None, str]]:
        """Clean one-line report of an unexpected exception."""
        self.errors += 1
        return [(ERROR, f"error: internal: {_failure_text(exc)}")]


def _shared(text: str) -> list[tuple[str | None, str]]:
    return [(None, text)] if text else []


def _place() -> str:
    """The working directory's name: ``home`` for the home directory, ``/`` for the root."""
    try:
        cwd = os.getcwd()
    except OSError:
        return "?"
    if cwd == os.path.expanduser("~"):
        return "home"
    return os.path.basename(cwd) or cwd


def _logical_lines(lines: Iterable[str]) -> Iterator[str]:
    """Joins each line ending in ``\\`` to the next with a single space; a comment line never
    continues."""
    parts: list[str] = []
    for raw in lines:
        stripped = raw.rstrip("\r\n").rstrip()
        if stripped.endswith("\\") and (parts or not stripped.lstrip().startswith("#")):
            parts.append(stripped[:-1].strip())
            continue
        parts.append(stripped.strip())
        yield " ".join(part for part in parts if part)
        parts = []
    if parts:
        yield " ".join(part for part in parts if part)


def _read_physical(source: TextIO, sink: TextIO, prompt: str, use_input: bool) -> str | None:
    """One physical line from ``source``, ``None`` at EOF."""
    if use_input:
        try:
            return input(prompt)
        except EOFError:
            return None
    if prompt:
        sink.write(prompt)
        sink.flush()
    raw = source.readline()
    return raw if raw else None


def _failure_text(exc: BaseException) -> str:
    """``<class>: <message>`` with whitespace runs collapsed."""
    return " ".join(f"{type(exc).__name__}: {exc}".split())


def _write(sink: TextIO, text: str) -> None:
    """Writes and flushes ``text``; characters the sink cannot encode become escapes."""
    try:
        sink.write(text)
    except UnicodeEncodeError:
        encoding = getattr(sink, "encoding", None) or "ascii"
        sink.write(text.encode(encoding, "backslashreplace").decode(encoding, "replace"))
    sink.flush()


def _isatty(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _start_readline() -> object | None:
    """Imports readline and loads :data:`HISTORY_FILE`; ``None`` when unavailable."""
    try:
        import readline
    except ImportError:
        return None
    try:
        readline.read_history_file(os.path.expanduser(HISTORY_FILE))
    except OSError:
        pass
    readline.set_history_length(_HISTORY_LENGTH)
    return readline


def _save_readline(module: object) -> None:
    """Writes the readline history to :data:`HISTORY_FILE`."""
    try:
        module.write_history_file(os.path.expanduser(HISTORY_FILE))  # type: ignore[attr-defined]
    except OSError:
        pass


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archytaszx",
        description="ArchytasZX symbolic qufinite ZX-calculus shell.",
        add_help=False,
    )
    parser.add_argument("--help", action="help", help="show this help message and exit")
    modes = parser.add_mutually_exclusive_group()
    for word in VIEWS:
        modes.add_argument(
            f"--{word}",
            f"-{word}",
            dest="mode",
            action="store_const",
            const=word,
            help=_VIEW_HELP[word] + (" (default)" if word == DEFAULT_MODE.value else ""),
        )
    parser.set_defaults(mode=DEFAULT_MODE.value)
    for word, style, text in (
        ("render", "plot", "show rendered matplotlib mathtext images (default)"),
        ("unicode", "unicode", "show Unicode text"),
        ("ascii", "ascii", "show ASCII text in the input syntax, without plots"),
        ("latex", "latex", "show LaTeX source"),
    ):
        parser.add_argument(
            f"--{word}",
            dest="styles",
            action="append_const",
            const=style,
            help=text + "; several style flags show every one, in order",
        )
    parser.add_argument(
        "--order",
        action="store_true",
        help="hold output until an error or the end, then show it grouped by style",
    )
    parser.add_argument(
        "--commands", action="store_true", help="read engine commands instead of the language"
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for rand")
    parser.add_argument(
        "--images",
        metavar="DIR",
        default=None,
        help="also save every rendered image (formulas and ZX drawings) as a PNG in DIR",
    )
    parser.add_argument(
        "-c",
        "--command",
        action="append",
        default=[],
        metavar="CMD",
        help="run CMD (repeatable, in order) instead of the interactive loop",
    )
    parser.add_argument("--verbose", action="store_true", help="echo each line's value, utop style")
    parser.add_argument(
        "scripts",
        nargs="*",
        metavar="FILE|ARG",
        help="script files (*.azx) to run; other values are arguments read by arg(k)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point.

    With ``-c`` or script files: runs the ``-c`` commands, then each file, in one session, and
    returns 1 when any line had ``ok=False``, else 0; an unreadable file returns 2 before
    anything runs. With neither: the interactive :meth:`Shell.loop`. A usage error returns 2,
    a closed standard output 1, and Ctrl-C outside a command 130.
    """
    try:
        args = _argument_parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
    styles = tuple(args.styles or ())
    paths = [p for p in args.scripts if args.commands or p.endswith(".azx")]
    script_args = [p for p in args.scripts if not (args.commands or p.endswith(".azx"))]
    try:
        D.CONFIG = D.load_config()
    except ValueError as exc:
        sys.stderr.write(f"archytaszx: {exc}; using the default display settings\n")
    D.SAVE_DIR = args.images
    mode, zx_style = VIEWS[args.mode]
    shell = Shell(
        mode=mode,
        zx_style=zx_style or "matplot",
        unicode="unicode" in styles,
        commands=args.commands,
        seed=args.seed,
        args=script_args,
        verbose=args.verbose,
        color=_isatty(sys.stdout) and not os.environ.get("NO_COLOR"),
        styles=styles,
        order=args.order,
    )
    drawn = mode is not NotationMode.DIRAC and zx_style == "matplot"
    if ("plot" in shell.interpreter.styles or drawn) and (paths or args.command):
        D.prepare()
    commands: list[str] = list(args.command)
    scripts: list[list[str]] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                scripts.append(handle.read().splitlines())
        except (OSError, UnicodeDecodeError) as exc:
            sys.stderr.write(f"archytaszx: cannot read {path}: {exc}\n")
            return 2
    _tolerant_streams()
    try:
        if not commands and not scripts:
            code = shell.loop()
        else:
            for lines in [commands, *scripts]:
                for _, output in shell.iter_script(lines):
                    if output:
                        _write(sys.stdout, output + "\n")
                if shell.done:
                    break
            held = shell.finish()
            if held:
                _write(sys.stdout, held + "\n")
            code = 1 if shell.errors else 0
        sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130
    except (OSError, ValueError):
        _discard_stdout()
        return 1
    return code


def _tolerant_streams() -> None:
    """Undecodable standard input becomes U+FFFD; unencodable output becomes escapes."""
    for stream, errors in ((sys.stdin, "replace"), (sys.stdout, "backslashreplace")):
        try:
            stream.reconfigure(errors=errors)  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def _discard_stdout() -> None:
    """Points the standard output descriptor at the null device."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (AttributeError, ValueError, OSError):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
