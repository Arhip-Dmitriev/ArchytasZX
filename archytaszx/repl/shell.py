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
* a line starting with ``:`` is an engine command (``:mode [dirac|zx]`` included), the rest
  of the line going to :func:`~archytaszx.repl.commands.run` and its outcome printed by
  :func:`~archytaszx.repl.commands.format_outcome` in the current mode;
* every other line is a :mod:`~archytaszx.repl.lang` statement; only ``show`` prints.

With ``commands=True`` every line is an engine command, without the ``:``.

The mode is passed to ``format_outcome`` only; commands never see it. A line ending in ``\\``
is joined to the next one with a single space. :func:`main` is the ``archytaszx`` console
script: ``-c`` commands and script files run non-interactively, otherwise :meth:`Shell.loop`
reads from standard input (prompt ``azx> ``, readline history on a terminal).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from typing import TextIO

from archytaszx.repl import display as D
from archytaszx.repl.commands import Outcome, Session, detail_text, format_outcome, run
from archytaszx.repl.lang import Interpreter, LangError
from archytaszx.repl.printer import DEFAULT_MODE, NotationMode

PROMPT = "azx> "
"""The primary interactive prompt."""

CONTINUATION_PROMPT = "...> "
"""The prompt after a line ending in ``\\``."""

HISTORY_FILE = "~/.archytaszx_history"
"""The readline history file, user-expanded."""

_HISTORY_LENGTH = 1000
_QUIT_WORDS = frozenset({"quit", "exit"})


class Shell:
    """A REPL session: engine :class:`Session`, notation mode, line history and error count."""

    def __init__(
        self,
        *,
        mode: NotationMode = DEFAULT_MODE,
        unicode: bool = False,
        session: Session | None = None,
        commands: bool = False,
        display: str | None = None,
        seed: int | None = None,
        plotter: D.Plotter | None = None,
    ) -> None:
        """Type checks: TypeError on a mode that is not a NotationMode, a non-bool unicode or
        a session that is not a Session; ValueError on an unknown display style.

        ``display`` defaults to ``unicode`` when ``unicode`` is set, else ``plot``.
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
        self.interpreter = Interpreter(style=style, seed=seed, plotter=plotter)
        self.history: list[str] = []
        self.errors = 0
        """Count of lines whose outcome had ``ok=False``, shell errors included."""
        self.last_outcome: Outcome | None = None
        """The outcome of the latest line sent to :func:`commands.run`."""
        self.done = False
        """Set by ``quit``/``exit``."""

    def run_line(self, line: str) -> str:
        """Runs one logical line and returns its printed output (``""`` for none)."""
        if not isinstance(line, str):
            self.errors += 1
            return f"error: a line must be a str, got {type(line).__name__}"
        text = line.strip()
        if not text or text.startswith("#"):
            return ""
        self.history.append(text)
        word, rest = (text.split(maxsplit=1) + [""])[:2]
        word = word.lower()
        rest = rest.strip()
        if word in _QUIT_WORDS and not rest:
            self.done = True
            return ""
        if word == "history" and not rest:
            return self._history()
        if not self.commands:
            if not text.startswith(":"):
                return self._statement(text)
            text = text[1:].strip()
            if not text:
                return ""
            word, rest = (text.split(maxsplit=1) + [""])[:2]
            word = word.lower()
            rest = rest.strip()
        if word == "mode":
            return self._mode(rest)
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
            if outcome.ok and outcome.command == "detail" and self.unicode:
                return detail_text(outcome.detail, unicode=True)
            return format_outcome(outcome, self.mode, unicode=self.unicode)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._internal(exc)

    def _statement(self, text: str) -> str:
        """Runs a language statement; errors become ``error: ...`` lines."""
        try:
            return self.interpreter.execute(text)
        except KeyboardInterrupt:
            raise
        except LangError as exc:
            self.errors += 1
            return f"error: {exc}"
        except Exception as exc:  # noqa: BLE001
            return self._internal(exc)

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
            if not text or text.startswith("#"):
                continue
            yield text, self.run_line(text)

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
        if interactive:
            if self.commands:
                banner = f"ArchytasZX shell, mode {self.mode.value}; type help for commands.\n"
            else:
                banner = "ArchytasZX Dirac shell; ':help' lists engine commands.\n"
            _write(sink, banner)
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
                    return 0
                output = self.run_line(line)
                if output:
                    _write(sink, output + "\n")
            except KeyboardInterrupt:
                _write(sink, "\ninterrupted\n" if interactive else "interrupted\n")
        return 0

    def _read_logical(
        self, source: TextIO, sink: TextIO, interactive: bool, use_input: bool
    ) -> str | None:
        """Reads physical lines up to one without a trailing ``\\`` (a comment line never
        continues); ``None`` at EOF."""
        parts: list[str] = []
        prompt = PROMPT
        while True:
            raw = _read_physical(source, sink, prompt if interactive else "", use_input)
            if raw is None:
                return " ".join(parts) if parts else None
            stripped = raw.rstrip("\r\n").rstrip()
            if stripped.endswith("\\") and (parts or not stripped.lstrip().startswith("#")):
                parts.append(stripped[:-1].strip())
                prompt = CONTINUATION_PROMPT
                continue
            parts.append(stripped.strip())
            return " ".join(part for part in parts if part)

    def _mode(self, rest: str) -> str:
        """Shows or sets the notation mode."""
        if rest:
            try:
                self.mode = NotationMode(rest.lower())
            except ValueError:
                self.errors += 1
                choices = "|".join(m.value for m in NotationMode)
                return f"error: unknown mode {rest!r}; usage: mode [{choices}]"
        return f"mode: {self.mode.value}"

    def _history(self) -> str:
        """Numbered list of recorded lines."""
        width = len(str(len(self.history)))
        return "\n".join(f"{i:>{width}}  {line}" for i, line in enumerate(self.history, 1))

    def _internal(self, exc: Exception) -> str:
        """Clean one-line report of an unexpected exception."""
        self.errors += 1
        return f"error: internal: {_failure_text(exc)}"


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
        prog="archytaszx", description="ArchytasZX symbolic qufinite ZX-calculus shell."
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in NotationMode],
        default=DEFAULT_MODE.value,
        help="output notation (default: %(default)s)",
    )
    parser.add_argument("--unicode", action="store_true", help="render with Unicode symbols")
    parser.add_argument("--ascii", action="store_true", help="show in ASCII, without plots")
    parser.add_argument(
        "--commands", action="store_true", help="read engine commands instead of the language"
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for numeric measurement")
    parser.add_argument(
        "-c",
        "--command",
        action="append",
        default=[],
        metavar="CMD",
        help="run CMD (repeatable, in order) instead of the interactive loop",
    )
    parser.add_argument("--echo", action="store_true", help="print 'azx> LINE' before each output")
    parser.add_argument("scripts", nargs="*", metavar="FILE", help="script files to run")
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
    display = "ascii" if args.ascii else None
    shell = Shell(
        mode=NotationMode(args.mode),
        unicode=args.unicode,
        commands=args.commands,
        display=display,
        seed=args.seed,
    )
    commands: list[str] = list(args.command)
    scripts: list[list[str]] = []
    for path in args.scripts:
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
                for line, output in shell.iter_script(lines):
                    if args.echo:
                        _write(sys.stdout, f"{PROMPT}{line}\n")
                    if output:
                        _write(sys.stdout, output + "\n")
                if shell.done:
                    break
            code = 1 if shell.errors else 0
        sys.stdout.flush()
        D.finish()
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
