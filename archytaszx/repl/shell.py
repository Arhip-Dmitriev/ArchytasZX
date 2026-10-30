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

"""Interactive shell over :mod:`archytaszx.repl.commands`: history, scripts and the console
entry point.

A :class:`Shell` owns one :class:`~archytaszx.repl.commands.Session` and a notation mode. Each
input line goes through :meth:`Shell.run_line`, a leading ``:`` optional:

* blank lines and ``#`` comment lines produce nothing and are not recorded;
* ``history`` and ``quit``/``exit`` are handled here;
* ``dirac``, ``zx`` (``zxrender``), ``zxunicode``, ``zxascii``, ``both`` and ``raw`` pick how
  diagrams are printed, ``listing`` toggles the graph listing under each drawing and
  ``config`` shows or sets the image size and colour;
* every other line is an engine command, run by :func:`~archytaszx.repl.commands.run` and
  printed by :func:`~archytaszx.repl.commands.format_outcome` in the current mode.

A line ending in ``\\`` is joined to the next one with a single space. ``error:`` lines are red
when colour is on. :func:`main` is the ``archytaszx`` console script: ``-c`` commands and script
files run non-interactively, otherwise :meth:`Shell.loop` reads from standard input (prompt
``dir(n): ``, readline history on a terminal).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from typing import TextIO

from archytaszx.repl import terminal as T
from archytaszx.repl.commands import Outcome, Session, detail_text, format_outcome, run
from archytaszx.repl.printer import DEFAULT_MODE, NotationMode

HISTORY_FILE = "~/.archytaszx_history"
"""The readline history file, user-expanded."""

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
        color: bool = False,
        listing: bool = False,
    ) -> None:
        """Type checks: TypeError on a mode that is not a NotationMode, a non-bool unicode or
        a session that is not a Session."""
        if not isinstance(mode, NotationMode):
            raise TypeError(f"mode must be a NotationMode, got {mode!r}")
        if not isinstance(unicode, bool):
            raise TypeError(f"unicode must be a bool, got {unicode!r}")
        if session is not None and not isinstance(session, Session):
            raise TypeError(f"session must be a Session, got {type(session).__name__}")
        self.mode = mode
        self.zx_style = zx_style
        self.unicode = unicode
        self.session = session if session is not None else Session()
        self.color = color
        self.listing = listing
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
            return self._red(f"error: a line must be a str, got {type(line).__name__}")
        text = line.strip()
        if not text or text.startswith("#"):
            return ""
        self.history.append(text)
        if text.startswith(":"):
            text = text[1:].strip()
            if not text:
                return ""
        word, rest = (text.split(maxsplit=1) + [""])[:2]
        word = word.lower()
        rest = rest.strip()
        if word in _QUIT_WORDS and not rest:
            self.done = True
            return ""
        if word == "history" and not rest:
            return self._history()
        if word in VIEWS and not rest:
            return self._mode(word)
        if word == "listing" and not rest:
            self.listing = not self.listing
            return f"listing: {'on' if self.listing else 'off'}"
        if word == "config":
            return self._red(self._config(rest))
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
            shown = self._outcome_text(outcome)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._internal(exc)
        return shown if outcome.ok else self._red(shown)

    def _outcome_text(self, outcome: Outcome) -> str:
        """An engine outcome in the current mode, Unicode text under :attr:`unicode`."""
        if outcome.ok and outcome.command == "detail" and self.unicode:
            return detail_text(outcome.detail, unicode=True)
        return format_outcome(
            outcome, self.mode, unicode=self.unicode, zx_style=self.zx_style, listing=self.listing
        )

    def _red(self, text: str) -> str:
        """``text`` with its ``error:`` lines in red when :attr:`color` is set."""
        if not self.color:
            return text
        return "\n".join(
            f"\x1b[31m{line}\x1b[0m" if line.startswith("error:") else line
            for line in text.split("\n")
        )

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
        if interactive and self.mode is not NotationMode.DIRAC and self.zx_style == "matplot":
            T.prepare()
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
        prompt = self.prompt(False)
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
            line = " ".join(part for part in parts if part)
            if interactive and _BLOCK_HEAD_RE.fullmatch(line):
                return self._read_block(line, source, sink, use_input)
            return line

    def _read_block(self, head: str, source: TextIO, sink: TextIO, use_input: bool) -> str:
        """``head`` and the lines after it up to a blank line or EOF, joined by newlines."""
        lines = [head]
        while True:
            raw = _read_physical(source, sink, self.prompt(True), use_input)
            code = "" if raw is None else raw.split("#", 1)[0].strip()
            if raw is None or not raw.strip():
                return "\n".join(lines)
            if code:
                lines.append(code)

    def _mode(self, word: str) -> str:
        """Sets the notation mode and the ZX drawing style of a view word."""
        self.mode, style = VIEWS[word]
        if style is not None:
            self.zx_style = style
        return f"mode: {word}"

    def _config(self, rest: str) -> str:
        """``config`` lists the image settings; ``config KEY VALUE`` sets one, ``config save``
        writes them to :func:`~archytaszx.repl.terminal.config_path`, ``config reset``
        restores the defaults."""
        words = rest.split(maxsplit=1)
        try:
            if words == ["save"]:
                return f"saved {T.save_config(T.CONFIG)}"
            if words == ["reset"]:
                T.CONFIG.size, T.CONFIG.color = T.Config().size, T.Config().color
            elif len(words) == 2:
                T.set_config(T.CONFIG, words[0].lower(), words[1])
            elif words:
                raise ValueError("usage: config [KEY VALUE | save | reset]")
        except (ValueError, OSError) as exc:
            self.errors += 1
            return f"error: {exc}"
        return f"size = {T.CONFIG.size:g}\ncolor = {T.CONFIG.color}\nfile: {T.config_path()}"

    def prompt(self, continuation: bool = False) -> str:
        """``dir(n): ``, the working directory's name and the next line number; dots replace
        the name on a continuation line."""
        number = len(self.history) + 1
        name = _place()
        if continuation:
            name = "." * len(name)
        return f"{name}({number}): "

    def _history(self) -> str:
        """Numbered list of recorded lines."""
        width = len(str(len(self.history)))
        return "\n".join(f"{i:>{width}}  {line}" for i, line in enumerate(self.history, 1))

    def _internal(self, exc: Exception) -> str:
        """Clean one-line report of an unexpected exception."""
        self.errors += 1
        return self._red(f"error: internal: {_failure_text(exc)}")


def _place() -> str:
    """The working directory's name: ``home`` for the home directory, ``/`` for the root."""
    try:
        cwd = os.getcwd()
    except OSError:
        return "?"
    if cwd == os.path.expanduser("~"):
        return "home"
    return os.path.basename(cwd) or cwd


_BLOCK_HEAD_RE = re.compile(r":?load|@[A-Za-z_]\w*\s*=")


def _logical_lines(lines: Iterable[str]) -> Iterator[str]:
    """Joins each line ending in ``\\`` to the next with a single space, and each indented
    line to the command above it with a newline; comments are cut from joined lines, and blank
    or comment lines never continue."""
    pending: str | None = None
    parts: list[str] = []
    for raw in lines:
        stripped = raw.rstrip("\r\n").rstrip()
        code = stripped.split("#", 1)[0].strip()
        if not parts and not code:
            yield stripped.strip()
            continue
        if not parts and pending is not None and stripped[:1] in (" ", "\t"):
            pending = pending.split("#", 1)[0].rstrip() + "\n" + code
            continue
        if stripped.endswith("\\") and (parts or not stripped.lstrip().startswith("#")):
            parts.append(stripped[:-1].strip())
            continue
        parts.append(stripped.strip())
        if pending is not None:
            yield pending
        pending = " ".join(part for part in parts if part)
        parts = []
    if parts:
        tail = " ".join(part for part in parts if part)
        pending = tail if pending is None else f"{pending}\n{tail}"
    if pending is not None:
        yield pending


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
    parser.add_argument("--unicode", action="store_true", help="print Unicode text")
    saving = parser.add_mutually_exclusive_group()
    saving.add_argument(
        "--images",
        metavar="DIR",
        default=None,
        help="also save every matplotlib ZX drawing as a PNG in DIR",
    )
    saving.add_argument(
        "--images_white",
        metavar="DIR",
        default=None,
        help="as --images, with black drawings on a white background",
    )
    parser.add_argument(
        "-c",
        "--command",
        action="append",
        default=[],
        metavar="CMD",
        help="run CMD (repeatable, in order) instead of the interactive loop",
    )
    parser.add_argument("scripts", nargs="*", metavar="FILE", help="command files to run")
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
    try:
        T.CONFIG = T.load_config()
    except ValueError as exc:
        sys.stderr.write(f"archytaszx: {exc}; using the default display settings\n")
    T.SAVE_DIR = args.images or args.images_white
    T.SAVE_WHITE = args.images_white is not None
    mode, zx_style = VIEWS[args.mode]
    shell = Shell(
        mode=mode,
        zx_style=zx_style or "matplot",
        unicode=args.unicode,
        color=_isatty(sys.stdout) and not os.environ.get("NO_COLOR"),
    )
    if mode is not NotationMode.DIRAC and zx_style == "matplot" and (args.scripts or args.command):
        T.prepare()
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
                for _, output in shell.iter_script(lines):
                    if output:
                        _write(sys.stdout, output + "\n")
                if shell.done:
                    break
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
