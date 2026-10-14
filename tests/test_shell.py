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

"""Unit tests for :mod:`archytaszx.repl.shell`: run_line, scripts, the IO loop and main."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from archytaszx.repl import commands as C
from archytaszx.repl import shell as S
from archytaszx.repl.printer import NotationMode
from archytaszx.repl.shell import PROMPT, Shell, main

EXAMPLE = "sum_{k=0}^{3-1} |k>^{30}; copy"
REPO = Path(__file__).resolve().parent.parent


def listing(text: str) -> bool:
    return any(ln.lstrip().startswith("graph:") for ln in text.splitlines())


class FakeStdin(io.StringIO):
    """A StringIO whose readline raises queued exceptions before reading."""

    def __init__(self, text: str, raises: list[BaseException], tty: bool = False) -> None:
        super().__init__(text)
        self.raises = list(raises)
        self.tty = tty

    def readline(self, size: int | None = -1) -> str:  # type: ignore[override]
        if self.raises:
            raise self.raises.pop(0)
        return super().readline()

    def isatty(self) -> bool:
        return self.tty


class BrokenStdout(io.StringIO):
    def write(self, text: str) -> int:
        raise OSError("closed")


# -- construction --------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        shell = Shell(commands=True)
        assert shell.mode is NotationMode.DIRAC
        assert shell.unicode is False
        assert isinstance(shell.session, C.Session)
        assert shell.history == [] and shell.errors == 0
        assert shell.last_outcome is None and shell.done is False

    def test_given_session_is_used(self) -> None:
        session = C.Session()
        assert Shell(commands=True, session=session).session is session

    @pytest.mark.parametrize(
        "kwargs",
        [{"mode": "zx"}, {"unicode": 1}, {"session": object()}],
        ids=["mode", "unicode", "session"],
    )
    def test_type_errors(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(TypeError):
            Shell(commands=True, **kwargs)  # type: ignore[arg-type]


# -- run_line ------------------------------------------------------------------------------


class TestRunLine:
    @pytest.mark.parametrize("line", ["", "   ", "\t", "# a comment", "   # indented comment"])
    def test_blank_and_comment_lines(self, line: str) -> None:
        shell = Shell(commands=True)
        assert shell.run_line(line) == ""
        assert shell.history == [] and shell.errors == 0 and shell.last_outcome is None

    def test_history_records_stripped_lines(self) -> None:
        shell = Shell(commands=True)
        shell.run_line("  help  ")
        shell.run_line("mode")
        shell.run_line("apply nosuchrule")
        assert shell.history == ["help", "mode", "apply nosuchrule"]

    def test_history_listing(self) -> None:
        shell = Shell(commands=True)
        for line in ("help", "mode", "", "# no", "rules"):
            shell.run_line(line)
        assert shell.run_line("history") == "1  help\n2  mode\n3  rules\n4  history"

    def test_history_listing_widths(self) -> None:
        shell = Shell(commands=True)
        for _ in range(9):
            shell.run_line("mode")
        lines = shell.run_line("history").splitlines()
        assert lines[0] == " 1  mode" and lines[-1] == "10  history"

    def test_mode_get_and_set(self) -> None:
        shell = Shell(commands=True)
        assert shell.run_line("mode") == "mode: dirac"
        assert shell.run_line("mode zx") == "mode: zx"
        assert shell.mode is NotationMode.ZX
        assert shell.run_line("MODE Dirac") == "mode: dirac"
        assert shell.mode is NotationMode.DIRAC
        assert shell.errors == 0

    def test_mode_bad_value(self) -> None:
        shell = Shell(commands=True, mode=NotationMode.ZX)
        out = shell.run_line("mode latex")
        assert out == "error: unknown mode 'latex'; usage: mode [dirac|zx|both]"
        assert shell.mode is NotationMode.ZX and shell.errors == 1

    def test_mode_drives_rendering(self) -> None:
        shell = Shell(commands=True)
        dirac = shell.run_line(EXAMPLE)
        shell.run_line("mode zx")
        zx = shell.run_line("show")
        assert not listing(dirac) and listing(zx)

    @pytest.mark.parametrize("word", ["quit", "exit", "QUIT", "  exit  "])
    def test_quit_and_exit(self, word: str) -> None:
        shell = Shell(commands=True)
        assert shell.run_line(word) == ""
        assert shell.done and shell.errors == 0

    def test_quit_with_arguments_is_not_quit(self) -> None:
        shell = Shell(commands=True)
        out = shell.run_line("quit now")
        assert not shell.done
        assert out.startswith("error: ")

    def test_help_lists_commands_and_shell_commands(self) -> None:
        out = Shell(commands=True).run_line("help")
        for usage in ("mode [dirac|zx]", "history --", "quit --", "apply RULE", "induct"):
            assert usage in out
        assert "aliases: list-rules = rules" in out

    def test_help_for_one_command(self) -> None:
        assert Shell(commands=True).run_line("help mode").startswith("mode [dirac|zx]")

    def test_command_output_and_last_outcome(self) -> None:
        shell = Shell(commands=True)
        out = shell.run_line(EXAMPLE)
        assert out.startswith("loaded dirac input")
        assert shell.last_outcome is not None and shell.last_outcome.command == "load"
        assert shell.last_outcome.ok and shell.errors == 0

    def test_errors_counter(self) -> None:
        shell = Shell(commands=True)
        shell.run_line("apply spider_fusion")
        assert shell.errors == 1 and not shell.last_outcome.ok
        shell.run_line(EXAMPLE)
        assert shell.errors == 1 and shell.last_outcome.ok
        shell.run_line("mode wrong")
        shell.run_line("check d=x")
        assert shell.errors == 3

    def test_failure_report_counts_as_error(self) -> None:
        shell = Shell(commands=True)
        shell.run_line(EXAMPLE)
        shell.run_line("check against sum_{k=0}^{d-1} |k>^{n} |k, k, k>")
        assert shell.errors == 1 and shell.last_outcome.command == "check"

    def test_non_str_line(self) -> None:
        shell = Shell(commands=True)
        assert shell.run_line(3) == "error: a line must be a str, got int"  # type: ignore[arg-type]
        assert shell.errors == 1 and shell.history == []

    def test_internal_error_is_reported_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(session: C.Session, line: str) -> C.Outcome:
            raise RuntimeError("boom\n  here")

        monkeypatch.setattr(S, "run", broken)
        shell = Shell(commands=True)
        assert shell.run_line("help") == "error: internal: RuntimeError: boom here"
        assert shell.errors == 1

    def test_keyboard_interrupt_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def interrupted(session: C.Session, line: str) -> C.Outcome:
            raise KeyboardInterrupt

        monkeypatch.setattr(S, "run", interrupted)
        with pytest.raises(KeyboardInterrupt):
            Shell(commands=True).run_line("help")

    def test_unicode_detail(self) -> None:
        shell = Shell(commands=True, unicode=True)
        shell.run_line(EXAMPLE)
        out = shell.run_line("detail")
        assert "graph:" in out and "Σ" in out
        plain = Shell(commands=True)
        plain.run_line(EXAMPLE)
        assert "Σ" not in plain.run_line("detail")


# -- scripts -------------------------------------------------------------------------------


class TestScripts:
    def test_run_script_skips_blank_and_comment_lines(self) -> None:
        shell = Shell(commands=True)
        outs = shell.run_script(["# header", "", "mode", "   ", "mode zx"])
        assert outs == ["mode: dirac", "mode: zx"]
        assert shell.history == ["mode", "mode zx"]

    def test_str_input_is_split(self) -> None:
        assert Shell(commands=True).run_script("mode\n\n# c\nmode zx\n") == [
            "mode: dirac",
            "mode: zx",
        ]

    def test_continuation_joins_lines(self) -> None:
        shell = Shell(commands=True)
        outs = shell.run_script(["sum_{k=0}^{3-1} \\", "  |k>^{30}; \\", "copy", "mode"])
        assert len(outs) == 2
        assert shell.history[0] == EXAMPLE
        assert shell.session.current is not None and shell.errors == 0

    def test_trailing_continuation_at_end_of_input(self) -> None:
        shell = Shell(commands=True)
        assert shell.run_script(["mode \\", "zx \\"]) == ["mode: zx"]

    def test_comment_line_never_continues(self) -> None:
        shell = Shell(commands=True)
        assert shell.run_script(["# note \\", "mode"]) == ["mode: dirac"]

    def test_stops_at_quit(self) -> None:
        shell = Shell(commands=True)
        outs = shell.run_script(["mode", "quit", "mode zx", "help"])
        assert outs == ["mode: dirac", ""]
        assert shell.done and shell.mode is NotationMode.DIRAC
        assert shell.history == ["mode", "quit"]

    def test_iter_script_yields_line_output_pairs(self) -> None:
        pairs = list(Shell(commands=True).iter_script("mode\nmode \\\n zx\n"))
        assert pairs == [("mode", "mode: dirac"), ("mode zx", "mode: zx")]

    def test_errors_do_not_stop_a_script(self) -> None:
        shell = Shell(commands=True)
        outs = shell.run_script(["apply nosuchrule", EXAMPLE, "apply spider_fusion"])
        assert outs[0].startswith("error: ")
        assert "GHZ_{n+2}" in outs[2]
        assert shell.errors == 1

    def test_crlf_lines(self) -> None:
        assert Shell(commands=True).run_script(["mode \\\r\n", "zx\r\n"]) == ["mode: zx"]


# -- loop ----------------------------------------------------------------------------------


class TestLoop:
    def test_non_tty_has_no_prompt_and_eof_returns_zero(self) -> None:
        stdout = io.StringIO()
        shell = Shell(commands=True)
        assert shell.loop(io.StringIO("mode\n\n# c\nmode zx\n"), stdout) == 0
        assert stdout.getvalue() == "mode: dirac\nmode: zx\n"
        assert PROMPT not in stdout.getvalue()
        assert not shell.done

    def test_quit_stops_reading(self) -> None:
        stdout = io.StringIO()
        shell = Shell(commands=True)
        assert shell.loop(io.StringIO("mode\nquit\nmode zx\n"), stdout) == 0
        assert stdout.getvalue() == "mode: dirac\n"
        assert shell.done and shell.mode is NotationMode.DIRAC

    def test_continuation_lines(self) -> None:
        stdout = io.StringIO()
        shell = Shell(commands=True)
        shell.loop(io.StringIO("sum_{k=0}^{3-1} \\\n|k>^{30}; copy\nmode \\\n"), stdout)
        assert stdout.getvalue().startswith("loaded dirac input")
        assert stdout.getvalue().endswith("mode: dirac\n")
        assert shell.history == [EXAMPLE, "mode"]

    def test_empty_input(self) -> None:
        stdout = io.StringIO()
        assert Shell(commands=True).loop(io.StringIO(""), stdout) == 0
        assert stdout.getvalue() == ""

    def test_keyboard_interrupt_while_reading(self) -> None:
        stdout = io.StringIO()
        stdin = FakeStdin("mode\n", [KeyboardInterrupt()])
        assert Shell(commands=True).loop(stdin, stdout) == 0
        assert stdout.getvalue() == "interrupted\nmode: dirac\n"

    def test_keyboard_interrupt_while_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def interrupted(session: C.Session, line: str) -> C.Outcome:
            calls.append(line)
            raise KeyboardInterrupt

        monkeypatch.setattr(S, "run", interrupted)
        stdout = io.StringIO()
        shell = Shell(commands=True)
        assert shell.loop(io.StringIO("help\nmode\n"), stdout) == 0
        assert stdout.getvalue() == "interrupted\nmode: dirac\n"
        assert calls == ["help"]

    def test_tty_shows_banner_and_prompts(self) -> None:
        stdout = io.StringIO()
        stdin = FakeStdin("mode \\\nzx\n", [], tty=True)
        assert Shell(commands=True).loop(stdin, stdout) == 0
        text = stdout.getvalue()
        assert text.startswith("ArchytasZX shell, mode dirac; type help for commands.\n")
        name = S._place()
        assert f"{name}(1): " in text and f"{'.' * len(name)}(1): " in text
        assert "mode: zx\n" in text
        assert text.endswith(f"{name}(2): \n")

    def test_tty_interrupt_message(self) -> None:
        stdout = io.StringIO()
        stdin = FakeStdin("", [KeyboardInterrupt()], tty=True)
        assert Shell(commands=True).loop(stdin, stdout) == 0
        assert "\ninterrupted\n" in stdout.getvalue()

    def test_unreadable_input_returns_one(self) -> None:
        stdout = io.StringIO()
        shell = Shell(commands=True)
        assert shell.loop(FakeStdin("", [OSError("gone")]), stdout) == 1
        assert stdout.getvalue() == "error: cannot read input: OSError: gone\n"
        assert shell.errors == 1

    def test_unwritable_output_returns_one(self) -> None:
        assert Shell(commands=True).loop(io.StringIO("mode\n"), BrokenStdout()) == 1

    def test_errors_continue_the_loop(self) -> None:
        stdout = io.StringIO()
        shell = Shell(commands=True)
        assert shell.loop(io.StringIO("apply x\nmode\n"), stdout) == 0
        lines = stdout.getvalue().splitlines()
        assert lines[0].startswith("error: ") and lines[-1] == "mode: dirac"
        assert shell.errors == 1


# -- main ----------------------------------------------------------------------------------


class TestMain:
    def test_repeated_c(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "-c", EXAMPLE, "-c", "apply spider_fusion", "-c", "induct"]) == 0
        out = capsys.readouterr().out
        assert "GHZ_{n+2}" in out and "induct: proved" in out
        assert not listing(out)

    def test_mode_zx(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "--mode", "zx", "-c", EXAMPLE]) == 0
        assert listing(capsys.readouterr().out)

    def test_bad_mode_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "--mode", "latex", "-c", "help"]) == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_echo(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "--echo", "-c", "mode", "-c", "# skipped"]) == 0
        assert capsys.readouterr().out == f"{PROMPT}mode\nmode: dirac\n"

    def test_unicode(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "--unicode", "-c", EXAMPLE]) == 0
        assert "Σ" in capsys.readouterr().out

    def test_script_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        script = tmp_path / "example.azx"
        script.write_text(
            f"# worked example\n{EXAMPLE}\napply \\\n  spider_fusion\ncheck\n", encoding="utf-8"
        )
        assert main(["--commands", str(script)]) == 0
        out = capsys.readouterr().out
        assert "check: no mismatch" in out and "GHZ_{n+2}" in out

    def test_c_then_scripts_share_one_session(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = tmp_path / "a.azx"
        second = tmp_path / "b.azx"
        first.write_text("apply spider_fusion\n", encoding="utf-8")
        second.write_text("check\n", encoding="utf-8")
        assert main(["--commands", "-c", EXAMPLE, str(first), str(second)]) == 0
        out = capsys.readouterr().out
        assert out.index("loaded") < out.index("spider_fusion") < out.index("check:")

    def test_quit_stops_later_scripts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = tmp_path / "later.azx"
        script.write_text("mode zx\n", encoding="utf-8")
        assert (
            main(["--commands", "-c", "mode", "-c", "quit", "-c", "apply nosuchrule", str(script)])
            == 0
        )
        assert capsys.readouterr().out == "mode: dirac\n"

    def test_missing_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        missing = tmp_path / "nope.azx"
        assert main(["--commands", "-c", "mode", str(missing)]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert f"archytaszx: cannot read {missing}" in captured.err

    def test_error_exits_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "-c", "apply nosuchrule", "-c", "mode"]) == 1
        out = capsys.readouterr().out
        assert out.startswith("error: ") and out.endswith("mode: dirac\n")
        assert "Traceback" not in out

    def test_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "--help"]) == 0
        out = capsys.readouterr().out
        assert "usage: archytaszx" in out and "--mode" in out

    def test_interrupt_returns_130(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def interrupted(session: C.Session, line: str) -> C.Outcome:
            raise KeyboardInterrupt

        monkeypatch.setattr(S, "run", interrupted)
        assert main(["--commands", "-c", "help", "-c", "mode"]) == 130
        captured = capsys.readouterr()
        assert captured.err == "interrupted\n" and captured.out == ""

    def test_interactive_without_commands(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("mode\napply nosuchrule\nquit\nmode\n"))
        assert main(["--commands"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("mode: dirac\nerror: ")
        assert out.count("mode: dirac") == 1


def test_module_runs_as_a_script() -> None:
    ok = subprocess.run(
        [
            sys.executable,
            "-m",
            "archytaszx.repl.shell",
            "--commands",
            "-c",
            EXAMPLE,
            "-c",
            "apply spider_fusion",
            "-c",
            "induct",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert "GHZ_{n+2}" in ok.stdout and "induct: proved" in ok.stdout
    assert ok.stderr == ""
    failed = subprocess.run(
        [sys.executable, "-m", "archytaszx.repl.shell", "--commands", "-c", "apply nosuchrule"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert failed.returncode == 1
    assert failed.stdout.startswith("error: ") and "Traceback" not in failed.stderr


class TestOutputMode:
    def test_echoes_the_last_statement(self) -> None:
        shell = Shell(display="ascii", output=True)
        assert shell.run_line("var n") == "val n : nat"
        assert shell.run_line("var a = 1; var b = 2") == "val b : nat = 2"
        assert shell.run_line("|p> = X|0>") == "val |p> : ket = |1>"
        assert shell.run_line("<p|p>") == "- : nat = 1"
        assert shell.run_line("seed 1") == ""

    def test_toggle(self) -> None:
        shell = Shell(display="ascii")
        assert shell.run_line("var n") == ""
        assert shell.run_line(":output on") == "output: on"
        assert shell.run_line("var m") == "val m : nat"
        assert shell.run_line(":output") == "output: on"
        assert shell.run_line(":output off") == "output: off"
        assert shell.run_line(":output maybe") == "error: usage: output [on|off]"

    def test_show_still_prints_with_the_echo_off(self) -> None:
        shell = Shell(display="ascii", output=True)
        assert shell.run_line("show(1 + 1)") == "1 + 1 = 2"

    def test_mode_drives_the_echo(self) -> None:
        shell = Shell(display="ascii", output=True)
        shell.run_line(":mode zx")
        out = shell.run_line("|p> = |1>").splitlines()
        assert out[0] == "val |p> : ket =" and out[1].startswith("  graph:")

    def test_colour_codes(self) -> None:
        shell = Shell(display="ascii", output=True, color=True)
        assert shell.run_line("var n") == "\x1b[1mval\x1b[0m n : \x1b[36mnat\x1b[0m"

    def test_bar_shape(self) -> None:
        bar = Shell().bar(60)
        assert len(bar) == 60 and bar.startswith("─( ") and "< command 1 >" in bar
        assert bar.endswith(f"{{ {S._place()} }}─")

    def test_tty_prints_the_bar_before_each_prompt(self) -> None:
        stdout = io.StringIO()
        Shell(output=True, display="ascii").loop(FakeStdin("var n\n", [], tty=True), stdout)
        text = stdout.getvalue()
        assert text.count("< command 1 >") == 1 and text.count("< command 2 >") == 1
        assert "val n : nat\n" in text


class TestPrompt:
    def test_counts_lines(self) -> None:
        shell = Shell(display="ascii")
        name = S._place()
        assert shell.prompt() == f"{name}(1): "
        shell.run_line("var n")
        shell.run_line("")
        assert shell.prompt() == f"{name}(2): "
        assert shell.prompt(True) == f"{'.' * len(name)}(2): "

    def test_place(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.chdir(tmp_path)
        assert S._place() == tmp_path.name
        monkeypatch.setenv("HOME", str(tmp_path))
        assert S._place() == "home"


class TestOracleFlag:
    def test_interactive_refused(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--oracle"]) == 2
        assert "scripts" in capsys.readouterr().err

    def test_script_checks_gates(self, capsys: pytest.CaptureFixture[str]) -> None:
        source = ["gate B = |x> -> |0>", "|a> = B|1>"]
        assert main(["--oracle", *(arg for line in source for arg in ("-c", line))]) == 1
        assert "not reversible" in capsys.readouterr().out
        assert main([*(arg for line in source for arg in ("-c", line))]) == 0
