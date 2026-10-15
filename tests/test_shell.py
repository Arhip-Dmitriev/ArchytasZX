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
from archytaszx.repl.shell import Shell, main

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
        [{"dirac": "zx"}, {"unicode": 1}, {"session": object()}],
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
        shell.run_line("dirac")
        shell.run_line("apply nosuchrule")
        assert shell.history == ["help", "dirac", "apply nosuchrule"]

    def test_history_listing(self) -> None:
        shell = Shell(commands=True)
        for line in ("help", "dirac", "", "# no", "rules"):
            shell.run_line(line)
        assert shell.run_line("history") == "1  help\n2  dirac\n3  rules\n4  history"

    def test_history_listing_widths(self) -> None:
        shell = Shell(commands=True)
        for _ in range(9):
            shell.run_line("dirac")
        lines = shell.run_line("history").splitlines()
        assert lines[0] == " 1  dirac" and lines[-1] == "10  history"

    def test_mode_get_and_set(self) -> None:
        shell = Shell(commands=True)
        assert shell.run_line("dirac") == "mode: dirac"
        assert shell.run_line("zx") == "mode: zx"
        assert shell.mode is NotationMode.ZX
        assert shell.run_line("BOTH") == "mode: both"
        assert shell.mode is NotationMode.BOTH
        assert shell.run_line("Dirac") == "mode: dirac"
        assert shell.mode is NotationMode.DIRAC
        assert shell.errors == 0

    def test_mode_bad_value(self) -> None:
        shell = Shell(commands=True, mode=NotationMode.ZX)
        assert shell.run_line("dirac now").startswith("error: ")
        assert shell.mode is NotationMode.ZX and shell.errors == 1

    def test_mode_drives_rendering(self) -> None:
        shell = Shell(commands=True)
        dirac = shell.run_line(EXAMPLE)
        shell.run_line("zx")
        zx = shell.run_line("show")
        assert not listing(dirac) and not listing(zx) and zx.startswith("scalar:")
        shell.run_line("raw")
        assert listing(shell.run_line("show"))
        shell.run_line("listing")
        shell.run_line("zx")
        both = shell.run_line("show")
        assert both.startswith("scalar:") and listing(both)

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
        for usage in (
            "zx -- ZX",
            "verbose_on --",
            "config [KEY",
            "history --",
            "quit --",
            "apply RULE",
            "induct",
        ):
            assert usage in out
        assert "aliases: list-rules = rules" in out

    def test_help_for_one_command(self) -> None:
        assert Shell(commands=True).run_line("help zx").startswith("zx -- ")

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
        shell.run_line("zx wrong")
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
        outs = shell.run_script(["# header", "", "dirac", "   ", "zx"])
        assert outs == ["mode: dirac", "mode: zx"]
        assert shell.history == ["dirac", "zx"]

    def test_str_input_is_split(self) -> None:
        assert Shell(commands=True).run_script("dirac\n\n# c\nzx\n") == [
            "mode: dirac",
            "mode: zx",
        ]

    def test_continuation_joins_lines(self) -> None:
        shell = Shell(commands=True)
        outs = shell.run_script(["sum_{k=0}^{3-1} \\", "  |k>^{30}; \\", "copy", "dirac"])
        assert len(outs) == 2
        assert shell.history[0] == EXAMPLE
        assert shell.session.current is not None and shell.errors == 0

    def test_trailing_continuation_at_end_of_input(self) -> None:
        shell = Shell(commands=True)
        assert shell.run_script(["help \\", "zx \\"]) == ["zx -- ZX diagrams drawn with matplotlib (alias of zxmatplot)"]

    def test_comment_line_never_continues(self) -> None:
        shell = Shell(commands=True)
        assert shell.run_script(["# note \\", "dirac"]) == ["mode: dirac"]

    def test_stops_at_quit(self) -> None:
        shell = Shell(commands=True)
        outs = shell.run_script(["dirac", "quit", "zx", "help"])
        assert outs == ["mode: dirac", ""]
        assert shell.done and shell.mode is NotationMode.DIRAC
        assert shell.history == ["dirac", "quit"]

    def test_iter_script_yields_line_output_pairs(self) -> None:
        pairs = list(Shell(commands=True).iter_script("dirac\nhelp \\\n zx\n"))
        assert pairs == [("dirac", "mode: dirac"), ("help zx", "zx -- ZX diagrams drawn with matplotlib (alias of zxmatplot)")]

    def test_errors_do_not_stop_a_script(self) -> None:
        shell = Shell(commands=True)
        outs = shell.run_script(["apply nosuchrule", EXAMPLE, "apply spider_fusion"])
        assert outs[0].startswith("error: ")
        assert "GHZ_{n+2}" in outs[2]
        assert shell.errors == 1

    def test_crlf_lines(self) -> None:
        assert Shell(commands=True).run_script(["help \\\r\n", "zx\r\n"]) == [
            "zx -- ZX diagrams drawn with matplotlib (alias of zxmatplot)"
        ]


# -- loop ----------------------------------------------------------------------------------


class TestLoop:
    def test_non_tty_has_no_prompt_and_eof_returns_zero(self) -> None:
        stdout = io.StringIO()
        shell = Shell(commands=True)
        assert shell.loop(io.StringIO("dirac\n\n# c\nzx\n"), stdout) == 0
        assert stdout.getvalue() == "mode: dirac\nmode: zx\n"
        assert "(1): " not in stdout.getvalue()
        assert not shell.done

    def test_quit_stops_reading(self) -> None:
        stdout = io.StringIO()
        shell = Shell(commands=True)
        assert shell.loop(io.StringIO("dirac\nquit\nzx\n"), stdout) == 0
        assert stdout.getvalue() == "mode: dirac\n"
        assert shell.done and shell.mode is NotationMode.DIRAC

    def test_continuation_lines(self) -> None:
        stdout = io.StringIO()
        shell = Shell(commands=True)
        shell.loop(io.StringIO("sum_{k=0}^{3-1} \\\n|k>^{30}; copy\ndirac \\\n"), stdout)
        assert stdout.getvalue().startswith("loaded dirac input")
        assert stdout.getvalue().endswith("mode: dirac\n")
        assert shell.history == [EXAMPLE, "dirac"]

    def test_empty_input(self) -> None:
        stdout = io.StringIO()
        assert Shell(commands=True).loop(io.StringIO(""), stdout) == 0
        assert stdout.getvalue() == ""

    def test_keyboard_interrupt_while_reading(self) -> None:
        stdout = io.StringIO()
        stdin = FakeStdin("dirac\n", [KeyboardInterrupt()])
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
        assert shell.loop(io.StringIO("help\ndirac\n"), stdout) == 0
        assert stdout.getvalue() == "interrupted\nmode: dirac\n"
        assert calls == ["help"]

    def test_tty_shows_banner_and_prompts(self) -> None:
        stdout = io.StringIO()
        stdin = FakeStdin("help \\\nzx\n", [], tty=True)
        assert Shell(commands=True).loop(stdin, stdout) == 0
        text = stdout.getvalue()
        assert text.startswith("ArchytasZX shell, mode dirac; type help for commands.\n")
        name = S._place()
        assert f"{name}(1): " in text and f"{'.' * len(name)}(1): " in text
        assert "zx -- ZX diagrams drawn with matplotlib (alias of zxmatplot)\n" in text
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
        assert Shell(commands=True).loop(io.StringIO("dirac\n"), BrokenStdout()) == 1

    def test_errors_continue_the_loop(self) -> None:
        stdout = io.StringIO()
        shell = Shell(commands=True)
        assert shell.loop(io.StringIO("apply x\ndirac\n"), stdout) == 0
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
        assert main(["--commands", "--zx", "-c", EXAMPLE]) == 0
        drawn = capsys.readouterr().out
        assert "scalar:" in drawn and not listing(drawn)
        assert main(["--commands", "-zxmatplot", "-c", EXAMPLE]) == 0
        assert capsys.readouterr().out == drawn
        assert main(["--commands", "--raw", "-c", EXAMPLE]) == 0
        assert listing(capsys.readouterr().out)

    def test_two_modes_are_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "--zx", "--dirac", "-c", "help"]) == 2
        assert "not allowed with" in capsys.readouterr().err

    @pytest.mark.parametrize("flag", ["--echo", "--mode", "--output", "--oracle"])
    def test_removed_flags(self, flag: str) -> None:
        assert main(["--commands", flag, "-c", "help"]) == 2

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
        script.write_text("zx\n", encoding="utf-8")
        assert (
            main(["--commands", "-c", "dirac", "-c", "quit", "-c", "apply nosuchrule", str(script)])
            == 0
        )
        assert capsys.readouterr().out == "mode: dirac\n"

    def test_missing_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        missing = tmp_path / "nope.azx"
        assert main(["--commands", "-c", "dirac", str(missing)]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert f"archytaszx: cannot read {missing}" in captured.err

    def test_error_exits_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "-c", "apply nosuchrule", "-c", "dirac"]) == 1
        out = capsys.readouterr().out
        assert out.startswith("error: ") and out.endswith("mode: dirac\n")
        assert "Traceback" not in out

    def test_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--commands", "--help"]) == 0
        out = capsys.readouterr().out
        assert "usage: archytaszx" in out and "--zx" in out and "--verbose" in out

    def test_interrupt_returns_130(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def interrupted(session: C.Session, line: str) -> C.Outcome:
            raise KeyboardInterrupt

        monkeypatch.setattr(S, "run", interrupted)
        assert main(["--commands", "-c", "help", "-c", "dirac"]) == 130
        captured = capsys.readouterr()
        assert captured.err == "interrupted\n" and captured.out == ""

    def test_interactive_without_commands(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("dirac\napply nosuchrule\nquit\ndirac\n"))
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


class TestVerboseMode:
    def test_echoes_the_last_statement(self) -> None:
        shell = Shell(display="ascii", verbose=True)
        assert shell.run_line("var n") == "val n : nat"
        assert shell.run_line("var a = 1; var b = 2") == "val b : nat = 2"
        assert shell.run_line("|p> = X|0>") == "val |p> : ket = |1>"
        assert shell.run_line("<p|p>") == "- : nat = 1"
        assert shell.run_line("seed 1") == ""

    def test_toggle(self) -> None:
        shell = Shell(display="ascii")
        assert shell.run_line("var n") == ""
        assert shell.run_line(":verbose_on") == "verbose: on"
        assert shell.run_line("var m") == "val m : nat"
        assert shell.run_line(":verbose_off") == "verbose: off"
        assert shell.run_line("var k") == ""

    def test_show_still_prints_with_the_echo_off(self) -> None:
        shell = Shell(display="ascii", verbose=True)
        assert shell.run_line("show(1 + 1)") == "1 + 1 = 2"

    def test_mode_drives_the_echo(self) -> None:
        shell = Shell(display="ascii", verbose=True)
        shell.run_line(":zx")
        out = shell.run_line("|p> = |1>").splitlines()
        assert out[0] == "val |p> : ket =" and out[1].startswith("  graph:")

    def test_colour_codes(self) -> None:
        shell = Shell(display="ascii", verbose=True, color=True)
        assert shell.run_line("var n") == "\x1b[1mval\x1b[0m n : \x1b[36mnat\x1b[0m"

    def test_bar_shape(self) -> None:
        bar = Shell().bar(60)
        assert len(bar) == 60 and bar.startswith("─( ") and "< command 1 >" in bar
        assert bar.endswith(f"{{ {S._place()} }}─")

    def test_tty_prints_the_bar_before_each_prompt(self) -> None:
        stdout = io.StringIO()
        Shell(verbose=True, display="ascii").loop(FakeStdin("var n\n", [], tty=True), stdout)
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


class TestElse:
    def test_else_on_the_next_line(self) -> None:
        shell = Shell(display="ascii")
        lines = ["var t = 0", "if t == 1 { print(1) }", "else { print(2) }"]
        assert shell.run_script(lines) == ["", "", "2"]

    def test_blank_lines_before_else_in_a_script(self) -> None:
        shell = Shell(display="ascii")
        assert shell.run_script("if 1 == 0 { print(1) }\n\n# c\nelse { print(2) }") == ["", "2"]

    def test_a_closed_if_runs_at_the_next_other_line(self) -> None:
        shell = Shell(display="ascii")
        assert shell.run_line("if 1 == 1 { print(1) }") == ""
        assert shell.run_line("print(2)") == "1\n2"

    def test_a_blank_line_runs_a_closed_if(self) -> None:
        shell = Shell(display="ascii")
        assert shell.run_line("if 1 == 1 { print(1) }") == ""
        assert shell.run_line("") == "1"

    def test_a_closed_if_runs_at_the_end_of_a_script(self) -> None:
        assert Shell(display="ascii").run_script(["if 1 == 1 { print(1) }"]) == ["", "1"]

    def test_header_brace_on_the_next_line(self) -> None:
        assert Shell(display="ascii").run_script(["if 1 == 1", "{ print(1) }"]) == ["", "", "1"]


class TestConfig:
    def test_show_set_and_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(S.D, "CONFIG", S.D.Config())
        shell = Shell(display="ascii")
        assert shell.run_line(":config").startswith("size = 1\ncolor = auto\nfile: ")
        assert shell.run_line(":config size 1.5").startswith("size = 1.5\n")
        assert shell.run_line(":config color #ff8800").splitlines()[1] == "color = #ff8800"
        assert shell.run_line(":config size huge").startswith("error: size must be a number")
        assert shell.run_line(":config shade red").startswith("error: unknown setting")
        assert shell.run_line(":config reset").startswith("size = 1\ncolor = auto")
        assert shell.errors == 2

    def test_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(S.D, "CONFIG", S.D.Config())
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        shell = Shell(display="ascii")
        shell.run_line(":config size 2")
        assert shell.run_line(":config save") == f"saved {tmp_path}/archytaszx/config.toml"
        loaded = S.D.load_config()
        assert loaded.size == 2.0 and loaded.color == "auto"

    def test_bad_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text("size = -1\n", encoding="utf-8")
        with pytest.raises(ValueError):
            S.D.load_config(str(path))
        assert S.D.load_config(str(tmp_path / "none.toml")) == S.D.Config()
