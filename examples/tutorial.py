"""Interactive terminal course on qufinite ZXW with !-boxes, every example run live on the engine.

python examples/tutorial.py                 start at chapter 1
python examples/tutorial.py --chapter 9     start at chapter 9
python examples/tutorial.py --menu          pick a chapter
python examples/tutorial.py --no-pause      print every chapter without stopping
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT))

try:
    from archytaszx import Program
    from archytaszx.diagram.bangbox import instantiate_symbol, peel_one
except ImportError as missing:
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.exists() and Path(sys.prefix).resolve() != venv.parent.parent.resolve():
        os.execv(str(venv), [str(venv), __file__, *sys.argv[1:]])
    sys.exit(f"{missing}: install the engine first, e.g. pip install -e {ROOT}")

INTERACTIVE = True
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
WIDTH = min(shutil.get_terminal_size((96, 24)).columns, 96) - 2


class Quit(Exception):
    """Leaving the course."""


class Menu(Exception):
    """Returning to the chapter menu."""


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def heading(number: int, title: str) -> None:
    """A chapter banner."""
    bar = "═" * WIDTH
    print("\n" + _c("1;36", bar))
    print(_c("1;36", f"  Chapter {number}. {title}"))
    print(_c("1;36", bar))


def section(title: str) -> None:
    """A sub-heading."""
    print("\n" + _c("1;33", f"── {title} " + "─" * max(0, WIDTH - len(title) - 4)))


def _dedent(text: str) -> str:
    lines = text.strip("\n").rstrip().split("\n")
    base = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[base:] if line[:base].isspace() else line.lstrip() for line in lines)


def say(text: str) -> None:
    """Prose wrapped per paragraph; a paragraph indented past the first line is kept as it is."""
    text = _dedent(text)
    for block in text.split("\n\n"):
        lines = block.split("\n")
        if all(line.startswith("  ") for line in lines):
            print("\n".join(_c("35", line) for line in lines))
        else:
            print(textwrap.fill(" ".join(line.strip() for line in lines), WIDTH))
        print()


def tip(text: str) -> None:
    """A highlighted remark."""
    body = textwrap.fill(" ".join(textwrap.dedent(text).split()), WIDTH - 4)
    print(_c("32", textwrap.indent(body, "  ▸ ", lambda _: True)))
    print()


class Lesson:
    """One chapter's engine session and Python namespace."""

    def __init__(self) -> None:
        self.P = Program()
        self.ns: dict[str, object] = {
            "P": self.P,
            "n": self.P.count("n"),
            "m": self.P.count("m"),
            "d": self.P.d,
            "instantiate_symbol": instantiate_symbol,
            "peel_one": peel_one,
            "show": self.show,
            "show_graph": self.show_graph,
        }

    def show(self, piece: object, drawing: bool = True) -> None:
        """Prints a piece's Dirac form, then its ZX drawing."""
        print(_c("2", "Dirac:"), str(piece))
        if drawing:
            print(_c("2", "ZX:"))
            print(piece.draw())  # type: ignore[attr-defined]

    def show_graph(self, diagram: object) -> None:
        """Prints a Diagram's Dirac form through the session."""
        text = self.P.load(diagram).text
        print("\n".join(line for line in text.splitlines() if not line.startswith("loaded")))

    def py(self, source: str) -> None:
        """Shows Python source, then runs it in the lesson namespace."""
        source = textwrap.dedent(source).strip("\n")
        print(_c("2", "python:"))
        print(_c("33", textwrap.indent(source, "  >>> ", lambda _: True)))
        try:
            exec(compile(source, "<tutorial>", "exec"), self.ns)  # noqa: S102
        except Exception:  # noqa: BLE001
            print(_c("31", traceback.format_exc(limit=1)))
        print()

    def run(self, command: str) -> None:
        """Shows an engine command, then runs it in the lesson session."""
        print(_c("2", "engine:"), _c("33", command))
        try:
            print(self.P(command))
        except Exception as exc:  # noqa: BLE001
            print(_c("31", f"error: {exc}"))
        print()

    def pause(self) -> None:
        """Waits for Enter; other input is run as an engine command or, after py, as Python."""
        if not INTERACTIVE:
            return
        hint = _c("2", "[Enter] next · type a command or py CODE to experiment · m menu · q quit")
        while True:
            try:
                line = input(hint + "\n" + _c("1;32", "› ")).strip()
            except EOFError:
                raise Quit from None
            if not line:
                return
            if line in ("q", "quit", "exit"):
                raise Quit
            if line in ("m", "menu"):
                raise Menu
            if line.startswith("py "):
                self.py(line[3:])
            else:
                self.run(line)

    def quiz(self, question: str, options: list[str], answer: int, why: str) -> None:
        """A multiple-choice question; the reader retries until right or types s to see it."""
        letters = "abcd"
        print(_c("1;34", "Quiz: ") + textwrap.fill(question, WIDTH - 6, subsequent_indent="      "))
        for letter, option in zip(letters, options):
            print(f"   {letter}) {option}")
        if INTERACTIVE:
            while True:
                try:
                    reply = input(_c("1;34", "your answer (a-d, s to show): ")).strip().lower()
                except EOFError:
                    raise Quit from None
                if reply == letters[answer]:
                    print(_c("32", "Correct. ") + textwrap.fill(why, WIDTH) + "\n")
                    return
                if reply == "s":
                    break
                if reply in letters[: len(options)]:
                    print(_c("31", "Not quite; try again."))
                else:
                    print(_c("2", "type one letter, a to d, or s to see the answer"))
        print(textwrap.fill(_c("32", f"Answer: {letters[answer]}) ") + why, WIDTH) + "\n")


def ch_welcome(L: Lesson) -> None:
    heading(1, "Welcome")
    say("""
        This course teaches the qufinite ZXW calculus from nothing: what the pictures mean,
        how to compute with them, and how !-boxes ("bang boxes") let one picture stand for a
        whole infinite family of pictures, one for each number of qudits. Every example on
        the screen is computed live by the ArchytasZX engine, so what you see is what the
        engine really does.

        You need no physics and no linear algebra beyond the idea of a list of numbers. Each
        new word is explained when it first appears.
    """)
    section("How to use this course")
    say("""
        The course stops after each step. Press Enter to go on. At any stop you can also
        experiment:

          load Z(0 -> 3)                 any engine command, run in this chapter's session
          py show(P.Z(0, 2))             any Python line, with P the engine and n, m, d symbols
          help                           the engine's list of commands
          m                              the chapter menu
          q                              quit

        Nothing you type can break the course: an engine error is printed and the course
        waits for the next line.
    """)
    L.pause()


def ch_qudits(L: Lesson) -> None:
    heading(2, "Qudits, kets and the notation the engine prints")
    say("""
        A qubit is a quantum system with two basic states, written |0⟩ and |1⟩. A qudit is
        the same idea with d basic states |0⟩, |1⟩, ..., |d-1⟩, where d is called the
        dimension. A qubit is a qudit with d = 2.

        A general state of one qudit is a list of d complex numbers, one weight per basic
        state, written as a sum:  a_0 |0⟩ + a_1 |1⟩ + ... + a_(d-1) |d-1⟩.  The symbol |k⟩
        (read "ket k") is just the basic state number k.

        Several qudits together have basic states written |k_1, k_2, ...⟩, one digit per
        qudit. Putting systems side by side is called the tensor product, written ⊗.
    """)
    L.pause()
    section("d stays a symbol")
    say("""
        Most tools pick a number for d. ArchytasZX never does: d stays a letter, so every
        answer holds for qubits, qutrits and every other dimension at once. Here is the
        simplest state, the equal superposition of all d basic states, which the engine
        calls "plus":
    """)
    L.py("show(P.Z(0, 1))")
    say("""
        Read the line after "Dirac:" as: the sum, for k running from 0 to d-1, of |k⟩. The
        engine always prints in this style, called Dirac notation. Three more pieces of
        notation appear everywhere:

          ⟨k|          "bra k", the input side of a map: |j⟩⟨k| sends |k⟩ to |j⟩
          [k0 = k1]    1 when the condition holds and 0 otherwise
          ω_{d}        the complex number e^(2πi/d), a d-th root of unity
    """)
    L.py("show(P.id)")
    say("""
        The identity wire is the sum of |k⟩⟨k|: it sends each |k⟩ to itself. The drawing
        under "ZX:" is the same thing as a picture, which is what the rest of the course is
        about.
    """)
    L.quiz(
        "A qudit has dimension d = 5. How many basic states does it have?",
        ["4", "5", "10", "25"],
        1,
        "|0⟩, |1⟩, |2⟩, |3⟩ and |4⟩: d basic states, numbered from 0.",
    )
    L.pause()


def ch_diagrams(L: Lesson) -> None:
    heading(3, "Diagrams: wires and boxes")
    say("""
        A ZX diagram is a picture of a computation. Each wire carries one qudit. Each node
        (a dot or a box) is an operation. Wires that end on the left are inputs, wires that
        leave on the right are outputs. A diagram with no inputs is a state (it prepares
        something), one with no outputs is an effect (it tests something), and one with
        neither is a plain number, called a scalar.

        The engine reads a picture as an exact linear map: you can always ask for its Dirac
        form, and two pictures are equal when their maps are equal, scalar included.
    """)
    L.pause()
    section("Building diagrams in Python")
    say("""
        In Python you build a diagram from pieces with two operators. A | B means "do B,
        then A": B's outputs are plugged into A's inputs (read it like function
        composition, A(B)). A @ B means "A and B side by side". Here F is the Fourier box,
        which we meet properly in chapter 5.
    """)
    L.py("show(P.F | P.Z(0, 1))           # first the plus state, then F")
    L.py("show(P.Z(0, 1) @ P.Z(0, 1))     # two plus states side by side")
    say("""
        The second result is |k0, k1⟩ summed over both indices: two independent qudits.
        A ^ K puts K copies in series and A * s multiplies by a number:
    """)
    L.py("show(P.F ^ 2, drawing=False)\nshow(P.Z(0, 1) * d ** '-1/2', drawing=False)")
    tip("""
        The same diagrams can be typed in the engine's own language (the DSL). Every piece
        prints its DSL with .dsl(); try   py print((P.F | P.Z(0, 1)).dsl())   at the prompt.
    """)
    L.quiz(
        "In Python, which expression first applies P.T and then P.F?",
        ["P.T | P.F", "P.F | P.T", "P.T @ P.F", "P.F @ P.T"],
        1,
        "A | B runs B first, so P.F | P.T applies T, then F.",
    )
    L.pause()


def ch_z(L: Lesson) -> None:
    heading(4, "The Z spider")
    say("""
        The first node is the Z spider, drawn Ⓩ. It may have any number of inputs and
        outputs, and it says one thing: every leg carries the same value. As a formula,

          Z with m inputs and n outputs  =  Σ_k |k, k, ..., k⟩⟨k, k, ..., k|

        with n k's in the ket and m in the bra. Different leg counts give familiar things:
    """)
    L.py("show(P.Z(1, 2))      # copy: |k⟩ goes to |k, k⟩")
    L.py("show(P.Z(0, 3))      # the GHZ state Σ_k |k, k, k⟩")
    L.pause()
    section("Phases")
    say("""
        A spider can carry a phase: a weight e^(2πi α_k) on each value k. Phases are
        measured in turns, so α = 1/2 is half a turn, the factor -1. Index 0 always has
        phase 0. Here the value 1 gets the phase alpha, and alpha stays a symbol:
    """)
    L.py("show(P.Z(0, 1, phase={1: 'alpha'}), drawing=False)")
    L.pause()
    section("Your first rewrite: spider fusion")
    say("""
        The basic rule of the calculus: two spiders of the same colour joined by a wire are
        one spider, and their phases add. Below, two phased spiders sit on a plus state.
        The engine finds where the rule spider_fusion applies (match), then applies it.
    """)
    L.py("P.load(P.Z(1, 1, phase={1: 'alpha'}) | P.Z(1, 1, phase={1: 'beta'}) | P.Z(0, 1))")
    L.run("match spider_fusion")
    L.run("apply spider_fusion")
    L.run("apply spider_fusion")
    say("""
        Three spiders became one, with phase alpha + beta. Each "step" line says which rule
        fired, which nodes it consumed and the exact number it multiplied in (here 1).
        "decide" now confirms that the first and last diagrams are equal for every d, alpha
        and beta:
    """)
    L.run("decide")
    tip("""
        The "refused" samples are at d = 1, where a phase at index 1 has no meaning; the
        oracle skips them and says so. The proof itself is the normal_form method.
    """)
    L.quiz(
        "Z(0, 3) is Σ_k |k, k, k⟩. What is Z(0, 1), a Z spider with a single output?",
        ["|0⟩", "Σ_k |k⟩, the plus state", "0", "the identity"],
        1,
        "Every leg carries the same k, and there is only one leg: Σ_k |k⟩.",
    )
    L.pause()


def ch_x(L: Lesson) -> None:
    heading(5, "The Fourier box and the X spider")
    say("""
        The Fourier box F, drawn [F], is the qudit version of the Hadamard gate. It sends
        |k⟩ to the even mixture of all |j⟩, with the weight ω^(jk)/√d:
    """)
    L.py("show(P.F)")
    say("""
        Applying F four times gives back the identity, F^4 = id. The engine knows this as
        the rule fourier_cancellation:
    """)
    L.py("P.load(P.F ^ 4)")
    L.run("apply fourier_cancellation")
    L.pause()
    section("The X spider is a Z spider seen through F")
    say("""
        The second spider, drawn Ⓧ, is the X spider: a Z spider with F on every output
        and F's inverse on every input. Its simplest case is the X state with one output:
    """)
    L.py("show(P.X(0, 1))")
    say("""
        The Dirac form looks heavy, but it hides one fact used everywhere in this calculus:
        the sum Σ_q ω^(q j) over q = 0..d-1 is d when j = 0 and 0 otherwise (the
        "character sum"). So the line above is d^(-1/2) · d · |0⟩ = √d |0⟩. The X state is
        |0⟩ times √d: the engine never drops such factors.

        Put F on the plus state and you get exactly the X state. The rule is called
        fourier_state_color_change, because it changes the colour of a spider (Z spiders are drawn
        green and X spiders red in the usual coloured pictures):
    """)
    L.py("P.load(P.F | P.Z(0, 1))")
    L.run("apply fourier_state_color_change")
    L.quiz(
        "The X state X(0, 1) is which state?",
        ["Σ_k |k⟩", "√d |0⟩", "|1⟩", "the scalar d"],
        1,
        "F applied to Σ_k |k⟩: only j = 0 survives the character sum, with weight √d.",
    )
    L.pause()


def ch_rewrite(L: Lesson) -> None:
    heading(6, "Rewriting, checking and deciding")
    say("""
        A proof in the ZX calculus is a chain of rewrites, each replacing a small piece of
        the picture by an equal one. Every rule has an exact scalar. Take an X state copied
        three times by a Z spider (rule state_copy):
    """)
    L.py("P.load(P.Z(1, 3) | P.X(0, 1))")
    L.run("apply state_copy")
    say("""
        The result is three separate X states and the factor 1/d. Why 1/d? Copying √d|0⟩
        gives √d|0,0,0⟩, while three X states are (√d)^3 |0,0,0⟩; the ratio is 1/d. The
        engine records every such factor, so its equalities are exact, never "up to a
        constant". "show derivation" prints the whole chain:
    """)
    L.run("show derivation")
    L.pause()
    section("check is evidence, decide is proof")
    say("""
        There are two ways to ask whether two diagrams are equal.

          check    evaluates small concrete cases (d = 2, 3, ...): a quick test, never a proof
          decide   proves equality for every value of every symbol, or finds a counterexample

        decide answers equal (with a proof), unequal (with a counterexample) or unknown (it
        could not settle the question, and says why). It never guesses. Here is a false
        claim, that the plus state is the X state:
    """)
    L.py("print(P.decide(P.Z(0, 1), P.X(0, 1)).text)")
    say("""
        The counterexample is at d = 2: Σ_k |k⟩ = |0⟩ + |1⟩ is not √2 |0⟩. And a true claim
        that needs no hand-picked rule: F^3 also turns plus into the X state, because the
        inverse Fourier transform has the same character sum.
    """)
    L.py("print(P.decide(P.F ^ 3 | P.Z(0, 1), P.X(0, 1)).text)")
    tip("""
        normalize runs the simplifying rules until nothing changes; decide uses that and
        stronger methods in turn. The method that settled the question is named after "by".
    """)
    L.quiz(
        "check reports no mismatch at d = 2, 3 and 4. What do you know?",
        [
            "The diagrams are equal for every d",
            "Nothing at all",
            "They agree at the tested dimensions only; decide is needed for a proof",
            "They are unequal",
        ],
        2,
        "check samples cases; only decide, prove and induct give statements for every d.",
    )
    L.pause()


def ch_w(L: Lesson) -> None:
    heading(7, "The W node and the triangle: the W in ZXW")
    say("""
        The ZXW calculus adds a third kind of node to Z and X: the W node, drawn Ⓦ, with
        one input and any number of outputs. It
        hands the input's value to exactly one output and zero to the others:

          W |0⟩ = |0, 0, ..., 0⟩
          W |i⟩ = |i, 0, ..., 0⟩ + |0, i, ..., 0⟩ + ... + |0, ..., 0, i⟩     (i ≥ 1)
    """)
    L.py("show(P.W(1, 2))")
    say("""
        In the Dirac form, [j0i0 = j0o0 + j0o1] says the outputs add up to the input and
        [nnz(...) <= 1] says at most one output is non-zero. For qubits, W applied to |1⟩
        gives the famous W state |1,0,0⟩ + |0,1,0⟩ + |0,0,1⟩. Because W adds values, it lets a
        diagram write a sum of states, and the complete rule sets of ZXW are built on that.
    """)
    L.pause()
    section("W rules")
    say("Two W nodes in a row fuse into one, like spiders (w_fusion):")
    L.py("P.load((P.id @ P.W(1, 2)) | P.W(1, 2))")
    L.run("apply w_fusion")
    say("And W fed with the X state √d|0⟩ hands out zeros (w_zero_copy):")
    L.py("P.load(P.W(1, 2) | P.X(0, 1))")
    L.run("apply w_zero_copy")
    section("The triangle")
    say("""
        The triangle T, drawn as a box, keeps |0⟩ and sends |i⟩ to |0⟩ + |i⟩ for i ≥ 1. It
        has an inverse Ti, and T followed by Ti is a plain wire (triangle_inverse_cancellation).
    """)
    L.py("show(P.T, drawing=False)")
    L.py("P.load(P.Z(1, 0) | P.Ti | P.T | P.Z(0, 1))")
    L.run("apply triangle_inverse_cancellation")
    say("""
        What is left is the plus state wired into its mirror image, the Z effect: the
        number Σ_k 1 = d, written as Σ_{k0,k3} [k0 = k3].
    """)
    tip("""
        The pair is capped with a state and an effect because the engine stores every wire
        end on some node: Ti then T with both ends open would leave a bare wire with no node
        to hold it. In a larger diagram the pair always has neighbours.
    """)
    L.pause()


def ch_qufinite(L: Lesson) -> None:
    heading(8, "Qufinite: wires of different dimensions")
    say("""
        "Qudit" ZXW fixes one dimension d for the whole picture. "Qufinite" ZXW lets every
        wire have its own finite dimension, and adds two connectives to move between them:

          B(s, t)   the binder: wires of dimensions s and t become one wire of dimension s·t
          S(s, t)   the splitter: the reverse

        The pair (a, b) corresponds to the single value a·t + b. A qubit and a qutrit
        together are one 6-level system.
    """)
    L.py("show(P.B(2, 3))")
    say("""
        The numbers 2 and 3 were turned into symbols d and d1, with their values recorded
        ("at d = 2, d1 = 3"). The engine works with the symbols and substitutes the numbers
        only for display, so a rewrite of this diagram holds for every s and t, not just 2 and 3.
    """)
    L.pause()
    section("Connective rules")
    say("Plus states on both inputs of a binder are one plus state on the bound wire:")
    L.py("P.load(P.B(2, 3) | (P.Z(0, 1, dim=2) @ P.Z(0, 1, dim=3)))")
    L.run("apply connective_states")
    say("""
        A wire whose two ends are joined is a loop. A loop of dimension D is the number D
        (the trace of the identity). Dimensions can be expressions too, such as d^2:
    """)
    L.run("load w = Z(1 -> 1, d^2); wire w.out0 -- w.in0")
    say("The sum of 1 over the d^2 values is the number d^2.")
    L.quiz(
        "A binder B(4, 5) has an output wire of which dimension?",
        ["9", "20", "4", "5"],
        1,
        "The binder multiplies dimensions: 4 · 5 = 20 levels.",
    )
    L.pause()


def ch_bang1(L: Lesson) -> None:
    heading(9, "!-boxes I: one picture for every n")
    say("""
        Quantum algorithms come in families: the GHZ state on n qudits, a circuit on n
        inputs. Drawing one picture per n would never end. A !-box (bang box) is a frame
        drawn around part of a diagram, labelled with a symbol n, meaning "n copies of what
        is inside". The symbol is called the multiplicity; n may be 0, 1, 2, ...

        There are two kinds.

          port-scope box   one leg of a node is repeated: the node gets n legs there
          node-scope box   whole nodes are repeated: n independent copies
    """)
    L.pause()
    section("Port-scope: a spider with n legs")
    L.py("show(P.Z(0, n))")
    say("""
        The frame ╔!═╗ marked n surrounds one output leg: the spider has n outputs. The
        Dirac form |k⟩^{n} means |k, k, ..., k⟩ with n copies of k: the GHZ state on n
        qudits, all in one line. Python prints the DSL in its long form, a box statement
        over the port v0.out0:
    """)
    L.py("print(P.Z(0, n).dsl())")
    say("By hand you would write the same with repeat, which reads back to the same family:")
    L.run("load count n; v = Z(0 -> 1); repeat n: v.out0")
    section("Node-scope: n copies of a piece")
    L.py("show(P.X(0, 1) ** n)")
    L.run("load count n; repeat n: X(0 -> 1)")
    say("""
        A ** n, or repeat n: A in the DSL, puts n copies of A side by side. The Dirac form
        now carries a copy index i0
        running from 1 to n: k[i0] is the index of copy number i0, ∏ multiplies one factor
        per copy and ⊗ puts one ket per copy side by side.
    """)
    L.pause()
    section("Unfolding a !-box")
    say("""
        A family is not evaluated by unfolding it, but you may unfold it to see what it
        means. instantiate_symbol replaces n by a number and expands every box:
    """)
    L.py("show_graph(instantiate_symbol(P.Z(0, n).diagram(), 'n', 3))")
    L.py("show_graph(instantiate_symbol((P.X(0, 1) ** n).diagram(), 'n', 3))")
    L.py("show_graph(instantiate_symbol(P.Z(0, n).diagram(), 'n', 0))")
    say("""
        At n = 3 we get GHZ_3 and three X states. At n = 0 the box disappears: a spider
        with no legs at all is the number Σ_k 1 = d. Remember that n = 0 is a member of the
        family; it is often where false claims break.
    """)
    L.pause()
    section("Wires into a !-box fan out")
    say("""
        When a node outside a box is wired to a node inside it, the outside node gets one
        leg per copy. This is how one spider can feed n copies of a gate. Here a Z spider
        sends its value to n CNOT-like targets (each an X spider with an X state):
    """)
    L.py("""
        target = P.X(2, 1) | (P.id @ P.X(0, 1))
        circuit = (P.id @ target ** n) | P.Z(1, 2) | P.F | P.X(0, 1)
        print(circuit.draw())
    """)
    L.quiz(
        "Which diagram is the n-qudit GHZ state Σ_k |k⟩^n?",
        ["P.Z(0, 1) ** n", "P.Z(0, n)", "P.X(0, n)", "P.Z(n, 0)"],
        1,
        "One Z spider with n outputs (a port-scope box); P.Z(0, 1) ** n is n separate plus "
        "states, with independent indices.",
    )
    L.pause()


def ch_bang2(L: Lesson) -> None:
    heading(10, "!-boxes II: rewriting and deciding whole families")
    say("""
        Rules work on families directly. Fusing a plus state into a spider with n outputs
        gives the n-output spider, for every n at once:
    """)
    L.py("print(P.decide(P.Z(1, n) | P.Z(0, 1), P.Z(0, n)).text)")
    L.pause()
    section("GHZ preparation for every n and every d")
    say("""
        The circuit from the last chapter starts from √d|0⟩, applies F to get the plus
        state, and copies the value into n+1 qudits with the CNOT-like targets. The claim
        is that it prepares the GHZ state on n+1 qudits, d^(-1/2) Σ_k |k⟩^(n+1):
    """)
    L.py("""
        target = P.X(2, 1) | (P.id @ P.X(0, 1))
        circuit = ((P.id @ target ** n) | P.Z(1, 2) | P.F | P.X(0, 1)) * d ** '-1/2'
        ghz = ((P.id @ P.Z(1, n)) | P.Z(0, 2)) * d ** '-1/2'
        print(ghz.draw())
        print(P.decide(circuit, ghz).text)
    """)
    say("""
        One answer, for all n and all d, with a proof. Nothing was simulated: no number of
        qudits was ever chosen. The kernel re-checks that proof from basic inference rules:
    """)
    L.py("print(P.kernel().text)")
    L.pause()
    section("False for some n, true for others")
    say("""
        A family claim must hold for every n, including 0. n loops of dimension d are the
        number d^n; one loop of dimension d^2 is the number d^2. They agree only at n = 2.
    """)
    L.run("@sq = w = Z(1 -> 1, d^2); wire w.out0 -- w.in0")
    L.run("load count n; repeat n: z = Z(1 -> 1); wire z.out0 -- z.in0")
    L.run("decide @sq")
    L.run("decide @sq for n = 2")
    say("""
        Without a restriction, decide finds the counterexample n = 0, d = 2 (1 against 4).
        With "for n = 2" it proves the restricted claim. Restrictions can also be n >= 2,
        n = 2*m + 1 and d = 3.
    """)
    L.quiz(
        "decide says unequal with counterexample n = 0. What should you check first?",
        [
            "Whether the engine is broken",
            "Whether your claim was only meant for n >= 1, and add 'for n >= 1'",
            "Whether d should be 2",
            "Nothing; the claim is false for every n",
        ],
        1,
        "Families include n = 0. A claim meant for n >= 1 is stated with a restriction.",
    )
    L.pause()


def ch_bang3(L: Lesson) -> None:
    heading(11, "!-boxes III: proof by induction")
    say("""
        Some family equalities need induction, the way mathematicians prove statements for
        every n: show it for n = 0 (the base case), then show that if it holds for n it
        holds for n + 1 (the step). For diagrams, the step works by peeling one copy off
        the !-box.
    """)
    section("Peeling one copy")
    say("""
        A box of multiplicity m + 1 is the same as a box of multiplicity m next to one
        separate copy. peel_one does exactly that:
    """)
    L.py("""
        family = (P.X(0, 1) ** (m + 1)).diagram()
        peeled = peel_one(family, next(iter(family.bang_boxes)))
        show_graph(peeled.diagram)
    """)
    say("""
        The Dirac form now has m boxed copies and one plain copy (the lone |j1o0⟩). The
        same works for a port-scope box: |k⟩^{m} |k⟩ is a spider with m boxed legs and one
        plain leg.
    """)
    L.py("show(P.Z(0, m + 1), drawing=False)")
    L.pause()
    section("Proving state copy for every n")
    say("""
        Chapter 6 copied an X state into 3 outputs. The general claim: an X state copied
        into n outputs is n X states times d^((1-n)/2). prove searches for a rewrite proof
        and, when the family needs it, inducts on n:
    """)
    L.py("""
        copies = P.Z(1, n) | P.X(0, 1)
        goal = (P.X(0, 1) ** n) * d ** ((1 - n) / 2)
        print(P.prove(copies, goal).text)
    """)
    say("Store the proof as a certificate; it prints as a mathematician would write it:")
    L.run("certificate emit induction")
    say("""
        Read the step: the left side at k + 1 contains its own instance at k (the
        port_box_unfusion step splits one leg off), the copy rule handles the new leg, and
        the induction hypothesis replaces the rest. Finally the kernel re-checks it all:
    """)
    L.run("certificate kernel")
    tip("""
        "certificate lean FILE" writes the same proof as a Lean 4 file, which the Lean proof
        assistant checks again on its own, so you need not trust the engine at all
        (docs/LEAN.txt).
    """)
    L.quiz(
        "In an induction over a !-box multiplicity n, what is the base case?",
        ["n = 1", "n = 0, where the box is removed", "n = d", "the largest n tested"],
        1,
        "The base case is n = 0: the box and everything in it vanish.",
    )
    L.pause()


def ch_next(L: Lesson) -> None:
    heading(12, "Where to go from here")
    say("""
        You have met every generator of qufinite ZXW (Z and X spiders, F, W, the triangle,
        binder and splitter), rewrites with exact scalars, check against decide, both kinds
        of !-box, family decisions, restrictions and induction.
    """)
    section("The engine shell")
    say("""
        The shell runs the same commands you typed here. Start it with  archytaszx  or run a
        file of commands with  archytaszx --unicode FILE.azx. A file may write a diagram
        over indented lines, for example the state-copy proof of chapter 11:

          @copies =
              count n
              repeat n: X(0 -> 1)
              scalar d^((1-n)/2)

          load
              count n
              X(0 -> 1) -> z = Z(1 -> 1)
              repeat n: z.out0

          prove against @copies
          certificate kernel
    """)
    section("Cheat sheet")
    say("""
        The commands you met, in the shell and at this course's prompt:

          load SOURCE          make a diagram current (DSL, Dirac or listing)
          match RULE           where a rule applies     apply RULE [at I]   rewrite there
          normalize            simplify to a fixpoint   show derivation     every step so far
          check / decide       evidence / proof         prove / induct      search / induction
          ... for n >= 2       restrict a claim         certificate kernel  re-check a proof
          rules [RULE]         the rule list            zxunicode / dirac   switch the view
    """)
    say("""
        Further reading, in docs/: SHELL.txt and ENGINE_COMMANDS.txt for commands,
        DIAGRAM.txt for !-boxes, RULES.txt for every rule, PROOFS.txt for decide, prove and
        induction, PYTHON_API.txt for the Python pieces, and MIXED.txt for discarding and
        measurement. The examples/ folder holds complete proofs of the qudit Deutsch-Jozsa
        and Bernstein-Vazirani algorithms for every n and d.
    """)
    L.pause()


CHAPTERS = [
    ("Welcome", ch_welcome),
    ("Qudits, kets and notation", ch_qudits),
    ("Diagrams: wires and boxes", ch_diagrams),
    ("The Z spider", ch_z),
    ("The Fourier box and the X spider", ch_x),
    ("Rewriting, checking and deciding", ch_rewrite),
    ("The W node and the triangle", ch_w),
    ("Qufinite: mixed dimensions", ch_qufinite),
    ("!-boxes I: one picture for every n", ch_bang1),
    ("!-boxes II: deciding whole families", ch_bang2),
    ("!-boxes III: proof by induction", ch_bang3),
    ("Where to go from here", ch_next),
]


def menu() -> int:
    """Prints the chapter list and reads a choice."""
    print("\n" + _c("1;36", "Chapters"))
    for number, (title, _) in enumerate(CHAPTERS, 1):
        print(f"  {number:2d}. {title}")
    while True:
        try:
            reply = input(_c("1;32", f"chapter (1-{len(CHAPTERS)}, q to quit): ")).strip()
        except EOFError:
            raise Quit from None
        if reply in ("q", "quit"):
            raise Quit
        if reply.isdigit() and 1 <= int(reply) <= len(CHAPTERS):
            return int(reply)


def main() -> None:
    """Runs the course from the chosen chapter to the end."""
    global INTERACTIVE
    parser = argparse.ArgumentParser(description="Qufinite ZXW with !-boxes, a terminal course.")
    parser.add_argument("--chapter", type=int, default=1, help="chapter to start at")
    parser.add_argument("--menu", action="store_true", help="choose a chapter first")
    parser.add_argument("--no-pause", action="store_true", help="print without stopping")
    args = parser.parse_args()
    INTERACTIVE = not args.no_pause and sys.stdin.isatty()
    start = max(1, min(args.chapter, len(CHAPTERS)))
    try:
        if args.menu and INTERACTIVE:
            start = menu()
        while True:
            try:
                for index in range(start - 1, len(CHAPTERS)):
                    CHAPTERS[index][1](Lesson())
                break
            except Menu:
                start = menu()
    except (Quit, KeyboardInterrupt):
        pass
    print(
        "\n" + _c("1;36", "End of the course. Run it again any time: python examples/tutorial.py")
    )


if __name__ == "__main__":
    main()
