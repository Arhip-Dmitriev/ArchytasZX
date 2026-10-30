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

"""Benchmark 6: C followed by C^-1 equals the identity, for a seeded random circuit C of ``s``
gates on ``w`` wires.

Gates: the phase Z(1 -> 1) phase{k: S k/(M d)} (S = +-1, M in 1, 2, 4), the Fourier box F, and
the controlled X (Z(1 -> 2) into X(2 -> 1), scaled by d^(1/2)). Inverses: the negated phase,
F^3, and the controlled X with F^2 on its control leg. Tools:

- ``azx-family``: ArchytasZX with d free and the whole circuit in a bang box of size n.
- ``azx-qubit``: ArchytasZX at ``dim d = 2``.
- ``pyzx``: PyZX at d = 2 (phase pi S/M, F as HAD, controlled X as CNOT), ``full_reduce`` to
  the identity with scalar 1.
"""

from __future__ import annotations

import random
import time
from fractions import Fraction

from benchmarks.harness import Inconclusive

SIZE = "s"
SIZES = [2**k for k in range(1, 11)]
WIDTHS = [2, 4, 8]
SEEDS = range(5)
TOOLS = ("azx-family", "azx-qubit", "pyzx")
QUICK_SIZES = [2, 8]
QUICK_WIDTHS = [2]
QUICK_SEEDS = range(1)

Gate = tuple


def cases(quick: bool) -> list[dict]:
    """Every tool at every (w, seed, s); ``quick`` keeps a few."""
    sizes, widths, seeds = (
        (QUICK_SIZES, QUICK_WIDTHS, QUICK_SEEDS)
        if quick
        else (
            SIZES,
            WIDTHS,
            SEEDS,
        )
    )
    return [
        {"suite": "size", "tool": tool, "w": w, "seed": seed, "s": s}
        for tool in TOOLS
        for w in widths
        for seed in seeds
        for s in sizes
    ]


def circuit(w: int, s: int, seed: int) -> list[Gate]:
    """C then C^-1: ("P", wire, S, M), ("F", wire, power) or ("C", control, target, inverse)."""
    rng = random.Random(f"{seed}-{w}-{s}")
    forward: list[Gate] = []
    for _ in range(s):
        kind = rng.choice("PFC") if w > 1 else rng.choice("PF")
        if kind == "P":
            forward.append(("P", rng.randrange(w), rng.choice((1, -1)), rng.choice((1, 2, 4))))
        elif kind == "F":
            forward.append(("F", rng.randrange(w), 1))
        else:
            c, t = rng.sample(range(w), 2)
            forward.append(("C", c, t, False))
    inverse: list[Gate] = []
    for gate in reversed(forward):
        if gate[0] == "P":
            inverse.append(("P", gate[1], -gate[2], gate[3]))
        elif gate[0] == "F":
            inverse.append(("F", gate[1], 3))
        else:
            inverse.append(("C", gate[1], gate[2], True))
    return forward + inverse


def azx_script(gates: list[Gate], w: int, family: bool) -> str:
    """The .azx script deciding the circuit against the identity."""
    head = ["dim d", "count n"] if family else ["dim d = 2"]
    body: list[str] = []
    names: list[str] = []
    cursor: list[str | None] = [None] * w
    start: list[str | None] = [None] * w

    def node(decl: str) -> str:
        name = f"g{len(names)}"
        names.append(name)
        body.append(f"{name} = {decl}")
        return name

    def through(wire: int, name: str, port: int = 0) -> None:
        if cursor[wire] is None:
            start[wire] = f"{name}.in{port}"
        else:
            body.append(f"wire {cursor[wire]} -- {name}.in{port}")
        cursor[wire] = f"{name}.out0"

    cx = 0
    for gate in gates:
        if gate[0] == "P":
            through(gate[1], node(f"Z(1 -> 1) phase{{k: {gate[2]}*k/({gate[3]}*d)}}"))
        elif gate[0] == "F":
            for _ in range(gate[2]):
                through(gate[1], node("F"))
        else:
            a, b = node("Z(1 -> 2)"), node("X(2 -> 1)")
            through(gate[1], a)
            through(gate[2], b)
            leg = f"{a}.out1"
            if gate[3]:
                for _ in range(2):
                    f = node("F")
                    body.append(f"wire {leg} -- {f}.in0")
                    leg = f"{f}.out0"
            body.append(f"wire {leg} -- {b}.in1")
            cx += 1
    for wire in range(w):
        if cursor[wire] is None:
            through(wire, node("Z(1 -> 1)"))
    body.append("inputs " + ", ".join(p for p in start if p is not None))
    body.append("outputs " + ", ".join(p for p in cursor if p is not None))
    if family:
        body.append("repeat n: " + ", ".join(names))
        body.append(f"scalar d^(n*{cx}/2)")
    elif cx:
        body.append(f"scalar d^({cx}/2)")
    ids = [f"e{k} = Z(1 -> 1)" for k in range(w)]
    if family:
        ids.append("repeat n: " + ", ".join(f"e{k}" for k in range(w)))
    indent = lambda lines: "\n".join("    " + ln for ln in lines)
    return f"@rhs =\n{indent(head + ids)}\n\nload\n{indent(head + body)}\n\ndecide against @rhs\n"


def pyzx_check(gates: list[Gate], w: int) -> dict:
    """PyZX: build the circuit, ``full_reduce`` its graph, require the identity with scalar 1."""
    import pyzx as zx

    def go() -> tuple[bool, object, int]:
        c = zx.Circuit(w)
        for gate in gates:
            if gate[0] == "P":
                c.add_gate("ZPhase", gate[1], phase=Fraction(gate[2], gate[3]))
            elif gate[0] == "F":
                for _ in range(gate[2]):
                    c.add_gate("HAD", gate[1])
            else:
                c.add_gate("CNOT", gate[1], gate[2])
        g = c.to_graph()
        size = g.num_vertices()
        zx.full_reduce(g)
        return g.is_id(), g.scalar, size

    c0, w0 = time.process_time(), time.perf_counter()
    is_id, scalar, size = go()
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    if not is_id:
        raise Inconclusive("full_reduce did not reach the identity")
    if abs(scalar.to_number() - 1) > 1e-9:
        raise Inconclusive(f"identity reached with scalar {scalar}")
    return {"cpu": cpu, "wall": wall, "nodes": size, "method": "full_reduce"}


def run(case: dict) -> dict:
    """One check of the case's circuit by its tool."""
    gates = circuit(case["w"], case["s"], case["seed"])
    if case["tool"] == "pyzx":
        return pyzx_check(gates, case["w"])
    from benchmarks.engine import decide_script

    return decide_script(azx_script(gates, case["w"], case["tool"] == "azx-family"))
