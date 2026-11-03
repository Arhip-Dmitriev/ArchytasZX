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

"""Benchmark: buggy algorithms. For every family of :mod:`benchmarks.families`, wrong
variants (a wrong phase, a dropped or wrong scalar, a missing gate inside the bang box, a gate
outside the box, an off-by-one multiplicity, a wrong dimension), and every claim of
examples/refutations.azx; each decided with n and d free and bound, then exported by
``certificate lean`` and checked by Lean.

Expected status ``refuted``; ``equal`` is ``wrong``; ``unknown`` is ``inconclusive``. Fields:
``method`` (interface, oracle_counterexample, symbolic_witness), ``time`` (the verdict's CPU
seconds), ``export_cpu``, ``lean_status``, ``lean_cpu``, ``refutation_kind`` (interface,
cyclotomic, ball arithmetic).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from benchmarks import engine, lean, workloads
from benchmarks.harness import Inconclusive, Wrong

SIZE = "n"
TIMED = ("refuted",)
RECORD_ENV = True
BINDINGS = [("free", "free"), (2, 1), (3, 4)]
QUICK_BINDINGS = [("free", "free")]
QUICK_VARIANTS = ("Bernstein-Vazirani: oracle phase doubled", "refutations.azx #1")

# family -> [(variant, kind, [(old, new), ...])]; replacements apply to the family's text.
VARIANTS: dict[str, list[tuple[str, str, list[tuple[str, str]]]]] = {
    "state copy": [
        ("copied state phased", "phase", [("X(0 -> 1) -> z", "X(0 -> 1) phase{{k: k/d}} -> z")]),
        ("scalar dropped", "scalar", [("    scalar d^((1-n)/2)\n", "")]),
        (
            "one copy too many",
            "multiplicity",
            [("repeat n: X(0 -> 1)\n", "repeat n: X(0 -> 1)\n    extra = X(0 -> 1)\n")],
        ),
        (
            "copies of dimension e",
            "dimension",
            [
                (
                    "    count n{N}\n    repeat n: X(0 -> 1)\n",
                    "    count n{N}\n    repeat n: X(0 -> 1, e)\n",
                )
            ],
        ),
    ],
    "spider fusion": [
        (
            "phase on the first spider",
            "phase",
            [("a = Z(0 -> 1)", "a = Z(0 -> 1) phase{{k: k/d}}")],
        ),
        (
            "scalar sqrt(d) added",
            "scalar",
            [("repeat n: b.out0\n", "repeat n: b.out0\n    scalar d^(1/2)\n")],
        ),
        ("one output too many", "multiplicity", [("z = Z(0 -> 1)", "z = Z(0 -> 2)")]),
    ],
    "Fourier order 4": [
        ("F^3 in the box", "missing gate", [("F -> F -> F -> F", "F -> F -> F")]),
        (
            "phased identity",
            "phase",
            [("repeat n: Z(1 -> 1)", "repeat n: Z(1 -> 1) phase{{k: k/d}}")],
        ),
        (
            "scalar d added",
            "scalar",
            [("repeat n: F -> F -> F -> F\n", "repeat n: F -> F -> F -> F\n    scalar d\n")],
        ),
    ],
    "Fourier superposition": [
        (
            "target phased",
            "phase",
            [("repeat n: Z(0 -> 1)", "repeat n: Z(0 -> 1) phase{{k: k/d}}")],
        ),
        (
            "left scalar dropped",
            "scalar",
            [
                (
                    "    repeat n: X(0 -> 1) -> F\n    scalar d^(-n/2)\n",
                    "    repeat n: X(0 -> 1) -> F\n",
                )
            ],
        ),
        (
            "F missing in the box",
            "missing gate",
            [("repeat n: X(0 -> 1) -> F", "repeat n: X(0 -> 1)")],
        ),
        (
            "one target state too many",
            "multiplicity",
            [("repeat n: Z(0 -> 1)\n", "repeat n: Z(0 -> 1)\n    extra = Z(0 -> 1)\n")],
        ),
    ],
    "Bernstein-Vazirani": [
        (
            "oracle phase doubled",
            "phase",
            [("Z(1 -> 1) phase{{k: k/d}}", "Z(1 -> 1) phase{{k: 2*k/d}}")],
        ),
        (
            "left scalar dropped",
            "scalar",
            [("phase{{k: k/d}} -> F^3\n    scalar d^(-n/2)\n", "phase{{k: k/d}} -> F^3\n")],
        ),
        ("final F^3 missing", "missing gate", [("phase{{k: k/d}} -> F^3", "phase{{k: k/d}}")]),
        (
            "one output too many",
            "multiplicity",
            [
                (
                    "repeat n: X(0 -> 1) phase{{k: -k/d}}\n",
                    "repeat n: X(0 -> 1) phase{{k: -k/d}}\n    extra = X(0 -> 1)\n",
                )
            ],
        ),
        (
            "target of dimension e",
            "dimension",
            [
                (
                    "repeat n: X(0 -> 1) phase{{k: -k/d}}\n",
                    "repeat n: X(0 -> 1, e) phase{{k: -k/e}}\n",
                )
            ],
        ),
    ],
    "Deutsch-Jozsa constant": [
        (
            "target phase negated",
            "phase",
            [("X(0 -> 1) phase{{k: -k/d}} -> F^3", "X(0 -> 1) phase{{k: k/d}} -> F^3")],
        ),
        (
            "left scalar off by sqrt(d)",
            "scalar",
            [
                (
                    "    inputs u.in1\n    scalar d^(-n/2)\n",
                    "    inputs u.in1\n    scalar d^(-(n+1)/2)\n",
                )
            ],
        ),
        (
            "F^3 in the data box",
            "missing gate",
            [("repeat n: X(0 -> 1) -> F^4", "repeat n: X(0 -> 1) -> F^3")],
        ),
        (
            "one data output too many",
            "multiplicity",
            [
                (
                    "    repeat n: X(0 -> 1)\n    t =",
                    "    repeat n: X(0 -> 1)\n    extra = X(0 -> 1)\n    t =",
                )
            ],
        ),
    ],
    "Deutsch-Jozsa balanced": [
        (
            "target phase negated",
            "phase",
            [("X(0 -> 1) phase{{k: -k/d}} -> F^3", "X(0 -> 1) phase{{k: k/d}} -> F^3")],
        ),
        (
            "left scalar dropped",
            "scalar",
            [("    c -> u\n    scalar d^(-(m+1)/2)\n", "    c -> u\n")],
        ),
        (
            "F^3 in the data box",
            "missing gate",
            [("repeat m: X(0 -> 1) -> F^4", "repeat m: X(0 -> 1) -> F^3")],
        ),
        (
            "one data output too many",
            "multiplicity",
            [
                (
                    "    repeat m: X(0 -> 1)\n    o =",
                    "    repeat m: X(0 -> 1)\n    extra = X(0 -> 1)\n    o =",
                )
            ],
        ),
    ],
    "GHZ": [
        ("GHZ phased", "phase", [("g = Z(0 -> 2)", "g = Z(0 -> 2) phase{{k: k/d}}")]),
        (
            "left scalar dropped",
            "scalar",
            [("X(0 -> 1) -> t\n    scalar d^(-1/2)\n", "X(0 -> 1) -> t\n")],
        ),
        ("F missing", "missing gate", [("X(0 -> 1) -> F -> z", "X(0 -> 1) -> z")]),
        (
            "target state outside the box",
            "outside box",
            [
                (
                    "    repeat n: z.out1 -> t = X(2 -> 1), X(0 -> 1) -> t\n",
                    "    s = X(0 -> 1)\n    repeat n: z.out1 -> t = X(2 -> 1)\n    s -> t\n",
                )
            ],
        ),
        (
            "one copy too many",
            "multiplicity",
            [("    repeat n: g.out1\n", "    repeat n: g.out1\n    h = Z(0 -> 1)\n")],
        ),
    ],
}


def _bind(value: object) -> str:
    """`` = VALUE``, or nothing for ``"free"``."""
    return "" if value == "free" else f" = {value}"


def variant_script(family: str, variant: str, n: object = "free", d: object = "free") -> str:
    """The variant's command file."""
    if family == "refutations.azx":
        index = int(variant.rsplit("#", 1)[1]) - 1
        return workloads.units(_examples_text())[index]
    _, text = workloads.FAMILIES[family]
    for name, _kind, edits in VARIANTS[family]:
        if name == variant:
            for old, new in edits:
                if old not in text:
                    raise ValueError(f"{family}: {variant}: {old!r} not in the family text")
                text = text.replace(old, new)
            return text.format(N=_bind(n), D=_bind(d))
    raise ValueError(variant)


def _examples_text() -> str:
    return (workloads.EXAMPLES / "refutations.azx").read_text(encoding="utf-8")


def variants() -> list[tuple[str, str, str]]:
    """Every (family, variant, kind)."""
    out = [(f, f"{f}: {v}", k) for f, vs in VARIANTS.items() for v, k, _ in vs]
    out += [
        ("refutations.azx", f"refutations.azx #{i + 1}", "example")
        for i in range(len(workloads.units(_examples_text())))
    ]
    return out


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every variant at every binding; ``quick`` keeps two variants, free."""
    out = []
    for family, variant, kind in variants():
        if quick and variant not in QUICK_VARIANTS:
            continue
        bindings = [("free", "free")] if family == "refutations.azx" else BINDINGS
        for d, n in QUICK_BINDINGS if quick else bindings:
            out.append(
                {
                    "suite": "refutations",
                    "family": family,
                    "variant": variant,
                    "kind": kind,
                    "d": d,
                    "n": n,
                }
            )
    return out


def refutation_kind(source: str) -> str:
    """interface, ball arithmetic or cyclotomic, from the export's header."""
    head = source.split("-/")[0]
    if "interface refutation" in head or "differ in size" in head:
        return "interface"
    if "ball arithmetic" in head:
        return "ball arithmetic"
    return "cyclotomic"


def run(case: dict[str, Any]) -> dict[str, Any]:
    """Decide, export the refutation, check it in Lean."""
    family, variant = case["family"], case["variant"]
    name = variant.split(": ", 1)[-1]
    script = variant_script(
        family, name if family != "refutations.azx" else variant, case["n"], case["d"]
    )
    with tempfile.TemporaryDirectory(prefix="bench-ref-") as tmp:
        path = Path(tmp) / "refutation.lean"
        run_ = engine.run_script(script + f"certificate lean {path}\n")
        v = engine.verdict(run_)
        if v["verdict"] == "equal":
            raise Wrong(f"equal by {v['method']}: {v['reason']}"[:400])
        if v["verdict"] != "unequal":
            raise Inconclusive(f"{v['method']}: {v['reason']}"[:400])
        cpu, wall = engine.until_verdict(run_)
        out: dict[str, Any] = {
            "status": "refuted",
            "cpu": cpu,
            "wall": wall,
            "method": v["method"],
            "nodes": v["nodes"],
            "wires": v["wires"],
            **engine.lean_export_fields(run_),
        }
        if not path.exists():
            out["lean_status"] = "no export"
            return out
        out["refutation_kind"] = refutation_kind(path.read_text(encoding="utf-8"))
        out |= lean.check(path)
    return out
