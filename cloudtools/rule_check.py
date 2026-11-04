"""Single-step Lean check of rules over the step route (lean_net.export, never lean_sym).

usage: rule_check.py [--lean] [--jobs N] [--out DIR] [--dims 2,3,5] [--arities K] [--leandir D]
                     RULE ... | all

Per rule and instance class (dimension, arity assignment; symbolic d where the template builds
there; older rules from fixed DSL inputs): the instance's first match applied, the step's
kernel theorem exported with lean_net.export, and with --lean checked by `lake env lean` (at
most N processes). One line per instance in OUT/RULE.txt:
  STATUS RULE class=... route=... time=... [note]
STATUS is PROVED (exported, no `unproved_`, Lean accepted or not run), UNPROVED (an
`unproved_` axiom, with its note), REJECTED (Lean exit non-zero), EXPORT (NetUnsupported or a
kernel error), NO-INSTANCE.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(20000)

from soundness import arity_choices, build, random_bindings, random_host  # noqa: E402

from archytaszx.algebra.dimension import Dim  # noqa: E402
from archytaszx.algebra.phase import PhaseVector  # noqa: E402
from archytaszx.repl.parser import parse_dsl  # noqa: E402
from archytaszx.rewrite.engine import apply  # noqa: E402
from archytaszx.rewrite.match_template import Var  # noqa: E402
from archytaszx.rewrite.rules_axioms import by_name  # noqa: E402
from archytaszx.rewrite.rules_library import RULES  # noqa: E402
from archytaszx.semantics import kernel, lean_net  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEANDIR = os.path.join(ROOT, "archytaszx", "lean")
LAKE = os.path.expanduser("~/.elan/bin/lake")

OLDER: dict[str, list[str]] = {
    "spider_fusion": ["a = Z(0 -> 2) phase{1: alpha}; b = Z(1 -> 2); a -> b"],
    "z_fusion_prime_d": [
        "a = Z(0 -> 1, literal 3) phase{1: 1/3}; b = Z(1 -> 1, literal 3); a -> b"
    ],
    "fourier_cancellation": ["X(0 -> 1) -> F^4"],
    "identity_removal": ["X(0 -> 1) -> Z(1 -> 1) -> F"],
    "zx_cap": ["Z(0 -> 1) -> X(1 -> 0)"],
    "zx_cap_swapped": ["X(0 -> 1) -> Z(1 -> 0)"],
    "state_copy": ["X(0 -> 1) -> c = Z(1 -> 3)"],
    "state_copy_swapped": ["Z(0 -> 1) -> c = X(1 -> 3)"],
    "hopf": ["u = Z(1 -> 2); v = X(2 -> 1); u.out0 -> v.in0; u.out1 -> F -> F -> v.in1"],
    "hopf_swapped": [
        "v = X(1 -> 2); u = Z(2 -> 1); v.out0 -> u.in0; v.out1 -> F -> F -> u.in1"
    ],
    "bialgebra": ["x = X(2 -> 1); z = Z(1 -> 2); x -> z"],
    "bialgebra_swapped": ["z = Z(2 -> 1); x = X(1 -> 2); z -> x"],
    "fourier_state_color_change": ["Z(0 -> 1) -> F"],
    "fourier_state_color_change_swapped": ["X(0 -> 1) -> F"],
    "triangle_inverse_cancellation": ["X(0 -> 1) -> T -> Ti -> F"],
    "w_fusion": ["w = W(1 -> 2); v = W(1 -> 2); w.out1 -> v"],
    "w_identity": ["X(0 -> 1) -> W(1 -> 1) -> F"],
    "w_zero_copy": ["X(0 -> 1) -> w = W(1 -> 2)"],
    "triangle_zero_state": ["X(0 -> 1) -> T"],
    "triangle_zero_effect": ["T -> X(1 -> 0)"],
    "w_zero_effect": ["w = W(1 -> 2); x = X(1 -> 0); wire w.out1 -- x.in0"],
    "w_z_effect": ["w = W(1 -> 3); z = Z(1 -> 0); wire w.out2 -- z.in0"],
    "wz_bialgebra": ["z = Z(2 -> 1); w = W(1 -> 2); z -> w"],
    "connective_inverse": [
        "z = Z(0 -> 1, literal 6) phase{1: 1/3}; p = S(literal 3, literal 2); "
        "b = B(literal 3, literal 2); f = F(literal 6); z -> p; "
        "wire p.out0 -- b.in0, p.out1 -- b.in1; b -> f"
    ],
    "connective_inverse_bind_first": [
        "x = Z(0 -> 1, literal 3) phase{1: 1/3}; y = Z(0 -> 1, literal 2) phase{1: 1/2}; "
        "b = B(literal 3, literal 2); p = S(literal 3, literal 2); x -> b.in0; y -> b.in1; "
        "b -> p; f = F(literal 3); g = F(literal 2); p.out0 -> f; p.out1 -> g"
    ],
    "connective_states": [
        "p = Z(0 -> 1, literal 3); q = Z(0 -> 1, literal 2); b = B(literal 3, literal 2); "
        "p -> b.in0; q -> b.in1"
    ],
    "connective_states_swapped": [
        "p = X(0 -> 1, literal 3); q = X(0 -> 1, literal 2); b = B(literal 3, literal 2); "
        "p -> b.in0; q -> b.in1"
    ],
    "port_box_unfusion": ["count n; z = Z(1 -> 3); repeat n: z.out0"],
}


def plain_bindings(template, dim: Dim) -> dict[str, object]:
    """Phaseless values for every vector variable of ``template``."""
    bindings: dict[str, object] = {}
    for tnode in template.nodes:
        p = tnode.param
        if p is None:
            continue
        if type(p) is Var:
            bindings.setdefault(p.name, PhaseVector(tnode.dim if tnode.dim is not None else dim))
    return bindings


def template_instances(name: str, dims: list[int], per_dim: int):
    """(class, diagram) pairs: concrete dims with random parameters, then symbolic d."""
    axiom = by_name()[name]
    rule = RULES[name]
    out = []
    for d in dims:
        dim = Dim.concrete(d)
        choices = arity_choices(axiom.left, dim)
        choices = sorted(choices, key=lambda a: sorted(a.items()))
        step = max(1, len(choices) // max(1, per_dim))
        choices = choices[::step] + [c for i, c in enumerate(choices) if i % step]
        found = 0
        for arities in choices:
            if found >= per_dim:
                break
            for seed in range(30):
                rng = random.Random(seed)
                template = axiom.left.template(dim, arities)
                try:
                    diagram = (
                        random_host(dim, rng)
                        if not template.nodes
                        else build(template, dim, random_bindings(template, dim, rng), rng)
                    )
                    if rule.pattern.find_matches(diagram):
                        out.append((f"d={d} arities={arities} seed={seed}", diagram))
                        found += 1
                        break
                except Exception:  # noqa: BLE001
                    continue
    if not axiom.concrete_only:
        dim = Dim.symbol("d")
        for arities in arity_choices(axiom.left, Dim.concrete(3))[:per_dim]:
            try:
                template = axiom.left.template(dim, arities)
                bindings = plain_bindings(template, dim)
                rng = random.Random(0)
                diagram = build(template, dim, bindings, rng)
                if rule.pattern.find_matches(diagram):
                    out.append((f"d=sym arities={arities}", diagram))
            except Exception:  # noqa: BLE001
                continue
    return out


def instances(name: str, dims: list[int], per_dim: int):
    if name in OLDER:
        return [(f"dsl#{i}", parse_dsl(src)) for i, src in enumerate(OLDER[name])]
    return template_instances(name, dims, per_dim)


def lean_check(path: str) -> tuple[int, float, str]:
    start = time.time()
    done = subprocess.run(
        [LAKE, "env", "lean", path], cwd=LEANDIR, capture_output=True, text=True, check=False
    )
    errors = [line for line in (done.stdout + done.stderr).splitlines() if "error" in line]
    return done.returncode, time.time() - start, " | ".join(errors)[:400]


def export_one(name: str, cls: str, diagram, workdir: str, index: int):
    """(status, route, note, lean path or None) for the first match of ``name`` in ``diagram``."""
    rule = RULES[name]
    route = "builder" if name in lean_net._RULES else "through_forward"
    try:
        result = apply(diagram, rule, rule.pattern.find_matches(diagram)[0])
        theorem = kernel.rule(diagram, result.step)
        source = lean_net.export(theorem, f"rc_{index}")
    except Exception as exc:  # noqa: BLE001
        return "EXPORT", route, f"{type(exc).__name__}: {exc}"[:300], None
    if "\naxiom unproved_" in source:
        notes = re.findall(r"/-- ((?:(?!-/).)*?) -/\naxiom unproved_", source, re.S)
        return "UNPROVED", route, " ; ".join(n.replace("\n", " ") for n in notes)[:300], None
    path = os.path.abspath(os.path.join(workdir, f"{name}__{index}.lean"))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    return "PROVED", route, "", path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rules", nargs="+")
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, ".cicero_work", "rule_check"))
    ap.add_argument("--dims", default="2,3,5")
    ap.add_argument("--arities", type=int, default=2)
    ap.add_argument("--leandir", default=LEANDIR)
    args = ap.parse_args()
    globals()["LEANDIR"] = args.leandir
    names = sorted(RULES) if args.rules == ["all"] else args.rules
    dims = [int(x) for x in args.dims.split(",")]
    os.makedirs(args.out, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="rc_", dir=args.out)
    pending = []
    lines: dict[str, list[str]] = {}
    for name in names:
        lines[name] = []
        found = instances(name, dims, args.arities)
        if not found:
            lines[name].append(f"NO-INSTANCE {name}")
        for i, (cls, diagram) in enumerate(found):
            start = time.time()
            status, route, note, path = export_one(name, cls, diagram, workdir, i)
            t = time.time() - start
            entry = [status, name, f"class={cls}", f"route={route}", f"export={t:.1f}s", note]
            if path is not None and args.lean:
                pending.append((name, len(lines[name]), path))
            lines[name].append(entry)  # type: ignore[arg-type]
    if pending:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            futures = {pool.submit(lean_check, p): (n, k) for n, k, p in pending}
            for fut in concurrent.futures.as_completed(futures):
                n, k = futures[fut]
                code, t, errors = fut.result()
                entry = lines[n][k]
                entry.insert(5, f"lean={code} {t:.0f}s")  # type: ignore[union-attr]
                if code != 0:
                    entry[0] = "REJECTED"  # type: ignore[index]
                    entry[-1] = errors  # type: ignore[index]
    for name, rows in lines.items():
        text = "\n".join(r if isinstance(r, str) else " ".join(x for x in r if x) for r in rows)
        tmp = os.path.join(args.out, f".{name}.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        os.replace(tmp, os.path.join(args.out, f"{name}.txt"))
        print(text, flush=True)


if __name__ == "__main__":
    main()
