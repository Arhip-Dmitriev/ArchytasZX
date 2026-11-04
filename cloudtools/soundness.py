"""Oracle soundness sweep of template rules.

usage: soundness.py RULE DIM SEED_START SEED_COUNT OUT_FILE

Per seed: a random instance of RULE's matched side at DIM (random arities, phases, weights,
0-2 extra legs on open nodes, random weighted context on boundary legs), the rule applied at a
random match, both sides contracted numerically and compared. One line per seed in OUT_FILE:
PASS, FAIL (with details), NO-MATCH, INVALID (no instance satisfies the conditions), SIZE
(contraction over the element limit), ERROR (with the exception).
"""

from __future__ import annotations

import os
import random
import sys
import traceback

import numpy as np
import sympy as sp

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.diagram.generators import W_NODE, X_SPIDER, Z_SPIDER
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.match_template import Derived, Kj, Var
from archytaszx.rewrite.rules_axioms import LastEntry, MixedVar, UnitIndex, by_name
from archytaszx.rewrite.rules_library import RULES
from archytaszx.semantics.contract_numeric import ContractSizeError, contract

MAX_ELEMENTS = 2**26
sys.setrecursionlimit(20000)
ATTEMPTS = 40
INT_NAMES = {"j", "k", "i", "e", "m", "l"}


def random_vector(dim: Dim, rng: random.Random) -> PhaseVector | None:
    """A phaseless, pure-phase or weighted vector over a concrete ``dim``."""
    r = dim.to_int()
    kind = rng.random()
    if kind < 0.15 or r < 2:
        return None
    if kind < 0.45:
        entries = {k: Phase(sp.Rational(rng.randrange(24), 24)) for k in range(1, r)}
        return PhaseVector(dim, entries)
    if kind < 0.6:
        units = (sp.Integer(1), sp.Integer(-1), sp.I, -sp.I)
        return PhaseVector.from_weights(dim, {k: Scalar(rng.choice(units)) for k in range(1, r)})
    weights = {}
    for k in range(1, r):
        if rng.random() < 0.2:
            weights[k] = Scalar.zero()
        else:
            re = sp.Rational(rng.randint(-6, 6), rng.randint(1, 4))
            im = sp.Rational(rng.randint(-6, 6), rng.randint(1, 4))
            weights[k] = Scalar(re + sp.I * im)
    return PhaseVector.from_weights(dim, weights)


def random_scalar(rng: random.Random) -> Scalar:
    re = sp.Rational(rng.randint(-6, 6), rng.randint(1, 4))
    im = sp.Rational(rng.randint(-6, 6), rng.randint(1, 4))
    value = re + sp.I * im
    return Scalar(value if value != 0 else sp.Integer(1))


def random_bindings(template, dim: Dim, rng: random.Random) -> dict[str, object]:
    """Random values for every variable the template's parameters read."""
    r = dim.to_int()
    bindings: dict[str, object] = {}
    for tnode in template.nodes:
        p = tnode.param
        own = tnode.dim if tnode.dim is not None else dim
        if isinstance(p, (Kj, UnitIndex)):
            bindings.setdefault(p.name, sp.Integer(rng.randrange(own.to_int())))
        elif isinstance(p, LastEntry):
            bindings.setdefault(p.name, random_scalar(rng))
        elif isinstance(p, MixedVar):
            low = Dim.concrete(p.low)
            bindings.setdefault(p.name, random_vector(low, rng) or PhaseVector(low))
        elif isinstance(p, Var):
            bindings.setdefault(p.name, random_vector(own, rng) or PhaseVector(own))
    for tnode in template.nodes:
        p = tnode.param
        if isinstance(p, Derived):
            for name in p.reads:
                if name not in bindings:
                    if name in INT_NAMES:
                        bindings[name] = sp.Integer(rng.randrange(r))
                    else:
                        bindings[name] = random_vector(dim, rng) or PhaseVector(dim)
    return bindings


def build(template, dim: Dim, bindings, rng: random.Random) -> Diagram:
    """The template as a diagram: terminals and extra legs on the boundary, a random context."""
    diagram = Diagram()
    port: dict[tuple, PortRef] = {}
    extra: list[PortRef] = []
    for tnode in template.nodes:
        own = tnode.dim if tnode.dim is not None else dim
        phase = tnode.param.build(own, bindings)
        ins, outs = tnode.ins, tnode.outs
        add_in = add_out = 0
        if tnode.open:
            n = rng.randint(0, 2)
            if tnode.gen.name == X_SPIDER.name:
                add_in = rng.randint(0, n)
                add_out = n - add_in
            elif tnode.gen.name in (Z_SPIDER.name, W_NODE.name):
                add_out = n
        if tnode.port_dims is not None:
            in_dims, out_dims = list(tnode.port_dims[0]), list(tnode.port_dims[1])
        else:
            in_dims = [own] * (ins + add_in)
            out_dims = [own] * (outs + add_out)
        if phase is None and not in_dims and not out_dims and tnode.gen.name in (
            Z_SPIDER.name,
            X_SPIDER.name,
        ):
            phase = PhaseVector(own)
        nid = diagram.add_node(tnode.gen, in_dims, out_dims, phase=phase)
        for i in range(ins):
            port[(tnode.key, Direction.INPUT, i)] = PortRef(nid, Direction.INPUT, i)
        for i in range(outs):
            port[(tnode.key, Direction.OUTPUT, i)] = PortRef(nid, Direction.OUTPUT, i)
        extra += [PortRef(nid, Direction.INPUT, ins + i) for i in range(add_in)]
        extra += [PortRef(nid, Direction.OUTPUT, outs + i) for i in range(add_out)]
    wired = set()
    for a, b in template.wires:
        diagram.add_wire(port[a], port[b])
        wired |= {a, b}
    free = [ref for key, ref in port.items() if key not in wired] + extra
    b_in: list[PortRef] = []
    b_out: list[PortRef] = []
    for ref in free:
        leg = diagram.nodes[ref.node_id].legs(ref.direction)[ref.index]
        if rng.random() < 0.3:
            ctx = diagram.add_node(Z_SPIDER, [leg.dim], [leg.dim], random_vector(leg.dim, rng))
            if ref.direction is Direction.OUTPUT:
                diagram.add_wire(ref, PortRef(ctx, Direction.INPUT, 0))
                b_out.append(PortRef(ctx, Direction.OUTPUT, 0))
            else:
                diagram.add_wire(PortRef(ctx, Direction.OUTPUT, 0), ref)
                b_in.append(PortRef(ctx, Direction.INPUT, 0))
        elif ref.direction is Direction.OUTPUT:
            b_out.append(ref)
        else:
            b_in.append(ref)
    diagram.set_boundary_inputs(b_in)
    diagram.set_boundary_outputs(b_out)
    return diagram


def random_host(dim: Dim, rng: random.Random) -> Diagram:
    """Two random spiders joined by a wire, one boundary leg each."""
    diagram = Diagram()
    a = diagram.add_node(Z_SPIDER, [dim], [dim], random_vector(dim, rng))
    gen = rng.choice((Z_SPIDER, X_SPIDER))
    phase = random_vector(dim, rng) if gen is Z_SPIDER else None
    b = diagram.add_node(gen, [dim], [dim], phase)
    diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(a, Direction.INPUT, 0)])
    diagram.set_boundary_outputs([PortRef(b, Direction.OUTPUT, 0)])
    return diagram


def random_free(axiom, dim: Dim, match, diagram, rng: random.Random):
    """The match with random values for the rule's free variables, or None."""
    pattern = RULES[axiom.name].pattern
    values = {}
    for name in axiom.free:
        if name in INT_NAMES:
            values[name] = sp.Integer(rng.randrange(dim.to_int()))
        else:
            values[name] = random_vector(dim, rng) or PhaseVector(dim)
    try:
        return pattern.with_values(diagram, match, values)
    except Exception:  # noqa: BLE001
        return None


def arity_choices(side, dim: Dim) -> list[dict[str, int]]:
    """Every arity assignment (values 0..3) for which the side has a template at ``dim``."""
    found: list[dict[str, int]] = []
    stack: list[dict[str, int]] = [{}]
    while stack:
        arities = stack.pop()
        try:
            template = side.template(dim, arities)
        except KeyError as exc:
            key = exc.args[0]
            if isinstance(key, str) and key not in arities and len(arities) < 4:
                stack += [{**arities, key: v} for v in range(4)]
            continue
        except Exception:  # noqa: BLE001
            continue
        if template is not None:
            found.append(arities)
    return found


def one(name: str, dim: Dim, seed: int) -> str:
    rng = random.Random(seed)
    axiom = by_name()[name]
    rule = RULES[name]
    choices = arity_choices(axiom.left, dim)
    if not choices:
        return f"INVALID {name} d={dim} seed={seed}: no template at this dimension"
    for _ in range(ATTEMPTS):
        arities = rng.choice(choices)
        template = axiom.left.template(dim, arities)
        try:
            if not template.nodes:
                diagram = random_host(dim, rng)
            else:
                bindings = random_bindings(template, dim, rng)
                diagram = build(template, dim, bindings, rng)
        except Exception:  # noqa: BLE001
            continue
        matches = rule.pattern.find_matches(diagram)
        if not matches:
            continue
        match = rng.choice(matches)
        free = "default"
        if axiom.free and rng.random() < 0.6:
            chosen = random_free(axiom, dim, match, diagram, rng)
            if chosen is not None:
                match, free = chosen, "random"
        try:
            result = apply(diagram, rule, match)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR {name} d={dim} seed={seed} arities={arities}: apply: {exc!r}"
        try:
            lhs = contract(diagram, max_elements=MAX_ELEMENTS)
            rhs = contract(result.diagram, max_elements=MAX_ELEMENTS)
        except ContractSizeError:
            return f"SIZE {name} d={dim} seed={seed} arities={arities}"
        a, b = lhs.tensor, rhs.tensor
        if a.shape != b.shape:
            return f"FAIL {name} d={dim} seed={seed} arities={arities}: shapes {a.shape} {b.shape}"
        scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        err = float(np.max(np.abs(a - b))) / scale if a.size else 0.0
        if err > 1e-9:
            return (
                f"FAIL {name} d={dim} seed={seed} arities={arities} free={free}: "
                f"max rel err {err:.3e}"
            )
        return f"PASS {name} d={dim} seed={seed} arities={arities} free={free}"
    return f"NO-MATCH {name} d={dim} seed={seed}"


def main() -> None:
    name, d, start, count, out = sys.argv[1:6]
    dim = Dim.concrete(int(d))
    lines = []
    for seed in range(int(start), int(start) + int(count)):
        try:
            lines.append(one(name, dim, seed))
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()[-1]
            lines.append(f"ERROR {name} d={dim} seed={seed}: {tb}")
    tmp = f"{out}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(tmp, out)


if __name__ == "__main__":
    main()
