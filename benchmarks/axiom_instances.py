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

"""Random instances of the matched side of every template rule (rules_axioms.by_name):
random phases and weights, terminals as boundary ports."""

from __future__ import annotations

import random

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite.match_template import Derived, Kj, Template, Var
from archytaszx.rewrite.rules_axioms import LastEntry, MixedVar, UnitIndex, by_name

ARITIES: dict[str, list[dict[str, int]]] = {
    "zxw_s1": [{"k": 1}, {"k": 2}],
    "zxw_k1": [{"n": 0}, {"n": 1}, {"n": 3}],
    "zxw_k1_converse": [{"n": 1}, {"n": 2}],
    "zxw_h1": [{"n": 1, "m": 1}, {"n": 2, "m": 1}, {"n": 0, "m": 2}],
    "zxw_h1_converse": [{"n": 1, "m": 1}, {"n": 1, "m": 2}],
    "zxw_ww": [{"n": 1, "m": 1}, {"n": 2, "m": 2}, {"n": 2, "m": 1}],
    "zxw_ww_converse": [{"n": 1, "m": 2}, {"n": 2, "m": 2}],
    "zxw_kz": [{"n": 1, "e0": 1}, {"n": 2, "e0": 1, "e1": 2}],
    "zxw_kz_converse": [{"n": 1, "e0": 1}, {"n": 2, "e0": 1, "e1": 1}],
    "qf_wf": [{"p": 0, "q": 1, "r": 2}, {"p": 0, "q": 0, "r": 0}, {"p": 0, "q": 2, "r": 1}],
    "qf_wf_converse": [{"p": 0, "q": 1, "r": 2}, {"p": 0, "q": 2, "r": 0}],
    "qf_w_sym": [{"n": 2, "k": 0}, {"n": 3, "k": 1}],
    "qf_w_sym_converse": [{"n": 3, "k": 0}],
    "qf_z_flex": [{"m": 1, "n": 1}, {"m": 2, "n": 0}],
    "qf_z_flex_converse": [{"m": 1, "n": 1}, {"m": 1, "n": 2}],
    "qf_mw": [{"e": 1}, {"e": 2}],
    "qf_mw_converse": [{"e": 1}, {"e": 2}],
    "qf_d1": [{"n": 1, "m": 1}, {"n": 2, "m": 1}, {"n": 1, "m": 0}],
    "qf_ww": [{"m": 1, "n": 1}, {"m": 2, "n": 2}, {"m": 1, "n": 3}],
    "qf_ww_converse": [{"m": 2, "n": 2}],
    "x_spider_definition": [{"n": 1, "m": 1}, {"n": 2, "m": 1}, {"n": 0, "m": 2}],
    "qf_s1": [
        {
            "na": 2,
            "a0d": 2,
            "a0o": 0,
            "a1d": 3,
            "a1o": 1,
            "nb": 2,
            "b0d": 3,
            "b0o": 0,
            "b1d": 2,
            "b1o": 1,
            "nc": 1,
            "c0a": 1,
            "c0b": 0,
        },
        {
            "na": 3,
            "a0d": 2,
            "a0o": 0,
            "a1d": 2,
            "a1o": 1,
            "a2d": 3,
            "a2o": 1,
            "nb": 2,
            "b0d": 2,
            "b0o": 0,
            "b1d": 3,
            "b1o": 0,
            "nc": 2,
            "c0a": 1,
            "c0b": 0,
            "c1a": 2,
            "c1b": 1,
        },
        {
            "na": 2,
            "a0d": 3,
            "a0o": 0,
            "a1d": 2,
            "a1o": 1,
            "nb": 2,
            "b0d": 2,
            "b0o": 0,
            "b1d": 4,
            "b1o": 1,
            "nc": 1,
            "c0a": 1,
            "c0b": 0,
        },
    ],
    "binder_definition": [{"s": 2, "t": 3}, {"s": 3, "t": 2}],
    "state_copy_converse": [{"n": 2}, {"n": 3}],
    "connective_states_states_converse": [{"s": 2, "t": 3}, {"s": 2, "t": 2}],
    "connective_states_effects_converse": [{"s": 2, "t": 3}, {"s": 3, "t": 2}],
    "connective_states_swapped_states_converse": [{"s": 2, "t": 3}, {"s": 2, "t": 2}],
    "connective_states_swapped_effects_converse": [{"s": 2, "t": 3}, {"s": 3, "t": 2}],
    "x_fusion_converse": [
        {"ni": 1, "no": 1, "mi": 1, "mo": 0},
        {"ni": 2, "no": 2, "mi": 2, "mo": 1},
        {"ni": 0, "no": 2, "mi": 0, "mo": 3},
    ],
    "state_copy_swapped_converse": [{"n": 2}, {"n": 3}],
    "w_zero_copy_converse": [{"n": 2}, {"n": 3}],
    "w_z_effect_converse": [{"m": 2}, {"m": 3}],
    "w_zero_effect_converse": [{"m": 2}, {"m": 3}],
    "wz_bialgebra_converse": [{"n": 1, "m": 2}, {"n": 2, "m": 2}, {"n": 2, "m": 3}],
    "binder_definition_converse": [{"s": 2, "t": 3}],
    "qf_dd": [{"m": 2, "n": 3}, {"m": 3, "n": 2}, {"m": 2, "n": 2}],
    "qf_dd_converse": [{"m": 2, "n": 3}, {"m": 2, "n": 2}],
    "qf_e": [{"n": 3, "e": 1}, {"n": 3, "e": 2}, {"n": 4, "e": 2}],
    "qf_e_converse": [{"n": 3, "e": 1}],
    "qf_dzx": [{"m": 2, "n": 2, "p": 1, "q": 1}, {"m": 2, "n": 3, "p": 1, "q": 2}],
    "qf_dzx_converse": [{"m": 2, "n": 2, "p": 1, "q": 1}],
    "qf_k0": [{"m": 2, "n1": 3, "n2": 3}, {"m": 3, "n1": 2, "n2": 4}],
    "qf_k0_converse": [{"m": 2, "n1": 3, "n2": 3}, {"m": 3, "n1": 2, "n2": 4}],
    "qf_bzw": [{"m": 2, "n": 3, "k": 2}, {"m": 3, "n": 2, "k": 3}],
    "qf_bzw_converse": [{"m": 2, "n": 3, "k": 2}],
    "qf_ta": [
        {"m": 2, "n": 3, "k": 2, "i": 2, "e0": 1, "e1": 2},
        {"m": 3, "n": 2, "k": 2, "i": 1, "e0": 1},
    ],
    "qf_ta_converse": [{"m": 2, "n": 3, "k": 2, "i": 1, "e0": 2}],
    "zxw_s1_converse": [
        {"n": 2, "mask": 1, "k": 1},
        {"n": 3, "mask": 5, "k": 2},
        {"n": 1, "mask": 0, "k": 1},
    ],
    "qf_d1_converse": [{"n": 1, "m": 1}, {"n": 2, "m": 1}, {"n": 0, "m": 2}],
    "x_spider_definition_converse": [{"n": 1, "m": 1}, {"n": 0, "m": 2}],
}

DIMS = (2, 3, 4, 5, 6)
MIXED = {
    "connective_states_states",
    "connective_states_effects",
    "connective_states_swapped_states",
    "connective_states_swapped_effects",
    "qf_s1",
    "binder_definition",
    "qf_dd",
    "qf_e",
    "qf_dzx",
    "qf_k0",
    "qf_bzw",
    "qf_ta",
}
BIG = {
    "zxw_hd",
    "zxw_hd_converse",
    "qf_hd",
    "qf_hd_converse",
    "zxw_va",
    "qf_va",
    "zxw_vw",
    "zxw_vw_converse",
    "zxw_zv",
}


def rand_scalar(rng: random.Random) -> Scalar:
    """A random exact complex scalar with small rational parts."""
    re = sp.Rational(rng.randint(-4, 4), rng.randint(1, 3))
    im = sp.Rational(rng.randint(-4, 4), rng.randint(1, 3))
    return Scalar(re + sp.I * im)


def rand_vector(rng: random.Random, dim: Dim, roots: bool = False) -> PhaseVector | None:
    """A random phase vector, weight vector, both, or None; with ``roots``, weights that are
    roots of unity."""
    r = dim.to_int()
    if roots:
        return PhaseVector(
            dim,
            weights={
                k: Scalar(sp.exp(2 * sp.pi * sp.I * sp.Rational(rng.randint(0, 11), 12)))
                for k in range(1, r)
            },
        )
    kind = rng.choice(["none", "phase", "weight", "both"])
    if kind == "none":
        return None
    phases = {}
    weights = {}
    if kind in ("phase", "both"):
        for k in range(1, r):
            if rng.random() < 0.7:
                phases[k] = Phase.turns(sp.Rational(rng.randint(0, 11), 12))
    if kind in ("weight", "both"):
        for k in range(1, r):
            if rng.random() < 0.8:
                weights[k] = rand_scalar(rng)
    return PhaseVector(dim, phases, weights=weights)


def instantiate(
    template: Template, dim: Dim, rng: random.Random, roots: bool = False
) -> tuple[Diagram, dict[str, object]]:
    """A diagram of the template with random parameters; terminals become boundary ports.
    An empty template gives a host of two weighted Z spiders joined by one wire."""
    r = dim.to_int()
    if not template.nodes:
        d = Diagram()
        a = d.add_node(Z_SPIDER, [dim], [dim], phase=rand_vector(rng, dim, roots))
        b = d.add_node(Z_SPIDER, [dim], [dim, dim], phase=rand_vector(rng, dim, roots))
        d.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        d.set_boundary_inputs([PortRef(a, Direction.INPUT, 0)])
        d.set_boundary_outputs([PortRef(b, Direction.OUTPUT, 0), PortRef(b, Direction.OUTPUT, 1)])
        return d, {}
    bindings: dict[str, object] = {}
    for node in template.nodes:
        p = node.param
        if isinstance(p, Var) and p.name not in bindings:
            bindings[p.name] = rand_vector(rng, dim, roots) or PhaseVector(dim)
        elif isinstance(p, (Kj, UnitIndex)) and p.name not in bindings:
            bindings[p.name] = sp.Integer(rng.randrange(r))
        elif isinstance(p, MixedVar) and p.name not in bindings:
            bindings[p.name] = rand_vector(rng, Dim.concrete(p.low), roots) or PhaseVector(
                Dim.concrete(p.low)
            )
        elif isinstance(p, LastEntry) and p.name not in bindings:
            bindings[p.name] = rand_scalar(rng)
    d = Diagram()
    ids = {}
    extras: list[PortRef] = []
    for node in template.nodes:
        p = node.param
        in_dims, out_dims = node.leg_dims(dim)
        own = node.dim if node.dim is not None else dim
        if isinstance(p, Derived):
            phase = p.compute(own, bindings)
        else:
            phase = p.build(own, bindings)
        if phase is None and node.ins == 0 and node.outs == 0:
            phase = PhaseVector(own)
        extra_in = extra_out = 0
        if node.open:
            extra_in, extra_out = rng.randint(0, 2), rng.randint(0, 2)
            own_dim = node.dim if node.dim is not None else dim
            in_dims = (*in_dims, *([own_dim] * extra_in))
            out_dims = (*out_dims, *([own_dim] * extra_out))
        ids[node.key] = d.add_node(node.gen, list(in_dims), list(out_dims), phase=phase)
        extras.extend(
            PortRef(ids[node.key], Direction.INPUT, node.ins + k) for k in range(extra_in)
        )
        extras.extend(
            PortRef(ids[node.key], Direction.OUTPUT, node.outs + k) for k in range(extra_out)
        )
        if node.pink:
            m, n = node.outs, node.ins
            d.multiply_scalar(Scalar(sp.Pow(r, sp.Rational(m + n, 2) - 1)) ** -1)
    for wa, wb in template.wires:
        d.add_wire(PortRef(ids[wa[0]], wa[1], wa[2]), PortRef(ids[wb[0]], wb[1], wb[2]))
    ins: list[PortRef] = []
    outs: list[PortRef] = []
    for name in sorted(template.terminals):
        key, direction, index = template.terminals[name]
        ref = PortRef(ids[key], direction, index)
        (ins if direction is Direction.INPUT else outs).append(ref)
    ins += [r for r in extras if r.direction is Direction.INPUT]
    outs += [r for r in extras if r.direction is Direction.OUTPUT]
    d.set_boundary_inputs(ins)
    d.set_boundary_outputs(outs)
    if template.scalar is not None:
        d.multiply_scalar(template.scalar(dim, bindings) ** -1)
    return d, bindings


def dims_for(name: str) -> tuple[int, ...]:
    """The dimensions the rule is instantiated at."""
    base = name.removesuffix("_converse")
    if base in MIXED:
        return (2,)
    if name == "connective_inverse_converse":
        return (4, 6)
    return (3,) if name not in BIG else (2, 3)


def instance(name: str, seed: int) -> Diagram | None:
    """A diagram holding one match of rule ``name``, from the seeded generator, or None."""
    from archytaszx.rewrite.rules_library import RULES

    axiom = by_name()[name]
    rule = RULES[name]
    rng = random.Random(f"{name}-{seed}")
    for r in dims_for(name):
        dim = Dim.concrete(r)
        for arities in ARITIES.get(name, [{}]):
            template = axiom.left.template(dim, arities)
            if template is None:
                continue
            diagram, _ = instantiate(template, dim, rng, roots=name == "weight_to_phase")
            if rule.pattern.find_matches(diagram):
                return diagram
    return None
