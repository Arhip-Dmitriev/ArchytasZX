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

"""The false set F: one break per S equation, applied to the circuit IR (BENCHMARKS.txt,
section 4).

A break kind lists the edits it tries on an equation (``variants``), each with the points
where it should differ (``probe``) and the record of where it stays equal (``equal_where``).
Generation applies the first variant whose witness search finds an entry that differs; a
kind with no such variant does not apply, and the assignment moves on to the next kind.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from benchmarks import ir, subject

TOL = 1e-8
DENSE_LIMIT = 1 << 18
ORACLE_D = 4
ORACLE_N = 3
CARMICHAEL = (561, 1105, 1729)


@dataclass(frozen=True)
class Edit:
    """One applied break: the kind, the variant label and its parameters."""

    kind: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)

    def apply(self, c: ir.Circuit, inst: Any, nv: int | None, dv: int | None) -> ir.Circuit:
        """The broken circuit."""
        return EDITS[self.kind](c, self.params, inst, nv, dv)

    def apply_spec(self, spec: str, inst: Any, nv: int | None, dv: int | None, ctx: Any) -> str:
        """The specification (no break edits it)."""
        return spec


def _outputs(c: ir.Circuit) -> list[str]:
    """Wires still open at the end."""
    closed = {g.q for g in c.gates if isinstance(g, (ir.Post, ir.Discard))}
    return [q for q in c.wires if q not in closed]


def _inputs(c: ir.Circuit) -> list[str]:
    """Wires without an Init."""
    init = {g.q for g in c.gates if isinstance(g, ir.Init)}
    return [q for q in c.wires if q not in init]


def _measured(c: ir.Circuit) -> set[str]:
    return {g.q for g in c.gates if isinstance(g, ir.Measure)}


def _role_lists(c: ir.Circuit) -> dict[str, list[str]]:
    """The wire lists a variant may name a wire from: open unmeasured outputs (singles first,
    reversed), inputs, every wire."""
    return {
        "phase": _wires_for_phase(c),
        "inputs": list(_inputs(c)),
        "all": list(c.wires),
    }


def _wire(c: ir.Circuit, p: dict[str, Any]) -> str:
    """The wire an edit names: by role and slot when recorded (unrolled equations, where wire
    names depend on n), else by name."""
    if "role" in p:
        return _role_lists(c)[p["role"]][p["slot"]]
    return p["wire"]


def _append(c: ir.Circuit, *gates: ir.Gate) -> ir.Circuit:
    return replace(c, gates=(*c.gates, *gates))


def _prepend(c: ir.Circuit, q: str, gate: ir.Gate) -> ir.Circuit:
    return replace(c, gates=(gate, *c.gates))


def _times(c: ir.Circuit, factor: str) -> ir.Circuit:
    return replace(c, scalar=f"({c.scalar}) * ({factor})")


def _product_poly(sym: str, top: int) -> str:
    """sym (sym - 1) ... (sym - top)."""
    return "*".join([sym] + [f"({sym}-{j})" for j in range(1, top + 1)])


def _e_phase(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    gen = ir.ZPhase if p.get("gen", "Z") == "Z" else ir.XPhase
    return _append(c, gen(_wire(c, p), rule=p["rule"]))


def _e_beyond_n(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    big_n = p["N"]
    if inst.equation.form == "unrolled":
        q = _outputs(c)[p["index"]]
        return _append(c, ir.X(q, 1)) if nv is not None and nv > big_n else c
    m = big_n + 2
    counts = [v for v in c.registers.values() if not str(v).isdigit()]
    if not counts:
        return _times(c, "-1") if nv is not None and nv % m == big_n + 1 else c
    count = counts[0]
    last = max(i for i, q in enumerate(c.wires) if str(c.count(q)) == str(count))
    wires = [*c.wires[: last + 1], "_cnt", *c.wires[last + 1 :], "_acc"]
    gates = (
        ir.Init("_cnt", 1),
        ir.Init("_acc", 0),
        ir.CX("_cnt", "_acc"),
        ir.ZPhase("_acc", entries={big_n + 1: Fraction(1, 2)}),
        ir.CX("_cnt", "_acc", -1),
        ir.Post("_acc", 0),
        ir.Post("_cnt", 1),
    )
    regs = {**c.registers, "_cnt": count}
    dims = {**c.dims, "_cnt": m, "_acc": m}
    return replace(c, wires=tuple(wires), gates=(*c.gates, *gates), registers=regs, dims=dims)


def _e_antipode(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    return _append(c, ir.F(_wire(c, p), 2))


def _e_scalar(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    return _times(c, p["factor"])


def _e_tiny(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    return _append(c, ir.ZPhase(_wire(c, p), entries={1: f"1/d^{p['k']}"}))


def _e_swap(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    if "pair" in p:
        a, b = p["pair"]
    else:
        outs = _outputs(c)
        a, b = outs[p["i"]], outs[p["j"]]
    return _append(c, ir.Swap(a, b))


def _e_input_z(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    return _prepend(c, _wire(c, p), ir.Z(_wire(c, p), 1))


def _entry_outputs(c: ir.Circuit) -> list[str]:
    """Open single outputs that are not measured."""
    m = _measured(c)
    return [q for q in _outputs(c) if c.count(q) is None and q not in m]


def _e_common(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    outs = tuple(_entry_outputs(c))
    if not outs:
        return c

    def f(x: tuple[int, ...]) -> Fraction:
        return Fraction(1, 2) if all(v == 1 for v in x) else Fraction(0)

    return _append(c, ir.PhaseFn(outs, f, support=((1,) * len(outs),)))


def _gate_at(c: ir.Circuit, index: int, new: ir.Gate | None) -> ir.Circuit:
    gates = list(c.gates)
    if new is None:
        gates.pop(index)
    else:
        gates[index] = new
    return replace(c, gates=tuple(gates))


def _e_wrong_phase(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    g = c.gates[p["index"]]
    if isinstance(g, (ir.X, ir.Z)):
        return _gate_at(c, p["index"], replace(g, a=f"({g.a}) + 1"))
    if isinstance(g, (ir.ZPhase, ir.XPhase)):
        if g.rule is not None:
            return _gate_at(c, p["index"], replace(g, rule=f"({g.rule}) + k/(3*d)"))
        entries = dict(g.entries)
        key = min(entries) if entries else 1
        entries[key] = f"({entries.get(key, 0)}) + 1/3"
        return _gate_at(c, p["index"], replace(g, entries=entries))
    if isinstance(g, (ir.CX, ir.CZ)):
        return _gate_at(c, p["index"], replace(g, s=-g.s))
    raise ValueError("no phase at the index")


def _e_missing(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    return _gate_at(c, p["index"], None)


def _e_extra(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    return _append(c, ir.X(_wire(c, p), 1))


def _e_control(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    g = c.gates[p["index"]]
    return _gate_at(c, p["index"], replace(g, c=g.t, t=g.c))


def _e_off_by_one(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    if inst.equation.form == "unrolled":
        d_arg = dv if dv is not None else "d"
        return inst.equation.circuit(nv + 1, d_arg)
    regs = {q: f"({v}) + 1" for q, v in c.registers.items()}
    return replace(c, registers=regs)


def _e_dimension(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    dims = dict(c.dims)
    w = _wire(c, p)
    dims[w] = f"({c.dim(w)}) + 1"
    return replace(c, dims=dims)


def _e_coincident(c: ir.Circuit, p: dict[str, Any], inst: Any, nv: Any, dv: Any) -> ir.Circuit:
    return _append(c, ir.ZPhase(_wire(c, p), rule=p["rule"]))


EDITS: dict[str, Callable[..., ir.Circuit]] = {
    "beyond_oracle": _e_phase,
    "beyond_N": _e_beyond_n,
    "prime_only": _e_phase,
    "qubit_only": _e_antipode,
    "coincident_dims": _e_coincident,
    "scalar_small_one": _e_scalar,
    "global_phase": _e_scalar,
    "tiny_phase": _e_tiny,
    "leg_order": _e_swap,
    "box_boundary": _e_scalar,
    "state_not_map": _e_input_z,
    "single_entry": _e_common,
    "wrong_phase": _e_wrong_phase,
    "missing_gate": _e_missing,
    "extra_gate": _e_extra,
    "swapped_control_target": _e_control,
    "off_by_one_n": _e_off_by_one,
    "wrong_dimension": _e_dimension,
}
KINDS = tuple(EDITS)


@dataclass(frozen=True)
class Variant:
    """One candidate edit: its parameters, probe points and equal_where record."""

    label: str
    params: dict[str, Any]
    probe: list[tuple[int | None, int]]
    equal_where: dict[str, Any]


def _small(inst: Any, count: int = 3, d_from: int = 2) -> list[tuple[int | None, int]]:
    """Small domain points: the first ``count`` n from n_min, d from ``d_from``."""
    dom = inst.domain
    has_n = inst.subject.size_parameter is not None
    ns: list[int | None] = (
        [n for n in range(dom.n_min, dom.n_min + 40) if dom.has_n(n)][:count] if has_n else [None]
    )
    ds = _ds(inst, d_from, 2)
    return [(n, d) for d in ds for n in ns]


def _ds(
    inst: Any, start: int, count: int, pred: Callable[[int], bool] = lambda v: True
) -> list[int]:
    if not inst.subject.general_d:
        dv = int(inst.subject.dimension)
        return [dv] if pred(dv) and dv >= start else []
    out = []
    v = max(start, inst.domain.d_min)
    while len(out) < count and v <= min(inst.domain.d_max or 64, 64):
        if inst.domain.has_d(v) and pred(v):
            out.append(v)
        v += 1
    return out


def _ns(inst: Any, start: int, count: int) -> list[int | None]:
    if inst.subject.size_parameter is None:
        return [None]
    dom = inst.domain
    out: list[int | None] = []
    v = max(start, dom.n_min)
    while len(out) < count and (dom.n_max is None or v <= dom.n_max) and v < start + 200:
        if dom.has_n(v):
            out.append(v)
        v += 1
    return out


def _wires_for_phase(c: ir.Circuit) -> list[str]:
    m = _measured(c)
    outs = [q for q in _outputs(c) if q not in m]
    singles = [q for q in outs if c.count(q) is None]
    return singles[::-1] + [q for q in outs if q not in singles]


def variants(kind: str, inst: Any, cfg: dict[str, Any]) -> Iterator[Variant]:
    """Every candidate edit of ``kind`` on ``inst``, in the order tried; on an unrolled
    equation a named wire also carries its role list and slot."""
    raw = _variants(kind, inst, cfg)
    if inst.equation.form != "unrolled":
        yield from raw
        return
    lists = _role_lists(subject.circuit_at(inst, _probe_n(inst), _probe_d(inst)))
    for var in raw:
        q = var.params.get("wire")
        if q is not None:
            role = {"state_not_map": "inputs", "wrong_dimension": "all"}.get(kind, "phase")
            var = replace(var, params={**var.params, "role": role, "slot": lists[role].index(q)})
        yield var


def _variants(kind: str, inst: Any, cfg: dict[str, Any]) -> Iterator[Variant]:
    c = subject.circuit_at(inst, _probe_n(inst), _probe_d(inst))
    has_n = inst.subject.size_parameter is not None
    gen_d = inst.subject.general_d
    small = _small(inst)
    wires = _wires_for_phase(c)
    if kind == "beyond_oracle":
        if gen_d:
            rule = "k*(d-1)*(d-2)*(d-3)*(d-4)/d"
            probe = [(n, d) for n in _ns(inst, 0, 1) for d in _ds(inst, ORACLE_D + 1, 2)]
            for q in wires:
                for gen in ("Z", "X"):
                    yield Variant(
                        f"{gen} phase on {q}",
                        {"wire": q, "rule": rule, "gen": gen},
                        probe,
                        {"grid": "oracle"},
                    )
        if has_n and inst.equation.form == "boxed":
            rule = f"k*{_product_poly('n', ORACLE_N)}/({math.factorial(ORACLE_N + 1)}*d)"
            probe = [(n, d) for n in _ns(inst, ORACLE_N + 1, 1) for d in _ds(inst, 2, 2)]
            for q in wires:
                for gen in ("Z", "X"):
                    yield Variant(
                        f"{gen} n-phase on {q}",
                        {"wire": q, "rule": rule, "gen": gen},
                        probe,
                        {"grid": "oracle"},
                    )
    elif kind == "beyond_N":
        if not has_n or inst.equation.form == "fixed":
            return
        big_n = int(cfg["false_set"].get("beyond_N", 32))
        probe = [(n, d) for n in _ns(inst, big_n + 1, 1) for d in _ds(inst, 2, 1)]
        if inst.equation.form == "unrolled":
            for i in range(1, len(_outputs(c)) + 1):
                yield Variant(
                    f"X on output -{i} beyond N", {"index": -i, "N": big_n}, probe, {"n_max": big_n}
                )
        else:
            yield Variant("sign at n = N + 1 mod N + 2", {"N": big_n}, probe, {"n_max": big_n})
    elif kind == "prime_only":
        if not gen_d:
            return
        probe = [
            (n, d)
            for n in _ns(inst, 0, 1)
            for d in _ds(inst, 4, 2, lambda v: not subject.is_prime(v) and v > 1)
        ]
        for q in wires:
            for gen in ("Z", "X"):
                yield Variant(
                    f"{gen} Fermat phase on {q}",
                    {"wire": q, "rule": "(k^d - k)/d", "gen": gen},
                    probe,
                    {"d_prime": True, "also": list(CARMICHAEL)},
                )
    elif kind == "qubit_only":
        if not gen_d:
            return
        probe = [(n, d) for n in _ns(inst, 0, 1) for d in _ds(inst, 3, 2)]
        for q in wires:
            yield Variant(f"antipode on {q}", {"wire": q}, probe, {"d_max": 2})
    elif kind == "coincident_dims":
        dims = sorted({str(c.dim(q)) for q in c.wires})
        if len(dims) < 2:
            return
        for q in wires:
            other = next(x for x in dims if x != str(c.dim(q)))
            rule = f"k*(({c.dim(q)}) - ({other}))/({c.dim(q)})"
            yield Variant(
                f"dimension gap on {q}",
                {"wire": q, "rule": rule},
                small,
                {"dims_equal": [str(c.dim(q)), other]},
            )
    elif kind == "scalar_small_one":
        if gen_d:
            probe = [(n, d) for n in _ns(inst, 0, 1) for d in _ds(inst, ORACLE_D + 1, 1)]
            yield Variant(
                "d scalar", {"factor": "1 + (d-1)*(d-2)*(d-3)*(d-4)"}, probe, {"grid": "oracle"}
            )
        if has_n and inst.equation.form == "boxed":
            probe = [(n, d) for n in _ns(inst, ORACLE_N + 1, 1) for d in _ds(inst, 2, 1)]
            yield Variant(
                "n scalar",
                {"factor": f"1 + {_product_poly('n', ORACLE_N)}"},
                probe,
                {"grid": "oracle"},
            )
    elif kind == "global_phase":
        yield Variant("phase 1/7", {"factor": "exp(2*pi*I/7)"}, small, {"up_to_global_phase": True})
    elif kind == "tiny_phase":
        k = int(cfg["false_set"].get("tiny_k", 3))
        for q in wires:
            yield Variant(f"1/d^{k} on {q}", {"wire": q, "k": k, "d_min": 2}, small, {})
    elif kind == "leg_order":
        regs: dict[str, list[str]] = {}
        for q in _outputs(c):
            if c.count(q) is not None:
                regs.setdefault(str(c.count(q)), []).append(q)
        pairs = [(v[0], v[1]) for v in regs.values() if len(v) >= 2]
        for a, b in pairs:
            yield Variant(f"swap {a}, {b} per copy", {"pair": [a, b]}, small, {})
        outs = [q for q in _outputs(c) if c.count(q) is None]
        if inst.equation.form != "boxed" and len(outs) >= 2:
            yield Variant(
                "swap last two outputs",
                {"i": len(_outputs(c)) - 2, "j": len(_outputs(c)) - 1},
                small,
                {},
            )
    elif kind == "box_boundary":
        if not has_n or inst.equation.form != "boxed":
            return
        n0 = inst.domain.n_min
        probe = [(n0, d) for d in _ds(inst, 2, 2)]
        yield Variant(f"factor at n = {n0}", {"factor": f"1 + 0^(n - {n0})"}, probe, {"n_gt": n0})
    elif kind == "state_not_map":
        m = _measured(c)
        for q in [w for w in _inputs(c) if w not in m]:
            yield Variant(f"Z on input {q}", {"wire": q}, small, {"input_zero": q})
    elif kind == "single_entry":
        if _entry_outputs(c) and not any(c.count(q) is not None for q in _outputs(c)):
            yield Variant(
                "phase where every output is 1", {"d_min": 2}, small, {"outputs_not_all": 1}
            )
    elif kind == "wrong_phase":
        for i, g in enumerate(c.gates):
            if isinstance(g, (ir.X, ir.Z, ir.ZPhase, ir.XPhase, ir.CX, ir.CZ)):
                yield Variant(f"gate {i} {type(g).__name__}", {"index": i}, small, {})
    elif kind == "missing_gate":
        for i in range(len(c.gates) - 1, -1, -1):
            g = c.gates[i]
            if not isinstance(g, (ir.Init, ir.Post, ir.Discard, ir.Measure, ir.Swap)):
                yield Variant(f"drop gate {i} {type(g).__name__}", {"index": i}, small, {})
    elif kind == "extra_gate":
        for q in wires:
            yield Variant(f"X on {q}", {"wire": q}, small, {})
    elif kind == "swapped_control_target":
        for i, g in enumerate(c.gates):
            if isinstance(g, ir.CX):
                yield Variant(f"gate {i} CX reversed", {"index": i}, small, {})
    elif kind == "off_by_one_n":
        if has_n and inst.equation.form != "fixed":
            yield Variant("n + 1 in the circuit", {}, small, {})
    elif kind == "wrong_dimension":
        ctrl = {
            q
            for g in c.gates
            if not isinstance(
                g,
                (ir.Init, ir.Post, ir.Discard, ir.Measure, ir.X, ir.Z, ir.F, ir.ZPhase, ir.XPhase),
            )
            for q in ir.gate_wires(g)
        }
        for q in c.wires:
            if q not in ctrl and inst.subject.general_d:
                yield Variant(f"dimension + 1 on {q}", {"wire": q}, small, {})


def _probe_n(inst: Any) -> int | None:
    if inst.subject.size_parameter is None:
        return None
    return next(iter(_ns(inst, 0, 1)), inst.domain.n_min)


def _probe_d(inst: Any) -> int | None:
    if not inst.subject.general_d:
        return int(inst.subject.dimension)
    ds = _ds(inst, 2, 1) or _ds(inst, 1, 1)
    return ds[0] if ds else inst.domain.d_min


# -- witness search --------------------------------------------------------------------------


def _phase_vector(dv: int, v: int, sign: int) -> Any:
    from archytaszx.algebra.dimension import Dim
    from archytaszx.algebra.phase import Phase, PhaseVector

    entries = {
        k: Phase.turns(sp.Rational(sign * v * k, dv) % 1) for k in range(1, dv) if (v * k) % dv
    }
    return PhaseVector(Dim(dv), entries)


def plugged(diagram: Any, outs: tuple[int, ...], ins: tuple[int, ...]) -> complex:
    """The entry <outs| D |ins> of a concrete pure diagram, by plugging basis effects and
    states and contracting the closed network."""
    from archytaszx.algebra.dimension import Dim
    from archytaszx.diagram.generators import REGISTRY
    from archytaszx.diagram.graph import Direction, PortRef
    from archytaszx.semantics.contract_numeric import contract
    from archytaszx.semantics.denote import leg_dims

    d2 = diagram.copy()
    xg = REGISTRY.get("X")
    norm = 1.0

    def dim_of(ref: Any) -> int:
        node = d2.nodes[ref.node_id]
        legs = node.outputs if ref.direction is Direction.OUTPUT else node.inputs
        return int(d2.resolve_dim(legs[ref.index].dim).to_int())

    for ref, v in zip(list(d2.boundary_outputs), outs):
        dv = dim_of(ref)
        nid = d2.add_node(xg, [Dim(dv)], [], _phase_vector(dv, v, 1))
        d2.add_wire(ref, PortRef(nid, Direction.INPUT, 0))
        norm /= math.sqrt(dv)
    for ref, v in zip(list(d2.boundary_inputs), ins):
        dv = dim_of(ref)
        nid = d2.add_node(xg, [], [Dim(dv)], _phase_vector(dv, v, -1))
        d2.add_wire(ref, PortRef(nid, Direction.OUTPUT, 0))
        norm /= math.sqrt(dv)
    d2.set_boundary_outputs([])
    d2.set_boundary_inputs([])
    _ = leg_dims
    t = contract(d2).tensor
    return complex(np.asarray(t).reshape(-1)[0]) * norm


def _pure(diagram: Any, doubled: bool = False) -> Any:
    """The instantiated diagram, doubled when it is mixed or ``doubled`` is set."""
    from archytaszx.diagram.mixed import double
    from archytaszx.semantics import check

    inst = check.instantiate(diagram, {})
    return inst if inst.is_pure() and not doubled else double(inst)


def _boundary_dims(d: Any) -> tuple[list[int], list[int]]:
    from archytaszx.diagram.graph import Direction

    def dims(refs: Any) -> list[int]:
        out = []
        for ref in refs:
            node = d.nodes[ref.node_id]
            legs = node.outputs if ref.direction is Direction.OUTPUT else node.inputs
            out.append(int(d.resolve_dim(legs[ref.index].dim).to_int()))
        return out

    return dims(d.boundary_outputs), dims(d.boundary_inputs)


@dataclass
class Witness:
    """A differing entry: the point, output and input indices (of the doubled diagram for a
    mixed equation), both values, or an interface difference; ``proportional`` says whether
    the pair differs only by a scalar there (None when no dense check ran)."""

    n: int | None
    d: int
    outs: list[int]
    ins: list[int]
    left: list[float]
    right: list[float]
    interface: str | None = None
    method: str = "entry"
    proportional: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "d": self.d,
            "outs": self.outs,
            "ins": self.ins,
            "left": self.left,
            "right": self.right,
            "interface": self.interface,
            "method": self.method,
            "proportional": self.proportional,
        }


def _pair(inst: Any, n: int | None, d: int) -> tuple[Any, Any]:
    """Both diagrams of ``inst`` at (n, d), as loaded (environment bound)."""
    from benchmarks import stages

    sc = subject.script(inst, n, d)
    return stages.diagrams(sc.text)


def _symbolic_entry(
    diagram: Any, outs: tuple[int, ...], ins: tuple[int, ...], doubled: bool = False
) -> complex:
    """One entry by symbolic contraction at the diagram's bound values, of the doubled diagram
    when it is mixed or ``doubled`` is set."""
    from archytaszx.diagram.mixed import double
    from archytaszx.semantics.contract_symbolic import contract_symbolic

    d = diagram if diagram.is_pure() and not doubled else double(diagram)
    t = contract_symbolic(d).substitute(dict(d.parameters))
    return complex(sp.N(t.value_at((*outs, *ins)).to_sympy()))


def _candidates(
    out_dims: list[int], in_dims: list[int], rng: random.Random
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    seen = set()
    for fill in (0, 1, -1):
        o = tuple(fill % v for v in out_dims)
        i = tuple(fill % v for v in in_dims)
        for cand in ((o, i), (o, tuple(0 for _ in in_dims)), (tuple(0 for _ in out_dims), i)):
            if cand not in seen:
                seen.add(cand)
                yield cand
    for _ in range(24):
        cand = (tuple(rng.randrange(v) for v in out_dims), tuple(rng.randrange(v) for v in in_dims))
        if cand not in seen:
            seen.add(cand)
            yield cand


def proportional(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether a = lambda b for one scalar lambda (both nonzero, within TOL entrywise)."""
    j = int(np.argmax(np.abs(b)))
    if abs(b[j]) <= TOL or float(np.max(np.abs(a))) <= TOL:
        return False
    lam = a[j] / b[j]
    return bool(np.allclose(a, lam * b, atol=TOL, rtol=0.0))


def differing_entry(
    raw_left: Any, raw_right: Any, n: int | None, d: int, rng: random.Random
) -> Witness | None:
    """A differing entry of two loaded diagrams, both doubled when either is mixed: dense when
    small, else candidate index tuples by plugging (small nodes) or symbolic contraction."""
    from archytaszx.semantics import check

    mixed = not (raw_left.is_pure() and raw_right.is_pure())
    left, right = _pure(raw_left, mixed), _pure(raw_right, mixed)
    lo, li = _boundary_dims(left)
    ro, ri = _boundary_dims(right)
    if (lo, li) != (ro, ri):
        return Witness(
            n,
            d,
            [],
            [],
            [],
            [],
            interface=f"left {lo}->{li}, right {ro}->{ri}",
            proportional=False,
        )
    size = math.prod(lo) * math.prod(li)
    if size <= DENSE_LIMIT:
        a = np.asarray(check.contract(left).tensor).reshape(-1)
        b = np.asarray(check.contract(right).tensor).reshape(-1)
        diff = np.abs(a - b)
        k = int(np.argmax(diff))
        if diff[k] <= TOL:
            return None
        idx = np.unravel_index(k, [*lo, *li]) if lo or li else ()
        idx = [int(v) for v in idx]
        return Witness(
            n,
            d,
            idx[: len(lo)],
            idx[len(lo) :],
            [a[k].real, a[k].imag],
            [b[k].real, b[k].imag],
            proportional=proportional(a, b),
        )

    def arity(dg: Any) -> int:
        return max((len(nd.inputs) + len(nd.outputs) for nd in dg.nodes.values()), default=0)

    narrow = {id(left): arity(left) <= 12, id(right): arity(right) <= 12}

    def entry(raw: Any, inst: Any, outs: tuple[int, ...], ins: tuple[int, ...]) -> complex:
        if narrow[id(inst)]:
            return plugged(inst, outs, ins)
        return _symbolic_entry(raw, outs, ins, mixed)

    for outs, ins in _candidates(lo, li, rng):
        x, y = entry(raw_left, left, outs, ins), entry(raw_right, right, outs, ins)
        if abs(x - y) > TOL:
            return Witness(n, d, list(outs), list(ins), [x.real, x.imag], [y.real, y.imag])
    return None


@dataclass
class FalseEquation:
    """One F record: the S equation's id, the edit and its witness."""

    equation: str
    edit: Edit
    witness: dict[str, Any]
    equal_where: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation": self.equation,
            "kind": self.edit.kind,
            "label": self.edit.label,
            "params": self.edit.params,
            "witness": self.witness,
            "equal_where": self.equal_where,
        }

    @classmethod
    def from_dict(cls, r: dict[str, Any]) -> FalseEquation:
        return cls(
            r["equation"], Edit(r["kind"], r["label"], r["params"]), r["witness"], r["equal_where"]
        )


def gadget_value(params: dict[str, Any], n: int) -> complex:
    """The scalar of the boxed beyond_N gadget at count n: -1 when n = N + 1 mod N + 2."""
    m = params["N"] + 2
    return -1.0 if n % m == params["N"] + 1 else 1.0


def gadget_diagram(params: dict[str, Any], n: int) -> Any:
    """The boxed beyond_N gadget alone, loaded with count n bound."""
    from benchmarks import stages, translate

    fake = type("I", (), {"equation": type("E", (), {"form": "boxed"})()})()
    base = ir.Circuit(("r",), (ir.Init("r"), ir.Post("r")), {"r": "n"}, {"r": 2}, "1")
    c = _e_beyond_n(base, params, fake, None, None)
    body = translate.translate(c, translate.Context()).body
    run_ = stages.new_run()
    text = "load\n" + "\n".join("    " + ln for ln in [f"count n = {n}", *body.splitlines()])
    out = stages.execute(run_, text)
    if not out.ok:
        raise RuntimeError(stages.text_of(out)[:200])
    return run_.session.current


def _factored_witness(
    broken: Any, params: dict[str, Any], n: int, d: int, rng: random.Random
) -> Witness | None:
    """The boxed beyond_N witness: a nonzero entry e of the specification at (n, d) and the
    gadget's value g there (|g|^2 on doubled diagrams when either side is mixed); the broken
    circuit's entry is g e."""
    left, right = _pair(broken, n, d)
    mixed = not (left.is_pure() and right.is_pure())
    g = gadget_value(params, n)
    g = abs(g) ** 2 if mixed else g
    if abs(g - 1) <= TOL:
        return None
    pr = _pure(right, mixed)
    ro, ri = _boundary_dims(pr)
    for outs, ins in [*_extrapolated(broken, n, d, len(ro)), *_candidates(ro, ri, rng)]:
        e = _symbolic_entry(right, outs, ins, mixed)
        if abs(e) > TOL:
            x = g * e
            return Witness(
                n,
                d,
                list(outs),
                list(ins),
                [x.real, x.imag],
                [e.real, e.imag],
                method="factored",
                proportional=True,
            )
    return None


def _first_nonzero(diagram: Any, doubled: bool = False) -> tuple[list[int], int] | None:
    """The first nonzero entry's index (outputs then inputs) and the output count, densely."""
    from archytaszx.semantics import check

    p = _pure(diagram, doubled)
    lo, li = _boundary_dims(p)
    if math.prod(lo) * math.prod(li) > DENSE_LIMIT:
        return None
    a = np.asarray(check.contract(p).tensor).reshape(-1)
    nz = np.flatnonzero(np.abs(a) > TOL)
    if not len(nz):
        return None
    return [int(v) for v in np.unravel_index(int(nz[0]), [*lo, *li])], len(lo)


def _extrapolated(
    broken: Any, n: int, d: int, outs_n: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Index tuples at count n extended from the first nonzero entries of the specification
    at two small counts, one block per copy."""
    small = [v for v in _ns(broken, 1, 2) if v is not None and v < n]
    if len(small) < 2:
        return []
    pats = []
    for v in small:
        left, right = _pair(broken, v, d)
        hit = _first_nonzero(right, not (left.is_pure() and right.is_pure()))
        if hit is None:
            return []
        pats.append((v, *hit))
    (n1, p1, _), (n2, p2, _) = pats
    step = n2 - n1
    b = (len(p2) - len(p1)) // step if step else 0
    if b <= 0:
        return []
    out = []
    for a in range(len(p1) - b * n1 + 1):
        block = p1[a : a + b]
        if p1[a : a + b * n1] != block * n1:
            continue
        if p1[:a] + block * n2 + p1[a + b * n1 :] != p2:
            continue
        full = p1[:a] + block * n + p1[a + b * n1 :]
        out.append((tuple(full[:outs_n]), tuple(full[outs_n:])))
    return out


_OVERFLOWS: list[str] = []


class _Overrun(BaseException):
    """A probe or an equation ran past its time budget."""


@contextmanager
def _limit(seconds: float) -> Iterator[None]:
    """Raises :class:`_Overrun` in the block after ``seconds`` (no limit when not positive)."""
    if seconds <= 0:
        yield
        return

    def fire(signum: int, frame: Any) -> None:
        raise _Overrun

    old = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def try_kind(
    kind: str, inst: Any, cfg: dict[str, Any], rng: random.Random, deadline: float = math.inf
) -> FalseEquation | None:
    """The first variant of ``kind`` with a witness, as an F record; each probe is limited to
    false_set.probe_seconds, and :class:`_Overrun` is raised past ``deadline``."""
    probe_seconds = float(cfg["false_set"].get("probe_seconds", 0))
    for var in variants(kind, inst, cfg):
        edit = Edit(kind, var.label, var.params)
        broken = subject.Instance(inst.subject, inst.equation, inst.restriction, edit)
        for n, d in var.probe:
            left_time = deadline - time.monotonic()
            if left_time <= 0:
                raise _Overrun
            budget = min(probe_seconds, left_time) if probe_seconds > 0 else left_time
            try:
                with _limit(budget if math.isfinite(budget) else 0):
                    if kind == "beyond_N" and inst.equation.form == "boxed":
                        w = _factored_witness(broken, var.params, n, d, rng)
                    else:
                        left, right = _pair(broken, n, d)
                        w = differing_entry(left, right, n, d, rng)
            except (_Overrun, Exception) as exc:  # noqa: BLE001
                if type(exc).__name__ in ("ContractSizeError", "MemoryError"):
                    _OVERFLOWS.append(inst.id)
                continue
            if w is not None:
                return FalseEquation(inst.id, edit, w.to_dict(), var.equal_where)
    return None


def kinds_allowed(cfg: dict[str, Any]) -> list[str]:
    """The registry kinds the config allows, in registry order."""
    import fnmatch

    pats = cfg["false_set"]["kinds"]
    return [k for k in KINDS if any(fnmatch.fnmatch(k, p) for p in pats)]


def order_for(index: int, eq_id: str, cfg: dict[str, Any]) -> list[str]:
    """The kinds tried for the ``index``-th S equation: round robin from index, or a seeded
    shuffle."""
    kinds = kinds_allowed(cfg)
    if not kinds:
        return []
    if cfg["false_set"]["assignment"] == "random":
        h = int.from_bytes(
            hashlib.sha256(f"{cfg['false_set']['seed']}|{eq_id}".encode()).digest()[:8], "big"
        )
        rng = random.Random(h)
        out = kinds[:]
        rng.shuffle(out)
        return out
    start = index % len(kinds)
    return kinds[start:] + kinds[:start]


def generate_one(job: tuple[int, str, dict[str, Any]]) -> dict[str, Any]:
    """The F record of the ``index``-th S equation, or a record of why none applies."""
    index, eq_id, cfg = job
    insts = {i.id: i for i in subject.instances(_selected(cfg))}
    inst = insts[eq_id]
    seed = int.from_bytes(
        hashlib.sha256(f"{cfg['false_set']['seed']}|{eq_id}".encode()).digest()[:8], "big"
    )
    rng = random.Random(seed)
    tried = []
    seconds = float(cfg["false_set"].get("equation_seconds", 0))
    deadline = time.monotonic() + seconds if seconds > 0 else math.inf
    for kind in order_for(index, eq_id, cfg):
        try:
            rec = try_kind(kind, inst, cfg, rng, deadline)
        except _Overrun:
            return {"equation": eq_id, "kind": None, "skipped": tried, "reason": "time budget"}
        if rec is not None:
            return {**rec.to_dict(), "skipped": tried}
        tried.append(kind)
    rec: dict[str, Any] = {"equation": eq_id, "kind": None, "skipped": tried}
    if eq_id in _OVERFLOWS:
        rec["reason"] = "oracle size"
    return rec


def _selected(cfg: dict[str, Any]) -> list[Any]:
    sc = cfg["subjects"]
    return subject.select(
        subject.catalogue(sc["catalogue"]), sc["include"], sc["exclude"], sc["categories"]
    )


def generate(cfg: dict[str, Any], path: Path, jobs: int) -> list[dict[str, Any]]:
    """Every F record, read from ``path`` when it holds this config's set, else generated in
    parallel and written there."""
    from concurrent.futures import ProcessPoolExecutor

    key = fingerprint(cfg)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("fingerprint") == key:
            return data["records"]
    eqs = [i.id for i in subject.instances(_selected(cfg))]
    work = [(k, e, cfg) for k, e in enumerate(eqs)]
    with ProcessPoolExecutor(max(1, jobs)) as pool:
        records = list(pool.map(generate_one, work, chunksize=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"fingerprint": key, "records": records}, indent=1, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)
    return records


def fingerprint(cfg: dict[str, Any]) -> str:
    """A hash of the subject sources and the subjects and false_set config."""
    h = hashlib.sha256()
    h.update(json.dumps({k: cfg[k] for k in ("subjects", "false_set")}, sort_keys=True).encode())
    base = Path(cfg["subjects"]["catalogue"])
    base = base if base.is_absolute() else subject.ROOT / base
    for p in sorted(base.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    for mod in ("ir.py", "translate.py", "breaks.py", "subject.py"):
        h.update((Path(__file__).parent / mod).read_bytes())
    return h.hexdigest()[:16]


def false_instance(rec: dict[str, Any], insts: dict[str, Any]) -> Any:
    """The F instance of an F record."""
    fe = FalseEquation.from_dict(rec)
    s = insts[rec["equation"]]
    return subject.Instance(s.subject, s.equation, s.restriction, fe.edit)


def oracle_grid(inst: Any) -> list[tuple[int | None, int]]:
    """The domain points with d <= 4 and n <= 3."""
    has_n = inst.subject.size_parameter is not None
    ns: list[int | None] = (
        [n for n in range(ORACLE_N + 1) if inst.domain.has_n(n)] if has_n else [None]
    )
    ds = (
        [d for d in range(1, ORACLE_D + 1) if inst.domain.has_d(d)]
        if inst.subject.general_d
        else [int(inst.subject.dimension)]
    )
    return [(n, d) for n in ns for d in ds]


__all__ = ["KINDS", "Edit", "FalseEquation", "generate", "product"]
