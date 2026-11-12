"""ZX driver for PyZX and QuiZX, run as ``python -I zx.py PAYLOAD`` in the tool's environment.

The circuit is read from QASM, its Init and Post wires plugged with basis states and effects,
the specification read from PyZX JSON. A map is checked by full reduction of circuit . spec^dagger
to the identity (equal when every input meets its output by a plain wire, unknown otherwise); a
state by the three closed inner products <s|c>, <c|c>, <s|s> (equal when |<s|c>|^2 = <c|c><s|s>,
unequal otherwise, unknown when one does not reduce to a scalar). Scalars are not compared.
Prints one JSON line.
"""

import json
import sys
import time


def main() -> None:
    sys.set_int_max_str_digits(0)
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    tool = p["tool"]
    c0, w0 = time.process_time(), time.perf_counter()
    if tool == "pyzx":
        import pyzx as zx

        lhs = zx.Circuit.from_qasm(p["qasm"]).to_graph()
        rhs = _build(zx.Graph(), p["rhs"])
        reduce = zx.full_reduce

        def number(g):
            return complex(g.scalar.to_number())
    else:
        import quizx

        lhs = quizx.qasm(p["qasm"])
        rhs = _build(quizx.VecGraph(), p["rhs"])
        reduce = quizx.full_simp

        def number(g):
            return complex(g.scalar.to_number())

    m = p["wires"]
    if p["inits"]:
        lhs.apply_state(
            "".join(str(p["inits"][str(i)]) if str(i) in p["inits"] else "/" for i in range(m))
        )
    if p["posts"]:
        lhs.apply_effect(
            "".join(str(p["posts"][str(i)]) if str(i) in p["posts"] else "/" for i in range(m))
        )
    nin, nout = len(lhs.inputs()), len(lhs.outputs())
    out = {"vertices_before": lhs.num_vertices()}
    if (nin, nout) != (m - len(p["inits"]), m - len(p["posts"])):
        out["verdict"] = "unknown"
        out["reason"] = "the tool read the circuit with another boundary"
    elif (nin, nout) != (len(rhs.inputs()), len(rhs.outputs())):
        out["verdict"] = "unequal"
        out["reason"] = "interface"
    elif nin == 0:
        s_cs = _ip(lhs, rhs, reduce, number)
        s_cc = _ip(lhs, lhs, reduce, number)
        s_ss = _ip(rhs, rhs, reduce, number)
        out["mode"] = "inner products"
        if None in (s_cs, s_cc, s_ss):
            out["verdict"] = "unknown"
            out["reason"] = "an inner product did not reduce to a scalar"
        else:
            same = abs(abs(s_cs) ** 2 - (s_cc * s_ss).real) <= 1e-9 * max(1.0, abs(s_cc * s_ss))
            out["verdict"] = "equal" if same else "unequal"
    else:
        g = lhs.copy()
        g.compose(rhs.adjoint())
        reduce(g)
        out["verdict"] = "equal" if _identity(g) else "unknown"
        out["mode"] = "identity up to scalar"
        lhs = g
    out["cpu"] = time.process_time() - c0
    out["wall"] = time.perf_counter() - w0
    reduce(lhs)
    out["vertices_after"] = lhs.num_vertices()
    out["edges_after"] = lhs.num_edges()
    print(json.dumps(out))


def _build(g, spec):
    """The specification graph in the tool's own graph class."""
    from fractions import Fraction

    kinds = {"B": 0, "Z": 1, "X": 2}
    ids = []
    for k, (kind, phase) in enumerate(spec["vertices"]):
        ids.append(g.add_vertex(kinds[kind], k, k, Fraction(phase)))
    for a, b, et in spec["edges"]:
        g.add_edge((ids[a], ids[b]), et)
    g.set_inputs(tuple(ids[i] for i in spec["inputs"]))
    g.set_outputs(tuple(ids[i] for i in spec["outputs"]))
    return g


def _ip(a, b, reduce, number):
    """<b|a> of two states as the closed graph b^dagger . a, fully reduced; None when vertices
    remain."""
    g = a.copy()
    g.compose(b.adjoint())
    reduce(g)
    return number(g) if g.num_vertices() == 0 else None


def _identity(g) -> bool:
    ins, outs = list(g.inputs()), list(g.outputs())
    if len(ins) != len(outs):
        return False
    for i, o in zip(ins, outs):
        nb = list(g.neighbors(i))
        if nb != [o]:
            return False
        if int(g.edge_type(g.edge(i, o))) != 1:
            return False
    return True


if __name__ == "__main__":
    main()
