"""ZX driver for PyZX and QuiZX, run as ``python -I zx.py PAYLOAD`` in the tool's environment.

The circuit is read from QASM, its Init and Post wires plugged with basis states and effects,
the specification read from PyZX JSON. A map is checked by full reduction of spec^dagger . circuit
and spec^dagger . spec (and circuit^dagger . circuit when inputs and outputs differ in number) to
multiples of the identity (every input meets its output by a plain wire; unknown otherwise); a
state by the three closed inner products <s|c>, <c|c>, <s|s> (unknown when one does not reduce
to a scalar). Both tests are exact: the circuit's global phase
and the specification's scalar (payload "phase" and rhs "scalar") are applied, a state is equal
when <s|c> = <c|c> = <s|s> (states_equal) and a map when the scalar of spec^dagger . circuit
equals that of spec^dagger . spec (map_scalar_match, with gram_match for a non-square map), each
value compared from the scalars' stored parts (in log space beyond float range);
equal_up_to_scalar records the comparison without them.
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
    else:
        import quizx

        lhs = quizx.qasm(p["qasm"])
        rhs = _build(quizx.VecGraph(), p["rhs"])
        reduce = quizx.full_simp

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
        s_cs = _ip(lhs, rhs, reduce)
        s_cc = _ip(lhs, lhs, reduce)
        s_ss = _ip(rhs, rhs, reduce)
        out["mode"] = "inner products"
        if None in (s_cs, s_cc, s_ss):
            out["verdict"] = "unknown"
            out["reason"] = "an inner product did not reduce to a scalar"
        else:
            out["equal_up_to_scalar"] = norm_identity(s_cs, s_cc, s_ss)
            out["verdict"] = "equal" if states_equal(s_cs, s_cc, s_ss, p) else "unequal"
    else:
        g, g_id = _gram(lhs, rhs, reduce)
        out["mode"] = "identity"
        r, r_id = _gram(rhs, rhs, reduce) if g_id else (None, False)
        l_, l_id = _gram(lhs, lhs, reduce) if r_id and nin != nout else (None, True)
        if not g_id:
            out["verdict"] = "unknown"
        elif not r_id or r.scalar.is_zero:
            out["verdict"] = "unknown"
            out["reason"] = "spec^dagger . spec did not reduce to a nonzero identity multiple"
        elif not l_id:
            out["verdict"] = "unknown"
            out["reason"] = "circuit^dagger . circuit did not reduce to a multiple of the identity"
        else:
            up = l_ is None or gram_match(g.scalar, r.scalar, l_.scalar)
            out["equal_up_to_scalar"] = up
            same = up and map_scalar_match(g.scalar, r.scalar, p)
            out["verdict"] = "equal" if same else "unequal"
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


LOG_LIMIT = 600.0
LOG_TOL = 1e-9
PHASE_TOL = 1e-6


def _ip(a, b, reduce):
    """<b|a> of two states as the closed graph b^dagger . a, fully reduced, as the graph's
    PyZX Scalar; None when vertices remain."""
    g = a.copy()
    g.compose(b.adjoint())
    reduce(g)
    return g.scalar if g.num_vertices() == 0 else None


def scalar_parts(s):
    """(zero, log|s|, arg s) of a PyZX Scalar from its stored parts (power2 exactly, the phase,
    phase nodes, sum of phases and float factor as to_number combines them), without
    to_number."""
    import cmath
    import math

    if s.is_zero:
        return True, -math.inf, 0.0
    logm = s.power2 * 0.5 * math.log(2.0)
    ang = math.pi * float(s.phase)
    factors = [complex(s.floatfactor)]
    factors += [1 + cmath.exp(1j * math.pi * float(p)) for p in s.phasenodes]
    if s.sum_of_phases:
        t = sum(c * cmath.exp(1j * math.pi * float(p)) for p, c in s.sum_of_phases.items())
        if t != 0:
            factors.append(t)
    for z in factors:
        if z == 0:
            return True, -math.inf, 0.0
        logm += math.log(abs(z))
        ang += cmath.phase(z)
    return False, logm, ang


def _shift(parts, log_add, arg_add, zero_factor=False):
    """``parts`` times a factor of log magnitude ``log_add`` and argument ``arg_add`` (zero
    when ``zero_factor``)."""
    zero, lg, ang = parts
    return zero or zero_factor, lg + log_add, ang + arg_add


def _same(x, y):
    """Whether two (zero, log, arg) values are equal: as complex numbers (relative 1e-9) when
    both are within LOG_LIMIT or zero, else in log space (LOG_TOL, PHASE_TOL)."""
    import cmath
    import math

    if all(z or abs(lg) <= LOG_LIMIT for z, lg, _ in (x, y)):
        a, b = (0j if z else cmath.rect(math.exp(lg), ang) for z, lg, ang in (x, y))
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    if x[0] or y[0]:
        return x[0] and y[0]
    return (
        abs(x[1] - y[1]) <= LOG_TOL and abs(math.remainder(x[2] - y[2], 2 * math.pi)) <= PHASE_TOL
    )


def states_equal(cs, cc, ss, p):
    """Whether the circuit's state c equals the specification's s exactly: <s|c> = <c|c> =
    <s|s>, the circuit's global phase phi, its scalar's magnitude m and the specification's
    scalar lambda applied (<s|c> m e^(i phi) conj(lambda), <c|c> m^2, <s|s> |lambda|^2)."""
    zero, lam_log, lam_arg = p["rhs"]["scalar"]
    c_zero, c_log = p.get("circuit_scalar", [False, 0.0])
    phi = float(p.get("phase", 0.0))
    a = _shift(scalar_parts(cs), lam_log + c_log, phi - lam_arg, zero or c_zero)
    b = _shift(scalar_parts(cc), 2 * c_log, 0.0, c_zero)
    c = _shift(scalar_parts(ss), 2 * lam_log, 0.0, zero)
    return _same(a, b) and _same(b, c)


def _gram(a, b, reduce):
    """(b^dagger . a fully reduced, whether it is a plain-wire identity)."""
    g = a.copy()
    g.compose(b.adjoint())
    reduce(g)
    return g, _identity(g)


def map_scalar_match(sigma, rho, p):
    """Whether sigma m e^(i phi) conj(lambda) = |lambda|^2 rho, sigma the scalar of the reduced
    spec^dagger . circuit, rho that of spec^dagger . spec (m the circuit scalar's magnitude)."""
    zero, lam_log, lam_arg = p["rhs"]["scalar"]
    c_zero, c_log = p.get("circuit_scalar", [False, 0.0])
    if zero or c_zero:
        return False
    phi = float(p.get("phase", 0.0))
    left = _shift(scalar_parts(sigma), lam_log + c_log, phi - lam_arg)
    right = _shift(scalar_parts(rho), 2 * lam_log, 0.0)
    return _same(left, right)


def gram_match(sigma, rho, tau):
    """Whether tau rho = |sigma|^2 for the scalars of spec^dagger . circuit (sigma),
    spec^dagger . spec (rho) and circuit^dagger . circuit (tau)."""
    zs, ls, _ = scalar_parts(sigma)
    zr, lr, ar = scalar_parts(rho)
    left = _shift(scalar_parts(tau), lr, ar, zr)
    return _same(left, (zs, 2 * ls, 0.0))


def norm_identity(cs, cc, ss):
    """Whether |cs|^2 = cc ss for three PyZX Scalars: as complex numbers (relative 1e-9) when
    every |log| is at most LOG_LIMIT, else in log space (log magnitudes within LOG_TOL, the
    phase of cc ss within PHASE_TOL of 0, exact zeros compared exactly)."""
    import math

    parts = [scalar_parts(s) for s in (cs, cc, ss)]
    if all(z or abs(lg) <= LOG_LIMIT for z, lg, _ in parts):
        a, b, c = (complex(s.to_number()) for s in (cs, cc, ss))
        return abs(abs(a) ** 2 - (b * c).real) <= 1e-9 * max(1.0, abs(b * c))
    (za, la, _), (zb, lb, ab), (zc, lc, ac) = parts
    if za or zb or zc:
        return za and (zb or zc)
    return (
        abs(2 * la - lb - lc) <= LOG_TOL and abs(math.remainder(ab + ac, 2 * math.pi)) <= PHASE_TOL
    )


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
