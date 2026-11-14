"""MQT QCEC driver, run as ``python -I qcec.py PAYLOAD``: both QASM circuits loaded as
QuantumComputations; when both list the same ancillary wires, the circuit followed by the
inverse reference (miter) against the empty circuit instead; the listed wires marked ancillary;
verify for full equivalence by the decision-diagram alternating and construction checkers only
(no simulation or ZX checker);
equivalent is equal, not_equivalent unequal, equivalent up to a global phase equal, up to a
relative phase unequal, anything else unknown. On equivalent up to a global phase, after the
timed check, ``global_phase`` is the phase of <0|B^-1 A|0> by decision-diagram simulation in a
child process limited to PHASE_SECONDS (None past it) and up_to_global_phase_only says whether
it exceeds PHASE_TOL (None when unread). Prints one JSON line."""

import cmath
import json
import multiprocessing
import sys
import time

PHASE_TOL = 1e-9
PHASE_SECONDS = 30.0


def _phase_child(qasm, reference, conn):
    from mqt.core import dd
    from mqt.core.ir import QuantumComputation

    a = QuantumComputation.from_qasm_str(qasm)
    b = QuantumComputation.from_qasm_str(reference)
    b.invert()
    for op in b:
        a.append(op)
    pkg = dd.DDPackage(a.num_qubits)
    v = dd.simulate(a, pkg.zero_state(a.num_qubits), pkg)
    amp = complex(v.get_amplitude(a.num_qubits, "0" * a.num_qubits))
    conn.send(cmath.phase(amp) if abs(amp) > 0.5 else None)


def global_phase(qasm, reference):
    """The phase of <0|B^-1 A|0> in a forked child, None past PHASE_SECONDS or on failure."""
    ctx = multiprocessing.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_phase_child, args=(qasm, reference, send))
    proc.start()
    send.close()
    out = recv.recv() if recv.poll(PHASE_SECONDS) else None
    if proc.is_alive():
        proc.kill()
    proc.join()
    return out


def main():
    sys.set_int_max_str_digits(0)
    from mqt.core.ir import QuantumComputation
    from mqt.qcec import verify

    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    c0, w0 = time.process_time(), time.perf_counter()
    a = QuantumComputation.from_qasm_str(p["qasm"])
    b = QuantumComputation.from_qasm_str(p["reference"])
    miter = bool(p["ancillary"]) and p["ancillary"] == p["reference_ancillary"]
    if miter:
        b.invert()
        for op in b:
            a.append(op)
        b = QuantumComputation(a.num_qubits)
    for q in p["ancillary"]:
        a.set_circuit_qubit_ancillary(q)
    for q in p["reference_ancillary"]:
        b.set_circuit_qubit_ancillary(q)
    res = verify(
        a,
        b,
        check_partial_equivalence=False,
        run_simulation_checker=False,
        run_zx_checker=False,
    )
    name = str(res.equivalence).split(".")[-1].lower()
    verdict = {
        "equivalent": "equal",
        "not_equivalent": "unequal",
        "equivalent_up_to_global_phase": "equal",
        "equivalent_up_to_phase": "unequal",
        "probably_not_equivalent": "unknown",
    }.get(name, "unknown")
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    phase, only = None, False
    if name == "equivalent_up_to_global_phase":
        try:
            phase = global_phase(p["qasm"], p["reference"])
        except EOFError:
            phase = None
        only = None if phase is None else abs(phase) > PHASE_TOL
    out = {
        "verdict": verdict,
        "equivalence": name,
        "up_to_global_phase_only": only,
        "global_phase": phase,
        "miter": miter,
        "cpu": cpu,
        "wall": wall,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
