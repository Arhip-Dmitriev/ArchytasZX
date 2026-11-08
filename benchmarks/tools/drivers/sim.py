"""Simulation driver for Qiskit and Cirq, run as ``python -I sim.py PAYLOAD`` in the tool's
environment: the circuit's state (every input initialised) or unitary, sliced at its Init
inputs and Post outputs, compared with the specification tensor (outputs then inputs, first
wire most significant): exactly, and up to a global phase. Prints one JSON line."""

import json
import sys
import time

import numpy as np


def qiskit_tensor(p):
    from qiskit import qasm2
    from qiskit.quantum_info import Operator, Statevector

    qc = qasm2.loads(p["qasm"])
    qc.global_phase = p["global_phase"]
    m = p["wires"]
    if len(p["inits"]) == m:
        for i, v in p["inits"].items():
            if v:
                qc_x = qc.copy_empty_like()
                qc_x.global_phase = 0
                qc_x.x(int(i))
                qc = qc_x.compose(qc)
        t = np.asarray(Statevector(qc).data).reshape([2] * m)
        t = t.transpose(list(range(m))[::-1])
        return t, True
    u = np.asarray(Operator(qc).data).reshape([2] * (2 * m))
    perm = list(range(m))[::-1] + [m + i for i in range(m)][::-1]
    return u.transpose(perm), False


def cirq_tensor(p):
    import cirq

    m, d = p["wires"], p["d"]
    qs = cirq.LineQid.range(m, dimension=d)
    ops = []
    for wires, re, im in p["gates"]:
        mat = np.asarray(re) + 1j * np.asarray(im)
        ops.append(cirq.MatrixGate(mat, qid_shape=(d,) * len(wires)).on(*[qs[w] for w in wires]))
    circuit = cirq.Circuit(ops)
    phase = complex(*p["scalar"])
    if len(p["inits"]) == m:
        init = [int(p["inits"][str(i)]) for i in range(m)]
        idx = 0
        for v in init:
            idx = idx * d + v
        state = cirq.final_state_vector(
            circuit, initial_state=idx, qubit_order=qs, dtype=np.complex128
        )
        return phase * np.asarray(state).reshape([d] * m), True
    u = cirq.unitary(circuit) if ops else np.eye(d**m)
    return phase * np.asarray(u).reshape([d] * (2 * m)), False


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    rhs = np.load(p["rhs"])
    c0, w0 = time.process_time(), time.perf_counter()
    t, is_state = (qiskit_tensor if p["tool"] == "qiskit" else cirq_tensor)(p)
    m = p["wires"]
    if not is_state:
        sl = [slice(None)] * (2 * m)
        for i, v in p["inits"].items():
            sl[m + int(i)] = int(v)
        for i, v in p["posts"].items():
            sl[int(i)] = int(v)
        t = t[tuple(sl)]
    else:
        sl = [slice(None)] * m
        for i, v in p["posts"].items():
            sl[int(i)] = int(v)
        t = t[tuple(sl)]
    a, b = t.reshape(-1), rhs.reshape(-1)
    out = {}
    if a.shape != b.shape:
        out["verdict"] = "unequal"
        out["reason"] = "shape"
    else:
        exact = bool(np.allclose(a, b, atol=1e-8))
        k = int(np.argmax(np.abs(b)))
        lam = a[k] / b[k] if abs(b[k]) > 1e-12 else 0
        phase = bool(abs(abs(lam) - 1) < 1e-8 and np.allclose(a, lam * b, atol=1e-8))
        out["verdict"] = "equal" if exact else "unequal"
        out["equal_up_to_global_phase"] = phase
    out["cpu"] = time.process_time() - c0
    out["wall"] = time.perf_counter() - w0
    print(json.dumps(out))


if __name__ == "__main__":
    main()
