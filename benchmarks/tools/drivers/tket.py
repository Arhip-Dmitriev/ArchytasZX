"""tket driver, run as ``python -I tket.py PAYLOAD``: the QASM circuit through
FullPeepholeOptimise; prints gate and two-qubit-gate counts before and after as one JSON line."""

import json
import sys
import time


def main():
    from pytket.passes import FullPeepholeOptimise
    from pytket.qasm import circuit_from_qasm_str

    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    c0, w0 = time.process_time(), time.perf_counter()
    c = circuit_from_qasm_str(p["qasm"])
    before = (c.n_gates, c.n_2qb_gates())
    FullPeepholeOptimise().apply(c)
    out = {
        "gates_before": before[0],
        "two_qubit_before": before[1],
        "gates_after": c.n_gates,
        "two_qubit_after": c.n_2qb_gates(),
        "cpu": time.process_time() - c0,
        "wall": time.perf_counter() - w0,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
