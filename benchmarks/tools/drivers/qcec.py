"""MQT QCEC driver, run as ``python -I qcec.py PAYLOAD``: both QASM circuits loaded as
QuantumComputations, the listed wires marked ancillary, verify with partial equivalence;
equivalent is equal, not_equivalent unequal, equivalent up to a global phase equal with
up_to_global_phase_only set (QASM 2 carries no global phase), up to a relative phase unequal,
anything else unknown. Prints one JSON line."""

import json
import sys
import time


def main():
    sys.set_int_max_str_digits(0)
    from mqt.core.ir import QuantumComputation
    from mqt.qcec import verify

    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    c0, w0 = time.process_time(), time.perf_counter()
    a = QuantumComputation.from_qasm_str(p["qasm"])
    b = QuantumComputation.from_qasm_str(p["reference"])
    for q in p["ancillary"]:
        a.set_circuit_qubit_ancillary(q)
    for q in p["reference_ancillary"]:
        b.set_circuit_qubit_ancillary(q)
    res = verify(a, b, check_partial_equivalence=True)
    name = str(res.equivalence).split(".")[-1].lower()
    verdict = {
        "equivalent": "equal",
        "not_equivalent": "unequal",
        "equivalent_up_to_global_phase": "equal",
        "equivalent_up_to_phase": "unequal",
        "probably_not_equivalent": "unknown",
    }.get(name, "unknown")
    out = {
        "verdict": verdict,
        "equivalence": name,
        "up_to_global_phase_only": name == "equivalent_up_to_global_phase",
        "cpu": time.process_time() - c0,
        "wall": time.perf_counter() - w0,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
