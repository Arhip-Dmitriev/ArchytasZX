"""DiZX driver, run as ``python -I dizx.py PAYLOAD``: circuit a then the inverse of circuit b,
built gate by gate, reduced by clifford_simp; equal when every input reaches its own output
through phase-free arity-2 Z spiders whose Hadamard edge weights compose to the identity mod d,
unknown otherwise. Prints one JSON line."""

import json
import sys
import time


def add(circuit, op, inverse, d):
    """Appends op (or its inverse) as a power of plain gates: X^p, Z^p, HAD^p, CX^s, CZ^s
    (DiZX's adjoint flag does not reach the graph)."""
    name, wires, power = op
    if name == "SWAP":
        circuit.add_gate("SWAP", wires[0], wires[1])
        return
    period = 4 if name == "HAD" else d
    k = (-power if inverse else power) % period
    for _ in range(k):
        circuit.add_gate(name, *wires)


def main():
    import dizx
    from dizx.simplify import clifford_simp
    from dizx.utils import VertexType

    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    d, n = p["d"], p["n"]
    dizx.settings.dim = d
    c0, w0 = time.process_time(), time.perf_counter()
    circuit = dizx.Circuit(n, dim=d)
    for op in p["a"]:
        add(circuit, op, False, d)
    for op in reversed(p["b"]):
        add(circuit, op, True, d)
    g = circuit.to_graph()
    before = g.num_vertices()
    clifford_simp(g)

    def chain(i):
        """The boundary a path from input i reaches through phase-free arity-2 Z spiders, and
        the Hadamard weights of its edges in order; None when it branches or meets anything
        else."""
        prev, cur, weights = None, i, []
        while True:
            nxt = [v for v in g.neighbors(cur) if v != prev]
            if len(nxt) != 1:
                return None
            v = nxt[0]
            e = g.edge_object(g.edge(cur, v))
            if e.had and e.simple:
                return None
            if e.had:
                weights.append(e.had % d)
            if g.type(v) == VertexType.BOUNDARY:
                return v, weights
            if g.type(v) != VertexType.Z or not g.phase(v).is_zero():
                return None
            if len(list(g.neighbors(v))) != 2:
                return None
            prev, cur = cur, v

    def identity_path(i, o):
        """Whether the chain from i ends at o and composes to the identity: an even number of
        Hadamard edges, consecutive pairs (a, b) mapping x to -(a/b) x, the product 1 mod d."""
        res = chain(i)
        if res is None or res[0] != o or len(res[1]) % 2:
            return False
        ws = res[1]
        total = 1
        for k in range(0, len(ws), 2):
            if ws[k + 1] == 0:
                return False
            total = total * (-ws[k]) * pow(ws[k + 1], -1, d) % d
        return total == 1

    ins, outs = list(g.inputs()), list(g.outputs())
    ok = len(ins) == len(outs) == n and all(identity_path(i, o) for i, o in zip(ins, outs))
    print(
        json.dumps(
            {
                "verdict": "equal" if ok else "unknown",
                "vertices_before": before,
                "vertices_after": g.num_vertices(),
                "cpu": time.process_time() - c0,
                "wall": time.perf_counter() - w0,
            }
        )
    )


if __name__ == "__main__":
    main()
