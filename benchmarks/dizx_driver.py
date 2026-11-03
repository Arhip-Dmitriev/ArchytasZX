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

"""DiZX driver, run as ``python -I dizx_driver.py JSON`` in the DiZX environment: builds a
Clifford circuit from a gate list at dimension d, reduces it with ``clifford_simp`` and prints
one JSON line: ``identity`` (every input joined to its output by a plain
edge, directly or through one phase-free Z spider), CPU and wall seconds,
the vertex count before reduction.

Gates: ["P", wire, S] (Z^S), ["F", wire, power] (HAD for power 1, HAD^dagger for 3) and
["C", control, target, inverse] (CX, or its adjoint).
"""

import json
import sys
import time


def main() -> None:
    spec = json.loads(sys.argv[1])
    import dizx
    from dizx.simplify import clifford_simp
    from dizx.utils import VertexType

    d, n = spec["d"], spec["n"]
    dizx.settings.dim = d
    c0, w0 = time.process_time(), time.perf_counter()
    circuit = dizx.Circuit(n, dim=d)
    for gate in spec["gates"]:
        if gate[0] == "P":
            for _ in range(gate[2] % d):
                circuit.add_gate("Z", gate[1])
        elif gate[0] == "F":
            circuit.add_gate("HAD", gate[1], adjoint=gate[2] == 3)
        else:
            circuit.add_gate("CX", gate[1], gate[2], adjoint=bool(gate[3]))
    g = circuit.to_graph()
    size = g.num_vertices()
    clifford_simp(g)

    def plain(a: object, b: object) -> bool:
        e = g.edge_object(g.edge(a, b))
        return bool(e.is_simple_edge() and e.simple == 1)

    def wired(i: object, o: object) -> bool:
        ns = list(g.neighbors(i))
        if ns == [o]:
            return plain(i, o)
        if len(ns) != 1 or g.type(ns[0]) != VertexType.Z or not g.phase(ns[0]).is_zero():
            return False
        v = ns[0]
        return set(g.neighbors(v)) == {i, o} and plain(i, v) and plain(v, o)

    identity = len(g.inputs()) == len(g.outputs()) == n and all(
        wired(i, o) for i, o in zip(g.inputs(), g.outputs())
    )
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    print(json.dumps({"identity": identity, "cpu": cpu, "wall": wall, "nodes": size}))


if __name__ == "__main__":
    main()
