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

"""Lattices shared by the topological code subjects: the honeycomb on an L x L torus (sites A(x, y),
B(x, y); hexagons; edges by direction and by colour) and the cubic L x L x L torus (qubits on
edges)."""

from __future__ import annotations

import itertools


class Honeycomb:
    """The honeycomb on the L x L torus, L a multiple of 3: 2 L^2 sites, L^2 hexagons
    3-coloured by (x - y) mod 3, 3 L^2 edges each with a direction letter (Z for A(x,y)-B(x,y),
    X for A(x,y)-B(x-1,y), Y for A(x,y)-B(x,y-1)) and the colour of neither hexagon it borders."""

    def __init__(self, L: int) -> None:
        self.L = L
        self.n = 2 * L * L
        self.hexes = {
            (x, y): [
                self.a(x, y),
                self.b(x, y),
                self.a(x, y + 1),
                self.b(x - 1, y + 1),
                self.a(x - 1, y + 1),
                self.b(x - 1, y),
            ]
            for x in range(L)
            for y in range(L)
        }
        self.edges: list[tuple[int, int, str, int]] = []
        for x, y in itertools.product(range(L), repeat=2):
            for u, v, t in (
                (self.a(x, y), self.b(x, y), "Z"),
                (self.a(x, y), self.b(x - 1, y), "X"),
                (self.a(x, y), self.b(x, y - 1), "Y"),
            ):
                around = [h for h, vs in self.hexes.items() if u in vs and v in vs]
                colour = ({0, 1, 2} - {(h[0] - h[1]) % 3 for h in around}).pop()
                self.edges.append((u, v, t, colour))

    def a(self, x: int, y: int) -> int:
        return 2 * ((x % self.L) * self.L + y % self.L)

    def b(self, x: int, y: int) -> int:
        return self.a(x, y) + 1


class Cubic:
    """The L x L x L cubic torus with one qubit per edge (site p, direction 0, 1, 2)."""

    def __init__(self, L: int) -> None:
        self.L = L
        self.n = 3 * L**3
        self.sites = list(itertools.product(range(L), repeat=3))

    def edge(self, p: tuple[int, ...], d: int) -> int:
        x, y, z = (c % self.L for c in p)
        return 3 * ((x * self.L + y) * self.L + z) + d

    @staticmethod
    def step(p: tuple[int, ...], d: int, s: int = 1) -> tuple[int, ...]:
        return tuple(c + s * (i == d) for i, c in enumerate(p))


__all__ = ["Cubic", "Honeycomb"]
