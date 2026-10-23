"""Teleporting a qudit along n hops is the identity, for every n and d and every Bell outcome
(u, v) with its Pauli correction.

Run:  python examples/teleport_chain.py
"""

from archytaszx import Program

azx = Program()
azx.var("n", "d", "u", "v")
azx("|B>_d = 1/sqrt(d) sum(k in 0..d){ |k>_d @ |k>_d }")
azx("|Buv>_d = 1/sqrt(d) sum(k in 0..d){ e^(2*pi*i*v*k/d) |(k + u) % d>_d @ |k>_d }")

# One hop: a fresh Bell pair, the Bell measurement reading (u, v), then the correction.
azx("gate HOP = |a>_d -> d sum(c in 0..d){ ((<Buv| @ <c|_d)(|a>_d @ |B>)) |c>_d }")
azx("gate FIX = |c>_d -> e^(2*pi*i*v*c/d) |(c + u) % d>_d")

azx("|psi>_d = 1/sqrt(d) sum(k in 0..d){ e^(2*pi*i*k^2/d) |k>_d }")
azx.show(azx.FIX | (azx.HOP | azx.psi))

azx.chain = azx.psi
azx("for j in 0..n { |chain> = FIX HOP |chain> }")
azx.show(azx.chain)
assert (azx.psi.dag() | azx.chain).sympy() == 1
assert (azx.chain.dag() | azx.chain).sympy() == 1
print("n hops of teleportation are the identity, for every n and d")
