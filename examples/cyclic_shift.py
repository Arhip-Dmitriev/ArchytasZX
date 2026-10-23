"""The cyclic shift of n qudits, composed with itself n times, is the identity, for every n
and d. On a value x < d^n - 1 the shift is x -> d x mod (d^n - 1); d^n - 1 is fixed.

Run:  python examples/cyclic_shift.py
"""

from archytaszx import Program

azx = Program()
azx.var("n", "d", "x")
azx("var M = d^n - 1")
azx("func rot(y) = ifelse(y < M, (y*d) % M, y)")
azx("gate ROT = |y> -> |rot(y)>")

azx("|zeros>_d = |0>_d^@n")
azx("|s>_d = 0 |zeros> + |x>")
azx("|x0>_d = 0 |zeros> + |x>")
azx.show(azx.ROT | azx.s)

azx("for j in 0..n { |s> = ROT |s> }")
azx.show(azx.s)
assert (azx.x0.dag() | azx.s).sympy() == 1
print("ROT^n = I, for every n and d")
