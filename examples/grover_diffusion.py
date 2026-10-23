"""The Grover diffusion operator, for every n:
H^@n (2|0><0| - I) H^@n = 2|s><s| - I,  |s> = H^@n |0>^@n.

Run:  python examples/grover_diffusion.py
"""

from archytaszx import Program

azx = Program()
n = azx.var("n", "x")[0]
azx.zeros = azx.ket(0) ** n
azx.s = azx.H**n | azx.zeros
azx.w = azx["0 |zeros> + |x>"]  # an arbitrary basis state

reflect_zero = 2 * (azx.zeros | azx.zeros.dag()) - azx.I
reflect_s = 2 * (azx.s | azx.s.dag()) - azx.I
azx.lhs = azx.H**n | (reflect_zero | (azx.H**n | azx.w))
azx.rhs = reflect_s | azx.w
azx.show(azx.rhs)
assert (azx.lhs.dag() | azx.rhs).sympy() == 1
assert (azx.lhs.dag() | azx.lhs).sympy() == 1
assert (azx.rhs.dag() | azx.rhs).sympy() == 1
print("H^@n (2|0><0| - I) H^@n = 2|s><s| - I, for every n")
