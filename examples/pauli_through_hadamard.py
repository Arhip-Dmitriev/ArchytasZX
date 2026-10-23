"""X^a Z^b passed through H^@n becomes Z^a X^b, for every n and all n-bit strings a, b:
H^@n X^a Z^b H^@n = Z^a X^b.

Run:  python examples/pauli_through_hadamard.py
"""

from archytaszx import Program

azx = Program()
n = azx.var("n", "x", "a", "b")[0]
azx("gate XA = |y> -> |y xor a>")
azx("gate ZB = |y> -> (-1)^dot(y, b) |y>")
azx("gate XB = |y> -> |y xor b>")
azx("gate ZA = |y> -> (-1)^dot(y, a) |y>")

azx.zeros = azx.ket(0) ** n
azx.w = azx["0 |zeros> + |x>"]  # an arbitrary basis state

hn = azx.H**n
azx.lhs = hn | (azx.XA | (azx.ZB | (hn | azx.w)))
azx.rhs = azx.ZA | (azx.XB | azx.w)
azx.show(azx.rhs)
assert (azx.rhs.dag() | azx.lhs).sympy() == 1
assert (azx.rhs.dag() | azx.rhs).sympy() == 1
print("H^@n X^a Z^b H^@n = Z^a X^b, for every n")
