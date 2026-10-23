"""Deutsch-Jozsa over Z_d^n, n and d symbolic: one oracle query separates a constant f from a
balanced f.

Run:  python examples/deutsch_jozsa_qudits.py
"""

from archytaszx import Program

azx = Program()
n, d, c = azx.var("n", "d", "c")
azx("gate F = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(2*pi*i*x*y/d) |y>_d }")
azx("gate Fd = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(-2*pi*i*x*y/d) |y>_d }")
azx("gate U_const = |x>|y>_d -> |x>|(y + c) % d>_d")  # f(x) = c
azx("gate U_bal = |x>|y>_d -> |x>|(y + x) % d>_d")  # f(x) = x mod d, balanced

start = azx.ket(0, "d") ** n @ azx.ket(1, "d")
first = azx.F ** n @ azx.Fd
last = azx.Fd ** n @ azx.I
target = (azx.ket(0, "d") ** n @ (azx.Fd | azx.ket(1, "d"))).dag()

azx.out_const = last | (azx.U_const | (first | start))
azx.out_bal = last | (azx.U_bal | (first | start))
azx.show(azx.out_const, azx.out_bal)

a_const = (target | azx.out_const).sympy()
a_bal = (target | azx.out_bal).sympy()
print(f"constant: <0...0|out> = {a_const}")
print(f"balanced: <0...0|out> = {a_bal}")
assert abs(a_const) == 1
assert a_bal == 0
