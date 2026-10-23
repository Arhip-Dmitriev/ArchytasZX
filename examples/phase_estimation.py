"""Phase estimation with n qudits of dimension d, n and d symbolic: for an eigenphase m/d^n the
controlled-U^(d^j) stages followed by the inverse QFT give |m> exactly (m read mod d^n).

Run:  python examples/phase_estimation.py
"""

from archytaszx import Program

azx = Program()
n = azx.var("n", "d", "m")[0]
azx("var Q = d^n")
azx("gate F = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(2*pi*i*x*y/d) |y>_d }")
azx("gate IQFT = |x>_d -> 1/sqrt(Q) sum(k in 0..Q){ e^(-2*pi*i*x*k/Q) |k>_d }")
azx("gate U = |y> -> e^(2*pi*i*m*y/Q) |y>")  # U|1> = e^(2 pi i m/Q) |1>
# U^(d^j) controlled by digit j of the counting register
azx("gate CU = |x>|y> -> e^(2*pi*i*m*y*d^j*(floor(x/d^j) % d)/Q) |x>|y>")

azx.show(azx.U | azx.ket(1))
azx.p = (azx.F**n | azx.ket(0, "d") ** n) @ azx.ket(1)
azx("for j in 0..n { |p> = CU |p> }")
azx.show(azx.p)
azx.p = (azx.IQFT @ azx.I) | azx.p
azx.show(azx.p)

azx("|zeros>_d = |0>_d^@n")
azx.want = azx["0 |zeros> + |m % Q>"] @ azx.ket(1)
assert (azx.want.dag() | azx.p).sympy() == 1
assert (azx.p.dag() | azx.p).sympy() == 1
print("phase estimation returns the eigenphase digits exactly, for every n and d")
