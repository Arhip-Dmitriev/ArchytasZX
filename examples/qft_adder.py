"""The QFT adder (Draper) on n qudits of dimension d implements |x> -> |x + 1 mod d^n>, for
every n and d: a phase on each Fourier digit, between the QFT and its inverse.

Run:  python examples/qft_adder.py
"""

from archytaszx import Program

azx = Program()
azx.var("n", "d", "x")
azx("var Q = d^n")
azx("gate QFT = |x>_d -> 1/sqrt(Q) sum(k in 0..Q){ e^(2*pi*i*x*k/Q) |k>_d }")
azx("gate IQFT = |x>_d -> 1/sqrt(Q) sum(k in 0..Q){ e^(-2*pi*i*x*k/Q) |k>_d }")
azx("gate P = |k> -> e^(2*pi*i*d^j*(floor(k/d^j) % d)/Q) |k>")  # phase on digit j

azx("|zeros>_d = |0>_d^@n")
azx("|a>_d = 0 |zeros> + |x>")
azx.a = azx.QFT | azx.a
azx("for j in 0..n { |a> = P |a> }")
azx.show(azx.a)
azx.a = azx.IQFT | azx.a
azx.show(azx.a)

azx("|want>_d = 0 |zeros> + |(x + 1) % Q>")
assert (azx.want.dag() | azx.a).sympy() == 1
assert (azx.a.dag() | azx.a).sympy() == 1
print("QFT adder: |x> -> |x + 1 mod d^n>, for every n and d")
