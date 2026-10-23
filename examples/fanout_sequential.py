"""n CNOTs from one control applied one after another equal the fanout gate that applies all of
them in one layer, for every n.

Run:  python examples/fanout_sequential.py
"""

from archytaszx import Program

azx = Program()
n, t = azx.var("n", "t")
azx("gate CX = |c>|y> -> |c>|y xor c*2^j>")  # CNOT onto bit j of the target register
azx("gate FAN = |c>|y> -> |c>|y xor c*(2^n - 1)>")  # every target bit at once

azx.zeros = azx.ket(0) ** n
azx.start = (azx.H | azx.ket(0)) @ azx["0 |zeros> + |t>"]  # control |+>, targets |t>

azx.seq = azx.start
azx("for j in 0..n { |seq> = CX |seq> }")
azx.par = azx.FAN | azx.start
azx.show(azx.seq, azx.par)
assert (azx.par.dag() | azx.seq).sympy() == 1
assert (azx.seq.dag() | azx.seq).sympy() == 1
print("sequential fanout = one-layer fanout, for every n")
