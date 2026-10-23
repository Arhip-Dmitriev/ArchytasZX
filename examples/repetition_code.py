"""The repetition-code encoder maps a logical qubit a|0> + b|1> to a|0...0> + b|1...1>, for a
symbolic code distance n + 1: CNOTs from the logical qubit onto n ancillas in |0>.

Run:  python examples/repetition_code.py
"""

from archytaszx import Program

azx = Program()
n = azx.var("n")
a, b = azx.var("a", "b", type="complex")
azx("gate CX = |c>|y> -> |c>|y xor c*2^j>")  # CNOT onto ancilla j

azx.enc = (a * azx.ket(0) + b * azx.ket(1)) @ azx.ket(0) ** n
azx("for j in 0..n { |enc> = CX |enc> }")
azx.show(azx.enc)

azx.want = a * azx.ket(0) @ azx.ket(0) ** n + b * azx.ket(1) @ azx.ket(1) ** n
norm = (azx.want.dag() | azx.want).sympy()
assert (azx.want.dag() | azx.enc).sympy() == norm
assert (azx.enc.dag() | azx.enc).sympy() == norm
print("the encoder maps a|0> + b|1> to a|0...0> + b|1...1>, for every distance")
