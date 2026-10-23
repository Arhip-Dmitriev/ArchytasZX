"""A guided tour of the azx Python embedding, ending in Deutsch-Jozsa for a symbolic n.

Run:  python examples/learn_azx.py
"""

from archytaszx import Program

# ---- 1. A program -------------------------------------------------------------------------
# A Program is one interpreter. `from archytaszx import azx` gives a shared default one.
azx = Program()

# Calling the program runs .azx statements and returns what show/print printed.
print(azx("var a = 3\nshow(a)"))

# ---- 2. Variables -------------------------------------------------------------------------
# azx.var declares symbols (default type nat) and returns them as Values.
n = azx.var("n")
t = azx.var("t", type="real")
azx.show(n, t)
azx("assume n > 0")

# Attribute assignment names a value in the program; Python numbers become exact numbers.
azx.b = azx.a ** 2 + 1          # ** on scalars is power
azx.show(azx.b)                  # show prints; render returns the text

# A Value has no ==; .sympy() gives the sympy expression, int()/float() a number.
assert azx.b.sympy() == 10
assert int(azx.b) == 10

# ---- 3. Language syntax inside Python -----------------------------------------------------
# azx["..."] evaluates one .azx expression; azx["|name>"] = ... assigns in .azx syntax.
print(azx["sqrt(8)"], azx["gcd(12, 18)"], azx["sum(k in 0..=4){ k }"])
azx["|phi>"] = azx["normalise(|0> + |1>)"]
azx.show(azx.phi)

# Python values can be spliced into statements with f-strings.
for k in range(3):
    azx(f"var c{k} = {k} ^ 2")
azx.show(azx.c0, azx.c1, azx.c2)

# ---- 4. Kets, bras and the operators ------------------------------------------------------
#   |   juxtaposition: apply, inner/outer product, tensor of kets
#   @   tensor product
#   **  tensor power of states and operators
#   .dag()  conjugate transpose
azx.p = azx.ket(0) + azx.ket(1)  # |0> + |1>
azx.v = azx.ket(2, 3)            # |2>_3
azx.show(azx.p, azx.v)
print(azx.bra(0) | azx.ket("+"))   # <0|+>
print(azx.ket("+").dag() | azx.ket("-"))  # <+|->
azx.zeros = azx.ket(0) ** n      # |0>^@n, n symbolic
azx.show(azx.zeros)

# ---- 5. Gates -----------------------------------------------------------------------------
# Built-in gates are attributes: H X Y Z S T CNOT SWAP I.
azx.c = azx.H | azx.ket(0)
azx.show(azx.c)

# Python's | groups from the left, the .azx language from the right:
#   CNOT (H @ I) |0>|0>  becomes  CNOT | (H @ I) | (|0> @ |0>)
azx.bell = azx.CNOT | (azx.H @ azx.I) | (azx.ket(0) @ azx.ket(0))
azx.show(azx.bell)

# New gates and functions are defined with .azx statements, then used as attributes.
azx("func f(x) = bit(x, 0)")
azx("gate U_f = |x>|y> -> |x>|y xor f(x)>")
azx("gate QFT = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(2*pi*i*x*y/d) |y>_d }")
azx.var("d")
azx.show(azx.QFT | azx.ket(0, "d"))

# ---- 6. Output ----------------------------------------------------------------------------
print(azx.c)                     # unicode
print(azx.c.ascii())
print(azx.c.latex())             # also rendered by Jupyter
print(azx.c.vector())            # numpy amplitudes of a concrete state
print(sorted(azx.names()))       # every user-defined name

# ---- 7. Measurement -----------------------------------------------------------------------
# azx.measure | state (or azx.measure(state)) returns the Born distribution and leaves the
# named state as the mixture of its outcomes.
azx.coin = azx.H | azx.ket(0)
print(azx.measure | azx.coin)
azx.show(azx.coin)
# Functions of the outcome go through the language syntax.
# In Python @ binds tighter than |, so (H |0>) @ |1> needs its parentheses.
azx.two = (azx.H | azx.ket(0)) @ azx.ket(1)
print(azx["floor(measure|two> / 2)"])

# ---- 8. Deutsch-Jozsa for n qubits, n symbolic --------------------------------------------
# f : {0,1}^n -> {0,1} is promised constant or balanced. One query of
#   U_f |x>|y> = |x>|y xor f(x)>
# inside H^@n ... H^@n decides which: the first register reads all zeros iff f is constant.
oracles = {"zero": "0", "one": "1", "bal": "bit(x, 0)"}

start = azx.ket(0) ** n @ azx.ket(1)
target = azx.bra(0) ** n @ azx.bra("-")
first = azx.H ** n @ azx.H
last = azx.H ** n @ azx.I
amps = {}

for name, body in oracles.items():
    azx(f"func f_{name}(x) = {body}")
    azx(f"gate O_{name} = |x>|y> -> |x>|y xor f_{name}(x)>")
    oracle = getattr(azx, f"O_{name}")

    # Building the circuit from Python Values.
    azx[f"|out_{name}>"] = last | oracle | first | start
    out = getattr(azx, f"out_{name}")
    azx.show(out)

    # Proof for every n: the amplitude of |0...0>|-> is +-1 if f is constant, 0 if balanced.
    amp = amps[name] = (target | out).sympy()
    kind = "constant" if abs(amp) == 1 else "balanced" if amp == 0 else "undecided"
    print(f"f_{name}: <0...0,-|out> = {amp}  ->  {kind}")

    # Measuring: the outcome of the n+1 qubits is 2x + y; floor(./2) keeps x.
    x = azx[f"floor(measure|out_{name}> / 2)"]
    print(f"f_{name}: first register reads {x}")

assert amps == {"zero": 1, "one": -1, "bal": 0}
print("Deutsch-Jozsa: one query separates constant from balanced, for every n")
