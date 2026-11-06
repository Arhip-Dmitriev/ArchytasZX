"""The examples of this directory through the Python library: each prints its decide verdict."""

from archytaszx import azx

d = azx.d

bell = azx.Z(0, 2) * d ** "-1/2"
print("partial trace of Bell", azx.decide(azx.ptrace(bell, 1), azx.maxmixed()).verdict)

ghz4 = azx.Z(0, 4) * d ** "-1/2"
print("partial trace of GHZ_4", azx.decide(azx.ptrace(ghz4, 1, 2, 3), azx.maxmixed()).verdict)

print("dephasing idempotent", azx.decide(azx.dephase() | azx.dephase(), azx.dephase()).verdict)

zero = azx.X(0, 1) * d ** "-1/2"
print("measure_x of |0>", azx.decide(azx.measure_x() | zero, azx.maxmixed()).verdict)

cx = (azx.id @ azx.X(2, 1)) | (azx.Z(1, 2) @ azx.id)
cx *= d ** "1/2"
grounds = azx.discard() @ azx.discard()
print("causality of CX", azx.decide(grounds | cx, grounds).verdict)

born = azx.ptrace(azx.Z(0, 4), 1, 2, 3)
born = (azx.X(1, 0) | born) * d ** "-1"
p = (azx.discard() | azx.X(0, 1)) * d ** "-1"
print("Born probability 1/d", azx.decide(born, p).verdict)

TELEPORTATION = """
za = Z(1 -> 2)
xb = X(2 -> 1)
bell = Z(0 -> 2)
wire za.out1 -- xb.in1, bell.out0 -- xb.in0
fa = F
wire za.out0 -- fa.in0
mb = Z(1 -> 2)
ma = Z(1 -> 2)
wire xb.out0 -- mb.in0, fa.out0 -- ma.in0
gb = G
ga = G
wire mb.out0 -- gb.in0, ma.out0 -- ga.in0
c1 = F
c2 = F
c1 -> c2
xc = X(2 -> 1)
wire bell.out1 -- c1.in0, c2.out0 -- xc.in0, mb.out1 -- xc.in1
zc = Z(1 -> 2)
g1 = F
g2 = F
g3 = F
g1 -> g2 -> g3
wire xc.out0 -- zc.in0, zc.out1 -- g1.in0, g3.out0 -- ma.out1
inputs za.in0
outputs zc.out0
scalar d
"""
print("teleportation", azx.decide(TELEPORTATION, "i = Z(1 -> 1)").verdict)
