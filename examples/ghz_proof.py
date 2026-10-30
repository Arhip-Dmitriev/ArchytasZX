"""GHZ-state preparation for every n and every qudit dimension d, through the Python library."""

from archytaszx import azx

n, d = azx.count("n"), azx.d

ghz = (azx.id @ azx.Z(1, n)) | azx.Z(0, 2)
ghz *= d ** "-1/2"

t = azx.X(2, 1) | (azx.id @ azx.X(0, 1))
diagram = (azx.id @ t**n) | azx.Z(1, 2) | azx.F | azx.X(0, 1)
diagram *= d ** "-1/2"

print(azx.decide(diagram, ghz).verdict)
print(azx.lean("ghz.lean"))
