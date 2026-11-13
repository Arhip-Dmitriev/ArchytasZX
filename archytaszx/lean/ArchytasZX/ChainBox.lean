/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Bridge

/-!
# Chain boxes

A `DChain` is a node box with its links: copy `j`'s port `l.1` meets copy `j + 1`'s port `l.2`.
Each link port of the template is wired to a port outside the box; a link port on the boundary
goes through an identity node first. `Diagram.expandChain D c k` copies the box `k` times
(`copyNodes` on the diagram without the link wires) and wires each link `l`: `l.2`'s outside
partner to copy `0`'s `l.2`, copy `j`'s `l.1` to copy `j + 1`'s `l.2`, copy `k - 1`'s `l.1` to
`l.1`'s outside partner, and at `k = 0` the two partners to each other.
-/

namespace ArchytasZX

/-- A chain box: a node box and its links `(l.1, l.2)`, copy `j`'s `l.1` meeting copy `j + 1`'s
`l.2`. -/
structure DChain where
  box : DBox
  links : List (Port × Port)
  deriving DecidableEq, Repr

namespace Diagram

/-- Whether a wire has an end at a link port of `c`. -/
def onLink (c : DChain) (w : Port × Port) : Bool :=
  c.links.any fun l => decide (w.1 = l.1) || decide (w.1 = l.2) || decide (w.2 = l.1) ||
    decide (w.2 = l.2)

/-- Copy `j` of a port on a node of the box `b`. -/
def cpPort (D : Diagram) (b : DBox) (j : ℕ) (p : Port) : Port := ⟨D.cpId b j p.node, p.out, p.pos⟩

/-- The wires of the link `l` at `k` copies, from the outside partner of `l.2` to that of
`l.1`. -/
def linkWires (D : Diagram) (c : DChain) (k : ℕ) (l : Port × Port) : List (Port × Port) :=
  match D.partner l.1, D.partner l.2 with
  | some P, some Q =>
    if k = 0 then [(Q, P)]
    else (Q, D.cpPort c.box 0 l.2) ::
      ((List.range (k - 1)).map fun j => (D.cpPort c.box j l.1, D.cpPort c.box (j + 1) l.2)) ++
      [(D.cpPort c.box (k - 1) l.1, P)]
  | _, _ => []

/-- The chain box `c` instantiated at `k`. -/
def expandChain (D : Diagram) (c : DChain) (k : ℕ) : Diagram :=
  let X := ({ D with wires := D.wires.filter fun w => !onLink c w } : Diagram).copyNodes c.box k
  { X with wires := X.wires ++ c.links.flatMap (D.linkWires c k) }

end Diagram

namespace ChainBoxExample

/-- One Z spider `1` in a chain box, its input `q` and output `p` linked, its second output on
the boundary; identity nodes `0` and `2` hold the chain's ends on the boundary. -/
def D : Diagram :=
  { nodes := [⟨0, .I, 0, 0, [0], [0]⟩, ⟨1, .Z, 0, 0, [0, 0], [0]⟩, ⟨2, .I, 0, 0, [0], [0]⟩],
    wires := [(⟨0, true, 0⟩, ⟨1, false, 0⟩), (⟨1, true, 0⟩, ⟨2, false, 0⟩)],
    outs := [⟨1, true, 1⟩, ⟨2, true, 0⟩],
    ins := [⟨0, false, 0⟩],
    boxes := [⟨0, 0, none, [1], []⟩] }

/-- The chain: output `0` of a copy feeds input `0` of the next. -/
def c : DChain := ⟨⟨0, 0, none, [1], []⟩, [(⟨1, true, 0⟩, ⟨1, false, 0⟩)]⟩

/-- Two copies `3`, `4` between the identities, their second outputs in copy order. -/
theorem expand_two : D.expandChain c 2 =
    { nodes := [⟨0, .I, 0, 0, [0], [0]⟩, ⟨3, .Z, 0, 0, [0, 0], [0]⟩, ⟨4, .Z, 0, 0, [0, 0], [0]⟩,
        ⟨2, .I, 0, 0, [0], [0]⟩],
      wires := [(⟨0, true, 0⟩, ⟨3, false, 0⟩), (⟨3, true, 0⟩, ⟨4, false, 0⟩),
        (⟨4, true, 0⟩, ⟨2, false, 0⟩)],
      outs := [⟨3, true, 1⟩, ⟨4, true, 1⟩, ⟨2, true, 0⟩],
      ins := [⟨0, false, 0⟩],
      boxes := [] } := by decide +kernel

/-- At `0` copies the identities meet. -/
theorem expand_zero : (D.expandChain c 0).wires = [(⟨0, true, 0⟩, ⟨2, false, 0⟩)] := by
  decide +kernel

end ChainBoxExample

end ArchytasZX
