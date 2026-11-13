/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainLink

/-!
# The one-spider chain expanded at every `k`

`expand_closed` gives `expandChain` of `ChainBoxExample` at every `k ≥ 1` in closed form: the
head identity `0`, the copies `3 + j` in copy order wired output `0` to input `0`, the tail
identity `2`, and the copies' second outputs on the boundary before the tail's output.
-/

namespace ArchytasZX

namespace ChainBoxExample

open Diagram

/-- The nodes at `k` copies. -/
def nodesK (k : ℕ) : List DNode :=
  ⟨0, .I, 0, 0, [0], [0]⟩ :: (List.range k).map (fun j => ⟨3 + j, .Z, 0, 0, [0, 0], [0]⟩) ++
    [⟨2, .I, 0, 0, [0], [0]⟩]

/-- The chain wires at `k + 1` copies, head to tail. -/
def wiresK (k : ℕ) : List (Port × Port) :=
  (⟨0, true, 0⟩, ⟨3, false, 0⟩) ::
    ((List.range k).map fun j => ((⟨3 + j, true, 0⟩ : Port), (⟨3 + (j + 1), false, 0⟩ : Port))) ++
    [(⟨3 + k, true, 0⟩, ⟨2, false, 0⟩)]

/-- The diagram at `k + 1` copies. -/
def closedK (k : ℕ) : Diagram :=
  { nodes := nodesK (k + 1),
    wires := wiresK k,
    outs := (List.range (k + 1)).map (fun j => ⟨3 + j, true, 1⟩) ++ [⟨2, true, 0⟩],
    ins := [⟨0, false, 0⟩],
    boxes := [] }

/-- The diagram without its link wires. -/
def D0 : Diagram := { D with wires := D.wires.filter fun w => !onLink c w }

theorem D0_eq : D0 = { D with wires := [] } := by decide

theorem cpId_D0 (j x : ℕ) (hx : x = 1) : D0.cpId c.box j x = 3 + j := by
  subst hx
  simp [cpId, D0_eq, fresh, D, c]

theorem expand_closed (k : ℕ) : D.expandChain c (k + 1) = closedK k := by
  have hX : D.expandChain c (k + 1) =
      { (D0.copyNodes c.box (k + 1)) with
        wires := (D0.copyNodes c.box (k + 1)).wires ++ c.links.flatMap (D.linkWires c (k + 1)) } :=
    rfl
  rw [hX]
  simp [copyNodes, D0_eq, closedK, nodesK, wiresK, linkWires, partner, cpPort, D, c, cpId, fresh,
    spliceN, nimg, inS, growImg, growSide, crossP, shiftPos]
  have hf : List.findIdx (inS ⟨0, 0, none, [1], []⟩)
      [(⟨1, true, 1⟩ : Port), ⟨2, true, 0⟩] = 0 := by decide
  rw [hf]
  simp [nimg, inS, growImg, crossP, shiftPos, List.flatMap_cons, List.flatMap_nil]
  induction (List.range (k + 1)) with
  | nil => rfl
  | cons j l ih => simp [ih]

end ChainBoxExample

end ArchytasZX
