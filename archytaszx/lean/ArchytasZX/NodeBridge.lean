/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.NodeVal

/-!
# A bundled node as its copies

`val_bnode`: the value of the bundled node `D.bnode off n` at a bundled assignment `β` is the
product over its copies `t` of the copy `D.dnode n t` at the digit assignment `ν`, which reads at
each digit label the digit of the leg's bundled value.
-/

namespace ArchytasZX

namespace Diagram

/-- The conditions on a node for `val_bnode`. -/
structure NodeOK (D : Diagram) (E : Env) (off : ℕ) (n : DNode) (β ν : ℕ → ℕ) : Prop where
  plain : n.kind.plain = true
  boxkind : D.nodeChain n.id ≠ [] → n.kind ≠ .B ∧ n.kind ≠ .S
  tailkind : ¬(tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = []) →
    n.kind = .Z ∨ n.kind = .X ∨ (n.kind = .W ∧ tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = [])
  dims : (D.nodeChain n.id ≠ [] ∨ ¬(tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = [])) → ∀ p ∈ n.legs, D.portDim p = n.dim
  posd : ∀ p ∈ n.legs, 0 < E.dimv (D.portDim p)
  posn : 0 < E.dimv n.dim
  range : ∀ p ∈ n.legs, β (D.label off p) < E.dimv (D.portDim p) ^ size E.mult (D.legChain p)
  dig : ∀ p ∈ n.legs, ∀ u ∈ cidx E.mult (D.legChain p),
    ν (dlab p u) = digit (E.dimv (D.portDim p)) (pos E.mult (D.legChain p) u) (β (D.label off p))

theorem leg_side {n : DNode} (o : Bool) (i : ℕ) (hi : i < (if o then n.outs else n.ins).length) :
    (⟨n.id, o, i⟩ : Port) ∈ n.legs := by
  rw [DNode.mem_legs]; cases o <;> simpa using hi

theorem pairs_mem_leg (D : Diagram) (off : ℕ) (n : DNode) (o : Bool) {q : ℕ × List ℕ}
    (hq : q ∈ pairs (if o then n.outs else n.ins).length (fun j => D.label off ⟨n.id, o, j⟩)
      (fun j => D.tail ⟨n.id, o, j⟩)) :
    ∃ p ∈ n.legs, q = (D.label off p, D.tail p) := by
  simp only [pairs, List.mem_map, List.mem_range] at hq
  obtain ⟨j, hj, rfl⟩ := hq
  exact ⟨_, leg_side o j hj, rfl⟩

theorem dnode_eq (D : Diagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) : D.dnode ks n t =
    ⟨n.kind, (List.range n.outs.length).flatMap fun i => (rcidx ks (D.tail ⟨n.id, true, i⟩)).map
      fun s => dlab ⟨n.id, true, i⟩ (t ++ s), (List.range n.ins.length).flatMap fun i =>
      (rcidx ks (D.tail ⟨n.id, false, i⟩)).map fun s => dlab ⟨n.id, false, i⟩ (t ++ s), n.dim, n.ph⟩ :=
  rfl

/-- `ν` reads the digits of a copy's legs, at the node's dimension. -/
theorem dig_side (D : Diagram) (E : Env) (off : ℕ) (n : DNode) (β ν : ℕ → ℕ)
    (h : D.NodeOK E off n β ν) (hd : ∀ p ∈ n.legs, D.portDim p = n.dim) (o : Bool) (t : List ℕ)
    (ht : t ∈ cidx E.mult (D.nodeChain n.id)) :
    ((List.range (if o then n.outs else n.ins).length).flatMap fun i =>
      (rcidx E.mult (D.tail ⟨n.id, o, i⟩)).map fun s => dlab ⟨n.id, o, i⟩ (t ++ s)).map ν =
    fvR E.mult (D.nodeChain n.id) t (E.dimv n.dim) β
      (pairs (if o then n.outs else n.ins).length (fun j => D.label off ⟨n.id, o, j⟩)
        (fun j => D.tail ⟨n.id, o, j⟩)) := by
  refine dnode_side D E.mult n t o _ _ _ β ν fun i hi s hs => ?_
  have hp := leg_side o i hi
  rw [h.dig _ hp (t ++ s) (mem_cidx_append _ _ _ _ _ ht ((mem_rcidx _ _ _).1 hs)), hd _ hp]
  rfl

end Diagram

end ArchytasZX
