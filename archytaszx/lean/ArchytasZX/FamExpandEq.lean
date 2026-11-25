/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamDiagram

/-!
# Expansion of parentless node boxes

`expand_one`: a diagram whose only box is a parentless node box `b` expands to
`D.copyNodes b (ks b.mult)`. `expand_two`: two disjoint parentless node boxes `b`, `c`,
`b.id < c.id`, expand to `copyNodes b` then `copyNodes c`.
-/

namespace ArchytasZX

namespace Diagram

theorem gather_none (D : Diagram) (b : DBox) (hb : b.parent = none) : D.gather b = D := by
  unfold gather
  cases h : D.boxes.length <;> simp [gather.go, hb]

theorem expandNode_none (D : Diagram) (b : DBox) (hb : b.parent = none) (k : ℕ) :
    D.expandNode b k = D.copyNodes b k := by
  unfold expandNode; split <;> simp [gather_none D b hb]

theorem copyNodes_boxes_one (D : Diagram) (b : DBox) (hD : D.boxes = [b]) (k : ℕ) :
    (D.copyNodes b k).boxes = [] := by
  simp [copyNodes, hD]

theorem expandFuel_nil (ks : ℕ → ℕ) (D : Diagram) (hD : D.boxes = []) :
    ∀ fuel, expandFuel ks fuel D = D
  | 0 => rfl
  | _ + 1 => by simp [expandFuel, hD]

/-- The only box a parentless node box: one `copyNodes`. -/
theorem expand_one (ks : ℕ → ℕ) (D : Diagram) (b : DBox) (hD : D.boxes = [b])
    (hb : b.parent = none) (hp : b.ports = []) : D.expand ks = D.copyNodes b (ks b.mult) := by
  simp only [expand, hD, List.length_singleton, expandFuel, List.any_cons, List.any_nil,
    hb, reduceCtorEq, decide_false, Bool.or_false, Bool.not_false, List.filter_cons_of_pos,
    List.filter_nil, List.foldr_cons, List.foldr_nil, hp, ite_true, expandNode_none D b hb]
  exact expandFuel_nil ks _ (copyNodes_boxes_one D b hD _) 1

theorem copyNodes_boxes_two (D : Diagram) (b c : DBox) (hD : D.boxes = [b, c]) (hid : b.id ≠ c.id)
    (hdj : ∀ x ∈ c.nodes, x ∉ b.nodes) (hc : c.nodes ≠ []) (hpc : c.ports = []) (k : ℕ) :
    (D.copyNodes b k).boxes = [c] := by
  have h1 : c.nodes.filter (fun x => !decide (x ∈ b.nodes)) = c.nodes :=
    List.filter_eq_self.2 fun x hx => by simpa using hdj x hx
  have h2 : ¬ ∃ x ∈ c.nodes, x ∈ b.nodes := fun ⟨x, hx, h⟩ => hdj x hx h
  simp only [copyNodes, hD]
  simp [hid.symm, h1, h2, hpc, hc]
  cases c; simp_all

/-- Two disjoint parentless node boxes, `b.id < c.id`: `copyNodes b`, then `copyNodes c`. -/
theorem expand_two (ks : ℕ → ℕ) (D : Diagram) (b c : DBox) (hD : D.boxes = [b, c])
    (hlt : b.id < c.id) (hb : b.parent = none) (hc : c.parent = none) (hpb : b.ports = [])
    (hpc : c.ports = []) (hdj : ∀ x ∈ c.nodes, x ∉ b.nodes) (hne : c.nodes ≠ []) :
    D.expand ks = (D.copyNodes b (ks b.mult)).copyNodes c (ks c.mult) := by
  have hB := copyNodes_boxes_two D b c hD hlt.ne hdj hne hpc (ks b.mult)
  simp only [expand, hD, List.length_cons, expandFuel, List.any_cons,
    List.any_nil, hb, hc, reduceCtorEq, decide_false, Bool.or_false, Bool.not_false,
    List.filter_cons_of_pos, List.filter_nil, List.foldr_cons, List.foldr_nil, hpb, ite_true,
    hlt, expandNode_none D b hb, hB, hpc, expandNode_none _ c hc]
  exact expandFuel_nil ks _ (copyNodes_boxes_one _ c hB _) 1

end Diagram

end ArchytasZX
