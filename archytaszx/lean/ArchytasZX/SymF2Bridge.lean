/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymF2
import ArchytasZX.DiagramBridge

/-!
# Diagrams as networks, checked in linear time

`bridgeQ` checks that a network is `D.toNet off` and that `D.check` holds for the labels
`(List.range M).map (·, 0)`, for a diagram whose legs all have dimension index `0`. A port's
key is `(2 node + [input]) P + pos`; the wires are packed as one number `WB` holding each
wire's two keys in `2K` bits; every boundary port and wire end is a distinct key, so a leg
found at its stated label (a boundary slot or a wire of `WB`) is located there by
`Diagram.locate`. `bridge_ok` reads `D.check` and `D.toNet off = N` from `bridgeQ`.
-/

namespace ArchytasZX
namespace SymF2

open DBridge

/-- The legs of `n` at the labels of `m`. -/
def nodeOKQ (P K off WB : ℕ) (D : Diagram) (n : DNode) (m : Node) : Bool :=
  decide (m.kind = n.kind) && m.dim == n.dim && m.ph == n.ph &&
    m.outs.length == n.outs.length && m.ins.length == n.ins.length &&
    ((List.range n.outs.length).zip m.outs).all (fun q => legOK P K off WB D ⟨n.id, true, q.1⟩ q.2) &&
    ((List.range n.ins.length).zip m.ins).all (fun q => legOK P K off WB D ⟨n.id, false, q.1⟩ q.2)

/-- `N` is `D.toNet off` and `D.check` holds, the labels below `M` of index `0`. -/
def bridgeQ (P K off M WB : ℕ) (D : Diagram) (N : Net) : Bool :=
  decide (D.bnd.length ≤ off) && decide (off + D.wires.length ≤ M) &&
    D.nodes.all (fun n => (n.outs ++ n.ins).all (· == 0) && decide (n.outs.length ≤ P) &&
      decide (n.ins.length ≤ P)) &&
    (ends D).all (fun p => decide (p.pos < P) && decide (pkey P p < 2 ^ K)) &&
    nodupS ((ends D).map (pkey P)) 0 && packW P K D.wires == WB &&
    N.int == (List.range D.wires.length).map (off + ·) && N.bnd == List.range D.bnd.length &&
    N.nodes.length == D.nodes.length &&
    (D.nodes.zip N.nodes).all fun q => nodeOKQ P K off WB D q.1 q.2

theorem lookup_range (M l : ℕ) :
    ((List.range M).map fun x => (x, 0)).lookup l = if l < M then some 0 else none := by
  have key : ∀ xs : List ℕ, (xs.map fun x => (x, 0)).lookup l = if l ∈ xs then some 0 else none := by
    intro xs
    induction xs with
    | nil => simp
    | cons x xs ih =>
      by_cases h : l = x
      · subst h; simp [List.lookup_cons]
      · have hb : (l == x) = false := by simp [h]
        rw [List.map_cons, List.lookup_cons, ih, hb]
        simp [h]
  rw [key]; simp

theorem portDim_zero (D : Diagram) (hz : ∀ n ∈ D.nodes, ∀ x ∈ n.outs ++ n.ins, x = 0) (p : Port) :
    D.portDim p = 0 := by
  unfold Diagram.portDim
  cases h : D.nodes.find? (fun n => n.id = p.node) with
  | none => rfl
  | some n =>
    have hn := List.mem_of_find?_eq_some h
    simp only
    split_ifs
    · rcases Nat.lt_or_ge p.pos n.outs.length with hl | hl
      · rw [List.getD_eq_getElem _ _ hl]; exact hz n hn _ (List.mem_append_left _ (List.getElem_mem hl))
      · rw [List.getD_eq_default _ _ hl]
    · rcases Nat.lt_or_ge p.pos n.ins.length with hl | hl
      · rw [List.getD_eq_getElem _ _ hl]; exact hz n hn _ (List.mem_append_right _ (List.getElem_mem hl))
      · rw [List.getD_eq_default _ _ hl]

theorem bridge_ok (P K off M WB : ℕ) (D : Diagram) (N : Net)
    (h : bridgeQ P K off M WB D N = true) :
    D.check ((List.range M).map fun l => (l, 0)) off = true ∧ D.toNet off = N := by
  unfold bridgeQ at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨hoff, hM⟩, hdims⟩, hpos⟩, hnd⟩, hW⟩, hint⟩, hbnd⟩, hlen⟩, hnodes⟩ := h
  rw [nodupS_eq] at hnd
  have hnd' : (ends D).Nodup := List.Nodup.of_map (pkey P) (nodupM_ok _ 0 hnd).1
  have hpos' : ∀ p ∈ ends D, p.pos < P ∧ pkey P p < 2 ^ K := fun p hp => hpos p hp
  have hz : ∀ n ∈ D.nodes, ∀ x ∈ n.outs ++ n.ins, x = 0 := fun n hn x hx => by
    have := (hdims n hn).1.1 x hx
    simpa using this
  have hnode : ∀ i (hi : i < D.nodes.length),
      nodeOKQ P K off WB D D.nodes[i] (N.nodes[i]'(by omega)) = true := fun i hi =>
    hnodes (D.nodes[i], N.nodes[i]'(by omega))
      (List.mem_iff_getElem.2 ⟨i, by simp [hi, hlen], by simp⟩)
  have hleg : ∀ i (hi : i < D.nodes.length) (o : Bool) (k : ℕ),
      k < (if o then D.nodes[i].outs.length else D.nodes[i].ins.length) →
      D.label off ⟨D.nodes[i].id, o, k⟩ =
        (if o then (N.nodes[i]'(by omega)).outs.getD k 0 else (N.nodes[i]'(by omega)).ins.getD k 0) ∧
      D.okPort ⟨D.nodes[i].id, o, k⟩ = true := by
    intro i hi o k hk
    have hn := hnode i hi
    unfold nodeOKQ at hn
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hn
    obtain ⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, ho⟩, hi'⟩, hao⟩, hai⟩ := hn
    obtain ⟨⟨-, hPo⟩, hPi⟩ := hdims _ (List.getElem_mem hi)
    cases o with
    | true =>
      simp only [if_true] at hk ⊢
      have := all_zip_get _ _ _ hao k (by simpa using hk) (by omega)
      simp only [List.getElem_range] at this
      rw [List.getD_eq_getElem _ _ (by omega)]
      exact (locate_of_leg P K off WB D hnd' hpos' hW _ (by simp; omega) _ this).1
    | false =>
      simp only [Bool.false_eq_true, if_false] at hk ⊢
      have := all_zip_get _ _ _ hai k (by simpa using hk) (by omega)
      simp only [List.getElem_range] at this
      rw [List.getD_eq_getElem _ _ (by omega)]
      exact (locate_of_leg P K off WB D hnd' hpos' hW _ (by simp; omega) _ this).1
  refine ⟨?_, ?_⟩
  · unfold Diagram.check
    simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range]
    refine ⟨⟨⟨⟨hoff, fun n hn => ?_⟩, fun w _ => by rw [portDim_zero D hz, portDim_zero D hz]⟩,
      fun j hj => ?_⟩, ?_⟩
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
      refine ⟨fun k hk => (hleg i hi true k (by simpa using hk)).2,
        fun k hk => (hleg i hi false k (by simpa using hk)).2⟩
    · rw [lookup_range, if_pos (by omega)]
      unfold Diagram.wireDim
      cases D.wires[j]? <;> simp [portDim_zero D hz]
    · apply List.ext_getElem
      · simp [Diagram.bdims, Diagram.bnd]
      · intro i h1 h2
        simp only [List.getElem_map, List.getElem_range, lookup_range]
        rw [if_pos (by simp at h1; omega)]
        simp [Diagram.bdims, portDim_zero D hz]
  · rw [show N = ⟨N.nodes, N.int, N.bnd⟩ from rfl]
    unfold Diagram.toNet
    rw [hint, hbnd]
    congr 1
    apply List.ext_getElem (by simp [hlen])
    intro i h1 h2
    rw [List.getElem_map]
    have hi : i < D.nodes.length := by simpa using h1
    have hn := hnode i hi
    unfold nodeOKQ at hn
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hn
    obtain ⟨⟨⟨⟨⟨⟨hk, hd⟩, hph⟩, ho⟩, hi'⟩, -⟩, -⟩ := hn
    rw [show N.nodes[i] = ⟨N.nodes[i].kind, N.nodes[i].outs, N.nodes[i].ins, N.nodes[i].dim,
      N.nodes[i].ph⟩ from rfl]
    unfold DNode.node
    rw [hk, hd, hph]
    congr 1
    · apply List.ext_getElem (by simp [ho])
      intro k hk1 hk2
      simp only [List.getElem_map, List.getElem_range]
      have := (hleg i hi true k (by simpa using hk1)).1
      simp only [if_true] at this
      rw [this, List.getD_eq_getElem _ _ hk2]
    · apply List.ext_getElem (by simp [hi'])
      intro k hk1 hk2
      simp only [List.getElem_map, List.getElem_range]
      have := (hleg i hi false k (by simpa using hk1)).1
      simp only [Bool.false_eq_true, if_false] at this
      rw [this, List.getD_eq_getElem _ _ hk2]

end SymF2
end ArchytasZX
