/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Induct

/-!
# The digit sum of a box-free diagram

`evalD_boxfree`: without boxes every leg has the single digit `[]`, and the digit sum is
`Diagram.eval` at the slot values `σ i []`.
-/

namespace ArchytasZX

theorem sumOver_dim_congr (D D' : ℕ → ℕ) (ls : List ℕ) (h : ∀ x ∈ ls, D x = D' x)
    (f : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) : sumOver D ls f σ = sumOver D' ls f σ := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l ls ih =>
    simp only [sumOver]
    rw [h l List.mem_cons_self]
    exact Finset.sum_congr rfl fun j _ => ih (fun x hx => h x (List.mem_cons_of_mem _ hx)) _

namespace Diagram

variable {D : Diagram} (hbx : D.boxes = [])

include hbx in
theorem walk_nil (p : DBox → Bool) (f : ℕ) (s : Option ℕ) : D.walk p f s = [] := by
  cases f with
  | zero => rfl
  | succ f => simp [walk, hbx]

include hbx in
theorem legChain_nil (p : Port) : D.legChain p = [] := by
  unfold legChain tail nodeChain nodeBoxes portBoxes
  simp only [walk_nil hbx, List.map_nil, List.nil_append]
  split <;> simp

include hbx in
theorem dlabs_nil (ks : ℕ → ℕ) : D.dlabs ks = D.legs.map (dlab · []) := by
  unfold dlabs
  rw [List.map_eq_flatMap]
  refine List.flatMap_congr fun p _ => ?_
  simp [legChain_nil hbx, cidx]

include hbx in
theorem dnode_nil (ks : ℕ → ℕ) (n : DNode) : D.dnode ks n [] = n.node (dlab · []) := by
  have ht : ∀ q, D.tail q = [] := fun q => by
    have := legChain_nil hbx q; unfold legChain at this; simp at this; exact this.2
  simp [dnode, DNode.node, ht, rcidx_nil, List.map_eq_flatMap]

theorem val_node (E : Env) (τ : ℕ → ℕ) (n : DNode) (g : Port → ℕ) :
    val E τ (n.node g) = val E id (n.node (τ ∘ g)) := by
  rw [← node_rename, val_rename]; rfl

include hbx in
/-- Without boxes the digit sum is the diagram's tensor entry. -/
theorem evalD_boxfree (hW : WF D) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < D.bnd.length), σ i [] < E.dimv (D.portDim D.bnd[i])) :
    D.evalD E ks σ = D.eval E (fun i => σ i []) := by
  set σ'' : ℕ → ℕ := fun i => σ i []
  have hlegs := legs_nodup_of D hW.ids
  have hcn : ∀ q, D.legChain q = [] := legChain_nil hbx
  have hidx : ∀ q, cidx ks (D.legChain q) = [[]] := fun q => by rw [hcn]; rfl
  -- the digit sum as the constrained sum over legs
  have h1 : D.evalD E ks σ = D.evalG (fun p => E.dimv (D.portDim p))
      (fun n f => val E id (n.node f)) σ'' := by
    unfold evalD evalG
    rw [dlabs_nil hbx]
    have hmap : D.legs.map (dlab · []) = (D.legs.map Port.code).map fun c => dlab (Port.decode c) [] := by
      simp [List.map_map, Function.comp_def, Port.decode_code]
    rw [hmap]
    refine sumOver_map_on _ _ _ _ (hlegs.map Port.code_inj) (fun x hx y hy e => ?_) (fun x hx => ?_)
      _ _ _ _ (fun τ τ' hτ hτ' hl => ?_) (fun x hx => ?_)
    · obtain ⟨p, -, rfl⟩ := List.mem_map.1 hx
      obtain ⟨q, -, rfl⟩ := List.mem_map.1 hy
      have := (dlab_inj e).1
      rw [Port.decode_code, Port.decode_code] at this
      rw [this]
    · simp [dport_dlab]
    · have hl' : ∀ p ∈ D.legs, τ' p.code = τ (dlab p []) := fun p hp => by
        rw [hl _ (List.mem_map.2 ⟨p, hp, rfl⟩), Port.decode_code]
      have hs : ∀ i, τ (slab i []) = σ i [] := fun i => by
        rw [hτ _ fun h => by
          obtain ⟨c, -, e⟩ := List.mem_map.1 h
          exact dlab_ne_slab _ _ i [] e, slotsD_slab]
      have hs' : ∀ i, τ' (2 * i + 1) = σ i [] := fun i => by
        rw [hτ' _ fun h => by
          obtain ⟨p, -, e⟩ := List.mem_map.1 h
          have := Port.code_even p; omega]
        simp only [slots]
        rw [if_pos (by omega), show (2 * i + 1) / 2 = i by omega]
      have hpins : D.pinsD ks τ ↔ ∀ q ∈ D.pins, τ' q.1 = τ' q.2 := by
        unfold pinsD pins
        simp only [hidx, List.mem_singleton, forall_eq, List.mem_append, List.mem_map]
        constructor
        · rintro ⟨hw, hb⟩ q (⟨w, hwm, rfl⟩ | ⟨⟨p, i⟩, hpi, rfl⟩)
          · obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hwm
            simp only
            rw [hl' _ l1, hl' _ l2]; exact (hw w hwm).symm
          · have hpb : p ∈ D.bnd := (by obtain ⟨h, e⟩ := (mem_zipIdx' _ _ _).1 hpi; exact e ▸ List.getElem_mem h)
            simp only
            rw [hl' _ (bnd_legs hW hpb), hs', hb _ hpi, hs]
        · intro h
          refine ⟨fun w hwm => ?_, fun q hq => ?_⟩
          · obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hwm
            have := h _ (Or.inl ⟨w, hwm, rfl⟩)
            simp only at this
            rw [hl' _ l1, hl' _ l2] at this; exact this.symm
          · obtain ⟨p, i⟩ := q
            have hpb : p ∈ D.bnd := (by obtain ⟨h, e⟩ := (mem_zipIdx' _ _ _).1 hq; exact e ▸ List.getElem_mem h)
            have := h _ (Or.inr ⟨(p, i), hq, rfl⟩)
            simp only at this ⊢
            rw [hl' _ (bnd_legs hW hpb), hs'] at this
            rw [this, hs]
      have hprod : D.prodD E ks τ = (D.nodes.map fun n => val E id (n.node (τ' ∘ Port.code))).prod := by
        unfold prodD
        congr 1
        refine List.map_congr_left fun n hn => ?_
        have hnc : D.nodeChain n.id = [] := by
          unfold nodeChain nodeBoxes; rw [walk_nil hbx]; rfl
        rw [hnc]
        simp only [cidx, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one]
        rw [dnode_nil hbx, val_node]
        show val E id (n.node (τ ∘ fun p => dlab p [])) = val E id (n.node (τ' ∘ Port.code))
        congr 1
        unfold DNode.node
        congr 1
        · refine List.map_congr_left fun k hk => ?_
          exact (hl' _ (leg_mem hn true k (by simpa [side] using hk))).symm
        · refine List.map_congr_left fun k hk => ?_
          exact (hl' _ (leg_mem hn false k (by simpa [side] using hk))).symm
      classical
      rw [hprod]
      exact if_congr hpins rfl rfl
    · obtain ⟨p, -, rfl⟩ := List.mem_map.1 hx
      simp only [slots, Port.code_even, Port.decode_code, slotsD_dlab]
      simp
  rw [h1, evalG_eq D _ _ (fun n _ f g h => ?_) hW.perm hlegs (fun w hw => by rw [hW.wdim w hw]) σ''
    (fun i hi => hσ i hi)]
  · unfold eval
    refine sumOver_dim_congr _ _ _ (fun j hj => ?_) _ _
    have := List.mem_range.1 hj
    simp [wireDim, List.getElem?_eq_getElem this]
  · simp only [DNode.node]
    congr 2
    · exact List.map_congr_left fun k hk => h _ (by simp [DNode.legs, List.mem_range.1 hk])
    · exact List.map_congr_left fun k hk => h _ (by simp [DNode.legs, List.mem_range.1 hk])

end Diagram

end ArchytasZX
