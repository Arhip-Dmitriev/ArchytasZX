/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainFam

/-!
# Two closed chain boxes equal at every copy count

`fam_pair`: two chain families on the same link labels and boundary, each side's block of one
copy and its head equal to a scalar times its head (`fam_step`), equal at `0` copies, are equal
at every `k`, the scalars growing by the other side's block factor. `chain_pair` states it over
the expansions of two closed chain boxes (`expand_chFam`). `Equal.env` moves an equation to an
environment that reads every node's dimension and phase alike and every label's dimension alike
(Z, X, F and identity nodes).
-/

namespace ArchytasZX

/-- A node whose entry reads the environment only through its dimension and phase indices. -/
def Node.plainKind (n : Node) : Bool :=
  match n.kind with
  | .Z | .X | .F | .I => true
  | _ => false

theorem val_plain (E E' : Env) (σ : ℕ → ℕ) (n : Node) (hk : n.plainKind = true)
    (hd : E'.dimv n.dim = E.dimv n.dim) (hp : E'.phase n.ph = E.phase n.ph) :
    val E' σ n = val E σ n := by
  obtain ⟨k, os, is, dm, ph⟩ := n
  simp only at hd hp
  have hk4 : k = .Z ∨ k = .X ∨ k = .F ∨ k = .I := by
    cases k <;> simp [Node.plainKind] at hk ⊢
  rcases hk4 with rfl | rfl | rfl | rfl <;> simp only [val, hd, hp] <;>
    (generalize List.map σ os = a; generalize List.map σ is = b
     rcases a with _ | ⟨o, _ | ⟨o', a⟩⟩ <;> rcases b with _ | ⟨i, _ | ⟨i', b⟩⟩ <;> rfl)

/-- An equation holds in every environment reading its nodes and labels alike. -/
theorem Equal.env {E E' : Env} {s t : ℂ} {A B : Net} (h : Equal E s A t B)
    (hl : ∀ x, labDim E' x = labDim E x)
    (hn : ∀ n ∈ A.nodes ++ B.nodes,
      n.plainKind = true ∧ E'.dimv n.dim = E.dimv n.dim ∧ E'.phase n.ph = E.phase n.ph) :
    Equal E' s A t B := by
  have he : ∀ N : Net, (∀ n ∈ N.nodes,
      n.plainKind = true ∧ E'.dimv n.dim = E.dimv n.dim ∧ E'.phase n.ph = E.phase n.ph) →
      ∀ σ, eval E' N σ = eval E N σ := by
    intro N hN σ
    unfold eval
    rw [show labDim E' = labDim E from funext hl]
    congr 1
    funext τ
    unfold prodVal
    congr 1
    exact List.map_congr_left fun n hm =>
      val_plain E E' τ n (hN n hm).1 (hN n hm).2.1 (hN n hm).2.2
  refine ⟨h.1, fun σ hσ => ?_⟩
  rw [he A (fun n hm => hn n (List.mem_append_left _ hm)),
    he B (fun n hm => hn n (List.mem_append_right _ hm))]
  exact h.2 σ fun l hm => by rw [← hl l]; exact hσ l hm

/-- `fam_step` with the block scaled: `r` times the block after `p` copies equal to the head
gives `r` times the family at `k + p` equal to the family at `k`. -/
theorem fam_stepR (E : Env) (hD : ∀ x y, labDim E x = labDim E y) {W : ℕ} {u bd : List ℕ}
    {tl cp hd : Net} (h : famOK W u tl cp hd bd = true) (p : ℕ) (r : ℂ)
    (h0 : Equal E r (blkNet W u cp hd p) 1 (headNet u hd)) (k : ℕ) :
    Equal E r (famNet W u tl cp hd bd (k + p)) 1 (famNet W u tl cp hd bd k) := by
  obtain ⟨hu, -, -, -, -, -, hbd⟩ := famOK_spec h
  have hk := Equal.mapL E (sh W k) (sh_inj W k) (fun x => hD _ _) h0
  rw [blk_mapL h k p, head_mapL h k] at hk
  rw [famNet_add]
  refine plug E _ _ _ _ _ _ bd _ r 1 (Equal.toLocal E _ _ hk) (fun x hx => ctx_off_blk h k p x hx)
    (fun x hx => ctx_off_head h k x hx) fun x hx => ?_
  rcases List.mem_append.1 hx with hx | hx
  · obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hx
    cases k with
    | zero =>
      rw [sh_zero]
      by_cases hb : v ∈ bd
      · exact Or.inr hb
      · left; simp [hv, hb]
    | succ k =>
      left
      simp only [List.mem_append, List.mem_flatMap, List.mem_range, List.mem_map]
      exact Or.inr ⟨k, by omega, Or.inr ⟨v, hv, rfl⟩⟩
  · exact Or.inr (hbd x hx)

/-- Two chain families equal at `0` copies, each side's block of one copy scaled by `r` equal to
its head, are equal at every `k`, the scalars growing by `r` per copy. -/
theorem fam_pair (E : Env) (hD : ∀ x y, labDim E x = labDim E y) {W : ℕ} {u bd : List ℕ}
    {tA cA hA tB cB hB : Net} (okA : famOK W u tA cA hA bd = true)
    (okB : famOK W u tB cB hB bd = true) {rA rB a b : ℂ}
    (sA : Equal E rA (blkNet W u cA hA 1) 1 (headNet u hA))
    (sB : Equal E rB (blkNet W u cB hB 1) 1 (headNet u hB))
    (b0 : Equal E a (famNet W u tA cA hA bd 0) b (famNet W u tB cB hB bd 0)) :
    ∀ k, Equal E (a * rA ^ k) (famNet W u tA cA hA bd k) (b * rB ^ k)
      (famNet W u tB cB hB bd k) := by
  intro k
  induction k with
  | zero => simpa using b0
  | succ k ih =>
    have fa := fam_stepR E hD okA 1 rA sA k
    have fb := fam_stepR E hD okB 1 rB sB k
    exact ((fa.mul_trans ih).mul_trans fb.symm).congr_scalar (by ring) (by ring)

namespace Diagram

/-- Two closed chain boxes on the same links and boundary size, each side's block of one copy
scaled by `r` equal to its head, equal at `0` copies, have equal expansions at every `k`, the
scalars growing by `r` per copy. -/
theorem chain_pair {DA DB : Diagram} {cA cB : DChain} {W : ℕ} (hA : ChainFamOK DA cA W)
    (hB : ChainFamOK DB cB W) (hL : cB.links.length = cA.links.length)
    (hm : DB.bnd.length = DA.bnd.length) (E : Env) (hD : ∀ x y, labDim E x = labDim E y)
    (o : ℕ) (ho : DA.bnd.length ≤ o)
    (okA : famOK W (chU cA W) (DA.chTl cA W) (DA.chCp cA W) (DA.chHdN cA W)
      (List.range DA.bnd.length) = true)
    (okB : famOK W (chU cA W) (DB.chTl cB W) (DB.chCp cB W) (DB.chHdN cB W)
      (List.range DA.bnd.length) = true) {rA rB a b : ℂ}
    (sA : Equal E rA (blkNet W (chU cA W) (DA.chCp cA W) (DA.chHdN cA W) 1) 1
      (headNet (chU cA W) (DA.chHdN cA W)))
    (sB : Equal E rB (blkNet W (chU cA W) (DB.chCp cB W) (DB.chHdN cB W) 1) 1
      (headNet (chU cA W) (DB.chHdN cB W)))
    (b0 : Equal E a (DA.chFam cA W 0) b (DB.chFam cB W 0)) (k : ℕ) :
    Equal E (a * rA ^ k) ((DA.expandChain cA k).toNet o) (b * rB ^ k)
      ((DB.expandChain cB k).toNet o) := by
  have hU : chU cB W = chU cA W := by simp only [chU, hL]
  have eB : ∀ j, DB.chFam cB W j = famNet W (chU cA W) (DB.chTl cB W) (DB.chCp cB W)
      (DB.chHdN cB W) (List.range DA.bnd.length) j := fun j => by
    simp only [chFam, hU, hm]
  have hf := fam_pair E hD okA okB sA sB (by rw [← eB 0]; exact b0) k
  rw [← eB k] at hf
  have eA := expand_chFam hA E hD o ho k
  have eBk := expand_chFam hB E hD o (by rw [hm]; exact ho) k
  exact ((eA.mul_trans hf).mul_trans eBk.symm).congr_scalar (by ring) (by ring)

theorem lookup_getD_all (l : List (ℕ × ℕ)) (v x : ℕ) (h : l.all (fun p => p.2 == v) = true) :
    (l.lookup x).getD v = v := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    simp only [List.all_cons, Bool.and_eq_true, beq_iff_eq] at h
    simp only [List.lookup]
    split
    · simp [h.1]
    · exact ih h.2

/-- An equation over networks whose nodes are Z, X, F or identity nodes of dimension and phase
index `0` holds in every environment reading those indices and every label's dimension alike. -/
theorem _root_.ArchytasZX.Equal.env0 {E E' : Env} {s t : ℂ} {A B : Net} (h : Equal E s A t B)
    (hl : ∀ x, labDim E' x = labDim E x)
    (hn : (A.nodes ++ B.nodes).all (fun n => n.plainKind && n.dim == 0 && n.ph == 0) = true)
    (hd : E'.dimv 0 = E.dimv 0) (hp : E'.phase 0 = E.phase 0) : Equal E' s A t B := by
  refine h.env hl fun n hm => ?_
  have := List.all_eq_true.1 hn n hm
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  obtain ⟨⟨h1, h2⟩, h3⟩ := this
  exact ⟨h1, by rw [h2]; exact hd, by rw [h3]; exact hp⟩

end Diagram

end ArchytasZX
