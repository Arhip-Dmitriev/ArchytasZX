/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Gather

/-!
# Splitting legs along a leaf box

`evalD_split`: when every leg `p` of `D` becomes the legs `img p` of `D'`, a leg `g` holds of
splitting its last copy coordinate (along the multiplicity `m`) over `ks m` images and any other
leg becoming one image with its chain, wires to wires as zips and boundary to boundary, the two
digit sums agree.
-/

namespace ArchytasZX

open Diagram

/-- The map of digits: the last coordinate of a split leg picks its image. -/
def φS (img : Port → List Port) (g : Port → Bool) (p : Port) (t : List ℕ) : Port × List ℕ :=
  if g p then ((img p).getD (t.getLastD 0) Port.dflt, t.dropLast) else ((img p).headD Port.dflt, t)

theorem mem_zip_idx {α β : Type*} {l₁ : List α} {l₂ : List β} {x : α × β} (h : x ∈ l₁.zip l₂) :
    ∃ j, ∃ h₁ : j < l₁.length, ∃ h₂ : j < l₂.length, x = (l₁[j], l₂[j]) := by
  obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 h
  simp only [List.length_zip, lt_min_iff] at hj
  exact ⟨j, hj.1, hj.2, by simp⟩

theorem getElem_mem_zip {α β : Type*} (l₁ : List α) (l₂ : List β) (j : ℕ) (h₁ : j < l₁.length)
    (h₂ : j < l₂.length) : (l₁[j], l₂[j]) ∈ l₁.zip l₂ := by
  have : j < (l₁.zip l₂).length := by simp [h₁, h₂]
  have e := List.getElem_mem this
  simpa using e

section split

variable (D D' : Diagram) (img : Port → List Port) (g : Port → Bool) (ks : ℕ → ℕ) (m : ℕ)

/-- The hypotheses of `evalD_split`. -/
structure SplitOK : Prop where
  legs : D.legs.Nodup
  legs' : D'.legs.Nodup
  bnd' : D'.bnd.Nodup
  memL : ∀ q, q ∈ D'.legs ↔ ∃ p ∈ D.legs, q ∈ img p
  disj : ∀ p ∈ D.legs, ∀ p' ∈ D.legs, ∀ q, q ∈ img p → q ∈ img p' → p = p'
  split : ∀ p ∈ D.legs, g p = true → ∃ c, D.legChain p = c ++ [m] ∧ (img p).length = ks m ∧
    (img p).Nodup ∧ ∀ q ∈ img p, D'.legChain q = c
  keep : ∀ p ∈ D.legs, g p = false → ∃ q, img p = [q] ∧ D'.legChain q = D.legChain p
  dim : ∀ p ∈ D.legs, ∀ q ∈ img p, D'.portDim q = D.portDim p
  wl : ∀ w ∈ D.wires, w.1 ∈ D.legs ∧ w.2 ∈ D.legs ∧ D.legChain w.1 = D.legChain w.2
  wg : ∀ w ∈ D.wires, g w.1 = g w.2
  memW : ∀ w', w' ∈ D'.wires ↔ ∃ w ∈ D.wires, w' ∈ (img w.1).zip (img w.2)
  bl : ∀ p ∈ D.bnd, p ∈ D.legs
  memB : ∀ q, q ∈ D'.bnd ↔ ∃ p ∈ D.bnd, q ∈ img p

variable {D D' img g ks m}

theorem SplitOK.into (h : SplitOK D D' img g ks m) (p : Port) (hp : p ∈ D.legs) (t : List ℕ)
    (ht : t ∈ cidx ks (D.legChain p)) :
    (φS img g p t).1 ∈ img p ∧ (φS img g p t).2 ∈ cidx ks (D'.legChain (φS img g p t).1) := by
  unfold φS
  by_cases hg : g p = true
  · rw [if_pos hg]
    obtain ⟨c, hcp, hlen, -, hc⟩ := h.split p hp hg
    rw [hcp, mem_cidx_single] at ht
    obtain ⟨u, j, rfl, hu, hj⟩ := ht
    have hj' : j < (img p).length := by omega
    simp only [getLastD_snoc, dropLast_snoc, List.getD_eq_getElem _ _ hj']
    exact ⟨List.getElem_mem hj', by rw [hc _ (List.getElem_mem hj')]; exact hu⟩
  · have hg' : g p = false := by simpa using hg
    rw [if_neg hg]
    obtain ⟨q, hq, hc⟩ := h.keep p hp hg'
    rw [hq]
    exact ⟨List.mem_singleton_self _, by simp only [List.headD_cons, hc]; exact ht⟩

theorem SplitOK.onto (h : SplitOK D D' img g ks m) (p : Port) (hp : p ∈ D.legs) (q : Port)
    (hq : q ∈ img p) (u : List ℕ) (hu : u ∈ cidx ks (D'.legChain q)) :
    ∃ t ∈ cidx ks (D.legChain p), φS img g p t = (q, u) := by
  unfold φS
  by_cases hg : g p = true
  · obtain ⟨c, hcp, hlen, -, hc⟩ := h.split p hp hg
    obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hq
    refine ⟨u ++ [j], ?_, ?_⟩
    · rw [hcp, mem_cidx_single]
      exact ⟨u, j, rfl, by rw [← hc _ (List.getElem_mem hj)]; exact hu, by omega⟩
    · rw [if_pos hg]
      simp only [getLastD_snoc, dropLast_snoc, List.getD_eq_getElem _ _ hj]
  · have hg' : g p = false := by simpa using hg
    obtain ⟨q', hq', hc⟩ := h.keep p hp hg'
    rw [hq', List.mem_singleton] at hq
    subst hq
    refine ⟨u, by rw [← hc]; exact hu, ?_⟩
    rw [if_neg hg, hq']
    rfl

theorem SplitOK.inj (h : SplitOK D D' img g ks m) (p : Port) (hp : p ∈ D.legs) (t : List ℕ)
    (ht : t ∈ cidx ks (D.legChain p)) (p' : Port) (hp' : p' ∈ D.legs) (t' : List ℕ)
    (ht' : t' ∈ cidx ks (D.legChain p')) (he : φS img g p t = φS img g p' t') :
    p = p' ∧ t = t' := by
  have h1 := (h.into p hp t ht).1
  have h2 := (h.into p' hp' t' ht').1
  rw [he] at h1
  have hpp := h.disj p hp p' hp' _ h1 h2
  subst hpp
  refine ⟨rfl, ?_⟩
  unfold φS at he
  by_cases hg : g p = true
  · rw [if_pos hg, if_pos hg] at he
    obtain ⟨c, hcp, hlen, hnd, -⟩ := h.split p hp hg
    rw [hcp, mem_cidx_single] at ht ht'
    obtain ⟨u, j, rfl, -, hj⟩ := ht
    obtain ⟨u', j', rfl, -, hj'⟩ := ht'
    simp only [getLastD_snoc, dropLast_snoc, Prod.mk.injEq,
      List.getD_eq_getElem _ _ (show j < (img p).length by omega),
      List.getD_eq_getElem _ _ (show j' < (img p).length by omega)] at he
    have := (List.Nodup.getElem_inj_iff hnd).1 he.1
    rw [this, he.2]
  · rw [if_neg hg, if_neg hg] at he
    exact (Prod.mk.inj he).2

/-- Splitting legs along a leaf box keeps the digit sum. -/
theorem evalD_split (h : SplitOK D D' img g ks m) (E : Env) (σ' : ℕ → List ℕ → ℕ)
    (hP : ∀ τ τ' : ℕ → ℕ, (∀ x ∈ D.dlabs ks, τ' x = τ (relab (φS img g) x)) →
      D'.prodD E ks τ = D.prodD E ks τ') :
    D'.evalD E ks σ' = D.evalD E ks (bndMap D D' (φS img g) σ') := by
  have mem' : ∀ p ∈ D.legs, ∀ t ∈ cidx ks (D.legChain p), (φS img g p t).1 ∈ D'.legs :=
    fun p hp t ht => (h.memL _).2 ⟨p, hp, (h.into p hp t ht).1⟩
  refine evalD_relabelφ D D' E ks ks σ' (φS img g) (dlabs_nodup D ks h.legs)
    (dlabs_nodup D' ks h.legs') h.bnd' (fun p hp t ht => ⟨mem' p hp t ht, (h.into p hp t ht).2⟩)
    (fun q hq u hu => ?_) (fun p hp t ht p' hp' t' ht' he => h.inj p hp t ht p' hp' t' ht' he)
    (fun p hp t ht => by rw [h.dim p hp _ (h.into p hp t ht).1]) h.wl (fun w hw t ht => ?_)
    (fun w' hw' u hu => ?_) h.bl (fun p hp t ht => (h.memB _).2 ⟨p, hp, (h.into p (h.bl p hp) t ht).1⟩)
    (fun q hq u hu => ?_) hP
  · obtain ⟨p, hp, hqp⟩ := (h.memL q).1 hq
    obtain ⟨t, ht, he⟩ := h.onto p hp q hqp u hu
    exact ⟨p, hp, t, ht, he⟩
  · obtain ⟨l1, l2, hcw⟩ := h.wl w hw
    by_cases hg : g w.1 = true
    · have hg2 : g w.2 = true := (h.wg w hw) ▸ hg
      obtain ⟨c, hcp, hlen, -, hc⟩ := h.split _ l1 hg
      obtain ⟨c2, hcp2, hlen2, -, hc2⟩ := h.split _ l2 hg2
      have ht0 := ht
      rw [hcp, mem_cidx_single] at ht
      obtain ⟨u, j, rfl, hu, hj⟩ := ht
      have j1 : j < (img w.1).length := by omega
      have j2 : j < (img w.2).length := by omega
      refine ⟨((img w.1)[j], (img w.2)[j]), (h.memW _).2 ⟨w, hw, getElem_mem_zip _ _ j j1 j2⟩, u,
        by rw [hc _ (List.getElem_mem j1)]; exact hu, ?_, ?_⟩
      · simp only [φS, if_pos hg, getLastD_snoc, dropLast_snoc, List.getD_eq_getElem _ _ j1]
      · simp only [φS, if_pos hg2, getLastD_snoc, dropLast_snoc, List.getD_eq_getElem _ _ j2]
    · have hg1 : g w.1 = false := by simpa using hg
      have hg2 : g w.2 = false := (h.wg w hw) ▸ hg1
      obtain ⟨q1, hq1, hc1⟩ := h.keep _ l1 hg1
      obtain ⟨q2, hq2, hc2⟩ := h.keep _ l2 hg2
      refine ⟨(q1, q2), (h.memW _).2 ⟨w, hw, by rw [hq1, hq2]; simp⟩, t,
        by simp only; rw [hc1]; exact ht, ?_, ?_⟩
      · simp only [φS, hg1, Bool.false_eq_true, ↓reduceIte, hq1, List.headD_cons]
      · simp only [φS, hg2, Bool.false_eq_true, ↓reduceIte, hq2, List.headD_cons]
  · obtain ⟨w, hw, hz⟩ := (h.memW w').1 hw'
    obtain ⟨j, j1, j2, rfl⟩ := mem_zip_idx hz
    obtain ⟨l1, l2, hcw⟩ := h.wl w hw
    simp only at hu
    obtain ⟨t, ht, he⟩ := h.onto _ l1 _ (List.getElem_mem j1) u hu
    refine ⟨w, hw, t, ht, he, ?_⟩
    by_cases hg : g w.1 = true
    · have hg2 : g w.2 = true := (h.wg w hw) ▸ hg
      obtain ⟨c, hcp, hlen, hnd, hc⟩ := h.split _ l1 hg
      unfold φS at he ⊢
      rw [if_pos hg] at he
      rw [if_pos hg2]
      rw [hcp, mem_cidx_single] at ht
      obtain ⟨u', j', rfl, -, hj'⟩ := ht
      simp only [getLastD_snoc, dropLast_snoc, Prod.mk.injEq,
        List.getD_eq_getElem _ _ (show j' < (img w.1).length by omega)] at he
      have := (List.Nodup.getElem_inj_iff hnd).1 he.1
      subst this
      simp only [getLastD_snoc, dropLast_snoc, List.getD_eq_getElem _ _ j2, he.2]
    · have hg1 : g w.1 = false := by simpa using hg
      have hg2 : g w.2 = false := (h.wg w hw) ▸ hg1
      obtain ⟨q1, hq1, -⟩ := h.keep _ l1 hg1
      obtain ⟨q2, hq2, -⟩ := h.keep _ l2 hg2
      unfold φS at he ⊢
      simp only [hg1, hg2, Bool.false_eq_true, ↓reduceIte, hq1, List.headD_cons, Prod.mk.injEq] at he ⊢
      simp only [hq2] at j2 ⊢
      have : j = 0 := by simp at j2; omega
      subst this
      simp [he.2]
  · obtain ⟨p, hp, hqp⟩ := (h.memB q).1 hq
    obtain ⟨t, ht, he⟩ := h.onto p (h.bl p hp) q hqp u hu
    exact ⟨p, hp, t, ht, he⟩

end split

end ArchytasZX
