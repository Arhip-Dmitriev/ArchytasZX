/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumRules

/-!
# Reduced sums compared through their phase difference

After `canonH` steps the rows are made but the phase still holds the pivots. `diffP` is the
left phase minus the right one with each row's pivot substituted, in order. `finalP_sound`
compares two reduced sums with one list of rows whose difference is a constant.
-/

namespace ArchytasZX
namespace PSem

open Complex

/-- `D` with each pivot replaced by the rest of its row, in order. -/
def substRows (M : ℕ) : List ℕ → List F2 → Poly → Poly
  | v :: vs, r :: rs, D => substRows M vs rs (normP M (substP M v (r.erase [v]) D))
  | _, _, D => D

/-- Whether the pivots and rows have one length and each row is pivotable at its pivot. -/
def rowsOK : List ℕ → List F2 → Bool
  | [], [] => true
  | v :: vs, r :: rs => pivotable v r && rowsOK vs rs
  | _, _ => false

theorem pn_substRows (M : ℕ) (hM : 2 ≤ M) (σ : ℕ → ZMod 2) :
    ∀ (vs : List ℕ) (rs : List F2) (D : Poly), rowsOK vs rs = true →
      (∀ r ∈ rs, fz σ r = 0) → peq M (pn σ (substRows M vs rs D)) (pn σ D)
  | [], [], D, _, _ => peq_refl M _
  | v :: vs, r :: rs, D, h, hr => by
    simp only [rowsOK, Bool.and_eq_true] at h
    obtain ⟨hpv, hrs⟩ := h
    simp only [pivotable, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at hpv
    have hr0 : fz σ r = 0 := hr r List.mem_cons_self
    have hqv : fz σ r = σ v + fz σ (r.erase [v]) := by
      rw [fz_perm _ (List.perm_cons_erase hpv.1), fz_cons]
      simp [mz]
    have hv : fz σ (r.erase [v]) = σ v :=
      (z2_eq_of_add _ _ (by rw [← hqv]; exact hr0)).symm
    have hτ : Function.update σ v (fz σ (r.erase [v])) = σ := by
      rw [hv, Function.update_eq_self]
    have ih := pn_substRows M hM σ vs rs (normP M (substP M v (r.erase [v]) D)) hrs
      (fun q hq => hr q (List.mem_cons_of_mem _ hq))
    simp only [substRows]
    refine peq_trans ih (peq_trans (pn_normP σ M _) ?_)
    have := pn_substP σ M v hM (r.erase [v]) D
    rwa [hτ] at this
  | [], _ :: _, _, h, _ => by simp [rowsOK] at h
  | _ :: _, [], _, h, _ => by simp [rowsOK] at h

/-- The left phase minus the right one, each pivot of `vs` replaced by the rest of its row. -/
def diffP (M : ℕ) (vs : List ℕ) (sL sR : Data) : Poly :=
  substRows M vs sL.rows (normP M (sL.phase ++ negP M sR.phase))

/-- Whether both sums have no path variable, only zero constraints, one list of rows pivotable
at `vs` and a constant phase difference. -/
def finalPOK (M : ℕ) (vs : List ℕ) (sL sR : Data) : Bool :=
  decide (sL.path = []) && decide (sR.path = []) && sL.cons.all (· = []) &&
    sR.cons.all (· = []) && decide (sL.rows = sR.rows) && rowsOK vs sL.rows &&
    decide (strip (diffP M vs sL sR) = [])

theorem finalP_sound (M : ℕ) (hM : 2 ≤ M) (cL cR : ℂ) (vs : List ℕ) (sL sR : Data)
    (σ : ℕ → ZMod 2) (k : ℕ) (h : finalPOK M vs sL sR = true)
    (hk : constK (diffP M vs sL sR) = k) (hc : cL * om M ^ k = cR) :
    val M cL sL σ = val M cR sR σ := by
  subst hk
  simp only [finalPOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨hpL, hpR⟩, hcL⟩, hcR⟩, hrows⟩, hok⟩, hD⟩ := h
  have hM0 : 0 < M := by omega
  unfold val summand
  rw [hpL, hpR]
  simp only [bsum]
  have hiff : ∀ (s : Data), (∀ q ∈ s.cons, q = []) →
      ((∀ q ∈ s.cons ++ s.rows, fz σ q = 0) ↔ (∀ q ∈ s.rows, fz σ q = 0)) := by
    intro s hs
    simp only [List.mem_append]
    constructor
    · intro H q hq; exact H q (Or.inr hq)
    · rintro H q (hq | hq)
      · rw [hs q hq]; rfl
      · exact H q hq
  rw [if_congr (hiff sL (fun q hq => by simpa using hcL q hq)) rfl rfl,
    if_congr (hiff sR (fun q hq => by simpa using hcR q hq)) rfl rfl, ← hrows]
  split_ifs with hR
  · obtain ⟨a, ha⟩ : peq M (pn σ (diffP M vs sL sR))
        (pn σ (normP M (sL.phase ++ negP M sR.phase))) :=
      pn_substRows M hM σ vs sL.rows _ hok hR
    obtain ⟨b, hb⟩ := pn_normP σ M (sL.phase ++ negP M sR.phase)
    obtain ⟨e, he⟩ := pn_negP σ M hM0 sR.phase
    have h2 := pn_strip σ (diffP M vs sL sR)
    rw [hD, pn_nil, add_zero] at h2
    rw [pn_append] at hb
    have key : peq M (pn σ sL.phase) (pn σ sR.phase + (constK (diffP M vs sL sR) : ℤ)) := by
      refine ⟨-(a + b + e), ?_⟩
      linear_combination -ha - hb - he + h2
    rw [ex_peq hM0 key]
    push_cast
    rw [add_div, ex_add, ← om_eq_ex]
    linear_combination ex (((pn σ sR.phase : ℤ) : ℝ) / M) * hc
  · rfl

end PSem
end ArchytasZX
