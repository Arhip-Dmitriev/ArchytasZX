/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumDigits

/-!
# The qubit W node as a fragment

On one-digit labels the W node with input `i` and outputs `os` is the fragment with no path
variable and the constraints: the digit of `i` plus the digits of `os` is 0, and the product of
the digits of any two outputs is 0 (`wFrag`, `wnode_frag`). `tW_bits`: on values below 2,
`tW 2 vs i` is 1 exactly when `i` plus the count of ones of `vs` is even and that count is at
most 1.
-/

namespace ArchytasZX
namespace PSem

open Complex Finset

/-- Each entry paired with each later entry. -/
def pairsOf {α : Type} : List α → List (α × α)
  | [] => []
  | a :: as => as.map (a, ·) ++ pairsOf as

theorem pairsOf_map {α β : Type} (f : α → β) :
    ∀ l : List α, pairsOf (l.map f) = (pairsOf l).map (Prod.map f f)
  | [] => rfl
  | a :: as => by
    simp only [List.map_cons, pairsOf, List.map_append, List.map_map, pairsOf_map f as]
    rfl

/-- The qubit W node's fragment: the input digit plus the output digits, and each pair of output
digits' product. -/
def wFrag (dig : ℕ → List ℕ) (os : List ℕ) (i : ℕ) : Frag :=
  ⟨[], [], ((dig i ++ os.flatMap dig).map fun x => [x]) ::
    (pairsOf (os.flatMap dig)).map fun p => [[p.1, p.2]], 1⟩

/-- The count of nonzero entries. -/
def ones (vs : List ℕ) : ℕ := (vs.filter (· ≠ 0)).length

theorem ones_cons (v : ℕ) (vs : List ℕ) (hv : v < 2) :
    ones (v :: vs) = v + ones vs := by
  unfold ones
  rcases bit_cases v hv with rfl | rfl
  · simp
  · simp
    omega

theorem sum_ones (vs : List ℕ) (hv : ∀ v ∈ vs, v < 2) :
    (vs.map fun v => ((v : ℕ) : ZMod 2)).sum = ((ones vs : ℕ) : ZMod 2) := by
  induction vs with
  | nil => rfl
  | cons v vs ih =>
    rw [List.map_cons, List.sum_cons, ih fun w hw => hv w (List.mem_cons_of_mem _ hw),
      ones_cons v vs (hv v List.mem_cons_self)]
    push_cast
    ring

theorem pairs_ones (vs : List ℕ) (hv : ∀ v ∈ vs, v < 2) :
    (∀ p ∈ pairsOf vs, ((p.1 : ℕ) : ZMod 2) * ((p.2 : ℕ) : ZMod 2) = 0) ↔ ones vs ≤ 1 := by
  induction vs with
  | nil => simp [pairsOf, ones]
  | cons v vs ih =>
    have hvs : ∀ w ∈ vs, w < 2 := fun w hw => hv w (List.mem_cons_of_mem _ hw)
    rw [ones_cons v vs (hv v List.mem_cons_self)]
    simp only [pairsOf, List.mem_append, List.mem_map]
    have hzero : (∀ w ∈ vs, w = 0) ↔ ones vs = 0 := by
      unfold ones
      rw [List.length_eq_zero_iff, List.filter_eq_nil_iff]
      simp
    rcases bit_cases v (hv v List.mem_cons_self) with rfl | rfl
    · constructor
      · intro H
        have := (ih hvs).1 fun p hp => H p (Or.inr hp)
        omega
      · rintro H p (⟨w, _, rfl⟩ | hp)
        · simp
        · exact (ih hvs).2 (by omega) p hp
    · constructor
      · intro H
        have h0 : ∀ w ∈ vs, w = 0 := by
          intro w hw
          have := H (1, w) (Or.inl ⟨w, hw, rfl⟩)
          rcases bit_cases w (hvs w hw) with rfl | rfl
          · rfl
          · exact absurd this (by decide)
        have := hzero.1 h0
        omega
      · intro H
        have h0 := hzero.2 (by omega)
        rintro p (⟨w, hw, rfl⟩ | hp)
        · simp [h0 w hw]
        · exact (ih hvs).2 (by omega) p hp

theorem cond0 (c : ℕ) : c = 0 ↔ ((0 : ℕ) : ZMod 2) + ((c : ℕ) : ZMod 2) = 0 ∧ c ≤ 1 := by
  rcases (show c = 0 ∨ c = 1 ∨ 2 ≤ c by omega) with rfl | rfl | h
  · decide
  · decide
  · exact ⟨fun h' => by omega, fun h' => by omega⟩

theorem cond1 (c : ℕ) : c = 1 ↔ ((1 : ℕ) : ZMod 2) + ((c : ℕ) : ZMod 2) = 0 ∧ c ≤ 1 := by
  rcases (show c = 0 ∨ c = 1 ∨ 2 ≤ c by omega) with rfl | rfl | h
  · decide
  · decide
  · exact ⟨fun h' => by omega, fun h' => by omega⟩

theorem tW_bits (vs : List ℕ) (i : ℕ) (hv : ∀ v ∈ vs, v < 2) (hi : i < 2) :
    tW 2 vs i = if ((i : ℕ) : ZMod 2) + (vs.map fun v => ((v : ℕ) : ZMod 2)).sum = 0 ∧
      ∀ p ∈ pairsOf vs, ((p.1 : ℕ) : ZMod 2) * ((p.2 : ℕ) : ZMod 2) = 0 then 1 else 0 := by
  rw [sum_ones vs hv, if_congr (and_congr Iff.rfl (pairs_ones vs hv)) rfl rfl]
  have hall : (∀ o ∈ vs, o = 0) ↔ ones vs = 0 := by
    unfold ones
    rw [List.length_eq_zero_iff, List.filter_eq_nil_iff]
    simp
  have hone : vs.filter (· ≠ 0) = [1] ↔ ones vs = 1 := by
    unfold ones
    rw [show [1] = List.replicate 1 1 from rfl, List.eq_replicate_iff]
    constructor
    · exact fun h => h.1
    · intro h
      refine ⟨h, fun b hb => ?_⟩
      have hb' := List.mem_filter.1 hb
      have := hv b hb'.1
      have : b ≠ 0 := by simpa using hb'.2
      omega
  unfold tW
  rcases bit_cases i hi with rfl | rfl
  · simp only [↓reduceIte]
    exact if_congr (hall.trans (cond0 _)) rfl rfl
  · simp only [one_ne_zero, ↓reduceIte, Nat.one_lt_two, true_and]
    exact if_congr (hone.trans (cond1 _)) rfl rfl

theorem dbits_flat {own : Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (hown : ownOK own labs dig = true) (σ : ℕ → ℕ) :
    ∀ os : List ℕ, (∀ l ∈ os, l ∈ labs ∧ (dig l).length = 1 ∧ σ l < 2) →
      (os.flatMap dig).map (dbits own σ) = os.map fun l => ((σ l : ℕ) : ZMod 2)
  | [], _ => rfl
  | l :: os, h => by
    obtain ⟨hl, hlen, hσ⟩ := h l List.mem_cons_self
    obtain ⟨x, hx⟩ : ∃ x, dig l = [x] := by
      match hd : dig l, hlen with
      | [x], _ => exact ⟨x, rfl⟩
    have hget := dbits_get hown σ hl (i := 0) (by rw [hlen]; norm_num)
    simp only [hx, List.getElem_cons_zero, digit, pow_zero, Nat.div_one,
      Nat.mod_eq_of_lt hσ] at hget
    rw [List.flatMap_cons, hx, List.singleton_append, List.map_cons, List.map_cons, hget,
      dbits_flat hown σ os fun l' hl' => h l' (List.mem_cons_of_mem _ hl')]

theorem wnode_frag (E : Env) (σ : ℕ → ℕ) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < 2 ^ (dig l).length)
    (n : Node) (hkind : n.kind = .W) (hd : E.dimv n.dim = 2) (i : ℕ) (hi : n.ins = [i])
    (hls : ∀ l ∈ i :: n.outs, l ∈ labs ∧ (dig l).length = 1)
    (M : ℕ) : ArchytasZX.val E σ n = fval M (wFrag dig n.outs i) (dbits own σ) := by
  have hl : ∀ l ∈ i :: n.outs, l ∈ labs ∧ (dig l).length = 1 ∧ σ l < 2 := by
    intro l hl
    obtain ⟨h1, h2⟩ := hls l hl
    exact ⟨h1, h2, by simpa [h2] using hσ l h1⟩
  have hv : ArchytasZX.val E σ n = tW 2 (n.outs.map σ) (σ i) := by
    unfold ArchytasZX.val
    rw [hkind, hi]
    simp only [hd, List.map_cons, List.map_nil]
  have hall := dbits_flat hown σ (i :: n.outs) hl
  have hos := dbits_flat hown σ n.outs fun l hl' => hl l (List.mem_cons_of_mem _ hl')
  rw [hv, tW_bits _ _ (by
      intro v hv'
      obtain ⟨l, hl', rfl⟩ := List.mem_map.1 hv'
      exact (hl l (List.mem_cons_of_mem _ hl')).2.2) (hl i List.mem_cons_self).2.2]
  unfold fval val wFrag
  simp only [bsum, summand, List.append_nil, pn_nil, Int.cast_zero, zero_div, ex_zero, mul_one]
  refine if_congr ?_ rfl rfl
  have hsum : fz (dbits own σ) ((dig i ++ n.outs.flatMap dig).map fun x => [x]) =
      ((σ i : ℕ) : ZMod 2) + (n.outs.map σ |>.map fun v => ((v : ℕ) : ZMod 2)).sum := by
    have h2 : ∀ xs : List ℕ,
        fz (dbits own σ) (xs.map fun x => [x]) = (xs.map (dbits own σ)).sum := by
      intro xs
      unfold fz
      rw [List.map_map]
      congr 1
      apply List.map_congr_left
      intro x _
      simp [mz]
    rw [h2, ← List.flatMap_cons, hall, List.map_cons, List.sum_cons, List.map_map]
    rfl
  have hpair : (∀ p ∈ pairsOf (n.outs.flatMap dig), fz (dbits own σ) [[p.1, p.2]] = 0) ↔
      ∀ p ∈ pairsOf (n.outs.map σ), ((p.1 : ℕ) : ZMod 2) * ((p.2 : ℕ) : ZMod 2) = 0 := by
    have h3 : ∀ p : ℕ × ℕ, fz (dbits own σ) [[p.1, p.2]] =
        (Prod.map (dbits own σ) (dbits own σ) p).1 *
          (Prod.map (dbits own σ) (dbits own σ) p).2 := by
      intro p
      simp [fz, mz]
    have h4 : pairsOf (n.outs.map fun l => ((σ l : ℕ) : ZMod 2)) =
        (pairsOf (n.outs.map σ)).map
          (Prod.map (fun v => ((v : ℕ) : ZMod 2)) fun v => ((v : ℕ) : ZMod 2)) := by
      rw [← pairsOf_map, List.map_map]
      rfl
    have e : (pairsOf (n.outs.flatMap dig)).map (Prod.map (dbits own σ) (dbits own σ)) =
        (pairsOf (n.outs.map σ)).map
          (Prod.map (fun v => ((v : ℕ) : ZMod 2)) fun v => ((v : ℕ) : ZMod 2)) := by
      rw [← pairsOf_map, hos, h4]
    simp only [h3]
    constructor
    · intro H p hp
      have hq : Prod.map (fun v => ((v : ℕ) : ZMod 2)) (fun v => ((v : ℕ) : ZMod 2)) p ∈
          (pairsOf (n.outs.flatMap dig)).map (Prod.map (dbits own σ) (dbits own σ)) := by
        rw [e]
        exact List.mem_map.2 ⟨p, hp, rfl⟩
      obtain ⟨p', hp', hpe⟩ := List.mem_map.1 hq
      have := H p' hp'
      rw [hpe] at this
      exact this
    · intro H p hp
      have hq : Prod.map (dbits own σ) (dbits own σ) p ∈ (pairsOf (n.outs.map σ)).map
          (Prod.map (fun v => ((v : ℕ) : ZMod 2)) fun v => ((v : ℕ) : ZMod 2)) := by
        rw [← e]
        exact List.mem_map.2 ⟨p, hp, rfl⟩
      obtain ⟨p', hp', hpe⟩ := List.mem_map.1 hq
      rw [← hpe]
      exact H p' hp'
  rw [List.forall_mem_cons, List.forall_mem_map, hsum, hpair]

end PSem
end ArchytasZX
