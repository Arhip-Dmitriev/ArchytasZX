/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.WRules

/-!
# W associativity over groups of outputs

A W node whose outputs `A ++ xs` meet, at each value `x_j` of `xs`, a weight `w x_j` and a W node
with input `x_j` and outputs `B_j`: summed over the values `xs` (`sumW`), this is the W node on
`A ++ B_0 ++ ... ++ B_(M-1)` with the weight `w i` on the outputs of the groups (`wAssoc`), for
every number `M` of groups; with no other output (`A = []`) it is `w i` times the W node on the
joined groups (`wAssoc_nil`).
-/

namespace ArchytasZX

open Complex Finset

/-- The sum over value lists of length `M`, each value below `d`. -/
noncomputable def sumW (d : ℕ) : ℕ → (List ℕ → ℂ) → ℂ
  | 0, f => f []
  | M + 1, f => ∑ x ∈ range d, sumW d M fun xs => f (x :: xs)

theorem sumW_congr (d : ℕ) : ∀ (M : ℕ) (f g : List ℕ → ℂ), (∀ xs, f xs = g xs) →
    sumW d M f = sumW d M g
  | 0, _, _, h => h []
  | M + 1, f, g, h => by
    simp only [sumW]
    exact Finset.sum_congr rfl fun x _ => sumW_congr d M _ _ fun xs => h (x :: xs)

theorem sumW_smul (d : ℕ) (c : ℂ) : ∀ (M : ℕ) (f : List ℕ → ℂ),
    sumW d M (fun xs => c * f xs) = c * sumW d M f
  | 0, _ => rfl
  | M + 1, f => by
    simp only [sumW, Finset.mul_sum]
    exact Finset.sum_congr rfl fun x _ => sumW_smul d c M _

/-- The product over the groups of the weight at the group's value times the W node from that
value to the group. -/
noncomputable def wTerm (d : ℕ) (w : ℕ → ℂ) : List ℕ → List (List ℕ) → ℂ
  | x :: xs, B :: Bs => w x * tW d B x * wTerm d w xs Bs
  | _, _ => 1

theorem nz_append_single (A : List ℕ) (x : ℕ) : nz (A ++ [x]) = [] ↔ nz A = [] ∧ x = 0 := by
  simp only [nz_nil_iff, List.mem_append, List.mem_singleton]
  constructor
  · intro h
    exact ⟨fun o ho => h o (Or.inl ho), h x (Or.inr rfl)⟩
  · rintro ⟨h, hx⟩ o (ho | rfl)
    · exact h o ho
    · exact hx

theorem tW_zero_ind (d : ℕ) (B : List ℕ) : tW d B 0 = if nz B = [] then 1 else 0 := by
  rw [tW_eq]
  simp

/-- W associativity over `M` groups of outputs, the weights on the groups. -/
theorem wAssoc (d : ℕ) (hd : 0 < d) (w : ℕ → ℂ) (hw : w 0 = 1) (i : ℕ) (hi : i < d) :
    ∀ (Bs : List (List ℕ)) (A : List ℕ),
      sumW d Bs.length (fun xs => tW d (A ++ xs) i * wTerm d w xs Bs) =
        tW d A i * tW d Bs.flatten 0 +
          (if i = 0 then 0 else (if nz A = [] then 1 else 0) * (w i * tW d Bs.flatten i))
  | [], A => by
    have e0 : tW d [] 0 = 1 := by simp [tW_zero_ind]
    by_cases h0 : i = 0
    · subst h0
      simp [sumW, wTerm, e0]
    · have e1 : tW d [] i = 0 := by simp [tW_eq, h0]
      simp [sumW, wTerm, e0, e1, h0]
  | B :: Bs, A => by
    have ih := wAssoc d hd w hw i hi Bs
    simp only [List.length_cons, sumW, wTerm]
    have step : ∀ x, sumW d Bs.length (fun xs =>
        tW d (A ++ x :: xs) i * (w x * tW d B x * wTerm d w xs Bs)) =
        (w x * tW d B x) * (tW d (x :: A) i * tW d Bs.flatten 0 +
          (if i = 0 then 0 else (if nz A = [] ∧ x = 0 then 1 else 0) *
            (w i * tW d Bs.flatten i))) := by
      intro x
      rw [sumW_congr d _ _ (fun xs => (w x * tW d B x) * (tW d ((A ++ [x]) ++ xs) i *
        wTerm d w xs Bs)) (fun xs => by rw [List.append_assoc, List.singleton_append]; ring),
        sumW_smul, ih (A ++ [x]), tW_perm d (A ++ [x]) (x :: A) i (List.perm_append_singleton x A)]
      simp only [nz_append_single]
    rw [Finset.sum_congr rfl fun x _ => step x]
    have hsplit : ∀ x, (w x * tW d B x) * (tW d (x :: A) i * tW d Bs.flatten 0 +
        (if i = 0 then 0 else (if nz A = [] ∧ x = 0 then 1 else 0) * (w i * tW d Bs.flatten i))) =
        tW d (x :: A) i * (w x * tW d B x * tW d Bs.flatten 0) +
          (if x = 0 then (w x * tW d B x) *
            (if i = 0 then 0 else (if nz A = [] then 1 else 0) * (w i * tW d Bs.flatten i))
          else 0) := by
      intro x
      by_cases hx : x = 0
      · subst hx; simp only [and_true, ↓reduceIte]; ring
      · simp only [hx, and_false, ↓reduceIte, add_zero]
        split_ifs <;> ring
    have h0d : 0 ∈ range d := Finset.mem_range.2 hd
    rw [Finset.sum_congr rfl fun x _ => hsplit x, Finset.sum_add_distrib,
      sum_tW_cons d hd A i _ hi, Finset.sum_ite_eq']
    simp only [h0d, ↓reduceIte, hw]
    rw [List.flatten_cons, tW_append d B Bs.flatten 0 hd, tW_append d B Bs.flatten i hi,
      tW_zero_ind d B]
    by_cases h0 : i = 0
    · simp [h0]
    · simp only [h0, ↓reduceIte]
      split_ifs <;> ring

/-- With no other output, `w i` times the W node on the joined groups. -/
theorem wAssoc_nil (d : ℕ) (hd : 0 < d) (w : ℕ → ℂ) (hw : w 0 = 1) (i : ℕ) (hi : i < d)
    (Bs : List (List ℕ)) :
    sumW d Bs.length (fun xs => tW d xs i * wTerm d w xs Bs) = w i * tW d Bs.flatten i := by
  have h := wAssoc d hd w hw i hi Bs []
  simp only [List.nil_append] at h
  rw [h]
  by_cases h0 : i = 0
  · subst h0
    simp [tW_eq, hw]
  · simp [tW_eq, h0]

end ArchytasZX
