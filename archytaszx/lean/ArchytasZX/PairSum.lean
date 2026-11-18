/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PairFam

/-!
# The pair product as a factor over the bundled sum

Over the copies' values `x 0 .. x (K - 1)`, `S = Σ_j x_j` and `Q = Σ_j x_j²`:
`two_pairS`: `2 Σ_{j<k} x_j x_k + Q = S²` over `ℕ`, before any reduction; `pair_fou`: the product of
the Fourier entries over the pairs `j < k` is `(√D)⁻¹ ^ (K (K - 1) / 2) ω_{2D} ^ (S² - Q)` for every
`D > 0`, odd or even (`om_two`: `ω_{2D}² = ω_D`). `eval_pairNet_sum`: the pair box's network with
both pair legs pinned to one slot `s` is its template at each copy's slots times that factor, `S`
and `Q` read at the copies' slot `s`.
-/

namespace ArchytasZX

open Diagram

/-- `Σ_{k<K} x_k Σ_{j<k} x_j`. -/
def pairS (x : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | K + 1 => pairS x K + x K * ((List.range K).map x).sum

theorem two_pairS (x : ℕ → ℕ) (K : ℕ) :
    2 * pairS x K + ((List.range K).map fun j => x j ^ 2).sum = ((List.range K).map x).sum ^ 2 := by
  induction K with
  | zero => simp [pairS]
  | succ K ih =>
    simp only [pairS, List.range_succ, List.map_append, List.sum_append, List.map_cons,
      List.map_nil, List.sum_cons, List.sum_nil, add_zero]
    nlinarith [ih]

theorem later_succ (K j : ℕ) (hj : j < K) : later (K + 1) j = later K j ++ [K] := by
  unfold later
  rw [show K + 1 - 1 - j = (K - 1 - j) + 1 by omega, List.range'_concat]
  congr 2
  omega

theorem later_self (K : ℕ) : later (K + 1) K = [] := by simp [later]

/-- The product over the pairs `j < k`, by the first copy then by the second. -/
theorem tri_prod (f : ℕ → ℕ → ℂ) (K : ℕ) :
    ((List.range K).map fun j => ((later K j).map (f j)).prod).prod =
      ((List.range K).map fun k => ((List.range k).map fun j => f j k).prod).prod := by
  induction K with
  | zero => simp
  | succ K ih =>
    rw [List.range_succ, List.map_append, List.map_append, List.prod_append, List.prod_append]
    simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, later_self]
    rw [List.map_congr_left (g := fun j => ((later K j).map (f j)).prod * f j K)
        fun j hj => by rw [later_succ K j (List.mem_range.1 hj)]; simp, List.prod_map_mul, ih]

theorem om_two (d : ℕ) (hd : 0 < d) : om (2 * d) ^ 2 = om d := by
  unfold om
  rw [← Complex.exp_nat_mul]
  congr 1
  have : (d : ℂ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  push_cast
  field_simp

theorem prod_fou_row (d : ℕ) (x : ℕ → ℕ) (c : ℕ) (l : List ℕ) :
    (l.map fun j => fou d c (x j)).prod =
      om d ^ (c * (l.map x).sum) * ((Real.sqrt d : ℂ)⁻¹) ^ l.length := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.prod_cons, List.sum_cons, List.length_cons]
    rw [ih]
    simp only [fou]
    rw [mul_add, pow_add, pow_succ]
    ring

theorem npair_succ (K : ℕ) : (K + 1) * (K + 1 - 1) / 2 = K * (K - 1) / 2 + K := by
  have : (K + 1) * (K + 1 - 1) = K * (K - 1) + 2 * K := by
    cases K with
    | zero => simp
    | succ k => simp only [Nat.add_sub_cancel]; ring
  rw [this, Nat.add_mul_div_left _ _ (by norm_num : 0 < 2)]

/-- The Fourier entries over the pairs `j < k` as one factor over `S = Σ x_j` and `Q = Σ x_j²`. -/
theorem pair_fou (d : ℕ) (hd : 0 < d) (x : ℕ → ℕ) (K : ℕ) :
    ((List.range K).map fun j => ((later K j).map fun k => fou d (x k) (x j)).prod).prod =
      ((Real.sqrt d : ℂ)⁻¹) ^ (K * (K - 1) / 2) *
        om (2 * d) ^ (((List.range K).map x).sum ^ 2 - ((List.range K).map fun j => x j ^ 2).sum) := by
  rw [tri_prod (fun j k => fou d (x k) (x j)) K]
  have h2 := two_pairS x K
  rw [show ((List.range K).map x).sum ^ 2 - ((List.range K).map fun j => x j ^ 2).sum =
    2 * pairS x K by omega, pow_mul, om_two d hd]
  clear h2
  induction K with
  | zero => simp [pairS]
  | succ K ih =>
    rw [List.range_succ, List.map_append, List.prod_append, ih]
    simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one]
    rw [prod_fou_row, List.length_range, npair_succ, pairS, pow_add, pow_add]
    ring

/-- The pair box's network with both pair legs pinned to the slot `s`: the template at each
copy's slots times `(√D)⁻¹ ^ (K (K - 1) / 2) ω_{2D} ^ (S² - Q)`, `S` and `Q` over the copies'
values at `s`. -/
theorem eval_pairNet_sum {D : Diagram} {p : DPair} {b : DBox} (h : D.PairOK p b = true) (E : Env)
    (hE : ∀ x, labDim E x = E.dimv 0) (hd : 0 < E.dimv 0) (K : ℕ) (cl cr : List (ℕ × ℕ)) (s : ℕ)
    (hcl : D.pinOK p p.left cl s = true) (hcr : D.pinOK p p.right cr s = true) (σ : ℕ → ℕ) :
    eval E (D.pairNet p K) σ =
      ((List.range K).map fun j =>
        eval E (D.tmpl p) (σ ∘ shJ K D.outs.length D.wires.length j)).prod *
      (((Real.sqrt (E.dimv 0) : ℂ)⁻¹) ^ (K * (K - 1) / 2) *
        om (2 * E.dimv 0) ^ (((List.range K).map fun j => σ (j * D.outs.length + s)).sum ^ 2 -
          ((List.range K).map fun j => σ (j * D.outs.length + s) ^ 2).sum)) := by
  rw [eval_pairNet h E hE K cl cr s s hcl hcr σ,
    ← pair_fou (E.dimv 0) hd (fun j => σ (j * D.outs.length + s)) K]

end ArchytasZX
