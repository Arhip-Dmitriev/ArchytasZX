/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Net
import ArchytasZX.Rules
import ArchytasZX.Closed
import ArchytasZX.ZXRules
import ArchytasZX.Cyclo

/-!
# The generators of [P1] and [P2] in their papers' conventions

Each generator's entry as the paper defines it, and its equality with the tensor of the
ArchytasZX generator it maps to, with the scalar of docs/COMPLETENESS.txt, section 2, for
every dimension `d`. Leg values are read below `d`.

[P1] B. Poór et al., Completeness for arbitrary finite dimensions of ZXW-calculus (LICS 2023).
[P2] Q. Wang, B. Poór, R. A. Shaikh, Completeness of qufinite ZXW calculus (arXiv:2309.13014).
-/

namespace ArchytasZX

open Complex Finset

/-- [P1] Hadamard box: `(1/√d) ω^{jk}`. -/
noncomputable def paperH (d j k : ℕ) : ℂ := (1 / (Real.sqrt d : ℂ)) * om d ^ (j * k)

theorem paperH_eq (d j k : ℕ) : paperH d j k = fou d j k := by
  simp only [paperH, fou]; ring

/-- [P1] green circle spider with phases `φ` (turns): `∑_j e^{2πiφ(j)} |j..j⟩⟨j..j|`. -/
noncomputable def paperZ (d : ℕ) (φ : ℕ → ℝ) (vs : List ℕ) : ℂ :=
  ∑ j ∈ range d, ex (φ j) * (vs.map fun v => if v = j then (1 : ℂ) else 0).prod

theorem prod_indicator (vs : List ℕ) (j : ℕ) :
    (vs.map fun v => if v = j then (1 : ℂ) else 0).prod = if ∀ v ∈ vs, v = j then 1 else 0 := by
  induction vs with
  | nil => simp
  | cons v vs ih =>
    simp only [List.map_cons, List.prod_cons, ih, List.mem_cons, forall_eq_or_imp]
    by_cases h : v = j <;> by_cases h' : ∀ a ∈ vs, a = j <;> simp [h, h']

theorem paperZ_eq (d : ℕ) (φ : ℕ → ℝ) (vs : List ℕ) : paperZ d φ vs = tZ d φ vs := by
  unfold paperZ tZ
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [prod_indicator]
  split_ifs <;> simp

/-- [P1] W node with one input and two outputs:
`|00⟩⟨0| + ∑_{i=1}^{d-1} (|0i⟩ + |i0⟩)⟨i|`. -/
def paperW (d o₁ o₂ i : ℕ) : ℂ :=
  (if o₁ = 0 ∧ o₂ = 0 ∧ i = 0 then 1 else 0) +
    (if 1 ≤ i ∧ i < d ∧ o₁ = 0 ∧ o₂ = i then 1 else 0) +
    (if 1 ≤ i ∧ i < d ∧ o₁ = i ∧ o₂ = 0 then 1 else 0)

theorem paperW_eq (d o₁ o₂ i : ℕ) : paperW d o₁ o₂ i = tW d [o₁, o₂] i := by
  unfold paperW tW
  by_cases hi : i = 0
  · subst hi; by_cases h1 : o₁ = 0 <;> by_cases h2 : o₂ = 0 <;> simp [h1, h2]
  · have hi1 : 1 ≤ i := Nat.one_le_iff_ne_zero.mpr hi
    by_cases hd : i < d
    · by_cases h1 : o₁ = 0 <;> by_cases h2 : o₂ = 0 <;>
        by_cases h3 : o₁ = i <;> by_cases h4 : o₂ = i <;>
        simp_all [List.filter] <;> (try omega)
    · simp [hi, hd]; (try omega)

/-- [P1] (YT) yellow triangle: `I + ∑_{i=1}^{d-1} |0⟩⟨i|`. -/
def paperT (o i : ℕ) : ℂ := (if o = i then 1 else 0) + (if o = 0 ∧ 1 ≤ i then 1 else 0)

theorem paperT_eq (o i : ℕ) : paperT o i = tT o i := by
  unfold paperT tT
  by_cases h1 : o = i <;> by_cases h2 : o = 0 <;> by_cases h3 : 1 ≤ i <;> simp_all <;>
    (try omega)

/-- [P1] (YT) inverse triangle: `I - ∑_{i=1}^{d-1} |0⟩⟨i|`. -/
def paperTi (o i : ℕ) : ℂ := (if o = i then 1 else 0) - (if o = 0 ∧ 1 ≤ i then 1 else 0)

theorem paperTi_eq (o i : ℕ) : paperTi o i = tTi o i := by
  unfold paperTi tTi
  by_cases h1 : o = i <;> by_cases h2 : o = 0 <;> by_cases h3 : 1 ≤ i <;> simp_all <;>
    (try omega)

/-- [P2] dimension splitter, input `de`, outputs `d`, `e`: `∑_{i<d} ∑_{j<e} |i, j⟩⟨i e + j|`. -/
def paperSplit (d e k i j : ℕ) : ℂ := if i < d ∧ j < e ∧ k = i * e + j then 1 else 0

theorem paperSplit_eq (d e k i j : ℕ) : paperSplit d e k i j = tB d e k i j := rfl

theorem prod_fou_out (d k : ℕ) (os : List ℕ) :
    (os.map fun o => fou d o k).prod = om d ^ (k * os.sum) / (Real.sqrt d : ℂ) ^ os.length := by
  induction os with
  | nil => simp
  | cons o os ih =>
    rw [List.map_cons, List.prod_cons, ih, List.sum_cons, List.length_cons]
    rw [show k * (o + os.sum) = o * k + k * os.sum by ring, pow_add, pow_succ]
    simp only [fou]
    ring

theorem prod_fou_in (d k : ℕ) (is : List ℕ) :
    (is.map fun i => (starRingEnd ℂ) (fou d i k)).prod =
      (starRingEnd ℂ) (om d) ^ (k * is.sum) / (Real.sqrt d : ℂ) ^ is.length := by
  induction is with
  | nil => simp
  | cons i is ih =>
    rw [List.map_cons, List.prod_cons, ih, List.sum_cons, List.length_cons]
    rw [show k * (i + is.sum) = i * k + k * is.sum by ring, pow_add, pow_succ]
    simp only [fou, map_div₀, map_pow, conj_sqrt]
    ring

/-- [P1] (HZ) pink K_j spider with outputs `os` and inputs `is`: `1` iff
`∑ os + j ≡ ∑ is (mod d)`. -/
def paperPink (d j : ℕ) (os is : List ℕ) : ℂ :=
  if (os.sum + j) % d = is.sum % d then 1 else 0

/-- The pink K_j spider is `d^((m+n)/2 - 1)` times the X spider with phases `j k / d` turns:
`√d^(m+n) · X = d · pink`. -/
theorem pink_eq (d j : ℕ) (hd : 0 < d) (os is : List ℕ) :
    (Real.sqrt d : ℂ) ^ (os.length + is.length) *
        tX d (fun k => ((j * k : ℕ) : ℝ) / d) os is =
      (d : ℂ) * paperPink d j os is := by
  have hs : (Real.sqrt d : ℂ) ≠ 0 := by
    have : (0 : ℝ) < Real.sqrt d := Real.sqrt_pos.mpr (by exact_mod_cast hd)
    exact_mod_cast this.ne'
  have hterm : ∀ k ∈ range d,
      ex (((j * k : ℕ) : ℝ) / d) * (os.map fun o => fou d o k).prod *
          (is.map fun i => (starRingEnd ℂ) (fou d i k)).prod =
        (om d ^ (k * (os.sum + j)) * (starRingEnd ℂ) (om d) ^ (k * is.sum)) /
          (Real.sqrt d : ℂ) ^ (os.length + is.length) := by
    intro k _
    rw [ex_of d (j * k) 0 hd (by push_cast; ring), prod_fou_out, prod_fou_in]
    rw [pow_add (Real.sqrt d : ℂ)]
    field_simp
    ring
  unfold tX
  rw [Finset.sum_congr rfl hterm, ← Finset.sum_div, char_sum d hd]
  unfold paperPink
  field_simp
  split_ifs <;> ring

/-- [P1] (Du) dualiser: `∑_i |i⟩⟨-i|`. -/
def paperD (d o i : ℕ) : ℂ := if (o + i) % d = 0 then 1 else 0

/-- D is F followed by F: `∑_k F[o][k] F[k][i] = D[o][i]`. -/
theorem paperD_eq (d o i : ℕ) (hd : 0 < d) :
    ∑ k ∈ range d, fou d o k * fou d k i = paperD d o i := by
  have hs : (Real.sqrt d : ℂ) * Real.sqrt d = d := sqrt_mul_sqrt d
  have hd' : (d : ℂ) ≠ 0 := by exact_mod_cast hd.ne'
  have hterm : ∀ k ∈ range d, fou d o k * fou d k i =
      om d ^ (k * (o + i)) * (starRingEnd ℂ) (om d) ^ (k * 0) / d := by
    intro k _
    simp only [fou, mul_zero, pow_zero, mul_one]
    rw [← hs, show k * (o + i) = o * k + k * i by ring, pow_add]
    ring
  rw [Finset.sum_congr rfl hterm, ← Finset.sum_div, char_sum d hd]
  unfold paperD
  simp only [Nat.zero_mod]
  by_cases h : (o + i) % d = 0
  · simp [h, hd']
  · simp [h]

end ArchytasZX
