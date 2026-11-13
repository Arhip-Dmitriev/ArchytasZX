/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Cyclo
import ArchytasZX.PathSumQudit

/-!
# The sign of the quadratic Gauss sum

For odd `p`, `G = Σ_{k<p} (k/p) ζ_p^k` with `G = Π_{j<h} (ζ^(2j+1) - ζ^(-(2j+1)))` and
`G^2 = (-1)^h p` (`h = (p-1)/2`) is `I^h (-1)^g √p`, `g` the number of `j < h` with
`p < 2 (2j + 1)` (`gauss_sign`): each factor is `2 I sin(2π(2j+1)/p)`, whose sign is that of
`p - 2(2j+1)`. `gauss_rt` takes the two hypotheses as coefficient-vector reductions by
`1 + x + ... + x^(p-1)` checked by `decide`.
-/

namespace ArchytasZX

open Complex Finset

/-- The number of `j < (p-1)/2` with `p < 2 (2j + 1)`. -/
def gneg (p : ℕ) : ℕ := ((range ((p - 1) / 2)).filter (fun j => p < 2 * (2 * j + 1))).card

/-- `x^a - x^(p-a)` as a coefficient vector of length `p + 1`. -/
def gfac (p a : ℕ) : List ℤ := addAt (addAt (List.replicate (p + 1) 0) a 1) (p - a) (-1)

/-- `Π_{j<n} (x^(2j+1) - x^(p-(2j+1)))` as a coefficient vector. -/
def gprod (p : ℕ) : ℕ → List ℤ
  | 0 => [1]
  | n + 1 => pmul (gprod p n) (gfac p (2 * n + 1))

/-- The Legendre symbols `(k/p)`, `k < p`. -/
def lgVec (p : ℕ) : List ℤ := (List.range p).map (QSem.lg p)

theorem cvz_map_range (z : ℂ) (f : ℕ → ℤ) (n : ℕ) :
    cvz z ((List.range n).map f) = ∑ k ∈ range n, (f k : ℂ) * z ^ k := by
  induction n generalizing f with
  | zero => simp [cvz]
  | succ n ih =>
    rw [List.range_succ_eq_map, List.map_cons, List.map_map, cvz, ih, sum_range_succ',
      Finset.mul_sum]
    simp only [Function.comp, pow_zero, mul_one, pow_succ]
    rw [add_comm]
    congr 1
    refine sum_congr rfl fun k _ => ?_
    ring

theorem cvz_gfac (p a : ℕ) (ha : a ≤ p) (z : ℂ) : cvz z (gfac p a) = z ^ a - z ^ (p - a) := by
  unfold gfac
  rw [cvz_addAt _ _ _ _ (by rw [addAt_length, List.length_replicate]; omega),
    cvz_addAt _ _ _ _ (by rw [List.length_replicate]; omega), cvz_replicate]
  push_cast; ring

theorem cvz_gprod (p n : ℕ) (hn : 2 * n ≤ p + 1) (z : ℂ) :
    cvz z (gprod p n) = ∏ j ∈ range n, (z ^ (2 * j + 1) - z ^ (p - (2 * j + 1))) := by
  induction n with
  | zero => simp [gprod, cvz]
  | succ n ih =>
    rw [gprod, cvz_pmul, ih (by omega), cvz_gfac p _ (by omega), prod_range_succ]

theorem om_factor (p a : ℕ) (hp : 0 < p) (ha : a ≤ p) :
    om p ^ a - om p ^ (p - a) = 2 * I * (Real.sin (2 * Real.pi * a / p) : ℂ) := by
  have h1 : om p ^ a = exp (((2 * Real.pi * a / p : ℝ) : ℂ) * I) := by
    unfold om; rw [← Complex.exp_nat_mul]; congr 1; push_cast; ring
  have h2 : om p ^ (p - a) = exp ((-(2 * Real.pi * a / p : ℝ) : ℂ) * I) := by
    unfold om; rw [← Complex.exp_nat_mul]
    have hpc : (p : ℂ) ≠ 0 := by exact_mod_cast hp.ne'
    rw [show ((p - a : ℕ) : ℂ) * (2 * Real.pi * I / p) =
        (-(2 * Real.pi * a / p : ℝ) : ℂ) * I + 2 * Real.pi * I by
      push_cast [Nat.cast_sub ha]; field_simp; ring]
    rw [Complex.exp_add, Complex.exp_two_pi_mul_I, mul_one]
  rw [h1, h2, Complex.exp_mul_I, Complex.exp_mul_I, ← ofReal_neg, ← ofReal_cos, ← ofReal_sin,
    ← ofReal_cos, ← ofReal_sin, Real.cos_neg, Real.sin_neg]
  push_cast; ring

theorem sin_sign (p a : ℕ) (hp : 0 < p) (ha : 0 < a) (hap : a < p) (hne : 2 * a ≠ p) :
    Real.sin (2 * Real.pi * a / p) =
      (if p < 2 * a then -1 else 1) * |Real.sin (2 * Real.pi * a / p)| := by
  have hpr : (0 : ℝ) < p := by exact_mod_cast hp
  have hπ := Real.pi_pos
  have hq : (0 : ℝ) < 2 * Real.pi * a / p := by positivity
  split_ifs with h
  · have h1 : Real.pi < 2 * Real.pi * a / p := by
      rw [lt_div_iff₀ hpr]
      have : (p : ℝ) < 2 * a := by exact_mod_cast h
      nlinarith
    have h2 : 2 * Real.pi * a / p < 2 * Real.pi := by
      rw [div_lt_iff₀ hpr]
      have : (a : ℝ) < p := by exact_mod_cast hap
      nlinarith
    have hpos : 0 < Real.sin (2 * Real.pi * a / p - Real.pi) :=
      Real.sin_pos_of_pos_of_lt_pi (by linarith) (by linarith)
    rw [Real.sin_sub_pi] at hpos
    rw [abs_of_neg (by linarith)]
    ring
  · have h1 : 2 * Real.pi * a / p < Real.pi := by
      rw [div_lt_iff₀ hpr]
      have : (2 * a : ℝ) < p := by exact_mod_cast (by omega : 2 * a < p)
      nlinarith
    rw [abs_of_pos (Real.sin_pos_of_pos_of_lt_pi hq h1)]
    ring

theorem prod_sign {ι : Type*} (s : Finset ι) (c : ι → Prop) [DecidablePred c] :
    ∏ j ∈ s, (if c j then (-1 : ℝ) else 1) = (-1) ^ (s.filter c).card := by
  rw [prod_ite, prod_const, prod_const_one, mul_one]

/-- `G = I^h (-1)^g √p` from the product form and the square of `G`. -/
theorem gauss_sign (p : ℕ) (hp : 0 < p) (hodd : p % 2 = 1) (G : ℂ)
    (hG : G = ∏ j ∈ range ((p - 1) / 2), (om p ^ (2 * j + 1) - om p ^ (p - (2 * j + 1))))
    (hsq : G ^ 2 = (-1) ^ ((p - 1) / 2) * p) :
    G = I ^ ((p - 1) / 2) * (-1) ^ gneg p * rt p := by
  set h := (p - 1) / 2 with hh
  set θ : ℕ → ℝ := fun j => 2 * Real.pi * ((2 * j + 1 : ℕ) : ℝ) / p with hθ
  have hfac : ∀ j ∈ range h, om p ^ (2 * j + 1) - om p ^ (p - (2 * j + 1)) =
      2 * I * (Real.sin (θ j) : ℂ) := fun j hj => by
    rw [mem_range] at hj
    exact om_factor p _ hp (by omega)
  set s : ℝ := ∏ j ∈ range h, Real.sin (θ j) with hs
  have hG2 : G = (2 * I) ^ h * (s : ℂ) := by
    rw [hG, prod_congr rfl hfac, prod_mul_distrib, prod_const, card_range, hs, ofReal_prod]
  have hsgn : s = (-1) ^ gneg p * |s| := by
    have : ∀ j ∈ range h, Real.sin (θ j) =
        (if p < 2 * (2 * j + 1) then -1 else 1) * |Real.sin (θ j)| := fun j hj => by
      rw [mem_range] at hj
      exact sin_sign p _ hp (by omega) (by omega) (by omega)
    conv_lhs => rw [hs, prod_congr rfl this, prod_mul_distrib, prod_sign, ← abs_prod]
    rfl
  have hI : ((2 * I) ^ h) ^ 2 = (-1) ^ h * (4 : ℂ) ^ h := by
    rw [← pow_mul, mul_comm h 2, pow_mul,
      show (2 * I) ^ 2 = (-1) * 4 by rw [mul_pow, I_sq]; norm_num, mul_pow]
  have h4 : (4 : ℝ) ^ h * s ^ 2 = p := by
    have e : ((2 * I) ^ h * (s : ℂ)) ^ 2 = (-1) ^ h * ((4 : ℂ) ^ h * (s : ℂ) ^ 2) := by
      rw [mul_pow, hI]
      ring
    rw [← hG2, hsq] at e
    have hne : ((-1 : ℂ)) ^ h ≠ 0 := pow_ne_zero _ (by norm_num)
    have e2 : ((p : ℂ)) = (4 : ℂ) ^ h * (s : ℂ) ^ 2 := mul_left_cancel₀ hne e
    exact_mod_cast e2.symm
  have hrt : (2 : ℝ) ^ h * |s| = Real.sqrt p := by
    rw [← h4, show (4 : ℝ) ^ h * s ^ 2 = ((2 : ℝ) ^ h * |s|) ^ 2 by
      rw [mul_pow, ← pow_mul, mul_comm h 2, pow_mul, sq_abs]; norm_num,
      Real.sqrt_sq (by positivity)]
  rw [hG2, hsgn, rt]
  rw [← hrt]
  push_cast
  rw [mul_pow]
  ring

/-- `Σ_{k<p} (k/p) ζ_p^k = I^h (-1)^g √p` from two reductions by `1 + x + ... + x^(p-1)`. -/
theorem gauss_rt (p N : ℕ) (hp : 1 < p) (hodd : p % 2 = 1)
    (hprod : (pmod (List.replicate p 1) N
      (kadd (lgVec p) (pscale (-1) (gprod p ((p - 1) / 2))))).all (· == 0) = true)
    (hsq : (pmod (List.replicate p 1) N
      (kadd (pmul (lgVec p) (lgVec p)) [-((-1) ^ ((p - 1) / 2) * p)])).all (· == 0) = true) :
    ∑ k ∈ range p, (QSem.lg p k : ℂ) * om p ^ k = I ^ ((p - 1) / 2) * (-1) ^ gneg p * rt p := by
  have hphi : cvz (om p) (List.replicate p 1) = 0 := by
    rw [show List.replicate p (1 : ℤ) = (List.range p).map (fun _ => 1) by simp,
      cvz_map_range]
    have := om_geom p p (by omega) hp dvd_rfl
    simpa [Nat.div_self (by omega : 0 < p)] using this
  have e1 := cvz_eq_zero_of_pmod _ _ _ hphi N hprod
  have e2 := cvz_eq_zero_of_pmod _ _ _ hphi N hsq
  rw [cvz_kadd, cvz_pscale, cvz_gprod p _ (by omega)] at e1
  rw [cvz_kadd, cvz_pmul] at e2
  simp only [cvz] at e2
  have hl : cvz (om p) (lgVec p) = ∑ k ∈ range p, (QSem.lg p k : ℂ) * om p ^ k :=
    cvz_map_range _ _ _
  rw [← hl]
  apply gauss_sign p (by omega) hodd
  · push_cast at e1; linear_combination e1
  · push_cast at e2; linear_combination e2

end ArchytasZX
