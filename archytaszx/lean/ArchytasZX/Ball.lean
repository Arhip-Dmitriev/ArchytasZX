/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Cyclo

/-!
# Ball arithmetic in ℂ

`Near z x y r` says `z` lies within `r` of `x + y i`, distance the sum of the real and
imaginary distances. The lemmas propagate balls through sums, products, real constants,
square roots, `π` and `exp (a i)`; `ne_zero_of_near` turns a ball away from zero into `z ≠ 0`.
-/

namespace ArchytasZX

open Complex Finset

/-- `z` lies within `r` of `x + y i`. -/
def Near (z : ℂ) (x y r : ℝ) : Prop := |z.re - x| + |z.im - y| ≤ r

theorem near_round {z : ℂ} {x y r : ℝ} (h : Near z x y r) (x' y' r' : ℝ)
    (hr : |x - x'| + |y - y'| + r ≤ r') : Near z x' y' r' := by
  unfold Near at *
  have h1 := abs_sub_le z.re x x'
  have h2 := abs_sub_le z.im y y'
  linarith

theorem near_const (c : ℝ) : Near (c : ℂ) c 0 0 := by simp [Near]

theorem near_I : Near Complex.I 0 1 0 := by simp [Near]

theorem near_real {a m δ : ℝ} (h : |a - m| ≤ δ) : Near (a : ℂ) m 0 δ := by
  simpa [Near] using h

theorem near_add {a b : ℂ} {x y r u v s : ℝ} (ha : Near a x y r) (hb : Near b u v s) :
    Near (a + b) (x + u) (y + v) (r + s) := by
  unfold Near at *
  simp only [Complex.add_re, Complex.add_im]
  have h1 : |a.re + b.re - (x + u)| ≤ |a.re - x| + |b.re - u| := by
    rw [show a.re + b.re - (x + u) = (a.re - x) + (b.re - u) by ring]; exact abs_add_le _ _
  have h2 : |a.im + b.im - (y + v)| ≤ |a.im - y| + |b.im - v| := by
    rw [show a.im + b.im - (y + v) = (a.im - y) + (b.im - v) by ring]; exact abs_add_le _ _
  linarith

theorem near_neg {a : ℂ} {x y r : ℝ} (ha : Near a x y r) : Near (-a) (-x) (-y) r := by
  unfold Near at *
  simp only [Complex.neg_re, Complex.neg_im]
  rw [show -a.re - -x = -(a.re - x) by ring, show -a.im - -y = -(a.im - y) by ring, abs_neg,
    abs_neg]
  exact ha

theorem near_sub {a b : ℂ} {x y r u v s : ℝ} (ha : Near a x y r) (hb : Near b u v s) :
    Near (a - b) (x - u) (y - v) (r + s) := by
  have := near_add ha (near_neg hb)
  simpa [sub_eq_add_neg] using this

theorem abs_re_add_abs_im_mul (a b : ℂ) :
    |(a * b).re| + |(a * b).im| ≤ (|a.re| + |a.im|) * (|b.re| + |b.im|) := by
  simp only [Complex.mul_re, Complex.mul_im]
  have e1 := abs_sub (a.re * b.re) (a.im * b.im)
  have e2 := abs_add_le (a.re * b.im) (a.im * b.re)
  rw [abs_mul, abs_mul] at e1 e2
  nlinarith [abs_nonneg a.re, abs_nonneg a.im, abs_nonneg b.re, abs_nonneg b.im]

theorem near_mul {a b : ℂ} {x y r u v s : ℝ} (ha : Near a x y r) (hb : Near b u v s) :
    Near (a * b) (x * u - y * v) (x * v + y * u) ((|x| + |y|) * s + (|u| + |v|) * r + r * s) := by
  set c : ℂ := ⟨x, y⟩
  set d : ℂ := ⟨u, v⟩
  have hc : Near (a - c) 0 0 r := by unfold Near at *; simpa [c] using ha
  have hd : Near (b - d) 0 0 s := by unfold Near at *; simpa [d] using hb
  have key : a * b - c * d = c * (b - d) + d * (a - c) + (a - c) * (b - d) := by ring
  unfold Near
  have hre : (a * b).re - (x * u - y * v) = (a * b - c * d).re := by simp [c, d]
  have him : (a * b).im - (x * v + y * u) = (a * b - c * d).im := by simp [c, d]
  rw [hre, him, key]
  simp only [Near, sub_zero] at hc hd
  have t1 := abs_re_add_abs_im_mul c (b - d)
  have t2 := abs_re_add_abs_im_mul d (a - c)
  have t3 := abs_re_add_abs_im_mul (a - c) (b - d)
  simp only [Complex.add_re, Complex.add_im]
  have s1 := abs_add_le ((c * (b - d)).re + (d * (a - c)).re) ((a - c) * (b - d)).re
  have s2 := abs_add_le (c * (b - d)).re (d * (a - c)).re
  have s3 := abs_add_le ((c * (b - d)).im + (d * (a - c)).im) ((a - c) * (b - d)).im
  have s4 := abs_add_le (c * (b - d)).im (d * (a - c)).im
  have hcx : |c.re| + |c.im| = |x| + |y| := by simp [c]
  have hdx : |d.re| + |d.im| = |u| + |v| := by simp [d]
  rw [hcx] at t1
  rw [hdx] at t2
  have p1 : (|x| + |y|) * (|(b - d).re| + |(b - d).im|) ≤ (|x| + |y|) * s :=
    mul_le_mul_of_nonneg_left hd (by positivity)
  have p2 : (|u| + |v|) * (|(a - c).re| + |(a - c).im|) ≤ (|u| + |v|) * r :=
    mul_le_mul_of_nonneg_left hc (by positivity)
  have p3 : (|(a - c).re| + |(a - c).im|) * (|(b - d).re| + |(b - d).im|) ≤ r * s :=
    mul_le_mul hc hd (by positivity) (le_trans (by positivity) hc)
  linarith

theorem ne_zero_of_near {z : ℂ} {x y r : ℝ} (h : Near z x y r) (hx : r < |x| ∨ r < |y|) :
    z ≠ 0 := by
  intro hz
  subst hz
  unfold Near at h
  simp only [Complex.zero_re, Complex.zero_im, zero_sub, abs_neg] at h
  rcases hx with hx | hx <;> linarith [abs_nonneg x, abs_nonneg y]

/-- The even part of the exponential series. -/
noncomputable def cosT (A : ℝ) (n : ℕ) : ℝ :=
  ∑ k ∈ range n, (-1) ^ k * A ^ (2 * k) / (2 * k).factorial

/-- The odd part of the exponential series. -/
noncomputable def sinT (A : ℝ) (n : ℕ) : ℝ :=
  ∑ k ∈ range n, (-1) ^ k * A ^ (2 * k + 1) / (2 * k + 1).factorial

theorem partial_eq (A : ℝ) (n : ℕ) :
    ∑ m ∈ range (2 * n), ((A : ℂ) * I) ^ m / (m.factorial : ℂ) =
      ((cosT A n : ℝ) : ℂ) + ((sinT A n : ℝ) : ℂ) * I := by
  induction n with
  | zero => simp [cosT, sinT]
  | succ n ih =>
    have e : 2 * (n + 1) = 2 * n + 1 + 1 := by ring
    rw [e, sum_range_succ, sum_range_succ, ih]
    have p0 : ((A : ℂ) * I) ^ (2 * n) = ((-1) ^ n * A ^ (2 * n) : ℂ) := by
      rw [mul_pow, pow_mul I, I_sq]; ring
    have p1 : ((A : ℂ) * I) ^ (2 * n + 1) = ((-1) ^ n * A ^ (2 * n + 1) : ℂ) * I := by
      rw [pow_succ, p0]; ring
    rw [p0, p1]
    simp only [cosT, sinT, sum_range_succ]
    push_cast
    ring

theorem partial_re_im (A : ℝ) (n : ℕ) :
    (∑ m ∈ range (2 * n), ((A : ℂ) * I) ^ m / (m.factorial : ℂ)).re = cosT A n ∧
    (∑ m ∈ range (2 * n), ((A : ℂ) * I) ^ m / (m.factorial : ℂ)).im = sinT A n := by
  rw [partial_eq]; simp

/-- `exp (a i)` near the series at `A` when `a` is near `A` and `|A| ≤ 1`. -/
theorem near_exp (a A ε : ℝ) (n : ℕ) (hn : 0 < n) (ha : |a - A| ≤ ε) (hA : |A| ≤ 1) :
    Near (Complex.exp ((a : ℂ) * I)) (cosT A n) (sinT A n)
      (2 * (ε + |A| ^ (2 * n) * (((2 * n).succ : ℝ) * (((2 * n).factorial : ℝ) *
        ((2 * n : ℕ) : ℝ))⁻¹))) := by
  have hx : ‖(A : ℂ) * I‖ ≤ 1 := by simpa using hA
  have hb := Complex.exp_bound hx (n := 2 * n) (by omega)
  obtain ⟨hre, him⟩ := partial_re_im A n
  have hnorm : ‖(A : ℂ) * I‖ = |A| := by simp
  rw [hnorm] at hb
  have r1 := le_trans (abs_re_le_norm _) hb
  have r2 := le_trans (abs_im_le_norm _) hb
  rw [Complex.sub_re, hre, Complex.exp_ofReal_mul_I_re] at r1
  rw [Complex.sub_im, him, Complex.exp_ofReal_mul_I_im] at r2
  have l1 := Real.abs_cos_sub_cos_le a A
  have l2 := Real.abs_sin_sub_sin_le a A
  unfold Near
  rw [Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im]
  have q1 := abs_sub_le (Real.cos a) (Real.cos A) (cosT A n)
  have q2 := abs_sub_le (Real.sin a) (Real.sin A) (sinT A n)
  linarith

theorem ex_eq_exp (g : ℝ) : ex g = Complex.exp (((2 * Real.pi * g : ℝ) : ℂ) * I) := by
  unfold ex; congr 1; push_cast; ring

theorem ex_pow_eight (g : ℝ) : ex g = ex (g / 8) ^ 8 := by
  unfold ex; rw [← Complex.exp_nat_mul]; congr 1; push_cast; ring

theorem om_eq_ex (m : ℕ) (e : ℕ) : om m ^ e = ex ((e : ℝ) / m) := by
  unfold om ex; rw [← Complex.exp_nat_mul]; congr 1; push_cast; ring

theorem ex_inv (g : ℝ) : (ex g)⁻¹ = ex (-g) := by
  unfold ex; rw [← Complex.exp_neg]; congr 1; push_cast; ring

theorem ex_zpow (g : ℝ) (k : ℤ) : ex g ^ k = ex (k * g) := by
  rw [ex_zmul]

theorem sqrt_near (p : ℕ) (m δ : ℝ) (h0 : 0 ≤ m - δ) (h1 : (m - δ) ^ 2 ≤ p) (h2 : (p : ℝ) ≤ (m + δ) ^ 2)
    (hδ : 0 ≤ δ) : |Real.sqrt p - m| ≤ δ := by
  rw [abs_le]
  constructor
  · have : m - δ ≤ Real.sqrt p := Real.le_sqrt_of_sq_le h1
    linarith
  · have : Real.sqrt p ≤ m + δ := by
      rw [Real.sqrt_le_left (by linarith)]
      exact h2
    linarith

theorem inv_near (x m δ m' δ' : ℝ) (hx : |x - m| ≤ δ) (hpos : 0 < m - δ)
    (h1 : 1 / (m - δ) ≤ m' + δ') (h2 : m' - δ' ≤ 1 / (m + δ)) : |x⁻¹ - m'| ≤ δ' := by
  rw [abs_le] at hx ⊢
  have hxpos : 0 < x := by linarith
  have a1 : x⁻¹ ≤ 1 / (m - δ) := by
    rw [inv_eq_one_div]; exact one_div_le_one_div_of_le hpos (by linarith)
  have a2 : 1 / (m + δ) ≤ x⁻¹ := by
    rw [inv_eq_one_div]; exact one_div_le_one_div_of_le hxpos (by linarith)
  constructor <;> linarith

theorem pi_near : |Real.pi - 3.14159265358979323846| ≤ 1 / 10 ^ 20 := by
  rw [abs_le]
  constructor <;> linarith [Real.pi_gt_d20, Real.pi_lt_d20]

/-- `a` lies within `δ` of `m`. -/
def NearR (a m δ : ℝ) : Prop := |a - m| ≤ δ

theorem nearR_const (c : ℝ) : NearR c c 0 := by simp [NearR]

theorem nearR_round {a m δ : ℝ} (h : NearR a m δ) (m' δ' : ℝ) (hr : |m - m'| + δ ≤ δ') :
    NearR a m' δ' := by
  unfold NearR at *; have := abs_sub_le a m m'; linarith

theorem nearR_add {a b m n δ ε : ℝ} (ha : NearR a m δ) (hb : NearR b n ε) :
    NearR (a + b) (m + n) (δ + ε) := by
  unfold NearR at *
  rw [show a + b - (m + n) = (a - m) + (b - n) by ring]
  exact le_trans (abs_add_le _ _) (by linarith)

theorem nearR_neg {a m δ : ℝ} (ha : NearR a m δ) : NearR (-a) (-m) δ := by
  unfold NearR at *; rw [show -a - -m = -(a - m) by ring, abs_neg]; exact ha

theorem nearR_mul {a b m n δ ε : ℝ} (ha : NearR a m δ) (hb : NearR b n ε) :
    NearR (a * b) (m * n) (|m| * ε + |n| * δ + δ * ε) := by
  unfold NearR at *
  rw [show a * b - m * n = m * (b - n) + n * (a - m) + (a - m) * (b - n) by ring]
  have h1 := abs_add_le (m * (b - n) + n * (a - m)) ((a - m) * (b - n))
  have h2 := abs_add_le (m * (b - n)) (n * (a - m))
  rw [abs_mul] at h1
  rw [abs_mul, abs_mul] at h2
  have p1 : |m| * |b - n| ≤ |m| * ε := mul_le_mul_of_nonneg_left hb (abs_nonneg _)
  have p2 : |n| * |a - m| ≤ |n| * δ := mul_le_mul_of_nonneg_left ha (abs_nonneg _)
  have p3 : |a - m| * |b - n| ≤ δ * ε := mul_le_mul ha hb (abs_nonneg _) (le_trans (abs_nonneg _) ha)
  linarith

theorem nearR_pi : NearR Real.pi (314159265358979323846 / 10 ^ 20) (1 / 10 ^ 20) := by
  unfold NearR; rw [abs_le]; constructor <;> linarith [Real.pi_gt_d20, Real.pi_lt_d20]

theorem nearR_sqrt (p : ℕ) (m δ : ℝ) (h0 : 0 ≤ m - δ) (h1 : (m - δ) ^ 2 ≤ p)
    (h2 : (p : ℝ) ≤ (m + δ) ^ 2) (hδ : 0 ≤ δ) : NearR (Real.sqrt p) m δ :=
  sqrt_near p m δ h0 h1 h2 hδ

theorem nearR_inv {a m δ m' δ' : ℝ} (ha : NearR a m δ) (hpos : 0 < m - δ)
    (h1 : 1 / (m - δ) ≤ m' + δ') (h2 : m' - δ' ≤ 1 / (m + δ)) : NearR a⁻¹ m' δ' :=
  inv_near a m δ m' δ' ha hpos h1 h2

theorem near_ofReal {a m δ : ℝ} (h : NearR a m δ) : Near (a : ℂ) m 0 δ := near_real h

theorem near_rat (a b : ℤ) : Near ((a : ℂ) / b) ((a : ℝ) / b) 0 0 := by
  have : ((a : ℂ) / b) = (((a : ℝ) / b : ℝ) : ℂ) := by push_cast; ring
  rw [this]; exact near_const _

theorem near_int (a : ℤ) : Near (a : ℂ) (a : ℝ) 0 0 := by
  have : (a : ℂ) = (((a : ℝ)) : ℂ) := by push_cast; ring
  rw [this]; exact near_const _

theorem ex_npow (g : ℝ) (k : ℕ) : ex g ^ k = ex (k * g) := by
  have := ex_zmul g k; simpa using this.symm

theorem rt_inv_ofReal (p : ℕ) : (rt p)⁻¹ = (((Real.sqrt p)⁻¹ : ℝ) : ℂ) := by
  simp [rt]

theorem re_pos_of_near {z : ℂ} {x y r : ℝ} (h : Near z x y r) (hx : r < x) : 0 < z.re := by
  unfold Near at h
  have := abs_nonneg (z.im - y)
  have h2 : |z.re - x| ≤ r := by linarith
  rw [abs_le] at h2
  linarith

/-- A square root of `p` with positive real part is `√p`. -/
theorem sqrt_of_sq (g : ℂ) (p : ℕ) (h : g ^ 2 = p) (hre : 0 < g.re) : g = rt p := by
  have hs : (rt p) ^ 2 = p := rt_sq' p
  have hz : (g - rt p) * (g + rt p) = 0 := by
    have : (g - rt p) * (g + rt p) = g ^ 2 - (rt p) ^ 2 := by ring
    rw [this, h, hs, sub_self]
  rcases mul_eq_zero.1 hz with h1 | h1
  · exact sub_eq_zero.1 h1
  · exfalso
    have hg : g = -rt p := eq_neg_of_add_eq_zero_left h1
    rw [hg] at hre
    simp only [rt, Complex.neg_re, Complex.ofReal_re] at hre
    have := Real.sqrt_nonneg (p : ℝ)
    linarith

end ArchytasZX
