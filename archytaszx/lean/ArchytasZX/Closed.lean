/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Rules

/-!
# Closed forms

The character sum, the phaseless X spider as a scaled sum-modulo-`d` delta, and the
Fourier box squared.
-/

namespace ArchytasZX

open Complex Finset

/-- `√d` as a complex number. -/
noncomputable abbrev rt (d : ℕ) : ℂ := (Real.sqrt d : ℂ)

theorem rt_ne (d : ℕ) (hd : 0 < d) : rt d ≠ 0 := by
  have : (0 : ℝ) < Real.sqrt d := Real.sqrt_pos.2 (by exact_mod_cast hd)
  show (Real.sqrt d : ℂ) ≠ 0
  exact_mod_cast this.ne'

theorem rt_cancel (d n : ℕ) (hd : 0 < d) :
    rt d * (rt d)⁻¹ ^ n * rt d ^ n = rt d := by
  rw [mul_assoc, ← mul_pow, inv_mul_cancel₀ (rt_ne d hd), one_pow, mul_one]

theorem nat_inv_mul (d : ℕ) (hd : 0 < d) : ((d : ℂ))⁻¹ * d = 1 :=
  inv_mul_cancel₀ (by exact_mod_cast hd.ne')

theorem rt_sq (d : ℕ) : rt d * rt d = d := sqrt_mul_sqrt d

theorem rt_sq' (d : ℕ) : rt d ^ 2 = d := by rw [sq, rt_sq]

theorem om_ne_zero (d : ℕ) (hd : 0 < d) : om d ≠ 0 := (om_prim d hd).ne_zero hd.ne'

theorem om_pow_self (d : ℕ) (hd : 0 < d) : om d ^ d = 1 := (om_prim d hd).pow_eq_one

theorem om_geom (m p : ℕ) (hm : 0 < m) (hp : 1 < p) (hpm : p ∣ m) :
    ∑ j ∈ range p, om m ^ (j * (m / p)) = 0 := by
  have hζ : IsPrimitiveRoot (om m ^ (m / p)) p := by
    have := (om_prim m hm).pow_of_dvd (Nat.div_pos (Nat.le_of_dvd hm hpm) (by omega)).ne'
      (Nat.div_dvd_of_dvd hpm)
    rwa [Nat.div_div_self hpm hm.ne'] at this
  have := hζ.geom_sum_eq_zero hp
  simpa [← pow_mul, mul_comm] using this

theorem om_pow_mod (d : ℕ) (hd : 0 < d) (n : ℕ) : om d ^ n = om d ^ (n % d) := by
  conv_lhs => rw [← Nat.div_add_mod n d]
  rw [pow_add, pow_mul, (om_prim d hd).pow_eq_one, one_pow, one_mul]

theorem om_pow_eq_iff (d : ℕ) (hd : 0 < d) (a b : ℕ) :
    om d ^ a = om d ^ b ↔ a % d = b % d := by
  rw [om_pow_mod d hd a, om_pow_mod d hd b]
  exact ⟨fun h => (om_prim d hd).pow_inj (Nat.mod_lt _ hd) (Nat.mod_lt _ hd) h,
    fun h => by rw [h]⟩

/-- The character sum. -/
theorem char_sum (d : ℕ) (hd : 0 < d) (a b : ℕ) :
    ∑ k ∈ range d, om d ^ (k * a) * (starRingEnd ℂ) (om d) ^ (k * b) =
      if a % d = b % d then (d : ℂ) else 0 := by
  have hp := om_prim d hd
  have hz : om d ≠ 0 := hp.ne_zero hd.ne'
  set u := om d ^ a * (om d ^ b)⁻¹
  have hterm : ∀ k, om d ^ (k * a) * (starRingEnd ℂ) (om d) ^ (k * b) = u ^ k := by
    intro k
    simp only [conj_om, u, mul_pow, inv_pow, ← pow_mul, mul_comm a k, mul_comm b k]
  simp only [hterm]
  split_ifs with h
  · have : u = 1 := by
      simp only [u]
      rw [(om_pow_eq_iff d hd a b).2 h, mul_inv_cancel₀ (pow_ne_zero _ hz)]
    simp [this]
  · have hu : u ≠ 1 := by
      intro h1
      exact h ((om_pow_eq_iff d hd a b).1 ((mul_inv_eq_one₀ (pow_ne_zero _ hz)).1 h1))
    have hud : u ^ d = 1 := by
      simp only [u, mul_pow, inv_pow, ← pow_mul]
      rw [mul_comm a d, mul_comm b d, pow_mul, pow_mul, hp.pow_eq_one]; simp
    rw [geom_sum_eq hu, hud]; simp

theorem prod_fou (d k : ℕ) (os : List ℕ) :
    (os.map fun o => fou d o k).prod = om d ^ (k * os.sum) * (rt d)⁻¹ ^ os.length := by
  induction os with
  | nil => simp
  | cons o os ih =>
    rw [List.map_cons, List.prod_cons, ih]
    simp only [List.sum_cons, List.length_cons, fou, rt]
    rw [mul_add, pow_add, pow_succ, mul_comm o k]
    ring

theorem prod_conj_fou (d k : ℕ) (is : List ℕ) :
    (is.map fun i => (starRingEnd ℂ) (fou d i k)).prod =
      (starRingEnd ℂ) (om d) ^ (k * is.sum) * (rt d)⁻¹ ^ is.length := by
  induction is with
  | nil => simp
  | cons i is ih =>
    rw [List.map_cons, List.prod_cons, ih]
    simp only [List.sum_cons, List.length_cons, fou, rt, map_div₀, map_pow, conj_sqrt]
    rw [mul_add, pow_add, pow_succ, mul_comm i k]
    ring

/-- A phaseless X spider is `√d^{-(n+m)} d [Σ outs ≡ Σ ins]`. -/
theorem tX_flat (d : ℕ) (hd : 0 < d) (φ : ℕ → ℝ) (hφ : ∀ k < d, ex (φ k) = 1)
    (os is : List ℕ) :
    tX d φ os is = (rt d)⁻¹ ^ (os.length + is.length) *
      (if os.sum % d = is.sum % d then (d : ℂ) else 0) := by
  unfold tX
  rw [← char_sum d hd, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [hφ k (Finset.mem_range.1 hk), prod_fou, prod_conj_fou, pow_add]
  ring

/-- A phaseless Z spider on one leg value. -/
theorem tZ_one (d : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k < d, ex (φ k) = 1) (x : ℕ) :
    tZ d φ [x] = if x < d then 1 else 0 := by
  rw [tZ_at d φ [x] x (by simp)]
  by_cases h : x < d <;> simp [h, hφ]

/-- The Fourier box squared is the antipode. -/
theorem fou_sq (d : ℕ) (hd : 0 < d) (x y : ℕ) :
    ∑ w ∈ range d, fou d x w * fou d w y = if (x + y) % d = 0 then 1 else 0 := by
  have h := char_sum d hd (x + y) 0
  simp only [Nat.zero_mod] at h
  have hrt := rt_sq d
  have hdc : (d : ℂ) ≠ 0 := by exact_mod_cast hd.ne'
  have : ∀ w ∈ range d, fou d x w * fou d w y =
      (om d ^ (w * (x + y)) * (starRingEnd ℂ) (om d) ^ (w * 0)) / d := by
    intro w _
    simp only [fou, mul_zero, pow_zero, mul_one]
    rw [← hrt]
    have := rt_ne d hd
    field_simp
    rw [← pow_add]; ring_nf
  rw [Finset.sum_congr rfl this, ← Finset.sum_div, h]
  by_cases hc : (x + y) % d = 0 <;> simp [hc, hdc]

/-- A value below `d` determined by its residue. -/
theorem sum_mod_eq (d : ℕ) (s : ℕ) (f : ℕ → ℂ) :
    ∑ w ∈ range d, (if s % d = w % d then f w else 0) =
      if 0 < d then f (s % d) else 0 := by
  rcases Nat.eq_zero_or_pos d with rfl | hd
  · simp
  have : ∀ w ∈ range d, (if s % d = w % d then f w else 0) = if s % d = w then f w else 0 := by
    intro w hw
    rw [Nat.mod_eq_of_lt (Finset.mem_range.1 hw)]
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq]
  simp [Nat.mod_lt s hd, hd]

end ArchytasZX
