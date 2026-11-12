/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Sym
import ArchytasZX.Cyclo

/-!
# Path-sum rules

One lemma per rule of `archytaszx.algebra.pathsum.reduce` and `reduce_qudit`, each the
identity the rule applies to the sum over its eliminated variables, at Boolean values `0`, `1`
of the variables its quotient polynomials read (`ps_*`) or at a residue mod `p` (`q_*`).
`(-1) ^ q` is `exp(2 pi i q / 2)`; `om 8 ^ e` is `exp(2 pi i e / 8)`.
-/

namespace ArchytasZX

open Complex Finset

/-- [Elim]: a sum over a variable no term reads. -/
theorem ps_elim (c : ℂ) : c + c = 2 * c := by ring

/-- [HH] on a constraint: `Sum_y (-1)^(y q) = 2 [q even]`. -/
theorem ps_hh (q : ℕ) : (1 : ℂ) + (-1) ^ q = if q % 2 = 0 then 2 else 0 := by
  rcases Nat.even_or_odd' q with ⟨k, rfl | rfl⟩
  · rw [if_pos (by omega), pow_mul]; norm_num
  · rw [if_neg (by omega), pow_succ, pow_mul]; norm_num

/-- [HH] on a path pivot: `Sum_z Sum_y (-1)^(y (z + q)) g z = 2 g q`, q a bit. -/
theorem ps_hh_pin (g : ℕ → ℂ) (q : ℕ) (hq : q ≤ 1) :
    ((1 : ℂ) + (-1) ^ (0 + q)) * g 0 + ((1 : ℂ) + (-1) ^ (1 + q)) * g 1 = 2 * g q := by
  interval_cases q <;> norm_num <;> ring

/-- [omega] at Q = 0: `1 + i = sqrt 2 zeta_8`. -/
theorem ps_omega0 : (1 : ℂ) + om 8 ^ 2 = rt 2 * om 8 := by
  rw [rt_sub2 8 1 (by norm_num) (by norm_num)]
  have h8 := om_pow_self 8 (by norm_num)
  linear_combination (-1 : ℂ) * h8

/-- [omega] at Q = 1: `1 + i (-1) = sqrt 2 zeta_8 zeta_8^(-2)`, `zeta_8^(-2) = zeta_8^6`. -/
theorem ps_omega1 : (1 : ℂ) + om 8 ^ 6 = rt 2 * om 8 * om 8 ^ 6 := by
  rw [rt_sub2 8 1 (by norm_num) (by norm_num)]
  have h8 := om_pow_self 8 (by norm_num)
  linear_combination (-(1 + om 8 ^ 6) : ℂ) * h8

/-- [omega] conjugate at Q = 0: `1 - i = sqrt 2 zeta_8^(-1)`. -/
theorem ps_omega_conj0 : (1 : ℂ) + om 8 ^ 6 = rt 2 * om 8 ^ 7 := by
  rw [rt_sub2 8 1 (by norm_num) (by norm_num)]
  have h8 := om_pow_self 8 (by norm_num)
  linear_combination (-(1 + om 8 ^ 6) : ℂ) * h8

/-- [omega] conjugate at Q = 1: `1 + i = sqrt 2 zeta_8^(-1) zeta_8^2`. -/
theorem ps_omega_conj1 : (1 : ℂ) + om 8 ^ 2 = rt 2 * om 8 ^ 7 * om 8 ^ 2 := by
  rw [rt_sub2 8 1 (by norm_num) (by norm_num)]
  have h8 := om_pow_self 8 (by norm_num)
  linear_combination (-(om 8 ^ 8 + 1 + om 8 ^ 2) : ℂ) * h8

/-- [HHgen]: with bits a, b not both 0, `Sum_z (1 + (-1)^(z a + b)) g z = (1 + (-1)^(b (1 +
a))) g b`. -/
theorem ps_hh_gen (g : ℕ → ℂ) (a b : ℕ) (ha : a ≤ 1) (hb : b ≤ 1) (hab : 1 ≤ a + b) :
    ((1 : ℂ) + (-1) ^ (0 * a + b)) * g 0 + ((1 : ℂ) + (-1) ^ (1 * a + b)) * g 1
      = ((1 : ℂ) + (-1) ^ (b * (1 + a))) * g b := by
  interval_cases a <;> interval_cases b
  all_goals first | (exfalso; omega) | (norm_num; done) | (norm_num; ring)

/-- [Rem]: a factor `c^m` beside `1 + (-1)^m`, m a bit, drops. -/
theorem ps_rem (m : ℕ) (hm : m ≤ 1) (c g : ℂ) :
    ((1 : ℂ) + (-1) ^ m) * (c ^ m * g) = ((1 : ℂ) + (-1) ^ m) * g := by
  interval_cases m <;> norm_num

/-- [HHnl]: `Sum_y Sum_y' (-1)^(y q + y' q') = 2 Sum_y (-1)^(y (q or q'))`, q and q' bits. -/
theorem ps_hh_nl (q q' : ℕ) (hq : q ≤ 1) (hq' : q' ≤ 1) :
    ((1 : ℂ) + (-1) ^ q) * ((1 : ℂ) + (-1) ^ q') = 2 * ((1 : ℂ) + (-1) ^ (q + q' + q * q')) := by
  interval_cases q <;> interval_cases q' <;> norm_num

/-- `Sum_{Y < N} zeta_N^(Y S) = N [S = 0 mod N]`. -/
theorem ps_geom (N S : ℕ) (hN : 0 < N) :
    ∑ Y ∈ range N, om N ^ (Y * S) = if S % N = 0 then (N : ℂ) else 0 := by
  have hz : (om N ^ S) ^ N = 1 := by
    rw [← pow_mul, mul_comm, pow_mul, om_pow_self N hN, one_pow]
  simp_rw [mul_comm _ S, pow_mul]
  split_ifs with h
  · have h1 : om N ^ S = 1 := by rw [om_pow_mod N hN, h, pow_zero]
    simp [h1]
  · have hne : om N ^ S ≠ 1 := by
      intro h1
      apply h
      have := (om_pow_eq_iff N hN S 0).1 (by simpa using h1)
      simpa using this
    rw [geom_sum_eq hne, hz, sub_self, zero_div]

/-- `Prod_{j < k} (1 + (-1)^(digit j of S)) = 2^k [S = 0 mod 2^k]`. -/
theorem ps_digits (S : ℕ) : ∀ k : ℕ,
    ∏ j ∈ range k, ((1 : ℂ) + (-1) ^ (S / 2 ^ j % 2)) =
      if S % 2 ^ k = 0 then (2 : ℂ) ^ k else 0
  | 0 => by simp [Nat.mod_one]
  | k + 1 => by
    rw [prod_range_succ, ps_digits S k, ps_hh]
    have hsplit : S % 2 ^ (k + 1) = S % 2 ^ k + 2 ^ k * (S / 2 ^ k % 2) := by
      rw [pow_succ, Nat.mod_mul]
    have hp : 0 < 2 ^ k := pow_pos (by norm_num) k
    rw [hsplit]
    rcases Nat.mod_two_eq_zero_or_one (S / 2 ^ k) with hb | hb <;>
      by_cases hs : S % 2 ^ k = 0 <;> simp [hb, hs, pow_succ] <;> omega

/-- [Dyadic]: `Sum_{Y < 2^k} zeta_(2^k)^(Y S) = Prod_{j < k} Sum_{y'_j} (-1)^(y'_j b_j)`, b_j
the j-th binary digit of S. -/
theorem ps_dyadic (k S : ℕ) :
    ∑ Y ∈ range (2 ^ k), om (2 ^ k) ^ (Y * S) =
      ∏ j ∈ range k, ((1 : ℂ) + (-1) ^ (S / 2 ^ j % 2)) := by
  rw [ps_geom _ _ (pow_pos (by norm_num) k), ps_digits]
  simp only [Nat.cast_pow, Nat.cast_ofNat]

/-- [Elim] over Z_p: a sum of p equal terms. -/
theorem q_elim (p : ℕ) (c : ℂ) : ∑ _y ∈ range p, c = p * c := by
  simp [sum_const, card_range]

/-- [HH] over Z_p: `Sum_{y < p} zeta_p^(y q) = p [q = 0 mod p]`. -/
theorem q_hh (p q : ℕ) (hp : 0 < p) :
    ∑ y ∈ range p, om p ^ (y * q) = if q % p = 0 then (p : ℂ) else 0 :=
  ps_geom p q hp

/-- A sum over `range p` of a p-periodic function is invariant under a shift. -/
theorem q_shift (p c : ℕ) (hp : 0 < p) (f : ℕ → ℂ) (hf : ∀ n, f (n + p) = f n) :
    ∑ y ∈ range p, f (y + c) = ∑ y ∈ range p, f y := by
  induction c with
  | zero => simp
  | succ c ih =>
    have h := Finset.sum_range_succ (fun y => f (y + c)) p
    have h0 := Finset.sum_range_succ' (fun y => f (y + c)) p
    beta_reduce at h h0
    rw [show p + c = c + p by ring, hf c] at h
    simp only [zero_add] at h0
    have : ∑ y ∈ range p, f (y + (c + 1)) = ∑ y ∈ range p, f (y + 1 + c) := by
      refine sum_congr rfl fun y _ => ?_
      rw [show y + (c + 1) = y + 1 + c by ring]
    rw [this, ← ih]
    have e := h.symm.trans h0
    linear_combination (-1 : ℂ) * e

/-- [Gauss] over Z_p: with `2 a c = q` and `e + a c^2 = 0` mod p, `Sum_y zeta_p^(a y^2 + q y) =
zeta_p^e Sum_y zeta_p^(a y^2)`. -/
theorem q_gauss (p a q c e : ℕ) (hp : 0 < p) (hc : (2 * a * c) % p = q % p)
    (he : (e + a * c * c) % p = 0) :
    ∑ y ∈ range p, om p ^ (a * y * y + q * y) = om p ^ e * ∑ y ∈ range p, om p ^ (a * y * y) := by
  have hstep : ∀ y, om p ^ (a * y * y + q * y) = om p ^ e * om p ^ (a * (y + c) * (y + c)) := by
    intro y
    rw [← pow_add, om_pow_eq_iff p hp, ← ZMod.natCast_eq_natCast_iff']
    have hc' : ((2 * a * c : ℕ) : ZMod p) = (q : ZMod p) :=
      (ZMod.natCast_eq_natCast_iff' _ _ _).2 hc
    have he' : ((e + a * c * c : ℕ) : ZMod p) = 0 := by
      have := (ZMod.natCast_eq_natCast_iff' (e + a * c * c) 0 p).2 (by simpa using he)
      simpa using this
    push_cast at hc' he' ⊢
    linear_combination (-1 : ZMod p) * he' - (y : ZMod p) * hc'
  simp_rw [hstep]
  rw [← mul_sum]
  congr 1
  apply q_shift p c hp (fun n => om p ^ (a * n * n))
  intro n
  rw [om_pow_eq_iff p hp]
  have : a * (n + p) * (n + p) = a * n * n + p * (a * (2 * n + p)) := by ring
  rw [this, Nat.add_mul_mod_self_left]

/-- Canonical-form data of a reduced qubit path sum: rows (pivot, monomials), phase terms in
turns, constant coefficients over `zeta_N^0 .. zeta_N^(phi N - 1)`, and N. -/
structure PSCanon where
  rows : List (ℕ × List (List ℕ))
  phase : List (List ℕ × ℚ)
  const : List ℚ
  n : ℕ

/-- Canonical-form data of a reduced qudit path sum over Z_p: rows (pivot, terms), phase
terms (monomial as (variable, exponent) pairs, coefficient mod p), constant, N and p. -/
structure QCanon where
  rows : List (ℕ × List (List (ℕ × ℕ) × ℕ))
  phase : List (List (ℕ × ℕ) × ℕ)
  const : List ℚ
  n : ℕ
  p : ℕ

end ArchytasZX
