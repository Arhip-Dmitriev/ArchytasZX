/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Closed

/-!
# Adder families: carry-chain telescoping and the Fourier shift

Lemmas of the every-`n` proofs of `draper_adder` and `fourier_constant_adder` in the boxed form
`C⁻¹ P C`, copies indexed MSB first (copy `k` reads root order `d ^ (k + 1)`), boundary
interleaved copy-major. A carry chain is `x k + e (k + 1) = x' k + d * e k` for `k < n`: copy `k`
takes carry-in `e (k + 1)` and gives carry-out `e k`.

* `om_tower`, `om_tower_mul`: `om (d ^ (m + 1)) ^ d = om (d ^ m)`.
* `digit_carry`, `carry_lt_two`, `chain_digit`: the digit-carry identity of one copy.
* `telescope`: the copy phases of a carry chain collapse to the two end terms.
* `chain_value`: the chain adds its values, `digVal x + e n = e 0 * d ^ n + digVal x'`.
* `fourier_shift`: `F³ · diag (om d ^ (y s)) · F` is the shift by `s`.
* `AdderIface`, `AdderIface.fam`: the family theorem by `Nat.rec` against an interface the engine
  export fills; `fam_8`, `fam_24` instantiate it.
-/

namespace ArchytasZX

open Complex Finset

/-- The root-of-unity tower step. -/
theorem om_tower (d m : ℕ) (hd : 0 < d) : om (d ^ (m + 1)) ^ d = om (d ^ m) :=
  by
  unfold om
  rw [← Complex.exp_nat_mul]
  congr 1
  have h1 : ((d : ℂ)) ≠ 0 := by exact_mod_cast hd.ne'
  push_cast
  rw [pow_succ]
  field_simp

theorem om_tower_mul (d k y c : ℕ) (hd : 0 < d) :
    om (d ^ (k + 1)) ^ (y * (d * c)) = om (d ^ k) ^ (y * c) := by
  rw [show y * (d * c) = d * (y * c) by ring, pow_mul, om_tower d k hd]

/-- The digit-carry identity. -/
theorem digit_carry (d x c : ℕ) : x + c = (x + c) % d + d * ((x + c) / d) :=
  (Nat.mod_add_div _ _).symm

theorem carry_lt_two (d x c : ℕ) (hx : x < d) (hc : c ≤ d) : (x + c) / d < 2 := by
  rw [Nat.div_lt_iff_lt_mul (by omega)]; omega

/-- Digits and carries from `x` and a carry-in `e n`, `e k = (x k + e (k + 1)) / d`, form a
carry chain. -/
theorem chain_digit (d n : ℕ) (x e : ℕ → ℕ) (he : ∀ k < n, e k = (x k + e (k + 1)) / d) :
    ∀ k < n, x k + e (k + 1) = (x k + e (k + 1)) % d + d * e k := fun k hk => by
  rw [he k hk]; exact digit_carry d _ _

/-- The copy phases of a carry chain: `∏ om (d^(k+1))^(y x k) · om (d^n)^(y e n)` equals
`om 1 ^ (y e 0) · ∏ om (d^(k+1))^(y x' k)`. -/
theorem telescope (d n y : ℕ) (hd : 0 < d) (x x' e : ℕ → ℕ)
    (hc : ∀ k < n, x k + e (k + 1) = x' k + d * e k) :
    (∏ k ∈ range n, om (d ^ (k + 1)) ^ (y * x k)) * om (d ^ n) ^ (y * e n) =
      om (d ^ 0) ^ (y * e 0) * ∏ k ∈ range n, om (d ^ (k + 1)) ^ (y * x' k) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have h := hc n (Nat.lt_succ_self n)
    rw [prod_range_succ, prod_range_succ, mul_assoc, ← pow_add, ← mul_add, h, mul_add, pow_add,
      om_tower_mul d n y _ hd]
    linear_combination om (d ^ (n + 1)) ^ (y * x' n) * ih fun k hk => hc k (Nat.lt_succ_of_lt hk)

/-- The base-`d` value of `n` copies, MSB first. -/
def digVal (d : ℕ) (x : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => d * digVal d x n + x n

/-- A carry chain adds: `digVal x + e n = e 0 * d ^ n + digVal x'`. -/
theorem chain_value (d n : ℕ) (x x' e : ℕ → ℕ)
    (hc : ∀ k < n, x k + e (k + 1) = x' k + d * e k) :
    digVal d x n + e n = e 0 * d ^ n + digVal d x' n := by
  induction n with
  | zero => simp [digVal]
  | succ n ih =>
    have h := hc n (Nat.lt_succ_self n)
    have ih' := ih fun k hk => hc k (Nat.lt_succ_of_lt hk)
    simp only [digVal]
    calc d * digVal d x n + x n + e (n + 1) = d * (digVal d x n + e n) + x' n := by
          rw [add_assoc, h]; ring
      _ = e 0 * d ^ (n + 1) + (d * digVal d x' n + x' n) := by rw [ih']; ring

/-- `F³ · diag (om d ^ (y s)) · F` is the shift `i ↦ i + s` mod `d`; `F³ = F⁻¹` has entry
`conj (fou d o y)`. -/
theorem fourier_shift (d s o i : ℕ) (hd : 0 < d) :
    ∑ y ∈ range d, (starRingEnd ℂ) (fou d o y) * om d ^ (y * s) * fou d y i =
      if (s + i) % d = o % d then 1 else 0 := by
  have hr : (starRingEnd ℂ) (rt d) = rt d := Complex.conj_ofReal _
  have hterm : ∀ y, (starRingEnd ℂ) (fou d o y) * om d ^ (y * s) * fou d y i =
      (rt d * rt d)⁻¹ * (om d ^ (y * (s + i)) * (starRingEnd ℂ) (om d) ^ (y * o)) := by
    intro y
    simp only [fou, map_div₀, map_pow]
    change (starRingEnd ℂ) (om d) ^ (o * y) / (starRingEnd ℂ) (rt d) * om d ^ (y * s) *
      (om d ^ (y * i) / rt d) = _
    rw [hr, mul_add, pow_add, mul_comm o y]
    field_simp
  simp only [hterm, ← mul_sum, char_sum d hd, rt_sq]
  split_ifs
  · exact nat_inv_mul d hd
  · simp

/-- The interface of an every-`n` adder family `C⁻¹ P C`: `lhs n` the boxed circuit, `rhs n`
the carry chain, `rel n` equality of their semantics, `peel` the copy-0 cut of the boxed side
around `glue n`, `close` the step identity (telescope, then the Fourier shift on copy 0). -/
structure AdderIface (α : ℕ → Type*) where
  lhs : (n : ℕ) → α n
  rhs : (n : ℕ) → α n
  rel : (n : ℕ) → α n → α n → Prop
  trans : ∀ n a b c, rel n a b → rel n b c → rel n a c
  glue : (n : ℕ) → α n → α (n + 1)
  base : rel 0 (lhs 0) (rhs 0)
  peel : ∀ n, rel (n + 1) (lhs (n + 1)) (glue n (lhs n))
  congr : ∀ n a b, rel n a b → rel (n + 1) (glue n a) (glue n b)
  close : ∀ n, rel (n + 1) (glue n (rhs n)) (rhs (n + 1))

/-- The family theorem: `lhs n` and `rhs n` are related at every `n`. -/
theorem AdderIface.fam {α : ℕ → Type*} (I : AdderIface α) : ∀ n, I.rel n (I.lhs n) (I.rhs n)
  | 0 => I.base
  | n + 1 => I.trans _ _ _ _ (I.peel n)
      (I.trans _ _ _ _ (I.congr n _ _ (AdderIface.fam I n)) (I.close n))

theorem AdderIface.fam_8 {α : ℕ → Type*} (I : AdderIface α) : I.rel 8 (I.lhs 8) (I.rhs 8) :=
  I.fam 8

theorem AdderIface.fam_24 {α : ℕ → Type*} (I : AdderIface α) : I.rel 24 (I.lhs 24) (I.rhs 24) :=
  I.fam 24

end ArchytasZX
