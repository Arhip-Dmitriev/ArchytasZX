/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Bundle

/-!
# Exact arithmetic in `ℚ(ζ)`

Integer coefficient vectors `l` stand for `Σ l_i ζ^i` (`cvz`); `pmul`, `kadd`, `pscale` and
`pmod` are their product, sum, scaling and reduction by a polynomial vanishing at `ζ`.
`cyc_ne` proves a nonzero vector of length at most `φ(M)` does not vanish at `om M`. The
scalar lemmas write `I`, `√2`, `√3` and `om m` as powers of `om M`.
-/

namespace ArchytasZX

open Complex Finset Polynomial

/-- `∑` over `range d` as a list sum. -/
theorem fsum_list (f : ℕ → ℂ) (d : ℕ) : ∑ k ∈ range d, f k = ((List.range d).map f).sum := by
  induction d with
  | zero => simp
  | succ d ih => rw [sum_range_succ, ih, List.range_succ, List.map_append, List.sum_append]; simp

/-- `c` added to entry `n`. -/
def addAt : List ℤ → ℕ → ℤ → List ℤ
  | [], _, _ => []
  | a :: l, 0, c => (a + c) :: l
  | a :: l, n + 1, c => a :: addAt l n c

/-- `Σ l_i z^i`. -/
noncomputable def cvz (z : ℂ) : List ℤ → ℂ
  | [] => 0
  | a :: l => a + z * cvz z l

theorem addAt_length (l : List ℤ) (n : ℕ) (c : ℤ) : (addAt l n c).length = l.length := by
  induction l generalizing n with
  | nil => rfl
  | cons a l ih => cases n <;> simp [addAt, ih]

theorem cvz_addAt (z : ℂ) (l : List ℤ) (n : ℕ) (c : ℤ) (h : n < l.length) :
    cvz z (addAt l n c) = cvz z l + c * z ^ n := by
  induction l generalizing n with
  | nil => simp at h
  | cons a l ih =>
    cases n with
    | zero => simp [addAt, cvz]; ring
    | succ n =>
      simp only [addAt, cvz]
      rw [ih n (by simpa using h)]
      ring

theorem cvz_replicate (z : ℂ) (M : ℕ) : cvz z (List.replicate M 0) = 0 := by
  induction M with
  | zero => rfl
  | succ M ih => simp [List.replicate_succ, cvz, ih]

theorem pow_mod_of (z : ℂ) (M n : ℕ) (hz : z ^ M = 1) : z ^ n = z ^ (n % M) := by
  conv_lhs => rw [← Nat.div_add_mod n M]
  rw [pow_add, pow_mul, hz, one_pow, one_mul]

/-- The polynomial `Σ l_i X^i`. -/
noncomputable def pz : List ℤ → ℚ[X]
  | [] => 0
  | a :: l => C (a : ℚ) + X * pz l

theorem aeval_pz (z : ℂ) (l : List ℤ) : aeval z (pz l) = cvz z l := by
  induction l with
  | nil => simp [pz, cvz]
  | cons a l ih => simp [pz, cvz, ih]

theorem coeff_pz (l : List ℤ) (m : ℕ) : (pz l).coeff m = ((l.getD m 0 : ℤ) : ℚ) := by
  induction l generalizing m with
  | nil => simp [pz]
  | cons a l ih =>
    cases m with
    | zero => simp [pz]
    | succ m =>
      simp only [pz, coeff_add, coeff_C, coeff_X_mul, ih]
      simp

theorem pz_ne (l : List ℤ) (h : l.any (· ≠ 0) = true) : pz l ≠ 0 := by
  obtain ⟨a, ha, hne⟩ := List.any_eq_true.1 h
  obtain ⟨m, hm, rfl⟩ := List.getElem_of_mem ha
  intro h0
  have := coeff_pz l m
  rw [h0, coeff_zero, List.getD_eq_getElem _ _ hm] at this
  simp at hne
  exact hne (by exact_mod_cast this.symm)

theorem degree_pz (l : List ℤ) : (pz l).degree < l.length := by
  rw [degree_lt_iff_coeff_zero]
  intro m hm
  rw [coeff_pz, List.getD_eq_default _ _ hm]
  simp

/-- A nonzero integer vector of length at most `φ(M)` does not vanish at `om M`. -/
theorem cyc_ne (M : ℕ) (hM : 0 < M) (r : List ℤ) (hlen : r.length ≤ M.totient)
    (hnz : r.any (· ≠ 0) = true) : cvz (om M) r ≠ 0 := by
  intro h0
  have hp := om_prim M hM
  have hne := pz_ne r hnz
  have hroot : aeval (om M) (pz r) = 0 := by rw [aeval_pz, h0]
  have hdvd := minpoly.dvd ℚ (om M) hroot
  rw [← cyclotomic_eq_minpoly_rat hp hM] at hdvd
  have hle := natDegree_le_of_dvd hdvd hne
  rw [natDegree_cyclotomic] at hle
  have hlt := (natDegree_lt_iff_degree_lt hne).2 (degree_pz r)
  omega

/-- `om m` as a power of `om M` when `M = m q`. -/
theorem om_sub (M m q : ℕ) (hM : 0 < M) (h : M = m * q) : om m = om M ^ q := by
  have hm : 0 < m := Nat.pos_of_mul_pos_right (h ▸ hM)
  have hq : 0 < q := Nat.pos_of_mul_pos_left (h ▸ hM)
  unfold om
  rw [← Complex.exp_nat_mul]
  congr 1
  subst h
  have hm' : (m : ℂ) ≠ 0 := by exact_mod_cast hm.ne'
  have hq' : (q : ℂ) ≠ 0 := by exact_mod_cast hq.ne'
  push_cast
  field_simp

/-- `ex θ` is `om M ^ p` when `θ ≡ p / M` mod 1. -/
theorem ex_of (M p : ℕ) (z : ℤ) {θ : ℝ} (_hM : 0 < M) (h : θ = p / M + z) : ex θ = om M ^ p := by
  rw [ex_congr_int ⟨z, h⟩]
  unfold ex om
  rw [← Complex.exp_nat_mul]
  congr 1
  push_cast
  ring

/-- `Complex.I` as a power of `om M` when `M = 4 q`. -/
theorem I_sub (M q : ℕ) (hM : 0 < M) (h : M = 4 * q) : Complex.I = om M ^ q := by
  rw [← om_sub M 4 q hM h]
  unfold om
  have : 2 * (Real.pi : ℂ) * Complex.I / ((4 : ℕ) : ℂ) = (Real.pi / 2 : ℝ) * Complex.I := by
    push_cast; ring
  rw [this, Complex.exp_mul_I]
  rw [← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_pi_div_two, Real.sin_pi_div_two]
  simp

theorem rt_sub2 (M q : ℕ) (hM : 0 < M) (h : M = 8 * q) :
    rt 2 = om M ^ q + om M ^ (7 * q) := by
  rw [pow_mul', ← om_sub M 8 q hM h]
  have e : om 8 ^ 7 = (starRingEnd ℂ) (om 8) := by
    rw [conj_om]; apply eq_inv_of_mul_eq_one_left
    rw [← pow_succ]; exact om_pow_self 8 (by norm_num)
  have hre : (om 8).re = Real.cos (Real.pi / 4) := by
    unfold om
    rw [Complex.exp_re]
    have h1 : (2 * (Real.pi : ℂ) * Complex.I / ((8 : ℕ) : ℂ)).re = 0 := by simp
    have h2 : (2 * (Real.pi : ℂ) * Complex.I / ((8 : ℕ) : ℂ)).im = Real.pi / 4 := by
      simp; ring
    rw [h1, h2, Real.exp_zero, one_mul]
  rw [e, Complex.add_conj, hre, Real.cos_pi_div_four]
  simp only [rt]
  push_cast
  ring

theorem rt_sub3 (M q : ℕ) (hM : 0 < M) (h : M = 12 * q) :
    rt 3 = om M ^ q + om M ^ (11 * q) := by
  rw [pow_mul', ← om_sub M 12 q hM h]
  have e : om 12 ^ 11 = (starRingEnd ℂ) (om 12) := by
    rw [conj_om]; apply eq_inv_of_mul_eq_one_left
    rw [← pow_succ]; exact om_pow_self 12 (by norm_num)
  have hre : (om 12).re = Real.cos (Real.pi / 6) := by
    unfold om
    rw [Complex.exp_re]
    have h1 : (2 * (Real.pi : ℂ) * Complex.I / ((12 : ℕ) : ℂ)).re = 0 := by simp
    have h2 : (2 * (Real.pi : ℂ) * Complex.I / ((12 : ℕ) : ℂ)).im = Real.pi / 6 := by
      simp; ring
    rw [h1, h2, Real.exp_zero, one_mul]
  rw [e, Complex.add_conj, hre, Real.cos_pi_div_six]
  simp only [rt]
  push_cast
  ring

theorem rt_split (x k s : ℕ) (h : x = k ^ 2 * s) : rt x = k * rt s := by
  subst h
  simp only [rt]
  push_cast
  rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
  push_cast; ring

theorem rt_one : rt 1 = 1 := by simp [rt]

theorem rt_prod (x a b : ℕ) (h : x = a * b) : rt x = rt a * rt b := by
  subst h
  simp only [rt]
  push_cast
  rw [Real.sqrt_mul (by positivity)]
  push_cast; ring

theorem conj_om_pow (d : ℕ) (hd : 0 < d) : (starRingEnd ℂ) (om d) = om d ^ (d - 1) := by
  rw [conj_om]; apply inv_eq_of_mul_eq_one_right
  rw [← pow_succ', Nat.sub_add_cancel hd]; exact om_pow_self d hd

/-- The W node's entry as an integer. -/
def tWZ (d : ℕ) (os : List ℕ) (i : ℕ) : ℤ :=
  if i = 0 then (if ∀ o ∈ os, o = 0 then 1 else 0)
  else if i < d ∧ os.filter (· ≠ 0) = [i] then 1 else 0

theorem tWZ_eq (d : ℕ) (os : List ℕ) (i : ℕ) : (tWZ d os i : ℂ) = tW d os i := by
  unfold tWZ tW; split_ifs <;> simp

theorem digs_length (d k v : ℕ) : (digs d k v).length = k := by simp [digs]

theorem digsC_length (mult : ℕ → ℕ) (d : ℕ) (c : List ℕ) (v : ℕ) :
    (digsC mult d c v).length = (c.map mult).prod := by
  induction c generalizing v with
  | nil => simp [digsC]
  | cons m c ih =>
    simp only [digsC, List.length_flatMap, ih, List.map_const', List.sum_replicate,
      digs_length, smul_eq_mul, List.map_cons, List.prod_cons]

theorem tailVals_length (mult : ℕ → ℕ) (d : ℕ) (vs ws : List ℕ) (cs : List (List ℕ))
    (h : vs.length = ws.length) :
    (tailVals mult d vs cs).length = (tailVals mult d ws cs).length := by
  induction vs generalizing ws cs with
  | nil => cases ws <;> simp_all [tailVals]
  | cons v vs ih =>
    cases ws with
    | nil => simp at h
    | cons w ws =>
      cases cs with
      | nil => simp [tailVals]
      | cons c cs =>
        have := ih ws cs (by simpa using h)
        simp only [tailVals, List.zipWith_cons_cons, List.flatten_cons, List.length_append,
          digsC_length] at this ⊢
        omega

theorem bundled_length (mult : ℕ → ℕ) (d : ℕ) (vs : List ℕ) (cs : List (List ℕ)) :
    (bundled mult d vs cs).length = (bundled mult d (List.replicate vs.length 0) cs).length := by
  unfold bundled
  rw [List.length_append, List.length_append, List.length_replicate, List.length_take,
    List.length_take, List.length_replicate,
    tailVals_length mult d _ ((List.replicate vs.length 0).drop (vs.length - cs.length)) cs
      (by simp)]

theorem outCs_length (b : Kind) (n : ℕ) : n ≤ (b.outCs n).length := by
  induction b <;> simp [Kind.outCs, *] <;> omega

theorem inCs_length (b : Kind) (n : ℕ) : n ≤ (b.inCs n).length := by
  induction b <;> simp [Kind.inCs, *] <;> omega

theorem copyOuts_length (E : Env) (b : Kind) (d i : ℕ) (os : List ℕ) :
    (copyOuts E b d i os).length = os.length := by
  simp [copyOuts, copyLegs, outCs_length]

theorem copyIns_length (E : Env) (b : Kind) (d i : ℕ) (is : List ℕ) :
    (copyIns E b d i is).length = is.length := by
  simp [copyIns, copyLegs, inCs_length]

theorem prod_pow_mul {α : Type*} (r : ℂ) (c : ℕ) (l : List α) (f : α → ℂ) :
    (l.map fun x => r ^ c * f x).prod = r ^ (l.length * c) * (l.map f).prod := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [List.map_cons, List.prod_cons, ih, List.length_cons]; ring

/-- The padded sum of two coefficient vectors. -/
def kadd : List ℤ → List ℤ → List ℤ
  | [], b => b
  | a :: as, [] => a :: as
  | x :: a, y :: b => (x + y) :: kadd a b

/-- A coefficient vector scaled by `c`. -/
def pscale (c : ℤ) (a : List ℤ) : List ℤ := a.map (c * ·)

/-- The product of two coefficient vectors. -/
def pmul : List ℤ → List ℤ → List ℤ
  | [], _ => []
  | a :: as, b => kadd (pscale a b) (0 :: pmul as b)

/-- A vector without its trailing zeros. -/
def ktrim : List ℤ → List ℤ
  | [] => []
  | k :: l => if ktrim l = [] ∧ k = 0 then [] else k :: ktrim l

/-- One step reducing the top coefficient of `a` by `phi`. -/
def pmodStep (phi a : List ℤ) : List ℤ :=
  if a.length < phi.length then a
  else match a.getLast? with
    | some top => ktrim (kadd a (List.replicate (a.length - phi.length) 0 ++ pscale (-top) phi))
    | none => a

/-- `a` reduced by `phi`, `n` steps. -/
def pmod (phi : List ℤ) : ℕ → List ℤ → List ℤ
  | 0, a => a
  | n + 1, a => pmod phi n (pmodStep phi (ktrim a))

theorem cvz_kadd (z : ℂ) (a b : List ℤ) : cvz z (kadd a b) = cvz z a + cvz z b := by
  induction a generalizing b with
  | nil => simp [kadd, cvz]
  | cons x a ih =>
    cases b with
    | nil => simp [kadd, cvz]
    | cons y b => simp only [kadd, cvz, ih]; push_cast; ring

theorem cvz_pscale (z : ℂ) (c : ℤ) (a : List ℤ) : cvz z (pscale c a) = c * cvz z a := by
  induction a with
  | nil => simp [pscale, cvz]
  | cons x a ih => simp only [pscale, List.map_cons, cvz] at ih ⊢; rw [ih]; push_cast; ring

theorem cvz_pmul (z : ℂ) (a b : List ℤ) : cvz z (pmul a b) = cvz z a * cvz z b := by
  induction a with
  | nil => simp [pmul, cvz]
  | cons x a ih => simp only [pmul, cvz_kadd, cvz_pscale, cvz, ih]; push_cast; ring

theorem cvz_ktrim (z : ℂ) (a : List ℤ) : cvz z (ktrim a) = cvz z a := by
  induction a with
  | nil => rfl
  | cons k l ih =>
    simp only [ktrim]
    split_ifs with h
    · simp only [cvz]; rw [h.2, ← ih, h.1]; simp [cvz]
    · simp only [cvz, ih]

theorem cvz_append_replicate (z : ℂ) (k : ℕ) (a : List ℤ) :
    cvz z (List.replicate k 0 ++ a) = z ^ k * cvz z a := by
  induction k with
  | zero => simp
  | succ k ih => simp [List.replicate_succ, cvz, ih, pow_succ]; ring

theorem cvz_pmodStep (z : ℂ) (phi a : List ℤ) (h : cvz z phi = 0) :
    cvz z (pmodStep phi a) = cvz z a := by
  unfold pmodStep
  split_ifs
  · rfl
  · split
    · rw [cvz_ktrim, cvz_kadd, cvz_append_replicate, cvz_pscale, h]; ring
    · rfl

theorem cvz_pmod (z : ℂ) (phi : List ℤ) (h : cvz z phi = 0) (n : ℕ) (a : List ℤ) :
    cvz z (pmod phi n a) = cvz z a := by
  induction n generalizing a with
  | zero => rfl
  | succ n ih => simp only [pmod]; rw [ih, cvz_pmodStep z phi _ h, cvz_ktrim]

theorem cvz_zeros (z : ℂ) (a : List ℤ) (h : a.all (· == 0) = true) : cvz z a = 0 := by
  induction a with
  | nil => rfl
  | cons x a ih =>
    simp only [List.all_cons, Bool.and_eq_true, beq_iff_eq] at h
    simp [cvz, h.1, ih h.2]

/-- A vector reducing to zero by `phi` vanishes where `phi` does. -/
theorem cvz_eq_zero_of_pmod (z : ℂ) (phi a : List ℤ) (h : cvz z phi = 0) (n : ℕ)
    (hz : (pmod phi n a).all (· == 0) = true) : cvz z a = 0 := by
  rw [← cvz_pmod z phi h n a]; exact cvz_zeros z _ hz

/-- `Π xs_i ^ ks_i`. -/
noncomputable def mono : List ℂ → List ℤ → ℂ
  | x :: xs, k :: ks => x ^ k * mono xs ks
  | _, _ => 1

theorem mono_nil (xs : List ℂ) : mono xs [] = 1 := by cases xs <;> rfl

theorem mono_kadd (xs : List ℂ) (hx : ∀ x ∈ xs, x ≠ 0) (a b : List ℤ) :
    mono xs (kadd a b) = mono xs a * mono xs b := by
  induction xs generalizing a b with
  | nil => simp [mono]
  | cons x xs ih =>
    have hx0 : x ≠ 0 := hx x (by simp)
    have hxs : ∀ y ∈ xs, y ≠ 0 := fun y hy => hx y (by simp [hy])
    cases a with
    | nil => simp [kadd, mono_nil]
    | cons k a =>
      cases b with
      | nil => simp [kadd, mono_nil]
      | cons j b =>
        simp only [kadd, mono, ih hxs, zpow_add₀ hx0]
        ring

theorem mono_ktrim (xs : List ℂ) (ks : List ℤ) : mono xs (ktrim ks) = mono xs ks := by
  induction ks generalizing xs with
  | nil => rfl
  | cons k l ih =>
    cases xs with
    | nil => simp [mono]
    | cons x xs =>
      simp only [ktrim]
      split_ifs with h
      · rw [mono_nil]; simp only [mono, h.2, zpow_zero, one_mul]; rw [← ih, h.1, mono_nil]
      · simp only [mono, ih]

/-- `ex (k g)` is `ex g ^ k`. -/
theorem ex_zmul (g : ℝ) (k : ℤ) : ex ((k : ℝ) * g) = ex g ^ k := by
  unfold ex
  rw [← Complex.exp_int_mul]
  congr 1
  push_cast
  ring

/-- `ex` of a sum of integer multiples. -/
theorem ex_lin (gs : List ℝ) (ks : List ℤ) :
    ex ((List.zipWith (fun (g : ℝ) (k : ℤ) => (k : ℝ) * g) gs ks).sum) = mono (gs.map ex) ks := by
  induction gs generalizing ks with
  | nil => simp [mono, ex_zero]
  | cons g gs ih =>
    cases ks with
    | nil => simp [mono, ex_zero]
    | cons k ks =>
      simp only [List.zipWith_cons_cons, List.sum_cons, List.map_cons, mono, ex_add, ex_zmul,
        ih]

/-- `ex θ` as a power of `om M` times a monomial in the atoms `ex g`. -/
theorem ex_atoms (M p : ℕ) (z : ℤ) (gs : List ℝ) (ks : List ℤ) {θ : ℝ} (hM : 0 < M)
    (h : θ = p / M + z + (List.zipWith (fun (g : ℝ) (k : ℤ) => (k : ℝ) * g) gs ks).sum) :
    ex θ = om M ^ p * mono (gs.map ex) ks := by
  rw [h, ex_add, ex_lin, ex_of M p z hM rfl]

/-- `Complex.exp w` is `ex θ` when `w = 2πiθ`. -/
theorem exp_eq_ex (w : ℂ) (θ : ℝ) (h : w = 2 * Real.pi * Complex.I * θ) :
    Complex.exp w = ex θ := by
  rw [h]; rfl

theorem ex_atoms_ne (gs : List ℝ) : ∀ x ∈ gs.map ex, x ≠ 0 := by
  intro x hx
  obtain ⟨g, _, rfl⟩ := List.mem_map.1 hx
  exact Complex.exp_ne_zero _

end ArchytasZX
