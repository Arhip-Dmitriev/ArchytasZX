/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymNet

/-!
# Qubit networks over F_2 masks

A network of Z and X spiders without phases and F nodes, every dimension `2`, as one term
`q √2^k Π_d [d = 0] Π_{(u, w)} (-1)^(u w)` of affine forms over F_2. A form is a mask: bit `0`
the constant, bit `v + 1` the variable `v`. `runQ` multiplies the nodes in order and sums
the labels listed with each node: by a delta reading the label, else by the character sum.
`canonQ` brings the deltas to reduced echelon form and the phases to their remainders, and
`equal_q` reads `Equal` from `qCheck` on the two final terms.
-/

namespace ArchytasZX
namespace SymF2

open Finset

/-! ## Affine forms -/

/-- The mask of variable `v`. -/
def vb (v : ℕ) : ℕ := 2 ^ (v + 1)

/-- Bit `i`'s value: `1` for the constant bit, else variable `i - 1`. -/
def wt (ν : ℕ → ZMod 2) (i : ℕ) : ZMod 2 := if i = 0 then 1 else ν (i - 1)

/-- The value of the affine form `m`. -/
def par (ν : ℕ → ZMod 2) (m : ℕ) : ZMod 2 :=
  ∑ i ∈ range m.size, if m.testBit i then wt ν i else 0

theorem two_self (w : ZMod 2) : w + w = 0 := by
  revert w; decide

theorem zmod2_cases (w : ZMod 2) : w = 0 ∨ w = 1 := by
  revert w; decide

theorem zmod2_two : (2 : ZMod 2) = 0 := by decide

theorem par_range (ν : ℕ → ZMod 2) (m N : ℕ) (h : m.size ≤ N) :
    par ν m = ∑ i ∈ range N, if m.testBit i then wt ν i else 0 := by
  unfold par
  refine Finset.sum_subset (fun x hx => by simp only [Finset.mem_range] at hx ⊢; omega) ?_
  intro i _ hi
  simp only [Finset.mem_range, not_lt] at hi
  have : m.testBit i = false :=
    Nat.testBit_lt_two_pow (lt_of_lt_of_le (Nat.lt_size_self m) (Nat.pow_le_pow_right (by norm_num) hi))
  simp [this]

theorem size_xor_le (a b : ℕ) : (a ^^^ b).size ≤ max a.size b.size := by
  rw [Nat.size_le]
  exact Nat.xor_lt_two_pow
    (lt_of_lt_of_le (Nat.lt_size_self a) (Nat.pow_le_pow_right (by norm_num) (le_max_left _ _)))
    (lt_of_lt_of_le (Nat.lt_size_self b) (Nat.pow_le_pow_right (by norm_num) (le_max_right _ _)))

theorem par_xor (ν : ℕ → ZMod 2) (a b : ℕ) : par ν (a ^^^ b) = par ν a + par ν b := by
  rw [par_range ν _ _ (size_xor_le a b), par_range ν a _ (le_max_left _ _),
    par_range ν b _ (le_max_right _ _), ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Nat.testBit_xor]
  cases a.testBit i <;> cases b.testBit i <;> simp [two_self]

theorem par_zero (ν : ℕ → ZMod 2) : par ν 0 = 0 := by simp [par]

theorem par_one (ν : ℕ → ZMod 2) : par ν 1 = 1 := by
  rw [par_range ν 1 1 (by decide)]
  simp [wt]

theorem par_vb (ν : ℕ → ZMod 2) (v : ℕ) : par ν (vb v) = ν v := by
  unfold vb
  rw [par_range ν _ (v + 2) (by rw [Nat.size_pow])]
  simp only [Nat.testBit_two_pow]
  rw [Finset.sum_eq_single (v + 1)]
  · simp [wt]
  · intro i _ hi
    simp [Ne.symm hi]
  · intro h; simp at h

theorem par_update_of_not (ν : ℕ → ZMod 2) (x : ℕ) (c : ZMod 2) (m : ℕ)
    (h : m.testBit (x + 1) = false) : par (Function.update ν x c) m = par ν m := by
  unfold par
  refine Finset.sum_congr rfl fun i _ => ?_
  by_cases hi : m.testBit i
  · have hix : i ≠ x + 1 := fun e => by rw [e, h] at hi; exact absurd hi (by simp)
    simp only [hi, if_true, wt]
    by_cases h0 : i = 0
    · simp [h0]
    · simp only [h0, if_false]
      rw [Function.update_of_ne (by omega)]
  · simp [hi]

/-- `m` without the variable `x`. -/
def clr (x m : ℕ) : ℕ := if m.testBit (x + 1) then m ^^^ vb x else m

/-- `m` with `x` replaced by the form `v`. -/
def sub (x v m : ℕ) : ℕ := if m.testBit (x + 1) then m ^^^ vb x ^^^ v else m

theorem testBit_vb (x i : ℕ) : (vb x).testBit i = decide (x + 1 = i) := by
  unfold vb; exact Nat.testBit_two_pow

theorem clr_not (x m : ℕ) : (clr x m).testBit (x + 1) = false := by
  unfold clr
  split_ifs with h
  · rw [Nat.testBit_xor, h, testBit_vb]; simp
  · simpa using h

theorem par_upd (ν : ℕ → ZMod 2) (x : ℕ) (b : ZMod 2) (m : ℕ) :
    par (Function.update ν x b) m = (if m.testBit (x + 1) then b else 0) + par ν (clr x m) := by
  by_cases h : m.testBit (x + 1)
  · have hm : m = clr x m ^^^ vb x := by
      simp only [clr, h, if_true, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
    conv_lhs => rw [hm]
    rw [par_xor, par_update_of_not _ _ _ _ (clr_not x m), par_vb, Function.update_self]
    simp only [h, if_true]; ring
  · have h' : m.testBit (x + 1) = false := by simpa using h
    simp only [clr, h', Bool.false_eq_true, if_false, zero_add]
    exact par_update_of_not _ _ _ _ h'

theorem par_sub (ν : ℕ → ZMod 2) (x v m : ℕ) :
    par ν (sub x v m) = par (Function.update ν x (par ν v)) m := by
  rw [par_upd]
  unfold sub clr
  split_ifs with h
  · rw [par_xor]; ring
  · simp

/-! ## Terms -/

/-- `(-1)^z`. -/
noncomputable def sgn (z : ZMod 2) : ℂ := if z = 0 then 1 else -1

/-- `[z = 0]`. -/
noncomputable def dlt (z : ZMod 2) : ℂ := if z = 0 then 1 else 0

theorem sgn_add (a b : ZMod 2) : sgn (a + b) = sgn a * sgn b := by
  rcases zmod2_cases a with rfl | rfl <;> rcases zmod2_cases b with rfl | rfl <;>
    simp [sgn] <;> decide

theorem sgn_sum {α : Type} (f : α → ZMod 2) (l : List α) :
    (l.map fun a => sgn (f a)).prod = sgn (l.map f).sum := by
  induction l with
  | nil => simp [sgn]
  | cons a l ih => rw [List.map_cons, List.prod_cons, ih, List.map_cons, List.sum_cons, sgn_add]

/-- `q √2^k Π_d [d = 0] Π_{(u, w)} (-1)^(u w)`. -/
structure QTm where
  q : ℤ
  k : ℤ
  dels : List ℕ
  phs : List (ℕ × ℕ)
  deriving DecidableEq, Repr

/-- The product of the deltas. -/
noncomputable def dval (ν : ℕ → ZMod 2) (ds : List ℕ) : ℂ := (ds.map fun d => dlt (par ν d)).prod

/-- The product of the phases. -/
noncomputable def pval (ν : ℕ → ZMod 2) (ps : List (ℕ × ℕ)) : ℂ :=
  (ps.map fun p => sgn (par ν p.1 * par ν p.2)).prod

/-- The value of a term. -/
noncomputable def QTm.val (t : QTm) (ν : ℕ → ZMod 2) : ℂ :=
  (t.q : ℂ) * rt 2 ^ t.k * dval ν t.dels * pval ν t.phs

/-- The value of a term, `0` for `none`. -/
noncomputable def oval : Option QTm → (ℕ → ZMod 2) → ℂ
  | none, _ => 0
  | some t, ν => t.val ν

@[simp] theorem oval_none (ν : ℕ → ZMod 2) : oval none ν = 0 := rfl

@[simp] theorem oval_some (t : QTm) (ν : ℕ → ZMod 2) : oval (some t) ν = t.val ν := rfl

/-- The product. -/
def mulQ (a b : QTm) : QTm := ⟨a.q * b.q, a.k + b.k, a.dels ++ b.dels, a.phs ++ b.phs⟩

theorem rt2_ne : rt 2 ≠ 0 := rt_ne 2 (by norm_num)

theorem val_mulQ (a b : QTm) (ν : ℕ → ZMod 2) : (mulQ a b).val ν = a.val ν * b.val ν := by
  simp only [QTm.val, mulQ, dval, pval, List.map_append, List.prod_append, zpow_add₀ rt2_ne]
  push_cast; ring

theorem dval_perm (ν : ℕ → ZMod 2) {a b : List ℕ} (h : a.Perm b) : dval ν a = dval ν b :=
  (h.map _).prod_eq

theorem pval_perm (ν : ℕ → ZMod 2) {a b : List (ℕ × ℕ)} (h : a.Perm b) : pval ν a = pval ν b :=
  (h.map _).prod_eq

theorem dval_cons (ν : ℕ → ZMod 2) (d : ℕ) (ds : List ℕ) :
    dval ν (d :: ds) = dlt (par ν d) * dval ν ds := by simp [dval]

theorem pval_cons (ν : ℕ → ZMod 2) (p : ℕ × ℕ) (ps : List (ℕ × ℕ)) :
    pval ν (p :: ps) = sgn (par ν p.1 * par ν p.2) * pval ν ps := by simp [pval]

theorem dval_eq (ν : ℕ → ZMod 2) (ds : List ℕ) :
    dval ν ds = if ∀ d ∈ ds, par ν d = 0 then 1 else 0 := by
  induction ds with
  | nil => simp [dval]
  | cons d ds ih =>
    rw [dval_cons, ih]
    by_cases h : par ν d = 0 <;> simp [dlt, h]

/-! ## Summing a variable -/

/-- The first delta reading `x` and the others. -/
def pickD (x : ℕ) : List ℕ → Option (ℕ × List ℕ)
  | [] => none
  | d :: ds => if d.testBit (x + 1) then some (d, ds) else (pickD x ds).map fun p => (p.1, d :: p.2)

theorem pickD_some (x : ℕ) : ∀ (l : List ℕ) (d : ℕ) (r : List ℕ), pickD x l = some (d, r) →
    d.testBit (x + 1) = true ∧ l.Perm (d :: r)
  | [], _, _, h => by simp [pickD] at h
  | e :: l, d, r, h => by
    unfold pickD at h
    split_ifs at h with he
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨he, List.Perm.refl _⟩
    · cases hp : pickD x l with
      | none => rw [hp] at h; simp at h
      | some p =>
        rw [hp] at h
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨h1, h2⟩ := pickD_some x l p.1 p.2 hp
        exact ⟨h1, (h2.cons e).trans (List.Perm.swap _ _ _)⟩

theorem pickD_none (x : ℕ) : ∀ (l : List ℕ), pickD x l = none → ∀ d ∈ l, d.testBit (x + 1) = false
  | [], _, d, hd => by simp at hd
  | e :: l, h, d, hd => by
    unfold pickD at h
    split_ifs at h with he
    cases hp : pickD x l with
    | some p => rw [hp] at h; simp at h
    | none =>
      rcases List.mem_cons.1 hd with rfl | hd'
      · simpa using he
      · exact pickD_none x l hp d hd'

/-- The product of the deltas, `0` for `none`. -/
noncomputable def odv (ν : ℕ → ZMod 2) : Option (List ℕ) → ℂ
  | none => 0
  | some R => dval ν R

@[simp] theorem odv_none (ν : ℕ → ZMod 2) : odv ν none = 0 := rfl

@[simp] theorem odv_some (ν : ℕ → ZMod 2) (R : List ℕ) : odv ν (some R) = dval ν R := rfl

/-- The deltas `ds` with `x` replaced by `v` onto `acc` in reverse, those now `0 = 0` dropped;
none when one is `1 = 0`. -/
def subDels (x v : ℕ) : List ℕ → List ℕ → Option (List ℕ)
  | acc, [] => some acc
  | acc, d :: ds =>
    match sub x v d with
    | 0 => subDels x v acc ds
    | 1 => none
    | e + 2 => subDels x v ((e + 2) :: acc) ds

theorem subDels_ok (ν : ℕ → ZMod 2) (x v : ℕ) : ∀ acc ds : List ℕ,
    dval (Function.update ν x (par ν v)) ds * dval ν acc = odv ν (subDels x v acc ds)
  | acc, [] => by simp [subDels, dval]
  | acc, d :: ds => by
    rw [dval_cons, ← par_sub]
    have h := subDels_ok ν x v acc ds
    have h' := fun e => subDels_ok ν x v ((e + 2) :: acc) ds
    rw [subDels]
    generalize sub x v d = m
    rcases m with _ | _ | e
    · rw [par_zero, ← h]; simp [dlt]
    · rw [par_one]; simp [dlt]
    · rw [← h' e, dval_cons]; ring

/-- `(u, w) :: l`, both numbers reduced first. -/
def pcons (u w : ℕ) (l : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  match u, w with
  | 0, 0 => (0, 0) :: l
  | 0, b + 1 => (0, b + 1) :: l
  | a + 1, 0 => (a + 1, 0) :: l
  | a + 1, b + 1 => (a + 1, b + 1) :: l

theorem pcons_eq (u w : ℕ) (l : List (ℕ × ℕ)) : pcons u w l = (u, w) :: l := by
  cases u <;> cases w <;> rfl

/-- `e :: l`, the number reduced first. -/
def scons (e : ℕ) (l : List ℕ) : List ℕ :=
  match e with
  | 0 => 0 :: l
  | k + 1 => (k + 1) :: l

theorem scons_eq (e : ℕ) (l : List ℕ) : scons e l = e :: l := by cases e <;> rfl

/-- The phases with `x` replaced by `v`. -/
def subPhs (x v : ℕ) : List (ℕ × ℕ) → List (ℕ × ℕ)
  | [] => []
  | p :: ps => pcons (sub x v p.1) (sub x v p.2) (subPhs x v ps)

theorem subPhs_eq (x v : ℕ) (ps : List (ℕ × ℕ)) :
    subPhs x v ps = ps.map fun p => (sub x v p.1, sub x v p.2) := by
  induction ps with
  | nil => rfl
  | cons p ps ih => rw [subPhs, pcons_eq, ih, List.map_cons]

theorem subPhs_ok (ν : ℕ → ZMod 2) (x v : ℕ) (ps : List (ℕ × ℕ)) :
    pval (Function.update ν x (par ν v)) ps = pval ν (subPhs x v ps) := by
  rw [subPhs_eq]
  simp only [pval, List.map_map, Function.comp_def, par_sub]

/-- The sum over `x` through the delta `d` reading it, the others `ds`. -/
def elimD (x d : ℕ) (ds : List ℕ) (t : QTm) : Option QTm :=
  let v := d ^^^ vb x
  (subDels x v [] ds).map fun r => ⟨t.q, t.k, r, subPhs x v t.phs⟩

/-- One phase's part of the character: the coefficient of `x`. -/
def cA (x : ℕ) (p : ℕ × ℕ) : ℕ :=
  let a := p.1.testBit (x + 1)
  let b := p.2.testBit (x + 1)
  (if a && b then 1 else 0) ^^^ (if a then clr x p.2 else 0) ^^^ (if b then clr x p.1 else 0)

/-- The phases without `x`, those with a zero form dropped. -/
def clrPhs (x : ℕ) : List (ℕ × ℕ) → List (ℕ × ℕ)
  | [] => []
  | p :: ps =>
    if clr x p.1 = 0 ∨ clr x p.2 = 0 then clrPhs x ps else pcons (clr x p.1) (clr x p.2) (clrPhs x ps)

/-- The character sum over `x`, read by no delta: `2 [A = 0]`. -/
def charX (x : ℕ) (t : QTm) : Option QTm :=
  match t.phs.foldr (fun p acc => acc ^^^ cA x p) 0 with
  | 0 => some ⟨t.q, t.k + 2, t.dels, clrPhs x t.phs⟩
  | 1 => none
  | a + 2 => some ⟨t.q, t.k + 2, (a + 2) :: t.dels, clrPhs x t.phs⟩

/-- The sum over the variable `x`. -/
def sumX (x : ℕ) (t : QTm) : Option QTm :=
  match pickD x t.dels with
  | some (d, ds) => elimD x d ds t
  | none => charX x t

theorem dlt_add_bit (y b : ZMod 2) : dlt (y + b) = if b = y then 1 else 0 := by
  rcases zmod2_cases y with rfl | rfl <;> rcases zmod2_cases b with rfl | rfl <;>
    simp [dlt] <;> decide

theorem elimD_ok (ν : ℕ → ZMod 2) (x d : ℕ) (ds : List ℕ) (t : QTm)
    (hd : d.testBit (x + 1) = true) (hp : t.dels.Perm (d :: ds)) :
    t.val (Function.update ν x 0) + t.val (Function.update ν x 1) = oval (elimD x d ds t) ν := by
  set v := d ^^^ vb x
  have hv : v.testBit (x + 1) = false := by
    simp only [v, Nat.testBit_xor, hd, testBit_vb]; simp
  have hdv : d = v ^^^ vb x := by simp [v, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  have hpar : ∀ b, par (Function.update ν x b) d = par ν v + b := fun b => by
    rw [hdv, par_xor, par_update_of_not _ _ _ _ hv, par_vb, Function.update_self]
  have hval : ∀ b, t.val (Function.update ν x b) = (if b = par ν v then 1 else 0) *
      ((t.q : ℂ) * rt 2 ^ t.k * dval (Function.update ν x b) ds *
        pval (Function.update ν x b) t.phs) := fun b => by
    unfold QTm.val
    rw [dval_perm _ hp, dval_cons, hpar, dlt_add_bit]; ring
  rw [hval, hval]
  have hy : (if (0 : ZMod 2) = par ν v then (1 : ℂ) else 0) * ((t.q : ℂ) * rt 2 ^ t.k *
      dval (Function.update ν x 0) ds * pval (Function.update ν x 0) t.phs) +
      (if (1 : ZMod 2) = par ν v then (1 : ℂ) else 0) * ((t.q : ℂ) * rt 2 ^ t.k *
      dval (Function.update ν x 1) ds * pval (Function.update ν x 1) t.phs) =
      (t.q : ℂ) * rt 2 ^ t.k * dval (Function.update ν x (par ν v)) ds *
        pval (Function.update ν x (par ν v)) t.phs := by
    rcases zmod2_cases (par ν v) with h | h <;> rw [h] <;> simp
  have he : elimD x d ds t =
      (subDels x v [] ds).map fun r => ⟨t.q, t.k, r, subPhs x v t.phs⟩ := rfl
  have hs := subDels_ok ν x v [] ds
  rw [show dval ν [] = 1 from rfl, mul_one] at hs
  rw [hy, hs, subPhs_ok, he]
  cases subDels x v [] ds with
  | none => simp
  | some r => simp [QTm.val]

theorem cA_par (ν : ℕ → ZMod 2) (x : ℕ) (p : ℕ × ℕ) (b : ZMod 2) :
    par (Function.update ν x b) p.1 * par (Function.update ν x b) p.2 =
      b * par ν (cA x p) + par ν (clr x p.1) * par ν (clr x p.2) := by
  rw [par_upd, par_upd]
  unfold cA
  simp only [par_xor]
  rcases zmod2_cases b with rfl | rfl <;>
    cases p.1.testBit (x + 1) <;> cases p.2.testBit (x + 1) <;> simp [par_one, par_zero] <;> ring

theorem clrPhs_ok (ν : ℕ → ZMod 2) (x : ℕ) (ps : List (ℕ × ℕ)) :
    sgn (ps.map fun p => par ν (clr x p.1) * par ν (clr x p.2)).sum = pval ν (clrPhs x ps) := by
  induction ps with
  | nil => simp [sgn, pval, clrPhs]
  | cons p ps ih =>
    rw [List.map_cons, List.sum_cons, sgn_add, ih, clrPhs]
    by_cases h : clr x p.1 = 0 ∨ clr x p.2 = 0
    · rw [if_pos h]
      rcases h with h | h <;> simp [h, par_zero, sgn]
    · rw [if_neg h, pcons_eq, pval_cons]

theorem fold_cA (ν : ℕ → ZMod 2) (x : ℕ) (ps : List (ℕ × ℕ)) :
    par ν (ps.foldr (fun p acc => acc ^^^ cA x p) 0) = (ps.map fun p => par ν (cA x p)).sum := by
  induction ps with
  | nil => simp [par_zero]
  | cons p ps ih => rw [List.foldr_cons, par_xor, ih, List.map_cons, List.sum_cons]; ring

theorem charX_ok (ν : ℕ → ZMod 2) (x : ℕ) (t : QTm)
    (hd : ∀ d ∈ t.dels, d.testBit (x + 1) = false) :
    t.val (Function.update ν x 0) + t.val (Function.update ν x 1) = oval (charX x t) ν := by
  obtain ⟨A, hA⟩ : ∃ A, t.phs.foldr (fun p acc => acc ^^^ cA x p) 0 = A := ⟨_, rfl⟩
  have hdv : ∀ b, dval (Function.update ν x b) t.dels = dval ν t.dels := fun b => by
    unfold dval
    congr 1
    exact List.map_congr_left fun d hd' => by rw [par_update_of_not _ _ _ _ (hd d hd')]
  have hpv : ∀ b, pval (Function.update ν x b) t.phs =
      sgn (b * par ν A) * sgn (t.phs.map fun p => par ν (clr x p.1) * par ν (clr x p.2)).sum := by
    intro b
    unfold pval
    rw [sgn_sum, ← sgn_add]
    congr 1
    simp only [cA_par, List.sum_map_add, ← hA, fold_cA, List.sum_map_mul_left]
  have hval : ∀ b, t.val (Function.update ν x b) = sgn (b * par ν A) *
      ((t.q : ℂ) * rt 2 ^ t.k * dval ν t.dels * pval ν (clrPhs x t.phs)) := fun b => by
    unfold QTm.val
    rw [hdv, hpv, clrPhs_ok]; ring
  rw [hval, hval]
  have h2 : rt 2 ^ (t.k + 2) = rt 2 ^ t.k * 2 := by
    rw [zpow_add₀ rt2_ne]; norm_num [rt_sq']
  have hc : charX x t = match A with
      | 0 => some ⟨t.q, t.k + 2, t.dels, clrPhs x t.phs⟩
      | 1 => none
      | a + 2 => some ⟨t.q, t.k + 2, (a + 2) :: t.dels, clrPhs x t.phs⟩ := by
    unfold charX; rw [hA]
  rw [hc]
  rcases A with _ | _ | a <;> dsimp only
  · rw [oval_some, par_zero]
    simp only [QTm.val]
    rw [h2]
    simp [sgn]; ring
  · rw [oval_none, par_one]
    simp [sgn]
  · rw [oval_some]
    simp only [QTm.val]
    rw [h2, dval_cons]
    rcases zmod2_cases (par ν (a + 2)) with h | h <;> simp [h, sgn, dlt] <;> ring

theorem sumX_ok (ν : ℕ → ZMod 2) (x : ℕ) (t : QTm) :
    t.val (Function.update ν x 0) + t.val (Function.update ν x 1) = oval (sumX x t) ν := by
  unfold sumX
  cases h : pickD x t.dels with
  | some p =>
    obtain ⟨h1, h2⟩ := pickD_some x t.dels p.1 p.2 h
    exact elimD_ok ν x p.1 p.2 t h1 h2
  | none => exact charX_ok ν x t (pickD_none x t.dels h)

/-! ## Generators -/

/-- The form `Σ_l l`. -/
def legsX (ls : List ℕ) : ℕ := ls.foldr (fun l m => m ^^^ vb l) 0

/-- The forms `l + l0`. -/
def zDels (l0 : ℕ) : List ℕ → List ℕ
  | [] => []
  | l :: ls => scons (vb l ^^^ vb l0) (zDels l0 ls)

theorem zDels_eq (l0 : ℕ) (ls : List ℕ) : zDels l0 ls = ls.map fun l => vb l ^^^ vb l0 := by
  induction ls with
  | nil => rfl
  | cons l ls ih => rw [zDels, scons_eq, ih, List.map_cons]

/-- A generator's term: the Z spider `Π_l [l + l0 = 0]`, the X spider `√2^(2 - n) [Σ l = 0]`,
F `√2^(-1) (-1)^(o i)`; none for any other generator. -/
def trQ (n : Node) : Option QTm :=
  match n.kind with
  | .Z => match n.outs ++ n.ins with
    | [] => some ⟨1, 2, [], []⟩
    | l0 :: ls => some ⟨1, 0, zDels l0 ls, []⟩
  | .X => some ⟨1, 2 - ((n.outs ++ n.ins).length : ℤ), scons (legsX (n.outs ++ n.ins)) [], []⟩
  | .F => match n.outs, n.ins with
    | [o], [i] => some ⟨1, -1, [], pcons (vb o) (vb i) []⟩
    | _, _ => none
  | _ => none

/-- The Boolean image of a label assignment. -/
def bits (σ : ℕ → ℕ) : ℕ → ZMod 2 := fun l => (σ l : ZMod 2)

theorem bit_cases (n : ℕ) (h : n < 2) : n = 0 ∨ n = 1 := by omega

theorem om_two : om 2 = -1 := by
  unfold om
  rw [show (2 : ℂ) * Real.pi * Complex.I / ((2 : ℕ) : ℂ) = Real.pi * Complex.I by push_cast; ring]
  exact Complex.exp_pi_mul_I

theorem fou_bits (j k : ℕ) (hj : j < 2) (hk : k < 2) :
    fou 2 j k = sgn ((j : ZMod 2) * (k : ZMod 2)) / rt 2 := by
  unfold fou
  rcases bit_cases j hj with rfl | rfl <;> rcases bit_cases k hk with rfl | rfl <;>
    simp [om_two, sgn, rt]

theorem fou_conj (j k : ℕ) (hj : j < 2) (hk : k < 2) :
    (starRingEnd ℂ) (fou 2 j k) = fou 2 j k := by
  rw [fou_bits j k hj hk]
  unfold sgn
  split_ifs <;> simp [map_div₀, rt, Complex.conj_ofReal]

theorem bits_eq_iff (a b : ℕ) (ha : a < 2) (hb : b < 2) :
    ((a : ZMod 2) + (b : ZMod 2) = 0) ↔ a = b := by
  rcases bit_cases a ha with rfl | rfl <;> rcases bit_cases b hb with rfl | rfl <;> decide

theorem z_val (σ : ℕ → ℕ) (hσ : ∀ l, σ l < 2) (l0 : ℕ) (ls : List ℕ) :
    dval (bits σ) (ls.map fun l => vb l ^^^ vb l0) =
      if ∀ w ∈ ls.map σ, w = σ l0 then 1 else 0 := by
  rw [dval_eq]
  simp only [List.mem_map, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, par_xor,
    par_vb, bits]
  congr 1
  apply propext
  exact forall₂_congr fun l _ => bits_eq_iff _ _ (hσ l) (hσ l0)

theorem tZ_zero (vs : List ℕ) (v : ℕ) (hv : v < 2) :
    tZ 2 (fun _ => 0) (v :: vs) = if ∀ w ∈ vs, w = v then 1 else 0 := by
  unfold tZ
  rw [Finset.sum_range_succ, Finset.sum_range_one]
  rcases bit_cases v hv with rfl | rfl <;> simp [ex_zero]

theorem par_legsX (ν : ℕ → ZMod 2) (ls : List ℕ) : par ν (legsX ls) = (ls.map ν).sum := by
  induction ls with
  | nil => simp [legsX, par_zero]
  | cons l ls ih =>
    simp only [legsX, List.foldr_cons] at ih ⊢
    rw [par_xor, ih, par_vb, List.map_cons, List.sum_cons]; ring

theorem prod_fou (σ : ℕ → ℕ) (hσ : ∀ l, σ l < 2) (k : ℕ) (hk : k < 2) (ls : List ℕ) :
    (ls.map fun l => fou 2 (σ l) k).prod =
      sgn ((ls.map (bits σ)).sum * (k : ZMod 2)) * (rt 2)⁻¹ ^ ls.length := by
  induction ls with
  | nil => simp [sgn]
  | cons l ls ih =>
    rw [List.map_cons, List.prod_cons, ih, fou_bits _ _ (hσ l) hk, List.map_cons, List.sum_cons,
      add_mul, sgn_add, List.length_cons, pow_succ]
    simp only [bits]; ring

/-- Both values agree on an assignment of bits, for a node of dimension `2` without phases. -/
theorem val_trQ (E : Env) (σ : ℕ → ℕ) (hσ : ∀ l, σ l < 2) (n : Node) (a : QTm)
    (hd : E.dimv n.dim = 2) (hph : ∀ j, E.phase n.ph j = 0) (h : trQ n = some a) :
    val E σ n = a.val (bits σ) := by
  have hphf : E.phase n.ph = fun _ => 0 := funext hph
  obtain ⟨kind, os, is, dim, ph⟩ := n
  simp only at hd hphf h
  cases kind with
  | Z =>
    simp only [trQ, zDels_eq] at h
    simp only [val, hd, hphf]
    rw [← List.map_append]
    cases hl : os ++ is with
    | nil =>
      rw [hl] at h
      simp only [Option.some.injEq] at h
      subst h
      simp [tZ, ex_zero, QTm.val, dval, pval, Finset.sum_range_succ, rt_sq']
    | cons l0 ls =>
      rw [hl] at h
      simp only [Option.some.injEq] at h
      subst h
      rw [List.map_cons, tZ_zero _ _ (hσ l0)]
      simp only [QTm.val, z_val σ hσ l0 ls, pval]
      simp
  | X =>
    simp only [trQ, scons_eq, Option.some.injEq] at h
    subst h
    simp only [val, hd, hphf, tX, ex_zero, one_mul, List.map_map, Function.comp_def]
    have hc : ∀ k < 2, (is.map fun i => (starRingEnd ℂ) (fou 2 (σ i) k)).prod =
        (is.map fun i => fou 2 (σ i) k).prod := fun k hk => by
      congr 1; exact List.map_congr_left fun i _ => fou_conj _ _ (hσ i) hk
    rw [Finset.sum_range_succ, Finset.sum_range_one, hc 0 (by norm_num), hc 1 (by norm_num),
      prod_fou σ hσ 0 (by norm_num) os, prod_fou σ hσ 0 (by norm_num) is,
      prod_fou σ hσ 1 (by norm_num) os, prod_fou σ hσ 1 (by norm_num) is]
    simp only [QTm.val, dval, pval, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
      par_legsX, List.map_append, List.sum_append, mul_one]
    have h2 : rt 2 ^ (2 - ((os ++ is).length : ℤ)) = 2 * (rt 2)⁻¹ ^ (os.length + is.length) := by
      rw [zpow_sub₀ rt2_ne, List.length_append, zpow_natCast,
        show rt 2 ^ (2 : ℤ) = 2 by norm_num [rt_sq'], inv_pow, div_eq_mul_inv]
    rw [h2]
    rcases zmod2_cases (os.map (bits σ)).sum with h1 | h1 <;>
      rcases zmod2_cases (is.map (bits σ)).sum with h3 | h3 <;>
      simp [h1, h3, sgn, dlt, pow_add, two_self] <;> (try rw [if_pos (by decide)]) <;> ring
  | F =>
    rcases os with _ | ⟨o, _ | _⟩ <;> rcases is with _ | ⟨i, _ | _⟩ <;>
      simp only [trQ, pcons_eq, reduceCtorEq, Option.some.injEq] at h
    subst h
    simp only [val, hd, List.map_cons, List.map_nil]
    rw [fou_bits _ _ (hσ o) (hσ i)]
    simp only [QTm.val, dval, pval, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
      par_vb, bits, zpow_neg, zpow_one]
    ring
  | _ => simp [trQ] at h

/-! ## Runs -/

/-- The summed labels as a mask, and the term so far, `none` when zero. -/
abbrev St := ℕ × Option QTm

/-- The labels `ls` summed in turn, each a bit of `IM` not yet in the mask. -/
def sumLs (IM : ℕ) : List ℕ → St → Option St
  | [], s => some s
  | l :: ls, s =>
    if IM.testBit l && !s.1.testBit l then
      match s.1 ||| 2 ^ l with
      | 0 => sumLs IM ls (0, s.2.bind (sumX l))
      | S + 1 => sumLs IM ls (S + 1, s.2.bind (sumX l))
    else none

theorem sumLs_cons (IM l : ℕ) (ls : List ℕ) (s : St) :
    sumLs IM (l :: ls) s = if IM.testBit l && !s.1.testBit l then
      sumLs IM ls (s.1 ||| 2 ^ l, s.2.bind (sumX l)) else none := by
  rw [sumLs]
  split_ifs
  · generalize s.1 ||| 2 ^ l = m
    cases m <;> rfl
  · rfl

/-- Node `p.1` multiplied in, none of its legs summed yet, then the labels `p.2` summed. -/
def stepQ (IM : ℕ) (p : Node × List ℕ) (s : St) : Option St :=
  if (p.1.outs ++ p.1.ins).all (fun l => !s.1.testBit l) then
    match trQ p.1 with
    | some a => sumLs IM p.2 (s.1, s.2.map fun t => mulQ a t)
    | none => none
  else none

/-- The steps in order. -/
def runQ (IM : ℕ) : List (Node × List ℕ) → St → Option St
  | [], s => some s
  | p :: ps, s => (stepQ IM p s).bind (runQ IM ps)

theorem runQ_append (IM : ℕ) : ∀ (a b : List (Node × List ℕ)) (s : St),
    runQ IM (a ++ b) s = (runQ IM a s).bind (runQ IM b)
  | [], _, _ => rfl
  | p :: a, b, s => by
    simp only [List.cons_append, runQ]
    cases stepQ IM p s with
    | none => rfl
    | some s' => exact runQ_append IM a b s'

theorem runQ_cut (IM : ℕ) (R : List (Node × List ℕ)) (k : ℕ) (s s' : St)
    (h : runQ IM (R.take k) s = some s') : runQ IM R s = runQ IM (R.drop k) s' := by
  conv_lhs => rw [← List.take_append_drop k R]
  rw [runQ_append, h]; rfl

theorem runQ_seg (IM : ℕ) (R : List (Node × List ℕ)) (i k j : ℕ) (s s' : St) (hj : i + k = j)
    (h : runQ IM ((R.drop i).take k) s = some s') :
    runQ IM (R.drop i) s = runQ IM (R.drop j) s' := by
  rw [runQ_cut IM (R.drop i) k s s' h, List.drop_drop, hj]

theorem runQ_end (IM : ℕ) (R : List (Node × List ℕ)) (j : ℕ) (s : St) (h : R.length ≤ j) :
    runQ IM (R.drop j) s = some s := by
  rw [List.drop_eq_nil_of_le h]; rfl

/-- The summed labels are `Ls`, each of `IM`, and the term is the sum over them of the nodes
`P` on assignments of bits. -/
def Inv (E : Env) (IM : ℕ) (P : List Node) (Ls : List ℕ) (s : St) : Prop :=
  (∀ l, s.1.testBit l = true ↔ l ∈ Ls) ∧ Ls.Nodup ∧ (∀ l ∈ Ls, IM.testBit l = true) ∧
    ∀ σ : ℕ → ℕ, (∀ l, σ l < 2) → oval s.2 (bits σ) = sumOver (labDim E) Ls (prodVal E P) σ

theorem bits_update (σ : ℕ → ℕ) (l j : ℕ) :
    bits (Function.update σ l j) = Function.update (bits σ) l (j : ZMod 2) := by
  funext v
  by_cases h : v = l
  · subst h; simp [bits]
  · simp [bits, Function.update_of_ne h]

theorem oval_sum (x : ℕ) (t : Option QTm) (ν : ℕ → ZMod 2) :
    oval (t.bind (sumX x)) ν = oval t (Function.update ν x 0) + oval t (Function.update ν x 1) := by
  cases t with
  | none => simp [oval]
  | some t => rw [Option.bind_some, oval_some, oval_some]; exact (sumX_ok ν x t).symm

theorem sumLs_ok (E : Env) (IM : ℕ) (hIM : ∀ l, IM.testBit l = true → labDim E l = 2)
    (P : List Node) : ∀ (ls Ls : List ℕ) (s s' : St), Inv E IM P Ls s → sumLs IM ls s = some s' →
      Inv E IM P (ls.reverse ++ Ls) s'
  | [], Ls, s, s', hI, h => by
    simp only [sumLs, Option.some.injEq] at h; subst h; simpa using hI
  | l :: ls, Ls, s, s', hI, h => by
    rw [sumLs_cons] at h
    split_ifs at h with hc
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at hc
    obtain ⟨hm, hn, hi, hv⟩ := hI
    have hlL : l ∉ Ls := fun h' => by rw [← hm] at h'; rw [h'] at hc; exact absurd hc.2 (by simp)
    have hI' : Inv E IM P (l :: Ls) (s.1 ||| 2 ^ l, s.2.bind (sumX l)) := by
      refine ⟨fun l' => ?_, List.nodup_cons.2 ⟨hlL, hn⟩, ?_, fun σ hσ => ?_⟩
      · rw [Nat.testBit_lor, Nat.testBit_two_pow, Bool.or_eq_true, hm, List.mem_cons]
        simp only [decide_eq_true_eq]; tauto
      · intro l' hl'
        rcases List.mem_cons.1 hl' with rfl | hl'
        · exact hc.1
        · exact hi l' hl'
      · simp only [sumOver, hIM l hc.1, Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
        have hup : ∀ j < 2, ∀ v, Function.update σ l j v < 2 := by
          intro j hj v
          by_cases hv' : v = l
          · subst hv'; simpa using hj
          · rw [Function.update_of_ne hv']; exact hσ v
        rw [← hv _ (hup 0 (by norm_num)), ← hv _ (hup 1 (by norm_num)), bits_update, bits_update,
          oval_sum]
        norm_num
    have := sumLs_ok E IM hIM P ls (l :: Ls) _ s' hI' h
    simpa using this

theorem prodVal_snoc (E : Env) (P : List Node) (n : Node) (σ : ℕ → ℕ) :
    prodVal E (P ++ [n]) σ = val E σ n * prodVal E P σ := by
  simp [prodVal, List.prod_append, mul_comm]

/-- Each node of dimension `2` without phases. -/
def NodesOK (E : Env) (ns : List Node) : Prop :=
  ∀ n ∈ ns, E.dimv n.dim = 2 ∧ ∀ j, E.phase n.ph j = 0

theorem stepQ_ok (E : Env) (IM : ℕ) (hIM : ∀ l, IM.testBit l = true → labDim E l = 2)
    (P : List Node) (Ls : List ℕ) (p : Node × List ℕ) (s s' : St)
    (hn : E.dimv p.1.dim = 2 ∧ ∀ j, E.phase p.1.ph j = 0)
    (hI : Inv E IM P Ls s) (h : stepQ IM p s = some s') :
    ∃ Ls', Inv E IM (P ++ [p.1]) Ls' s' := by
  unfold stepQ at h
  split_ifs at h with hleg
  cases ha : trQ p.1 with
  | none => rw [ha] at h; simp at h
  | some a =>
    rw [ha] at h
    simp only at h
    obtain ⟨hm, hnd, hi, hv⟩ := hI
    have hdis : ∀ x ∈ p.1.labels, x ∉ Ls := by
      intro x hx hxL
      rw [← hm] at hxL
      have := List.all_eq_true.1 hleg x hx
      rw [hxL] at this; simp at this
    have hI' : Inv E IM (P ++ [p.1]) Ls (s.1, s.2.map fun t => mulQ a t) := by
      refine ⟨hm, hnd, hi, fun σ hσ => ?_⟩
      have e1 : sumOver (labDim E) Ls (prodVal E (P ++ [p.1])) σ =
          sumOver (labDim E) Ls (fun τ => val E τ p.1 * prodVal E P τ) σ :=
        congrArg (fun f => sumOver (labDim E) Ls f σ) (funext fun τ => prodVal_snoc E P p.1 τ)
      rw [e1, sumOver_const_mul (labDim E) Ls p.1.labels _ _ (val_reads E p.1) hdis σ, ← hv σ hσ,
        val_trQ E σ hσ p.1 a hn.1 hn.2 ha]
      cases s.2 with
      | none => simp
      | some t => rw [Option.map_some, oval_some, oval_some, val_mulQ]
    exact ⟨_, sumLs_ok E IM hIM (P ++ [p.1]) p.2 Ls _ s' hI' h⟩

theorem runQ_ok (E : Env) (IM : ℕ) (hIM : ∀ l, IM.testBit l = true → labDim E l = 2) :
    ∀ (ps : List (Node × List ℕ)) (P : List Node) (Ls : List ℕ) (s s' : St),
      NodesOK E (ps.map Prod.fst) → Inv E IM P Ls s → runQ IM ps s = some s' →
      ∃ Ls', Inv E IM (P ++ ps.map Prod.fst) Ls' s'
  | [], P, Ls, s, s', _, hI, h => by
    simp only [runQ, Option.some.injEq] at h; subst h; exact ⟨Ls, by simpa using hI⟩
  | p :: ps, P, Ls, s, s', hn, hI, h => by
    simp only [runQ] at h
    cases hs : stepQ IM p s with
    | none => rw [hs] at h; simp at h
    | some s1 =>
      rw [hs] at h
      simp only [Option.bind_some] at h
      obtain ⟨L1, hI1⟩ := stepQ_ok E IM hIM P Ls p s s1 (hn p.1 (by simp)) hI hs
      obtain ⟨L2, hI2⟩ := runQ_ok E IM hIM ps (P ++ [p.1]) L1 s1 s'
        (fun n hn' => hn n (by simp only [List.map_cons, List.mem_cons]; exact Or.inr hn')) hI1 h
      exact ⟨L2, by simpa using hI2⟩

/-! ## Networks -/

/-- The mask of the labels `ls`. -/
def maskL (ls : List ℕ) : ℕ := ls.foldr (fun l m => m ||| 2 ^ l) 0

theorem maskL_bit (ls : List ℕ) (l : ℕ) : (maskL ls).testBit l = true ↔ l ∈ ls := by
  induction ls with
  | nil => simp [maskL]
  | cons x ls ih =>
    simp only [maskL, List.foldr_cons] at ih ⊢
    rw [Nat.testBit_lor, Nat.testBit_two_pow, Bool.or_eq_true, ih, List.mem_cons]
    simp only [decide_eq_true_eq]; tauto

/-- No label twice, none of the mask `m`. -/
def nodupM : List ℕ → ℕ → Bool
  | [], _ => true
  | l :: ls, m => !m.testBit l && nodupM ls (m ||| 2 ^ l)

theorem nodupM_ok : ∀ (ls : List ℕ) (m : ℕ), nodupM ls m = true →
    ls.Nodup ∧ ∀ l ∈ ls, m.testBit l = false
  | [], _, _ => ⟨List.nodup_nil, by simp⟩
  | l :: ls, m, h => by
    simp only [nodupM, Bool.and_eq_true, Bool.not_eq_true'] at h
    obtain ⟨h1, h2⟩ := nodupM_ok ls _ h.2
    refine ⟨List.nodup_cons.2 ⟨fun hl => ?_, h1⟩, fun l' hl' => ?_⟩
    · have := h2 l hl
      rw [Nat.testBit_lor, Nat.testBit_two_pow] at this; simp at this
    · rcases List.mem_cons.1 hl' with rfl | hl'
      · exact h.1
      · have := h2 l' hl'
        rw [Nat.testBit_lor] at this; simp only [Bool.or_eq_false_iff] at this; exact this.1

/-- The checks on a network and its step labels: as many label lists as nodes, every node of
dimension index `0`, every leg a boundary or summed label, no summed label twice. -/
def netQ (N : Net) (H : List (List ℕ)) : Bool :=
  H.length == N.nodes.length && nodupM N.int 0 &&
    N.nodes.all fun n => n.dim == 0 && (n.outs ++ n.ins).all fun l =>
      (maskL N.int).testBit l || (maskL N.bnd).testBit l

/-- The final term: the steps run from the empty sum, then the labels no step sums. -/
def finQ (N : Net) (s : St) : Option (Option QTm) :=
  (sumLs (maskL N.int) (N.int.filter fun l => !s.1.testBit l) s).map Prod.snd

theorem labDim_two (E : Env) (hdim : E.dimv 0 = 2) (hL : E.labs.all (fun p => p.2 == 0) = true)
    (l : ℕ) : labDim E l = 2 := by
  unfold labDim
  cases h : E.labs.lookup l with
  | none => simpa using hdim
  | some j =>
    obtain ⟨l1, l2, he, -⟩ := List.lookup_eq_some_iff.1 h
    have := List.all_eq_true.1 hL (l, j) (by rw [he]; simp)
    simp only [beq_iff_eq] at this
    simpa [this] using hdim

theorem prodVal_reads (E : Env) (ns : List Node) : Reads (labelsOf ns) (prodVal E ns) := by
  intro τ τ' h
  unfold prodVal
  congr 1
  refine List.map_congr_left fun n hn => val_reads E n τ τ' fun x hx => h x ?_
  exact List.mem_flatMap.2 ⟨n, hn, hx⟩

/-- `σ` with every value above `1` replaced by `0`. -/
def trunc (σ : ℕ → ℕ) : ℕ → ℕ := fun l => if σ l < 2 then σ l else 0

theorem trunc_lt (σ : ℕ → ℕ) (l : ℕ) : trunc σ l < 2 := by
  unfold trunc; split_ifs with h <;> omega

theorem eval_q (E : Env) (hdim : E.dimv 0 = 2) (hph : ∀ ph j, E.phase ph j = 0)
    (hL : E.labs.all (fun p => p.2 == 0) = true) (N : Net) (H : List (List ℕ)) (s : St)
    (t : Option QTm) (hc : netQ N H = true)
    (hr : runQ (maskL N.int) (N.nodes.zip H) (0, some ⟨1, 0, [], []⟩) = some s)
    (hf : finQ N s = some t) (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = oval t (bits (trunc σ)) := by
  simp only [netQ, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, Bool.or_eq_true] at hc
  obtain ⟨⟨hlen, hnd⟩, hnodes⟩ := hc
  obtain ⟨hnd, -⟩ := nodupM_ok N.int 0 hnd
  have hld := labDim_two E hdim hL
  have hIM : ∀ l, (maskL N.int).testBit l = true → labDim E l = 2 := fun l _ => hld l
  have hfst : (N.nodes.zip H).map Prod.fst = N.nodes := List.map_fst_zip (by omega)
  have hok : NodesOK E ((N.nodes.zip H).map Prod.fst) := by
    rw [hfst]
    intro n hn
    have := (hnodes n hn).1
    rw [this]
    exact ⟨hdim, hph n.ph⟩
  have hI0 : Inv E (maskL N.int) [] [] (0, some ⟨1, 0, [], []⟩) := by
    refine ⟨fun l => by simp, List.nodup_nil, by simp, fun σ _ => ?_⟩
    simp [oval, QTm.val, dval, pval, sumOver, prodVal]
  obtain ⟨Ls, hI⟩ := runQ_ok E _ hIM _ [] [] _ _ hok hI0 hr
  rw [List.nil_append, hfst] at hI
  unfold finQ at hf
  cases hs : sumLs (maskL N.int) (N.int.filter fun l => !s.1.testBit l) s with
  | none => rw [hs] at hf; simp at hf
  | some s' =>
    rw [hs] at hf
    simp only [Option.map_some, Option.some.injEq] at hf
    subst hf
    obtain ⟨hm, hn, hi, hv⟩ := sumLs_ok E _ hIM N.nodes _ Ls s s' hI hs
    have hperm : N.int.Perm ((N.int.filter fun l => !s.1.testBit l).reverse ++ Ls) := by
      refine (List.perm_ext_iff_of_nodup hnd hn).2 fun l => ?_
      constructor
      · intro hl
        rw [List.mem_append, List.mem_reverse, List.mem_filter]
        by_cases hb : s.1.testBit l
        · exact Or.inr ((hI.1 l).1 hb)
        · exact Or.inl ⟨hl, by simpa using hb⟩
      · intro hl
        exact (maskL_bit N.int l).1 (hi l hl)
    have htr : eval E N σ = eval E N (trunc σ) := by
      unfold eval
      refine sumOver_ext (labDim E) _ _ _ (prodVal_reads E N.nodes) _ _ fun x hx hxi => ?_
      obtain ⟨n, hn', hxn⟩ := List.mem_flatMap.1 hx
      have hb := (hnodes n hn').2 x hxn
      rw [maskL_bit, maskL_bit] at hb
      rcases hb with hb | hb
      · exact absurd hb hxi
      · have := hσ x hb
        rw [hld x] at this
        simp [trunc, this]
    rw [htr, hv _ (trunc_lt σ)]
    unfold eval
    exact sumOver_perm (labDim E) hperm _ _

/-! ## Canonical forms -/

/-- A row's pivot: its top bit. -/
def piv (r : ℕ) : ℕ := Nat.log2 r

/-- `m` with each row of `R` added where `m` holds its pivot. -/
def redR (R : List ℕ) (m : ℕ) : ℕ := R.foldl (fun m r => if m.testBit (piv r) then m ^^^ r else m) m

theorem redR_par (ν : ℕ → ZMod 2) : ∀ (R : List ℕ) (m : ℕ), (∀ r ∈ R, par ν r = 0) →
    par ν (redR R m) = par ν m
  | [], _, _ => rfl
  | r :: R, m, h => by
    show par ν (redR R (if m.testBit (piv r) then m ^^^ r else m)) = par ν m
    rw [redR_par ν R _ (fun r' hr' => h r' (List.mem_cons_of_mem _ hr'))]
    split_ifs
    · rw [par_xor, h r List.mem_cons_self, add_zero]
    · rfl

/-- The rows `R` with the row `a` added, reduced; none when it reads `1 = 0`. -/
def insR (R : List ℕ) (a : ℕ) : Option (List ℕ) :=
  let a' := redR R a
  if a' = 0 then some R
  else if a' = 1 then none
  else some (a' :: R.map fun r => if r.testBit (piv a') then r ^^^ a' else r)

/-- The deltas in reduced echelon form; none when inconsistent. -/
def echR : List ℕ → Option (List ℕ)
  | [] => some []
  | a :: ds => (echR ds).bind fun R => insR R a

theorem insR_ok (ν : ℕ → ZMod 2) (R : List ℕ) (a : ℕ) :
    dval ν R * dlt (par ν a) = odv ν (insR R a) := by
  unfold insR
  simp only
  by_cases hz : ∀ r ∈ R, par ν r = 0
  · have hred := redR_par ν R a hz
    have h1 : dval ν R = 1 := by rw [dval_eq, if_pos hz]
    rw [h1, one_mul, ← hred]
    split_ifs with h0 h1'
    · rw [odv_some, h1, h0, par_zero]; simp [dlt]
    · rw [odv_none, h1', par_one]; simp [dlt]
    · rw [odv_some, dval_cons]
      have hm : dval ν (R.map fun r => if r.testBit (piv (redR R a)) then r ^^^ redR R a else r) =
          1 ∨ par ν (redR R a) ≠ 0 := by
        by_cases hp : par ν (redR R a) = 0
        · left
          rw [dval_eq, if_pos]
          intro d hd
          obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hd
          split_ifs
          · rw [par_xor, hz r hr, hp, add_zero]
          · exact hz r hr
        · exact Or.inr hp
      rcases hm with hm | hm
      · rw [hm]; ring
      · simp [dlt, hm]
  · have h0 : dval ν R = 0 := by rw [dval_eq, if_neg hz]
    rw [h0, zero_mul]
    split_ifs with ha0 ha1
    · rw [odv_some, h0]
    · rw [odv_none]
    · rw [odv_some, dval_eq, if_neg]
      intro hall
      apply hz
      intro r hr
      have hr' := hall _ (List.mem_cons_of_mem _ (List.mem_map_of_mem hr))
      have ha := hall _ List.mem_cons_self
      split_ifs at hr'
      · rw [par_xor, ha, add_zero] at hr'; exact hr'
      · exact hr'

theorem echR_ok (ν : ℕ → ZMod 2) : ∀ ds : List ℕ, dval ν ds = odv ν (echR ds)
  | [] => rfl
  | a :: ds => by
    have ih := echR_ok ν ds
    rw [dval_cons, mul_comm]
    show _ = odv ν ((echR ds).bind fun R => insR R a)
    cases h : echR ds with
    | none => rw [h] at ih; rw [ih]; simp
    | some R => rw [h] at ih; rw [ih, odv_some, Option.bind_some]; exact insR_ok ν R a

/-- A phase with its forms reduced by the rows `R` and ordered. -/
def redP (R : List ℕ) (p : ℕ × ℕ) : ℕ × ℕ :=
  let u := redR R p.1
  let w := redR R p.2
  if u ≤ w then (u, w) else (w, u)

/-- The reduced phases without those of value `1`, and the number of value `-1`. -/
def cleanP : List (ℕ × ℕ) → ℕ × List (ℕ × ℕ)
  | [] => (0, [])
  | p :: ps =>
    let r := cleanP ps
    if p.1 = 0 ∨ p.2 = 0 then r
    else if p.1 = 1 ∧ p.2 = 1 then (r.1 + 1, r.2)
    else (r.1, p :: r.2)

theorem cleanP_ok (ν : ℕ → ZMod 2) : ∀ ps : List (ℕ × ℕ),
    pval ν ps = (-1) ^ (cleanP ps).1 * pval ν (cleanP ps).2
  | [] => by simp [pval, cleanP]
  | p :: ps => by
    rw [pval_cons, cleanP_ok ν ps]
    simp only [cleanP]
    split_ifs with h0 h1
    · rcases h0 with h | h <;> simp [h, par_zero, sgn]
    · rw [h1.1, h1.2, par_one]; simp [sgn, pow_succ]
    · rw [pval_cons]; ring

/-- Lexicographic order on pairs. -/
def pLe (a b : ℕ × ℕ) : Bool := a.1 < b.1 || (a.1 == b.1 && a.2 ≤ b.2)

/-- The canonical term: deltas in reduced echelon form sorted, phases reduced by them,
cleaned and sorted. -/
def canonQ (t : QTm) : Option QTm :=
  (echR t.dels).map fun R =>
    let c := cleanP (t.phs.map (redP R))
    ⟨if c.1 % 2 = 0 then t.q else -t.q, t.k, R.insertionSort (· ≥ ·),
      c.2.insertionSort fun a b => pLe a b = true⟩

theorem redP_val (ν : ℕ → ZMod 2) (R : List ℕ) (hR : ∀ r ∈ R, par ν r = 0) (p : ℕ × ℕ) :
    par ν (redP R p).1 * par ν (redP R p).2 = par ν p.1 * par ν p.2 := by
  unfold redP
  simp only
  split_ifs
  · simp [redR_par ν R _ hR]
  · simp only [redR_par ν R _ hR]; ring


theorem canonQ_ok (ν : ℕ → ZMod 2) (t : QTm) : t.val ν = oval (canonQ t) ν := by
  unfold canonQ
  have he := echR_ok ν t.dels
  cases h : echR t.dels with
  | none => rw [h, odv_none] at he; simp [QTm.val, he]
  | some R =>
    rw [h, odv_some] at he
    rw [Option.map_some, oval_some]
    simp only [QTm.val]
    rw [dval_perm ν (List.perm_insertionSort _ R), pval_perm ν (List.perm_insertionSort _ _), he]
    by_cases hR : ∀ r ∈ R, par ν r = 0
    · have hp : pval ν t.phs = pval ν (t.phs.map (redP R)) := by
        unfold pval
        rw [List.map_map]
        congr 1
        exact List.map_congr_left fun p _ => by simp only [Function.comp_def, redP_val ν R hR p]
      rw [hp, cleanP_ok ν (t.phs.map (redP R))]
      rcases Nat.even_or_odd (cleanP (t.phs.map (redP R))).1 with he' | he'
      · rw [he'.neg_one_pow, if_pos (Nat.even_iff.1 he')]; ring
      · rw [he'.neg_one_pow, if_neg (by have := Nat.odd_iff.1 he'; omega)]; push_cast; ring
    · have h0 : dval ν R = 0 := by rw [dval_eq, if_neg hR]
      simp [h0]

/-! ## The equality -/

/-- `A0 √2^e = A1` for an even `e`, in integers. -/
def scalEq (A0 A1 e : ℤ) : Bool :=
  e % 2 == 0 && if 0 ≤ e then A0 * 2 ^ (e / 2).toNat == A1 else A0 == A1 * 2 ^ (-e / 2).toNat

theorem rt2_two_pow (m : ℕ) : rt 2 ^ (2 * (m : ℤ)) = 2 ^ m := by
  rw [zpow_mul]; norm_num [rt_sq']

theorem scalEq_ok (A0 A1 e : ℤ) (h : scalEq A0 A1 e = true) : (A0 : ℂ) * rt 2 ^ e = A1 := by
  unfold scalEq at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨he, hq⟩ := h
  split_ifs at hq with hpos
  · obtain ⟨k, hk⟩ : ∃ k : ℕ, e = 2 * (k : ℤ) := ⟨(e / 2).toNat, by omega⟩
    have hk' : (e / 2).toNat = k := by omega
    rw [hk', beq_iff_eq] at hq
    rw [hk, rt2_two_pow]
    exact_mod_cast hq
  · obtain ⟨k, hk⟩ : ∃ k : ℕ, e = -(2 * (k : ℤ)) := ⟨(-e / 2).toNat, by omega⟩
    have hk' : (-e / 2).toNat = k := by omega
    rw [hk', beq_iff_eq] at hq
    have hq' : (A0 : ℂ) = A1 * 2 ^ k := by exact_mod_cast hq
    rw [hk, zpow_neg, rt2_two_pow, hq']
    field_simp

/-- The two final terms are equal after their scalars `a / b √2^K`: both zero, or the same
canonical factors and equal coefficients. -/
def cmpQ (a0 : ℤ) (b0 : ℕ) (K0 : ℤ) (a1 : ℤ) (b1 : ℕ) (K1 : ℤ) (t0 t1 : Option QTm) : Bool :=
  b0 != 0 && b1 != 0 &&
  match t0.bind canonQ, t1.bind canonQ with
  | none, none => true
  | some u0, some u1 =>
    u0.dels == u1.dels && u0.phs == u1.phs &&
      scalEq (a0 * u0.q * b1) (a1 * u1.q * b0) ((K0 + u0.k) - (K1 + u1.k))
  | _, _ => false

theorem oval_bind_canon (t : Option QTm) (ν : ℕ → ZMod 2) : oval (t.bind canonQ) ν = oval t ν := by
  cases t with
  | none => rfl
  | some t => rw [Option.bind_some, oval_some]; exact (canonQ_ok ν t).symm

theorem cmpQ_ok (a0 : ℤ) (b0 : ℕ) (K0 : ℤ) (a1 : ℤ) (b1 : ℕ) (K1 : ℤ) (t0 t1 : Option QTm)
    (h : cmpQ a0 b0 K0 a1 b1 K1 t0 t1 = true) (ν : ℕ → ZMod 2) :
    (a0 : ℂ) / b0 * rt 2 ^ K0 * oval t0 ν = (a1 : ℂ) / b1 * rt 2 ^ K1 * oval t1 ν := by
  rw [← oval_bind_canon t0, ← oval_bind_canon t1]
  unfold cmpQ at h
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at h
  obtain ⟨⟨hb0, hb1⟩, h⟩ := h
  have hb0' : (b0 : ℂ) ≠ 0 := by exact_mod_cast hb0
  have hb1' : (b1 : ℂ) ≠ 0 := by exact_mod_cast hb1
  split at h
  · next h0 h1 => rw [h0, h1]; simp
  · next u0 u1 h0 h1 =>
    rw [h0, h1, oval_some, oval_some]
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    have hd : u0.dels = u1.dels := h.1.1
    have hp : u0.phs = u1.phs := h.1.2
    have key := scalEq_ok _ _ _ h.2
    push_cast at key
    have hz : rt 2 ^ (K0 + u0.k - (K1 + u1.k)) * (rt 2 ^ K1 * rt 2 ^ u1.k) =
        rt 2 ^ K0 * rt 2 ^ u0.k := by
      rw [← zpow_add₀ rt2_ne, ← zpow_add₀ rt2_ne, ← zpow_add₀ rt2_ne]; congr 1; ring
    unfold QTm.val
    rw [hd, hp]
    have e1 : (a0 : ℂ) / b0 * rt 2 ^ K0 * (u0.q * rt 2 ^ u0.k * dval ν u1.dels * pval ν u1.phs) =
        ((a0 : ℂ) * u0.q * b1 * rt 2 ^ (K0 + u0.k - (K1 + u1.k))) *
          (rt 2 ^ K1 * rt 2 ^ u1.k * dval ν u1.dels * pval ν u1.phs) / (b0 * b1) := by
      rw [show ((a0 : ℂ) * u0.q * b1 * rt 2 ^ (K0 + u0.k - (K1 + u1.k))) *
          (rt 2 ^ K1 * rt 2 ^ u1.k * dval ν u1.dels * pval ν u1.phs) =
          (a0 : ℂ) * u0.q * b1 * (rt 2 ^ (K0 + u0.k - (K1 + u1.k)) * (rt 2 ^ K1 * rt 2 ^ u1.k)) *
          (dval ν u1.dels * pval ν u1.phs) by ring, hz]
      field_simp
    rw [e1, key]
    field_simp
  · simp at h

/-- The check `equal_q` reads: both networks run to their final terms, equal after the scalars. -/
def qCheck (N0 N1 : Net) (H0 H1 : List (List ℕ)) (s0 s1 : St) (a0 : ℤ) (b0 : ℕ) (K0 : ℤ)
    (a1 : ℤ) (b1 : ℕ) (K1 : ℤ) : Bool :=
  N0.bnd.isPerm N1.bnd && netQ N0 H0 && netQ N1 H1 &&
    match finQ N0 s0, finQ N1 s1 with
    | some t0, some t1 => cmpQ a0 b0 K0 a1 b1 K1 t0 t1
    | _, _ => false

/-- The empty sum. -/
def st0 : St := (0, some ⟨1, 0, [], []⟩)

theorem equal_q (E : Env) (hdim : E.dimv 0 = 2) (hph : ∀ ph j, E.phase ph j = 0)
    (hL : E.labs.all (fun p => p.2 == 0) = true) (N0 N1 : Net) (H0 H1 : List (List ℕ))
    (s0 s1 : St) (a0 : ℤ) (b0 : ℕ) (K0 : ℤ) (a1 : ℤ) (b1 : ℕ) (K1 : ℤ)
    (hr0 : runQ (maskL N0.int) (N0.nodes.zip H0) st0 = some s0)
    (hr1 : runQ (maskL N1.int) (N1.nodes.zip H1) st0 = some s1)
    (h : qCheck N0 N1 H0 H1 s0 s1 a0 b0 K0 a1 b1 K1 = true) :
    Equal E ((a0 : ℂ) / b0 * rt 2 ^ K0) N0 ((a1 : ℂ) / b1 * rt 2 ^ K1) N1 := by
  unfold qCheck at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨hp, hc0⟩, hc1⟩, h⟩ := h
  split at h
  · next t0 t1 hf0 hf1 =>
    refine ⟨List.isPerm_iff.1 hp, fun σ hσ => ?_⟩
    have hσ1 : InRange E N1.bnd σ := hσ.perm (List.isPerm_iff.1 hp)
    rw [eval_q E hdim hph hL N0 H0 s0 t0 hc0 hr0 hf0 σ hσ,
      eval_q E hdim hph hL N1 H1 s1 t1 hc1 hr1 hf1 σ hσ1]
    exact cmpQ_ok a0 b0 K0 a1 b1 K1 t0 t1 h _
  · simp at h

/-- Both runs and `qCheck` on their final states. -/
def qAll (N0 N1 : Net) (H0 H1 : List (List ℕ)) (a0 : ℤ) (b0 : ℕ) (K0 : ℤ) (a1 : ℤ) (b1 : ℕ)
    (K1 : ℤ) : Bool :=
  match runQ (maskL N0.int) (N0.nodes.zip H0) st0, runQ (maskL N1.int) (N1.nodes.zip H1) st0 with
  | some s0, some s1 => qCheck N0 N1 H0 H1 s0 s1 a0 b0 K0 a1 b1 K1
  | _, _ => false

theorem equal_qAll (E : Env) (hdim : E.dimv 0 = 2) (hph : ∀ ph j, E.phase ph j = 0)
    (hL : E.labs.all (fun p => p.2 == 0) = true) (N0 N1 : Net) (H0 H1 : List (List ℕ))
    (a0 : ℤ) (b0 : ℕ) (K0 : ℤ) (a1 : ℤ) (b1 : ℕ) (K1 : ℤ)
    (h : qAll N0 N1 H0 H1 a0 b0 K0 a1 b1 K1 = true) :
    Equal E ((a0 : ℂ) / b0 * rt 2 ^ K0) N0 ((a1 : ℂ) / b1 * rt 2 ^ K1) N1 := by
  unfold qAll at h
  split at h
  · next s0 s1 h0 h1 => exact equal_q E hdim hph hL N0 N1 H0 H1 s0 s1 a0 b0 K0 a1 b1 K1 h0 h1 h
  · simp at h

end SymF2
end ArchytasZX
