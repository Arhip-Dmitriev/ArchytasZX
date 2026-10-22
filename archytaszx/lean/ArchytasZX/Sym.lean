/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Closed

/-!
# Symbolic sums

A term `STm` is a coefficient times nested sums over bound index variables of a product of
Kronecker deltas `[M ∣ p]` and roots of unity `ω_D^p`; `p` is an integer polynomial in index
variables, `M` and `D` are monomials in dimension symbols. An expression is a list of terms,
denoting their sum under a valuation of the dimension symbols, coefficient atoms and index
variables.
-/

namespace ArchytasZX.Sym

open Complex Finset

/-! ## Valuations -/

/-- Dimension symbol values and coefficient atom values. -/
structure Val where
  dv : ℕ → ℕ
  atom : ℕ → ℂ

/-! ## Dense exponent vectors -/

/-- `Π_i f (k + i) ^ v_i`. -/
def npow (f : ℕ → ℕ) : ℕ → List ℕ → ℕ
  | _, [] => 1
  | k, a :: v => f k ^ a * npow f (k + 1) v

/-- `Π_i f (k + i) ^ v_i` with integer exponents. -/
noncomputable def zpw (f : ℕ → ℂ) : ℕ → List ℤ → ℂ
  | _, [] => 1
  | k, a :: v => f k ^ a * zpw f (k + 1) v

/-- Pointwise sum, the shorter vector padded with zeros. -/
def nadd : List ℕ → List ℕ → List ℕ
  | [], b => b
  | a, [] => a
  | x :: a, y :: b => (x + y) :: nadd a b

/-- Pointwise sum, the shorter vector padded with zeros. -/
def zadd : List ℤ → List ℤ → List ℤ
  | [], b => b
  | a, [] => a
  | x :: a, y :: b => (x + y) :: zadd a b

/-- Pointwise truncated difference. -/
def nsub : List ℕ → List ℕ → List ℕ
  | [], _ => []
  | a, [] => a
  | x :: a, y :: b => (x - y) :: nsub a b

/-- Pointwise `≤`, missing entries zero. -/
def nle : List ℕ → List ℕ → Bool
  | [], _ => true
  | x :: a, [] => x == 0 && nle a []
  | x :: a, y :: b => decide (x ≤ y) && nle a b

/-- Without trailing zeros. -/
def ntrim : List ℕ → List ℕ
  | [] => []
  | x :: a => match ntrim a with
    | [] => if x = 0 then [] else [x]
    | t => x :: t

/-- Without trailing zeros. -/
def ztrim : List ℤ → List ℤ
  | [] => []
  | x :: a => match ztrim a with
    | [] => if x = 0 then [] else [x]
    | t => x :: t

/-- Scaled by `c`. -/
def nscale (c : ℕ) (v : List ℕ) : List ℤ := v.map fun x => (c : ℤ) * x

theorem npow_nadd (f : ℕ → ℕ) (a b : List ℕ) (k : ℕ) :
    npow f k (nadd a b) = npow f k a * npow f k b := by
  induction a generalizing b k with
  | nil => simp [nadd, npow]
  | cons x a ih =>
    cases b with
    | nil => simp [nadd, npow]
    | cons y b => simp only [nadd, npow, ih, pow_add]; ring

theorem zpw_zadd (f : ℕ → ℂ) (hf : ∀ i, f i ≠ 0) (a b : List ℤ) (k : ℕ) :
    zpw f k (zadd a b) = zpw f k a * zpw f k b := by
  induction a generalizing b k with
  | nil => simp [zadd, zpw]
  | cons x a ih =>
    cases b with
    | nil => simp [zadd, zpw]
    | cons y b => simp only [zadd, zpw, ih, zpow_add₀ (hf k)]; ring

theorem npow_ntrim (f : ℕ → ℕ) (a : List ℕ) (k : ℕ) : npow f k (ntrim a) = npow f k a := by
  induction a generalizing k with
  | nil => rfl
  | cons x a ih =>
    have h := ih (k + 1)
    simp only [ntrim]
    split
    · next heq =>
      rw [heq] at h
      split_ifs with hx
      · subst hx; simp [npow, ← h]
      · simp [npow, ← h]
    · next t ht => simp only [npow]; rw [← h]

theorem zpw_ztrim (f : ℕ → ℂ) (a : List ℤ) (k : ℕ) : zpw f k (ztrim a) = zpw f k a := by
  induction a generalizing k with
  | nil => rfl
  | cons x a ih =>
    have h := ih (k + 1)
    simp only [ztrim]
    split
    · next heq =>
      rw [heq] at h
      split_ifs with hx
      · subst hx; simp [zpw, ← h]
      · simp [zpw, ← h]
    · next t ht => simp only [zpw]; rw [← h]

theorem npow_pos (f : ℕ → ℕ) (hf : ∀ i, 0 < f i) (a : List ℕ) (k : ℕ) : 0 < npow f k a := by
  induction a generalizing k with
  | nil => simp [npow]
  | cons x a ih => exact Nat.mul_pos (pow_pos (hf k) _) (ih _)

theorem npow_mono (f g : ℕ → ℕ) (h : ∀ i, f i ≤ g i) (a : List ℕ) (k : ℕ) :
    npow f k a ≤ npow g k a := by
  induction a generalizing k with
  | nil => simp [npow]
  | cons x a ih => exact Nat.mul_le_mul (Nat.pow_le_pow_left (h k) _) (ih _)

theorem npow_of_nle (f : ℕ → ℕ) (b : List ℕ) (k : ℕ) (h : nle b [] = true) :
    npow f k b = 1 := by
  induction b generalizing k with
  | nil => rfl
  | cons y b ih =>
    simp only [nle, Bool.and_eq_true, beq_iff_eq] at h
    simp [npow, h.1, ih _ h.2]

theorem npow_nsub (f : ℕ → ℕ) (a b : List ℕ) (k : ℕ) (h : nle b a = true) :
    npow f k a = npow f k b * npow f k (nsub a b) := by
  induction a generalizing b k with
  | nil =>
    cases b with
    | nil => simp [npow, nsub]
    | cons y b => rw [npow_of_nle f _ k h]; simp [npow, nsub]
  | cons x a ih =>
    cases b with
    | nil => simp [npow, nsub]
    | cons y b =>
      simp only [nle, Bool.and_eq_true, decide_eq_true_eq] at h
      simp only [npow, nsub, ih b (k + 1) h.2]
      rw [← Nat.add_sub_cancel' h.1, pow_add, Nat.add_sub_cancel_left]; ring

theorem npow_cast (f : ℕ → ℕ) (a : List ℕ) (k : ℕ) :
    ((npow f k a : ℕ) : ℂ) = zpw (fun i => (f i : ℂ)) k (a.map fun x => (x : ℤ)) := by
  induction a generalizing k with
  | nil => simp [npow, zpw]
  | cons x a ih => simp [npow, zpw, ih]

/-! ## Dimension monomials -/

/-- `c · Π_i d_i ^ e_i`. -/
structure DMon where
  c : ℕ
  e : List ℕ
  deriving DecidableEq, Repr

namespace DMon

/-- The value. -/
def den (dv : ℕ → ℕ) (m : DMon) : ℕ := m.c * npow dv 0 m.e

/-- The product. -/
def mul (a b : DMon) : DMon := ⟨a.c * b.c, ntrim (nadd a.e b.e)⟩

/-- Whether `a` divides `b` for every valuation. -/
def dvd (a b : DMon) : Bool := a.c != 0 && b.c % a.c == 0 && nle a.e b.e

/-- The quotient `b / a`. -/
def div (b a : DMon) : DMon := ⟨b.c / a.c, ntrim (nsub b.e a.e)⟩

/-- The value with every symbol one. -/
def den1 (m : DMon) : ℕ := m.c

/-- Whether the monomial is a constant. -/
def isConst (m : DMon) : Bool := m.e.all (· == 0)

theorem den_mul (dv : ℕ → ℕ) (a b : DMon) : (mul a b).den dv = a.den dv * b.den dv := by
  simp only [den, mul, npow_ntrim, npow_nadd]; ring

theorem den_pos (dv : ℕ → ℕ) (hdv : ∀ i, 0 < dv i) (m : DMon) (hc : 0 < m.c) :
    0 < m.den dv := Nat.mul_pos hc (npow_pos dv hdv _ _)

theorem den_div (dv : ℕ → ℕ) (a b : DMon) (h : dvd a b = true) :
    b.den dv = a.den dv * (div b a).den dv := by
  simp only [dvd, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at h
  obtain ⟨⟨_, hc⟩, he⟩ := h
  simp only [den, div, npow_ntrim]
  have hd : a.c ∣ b.c := Nat.dvd_of_mod_eq_zero hc
  rw [npow_nsub dv b.e a.e 0 he]
  conv_lhs => rw [← Nat.mul_div_cancel' hd]
  ring

theorem dvd_den (dv : ℕ → ℕ) (a b : DMon) (h : dvd a b = true) : a.den dv ∣ b.den dv :=
  ⟨_, den_div dv a b h⟩

theorem den_const (dv dv' : ℕ → ℕ) (m : DMon) (h : m.isConst = true) : m.den dv = m.den dv' := by
  obtain ⟨c, e⟩ := m
  simp only [isConst, den] at h ⊢
  congr 1
  generalize 0 = k
  induction e generalizing k with
  | nil => rfl
  | cons x e ih =>
    simp only [List.all_cons, Bool.and_eq_true, beq_iff_eq] at h
    simp [npow, h.1, ih h.2]

theorem den_mono (f g : ℕ → ℕ) (h : ∀ i, f i ≤ g i) (m : DMon) : m.den f ≤ m.den g :=
  Nat.mul_le_mul_left _ (npow_mono f g h _ _)

end DMon

/-! ## Index polynomials -/

/-- A monomial in index variables, as a sorted list. -/
abbrev IMon := List ℕ

/-- An integer polynomial in index variables: sorted monomials, nonzero coefficients. -/
abbrev IPoly := List (IMon × ℤ)

/-- The value of a monomial. -/
def imDen (ν : ℕ → ℤ) (m : IMon) : ℤ := (m.map ν).prod

/-- The value of a polynomial. -/
def ipDen (ν : ℕ → ℤ) (p : IPoly) : ℤ := (p.map fun t => t.2 * imDen ν t.1).sum

/-- Insertion keeping a monomial sorted. -/
def imIns (x : ℕ) : IMon → IMon
  | [] => [x]
  | y :: m => if x ≤ y then x :: y :: m else y :: imIns x m

/-- The product of monomials. -/
def imMul (a b : IMon) : IMon := b.foldr imIns a

/-- Lexicographic order on monomials. -/
def lexLt : List ℕ → List ℕ → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | x :: a, y :: b => decide (x < y) || (x == y && lexLt a b)

/-- `c · m` added to a polynomial. -/
def ipIns (m : IMon) (c : ℤ) : IPoly → IPoly
  | [] => if c = 0 then [] else [(m, c)]
  | (m', c') :: p =>
    if m = m' then (if c + c' = 0 then p else (m, c + c') :: p)
    else if lexLt m m' then (if c = 0 then (m', c') :: p else (m, c) :: (m', c') :: p)
    else (m', c') :: ipIns m c p

/-- The sum. -/
def ipAdd (p q : IPoly) : IPoly := q.foldr (fun t acc => ipIns t.1 t.2 acc) p

/-- `k · p`. -/
def ipSmul (k : ℤ) (p : IPoly) : IPoly := if k = 0 then [] else p.map fun t => (t.1, k * t.2)

/-- The product. -/
def ipMul (p q : IPoly) : IPoly :=
  p.foldr (fun t acc => q.foldr (fun s acc' => ipIns (imMul t.1 s.1) (t.2 * s.2) acc') acc) []

/-- A constant. -/
def ipC (c : ℤ) : IPoly := ipIns [] c []

/-- A variable. -/
def ipV (x : ℕ) : IPoly := [([x], 1)]

/-- The difference. -/
def ipSub (p q : IPoly) : IPoly := ipAdd p (ipSmul (-1) q)

/-- The `k`-th power. -/
def ipPow (r : IPoly) : ℕ → IPoly
  | 0 => ipC 1
  | k + 1 => ipMul r (ipPow r k)

/-- `x` replaced by `r`, term by term. -/
def ipSubstF (x : ℕ) (r : IPoly) (p : IPoly) : IPoly :=
  p.foldr (fun t acc => ipAdd acc
    (ipMul (ipSmul t.2 [(t.1.filter (· != x), 1)]) (ipPow r (t.1.count x)))) []

/-- Whether `x` occurs. -/
def ipHas (x : ℕ) (p : IPoly) : Bool := p.any fun t => t.1.contains x

/-- `x` replaced by `r`; `p` itself when `x` does not occur. -/
def ipSubst (x : ℕ) (r : IPoly) (p : IPoly) : IPoly := if ipHas x p then ipSubstF x r p else p

/-- `p = a · x + b` with `a`, `b` free of `x`, when `x` occurs at most linearly. -/
def ipLin (x : ℕ) (p : IPoly) : Option (IPoly × IPoly) :=
  if p.all (fun t => decide (t.1.count x ≤ 1)) then
    some (p.foldr (fun t acc => if t.1.contains x then ipIns (t.1.erase x) t.2 acc else acc) [],
      p.foldr (fun t acc => if t.1.contains x then acc else ipIns t.1 t.2 acc) [])
  else none

/-- The constant value, if constant. -/
def ipConst : IPoly → Option ℤ
  | [] => some 0
  | [([], c)] => some c
  | _ => none

/-- Coefficients reduced modulo `k`. -/
def ipModC (k : ℕ) (p : IPoly) : IPoly := p.foldr (fun t acc => ipIns t.1 (t.2 % (k : ℤ)) acc) []

/-- Every coefficient a multiple of `k`. -/
def ipDivBy (k : ℤ) (p : IPoly) : Bool := p.all fun t => t.2 % k == 0

/-- The polynomial with a nonnegative leading coefficient. -/
def ipSign (p : IPoly) : IPoly :=
  match p with
  | (_, c) :: _ => if c < 0 then ipSmul (-1) p else p
  | [] => []

theorem imDen_cons (ν : ℕ → ℤ) (x : ℕ) (m : IMon) : imDen ν (x :: m) = ν x * imDen ν m := by
  simp [imDen]

theorem imDen_ins (ν : ℕ → ℤ) (x : ℕ) (m : IMon) : imDen ν (imIns x m) = ν x * imDen ν m := by
  induction m with
  | nil => simp [imIns, imDen]
  | cons y m ih =>
    simp only [imIns]
    split_ifs
    · rfl
    · simp only [imDen_cons, ih]; ring

theorem imDen_mul (ν : ℕ → ℤ) (a b : IMon) : imDen ν (imMul a b) = imDen ν a * imDen ν b := by
  induction b with
  | nil => simp [imMul, imDen]
  | cons y b ih =>
    simp only [imMul, List.foldr_cons] at ih ⊢
    rw [imDen_ins, ih, imDen_cons]; ring

theorem ipDen_nil (ν : ℕ → ℤ) : ipDen ν [] = 0 := rfl

theorem ipDen_cons (ν : ℕ → ℤ) (m : IMon) (c : ℤ) (p : IPoly) :
    ipDen ν ((m, c) :: p) = c * imDen ν m + ipDen ν p := by simp [ipDen]

theorem ipDen_ins (ν : ℕ → ℤ) (m : IMon) (c : ℤ) (p : IPoly) :
    ipDen ν (ipIns m c p) = c * imDen ν m + ipDen ν p := by
  induction p with
  | nil => simp only [ipIns]; split_ifs with h <;> simp [ipDen, h]
  | cons t p ih =>
    obtain ⟨m', c'⟩ := t
    simp only [ipIns]
    split_ifs with h1 h2 h3 h4
    · subst h1; rw [ipDen_cons]
      have : c' = -c := by omega
      subst this; ring
    · subst h1; rw [ipDen_cons, ipDen_cons]; ring
    · subst h4; simp [ipDen_cons]
    · rw [ipDen_cons, ipDen_cons]
    · rw [ipDen_cons, ipDen_cons, ih]; ring

theorem ipDen_add (ν : ℕ → ℤ) (p q : IPoly) : ipDen ν (ipAdd p q) = ipDen ν p + ipDen ν q := by
  induction q with
  | nil => simp [ipAdd, ipDen]
  | cons t q ih =>
    simp only [ipAdd, List.foldr_cons] at ih ⊢
    rw [ipDen_ins, ih, ipDen_cons]; ring

theorem ipDen_smul (ν : ℕ → ℤ) (k : ℤ) (p : IPoly) : ipDen ν (ipSmul k p) = k * ipDen ν p := by
  unfold ipSmul
  split_ifs with h
  · simp [h, ipDen]
  · induction p with
    | nil => simp [ipDen]
    | cons t p ih =>
      obtain ⟨m, c⟩ := t
      simp only [List.map_cons, ipDen_cons] at ih ⊢; rw [ih]; ring

theorem ipDen_mul (ν : ℕ → ℤ) (p q : IPoly) : ipDen ν (ipMul p q) = ipDen ν p * ipDen ν q := by
  have inner : ∀ (t : IMon × ℤ) (acc : IPoly), ipDen ν (q.foldr (fun s acc' =>
      ipIns (imMul t.1 s.1) (t.2 * s.2) acc') acc) = t.2 * imDen ν t.1 * ipDen ν q +
      ipDen ν acc := by
    intro t acc
    induction q with
    | nil => simp [ipDen]
    | cons s q ih => simp only [List.foldr_cons]; rw [ipDen_ins, ih, imDen_mul, ipDen_cons]; ring
  induction p with
  | nil => simp [ipMul, ipDen]
  | cons t p ih =>
    simp only [ipMul, List.foldr_cons] at ih ⊢
    rw [inner, ih, ipDen_cons]; ring

theorem ipDen_C (ν : ℕ → ℤ) (c : ℤ) : ipDen ν (ipC c) = c := by
  simp [ipC, ipDen_ins, ipDen_nil, imDen]

theorem ipDen_V (ν : ℕ → ℤ) (x : ℕ) : ipDen ν (ipV x) = ν x := by simp [ipV, ipDen, imDen]

theorem ipDen_sub (ν : ℕ → ℤ) (p q : IPoly) : ipDen ν (ipSub p q) = ipDen ν p - ipDen ν q := by
  simp only [ipSub, ipDen_add, ipDen_smul]; ring

theorem ipDen_pow (ν : ℕ → ℤ) (r : IPoly) (k : ℕ) : ipDen ν (ipPow r k) = ipDen ν r ^ k := by
  induction k with
  | zero => simp [ipPow, ipDen_C]
  | succ k ih => simp only [ipPow, ipDen_mul, ih]; ring

theorem imDen_upd (ν : ℕ → ℤ) (x : ℕ) (v : ℤ) (m : IMon) :
    imDen (Function.update ν x v) m = v ^ m.count x * imDen ν (m.filter (· != x)) := by
  induction m with
  | nil => simp [imDen]
  | cons y m ih =>
    by_cases h : y = x
    · subst h; simp only [imDen_cons, ih, Function.update_self, List.count_cons_self,
        List.filter_cons, bne_self_eq_false, Bool.false_eq_true, ite_false]; ring
    · have h' : (y != x) = true := by simpa using h
      simp only [imDen_cons, ih, Function.update_of_ne h, List.filter_cons, h', ite_true,
        List.count_cons, beq_iff_eq, h, ite_false, Nat.add_zero]; ring

theorem ipDen_substF (ν : ℕ → ℤ) (x : ℕ) (r p : IPoly) :
    ipDen ν (ipSubstF x r p) = ipDen (Function.update ν x (ipDen ν r)) p := by
  induction p with
  | nil => simp [ipSubstF, ipDen]
  | cons t p ih =>
    simp only [ipSubstF, List.foldr_cons] at ih ⊢
    rw [ipDen_add, ih, ipDen_mul, ipDen_smul, ipDen_pow, ipDen_cons, ipDen_cons, ipDen_nil,
      imDen_upd]
    ring

theorem imDen_upd_of (ν : ℕ → ℤ) (x : ℕ) (v : ℤ) (m : IMon) (h : m.contains x = false) :
    imDen (Function.update ν x v) m = imDen ν m := by
  induction m with
  | nil => rfl
  | cons y m ih =>
    simp only [List.contains_cons, Bool.or_eq_false_iff, beq_eq_false_iff_ne] at h
    rw [imDen_cons, imDen_cons, ih h.2, Function.update_of_ne (Ne.symm h.1)]

theorem ipDen_upd_of (ν : ℕ → ℤ) (x : ℕ) (v : ℤ) (p : IPoly) (h : ipHas x p = false) :
    ipDen (Function.update ν x v) p = ipDen ν p := by
  induction p with
  | nil => rfl
  | cons t p ih =>
    simp only [ipHas, List.any_cons, Bool.or_eq_false_iff] at h ih
    rw [ipDen_cons, ipDen_cons, ih h.2, imDen_upd_of ν x v _ h.1]

theorem ipDen_subst (ν : ℕ → ℤ) (x : ℕ) (r p : IPoly) :
    ipDen ν (ipSubst x r p) = ipDen (Function.update ν x (ipDen ν r)) p := by
  unfold ipSubst
  split_ifs with h
  · exact ipDen_substF ν x r p
  · exact (ipDen_upd_of ν x _ p (by simpa using h)).symm

theorem mem_ipIns (m : IMon) (c : ℤ) (p : IPoly) (t : IMon × ℤ) (ht : t ∈ ipIns m c p) :
    t.1 = m ∨ t ∈ p := by
  induction p with
  | nil => simp only [ipIns] at ht; split_ifs at ht <;> simp_all
  | cons s p ih =>
    obtain ⟨m', c'⟩ := s
    simp only [ipIns] at ht
    split_ifs at ht <;> (try simp only [List.mem_cons] at ht)
    · exact Or.inr (List.mem_cons_of_mem _ ht)
    · rcases ht with rfl | ht
      · exact Or.inl rfl
      · exact Or.inr (List.mem_cons_of_mem _ ht)
    · rcases ht with rfl | ht
      · exact Or.inr List.mem_cons_self
      · exact Or.inr (List.mem_cons_of_mem _ ht)
    · rcases ht with rfl | rfl | ht
      · exact Or.inl rfl
      · exact Or.inr List.mem_cons_self
      · exact Or.inr (List.mem_cons_of_mem _ ht)
    · rcases ht with rfl | ht
      · exact Or.inr List.mem_cons_self
      · rcases ih ht with h | h
        · exact Or.inl h
        · exact Or.inr (List.mem_cons_of_mem _ h)

theorem ipHas_iff (x : ℕ) (p : IPoly) : ipHas x p = false ↔ ∀ t ∈ p, t.1.contains x = false := by
  simp [ipHas]

theorem ipHas_ins (x : ℕ) (m : IMon) (c : ℤ) (p : IPoly) (hm : m.contains x = false)
    (hp : ipHas x p = false) : ipHas x (ipIns m c p) = false := by
  rw [ipHas_iff] at hp ⊢
  intro t ht
  rcases mem_ipIns m c p t ht with h | h
  · rw [h]; exact hm
  · exact hp t h

theorem imDen_erase (ν : ℕ → ℤ) (x : ℕ) (m : IMon) (h : m.contains x = true) :
    imDen ν m = ν x * imDen ν (m.erase x) := by
  induction m with
  | nil => simp at h
  | cons y m ih =>
    by_cases hy : y = x
    · subst hy; simp [imDen_cons]
    · simp only [List.contains_cons, Bool.or_eq_true, beq_iff_eq, Ne.symm hy, false_or] at h
      rw [List.erase_cons_tail (by simpa using hy), imDen_cons, imDen_cons, ih h]; ring

theorem count_erase_zero (x : ℕ) (m : IMon) (h : m.count x ≤ 1) :
    (m.erase x).contains x = false := by
  by_cases hx : x ∈ m
  · have h1 := List.count_erase_self (a := x) (l := m)
    have : (m.erase x).count x = 0 := by omega
    simpa using List.count_eq_zero.1 this
  · rw [List.erase_of_not_mem hx]; simpa using hx

theorem ipLin_spec (x : ℕ) (p a b : IPoly) (h : ipLin x p = some (a, b)) :
    ipHas x a = false ∧ ipHas x b = false ∧ ∀ ν, ipDen ν p = ipDen ν a * ν x + ipDen ν b := by
  unfold ipLin at h
  split_ifs at h with hall
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  rw [List.all_eq_true] at hall
  induction p with
  | nil => simp [ipHas, ipDen]
  | cons t p ih =>
    have ht := hall t List.mem_cons_self
    simp only [decide_eq_true_eq] at ht
    obtain ⟨ih1, ih2, ih3⟩ := ih fun s hs => hall s (List.mem_cons_of_mem _ hs)
    simp only [List.foldr_cons]
    by_cases hc : t.1.contains x = true
    · simp only [hc, ite_true]
      refine ⟨ipHas_ins _ _ _ _ (count_erase_zero x t.1 ht) ih1, ih2, fun ν => ?_⟩
      rw [ipDen_ins, ipDen_cons, ih3, imDen_erase ν x t.1 hc]; ring
    · simp only [Bool.not_eq_true] at hc
      simp only [hc, Bool.false_eq_true, ite_false]
      refine ⟨ih1, ipHas_ins _ _ _ _ hc ih2, fun ν => ?_⟩
      rw [ipDen_ins, ipDen_cons, ih3]; ring

theorem ipConst_spec (p : IPoly) (c : ℤ) (h : ipConst p = some c) (ν : ℕ → ℤ) :
    ipDen ν p = c := by
  match p, h with
  | [], h => simp [ipConst] at h; simp [ipDen, h]
  | [([], c')], h => simp [ipConst] at h; simp [ipDen, imDen, h]

theorem ipDen_modC (ν : ℕ → ℤ) (k : ℕ) (p : IPoly) :
    (k : ℤ) ∣ ipDen ν p - ipDen ν (ipModC k p) := by
  induction p with
  | nil => simp [ipModC, ipDen]
  | cons t p ih =>
    simp only [ipModC, List.foldr_cons] at ih ⊢
    rw [ipDen_ins, ipDen_cons]
    have : t.2 * imDen ν t.1 + ipDen ν p - (t.2 % (k : ℤ) * imDen ν t.1 +
        ipDen ν (List.foldr (fun t acc => ipIns t.1 (t.2 % (k : ℤ)) acc) [] p)) =
        (t.2 - t.2 % (k : ℤ)) * imDen ν t.1 + (ipDen ν p -
        ipDen ν (List.foldr (fun t acc => ipIns t.1 (t.2 % (k : ℤ)) acc) [] p)) := by ring
    rw [this]
    refine dvd_add (Dvd.dvd.mul_right ?_ _) ih
    rw [Int.emod_def]; simp

theorem ipDen_sign (ν : ℕ → ℤ) (p : IPoly) :
    ipDen ν (ipSign p) = ipDen ν p ∨ ipDen ν (ipSign p) = -ipDen ν p := by
  unfold ipSign
  split
  · split_ifs
    · right; rw [ipDen_smul]; ring
    · left; rfl
  · left; rfl

/-- The value moves by a multiple of `a - b` when `x` moves from `b` to `a`. -/
theorem ipDen_upd_dvd (ν : ℕ → ℤ) (x : ℕ) (a b : ℤ) (p : IPoly) :
    (a - b) ∣ ipDen (Function.update ν x a) p - ipDen (Function.update ν x b) p := by
  induction p with
  | nil => simp [ipDen]
  | cons t p ih =>
    rw [ipDen_cons, ipDen_cons, imDen_upd, imDen_upd]
    have : t.2 * (a ^ t.1.count x * imDen ν (t.1.filter (· != x))) +
        ipDen (Function.update ν x a) p - (t.2 * (b ^ t.1.count x *
        imDen ν (t.1.filter (· != x))) + ipDen (Function.update ν x b) p) =
        (a ^ t.1.count x - b ^ t.1.count x) * (t.2 * imDen ν (t.1.filter (· != x))) +
        (ipDen (Function.update ν x a) p - ipDen (Function.update ν x b) p) := by ring
    rw [this]
    exact dvd_add (Dvd.dvd.mul_right (sub_dvd_pow_sub_pow a b _) _) ih

/-! ## Coefficient monomials -/

/-- `q · ex r · Π_i √(d_i)^(sq_i) · Π_{c ∈ rc} √c · Π_j atom_j^(am_j) · Π_{(a, D) ∈ gc} gcd(a, D)`. -/
structure CMon where
  q : ℚ
  r : ℚ
  sq : List ℤ
  rc : List ℕ
  am : List ℤ
  gc : List (ℤ × DMon)
  deriving DecidableEq, Repr

/-- `gcd(a, D)` as a complex number. -/
noncomputable def gcdV (dv : ℕ → ℕ) (g : ℤ × DMon) : ℂ := (Nat.gcd g.1.natAbs (g.2.den dv) : ℂ)

/-- `√(d_i)` as a function of `i`. -/
noncomputable def rtv (dv : ℕ → ℕ) (i : ℕ) : ℂ := rt (dv i)

namespace CMon

/-- The value. -/
noncomputable def den (ρ : Val) (m : CMon) : ℂ :=
  (m.q : ℂ) * ex m.r * zpw (rtv ρ.dv) 0 m.sq * (m.rc.map rt).prod * zpw ρ.atom 0 m.am *
    (m.gc.map (gcdV ρ.dv)).prod

/-- The rational `q`. -/
def ofQ (q : ℚ) : CMon := ⟨q, 0, [], [], [], []⟩

/-- `q · ex r` with `r` reduced into `[0, 1/2)`. -/
def rnorm (q r : ℚ) : ℚ × ℚ :=
  let r' := r - (⌊r⌋ : ℚ)
  if 1 / 2 ≤ r' then (-q, r' - 1 / 2) else (q, r')

/-- `√c` merged into a sorted list of square roots: the rational factor and the list. -/
def rcIns (c : ℕ) : List ℕ → ℕ × List ℕ
  | [] => (1, [c])
  | x :: l =>
    if c = x then (c, l)
    else if c < x then (1, c :: x :: l)
    else ((rcIns c l).1, x :: (rcIns c l).2)

/-- The product of two lists of square roots: the rational factor and the list. -/
def rcMul (a b : List ℕ) : ℕ × List ℕ :=
  b.foldr (fun c acc => ((rcIns c acc.2).1 * acc.1, (rcIns c acc.2).2)) (1, a)

/-- Order on gcd atoms. -/
def gcLt (a b : ℤ × DMon) : Bool :=
  decide (a.1 < b.1) || (a.1 == b.1 && (decide (a.2.c < b.2.c) ||
    (a.2.c == b.2.c && lexLt a.2.e b.2.e)))

/-- Sorted insertion of a gcd atom. -/
def gcIns (g : ℤ × DMon) : List (ℤ × DMon) → List (ℤ × DMon)
  | [] => [g]
  | h :: l => if gcLt h g then h :: gcIns g l else g :: h :: l

/-- The product. -/
def mul (a b : CMon) : CMon :=
  let qr := rnorm (a.q * b.q) (a.r + b.r)
  let rr := rcMul a.rc b.rc
  ⟨qr.1 * rr.1, qr.2, ztrim (zadd a.sq b.sq), rr.2, ztrim (zadd a.am b.am),
    b.gc.foldr gcIns a.gc⟩

/-- The square root of `c` found by search, `0` when `c` is no square. -/
def sqrtS (c : ℕ) : ℕ := ((List.range (c + 1)).find? fun r => r * r == c).getD 0

/-- `(√c)^k` for a constant `c ≥ 1`. -/
def rtC (c : ℕ) (k : ℤ) : CMon :=
  if k % 2 = 0 then ofQ ((c : ℚ) ^ (k / 2))
  else if sqrtS c * sqrtS c = c then ofQ ((c : ℚ) ^ (k / 2) * sqrtS c)
  else ⟨(c : ℚ) ^ (k / 2), 0, [], [c], [], []⟩

/-- `(√D)^k`. -/
def rtD (D : DMon) (k : ℤ) : CMon :=
  let m := rtC D.c k
  { m with sq := ztrim (D.e.map fun x : ℕ => k * (x : ℤ)) }

/-- `gcd(a, D)`. -/
def gcdA (a : ℤ) (D : DMon) : CMon :=
  if D.isConst ∨ a.natAbs = 1 then ofQ (Nat.gcd a.natAbs D.den1) else
    ⟨1, 0, [], [], [], [((a.natAbs : ℤ), D)]⟩

/-- `ex r`. -/
def exQ (r : ℚ) : CMon := let qr := rnorm 1 r; ⟨qr.1, qr.2, [], [], [], []⟩

/-- Coefficient atom `j` to the power `k`. -/
def atomP (j : ℕ) (k : ℤ) : CMon := ⟨1, 0, [], [], ztrim (List.replicate j 0 ++ [k]), []⟩

end CMon

/-- What the valuation must satisfy: positive dimension symbols, nonzero atoms. -/
structure Val.OK (ρ : Val) : Prop where
  dpos : ∀ i, 0 < ρ.dv i
  anz : ∀ j, ρ.atom j ≠ 0

theorem ex_half : ex (1 / 2) = -1 := by
  simp only [ex]
  have : 2 * (Real.pi : ℂ) * Complex.I * ((1 / 2 : ℝ) : ℂ) = Real.pi * Complex.I := by
    push_cast; ring
  rw [this, Complex.exp_pi_mul_I]

theorem rnorm_den (q r : ℚ) :
    ((CMon.rnorm q r).1 : ℂ) * ex ((CMon.rnorm q r).2 : ℝ) = (q : ℂ) * ex (r : ℝ) := by
  have h0 : ex ((r - (⌊r⌋ : ℚ) : ℚ) : ℝ) = ex (r : ℝ) := by
    have := ex_int ((r - (⌊r⌋ : ℚ) : ℚ) : ℝ) ⌊r⌋
    rw [← this]; congr 1; push_cast; ring
  unfold CMon.rnorm
  simp only
  split_ifs
  · have h1 : ex (((r - (⌊r⌋ : ℚ) - 1 / 2 : ℚ)) : ℝ) * ex (1 / 2) =
        ex ((r - (⌊r⌋ : ℚ) : ℚ) : ℝ) := by
      rw [← ex_add]; congr 1; push_cast; ring
    rw [ex_half] at h1
    rw [← h0, ← h1]; push_cast; ring
  · rw [h0]

theorem rt_ne' (ρ : Val) (h : ρ.OK) (i : ℕ) : rtv ρ.dv i ≠ 0 := rt_ne _ (h.dpos i)

theorem rcIns_den (c : ℕ) (l : List ℕ) :
    ((CMon.rcIns c l).2.map rt).prod * ((CMon.rcIns c l).1 : ℂ) = rt c * (l.map rt).prod := by
  induction l with
  | nil => simp [CMon.rcIns]
  | cons x l ih =>
    simp only [CMon.rcIns]
    split_ifs with h1 h2
    · subst h1; simp only [List.map_cons, List.prod_cons]; rw [← mul_assoc, rt_sq]; ring
    · simp
    · simp only [List.map_cons, List.prod_cons]
      rw [mul_assoc, ih]; ring

theorem rcMul_den (a b : List ℕ) :
    ((CMon.rcMul a b).2.map rt).prod * ((CMon.rcMul a b).1 : ℂ) = (a.map rt).prod * (b.map rt).prod := by
  induction b with
  | nil => simp [CMon.rcMul]
  | cons c b ih =>
    simp only [CMon.rcMul, List.foldr_cons] at ih ⊢
    push_cast
    rw [← mul_assoc, rcIns_den]
    simp only [List.map_cons, List.prod_cons]
    rw [mul_assoc, ih]; ring

theorem gcIns_perm (g : ℤ × DMon) (l : List (ℤ × DMon)) : (CMon.gcIns g l).Perm (g :: l) := by
  induction l with
  | nil => simp [CMon.gcIns]
  | cons h l ih =>
    simp only [CMon.gcIns]
    split_ifs
    · exact (ih.cons h).trans (List.Perm.swap g h l)
    · exact List.Perm.refl _

theorem gc_prod (dv : ℕ → ℕ) (a b : List (ℤ × DMon)) :
    ((b.foldr CMon.gcIns a).map (gcdV dv)).prod = (a.map (gcdV dv)).prod * (b.map (gcdV dv)).prod := by
  induction b with
  | nil => simp
  | cons g b ih =>
    simp only [List.foldr_cons]
    rw [((gcIns_perm g _).map _).prod_eq]
    simp only [List.map_cons, List.prod_cons, ih]; ring

theorem CMon.den_mul (ρ : Val) (h : ρ.OK) (a b : CMon) :
    (CMon.mul a b).den ρ = a.den ρ * b.den ρ := by
  have hr := rnorm_den (a.q * b.q) (a.r + b.r)
  have hc := rcMul_den a.rc b.rc
  simp only [CMon.den, CMon.mul, zpw_ztrim, zpw_zadd _ (rt_ne' ρ h), zpw_zadd _ h.anz, gc_prod]
  push_cast
  have e1 : ((CMon.rnorm (a.q * b.q) (a.r + b.r)).1 : ℂ) * ((CMon.rcMul a.rc b.rc).1 : ℂ) *
      ex ((CMon.rnorm (a.q * b.q) (a.r + b.r)).2 : ℝ) =
      ((CMon.rnorm (a.q * b.q) (a.r + b.r)).1 : ℂ) * ex ((CMon.rnorm (a.q * b.q) (a.r + b.r)).2 : ℝ) *
      ((CMon.rcMul a.rc b.rc).1 : ℂ) := by ring
  have e2 : ex ((a.r + b.r : ℚ) : ℝ) = ex (a.r : ℝ) * ex (b.r : ℝ) := by
    rw [← ex_add]; push_cast; ring_nf
  calc _ = (((CMon.rnorm (a.q * b.q) (a.r + b.r)).1 : ℂ) * ex ((CMon.rnorm (a.q * b.q) (a.r + b.r)).2 : ℝ))
        * (((CMon.rcMul a.rc b.rc).2.map rt).prod * ((CMon.rcMul a.rc b.rc).1 : ℂ)) *
        (zpw (rtv ρ.dv) 0 a.sq * zpw (rtv ρ.dv) 0 b.sq) * (zpw ρ.atom 0 a.am * zpw ρ.atom 0 b.am) *
        ((a.gc.map (gcdV ρ.dv)).prod * (b.gc.map (gcdV ρ.dv)).prod) := by push_cast; ring
    _ = _ := by rw [hr, hc, e2]; push_cast; ring

theorem rt_mul (a b : ℕ) : rt (a * b) = rt a * rt b := by
  simp only [rt]; push_cast; rw [Real.sqrt_mul (Nat.cast_nonneg a)]; push_cast; ring

theorem rt_pow (a n : ℕ) : rt (a ^ n) = rt a ^ n := by
  induction n with
  | zero => simp [rt]
  | succ n ih => rw [pow_succ, rt_mul, ih, pow_succ]

theorem rt_npow (dv : ℕ → ℕ) (e : List ℕ) (k : ℕ) (z : ℤ) :
    rt (npow dv k e) ^ z = zpw (rtv dv) k (e.map fun x : ℕ => z * (x : ℤ)) := by
  induction e generalizing k with
  | nil => simp [npow, zpw, rt]
  | cons x e ih =>
    simp only [npow, rt_mul, rt_pow, List.map_cons, zpw, rtv]
    rw [mul_zpow, ← ih, ← zpow_natCast, ← zpow_mul, mul_comm (x : ℤ) z]

theorem rtC_den (ρ : Val) (c : ℕ) (hc : 0 < c) (k : ℤ) : (CMon.rtC c k).den ρ = rt c ^ k := by
  have hne : rt c ≠ 0 := rt_ne c hc
  have hk : rt c ^ k = (c : ℂ) ^ (k / 2) * rt c ^ (k % 2) := by
    conv_lhs => rw [← Int.emod_add_mul_ediv k 2]
    rw [zpow_add₀ hne, zpow_mul, show rt c ^ (2 : ℤ) = c from by
      rw [zpow_ofNat]; exact rt_sq' c]
    ring
  unfold CMon.rtC
  split_ifs with h1 h2
  · rw [hk, h1]; simp [CMon.den, CMon.ofQ, zpw, ex_zero]
  · have hk1 : k % 2 = 1 := by omega
    have hs : rt c = (CMon.sqrtS c : ℂ) := by
      simp only [rt]
      have : (c : ℝ) = (CMon.sqrtS c : ℝ) * (CMon.sqrtS c) := by exact_mod_cast h2.symm
      rw [this, Real.sqrt_mul_self (by positivity)]; push_cast; rfl
    rw [hk, hk1, zpow_one, hs]; simp [CMon.den, CMon.ofQ, zpw, ex_zero]
  · have hk1 : k % 2 = 1 := by omega
    rw [hk, hk1, zpow_one]; simp [CMon.den, zpw, ex_zero]

theorem rtD_den (ρ : Val) (D : DMon) (hc : 0 < D.c) (k : ℤ) :
    (CMon.rtD D k).den ρ = rt (D.den ρ.dv) ^ k := by
  have h := rtC_den ρ D.c hc k
  simp only [CMon.rtD, CMon.den] at h ⊢
  rw [zpw_ztrim, DMon.den, rt_mul, mul_zpow, ← h, rt_npow]
  unfold CMon.rtC
  split_ifs <;> simp [CMon.ofQ, zpw] <;> ring

theorem npow_one (e : List ℕ) (k : ℕ) : npow (fun _ => 1) k e = 1 := by
  induction e generalizing k with
  | nil => rfl
  | cons x e ih => simp [npow, ih]

theorem gcdA_den (ρ : Val) (a : ℤ) (D : DMon) :
    (CMon.gcdA a D).den ρ = (Nat.gcd a.natAbs (D.den ρ.dv) : ℂ) := by
  unfold CMon.gcdA
  split_ifs with h
  · rcases h with h | h
    · have : D.den ρ.dv = D.c := by
        rw [DMon.den_const ρ.dv (fun _ => 1) D h]; simp [DMon.den, npow_one]
      simp [CMon.den, CMon.ofQ, zpw, ex_zero, this, DMon.den1]
    · simp [CMon.den, CMon.ofQ, zpw, ex_zero, h]
  · simp [CMon.den, zpw, ex_zero, gcdV, Int.natAbs_abs]

theorem exQ_den (ρ : Val) (r : ℚ) : (CMon.exQ r).den ρ = ex (r : ℝ) := by
  have := rnorm_den 1 r
  simp only [CMon.exQ, CMon.den, zpw, List.map_nil, List.prod_nil, mul_one] at this ⊢
  rw [this]; simp

theorem atomP_den (ρ : Val) (j : ℕ) (k : ℤ) : (CMon.atomP j k).den ρ = ρ.atom j ^ k := by
  simp only [CMon.atomP, CMon.den, zpw_ztrim]
  suffices h : ∀ i, zpw ρ.atom i (List.replicate j 0 ++ [k]) = ρ.atom (i + j) ^ k by
    simp [h, zpw, ex_zero]
  intro i
  induction j generalizing i with
  | zero => simp [zpw]
  | succ j ih => simp [zpw, List.replicate_succ, ih, Nat.add_assoc, Nat.add_comm 1 j]

theorem ofQ_den (ρ : Val) (q : ℚ) : (CMon.ofQ q).den ρ = q := by
  simp [CMon.den, CMon.ofQ, zpw, ex_zero]

/-! ## Terms -/

/-- A delta `[M ∣ p]` or a root of unity `ω_D^p`. -/
abbrev Fac := DMon × IPoly

/-- `c · Σ_{sums} Π dels · Π phs`. -/
structure STm where
  c : CMon
  sums : List (ℕ × DMon)
  dels : List Fac
  phs : List Fac
  deriving DecidableEq, Repr

/-- The value of a delta. -/
noncomputable def delV (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f : Fac) : ℂ :=
  if ((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν f.2 then 1 else 0

/-- The value of a root of unity. -/
noncomputable def phV (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f : Fac) : ℂ := om (f.1.den dv) ^ ipDen ν f.2

/-- The summand. -/
noncomputable def body (dv : ℕ → ℕ) (t : STm) (ν : ℕ → ℤ) : ℂ :=
  (t.dels.map (delV dv ν)).prod * (t.phs.map (phV dv ν)).prod

/-- Nested sums, each variable over its range. -/
noncomputable def sumsV (dv : ℕ → ℕ) : List (ℕ × DMon) → ((ℕ → ℤ) → ℂ) → (ℕ → ℤ) → ℂ
  | [], f, ν => f ν
  | p :: s, f, ν => ∑ j ∈ range (p.2.den dv), sumsV dv s f (Function.update ν p.1 j)

/-- The value of a term. -/
noncomputable def STm.den (ρ : Val) (ν : ℕ → ℤ) (t : STm) : ℂ :=
  t.c.den ρ * sumsV ρ.dv t.sums (body ρ.dv t) ν

/-- The value of an expression. -/
noncomputable def exDen (ρ : Val) (ν : ℕ → ℤ) (e : List STm) : ℂ := (e.map (STm.den ρ ν)).sum

/-- The variables of a factor. -/
def facVars (f : Fac) : List ℕ := f.2.flatMap (·.1)

/-- The variables the summand reads. -/
def tmVars (t : STm) : List ℕ := (t.dels ++ t.phs).flatMap facVars

/-- The bound variables. -/
def bvs (t : STm) : List ℕ := t.sums.map (·.1)

/-- `f` reads `ν` only on `S`. -/
def RD (S : List ℕ) (f : (ℕ → ℤ) → ℂ) : Prop := ∀ ν ν', (∀ x ∈ S, ν x = ν' x) → f ν = f ν'

theorem RD.upd {S : List ℕ} {f : (ℕ → ℤ) → ℂ} (h : RD S f) (x : ℕ) (hx : x ∉ S) (ν : ℕ → ℤ)
    (v : ℤ) : f (Function.update ν x v) = f ν :=
  h _ _ fun y hy => Function.update_of_ne (fun e : y = x => hx (e ▸ hy)) _ _

theorem imDen_reads (m : IMon) (ν ν' : ℕ → ℤ) (h : ∀ x ∈ m, ν x = ν' x) : imDen ν m = imDen ν' m := by
  unfold imDen; congr 1; exact List.map_congr_left h

theorem ipDen_reads (p : IPoly) (ν ν' : ℕ → ℤ) (h : ∀ x ∈ p.flatMap (·.1), ν x = ν' x) :
    ipDen ν p = ipDen ν' p := by
  unfold ipDen; congr 1
  refine List.map_congr_left fun t ht => ?_
  rw [imDen_reads t.1 ν ν' fun x hx => h x (List.mem_flatMap.2 ⟨t, ht, hx⟩)]

theorem body_reads (dv : ℕ → ℕ) (t : STm) : RD (tmVars t) (body dv t) := by
  intro ν ν' h
  have hf : ∀ f ∈ t.dels ++ t.phs, ipDen ν f.2 = ipDen ν' f.2 := fun f hf =>
    ipDen_reads f.2 ν ν' fun x hx => h x (List.mem_flatMap.2 ⟨f, hf, hx⟩)
  unfold body delV phV
  congr 1
  · congr 1; exact List.map_congr_left fun f hf' => by rw [hf f (List.mem_append_left _ hf')]
  · congr 1; exact List.map_congr_left fun f hf' => by rw [hf f (List.mem_append_right _ hf')]

theorem sumsV_reads (dv : ℕ → ℕ) (S : List ℕ) (f : (ℕ → ℤ) → ℂ) (hf : RD S f)
    (s : List (ℕ × DMon)) : RD S (sumsV dv s f) := by
  induction s with
  | nil => exact hf
  | cons p s ih =>
    intro ν ν' h
    refine Finset.sum_congr rfl fun j _ => ih _ _ fun x hx => ?_
    by_cases e : x = p.1
    · subst e; simp
    · rw [Function.update_of_ne e, Function.update_of_ne e]; exact h x hx

theorem sumsV_congr (dv : ℕ → ℕ) (s : List (ℕ × DMon)) (f g : (ℕ → ℤ) → ℂ) (h : ∀ ν, f ν = g ν)
    (ν : ℕ → ℤ) : sumsV dv s f ν = sumsV dv s g ν := by
  rw [show f = g from funext h]

theorem sumsV_mul_left (dv : ℕ → ℕ) (s : List (ℕ × DMon)) (k : ℂ) (f : (ℕ → ℤ) → ℂ)
    (ν : ℕ → ℤ) : sumsV dv s (fun ν => k * f ν) ν = k * sumsV dv s f ν := by
  induction s generalizing ν with
  | nil => rfl
  | cons p s ih => simp only [sumsV, ih, Finset.mul_sum]

theorem sumsV_zero (dv : ℕ → ℕ) (s : List (ℕ × DMon)) (ν : ℕ → ℤ) :
    sumsV dv s (fun _ => 0) ν = 0 := by
  have := sumsV_mul_left dv s 0 (fun _ => 0) ν
  simpa using this

theorem sumsV_fsum {ι : Type*} (dv : ℕ → ℕ) (s : List (ℕ × DMon)) (I : Finset ι)
    (g : ι → (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ) :
    sumsV dv s (fun ν => ∑ i ∈ I, g i ν) ν = ∑ i ∈ I, sumsV dv s (g i) ν := by
  induction s generalizing ν with
  | nil => rfl
  | cons p s ih => simp only [sumsV, ih]; rw [Finset.sum_comm]

theorem sumsV_add (dv : ℕ → ℕ) (s : List (ℕ × DMon)) (f g : (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ) :
    sumsV dv s (fun ν => f ν + g ν) ν = sumsV dv s f ν + sumsV dv s g ν := by
  induction s generalizing ν with
  | nil => rfl
  | cons p s ih => simp only [sumsV, ih, Finset.sum_add_distrib]

theorem sumsV_lsum (dv : ℕ → ℕ) (s : List (ℕ × DMon)) (l : List ((ℕ → ℤ) → ℂ)) (ν : ℕ → ℤ) :
    sumsV dv s (fun ν => (l.map fun g => g ν).sum) ν = (l.map fun g => sumsV dv s g ν).sum := by
  induction l generalizing ν with
  | nil => simpa using sumsV_zero dv s ν
  | cons g l ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [sumsV_add, ih]

theorem sumsV_append (dv : ℕ → ℕ) (a b : List (ℕ × DMon)) (f : (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ) :
    sumsV dv (a ++ b) f ν = sumsV dv a (sumsV dv b f) ν := by
  induction a generalizing ν with
  | nil => rfl
  | cons p a ih => simp only [List.cons_append, sumsV, ih]

theorem sumsV_const_mul (dv : ℕ → ℕ) (s : List (ℕ × DMon)) (S : List ℕ) (g f : (ℕ → ℤ) → ℂ)
    (hg : RD S g) (hS : ∀ x ∈ S, x ∉ s.map (·.1)) (ν : ℕ → ℤ) :
    sumsV dv s (fun ν => g ν * f ν) ν = g ν * sumsV dv s f ν := by
  induction s generalizing ν with
  | nil => rfl
  | cons p s ih =>
    simp only [sumsV, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [ih (fun x hx hm => hS x hx (by simp [hm]))]
    congr 1
    exact hg.upd p.1 (fun hm => hS p.1 hm (by simp)) ν j

/-- Moving the sum over `x` innermost. -/
theorem sumsV_pull (dv : ℕ → ℕ) (a b : List (ℕ × DMon)) (x : ℕ) (D : DMon)
    (hx : x ∉ (a ++ b).map (·.1)) (f : (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ) :
    sumsV dv (a ++ (x, D) :: b) f ν =
      sumsV dv (a ++ b) (fun ν' => ∑ j ∈ range (D.den dv), f (Function.update ν' x j)) ν := by
  have comm : ∀ (b : List (ℕ × DMon)), x ∉ b.map (·.1) → ∀ (g : (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ) (j : ℤ),
      sumsV dv b g (Function.update ν x j) =
        sumsV dv b (fun ν' => g (Function.update ν' x j)) ν := by
    intro b hb
    induction b with
    | nil => intro g ν j; rfl
    | cons p b ih =>
      intro g ν j
      simp only [List.map_cons, List.mem_cons, not_or] at hb
      simp only [sumsV]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Function.update_comm hb.1, ih hb.2]
  induction a generalizing ν with
  | nil =>
    simp only [List.nil_append, sumsV] at hx ⊢
    rw [sumsV_fsum]
    exact Finset.sum_congr rfl fun j _ => comm b hx f ν j
  | cons p a ih =>
    simp only [List.cons_append, List.map_cons, List.mem_cons, not_or] at hx ⊢
    simp only [sumsV]
    exact Finset.sum_congr rfl fun j _ => ih (by simpa using hx.2) _

/-- The product of two terms. -/
def STm.mul (s t : STm) : STm := ⟨CMon.mul s.c t.c, s.sums ++ t.sums, s.dels ++ t.dels, s.phs ++ t.phs⟩

/-- The bound variables of `s` and `t` capture nothing of the other. -/
def mulOK (s t : STm) : Bool :=
  (bvs s).all (fun x => !(tmVars t).contains x) && (bvs t).all (fun x => !(tmVars s).contains x)

theorem body_mul (dv : ℕ → ℕ) (s t : STm) (ν : ℕ → ℤ) :
    body dv (STm.mul s t) ν = body dv s ν * body dv t ν := by
  simp only [body, STm.mul, List.map_append, List.prod_append]; ring

theorem STm.den_mul (ρ : Val) (hρ : ρ.OK) (s t : STm) (h : mulOK s t = true) (ν : ℕ → ℤ) :
    (STm.mul s t).den ρ ν = s.den ρ ν * t.den ρ ν := by
  have h' : (∀ x ∈ bvs s, x ∉ tmVars t) ∧ (∀ x ∈ bvs t, x ∉ tmVars s) := by
    simpa [mulOK] using h
  have inner : ∀ ν', sumsV ρ.dv t.sums (fun ν => body ρ.dv s ν * body ρ.dv t ν) ν' =
      body ρ.dv s ν' * sumsV ρ.dv t.sums (body ρ.dv t) ν' := fun ν' =>
    sumsV_const_mul ρ.dv t.sums (tmVars s) _ _ (body_reads ρ.dv s)
      (fun x hx hb => h'.2 x (by simpa [bvs] using hb) hx) ν'
  have outer : sumsV ρ.dv s.sums (fun ν' => sumsV ρ.dv t.sums (body ρ.dv t) ν' * body ρ.dv s ν') ν
      = sumsV ρ.dv t.sums (body ρ.dv t) ν * sumsV ρ.dv s.sums (body ρ.dv s) ν :=
    sumsV_const_mul ρ.dv s.sums (tmVars t) _ _ (sumsV_reads _ _ _ (body_reads ρ.dv t) _)
      (fun x hx hb => h'.1 x (by simpa [bvs] using hb) hx) ν
  show (CMon.mul s.c t.c).den ρ * sumsV ρ.dv (s.sums ++ t.sums) (body ρ.dv (STm.mul s t)) ν =
    s.den ρ ν * t.den ρ ν
  simp only [STm.den, CMon.den_mul ρ hρ]
  rw [sumsV_congr _ _ _ _ (body_mul ρ.dv s t), sumsV_append, sumsV_congr _ _ _ _ inner]
  rw [sumsV_congr _ _ _ (fun ν' => sumsV ρ.dv t.sums (body ρ.dv t) ν' * body ρ.dv s ν')
    (fun ν' => mul_comm _ _), outer]
  ring

theorem ipHas_add (x : ℕ) (p q : IPoly) (hp : ipHas x p = false) (hq : ipHas x q = false) :
    ipHas x (ipAdd p q) = false := by
  induction q with
  | nil => simpa [ipAdd] using hp
  | cons t q ih =>
    simp only [ipHas, List.any_cons, Bool.or_eq_false_iff] at hq
    simp only [ipAdd, List.foldr_cons] at ih ⊢
    exact ipHas_ins x _ _ _ hq.1 (ih hq.2)

end ArchytasZX.Sym
