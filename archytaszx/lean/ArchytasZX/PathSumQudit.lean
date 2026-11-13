/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSum
import Mathlib.NumberTheory.LegendreSymbol.QuadraticChar.Basic

/-!
# Qudit path sums over Z_p

A path sum over `σ : ℕ → ZMod p` is `Data`: path variables, phase terms `(monomial, k)` worth
`k / p` turns, constraints and support rows, each a polynomial over F_p whose monomials are
(variable, exponent) lists. `val p c s σ` sums `summand` over the path variables, each over
`ZMod p`. `elimS`, `hhConsS`, `hhPivotS`, `gaussS`, `canonS`, `pinS` and `normS` compute one step
and each `*_sound` theorem says the value is unchanged up to the step's factor under a `Bool`
side condition checked by `decide`. `qg_legendre` is the Legendre form of the quadratic Gauss
sum `G(a) = Sum_y zeta_p^(a y^2) = (a/p) Sum_k (k/p) zeta_p^k`.
-/

namespace ArchytasZX
namespace QSem

open Complex Finset

/-- A monomial: (variable, exponent) pairs. -/
abbrev Mono := List (ℕ × ℕ)
/-- A polynomial: monomials with coefficients mod p. -/
abbrev Poly := List (Mono × ℕ)

section Values

variable {p : ℕ}

/-- A monomial's value. -/
def qm (σ : ℕ → ZMod p) (m : Mono) : ZMod p := (m.map fun t => σ t.1 ^ t.2).prod

/-- A polynomial's value. -/
def qv (σ : ℕ → ZMod p) (P : Poly) : ZMod p := (P.map fun t => (t.2 : ZMod p) * qm σ t.1).sum

theorem qm_nil (σ : ℕ → ZMod p) : qm σ [] = 1 := rfl

theorem qm_cons (σ : ℕ → ZMod p) (t : ℕ × ℕ) (m : Mono) :
    qm σ (t :: m) = σ t.1 ^ t.2 * qm σ m := by simp [qm]

theorem qm_append (σ : ℕ → ZMod p) (a b : Mono) : qm σ (a ++ b) = qm σ a * qm σ b := by
  simp [qm, List.map_append, List.prod_append]

theorem qv_nil (σ : ℕ → ZMod p) : qv σ [] = 0 := rfl

theorem qv_cons (σ : ℕ → ZMod p) (t : Mono × ℕ) (P : Poly) :
    qv σ (t :: P) = (t.2 : ZMod p) * qm σ t.1 + qv σ P := by simp [qv]

theorem qv_append (σ : ℕ → ZMod p) (P R : Poly) : qv σ (P ++ R) = qv σ P + qv σ R := by
  simp [qv, List.map_append, List.sum_append]

theorem qv_perm (σ : ℕ → ZMod p) {P R : Poly} (h : P.Perm R) : qv σ P = qv σ R :=
  (h.map _).sum_eq

/-- The exponent of `y` in a monomial. -/
def yexp (y : ℕ) (m : Mono) : ℕ := (m.map fun t => if t.1 = y then t.2 else 0).sum

/-- A monomial without `y`. -/
def strip (y : ℕ) (m : Mono) : Mono := m.filter fun t => t.1 ≠ y

theorem yexp_nil (y : ℕ) : yexp y [] = 0 := rfl

theorem yexp_cons (y : ℕ) (t : ℕ × ℕ) (m : Mono) :
    yexp y (t :: m) = (if t.1 = y then t.2 else 0) + yexp y m := by simp [yexp]

theorem strip_cons (y : ℕ) (t : ℕ × ℕ) (m : Mono) :
    strip y (t :: m) = if t.1 = y then strip y m else t :: strip y m := by
  by_cases h : t.1 = y <;> simp [strip, h]

theorem qm_update (σ : ℕ → ZMod p) (y : ℕ) (b : ZMod p) :
    ∀ m : Mono, qm (Function.update σ y b) m = b ^ yexp y m * qm σ (strip y m)
  | [] => by simp [qm, yexp, strip]
  | t :: m => by
    rw [qm_cons, yexp_cons, strip_cons, qm_update σ y b m]
    by_cases h : t.1 = y
    · rw [if_pos h, if_pos h, h, Function.update_self, pow_add]
      ring
    · rw [if_neg h, if_neg h, qm_cons, Function.update_apply, if_neg h]
      ring

/-- The variables of a polynomial. -/
def pvars (P : Poly) : List ℕ := P.flatMap fun t => t.1.map Prod.fst

/-- The variables of a list of polynomials. -/
def cvars (C : List Poly) : List ℕ := C.flatMap pvars

theorem qm_update_not (σ : ℕ → ZMod p) (y : ℕ) (b : ZMod p) (m : Mono)
    (h : y ∉ m.map Prod.fst) : qm (Function.update σ y b) m = qm σ m := by
  unfold qm
  congr 1
  apply List.map_congr_left
  intro t ht
  rw [Function.update_apply, if_neg]
  intro e
  exact h (List.mem_map.2 ⟨t, ht, e⟩)

theorem qv_update_not (σ : ℕ → ZMod p) (y : ℕ) (b : ZMod p) (P : Poly) (h : y ∉ pvars P) :
    qv (Function.update σ y b) P = qv σ P := by
  unfold qv
  congr 1
  apply List.map_congr_left
  intro t ht
  rw [qm_update_not σ y b t.1 (fun e => h (List.mem_flatMap.2 ⟨t, ht, e⟩))]

theorem holds_update_not (σ : ℕ → ZMod p) (y : ℕ) (b : ZMod p) (C : List Poly)
    (h : y ∉ cvars C) :
    (∀ q ∈ C, qv (Function.update σ y b) q = 0) ↔ (∀ q ∈ C, qv σ q = 0) := by
  have hq : ∀ q ∈ C, y ∉ pvars q := fun q hq e => h (List.mem_flatMap.2 ⟨q, hq, e⟩)
  constructor
  · intro H q hqC
    rw [← qv_update_not σ y b q (hq q hqC)]
    exact H q hqC
  · intro H q hqC
    rw [qv_update_not σ y b q (hq q hqC)]
    exact H q hqC

/-! ## Splitting at a variable -/

/-- The terms whose exponent of `y` is `k`, `y` removed from each monomial. -/
def yk (y k : ℕ) (P : Poly) : Poly :=
  (P.filter fun t => yexp y t.1 = k).map fun t => (strip y t.1, t.2)

theorem yk_cons (y k : ℕ) (t : Mono × ℕ) (P : Poly) :
    yk y k (t :: P) = if yexp y t.1 = k then (strip y t.1, t.2) :: yk y k P else yk y k P := by
  by_cases h : yexp y t.1 = k <;> simp [yk, h]

theorem not_mem_strip (y : ℕ) (m : Mono) : y ∉ (strip y m).map Prod.fst := by
  intro hm
  obtain ⟨t, ht, e⟩ := List.mem_map.1 hm
  exact (List.mem_filter.1 ht).2 |> fun h => by simp at h; exact h e

theorem not_mem_yk (y k : ℕ) (P : Poly) : y ∉ pvars (yk y k P) := by
  intro hm
  obtain ⟨t, ht, e⟩ := List.mem_flatMap.1 hm
  obtain ⟨u, _, rfl⟩ := List.mem_map.1 ht
  exact not_mem_strip y u.1 e

theorem qv_yk_update (σ : ℕ → ZMod p) (y k : ℕ) (b : ZMod p) (P : Poly) :
    qv (Function.update σ y b) (yk y k P) = qv σ (yk y k P) :=
  qv_update_not σ y b _ (not_mem_yk y k P)

theorem qv_split (σ : ℕ → ZMod p) (y : ℕ) (b : ZMod p) :
    ∀ P : Poly, (∀ t ∈ P, yexp y t.1 ≤ 2) →
      qv (Function.update σ y b) P =
        qv σ (yk y 0 P) + b * qv σ (yk y 1 P) + b ^ 2 * qv σ (yk y 2 P)
  | [] => fun _ => by simp [yk, qv]
  | t :: P => fun h => by
    have ht := h t List.mem_cons_self
    have ih := qv_split σ y b P (fun u hu => h u (List.mem_cons_of_mem _ hu))
    rw [qv_cons, qm_update, ih, yk_cons, yk_cons, yk_cons]
    rcases (show yexp y t.1 = 0 ∨ yexp y t.1 = 1 ∨ yexp y t.1 = 2 by omega) with e | e | e <;>
      simp [e, qv_cons] <;> ring

theorem yk_two_nil (y : ℕ) (P : Poly) (h : ∀ t ∈ P, yexp y t.1 ≤ 1) : yk y 2 P = [] := by
  unfold yk
  rw [List.map_eq_nil_iff, List.filter_eq_nil_iff]
  intro t ht
  have := h t ht
  simp only [decide_eq_true_eq]
  omega

theorem qv_split1 (σ : ℕ → ZMod p) (y : ℕ) (b : ZMod p) (P : Poly)
    (h : ∀ t ∈ P, yexp y t.1 ≤ 1) :
    qv (Function.update σ y b) P = qv σ (yk y 0 P) + b * qv σ (yk y 1 P) := by
  rw [qv_split σ y b P (fun t ht => le_trans (h t ht) (by norm_num)), yk_two_nil y P h, qv_nil]
  ring

/-- The sum of the coefficients. -/
def csum (P : Poly) : ℕ := (P.map (·.2)).sum

theorem qv_const (σ : ℕ → ZMod p) : ∀ P : Poly, (∀ t ∈ P, t.1 = []) → qv σ P = (csum P : ZMod p)
  | [] => fun _ => by simp [qv, csum]
  | t :: P => fun h => by
    have ih := qv_const σ P (fun u hu => h u (List.mem_cons_of_mem _ hu))
    rw [qv_cons, ih, h t List.mem_cons_self, qm_nil]
    simp [csum]

/-! ## Polynomial operations -/

/-- A polynomial times a constant. -/
def scaleQ (k : ℕ) (P : Poly) : Poly := P.map fun t => (t.1, k * t.2)

theorem qv_scaleQ (σ : ℕ → ZMod p) (k : ℕ) : ∀ P : Poly, qv σ (scaleQ k P) = k * qv σ P
  | [] => by simp [scaleQ, qv]
  | t :: P => by
    have ih := qv_scaleQ σ k P
    simp only [scaleQ, List.map_cons] at ih ⊢
    rw [qv_cons, qv_cons, ih]
    push_cast
    ring

/-- The product of two polynomials. -/
def mulQ (A B : Poly) : Poly := A.flatMap fun s => B.map fun t => (s.1 ++ t.1, s.2 * t.2)

theorem qv_mapmul (σ : ℕ → ZMod p) (m : Mono) (c : ℕ) :
    ∀ B : Poly, qv σ (B.map fun t => (m ++ t.1, c * t.2)) = c * qm σ m * qv σ B
  | [] => by simp [qv]
  | t :: B => by
    rw [List.map_cons, qv_cons, qv_cons, qv_mapmul σ m c B, qm_append]
    push_cast
    ring

theorem qv_mulQ (σ : ℕ → ZMod p) (B : Poly) : ∀ A : Poly, qv σ (mulQ A B) = qv σ A * qv σ B
  | [] => by simp [mulQ, qv]
  | s :: A => by
    have ih := qv_mulQ σ B A
    unfold mulQ at ih ⊢
    rw [List.flatMap_cons, qv_append, qv_mapmul, ih, qv_cons]
    ring

/-- A polynomial to a power. -/
def powQ (B : Poly) : ℕ → Poly
  | 0 => [([], 1)]
  | n + 1 => mulQ (powQ B n) B

theorem qv_powQ (σ : ℕ → ZMod p) (B : Poly) : ∀ n : ℕ, qv σ (powQ B n) = qv σ B ^ n
  | 0 => by simp [powQ, qv, qm]
  | n + 1 => by rw [powQ, qv_mulQ, qv_powQ σ B n, pow_succ]

/-- `z` replaced by `B`. -/
def substQ (z : ℕ) (B : Poly) (P : Poly) : Poly :=
  P.flatMap fun t => mulQ [(strip z t.1, t.2)] (powQ B (yexp z t.1))

theorem qv_substQ (σ : ℕ → ZMod p) (z : ℕ) (B : Poly) :
    ∀ P : Poly, qv σ (substQ z B P) = qv (Function.update σ z (qv σ B)) P
  | [] => by simp [substQ, qv]
  | t :: P => by
    have ih := qv_substQ σ z B P
    unfold substQ at ih ⊢
    rw [List.flatMap_cons, qv_append, ih, qv_mulQ, qv_powQ, qv_cons, qv_cons, qv_nil, qm_update]
    ring

/-! ## Normal form -/

/-- `t` merged into the head of a monomial when it has the same variable. -/
def mergeE (t : ℕ × ℕ) : Mono → Mono
  | [] => [t]
  | u :: l => if u.1 = t.1 then (t.1, t.2 + u.2) :: l else t :: u :: l

theorem qm_mergeE (σ : ℕ → ZMod p) (t : ℕ × ℕ) :
    ∀ l : Mono, qm σ (mergeE t l) = σ t.1 ^ t.2 * qm σ l
  | [] => by simp [mergeE, qm]
  | u :: l => by
    simp only [mergeE]
    split_ifs with h
    · simp only [qm_cons, h, pow_add]
      ring
    · rw [qm_cons]

/-- An exponent reduced by `x^p = x`. -/
def redE (p e : ℕ) : ℕ := (e - 1) % (p - 1) + 1

theorem pow_shift [Fact p.Prime] (x : ZMod p) (r : ℕ) :
    ∀ n : ℕ, x ^ (n * (p - 1) + r + 1) = x ^ (r + 1)
  | 0 => by simp
  | n + 1 => by
    have hp : 2 ≤ p := (Fact.out : p.Prime).two_le
    have e : (n + 1) * (p - 1) + r + 1 = (n * (p - 1) + r) + p := by
      rw [Nat.succ_mul]; omega
    rw [e, pow_add, ZMod.pow_card, ← pow_succ, pow_shift x r n]

theorem pow_redE [Fact p.Prime] (x : ZMod p) (e : ℕ) (he : e ≠ 0) : x ^ redE p e = x ^ e := by
  unfold redE
  have h := Nat.div_add_mod (e - 1) (p - 1)
  conv_rhs => rw [show e = (e - 1) / (p - 1) * (p - 1) + (e - 1) % (p - 1) + 1 by
    rw [Nat.mul_comm]; omega]
  rw [pow_shift]

/-- A monomial sorted by variable, equal variables merged, zero exponents dropped, each
exponent reduced. -/
def mnormQ (p : ℕ) (m : Mono) : Mono :=
  (((m.insertionSort fun a b => a.1 ≤ b.1).foldr mergeE []).filter fun t => t.2 ≠ 0).map
    fun t => (t.1, redE p t.2)

theorem qm_foldr_mergeE (σ : ℕ → ZMod p) : ∀ m : Mono, qm σ (m.foldr mergeE []) = qm σ m
  | [] => rfl
  | t :: m => by rw [List.foldr_cons, qm_mergeE, qm_foldr_mergeE σ m, qm_cons]

theorem qm_filter_ne (σ : ℕ → ZMod p) : ∀ m : Mono, qm σ (m.filter fun t => t.2 ≠ 0) = qm σ m
  | [] => rfl
  | t :: m => by
    have ih := qm_filter_ne σ m
    by_cases h : t.2 = 0 <;> simp_all [qm_cons]

theorem qm_mnormQ [Fact p.Prime] (σ : ℕ → ZMod p) (m : Mono) : qm σ (mnormQ p m) = qm σ m := by
  unfold mnormQ
  have hred : ∀ l : Mono, (∀ t ∈ l, t.2 ≠ 0) →
      qm σ (l.map fun t => (t.1, redE p t.2)) = qm σ l := by
    intro l
    induction l with
    | nil => intro _; rfl
    | cons t l ih =>
      intro hl
      rw [List.map_cons, qm_cons, qm_cons, pow_redE _ _ (hl t List.mem_cons_self),
        ih (fun u hu => hl u (List.mem_cons_of_mem _ hu))]
  rw [hred _ (fun t ht => by simpa using (List.mem_filter.1 ht).2), qm_filter_ne,
    qm_foldr_mergeE]
  unfold qm
  exact ((List.perm_insertionSort _ m).map _).prod_eq

/-- `t` merged into a polynomial: coefficients added mod p, zero terms dropped. -/
def insQ (p : ℕ) (t : Mono × ℕ) : Poly → Poly
  | [] => if t.2 % p = 0 then [] else [(t.1, t.2 % p)]
  | u :: P =>
    if u.1 = t.1 then (if (u.2 + t.2) % p = 0 then P else (u.1, (u.2 + t.2) % p) :: P)
    else u :: insQ p t P

theorem cast_mod_zero (n : ℕ) (h : n % p = 0) : (n : ZMod p) = 0 := by
  rw [← ZMod.natCast_mod, h, Nat.cast_zero]

theorem qv_insQ (σ : ℕ → ZMod p) (t : Mono × ℕ) :
    ∀ P : Poly, qv σ (insQ p t P) = (t.2 : ZMod p) * qm σ t.1 + qv σ P
  | [] => by
    unfold insQ
    split_ifs with h
    · rw [cast_mod_zero _ h, qv_nil]; ring
    · rw [qv_cons, ZMod.natCast_mod]
  | u :: P => by
    unfold insQ
    split_ifs with h1 h2
    · rw [qv_cons, ← h1]
      have := cast_mod_zero _ h2
      push_cast at this
      linear_combination (-qm σ u.1) * this
    · rw [qv_cons, qv_cons, ZMod.natCast_mod, h1]
      push_cast
      ring
    · rw [qv_cons, qv_cons, qv_insQ σ t P]
      ring

/-- A lexicographic order on monomials. -/
def mleQ : Mono → Mono → Bool
  | [], _ => true
  | _ :: _, [] => false
  | a :: l, b :: r => a.1 < b.1 || (a.1 = b.1 && (a.2 < b.2 || (a.2 = b.2 && mleQ l r)))

/-- A polynomial with normalised monomials, terms merged mod p and sorted. -/
def normQ (p : ℕ) (P : Poly) : Poly :=
  (P.foldr (fun t acc => insQ p (mnormQ p t.1, t.2) acc) []).insertionSort fun a b => mleQ a.1 b.1

theorem qv_normQ [Fact p.Prime] (σ : ℕ → ZMod p) (P : Poly) : qv σ (normQ p P) = qv σ P := by
  unfold normQ
  rw [qv_perm σ (List.perm_insertionSort _ _)]
  induction P with
  | nil => rfl
  | cons t P ih =>
    rw [List.foldr_cons, qv_insQ, ih, qm_mnormQ, qv_cons]

end Values

/-! ## Roots of unity indexed by Z_p -/

section Psi

variable {p : ℕ}

/-- `zeta_p^x`. -/
noncomputable def ψ (p : ℕ) (x : ZMod p) : ℂ := om p ^ x.val

theorem ψ_add [NeZero p] (x y : ZMod p) : ψ p (x + y) = ψ p x * ψ p y := by
  unfold ψ
  rw [← pow_add, om_pow_eq_iff p (NeZero.pos p), ZMod.val_add, Nat.mod_mod]

theorem ψ_zero [NeZero p] : ψ p 0 = 1 := by simp [ψ]

theorem ψ_nat [NeZero p] (k : ℕ) : ψ p (k : ZMod p) = om p ^ k := by
  unfold ψ
  rw [ZMod.val_natCast, ← om_pow_mod p (NeZero.pos p)]

theorem zsum_range [NeZero p] (f : ℕ → ℂ) : ∑ x : ZMod p, f x.val = ∑ k ∈ range p, f k := by
  apply Finset.sum_nbij' (s := Finset.univ) (t := range p) (fun x : ZMod p => x.val)
    (fun k : ℕ => (k : ZMod p))
  · intro x _; simp [ZMod.val_lt]
  · intro k _; simp
  · intro x _; simp
  · intro k hk; rw [Finset.mem_range] at hk; exact ZMod.val_cast_of_lt hk
  · intro x _; rfl

theorem sum_ψ [Fact p.Prime] : ∑ x : ZMod p, ψ p x = 0 := by
  have hp := (Fact.out : p.Prime)
  rw [show (∑ x : ZMod p, ψ p x) = ∑ x : ZMod p, om p ^ x.val from rfl, zsum_range]
  have := om_geom p p hp.pos hp.one_lt dvd_rfl
  simpa [Nat.div_self hp.pos] using this

/-- The character sum over Z_p. -/
theorem char_sum [Fact p.Prime] (Q : ZMod p) :
    ∑ b : ZMod p, ψ p (b * Q) = if Q = 0 then (p : ℂ) else 0 := by
  split_ifs with hQ
  · simp [hQ, ψ_zero, Finset.card_univ, ZMod.card]
  · rw [Fintype.sum_equiv (Equiv.mulRight₀ Q hQ) (fun b => ψ p (b * Q)) (ψ p) (fun _ => rfl),
      sum_ψ]

/-- The quadratic Gauss sum `Sum_y zeta_p^(a y^2)`. -/
noncomputable def qg (p a : ℕ) [NeZero p] : ℂ := ∑ y : ZMod p, ψ p ((a : ZMod p) * y ^ 2)

/-- The Legendre symbol by Euler's criterion. -/
def lg (p a : ℕ) : ℤ :=
  if (a : ZMod p) = 0 then 0 else if (a : ZMod p) ^ (p / 2) = 1 then 1 else -1

theorem lg_eq [Fact p.Prime] (hp : p ≠ 2) (a : ℕ) :
    lg p a = quadraticChar (ZMod p) (a : ZMod p) := by
  unfold lg
  by_cases ha : (a : ZMod p) = 0
  · rw [if_pos ha, ha, quadraticChar_zero]
  · rw [if_neg ha, quadraticChar_eq_pow_of_char_ne_two (by rwa [ZMod.ringChar_zmod_n]) ha,
      ZMod.card]

theorem qg_char [Fact p.Prime] (hp : p ≠ 2) (a : ZMod p) (ha : a ≠ 0) :
    ∑ y : ZMod p, ψ p (a * y ^ 2) =
      (quadraticChar (ZMod p) a : ℂ) *
        ∑ x : ZMod p, (quadraticChar (ZMod p) x : ℂ) * ψ p x := by
  have hF : ringChar (ZMod p) ≠ 2 := by rwa [ZMod.ringChar_zmod_n]
  set χ := quadraticChar (ZMod p)
  have hfib : ∑ y : ZMod p, ψ p (a * y ^ 2) =
      ∑ x : ZMod p, ((χ x : ℂ) + 1) * ψ p (a * x) := by
    rw [← Finset.sum_fiberwise' Finset.univ (fun y : ZMod p => y ^ 2) (fun x => ψ p (a * x))]
    apply Finset.sum_congr rfl
    intro x _
    rw [Finset.sum_const, nsmul_eq_mul]
    congr 1
    have h := quadraticChar_card_sqrts hF x
    have e : (#{y : ZMod p | y ^ 2 = x}.toFinset : ℤ) =
        ((Finset.univ.filter fun y : ZMod p => y ^ 2 = x).card : ℤ) := by
      congr 2
      ext y
      simp
    rw [e] at h
    exact_mod_cast h
  have hlin : ∑ x : ZMod p, ψ p (a * x) = 0 := by
    rw [Fintype.sum_equiv (Equiv.mulLeft₀ a ha) (fun x => ψ p (a * x)) (ψ p) (fun _ => rfl),
      sum_ψ]
  have hsq : (χ a : ℂ) * (χ a : ℂ) = 1 := by
    have := quadraticChar_sq_one ha
    have h' : ((χ a * χ a : ℤ) : ℂ) = 1 := by rw [← sq, this]; simp
    exact_mod_cast h'
  have hmul : (χ a : ℂ) * ∑ x : ZMod p, (χ x : ℂ) * ψ p (a * x) =
      ∑ x : ZMod p, (χ x : ℂ) * ψ p x := by
    rw [Finset.mul_sum]
    rw [Fintype.sum_equiv (Equiv.mulLeft₀ a ha) (fun x => (χ a : ℂ) * ((χ x : ℂ) * ψ p (a * x)))
      (fun x => (χ x : ℂ) * ψ p x) (fun x => by
        show (χ a : ℂ) * ((χ x : ℂ) * ψ p (a * x)) = (χ (a * x) : ℂ) * ψ p (a * x)
        rw [map_mul, Int.cast_mul]
        ring)]
  rw [hfib]
  simp only [add_mul, one_mul, Finset.sum_add_distrib, hlin, add_zero]
  rw [← hmul, ← mul_assoc, hsq, one_mul]

/-- The Legendre form of `G(a)`: `G(a) = (a/p) Sum_{k < p} (k/p) zeta_p^k`. -/
theorem qg_legendre [Fact p.Prime] (hp : p ≠ 2) (a : ℕ) (ha : (a : ZMod p) ≠ 0) :
    qg p a = (lg p a : ℂ) * ∑ k ∈ range p, (lg p k : ℂ) * om p ^ k := by
  unfold qg
  rw [qg_char hp _ ha, lg_eq hp a]
  congr 1
  rw [← zsum_range (fun k => (lg p k : ℂ) * om p ^ k)]
  apply Finset.sum_congr rfl
  intro x _
  rw [lg_eq hp, ZMod.natCast_zmod_val]
  rfl

end Psi

/-! ## Sums over path variables -/

section Sums

variable {p : ℕ} [NeZero p]

/-- The sum over Z_p of each listed variable. -/
noncomputable def qsum : List ℕ → ((ℕ → ZMod p) → ℂ) → (ℕ → ZMod p) → ℂ
  | [], f, σ => f σ
  | y :: ys, f, σ => ∑ b : ZMod p, qsum ys f (Function.update σ y b)

theorem qsum_congr (f g : (ℕ → ZMod p) → ℂ) (h : ∀ τ, f τ = g τ) :
    ∀ (ys : List ℕ) (σ : ℕ → ZMod p), qsum ys f σ = qsum ys g σ
  | [], σ => h σ
  | y :: ys, σ => by simp only [qsum, qsum_congr f g h ys]

theorem qsum_fsum (F : ZMod p → (ℕ → ZMod p) → ℂ) :
    ∀ (ys : List ℕ) (σ : ℕ → ZMod p),
      qsum ys (fun τ => ∑ b : ZMod p, F b τ) σ = ∑ b : ZMod p, qsum ys (F b) σ
  | [], σ => rfl
  | y :: ys, σ => by
    simp only [qsum, qsum_fsum F ys]
    exact Finset.sum_comm

theorem qsum_out (f : (ℕ → ZMod p) → ℂ) (y : ℕ) (b : ZMod p) :
    ∀ (ys : List ℕ) (σ : ℕ → ZMod p), y ∉ ys →
      qsum ys f (Function.update σ y b) = qsum ys (fun τ => f (Function.update τ y b)) σ
  | [], σ, _ => rfl
  | x :: xs, σ, h => by
    have hx : y ≠ x := fun e => h (e ▸ List.mem_cons_self)
    have hxs : y ∉ xs := fun e => h (List.mem_cons_of_mem _ e)
    simp only [qsum]
    apply Finset.sum_congr rfl
    intro c _
    rw [Function.update_comm hx, qsum_out f y b xs _ hxs]

theorem qsum_perm (f : (ℕ → ZMod p) → ℂ) {l l' : List ℕ} (h : l.Perm l') :
    ∀ σ, qsum l f σ = qsum l' f σ := by
  induction h with
  | nil => intro σ; rfl
  | cons x _ ih => intro σ; simp only [qsum, ih]
  | swap x y l =>
    intro σ
    simp only [qsum]
    by_cases hxy : x = y
    · subst hxy
      simp only [Function.update_idem]
    · rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro a _
      apply Finset.sum_congr rfl
      intro b _
      rw [Function.update_comm hxy]
  | trans _ _ ih1 ih2 => intro σ; rw [ih1, ih2]

theorem qsum_head (f : (ℕ → ZMod p) → ℂ) (y : ℕ) (ys : List ℕ) (σ : ℕ → ZMod p) (h : y ∉ ys) :
    qsum (y :: ys) f σ = qsum ys (fun τ => ∑ b : ZMod p, f (Function.update τ y b)) σ := by
  simp only [qsum]
  rw [qsum_fsum]
  apply Finset.sum_congr rfl
  intro b _
  exact qsum_out f y b ys σ h

end Sums

/-! ## Path sums -/

/-- Path variables, phase, constraints and support rows. -/
structure Data where
  path : List ℕ
  phase : Poly
  cons : List Poly
  rows : List Poly
deriving DecidableEq

section Steps

variable {p : ℕ} [NeZero p]

/-- The summand at one assignment. -/
noncomputable def summand (p : ℕ) [NeZero p] (c : ℂ) (s : Data) (σ : ℕ → ZMod p) : ℂ :=
  if ∀ q ∈ s.cons ++ s.rows, qv σ q = 0 then c * ψ p (qv σ s.phase) else 0

/-- The value of a path sum with constant `c` at a boundary assignment. -/
noncomputable def val (p : ℕ) [NeZero p] (c : ℂ) (s : Data) (σ : ℕ → ZMod p) : ℂ :=
  qsum s.path (summand p c s) σ

theorem summand_update_not (c : ℂ) (s : Data) (τ : ℕ → ZMod p) (y : ℕ) (b : ZMod p)
    (hp : y ∉ pvars s.phase) (hc : y ∉ cvars (s.cons ++ s.rows)) :
    summand p c s (Function.update τ y b) = summand p c s τ := by
  unfold summand
  rw [qv_update_not τ y b s.phase hp]
  simp only [holds_update_not τ y b _ hc]

/-! ### Permutation -/

/-- The path reordered. -/
def permS (l : List ℕ) (s : Data) : Data := { s with path := l }

theorem perm_sound (c : ℂ) (l : List ℕ) (s : Data) (σ : ℕ → ZMod p)
    (h : decide (s.path.Perm l) = true) : val p c s σ = val p c (permS l s) σ := by
  unfold val permS
  exact qsum_perm _ (of_decide_eq_true h) σ

/-! ### Normalisation -/

/-- Phase, constraints and rows normalised. -/
def normS (p : ℕ) (s : Data) : Data :=
  { path := s.path, phase := normQ p s.phase, cons := s.cons.map (normQ p),
    rows := s.rows.map (normQ p) }

omit [NeZero p] in
theorem holds_map {σ σ' : ℕ → ZMod p} (g : Poly → Poly) (hg : ∀ q, qv σ (g q) = qv σ' q)
    (C : List Poly) : (∀ q ∈ C.map g, qv σ q = 0) ↔ (∀ q ∈ C, qv σ' q = 0) := by
  simp only [List.mem_map, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, hg]

theorem norm_sound [Fact p.Prime] (c : ℂ) (s : Data) (σ : ℕ → ZMod p) :
    val p c s σ = val p c (normS p s) σ := by
  unfold val normS
  apply qsum_congr
  intro τ
  unfold summand
  simp only [← List.map_append, holds_map (normQ p) (qv_normQ τ), qv_normQ]

/-! ### [Elim] -/

/-- Whether the head path variable is in no term, constraint or row and not in the tail. -/
def elimOK (s : Data) : Bool :=
  match s.path with
  | [] => false
  | y :: ys => !decide (y ∈ ys) && !decide (y ∈ pvars s.phase) &&
      !decide (y ∈ cvars (s.cons ++ s.rows))

/-- The head path variable dropped. -/
def elimS (s : Data) : Data := { s with path := s.path.tail }

theorem elim_sound (c : ℂ) (s : Data) (σ : ℕ → ZMod p) (h : elimOK s = true) :
    val p c s σ = val p (p * c) (elimS s) σ := by
  unfold val elimS
  rcases hp : s.path with _ | ⟨y, ys⟩
  · simp [elimOK, hp] at h
  · simp only [elimOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at h
    obtain ⟨⟨h1, h2⟩, h3⟩ := h
    rw [qsum_head _ y ys σ h1, List.tail_cons]
    apply qsum_congr
    intro τ
    simp only [summand_update_not c s τ y _ h2 h3]
    rw [Finset.sum_const, Finset.card_univ, ZMod.card, nsmul_eq_mul]
    unfold summand
    split_ifs <;> ring

/-! ### [HH] on a constraint -/

/-- Whether the head variable is linear in every term, in no constraint or row and not in the
tail. -/
def hhConsOK (s : Data) : Bool :=
  match s.path with
  | [] => false
  | y :: ys => !decide (y ∈ ys) && s.phase.all (fun t => decide (yexp y t.1 ≤ 1)) &&
      !decide (y ∈ cvars (s.cons ++ s.rows))

/-- The head variable summed: its coefficient becomes a constraint. -/
def hhConsS (s : Data) : Data :=
  match s.path with
  | [] => s
  | y :: ys =>
    { path := ys, phase := yk y 0 s.phase, cons := yk y 1 s.phase :: s.cons, rows := s.rows }

theorem hhCons_sound [Fact p.Prime] (c : ℂ) (s : Data) (σ : ℕ → ZMod p)
    (h : hhConsOK s = true) : val p c s σ = val p (p * c) (hhConsS s) σ := by
  unfold val
  rcases hp : s.path with _ | ⟨y, ys⟩
  · simp [hhConsOK, hp] at h
  · simp only [hhConsOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      List.all_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨h1, hlin⟩, h3⟩ := h
    simp only [hhConsS, hp]
    rw [qsum_head _ y ys σ h1]
    apply qsum_congr
    intro τ
    unfold summand
    simp only [holds_update_not τ y _ _ h3, qv_split1 τ y _ s.phase hlin]
    simp only [List.cons_append, List.forall_mem_cons]
    by_cases hC : ∀ q ∈ s.cons ++ s.rows, qv τ q = 0
    · simp only [if_pos hC, ψ_add]
      rw [← Finset.mul_sum, ← Finset.mul_sum, char_sum]
      by_cases hQ : qv τ (yk y 1 s.phase) = 0
      · rw [if_pos hQ, if_pos ⟨hQ, hC⟩]; ring
      · rw [if_neg hQ, if_neg (fun h => hQ h.1)]; ring
    · simp only [if_neg hC]
      rw [if_neg (fun h => hC h.2)]
      simp

/-! ### Pivots: a variable linear with an invertible constant coefficient -/

/-- `a^(p - 2) mod p`, the inverse of `a` mod a prime p. -/
def inv (p a : ℕ) : ℕ := a ^ (p - 2) % p

/-- Whether `v` is linear in every term of `q` with a constant coefficient invertible mod p. -/
def pivOK (p v : ℕ) (q : Poly) : Bool :=
  q.all (fun t => decide (yexp v t.1 ≤ 1)) && (yk v 1 q).all (fun t => decide (t.1 = [])) &&
    decide (csum (yk v 1 q) * inv p (csum (yk v 1 q)) % p = 1)

/-- The value `v` takes where `q` vanishes, `q` pivotable at `v`. -/
def pivVal (p v : ℕ) (q : Poly) : Poly := scaleQ ((p - 1) * inv p (csum (yk v 1 q))) (yk v 0 q)

theorem cast_pred_one : ((p - 1 : ℕ) : ZMod p) = -1 := by
  have hp := NeZero.pos p
  rw [Nat.cast_sub hp, ZMod.natCast_self]
  simp

theorem piv_iff (v : ℕ) (q : Poly) (hq : pivOK p v q = true) (τ : ℕ → ZMod p) (w : ZMod p) :
    qv (Function.update τ v w) q = 0 ↔ w = qv τ (pivVal p v q) := by
  simp only [pivOK, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hq
  obtain ⟨⟨hlin, hconst⟩, hinv⟩ := hq
  set a := csum (yk v 1 q)
  have ha : (a : ZMod p) * (inv p a : ZMod p) = 1 := by
    have := congrArg (fun n : ℕ => (n : ZMod p)) hinv
    simp only [ZMod.natCast_mod, Nat.cast_mul, Nat.cast_one] at this
    exact this
  rw [qv_split1 τ v w q hlin, qv_const τ _ hconst]
  unfold pivVal
  rw [qv_scaleQ]
  push_cast
  rw [cast_pred_one]
  constructor
  · intro h0
    linear_combination (inv p a : ZMod p) * h0 - w * ha
  · intro hw
    rw [hw]
    linear_combination (-(qv τ (yk v 0 q))) * ha

theorem sum_pin (F : ZMod p → ℂ) (B : ZMod p) (G : ZMod p → Prop) [DecidablePred G]
    (hG : ∀ w, G w ↔ w = B) : ∑ w : ZMod p, (if G w then (p : ℂ) else 0) * F w = p * F B := by
  simp only [hG, ite_mul, zero_mul]
  rw [Finset.sum_ite_eq' Finset.univ B (fun w => (p : ℂ) * F w)]
  simp

/-! ### [HH] on a path pivot -/

/-- Whether the head variable `y` is linear in every term and its coefficient is pivotable at
the next variable `z`; `y` in no constraint or row, distinct path. -/
def hhPivotOK (p : ℕ) (s : Data) : Bool :=
  match s.path with
  | y :: z :: ys => !decide (y = z) && !decide (y ∈ ys) && !decide (z ∈ ys) &&
      s.phase.all (fun t => decide (yexp y t.1 ≤ 1)) &&
      !decide (y ∈ cvars (s.cons ++ s.rows)) && pivOK p z (yk y 1 s.phase)
  | _ => false

/-- The head two variables summed: the second set to where the first's coefficient vanishes. -/
def hhPivotS (p : ℕ) (s : Data) : Data :=
  match s.path with
  | y :: z :: ys =>
    let B := pivVal p z (yk y 1 s.phase)
    { path := ys, phase := normQ p (substQ z B (yk y 0 s.phase)),
      cons := s.cons.map fun q => normQ p (substQ z B q),
      rows := s.rows.map fun q => normQ p (substQ z B q) }
  | _ => s

theorem hhPivot_sound [Fact p.Prime] (c : ℂ) (s : Data) (σ : ℕ → ZMod p)
    (h : hhPivotOK p s = true) : val p c s σ = val p (p * c) (hhPivotS p s) σ := by
  unfold val
  match hp : s.path, h with
  | [], h => simp [hhPivotOK, hp] at h
  | [_], h => simp [hhPivotOK, hp] at h
  | y :: z :: ys, h =>
    simp only [hhPivotOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      List.all_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨⟨⟨⟨hyz, hy⟩, hz⟩, hlin⟩, hc⟩, hpiv⟩ := h
    simp only [hhPivotS, hp]
    have hyzs : y ∉ z :: ys := by
      intro hm
      rcases List.mem_cons.1 hm with e | e
      · exact hyz e
      · exact hy e
    rw [qsum_head _ y (z :: ys) σ hyzs, qsum_head _ z ys σ hz]
    apply qsum_congr
    intro τ
    set Q := yk y 1 s.phase
    set R := yk y 0 s.phase
    set B := pivVal p z Q
    let G : ZMod p → ℂ := fun w =>
      if ∀ q ∈ s.cons ++ s.rows, qv (Function.update τ z w) q = 0 then
        c * ψ p (qv (Function.update τ z w) R) else 0
    have hstep : ∀ w : ZMod p,
        ∑ b : ZMod p, summand p c s (Function.update (Function.update τ z w) y b) =
          (if qv (Function.update τ z w) Q = 0 then (p : ℂ) else 0) * G w := by
      intro w
      unfold summand
      simp only [holds_update_not (Function.update τ z w) y _ _ hc,
        qv_split1 (Function.update τ z w) y _ s.phase hlin]
      show _ = _ * (if _ then _ else _)
      by_cases hC : ∀ q ∈ s.cons ++ s.rows, qv (Function.update τ z w) q = 0
      · simp only [if_pos hC, ψ_add]
        rw [← Finset.mul_sum, ← Finset.mul_sum, char_sum]
        split_ifs <;> ring
      · simp only [if_neg hC]
        simp
    simp only [hstep]
    rw [sum_pin G (qv τ B) _ (piv_iff z Q hpiv τ)]
    unfold summand
    simp only [G]
    have hcons : (∀ q ∈ (s.cons.map fun q => normQ p (substQ z B q)) ++
        (s.rows.map fun q => normQ p (substQ z B q)), qv τ q = 0) ↔
        (∀ q ∈ s.cons ++ s.rows, qv (Function.update τ z (qv τ B)) q = 0) := by
      rw [← List.map_append, holds_map (fun q => normQ p (substQ z B q))
        (fun q => by rw [qv_normQ, qv_substQ])]
    simp only [hcons, qv_normQ, qv_substQ]
    split_ifs <;> ring

/-! ### [Gauss] -/

/-- Whether the head variable has exponent at most 2 in every term, its square's coefficient
is the constant `A` with `4 A` invertible mod p, and it is in no constraint or row and not in
the tail. -/
def gaussOK (p A : ℕ) (s : Data) : Bool :=
  match s.path with
  | [] => false
  | y :: ys => !decide (y ∈ ys) && s.phase.all (fun t => decide (yexp y t.1 ≤ 2)) &&
      !decide (y ∈ cvars (s.cons ++ s.rows)) &&
      (yk y 2 s.phase).all (fun t => decide (t.1 = [])) && decide (csum (yk y 2 s.phase) = A) &&
      decide (4 * A * inv p (4 * A) % p = 1)

/-- The head variable summed by completing the square: `A y^2 + Q y + R` becomes
`R - Q^2 / (4 A)`. -/
def gaussS (p A : ℕ) (s : Data) : Data :=
  match s.path with
  | [] => s
  | y :: ys =>
    { path := ys,
      phase := normQ p (yk y 0 s.phase ++
        scaleQ ((p - 1) * inv p (4 * A)) (mulQ (yk y 1 s.phase) (yk y 1 s.phase))),
      cons := s.cons, rows := s.rows }

theorem complete_square (A Q R i : ZMod p) (h : 4 * A * i = 1) :
    ∑ b : ZMod p, ψ p (R + b * Q + b ^ 2 * A) =
      ψ p (R + (-1) * i * (Q * Q)) * ∑ b : ZMod p, ψ p (A * b ^ 2) := by
  have hb : ∀ b : ZMod p, R + b * Q + b ^ 2 * A =
      (R + (-1) * i * (Q * Q)) + A * (b + 2 * i * Q) ^ 2 := by
    intro b
    linear_combination (-(Q * Q * i) - b * Q) * h
  simp only [hb, ψ_add]
  rw [← Finset.mul_sum]
  congr 1
  exact Fintype.sum_equiv (Equiv.addRight (2 * i * Q)) _ _ (fun _ => rfl)

theorem gauss_sound [Fact p.Prime] (A : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod p) (h : gaussOK p A s = true) :
    val p c s σ = val p (qg p A * c) (gaussS p A s) σ := by
  unfold val
  rcases hp : s.path with _ | ⟨y, ys⟩
  · simp [gaussOK, hp] at h
  · simp only [gaussOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      List.all_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨⟨⟨⟨h1, hdeg⟩, h3⟩, hconst⟩, hA⟩, hinv⟩ := h
    simp only [gaussS, hp]
    rw [qsum_head _ y ys σ h1]
    apply qsum_congr
    intro τ
    unfold summand
    simp only [holds_update_not τ y _ _ h3, qv_split τ y _ s.phase hdeg]
    rw [qv_const τ _ hconst, hA]
    have hi : 4 * (A : ZMod p) * (inv p (4 * A) : ZMod p) = 1 := by
      have := congrArg (fun n : ℕ => (n : ZMod p)) hinv
      simp only [ZMod.natCast_mod, Nat.cast_mul, Nat.cast_one, Nat.cast_ofNat] at this
      exact this
    split_ifs
    · rw [← Finset.mul_sum, complete_square _ _ _ _ hi, qv_normQ, qv_append, qv_scaleQ, qv_mulQ]
      unfold qg
      push_cast
      rw [cast_pred_one]
      ring
    · simp

/-! ### Canonical-form pivot -/

/-- Whether some constraint is pivotable at `v`, `v` not a path variable. -/
def canonOK (p v : ℕ) (s : Data) : Bool :=
  (s.cons.find? (pivOK p v)).isSome && !decide (v ∈ s.path)

/-- The first constraint pivotable at `v` made a row with `v`'s coefficient 1: `v` substituted
out of the phase, the other constraints and the rows. -/
def canonS (p v : ℕ) (s : Data) : Data :=
  match s.cons.find? (pivOK p v) with
  | none => s
  | some q =>
    let B := pivVal p v q
    { path := s.path, phase := normQ p (substQ v B s.phase),
      cons := (s.cons.erase q).map fun r => normQ p (substQ v B r),
      rows := (s.rows.map fun r => normQ p (substQ v B r)) ++
        [normQ p (scaleQ (inv p (csum (yk v 1 q))) q)] }

theorem canon_sound [Fact p.Prime] (c : ℂ) (v : ℕ) (s : Data) (σ : ℕ → ZMod p)
    (h : canonOK p v s = true) : val p c s σ = val p c (canonS p v s) σ := by
  unfold val
  simp only [canonOK, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    Option.isSome_iff_exists] at h
  obtain ⟨⟨q, hq⟩, _⟩ := h
  have hiq : q ∈ s.cons := List.mem_of_find?_eq_some hq
  have hpq : pivOK p v q = true := List.find?_some hq
  simp only [canonS, hq]
  apply qsum_congr
  intro τ
  set B := pivVal p v q
  have hτ0 : qv τ q = 0 ↔ τ v = qv τ B := by
    have := piv_iff v q hpq τ (τ v)
    rwa [Function.update_eq_self] at this
  have hinv : ((csum (yk v 1 q) : ℕ) : ZMod p) * (inv p (csum (yk v 1 q)) : ZMod p) = 1 := by
    simp only [pivOK, Bool.and_eq_true, decide_eq_true_eq] at hpq
    have := congrArg (fun n : ℕ => (n : ZMod p)) hpq.2
    simp only [ZMod.natCast_mod, Nat.cast_mul, Nat.cast_one] at this
    exact this
  have hrow : qv τ (normQ p (scaleQ (inv p (csum (yk v 1 q))) q)) = 0 ↔ qv τ q = 0 := by
    rw [qv_normQ, qv_scaleQ]
    constructor
    · intro h0
      linear_combination ((csum (yk v 1 q) : ℕ) : ZMod p) * h0 - qv τ q * hinv
    · intro h0
      rw [h0, mul_zero]
  have hperm := List.perm_cons_erase hiq
  unfold summand
  by_cases h0 : qv τ q = 0
  · have hv : Function.update τ v (qv τ B) = τ := by
      rw [← hτ0.1 h0, Function.update_eq_self]
    have hsub : ∀ r, qv τ (normQ p (substQ v B r)) = qv τ r := by
      intro r
      rw [qv_normQ, qv_substQ, hv]
    have hcond : (∀ r ∈ (s.cons.erase q).map (fun r => normQ p (substQ v B r)) ++
        ((s.rows.map fun r => normQ p (substQ v B r)) ++
          [normQ p (scaleQ (inv p (csum (yk v 1 q))) q)]), qv τ r = 0) ↔
        (∀ r ∈ s.cons ++ s.rows, qv τ r = 0) := by
      simp only [List.mem_append, List.mem_map, List.mem_singleton]
      constructor
      · rintro H r (hr | hr)
        · rcases List.mem_cons.1 (hperm.mem_iff.1 hr) with e | hr'
          · rw [e]; exact h0
          · have := H _ (Or.inl ⟨r, hr', rfl⟩)
            rwa [hsub] at this
        · have := H _ (Or.inr (Or.inl ⟨r, hr, rfl⟩))
          rwa [hsub] at this
      · rintro H r (⟨r', hr', rfl⟩ | ⟨r', hr', rfl⟩ | rfl)
        · rw [hsub]; exact H r' (Or.inl (List.mem_of_mem_erase hr'))
        · rw [hsub]; exact H r' (Or.inr hr')
        · exact hrow.2 h0
    simp only [hcond]
    split_ifs
    · rw [qv_normQ, qv_substQ, hv]
    · rfl
  · have hl : ¬ (∀ r ∈ s.cons ++ s.rows, qv τ r = 0) :=
      fun H => h0 (H q (List.mem_append_left _ hiq))
    have hr : ¬ (∀ r ∈ (s.cons.erase q).map (fun r => normQ p (substQ v B r)) ++
        ((s.rows.map fun r => normQ p (substQ v B r)) ++
          [normQ p (scaleQ (inv p (csum (yk v 1 q))) q)]), qv τ r = 0) := by
      intro H
      apply h0
      exact hrow.1 (H _ (by simp))
    rw [if_neg hl, if_neg hr]

/-! ### The final comparison -/

/-- The constant term's coefficient. -/
def constK (P : Poly) : ℕ := ((P.filter fun t => decide (t.1 = [])).map (·.2)).sum

/-- The terms with a nonempty monomial. -/
def stripC (P : Poly) : Poly := P.filter fun t => !decide (t.1 = [])

omit [NeZero p] in
theorem qv_stripC (τ : ℕ → ZMod p) :
    ∀ P : Poly, qv τ P = (constK P : ZMod p) + qv τ (stripC P)
  | [] => by simp [qv, constK, stripC]
  | t :: P => by
    have ih := qv_stripC τ P
    by_cases ht : t.1 = []
    · have hc : constK (t :: P) = t.2 + constK P := by simp [constK, ht]
      have hs : stripC (t :: P) = stripC P := by simp [stripC, ht]
      rw [hc, hs, qv_cons, ih, ht, qm_nil]
      push_cast
      ring
    · have hc : constK (t :: P) = constK P := by simp [constK, ht]
      have hs : stripC (t :: P) = t :: stripC P := by simp [stripC, ht]
      rw [hc, hs, qv_cons, qv_cons, ih]
      ring

/-- Whether both sums have no path variable, only zero constraints, one list of rows and one
phase up to the constant term. -/
def finalOK (sL sR : Data) : Bool :=
  decide (sL.path = []) && decide (sR.path = []) && sL.cons.all (· = []) &&
    sR.cons.all (· = []) && decide (sL.rows = sR.rows) &&
    decide (stripC sL.phase = stripC sR.phase)

theorem final_sound (cL cR : ℂ) (sL sR : Data) (σ : ℕ → ZMod p) (kL kR : ℕ)
    (h : finalOK sL sR = true) (hkL : constK sL.phase = kL) (hkR : constK sR.phase = kR)
    (hc : cL * om p ^ kL = cR * om p ^ kR) : val p cL sL σ = val p cR sR σ := by
  subst hkL hkR
  simp only [finalOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hpL, hpR⟩, hcL⟩, hcR⟩, hrows⟩, hph⟩ := h
  unfold val summand
  rw [hpL, hpR]
  simp only [qsum]
  have hiff : ∀ (s : Data), (∀ q ∈ s.cons, q = []) →
      ((∀ q ∈ s.cons ++ s.rows, qv σ q = 0) ↔ (∀ q ∈ s.rows, qv σ q = 0)) := by
    intro s hs
    simp only [List.mem_append]
    constructor
    · intro H q hq; exact H q (Or.inr hq)
    · rintro H q (hq | hq)
      · rw [hs q hq]; rfl
      · exact H q hq
  rw [if_congr (hiff sL (fun q hq => by simpa using hcL q hq)) rfl rfl,
    if_congr (hiff sR (fun q hq => by simpa using hcR q hq)) rfl rfl, hrows]
  split_ifs
  · rw [qv_stripC σ sL.phase, qv_stripC σ sR.phase, hph, ψ_add, ψ_add, ψ_nat, ψ_nat]
    linear_combination ψ p (qv σ (stripC sR.phase)) * hc
  · rfl

/-! ### Pinning a path variable to another variable -/

/-- Whether `q` is pivotable at `y` with value `x`. -/
def pinTo (p y x : ℕ) (q : Poly) : Bool :=
  pivOK p y q && decide (normQ p (pivVal p y q) = [([(x, 1)], 1)])

/-- Whether the head path variable is not in the tail and some constraint sets it to `x`. -/
def pinOK (p x : ℕ) (s : Data) : Bool :=
  match s.path with
  | [] => false
  | y :: ys => !decide (y ∈ ys) && (s.cons.find? (pinTo p y x)).isSome

/-- The head path variable summed against the first constraint setting it to `x`: that
constraint dropped, the variable's value substituted into the phase, the other constraints and
the rows. -/
def pinS (p x : ℕ) (s : Data) : Data :=
  match s.path with
  | [] => s
  | y :: ys =>
    match s.cons.find? (pinTo p y x) with
    | none => s
    | some q =>
      let B := pivVal p y q
      { path := ys, phase := normQ p (substQ y B s.phase),
        cons := (s.cons.erase q).map fun r => normQ p (substQ y B r),
        rows := s.rows.map fun r => normQ p (substQ y B r) }

theorem pin_sound [Fact p.Prime] (c : ℂ) (x : ℕ) (s : Data) (σ : ℕ → ZMod p)
    (h : pinOK p x s = true) : val p c s σ = val p c (pinS p x s) σ := by
  unfold val
  rcases hp : s.path with _ | ⟨y, ys⟩
  · simp [pinOK, hp] at h
  · simp only [pinOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      Option.isSome_iff_exists] at h
    obtain ⟨h1, ⟨q, hq⟩⟩ := h
    have hiq : q ∈ s.cons := List.mem_of_find?_eq_some hq
    have hpq : pivOK p y q = true := by
      have := List.find?_some hq
      simp only [pinTo, Bool.and_eq_true] at this
      exact this.1
    simp only [pinS, hp, hq]
    rw [qsum_head _ y ys σ h1]
    apply qsum_congr
    intro τ
    set B := pivVal p y q
    set τB := Function.update τ y (qv τ B)
    have hperm := List.perm_cons_erase hiq
    have hsplit : ∀ τ' : ℕ → ZMod p, (∀ r ∈ s.cons ++ s.rows, qv τ' r = 0) ↔
        (qv τ' q = 0 ∧ ∀ r ∈ s.cons.erase q ++ s.rows, qv τ' r = 0) := by
      intro τ'
      simp only [List.mem_append]
      constructor
      · intro H
        refine ⟨H q (Or.inl hiq), fun r hr => ?_⟩
        rcases hr with hr | hr
        · exact H r (Or.inl (List.mem_of_mem_erase hr))
        · exact H r (Or.inr hr)
      · rintro ⟨H0, H⟩ r (hr | hr)
        · rcases List.mem_cons.1 (hperm.mem_iff.1 hr) with e | hr'
          · rw [e]; exact H0
          · exact H r (Or.inl hr')
        · exact H r (Or.inr hr)
    have hnew : (∀ r ∈ (s.cons.erase q).map (fun r => normQ p (substQ y B r)) ++
        s.rows.map (fun r => normQ p (substQ y B r)), qv τ r = 0) ↔
        (∀ r ∈ s.cons.erase q ++ s.rows, qv τB r = 0) := by
      rw [← List.map_append]
      exact holds_map (fun r => normQ p (substQ y B r)) (fun r => by rw [qv_normQ, qv_substQ]) _
    unfold summand
    simp only [hsplit, piv_iff y q hpq τ]
    rw [Finset.sum_eq_single (qv τ B)]
    · simp only [hnew, qv_normQ, qv_substQ]
      exact if_congr ⟨fun H => H.2, fun H => ⟨rfl, H⟩⟩ rfl rfl
    · intro b _ hb
      rw [if_neg (fun H => hb H.1)]
    · intro hm
      exact absurd (Finset.mem_univ _) hm

/-! ## Step lists -/

/-- One step of a replayed qudit reduction. -/
inductive Step where
  | perm (l : List ℕ)
  | norm
  | elim
  | hhCons
  | hhPivot
  | gauss (A : ℕ)
  | canon (v : ℕ)
  | pin (x : ℕ)

/-- A step's side condition. -/
def stepOK (p : ℕ) : Step → Data → Bool
  | .perm l, s => decide (s.path.Perm l)
  | .norm, _ => true
  | .elim, s => elimOK s
  | .hhCons, s => hhConsOK s
  | .hhPivot, s => hhPivotOK p s
  | .gauss A, s => gaussOK p A s
  | .canon v, s => canonOK p v s
  | .pin x, s => pinOK p x s

/-- A step applied. -/
def stepS (p : ℕ) : Step → Data → Data
  | .perm l, s => permS l s
  | .norm, s => normS p s
  | .elim, s => elimS s
  | .hhCons, s => hhConsS s
  | .hhPivot, s => hhPivotS p s
  | .gauss A, s => gaussS p A s
  | .canon v, s => canonS p v s
  | .pin x, s => pinS p x s

/-- The factor a step multiplies the constant by. -/
noncomputable def fac (p : ℕ) [NeZero p] : Step → ℂ
  | .elim | .hhCons | .hhPivot => p
  | .gauss A => qg p A
  | .perm _ | .norm | .canon _ | .pin _ => 1

/-- Every side condition along a step list. -/
def runOK (p : ℕ) : List Step → Data → Bool
  | [], _ => true
  | st :: sts, s => stepOK p st s && runOK p sts (stepS p st s)

/-- A step list applied. -/
def run (p : ℕ) : List Step → Data → Data
  | [], s => s
  | st :: sts, s => run p sts (stepS p st s)

/-- The product of the steps' factors. -/
noncomputable def facs (p : ℕ) [NeZero p] : List Step → ℂ
  | [] => 1
  | st :: sts => fac p st * facs p sts

theorem step_sound [Fact p.Prime] (st : Step) (s : Data) (c : ℂ) (σ : ℕ → ZMod p)
    (h : stepOK p st s = true) : val p c s σ = val p (fac p st * c) (stepS p st s) σ := by
  cases st with
  | perm l => simpa [fac, stepS] using perm_sound c l s σ h
  | norm => simpa [fac, stepS] using norm_sound c s σ
  | elim => simpa [fac, stepS] using elim_sound c s σ h
  | hhCons => simpa [fac, stepS] using hhCons_sound c s σ h
  | hhPivot => simpa [fac, stepS] using hhPivot_sound c s σ h
  | gauss A => simpa [fac, stepS] using gauss_sound A c s σ h
  | canon v => simpa [fac, stepS] using canon_sound c v s σ h
  | pin x => simpa [fac, stepS] using pin_sound c x s σ h

theorem run_sound [Fact p.Prime] :
    ∀ (sts : List Step) (s : Data) (c : ℂ) (σ : ℕ → ZMod p), runOK p sts s = true →
      val p c s σ = val p (facs p sts * c) (run p sts s) σ
  | [], s, c, σ, _ => by simp [facs, run]
  | st :: sts, s, c, σ, h => by
    simp only [runOK, Bool.and_eq_true] at h
    rw [step_sound st s c σ h.1, run_sound sts _ _ σ h.2]
    simp only [facs, run]
    ring_nf

end Steps

end QSem
end ArchytasZX
