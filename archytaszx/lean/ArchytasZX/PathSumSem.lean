/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSum

/-!
# Path-sum values and step soundness

A path sum over Boolean variables `σ : ℕ → ZMod 2` is `Data`: path variables, phase terms
`(monomial, k)` worth `k / M` turns, constraints and support rows (F_2 polynomials, sums of
monomials). `val M c s σ` sums `summand` over the path variables. `elimS`, `hhPivotS`,
`hhConsS`, `omegaS`, `omegaConjS` and `canonS` compute one rule step, and each `*_sound`
theorem says the value is unchanged under the step's syntactic side condition, a `Bool`
checked by `decide`. No lemma enumerates the path variables.
-/

namespace ArchytasZX
namespace PSem

open Complex

/-- A monomial: a list of variables. -/
abbrev Mono := List ℕ
/-- Phase terms: a monomial and its coefficient `k`, worth `k / M` turns. -/
abbrev Poly := List (Mono × ℕ)
/-- An F_2 polynomial: the sum of its monomials. -/
abbrev F2 := List Mono

/-- `0` or `1` as an integer. -/
def zq (x : ZMod 2) : ℤ := if x = 0 then 0 else 1

theorem zq_add : ∀ x y : ZMod 2, zq (x + y) = zq x + zq y - 2 * zq x * zq y := by decide
theorem zq_mul : ∀ x y : ZMod 2, zq (x * y) = zq x * zq y := by decide
theorem zq_cast : ∀ x : ZMod 2, ((zq x : ℤ) : ZMod 2) = x := by decide
theorem z2_cases : ∀ x : ZMod 2, x = 0 ∨ x = 1 := by decide
theorem z2_idem : ∀ x : ZMod 2, x * x = x := by decide

/-- A monomial's value. -/
def mz (σ : ℕ → ZMod 2) (m : Mono) : ZMod 2 := (m.map σ).prod

/-- An F_2 polynomial's value. -/
def fz (σ : ℕ → ZMod 2) (Q : F2) : ZMod 2 := (Q.map (mz σ)).sum

/-- A phase polynomial's value times M. -/
def pn (σ : ℕ → ZMod 2) (P : Poly) : ℤ := (P.map fun t => (t.2 : ℤ) * zq (mz σ t.1)).sum

/-- Two integers equal modulo `M`. -/
def peq (M : ℕ) (a b : ℤ) : Prop := ∃ n : ℤ, a = b + n * M

theorem peq_refl (M : ℕ) (a : ℤ) : peq M a a := ⟨0, by ring⟩

theorem peq_trans {M : ℕ} {a b c : ℤ} (h1 : peq M a b) (h2 : peq M b c) : peq M a c := by
  obtain ⟨n, hn⟩ := h1
  obtain ⟨k, hk⟩ := h2
  exact ⟨n + k, by rw [hn, hk]; ring⟩

theorem peq_add {M : ℕ} {a b c d : ℤ} (h1 : peq M a b) (h2 : peq M c d) :
    peq M (a + c) (b + d) := by
  obtain ⟨n, hn⟩ := h1
  obtain ⟨k, hk⟩ := h2
  exact ⟨n + k, by rw [hn, hk]; ring⟩

theorem peq_mul {M : ℕ} {a b : ℤ} (c : ℤ) (h : peq M a b) : peq M (c * a) (c * b) := by
  obtain ⟨n, hn⟩ := h
  exact ⟨c * n, by rw [hn]; ring⟩

/-- Equal phases modulo M give equal exponentials. -/
theorem ex_peq {M : ℕ} (hM : 0 < M) {a b : ℤ} (h : peq M a b) :
    ex ((a : ℝ) / M) = ex ((b : ℝ) / M) := by
  obtain ⟨n, hn⟩ := h
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  rw [hn]
  have : (((b + n * M : ℤ) : ℝ) / M) = (b : ℝ) / M + (n : ℤ) := by
    push_cast; field_simp
  rw [this, ex_int]

theorem pn_nil (σ : ℕ → ZMod 2) : pn σ [] = 0 := rfl

theorem pn_cons (σ : ℕ → ZMod 2) (t : Mono × ℕ) (P : Poly) :
    pn σ (t :: P) = (t.2 : ℤ) * zq (mz σ t.1) + pn σ P := by
  simp [pn]

theorem pn_append (σ : ℕ → ZMod 2) (P R : Poly) : pn σ (P ++ R) = pn σ P + pn σ R := by
  simp [pn, List.map_append, List.sum_append]

theorem mz_append (σ : ℕ → ZMod 2) (a b : Mono) : mz σ (a ++ b) = mz σ a * mz σ b := by
  simp [mz, List.map_append, List.prod_append]

theorem fz_cons (σ : ℕ → ZMod 2) (m : Mono) (Q : F2) : fz σ (m :: Q) = mz σ m + fz σ Q := by
  simp [fz]

theorem fz_append (σ : ℕ → ZMod 2) (P Q : F2) : fz σ (P ++ Q) = fz σ P + fz σ Q := by
  simp [fz, List.map_append, List.sum_append]

/-! ## Updates -/

theorem mz_update_not (σ : ℕ → ZMod 2) (z : ℕ) (w : ZMod 2) (m : Mono) (h : z ∉ m) :
    mz (Function.update σ z w) m = mz σ m := by
  unfold mz
  congr 1
  apply List.map_congr_left
  intro v hv
  rw [Function.update_apply, if_neg]
  rintro rfl
  exact h hv

theorem mz_update_in (σ : ℕ → ZMod 2) (z : ℕ) (w : ZMod 2) :
    ∀ m : Mono, z ∈ m → mz (Function.update σ z w) m = w * mz σ (m.filter (· != z))
  | [], h => by simp at h
  | v :: m, h => by
    by_cases hv : v = z
    · subst hv
      have hfil : (v :: m).filter (· != v) = m.filter (· != v) := by simp
      rw [hfil]
      show Function.update σ v w v * mz (Function.update σ v w) m = _
      rw [Function.update_self]
      by_cases hm : v ∈ m
      · rw [mz_update_in σ v w m hm, ← mul_assoc, z2_idem]
      · rw [mz_update_not σ v w m hm]
        congr 1
        unfold mz
        congr 1
        rw [List.filter_eq_self.2]
        intro a ha
        simp only [bne_iff_ne, ne_eq]
        rintro rfl
        exact hm ha
    · have hm : z ∈ m := by
        rcases List.mem_cons.1 h with h | h
        · exact absurd h.symm hv
        · exact h
      have hfil : (v :: m).filter (· != z) = v :: m.filter (· != z) := by
        simp [hv]
      rw [hfil]
      show Function.update σ z w v * mz (Function.update σ z w) m = w * (σ v * mz σ _)
      rw [Function.update_apply, if_neg hv, mz_update_in σ z w m hm]
      ring

theorem pn_update_not (σ : ℕ → ZMod 2) (z : ℕ) (w : ZMod 2) (P : Poly)
    (h : ∀ t ∈ P, z ∉ t.1) : pn (Function.update σ z w) P = pn σ P := by
  unfold pn
  congr 1
  apply List.map_congr_left
  intro t ht
  rw [mz_update_not σ z w t.1 (h t ht)]

theorem fz_update_not (σ : ℕ → ZMod 2) (z : ℕ) (w : ZMod 2) (Q : F2)
    (h : ∀ m ∈ Q, z ∉ m) : fz (Function.update σ z w) Q = fz σ Q := by
  unfold fz
  congr 1
  apply List.map_congr_left
  intro m hm
  exact mz_update_not σ z w m (h m hm)

/-! ## The Boolean lift and substitution -/

/-- The integer polynomial of an F_2 polynomial, `-2` written as `M - 2`. -/
def lift (M : ℕ) : F2 → Poly
  | [] => []
  | m :: B => (m, 1) :: (lift M B ++ (lift M B).map fun t => (m ++ t.1, (M - 2) * t.2))

theorem pn_mapmul (σ : ℕ → ZMod 2) (m : Mono) (c : ℕ) :
    ∀ L : Poly, pn σ (L.map fun t => (m ++ t.1, c * t.2)) = (c : ℤ) * zq (mz σ m) * pn σ L
  | [] => by simp [pn]
  | t :: L => by
    rw [List.map_cons, pn_cons, pn_cons, pn_mapmul σ m c L, mz_append, zq_mul]
    push_cast
    ring

theorem pn_lift (σ : ℕ → ZMod 2) (M : ℕ) (hM : 2 ≤ M) :
    ∀ B : F2, peq M (pn σ (lift M B)) (zq (fz σ B))
  | [] => ⟨0, by simp [lift, pn, fz, zq]⟩
  | m :: B => by
    obtain ⟨k, hk⟩ := pn_lift σ M hM B
    simp only [lift]
    rw [pn_cons, pn_append, pn_mapmul, fz_cons, zq_add]
    have hc : ((M - 2 : ℕ) : ℤ) = (M : ℤ) - 2 := by push_cast [Nat.cast_sub hM]; ring
    rw [hc]
    refine ⟨k + zq (mz σ m) * zq (fz σ B) + ((M : ℤ) - 2) * zq (mz σ m) * k, ?_⟩
    push_cast
    linear_combination (1 + ((M : ℤ) - 2) * zq (mz σ m)) * hk

/-- `z` replaced by the F_2 polynomial `B` in a phase polynomial. -/
def substP (M z : ℕ) (B : F2) (P : Poly) : Poly :=
  P.flatMap fun t =>
    if z ∈ t.1 then (lift M B).map fun u => (t.1.filter (· != z) ++ u.1, t.2 * u.2) else [t]

/-- `z` replaced by `B` in an F_2 polynomial. -/
def substF (z : ℕ) (B : F2) (Q : F2) : F2 :=
  Q.flatMap fun m => if z ∈ m then B.map fun u => m.filter (· != z) ++ u else [m]

theorem pn_substP (σ : ℕ → ZMod 2) (M z : ℕ) (hM : 2 ≤ M) (B : F2) :
    ∀ P : Poly, peq M (pn σ (substP M z B P)) (pn (Function.update σ z (fz σ B)) P)
  | [] => peq_refl M 0
  | t :: P => by
    have ih := pn_substP σ M z hM B P
    have hsplit : substP M z B (t :: P) = (if z ∈ t.1 then (lift M B).map fun u =>
        (t.1.filter (· != z) ++ u.1, t.2 * u.2) else [t]) ++ substP M z B P := by
      simp [substP, List.flatMap_cons]
    rw [hsplit, pn_append, pn_cons]
    apply peq_add _ ih
    by_cases hz : z ∈ t.1
    · rw [if_pos hz]
      have hmap : ((lift M B).map fun u => (t.1.filter (· != z) ++ u.1, t.2 * u.2)) =
          (lift M B).map fun u => (t.1.filter (· != z) ++ u.1, t.2 * u.2) := rfl
      rw [hmap, pn_mapmul, mz_update_in σ z _ t.1 hz, zq_mul]
      have := peq_mul ((t.2 : ℤ) * zq (mz σ (t.1.filter (· != z)))) (pn_lift σ M hM B)
      obtain ⟨n, hn⟩ := this
      exact ⟨n, by rw [hn]; ring⟩
    · rw [if_neg hz, pn_cons, pn_nil, mz_update_not σ z _ t.1 hz]
      exact ⟨0, by ring⟩

theorem fz_mapprefix (σ : ℕ → ZMod 2) (m : Mono) :
    ∀ B : F2, fz σ (B.map fun u => m ++ u) = mz σ m * fz σ B
  | [] => by simp [fz]
  | u :: B => by
    rw [List.map_cons, fz_cons, fz_cons, fz_mapprefix σ m B, mz_append]
    ring

theorem fz_substF (σ : ℕ → ZMod 2) (z : ℕ) (B : F2) :
    ∀ Q : F2, fz σ (substF z B Q) = fz (Function.update σ z (fz σ B)) Q
  | [] => rfl
  | m :: Q => by
    have hsplit : substF z B (m :: Q) =
        (if z ∈ m then B.map fun u => m.filter (· != z) ++ u else [m]) ++ substF z B Q := by
      simp [substF, List.flatMap_cons]
    rw [hsplit, fz_append, fz_cons, fz_substF σ z B Q]
    congr 1
    by_cases hz : z ∈ m
    · rw [if_pos hz, fz_mapprefix, mz_update_in σ z _ m hz]
      ring
    · rw [if_neg hz, fz_cons, mz_update_not σ z _ m hz]
      simp [fz]

/-! ## Normalisation -/

/-- Adjacent duplicates removed. -/
def squash : List ℕ → List ℕ
  | a :: b :: l => if a = b then squash (b :: l) else a :: squash (b :: l)
  | l => l

theorem mz_squash (σ : ℕ → ZMod 2) : ∀ l : List ℕ, mz σ (squash l) = mz σ l
  | [] => rfl
  | [a] => rfl
  | a :: b :: l => by
    have ih := mz_squash σ (b :: l)
    by_cases h : a = b
    · subst h
      simp only [squash, if_pos rfl]
      rw [ih]
      show mz σ (a :: l) = σ a * (σ a * mz σ l)
      show σ a * mz σ l = σ a * (σ a * mz σ l)
      rw [← mul_assoc, z2_idem]
    · simp only [squash, if_neg h]
      show σ a * mz σ (squash (b :: l)) = σ a * mz σ (b :: l)
      rw [ih]

/-- A monomial sorted with duplicates removed. -/
def mnorm (m : Mono) : Mono := squash (m.insertionSort (· ≤ ·))

theorem mz_mnorm (σ : ℕ → ZMod 2) (m : Mono) : mz σ (mnorm m) = mz σ m := by
  unfold mnorm
  rw [mz_squash]
  unfold mz
  exact ((List.perm_insertionSort (· ≤ ·) m).map σ).prod_eq

/-- `t` merged into `P`: coefficients added mod M, zero terms dropped. -/
def ins (M : ℕ) (t : Mono × ℕ) : Poly → Poly
  | [] => if t.2 % M = 0 then [] else [(t.1, t.2 % M)]
  | u :: P =>
    if u.1 = t.1 then (if (u.2 + t.2) % M = 0 then P else (u.1, (u.2 + t.2) % M) :: P)
    else u :: ins M t P

theorem peq_mod (M : ℕ) (k : ℕ) (z : ℤ) : peq M ((k % M : ℕ) * z) ((k : ℤ) * z) := by
  refine ⟨-((k / M : ℕ) : ℤ) * z, ?_⟩
  have h := Nat.mod_add_div k M
  have h' : ((k % M : ℕ) : ℤ) + (M : ℤ) * ((k / M : ℕ) : ℤ) = (k : ℤ) := by exact_mod_cast h
  linear_combination z * h'

theorem pn_ins (σ : ℕ → ZMod 2) (M : ℕ) (t : Mono × ℕ) :
    ∀ P : Poly, peq M (pn σ (ins M t P)) ((t.2 : ℤ) * zq (mz σ t.1) + pn σ P)
  | [] => by
    simp only [ins]
    split_ifs with h
    · have := peq_mod M t.2 (zq (mz σ t.1))
      rw [h] at this
      obtain ⟨n, hn⟩ := this
      exact ⟨-n, by simp [pn] at *; linarith⟩
    · rw [pn_cons, pn_nil]
      obtain ⟨n, hn⟩ := peq_mod M t.2 (zq (mz σ t.1))
      exact ⟨n, by simp [pn]; linarith⟩
  | u :: P => by
    simp only [ins]
    split_ifs with h1 h2
    · obtain ⟨n, hn⟩ := peq_mod M (u.2 + t.2) (zq (mz σ t.1))
      rw [h2] at hn
      refine ⟨-n, ?_⟩
      rw [pn_cons, h1]
      push_cast at hn
      linarith
    · obtain ⟨n, hn⟩ := peq_mod M (u.2 + t.2) (zq (mz σ t.1))
      refine ⟨n, ?_⟩
      rw [pn_cons, pn_cons, h1]
      push_cast at hn ⊢
      linarith
    · obtain ⟨n, hn⟩ := pn_ins σ M t P
      refine ⟨n, ?_⟩
      rw [pn_cons, pn_cons, hn]
      ring

/-- A lexicographic order on monomials. -/
def mle : Mono → Mono → Bool
  | [], _ => true
  | _ :: _, [] => false
  | a :: l, b :: r => a < b || (a = b && mle l r)

/-- A phase polynomial with sorted monomials, terms merged mod M and sorted. -/
def normP (M : ℕ) (P : Poly) : Poly :=
  (P.foldr (fun t acc => ins M (mnorm t.1, t.2) acc) []).insertionSort fun a b => mle a.1 b.1

theorem pn_perm (σ : ℕ → ZMod 2) {P R : Poly} (h : P.Perm R) : pn σ P = pn σ R :=
  (h.map _).sum_eq

theorem pn_normP (σ : ℕ → ZMod 2) (M : ℕ) (P : Poly) : peq M (pn σ (normP M P)) (pn σ P) := by
  unfold normP
  rw [pn_perm σ (List.perm_insertionSort _ _)]
  induction P with
  | nil => exact peq_refl M 0
  | cons t P ih =>
    simp only [List.foldr_cons]
    refine peq_trans (pn_ins σ M _ _) ?_
    rw [pn_cons, mz_mnorm]
    exact peq_add (peq_refl M _) ih

/-- An F_2 polynomial with sorted monomials, equal pairs cancelled, sorted. -/
def normF (Q : F2) : F2 :=
  (Q.foldr (fun m acc => let k := mnorm m; if k ∈ acc then acc.erase k else k :: acc) []
    ).insertionSort mle

theorem fz_perm (σ : ℕ → ZMod 2) {P Q : F2} (h : P.Perm Q) : fz σ P = fz σ Q :=
  (h.map _).sum_eq

theorem fz_normF (σ : ℕ → ZMod 2) (Q : F2) : fz σ (normF Q) = fz σ Q := by
  unfold normF
  rw [fz_perm σ (List.perm_insertionSort _ _)]
  induction Q with
  | nil => rfl
  | cons m Q ih =>
    simp only [List.foldr_cons]
    split_ifs with h
    · have hp := List.perm_cons_erase h
      rw [fz_perm σ hp, fz_cons] at ih
      rw [fz_cons, ← ih, mz_mnorm, ← add_assoc, ← two_mul]
      have : (2 : ZMod 2) = 0 := by decide
      rw [this, zero_mul, zero_add]
    · rw [fz_cons, fz_cons, ih, mz_mnorm]

/-! ## Sums over path variables -/

/-- The sum over the values 0, 1 of each listed variable. -/
noncomputable def bsum : List ℕ → ((ℕ → ZMod 2) → ℂ) → (ℕ → ZMod 2) → ℂ
  | [], f, σ => f σ
  | y :: ys, f, σ => bsum ys f (Function.update σ y 0) + bsum ys f (Function.update σ y 1)

theorem bsum_congr (f g : (ℕ → ZMod 2) → ℂ) (h : ∀ τ, f τ = g τ) :
    ∀ (ys : List ℕ) (σ : ℕ → ZMod 2), bsum ys f σ = bsum ys g σ
  | [], σ => h σ
  | y :: ys, σ => by simp only [bsum, bsum_congr f g h ys]

theorem bsum_add (f g : (ℕ → ZMod 2) → ℂ) :
    ∀ (ys : List ℕ) (σ : ℕ → ZMod 2), bsum ys (fun τ => f τ + g τ) σ = bsum ys f σ + bsum ys g σ
  | [], σ => rfl
  | y :: ys, σ => by simp only [bsum, bsum_add f g ys]; ring

theorem bsum_smul (a : ℂ) (f : (ℕ → ZMod 2) → ℂ) :
    ∀ (ys : List ℕ) (σ : ℕ → ZMod 2), bsum ys (fun τ => a * f τ) σ = a * bsum ys f σ
  | [], σ => rfl
  | y :: ys, σ => by simp only [bsum, bsum_smul a f ys]; ring

theorem bsum_out (f : (ℕ → ZMod 2) → ℂ) (y : ℕ) (b : ZMod 2) :
    ∀ (ys : List ℕ) (σ : ℕ → ZMod 2), y ∉ ys →
      bsum ys f (Function.update σ y b) = bsum ys (fun τ => f (Function.update τ y b)) σ
  | [], σ, _ => rfl
  | x :: xs, σ, h => by
    have hx : y ≠ x := fun e => h (e ▸ List.mem_cons_self)
    have hxs : y ∉ xs := fun e => h (List.mem_cons_of_mem _ e)
    simp only [bsum]
    rw [Function.update_comm hx, Function.update_comm hx, bsum_out f y b xs _ hxs,
      bsum_out f y b xs _ hxs]

theorem bsum_perm (f : (ℕ → ZMod 2) → ℂ) {l l' : List ℕ} (h : l.Perm l') :
    ∀ σ, bsum l f σ = bsum l' f σ := by
  induction h with
  | nil => intro σ; rfl
  | cons x _ ih => intro σ; simp only [bsum, ih]
  | swap x y l =>
    intro σ
    simp only [bsum]
    by_cases hxy : x = y
    · subst hxy
      simp only [Function.update_idem]
      ring
    · rw [Function.update_comm hxy (0 : ZMod 2) (0 : ZMod 2),
        Function.update_comm hxy (0 : ZMod 2) (1 : ZMod 2),
        Function.update_comm hxy (1 : ZMod 2) (0 : ZMod 2),
        Function.update_comm hxy (1 : ZMod 2) (1 : ZMod 2)]
      ring
  | trans _ _ ih1 ih2 => intro σ; rw [ih1, ih2]

/-- The head variable's sum moved inside, for a head variable absent from the tail. -/
theorem bsum_head (f : (ℕ → ZMod 2) → ℂ) (y : ℕ) (ys : List ℕ) (σ : ℕ → ZMod 2)
    (h : y ∉ ys) : bsum (y :: ys) f σ =
      bsum ys (fun τ => f (Function.update τ y 0) + f (Function.update τ y 1)) σ := by
  simp only [bsum]
  rw [bsum_out f y 0 ys σ h, bsum_out f y 1 ys σ h, bsum_add]

/-! ## Path sums -/

/-- Path variables, phase, constraints and support rows. -/
structure Data where
  path : List ℕ
  phase : Poly
  cons : List F2
  rows : List F2
deriving DecidableEq

/-- The summand at one assignment. -/
noncomputable def summand (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) : ℂ :=
  if ∀ q ∈ s.cons ++ s.rows, fz σ q = 0 then c * ex ((pn σ s.phase : ℝ) / M) else 0

/-- The value of a path sum with constant `c` at a boundary assignment. -/
noncomputable def val (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) : ℂ :=
  bsum s.path (summand M c s) σ

/-- The variables of a phase polynomial. -/
def pvars (P : Poly) : List ℕ := P.flatMap (·.1)

/-- The variables of a list of F_2 polynomials. -/
def fvars (C : List F2) : List ℕ := C.flatMap fun q => q.flatMap id

theorem not_mem_pvars {y : ℕ} {P : Poly} (h : y ∉ pvars P) : ∀ t ∈ P, y ∉ t.1 := by
  intro t ht hy
  exact h (List.mem_flatMap.2 ⟨t, ht, hy⟩)

theorem not_mem_fvars {y : ℕ} {C : List F2} (h : y ∉ fvars C) : ∀ q ∈ C, ∀ m ∈ q, y ∉ m := by
  intro q hq m hm hy
  exact h (List.mem_flatMap.2 ⟨q, hq, List.mem_flatMap.2 ⟨m, hm, hy⟩⟩)

theorem holds_update_not (σ : ℕ → ZMod 2) (y : ℕ) (b : ZMod 2) (C : List F2)
    (h : y ∉ fvars C) :
    (∀ q ∈ C, fz (Function.update σ y b) q = 0) ↔ (∀ q ∈ C, fz σ q = 0) := by
  have hq := not_mem_fvars h
  constructor
  · intro H q hqC
    rw [← fz_update_not σ y b q (hq q hqC)]
    exact H q hqC
  · intro H q hqC
    rw [fz_update_not σ y b q (hq q hqC)]
    exact H q hqC

theorem summand_update_not (M : ℕ) (c : ℂ) (s : Data) (τ : ℕ → ZMod 2) (y : ℕ) (b : ZMod 2)
    (hp : y ∉ pvars s.phase) (hc : y ∉ fvars (s.cons ++ s.rows)) :
    summand M c s (Function.update τ y b) = summand M c s τ := by
  unfold summand
  rw [pn_update_not τ y b s.phase (not_mem_pvars hp)]
  simp only [holds_update_not τ y b _ hc]

/-! ## Permutation of the path -/

/-- The path reordered. -/
def permS (l : List ℕ) (s : Data) : Data := { s with path := l }

theorem perm_sound (M : ℕ) (c : ℂ) (l : List ℕ) (s : Data) (σ : ℕ → ZMod 2)
    (h : decide (s.path.Perm l) = true) : val M c s σ = val M c (permS l s) σ := by
  unfold val permS
  exact bsum_perm _ (of_decide_eq_true h) σ

/-! ## [Elim] -/

/-- Whether the head path variable is in no term, constraint or row and not in the tail. -/
def elimOK (s : Data) : Bool :=
  match s.path with
  | [] => false
  | y :: ys => !decide (y ∈ ys) && !decide (y ∈ pvars s.phase) &&
      !decide (y ∈ fvars (s.cons ++ s.rows))

/-- The head path variable dropped. -/
def elimS (s : Data) : Data := { s with path := s.path.tail }

theorem elim_sound (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) (h : elimOK s = true) :
    val M c s σ = val M (2 * c) (elimS s) σ := by
  unfold val elimS
  rcases hp : s.path with _ | ⟨y, ys⟩
  · simp [elimOK, hp] at h
  · simp only [elimOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at h
    obtain ⟨⟨h1, h2⟩, h3⟩ := h
    rw [bsum_head _ y ys σ h1, List.tail_cons]
    apply bsum_congr
    intro τ
    rw [summand_update_not M c s τ y 0 h2 h3, summand_update_not M c s τ y 1 h2 h3]
    unfold summand
    split_ifs <;> ring

/-! ## The half-coefficient quotient -/

/-- The terms of `y`. -/
def yTerms (y : ℕ) (P : Poly) : Poly := P.filter fun t => decide (y ∈ t.1)

/-- The terms without `y`. -/
def rest (y : ℕ) (P : Poly) : Poly := P.filter fun t => !decide (y ∈ t.1)

/-- The quotient of the terms of `y`, `y` removed from each monomial. -/
def quot (y : ℕ) (P : Poly) : F2 := (yTerms y P).map fun t => t.1.filter (· != y)

/-- Whether every term of `y` has coefficient M / 2. -/
def halfOK (M y : ℕ) (P : Poly) : Bool := (yTerms y P).all fun t => decide (2 * t.2 = M)

theorem zsum_cast : ∀ L : List (ZMod 2), (((L.map zq).sum : ℤ) : ZMod 2) = L.sum
  | [] => by simp
  | x :: L => by simp only [List.map_cons, List.sum_cons]; push_cast; rw [zq_cast, zsum_cast L]

theorem pn_half (M : ℕ) (hM : 0 < M) (τ : ℕ → ZMod 2) (y : ℕ) (b : ZMod 2) :
    ∀ P : Poly, halfOK M y P = true →
      ((pn (Function.update τ y b) P : ℤ) : ℝ) / M =
        (zq b : ℝ) * (((quot y P).map fun m => zq (mz τ m)).sum : ℤ) / 2 +
          ((pn τ (rest y P) : ℤ) : ℝ) / M
  | [] => fun _ => by simp [pn, quot, rest, yTerms]
  | t :: P => fun h => by
    have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
    by_cases hy : y ∈ t.1
    · have h2 : 2 * t.2 = M := by
        simp only [halfOK, yTerms, List.filter_cons, hy, decide_true, if_true, List.all_cons,
          Bool.and_eq_true, decide_eq_true_eq] at h
        exact h.1
      have hP : halfOK M y P = true := by
        simp only [halfOK, yTerms, List.filter_cons, hy, decide_true, if_true, List.all_cons,
          Bool.and_eq_true] at h
        exact h.2
      have ih := pn_half M hM τ y b P hP
      have hq : quot y (t :: P) = t.1.filter (· != y) :: quot y P := by
        simp [quot, yTerms, hy]
      have hr : rest y (t :: P) = rest y P := by simp [rest, hy]
      rw [hq, hr, pn_cons, mz_update_in τ y b t.1 hy, zq_mul, List.map_cons, List.sum_cons]
      have hMt : (M : ℝ) = 2 * (t.2 : ℝ) := by exact_mod_cast h2.symm
      push_cast
      rw [add_div, ih, hMt]
      field_simp
      ring
    · have hP : halfOK M y P = true := by
        simpa [halfOK, yTerms, List.filter_cons, hy] using h
      have ih := pn_half M hM τ y b P hP
      have hq : quot y (t :: P) = quot y P := by simp [quot, yTerms, hy]
      have hr : rest y (t :: P) = t :: rest y P := by simp [rest, hy]
      rw [hq, hr, pn_cons, pn_cons, mz_update_not τ y b t.1 hy]
      push_cast
      rw [add_div, ih, add_div]
      ring

theorem ex_halfN (N : ℤ) : ex ((N : ℝ) / 2) = if (N : ZMod 2) = 0 then 1 else -1 := by
  rcases Int.even_or_odd' N with ⟨k, rfl | rfl⟩
  · have e : (((2 * k : ℤ) : ℝ) / 2) = 0 + ((k : ℤ) : ℝ) := by push_cast; ring
    have z : (((2 * k : ℤ)) : ZMod 2) = 0 := by
      push_cast
      rw [show (2 : ZMod 2) = 0 by decide, zero_mul]
    rw [e, ex_int, ex_zero, if_pos z]
  · have e : (((2 * k + 1 : ℤ) : ℝ) / 2) = 1 / 2 + ((k : ℤ) : ℝ) := by push_cast; ring
    have z : (((2 * k + 1 : ℤ)) : ZMod 2) ≠ 0 := by
      push_cast
      rw [show (2 : ZMod 2) = 0 by decide, zero_mul, zero_add]
      decide
    rw [e, ex_int, ex_half, if_neg z]

/-- The pair sum over `y` of a half-coefficient phase. -/
theorem pair_half (c : ℂ) (r : ℝ) (N : ℤ) :
    c * ex ((zq (0 : ZMod 2) : ℝ) * N / 2 + r) + c * ex ((zq (1 : ZMod 2) : ℝ) * N / 2 + r) =
      (if (N : ZMod 2) = 0 then 2 else 0) * (c * ex r) := by
  have h0 : zq (0 : ZMod 2) = 0 := by decide
  have h1 : zq (1 : ZMod 2) = 1 := by decide
  rw [h0, h1, ex_add, ex_add]
  push_cast
  rw [zero_mul, zero_div, ex_zero, one_mul, one_mul, ex_halfN]
  split_ifs <;> ring

/-! ## [HH] on a constraint -/

/-- Whether the head variable carries only half coefficients, is in no constraint or row and
not in the tail, with `0 < M`. -/
def hhConsOK (M : ℕ) (s : Data) : Bool :=
  match s.path with
  | [] => false
  | y :: ys => decide (0 < M) && !decide (y ∈ ys) && halfOK M y s.phase &&
      !decide (y ∈ fvars (s.cons ++ s.rows))

/-- The head variable summed: its quotient becomes a constraint. -/
def hhConsS (s : Data) : Data :=
  match s.path with
  | [] => s
  | y :: ys => { path := ys, phase := rest y s.phase, cons := quot y s.phase :: s.cons,
      rows := s.rows }

theorem hhCons_sound (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) (h : hhConsOK M s = true) :
    val M c s σ = val M (2 * c) (hhConsS s) σ := by
  unfold val
  rcases hp : s.path with _ | ⟨y, ys⟩
  · simp [hhConsOK, hp] at h
  · simp only [hhConsOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at h
    obtain ⟨⟨⟨hM, h1⟩, hh⟩, h3⟩ := h
    simp only [hhConsS, hp]
    rw [bsum_head _ y ys σ h1]
    apply bsum_congr
    intro τ
    unfold summand
    simp only [holds_update_not τ y _ _ h3]
    simp only [List.cons_append, List.forall_mem_cons]
    rw [pn_half M hM τ y 0 s.phase hh, pn_half M hM τ y 1 s.phase hh]
    have hQ : ((((quot y s.phase).map fun m => zq (mz τ m)).sum : ℤ) : ZMod 2) =
        fz τ (quot y s.phase) := by
      rw [show ((quot y s.phase).map fun m => zq (mz τ m)) =
        ((quot y s.phase).map (mz τ)).map zq by simp [List.map_map]]
      rw [zsum_cast]
      rfl
    split_ifs with hC hC'
    · rw [pair_half, ← hQ]
      simp only [hC'.1, if_true]
    · rw [pair_half]
      have : ¬ (((((quot y s.phase).map fun m => zq (mz τ m)).sum : ℤ) : ZMod 2) = 0) := by
        rw [hQ]
        intro e
        exact hC' ⟨e, hC⟩
      rw [if_neg this]
      ring
    · exact absurd hC'.2 hC
    · ring

/-! ## [HH] on a path pivot -/

/-- Whether the head variable `y` carries only half coefficients and the next variable `z` is
in its quotient only as the monomial `[z]`; `y` in no constraint or row, distinct path. -/
def hhPivotOK (M : ℕ) (s : Data) : Bool :=
  match s.path with
  | y :: z :: ys => decide (2 ≤ M) && !decide (y = z) && !decide (y ∈ ys) && !decide (z ∈ ys) &&
      halfOK M y s.phase && !decide (y ∈ fvars (s.cons ++ s.rows)) &&
      decide ([z] ∈ quot y s.phase) && !decide (z ∈ ((quot y s.phase).erase [z]).flatMap id)
  | _ => false

/-- The head two variables summed: the second set to the rest of the first's quotient. -/
def hhPivotS (M : ℕ) (s : Data) : Data :=
  match s.path with
  | y :: z :: ys =>
    let B := (quot y s.phase).erase [z]
    { path := ys, phase := normP M (substP M z B (rest y s.phase)),
      cons := s.cons.map fun q => normF (substF z B q),
      rows := s.rows.map fun q => normF (substF z B q) }
  | _ => s

theorem pick (v : ZMod 2) (G : ZMod 2 → ℂ) :
    (if (0 : ZMod 2) + v = 0 then (2 : ℂ) else 0) * G 0 +
      (if (1 : ZMod 2) + v = 0 then (2 : ℂ) else 0) * G 1 = 2 * G v := by
  rcases z2_cases v with rfl | rfl <;> simp [show (1 : ZMod 2) + 1 = 0 by decide]

theorem hhPivot_sound (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2)
    (h : hhPivotOK M s = true) : val M c s σ = val M (2 * c) (hhPivotS M s) σ := by
  unfold val
  match hp : s.path, h with
  | [], h => simp [hhPivotOK, hp] at h
  | [_], h => simp [hhPivotOK, hp] at h
  | y :: z :: ys, h =>
    simp only [hhPivotOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at h
    obtain ⟨⟨⟨⟨⟨⟨⟨hM, hyz⟩, hy⟩, hz⟩, hh⟩, hc⟩, hzq⟩, hzB⟩ := h
    have hM0 : 0 < M := by omega
    set Q := quot y s.phase with hQdef
    set B := Q.erase [z] with hBdef
    simp only [hhPivotS, hp]
    have hyzs : y ∉ z :: ys := by
      rintro (h | h)
      · exact hyz rfl
      · exact hy h
    rw [bsum_head _ y (z :: ys) σ hyzs, bsum_head _ z ys σ hz]
    apply bsum_congr
    intro τ
    have hBz : ∀ m ∈ B, z ∉ m := by
      intro m hm hzm
      exact hzB (List.mem_flatMap.2 ⟨m, hm, hzm⟩)
    have hfB : ∀ w : ZMod 2, fz (Function.update τ z w) B = fz τ B :=
      fun w => fz_update_not τ z w B hBz
    have hfQ : ∀ w : ZMod 2, fz (Function.update τ z w) Q = w + fz τ B := by
      intro w
      rw [fz_perm _ (List.perm_cons_erase hzq), fz_cons, hfB]
      simp [mz, Function.update_self]
    have hQcast : ∀ w : ZMod 2,
        ((((Q.map fun m => zq (mz (Function.update τ z w) m)).sum : ℤ)) : ZMod 2) =
          w + fz τ B := by
      intro w
      rw [show (Q.map fun m => zq (mz (Function.update τ z w) m)) =
        (Q.map (mz (Function.update τ z w))).map zq by simp [List.map_map], zsum_cast]
      exact hfQ w
    let G : ZMod 2 → ℂ := fun w =>
      if ∀ q ∈ s.cons ++ s.rows, fz (Function.update τ z w) q = 0 then
        c * ex ((pn (Function.update τ z w) (rest y s.phase) : ℝ) / M) else 0
    have hstep : ∀ w : ZMod 2,
        summand M c s (Function.update (Function.update τ z w) y 0) +
          summand M c s (Function.update (Function.update τ z w) y 1) =
        (if w + fz τ B = 0 then (2 : ℂ) else 0) * G w := by
      intro w
      unfold summand
      simp only [holds_update_not (Function.update τ z w) y _ _ hc]
      rw [pn_half M hM0 (Function.update τ z w) y 0 s.phase hh,
        pn_half M hM0 (Function.update τ z w) y 1 s.phase hh]
      show _ = _ * (if _ then _ else _)
      split_ifs with hC
      · rw [pair_half, hQcast]
      · simp
    rw [hstep 0, hstep 1, pick]
    show 2 * G (fz τ B) = 2 * summand M c _ τ
    congr 1
    unfold summand
    simp only [G]
    have hcons : (∀ q ∈ (s.cons.map fun q => normF (substF z B q)) ++
        (s.rows.map fun q => normF (substF z B q)), fz τ q = 0) ↔
        (∀ q ∈ s.cons ++ s.rows, fz (Function.update τ z (fz τ B)) q = 0) := by
      simp only [List.mem_append, List.mem_map]
      constructor
      · intro H q hq
        have := H (normF (substF z B q)) (hq.imp (fun h => ⟨q, h, rfl⟩) (fun h => ⟨q, h, rfl⟩))
        rwa [fz_normF, fz_substF] at this
      · rintro H q (⟨q', hq', rfl⟩ | ⟨q', hq', rfl⟩)
        · rw [fz_normF, fz_substF]; exact H q' (Or.inl hq')
        · rw [fz_normF, fz_substF]; exact H q' (Or.inr hq')
    simp only [hcons]
    split_ifs
    · congr 1
      symm
      apply ex_peq hM0
      exact peq_trans (pn_normP τ M _) (pn_substP τ M z hM B _)
    · rfl

/-! ## Canonical-form pivot -/

/-- Whether a constraint is `[v]` plus a polynomial free of `v`. -/
def pivotable (v : ℕ) (q : F2) : Bool :=
  decide ([v] ∈ q) && !decide (v ∈ (q.erase [v]).flatMap id)

/-- Whether some constraint is pivotable at `v`, `v` not a path variable. -/
def canonOK (v : ℕ) (s : Data) : Bool :=
  (s.cons.find? (pivotable v)).isSome && !decide (v ∈ s.path)

/-- The first constraint pivotable at `v` made a row: `v` substituted out of the phase, the
other constraints and the rows. -/
def canonS (M v : ℕ) (s : Data) : Data :=
  match s.cons.find? (pivotable v) with
  | none => s
  | some q =>
    let B := q.erase [v]
    { path := s.path, phase := normP M (substP M v B s.phase),
      cons := (s.cons.erase q).map fun r => normF (substF v B r),
      rows := (s.rows.map fun r => normF (substF v B r)) ++ [normF q] }

theorem z2_eq_of_add : ∀ a b : ZMod 2, a + b = 0 → a = b := by decide

theorem canon_sound (M : ℕ) (hM : 2 ≤ M) (c : ℂ) (v : ℕ) (s : Data) (σ : ℕ → ZMod 2)
    (h : canonOK v s = true) : val M c s σ = val M c (canonS M v s) σ := by
  unfold val
  simp only [canonOK, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    Option.isSome_iff_exists] at h
  obtain ⟨⟨q, hq⟩, _⟩ := h
  have hiq : q ∈ s.cons := List.mem_of_find?_eq_some hq
  have hpq : pivotable v q = true := List.find?_some hq
  simp only [pivotable, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    decide_eq_true_eq] at hpq
  obtain ⟨hvq, hvB⟩ := hpq
  have hM0 : 0 < M := by omega
  simp only [canonS, hq]
  apply bsum_congr
  intro τ
  set B := q.erase [v]
  have hqv : fz τ q = τ v + fz τ B := by
    rw [fz_perm _ (List.perm_cons_erase hvq), fz_cons]
    simp [mz]
  have hperm := List.perm_cons_erase hiq
  unfold summand
  by_cases h0 : fz τ q = 0
  · have hv : fz τ B = τ v := (z2_eq_of_add _ _ (by rw [← hqv]; exact h0)).symm
    have hτ : Function.update τ v (fz τ B) = τ := by rw [hv, Function.update_eq_self]
    have hsub : ∀ r, fz τ (normF (substF v B r)) = fz τ r := by
      intro r
      rw [fz_normF, fz_substF, hτ]
    have hcond : (∀ r ∈ (s.cons.erase q).map (fun r => normF (substF v B r)) ++
        ((s.rows.map fun r => normF (substF v B r)) ++ [normF q]), fz τ r = 0) ↔
        (∀ r ∈ s.cons ++ s.rows, fz τ r = 0) := by
      simp only [List.mem_append, List.mem_map, List.mem_singleton]
      constructor
      · rintro H r (hr | hr)
        · rcases (hperm.mem_iff.1 hr) with rfl | hr'
          · exact h0
          · have := H _ (Or.inl ⟨r, hr', rfl⟩)
            rwa [hsub] at this
        · have := H _ (Or.inr (Or.inl ⟨r, hr, rfl⟩))
          rwa [hsub] at this
      · rintro H r (⟨r', hr', rfl⟩ | ⟨r', hr', rfl⟩ | rfl)
        · rw [hsub]; exact H r' (Or.inl (List.mem_of_mem_erase hr'))
        · rw [hsub]; exact H r' (Or.inr hr')
        · rw [fz_normF]; exact h0
    simp only [hcond]
    split_ifs
    · congr 1
      apply ex_peq hM0
      have := peq_trans (pn_normP τ M _) (pn_substP τ M v hM B s.phase)
      rwa [hτ] at this
    · rfl
  · have hl : ¬ (∀ r ∈ s.cons ++ s.rows, fz τ r = 0) :=
      fun H => h0 (H q (List.mem_append_left _ hiq))
    have hr : ¬ (∀ r ∈ (s.cons.erase q).map (fun r => normF (substF v B r)) ++
        ((s.rows.map fun r => normF (substF v B r)) ++ [normF q]), fz τ r = 0) := by
      intro H
      apply h0
      have := H (normF q) (by simp)
      rwa [fz_normF] at this
    rw [if_neg hl, if_neg hr]

/-! ## The final comparison -/

/-- The constant term's coefficient. -/
def constK (P : Poly) : ℕ := ((P.filter fun t => decide (t.1 = [])).map (·.2)).sum

/-- The terms with a nonempty monomial. -/
def strip (P : Poly) : Poly := P.filter fun t => !decide (t.1 = [])

theorem pn_strip (τ : ℕ → ZMod 2) :
    ∀ P : Poly, pn τ P = (constK P : ℤ) + pn τ (strip P)
  | [] => by simp [pn, constK, strip]
  | t :: P => by
    have ih := pn_strip τ P
    by_cases ht : t.1 = []
    · have hc : constK (t :: P) = t.2 + constK P := by simp [constK, ht]
      have hs : strip (t :: P) = strip P := by simp [strip, ht]
      rw [hc, hs, pn_cons, ih, ht]
      have : zq (mz τ []) = 1 := by simp [mz]; decide
      rw [this]
      push_cast
      ring
    · have hc : constK (t :: P) = constK P := by simp [constK, ht]
      have hs : strip (t :: P) = t :: strip P := by simp [strip, ht]
      rw [hc, hs, pn_cons, pn_cons, ih]
      ring

/-- Whether both sums have no path variable, only zero constraints, one list of rows and one
phase up to the constant term. -/
def finalOK (sL sR : Data) : Bool :=
  decide (sL.path = []) && decide (sR.path = []) && sL.cons.all (· = []) &&
    sR.cons.all (· = []) && decide (sL.rows = sR.rows) &&
    decide (strip sL.phase = strip sR.phase)

theorem final_sound (M : ℕ) (hM : 0 < M) (cL cR : ℂ) (sL sR : Data) (σ : ℕ → ZMod 2)
    (kL kR : ℕ) (h : finalOK sL sR = true) (hkL : constK sL.phase = kL)
    (hkR : constK sR.phase = kR) (hc : cL * om M ^ kL = cR * om M ^ kR) :
    val M cL sL σ = val M cR sR σ := by
  subst hkL hkR
  simp only [finalOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hpL, hpR⟩, hcL⟩, hcR⟩, hrows⟩, hph⟩ := h
  unfold val summand
  rw [hpL, hpR]
  simp only [bsum]
  have hzero : ∀ (C : List F2), (∀ q ∈ C, q = []) → ∀ q ∈ C, fz σ q = 0 := by
    intro C hC q hq
    rw [hC q hq]
    rfl
  have hiff : ∀ (s : Data), (∀ q ∈ s.cons, q = []) →
      ((∀ q ∈ s.cons ++ s.rows, fz σ q = 0) ↔ (∀ q ∈ s.rows, fz σ q = 0)) := by
    intro s hs
    simp only [List.mem_append]
    constructor
    · intro H q hq; exact H q (Or.inr hq)
    · rintro H q (hq | hq)
      · exact hzero _ hs q hq
      · exact H q hq
  rw [if_congr (hiff sL (fun q hq => by simpa using hcL q hq)) rfl rfl,
    if_congr (hiff sR (fun q hq => by simpa using hcR q hq)) rfl rfl, hrows]
  split_ifs
  · rw [pn_strip σ sL.phase, pn_strip σ sR.phase, hph]
    push_cast
    rw [add_div, add_div, ex_add, ex_add, ← om_eq_ex, ← om_eq_ex]
    linear_combination ex (((pn σ (strip sR.phase) : ℤ) : ℝ) / M) * hc
  · rfl

/-! ## The exact split at the head variable -/

/-- The terms of `y` with `y` removed from each monomial. -/
def yq (y : ℕ) (P : Poly) : Poly := (yTerms y P).map fun t => (t.1.filter (· != y), t.2)

theorem pn_yupd (τ : ℕ → ZMod 2) (y : ℕ) (b : ZMod 2) :
    ∀ P : Poly, pn (Function.update τ y b) P = zq b * pn τ (yq y P) + pn τ (rest y P)
  | [] => by simp [pn, yq, yTerms, rest]
  | t :: P => by
    have ih := pn_yupd τ y b P
    by_cases hy : y ∈ t.1
    · have hq : yq y (t :: P) = (t.1.filter (· != y), t.2) :: yq y P := by
        simp [yq, yTerms, hy]
      have hr : rest y (t :: P) = rest y P := by simp [rest, hy]
      rw [hq, hr, pn_cons, pn_cons, ih, mz_update_in τ y b t.1 hy, zq_mul]
      ring
    · have hq : yq y (t :: P) = yq y P := by simp [yq, yTerms, hy]
      have hr : rest y (t :: P) = t :: rest y P := by simp [rest, hy]
      rw [hq, hr, pn_cons, pn_cons, ih, mz_update_not τ y b t.1 hy]
      ring

theorem pn_allhalf (τ : ℕ → ZMod 2) (M : ℕ) :
    ∀ P : Poly, (∀ t ∈ P, 2 * t.2 = M) →
      2 * pn τ P = (M : ℤ) * ((P.map fun t => zq (mz τ t.1)).sum)
  | [] => fun _ => by simp [pn]
  | t :: P => fun h => by
    have ih := pn_allhalf τ M P (fun u hu => h u (List.mem_cons_of_mem _ hu))
    have ht : (2 : ℤ) * t.2 = M := by exact_mod_cast h t List.mem_cons_self
    rw [pn_cons, List.map_cons, List.sum_cons, mul_add, ih, ← ht]
    ring

theorem zsum_mz (τ : ℕ → ZMod 2) (Q : F2) :
    ((((Q.map fun m => zq (mz τ m)).sum : ℤ)) : ZMod 2) = fz τ Q := by
  rw [show (Q.map fun m => zq (mz τ m)) = (Q.map (mz τ)).map zq by simp [List.map_map],
    zsum_cast]
  rfl

/-! ## [omega] at Q = 0 and Q = 1, as powers of zeta_8 -/

theorem om8_four : om 8 ^ 4 = -1 := by
  rw [om_eq_ex, show ((4 : ℕ) : ℝ) / ((8 : ℕ) : ℝ) = 1 / 2 by norm_num]
  exact ex_half

theorem omega_e1F0 : (1 : ℂ) + om 8 ^ 2 = rt 2 * om 8 := by
  rw [rt_sub2 8 1 (by norm_num) (by norm_num)]
  linear_combination (1 - om 8 ^ 4) * om8_four

theorem omega_e1F1 : (1 : ℂ) - om 8 ^ 2 = rt 2 * om 8 * om 8 ^ 6 := by
  rw [rt_sub2 8 1 (by norm_num) (by norm_num)]
  linear_combination (-om 8 ^ 10 + om 8 ^ 6 - om 8 ^ 4 - om 8 ^ 2 + 1) * om8_four

theorem omega_e3F0 : (1 : ℂ) + om 8 ^ 6 = rt 2 * om 8 ^ 7 := by
  rw [rt_sub2 8 1 (by norm_num) (by norm_num)]
  linear_combination (-om 8 ^ 10 + om 8 ^ 6 - om 8 ^ 4 + 1) * om8_four

theorem omega_e3F1 : (1 : ℂ) - om 8 ^ 6 = rt 2 * om 8 ^ 7 * om 8 ^ 2 := by
  rw [rt_sub2 8 1 (by norm_num) (by norm_num)]
  linear_combination (-om 8 ^ 12 + om 8 ^ 8 - om 8 ^ 6 - om 8 ^ 4 + 1) * om8_four

/-! ## Boolean elimination -/

/-- The product of two F_2 polynomials. -/
def f2mul (a b : F2) : F2 := a.flatMap fun m => b.map fun u => m ++ u

theorem fz_f2mul (τ : ℕ → ZMod 2) (b : F2) : ∀ a : F2, fz τ (f2mul a b) = fz τ a * fz τ b
  | [] => by simp [f2mul, fz]
  | m :: a => by
    have ih := fz_f2mul τ b a
    have hs : f2mul (m :: a) b = (b.map fun u => m ++ u) ++ f2mul a b := by
      simp [f2mul, List.flatMap_cons]
    rw [hs, fz_append, fz_mapprefix, ih, fz_cons]
    ring

/-- The support indicator `Prod (1 + c)` of a list of constraints. -/
def gatherPoly (C : List F2) : F2 := C.foldr (fun c acc => f2mul ([] :: c) acc) [[]]

theorem fz_gather (τ : ℕ → ZMod 2) :
    ∀ C : List F2, (fz τ (gatherPoly C) = 1 ↔ ∀ q ∈ C, fz τ q = 0)
  | [] => by simp [gatherPoly, fz, mz]
  | c :: C => by
    have ih := fz_gather τ C
    have hs : gatherPoly (c :: C) = f2mul ([] :: c) (gatherPoly C) := rfl
    rw [hs, fz_f2mul, fz_cons, show mz τ [] = 1 from rfl, List.forall_mem_cons, ← ih]
    rcases z2_cases (fz τ c) with h | h <;> rcases z2_cases (fz τ (gatherPoly C)) with g | g <;>
      simp [h, g] <;> decide

/-- Whether there is a constraint to gather and no path variable. -/
def gatherOK (s : Data) : Bool := decide (s.path = [])

/-- Every constraint replaced by the one constraint `1 + Prod (1 + c)`. -/
def gatherS (s : Data) : Data := { s with cons := [normF ([] :: gatherPoly s.cons)] }

theorem gather_sound (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) (_h : gatherOK s = true) :
    val M c s σ = val M c (gatherS s) σ := by
  unfold val gatherS
  apply bsum_congr
  intro τ
  unfold summand
  have hiff : (∀ q ∈ s.cons ++ s.rows, fz τ q = 0) ↔
      (∀ q ∈ [normF ([] :: gatherPoly s.cons)] ++ s.rows, fz τ q = 0) := by
    simp only [List.mem_append, List.mem_singleton]
    have hg : fz τ (normF ([] :: gatherPoly s.cons)) = 0 ↔ ∀ q ∈ s.cons, fz τ q = 0 := by
      rw [fz_normF, fz_cons, show mz τ [] = 1 from rfl, ← fz_gather τ s.cons]
      rcases z2_cases (fz τ (gatherPoly s.cons)) with g | g <;> simp [g] <;> decide
    constructor
    · rintro H q (rfl | hq)
      · exact hg.2 fun q hq => H q (Or.inl hq)
      · exact H q (Or.inr hq)
    · rintro H q (hq | hq)
      · exact hg.1 (H _ (Or.inl rfl)) q hq
      · exact H q (Or.inr hq)
  simp only [hiff]
  rfl

/-- The monomials of `g` with `v`, `v` removed, and those without `v`. -/
def splitAt (v : ℕ) (g : F2) : F2 × F2 :=
  ((g.filter fun m => decide (v ∈ m)).map fun m => m.filter (· != v),
    g.filter fun m => !decide (v ∈ m))

theorem fz_splitAt (τ : ℕ → ZMod 2) (v : ℕ) :
    ∀ g : F2, fz τ g = τ v * fz τ (splitAt v g).1 + fz τ (splitAt v g).2
  | [] => by simp [splitAt, fz]
  | m :: g => by
    have ih := fz_splitAt τ v g
    by_cases hv : v ∈ m
    · have h1 : (splitAt v (m :: g)).1 = m.filter (· != v) :: (splitAt v g).1 := by
        simp [splitAt, hv]
      have h2 : (splitAt v (m :: g)).2 = (splitAt v g).2 := by simp [splitAt, hv]
      rw [h1, h2, fz_cons, fz_cons, ih]
      have hm := mz_update_in τ v (τ v) m hv
      rw [Function.update_eq_self] at hm
      rw [hm]
      ring
    · have h1 : (splitAt v (m :: g)).1 = (splitAt v g).1 := by simp [splitAt, hv]
      have h2 : (splitAt v (m :: g)).2 = m :: (splitAt v g).2 := by simp [splitAt, hv]
      rw [h1, h2, fz_cons, fz_cons, ih]
      ring

/-- The value `A + B` of a determined variable `v` of the indicator `g = 1 + q`. -/
def detValue (v : ℕ) (q : F2) : F2 :=
  let g := normF ([] :: q)
  (splitAt v g).1 ++ (splitAt v g).2

/-- Whether `v`, not a path variable, is determined on the support of the first constraint:
`B (A + B) = 0` for its indicator `v A + B`. -/
def bcanonOK (v : ℕ) (s : Data) : Bool :=
  match s.cons with
  | [] => false
  | q :: _ =>
    let g := normF ([] :: q)
    !decide (v ∈ s.path) && decide (normF (f2mul (splitAt v g).2 (detValue v q)) = [])

/-- The first constraint's determined variable `v` made a row `v + A + B`, substituted out of
the phase, the constraints and the rows. -/
def bcanonS (M v : ℕ) (s : Data) : Data :=
  match s.cons with
  | [] => s
  | q :: rest =>
    let val := detValue v q
    { path := s.path, phase := normP M (substP M v val s.phase),
      cons := (q :: rest).map fun r => normF (substF v val r),
      rows := (s.rows.map fun r => normF (substF v val r)) ++ [normF ([v] :: val)] }

theorem z2_add_self : ∀ x : ZMod 2, x + x = 0 := by decide

theorem det_bits : ∀ x a b : ZMod 2, x * a + b = 1 → b * (a + b) = 0 → x = a + b := by decide

theorem bcanon_sound (M : ℕ) (hM : 2 ≤ M) (c : ℂ) (v : ℕ) (s : Data) (σ : ℕ → ZMod 2)
    (h : bcanonOK v s = true) : val M c s σ = val M c (bcanonS M v s) σ := by
  unfold val
  rcases hq : s.cons with _ | ⟨q, rest⟩
  · simp [bcanonOK, hq] at h
  · simp only [bcanonOK, hq, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at h
    obtain ⟨_, hdet⟩ := h
    simp only [bcanonS, hq]
    apply bsum_congr
    intro τ
    set val := detValue v q
    set g := normF ([] :: q)
    have hg : fz τ g = 1 + fz τ q := by
      rw [fz_normF, fz_cons]; rfl
    have hval : fz τ val = fz τ (splitAt v g).1 + fz τ (splitAt v g).2 := by
      simp only [val, detValue]; rw [fz_append]
    have hzero : fz τ (splitAt v g).2 * fz τ val = 0 := by
      have := congrArg (fz τ) hdet
      rw [fz_normF, fz_f2mul] at this
      simpa [fz] using this
    unfold summand
    simp only [hq]
    have hM0 : 0 < M := by omega
    by_cases hrow : τ v = fz τ val
    · have hτ : Function.update τ v (fz τ val) = τ := by rw [← hrow, Function.update_eq_self]
      have hsub : ∀ r, fz τ (normF (substF v val r)) = fz τ r := by
        intro r; rw [fz_normF, fz_substF, hτ]
      have hrowz : fz τ (normF ([v] :: val)) = 0 := by
        have e : mz τ [v] = τ v := by simp [mz]
        rw [fz_normF, fz_cons, e, hrow]
        exact z2_add_self _
      have hcond : (∀ r ∈ (q :: rest).map (fun r => normF (substF v val r)) ++
          ((s.rows.map fun r => normF (substF v val r)) ++ [normF ([v] :: val)]), fz τ r = 0) ↔
          (∀ r ∈ (q :: rest) ++ s.rows, fz τ r = 0) := by
        simp only [List.mem_append, List.mem_map, List.mem_singleton]
        constructor
        · rintro H r (hr | hr)
          · have := H _ (Or.inl ⟨r, hr, rfl⟩); rwa [hsub] at this
          · have := H _ (Or.inr (Or.inl ⟨r, hr, rfl⟩)); rwa [hsub] at this
        · rintro H r (⟨r', hr', rfl⟩ | ⟨r', hr', rfl⟩ | rfl)
          · rw [hsub]; exact H r' (Or.inl hr')
          · rw [hsub]; exact H r' (Or.inr hr')
          · exact hrowz
      simp only [hcond]
      split_ifs
      · congr 1
        apply ex_peq hM0
        have := peq_trans (pn_normP τ M _) (pn_substP τ M v hM val s.phase)
        rwa [hτ] at this
      · rfl
    · have hr : ¬ (∀ r ∈ (q :: rest).map (fun r => normF (substF v val r)) ++
          ((s.rows.map fun r => normF (substF v val r)) ++ [normF ([v] :: val)]),
            fz τ r = 0) := by
        intro H
        apply hrow
        have := H (normF ([v] :: val)) (by simp)
        rw [fz_normF, fz_cons] at this
        have e : mz τ [v] = τ v := by simp [mz]
        rw [e] at this
        exact z2_eq_of_add _ _ this
      have hl : ¬ (∀ r ∈ (q :: rest) ++ s.rows, fz τ r = 0) := by
        intro H
        apply hrow
        have hq0 : fz τ q = 0 := H q (by simp)
        have hg1 : fz τ g = 1 := by rw [hg, hq0, add_zero]
        rw [fz_splitAt τ v g] at hg1
        rw [hval]
        exact det_bits _ _ _ hg1 (by rw [← hval]; exact hzero)
      rw [if_neg hr, if_neg hl]

/-! ## Step lists -/

/-- One step of a replayed reduction. -/
inductive Step where
  | perm (l : List ℕ)
  | elim
  | hhCons
  | hhPivot
  | canon (v : ℕ)
  | gather
  | bcanon (v : ℕ)

/-- A step's side condition. -/
def stepOK (M : ℕ) : Step → Data → Bool
  | .perm l, s => decide (s.path.Perm l)
  | .elim, s => elimOK s
  | .hhCons, s => hhConsOK M s
  | .hhPivot, s => hhPivotOK M s
  | .canon v, s => canonOK v s
  | .gather, s => gatherOK s
  | .bcanon v, s => bcanonOK v s

/-- A step applied. -/
def stepS (M : ℕ) : Step → Data → Data
  | .perm l, s => permS l s
  | .elim, s => elimS s
  | .hhCons, s => hhConsS s
  | .hhPivot, s => hhPivotS M s
  | .canon v, s => canonS M v s
  | .gather, s => gatherS s
  | .bcanon v, s => bcanonS M v s

/-- The power of 2 a step multiplies the constant by. -/
def stepTwo : Step → ℕ
  | .elim | .hhCons | .hhPivot => 1
  | _ => 0

/-- Every side condition along a step list. -/
def runOK (M : ℕ) : List Step → Data → Bool
  | [], _ => true
  | st :: sts, s => stepOK M st s && runOK M sts (stepS M st s)

/-- A step list applied. -/
def run (M : ℕ) : List Step → Data → Data
  | [], s => s
  | st :: sts, s => run M sts (stepS M st s)

/-- The power of 2 a step list multiplies the constant by. -/
def twos : List Step → ℕ
  | [] => 0
  | st :: sts => stepTwo st + twos sts

theorem step_sound (M : ℕ) (hM : 2 ≤ M) (st : Step) (s : Data) (c : ℂ) (σ : ℕ → ZMod 2)
    (h : stepOK M st s = true) : val M c s σ = val M (2 ^ stepTwo st * c) (stepS M st s) σ := by
  cases st with
  | perm l => simpa [stepTwo, stepS] using perm_sound M c l s σ h
  | elim => simpa [stepTwo, stepS] using elim_sound M c s σ h
  | hhCons => simpa [stepTwo, stepS] using hhCons_sound M c s σ h
  | hhPivot => simpa [stepTwo, stepS] using hhPivot_sound M c s σ h
  | canon v => simpa [stepTwo, stepS] using canon_sound M hM c v s σ h
  | gather => simpa [stepTwo, stepS] using gather_sound M c s σ h
  | bcanon v => simpa [stepTwo, stepS] using bcanon_sound M hM c v s σ h

theorem run_sound (M : ℕ) (hM : 2 ≤ M) :
    ∀ (sts : List Step) (s : Data) (c : ℂ) (σ : ℕ → ZMod 2), runOK M sts s = true →
      val M c s σ = val M (2 ^ twos sts * c) (run M sts s) σ
  | [], s, c, σ, _ => by simp [twos, run]
  | st :: sts, s, c, σ, h => by
    simp only [runOK, Bool.and_eq_true] at h
    rw [step_sound M hM st s c σ h.1, run_sound M hM sts _ _ σ h.2]
    simp only [twos, run, pow_add]
    ring_nf

end PSem
end ArchytasZX
