/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSum
import ArchytasZX.Ball

/-!
# Path-sum values and step soundness

A path sum over Boolean variables `σ : ℕ → ZMod 2` is `Data`: path variables, phase terms
`(monomial, k)` worth `k / M` turns, constraints and support rows (F_2 polynomials, sums of
monomials). `val M c s σ` sums `summand` over the path variables. `elimS`, `hhPivotS`,
`hhConsS`, `canonS`, `gatherS` and `bcanonS` compute one rule step, and each `*_sound`
theorem says the value is unchanged under the step's syntactic side condition, a `Bool`
checked by `decide`. No lemma enumerates the path variables.
-/

namespace ArchytasZX
namespace PSem

open Complex
open ArchytasZX.Sym (ex_half)

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

/-! ## Bitmask monomials -/

/-- The mask of `m` over `vs`: bit `i` set when the `i`-th entry of `vs` is in `m`. -/
def toM : List ℕ → Mono → ℕ
  | [], _ => 0
  | v :: vs, m => (if v ∈ m then 1 else 0) + 2 * toM vs m

/-- The monomial of the mask `k` over `vs`: the entries of `vs` at the set bits of `k`. -/
def ofM : List ℕ → ℕ → Mono
  | [], _ => []
  | v :: vs, k => if k % 2 = 1 then v :: ofM vs (k / 2) else ofM vs (k / 2)

theorem mem_ofM_toM (w : ℕ) (m : Mono) : ∀ vs : List ℕ, w ∈ ofM vs (toM vs m) ↔ w ∈ vs ∧ w ∈ m
  | [] => by simp [ofM]
  | v :: vs => by
    have ih := mem_ofM_toM w m vs
    by_cases hv : v ∈ m
    · have h1 : (1 + 2 * toM vs m) % 2 = 1 := by omega
      have h2 : (1 + 2 * toM vs m) / 2 = toM vs m := by omega
      simp only [toM, hv, ↓reduceIte, ofM, h1, h2, List.mem_cons, ih]
      constructor
      · rintro (rfl | ⟨a, b⟩)
        · exact ⟨Or.inl rfl, hv⟩
        · exact ⟨Or.inr a, b⟩
      · rintro ⟨rfl | a, b⟩
        · exact Or.inl rfl
        · exact Or.inr ⟨a, b⟩
    · have h1 : ¬ (0 + 2 * toM vs m) % 2 = 1 := by omega
      have h2 : (0 + 2 * toM vs m) / 2 = toM vs m := by omega
      simp only [toM, hv, ↓reduceIte, ofM, h1, h2, List.mem_cons, ih]
      constructor
      · rintro ⟨a, b⟩
        exact ⟨Or.inr a, b⟩
      · rintro ⟨rfl | a, b⟩
        · exact absurd b hv
        · exact ⟨a, b⟩

theorem or_div_two (a b : ℕ) : (a ||| b) / 2 = a / 2 ||| b / 2 := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [← Nat.testBit_succ, Nat.testBit_or, Nat.testBit_or, Nat.testBit_succ, Nat.testBit_succ]

theorem mem_ofM_or (w : ℕ) : ∀ (vs : List ℕ) (a b : ℕ),
    w ∈ ofM vs (a ||| b) ↔ w ∈ ofM vs a ∨ w ∈ ofM vs b
  | [], _, _ => by simp [ofM]
  | v :: vs, a, b => by
    have ih := mem_ofM_or w vs (a / 2) (b / 2)
    simp only [ofM, or_div_two]
    by_cases ha : a % 2 = 1 <;> by_cases hb : b % 2 = 1 <;>
      simp [ha, hb, ih] <;> tauto

theorem ofM_zero : ∀ vs : List ℕ, ofM vs 0 = []
  | [] => rfl
  | _ :: vs => by simp [ofM, ofM_zero vs]

theorem mz_dedup (σ : ℕ → ZMod 2) : ∀ l : Mono, mz σ l.dedup = mz σ l
  | [] => rfl
  | a :: l => by
    by_cases h : a ∈ l
    · rw [List.dedup_cons_of_mem h, mz_dedup σ l]
      show mz σ l = σ a * mz σ l
      have hp : mz σ l = σ a * mz σ (l.erase a) := ((List.perm_cons_erase h).map σ).prod_eq
      rw [hp, ← mul_assoc, z2_idem]
    · rw [List.dedup_cons_of_notMem h]
      show σ a * mz σ l.dedup = σ a * mz σ l
      rw [mz_dedup σ l]

theorem mz_congr (σ : ℕ → ZMod 2) {l l' : Mono} (h : ∀ v, v ∈ l ↔ v ∈ l') :
    mz σ l = mz σ l' := by
  rw [← mz_dedup σ l, ← mz_dedup σ l']
  have hp : l.dedup.Perm l'.dedup :=
    (List.perm_ext_iff_of_nodup (List.nodup_dedup _) (List.nodup_dedup _)).2
      fun v => by rw [List.mem_dedup, List.mem_dedup]; exact h v
  exact (hp.map σ).prod_eq

theorem mz_ofM_toM (σ : ℕ → ZMod 2) {vs : List ℕ} {m : Mono} (h : ∀ v ∈ m, v ∈ vs) :
    mz σ (ofM vs (toM vs m)) = mz σ m :=
  mz_congr σ fun v => by
    rw [mem_ofM_toM]
    exact ⟨fun h' => h'.2, fun h' => ⟨h v h', h'⟩⟩

theorem mz_ofM_or (σ : ℕ → ZMod 2) (vs : List ℕ) (a b : ℕ) :
    mz σ (ofM vs (a ||| b)) = mz σ (ofM vs a) * mz σ (ofM vs b) := by
  rw [← mz_append]
  apply mz_congr
  intro v
  rw [mem_ofM_or, List.mem_append]

/-- `mle` on the monomials of two masks over one ascending variable list. -/
def mleM (a b : ℕ) : Bool :=
  a == b ||
    let d := a ^^^ b
    let t := d &&& (d ^^^ (d - 1))
    if a &&& t = 0 then decide (a < t) else decide (t ≤ b)

/-! ## Normalisation -/

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
      exact ⟨n, by simp [pn] at *; linarith⟩
    · rw [pn_cons, pn_nil]
      obtain ⟨n, hn⟩ := peq_mod M t.2 (zq (mz σ t.1))
      exact ⟨n, by simp [pn] at *; push_cast at *; linarith⟩
  | u :: P => by
    simp only [ins]
    split_ifs with h1 h2
    · obtain ⟨n, hn⟩ := peq_mod M (u.2 + t.2) (zq (mz σ t.1))
      rw [h2] at hn
      refine ⟨n, ?_⟩
      rw [pn_cons, h1]
      push_cast at hn
      linear_combination hn
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

/-- `xs` and `ys` merged by `le`, at most `n` heads compared, the rest appended. -/
def mergeN {α : Type} (le : α → α → Bool) : ℕ → List α → List α → List α
  | 0, xs, ys => xs ++ ys
  | _ + 1, [], ys => ys
  | _ + 1, x :: xs, [] => x :: xs
  | n + 1, x :: xs, y :: ys =>
    if le x y then x :: mergeN le n xs (y :: ys) else y :: mergeN le n (x :: xs) ys

theorem mergeN_perm {α : Type} (le : α → α → Bool) :
    ∀ (n : ℕ) (xs ys : List α), (mergeN le n xs ys).Perm (xs ++ ys)
  | 0, _, _ => List.Perm.refl _
  | _ + 1, [], _ => List.Perm.refl _
  | _ + 1, _ :: _, [] => by simp [mergeN]
  | n + 1, x :: xs, y :: ys => by
    simp only [mergeN]
    split
    · exact (mergeN_perm le n xs (y :: ys)).cons x
    · exact ((mergeN_perm le n (x :: xs) ys).cons y).trans List.perm_middle.symm

/-- The entries at even and at odd positions. -/
def halve {α : Type} : List α → List α × List α
  | [] => ([], [])
  | [a] => ([a], [])
  | a :: b :: l => (a :: (halve l).1, b :: (halve l).2)

theorem halve_perm {α : Type} : ∀ l : List α, ((halve l).1 ++ (halve l).2).Perm l
  | [] => List.Perm.refl _
  | [_] => by simp [halve]
  | a :: b :: l => by
    simp only [halve, List.cons_append]
    exact (List.perm_middle.trans ((halve_perm l).cons b)).cons a

/-- Merge sort by `le`, at most `n` levels deep. -/
def msortN {α : Type} (le : α → α → Bool) : ℕ → List α → List α
  | 0, l => l
  | _ + 1, [] => []
  | _ + 1, [a] => [a]
  | n + 1, a :: b :: l =>
    mergeN le (l.length + 2) (msortN le n (halve (a :: b :: l)).1)
      (msortN le n (halve (a :: b :: l)).2)

theorem msortN_perm {α : Type} (le : α → α → Bool) :
    ∀ (n : ℕ) (l : List α), (msortN le n l).Perm l
  | 0, _ => List.Perm.refl _
  | _ + 1, [] => List.Perm.refl _
  | _ + 1, [_] => List.Perm.refl _
  | n + 1, a :: b :: l => by
    simp only [msortN]
    exact (mergeN_perm le _ _ _).trans
      (((msortN_perm le n _).append (msortN_perm le n _)).trans (halve_perm _))

/-- Merge sort by `le`. -/
def sortB {α : Type} (le : α → α → Bool) (l : List α) : List α := msortN le l.length l

theorem sortB_perm {α : Type} (le : α → α → Bool) (l : List α) : (sortB le l).Perm l :=
  msortN_perm le _ l

/-- `v` inserted into an ascending list without repeats. -/
def insU (v : ℕ) : List ℕ → List ℕ
  | [] => [v]
  | a :: l => if v < a then v :: a :: l else if v = a then a :: l else a :: insU v l

theorem mem_insU (w v : ℕ) : ∀ l : List ℕ, w ∈ insU v l ↔ w = v ∨ w ∈ l
  | [] => by simp [insU]
  | a :: l => by
    have ih := mem_insU w v l
    simp only [insU]
    split_ifs with h1 h2
    · simp
    · subst h2
      simp only [List.mem_cons]
      tauto
    · simp only [List.mem_cons, ih]
      tauto

theorem mem_foldr_insU (w : ℕ) (acc : List ℕ) : ∀ m : Mono,
    w ∈ m.foldr insU acc ↔ w ∈ m ∨ w ∈ acc
  | [] => by simp
  | v :: m => by
    rw [List.foldr_cons, mem_insU, mem_foldr_insU w acc m, List.mem_cons]
    tauto

/-- The variables of a list of monomials, ascending, without repeats. -/
def vlist (ms : List Mono) : List ℕ := ms.foldr (fun m acc => m.foldr insU acc) []

theorem mem_vlist {ms : List Mono} {m : Mono} (hm : m ∈ ms) : ∀ v ∈ m, v ∈ vlist ms := by
  intro v hv
  induction ms with
  | nil => exact absurd hm List.not_mem_nil
  | cons a ms ih =>
    rw [vlist, List.foldr_cons, mem_foldr_insU]
    rcases List.mem_cons.1 hm with rfl | hm
    · exact Or.inl hv
    · exact Or.inr (ih hm)

/-- `t` added to the head term of `P` when their monomials agree, else consed; zero terms
dropped. -/
def insH (M : ℕ) (t : Mono × ℕ) : Poly → Poly
  | [] => if t.2 % M = 0 then [] else [(t.1, t.2 % M)]
  | u :: P =>
    if u.1 = t.1 then (if (u.2 + t.2) % M = 0 then P else (u.1, (u.2 + t.2) % M) :: P)
    else if t.2 % M = 0 then u :: P else (t.1, t.2 % M) :: u :: P

theorem pn_insH (σ : ℕ → ZMod 2) (M : ℕ) (t : Mono × ℕ) :
    ∀ P : Poly, peq M (pn σ (insH M t P)) ((t.2 : ℤ) * zq (mz σ t.1) + pn σ P)
  | [] => pn_ins σ M t []
  | u :: P => by
    simp only [insH]
    split_ifs with h1 h2 h3
    · obtain ⟨n, hn⟩ := peq_mod M (u.2 + t.2) (zq (mz σ t.1))
      rw [h2] at hn
      refine ⟨n, ?_⟩
      rw [pn_cons, h1]
      push_cast at hn
      linear_combination hn
    · obtain ⟨n, hn⟩ := peq_mod M (u.2 + t.2) (zq (mz σ t.1))
      refine ⟨n, ?_⟩
      rw [pn_cons, pn_cons, h1]
      push_cast at hn ⊢
      linarith
    · obtain ⟨n, hn⟩ := peq_mod M t.2 (zq (mz σ t.1))
      rw [h3] at hn
      refine ⟨n, ?_⟩
      push_cast at hn
      linear_combination hn
    · obtain ⟨n, hn⟩ := peq_mod M t.2 (zq (mz σ t.1))
      refine ⟨n, ?_⟩
      rw [pn_cons (t := (t.1, t.2 % M))]
      dsimp only
      push_cast at hn ⊢
      linear_combination hn

theorem pn_foldr_insH (σ : ℕ → ZMod 2) (M : ℕ) :
    ∀ L : Poly, peq M (pn σ (L.foldr (insH M) [])) (pn σ L)
  | [] => peq_refl M 0
  | t :: L => by
    rw [List.foldr_cons]
    refine peq_trans (pn_insH σ M t _) ?_
    rw [pn_cons]
    exact peq_add (peq_refl M _) (pn_foldr_insH σ M L)

/-- A mask term over `vs` as a monomial term. -/
def conv (vs : List ℕ) (t : ℕ × ℕ) : Mono × ℕ := (ofM vs t.1, t.2)

/-- A phase polynomial over masks of `vs`, times M. -/
def pnM (σ : ℕ → ZMod 2) (vs : List ℕ) (P : List (ℕ × ℕ)) : ℤ := pn σ (P.map (conv vs))

theorem pnM_nil (σ : ℕ → ZMod 2) (vs : List ℕ) : pnM σ vs [] = 0 := rfl

theorem pnM_cons (σ : ℕ → ZMod 2) (vs : List ℕ) (t : ℕ × ℕ) (P : List (ℕ × ℕ)) :
    pnM σ vs (t :: P) = (t.2 : ℤ) * zq (mz σ (ofM vs t.1)) + pnM σ vs P := by
  unfold pnM
  rw [List.map_cons, pn_cons]
  rfl

theorem pnM_append (σ : ℕ → ZMod 2) (vs : List ℕ) (P R : List (ℕ × ℕ)) :
    pnM σ vs (P ++ R) = pnM σ vs P + pnM σ vs R := by
  unfold pnM
  rw [List.map_append, pn_append]

/-- `insH` on mask terms. -/
def insHM (M : ℕ) (t : ℕ × ℕ) : List (ℕ × ℕ) → List (ℕ × ℕ)
  | [] => if t.2 % M = 0 then [] else [(t.1, t.2 % M)]
  | u :: P =>
    if u.1 = t.1 then (if (u.2 + t.2) % M = 0 then P else (u.1, (u.2 + t.2) % M) :: P)
    else if t.2 % M = 0 then u :: P else (t.1, t.2 % M) :: u :: P

theorem pnM_insHM (σ : ℕ → ZMod 2) (vs : List ℕ) (M : ℕ) (t : ℕ × ℕ) :
    ∀ P : List (ℕ × ℕ),
      peq M (pnM σ vs (insHM M t P)) ((t.2 : ℤ) * zq (mz σ (ofM vs t.1)) + pnM σ vs P)
  | [] => by
    simp only [insHM]
    obtain ⟨n, hn⟩ := peq_mod M t.2 (zq (mz σ (ofM vs t.1)))
    split_ifs with h
    · rw [h] at hn
      refine ⟨n, ?_⟩
      rw [pnM_nil]
      push_cast at hn
      linear_combination hn
    · refine ⟨n, ?_⟩
      rw [pnM_cons, pnM_nil]
      dsimp only
      push_cast at hn ⊢
      linear_combination hn
  | u :: P => by
    simp only [insHM]
    split_ifs with h1 h2 h3
    · obtain ⟨n, hn⟩ := peq_mod M (u.2 + t.2) (zq (mz σ (ofM vs t.1)))
      rw [h2] at hn
      refine ⟨n, ?_⟩
      rw [pnM_cons, h1]
      push_cast at hn
      linear_combination hn
    · obtain ⟨n, hn⟩ := peq_mod M (u.2 + t.2) (zq (mz σ (ofM vs t.1)))
      refine ⟨n, ?_⟩
      rw [pnM_cons, pnM_cons, h1]
      push_cast at hn ⊢
      linarith
    · obtain ⟨n, hn⟩ := peq_mod M t.2 (zq (mz σ (ofM vs t.1)))
      rw [h3] at hn
      refine ⟨n, ?_⟩
      push_cast at hn
      linear_combination hn
    · obtain ⟨n, hn⟩ := peq_mod M t.2 (zq (mz σ (ofM vs t.1)))
      refine ⟨n, ?_⟩
      rw [pnM_cons σ vs (t.1, t.2 % M)]
      dsimp only
      push_cast at hn ⊢
      linear_combination hn

theorem pnM_foldr_insHM (σ : ℕ → ZMod 2) (vs : List ℕ) (M : ℕ) :
    ∀ L : List (ℕ × ℕ), peq M (pnM σ vs (L.foldr (insHM M) [])) (pnM σ vs L)
  | [] => peq_refl M 0
  | t :: L => by
    rw [List.foldr_cons]
    refine peq_trans (pnM_insHM σ vs M t _) ?_
    rw [pnM_cons]
    exact peq_add (peq_refl M _) (pnM_foldr_insHM σ vs M L)

/-- A mask phase polynomial sorted by `mleM` (merge sort), terms of one mask merged mod M and
zero terms dropped. -/
def normPM (M : ℕ) (P : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  (sortB (fun a b => mleM a.1 b.1) P).foldr (insHM M) []

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
      simp only [squash, ↓reduceIte]
      rw [ih]
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

/-- A phase polynomial with sorted monomials, sorted by `mle` (merge sort), terms of one
monomial merged mod M and zero terms dropped. -/
def normP (M : ℕ) (P : Poly) : Poly :=
  (sortB (fun a b => mle a.1 b.1) (P.map fun t => (mnorm t.1, t.2))).foldr (insH M) []

theorem pn_perm (σ : ℕ → ZMod 2) {P R : Poly} (h : P.Perm R) : pn σ P = pn σ R :=
  (h.map _).sum_eq

theorem pnM_normPM (σ : ℕ → ZMod 2) (vs : List ℕ) (M : ℕ) (P : List (ℕ × ℕ)) :
    peq M (pnM σ vs (normPM M P)) (pnM σ vs P) := by
  unfold normPM
  refine peq_trans (pnM_foldr_insHM σ vs M _) ?_
  unfold pnM
  rw [pn_perm σ ((sortB_perm _ _).map (conv vs))]
  exact peq_refl M _

theorem pn_normP (σ : ℕ → ZMod 2) (M : ℕ) (P : Poly) : peq M (pn σ (normP M P)) (pn σ P) := by
  unfold normP
  refine peq_trans (pn_foldr_insH σ M _) ?_
  rw [pn_perm σ (sortB_perm _ _)]
  have hm : pn σ (P.map fun t => (mnorm t.1, t.2)) = pn σ P := by
    induction P with
    | nil => rfl
    | cons t P ih => rw [List.map_cons, pn_cons, pn_cons, ih, mz_mnorm]
  rw [hm]
  exact peq_refl M _

/-- `m` cancelled against the head monomial of `Q` when equal, else consed. -/
def cancelH (m : Mono) : F2 → F2
  | [] => [m]
  | u :: Q => if u = m then Q else m :: u :: Q

theorem fz_perm (σ : ℕ → ZMod 2) {P Q : F2} (h : P.Perm Q) : fz σ P = fz σ Q :=
  (h.map _).sum_eq

theorem fz_cancelH (σ : ℕ → ZMod 2) (m : Mono) :
    ∀ Q : F2, fz σ (cancelH m Q) = mz σ m + fz σ Q
  | [] => by simp [cancelH, fz]
  | u :: Q => by
    simp only [cancelH]
    split_ifs with h
    · subst h
      rw [fz_cons, ← add_assoc, ← two_mul]
      have : (2 : ZMod 2) = 0 := by decide
      rw [this, zero_mul, zero_add]
    · rw [fz_cons]

theorem fz_foldr_cancelH (σ : ℕ → ZMod 2) : ∀ Q : F2, fz σ (Q.foldr cancelH []) = fz σ Q
  | [] => rfl
  | m :: Q => by rw [List.foldr_cons, fz_cancelH, fz_foldr_cancelH σ Q, fz_cons]

/-- `cancelH` on masks. -/
def cancelHM (m : ℕ) : List ℕ → List ℕ
  | [] => [m]
  | u :: Q => if u = m then Q else m :: u :: Q

theorem fz_cancelHM (σ : ℕ → ZMod 2) (vs : List ℕ) (m : ℕ) : ∀ Q : List ℕ,
    fz σ ((cancelHM m Q).map (ofM vs)) = mz σ (ofM vs m) + fz σ (Q.map (ofM vs))
  | [] => by simp [cancelHM, fz]
  | u :: Q => by
    simp only [cancelHM]
    split_ifs with h
    · subst h
      rw [List.map_cons, fz_cons, ← add_assoc, ← two_mul]
      have : (2 : ZMod 2) = 0 := by decide
      rw [this, zero_mul, zero_add]
    · rw [List.map_cons, fz_cons]

theorem fz_foldr_cancelHM (σ : ℕ → ZMod 2) (vs : List ℕ) : ∀ Q : List ℕ,
    fz σ ((Q.foldr cancelHM []).map (ofM vs)) = fz σ (Q.map (ofM vs))
  | [] => rfl
  | m :: Q => by
    rw [List.foldr_cons, fz_cancelHM, fz_foldr_cancelHM σ vs Q, List.map_cons, fz_cons]

/-- The xor of `2 ^ k` over the masks `k` of `Q`. -/
def xorF (Q : List ℕ) : ℕ := Q.foldr (fun k acc => acc ^^^ 2 ^ k) 0

/-- The or of the masks of `Q`. -/
def orF (Q : List ℕ) : ℕ := Q.foldr (fun k acc => k ||| acc) 0

/-- `0` and the `or` of every nonempty sublist of `bs`; in `mleM` order when `bs` holds
ascending powers of two. -/
def subsM : List ℕ → List ℕ
  | [] => [0]
  | b :: R => 0 :: ((subsM R).map (b ||| ·) ++ (subsM R).tail)

/-- The powers of two of the set bits of `m`, ascending. -/
def bitsOf (m : ℕ) : List ℕ :=
  (List.range (Nat.log2 m + 1)).filterMap fun i => if m.testBit i then some (2 ^ i) else none

/-- The submasks `k` of `b` with bit `k` of `x` set. -/
def fromX (x b : ℕ) : List ℕ := (subsM (bitsOf b)).filter fun k => x.testBit k

/-- `fromX (x ()) b` when `b < 4 n + 64` and `xorF (fromX (x ()) b) = x ()`, else `R`; `x` is
evaluated only when `b < 4 n + 64`. -/
def pickX (x : Unit → ℕ) (b n : ℕ) (R : List ℕ) : List ℕ :=
  if (decide (b < 4 * n + 64) && decide (xorF (fromX (x ()) b) = x ())) = true then
    fromX (x ()) b
  else R

/-- The masks of odd multiplicity in `Q`, sorted by `mleM`: the submasks of `orF Q` in `mleM`
order with their bit of `xorF Q` set, else the set bits of `xorF Q` up to `orF Q` merge
sorted. -/
def normFD (Q : List ℕ) : List ℕ :=
  pickX (fun _ => xorF Q) (orF Q) Q.length
    (sortB mleM ((List.range (orF Q + 1)).filter fun k => (xorF Q).testBit k))

/-- A mask F_2 polynomial sorted by `mleM`, equal pairs cancelled: through `xorF` when
`orF Q` is below `4 * Q.length + 64`, else by merge sort. -/
def normFM (Q : List ℕ) : List ℕ :=
  if orF Q < 4 * Q.length + 64 then normFD Q else (sortB mleM Q).foldr cancelHM []

theorem sum_filter_ite {f : ℕ → ZMod 2} (p : ℕ → Bool) : ∀ l : List ℕ,
    ((l.filter p).map f).sum = (l.map fun x => if p x then f x else 0).sum
  | [] => rfl
  | a :: l => by
    by_cases h : p a <;> simp [List.filter_cons, h, sum_filter_ite p l]

theorem sum_ite_eq_none {f : ℕ → ZMod 2} (k : ℕ) : ∀ l : List ℕ, k ∉ l →
    (l.map fun x => if x = k then f x else 0).sum = 0
  | [], _ => rfl
  | b :: l, hk => by
    have hb : b ≠ k := fun e => hk (e ▸ List.mem_cons_self)
    rw [List.map_cons, List.sum_cons, if_neg hb,
      sum_ite_eq_none k l (fun e => hk (List.mem_cons_of_mem _ e)), add_zero]

theorem sum_ite_eq_self {f : ℕ → ZMod 2} (k : ℕ) : ∀ l : List ℕ, l.Nodup → k ∈ l →
    (l.map fun x => if x = k then f x else 0).sum = f k
  | [], _, h => absurd h List.not_mem_nil
  | a :: l, hn, h => by
    rw [List.map_cons, List.sum_cons]
    by_cases hak : a = k
    · subst hak
      rw [if_pos rfl, sum_ite_eq_none a l (List.nodup_cons.1 hn).1, add_zero]
    · rw [if_neg hak, zero_add]
      exact sum_ite_eq_self k l (List.nodup_cons.1 hn).2
        ((List.mem_cons.1 h).resolve_left (Ne.symm hak))

theorem testBit_xorF (k x : ℕ) (Q : List ℕ) :
    (xorF (k :: Q)).testBit x = ((xorF Q).testBit x ^^ decide (k = x)) := by
  simp only [xorF, List.foldr_cons, Nat.testBit_xor, Nat.testBit_two_pow]

theorem sum_xorF {f : ℕ → ZMod 2} (cand : List ℕ) (hc : cand.Nodup) : ∀ Q : List ℕ,
    (∀ k ∈ Q, k ∈ cand) →
      ((cand.filter fun k => (xorF Q).testBit k).map f).sum = (Q.map f).sum
  | [], _ => by simp [xorF]
  | k :: Q, h => by
    have ih := sum_xorF (f := f) cand hc Q (fun x hx => h x (List.mem_cons_of_mem _ hx))
    rw [sum_filter_ite] at ih ⊢
    have e : ∀ x, (if (xorF (k :: Q)).testBit x then f x else 0) =
        (if (xorF Q).testBit x then f x else 0) + (if x = k then f x else 0) := by
      intro x
      rw [testBit_xorF]
      by_cases h1 : (xorF Q).testBit x <;> by_cases h2 : x = k
      · subst h2
        simp only [h1, Bool.true_xor, decide_true, Bool.not_true, ↓reduceIte]
        exact (show ∀ y : ZMod 2, 0 = y + y by decide) _
      · simp [h1, h2, Ne.symm h2]
      · subst h2
        simp [h1]
      · simp [h1, h2, Ne.symm h2]
    simp only [e]
    rw [List.sum_map_add, ih, sum_ite_eq_self k cand hc (h k List.mem_cons_self),
      List.map_cons, List.sum_cons, add_comm]

theorem le_orF : ∀ (Q : List ℕ) (k : ℕ), k ∈ Q → k ≤ orF Q
  | [], _, h => absurd h List.not_mem_nil
  | a :: Q, k, h => by
    rw [orF, List.foldr_cons]
    rcases List.mem_cons.1 h with rfl | h
    · exact Nat.left_le_or
    · exact le_trans (le_orF Q k h) Nat.right_le_or

theorem fz_xorF_eq (σ : ℕ → ZMod 2) (vs : List ℕ) {R Q : List ℕ} (h : xorF R = xorF Q) :
    fz σ (R.map (ofM vs)) = fz σ (Q.map (ofM vs)) := by
  have hR := sum_xorF (f := fun k => mz σ (ofM vs k)) (List.range (orF R + orF Q + 1))
    List.nodup_range R (fun k hk => List.mem_range.2 (by have := le_orF R k hk; omega))
  have hQ := sum_xorF (f := fun k => mz σ (ofM vs k)) (List.range (orF R + orF Q + 1))
    List.nodup_range Q (fun k hk => List.mem_range.2 (by have := le_orF Q k hk; omega))
  unfold fz
  rw [List.map_map, List.map_map]
  exact hR.symm.trans (h ▸ hQ)

theorem fz_pickX (σ : ℕ → ZMod 2) (vs : List ℕ) (x : Unit → ℕ) (b n : ℕ) (R Q : List ℕ)
    (hx : x () = xorF Q) (hR : fz σ (R.map (ofM vs)) = fz σ (Q.map (ofM vs))) :
    fz σ ((pickX x b n R).map (ofM vs)) = fz σ (Q.map (ofM vs)) := by
  unfold pickX
  split_ifs with h
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact fz_xorF_eq σ vs (h.2.trans hx)
  · exact hR

/-- An F_2 polynomial with sorted monomials, sorted by `mle` (merge sort), equal pairs
cancelled, on lists. -/
def normFL (Q : F2) : F2 := (sortB (fun a b => mle a b) (Q.map mnorm)).foldr cancelH []

/-- An F_2 polynomial with sorted monomials, sorted by `mle` order, equal pairs cancelled:
`normFL` below 32 monomials, else on masks. -/
def normF (Q : F2) : F2 :=
  if Q.length < 32 then normFL Q else
    let vs := vlist Q
    (normFM (Q.map (toM vs))).map (ofM vs)

theorem fz_normFM (σ : ℕ → ZMod 2) (vs : List ℕ) (Q : List ℕ) :
    fz σ ((normFM Q).map (ofM vs)) = fz σ (Q.map (ofM vs)) := by
  unfold normFM
  split_ifs
  · unfold normFD
    apply fz_pickX σ vs (fun _ => xorF Q) _ _ _ Q rfl
    rw [fz_perm σ ((sortB_perm _ _).map (ofM vs))]
    unfold fz
    rw [List.map_map, List.map_map]
    exact sum_xorF _ List.nodup_range Q
      (fun k hk => List.mem_range.2 (Nat.lt_succ_of_le (le_orF Q k hk)))
  · rw [fz_foldr_cancelHM, fz_perm σ ((sortB_perm _ _).map (ofM vs))]

theorem fz_toM (σ : ℕ → ZMod 2) (vs : List ℕ) : ∀ Q : F2, (∀ m ∈ Q, ∀ v ∈ m, v ∈ vs) →
    fz σ ((Q.map (toM vs)).map (ofM vs)) = fz σ Q
  | [] => fun _ => rfl
  | m :: Q => fun h => by
    rw [List.map_cons, List.map_cons, fz_cons, fz_cons,
      fz_toM σ vs Q (fun u hu => h u (List.mem_cons_of_mem _ hu)),
      mz_ofM_toM σ (h m List.mem_cons_self)]

theorem fz_normF (σ : ℕ → ZMod 2) (Q : F2) : fz σ (normF Q) = fz σ Q := by
  unfold normF
  split_ifs with h
  · clear h
    unfold normFL
    rw [fz_foldr_cancelH, fz_perm σ (sortB_perm _ _)]
    induction Q with
    | nil => rfl
    | cons m Q ih => rw [List.map_cons, fz_cons, fz_cons, ih, mz_mnorm]
  · rw [fz_normFM, fz_toM σ _ Q (fun m hm => mem_vlist hm)]

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

/-- `c` times `lift M` of a mask polynomial, normalised mod M after each mask. -/
def liftNM (M c : ℕ) : List ℕ → List (ℕ × ℕ)
  | [] => []
  | m :: B =>
    let L := liftNM M c B
    normPM M ((m, c) :: (L ++ L.map fun t => (m ||| t.1, (M - 2) * t.2)))

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

/-! ## The dense lift -/

/-- A table over a list of variables: `node lo hi` holds `lo` where the head variable is 0 and
`hi` where it is 1. -/
inductive Tb where
  | leaf (k : ℕ)
  | node (lo hi : Tb)

/-- Whether `t` is a complete tree of depth `n`. -/
def Tb.dep : ℕ → Tb → Prop
  | 0, .leaf _ => True
  | n + 1, .node l h => Tb.dep n l ∧ Tb.dep n h
  | _, _ => False

/-- The entry of `t` at the assignment `σ` of `vs`. -/
def Tb.at (σ : ℕ → ZMod 2) : List ℕ → Tb → ℕ
  | _, .leaf k => k
  | [], .node l _ => Tb.at σ [] l
  | v :: vs, .node l h => if σ v = 0 then Tb.at σ vs l else Tb.at σ vs h

/-- Leafwise `f` on two trees of one shape. -/
def Tb.zip (f : ℕ → ℕ → ℕ) : Tb → Tb → Tb
  | .leaf a, .leaf b => .leaf (f a b)
  | .node l h, .node l' h' => .node (Tb.zip f l l') (Tb.zip f h h')
  | t, _ => t

/-- Every leaf times `c`. -/
def Tb.scale (c : ℕ) : Tb → Tb
  | .leaf k => .leaf (c * k)
  | .node l h => .node (Tb.scale c l) (Tb.scale c h)

/-- `node a (a + b mod 2)`. -/
def Tb.xorN (a b : Tb) : Tb := .node a (Tb.zip (fun x y => (x + y) % 2) a b)

/-- `node a (b - a mod M)`. -/
def Tb.subN (M : ℕ) (a b : Tb) : Tb := .node a (Tb.zip (fun x y => x + (M - y % M)) b a)

/-- The values 0 or 1 of the F_2 polynomial `B` on the assignments of `vs`. -/
def tabF : List ℕ → F2 → Tb
  | [], B => .leaf (B.length % 2)
  | v :: vs, B => Tb.xorN (tabF vs (splitAt v B).2) (tabF vs (splitAt v B).1)

/-- The multilinear coefficients mod `M` of a table (Moebius transform). -/
def Tb.mob (M : ℕ) : Tb → Tb
  | .leaf k => .leaf k
  | .node l h => Tb.subN M (Tb.mob M l) (Tb.mob M h)

/-- The terms of the leaves nonzero mod `M` of a coefficient table over `vs`, each monomial the
variables at 1. -/
def Tb.poly (M : ℕ) : List ℕ → Tb → Poly
  | _, .leaf k => if k % M = 0 then [] else [([], k % M)]
  | [], .node l _ => Tb.poly M [] l
  | v :: vs, .node l h => Tb.poly M vs l ++ (Tb.poly M vs h).map fun t => (v :: t.1, t.2)

/-- `c` times `lift M B` through the table of `B` on `vs`. -/
def liftD (M c : ℕ) (vs : List ℕ) (B : F2) : Poly :=
  Tb.poly M vs (Tb.mob M (Tb.scale c (tabF vs B)))

theorem Tb.dep_zip (f : ℕ → ℕ → ℕ) : ∀ (n : ℕ) (a b : Tb), Tb.dep n a → Tb.dep n b →
    Tb.dep n (Tb.zip f a b)
  | 0, .leaf _, .leaf _, _, _ => by simp [Tb.zip, Tb.dep]
  | 0, .node _ _, _, ha, _ => by simp [Tb.dep] at ha
  | 0, .leaf _, .node _ _, _, hb => by simp [Tb.dep] at hb
  | _ + 1, .leaf _, _, ha, _ => by simp [Tb.dep] at ha
  | _ + 1, .node _ _, .leaf _, _, hb => by simp [Tb.dep] at hb
  | n + 1, .node l h, .node l' h', ha, hb => by
    simp only [Tb.dep] at ha hb
    simp only [Tb.zip, Tb.dep]
    exact ⟨Tb.dep_zip f n l l' ha.1 hb.1, Tb.dep_zip f n h h' ha.2 hb.2⟩

theorem Tb.dep_scale (c : ℕ) : ∀ (n : ℕ) (t : Tb), Tb.dep n t → Tb.dep n (Tb.scale c t)
  | 0, .leaf _, _ => by simp [Tb.scale, Tb.dep]
  | 0, .node _ _, h => by simp [Tb.dep] at h
  | _ + 1, .leaf _, h => by simp [Tb.dep] at h
  | n + 1, .node l r, h => by
    simp only [Tb.dep] at h
    simp only [Tb.scale, Tb.dep]
    exact ⟨Tb.dep_scale c n l h.1, Tb.dep_scale c n r h.2⟩

theorem Tb.dep_mob (M : ℕ) : ∀ (n : ℕ) (t : Tb), Tb.dep n t → Tb.dep n (Tb.mob M t)
  | 0, .leaf _, _ => by simp [Tb.mob, Tb.dep]
  | 0, .node _ _, h => by simp [Tb.dep] at h
  | _ + 1, .leaf _, h => by simp [Tb.dep] at h
  | n + 1, .node l r, h => by
    simp only [Tb.dep] at h
    have hl := Tb.dep_mob M n l h.1
    have hr := Tb.dep_mob M n r h.2
    simp only [Tb.mob, Tb.subN, Tb.dep]
    exact ⟨hl, Tb.dep_zip _ n _ _ hr hl⟩

theorem dep_tabF : ∀ (vs : List ℕ) (B : F2), Tb.dep vs.length (tabF vs B)
  | [], _ => by simp [tabF, Tb.dep]
  | v :: vs, B => by
    have h1 := dep_tabF vs (splitAt v B).2
    have h2 := dep_tabF vs (splitAt v B).1
    simp only [tabF, Tb.xorN, List.length_cons, Tb.dep]
    exact ⟨h1, Tb.dep_zip _ _ _ _ h1 h2⟩

theorem Tb.at_zip (σ : ℕ → ZMod 2) (f : ℕ → ℕ → ℕ) : ∀ (vs : List ℕ) (a b : Tb),
    Tb.dep vs.length a → Tb.dep vs.length b →
      Tb.at σ vs (Tb.zip f a b) = f (Tb.at σ vs a) (Tb.at σ vs b)
  | [], .leaf _, .leaf _, _, _ => by simp [Tb.zip, Tb.at]
  | [], .node _ _, _, ha, _ => by simp [Tb.dep] at ha
  | [], .leaf _, .node _ _, _, hb => by simp [Tb.dep] at hb
  | _ :: _, .leaf _, _, ha, _ => by simp [Tb.dep] at ha
  | _ :: _, .node _ _, .leaf _, _, hb => by simp [Tb.dep] at hb
  | v :: vs, .node l h, .node l' h', ha, hb => by
    simp only [List.length_cons, Tb.dep] at ha hb
    simp only [Tb.zip, Tb.at]
    split_ifs
    · exact Tb.at_zip σ f vs l l' ha.1 hb.1
    · exact Tb.at_zip σ f vs h h' ha.2 hb.2

theorem Tb.at_scale (σ : ℕ → ZMod 2) (c : ℕ) : ∀ (vs : List ℕ) (t : Tb),
    Tb.at σ vs (Tb.scale c t) = c * Tb.at σ vs t
  | [], .leaf _ => by simp [Tb.scale, Tb.at]
  | _ :: _, .leaf _ => by simp [Tb.scale, Tb.at]
  | [], .node l _ => by
    simp only [Tb.scale, Tb.at]
    exact Tb.at_scale σ c [] l
  | v :: vs, .node l h => by
    simp only [Tb.scale, Tb.at]
    split_ifs
    · exact Tb.at_scale σ c vs l
    · exact Tb.at_scale σ c vs h

theorem zn_empty (σ : ℕ → ZMod 2) : ∀ B : F2, (∀ m ∈ B, m = []) →
    (if fz σ B = 0 then 0 else 1) = B.length % 2
  | [], _ => by simp [fz]
  | m :: B, h => by
    have ih := zn_empty σ B (fun m' hm' => h m' (List.mem_cons_of_mem _ hm'))
    have hm : m = [] := h m List.mem_cons_self
    subst hm
    have h10 : (1 : ZMod 2) ≠ 0 := by decide
    rw [fz_cons, show mz σ [] = 1 from rfl, List.length_cons]
    rcases z2_cases (fz σ B) with e | e
    · rw [e] at ih ⊢
      rw [if_pos rfl] at ih
      rw [add_zero, if_neg h10]
      omega
    · rw [e] at ih ⊢
      rw [if_neg h10] at ih
      rw [show (1 : ZMod 2) + 1 = 0 by decide, if_pos rfl]
      omega

theorem at_tabF (σ : ℕ → ZMod 2) : ∀ (vs : List ℕ) (B : F2), (∀ m ∈ B, ∀ x ∈ m, x ∈ vs) →
    Tb.at σ vs (tabF vs B) = if fz σ B = 0 then 0 else 1
  | [], B, hB => by
    simp only [tabF, Tb.at]
    refine (zn_empty σ B fun m hm => ?_).symm
    rcases m with _ | ⟨x, m⟩
    · rfl
    · exact absurd (hB _ hm x List.mem_cons_self) List.not_mem_nil
  | v :: vs, B, hB => by
    have h2 : ∀ m ∈ (splitAt v B).2, ∀ x ∈ m, x ∈ vs := by
      intro m hm x hx
      simp only [splitAt, List.mem_filter, Bool.not_eq_true', decide_eq_false_iff_not] at hm
      rcases List.mem_cons.1 (hB m hm.1 x hx) with e | e
      · exact absurd (e ▸ hx) hm.2
      · exact e
    have h1 : ∀ m ∈ (splitAt v B).1, ∀ x ∈ m, x ∈ vs := by
      intro m hm x hx
      simp only [splitAt, List.mem_map, List.mem_filter, decide_eq_true_eq] at hm
      obtain ⟨m', ⟨hm', _⟩, rfl⟩ := hm
      simp only [List.mem_filter, bne_iff_ne, ne_eq] at hx
      rcases List.mem_cons.1 (hB m' hm' x hx.1) with e | e
      · exact absurd e hx.2
      · exact e
    have i2 := at_tabF σ vs _ h2
    have i1 := at_tabF σ vs _ h1
    simp only [tabF, Tb.xorN, Tb.at]
    rw [fz_splitAt σ v B]
    rcases z2_cases (σ v) with e | e
    · rw [if_pos e, i2, e, zero_mul, zero_add]
    · have hne : σ v ≠ 0 := by rw [e]; decide
      rw [if_neg hne, Tb.at_zip σ _ vs _ _ (dep_tabF vs _) (dep_tabF vs _), i2, i1, e, one_mul]
      rcases z2_cases (fz σ (splitAt v B).1) with a | a <;>
        rcases z2_cases (fz σ (splitAt v B).2) with b | b <;>
        simp [a, b, show (1 : ZMod 2) + 1 = 0 by decide]

theorem pn_mapcons (σ : ℕ → ZMod 2) (v : ℕ) :
    ∀ L : Poly, pn σ (L.map fun t => (v :: t.1, t.2)) = zq (σ v) * pn σ L
  | [] => by simp [pn]
  | t :: L => by
    rw [List.map_cons, pn_cons, pn_cons, pn_mapcons σ v L]
    have hm : mz σ (v :: t.1) = σ v * mz σ t.1 := by simp [mz]
    simp only [hm, zq_mul]
    ring

theorem pn_polyLeaf (σ : ℕ → ZMod 2) (M k : ℕ) (vs : List ℕ) :
    pn σ (Tb.poly M vs (.leaf k)) = ((k % M : ℕ) : ℤ) := by
  have h : pn σ (if k % M = 0 then [] else [([], k % M)]) = ((k % M : ℕ) : ℤ) := by
    split_ifs with h
    · rw [h]; simp [pn]
    · simp [pn, mz, zq]
  cases vs <;> exact h

theorem peq_natmod (M k : ℕ) : peq M ((k % M : ℕ) : ℤ) (k : ℤ) := by
  have := peq_mod M k 1
  simpa using this

theorem pn_poly_sub (σ : ℕ → ZMod 2) (M : ℕ) (hM : 0 < M) : ∀ (vs : List ℕ) (a b : Tb),
    Tb.dep vs.length a → Tb.dep vs.length b →
      peq M (pn σ (Tb.poly M vs (Tb.zip (fun x y => x + (M - y % M)) a b)))
        (pn σ (Tb.poly M vs a) - pn σ (Tb.poly M vs b))
  | [], .leaf x, .leaf y, _, _ => by
    simp only [Tb.zip, pn_polyLeaf]
    obtain ⟨n1, h1⟩ := peq_natmod M (x + (M - y % M))
    obtain ⟨n2, h2⟩ := peq_natmod M x
    have hy : y % M < M := Nat.mod_lt y hM
    refine ⟨n1 - n2 + 1, ?_⟩
    rw [h1, h2]
    push_cast [Nat.cast_sub hy.le]
    ring
  | [], .node _ _, _, ha, _ => by simp [Tb.dep] at ha
  | [], .leaf _, .node _ _, _, hb => by simp [Tb.dep] at hb
  | _ :: _, .leaf _, _, ha, _ => by simp [Tb.dep] at ha
  | _ :: _, .node _ _, .leaf _, _, hb => by simp [Tb.dep] at hb
  | v :: vs, .node l h, .node l' h', ha, hb => by
    simp only [List.length_cons, Tb.dep] at ha hb
    simp only [Tb.zip, Tb.poly, pn_append, pn_mapcons]
    obtain ⟨n1, e1⟩ := pn_poly_sub σ M hM vs l l' ha.1 hb.1
    obtain ⟨n2, e2⟩ := pn_poly_sub σ M hM vs h h' ha.2 hb.2
    exact ⟨n1 + zq (σ v) * n2, by rw [e1, e2]; ring⟩

theorem pn_poly_mob (σ : ℕ → ZMod 2) (M : ℕ) (hM : 0 < M) : ∀ (vs : List ℕ) (t : Tb),
    Tb.dep vs.length t → peq M (pn σ (Tb.poly M vs (Tb.mob M t))) (Tb.at σ vs t)
  | [], .leaf k, _ => by
    simp only [Tb.mob, pn_polyLeaf, Tb.at]
    exact peq_natmod M k
  | [], .node _ _, ht => by simp [Tb.dep] at ht
  | _ :: _, .leaf _, ht => by simp [Tb.dep] at ht
  | v :: vs, .node l h, ht => by
    simp only [List.length_cons, Tb.dep] at ht
    have dl := Tb.dep_mob M _ _ ht.1
    have dh := Tb.dep_mob M _ _ ht.2
    obtain ⟨n1, e1⟩ := pn_poly_mob σ M hM vs l ht.1
    obtain ⟨n2, e2⟩ := pn_poly_mob σ M hM vs h ht.2
    obtain ⟨n3, e3⟩ := pn_poly_sub σ M hM vs _ _ dh dl
    simp only [Tb.mob, Tb.subN, Tb.poly, pn_append, pn_mapcons, Tb.at]
    rw [e3, e1, e2]
    rcases z2_cases (σ v) with e | e
    · rw [if_pos e, e, show zq (0 : ZMod 2) = 0 by decide]
      exact ⟨n1, by ring⟩
    · have hne : σ v ≠ 0 := by rw [e]; decide
      rw [if_neg hne, e, show zq (1 : ZMod 2) = 1 by decide]
      exact ⟨n3 + n2, by ring⟩

theorem pn_liftD (σ : ℕ → ZMod 2) (M c : ℕ) (hM : 0 < M) (vs : List ℕ) (B : F2)
    (hB : ∀ m ∈ B, ∀ x ∈ m, x ∈ vs) : peq M (pn σ (liftD M c vs B)) (c * zq (fz σ B)) := by
  unfold liftD
  refine peq_trans (pn_poly_mob σ M hM vs _ (Tb.dep_scale c _ _ (dep_tabF vs B))) ?_
  rw [Tb.at_scale, at_tabF σ vs B hB]
  refine ⟨0, ?_⟩
  rcases z2_cases (fz σ B) with e | e <;> simp [e, zq]

/-- `c` times `lift M B`: on its table when `B`'s variables have at most `16 |B|` assignments,
else normalised mod M after each monomial. -/
def liftN (M c : ℕ) (B : F2) : Poly :=
  if 2 * c % M = 0 then B.map fun m => (m, c) else
    let vs := vlist B
    if 2 ^ vs.length ≤ 16 * B.length then liftD M c vs B else
    (liftNM M c (B.map (toM vs))).map (conv vs)

theorem pn_linlift (σ : ℕ → ZMod 2) (M c : ℕ) (h : 2 * c % M = 0) :
    ∀ B : F2, peq M (pn σ (B.map fun m => (m, c))) (c * zq (fz σ B))
  | [] => ⟨0, by simp [pn, fz, zq]⟩
  | m :: B => by
    obtain ⟨k, hk⟩ := pn_linlift σ M c h B
    obtain ⟨j, hj⟩ := Nat.dvd_of_mod_eq_zero h
    rw [List.map_cons, pn_cons, fz_cons, zq_add]
    refine ⟨k + j * zq (mz σ m) * zq (fz σ B), ?_⟩
    have hj2 : (2 : ℤ) * c = M * j := by exact_mod_cast hj
    dsimp only
    rw [hk]
    linear_combination zq (mz σ m) * zq (fz σ B) * hj2

theorem pnM_mapor (σ : ℕ → ZMod 2) (vs : List ℕ) (m c : ℕ) : ∀ L : List (ℕ × ℕ),
    pnM σ vs (L.map fun t => (m ||| t.1, c * t.2)) =
      (c : ℤ) * zq (mz σ (ofM vs m)) * pnM σ vs L
  | [] => by simp [pnM, pn]
  | t :: L => by
    rw [List.map_cons, pnM_cons, pnM_cons, pnM_mapor σ vs m c L, mz_ofM_or, zq_mul]
    push_cast
    ring

theorem pn_liftNM (σ : ℕ → ZMod 2) (vs : List ℕ) (M c : ℕ) (hM : 2 ≤ M) :
    ∀ B : List ℕ, peq M (pnM σ vs (liftNM M c B)) (c * zq (fz σ (B.map (ofM vs))))
  | [] => ⟨0, by simp [liftNM, pnM, pn, fz, zq]⟩
  | m :: B => by
    obtain ⟨k, hk⟩ := pn_liftNM σ vs M c hM B
    simp only [liftNM]
    refine peq_trans (pnM_normPM σ vs M _) ?_
    rw [pnM_cons, pnM_append, pnM_mapor, List.map_cons, fz_cons, zq_add]
    have hc : ((M - 2 : ℕ) : ℤ) = (M : ℤ) - 2 := by push_cast [Nat.cast_sub hM]; ring
    rw [hc]
    refine ⟨c * zq (mz σ (ofM vs m)) * zq (fz σ (B.map (ofM vs))) + k +
      ((M : ℤ) - 2) * zq (mz σ (ofM vs m)) * k, ?_⟩
    push_cast
    linear_combination (1 + ((M : ℤ) - 2) * zq (mz σ (ofM vs m))) * hk

theorem pn_liftN (σ : ℕ → ZMod 2) (M c : ℕ) (hM : 2 ≤ M) (B : F2) :
    peq M (pn σ (liftN M c B)) (c * zq (fz σ B)) := by
  unfold liftN
  by_cases hc : 2 * c % M = 0
  · rw [if_pos hc]
    exact pn_linlift σ M c hc B
  rw [if_neg hc]
  show peq M (pn σ (if 2 ^ (vlist B).length ≤ 16 * B.length then liftD M c (vlist B) B else
    (liftNM M c (B.map (toM (vlist B)))).map (conv (vlist B)))) _
  split_ifs
  · exact pn_liftD σ M c (by omega) _ B (fun m hm => mem_vlist hm)
  have h := pn_liftNM σ (vlist B) M c hM (B.map (toM (vlist B)))
  rw [fz_toM σ _ B (fun m hm => mem_vlist hm)] at h
  exact h

/-- `z` replaced by the F_2 polynomial `B` in a phase polynomial. -/
def substP (M z : ℕ) (B : F2) (P : Poly) : Poly :=
  P.flatMap fun t =>
    if z ∈ t.1 then (liftN M t.2 B).map fun u => (t.1.filter (· != z) ++ u.1, u.2) else [t]

/-- `z` replaced by `B` in an F_2 polynomial. -/
def substF (z : ℕ) (B : F2) (Q : F2) : F2 :=
  Q.flatMap fun m => if z ∈ m then B.map fun u => m.filter (· != z) ++ u else [m]

theorem pn_substP (σ : ℕ → ZMod 2) (M z : ℕ) (hM : 2 ≤ M) (B : F2) :
    ∀ P : Poly, peq M (pn σ (substP M z B P)) (pn (Function.update σ z (fz σ B)) P)
  | [] => peq_refl M 0
  | t :: P => by
    have ih := pn_substP σ M z hM B P
    have hsplit : substP M z B (t :: P) = (if z ∈ t.1 then (liftN M t.2 B).map fun u =>
        (t.1.filter (· != z) ++ u.1, 1 * u.2) else [t]) ++ substP M z B P := by
      simp [substP, List.flatMap_cons]
    rw [hsplit, pn_append, pn_cons]
    apply peq_add _ ih
    by_cases hz : z ∈ t.1
    · rw [if_pos hz]
      rw [pn_mapmul, mz_update_in σ z _ t.1 hz, zq_mul]
      have := peq_mul (zq (mz σ (t.1.filter (· != z)))) (pn_liftN σ M t.2 hM B)
      obtain ⟨n, hn⟩ := this
      exact ⟨n, by push_cast; rw [one_mul, hn]; ring⟩
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

/-- `xorF` of the masks of `substF z B Q` over `vs` (`Bm` the masks of `B`), accumulated
without that list. -/
def xorSub (z : ℕ) (vs Bm : List ℕ) (Q : F2) : ℕ :=
  Q.foldr (fun m acc => if z ∈ m then
    Bm.foldr (fun u a => a ^^^ 2 ^ (toM vs (m.filter (· != z)) ||| u)) acc
    else acc ^^^ 2 ^ toM vs m) 0

theorem xorF_append (a b : List ℕ) : xorF (a ++ b) = a.foldr (fun k acc => acc ^^^ 2 ^ k) (xorF b) :=
  List.foldr_append

theorem xorSub_eq (z : ℕ) (vs Bm : List ℕ) : ∀ Q : F2, xorSub z vs Bm Q = xorF (Q.flatMap fun m =>
    if z ∈ m then Bm.map fun u => toM vs (m.filter (· != z)) ||| u else [toM vs m])
  | [] => rfl
  | m :: Q => by
    rw [List.flatMap_cons, xorF_append, ← xorSub_eq z vs Bm Q]
    show (if z ∈ m then _ else _) = _
    split_ifs <;> simp [List.foldr_map] <;> rfl

/-- `normF (substF z B Q)`, computed on masks over `vs`. -/
def substFV (vs : List ℕ) (z : ℕ) (B : F2) (Q : F2) : F2 :=
  let Bm := B.map (toM vs)
  (pickX (fun _ => xorSub z vs Bm Q) (orF (Q.map (toM vs)) ||| orF Bm) (Q.length * (B.length + 1))
    (normFM (Q.flatMap fun m =>
      if z ∈ m then Bm.map fun u => toM vs (m.filter (· != z)) ||| u else [toM vs m]))).map
    (ofM vs)

/-- `normF (substF z B Q)`, computed on masks. -/
def substFN (z : ℕ) (B : F2) (Q : F2) : F2 := substFV (vlist (Q ++ B)) z B Q

theorem fz_substFV (σ : ℕ → ZMod 2) (vs : List ℕ) (z : ℕ) (B : F2) (Q : F2)
    (hQ : ∀ m ∈ Q, ∀ v ∈ m, v ∈ vs) (hB : ∀ m ∈ B, ∀ v ∈ m, v ∈ vs) :
    fz σ (substFV vs z B Q) = fz (Function.update σ z (fz σ B)) Q := by
  unfold substFV
  rw [fz_pickX σ _ (fun _ => xorSub z vs (B.map (toM vs)) Q) _ _ _ _ (xorSub_eq _ _ _ Q)
    (fz_normFM σ _ _)]
  have hBf : fz σ ((B.map (toM vs)).map (ofM vs)) = fz σ B := fz_toM σ _ B hB
  have key : ∀ R : F2, (∀ m ∈ R, ∀ v ∈ m, v ∈ vs) →
      fz σ ((R.flatMap fun m => if z ∈ m then (B.map (toM vs)).map fun u =>
        toM vs (m.filter (· != z)) ||| u else [toM vs m]).map
          (ofM vs)) = fz (Function.update σ z (fz σ B)) R := by
    intro R
    induction R with
    | nil => intro _; rfl
    | cons m R ih =>
      intro hR
      have hm : ∀ v ∈ m, v ∈ vs := hR m List.mem_cons_self
      rw [List.flatMap_cons, List.map_append, fz_append,
        ih (fun r hr => hR r (List.mem_cons_of_mem _ hr)), fz_cons]
      congr 1
      by_cases hz : z ∈ m
      · rw [if_pos hz, mz_update_in σ z _ m hz, ← hBf]
        have hf : ∀ v ∈ m.filter (· != z), v ∈ vs :=
          fun v hv => hm v (List.mem_of_mem_filter hv)
        generalize B.map (toM vs) = L
        induction L with
        | nil => simp [fz]
        | cons u L ihL =>
          rw [List.map_cons, List.map_cons, List.map_cons, fz_cons, fz_cons, ihL, mz_ofM_or,
            mz_ofM_toM σ hf]
          ring
      · rw [if_neg hz, List.map_cons, List.map_nil, fz_cons, mz_ofM_toM σ hm,
          mz_update_not σ z _ m hz]
        simp [fz]
  exact key Q hQ

theorem fz_substFN (σ : ℕ → ZMod 2) (z : ℕ) (B : F2) (Q : F2) :
    fz σ (substFN z B Q) = fz (Function.update σ z (fz σ B)) Q :=
  fz_substFV σ _ z B Q (fun m hm => mem_vlist (List.mem_append_left _ hm))
    (fun m hm => mem_vlist (List.mem_append_right _ hm))

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
    · have hyx : y ≠ x := Ne.symm hxy
      rw [Function.update_comm hyx (0 : ZMod 2) (0 : ZMod 2) σ,
        Function.update_comm hyx (0 : ZMod 2) (1 : ZMod 2) σ,
        Function.update_comm hyx (1 : ZMod 2) (0 : ZMod 2) σ,
        Function.update_comm hyx (1 : ZMod 2) (1 : ZMod 2) σ]
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

/-- `x` moved to the front of the path. -/
def frontS (x : ℕ) (s : Data) : Data := { s with path := x :: s.path.erase x }

theorem front_sound (M : ℕ) (c : ℂ) (x : ℕ) (s : Data) (σ : ℕ → ZMod 2)
    (h : decide (x ∈ s.path) = true) : val M c s σ = val M c (frontS x s) σ := by
  unfold val frontS
  exact bsum_perm _ (List.perm_cons_erase (of_decide_eq_true h)) σ

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
  | x :: L => by rw [List.map_cons, List.sum_cons, Int.cast_add, zq_cast, zsum_cast L, List.sum_cons]

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
      have ht : (t.2 : ℝ) ≠ 0 := by
        intro e
        apply hM'
        rw [hMt, e, mul_zero]
      simp only [Int.cast_list_sum, List.map_map]
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
      simp only [Int.cast_list_sum, List.map_map]
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
  | y :: ys =>
    { path := ys
      phase := rest y s.phase
      cons := quot y s.phase :: s.cons
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
    by_cases hC : ∀ q ∈ s.cons ++ s.rows, fz τ q = 0
    · simp only [if_pos hC]
      rw [pair_half, hQ]
      by_cases hz : fz τ (quot y s.phase) = 0
      · rw [if_pos hz, if_pos (And.intro hz hC)]
        ring
      · rw [if_neg hz, if_neg (fun h => hz h.1)]
        ring
    · simp only [if_neg hC]
      rw [if_neg (fun h => hC h.2)]
      ring

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
      intro hm
      rcases List.mem_cons.1 hm with e | e
      · exact hyz e
      · exact hy e
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
      by_cases hC : ∀ q ∈ s.cons ++ s.rows, fz (Function.update τ z w) q = 0
      · simp only [if_pos hC]
        rw [pair_half, hQcast]
      · simp only [if_neg hC]
        simp
    rw [hstep 0, hstep 1, pick]
    show 2 * G (fz τ B) = summand M (2 * c) _ τ
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
    simp only [← hQdef, ← hBdef]
    simp only [hcons]
    split_ifs
    · rw [← mul_assoc]
      congr 1
      symm
      apply ex_peq hM0
      exact peq_trans (pn_normP τ M _) (pn_substP τ M z hM B _)
    · ring

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
    simp [mz, B]
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
        · rcases List.mem_cons.1 (hperm.mem_iff.1 hr) with e | hr'
          · rw [e]; exact h0
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
      symm
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

/-- The product of two mask F_2 polynomials. -/
def f2mulM (a b : List ℕ) : List ℕ := a.flatMap fun m => b.map fun u => m ||| u

theorem fz_f2mulM (τ : ℕ → ZMod 2) (vs : List ℕ) (b : List ℕ) : ∀ a : List ℕ,
    fz τ ((f2mulM a b).map (ofM vs)) = fz τ (a.map (ofM vs)) * fz τ (b.map (ofM vs))
  | [] => by simp [f2mulM, fz]
  | m :: a => by
    have ih := fz_f2mulM τ vs b a
    have hs : f2mulM (m :: a) b = (b.map fun u => m ||| u) ++ f2mulM a b := by
      simp [f2mulM, List.flatMap_cons]
    have hm : ∀ L : List ℕ, fz τ ((L.map fun u => m ||| u).map (ofM vs)) =
        mz τ (ofM vs m) * fz τ (L.map (ofM vs)) := by
      intro L
      induction L with
      | nil => simp [fz]
      | cons u L ihL =>
        rw [List.map_cons, List.map_cons, List.map_cons, fz_cons, fz_cons, ihL, mz_ofM_or]
        ring
    rw [hs, List.map_append, fz_append, hm, ih, List.map_cons, fz_cons]
    ring

/-- `xorF (f2mulM a b)`, accumulated without the product list. -/
def xorMul (a b : List ℕ) : ℕ :=
  a.foldr (fun m acc => b.foldr (fun u acc' => acc' ^^^ 2 ^ (m ||| u)) acc) 0

theorem xorMul_eq (b : List ℕ) : ∀ a : List ℕ, xorMul a b = xorF (f2mulM a b)
  | [] => rfl
  | m :: a => by
    rw [show f2mulM (m :: a) b = (b.map fun u => m ||| u) ++ f2mulM a b by
      simp [f2mulM, List.flatMap_cons], xorF_append, ← xorMul_eq b a]
    simp [xorMul, List.foldr_map]

/-- `normFM (f2mulM a b)`: through `pickX` on `xorMul a b`. -/
def mulM (a b : List ℕ) : List ℕ :=
  pickX (fun _ => xorMul a b) (orF a ||| orF b) (a.length * b.length) (normFM (f2mulM a b))

theorem fz_mulM (τ : ℕ → ZMod 2) (vs a b : List ℕ) :
    fz τ ((mulM a b).map (ofM vs)) = fz τ (a.map (ofM vs)) * fz τ (b.map (ofM vs)) := by
  unfold mulM
  rw [fz_pickX τ vs (fun _ => xorMul a b) _ _ _ _ (xorMul_eq b a) (fz_normFM τ vs _), fz_f2mulM]

/-- `normF (f2mul a b)`, computed on masks. -/
def f2mulN (a b : F2) : F2 :=
  let vs := vlist (a ++ b)
  (mulM (a.map (toM vs)) (b.map (toM vs))).map (ofM vs)

theorem fz_f2mulN (τ : ℕ → ZMod 2) (a b : F2) : fz τ (f2mulN a b) = fz τ a * fz τ b := by
  unfold f2mulN
  rw [fz_mulM,
    fz_toM τ _ a (fun m hm => mem_vlist (List.mem_append_left _ hm)),
    fz_toM τ _ b (fun m hm => mem_vlist (List.mem_append_right _ hm))]

/-- The mask support indicator `Prod (1 + c)` over `vs` of a list of constraints, normalised
after each factor, empty constraints skipped. -/
def gatherM (vs : List ℕ) (C : List F2) : List ℕ :=
  C.foldr (fun c acc => if c = [] then acc else mulM (0 :: c.map (toM vs)) acc) [0]

/-- The support indicator `Prod (1 + c)` of a list of constraints, normalised after each
factor. -/
def gatherPoly (C : List F2) : F2 :=
  let vs := vlist (C.flatMap id)
  (gatherM vs C).map (ofM vs)

theorem fz_gatherM (τ : ℕ → ZMod 2) (vs : List ℕ) :
    ∀ C : List F2, (∀ q ∈ C, ∀ m ∈ q, ∀ v ∈ m, v ∈ vs) →
      (fz τ ((gatherM vs C).map (ofM vs)) = 1 ↔ ∀ q ∈ C, fz τ q = 0)
  | [], _ => by simp [gatherM, fz, ofM_zero, mz]
  | c :: C, hC => by
    have ih := fz_gatherM τ vs C (fun q hq => hC q (List.mem_cons_of_mem _ hq))
    by_cases hc : c = []
    · subst hc
      have hs : gatherM vs ([] :: C) = gatherM vs C := by simp [gatherM]
      rw [hs, List.forall_mem_cons, ih]
      simp [fz]
    have hs : gatherM vs (c :: C) = mulM (0 :: c.map (toM vs)) (gatherM vs C) := by
      simp [gatherM, hc]
    rw [hs, fz_mulM, List.map_cons, fz_cons,
      fz_toM τ vs c (hC c List.mem_cons_self), ofM_zero,
      show mz τ [] = 1 from rfl, List.forall_mem_cons, ← ih]
    rcases z2_cases (fz τ c) with h | h <;>
      rcases z2_cases (fz τ ((gatherM vs C).map (ofM vs))) with g | g <;> simp [h, g]

theorem fz_gather (τ : ℕ → ZMod 2) (C : List F2) :
    fz τ (gatherPoly C) = 1 ↔ ∀ q ∈ C, fz τ q = 0 :=
  fz_gatherM τ _ C fun q hq m hm =>
    mem_vlist (List.mem_flatMap.2 ⟨q, hq, by simpa using hm⟩)

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
    !decide (v ∈ s.path) && decide (f2mulN (splitAt v g).2 (detValue v q) = [])

/-- The first constraint's determined variable `v` made a row `v + A + B`, substituted out of
the phase, the constraints and the rows. -/
def bcanonS (M v : ℕ) (s : Data) : Data :=
  match s.cons with
  | [] => s
  | q :: rest =>
    let val := detValue v q
    { path := s.path, phase := normP M (substP M v val s.phase),
      cons := (q :: rest).map fun r => substFN v val r,
      rows := (s.rows.map fun r => substFN v val r) ++ [normF ([v] :: val)] }

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
      rw [fz_f2mulN] at this
      simpa [fz] using this
    unfold summand
    simp only [hq]
    have hM0 : 0 < M := by omega
    by_cases hrow : τ v = fz τ val
    · have hτ : Function.update τ v (fz τ val) = τ := by rw [← hrow, Function.update_eq_self]
      have hsub : ∀ r, fz τ (substFN v val r) = fz τ r := by
        intro r; rw [fz_substFN, hτ]
      have hrowz : fz τ (normF ([v] :: val)) = 0 := by
        have e : mz τ [v] = τ v := by simp [mz]
        rw [fz_normF, fz_cons, e, hrow]
        exact z2_add_self _
      have hcond : (∀ r ∈ (q :: rest).map (fun r => substFN v val r) ++
          ((s.rows.map fun r => substFN v val r) ++ [normF ([v] :: val)]), fz τ r = 0) ↔
          (∀ r ∈ (q :: rest) ++ s.rows, fz τ r = 0) := by
        simp only [List.mem_append, List.mem_map, List.mem_singleton]
        constructor
        · rintro H r (hr | hr)
          · have := H _ (Or.inl ⟨r, hr, rfl⟩); rwa [hsub] at this
          · have := H _ (Or.inr (Or.inl ⟨r, hr, rfl⟩)); rwa [hsub] at this
        · rintro H r (⟨r', hr', rfl⟩ | ⟨r', hr', rfl⟩ | hr)
          · rw [hsub]; exact H r' (Or.inl hr')
          · rw [hsub]; exact H r' (Or.inr hr')
          · rw [hr]; exact hrowz
      simp only [hcond]
      split_ifs
      · refine congrArg (c * ·) ?_
        symm
        apply ex_peq hM0
        have := peq_trans (pn_normP τ M _) (pn_substP τ M v hM val s.phase)
        rwa [hτ] at this
      · rfl
    · have hr : ¬ (∀ r ∈ (q :: rest).map (fun r => substFN v val r) ++
          ((s.rows.map fun r => substFN v val r) ++ [normF ([v] :: val)]),
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

end PSem
end ArchytasZX
