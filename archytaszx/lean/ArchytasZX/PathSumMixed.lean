/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumQudit
import ArchytasZX.PathSumAmp

/-!
# Mixed path sums on dimensions 2, 3 and 6

A label of dimension 6 is the digit pair `(s, f)` of its value `2 s + f`, `s : ZMod 3`,
`f : ZMod 2` (`sum_six`). `ind k s` is the indicator `[s = k]` as an F_2 value; `ind_zero`
writes `I_0 = 1 - I_1 - I_2` over the integers, `ind_one_two` gives `I_1 I_2 = 0` and
`ind_affine` reads an indicator through `s = a z + b`. `elim3` is [Elim3], the sum over `s` of
`zeta_3^(s L)` times a function free of `s` being `3 [L = 0]` times it; `count3` is [Count3],
the sum over `s` of a function of `(I_1 s, I_2 s)` being the sum over two F_2 values with the
zero atom `0^(u v)`. `lift` sets the indicator variables of an F_2 assignment from a Z_3
assignment; `lift_count` and `lift_elim` are [Count3] and [Elim3] on the sums of `qsum`.
-/

namespace ArchytasZX
namespace MSem

open Complex Finset

/-- `[s = k]` in F_2. -/
def ind (k : ℕ) (s : ZMod 3) : ZMod 2 := if s = (k : ZMod 3) then 1 else 0

theorem ind_one_two (s : ZMod 3) : ind 1 s * ind 2 s = 0 := by
  fin_cases s <;> decide

theorem ind_zero (s : ZMod 3) :
    PSem.zq (ind 0 s) = 1 - PSem.zq (ind 1 s) - PSem.zq (ind 2 s) := by
  fin_cases s <;> decide

theorem ind_affine (k : ℕ) (a b z : ZMod 3) (ha : a ≠ 0) :
    ind k (a * z + b) = ind ((((k : ZMod 3) - b) * a⁻¹).val) z := by
  unfold ind
  have e : a * z + b = (k : ZMod 3) ↔ z = ((k : ZMod 3) - b) * a⁻¹ := by
    constructor
    · intro h
      rw [← h]
      field_simp
      ring
    · intro h
      rw [h]
      field_simp
      ring
  simp only [ZMod.natCast_val, ZMod.cast_id', id, e]

theorem ind_const (k : ℕ) (b : ZMod 3) : ind k b = if b = (k : ZMod 3) then 1 else 0 := rfl

/-- A sum over `range 6` as a sum over the digit pairs `(s, f)` of `2 s + f`. -/
theorem sum_six (G : ℕ → ℂ) :
    ∑ j ∈ range 6, G j = ∑ s : ZMod 3, ∑ f : ZMod 2, G (2 * s.val + f.val) := by
  show _ = ∑ s : Fin 3, ∑ f : Fin 2, G (2 * (s : ℕ) + (f : ℕ))
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Fin.sum_univ_three,
    Fin.sum_univ_two, zero_add]
  norm_num
  ring

/-- [Elim3]: `Sum_s zeta_3^(s L) H = 3 [L = 0] H`. -/
theorem elim3 (L : ZMod 3) (H : ℂ) :
    ∑ s : ZMod 3, QSem.ψ 3 (s * L) * H = if L = 0 then 3 * H else 0 := by
  have : Fact (Nat.Prime 3) := ⟨Nat.prime_three⟩
  rw [← Finset.sum_mul, QSem.char_sum]
  split_ifs <;> simp

/-- [Count3]: the sum over `s` of a function of `(I_1 s, I_2 s)`. -/
theorem count3 (F : ZMod 2 → ZMod 2 → ℂ) :
    ∑ s : ZMod 3, F (ind 1 s) (ind 2 s) =
      ∑ u : ZMod 2, ∑ v : ZMod 2, if u * v = 0 then F u v else 0 := by
  have u3 : (Finset.univ : Finset (ZMod 3)) = {0, 1, 2} := by decide
  have u2 : (Finset.univ : Finset (ZMod 2)) = {0, 1} := by decide
  have h0 : ind 1 (0 : ZMod 3) = 0 ∧ ind 2 (0 : ZMod 3) = 0 := by decide
  have h1 : ind 1 (1 : ZMod 3) = 1 ∧ ind 2 (1 : ZMod 3) = 0 := by decide
  have h2 : ind 1 (2 : ZMod 3) = 0 ∧ ind 2 (2 : ZMod 3) = 1 := by decide
  have e11 : ¬ (1 : ZMod 2) * 1 = 0 := by decide
  rw [u3, u2]
  rw [Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton,
    Finset.sum_insert (by decide), Finset.sum_singleton, Finset.sum_insert (by decide),
    Finset.sum_singleton, Finset.sum_insert (by decide), Finset.sum_singleton]
  rw [h0.1, h0.2, h1.1, h1.2, h2.1, h2.2]
  simp only [mul_zero, zero_mul, if_true, if_neg e11]
  ring

/-! ## Indicator variables of an F_2 assignment -/

/-- `τ` with the variable `u` of each entry `(u, s, k)` set to `ind k (ρ s)`. -/
def lift : List (ℕ × ℕ × ℕ) → (ℕ → ZMod 3) → (ℕ → ZMod 2) → ℕ → ZMod 2
  | [], _, τ => τ
  | (u, s, k) :: es, ρ, τ => Function.update (lift es ρ τ) u (ind k (ρ s))

theorem lift_out (ρ : ℕ → ZMod 3) (τ : ℕ → ZMod 2) (x : ℕ) :
    ∀ es : List (ℕ × ℕ × ℕ), x ∉ es.map (·.1) → lift es ρ τ x = τ x
  | [], _ => rfl
  | (u, s, k) :: es, h => by
    simp only [List.map_cons, List.mem_cons, not_or] at h
    rw [lift, Function.update_of_ne h.1, lift_out ρ τ x es h.2]

theorem lift_rho (ρ ρ' : ℕ → ZMod 3) (τ : ℕ → ZMod 2) :
    ∀ es : List (ℕ × ℕ × ℕ), (∀ e ∈ es, ρ e.2.1 = ρ' e.2.1) → lift es ρ τ = lift es ρ' τ
  | [], _ => rfl
  | (u, s, k) :: es, h => by
    rw [lift, lift, lift_rho ρ ρ' τ es (fun e he => h e (List.mem_cons_of_mem _ he)),
      h (u, s, k) List.mem_cons_self]

/-- [Count3] on one Z_3 variable `s` read only through its indicators `u₁ = I_1 s` and
`u₂ = I_2 s` (the first two entries): the sum over `s` is the sum over `u₁, u₂` with the zero
atom `0^(u₁ u₂)`. -/
theorem lift_count (G : (ℕ → ZMod 2) → ℂ) (ρ : ℕ → ZMod 3) (τ : ℕ → ZMod 2)
    (u₁ u₂ s : ℕ) (es : List (ℕ × ℕ × ℕ)) (_hne : u₁ ≠ u₂)
    (hs : ∀ e ∈ es, e.2.1 ≠ s) (_hu : u₁ ∉ es.map (·.1) ∧ u₂ ∉ es.map (·.1)) :
    ∑ b : ZMod 3, G (lift ((u₁, s, 1) :: (u₂, s, 2) :: es) (Function.update ρ s b) τ) =
      ∑ x : ZMod 2, ∑ y : ZMod 2,
        if x * y = 0 then
          G (Function.update (Function.update (lift es ρ τ) u₂ y) u₁ x) else 0 := by
  have hr : ∀ b, lift es (Function.update ρ s b) τ = lift es ρ τ := fun b =>
    lift_rho _ _ τ es fun e he => Function.update_of_ne (hs e he) _ _
  have hl : ∀ b, lift ((u₁, s, 1) :: (u₂, s, 2) :: es) (Function.update ρ s b) τ =
      Function.update (Function.update (lift es ρ τ) u₂ (ind 2 b)) u₁ (ind 1 b) := by
    intro b
    simp only [lift, Function.update_self, hr]
  simp only [hl]
  exact count3 (fun x y => G (Function.update (Function.update (lift es ρ τ) u₂ y) u₁ x))

/-- [Elim3] on one Z_3 variable `s` outside the indicators: the sum over `s` of
`zeta_3^(s L)` times a function of the other variables. -/
theorem lift_elim (H : (ℕ → ZMod 3) → ℂ) (L : (ℕ → ZMod 3) → ZMod 3) (ρ : ℕ → ZMod 3) (s : ℕ)
    (hH : ∀ b, H (Function.update ρ s b) = H ρ) (hL : ∀ b, L (Function.update ρ s b) = L ρ) :
    ∑ b : ZMod 3, QSem.ψ 3 (b * L (Function.update ρ s b)) * H (Function.update ρ s b) =
      if L ρ = 0 then 3 * H ρ else 0 := by
  simp only [hH, hL]
  exact elim3 (L ρ) (H ρ)

/-! ## Indicators read through affine maps -/

/-- An indicator entry `(u, z, a, b, k)`: the F_2 variable `u` is `ind k (a z + b)`. -/
abbrev Ent := ℕ × ℕ × ℕ × ℕ × ℕ

/-- An entry's value under `ρ`. -/
def entVal (ρ : ℕ → ZMod 3) (e : Ent) : ZMod 2 :=
  ind e.2.2.2.2 ((e.2.2.1 : ZMod 3) * ρ e.2.1 + (e.2.2.2.1 : ZMod 3))

/-- `τ` with each entry's variable set to its value under `ρ`. -/
def liftA : List Ent → (ℕ → ZMod 3) → (ℕ → ZMod 2) → ℕ → ZMod 2
  | [], _, τ => τ
  | e :: es, ρ, τ => Function.update (liftA es ρ τ) e.1 (entVal ρ e)

/-- An entry on `x` read through `x = a' z' + b'`. -/
def pinE (x z' a' b' : ℕ) (e : Ent) : Ent :=
  if e.2.1 = x then (e.1, z', e.2.2.1 * a' % 3, (e.2.2.1 * b' + e.2.2.2.1) % 3, e.2.2.2.2)
  else e

theorem entVal_pinE (ρ : ℕ → ZMod 3) (x z' a' b' : ℕ)
    (h : ρ x = (a' : ZMod 3) * ρ z' + (b' : ZMod 3)) (e : Ent) :
    entVal ρ (pinE x z' a' b' e) = entVal ρ e := by
  unfold pinE entVal
  split_ifs with hx
  · simp only
    rw [hx, h]
    congr 1
    simp only [ZMod.natCast_mod, Nat.cast_mul, Nat.cast_add]
    ring
  · rfl

theorem liftA_pinE (ρ : ℕ → ZMod 3) (τ : ℕ → ZMod 2) (x z' a' b' : ℕ)
    (h : ρ x = (a' : ZMod 3) * ρ z' + (b' : ZMod 3)) :
    ∀ es : List Ent, liftA (es.map (pinE x z' a' b')) ρ τ = liftA es ρ τ
  | [] => rfl
  | e :: es => by
    have h1 : (pinE x z' a' b' e).1 = e.1 := by unfold pinE; split_ifs <;> rfl
    rw [List.map_cons, liftA, liftA, liftA_pinE ρ τ x z' a' b' h es, h1,
      entVal_pinE ρ x z' a' b' h e]

theorem liftA_rho (ρ ρ' : ℕ → ZMod 3) (τ : ℕ → ZMod 2) :
    ∀ es : List Ent, (∀ e ∈ es, ρ e.2.1 = ρ' e.2.1) → liftA es ρ τ = liftA es ρ' τ
  | [], _ => rfl
  | e :: es, h => by
    rw [liftA, liftA, liftA_rho ρ ρ' τ es (fun e' he => h e' (List.mem_cons_of_mem _ he))]
    unfold entVal
    rw [h e List.mem_cons_self]

/-- The normal form `ind j z` of an entry with a unit coefficient. -/
theorem entVal_norm (ρ : ℕ → ZMod 3) (e : Ent) (ha : (e.2.2.1 : ZMod 3) ≠ 0) :
    entVal ρ e = ind ((((e.2.2.2.2 : ZMod 3) - e.2.2.2.1) * (e.2.2.1 : ZMod 3)⁻¹).val)
      (ρ e.2.1) := by
  unfold entVal
  exact ind_affine _ _ _ _ ha

/-! ## Mixed path sums -/

open PSem in
theorem bsum_zero : ∀ (ys : List ℕ) (σ : ℕ → ZMod 2), bsum ys (fun _ => (0 : ℂ)) σ = 0
  | [], _ => rfl
  | y :: ys, σ => by simp only [bsum, bsum_zero ys, add_zero]

open PSem in
/-- Two held variables `u₁ ≠ u₂` with the zero atom `0^(u₁ u₂)` (`A j = 0`). -/
theorem valA_zero_atom (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (q : AData) (u₁ u₂ j : ℕ) (hA : A j = 0)
    (hne : u₁ ≠ u₂) (h₁ : u₁ ∉ q.held) (h₂ : u₂ ∉ q.held) (τ : ℕ → ZMod 2) :
    valA M c A ⟨q.s, u₁ :: u₂ :: q.held, (j, [[u₁, u₂]]) :: q.atoms⟩ τ =
      ∑ x : ZMod 2, ∑ y : ZMod 2,
        if x * y = 0 then valA M c A q (Function.update (Function.update τ u₂ y) u₁ x)
        else 0 := by
  have key : ∀ x y : ZMod 2,
      bsum q.held (fun τ' => aval A ((j, [[u₁, u₂]]) :: q.atoms) τ' * PSem.val M c q.s τ')
        (Function.update (Function.update τ u₁ x) u₂ y) =
      if x * y = 0 then valA M c A q (Function.update (Function.update τ u₂ y) u₁ x)
      else 0 := by
    intro x y
    rw [Function.update_comm hne, bsum_out _ u₁ x _ _ h₁, bsum_out _ u₂ y _ _ h₂]
    have hf : ∀ τ' : ℕ → ZMod 2,
        fz (Function.update (Function.update τ' u₂ y) u₁ x) [[u₁, u₂]] = x * y := by
      intro τ'
      simp [fz, mz, Function.update_of_ne hne, Function.update_of_ne (Ne.symm hne),
        Function.update_self]
    split_ifs with hxy
    · unfold valA
      rw [bsum_out _ u₁ x _ _ h₁, bsum_out _ u₂ y _ _ h₂]
      · apply bsum_congr
        intro τ'
        simp only [aval_cons, hf, if_pos hxy, one_mul]
    · rw [← bsum_zero q.held τ]
      apply bsum_congr
      intro τ'
      simp only [aval_cons, hf, if_neg hxy, hA, zero_mul]
  show bsum (u₁ :: u₂ :: q.held) _ τ = _
  simp only [bsum]
  rw [key 0 0, key 0 1, key 1 0, key 1 1]
  have u2 : (Finset.univ : Finset (ZMod 2)) = {0, 1} := by decide
  rw [u2, Finset.sum_pair (by decide), Finset.sum_pair (by decide), Finset.sum_pair (by decide)]

/-- The value of a mixed path sum: the qubit part `q` read at the indicator entries `es` of
the Z_3 assignment, summed with the Z_3 part `t` over its path variables. -/
noncomputable def mval (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (q : PSem.AData) (t : QSem.Data)
    (es : List Ent) (τ : ℕ → ZMod 2) (ρ : ℕ → ZMod 3) : ℂ :=
  QSem.qsum t.path (fun ρ' => (if ∀ r ∈ t.cons ++ t.rows, QSem.qv ρ' r = 0 then
    QSem.ψ 3 (QSem.qv ρ' t.phase) else 0) * PSem.valA M c A q (liftA es ρ' τ)) ρ

/-- The Z_3 weight of a mixed summand. -/
noncomputable def wt3 (t : QSem.Data) (ρ : ℕ → ZMod 3) : ℂ :=
  if ∀ r ∈ t.cons ++ t.rows, QSem.qv ρ r = 0 then QSem.ψ 3 (QSem.qv ρ t.phase) else 0

theorem wt3_update (t : QSem.Data) (ρ : ℕ → ZMod 3) (s : ℕ) (b : ZMod 3)
    (hph : s ∉ QSem.pvars t.phase) (hc : s ∉ QSem.cvars (t.cons ++ t.rows)) :
    wt3 t (Function.update ρ s b) = wt3 t ρ := by
  unfold wt3
  rw [QSem.qv_update_not ρ s b _ hph]
  simp only [QSem.holds_update_not ρ s b _ hc]

/-- [Count3] on a mixed path sum: the Z_3 path variable `s`, outside the Z_3 phase and
constraints, read by the entries `(u₁, s, 1, 0, 1)` and `(u₂, s, 1, 0, 2)` only, becomes the
held F_2 variables `u₁`, `u₂` with the zero atom `0^(u₁ u₂)`. -/
theorem mval_count3 (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (q : PSem.AData) (t : QSem.Data)
    (es : List Ent) (u₁ u₂ s j : ℕ) (rest : List ℕ) (hpath : t.path = s :: rest)
    (hs : s ∉ rest) (hph : s ∉ QSem.pvars t.phase) (hc : s ∉ QSem.cvars (t.cons ++ t.rows))
    (hes : ∀ e ∈ es, e.2.1 ≠ s) (hA : A j = 0) (hne : u₁ ≠ u₂) (h₁ : u₁ ∉ q.held)
    (h₂ : u₂ ∉ q.held) (τ : ℕ → ZMod 2) (ρ : ℕ → ZMod 3) :
    mval M c A q t ((u₁, s, 1, 0, 1) :: (u₂, s, 1, 0, 2) :: es) τ ρ =
      mval M c A ⟨q.s, u₁ :: u₂ :: q.held, (j, [[u₁, u₂]]) :: q.atoms⟩
        ⟨rest, t.phase, t.cons, t.rows⟩ es τ ρ := by
  unfold mval
  rw [hpath]
  show ∑ b : ZMod 3, QSem.qsum rest _ (Function.update ρ s b) = _
  simp only [QSem.qsum_out _ s _ rest _ hs]
  rw [← QSem.qsum_fsum]
  apply QSem.qsum_congr
  intro ρ'
  change ∑ b : ZMod 3, wt3 t (Function.update ρ' s b) * PSem.valA M c A q
      (liftA ((u₁, s, 1, 0, 1) :: (u₂, s, 1, 0, 2) :: es) (Function.update ρ' s b) τ) =
    wt3 ⟨rest, t.phase, t.cons, t.rows⟩ ρ' * _
  have hw : wt3 ⟨rest, t.phase, t.cons, t.rows⟩ ρ' = wt3 t ρ' := rfl
  simp only [wt3_update t ρ' s _ hph hc, hw]
  rw [← Finset.mul_sum, valA_zero_atom M c A q u₁ u₂ j hA hne h₁ h₂]
  congr 1
  have hr : ∀ b, liftA es (Function.update ρ' s b) τ = liftA es ρ' τ := fun b =>
    liftA_rho _ _ τ es fun e he => Function.update_of_ne (hes e he) _ _
  have hl : ∀ b : ZMod 3, liftA ((u₁, s, 1, 0, 1) :: (u₂, s, 1, 0, 2) :: es)
      (Function.update ρ' s b) τ =
      Function.update (Function.update (liftA es ρ' τ) u₂ (ind 2 b)) u₁ (ind 1 b) := by
    intro b
    simp only [liftA, entVal, hr, Function.update_self, Nat.cast_one, Nat.cast_zero, one_mul,
      add_zero]
  simp only [hl]
  exact count3 (fun x y => PSem.valA M c A q
    (Function.update (Function.update (liftA es ρ' τ) u₂ y) u₁ x))

instance fact3 : Fact (Nat.Prime 3) := ⟨Nat.prime_three⟩

theorem qsum_const_mul (k : ℂ) (f : (ℕ → ZMod 3) → ℂ) :
    ∀ (ys : List ℕ) (σ : ℕ → ZMod 3), QSem.qsum ys (fun τ => k * f τ) σ = k * QSem.qsum ys f σ
  | [], _ => rfl
  | y :: ys, σ => by
    simp only [QSem.qsum, qsum_const_mul k f ys, Finset.mul_sum]

open QSem in
/-- [HH] on a path pivot (`hhPivotS`) on a mixed path sum: the head Z_3 variable `y` outside
the entries, the next variable `z` set to the pivot value, the entries read through it
(`hL`). -/
theorem mval_hhPivot (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (q : PSem.AData) (t : QSem.Data)
    (es es' : List Ent) (y z : ℕ) (ys : List ℕ) (hp : t.path = y :: z :: ys)
    (h : hhPivotOK 3 t = true) (hy : ∀ e ∈ es, e.2.1 ≠ y) (τ : ℕ → ZMod 2)
    (hL : ∀ ρ : ℕ → ZMod 3, liftA es (Function.update ρ z
      (qv ρ (pivVal 3 z (yk y 1 t.phase)))) τ = liftA es' ρ τ) (ρ : ℕ → ZMod 3) :
    mval M c A q t es τ ρ = 3 * mval M c A q (hhPivotS 3 t) es' τ ρ := by
  have h' := h
  simp only [hhPivotOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    List.all_eq_true, decide_eq_true_eq] at h'
  obtain ⟨⟨⟨⟨⟨hyz, hy'⟩, hz⟩, hlin⟩, hc⟩, hpiv⟩ := h'
  unfold mval
  simp only [hhPivotS, hp]
  have hyzs : y ∉ z :: ys := by
    intro hm
    rcases List.mem_cons.1 hm with e | e
    · exact hyz e
    · exact hy' e
  rw [qsum_head _ y (z :: ys) ρ hyzs, qsum_head _ z ys ρ hz, ← qsum_const_mul]
  apply qsum_congr
  intro τ3
  set Q := yk y 1 t.phase
  set R := yk y 0 t.phase
  set B := pivVal 3 z Q
  have hg : ∀ w b, liftA es (Function.update (Function.update τ3 z w) y b) τ =
      liftA es (Function.update τ3 z w) τ := fun w b =>
    liftA_rho _ _ τ es fun e he => Function.update_of_ne (hy e he) _ _
  let G : ZMod 3 → ℂ := fun w =>
    (if ∀ r ∈ t.cons ++ t.rows, qv (Function.update τ3 z w) r = 0 then
      ψ 3 (qv (Function.update τ3 z w) R) else 0) *
      PSem.valA M c A q (liftA es (Function.update τ3 z w) τ)
  have hstep : ∀ w : ZMod 3,
      ∑ b : ZMod 3, (if ∀ r ∈ t.cons ++ t.rows,
          qv (Function.update (Function.update τ3 z w) y b) r = 0 then
        ψ 3 (qv (Function.update (Function.update τ3 z w) y b) t.phase) else 0) *
        PSem.valA M c A q (liftA es (Function.update (Function.update τ3 z w) y b) τ) =
        (if qv (Function.update τ3 z w) Q = 0 then ((3 : ℕ) : ℂ) else 0) * G w := by
    intro w
    simp only [hg, holds_update_not (Function.update τ3 z w) y _ _ hc,
      qv_split1 (Function.update τ3 z w) y _ t.phase hlin]
    by_cases hC : ∀ r ∈ t.cons ++ t.rows, qv (Function.update τ3 z w) r = 0
    · simp only [G, if_pos hC, ψ_add]
      rw [← Finset.sum_mul, ← Finset.mul_sum, QSem.char_sum]
      split_ifs <;> ring
    · simp only [G, if_neg hC]
      simp
  simp only [hstep]
  rw [sum_pin G (qv τ3 B) _ (piv_iff z Q hpiv τ3)]
  simp only [G]
  have hcons : (∀ r ∈ (t.cons.map fun r => normQ 3 (substQ z B r)) ++
      (t.rows.map fun r => normQ 3 (substQ z B r)), qv τ3 r = 0) ↔
      (∀ r ∈ t.cons ++ t.rows, qv (Function.update τ3 z (qv τ3 B)) r = 0) := by
    rw [← List.map_append, holds_map (fun r => normQ 3 (substQ z B r))
      (fun r => by rw [qv_normQ, qv_substQ])]
  simp only [hcons, qv_normQ, qv_substQ, hL]
  push_cast
  ring

open QSem in
/-- [HH] on a constraint (`hhConsS`) on a mixed path sum, the head variable outside the
entries. -/
theorem mval_hhCons (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (q : PSem.AData) (t : QSem.Data)
    (es : List Ent) (y : ℕ) (ys : List ℕ) (hp : t.path = y :: ys) (h : hhConsOK t = true)
    (hy : ∀ e ∈ es, e.2.1 ≠ y) (τ : ℕ → ZMod 2) (ρ : ℕ → ZMod 3) :
    mval M c A q t es τ ρ = 3 * mval M c A q (hhConsS t) es τ ρ := by
  have h' := h
  simp only [hhConsOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    List.all_eq_true, decide_eq_true_eq] at h'
  obtain ⟨⟨h1, hlin⟩, h3⟩ := h'
  unfold mval
  simp only [hhConsS, hp]
  rw [qsum_head _ y ys ρ h1, ← qsum_const_mul]
  apply qsum_congr
  intro τ3
  have hg : ∀ b, liftA es (Function.update τ3 y b) τ = liftA es τ3 τ := fun b =>
    liftA_rho _ _ τ es fun e he => Function.update_of_ne (hy e he) _ _
  simp only [hg, holds_update_not τ3 y _ _ h3, qv_split1 τ3 y _ t.phase hlin]
  simp only [List.cons_append, List.forall_mem_cons]
  by_cases hC : ∀ r ∈ t.cons ++ t.rows, qv τ3 r = 0
  · simp only [if_pos hC, ψ_add]
    rw [← Finset.sum_mul, ← Finset.mul_sum, QSem.char_sum]
    by_cases hQ : qv τ3 (yk y 1 t.phase) = 0
    · rw [if_pos hQ, if_pos ⟨hQ, hC⟩]; push_cast; ring
    · rw [if_neg hQ, if_neg (fun h => hQ h.1)]; ring
  · simp only [if_neg hC]
    rw [if_neg (fun h => hC h.2)]
    simp

open QSem in
/-- [Elim] (`elimS`) on a mixed path sum, the head variable outside the entries. -/
theorem mval_elim (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (q : PSem.AData) (t : QSem.Data)
    (es : List Ent) (y : ℕ) (ys : List ℕ) (hp : t.path = y :: ys) (h : elimOK t = true)
    (hy : ∀ e ∈ es, e.2.1 ≠ y) (τ : ℕ → ZMod 2) (ρ : ℕ → ZMod 3) :
    mval M c A q t es τ ρ = 3 * mval M c A q (elimS t) es τ ρ := by
  have h' := h
  simp only [elimOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at h'
  obtain ⟨⟨h1, h2⟩, h3⟩ := h'
  unfold mval
  simp only [elimS, hp, List.tail_cons]
  rw [qsum_head _ y ys ρ h1, ← qsum_const_mul]
  apply qsum_congr
  intro τ3
  have hg : ∀ b, liftA es (Function.update τ3 y b) τ = liftA es τ3 τ := fun b =>
    liftA_rho _ _ τ es fun e he => Function.update_of_ne (hy e he) _ _
  simp only [hg, holds_update_not τ3 y _ _ h3, qv_update_not τ3 y _ _ h2]
  rw [Finset.sum_const, Finset.card_univ, ZMod.card, nsmul_eq_mul]
  push_cast
  ring

open QSem in
/-- Normalisation (`normS`) of the Z_3 part of a mixed path sum. -/
theorem mval_norm (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (q : PSem.AData) (t : QSem.Data)
    (es : List Ent) (τ : ℕ → ZMod 2) (ρ : ℕ → ZMod 3) :
    mval M c A q t es τ ρ = mval M c A q (normS 3 t) es τ ρ := by
  unfold mval normS
  apply qsum_congr
  intro τ3
  simp only [← List.map_append, holds_map (normQ 3) (qv_normQ τ3), qv_normQ]

open QSem in
/-- A pin (`pinS`) on a mixed path sum: the head variable `y` set to `x`, the entries read
through it. -/
theorem mval_pin (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (q : PSem.AData) (t : QSem.Data)
    (es : List Ent) (x : ℕ) (h : pinOK 3 x t = true) (τ : ℕ → ZMod 2) (ρ : ℕ → ZMod 3)
    (y : ℕ) (ys : List ℕ) (hp : t.path = y :: ys) (hxy : x ≠ y) :
    mval M c A q t es τ ρ = mval M c A q (pinS 3 x t) (es.map (pinE y x 1 0)) τ ρ := by
  have h' := h
  simp only [pinOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    Option.isSome_iff_exists] at h'
  obtain ⟨h1, ⟨r, hr⟩⟩ := h'
  have hir : r ∈ t.cons := List.mem_of_find?_eq_some hr
  have hpr : pinTo 3 y x r = true := List.find?_some hr
  have hpq : pivOK 3 y r = true := by
    simp only [pinTo, Bool.and_eq_true] at hpr
    exact hpr.1
  have hB : ∀ τ3 : ℕ → ZMod 3, qv τ3 (pivVal 3 y r) = τ3 x := by
    intro τ3
    simp only [pinTo, Bool.and_eq_true, decide_eq_true_eq] at hpr
    rw [← qv_normQ τ3 (pivVal 3 y r), hpr.2]
    simp [qv, qm]
  unfold mval
  simp only [pinS, hp, hr]
  rw [qsum_head _ y ys ρ h1]
  apply qsum_congr
  intro τ3
  set B := pivVal 3 y r
  have hperm := List.perm_cons_erase hir
  have hsplit : ∀ τ' : ℕ → ZMod 3, (∀ r' ∈ t.cons ++ t.rows, qv τ' r' = 0) ↔
      (qv τ' r = 0 ∧ ∀ r' ∈ t.cons.erase r ++ t.rows, qv τ' r' = 0) := by
    intro τ'
    simp only [List.mem_append]
    constructor
    · intro H
      refine ⟨H r (Or.inl hir), fun r' hr' => ?_⟩
      rcases hr' with hr' | hr'
      · exact H r' (Or.inl (List.mem_of_mem_erase hr'))
      · exact H r' (Or.inr hr')
    · rintro ⟨H0, H⟩ r' (hr' | hr')
      · rcases List.mem_cons.1 (hperm.mem_iff.1 hr') with e | hr''
        · rw [e]; exact H0
        · exact H r' (Or.inl hr'')
      · exact H r' (Or.inr hr')
  have hnew : (∀ r' ∈ (t.cons.erase r).map (fun r' => normQ 3 (substQ y B r')) ++
      t.rows.map (fun r' => normQ 3 (substQ y B r')), qv τ3 r' = 0) ↔
      (∀ r' ∈ t.cons.erase r ++ t.rows, qv (Function.update τ3 y (qv τ3 B)) r' = 0) := by
    rw [← List.map_append]
    exact holds_map (fun r' => normQ 3 (substQ y B r')) (fun r' => by rw [qv_normQ, qv_substQ]) _
  have hlift : liftA es (Function.update τ3 y (qv τ3 B)) τ =
      liftA (es.map (pinE y x 1 0)) τ3 τ := by
    have hx : (Function.update τ3 y (qv τ3 B)) y =
        ((1 : ℕ) : ZMod 3) * (Function.update τ3 y (qv τ3 B)) x + ((0 : ℕ) : ZMod 3) := by
      rw [Function.update_self, hB, Function.update_of_ne hxy]
      simp
    rw [← liftA_pinE _ τ y x 1 0 hx es]
    apply liftA_rho
    intro e he
    simp only [List.mem_map] at he
    obtain ⟨e₀, _, rfl⟩ := he
    unfold pinE
    split_ifs with h0
    · exact Function.update_of_ne hxy _ _
    · exact Function.update_of_ne h0 _ _
  simp only [hsplit, piv_iff y r hpq τ3, ite_and]
  rw [Finset.sum_eq_single (qv τ3 B)]
  · simp only [hnew, qv_normQ, qv_substQ, hlift]
    rw [if_pos rfl]
  · intro b _ hb
    rw [if_neg hb, zero_mul]
  · intro hm
    exact absurd (Finset.mem_univ _) hm

end MSem
end ArchytasZX
