/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumSem

/-!
# Path-sum steps [omega], [HHgen], [Rem], [HHnl] and step lists

Each step reads the head path variables of a `Data`; `*OK` is its syntactic side condition, a
`Bool` checked by `decide`, and `*_sound` says the value is unchanged up to the step's constant
factor `fac`. `run_sound` composes a step list.
-/

namespace ArchytasZX
namespace PSem

open Complex
open ArchytasZX.Sym (ex_half)

theorem zq0 : zq (0 : ZMod 2) = 0 := by decide
theorem zq1 : zq (1 : ZMod 2) = 1 := by decide

theorem summand_path (M : ℕ) (c : ℂ) (s : Data) (l : List ℕ) (τ : ℕ → ZMod 2) :
    summand M c { s with path := l } τ = summand M c s τ := rfl

/-- The summand at `y = 0` plus at `y = 1`, for `y` in no constraint or row. -/
theorem summand_pair (M : ℕ) (c : ℂ) (s : Data) (τ : ℕ → ZMod 2) (y : ℕ)
    (hc : y ∉ fvars (s.cons ++ s.rows)) :
    summand M c s (Function.update τ y 0) + summand M c s (Function.update τ y 1) =
      (1 + ex ((pn τ (yq y s.phase) : ℝ) / M)) *
        (if ∀ q ∈ s.cons ++ s.rows, fz τ q = 0 then c * ex ((pn τ (rest y s.phase) : ℝ) / M)
          else 0) := by
  unfold summand
  simp only [holds_update_not τ y _ _ hc]
  rw [pn_yupd τ y 0, pn_yupd τ y 1, zq0, zq1]
  split_ifs
  · have e0 : (((0 * pn τ (yq y s.phase) + pn τ (rest y s.phase) : ℤ) : ℝ)) / M =
        (pn τ (rest y s.phase) : ℝ) / M := by push_cast; ring
    have e1 : (((1 * pn τ (yq y s.phase) + pn τ (rest y s.phase) : ℤ) : ℝ)) / M =
        (pn τ (yq y s.phase) : ℝ) / M + (pn τ (rest y s.phase) : ℝ) / M := by push_cast; ring
    rw [e0, e1, ex_add]
    ring
  · ring

theorem yq_map (τ : ℕ → ZMod 2) (y : ℕ) (P : Poly) :
    ((yq y P).map fun t => zq (mz τ t.1)) = (quot y P).map fun m => zq (mz τ m) := by
  simp [yq, quot, List.map_map, Function.comp_def]

/-- `1 + (-1)^Q` for a head variable with half coefficients. -/
theorem one_add_half (M : ℕ) (hM : 0 < M) (τ : ℕ → ZMod 2) (y : ℕ) (P : Poly)
    (h : halfOK M y P = true) :
    1 + ex ((pn τ (yq y P) : ℝ) / M) = if fz τ (quot y P) = 0 then 2 else 0 := by
  have hall : ∀ t ∈ yq y P, 2 * t.2 = M := by
    intro t ht
    simp only [yq, List.mem_map] at ht
    obtain ⟨u, hu, rfl⟩ := ht
    simp only [halfOK, List.all_eq_true, decide_eq_true_eq] at h
    exact h u hu
  have h2 := pn_allhalf τ M (yq y P) hall
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  rw [yq_map] at h2
  set N := ((quot y P).map fun m => zq (mz τ m)).sum with hN
  have e : ((pn τ (yq y P) : ℤ) : ℝ) / M = (N : ℝ) / 2 := by
    have h2' : (2 : ℝ) * (pn τ (yq y P) : ℝ) = (M : ℝ) * (N : ℝ) := by exact_mod_cast h2
    rw [div_eq_div_iff hM' two_ne_zero]
    linear_combination h2'
  rw [e, ex_halfN, hN, zsum_mz]
  split_ifs <;> norm_num

/-- `ex (x N / 2)` for a bit `x`. -/
theorem ex_zqN (x : ZMod 2) (N : ℤ) :
    ex ((zq x : ℝ) * N / 2) = if x * (N : ZMod 2) = 0 then 1 else -1 := by
  rcases z2_cases x with rfl | rfl
  · rw [zq0]; simp [ex_zero]
  · rw [zq1, Int.cast_one, one_mul, one_mul, ex_halfN]

/-! ## Substitution of a variable throughout -/

/-- `z` replaced by `B` in the phase, the constraints and the rows. -/
def substD (M z : ℕ) (B : F2) (s : Data) : Data :=
  { path := s.path, phase := normP M (substP M z B s.phase),
    cons := s.cons.map fun q => normF (substF z B q),
    rows := s.rows.map fun q => normF (substF z B q) }

theorem summand_substD (M : ℕ) (hM : 2 ≤ M) (c : ℂ) (z : ℕ) (B : F2) (s : Data)
    (τ : ℕ → ZMod 2) :
    summand M c (substD M z B s) τ = summand M c s (Function.update τ z (fz τ B)) := by
  unfold summand substD
  dsimp only
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
    apply ex_peq (by omega)
    exact peq_trans (pn_normP τ M _) (pn_substP τ M z hM B _)
  · rfl

theorem quot_free (y : ℕ) (P : Poly) : ∀ m ∈ quot y P, y ∉ m := by
  intro m hm hy
  simp only [quot, List.mem_map] at hm
  obtain ⟨t, _, rfl⟩ := hm
  simp at hy

theorem fz_free (τ : ℕ → ZMod 2) (y : ℕ) (b : ZMod 2) (Q : F2) (h : ∀ m ∈ Q, y ∉ m) :
    fz (Function.update τ y b) Q = fz τ Q := fz_update_not τ y b Q h

theorem splitAt_free1 (z : ℕ) (Q : F2) : ∀ m ∈ (splitAt z Q).1, z ∉ m := by
  intro m hm hz
  simp only [splitAt, List.mem_map, List.mem_filter] at hm
  obtain ⟨u, _, rfl⟩ := hm
  simp at hz

theorem splitAt_free2 (z : ℕ) (Q : F2) : ∀ m ∈ (splitAt z Q).2, z ∉ m := by
  intro m hm hz
  simp only [splitAt, List.mem_filter, Bool.not_eq_true', decide_eq_false_iff_not] at hm
  exact hm.2 hz

theorem splitAt_sub1 (z : ℕ) (Q : F2) (y : ℕ) (h : ∀ m ∈ Q, y ∉ m) :
    ∀ m ∈ (splitAt z Q).1, y ∉ m := by
  intro m hm hy
  simp only [splitAt, List.mem_map, List.mem_filter] at hm
  obtain ⟨u, ⟨hu, _⟩, rfl⟩ := hm
  exact h u hu (List.mem_of_mem_filter hy)

theorem splitAt_sub2 (z : ℕ) (Q : F2) (y : ℕ) (h : ∀ m ∈ Q, y ∉ m) :
    ∀ m ∈ (splitAt z Q).2, y ∉ m := by
  intro m hm hy
  simp only [splitAt, List.mem_filter] at hm
  exact h m hm.1 hy

/-! ## [HHgen] -/

/-- Whether the head variable `y` carries only half coefficients, its quotient is `z A + B` in
the next variable `z` with `(1 + A)(1 + B) = 0`, `y` in no constraint or row, distinct path. -/
def hhGenOK (M : ℕ) (s : Data) : Bool :=
  match s.path with
  | y :: z :: ys => decide (2 ≤ M) && !decide (y = z) && !decide (y ∈ ys) && !decide (z ∈ ys) &&
      halfOK M y s.phase && !decide (y ∈ fvars (s.cons ++ s.rows)) &&
      decide (normF (f2mul ([] :: (splitAt z (quot y s.phase)).1)
        ([] :: (splitAt z (quot y s.phase)).2)) = [])
  | _ => false

/-- The second variable `z` set to `B` and dropped from the path. -/
def hhGenS (M : ℕ) (s : Data) : Data :=
  match s.path with
  | y :: z :: ys => substD M z (splitAt z (quot y s.phase)).2 { s with path := y :: ys }
  | _ => s

theorem hhgen_pick (a b : ZMod 2) (hab : (1 + a) * (1 + b) = 0) (R : ZMod 2 → ℂ) :
    (if 0 * a + b = 0 then (2 : ℂ) else 0) * R 0 + (if 1 * a + b = 0 then (2 : ℂ) else 0) * R 1 =
      (if b * a + b = 0 then (2 : ℂ) else 0) * R b := by
  have h10 : (1 : ZMod 2) ≠ 0 := by decide
  have h11 : (1 : ZMod 2) + 1 = 0 := by decide
  rcases z2_cases a with rfl | rfl <;> rcases z2_cases b with rfl | rfl
  · exact absurd hab (by decide)
  all_goals simp [h10, h11]

theorem hhGen_sound (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) (h : hhGenOK M s = true) :
    val M c s σ = val M c (hhGenS M s) σ := by
  unfold val
  match hp : s.path, h with
  | [], h => simp [hhGenOK, hp] at h
  | [_], h => simp [hhGenOK, hp] at h
  | y :: z :: ys, h =>
    simp only [hhGenOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at h
    obtain ⟨⟨⟨⟨⟨⟨hM, hyz⟩, hy⟩, hz⟩, hh⟩, hc⟩, hAB⟩ := h
    have hM0 : 0 < M := by omega
    simp only [hhGenS, hp]
    set Q := quot y s.phase with hQ
    set A := (splitAt z Q).1 with hA
    set B := (splitAt z Q).2 with hB
    have hyzs : y ∉ z :: ys := by
      intro hm
      rcases List.mem_cons.1 hm with e | e
      · exact hyz e
      · exact hy e
    show bsum (y :: z :: ys) _ σ = bsum (y :: ys) _ σ
    rw [bsum_head _ y (z :: ys) σ hyzs, bsum_head _ z ys σ hz, bsum_head _ y ys σ hy]
    apply bsum_congr
    intro τ
    have hQy := quot_free y s.phase
    have hBy : ∀ b : ZMod 2, fz (Function.update τ y b) B = fz τ B :=
      fun b => fz_free τ y b B (splitAt_sub2 z Q y hQy)
    simp only [summand_substD M hM c z B, summand_path, hBy]
    rw [Function.update_comm hyz _ _ τ, Function.update_comm hyz _ _ τ]
    have hg : ∀ w : ZMod 2,
        summand M c s (Function.update (Function.update τ z w) y 0) +
          summand M c s (Function.update (Function.update τ z w) y 1) =
        (if w * fz τ A + fz τ B = 0 then (2 : ℂ) else 0) *
          (if ∀ q ∈ s.cons ++ s.rows, fz (Function.update τ z w) q = 0 then
            c * ex ((pn (Function.update τ z w) (rest y s.phase) : ℝ) / M) else 0) := by
      intro w
      rw [summand_pair M c s _ y hc, one_add_half M hM0 _ y s.phase hh, ← hQ,
        fz_splitAt _ z Q, Function.update_self, ← hA, ← hB,
        fz_free τ z w A (splitAt_free1 z Q), fz_free τ z w B (splitAt_free2 z Q)]
    rw [hg 0, hg 1, hg (fz τ B)]
    have hab : (1 + fz τ A) * (1 + fz τ B) = 0 := by
      have := congrArg (fz τ) hAB
      rw [fz_normF, fz_f2mul, fz_cons, fz_cons] at this
      simpa [fz, mz] using this
    exact hhgen_pick (fz τ A) (fz τ B) hab
      (fun w => if ∀ q ∈ s.cons ++ s.rows, fz (Function.update τ z w) q = 0 then
        c * ex ((pn (Function.update τ z w) (rest y s.phase) : ℝ) / M) else 0)

/-! ## [Rem] -/

/-- Whether a term is kept: it reads `y` or misses a variable of `m`. -/
def remKeep (y : ℕ) (m : Mono) (t : Mono × ℕ) : Bool :=
  decide (y ∈ t.1) || !(m.all fun v => decide (v ∈ t.1))

/-- Whether the head variable `y` carries only half coefficients with a one-monomial quotient,
`y` in no constraint or row and not in the tail. -/
def remOK (M : ℕ) (s : Data) : Bool :=
  match s.path with
  | [] => false
  | y :: ys => decide (0 < M) && !decide (y ∈ ys) && halfOK M y s.phase &&
      !decide (y ∈ fvars (s.cons ++ s.rows)) && decide ((quot y s.phase).length = 1)

/-- The terms without `y` divisible by the quotient monomial dropped. -/
def remS (s : Data) : Data :=
  match s.path with
  | [] => s
  | y :: _ => { s with phase := s.phase.filter (remKeep y ((quot y s.phase).headD [])) }

theorem yTerms_filter (y : ℕ) (p : Mono × ℕ → Bool) (hp : ∀ t, y ∈ t.1 → p t = true)
    (P : Poly) : yTerms y (P.filter p) = yTerms y P := by
  simp only [yTerms, List.filter_filter]
  congr 1
  funext t
  by_cases ht : y ∈ t.1 <;> simp [ht, hp]

theorem rest_filter (y : ℕ) (p : Mono × ℕ → Bool) (P : Poly) :
    rest y (P.filter p) = (rest y P).filter p := by
  simp only [rest, List.filter_filter, Bool.and_comm]

theorem quot_filter (y : ℕ) (p : Mono × ℕ → Bool) (hp : ∀ t, y ∈ t.1 → p t = true)
    (P : Poly) : quot y (P.filter p) = quot y P := by
  simp only [quot, yTerms_filter y p hp]

theorem yq_filter (y : ℕ) (p : Mono × ℕ → Bool) (hp : ∀ t, y ∈ t.1 → p t = true)
    (P : Poly) : yq y (P.filter p) = yq y P := by
  simp only [yq, yTerms_filter y p hp]

theorem halfOK_filter (M y : ℕ) (p : Mono × ℕ → Bool) (hp : ∀ t, y ∈ t.1 → p t = true)
    (P : Poly) : halfOK M y (P.filter p) = halfOK M y P := by
  simp only [halfOK, yTerms_filter y p hp]

theorem pn_filter_zero (τ : ℕ → ZMod 2) (p : Mono × ℕ → Bool) :
    ∀ L : Poly, (∀ t ∈ L, p t = false → mz τ t.1 = 0) → pn τ (L.filter p) = pn τ L
  | [] => fun _ => rfl
  | t :: L => fun h => by
    have ih := pn_filter_zero τ p L (fun u hu => h u (List.mem_cons_of_mem _ hu))
    by_cases ht : p t = true
    · rw [List.filter_cons_of_pos ht, pn_cons, pn_cons, ih]
    · have hf : p t = false := by simpa using ht
      rw [List.filter_cons_of_neg ht, pn_cons, ih, h t List.mem_cons_self hf, zq0]
      ring

theorem mz_zero_mem (τ : ℕ → ZMod 2) : ∀ m : Mono, mz τ m = 0 → ∃ v ∈ m, τ v = 0
  | [], h => by simp [mz] at h
  | v :: m, h => by
    have h' : τ v * mz τ m = 0 := h
    rcases z2_cases (τ v) with e | e
    · exact ⟨v, List.mem_cons_self, e⟩
    · rw [e, one_mul] at h'
      obtain ⟨u, hu, e'⟩ := mz_zero_mem τ m h'
      exact ⟨u, List.mem_cons_of_mem _ hu, e'⟩

theorem mz_zero_of_mem (τ : ℕ → ZMod 2) (v : ℕ) (hv : τ v = 0) :
    ∀ t : Mono, v ∈ t → mz τ t = 0
  | [], h => by simp at h
  | u :: t, h => by
    show τ u * mz τ t = 0
    rcases List.mem_cons.1 h with e | e
    · rw [← e, hv, zero_mul]
    · rw [mz_zero_of_mem τ v hv t e, mul_zero]

theorem rem_sound (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) (h : remOK M s = true) :
    val M c s σ = val M c (remS s) σ := by
  unfold val
  rcases hp : s.path with _ | ⟨y, ys⟩
  · simp [remOK, hp] at h
  · simp only [remOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at h
    obtain ⟨⟨⟨⟨hM, hy⟩, hh⟩, hc⟩, hl⟩ := h
    obtain ⟨m, hm⟩ := List.length_eq_one_iff.1 hl
    simp only [remS, hp, hm, List.headD_cons]
    set p := remKeep y m with hpdef
    have hpy : ∀ t, y ∈ t.1 → p t = true := by
      intro t ht; simp [hpdef, remKeep, ht]
    show bsum (y :: ys) _ σ = bsum (y :: ys) _ σ
    rw [bsum_head _ y ys σ hy, bsum_head _ y ys σ hy]
    apply bsum_congr
    intro τ
    have hc' : y ∉ fvars ({ s with phase := s.phase.filter p }.cons ++
        { s with phase := s.phase.filter p }.rows) := hc
    rw [summand_pair M c s τ y hc,
      summand_pair M c ⟨y :: ys, s.phase.filter p, s.cons, s.rows⟩ τ y hc]
    dsimp only
    rw [yq_filter y p hpy, rest_filter, one_add_half M hM τ y s.phase hh, hm]
    by_cases hz : fz τ [m] = 0
    · have hmz : mz τ m = 0 := by simpa [fz] using hz
      obtain ⟨v, hvm, hv⟩ := mz_zero_mem τ m hmz
      rw [pn_filter_zero τ p (rest y s.phase)]
      intro t _ hf
      simp only [hpdef, remKeep, Bool.or_eq_false_iff, decide_eq_false_iff_not,
        Bool.not_eq_false', List.all_eq_true, decide_eq_true_eq] at hf
      exact mz_zero_of_mem τ v hv t.1 (hf.2 v hvm)
    · simp only [if_neg hz, zero_mul]

/-! ## [omega] -/

theorem om8_ex (k : ℕ) : om 8 ^ k = ex ((k : ℝ) / 8) := om_eq_ex 8 k

/-- `1 + i^e (-1)^q` as `sqrt 2 zeta_8^f zeta_8^(2 (4 - e) q)`. -/
theorem omega_id (e : ℕ) (he : e = 1 ∨ e = 3) (b : ZMod 2) :
    1 + om 8 ^ (2 * e) * (if b = 0 then 1 else -1) =
      rt 2 * om 8 ^ (if e = 1 then 1 else 7) * (if b = 0 then 1 else om 8 ^ (2 * (4 - e))) := by
  rcases he with rfl | rfl <;> rcases z2_cases b with rfl | rfl
  · simpa using omega_e1F0
  · have h10 : (1 : ZMod 2) ≠ 0 := by decide
    simp only [if_neg h10]
    have := omega_e1F1
    norm_num at this ⊢
    linear_combination this
  · simpa using omega_e3F0
  · have h10 : (1 : ZMod 2) ≠ 0 := by decide
    simp only [if_neg h10]
    have := omega_e3F1
    norm_num at this ⊢
    linear_combination this

/-- Whether the head variable `y` has the linear coefficient `e M / 4` and half coefficients
elsewhere, `4 ∣ M`, `y` in no constraint or row and not in the tail. -/
def omegaOK (M e : ℕ) (s : Data) : Bool :=
  match s.path with
  | [] => false
  | y :: ys => decide (0 < M) && decide (M % 4 = 0) && !decide (y ∈ ys) &&
      decide (constK (yq y s.phase) % M = e * (M / 4)) &&
      (strip (yq y s.phase)).all (fun t => decide (2 * t.2 = M)) &&
      !decide (y ∈ fvars (s.cons ++ s.rows))

/-- The quotient of the non-linear terms of `y`. -/
def omQ (y : ℕ) (P : Poly) : F2 := (strip (yq y P)).map (·.1)

/-- The head variable summed: its quotient's lift added with coefficient `(4 - e) M / 4`. -/
def omegaS (M e : ℕ) (s : Data) : Data :=
  match s.path with
  | [] => s
  | y :: ys =>
    { path := ys,
      phase := normP M (rest y s.phase ++ (lift M (omQ y s.phase)).map
        fun u => ([] ++ u.1, ((4 - e) * (M / 4)) * u.2)),
      cons := s.cons, rows := s.rows }

theorem ex_div4 (M r e : ℕ) (hM : 0 < M) (hr : M = 4 * r) :
    ex (((e * r : ℕ) : ℝ) / M) = om 8 ^ (2 * e) := by
  rw [om8_ex]
  congr 1
  have hr0 : (r : ℝ) ≠ 0 := by
    intro e0
    have : r = 0 := by exact_mod_cast e0
    omega
  rw [hr]
  push_cast
  field_simp
  ring

theorem omega_sound (M e : ℕ) (hM2 : 2 ≤ M) (he : e = 1 ∨ e = 3) (c : ℂ) (s : Data)
    (σ : ℕ → ZMod 2) (h : omegaOK M e s = true) :
    val M c s σ = val M (rt 2 * om 8 ^ (if e = 1 then 1 else 7) * c) (omegaS M e s) σ := by
  unfold val
  rcases hp : s.path with _ | ⟨y, ys⟩
  · simp [omegaOK, hp] at h
  · simp only [omegaOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq, List.all_eq_true] at h
    obtain ⟨⟨⟨⟨⟨hM, h4⟩, hy⟩, hK⟩, hst⟩, hc⟩ := h
    obtain ⟨r, hr⟩ : ∃ r, M = 4 * r := ⟨M / 4, by omega⟩
    have hr4 : M / 4 = r := by omega
    simp only [omegaS, hp]
    rw [hr4] at hK ⊢
    rw [bsum_head _ y ys σ hy]
    apply bsum_congr
    intro τ
    rw [summand_pair M c s τ y hc]
    unfold summand
    dsimp only
    have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
    set Q := omQ y s.phase with hQ
    have h2 := pn_allhalf τ M (strip (yq y s.phase)) hst
    set N := ((strip (yq y s.phase)).map fun t => zq (mz τ t.1)).sum with hN
    have hNQ : ((N : ℤ) : ZMod 2) = fz τ Q := by
      rw [hN, show ((strip (yq y s.phase)).map fun t => zq (mz τ t.1)) =
        Q.map (fun m => zq (mz τ m)) by simp [hQ, omQ, List.map_map, Function.comp_def],
        zsum_mz]
    have hsplitK := Nat.mod_add_div (constK (yq y s.phase)) M
    rw [hK] at hsplitK
    generalize hj : constK (yq y s.phase) / M = j at hsplitK
    have hA : ex ((pn τ (yq y s.phase) : ℝ) / M) =
        om 8 ^ (2 * e) * (if fz τ Q = 0 then 1 else -1) := by
      rw [pn_strip τ (yq y s.phase), ← hsplitK]
      have h2' : ((pn τ (strip (yq y s.phase)) : ℤ) : ℝ) = (M : ℝ) * (N : ℝ) / 2 := by
        have : (2 : ℝ) * (pn τ (strip (yq y s.phase)) : ℝ) = (M : ℝ) * (N : ℝ) := by
          exact_mod_cast h2
        linarith
      have e1 : ((((e * r + M * j : ℕ) : ℤ) + pn τ (strip (yq y s.phase)) : ℤ) : ℝ) / M =
          ((e * r : ℕ) : ℝ) / M + (N : ℝ) / 2 + ((j : ℤ) : ℝ) := by
        rw [Int.cast_add, h2']
        push_cast
        field_simp
        ring
      rw [e1, ex_int, ex_add, ex_div4 M r e hM hr, ex_halfN, hNQ]
    have hB : ex ((pn τ (normP M (rest y s.phase ++ (lift M Q).map
        fun u => ([] ++ u.1, ((4 - e) * r) * u.2))) : ℝ) / M) =
        ex ((pn τ (rest y s.phase) : ℝ) / M) *
          (if fz τ Q = 0 then 1 else om 8 ^ (2 * (4 - e))) := by
      have hpe : peq M (pn τ (normP M (rest y s.phase ++ (lift M Q).map
          fun u => ([] ++ u.1, ((4 - e) * r) * u.2))))
          (pn τ (rest y s.phase) + (((4 - e) * r : ℕ) : ℤ) * zq (fz τ Q)) := by
        refine peq_trans (pn_normP τ M _) ?_
        rw [pn_append, pn_mapmul]
        have hm1 : zq (mz τ []) = 1 := by simp [mz]; decide
        rw [hm1, mul_one]
        exact peq_add (peq_refl M _) (peq_mul _ (pn_lift τ M hM2 Q))
      rw [ex_peq hM hpe]
      rcases z2_cases (fz τ Q) with hq | hq
      · rw [hq, zq0, mul_zero, add_zero, if_pos rfl, mul_one]
      · rw [hq, zq1, mul_one, Int.cast_add, add_div, ex_add, if_neg (by decide),
          ← ex_div4 M r (4 - e) hM hr]
        norm_cast
    rw [hA, hB]
    have hid := omega_id e he (fz τ Q)
    by_cases hC : ∀ q ∈ s.cons ++ s.rows, fz τ q = 0
    · rw [if_pos hC, if_pos hC]
      linear_combination (c * ex ((pn τ (rest y s.phase) : ℝ) / M)) * hid
    · rw [if_neg hC, if_neg hC]
      ring

/-! ## [HHnl] -/

/-- Whether the head variables `y`, `y'` carry only half coefficients, neither in the other's
quotient nor in a constraint or row, distinct path. -/
def hhNlOK (M : ℕ) (s : Data) : Bool :=
  match s.path with
  | y :: y' :: ys => decide (2 ≤ M) && !decide (y = y') && !decide (y ∈ ys) &&
      !decide (y' ∈ ys) && halfOK M y s.phase && halfOK M y' s.phase &&
      !decide (y ∈ fvars (s.cons ++ s.rows)) && !decide (y' ∈ fvars (s.cons ++ s.rows)) &&
      !decide (y ∈ (quot y' s.phase).flatMap id) && !decide (y' ∈ (quot y s.phase).flatMap id)
  | _ => false

/-- The value `y + y Q` of `y'`. -/
def nlValue (y : ℕ) (P : Poly) : F2 := [y] :: (quot y P).map fun u => [y] ++ u

/-- `y'` set to `y + y Q` and dropped from the path. -/
def hhNlS (M : ℕ) (s : Data) : Data :=
  match s.path with
  | y :: y' :: ys =>
    { path := y :: ys, phase := normP M (substP M y' (nlValue y s.phase) s.phase),
      cons := s.cons, rows := s.rows }
  | _ => s

theorem not_mem_flat {y : ℕ} {Q : F2} (h : y ∉ Q.flatMap id) : ∀ m ∈ Q, y ∉ m := by
  intro m hm hy
  exact h (List.mem_flatMap.2 ⟨m, hm, hy⟩)

theorem yTerms_rest (y y' : ℕ) (P : Poly) (h : ∀ m ∈ quot y P, y' ∉ m) (hne : y ≠ y') :
    yTerms y (rest y' P) = yTerms y P := by
  simp only [yTerms, rest, List.filter_filter]
  apply List.filter_congr
  intro t ht
  by_cases hy : y ∈ t.1
  · have : y' ∉ t.1 := by
      intro hy'
      apply h (t.1.filter (· != y))
      · simp only [quot, yTerms, List.mem_map, List.mem_filter]
        exact ⟨t, ⟨ht, by simpa using hy⟩, rfl⟩
      · simp [hy', Ne.symm hne]
    simp [hy, this]
  · simp [hy]

theorem hhNl_sound (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) (h : hhNlOK M s = true) :
    val M c s σ = val M (2 * c) (hhNlS M s) σ := by
  unfold val
  match hp : s.path, h with
  | [], h => simp [hhNlOK, hp] at h
  | [_], h => simp [hhNlOK, hp] at h
  | y :: y' :: ys, h =>
    simp only [hhNlOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at h
    obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨hM, hne⟩, hy⟩, hy'⟩, hh⟩, hh'⟩, hc⟩, hc'⟩, hq'⟩, hq⟩ := h
    have hM0 : 0 < M := by omega
    have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM0.ne'
    simp only [hhNlS, hp]
    have hyy : y ∉ y' :: ys := by
      intro hm
      rcases List.mem_cons.1 hm with e | e
      · exact hne e
      · exact hy e
    show bsum (y :: y' :: ys) _ σ = bsum (y :: ys) _ σ
    rw [bsum_head _ y (y' :: ys) σ hyy, bsum_head _ y' ys σ hy', bsum_head _ y ys σ hy]
    apply bsum_congr
    intro τ
    set Q := quot y s.phase with hQ
    set Q' := quot y' s.phase with hQ'
    set N := (Q.map fun m => zq (mz τ m)).sum with hN
    set N' := (Q'.map fun m => zq (mz τ m)).sum with hN'
    set R := pn τ (rest y (rest y' s.phase)) with hR
    set K := (if ∀ q ∈ s.cons ++ s.rows, fz τ q = 0 then c * ex ((R : ℝ) / M) else 0) with hK
    have hQy' := not_mem_flat hq
    have hQ'y := not_mem_flat hq'
    have hall : ∀ (z : ℕ), halfOK M z s.phase = true → ∀ t ∈ yq z s.phase, 2 * t.2 = M := by
      intro z hz t ht
      simp only [yq, List.mem_map] at ht
      obtain ⟨u, hu, rfl⟩ := ht
      simp only [halfOK, List.all_eq_true, decide_eq_true_eq] at hz
      exact hz u hu
    have h2 := pn_allhalf τ M (yq y s.phase) (hall y hh)
    have h2' := pn_allhalf τ M (yq y' s.phase) (hall y' hh')
    rw [yq_map] at h2 h2'
    have hv : ∀ b w : ZMod 2,
        summand M c s (Function.update (Function.update τ y b) y' w) =
          K * (ex ((zq w : ℝ) * N' / 2) * ex ((zq b : ℝ) * N / 2)) := by
      intro b w
      unfold summand
      simp only [holds_update_not _ y' _ _ hc', holds_update_not _ y _ _ hc]
      rw [pn_yupd _ y' w, pn_update_not τ y b (yq y' s.phase), pn_yupd τ y b]
      · have hyq : yq y (rest y' s.phase) = yq y s.phase := by
          simp only [yq, yTerms_rest y y' s.phase hQy' hne]
        rw [hyq, ← hR]
        have e1 : ((zq w * pn τ (yq y' s.phase) + (zq b * pn τ (yq y s.phase) + R) : ℤ) : ℝ) /
            M = (zq w : ℝ) * N' / 2 + (zq b : ℝ) * N / 2 + (R : ℝ) / M := by
          have a1 : (2 : ℝ) * (pn τ (yq y s.phase) : ℝ) = (M : ℝ) * (N : ℝ) := by
            exact_mod_cast h2
          have a2 : (2 : ℝ) * (pn τ (yq y' s.phase) : ℝ) = (M : ℝ) * (N' : ℝ) := by
            exact_mod_cast h2'
          push_cast
          field_simp
          linear_combination (zq w : ℝ) * a2 + (zq b : ℝ) * a1
        rw [e1, ex_add, ex_add, hK]
        split_ifs <;> ring
      · intro t ht hyt
        simp only [yq, List.mem_map] at ht
        obtain ⟨u, hu, rfl⟩ := ht
        apply hQ'y (u.1.filter (· != y'))
        · simp only [hQ', quot, List.mem_map]; exact ⟨u, hu, rfl⟩
        · exact hyt
    have hr : ∀ b : ZMod 2,
        summand M (2 * c) ⟨y :: ys, normP M (substP M y' (nlValue y s.phase) s.phase),
          s.cons, s.rows⟩ (Function.update τ y b) =
        2 * summand M c s (Function.update (Function.update τ y b) y'
          (b + b * fz τ Q)) := by
      intro b
      have hval : fz (Function.update τ y b) (nlValue y s.phase) = b + b * fz τ Q := by
        rw [nlValue, fz_cons, fz_mapprefix, fz_free τ y b Q (quot_free y s.phase)]
        simp [mz, Function.update_self]
      unfold summand
      dsimp only
      simp only [holds_update_not _ y' _ _ hc']
      rw [← hval]
      split_ifs
      · rw [ex_peq hM0 (peq_trans (pn_normP _ M _) (pn_substP _ M y' hM _ _))]
        ring
      · ring
    rw [hr 0, hr 1]
    rw [Function.update_comm (Ne.symm hne) _ _ τ, Function.update_comm (Ne.symm hne) _ _ τ,
      Function.update_comm (Ne.symm hne) _ _ τ, Function.update_comm (Ne.symm hne) _ _ τ]
    rw [hv 0 0, hv 0 1, hv 1 0, hv 1 1, hv 0, hv 1]
    simp only [ex_zqN]
    have hn : ((N : ℤ) : ZMod 2) = fz τ Q := by rw [hN, zsum_mz]
    have hn' : ((N' : ℤ) : ZMod 2) = fz τ Q' := by rw [hN', zsum_mz]
    rw [hn, hn']
    have h10 : (1 : ZMod 2) ≠ 0 := by decide
    have h11 : (1 : ZMod 2) + 1 = 0 := by decide
    rcases z2_cases (fz τ Q) with e1 | e1 <;> rcases z2_cases (fz τ Q') with e2 | e2 <;>
      simp [e1, e2, h10, h11] <;> ring

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
  | omega
  | omegaConj
  | hhGen
  | rem
  | hhNl

/-- A step's side condition. -/
def stepOK (M : ℕ) : Step → Data → Bool
  | .perm l, s => decide (s.path.Perm l)
  | .elim, s => elimOK s
  | .hhCons, s => hhConsOK M s
  | .hhPivot, s => hhPivotOK M s
  | .canon v, s => canonOK v s
  | .gather, s => gatherOK s
  | .bcanon v, s => bcanonOK v s
  | .omega, s => omegaOK M 1 s
  | .omegaConj, s => omegaOK M 3 s
  | .hhGen, s => hhGenOK M s
  | .rem, s => remOK M s
  | .hhNl, s => hhNlOK M s

/-- A step applied. -/
def stepS (M : ℕ) : Step → Data → Data
  | .perm l, s => permS l s
  | .elim, s => elimS s
  | .hhCons, s => hhConsS s
  | .hhPivot, s => hhPivotS M s
  | .canon v, s => canonS M v s
  | .gather, s => gatherS s
  | .bcanon v, s => bcanonS M v s
  | .omega, s => omegaS M 1 s
  | .omegaConj, s => omegaS M 3 s
  | .hhGen, s => hhGenS M s
  | .rem, s => remS s
  | .hhNl, s => hhNlS M s

/-- The factor a step multiplies the constant by. -/
noncomputable def fac : Step → ℂ
  | .elim | .hhCons | .hhPivot | .hhNl => 2
  | .omega => rt 2 * om 8
  | .omegaConj => rt 2 * om 8 ^ 7
  | .perm _ | .canon _ | .gather | .bcanon _ | .hhGen | .rem => 1

/-- Every side condition along a step list. -/
def runOK (M : ℕ) : List Step → Data → Bool
  | [], _ => true
  | st :: sts, s => stepOK M st s && runOK M sts (stepS M st s)

/-- A step list applied. -/
def run (M : ℕ) : List Step → Data → Data
  | [], s => s
  | st :: sts, s => run M sts (stepS M st s)

/-- The product of the steps' factors. -/
noncomputable def facs : List Step → ℂ
  | [] => 1
  | st :: sts => fac st * facs sts

theorem step_sound (M : ℕ) (hM : 2 ≤ M) (st : Step) (s : Data) (c : ℂ) (σ : ℕ → ZMod 2)
    (h : stepOK M st s = true) : val M c s σ = val M (fac st * c) (stepS M st s) σ := by
  cases st with
  | perm l => simpa [fac, stepS] using perm_sound M c l s σ h
  | elim => simpa [fac, stepS] using elim_sound M c s σ h
  | hhCons => simpa [fac, stepS] using hhCons_sound M c s σ h
  | hhPivot => simpa [fac, stepS] using hhPivot_sound M c s σ h
  | canon v => simpa [fac, stepS] using canon_sound M hM c v s σ h
  | gather => simpa [fac, stepS] using gather_sound M c s σ h
  | bcanon v => simpa [fac, stepS] using bcanon_sound M hM c v s σ h
  | omega => simpa [fac, stepS] using omega_sound M 1 hM (Or.inl rfl) c s σ h
  | omegaConj => simpa [fac, stepS] using omega_sound M 3 hM (Or.inr rfl) c s σ h
  | hhGen => simpa [fac, stepS] using hhGen_sound M c s σ h
  | rem => simpa [fac, stepS] using rem_sound M c s σ h
  | hhNl => simpa [fac, stepS] using hhNl_sound M c s σ h

theorem run_sound (M : ℕ) (hM : 2 ≤ M) :
    ∀ (sts : List Step) (s : Data) (c : ℂ) (σ : ℕ → ZMod 2), runOK M sts s = true →
      val M c s σ = val M (facs sts * c) (run M sts s) σ
  | [], s, c, σ, _ => by simp [facs, run]
  | st :: sts, s, c, σ, h => by
    simp only [runOK, Bool.and_eq_true] at h
    rw [step_sound M hM st s c σ h.1, run_sound M hM sts _ _ σ h.2]
    simp only [facs, run]
    ring_nf

end PSem
end ArchytasZX
