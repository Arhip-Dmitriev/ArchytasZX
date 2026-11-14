/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumSem

/-!
# Path-sum steps [omega], [HHgen], [Rem], [HHnl], [Dyadic], pin, norm, phase drop and step lists

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

/-! ## [pin] and normalisation -/

/-- Whether `q` is `[v]` plus `B`, `B` free of `v`. -/
def pinAt (v : ℕ) (B : F2) (q : F2) : Bool := pivotable v q && decide (q.erase [v] = B)

/-- Whether the head variable `v` is not in the tail and some constraint is `[v] + B`. -/
def pinOK (B : F2) (s : Data) : Bool :=
  match s.path with
  | [] => false
  | v :: ys => !decide (v ∈ ys) && (s.cons.find? (pinAt v B)).isSome

/-- The head variable `v` set to `B` throughout and dropped, its constraint removed. -/
def pinS (M : ℕ) (B : F2) (s : Data) : Data :=
  match s.path with
  | [] => s
  | v :: ys =>
    match s.cons.find? (pinAt v B) with
    | none => s
    | some q =>
      { path := ys, phase := normP M (substP M v B s.phase),
        cons := (s.cons.erase q).map fun r => normF (substF v B r),
        rows := s.rows.map fun r => normF (substF v B r) }

theorem pin_sound (M : ℕ) (hM : 2 ≤ M) (c : ℂ) (B : F2) (s : Data) (σ : ℕ → ZMod 2)
    (h : pinOK B s = true) : val M c s σ = val M c (pinS M B s) σ := by
  unfold val
  rcases hp : s.path with _ | ⟨v, ys⟩
  · simp [pinOK, hp] at h
  · simp only [pinOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      Option.isSome_iff_exists] at h
    obtain ⟨hv, ⟨q, hq⟩⟩ := h
    have hiq : q ∈ s.cons := List.mem_of_find?_eq_some hq
    have hpq : pinAt v B q = true := List.find?_some hq
    simp only [pinAt, pivotable, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at hpq
    obtain ⟨⟨hvq, hvB⟩, hB⟩ := hpq
    subst hB
    have hM0 : 0 < M := by omega
    simp only [pinS, hp, hq]
    rw [bsum_head _ v ys σ hv]
    apply bsum_congr
    intro τ
    set B := q.erase [v] with hBdef
    have hBv : ∀ m ∈ B, v ∉ m := by
      intro m hm hvm
      exact hvB (List.mem_flatMap.2 ⟨m, hm, hvm⟩)
    have hfB : ∀ w : ZMod 2, fz (Function.update τ v w) B = fz τ B :=
      fun w => fz_update_not τ v w B hBv
    have hqv : ∀ w : ZMod 2, fz (Function.update τ v w) q = w + fz τ B := by
      intro w
      rw [fz_perm _ (List.perm_cons_erase hvq), fz_cons, ← hBdef, hfB]
      simp [mz, Function.update_self]
    have hperm := List.perm_cons_erase hiq
    have key : ∀ w : ZMod 2, summand M c s (Function.update τ v w) =
        if w = fz τ B then summand M c
          { path := ys, phase := normP M (substP M v B s.phase),
            cons := (s.cons.erase q).map fun r => normF (substF v B r),
            rows := s.rows.map fun r => normF (substF v B r) } τ else 0 := by
      intro w
      unfold summand
      by_cases hw : w = fz τ B
      · rw [if_pos hw]
        subst hw
        have hsub : ∀ r, fz τ (normF (substF v B r)) = fz (Function.update τ v (fz τ B)) r := by
          intro r
          rw [fz_normF, fz_substF]
        have hcond : (∀ r ∈ (s.cons.erase q).map (fun r => normF (substF v B r)) ++
            s.rows.map (fun r => normF (substF v B r)), fz τ r = 0) ↔
            (∀ r ∈ s.cons ++ s.rows, fz (Function.update τ v (fz τ B)) r = 0) := by
          simp only [List.mem_append, List.mem_map]
          constructor
          · rintro H r (hr | hr)
            · rcases List.mem_cons.1 (hperm.mem_iff.1 hr) with e | hr'
              · rw [e, hqv]; exact z2_add_self _
              · have := H _ (Or.inl ⟨r, hr', rfl⟩)
                rwa [hsub] at this
            · have := H _ (Or.inr ⟨r, hr, rfl⟩)
              rwa [hsub] at this
          · rintro H r (⟨r', hr', rfl⟩ | ⟨r', hr', rfl⟩)
            · rw [hsub]; exact H r' (Or.inl (List.mem_of_mem_erase hr'))
            · rw [hsub]; exact H r' (Or.inr hr')
        simp only [hcond]
        split_ifs
        · congr 1
          exact (ex_peq hM0 (peq_trans (pn_normP τ M _) (pn_substP τ M v hM B s.phase))).symm
        · rfl
      · rw [if_neg hw]
        have hne : ¬ (∀ r ∈ s.cons ++ s.rows, fz (Function.update τ v w) r = 0) := by
          intro H
          apply hw
          have := H q (List.mem_append_left _ hiq)
          rw [hqv] at this
          exact z2_eq_of_add _ _ this
        rw [if_neg hne]
    rw [key 0, key 1]
    rcases z2_cases (fz τ B) with e | e <;> rw [e] <;> simp

/-- The phase and every constraint and row normalised. -/
def normS (M : ℕ) (s : Data) : Data :=
  { path := s.path, phase := normP M s.phase, cons := s.cons.map normF, rows := s.rows.map normF }

theorem norm_sound (M : ℕ) (hM : 0 < M) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) :
    val M c s σ = val M c (normS M s) σ := by
  unfold val normS
  apply bsum_congr
  intro τ
  unfold summand
  dsimp only
  have hcond : (∀ r ∈ s.cons.map normF ++ s.rows.map normF, fz τ r = 0) ↔
      (∀ r ∈ s.cons ++ s.rows, fz τ r = 0) := by
    simp only [List.mem_append, List.mem_map]
    constructor
    · rintro H r (hr | hr)
      · have := H _ (Or.inl ⟨r, hr, rfl⟩); rwa [fz_normF] at this
      · have := H _ (Or.inr ⟨r, hr, rfl⟩); rwa [fz_normF] at this
    · rintro H r (⟨r', hr', rfl⟩ | ⟨r', hr', rfl⟩)
      · rw [fz_normF]; exact H r' (Or.inl hr')
      · rw [fz_normF]; exact H r' (Or.inr hr')
  simp only [hcond]
  split_ifs
  · congr 1
    exact (ex_peq hM (pn_normP τ M s.phase)).symm
  · rfl

theorem val_smul (M : ℕ) (a c : ℂ) (s : Data) (σ : ℕ → ZMod 2) :
    val M (a * c) s σ = a * val M c s σ := by
  unfold val
  rw [← bsum_smul]
  apply bsum_congr
  intro τ
  unfold summand
  split_ifs <;> ring



theorem bsum_append (f : (ℕ → ZMod 2) → ℂ) :
    ∀ (xs ys : List ℕ) (σ : ℕ → ZMod 2), bsum (xs ++ ys) f σ = bsum xs (bsum ys f) σ
  | [], _, _ => rfl
  | x :: xs, ys, σ => by simp only [List.cons_append, bsum, bsum_append f xs ys]

theorem bsum_mul_left (g f : (ℕ → ZMod 2) → ℂ) :
    ∀ (ys : List ℕ), (∀ τ, ∀ y ∈ ys, ∀ b, g (Function.update τ y b) = g τ) →
      ∀ σ, bsum ys (fun τ => g τ * f τ) σ = g σ * bsum ys f σ
  | [], _, _ => rfl
  | y :: ys, hg, σ => by
    have hg' : ∀ τ, ∀ z ∈ ys, ∀ b, g (Function.update τ z b) = g τ :=
      fun τ z hz b => hg τ z (List.mem_cons_of_mem _ hz) b
    simp only [bsum, bsum_mul_left g f ys hg', hg σ y List.mem_cons_self]
    ring

/-! ## [Dyadic] -/

/-- `P` with every coefficient negated modulo `N`. -/
def negP (N : ℕ) (P : Poly) : Poly := P.map fun t => (t.1, N - t.2 % N)

theorem pn_negP (τ : ℕ → ZMod 2) (N : ℕ) (hN : 0 < N) :
    ∀ P : Poly, peq N (pn τ (negP N P)) (-(pn τ P))
  | [] => peq_refl N 0
  | t :: P => by
    have ih := pn_negP τ N hN P
    simp only [negP, List.map_cons] at ih ⊢
    rw [pn_cons, pn_cons]
    obtain ⟨n, hn⟩ := ih
    have hlt : t.2 % N ≤ N := (Nat.mod_lt _ hN).le
    have hd := Nat.mod_add_div t.2 N
    refine ⟨n + zq (mz τ t.1) * (1 + (t.2 / N : ℕ)), ?_⟩
    have hc : ((N - t.2 % N : ℕ) : ℤ) = (N : ℤ) - ((t.2 % N : ℕ) : ℤ) := by
      push_cast [Nat.cast_sub hlt]; ring
    have hd' : ((t.2 % N : ℕ) : ℤ) + (N : ℤ) * ((t.2 / N : ℕ) : ℤ) = (t.2 : ℤ) := by
      exact_mod_cast hd
    rw [hc, hn]
    linear_combination (-(zq (mz τ t.1))) * hd'

/-- The binary digits of `S` modulo `2^k` as F_2 polynomials, low digit first; none when a
remainder is odd. -/
def dyBits : ℕ → Poly → Option (List F2)
  | 0, _ => some []
  | k + 1, S =>
    let b : F2 := (S.filter fun t => t.2 % 2 = 1).map (·.1)
    let D := normP (2 ^ (k + 1)) (S ++ negP (2 ^ (k + 1)) (lift (2 ^ (k + 1)) b))
    if D.all (fun t => t.2 % 2 = 0) then
      (dyBits k (D.map fun t => (t.1, t.2 / 2))).map (b :: ·)
    else none

theorem pn_odd (τ : ℕ → ZMod 2) :
    ∀ S : Poly, ((pn τ S : ℤ) : ZMod 2) = fz τ ((S.filter fun t => t.2 % 2 = 1).map (·.1))
  | [] => by simp [pn, fz]
  | t :: S => by
    have ih := pn_odd τ S
    rw [pn_cons]
    push_cast
    rw [ih]
    have hz : ((zq (mz τ t.1) : ℤ) : ZMod 2) = mz τ t.1 := zq_cast _
    by_cases h : t.2 % 2 = 1
    · rw [List.filter_cons_of_pos (by simpa using h), List.map_cons, fz_cons, hz]
      have : ((t.2 : ℕ) : ZMod 2) = 1 := by
        rw [← ZMod.natCast_mod, h]
        rfl
      rw [this, one_mul]
    · rw [List.filter_cons_of_neg (by simpa using h)]
      have h0 : t.2 % 2 = 0 := by omega
      have : ((t.2 : ℕ) : ZMod 2) = 0 := by
        rw [← ZMod.natCast_mod, h0]
        rfl
      rw [this, zero_mul, zero_add]

theorem pn_half_all (τ : ℕ → ZMod 2) :
    ∀ D : Poly, D.all (fun t => t.2 % 2 = 0) = true →
      pn τ D = 2 * pn τ (D.map fun t => (t.1, t.2 / 2))
  | [] => fun _ => by simp [pn]
  | t :: D => fun h => by
    simp only [List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at h
    rw [List.map_cons, pn_cons, pn_cons, pn_half_all τ D h.2]
    have := Nat.mod_add_div t.2 2
    rw [h.1, zero_add] at this
    have h' : (t.2 : ℤ) = 2 * ((t.2 / 2 : ℕ) : ℤ) := by exact_mod_cast this.symm
    rw [h']
    ring

theorem zq_le (x : ZMod 2) : zq x = 0 ∨ zq x = 1 := by
  rcases z2_cases x with rfl | rfl <;> decide

theorem dy_bits (τ : ℕ → ZMod 2) :
    ∀ (k : ℕ) (S : Poly) (bs : List F2), dyBits k S = some bs →
      (((2 ^ k : ℕ) : ℤ) ∣ pn τ S ↔ ∀ b ∈ bs, fz τ b = 0)
  | 0, S, bs, h => by
    simp only [dyBits, Option.some.injEq] at h
    subst h
    simp
  | k + 1, S, bs, h => by
    simp only [dyBits] at h
    set N := 2 ^ (k + 1) with hN
    set b : F2 := (S.filter fun t => t.2 % 2 = 1).map (·.1) with hb
    set D := normP N (S ++ negP N (lift N b)) with hD
    split_ifs at h with hev
    obtain ⟨bs', hbs', rfl⟩ := Option.map_eq_some_iff.1 h
    have ih := dy_bits τ k _ bs' hbs'
    have hN2 : 2 ≤ N := by
      rw [hN, pow_succ]
      have : 1 ≤ 2 ^ k := Nat.one_le_two_pow
      omega
    have hN0 : 0 < N := by omega
    have hpe : peq N (pn τ D) (pn τ S - zq (fz τ b)) := by
      refine peq_trans (pn_normP τ N _) ?_
      rw [pn_append]
      obtain ⟨n1, h1⟩ := pn_negP τ N hN0 (lift N b)
      obtain ⟨n2, h2⟩ := pn_lift τ N hN2 b
      refine ⟨n1 - n2, ?_⟩
      rw [h1, h2]
      ring
    obtain ⟨n, hn⟩ := hpe
    have hdiv := pn_half_all τ D hev
    have hNk : ((N : ℕ) : ℤ) = 2 * ((2 ^ k : ℕ) : ℤ) := by
      rw [hN, pow_succ]; push_cast; ring
    have hodd : ((pn τ S : ℤ) : ZMod 2) = fz τ b := pn_odd τ S
    simp only [List.forall_mem_cons]
    rcases z2_cases (fz τ b) with hz | hz
    · have hzq : zq (fz τ b) = 0 := by rw [hz]; decide
      rw [hzq, sub_zero] at hn
      rw [hz]
      simp only [true_and]
      rw [← ih]
      constructor
      · rintro ⟨q, hq⟩
        refine ⟨q + n, ?_⟩
        have hq' : pn τ S = 2 * ((2 ^ k : ℕ) : ℤ) * q := by rw [hq, ← hNk]
        rw [hNk] at hn
        linarith
      · rintro ⟨q, hq⟩
        refine ⟨q - n, ?_⟩
        rw [hNk] at hn
        show pn τ S = ((N : ℕ) : ℤ) * (q - n)
        rw [hNk]
        linarith
    · have hzq : zq (fz τ b) = 1 := by rw [hz]; decide
      constructor
      · rintro ⟨q, hq⟩
        exfalso
        have h2 : ((pn τ S : ℤ) : ZMod 2) = 0 := by
          rw [hq, hNk]; push_cast; rw [show (2 : ZMod 2) = 0 by decide]; ring
        rw [hodd, hz] at h2
        exact absurd h2 (by decide)
      · rintro ⟨h0, _⟩
        rw [hz] at h0
        exact absurd h0 (by decide)

/-- The product `Prod_a (1 + ex (w 2^a N / K))` over `k` digits, weights doubling from `w`. -/
noncomputable def dyProd (N : ℤ) (K : ℕ) : ℕ → ℕ → ℂ
  | 0, _ => 1
  | k + 1, w => (1 + ex ((w : ℝ) * N / K)) * dyProd N K k (2 * w)

theorem dyProd_double (N : ℤ) (K : ℕ) (hK : 0 < K) :
    ∀ (k w : ℕ), dyProd N (2 * K) k (2 * w) = dyProd N K k w
  | 0, _ => rfl
  | k + 1, w => by
    simp only [dyProd]
    rw [dyProd_double N K hK k (2 * w)]
    congr 2
    have : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
    push_cast
    field_simp

theorem dyProd_geom (N : ℤ) :
    ∀ k : ℕ, dyProd N (2 ^ k) k 1 = if ((2 ^ k : ℕ) : ℤ) ∣ N then (2 ^ k : ℂ) else 0
  | 0 => by simp [dyProd]
  | k + 1 => by
    simp only [dyProd]
    rw [show 2 ^ (k + 1) = 2 * 2 ^ k by ring, show (2 * 1 : ℕ) = 2 * 1 from rfl,
      dyProd_double N (2 ^ k) (by positivity) k 1, dyProd_geom N k]
    by_cases hk : ((2 ^ k : ℕ) : ℤ) ∣ N
    · obtain ⟨m, rfl⟩ := hk
      rw [if_pos ⟨m, rfl⟩]
      have e : ((1 : ℕ) : ℝ) * (((2 ^ k : ℕ) : ℤ) * m : ℤ) / ((2 * 2 ^ k : ℕ) : ℝ) =
          (m : ℝ) / 2 := by
        have : ((2 ^ k : ℕ) : ℝ) ≠ 0 := by positivity
        push_cast
        field_simp
      rw [e, ex_halfN]
      by_cases hm : (m : ZMod 2) = 0
      · rw [if_pos hm]
        have hm2 : (2 : ℤ) ∣ m := (ZMod.intCast_zmod_eq_zero_iff_dvd m 2).1 hm
        obtain ⟨r, rfl⟩ := hm2
        rw [if_pos ⟨r, by push_cast; ring⟩]
        push_cast
        ring
      · rw [if_neg hm]
        rw [if_neg]
        · ring
        rintro ⟨r, hr⟩
        apply hm
        have : (((2 ^ k : ℕ) : ℤ)) * m = ((2 ^ k : ℕ) : ℤ) * (2 * r) := by
          rw [hr]; push_cast; ring
        have hpos : ((2 ^ k : ℕ) : ℤ) ≠ 0 := by positivity
        have := mul_left_cancel₀ hpos this
        rw [this]
        push_cast
        rw [show (2 : ZMod 2) = 0 by decide, zero_mul]
    · rw [if_neg hk, mul_zero, if_neg]
      rintro ⟨r, hr⟩
      apply hk
      exact ⟨2 * r, by rw [hr]; push_cast; ring⟩


/-- Whether a monomial reads a variable of `ds`. -/
def hasAny (ds : List ℕ) (m : Mono) : Bool := m.any fun v => decide (v ∈ ds)

/-- The terms `d_a S 2^a w / K` of the digits `ds`, in units of `1 / M`. -/
def dyTerms (M K : ℕ) (S : Poly) : List ℕ → ℕ → Poly
  | [], _ => []
  | d :: ds, w => S.map (fun t => ([d] ++ t.1, (w * (M / K)) * t.2)) ++ dyTerms M K S ds (2 * w)

/-- The terms `f_j b_j / 2` of the fresh variables, in units of `1 / M`. -/
def dyNew (M : ℕ) : List ℕ → List F2 → Poly
  | f :: fs, b :: bs =>
    (b.map fun m => (m, 1)).map (fun t => ([f] ++ t.1, (M / 2) * t.2)) ++ dyNew M fs bs
  | _, _ => []

theorem pn_filter_split (τ : ℕ → ZMod 2) (p : Mono × ℕ → Bool) :
    ∀ P : Poly, pn τ P = pn τ (P.filter fun t => !p t) + pn τ (P.filter p)
  | [] => by simp [pn]
  | t :: P => by
    have ih := pn_filter_split τ p P
    by_cases h : p t = true
    · rw [List.filter_cons_of_neg (by simp [h]), List.filter_cons_of_pos h, pn_cons, pn_cons, ih]
      ring
    · rw [List.filter_cons_of_pos (by simpa using h), List.filter_cons_of_neg h, pn_cons, pn_cons,
        ih]
      ring

theorem not_hasAny {ds : List ℕ} {m : Mono} (h : hasAny ds m = false) : ∀ d ∈ ds, d ∉ m := by
  intro d hd hm
  have : hasAny ds m = true := List.any_eq_true.2 ⟨d, hm, by simpa using hd⟩
  rw [h] at this
  exact absurd this (by decide)

theorem pn_update_free (τ : ℕ → ZMod 2) (y : ℕ) (b : ZMod 2) (P : Poly)
    (h : ∀ t ∈ P, y ∉ t.1) : pn (Function.update τ y b) P = pn τ P := pn_update_not τ y b P h

theorem dyProd_succ (N : ℤ) (K k w : ℕ) :
    dyProd N K (k + 1) w = (1 + ex ((w : ℝ) * N / K)) * dyProd N K k (2 * w) := rfl

theorem bsum_dy (M K q : ℕ) (hM : 0 < M) (hK : 0 < K) (hq : M = K * q) (S : Poly) :
    ∀ (ds : List ℕ) (w : ℕ), ds.Nodup → (∀ t ∈ S, ∀ d ∈ ds, d ∉ t.1) → ∀ τ : ℕ → ZMod 2,
      bsum ds (fun τ' => ex ((pn τ' (dyTerms M K S ds w) : ℝ) / M)) τ =
        dyProd (pn τ S) K ds.length w
  | [], w, _, _, τ => by simp [bsum, dyTerms, pn, ex_zero, dyProd]
  | d :: ds, w, hnd, hS, τ => by
    obtain ⟨hd, hnd'⟩ := List.nodup_cons.1 hnd
    have hS' : ∀ t ∈ S, ∀ e ∈ ds, e ∉ t.1 := fun t ht e he => hS t ht e (List.mem_cons_of_mem _ he)
    have ih := bsum_dy M K q hM hK hq S ds (2 * w) hnd' hS'
    have hSd : ∀ t ∈ S, d ∉ t.1 := fun t ht => hS t ht d List.mem_cons_self
    have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
    have hK' : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
    have hdivq : M / K = q := by rw [hq, Nat.mul_div_cancel_left _ hK]
    simp only [bsum, List.length_cons, dyProd_succ]
    have split : ∀ b : ZMod 2,
        bsum ds (fun τ' => ex ((pn τ' (dyTerms M K S (d :: ds) w) : ℝ) / M))
          (Function.update τ d b) =
        ex ((zq b : ℝ) * w * pn τ S / K) * dyProd (pn τ S) K ds.length (2 * w) := by
      intro b
      have hform : ∀ τ' : ℕ → ZMod 2,
          ex ((pn τ' (dyTerms M K S (d :: ds) w) : ℝ) / M) =
            ex ((zq (τ' d) : ℝ) * w * pn τ' S / K) *
              ex ((pn τ' (dyTerms M K S ds (2 * w)) : ℝ) / M) := by
        intro τ'
        simp only [dyTerms]
        rw [pn_append, pn_mapmul, ← ex_add]
        congr 1
        have hmz : zq (mz τ' [d]) = zq (τ' d) := by simp [mz]
        rw [hmz, hdivq]
        have hq' : (M : ℝ) = K * q := by rw [hq]; push_cast; ring
        have hq0 : (q : ℝ) ≠ 0 := by
          intro e
          apply hM'
          rw [hq', e, mul_zero]
        push_cast
        rw [hq']
        field_simp
      rw [bsum_congr _ _ hform]
      have hg : ∀ τ', ∀ y ∈ ds, ∀ v,
          (fun τ'' : ℕ → ZMod 2 => ex ((zq (τ'' d) : ℝ) * w * pn τ'' S / K))
            (Function.update τ' y v) =
          (fun τ'' : ℕ → ZMod 2 => ex ((zq (τ'' d) : ℝ) * w * pn τ'' S / K)) τ' := by
        intro τ' y hy v
        have hdy : d ≠ y := fun e => hd (e ▸ hy)
        simp only [Function.update_of_ne hdy]
        rw [pn_update_free τ' y v S (fun t ht => hS' t ht y hy)]
      rw [bsum_mul_left _ _ ds hg, ih, Function.update_self,
        pn_update_free τ d b S hSd]
    rw [split 0, split 1, zq0, zq1]
    push_cast
    rw [zero_mul, zero_mul, zero_div, ex_zero]
    ring

theorem prod_ite (τ : ℕ → ZMod 2) :
    ∀ bs : List F2, (bs.map fun b => if fz τ b = 0 then (2 : ℂ) else 0).prod =
      if ∀ b ∈ bs, fz τ b = 0 then (2 : ℂ) ^ bs.length else 0
  | [] => by simp
  | b :: bs => by
    rw [List.map_cons, List.prod_cons, prod_ite τ bs]
    by_cases h : fz τ b = 0
    · simp only [h, if_true, List.forall_mem_cons, true_and, List.length_cons]
      split_ifs <;> ring
    · simp [h]

theorem bsum_new (M r : ℕ) (hM : 0 < M) (hr : M = 2 * r) :
    ∀ (fs : List ℕ) (bs : List F2), fs.Nodup → fs.length = bs.length →
      (∀ f ∈ fs, ∀ b ∈ bs, ∀ m ∈ b, f ∉ m) → ∀ τ : ℕ → ZMod 2,
      bsum fs (fun τ' => ex ((pn τ' (dyNew M fs bs) : ℝ) / M)) τ =
        (bs.map fun b => if fz τ b = 0 then (2 : ℂ) else 0).prod
  | [], [], _, _, _, τ => by simp [bsum, dyNew, pn, ex_zero]
  | [], _ :: _, _, h, _, _ => by simp at h
  | _ :: _, [], _, h, _, _ => by simp at h
  | f :: fs, b :: bs, hnd, hl, hfree, τ => by
    obtain ⟨hf, hnd'⟩ := List.nodup_cons.1 hnd
    have hl' : fs.length = bs.length := by simpa using hl
    have hfree' : ∀ g ∈ fs, ∀ c ∈ bs, ∀ m ∈ c, g ∉ m := fun g hg c hc =>
      hfree g (List.mem_cons_of_mem _ hg) c (List.mem_cons_of_mem _ hc)
    have ih := bsum_new M r hM hr fs bs hnd' hl' hfree'
    have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
    have hr2 : M / 2 = r := by omega
    set N : (ℕ → ZMod 2) → ℤ := fun τ' => pn τ' (b.map fun m => (m, 1)) with hN
    have hNinv : ∀ τ', ∀ g ∈ f :: fs, ∀ v, N (Function.update τ' g v) = N τ' := by
      intro τ' g hg v
      simp only [hN]
      apply pn_update_free
      intro t ht
      simp only [List.mem_map] at ht
      obtain ⟨m, hm, rfl⟩ := ht
      exact hfree g hg b List.mem_cons_self m hm
    have hNb : ∀ τ', ((N τ' : ℤ) : ZMod 2) = fz τ' b := by
      intro τ'
      simp only [hN]
      have : pn τ' (b.map fun m => (m, 1)) = (b.map fun m => zq (mz τ' m)).sum := by
        simp [pn, List.map_map, Function.comp_def]
      rw [this, zsum_mz]
    have hform : ∀ τ' : ℕ → ZMod 2,
        ex ((pn τ' (dyNew M (f :: fs) (b :: bs)) : ℝ) / M) =
          ex ((zq (τ' f) : ℝ) * N τ' / 2) * ex ((pn τ' (dyNew M fs bs) : ℝ) / M) := by
      intro τ'
      simp only [dyNew]
      rw [pn_append, pn_mapmul, ← ex_add]
      congr 1
      have hmz : zq (mz τ' [f]) = zq (τ' f) := by simp [mz]
      rw [hmz, hr2]
      have hr' : (M : ℝ) = 2 * r := by rw [hr]; push_cast; ring
      have hr0 : (r : ℝ) ≠ 0 := by
        intro e
        apply hM'
        rw [hr', e, mul_zero]
      push_cast
      rw [hr']
      field_simp
      rfl
    simp only [bsum]
    rw [bsum_congr _ _ hform, bsum_congr _ _ hform]
    have hg : ∀ τ', ∀ y ∈ fs, ∀ v,
        (fun τ'' : ℕ → ZMod 2 => ex ((zq (τ'' f) : ℝ) * N τ'' / 2)) (Function.update τ' y v) =
          (fun τ'' : ℕ → ZMod 2 => ex ((zq (τ'' f) : ℝ) * N τ'' / 2)) τ' := by
      intro τ' y hy v
      have hfy : f ≠ y := fun e => hf (e ▸ hy)
      simp only [Function.update_of_ne hfy]
      rw [hNinv τ' y (List.mem_cons_of_mem _ hy) v]
    rw [bsum_mul_left _ _ fs hg, bsum_mul_left _ _ fs hg, ih, ih, Function.update_self,
      Function.update_self, hNinv τ f List.mem_cons_self, hNinv τ f List.mem_cons_self, zq0, zq1]
    have hinv : ∀ v : ZMod 2,
        (bs.map fun c => if fz (Function.update τ f v) c = 0 then (2 : ℂ) else 0) =
          bs.map fun c => if fz τ c = 0 then (2 : ℂ) else 0 := by
      intro v
      apply List.map_congr_left
      intro c hc
      rw [fz_update_not τ f v c (fun m hm => hfree f List.mem_cons_self c
        (List.mem_cons_of_mem _ hc) m hm)]
    rw [hinv 0, hinv 1]
    push_cast
    rw [zero_mul, zero_div, ex_zero]
    simp only [one_mul]
    rw [ex_halfN, hNb, List.map_cons, List.prod_cons]
    split_ifs <;> ring

theorem dyBits_length : ∀ (k : ℕ) (S : Poly) (bs : List F2), dyBits k S = some bs →
    bs.length = k
  | 0, S, bs, h => by
    simp only [dyBits, Option.some.injEq] at h
    subst h
    rfl
  | k + 1, S, bs, h => by
    simp only [dyBits] at h
    split_ifs at h
    obtain ⟨bs', hbs', rfl⟩ := Option.map_eq_some_iff.1 h
    simp [dyBits_length k _ bs' hbs']

/-- The number the digits' values spell, low digit first. -/
def dnum (τ : ℕ → ZMod 2) : List F2 → ℕ
  | [] => 0
  | B :: Bs => (fz τ B).val + 2 * dnum τ Bs

/-- `w 2^i` times the lift of the digit `B_i`, summed over the digits. -/
def tLift (M : ℕ) : ℕ → List F2 → Poly
  | _, [] => []
  | w, B :: Bs => (lift M B).map (fun u => ([] ++ u.1, w * u.2)) ++ tLift M (2 * w) Bs

theorem dnum_lt (τ : ℕ → ZMod 2) : ∀ Bs : List F2, dnum τ Bs < 2 ^ Bs.length
  | [] => by simp [dnum]
  | B :: Bs => by
    have ih := dnum_lt τ Bs
    have hv : (fz τ B).val < 2 := ZMod.val_lt _
    simp only [dnum, List.length_cons, pow_succ]
    omega

theorem dnum_update_not (τ : ℕ → ZMod 2) (x : ℕ) (b : ZMod 2) :
    ∀ Bs : List F2, (∀ B ∈ Bs, ∀ m ∈ B, x ∉ m) → dnum (Function.update τ x b) Bs = dnum τ Bs
  | [] => fun _ => rfl
  | B :: Bs => fun h => by
    simp only [dnum]
    rw [fz_update_not τ x b B (h B List.mem_cons_self),
      dnum_update_not τ x b Bs (fun B' hB => h B' (List.mem_cons_of_mem _ hB))]

theorem zq_val (x : ZMod 2) : zq x = (x.val : ℤ) := by
  rcases z2_cases x with rfl | rfl <;> decide

theorem pn_tLift (τ : ℕ → ZMod 2) (M : ℕ) (hM : 2 ≤ M) :
    ∀ (Bs : List F2) (w : ℕ), peq M (pn τ (tLift M w Bs)) ((w : ℤ) * dnum τ Bs)
  | [], w => ⟨0, by simp [tLift, pn, dnum]⟩
  | B :: Bs, w => by
    obtain ⟨n1, e1⟩ := pn_lift τ M hM B
    obtain ⟨n2, e2⟩ := pn_tLift τ M hM Bs (2 * w)
    simp only [tLift, dnum]
    have hm1 : zq (mz τ []) = 1 := by simp [mz]; decide
    rw [pn_append, pn_mapmul, hm1, e1, e2, zq_val]
    refine ⟨w * n1 + n2, ?_⟩
    push_cast
    ring

theorem peq_symm {M : ℕ} {a b : ℤ} (h : peq M a b) : peq M b a := by
  obtain ⟨n, hn⟩ := h
  exact ⟨-n, by rw [hn]; ring⟩

theorem dnum_eq_zero (τ : ℕ → ZMod 2) :
    ∀ bs : List F2, dnum τ bs = 0 ↔ ∀ b ∈ bs, fz τ b = 0
  | [] => by simp [dnum]
  | B :: Bs => by
    have ih := dnum_eq_zero τ Bs
    simp only [dnum, List.forall_mem_cons]
    rw [← ih]
    have v0 : (0 : ZMod 2).val = 0 := rfl
    have v1 : (1 : ZMod 2).val = 1 := rfl
    rcases z2_cases (fz τ B) with h | h <;> rw [h]
    · rw [v0]
      constructor
      · intro e; exact ⟨rfl, by omega⟩
      · rintro ⟨-, e⟩; omega
    · rw [v1]
      constructor
      · intro e; omega
      · rintro ⟨e, -⟩; exact absurd e (by decide)

/-- The lift of `B` keeping the products of at most `d` of its monomials. -/
def liftT (M : ℕ) : ℕ → F2 → Poly
  | 0, _ => []
  | _ + 1, [] => []
  | d + 1, m :: B =>
    (m, 1) :: (liftT M (d + 1) B ++ (liftT M d B).map fun t => (m ++ t.1, (M - 2) * t.2))

theorem pn_liftT (τ : ℕ → ZMod 2) (M : ℕ) :
    ∀ (d : ℕ) (B : F2), ∃ z : ℤ,
      pn τ (lift M B) = pn τ (liftT M d B) + ((M - 2 : ℕ) : ℤ) ^ d * z
  | 0, B => ⟨pn τ (lift M B), by simp [liftT, pn]⟩
  | d + 1, [] => ⟨0, by simp [liftT, lift, pn]⟩
  | d + 1, m :: B => by
    obtain ⟨z1, h1⟩ := pn_liftT τ M (d + 1) B
    obtain ⟨z2, h2⟩ := pn_liftT τ M d B
    refine ⟨z1 + zq (mz τ m) * z2, ?_⟩
    simp only [lift, liftT]
    rw [pn_cons, pn_cons, pn_append, pn_append, pn_mapmul, pn_mapmul]
    linear_combination (1 + ((M - 2 : ℕ) : ℤ) * zq (mz τ m)) * h1 -
      (((M - 2 : ℕ) : ℤ) * zq (mz τ m)) * (h1.symm.trans h2).symm

/-- `w 2^i` times the truncated lift of the digit `B_i`, keeping products of at most `d - i`
monomials, summed over the digits. -/
def bLift (M : ℕ) : ℕ → ℕ → List F2 → Poly
  | _, _, [] => []
  | w, d, B :: Bs => (liftT M d B).map (fun u => ([] ++ u.1, w * u.2)) ++ bLift M (2 * w) (d - 1) Bs

theorem pn_bLift (τ : ℕ → ZMod 2) (k : ℕ) (hk : 1 ≤ k) :
    ∀ (bs : List F2) (w d : ℕ), w * 2 ^ d = 2 ^ k → bs.length ≤ d →
      peq (2 ^ k) (pn τ (bLift (2 ^ k) w d bs)) ((w : ℤ) * dnum τ bs)
  | [], w, d, _, _ => ⟨0, by simp [bLift, pn, dnum]⟩
  | B :: Bs, w, d, hw, hl => by
    have hd : 1 ≤ d := by simp at hl; omega
    have h2 : 2 ≤ 2 ^ k := by
      calc 2 = 2 ^ 1 := by norm_num
        _ ≤ 2 ^ k := Nat.pow_le_pow_right (by norm_num) hk
    have hw' : 2 * w * 2 ^ (d - 1) = 2 ^ k := by
      rw [← hw]
      obtain ⟨e, rfl⟩ : ∃ e, d = e + 1 := ⟨d - 1, by omega⟩
      simp only [Nat.add_sub_cancel, pow_succ]
      ring
    obtain ⟨n2, e2⟩ := pn_bLift τ k hk Bs (2 * w) (d - 1) hw' (by simp at hl; omega)
    obtain ⟨z, ez⟩ := pn_liftT τ (2 ^ k) d B
    obtain ⟨n1, e1⟩ := pn_lift τ (2 ^ k) h2 B
    obtain ⟨j, hj⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
    have hc : ((2 ^ k - 2 : ℕ) : ℤ) = 2 * ((2 ^ j - 1 : ℕ) : ℤ) := by
      have h1 : 1 ≤ 2 ^ j := Nat.one_le_two_pow
      rw [hj, pow_succ]
      push_cast [Nat.cast_sub (show 2 ≤ 2 ^ j * 2 by omega), Nat.cast_sub h1]
      ring
    have hwz : ((w : ℕ) : ℤ) * 2 ^ d = ((2 ^ k : ℕ) : ℤ) := by exact_mod_cast hw
    simp only [bLift, dnum]
    have hm1 : zq (mz τ []) = 1 := by simp [mz]; decide
    rw [pn_append, pn_mapmul, hm1, e2]
    refine ⟨w * n1 - ((2 ^ j - 1 : ℕ) : ℤ) ^ d * z + n2, ?_⟩
    have hL : pn τ (liftT (2 ^ k) d B) =
        ((fz τ B).val : ℤ) + n1 * ((2 ^ k : ℕ) : ℤ) - ((2 ^ k - 2 : ℕ) : ℤ) ^ d * z := by
      rw [← zq_val]; linarith
    rw [hL, hc, mul_pow]
    generalize ((2 ^ j - 1 : ℕ) : ℤ) = o
    simp only [Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat]
    linear_combination (-(o ^ d * z)) * hwz

/-- Whether `bs` are `k ≥ 1` binary digits of `S` modulo `2^k`, low digit first: `S` and
`Sum_j 2^j bs_j` (bLift) agree after normalising modulo `2^k`. -/
def bitsOK (k : ℕ) (S : Poly) (bs : List F2) : Bool :=
  decide (1 ≤ k) && decide (bs.length = k) &&
    decide (normP (2 ^ k) S = normP (2 ^ k) (bLift (2 ^ k) 1 k bs))

theorem bits_iff (τ : ℕ → ZMod 2) (k : ℕ) (S : Poly) (bs : List F2)
    (h : bitsOK k S bs = true) : (((2 ^ k : ℕ) : ℤ) ∣ pn τ S ↔ ∀ b ∈ bs, fz τ b = 0) := by
  simp only [bitsOK, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hk, hl⟩, hn⟩ := h
  have e : peq (2 ^ k) (pn τ S) ((dnum τ bs : ℕ) : ℤ) := by
    have h1 := pn_bLift τ k hk bs 1 k (by ring) (by omega)
    refine peq_trans (peq_symm (pn_normP τ _ S)) ?_
    rw [hn]
    refine peq_trans (pn_normP τ _ _) ?_
    simpa using h1
  obtain ⟨n, hn'⟩ := e
  have hlt := dnum_lt τ bs
  rw [hl] at hlt
  rw [← dnum_eq_zero τ bs]
  have hK : (0 : ℤ) < ((2 ^ k : ℕ) : ℤ) := by positivity
  have hD : ((dnum τ bs : ℕ) : ℤ) < ((2 ^ k : ℕ) : ℤ) := by exact_mod_cast hlt
  have hD0 : (0 : ℤ) ≤ ((dnum τ bs : ℕ) : ℤ) := by positivity
  constructor
  · rintro ⟨q, hq⟩
    have hx : ((dnum τ bs : ℕ) : ℤ) = ((2 ^ k : ℕ) : ℤ) * (q - n) := by
      push_cast at hn' hq ⊢
      linarith
    rcases lt_trichotomy (q - n) 0 with hc | hc | hc
    · nlinarith
    · rw [hc, mul_zero] at hx; exact_mod_cast hx
    · nlinarith
  · intro h0
    refine ⟨n, ?_⟩
    rw [hn', h0]
    push_cast
    ring

/-- The side conditions of [Dyadic] with the binary digits `bs` of `S`. -/
def dyRest (M : ℕ) (ds : List ℕ) (S : Poly) (fs : List ℕ) (bs : List F2) (s : Data) : Bool :=
  decide (0 < M) && decide (M % (2 ^ ds.length) = 0) && decide (M % 2 = 0) &&
    decide (s.path.take ds.length = ds) && decide s.path.Nodup &&
    decide ((fs ++ s.path.drop ds.length).Nodup) && decide (fs.length = ds.length) &&
    ds.all (fun d => !decide (d ∈ fvars (s.cons ++ s.rows))) &&
    S.all (fun t => !hasAny ds t.1) &&
    fs.all (fun f => !decide (f ∈ pvars s.phase) && !decide (f ∈ fvars (s.cons ++ s.rows)) &&
      !decide (f ∈ fvars bs)) &&
    decide (normP M (s.phase.filter fun t => hasAny ds t.1) =
      normP M (dyTerms M (2 ^ ds.length) S ds 1))

/-- Whether the head `k = ds.length` path variables are digits `ds` with phase `Y S / 2^k`
(Y = Sum_a 2^a d_a, `S` free of them, `2^k ∣ M`), in no constraint or row, the fresh variables
`fs` in no phase term, constraint or row of `S`'s digits, and `S`'s binary digits exist. -/
def dyadicOK (M : ℕ) (ds : List ℕ) (S : Poly) (fs : List ℕ) (s : Data) : Bool :=
  match dyBits ds.length S with
  | none => false
  | some bs => dyRest M ds S fs bs s

/-- The digits `ds` summed: their terms replaced by `f_j b_j / 2` over the fresh `fs`. -/
def dyadicS (M : ℕ) (ds : List ℕ) (S : Poly) (fs : List ℕ) (s : Data) : Data :=
  match dyBits ds.length S with
  | none => s
  | some bs =>
    { path := fs ++ s.path.drop ds.length,
      phase := s.phase.filter (fun t => !hasAny ds t.1) ++ dyNew M fs bs,
      cons := s.cons, rows := s.rows }

theorem fvars_mem {q : F2} {C : List F2} {m : Mono} {f : ℕ} (hq : q ∈ C) (hm : m ∈ q)
    (hf : f ∈ m) : f ∈ fvars C :=
  List.mem_flatMap.2 ⟨q, hq, List.mem_flatMap.2 ⟨m, hm, hf⟩⟩

/-- [Dyadic] with binary digits `bs` of `S` given by their divisibility fact. -/
theorem dyadic_core (M : ℕ) (c : ℂ) (ds : List ℕ) (S : Poly) (fs : List ℕ) (bs : List F2)
    (s : Data) (σ : ℕ → ZMod 2) (hbl : bs.length = ds.length)
    (hbits : ∀ τ : ℕ → ZMod 2, (((2 ^ ds.length : ℕ) : ℤ) ∣ pn τ S ↔ ∀ b ∈ bs, fz τ b = 0))
    (h : dyRest M ds S fs bs s = true) :
    val M c s σ = val M c ⟨fs ++ s.path.drop ds.length,
      s.phase.filter (fun t => !hasAny ds t.1) ++ dyNew M fs bs, s.cons, s.rows⟩ σ := by
  unfold dyRest at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, Bool.not_eq_true',
    decide_eq_false_iff_not] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hM, hKM⟩, hM2⟩, htake⟩, hnd⟩, hnd2⟩, hfl⟩, hdc⟩, hSd⟩, hfs⟩, hP⟩ := h
  set k := ds.length with hk
  set K := 2 ^ k with hK
  set ys := s.path.drop k with hys
  have hpath : s.path = ds ++ ys := by rw [← htake, List.take_append_drop]
  have hK0 : 0 < K := by positivity
  obtain ⟨q, hq⟩ : ∃ q, M = K * q := ⟨M / K, (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hKM)).symm⟩
  obtain ⟨r, hr⟩ : ∃ r, M = 2 * r := ⟨M / 2, by omega⟩
  have hdsnd : ds.Nodup := by rw [hpath] at hnd; exact (List.nodup_append.1 hnd).1
  have hfsnd : fs.Nodup := (List.nodup_append.1 hnd2).1
  have hSfree : ∀ t ∈ S, ∀ d ∈ ds, d ∉ t.1 := fun t ht => not_hasAny (by simpa using hSd t ht)
  set R := s.phase.filter (fun t => !hasAny ds t.1) with hR
  set Pd := s.phase.filter (fun t => hasAny ds t.1) with hPd
  have hRfree : ∀ t ∈ R, ∀ d ∈ ds, d ∉ t.1 := by
    intro t ht
    simp only [hR, List.mem_filter, Bool.not_eq_true'] at ht
    exact not_hasAny ht.2
  -- the summand as the digits' factor times the rest
  set G : (ℕ → ZMod 2) → ℂ := fun τ => if ∀ q ∈ s.cons ++ s.rows, fz τ q = 0 then
    c * ex ((pn τ R : ℝ) / M) else 0 with hG
  have hsplitL : ∀ τ, summand M c s τ = G τ * ex ((pn τ Pd : ℝ) / M) := by
    intro τ
    unfold summand
    simp only [hG]
    rw [pn_filter_split τ (fun t => hasAny ds t.1) s.phase]
    split_ifs
    · push_cast; rw [add_div, ex_add]; ring
    · ring
  have hsplitR : ∀ τ, summand M c ⟨fs ++ ys, R ++ dyNew M fs bs, s.cons, s.rows⟩ τ =
      G τ * ex ((pn τ (dyNew M fs bs) : ℝ) / M) := by
    intro τ
    unfold summand
    simp only [hG, pn_append]
    split_ifs
    · push_cast; rw [add_div, ex_add]; ring
    · ring
  have hGinv : ∀ (vs : List ℕ), (∀ v ∈ vs, v ∉ fvars (s.cons ++ s.rows)) →
      (∀ t ∈ R, ∀ v ∈ vs, v ∉ t.1) → ∀ τ, ∀ y ∈ vs, ∀ b, G (Function.update τ y b) = G τ := by
    intro vs hc hr τ y hy b
    simp only [hG]
    rw [pn_update_not τ y b R (fun t ht => hr t ht y hy)]
    simp only [holds_update_not τ y b _ (hc y hy)]
  have hGds := hGinv ds hdc hRfree
  have hRfs : ∀ t ∈ R, ∀ f ∈ fs, f ∉ t.1 := by
    intro t ht f hf hft
    have hfp := (hfs f hf).1.1
    apply hfp
    simp only [hR, List.mem_filter] at ht
    exact List.mem_flatMap.2 ⟨t, ht.1, hft⟩
  have hGfs := hGinv fs (fun f hf => (hfs f hf).1.2) hRfs
  have hpeq : ∀ τ, peq M (pn τ Pd) (pn τ (dyTerms M K S ds 1)) := by
    intro τ
    have h1 := pn_normP τ M Pd
    have h2 := pn_normP τ M (dyTerms M K S ds 1)
    rw [hP] at h1
    obtain ⟨a1, ha1⟩ := h1
    obtain ⟨b1, hb1⟩ := h2
    exact ⟨b1 - a1, by linarith⟩
  unfold val
  dsimp only
  rw [hpath, bsum_perm _ List.perm_append_comm σ, bsum_append,
    bsum_perm _ List.perm_append_comm σ, bsum_append]
  apply bsum_congr
  intro τ
  rw [bsum_congr _ _ hsplitL, bsum_mul_left _ _ ds hGds, bsum_congr _ _ hsplitR,
    bsum_mul_left _ _ fs hGfs]
  congr 1
  have hL : bsum ds (fun τ' => ex ((pn τ' Pd : ℝ) / M)) τ =
      bsum ds (fun τ' => ex ((pn τ' (dyTerms M K S ds 1) : ℝ) / M)) τ :=
    bsum_congr _ _ (fun τ' => ex_peq hM (hpeq τ')) ds τ
  rw [hL, bsum_dy M K q hM hK0 hq S ds 1 hdsnd hSfree τ, dyProd_geom,
    bsum_new M r hM hr fs bs hfsnd (by rw [hfl, hbl]) (fun f hf b hb m hm hfm =>
      (hfs f hf).2 (fvars_mem hb hm hfm)) τ, prod_ite, hbl]
  have h3 := hbits τ
  rw [hK] at h3
  simp only [h3]


theorem dyadic_sound (M : ℕ) (c : ℂ) (ds : List ℕ) (S : Poly) (fs : List ℕ) (s : Data)
    (σ : ℕ → ZMod 2) (h : dyadicOK M ds S fs s = true) :
    val M c s σ = val M c (dyadicS M ds S fs s) σ := by
  unfold dyadicOK at h
  unfold dyadicS
  rcases hb : dyBits ds.length S with _ | bs
  · simp [hb] at h
  simp only [hb] at h
  exact dyadic_core M c ds S fs bs s σ (dyBits_length _ S bs hb) (fun τ => dy_bits τ _ S bs hb) h

/-- Whether `bs` are the binary digits of `S` (bitsOK) and [Dyadic]'s side conditions hold. -/
def dyadicBOK (M : ℕ) (ds : List ℕ) (S : Poly) (fs : List ℕ) (bs : List F2) (s : Data) : Bool :=
  bitsOK ds.length S bs && dyRest M ds S fs bs s

/-- The digits `ds` summed with the given binary digits `bs`. -/
def dyadicBS (M : ℕ) (ds : List ℕ) (fs : List ℕ) (bs : List F2) (s : Data) : Data :=
  { path := fs ++ s.path.drop ds.length,
    phase := s.phase.filter (fun t => !hasAny ds t.1) ++ dyNew M fs bs,
    cons := s.cons, rows := s.rows }

theorem dyadicB_sound (M : ℕ) (c : ℂ) (ds : List ℕ) (S : Poly) (fs : List ℕ) (bs : List F2)
    (s : Data) (σ : ℕ → ZMod 2) (h : dyadicBOK M ds S fs bs s = true) :
    val M c s σ = val M c (dyadicBS M ds fs bs s) σ := by
  simp only [dyadicBOK, Bool.and_eq_true] at h
  have hl : bs.length = ds.length := by
    have := h.1
    simp only [bitsOK, Bool.and_eq_true, decide_eq_true_eq] at this
    exact this.1.2
  exact dyadic_core M c ds S fs bs s σ hl (fun τ => bits_iff τ _ S bs h.1) h.2


/-! ## Phase terms off the constraints -/

/-- Whether `m (1 + q) = 0` for some constraint `q`. -/
def dropT (C : List F2) (m : Mono) : Bool :=
  C.any fun q => decide (normF (f2mul [m] ([] :: q)) = [])

/-- The phase terms whose monomial is 0 where some constraint holds dropped. -/
def pdropS (s : Data) : Data := { s with phase := s.phase.filter fun t => !dropT s.cons t.1 }

theorem pdrop_sound (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) :
    val M c s σ = val M c (pdropS s) σ := by
  unfold val pdropS
  apply bsum_congr
  intro τ
  unfold summand
  dsimp only
  split_ifs with hC
  · rw [pn_filter_zero τ _ s.phase]
    intro t _ ht
    simp only [Bool.not_eq_false', dropT, List.any_eq_true, decide_eq_true_eq] at ht
    obtain ⟨q, hq, e⟩ := ht
    have h0 : fz τ q = 0 := hC q (List.mem_append_left _ hq)
    have e2 : fz τ [t.1] * (mz τ [] + fz τ q) = 0 := by
      rw [← fz_cons, ← fz_f2mul, ← fz_normF, e]
      rfl
    rw [h0, show mz τ [] = 1 by simp [mz], add_zero, mul_one, fz_cons,
      show fz τ ([] : F2) = 0 from rfl, add_zero] at e2
    exact e2
  · rfl

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
  | pin (B : F2)
  | norm
  | dyadic (ds : List ℕ) (S : Poly) (fs : List ℕ)
  | pdrop
  | dyadicB (ds : List ℕ) (S : Poly) (fs : List ℕ) (bs : List F2)

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
  | .pin B, s => pinOK B s
  | .norm, _ => true
  | .dyadic ds S fs, s => dyadicOK M ds S fs s
  | .pdrop, _ => true
  | .dyadicB ds S fs bs, s => dyadicBOK M ds S fs bs s

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
  | .pin B, s => pinS M B s
  | .norm, s => normS M s
  | .dyadic ds S fs, s => dyadicS M ds S fs s
  | .pdrop, s => pdropS s
  | .dyadicB ds _ fs bs, s => dyadicBS M ds fs bs s

/-- The factor a step multiplies the constant by. -/
noncomputable def fac : Step → ℂ
  | .elim | .hhCons | .hhPivot | .hhNl => 2
  | .omega => rt 2 * om 8
  | .omegaConj => rt 2 * om 8 ^ 7
  | .perm _ | .canon _ | .gather | .bcanon _ | .hhGen | .rem | .pin _ | .norm | .dyadic _ _ _
    | .pdrop | .dyadicB _ _ _ _ => 1

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
  | pin B => simpa [fac, stepS] using pin_sound M hM c B s σ h
  | norm => simpa [fac, stepS] using norm_sound M (by omega) c s σ
  | dyadic ds S fs => simpa [fac, stepS] using dyadic_sound M c ds S fs s σ h
  | pdrop => simpa [fac, stepS] using pdrop_sound M c s σ
  | dyadicB ds S fs bs => simpa [fac, stepS] using dyadicB_sound M c ds S fs bs s σ h

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
