/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumRules

/-!
# Path sums with amplitude atoms

`AData` is a path sum `Data` with amplitude atoms: the atom `(i, f)` is `A i` where the F_2
polynomial `f` is 1 and `1` where it is 0. The path variables the atoms read are `held`,
summed outside the inner sum: `valA M c A t σ` is the sum over `held` of the atoms' product
`aval` times `val M c t.s`. `astep_sound` and `arun_sound` carry every `PSem` step list to the
inner sum, `release_sound` moves a held variable no atom reads into the inner path, `adrop_sound`
drops atoms on the zero polynomial, `aconst_sound` moves an atom on the polynomial 1 into the
constant, `average_sound` is [Average] and `afinal_sound` compares two reduced sums with the
same atoms. `valA_flat` reads `valA` as one sum over `held ++ t.s.path`.
-/

namespace ArchytasZX
namespace PSem

open Complex

/-- Atoms: a coefficient index and an F_2 polynomial. -/
abbrev Atoms := List (ℕ × F2)

/-- A path sum, its held variables and its atoms. -/
structure AData where
  s : Data
  held : List ℕ
  atoms : Atoms

/-- The product of the atoms at one assignment. -/
noncomputable def aval (A : ℕ → ℂ) (as : Atoms) (τ : ℕ → ZMod 2) : ℂ :=
  (as.map fun a => if fz τ a.2 = 0 then 1 else A a.1).prod

/-- The value of a path sum with atoms at a boundary assignment. -/
noncomputable def valA (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (t : AData) (σ : ℕ → ZMod 2) : ℂ :=
  bsum t.held (fun τ => aval A t.atoms τ * val M c t.s τ) σ

theorem aval_nil (A : ℕ → ℂ) (τ : ℕ → ZMod 2) : aval A [] τ = 1 := rfl

theorem aval_cons (A : ℕ → ℂ) (a : ℕ × F2) (as : Atoms) (τ : ℕ → ZMod 2) :
    aval A (a :: as) τ = (if fz τ a.2 = 0 then 1 else A a.1) * aval A as τ := by
  simp [aval]

theorem aval_append (A : ℕ → ℂ) (as bs : Atoms) (τ : ℕ → ZMod 2) :
    aval A (as ++ bs) τ = aval A as τ * aval A bs τ := by
  simp [aval, List.map_append, List.prod_append]

/-- The variables the atoms read. -/
def avars (as : Atoms) : List ℕ := as.flatMap fun a => a.2.flatMap id

theorem aval_update_not (A : ℕ → ℂ) (τ : ℕ → ZMod 2) (y : ℕ) (b : ZMod 2) :
    ∀ as : Atoms, y ∉ avars as → aval A as (Function.update τ y b) = aval A as τ
  | [], _ => rfl
  | a :: as, h => by
    have h1 : ∀ m ∈ a.2, y ∉ m := fun m hm hy =>
      h (List.mem_flatMap.2 ⟨a, List.mem_cons_self, List.mem_flatMap.2 ⟨m, hm, hy⟩⟩)
    have h2 : y ∉ avars as := fun hy => h (by
      simp only [avars, List.flatMap_cons, List.mem_append] at hy ⊢; exact Or.inr hy)
    rw [aval_cons, aval_cons, fz_update_not τ y b a.2 h1, aval_update_not A τ y b as h2]

/-- `val` unchanged by a variable off the path, the phase, the constraints and the rows. -/
theorem val_update_not (M : ℕ) (c : ℂ) (s : Data) (τ : ℕ → ZMod 2) (y : ℕ) (b : ZMod 2)
    (hy : y ∉ s.path) (hp : y ∉ pvars s.phase) (hc : y ∉ fvars (s.cons ++ s.rows)) :
    val M c s (Function.update τ y b) = val M c s τ := by
  unfold val
  rw [bsum_out _ y b s.path τ hy]
  exact bsum_congr _ _ (fun τ' => summand_update_not M c s τ' y b hp hc) _ _

/-! ## The `PSem` steps on the inner sum -/

theorem astep_sound (M : ℕ) (hM : 2 ≤ M) (st : Step) (t : AData) (c : ℂ) (A : ℕ → ℂ)
    (σ : ℕ → ZMod 2) (h : stepOK M st t.s = true) :
    valA M c A t σ = valA M (fac st * c) A { t with s := stepS M st t.s } σ := by
  unfold valA
  exact bsum_congr _ _ (fun τ => by rw [step_sound M hM st t.s c τ h]) _ _

theorem arun_sound (M : ℕ) (hM : 2 ≤ M) (sts : List Step) (t : AData) (c : ℂ) (A : ℕ → ℂ)
    (σ : ℕ → ZMod 2) (h : runOK M sts t.s = true) :
    valA M c A t σ = valA M (facs sts * c) A { t with s := run M sts t.s } σ := by
  unfold valA
  exact bsum_congr _ _ (fun τ => by rw [run_sound M hM sts t.s c τ h]) _ _

/-! ## Releasing a held variable -/

/-- Whether the head held variable is read by no atom and is in neither the rest of `held`
nor the path. -/
def releaseOK (t : AData) : Bool :=
  match t.held with
  | [] => false
  | y :: hs => !decide (y ∈ hs) && !decide (y ∈ t.s.path) && !decide (y ∈ avars t.atoms)

/-- The head held variable moved to the front of the path. -/
def releaseS (t : AData) : AData :=
  match t.held with
  | [] => t
  | y :: hs => ⟨{ t.s with path := y :: t.s.path }, hs, t.atoms⟩

theorem release_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (t : AData) (σ : ℕ → ZMod 2)
    (h : releaseOK t = true) : valA M c A t σ = valA M c A (releaseS t) σ := by
  rcases t with ⟨s, held, as⟩
  rcases held with _ | ⟨y, hs⟩
  · simp [releaseOK] at h
  simp only [releaseOK, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at h
  obtain ⟨⟨h1, -⟩, h3⟩ := h
  simp only [valA, releaseS]
  rw [bsum_head _ y hs σ h1]
  apply bsum_congr
  intro τ
  rw [aval_update_not A τ y 0 as h3, aval_update_not A τ y 1 as h3, ← mul_add]
  rfl

/-! ## Atoms on constant polynomials -/

/-- The atoms on a nonzero polynomial kept. -/
def adropS (t : AData) : AData := { t with atoms := t.atoms.filter fun a => !decide (a.2 = []) }

theorem aval_drop (A : ℕ → ℂ) (τ : ℕ → ZMod 2) :
    ∀ as : Atoms, aval A (as.filter fun a => !decide (a.2 = [])) τ = aval A as τ
  | [] => rfl
  | a :: as => by
    by_cases ha : a.2 = []
    · simp only [List.filter_cons, ha, decide_true, Bool.not_true, Bool.false_eq_true,
        ↓reduceIte]
      rw [aval_drop A τ as, aval_cons, ha]
      simp [fz]
    · simp only [List.filter_cons, ha, decide_false, Bool.not_false, ↓reduceIte]
      rw [aval_cons, aval_cons, aval_drop A τ as]

theorem adrop_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (t : AData) (σ : ℕ → ZMod 2) :
    valA M c A t σ = valA M c A (adropS t) σ := by
  unfold valA adropS
  exact bsum_congr _ _ (fun τ => by rw [aval_drop]) _ _

/-- The head atom, on the polynomial 1, dropped. -/
def aconstS (t : AData) : AData := { t with atoms := t.atoms.tail }

/-- Whether the head atom is on the polynomial 1. -/
def aconstOK (t : AData) : Bool :=
  match t.atoms with
  | [] => false
  | a :: _ => decide (a.2 = [[]])

/-- The coefficient of the head atom. -/
def aconstIdx (t : AData) : ℕ := (t.atoms.headD (0, [])).1

theorem aconst_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (t : AData) (σ : ℕ → ZMod 2)
    (h : aconstOK t = true) :
    valA M c A t σ = valA M (A (aconstIdx t) * c) A (aconstS t) σ := by
  rcases t with ⟨s, held, as⟩
  rcases as with _ | ⟨a, as⟩
  · simp [aconstOK] at h
  simp only [aconstOK, decide_eq_true_eq] at h
  simp only [valA, aconstS, aconstIdx, List.tail_cons, List.headD_cons]
  apply bsum_congr
  intro τ
  rw [aval_cons, h, val_smul]
  have : fz τ [[]] = 1 := by simp [fz, mz]
  rw [this]
  simp only [one_ne_zero, ↓reduceIte]
  ring

/-! ## [Average] -/

/-- Whether atom `a` reads `y`. -/
def reads (y : ℕ) (a : ℕ × F2) : Bool := a.2.any fun m => decide (y ∈ m)

/-- `(1 + y) g`. -/
def oneY (y : ℕ) (g : F2) : F2 := f2mul [[], [y]] g

/-- `y g`. -/
def onlyY (y : ℕ) (g : F2) : F2 := f2mul [[y]] g

/-- Whether `y` is the head held variable, absent from the rest of `held`, the path, the
phase, the constraints, the rows and `g`, and every atom reading it is on `(1 + y) g` or
`y g`. -/
def averageOK (y : ℕ) (g : F2) (t : AData) : Bool :=
  decide (t.held.head? = some y) && !decide (y ∈ t.held.tail) && !decide (y ∈ t.s.path) &&
    !decide (y ∈ pvars t.s.phase) && !decide (y ∈ fvars (t.s.cons ++ t.s.rows)) &&
    !decide (y ∈ g.flatMap id) &&
    (t.atoms.filter (reads y)).all fun a => decide (a.2 = oneY y g) || decide (a.2 = onlyY y g)

/-- The product of the coefficients of the atoms on `(1 + y) g`. -/
noncomputable def alphaY (A : ℕ → ℂ) (y : ℕ) (g : F2) (as : Atoms) : ℂ :=
  ((as.filter (reads y)).filter (fun a => decide (a.2 = oneY y g))).map (A ·.1) |>.prod

/-- The product of the coefficients of the other atoms reading `y`. -/
noncomputable def betaY (A : ℕ → ℂ) (y : ℕ) (g : F2) (as : Atoms) : ℂ :=
  ((as.filter (reads y)).filter (fun a => !decide (a.2 = oneY y g))).map (A ·.1) |>.prod

/-- The held variable `y` summed: the atoms reading it replaced by the atom `(j, g)`. -/
def averageS (y : ℕ) (g : F2) (j : ℕ) (t : AData) : AData :=
  ⟨t.s, t.held.tail, t.atoms.filter (fun a => !reads y a) ++ [(j, g)]⟩

theorem z2_one_add_one : (1 + 1 : ZMod 2) = 0 := by decide

theorem fz_oneY (τ : ℕ → ZMod 2) (y : ℕ) (g : F2) : fz τ (oneY y g) = (1 + τ y) * fz τ g := by
  rw [oneY, fz_f2mul]
  simp [fz, mz]

theorem fz_onlyY (τ : ℕ → ZMod 2) (y : ℕ) (g : F2) : fz τ (onlyY y g) = τ y * fz τ g := by
  rw [onlyY, fz_f2mul]
  simp [fz, mz]

theorem aval_split (A : ℕ → ℂ) (y : ℕ) (τ : ℕ → ZMod 2) :
    ∀ as : Atoms, aval A as τ =
      aval A (as.filter (reads y)) τ * aval A (as.filter fun a => !reads y a) τ
  | [] => by simp [aval]
  | a :: as => by
    have ih := aval_split A y τ as
    by_cases ha : reads y a = true
    · simp only [List.filter_cons, ha, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
      rw [aval_cons, aval_cons, ih]
      ring
    · simp only [List.filter_cons, ha, Bool.not_false, Bool.false_eq_true, ↓reduceIte]
      rw [aval_cons, aval_cons, ih]
      ring

theorem avars_not_reads (y : ℕ) :
    ∀ as : Atoms, y ∉ avars (as.filter fun a => !reads y a)
  | [] => by simp [avars]
  | a :: as => by
    have ih := avars_not_reads y as
    by_cases ha : reads y a = true
    · simpa [List.filter_cons, ha] using ih
    · have hn : ∀ m ∈ a.2, y ∉ m := by
        intro m hm hy
        apply ha
        simp only [reads, List.any_eq_true, decide_eq_true_eq]
        exact ⟨m, hm, hy⟩
      simp only [List.filter_cons, ha, Bool.not_false, ↓reduceIte, avars, List.flatMap_cons,
        List.mem_append, not_or]
      refine ⟨fun hy => ?_, ih⟩
      obtain ⟨m, hm, hy⟩ := List.mem_flatMap.1 hy
      exact hn m hm hy

/-- The atoms reading `y` at `y = b`: `1` where `g` is 0, else the coefficients of the atoms
on `(1 + y) g` at `b = 0` and of the others at `b = 1`. -/
theorem aval_reads (A : ℕ → ℂ) (y : ℕ) (g : F2) (τ : ℕ → ZMod 2) (hg : ∀ m ∈ g, y ∉ m) :
    ∀ as : Atoms, (as.filter (reads y)).all
        (fun a => decide (a.2 = oneY y g) || decide (a.2 = onlyY y g)) = true →
      aval A (as.filter (reads y)) (Function.update τ y 0) =
          (if fz τ g = 0 then 1 else alphaY A y g as) ∧
        aval A (as.filter (reads y)) (Function.update τ y 1) =
          (if fz τ g = 0 then 1 else betaY A y g as)
  | [], _ => by simp [aval, alphaY, betaY]
  | a :: as, h => by
    have hgu : ∀ b : ZMod 2, fz (Function.update τ y b) g = fz τ g :=
      fun b => fz_update_not τ y b g hg
    by_cases ha : reads y a = true
    · simp only [List.filter_cons, ha, ↓reduceIte, List.all_cons, Bool.and_eq_true,
        Bool.or_eq_true, decide_eq_true_eq] at h
      obtain ⟨hform, hrest⟩ := h
      obtain ⟨ih0, ih1⟩ := aval_reads A y g τ hg as (by simpa [List.all_eq_true] using hrest)
      simp only [alphaY, betaY, List.filter_cons, ha, ↓reduceIte] at ih0 ih1 ⊢
      rcases hform with hf | hf
      · simp only [hf, decide_true, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
          List.map_cons, List.prod_cons]
        rw [aval_cons, aval_cons, ih0, ih1, hf, fz_oneY, fz_oneY, hgu, hgu]
        simp only [Function.update_self]
        rcases z2_cases (fz τ g) with e | e <;> simp [e, z2_one_add_one]
      · have hne : ¬ a.2 = oneY y g := by
          intro e
          have hl := congrArg List.length (hf.symm.trans e)
          simp only [oneY, onlyY, f2mul, List.flatMap_cons, List.flatMap_nil, List.length_append,
            List.length_map, List.append_nil] at hl
          have hg0 : g = [] := List.length_eq_zero_iff.1 (by omega)
          unfold reads at ha
          rw [hf, hg0] at ha
          simp [onlyY, f2mul] at ha
        simp only [hne, decide_false, Bool.not_false, ↓reduceIte, List.map_cons, List.prod_cons]
        rw [aval_cons, aval_cons, ih0, ih1, hf, fz_onlyY, fz_onlyY, hgu, hgu]
        simp only [Function.update_self]
        rcases z2_cases (fz τ g) with e | e <;> simp [e]
    · simp only [List.filter_cons, ha, Bool.false_eq_true, ↓reduceIte] at h ⊢
      have := aval_reads A y g τ hg as h
      simpa [alphaY, betaY, List.filter_cons, ha] using this

theorem average_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (y : ℕ) (g : F2) (j : ℕ) (t : AData)
    (σ : ℕ → ZMod 2) (h : averageOK y g t = true)
    (hA : 2 * A j = alphaY A y g t.atoms + betaY A y g t.atoms) :
    valA M c A t σ = valA M (2 * c) A (averageS y g j t) σ := by
  rcases t with ⟨s, held, as⟩
  simp only [averageOK, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨hhd, htl⟩, hpath⟩, hph⟩, hcons⟩, hgy⟩, hall⟩ := h
  rcases held with _ | ⟨z, hs⟩
  · simp at hhd
  simp only [List.head?_cons, Option.some.injEq] at hhd
  subst hhd
  simp only [List.tail_cons] at htl
  have hg : ∀ m ∈ g, z ∉ m := fun m hm hz => hgy (List.mem_flatMap.2 ⟨m, hm, hz⟩)
  simp only [valA, averageS, List.tail_cons]
  rw [bsum_head _ z hs σ htl]
  apply bsum_congr
  intro τ
  obtain ⟨h0, h1⟩ := aval_reads A z g τ hg as hall
  have hrest : ∀ b : ZMod 2, aval A (as.filter fun a => !reads z a) (Function.update τ z b) =
      aval A (as.filter fun a => !reads z a) τ :=
    fun b => aval_update_not A τ z b _ (avars_not_reads z as)
  rw [aval_split A z (Function.update τ z 0) as, aval_split A z (Function.update τ z 1) as,
    h0, h1, hrest, hrest, val_update_not M c s τ z 0 hpath hph hcons,
    val_update_not M c s τ z 1 hpath hph hcons, aval_append, aval_cons, aval_nil, val_smul]
  rcases z2_cases (fz τ g) with e | e
  · simp only [e, ↓reduceIte]
    ring
  · simp only [e, one_ne_zero, ↓reduceIte]
    linear_combination (aval A (List.filter (fun a => !reads z a) as) τ * val M c s τ) * hA.symm

/-! ## Reduced sums -/

theorem afinal_sound (M : ℕ) (hM : 0 < M) (cL cR : ℂ) (A : ℕ → ℂ) (tL tR : AData)
    (σ : ℕ → ZMod 2) (kL kR : ℕ) (hh : tL.held = [] ∧ tR.held = [])
    (ha : tL.atoms = tR.atoms) (h : finalOK tL.s tR.s = true) (hkL : constK tL.s.phase = kL)
    (hkR : constK tR.s.phase = kR) (hc : cL * om M ^ kL = cR * om M ^ kR) :
    valA M cL A tL σ = valA M cR A tR σ := by
  simp only [valA, hh.1, hh.2, bsum, ha]
  rw [final_sound M hM cL cR tL.s tR.s σ kL kR h hkL hkR hc]

/-! ## One sum -/

theorem valA_flat (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (t : AData) (σ : ℕ → ZMod 2)
    (h : ∀ y ∈ t.s.path, y ∉ avars t.atoms) :
    valA M c A t σ = bsum (t.held ++ t.s.path) (fun τ => aval A t.atoms τ * summand M c t.s τ) σ := by
  unfold valA val
  rw [bsum_append]
  apply bsum_congr
  intro τ
  rw [bsum_mul_left _ _ t.s.path (fun τ' y hy b => aval_update_not A τ' y b t.atoms (h y hy))]

end PSem
end ArchytasZX
