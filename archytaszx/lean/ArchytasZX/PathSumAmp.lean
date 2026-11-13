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
constant, `average_sound` is [Average], `averageH_sound` is [Average] through a half phase,
`quarter_sound` sums a path variable with quarter coefficients into the atom `((1 + i) / 2)^B0`,
`acomp_sound` moves two atoms of one index on complementary polynomials into the constant and
`afinal_sound` compares two reduced sums with the
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

/-- Whether an atom's polynomial normalises to that of `(1 + y) g`. -/
def isAlpha (y : ℕ) (g : F2) (a : ℕ × F2) : Bool := decide (normF a.2 = normF (oneY y g))

/-- Whether `y` is the head held variable, absent from the rest of `held`, the path, the
phase, the constraints, the rows and `g`, and every atom reading it normalises to `(1 + y) g`
or `y g`. -/
def averageOK (y : ℕ) (g : F2) (t : AData) : Bool :=
  decide (t.held.head? = some y) && !decide (y ∈ t.held.tail) && !decide (y ∈ t.s.path) &&
    !decide (y ∈ pvars t.s.phase) && !decide (y ∈ fvars (t.s.cons ++ t.s.rows)) &&
    !decide (y ∈ g.flatMap id) &&
    (t.atoms.filter (reads y)).all fun a =>
      isAlpha y g a || decide (normF a.2 = normF (onlyY y g))

/-- The product of the coefficients of the atoms on `(1 + y) g`. -/
noncomputable def alphaY (A : ℕ → ℂ) (y : ℕ) (g : F2) (as : Atoms) : ℂ :=
  ((as.filter (reads y)).filter (isAlpha y g)).map (A ·.1) |>.prod

/-- The product of the coefficients of the other atoms reading `y`. -/
noncomputable def betaY (A : ℕ → ℂ) (y : ℕ) (g : F2) (as : Atoms) : ℂ :=
  ((as.filter (reads y)).filter (fun a => !isAlpha y g a)).map (A ·.1) |>.prod

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
theorem fz_of_normF {P Q : F2} (h : normF P = normF Q) (τ : ℕ → ZMod 2) :
    fz τ P = fz τ Q := by
  rw [← fz_normF τ P, h, fz_normF]

theorem aval_reads (A : ℕ → ℂ) (y : ℕ) (g : F2) (τ : ℕ → ZMod 2) (hg : ∀ m ∈ g, y ∉ m) :
    ∀ as : Atoms, (as.filter (reads y)).all
        (fun a => isAlpha y g a || decide (normF a.2 = normF (onlyY y g))) = true →
      aval A (as.filter (reads y)) (Function.update τ y 0) =
          (if fz τ g = 0 then 1 else alphaY A y g as) ∧
        aval A (as.filter (reads y)) (Function.update τ y 1) =
          (if fz τ g = 0 then 1 else betaY A y g as)
  | [], _ => by simp [aval, alphaY, betaY]
  | a :: as, h => by
    have hgu : ∀ b : ZMod 2, fz (Function.update τ y b) g = fz τ g :=
      fun b => fz_update_not τ y b g hg
    by_cases ha : reads y a = true
    · simp only [List.filter_cons, ha, ↓reduceIte, List.all_cons, Bool.and_eq_true] at h
      obtain ⟨hform, hrest⟩ := h
      obtain ⟨ih0, ih1⟩ := aval_reads A y g τ hg as hrest
      simp only [alphaY, betaY, List.filter_cons, ha, ↓reduceIte] at ih0 ih1 ⊢
      by_cases hal : isAlpha y g a = true
      · have hf : ∀ τ', fz τ' a.2 = (1 + τ' y) * fz τ' g := fun τ' => by
          rw [fz_of_normF (of_decide_eq_true hal), fz_oneY]
        simp only [hal, Bool.not_true, Bool.false_eq_true, ↓reduceIte, List.map_cons,
          List.prod_cons]
        rw [aval_cons, aval_cons, ih0, ih1, hf, hf, hgu, hgu]
        simp only [Function.update_self]
        rcases z2_cases (fz τ g) with e | e <;> simp [e, z2_one_add_one]
      · have hal' : isAlpha y g a = false := by simpa using hal
        have hb : normF a.2 = normF (onlyY y g) := by
          simpa [hal'] using hform
        have hf : ∀ τ', fz τ' a.2 = τ' y * fz τ' g := fun τ' => by
          rw [fz_of_normF hb, fz_onlyY]
        simp only [hal', Bool.not_false, Bool.false_eq_true, ↓reduceIte, List.map_cons,
          List.prod_cons]
        rw [aval_cons, aval_cons, ih0, ih1, hf, hf, hgu, hgu]
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

/-! ## Holding a path variable -/

/-- Whether the head path variable is in neither the rest of the path, `held` nor an atom. -/
def holdOK (t : AData) : Bool :=
  match t.s.path with
  | [] => false
  | p :: ps => !decide (p ∈ ps) && !decide (p ∈ t.held) && !decide (p ∈ avars t.atoms)

/-- The head path variable moved to the front of `held`. -/
def holdS (t : AData) : AData :=
  match t.s.path with
  | [] => t
  | p :: ps => ⟨{ t.s with path := ps }, p :: t.held, t.atoms⟩

theorem hold_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (t : AData) (σ : ℕ → ZMod 2)
    (h : holdOK t = true) : valA M c A t σ = valA M c A (holdS t) σ := by
  rcases t with ⟨⟨path, phase, cons, rows⟩, held, as⟩
  rcases path with _ | ⟨p, ps⟩
  · simp [holdOK] at h
  simp only [holdOK, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  have hr : releaseOK (holdS ⟨⟨p :: ps, phase, cons, rows⟩, held, as⟩) = true := by
    simp [releaseOK, holdS, h1, h2, h3]
  rw [release_sound M c A _ σ hr]
  rfl

/-! ## Atoms substituted along a constraint -/

/-- Whether no variable of `q` is a path variable of `s`. -/
def offPath (s : Data) (q : F2) : Bool := (q.flatMap id).all fun x => !decide (x ∈ s.path)

/-- The first constraint or row pivotable at `v` and off the path. -/
def asubQ (v : ℕ) (s : Data) : Option F2 :=
  (s.cons ++ s.rows).find? fun q => pivotable v q && offPath s q

/-- Whether some constraint or row off the path is pivotable at `v`. -/
def asubOK (v : ℕ) (t : AData) : Bool := (asubQ v t.s).isSome

/-- `v` set to the rest of that constraint in every atom. -/
def asubS (v : ℕ) (t : AData) : AData :=
  match asubQ v t.s with
  | none => t
  | some q => { t with atoms := t.atoms.map fun a => (a.1, normF (substF v (q.erase [v]) a.2)) }

theorem aval_map (A : ℕ → ℂ) (τ : ℕ → ZMod 2) (F : F2 → F2) (hF : ∀ f, fz τ (F f) = fz τ f) :
    ∀ as : Atoms, aval A (as.map fun a => (a.1, F a.2)) τ = aval A as τ
  | [] => rfl
  | a :: as => by
    rw [List.map_cons, aval_cons, aval_cons, hF, aval_map A τ F hF as]

theorem offPath_not {s : Data} {q : F2} (h : offPath s q = true) :
    ∀ y ∈ s.path, ∀ m ∈ q, y ∉ m := by
  intro y hy m hm hym
  have := List.all_eq_true.1 h y (List.mem_flatMap.2 ⟨m, hm, hym⟩)
  simp [hy] at this

/-- A constraint or row off the path as a factor of the value. -/
theorem val_ind (M : ℕ) (c : ℂ) (s : Data) (q : F2) (hq : q ∈ s.cons ++ s.rows)
    (ho : offPath s q = true) (τ : ℕ → ZMod 2) :
    val M c s τ = (if fz τ q = 0 then 1 else 0) * val M c s τ := by
  unfold val
  rw [← bsum_mul_left (fun τ' => if fz τ' q = 0 then (1 : ℂ) else 0) _ s.path
    (fun τ' y hy b => by rw [fz_update_not τ' y b q (offPath_not ho y hy)]) τ]
  apply bsum_congr
  intro τ'
  unfold summand
  by_cases h0 : fz τ' q = 0
  · simp [h0]
  · have : ¬ ∀ r ∈ s.cons ++ s.rows, fz τ' r = 0 := fun H => h0 (H q hq)
    rw [if_neg this, if_neg h0, zero_mul]

theorem asub_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (v : ℕ) (t : AData) (σ : ℕ → ZMod 2)
    (h : asubOK v t = true) : valA M c A t σ = valA M c A (asubS v t) σ := by
  simp only [asubOK, Option.isSome_iff_exists] at h
  obtain ⟨q, hq⟩ := h
  have hiq : q ∈ t.s.cons ++ t.s.rows := List.mem_of_find?_eq_some hq
  have hpq := List.find?_some hq
  simp only [Bool.and_eq_true] at hpq
  obtain ⟨hpv, ho⟩ := hpq
  simp only [pivotable, Bool.and_eq_true, decide_eq_true_eq] at hpv
  obtain ⟨hvq, -⟩ := hpv
  simp only [valA, asubS, hq]
  apply bsum_congr
  intro τ
  rw [val_ind M c t.s q hiq ho τ]
  by_cases h0 : fz τ q = 0
  · have hqv : fz τ q = τ v + fz τ (q.erase [v]) := by
      rw [fz_perm _ (List.perm_cons_erase hvq), fz_cons]
      simp [mz]
    have hv : fz τ (q.erase [v]) = τ v :=
      (z2_eq_of_add _ _ (by rw [← hqv]; exact h0)).symm
    rw [aval_map A τ (fun f => normF (substF v (q.erase [v]) f))
      (fun f => by rw [fz_normF, fz_substF, hv, Function.update_eq_self])]
  · simp [h0]

/-- Whether some constraint or row off the path is `[v]` plus `B`. -/
def asubBOK (v : ℕ) (B : F2) (t : AData) : Bool :=
  (t.s.cons ++ t.s.rows).any fun q => pinAt v B q && offPath t.s q

/-- `v` set to `B` in every atom. -/
def asubBS (v : ℕ) (B : F2) (t : AData) : AData :=
  { t with atoms := t.atoms.map fun a => (a.1, normF (substF v B a.2)) }

theorem asubB_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (v : ℕ) (B : F2) (t : AData)
    (σ : ℕ → ZMod 2) (h : asubBOK v B t = true) :
    valA M c A t σ = valA M c A (asubBS v B t) σ := by
  simp only [asubBOK, List.any_eq_true, Bool.and_eq_true] at h
  obtain ⟨q, hiq, hpv, ho⟩ := h
  simp only [pinAt, pivotable, Bool.and_eq_true, decide_eq_true_eq] at hpv
  obtain ⟨⟨hvq, -⟩, hB⟩ := hpv
  subst hB
  simp only [valA, asubBS]
  apply bsum_congr
  intro τ
  rw [val_ind M c t.s q hiq ho τ]
  by_cases h0 : fz τ q = 0
  · have hqv : fz τ q = τ v + fz τ (q.erase [v]) := by
      rw [fz_perm _ (List.perm_cons_erase hvq), fz_cons]
      simp [mz]
    have hv : fz τ (q.erase [v]) = τ v :=
      (z2_eq_of_add _ _ (by rw [← hqv]; exact h0)).symm
    rw [aval_map A τ (fun f => normF (substF v (q.erase [v]) f))
      (fun f => by rw [fz_normF, fz_substF, hv, Function.update_eq_self])]
  · simp [h0]

/-! ## Atoms on a zero coefficient, the coefficient 1, and merged atoms -/

theorem summand_consApp (M : ℕ) (c : ℂ) (s : Data) (f : F2) (τ : ℕ → ZMod 2) :
    summand M c { s with cons := s.cons ++ [f] } τ =
      (if fz τ f = 0 then 1 else 0) * summand M c s τ := by
  unfold summand
  dsimp only
  have e : (∀ q ∈ s.cons ++ [f] ++ s.rows, fz τ q = 0) ↔
      (fz τ f = 0 ∧ ∀ q ∈ s.cons ++ s.rows, fz τ q = 0) := by
    simp only [List.mem_append, List.mem_singleton]
    constructor
    · intro H
      exact ⟨H f (Or.inl (Or.inr rfl)), fun q hq => hq.elim (fun h => H q (Or.inl (Or.inl h)))
        (fun h => H q (Or.inr h))⟩
    · rintro ⟨hf, H⟩ q ((hq | rfl) | hq)
      · exact H q (Or.inl hq)
      · exact hf
      · exact H q (Or.inr hq)
  rw [if_congr e rfl rfl]
  by_cases hf : fz τ f = 0
  · simp only [hf, true_and, if_true, one_mul]
  · simp only [hf, false_and, if_false, zero_mul]

/-- Whether the head atom has coefficient index `j` and a polynomial off the path. -/
def azeroOK (j : ℕ) (t : AData) : Bool :=
  match t.atoms with
  | [] => false
  | a :: _ => decide (a.1 = j) && offPath t.s a.2

/-- The head atom's polynomial made a constraint. -/
def azeroS (t : AData) : AData :=
  match t.atoms with
  | [] => t
  | a :: as => ⟨{ t.s with cons := t.s.cons ++ [a.2] }, t.held, as⟩

theorem azero_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (j : ℕ) (t : AData) (σ : ℕ → ZMod 2)
    (hA : A j = 0) (h : azeroOK j t = true) : valA M c A t σ = valA M c A (azeroS t) σ := by
  rcases t with ⟨s, held, as⟩
  rcases as with _ | ⟨a, as⟩
  · simp [azeroOK] at h
  simp only [azeroOK, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨hj, ho⟩ := h
  simp only [valA, azeroS]
  apply bsum_congr
  intro τ
  have hv : val M c { s with cons := s.cons ++ [a.2] } τ =
      (if fz τ a.2 = 0 then 1 else 0) * val M c s τ := by
    unfold val
    rw [← bsum_mul_left (fun τ' => if fz τ' a.2 = 0 then (1 : ℂ) else 0) _ s.path
      (fun τ' y hy b => by rw [fz_update_not τ' y b a.2 (offPath_not ho y hy)]) τ]
    exact bsum_congr _ _ (fun τ' => summand_consApp M c s a.2 τ') _ _
  rw [hv, aval_cons, hj, hA]
  by_cases h0 : fz τ a.2 = 0 <;> simp [h0]

/-- Whether the head atom has coefficient index `j`. -/
def aheadOK (j : ℕ) (t : AData) : Bool :=
  match t.atoms with
  | [] => false
  | a :: _ => decide (a.1 = j)

theorem aone_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (j : ℕ) (t : AData) (σ : ℕ → ZMod 2)
    (hA : A j = 1) (h : aheadOK j t = true) : valA M c A t σ = valA M c A (aconstS t) σ := by
  rcases t with ⟨s, held, as⟩
  rcases as with _ | ⟨a, as⟩
  · simp [aheadOK] at h
  simp only [aheadOK, decide_eq_true_eq] at h
  simp only [valA, aconstS, List.tail_cons]
  apply bsum_congr
  intro τ
  rw [aval_cons, h, hA]
  simp

/-- Whether the head two atoms have indices `j1`, `j2` and one polynomial. -/
def amergeOK (j1 j2 : ℕ) (t : AData) : Bool :=
  match t.atoms with
  | a :: b :: _ => decide (a.1 = j1) && decide (b.1 = j2) && decide (a.2 = b.2)
  | _ => false

/-- The head two atoms replaced by one atom of index `j`. -/
def amergeS (j : ℕ) (t : AData) : AData :=
  match t.atoms with
  | a :: _ :: as => { t with atoms := (j, a.2) :: as }
  | _ => t

theorem amerge_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (j j1 j2 : ℕ) (t : AData) (σ : ℕ → ZMod 2)
    (hA : A j = A j1 * A j2) (h : amergeOK j1 j2 t = true) :
    valA M c A t σ = valA M c A (amergeS j t) σ := by
  rcases t with ⟨s, held, as⟩
  rcases as with _ | ⟨a, _ | ⟨b, as⟩⟩
  · simp [amergeOK] at h
  · simp [amergeOK] at h
  simp only [amergeOK, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  simp only [valA, amergeS]
  apply bsum_congr
  intro τ
  rw [aval_cons, aval_cons, aval_cons, ← h3, h1, h2, hA]
  by_cases h0 : fz τ a.2 = 0
  · simp only [h0, if_true, one_mul]
  · simp only [h0, if_false]
    ring

/-! ## Reordering and normalising atoms -/

theorem aval_perm (A : ℕ → ℂ) (τ : ℕ → ZMod 2) {as bs : Atoms} (h : as.Perm bs) :
    aval A as τ = aval A bs τ :=
  (h.map _).prod_eq

/-- The atom at position `i` moved to the front. -/
def apickS (i : ℕ) (t : AData) : AData :=
  { t with atoms := t.atoms.getD i (0, []) :: t.atoms.erase (t.atoms.getD i (0, [])) }

theorem apick_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (i : ℕ) (t : AData) (σ : ℕ → ZMod 2)
    (h : decide (i < t.atoms.length) = true) : valA M c A t σ = valA M c A (apickS i t) σ := by
  have hi : i < t.atoms.length := of_decide_eq_true h
  have hm : t.atoms.getD i (0, []) ∈ t.atoms := by
    rw [List.getD_eq_getElem _ _ hi]
    exact List.getElem_mem hi
  unfold valA apickS
  exact bsum_congr _ _ (fun τ => by rw [aval_perm A τ (List.perm_cons_erase hm)]) _ _

/-- Every atom's polynomial normalised. -/
def anormS (t : AData) : AData := { t with atoms := t.atoms.map fun a => (a.1, normF a.2) }

theorem anorm_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (t : AData) (σ : ℕ → ZMod 2) :
    valA M c A t σ = valA M c A (anormS t) σ := by
  unfold valA anormS
  exact bsum_congr _ _
    (fun τ => by rw [aval_map A τ normF (fun f => fz_normF τ f)]) _ _

/-- `held` reordered. -/
def hpermS (l : List ℕ) (t : AData) : AData := { t with held := l }

theorem hperm_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (l : List ℕ) (t : AData) (σ : ℕ → ZMod 2)
    (h : decide (t.held.Perm l) = true) : valA M c A t σ = valA M c A (hpermS l t) σ := by
  unfold valA hpermS
  exact bsum_perm _ (of_decide_eq_true h) σ

/-! ## [Average] through a half phase -/

/-- A held variable's half phase `y Q / 2`, `Q` off the path, as a sign outside the inner sum. -/
theorem val_half_held (M : ℕ) (hM : 0 < M) (c : ℂ) (s : Data) (y : ℕ) (τ : ℕ → ZMod 2)
    (b : ZMod 2) (hy : y ∉ s.path) (hc : y ∉ fvars (s.cons ++ s.rows))
    (hh : halfOK M y s.phase = true) (ho : offPath s (quot y s.phase) = true) :
    val M c s (Function.update τ y b) =
      (if b * fz τ (quot y s.phase) = 0 then 1 else -1) *
        val M c { s with phase := rest y s.phase } τ := by
  unfold val
  dsimp only
  rw [bsum_out _ y b s.path τ hy,
    ← bsum_mul_left (fun τ' => if b * fz τ' (quot y s.phase) = 0 then (1 : ℂ) else -1) _ s.path
      (fun τ' x hx w => by rw [fz_update_not τ' x w _ (offPath_not ho x hx)]) τ]
  apply bsum_congr
  intro τ'
  unfold summand
  dsimp only
  simp only [holds_update_not τ' y b _ hc]
  have hQ : ((((quot y s.phase).map fun m => zq (mz τ' m)).sum : ℤ) : ZMod 2) =
      fz τ' (quot y s.phase) := by
    rw [show ((quot y s.phase).map fun m => zq (mz τ' m)) =
      ((quot y s.phase).map (mz τ')).map zq by simp [List.map_map]]
    rw [zsum_cast]
    rfl
  split_ifs with hC h1 h1
  · rw [pn_half M hM τ' y b s.phase hh, ex_add]
    rcases z2_cases b with rfl | rfl
    · have : zq (0 : ZMod 2) = 0 := by decide
      rw [this]
      simp [ex_zero]
    · have : zq (1 : ZMod 2) = 1 := by decide
      rw [this, one_mul] at *
      rw [Int.cast_one, one_mul, ex_halfN, hQ, if_pos h1]
      ring
  · rw [pn_half M hM τ' y b s.phase hh, ex_add]
    rcases z2_cases b with rfl | rfl
    · simp at h1
    · have : zq (1 : ZMod 2) = 1 := by decide
      rw [one_mul] at h1
      rw [this, Int.cast_one, one_mul, ex_halfN, hQ, if_neg h1]
      ring
  · ring
  · ring

/-- Whether `y` is the head held variable, absent from the rest of `held`, the path, the
constraints and the rows, its phase terms of coefficient `M / 2` with a quotient off the path,
and every atom reading it normalises to `1 + y` or `y`. -/
def averageHOK (M y : ℕ) (t : AData) : Bool :=
  decide (0 < M) && decide (t.held.head? = some y) && !decide (y ∈ t.held.tail) &&
    !decide (y ∈ t.s.path) && !decide (y ∈ fvars (t.s.cons ++ t.s.rows)) &&
    halfOK M y t.s.phase && offPath t.s (quot y t.s.phase) &&
    (t.atoms.filter (reads y)).all fun a =>
      isAlpha y [[]] a || decide (normF a.2 = normF (onlyY y [[]]))

/-- The held variable `y` summed through its half phase `y Q / 2`: its phase terms dropped and
the atoms reading it replaced by `(j0, 1 + Q)` and `(j1, Q)`. -/
def averageHS (y j0 j1 : ℕ) (t : AData) : AData :=
  ⟨{ t.s with phase := rest y t.s.phase }, t.held.tail,
    t.atoms.filter (fun a => !reads y a) ++
      [(j0, normF ([] :: quot y t.s.phase)), (j1, normF (quot y t.s.phase))]⟩

theorem averageH_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (y j0 j1 : ℕ) (t : AData)
    (σ : ℕ → ZMod 2) (h : averageHOK M y t = true)
    (h0 : 2 * A j0 = alphaY A y [[]] t.atoms + betaY A y [[]] t.atoms)
    (h1 : 2 * A j1 = alphaY A y [[]] t.atoms - betaY A y [[]] t.atoms) :
    valA M c A t σ = valA M (2 * c) A (averageHS y j0 j1 t) σ := by
  rcases t with ⟨s, held, as⟩
  simp only [averageHOK, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨hM, hhd⟩, htl⟩, hpath⟩, hcons⟩, hh⟩, ho⟩, hall⟩ := h
  rcases held with _ | ⟨z, hs⟩
  · simp at hhd
  simp only [List.head?_cons, Option.some.injEq] at hhd
  subst hhd
  simp only [List.tail_cons] at htl
  have hg : ∀ m ∈ ([[]] : F2), z ∉ m := by simp
  simp only [valA, averageHS, List.tail_cons]
  rw [bsum_head _ z hs σ htl]
  apply bsum_congr
  intro τ
  obtain ⟨e0, e1⟩ := aval_reads A z [[]] τ hg as hall
  have hrest : ∀ b : ZMod 2, aval A (as.filter fun a => !reads z a) (Function.update τ z b) =
      aval A (as.filter fun a => !reads z a) τ :=
    fun b => aval_update_not A τ z b _ (avars_not_reads z as)
  have hone : fz τ [[]] = 1 := by simp [fz, mz]
  have hm : mz τ [] = 1 := by simp [mz]
  rw [aval_split A z (Function.update τ z 0) as, aval_split A z (Function.update τ z 1) as,
    e0, e1, hrest, hrest, val_half_held M hM c s z τ 0 hpath hcons hh ho,
    val_half_held M hM c s z τ 1 hpath hcons hh ho, aval_append, aval_cons, aval_cons, aval_nil,
    val_smul, fz_normF, fz_normF, hone, fz_cons, hm]
  simp only [one_ne_zero, ↓reduceIte, zero_mul, one_mul]
  rcases z2_cases (fz τ (quot z s.phase)) with e | e
  · simp only [e, add_zero, one_ne_zero, ↓reduceIte, mul_one, one_mul]
    linear_combination (aval A (List.filter (fun a => !reads z a) as) τ *
      val M c { s with phase := rest z s.phase } τ) * h0.symm
  · simp only [e, z2_one_add_one, one_ne_zero, ↓reduceIte, one_mul]
    linear_combination (aval A (List.filter (fun a => !reads z a) as) τ *
      val M c { s with phase := rest z s.phase } τ) * h1.symm

/-! ## [Quarter] -/

/-- `M / 4` times the lift of `B0` plus `M / 2` times the lift of `B1`. -/
def qLift (M : ℕ) (B0 B1 : F2) : Poly :=
  (lift M B0).map (fun u => ([] ++ u.1, (M / 4) * u.2)) ++
    (lift M B1).map (fun u => ([] ++ u.1, (M / 2) * u.2))

/-- Whether the head path variable `y` is not in the tail, a constraint or a row, `4 ∣ M`, `B0`
and `B1` are off the path, and the terms of `y` normalise to `qLift M B0 B1`. -/
def quarterOK (M : ℕ) (B0 B1 : F2) (t : AData) : Bool :=
  match t.s.path with
  | [] => false
  | y :: ys => decide (0 < M) && decide (M % 4 = 0) && !decide (y ∈ ys) &&
      !decide (y ∈ fvars (t.s.cons ++ t.s.rows)) && offPath t.s B0 && offPath t.s B1 &&
      decide (normP M (yq y t.s.phase) = normP M (qLift M B0 B1))

/-- The head path variable summed: the phase gains `3 M / 4` times the lift of `B0 B1` and the
constraint `(1 + B0) B1` is appended. -/
def quarterD (M : ℕ) (B0 B1 : F2) (s : Data) : Data :=
  match s.path with
  | [] => s
  | y :: ys =>
    { path := ys,
      phase := normP M (rest y s.phase ++ (lift M (f2mul B0 B1)).map
        fun u => ([] ++ u.1, (3 * (M / 4)) * u.2)),
      cons := s.cons ++ [normF (f2mul ([] :: B0) B1)], rows := s.rows }

/-- `quarterD` on the inner sum and the atom `(j, B0)` appended. -/
def quarterS (M : ℕ) (B0 B1 : F2) (j : ℕ) (t : AData) : AData :=
  ⟨quarterD M B0 B1 t.s, t.held, t.atoms ++ [(j, B0)]⟩

theorem peq_symm {M : ℕ} {a b : ℤ} (h : peq M a b) : peq M b a := by
  obtain ⟨n, hn⟩ := h
  exact ⟨-n, by rw [hn]; ring⟩

theorem ex_quarter_sq : ex (1 / 4) * ex (1 / 4) = -1 := by
  rw [← ex_add, show (1 / 4 + 1 / 4 : ℝ) = 1 / 2 by norm_num, Sym.ex_half]

/-- `ex (r N / M)` for `M = 4 r`. -/
theorem ex_quarters (M r : ℕ) (hr : M = 4 * r) (hr0 : 0 < r) (N : ℤ) :
    ex ((((r : ℤ) * N : ℤ) : ℝ) / M) = ex ((N : ℝ) / 4) := by
  congr 1
  have : (r : ℝ) ≠ 0 := by exact_mod_cast hr0.ne'
  rw [hr]
  push_cast
  field_simp

theorem om_quarter (M r : ℕ) (hr : M = 4 * r) (hr0 : 0 < r) : om M ^ (M / 4) = ex (1 / 4) := by
  have h4 : M / 4 = r := by omega
  rw [h4, om_eq_ex]
  have := ex_quarters M r hr hr0 1
  simpa using this

theorem pn_qLift (τ : ℕ → ZMod 2) (M r : ℕ) (hM : 2 ≤ M) (hr : M = 4 * r) (B0 B1 : F2) :
    peq M (pn τ (qLift M B0 B1)) ((r : ℤ) * (zq (fz τ B0) + 2 * zq (fz τ B1))) := by
  have h4 : M / 4 = r := by omega
  have h2 : M / 2 = 2 * r := by omega
  unfold qLift
  rw [pn_append, pn_mapmul, pn_mapmul, h4, h2]
  have hm1 : zq (mz τ []) = 1 := by simp [mz]; decide
  rw [hm1]
  obtain ⟨n0, e0⟩ := pn_lift τ M hM B0
  obtain ⟨n1, e1⟩ := pn_lift τ M hM B1
  refine ⟨r * n0 + 2 * r * n1, ?_⟩
  rw [e0, e1]
  push_cast
  ring

/-- The sum over a bit `y` of `ex (y (b0 + 2 b1) / 4)`. -/
theorem quarter_pt (a c e : ℂ) (b0 b1 : ZMod 2) (ha : 2 * a = 1 + ex (1 / 4)) (P : Prop)
    [Decidable P] :
    (1 + ex (((zq b0 + 2 * zq b1 : ℤ) : ℝ) / 4)) * (if P then c * e else 0) =
      2 * (if b0 = 0 then 1 else a) * ((if (1 + b0) * b1 = 0 then 1 else 0) *
        (if P then c * (e * ex (((3 * zq (b0 * b1) : ℤ) : ℝ) / 4)) else 0)) := by
  have hI := ex_quarter_sq
  have h3 : ex (3 / 4) = -ex (1 / 4) := by
    rw [show (3 / 4 : ℝ) = 1 / 4 + 1 / 2 by norm_num, ex_add, Sym.ex_half]; ring
  have n1 : (1 : ZMod 2) ≠ 0 := by decide
  by_cases hP : P
  · rw [if_pos hP, if_pos hP]
    rcases z2_cases b0 with rfl | rfl <;> rcases z2_cases b1 with rfl | rfl
    · have hx : ex (((zq (0 : ZMod 2) + 2 * zq (0 : ZMod 2) : ℤ) : ℝ) / 4) = 1 := by
        rw [zq0]; norm_num [ex_zero]
      have hy : ex (((3 * zq ((0 : ZMod 2) * 0) : ℤ) : ℝ) / 4) = 1 := by
        rw [mul_zero, zq0]; norm_num [ex_zero]
      rw [hx, hy, if_pos rfl, if_pos (show (1 + (0 : ZMod 2)) * 0 = 0 by decide)]
      ring
    · have hx : ex (((zq (0 : ZMod 2) + 2 * zq (1 : ZMod 2) : ℤ) : ℝ) / 4) = -1 := by
        rw [zq0, zq1]; norm_num; exact Sym.ex_half
      rw [hx, if_neg (show (1 + (0 : ZMod 2)) * 1 ≠ 0 by decide)]
      ring
    · have hx : ex (((zq (1 : ZMod 2) + 2 * zq (0 : ZMod 2) : ℤ) : ℝ) / 4) = ex (1 / 4) := by
        rw [zq0, zq1]; norm_num
      have hy : ex (((3 * zq ((1 : ZMod 2) * 0) : ℤ) : ℝ) / 4) = 1 := by
        rw [mul_zero, zq0]; norm_num [ex_zero]
      rw [hx, hy, if_neg n1, if_pos (show (1 + (1 : ZMod 2)) * 0 = 0 by decide)]
      linear_combination (c * e) * ha.symm
    · have hx : ex (((zq (1 : ZMod 2) + 2 * zq (1 : ZMod 2) : ℤ) : ℝ) / 4) = -ex (1 / 4) := by
        rw [zq1]; norm_num; exact h3
      have hy : ex (((3 * zq ((1 : ZMod 2) * 1) : ℤ) : ℝ) / 4) = -ex (1 / 4) := by
        rw [mul_one, zq1]; norm_num; exact h3
      rw [hx, hy, if_neg n1, if_pos (show (1 + (1 : ZMod 2)) * 1 = 0 by decide)]
      linear_combination (c * e * ex (1 / 4)) * ha + c * e * hI
  · rw [if_neg hP, if_neg hP]
    ring

/-- The value with the head path variable summed by [Quarter]. -/
theorem quarter_val (M : ℕ) (hM : 2 ≤ M) (c a : ℂ) (B0 B1 : F2) (s : Data) (τ : ℕ → ZMod 2)
    (y : ℕ) (ys : List ℕ) (hp : s.path = y :: ys) (h4 : M % 4 = 0) (hy : y ∉ ys)
    (hc : y ∉ fvars (s.cons ++ s.rows)) (h0 : offPath s B0 = true) (h1 : offPath s B1 = true)
    (hn : normP M (yq y s.phase) = normP M (qLift M B0 B1)) (ha : 2 * a = 1 + ex (1 / 4)) :
    val M c s τ = (2 * (if fz τ B0 = 0 then 1 else a)) * val M c (quarterD M B0 B1 s) τ := by
  obtain ⟨r, hr⟩ : ∃ r, M = 4 * r := ⟨M / 4, by omega⟩
  have hr0 : 0 < r := by omega
  have hM0 : 0 < M := by omega
  have h4r : M / 4 = r := by omega
  have off0 : ∀ x ∈ ys, ∀ m ∈ B0, x ∉ m := fun x hx =>
    offPath_not h0 x (by rw [hp]; exact List.mem_cons_of_mem _ hx)
  have hinv : ∀ τ', ∀ x ∈ ys, ∀ b : ZMod 2,
      (2 * (if fz (Function.update τ' x b) B0 = 0 then 1 else a)) =
        2 * (if fz τ' B0 = 0 then 1 else a) := by
    intro τ' x hx b
    rw [fz_update_not τ' x b B0 (off0 x hx)]
  set P' := normP M (rest y s.phase ++ (lift M (f2mul B0 B1)).map
    fun u => ([] ++ u.1, (3 * (M / 4)) * u.2)) with hP'
  set K := normF (f2mul ([] :: B0) B1) with hK
  have hD : quarterD M B0 B1 s = ⟨ys, P', s.cons ++ [K], s.rows⟩ := by
    unfold quarterD
    rw [hp]
  rw [hD]
  unfold val
  rw [hp, bsum_head _ y ys τ hy]
  have hm := bsum_mul_left (fun τ' => 2 * (if fz τ' B0 = 0 then 1 else a))
    (summand M c ⟨ys, P', s.cons ++ [K], s.rows⟩) ys hinv τ
  rw [show (⟨ys, P', s.cons ++ [K], s.rows⟩ : Data).path = ys from rfl, ← hm]
  apply bsum_congr
  intro τ'
  have hs : summand M c ⟨ys, P', s.cons ++ [K], s.rows⟩ τ' =
      (if fz τ' K = 0 then 1 else 0) * summand M c ⟨ys, P', s.cons, s.rows⟩ τ' :=
    summand_consApp M c ⟨ys, P', s.cons, s.rows⟩ K τ'
  rw [hs, summand_pair M c s τ' y hc]
  unfold summand
  dsimp only
  have hY : ex ((pn τ' (yq y s.phase) : ℝ) / M) =
      ex (((zq (fz τ' B0) + 2 * zq (fz τ' B1) : ℤ) : ℝ) / 4) := by
    have e1 : peq M (pn τ' (yq y s.phase)) ((r : ℤ) * (zq (fz τ' B0) + 2 * zq (fz τ' B1))) :=
      peq_trans (peq_symm (pn_normP τ' M _))
        (by rw [hn]; exact peq_trans (pn_normP τ' M _) (pn_qLift τ' M r hM hr B0 B1))
    rw [ex_peq hM0 e1, ex_quarters M r hr hr0]
  have hPx : ex ((pn τ' P' : ℝ) / M) = ex ((pn τ' (rest y s.phase) : ℝ) / M) *
      ex (((3 * zq (fz τ' B0 * fz τ' B1) : ℤ) : ℝ) / 4) := by
    have e2 : peq M (pn τ' P')
        (pn τ' (rest y s.phase) + (r : ℤ) * (3 * zq (fz τ' B0 * fz τ' B1))) := by
      refine peq_trans (pn_normP τ' M _) ?_
      rw [pn_append, pn_mapmul, h4r]
      have hm1 : zq (mz τ' []) = 1 := by simp [mz]; decide
      rw [hm1]
      obtain ⟨n, hn'⟩ := pn_lift τ' M hM (f2mul B0 B1)
      rw [fz_f2mul] at hn'
      refine ⟨3 * r * n, ?_⟩
      rw [hn']
      push_cast
      ring
    rw [ex_peq hM0 e2, Int.cast_add, add_div, ex_add, ex_quarters M r hr hr0]
  have hKv : fz τ' K = (1 + fz τ' B0) * fz τ' B1 := by
    rw [hK, fz_normF, fz_f2mul, fz_cons, show mz τ' [] = 1 by simp [mz]]
  rw [hY, hPx, hKv]
  exact quarter_pt a c _ (fz τ' B0) (fz τ' B1) ha _

theorem quarter_sound (M : ℕ) (hM : 2 ≤ M) (c : ℂ) (A : ℕ → ℂ) (B0 B1 : F2) (j : ℕ) (t : AData)
    (σ : ℕ → ZMod 2) (h : quarterOK M B0 B1 t = true) (hA : 2 * A j = 1 + om M ^ (M / 4)) :
    valA M c A t σ = valA M (2 * c) A (quarterS M B0 B1 j t) σ := by
  rcases t with ⟨s, held, as⟩
  rcases hp : s.path with _ | ⟨y, ys⟩
  · simp [quarterOK, hp] at h
  simp only [quarterOK, hp, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨_, h4⟩, hy⟩, hc⟩, h0⟩, h1⟩, hn⟩ := h
  obtain ⟨r, hr⟩ : ∃ r, M = 4 * r := ⟨M / 4, by omega⟩
  have ha : 2 * A j = 1 + ex (1 / 4) := by rw [hA, om_quarter M r hr (by omega)]
  simp only [valA, quarterS]
  apply bsum_congr
  intro τ
  rw [quarter_val M hM c (A j) B0 B1 s τ y ys hp h4 hy hc h0 h1 hn ha, aval_append, aval_cons,
    aval_nil, val_smul]
  ring

/-! ## Complementary atoms -/

/-- Whether the head two atoms have the index `j` and polynomials adding to 1. -/
def acompOK (j : ℕ) (t : AData) : Bool :=
  match t.atoms with
  | a :: b :: _ => decide (a.1 = j) && decide (b.1 = j) && decide (normF (a.2 ++ b.2) = [[]])
  | _ => false

/-- The head two atoms dropped. -/
def acompS (t : AData) : AData := { t with atoms := t.atoms.drop 2 }

theorem acomp_sound (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (j : ℕ) (t : AData) (σ : ℕ → ZMod 2)
    (h : acompOK j t = true) : valA M c A t σ = valA M (A j * c) A (acompS t) σ := by
  rcases t with ⟨s, held, as⟩
  rcases as with _ | ⟨a, _ | ⟨b, as⟩⟩
  · simp [acompOK] at h
  · simp [acompOK] at h
  simp only [acompOK, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨ha, hb⟩, hab⟩ := h
  simp only [valA, acompS, List.drop_succ_cons, List.drop_zero]
  apply bsum_congr
  intro τ
  have hs : fz τ a.2 + fz τ b.2 = 1 := by
    rw [← fz_append, ← fz_normF, hab]
    simp [fz, mz]
  rw [aval_cons, aval_cons, ha, hb, val_smul]
  have n1 : (1 : ZMod 2) ≠ 0 := by decide
  rcases z2_cases (fz τ a.2) with e | e <;> rcases z2_cases (fz τ b.2) with e' | e' <;>
    rw [e, e'] at hs ⊢
  · exact absurd hs (by decide)
  · rw [if_pos rfl, if_neg n1]; ring
  · rw [if_pos rfl, if_neg n1]; ring
  · exact absurd hs (by decide)

/-- A path sum with no held variable and no atom. -/
theorem valA_nil (M : ℕ) (c : ℂ) (A : ℕ → ℂ) (s : Data) (σ : ℕ → ZMod 2) :
    valA M c A ⟨s, [], []⟩ σ = val M c s σ := by
  simp [valA, bsum, aval_nil]

/-! ## Step lists -/

/-- A step on a path sum with atoms. -/
inductive AStep where
  | inner (st : Step)
  | hperm (l : List ℕ)
  | release
  | hold
  | asub (v : ℕ)
  | asubB (v : ℕ) (B : F2)
  | average (y : ℕ) (g : F2) (j : ℕ) (as bs : List ℕ)
  | averageH (y j0 j1 jn : ℕ) (as bs : List ℕ)
  | azero (j : ℕ)
  | aconst (j : ℕ)
  | aone (j : ℕ)
  | adrop
  | amerge (j j1 j2 : ℕ)
  | apick (i : ℕ)
  | anorm
  | quarter (B0 B1 : F2) (j : ℕ)
  | acomp (j : ℕ)

/-- The indices `j` a step assumes `2 * A j = 1 + om M ^ (M / 4)` of. -/
def AStep.qreqs : AStep → List ℕ
  | .quarter _ _ j => [j]
  | _ => []

/-- The `qreqs` of a step list. -/
def aqreqs (sts : List AStep) : List ℕ := sts.flatMap AStep.qreqs

/-- A coefficient identity `k * A j = Sum_ps Prod_{i in ps} A i`. -/
abbrev Req := ℕ × ℕ × List (List ℕ)

/-- Whether a coefficient identity holds. -/
def reqHolds (A : ℕ → ℂ) (r : Req) : Prop :=
  (r.1 : ℂ) * A r.2.1 = (r.2.2.map fun ps => (ps.map A).prod).sum

/-- The coefficient identities a step assumes. -/
def AStep.reqs : AStep → List Req
  | .average _ _ j as bs => [(2, j, [as, bs])]
  | .averageH _ j0 j1 jn as bs => [(2, j0, [as, bs]), (2, j1, [as, bs ++ [jn]]), (0, jn, [[], [jn]])]
  | .azero j => [(1, j, [])]
  | .aone j => [(1, j, [[]])]
  | .amerge j j1 j2 => [(1, j, [[j1, j2]])]
  | _ => []

/-- A step's side condition. -/
def AStep.ok (M : ℕ) : AStep → AData → Bool
  | .inner st, t => stepOK M st t.s
  | .hperm l, t => decide (t.held.Perm l)
  | .release, t => releaseOK t
  | .hold, t => holdOK t
  | .asub v, t => asubOK v t
  | .asubB v B, t => asubBOK v B t
  | .average y g _ as bs, t => averageOK y g t &&
      decide (((t.atoms.filter (reads y)).filter (isAlpha y g)).map (·.1) = as) &&
      decide (((t.atoms.filter (reads y)).filter (fun a => !isAlpha y g a)).map (·.1) = bs)
  | .averageH y _ _ _ as bs, t => averageHOK M y t &&
      decide (((t.atoms.filter (reads y)).filter (isAlpha y [[]])).map (·.1) = as) &&
      decide (((t.atoms.filter (reads y)).filter (fun a => !isAlpha y [[]] a)).map (·.1) = bs)
  | .azero j, t => azeroOK j t
  | .aconst j, t => aconstOK t && decide (aconstIdx t = j)
  | .aone j, t => aheadOK j t
  | .adrop, _ => true
  | .amerge _ j1 j2, t => amergeOK j1 j2 t
  | .apick i, t => decide (i < t.atoms.length)
  | .anorm, _ => true
  | .quarter B0 B1 _, t => quarterOK M B0 B1 t
  | .acomp j, t => acompOK j t

/-- A step applied. -/
def AStep.apply (M : ℕ) : AStep → AData → AData
  | .inner st, t => { t with s := stepS M st t.s }
  | .hperm l, t => hpermS l t
  | .release, t => releaseS t
  | .hold, t => holdS t
  | .asub v, t => asubS v t
  | .asubB v B, t => asubBS v B t
  | .average y g j _ _, t => averageS y g j t
  | .averageH y j0 j1 _ _ _, t => averageHS y j0 j1 t
  | .azero _, t => azeroS t
  | .aconst _, t => aconstS t
  | .aone _, t => aconstS t
  | .adrop, t => adropS t
  | .amerge j _ _, t => amergeS j t
  | .apick i, t => apickS i t
  | .anorm, t => anormS t
  | .quarter B0 B1 j, t => quarterS M B0 B1 j t
  | .acomp _, t => acompS t

/-- The factor a step multiplies the constant by. -/
noncomputable def AStep.fac (A : ℕ → ℂ) : AStep → ℂ
  | .inner st => PSem.fac st
  | .average _ _ _ _ _ => 2
  | .averageH _ _ _ _ _ _ => 2
  | .aconst j => A j
  | .quarter _ _ _ => 2
  | .acomp j => A j
  | _ => 1

/-- Every side condition along a step list. -/
def arunsOK (M : ℕ) : List AStep → AData → Bool
  | [], _ => true
  | st :: sts, t => st.ok M t && arunsOK M sts (st.apply M t)

/-- A step list applied. -/
def aruns (M : ℕ) : List AStep → AData → AData
  | [], t => t
  | st :: sts, t => aruns M sts (st.apply M t)

/-- The product of the steps' factors. -/
noncomputable def afacs (A : ℕ → ℂ) : List AStep → ℂ
  | [] => 1
  | st :: sts => st.fac A * afacs A sts

/-- The coefficient identities a step list assumes. -/
def areqs (sts : List AStep) : List Req := sts.flatMap AStep.reqs

theorem AStep.sound (M : ℕ) (hM : 2 ≤ M) (A : ℕ → ℂ) (st : AStep) (t : AData) (c : ℂ)
    (σ : ℕ → ZMod 2) (hr : ∀ r ∈ st.reqs, reqHolds A r)
    (hq : ∀ j ∈ st.qreqs, 2 * A j = 1 + om M ^ (M / 4)) (h : st.ok M t = true) :
    valA M c A t σ = valA M (st.fac A * c) A (st.apply M t) σ := by
  cases st with
  | quarter B0 B1 j =>
    simpa [AStep.fac, AStep.apply] using
      quarter_sound M hM c A B0 B1 j t σ h (hq j List.mem_cons_self)
  | acomp j => simpa [AStep.fac, AStep.apply] using acomp_sound M c A j t σ h
  | inner st => simpa [AStep.fac, AStep.apply] using astep_sound M hM st t c A σ h
  | hperm l => simpa [AStep.fac, AStep.apply] using hperm_sound M c A l t σ h
  | release => simpa [AStep.fac, AStep.apply] using release_sound M c A t σ h
  | hold => simpa [AStep.fac, AStep.apply] using hold_sound M c A t σ h
  | asub v => simpa [AStep.fac, AStep.apply] using asub_sound M c A v t σ h
  | asubB v B => simpa [AStep.fac, AStep.apply] using asubB_sound M c A v B t σ h
  | average y g j as bs =>
    simp only [AStep.ok, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨h1, h2⟩, h3⟩ := h
    have hreq := hr _ List.mem_cons_self
    simp only [reqHolds, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
      add_zero] at hreq
    have hA : 2 * A j = alphaY A y g t.atoms + betaY A y g t.atoms := by
      unfold alphaY betaY
      rw [← h2, ← h3, List.map_map, List.map_map] at hreq
      push_cast at hreq
      exact hreq
    simpa [AStep.fac, AStep.apply] using average_sound M c A y g j t σ h1 hA
  | averageH y j0 j1 jn as bs =>
    simp only [AStep.ok, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨h1, h2⟩, h3⟩ := h
    have r0 := hr _ List.mem_cons_self
    have r1 := hr _ (List.mem_cons_of_mem _ List.mem_cons_self)
    have r2 := hr _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self))
    simp only [reqHolds, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
      List.map_append, List.prod_append, List.prod_cons, List.prod_nil, add_zero] at r0 r1 r2
    push_cast at r0 r1 r2
    have hn : A jn = -1 := by linear_combination -r2
    have hα : (as.map A).prod = alphaY A y [[]] t.atoms := by
      unfold alphaY
      rw [← h2, List.map_map]
      rfl
    have hβ : (bs.map A).prod = betaY A y [[]] t.atoms := by
      unfold betaY
      rw [← h3, List.map_map]
      rfl
    have hA0 : 2 * A j0 = alphaY A y [[]] t.atoms + betaY A y [[]] t.atoms := by
      rw [← hα, ← hβ]
      exact r0
    have hA1 : 2 * A j1 = alphaY A y [[]] t.atoms - betaY A y [[]] t.atoms := by
      rw [← hα, ← hβ, r1, hn]
      ring
    simpa [AStep.fac, AStep.apply] using averageH_sound M c A y j0 j1 t σ h1 hA0 hA1
  | azero j =>
    have hreq := hr _ List.mem_cons_self
    simp only [reqHolds, List.map_nil, List.sum_nil, Nat.cast_one, one_mul] at hreq
    simpa [AStep.fac, AStep.apply] using azero_sound M c A j t σ hreq h
  | aconst j =>
    simp only [AStep.ok, Bool.and_eq_true, decide_eq_true_eq] at h
    have := aconst_sound M c A t σ h.1
    rw [h.2] at this
    simpa [AStep.fac, AStep.apply] using this
  | aone j =>
    have hreq := hr _ List.mem_cons_self
    simp only [reqHolds, List.map_cons, List.map_nil, List.prod_nil, List.sum_cons,
      List.sum_nil, add_zero, Nat.cast_one, one_mul] at hreq
    simpa [AStep.fac, AStep.apply] using aone_sound M c A j t σ hreq h
  | adrop => simpa [AStep.fac, AStep.apply] using adrop_sound M c A t σ
  | amerge j j1 j2 =>
    have hreq := hr _ List.mem_cons_self
    simp only [reqHolds, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
      List.sum_cons, List.sum_nil, add_zero, Nat.cast_one, one_mul, mul_one] at hreq
    simpa [AStep.fac, AStep.apply] using amerge_sound M c A j j1 j2 t σ hreq h
  | apick i => simpa [AStep.fac, AStep.apply] using apick_sound M c A i t σ h
  | anorm => simpa [AStep.fac, AStep.apply] using anorm_sound M c A t σ

theorem aruns_sound (M : ℕ) (hM : 2 ≤ M) (A : ℕ → ℂ) :
    ∀ (sts : List AStep) (t : AData) (c : ℂ) (σ : ℕ → ZMod 2),
      (∀ r ∈ areqs sts, reqHolds A r) → (∀ j ∈ aqreqs sts, 2 * A j = 1 + om M ^ (M / 4)) →
      arunsOK M sts t = true →
      valA M c A t σ = valA M (afacs A sts * c) A (aruns M sts t) σ
  | [], t, c, σ, _, _, _ => by simp [afacs, aruns]
  | st :: sts, t, c, σ, hr, hq, h => by
    simp only [arunsOK, Bool.and_eq_true] at h
    have hr1 : ∀ r ∈ st.reqs, reqHolds A r :=
      fun r hm => hr r (List.mem_flatMap.2 ⟨st, List.mem_cons_self, hm⟩)
    have hr2 : ∀ r ∈ areqs sts, reqHolds A r := fun r hm => by
      obtain ⟨st', hst', hm'⟩ := List.mem_flatMap.1 hm
      exact hr r (List.mem_flatMap.2 ⟨st', List.mem_cons_of_mem _ hst', hm'⟩)
    have hq1 : ∀ j ∈ st.qreqs, 2 * A j = 1 + om M ^ (M / 4) :=
      fun j hm => hq j (List.mem_flatMap.2 ⟨st, List.mem_cons_self, hm⟩)
    have hq2 : ∀ j ∈ aqreqs sts, 2 * A j = 1 + om M ^ (M / 4) := fun j hm => by
      obtain ⟨st', hst', hm'⟩ := List.mem_flatMap.1 hm
      exact hq j (List.mem_flatMap.2 ⟨st', List.mem_cons_of_mem _ hst', hm'⟩)
    rw [AStep.sound M hM A st t c σ hr1 hq1 h.1, aruns_sound M hM A sts _ _ σ hr2 hq2 h.2]
    simp only [afacs, aruns]
    ring_nf

/-- Two reduced sums with no held variable, atoms equal up to order, compared. -/
theorem afinal_perm (M : ℕ) (hM : 0 < M) (cL cR : ℂ) (A : ℕ → ℂ) (tL tR : AData)
    (σ : ℕ → ZMod 2) (kL kR : ℕ) (hL : tL.held = []) (hR : tR.held = [])
    (ha : decide (tL.atoms.Perm tR.atoms) = true) (h : finalOK tL.s tR.s = true)
    (hkL : constK tL.s.phase = kL) (hkR : constK tR.s.phase = kR)
    (hc : cL * om M ^ kL = cR * om M ^ kR) :
    valA M cL A tL σ = valA M cR A tR σ := by
  simp only [valA, hL, hR, bsum]
  rw [aval_perm A σ (of_decide_eq_true ha), final_sound M hM cL cR tL.s tR.s σ kL kR h hkL hkR hc]

end PSem
end ArchytasZX
