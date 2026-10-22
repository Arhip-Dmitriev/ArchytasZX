/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Refute

/-!
# Exact evaluation by contraction

A factor is a scope of labels and a table of formal sums, one per assignment of the scope.
`leafOK` checks a node's factor against `valT`; `elimOK` checks that summing one label out
of the factors containing it gives a new factor; `eval_ve` proves that a checked chain of
eliminations, in an order permuting the summed labels, gives the network's entry.
-/

namespace ArchytasZX

open Complex Finset

/-- A scope of labels and one formal sum per assignment of it. -/
structure Fac where
  scope : List ℕ
  table : List Terms
  deriving DecidableEq, Repr

/-- The index of an assignment of `ls` in mixed radix, the first label least significant. -/
def idx (D : ℕ → ℕ) : List ℕ → (ℕ → ℕ) → ℕ
  | [], _ => 0
  | l :: ls, σ => σ l + D l * idx D ls σ

/-- A factor's entry at an assignment. -/
def Fac.at (D : ℕ → ℕ) (F : Fac) (σ : ℕ → ℕ) : Terms := F.table.getD (idx D F.scope σ) []

/-- A factor's value. -/
noncomputable def fv (z : ℂ) (xs : List ℂ) (D : ℕ → ℕ) (F : Fac) (σ : ℕ → ℕ) : ℂ :=
  tv z xs (F.at D σ)

/-- The product of factor values. -/
noncomputable def fprod (z : ℂ) (xs : List ℂ) (D : ℕ → ℕ) (Fs : List Fac) (σ : ℕ → ℕ) : ℂ :=
  (Fs.map fun F => fv z xs D F σ).prod

/-- The product of factor entries as one formal sum. -/
def ffold (D : ℕ → ℕ) (Fs : List Fac) (σ : ℕ → ℕ) : Terms :=
  Fs.foldr (fun F acc => tmul (F.at D σ) acc) (tc 1)

theorem idx_congr (D : ℕ → ℕ) (ls : List ℕ) (σ τ : ℕ → ℕ) (h : ∀ l ∈ ls, σ l = τ l) :
    idx D ls σ = idx D ls τ := by
  induction ls with
  | nil => rfl
  | cons l ls ih =>
    simp only [idx, h l List.mem_cons_self, ih fun x hx => h x (List.mem_cons_of_mem _ hx)]

theorem fv_reads (z : ℂ) (xs : List ℂ) (D : ℕ → ℕ) (F : Fac) :
    Reads F.scope (fv z xs D F) := by
  intro σ τ h
  simp only [fv, Fac.at, idx_congr D F.scope σ τ h]

theorem fprod_reads (z : ℂ) (xs : List ℂ) (D : ℕ → ℕ) (Fs : List Fac) (σ τ : ℕ → ℕ)
    (h : ∀ F ∈ Fs, ∀ l ∈ F.scope, σ l = τ l) : fprod z xs D Fs σ = fprod z xs D Fs τ := by
  unfold fprod
  congr 1
  exact List.map_congr_left fun F hF => fv_reads z xs D F σ τ (h F hF)

theorem tv_ffold (z : ℂ) (xs : List ℂ) (hx : ∀ x ∈ xs, x ≠ 0) (D : ℕ → ℕ) (Fs : List Fac)
    (σ : ℕ → ℕ) : tv z xs (ffold D Fs σ) = fprod z xs D Fs σ := by
  induction Fs with
  | nil => simp [ffold, fprod, tv_tc]
  | cons F Fs ih =>
    simp only [ffold, List.foldr_cons, fprod, List.map_cons, List.prod_cons] at ih ⊢
    rw [tv_mul z xs hx, ih]; rfl

/-- A node's factor holds its entries and covers its labels. -/
def leafOK (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (n : Node) (F : Fac) : Bool :=
  (n.outs ++ n.ins).all (F.scope.contains ·) &&
    (assigns S.dim F.scope).all fun τ => F.at S.dim τ == compact M (valT S M P τ n)

theorem valT_congr (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (n : Node) (σ τ : ℕ → ℕ)
    (h : ∀ l ∈ n.outs ++ n.ins, σ l = τ l) : valT S M P σ n = valT S M P τ n := by
  have ho : n.outs.map σ = n.outs.map τ :=
    List.map_congr_left fun l hl => h l (List.mem_append_left _ hl)
  have hi : n.ins.map σ = n.ins.map τ :=
    List.map_congr_left fun l hl => h l (List.mem_append_right _ hl)
  simp only [valT, valBS, ho, hi]

theorem leaf_val (S : Shape) (M : ℕ) (hM : 0 < M) (xs : List ℂ) (P : ℕ → ℕ → ℕ × List ℤ)
    (n : Node) (F : Fac) (h : leafOK S M P n F = true) (σ : ℕ → ℕ)
    (hσ : ∀ l ∈ F.scope, σ l < S.dim l) :
    tv (om M) xs (valT S M P σ n) = fv (om M) xs S.dim F σ := by
  simp only [leafOK, Bool.and_eq_true, List.all_eq_true, List.contains_iff_mem,
    beq_iff_eq] at h
  obtain ⟨τ, hτ, hτσ⟩ := assigns_cover S.dim F.scope σ hσ
  have e1 : fv (om M) xs S.dim F σ = fv (om M) xs S.dim F τ :=
    fv_reads _ _ _ F σ τ fun l hl => (hτσ l hl).symm
  rw [e1, fv, h.2 τ hτ, tv_compact M hM]
  rw [valT_congr S M P n σ τ fun l hl => (hτσ l (h.1 l hl)).symm]

/-- Summing `l` out of the factors containing it gives the last factor of `Fs'`, the others
kept in order. -/
def elimOK (M : ℕ) (D : ℕ → ℕ) (l : ℕ) (Fs Fs' : List Fac) : Bool :=
  let ins := Fs.filter fun F => F.scope.contains l
  let outs := Fs.filter fun F => !F.scope.contains l
  match Fs'.reverse with
  | [] => false
  | new :: rest =>
    rest.reverse == outs && !new.scope.contains l &&
      (new.scope.all fun x => ins.any fun F => F.scope.contains x) &&
      (ins.all fun F => F.scope.all fun x => x == l || new.scope.contains x) &&
      (assigns D new.scope).all fun τ => new.at D τ ==
        compact M ((List.range (D l)).flatMap fun j => ffold D ins (Function.update τ l j))

theorem fprod_filter (z : ℂ) (xs : List ℂ) (D : ℕ → ℕ) (p : Fac → Bool) (Fs : List Fac)
    (σ : ℕ → ℕ) :
    fprod z xs D Fs σ = fprod z xs D (Fs.filter fun F => !p F) σ * fprod z xs D (Fs.filter p) σ := by
  induction Fs with
  | nil => simp [fprod]
  | cons F Fs ih =>
    simp only [fprod, List.filter_cons] at ih ⊢
    cases p F <;> simp [ih] <;> ring

theorem elim_val (M : ℕ) (hM : 0 < M) (xs : List ℂ) (hx : ∀ x ∈ xs, x ≠ 0) (D : ℕ → ℕ)
    (l : ℕ) (Fs Fs' : List Fac) (h : elimOK M D l Fs Fs' = true) (σ : ℕ → ℕ)
    (hσ : ∀ F ∈ Fs', ∀ x ∈ F.scope, σ x < D x) :
    ∑ j ∈ range (D l), fprod (om M) xs D Fs (Function.update σ l j) = fprod (om M) xs D Fs' σ := by
  unfold elimOK at h
  generalize hr : Fs'.reverse = R at h
  rcases R with _ | ⟨new, rest⟩
  · simp at h
  simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true', List.all_eq_true,
    List.any_eq_true, Bool.or_eq_true, List.contains_iff_mem] at h
  obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := h
  have hFs' : Fs' = rest.reverse ++ [new] := by
    rw [← List.reverse_reverse Fs', hr]; simp
  set ins := Fs.filter fun F => F.scope.contains l
  set outs := Fs.filter fun F => !F.scope.contains l
  have hnl : l ∉ new.scope := by simpa using h2
  have hnew : ∀ x ∈ new.scope, σ x < D x := hσ new (by rw [hFs']; simp)
  -- the kept factors do not read `l`
  have hout : ∀ j, fprod (om M) xs D outs (Function.update σ l j) = fprod (om M) xs D outs σ := by
    intro j
    refine fprod_reads _ _ _ _ _ _ fun F hF x hxF => ?_
    have : l ∉ F.scope := by
      have := (List.mem_filter.1 hF).2; simpa using this
    have hxl : x ≠ l := fun e => this (e ▸ hxF)
    exact Function.update_of_ne hxl _ _
  have split : ∀ τ, fprod (om M) xs D Fs τ =
      fprod (om M) xs D outs τ * fprod (om M) xs D ins τ := by
    intro τ
    have := fprod_filter (om M) xs D (fun F => F.scope.contains l) Fs τ
    simpa [outs, ins] using this
  simp only [split, hout, ← Finset.mul_sum]
  rw [hFs', h1]
  simp only [fprod, List.map_append, List.prod_append, List.map_cons, List.map_nil,
    List.prod_cons, List.prod_nil, mul_one]
  congr 1
  obtain ⟨τ, hτ, hτσ⟩ := assigns_cover D new.scope σ hnew
  rw [fv_reads _ _ _ new σ τ fun x hx => (hτσ x hx).symm, fv, h5 τ hτ, tv_compact M hM,
    tv_flatMap, ← fsum_list]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [tv_ffold _ _ hx]
  refine fprod_reads _ _ _ _ _ _ fun F hF x hxF => ?_
  rcases h4 F hF x hxF with e | e
  · subst e; simp
  · have hxl : x ≠ l := fun e' => hnl (e' ▸ e)
    rw [Function.update_of_ne hxl, Function.update_of_ne hxl]
    exact (hτσ x e).symm

/-- A chain of eliminations from `Fs` ending at `final`. -/
def chainOK (M : ℕ) (D : ℕ → ℕ) : List Fac → List (ℕ × List Fac) → List Fac → Bool
  | Fs, [], final => Fs == final
  | Fs, (l, Fs') :: steps, final => elimOK M D l Fs Fs' && chainOK M D Fs' steps final

/-- Every label of every factor in a chain lies in `U` or is eliminated later. -/
def scopesOK (U : List ℕ) : List Fac → List (ℕ × List Fac) → Bool
  | Fs, [] => Fs.all fun F => F.scope.all (U.contains ·)
  | Fs, (l, Fs') :: steps =>
    (Fs.all fun F => F.scope.all fun x => U.contains x || x == l ||
      (steps.map Prod.fst).contains x) && scopesOK U Fs' steps

theorem chain_val (M : ℕ) (hM : 0 < M) (xs : List ℂ) (hx : ∀ x ∈ xs, x ≠ 0) (D : ℕ → ℕ)
    (U : List ℕ) : ∀ (steps : List (ℕ × List Fac)) (Fs final : List Fac),
    chainOK M D Fs steps final = true → scopesOK U Fs steps = true →
    ∀ σ, (∀ x ∈ U, σ x < D x) →
      sumOver D (steps.map Prod.fst).reverse (fprod (om M) xs D Fs) σ =
        fprod (om M) xs D final σ := by
  intro steps
  induction steps with
  | nil =>
    intro Fs final h _ σ _
    simp only [chainOK, beq_iff_eq] at h
    subst h; rfl
  | cons st steps ih =>
    obtain ⟨l, Fs'⟩ := st
    intro Fs final h hs σ hσ
    simp only [chainOK, Bool.and_eq_true] at h
    simp only [scopesOK, Bool.and_eq_true] at hs
    simp only [List.map_cons, List.reverse_cons]
    rw [sumOver_append]
    have inner : ∀ τ, (∀ x ∈ U, τ x < D x) →
        (∀ x ∈ (steps.map Prod.fst), τ x < D x) →
        sumOver D [l] (fprod (om M) xs D Fs) τ = fprod (om M) xs D Fs' τ := by
      intro τ hU hS
      simp only [sumOver]
      apply elim_val M hM xs hx D l Fs Fs' h.1 τ
      intro F hF x hxF
      have hall := hs.2
      -- every label of Fs' is in U or eliminated later
      have key : ∀ F ∈ Fs', ∀ x ∈ F.scope, x ∈ U ∨ x ∈ steps.map Prod.fst := by
        cases steps with
        | nil =>
          simp only [scopesOK, List.all_eq_true, List.contains_iff_mem] at hall
          exact fun F hF x hx => Or.inl (hall F hF x hx)
        | cons s2 rest =>
          simp only [scopesOK, Bool.and_eq_true, List.all_eq_true, Bool.or_eq_true,
            List.contains_iff_mem, beq_iff_eq, List.map_cons] at hall
          intro F hF x hx
          rcases hall.1 F hF x hx with (h1 | h1) | h1
          · exact Or.inl h1
          · exact Or.inr (by simp [h1])
          · exact Or.inr (by simp [h1])
      rcases key F hF x hxF with hU' | hS'
      · exact hU x hU'
      · exact hS x hS'
    calc sumOver D (steps.map Prod.fst).reverse (sumOver D [l] (fprod (om M) xs D Fs)) σ
        = sumOver D (steps.map Prod.fst).reverse (fprod (om M) xs D Fs') σ := by
          apply sumOver_congr
          intro τ hτ hr
          apply inner τ
          · intro x hxU
            by_cases hm : x ∈ (steps.map Prod.fst).reverse
            · exact hr x hm
            · rw [hτ x hm]; exact hσ x hxU
          · intro x hx; exact hr x (by simpa using hx)
      _ = fprod (om M) xs D final σ := ih Fs' final h.2 hs.2 σ hσ

theorem prodVal_leaves (E : Env) (M : ℕ) (gs : List ℝ) (P : ℕ → ℕ → ℕ × List ℤ)
    (keys : List (ℕ × ℕ)) (hM : 0 < M) (hph : PhOK E M gs P keys) :
    ∀ (ns : List Node) (leaves : List Fac), leaves.length = ns.length →
      ns.all (nodeOK E.shape M keys) = true →
      (ns.zip leaves).all (fun p => leafOK E.shape M P p.1 p.2) = true →
      ∀ τ, (∀ F ∈ leaves, ∀ x ∈ F.scope, τ x < E.shape.dim x) →
        prodVal E ns τ = (ns.map fun n => (rt (E.dimv n.dim))⁻¹ ^ sqN E.shape n).prod *
          fprod (om M) (gs.map ex) E.shape.dim leaves τ := by
  intro ns
  induction ns with
  | nil => intro leaves hlen _ _ τ _; cases leaves <;> simp_all [prodVal, fprod]
  | cons n ns ih =>
    intro leaves hlen hok hleaf τ hτ
    cases leaves with
    | nil => simp at hlen
    | cons F Fs =>
      simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at hleaf hok
      have h1 := ih Fs (by simpa using hlen) hok.2 hleaf.2 τ (fun F hF => hτ F (by simp [hF]))
      simp only [prodVal, fprod, List.map_cons, List.prod_cons] at h1 ⊢
      rw [h1, val_cyc E M gs P keys hM hph τ n hok.1,
        leaf_val E.shape M hM (gs.map ex) P n F hleaf.1 τ (hτ F (by simp))]
      simp only [Env.shape]
      ring

theorem scopesOK_leaves (U : List ℕ) (Fs : List Fac) (steps : List (ℕ × List Fac))
    (h : scopesOK U Fs steps = true) :
    ∀ F ∈ Fs, ∀ x ∈ F.scope, x ∈ U ∨ x ∈ steps.map Prod.fst := by
  cases steps with
  | nil =>
    simp only [scopesOK, List.all_eq_true, List.contains_iff_mem] at h
    exact fun F hF x hx => Or.inl (h F hF x hx)
  | cons st rest =>
    obtain ⟨l, Fs'⟩ := st
    simp only [scopesOK, Bool.and_eq_true, List.all_eq_true, Bool.or_eq_true,
      List.contains_iff_mem, beq_iff_eq] at h
    intro F hF x hx
    rcases h.1 F hF x hx with (h1 | h1) | h1
    · exact Or.inl h1
    · exact Or.inr (by simp [h1])
    · exact Or.inr (by simp [h1])

/-- A network's entry by a checked contraction. -/
theorem eval_ve (E : Env) (M : ℕ) (gs : List ℝ) (P : ℕ → ℕ → ℕ × List ℤ)
    (keys : List (ℕ × ℕ)) (N : Net) (hM : 0 < M)
    (hok : N.nodes.all (nodeOK E.shape M keys) = true) (hph : PhOK E M gs P keys)
    (leaves : List Fac) (hlen : leaves.length = N.nodes.length)
    (hleaf : (N.nodes.zip leaves).all (fun p => leafOK E.shape M P p.1 p.2) = true)
    (steps : List (ℕ × List Fac)) (final : List Fac)
    (hperm : N.int.Perm (steps.map Prod.fst).reverse)
    (hchain : chainOK M E.shape.dim leaves steps final = true)
    (hscope : scopesOK N.bnd leaves steps = true) (σ : ℕ → ℕ)
    (hσ : ∀ x ∈ N.bnd, σ x < E.shape.dim x) :
    eval E N σ = sqK E.shape N * fprod (om M) (gs.map ex) E.shape.dim final σ := by
  have hD : labDim E = E.shape.dim := rfl
  have hK : sqK E.shape N = (N.nodes.map fun n => (rt (E.dimv n.dim))⁻¹ ^ sqN E.shape n).prod := by
    simp only [sqK, sqL, List.map_map]; rfl
  unfold eval
  rw [sumOver_perm _ hperm, hD, hK]
  have := chain_val M hM (gs.map ex) (ex_atoms_ne gs) E.shape.dim N.bnd steps leaves final
    hchain hscope σ hσ
  rw [← this, mul_sumOver]
  apply sumOver_congr
  intro τ hτ hr
  apply prodVal_leaves E M gs P keys hM hph N.nodes leaves hlen hok hleaf τ
  intro F hF x hxF
  rcases scopesOK_leaves N.bnd leaves steps hscope F hF x hxF with hb | hs
  · by_cases hm : x ∈ (steps.map Prod.fst).reverse
    · exact hr x hm
    · rw [hτ x hm]; exact hσ x hb
  · exact hr x (by simpa using hs)

/-- A network's entry as groups from its final factors. -/
theorem eval_ve_groups (E : Env) (M : ℕ) (gs : List ℝ) (hM : 0 < M) (final : List Fac)
    (σ : ℕ → ℕ) : fprod (om M) (gs.map ex) E.shape.dim final σ =
      gval (om M) (gs.map ex) (gnorm M (ffold E.shape.dim final σ)) := by
  rw [gval_gnorm _ _ M hM (om_pow_self M hM), tv_ffold _ _ (ex_atoms_ne gs)]

end ArchytasZX
