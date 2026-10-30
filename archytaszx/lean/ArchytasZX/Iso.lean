/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Net

/-!
# Isomorphic networks

`isoOK` is a decidable check that one network is another up to a bijective renaming of the
summed labels, a reordering of the nodes, and spider leg order; `iso` proves it sound.
-/

namespace ArchytasZX

open Finset

/-- The renaming by the pairs `π`, the identity off its keys. -/
def ren : List (ℕ × ℕ) → ℕ → ℕ
  | [], x => x
  | (a, b) :: π, x => if x = a then b else ren π x

theorem ren_of_not_mem (π : List (ℕ × ℕ)) (x : ℕ) (h : x ∉ π.map Prod.fst) : ren π x = x := by
  induction π with
  | nil => rfl
  | cons p π ih =>
    obtain ⟨a, b⟩ := p
    have hx : x ≠ a := fun h' => h (by simp [h'])
    simp only [ren, hx, ite_false]
    exact ih fun h' => h (List.mem_cons_of_mem _ h')

theorem ren_injective (π : List (ℕ × ℕ)) (hk : (π.map Prod.fst).Nodup)
    (hp : ((π.map Prod.fst).map (ren π)).Perm (π.map Prod.fst)) : Function.Injective (ren π) := by
  set K := π.map Prod.fst
  have hn : (K.map (ren π)).Nodup := hp.nodup_iff.2 hk
  have hin : ∀ x ∈ K, ren π x ∈ K := fun x hx => hp.subset (List.mem_map_of_mem hx)
  intro x y hxy
  by_cases hx : x ∈ K <;> by_cases hy : y ∈ K
  · exact List.inj_on_of_nodup_map hn hx hy hxy
  · rw [ren_of_not_mem π y hy] at hxy
    exact absurd (hxy ▸ hin x hx) hy
  · rw [ren_of_not_mem π x hx] at hxy
    exact absurd (hxy.symm ▸ hin y hy) hx
  · rwa [ren_of_not_mem π x hx, ren_of_not_mem π y hy] at hxy

/-- Same generator, dimension and phase, legs equal up to the generator's leg symmetry. -/
def nodeEquiv (n m : Node) : Bool :=
  n.kind == m.kind && n.dim == m.dim && n.ph == m.ph &&
    match n.kind with
    | .Z | .ZW => (n.outs ++ n.ins).isPerm (m.outs ++ m.ins)
    | .X => n.outs.isPerm m.outs && n.ins.isPerm m.ins
    | _ => n.outs == m.outs && n.ins == m.ins

theorem val_nodeEquiv (E : Env) (σ : ℕ → ℕ) (n m : Node) (h : nodeEquiv n m = true) :
    val E σ n = val E σ m := by
  obtain ⟨kn, on, inn, dn, pn⟩ := n
  obtain ⟨km, om, im, dm, pm⟩ := m
  simp only [nodeEquiv, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨rfl, rfl⟩, rfl⟩, h⟩ := h
  cases kn with
  | Z =>
    simp only [List.isPerm_iff] at h
    simp only [val, tZ, ← List.map_append]
    refine Finset.sum_congr rfl fun k _ => ?_
    have : (∀ v ∈ (on ++ inn).map σ, v = k) ↔ (∀ v ∈ (om ++ im).map σ, v = k) :=
      ⟨fun h' v hv => h' v ((h.map σ).mem_iff.2 hv),
        fun h' v hv => h' v ((h.map σ).mem_iff.1 hv)⟩
    simp only [this]
  | ZW =>
    simp only [List.isPerm_iff] at h
    simp only [val, tZw, ← List.map_append]
    refine Finset.sum_congr rfl fun k _ => ?_
    have : (∀ v ∈ (on ++ inn).map σ, v = k) ↔ (∀ v ∈ (om ++ im).map σ, v = k) :=
      ⟨fun h' v hv => h' v ((h.map σ).mem_iff.2 hv),
        fun h' v hv => h' v ((h.map σ).mem_iff.1 hv)⟩
    simp only [this]
  | X =>
    simp only [Bool.and_eq_true, List.isPerm_iff] at h
    simp only [val, tX]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [((h.1.map σ).map _).prod_eq, ((h.2.map σ).map _).prod_eq]
  | _ =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, rfl⟩ := h
    rfl

/-- A placeholder node. -/
def Node.none : Node := ⟨.I, [], [], 0, 0⟩

/-- The dimension index of a label. -/
def labIdx (labs : List (ℕ × ℕ)) (l : ℕ) : ℕ := (labs.lookup l).getD 0

/-- `B` is `A` with its summed labels renamed by `π`, nodes reordered by `p`. -/
def isoOK (labs : List (ℕ × ℕ)) (A B : Net) (π : List (ℕ × ℕ)) (p : List ℕ) : Bool :=
  let f := ren π
  let K := π.map Prod.fst
  let A' := A.nodes.map (Node.rename f)
  let P := p.map fun j => A'.getD j Node.none
  decide K.Nodup && (K.map f).isPerm K &&
    (labelsOf A.nodes).all (fun x => decide (x ∈ A.int) || !decide (x ∈ K)) &&
    A.int.all (fun x => labIdx labs (f x) == labIdx labs x) &&
    A.bnd == B.bnd && B.int.isPerm (A.int.map f) && P.isPerm A' &&
    P.length == B.nodes.length && (List.zip P B.nodes).all fun q => nodeEquiv q.1 q.2

theorem prodVal_equiv (E : Env) (σ : ℕ → ℕ) (P Q : List Node) (hl : P.length = Q.length)
    (h : (List.zip P Q).all (fun q => nodeEquiv q.1 q.2) = true) :
    prodVal E P σ = prodVal E Q σ := by
  induction P generalizing Q with
  | nil => cases Q <;> simp_all
  | cons n P ih =>
    cases Q with
    | nil => simp at hl
    | cons m Q =>
      simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at h
      simp only [prodVal, List.map_cons, List.prod_cons] at ih ⊢
      rw [val_nodeEquiv E σ n m h.1, ih Q (by simpa using hl) h.2]

theorem iso_eval (E : Env) (A B : Net) (π : List (ℕ × ℕ)) (p : List ℕ)
    (h : isoOK E.labs A B π p = true) (σ : ℕ → ℕ) : eval E A σ = eval E B σ := by
  simp only [isoOK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.isPerm_iff,
    List.all_eq_true, Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hk, hp⟩, hfree⟩, hdim⟩, -⟩, hint⟩, hP⟩, hlen⟩, hnodes⟩ := h
  set f := ren π
  have hf := ren_injective π hk hp
  have hB : prodVal E B.nodes = prodVal E (A.nodes.map (Node.rename f)) := by
    funext τ
    rw [← prodVal_equiv E τ _ _ hlen (by simpa using hnodes)]
    exact (hP.map (val E τ)).prod_eq
  simp only [eval]
  rw [sumOver_perm _ hint, hB]
  have hr : prodVal E (A.nodes.map (Node.rename f)) =
      fun τ => prodVal E A.nodes (τ ∘ f) := funext fun τ => prodVal_rename E f τ A.nodes
  rw [hr, sumOver_map _ f hf A.int (fun x hx => by have := hdim x hx; simp only [labIdx] at this; simp only [labDim, this])]
  refine sumOver_ext _ _ (labelsOf A.nodes) _ (prodVal_reads E A.nodes) _ _ fun x hx hn => ?_
  rcases hfree x hx with h' | h'
  · exact absurd h' hn
  · have : f x = x := ren_of_not_mem π x h'
    simp [Function.comp, this]

theorem iso (E : Env) (s : ℂ) (A B : Net) (π : List (ℕ × ℕ)) (p : List ℕ)
    (h : isoOK E.labs A B π p = true) : Equal E s A s B := by
  refine ⟨?_, fun σ _ => by rw [iso_eval E A B π p h σ]⟩
  simp only [isoOK, Bool.and_eq_true, beq_iff_eq] at h
  exact List.Perm.of_eq h.1.1.1.1.2

/-- Equal networks under scalars proven equal. -/
theorem iso_scalar (E : Env) (s s' : ℂ) (A B : Net) (π : List (ℕ × ℕ)) (p : List ℕ)
    (h : isoOK E.labs A B π p = true) (hs : s = s') : Equal E s A s' B :=
  hs ▸ iso E s A B π p h

end ArchytasZX
