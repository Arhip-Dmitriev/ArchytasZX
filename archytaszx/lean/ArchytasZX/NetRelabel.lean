/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.EqualLabs
import ArchytasZX.Iso

/-!
# Relabelled networks

`Net.relabel f` renames every label of a network, summed and boundary, by `f`.
`Equal.relabel` moves an equation between two networks to their relabellings by one injective
`f`, at the environment with a label table giving each renamed label its old dimension index
(`relabelOK`).
-/

namespace ArchytasZX

/-- Every label of `N` renamed by `f`. -/
def Net.relabel (f : ℕ → ℕ) (N : Net) : Net :=
  ⟨N.nodes.map (Node.rename f), N.int.map f, N.bnd.map f⟩

/-- Each label in `ls` renamed by `f` has, in `L`, its dimension index in `L₀`. -/
def relabelOK (L L₀ : List (ℕ × ℕ)) (f : ℕ → ℕ) (ls : List ℕ) : Bool :=
  ls.all fun x => (L.lookup (f x)).getD 0 == (L₀.lookup x).getD 0

theorem labDim_of_relabelOK {E : Env} {L : List (ℕ × ℕ)} {f : ℕ → ℕ} {ls : List ℕ}
    (h : relabelOK L E.labs f ls = true) :
    ∀ x ∈ ls, labDim { E with labs := L } (f x) = labDim E x := by
  intro x hx
  simp only [relabelOK, List.all_eq_true, beq_iff_eq] at h
  simp only [labDim, h x hx]

theorem sumOver_map2 (D D' : ℕ → ℕ) (f : ℕ → ℕ) (hf : Function.Injective f) (ls : List ℕ)
    (hd : ∀ x ∈ ls, D' (f x) = D x) (g : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver D' (ls.map f) (fun τ => g (τ ∘ f)) σ = sumOver D ls g (σ ∘ f) := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l ls ih =>
    simp only [List.map_cons, sumOver]
    rw [hd l List.mem_cons_self]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [ih (fun x hx => hd x (List.mem_cons_of_mem _ hx))]
    congr 1
    funext x
    simp only [Function.comp, Function.update]
    by_cases hx : x = l
    · subst hx; simp
    · have : f x ≠ f l := fun h' => hx (hf h')
      simp [hx, this]

theorem eval_relabel (E : Env) (L : List (ℕ × ℕ)) (f : ℕ → ℕ) (hf : Function.Injective f)
    (N : Net) (hd : ∀ x ∈ N.int, labDim { E with labs := L } (f x) = labDim E x)
    (σ : ℕ → ℕ) : eval { E with labs := L } (N.relabel f) σ = eval E N (σ ∘ f) := by
  have hp : prodVal { E with labs := L } (N.nodes.map (Node.rename f)) =
      fun τ => prodVal E N.nodes (τ ∘ f) := by
    funext τ
    rw [prodVal_rename]
    simp only [prodVal]
    congr 1
    refine List.map_congr_left fun n _ => ?_
    have := val_mapDim E id (fun _ => rfl) (τ ∘ f) n L
    rwa [Node.mapDim_id] at this
  simp only [eval, Net.relabel, hp]
  exact sumOver_map2 _ _ f hf _ hd _ σ

/-- An equation between two networks holds between their relabellings by `f`. -/
theorem Equal.relabel (E : Env) (L : List (ℕ × ℕ)) (f : ℕ → ℕ) (hf : Function.Injective f)
    {s s' : ℂ} {N M : Net} (hN : relabelOK L E.labs f (N.int ++ N.bnd) = true)
    (hM : relabelOK L E.labs f M.int = true) (h : Equal E s N s' M) :
    Equal { E with labs := L } s (N.relabel f) s' (M.relabel f) := by
  have dN := labDim_of_relabelOK hN
  have dM := labDim_of_relabelOK hM
  refine ⟨h.1.map f, fun σ hσ => ?_⟩
  rw [eval_relabel E L f hf N (fun x hx => dN x (List.mem_append_left _ hx)),
    eval_relabel E L f hf M dM]
  refine h.2 _ fun l hl => ?_
  have := hσ (f l) (List.mem_map_of_mem hl)
  rwa [dN l (List.mem_append_right _ hl)] at this

end ArchytasZX
