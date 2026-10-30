/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Bundle

/-!
# Dimensions assumed equal

A proof over a network whose dimension indices were merged by `f` gives the equation over the
unmerged network, provided each merged pair has equal values.
-/

namespace ArchytasZX

/-- `x` through the pairs `ps`, itself when it has no pair. -/
def dimMap (ps : List (ℕ × ℕ)) (x : ℕ) : ℕ := (ps.lookup x).getD x

/-- A node's dimension indices through `f`. -/
def Node.mapDim (f : ℕ → ℕ) (n : Node) : Node :=
  match n.kind with
  | .B | .S => { n with dim := f n.dim, ph := f n.ph }
  | _ => { n with dim := f n.dim }

/-- Every node's dimension indices through `f`. -/
def Net.mapDim (f : ℕ → ℕ) (N : Net) : Net := ⟨N.nodes.map (Node.mapDim f), N.int, N.bnd⟩

/-- Each label's dimension index through `f`. -/
def mapLabs (f : ℕ → ℕ) (labs : List (ℕ × ℕ)) : List (ℕ × ℕ) := labs.map fun p => (p.1, f p.2)

theorem lookup_mapLabs (f : ℕ → ℕ) (labs : List (ℕ × ℕ)) (l : ℕ) :
    (mapLabs f labs).lookup l = (labs.lookup l).map f := by
  induction labs with
  | nil => rfl
  | cons p labs ih =>
    obtain ⟨a, b⟩ := p
    by_cases h : l = a
    · subst h; simp [mapLabs, List.lookup]
    · have : (l == a) = false := by simpa using h
      simp only [mapLabs, List.map_cons, List.lookup, this] at ih ⊢
      exact ih

theorem val_mapDim (E : Env) (f : ℕ → ℕ) (hf : ∀ x, E.dimv (f x) = E.dimv x) (σ : ℕ → ℕ)
    (n : Node) (labs : List (ℕ × ℕ)) :
    val { E with labs := labs } σ (n.mapDim f) = val E σ n := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  by_cases hk : kind ≠ .B ∧ kind ≠ .S
  · have e : Node.mapDim f ⟨kind, os, is, dim, ph⟩ = ⟨kind, os, is, f dim, ph⟩ := by
      cases kind <;> simp_all [Node.mapDim]
    rw [e, val_tensor _ σ kind hk, val_tensor E σ kind hk]
    exact (congrArg (fun d => tensor _ kind d ph _ _) (hf dim)).trans
      (tensor_env { E with labs := labs } E rfl rfl kind (fun _ _ => rfl) _ _ _ _)
  · have hbs : kind = .B ∨ kind = .S := by
      by_contra h'; exact hk ⟨fun e => h' (Or.inl e), fun e => h' (Or.inr e)⟩
    rcases hbs with rfl | rfl <;>
      rcases os with _ | ⟨a, _ | ⟨b, _ | _⟩⟩ <;> rcases is with _ | ⟨c, _ | ⟨e, _ | _⟩⟩ <;>
      simp [val, Node.mapDim, hf]

theorem Equal.of_mapDim (E : Env) (ps : List (ℕ × ℕ)) (hps : ∀ p ∈ ps, E.dimv p.2 = E.dimv p.1)
    {s s' : ℂ} {N M : Net}
    (h : Equal { E with labs := mapLabs (dimMap ps) E.labs } s (N.mapDim (dimMap ps)) s'
      (M.mapDim (dimMap ps))) : Equal E s N s' M := by
  have hf : ∀ x, E.dimv (dimMap ps x) = E.dimv x := by
    intro x
    unfold dimMap
    cases hl : ps.lookup x with
    | none => rfl
    | some y => simpa using (hps (x, y) (lookup_mem hl))
  have hlab : ∀ l, labDim { E with labs := mapLabs (dimMap ps) E.labs } l = labDim E l := by
    intro l
    simp only [labDim, lookup_mapLabs]
    cases E.labs.lookup l with
    | none => simp
    | some i => simp [hf]
  have hD : labDim { E with labs := mapLabs (dimMap ps) E.labs } = labDim E := funext hlab
  have hv : ∀ (X : Net) (σ : ℕ → ℕ), eval { E with labs := mapLabs (dimMap ps) E.labs }
      (X.mapDim (dimMap ps)) σ = eval E X σ := by
    intro X σ
    simp only [eval, Net.mapDim, hD]
    refine sumOver_congr _ _ _ _ _ fun τ _ _ => ?_
    simp only [prodVal, List.map_map]
    congr 1
    exact List.map_congr_left fun n _ => val_mapDim E _ hf τ n _
  refine ⟨h.1, fun σ hσ => ?_⟩
  have := h.2 σ (fun l hl => by rw [hlab]; exact hσ l hl)
  rwa [hv, hv] at this

/-- `Equal.of_mapDim` with the merged environment and networks named. -/
theorem Equal.of_mapDim' (E E' : Env) (ps : List (ℕ × ℕ)) (hps : ∀ p ∈ ps, E.dimv p.2 = E.dimv p.1)
    (N M N' M' : Net) (hN : N.mapDim (dimMap ps) = N') (hM : M.mapDim (dimMap ps) = M')
    (hE : { E with labs := mapLabs (dimMap ps) E.labs } = E') {s s' : ℂ}
    (h : Equal E' s N' s' M') : Equal E s N s' M := by
  subst hN hM hE
  exact Equal.of_mapDim E ps hps h

end ArchytasZX
