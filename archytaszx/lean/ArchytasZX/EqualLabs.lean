/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Assume
import ArchytasZX.Diagram

/-!
# Label tables of one dimension

`Equal` at an environment and at the same environment with another label table are the same
statement when every label has the same dimension under both tables (`Equal.labs_iff`).
`DEqual_of_labs` gives `DEqual` of two diagrams from `Equal` of their networks at an environment
whose label table differs from the diagrams' check table only in that way.
-/

namespace ArchytasZX

theorem Node.mapDim_id (n : Node) : n.mapDim id = n := by
  obtain ⟨k, o, i, d, p⟩ := n
  cases k <;> rfl

theorem eval_labs (E : Env) (L : List (ℕ × ℕ))
    (hL : ∀ l, labDim { E with labs := L } l = labDim E l) (N : Net) (σ : ℕ → ℕ) :
    eval { E with labs := L } N σ = eval E N σ := by
  have hD : labDim { E with labs := L } = labDim E := funext hL
  simp only [eval, hD]
  refine sumOver_congr _ _ _ _ _ fun τ _ _ => ?_
  simp only [prodVal]
  congr 1
  refine List.map_congr_left fun n _ => ?_
  have := val_mapDim E id (fun _ => rfl) τ n L
  rwa [Node.mapDim_id] at this

/-- `Equal` does not depend on the label table beyond each label's dimension. -/
theorem Equal.labs_iff (E : Env) (L : List (ℕ × ℕ))
    (hL : ∀ l, labDim { E with labs := L } l = labDim E l) {s s' : ℂ} {N M : Net} :
    Equal { E with labs := L } s N s' M ↔ Equal E s N s' M := by
  have hr : ∀ ls σ, InRange { E with labs := L } ls σ ↔ InRange E ls σ := fun ls σ => by
    simp only [InRange, hL]
  simp only [Equal, eval_labs E L hL, hr]

/-- `DEqual` over the check table `L` from `Equal` of the networks at `E`, every label of one
dimension under `E.labs` and `L`. -/
theorem DEqual_of_labs {E : Env} {L : List (ℕ × ℕ)} {s₁ s₂ : ℂ} {D₁ D₂ : Diagram} {o₁ o₂ : ℕ}
    (hL : ∀ l, labDim { E with labs := L } l = labDim E l) (h₁ : D₁.check L o₁ = true)
    (h₂ : D₂.check L o₂ = true) (h : Equal E s₁ (D₁.toNet o₁) s₂ (D₂.toNet o₂)) :
    DEqual { E with labs := L } s₁ D₁ s₂ D₂ :=
  (DEqual_iff rfl h₁ h₂).2 ((Equal.labs_iff E L hL).2 h)

/-- Two diagrams whose boundary dimension values differ are not equal. -/
theorem not_DEqual_of_dims {E : Env} {s₁ s₂ : ℂ} {D₁ D₂ : Diagram}
    (h : D₁.bdims.map E.dimv ≠ D₂.bdims.map E.dimv) : ¬ DEqual E s₁ D₁ s₂ D₂ :=
  fun hd => h (congrArg _ hd.1)

end ArchytasZX
