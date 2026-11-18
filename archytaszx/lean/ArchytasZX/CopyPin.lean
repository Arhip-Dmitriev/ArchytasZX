/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Ball
import ArchytasZX.Mixed
import ArchytasZX.FamK

/-!
# Copy-index phases as pinned boundary legs

`tZ_pin`: a phaseless Z spider with one more leg `w`, `w` through `F` to the value `u`, summed
over `w`, is the Z spider of phase `u k / d` over `√d`. `tX_pin`: a phaseless X spider with one
more input at the value `u` is the X spider of phase `-u k / d` over `√d`. `ex_mod`: the phase
`u k / d` reads `u` modulo `d`.

`Equal.pin`: equal networks agree at every boundary assignment in range, a boundary label pinned
to any value in range. `Diagram.evalD_eq_of_checks`: equal family networks of two checked boxed
diagrams with the same boundary data have equal digit sums at every slot digits in range.
-/

namespace ArchytasZX

open Finset

theorem conj_ex (θ : ℝ) : (starRingEnd ℂ) (ex θ) = ex (-θ) := by
  rw [← ex_inv]
  exact eq_inv_of_mul_eq_one_right (ex_mul_conj θ)

theorem ex_mod (d u k : ℕ) (hd : 0 < d) :
    ex (((u % d : ℕ) : ℝ) * k / d) = ex ((u : ℝ) * k / d) := by
  rw [show ((u % d : ℕ) : ℝ) * k = ((u % d * k : ℕ) : ℝ) by push_cast; ring, ← om_eq_ex,
    show (u : ℝ) * k = ((u * k : ℕ) : ℝ) by push_cast; ring, ← om_eq_ex, om_pow_mod d hd,
    om_pow_mod d hd (u * k), Nat.mod_mul_mod]

/-- A phaseless Z spider with one more leg, that leg through `F` to `u`, summed over the leg. -/
theorem tZ_pin (d : ℕ) (vs : List ℕ) (u : ℕ) :
    ∑ w ∈ range d, tZ d (fun _ => 0) (vs ++ [w]) * fou d w u =
      tZ d (fun k => (u : ℝ) * k / d) vs / rt d := by
  unfold tZ
  simp only [Finset.sum_mul, Finset.sum_div]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k hk => ?_
  by_cases hv : ∀ v ∈ vs, v = k
  · rw [if_pos hv]
    have h : ∀ w ∈ range d, (if ∀ v ∈ vs ++ [w], v = k then ex 0 else 0) * fou d w u =
        if k = w then fou d k u else 0 := fun w _ => by
      by_cases hw : k = w
      · subst hw
        rw [if_pos (fun v hv' => by
          rcases List.mem_append.1 hv' with h | h
          · exact hv v h
          · exact List.mem_singleton.1 h), if_pos rfl, ex_zero, one_mul]
      · rw [if_neg (fun h => hw (h w (List.mem_append_right _ (List.mem_singleton_self _))).symm),
          if_neg hw, zero_mul]
    rw [Finset.sum_congr rfl h, Finset.sum_ite_eq (range d) k, if_pos hk]
    unfold fou
    rw [om_eq_ex]
    congr 2
    push_cast
    ring
  · rw [if_neg hv, zero_div]
    refine Finset.sum_eq_zero fun w _ => ?_
    rw [if_neg (fun h => hv fun v hv' => h v (List.mem_append_left _ hv')), zero_mul]

/-- A phaseless X spider with one more input at `u`. -/
theorem tX_pin (d : ℕ) (os is : List ℕ) (u : ℕ) :
    tX d (fun _ => 0) os (is ++ [u]) = tX d (fun k => -((u : ℝ) * k / d)) os is / rt d := by
  unfold tX
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [List.map_append, List.prod_append]
  simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one]
  have hc : (starRingEnd ℂ) (fou d u k) = ex (-((u : ℝ) * k / d)) / rt d := by
    unfold fou
    rw [map_div₀, om_eq_ex, conj_ex, Complex.conj_ofReal]
    congr 2
    push_cast
    ring
  rw [hc, ex_zero]
  ring

/-- Equal networks agree at a boundary assignment in range pinning a boundary label to `v`. -/
theorem Equal.pin {E : Env} {s₁ s₂ : ℂ} {N₁ N₂ : Net} (h : Equal E s₁ N₁ s₂ N₂) (K v : ℕ)
    (σ : ℕ → ℕ) (hσ : InRange E N₁.bnd σ) (hK : σ K = v) :
    s₁ * eval E N₁ (Function.update σ K v) = s₂ * eval E N₂ (Function.update σ K v) := by
  rw [← hK, Function.update_eq_self]
  exact h.2 σ hσ

namespace Diagram

/-- Equal family networks of two checked boxed diagrams with the same boundary data have equal
digit sums at every slot digits in range. -/
theorem evalD_eq_of_checks {F : Fam} {D₁ D₂ : Diagram} {o₁ o₂ : ℕ} (ks : ℕ → ℕ) (s₁ s₂ : ℂ)
    (hc₁ : D₁.FamOK F o₁) (hc₂ : D₂.FamOK F o₂) (hbd : D₁.bndData = D₂.bndData)
    (hpos : ∀ i, 0 < F.dims i)
    (hEq : Equal (F.env ks) s₁ (D₁.toNetB o₁) s₂ (D₂.toNetB o₂)) (ρ : ℕ → List ℕ → ℕ)
    (hin : ∀ i (hi : i < D₁.bnd.length), ∀ t ∈ cidx ks (D₁.legChain D₁.bnd[i]),
      ρ i t < (F.env ks).dimv (D₁.portDim D₁.bnd[i])) :
    s₁ * D₁.evalD (F.env ks) ks ρ = s₂ * D₂.evalD (F.env ks) ks ρ := by
  have hp : ∀ i, 0 < (F.env ks).dimv i := fun i => Fam.env_dimv_pos F ks i (hpos _)
  obtain ⟨hlen, hd⟩ := bndData_at hbd
  obtain ⟨w₁, b₁, n₁, l₁, s₁'⟩ := hc₁
  obtain ⟨w₂, b₂, n₂, l₂, -⟩ := hc₂
  have hin₂ : ∀ i (hi : i < D₂.bnd.length), ∀ t ∈ cidx ks (D₂.legChain D₂.bnd[i]),
      ρ i t < (F.env ks).dimv (D₂.portDim D₂.bnd[i]) := fun i hi t ht => by
    have h₁ : i < D₁.bnd.length := hlen ▸ hi
    obtain ⟨e1, e2⟩ := hd i h₁ hi
    rw [← e2] at ht
    rw [← e1]
    exact hin i h₁ t ht
  have e₁ := evalD_net (WF_of_WFb w₁) (F.env ks) o₁ b₁ (fun n hn => NodeS_of (n₁ n hn) hp)
    (fun j hj => labDim_of F ks _ _ _ (l₁ j hj).1 (l₁ j hj).2) ρ hin
  have e₂ := evalD_net (WF_of_WFb w₂) (F.env ks) o₂ b₂ (fun n hn => NodeS_of (n₂ n hn) hp)
    (fun j hj => labDim_of F ks _ _ _ (l₂ j hj).1 (l₂ j hj).2) ρ hin₂
  change D₁.evalD (F.env ks) (F.env ks).mult ρ = _ at e₁
  change D₂.evalD (F.env ks) (F.env ks).mult ρ = _ at e₂
  change s₁ * D₁.evalD (F.env ks) (F.env ks).mult ρ = s₂ * D₂.evalD (F.env ks) (F.env ks).mult ρ
  rw [e₁, e₂, ← packB_congr hbd]
  refine hEq.2 _ fun l hl => ?_
  have hi : l < D₁.bnd.length := by simpa [toNetB] using hl
  rw [labDim_of F ks _ _ _ (s₁' l hi).1 (s₁' l hi).2]
  simp only [packB, if_pos hi, List.getD_eq_getElem _ _ hi]
  exact packC_lt _ _ _ _ fun u hu => hin l hi u hu

end Diagram

end ArchytasZX
