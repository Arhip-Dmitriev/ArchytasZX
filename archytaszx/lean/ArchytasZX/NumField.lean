/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Weights

/-!
# One real root of an integer polynomial, and networks equal up to spider coefficients

`pev cs` is the polynomial with integer coefficients `cs` (constant term first) on ℝ, `pdv cs`
its derivative, `pevC cs` the same polynomial on ℂ. `root_unique` gives exactly one root in an
interval where the polynomial changes sign and its derivative is positive. `netCoef` equates two
networks whose node lists differ only in the coefficient tables of Z and X spiders.
-/

namespace ArchytasZX

open Complex Finset

/-- `Σ cs_k x^k` on ℝ, by Horner's rule. -/
noncomputable def pev : List ℤ → ℝ → ℝ
  | [], _ => 0
  | c :: cs, x => c + x * pev cs x

/-- The derivative of `pev cs`. -/
noncomputable def pdv : List ℤ → ℝ → ℝ
  | [], _ => 0
  | _ :: cs, x => pev cs x + x * pdv cs x

/-- `Σ cs_k z^k` on ℂ, by Horner's rule. -/
noncomputable def pevC : List ℤ → ℂ → ℂ
  | [], _ => 0
  | c :: cs, z => c + z * pevC cs z

theorem hasDerivAt_pev (cs : List ℤ) (x : ℝ) : HasDerivAt (pev cs) (pdv cs x) x := by
  induction cs generalizing x with
  | nil => simpa [pev, pdv] using hasDerivAt_const x (0 : ℝ)
  | cons c cs ih =>
    have h := ((hasDerivAt_id x).mul (ih x)).const_add (c : ℝ)
    have e : pev (c :: cs) = fun y => (c : ℝ) + ((id : ℝ → ℝ) * pev cs) y := by
      funext y; simp [pev]
    rw [e]
    convert h using 1
    simp [pdv]

theorem pevC_ofReal (cs : List ℤ) (x : ℝ) : pevC cs (x : ℂ) = (pev cs x : ℂ) := by
  induction cs with
  | nil => simp [pevC, pev]
  | cons c cs ih => simp [pevC, pev, ih]

theorem pevC_root (cs : List ℤ) (x : ℝ) (h : pev cs x = 0) : pevC cs (x : ℂ) = 0 := by
  rw [pevC_ofReal, h, Complex.ofReal_zero]

/-- Exactly one root of `pev cs` in `(lo, hi)`: a sign change from negative to positive and a
positive derivative there. -/
theorem root_unique (cs : List ℤ) (lo hi : ℝ) (hlh : lo < hi) (hlo : pev cs lo < 0) (hhi : 0 < pev cs hi)
    (hd : ∀ x, lo < x → x < hi → 0 < pdv cs x) :
    ∃! x, lo < x ∧ x < hi ∧ pev cs x = 0 := by
  have hc : Continuous (pev cs) := continuous_iff_continuousAt.2 fun x =>
    (hasDerivAt_pev cs x).continuousAt
  have hm : StrictMonoOn (pev cs) (Set.Icc lo hi) := by
    refine strictMonoOn_of_deriv_pos (convex_Icc _ _) hc.continuousOn fun x hx => ?_
    rw [interior_Icc] at hx
    rw [(hasDerivAt_pev cs x).deriv]
    exact hd x hx.1 hx.2
  obtain ⟨x, hx, hx0⟩ := intermediate_value_Ioo hlh.le hc.continuousOn ⟨hlo, hhi⟩
  refine ⟨x, ⟨hx.1, hx.2, hx0⟩, fun y ⟨hy1, hy2, hy0⟩ => ?_⟩
  exact hm.injOn ⟨hy1.le, hy2.le⟩ ⟨hx.1.le, hx.2.le⟩ (hy0.trans hx0.symm)

/-- Two Z or weighted Z spiders on the same legs with equal coefficients have equal entries. -/
theorem valCoefZ (E : Env) (n m : Node) (hn : isZc n = true) (hm : isZc m = true)
    (ho : n.outs = m.outs) (hi : n.ins = m.ins) (hd : n.dim = m.dim)
    (hc : ∀ k < E.dimv n.dim, zc E n k = zc E m k) : ∀ σ, val E σ n = val E σ m := by
  intro σ
  rw [val_zc E σ n hn, val_zc E σ m hm, ho, hi, hd]
  unfold tZw
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [hc k (hd ▸ Finset.mem_range.1 hk)]

/-- Two X or weighted X spiders on the same legs with equal coefficients have equal entries. -/
theorem valCoefX (E : Env) (n m : Node) (hn : isXc n = true) (hm : isXc m = true)
    (ho : n.outs = m.outs) (hi : n.ins = m.ins) (hd : n.dim = m.dim)
    (hc : ∀ k < E.dimv n.dim, xc E n k = xc E m k) : ∀ σ, val E σ n = val E σ m := by
  intro σ
  rw [val_xc E σ n hn, val_xc E σ m hm, ho, hi, hd]
  unfold tXw
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [hc k (hd ▸ Finset.mem_range.1 hk)]

/-- Networks whose nodes have pairwise equal entries are equal. -/
theorem netCoef (E : Env) (A B : Net) (hi : A.int = B.int) (hb : A.bnd = B.bnd)
    (h : List.Forall₂ (fun n m => ∀ σ, val E σ n = val E σ m) A.nodes B.nodes) :
    Equal E 1 A 1 B := by
  have hl : ∀ (ns ms : List Node), List.Forall₂ (fun n m => ∀ σ, val E σ n = val E σ m) ns ms →
      ∀ σ, prodVal E ns σ = prodVal E ms σ := by
    intro ns ms h' σ
    unfold prodVal
    induction h' with
    | nil => rfl
    | cons hab _ ih => simp only [List.map_cons, List.prod_cons, hab σ, ih]
  have hp := hl _ _ h
  refine ⟨hb ▸ List.Perm.refl _, fun σ _ => ?_⟩
  simp only [one_mul, eval, hi]
  exact congrArg (fun f => sumOver (labDim E) B.int f σ) (funext hp)

end ArchytasZX
