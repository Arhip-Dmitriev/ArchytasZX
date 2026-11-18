/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PairEval

/-!
# Two pair boxes equal at every copy count

`pair_family`: two diagrams of the pair bridge's form (`PairOK`) whose pair legs are pinned to the
same slots (`pinOK`) and whose templates satisfy `a • TA = b • TB` have expansions equal at every
`K` copies, as `c a ^ K` against `c b ^ K`: each side's network is its template at each copy's
slots times the same pair factor (`eval_pairNet`).
-/

namespace ArchytasZX

namespace Diagram

variable {D : Diagram} {p : DPair}

theorem tmpl_read (s : PairSpec D p) (E : Env) (ρ ρ' : ℕ → ℕ)
    (h : ∀ x < D.outs.length, ρ x = ρ' x) :
    ArchytasZX.eval E (D.tmpl p) ρ = ArchytasZX.eval E (D.tmpl p) ρ' := by
  unfold ArchytasZX.eval
  refine sumOver_ext _ _ _ _ (prodVal_reads E _) _ _ fun x hx hni => h x ?_
  obtain ⟨n', hn', hx'⟩ := List.mem_flatMap.1 hx
  obtain ⟨n, hn, rfl⟩ := List.mem_map.1 hn'
  have := tNode_lt s hn x hx'
  by_contra hc
  exact hni (List.mem_map.2 ⟨x - D.outs.length, List.mem_range.2 (by omega), by omega⟩)

theorem pow_prod (K : ℕ) (a : ℂ) (f : ℕ → ℂ) :
    a ^ K * ((List.range K).map f).prod = ((List.range K).map fun j => a * f j).prod := by
  rw [List.prod_map_mul]
  simp

/-- Two pair boxes over node boxes holding every node, their pair legs pinned to the same slots,
their templates equal as `a • TA = b • TB`: their expansions are equal at every `K`, the scalars
`c a ^ K` and `c b ^ K`. -/
theorem pair_family {DA DB : Diagram} {pA pB : DPair} {bA bB : DBox}
    (hA : DA.PairOK pA bA = true) (hB : DB.PairOK pB bB = true)
    (hm : DB.outs.length = DA.outs.length) (clA crA clB crB : List (ℕ × ℕ)) (sL sR : ℕ)
    (hlA : DA.pinOK pA pA.left clA sL = true) (hrA : DA.pinOK pA pA.right crA sR = true)
    (hlB : DB.pinOK pB pB.left clB sL = true) (hrB : DB.pinOK pB pB.right crB sR = true)
    (E : Env) (hE : ∀ x, labDim E x = E.dimv 0) {a b : ℂ}
    (ht : Equal E a (DA.tmpl pA) b (DB.tmpl pB)) (c : ℂ) (K : ℕ) :
    Equal E (c * a ^ K) ((DA.expandPair pA K).toNet (K * DA.outs.length))
      (c * b ^ K) ((DB.expandPair pB K).toNet (K * DA.outs.length)) := by
  have sA := PairOK.spec hA
  have sB := PairOK.spec hB
  have hE' : ∀ x y, labDim E x = labDim E y := fun x y => by rw [hE x, hE y]
  have eA := equal_expandPair hA E hE' K _ le_rfl
  have eB := equal_expandPair hB E hE' K (K * DA.outs.length) (by rw [hm])
  have hbA : ((DA.expandPair pA K).toNet (K * DA.outs.length)).bnd =
      List.range (K * DA.outs.length) := by
    simp [toNet, bnd_len sA]
  have hbB : ((DB.expandPair pB K).toNet (K * DA.outs.length)).bnd =
      List.range (K * DA.outs.length) := by
    simp [toNet, bnd_len sB, hm]
  refine ⟨by rw [hbA, hbB], fun σ hσ => ?_⟩
  have h1 := eA.2 σ hσ
  have h2 := eB.2 σ (by rw [hbB, ← hbA]; exact hσ)
  rw [one_mul, one_mul] at h1 h2
  rw [h1, h2, eval_pairNet hA E hE K clA crA sL sR hlA hrA σ,
    eval_pairNet hB E hE K clB crB sL sR hlB hrB σ, hm]
  have tpl : ∀ j ∈ List.range K, a * ArchytasZX.eval E (DA.tmpl pA)
      (σ ∘ shJ K DA.outs.length DA.wires.length j) =
      b * ArchytasZX.eval E (DB.tmpl pB) (σ ∘ shJ K DA.outs.length DB.wires.length j) := by
    intro j hj
    have hj' := List.mem_range.1 hj
    have hjm : (j + 1) * DA.outs.length ≤ K * DA.outs.length := Nat.mul_le_mul_right _ hj'
    rw [Nat.succ_mul] at hjm
    rw [ht.2 _ fun x hx => by
      have hx' : x < DA.outs.length := by simpa [tmpl] using hx
      simp only [Function.comp, shJ, hx', ite_true]
      rw [hE' x]
      exact hσ _ (by rw [hbA]; exact List.mem_range.2 (by omega))]
    congr 1
    refine tmpl_read sB E _ _ fun x hx => ?_
    rw [hm] at hx
    simp [shJ, hx]
  have e2 : ((List.range K).map fun j => a * ArchytasZX.eval E (DA.tmpl pA)
      (σ ∘ shJ K DA.outs.length DA.wires.length j)) =
      (List.range K).map fun j => b * ArchytasZX.eval E (DB.tmpl pB)
      (σ ∘ shJ K DA.outs.length DB.wires.length j) := List.map_congr_left tpl
  rw [show ∀ (u w : ℂ), c * a ^ K * (u * w) = c * (a ^ K * u) * w from fun u w => by ring,
    show ∀ (u w : ℂ), c * b ^ K * (u * w) = c * (b ^ K * u) * w from fun u w => by ring,
    pow_prod, pow_prod, e2]

end Diagram

end ArchytasZX
