/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainPair
import ArchytasZX.FamEval
import ArchytasZX.PairFam

/-!
# Chain and pair families as diagram equalities

`DEqual_of_chainNet` and `DEqual_of_pairNet`: `Equal` of the networks of two closed chain (or
pair box) expansions at one copy count as `DEqual` of the expansions, for an environment of one
dimension and sides whose every leg has dimension index `0` (`legsZero`): the expansions'
boundary dimension lists are both zeros of one length, and each expansion's entry is its
network's (`eval_toNet_dim`, ports located by `ChainFamOK.portsX` and `PairSpec.ports_ok`).
-/

namespace ArchytasZX

namespace Diagram

/-- Every leg of every node has dimension index `0`. -/
def legsZero (D : Diagram) : Prop := ∀ n ∈ D.nodes, ∀ x ∈ n.outs ++ n.ins, x = 0

instance (D : Diagram) : Decidable D.legsZero := by unfold legsZero; infer_instance

theorem mem_growSide {g : Port → Bool} {k x : ℕ} {out : Bool} {ds : List ℕ} {v : ℕ}
    (h : v ∈ growSide g k x out ds) : v ∈ ds := by
  unfold growSide at h
  obtain ⟨⟨d, i⟩, hp, hv⟩ := List.mem_flatMap.1 h
  have hd : d ∈ ds := List.fst_mem_of_mem_zipIdx hp
  by_cases hg : g ⟨x, out, i⟩ = true
  · simp only [hg, ↓reduceIte] at hv; rw [(List.mem_replicate.1 hv).2]; exact hd
  · simp only [hg, Bool.false_eq_true, ↓reduceIte] at hv; rw [List.mem_singleton.1 hv]; exact hd

theorem legsZero_expandChain {D : Diagram} (h : D.legsZero) (c : DChain) (k : ℕ) :
    (D.expandChain c k).legsZero := by
  intro n hn x hx
  simp only [expandChain, copyNodes, List.mem_flatMap] at hn
  obtain ⟨m, hm, hn⟩ := hn
  split at hn
  · obtain ⟨j, _, rfl⟩ := List.mem_map.1 hn
    exact h m hm x hx
  · rw [List.mem_singleton.1 hn] at hx
    rcases List.mem_append.1 hx with hx | hx
    · exact h m hm x (List.mem_append_left _ (mem_growSide hx))
    · exact h m hm x (List.mem_append_right _ (mem_growSide hx))

theorem portDim_of_legsZero {D : Diagram} (h : D.legsZero) (p : Port) : D.portDim p = 0 := by
  unfold portDim
  split
  · rename_i n hf
    have hn := List.mem_of_find?_eq_some hf
    split
    · unfold List.getD
      cases hg : n.outs[p.pos]? with
      | none => rfl
      | some v => exact h n hn v (List.mem_append_left _ (List.mem_of_getElem? hg))
    · unfold List.getD
      cases hg : n.ins[p.pos]? with
      | none => rfl
      | some v => exact h n hn v (List.mem_append_right _ (List.mem_of_getElem? hg))
  · rfl

theorem bdims_of_legsZero {D : Diagram} (h : D.legsZero) :
    D.bdims = List.replicate D.bnd.length 0 := by
  unfold bdims
  exact List.eq_replicate_iff.2 ⟨by simp, fun x hx => by
    obtain ⟨p, _, rfl⟩ := List.mem_map.1 hx; exact portDim_of_legsZero h p⟩

/-- `DEqual` of two diagrams of zero legs from `Equal` of their networks at one offset, at an
environment of one dimension. -/
theorem DEqual_of_toNet {X Y : Diagram} {E : Env} (hE : ∀ x y, labDim E x = E.dimv y)
    (zX : X.legsZero) (zY : Y.legsZero) (hl : X.bnd.length = Y.bnd.length) {o : ℕ}
    (ho : X.bnd.length ≤ o) (pX : X.portsOK) (pY : Y.portsOK) {s₁ s₂ : ℂ}
    (h : Equal E s₁ (X.toNet o) s₂ (Y.toNet o)) : DEqual E s₁ X s₂ Y := by
  refine ⟨by rw [bdims_of_legsZero zX, bdims_of_legsZero zY, hl], fun σ hσ => ?_⟩
  have hin : ArchytasZX.InRange E (X.toNet o).bnd σ := fun l hl' => by
    rw [hE l (X.bdims.getD l 0)]
    exact hσ l (List.mem_range.1 hl')
  have e := h.2 σ hin
  rwa [eval_toNet_dim _ E o ho pX (fun j _ => hE _ _) σ,
    eval_toNet_dim _ E o (hl ▸ ho) pY (fun j _ => hE _ _) σ] at e

/-- `DEqual` of two closed chain expansions from `Equal` of their networks. -/
theorem DEqual_of_chainNet {DA DB : Diagram} {cA cB : DChain} {W W' : ℕ}
    (hA : ChainFamOK DA cA W) (hB : ChainFamOK DB cB W') (zA : DA.legsZero) (zB : DB.legsZero)
    (hm : DB.bnd.length = DA.bnd.length) {E : Env} (hE : ∀ x y, labDim E x = E.dimv y)
    {o : ℕ} (ho : DA.bnd.length ≤ o) (k : ℕ) {s₁ s₂ : ℂ}
    (h : Equal E s₁ ((DA.expandChain cA k).toNet o) s₂ ((DB.expandChain cB k).toNet o)) :
    DEqual E s₁ (DA.expandChain cA k) s₂ (DB.expandChain cB k) :=
  DEqual_of_toNet hE (legsZero_expandChain zA cA k) (legsZero_expandChain zB cB k)
    (by rw [hA.closed.bnd_expand k, hB.closed.bnd_expand k, hm])
    (by rw [hA.closed.bnd_expand k]; exact ho) (hA.portsX k) (hB.portsX k) h

theorem zero_of_mem_growC {c : Port → ℕ} {x : ℕ} {out : Bool} {ds : List ℕ} {v : ℕ}
    (hz : ∀ y ∈ ds, y = 0) (h : v ∈ growC c x out ds) : v = 0 := by
  unfold growC at h
  obtain ⟨i, -, hv⟩ := List.mem_flatMap.1 h
  rw [(List.mem_replicate.1 hv).2]
  unfold List.getD
  cases hg : ds[i]? with
  | none => rfl
  | some y => exact hz y (List.mem_of_getElem? hg)

theorem legsZero_expandPair {D : Diagram} (h : D.legsZero) (p : DPair)
    (hp : ∀ x ∈ p.node.outs ++ p.node.ins, x = 0) (K : ℕ) : (D.expandPair p K).legsZero := by
  intro n hn x hx
  simp only [expandPair, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range] at hn
  rcases hn with ⟨j, -, m, hm, rfl⟩ | ⟨j, -, k, -, rfl⟩
  · have hz : ∀ y ∈ m.outs ++ m.ins, y = 0 := h m hm
    rcases List.mem_append.1 hx with hx | hx
    · exact zero_of_mem_growC (fun y hy => hz y (List.mem_append_left _ hy)) hx
    · exact zero_of_mem_growC (fun y hy => hz y (List.mem_append_right _ hy)) hx
  · exact hp x hx

/-- `DEqual` of two pair box expansions from `Equal` of their networks. -/
theorem DEqual_of_pairNet {DA DB : Diagram} {pA pB : DPair} {bA bB : DBox}
    (hA : DA.PairOK pA bA = true) (hB : DB.PairOK pB bB = true) (zA : DA.legsZero)
    (zB : DB.legsZero) (zpA : ∀ x ∈ pA.node.outs ++ pA.node.ins, x = 0)
    (zpB : ∀ x ∈ pB.node.outs ++ pB.node.ins, x = 0)
    (hm : DB.outs.length = DA.outs.length) {E : Env} (hE : ∀ x y, labDim E x = E.dimv y)
    (K : ℕ) {s₁ s₂ : ℂ}
    (h : Equal E s₁ ((DA.expandPair pA K).toNet (K * DA.outs.length)) s₂
      ((DB.expandPair pB K).toNet (K * DA.outs.length))) :
    DEqual E s₁ (DA.expandPair pA K) s₂ (DB.expandPair pB K) :=
  DEqual_of_toNet hE (legsZero_expandPair zA pA zpA K) (legsZero_expandPair zB pB zpB K)
    (by rw [bnd_len (PairOK.spec hA) K, bnd_len (PairOK.spec hB) K, hm])
    (by rw [bnd_len (PairOK.spec hA) K]) (ports_ok (PairOK.spec hA) K)
    (ports_ok (PairOK.spec hB) K) h

end Diagram

end ArchytasZX
