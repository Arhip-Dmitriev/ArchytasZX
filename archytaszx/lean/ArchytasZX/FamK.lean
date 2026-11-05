/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.OrderK

/-!
# Family theorems on the expansions in key order

`DEqual_expandK_of_checks`: equal family networks of two checked boxed diagrams with the same
boundary data and key data are equal expansions in key order, at every multiplicity.
-/

namespace ArchytasZX

namespace Diagram

variable {D : Diagram}

/-- The digit sum at slot digits in range is the family's network at their packing. -/
theorem evalD_net (hW : WF D) (E : Env) (off : ℕ) (hoff : D.bnd.length ≤ off)
    (hN : ∀ n ∈ D.nodes, D.NodeS E n)
    (hlab : ∀ j (hj : j < D.wires.length), labDim E (off + j) =
      E.dimv (D.portDim D.wires[j].1) ^ size E.mult (D.legChain D.wires[j].1))
    (ρ : ℕ → List ℕ → ℕ)
    (hin : ∀ i (hi : i < D.bnd.length), ∀ t ∈ cidx E.mult (D.legChain D.bnd[i]),
      ρ i t < E.dimv (D.portDim D.bnd[i])) :
    D.evalD E E.mult ρ = ArchytasZX.eval E (D.toNetB off) (D.packB E ρ) := by
  have hpd : ∀ i (hi : i < D.bnd.length), 0 < E.dimv (D.portDim D.bnd[i]) := fun i hi => by
    obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 (bnd_legs hW (List.getElem_mem hi))
    exact (hN n hn).posd _ hpn
  have hg : ∀ i (hi : i < D.bnd.length), D.bnd.getD i ⟨0, true, 0⟩ = D.bnd[i] := fun i hi => by
    rw [List.getD_eq_getElem _ _ hi]
  rw [eval_toNetB_evalD hW E off hoff hN hlab _ fun i hi => ?_]
  · refine evalD_slots hW E E.mult _ _ fun i hi t ht => ?_
    simp only [slotDig, packB, if_pos hi, hg i hi]
    exact (digit_packC _ _ (hpd i hi) _ _ (fun u hu => hin i hi u hu) t ht).symm
  · simp only [packB, if_pos hi, hg i hi]
    exact packC_lt _ _ _ _ fun u hu => hin i hi u hu

theorem bndData_at {D₁ D₂ : Diagram} (hbd : D₁.bndData = D₂.bndData) :
    D₁.bnd.length = D₂.bnd.length ∧ ∀ i (h₁ : i < D₁.bnd.length) (h₂ : i < D₂.bnd.length),
      D₁.portDim D₁.bnd[i] = D₂.portDim D₂.bnd[i] ∧ D₁.legChain D₁.bnd[i] = D₂.legChain D₂.bnd[i] := by
  refine ⟨by simpa [bndData] using congrArg List.length hbd, fun i h₁ h₂ => ?_⟩
  have := congrArg (·[i]?) hbd
  simp only [bndData, List.getElem?_map, List.getElem?_eq_getElem h₁,
    List.getElem?_eq_getElem h₂, Option.map_some, Option.some.injEq, Prod.mk.injEq] at this
  exact this

theorem SDom_congr {D₁ D₂ : Diagram} (hbd : D₁.bndData = D₂.bndData) (ks : ℕ → ℕ) :
    D₁.SDom ks = D₂.SDom ks := by
  obtain ⟨hl, hd⟩ := bndData_at hbd
  unfold SDom
  rw [hl]
  refine List.flatMap_congr fun i hi => ?_
  have hi₂ := List.mem_range.1 hi
  rw [List.getD_eq_getElem _ _ (hl ▸ hi₂), List.getD_eq_getElem _ _ hi₂, (hd i (hl ▸ hi₂) hi₂).2]

theorem SL_congr {D₁ D₂ : Diagram} (hbd : D₁.bndData = D₂.bndData) (hks : D₁.keySig = D₂.keySig)
    (ks : ℕ → ℕ) : D₁.SL ks = D₂.SL ks := by
  unfold SL
  rw [SDom_congr hbd, hks]

theorem packB_congr {D₁ D₂ : Diagram} (hbd : D₁.bndData = D₂.bndData) (E : Env) (ρ : ℕ → List ℕ → ℕ) :
    D₁.packB E ρ = D₂.packB E ρ := by
  obtain ⟨hl, hd⟩ := bndData_at hbd
  funext i
  unfold packB
  by_cases hi : i < D₁.bnd.length
  · have hi₂ : i < D₂.bnd.length := hl ▸ hi
    rw [if_pos hi, if_pos hi₂, List.getD_eq_getElem _ _ hi, List.getD_eq_getElem _ _ hi₂,
      (hd i hi hi₂).1, (hd i hi hi₂).2]
  · rw [if_neg hi, if_neg (hl ▸ hi)]

/-- Equal family networks give equal expansions in key order when the boundary data and key data
agree. -/
theorem DEqual_expandK_of_net {D₁ D₂ : Diagram} (hW₁ : WF D₁) (hW₂ : WF D₂) (E : Env) (o₁ o₂ : ℕ)
    (s₁ s₂ : ℂ) (ho₁ : D₁.bnd.length ≤ o₁) (ho₂ : D₂.bnd.length ≤ o₂)
    (hN₁ : ∀ n ∈ D₁.nodes, D₁.NodeS E n) (hN₂ : ∀ n ∈ D₂.nodes, D₂.NodeS E n)
    (hl₁ : ∀ j (hj : j < D₁.wires.length), labDim E (o₁ + j) =
      E.dimv (D₁.portDim D₁.wires[j].1) ^ size E.mult (D₁.legChain D₁.wires[j].1))
    (hl₂ : ∀ j (hj : j < D₂.wires.length), labDim E (o₂ + j) =
      E.dimv (D₂.portDim D₂.wires[j].1) ^ size E.mult (D₂.legChain D₂.wires[j].1))
    (hb : ∀ i (hi : i < D₁.bnd.length), labDim E i =
      E.dimv (D₁.portDim D₁.bnd[i]) ^ size E.mult (D₁.legChain D₁.bnd[i]))
    (hbd : D₁.bndData = D₂.bndData) (hks : D₁.keySig = D₂.keySig)
    (hEq : Equal E s₁ (D₁.toNetB o₁) s₂ (D₂.toNetB o₂)) :
    DEqual E s₁ (D₁.expandK E.mult) s₂ (D₂.expandK E.mult) := by
  obtain ⟨hlen, hd⟩ := bndData_at hbd
  have hSL := SL_congr hbd hks E.mult
  have hexp : (D₁.expandK E.mult).bdims = (D₂.expandK E.mult).bdims := by
    rw [bdims_expandK hW₁, bdims_expandK hW₂, hSL]
    refine List.map_congr_left fun x hx => ?_
    have hx' := (SL_perm E.mult).mem_iff.1 (hSL ▸ hx : x ∈ D₁.SL E.mult)
    obtain ⟨hi, -⟩ := (mem_SDom D₁ E.mult x).1 hx'
    rw [List.getD_eq_getElem _ _ hi, List.getD_eq_getElem _ _ (hlen ▸ hi), (hd _ hi (hlen ▸ hi)).1]
  refine ⟨hexp, fun σ'' hσ => ?_⟩
  have hσ₂ : (D₂.expandK E.mult).InRange E σ'' := by
    intro i hi
    have hl : (D₂.expandK E.mult).bnd.length = (D₁.expandK E.mult).bnd.length := by
      simpa [bdims] using (congrArg List.length hexp).symm
    rw [← hexp]; exact hσ i (hl ▸ hi)
  have hρ : D₂.readK E.mult σ'' = D₁.readK E.mult σ'' := by
    funext i t; simp only [readK, hSL]
  set ρ := D₁.readK E.mult σ''
  rw [eval_expandK_evalD hW₁ E σ'' hσ, eval_expandK_evalD hW₂ E σ'' hσ₂, hρ]
  by_cases hin : ∀ i (hi : i < D₁.bnd.length), ∀ t ∈ cidx E.mult (D₁.legChain D₁.bnd[i]),
      ρ i t < E.dimv (D₁.portDim D₁.bnd[i])
  · have hin₂ : ∀ i (hi : i < D₂.bnd.length), ∀ t ∈ cidx E.mult (D₂.legChain D₂.bnd[i]),
        ρ i t < E.dimv (D₂.portDim D₂.bnd[i]) := fun i hi t ht => by
      have h₁ : i < D₁.bnd.length := hlen ▸ hi
      obtain ⟨e1, e2⟩ := hd i h₁ hi
      rw [← e2] at ht
      rw [← e1]
      exact hin i h₁ t ht
    rw [evalD_net hW₁ E o₁ ho₁ hN₁ hl₁ ρ hin, evalD_net hW₂ E o₂ ho₂ hN₂ hl₂ ρ hin₂,
      ← packB_congr hbd]
    refine hEq.2 _ fun l hl => ?_
    have hi : l < D₁.bnd.length := by simpa [toNetB] using hl
    rw [hb l hi]
    simp only [packB, if_pos hi, List.getD_eq_getElem _ _ hi]
    exact packC_lt _ _ _ _ fun u hu => hin l hi u hu
  · push_neg at hin
    obtain ⟨i, hi, t, ht, hout⟩ := hin
    have hi₂ : i < D₂.bnd.length := hlen ▸ hi
    obtain ⟨e1, e2⟩ := hd i hi hi₂
    rw [evalD_out E E.mult ρ i hi hW₁ t ht hout,
      evalD_out E E.mult ρ i hi₂ hW₂ t (by rw [← e2]; exact ht) (by rw [← e1]; exact hout)]
    simp

/-- Equal family networks of two checked boxed diagrams with the same boundary data and key data
are equal expansions in key order at `ks`. -/
theorem DEqual_expandK_of_checks {F : Fam} {D₁ D₂ : Diagram} {o₁ o₂ : ℕ} (ks : ℕ → ℕ) (s₁ s₂ : ℂ)
    (hc₁ : D₁.FamOK F o₁) (hc₂ : D₂.FamOK F o₂) (hbd : D₁.bndData = D₂.bndData)
    (hks : D₁.keySig = D₂.keySig) (hpos : ∀ i, 0 < F.dims i)
    (hEq : Equal (F.env ks) s₁ (D₁.toNetB o₁) s₂ (D₂.toNetB o₂)) :
    DEqual (F.env ks) s₁ (D₁.expandK ks) s₂ (D₂.expandK ks) := by
  have hp : ∀ i, 0 < (F.env ks).dimv i := fun i => Fam.env_dimv_pos F ks i (hpos _)
  obtain ⟨w₁, b₁, n₁, l₁, s₁'⟩ := hc₁
  obtain ⟨w₂, b₂, n₂, l₂, -⟩ := hc₂
  exact DEqual_expandK_of_net (WF_of_WFb w₁) (WF_of_WFb w₂) (F.env ks) o₁ o₂ s₁ s₂ b₁ b₂
    (fun n hn => NodeS_of (n₁ n hn) hp) (fun n hn => NodeS_of (n₂ n hn) hp)
    (fun j hj => labDim_of F ks _ _ _ (l₁ j hj).1 (l₁ j hj).2)
    (fun j hj => labDim_of F ks _ _ _ (l₂ j hj).1 (l₂ j hj).2)
    (fun i hi => labDim_of F ks _ _ _ (s₁' i hi).1 (s₁' i hi).2) hbd hks hEq

theorem slotDig_lt (E : Env) (β : ℕ → ℕ) (hpos : ∀ i, 0 < E.dimv i) (i : ℕ) (t : List ℕ) :
    D.slotDig E β i t < E.dimv (D.portDim (D.bnd.getD i ⟨0, true, 0⟩)) :=
  digit_lt _ _ _ (hpos _)

variable (D) in
/-- The slot digits of `β` in key order. -/
def slotK (E : Env) (β : ℕ → ℕ) (r : ℕ) : ℕ :=
  if r < (D.SL E.mult).length then
    D.slotDig E β ((D.SL E.mult).getD r (0, [])).1 ((D.SL E.mult).getD r (0, [])).2
  else 0

theorem slotK_lt (hW : WF D) (E : Env) (β : ℕ → ℕ) (hpos : ∀ i, 0 < E.dimv i) :
    (D.expandK E.mult).InRange E (D.slotK E β) := by
  intro r hr
  have hb := bdims_expandK hW E.mult
  have hlen : (D.expandK E.mult).bnd.length = (D.SL E.mult).length := by
    simpa [bdims] using congrArg List.length hb
  have hr' : r < (D.SL E.mult).length := hlen ▸ hr
  have hm := (SL_perm E.mult).mem_iff.1 (List.getElem_mem hr')
  obtain ⟨hi, -⟩ := (mem_SDom D _ _).1 hm
  have := slotDig_lt (D := D) E β hpos (D.SL E.mult)[r].1 (D.SL E.mult)[r].2
  rw [List.getD_eq_getElem _ _ hi] at this
  simp only [slotK, if_pos hr', hb, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem hr', Option.map_some, Option.getD_some]
  simpa [List.getElem?_eq_getElem hi] using this

/-- The tensor entry of the family's network at `β` is the expansion in key order at the slot
digits of `β` in key order. -/
theorem eval_net_expandK (hW : WF D) (E : Env) (off : ℕ) (hoff : D.bnd.length ≤ off)
    (hN : ∀ n ∈ D.nodes, D.NodeS E n)
    (hlab : ∀ j (hj : j < D.wires.length), labDim E (off + j) =
      E.dimv (D.portDim D.wires[j].1) ^ size E.mult (D.legChain D.wires[j].1))
    (hpos : ∀ i, 0 < E.dimv i) (β : ℕ → ℕ) (hβ : ∀ i (hi : i < D.bnd.length),
      β i < E.dimv (D.portDim D.bnd[i]) ^ size E.mult (D.legChain D.bnd[i])) :
    ArchytasZX.eval E (D.toNetB off) β = (D.expandK E.mult).eval E (D.slotK E β) := by
  rw [eval_toNetB_evalD hW E off hoff hN hlab β hβ, eval_expandK_evalD hW E _ (slotK_lt hW E β hpos)]
  refine evalD_slots hW E E.mult _ _ fun i hi t ht => ?_
  have hx : (i, t) ∈ D.SL E.mult := (SL_perm E.mult).mem_iff.2 ((mem_SDom D _ _).2 ⟨hi, ht⟩)
  have hlt := List.idxOf_lt_length_of_mem hx
  simp only [readK, slotK, if_pos hlt]
  rw [List.getD_eq_getElem _ _ hlt, List.getElem_idxOf]

theorem slotK_congr {D₁ D₂ : Diagram} (hbd : D₁.bndData = D₂.bndData) (hks : D₁.keySig = D₂.keySig)
    (E : Env) (β : ℕ → ℕ) : D₁.slotK E β = D₂.slotK E β := by
  obtain ⟨hlen, hd⟩ := bndData_at hbd
  have hSL := SL_congr hbd hks E.mult
  funext r
  unfold slotK
  rw [← hSL]
  by_cases hr : r < (D₁.SL E.mult).length
  · rw [if_pos hr, if_pos hr, List.getD_eq_getElem _ _ hr]
    have hm := (SL_perm E.mult).mem_iff.1 (List.getElem_mem hr)
    obtain ⟨hi, -⟩ := (mem_SDom D₁ _ _).1 hm
    have hi₂ : (D₁.SL E.mult)[r].1 < D₂.bnd.length := hlen ▸ hi
    simp only [slotDig, List.getD_eq_getElem _ _ hi, List.getD_eq_getElem _ _ hi₂,
      (hd _ hi hi₂).1, (hd _ hi hi₂).2]
  · rw [if_neg hr, if_neg hr]

/-- Equal expansions in key order of two checked boxed diagrams with the same boundary data and
key data have equal family networks at `ks`. -/
theorem Equal_of_DEqual_expandK {F : Fam} {D₁ D₂ : Diagram} {o₁ o₂ : ℕ} (ks : ℕ → ℕ) (s₁ s₂ : ℂ)
    (hc₁ : D₁.FamOK F o₁) (hc₂ : D₂.FamOK F o₂) (hbd : D₁.bndData = D₂.bndData)
    (hks : D₁.keySig = D₂.keySig) (hpos : ∀ i, 0 < F.dims i)
    (h : DEqual (F.env ks) s₁ (D₁.expandK ks) s₂ (D₂.expandK ks)) :
    Equal (F.env ks) s₁ (D₁.toNetB o₁) s₂ (D₂.toNetB o₂) := by
  have hp : ∀ i, 0 < (F.env ks).dimv i := fun i => Fam.env_dimv_pos F ks i (hpos _)
  obtain ⟨hlen, hd⟩ := bndData_at hbd
  obtain ⟨w₁, b₁, n₁, l₁, s₁'⟩ := hc₁
  obtain ⟨w₂, b₂, n₂, l₂, -⟩ := hc₂
  have hW₁ := WF_of_WFb w₁
  have hW₂ := WF_of_WFb w₂
  refine ⟨by simp [toNetB, hlen], fun β hβ => ?_⟩
  have hβ₁ : ∀ i (hi : i < D₁.bnd.length), β i < (F.env ks).dimv (D₁.portDim D₁.bnd[i]) ^
      size (F.env ks).mult (D₁.legChain D₁.bnd[i]) := fun i hi => by
    have := hβ i (by simpa [toNetB] using hi)
    rwa [labDim_of F ks _ _ _ (s₁' i hi).1 (s₁' i hi).2] at this
  have hβ₂ : ∀ i (hi : i < D₂.bnd.length), β i < (F.env ks).dimv (D₂.portDim D₂.bnd[i]) ^
      size (F.env ks).mult (D₂.legChain D₂.bnd[i]) := fun i hi => by
    have h₁ : i < D₁.bnd.length := hlen ▸ hi
    rw [← (hd i h₁ hi).1, ← (hd i h₁ hi).2]
    exact hβ₁ i h₁
  rw [eval_net_expandK hW₁ (F.env ks) o₁ b₁ (fun n hn => NodeS_of (n₁ n hn) hp)
      (fun j hj => labDim_of F ks _ _ _ (l₁ j hj).1 (l₁ j hj).2) hp β hβ₁,
    eval_net_expandK hW₂ (F.env ks) o₂ b₂ (fun n hn => NodeS_of (n₂ n hn) hp)
      (fun j hj => labDim_of F ks _ _ _ (l₂ j hj).1 (l₂ j hj).2) hp β hβ₂,
    ← slotK_congr hbd hks]
  exact h.2 _ (slotK_lt hW₁ (F.env ks) β hp)

end Diagram

/-- The families `s₁ • D₁` and `s₂ • D₂` agree at every assignment `σ` of their multiplicity
symbols: the expansions in key order at the multiplicities `M σ`, at the environment `E σ`. -/
def DEqualF (E : (ℕ → ℕ) → Env) (M : (ℕ → ℕ) → ℕ → ℕ) (s₁ : (ℕ → ℕ) → ℂ) (D₁ : Diagram)
    (s₂ : (ℕ → ℕ) → ℂ) (D₂ : Diagram) : Prop :=
  ∀ σ, DEqual (E σ) (s₁ σ) (D₁.expandK (M σ)) (s₂ σ) (D₂.expandK (M σ))

end ArchytasZX
