/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.NetDigits

/-!
# The expansion through the family's network

`eval_expand_net`: the expansion of a boxed diagram at the multiplicities `E.mult` is the family's
network `D.toNetB off` at the bundled boundary assignment packing the slot digits that `bmap`
reads.
-/

namespace ArchytasZX

open Finset

theorem digit_ofDigits (d : ℕ) (hd : 0 < d) (L : List ℕ) (hL : ∀ x ∈ L, x < d) (i : ℕ) :
    digit d i (Nat.ofDigits d L) = L.getD i 0 := by
  induction L generalizing i with
  | nil => simp [digit, Nat.ofDigits]
  | cons a L ih =>
    rw [Nat.ofDigits_cons]
    have ha := hL a List.mem_cons_self
    cases i with
    | zero => simp [digit_zero_add d a _ ha]
    | succ i =>
      rw [digit_succ_add d a _ i hd ha, ih (fun x hx => hL x (List.mem_cons_of_mem _ hx))]
      simp

theorem ofDigits_lt (d : ℕ) (L : List ℕ) (hL : ∀ x ∈ L, x < d) : Nat.ofDigits d L < d ^ L.length := by
  induction L with
  | nil => simp [Nat.ofDigits]
  | cons a L ih =>
    rw [Nat.ofDigits_cons, List.length_cons, pow_succ']
    have ha := hL a List.mem_cons_self
    have h2 := ih fun x hx => hL x (List.mem_cons_of_mem _ hx)
    have : d * Nat.ofDigits d L + d ≤ d * d ^ L.length := by
      rw [← Nat.mul_succ]; exact Nat.mul_le_mul_left _ h2
    omega

theorem cidx_getD_pos (ks : ℕ → ℕ) (c t : List ℕ) (ht : t ∈ cidx ks c) :
    (cidx ks c).getD (pos ks c t) [] = t := by
  have hl : pos ks c t < (cidx ks c).length := by rw [cidx_length]; exact pos_lt ks c t ht
  rw [List.getD_eq_getElem _ _ hl]
  exact pos_inj ks c _ _ (List.getElem_mem hl) ht (getElem_cidx_pos ks c _ hl)

/-- The number whose base-`d` digits along `c` are `ρ`. -/
def packC (ks : ℕ → ℕ) (d : ℕ) (c : List ℕ) (ρ : List ℕ → ℕ) : ℕ :=
  Nat.ofDigits d ((List.range (size ks c)).map fun j => ρ ((cidx ks c).getD j []))

theorem digit_packC (ks : ℕ → ℕ) (d : ℕ) (hd : 0 < d) (c : List ℕ) (ρ : List ℕ → ℕ)
    (hρ : ∀ t ∈ cidx ks c, ρ t < d) (t : List ℕ) (ht : t ∈ cidx ks c) :
    digit d (pos ks c t) (packC ks d c ρ) = ρ t := by
  have hL : ∀ x ∈ (List.range (size ks c)).map (fun j => ρ ((cidx ks c).getD j [])), x < d := by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hx
    have hj' : j < (cidx ks c).length := by rw [cidx_length]; exact List.mem_range.1 hj
    rw [List.getD_eq_getElem _ _ hj']
    exact hρ _ (List.getElem_mem hj')
  rw [packC, digit_ofDigits d hd _ hL, List.getD_eq_getElem _ _ (by simpa using pos_lt ks c t ht)]
  simp only [List.getElem_map, List.getElem_range]
  rw [cidx_getD_pos ks c t ht]

theorem packC_lt (ks : ℕ → ℕ) (d : ℕ) (c : List ℕ) (ρ : List ℕ → ℕ)
    (hρ : ∀ t ∈ cidx ks c, ρ t < d) : packC ks d c ρ < d ^ size ks c := by
  have := ofDigits_lt d ((List.range (size ks c)).map fun j => ρ ((cidx ks c).getD j [])) (by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hx
    have hj' : j < (cidx ks c).length := by rw [cidx_length]; exact List.mem_range.1 hj
    rw [List.getD_eq_getElem _ _ hj']
    exact hρ _ (List.getElem_mem hj'))
  rw [List.length_map, List.length_range] at this
  exact this

namespace Diagram

variable {D : Diagram}

/-- The digit sum reads the slots only at their copy indices. -/
theorem evalD_slots (hW : WF D) (E : Env) (ks : ℕ → ℕ) (σ σ' : ℕ → List ℕ → ℕ)
    (h : ∀ i (hi : i < D.bnd.length), ∀ t ∈ cidx ks (D.legChain D.bnd[i]), σ i t = σ' i t) :
    D.evalD E ks σ = D.evalD E ks σ' := by
  set S2 := D.bnd.zipIdx.flatMap fun q => (cidx ks (D.legChain q.1)).map (slab q.2)
  have hS2 : ∀ q ∈ D.bnd.zipIdx, ∀ t ∈ cidx ks (D.legChain q.1), slab q.2 t ∈ S2 := fun q hq t ht =>
    List.mem_flatMap.2 ⟨q, hq, List.mem_map.2 ⟨t, ht, rfl⟩⟩
  have hw1 : ∀ w ∈ D.wires, ∀ t ∈ cidx ks (D.legChain w.1), dlab w.1 t ∈ D.dlabs ks :=
    fun w hw t ht => (mem_dlabs D ks _).2 ⟨w.1, (wire_not_bnd hW hw).2.2.1, t, ht, rfl⟩
  have hw2 : ∀ w ∈ D.wires, ∀ t ∈ cidx ks (D.legChain w.1), dlab w.2 t ∈ D.dlabs ks :=
    fun w hw t ht => (mem_dlabs D ks _).2 ⟨w.2, (wire_not_bnd hW hw).2.2.2, t,
      by rw [← legChain_wire hW hw]; exact ht, rfl⟩
  have hb1 : ∀ q ∈ D.bnd.zipIdx, ∀ t ∈ cidx ks (D.legChain q.1), dlab q.1 t ∈ D.dlabs ks :=
    fun q hq t ht => by
      obtain ⟨hi, e⟩ := (mem_zipIdx' _ _ _).1 hq
      exact (mem_dlabs D ks _).2 ⟨q.1, bnd_legs hW (e ▸ List.getElem_mem hi), t, ht, rfl⟩
  have dir : ∀ a b : ℕ → ℕ, (∀ x ∈ D.dlabs ks ++ S2, a x = b x) → D.pinsD ks a → D.pinsD ks b := by
    rintro a b hab ⟨hw, hb⟩
    refine ⟨fun w hwm t ht => ?_, fun q hq t ht => ?_⟩
    · rw [← hab _ (List.mem_append_left _ (hw1 w hwm t ht)),
        ← hab _ (List.mem_append_left _ (hw2 w hwm t ht))]
      exact hw w hwm t ht
    · rw [← hab _ (List.mem_append_left _ (hb1 q hq t ht)),
        ← hab _ (List.mem_append_right _ (hS2 q hq t ht))]
      exact hb q hq t ht
  unfold evalD
  refine sumOver_ext _ _ (D.dlabs ks ++ S2) _ (fun τ τ' hτ => ?_) _ _ (fun x hx hn => ?_)
  · classical
    refine if_congr ⟨dir τ τ' hτ, dir τ' τ fun x hx => (hτ x hx).symm⟩
      (prodD_congr D E ks τ τ' fun x hx => hτ x (List.mem_append_left _ hx)) rfl
  · rcases List.mem_append.1 hx with hx | hx
    · exact absurd hx hn
    · obtain ⟨q, hq, t, ht, rfl⟩ : ∃ q ∈ D.bnd.zipIdx, ∃ t ∈ cidx ks (D.legChain q.1), x = slab q.2 t := by
        obtain ⟨q, hq, hx⟩ := List.mem_flatMap.1 hx
        obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hx
        exact ⟨q, hq, t, ht, rfl⟩
      obtain ⟨hi, e⟩ := (mem_zipIdx' _ _ _).1 hq
      rw [slotsD_slab, slotsD_slab]
      exact h q.2 hi t (by rw [e]; exact ht)

/-- A slot digit beyond its dimension makes the digit sum vanish. -/
theorem evalD_out (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) (i : ℕ) (hi : i < D.bnd.length)
    (hW : WF D) (t : List ℕ) (ht : t ∈ cidx ks (D.legChain D.bnd[i]))
    (hout : E.dimv (D.portDim D.bnd[i]) ≤ σ i t) : D.evalD E ks σ = 0 := by
  unfold evalD
  have hmem : dlab D.bnd[i] t ∈ D.dlabs ks :=
    (mem_dlabs D ks _).2 ⟨_, bnd_legs hW (List.getElem_mem hi), t, ht, rfl⟩
  rw [sumOver_congr _ _ _ (fun _ => 0) _ fun τ hτ hr => ?_, sumOver_zero]
  classical
  rw [if_neg]
  rintro ⟨-, hb⟩
  have e := hb (D.bnd[i], i) ((mem_zipIdx' _ _ _).2 ⟨hi, rfl⟩) t ht
  have hr' := hr _ hmem
  simp only [dport_dlab] at hr'
  simp only at e
  rw [hτ (slab i t) fun h => by
    obtain ⟨p, -, u, -, e'⟩ := (mem_dlabs D ks _).1 h
    exact dlab_ne_slab p u i t e'.symm, slotsD_slab] at e
  omega

variable (D) in
/-- The bundled boundary assignment packing the slot digits `ρ`. -/
def packB (E : Env) (ρ : ℕ → List ℕ → ℕ) (i : ℕ) : ℕ :=
  if i < D.bnd.length then
    packC E.mult (E.dimv (D.portDim (D.bnd.getD i ⟨0, true, 0⟩)))
      (D.legChain (D.bnd.getD i ⟨0, true, 0⟩)) (ρ i)
  else 0

/-- The expansion's tensor entry is the digit sum at the slots `bmap` reads. -/
theorem eval_expand_evalD (hW : WF D) (E : Env) (σ'' : ℕ → ℕ) (hσ : (D.expand E.mult).InRange E σ'') :
    (D.expand E.mult).eval E σ'' = D.evalD E E.mult (D.bmap E.mult fun j _ => σ'' j) := by
  obtain ⟨he, hWe, hbe⟩ := evalD_expand D E.mult E hW (fun j _ => σ'' j)
  have h1 := evalD_boxfree hbe hWe E E.mult (fun j _ => σ'' j) fun i hi => by
    have := hσ i hi
    simpa [bdims, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi] using this
  rw [← he, h1]

/-- The expansion is the family's network at the packed slot digits. -/
theorem eval_expand_net (hW : WF D) (E : Env) (off : ℕ) (hoff : D.bnd.length ≤ off)
    (hN : ∀ n ∈ D.nodes, D.NodeS E n)
    (hlab : ∀ j (hj : j < D.wires.length), labDim E (off + j) =
      E.dimv (D.portDim D.wires[j].1) ^ size E.mult (D.legChain D.wires[j].1))
    (σ'' : ℕ → ℕ) (hσ : (D.expand E.mult).InRange E σ'')
    (hin : ∀ i (hi : i < D.bnd.length), ∀ t ∈ cidx E.mult (D.legChain D.bnd[i]),
      D.bmap E.mult (fun j _ => σ'' j) i t < E.dimv (D.portDim D.bnd[i])) :
    (D.expand E.mult).eval E σ'' =
      ArchytasZX.eval E (D.toNetB off) (D.packB E (D.bmap E.mult fun j _ => σ'' j)) := by
  set ρ := D.bmap E.mult fun j _ => σ'' j
  have hpd : ∀ i (hi : i < D.bnd.length), 0 < E.dimv (D.portDim D.bnd[i]) := fun i hi => by
    obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 (bnd_legs hW (List.getElem_mem hi))
    exact (hN n hn).posd _ hpn
  have hg : ∀ i (hi : i < D.bnd.length), D.bnd.getD i ⟨0, true, 0⟩ = D.bnd[i] := fun i hi => by
    rw [List.getD_eq_getElem _ _ hi]
  rw [eval_expand_evalD hW E σ'' hσ, eval_toNetB_evalD hW E off hoff hN hlab _ fun i hi => ?_]
  · refine evalD_slots hW E E.mult _ _ fun i hi t ht => ?_
    simp only [slotDig, packB, if_pos hi, hg i hi]
    exact (digit_packC _ _ (hpd i hi) _ _ (fun u hu => hin i hi u hu) t ht).symm
  · simp only [packB, if_pos hi, hg i hi]
    exact packC_lt _ _ _ _ fun u hu => hin i hi u hu

theorem packC_congr (ks : ℕ → ℕ) (d : ℕ) (c : List ℕ) (ρ ρ' : List ℕ → ℕ)
    (h : ∀ t ∈ cidx ks c, ρ t = ρ' t) : packC ks d c ρ = packC ks d c ρ' := by
  unfold packC
  congr 1
  refine List.map_congr_left fun j hj => h _ ?_
  have hj' : j < (cidx ks c).length := by rw [cidx_length]; exact List.mem_range.1 hj
  rw [List.getD_eq_getElem _ _ hj']
  exact List.getElem_mem hj'

/-- The boundary data the packing reads: each slot's dimension and chain. -/
def bndData (D : Diagram) : List (ℕ × List ℕ) := D.bnd.map fun q => (D.portDim q, D.legChain q)

/-- Equal family networks give equal expansions when the boundary data and the slot maps agree. -/
theorem DEqual_expand_of_net {D₁ D₂ : Diagram} (hW₁ : WF D₁) (hW₂ : WF D₂) (E : Env) (o₁ o₂ : ℕ)
    (s₁ s₂ : ℂ) (ho₁ : D₁.bnd.length ≤ o₁) (ho₂ : D₂.bnd.length ≤ o₂)
    (hN₁ : ∀ n ∈ D₁.nodes, D₁.NodeS E n) (hN₂ : ∀ n ∈ D₂.nodes, D₂.NodeS E n)
    (hl₁ : ∀ j (hj : j < D₁.wires.length), labDim E (o₁ + j) =
      E.dimv (D₁.portDim D₁.wires[j].1) ^ size E.mult (D₁.legChain D₁.wires[j].1))
    (hl₂ : ∀ j (hj : j < D₂.wires.length), labDim E (o₂ + j) =
      E.dimv (D₂.portDim D₂.wires[j].1) ^ size E.mult (D₂.legChain D₂.wires[j].1))
    (hb : ∀ i (hi : i < D₁.bnd.length), labDim E i =
      E.dimv (D₁.portDim D₁.bnd[i]) ^ size E.mult (D₁.legChain D₁.bnd[i]))
    (hbd : D₁.bndData = D₂.bndData)
    (hmap : ∀ σ'' : ℕ → ℕ, ∀ i (hi : i < D₁.bnd.length), ∀ t ∈ cidx E.mult (D₁.legChain D₁.bnd[i]),
      D₁.bmap E.mult (fun j _ => σ'' j) i t = D₂.bmap E.mult (fun j _ => σ'' j) i t)
    (hexp : (D₁.expand E.mult).bdims = (D₂.expand E.mult).bdims)
    (hEq : Equal E s₁ (D₁.toNetB o₁) s₂ (D₂.toNetB o₂)) :
    DEqual E s₁ (D₁.expand E.mult) s₂ (D₂.expand E.mult) := by
  have hlen : D₁.bnd.length = D₂.bnd.length := by
    simpa [bndData] using congrArg List.length hbd
  have hd : ∀ i (h₁ : i < D₁.bnd.length) (h₂ : i < D₂.bnd.length),
      D₁.portDim D₁.bnd[i] = D₂.portDim D₂.bnd[i] ∧ D₁.legChain D₁.bnd[i] = D₂.legChain D₂.bnd[i] :=
    fun i h₁ h₂ => by
      have := congrArg (·[i]?) hbd
      simp only [bndData, List.getElem?_map, List.getElem?_eq_getElem h₁,
        List.getElem?_eq_getElem h₂, Option.map_some, Option.some.injEq, Prod.mk.injEq] at this
      exact this
  refine ⟨hexp, fun σ'' hσ => ?_⟩
  have hσ₂ : (D₂.expand E.mult).InRange E σ'' := by
    intro i hi
    have hl : (D₂.expand E.mult).bnd.length = (D₁.expand E.mult).bnd.length := by
      simpa [bdims] using (congrArg List.length hexp).symm
    rw [← hexp]; exact hσ i (hl ▸ hi)
  set ρ₁ := D₁.bmap E.mult fun j _ => σ'' j
  set ρ₂ := D₂.bmap E.mult fun j _ => σ'' j
  by_cases hin : ∀ i (hi : i < D₁.bnd.length), ∀ t ∈ cidx E.mult (D₁.legChain D₁.bnd[i]),
      ρ₁ i t < E.dimv (D₁.portDim D₁.bnd[i])
  · have hin₂ : ∀ i (hi : i < D₂.bnd.length), ∀ t ∈ cidx E.mult (D₂.legChain D₂.bnd[i]),
        ρ₂ i t < E.dimv (D₂.portDim D₂.bnd[i]) := fun i hi t ht => by
      have h₁ : i < D₁.bnd.length := hlen ▸ hi
      obtain ⟨e1, e2⟩ := hd i h₁ hi
      rw [← e2] at ht
      have hm : ρ₁ i t = ρ₂ i t := hmap σ'' i h₁ t ht
      rw [← hm, ← e1]
      exact hin i h₁ t ht
    have hpack : D₂.packB E ρ₂ = D₁.packB E ρ₁ := by
      funext i
      unfold packB
      by_cases hi : i < D₁.bnd.length
      · have hi₂ : i < D₂.bnd.length := hlen ▸ hi
        obtain ⟨e1, e2⟩ := hd i hi hi₂
        rw [if_pos hi, if_pos hi₂, List.getD_eq_getElem _ _ hi, List.getD_eq_getElem _ _ hi₂, ← e1, ← e2]
        exact packC_congr _ _ _ _ _ fun t ht => (hmap σ'' i hi t ht).symm
      · rw [if_neg hi, if_neg (hlen ▸ hi)]
    rw [eval_expand_net hW₁ E o₁ ho₁ hN₁ hl₁ σ'' hσ hin, eval_expand_net hW₂ E o₂ ho₂ hN₂ hl₂ σ'' hσ₂ hin₂,
      hpack]
    refine hEq.2 _ fun l hl => ?_
    have hi : l < D₁.bnd.length := by simpa [toNetB] using hl
    have hpd : 0 < E.dimv (D₁.portDim D₁.bnd[l]) := by
      obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 (bnd_legs hW₁ (List.getElem_mem hi))
      exact (hN₁ n hn).posd _ hpn
    rw [hb l hi]
    simp only [packB, if_pos hi, List.getD_eq_getElem _ _ hi]
    exact packC_lt _ _ _ _ fun u hu => hin l hi u hu
  · push_neg at hin
    obtain ⟨i, hi, t, ht, hout⟩ := hin
    have hi₂ : i < D₂.bnd.length := hlen ▸ hi
    obtain ⟨e1, e2⟩ := hd i hi hi₂
    rw [eval_expand_evalD hW₁ E σ'' hσ, eval_expand_evalD hW₂ E σ'' hσ₂,
      evalD_out E E.mult ρ₁ i hi hW₁ t ht hout,
      evalD_out E E.mult ρ₂ i hi₂ hW₂ t (by rw [← e2]; exact ht)
        (by have hm : ρ₁ i t = ρ₂ i t := hmap σ'' i hi t ht
            rw [← e1, ← hm]; exact hout)]
    simp

end Diagram

end ArchytasZX
