/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Pin

/-!
# The bundled network as a digit sum

`eval_toNetB`: the family's network `D.toNetB off` at a bundled boundary assignment is the digit
sum of `evalD_pin`, each bundled value read as its digits along its chain.
-/

namespace ArchytasZX

open Finset

/-- Summing labels `L` whose values are numbers of `B l`'s digits is summing the digits. -/
theorem sumOver_blocks (Dm Dm' : ℕ → ℕ) (L : List ℕ) (B : ℕ → List ℕ) (dd : ℕ → ℕ)
    (hdd : ∀ l ∈ L, 0 < dd l) (hL : L.Nodup) (hB : (L.flatMap B).Nodup)
    (hD : ∀ l ∈ L, Dm l = dd l ^ (B l).length) (hD' : ∀ l ∈ L, ∀ x ∈ B l, Dm' x = dd l)
    (f g : (ℕ → ℕ) → ℂ) (σ σ' : ℕ → ℕ)
    (H : ∀ τ τ' : ℕ → ℕ, (∀ x ∉ L, τ x = σ x) → (∀ x ∉ L.flatMap B, τ' x = σ' x) →
      (∀ l ∈ L, τ l < Dm l) → (∀ l ∈ L, ∀ x ∈ B l, τ' x = digit (dd l) ((B l).idxOf x) (τ l)) →
      f τ = g τ') :
    sumOver Dm L f σ = sumOver Dm' (L.flatMap B) g σ' := by
  induction L generalizing σ σ' with
  | nil => exact H σ σ' (fun _ _ => rfl) (fun _ _ => rfl) (by simp) (by simp)
  | cons l L ih =>
    obtain ⟨hl, hL'⟩ := List.nodup_cons.1 hL
    rw [List.flatMap_cons] at hB ⊢
    obtain ⟨hBl, hBL, hdis⟩ := List.nodup_append.1 hB
    simp only [sumOver]
    rw [sumOver_append, sumOver_digits Dm' (dd l) (hdd l List.mem_cons_self) (B l) hBl
      (hD' l List.mem_cons_self), hD l List.mem_cons_self]
    refine Finset.sum_congr rfl fun v hv => ?_
    refine ih (fun x hx => hdd x (List.mem_cons_of_mem _ hx)) hL' hBL
      (fun x hx => hD x (List.mem_cons_of_mem _ hx)) (fun x hx => hD' x (List.mem_cons_of_mem _ hx))
      _ _ fun τ τ' hτ hτ' hr hdg => H τ τ' (fun x hx => ?_) (fun x hx => ?_) (fun x hx => ?_)
        (fun x hx => ?_)
    · have h1 : x ≠ l := fun e => hx (e ▸ List.mem_cons_self)
      rw [hτ x (fun h => hx (List.mem_cons_of_mem _ h)), Function.update_of_ne h1]
    · rw [List.flatMap_cons] at hx
      have hx' : x ∉ B l := fun h => hx (List.mem_append_left _ h)
      rw [hτ' x (fun h => hx (List.mem_append_right _ h))]
      simp [setDigits, hx']
    · rcases List.mem_cons.1 hx with rfl | hx
      · rw [hτ x hl, Function.update_self, hD x List.mem_cons_self]; exact Finset.mem_range.1 hv
      · exact hr x hx
    · intro y hy
      rcases List.mem_cons.1 hx with rfl | hx
      · rw [hτ' y (fun h => hdis _ hy _ h rfl), hτ x hl, Function.update_self]
        simp [setDigits, hy]
      · exact hdg x hx y hy

theorem flatMap_range_off {α : Type*} (ws : List α) (off : ℕ) (g : α → List ℕ) :
    ((List.range ws.length).map (off + ·)).flatMap (fun l => ((ws[l - off]?).map g).getD []) =
      ws.flatMap g := by
  induction ws generalizing off with
  | nil => rfl
  | cons w ws ih =>
    rw [List.length_cons, List.range_succ_eq_map, List.map_cons, List.flatMap_cons,
      List.flatMap_cons, List.map_map]
    simp only [Nat.add_zero, Nat.sub_self, List.getElem?_cons_zero, Option.map_some,
      Option.getD_some]
    congr 1
    rw [← ih (off + 1)]
    have e : (List.range ws.length).map ((off + ·) ∘ Nat.succ) =
        (List.range ws.length).map (off + 1 + ·) :=
      List.map_congr_left fun j _ => by simp only [Function.comp]; omega
    rw [e]
    refine List.flatMap_congr fun l hl => ?_
    obtain ⟨j, -, rfl⟩ := List.mem_map.1 hl
    rw [show off + 1 + j - off = j + 1 by omega, show off + 1 + j - (off + 1) = j by omega]
    rfl

theorem idxOf_cidx (ks : ℕ → ℕ) (c : List ℕ) (p : Port) (u : List ℕ) (hu : u ∈ cidx ks c) :
    ((cidx ks c).map (dlab p)).idxOf (dlab p u) = pos ks c u := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hu
  rw [getElem_cidx_pos]
  have hnd : ((cidx ks c).map (dlab p)).Nodup := (cidx_nodup ks c).map fun a b e => (dlab_inj e).2
  have := List.get_idxOf hnd ⟨i, by simpa using hi⟩
  simpa using this

namespace Diagram

variable (D : Diagram)

/-- The structural conditions on a node for `val_bnode`. -/
structure NodeS (E : Env) (n : DNode) : Prop where
  plain : n.kind.plain = true
  boxkind : D.nodeChain n.id ≠ [] → n.kind ≠ .B ∧ n.kind ≠ .S
  tailkind : ¬(tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = []) →
    n.kind = .Z ∨ n.kind = .X ∨ (n.kind = .W ∧ tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = [])
  dims : (D.nodeChain n.id ≠ [] ∨ ¬(tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = [])) → ∀ p ∈ n.legs, D.portDim p = n.dim
  posd : ∀ p ∈ n.legs, 0 < E.dimv (D.portDim p)
  posn : 0 < E.dimv n.dim

theorem label_eq (off : ℕ) (p : Port) : D.label off p = D.value id (off + ·) p := by
  unfold label value
  rcases D.locate p with _ | (i | j) <;> rfl

/-- Slot `i`'s digits of the bundled boundary assignment `β`. -/
def slotDig (E : Env) (β : ℕ → ℕ) (i : ℕ) (t : List ℕ) : ℕ :=
  digit (E.dimv (D.portDim (D.bnd.getD i ⟨0, true, 0⟩)))
    (pos E.mult (D.legChain (D.bnd.getD i ⟨0, true, 0⟩)) t) (β i)

variable {D}

/-- The family's network is the digit sum over the wires' first-end digits. -/
theorem eval_toNetB (hW : WF D) (E : Env) (off : ℕ) (hoff : D.bnd.length ≤ off)
    (hN : ∀ n ∈ D.nodes, D.NodeS E n)
    (hlab : ∀ j (hj : j < D.wires.length), labDim E (off + j) =
      E.dimv (D.portDim D.wires[j].1) ^ size E.mult (D.legChain D.wires[j].1))
    (β : ℕ → ℕ) (hβ : ∀ i (hi : i < D.bnd.length),
      β i < E.dimv (D.portDim D.bnd[i]) ^ size E.mult (D.legChain D.bnd[i])) :
    ArchytasZX.eval E (D.toNetB off) β = sumOver (fun l => E.dimv (D.portDim (dport l))) (D.wdl E.mult)
      (fun τ => D.prodD E E.mult fun x => if x ∈ D.rdl E.mult then τ (D.pinD x) else τ x)
      (slotsD (D.slotDig E β)) := by
  have hall := hW.perm.nodup_iff.2 (legs_nodup_of D hW.ids)
  obtain ⟨n1, n23, d1⟩ := List.nodup_append.1 hall
  obtain ⟨n2, n3, d23⟩ := List.nodup_append.1 n23
  have n12 : (D.wires.map (·.1) ++ D.wires.map (·.2)).Nodup :=
    List.nodup_append.2 ⟨n1, n2, fun a ha b hb => d1 a ha b (List.mem_append_left _ hb)⟩
  have hnd := (dlabs_perm hW E.mult).nodup_iff.1 (dlabs_nodup D E.mult (legs_nodup_of D hW.ids))
  obtain ⟨hwnd, -, hdis⟩ := List.nodup_append.1 hnd
  set B : ℕ → List ℕ := fun l => ((D.wires[l - off]?).map fun w =>
    (cidx E.mult (D.legChain w.1)).map (dlab w.1)).getD [] with hB_def
  set dd : ℕ → ℕ := fun l => ((D.wires[l - off]?).map fun w => E.dimv (D.portDim w.1)).getD 1
  have hflat : ((List.range D.wires.length).map (off + ·)).flatMap B = D.wdl E.mult :=
    flatMap_range_off D.wires off _
  have hBj : ∀ j (hj : j < D.wires.length),
      B (off + j) = (cidx E.mult (D.legChain D.wires[j].1)).map (dlab D.wires[j].1) := fun j hj => by
    simp [hB_def, List.getElem?_eq_getElem hj]
  have hdj : ∀ j (hj : j < D.wires.length), dd (off + j) = E.dimv (D.portDim D.wires[j].1) :=
    fun j hj => by simp [dd, List.getElem?_eq_getElem hj]
  have hmemL : ∀ l ∈ (List.range D.wires.length).map (off + ·), ∃ j, ∃ hj : j < D.wires.length,
      l = off + j := fun l hl => by
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hl
    exact ⟨j, List.mem_range.1 hj, rfl⟩
  have hlab1 : ∀ j (hj : j < D.wires.length), D.label off D.wires[j].1 = off + j := fun j hj => by
    rw [label_eq, value_wire D n12 (wire_not_bnd hW (List.getElem_mem hj)).1 _ _ j hj (Or.inl rfl)]
  have hlab2 : ∀ j (hj : j < D.wires.length), D.label off D.wires[j].2 = off + j := fun j hj => by
    rw [label_eq, value_wire D n12 (wire_not_bnd hW (List.getElem_mem hj)).2.1 _ _ j hj (Or.inr rfl)]
  have hlabB : ∀ i (hi : i < D.bnd.length), D.label off D.bnd[i] = i := fun i hi => by
    rw [label_eq, value_bnd D n3 _ _ i hi]; rfl
  have hposd : ∀ p ∈ D.legs, 0 < E.dimv (D.portDim p) := fun p hp => by
    obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 hp
    exact (hN n hn).posd p hpn
  unfold ArchytasZX.eval
  simp only [toNetB]
  rw [← hflat]
  refine sumOver_blocks (labDim E) _ _ B dd (fun l hl => ?_) ((List.nodup_range).map
    fun a b e => by simpa using e) (hflat ▸ hwnd) (fun l hl => ?_) (fun l hl x hx => ?_) _ _ _ _
    fun τ τ' hτ hτ' hr hdg => ?_
  · obtain ⟨j, hj, rfl⟩ := hmemL l hl
    rw [hdj j hj]
    exact hposd _ (wire_not_bnd hW (List.getElem_mem hj)).2.2.1
  · obtain ⟨j, hj, rfl⟩ := hmemL l hl
    rw [hBj j hj, hdj j hj, hlab j hj, List.length_map, cidx_length]
  · obtain ⟨j, hj, rfl⟩ := hmemL l hl
    rw [hBj j hj] at hx
    obtain ⟨t, -, rfl⟩ := List.mem_map.1 hx
    rw [hdj j hj, dport_dlab]
  · -- the integrands
    set ν : ℕ → ℕ := fun x => if x ∈ D.rdl E.mult then τ' (D.pinD x) else τ' x
    have hτB : ∀ i (hi : i < D.bnd.length), τ i = β i := fun i hi => hτ i fun h => by
      obtain ⟨j, -, e⟩ := hmemL i h; omega
    have hw1 : ∀ j (hj : j < D.wires.length), ∀ u ∈ cidx E.mult (D.legChain D.wires[j].1),
        τ' (dlab D.wires[j].1 u) = digit (E.dimv (D.portDim D.wires[j].1))
          (pos E.mult (D.legChain D.wires[j].1) u) (τ (off + j)) := fun j hj u hu => by
      have hm : off + j ∈ (List.range D.wires.length).map (off + ·) :=
        List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩
      have := hdg _ hm (dlab D.wires[j].1 u) (by rw [hBj j hj]; exact List.mem_map.2 ⟨u, hu, rfl⟩)
      rw [this, hBj j hj, hdj j hj, idxOf_cidx _ _ _ _ hu]
    have hwr : ∀ j (hj : j < D.wires.length), ∀ u ∈ cidx E.mult (D.legChain D.wires[j].1),
        dlab D.wires[j].1 u ∈ D.wdl E.mult := fun j hj u hu =>
      (D.mem_wdl _ _).2 ⟨_, List.getElem_mem hj, u, hu, rfl⟩
    have key : ∀ p ∈ D.legs, (τ (D.label off p) < E.dimv (D.portDim p) ^ size E.mult (D.legChain p)) ∧
        ∀ u ∈ cidx E.mult (D.legChain p), ν (dlab p u) =
          digit (E.dimv (D.portDim p)) (pos E.mult (D.legChain p) u) (τ (D.label off p)) := by
      intro p hp
      have hm : off + 0 = off := rfl
      rcases List.mem_append.1 (hW.perm.mem_iff.2 hp) with hp1 | hp23
      · obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hp1
        simp only [List.getElem_map, List.length_map] at hj ⊢
        rw [hlab1 j hj]
        refine ⟨?_, fun u hu => ?_⟩
        · have := hr _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩)
          rwa [hlab j hj] at this
        · have hnr : dlab D.wires[j].1 u ∉ D.rdl E.mult := fun h => hdis _ (hwr j hj u hu) _ h rfl
          simp only [ν, if_neg hnr]
          exact hw1 j hj u hu
      rcases List.mem_append.1 hp23 with hp2 | hp3
      · obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hp2
        simp only [List.getElem_map, List.length_map] at hj ⊢
        have hwm := List.getElem_mem hj
        rw [hlab2 j hj, ← legChain_wire hW hwm, ← hW.wdim _ hwm]
        refine ⟨?_, fun u hu => ?_⟩
        · have := hr _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩)
          rwa [hlab j hj] at this
        · have hrm : dlab D.wires[j].2 u ∈ D.rdl E.mult := (D.mem_rdl _ _).2 ⟨_,
            List.mem_append_left _ (List.mem_map.2 ⟨_, hwm, rfl⟩), u,
            by rw [← legChain_wire hW hwm]; exact hu, rfl⟩
          simp only [ν, if_pos hrm, pinD_wire hW hwm]
          exact hw1 j hj u hu
      · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp3
        rw [hlabB i hi, hτB i hi]
        refine ⟨hβ i hi, fun u hu => ?_⟩
        have hrm : dlab D.bnd[i] u ∈ D.rdl E.mult := (D.mem_rdl _ _).2 ⟨_,
          List.mem_append_right _ (List.getElem_mem hi), u, hu, rfl⟩
        simp only [ν, if_pos hrm, pinD_bnd hW i hi]
        rw [hτ' _ fun h => by
          obtain ⟨w, -, v, -, e⟩ := (D.mem_wdl _ _).1 (hflat ▸ h)
          exact dlab_ne_slab _ v i u e.symm, slotsD_slab]
        simp [slotDig, List.getElem?_eq_getElem hi]
    unfold prodVal prodD
    rw [List.map_map]
    congr 1
    refine List.map_congr_left fun n hn => ?_
    have hleg : ∀ p ∈ n.legs, p ∈ D.legs := fun p hp => List.mem_flatMap.2 ⟨n, hn, hp⟩
    exact val_bnode D E off n τ ν ⟨(hN n hn).plain, (hN n hn).boxkind, (hN n hn).tailkind,
      (hN n hn).dims, (hN n hn).posd, (hN n hn).posn, fun p hp => (key p (hleg p hp)).1,
      fun p hp => (key p (hleg p hp)).2⟩

/-- The family's network at a bundled boundary assignment is the digit sum at its digits. -/
theorem eval_toNetB_evalD (hW : WF D) (E : Env) (off : ℕ) (hoff : D.bnd.length ≤ off)
    (hN : ∀ n ∈ D.nodes, D.NodeS E n)
    (hlab : ∀ j (hj : j < D.wires.length), labDim E (off + j) =
      E.dimv (D.portDim D.wires[j].1) ^ size E.mult (D.legChain D.wires[j].1))
    (β : ℕ → ℕ) (hβ : ∀ i (hi : i < D.bnd.length),
      β i < E.dimv (D.portDim D.bnd[i]) ^ size E.mult (D.legChain D.bnd[i])) :
    ArchytasZX.eval E (D.toNetB off) β = D.evalD E E.mult (D.slotDig E β) := by
  rw [eval_toNetB hW E off hoff hN hlab β hβ, evalD_pin hW E E.mult _ fun i hi t _ => ?_]
  have hp : D.bnd[i] ∈ D.legs := bnd_legs hW (List.getElem_mem hi)
  obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 hp
  simp only [slotDig, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
  exact digit_lt _ _ _ ((hN n hn).posd _ hpn)

end Diagram

end ArchytasZX
