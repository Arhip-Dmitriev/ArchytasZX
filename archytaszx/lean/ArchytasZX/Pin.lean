/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.NodeBridge

/-!
# Pins of the digit sum

`evalD_pin`: the digit sum is the sum over the digits of the wires' first ends alone, every other
digit read through `pinD`: a wire's second end at its first end's digit, a boundary leg at its
slot's digit.
-/

namespace ArchytasZX

/-- The copy index of a digit label. -/
def dtl (l : ℕ) : List ℕ := ((Encodable.decode (l / 2).unpair.2 : Option (List ℕ))).getD []

theorem dtl_dlab (p : Port) (t : List ℕ) : dtl (dlab p t) = t := by
  have h2 : dlab p t / 2 = Nat.pair p.code (Encodable.encode t) := by simp only [dlab]; omega
  simp [dtl, h2, Nat.unpair_pair]

namespace Diagram

variable (D : Diagram)

/-- The digits of the wires' first ends. -/
def wdl (ks : ℕ → ℕ) : List ℕ := D.wires.flatMap fun w => (cidx ks (D.legChain w.1)).map (dlab w.1)

/-- The digits of the wires' second ends and of the boundary legs. -/
def rdl (ks : ℕ → ℕ) : List ℕ :=
  (D.wires.map (·.2) ++ D.bnd).flatMap fun p => (cidx ks (D.legChain p)).map (dlab p)

/-- A boundary leg's digit to its slot's, any other leg's to its partner's. -/
def pinD (l : ℕ) : ℕ :=
  match D.bnd.idxOf? (dport l) with
  | some i => slab i (dtl l)
  | none => dlab ((D.partner (dport l)).getD (dport l)) (dtl l)

theorem idxOf?_get {l : List Port} (hnd : l.Nodup) (i : ℕ) (hi : i < l.length) :
    l.idxOf? l[i] = some i := by
  rw [List.idxOf?_eq_some_iff]
  exact ⟨hi, rfl, fun j hj e => by
    have := (List.Nodup.getElem_inj_iff hnd (i := j) (j := i) (hi := by omega) (hj := hi)).1 e
    omega⟩

theorem mem_rdl (ks : ℕ → ℕ) (x : ℕ) : x ∈ D.rdl ks ↔
    ∃ p ∈ D.wires.map (·.2) ++ D.bnd, ∃ t ∈ cidx ks (D.legChain p), x = dlab p t := by
  simp only [rdl, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨p, hp, t, ht, rfl⟩; exact ⟨p, hp, t, ht, rfl⟩
  · rintro ⟨p, hp, t, ht, rfl⟩; exact ⟨p, hp, t, ht, rfl⟩

theorem mem_wdl (ks : ℕ → ℕ) (x : ℕ) : x ∈ D.wdl ks ↔
    ∃ w ∈ D.wires, ∃ t ∈ cidx ks (D.legChain w.1), x = dlab w.1 t := by
  simp only [wdl, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨w, hw, t, ht, rfl⟩; exact ⟨w, hw, t, ht, rfl⟩
  · rintro ⟨w, hw, t, ht, rfl⟩; exact ⟨w, hw, t, ht, rfl⟩

variable {D}

theorem pinD_wire (hW : WF D) {w : Port × Port} (hw : w ∈ D.wires) (t : List ℕ) :
    D.pinD (dlab w.2 t) = dlab w.1 t := by
  have hb : D.bnd.idxOf? w.2 = none := List.idxOf?_eq_none_iff.2 (wire_not_bnd hW hw).2.1
  simp [pinD, dport_dlab, dtl_dlab, hb, (partner_of_wire hW hw).2]

theorem pinD_bnd (hW : WF D) (i : ℕ) (hi : i < D.bnd.length) (t : List ℕ) :
    D.pinD (dlab D.bnd[i] t) = slab i t := by
  have hnd : D.bnd.Nodup := by
    have := hW.perm.nodup_iff.2 (legs_nodup_of D hW.ids)
    exact (List.nodup_append.1 (List.nodup_append.1 this).2.1).2.1
  simp [pinD, dport_dlab, dtl_dlab, idxOf?_get hnd i hi]

theorem dlabs_perm (hW : WF D) (ks : ℕ → ℕ) : (D.dlabs ks).Perm (D.wdl ks ++ D.rdl ks) := by
  unfold dlabs wdl rdl
  refine (hW.perm.symm.flatMap_right _).trans ?_
  rw [List.flatMap_append, List.flatMap_map]

/-- The digit sum over the first ends' digits, the others pinned. -/
theorem evalD_pin (hW : WF D) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < D.bnd.length), ∀ t ∈ cidx ks (D.legChain D.bnd[i]),
      σ i t < E.dimv (D.portDim D.bnd[i])) :
    D.evalD E ks σ = sumOver (fun l => E.dimv (D.portDim (dport l))) (D.wdl ks)
      (fun τ => D.prodD E ks fun x => if x ∈ D.rdl ks then τ (D.pinD x) else τ x) (slotsD σ) := by
  have hnd := (dlabs_perm hW ks).nodup_iff.1 (dlabs_nodup D ks (legs_nodup_of D hW.ids))
  obtain ⟨-, hrnd, hdis⟩ := List.nodup_append.1 hnd
  have hbnd : D.bnd.Nodup := by
    have := hW.perm.nodup_iff.2 (legs_nodup_of D hW.ids)
    exact (List.nodup_append.1 (List.nodup_append.1 this).2.1).2.1
  have hpins : ∀ ν : ℕ → ℕ, D.pinsD ks ν ↔ ∀ l ∈ D.rdl ks, ν l = ν (D.pinD l) := by
    intro ν
    constructor
    · rintro ⟨hw, hb⟩ l hl
      obtain ⟨p, hp, t, ht, rfl⟩ := (D.mem_rdl ks l).1 hl
      rcases List.mem_append.1 hp with hp | hp
      · obtain ⟨w, hwm, rfl⟩ := List.mem_map.1 hp
        rw [pinD_wire hW hwm]
        exact (hw w hwm t (by rw [legChain_wire hW hwm]; exact ht)).symm
      · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
        rw [pinD_bnd hW i hi]
        exact hb (D.bnd[i], i) ((mem_zipIdx' _ _ _).2 ⟨hi, rfl⟩) t ht
    · intro h
      refine ⟨fun w hwm t ht => ?_, fun q hq t ht => ?_⟩
      · have hl : dlab w.2 t ∈ D.rdl ks := (D.mem_rdl ks _).2 ⟨w.2,
          List.mem_append_left _ (List.mem_map.2 ⟨w, hwm, rfl⟩), t,
          by rw [← legChain_wire hW hwm]; exact ht, rfl⟩
        have := h _ hl
        rw [pinD_wire hW hwm] at this
        exact this.symm
      · obtain ⟨p, i⟩ := q
        obtain ⟨hi, rfl⟩ := (mem_zipIdx' _ _ _).1 hq
        have hl : dlab D.bnd[i] t ∈ D.rdl ks := (D.mem_rdl ks _).2 ⟨D.bnd[i],
          List.mem_append_right _ (List.getElem_mem hi), t, ht, rfl⟩
        have := h _ hl
        rw [pinD_bnd hW i hi] at this
        exact this
  unfold evalD
  rw [sumOver_perm _ (dlabs_perm hW ks), sumOver_append]
  refine sumOver_congr _ _ _ _ _ fun τ hτ hr => ?_
  classical
  rw [show (fun ν => if D.pinsD ks ν then D.prodD E ks ν else 0) =
      fun ν => if ∀ l ∈ D.rdl ks, ν l = ν (D.pinD l) then D.prodD E ks ν else 0 from
    funext fun ν => if_congr (hpins ν) rfl rfl]
  refine sumOver_pin _ _ hrnd _ (fun l hl => ?_) _ τ (fun l hl => ?_)
  · obtain ⟨p, hp, t, ht, rfl⟩ := (D.mem_rdl ks l).1 hl
    rcases List.mem_append.1 hp with hp | hp
    · obtain ⟨w, hwm, rfl⟩ := List.mem_map.1 hp
      rw [pinD_wire hW hwm]
      exact fun h => hdis _ ((D.mem_wdl ks _).2 ⟨w, hwm, t,
        by rw [legChain_wire hW hwm]; exact ht, rfl⟩) _ h rfl
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
      rw [pinD_bnd hW i hi]
      intro h
      obtain ⟨q, -, u, -, e⟩ := (D.mem_rdl ks _).1 h
      exact dlab_ne_slab q u i t e.symm
  · obtain ⟨p, hp, t, ht, rfl⟩ := (D.mem_rdl ks l).1 hl
    rcases List.mem_append.1 hp with hp | hp
    · obtain ⟨w, hwm, rfl⟩ := List.mem_map.1 hp
      rw [pinD_wire hW hwm]
      have := hr _ ((D.mem_wdl ks _).2 ⟨w, hwm, t, by rw [legChain_wire hW hwm]; exact ht, rfl⟩)
      simpa [dport_dlab, hW.wdim w hwm] using this
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
      rw [pinD_bnd hW i hi, hτ _ fun h => by
        obtain ⟨w, -, u, -, e⟩ := (D.mem_wdl ks _).1 h
        exact dlab_ne_slab _ u i t e.symm, slotsD_slab, dport_dlab]
      exact hσ i hi t ht

end Diagram

end ArchytasZX
