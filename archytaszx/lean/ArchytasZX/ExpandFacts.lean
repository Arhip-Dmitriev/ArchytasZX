/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Digits

/-!
# Facts about the image form of the expansion

Positions after growing legs in place, the legs of grown nodes as images of the old legs.
-/

namespace ArchytasZX

/-- The first `n` widths summed. -/
def wsum (w : ℕ → ℕ) (n : ℕ) : ℕ := ((List.range n).map w).sum

theorem wsum_succ (w : ℕ → ℕ) (n : ℕ) : wsum w (n + 1) = wsum w n + w n := by
  simp [wsum, List.range_succ]

/-- Consecutive blocks of widths `w` fill the range of their total. -/
theorem range_blocks (w : ℕ → ℕ) (n : ℕ) :
    (List.range n).flatMap (fun i => (List.range (w i)).map (wsum w i + ·)) =
      List.range (wsum w n) := by
  induction n with
  | zero => simp [wsum]
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append, ih, wsum_succ, List.range_add]
    simp

theorem find?_unique {α : Type*} (l : List α) (q : α → Bool) (b : α) (hb : b ∈ l) (hq : q b)
    (hu : ∀ y ∈ l, q y → y = b) : l.find? q = some b := by
  induction l with
  | nil => simp at hb
  | cons a l ih =>
    by_cases ha : q a
    · simp only [List.find?_cons, ha]; rw [hu a List.mem_cons_self ha]
    · simp only [List.find?_cons, ha]
      rcases List.mem_cons.1 hb with rfl | hb
      · exact absurd hq ha
      · exact ih hb fun y hy => hu y (List.mem_cons_of_mem _ hy)

theorem find?_none {α : Type*} (l : List α) (q : α → Bool) (h : ∀ y ∈ l, q y = false) :
    l.find? q = none := List.find?_eq_none.2 fun y hy => by simp [h y hy]

namespace Diagram

/-- The walk is the chain `L` when `L` holds every box satisfying `p` whose parent is a start
of the chain, each box's parent is the previous box, and the starts are distinct. -/
theorem walk_eq (D : Diagram) (p : DBox → Bool) (L : List DBox) (start : Option ℕ) (fuel : ℕ)
    (hf : L.length ≤ fuel) (hsub : ∀ y ∈ L, y ∈ D.boxes ∧ p y = true)
    (hall : ∀ y ∈ D.boxes, p y = true → y.parent ∈ start :: L.map (fun b => some b.id) → y ∈ L)
    (hpar : ∀ i (h : i < L.length), L[i].parent = (start :: L.map (fun b => some b.id))[i]'(by simp; omega))
    (hnd : (start :: L.map (fun b => some b.id)).Nodup) :
    D.walk p fuel start = L := by
  induction L generalizing start fuel with
  | nil =>
    cases fuel with
    | zero => rfl
    | succ f =>
    simp only [walk]
    rw [find?_none _ _ fun y hy => by
        by_contra h
        simp only [Bool.not_eq_false, Bool.and_eq_true, beq_iff_eq] at h
        exact absurd (hall y hy h.1 (by simp [h.2])) (by simp)]
  | cons b L ih =>
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    have hb0 : b.parent = start := by
      have h := hpar 0 (by simp)
      simp only [List.getElem_cons_zero] at h
      exact h
    simp only [walk]
    rw [find?_unique _ _ b (hsub b List.mem_cons_self).1 (by
        simp [(hsub b List.mem_cons_self).2, hb0]) fun y hy hq => by
      simp only [Bool.and_eq_true, beq_iff_eq] at hq
      have hy' := hall y hy hq.1 (by simp [hq.2])
      rcases List.mem_cons.1 hy' with rfl | hy'
      · rfl
      · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hy'
        have h1 := hpar (i + 1) (by simp; omega)
        simp only [List.getElem_cons_succ, List.map_cons, List.getElem_cons_succ] at h1
        rw [hq.2] at h1
        have := List.Nodup.getElem_inj_iff hnd (i := 0) (j := i + 1) (hi := by simp)
          (hj := by simp; omega)
        simp only [List.getElem_cons_zero, List.getElem_cons_succ] at this
        exact absurd (this.1 (by rw [h1]; simp)) (by omega)]
    simp only [List.cons.injEq, true_and]
    simp only [List.map_cons, List.nodup_cons, List.mem_cons, List.mem_map, not_or] at hnd
    refine ih (some b.id) f (by simp at hf; omega)
      (fun y hy => hsub y (List.mem_cons_of_mem _ hy)) (fun y hy hq hp => ?_)
      (fun i hi => by
        have h := hpar (i + 1) (by simp; omega)
        simp only [List.getElem_cons_succ] at h
        exact h) (by simpa using hnd.2)
    have hy' := hall y hy hq (by
      simp only [List.map_cons, List.mem_cons] at hp ⊢
      exact Or.inr hp)
    rcases List.mem_cons.1 hy' with rfl | hy'
    · exfalso
      rw [hb0] at hp
      simp only [List.mem_cons, List.mem_map] at hp
      rcases hp with h | ⟨a, ha, h⟩
      · exact hnd.1.1 h
      · exact hnd.1.2 ⟨a, ha, h⟩
    · exact hy'

/-- `L` is the chain of boxes satisfying `pr` below `start`, in the sense of `walk_eq`. -/
def ChainOK (D : Diagram) (pr : DBox → Bool) (start : Option ℕ) (L : List DBox) : Prop :=
  (∀ y ∈ L, y ∈ D.boxes ∧ pr y = true) ∧
  (∀ y ∈ D.boxes, pr y = true → y.parent ∈ start :: L.map (fun b => some b.id) → y ∈ L) ∧
  (∀ i (h : i < L.length), L[i].parent = (start :: L.map (fun b => some b.id))[i]'(by simp; omega)) ∧
  (start :: L.map (fun b => some b.id)).Nodup

theorem ChainOK.length_le {D : Diagram} {pr : DBox → Bool} {start : Option ℕ} {L : List DBox}
    (h : ChainOK D pr start L) : L.length ≤ D.boxes.length := by
  have hL : L.Nodup := by
    have := (List.nodup_cons.1 h.2.2.2).2
    exact List.Nodup.of_map _ this
  exact List.Nodup.length_le_of_subset hL fun y hy => (h.1 y hy).1
  
theorem ChainOK.walk {D : Diagram} {pr : DBox → Bool} {start : Option ℕ} {L : List DBox}
    (h : ChainOK D pr start L) (fuel : ℕ) (hf : L.length ≤ fuel) : D.walk pr fuel start = L :=
  walk_eq D pr L start fuel hf h.1 h.2.1 h.2.2.1 h.2.2.2

theorem filter_last {α : Type*} (L : List α) (keep : α → Bool)
    (hlast : ∀ i (h : i < L.length), keep L[i] = false → i + 1 = L.length) :
    ∃ n ≤ L.length, L.filter keep = L.take n := by
  induction L with
  | nil => exact ⟨0, le_refl _, rfl⟩
  | cons a L ih =>
    by_cases ha : keep a = true
    · obtain ⟨n, hn, e⟩ := ih fun i hi hk => by
        have := hlast (i + 1) (by simp; omega) (by simpa using hk); simp at this; omega
      exact ⟨n + 1, by simp; omega, by simp [ha, e]⟩
    · have := hlast 0 (by simp) (by simpa using ha)
      simp only [List.length_cons, zero_add] at this
      have h0 : L.length = 0 := by omega
      rw [List.length_eq_zero_iff] at h0
      subst h0
      exact ⟨0, by simp, by simp [ha]⟩

/-- A chain's prefix survives a change of boxes that keeps ids, parents and `pr` and keeps no box
it did not have. -/
theorem ChainOK.transport {D D' : Diagram} {pr pr' : DBox → Bool} {start : Option ℕ}
    {L : List DBox} (f : DBox → DBox) (keep : DBox → Bool)
    (hD' : ∀ y', y' ∈ D'.boxes ↔ ∃ y ∈ D.boxes, keep y = true ∧ y' = f y ∧
      ((f y).nodes ≠ [] ∨ (f y).ports ≠ []))
    (hf : ∀ y ∈ D.boxes, (f y).id = y.id ∧ (f y).parent = y.parent ∧ pr' (f y) = pr y)
    (hne : ∀ y, pr' y = true → y.nodes ≠ [] ∨ y.ports ≠ [])
    (hlast : ∀ i (h : i < L.length), keep L[i] = false → i + 1 = L.length)
    (h : ChainOK D pr start L) : ChainOK D' pr' start ((L.filter keep).map f) := by
  obtain ⟨hsub, hall, hpar, hnd⟩ := h
  obtain ⟨n, hn, e⟩ := filter_last L keep hlast
  have hids : ((L.take n).map f).map (fun b => some b.id) = (L.take n).map (fun b => some b.id) := by
    rw [List.map_map]
    exact List.map_congr_left fun y hy => by
      simp [(hf y (hsub y (List.mem_of_mem_take hy)).1).1]
  have hsl : (start :: (L.take n).map (fun b => some b.id)).Sublist
      (start :: L.map (fun b => some b.id)) :=
    ((List.take_sublist n L).map _).cons_cons _
  rw [e]
  refine ⟨fun y' hy' => ?_, fun y' hy' hp hpa => ?_, fun i hi => ?_, ?_⟩
  · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hy'
    have hyL := List.mem_of_mem_take hy
    have hk : keep y = true := by
      have : y ∈ L.filter keep := e ▸ hy
      exact (List.mem_filter.1 this).2
    have hpr : pr' (f y) = true := by rw [(hf y (hsub y hyL).1).2.2]; exact (hsub y hyL).2
    exact ⟨(hD' _).2 ⟨y, (hsub y hyL).1, hk, rfl, hne _ hpr⟩, hpr⟩
  · obtain ⟨y, hy, hk, rfl, -⟩ := (hD' y').1 hy'
    rw [(hf y hy).2.2] at hp
    rw [hids, (hf y hy).2.1] at hpa
    have hyL := hall y hy hp (hsl.subset hpa)
    have : y ∈ L.filter keep := List.mem_filter.2 ⟨hyL, hk⟩
    rw [e] at this
    exact List.mem_map.2 ⟨y, this, rfl⟩
  · have hi' : i < (L.take n).length := by simpa using hi
    have hin : i < L.length := by simp at hi'; omega
    have hp := hpar i hin
    have hmem := (hsub _ (List.getElem_mem hin)).1
    cases i with
    | zero => simpa [(hf _ hmem).2.1] using hp
    | succ i =>
      have hmem' := (hsub _ (List.getElem_mem (show i < L.length by omega))).1
      simpa [(hf _ hmem).2.1, (hf _ hmem').1, List.getElem_take] using hp
  · rw [hids]; exact hnd.sublist hsl

theorem shiftPos_eq (g : Port → Bool) (k x : ℕ) (out : Bool) (i : ℕ) :
    shiftPos g k ⟨x, out, i⟩ = wsum (fun i' => if g ⟨x, out, i'⟩ then k else 1) i := by
  unfold shiftPos
  simp only
  induction i with
  | zero => simp [wsum]
  | succ i ih =>
    rw [wsum_succ, ← ih, List.range_succ, List.filter_append]
    have hle : ((List.range i).filter fun i' => g ⟨x, out, i'⟩).length ≤ i := by
      calc _ ≤ (List.range i).length := List.length_filter_le _ _
        _ = i := List.length_range
    by_cases h : g ⟨x, out, i⟩ <;> simp [h, Nat.mul_succ] <;> omega

theorem growSide_length (g : Port → Bool) (k x : ℕ) (out : Bool) (ds : List ℕ) :
    (growSide g k x out ds).length = wsum (fun i' => if g ⟨x, out, i'⟩ then k else 1) ds.length := by
  unfold growSide
  induction ds using List.reverseRecOn with
  | nil => simp [wsum]
  | append_singleton ds d ih =>
    rw [List.zipIdx_append, List.flatMap_append, List.length_append, ih]
    simp only [List.length_append, List.length_singleton, wsum_succ]
    by_cases h : g ⟨x, out, ds.length⟩ <;> simp [h]

/-- A grown side's legs are the images of the old side's legs, block by block. -/
theorem side_grow (g : Port → Bool) (k x : ℕ) (out : Bool) (len : ℕ) :
    (List.range (wsum (fun i' => if g ⟨x, out, i'⟩ then k else 1) len)).map
        (fun i => (⟨x, out, i⟩ : Port)) =
      ((List.range len).map fun i => (⟨x, out, i⟩ : Port)).flatMap (growImg g k) := by
  rw [← range_blocks, List.map_flatMap, List.flatMap_map]
  refine List.flatMap_congr fun i _ => ?_
  unfold growImg
  rw [shiftPos_eq]
  by_cases h : g ⟨x, out, i⟩ <;> simp [h]

/-- A grown node's legs are the images of its legs. -/
theorem legs_grow (g : Port → Bool) (k : ℕ) (n : DNode) :
    DNode.legs { n with outs := growSide g k n.id true n.outs, ins := growSide g k n.id false n.ins } =
      n.legs.flatMap (growImg g k) := by
  simp only [DNode.legs, List.flatMap_append, growSide_length, side_grow]

end Diagram

end ArchytasZX
