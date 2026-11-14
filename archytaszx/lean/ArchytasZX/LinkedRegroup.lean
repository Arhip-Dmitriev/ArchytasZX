/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Linked

/-!
# Slot layouts of box groups

A slot block `(c, is)` lays out the slots `is`, all of chain `c`, copy-major: for each copy index
`t` of `c`, every slot of `is` at `t`. A layout `G` is a list of blocks; `layout ks G` lists the
instantiated slots `(i, t)` block by block, and position `p` of a flat boundary assignment reads
the slot digit at `(layout ks G)[p]` (`readAt`). `Covers L G`: every slot of `L` lies in exactly one
block of `G`, of its leg's chain. `regroup_layout`: two covering layouts list the same slots in
another order, the position map `lperm` is one to one and onto, and `L` read through the one is
`L` read through the other after `lperm`.
-/

namespace ArchytasZX

/-- A slot block: a chain of multiplicity indices and its slots. -/
abbrev Blk := List ℕ × List ℕ

/-- The instantiated slots of the blocks `G`, block by block, each copy-major. -/
def layout (ks : ℕ → ℕ) (G : List Blk) : List (ℕ × List ℕ) :=
  G.flatMap fun b => (cidx ks b.1).flatMap fun t => b.2.map (·, t)

/-- Slot digits read from the flat assignment `x` at their positions in `lay`. -/
def readAt (lay : List (ℕ × List ℕ)) (x : ℕ → ℕ) (i : ℕ) (t : List ℕ) : ℕ :=
  x (lay.idxOf (i, t))

/-- Position `p` of `lay` to the position of its slot in `lay'`; past `lay`, `p` itself. -/
def lperm (lay lay' : List (ℕ × List ℕ)) (p : ℕ) : ℕ :=
  if h : p < lay.length then lay'.idxOf lay[p] else p

/-- Every block slot a boundary slot of its block's chain, every boundary slot in a block, no
slot twice. -/
def Covers (L : LDiagram) (G : List Blk) : Prop :=
  (∀ b ∈ G, ∀ i ∈ b.2, (L.base.bnd[i]?).map L.base.legChain = some b.1) ∧
    (∀ i < L.base.bnd.length, ∃ b ∈ G, i ∈ b.2) ∧ (G.flatMap (·.2)).Nodup

instance (L : LDiagram) (G : List Blk) : Decidable (Covers L G) := by
  unfold Covers; infer_instance

theorem mem_layout (ks : ℕ → ℕ) (G : List Blk) (i : ℕ) (t : List ℕ) :
    (i, t) ∈ layout ks G ↔ ∃ b ∈ G, t ∈ cidx ks b.1 ∧ i ∈ b.2 := by
  simp only [layout, List.mem_flatMap, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨b, hb, u, hu, j, hj, rfl, rfl⟩
    exact ⟨b, hb, hu, hj⟩
  · rintro ⟨b, hb, hu, hj⟩
    exact ⟨b, hb, t, hu, i, hj, rfl, rfl⟩

/-- A covering layout lists exactly the boundary digits. -/
theorem mem_layout_covers {L : LDiagram} {G : List Blk} (hG : Covers L G) (ks : ℕ → ℕ) (i : ℕ)
    (t : List ℕ) :
    (i, t) ∈ layout ks G ↔ ∃ p, L.base.bnd[i]? = some p ∧ t ∈ cidx ks (L.base.legChain p) := by
  rw [mem_layout]
  constructor
  · rintro ⟨b, hb, ht, hi⟩
    have h := hG.1 b hb i hi
    obtain ⟨p, hp, hc⟩ := Option.map_eq_some_iff.1 h
    exact ⟨p, hp, hc ▸ ht⟩
  · rintro ⟨p, hp, ht⟩
    have hlt : i < L.base.bnd.length := (List.getElem?_eq_some_iff.1 hp).1
    obtain ⟨b, hb, hi⟩ := hG.2.1 i hlt
    have h := hG.1 b hb i hi
    rw [hp, Option.map_some, Option.some_inj] at h
    exact ⟨b, hb, h ▸ ht, hi⟩

theorem layout_nodup {L : LDiagram} {G : List Blk} (hG : Covers L G) (ks : ℕ → ℕ) :
    (layout ks G).Nodup := by
  obtain ⟨hn, hd⟩ := List.nodup_flatMap.1 hG.2.2
  refine List.nodup_flatMap.2 ⟨fun b hb => ?_, hd.imp fun {b b'} h => ?_⟩
  · refine List.nodup_flatMap.2 ⟨fun t _ => (hn b hb).map fun i j e => by simpa using e, ?_⟩
    refine (cidx_nodup ks b.1).pairwise_of_forall_ne fun t _ u _ htu => ?_
    simp only [Function.onFun, List.disjoint_left, List.mem_map]
    rintro _ ⟨i, -, rfl⟩ ⟨j, -, e⟩
    exact htu (Prod.mk.inj e).2.symm
  · simp only [Function.onFun, List.disjoint_left, List.mem_flatMap, List.mem_map] at h ⊢
    rintro _ ⟨t, -, i, hi, rfl⟩ ⟨u, -, j, hj, e⟩
    obtain ⟨rfl, -⟩ := Prod.mk.inj e
    exact h hi hj

/-- Reading through a permuted layout is reading through the other after `lperm`. -/
theorem readAt_perm {lay lay' : List (ℕ × List ℕ)} (h : lay'.Perm lay) (x : ℕ → ℕ) :
    readAt lay' x = readAt lay (x ∘ lperm lay lay') := by
  funext i t
  simp only [readAt, Function.comp_apply, lperm]
  by_cases hm : (i, t) ∈ lay
  · rw [dite_eq_left_of_eq_true (eq_true (List.idxOf_lt_length_of_mem hm)),
      List.getElem_idxOf]
  · rw [dite_eq_right_of_eq_false (eq_false (by rw [List.idxOf_eq_length hm]; omega)),
      List.idxOf_eq_length hm, List.idxOf_eq_length (fun h' => hm (h.mem_iff.1 h')),
      h.length_eq]

theorem lperm_bijective {lay lay' : List (ℕ × List ℕ)} (h : lay'.Perm lay) (hn : lay.Nodup)
    (hn' : lay'.Nodup) : Function.Bijective (lperm lay lay') := by
  have hl := h.length_eq
  have lt : ∀ p (hp : p < lay.length), lay'.idxOf lay[p] < lay.length := fun p hp => by
    rw [← hl]; exact List.idxOf_lt_length_of_mem (h.mem_iff.2 (List.getElem_mem hp))
  constructor
  · intro p q e
    unfold lperm at e
    split_ifs at e with hp hq hq
    · have e' := (List.idxOf_inj (h.mem_iff.2 (List.getElem_mem hp))).1 e
      exact (List.Nodup.getElem_inj_iff hn).1 e'
    · exact absurd (e ▸ lt p hp) hq
    · exact absurd (e ▸ lt q hq) hp
    · exact e
  · intro q
    by_cases hq : q < lay.length
    · have hm : lay'[q]'(hl ▸ hq) ∈ lay := h.mem_iff.1 (List.getElem_mem _)
      refine ⟨lay.idxOf lay'[q], ?_⟩
      rw [lperm, dite_eq_left_of_eq_true (eq_true (List.idxOf_lt_length_of_mem hm)),
        List.getElem_idxOf]
      exact List.Nodup.idxOf_getElem hn' _ _
    · exact ⟨q, by rw [lperm, dite_eq_right_of_eq_false (eq_false hq)]⟩

/-- Two covering layouts of `L`: the one is a permutation of the other's slots, `lperm` is one to
one and onto, and `L` read through the one is `L` read through the other after `lperm`. -/
theorem regroup_layout (L : LDiagram) (G G' : List Blk) (hG : Covers L G) (hG' : Covers L G')
    (E : Env) (ks : ℕ → ℕ) (x : ℕ → ℕ) :
    (layout ks G').Perm (layout ks G) ∧ Function.Bijective (lperm (layout ks G) (layout ks G')) ∧
      L.evalL E ks (readAt (layout ks G') x) =
        L.evalL E ks (readAt (layout ks G) (x ∘ lperm (layout ks G) (layout ks G'))) := by
  have hp : (layout ks G').Perm (layout ks G) := by
    refine (List.perm_ext_iff_of_nodup (layout_nodup hG' ks) (layout_nodup hG ks)).2 ?_
    rintro ⟨i, t⟩
    rw [mem_layout_covers hG', mem_layout_covers hG]
  exact ⟨hp, lperm_bijective hp (layout_nodup hG ks) (layout_nodup hG' ks),
    by rw [readAt_perm hp]⟩

end ArchytasZX
