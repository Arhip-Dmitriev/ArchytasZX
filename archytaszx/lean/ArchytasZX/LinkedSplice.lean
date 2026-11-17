/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedGhost
import ArchytasZX.LinkedPeel

/-!
# The flat boundary of an instantiated linked diagram

The engine instantiates a box by splicing, for each of its slot blocks, the block's boundary legs
into one run at the place of its first leg, copy-major (`bangbox._splice_box`,
`_splice_one_block`), on the outputs and on the inputs apart, outermost box first. The resulting
order of the instantiated slots `(i, t)` depends only on the slot data: slot `i`'s anchors
`anch sd i` are, per box holding it (outermost first), the least slot of its block on its side,
then `i`; its key is its anchors interleaved with its copy indices `t` (`key`), and the engine
lists the slots by increasing key. `fpos sd ks i t`, the number of instantiated slots of smaller
key, is slot `i`'s digit `t`'s flat position, and `flatAt sd ks y` reads every slot digit from
the flat assignment `y` there.

`evalL_congr`: the linked digit sum reads only the slot digits of its boundary legs' copies.
`evalL_peel_flat`: under `PeelFlat` (slot images, their chains and their anchor orders checked
slot by slot), a peel keeps the digit sum read through the flat positions of both sides.
`evalL_peel_first`: a peel of the first copy (`evalL_rev`, `evalL_peel` of the reversal,
`evalL_rev`) does the same under `PeelFirst`.
`evalG_flatAt`: equal digit sums and equal slot data give equal digit sums read through the flat
positions.
-/

namespace ArchytasZX

namespace LDiagram

/-- Per box of slot data: id, multiplicity index, parent and slot blocks. -/
abbrev BoxD := ℕ × ℕ × Option ℕ × List (List ℕ)

/-- The least slot of `P` on the side (outputs below `no`) of the slot `i`, at most `i`. -/
def anchorOf (no i : ℕ) (P : List ℕ) : ℕ :=
  (P.filter fun j => decide (j < no) = decide (i < no)).foldl min i

/-- Slot `i`'s anchors and multiplicity indices in the boxes holding it, from the box of parent
`p` down, then `i`. -/
def walkS (no i : ℕ) (bs : List BoxD) : ℕ → Option ℕ → List ℕ × List ℕ
  | 0, _ => ([i], [])
  | f + 1, p =>
    match bs.find? (fun b => b.2.2.1 == p && b.2.2.2.any fun P => P.contains i) with
    | none => ([i], [])
    | some b =>
      (anchorOf no i ((b.2.2.2.find? fun P => P.contains i).getD []) ::
          (walkS no i bs f (some b.1)).1,
        b.2.1 :: (walkS no i bs f (some b.1)).2)

theorem walkS_length (no i : ℕ) (bs : List BoxD) (f : ℕ) (p : Option ℕ) :
    (walkS no i bs f p).1.length = (walkS no i bs f p).2.length + 1 := by
  induction f generalizing p with
  | zero => rfl
  | succ f ih =>
    simp only [walkS]
    split
    · rfl
    · simp only [List.length_cons, ih]

/-- The anchors of slot `i`. -/
def anch (sd : SlotData) (i : ℕ) : List ℕ := (walkS sd.1 i sd.2.2 sd.2.2.length none).1

/-- The multiplicity indices of the boxes holding slot `i`, outermost first. -/
def chainS (sd : SlotData) (i : ℕ) : List ℕ := (walkS sd.1 i sd.2.2 sd.2.2.length none).2

theorem anch_length (sd : SlotData) (i : ℕ) : (anch sd i).length = (chainS sd i).length + 1 :=
  walkS_length _ _ _ _ _

/-- `a₀ j₀ a₁ j₁ …`: anchors interleaved with copy indices. -/
def ileave : List ℕ → List ℕ → List ℕ
  | a :: as, j :: js => a :: j :: ileave as js
  | as, _ => as

/-- Slot `i`'s digit `t`'s key. -/
def key (sd : SlotData) (i : ℕ) (t : List ℕ) : List ℕ := ileave (anch sd i) t

/-- The instantiated slots: each slot with each copy index of its chain. -/
def slots (sd : SlotData) (ks : ℕ → ℕ) : List (ℕ × List ℕ) :=
  (List.range (sd.1 + sd.2.1)).flatMap fun i => (cidx ks (chainS sd i)).map (i, ·)

/-- Slot `i`'s digit `t`'s flat position: the number of instantiated slots of smaller key. -/
def fpos (sd : SlotData) (ks : ℕ → ℕ) (i : ℕ) (t : List ℕ) : ℕ :=
  ((slots sd ks).filter fun v => decide (List.Lex (· < ·) (key sd v.1 v.2) (key sd i t))).length

/-- Slot digits read from the flat assignment `y` at their flat positions. -/
def flatAt (sd : SlotData) (ks : ℕ → ℕ) (y : ℕ → ℕ) (i : ℕ) (t : List ℕ) : ℕ :=
  y (fpos sd ks i t)

theorem mem_slots (sd : SlotData) (ks : ℕ → ℕ) (i : ℕ) (t : List ℕ) :
    (i, t) ∈ slots sd ks ↔ i < sd.1 + sd.2.1 ∧ t ∈ cidx ks (chainS sd i) := by
  simp only [slots, List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨j, hj, u, hu, rfl, rfl⟩
    exact ⟨hj, hu⟩
  · rintro ⟨hi, ht⟩
    exact ⟨i, hi, t, ht, rfl, rfl⟩

theorem slots_nodup (sd : SlotData) (ks : ℕ → ℕ) : (slots sd ks).Nodup := by
  unfold slots
  refine List.nodup_flatMap.2 ⟨fun i _ => (cidx_nodup ks _).map fun a b e => by simpa using e, ?_⟩
  refine List.nodup_range.pairwise_of_forall_ne fun i _ j _ hij => ?_
  simp only [Function.onFun, List.disjoint_left, List.mem_map]
  rintro _ ⟨a, -, rfl⟩ ⟨b, -, e⟩
  exact hij (Prod.mk.inj e).1.symm

theorem length_of_cidx {ks : ℕ → ℕ} {c t : List ℕ} (h : t ∈ cidx ks c) : t.length = c.length :=
  ((mem_cidx ks c t).1 h).length_eq

/-! ## Lexicographic order of interleaved keys -/

theorem lex_cc {a b : ℕ} {l m : List ℕ} :
    List.Lex (· < ·) (a :: l) (b :: m) ↔ a < b ∨ a = b ∧ List.Lex (· < ·) l m := by
  constructor
  · intro h
    cases h with
    | rel h => exact Or.inl h
    | cons h => exact Or.inr ⟨rfl, h⟩
  · rintro (h | ⟨rfl, h⟩)
    · exact List.Lex.rel h
    · exact List.Lex.cons h

theorem not_lex_nil {l : List ℕ} : ¬ List.Lex (· < ·) l [] := by
  intro h
  cases h

theorem lex_nil_cons {a : ℕ} {l : List ℕ} : List.Lex (· < ·) [] (a :: l) := List.Lex.nil

theorem lex_ne_heads {a b : ℕ} {l m : List ℕ} (h : a ≠ b) :
    List.Lex (· < ·) (a :: l) (b :: m) ↔ a < b := by
  rw [lex_cc]
  exact ⟨fun e => e.resolve_right fun e' => h e'.1, Or.inl⟩

/-- `A` against `A'` level by level as `B` against `B'`, down to the first unequal level. -/
def agree : List ℕ → List ℕ → List ℕ → List ℕ → Bool
  | a :: A, a' :: A', b :: B, b' :: B' =>
    (decide (a < a') == decide (b < b')) && (decide (a = a') == decide (b = b')) &&
      (!decide (a = a') || agree A A' B B')
  | _, _, _, _ => true

theorem agree_cons {a a' b b' : ℕ} {A A' B B' : List ℕ}
    (h : agree (a :: A) (a' :: A') (b :: B) (b' :: B') = true) :
    (a < a' ↔ b < b') ∧ (a = a' ↔ b = b') ∧ (a = a' → agree A A' B B' = true) := by
  simp only [agree, Bool.and_eq_true, beq_iff_eq, decide_eq_decide, Bool.or_eq_true,
    Bool.not_eq_true', decide_eq_false_iff_not] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  exact ⟨h1, h2, fun e => h3.resolve_left (not_not.2 e)⟩

theorem cons_of_len {l : List ℕ} {n : ℕ} (h : l.length = n + 1) :
    ∃ a m, l = a :: m ∧ m.length = n := by
  cases l with
  | nil => simp at h
  | cons a m => exact ⟨a, m, rfl, by simpa using h⟩

theorem ileave_cc (a j : ℕ) (A u : List ℕ) : ileave (a :: A) (j :: u) = a :: j :: ileave A u :=
  rfl

theorem ileave_nil (A : List ℕ) : ileave A [] = A := by
  cases A <;> rfl

/-- Interleaved keys compare alike when their anchors agree level by level. -/
theorem agree_lex (t : List ℕ) : ∀ (t' A A' B B' : List ℕ), agree A A' B B' = true →
    A.length = t.length + 1 → A'.length = t'.length + 1 → B.length = A.length →
    B'.length = A'.length →
    (List.Lex (· < ·) (ileave A t) (ileave A' t') ↔
      List.Lex (· < ·) (ileave B t) (ileave B' t')) := by
  induction t with
  | nil =>
    intro t' A A' B B' h hA hA' hB hB'
    rw [hA] at hB
    rw [hA'] at hB'
    obtain ⟨a, A1, rfl, hA1⟩ := cons_of_len hA
    obtain ⟨b, B1, rfl, hB1⟩ := cons_of_len hB
    obtain ⟨a', A1', rfl, hA1'⟩ := cons_of_len hA'
    obtain ⟨b', B1', rfl, hB1'⟩ := cons_of_len hB'
    obtain ⟨h1, h2, -⟩ := agree_cons h
    rw [List.length_eq_zero_iff.1 hA1, List.length_eq_zero_iff.1 hB1]
    cases t' with
    | nil =>
      rw [List.length_eq_zero_iff.1 hA1', List.length_eq_zero_iff.1 hB1']
      simp only [ileave_nil, lex_cc, not_lex_nil, and_false, or_false]
      exact h1
    | cons j' u' =>
      obtain ⟨c1, C1, rfl, -⟩ := cons_of_len hA1'
      obtain ⟨d1, D1, rfl, -⟩ := cons_of_len hB1'
      simp only [ileave_nil, ileave_cc, lex_cc, lex_nil_cons, and_true]
      rw [h1, h2]
  | cons j u ih =>
    intro t' A A' B B' h hA hA' hB hB'
    rw [hA] at hB
    rw [hA'] at hB'
    obtain ⟨a, A1, rfl, hA1⟩ := cons_of_len hA
    obtain ⟨b, B1, rfl, hB1⟩ := cons_of_len hB
    obtain ⟨a', A1', rfl, hA1'⟩ := cons_of_len hA'
    obtain ⟨b', B1', rfl, hB1'⟩ := cons_of_len hB'
    obtain ⟨h1, h2, h3⟩ := agree_cons h
    obtain ⟨a1, A2, rfl, hA2⟩ := cons_of_len hA1
    obtain ⟨b1, B2, rfl, hB2⟩ := cons_of_len hB1
    cases t' with
    | nil =>
      rw [List.length_eq_zero_iff.1 hA1', List.length_eq_zero_iff.1 hB1']
      simp only [ileave_nil, ileave_cc, lex_cc, not_lex_nil, and_false, or_false]
      exact h1
    | cons j' u' =>
      obtain ⟨c1, C1, rfl, hC1⟩ := cons_of_len hA1'
      obtain ⟨d1, D1, rfl, hD1⟩ := cons_of_len hB1'
      have key := fun e => ih u' (a1 :: A2) (c1 :: C1) (b1 :: B2) (d1 :: D1) (h3 e)
        (by simp [hA2]) (by simp [hC1]) (by simp [hB2, hA2]) (by simp [hD1, hC1])
      simp only [ileave_cc, lex_cc]
      constructor
      · rintro (e | ⟨e, f⟩)
        · exact Or.inl (h1.1 e)
        · refine Or.inr ⟨h2.1 e, ?_⟩
          rcases f with f | ⟨rfl, f⟩
          · exact Or.inl f
          · exact Or.inr ⟨rfl, (key e).1 f⟩
      · rintro (e | ⟨e, f⟩)
        · exact Or.inl (h1.2 e)
        · refine Or.inr ⟨h2.2 e, ?_⟩
          rcases f with f | ⟨rfl, f⟩
          · exact Or.inl f
          · exact Or.inr ⟨rfl, (key (h2.2 e)).2 f⟩

/-! ## The linked digit sum reads only the boundary legs' slot digits -/

theorem sumOver_extP (D : ℕ → ℕ) (ls : List ℕ) (S : ℕ → Prop) (f : (ℕ → ℕ) → ℂ)
    (hf : ∀ τ τ', (∀ x, S x → τ x = τ' x) → f τ = f τ') (σ σ' : ℕ → ℕ)
    (h : ∀ x, S x → x ∉ ls → σ x = σ' x) : sumOver D ls f σ = sumOver D ls f σ' := by
  induction ls generalizing σ σ' with
  | nil => exact hf σ σ' fun x hx => h x hx (by simp)
  | cons l ls ih =>
    refine Finset.sum_congr rfl fun j _ => ih _ _ fun x hx hn => ?_
    by_cases hxl : x = l
    · subst hxl; simp
    · rw [Function.update_of_ne hxl, Function.update_of_ne hxl]
      exact h x hx (by simp [hxl, hn])

theorem dlab_even (p : Port) (t : List ℕ) : dlab p t % 2 = 0 := by
  simp only [dlab]; omega

theorem dnodeL_even (L : LDiagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    (∀ x ∈ (L.dnodeL ks n t).outs, x % 2 = 0) ∧ ∀ x ∈ (L.dnodeL ks n t).ins, x % 2 = 0 := by
  constructor <;>
  · intro x hx
    simp only [dnodeL, Diagram.dnode, List.mem_flatMap, List.mem_map] at hx
    obtain ⟨_, _, _, _, rfl⟩ := hx
    exact dlab_even _ _

/-- The linked digit sum reads only the slot digits of its boundary legs' copies. -/
theorem evalL_congr (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (σ σ' : ℕ → List ℕ → ℕ)
    (h : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t = σ' i t) :
    L.evalL E ks σ = L.evalL E ks σ' := by
  unfold evalL
  refine sumOver_extP _ _ (fun x => x % 2 = 0 ∨ ∃ i, ∃ hi : i < L.base.bnd.length,
    ∃ t ∈ cidx ks (L.base.legChain L.base.bnd[i]), x = slab i t) _ (fun τ τ' hτ => ?_) _ _
    (fun x hx _ => ?_)
  · have hd : ∀ p t, τ (dlab p t) = τ' (dlab p t) := fun p t => hτ _ (Or.inl (dlab_even p t))
    have hp : L.pinsL ks τ ↔ L.pinsL ks τ' := by
      unfold pinsL
      refine and_congr (forall₂_congr fun y _ => by rw [hd, hd])
        (forall₂_congr fun q hq => forall₂_congr fun t ht => ?_)
      obtain ⟨p, i⟩ := q
      obtain ⟨hi, e⟩ := (mem_zipIdx' _ _ _).1 hq
      subst e
      rw [hd, hτ _ (Or.inr ⟨i, hi, t, ht, rfl⟩)]
    have hP : L.prodL E ks τ = L.prodL E ks τ' := by
      unfold prodL
      refine congrArg List.prod (List.map_congr_left fun n _ =>
        congrArg List.prod (List.map_congr_left fun t _ => ?_))
      apply val_congr
      · exact List.map_congr_left fun x hx => hτ _ (Or.inl ((dnodeL_even L ks n t).1 x hx))
      · exact List.map_congr_left fun x hx => hτ _ (Or.inl ((dnodeL_even L ks n t).2 x hx))
    rw [hP]
    classical
    exact if_congr hp rfl rfl
  · rcases hx with hx | ⟨i, hi, t, ht, rfl⟩
    · simp [slotsD, hx]
    · rw [slotsD_slab, slotsD_slab]
      exact h i hi t ht

/-! ## Peels keep the flat positions -/

/-- Boundary place `i` of `L`. -/
def bndAt (L : LDiagram) (i : ℕ) : Port := L.base.bnd.getD i Port.dflt

/-- Whether boundary place `i` of `L` is a leg of a node of `B`. -/
def inB (L : LDiagram) (B : List ℕ) (i : ℕ) : Bool := decide ((L.bndAt i).node ∈ B)

/-- The place in `L'` of `L`'s boundary leg `i`. -/
def mI (L L' : LDiagram) (i : ℕ) : ℕ := L'.base.bnd.idxOf (L.bndAt i)

/-- The place in `L'` of the copy of `L`'s boundary leg `i`. -/
def cI (L L' : LDiagram) (cp : ℕ → ℕ) (i : ℕ) : ℕ := L'.base.bnd.idxOf (cpP cp (L.bndAt i))

/-- The places of `L'` from `L`'s: every place, then every copied place. -/
def imgs (L L' : LDiagram) (B : List ℕ) (cp : ℕ → ℕ) : List ℕ :=
  (List.range L.base.bnd.length).map (mI L L') ++
    ((List.range L.base.bnd.length).filter (inB L B)).map (cI L L' cp)

/-- A copied slot of anchors `A` against a kept slot of anchors `A'` (in `B` or not, `inB'`), as
the copy's anchors `E` against the kept slot's `F`: by their first anchors, the copy after the
kept slot when those agree. -/
def copyVs (A A' E F : List ℕ) (inB' : Bool) : Bool :=
  if A.headD 0 = A'.headD 0 then inB' && decide (F.headD 0 < E.headD 0)
  else (decide (A.headD 0 < A'.headD 0) == decide (E.headD 0 < F.headD 0)) &&
    !decide (E.headD 0 = F.headD 0)

/-- Two copied slots of anchors `A`, `A'` as their copies' anchors `E`, `E'`: by their first
anchors, by the rest when those agree. -/
def copyCopy (A A' E E' : List ℕ) : Bool :=
  if A.headD 0 = A'.headD 0 then agree A.tail A'.tail E E'
  else (decide (A.headD 0 < A'.headD 0) == decide (E.headD 0 < E'.headD 0)) &&
    !decide (E.headD 0 = E'.headD 0)

/-- `L'` holds `L`'s boundary slots after a peel of the box of multiplicity index `mb` (index `c`
after) over the nodes `B` into the copies `cp`, with slot data `sd` and `sd'`: chains and places
of the kept and copied slots, the places one to one and onto, the anchors ordered alike. -/
structure PeelFlat (L L' : LDiagram) (B : List ℕ) (cp : ℕ → ℕ) (mb c : ℕ) (sd sd' : SlotData) :
    Prop where
  n : sd.1 + sd.2.1 = L.base.bnd.length
  n' : sd'.1 + sd'.2.1 = L'.base.bnd.length
  ch : ∀ i < L.base.bnd.length, chainS sd i = L.base.legChain (L.bndAt i)
  out : ∀ i < L.base.bnd.length, inB L B i = false →
    mI L L' i < L'.base.bnd.length ∧ chainS sd' (mI L L' i) = chainS sd i
  inn : ∀ i < L.base.bnd.length, inB L B i = true →
    mI L L' i < L'.base.bnd.length ∧ cI L L' cp i < L'.base.bnd.length ∧
      chainS sd i = mb :: chainS sd' (cI L L' cp i) ∧
      chainS sd' (mI L L' i) = c :: chainS sd' (cI L L' cp i)
  nodup : (imgs L L' B cp).Nodup
  onto : ∀ j < L'.base.bnd.length, j ∈ imgs L L' B cp
  keyM : ∀ i < L.base.bnd.length, ∀ i' < L.base.bnd.length,
    agree (anch sd i) (anch sd i') (anch sd' (mI L L' i)) (anch sd' (mI L L' i')) = true
  keyC : ∀ i < L.base.bnd.length, inB L B i = true → ∀ i' < L.base.bnd.length,
    copyVs (anch sd i) (anch sd i') (anch sd' (cI L L' cp i)) (anch sd' (mI L L' i'))
      (inB L B i') = true
  keyCC : ∀ i < L.base.bnd.length, inB L B i = true → ∀ i' < L.base.bnd.length,
    inB L B i' = true →
    copyCopy (anch sd i) (anch sd i') (anch sd' (cI L L' cp i)) (anch sd' (cI L L' cp i')) = true

/-- A slot digit's place and copy index in `L'`: its copy's at the copy index `s` of the peeled
box, else its own. -/
def psi (L L' : LDiagram) (B : List ℕ) (cp : ℕ → ℕ) (s : ℕ) (v : ℕ × List ℕ) : ℕ × List ℕ :=
  if inB L B v.1 = true ∧ v.2.head? = some s then (cI L L' cp v.1, v.2.tail)
  else (mI L L' v.1, v.2)

theorem bndMap_psi (L L' : LDiagram) (B : List ℕ) (cp : ℕ → ℕ) (s : ℕ) (σ' : ℕ → List ℕ → ℕ)
    (i : ℕ) (t : List ℕ) :
    bndMap L.base L'.base (peelφ B cp s) σ' i t =
      σ' (psi L L' B cp s (i, t)).1 (psi L L' B cp s (i, t)).2 := by
  unfold bndMap peelφ psi mI cI inB bndAt
  by_cases hc : (L.base.bnd.getD i Port.dflt).node ∈ B ∧ t.head? = some s
  · simp only [hc, decide_true, and_self, ite_true]
  · have : ¬(decide ((L.base.bnd.getD i Port.dflt).node ∈ B) = true ∧ t.head? = some s) := by
      simpa using hc
    simp only [hc, this, ite_false]

theorem ileave_head (A t : List ℕ) (hA : A ≠ []) : ∃ r, ileave A t = A.headD 0 :: r := by
  obtain ⟨a, A, rfl⟩ := List.exists_cons_of_ne_nil hA
  cases t with
  | nil => exact ⟨A, by simp [ileave_nil]⟩
  | cons j u => exact ⟨j :: ileave A u, rfl⟩

theorem anch_ne_nil (sd : SlotData) (i : ℕ) : anch sd i ≠ [] := by
  intro h
  have := anch_length sd i
  rw [h] at this
  simp at this

section peel

variable {L L' : LDiagram} {B : List ℕ} {cp : ℕ → ℕ} {mb c : ℕ} {sd sd' : SlotData}
  (hf : PeelFlat L L' B cp mb c sd sd') {ks : ℕ → ℕ} (hk : ks mb = ks c + 1)

include hf hk

theorem PeelFlat.psi_mem {v : ℕ × List ℕ} (hv : v ∈ slots sd ks) :
    psi L L' B cp (ks c) v ∈ slots sd' ks := by
  obtain ⟨i, t⟩ := v
  rw [mem_slots] at hv
  obtain ⟨hi, ht⟩ := hv
  rw [hf.n] at hi
  unfold psi
  cases hb : inB L B i
  · obtain ⟨h1, h2⟩ := hf.out i hi hb
    simp only [Bool.false_eq_true, false_and, ite_false]
    rw [mem_slots, hf.n', h2]
    exact ⟨h1, ht⟩
  · obtain ⟨h1, h2, h3, h4⟩ := hf.inn i hi hb
    rw [h3] at ht
    obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
    by_cases hjs : j = ks c
    · subst hjs
      simp only [List.head?_cons, true_and, ite_true, List.tail_cons]
      rw [mem_slots, hf.n']
      exact ⟨h2, hu⟩
    · simp only [List.head?_cons, Option.some.injEq, hjs, and_false, ite_false]
      rw [mem_slots, hf.n', h4]
      exact ⟨h1, cons_mem_cidx ks c j _ u (by omega) hu⟩

omit hk in
theorem PeelFlat.mI_inj {i i' : ℕ} (hi : i < L.base.bnd.length) (hi' : i' < L.base.bnd.length)
    (e : mI L L' i = mI L L' i') : i = i' :=
  List.inj_on_of_nodup_map (List.nodup_append.1 hf.nodup).1 (List.mem_range.2 hi)
    (List.mem_range.2 hi') e

omit hk in
theorem PeelFlat.cI_inj {i i' : ℕ} (hi : i < L.base.bnd.length) (hi' : i' < L.base.bnd.length)
    (hb : inB L B i = true) (hb' : inB L B i' = true) (e : cI L L' cp i = cI L L' cp i') :
    i = i' :=
  List.inj_on_of_nodup_map (List.nodup_append.1 hf.nodup).2.1
    (List.mem_filter.2 ⟨List.mem_range.2 hi, hb⟩)
    (List.mem_filter.2 ⟨List.mem_range.2 hi', hb'⟩) e

omit hk in
theorem PeelFlat.mI_ne_cI {i i' : ℕ} (hi : i < L.base.bnd.length) (hi' : i' < L.base.bnd.length)
    (hb' : inB L B i' = true) : mI L L' i ≠ cI L L' cp i' := fun e =>
  (List.nodup_append.1 hf.nodup).2.2 _ (List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩) _
    (List.mem_map.2 ⟨i', List.mem_filter.2 ⟨List.mem_range.2 hi', hb'⟩, rfl⟩) e

omit hk in
theorem PeelFlat.psi_inj {v v' : ℕ × List ℕ} (hv : v ∈ slots sd ks) (hv' : v' ∈ slots sd ks)
    (e : psi L L' B cp (ks c) v = psi L L' B cp (ks c) v') : v = v' := by
  obtain ⟨i, t⟩ := v
  obtain ⟨i', t'⟩ := v'
  rw [mem_slots, hf.n] at hv hv'
  unfold psi at e
  simp only at e
  split_ifs at e with h1 h2 h2
  · obtain ⟨e1, e2⟩ := Prod.mk.inj e
    have := hf.cI_inj hv.1 hv'.1 h1.1 h2.1 e1
    subst this
    refine Prod.ext rfl ?_
    obtain ⟨j, u, rfl⟩ : ∃ j u, t = j :: u := by
      cases t with
      | nil => simp at h1
      | cons j u => exact ⟨j, u, rfl⟩
    obtain ⟨j', u', rfl⟩ : ∃ j u, t' = j :: u := by
      cases t' with
      | nil => simp at h2
      | cons j u => exact ⟨j, u, rfl⟩
    simp only [List.head?_cons, Option.some.injEq] at h1 h2
    simp only [List.tail_cons] at e2
    simp only [h1.2, h2.2, e2]
  · exact absurd (Prod.mk.inj e).1.symm (hf.mI_ne_cI hv'.1 hv.1 h1.1)
  · exact absurd (Prod.mk.inj e).1 (hf.mI_ne_cI hv.1 hv'.1 h2.1)
  · obtain ⟨e1, e2⟩ := Prod.mk.inj e
    simp only [hf.mI_inj hv.1 hv'.1 e1, e2]

theorem PeelFlat.psi_onto {w : ℕ × List ℕ} (hw : w ∈ slots sd' ks) :
    ∃ v ∈ slots sd ks, psi L L' B cp (ks c) v = w := by
  obtain ⟨j, u⟩ := w
  rw [mem_slots, hf.n'] at hw
  obtain ⟨hj, hu⟩ := hw
  have hm := hf.onto j hj
  unfold imgs at hm
  rcases List.mem_append.1 hm with hm | hm
  · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hm
    rw [List.mem_range] at hi
    cases hb : inB L B i
    · obtain ⟨-, h2⟩ := hf.out i hi hb
      refine ⟨(i, u), (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, h2 ▸ hu⟩, ?_⟩
      simp [psi, hb]
    · obtain ⟨-, -, h3, h4⟩ := hf.inn i hi hb
      rw [h4] at hu
      obtain ⟨j0, u0, rfl, hj0, hu0⟩ := (mem_cidx_cons ks c _ _).1 hu
      refine ⟨(i, j0 :: u0), (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, by
        rw [h3]; exact cons_mem_cidx ks mb j0 _ u0 (by omega) hu0⟩, ?_⟩
      have : j0 ≠ ks c := by omega
      simp [psi, this]
  · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hm
    obtain ⟨hi, hb⟩ := List.mem_filter.1 hi
    rw [List.mem_range] at hi
    obtain ⟨-, -, h3, -⟩ := hf.inn i hi hb
    refine ⟨(i, ks c :: u), (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, by
      rw [h3]; exact cons_mem_cidx ks mb _ _ u (by omega) hu⟩, ?_⟩
    simp [psi, hb]

theorem PeelFlat.perm :
    (slots sd' ks).Perm ((slots sd ks).map (psi L L' B cp (ks c))) := by
  refine (List.perm_ext_iff_of_nodup (slots_nodup _ _)
    ((slots_nodup _ _).map_on fun v hv v' hv' e => hf.psi_inj hv hv' e)).2 fun w => ?_
  constructor
  · intro hw
    obtain ⟨v, hv, rfl⟩ := hf.psi_onto hk hw
    exact List.mem_map_of_mem hv
  · intro hw
    obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hw
    exact hf.psi_mem hk hv

omit hk in
/-- A kept slot digit: it keeps its copy index, its anchors in `L` and `L'` one longer. -/
theorem PeelFlat.kept {i : ℕ} {t : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hn : ¬(inB L B i = true ∧ t.head? = some (ks c))) :
    psi L L' B cp (ks c) (i, t) = (mI L L' i, t) ∧
      (anch sd i).length = t.length + 1 ∧ (anch sd' (mI L L' i)).length = t.length + 1 := by
  rw [mem_slots, hf.n] at hv
  refine ⟨by simp only [psi, hn, ite_false], by rw [anch_length, length_of_cidx hv.2], ?_⟩
  rw [anch_length, length_of_cidx hv.2]
  cases hb : inB L B i
  · rw [(hf.out i hv.1 hb).2]
  · obtain ⟨-, -, h3, h4⟩ := hf.inn i hv.1 hb
    rw [h3, h4]
    rfl

omit hk in
/-- A copied slot digit: its copy index is `ks c :: u`, its first anchor `a`, its copy's anchors
one longer than `u`. -/
theorem PeelFlat.copied {i : ℕ} {t : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hc : inB L B i = true ∧ t.head? = some (ks c)) :
    ∃ u a A, t = ks c :: u ∧ anch sd i = a :: A ∧ A.length = u.length + 1 ∧
      (anch sd' (cI L L' cp i)).length = u.length + 1 ∧
      psi L L' B cp (ks c) (i, t) = (cI L L' cp i, u) := by
  rw [mem_slots, hf.n] at hv
  obtain ⟨-, -, h3, -⟩ := hf.inn i hv.1 hc.1
  obtain ⟨u, rfl⟩ : ∃ u, t = ks c :: u := by
    obtain ⟨j, u, rfl⟩ : ∃ j u, t = j :: u := by
      cases t with
      | nil => simp at hc
      | cons j u => exact ⟨j, u, rfl⟩
    simp only [List.head?_cons, Option.some.injEq] at hc
    exact ⟨u, by rw [hc.2]⟩
  have hl := length_of_cidx hv.2
  rw [h3] at hl
  simp only [List.length_cons, Nat.add_right_cancel_iff] at hl
  obtain ⟨a, A, e, hA⟩ := cons_of_len (anch_length sd i)
  refine ⟨u, a, A, rfl, e, ?_, by rw [anch_length, hl], by simp [psi, hc.1]⟩
  rw [hA, h3, List.length_cons, hl]

/-- A copied slot digit against a kept one: their keys compare alike in `L` and `L'`, both ways
round. -/
theorem PeelFlat.copy_kept {i i' : ℕ} {t t' : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hx : (i', t') ∈ slots sd ks) (hc : inB L B i = true ∧ t.head? = some (ks c))
    (hn : ¬(inB L B i' = true ∧ t'.head? = some (ks c))) :
    (List.Lex (· < ·) (key sd' (cI L L' cp i) t.tail) (key sd' (mI L L' i') t') ↔
      List.Lex (· < ·) (key sd i t) (key sd i' t')) ∧
    (List.Lex (· < ·) (key sd' (mI L L' i') t') (key sd' (cI L L' cp i) t.tail) ↔
      List.Lex (· < ·) (key sd i' t') (key sd i t)) := by
  obtain ⟨u, a, A, rfl, eA, -, -, -⟩ := hf.copied hv hc
  have hi := ((mem_slots _ _ _ _).1 hv).1
  have hx' := (mem_slots _ _ _ _).1 hx
  rw [hf.n] at hi hx'
  have hC := hf.keyC i hi hc.1 i' hx'.1
  obtain ⟨rE, eE⟩ := ileave_head (anch sd' (cI L L' cp i)) u (anch_ne_nil _ _)
  obtain ⟨rF, eF⟩ := ileave_head (anch sd' (mI L L' i')) t' (anch_ne_nil _ _)
  obtain ⟨a', A', eA2⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd i')
  simp only [key, List.tail_cons]
  rw [eE, eF, eA, eA2, ileave_cc]
  rw [eA, eA2] at hC
  unfold copyVs at hC
  simp only [List.headD_cons] at hC
  by_cases h0 : a = a'
  · simp only [h0, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at hC
    obtain ⟨hb', hlt⟩ := hC
    have hn' : t'.head? ≠ some (ks c) := fun e => hn ⟨hb', e⟩
    obtain ⟨-, -, h3, -⟩ := hf.inn i' hx'.1 hb'
    rw [h3] at hx'
    obtain ⟨j', u', rfl, hj', -⟩ := (mem_cidx_cons ks mb _ _).1 hx'.2
    have hj : j' < ks c := by
      have : j' ≠ ks c := fun e => hn' (by rw [e]; rfl)
      omega
    subst h0
    rw [ileave_cc]
    simp only [lex_cc]
    refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
    · rintro (e | ⟨e, -⟩) <;> omega
    · rintro (e | ⟨-, e | ⟨e, -⟩⟩) <;> omega
    · intro _; exact Or.inr ⟨by simp, Or.inl hj⟩
    · intro _; exact Or.inl hlt
  · simp only [h0, ↓reduceIte, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true',
      decide_eq_false_iff_not] at hC
    obtain ⟨h1, h2⟩ := hC
    replace h1 := decide_eq_decide.1 h1
    obtain ⟨rA, eA3⟩ := ileave_head (a' :: A') t' (by simp)
    rw [eA3]
    simp only [List.headD_cons]
    rw [lex_ne_heads h2, lex_ne_heads (Ne.symm h2), lex_ne_heads h0, lex_ne_heads (Ne.symm h0)]
    refine ⟨h1.symm, ⟨fun e => ?_, fun e => ?_⟩⟩
    · have := h1.not; omega
    · have := h1.not; omega

omit hk in
/-- Two kept slot digits: their keys compare alike in `L` and `L'`. -/
theorem PeelFlat.kept_kept {i i' : ℕ} {t t' : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hx : (i', t') ∈ slots sd ks) (hn : ¬(inB L B i = true ∧ t.head? = some (ks c)))
    (hn' : ¬(inB L B i' = true ∧ t'.head? = some (ks c))) :
    List.Lex (· < ·) (key sd' (mI L L' i) t) (key sd' (mI L L' i') t') ↔
      List.Lex (· < ·) (key sd i t) (key sd i' t') := by
  obtain ⟨-, l1, l2⟩ := hf.kept hv hn
  obtain ⟨-, l1', l2'⟩ := hf.kept hx hn'
  have hi := ((mem_slots _ _ _ _).1 hv).1
  have hi' := ((mem_slots _ _ _ _).1 hx).1
  rw [hf.n] at hi hi'
  exact (agree_lex t t' _ _ _ _ (hf.keyM i hi i' hi') l1 l1' (l2.trans l1.symm)
    (l2'.trans l1'.symm)).symm

omit hk in
/-- Two copied slot digits: their keys compare alike in `L` and `L'`. -/
theorem PeelFlat.copy_copy {i i' : ℕ} {t t' : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hx : (i', t') ∈ slots sd ks) (hc : inB L B i = true ∧ t.head? = some (ks c))
    (hc' : inB L B i' = true ∧ t'.head? = some (ks c)) :
    List.Lex (· < ·) (key sd' (cI L L' cp i) t.tail) (key sd' (cI L L' cp i') t'.tail) ↔
      List.Lex (· < ·) (key sd i t) (key sd i' t') := by
  obtain ⟨u, a, A, rfl, eA, lA, lE, -⟩ := hf.copied hv hc
  obtain ⟨u', a', A', rfl, eA', lA', lE', -⟩ := hf.copied hx hc'
  have hi := ((mem_slots _ _ _ _).1 hv).1
  have hi' := ((mem_slots _ _ _ _).1 hx).1
  rw [hf.n] at hi hi'
  have hC := hf.keyCC i hi hc.1 i' hi' hc'.1
  rw [eA, eA'] at hC
  unfold copyCopy at hC
  simp only [key, List.tail_cons]
  rw [eA, eA', ileave_cc, ileave_cc]
  by_cases h0 : a = a'
  · simp only [List.headD_cons, h0, ↓reduceIte, List.tail_cons] at hC
    subst h0
    rw [← agree_lex u u' A A' _ _ hC lA lA' (lE.trans lA.symm) (lE'.trans lA'.symm)]
    simp only [lex_cc, lt_irrefl, false_or, true_and]
  · simp only [List.headD_cons, h0, ↓reduceIte, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true',
      decide_eq_false_iff_not] at hC
    obtain ⟨h1, h2⟩ := hC
    replace h1 := decide_eq_decide.1 h1
    obtain ⟨rE, eE⟩ := ileave_head (anch sd' (cI L L' cp i)) u (anch_ne_nil _ _)
    obtain ⟨rE', eE'⟩ := ileave_head (anch sd' (cI L L' cp i')) u' (anch_ne_nil _ _)
    rw [eE, eE', lex_ne_heads h2, lex_ne_heads h0]
    exact h1.symm

/-- The peel keeps the order of the slot digits' keys. -/
theorem PeelFlat.ord {v x : ℕ × List ℕ} (hv : v ∈ slots sd ks) (hx : x ∈ slots sd ks) :
    List.Lex (· < ·) (key sd' (psi L L' B cp (ks c) v).1 (psi L L' B cp (ks c) v).2)
        (key sd' (psi L L' B cp (ks c) x).1 (psi L L' B cp (ks c) x).2) ↔
      List.Lex (· < ·) (key sd v.1 v.2) (key sd x.1 x.2) := by
  obtain ⟨i, t⟩ := v
  obtain ⟨i', t'⟩ := x
  simp only [psi]
  by_cases hc : inB L B i = true ∧ t.head? = some (ks c) <;>
    by_cases hc' : inB L B i' = true ∧ t'.head? = some (ks c) <;>
    simp only [hc, hc', ↓reduceIte]
  · exact hf.copy_copy hv hx hc hc'
  · exact (hf.copy_kept hk hv hx hc hc').1
  · exact (hf.copy_kept hk hx hv hc' hc).2
  · exact hf.kept_kept hv hx hc hc'

/-- The peel keeps every slot digit's flat position. -/
theorem PeelFlat.fpos_psi {v : ℕ × List ℕ} (hv : v ∈ slots sd ks) :
    fpos sd' ks (psi L L' B cp (ks c) v).1 (psi L L' B cp (ks c) v).2 = fpos sd ks v.1 v.2 := by
  unfold fpos
  rw [((hf.perm hk).filter _).length_eq, List.filter_map, List.length_map]
  congr 1
  refine List.filter_congr fun w hw => ?_
  simp only [Function.comp_apply]
  exact decide_eq_decide.2 (hf.ord hk hw hv)

end peel

/-- A peel keeps the linked digit sum read through the flat positions of both sides. -/
theorem evalL_peel_flat {L L' : LDiagram} {bid mb c : ℕ} {B : List ℕ} {cp : ℕ → ℕ}
    {sd sd' : SlotData} (h : PeelOK L L' bid mb c B cp) (hf : PeelFlat L L' B cp mb c sd sd')
    (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (y : ℕ → ℕ) :
    L'.evalL E ks (flatAt sd' ks y) = L.evalL E ks (flatAt sd ks y) := by
  rw [evalL_peel h E ks hk]
  refine evalL_congr L E ks _ _ fun i hi t ht => ?_
  rw [bndMap_psi]
  have hv : (i, t) ∈ slots sd ks := by
    refine (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, ?_⟩
    rw [hf.ch i hi, bndAt, List.getD_eq_getElem _ _ hi]
    exact ht
  exact congrArg y (hf.fpos_psi hk hv)

/-! ## Peels of the first copy keep the flat positions -/

/-- `t` with its first coordinate one lower. -/
def decT : List ℕ → List ℕ
  | [] => []
  | j :: u => (j - 1) :: u

/-- A copied slot of anchors `A` against a kept slot of anchors `A'` (in `B` or not, `inB'`), as
the copy's anchors `E` against the kept slot's `F`: by their first anchors, the copy before the
kept slot when those agree. -/
def copyVs0 (A A' E F : List ℕ) (inB' : Bool) : Bool :=
  if A.headD 0 = A'.headD 0 then inB' && decide (E.headD 0 < F.headD 0)
  else (decide (A.headD 0 < A'.headD 0) == decide (E.headD 0 < F.headD 0)) &&
    !decide (E.headD 0 = F.headD 0)

/-- `L'` holds `L`'s boundary slots after a peel of the first copy of the box of multiplicity
index `mb` (index `c` after) over the nodes `B` into the copies `cp`, with slot data `sd` and
`sd'`: `PeelFlat` with the copy before the box's copies, and slots of one first anchor all in `B`
or all off it. -/
structure PeelFirst (L L' : LDiagram) (B : List ℕ) (cp : ℕ → ℕ) (mb c : ℕ) (sd sd' : SlotData) :
    Prop where
  n : sd.1 + sd.2.1 = L.base.bnd.length
  n' : sd'.1 + sd'.2.1 = L'.base.bnd.length
  ch : ∀ i < L.base.bnd.length, chainS sd i = L.base.legChain (L.bndAt i)
  out : ∀ i < L.base.bnd.length, inB L B i = false →
    mI L L' i < L'.base.bnd.length ∧ chainS sd' (mI L L' i) = chainS sd i
  inn : ∀ i < L.base.bnd.length, inB L B i = true →
    mI L L' i < L'.base.bnd.length ∧ cI L L' cp i < L'.base.bnd.length ∧
      chainS sd i = mb :: chainS sd' (cI L L' cp i) ∧
      chainS sd' (mI L L' i) = c :: chainS sd' (cI L L' cp i)
  nodup : (imgs L L' B cp).Nodup
  onto : ∀ j < L'.base.bnd.length, j ∈ imgs L L' B cp
  keyM : ∀ i < L.base.bnd.length, ∀ i' < L.base.bnd.length,
    agree (anch sd i) (anch sd i') (anch sd' (mI L L' i)) (anch sd' (mI L L' i')) = true
  keyC : ∀ i < L.base.bnd.length, inB L B i = true → ∀ i' < L.base.bnd.length,
    copyVs0 (anch sd i) (anch sd i') (anch sd' (cI L L' cp i)) (anch sd' (mI L L' i'))
      (inB L B i') = true
  keyCC : ∀ i < L.base.bnd.length, inB L B i = true → ∀ i' < L.base.bnd.length,
    inB L B i' = true →
    copyCopy (anch sd i) (anch sd i') (anch sd' (cI L L' cp i)) (anch sd' (cI L L' cp i')) = true
  same : ∀ i < L.base.bnd.length, ∀ i' < L.base.bnd.length,
    (anch sd i).headD 0 = (anch sd i').headD 0 → inB L B i = inB L B i'

/-- A slot digit's place and copy index in `L'` after a peel of the first copy: its copy's at the
copy index `0`, one copy lower on `B` otherwise. -/
def psi0 (L L' : LDiagram) (B : List ℕ) (cp : ℕ → ℕ) (v : ℕ × List ℕ) : ℕ × List ℕ :=
  if inB L B v.1 = true ∧ v.2.head? = some 0 then (cI L L' cp v.1, v.2.tail)
  else if inB L B v.1 = true then (mI L L' v.1, decT v.2) else (mI L L' v.1, v.2)

theorem lex_dec {a a' j j' : ℕ} {X Y : List ℕ} (hj : 1 ≤ j) (hj' : 1 ≤ j') :
    List.Lex (· < ·) (a :: (j - 1) :: X) (a' :: (j' - 1) :: Y) ↔
      List.Lex (· < ·) (a :: j :: X) (a' :: j' :: Y) := by
  simp only [lex_cc]
  constructor
  · rintro (e | ⟨e, f | ⟨f, g⟩⟩)
    · exact Or.inl e
    · exact Or.inr ⟨e, Or.inl (by omega)⟩
    · exact Or.inr ⟨e, Or.inr ⟨by omega, g⟩⟩
  · rintro (e | ⟨e, f | ⟨f, g⟩⟩)
    · exact Or.inl e
    · exact Or.inr ⟨e, Or.inl (by omega)⟩
    · exact Or.inr ⟨e, Or.inr ⟨by omega, g⟩⟩

section first

variable {L L' : LDiagram} {B : List ℕ} {cp : ℕ → ℕ} {mb c : ℕ} {sd sd' : SlotData}
  (hf : PeelFirst L L' B cp mb c sd sd') {ks : ℕ → ℕ} (hk : ks mb = ks c + 1)

include hf hk

theorem PeelFirst.psi_mem {v : ℕ × List ℕ} (hv : v ∈ slots sd ks) :
    psi0 L L' B cp v ∈ slots sd' ks := by
  obtain ⟨i, t⟩ := v
  rw [mem_slots] at hv
  obtain ⟨hi, ht⟩ := hv
  rw [hf.n] at hi
  unfold psi0
  cases hb : inB L B i
  · obtain ⟨h1, h2⟩ := hf.out i hi hb
    simp only [Bool.false_eq_true, false_and, ite_false]
    rw [mem_slots, hf.n', h2]
    exact ⟨h1, ht⟩
  · obtain ⟨h1, h2, h3, h4⟩ := hf.inn i hi hb
    rw [h3] at ht
    obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
    by_cases hj0 : j = 0
    · subst hj0
      simp only [List.head?_cons, true_and, ite_true, List.tail_cons]
      rw [mem_slots, hf.n']
      exact ⟨h2, hu⟩
    · simp only [List.head?_cons, Option.some.injEq, hj0, and_false, ite_true, ite_false, decT]
      rw [mem_slots, hf.n', h4]
      exact ⟨h1, cons_mem_cidx ks c (j - 1) _ u (by omega) hu⟩

omit hk in
theorem PeelFirst.mI_inj {i i' : ℕ} (hi : i < L.base.bnd.length) (hi' : i' < L.base.bnd.length)
    (e : mI L L' i = mI L L' i') : i = i' :=
  List.inj_on_of_nodup_map (List.nodup_append.1 hf.nodup).1 (List.mem_range.2 hi)
    (List.mem_range.2 hi') e

omit hk in
theorem PeelFirst.cI_inj {i i' : ℕ} (hi : i < L.base.bnd.length) (hi' : i' < L.base.bnd.length)
    (hb : inB L B i = true) (hb' : inB L B i' = true) (e : cI L L' cp i = cI L L' cp i') :
    i = i' :=
  List.inj_on_of_nodup_map (List.nodup_append.1 hf.nodup).2.1
    (List.mem_filter.2 ⟨List.mem_range.2 hi, hb⟩)
    (List.mem_filter.2 ⟨List.mem_range.2 hi', hb'⟩) e

omit hk in
theorem PeelFirst.mI_ne_cI {i i' : ℕ} (hi : i < L.base.bnd.length) (hi' : i' < L.base.bnd.length)
    (hb' : inB L B i' = true) : mI L L' i ≠ cI L L' cp i' := fun e =>
  (List.nodup_append.1 hf.nodup).2.2 _ (List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩) _
    (List.mem_map.2 ⟨i', List.mem_filter.2 ⟨List.mem_range.2 hi', hb'⟩, rfl⟩) e

omit hk in
/-- A slot digit on `B` kept by the peel: its first copy index is at least `1`. -/
theorem PeelFirst.kept_head {i : ℕ} {t : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hb : inB L B i = true) (hn : ¬(inB L B i = true ∧ t.head? = some 0)) :
    ∃ j u, t = j :: u ∧ 1 ≤ j ∧ j < ks mb ∧ u ∈ cidx ks (chainS sd' (cI L L' cp i)) := by
  rw [mem_slots, hf.n] at hv
  obtain ⟨-, -, h3, -⟩ := hf.inn i hv.1 hb
  have ht := hv.2
  rw [h3] at ht
  obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
  refine ⟨j, u, rfl, ?_, hj, hu⟩
  by_contra h0
  exact hn ⟨hb, by rw [List.head?_cons, show j = 0 by omega]⟩

omit hk in
theorem PeelFirst.copied {i : ℕ} {t : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hc : inB L B i = true ∧ t.head? = some 0) :
    ∃ u a A, t = 0 :: u ∧ anch sd i = a :: A ∧ A.length = u.length + 1 ∧
      (anch sd' (cI L L' cp i)).length = u.length + 1 ∧
      psi0 L L' B cp (i, t) = (cI L L' cp i, u) := by
  rw [mem_slots, hf.n] at hv
  obtain ⟨-, -, h3, -⟩ := hf.inn i hv.1 hc.1
  obtain ⟨u, rfl⟩ : ∃ u, t = 0 :: u := by
    obtain ⟨j, u, rfl⟩ : ∃ j u, t = j :: u := by
      cases t with
      | nil => simp at hc
      | cons j u => exact ⟨j, u, rfl⟩
    simp only [List.head?_cons, Option.some.injEq] at hc
    exact ⟨u, by rw [hc.2]⟩
  have hl := length_of_cidx hv.2
  rw [h3] at hl
  simp only [List.length_cons, Nat.add_right_cancel_iff] at hl
  obtain ⟨a, A, e, hA⟩ := cons_of_len (anch_length sd i)
  refine ⟨u, a, A, rfl, e, ?_, by rw [anch_length, hl], by simp [psi0, hc.1]⟩
  rw [hA, h3, List.length_cons, hl]

omit hk in
/-- A kept slot digit: its image, and its anchors in `L` and `L'` one longer than its copy
index. -/
theorem PeelFirst.kept {i : ℕ} {t : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hn : ¬(inB L B i = true ∧ t.head? = some 0)) :
    psi0 L L' B cp (i, t) = (mI L L' i, if inB L B i = true then decT t else t) ∧
      (anch sd i).length = t.length + 1 ∧ (anch sd' (mI L L' i)).length = t.length + 1 := by
  have hv' := hv
  rw [mem_slots, hf.n] at hv
  refine ⟨by unfold psi0; split_ifs <;> simp_all, by rw [anch_length, length_of_cidx hv.2], ?_⟩
  rw [anch_length, length_of_cidx hv.2]
  cases hb : inB L B i
  · rw [(hf.out i hv.1 hb).2]
  · obtain ⟨-, -, h3, h4⟩ := hf.inn i hv.1 hb
    rw [h3, h4]
    rfl

theorem PeelFirst.psi_inj {v v' : ℕ × List ℕ} (hv : v ∈ slots sd ks) (hv' : v' ∈ slots sd ks)
    (e : psi0 L L' B cp v = psi0 L L' B cp v') : v = v' := by
  obtain ⟨i, t⟩ := v
  obtain ⟨i', t'⟩ := v'
  have hi := ((mem_slots _ _ _ _).1 hv).1
  have hi' := ((mem_slots _ _ _ _).1 hv').1
  rw [hf.n] at hi hi'
  by_cases hc : inB L B i = true ∧ t.head? = some 0 <;>
    by_cases hc' : inB L B i' = true ∧ t'.head? = some 0
  · have p : psi0 L L' B cp (i, t) = (cI L L' cp i, t.tail) := by simp [psi0, hc]
    have p' : psi0 L L' B cp (i', t') = (cI L L' cp i', t'.tail) := by simp [psi0, hc']
    rw [p, p'] at e
    obtain ⟨e1, e2⟩ := Prod.mk.inj e
    have := hf.cI_inj hi hi' hc.1 hc'.1 e1
    subst this
    refine Prod.ext rfl ?_
    cases t with
    | nil => simp at hc
    | cons j u =>
      cases t' with
      | nil => simp at hc'
      | cons j' u' =>
        simp only [List.head?_cons, Option.some.injEq] at hc hc'
        simp only [List.tail_cons] at e2
        simp [hc.2, hc'.2, e2]
  · have p : psi0 L L' B cp (i, t) = (cI L L' cp i, t.tail) := by simp [psi0, hc]
    obtain ⟨p', -, -⟩ := hf.kept hv' hc'
    rw [p, p'] at e
    exact absurd (Prod.mk.inj e).1.symm (hf.mI_ne_cI hi' hi hc.1)
  · obtain ⟨p, -, -⟩ := hf.kept hv hc
    have p' : psi0 L L' B cp (i', t') = (cI L L' cp i', t'.tail) := by simp [psi0, hc']
    rw [p, p'] at e
    exact absurd (Prod.mk.inj e).1 (hf.mI_ne_cI hi hi' hc'.1)
  · obtain ⟨p, -, -⟩ := hf.kept hv hc
    obtain ⟨p', -, -⟩ := hf.kept hv' hc'
    rw [p, p'] at e
    obtain ⟨e1, e2⟩ := Prod.mk.inj e
    have := hf.mI_inj hi hi' e1
    subst this
    refine Prod.ext rfl ?_
    cases hb : inB L B i
    · simpa [hb] using e2
    · obtain ⟨j, u, rfl, hj, -, -⟩ := hf.kept_head hv hb hc
      obtain ⟨j', u', rfl, hj', -, -⟩ := hf.kept_head hv' hb hc'
      simp only [hb, if_true, decT, List.cons.injEq] at e2
      simp only [List.cons.injEq]
      exact ⟨by omega, e2.2⟩

theorem PeelFirst.psi_onto {w : ℕ × List ℕ} (hw : w ∈ slots sd' ks) :
    ∃ v ∈ slots sd ks, psi0 L L' B cp v = w := by
  obtain ⟨j, u⟩ := w
  rw [mem_slots, hf.n'] at hw
  obtain ⟨hj, hu⟩ := hw
  have hm := hf.onto j hj
  unfold imgs at hm
  rcases List.mem_append.1 hm with hm | hm
  · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hm
    rw [List.mem_range] at hi
    cases hb : inB L B i
    · obtain ⟨-, h2⟩ := hf.out i hi hb
      refine ⟨(i, u), (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, h2 ▸ hu⟩, ?_⟩
      simp [psi0, hb]
    · obtain ⟨-, -, h3, h4⟩ := hf.inn i hi hb
      rw [h4] at hu
      obtain ⟨j0, u0, rfl, hj0, hu0⟩ := (mem_cidx_cons ks c _ _).1 hu
      refine ⟨(i, (j0 + 1) :: u0), (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, by
        rw [h3]; exact cons_mem_cidx ks mb (j0 + 1) _ u0 (by omega) hu0⟩, ?_⟩
      simp [psi0, hb, decT]
  · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hm
    obtain ⟨hi, hb⟩ := List.mem_filter.1 hi
    rw [List.mem_range] at hi
    obtain ⟨-, -, h3, -⟩ := hf.inn i hi hb
    refine ⟨(i, 0 :: u), (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, by
      rw [h3]; exact cons_mem_cidx ks mb 0 _ u (by omega) hu⟩, ?_⟩
    simp [psi0, hb]

theorem PeelFirst.perm :
    (slots sd' ks).Perm ((slots sd ks).map (psi0 L L' B cp)) := by
  refine (List.perm_ext_iff_of_nodup (slots_nodup _ _)
    ((slots_nodup _ _).map_on fun v hv v' hv' e => hf.psi_inj hk hv hv' e)).2 fun w => ?_
  constructor
  · intro hw
    obtain ⟨v, hv, rfl⟩ := hf.psi_onto hk hw
    exact List.mem_map_of_mem hv
  · intro hw
    obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hw
    exact hf.psi_mem hk hv

/-- A copied slot digit against a kept one: their keys compare alike in `L` and `L'`, both ways
round. -/
theorem PeelFirst.copy_kept {i i' : ℕ} {t t' : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hx : (i', t') ∈ slots sd ks) (hc : inB L B i = true ∧ t.head? = some 0)
    (hn : ¬(inB L B i' = true ∧ t'.head? = some 0)) :
    (List.Lex (· < ·) (key sd' (cI L L' cp i) t.tail) (key sd' (psi0 L L' B cp (i', t')).1
        (psi0 L L' B cp (i', t')).2) ↔
      List.Lex (· < ·) (key sd i t) (key sd i' t')) ∧
    (List.Lex (· < ·) (key sd' (psi0 L L' B cp (i', t')).1 (psi0 L L' B cp (i', t')).2)
        (key sd' (cI L L' cp i) t.tail) ↔
      List.Lex (· < ·) (key sd i' t') (key sd i t)) := by
  obtain ⟨u, a, A, rfl, eA, -, -, -⟩ := hf.copied hv hc
  obtain ⟨hp, -, -⟩ := hf.kept hx hn
  rw [hp]
  simp only
  have hi := ((mem_slots _ _ _ _).1 hv).1
  have hx' := (mem_slots _ _ _ _).1 hx
  rw [hf.n] at hi hx'
  have hC := hf.keyC i hi hc.1 i' hx'.1
  set t'' := if inB L B i' = true then decT t' else t'
  obtain ⟨rE, eE⟩ := ileave_head (anch sd' (cI L L' cp i)) u (anch_ne_nil _ _)
  obtain ⟨rF, eF⟩ := ileave_head (anch sd' (mI L L' i')) t'' (anch_ne_nil _ _)
  obtain ⟨a', A', eA2⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd i')
  simp only [key, List.tail_cons]
  rw [eE, eF, eA, eA2, ileave_cc]
  rw [eA, eA2] at hC
  unfold copyVs0 at hC
  simp only [List.headD_cons] at hC
  by_cases h0 : a = a'
  · simp only [h0, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at hC
    obtain ⟨hb', hlt⟩ := hC
    obtain ⟨j', u', rfl, hj1, -, -⟩ := hf.kept_head hx hb' hn
    subst h0
    rw [ileave_cc]
    refine ⟨⟨fun _ => lex_cc.2 (Or.inr ⟨rfl, lex_cc.2 (Or.inl (by omega))⟩),
      fun _ => lex_cc.2 (Or.inl hlt)⟩, ⟨fun e => ?_, fun e => ?_⟩⟩
    · exfalso
      rcases lex_cc.1 e with e | ⟨e, -⟩ <;> omega
    · exfalso
      rcases lex_cc.1 e with e | ⟨-, e⟩
      · omega
      · rcases lex_cc.1 e with e | ⟨e, -⟩ <;> omega
  · simp only [h0, ↓reduceIte, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true',
      decide_eq_false_iff_not] at hC
    obtain ⟨h1, h2⟩ := hC
    replace h1 := decide_eq_decide.1 h1
    obtain ⟨rA, eA3⟩ := ileave_head (a' :: A') t' (by simp)
    rw [eA3]
    simp only [List.headD_cons]
    rw [lex_ne_heads h2, lex_ne_heads (Ne.symm h2), lex_ne_heads h0, lex_ne_heads (Ne.symm h0)]
    refine ⟨h1.symm, ⟨fun e => ?_, fun e => ?_⟩⟩
    · have := h1.not; omega
    · have := h1.not; omega

/-- Two kept slot digits: their keys compare alike in `L` and `L'`. -/
theorem PeelFirst.kept_kept {i i' : ℕ} {t t' : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hx : (i', t') ∈ slots sd ks) (hn : ¬(inB L B i = true ∧ t.head? = some 0))
    (hn' : ¬(inB L B i' = true ∧ t'.head? = some 0)) :
    List.Lex (· < ·) (key sd' (psi0 L L' B cp (i, t)).1 (psi0 L L' B cp (i, t)).2)
        (key sd' (psi0 L L' B cp (i', t')).1 (psi0 L L' B cp (i', t')).2) ↔
      List.Lex (· < ·) (key sd i t) (key sd i' t') := by
  obtain ⟨p1, l1, l2⟩ := hf.kept hv hn
  obtain ⟨p1', l1', l2'⟩ := hf.kept hx hn'
  have hi := ((mem_slots _ _ _ _).1 hv).1
  have hi' := ((mem_slots _ _ _ _).1 hx).1
  rw [hf.n] at hi hi'
  have hag := hf.keyM i hi i' hi'
  have base := agree_lex t t' _ _ _ _ hag l1 l1' (l2.trans l1.symm) (l2'.trans l1'.symm)
  rw [p1, p1']
  simp only
  cases hb : inB L B i <;> cases hb' : inB L B i'
  · simp only [Bool.false_eq_true, if_false]
    exact base.symm
  · obtain ⟨j', u', rfl, -, -, -⟩ := hf.kept_head hx hb' hn'
    have hne : (anch sd i).headD 0 ≠ (anch sd i').headD 0 := fun e => by
      have := hf.same i hi i' hi' e; rw [hb, hb'] at this; exact Bool.false_ne_true this
    obtain ⟨a, A, eA⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd i)
    obtain ⟨a', A', eA'⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd i')
    obtain ⟨b, Bs, eB⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd' (mI L L' i))
    obtain ⟨b', Bs', eB'⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd' (mI L L' i'))
    rw [eA, eA', eB, eB'] at hag
    rw [eA, eA'] at hne
    simp only [List.headD_cons] at hne
    obtain ⟨g1, g2, -⟩ := agree_cons hag
    have hbne : b ≠ b' := fun e => hne (g2.2 e)
    simp only [Bool.false_eq_true, if_false, if_true, key, decT]
    rw [eA, eA', eB, eB']
    obtain ⟨r1, e1⟩ := ileave_head (b :: Bs) t (by simp)
    obtain ⟨r2, e2⟩ := ileave_head (b' :: Bs') ((j' - 1) :: u') (by simp)
    obtain ⟨r3, e3⟩ := ileave_head (a :: A) t (by simp)
    obtain ⟨r4, e4⟩ := ileave_head (a' :: A') (j' :: u') (by simp)
    rw [e1, e2, e3, e4]
    simp only [List.headD_cons]
    rw [lex_ne_heads hbne, lex_ne_heads hne]
    exact g1.symm
  · obtain ⟨j, u, rfl, -, -, -⟩ := hf.kept_head hv hb hn
    have hne : (anch sd i).headD 0 ≠ (anch sd i').headD 0 := fun e => by
      have := hf.same i hi i' hi' e; rw [hb, hb'] at this; exact Bool.false_ne_true this.symm
    obtain ⟨a, A, eA⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd i)
    obtain ⟨a', A', eA'⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd i')
    obtain ⟨b, Bs, eB⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd' (mI L L' i))
    obtain ⟨b', Bs', eB'⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd' (mI L L' i'))
    rw [eA, eA', eB, eB'] at hag
    rw [eA, eA'] at hne
    simp only [List.headD_cons] at hne
    obtain ⟨g1, g2, -⟩ := agree_cons hag
    have hbne : b ≠ b' := fun e => hne (g2.2 e)
    simp only [Bool.false_eq_true, if_false, if_true, key, decT]
    rw [eA, eA', eB, eB']
    obtain ⟨r1, e1⟩ := ileave_head (b :: Bs) ((j - 1) :: u) (by simp)
    obtain ⟨r2, e2⟩ := ileave_head (b' :: Bs') t' (by simp)
    obtain ⟨r3, e3⟩ := ileave_head (a :: A) (j :: u) (by simp)
    obtain ⟨r4, e4⟩ := ileave_head (a' :: A') t' (by simp)
    rw [e1, e2, e3, e4]
    simp only [List.headD_cons]
    rw [lex_ne_heads hbne, lex_ne_heads hne]
    exact g1.symm
  · obtain ⟨j, u, rfl, hj, -, -⟩ := hf.kept_head hv hb hn
    obtain ⟨j', u', rfl, hj', -, -⟩ := hf.kept_head hx hb' hn'
    simp only [if_true, decT]
    obtain ⟨b, Bs, eB⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd' (mI L L' i))
    obtain ⟨b', Bs', eB'⟩ := List.exists_cons_of_ne_nil (anch_ne_nil sd' (mI L L' i'))
    have k1 : key sd' (mI L L' i) ((j - 1) :: u) = b :: (j - 1) :: ileave Bs u := by
      simp only [key, eB, ileave_cc]
    have k2 : key sd' (mI L L' i') ((j' - 1) :: u') = b' :: (j' - 1) :: ileave Bs' u' := by
      simp only [key, eB', ileave_cc]
    have k3 : key sd' (mI L L' i) (j :: u) = b :: j :: ileave Bs u := by
      simp only [key, eB, ileave_cc]
    have k4 : key sd' (mI L L' i') (j' :: u') = b' :: j' :: ileave Bs' u' := by
      simp only [key, eB', ileave_cc]
    rw [k1, k2, lex_dec hj hj', ← k3, ← k4]
    exact base.symm

omit hk in
/-- Two copied slot digits: their keys compare alike in `L` and `L'`. -/
theorem PeelFirst.copy_copy {i i' : ℕ} {t t' : List ℕ} (hv : (i, t) ∈ slots sd ks)
    (hx : (i', t') ∈ slots sd ks) (hc : inB L B i = true ∧ t.head? = some 0)
    (hc' : inB L B i' = true ∧ t'.head? = some 0) :
    List.Lex (· < ·) (key sd' (cI L L' cp i) t.tail) (key sd' (cI L L' cp i') t'.tail) ↔
      List.Lex (· < ·) (key sd i t) (key sd i' t') := by
  obtain ⟨u, a, A, rfl, eA, lA, lE, -⟩ := hf.copied hv hc
  obtain ⟨u', a', A', rfl, eA', lA', lE', -⟩ := hf.copied hx hc'
  have hi := ((mem_slots _ _ _ _).1 hv).1
  have hi' := ((mem_slots _ _ _ _).1 hx).1
  rw [hf.n] at hi hi'
  have hC := hf.keyCC i hi hc.1 i' hi' hc'.1
  rw [eA, eA'] at hC
  unfold copyCopy at hC
  simp only [key, List.tail_cons]
  rw [eA, eA', ileave_cc, ileave_cc]
  by_cases h0 : a = a'
  · simp only [List.headD_cons, h0, ↓reduceIte, List.tail_cons] at hC
    subst h0
    rw [← agree_lex u u' A A' _ _ hC lA lA' (lE.trans lA.symm) (lE'.trans lA'.symm)]
    simp only [lex_cc, lt_irrefl, false_or, true_and]
  · simp only [List.headD_cons, h0, ↓reduceIte, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true',
      decide_eq_false_iff_not] at hC
    obtain ⟨h1, h2⟩ := hC
    replace h1 := decide_eq_decide.1 h1
    obtain ⟨rE, eE⟩ := ileave_head (anch sd' (cI L L' cp i)) u (anch_ne_nil _ _)
    obtain ⟨rE', eE'⟩ := ileave_head (anch sd' (cI L L' cp i')) u' (anch_ne_nil _ _)
    rw [eE, eE', lex_ne_heads h2, lex_ne_heads h0]
    exact h1.symm

/-- The peel of the first copy keeps the order of the slot digits' keys. -/
theorem PeelFirst.ord {v x : ℕ × List ℕ} (hv : v ∈ slots sd ks) (hx : x ∈ slots sd ks) :
    List.Lex (· < ·) (key sd' (psi0 L L' B cp v).1 (psi0 L L' B cp v).2)
        (key sd' (psi0 L L' B cp x).1 (psi0 L L' B cp x).2) ↔
      List.Lex (· < ·) (key sd v.1 v.2) (key sd x.1 x.2) := by
  obtain ⟨i, t⟩ := v
  obtain ⟨i', t'⟩ := x
  by_cases hc : inB L B i = true ∧ t.head? = some 0 <;>
    by_cases hc' : inB L B i' = true ∧ t'.head? = some 0
  · have p : psi0 L L' B cp (i, t) = (cI L L' cp i, t.tail) := by simp [psi0, hc]
    have p' : psi0 L L' B cp (i', t') = (cI L L' cp i', t'.tail) := by simp [psi0, hc']
    rw [p, p']
    exact hf.copy_copy hv hx hc hc'
  · have p : psi0 L L' B cp (i, t) = (cI L L' cp i, t.tail) := by simp [psi0, hc]
    rw [p]
    exact (hf.copy_kept hk hv hx hc hc').1
  · have p' : psi0 L L' B cp (i', t') = (cI L L' cp i', t'.tail) := by simp [psi0, hc']
    rw [p']
    exact (hf.copy_kept hk hx hv hc' hc).2
  · exact hf.kept_kept hk hv hx hc hc'

/-- The peel of the first copy keeps every slot digit's flat position. -/
theorem PeelFirst.fpos_psi {v : ℕ × List ℕ} (hv : v ∈ slots sd ks) :
    fpos sd' ks (psi0 L L' B cp v).1 (psi0 L L' B cp v).2 = fpos sd ks v.1 v.2 := by
  unfold fpos
  rw [((hf.perm hk).filter _).length_eq, List.filter_map, List.length_map]
  congr 1
  refine List.filter_congr fun w hw => ?_
  simp only [Function.comp_apply]
  exact decide_eq_decide.2 (hf.ord hk hw hv)

end first
/-- The reversal of `L`, the peel of the last copy of the reversal and the reversal of `L'` read
slot digit `(i, t)` of `L` at `psi0`. -/
theorem bndMap_rev_peel {L Lr R Rr : LDiagram} {B : List ℕ} {cp : ℕ → ℕ} {mb c : ℕ}
    {sd sd' : SlotData} (hf : PeelFirst L R B cp mb c sd sd') (hLb : Lr.base = L.base)
    (hRb : Rr.base = R.base) (hbnd : L.base.bnd.Nodup)
    (hcp : ∀ i < L.base.bnd.length, inB L B i = true → (cpP cp (L.bndAt i)).node ∉ B)
    {ks : ℕ → ℕ} (hk : ks mb = ks c + 1) (σ : ℕ → List ℕ → ℕ) (i : ℕ) (t : List ℕ)
    (hv : (i, t) ∈ slots sd ks) :
    bndMap L.base Lr.base (revφ B (ks mb)) (bndMap Lr.base Rr.base (peelφ B cp (ks c))
      (bndMap Rr.base R.base (revφ B (ks c)) σ)) i t =
      σ (psi0 L R B cp (i, t)).1 (psi0 L R B cp (i, t)).2 := by
  have hi : i < L.base.bnd.length := hf.n ▸ ((mem_slots _ _ _ _).1 hv).1
  have hp : L.base.bnd.getD i Port.dflt = L.bndAt i := rfl
  have hidx : L.base.bnd.idxOf (L.bndAt i) = i := by
    rw [bndAt, List.getD_eq_getElem _ _ hi, List.Nodup.idxOf_getElem hbnd]
  have hget : ∀ q ∈ R.base.bnd, R.base.bnd.getD (R.base.bnd.idxOf q) Port.dflt = q :=
    fun q hq => by
      rw [List.getD_eq_getElem _ _ (List.idxOf_lt_length_of_mem hq), List.getElem_idxOf]
  unfold bndMap
  rw [hLb, hRb]
  simp only [hp, revφ_fst, hidx]
  by_cases hc : inB L B i = true ∧ t.head? = some 0
  · have hB : (L.bndAt i).node ∈ B := by simpa [inB] using hc.1
    obtain ⟨u, rfl⟩ : ∃ u, t = 0 :: u := by
      cases t with
      | nil => simp at hc
      | cons j u => simp only [List.head?_cons, Option.some.injEq] at hc; exact ⟨u, by rw [hc.2]⟩
    obtain ⟨-, hci, -, -⟩ := hf.inn i hi hc.1
    have hq : cpP cp (L.bndAt i) ∈ R.base.bnd := List.idxOf_lt_length_iff.1 hci
    have e1 : revφ B (ks mb) (L.bndAt i) (0 :: u) = (L.bndAt i, ks c :: u) := by
      simp [revφ, hB, revT, hk]
    rw [e1]
    simp only [peelφ_B (ks c) _ hB u, hget _ hq]
    have hnB := hcp i hi hc.1
    rw [show revφ B (ks c) (cpP cp (L.bndAt i)) u = (cpP cp (L.bndAt i), u) from if_neg hnB]
    simp only [psi0, if_pos hc, List.tail_cons, cI]
  · by_cases hb : inB L B i = true
    · obtain ⟨j, u, rfl, hj, hjm, -⟩ := hf.kept_head hv hb hc
      have hB : (L.bndAt i).node ∈ B := by simpa [inB] using hb
      obtain ⟨hmi, -, -, -⟩ := hf.inn i hi hb
      have hq : L.bndAt i ∈ R.base.bnd := List.idxOf_lt_length_iff.1 hmi
      have e1 : revφ B (ks mb) (L.bndAt i) (j :: u) = (L.bndAt i, (ks c - j) :: u) := by
        simp [revφ, hB, revT, hk]
      rw [e1, peelφ_ne (ks c) _ _ u (by omega), hget _ hq]
      have e2 : revφ B (ks c) (L.bndAt i) ((ks c - j) :: u) = (L.bndAt i, (j - 1) :: u) := by
        simp only [revφ, if_pos hB, revT, Prod.mk.injEq, List.cons.injEq, and_true, true_and]
        omega
      rw [e2]
      simp only [psi0, if_neg hc, if_pos hb, decT, mI]
    · have hB : (L.bndAt i).node ∉ B := by simpa [inB] using hb
      have hb' : inB L B i = false := by simpa using hb
      obtain ⟨hmi, -⟩ := hf.out i hi hb'
      have hq : L.bndAt i ∈ R.base.bnd := List.idxOf_lt_length_iff.1 hmi
      rw [show revφ B (ks mb) (L.bndAt i) t = (L.bndAt i, t) from if_neg hB,
        peelφ_O (ks c) _ hB t, hget _ hq,
        show revφ B (ks c) (L.bndAt i) t = (L.bndAt i, t) from if_neg hB]
      simp only [psi0, if_neg hc, if_neg hb, mI]

/-- A peel of the first copy of an outermost chain box (the peel of the last copy of its
reversal, between reversals) keeps the linked digit sum read through the flat positions of both
sides. -/
theorem evalL_peel_first {L Lr R Rr : LDiagram} {bid mb c : ℕ} {B : List ℕ} {cp : ℕ → ℕ}
    {sd sd' : SlotData} (hL : RevOK L Lr bid mb B) (hL' : RevOK Lr L bid mb B)
    (hR : RevOK R Rr bid c B) (hR' : RevOK Rr R bid c B) (h : PeelOK Lr Rr bid mb c B cp)
    (hf : PeelFirst L R B cp mb c sd sd')
    (hcp : ∀ i < L.base.bnd.length, inB L B i = true → (cpP cp (L.bndAt i)).node ∉ B)
    (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (y : ℕ → ℕ) :
    R.evalL E ks (flatAt sd' ks y) = L.evalL E ks (flatAt sd ks y) := by
  rw [evalL_rev hR' hR, evalL_peel h E ks hk, evalL_rev hL hL']
  refine evalL_congr L E ks _ _ fun i hi t ht => ?_
  have hv : (i, t) ∈ slots sd ks := by
    refine (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, ?_⟩
    rw [hf.ch i hi, bndAt, List.getD_eq_getElem _ _ hi]
    exact ht
  rw [bndMap_rev_peel hf hL.base hR.base hL.bnd hcp hk _ i t hv]
  exact congrArg y (hf.fpos_psi hk hv)

/-- Equal digit sums with ghosts and equal slot data give equal digit sums read through the flat
positions. -/
theorem evalG_flatAt {L L' : LDiagram} {gs gs' : Ghosts} {G G' : List (ℕ × List (List ℕ))}
    (hs : L'.slotData G' = L.slotData G) (E : Env) (ks : ℕ → ℕ)
    (h : ∀ σ, L'.evalG gs' E ks σ = L.evalG gs E ks σ) (y : ℕ → ℕ) :
    L'.evalG gs' E ks (flatAt (L'.slotData G') ks y) =
      L.evalG gs E ks (flatAt (L.slotData G) ks y) := by
  rw [hs, h]

end LDiagram

end ArchytasZX
