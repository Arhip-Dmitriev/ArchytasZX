/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Order

/-!
# The expansion in key order

`expandK D ks`: the expansion with its boundary listed in key order: slot `i`'s copy `t` at the
place of `keyOf (slotSig i) i t` among all slot copies, lexicographically, where the key reads the
least slot of each node box of `i` (outermost first, up to the first that is `i` itself) with the
copy index in that box, then `i`, then the remaining node-box copy indices, then the port-box copy
indices innermost first. `eval_expandK_evalD`: its tensor entry is the digit sum with slot `i`'s copy `t`
read at that place.
-/

namespace ArchytasZX

namespace Diagram

variable (D : Diagram)

/-- The least slot on the side `o` (outputs when true) whose node is in the box `y`. -/
def mpB (y : DBox) (o : Bool) : ℕ :=
  ((List.range D.bnd.length).filter fun s =>
    decide (decide (s < D.outs.length) = o) && decide ((D.bnd.getD s Port.dflt).node ∈ y.nodes)).headD 0

/-- Slot `i`'s key data: the least slots of its node boxes, outermost first, up to the first that
is `i`, and the copy coordinates after `i` in key order: the remaining node boxes' outermost first,
then the port boxes' innermost first. -/
def slotSig (i : ℕ) : List ℕ × List ℕ :=
  let q := D.bnd.getD i Port.dflt
  let bs := D.nodeBoxes q.node
  let mps := bs.map (D.mpB · (decide (i < D.outs.length)))
  let r := mps.findIdx (· = i)
  (mps.take r, List.range' r (bs.length - r) ++ (List.range' bs.length (D.portBoxes q).length).reverse)

/-- Every slot's key data. -/
def keySig : List (List ℕ × List ℕ) := (List.range D.bnd.length).map D.slotSig

/-- The key of slot `i`'s copy `t` from the key data `sg`. -/
def keyOf (sg : List ℕ × List ℕ) (i : ℕ) (t : List ℕ) : List ℕ :=
  (sg.1.zip t).flatMap (fun p => [p.1, p.2]) ++ [i] ++ sg.2.map (t.getD · 0)

/-- Lexicographic order on lists of naturals. -/
def lexLe : List ℕ → List ℕ → Bool
  | [], _ => true
  | _ :: _, [] => false
  | a :: as, b :: bs => a < b || (a = b && lexLe as bs)

/-- Every slot copy, in key order. -/
def SL (ks : ℕ → ℕ) : List (ℕ × List ℕ) :=
  (D.SDom ks).insertionSort fun x y =>
    lexLe (keyOf (D.keySig.getD x.1 ([], [])) x.1 x.2) (keyOf (D.keySig.getD y.1 ([], [])) y.1 y.2)

/-- The expansion with its boundary in key order. -/
def expandK (ks : ℕ → ℕ) : Diagram :=
  let E := D.expand ks
  let L := (D.SL ks).map fun x => E.bnd.getD (D.bidx ks x).1 Port.dflt
  { E with outs := L.take E.outs.length, ins := L.drop E.outs.length }

/-- The slots of the expansion read in key order. -/
def readK (ks : ℕ → ℕ) (σ'' : ℕ → ℕ) (i : ℕ) (t : List ℕ) : ℕ := σ'' ((D.SL ks).idxOf (i, t))

variable {D}

theorem SDom_nodup (ks : ℕ → ℕ) : (D.SDom ks).Nodup := by
  unfold SDom
  refine List.nodup_flatMap.2 ⟨fun i _ => (cidx_nodup ks _).map fun a b e => by simpa using e, ?_⟩
  refine List.nodup_range.pairwise_of_forall_ne fun i _ j _ hij => ?_
  simp only [Function.onFun, List.disjoint_left, List.mem_map]
  rintro _ ⟨a, -, rfl⟩ ⟨b, -, e⟩
  exact hij (Prod.mk.inj e).1.symm

theorem SDom_boxfree (hbx : D.boxes = []) (ks : ℕ → ℕ) :
    D.SDom ks = (List.range D.bnd.length).map (·, []) := by
  unfold SDom
  rw [List.map_eq_flatMap]
  refine List.flatMap_congr fun i _ => ?_
  rw [legChain_nil hbx]
  rfl

theorem idxOf_map_inj {α β : Type*} [BEq α] [LawfulBEq α] [BEq β] [LawfulBEq β] (f : α → β)
    (l : List α) (hf : ∀ x ∈ l, ∀ y ∈ l, f x = f y → x = y) (x : α) (hx : x ∈ l) :
    (l.map f).idxOf (f x) = l.idxOf x := by
  induction l with
  | nil => simp at hx
  | cons a l ih =>
    by_cases h : a = x
    · subst h; simp
    · have hfa : f a ≠ f x := fun e => h (hf a (by simp) x hx e)
      simp only [List.map_cons, List.idxOf_cons, beq_iff_eq, hfa, h, cond_false]
      rw [ih (fun y hy z hz e => hf y (List.mem_cons_of_mem _ hy) z (List.mem_cons_of_mem _ hz) e)
        ((List.mem_cons.1 hx).resolve_left (Ne.symm h))]

theorem idxOf?_of_mem {l : List Port} (hnd : l.Nodup) {a : Port} (h : a ∈ l) :
    l.idxOf? a = some (l.idxOf a) := by
  rw [List.idxOf?_eq_some_iff]
  refine ⟨List.idxOf_lt_length_of_mem h, List.getElem_idxOf _, fun j hj e => ?_⟩
  have hj' : j < l.length := lt_trans hj (List.idxOf_lt_length_of_mem h)
  have := idxOf_getElem' hnd j hj'
  rw [e] at this
  omega

theorem value_perm (D D' : Diagram) (hw : D'.wires = D.wires) (hp : D'.bnd.Perm D.bnd)
    (hnd : D.bnd.Nodup) (σ τ : ℕ → ℕ) (p : Port) :
    D'.value σ τ p = D.value (fun s => σ (D'.bnd.idxOf (D.bnd.getD s Port.dflt))) τ p := by
  have hnd' : D'.bnd.Nodup := hp.nodup_iff.2 hnd
  unfold value locate
  by_cases hm : p ∈ D.bnd
  · rw [idxOf?_of_mem hnd hm, idxOf?_of_mem hnd' (hp.mem_iff.2 hm)]
    simp only
    rw [List.getD_eq_getElem _ _ (List.idxOf_lt_length_of_mem hm), List.getElem_idxOf]
  · rw [List.idxOf?_eq_none_iff.2 hm, List.idxOf?_eq_none_iff.2 (fun h => hm (hp.mem_iff.1 h)), hw]
    generalize List.findIdx? (fun w => decide (w.1 = p ∨ w.2 = p)) D.wires = r
    cases r <;> rfl

/-- A diagram with its boundary permuted reads the slots through the permutation. -/
theorem eval_perm (D D' : Diagram) (hn : D'.nodes = D.nodes) (hw : D'.wires = D.wires)
    (hp : D'.bnd.Perm D.bnd) (hnd : D.bnd.Nodup) (E : Env) (σ : ℕ → ℕ) :
    D'.eval E σ = D.eval E (fun s => σ (D'.bnd.idxOf (D.bnd.getD s Port.dflt))) := by
  have hwd : ∀ j, D'.wireDim j = D.wireDim j := fun j => by
    unfold wireDim portDim; rw [hn, hw]
  unfold Diagram.eval
  rw [show (fun j => E.dimv (D'.wireDim j)) = fun j => E.dimv (D.wireDim j) from funext fun j => by
    rw [hwd], hw, hn]
  congr 1
  funext τ
  congr 1
  refine List.map_congr_left fun n _ => ?_
  congr 2
  funext p
  exact value_perm D D' hw hp hnd σ τ p

theorem range_getD (l : List Port) : (List.range l.length).map (fun i => l.getD i Port.dflt) = l := by
  refine List.ext_getElem (by simp) fun i h1 h2 => ?_
  simp [List.getElem?_eq_getElem h2]

theorem SL_perm (ks : ℕ → ℕ) : (D.SL ks).Perm (D.SDom ks) := List.perm_insertionSort _ _

section ord

variable (hW : WF D) (ks : ℕ → ℕ)

include hW in
theorem bidx_slot (x : ℕ × List ℕ) (hx : x ∈ D.SDom ks) :
    (D.bidx ks x).2 = [] ∧ (D.bidx ks x).1 < (D.expand ks).bnd.length := by
  obtain ⟨-, -, hbe⟩ := evalD_expand D ks ⟨fun _ => 0, fun _ _ => 0, [], fun _ => 0, fun _ _ => 1⟩ hW
    (fun _ _ => 0)
  have hm := (bidx_bij hW).mem x hx
  rw [SDom_boxfree hbe] at hm
  obtain ⟨s, hs, e⟩ := List.mem_map.1 hm
  rw [← e]
  exact ⟨rfl, List.mem_range.1 hs⟩

include hW in
theorem L_perm : ((D.SL ks).map fun x => (D.expand ks).bnd.getD (D.bidx ks x).1 Port.dflt).Perm
    (D.expand ks).bnd := by
  obtain ⟨-, hWe, hbe⟩ := evalD_expand D ks ⟨fun _ => 0, fun _ _ => 0, [], fun _ => 0, fun _ _ => 1⟩
    hW (fun _ _ => 0)
  have hb := bidx_bij (ks := ks) hW
  have h1 : ((D.SDom ks).map (D.bidx ks)).Perm ((D.expand ks).SDom ks) := by
    refine (List.perm_ext_iff_of_nodup ((SDom_nodup ks).map_on fun x hx y hy e => hb.inj x hx y hy e)
      (SDom_nodup ks)).2 fun y => ?_
    constructor
    · intro hy
      obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hy
      exact hb.mem x hx
    · intro hy
      obtain ⟨x, hx, rfl⟩ := hb.onto y hy
      exact List.mem_map.2 ⟨x, hx, rfl⟩
  have h2 := (SL_perm (D := D) ks).map
    (fun x => (D.expand ks).bnd.getD (D.bidx ks x).1 Port.dflt)
  refine h2.trans ?_
  have h3 := h1.map (fun y => (D.expand ks).bnd.getD y.1 Port.dflt)
  rw [List.map_map, SDom_boxfree hbe, List.map_map] at h3
  simp only [Function.comp_def] at h3
  rw [range_getD] at h3
  exact h3

include hW in
theorem L_inj : ∀ x ∈ D.SL ks, ∀ y ∈ D.SL ks,
    (D.expand ks).bnd.getD (D.bidx ks x).1 Port.dflt = (D.expand ks).bnd.getD (D.bidx ks y).1 Port.dflt →
      x = y := by
  obtain ⟨-, hWe, -⟩ := evalD_expand D ks ⟨fun _ => 0, fun _ _ => 0, [], fun _ => 0, fun _ _ => 1⟩
    hW (fun _ _ => 0)
  intro x hx y hy e
  have hx' := (SL_perm ks).mem_iff.1 hx
  have hy' := (SL_perm ks).mem_iff.1 hy
  obtain ⟨x2, hx1⟩ := bidx_slot hW ks x hx'
  obtain ⟨y2, hy1⟩ := bidx_slot hW ks y hy'
  rw [List.getD_eq_getElem _ _ hx1, List.getD_eq_getElem _ _ hy1] at e
  have := (List.Nodup.getElem_inj_iff (WF.bnd_nodup hWe)).1 e
  exact (bidx_bij hW).inj x hx' y hy' (Prod.ext this (x2.trans y2.symm))

end ord

/-- The expansion in key order is the digit sum with the slots read in key order. -/
theorem eval_expandK_evalD (hW : WF D) (E : Env) (σ'' : ℕ → ℕ)
    (hσ : (D.expandK E.mult).InRange E σ'') :
    (D.expandK E.mult).eval E σ'' = D.evalD E E.mult (D.readK E.mult σ'') := by
  set ks := E.mult
  obtain ⟨-, hWe, -⟩ := evalD_expand D ks ⟨fun _ => 0, fun _ _ => 0, [], fun _ => 0, fun _ _ => 1⟩
    hW (fun _ _ => 0)
  set Ex := D.expand ks
  set h : ℕ × List ℕ → Port := fun x => Ex.bnd.getD (D.bidx ks x).1 Port.dflt
  set L := (D.SL ks).map h
  have hbK : (D.expandK ks).bnd = L := by
    simp only [expandK, bnd]; exact List.take_append_drop _ _
  have hpL : L.Perm Ex.bnd := L_perm hW ks
  have hndE := WF.bnd_nodup hWe
  have hdimK : (D.expandK ks).portDim = Ex.portDim := rfl
  rw [eval_perm Ex (D.expandK ks) rfl rfl (hbK ▸ hpL) hndE, hbK]
  rw [eval_expand_evalD hW E _ fun s hs => ?_]
  · refine evalD_slots hW E ks _ _ fun i hi t ht => ?_
    have hx : (i, t) ∈ D.SDom ks := (mem_SDom D ks _).2 ⟨hi, ht⟩
    have hx' : (i, t) ∈ D.SL ks := (SL_perm ks).mem_iff.2 hx
    rw [bmap_read]
    simp only [readK]
    exact congrArg σ'' (idxOf_map_inj h _ (L_inj hW ks) _ hx')
  · have hs' : s < Ex.bnd.length := hs
    have hmem : Ex.bnd[s] ∈ L := hpL.mem_iff.2 (List.getElem_mem hs')
    have hr := List.idxOf_lt_length_of_mem hmem
    have := hσ (L.idxOf Ex.bnd[s]) (by rw [hbK]; exact hr)
    simp only [bdims, hbK, List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_eq_getElem hr, Option.map_some, Option.getD_some, List.getElem_idxOf, hdimK]
      at this
    have key : σ'' (L.idxOf (Ex.bnd.getD s Port.dflt)) < E.dimv (Ex.bdims.getD s 0) := by
      simp only [List.getD_eq_getElem _ _ hs', bdims, List.getD_eq_getElem?_getD, List.getElem?_map,
        List.getElem?_eq_getElem hs', Option.map_some, Option.getD_some]
      exact this
    exact key

theorem bdims_expandK (hW : WF D) (ks : ℕ → ℕ) :
    (D.expandK ks).bdims = (D.SL ks).map fun x => D.portDim (D.bnd.getD x.1 Port.dflt) := by
  have hbK : (D.expandK ks).bnd = (D.SL ks).map fun x => (D.expand ks).bnd.getD (D.bidx ks x).1 Port.dflt := by
    simp only [expandK, bnd]; exact List.take_append_drop _ _
  have hdimK : (D.expandK ks).portDim = (D.expand ks).portDim := rfl
  rw [bdims, hbK, hdimK, List.map_map]
  refine List.map_congr_left fun x hx => ?_
  exact (bidx_bij hW).dim x ((SL_perm ks).mem_iff.1 hx)

end Diagram

end ArchytasZX
