/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PortBox

/-!
# Gathering ancestors' boundary ports

`gather` only reorders the boundary: the digit sum of the gathered diagram is the digit sum of the
diagram with the slots read through the reordering.
-/

namespace ArchytasZX

namespace Diagram

theorem lookup_ident (l : List ℕ) (x : ℕ) (hx : x ∈ l) : (l.map fun y => (y, y)).lookup x = some x := by
  induction l with
  | nil => simp at hx
  | cons a l ih =>
    by_cases h : x = a
    · subst h; simp [List.lookup]
    · have hb : (x == a) = false := by simpa using h
      simp only [List.map_cons, List.lookup, hb]
      exact ih (by simpa [h] using hx)

theorem splice_ident_perm (refs : List Port) (nodes : List ℕ) (old : List Port)
    (hold : ∀ r ∈ old, r.node ∈ nodes) :
    (splice refs old [nodes.map fun x => (x, x)]).Perm refs := by
  unfold splice
  simp only
  split_ifs with h
  · exact List.Perm.refl _
  · set at_ := refs.findIdx (· ∈ old)
    have hpre : ∀ r ∈ refs.take at_, r ∉ old := fun r hr => by
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hr
      simp only [List.getElem_take]
      have hlt : i < at_ := by simp at hi; omega
      have := List.not_of_lt_findIdx (xs := refs) (p := fun r => decide (r ∈ old)) hlt
      simpa using this
    have hblock : ([nodes.map fun x => (x, x)].flatMap fun mp => (refs.filter (· ∈ old)).map
        fun r => (⟨((mp.lookup r.node).getD r.node), r.out, r.pos⟩ : Port)) =
        refs.filter (· ∈ old) := by
      simp only [List.flatMap_singleton]
      conv_rhs => rw [← List.map_id (refs.filter (· ∈ old))]
      refine List.map_congr_left fun r hr => ?_
      have := hold r (of_decide_eq_true (List.mem_filter.1 hr).2)
      simp [lookup_ident nodes r.node this]
    rw [hblock]
    have ht : (refs.take at_).filter (· ∉ old) = refs.take at_ :=
      List.filter_eq_self.2 fun r hr => by simpa using hpre r hr
    have hd : refs.filter (· ∈ old) = (refs.drop at_).filter (· ∈ old) := by
      conv_lhs => rw [← List.take_append_drop at_ refs]
      rw [List.filter_append, List.filter_eq_nil_iff.2 fun r hr => by simpa using hpre r hr,
        List.nil_append]
    rw [ht, hd]
    conv_rhs => rw [← List.take_append_drop at_ refs]
    rw [List.append_assoc]
    refine List.Perm.append_left _ ?_
    exact (List.filter_append_perm _ _).trans (by simp)

theorem gather_go (D : Diagram) (anc : Option ℕ) (fuel : ℕ) :
    ∃ o i, gather.go D anc fuel = { D with outs := o, ins := i } ∧ o.Perm D.outs ∧ i.Perm D.ins := by
  induction fuel generalizing D anc with
  | zero => exact ⟨D.outs, D.ins, rfl, List.Perm.refl _, List.Perm.refl _⟩
  | succ fuel ih =>
    simp only [gather.go]
    split
    · exact ⟨D.outs, D.ins, rfl, List.Perm.refl _, List.Perm.refl _⟩
    · next outer _ =>
      split_ifs
      · exact ih D outer.parent
      · obtain ⟨o, i, e, ho, hi⟩ := ih { D with
            ins := (splice D.ins (D.scopeBnd outer.nodes) [outer.nodes.map (fun x => (x, x))]),
            outs := (splice D.outs (D.scopeBnd outer.nodes) [outer.nodes.map (fun x => (x, x))]) }
          outer.parent
        have hold : ∀ r ∈ D.scopeBnd outer.nodes, r.node ∈ outer.nodes := fun r hr => by
          unfold scopeBnd at hr
          exact of_decide_eq_true (List.mem_filter.1 ((List.mergeSort_perm _ _).mem_iff.1 hr)).2
        refine ⟨o, i, e, ho.trans (splice_ident_perm _ _ _ hold),
          hi.trans (splice_ident_perm _ _ _ hold)⟩

theorem gather_eq (D : Diagram) (b : DBox) :
    ∃ o i, D.gather b = { D with outs := o, ins := i } ∧ o.Perm D.outs ∧ i.Perm D.ins :=
  gather_go D b.parent D.boxes.length

section congr

variable {D D' : Diagram} (hn : D'.nodes = D.nodes) (hw : D'.wires = D.wires)
  (hb : D'.boxes = D.boxes)

include hb in
theorem walk_congr (p : DBox → Bool) (f : ℕ) (s : Option ℕ) : D'.walk p f s = D.walk p f s := by
  induction f generalizing s with
  | zero => rfl
  | succ f ih => simp only [walk, hb, ih]

include hb in
theorem nodeBoxes_congr (x : ℕ) : D'.nodeBoxes x = D.nodeBoxes x := by
  unfold nodeBoxes; rw [walk_congr hb, hb]

include hb in
theorem nodeChain_congr (x : ℕ) : D'.nodeChain x = D.nodeChain x := by
  unfold nodeChain; rw [nodeBoxes_congr hb]

include hb in
theorem portBoxes_congr (q : Port) : D'.portBoxes q = D.portBoxes q := by
  unfold portBoxes; rw [walk_congr hb, hb, nodeBoxes_congr hb]

include hw in
theorem partner_congr (q : Port) : D'.partner q = D.partner q := by
  unfold partner; rw [hw]

include hw hb in
theorem tail_congr (q : Port) : D'.tail q = D.tail q := by
  unfold tail; simp only [partner_congr hw, nodeChain_congr hb, portBoxes_congr hb]

include hw hb in
theorem legChain_congr (q : Port) : D'.legChain q = D.legChain q := by
  unfold legChain; rw [nodeChain_congr hb, tail_congr hw hb]

include hn in
theorem legs_congr : D'.legs = D.legs := by unfold legs; rw [hn]

include hn hw hb in
theorem dlabs_congr (ks : ℕ → ℕ) : D'.dlabs ks = D.dlabs ks := by
  unfold dlabs; rw [legs_congr hn]; simp only [legChain_congr hw hb]

include hn in
theorem portDim_congr (q : Port) : D'.portDim q = D.portDim q := by unfold portDim; rw [hn]

include hw hb in
theorem dnode_congr (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) : D'.dnode ks n t = D.dnode ks n t := by
  unfold dnode; simp only [tail_congr hw hb]

include hn hw hb in
theorem prodD_congr' (E : Env) (ks : ℕ → ℕ) (τ : ℕ → ℕ) : D'.prodD E ks τ = D.prodD E ks τ := by
  unfold prodD; rw [hn]; simp only [nodeChain_congr hb, dnode_congr hw hb]

include hb in
theorem ChainOK_congr (pr : DBox → Bool) (start : Option ℕ) (L : List DBox) :
    ChainOK D' pr start L ↔ ChainOK D pr start L := by unfold ChainOK; rw [hb]

end congr

theorem prodD_congr (D : Diagram) (E : Env) (ks : ℕ → ℕ) (τ τ' : ℕ → ℕ)
    (h : ∀ x ∈ D.dlabs ks, τ x = τ' x) : D.prodD E ks τ = D.prodD E ks τ' := by
  unfold prodD
  congr 1
  refine List.map_congr_left fun n hn => ?_
  congr 1
  refine List.map_congr_left fun t ht => ?_
  exact val_reads E _ _ _ fun x hx => h x (dnode_labels_mem ks hn ht hx)

theorem bnd_perm (D : Diagram) (o i : List Port) (ho : o.Perm D.outs) (hi : i.Perm D.ins) :
    ({ D with outs := o, ins := i } : Diagram).bnd.Perm D.bnd := ho.append hi

theorem WF_perm_bnd (D : Diagram) (o i : List Port) (ho : o.Perm D.outs) (hi : i.Perm D.ins)
    (hW : WF D) : WF { D with outs := o, ins := i } := by
  have hp := bnd_perm D o i ho hi
  set D' : Diagram := { D with outs := o, ins := i }
  have hb : D'.boxes = D.boxes := rfl
  have hw : D'.wires = D.wires := rfl
  have hn : D'.nodes = D.nodes := rfl
  refine ⟨hW.ids, hW.bids, ?_, hW.kinds, hW.nsub, fun x hx => ?_, fun x y hy hx => ?_,
    fun p h => ?_, fun p h y hy hpy => ?_, fun y hy p h => hp.mem_iff.2 (hW.pbnd y hy p h), ?_, ?_, ?_⟩
  · rw [legs_congr hn]; exact ((hp.append_left _).append_left _).trans hW.perm
  · rw [ChainOK_congr hb, nodeBoxes_congr hb]; exact hW.nchain x hx
  · rw [nodeBoxes_congr hb]; exact hW.nall x y hy hx
  · rw [ChainOK_congr hb, nodeBoxes_congr hb, portBoxes_congr hb]; exact hW.pchain p (hp.mem_iff.1 h)
  · rw [portBoxes_congr hb]; exact hW.pall p (hp.mem_iff.1 h) y hy hpy
  · intro w hw'; rw [nodeBoxes_congr hb, nodeBoxes_congr hb]; exact hW.wpre w hw'
  · exact hW.ne
  · intro w hw'; rw [portDim_congr hn, portDim_congr hn]; exact hW.wdim w hw'

/-- Reordering the boundary reads the slots through the reordering. -/
theorem evalD_perm_bnd (D : Diagram) (o i : List Port) (ho : o.Perm D.outs) (hi : i.Perm D.ins)
    (hW : WF D) (E : Env) (ks : ℕ → ℕ) (σ' : ℕ → List ℕ → ℕ) :
    ({ D with outs := o, ins := i } : Diagram).evalD E ks σ' =
      D.evalD E ks (bndMap D { D with outs := o, ins := i } (fun p t => (p, t)) σ') := by
  have hp := bnd_perm D o i ho hi
  set D' : Diagram := { D with outs := o, ins := i }
  have hb : D'.boxes = D.boxes := rfl
  have hw : D'.wires = D.wires := rfl
  have hn : D'.nodes = D.nodes := rfl
  have hlegs := legs_nodup_of D hW.ids
  have hbnd : D.bnd.Nodup := by
    have := hW.perm.nodup_iff.2 hlegs
    exact ((List.nodup_append.1 this).2.1 |> List.nodup_append.1).2.1
  have hlc := legChain_congr hw hb
  refine evalD_relabelφ D D' E ks ks σ' (fun p t => (p, t)) (dlabs_nodup D ks hlegs)
    (by rw [dlabs_congr hn hw hb]; exact dlabs_nodup D ks hlegs) (hp.nodup_iff.2 hbnd)
    (fun p hp t ht => ⟨by rw [legs_congr hn]; exact hp, by simp only [hlc]; exact ht⟩)
    (fun q hq u hu => ⟨q, by rw [legs_congr hn] at hq; exact hq, u, by rw [hlc] at hu; exact hu, rfl⟩)
    (fun p _ t _ p' _ t' _ h => Prod.mk.inj h)
    (fun p _ t _ => by simp only [portDim_congr hn]) (fun w hw => ?_)
    (fun w hw t ht => ⟨w, hw, t, by simp only [hlc]; exact ht, rfl, rfl⟩)
    (fun w hw' u hu => ⟨w, hw', u, by rw [hlc] at hu; exact hu, rfl, rfl⟩)
    (fun p hp' => bnd_legs hW hp')
    (fun p hp' t _ => hp.mem_iff.2 hp')
    (fun q hq u hu => ⟨q, hp.mem_iff.1 hq, u, by rw [hlc] at hu; exact hu, rfl⟩)
    (fun τ τ' hl => ?_)
  · obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
    exact ⟨l1, l2, legChain_wire hW hw⟩
  · rw [prodD_congr' hn hw hb]
    refine prodD_congr D E ks τ τ' fun x hx => ?_
    rw [hl x hx]
    obtain ⟨p, -, t, -, rfl⟩ := (mem_dlabs D ks x).1 hx
    rw [relab_dlab]

/-- The slot reading of a port box's instantiation at `k`. -/
def portMap (D : Diagram) (b : DBox) (k : ℕ) (σ' : ℕ → List ℕ → ℕ) : ℕ → List ℕ → ℕ :=
  let D₀ := if k = 0 then D.gather b else D
  bndMap D D₀ (fun p t => (p, t)) (bndMap D₀ (D₀.growPorts b k) (φP b k) σ')

/-- Instantiating a leaf port box keeps the digit sum, the slots read through `portMap`. -/
theorem evalD_expandPort (D : Diagram) (b : DBox) (ks : ℕ → ℕ) (hW : WF D) (hb : b ∈ D.boxes)
    (hbn : b.nodes = []) (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (E : Env)
    (σ' : ℕ → List ℕ → ℕ) :
    (D.expandPort b (ks b.mult)).evalD E ks σ' = D.evalD E ks (portMap D b (ks b.mult) σ') ∧
      WF (D.expandPort b (ks b.mult)) := by
  obtain ⟨o, i, e, ho, hi⟩ := gather_eq D b
  set D₀ := if ks b.mult = 0 then D.gather b else D with hD₀
  set o' := if ks b.mult = 0 then o else D.outs
  set i' := if ks b.mult = 0 then i else D.ins
  have hD₀' : D₀ = { D with outs := o', ins := i' } := by
    rw [hD₀]; split_ifs with h
    · simp only [o', i', h, ↓reduceIte]; exact e
    · simp only [o', i', h, ↓reduceIte]
  have ho' : o'.Perm D.outs := by simp only [o']; split_ifs; exacts [ho, List.Perm.refl _]
  have hi' : i'.Perm D.ins := by simp only [i']; split_ifs; exacts [hi, List.Perm.refl _]
  have hW₀ : WF D₀ := hD₀' ▸ WF_perm_bnd D _ _ ho' hi' hW
  have hb₀ : D₀.boxes = D.boxes := by rw [hD₀']
  have hexp : D.expandPort b (ks b.mult) = D₀.growPorts b (ks b.mult) := rfl
  rw [hexp]
  refine ⟨?_, ep_WF D₀ b _ hW₀ (hb₀ ▸ hb) hbn (hb₀ ▸ hleaf)⟩
  rw [evalD_growPorts D₀ b (ks := ks) hW₀ (hb₀ ▸ hb) hbn (hb₀ ▸ hleaf) E σ']
  unfold portMap
  simp only [← hD₀]
  rw [hD₀']
  exact evalD_perm_bnd D _ _ ho' hi' hW E ks _

end Diagram

end ArchytasZX
