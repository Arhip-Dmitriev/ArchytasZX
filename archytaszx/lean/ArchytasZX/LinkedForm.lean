/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedSlide
import ArchytasZX.LinkedRegroup

/-!
# Node order, identity drops in place and slot data of linked diagrams

`withNodes L ns`: `L` with the node list `ns`; `evalL_withNodes`: a permutation of the nodes keeps
the linked digit sum. `dropAt L x w v P`: `x` moved last, then dropped (`dropId`);
`evalL_dropAt`: `DropOK` there keeps the sum when `x`'s phase index reads zero.

`slotData L gs`: the boundary sizes and, per box, its id, multiplicity index, parent and slot
blocks, the box groups `gs` (box id, node lists) first, its ungrouped nodes last, each block the
boundary slots of its nodes in the box, empty blocks left out. `evalL_flat`: two linked diagrams
with equal digit sums at every slot assignment and equal slot data have equal digit sums read
through any layout `lay` of their slot data.
-/

namespace ArchytasZX

namespace LDiagram

/-- `L` with the node list `ns`. -/
def withNodes (L : LDiagram) (ns : List DNode) : LDiagram :=
  { L with base := { L.base with nodes := ns } }

/-- `L` with the nodes of id `x` last. -/
def moveLast (L : LDiagram) (x : ℕ) : LDiagram :=
  L.withNodes (L.base.nodes.filter (fun n => n.id ≠ x) ++ L.base.nodes.filter (fun n => n.id = x))

/-- `x` moved last, then dropped. -/
def dropAt (L : LDiagram) (x : DNode) (w : Port × Port) (v P : Port) : LDiagram :=
  (L.moveLast x.id).dropId x.id w v P

theorem portDim_withNodes (D : Diagram) (ns : List DNode) (hp : ns.Perm D.nodes)
    (hid : (D.nodes.map (·.id)).Nodup) (p : Port) :
    ({ D with nodes := ns } : Diagram).portDim p = D.portDim p := by
  have hid' : (({ D with nodes := ns } : Diagram).nodes.map (·.id)).Nodup :=
    (hp.map _).nodup_iff.2 hid
  by_cases h : ∃ n ∈ D.nodes, p.node = n.id
  · obtain ⟨n, hn, e⟩ := h
    rw [portDim_of _ hid' n (hp.mem_iff.2 hn) p e, portDim_of D hid n hn p e]
  · have h1 : D.nodes.find? (fun m => decide (m.id = p.node)) = none :=
      List.find?_eq_none.2 fun m hm e => h ⟨m, hm, (of_decide_eq_true e).symm⟩
    have h2 : ns.find? (fun m => decide (m.id = p.node)) = none :=
      List.find?_eq_none.2 fun m hm e => h ⟨m, hp.mem_iff.1 hm, (of_decide_eq_true e).symm⟩
    unfold Diagram.portDim
    rw [h1]
    exact (congrArg (fun o : Option DNode => match o with
      | some n => if p.out then n.outs.getD p.pos 0 else n.ins.getD p.pos 0
      | none => 0) h2)

section withNodes

variable (L : LDiagram) (ns : List DNode)

theorem nodeBoxes_wn : (L.withNodes ns).base.nodeBoxes = L.base.nodeBoxes :=
  funext fun _ => Diagram.walk_boxes_congr (L.withNodes ns).base L.base rfl _ _ _

theorem portBoxes_wn : (L.withNodes ns).base.portBoxes = L.base.portBoxes := by
  funext q
  unfold Diagram.portBoxes
  rw [nodeBoxes_wn]
  exact Diagram.walk_boxes_congr (L.withNodes ns).base L.base rfl _ _ _

theorem nodeChain_wn : (L.withNodes ns).base.nodeChain = L.base.nodeChain := by
  funext x
  unfold Diagram.nodeChain
  rw [nodeBoxes_wn]

theorem tail_wn : (L.withNodes ns).base.tail = L.base.tail := by
  funext q
  unfold Diagram.tail
  rw [nodeChain_wn, portBoxes_wn]
  rfl

theorem legChain_wn : (L.withNodes ns).base.legChain = L.base.legChain := by
  funext q
  unfold Diagram.legChain
  rw [nodeChain_wn, tail_wn]

theorem depth_wn : (L.withNodes ns).depth = L.depth := by
  funext y x
  unfold depth
  rw [nodeBoxes_wn]

theorem pinsL_wn : (L.withNodes ns).pinsL = L.pinsL := by
  funext ks ν
  unfold pinsL dpairs linkPairs
  rw [legChain_wn, depth_wn]
  rfl

theorem dnodeL_wn : (L.withNodes ns).dnodeL = L.dnodeL := by
  funext ks n t
  unfold dnodeL Diagram.dnode phAt
  rw [tail_wn, depth_wn]
  rfl

end withNodes

/-- A permutation of the nodes keeps the linked digit sum. -/
theorem evalL_withNodes (L : LDiagram) (ns : List DNode) (hp : ns.Perm L.base.nodes)
    (hid : (L.base.nodes.map (·.id)).Nodup) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    (L.withNodes ns).evalL E ks σ = L.evalL E ks σ := by
  have hl : ((L.withNodes ns).base.dlabs ks).Perm (L.base.dlabs ks) := by
    unfold Diagram.dlabs Diagram.legs
    rw [legChain_wn]
    exact (hp.flatMap_right DNode.legs).flatMap_right
      (fun p => (cidx ks (L.base.legChain p)).map (dlab p))
  have hd : (fun l => E.dimv ((L.withNodes ns).base.portDim (dport l))) =
      fun l => E.dimv (L.base.portDim (dport l)) :=
    funext fun l => by rw [withNodes, portDim_withNodes L.base ns hp hid]
  have hP : ∀ ν, (L.withNodes ns).prodL E ks ν = L.prodL E ks ν := fun ν => by
    unfold prodL
    rw [dnodeL_wn, nodeChain_wn]
    exact (hp.map fun (n : DNode) => ((cidx ks (L.base.nodeChain n.id)).map fun t =>
      val E ν (L.dnodeL ks n t)).prod).prod_eq
  unfold evalL
  rw [sumOver_perm _ hl, hd]
  congr 1
  funext ν
  rw [hP ν, pinsL_wn]

theorem moveLast_perm (L : LDiagram) (x : ℕ) :
    (L.base.nodes.filter (fun n => n.id ≠ x) ++ L.base.nodes.filter (fun n => n.id = x)).Perm
      L.base.nodes := by
  have e : L.base.nodes.filter (fun n => n.id = x) =
      L.base.nodes.filter (fun n => !decide (n.id ≠ x)) :=
    List.filter_congr fun n _ => by simp
  rw [e]
  exact List.filter_append_perm _ _

/-- Dropping `x` after moving it last keeps the linked digit sum. -/
theorem evalL_dropAt {L : LDiagram} {x : DNode} {w : Port × Port} {u v P : Port}
    (h : DropOK (L.moveLast x.id) x w u v P) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalL E ks σ = (L.dropAt x w v P).evalL E ks σ := by
  have hp := moveLast_perm L x.id
  have hid : (L.base.nodes.map (·.id)).Nodup := (hp.map _).nodup_iff.1 h.ids
  rw [← evalL_withNodes L _ hp hid]
  exact h.evalL_drop E hE ks σ

/-- The boundary slots whose node lies in `g` and in the box `b`. -/
def slotsIn (L : LDiagram) (b : DBox) (g : List ℕ) : List ℕ :=
  (List.range L.base.bnd.length).filter fun i =>
    (L.base.bnd[i]?).any fun p => decide (p.node ∈ g ∧ p.node ∈ b.nodes)

/-- The slot blocks of the box `b`: its groups in `gs`, then its ungrouped nodes, empty blocks
left out. -/
def slotParts (L : LDiagram) (gs : List (ℕ × List (List ℕ))) (b : DBox) : List (List ℕ) :=
  let g := (gs.lookup b.id).getD []
  ((g ++ [b.nodes.filter fun n => g.all fun h => decide (n ∉ h)]).map (L.slotsIn b)).filter
    fun s => !s.isEmpty

/-- Slot data: output and input counts, then per box its id, multiplicity index, parent and slot
blocks. -/
abbrev SlotData := ℕ × ℕ × List (ℕ × ℕ × Option ℕ × List (List ℕ))

instance : DecidableEq SlotData := fun a b => by
  set_option synthInstance.maxSize 512 in exact inferInstance

/-- The slot data of `L` under the box groups `gs`. -/
def slotData (L : LDiagram) (gs : List (ℕ × List (List ℕ))) : SlotData :=
  (L.base.outs.length, L.base.ins.length,
    L.base.boxes.map fun b => (b.id, b.mult, b.parent, L.slotParts gs b))

/-- Equal digit sums and equal slot data give equal digit sums read through any layout. -/
theorem evalL_flat {L L' : LDiagram} {gs gs' : List (ℕ × List (List ℕ))}
    (hs : L'.slotData gs' = L.slotData gs) (E : Env) (ks : ℕ → ℕ)
    (h : ∀ σ, L'.evalL E ks σ = L.evalL E ks σ) (lay : SlotData → (ℕ → ℕ) → List (ℕ × List ℕ))
    (x : ℕ → ℕ) :
    L'.evalL E ks (readAt (lay (L'.slotData gs') ks) x) =
      L.evalL E ks (readAt (lay (L.slotData gs) ks) x) := by
  rw [hs, h]

end LDiagram

end ArchytasZX
