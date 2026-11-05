/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Relabel
import ArchytasZX.ExpandFacts

/-!
# Instantiating a port box

The digits of a diagram and of its port box `b` instantiated at `k ≥ 1` correspond: digit
`(p, t ++ [j])` of a leg `p` in `b` is digit `(j-th leg p grows to, t)`.
-/

namespace ArchytasZX

namespace Diagram

theorem legs_nodup_of (D : Diagram) (h : (D.nodes.map (·.id)).Nodup) : D.legs.Nodup := by
  unfold legs
  refine List.nodup_flatMap.2 ⟨fun n _ => ?_, ?_⟩
  · simp only [DNode.legs]
    refine List.nodup_append.2 ⟨(List.nodup_range).map fun a b e => by simpa using e,
      (List.nodup_range).map fun a b e => by simpa using e, ?_⟩
    simp
  · refine (List.pairwise_map.1 h).imp fun {a c} hac => ?_
    simp only [Function.onFun, List.disjoint_left, DNode.legs, List.mem_append, List.mem_map]
    rintro q (⟨i, -, rfl⟩ | ⟨i, -, rfl⟩) (⟨j, -, e⟩ | ⟨j, -, e⟩) <;>
      exact hac (congrArg Port.node e).symm

/-- Whether a port is in the port box `b`. -/
def pg (b : DBox) (p : Port) : Bool := decide (p ∈ b.ports)

theorem pg_def (b : DBox) : pg b = fun p => decide (p ∈ b.ports) := rfl

/-- The grown node of the port box `b` at `k`. -/
def pgrow (b : DBox) (k : ℕ) (n : DNode) : DNode :=
  { n with outs := growSide (pg b) k n.id true n.outs, ins := growSide (pg b) k n.id false n.ins }

/-- A port box's boxes: `b` dropped, every port its image. -/
def pbox (b : DBox) (k : ℕ) (x : DBox) : DBox := { x with ports := x.ports.flatMap (growImg (pg b) k) }

variable (D : Diagram) (b : DBox) (k : ℕ)

theorem ep_nodes : (D.growPorts b k).nodes = D.nodes.map (pgrow b k) := by
  simp [growPorts, pgrow, pg_def]

theorem ep_legs : (D.growPorts b k).legs = D.legs.flatMap (growImg (pg b) k) := by
  simp only [legs, ep_nodes D b k, List.flatMap_map, pgrow, legs_grow, List.flatMap_assoc]

theorem ep_wires : (D.growPorts b k).wires =
    D.wires.flatMap fun w => (growImg (pg b) k w.1).zip (growImg (pg b) k w.2) := by
  simp [growPorts, pg_def]

theorem ep_bnd : (D.growPorts b k).bnd = D.bnd.flatMap (growImg (pg b) k) := by
  simp [growPorts, pg_def, bnd, List.flatMap_append]

theorem ep_boxes : (D.growPorts b k).boxes =
    ((D.boxes.filter (·.id ≠ b.id)).map (pbox b k)).filter fun x => x.nodes ≠ [] ∨ x.ports ≠ [] := by
  rfl

theorem growImg_of_not (g : Port → Bool) (k : ℕ) (p : Port) (h : g p = false) :
    growImg g k p = [⟨p.node, p.out, shiftPos g k p⟩] := by simp [growImg, h]

theorem growImg_of_pos (g : Port → Bool) (k : ℕ) (p : Port) (h : g p = true) :
    growImg g k p = (List.range k).map fun j => ⟨p.node, p.out, shiftPos g k p + j⟩ := by
  simp [growImg, h]

theorem mem_growImg {g : Port → Bool} {k : ℕ} {p q : Port} (h : q ∈ growImg g k p) :
    q.node = p.node ∧ q.out = p.out := by
  unfold growImg at h
  split at h <;> simp at h
  · obtain ⟨j, -, rfl⟩ := h; exact ⟨rfl, rfl⟩
  · subst h; exact ⟨rfl, rfl⟩

theorem ep_ids : (D.growPorts b k).nodes.map (·.id) = D.nodes.map (·.id) := by
  rw [ep_nodes D b k, List.map_map]; rfl

/-- Distinct legs grow into disjoint legs. -/
theorem ep_disj (hids : (D.nodes.map (·.id)).Nodup) {p p' q : Port}
    (hp : p ∈ D.legs) (hp' : p' ∈ D.legs) (hq : q ∈ growImg (pg b) k p)
    (hq' : q ∈ growImg (pg b) k p') : p = p' := by
  have hnd := legs_nodup_of (D.growPorts b k) (by rw [ep_ids D b k]; exact hids)
  rw [ep_legs D b k, List.nodup_flatMap] at hnd
  by_contra hne
  have : Std.Symm (Function.onFun List.Disjoint (growImg (pg b) k)) :=
    ⟨fun _ _ h => List.disjoint_comm.1 h⟩
  exact (hnd.2.forall hp hp' hne) hq hq'

end Diagram

/-- The box structure the expansion keeps: node and box ids once each, the legs exactly the wire
ends and boundary ports, every box a node box or a port box, the boxes holding a node or a
boundary port a chain (`ChainOK`) and all of them, port boxes only on boundary ports. -/
structure WF (D : Diagram) : Prop where
  ids : (D.nodes.map (·.id)).Nodup
  bids : (D.boxes.map (·.id)).Nodup
  perm : (D.wires.map (·.1) ++ (D.wires.map (·.2) ++ D.bnd)).Perm D.legs
  kinds : ∀ y ∈ D.boxes, y.nodes = [] ∨ y.ports = []
  nsub : ∀ y ∈ D.boxes, ∀ x ∈ y.nodes, x ∈ D.nodes.map (·.id)
  nchain : ∀ x ∈ D.nodes.map (·.id), Diagram.ChainOK D (fun y => decide (x ∈ y.nodes)) none (D.nodeBoxes x)
  nall : ∀ x, ∀ y ∈ D.boxes, x ∈ y.nodes → y ∈ D.nodeBoxes x
  pchain : ∀ p ∈ D.bnd, Diagram.ChainOK D (fun y => decide (p ∈ y.ports))
    ((D.nodeBoxes p.node).getLast?.map (·.id)) (D.portBoxes p)
  pall : ∀ p ∈ D.bnd, ∀ y ∈ D.boxes, p ∈ y.ports → y ∈ D.portBoxes p
  pbnd : ∀ y ∈ D.boxes, ∀ p ∈ y.ports, p ∈ D.bnd
  wpre : ∀ w ∈ D.wires, D.nodeBoxes w.1.node <+: D.nodeBoxes w.2.node ∨
    D.nodeBoxes w.2.node <+: D.nodeBoxes w.1.node

end ArchytasZX

namespace ArchytasZX

namespace Diagram

variable (D : Diagram) (b : DBox) (k : ℕ)

theorem box_eq_of_id {D : Diagram} (hb : (D.boxes.map (·.id)).Nodup) {y z : DBox}
    (hy : y ∈ D.boxes) (hz : z ∈ D.boxes) (h : y.id = z.id) : y = z :=
  List.inj_on_of_nodup_map hb hy hz h

theorem ep_mem_boxes (y' : DBox) : y' ∈ (D.growPorts b k).boxes ↔
    ∃ y ∈ D.boxes, decide (y.id ≠ b.id) = true ∧ y' = pbox b k y ∧
      ((pbox b k y).nodes ≠ [] ∨ (pbox b k y).ports ≠ []) := by
  rw [ep_boxes D b k]
  simp only [List.mem_filter, List.mem_map, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨y, ⟨hy, hne⟩, rfl⟩, h⟩
    exact ⟨y, hy, by simpa using hne, rfl, by simpa using h⟩
  · rintro ⟨y, hy, hne, rfl, h⟩
    exact ⟨⟨y, ⟨hy, by simpa using hne⟩, rfl⟩, by simpa using h⟩

theorem ep_nchain (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (x : ℕ) (hx : x ∈ D.nodes.map (·.id)) :
    ChainOK (D.growPorts b k) (fun y => decide (x ∈ y.nodes)) none
      ((D.nodeBoxes x).map (pbox b k)) := by
  have hc := hW.nchain x hx
  have h' := hc.transport (D' := D.growPorts b k) (pr' := fun y => decide (x ∈ y.nodes))
    (pbox b k) (fun y => decide (y.id ≠ b.id)) (ep_mem_boxes D b k) (fun y _ => ⟨rfl, rfl, rfl⟩)
    (fun y hy => Or.inl (List.ne_nil_of_mem (of_decide_eq_true hy))) (fun i hi hkp => by
      have hy := hc.1 _ (List.getElem_mem hi)
      have : (D.nodeBoxes x)[i] = b := box_eq_of_id hW.bids hy.1 hb (by simpa using hkp)
      have hm := of_decide_eq_true hy.2
      rw [this, hbn] at hm
      simp at hm)
  have hf : (D.nodeBoxes x).filter (fun y => decide (y.id ≠ b.id)) = D.nodeBoxes x := by
    refine List.filter_eq_self.2 fun y hy => ?_
    have hy' := hc.1 y hy
    simp only [decide_eq_true_eq]
    intro e
    have : y = b := box_eq_of_id hW.bids hy'.1 hb e
    have hm := of_decide_eq_true hy'.2
    rw [this, hbn] at hm
    simp at hm
  rwa [hf] at h'

theorem ep_nodeBoxes (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (x : ℕ) (hx : x ∈ D.nodes.map (·.id)) :
    (D.growPorts b k).nodeBoxes x = (D.nodeBoxes x).map (pbox b k) := by
  have h' := ep_nchain D b k hW hb hbn x hx
  exact h'.walk _ h'.length_le

theorem ep_nodeChain (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (x : ℕ) (hx : x ∈ D.nodes.map (·.id)) :
    (D.growPorts b k).nodeChain x = D.nodeChain x := by
  simp only [nodeChain, ep_nodeBoxes D b k hW hb hbn x hx, List.map_map]
  rfl

theorem node_of_leg {D : Diagram} {p : Port} (hp : p ∈ D.legs) : p.node ∈ D.nodes.map (·.id) := by
  obtain ⟨n, hn, hp⟩ := List.mem_flatMap.1 hp
  simp only [DNode.legs, List.mem_append, List.mem_map] at hp
  rcases hp with ⟨i, -, rfl⟩ | ⟨i, -, rfl⟩ <;> exact List.mem_map.2 ⟨n, hn, rfl⟩

theorem bnd_legs {D : Diagram} (hW : WF D) {p : Port} (hp : p ∈ D.bnd) : p ∈ D.legs :=
  hW.perm.subset (List.mem_append_right _ (List.mem_append_right _ hp))

/-- In a chain under a leaf, the leaf can only be last. -/
theorem chain_leaf_last {D : Diagram} {pr : DBox → Bool} {start : Option ℕ} {L : List DBox}
    (hc : ChainOK D pr start L)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (i : ℕ) (hi : i < L.length)
    (hk : decide (L[i].id ≠ b.id) = false) : i + 1 = L.length := by
  by_contra hne
  have hi1 : i + 1 < L.length := by omega
  have hp := hc.2.2.1 (i + 1) hi1
  simp only [List.getElem_cons_succ, List.getElem_map] at hp
  have : L[i].id = b.id := by simpa using hk
  rw [this] at hp
  exact hleaf _ (hc.1 _ (List.getElem_mem hi1)).1 hp

theorem ep_pchain (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port) (hp : p ∈ D.bnd) (q : Port)
    (hq : q ∈ growImg (pg b) k p) :
    ChainOK (D.growPorts b k) (fun y => decide (q ∈ y.ports))
      (((D.growPorts b k).nodeBoxes q.node).getLast?.map (·.id))
      (((D.portBoxes p).filter (fun y => decide (y.id ≠ b.id))).map (pbox b k)) := by
  have hpl := bnd_legs hW hp
  obtain ⟨hqn, -⟩ := mem_growImg hq
  have hstart : ((D.growPorts b k).nodeBoxes q.node).getLast?.map (·.id) =
      (D.nodeBoxes p.node).getLast?.map (·.id) := by
    rw [hqn, ep_nodeBoxes D b k hW hb hbn _ (node_of_leg hpl), List.getLast?_map, Option.map_map]
    rfl
  rw [hstart]
  have hc := hW.pchain p hp
  have h' := hc.transport (D' := D.growPorts b k) (pr' := fun y => decide (q ∈ y.ports))
    (pbox b k) (fun y => decide (y.id ≠ b.id)) (ep_mem_boxes D b k) (fun y hy => ⟨rfl, rfl, by
      simp only [pbox, List.mem_flatMap, decide_eq_decide]
      constructor
      · rintro ⟨p'', hp'', hq''⟩
        have := ep_disj D b k hW.ids hpl (bnd_legs hW (hW.pbnd y hy p'' hp'')) hq hq''
        exact this ▸ hp''
      · exact fun h => ⟨p, h, hq⟩⟩)
    (fun y hy => Or.inr (List.ne_nil_of_mem (of_decide_eq_true hy)))
    (chain_leaf_last b hc hleaf)
  exact h'

theorem ep_portBoxes (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port) (hp : p ∈ D.bnd) (q : Port)
    (hq : q ∈ growImg (pg b) k p) :
    (D.growPorts b k).portBoxes q =
      ((D.portBoxes p).filter (fun y => decide (y.id ≠ b.id))).map (pbox b k) := by
  have h' := ep_pchain D b k hW hb hbn hleaf p hp q hq
  exact h'.walk _ h'.length_le

theorem ep_portChain (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port) (hp : p ∈ D.bnd) (q : Port)
    (hq : q ∈ growImg (pg b) k p) :
    (D.portBoxes p).map (·.mult) = ((D.growPorts b k).portBoxes q).map (·.mult) ++
      (if pg b p then [b.mult] else []) := by
  rw [ep_portBoxes D b k hW hb hbn hleaf p hp q hq, List.map_map]
  have hc := hW.pchain p hp
  set L := D.portBoxes p
  have hndL : (L.map (·.id)).Nodup := by
    have := (List.nodup_cons.1 hc.2.2.2).2
    rw [show (L.map fun b => some b.id) = (L.map (·.id)).map some by simp] at this
    exact this.of_map _
  by_cases hpb : pg b p = true
  · have hbL : b ∈ L := hW.pall p hp b hb (of_decide_eq_true hpb)
    obtain ⟨i, hi, hib⟩ := List.mem_iff_getElem.1 hbL
    have hlast := chain_leaf_last b hc hleaf i hi (by simp [hib])
    have hL : L = L.take i ++ [b] := by
      conv_lhs => rw [← List.take_append_drop i L]
      congr 1
      rw [List.drop_eq_getElem_cons hi, hib, List.drop_eq_nil_of_le (by omega)]
    have hf : (L.take i).filter (fun y => decide (y.id ≠ b.id)) = L.take i := by
      refine List.filter_eq_self.2 fun y hy => ?_
      simp only [decide_eq_true_eq]
      intro e
      have hnd' : ((L.take i ++ [b]).map (·.id)).Nodup := hL ▸ hndL
      rw [List.map_append, List.nodup_append] at hnd'
      exact hnd'.2.2 _ (List.mem_map.2 ⟨y, hy, rfl⟩) _ (by simp) e
    rw [hL, List.filter_append, hf]
    simp [hpb, pbox, Function.comp_def]
  · have hf : L.filter (fun y => decide (y.id ≠ b.id)) = L := by
      refine List.filter_eq_self.2 fun y hy => ?_
      simp only [decide_eq_true_eq]
      intro e
      have : y = b := box_eq_of_id hW.bids (hc.1 y hy).1 hb e
      subst this
      exact hpb (by simpa [pg_def] using (hc.1 _ hy).2)
    rw [hf]
    simp [hpb, pbox, Function.comp_def]

/-- A leg's position after the port box grows. -/
def pshift (b : DBox) (k : ℕ) (p : Port) : Port := ⟨p.node, p.out, shiftPos (pg b) k p⟩

theorem wire_not_bnd {D : Diagram} (hW : WF D) {w : Port × Port} (hw : w ∈ D.wires) :
    w.1 ∉ D.bnd ∧ w.2 ∉ D.bnd ∧ w.1 ∈ D.legs ∧ w.2 ∈ D.legs := by
  have hnd := hW.perm.nodup_iff.2 (legs_nodup_of D hW.ids)
  rw [List.nodup_append] at hnd
  obtain ⟨-, n23, d1⟩ := hnd
  rw [List.nodup_append] at n23
  have h1 : w.1 ∈ D.wires.map (·.1) := List.mem_map.2 ⟨w, hw, rfl⟩
  have h2 : w.2 ∈ D.wires.map (·.2) := List.mem_map.2 ⟨w, hw, rfl⟩
  refine ⟨fun h => d1 _ h1 _ (List.mem_append_right _ h) rfl, fun h => n23.2.2 _ h2 _ h rfl,
    hW.perm.subset (List.mem_append_left _ h1),
    hW.perm.subset (List.mem_append_right _ (List.mem_append_left _ h2))⟩

theorem not_pg_of_wire {D : Diagram} (hW : WF D) (hb : b ∈ D.boxes) {w : Port × Port}
    (hw : w ∈ D.wires) : pg b w.1 = false ∧ pg b w.2 = false := by
  obtain ⟨h1, h2, -, -⟩ := wire_not_bnd hW hw
  constructor <;> simp only [pg_def, decide_eq_false_iff_not] <;> intro h
  · exact h1 (hW.pbnd b hb _ h)
  · exact h2 (hW.pbnd b hb _ h)

theorem ep_wires_map (hW : WF D) (hb : b ∈ D.boxes) :
    (D.growPorts b k).wires = D.wires.map fun w => (pshift b k w.1, pshift b k w.2) := by
  rw [ep_wires D b k]
  rw [List.map_eq_flatMap]
  refine List.flatMap_congr fun w hw => ?_
  obtain ⟨h1, h2⟩ := not_pg_of_wire b hW hb hw
  rw [growImg_of_not _ _ _ h1, growImg_of_not _ _ _ h2]
  rfl

theorem find?_congr' {α : Type*} (l : List α) (P Q : α → Bool) (h : ∀ x ∈ l, P x = Q x) :
    l.find? P = l.find? Q := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.find?_cons, List.find?_cons, h a List.mem_cons_self,
      ih fun x hx => h x (List.mem_cons_of_mem _ hx)]

theorem ep_partner (hW : WF D) (hb : b ∈ D.boxes) (p : Port) (hp : p ∈ D.legs)
    (q : Port) (hq : q ∈ growImg (pg b) k p) :
    (D.growPorts b k).partner q = (D.partner p).map (pshift b k) := by
  have key : ∀ w ∈ D.wires, ∀ e : Port, (e = w.1 ∨ e = w.2) → (pshift b k e = q ↔ e = p) := by
    intro w hw e he
    obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
    obtain ⟨n1, n2⟩ := not_pg_of_wire b hW hb hw
    have hel : e ∈ D.legs := by rcases he with rfl | rfl <;> assumption
    have hen : pg b e = false := by rcases he with rfl | rfl <;> assumption
    constructor
    · intro h
      refine ep_disj D b k hW.ids hel hp ?_ hq
      rw [growImg_of_not _ _ _ hen, ← h]; simp [pshift]
    · rintro rfl
      rw [growImg_of_not _ _ _ hen] at hq
      simp only [List.mem_singleton] at hq
      rw [hq]; rfl
  unfold partner
  rw [ep_wires_map D b k hW hb, List.find?_map, Option.map_map]
  rw [find?_congr' _ _ (fun w => decide (w.1 = p ∨ w.2 = p)) fun w hw => by
    simp only [Function.comp, decide_eq_decide]
    rw [key w hw w.1 (Or.inl rfl), key w hw w.2 (Or.inr rfl)]]
  cases hf : D.wires.find? (fun w => decide (w.1 = p ∨ w.2 = p)) with
  | none => rfl
  | some w =>
    have hw := List.mem_of_find?_eq_some hf
    simp only [Option.map_some, Function.comp]
    by_cases h1 : w.1 = p
    · have : pshift b k w.1 = q := (key w hw w.1 (Or.inl rfl)).2 h1
      rw [if_pos this, if_pos h1]
    · have : pshift b k w.1 ≠ q := fun e => h1 ((key w hw w.1 (Or.inl rfl)).1 e)
      rw [if_neg this, if_neg h1]

theorem partner_some {D : Diagram} {p r : Port} (h : D.partner p = some r) :
    ∃ w ∈ D.wires, (w.1 = p ∧ r = w.2) ∨ (w.2 = p ∧ r = w.1) := by
  unfold partner at h
  cases hf : D.wires.find? (fun w => decide (w.1 = p ∨ w.2 = p)) with
  | none => rw [hf] at h; simp at h
  | some w =>
    rw [hf] at h
    have hw := List.mem_of_find?_eq_some hf
    have hp := List.find?_some hf
    simp only [decide_eq_true_eq] at hp
    simp only [Option.map_some, Option.some.injEq] at h
    refine ⟨w, hw, ?_⟩
    by_cases h1 : w.1 = p
    · rw [if_pos h1] at h; exact Or.inl ⟨h1, h.symm⟩
    · rw [if_neg h1] at h; exact Or.inr ⟨hp.resolve_left h1, h.symm⟩

theorem partner_none {D : Diagram} {p : Port} (h : D.partner p = none) :
    ∀ w ∈ D.wires, w.1 ≠ p ∧ w.2 ≠ p := by
  unfold partner at h
  rw [Option.map_eq_none_iff, List.find?_eq_none] at h
  intro w hw
  have := h w hw
  simp only [decide_eq_true_eq, not_or] at this
  exact this

theorem bnd_of_partner_none {D : Diagram} (hW : WF D) {p : Port} (hp : p ∈ D.legs)
    (h : D.partner p = none) : p ∈ D.bnd := by
  have hn := partner_none h
  rcases List.mem_append.1 (hW.perm.mem_iff.2 hp) with h1 | h1
  · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 h1; exact absurd rfl (hn w hw).1
  · rcases List.mem_append.1 h1 with h2 | h2
    · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 h2; exact absurd rfl (hn w hw).2
    · exact h2

theorem ep_tail (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port) (hp : p ∈ D.legs) (q : Port)
    (hq : q ∈ growImg (pg b) k p) :
    D.tail p = (D.growPorts b k).tail q ++ (if pg b p then [b.mult] else []) := by
  obtain ⟨hqn, -⟩ := mem_growImg hq
  unfold tail
  rw [ep_partner D b k hW hb p hp q hq]
  cases hpart : D.partner p with
  | none =>
    simp only [Option.map_none]
    exact ep_portChain D b k hW hb hbn hleaf p (bnd_of_partner_none hW hp hpart) q hq
  | some r =>
    obtain ⟨w, hw, hwr⟩ := partner_some hpart
    obtain ⟨n1, n2⟩ := not_pg_of_wire b hW hb hw
    obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
    have hpg : pg b p = false := by rcases hwr with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> assumption
    have hrl : r ∈ D.legs := by rcases hwr with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> assumption
    simp only [Option.map_some, hpg, Bool.false_eq_true, ↓reduceIte, List.append_nil]
    rw [hqn, show (pshift b k r).node = r.node from rfl,
      ep_nodeChain D b k hW hb hbn _ (node_of_leg hp),
      ep_nodeChain D b k hW hb hbn _ (node_of_leg hrl)]

theorem ep_legChain (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port) (hp : p ∈ D.legs) (q : Port)
    (hq : q ∈ growImg (pg b) k p) :
    D.legChain p = (D.growPorts b k).legChain q ++ (if pg b p then [b.mult] else []) := by
  obtain ⟨hqn, -⟩ := mem_growImg hq
  unfold legChain
  rw [ep_tail D b k hW hb hbn hleaf p hp q hq, hqn,
    ep_nodeChain D b k hW hb hbn _ (node_of_leg hp), List.append_assoc]

theorem flatMap_blocks_length {α : Type*} (F : ℕ → List α) (w : ℕ → ℕ)
    (hw : ∀ i, (F i).length = w i) (n : ℕ) : ((List.range n).flatMap F).length = wsum w n := by
  induction n with
  | zero => simp [wsum]
  | succ n ih => rw [List.range_succ, List.flatMap_append, List.length_append, ih, wsum_succ]; simp [hw]

theorem flatMap_blocks_get {α : Type*} (F : ℕ → List α) (w : ℕ → ℕ)
    (hw : ∀ i, (F i).length = w i) (n i j : ℕ) (hi : i < n) (hj : j < w i) :
    ((List.range n).flatMap F)[wsum w i + j]? = (F i)[j]? := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append]
    by_cases hin : i < n
    · have hlt : wsum w i + j < ((List.range n).flatMap F).length := by
        rw [flatMap_blocks_length F w hw]
        have : wsum w (i + 1) ≤ wsum w n :=
          monotone_nat_of_le_succ (fun m => by rw [wsum_succ]; omega) (show i + 1 ≤ n by omega)
        rw [wsum_succ] at this
        omega
      rw [List.getElem?_append_left hlt, ih hin]
    · have : i = n := by omega
      subst this
      rw [List.getElem?_append_right (by rw [flatMap_blocks_length F w hw]; omega),
        flatMap_blocks_length F w hw]
      simp

theorem growSide_eq (g : Port → Bool) (k x : ℕ) (out : Bool) (ds : List ℕ) :
    growSide g k x out ds = (List.range ds.length).flatMap fun i =>
      List.replicate (if g ⟨x, out, i⟩ then k else 1) (ds.getD i 0) := by
  unfold growSide
  induction ds using List.reverseRecOn with
  | nil => simp
  | append_singleton ds d ih =>
    rw [List.zipIdx_append, List.flatMap_append, ih, List.length_append, List.length_singleton,
      List.range_succ, List.flatMap_append]
    congr 1
    · refine List.flatMap_congr fun i hi => ?_
      rw [List.getD_append _ _ _ _ (List.mem_range.1 hi)]
    · by_cases h : g ⟨x, out, ds.length⟩ <;> simp [h]

theorem growSide_getD (g : Port → Bool) (k x : ℕ) (out : Bool) (ds : List ℕ) (i j : ℕ)
    (hi : i < ds.length) (hj : j < if g ⟨x, out, i⟩ then k else 1) :
    (growSide g k x out ds).getD (shiftPos g k ⟨x, out, i⟩ + j) 0 = ds.getD i 0 := by
  rw [growSide_eq, shiftPos_eq, List.getD_eq_getElem?_getD,
    flatMap_blocks_get _ (fun i' => if g ⟨x, out, i'⟩ then k else 1) (fun i' => by simp) _ _ _ hi hj]
  simp [List.getElem?_replicate, hj]

theorem mem_growImg' {g : Port → Bool} {k : ℕ} {p q : Port} (h : q ∈ growImg g k p) :
    ∃ j < (if g p then k else 1), q = ⟨p.node, p.out, shiftPos g k p + j⟩ := by
  unfold growImg at h
  by_cases hg : g p = true
  · rw [if_pos hg] at h
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 h
    exact ⟨j, by simpa [hg] using hj, rfl⟩
  · rw [if_neg hg] at h
    rw [List.mem_singleton.1 h]
    exact ⟨0, by simp [hg], rfl⟩

theorem leg_of_mem {D : Diagram} {p : Port} (hp : p ∈ D.legs) :
    ∃ n ∈ D.nodes, p.node = n.id ∧ p.pos < (if p.out then n.outs.length else n.ins.length) := by
  obtain ⟨n, hn, hp⟩ := List.mem_flatMap.1 hp
  simp only [DNode.legs, List.mem_append, List.mem_map, List.mem_range] at hp
  rcases hp with ⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩ <;> exact ⟨n, hn, rfl, by simpa⟩

theorem find_node {D : Diagram} (hids : (D.nodes.map (·.id)).Nodup) {n : DNode} (hn : n ∈ D.nodes) :
    D.nodes.find? (fun m => decide (m.id = n.id)) = some n :=
  find?_unique _ _ n hn (by simp) fun y hy h =>
    List.inj_on_of_nodup_map hids hy hn (by simpa using h)

theorem ep_portDim (hW : WF D) (p : Port) (hp : p ∈ D.legs) (q : Port)
    (hq : q ∈ growImg (pg b) k p) : (D.growPorts b k).portDim q = D.portDim p := by
  obtain ⟨n, hn, hpn, hpos⟩ := leg_of_mem hp
  obtain ⟨j, hj, rfl⟩ := mem_growImg' hq
  unfold portDim
  rw [ep_nodes D b k, List.find?_map]
  have e1 : (D.nodes.find? ((fun m => decide (m.id = p.node)) ∘ pgrow b k)) =
      D.nodes.find? (fun m => decide (m.id = p.node)) := rfl
  simp only at e1 ⊢
  rw [e1, hpn, find_node hW.ids hn]
  simp only [Option.map_some]
  obtain ⟨x, o, i⟩ := p
  simp only at hpn hpos hj ⊢
  subst hpn
  cases o
  · simp only [Bool.false_eq_true, ↓reduceIte] at hpos ⊢
    exact growSide_getD _ _ _ _ _ _ _ hpos hj
  · simp only [↓reduceIte] at hpos ⊢
    exact growSide_getD _ _ _ _ _ _ _ hpos hj

theorem dlabs_nodup (D : Diagram) (ks : ℕ → ℕ) (h : D.legs.Nodup) : (D.dlabs ks).Nodup := by
  unfold dlabs
  refine List.nodup_flatMap.2 ⟨fun p _ => (cidx_nodup ks _).map fun a b e => (dlab_inj e).2, ?_⟩
  refine h.pairwise_of_forall_ne fun p _ q _ hpq => ?_
  simp only [Function.onFun, List.disjoint_left, List.mem_map]
  rintro _ ⟨t, -, rfl⟩ ⟨u, -, e⟩
  exact hpq (dlab_inj e).1.symm

theorem partner_of_wire_nd {D : Diagram}
    (hnd : (D.wires.map (·.1) ++ (D.wires.map (·.2) ++ D.bnd)).Nodup) {w : Port × Port}
    (hw : w ∈ D.wires) : D.partner w.1 = some w.2 ∧ D.partner w.2 = some w.1 := by
  rw [List.nodup_append] at hnd
  obtain ⟨n1, n23, d1⟩ := hnd
  rw [List.nodup_append] at n23
  obtain ⟨n2, -, -⟩ := n23
  have m1 : ∀ v ∈ D.wires, v.1 ∈ D.wires.map (·.1) := fun v hv => List.mem_map.2 ⟨v, hv, rfl⟩
  have m2 : ∀ v ∈ D.wires, v.2 ∈ D.wires.map (·.2) := fun v hv => List.mem_map.2 ⟨v, hv, rfl⟩
  have x12 : ∀ v ∈ D.wires, ∀ v' ∈ D.wires, v.1 ≠ v'.2 := fun v hv v' hv' =>
    d1 _ (m1 v hv) _ (List.mem_append_left _ (m2 v' hv'))
  unfold partner
  constructor
  · rw [find?_unique _ _ w hw (by simp) fun v hv h => by
      simp only [decide_eq_true_eq] at h
      rcases h with h | h
      · exact List.inj_on_of_nodup_map n1 hv hw h
      · exact absurd h.symm (x12 w hw v hv)]
    simp
  · rw [find?_unique _ _ w hw (by simp) fun v hv h => by
      simp only [decide_eq_true_eq] at h
      rcases h with h | h
      · exact absurd h (x12 v hv w hw)
      · exact List.inj_on_of_nodup_map n2 hv hw h]
    simp [x12 w hw w hw]

theorem partner_of_wire {D : Diagram} (hW : WF D) {w : Port × Port} (hw : w ∈ D.wires) :
    D.partner w.1 = some w.2 ∧ D.partner w.2 = some w.1 :=
  partner_of_wire_nd (hW.perm.nodup_iff.2 (legs_nodup_of D hW.ids)) hw

theorem legChain_wire {D : Diagram} (hW : WF D) {w : Port × Port} (hw : w ∈ D.wires) :
    D.legChain w.1 = D.legChain w.2 := by
  obtain ⟨h1, h2⟩ := partner_of_wire hW hw
  unfold legChain tail
  rw [h1, h2]
  simp only
  have hpre : D.nodeChain w.1.node <+: D.nodeChain w.2.node ∨
      D.nodeChain w.2.node <+: D.nodeChain w.1.node := by
    unfold nodeChain
    rcases hW.wpre w hw with h | h
    · exact Or.inl (h.map _)
    · exact Or.inr (h.map _)
  rcases hpre with h | h
  · obtain ⟨c, hc⟩ := h
    rw [← hc]
    by_cases hc0 : c = []
    · subst hc0; simp
    · have : (D.nodeChain w.1.node).length < (D.nodeChain w.1.node ++ c).length := by
        simp; exact List.length_pos_iff.2 hc0
      simp [this]
  · obtain ⟨c, hc⟩ := h
    rw [← hc]
    by_cases hc0 : c = []
    · subst hc0; simp
    · have : (D.nodeChain w.2.node).length < (D.nodeChain w.2.node ++ c).length := by
        simp; exact List.length_pos_iff.2 hc0
      simp [this]

theorem getLastD_snoc (u : List ℕ) (j d : ℕ) : (u ++ [j]).getLastD d = j := by
  simp [List.getLastD_eq_getLast?]

theorem dropLast_snoc (u : List ℕ) (j : ℕ) : (u ++ [j]).dropLast = u := List.dropLast_concat

/-- The port box's map of digits: digit `t ++ [j]` of a leg in `b` is digit `t` of its `j`-th
image, any other digit that of its single image. -/
def φP (b : DBox) (k : ℕ) (p : Port) (t : List ℕ) : Port × List ℕ :=
  if pg b p then ((growImg (pg b) k p).getD (t.getLastD 0) Port.dflt, t.dropLast)
  else ((growImg (pg b) k p).headD Port.dflt, t)

theorem growImg_getD_pos (g : Port → Bool) (k : ℕ) (p : Port) (hg : g p = true) (j : ℕ)
    (hj : j < k) : (growImg g k p).getD j Port.dflt = ⟨p.node, p.out, shiftPos g k p + j⟩ := by
  rw [growImg_of_pos _ _ _ hg, List.getD_eq_getElem _ _ (by simpa using hj)]
  simp

theorem tail_pg (hW : WF D) (hb : b ∈ D.boxes)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port) (hp : p ∈ D.legs)
    (hpg : pg b p = true) : ∃ tl, D.tail p = tl ++ [b.mult] := by
  have hpb : p ∈ b.ports := of_decide_eq_true hpg
  have hpbnd := hW.pbnd b hb p hpb
  have hnone : D.partner p = none := by
    cases h : D.partner p with
    | none => rfl
    | some r =>
      obtain ⟨w, hw, hwr⟩ := partner_some h
      obtain ⟨h1, h2, -, -⟩ := wire_not_bnd hW hw
      rcases hwr with ⟨rfl, -⟩ | ⟨rfl, -⟩
      · exact absurd hpbnd h1
      · exact absurd hpbnd h2
  unfold tail
  rw [hnone]
  simp only
  have hc := hW.pchain p hpbnd
  have hbL : b ∈ D.portBoxes p := hW.pall p hpbnd b hb hpb
  obtain ⟨i, hi, hib⟩ := List.mem_iff_getElem.1 hbL
  have hlast := chain_leaf_last b hc hleaf i hi (by simp [hib])
  refine ⟨((D.portBoxes p).take i).map (·.mult), ?_⟩
  conv_lhs => rw [← List.take_append_drop i (D.portBoxes p)]
  rw [List.drop_eq_getElem_cons hi, hib, List.drop_eq_nil_of_le (by omega)]
  simp

theorem legChain_pg (hW : WF D) (hb : b ∈ D.boxes)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port) (hp : p ∈ D.legs)
    (hpg : pg b p = true) : ∃ c, D.legChain p = c ++ [b.mult] := by
  obtain ⟨tl, htl⟩ := tail_pg D b hW hb hleaf p hp hpg
  exact ⟨D.nodeChain p.node ++ tl, by simp [legChain, htl]⟩

theorem legChain_img (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port) (hp : p ∈ D.legs) (q : Port)
    (hq : q ∈ growImg (pg b) k p) (hpg : pg b p = true) :
    D.legChain p = (D.growPorts b k).legChain q ++ [b.mult] := by
  rw [ep_legChain D b k hW hb hbn hleaf p hp q hq, if_pos hpg]

theorem φP_spec (hkm : ks b.mult = k) (hW : WF D) (hb : b ∈ D.boxes)
    (hbn : b.nodes = []) (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port)
    (hp : p ∈ D.legs) (t : List ℕ) (ht : t ∈ cidx ks (D.legChain p)) :
    (φP b k p t).1 ∈ growImg (pg b) k p ∧
      (φP b k p t).2 ∈ cidx ks ((D.growPorts b k).legChain (φP b k p t).1) := by
  unfold φP
  by_cases hpg : pg b p = true
  · rw [if_pos hpg]
    obtain ⟨c, hc⟩ := legChain_pg D b hW hb hleaf p hp hpg
    rw [hc, mem_cidx_single] at ht
    obtain ⟨u, j, rfl, hu, hj⟩ := ht
    rw [hkm] at hj
    simp only [getLastD_snoc, dropLast_snoc]
    rw [growImg_getD_pos _ _ _ hpg j hj]
    have hqj : (⟨p.node, p.out, shiftPos (pg b) k p + j⟩ : Port) ∈ growImg (pg b) k p := by
      rw [growImg_of_pos _ _ _ hpg]; exact List.mem_map.2 ⟨j, by simpa using hj, rfl⟩
    refine ⟨hqj, ?_⟩
    have e := hc.symm.trans (legChain_img D b k hW hb hbn hleaf p hp _ hqj hpg)
    rw [List.append_cancel_right e] at hu
    exact hu
  · rw [if_neg hpg]
    have hpg' : pg b p = false := by simpa using hpg
    rw [growImg_of_not _ _ _ hpg']
    simp only [List.headD_cons]
    have hq : (⟨p.node, p.out, shiftPos (pg b) k p⟩ : Port) ∈ growImg (pg b) k p := by
      rw [growImg_of_not _ _ _ hpg']; simp
    refine ⟨List.mem_singleton_self _, ?_⟩
    rw [ep_legChain D b k hW hb hbn hleaf p hp _ hq, if_neg hpg, List.append_nil] at ht
    exact ht

theorem φP_onto (hkm : ks b.mult = k) (hW : WF D) (hb : b ∈ D.boxes)
    (hbn : b.nodes = []) (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port)
    (hp : p ∈ D.legs) (q : Port) (hq : q ∈ growImg (pg b) k p) (u : List ℕ)
    (hu : u ∈ cidx ks ((D.growPorts b k).legChain q)) :
    ∃ t ∈ cidx ks (D.legChain p), φP b k p t = (q, u) := by
  unfold φP
  obtain ⟨j, hj, rfl⟩ := mem_growImg' hq
  by_cases hpg : pg b p = true
  · rw [if_pos hpg] at hj
    refine ⟨u ++ [j], ?_, ?_⟩
    · rw [legChain_img D b k hW hb hbn hleaf p hp _ hq hpg, mem_cidx_single]
      exact ⟨u, j, rfl, hu, by rw [hkm]; exact hj⟩
    · rw [if_pos hpg]
      simp only [getLastD_snoc, dropLast_snoc]
      rw [growImg_getD_pos _ _ _ hpg j hj]
  · have hpg' : pg b p = false := by simpa using hpg
    rw [if_neg hpg] at hj
    have hj0 : j = 0 := by omega
    subst hj0
    refine ⟨u, ?_, ?_⟩
    · rw [ep_legChain D b k hW hb hbn hleaf p hp _ hq, if_neg hpg, List.append_nil]; exact hu
    · rw [if_neg hpg, growImg_of_not _ _ _ hpg']
      simp

theorem φP_inj (hkm : ks b.mult = k) (hW : WF D) (hb : b ∈ D.boxes)
    (hbn : b.nodes = []) (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (p : Port)
    (hp : p ∈ D.legs) (t t' : List ℕ) (ht : t ∈ cidx ks (D.legChain p))
    (ht' : t' ∈ cidx ks (D.legChain p)) (h : φP b k p t = φP b k p t') : t = t' := by
  unfold φP at h
  by_cases hpg : pg b p = true
  · rw [if_pos hpg, if_pos hpg] at h
    obtain ⟨c, hc⟩ := legChain_pg D b hW hb hleaf p hp hpg
    rw [hc, mem_cidx_single] at ht ht'
    obtain ⟨u, j, rfl, -, hj⟩ := ht
    obtain ⟨u', j', rfl, -, hj'⟩ := ht'
    rw [hkm] at hj hj'
    simp only [getLastD_snoc, dropLast_snoc, Prod.mk.injEq] at h
    rw [growImg_getD_pos _ _ _ hpg j hj, growImg_getD_pos _ _ _ hpg j' hj'] at h
    simp only [Port.mk.injEq, true_and, Nat.add_left_cancel_iff] at h
    rw [h.1, h.2]
  · rw [if_neg hpg, if_neg hpg] at h
    exact (Prod.mk.inj h).2

/-- A node's side: its outputs or its inputs. -/
def side (n : DNode) (out : Bool) : List ℕ := if out then n.outs else n.ins

theorem leg_mem {D : Diagram} {n : DNode} (hn : n ∈ D.nodes) (out : Bool) (i : ℕ)
    (hi : i < (side n out).length) : (⟨n.id, out, i⟩ : Port) ∈ D.legs := by
  refine List.mem_flatMap.2 ⟨n, hn, ?_⟩
  cases out
  · simp only [side, Bool.false_eq_true, ↓reduceIte] at hi
    simp [DNode.legs, hi]
  · simp only [side, ↓reduceIte] at hi
    simp [DNode.legs, hi]

theorem ep_dnode_side (hkm : ks b.mult = k) (hW : WF D) (hb : b ∈ D.boxes)
    (hbn : b.nodes = []) (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (n : DNode)
    (hn : n ∈ D.nodes) (out : Bool) (t : List ℕ) :
    (List.range (growSide (pg b) k n.id out (side n out)).length).flatMap (fun i' =>
      (rcidx ks ((D.growPorts b k).tail ⟨n.id, out, i'⟩)).map fun s => dlab ⟨n.id, out, i'⟩ (t ++ s)) =
    ((List.range (side n out).length).flatMap fun i =>
      (rcidx ks (D.tail ⟨n.id, out, i⟩)).map fun s => dlab ⟨n.id, out, i⟩ (t ++ s)).map
        (relab (φP b k)) := by
  rw [growSide_length, ← range_blocks, List.flatMap_assoc, List.map_flatMap]
  refine List.flatMap_congr fun i hi => ?_
  have hi' := List.mem_range.1 hi
  set p : Port := ⟨n.id, out, i⟩
  have hp : p ∈ D.legs := leg_mem hn out i hi'
  rw [List.flatMap_map, List.map_map]
  by_cases hpg : pg b p = true
  · simp only [hpg, ↓reduceIte]
    obtain ⟨tl, htl⟩ := tail_pg D b hW hb hleaf p hp hpg
    rw [htl, rcidx_append_single, hkm, List.map_flatMap]
    refine List.flatMap_congr fun j hj => ?_
    have hj' := List.mem_range.1 hj
    have hqj : (⟨p.node, p.out, shiftPos (pg b) k p + j⟩ : Port) ∈ growImg (pg b) k p := by
      rw [growImg_of_pos _ _ _ hpg]; exact List.mem_map.2 ⟨j, by simpa using hj', rfl⟩
    have htl' := ep_tail D b k hW hb hbn hleaf p hp _ hqj
    rw [if_pos hpg, htl] at htl'
    have e := List.append_cancel_right htl'
    rw [← shiftPos_eq]
    change List.map (fun s => dlab ⟨p.node, p.out, shiftPos (pg b) k p + j⟩ (t ++ s))
      (rcidx ks ((D.growPorts b k).tail ⟨p.node, p.out, shiftPos (pg b) k p + j⟩)) = _
    rw [← e, List.map_map]
    refine List.map_congr_left fun s _ => ?_
    simp only [Function.comp, relab_dlab, φP, if_pos hpg, ← List.append_assoc, getLastD_snoc,
      dropLast_snoc, growImg_getD_pos _ _ _ hpg j hj']
  · have hpg' : pg b p = false := by simpa using hpg
    simp only [hpg', Bool.false_eq_true, ↓reduceIte]
    have hq : (⟨p.node, p.out, shiftPos (pg b) k p⟩ : Port) ∈ growImg (pg b) k p := by
      rw [growImg_of_not _ _ _ hpg']; simp
    have htl := ep_tail D b k hW hb hbn hleaf p hp _ hq
    rw [if_neg hpg, List.append_nil] at htl
    simp only [List.range_one, List.flatMap_singleton, add_zero]
    rw [← shiftPos_eq]
    change List.map (fun s => dlab ⟨p.node, p.out, shiftPos (pg b) k p⟩ (t ++ s))
      (rcidx ks ((D.growPorts b k).tail ⟨p.node, p.out, shiftPos (pg b) k p⟩)) = _
    rw [← htl]
    refine List.map_congr_left fun s _ => ?_
    simp only [Function.comp, relab_dlab, φP, if_neg hpg, growImg_of_not _ _ _ hpg',
      List.headD_cons]

theorem dnode_labels_mem {D : Diagram} (ks : ℕ → ℕ) {n : DNode} (hn : n ∈ D.nodes) {t : List ℕ}
    (ht : t ∈ cidx ks (D.nodeChain n.id)) {x : ℕ} (hx : x ∈ (D.dnode ks n t).labels) :
    x ∈ D.dlabs ks := by
  simp only [Node.labels, dnode, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range] at hx
  rw [mem_dlabs]
  rcases hx with ⟨i, hi, s, hs, rfl⟩ | ⟨i, hi, s, hs, rfl⟩
  · refine ⟨_, leg_mem hn true i (by simpa [side] using hi), t ++ s, ?_, rfl⟩
    exact mem_cidx_append ks _ _ _ _ ht ((mem_rcidx ks _ s).1 hs)
  · refine ⟨_, leg_mem hn false i (by simpa [side] using hi), t ++ s, ?_, rfl⟩
    exact mem_cidx_append ks _ _ _ _ ht ((mem_rcidx ks _ s).1 hs)

theorem ep_prodD (hkm : ks b.mult = k) (hW : WF D) (hb : b ∈ D.boxes)
    (hbn : b.nodes = []) (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (E : Env)
    (τ τ' : ℕ → ℕ) (hl : ∀ x ∈ D.dlabs ks, τ' x = τ (relab (φP b k) x)) :
    (D.growPorts b k).prodD E ks τ = D.prodD E ks τ' := by
  unfold prodD
  rw [ep_nodes D b k, List.map_map]
  congr 1
  refine List.map_congr_left fun n hn => ?_
  simp only [Function.comp]
  rw [show (pgrow b k n).id = n.id from rfl,
    ep_nodeChain D b k hW hb hbn n.id (List.mem_map_of_mem hn)]
  congr 1
  refine List.map_congr_left fun t ht => ?_
  have hnode : (D.growPorts b k).dnode ks (pgrow b k n) t =
      (D.dnode ks n t).rename (relab (φP b k)) := by
    have h1 := ep_dnode_side D b k hkm hW hb hbn hleaf n hn true t
    have h2 := ep_dnode_side D b k hkm hW hb hbn hleaf n hn false t
    simp only [side, ↓reduceIte, Bool.false_eq_true] at h1 h2
    simp only [dnode, Node.rename, pgrow]
    rw [h1, h2]
  rw [hnode, val_rename]
  exact val_reads E _ _ _ fun x hx => (hl x (dnode_labels_mem ks hn ht hx)).symm

theorem φP_not (p : Port) (t : List ℕ) (h : pg b p = false) : φP b k p t = (pshift b k p, t) := by
  simp [φP, h, growImg_of_not, pshift]

theorem pshift_mem (p : Port) (h : pg b p = false) : pshift b k p ∈ growImg (pg b) k p := by
  rw [growImg_of_not _ _ _ h]; simp [pshift]

/-- Instantiating a port box `b` at `ks b.mult ≥ 1` keeps the digit sum: digit `t ++ [j]` of a leg
in `b` is digit `t` of its `j`-th image. -/
theorem evalD_growPorts (hW : WF D) (hb : b ∈ D.boxes)
    (hbn : b.nodes = []) (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (E : Env)
    (σ' : ℕ → List ℕ → ℕ) :
    (D.growPorts b (ks b.mult)).evalD E ks σ' =
      D.evalD E ks (bndMap D (D.growPorts b (ks b.mult)) (φP b (ks b.mult)) σ') := by
  set k := ks b.mult with hkdef
  have hkm : ks b.mult = k := rfl
  have hlegs := legs_nodup_of D hW.ids
  have hlegs' := legs_nodup_of (D.growPorts b k) (by rw [ep_ids D b k]; exact hW.ids)
  have himg : ∀ p ∈ D.legs, (growImg (pg b) k p).Nodup := by
    have := hlegs'; rw [ep_legs D b k, List.nodup_flatMap] at this; exact this.1
  have hbnd : D.bnd.Nodup := by
    have := hW.perm.nodup_iff.2 hlegs
    exact ((List.nodup_append.1 this).2.1 |> List.nodup_append.1).2.1
  refine evalD_relabelφ D (D.growPorts b k) E ks ks σ' (φP b k) (dlabs_nodup D ks hlegs)
    (dlabs_nodup _ ks hlegs') ?_ (fun p hp t ht => ?_) (fun q hq u hu => ?_)
    (fun p hp t ht p' hp' t' ht' he => ?_) (fun p hp t ht => ?_) (fun w hw => ?_)
    (fun w hw t ht => ?_) (fun w' hw' u hu => ?_) (fun p hp => bnd_legs hW hp)
    (fun p hp t ht => ?_) (fun q hq u hu => ?_)
    (fun τ τ' hl => ep_prodD D b k hkm hW hb hbn hleaf E τ τ' hl)
  · rw [ep_bnd D b k, List.nodup_flatMap]
    refine ⟨fun p hp => himg p (bnd_legs hW hp), hbnd.pairwise_of_forall_ne fun p hp p' hp' hne => ?_⟩
    simp only [Function.onFun, List.disjoint_left]
    exact fun q hq hq' => hne (ep_disj D b k hW.ids (bnd_legs hW hp) (bnd_legs hW hp') hq hq')
  · obtain ⟨h1, h2⟩ := φP_spec D b k hkm hW hb hbn hleaf p hp t ht
    exact ⟨by rw [ep_legs D b k]; exact List.mem_flatMap.2 ⟨p, hp, h1⟩, h2⟩
  · rw [ep_legs D b k] at hq
    obtain ⟨p, hp, hqp⟩ := List.mem_flatMap.1 hq
    obtain ⟨t, ht, he⟩ := φP_onto D b k hkm hW hb hbn hleaf p hp q hqp u hu
    exact ⟨p, hp, t, ht, he⟩
  · have h1 := (φP_spec D b k hkm hW hb hbn hleaf p hp t ht).1
    have h2 := (φP_spec D b k hkm hW hb hbn hleaf p' hp' t' ht').1
    rw [he] at h1
    have hpp := ep_disj D b k hW.ids hp hp' h1 h2
    subst hpp
    exact ⟨rfl, φP_inj D b k hkm hW hb hbn hleaf p hp t t' ht ht' he⟩
  · rw [ep_portDim D b k hW p hp _ (φP_spec D b k hkm hW hb hbn hleaf p hp t ht).1]
  · obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
    exact ⟨l1, l2, legChain_wire hW hw⟩
  · obtain ⟨n1, n2⟩ := not_pg_of_wire b hW hb hw
    obtain ⟨-, -, l1, -⟩ := wire_not_bnd hW hw
    refine ⟨(pshift b k w.1, pshift b k w.2), ?_, t, ?_, φP_not b k _ t n1, φP_not b k _ t n2⟩
    · rw [ep_wires_map D b k hW hb]; exact List.mem_map.2 ⟨w, hw, rfl⟩
    · have := (φP_spec D b k hkm hW hb hbn hleaf w.1 l1 t ht).2
      rwa [φP_not b k _ t n1] at this
  · rw [ep_wires_map D b k hW hb] at hw'
    obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hw'
    obtain ⟨n1, n2⟩ := not_pg_of_wire b hW hb hw
    obtain ⟨-, -, l1, -⟩ := wire_not_bnd hW hw
    refine ⟨w, hw, u, ?_, φP_not b k _ u n1, φP_not b k _ u n2⟩
    rw [ep_legChain D b k hW hb hbn hleaf w.1 l1 _ (pshift_mem b k _ n1), n1]
    simpa using hu
  · rw [ep_bnd D b k]
    exact List.mem_flatMap.2 ⟨p, hp, (φP_spec D b k hkm hW hb hbn hleaf p (bnd_legs hW hp) t ht).1⟩
  · rw [ep_bnd D b k] at hq
    obtain ⟨p, hp, hqp⟩ := List.mem_flatMap.1 hq
    rw [← ep_bnd D b k] at hq
    obtain ⟨t, ht, he⟩ := φP_onto D b k hkm hW hb hbn hleaf p (bnd_legs hW hp) q hqp u hu
    exact ⟨p, hp, t, ht, he⟩

theorem ep_wire_ends (hW : WF D) (hb : b ∈ D.boxes) :
    ((D.growPorts b k).wires.map (·.1) ++ ((D.growPorts b k).wires.map (·.2) ++
      (D.growPorts b k).bnd)) =
      (D.wires.map (·.1) ++ (D.wires.map (·.2) ++ D.bnd)).flatMap (growImg (pg b) k) := by
  rw [ep_wires_map D b k hW hb, ep_bnd D b k]
  simp only [List.flatMap_append, List.map_map]
  congr 1
  · rw [List.map_eq_flatMap, List.flatMap_map]
    refine List.flatMap_congr fun w hw => ?_
    rw [growImg_of_not _ _ _ (not_pg_of_wire b hW hb hw).1]; rfl
  · congr 1
    rw [List.map_eq_flatMap, List.flatMap_map]
    refine List.flatMap_congr fun w hw => ?_
    rw [growImg_of_not _ _ _ (not_pg_of_wire b hW hb hw).2]; rfl

/-- A port box's instantiation at `k ≥ 1` keeps the box structure. -/
theorem ep_WF (hW : WF D) (hb : b ∈ D.boxes) (hbn : b.nodes = [])
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) : WF (D.growPorts b k) := by
  have hmem := ep_mem_boxes D b k
  refine ⟨by rw [ep_ids D b k]; exact hW.ids, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ep_boxes D b k]
    refine (List.Sublist.map _ List.filter_sublist).nodup ?_
    rw [List.map_map]
    exact (List.Sublist.map _ List.filter_sublist).nodup hW.bids
  · rw [ep_wire_ends D b k hW hb, ep_legs D b k]
    exact hW.perm.flatMap_right _
  · intro y' hy'
    obtain ⟨y, hy, -, rfl, -⟩ := (hmem y').1 hy'
    rcases hW.kinds y hy with h | h
    · exact Or.inl h
    · exact Or.inr (by simp [pbox, h])
  · intro y' hy' x hx
    obtain ⟨y, hy, -, rfl, -⟩ := (hmem y').1 hy'
    rw [ep_ids D b k]
    exact hW.nsub y hy x hx
  · intro x hx
    rw [ep_ids D b k] at hx
    rw [ep_nodeBoxes D b k hW hb hbn x hx]
    exact ep_nchain D b k hW hb hbn x hx
  · intro x y' hy' hx
    obtain ⟨y, hy, -, rfl, -⟩ := (hmem y').1 hy'
    have hxi := hW.nsub y hy x hx
    rw [ep_nodeBoxes D b k hW hb hbn x hxi]
    exact List.mem_map.2 ⟨y, hW.nall x y hy hx, rfl⟩
  · intro q hq
    rw [ep_bnd D b k] at hq
    obtain ⟨p, hp, hqp⟩ := List.mem_flatMap.1 hq
    unfold portBoxes
    have h' := ep_pchain D b k hW hb hbn hleaf p hp q hqp
    rw [h'.walk _ h'.length_le]
    exact h'
  · intro q hq y' hy' hqy
    rw [ep_bnd D b k] at hq
    obtain ⟨p, hp, hqp⟩ := List.mem_flatMap.1 hq
    obtain ⟨y, hy, hky, rfl, -⟩ := (hmem y').1 hy'
    rw [ep_portBoxes D b k hW hb hbn hleaf p hp q hqp]
    obtain ⟨p'', hp'', hq''⟩ := List.mem_flatMap.1 hqy
    have := ep_disj D b k hW.ids (bnd_legs hW hp) (bnd_legs hW (hW.pbnd y hy p'' hp'')) hqp hq''
    subst this
    exact List.mem_map.2 ⟨y, List.mem_filter.2 ⟨hW.pall p hp y hy hp'', hky⟩, rfl⟩
  · intro y' hy' q hq
    obtain ⟨y, hy, -, rfl, -⟩ := (hmem y').1 hy'
    obtain ⟨p, hp, hqp⟩ := List.mem_flatMap.1 hq
    rw [ep_bnd D b k]
    exact List.mem_flatMap.2 ⟨p, hW.pbnd y hy p hp, hqp⟩
  · intro w' hw'
    rw [ep_wires_map D b k hW hb] at hw'
    obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hw'
    obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
    simp only [pshift]
    rw [ep_nodeBoxes D b k hW hb hbn _ (node_of_leg l1),
      ep_nodeBoxes D b k hW hb hbn _ (node_of_leg l2)]
    rcases hW.wpre w hw with h | h
    · exact Or.inl (h.map _)
    · exact Or.inr (h.map _)

end Diagram

end ArchytasZX
