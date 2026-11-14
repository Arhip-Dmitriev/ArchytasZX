/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.BoxFree
import ArchytasZX.LinkedPeel
import ArchytasZX.Weights

/-!
# Dropping a boxed identity spider

`evalL_dropId`: a Z(1 -> 1) node `x` whose phase index reads zero, its leg `u` wired (the wire
`w`) to the leg `P` of another node and its other leg `v` anywhere (a wire, a link end or a
boundary place), dropped by `dropId`: `x` and `w` removed and `v` read as `P` in every wire, link
and boundary place. The linked digit sum is unchanged: the digits of `u` and `v` are summed out,
each pinned to `P`'s.
-/

namespace ArchytasZX

theorem find?_filter_of {α : Type*} (f g : α → Bool) (l : List α)
    (h : ∀ a ∈ l, f a = true → g a = true) : (l.filter g).find? f = l.find? f := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    have ih' := ih fun b hb => h b (List.mem_cons_of_mem _ hb)
    by_cases hf : f a = true
    · rw [List.filter_cons_of_pos (h a List.mem_cons_self hf), List.find?_cons_of_pos hf,
        List.find?_cons_of_pos hf]
    · by_cases hg : g a = true
      · rw [List.filter_cons_of_pos hg, List.find?_cons_of_neg hf, List.find?_cons_of_neg hf, ih']
      · rw [List.filter_cons_of_neg hg, List.find?_cons_of_neg hf, ih']

theorem tZ_id_pair (d : ℕ) (φ : ℕ → ℝ) (a b : ℕ) :
    tZ d φ [a, b] = if a = b ∧ a < d then ex (φ a) else 0 := by
  unfold tZ
  simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  split_ifs with hab
  · obtain ⟨rfl, ha⟩ := hab
    rw [Finset.sum_eq_single a]
    · simp
    · intro k _ hk
      rw [if_neg]
      rintro ⟨h1, -⟩
      exact hk h1.symm
    · intro ha'
      exact absurd (Finset.mem_range.2 ha) ha'
  · refine Finset.sum_eq_zero fun k hk => ?_
    rw [if_neg]
    rintro ⟨rfl, rfl⟩
    exact hab ⟨rfl, Finset.mem_range.1 hk⟩

theorem val_congr (E : Env) (σ σ' : ℕ → ℕ) (n : Node) (ho : n.outs.map σ = n.outs.map σ')
    (hi : n.ins.map σ = n.ins.map σ') : val E σ n = val E σ' n := by
  simp only [val, ho, hi]

namespace Diagram

theorem walk_boxes_congr (D D' : Diagram) (hb : D.boxes = D'.boxes) (p : DBox → Bool) (f : ℕ)
    (s : Option ℕ) : D.walk p f s = D'.walk p f s := by
  induction f generalizing s with
  | zero => rfl
  | succ f ih => simp only [walk, hb, ih]

end Diagram

namespace LDiagram

/-- `p` read as `P` when it is `v`. -/
def substP (v P : Port) (p : Port) : Port := if p = v then P else p

/-- `L` with the node `x` and the wire `w` removed and the leg `v` read as `P`. -/
def dropId (L : LDiagram) (x : ℕ) (w : Port × Port) (v P : Port) : LDiagram :=
  { base :=
      { nodes := L.base.nodes.filter fun n => n.id ≠ x
        wires := (L.base.wires.erase w).map fun e => (substP v P e.1, substP v P e.2)
        outs := L.base.outs.map (substP v P)
        ins := L.base.ins.map (substP v P)
        boxes := L.base.boxes }
    links := L.links.map (DLink.mapP (substP v P))
    cph := L.cph }

/-- The pairs of `L`'s wires other than `w`, then of its links. -/
def restP (L : LDiagram) (w : Port × Port) (ks : ℕ → ℕ) : List (Dig × Dig) :=
  (L.base.wires.erase w).flatMap (fun e =>
    (cidx ks (L.base.legChain e.1)).map fun t => ((e.1, t), (e.2, t))) ++
    L.links.flatMap (L.linkPairs ks)

/-- `L` drops the Z(1 -> 1) node `x` whose leg `u` meets the leg `P` by the wire `w` (`v` its
other leg): plain node boxes, wires within one chain, `x` the last node, `u` nowhere else, `v` no
link port. -/
structure DropOK (L : LDiagram) (x : DNode) (w : Port × Port) (u v P : Port) : Prop where
  ports : ∀ y ∈ L.base.boxes, y.ports = []
  wch : ∀ e ∈ L.base.wires, L.base.nodeChain e.1.node = L.base.nodeChain e.2.node
  cph : L.cph = []
  ids : (L.base.nodes.map (·.id)).Nodup
  last : L.base.nodes = (L.base.nodes.filter fun n => n.id ≠ x.id) ++ [x]
  kind : x.kind = .Z
  outs : x.outs = [x.dim]
  ins : x.ins = [x.dim]
  uv : (u = ⟨x.id, true, 0⟩ ∧ v = ⟨x.id, false, 0⟩) ∨ (u = ⟨x.id, false, 0⟩ ∧ v = ⟨x.id, true, 0⟩)
  hw : w ∈ L.base.wires
  wuP : w = (u, P) ∨ w = (P, u)
  hP : P.node ≠ x.id
  hPl : P ∈ L.base.legs
  pdim : L.base.portDim P = x.dim
  uw : ∀ e ∈ L.base.wires.erase w, e.1 ≠ u ∧ e.2 ≠ u
  ub : u ∉ L.base.bnd
  ul : ∀ l ∈ L.links, l.l1 ≠ u ∧ l.l2 ≠ u ∧ l.e1 ≠ u ∧ l.e2 ≠ u
  vl : ∀ l ∈ L.links, l.l1 ≠ v ∧ l.l2 ≠ v

section drop

variable {L : LDiagram} {x : DNode} {w : Port × Port} {u v P : Port}

local notation "Ld" => L.dropId x.id w v P

theorem nodes_drop (y : ℕ) :
    (L.dropId y w v P).base.nodes = L.base.nodes.filter fun n => n.id ≠ y := rfl

theorem wires_drop (y : ℕ) : (L.dropId y w v P).base.wires =
    (L.base.wires.erase w).map fun e => (substP v P e.1, substP v P e.2) := rfl

theorem links_drop (y : ℕ) : (L.dropId y w v P).links = L.links.map (DLink.mapP (substP v P)) :=
  rfl

theorem nodeBoxes_drop (y : ℕ) (z : ℕ) : (L.dropId y w v P).base.nodeBoxes z = L.base.nodeBoxes z :=
  Diagram.walk_boxes_congr (L.dropId y w v P).base L.base rfl _ _ _

theorem nodeChain_drop (y : ℕ) (z : ℕ) :
    (L.dropId y w v P).base.nodeChain z = L.base.nodeChain z := by
  simp only [Diagram.nodeChain, nodeBoxes_drop]

theorem depth_drop (y b z : ℕ) : (L.dropId y w v P).depth b z = L.depth b z := by
  simp only [depth, nodeBoxes_drop]

theorem multOf_drop (y b : ℕ) : (L.dropId y w v P).multOf b = L.multOf b := rfl

theorem DropOK.unode (h : DropOK L x w u v P) : u.node = x.id := by
  rcases h.uv with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> rfl

theorem DropOK.vnode (h : DropOK L x w u v P) : v.node = x.id := by
  rcases h.uv with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> rfl

theorem DropOK.uv_ne (h : DropOK L x w u v P) : u ≠ v := by
  rcases h.uv with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp

theorem DropOK.Pu (h : DropOK L x w u v P) : P ≠ u := fun e => h.hP (e ▸ h.unode)

theorem DropOK.Pv (h : DropOK L x w u v P) : P ≠ v := fun e => h.hP (e ▸ h.vnode)

theorem DropOK.xmem (h : DropOK L x w u v P) : x ∈ L.base.nodes := by
  rw [h.last]; simp

theorem DropOK.chP (h : DropOK L x w u v P) :
    L.base.nodeChain P.node = L.base.nodeChain x.id := by
  have hc := h.wch w h.hw
  rcases h.wuP with rfl | rfl
  · rw [← hc, h.unode]
  · rw [hc, h.unode]

theorem DropOK.chain_subst (h : DropOK L x w u v P) (a : Port) :
    L.base.nodeChain (substP v P a).node = L.base.nodeChain a.node := by
  unfold substP
  split_ifs with ha
  · rw [h.chP, ha, h.vnode]
  · rfl

theorem DropOK.lc (h : DropOK L x w u v P) (q : Port) :
    L.base.legChain q = L.base.nodeChain q.node := L.base.legChain_eq h.ports h.wch q

theorem DropOK.wch' (h : DropOK L x w u v P) :
    ∀ e ∈ (Ld).base.wires, (Ld).base.nodeChain e.1.node = (Ld).base.nodeChain e.2.node := by
  intro e he
  rw [wires_drop, List.mem_map] at he
  obtain ⟨e0, he0, rfl⟩ := he
  rw [nodeChain_drop, nodeChain_drop, h.chain_subst, h.chain_subst]
  exact h.wch e0 (List.mem_of_mem_erase he0)

theorem DropOK.lc' (h : DropOK L x w u v P) (q : Port) :
    (Ld).base.legChain q = L.base.nodeChain q.node :=
  ((Ld).base.legChain_eq h.ports h.wch' q).trans (nodeChain_drop _ _)

theorem DropOK.xlegs (h : DropOK L x w u v P) :
    x.legs = [⟨x.id, true, 0⟩, ⟨x.id, false, 0⟩] := by
  simp [DNode.legs, h.outs, h.ins]

theorem DropOK.legs_eq (h : DropOK L x w u v P) : L.base.legs = (Ld).base.legs ++ x.legs := by
  have e : L.base.legs = ((L.base.nodes.filter fun n => n.id ≠ x.id) ++ [x]).flatMap DNode.legs := by
    unfold Diagram.legs
    rw [← h.last]
  rw [e, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil]
  rfl

theorem DropOK.node_ne (h : DropOK L x w u v P) (p : Port) (hp : p ∈ (Ld).base.legs) :
    p.node ≠ x.id := by
  obtain ⟨n, hn, e⟩ := node_of_legs _ p hp
  rw [nodes_drop, List.mem_filter, decide_eq_true_eq] at hn
  rw [e]; exact hn.2

theorem DropOK.Pmem (h : DropOK L x w u v P) : P ∈ (Ld).base.legs := by
  have := h.hPl
  rw [h.legs_eq, h.xlegs] at this
  rcases List.mem_append.1 this with h1 | h1
  · exact h1
  · simp only [List.mem_cons, List.mem_nil_iff, or_false] at h1
    rcases h1 with e | e <;> exact absurd (congrArg Port.node e) h.hP

theorem DropOK.portDim_drop (h : DropOK L x w u v P) (z : Port) (hz : z.node ≠ x.id) :
    (Ld).base.portDim z = L.base.portDim z := by
  unfold Diagram.portDim
  rw [nodes_drop, find?_filter_of]
  intro n _ hn
  simp only [decide_eq_true_eq] at hn ⊢
  rw [hn]; exact hz

theorem DropOK.portDim_x (h : DropOK L x w u v P) (b : Bool) :
    L.base.portDim ⟨x.id, b, 0⟩ = x.dim := by
  rw [portDim_of L.base h.ids x h.xmem _ rfl]
  cases b <;> simp [h.outs, h.ins]

/-- The digits of `x`'s two legs. -/
def xdigs (L : LDiagram) (ks : ℕ → ℕ) (x : DNode) : List ℕ :=
  (cidx ks (L.base.nodeChain x.id)).map (dlab ⟨x.id, true, 0⟩) ++
    (cidx ks (L.base.nodeChain x.id)).map (dlab ⟨x.id, false, 0⟩)

theorem DropOK.dlabs_eq (h : DropOK L x w u v P) (ks : ℕ → ℕ) :
    L.base.dlabs ks = (Ld).base.dlabs ks ++ L.xdigs ks x := by
  unfold Diagram.dlabs
  rw [h.legs_eq, List.flatMap_append, h.xlegs]
  congr 1
  · refine List.flatMap_congr fun p _ => ?_
    rw [h.lc, h.lc']
  · simp [xdigs, h.lc]

theorem mem_xdigs (L : LDiagram) (ks : ℕ → ℕ) (x : DNode) (l : ℕ) :
    l ∈ L.xdigs ks x ↔ ∃ t ∈ cidx ks (L.base.nodeChain x.id),
      l = dlab ⟨x.id, true, 0⟩ t ∨ l = dlab ⟨x.id, false, 0⟩ t := by
  simp only [xdigs, List.mem_append, List.mem_map]
  constructor
  · rintro (⟨t, ht, rfl⟩ | ⟨t, ht, rfl⟩)
    · exact ⟨t, ht, Or.inl rfl⟩
    · exact ⟨t, ht, Or.inr rfl⟩
  · rintro ⟨t, ht, rfl | rfl⟩
    · exact Or.inl ⟨t, ht, rfl⟩
    · exact Or.inr ⟨t, ht, rfl⟩

theorem xdigs_nodup (L : LDiagram) (ks : ℕ → ℕ) (x : DNode) : (L.xdigs ks x).Nodup := by
  unfold xdigs
  refine List.nodup_append.2 ⟨?_, ?_, ?_⟩
  · exact (cidx_nodup ks _).map fun a b e => (dlab_inj e).2
  · exact (cidx_nodup ks _).map fun a b e => (dlab_inj e).2
  · simp only [List.mem_map]
    rintro _ ⟨a, -, rfl⟩ _ ⟨b, -, rfl⟩ e
    have := (dlab_inj e).1
    simp at this

/-- A digit label of a leg other than `u` and `v` is no digit of `x`. -/
theorem DropOK.not_xdigs (h : DropOK L x w u v P) (ks : ℕ → ℕ) (p : Port) (hu : p ≠ u)
    (hv : p ≠ v) (t : List ℕ) : dlab p t ∉ L.xdigs ks x := by
  rw [mem_xdigs]
  rintro ⟨t', -, e⟩
  have hp : p = ⟨x.id, true, 0⟩ ∨ p = ⟨x.id, false, 0⟩ := by
    rcases e with e | e
    · exact Or.inl (dlab_inj e).1
    · exact Or.inr (dlab_inj e).1
  rcases h.uv with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> rcases hp with rfl | rfl <;> simp_all

theorem slab_not_xdigs (L : LDiagram) (ks : ℕ → ℕ) (x : DNode) (j : ℕ) (t : List ℕ) :
    slab j t ∉ L.xdigs ks x := by
  rw [mem_xdigs]
  rintro ⟨t', -, e | e⟩ <;> exact dlab_ne_slab _ _ _ _ e.symm

theorem DropOK.mem_dpairs (h : DropOK L x w u v P) (ks : ℕ → ℕ) (y : Dig × Dig) :
    y ∈ L.dpairs ks ↔ (∃ t ∈ cidx ks (L.base.legChain w.1), y = ((w.1, t), (w.2, t))) ∨
      y ∈ L.restP w ks := by
  have hp : List.Perm (L.base.wires.flatMap (fun e =>
      (cidx ks (L.base.legChain e.1)).map fun t => ((e.1, t), (e.2, t))))
      ((w :: L.base.wires.erase w).flatMap (fun e =>
        (cidx ks (L.base.legChain e.1)).map fun t => ((e.1, t), (e.2, t)))) :=
    (List.perm_cons_erase h.hw).flatMap_right _
  simp only [dpairs, restP, List.mem_append]
  rw [hp.mem_iff]
  simp only [List.flatMap_cons, List.mem_append, List.mem_map]
  constructor
  · rintro ((⟨t, ht, rfl⟩ | h1) | h2)
    · exact Or.inl ⟨t, ht, rfl⟩
    · exact Or.inr (Or.inl h1)
    · exact Or.inr (Or.inr h2)
  · rintro (⟨t, ht, rfl⟩ | h1 | h2)
    · exact Or.inl (Or.inl ⟨t, ht, rfl⟩)
    · exact Or.inl (Or.inr h1)
    · exact Or.inr h2

theorem DropOK.dpairs_drop (h : DropOK L x w u v P) (ks : ℕ → ℕ) :
    (Ld).dpairs ks = (L.restP w ks).map (portsP (substP v P)) := by
  simp only [dpairs, restP, List.map_append, List.map_flatMap]
  congr 1
  · rw [wires_drop, List.flatMap_map]
    refine List.flatMap_congr fun e _ => ?_
    rw [h.lc', h.chain_subst, ← h.lc, List.map_map]
    rfl
  · rw [links_drop, List.flatMap_map]
    refine List.flatMap_congr fun l hl => ?_
    have h1 : substP v P l.l1 = l.l1 := if_neg (h.vl l hl).1
    have hb : (l.mapP (substP v P)).box = l.box := rfl
    have hl1 : (l.mapP (substP v P)).l1 = l.l1 := h1
    unfold linkPairs
    rw [hb, hl1, depth_drop, multOf_drop, h.lc', ← h.lc]
    exact lp_mapP ks _ _ _ l _

theorem DropOK.bnd_drop (h : DropOK L x w u v P) : (Ld).base.bnd = L.base.bnd.map (substP v P) := by
  simp only [Diagram.bnd, List.map_append]
  rfl

theorem DropOK.rest_avoid (h : DropOK L x w u v P) (ks : ℕ → ℕ) (y : Dig × Dig)
    (hy : y ∈ L.restP w ks) : y.1.1 ≠ u ∧ y.2.1 ≠ u := by
  simp only [restP, List.mem_append, List.mem_flatMap, List.mem_map] at hy
  rcases hy with ⟨e, he, t, -, rfl⟩ | ⟨l, hl, hy⟩
  · exact h.uw e he
  · obtain ⟨a1, a2⟩ := lp_ports ks _ _ _ l y hy
    obtain ⟨b1, b2, b3, b4⟩ := h.ul l hl
    exact ⟨by rcases a1 with e | e <;> rw [e] <;> assumption,
      by rcases a2 with e | e <;> rw [e] <;> assumption⟩

theorem DropOK.subst_ne (h : DropOK L x w u v P) (a : Port) (ha : a ≠ u) :
    substP v P a ≠ u ∧ substP v P a ≠ v := by
  unfold substP
  split_ifs with e
  · exact ⟨h.Pu, h.Pv⟩
  · exact ⟨ha, e⟩

theorem DropOK.pairs_avoid (h : DropOK L x w u v P) (ks : ℕ → ℕ) (y : Dig × Dig)
    (hy : y ∈ (Ld).dpairs ks) : (y.1.1 ≠ u ∧ y.1.1 ≠ v) ∧ (y.2.1 ≠ u ∧ y.2.1 ≠ v) := by
  rw [h.dpairs_drop, List.mem_map] at hy
  obtain ⟨y0, hy0, rfl⟩ := hy
  obtain ⟨a1, a2⟩ := h.rest_avoid ks y0 hy0
  exact ⟨h.subst_ne _ a1, h.subst_ne _ a2⟩

/-- The integrand of `L`'s digit sum. -/
noncomputable def integ (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : ℂ :=
  open Classical in if L.pinsL ks ν then L.prodL E ks ν else 0

theorem DropOK.prodL_eq (h : DropOK L x w u v P) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.prodL E ks ν = (Ld).prodL E ks ν * ((cidx ks (L.base.nodeChain x.id)).map fun t =>
      tZ (E.dimv x.dim) (E.phase x.ph) [ν (dlab ⟨x.id, true, 0⟩ t), ν (dlab ⟨x.id, false, 0⟩ t)]).prod := by
  have hcph' : (Ld).cph = [] := h.cph
  unfold prodL
  conv_lhs => rw [h.last]
  rw [List.map_append, List.prod_append, List.map_singleton, List.prod_singleton]
  congr 1
  · rw [nodes_drop]
    refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
    rw [nodeChain_drop]
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [L.dnodeL_plain h.cph, (Ld).dnodeL_plain hcph',
      L.base.dnode_notail h.ports h.wch, (Ld).base.dnode_notail h.ports h.wch']
  · refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [L.dnodeL_plain h.cph, L.base.dnode_notail h.ports h.wch]
    simp [val, h.kind, h.outs, h.ins]

theorem DropOK.integ_reads (h : DropOK L x w u v P) (E : Env) (ks : ℕ → ℕ) (ν ν' : ℕ → ℕ)
    (hag : ∀ l ∉ L.xdigs ks x, ν l = ν' l) : (Ld).integ E ks ν = (Ld).integ E ks ν' := by
  have hd : ∀ p t, p ≠ u ∧ p ≠ v → ν (dlab p t) = ν' (dlab p t) :=
    fun p t hp => hag _ (h.not_xdigs ks p hp.1 hp.2 t)
  have hs : ∀ j t, ν (slab j t) = ν' (slab j t) := fun j t => hag _ (slab_not_xdigs L ks x j t)
  have hpins : (Ld).pinsL ks ν ↔ (Ld).pinsL ks ν' := by
    unfold pinsL
    refine and_congr (forall₂_congr fun y hy => ?_) (forall₂_congr fun q hq => forall₂_congr
      fun t _ => ?_)
    · obtain ⟨a, b⟩ := h.pairs_avoid ks y hy
      rw [hd _ _ a, hd _ _ b]
    · have hq' : q.1 ∈ (Ld).base.bnd := (List.mem_zipIdx hq).2.2 ▸ List.getElem_mem _
      rw [h.bnd_drop, List.mem_map] at hq'
      obtain ⟨q0, hq0, e⟩ := hq'
      rw [hd _ _ (e ▸ h.subst_ne q0 (fun e' => h.ub (e' ▸ hq0))), hs]
  have hprod : (Ld).prodL E ks ν = (Ld).prodL E ks ν' := by
    have hcph' : (Ld).cph = [] := h.cph
    unfold prodL
    refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [(Ld).dnodeL_plain hcph', (Ld).base.dnode_notail h.ports h.wch']
    have hn' : n.id ≠ x.id := by
      rw [nodes_drop, List.mem_filter, decide_eq_true_eq] at hn
      exact hn.2
    have hne : ∀ b i, (⟨n.id, b, i⟩ : Port) ≠ u ∧ (⟨n.id, b, i⟩ : Port) ≠ v := fun b i =>
      ⟨fun e => hn' (by rw [← h.unode, ← e]), fun e => hn' (by rw [← h.vnode, ← e])⟩
    apply val_congr
    · simp only [List.map_map]
      exact List.map_congr_left fun i _ => hd _ _ (hne _ _)
    · simp only [List.map_map]
      exact List.map_congr_left fun i _ => hd _ _ (hne _ _)
  unfold integ
  rw [hprod]
  classical
  exact if_congr hpins rfl rfl

theorem DropOK.lw1 (h : DropOK L x w u v P) : L.base.legChain w.1 = L.base.nodeChain x.id := by
  rw [h.lc]
  rcases h.wuP with e | e <;> rw [e]
  · rw [h.unode]
  · exact h.chP

theorem DropOK.pins_iff (h : DropOK L x w u v P) (ks : ℕ → ℕ) (ν : ℕ → ℕ)
    (hsub : ∀ a t, ν (dlab (substP v P a) t) = ν (dlab a t))
    (huP : ∀ t ∈ cidx ks (L.base.nodeChain x.id), ν (dlab u t) = ν (dlab P t)) :
    L.pinsL ks ν ↔ (Ld).pinsL ks ν := by
  have hlc : ∀ p, (Ld).base.legChain (substP v P p) = L.base.legChain p := fun p => by
    rw [h.lc', h.chain_subst, h.lc]
  unfold pinsL
  refine and_congr ⟨fun hp y hy => ?_, fun hp y hy => ?_⟩
    ⟨fun hb q hq t ht => ?_, fun hb q hq t ht => ?_⟩
  · rw [h.dpairs_drop, List.mem_map] at hy
    obtain ⟨y0, hy0, rfl⟩ := hy
    have := hp y0 ((h.mem_dpairs ks y0).2 (Or.inr hy0))
    simp only [portsP]
    rw [hsub, hsub]
    exact this
  · rcases (h.mem_dpairs ks y).1 hy with ⟨t, ht, rfl⟩ | hy0
    · rw [h.lw1] at ht
      rcases h.wuP with e | e <;> rw [e]
      · exact huP t ht
      · exact (huP t ht).symm
    · have := hp _ (by rw [h.dpairs_drop]; exact List.mem_map.2 ⟨y, hy0, rfl⟩)
      simp only [portsP] at this
      rwa [hsub, hsub] at this
  · obtain ⟨q, j⟩ := q
    rw [h.bnd_drop] at hq
    obtain ⟨hj, e⟩ := (mem_zipIdx' _ _ _).1 hq
    rw [List.length_map] at hj
    rw [List.getElem_map] at e
    subst e
    rw [hlc] at ht
    have := hb (L.base.bnd[j], j) ((mem_zipIdx' _ _ _).2 ⟨hj, rfl⟩) t ht
    simp only at this ⊢
    rw [hsub]
    exact this
  · obtain ⟨p, j⟩ := q
    obtain ⟨hj, e⟩ := (mem_zipIdx' _ _ _).1 hq
    subst e
    have hq' : (substP v P L.base.bnd[j], j) ∈ (Ld).base.bnd.zipIdx := by
      rw [h.bnd_drop]
      exact (mem_zipIdx' _ _ _).2 ⟨by simpa using hj, by simp⟩
    have := hb _ hq' t (by rw [hlc]; exact ht)
    simp only at this ⊢
    rw [hsub] at this
    exact this

theorem DropOK.integ_point (h : DropOK L x w u v P) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) (τ ν : ℕ → ℕ)
    (hτ : ∀ y ∉ (Ld).base.dlabs ks, τ y = slotsD σ y)
    (hν : ∀ y ∉ L.xdigs ks x, ν y = τ y) (hνr : ∀ y ∈ L.xdigs ks x, ν y < E.dimv x.dim) :
    L.integ E ks ν =
      if ∀ l ∈ L.xdigs ks x, ν l = τ (dlab P (dtup l)) then (Ld).integ E ks ν else 0 := by
  have hPx : ∀ t, dlab P t ∉ L.xdigs ks x := fun t => h.not_xdigs ks P h.Pu h.Pv t
  have hPt : ∀ t, ν (dlab P t) = τ (dlab P t) := fun t => hν _ (hPx t)
  have hux : ∀ t ∈ cidx ks (L.base.nodeChain x.id), dlab u t ∈ L.xdigs ks x := by
    intro t ht
    rw [mem_xdigs]
    rcases h.uv with ⟨rfl, -⟩ | ⟨rfl, -⟩
    · exact ⟨t, ht, Or.inl rfl⟩
    · exact ⟨t, ht, Or.inr rfl⟩
  have hvx : ∀ t ∈ cidx ks (L.base.nodeChain x.id), dlab v t ∈ L.xdigs ks x := by
    intro t ht
    rw [mem_xdigs]
    rcases h.uv with ⟨-, rfl⟩ | ⟨-, rfl⟩
    · exact ⟨t, ht, Or.inr rfl⟩
    · exact ⟨t, ht, Or.inl rfl⟩
  have hvx' : ∀ t, t ∉ cidx ks (L.base.nodeChain x.id) → dlab v t ∉ L.xdigs ks x := by
    intro t ht hm
    rw [mem_xdigs] at hm
    obtain ⟨t', ht', e | e⟩ := hm <;> exact ht ((dlab_inj e).2 ▸ ht')
  have hwP : L.pinsL ks ν → ∀ t ∈ cidx ks (L.base.nodeChain x.id),
      ν (dlab u t) = ν (dlab P t) := by
    intro hp t ht
    have := hp.1 _ ((h.mem_dpairs ks _).2 (Or.inl ⟨t, h.lw1 ▸ ht, rfl⟩))
    rcases h.wuP with e | e <;> rw [e] at this
    · exact this
    · exact this.symm
  unfold integ
  by_cases hA : ∀ l ∈ L.xdigs ks x, ν l = τ (dlab P (dtup l))
  · rw [if_pos hA]
    have hsub : ∀ a t, ν (dlab (substP v P a) t) = ν (dlab a t) := by
      intro a t
      unfold substP
      split_ifs with ha
      · rw [ha]
        by_cases ht : t ∈ cidx ks (L.base.nodeChain x.id)
        · rw [hA _ (hvx t ht), dtup_dlab, hPt]
        · have h1 : dlab v t ∉ (Ld).base.dlabs ks := by
            rw [mem_dlabs]
            rintro ⟨p, hp, t', -, e⟩
            obtain ⟨rfl, -⟩ := dlab_inj e
            exact h.node_ne _ hp h.vnode
          have h2 : dlab P t ∉ (Ld).base.dlabs ks := by
            rw [mem_dlabs]
            rintro ⟨p, -, t', ht', e⟩
            obtain ⟨rfl, rfl⟩ := dlab_inj e
            rw [h.lc', h.chP] at ht'
            exact ht ht'
          rw [hPt, hτ _ h2, hν _ (hvx' t ht), hτ _ h1, slotsD_dlab, slotsD_dlab]
      · rfl
    have huP : ∀ t ∈ cidx ks (L.base.nodeChain x.id), ν (dlab u t) = ν (dlab P t) :=
      fun t ht => by rw [hA _ (hux t ht), dtup_dlab, hPt]
    have hone : ((cidx ks (L.base.nodeChain x.id)).map fun t =>
        tZ (E.dimv x.dim) (E.phase x.ph)
          [ν (dlab ⟨x.id, true, 0⟩ t), ν (dlab ⟨x.id, false, 0⟩ t)]).prod = 1 := by
      refine List.prod_eq_one fun z hz => ?_
      obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hz
      have ho : dlab ⟨x.id, true, 0⟩ t ∈ L.xdigs ks x := (mem_xdigs _ _ _ _).2 ⟨t, ht, Or.inl rfl⟩
      have hi : dlab ⟨x.id, false, 0⟩ t ∈ L.xdigs ks x :=
        (mem_xdigs _ _ _ _).2 ⟨t, ht, Or.inr rfl⟩
      rw [tZ_id_pair, if_pos ⟨by rw [hA _ ho, hA _ hi, dtup_dlab, dtup_dlab], hνr _ ho⟩, hE]
      simp [ex]
    rw [h.prodL_eq, hone, mul_one]
    classical
    exact if_congr (h.pins_iff ks ν hsub huP) rfl rfl
  · rw [if_neg hA]
    split_ifs with hp
    · rw [h.prodL_eq]
      refine mul_eq_zero_of_right _ (List.prod_eq_zero ?_)
      have hu := hwP hp
      obtain ⟨l, hl, hne⟩ : ∃ l ∈ L.xdigs ks x, ¬ ν l = τ (dlab P (dtup l)) := by
        by_contra hc
        exact hA fun l hl => by_contra fun hn => hc ⟨l, hl, hn⟩
      obtain ⟨t, ht, hlt⟩ := (mem_xdigs _ _ _ _).1 hl
      refine List.mem_map.2 ⟨t, ht, ?_⟩
      rw [tZ_id_pair, if_neg]
      rintro ⟨heq, -⟩
      apply hne
      have hu' := hu t ht
      rw [hPt] at hu'
      rcases h.uv with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> rcases hlt with rfl | rfl <;> rw [dtup_dlab]
      · exact hu'
      · rw [← heq]; exact hu'
      · rw [heq]; exact hu'
      · exact hu'
    · rfl

/-- Dropping `x` keeps the linked digit sum. -/
theorem DropOK.evalL_drop (h : DropOK L x w u v P) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) : L.evalL E ks σ = (Ld).evalL E ks σ := by
  have hX := xdigs_nodup L ks x
  have hdimX : ∀ l ∈ L.xdigs ks x, E.dimv (L.base.portDim (dport l)) = E.dimv x.dim := by
    intro l hl
    obtain ⟨t, -, rfl | rfl⟩ := (mem_xdigs L ks x l).1 hl <;> rw [dport_dlab, h.portDim_x]
  have hdims : ∀ l ∈ (Ld).base.dlabs ks, E.dimv (L.base.portDim (dport l)) =
      E.dimv ((Ld).base.portDim (dport l)) := by
    intro l hl
    obtain ⟨p, hp, t, -, rfl⟩ := (mem_dlabs _ ks l).1 hl
    rw [dport_dlab, h.portDim_drop p (h.node_ne p hp)]
  change sumOver _ (L.base.dlabs ks) (L.integ E ks) (slotsD σ) =
    sumOver _ ((Ld).base.dlabs ks) ((Ld).integ E ks) (slotsD σ)
  rw [h.dlabs_eq ks, sumOver_append, sumOver_dim_congr _ _ _ hdims]
  refine sumOver_congr _ _ _ _ _ fun τ hτ hr => ?_
  have hvf : ∀ l ∈ L.xdigs ks x, τ (dlab P (dtup l)) < E.dimv (L.base.portDim (dport l)) := by
    intro l hl
    rw [hdimX l hl]
    obtain ⟨t, ht, e⟩ := (mem_xdigs L ks x l).1 hl
    have ht' : dtup l = t := by rcases e with rfl | rfl <;> exact dtup_dlab _ _
    rw [ht']
    have hm : dlab P t ∈ (Ld).base.dlabs ks :=
      (mem_dlabs _ ks _).2 ⟨P, h.Pmem, t, by rw [h.lc', h.chP]; exact ht, rfl⟩
    have := hr _ hm
    rwa [dport_dlab, h.portDim_drop P h.hP, h.pdim] at this
  rw [sumOver_congr _ _ _ (fun ν => if ∀ l ∈ L.xdigs ks x, ν l = τ (dlab P (dtup l)) then
      (Ld).integ E ks ν else 0) _ fun ν hν hνr => h.integ_point E hE ks σ τ ν hτ hν
      fun y hy => by rw [← hdimX y hy]; exact hνr y hy]
  rw [sumOver_pinV _ hX _ hvf]
  exact h.integ_reads E ks _ _ fun l hl => if_neg hl

end drop

/-- Two linked diagrams that drop the same node `x` to one diagram have equal digit sums. -/
theorem evalL_slide {L L' : LDiagram} {x : DNode} {w w' : Port × Port} {u v P u' v' P' : Port}
    (h : DropOK L x w u v P) (h' : DropOK L' x w' u' v' P')
    (he : L.dropId x.id w v P = L'.dropId x.id w' v' P') (E : Env)
    (hE : ∀ k, E.phase x.ph k = 0) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L'.evalL E ks σ = L.evalL E ks σ := by
  rw [h.evalL_drop E hE, h'.evalL_drop E hE, he]

end LDiagram

end ArchytasZX
