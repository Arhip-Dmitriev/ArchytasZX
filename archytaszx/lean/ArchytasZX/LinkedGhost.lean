/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedForm

/-!
# Link ends on the boundary

A ghost `(g, i, k)` is a port `g` of no node standing for boundary slot `i`, of dimension index
`k`: `labG gs` reads a ghost's digit `t` as slot `i`'s digit `t` and every other digit as itself.
`evalG L gs` is the digit sum of `L` with every digit pair and boundary place read through
`labG gs` and every ghost's slot digit below its dimension; `evalG_nil`: with no ghost it is
`evalL`.

`ghostDrop L x a c g`: the node `x` removed and its legs `a` and `c` read as `g`. `GhostOK`: `x`
is an unboxed Z(1 -> 1) without wires, its leg `a` only at link ends of links whose ends carry no
copy index, its leg `c` only at boundary place `i`, `g` fresh. `GhostOK.evalG_ghost`: dropping `x`
with the ghost `(g, i, x.dim)` keeps the digit sum when `x`'s phase index reads zero;
`evalG_ghostAt`: the same with `x` moved last first (`evalG_withNodes`).
-/

namespace ArchytasZX

namespace LDiagram

/-- Ghost ports with their slots and dimension indices. -/
abbrev Ghosts := List (Port × ℕ × ℕ)

/-- Digit `t` of `p`: its slot's digit `t` when `p` is a ghost. -/
def labG (gs : Ghosts) (p : Port) (t : List ℕ) : ℕ :=
  match gs.lookup p with
  | some q => slab q.1 t
  | none => dlab p t

theorem labG_nil (p : Port) (t : List ℕ) : labG [] p t = dlab p t := rfl

theorem labG_self (g : Port) (i k : ℕ) (gs : Ghosts) (t : List ℕ) :
    labG ((g, i, k) :: gs) g t = slab i t := by
  simp [labG, List.lookup]

theorem labG_ne {g p : Port} (h : p ≠ g) (i k : ℕ) (gs : Ghosts) (t : List ℕ) :
    labG ((g, i, k) :: gs) p t = labG gs p t := by
  have : (p == g) = false := by simpa using h
  simp only [labG, List.lookup, this]

theorem labG_none {gs : Ghosts} {p : Port} (h : ∀ q ∈ gs, q.1 ≠ p) (t : List ℕ) :
    labG gs p t = dlab p t := by
  have : gs.lookup p = none := by
    induction gs with
    | nil => rfl
    | cons q gs ih =>
      have hq : (p == q.1) = false := by simpa using (h q List.mem_cons_self).symm
      simp only [List.lookup, hq]
      exact ih fun r hr => h r (List.mem_cons_of_mem _ hr)
  simp only [labG, this]

/-- Digit pairs agreeing and boundary places reading their slots through `labG gs`, every ghost's
slot digit below its dimension. -/
def pinsG (L : LDiagram) (gs : Ghosts) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : Prop :=
  (∀ x ∈ L.dpairs ks, ν (labG gs x.1.1 x.1.2) = ν (labG gs x.2.1 x.2.2)) ∧
    (∀ q ∈ L.base.bnd.zipIdx, ∀ t ∈ cidx ks (L.base.legChain q.1),
      ν (labG gs q.1 t) = ν (slab q.2 t)) ∧
    ∀ q ∈ gs, ν (slab q.2.1 []) < E.dimv q.2.2

/-- The integrand of `L`'s digit sum with the ghosts `gs`. -/
noncomputable def integG (L : LDiagram) (gs : Ghosts) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : ℂ :=
  open Classical in if L.pinsG gs E ks ν then L.prodL E ks ν else 0

/-- The linked digit sum with the ghosts `gs`. -/
noncomputable def evalG (L : LDiagram) (gs : Ghosts) (E : Env) (ks : ℕ → ℕ)
    (σ : ℕ → List ℕ → ℕ) : ℂ :=
  sumOver (fun l => E.dimv (L.base.portDim (dport l))) (L.base.dlabs ks) (L.integG gs E ks)
    (slotsD σ)

/-- With no ghost the digit sum is `evalL`. -/
theorem evalG_nil (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalG [] E ks σ = L.evalL E ks σ := by
  unfold evalG evalL
  congr 1
  funext ν
  have h : L.pinsG [] E ks ν ↔ L.pinsL ks ν := by
    simp only [pinsG, pinsL, labG_nil]
    exact ⟨fun h => ⟨h.1, h.2.1⟩, fun h => ⟨h.1, h.2, fun q hq => by simp at hq⟩⟩
  unfold integG
  classical
  exact if_congr h rfl rfl

/-- `p` read as `g` when it is `a` or `c`. -/
def substG (a c g : Port) (p : Port) : Port := if p = a ∨ p = c then g else p

/-- `L` with the node `x` removed and the legs `a` and `c` read as `g`. -/
def ghostDrop (L : LDiagram) (x : ℕ) (a c g : Port) : LDiagram :=
  { base :=
      { nodes := L.base.nodes.filter fun n => n.id ≠ x
        wires := L.base.wires
        outs := L.base.outs.map (substG a c g)
        ins := L.base.ins.map (substG a c g)
        boxes := L.base.boxes }
    links := L.links.map (DLink.mapP (substG a c g))
    cph := L.cph }

/-- `L` drops the unboxed, unwired Z(1 -> 1) node `x` whose leg `a` lies only at link ends of
links whose ends carry no copy index and whose leg `c` lies only at boundary place `i`, reading
both as the fresh ghost `g`. -/
structure GhostOK (L : LDiagram) (gs : Ghosts) (x : DNode) (a c g : Port) (i : ℕ) : Prop where
  ports : ∀ y ∈ L.base.boxes, y.ports = []
  wch : ∀ e ∈ L.base.wires, L.base.nodeChain e.1.node = L.base.nodeChain e.2.node
  cph : L.cph = []
  ids : (L.base.nodes.map (·.id)).Nodup
  last : L.base.nodes = (L.base.nodes.filter fun n => n.id ≠ x.id) ++ [x]
  kind : x.kind = .Z
  outs : x.outs = [x.dim]
  ins : x.ins = [x.dim]
  ac : (a = ⟨x.id, true, 0⟩ ∧ c = ⟨x.id, false, 0⟩) ∨ (a = ⟨x.id, false, 0⟩ ∧ c = ⟨x.id, true, 0⟩)
  xbox : ∀ y ∈ L.base.boxes, x.id ∉ y.nodes
  xw : ∀ e ∈ L.base.wires, e.1.node ≠ x.id ∧ e.2.node ≠ x.id
  bnd : L.base.bnd.Nodup
  ci : L.base.bnd[i]? = some c
  ab : a ∉ L.base.bnd
  gb : g ∉ L.base.bnd
  gw : ∀ e ∈ L.base.wires, e.1 ≠ g ∧ e.2 ≠ g
  gx : g.node ≠ x.id
  cl : ∀ l ∈ L.links, l.l1 ≠ c ∧ l.l2 ≠ c ∧ l.e1 ≠ c ∧ l.e2 ≠ c
  al : ∀ l ∈ L.links, l.l1 ≠ a ∧ l.l2 ≠ a
  gl : ∀ l ∈ L.links, l.l1 ≠ g ∧ l.l2 ≠ g ∧ l.e1 ≠ g ∧ l.e2 ≠ g
  arest : ∀ l ∈ L.links, l.e1 = a ∨ l.e2 = a →
    (L.base.legChain l.l1).eraseIdx (L.depth l.box l.l1.node) = []
  gsf : ∀ q ∈ gs, q.1 ≠ a ∧ q.1 ≠ c ∧ q.1 ≠ g

section ghost

variable {L : LDiagram} {gs : Ghosts} {x : DNode} {a c g : Port} {i : ℕ}

local notation "Lg" => L.ghostDrop x.id a c g

theorem nodeChain_gd (y : ℕ) (z : ℕ) (a c g : Port) :
    (L.ghostDrop y a c g).base.nodeChain z = L.base.nodeChain z := by
  simp only [Diagram.nodeChain, Diagram.nodeBoxes]
  rw [Diagram.walk_boxes_congr (L.ghostDrop y a c g).base L.base rfl]
  rfl

theorem depth_gd (y b z : ℕ) (a c g : Port) : (L.ghostDrop y a c g).depth b z = L.depth b z := by
  simp only [depth, Diagram.nodeBoxes]
  rw [Diagram.walk_boxes_congr (L.ghostDrop y a c g).base L.base rfl]
  rfl

theorem multOf_gd (y b : ℕ) (a c g : Port) : (L.ghostDrop y a c g).multOf b = L.multOf b := rfl

theorem GhostOK.lc (h : GhostOK L gs x a c g i) (q : Port) :
    L.base.legChain q = L.base.nodeChain q.node := L.base.legChain_eq h.ports h.wch q

theorem GhostOK.wch' (h : GhostOK L gs x a c g i) :
    ∀ e ∈ (Lg).base.wires, (Lg).base.nodeChain e.1.node = (Lg).base.nodeChain e.2.node := by
  intro e he
  rw [nodeChain_gd, nodeChain_gd]
  exact h.wch e he

theorem GhostOK.lc' (h : GhostOK L gs x a c g i) (q : Port) :
    (Lg).base.legChain q = L.base.nodeChain q.node :=
  ((Lg).base.legChain_eq h.ports h.wch' q).trans (nodeChain_gd _ _ _ _ _)

theorem walk_none (D : Diagram) (p : DBox → Bool) (hp : ∀ y ∈ D.boxes, p y = false) (f : ℕ)
    (s : Option ℕ) : D.walk p f s = [] := by
  cases f with
  | zero => rfl
  | succ f =>
    have : D.boxes.find? (fun b => p b && b.parent == s) = none :=
      List.find?_eq_none.2 fun y hy => by simp [hp y hy]
    simp [Diagram.walk, this]

theorem GhostOK.xchain (h : GhostOK L gs x a c g i) : L.base.nodeChain x.id = [] := by
  simp only [Diagram.nodeChain, Diagram.nodeBoxes]
  rw [walk_none _ _ fun y hy => by simpa using h.xbox y hy]
  rfl

theorem GhostOK.anode (h : GhostOK L gs x a c g i) : a.node = x.id := by
  rcases h.ac with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> rfl

theorem GhostOK.cnode (h : GhostOK L gs x a c g i) : c.node = x.id := by
  rcases h.ac with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> rfl

theorem GhostOK.ac_ne (h : GhostOK L gs x a c g i) : a ≠ c := by
  rcases h.ac with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp

theorem GhostOK.ga (h : GhostOK L gs x a c g i) : g ≠ a := fun e => h.gx (e ▸ h.anode)

theorem GhostOK.gc (h : GhostOK L gs x a c g i) : g ≠ c := fun e => h.gx (e ▸ h.cnode)

theorem GhostOK.xmem (h : GhostOK L gs x a c g i) : x ∈ L.base.nodes := by
  rw [h.last]; simp

theorem GhostOK.legs_eq (h : GhostOK L gs x a c g i) : L.base.legs = (Lg).base.legs ++ x.legs := by
  have e : L.base.legs = ((L.base.nodes.filter fun n => n.id ≠ x.id) ++ [x]).flatMap DNode.legs := by
    unfold Diagram.legs
    rw [← h.last]
  rw [e, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil]
  rfl

theorem GhostOK.xlegs (h : GhostOK L gs x a c g i) :
    x.legs = [⟨x.id, true, 0⟩, ⟨x.id, false, 0⟩] := by
  simp [DNode.legs, h.outs, h.ins]

theorem GhostOK.node_ne (h : GhostOK L gs x a c g i) (p : Port) (hp : p ∈ (Lg).base.legs) :
    p.node ≠ x.id := by
  obtain ⟨n, hn, e⟩ := node_of_legs _ p hp
  have hn' : n ∈ L.base.nodes.filter fun n => n.id ≠ x.id := hn
  rw [List.mem_filter, decide_eq_true_eq] at hn'
  rw [e]; exact hn'.2

theorem nodes_gd (y : ℕ) (a c g : Port) :
    (L.ghostDrop y a c g).base.nodes = L.base.nodes.filter fun n => n.id ≠ y := rfl

theorem GhostOK.portDim_gd (h : GhostOK L gs x a c g i) (z : Port) (hz : z.node ≠ x.id) :
    (Lg).base.portDim z = L.base.portDim z := by
  unfold Diagram.portDim
  rw [nodes_gd, find?_filter_of]
  intro n _ hn
  simp only [decide_eq_true_eq] at hn ⊢
  rw [hn]; exact hz

theorem GhostOK.portDim_x (h : GhostOK L gs x a c g i) (b : Bool) :
    L.base.portDim ⟨x.id, b, 0⟩ = x.dim := by
  rw [portDim_of L.base h.ids x h.xmem _ rfl]
  cases b <;> simp [h.outs, h.ins]

/-- `x`'s two digits. -/
def gdigs (x : DNode) : List ℕ := [dlab ⟨x.id, true, 0⟩ [], dlab ⟨x.id, false, 0⟩ []]

theorem gdigs_nodup (x : DNode) : (gdigs x).Nodup := by
  simp only [gdigs, List.nodup_cons, List.mem_cons, List.mem_nil_iff, or_false, List.not_mem_nil,
    not_false_eq_true, List.nodup_nil, and_true]
  intro e
  have := (dlab_inj e).1
  simp at this

theorem GhostOK.dlabs_eq (h : GhostOK L gs x a c g i) (ks : ℕ → ℕ) :
    L.base.dlabs ks = (Lg).base.dlabs ks ++ gdigs x := by
  unfold Diagram.dlabs
  rw [h.legs_eq, List.flatMap_append, h.xlegs]
  congr 1
  · refine List.flatMap_congr fun p _ => ?_
    rw [h.lc, h.lc']
  · simp [gdigs, h.lc, h.xchain, cidx]

theorem GhostOK.not_gdigs (h : GhostOK L gs x a c g i) (p : Port) (hp : p.node ≠ x.id)
    (t : List ℕ) : dlab p t ∉ gdigs x := by
  simp only [gdigs, List.mem_cons, List.mem_nil_iff, or_false]
  rintro (e | e) <;> exact hp (congrArg Port.node (dlab_inj e).1)

theorem GhostOK.not_gdigs' (h : GhostOK L gs x a c g i) (p : Port) (ha : p ≠ a) (hc : p ≠ c)
    (t : List ℕ) : dlab p t ∉ gdigs x := by
  simp only [gdigs, List.mem_cons, List.mem_nil_iff, or_false]
  rintro (e | e) <;> obtain ⟨rfl, -⟩ := dlab_inj e <;>
    rcases h.ac with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp_all

theorem slab_not_gdigs (x : DNode) (j : ℕ) (t : List ℕ) : slab j t ∉ gdigs x := by
  simp only [gdigs, List.mem_cons, List.mem_nil_iff, or_false]
  rintro (e | e) <;> exact dlab_ne_slab _ _ _ _ e.symm

theorem GhostOK.prodL_eq (h : GhostOK L gs x a c g i) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.prodL E ks ν = (Lg).prodL E ks ν *
      tZ (E.dimv x.dim) (E.phase x.ph) [ν (dlab ⟨x.id, true, 0⟩ []), ν (dlab ⟨x.id, false, 0⟩ [])] := by
  have hcph' : (Lg).cph = [] := h.cph
  unfold prodL
  conv_lhs => rw [h.last]
  rw [List.map_append, List.prod_append, List.map_singleton, List.prod_singleton]
  congr 1
  · refine congrArg List.prod (List.map_congr_left fun n _ => ?_)
    rw [nodeChain_gd]
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [L.dnodeL_plain h.cph, (Lg).dnodeL_plain hcph',
      L.base.dnode_notail h.ports h.wch, (Lg).base.dnode_notail h.ports h.wch']
  · rw [h.xchain]
    simp [cidx, L.dnodeL_plain h.cph, L.base.dnode_notail h.ports h.wch, val, h.kind, h.outs,
      h.ins]

theorem GhostOK.substG_of (h : GhostOK L gs x a c g i) {p : Port} (ha : p ≠ a) (hc : p ≠ c) :
    substG a c g p = p := if_neg (by tauto)

theorem GhostOK.dpairs_gd (h : GhostOK L gs x a c g i) (ks : ℕ → ℕ) :
    (Lg).dpairs ks = (L.dpairs ks).map (portsP (substG a c g)) := by
  simp only [dpairs, List.map_append, List.map_flatMap]
  congr 1
  · refine List.flatMap_congr fun e he => ?_
    have hc' : (Lg).base.legChain e.1 = L.base.legChain e.1 := by rw [h.lc', h.lc]
    have h1 : e.1 ≠ a ∧ e.1 ≠ c := ⟨fun q => (h.xw e he).1 (q ▸ h.anode),
      fun q => (h.xw e he).1 (q ▸ h.cnode)⟩
    have h2 : e.2 ≠ a ∧ e.2 ≠ c := ⟨fun q => (h.xw e he).2 (q ▸ h.anode),
      fun q => (h.xw e he).2 (q ▸ h.cnode)⟩
    change ((cidx ks ((Lg).base.legChain e.1)).map fun t => ((e.1, t), (e.2, t))) = _
    rw [hc', List.map_map]
    refine List.map_congr_left fun t _ => ?_
    simp only [Function.comp_apply, portsP, h.substG_of h1.1 h1.2, h.substG_of h2.1 h2.2]
  · change (L.links.map (DLink.mapP (substG a c g))).flatMap ((Lg).linkPairs ks) = _
    rw [List.flatMap_map]
    refine List.flatMap_congr fun l hl => ?_
    have h1 : substG a c g l.l1 = l.l1 := h.substG_of (h.al l hl).1 (h.cl l hl).1
    have hb : (l.mapP (substG a c g)).box = l.box := rfl
    have hl1 : (l.mapP (substG a c g)).l1 = l.l1 := h1
    unfold linkPairs
    rw [hb, hl1, depth_gd, multOf_gd, h.lc', ← h.lc]
    exact lp_mapP ks _ _ _ l _

theorem GhostOK.bnd_gd (h : GhostOK L gs x a c g i) : (Lg).base.bnd = L.base.bnd.map (substG a c g) := by
  simp only [Diagram.bnd, List.map_append]
  rfl

/-- The ports of `L`'s digit pairs: neither `c` nor `g`, and `a` only with no copy index. -/
theorem GhostOK.pair_ports (h : GhostOK L gs x a c g i) (ks : ℕ → ℕ) (y : Dig × Dig)
    (hy : y ∈ L.dpairs ks) :
    (y.1.1 ≠ c ∧ y.1.1 ≠ g ∧ (y.1.1 = a → y.1.2 = [])) ∧
      (y.2.1 ≠ c ∧ y.2.1 ≠ g ∧ (y.2.1 = a → y.2.2 = [])) := by
  simp only [dpairs, List.mem_append, List.mem_flatMap, List.mem_map] at hy
  rcases hy with ⟨e, he, t, -, rfl⟩ | ⟨l, hl, hy⟩
  · have hxa : ∀ p : Port, p.node ≠ x.id → p ≠ a ∧ p ≠ c := fun p hp =>
      ⟨fun q => hp (q ▸ h.anode), fun q => hp (q ▸ h.cnode)⟩
    obtain ⟨n1, n2⟩ := h.xw e he
    obtain ⟨g1, g2⟩ := h.gw e he
    exact ⟨⟨(hxa _ n1).2, g1, fun q => absurd q (hxa _ n1).1⟩,
      ⟨(hxa _ n2).2, g2, fun q => absurd q (hxa _ n2).1⟩⟩
  · obtain ⟨c1, c2, c3, c4⟩ := h.cl l hl
    obtain ⟨g1, g2, g3, g4⟩ := h.gl l hl
    obtain ⟨a1, a2⟩ := h.al l hl
    have hr : ∀ u, (∃ t ∈ cidx ks (L.base.legChain l.l1),
        u = t.eraseIdx (L.depth l.box l.l1.node)) ∨
        u ∈ cidx ks ((L.base.legChain l.l1).eraseIdx (L.depth l.box l.l1.node)) →
        (l.e1 = a ∨ l.e2 = a) → u = [] := by
      intro u hu ha
      have hz := h.arest l hl ha
      rcases hu with ⟨t, ht, rfl⟩ | hu
      · have := cidx_eraseIdx ks _ t (L.depth l.box l.l1.node) ht
        rw [hz] at this
        simpa [cidx] using this
      · rw [hz] at hu
        simpa [cidx] using hu
    unfold linkPairs at hy
    rcases (mem_lp ks _ _ _ l y).1 hy with ⟨t, ht, -, rfl⟩ | ⟨t, ht, -, rfl⟩ | ⟨t, ht, -, rfl⟩ |
      ⟨-, u, hu, rfl⟩
    · exact ⟨⟨c1, g1, fun q => absurd q a1⟩, ⟨c2, g2, fun q => absurd q a2⟩⟩
    · exact ⟨⟨c1, g1, fun q => absurd q a1⟩,
        ⟨c3, g3, fun q => hr _ (Or.inl ⟨t, ht, rfl⟩) (Or.inl q)⟩⟩
    · exact ⟨⟨c4, g4, fun q => hr _ (Or.inl ⟨t, ht, rfl⟩) (Or.inr q)⟩,
        ⟨c2, g2, fun q => absurd q a2⟩⟩
    · exact ⟨⟨c4, g4, fun q => hr _ (Or.inr hu) (Or.inr q)⟩,
        ⟨c3, g3, fun q => hr _ (Or.inr hu) (Or.inl q)⟩⟩

theorem GhostOK.c_pin (h : GhostOK L gs x a c g i) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ)
    (hp : L.pinsG gs E ks ν) : ν (dlab c []) = ν (slab i []) := by
  obtain ⟨hi, e⟩ := List.getElem?_eq_some_iff.1 h.ci
  have hm : (c, i) ∈ L.base.bnd.zipIdx := (mem_zipIdx' _ _ _).2 ⟨hi, e⟩
  have := hp.2.1 _ hm [] (by rw [h.lc, h.cnode, h.xchain]; simp [cidx])
  rwa [labG_none (fun q hq => (h.gsf q hq).2.1)] at this

theorem GhostOK.pins_iff (h : GhostOK L gs x a c g i) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ)
    (ha : ν (dlab a []) = ν (slab i [])) (hc : ν (dlab c []) = ν (slab i []))
    (hd : ν (slab i []) < E.dimv x.dim) :
    L.pinsG gs E ks ν ↔ (Lg).pinsG ((g, i, x.dim) :: gs) E ks ν := by
  have hsub : ∀ p t, p ≠ c → p ≠ g → (p = a → t = []) →
      ν (labG ((g, i, x.dim) :: gs) (substG a c g p) t) = ν (labG gs p t) := by
    intro p t hpc hpg hpa
    by_cases hpa' : p = a
    · subst hpa'
      rw [hpa rfl, show substG p c g p = g from if_pos (Or.inl rfl), labG_self,
        labG_none (fun q hq => (h.gsf q hq).1), ha]
    · rw [h.substG_of hpa' hpc, labG_ne hpg]
  obtain ⟨hi, hci⟩ := List.getElem?_eq_some_iff.1 h.ci
  unfold pinsG
  refine ⟨fun hp => ⟨fun y hy => ?_, fun q hq t ht => ?_, ?_⟩,
    fun hp => ⟨fun y hy => ?_, fun q hq t ht => ?_, fun q hq => hp.2.2 q (List.mem_cons_of_mem _ hq)⟩⟩
  · rw [h.dpairs_gd, List.mem_map] at hy
    obtain ⟨y0, hy0, rfl⟩ := hy
    obtain ⟨⟨p1, p2, p3⟩, ⟨q1, q2, q3⟩⟩ := h.pair_ports ks y0 hy0
    simp only [portsP]
    rw [hsub _ _ p1 p2 p3, hsub _ _ q1 q2 q3]
    exact hp.1 y0 hy0
  · obtain ⟨p, j⟩ := q
    rw [h.bnd_gd] at hq
    obtain ⟨hj, e⟩ := (mem_zipIdx' _ _ _).1 hq
    rw [List.length_map] at hj
    rw [List.getElem_map] at e
    subst e
    by_cases hji : j = i
    · subst hji
      simp only [hci, show substG a c g c = g from if_pos (Or.inr rfl), labG_self]
    · have hpc : L.base.bnd[j] ≠ c := fun e' =>
        hji ((List.Nodup.getElem_inj_iff h.bnd).1 (e'.trans hci.symm))
      have hpa : L.base.bnd[j] ≠ a := fun e' => h.ab (e' ▸ List.getElem_mem hj)
      have hpg : L.base.bnd[j] ≠ g := fun e' => h.gb (e' ▸ List.getElem_mem hj)
      simp only at ht ⊢
      rw [h.substG_of hpa hpc] at ht ⊢
      rw [labG_ne hpg]
      exact hp.2.1 _ ((mem_zipIdx' _ _ _).2 ⟨hj, rfl⟩) t (by rw [h.lc, ← h.lc']; exact ht)
  · intro q hq
    rcases List.mem_cons.1 hq with rfl | hq
    · exact hd
    · exact hp.2.2 q hq
  · have := hp.1 _ (by rw [h.dpairs_gd]; exact List.mem_map.2 ⟨y, hy, rfl⟩)
    obtain ⟨⟨p1, p2, p3⟩, ⟨q1, q2, q3⟩⟩ := h.pair_ports ks y hy
    simp only [portsP] at this
    rwa [hsub _ _ p1 p2 p3, hsub _ _ q1 q2 q3] at this
  · obtain ⟨p, j⟩ := q
    obtain ⟨hj, e⟩ := (mem_zipIdx' _ _ _).1 hq
    subst e
    by_cases hji : j = i
    · subst hji
      simp only [hci] at ht ⊢
      rw [h.lc, h.cnode, h.xchain] at ht
      simp only [cidx, List.mem_singleton] at ht
      subst ht
      rw [labG_none (fun q hq => (h.gsf q hq).2.1), hc]
    · have hpc : L.base.bnd[j] ≠ c := fun e' =>
        hji ((List.Nodup.getElem_inj_iff h.bnd).1 (e'.trans hci.symm))
      have hpa : L.base.bnd[j] ≠ a := fun e' => h.ab (e' ▸ List.getElem_mem hj)
      have hpg : L.base.bnd[j] ≠ g := fun e' => h.gb (e' ▸ List.getElem_mem hj)
      have hq' : (substG a c g L.base.bnd[j], j) ∈ (Lg).base.bnd.zipIdx := by
        rw [h.bnd_gd]
        exact (mem_zipIdx' _ _ _).2 ⟨by simpa using hj, by simp⟩
      have := hp.2.1 _ hq' t (by rw [h.lc', ← h.lc, h.substG_of hpa hpc]; exact ht)
      simp only at this ⊢
      rwa [h.substG_of hpa hpc, labG_ne hpg] at this

theorem GhostOK.integ_reads (h : GhostOK L gs x a c g i) (E : Env) (ks : ℕ → ℕ) (ν ν' : ℕ → ℕ)
    (hag : ∀ l ∉ gdigs x, ν l = ν' l) :
    (Lg).integG ((g, i, x.dim) :: gs) E ks ν = (Lg).integG ((g, i, x.dim) :: gs) E ks ν' := by
  have hl : ∀ p t, p ≠ a → p ≠ c → ∀ gs' : Ghosts, ν (labG gs' p t) = ν' (labG gs' p t) := by
    intro p t ha hc gs'
    unfold labG
    split
    · exact hag _ (slab_not_gdigs x _ _)
    · exact hag _ (h.not_gdigs' p ha hc t)
  have hs : ∀ j t, ν (slab j t) = ν' (slab j t) := fun j t => hag _ (slab_not_gdigs x j t)
  have hne : ∀ p, substG a c g p ≠ a ∧ substG a c g p ≠ c := by
    intro p
    unfold substG
    split_ifs with hp
    · exact ⟨h.ga, h.gc⟩
    · exact ⟨fun e => hp (Or.inl e), fun e => hp (Or.inr e)⟩
  have hpins : (Lg).pinsG ((g, i, x.dim) :: gs) E ks ν ↔
      (Lg).pinsG ((g, i, x.dim) :: gs) E ks ν' := by
    unfold pinsG
    refine and_congr (forall₂_congr fun y hy => ?_) (and_congr (forall₂_congr fun q hq =>
      forall₂_congr fun t _ => ?_) (forall₂_congr fun q _ => by rw [hs]))
    · rw [h.dpairs_gd, List.mem_map] at hy
      obtain ⟨y0, -, rfl⟩ := hy
      simp only [portsP]
      rw [hl _ _ (hne _).1 (hne _).2, hl _ _ (hne _).1 (hne _).2]
    · have hq' : q.1 ∈ (Lg).base.bnd := (List.mem_zipIdx hq).2.2 ▸ List.getElem_mem _
      rw [h.bnd_gd, List.mem_map] at hq'
      obtain ⟨q0, -, e⟩ := hq'
      rw [← e, hl _ _ (hne _).1 (hne _).2, hs]
  have hprod : (Lg).prodL E ks ν = (Lg).prodL E ks ν' := by
    have hcph' : (Lg).cph = [] := h.cph
    unfold prodL
    refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [(Lg).dnodeL_plain hcph', (Lg).base.dnode_notail h.ports h.wch']
    have hn' : n.id ≠ x.id := by
      have hn'' : n ∈ L.base.nodes.filter fun n => n.id ≠ x.id := hn
      rw [List.mem_filter, decide_eq_true_eq] at hn''
      exact hn''.2
    apply val_congr
    · simp only [List.map_map]
      exact List.map_congr_left fun j _ => hag _ (h.not_gdigs _ hn' _)
    · simp only [List.map_map]
      exact List.map_congr_left fun j _ => hag _ (h.not_gdigs _ hn' _)
  unfold integG
  rw [hprod]
  classical
  exact if_congr hpins rfl rfl

theorem GhostOK.integ_point (h : GhostOK L gs x a c g i) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.integG gs E ks ν =
      if ∀ l ∈ gdigs x, ν l = ν (slab i []) then (Lg).integG ((g, i, x.dim) :: gs) E ks ν
      else 0 := by
  have hdx : ∀ b, dlab ⟨x.id, b, 0⟩ [] ∈ gdigs x := fun b => by cases b <;> simp [gdigs]
  have hax : dlab a [] ∈ gdigs x := by rcases h.ac with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> exact hdx _
  have hcx : dlab c [] ∈ gdigs x := by rcases h.ac with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> exact hdx _
  unfold integG
  by_cases hA : ∀ l ∈ gdigs x, ν l = ν (slab i [])
  · rw [if_pos hA]
    by_cases hd : ν (slab i []) < E.dimv x.dim
    · have hone : tZ (E.dimv x.dim) (E.phase x.ph)
          [ν (dlab ⟨x.id, true, 0⟩ []), ν (dlab ⟨x.id, false, 0⟩ [])] = 1 := by
        rw [tZ_id_pair, if_pos ⟨by rw [hA _ (hdx _), hA _ (hdx _)], by rw [hA _ (hdx _)]; exact hd⟩,
          hE]
        simp [ex]
      rw [h.prodL_eq, hone, mul_one]
      classical
      exact if_congr (h.pins_iff E ks ν (hA _ hax) (hA _ hcx) hd) rfl rfl
    · have h1 : ¬ (Lg).pinsG ((g, i, x.dim) :: gs) E ks ν := fun hp =>
        hd (hp.2.2 _ List.mem_cons_self)
      rw [if_neg h1]
      split_ifs with hp
      · rw [h.prodL_eq]
        refine mul_eq_zero_of_right _ ?_
        rw [tZ_id_pair, if_neg]
        rintro ⟨-, hlt⟩
        exact hd (hA _ (hdx true) ▸ hlt)
      · rfl
  · rw [if_neg hA]
    split_ifs with hp
    · rw [h.prodL_eq]
      refine mul_eq_zero_of_right _ ?_
      rw [tZ_id_pair, if_neg]
      rintro ⟨heq, -⟩
      have hc' := h.c_pin E ks ν hp
      apply hA
      intro l hl
      simp only [gdigs, List.mem_cons, List.mem_nil_iff, or_false] at hl
      have hcv : ∀ b, (⟨x.id, b, 0⟩ : Port) = c → ν (dlab ⟨x.id, b, 0⟩ []) = ν (slab i []) :=
        fun b e => e ▸ hc'
      rcases h.ac with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> rcases hl with rfl | rfl
      · rw [heq]; exact hc'
      · exact hc'
      · exact hc'
      · rw [← heq]; exact hc'
    · rfl

/-- Dropping `x` for the ghost `(g, i, x.dim)` keeps the digit sum. -/
theorem GhostOK.evalG_ghost (h : GhostOK L gs x a c g i) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalG gs E ks σ = (Lg).evalG ((g, i, x.dim) :: gs) E ks σ := by
  have hX := gdigs_nodup x
  have hdimX : ∀ l ∈ gdigs x, E.dimv (L.base.portDim (dport l)) = E.dimv x.dim := by
    intro l hl
    simp only [gdigs, List.mem_cons, List.mem_nil_iff, or_false] at hl
    rcases hl with rfl | rfl <;> rw [dport_dlab, h.portDim_x]
  have hdims : ∀ l ∈ (Lg).base.dlabs ks, E.dimv (L.base.portDim (dport l)) =
      E.dimv ((Lg).base.portDim (dport l)) := by
    intro l hl
    obtain ⟨p, hp, t, -, rfl⟩ := (mem_dlabs _ ks l).1 hl
    rw [dport_dlab, h.portDim_gd p (h.node_ne p hp)]
  unfold evalG
  rw [h.dlabs_eq ks, sumOver_append, sumOver_dim_congr _ _ _ hdims]
  refine sumOver_congr _ _ _ _ _ fun τ hτ hr => ?_
  have hs : τ (slab i []) = σ i [] := by
    rw [hτ _ fun hm => by
      obtain ⟨p, -, t, -, e⟩ := (mem_dlabs _ ks _).1 hm
      exact dlab_ne_slab p t i [] e.symm, slotsD_slab]
  rw [sumOver_congr _ _ _ (fun ν => if ∀ l ∈ gdigs x, ν l = (fun _ => τ (slab i [])) l then
      (Lg).integG ((g, i, x.dim) :: gs) E ks τ else 0) _ fun ν hν hνr => by
    rw [h.integ_point E hE ks ν, hν _ (slab_not_gdigs x i [])]
    exact if_congr Iff.rfl (h.integ_reads E ks ν τ hν) rfl]
  by_cases hd : τ (slab i []) < E.dimv x.dim
  · rw [sumOver_pinV _ hX _ fun l hl => by rw [hdimX l hl]; exact hd]
  · have h0 : (Lg).integG ((g, i, x.dim) :: gs) E ks τ = 0 := by
      unfold integG
      rw [if_neg fun hp => hd (hp.2.2 _ List.mem_cons_self)]
    rw [h0]
    simp only [ite_self]
    exact sumOver_zero _ _ _

end ghost

theorem pinsG_wn (L : LDiagram) (ns : List DNode) : (L.withNodes ns).pinsG = L.pinsG := by
  funext gs E ks ν
  unfold pinsG dpairs linkPairs
  rw [legChain_wn, depth_wn]
  rfl

/-- A permutation of the nodes keeps the digit sum with ghosts. -/
theorem evalG_withNodes (L : LDiagram) (ns : List DNode) (hp : ns.Perm L.base.nodes)
    (hid : (L.base.nodes.map (·.id)).Nodup) (gs : Ghosts) (E : Env) (ks : ℕ → ℕ)
    (σ : ℕ → List ℕ → ℕ) : (L.withNodes ns).evalG gs E ks σ = L.evalG gs E ks σ := by
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
  unfold evalG
  rw [sumOver_perm _ hl, hd]
  congr 1
  funext ν
  unfold integG
  rw [hP ν, pinsG_wn]

/-- `x` moved last, then dropped for the ghost `g`. -/
def ghostAt (L : LDiagram) (x : DNode) (a c g : Port) : LDiagram :=
  (L.moveLast x.id).ghostDrop x.id a c g

/-- Dropping `x` for a ghost after moving it last keeps the digit sum. -/
theorem evalG_ghostAt {L : LDiagram} {gs : Ghosts} {x : DNode} {a c g : Port} {i : ℕ}
    (h : GhostOK (L.moveLast x.id) gs x a c g i) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalG gs E ks σ = (L.ghostAt x a c g).evalG ((g, i, x.dim) :: gs) E ks σ := by
  have hp := moveLast_perm L x.id
  have hid : (L.base.nodes.map (·.id)).Nodup := (hp.map _).nodup_iff.1 h.ids
  rw [← evalG_withNodes L _ hp hid]
  exact h.evalG_ghost E hE ks σ

end LDiagram

end ArchytasZX
