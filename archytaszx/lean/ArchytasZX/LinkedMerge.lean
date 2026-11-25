/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedPair

/-!
# Aligned boxes merged into one box

`MergeOK L L' ps`: `L'` has the nodes, boundary, leg chains, tails, copy-indexed phases and leg
dimensions (each read at its box's depth) of `L`, no links on either side, and the wires of `L`
together with every aligned wire of `L` reading copy `t` at copy `t` (`idAl`) as a wire; every
other aligned wire is kept; each pair `ps'` is the pair of `ps` at its place over a box of
one copy count.
`evalP_merge`: then `L` and `L'` have equal `evalP` digit sums. `SameSlots sd sd'`: one output
and input count, and every slot's anchors and box chain alike. `evalP_merge_flat`: under both,
the digit sums read at the flat positions of the two sides' slot data agree.
-/

namespace ArchytasZX

theorem pick_range (t : List ℕ) : pick t (List.range t.length) = t := by
  apply List.ext_getElem
  · simp [pick]
  · intro i h1 h2
    simp [pick, List.getElem?_eq_getElem h2]

theorem length_of_mem_cidx {ks : ℕ → ℕ} {c t : List ℕ} (h : t ∈ cidx ks c) :
    t.length = c.length :=
  ((mem_cidx ks c t).1 h).length_eq

namespace LDiagram

/-- An aligned wire reading copy `t` of one end at copy `t` of the other. -/
def idAl (L : LDiagram) (a : DAlign) : Bool :=
  a.perm == List.range (L.base.legChain a.a).length &&
    L.base.legChain a.a == L.base.legChain a.b

/-- The copy-indexed phase entry of the node `x`, its box read as its depth. -/
def cphAt (L : LDiagram) (x : ℕ) : Option (ℕ × ℕ × ℕ) :=
  (L.cph.find? fun q => decide (q.1 = x)).map fun q => (L.depth q.2.1 x, q.2.2)

/-- The copy-indexed dimension entry of the leg `p`, its box read as its depth. -/
def cdimAt (L : LDiagram) (p : Port) : Option (ℕ × ℕ × ℕ) :=
  (L.cdim.find? fun q => decide (q.1 = p)).map fun q => (L.depth q.2.1 p.node, q.2.2)

/-- The legs whose chains the digit sums read. -/
def readLegs (L : LDiagram) : List Port :=
  L.base.legs ++ L.base.bnd ++ L.base.wires.map (·.1) ++ L.aligned.map (·.a)

theorem mem_readLegs_wire (L : LDiagram) {w : Port × Port} (hw : w ∈ L.base.wires) :
    w.1 ∈ L.readLegs := by
  simp only [readLegs, List.mem_append, List.mem_map]
  exact Or.inl (Or.inr ⟨w, hw, rfl⟩)

theorem mem_readLegs_al (L : LDiagram) {a : DAlign} (ha : a ∈ L.aligned) : a.a ∈ L.readLegs := by
  simp only [readLegs, List.mem_append, List.mem_map]
  exact Or.inr ⟨a, ha, rfl⟩

theorem mem_readLegs_leg (L : LDiagram) {p : Port} (hp : p ∈ L.base.legs) : p ∈ L.readLegs := by
  simp only [readLegs, List.mem_append]
  exact Or.inl (Or.inl (Or.inl hp))

theorem mem_readLegs_bnd (L : LDiagram) {p : Port} (hp : p ∈ L.base.bnd) : p ∈ L.readLegs := by
  simp only [readLegs, List.mem_append]
  exact Or.inl (Or.inl (Or.inr hp))

/-- `L'` is `L` with aligned boxes merged: see the module docstring. -/
structure MergeOK (L L' : LDiagram) (ps ps' : List LPair) : Prop where
  nodes : L'.base.nodes = L.base.nodes
  outs : L'.base.outs = L.base.outs
  ins : L'.base.ins = L.base.ins
  chain : ∀ n ∈ L.base.nodes, L'.base.nodeChain n.id = L.base.nodeChain n.id
  tail : ∀ p ∈ L.base.legs, L'.base.tail p = L.base.tail p
  lc : ∀ p ∈ L.readLegs, L'.base.legChain p = L.base.legChain p
  cph : ∀ n ∈ L.base.nodes, L'.cphAt n.id = L.cphAt n.id
  cdim : ∀ p ∈ L.base.legs, L'.cdimAt p = L.cdimAt p
  links : L.links = []
  links' : L'.links = []
  wires : ∀ w ∈ L.base.wires, w ∈ L'.base.wires
  al : ∀ a ∈ L.aligned, a ∈ L'.aligned ∨ (L.idAl a = true ∧ (a.a, a.b) ∈ L'.base.wires)
  wires' : ∀ w ∈ L'.base.wires,
    w ∈ L.base.wires ∨ ∃ a ∈ L.aligned, L.idAl a = true ∧ (a.a, a.b) = w
  al' : ∀ a ∈ L'.aligned, a ∈ L.aligned
  len : ps'.length = ps.length
  pairs : ∀ q ∈ ps.zip ps', q.2 = { q.1 with box := q.2.box } ∧
    L'.multOf q.2.box = L.multOf q.1.box

theorem phAt_cphAt (L : LDiagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    L.phAt ks n t = match L.cphAt n.id with
      | some q => Nat.pair n.ph (t.getD q.1 0 + q.2.1 * ks q.2.2)
      | none => n.ph := by
  unfold phAt cphAt
  cases L.cph.find? fun q => decide (q.1 = n.id) <;> rfl

theorem dimAt_cdimAt (L : LDiagram) (ks : ℕ → ℕ) (p : Port) (t : List ℕ) :
    L.dimAt ks p t = match L.cdimAt p with
      | some q => Nat.pair (L.base.portDim p) (t.getD q.1 0 + q.2.1 * ks q.2.2)
      | none => L.base.portDim p := by
  unfold dimAt cdimAt
  cases L.cdim.find? fun q => decide (q.1 = p) <;> rfl

theorem mem_dpairsX_nil (L : LDiagram) (hl : L.links = []) (ks : ℕ → ℕ) (x : Dig × Dig) :
    x ∈ L.dpairsX ks ↔
      (∃ w ∈ L.base.wires, ∃ t ∈ cidx ks (L.base.legChain w.1), x = ((w.1, t), (w.2, t))) ∨
        ∃ a ∈ L.aligned, ∃ t ∈ cidx ks (L.base.legChain a.a),
          x = ((a.a, t), (a.b, pick t a.perm)) := by
  simp only [dpairsX, dpairs, hl, List.flatMap_nil, List.append_nil, List.mem_append,
    List.mem_flatMap, List.mem_map, alignPairs]
  constructor
  · rintro (⟨w, hw, t, ht, rfl⟩ | ⟨a, ha, t, ht, rfl⟩)
    · exact Or.inl ⟨w, hw, t, ht, rfl⟩
    · exact Or.inr ⟨a, ha, t, ht, rfl⟩
  · rintro (⟨w, hw, t, ht, rfl⟩ | ⟨a, ha, t, ht, rfl⟩)
    · exact Or.inl ⟨w, hw, t, ht, rfl⟩
    · exact Or.inr ⟨a, ha, t, ht, rfl⟩

section merge

variable {L L' : LDiagram} {ps ps' : List LPair}

theorem idAl_pick {a : DAlign} (hi : L.idAl a = true)
    {ks : ℕ → ℕ} {t : List ℕ} (ht : t ∈ cidx ks (L.base.legChain a.a)) : pick t a.perm = t := by
  simp only [idAl, Bool.and_eq_true, beq_iff_eq] at hi
  rw [hi.1, ← length_of_mem_cidx ht, pick_range]

theorem MergeOK.dpairs_sub (h : MergeOK L L' ps ps') (ks : ℕ → ℕ) (x : Dig × Dig)
    (hx : x ∈ L.dpairsX ks) : x ∈ L'.dpairsX ks := by
  rw [mem_dpairsX_nil L h.links] at hx
  rw [mem_dpairsX_nil L' h.links']
  rcases hx with ⟨w, hw, t, ht, rfl⟩ | ⟨a, ha, t, ht, rfl⟩
  · have e := h.lc w.1 (L.mem_readLegs_wire hw)
    exact Or.inl ⟨w, h.wires w hw, t, e ▸ ht, rfl⟩
  · have e := h.lc a.a (L.mem_readLegs_al ha)
    rcases h.al a ha with ha' | ⟨hi, hw⟩
    · exact Or.inr ⟨a, ha', t, e ▸ ht, rfl⟩
    · rw [idAl_pick hi ht]
      exact Or.inl ⟨(a.a, a.b), hw, t, e ▸ ht, rfl⟩

theorem MergeOK.dpairs_sup (h : MergeOK L L' ps ps') (ks : ℕ → ℕ) (x : Dig × Dig)
    (hx : x ∈ L'.dpairsX ks) : x ∈ L.dpairsX ks := by
  rw [mem_dpairsX_nil L' h.links'] at hx
  rw [mem_dpairsX_nil L h.links]
  rcases hx with ⟨w, hw, t, ht, rfl⟩ | ⟨a, ha, t, ht, rfl⟩
  · rcases h.wires' w hw with hw' | ⟨a, ha, hi, rfl⟩
    · have e := h.lc w.1 (L.mem_readLegs_wire hw')
      exact Or.inl ⟨w, hw', t, e ▸ ht, rfl⟩
    · have e := h.lc a.a (L.mem_readLegs_al ha)
      rw [e] at ht
      refine Or.inr ⟨a, ha, t, ht, ?_⟩
      rw [idAl_pick hi ht]
  · have ha' := h.al' a ha
    have e := h.lc a.a (L.mem_readLegs_al ha')
    exact Or.inr ⟨a, ha', t, e ▸ ht, rfl⟩

theorem MergeOK.legs (h : MergeOK L L' ps ps') : L'.base.legs = L.base.legs := by
  simp only [Diagram.legs, h.nodes]

theorem MergeOK.bnd (h : MergeOK L L' ps ps') : L'.base.bnd = L.base.bnd := by
  simp only [Diagram.bnd, h.outs, h.ins]

theorem MergeOK.dlabs (h : MergeOK L L' ps ps') (ks : ℕ → ℕ) :
    L'.base.dlabs ks = L.base.dlabs ks := by
  unfold Diagram.dlabs
  rw [h.legs]
  exact List.flatMap_congr fun p hp => by
    rw [h.lc p (L.mem_readLegs_leg hp)]

theorem MergeOK.pins (h : MergeOK L L' ps ps') (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L'.pinsX ks ν ↔ L.pinsX ks ν := by
  unfold pinsX
  rw [h.bnd]
  refine and_congr ⟨fun H x hx => H x (h.dpairs_sub ks x hx), fun H x hx =>
    H x (h.dpairs_sup ks x hx)⟩ (forall₂_congr fun q hq => ?_)
  obtain ⟨p, i⟩ := q
  obtain ⟨hi, e⟩ := (mem_zipIdx' _ _ _).1 hq
  have hm : p ∈ L.readLegs := L.mem_readLegs_bnd (e ▸ List.getElem_mem hi)
  rw [h.lc p hm]

theorem MergeOK.mem_legs (n : DNode) (hn : n ∈ L.base.nodes) (o : Bool) (i : ℕ)
    (hi : i < (if o then n.outs.length else n.ins.length)) : ⟨n.id, o, i⟩ ∈ L.base.legs := by
  simp only [Diagram.legs, List.mem_flatMap]
  refine ⟨n, hn, ?_⟩
  cases o <;> simp_all [DNode.legs]

theorem MergeOK.dnode (h : MergeOK L L' ps ps') (ks : ℕ → ℕ) (n : DNode) (hn : n ∈ L.base.nodes)
    (t : List ℕ) : L'.base.dnode ks n t = L.base.dnode ks n t := by
  unfold Diagram.dnode
  simp only []
  have ho : ∀ i ∈ List.range n.outs.length, L'.base.tail ⟨n.id, true, i⟩ =
      L.base.tail ⟨n.id, true, i⟩ := fun i hi =>
    h.tail _ (MergeOK.mem_legs n hn true i (by simpa using hi))
  have hi : ∀ i ∈ List.range n.ins.length, L'.base.tail ⟨n.id, false, i⟩ =
      L.base.tail ⟨n.id, false, i⟩ := fun i hi =>
    h.tail _ (MergeOK.mem_legs n hn false i (by simpa using hi))
  congr 1
  · exact List.flatMap_congr fun i hi' => by simp only [ho i hi']
  · exact List.flatMap_congr fun i hi' => by simp only [hi i hi']

theorem MergeOK.prodL (h : MergeOK L L' ps ps') (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L'.prodL E ks ν = L.prodL E ks ν := by
  unfold LDiagram.prodL
  rw [h.nodes]
  refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
  rw [h.chain n hn]
  refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
  unfold dnodeL
  rw [h.dnode ks n hn t, phAt_cphAt, phAt_cphAt, h.cph n hn]

theorem pairF_zip (L L' : LDiagram) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    ∀ (ps ps' : List LPair), ps'.length = ps.length →
      (∀ q ∈ ps.zip ps', q.2 = { q.1 with box := q.2.box } ∧
        L'.multOf q.2.box = L.multOf q.1.box) →
      L'.pairF ps' E ks ν = L.pairF ps E ks ν
  | [], [], _, _ => rfl
  | p :: ps, p' :: ps', hl, h => by
    have hp := h (p, p') List.mem_cons_self
    have ih := pairF_zip L L' E ks ν ps ps' (by simpa using hl)
      fun q hq => h q (List.mem_cons_of_mem _ hq)
    unfold LDiagram.pairF at ih ⊢
    simp only [List.map_cons, List.prod_cons]
    obtain ⟨h1, h2⟩ := hp
    simp only at h1 h2
    have e : p'.val E ks (ks (L.multOf p.box)) ν = p.val E ks (ks (L.multOf p.box)) ν := by
      rw [h1]; rfl
    rw [ih, h2, e]
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

theorem MergeOK.pairF (h : MergeOK L L' ps ps') (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L'.pairF ps' E ks ν = L.pairF ps E ks ν :=
  pairF_zip L L' E ks ν ps ps' h.len h.pairs

theorem MergeOK.portDim (h : MergeOK L L' ps ps') (p : Port) :
    L'.base.portDim p = L.base.portDim p := by
  simp only [Diagram.portDim, h.nodes]

/-- Merged aligned boxes keep the pair digit sum. -/
theorem evalP_merge (h : MergeOK L L' ps ps') (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L'.evalP ps' E ks σ = L.evalP ps E ks σ := by
  unfold evalP evalW
  rw [h.dlabs]
  rw [sumOver_dim_congr _ (fun l => E.dimv (L.dimAt ks (dport l) (dtup l))) _ ?_]
  · congr 1
    funext ν
    beta_reduce
    rw [h.prodL, h.pairF]
    classical
    exact if_congr (h.pins ks ν) rfl rfl
  · intro x hx
    obtain ⟨p, hp, t, _, rfl⟩ := (mem_dlabs _ _ _).1 hx
    rw [dport_dlab, dtup_dlab, dimAt_cdimAt, dimAt_cdimAt, h.cdim p hp, h.portDim]

end merge

/-- One output and input count, and every slot's anchors and box chain alike. -/
structure SameSlots (sd sd' : SlotData) : Prop where
  no : sd'.1 = sd.1
  ni : sd'.2.1 = sd.2.1
  per : ∀ i < sd.1 + sd.2.1, anch sd' i = anch sd i ∧ chainS sd' i = chainS sd i

theorem SameSlots.slots {sd sd' : SlotData} (h : SameSlots sd sd') (ks : ℕ → ℕ) :
    slots sd' ks = slots sd ks := by
  unfold LDiagram.slots
  rw [h.no, h.ni]
  exact List.flatMap_congr fun i hi => by rw [(h.per i (List.mem_range.1 hi)).2]

theorem SameSlots.fpos {sd sd' : SlotData} (h : SameSlots sd sd') (ks : ℕ → ℕ) (i : ℕ)
    (hi : i < sd.1 + sd.2.1) (t : List ℕ) : fpos sd' ks i t = fpos sd ks i t := by
  unfold LDiagram.fpos
  rw [h.slots]
  have hk : ∀ j < sd.1 + sd.2.1, ∀ u, key sd' j u = key sd j u := fun j hj u => by
    unfold key; rw [(h.per j hj).1]
  rw [hk i hi t]
  congr 1
  refine List.filter_congr fun v hv => ?_
  have hv' := (mem_slots sd ks v.1 v.2).1 hv
  rw [hk v.1 hv'.1 v.2]

/-- `evalP_merge` read at the flat positions of the two sides' slot data. -/
theorem evalP_merge_flat {L L' : LDiagram} {ps ps' : List LPair} {G G' : List (ℕ × List (List ℕ))}
    (h : MergeOK L L' ps ps') (hs : SameSlots (L.slotData G) (L'.slotData G')) (E : Env)
    (ks : ℕ → ℕ) (y : ℕ → ℕ) :
    L'.evalP ps' E ks (flatAt (L'.slotData G') ks y) =
      L.evalP ps E ks (flatAt (L.slotData G) ks y) := by
  rw [evalP_merge h]
  refine evalP_congr L ps E ks _ _ fun i hi t _ => ?_
  have hi' : i < (L.slotData G).1 + (L.slotData G).2.1 := by
    simpa [slotData, Diagram.bnd] using hi
  simp only [flatAt, hs.fpos ks i hi' t]

end LDiagram

end ArchytasZX
