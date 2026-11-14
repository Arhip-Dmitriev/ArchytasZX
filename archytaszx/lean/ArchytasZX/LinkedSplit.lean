/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedPeel

/-!
# Splitting an outermost box in two

`evalL_split`: for an outermost node box `bid` over the nodes `B` at `ks c₁ + ks c₂` copies
(multiplicity index `mb`) and a linked diagram `L'` holding `B` at `ks c₁` copies and a copy of `B`
(the nodes `cp x`, outermost box `bid₂`) at `ks c₂` copies (`BoxSplitOK`), `L'` is `L` with `B`'s
copy `j + ks c₁` read at the copy's copy `j`. A copy-indexed phase of `B` reads the copy index of
`bid`; on the copy it reads the copy index of `bid₂` plus `ks c₁`. `bid` has no links; links of
boxes inside it hold once on `B` and once on the copy.
-/

namespace ArchytasZX

/-- A copy index with its first coordinate raised by `K`. -/
def addH (K : ℕ) : List ℕ → List ℕ
  | [] => []
  | j :: u => (j + K) :: u

theorem cidx_add (ks : ℕ → ℕ) (mb c₁ c₂ : ℕ) (r : List ℕ) (hk : ks mb = ks c₁ + ks c₂) :
    cidx ks (mb :: r) = cidx ks (c₁ :: r) ++ (cidx ks (c₂ :: r)).map (addH (ks c₁)) := by
  simp only [cidx, hk, List.range_add, List.flatMap_append, List.flatMap_map, List.map_flatMap,
    List.map_map]
  congr 1
  refine List.flatMap_congr fun i _ => List.map_congr_left fun u _ => ?_
  simp [addH, Nat.add_comm]

theorem find?_congr' {α : Type*} {l : List α} {p q : α → Bool} (h : ∀ x ∈ l, p x = q x) :
    l.find? p = l.find? q := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.find?_cons, h a List.mem_cons_self,
      ih fun x hx => h x (List.mem_cons_of_mem _ hx)]

namespace LDiagram

/-- Both digits of a pair with their first coordinate raised by `K`. -/
def shiftS (K : ℕ) (y : Dig × Dig) : Dig × Dig :=
  ((y.1.1, addH K y.1.2), (y.2.1, addH K y.2.2))

/-- The pairs of a link inside the outermost box: those of the first `ks c₁` copies, and those
of the other `ks c₂`, raised by `ks c₁`. -/
theorem mem_lp_split2 (ks : ℕ → ℕ) (d k : ℕ) (mb c₁ c₂ : ℕ) (r : List ℕ) (l : DLink)
    (hk : ks mb = ks c₁ + ks c₂) (x : Dig × Dig) :
    x ∈ lp ks (d + 1) k (mb :: r) l ↔
      x ∈ lp ks (d + 1) k (c₁ :: r) l ∨ ∃ y ∈ lp ks (d + 1) k (c₂ :: r) l, x = shiftS (ks c₁) y := by
  rw [mem_lp, mem_lp]
  simp only [List.eraseIdx_cons_succ, mem_cidx_cons, List.getD_cons_succ, List.set_cons_succ]
  have hy := fun y => (mem_lp ks (d + 1) k (c₂ :: r) l y).2
  simp only [List.eraseIdx_cons_succ, mem_cidx_cons, List.getD_cons_succ,
    List.set_cons_succ] at hy
  constructor
  · rintro (⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ |
      ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨h1, _, ⟨j, u, rfl, hj, hu⟩, rfl⟩)
    · by_cases hj1 : j < ks c₁
      · exact Or.inl (Or.inl ⟨_, ⟨j, u, rfl, hj1, hu⟩, h1, rfl⟩)
      · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
        exact Or.inr ⟨_, hy _ (Or.inl ⟨_, ⟨j', u, rfl, by omega, hu⟩, h1, rfl⟩),
          by simp [shiftS, addH]⟩
    · by_cases hj1 : j < ks c₁
      · exact Or.inl (Or.inr (Or.inl ⟨_, ⟨j, u, rfl, hj1, hu⟩, h1, rfl⟩))
      · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
        exact Or.inr ⟨_, hy _ (Or.inr (Or.inl ⟨_, ⟨j', u, rfl, by omega, hu⟩, h1, rfl⟩)),
          by simp [shiftS, addH]⟩
    · by_cases hj1 : j < ks c₁
      · exact Or.inl (Or.inr (Or.inr (Or.inl ⟨_, ⟨j, u, rfl, hj1, hu⟩, h1, rfl⟩)))
      · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
        exact Or.inr ⟨_, hy _ (Or.inr (Or.inr (Or.inl ⟨_, ⟨j', u, rfl, by omega, hu⟩, h1, rfl⟩))),
          by simp [shiftS, addH]⟩
    · by_cases hj1 : j < ks c₁
      · exact Or.inl (Or.inr (Or.inr (Or.inr ⟨h1, _, ⟨j, u, rfl, hj1, hu⟩, rfl⟩)))
      · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
        exact Or.inr ⟨_, hy _ (Or.inr (Or.inr (Or.inr ⟨h1, _, ⟨j', u, rfl, by omega, hu⟩, rfl⟩))),
          by simp [shiftS, addH]⟩
  · rintro ((⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ |
      ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨h1, _, ⟨j, u, rfl, hj, hu⟩, rfl⟩) | ⟨y, hy', rfl⟩)
    · exact Or.inl ⟨_, ⟨j, u, rfl, by omega, hu⟩, h1, rfl⟩
    · exact Or.inr (Or.inl ⟨_, ⟨j, u, rfl, by omega, hu⟩, h1, rfl⟩)
    · exact Or.inr (Or.inr (Or.inl ⟨_, ⟨j, u, rfl, by omega, hu⟩, h1, rfl⟩))
    · exact Or.inr (Or.inr (Or.inr ⟨h1, _, ⟨j, u, rfl, by omega, hu⟩, rfl⟩))
    · rw [mem_lp] at hy'
      simp only [List.eraseIdx_cons_succ, mem_cidx_cons, List.getD_cons_succ,
        List.set_cons_succ] at hy'
      rcases hy' with ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ |
        ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨h1, _, ⟨j, u, rfl, hj, hu⟩, rfl⟩
      · exact Or.inl ⟨_, ⟨j + ks c₁, u, rfl, by omega, hu⟩, by simpa using h1,
          by simp [shiftS, addH]⟩
      · exact Or.inr (Or.inl ⟨_, ⟨j + ks c₁, u, rfl, by omega, hu⟩, by simpa using h1,
          by simp [shiftS, addH]⟩)
      · exact Or.inr (Or.inr (Or.inl ⟨_, ⟨j + ks c₁, u, rfl, by omega, hu⟩, by simpa using h1,
          by simp [shiftS, addH]⟩))
      · exact Or.inr (Or.inr (Or.inr ⟨h1, _, ⟨j + ks c₁, u, rfl, by omega, hu⟩,
          by simp [shiftS, addH]⟩))

/-- The split's map of digits: a leg on `B` at a copy `j ≥ K` of the outermost box goes to its
copy at `j - K`. -/
def splitφ (B : List ℕ) (cp : ℕ → ℕ) (K : ℕ) (p : Port) : List ℕ → Dig
  | [] => (p, [])
  | j :: u => if p.node ∈ B ∧ K ≤ j then (cpP cp p, (j - K) :: u) else (p, j :: u)

/-- `L'` is `L` with the outermost node box `bid` (multiplicity index `mb`, nodes `B`) left at the
index `c₁` and copied into the nodes `cp x` under the outermost box `bid₂` at the index `c₂`. -/
structure BoxSplitOK (L L' : LDiagram) (bid bid₂ mb c₁ c₂ : ℕ) (B : List ℕ) (cp : ℕ → ℕ) : Prop where
  ports : ∀ y ∈ L.base.boxes, y.ports = []
  ports' : ∀ y ∈ L'.base.boxes, y.ports = []
  wch : ∀ w ∈ L.base.wires, L.base.nodeChain w.1.node = L.base.nodeChain w.2.node
  wch' : ∀ w ∈ L'.base.wires, L'.base.nodeChain w.1.node = L'.base.nodeChain w.2.node
  cph : ∀ q ∈ L.cph, (∃ n ∈ L.base.nodes, n.id = q.1) ∧ q.1 ∈ B ∧ q.2.1 = bid ∧ q.2.2.1 = 0
  cph' : L'.cph = L.cph ++ L.cph.map fun q => (cp q.1, bid₂, 1, c₁)
  nodes : L'.base.nodes = L.base.nodes ++ (L.base.nodes.filter fun n => n.id ∈ B).map (cpNode cp)
  ids : (L'.base.nodes.map (·.id)).Nodup
  legs : L'.base.legs.Nodup
  bnd : L'.base.bnd.Nodup
  chB : ∀ n ∈ L.base.nodes, n.id ∈ B →
    L.base.nodeChain n.id = mb :: (L.base.nodeChain n.id).tail ∧
      L'.base.nodeChain n.id = c₁ :: (L.base.nodeChain n.id).tail ∧
      L'.base.nodeChain (cp n.id) = c₂ :: (L.base.nodeChain n.id).tail ∧
      L.depth bid n.id = 0 ∧ L'.depth bid n.id = 0 ∧ L'.depth bid₂ (cp n.id) = 0
  chO : ∀ n ∈ L.base.nodes, n.id ∉ B → L'.base.nodeChain n.id = L.base.nodeChain n.id
  wires : ∀ w ∈ L.base.wires, w.1 ∈ L.base.legs ∧ w.2 ∈ L.base.legs ∧
    (w.1.node ∈ B ↔ w.2.node ∈ B) ∧ WireIn L'.base.wires w ∧
    (w.1.node ∈ B → WireIn L'.base.wires (cpP cp w.1, cpP cp w.2))
  wires' : ∀ w ∈ L'.base.wires, w ∈ L.base.wires ∨
    ∃ w0 ∈ L.base.wires, w0.1.node ∈ B ∧ w = (cpP cp w0.1, cpP cp w0.2)
  lwf : ∀ l ∈ L.links, l.l1 ∈ L.base.legs ∧ l.l2 ∈ L.base.legs ∧ l.e1 ∈ L.base.legs ∧
    l.e2 ∈ L.base.legs ∧ L.base.nodeChain l.l2.node = L.base.nodeChain l.l1.node ∧
    L.depth l.box l.l1.node < (L.base.nodeChain l.l1.node).length ∧
    (L.base.nodeChain l.l1.node).getD (L.depth l.box l.l1.node) 0 = L.multOf l.box ∧
    L.base.nodeChain l.e1.node = (L.base.nodeChain l.l1.node).eraseIdx (L.depth l.box l.l1.node) ∧
    L.base.nodeChain l.e2.node = (L.base.nodeChain l.l1.node).eraseIdx (L.depth l.box l.l1.node)
  lin : ∀ l ∈ L.links, l.l1.node ∈ B → l.l2.node ∈ B ∧ l.e1.node ∈ B ∧
    l.e2.node ∈ B ∧ 1 ≤ L.depth l.box l.l1.node ∧
    (∃ l' ∈ L'.links, l'.l1 = l.l1 ∧ l'.l2 = l.l2 ∧ l'.e1 = l.e1 ∧ l'.e2 = l.e2 ∧
      L'.depth l'.box l.l1.node = L.depth l.box l.l1.node ∧ L'.multOf l'.box = L.multOf l.box) ∧
    ∃ l' ∈ L'.links, l'.l1 = cpP cp l.l1 ∧ l'.l2 = cpP cp l.l2 ∧ l'.e1 = cpP cp l.e1 ∧
      l'.e2 = cpP cp l.e2 ∧ L'.depth l'.box (cp l.l1.node) = L.depth l.box l.l1.node ∧
      L'.multOf l'.box = L.multOf l.box
  lout : ∀ l ∈ L.links, l.l1.node ∉ B → l.l2.node ∉ B ∧ l.e1.node ∉ B ∧
    l.e2.node ∉ B ∧ ∃ l' ∈ L'.links, l'.l1 = l.l1 ∧ l'.l2 = l.l2 ∧ l'.e1 = l.e1 ∧ l'.e2 = l.e2 ∧
      L'.depth l'.box l.l1.node = L.depth l.box l.l1.node ∧ L'.multOf l'.box = L.multOf l.box
  links' : ∀ l' ∈ L'.links,
    (∃ l ∈ L.links, l'.l1 = l.l1 ∧ l'.l2 = l.l2 ∧ l'.e1 = l.e1 ∧ l'.e2 = l.e2 ∧
      L'.depth l'.box l.l1.node = L.depth l.box l.l1.node ∧ L'.multOf l'.box = L.multOf l.box) ∨
    (∃ l ∈ L.links, l.l1.node ∈ B ∧ l'.l1 = cpP cp l.l1 ∧ l'.l2 = cpP cp l.l2 ∧
      l'.e1 = cpP cp l.e1 ∧ l'.e2 = cpP cp l.e2 ∧
      L'.depth l'.box (cp l.l1.node) = L.depth l.box l.l1.node ∧
      L'.multOf l'.box = L.multOf l.box)
  bndL : ∀ p ∈ L.base.bnd, p ∈ L.base.legs ∧ p ∈ L'.base.bnd ∧ (p.node ∈ B → cpP cp p ∈ L'.base.bnd)
  bnd' : ∀ q ∈ L'.base.bnd, q ∈ L.base.bnd ∨ ∃ p ∈ L.base.bnd, p.node ∈ B ∧ q = cpP cp p

section split

variable {L L' : LDiagram} {bid bid₂ mb c₁ c₂ : ℕ} {B : List ℕ} {cp : ℕ → ℕ}

theorem splitφ_lo (K : ℕ) (p : Port) (j : ℕ) (u : List ℕ) (hj : j < K) :
    splitφ B cp K p (j :: u) = (p, j :: u) := by
  simp [splitφ, Nat.not_le.2 hj]

theorem splitφ_hi (K : ℕ) (p : Port) (hp : p.node ∈ B) (j : ℕ) (u : List ℕ) :
    splitφ B cp K p ((j + K) :: u) = (cpP cp p, j :: u) := by
  simp [splitφ, hp]

theorem splitφ_O (K : ℕ) (p : Port) (hp : p.node ∉ B) (t : List ℕ) :
    splitφ B cp K p t = (p, t) := by
  cases t <;> simp [splitφ, hp]

theorem pmap_lo (K : ℕ) (x : Dig × Dig) (h1 : ∃ j u, x.1.2 = j :: u ∧ j < K)
    (h2 : ∃ j u, x.2.2 = j :: u ∧ j < K) : pmap (splitφ B cp K) x = x := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := x
  obtain ⟨j1, u1, e1, hj1⟩ := h1
  obtain ⟨j2, u2, e2, hj2⟩ := h2
  simp only at e1 e2
  subst e1 e2
  simp [pmap, splitφ_lo K p1 j1 u1 hj1, splitφ_lo K p2 j2 u2 hj2]

theorem pmap_O (K : ℕ) (x : Dig × Dig) (h1 : x.1.1.node ∉ B) (h2 : x.2.1.node ∉ B) :
    pmap (splitφ B cp K) x = x := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := x
  simp [pmap, splitφ_O K p1 h1, splitφ_O K p2 h2]

theorem pmap_shift (K : ℕ) (y : Dig × Dig) (h1 : y.1.1.node ∈ B) (h2 : y.2.1.node ∈ B)
    (e1 : ∃ j u, y.1.2 = j :: u) (e2 : ∃ j u, y.2.2 = j :: u) :
    pmap (splitφ B cp K) (shiftS K y) = portsP (cpP cp) y := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := y
  obtain ⟨j1, u1, e1⟩ := e1
  obtain ⟨j2, u2, e2⟩ := e2
  simp only at e1 e2 h1 h2
  subst e1 e2
  simp [pmap, shiftS, portsP, addH, splitφ_hi K p1 h1, splitφ_hi K p2 h2]

theorem BoxSplitOK.legs_eq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) :
    L'.base.legs = L.base.legs ++
      ((L.base.nodes.filter fun n => n.id ∈ B).flatMap fun n => n.legs.map (cpP cp)) := by
  simp only [Diagram.legs, h.nodes, List.flatMap_append, List.flatMap_map, legs_cpNode]

theorem BoxSplitOK.idsL (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) :
    (L.base.nodes.map (·.id)).Nodup := by
  have := h.ids
  rw [h.nodes, List.map_append] at this
  exact this.of_append_left

theorem BoxSplitOK.cp_fresh (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (n : DNode)
    (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) :
    cp n.id ≠ m.id := by
  have := h.ids
  rw [h.nodes, List.map_append, List.nodup_append] at this
  intro e
  refine this.2.2 m.id (List.mem_map.2 ⟨m, hm, rfl⟩) (cp n.id) ?_ e.symm
  exact List.mem_map.2 ⟨cpNode cp n,
    List.mem_map.2 ⟨n, List.mem_filter.2 ⟨hn, by simpa using hB⟩, rfl⟩, rfl⟩

theorem BoxSplitOK.cp_inj (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (n : DNode)
    (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) (hmB : m.id ∈ B)
    (e : cp n.id = cp m.id) : n = m := by
  have := h.ids
  rw [h.nodes, List.map_append, List.nodup_append] at this
  have h2 := this.2.1
  rw [List.map_map] at h2
  exact (List.inj_on_of_nodup_map h2) (List.mem_filter.2 ⟨hn, by simpa using hB⟩)
    (List.mem_filter.2 ⟨hm, by simpa using hmB⟩) (by simpa [cpNode] using e)

theorem BoxSplitOK.lc (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (q : Port) :
    L.base.legChain q = L.base.nodeChain q.node := L.base.legChain_eq h.ports h.wch q

theorem BoxSplitOK.lc' (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (q : Port) :
    L'.base.legChain q = L'.base.nodeChain q.node := L'.base.legChain_eq h.ports' h.wch' q

theorem BoxSplitOK.legsL (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) : L.base.legs.Nodup := by
  have := h.legs
  rw [h.legs_eq] at this
  exact this.of_append_left

theorem BoxSplitOK.mem' (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (p : Port) (hp : p ∈ L.base.legs) :
    p ∈ L'.base.legs := by
  rw [h.legs_eq]; exact List.mem_append_left _ hp

theorem BoxSplitOK.memC (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∈ B) : cpP cp p ∈ L'.base.legs := by
  rw [h.legs_eq]
  obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 hp
  have e := node_of_leg n p hpn
  exact List.mem_append_right _ (List.mem_flatMap.2 ⟨n,
    List.mem_filter.2 ⟨hn, by simpa [← e] using hB⟩, List.mem_map.2 ⟨p, hpn, rfl⟩⟩)

theorem BoxSplitOK.chB' (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∈ B) : ∃ r, L.base.nodeChain p.node = mb :: r ∧
      L'.base.nodeChain p.node = c₁ :: r ∧ L'.base.nodeChain (cp p.node) = c₂ :: r := by
  obtain ⟨n, hn, e⟩ := node_of_legs _ p hp
  rw [e] at hB ⊢
  obtain ⟨h1, h2, h3, -⟩ := h.chB n hn hB
  exact ⟨_, h1, h2, h3⟩

theorem BoxSplitOK.chO' (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∉ B) : L'.base.nodeChain p.node = L.base.nodeChain p.node := by
  obtain ⟨n, hn, e⟩ := node_of_legs _ p hp
  rw [e] at hB ⊢
  exact h.chO n hn hB

theorem BoxSplitOK.lp_eq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (ks : ℕ → ℕ) (l : DLink) :
    L.linkPairs ks l = lp ks (L.depth l.box l.l1.node) (ks (L.multOf l.box))
      (L.base.nodeChain l.l1.node) l := by
  rw [linkPairs, h.lc]

theorem BoxSplitOK.lp_eq' (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (ks : ℕ → ℕ) (l : DLink) :
    L'.linkPairs ks l = lp ks (L'.depth l.box l.l1.node) (ks (L'.multOf l.box))
      (L'.base.nodeChain l.l1.node) l := by
  rw [linkPairs, h.lc']

/-- The heads of a link's pairs inside the box, at the depth `d + 1`. -/
theorem lp_heads' (ks : ℕ → ℕ) (d k c0 : ℕ) (r : List ℕ) (l : DLink) (x : Dig × Dig)
    (hx : x ∈ lp ks (d + 1) k (c0 :: r) l) :
    (∃ j u, x.1.2 = j :: u ∧ j < ks c0) ∧ (∃ j u, x.2.2 = j :: u ∧ j < ks c0) :=
  lp_heads ks d k c0 r l x hx

/-- Every digit pair of `L` goes to a digit pair of `L'`. -/
theorem BoxSplitOK.fwd (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (x : Dig × Dig) (hx : x ∈ L.dpairs ks) :
    pmap (splitφ B cp (ks c₁)) x ∈ L'.dpairs ks ∨
      pswap (pmap (splitφ B cp (ks c₁)) x) ∈ L'.dpairs ks := by
  rcases (mem_dpairs L ks x).1 hx with ⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩
  · obtain ⟨hw1, hw2, hiff, hin, hcp⟩ := h.wires w hw
    rw [h.lc] at ht
    by_cases hB : w.1.node ∈ B
    · have hB2 := hiff.1 hB
      obtain ⟨r, c1, c1', c1''⟩ := h.chB' w.1 hw1 hB
      obtain ⟨r2, c2, c2', c2''⟩ := h.chB' w.2 hw2 hB2
      have er : r2 = r := by
        have := h.wch w hw; rw [c1, c2] at this; exact (List.cons.inj this).2.symm
      subst er
      rw [c1] at ht
      obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
      by_cases hj1 : j < ks c₁
      · rw [pmap_lo (ks c₁) _ ⟨j, u, rfl, hj1⟩ ⟨j, u, rfl, hj1⟩]
        have hju : j :: u ∈ cidx ks (c₁ :: r2) := cons_mem_cidx ks c₁ j _ u hj1 hu
        exact wireIn_mem L' ks _ _ hin _ (by rw [h.lc', c1']; exact hju)
          (by rw [h.lc', c2']; exact hju)
      · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
        have e : pmap (splitφ B cp (ks c₁)) ((w.1, (j' + ks c₁) :: u), (w.2, (j' + ks c₁) :: u)) =
            ((cpP cp w.1, j' :: u), (cpP cp w.2, j' :: u)) := by
          simp [pmap, splitφ_hi (ks c₁) w.1 hB, splitφ_hi (ks c₁) w.2 hB2]
        rw [e]
        have hju : j' :: u ∈ cidx ks (c₂ :: r2) := cons_mem_cidx ks c₂ j' _ u (by omega) hu
        exact wireIn_mem L' ks _ _ (hcp hB) _
          (by rw [h.lc']; show _ ∈ cidx ks (L'.base.nodeChain (cp w.1.node)); rw [c1'']; exact hju)
          (by rw [h.lc']; show _ ∈ cidx ks (L'.base.nodeChain (cp w.2.node)); rw [c2'']; exact hju)
    · have hB2 : w.2.node ∉ B := fun e => hB (hiff.2 e)
      rw [pmap_O (ks c₁) _ hB hB2]
      exact wireIn_mem L' ks _ _ hin _ (by rw [h.lc', h.chO' w.1 hw1 hB]; exact ht)
        (by rw [h.lc', h.chO' w.2 hw2 hB2, ← h.wch w hw]; exact ht)
  · obtain ⟨hl1, hl2, he1, he2, hc2, hdl, hgd, hce1, hce2⟩ := h.lwf l hl
    rw [h.lp_eq] at hx
    by_cases hin : l.l1.node ∈ B
    · obtain ⟨hl2B, he1B, he2B, hd1, ⟨lt, hltm, r1, r2, r3, r4, hd', hm'⟩, l', hl'm, q1, q2, q3,
        q4, hd'', hm''⟩ := h.lin l hl hin
      obtain ⟨r, c1, c1', c1''⟩ := h.chB' l.l1 hl1 hin
      obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
        ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
      rw [hd0, c1] at hx
      left
      rcases (mem_lp_split2 ks d0 _ mb c₁ c₂ r l hk x).1 hx with hx | ⟨y, hy, rfl⟩
      · obtain ⟨⟨j1, u1, e1, hj1⟩, ⟨j2, u2, e2, hj2⟩⟩ := lp_heads' ks d0 _ c₁ r l x hx
        rw [pmap_lo (ks c₁) x ⟨j1, u1, e1, hj1⟩ ⟨j2, u2, e2, hj2⟩]
        exact link_mem_dpairs L' ks lt hltm x (by
          rw [h.lp_eq', r1, hd', hd0, hm', c1', lp_congr ks _ _ _ l lt r1 r2 r3 r4]; exact hx)
      · obtain ⟨p1, p2⟩ := lp_ports_in ks _ _ _ l (· ∈ B) y hy hin hl2B he1B he2B
        obtain ⟨⟨j1, u1, e1, -⟩, ⟨j2, u2, e2, -⟩⟩ := lp_heads' ks d0 _ c₂ r l y hy
        rw [pmap_shift (ks c₁) y p1 p2 ⟨j1, u1, e1⟩ ⟨j2, u2, e2⟩]
        refine link_mem_dpairs L' ks l' hl'm _ ?_
        rw [h.lp_eq', q1]
        have hd3 : L'.depth l'.box (cpP cp l.l1).node = d0 + 1 := by
          show L'.depth l'.box (cp l.l1.node) = d0 + 1; omega
        have hc3 : L'.base.nodeChain (cpP cp l.l1).node = c₂ :: r := c1''
        rw [hd3, hm'', hc3, lp_congr ks _ _ _ (l.mapP (cpP cp)) l' q1 q2 q3 q4, lp_mapP]
        exact List.mem_map.2 ⟨y, hy, rfl⟩
    · obtain ⟨hl2O, he1O, he2O, lt, hltm, r1, r2, r3, r4, hd', hm'⟩ := h.lout l hl hin
      left
      obtain ⟨p1, p2⟩ := lp_ports_in ks _ _ _ l (· ∉ B) x hx hin hl2O he1O he2O
      rw [pmap_O (ks c₁) x p1 p2]
      exact link_mem_dpairs L' ks lt hltm x (by
        rw [h.lp_eq', r1, hd', hm', h.chO' l.l1 hl1 hin, lp_congr ks _ _ _ l lt r1 r2 r3 r4]
        exact hx)

/-- Every digit pair of `L'` comes from a digit pair of `L`. -/
theorem BoxSplitOK.bwd (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (y : Dig × Dig) (hy : y ∈ L'.dpairs ks) : ∃ x ∈ L.dpairs ks,
      pmap (splitφ B cp (ks c₁)) x = y ∨ pswap (pmap (splitφ B cp (ks c₁)) x) = y := by
  rcases (mem_dpairs L' ks y).1 hy with ⟨w, hw, u, hu, rfl⟩ | ⟨l', hl', hy⟩
  · rw [h.lc'] at hu
    rcases h.wires' w hw with hw0 | ⟨w0, hw0, hB, rfl⟩
    · obtain ⟨hw1, hw2, hiff, -, -⟩ := h.wires w hw0
      by_cases hB : w.1.node ∈ B
      · obtain ⟨r, c1, c1', -⟩ := h.chB' w.1 hw1 hB
        rw [c1'] at hu
        obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c₁ _ u).1 hu
        refine ⟨((w.1, j :: v), (w.2, j :: v)), wire_mem_dpairs L ks w hw0 _ ?_,
          Or.inl (pmap_lo (ks c₁) _ ⟨j, v, rfl, hj⟩ ⟨j, v, rfl, hj⟩)⟩
        rw [h.lc, c1]; exact cons_mem_cidx ks mb j _ v (by omega) hv
      · refine ⟨((w.1, u), (w.2, u)), wire_mem_dpairs L ks w hw0 _ ?_,
          Or.inl (pmap_O (ks c₁) _ hB (fun e => hB (hiff.2 e)))⟩
        rw [h.lc, ← h.chO' w.1 hw1 hB]; exact hu
    · obtain ⟨hw1, -, hiff, -, -⟩ := h.wires w0 hw0
      obtain ⟨r, c1, -, c1''⟩ := h.chB' w0.1 hw1 hB
      have hu' : u ∈ cidx ks (c₂ :: r) := by rw [← c1'']; exact hu
      obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c₂ _ u).1 hu'
      refine ⟨((w0.1, (j + ks c₁) :: v), (w0.2, (j + ks c₁) :: v)), wire_mem_dpairs L ks w0 hw0 _ ?_,
        Or.inl ?_⟩
      · rw [h.lc, c1]; exact cons_mem_cidx ks mb _ _ v (by omega) hv
      · simp [pmap, splitφ_hi (ks c₁) w0.1 hB, splitφ_hi (ks c₁) w0.2 (hiff.1 hB)]
  · rcases h.links' l' hl' with ⟨l, hl, r1, r2, r3, r4, hd', hm'⟩ |
      ⟨l, hl, hin, q1, q2, q3, q4, hd'', hm''⟩
    · obtain ⟨hl1, -, -, -, -, -, -, -, -⟩ := h.lwf l hl
      rw [h.lp_eq', r1, hd', hm', lp_congr ks _ _ _ l l' r1 r2 r3 r4] at hy
      by_cases hin : l.l1.node ∈ B
      · obtain ⟨-, -, -, hd1, -⟩ := h.lin l hl hin
        obtain ⟨r, c1, c1', -⟩ := h.chB' l.l1 hl1 hin
        obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
          ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
        rw [hd0, c1'] at hy
        refine ⟨y, link_mem_dpairs L ks l hl y ?_, Or.inl ?_⟩
        · rw [h.lp_eq, hd0, c1]; exact (mem_lp_split2 ks d0 _ mb c₁ c₂ r l hk y).2 (Or.inl hy)
        · obtain ⟨⟨j1, u1, e1, hj1⟩, ⟨j2, u2, e2, hj2⟩⟩ := lp_heads' ks d0 _ c₁ r l y hy
          exact pmap_lo (ks c₁) y ⟨j1, u1, e1, hj1⟩ ⟨j2, u2, e2, hj2⟩
      · obtain ⟨hl2O, he1O, he2O, -⟩ := h.lout l hl hin
        rw [h.chO' l.l1 hl1 hin] at hy
        obtain ⟨p1, p2⟩ := lp_ports_in ks _ _ _ l (· ∉ B) y hy hin hl2O he1O he2O
        exact ⟨y, link_mem_dpairs L ks l hl y (by rw [h.lp_eq]; exact hy),
          Or.inl (pmap_O (ks c₁) y p1 p2)⟩
    · obtain ⟨hl1, -, -, -, -, -, -, -, -⟩ := h.lwf l hl
      obtain ⟨hl2B, he1B, he2B, hd1, -⟩ := h.lin l hl hin
      obtain ⟨r, c1, -, c1''⟩ := h.chB' l.l1 hl1 hin
      obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
        ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
      rw [h.lp_eq', q1] at hy
      have hd3 : L'.depth l'.box (cpP cp l.l1).node = d0 + 1 := by
        show L'.depth l'.box (cp l.l1.node) = d0 + 1; omega
      have hc3 : L'.base.nodeChain (cpP cp l.l1).node = c₂ :: r := c1''
      rw [hd3, hm'', hc3, lp_congr ks _ _ _ (l.mapP (cpP cp)) l' q1 q2 q3 q4, lp_mapP] at hy
      obtain ⟨y0, hy0, rfl⟩ := List.mem_map.1 hy
      obtain ⟨p1, p2⟩ := lp_ports_in ks _ _ _ l (· ∈ B) y0 hy0 hin hl2B he1B he2B
      obtain ⟨⟨j1, u1, e1, -⟩, ⟨j2, u2, e2, -⟩⟩ := lp_heads' ks d0 _ c₂ r l y0 hy0
      refine ⟨shiftS (ks c₁) y0, link_mem_dpairs L ks l hl _ ?_,
        Or.inl (pmap_shift (ks c₁) y0 p1 p2 ⟨j1, u1, e1⟩ ⟨j2, u2, e2⟩)⟩
      rw [h.lp_eq, hd0, c1]
      exact (mem_lp_split2 ks d0 _ mb c₁ c₂ r l hk _).2 (Or.inr ⟨y0, hy0, rfl⟩)

/-- A node's copy with no tails, its phase index read by `phAt`. -/
theorem dnodeL_notail (L : LDiagram) (hp : ∀ y ∈ L.base.boxes, y.ports = [])
    (hw : ∀ w ∈ L.base.wires, L.base.nodeChain w.1.node = L.base.nodeChain w.2.node)
    (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) : L.dnodeL ks n t =
      ⟨n.kind, (List.range n.outs.length).map fun i => dlab ⟨n.id, true, i⟩ t,
        (List.range n.ins.length).map fun i => dlab ⟨n.id, false, i⟩ t, n.dim, L.phAt ks n t⟩ := by
  rw [dnodeL, L.base.dnode_notail hp hw]

/-- Outside the copies, `L'` reads the phase indices of `L`. -/
theorem BoxSplitOK.phAt_eq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (n : DNode)
    (hn : n ∈ L.base.nodes) (ks : ℕ → ℕ) (t : List ℕ) : L'.phAt ks n t = L.phAt ks n t := by
  have hc : (L.cph.map fun q => (cp q.1, bid₂, 1, c₁)).find? (fun q => decide (q.1 = n.id)) =
      none := by
    rw [List.find?_eq_none]
    intro q hq
    obtain ⟨q0, hq0, rfl⟩ := List.mem_map.1 hq
    obtain ⟨⟨m, hm, em⟩, hB, -, -⟩ := h.cph q0 hq0
    simp only [decide_eq_true_eq]
    rw [← em]
    exact h.cp_fresh m hm (em ▸ hB) n hn
  unfold phAt
  rw [h.cph', List.find?_append, hc, Option.or_none]
  cases hf : L.cph.find? (fun q => decide (q.1 = n.id)) with
  | none => rfl
  | some q =>
    have hq := List.mem_of_find?_eq_some hf
    have he := List.find?_some hf
    simp only [decide_eq_true_eq] at he
    obtain ⟨-, hB, hbid, -⟩ := h.cph q hq
    obtain ⟨-, -, -, d1, d2, -⟩ := h.chB n hn (he ▸ hB)
    simp only [hbid, d1, d2]

/-- The copy of a node reads the phase index of the node at its copy index raised by `ks c₁`. -/
theorem BoxSplitOK.phAt_cp (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (n : DNode)
    (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) (ks : ℕ → ℕ) (j : ℕ) (u : List ℕ) :
    L'.phAt ks (cpNode cp n) (j :: u) = L.phAt ks n ((j + ks c₁) :: u) := by
  have hc : L.cph.find? (fun q => decide (q.1 = (cpNode cp n).id)) = none := by
    rw [List.find?_eq_none]
    intro q hq
    obtain ⟨⟨m, hm, em⟩, -, -, -⟩ := h.cph q hq
    simp only [decide_eq_true_eq]
    rw [← em]
    exact fun e => h.cp_fresh n hn hB m hm e.symm
  have hcg : L.cph.find? ((fun q => decide (q.1 = (cpNode cp n).id)) ∘
      fun q => (cp q.1, bid₂, 1, c₁)) = L.cph.find? (fun q => decide (q.1 = n.id)) := by
    refine find?_congr' fun q hq => ?_
    obtain ⟨⟨m, hm, em⟩, hmB, -, -⟩ := h.cph q hq
    simp only [Function.comp_apply, cpNode]
    rw [← em]
    by_cases e : m.id = n.id
    · simp [e]
    · have : cp m.id ≠ cp n.id := fun e' => e (by rw [h.cp_inj m hm (em ▸ hmB) n hn hB e'])
      simp [e, this]
  unfold phAt
  rw [h.cph', List.find?_append, hc, Option.none_or, List.find?_map, hcg]
  cases hf : L.cph.find? (fun q => decide (q.1 = n.id)) with
  | none => rfl
  | some q =>
    have hq := List.mem_of_find?_eq_some hf
    have he := List.find?_some hf
    simp only [decide_eq_true_eq] at he
    obtain ⟨-, hqB, hbid, hc0⟩ := h.cph q hq
    obtain ⟨-, -, -, d1, -, d3⟩ := h.chB n hn (he ▸ hqB)
    simp only [cpNode]
    simp only [Option.map_some, hbid, hc0, d1, d3, List.getD_cons_zero]
    ring_nf

/-- The node products agree under the split's relabelling. -/
theorem BoxSplitOK.prod (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (splitφ B cp (ks c₁)) x)) :
    L'.prodL E ks τ = L.prodL E ks τ' := by
  set φ := splitφ B cp (ks c₁)
  have hv : ∀ n ∈ L.base.nodes, ∀ t ∈ cidx ks (L.base.nodeChain n.id),
      val E τ' (L.dnodeL ks n t) = val E τ ⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        n.dim, L.phAt ks n t⟩ := by
    intro n hn t ht
    have e : (⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        n.dim, L.phAt ks n t⟩ : Node) = (L.dnodeL ks n t).rename (relab φ) := by
      rw [dnodeL_notail L h.ports h.wch]
      simp [Node.rename, List.map_map, Function.comp_def, relab_dlab]
    rw [e, val_rename]
    refine val_reads E _ _ _ fun x hx => ?_
    rw [dnodeL_notail L h.ports h.wch] at hx
    simp only [Node.labels, List.mem_append, List.mem_map, List.mem_range] at hx
    have hmem : x ∈ L.base.dlabs ks := by
      rcases hx with ⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩
      · refine (mem_dlabs _ _ _).2 ⟨_, List.mem_flatMap.2 ⟨n, hn, ?_⟩, t, by rw [h.lc]; exact ht, rfl⟩
        simp [DNode.legs, hi]
      · refine (mem_dlabs _ _ _).2 ⟨_, List.mem_flatMap.2 ⟨n, hn, ?_⟩, t, by rw [h.lc]; exact ht, rfl⟩
        simp [DNode.legs, hi]
    exact hτ x hmem
  have hd' : ∀ m : DNode, ∀ t, L'.dnodeL ks m t = ⟨m.kind,
      (List.range m.outs.length).map (fun i => dlab ⟨m.id, true, i⟩ t),
      (List.range m.ins.length).map (fun i => dlab ⟨m.id, false, i⟩ t), m.dim, L'.phAt ks m t⟩ :=
    fun m t => dnodeL_notail L' h.ports' h.wch' ks m t
  set F' : DNode → ℂ := fun m =>
    ((cidx ks (L'.base.nodeChain m.id)).map fun t => val E τ (L'.dnodeL ks m t)).prod
  have key : ∀ n ∈ L.base.nodes,
      ((cidx ks (L.base.nodeChain n.id)).map fun t => val E τ' (L.dnodeL ks n t)).prod =
        F' n * (if n.id ∈ B then F' (cpNode cp n) else 1) := by
    intro n hn
    by_cases hB : n.id ∈ B
    · obtain ⟨c1, c1', c1'', -⟩ := h.chB n hn hB
      simp only [hB, ↓reduceIte]
      set r := (L.base.nodeChain n.id).tail
      have hcn : (cpNode cp n).id = cp n.id := rfl
      simp only [F', hcn, c1'', c1']
      rw [c1, cidx_add ks mb c₁ c₂ r hk, List.map_append, List.prod_append, List.map_map]
      congr 1
      · refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c₁ _ t).1 ht
        rw [hv n hn _ (by rw [c1, cidx_add ks mb c₁ c₂ r hk]; exact List.mem_append_left _ ht), hd',
          h.phAt_eq n hn]
        simp [φ, splitφ_lo (ks c₁) _ j u hj]
      · refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c₂ _ t).1 ht
        simp only [Function.comp_apply, addH]
        have hm : (j + ks c₁) :: u ∈ cidx ks (L.base.nodeChain n.id) := by
          rw [c1, cidx_add ks mb c₁ c₂ r hk]
          exact List.mem_append_right _ (List.mem_map.2 ⟨j :: u, ht, rfl⟩)
        rw [hv n hn _ hm, hd', h.phAt_cp n hn hB]
        have hp : ∀ o i, φ ⟨n.id, o, i⟩ ((j + ks c₁) :: u) = (⟨cp n.id, o, i⟩, j :: u) :=
          fun o i => splitφ_hi (ks c₁) ⟨n.id, o, i⟩ hB j u
        simp [hp, cpNode]
    · simp only [hB, ↓reduceIte, mul_one]
      simp only [F', h.chO n hn hB]
      refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
      rw [hv n hn _ ht, hd', h.phAt_eq n hn]
      have hp : ∀ o i, φ ⟨n.id, o, i⟩ t = (⟨n.id, o, i⟩, t) := fun o i =>
        splitφ_O (ks c₁) ⟨n.id, o, i⟩ hB t
      simp [hp]
  unfold prodL
  rw [List.map_congr_left key, List.prod_map_mul, prod_ite_filter, h.nodes, List.map_append,
    List.prod_append, List.map_map]
  rfl

/-- Splitting an outermost box in two keeps the linked digit sum, `L`'s slots read from `L'`'s
through `splitφ`. -/
theorem evalL_split (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (σ' : ℕ → List ℕ → ℕ) :
    L'.evalL E ks σ' = L.evalL E ks (bndMap L.base L'.base (splitφ B cp (ks c₁)) σ') := by
  refine evalL_relabelφ L L' E ks ks σ' (splitφ B cp (ks c₁)) (Diagram.dlabs_nodup _ _ h.legsL)
    (Diagram.dlabs_nodup _ _ h.legs) h.bnd ?_ ?_ ?_ ?_ ?_ (fun x hx => h.fwd ks hk x hx)
    (fun y hy => h.bwd ks hk y hy) (fun p hp => (h.bndL p hp).1) ?_ ?_
    (fun τ τ' hτ => h.prod E ks hk τ τ' hτ)
  · intro p hp t ht
    rw [h.lc] at ht
    by_cases hB : p.node ∈ B
    · obtain ⟨r, c1, c1', c1''⟩ := h.chB' p hp hB
      rw [c1] at ht
      obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
      by_cases hj1 : j < ks c₁
      · rw [splitφ_lo (ks c₁) p j u hj1]
        exact ⟨h.mem' p hp, by rw [h.lc', c1']; exact cons_mem_cidx ks c₁ j _ u hj1 hu⟩
      · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
        rw [splitφ_hi (ks c₁) p hB j' u]
        exact ⟨h.memC p hp hB, by
          rw [h.lc']; show _ ∈ cidx ks (L'.base.nodeChain (cp p.node)); rw [c1'']
          exact cons_mem_cidx ks c₂ j' _ u (by omega) hu⟩
    · rw [splitφ_O (ks c₁) p hB t]
      exact ⟨h.mem' p hp, by rw [h.lc', h.chO' p hp hB]; exact ht⟩
  · intro q hq u hu
    rw [h.legs_eq, List.mem_append] at hq
    rcases hq with hq | hq
    · by_cases hB : q.node ∈ B
      · obtain ⟨r, c1, c1', -⟩ := h.chB' q hq hB
        rw [h.lc', c1'] at hu
        obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c₁ _ _).1 hu
        exact ⟨q, hq, j :: v, by rw [h.lc, c1]; exact cons_mem_cidx ks mb j _ v (by omega) hv,
          splitφ_lo (ks c₁) q j v hj⟩
      · exact ⟨q, hq, u, by rw [h.lc, ← h.chO' q hq hB, ← h.lc']; exact hu,
          splitφ_O (ks c₁) q hB u⟩
    · obtain ⟨n, hn, hq⟩ := List.mem_flatMap.1 hq
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hq
      obtain ⟨hn, hnB⟩ := List.mem_filter.1 hn
      have hpn := node_of_leg n p hp
      have hpB : p.node ∈ B := by rw [hpn]; simpa using hnB
      have hpL : p ∈ L.base.legs := List.mem_flatMap.2 ⟨n, hn, hp⟩
      obtain ⟨r, c1, -, c1''⟩ := h.chB' p hpL hpB
      have hu' : u ∈ cidx ks (c₂ :: r) := by
        rw [h.lc'] at hu; rw [← c1'']; exact hu
      obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c₂ _ _).1 hu'
      refine ⟨p, hpL, (j + ks c₁) :: v, ?_, splitφ_hi (ks c₁) p hpB j v⟩
      rw [h.lc, c1]
      exact cons_mem_cidx ks mb _ _ v (by omega) hv
  · intro p hp t ht p' hp' t' ht' he
    have nodeB : ∀ q ∈ L.base.legs, q.node ∈ B → ∃ n ∈ L.base.nodes, q.node = n.id ∧ n.id ∈ B :=
      fun q hq hB => by
        obtain ⟨n, hn, e⟩ := node_of_legs _ q hq
        exact ⟨n, hn, e, e ▸ hB⟩
    cases t with
    | nil =>
      cases t' with
      | nil => simp only [splitφ, Prod.mk.injEq] at he; exact ⟨he.1, rfl⟩
      | cons j' u' =>
        simp only [splitφ] at he
        split_ifs at he with h2
        · obtain ⟨n, hn, en, hnB⟩ := nodeB p' hp' h2.1
          obtain ⟨m, hm, em⟩ := node_of_legs _ p hp
          simp only [Prod.mk.injEq] at he
          exact (h.cp_fresh n hn hnB m hm (by rw [← em, he.1, ← en]; rfl)).elim
        · simp at he
    | cons j u =>
      cases t' with
      | nil =>
        simp only [splitφ] at he
        split_ifs at he with h1
        · obtain ⟨n, hn, en, hnB⟩ := nodeB p hp h1.1
          obtain ⟨m, hm, em⟩ := node_of_legs _ p' hp'
          simp only [Prod.mk.injEq] at he
          exact (h.cp_fresh n hn hnB m hm (by rw [← em, ← he.1, ← en]; rfl)).elim
        · simp at he
      | cons j' u' =>
        simp only [splitφ] at he
        split_ifs at he with h1 h2 h2
        · obtain ⟨n, hn, en, hnB⟩ := nodeB p hp h1.1
          obtain ⟨n', hn', en', hnB'⟩ := nodeB p' hp' h2.1
          simp only [cpP, Prod.mk.injEq, Port.mk.injEq, List.cons.injEq] at he
          obtain ⟨⟨e1, e2, e3⟩, e4, e5⟩ := he
          have := h.cp_inj n hn hnB n' hn' hnB' (by rw [← en, ← en']; exact e1)
          refine ⟨by cases p; cases p'; simp_all, ?_⟩
          have : j = j' := by omega
          subst this; rw [e5]
        · obtain ⟨n, hn, en, hnB⟩ := nodeB p hp h1.1
          obtain ⟨m, hm, em⟩ := node_of_legs _ p' hp'
          simp only [Prod.mk.injEq] at he
          exact (h.cp_fresh n hn hnB m hm (by rw [← em, ← he.1, ← en]; rfl)).elim
        · obtain ⟨n, hn, en, hnB⟩ := nodeB p' hp' h2.1
          obtain ⟨m, hm, em⟩ := node_of_legs _ p hp
          simp only [Prod.mk.injEq] at he
          exact (h.cp_fresh n hn hnB m hm (by rw [← em, he.1, ← en]; rfl)).elim
        · simp only [Prod.mk.injEq] at he
          exact he
  · intro p hp t ht
    obtain ⟨n, hn, en⟩ := node_of_legs _ p hp
    have hc : n ∈ L'.base.nodes := by rw [h.nodes]; exact List.mem_append_left _ hn
    have hcp : n.id ∈ B → cpNode cp n ∈ L'.base.nodes := fun hnB => by
      rw [h.nodes]
      exact List.mem_append_right _
        (List.mem_map.2 ⟨n, List.mem_filter.2 ⟨hn, by simpa using hnB⟩, rfl⟩)
    have hO : E.dimv (L'.base.portDim p) = E.dimv (L.base.portDim p) := by
      rw [portDim_of _ h.ids _ hc p en, portDim_of _ h.idsL _ hn p en]
    cases t with
    | nil => exact hO
    | cons j u =>
      simp only [splitφ]
      split_ifs with h1
      · have hnB : n.id ∈ B := en ▸ h1.1
        rw [portDim_of _ h.ids _ (hcp hnB) (cpP cp p) (by simp [cpP, cpNode, en]),
          portDim_of _ h.idsL _ hn p en]
        simp [cpP, cpNode]
      · exact hO
  · intro x hx
    rcases (mem_dpairs L ks x).1 hx with ⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩
    · obtain ⟨hw1, hw2, -⟩ := h.wires w hw
      refine ⟨hw1, ht, hw2, ?_⟩
      rw [h.lc] at ht ⊢
      rw [← h.wch w hw]
      exact ht
    · obtain ⟨hl1, hl2, he1, he2, hc2, hdl, hgd, hce1, hce2⟩ := h.lwf l hl
      rw [h.lp_eq] at hx
      rcases (mem_lp ks _ _ _ l x).1 hx with ⟨t, ht, h1, rfl⟩ | ⟨t, ht, h1, rfl⟩ |
        ⟨t, ht, h1, rfl⟩ | ⟨-, u, hu, rfl⟩
      · refine ⟨hl1, by rw [h.lc]; exact ht, hl2, ?_⟩
        rw [h.lc, hc2]
        exact cidx_set ks _ t _ _ ht (by rw [hgd]; exact h1) hdl
      · exact ⟨hl1, by rw [h.lc]; exact ht, he1,
          by rw [h.lc, hce1]; exact cidx_eraseIdx ks _ t _ ht⟩
      · exact ⟨he2, by rw [h.lc, hce2]; exact cidx_eraseIdx ks _ t _ ht, hl2,
          by rw [h.lc, hc2]; exact ht⟩
      · exact ⟨he2, by rw [h.lc, hce2]; exact hu, he1, by rw [h.lc, hce1]; exact hu⟩
  · intro p hp t ht
    obtain ⟨-, h2, h3⟩ := h.bndL p hp
    cases t with
    | nil => exact h2
    | cons j u =>
      simp only [splitφ]
      split_ifs with h1
      · exact h3 h1.1
      · exact h2
  · intro q hq u hu
    rcases h.bnd' q hq with hq' | ⟨p, hp, hB, rfl⟩
    · have hql := (h.bndL q hq').1
      by_cases hB : q.node ∈ B
      · obtain ⟨r, c1, c1', -⟩ := h.chB' q hql hB
        rw [h.lc', c1'] at hu
        obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c₁ _ _).1 hu
        exact ⟨q, hq', j :: v, by rw [h.lc, c1]; exact cons_mem_cidx ks mb j _ v (by omega) hv,
          splitφ_lo (ks c₁) q j v hj⟩
      · exact ⟨q, hq', u, by rw [h.lc, ← h.chO' q hql hB, ← h.lc']; exact hu,
          splitφ_O (ks c₁) q hB u⟩
    · have hpl := (h.bndL p hp).1
      obtain ⟨r, c1, -, c1''⟩ := h.chB' p hpl hB
      have hu' : u ∈ cidx ks (c₂ :: r) := by
        rw [h.lc'] at hu; rw [← c1'']; exact hu
      obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c₂ _ _).1 hu'
      refine ⟨p, hp, (j + ks c₁) :: v, ?_, splitφ_hi (ks c₁) p hB j v⟩
      rw [h.lc, c1]
      exact cons_mem_cidx ks mb _ _ v (by omega) hv

end split

end LDiagram

end ArchytasZX
