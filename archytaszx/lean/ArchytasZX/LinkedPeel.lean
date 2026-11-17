/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Linked
import ArchytasZX.PortBox

/-!
# Peeling the last copy of an outermost chain box

`evalL_peel`: for an outermost node box `bid` over the nodes `B` at `s + 1` copies (multiplicity
index `mb`) and a linked diagram `L'` holding `B`'s template at `s` copies (index `c`) and its last
copy as the nodes `cp x` (`PeelOK`), `L'` is `L` with `B`'s last-copy digits read at the copies.
The box's links end in the copy: its last template copy's `l1` meets the copy's `l2`, the copy's
`l1` meets the old end `e1` by a wire; the links of boxes inside it hold once on the template and
once on the copy.

`evalL_rev`: for `L'` the diagram `L` with the links of the outermost box `bid` flipped
(`RevOK`, both ways), `L'` is `L` with the box's copy `t` read at `revT t`, the first coordinate
counted from the far end.
-/

namespace ArchytasZX

theorem mem_cidx_cons (ks : ℕ → ℕ) (m : ℕ) (c t : List ℕ) :
    t ∈ cidx ks (m :: c) ↔ ∃ j u, t = j :: u ∧ j < ks m ∧ u ∈ cidx ks c := by
  rw [mem_cidx]
  constructor
  · intro h
    cases h with
    | cons hj hu => exact ⟨_, _, rfl, hj, (mem_cidx ks c _).2 hu⟩
  · rintro ⟨j, u, rfl, hj, hu⟩
    exact List.Forall₂.cons hj ((mem_cidx ks c u).1 hu)

theorem cons_mem_cidx (ks : ℕ → ℕ) (m j : ℕ) (c u : List ℕ) (hj : j < ks m) (hu : u ∈ cidx ks c) :
    j :: u ∈ cidx ks (m :: c) := (mem_cidx_cons ks m c _).2 ⟨j, u, rfl, hj, hu⟩

theorem cidx_eraseIdx (ks : ℕ → ℕ) (c t : List ℕ) (d : ℕ) (h : t ∈ cidx ks c) :
    t.eraseIdx d ∈ cidx ks (c.eraseIdx d) := by
  induction c generalizing t d with
  | nil => simp [cidx] at h; subst h; simp [cidx]
  | cons m c ih =>
    obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks m c t).1 h
    cases d with
    | zero => simpa using hu
    | succ d => simpa using cons_mem_cidx ks m j _ _ hj (ih u d hu)

theorem cidx_set (ks : ℕ → ℕ) (c t : List ℕ) (d v : ℕ) (h : t ∈ cidx ks c)
    (hv : v < ks (c.getD d 0)) (hd : d < c.length) : t.set d v ∈ cidx ks c := by
  induction c generalizing t d with
  | nil => simp at hd
  | cons m c ih =>
    obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks m c t).1 h
    cases d with
    | zero => exact cons_mem_cidx ks m v c u (by simpa using hv) hu
    | succ d =>
      exact cons_mem_cidx ks m j c _ hj (ih u d hu (by simpa using hv) (by simpa using hd))

namespace Diagram

variable (D : Diagram)

theorem walk_nil_of (p : DBox → Bool) (hp : ∀ y ∈ D.boxes, p y = false) (fuel : ℕ)
    (start : Option ℕ) : D.walk p fuel start = [] := by
  cases fuel with
  | zero => rfl
  | succ fuel =>
    simp only [walk]
    have : D.boxes.find? (fun b => p b && b.parent == start) = none := by
      rw [List.find?_eq_none]
      intro y hy
      simp [hp y hy]
    rw [this]

theorem portBoxes_nil (hp : ∀ y ∈ D.boxes, y.ports = []) (q : Port) : D.portBoxes q = [] :=
  D.walk_nil_of _ (fun y hy => by simp [hp y hy]) _ _

/-- With no port boxes and every wire's ends of one node chain, no leg has a tail. -/
theorem tail_nil (hp : ∀ y ∈ D.boxes, y.ports = [])
    (hw : ∀ w ∈ D.wires, D.nodeChain w.1.node = D.nodeChain w.2.node) (q : Port) :
    D.tail q = [] := by
  unfold tail partner
  cases h : D.wires.find? (fun w => decide (w.1 = q ∨ w.2 = q)) with
  | none => simp [D.portBoxes_nil hp]
  | some w =>
    have hw' := hw w (List.mem_of_find?_eq_some h)
    have hq := List.find?_some h
    simp only [decide_eq_true_eq] at hq
    simp only [Option.map_some]
    split_ifs with h1 <;> simp_all

theorem legChain_eq (hp : ∀ y ∈ D.boxes, y.ports = [])
    (hw : ∀ w ∈ D.wires, D.nodeChain w.1.node = D.nodeChain w.2.node) (q : Port) :
    D.legChain q = D.nodeChain q.node := by
  simp [legChain, D.tail_nil hp hw]

/-- A node's copy `t` with no tails: each leg read as its digit `t`. -/
theorem dnode_notail (hp : ∀ y ∈ D.boxes, y.ports = [])
    (hw : ∀ w ∈ D.wires, D.nodeChain w.1.node = D.nodeChain w.2.node) (ks : ℕ → ℕ) (n : DNode)
    (t : List ℕ) : D.dnode ks n t =
      ⟨n.kind, (List.range n.outs.length).map fun i => dlab ⟨n.id, true, i⟩ t,
        (List.range n.ins.length).map fun i => dlab ⟨n.id, false, i⟩ t, n.dim, n.ph⟩ := by
  simp [dnode, D.tail_nil hp hw, rcidx, cidx, List.map_eq_flatMap]

end Diagram

namespace LDiagram

theorem mem_lp (ks : ℕ → ℕ) (d k : ℕ) (c : List ℕ) (l : DLink) (x : Dig × Dig) :
    x ∈ lp ks d k c l ↔
      (∃ t ∈ cidx ks c, t.getD d 0 + 1 < k ∧ x = ((l.l1, t), (l.l2, t.set d (t.getD d 0 + 1)))) ∨
      (∃ t ∈ cidx ks c, ¬ t.getD d 0 + 1 < k ∧ x = ((l.l1, t), (l.e1, t.eraseIdx d))) ∨
      (∃ t ∈ cidx ks c, t.getD d 0 = 0 ∧ x = ((l.e2, t.eraseIdx d), (l.l2, t))) ∨
      (k = 0 ∧ ∃ u ∈ cidx ks (c.eraseIdx d), x = ((l.e2, u), (l.e1, u))) := by
  simp only [lp, List.mem_append, List.mem_flatMap]
  constructor
  · rintro (⟨t, ht, hx⟩ | hx)
    · rcases hx with hx | hx
      · split_ifs at hx with h1
        · exact Or.inl ⟨t, ht, h1, by simpa using hx⟩
        · exact Or.inr (Or.inl ⟨t, ht, h1, by simpa using hx⟩)
      · split_ifs at hx with h1
        · exact Or.inr (Or.inr (Or.inl ⟨t, ht, h1, by simpa using hx⟩))
        · simp at hx
    · split_ifs at hx with h1
      · obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hx
        exact Or.inr (Or.inr (Or.inr ⟨h1, u, hu, rfl⟩))
      · simp at hx
  · rintro (⟨t, ht, h1, rfl⟩ | ⟨t, ht, h1, rfl⟩ | ⟨t, ht, h1, rfl⟩ | ⟨h1, u, hu, rfl⟩)
    · exact Or.inl ⟨t, ht, Or.inl (by split_ifs with h2 <;> simp_all)⟩
    · exact Or.inl ⟨t, ht, Or.inl (by split_ifs with h2 <;> simp_all)⟩
    · exact Or.inl ⟨t, ht, Or.inr (by split_ifs with h2 <;> simp_all)⟩
    · exact Or.inr (by rw [if_pos h1]; exact List.mem_map.2 ⟨u, hu, rfl⟩)

/-- A port moved to the node `cp p.node`. -/
def cpP (cp : ℕ → ℕ) (p : Port) : Port := ⟨cp p.node, p.out, p.pos⟩

/-- A node renamed `cp n.id`. -/
def cpNode (cp : ℕ → ℕ) (n : DNode) : DNode := { n with id := cp n.id }

/-- A link with every port moved by `f`. -/
def _root_.ArchytasZX.DLink.mapP (f : Port → Port) (l : DLink) : DLink :=
  { l with l1 := f l.l1, l2 := f l.l2, e1 := f l.e1, e2 := f l.e2 }

/-- Both digits of a pair with the copy index `s` put first. -/
def consS (s : ℕ) (y : Dig × Dig) : Dig × Dig := ((y.1.1, s :: y.1.2), (y.2.1, s :: y.2.2))

/-- Both ports of a pair moved by `f`. -/
def portsP (f : Port → Port) (y : Dig × Dig) : Dig × Dig := ((f y.1.1, y.1.2), (f y.2.1, y.2.2))

theorem lp_mapP (ks : ℕ → ℕ) (d k : ℕ) (c : List ℕ) (l : DLink) (f : Port → Port) :
    lp ks d k c (l.mapP f) = (lp ks d k c l).map (portsP f) := by
  simp only [lp, DLink.mapP, List.map_append, List.map_flatMap, List.flatMap_map]
  congr 1
  · refine List.flatMap_congr fun t _ => ?_
    split_ifs <;> simp [portsP]
  · split_ifs <;> simp [portsP]

/-- The pairs of a link inside the outermost box: those of the template copies, and those of the
last copy, its copy index `s` put first. -/
theorem mem_lp_split (ks : ℕ → ℕ) (d k : ℕ) (m0 c0 : ℕ) (r : List ℕ) (l : DLink)
    (hs : ks m0 = ks c0 + 1) (x : Dig × Dig) :
    x ∈ lp ks (d + 1) k (m0 :: r) l ↔
      x ∈ lp ks (d + 1) k (c0 :: r) l ∨ ∃ y ∈ lp ks d k r l, x = consS (ks c0) y := by
  rw [mem_lp, mem_lp]
  simp only [List.eraseIdx_cons_succ, mem_cidx_cons, List.getD_cons_succ, List.set_cons_succ]
  constructor
  · rintro (⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ |
      ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨h1, _, ⟨j, u, rfl, hj, hu⟩, rfl⟩)
    · rcases Nat.lt_succ_iff_lt_or_eq.1 (hs ▸ hj) with hj | rfl
      · exact Or.inl (Or.inl ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩)
      · exact Or.inr ⟨_, (mem_lp ks d k r l _).2 (Or.inl ⟨u, hu, h1, rfl⟩), rfl⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.1 (hs ▸ hj) with hj | rfl
      · exact Or.inl (Or.inr (Or.inl ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩))
      · exact Or.inr ⟨_, (mem_lp ks d k r l _).2 (Or.inr (Or.inl ⟨u, hu, h1, rfl⟩)), rfl⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.1 (hs ▸ hj) with hj | rfl
      · exact Or.inl (Or.inr (Or.inr (Or.inl ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩)))
      · exact Or.inr ⟨_, (mem_lp ks d k r l _).2 (Or.inr (Or.inr (Or.inl ⟨u, hu, h1, rfl⟩))),
          rfl⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.1 (hs ▸ hj) with hj | rfl
      · exact Or.inl (Or.inr (Or.inr (Or.inr ⟨h1, _, ⟨j, u, rfl, hj, hu⟩, rfl⟩)))
      · exact Or.inr ⟨_, (mem_lp ks d k r l _).2 (Or.inr (Or.inr (Or.inr ⟨h1, u, hu, rfl⟩))),
          rfl⟩
  · rintro ((⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ |
      ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨h1, _, ⟨j, u, rfl, hj, hu⟩, rfl⟩) | ⟨y, hy, rfl⟩)
    · exact Or.inl ⟨_, ⟨j, u, rfl, by omega, hu⟩, h1, rfl⟩
    · exact Or.inr (Or.inl ⟨_, ⟨j, u, rfl, by omega, hu⟩, h1, rfl⟩)
    · exact Or.inr (Or.inr (Or.inl ⟨_, ⟨j, u, rfl, by omega, hu⟩, h1, rfl⟩))
    · exact Or.inr (Or.inr (Or.inr ⟨h1, _, ⟨j, u, rfl, by omega, hu⟩, rfl⟩))
    · rcases (mem_lp ks d k r l y).1 hy with ⟨u, hu, h1, rfl⟩ | ⟨u, hu, h1, rfl⟩ |
        ⟨u, hu, h1, rfl⟩ | ⟨h1, u, hu, rfl⟩
      · exact Or.inl ⟨_, ⟨_, u, rfl, by omega, hu⟩, h1, rfl⟩
      · exact Or.inr (Or.inl ⟨_, ⟨_, u, rfl, by omega, hu⟩, h1, rfl⟩)
      · exact Or.inr (Or.inr (Or.inl ⟨_, ⟨_, u, rfl, by omega, hu⟩, h1, rfl⟩))
      · exact Or.inr (Or.inr (Or.inr ⟨h1, _, ⟨_, u, rfl, by omega, hu⟩, rfl⟩))

/-- In the pairs of a link inside the outermost box at `ks c0` template copies, every digit's
first coordinate is below `ks c0`. -/
theorem lp_heads (ks : ℕ → ℕ) (d k c0 : ℕ) (r : List ℕ) (l : DLink) (x : Dig × Dig)
    (hx : x ∈ lp ks (d + 1) k (c0 :: r) l) :
    (∃ j u, x.1.2 = j :: u ∧ j < ks c0) ∧ (∃ j u, x.2.2 = j :: u ∧ j < ks c0) := by
  rw [mem_lp] at hx
  simp only [List.eraseIdx_cons_succ, mem_cidx_cons, List.getD_cons_succ, List.set_cons_succ] at hx
  rcases hx with ⟨_, ⟨j, u, rfl, hj, -⟩, -, rfl⟩ | ⟨_, ⟨j, u, rfl, hj, -⟩, -, rfl⟩ |
    ⟨_, ⟨j, u, rfl, hj, -⟩, -, rfl⟩ | ⟨-, _, ⟨j, u, rfl, hj, -⟩, rfl⟩ <;>
  exact ⟨⟨j, _, rfl, hj⟩, ⟨j, _, rfl, hj⟩⟩

/-- The peel's map of digits: a leg on `B` at the copy `s` of the outermost box goes to its
copy. -/
def peelφ (B : List ℕ) (cp : ℕ → ℕ) (s : ℕ) (p : Port) (t : List ℕ) : Dig :=
  if p.node ∈ B ∧ t.head? = some s then (cpP cp p, t.tail) else (p, t)

/-- A wire or its reverse in `ws`. -/
def WireIn (ws : List (Port × Port)) (w : Port × Port) : Prop := w ∈ ws ∨ (w.2, w.1) ∈ ws

instance (ws : List (Port × Port)) (w : Port × Port) : Decidable (WireIn ws w) := by
  unfold WireIn; infer_instance

/-- `L'` is `L` with the last copy of the outermost node box `bid` (multiplicity index `mb`,
nodes `B`) peeled into the nodes `cp x`, the box left at the index `c`. -/
structure PeelOK (L L' : LDiagram) (bid mb c : ℕ) (B : List ℕ) (cp : ℕ → ℕ) : Prop where
  ports : ∀ y ∈ L.base.boxes, y.ports = []
  ports' : ∀ y ∈ L'.base.boxes, y.ports = []
  wch : ∀ w ∈ L.base.wires, L.base.nodeChain w.1.node = L.base.nodeChain w.2.node
  wch' : ∀ w ∈ L'.base.wires, L'.base.nodeChain w.1.node = L'.base.nodeChain w.2.node
  cph : L.cph = []
  cph' : L'.cph = []
  nodes : L'.base.nodes = L.base.nodes ++ (L.base.nodes.filter fun n => n.id ∈ B).map (cpNode cp)
  ids : (L'.base.nodes.map (·.id)).Nodup
  legs : L'.base.legs.Nodup
  bnd : L'.base.bnd.Nodup
  chB : ∀ n ∈ L.base.nodes, n.id ∈ B →
    L.base.nodeChain n.id = mb :: L'.base.nodeChain (cp n.id) ∧
      L'.base.nodeChain n.id = c :: L'.base.nodeChain (cp n.id)
  chO : ∀ n ∈ L.base.nodes, n.id ∉ B → L'.base.nodeChain n.id = L.base.nodeChain n.id
  wires : ∀ w ∈ L.base.wires, w.1 ∈ L.base.legs ∧ w.2 ∈ L.base.legs ∧
    (w.1.node ∈ B ↔ w.2.node ∈ B) ∧ WireIn L'.base.wires w ∧
    (w.1.node ∈ B → WireIn L'.base.wires (cpP cp w.1, cpP cp w.2))
  wires' : ∀ w ∈ L'.base.wires, w ∈ L.base.wires ∨
    (∃ w0 ∈ L.base.wires, w0.1.node ∈ B ∧ w = (cpP cp w0.1, cpP cp w0.2)) ∨
    (∃ l ∈ L.links, l.box = bid ∧ (w = (cpP cp l.l1, l.e1) ∨ w = (l.e1, cpP cp l.l1)))
  lwf : ∀ l ∈ L.links, l.l1 ∈ L.base.legs ∧ l.l2 ∈ L.base.legs ∧ l.e1 ∈ L.base.legs ∧
    l.e2 ∈ L.base.legs ∧ L.base.nodeChain l.l2.node = L.base.nodeChain l.l1.node ∧
    L.depth l.box l.l1.node < (L.base.nodeChain l.l1.node).length ∧
    (L.base.nodeChain l.l1.node).getD (L.depth l.box l.l1.node) 0 = L.multOf l.box ∧
    L.base.nodeChain l.e1.node = (L.base.nodeChain l.l1.node).eraseIdx (L.depth l.box l.l1.node) ∧
    L.base.nodeChain l.e2.node = (L.base.nodeChain l.l1.node).eraseIdx (L.depth l.box l.l1.node)
  lb : ∀ l ∈ L.links, l.box = bid → l.l1.node ∈ B ∧ l.l2.node ∈ B ∧ l.e1.node ∉ B ∧
    l.e2.node ∉ B ∧ L.depth bid l.l1.node = 0 ∧ L.multOf bid = mb ∧
    { l with e1 := cpP cp l.l2 } ∈ L'.links ∧ L'.depth bid l.l1.node = 0 ∧ L'.multOf bid = c ∧
    WireIn L'.base.wires (cpP cp l.l1, l.e1)
  lin : ∀ l ∈ L.links, l.box ≠ bid → l.l1.node ∈ B → l.l2.node ∈ B ∧ l.e1.node ∈ B ∧
    l.e2.node ∈ B ∧ 1 ≤ L.depth l.box l.l1.node ∧
    (∃ l' ∈ L'.links, l'.l1 = l.l1 ∧ l'.l2 = l.l2 ∧ l'.e1 = l.e1 ∧ l'.e2 = l.e2 ∧
      L'.depth l'.box l.l1.node = L.depth l.box l.l1.node ∧ L'.multOf l'.box = L.multOf l.box) ∧
    ∃ l' ∈ L'.links, l'.l1 = cpP cp l.l1 ∧ l'.l2 = cpP cp l.l2 ∧ l'.e1 = cpP cp l.e1 ∧
      l'.e2 = cpP cp l.e2 ∧ L'.depth l'.box (cp l.l1.node) + 1 = L.depth l.box l.l1.node ∧
      L'.multOf l'.box = L.multOf l.box
  lout : ∀ l ∈ L.links, l.box ≠ bid → l.l1.node ∉ B → l.l2.node ∉ B ∧ l.e1.node ∉ B ∧
    l.e2.node ∉ B ∧ ∃ l' ∈ L'.links, l'.l1 = l.l1 ∧ l'.l2 = l.l2 ∧ l'.e1 = l.e1 ∧ l'.e2 = l.e2 ∧
      L'.depth l'.box l.l1.node = L.depth l.box l.l1.node ∧ L'.multOf l'.box = L.multOf l.box
  links' : ∀ l' ∈ L'.links, (∃ l ∈ L.links, l.box = bid ∧ l' = { l with e1 := cpP cp l.l2 }) ∨
    (∃ l ∈ L.links, l.box ≠ bid ∧ l'.l1 = l.l1 ∧ l'.l2 = l.l2 ∧ l'.e1 = l.e1 ∧ l'.e2 = l.e2 ∧
      L'.depth l'.box l.l1.node = L.depth l.box l.l1.node ∧ L'.multOf l'.box = L.multOf l.box) ∨
    (∃ l ∈ L.links, l.box ≠ bid ∧ l.l1.node ∈ B ∧ l'.l1 = cpP cp l.l1 ∧ l'.l2 = cpP cp l.l2 ∧
      l'.e1 = cpP cp l.e1 ∧ l'.e2 = cpP cp l.e2 ∧
      L'.depth l'.box (cp l.l1.node) + 1 = L.depth l.box l.l1.node ∧
      L'.multOf l'.box = L.multOf l.box)
  bndL : ∀ p ∈ L.base.bnd, p ∈ L.base.legs ∧ p ∈ L'.base.bnd ∧ (p.node ∈ B → cpP cp p ∈ L'.base.bnd)
  bnd' : ∀ q ∈ L'.base.bnd, q ∈ L.base.bnd ∨ ∃ p ∈ L.base.bnd, p.node ∈ B ∧ q = cpP cp p

theorem legs_cpNode (cp : ℕ → ℕ) (n : DNode) : (cpNode cp n).legs = n.legs.map (cpP cp) := by
  simp [DNode.legs, cpNode, cpP, List.map_map, Function.comp_def]

theorem node_of_leg (n : DNode) (p : Port) (h : p ∈ n.legs) : p.node = n.id := by
  simp only [DNode.legs, List.mem_append, List.mem_map] at h
  rcases h with ⟨_, _, rfl⟩ | ⟨_, _, rfl⟩ <;> rfl

theorem node_of_legs (D : Diagram) (p : Port) (h : p ∈ D.legs) : ∃ n ∈ D.nodes, p.node = n.id := by
  obtain ⟨n, hn, hp⟩ := List.mem_flatMap.1 h
  exact ⟨n, hn, node_of_leg n p hp⟩

theorem portDim_of (D : Diagram) (hid : (D.nodes.map (·.id)).Nodup) (n : DNode) (hn : n ∈ D.nodes)
    (p : Port) (hp : p.node = n.id) :
    D.portDim p = if p.out then n.outs.getD p.pos 0 else n.ins.getD p.pos 0 := by
  have : D.nodes.find? (fun m => decide (m.id = p.node)) = some n := by
    rw [List.find?_eq_some_iff_getElem]
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
    refine ⟨by simp [hp], i, hi, rfl, fun j hj => ?_⟩
    have hne := (List.Nodup.getElem_inj_iff hid (i := j) (j := i)
      (hi := by simpa using lt_trans hj hi) (hj := by simpa using hi)).not.2 (by omega)
    simp only [List.getElem_map] at hne
    simp [hp, hne]
  simp [Diagram.portDim, this]

section peel

variable {L L' : LDiagram} {bid mb c : ℕ} {B : List ℕ} {cp : ℕ → ℕ}

theorem PeelOK.legs_eq (h : PeelOK L L' bid mb c B cp) :
    L'.base.legs = L.base.legs ++
      ((L.base.nodes.filter fun n => n.id ∈ B).flatMap fun n => n.legs.map (cpP cp)) := by
  simp only [Diagram.legs, h.nodes, List.flatMap_append, List.flatMap_map, legs_cpNode]

theorem PeelOK.idsL (h : PeelOK L L' bid mb c B cp) : (L.base.nodes.map (·.id)).Nodup := by
  have := h.ids
  rw [h.nodes, List.map_append] at this
  exact this.of_append_left

theorem PeelOK.cp_fresh (h : PeelOK L L' bid mb c B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) : cp n.id ≠ m.id := by
  have := h.ids
  rw [h.nodes, List.map_append, List.nodup_append] at this
  intro e
  refine this.2.2 m.id (List.mem_map.2 ⟨m, hm, rfl⟩) (cp n.id) ?_ e.symm
  exact List.mem_map.2 ⟨cpNode cp n,
    List.mem_map.2 ⟨n, List.mem_filter.2 ⟨hn, by simpa using hB⟩, rfl⟩, rfl⟩

theorem PeelOK.cp_inj (h : PeelOK L L' bid mb c B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) (hmB : m.id ∈ B) (e : cp n.id = cp m.id) :
    n = m := by
  have := h.ids
  rw [h.nodes, List.map_append, List.nodup_append] at this
  have h2 := this.2.1
  rw [List.map_map] at h2
  have hf : (L.base.nodes.filter fun n => n.id ∈ B).Nodup :=
    (List.Nodup.of_map _ h.idsL).filter _
  have := (List.inj_on_of_nodup_map h2) (List.mem_filter.2 ⟨hn, by simpa using hB⟩)
    (List.mem_filter.2 ⟨hm, by simpa using hmB⟩) (by simpa [cpNode] using e)
  exact this

theorem PeelOK.lc (h : PeelOK L L' bid mb c B cp) (q : Port) :
    L.base.legChain q = L.base.nodeChain q.node := L.base.legChain_eq h.ports h.wch q

theorem PeelOK.lc' (h : PeelOK L L' bid mb c B cp) (q : Port) :
    L'.base.legChain q = L'.base.nodeChain q.node := L'.base.legChain_eq h.ports' h.wch' q

theorem PeelOK.legsL (h : PeelOK L L' bid mb c B cp) : L.base.legs.Nodup := by
  have := h.legs
  rw [h.legs_eq] at this
  exact this.of_append_left

theorem PeelOK.mem' (h : PeelOK L L' bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs) :
    p ∈ L'.base.legs := by
  rw [h.legs_eq]; exact List.mem_append_left _ hp

theorem PeelOK.memC (h : PeelOK L L' bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∈ B) : cpP cp p ∈ L'.base.legs := by
  rw [h.legs_eq]
  obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 hp
  have e := node_of_leg n p hpn
  exact List.mem_append_right _ (List.mem_flatMap.2 ⟨n, List.mem_filter.2 ⟨hn, by simpa [← e] using hB⟩,
    List.mem_map.2 ⟨p, hpn, rfl⟩⟩)

theorem PeelOK.chB' (h : PeelOK L L' bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∈ B) : L.base.nodeChain p.node = mb :: L'.base.nodeChain (cp p.node) ∧
      L'.base.nodeChain p.node = c :: L'.base.nodeChain (cp p.node) := by
  obtain ⟨n, hn, e⟩ := node_of_legs _ p hp
  rw [e] at hB ⊢
  exact h.chB n hn hB

theorem PeelOK.chO' (h : PeelOK L L' bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∉ B) : L'.base.nodeChain p.node = L.base.nodeChain p.node := by
  obtain ⟨n, hn, e⟩ := node_of_legs _ p hp
  rw [e] at hB ⊢
  exact h.chO n hn hB

theorem peelφ_B (s : ℕ) (p : Port) (hp : p.node ∈ B) (u : List ℕ) :
    peelφ B cp s p (s :: u) = (cpP cp p, u) := by
  simp [peelφ, hp]

theorem peelφ_ne (s : ℕ) (p : Port) (j : ℕ) (u : List ℕ) (hj : j ≠ s) :
    peelφ B cp s p (j :: u) = (p, j :: u) := by
  simp [peelφ, hj]

theorem peelφ_O (s : ℕ) (p : Port) (hp : p.node ∉ B) (t : List ℕ) :
    peelφ B cp s p t = (p, t) := by
  simp [peelφ, hp]

theorem lp_ports (ks : ℕ → ℕ) (d k : ℕ) (cs : List ℕ) (l : DLink) (x : Dig × Dig)
    (hx : x ∈ lp ks d k cs l) : (x.1.1 = l.l1 ∨ x.1.1 = l.e2) ∧ (x.2.1 = l.l2 ∨ x.2.1 = l.e1) := by
  rcases (mem_lp ks d k cs l x).1 hx with ⟨_, _, _, rfl⟩ | ⟨_, _, _, rfl⟩ | ⟨_, _, _, rfl⟩ |
    ⟨_, _, _, rfl⟩ <;> simp

theorem lp_congr (ks : ℕ → ℕ) (d k : ℕ) (cs : List ℕ) (l l' : DLink) (h1 : l'.l1 = l.l1)
    (h2 : l'.l2 = l.l2) (h3 : l'.e1 = l.e1) (h4 : l'.e2 = l.e2) : lp ks d k cs l' = lp ks d k cs l := by
  simp only [lp, h1, h2, h3, h4]

theorem PeelOK.lp_eq (h : PeelOK L L' bid mb c B cp) (ks : ℕ → ℕ) (l : DLink) :
    L.linkPairs ks l = lp ks (L.depth l.box l.l1.node) (ks (L.multOf l.box))
      (L.base.nodeChain l.l1.node) l := by
  rw [linkPairs, h.lc]

theorem PeelOK.lp_eq' (h : PeelOK L L' bid mb c B cp) (ks : ℕ → ℕ) (l : DLink) :
    L'.linkPairs ks l = lp ks (L'.depth l.box l.l1.node) (ks (L'.multOf l.box))
      (L'.base.nodeChain l.l1.node) l := by
  rw [linkPairs, h.lc']

theorem mem_dpairs (L : LDiagram) (ks : ℕ → ℕ) (x : Dig × Dig) : x ∈ L.dpairs ks ↔
    (∃ w ∈ L.base.wires, ∃ t ∈ cidx ks (L.base.legChain w.1), x = ((w.1, t), (w.2, t))) ∨
      ∃ l ∈ L.links, x ∈ L.linkPairs ks l := by
  simp only [dpairs, List.mem_append, List.mem_flatMap, List.mem_map]
  constructor
  · rintro (⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩)
    · exact Or.inl ⟨w, hw, t, ht, rfl⟩
    · exact Or.inr ⟨l, hl, hx⟩
  · rintro (⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩)
    · exact Or.inl ⟨w, hw, t, ht, rfl⟩
    · exact Or.inr ⟨l, hl, hx⟩

theorem wire_mem_dpairs (L : LDiagram) (ks : ℕ → ℕ) (w : Port × Port) (hw : w ∈ L.base.wires)
    (t : List ℕ) (ht : t ∈ cidx ks (L.base.legChain w.1)) : ((w.1, t), (w.2, t)) ∈ L.dpairs ks :=
  (mem_dpairs L ks _).2 (Or.inl ⟨w, hw, t, ht, rfl⟩)

theorem link_mem_dpairs (L : LDiagram) (ks : ℕ → ℕ) (l : DLink) (hl : l ∈ L.links)
    (x : Dig × Dig) (hx : x ∈ L.linkPairs ks l) : x ∈ L.dpairs ks :=
  (mem_dpairs L ks _).2 (Or.inr ⟨l, hl, hx⟩)

/-- A wire of `L'` (either way round) at the digit `u` of its first end's chain. -/
theorem wireIn_mem (L' : LDiagram) (ks : ℕ → ℕ) (a b : Port) (hw : WireIn L'.base.wires (a, b))
    (u : List ℕ) (ha : u ∈ cidx ks (L'.base.legChain a)) (hb : u ∈ cidx ks (L'.base.legChain b)) :
    ((a, u), (b, u)) ∈ L'.dpairs ks ∨ pswap ((a, u), (b, u)) ∈ L'.dpairs ks := by
  rcases hw with hw | hw
  · exact Or.inl (wire_mem_dpairs L' ks _ hw u ha)
  · exact Or.inr (wire_mem_dpairs L' ks _ hw u hb)

theorem pmap_fix (s : ℕ) (x : Dig × Dig) (h1 : ∃ j u, x.1.2 = j :: u ∧ j ≠ s)
    (h2 : ∃ j u, x.2.2 = j :: u ∧ j ≠ s) : pmap (peelφ B cp s) x = x := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := x
  obtain ⟨j1, u1, e1, hj1⟩ := h1
  obtain ⟨j2, u2, e2, hj2⟩ := h2
  simp only at e1 e2
  subst e1 e2
  simp [pmap, peelφ_ne s p1 j1 u1 hj1, peelφ_ne s p2 j2 u2 hj2]

theorem pmap_out (s : ℕ) (x : Dig × Dig) (h1 : x.1.1.node ∉ B) (h2 : x.2.1.node ∉ B) :
    pmap (peelφ B cp s) x = x := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := x
  simp [pmap, peelφ_O s p1 h1, peelφ_O s p2 h2]

theorem pmap_consS (s : ℕ) (y : Dig × Dig) (h1 : y.1.1.node ∈ B) (h2 : y.2.1.node ∈ B) :
    pmap (peelφ B cp s) (consS s y) = portsP (cpP cp) y := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := y
  simp [pmap, consS, portsP, peelφ_B s p1 h1, peelφ_B s p2 h2]

theorem lp_ports_in (ks : ℕ → ℕ) (d k : ℕ) (cs : List ℕ) (l : DLink) (P : ℕ → Prop) (y : Dig × Dig)
    (hy : y ∈ lp ks d k cs l) (h1 : P l.l1.node) (h2 : P l.l2.node) (h3 : P l.e1.node)
    (h4 : P l.e2.node) : P y.1.1.node ∧ P y.2.1.node := by
  obtain ⟨a, b⟩ := lp_ports ks d k cs l y hy
  exact ⟨by rcases a with e | e <;> rw [e] <;> assumption,
    by rcases b with e | e <;> rw [e] <;> assumption⟩

/-- Every digit pair of `L` goes to a digit pair of `L'`. -/
theorem PeelOK.fwd (h : PeelOK L L' bid mb c B cp) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1)
    (x : Dig × Dig) (hx : x ∈ L.dpairs ks) :
    pmap (peelφ B cp (ks c)) x ∈ L'.dpairs ks ∨
      pswap (pmap (peelφ B cp (ks c)) x) ∈ L'.dpairs ks := by
  rcases (mem_dpairs L ks x).1 hx with ⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩
  · obtain ⟨hw1, hw2, hiff, hin, hcp⟩ := h.wires w hw
    rw [h.lc] at ht
    by_cases hB : w.1.node ∈ B
    · have hB2 := hiff.1 hB
      obtain ⟨c1, c1'⟩ := h.chB' w.1 hw1 hB
      obtain ⟨c2, c2'⟩ := h.chB' w.2 hw2 hB2
      have e12 : L'.base.nodeChain (cp w.1.node) = L'.base.nodeChain (cp w.2.node) := by
        have := h.wch w hw; rw [c1, c2] at this; exact (List.cons.inj this).2
      rw [c1] at ht
      obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
      by_cases hjs : j = ks c
      · subst hjs
        have : pmap (peelφ B cp (ks c)) ((w.1, ks c :: u), (w.2, ks c :: u)) =
            ((cpP cp w.1, u), (cpP cp w.2, u)) := by
          simp [pmap, peelφ_B (ks c) w.1 hB, peelφ_B (ks c) w.2 hB2]
        rw [this]
        exact wireIn_mem L' ks _ _ (hcp hB) u (by rw [h.lc']; exact hu)
          (by rw [h.lc']; show u ∈ cidx ks (L'.base.nodeChain (cp w.2.node)); rw [← e12]; exact hu)
      · rw [pmap_fix (ks c) _ ⟨j, u, rfl, hjs⟩ ⟨j, u, rfl, hjs⟩]
        have hju : j :: u ∈ cidx ks (c :: L'.base.nodeChain (cp w.1.node)) :=
          cons_mem_cidx ks c j _ u (by omega) hu
        exact wireIn_mem L' ks _ _ hin _ (by rw [h.lc', c1']; exact hju)
          (by rw [h.lc', c2', ← e12]; exact hju)
    · have hB2 : w.2.node ∉ B := fun e => hB (hiff.2 e)
      rw [pmap_out (ks c) _ hB hB2]
      exact wireIn_mem L' ks _ _ hin _ (by rw [h.lc', h.chO' w.1 hw1 hB]; exact ht)
        (by rw [h.lc', h.chO' w.2 hw2 hB2, ← h.wch w hw]; exact ht)
  · obtain ⟨hl1, hl2, he1, he2, hc2, hdl, hgd, hce1, hce2⟩ := h.lwf l hl
    rw [h.lp_eq] at hx
    by_cases hb : l.box = bid
    · obtain ⟨hl1B, hl2B, he1B, he2B, hd0, hm, hl'm, hd0', hm', hwire⟩ := h.lb l hl hb
      obtain ⟨c1, c1'⟩ := h.chB' l.l1 hl1 hl1B
      have hL' : ∀ y ∈ lp ks 0 (ks c) (c :: L'.base.nodeChain (cp l.l1.node))
          { l with e1 := cpP cp l.l2 }, y ∈ L'.dpairs ks := fun y hy =>
        link_mem_dpairs L' ks _ hl'm y (by
          rw [h.lp_eq']
          show y ∈ lp ks (L'.depth l.box l.l1.node) (ks (L'.multOf l.box)) (L'.base.nodeChain l.l1.node) _
          rw [hb, hd0', hm', c1']; exact hy)
      rw [hb, hd0, hm, hk, c1] at hx
      rcases (mem_lp ks 0 _ _ l x).1 hx with ⟨t, ht, h1, rfl⟩ | ⟨t, ht, h1, rfl⟩ |
        ⟨t, ht, h1, rfl⟩ | ⟨h1, -⟩
      · obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ t).1 ht
        simp only [List.getD_cons_zero, List.set_cons_zero] at h1 ⊢
        left
        by_cases hj1 : j + 1 = ks c
        · have : pmap (peelφ B cp (ks c)) ((l.l1, j :: u), (l.l2, (j + 1) :: u)) =
              ((l.l1, j :: u), (cpP cp l.l2, u)) := by
            simp only [pmap, peelφ_ne (ks c) l.l1 j u (by omega), hj1, peelφ_B (ks c) l.l2 hl2B u]
          rw [this]
          exact hL' _ ((mem_lp ks 0 _ _ _ _).2 (Or.inr (Or.inl ⟨j :: u,
            cons_mem_cidx ks c j _ u (by omega) hu, by simp; omega, by simp⟩)))
        · rw [pmap_fix (ks c) _ ⟨j, u, rfl, by omega⟩ ⟨j + 1, u, rfl, hj1⟩]
          exact hL' _ ((mem_lp ks 0 _ _ _ _).2 (Or.inl ⟨j :: u,
            cons_mem_cidx ks c j _ u (by omega) hu, by simp; omega, by simp⟩))
      · obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ t).1 ht
        simp only [List.getD_cons_zero, List.eraseIdx_cons_zero] at h1 ⊢
        have hjs : j = ks c := by omega
        subst hjs
        have : pmap (peelφ B cp (ks c)) ((l.l1, ks c :: u), (l.e1, u)) =
            ((cpP cp l.l1, u), (l.e1, u)) := by
          simp only [pmap, peelφ_B (ks c) l.l1 hl1B u, peelφ_O (ks c) l.e1 he1B u]
        rw [this]
        exact wireIn_mem L' ks _ _ hwire u (by rw [h.lc']; exact hu)
          (by rw [h.lc', h.chO' l.e1 he1 he1B, hce1, hb, hd0, c1]; exact hu)
      · obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ t).1 ht
        simp only [List.getD_cons_zero, List.eraseIdx_cons_zero] at h1 ⊢
        subst h1
        left
        by_cases hs0 : ks c = 0
        · have : pmap (peelφ B cp (ks c)) ((l.e2, u), (l.l2, 0 :: u)) =
              ((l.e2, u), (cpP cp l.l2, u)) := by
            have e0 : (0 : ℕ) = ks c := hs0.symm
            simp only [pmap, peelφ_O (ks c) l.e2 he2B u]
            rw [e0, peelφ_B (ks c) l.l2 hl2B u]
          rw [this]
          exact hL' _ ((mem_lp ks 0 _ _ _ _).2 (Or.inr (Or.inr (Or.inr ⟨hs0, u, by simpa using hu,
            rfl⟩))))
        · have : pmap (peelφ B cp (ks c)) ((l.e2, u), (l.l2, 0 :: u)) = ((l.e2, u), (l.l2, 0 :: u)) := by
            simp only [pmap, peelφ_O (ks c) l.e2 he2B u, peelφ_ne (ks c) l.l2 0 u (Ne.symm hs0)]
          rw [this]
          exact hL' _ ((mem_lp ks 0 _ _ _ _).2 (Or.inr (Or.inr (Or.inl ⟨0 :: u,
            cons_mem_cidx ks c 0 _ u (by omega) hu, by simp, by simp⟩))))
      · omega
    · by_cases hin : l.l1.node ∈ B
      · obtain ⟨hl2B, he1B, he2B, hd1, ⟨lt, hltm, r1, r2, r3, r4, hd', hm'⟩, l', hl'm, q1, q2, q3,
          q4, hd'', hm''⟩ := h.lin l hl hb hin
        obtain ⟨c1, c1'⟩ := h.chB' l.l1 hl1 hin
        obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
          ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
        rw [hd0, c1] at hx
        left
        rcases (mem_lp_split ks d0 _ mb c _ l hk x).1 hx with hx | ⟨y, hy, rfl⟩
        · obtain ⟨⟨j1, u1, e1, hj1⟩, ⟨j2, u2, e2, hj2⟩⟩ := lp_heads ks d0 _ c _ l x hx
          rw [pmap_fix (ks c) x ⟨j1, u1, e1, by omega⟩ ⟨j2, u2, e2, by omega⟩]
          exact link_mem_dpairs L' ks lt hltm x (by
            rw [h.lp_eq', r1, hd', hd0, hm', c1', lp_congr ks _ _ _ l lt r1 r2 r3 r4]; exact hx)
        · obtain ⟨p1, p2⟩ := lp_ports_in ks d0 _ _ l (· ∈ B) y hy hin hl2B he1B he2B
          rw [pmap_consS (ks c) y p1 p2]
          refine link_mem_dpairs L' ks l' hl'm _ ?_
          rw [h.lp_eq', q1]
          have hd3 : L'.depth l'.box (cpP cp l.l1).node = d0 := by
            show L'.depth l'.box (cp l.l1.node) = d0; omega
          have hc3 : L'.base.nodeChain (cpP cp l.l1).node = L'.base.nodeChain (cp l.l1.node) := rfl
          rw [hd3, hm'', hc3, lp_congr ks d0 _ _ (l.mapP (cpP cp)) l' q1 q2 q3 q4, lp_mapP]
          exact List.mem_map.2 ⟨y, hy, rfl⟩
      · obtain ⟨hl2O, he1O, he2O, lt, hltm, r1, r2, r3, r4, hd', hm'⟩ := h.lout l hl hb hin
        left
        obtain ⟨p1, p2⟩ := lp_ports_in ks _ _ _ l (· ∉ B) x hx hin hl2O he1O he2O
        rw [pmap_out (ks c) x p1 p2]
        exact link_mem_dpairs L' ks lt hltm x (by
          rw [h.lp_eq', r1, hd', hm', h.chO' l.l1 hl1 hin, lp_congr ks _ _ _ l lt r1 r2 r3 r4]
          exact hx)

/-- Every digit pair of `L'` comes from a digit pair of `L`. -/
theorem PeelOK.bwd (h : PeelOK L L' bid mb c B cp) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1)
    (y : Dig × Dig) (hy : y ∈ L'.dpairs ks) : ∃ x ∈ L.dpairs ks,
      pmap (peelφ B cp (ks c)) x = y ∨ pswap (pmap (peelφ B cp (ks c)) x) = y := by
  rcases (mem_dpairs L' ks y).1 hy with ⟨w, hw, u, hu, rfl⟩ | ⟨l', hl', hy⟩
  · rw [h.lc'] at hu
    rcases h.wires' w hw with hw0 | ⟨w0, hw0, hB, rfl⟩ | ⟨l, hl, hb, e | e⟩
    · obtain ⟨hw1, hw2, hiff, -, -⟩ := h.wires w hw0
      by_cases hB : w.1.node ∈ B
      · obtain ⟨c1, c1'⟩ := h.chB' w.1 hw1 hB
        rw [c1'] at hu
        obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c _ u).1 hu
        refine ⟨((w.1, j :: v), (w.2, j :: v)), wire_mem_dpairs L ks w hw0 _ ?_,
          Or.inl (pmap_fix (ks c) _ ⟨j, v, rfl, by omega⟩ ⟨j, v, rfl, by omega⟩)⟩
        rw [h.lc, c1]; exact cons_mem_cidx ks mb j _ v (by omega) hv
      · refine ⟨((w.1, u), (w.2, u)), wire_mem_dpairs L ks w hw0 _ ?_,
          Or.inl (pmap_out (ks c) _ hB (fun e => hB (hiff.2 e)))⟩
        rw [h.lc, ← h.chO' w.1 hw1 hB]; exact hu
    · obtain ⟨hw1, -, hiff, -, -⟩ := h.wires w0 hw0
      obtain ⟨c1, -⟩ := h.chB' w0.1 hw1 hB
      refine ⟨((w0.1, ks c :: u), (w0.2, ks c :: u)), wire_mem_dpairs L ks w0 hw0 _ ?_,
        Or.inl (pmap_consS (ks c) ((w0.1, u), (w0.2, u)) hB (hiff.1 hB))⟩
      rw [h.lc, c1]; exact cons_mem_cidx ks mb _ _ u (by omega) hu
    · subst e
      obtain ⟨hl1B, -, he1B, -, hd0, hm, -, -, -, -⟩ := h.lb l hl hb
      obtain ⟨hl1, -, -, -, -, -, -, -, -⟩ := h.lwf l hl
      obtain ⟨c1, -⟩ := h.chB' l.l1 hl1 hl1B
      refine ⟨((l.l1, ks c :: u), (l.e1, u)), link_mem_dpairs L ks l hl _ ?_, Or.inl ?_⟩
      · rw [h.lp_eq, hb, hd0, hm, hk, c1]
        exact (mem_lp ks 0 _ _ l _).2 (Or.inr (Or.inl ⟨ks c :: u,
          cons_mem_cidx ks mb _ _ u (by omega) hu, by simp, by simp⟩))
      · simp only [pmap, peelφ_B (ks c) l.l1 hl1B u, peelφ_O (ks c) l.e1 he1B u]
    · subst e
      obtain ⟨hl1B, -, he1B, -, hd0, hm, -, -, -, -⟩ := h.lb l hl hb
      obtain ⟨hl1, -, he1, -, -, -, -, hce1, -⟩ := h.lwf l hl
      obtain ⟨c1, -⟩ := h.chB' l.l1 hl1 hl1B
      have hu' : u ∈ cidx ks (L'.base.nodeChain (cp l.l1.node)) := by
        have := hu
        rw [h.chO' l.e1 he1 he1B, hce1, hb, hd0, c1] at this
        simpa using this
      refine ⟨((l.l1, ks c :: u), (l.e1, u)), link_mem_dpairs L ks l hl _ ?_, Or.inr ?_⟩
      · rw [h.lp_eq, hb, hd0, hm, hk, c1]
        exact (mem_lp ks 0 _ _ l _).2 (Or.inr (Or.inl ⟨ks c :: u,
          cons_mem_cidx ks mb _ _ u (by omega) hu', by simp, by simp⟩))
      · simp only [pswap, pmap, peelφ_B (ks c) l.l1 hl1B u, peelφ_O (ks c) l.e1 he1B u]
  · rcases h.links' l' hl' with ⟨l, hl, hb, rfl⟩ | ⟨l, hl, hb, r1, r2, r3, r4, hd', hm'⟩ |
      ⟨l, hl, hb, hin, q1, q2, q3, q4, hd'', hm''⟩
    · obtain ⟨hl1B, hl2B, he1B, he2B, hd0, hm, -, hd0', hm', -⟩ := h.lb l hl hb
      obtain ⟨hl1, -, -, -, -, -, -, -, -⟩ := h.lwf l hl
      obtain ⟨c1, c1'⟩ := h.chB' l.l1 hl1 hl1B
      have hLx : ∀ x ∈ lp ks 0 (ks c + 1) (mb :: L'.base.nodeChain (cp l.l1.node)) l,
          x ∈ L.dpairs ks := fun x hx =>
        link_mem_dpairs L ks l hl x (by rw [h.lp_eq, hb, hd0, hm, hk, c1]; exact hx)
      rw [h.lp_eq'] at hy
      change y ∈ lp ks (L'.depth l.box l.l1.node) (ks (L'.multOf l.box))
        (L'.base.nodeChain l.l1.node) _ at hy
      rw [hb, hd0', hm', c1'] at hy
      rcases (mem_lp ks 0 _ _ _ y).1 hy with ⟨t, ht, h1, rfl⟩ | ⟨t, ht, h1, rfl⟩ |
        ⟨t, ht, h1, rfl⟩ | ⟨h1, u, hu, rfl⟩
      · obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c _ t).1 ht
        simp only [List.getD_cons_zero, List.set_cons_zero] at h1 ⊢
        refine ⟨((l.l1, j :: u), (l.l2, (j + 1) :: u)), hLx _ ((mem_lp ks 0 _ _ l _).2
          (Or.inl ⟨j :: u, cons_mem_cidx ks mb j _ u (by omega) hu, by simp; omega, by simp⟩)),
          Or.inl ?_⟩
        exact pmap_fix (ks c) _ ⟨j, u, rfl, by omega⟩ ⟨j + 1, u, rfl, by omega⟩
      · obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c _ t).1 ht
        simp only [List.getD_cons_zero, List.eraseIdx_cons_zero] at h1 ⊢
        have hj1 : j + 1 = ks c := by omega
        refine ⟨((l.l1, j :: u), (l.l2, (j + 1) :: u)), hLx _ ((mem_lp ks 0 _ _ l _).2
          (Or.inl ⟨j :: u, cons_mem_cidx ks mb j _ u (by omega) hu, by simp; omega, by simp⟩)),
          Or.inl ?_⟩
        simp only [pmap, peelφ_ne (ks c) l.l1 j u (by omega), hj1, peelφ_B (ks c) l.l2 hl2B u]
      · obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c _ t).1 ht
        simp only [List.getD_cons_zero, List.eraseIdx_cons_zero] at h1 ⊢
        subst h1
        refine ⟨((l.e2, u), (l.l2, 0 :: u)), hLx _ ((mem_lp ks 0 _ _ l _).2
          (Or.inr (Or.inr (Or.inl ⟨0 :: u, cons_mem_cidx ks mb 0 _ u (by omega) hu, by simp,
            by simp⟩)))), Or.inl ?_⟩
        simp only [pmap, peelφ_O (ks c) l.e2 he2B u, peelφ_ne (ks c) l.l2 0 u (by omega)]
      · simp only [List.eraseIdx_cons_zero] at hu
        refine ⟨((l.e2, u), (l.l2, 0 :: u)), hLx _ ((mem_lp ks 0 _ _ l _).2
          (Or.inr (Or.inr (Or.inl ⟨0 :: u, cons_mem_cidx ks mb 0 _ u (by omega) hu, by simp,
            by simp⟩)))), Or.inl ?_⟩
        have e0 : (0 : ℕ) = ks c := h1.symm
        simp only [pmap, peelφ_O (ks c) l.e2 he2B u]
        rw [e0, peelφ_B (ks c) l.l2 hl2B u]
    · obtain ⟨hl1, -, -, -, -, -, -, -, -⟩ := h.lwf l hl
      rw [h.lp_eq', r1, hd', hm', lp_congr ks _ _ _ l l' r1 r2 r3 r4] at hy
      by_cases hin : l.l1.node ∈ B
      · obtain ⟨-, -, -, hd1, -⟩ := h.lin l hl hb hin
        obtain ⟨c1, c1'⟩ := h.chB' l.l1 hl1 hin
        obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
          ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
        rw [hd0, c1'] at hy
        refine ⟨y, link_mem_dpairs L ks l hl y ?_, Or.inl ?_⟩
        · rw [h.lp_eq, hd0, c1]; exact (mem_lp_split ks d0 _ mb c _ l hk y).2 (Or.inl hy)
        · obtain ⟨⟨j1, u1, e1, hj1⟩, ⟨j2, u2, e2, hj2⟩⟩ := lp_heads ks d0 _ c _ l y hy
          exact pmap_fix (ks c) y ⟨j1, u1, e1, by omega⟩ ⟨j2, u2, e2, by omega⟩
      · obtain ⟨hl2O, he1O, he2O, -⟩ := h.lout l hl hb hin
        rw [h.chO' l.l1 hl1 hin] at hy
        obtain ⟨p1, p2⟩ := lp_ports_in ks _ _ _ l (· ∉ B) y hy hin hl2O he1O he2O
        exact ⟨y, link_mem_dpairs L ks l hl y (by rw [h.lp_eq]; exact hy),
          Or.inl (pmap_out (ks c) y p1 p2)⟩
    · obtain ⟨hl1, -, -, -, -, -, -, -, -⟩ := h.lwf l hl
      obtain ⟨hl2B, he1B, he2B, hd1, -⟩ := h.lin l hl hb hin
      obtain ⟨c1, -⟩ := h.chB' l.l1 hl1 hin
      obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
        ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
      rw [h.lp_eq', q1] at hy
      have hd3 : L'.depth l'.box (cpP cp l.l1).node = d0 := by
        show L'.depth l'.box (cp l.l1.node) = d0; omega
      have hc3 : L'.base.nodeChain (cpP cp l.l1).node = L'.base.nodeChain (cp l.l1.node) := rfl
      rw [hd3, hm'', hc3, lp_congr ks d0 _ _ (l.mapP (cpP cp)) l' q1 q2 q3 q4, lp_mapP] at hy
      obtain ⟨y0, hy0, rfl⟩ := List.mem_map.1 hy
      obtain ⟨p1, p2⟩ := lp_ports_in ks d0 _ _ l (· ∈ B) y0 hy0 hin hl2B he1B he2B
      refine ⟨consS (ks c) y0, link_mem_dpairs L ks l hl _ ?_, Or.inl (pmap_consS (ks c) y0 p1 p2)⟩
      rw [h.lp_eq, hd0, c1]
      exact (mem_lp_split ks d0 _ mb c _ l hk _).2 (Or.inr ⟨y0, hy0, rfl⟩)

theorem prod_ite_filter {α : Type*} (l : List α) (P : α → Prop) [DecidablePred P] (f : α → ℂ) :
    (l.map fun a => if P a then f a else 1).prod = ((l.filter fun a => decide (P a)).map f).prod := by
  induction l with
  | nil => rfl
  | cons a l ih => by_cases hp : P a <;> simp [hp, ih, List.filter_cons]

theorem cidx_succ (ks : ℕ → ℕ) (mb c : ℕ) (r : List ℕ) (hk : ks mb = ks c + 1) :
    cidx ks (mb :: r) = cidx ks (c :: r) ++ (cidx ks r).map (ks c :: ·) := by
  simp [cidx, hk, List.range_succ, List.flatMap_append]

/-- The node products agree under the peel's relabelling. -/
theorem PeelOK.prod (h : PeelOK L L' bid mb c B cp) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c + 1) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (peelφ B cp (ks c)) x)) :
    L'.prodL E ks τ = L.prodL E ks τ' := by
  set φ := peelφ B cp (ks c)
  have hv : ∀ n ∈ L.base.nodes, ∀ t ∈ cidx ks (L.base.nodeChain n.id),
      val E τ' (L.dnodeL ks n t) = val E τ ⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        n.dim, n.ph⟩ := by
    intro n hn t ht
    have e : (⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        n.dim, n.ph⟩ : Node) = (L.base.dnode ks n t).rename (relab φ) := by
      rw [L.base.dnode_notail h.ports h.wch]
      simp [Node.rename, List.map_map, Function.comp_def, relab_dlab]
    rw [e, val_rename, L.dnodeL_plain h.cph]
    refine val_reads E _ _ _ fun x hx => ?_
    rw [L.base.dnode_notail h.ports h.wch] at hx
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
      (List.range m.ins.length).map (fun i => dlab ⟨m.id, false, i⟩ t), m.dim, m.ph⟩ := by
    intro m t
    rw [L'.dnodeL_plain h.cph', L'.base.dnode_notail h.ports' h.wch']
  set F' : DNode → ℂ := fun m =>
    ((cidx ks (L'.base.nodeChain m.id)).map fun t => val E τ (L'.dnodeL ks m t)).prod
  have key : ∀ n ∈ L.base.nodes,
      ((cidx ks (L.base.nodeChain n.id)).map fun t => val E τ' (L.dnodeL ks n t)).prod =
        F' n * (if n.id ∈ B then F' (cpNode cp n) else 1) := by
    intro n hn
    by_cases hB : n.id ∈ B
    · obtain ⟨c1, c1'⟩ := h.chB n hn hB
      rw [if_pos hB]
      simp only [F', c1', c1, cidx_succ ks mb c _ hk, List.map_append, List.prod_append,
        List.map_map]
      have hcn : (cpNode cp n).id = cp n.id := rfl
      rw [hcn]
      congr 1
      · refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c _ t).1 ht
        rw [hv n hn _ (by rw [c1, cidx_succ ks mb c _ hk]; exact List.mem_append_left _ ht), hd']
        simp [φ, peelφ_ne (ks c) _ j u (by omega)]
      · refine congrArg List.prod (List.map_congr_left fun u hu => ?_)
        simp only [Function.comp_apply]
        have hm : ks c :: u ∈ cidx ks (L.base.nodeChain n.id) := by
          rw [c1, cidx_succ ks mb c _ hk]; exact List.mem_append_right _ (List.mem_map.2 ⟨u, hu, rfl⟩)
        rw [hv n hn _ hm, hd']
        have hp : ∀ o i, φ ⟨n.id, o, i⟩ (ks c :: u) = (⟨cp n.id, o, i⟩, u) := fun o i =>
          peelφ_B (ks c) ⟨n.id, o, i⟩ hB u
        simp [hp, cpNode]
    · rw [if_neg hB, mul_one]
      simp only [F', h.chO n hn hB]
      refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
      rw [hv n hn _ ht, hd']
      have hp : ∀ o i, φ ⟨n.id, o, i⟩ t = (⟨n.id, o, i⟩, t) := fun o i =>
        peelφ_O (ks c) ⟨n.id, o, i⟩ hB t
      simp [hp]
  unfold prodL
  rw [List.map_congr_left key, List.prod_map_mul, prod_ite_filter, h.nodes, List.map_append,
    List.prod_append, List.map_map]
  rfl

/-- Peeling the last copy of an outermost chain box keeps the linked digit sum, `L`'s slots read
from `L'`'s through `peelφ`. -/
theorem evalL_peel (h : PeelOK L L' bid mb c B cp) (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1)
    (σ' : ℕ → List ℕ → ℕ) :
    L'.evalL E ks σ' = L.evalL E ks (bndMap L.base L'.base (peelφ B cp (ks c)) σ') := by
  refine evalL_relabelφ L L' E ks ks σ' (peelφ B cp (ks c)) (Diagram.dlabs_nodup _ _ h.legsL)
    (Diagram.dlabs_nodup _ _ h.legs) h.bnd ?_ ?_ ?_ ?_ ?_ (fun x hx => h.fwd ks hk x hx)
    (fun y hy => h.bwd ks hk y hy) (fun p hp => (h.bndL p hp).1) ?_ ?_ (fun τ τ' hτ => h.prod E ks hk τ τ' hτ)
  · intro p hp t ht
    rw [h.lc] at ht
    by_cases hB : p.node ∈ B
    · obtain ⟨c1, c1'⟩ := h.chB' p hp hB
      rw [c1] at ht
      obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
      by_cases hjs : j = ks c
      · subst hjs
        rw [peelφ_B (ks c) p hB u]
        exact ⟨h.memC p hp hB, by rw [h.lc']; exact hu⟩
      · rw [peelφ_ne (ks c) p j u hjs]
        exact ⟨h.mem' p hp, by rw [h.lc', c1']; exact cons_mem_cidx ks c j _ u (by omega) hu⟩
    · rw [peelφ_O (ks c) p hB t]
      exact ⟨h.mem' p hp, by rw [h.lc', h.chO' p hp hB]; exact ht⟩
  · intro q hq u hu
    rw [h.legs_eq, List.mem_append] at hq
    rcases hq with hq | hq
    · by_cases hB : q.node ∈ B
      · obtain ⟨c1, c1'⟩ := h.chB' q hq hB
        rw [h.lc', c1'] at hu
        obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c _ _).1 hu
        exact ⟨q, hq, j :: v, by rw [h.lc, c1]; exact cons_mem_cidx ks mb j _ v (by omega) hv,
          peelφ_ne (ks c) q j v (by omega)⟩
      · exact ⟨q, hq, u, by rw [h.lc, ← h.chO' q hq hB, ← h.lc']; exact hu, peelφ_O (ks c) q hB u⟩
    · obtain ⟨n, hn, hq⟩ := List.mem_flatMap.1 hq
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hq
      obtain ⟨hn, hnB⟩ := List.mem_filter.1 hn
      have hpn := node_of_leg n p hp
      have hpB : p.node ∈ B := by rw [hpn]; simpa using hnB
      have hpL : p ∈ L.base.legs := List.mem_flatMap.2 ⟨n, hn, hp⟩
      obtain ⟨c1, -⟩ := h.chB' p hpL hpB
      refine ⟨p, hpL, ks c :: u, ?_, peelφ_B (ks c) p hpB u⟩
      rw [h.lc, c1]
      exact cons_mem_cidx ks mb _ _ u (by omega) (by rw [h.lc'] at hu; exact hu)
  · intro p hp t ht p' hp' t' ht' he
    have nodeB : ∀ q ∈ L.base.legs, q.node ∈ B → ∃ n ∈ L.base.nodes, q.node = n.id ∧ n.id ∈ B :=
      fun q hq hB => by
        obtain ⟨n, hn, e⟩ := node_of_legs _ q hq
        exact ⟨n, hn, e, e ▸ hB⟩
    unfold peelφ at he
    split_ifs at he with h1 h2 h2
    · obtain ⟨n, hn, en, hnB⟩ := nodeB p hp h1.1
      obtain ⟨n', hn', en', hnB'⟩ := nodeB p' hp' h2.1
      simp only [cpP, Prod.mk.injEq, Port.mk.injEq] at he
      obtain ⟨⟨e1, e2, e3⟩, e4⟩ := he
      have := h.cp_inj n hn hnB n' hn' hnB' (by rw [← en, ← en']; exact e1)
      refine ⟨by cases p; cases p'; simp_all, ?_⟩
      obtain ⟨-, ht1⟩ := h1
      obtain ⟨-, ht2⟩ := h2
      cases t <;> cases t' <;> simp_all
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
    unfold peelφ
    split_ifs with h1
    · have hnB : n.id ∈ B := en ▸ h1.1
      have hc : cpNode cp n ∈ L'.base.nodes := by
        rw [h.nodes]
        exact List.mem_append_right _ (List.mem_map.2 ⟨n, List.mem_filter.2 ⟨hn, by simpa using hnB⟩, rfl⟩)
      rw [portDim_of _ h.ids _ hc (cpP cp p) (by simp [cpP, cpNode, en]),
        portDim_of _ h.idsL _ hn p en]
      simp [cpP, cpNode]
    · have hc : n ∈ L'.base.nodes := by rw [h.nodes]; exact List.mem_append_left _ hn
      rw [portDim_of _ h.ids _ hc p en, portDim_of _ h.idsL _ hn p en]
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
    unfold peelφ
    split_ifs with h1
    · exact h3 h1.1
    · exact h2
  · intro q hq u hu
    rcases h.bnd' q hq with hq' | ⟨p, hp, hB, rfl⟩
    · have hql := (h.bndL q hq').1
      by_cases hB : q.node ∈ B
      · obtain ⟨c1, c1'⟩ := h.chB' q hql hB
        rw [h.lc', c1'] at hu
        obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c _ _).1 hu
        exact ⟨q, hq', j :: v, by rw [h.lc, c1]; exact cons_mem_cidx ks mb j _ v (by omega) hv,
          peelφ_ne (ks c) q j v (by omega)⟩
      · exact ⟨q, hq', u, by rw [h.lc, ← h.chO' q hql hB, ← h.lc']; exact hu, peelφ_O (ks c) q hB u⟩
    · have hpl := (h.bndL p hp).1
      obtain ⟨c1, -⟩ := h.chB' p hpl hB
      refine ⟨p, hp, ks c :: u, ?_, peelφ_B (ks c) p hB u⟩
      rw [h.lc, c1]
      exact cons_mem_cidx ks mb _ _ u (by omega) (by rw [h.lc'] at hu; exact hu)

end peel

/-! ## Reversing the copies of an outermost chain box -/

/-- `t` with its first coordinate counted from the far end of `k` copies. -/
def revT (k : ℕ) : List ℕ → List ℕ
  | [] => []
  | j :: u => (k - 1 - j) :: u

theorem revT_mem (ks : ℕ → ℕ) (m : ℕ) (r t : List ℕ) (ht : t ∈ cidx ks (m :: r)) :
    revT (ks m) t ∈ cidx ks (m :: r) := by
  obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks m r t).1 ht
  exact cons_mem_cidx ks m _ r u (by omega) hu

theorem revT_revT (ks : ℕ → ℕ) (m : ℕ) (r t : List ℕ) (ht : t ∈ cidx ks (m :: r)) :
    revT (ks m) (revT (ks m) t) = t := by
  obtain ⟨j, u, rfl, hj, -⟩ := (mem_cidx_cons ks m r t).1 ht
  simp only [revT, List.cons.injEq, and_true]
  omega

theorem cidx_revT_perm (ks : ℕ → ℕ) (m : ℕ) (r : List ℕ) :
    ((cidx ks (m :: r)).map (revT (ks m))).Perm (cidx ks (m :: r)) := by
  refine (List.perm_ext_iff_of_nodup ((cidx_nodup ks _).map_on fun a ha b hb e => ?_)
    (cidx_nodup ks _)).2 fun t => ?_
  · rw [← revT_revT ks m r a ha, e, revT_revT ks m r b hb]
  · constructor
    · intro h
      obtain ⟨s, hs, rfl⟩ := List.mem_map.1 h
      exact revT_mem ks m r s hs
    · intro h
      exact List.mem_map.2 ⟨revT (ks m) t, revT_mem ks m r t h, revT_revT ks m r t h⟩

/-- A link with its two ports and its two ends swapped. -/
def flipL (l : DLink) : DLink := ⟨l.box, l.l2, l.l1, l.e2, l.e1⟩

/-- The links with those of the box `bid` flipped. -/
def flipLinks (bid : ℕ) (ls : List DLink) : List DLink :=
  ls.map fun l => if l.box = bid then flipL l else l

/-- A leg on `B` at copy `t` of the outermost box of `k` copies reads copy `revT k t`. -/
def revφ (B : List ℕ) (k : ℕ) (p : Port) (t : List ℕ) : Dig :=
  if p.node ∈ B then (p, revT k t) else (p, t)

theorem revφ_fst (B : List ℕ) (k : ℕ) (p : Port) (t : List ℕ) : (revφ B k p t).1 = p := by
  unfold revφ; split <;> rfl

/-- The links between two copies of each nested box in `B` are kept by `revφ`: the pairs of a
link at coordinate `d + 1` are those at coordinate `d` with a first coordinate put before. -/
theorem mem_lp_cons (ks : ℕ → ℕ) (d k m : ℕ) (r : List ℕ) (l : DLink) (x : Dig × Dig) :
    x ∈ lp ks (d + 1) k (m :: r) l ↔ ∃ j < ks m, ∃ y ∈ lp ks d k r l, x = consS j y := by
  rw [mem_lp]
  simp only [List.eraseIdx_cons_succ, mem_cidx_cons, List.getD_cons_succ, List.set_cons_succ]
  constructor
  · rintro (⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ |
      ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩ | ⟨h1, _, ⟨j, u, rfl, hj, hu⟩, rfl⟩)
    · exact ⟨j, hj, _, (mem_lp ks d k r l _).2 (Or.inl ⟨u, hu, h1, rfl⟩), rfl⟩
    · exact ⟨j, hj, _, (mem_lp ks d k r l _).2 (Or.inr (Or.inl ⟨u, hu, h1, rfl⟩)), rfl⟩
    · exact ⟨j, hj, _, (mem_lp ks d k r l _).2 (Or.inr (Or.inr (Or.inl ⟨u, hu, h1, rfl⟩))), rfl⟩
    · exact ⟨j, hj, _, (mem_lp ks d k r l _).2 (Or.inr (Or.inr (Or.inr ⟨h1, u, hu, rfl⟩))), rfl⟩
  · rintro ⟨j, hj, y, hy, rfl⟩
    rcases (mem_lp ks d k r l y).1 hy with ⟨u, hu, h1, rfl⟩ | ⟨u, hu, h1, rfl⟩ |
      ⟨u, hu, h1, rfl⟩ | ⟨h1, u, hu, rfl⟩
    · exact Or.inl ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩
    · exact Or.inr (Or.inl ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩)
    · exact Or.inr (Or.inr (Or.inl ⟨_, ⟨j, u, rfl, hj, hu⟩, h1, rfl⟩))
    · exact Or.inr (Or.inr (Or.inr ⟨h1, _, ⟨j, u, rfl, hj, hu⟩, rfl⟩))

/-- `L'` is `L` with the links of the outermost node box `bid` (multiplicity index `mb`, nodes
`B`) flipped. -/
structure RevOK (L L' : LDiagram) (bid mb : ℕ) (B : List ℕ) : Prop where
  ports : ∀ y ∈ L.base.boxes, y.ports = []
  wch : ∀ w ∈ L.base.wires, L.base.nodeChain w.1.node = L.base.nodeChain w.2.node
  cph : L.cph = []
  cph' : L'.cph = []
  base : L'.base = L.base
  links : L'.links = flipLinks bid L.links
  ids : (L.base.nodes.map (·.id)).Nodup
  legs : L.base.legs.Nodup
  bnd : L.base.bnd.Nodup
  bl : ∀ p ∈ L.base.bnd, p ∈ L.base.legs
  chB : ∀ n ∈ L.base.nodes, n.id ∈ B → (L.base.nodeChain n.id).head? = some mb
  wires : ∀ w ∈ L.base.wires, w.1 ∈ L.base.legs ∧ w.2 ∈ L.base.legs ∧ (w.1.node ∈ B ↔ w.2.node ∈ B)
  lwf : ∀ l ∈ L.links, l.l1 ∈ L.base.legs ∧ l.l2 ∈ L.base.legs ∧ l.e1 ∈ L.base.legs ∧
    l.e2 ∈ L.base.legs ∧ L.base.nodeChain l.l2.node = L.base.nodeChain l.l1.node ∧
    L.depth l.box l.l1.node < (L.base.nodeChain l.l1.node).length ∧
    (L.base.nodeChain l.l1.node).getD (L.depth l.box l.l1.node) 0 = L.multOf l.box ∧
    L.base.nodeChain l.e1.node = (L.base.nodeChain l.l1.node).eraseIdx (L.depth l.box l.l1.node) ∧
    L.base.nodeChain l.e2.node = (L.base.nodeChain l.l1.node).eraseIdx (L.depth l.box l.l1.node)
  lb : ∀ l ∈ L.links, l.box = bid → l.l1.node ∈ B ∧ l.l2.node ∈ B ∧ l.e1.node ∉ B ∧
    l.e2.node ∉ B ∧ L.depth bid l.l1.node = 0 ∧ L.depth bid l.l2.node = 0 ∧ L.multOf bid = mb
  lin : ∀ l ∈ L.links, l.box ≠ bid → l.l1.node ∈ B → l.l2.node ∈ B ∧ l.e1.node ∈ B ∧
    l.e2.node ∈ B ∧ 1 ≤ L.depth l.box l.l1.node
  lout : ∀ l ∈ L.links, l.box ≠ bid → l.l1.node ∉ B → l.l2.node ∉ B ∧ l.e1.node ∉ B ∧
    l.e2.node ∉ B

section rev

variable {L L' : LDiagram} {bid mb : ℕ} {B : List ℕ}

theorem RevOK.lc (h : RevOK L L' bid mb B) (q : Port) :
    L.base.legChain q = L.base.nodeChain q.node := L.base.legChain_eq h.ports h.wch q

theorem RevOK.lc' (h : RevOK L L' bid mb B) (q : Port) :
    L'.base.legChain q = L.base.nodeChain q.node := by rw [h.base, h.lc]

theorem RevOK.depth_eq (h : RevOK L L' bid mb B) : L'.depth = L.depth := by
  funext y x; simp only [depth, h.base]

theorem RevOK.multOf_eq (h : RevOK L L' bid mb B) : L'.multOf = L.multOf := by
  funext y; simp only [multOf, h.base]

theorem RevOK.chBn (h : RevOK L L' bid mb B) (n : DNode) (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) :
    ∃ r, L.base.nodeChain n.id = mb :: r := by
  have e := h.chB n hn hB
  cases hc : L.base.nodeChain n.id with
  | nil => rw [hc] at e; simp at e
  | cons a r =>
    rw [hc] at e
    simp only [List.head?_cons, Option.some.injEq] at e
    exact ⟨r, by rw [e]⟩

theorem RevOK.chB' (h : RevOK L L' bid mb B) (p : Port) (hp : p ∈ L.base.legs) (hB : p.node ∈ B) :
    ∃ r, L.base.nodeChain p.node = mb :: r := by
  obtain ⟨n, hn, e⟩ := node_of_legs _ p hp
  rw [e] at hB ⊢
  exact h.chBn n hn hB

theorem RevOK.flip_mem (h : RevOK L L' bid mb B) (l : DLink) (hl : l ∈ L.links)
    (hb : l.box = bid) : flipL l ∈ L'.links := by
  rw [h.links]; exact List.mem_map.2 ⟨l, hl, by simp [hb]⟩

theorem RevOK.keep_mem (h : RevOK L L' bid mb B) (l : DLink) (hl : l ∈ L.links)
    (hb : l.box ≠ bid) : l ∈ L'.links := by
  rw [h.links]; exact List.mem_map.2 ⟨l, hl, by simp [hb]⟩

theorem RevOK.lp_eq (h : RevOK L L' bid mb B) (ks : ℕ → ℕ) (l : DLink) :
    L.linkPairs ks l = lp ks (L.depth l.box l.l1.node) (ks (L.multOf l.box))
      (L.base.nodeChain l.l1.node) l := by
  rw [linkPairs, h.lc]

theorem RevOK.lp_eq' (h : RevOK L L' bid mb B) (ks : ℕ → ℕ) (l : DLink) :
    L'.linkPairs ks l = lp ks (L.depth l.box l.l1.node) (ks (L.multOf l.box))
      (L.base.nodeChain l.l1.node) l := by
  rw [linkPairs, h.depth_eq, h.multOf_eq, h.lc']

theorem pmap_rev_consS (k j : ℕ) (y : Dig × Dig) (h1 : y.1.1.node ∈ B) (h2 : y.2.1.node ∈ B) :
    pmap (revφ B k) (consS j y) = consS (k - 1 - j) y := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := y
  simp only at h1 h2
  simp [pmap, consS, revφ, revT, h1, h2]

theorem pmap_rev_out (k : ℕ) (x : Dig × Dig) (h1 : x.1.1.node ∉ B) (h2 : x.2.1.node ∉ B) :
    pmap (revφ B k) x = x := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := x
  simp only at h1 h2
  simp only [pmap, revφ, if_neg h1, if_neg h2]

/-- The legs and copy indices of `L`'s digit pairs. -/
theorem RevOK.pl (h : RevOK L L' bid mb B) (ks : ℕ → ℕ) (x : Dig × Dig) (hx : x ∈ L.dpairs ks) :
    x.1.1 ∈ L.base.legs ∧ x.1.2 ∈ cidx ks (L.base.legChain x.1.1) ∧
      x.2.1 ∈ L.base.legs ∧ x.2.2 ∈ cidx ks (L.base.legChain x.2.1) := by
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

theorem RevOK.inv (h : RevOK L L' bid mb B) (ks : ℕ → ℕ) (p : Port) (hp : p ∈ L.base.legs)
    (t : List ℕ) (ht : t ∈ cidx ks (L.base.legChain p)) :
    revφ B (ks mb) (revφ B (ks mb) p t).1 (revφ B (ks mb) p t).2 = (p, t) := by
  by_cases hB : p.node ∈ B
  · obtain ⟨r, c1⟩ := h.chB' p hp hB
    rw [h.lc, c1] at ht
    simp only [revφ, if_pos hB, revT_revT ks mb r t ht]
  · simp only [revφ, if_neg hB]

/-- Every digit pair of `L` goes to a digit pair of `L'`. -/
theorem RevOK.fwd (h : RevOK L L' bid mb B) (ks : ℕ → ℕ) (x : Dig × Dig) (hx : x ∈ L.dpairs ks) :
    pmap (revφ B (ks mb)) x ∈ L'.dpairs ks ∨ pswap (pmap (revφ B (ks mb)) x) ∈ L'.dpairs ks := by
  have hw' : ∀ w ∈ L.base.wires, w ∈ L'.base.wires := fun w hw => by rw [h.base]; exact hw
  rcases (mem_dpairs L ks x).1 hx with ⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩
  · obtain ⟨hw1, -, hiff⟩ := h.wires w hw
    rw [h.lc] at ht
    left
    by_cases hB : w.1.node ∈ B
    · have hB2 := hiff.1 hB
      obtain ⟨r, c1⟩ := h.chB' w.1 hw1 hB
      have e : pmap (revφ B (ks mb)) ((w.1, t), (w.2, t)) =
          ((w.1, revT (ks mb) t), (w.2, revT (ks mb) t)) := by
        simp only [pmap, revφ, if_pos hB, if_pos hB2]
      rw [e]
      refine wire_mem_dpairs L' ks w (hw' w hw) _ ?_
      rw [h.lc', c1]
      rw [c1] at ht
      exact revT_mem ks mb r t ht
    · have hB2 : w.2.node ∉ B := fun e => hB (hiff.2 e)
      have e : pmap (revφ B (ks mb)) ((w.1, t), (w.2, t)) = ((w.1, t), (w.2, t)) := by
        simp only [pmap, revφ, if_neg hB, if_neg hB2]
      rw [e]
      exact wire_mem_dpairs L' ks w (hw' w hw) _ (by rw [h.lc']; exact ht)
  · obtain ⟨hl1, -, -, -, hc2, -, -, -, -⟩ := h.lwf l hl
    rw [h.lp_eq] at hx
    by_cases hb : l.box = bid
    · obtain ⟨hl1B, hl2B, he1B, he2B, hd0, hd0', hm⟩ := h.lb l hl hb
      obtain ⟨r, c1⟩ := h.chB' l.l1 hl1 hl1B
      have hL' : ∀ y ∈ lp ks 0 (ks mb) (mb :: r) (flipL l), y ∈ L'.dpairs ks := fun y hy =>
        link_mem_dpairs L' ks _ (h.flip_mem l hl hb) y (by
          rw [h.lp_eq']
          show y ∈ lp ks (L.depth l.box l.l2.node) (ks (L.multOf l.box))
            (L.base.nodeChain l.l2.node) _
          rw [hb, hd0', hm, hc2, c1]; exact hy)
      rw [hb, hd0, hm, c1] at hx
      right
      rcases (mem_lp ks 0 _ _ l x).1 hx with ⟨t, ht, h1, rfl⟩ | ⟨t, ht, h1, rfl⟩ |
        ⟨t, ht, h1, rfl⟩ | ⟨h1, u, hu, rfl⟩
      · obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ t).1 ht
        simp only [List.getD_cons_zero, List.set_cons_zero] at h1 ⊢
        have e : pswap (pmap (revφ B (ks mb)) ((l.l1, j :: u), (l.l2, (j + 1) :: u))) =
            ((l.l2, (ks mb - 1 - (j + 1)) :: u), (l.l1, (ks mb - 1 - j) :: u)) := by
          simp only [pswap, pmap, revφ, if_pos hl1B, if_pos hl2B, revT]
        rw [e]
        have e2 : ks mb - 1 - (j + 1) + 1 = ks mb - 1 - j := by omega
        refine hL' _ ((mem_lp ks 0 _ _ _ _).2 (Or.inl ⟨(ks mb - 1 - (j + 1)) :: u,
          cons_mem_cidx ks mb _ _ u (by omega) hu, by simp only [List.getD_cons_zero]; omega, ?_⟩))
        simp only [flipL, List.getD_cons_zero, List.set_cons_zero, e2]
      · obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ t).1 ht
        simp only [List.getD_cons_zero, List.eraseIdx_cons_zero] at h1 ⊢
        have hz : ks mb - 1 - j = 0 := by omega
        have e : pswap (pmap (revφ B (ks mb)) ((l.l1, j :: u), (l.e1, u))) =
            ((l.e1, u), (l.l1, 0 :: u)) := by
          simp only [pswap, pmap, revφ, if_pos hl1B, if_neg he1B, revT, hz]
        rw [e]
        exact hL' _ ((mem_lp ks 0 _ _ _ _).2 (Or.inr (Or.inr (Or.inl ⟨0 :: u,
          cons_mem_cidx ks mb 0 _ u (by omega) hu, by simp only [List.getD_cons_zero],
          by simp only [flipL, List.eraseIdx_cons_zero]⟩))))
      · obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ t).1 ht
        simp only [List.getD_cons_zero, List.eraseIdx_cons_zero] at h1 ⊢
        subst h1
        have e : pswap (pmap (revφ B (ks mb)) ((l.e2, u), (l.l2, 0 :: u))) =
            ((l.l2, (ks mb - 1) :: u), (l.e2, u)) := by
          simp only [pswap, pmap, revφ, if_neg he2B, if_pos hl2B, revT, Nat.sub_zero]
        rw [e]
        exact hL' _ ((mem_lp ks 0 _ _ _ _).2 (Or.inr (Or.inl ⟨(ks mb - 1) :: u,
          cons_mem_cidx ks mb _ _ u (by omega) hu, by simp only [List.getD_cons_zero]; omega,
          by simp only [flipL, List.eraseIdx_cons_zero]⟩)))
      · have e : pswap (pmap (revφ B (ks mb)) ((l.e2, u), (l.e1, u))) = ((l.e1, u), (l.e2, u)) := by
          simp only [pswap, pmap, revφ, if_neg he2B, if_neg he1B]
        rw [e]
        exact hL' _ ((mem_lp ks 0 _ _ _ _).2 (Or.inr (Or.inr (Or.inr ⟨h1, u, hu,
          by simp only [flipL]⟩))))
    · have hl' := h.keep_mem l hl hb
      by_cases hin : l.l1.node ∈ B
      · obtain ⟨hl2B, he1B, he2B, hd1⟩ := h.lin l hl hb hin
        obtain ⟨r, c1⟩ := h.chB' l.l1 hl1 hin
        obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
          ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
        rw [hd0, c1] at hx
        obtain ⟨j, hj, y, hy, rfl⟩ := (mem_lp_cons ks d0 _ mb r l x).1 hx
        obtain ⟨p1, p2⟩ := lp_ports_in ks d0 _ _ l (· ∈ B) y hy hin hl2B he1B he2B
        left
        rw [pmap_rev_consS (ks mb) j y p1 p2]
        refine link_mem_dpairs L' ks l hl' _ ?_
        rw [h.lp_eq', hd0, c1]
        exact (mem_lp_cons ks d0 _ mb r l _).2 ⟨_, by omega, y, hy, rfl⟩
      · obtain ⟨hl2O, he1O, he2O⟩ := h.lout l hl hb hin
        obtain ⟨p1, p2⟩ := lp_ports_in ks _ _ _ l (· ∉ B) x hx hin hl2O he1O he2O
        left
        rw [pmap_rev_out (ks mb) x p1 p2]
        exact link_mem_dpairs L' ks l hl' x (by rw [h.lp_eq']; exact hx)

/-- Every digit pair of `L'` comes from a digit pair of `L`. -/
theorem RevOK.bwd (h' : RevOK L' L bid mb B) (ks : ℕ → ℕ) (y : Dig × Dig)
    (hy : y ∈ L'.dpairs ks) : ∃ x ∈ L.dpairs ks,
      pmap (revφ B (ks mb)) x = y ∨ pswap (pmap (revφ B (ks mb)) x) = y := by
  obtain ⟨a1, a2, a3, a4⟩ := h'.pl ks y hy
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := y
  simp only at a1 a2 a3 a4
  rcases h'.fwd ks _ hy with hx | hx
  · refine ⟨_, hx, Or.inl ?_⟩
    simp only [pmap]
    rw [h'.inv ks p1 a1 t1 a2, h'.inv ks p2 a3 t2 a4]
  · refine ⟨_, hx, Or.inr ?_⟩
    simp only [pswap, pmap]
    rw [h'.inv ks p1 a1 t1 a2, h'.inv ks p2 a3 t2 a4]

/-- The node products agree under the reversal. -/
theorem RevOK.prod (h : RevOK L L' bid mb B) (E : Env) (ks : ℕ → ℕ) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (revφ B (ks mb)) x)) :
    L'.prodL E ks τ = L.prodL E ks τ' := by
  set φ := revφ B (ks mb)
  have hv : ∀ n ∈ L.base.nodes, ∀ t ∈ cidx ks (L.base.nodeChain n.id),
      val E τ' (L.dnodeL ks n t) = val E τ ⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        n.dim, n.ph⟩ := by
    intro n hn t ht
    have e : (⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        n.dim, n.ph⟩ : Node) = (L.base.dnode ks n t).rename (relab φ) := by
      rw [L.base.dnode_notail h.ports h.wch]
      simp [Node.rename, List.map_map, Function.comp_def, relab_dlab]
    rw [e, val_rename, L.dnodeL_plain h.cph]
    refine val_reads E _ _ _ fun x hx => ?_
    rw [L.base.dnode_notail h.ports h.wch] at hx
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
      (List.range m.ins.length).map (fun i => dlab ⟨m.id, false, i⟩ t), m.dim, m.ph⟩ := by
    intro m t
    rw [L'.dnodeL_plain h.cph', h.base, L.base.dnode_notail h.ports h.wch]
  unfold prodL
  rw [h.base]
  refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
  by_cases hB : n.id ∈ B
  · obtain ⟨r, c1⟩ := h.chBn n hn hB
    rw [c1, ← ((cidx_revT_perm ks mb r).map fun t => val E τ (L'.dnodeL ks n t)).prod_eq,
      List.map_map]
    refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
    simp only [Function.comp_apply]
    rw [hv n hn t (by rw [c1]; exact ht), hd']
    have hp : ∀ o i, φ ⟨n.id, o, i⟩ t = (⟨n.id, o, i⟩, revT (ks mb) t) := fun o i => if_pos hB
    simp [hp]
  · refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
    rw [hv n hn t ht, hd']
    have hp : ∀ o i, φ ⟨n.id, o, i⟩ t = (⟨n.id, o, i⟩, t) := fun o i => if_neg hB
    simp [hp]

/-- Flipping the links of an outermost chain box keeps the linked digit sum, `L`'s slot digits
read from `L'`'s at the copy counted from the far end. -/
theorem evalL_rev (h : RevOK L L' bid mb B) (h' : RevOK L' L bid mb B) (E : Env) (ks : ℕ → ℕ)
    (σ' : ℕ → List ℕ → ℕ) :
    L'.evalL E ks σ' = L.evalL E ks (bndMap L.base L'.base (revφ B (ks mb)) σ') := by
  refine evalL_relabelφ L L' E ks ks σ' (revφ B (ks mb)) (Diagram.dlabs_nodup _ _ h.legs)
    (by rw [h.base]; exact Diagram.dlabs_nodup _ _ h.legs) (by rw [h.base]; exact h.bnd)
    ?_ ?_ ?_ ?_ (fun x hx => h.pl ks x hx) (fun x hx => h.fwd ks x hx)
    (fun y hy => h'.bwd ks y hy) h.bl ?_ ?_ (fun τ τ' hτ => h.prod E ks τ τ' hτ)
  · intro p hp t ht
    rw [h.base]
    by_cases hB : p.node ∈ B
    · obtain ⟨r, c1⟩ := h.chB' p hp hB
      rw [h.lc, c1] at ht
      simp only [revφ, if_pos hB]
      exact ⟨hp, by rw [h.lc, c1]; exact revT_mem ks mb r t ht⟩
    · simp only [revφ, if_neg hB]
      exact ⟨hp, ht⟩
  · intro q hq u hu
    rw [h.base] at hq hu
    by_cases hB : q.node ∈ B
    · obtain ⟨r, c1⟩ := h.chB' q hq hB
      rw [h.lc, c1] at hu
      exact ⟨q, hq, revT (ks mb) u, by rw [h.lc, c1]; exact revT_mem ks mb r u hu,
        by simp only [revφ, if_pos hB, revT_revT ks mb r u hu]⟩
    · exact ⟨q, hq, u, hu, by simp only [revφ, if_neg hB]⟩
  · intro p hp t ht p' hp' t' ht' he
    have e := congrArg (fun z : Dig => revφ B (ks mb) z.1 z.2) he
    simp only [h.inv ks p hp t ht, h.inv ks p' hp' t' ht'] at e
    exact ⟨congrArg Prod.fst e, congrArg Prod.snd e⟩
  · intro p _ t _
    rw [revφ_fst, h.base]
  · intro p hp t _
    rw [revφ_fst, h.base]
    exact hp
  · intro q hq u hu
    rw [h.base] at hq hu
    have hq' := h.bl q hq
    by_cases hB : q.node ∈ B
    · obtain ⟨r, c1⟩ := h.chB' q hq' hB
      rw [h.lc, c1] at hu
      exact ⟨q, hq, revT (ks mb) u, by rw [h.lc, c1]; exact revT_mem ks mb r u hu,
        by simp only [revφ, if_pos hB, revT_revT ks mb r u hu]⟩
    · exact ⟨q, hq, u, hu, by simp only [revφ, if_neg hB]⟩

end rev

end LDiagram

end ArchytasZX
