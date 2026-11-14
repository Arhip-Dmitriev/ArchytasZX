/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedPeel
import ArchytasZX.Iso

/-!
# Splitting an outermost box in two

`evalL_split`: for an outermost node box `bid` over the nodes `B` at `ks c₁ + ks c₂` copies
(multiplicity index `mb`) and a linked diagram `L'` holding `B` at `ks c₁` copies and a copy of `B`
(the nodes `cp x`, outermost box `bid₂`) at `ks c₂` copies (`BoxSplitOK`), `L'` is `L` with `B`'s
copy `j + ks c₁` read at the copy's copy `j`. A leg `e.1` of a top-level Z or X spider outside `B`
bundled over `bid` (an entry `e` of `T`) becomes two legs, `e.1` over `bid` and the appended leg
`e.2` over `bid₂` (`extT`); the spider's value is kept up to its leg order (`BoxSplitOK.tnode`). A
copy-indexed phase `(x, bid, c, c₁)` of `B` reads the copy index of `bid` plus `c * ks c₁`; on the
copy it reads the copy index of `bid₂` plus `(c + 1) * ks c₁`. `bid` has no links; links of boxes inside it hold once on `B` and once on the
copy.
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

/-- The digits of the legs `⟨x, o, i⟩`, `i < len`, of a node `x` at copy `t`, each leg's tail in
`rcidx` order. -/
def dlegs (D : Diagram) (ks : ℕ → ℕ) (x : ℕ) (t : List ℕ) (o : Bool) (len : ℕ) : List ℕ :=
  (List.range len).flatMap fun i =>
    (rcidx ks (D.tail ⟨x, o, i⟩)).map fun s => dlab ⟨x, o, i⟩ (t ++ s)

/-- The number of output (`o`) or input legs of `n`. -/
def legLen (n : DNode) (o : Bool) : ℕ := if o then n.outs.length else n.ins.length

theorem dnode_outs (D : Diagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    (D.dnode ks n t).outs = dlegs D ks n.id t true n.outs.length := rfl

theorem dnode_ins (D : Diagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    (D.dnode ks n t).ins = dlegs D ks n.id t false n.ins.length := rfl

theorem mem_dlegs (D : Diagram) (ks : ℕ → ℕ) (x : ℕ) (t : List ℕ) (o : Bool) (len : ℕ) (y : ℕ) :
    y ∈ dlegs D ks x t o len ↔
      ∃ i < len, ∃ s ∈ cidx ks (D.tail ⟨x, o, i⟩), y = dlab ⟨x, o, i⟩ (t ++ s) := by
  simp only [dlegs, List.mem_flatMap, List.mem_range, List.mem_map, mem_rcidx]
  constructor
  · rintro ⟨i, hi, s, hs, rfl⟩; exact ⟨i, hi, s, hs, rfl⟩
  · rintro ⟨i, hi, s, hs, rfl⟩; exact ⟨i, hi, s, hs, rfl⟩

theorem rcidx_nodup (ks : ℕ → ℕ) (c : List ℕ) : (rcidx ks c).Nodup :=
  (cidx_nodup ks _).map fun _ _ e => List.reverse_injective e

theorem dlegs_nodup (D : Diagram) (ks : ℕ → ℕ) (x : ℕ) (t : List ℕ) (o : Bool) (len : ℕ) :
    (dlegs D ks x t o len).Nodup := by
  unfold dlegs
  refine List.nodup_flatMap.2 ⟨fun i _ => (rcidx_nodup ks _).map fun a b e =>
    List.append_cancel_left (dlab_inj e).2, ?_⟩
  refine List.nodup_range.pairwise_of_forall_ne fun i _ j _ hij => ?_
  simp only [Function.onFun, List.disjoint_left, List.mem_map]
  rintro _ ⟨a, -, rfl⟩ ⟨b, -, e⟩
  exact hij (by simpa using (dlab_inj e).1.symm)

/-- The digits of a node's legs renamed leg by leg. -/
theorem dlegs_rename (D D' : Diagram) (ks : ℕ → ℕ) (φ : Port → List ℕ → Dig) (x x' : ℕ)
    (t t' : List ℕ) (o : Bool) (len : ℕ)
    (h : ∀ i < len, D'.tail ⟨x', o, i⟩ = D.tail ⟨x, o, i⟩ ∧
      ∀ s, φ ⟨x, o, i⟩ (t ++ s) = (⟨x', o, i⟩, t' ++ s)) :
    (dlegs D ks x t o len).map (relab φ) = dlegs D' ks x' t' o len := by
  unfold dlegs
  rw [List.map_flatMap]
  refine List.flatMap_congr fun i hi => ?_
  obtain ⟨h1, h2⟩ := h i (List.mem_range.1 hi)
  rw [h1, List.map_map]
  refine List.map_congr_left fun s _ => ?_
  simp [relab_dlab, h2]

theorem mem_legs_iff (D : Diagram) (hid : (D.nodes.map (·.id)).Nodup) (n : DNode)
    (hn : n ∈ D.nodes) (o : Bool) (i : ℕ) : (⟨n.id, o, i⟩ : Port) ∈ D.legs ↔ i < legLen n o := by
  constructor
  · intro h
    obtain ⟨m, hm, hp⟩ := List.mem_flatMap.1 h
    have e : n.id = m.id := LDiagram.node_of_leg m _ hp
    have : m = n := List.inj_on_of_nodup_map hid hm hn e.symm
    subst this
    simp only [DNode.legs, List.mem_append, List.mem_map, List.mem_range] at hp
    rcases hp with ⟨j, hj, e⟩ | ⟨j, hj, e⟩
    · cases e; simpa [legLen] using hj
    · cases e; simpa [legLen] using hj
  · intro h
    refine List.mem_flatMap.2 ⟨n, hn, ?_⟩
    cases o <;> simp_all [DNode.legs, legLen]

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

/-- A leg the split moves: on a node of `B`, or the first leg of an entry of `T`. -/
def SplitL (B : List ℕ) (T : List (Port × Port)) (p : Port) : Prop :=
  p.node ∈ B ∨ p ∈ T.map Prod.fst

instance (B : List ℕ) (T : List (Port × Port)) (p : Port) : Decidable (SplitL B T p) := by
  unfold SplitL; infer_instance

/-- The leg of `L'` taking a moved leg's copies from the first box's copy count on: the copy's
leg on `B`, the second leg of its `T` entry otherwise. -/
def cq (B : List ℕ) (cp : ℕ → ℕ) (T : List (Port × Port)) (p : Port) : Port :=
  if p.node ∈ B then cpP cp p else ((T.find? fun e => decide (e.1 = p)).map Prod.snd).getD p

/-- The split's map of digits: a moved leg at a copy `j ≥ K` of the outermost box goes to `cq`
at `j - K`. -/
def splitφ (B : List ℕ) (cp : ℕ → ℕ) (T : List (Port × Port)) (K : ℕ) (p : Port) :
    List ℕ → Dig
  | [] => (p, [])
  | j :: u => if SplitL B T p ∧ K ≤ j then (cq B cp T p, (j - K) :: u) else (p, j :: u)

/-- The node `n` with a leg appended, of the same dimension, for each entry of `T` on it. -/
def extT (T : List (Port × Port)) (n : DNode) : DNode :=
  { n with
    outs := n.outs ++ ((T.filter fun e => decide (e.1.node = n.id) && e.1.out).map
      fun e => n.outs.getD e.1.pos 0)
    ins := n.ins ++ ((T.filter fun e => decide (e.1.node = n.id) && !e.1.out).map
      fun e => n.ins.getD e.1.pos 0) }

/-- `L'` is `L` with the outermost node box `bid` (multiplicity index `mb`, nodes `B`) left at the
index `c₁` and copied into the nodes `cp x` under the outermost box `bid₂` at the index `c₂`; each
entry `e` of `T` a leg `e.1` of a top-level Z or X spider bundled over `bid`, its part over `bid₂`
the appended leg `e.2`. -/
structure BoxSplitOK (L L' : LDiagram) (bid bid₂ mb c₁ c₂ : ℕ) (B : List ℕ) (cp : ℕ → ℕ)
    (T : List (Port × Port)) : Prop where
  cph : ∀ q ∈ L.cph, (∃ n ∈ L.base.nodes, n.id = q.1) ∧ (q.1 ∈ B → q.2.1 = bid ∧ q.2.2.2 = c₁)
  cph' : L'.cph = L.cph ++ (L.cph.filter fun q => decide (q.1 ∈ B)).map
    fun q => (cp q.1, bid₂, q.2.2.1 + 1, c₁)
  cphO : ∀ q ∈ L.cph, q.1 ∉ B → L'.depth q.2.1 q.1 = L.depth q.2.1 q.1
  nodes : L'.base.nodes = L.base.nodes.map (extT T) ++
    (L.base.nodes.filter fun n => n.id ∈ B).map (cpNode cp)
  ids : (L'.base.nodes.map (·.id)).Nodup
  legsL : L.base.legs.Nodup
  legs : L'.base.legs.Nodup
  bnd : L'.base.bnd.Nodup
  chB : ∀ n ∈ L.base.nodes, n.id ∈ B →
    L.base.nodeChain n.id = mb :: (L.base.nodeChain n.id).tail ∧
      L'.base.nodeChain n.id = c₁ :: (L.base.nodeChain n.id).tail ∧
      L'.base.nodeChain (cp n.id) = c₂ :: (L.base.nodeChain n.id).tail ∧
      L.depth bid n.id = 0 ∧ L'.depth bid n.id = 0 ∧ L'.depth bid₂ (cp n.id) = 0
  chO : ∀ n ∈ L.base.nodes, n.id ∉ B → L'.base.nodeChain n.id = L.base.nodeChain n.id
  tl : ∀ e ∈ T, e.1 ∈ L.base.legs ∧ e.1.node ∉ B ∧ L.base.nodeChain e.1.node = [] ∧
    e.2.node = e.1.node ∧ e.2.out = e.1.out ∧
    ∃ n ∈ L.base.nodes, n.id = e.1.node ∧ (n.kind = .Z ∨ n.kind = .X)
  legS : ∀ p ∈ L.base.legs, SplitL B T p →
    L.base.legChain p = mb :: (L.base.legChain p).tail ∧
      L'.base.legChain p = c₁ :: (L.base.legChain p).tail ∧ cq B cp T p ∈ L'.base.legs ∧
      L'.base.legChain (cq B cp T p) = c₂ :: (L.base.legChain p).tail ∧
      L'.base.portDim (cq B cp T p) = L.base.portDim p
  legO : ∀ p ∈ L.base.legs, ¬ SplitL B T p → L'.base.legChain p = L.base.legChain p
  legL : ∀ p ∈ L.base.legs, p ∈ L'.base.legs ∧ L'.base.portDim p = L.base.portDim p
  legs' : ∀ q ∈ L'.base.legs, q ∈ L.base.legs ∨
    ∃ p ∈ L.base.legs, SplitL B T p ∧ q = cq B cp T p
  cqinj : ∀ p ∈ L.base.legs, SplitL B T p → ∀ p' ∈ L.base.legs, SplitL B T p' →
    cq B cp T p = cq B cp T p' → p = p'
  cqnew : ∀ p ∈ L.base.legs, SplitL B T p → cq B cp T p ∉ L.base.legs
  wires : ∀ w ∈ L.base.wires, w.1 ∈ L.base.legs ∧ w.2 ∈ L.base.legs ∧
    (SplitL B T w.1 ↔ SplitL B T w.2) ∧ L.base.legChain w.2 = L.base.legChain w.1 ∧
    WireIn L'.base.wires w ∧ (SplitL B T w.1 → WireIn L'.base.wires (cq B cp T w.1, cq B cp T w.2))
  wires' : ∀ w ∈ L'.base.wires, WireIn L.base.wires w ∨
    ∃ w0 ∈ L.base.wires, SplitL B T w0.1 ∧ WireIn [(cq B cp T w0.1, cq B cp T w0.2)] w
  lwf : ∀ l ∈ L.links, l.l1 ∈ L.base.legs ∧ l.l2 ∈ L.base.legs ∧ l.e1 ∈ L.base.legs ∧
    l.e2 ∈ L.base.legs ∧ L.base.legChain l.l2 = L.base.legChain l.l1 ∧
    L.depth l.box l.l1.node < (L.base.legChain l.l1).length ∧
    (L.base.legChain l.l1).getD (L.depth l.box l.l1.node) 0 = L.multOf l.box ∧
    L.base.legChain l.e1 = (L.base.legChain l.l1).eraseIdx (L.depth l.box l.l1.node) ∧
    L.base.legChain l.e2 = (L.base.legChain l.l1).eraseIdx (L.depth l.box l.l1.node)
  lin : ∀ l ∈ L.links, SplitL B T l.l1 → SplitL B T l.l2 ∧ SplitL B T l.e1 ∧
    SplitL B T l.e2 ∧ 1 ≤ L.depth l.box l.l1.node ∧
    (∃ l' ∈ L'.links, l'.l1 = l.l1 ∧ l'.l2 = l.l2 ∧ l'.e1 = l.e1 ∧ l'.e2 = l.e2 ∧
      L'.depth l'.box l.l1.node = L.depth l.box l.l1.node ∧ L'.multOf l'.box = L.multOf l.box) ∧
    ∃ l' ∈ L'.links, l'.l1 = cq B cp T l.l1 ∧ l'.l2 = cq B cp T l.l2 ∧
      l'.e1 = cq B cp T l.e1 ∧ l'.e2 = cq B cp T l.e2 ∧
      L'.depth l'.box (cq B cp T l.l1).node = L.depth l.box l.l1.node ∧
      L'.multOf l'.box = L.multOf l.box
  lout : ∀ l ∈ L.links, ¬ SplitL B T l.l1 → ¬ SplitL B T l.l2 ∧ ¬ SplitL B T l.e1 ∧
    ¬ SplitL B T l.e2 ∧ ∃ l' ∈ L'.links, l'.l1 = l.l1 ∧ l'.l2 = l.l2 ∧ l'.e1 = l.e1 ∧
      l'.e2 = l.e2 ∧ L'.depth l'.box l.l1.node = L.depth l.box l.l1.node ∧
      L'.multOf l'.box = L.multOf l.box
  links' : ∀ l' ∈ L'.links,
    (∃ l ∈ L.links, l'.l1 = l.l1 ∧ l'.l2 = l.l2 ∧ l'.e1 = l.e1 ∧ l'.e2 = l.e2 ∧
      L'.depth l'.box l.l1.node = L.depth l.box l.l1.node ∧ L'.multOf l'.box = L.multOf l.box) ∨
    (∃ l ∈ L.links, SplitL B T l.l1 ∧ l'.l1 = cq B cp T l.l1 ∧ l'.l2 = cq B cp T l.l2 ∧
      l'.e1 = cq B cp T l.e1 ∧ l'.e2 = cq B cp T l.e2 ∧
      L'.depth l'.box (cq B cp T l.l1).node = L.depth l.box l.l1.node ∧
      L'.multOf l'.box = L.multOf l.box)
  bndL : ∀ p ∈ L.base.bnd, p ∈ L.base.legs ∧ p ∈ L'.base.bnd ∧
    (SplitL B T p → cq B cp T p ∈ L'.base.bnd)
  bnd' : ∀ q ∈ L'.base.bnd, q ∈ L.base.bnd ∨ ∃ p ∈ L.base.bnd, SplitL B T p ∧ q = cq B cp T p

section split

variable {L L' : LDiagram} {bid bid₂ mb c₁ c₂ : ℕ} {B : List ℕ} {cp : ℕ → ℕ}
  {T : List (Port × Port)}

theorem splitφ_lo (K : ℕ) (p : Port) (j : ℕ) (u : List ℕ) (hj : j < K) :
    splitφ B cp T K p (j :: u) = (p, j :: u) := by
  simp [splitφ, Nat.not_le.2 hj]

theorem splitφ_hi (K : ℕ) (p : Port) (hp : SplitL B T p) (j : ℕ) (u : List ℕ) :
    splitφ B cp T K p ((j + K) :: u) = (cq B cp T p, j :: u) := by
  simp [splitφ, hp]

theorem splitφ_O (K : ℕ) (p : Port) (hp : ¬ SplitL B T p) (t : List ℕ) :
    splitφ B cp T K p t = (p, t) := by
  cases t <;> simp [splitφ, hp]

theorem splitφ_cases (K : ℕ) (p : Port) (t : List ℕ) :
    splitφ B cp T K p t = (p, t) ∨
      ∃ j u, t = (j + K) :: u ∧ SplitL B T p ∧ splitφ B cp T K p t = (cq B cp T p, j :: u) := by
  cases t with
  | nil => exact Or.inl rfl
  | cons j u =>
    by_cases h : SplitL B T p ∧ K ≤ j
    · obtain ⟨j', rfl⟩ : ∃ j', j = j' + K := ⟨j - K, by omega⟩
      exact Or.inr ⟨j', u, rfl, h.1, splitφ_hi K p h.1 j' u⟩
    · left; simp only [splitφ, h, ↓reduceIte]

theorem cq_B (p : Port) (hp : p.node ∈ B) : cq B cp T p = cpP cp p := by
  simp [cq, hp]

theorem cq_T (p : Port) (hp : p.node ∉ B) (hT : p ∈ T.map Prod.fst) :
    ∃ e ∈ T, e.1 = p ∧ cq B cp T p = e.2 := by
  have hf : (T.find? fun e => decide (e.1 = p)).isSome := by
    rw [List.find?_isSome]
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hT
    exact ⟨e, he, by simp⟩
  obtain ⟨e, he⟩ := Option.isSome_iff_exists.1 hf
  have h1 := List.mem_of_find?_eq_some he
  have h2 := List.find?_some he
  simp only [decide_eq_true_eq] at h2
  exact ⟨e, h1, h2, by simp [cq, hp, he]⟩

theorem pmap_lo (K : ℕ) (x : Dig × Dig) (h1 : ∃ j u, x.1.2 = j :: u ∧ j < K)
    (h2 : ∃ j u, x.2.2 = j :: u ∧ j < K) : pmap (splitφ B cp T K) x = x := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := x
  obtain ⟨j1, u1, e1, hj1⟩ := h1
  obtain ⟨j2, u2, e2, hj2⟩ := h2
  simp only at e1 e2
  subst e1 e2
  simp [pmap, splitφ_lo K p1 j1 u1 hj1, splitφ_lo K p2 j2 u2 hj2]

theorem pmap_O (K : ℕ) (x : Dig × Dig) (h1 : ¬ SplitL B T x.1.1) (h2 : ¬ SplitL B T x.2.1) :
    pmap (splitφ B cp T K) x = x := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := x
  simp [pmap, splitφ_O K p1 h1, splitφ_O K p2 h2]

theorem pmap_shift (K : ℕ) (y : Dig × Dig) (h1 : SplitL B T y.1.1) (h2 : SplitL B T y.2.1)
    (e1 : ∃ j u, y.1.2 = j :: u) (e2 : ∃ j u, y.2.2 = j :: u) :
    pmap (splitφ B cp T K) (shiftS K y) = portsP (cq B cp T) y := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := y
  obtain ⟨j1, u1, e1⟩ := e1
  obtain ⟨j2, u2, e2⟩ := e2
  simp only at e1 e2 h1 h2
  subst e1 e2
  simp [pmap, shiftS, portsP, addH, splitφ_hi K p1 h1, splitφ_hi K p2 h2]

/-- The ports of a link's pairs satisfy a property of ports its four ports satisfy. -/
theorem lp_ports_inP (ks : ℕ → ℕ) (d k : ℕ) (cs : List ℕ) (l : DLink) (P : Port → Prop)
    (y : Dig × Dig) (hy : y ∈ lp ks d k cs l) (h1 : P l.l1) (h2 : P l.l2) (h3 : P l.e1)
    (h4 : P l.e2) : P y.1.1 ∧ P y.2.1 := by
  obtain ⟨a, b⟩ := lp_ports ks d k cs l y hy
  exact ⟨by rcases a with e | e <;> rw [e] <;> assumption,
    by rcases b with e | e <;> rw [e] <;> assumption⟩

theorem BoxSplitOK.idsEq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) :
    L'.base.nodes.map (·.id) = L.base.nodes.map (·.id) ++
      ((L.base.nodes.filter fun n => n.id ∈ B).map (cpNode cp)).map (·.id) := by
  rw [h.nodes, List.map_append, List.map_map]
  rfl

theorem BoxSplitOK.idsL (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) :
    (L.base.nodes.map (·.id)).Nodup := by
  have := h.ids
  rw [h.idsEq] at this
  exact this.of_append_left

theorem BoxSplitOK.cp_fresh (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (n : DNode)
    (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) :
    cp n.id ≠ m.id := by
  have := h.ids
  rw [h.idsEq, List.nodup_append] at this
  intro e
  refine this.2.2 m.id (List.mem_map.2 ⟨m, hm, rfl⟩) (cp n.id) ?_ e.symm
  exact List.mem_map.2 ⟨cpNode cp n,
    List.mem_map.2 ⟨n, List.mem_filter.2 ⟨hn, by simpa using hB⟩, rfl⟩, rfl⟩

theorem BoxSplitOK.cp_inj (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (n : DNode)
    (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) (hmB : m.id ∈ B)
    (e : cp n.id = cp m.id) : n = m := by
  have := h.ids
  rw [h.idsEq, List.nodup_append] at this
  have h2 := this.2.1
  rw [List.map_map] at h2
  exact (List.inj_on_of_nodup_map h2) (List.mem_filter.2 ⟨hn, by simpa using hB⟩)
    (List.mem_filter.2 ⟨hm, by simpa using hmB⟩) (by simpa [cpNode] using e)

theorem BoxSplitOK.lS (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (p : Port)
    (hp : p ∈ L.base.legs) (hs : SplitL B T p) : ∃ r, L.base.legChain p = mb :: r ∧
      L'.base.legChain p = c₁ :: r ∧ L'.base.legChain (cq B cp T p) = c₂ :: r := by
  obtain ⟨h1, h2, -, h4, -⟩ := h.legS p hp hs
  exact ⟨_, h1, h2, h4⟩

/-- A leg moved by `splitφ` stays on its node and side off `B`. -/
theorem BoxSplitOK.φ_node (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (K : ℕ) (p : Port)
    (hp : p.node ∉ B) (t : List ℕ) :
    (splitφ B cp T K p t).1.node = p.node ∧ (splitφ B cp T K p t).1.out = p.out := by
  rcases splitφ_cases (B := B) (cp := cp) (T := T) K p t with e | ⟨j, u, -, hs, e⟩
  · rw [e]; exact ⟨rfl, rfl⟩
  · rw [e]
    have hT : p ∈ T.map Prod.fst := hs.resolve_left hp
    obtain ⟨q, hq, hq1, he⟩ := cq_T (B := B) (cp := cp) p hp hT
    simp only
    rw [he]
    obtain ⟨-, -, -, h4, h5, -⟩ := h.tl q hq
    rw [h4, h5, hq1]
    exact ⟨rfl, rfl⟩

/-- A leg of `L` at a digit of `L'`'s chain of it, read back at a digit of its own. -/
theorem BoxSplitOK.onto_same (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (q : Port) (hq : q ∈ L.base.legs) (u : List ℕ)
    (hu : u ∈ cidx ks (L'.base.legChain q)) :
    ∃ t ∈ cidx ks (L.base.legChain q), splitφ B cp T (ks c₁) q t = (q, u) := by
  by_cases hs : SplitL B T q
  · obtain ⟨r, c1, c1', -⟩ := h.lS q hq hs
    rw [c1'] at hu
    obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c₁ _ _).1 hu
    exact ⟨j :: v, by rw [c1]; exact cons_mem_cidx ks mb j _ v (by omega) hv,
      splitφ_lo (ks c₁) q j v hj⟩
  · exact ⟨u, by rw [← h.legO q hq hs]; exact hu, splitφ_O (ks c₁) q hs u⟩

/-- A leg `cq p` of `L'` at a digit of its chain, read back at a digit of `p`. -/
theorem BoxSplitOK.onto_cq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (p : Port) (hp : p ∈ L.base.legs) (hs : SplitL B T p)
    (u : List ℕ) (hu : u ∈ cidx ks (L'.base.legChain (cq B cp T p))) :
    ∃ t ∈ cidx ks (L.base.legChain p), splitφ B cp T (ks c₁) p t = (cq B cp T p, u) := by
  obtain ⟨r, c1, -, c1''⟩ := h.lS p hp hs
  rw [c1''] at hu
  obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c₂ _ _).1 hu
  exact ⟨(j + ks c₁) :: v, by rw [c1]; exact cons_mem_cidx ks mb _ _ v (by omega) hv,
    splitφ_hi (ks c₁) p hs j v⟩

theorem BoxSplitOK.into (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (p : Port) (hp : p ∈ L.base.legs) (t : List ℕ)
    (ht : t ∈ cidx ks (L.base.legChain p)) :
    (splitφ B cp T (ks c₁) p t).1 ∈ L'.base.legs ∧
      (splitφ B cp T (ks c₁) p t).2 ∈
        cidx ks (L'.base.legChain (splitφ B cp T (ks c₁) p t).1) := by
  by_cases hs : SplitL B T p
  · obtain ⟨r, c1, c1', c1''⟩ := h.lS p hp hs
    rw [c1] at ht
    obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
    by_cases hj1 : j < ks c₁
    · rw [splitφ_lo (ks c₁) p j u hj1]
      exact ⟨(h.legL p hp).1, by rw [c1']; exact cons_mem_cidx ks c₁ j _ u hj1 hu⟩
    · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
      rw [splitφ_hi (ks c₁) p hs j' u]
      exact ⟨(h.legS p hp hs).2.2.1,
        by rw [c1'']; exact cons_mem_cidx ks c₂ j' _ u (by omega) hu⟩
  · rw [splitφ_O (ks c₁) p hs t]
    exact ⟨(h.legL p hp).1, by rw [h.legO p hp hs]; exact ht⟩

theorem BoxSplitOK.onto (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (q : Port) (hq : q ∈ L'.base.legs) (u : List ℕ)
    (hu : u ∈ cidx ks (L'.base.legChain q)) :
    ∃ p ∈ L.base.legs, ∃ t ∈ cidx ks (L.base.legChain p), splitφ B cp T (ks c₁) p t = (q, u) := by
  rcases h.legs' q hq with hq' | ⟨p, hp, hs, rfl⟩
  · obtain ⟨t, ht, e⟩ := h.onto_same ks hk q hq' u hu
    exact ⟨q, hq', t, ht, e⟩
  · obtain ⟨t, ht, e⟩ := h.onto_cq ks hk p hp hs u hu
    exact ⟨p, hp, t, ht, e⟩

theorem BoxSplitOK.inj (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (K : ℕ) (p : Port)
    (hp : p ∈ L.base.legs) (t : List ℕ) (p' : Port) (hp' : p' ∈ L.base.legs) (t' : List ℕ)
    (he : splitφ B cp T K p t = splitφ B cp T K p' t') : p = p' ∧ t = t' := by
  rcases splitφ_cases (B := B) (cp := cp) (T := T) K p t with e1 | ⟨j, u, rfl, hs, e1⟩ <;>
  rcases splitφ_cases (B := B) (cp := cp) (T := T) K p' t' with e2 | ⟨j', u', rfl, hs', e2⟩ <;>
  rw [e1, e2] at he <;> simp only [Prod.mk.injEq] at he
  · exact he
  · exact absurd (he.1 ▸ hp) (h.cqnew p' hp' hs')
  · exact absurd (he.1 ▸ hp') (h.cqnew p hp hs)
  · obtain ⟨e3, e4⟩ := he
    obtain ⟨rfl, rfl⟩ := List.cons.inj e4
    exact ⟨h.cqinj p hp hs p' hp' hs' e3, rfl⟩

/-- Every digit pair of `L` goes to a digit pair of `L'`. -/
theorem BoxSplitOK.fwd (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (x : Dig × Dig) (hx : x ∈ L.dpairs ks) :
    pmap (splitφ B cp T (ks c₁)) x ∈ L'.dpairs ks ∨
      pswap (pmap (splitφ B cp T (ks c₁)) x) ∈ L'.dpairs ks := by
  rcases (mem_dpairs L ks x).1 hx with ⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩
  · obtain ⟨hw1, hw2, hiff, hch, hin, hcp⟩ := h.wires w hw
    by_cases hs : SplitL B T w.1
    · have hs2 := hiff.1 hs
      obtain ⟨r, c1, c1', c1''⟩ := h.lS w.1 hw1 hs
      obtain ⟨r2, c2, c2', c2''⟩ := h.lS w.2 hw2 hs2
      have er : r2 = r := by rw [hch, c1] at c2; exact (List.cons.inj c2).2.symm
      subst er
      rw [c1] at ht
      obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
      by_cases hj1 : j < ks c₁
      · rw [pmap_lo (ks c₁) _ ⟨j, u, rfl, hj1⟩ ⟨j, u, rfl, hj1⟩]
        have hju : j :: u ∈ cidx ks (c₁ :: r2) := cons_mem_cidx ks c₁ j _ u hj1 hu
        exact wireIn_mem L' ks _ _ hin _ (by rw [c1']; exact hju) (by rw [c2']; exact hju)
      · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
        have e : pmap (splitφ B cp T (ks c₁)) ((w.1, (j' + ks c₁) :: u),
            (w.2, (j' + ks c₁) :: u)) = ((cq B cp T w.1, j' :: u), (cq B cp T w.2, j' :: u)) := by
          simp [pmap, splitφ_hi (ks c₁) w.1 hs, splitφ_hi (ks c₁) w.2 hs2]
        rw [e]
        have hju : j' :: u ∈ cidx ks (c₂ :: r2) := cons_mem_cidx ks c₂ j' _ u (by omega) hu
        exact wireIn_mem L' ks _ _ (hcp hs) _ (by rw [c1'']; exact hju) (by rw [c2'']; exact hju)
    · have hs2 : ¬ SplitL B T w.2 := fun e => hs (hiff.2 e)
      rw [pmap_O (ks c₁) _ hs hs2]
      exact wireIn_mem L' ks _ _ hin _ (by rw [h.legO w.1 hw1 hs]; exact ht)
        (by rw [h.legO w.2 hw2 hs2, hch]; exact ht)
  · obtain ⟨hl1, hl2, he1, he2, hc2, hdl, hgd, hce1, hce2⟩ := h.lwf l hl
    unfold linkPairs at hx
    by_cases hin : SplitL B T l.l1
    · obtain ⟨hl2B, he1B, he2B, hd1, ⟨lt, hltm, r1, r2, r3, r4, hd', hm'⟩, l', hl'm, q1, q2, q3,
        q4, hd'', hm''⟩ := h.lin l hl hin
      obtain ⟨r, c1, c1', c1''⟩ := h.lS l.l1 hl1 hin
      obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
        ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
      rw [hd0, c1] at hx
      left
      rcases (mem_lp_split2 ks d0 _ mb c₁ c₂ r l hk x).1 hx with hx | ⟨y, hy, rfl⟩
      · obtain ⟨⟨j1, u1, e1, hj1⟩, ⟨j2, u2, e2, hj2⟩⟩ := lp_heads ks d0 _ c₁ r l x hx
        rw [pmap_lo (ks c₁) x ⟨j1, u1, e1, hj1⟩ ⟨j2, u2, e2, hj2⟩]
        refine link_mem_dpairs L' ks lt hltm x ?_
        unfold linkPairs
        rw [r1, hd', hd0, hm', c1', lp_congr ks _ _ _ l lt r1 r2 r3 r4]; exact hx
      · obtain ⟨p1, p2⟩ := lp_ports_inP ks _ _ _ l (SplitL B T) y hy hin hl2B he1B he2B
        obtain ⟨⟨j1, u1, e1, -⟩, ⟨j2, u2, e2, -⟩⟩ := lp_heads ks d0 _ c₂ r l y hy
        rw [pmap_shift (ks c₁) y p1 p2 ⟨j1, u1, e1⟩ ⟨j2, u2, e2⟩]
        refine link_mem_dpairs L' ks l' hl'm _ ?_
        unfold linkPairs
        rw [q1, hd'', hd0, hm'', c1'', lp_congr ks _ _ _ (l.mapP (cq B cp T)) l' q1 q2 q3 q4,
          lp_mapP]
        exact List.mem_map.2 ⟨y, hy, rfl⟩
    · obtain ⟨hl2O, he1O, he2O, lt, hltm, r1, r2, r3, r4, hd', hm'⟩ := h.lout l hl hin
      left
      obtain ⟨p1, p2⟩ :=
        lp_ports_inP ks _ _ _ l (fun p => ¬ SplitL B T p) x hx hin hl2O he1O he2O
      rw [pmap_O (ks c₁) x p1 p2]
      refine link_mem_dpairs L' ks lt hltm x ?_
      unfold linkPairs
      rw [r1, hd', hm', h.legO l.l1 hl1 hin, lp_congr ks _ _ _ l lt r1 r2 r3 r4]
      exact hx

/-- A wire of `L` at a digit of `L'`'s chain of its first end, read back. -/
theorem BoxSplitOK.wire_back (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (v : Port × Port) (hv : v ∈ L.base.wires) (u : List ℕ)
    (hu : u ∈ cidx ks (L'.base.legChain v.1)) :
    ∃ x ∈ L.dpairs ks, pmap (splitφ B cp T (ks c₁)) x = ((v.1, u), (v.2, u)) := by
  obtain ⟨hw1, hw2, hiff, -, -, -⟩ := h.wires v hv
  by_cases hs : SplitL B T v.1
  · obtain ⟨r, c1, c1', -⟩ := h.lS v.1 hw1 hs
    rw [c1'] at hu
    obtain ⟨j, w, rfl, hj, hw⟩ := (mem_cidx_cons ks c₁ _ u).1 hu
    refine ⟨((v.1, j :: w), (v.2, j :: w)), wire_mem_dpairs L ks v hv _ ?_,
      pmap_lo (ks c₁) _ ⟨j, w, rfl, hj⟩ ⟨j, w, rfl, hj⟩⟩
    rw [c1]; exact cons_mem_cidx ks mb j _ w (by omega) hw
  · refine ⟨((v.1, u), (v.2, u)), wire_mem_dpairs L ks v hv _ ?_,
      pmap_O (ks c₁) _ hs (fun e => hs (hiff.2 e))⟩
    rw [← h.legO v.1 hw1 hs]; exact hu

/-- The copy of a wire of `L` at a digit of its chain, read back. -/
theorem BoxSplitOK.wire_back_cq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (v : Port × Port) (hv : v ∈ L.base.wires)
    (hs : SplitL B T v.1) (u : List ℕ) (hu : u ∈ cidx ks (L'.base.legChain (cq B cp T v.1))) :
    ∃ x ∈ L.dpairs ks, pmap (splitφ B cp T (ks c₁)) x =
      ((cq B cp T v.1, u), (cq B cp T v.2, u)) := by
  obtain ⟨hw1, -, hiff, -, -, -⟩ := h.wires v hv
  obtain ⟨r, c1, -, c1''⟩ := h.lS v.1 hw1 hs
  rw [c1''] at hu
  obtain ⟨j, w, rfl, hj, hw⟩ := (mem_cidx_cons ks c₂ _ u).1 hu
  refine ⟨((v.1, (j + ks c₁) :: w), (v.2, (j + ks c₁) :: w)), wire_mem_dpairs L ks v hv _ ?_, ?_⟩
  · rw [c1]; exact cons_mem_cidx ks mb _ _ w (by omega) hw
  · simp [pmap, splitφ_hi (ks c₁) v.1 hs, splitφ_hi (ks c₁) v.2 (hiff.1 hs)]

/-- The two ends of a wire of `L` have one chain in `L'`, and one at their copies. -/
theorem BoxSplitOK.wire_ch (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (v : Port × Port)
    (hv : v ∈ L.base.wires) : L'.base.legChain v.2 = L'.base.legChain v.1 ∧
      (SplitL B T v.1 → L'.base.legChain (cq B cp T v.2) = L'.base.legChain (cq B cp T v.1)) := by
  obtain ⟨hw1, hw2, hiff, hch, -, -⟩ := h.wires v hv
  by_cases hs : SplitL B T v.1
  · obtain ⟨r, c1, c1', c1''⟩ := h.lS v.1 hw1 hs
    obtain ⟨r2, c2, c2', c2''⟩ := h.lS v.2 hw2 (hiff.1 hs)
    have er : r2 = r := by rw [hch, c1] at c2; exact (List.cons.inj c2).2.symm
    subst er
    exact ⟨by rw [c1', c2'], fun _ => by rw [c1'', c2'']⟩
  · exact ⟨by rw [h.legO v.2 hw2 (fun e => hs (hiff.2 e)), h.legO v.1 hw1 hs, hch],
      fun e => absurd e hs⟩

/-- Every digit pair of `L'` comes from a digit pair of `L`. -/
theorem BoxSplitOK.bwd (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (y : Dig × Dig) (hy : y ∈ L'.dpairs ks) : ∃ x ∈ L.dpairs ks,
      pmap (splitφ B cp T (ks c₁)) x = y ∨ pswap (pmap (splitφ B cp T (ks c₁)) x) = y := by
  rcases (mem_dpairs L' ks y).1 hy with ⟨w, hw, u, hu, rfl⟩ | ⟨l', hl', hy⟩
  · have hw' := h.wires' w hw
    simp only [WireIn, List.mem_singleton] at hw'
    rcases hw' with (hw0 | hw0) | ⟨w0, hw0, hs, hw1 | hw1⟩
    · obtain ⟨x, hx, e⟩ := h.wire_back ks hk w hw0 u hu
      exact ⟨x, hx, Or.inl e⟩
    · obtain ⟨x, hx, e⟩ := h.wire_back ks hk (w.2, w.1) hw0 u
        (by have e := (h.wire_ch _ hw0).1; simp only at e ⊢; rw [← e]; exact hu)
      exact ⟨x, hx, Or.inr (by rw [e]; rfl)⟩
    · subst hw1
      obtain ⟨x, hx, e⟩ := h.wire_back_cq ks hk w0 hw0 hs u hu
      exact ⟨x, hx, Or.inl e⟩
    · obtain ⟨w1, w2⟩ := w
      simp only [Prod.mk.injEq] at hw1
      obtain ⟨rfl, rfl⟩ := hw1
      obtain ⟨x, hx, e⟩ := h.wire_back_cq ks hk w0 hw0 hs u (by
        rw [← (h.wire_ch w0 hw0).2 hs]; exact hu)
      exact ⟨x, hx, Or.inr (by rw [e]; rfl)⟩
  · unfold linkPairs at hy
    rcases h.links' l' hl' with ⟨l, hl, r1, r2, r3, r4, hd', hm'⟩ |
      ⟨l, hl, hin, q1, q2, q3, q4, hd'', hm''⟩
    · obtain ⟨hl1, -, -, -, -, -, -, -, -⟩ := h.lwf l hl
      rw [r1, hd', hm', lp_congr ks _ _ _ l l' r1 r2 r3 r4] at hy
      by_cases hin : SplitL B T l.l1
      · obtain ⟨-, -, -, hd1, -⟩ := h.lin l hl hin
        obtain ⟨r, c1, c1', -⟩ := h.lS l.l1 hl1 hin
        obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
          ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
        rw [hd0, c1'] at hy
        refine ⟨y, link_mem_dpairs L ks l hl y ?_, Or.inl ?_⟩
        · unfold linkPairs
          rw [hd0, c1]; exact (mem_lp_split2 ks d0 _ mb c₁ c₂ r l hk y).2 (Or.inl hy)
        · obtain ⟨⟨j1, u1, e1, hj1⟩, ⟨j2, u2, e2, hj2⟩⟩ := lp_heads ks d0 _ c₁ r l y hy
          exact pmap_lo (ks c₁) y ⟨j1, u1, e1, hj1⟩ ⟨j2, u2, e2, hj2⟩
      · obtain ⟨hl2O, he1O, he2O, -⟩ := h.lout l hl hin
        rw [h.legO l.l1 hl1 hin] at hy
        obtain ⟨p1, p2⟩ :=
          lp_ports_inP ks _ _ _ l (fun p => ¬ SplitL B T p) y hy hin hl2O he1O he2O
        exact ⟨y, link_mem_dpairs L ks l hl y (by unfold linkPairs; exact hy),
          Or.inl (pmap_O (ks c₁) y p1 p2)⟩
    · obtain ⟨hl1, -, -, -, -, -, -, -, -⟩ := h.lwf l hl
      obtain ⟨hl2B, he1B, he2B, hd1, -⟩ := h.lin l hl hin
      obtain ⟨r, c1, -, c1''⟩ := h.lS l.l1 hl1 hin
      obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
        ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
      rw [q1, hd'', hd0, hm'', c1'', lp_congr ks _ _ _ (l.mapP (cq B cp T)) l' q1 q2 q3 q4,
        lp_mapP] at hy
      obtain ⟨y0, hy0, rfl⟩ := List.mem_map.1 hy
      obtain ⟨p1, p2⟩ := lp_ports_inP ks _ _ _ l (SplitL B T) y0 hy0 hin hl2B he1B he2B
      obtain ⟨⟨j1, u1, e1, -⟩, ⟨j2, u2, e2, -⟩⟩ := lp_heads ks d0 _ c₂ r l y0 hy0
      refine ⟨shiftS (ks c₁) y0, link_mem_dpairs L ks l hl _ ?_,
        Or.inl (pmap_shift (ks c₁) y0 p1 p2 ⟨j1, u1, e1⟩ ⟨j2, u2, e2⟩)⟩
      unfold linkPairs
      rw [hd0, c1]
      exact (mem_lp_split2 ks d0 _ mb c₁ c₂ r l hk _).2 (Or.inr ⟨y0, hy0, rfl⟩)

theorem phAt_congr (L : LDiagram) (ks : ℕ → ℕ) (n m : DNode) (h1 : m.id = n.id)
    (h2 : m.ph = n.ph) (t : List ℕ) : L.phAt ks m t = L.phAt ks n t := by
  unfold phAt; rw [h1, h2]

theorem find?_filter_of {α : Type*} (l : List α) (p g : α → Bool)
    (h : ∀ x ∈ l, p x = true → g x = true) : (l.filter g).find? p = l.find? p := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    have ih' := ih fun x hx => h x (List.mem_cons_of_mem _ hx)
    by_cases hp : p a = true
    · simp [h a List.mem_cons_self hp, hp]
    · by_cases hg : g a = true
      · simp [hg, hp, ih']
      · simp [hg, hp, ih']

/-- Outside the copies, `L'` reads the phase indices of `L`. -/
theorem BoxSplitOK.phAt_eq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (n : DNode)
    (hn : n ∈ L.base.nodes) (ks : ℕ → ℕ) (t : List ℕ) : L'.phAt ks n t = L.phAt ks n t := by
  have hc : ((L.cph.filter fun q => decide (q.1 ∈ B)).map
      fun q => (cp q.1, bid₂, q.2.2.1 + 1, c₁)).find?
      (fun q => decide (q.1 = n.id)) = none := by
    rw [List.find?_eq_none]
    intro q hq
    obtain ⟨q0, hq0, rfl⟩ := List.mem_map.1 hq
    obtain ⟨hq0, hB⟩ := List.mem_filter.1 hq0
    simp only [decide_eq_true_eq] at hB
    obtain ⟨⟨m, hm, em⟩, -⟩ := h.cph q0 hq0
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
    by_cases hB : q.1 ∈ B
    · obtain ⟨-, hb⟩ := h.cph q hq
      obtain ⟨hbid, -⟩ := hb hB
      obtain ⟨-, -, -, d1, d2, -⟩ := h.chB n hn (he ▸ hB)
      simp only [hbid, d1, d2]
    · have := h.cphO q hq hB
      rw [he] at this
      simp only [this]

/-- The copy of a node reads the phase index of the node at its copy index raised by `ks c₁`. -/
theorem BoxSplitOK.phAt_cp (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (n : DNode)
    (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) (ks : ℕ → ℕ) (j : ℕ) (u : List ℕ) :
    L'.phAt ks (cpNode cp n) (j :: u) = L.phAt ks n ((j + ks c₁) :: u) := by
  have hc : L.cph.find? (fun q => decide (q.1 = (cpNode cp n).id)) = none := by
    rw [List.find?_eq_none]
    intro q hq
    obtain ⟨⟨m, hm, em⟩, -⟩ := h.cph q hq
    simp only [decide_eq_true_eq]
    rw [← em]
    exact fun e => h.cp_fresh n hn hB m hm e.symm
  have hcg : (L.cph.filter fun q => decide (q.1 ∈ B)).find?
      ((fun q => decide (q.1 = (cpNode cp n).id)) ∘ fun q => (cp q.1, bid₂, q.2.2.1 + 1, c₁)) =
        L.cph.find? (fun q => decide (q.1 = n.id)) := by
    rw [← find?_filter_of L.cph (fun q => decide (q.1 = n.id)) (fun q => decide (q.1 ∈ B))
      fun x _ e => by simp only [decide_eq_true_eq] at e ⊢; rw [e]; exact hB]
    refine find?_congr' fun q hq => ?_
    obtain ⟨hq, hqB⟩ := List.mem_filter.1 hq
    simp only [decide_eq_true_eq] at hqB
    obtain ⟨⟨m, hm, em⟩, -⟩ := h.cph q hq
    simp only [Function.comp_apply, cpNode]
    rw [← em]
    by_cases e : m.id = n.id
    · simp [e]
    · have : cp m.id ≠ cp n.id := fun e' => e (by rw [h.cp_inj m hm (em ▸ hqB) n hn hB e'])
      simp [e, this]
  unfold phAt
  rw [h.cph', List.find?_append, hc, Option.none_or, List.find?_map, hcg]
  cases hf : L.cph.find? (fun q => decide (q.1 = n.id)) with
  | none => rfl
  | some q =>
    have hq := List.mem_of_find?_eq_some hf
    have he := List.find?_some hf
    simp only [decide_eq_true_eq] at he
    obtain ⟨-, hb⟩ := h.cph q hq
    obtain ⟨hbid, hc0⟩ := hb (he ▸ hB)
    obtain ⟨-, -, -, d1, -, d3⟩ := h.chB n hn hB
    simp only [cpNode]
    simp only [Option.map_some, hbid, hc0, d1, d3, List.getD_cons_zero]
    ring_nf

/-- A node's copy `t` renamed leg by leg is the copy `t'` of `n'`. -/
theorem dnodeL_rename (L L' : LDiagram) (ks : ℕ → ℕ) (φ : Port → List ℕ → Dig) (n n' : DNode)
    (t t' : List ℕ) (hk : n'.kind = n.kind) (hd : n'.dim = n.dim)
    (ho : n'.outs.length = n.outs.length) (hi : n'.ins.length = n.ins.length)
    (hph : L'.phAt ks n' t' = L.phAt ks n t)
    (hl : ∀ o i, i < legLen n o → L'.base.tail ⟨n'.id, o, i⟩ = L.base.tail ⟨n.id, o, i⟩ ∧
      ∀ s, φ ⟨n.id, o, i⟩ (t ++ s) = (⟨n'.id, o, i⟩, t' ++ s)) :
    (L.dnodeL ks n t).rename (relab φ) = L'.dnodeL ks n' t' := by
  have e1 := dlegs_rename L.base L'.base ks φ n.id n'.id t t' true n.outs.length
    fun i hi => hl true i (by simpa [legLen] using hi)
  have e2 := dlegs_rename L.base L'.base ks φ n.id n'.id t t' false n.ins.length
    fun i hi' => hl false i (by simpa [legLen] using hi')
  have a1 : (L.dnodeL ks n t).outs = dlegs L.base ks n.id t true n.outs.length := rfl
  have a2 : (L.dnodeL ks n t).ins = dlegs L.base ks n.id t false n.ins.length := rfl
  have b1 : (L'.dnodeL ks n' t').outs = dlegs L'.base ks n'.id t' true n'.outs.length := rfl
  have b2 : (L'.dnodeL ks n' t').ins = dlegs L'.base ks n'.id t' false n'.ins.length := rfl
  have a3 : (L.dnodeL ks n t).kind = n.kind := rfl
  have b3 : (L'.dnodeL ks n' t').kind = n'.kind := rfl
  have a4 : (L.dnodeL ks n t).dim = n.dim := rfl
  have b4 : (L'.dnodeL ks n' t').dim = n'.dim := rfl
  have a5 : (L.dnodeL ks n t).ph = L.phAt ks n t := rfl
  have b5 : (L'.dnodeL ks n' t').ph = L'.phAt ks n' t' := rfl
  cases hA : L.dnodeL ks n t
  cases hB : L'.dnodeL ks n' t'
  rw [hA] at a1 a2 a3 a4 a5
  rw [hB] at b1 b2 b3 b4 b5
  simp only [Node.rename, Node.mk.injEq] at *
  refine ⟨by rw [a3, b3, hk], ?_, ?_, by rw [a4, b4, hd], by rw [a5, b5, hph]⟩
  · rw [a1, b1, e1, ho]
  · rw [a2, b2, e2, hi]

/-- A leg's tail is its leg chain past its node chain. -/
theorem tail_eq (D : Diagram) (p : Port) :
    D.tail p = (D.legChain p).drop (D.nodeChain p.node).length := by
  simp [Diagram.legChain]

/-- The digits of one side `o` of a top-level spider off `B`, read through the split's
relabelling, are its digits in `L'` up to order. -/
theorem BoxSplitOK.tside (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (n : DNode) (hn : n ∈ L.base.nodes) (hnB : n.id ∉ B)
    (hc : L.base.nodeChain n.id = []) (o : Bool) :
    ((dlegs L.base ks n.id [] o (legLen n o)).map (relab (splitφ B cp T (ks c₁)))).Perm
      (dlegs L'.base ks n.id [] o (legLen (extT T n) o)) := by
  set φ := splitφ B cp T (ks c₁)
  have hn' : extT T n ∈ L'.base.nodes := by
    rw [h.nodes]; exact List.mem_append_left _ (List.mem_map.2 ⟨n, hn, rfl⟩)
  have hc' : L'.base.nodeChain n.id = [] := by rw [h.chO n hn hnB, hc]
  have tl : ∀ q : Port, q.node = n.id → L.base.tail q = L.base.legChain q := fun q e => by
    simp [Diagram.legChain, e, hc]
  have tl' : ∀ q : Port, q.node = n.id → L'.base.tail q = L'.base.legChain q := fun q e => by
    simp [Diagram.legChain, e, hc']
  have mem1 : ∀ x, x ∈ dlegs L.base ks n.id [] o (legLen n o) ↔ ∃ p ∈ L.base.legs,
      p.node = n.id ∧ p.out = o ∧ ∃ s ∈ cidx ks (L.base.legChain p), x = dlab p s := by
    intro x
    rw [mem_dlegs]
    constructor
    · rintro ⟨i, hi, s, hs, rfl⟩
      exact ⟨_, (mem_legs_iff L.base h.idsL n hn o i).2 hi, rfl, rfl, s,
        by rw [← tl _ rfl]; exact hs, rfl⟩
    · rintro ⟨⟨x0, o0, i⟩, hp, e1, e2, s, hs, rfl⟩
      simp only at e1 e2
      subst e1 e2
      exact ⟨i, (mem_legs_iff L.base h.idsL n hn _ i).1 hp, s, by rw [tl _ rfl]; exact hs, rfl⟩
  have mem2 : ∀ x, x ∈ dlegs L'.base ks n.id [] o (legLen (extT T n) o) ↔ ∃ q ∈ L'.base.legs,
      q.node = n.id ∧ q.out = o ∧ ∃ u ∈ cidx ks (L'.base.legChain q), x = dlab q u := by
    intro x
    rw [mem_dlegs]
    constructor
    · rintro ⟨i, hi, s, hs, rfl⟩
      exact ⟨_, (mem_legs_iff L'.base h.ids (extT T n) hn' o i).2 hi, rfl, rfl, s,
        by rw [← tl' _ rfl]; exact hs, rfl⟩
    · rintro ⟨⟨x0, o0, i⟩, hp, e1, e2, s, hs, rfl⟩
      simp only at e1 e2
      subst e1 e2
      exact ⟨i, (mem_legs_iff L'.base h.ids (extT T n) hn' _ i).1 hp, s,
        by rw [tl' _ rfl]; exact hs, rfl⟩
  refine (List.perm_ext_iff_of_nodup ?_ (dlegs_nodup _ _ _ _ _ _)).2 fun x => ?_
  · refine (dlegs_nodup _ _ _ _ _ _).map_on fun a ha b hb e => ?_
    obtain ⟨p, hp, -, -, s, hs, rfl⟩ := (mem1 a).1 ha
    obtain ⟨p', hp', -, -, s', hs', rfl⟩ := (mem1 b).1 hb
    rw [relab_dlab, relab_dlab] at e
    obtain ⟨e1, e2⟩ := dlab_inj e
    obtain ⟨rfl, rfl⟩ := h.inj (ks c₁) p hp s p' hp' s' (Prod.ext e1 e2)
    rfl
  · rw [List.mem_map, mem2]
    constructor
    · rintro ⟨y, hy, rfl⟩
      obtain ⟨p, hp, e1, e2, s, hs, rfl⟩ := (mem1 y).1 hy
      obtain ⟨i1, i2⟩ := h.into ks hk p hp s hs
      obtain ⟨f1, f2⟩ := h.φ_node (ks c₁) p (e1 ▸ hnB) s
      exact ⟨_, i1, f1.trans e1, f2.trans e2, _, i2, relab_dlab φ p s⟩
    · rintro ⟨q, hq, e1, e2, u, hu, rfl⟩
      obtain ⟨p, hp, s, hs, he⟩ := h.onto ks hk q hq u hu
      have hpB : p.node ∉ B := by
        intro hpB
        rcases splitφ_cases (B := B) (cp := cp) (T := T) (ks c₁) p s with e | ⟨j, v, -, -, e⟩
        · rw [e] at he
          obtain ⟨rfl, -⟩ := Prod.mk.inj he
          exact hnB (e1 ▸ hpB)
        · rw [e, cq_B p hpB] at he
          obtain ⟨rfl, -⟩ := Prod.mk.inj he
          obtain ⟨m, hm, em⟩ := node_of_legs _ p hp
          exact h.cp_fresh m hm (em ▸ hpB) n hn (by rw [← em]; exact e1)
      obtain ⟨f1, f2⟩ := h.φ_node (ks c₁) p hpB s
      rw [he] at f1 f2
      refine ⟨dlab p s, (mem1 _).2 ⟨p, hp, f1.symm.trans e1, f2.symm.trans e2, s, hs, rfl⟩, ?_⟩
      rw [relab_dlab]
      show dlab (splitφ B cp T (ks c₁) p s).1 (splitφ B cp T (ks c₁) p s).2 = dlab q u
      rw [he]

/-- A top-level Z or X spider of `T`, its bundled legs split, keeps its value under the split's
relabelling. -/
theorem BoxSplitOK.tnode (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (τ : ℕ → ℕ) (n : DNode) (hn : n ∈ L.base.nodes)
    (hT : ∃ e ∈ T, e.1.node = n.id) :
    val E τ ((L.dnodeL ks n []).rename (relab (splitφ B cp T (ks c₁)))) =
      val E τ (L'.dnodeL ks (extT T n) []) := by
  obtain ⟨e, he, en⟩ := hT
  obtain ⟨-, hnB, hc, -, -, m, hm, em, hkind⟩ := h.tl e he
  have hmn : m = n := List.inj_on_of_nodup_map h.idsL hm hn (by rw [em, en])
  subst hmn
  rw [en] at hnB hc
  have po := h.tside ks hk m hn hnB hc true
  have pi := h.tside ks hk m hn hnB hc false
  have hph : L'.phAt ks (extT T m) [] = L.phAt ks m [] := by
    rw [phAt_congr L' ks m (extT T m) rfl rfl, h.phAt_eq m hn]
  refine val_nodeEquiv E τ _ _ ?_
  have a1 : ((L.dnodeL ks m []).rename (relab (splitφ B cp T (ks c₁)))).outs =
      (dlegs L.base ks m.id [] true (legLen m true)).map (relab (splitφ B cp T (ks c₁))) := rfl
  have a2 : ((L.dnodeL ks m []).rename (relab (splitφ B cp T (ks c₁)))).ins =
      (dlegs L.base ks m.id [] false (legLen m false)).map (relab (splitφ B cp T (ks c₁))) := rfl
  have b1 : (L'.dnodeL ks (extT T m) []).outs =
      dlegs L'.base ks m.id [] true (legLen (extT T m) true) := rfl
  have b2 : (L'.dnodeL ks (extT T m) []).ins =
      dlegs L'.base ks m.id [] false (legLen (extT T m) false) := rfl
  have a3 : ((L.dnodeL ks m []).rename (relab (splitφ B cp T (ks c₁)))).kind = m.kind := rfl
  have b3 : (L'.dnodeL ks (extT T m) []).kind = m.kind := rfl
  have a4 : ((L.dnodeL ks m []).rename (relab (splitφ B cp T (ks c₁)))).dim = m.dim := rfl
  have b4 : (L'.dnodeL ks (extT T m) []).dim = m.dim := rfl
  have a5 : ((L.dnodeL ks m []).rename (relab (splitφ B cp T (ks c₁)))).ph = L.phAt ks m [] :=
    rfl
  have b5 : (L'.dnodeL ks (extT T m) []).ph = L.phAt ks m [] := hph
  unfold nodeEquiv
  rw [a3, b3, a4, b4, a5, b5]
  rcases hkind with hz | hx
  · rw [hz]
    simp only [beq_self_eq_true, Bool.true_and, List.isPerm_iff]
    rw [a1, a2, b1, b2]
    exact po.append pi
  · rw [hx]
    simp only [beq_self_eq_true, Bool.true_and, List.isPerm_iff, Bool.and_eq_true]
    rw [a1, a2, b1, b2]
    exact ⟨po, pi⟩

/-- The node products agree under the split's relabelling. -/
theorem BoxSplitOK.prod (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (splitφ B cp T (ks c₁)) x)) :
    L'.prodL E ks τ = L.prodL E ks τ' := by
  set φ := splitφ B cp T (ks c₁)
  have legOf : ∀ n ∈ L.base.nodes, ∀ o i, i < legLen n o →
      (⟨n.id, o, i⟩ : Port) ∈ L.base.legs :=
    fun n hn o i hi => (mem_legs_iff L.base h.idsL n hn o i).2 hi
  have hv : ∀ n ∈ L.base.nodes, ∀ t ∈ cidx ks (L.base.nodeChain n.id),
      val E τ' (L.dnodeL ks n t) = val E τ ((L.dnodeL ks n t).rename (relab φ)) := by
    intro n hn t ht
    rw [val_rename]
    refine val_reads E _ _ _ fun x hx => ?_
    have a1 : (L.dnodeL ks n t).outs = dlegs L.base ks n.id t true n.outs.length := rfl
    have a2 : (L.dnodeL ks n t).ins = dlegs L.base ks n.id t false n.ins.length := rfl
    simp only [Node.labels, List.mem_append, a1, a2] at hx
    have hmem : x ∈ L.base.dlabs ks := by
      rcases hx with hx | hx
      · obtain ⟨i, hi, s, hs, rfl⟩ := (mem_dlegs _ _ _ _ _ _ _).1 hx
        exact (mem_dlabs _ _ _).2 ⟨_, legOf n hn true i (by simpa [legLen] using hi), _,
          mem_cidx_append ks _ _ t s ht hs, rfl⟩
      · obtain ⟨i, hi, s, hs, rfl⟩ := (mem_dlegs _ _ _ _ _ _ _).1 hx
        exact (mem_dlabs _ _ _).2 ⟨_, legOf n hn false i (by simpa [legLen] using hi), _,
          mem_cidx_append ks _ _ t s ht hs, rfl⟩
    exact hτ x hmem
  have tN : ∀ n : DNode, (∀ e ∈ T, e.1.node ≠ n.id) → extT T n = n := by
    intro n hT
    have f1 : (T.filter fun e => decide (e.1.node = n.id) && e.1.out) = [] :=
      List.filter_eq_nil_iff.2 fun e he => by simp [hT e he]
    have f2 : (T.filter fun e => decide (e.1.node = n.id) && !e.1.out) = [] :=
      List.filter_eq_nil_iff.2 fun e he => by simp [hT e he]
    unfold extT
    rw [f1, f2]
    simp
  set F' : DNode → ℂ := fun m =>
    ((cidx ks (L'.base.nodeChain m.id)).map fun t => val E τ (L'.dnodeL ks m t)).prod
  have key : ∀ n ∈ L.base.nodes,
      ((cidx ks (L.base.nodeChain n.id)).map fun t => val E τ' (L.dnodeL ks n t)).prod =
        F' (extT T n) * (if n.id ∈ B then F' (cpNode cp n) else 1) := by
    intro n hn
    by_cases hB : n.id ∈ B
    · obtain ⟨c1, c1', c1'', -⟩ := h.chB n hn hB
      have hx : extT T n = n := tN n fun e he en => (h.tl e he).2.1 (en ▸ hB)
      simp only [hB, ↓reduceIte, hx]
      set r := (L.base.nodeChain n.id).tail
      have hcn : (cpNode cp n).id = cp n.id := rfl
      simp only [F', hcn, c1'', c1']
      rw [c1, cidx_add ks mb c₁ c₂ r hk, List.map_append, List.prod_append, List.map_map]
      have hsl : ∀ o i, SplitL B T ⟨n.id, o, i⟩ := fun o i => Or.inl hB
      have tails : ∀ o i, i < legLen n o → L'.base.tail ⟨n.id, o, i⟩ =
          L.base.tail ⟨n.id, o, i⟩ ∧ L'.base.tail ⟨cp n.id, o, i⟩ = L.base.tail ⟨n.id, o, i⟩ := by
        intro o i hi
        obtain ⟨rl, l1, l1', l1''⟩ := h.lS _ (legOf n hn o i hi) (hsl o i)
        rw [cq_B _ hB] at l1''
        have hcl : (cpP cp ⟨n.id, o, i⟩ : Port) = ⟨cp n.id, o, i⟩ := rfl
        rw [hcl] at l1''
        refine ⟨?_, ?_⟩
        · rw [tail_eq, tail_eq, l1, l1']
          simp only
          rw [c1, c1']
          rfl
        · rw [tail_eq, tail_eq, l1, l1'']
          simp only
          rw [c1, c1'']
          rfl
      congr 1
      · refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c₁ _ t).1 ht
        rw [hv n hn _ (by rw [c1, cidx_add ks mb c₁ c₂ r hk]; exact List.mem_append_left _ ht)]
        refine congrArg (val E τ) (dnodeL_rename L L' ks φ n n (j :: u) (j :: u) rfl rfl rfl rfl
          (h.phAt_eq n hn ks _) fun o i hi => ⟨(tails o i hi).1, fun s => ?_⟩)
        exact splitφ_lo (ks c₁) _ j (u ++ s) hj
      · refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c₂ _ t).1 ht
        simp only [Function.comp_apply, addH]
        have hm : (j + ks c₁) :: u ∈ cidx ks (L.base.nodeChain n.id) := by
          rw [c1, cidx_add ks mb c₁ c₂ r hk]
          exact List.mem_append_right _ (List.mem_map.2 ⟨j :: u, ht, rfl⟩)
        rw [hv n hn _ hm]
        refine congrArg (val E τ) (dnodeL_rename L L' ks φ n (cpNode cp n) ((j + ks c₁) :: u)
          (j :: u) rfl rfl rfl rfl (h.phAt_cp n hn hB ks j u)
          fun o i hi => ⟨(tails o i hi).2, fun s => ?_⟩)
        have := splitφ_hi (B := B) (cp := cp) (T := T) (ks c₁) ⟨n.id, o, i⟩ (hsl o i) j (u ++ s)
        rw [cq_B _ hB] at this
        exact this
    · simp only [hB, ↓reduceIte, mul_one]
      by_cases hT : ∃ e ∈ T, e.1.node = n.id
      · obtain ⟨e, he, en⟩ := hT
        obtain ⟨-, -, hc, -⟩ := h.tl e he
        rw [en] at hc
        have hc' : L'.base.nodeChain (extT T n).id = [] := by
          show L'.base.nodeChain n.id = []
          rw [h.chO n hn hB, hc]
        simp only [F', hc', hc, cidx, List.map_cons, List.map_nil, List.prod_cons,
          List.prod_nil, mul_one]
        rw [hv n hn [] (by rw [hc]; simp [cidx]), h.tnode E ks hk τ n hn ⟨e, he, en⟩]
      · have hx : extT T n = n := tN n fun e he en => hT ⟨e, he, en⟩
        simp only [F', hx, h.chO n hn hB]
        refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        rw [hv n hn t ht]
        refine congrArg (val E τ) (dnodeL_rename L L' ks φ n n t t rfl rfl rfl rfl
          (h.phAt_eq n hn ks t) fun o i hi => ?_)
        have hs : ¬ SplitL B T ⟨n.id, o, i⟩ := by
          rintro (hb | ht')
          · exact hB hb
          · obtain ⟨e, he, e1⟩ := List.mem_map.1 ht'
            exact hT ⟨e, he, by rw [e1]⟩
        refine ⟨?_, fun s => splitφ_O (ks c₁) _ hs _⟩
        rw [tail_eq, tail_eq, h.legO _ (legOf n hn o i hi) hs]
        simp only
        rw [h.chO n hn hB]
  unfold prodL
  rw [List.map_congr_left key, List.prod_map_mul, prod_ite_filter, h.nodes, List.map_append,
    List.prod_append, List.map_map, List.map_map]
  rfl

/-- Splitting an outermost box in two keeps the linked digit sum, `L`'s slots read from `L'`'s
through `splitφ`. -/
theorem evalL_split (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp T) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (σ' : ℕ → List ℕ → ℕ) :
    L'.evalL E ks σ' = L.evalL E ks (bndMap L.base L'.base (splitφ B cp T (ks c₁)) σ') := by
  refine evalL_relabelφ L L' E ks ks σ' (splitφ B cp T (ks c₁))
    (Diagram.dlabs_nodup _ _ h.legsL) (Diagram.dlabs_nodup _ _ h.legs) h.bnd
    (fun p hp t ht => h.into ks hk p hp t ht) (fun q hq u hu => h.onto ks hk q hq u hu)
    (fun p hp t _ p' hp' t' _ he => h.inj (ks c₁) p hp t p' hp' t' he) ?_ ?_
    (fun x hx => h.fwd ks hk x hx) (fun y hy => h.bwd ks hk y hy) (fun p hp => (h.bndL p hp).1)
    ?_ ?_ (fun τ τ' hτ => h.prod E ks hk τ τ' hτ)
  · intro p hp t _
    rcases splitφ_cases (B := B) (cp := cp) (T := T) (ks c₁) p t with e | ⟨j, u, -, hs, e⟩
    · rw [e, (h.legL p hp).2]
    · rw [e, (h.legS p hp hs).2.2.2.2]
  · intro x hx
    rcases (mem_dpairs L ks x).1 hx with ⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩
    · obtain ⟨hw1, hw2, -, hch, -⟩ := h.wires w hw
      exact ⟨hw1, ht, hw2, by rw [hch]; exact ht⟩
    · obtain ⟨hl1, hl2, he1, he2, hc2, hdl, hgd, hce1, hce2⟩ := h.lwf l hl
      unfold linkPairs at hx
      rcases (mem_lp ks _ _ _ l x).1 hx with ⟨t, ht, h1, rfl⟩ | ⟨t, ht, h1, rfl⟩ |
        ⟨t, ht, h1, rfl⟩ | ⟨-, u, hu, rfl⟩
      · refine ⟨hl1, ht, hl2, ?_⟩
        rw [hc2]
        exact cidx_set ks _ t _ _ ht (by rw [hgd]; exact h1) hdl
      · exact ⟨hl1, ht, he1, by rw [hce1]; exact cidx_eraseIdx ks _ t _ ht⟩
      · exact ⟨he2, by rw [hce2]; exact cidx_eraseIdx ks _ t _ ht, hl2, by rw [hc2]; exact ht⟩
      · exact ⟨he2, by rw [hce2]; exact hu, he1, by rw [hce1]; exact hu⟩
  · intro p hp t _
    obtain ⟨-, h2, h3⟩ := h.bndL p hp
    rcases splitφ_cases (B := B) (cp := cp) (T := T) (ks c₁) p t with e | ⟨j, u, -, hs, e⟩
    · rw [e]; exact h2
    · rw [e]; exact h3 hs
  · intro q hq u hu
    rcases h.bnd' q hq with hq' | ⟨p, hp, hs, rfl⟩
    · obtain ⟨t, ht, e⟩ := h.onto_same ks hk q (h.bndL q hq').1 u hu
      exact ⟨q, hq', t, ht, e⟩
    · obtain ⟨t, ht, e⟩ := h.onto_cq ks hk p (h.bndL p hp).1 hs u hu
      exact ⟨p, hp, t, ht, e⟩

end split

end LDiagram

end ArchytasZX
