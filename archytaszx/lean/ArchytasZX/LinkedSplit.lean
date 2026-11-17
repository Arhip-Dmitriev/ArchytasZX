/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedPeel
import ArchytasZX.Iso

/-!
# Splitting an outermost box in two

`evalX_split`: for an outermost node box `bid` over the nodes `B` at `ks c₁ + ks c₂` copies
(multiplicity index `mb`) and a linked diagram `L'` holding `B` at `ks c₁` copies and a copy of `B`
(the nodes `cp x`, outermost box `bid₂`, inner boxes `cb y`) at `ks c₂` copies (`BoxSplitOK`), `L'`
is `L` with `B`'s copy `j + ks c₁` read at the copy's copy `j`. A leg `e.1` of a Z or X spider
outside `B` bundled over `bid` (an entry `e` of `T`) becomes two legs, `e.1` over `bid` and the
appended leg `e.2` over `bid₂` (`extT`); the spider's value is kept up to its leg order
(`BoxSplitOK.tnode`). The split coordinate of a leg's copy index sits at `spos`: first on `B`,
past the node chain on a spider of `T`. A copy-indexed phase or leg dimension `(_, y, c, m)` of
`B` reading the copy index of `bid` (place `0`) reads it plus `c * ks c₁`; on the copy it reads
the copy index of `cb y = bid₂` plus `(c + 1) * ks c₁`; one reading an inner box `y` reads `cb y`
on the copy. `bid` has no links; links of boxes inside it hold once on `B` and once on the copy;
an aligned wire into `B` holds once on `B` and once, between the legs `cq`, on the copy.
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

theorem cidx_len (ks : ℕ → ℕ) (c t : List ℕ) (h : t ∈ cidx ks c) : t.length = c.length :=
  ((mem_cidx ks c t).1 h).length_eq

theorem mem_cidx_append_iff (ks : ℕ → ℕ) (a b t : List ℕ) :
    t ∈ cidx ks (a ++ b) ↔ ∃ u v, t = u ++ v ∧ u ∈ cidx ks a ∧ v ∈ cidx ks b := by
  induction a generalizing t with
  | nil => simp [cidx]
  | cons m a ih =>
    rw [List.cons_append, mem_cidx_cons]
    constructor
    · rintro ⟨j, w, rfl, hj, hw⟩
      obtain ⟨u, v, rfl, hu, hv⟩ := (ih w).1 hw
      exact ⟨j :: u, v, rfl, cons_mem_cidx ks m j a u hj hu, hv⟩
    · rintro ⟨u, v, rfl, hu, hv⟩
      obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks m a _).1 hu
      exact ⟨j, u ++ v, rfl, hj, (ih _).2 ⟨u, v, rfl, hu, hv⟩⟩

theorem mem_cidx_mid (ks : ℕ → ℕ) (pre r : List ℕ) (m : ℕ) (t : List ℕ) :
    t ∈ cidx ks (pre ++ m :: r) ↔
      ∃ u j w, t = u ++ j :: w ∧ u ∈ cidx ks pre ∧ j < ks m ∧ w ∈ cidx ks r := by
  rw [mem_cidx_append_iff]
  constructor
  · rintro ⟨u, v, rfl, hu, hv⟩
    obtain ⟨j, w, rfl, hj, hw⟩ := (mem_cidx_cons ks m r v).1 hv
    exact ⟨u, j, w, rfl, hu, hj, hw⟩
  · rintro ⟨u, j, w, rfl, hu, hj, hw⟩
    exact ⟨u, j :: w, rfl, hu, cons_mem_cidx ks m j r w hj hw⟩

theorem eq_mid (c : List ℕ) (s : ℕ) (hs : s < c.length) :
    c = c.take s ++ c.getD s 0 :: c.drop (s + 1) := by
  rw [List.getD_eq_getElem _ _ hs, ← List.drop_eq_getElem_cons hs, List.take_append_drop]

theorem set_eq_mid (c : List ℕ) (s x : ℕ) (hs : s < c.length) :
    c.set s x = c.take s ++ x :: c.drop (s + 1) := by
  rw [List.set_eq_take_append_cons_drop, if_pos hs]

theorem cidx_getD (ks : ℕ → ℕ) (c t : List ℕ) (h : t ∈ cidx ks c) (i : ℕ) (hi : i < c.length) :
    t.getD i 0 < ks (c.getD i 0) := by
  induction c generalizing t i with
  | nil => simp at hi
  | cons m c ih =>
    obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks m c t).1 h
    cases i with
    | zero => simpa using hj
    | succ i => simpa using ih u hu i (by simpa using hi)

theorem pick_length (t perm : List ℕ) : (pick t perm).length = perm.length := by simp [pick]

theorem pick_getD (t perm : List ℕ) (i : ℕ) (hi : i < perm.length) :
    (pick t perm).getD i 0 = t.getD (perm.getD i 0) 0 := by
  simp [pick, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]

theorem pick_set (t perm : List ℕ) (s s' x : ℕ) (hs : s < t.length)
    (hp : ∀ i ∈ List.range perm.length, perm.getD i 0 = s ↔ i = s') :
    pick (t.set s x) perm = (pick t perm).set s' x := by
  apply List.ext_getElem
  · simp [pick]
  · intro i h1 h2
    have hi : i < perm.length := by simpa [pick] using h1
    have e := hp i (List.mem_range.2 hi)
    rw [List.getD_eq_getElem _ _ hi] at e
    simp only [pick, List.getElem_map, List.getElem_set]
    by_cases hq : perm[i] = s
    · have hi' : s' = i := (e.1 hq).symm
      simp [hi', hq, List.getD_eq_getElem?_getD, hs]
    · have hi' : ¬ s' = i := fun h => hq (e.2 h.symm)
      simp only [hi', ↓reduceIte, List.getElem_map]
      simp [List.getD_eq_getElem?_getD, List.getElem?_set, Ne.symm hq]

theorem pick_mem (ks : ℕ → ℕ) (c t perm : List ℕ) (h : t ∈ cidx ks c)
    (hp : ∀ i ∈ perm, i < c.length) : pick t perm ∈ cidx ks (pick c perm) := by
  rw [mem_cidx]
  unfold pick
  rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff]
  exact List.forall₂_same.2 fun i hi => cidx_getD ks c t h i (hp i hi)

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

theorem mem_dpairsX (L : LDiagram) (ks : ℕ → ℕ) (x : Dig × Dig) : x ∈ L.dpairsX ks ↔
    x ∈ L.dpairs ks ∨ ∃ w ∈ L.aligned, ∃ t ∈ cidx ks (L.base.legChain w.a),
      x = ((w.a, t), (w.b, pick t w.perm)) := by
  simp only [dpairsX, alignPairs, List.mem_append, List.mem_flatMap, List.mem_map]
  constructor
  · rintro (hx | ⟨w, hw, t, ht, rfl⟩)
    · exact Or.inl hx
    · exact Or.inr ⟨w, hw, t, ht, rfl⟩
  · rintro (hx | ⟨w, hw, t, ht, rfl⟩)
    · exact Or.inl hx
    · exact Or.inr ⟨w, hw, t, ht, rfl⟩

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

/-- The place of the split coordinate in a copy index of the leg `p`: `0` on `B`, past the node
chain otherwise. -/
def spos (L : LDiagram) (B : List ℕ) (p : Port) : ℕ :=
  if p.node ∈ B then 0 else (L.base.nodeChain p.node).length

/-- The split's map of digits: a moved leg whose split coordinate is `j ≥ K` goes to `cq` with
that coordinate `j - K`. -/
def splitφ (L : LDiagram) (B : List ℕ) (cp : ℕ → ℕ) (T : List (Port × Port)) (K : ℕ) (p : Port)
    (t : List ℕ) : Dig :=
  if SplitL B T p ∧ spos L B p < t.length ∧ K ≤ t.getD (spos L B p) 0 then
    (cq B cp T p, t.set (spos L B p) (t.getD (spos L B p) 0 - K))
  else (p, t)

/-- The node `n` with a leg appended, of the same dimension, for each entry of `T` on it. -/
def extT (T : List (Port × Port)) (n : DNode) : DNode :=
  { n with
    outs := n.outs ++ ((T.filter fun e => decide (e.1.node = n.id) && e.1.out).map
      fun e => n.outs.getD e.1.pos 0)
    ins := n.ins ++ ((T.filter fun e => decide (e.1.node = n.id) && !e.1.out).map
      fun e => n.ins.getD e.1.pos 0) }

/-- `L'` is `L` with the outermost node box `bid` (multiplicity index `mb`, nodes `B`) left at the
index `c₁` and copied into the nodes `cp x` under the outermost box `bid₂` at the index `c₂`, each
box `y` of `B` read as `cb y` on the copy; each entry `e` of `T` a leg `e.1` of a Z or X spider
bundled over `bid`, its part over `bid₂` the appended leg `e.2`. -/
structure BoxSplitOK (L L' : LDiagram) (bid bid₂ mb c₁ c₂ : ℕ) (B : List ℕ) (cp cb : ℕ → ℕ)
    (T : List (Port × Port)) : Prop where
  cph : ∀ q ∈ L.cph, (∃ n ∈ L.base.nodes, n.id = q.1) ∧ L'.depth q.2.1 q.1 = L.depth q.2.1 q.1 ∧
    (q.1 ∈ B → L'.depth (cb q.2.1) (cp q.1) = L.depth q.2.1 q.1 ∧
      (L.depth q.2.1 q.1 = 0 → q.2.2.2 = c₁))
  cph' : L'.cph = L.cph ++ (L.cph.filter fun q => decide (q.1 ∈ B)).map
    fun q => (cp q.1, cb q.2.1, q.2.2.1 + (if L.depth q.2.1 q.1 = 0 then 1 else 0), q.2.2.2)
  cdim : ∀ q ∈ L.cdim, q.1 ∈ L.base.legs ∧ q.1 ∉ T.map Prod.fst ∧
    L'.depth q.2.1 q.1.node = L.depth q.2.1 q.1.node ∧
    (q.1.node ∈ B → L'.depth (cb q.2.1) (cp q.1.node) = L.depth q.2.1 q.1.node ∧
      (L.depth q.2.1 q.1.node = 0 → q.2.2.2 = c₁))
  cdim' : L'.cdim = L.cdim ++ (L.cdim.filter fun q => decide (q.1.node ∈ B)).map
    fun q => (cpP cp q.1, cb q.2.1, q.2.2.1 + (if L.depth q.2.1 q.1.node = 0 then 1 else 0),
      q.2.2.2)
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
  tl : ∀ e ∈ T, e.1 ∈ L.base.legs ∧ e.1.node ∉ B ∧ e.2.node = e.1.node ∧ e.2.out = e.1.out ∧
    ∃ n ∈ L.base.nodes, n.id = e.1.node ∧ (n.kind = .Z ∨ n.kind = .X)
  legS : ∀ p ∈ L.base.legs, SplitL B T p →
    spos L B p < (L.base.legChain p).length ∧ (L.base.legChain p).getD (spos L B p) 0 = mb ∧
      L'.base.legChain p = (L.base.legChain p).set (spos L B p) c₁ ∧ cq B cp T p ∈ L'.base.legs ∧
      L'.base.legChain (cq B cp T p) = (L.base.legChain p).set (spos L B p) c₂ ∧
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
    WireIn L'.base.wires w ∧ (SplitL B T w.1 → spos L B w.1 = 0 ∧ spos L B w.2 = 0 ∧
      WireIn L'.base.wires (cq B cp T w.1, cq B cp T w.2))
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
    (spos L B l.l1 = 0 ∧ spos L B l.l2 = 0 ∧ spos L B l.e1 = 0 ∧ spos L B l.e2 = 0) ∧
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
  al : ∀ w ∈ L.aligned, w.a ∈ L.base.legs ∧ w.b ∈ L.base.legs ∧
    (SplitL B T w.a ↔ SplitL B T w.b) ∧ (∀ i ∈ w.perm, i < (L.base.legChain w.a).length) ∧
    L.base.legChain w.b = pick (L.base.legChain w.a) w.perm ∧ w ∈ L'.aligned ∧
    (SplitL B T w.a → spos L B w.b < w.perm.length ∧
      (∀ i ∈ List.range w.perm.length, w.perm.getD i 0 = spos L B w.a ↔ i = spos L B w.b) ∧
      ⟨cq B cp T w.a, cq B cp T w.b, w.perm⟩ ∈ L'.aligned)
  al' : ∀ w ∈ L'.aligned, w ∈ L.aligned ∨
    ∃ w0 ∈ L.aligned, SplitL B T w0.a ∧ w = ⟨cq B cp T w0.a, cq B cp T w0.b, w0.perm⟩
  bndL : ∀ p ∈ L.base.bnd, p ∈ L.base.legs ∧ p ∈ L'.base.bnd ∧
    (SplitL B T p → cq B cp T p ∈ L'.base.bnd)
  bnd' : ∀ q ∈ L'.base.bnd, q ∈ L.base.bnd ∨ ∃ p ∈ L.base.bnd, SplitL B T p ∧ q = cq B cp T p

section split

variable {L L' : LDiagram} {bid bid₂ mb c₁ c₂ : ℕ} {B : List ℕ} {cp cb : ℕ → ℕ}
  {T : List (Port × Port)}

theorem spos_B (p : Port) (hp : p.node ∈ B) : spos L B p = 0 := by simp [spos, hp]

theorem splitφ_O (K : ℕ) (p : Port) (hp : ¬ SplitL B T p) (t : List ℕ) :
    splitφ L B cp T K p t = (p, t) := by
  simp [splitφ, hp]

theorem splitφ_low (K : ℕ) (p : Port) (t : List ℕ) (h : t.getD (spos L B p) 0 < K) :
    splitφ L B cp T K p t = (p, t) := by
  unfold splitφ
  rw [if_neg fun h' => absurd h'.2.2 (by omega)]

theorem splitφ_high (K : ℕ) (p : Port) (t : List ℕ) (hs : SplitL B T p)
    (hl : spos L B p < t.length) (hK : K ≤ t.getD (spos L B p) 0) :
    splitφ L B cp T K p t = (cq B cp T p, t.set (spos L B p) (t.getD (spos L B p) 0 - K)) := by
  unfold splitφ
  rw [if_pos ⟨hs, hl, hK⟩]

theorem splitφ_mid_lo (K : ℕ) (p : Port) (u w : List ℕ) (j : ℕ) (hu : u.length = spos L B p)
    (hj : j < K) : splitφ L B cp T K p (u ++ j :: w) = (p, u ++ j :: w) :=
  splitφ_low K p _ (by rw [← hu]; simpa using hj)

theorem splitφ_mid_hi (K : ℕ) (p : Port) (hp : SplitL B T p) (u w : List ℕ) (j : ℕ)
    (hu : u.length = spos L B p) :
    splitφ L B cp T K p (u ++ (j + K) :: w) = (cq B cp T p, u ++ j :: w) := by
  rw [splitφ_high K p _ hp (by rw [← hu]; simp) (by rw [← hu]; simp), ← hu]
  simp

theorem splitφ_lo0 (K : ℕ) (p : Port) (hp : spos L B p = 0) (j : ℕ) (u : List ℕ) (hj : j < K) :
    splitφ L B cp T K p (j :: u) = (p, j :: u) :=
  splitφ_mid_lo K p [] u j hp.symm hj

theorem splitφ_hi0 (K : ℕ) (p : Port) (hs : SplitL B T p) (hp : spos L B p = 0) (j : ℕ)
    (u : List ℕ) : splitφ L B cp T K p ((j + K) :: u) = (cq B cp T p, j :: u) :=
  splitφ_mid_hi K p hs [] u j hp.symm

theorem splitφ_cases (K : ℕ) (p : Port) (t : List ℕ) :
    splitφ L B cp T K p t = (p, t) ∨ ∃ u j w, t = u ++ (j + K) :: w ∧ u.length = spos L B p ∧
      SplitL B T p ∧ splitφ L B cp T K p t = (cq B cp T p, u ++ j :: w) := by
  by_cases h : SplitL B T p ∧ spos L B p < t.length ∧ K ≤ t.getD (spos L B p) 0
  · right
    obtain ⟨hs, hl, hK⟩ := h
    obtain ⟨j, hj⟩ : ∃ j, t.getD (spos L B p) 0 = j + K := ⟨_, (Nat.sub_add_cancel hK).symm⟩
    have e := eq_mid t (spos L B p) hl
    rw [hj] at e
    have hlen : (t.take (spos L B p)).length = spos L B p := by simp; omega
    generalize t.take (spos L B p) = u at e hlen
    generalize t.drop (spos L B p + 1) = w at e
    subst e
    exact ⟨u, j, w, rfl, hlen, hs, splitφ_mid_hi K p hs u w j hlen⟩
  · left
    unfold splitφ
    rw [if_neg h]

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

theorem pmap_lo (K : ℕ) (x : Dig × Dig) (z1 : spos L B x.1.1 = 0) (z2 : spos L B x.2.1 = 0)
    (h1 : ∃ j u, x.1.2 = j :: u ∧ j < K) (h2 : ∃ j u, x.2.2 = j :: u ∧ j < K) :
    pmap (splitφ L B cp T K) x = x := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := x
  obtain ⟨j1, u1, e1, hj1⟩ := h1
  obtain ⟨j2, u2, e2, hj2⟩ := h2
  simp only at e1 e2 z1 z2
  subst e1 e2
  simp [pmap, splitφ_lo0 K p1 z1 j1 u1 hj1, splitφ_lo0 K p2 z2 j2 u2 hj2]

theorem pmap_O (K : ℕ) (x : Dig × Dig) (h1 : ¬ SplitL B T x.1.1) (h2 : ¬ SplitL B T x.2.1) :
    pmap (splitφ L B cp T K) x = x := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := x
  simp [pmap, splitφ_O K p1 h1, splitφ_O K p2 h2]

theorem pmap_shift (K : ℕ) (y : Dig × Dig) (h1 : SplitL B T y.1.1) (h2 : SplitL B T y.2.1)
    (z1 : spos L B y.1.1 = 0) (z2 : spos L B y.2.1 = 0)
    (e1 : ∃ j u, y.1.2 = j :: u) (e2 : ∃ j u, y.2.2 = j :: u) :
    pmap (splitφ L B cp T K) (shiftS K y) = portsP (cq B cp T) y := by
  obtain ⟨⟨p1, t1⟩, ⟨p2, t2⟩⟩ := y
  obtain ⟨j1, u1, e1⟩ := e1
  obtain ⟨j2, u2, e2⟩ := e2
  simp only at e1 e2 h1 h2 z1 z2
  subst e1 e2
  simp [pmap, shiftS, portsP, addH, splitφ_hi0 K p1 h1 z1, splitφ_hi0 K p2 h2 z2]

/-- The ports of a link's pairs satisfy a property of ports its four ports satisfy. -/
theorem lp_ports_inP (ks : ℕ → ℕ) (d k : ℕ) (cs : List ℕ) (l : DLink) (P : Port → Prop)
    (y : Dig × Dig) (hy : y ∈ lp ks d k cs l) (h1 : P l.l1) (h2 : P l.l2) (h3 : P l.e1)
    (h4 : P l.e2) : P y.1.1 ∧ P y.2.1 := by
  obtain ⟨a, b⟩ := lp_ports ks d k cs l y hy
  exact ⟨by rcases a with e | e <;> rw [e] <;> assumption,
    by rcases b with e | e <;> rw [e] <;> assumption⟩

theorem BoxSplitOK.idsEq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) :
    L'.base.nodes.map (·.id) = L.base.nodes.map (·.id) ++
      ((L.base.nodes.filter fun n => n.id ∈ B).map (cpNode cp)).map (·.id) := by
  rw [h.nodes, List.map_append, List.map_map]
  rfl

theorem BoxSplitOK.idsL (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) :
    (L.base.nodes.map (·.id)).Nodup := by
  have := h.ids
  rw [h.idsEq] at this
  exact this.of_append_left

theorem BoxSplitOK.cp_fresh (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (n : DNode)
    (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) :
    cp n.id ≠ m.id := by
  have := h.ids
  rw [h.idsEq, List.nodup_append] at this
  intro e
  refine this.2.2 m.id (List.mem_map.2 ⟨m, hm, rfl⟩) (cp n.id) ?_ e.symm
  exact List.mem_map.2 ⟨cpNode cp n,
    List.mem_map.2 ⟨n, List.mem_filter.2 ⟨hn, by simpa using hB⟩, rfl⟩, rfl⟩

theorem BoxSplitOK.cp_inj (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (n : DNode)
    (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) (hmB : m.id ∈ B)
    (e : cp n.id = cp m.id) : n = m := by
  have := h.ids
  rw [h.idsEq, List.nodup_append] at this
  have h2 := this.2.1
  rw [List.map_map] at h2
  exact (List.inj_on_of_nodup_map h2) (List.mem_filter.2 ⟨hn, by simpa using hB⟩)
    (List.mem_filter.2 ⟨hm, by simpa using hmB⟩) (by simpa [cpNode] using e)

/-- A leg of `B` and a leg of `L` with equal copies are one leg. -/
theorem BoxSplitOK.cpP_inj (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (p : Port)
    (hp : p ∈ L.base.legs) (hB : p.node ∈ B) (q : Port) (hq : q ∈ L.base.legs) (hqB : q.node ∈ B)
    (e : cpP cp p = cpP cp q) : p = q := by
  obtain ⟨n, hn, en⟩ := node_of_legs _ p hp
  obtain ⟨m, hm, em⟩ := node_of_legs _ q hq
  simp only [cpP, Port.mk.injEq] at e
  obtain ⟨e1, e2, e3⟩ := e
  have := h.cp_inj n hn (en ▸ hB) m hm (em ▸ hqB) (by rw [← en, ← em]; exact e1)
  obtain ⟨p1, p2, p3⟩ := p
  obtain ⟨q1, q2, q3⟩ := q
  simp only at en em e2 e3
  subst e2 e3
  rw [en, em, this]

/-- The copy of a leg of `B` is a leg of no node of `L`. -/
theorem BoxSplitOK.cpP_fresh (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (p : Port)
    (hp : p ∈ L.base.legs) (hB : p.node ∈ B) (q : Port) (hq : q ∈ L.base.legs) :
    cpP cp p ≠ q := by
  obtain ⟨n, hn, en⟩ := node_of_legs _ p hp
  obtain ⟨m, hm, em⟩ := node_of_legs _ q hq
  intro e
  have e1 : cp p.node = q.node := by rw [← e]; rfl
  rw [en, em] at e1
  exact h.cp_fresh n hn (en ▸ hB) m hm e1

theorem BoxSplitOK.lS (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (p : Port)
    (hp : p ∈ L.base.legs) (hs : SplitL B T p) : ∃ pre r, pre.length = spos L B p ∧
      L.base.legChain p = pre ++ mb :: r ∧ L'.base.legChain p = pre ++ c₁ :: r ∧
      L'.base.legChain (cq B cp T p) = pre ++ c₂ :: r := by
  obtain ⟨hl, hg, h1, -, h2, -⟩ := h.legS p hp hs
  refine ⟨(L.base.legChain p).take (spos L B p), (L.base.legChain p).drop (spos L B p + 1),
    by simp; omega, ?_, ?_, ?_⟩
  · conv_lhs => rw [eq_mid _ _ hl]
    rw [hg]
  · rw [h1, set_eq_mid _ _ _ hl]
  · rw [h2, set_eq_mid _ _ _ hl]

theorem BoxSplitOK.lS0 (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (p : Port)
    (hp : p ∈ L.base.legs) (hs : SplitL B T p) (z : spos L B p = 0) : ∃ r,
      L.base.legChain p = mb :: r ∧ L'.base.legChain p = c₁ :: r ∧
      L'.base.legChain (cq B cp T p) = c₂ :: r := by
  obtain ⟨pre, r, hl, c1, c1', c1''⟩ := h.lS p hp hs
  rw [z] at hl
  obtain rfl := List.eq_nil_of_length_eq_zero hl
  exact ⟨r, c1, c1', c1''⟩

/-- A leg moved by `splitφ` stays on its node and side off `B`. -/
theorem BoxSplitOK.φ_node (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (K : ℕ) (p : Port)
    (hp : p.node ∉ B) (t : List ℕ) :
    (splitφ L B cp T K p t).1.node = p.node ∧ (splitφ L B cp T K p t).1.out = p.out := by
  rcases splitφ_cases (L := L) (B := B) (cp := cp) (T := T) K p t with e | ⟨u, j, w, -, -, hs, e⟩
  · rw [e]; exact ⟨rfl, rfl⟩
  · rw [e]
    have hT : p ∈ T.map Prod.fst := hs.resolve_left hp
    obtain ⟨q, hq, hq1, he⟩ := cq_T (B := B) (cp := cp) p hp hT
    simp only
    rw [he]
    obtain ⟨-, -, h4, h5, -⟩ := h.tl q hq
    rw [h4, h5, hq1]
    exact ⟨rfl, rfl⟩

/-- A leg of `L` at a digit of `L'`'s chain of it, read back at a digit of its own. -/
theorem BoxSplitOK.onto_same (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (q : Port) (hq : q ∈ L.base.legs) (u : List ℕ)
    (hu : u ∈ cidx ks (L'.base.legChain q)) :
    ∃ t ∈ cidx ks (L.base.legChain q), splitφ L B cp T (ks c₁) q t = (q, u) := by
  by_cases hs : SplitL B T q
  · obtain ⟨pre, r, hl, c1, c1', -⟩ := h.lS q hq hs
    rw [c1'] at hu
    obtain ⟨v, j, w, rfl, hv, hj, hw⟩ := (mem_cidx_mid ks pre r c₁ _).1 hu
    have hvl : v.length = spos L B q := by rw [cidx_len ks _ _ hv, hl]
    have hm : v ++ j :: w ∈ cidx ks (L.base.legChain q) := by
      rw [c1]; exact (mem_cidx_mid ks pre r mb _).2 ⟨v, j, w, rfl, hv, by omega, hw⟩
    exact ⟨v ++ j :: w, hm, splitφ_mid_lo (ks c₁) q v w j hvl hj⟩
  · exact ⟨u, by rw [← h.legO q hq hs]; exact hu, splitφ_O (ks c₁) q hs u⟩

/-- A leg `cq p` of `L'` at a digit of its chain, read back at a digit of `p`. -/
theorem BoxSplitOK.onto_cq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (p : Port) (hp : p ∈ L.base.legs) (hs : SplitL B T p)
    (u : List ℕ) (hu : u ∈ cidx ks (L'.base.legChain (cq B cp T p))) :
    ∃ t ∈ cidx ks (L.base.legChain p), splitφ L B cp T (ks c₁) p t = (cq B cp T p, u) := by
  obtain ⟨pre, r, hl, c1, -, c1''⟩ := h.lS p hp hs
  rw [c1''] at hu
  obtain ⟨v, j, w, rfl, hv, hj, hw⟩ := (mem_cidx_mid ks pre r c₂ _).1 hu
  have hvl : v.length = spos L B p := by rw [cidx_len ks _ _ hv, hl]
  have hm : v ++ (j + ks c₁) :: w ∈ cidx ks (L.base.legChain p) := by
    rw [c1]; exact (mem_cidx_mid ks pre r mb _).2 ⟨v, _, w, rfl, hv, by omega, hw⟩
  exact ⟨v ++ (j + ks c₁) :: w, hm, splitφ_mid_hi (ks c₁) p hs v w j hvl⟩

theorem BoxSplitOK.into (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (p : Port) (hp : p ∈ L.base.legs) (t : List ℕ)
    (ht : t ∈ cidx ks (L.base.legChain p)) :
    (splitφ L B cp T (ks c₁) p t).1 ∈ L'.base.legs ∧
      (splitφ L B cp T (ks c₁) p t).2 ∈
        cidx ks (L'.base.legChain (splitφ L B cp T (ks c₁) p t).1) := by
  by_cases hs : SplitL B T p
  · obtain ⟨pre, r, hl, c1, c1', c1''⟩ := h.lS p hp hs
    rw [c1] at ht
    obtain ⟨u, j, w, rfl, hu, hj, hw⟩ := (mem_cidx_mid ks pre r mb _).1 ht
    have hul : u.length = spos L B p := by rw [cidx_len ks _ _ hu, hl]
    by_cases hj1 : j < ks c₁
    · rw [splitφ_mid_lo (ks c₁) p u w j hul hj1]
      exact ⟨(h.legL p hp).1, by
        rw [c1']; exact (mem_cidx_mid ks pre r c₁ _).2 ⟨u, j, w, rfl, hu, hj1, hw⟩⟩
    · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
      rw [splitφ_mid_hi (ks c₁) p hs u w j' hul]
      exact ⟨(h.legS p hp hs).2.2.2.1, by
        rw [c1'']; exact (mem_cidx_mid ks pre r c₂ _).2 ⟨u, j', w, rfl, hu, by omega, hw⟩⟩
  · rw [splitφ_O (ks c₁) p hs t]
    exact ⟨(h.legL p hp).1, by rw [h.legO p hp hs]; exact ht⟩

theorem BoxSplitOK.onto (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (q : Port) (hq : q ∈ L'.base.legs) (u : List ℕ)
    (hu : u ∈ cidx ks (L'.base.legChain q)) :
    ∃ p ∈ L.base.legs, ∃ t ∈ cidx ks (L.base.legChain p),
      splitφ L B cp T (ks c₁) p t = (q, u) := by
  rcases h.legs' q hq with hq' | ⟨p, hp, hs, rfl⟩
  · obtain ⟨t, ht, e⟩ := h.onto_same ks hk q hq' u hu
    exact ⟨q, hq', t, ht, e⟩
  · obtain ⟨t, ht, e⟩ := h.onto_cq ks hk p hp hs u hu
    exact ⟨p, hp, t, ht, e⟩

theorem BoxSplitOK.inj (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (K : ℕ) (p : Port)
    (hp : p ∈ L.base.legs) (t : List ℕ) (p' : Port) (hp' : p' ∈ L.base.legs) (t' : List ℕ)
    (he : splitφ L B cp T K p t = splitφ L B cp T K p' t') : p = p' ∧ t = t' := by
  rcases splitφ_cases (L := L) (B := B) (cp := cp) (T := T) K p t with
    e1 | ⟨u, j, w, rfl, hu, hs, e1⟩ <;>
  rcases splitφ_cases (L := L) (B := B) (cp := cp) (T := T) K p' t' with
    e2 | ⟨u', j', w', rfl, hu', hs', e2⟩ <;>
  rw [e1, e2] at he <;> simp only [Prod.mk.injEq] at he
  · exact he
  · exact absurd (he.1 ▸ hp) (h.cqnew p' hp' hs')
  · exact absurd (he.1 ▸ hp') (h.cqnew p hp hs)
  · obtain ⟨e3, e4⟩ := he
    have hpp := h.cqinj p hp hs p' hp' hs' e3
    subst hpp
    obtain ⟨rfl, e5⟩ := List.append_inj e4 (by rw [hu, hu'])
    obtain ⟨rfl, rfl⟩ := List.cons.inj e5
    exact ⟨rfl, rfl⟩

/-- Every digit pair of `L` goes to a digit pair of `L'`. -/
theorem BoxSplitOK.fwd (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (x : Dig × Dig) (hx : x ∈ L.dpairs ks) :
    pmap (splitφ L B cp T (ks c₁)) x ∈ L'.dpairs ks ∨
      pswap (pmap (splitφ L B cp T (ks c₁)) x) ∈ L'.dpairs ks := by
  rcases (mem_dpairs L ks x).1 hx with ⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩
  · obtain ⟨hw1, hw2, hiff, hch, hin, hcp⟩ := h.wires w hw
    by_cases hs : SplitL B T w.1
    · have hs2 := hiff.1 hs
      obtain ⟨z1, z2, hcq⟩ := hcp hs
      obtain ⟨r, c1, c1', c1''⟩ := h.lS0 w.1 hw1 hs z1
      obtain ⟨r2, c2, c2', c2''⟩ := h.lS0 w.2 hw2 hs2 z2
      have er : r2 = r := by rw [hch, c1] at c2; exact (List.cons.inj c2).2.symm
      subst er
      rw [c1] at ht
      obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
      by_cases hj1 : j < ks c₁
      · rw [pmap_lo (ks c₁) _ z1 z2 ⟨j, u, rfl, hj1⟩ ⟨j, u, rfl, hj1⟩]
        have hju : j :: u ∈ cidx ks (c₁ :: r2) := cons_mem_cidx ks c₁ j _ u hj1 hu
        exact wireIn_mem L' ks _ _ hin _ (by rw [c1']; exact hju) (by rw [c2']; exact hju)
      · obtain ⟨j', rfl⟩ : ∃ j', j = j' + ks c₁ := ⟨j - ks c₁, by omega⟩
        have e : pmap (splitφ L B cp T (ks c₁)) ((w.1, (j' + ks c₁) :: u),
            (w.2, (j' + ks c₁) :: u)) = ((cq B cp T w.1, j' :: u), (cq B cp T w.2, j' :: u)) := by
          simp [pmap, splitφ_hi0 (ks c₁) w.1 hs z1, splitφ_hi0 (ks c₁) w.2 hs2 z2]
        rw [e]
        have hju : j' :: u ∈ cidx ks (c₂ :: r2) := cons_mem_cidx ks c₂ j' _ u (by omega) hu
        exact wireIn_mem L' ks _ _ hcq _ (by rw [c1'']; exact hju) (by rw [c2'']; exact hju)
    · have hs2 : ¬ SplitL B T w.2 := fun e => hs (hiff.2 e)
      rw [pmap_O (ks c₁) _ hs hs2]
      exact wireIn_mem L' ks _ _ hin _ (by rw [h.legO w.1 hw1 hs]; exact ht)
        (by rw [h.legO w.2 hw2 hs2, hch]; exact ht)
  · obtain ⟨hl1, hl2, he1, he2, hc2, hdl, hgd, hce1, hce2⟩ := h.lwf l hl
    unfold linkPairs at hx
    by_cases hin : SplitL B T l.l1
    · obtain ⟨hl2B, he1B, he2B, hd1, ⟨z1, z2, z3, z4⟩, ⟨lt, hltm, r1, r2, r3, r4, hd', hm'⟩, l',
        hl'm, q1, q2, q3, q4, hd'', hm''⟩ := h.lin l hl hin
      obtain ⟨r, c1, c1', c1''⟩ := h.lS0 l.l1 hl1 hin z1
      obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
        ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
      rw [hd0, c1] at hx
      left
      rcases (mem_lp_split2 ks d0 _ mb c₁ c₂ r l hk x).1 hx with hx | ⟨y, hy, rfl⟩
      · obtain ⟨⟨j1, u1, e1, hj1⟩, ⟨j2, u2, e2, hj2⟩⟩ := lp_heads ks d0 _ c₁ r l x hx
        obtain ⟨y1, y2⟩ := lp_ports_inP ks _ _ _ l (fun p => spos L B p = 0) x hx z1 z2 z3 z4
        rw [pmap_lo (ks c₁) x y1 y2 ⟨j1, u1, e1, hj1⟩ ⟨j2, u2, e2, hj2⟩]
        refine link_mem_dpairs L' ks lt hltm x ?_
        unfold linkPairs
        rw [r1, hd', hd0, hm', c1', lp_congr ks _ _ _ l lt r1 r2 r3 r4]; exact hx
      · obtain ⟨p1, p2⟩ := lp_ports_inP ks _ _ _ l (SplitL B T) y hy hin hl2B he1B he2B
        obtain ⟨y1, y2⟩ := lp_ports_inP ks _ _ _ l (fun p => spos L B p = 0) y hy z1 z2 z3 z4
        obtain ⟨⟨j1, u1, e1, -⟩, ⟨j2, u2, e2, -⟩⟩ := lp_heads ks d0 _ c₂ r l y hy
        rw [pmap_shift (ks c₁) y p1 p2 y1 y2 ⟨j1, u1, e1⟩ ⟨j2, u2, e2⟩]
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
theorem BoxSplitOK.wire_back (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (v : Port × Port) (hv : v ∈ L.base.wires) (u : List ℕ)
    (hu : u ∈ cidx ks (L'.base.legChain v.1)) :
    ∃ x ∈ L.dpairs ks, pmap (splitφ L B cp T (ks c₁)) x = ((v.1, u), (v.2, u)) := by
  obtain ⟨hw1, hw2, hiff, -, -, hcp⟩ := h.wires v hv
  by_cases hs : SplitL B T v.1
  · obtain ⟨z1, z2, -⟩ := hcp hs
    obtain ⟨r, c1, c1', -⟩ := h.lS0 v.1 hw1 hs z1
    rw [c1'] at hu
    obtain ⟨j, w, rfl, hj, hw⟩ := (mem_cidx_cons ks c₁ _ u).1 hu
    refine ⟨((v.1, j :: w), (v.2, j :: w)), wire_mem_dpairs L ks v hv _ ?_,
      pmap_lo (ks c₁) _ z1 z2 ⟨j, w, rfl, hj⟩ ⟨j, w, rfl, hj⟩⟩
    rw [c1]; exact cons_mem_cidx ks mb j _ w (by omega) hw
  · refine ⟨((v.1, u), (v.2, u)), wire_mem_dpairs L ks v hv _ ?_,
      pmap_O (ks c₁) _ hs (fun e => hs (hiff.2 e))⟩
    rw [← h.legO v.1 hw1 hs]; exact hu

/-- The copy of a wire of `L` at a digit of its chain, read back. -/
theorem BoxSplitOK.wire_back_cq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (v : Port × Port) (hv : v ∈ L.base.wires)
    (hs : SplitL B T v.1) (u : List ℕ) (hu : u ∈ cidx ks (L'.base.legChain (cq B cp T v.1))) :
    ∃ x ∈ L.dpairs ks, pmap (splitφ L B cp T (ks c₁)) x =
      ((cq B cp T v.1, u), (cq B cp T v.2, u)) := by
  obtain ⟨hw1, -, hiff, -, -, hcp⟩ := h.wires v hv
  obtain ⟨z1, z2, -⟩ := hcp hs
  obtain ⟨r, c1, -, c1''⟩ := h.lS0 v.1 hw1 hs z1
  rw [c1''] at hu
  obtain ⟨j, w, rfl, hj, hw⟩ := (mem_cidx_cons ks c₂ _ u).1 hu
  refine ⟨((v.1, (j + ks c₁) :: w), (v.2, (j + ks c₁) :: w)), wire_mem_dpairs L ks v hv _ ?_, ?_⟩
  · rw [c1]; exact cons_mem_cidx ks mb _ _ w (by omega) hw
  · simp [pmap, splitφ_hi0 (ks c₁) v.1 hs z1, splitφ_hi0 (ks c₁) v.2 (hiff.1 hs) z2]

/-- The two ends of a wire of `L` have one chain in `L'`, and one at their copies. -/
theorem BoxSplitOK.wire_ch (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (v : Port × Port)
    (hv : v ∈ L.base.wires) : L'.base.legChain v.2 = L'.base.legChain v.1 ∧
      (SplitL B T v.1 → L'.base.legChain (cq B cp T v.2) = L'.base.legChain (cq B cp T v.1)) := by
  obtain ⟨hw1, hw2, hiff, hch, -, hcp⟩ := h.wires v hv
  by_cases hs : SplitL B T v.1
  · obtain ⟨z1, z2, -⟩ := hcp hs
    obtain ⟨r, c1, c1', c1''⟩ := h.lS0 v.1 hw1 hs z1
    obtain ⟨r2, c2, c2', c2''⟩ := h.lS0 v.2 hw2 (hiff.1 hs) z2
    have er : r2 = r := by rw [hch, c1] at c2; exact (List.cons.inj c2).2.symm
    subst er
    exact ⟨by rw [c1', c2'], fun _ => by rw [c1'', c2'']⟩
  · exact ⟨by rw [h.legO v.2 hw2 (fun e => hs (hiff.2 e)), h.legO v.1 hw1 hs, hch],
      fun e => absurd e hs⟩

/-- Every digit pair of `L'` comes from a digit pair of `L`. -/
theorem BoxSplitOK.bwd (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (y : Dig × Dig) (hy : y ∈ L'.dpairs ks) : ∃ x ∈ L.dpairs ks,
      pmap (splitφ L B cp T (ks c₁)) x = y ∨ pswap (pmap (splitφ L B cp T (ks c₁)) x) = y := by
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
      · obtain ⟨-, -, -, hd1, ⟨z1, z2, z3, z4⟩, -⟩ := h.lin l hl hin
        obtain ⟨r, c1, c1', -⟩ := h.lS0 l.l1 hl1 hin z1
        obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
          ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
        rw [hd0, c1'] at hy
        refine ⟨y, link_mem_dpairs L ks l hl y ?_, Or.inl ?_⟩
        · unfold linkPairs
          rw [hd0, c1]; exact (mem_lp_split2 ks d0 _ mb c₁ c₂ r l hk y).2 (Or.inl hy)
        · obtain ⟨⟨j1, u1, e1, hj1⟩, ⟨j2, u2, e2, hj2⟩⟩ := lp_heads ks d0 _ c₁ r l y hy
          obtain ⟨y1, y2⟩ := lp_ports_inP ks _ _ _ l (fun p => spos L B p = 0) y hy z1 z2 z3 z4
          exact pmap_lo (ks c₁) y y1 y2 ⟨j1, u1, e1, hj1⟩ ⟨j2, u2, e2, hj2⟩
      · obtain ⟨hl2O, he1O, he2O, -⟩ := h.lout l hl hin
        rw [h.legO l.l1 hl1 hin] at hy
        obtain ⟨p1, p2⟩ :=
          lp_ports_inP ks _ _ _ l (fun p => ¬ SplitL B T p) y hy hin hl2O he1O he2O
        exact ⟨y, link_mem_dpairs L ks l hl y (by unfold linkPairs; exact hy),
          Or.inl (pmap_O (ks c₁) y p1 p2)⟩
    · obtain ⟨hl1, -, -, -, -, -, -, -, -⟩ := h.lwf l hl
      obtain ⟨hl2B, he1B, he2B, hd1, ⟨z1, z2, z3, z4⟩, -⟩ := h.lin l hl hin
      obtain ⟨r, c1, -, c1''⟩ := h.lS0 l.l1 hl1 hin z1
      obtain ⟨d0, hd0⟩ : ∃ d0, L.depth l.box l.l1.node = d0 + 1 :=
        ⟨_, (Nat.succ_pred_eq_of_pos hd1).symm⟩
      rw [q1, hd'', hd0, hm'', c1'', lp_congr ks _ _ _ (l.mapP (cq B cp T)) l' q1 q2 q3 q4,
        lp_mapP] at hy
      obtain ⟨y0, hy0, rfl⟩ := List.mem_map.1 hy
      obtain ⟨p1, p2⟩ := lp_ports_inP ks _ _ _ l (SplitL B T) y0 hy0 hin hl2B he1B he2B
      obtain ⟨y1, y2⟩ := lp_ports_inP ks _ _ _ l (fun p => spos L B p = 0) y0 hy0 z1 z2 z3 z4
      obtain ⟨⟨j1, u1, e1, -⟩, ⟨j2, u2, e2, -⟩⟩ := lp_heads ks d0 _ c₂ r l y0 hy0
      refine ⟨shiftS (ks c₁) y0, link_mem_dpairs L ks l hl _ ?_,
        Or.inl (pmap_shift (ks c₁) y0 p1 p2 y1 y2 ⟨j1, u1, e1⟩ ⟨j2, u2, e2⟩)⟩
      unfold linkPairs
      rw [hd0, c1]
      exact (mem_lp_split2 ks d0 _ mb c₁ c₂ r l hk _).2 (Or.inr ⟨y0, hy0, rfl⟩)

/-- An aligned wire's pair below the first box's copy count is kept. -/
theorem BoxSplitOK.pmapA_lo (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (K : ℕ)
    (w : DAlign) (hw : w ∈ L.aligned) (hs : SplitL B T w.a) (u v : List ℕ) (j : ℕ)
    (hul : u.length = spos L B w.a) (hj : j < K) :
    pmap (splitφ L B cp T K) ((w.a, u ++ j :: v), (w.b, pick (u ++ j :: v) w.perm)) =
      ((w.a, u ++ j :: v), (w.b, pick (u ++ j :: v) w.perm)) := by
  obtain ⟨-, -, -, -, -, -, hsp⟩ := h.al w hw
  obtain ⟨hsb, hperm, -⟩ := hsp hs
  have hpb : (pick (u ++ j :: v) w.perm).getD (spos L B w.b) 0 = j := by
    rw [pick_getD _ _ _ hsb, (hperm _ (List.mem_range.2 hsb)).2 rfl, ← hul]; simp
  simp only [pmap]
  rw [splitφ_mid_lo K w.a u v j hul hj, splitφ_low K w.b _ (by rw [hpb]; exact hj)]

/-- An aligned wire's pair from the first box's copy count on goes to the copied aligned wire. -/
theorem BoxSplitOK.pmapA_hi (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (K : ℕ)
    (w : DAlign) (hw : w ∈ L.aligned) (hs : SplitL B T w.a) (u v : List ℕ) (j : ℕ)
    (hul : u.length = spos L B w.a) :
    pmap (splitφ L B cp T K) ((w.a, u ++ (j + K) :: v), (w.b, pick (u ++ (j + K) :: v) w.perm)) =
      ((cq B cp T w.a, u ++ j :: v), (cq B cp T w.b, pick (u ++ j :: v) w.perm)) := by
  obtain ⟨-, -, hiff, -, -, -, hsp⟩ := h.al w hw
  obtain ⟨hsb, hperm, -⟩ := hsp hs
  have hpb : (pick (u ++ (j + K) :: v) w.perm).getD (spos L B w.b) 0 = j + K := by
    rw [pick_getD _ _ _ hsb, (hperm _ (List.mem_range.2 hsb)).2 rfl, ← hul]; simp
  have hset : (u ++ (j + K) :: v).set (spos L B w.a) j = u ++ j :: v := by rw [← hul]; simp
  have hlt : spos L B w.a < (u ++ (j + K) :: v).length := by rw [← hul]; simp
  simp only [pmap]
  rw [splitφ_mid_hi K w.a hs u v j hul, splitφ_high K w.b _ (hiff.1 hs)
    (by rw [pick_length]; exact hsb) (by rw [hpb]; omega), hpb, Nat.add_sub_cancel,
    ← pick_set _ _ _ _ _ hlt hperm, hset]

/-- Every aligned pair of `L` goes to an aligned pair of `L'`. -/
theorem BoxSplitOK.fwdA (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (w : DAlign) (hw : w ∈ L.aligned) (t : List ℕ)
    (ht : t ∈ cidx ks (L.base.legChain w.a)) :
    pmap (splitφ L B cp T (ks c₁)) ((w.a, t), (w.b, pick t w.perm)) ∈ L'.dpairsX ks := by
  obtain ⟨ha, -, hiff, -, -, hw', hsp⟩ := h.al w hw
  rw [mem_dpairsX]
  right
  by_cases hs : SplitL B T w.a
  · obtain ⟨-, -, hw2⟩ := hsp hs
    obtain ⟨pre, r, hl, c1, c1', c1''⟩ := h.lS w.a ha hs
    rw [c1] at ht
    obtain ⟨u, x0, v, rfl, hu, hx0, hv⟩ := (mem_cidx_mid ks pre r mb _).1 ht
    have hul : u.length = spos L B w.a := by rw [cidx_len ks _ _ hu, hl]
    by_cases hj : x0 < ks c₁
    · rw [h.pmapA_lo (ks c₁) w hw hs u v x0 hul hj]
      have hm : u ++ x0 :: v ∈ cidx ks (L'.base.legChain w.a) := by
        rw [c1']; exact (mem_cidx_mid ks pre r c₁ _).2 ⟨u, x0, v, rfl, hu, hj, hv⟩
      exact ⟨w, hw', _, hm, rfl⟩
    · obtain ⟨j, rfl⟩ : ∃ j, x0 = j + ks c₁ := ⟨x0 - ks c₁, by omega⟩
      rw [h.pmapA_hi (ks c₁) w hw hs u v j hul]
      have hm : u ++ j :: v ∈ cidx ks (L'.base.legChain (cq B cp T w.a)) := by
        rw [c1'']; exact (mem_cidx_mid ks pre r c₂ _).2 ⟨u, j, v, rfl, hu, by omega, hv⟩
      exact ⟨_, hw2, u ++ j :: v, hm, rfl⟩
  · rw [pmap_O (ks c₁) _ hs (fun e => hs (hiff.2 e))]
    exact ⟨w, hw', t, by rw [h.legO w.a ha hs]; exact ht, rfl⟩

/-- Every aligned pair of `L'` comes from an aligned pair of `L`. -/
theorem BoxSplitOK.bwdA (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (w' : DAlign) (hw' : w' ∈ L'.aligned) (u : List ℕ)
    (hu : u ∈ cidx ks (L'.base.legChain w'.a)) : ∃ x ∈ L.dpairsX ks,
      pmap (splitφ L B cp T (ks c₁)) x = ((w'.a, u), (w'.b, pick u w'.perm)) := by
  rcases h.al' w' hw' with hw0 | ⟨w0, hw0, hs, rfl⟩
  · obtain ⟨ha, -, hiff, -, -, -, -⟩ := h.al w' hw0
    by_cases hs : SplitL B T w'.a
    · obtain ⟨pre, r, hl, c1, c1', -⟩ := h.lS w'.a ha hs
      rw [c1'] at hu
      obtain ⟨u1, j, v, rfl, hu1, hj, hv⟩ := (mem_cidx_mid ks pre r c₁ _).1 hu
      have hul : u1.length = spos L B w'.a := by rw [cidx_len ks _ _ hu1, hl]
      have hm : u1 ++ j :: v ∈ cidx ks (L.base.legChain w'.a) := by
        rw [c1]; exact (mem_cidx_mid ks pre r mb _).2 ⟨u1, j, v, rfl, hu1, by omega, hv⟩
      exact ⟨_, (mem_dpairsX L ks _).2 (Or.inr ⟨w', hw0, u1 ++ j :: v, hm, rfl⟩),
        h.pmapA_lo (ks c₁) w' hw0 hs u1 v j hul hj⟩
    · have hm : u ∈ cidx ks (L.base.legChain w'.a) := by rw [← h.legO w'.a ha hs]; exact hu
      exact ⟨_, (mem_dpairsX L ks _).2 (Or.inr ⟨w', hw0, u, hm, rfl⟩),
        pmap_O (ks c₁) _ hs (fun e => hs (hiff.2 e))⟩
  · obtain ⟨ha, -, -, -, -, -, -⟩ := h.al w0 hw0
    obtain ⟨pre, r, hl, c1, -, c1''⟩ := h.lS w0.a ha hs
    simp only at hu ⊢
    rw [c1''] at hu
    obtain ⟨u1, j, v, rfl, hu1, hj, hv⟩ := (mem_cidx_mid ks pre r c₂ _).1 hu
    have hul : u1.length = spos L B w0.a := by rw [cidx_len ks _ _ hu1, hl]
    have hm : u1 ++ (j + ks c₁) :: v ∈ cidx ks (L.base.legChain w0.a) := by
      rw [c1]; exact (mem_cidx_mid ks pre r mb _).2 ⟨u1, _, v, rfl, hu1, by omega, hv⟩
    exact ⟨_, (mem_dpairsX L ks _).2 (Or.inr ⟨w0, hw0, u1 ++ (j + ks c₁) :: v, hm, rfl⟩),
      h.pmapA_hi (ks c₁) w0 hw0 hs u1 v j hul⟩

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
theorem BoxSplitOK.phAt_eq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (n : DNode)
    (hn : n ∈ L.base.nodes) (ks : ℕ → ℕ) (t : List ℕ) : L'.phAt ks n t = L.phAt ks n t := by
  have hc : ((L.cph.filter fun q => decide (q.1 ∈ B)).map
      fun q => (cp q.1, cb q.2.1, q.2.2.1 + (if L.depth q.2.1 q.1 = 0 then 1 else 0),
        q.2.2.2)).find? (fun q => decide (q.1 = n.id)) = none := by
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
    obtain ⟨-, hd, -⟩ := h.cph q hq
    rw [he] at hd
    simp only [hd]

/-- The copy of a node reads the phase index of the node at its copy index raised by `ks c₁`. -/
theorem BoxSplitOK.phAt_cp (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (n : DNode)
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
      ((fun q => decide (q.1 = (cpNode cp n).id)) ∘ fun q => (cp q.1, cb q.2.1,
        q.2.2.1 + (if L.depth q.2.1 q.1 = 0 then 1 else 0), q.2.2.2)) =
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
    obtain ⟨-, -, hb⟩ := h.cph q hq
    obtain ⟨hdc, hm⟩ := hb (he ▸ hB)
    rw [he] at hdc hm
    simp only [cpNode, Option.map_some, he, hdc]
    rcases hd : L.depth q.2.1 n.id with _ | d
    · rw [hm hd]
      simp only [List.getD_cons_zero, ↓reduceIte]
      ring_nf
    · simp

/-- Off the copies, `L'` reads the leg dimensions of `L`. -/
theorem BoxSplitOK.dimAt_eq (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (p : Port) (hp : p ∈ L.base.legs) (t : List ℕ) : L'.dimAt ks p t = L.dimAt ks p t := by
  have hc : ((L.cdim.filter fun q => decide (q.1.node ∈ B)).map
      fun q => (cpP cp q.1, cb q.2.1, q.2.2.1 + (if L.depth q.2.1 q.1.node = 0 then 1 else 0),
        q.2.2.2)).find? (fun q => decide (q.1 = p)) = none := by
    rw [List.find?_eq_none]
    intro q hq
    obtain ⟨q0, hq0, rfl⟩ := List.mem_map.1 hq
    obtain ⟨hq0, hB⟩ := List.mem_filter.1 hq0
    simp only [decide_eq_true_eq] at hB ⊢
    exact h.cpP_fresh q0.1 (h.cdim q0 hq0).1 hB p hp
  unfold dimAt
  rw [h.cdim', List.find?_append, hc, Option.or_none, (h.legL p hp).2]
  cases hf : L.cdim.find? (fun q => decide (q.1 = p)) with
  | none => rfl
  | some q =>
    have hq := List.mem_of_find?_eq_some hf
    have he := List.find?_some hf
    simp only [decide_eq_true_eq] at he
    obtain ⟨-, -, hd, -⟩ := h.cdim q hq
    rw [he] at hd
    simp only [hd]

/-- The copy of a leg of `B` reads the leg dimension of the leg at its copy index raised by
`ks c₁`. -/
theorem BoxSplitOK.dimAt_cp (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (p : Port) (hp : p ∈ L.base.legs) (hB : p.node ∈ B) (j : ℕ) (u : List ℕ) :
    L'.dimAt ks (cpP cp p) (j :: u) = L.dimAt ks p ((j + ks c₁) :: u) := by
  have hc : L.cdim.find? (fun q => decide (q.1 = cpP cp p)) = none := by
    rw [List.find?_eq_none]
    intro q hq
    simp only [decide_eq_true_eq]
    exact fun e => h.cpP_fresh p hp hB q.1 (h.cdim q hq).1 e.symm
  have hcg : (L.cdim.filter fun q => decide (q.1.node ∈ B)).find?
      ((fun q => decide (q.1 = cpP cp p)) ∘ fun q => (cpP cp q.1, cb q.2.1,
        q.2.2.1 + (if L.depth q.2.1 q.1.node = 0 then 1 else 0), q.2.2.2)) =
        L.cdim.find? (fun q => decide (q.1 = p)) := by
    rw [← find?_filter_of L.cdim (fun q => decide (q.1 = p)) (fun q => decide (q.1.node ∈ B))
      fun x _ e => by simp only [decide_eq_true_eq] at e ⊢; rw [e]; exact hB]
    refine find?_congr' fun q hq => ?_
    obtain ⟨hq, hqB⟩ := List.mem_filter.1 hq
    simp only [decide_eq_true_eq] at hqB
    simp only [Function.comp_apply]
    by_cases e : q.1 = p
    · simp [e]
    · have : cpP cp q.1 ≠ cpP cp p := fun e' => e (h.cpP_inj q.1 (h.cdim q hq).1 hqB p hp hB e')
      simp [e, this]
  have hpd : L'.base.portDim (cpP cp p) = L.base.portDim p := by
    have := (h.legS p hp (Or.inl hB)).2.2.2.2.2
    rwa [cq_B p hB] at this
  unfold dimAt
  rw [h.cdim', List.find?_append, hc, Option.none_or, List.find?_map, hcg]
  cases hf : L.cdim.find? (fun q => decide (q.1 = p)) with
  | none => exact hpd
  | some q =>
    have hq := List.mem_of_find?_eq_some hf
    have he := List.find?_some hf
    simp only [decide_eq_true_eq] at he
    obtain ⟨-, -, -, hb⟩ := h.cdim q hq
    obtain ⟨hdc, hm⟩ := hb (he ▸ hB)
    rw [he] at hdc hm
    simp only [Option.map_some, hpd, he]
    have hn : (cpP cp p).node = cp p.node := rfl
    rw [hn, hdc]
    rcases hd : L.depth q.2.1 p.node with _ | d
    · rw [hm hd]
      simp only [List.getD_cons_zero, ↓reduceIte]
      ring_nf
    · simp

/-- A leg of an entry of `T`, and its appended leg, read their plain dimensions. -/
theorem BoxSplitOK.dimAt_T (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (p : Port) (hp : p ∈ L.base.legs) (hpB : p.node ∉ B) (hT : p ∈ T.map Prod.fst)
    (t t' : List ℕ) : L.dimAt ks p t = L.base.portDim p ∧
      L'.dimAt ks (cq B cp T p) t' = L'.base.portDim (cq B cp T p) := by
  obtain ⟨e, he, e1, ecq⟩ := cq_T (B := B) (cp := cp) p hpB hT
  obtain ⟨-, -, e2, -⟩ := h.tl e he
  constructor
  · have : L.cdim.find? (fun q => decide (q.1 = p)) = none := by
      rw [List.find?_eq_none]
      intro q hq
      simp only [decide_eq_true_eq]
      intro eq
      exact (h.cdim q hq).2.1 (eq ▸ hT)
    simp [dimAt, this]
  · have hn : (cq B cp T p) ∉ L.base.legs := h.cqnew p hp (Or.inr hT)
    have h1 : L.cdim.find? (fun q => decide (q.1 = cq B cp T p)) = none := by
      rw [List.find?_eq_none]
      intro q hq
      simp only [decide_eq_true_eq]
      intro eq
      exact hn (eq ▸ (h.cdim q hq).1)
    have h2 : ((L.cdim.filter fun q => decide (q.1.node ∈ B)).map
        fun q => (cpP cp q.1, cb q.2.1, q.2.2.1 + (if L.depth q.2.1 q.1.node = 0 then 1 else 0),
          q.2.2.2)).find? (fun q => decide (q.1 = cq B cp T p)) = none := by
      rw [List.find?_eq_none]
      intro q hq
      obtain ⟨q0, hq0, rfl⟩ := List.mem_map.1 hq
      obtain ⟨hq0, hB⟩ := List.mem_filter.1 hq0
      simp only [decide_eq_true_eq] at hB ⊢
      intro eq
      have hnode : (cpP cp q0.1).node = (cq B cp T p).node := by rw [eq]
      rw [ecq, e2, e1] at hnode
      obtain ⟨n, hn, en⟩ := node_of_legs _ q0.1 (h.cdim q0 hq0).1
      obtain ⟨m, hm, em⟩ := node_of_legs _ p hp
      exact h.cp_fresh n hn (en ▸ hB) m hm (by rw [← en, ← em]; exact hnode)
    unfold dimAt
    rw [h.cdim', List.find?_append, h1, Option.none_or, h2]

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

/-- The digits of one side `o` of a spider off `B` at its copy `t`, read through the split's
relabelling, are its digits in `L'` at `t` up to order. -/
theorem BoxSplitOK.tside (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (n : DNode) (hn : n ∈ L.base.nodes) (hnB : n.id ∉ B)
    (t : List ℕ) (ht : t ∈ cidx ks (L.base.nodeChain n.id)) (o : Bool) :
    ((dlegs L.base ks n.id t o (legLen n o)).map (relab (splitφ L B cp T (ks c₁)))).Perm
      (dlegs L'.base ks n.id t o (legLen (extT T n) o)) := by
  have hn' : extT T n ∈ L'.base.nodes := by
    rw [h.nodes]; exact List.mem_append_left _ (List.mem_map.2 ⟨n, hn, rfl⟩)
  have hc' : L'.base.nodeChain n.id = L.base.nodeChain n.id := h.chO n hn hnB
  have tl : ∀ q : Port, q.node = n.id →
      L.base.legChain q = L.base.nodeChain n.id ++ L.base.tail q := fun q e => by
    simp [Diagram.legChain, e]
  have tl' : ∀ q : Port, q.node = n.id →
      L'.base.legChain q = L.base.nodeChain n.id ++ L'.base.tail q := fun q e => by
    simp [Diagram.legChain, e, hc']
  have htl : t.length = (L.base.nodeChain n.id).length := cidx_len ks _ _ ht
  have pre : ∀ p : Port, p.node = n.id → ∀ t1 s, t1.length = t.length →
      ∃ s', (splitφ L B cp T (ks c₁) p (t1 ++ s)).2 = t1 ++ s' := by
    intro p e t1 s hl1
    rcases splitφ_cases (L := L) (B := B) (cp := cp) (T := T) (ks c₁) p (t1 ++ s) with
      e1 | ⟨u, j, w, e2, hu, -, e1⟩
    · exact ⟨s, by rw [e1]⟩
    · have hsp : spos L B p = t1.length := by
        rw [spos, if_neg (by rw [e]; exact hnB), e, hl1, htl]
      obtain ⟨rfl, rfl⟩ := List.append_inj e2.symm (by rw [hu, hsp])
      exact ⟨j :: w, by rw [e1]⟩
  have mem1 : ∀ x, x ∈ dlegs L.base ks n.id t o (legLen n o) ↔ ∃ p ∈ L.base.legs,
      p.node = n.id ∧ p.out = o ∧ ∃ s ∈ cidx ks (L.base.tail p), x = dlab p (t ++ s) := by
    intro x
    rw [mem_dlegs]
    constructor
    · rintro ⟨i, hi, s, hs, rfl⟩
      exact ⟨_, (mem_legs_iff L.base h.idsL n hn o i).2 hi, rfl, rfl, s, hs, rfl⟩
    · rintro ⟨⟨x0, o0, i⟩, hp, e1, e2, s, hs, rfl⟩
      simp only at e1 e2
      subst e1 e2
      exact ⟨i, (mem_legs_iff L.base h.idsL n hn _ i).1 hp, s, hs, rfl⟩
  have mem2 : ∀ x, x ∈ dlegs L'.base ks n.id t o (legLen (extT T n) o) ↔ ∃ q ∈ L'.base.legs,
      q.node = n.id ∧ q.out = o ∧ ∃ s ∈ cidx ks (L'.base.tail q), x = dlab q (t ++ s) := by
    intro x
    rw [mem_dlegs]
    constructor
    · rintro ⟨i, hi, s, hs, rfl⟩
      exact ⟨_, (mem_legs_iff L'.base h.ids (extT T n) hn' o i).2 hi, rfl, rfl, s, hs, rfl⟩
    · rintro ⟨⟨x0, o0, i⟩, hp, e1, e2, s, hs, rfl⟩
      simp only at e1 e2
      subst e1 e2
      exact ⟨i, (mem_legs_iff L'.base h.ids (extT T n) hn' _ i).1 hp, s, hs, rfl⟩
  refine (List.perm_ext_iff_of_nodup ?_ (dlegs_nodup _ _ _ _ _ _)).2 fun x => ?_
  · refine (dlegs_nodup _ _ _ _ _ _).map_on fun a ha b hb e => ?_
    obtain ⟨p, hp, ep, -, s, hs, rfl⟩ := (mem1 a).1 ha
    obtain ⟨p', hp', ep', -, s', hs', rfl⟩ := (mem1 b).1 hb
    rw [relab_dlab, relab_dlab] at e
    obtain ⟨e1, e2⟩ := dlab_inj e
    obtain ⟨rfl, e3⟩ := h.inj (ks c₁) p hp (t ++ s) p' hp' (t ++ s') (Prod.ext e1 e2)
    rw [e3]
  · rw [List.mem_map, mem2]
    constructor
    · rintro ⟨y, hy, rfl⟩
      obtain ⟨p, hp, e1, e2, s, hs, rfl⟩ := (mem1 y).1 hy
      have hts : t ++ s ∈ cidx ks (L.base.legChain p) := by
        rw [tl p e1]; exact mem_cidx_append ks _ _ t s ht hs
      obtain ⟨i1, i2⟩ := h.into ks hk p hp (t ++ s) hts
      obtain ⟨f1, f2⟩ := h.φ_node (ks c₁) p (by rw [e1]; exact hnB) (t ++ s)
      obtain ⟨s', hs'⟩ := pre p e1 t s rfl
      refine ⟨_, i1, f1.trans e1, f2.trans e2, s', ?_, by rw [relab_dlab, ← hs']⟩
      rw [hs', tl' _ (f1.trans e1)] at i2
      obtain ⟨u1, v1, e3, hu1, hv1⟩ := (mem_cidx_append_iff ks _ _ _).1 i2
      have := List.append_inj e3 (by rw [htl, cidx_len ks _ _ hu1])
      rw [this.2]; exact hv1
    · rintro ⟨q, hq, e1, e2, s', hs', rfl⟩
      have hu : t ++ s' ∈ cidx ks (L'.base.legChain q) := by
        rw [tl' q e1]; exact mem_cidx_append ks _ _ t s' ht hs'
      obtain ⟨p, hp, t0, ht0, he⟩ := h.onto ks hk q hq _ hu
      have hpB : p.node ∉ B := by
        intro hpB
        rcases splitφ_cases (L := L) (B := B) (cp := cp) (T := T) (ks c₁) p t0 with
          e | ⟨u, j, w, -, -, -, e⟩
        · rw [e] at he
          obtain ⟨rfl, -⟩ := Prod.mk.inj he
          exact hnB (e1 ▸ hpB)
        · rw [e, cq_B p hpB] at he
          obtain ⟨rfl, -⟩ := Prod.mk.inj he
          obtain ⟨m, hm, em⟩ := node_of_legs _ p hp
          exact h.cp_fresh m hm (em ▸ hpB) n hn (by rw [← em]; exact e1)
      obtain ⟨f1, f2⟩ := h.φ_node (ks c₁) p hpB t0
      rw [he] at f1 f2
      have e1' : p.node = n.id := f1.symm.trans e1
      rw [tl p e1'] at ht0
      obtain ⟨t1, s0, rfl, ht1, hs0⟩ := (mem_cidx_append_iff ks _ _ _).1 ht0
      have hl1 : t1.length = t.length := by rw [cidx_len ks _ _ ht1, htl]
      obtain ⟨s'', hs''⟩ := pre p e1' t1 s0 hl1
      have ht1e : t1 = t := by
        have h2 := congrArg Prod.snd he
        simp only at h2
        rw [hs''] at h2
        exact (List.append_inj h2 hl1).1
      subst ht1e
      refine ⟨dlab p (t1 ++ s0), (mem1 _).2 ⟨p, hp, e1', f2.symm.trans e2, s0, ?_, rfl⟩, ?_⟩
      · exact hs0
      · rw [relab_dlab]
        show dlab (splitφ L B cp T (ks c₁) p (t1 ++ s0)).1
          (splitφ L B cp T (ks c₁) p (t1 ++ s0)).2 = dlab q (t1 ++ s')
        rw [he]

/-- A Z or X spider of `T`, its bundled legs split, keeps its value at each copy under the
split's relabelling. -/
theorem BoxSplitOK.tnode (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (τ : ℕ → ℕ) (n : DNode) (hn : n ∈ L.base.nodes)
    (hT : ∃ e ∈ T, e.1.node = n.id) (t : List ℕ) (ht : t ∈ cidx ks (L.base.nodeChain n.id)) :
    val E τ ((L.dnodeL ks n t).rename (relab (splitφ L B cp T (ks c₁)))) =
      val E τ (L'.dnodeL ks (extT T n) t) := by
  obtain ⟨e, he, en⟩ := hT
  obtain ⟨-, hnB, -, -, m, hm, em, hkind⟩ := h.tl e he
  have hmn : m = n := List.inj_on_of_nodup_map h.idsL hm hn (by rw [em, en])
  subst hmn
  rw [en] at hnB
  have po := h.tside ks hk m hn hnB t ht true
  have pi := h.tside ks hk m hn hnB t ht false
  have hph : L'.phAt ks (extT T m) t = L.phAt ks m t := by
    rw [phAt_congr L' ks m (extT T m) rfl rfl, h.phAt_eq m hn]
  refine val_nodeEquiv E τ _ _ ?_
  have a1 : ((L.dnodeL ks m t).rename (relab (splitφ L B cp T (ks c₁)))).outs =
      (dlegs L.base ks m.id t true (legLen m true)).map (relab (splitφ L B cp T (ks c₁))) := rfl
  have a2 : ((L.dnodeL ks m t).rename (relab (splitφ L B cp T (ks c₁)))).ins =
      (dlegs L.base ks m.id t false (legLen m false)).map (relab (splitφ L B cp T (ks c₁))) :=
    rfl
  have b1 : (L'.dnodeL ks (extT T m) t).outs =
      dlegs L'.base ks m.id t true (legLen (extT T m) true) := rfl
  have b2 : (L'.dnodeL ks (extT T m) t).ins =
      dlegs L'.base ks m.id t false (legLen (extT T m) false) := rfl
  have a3 : ((L.dnodeL ks m t).rename (relab (splitφ L B cp T (ks c₁)))).kind = m.kind := rfl
  have b3 : (L'.dnodeL ks (extT T m) t).kind = m.kind := rfl
  have a4 : ((L.dnodeL ks m t).rename (relab (splitφ L B cp T (ks c₁)))).dim = m.dim := rfl
  have b4 : (L'.dnodeL ks (extT T m) t).dim = m.dim := rfl
  have a5 : ((L.dnodeL ks m t).rename (relab (splitφ L B cp T (ks c₁)))).ph = L.phAt ks m t :=
    rfl
  have b5 : (L'.dnodeL ks (extT T m) t).ph = L.phAt ks m t := hph
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
theorem BoxSplitOK.prod (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (E : Env)
    (ks : ℕ → ℕ) (hk : ks mb = ks c₁ + ks c₂) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (splitφ L B cp T (ks c₁)) x)) :
    L'.prodL E ks τ = L.prodL E ks τ' := by
  set φ := splitφ L B cp T (ks c₁)
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
      have hz : ∀ o i, spos L B ⟨n.id, o, i⟩ = 0 := fun o i => spos_B _ hB
      have tails : ∀ o i, i < legLen n o → L'.base.tail ⟨n.id, o, i⟩ =
          L.base.tail ⟨n.id, o, i⟩ ∧ L'.base.tail ⟨cp n.id, o, i⟩ = L.base.tail ⟨n.id, o, i⟩ := by
        intro o i hi
        obtain ⟨rl, l1, l1', l1''⟩ := h.lS0 _ (legOf n hn o i hi) (hsl o i) (hz o i)
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
        exact splitφ_lo0 (ks c₁) _ (hz o i) j (u ++ s) hj
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
        have := splitφ_hi0 (L := L) (B := B) (cp := cp) (T := T) (ks c₁) ⟨n.id, o, i⟩ (hsl o i)
          (hz o i) j (u ++ s)
        rw [cq_B _ hB] at this
        exact this
    · simp only [hB, ↓reduceIte, mul_one]
      by_cases hT : ∃ e ∈ T, e.1.node = n.id
      · have hc' : L'.base.nodeChain (extT T n).id = L.base.nodeChain n.id := h.chO n hn hB
        simp only [F', hc']
        refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        rw [hv n hn t ht, h.tnode E ks hk τ n hn hT t ht]
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

/-- Splitting an outermost box in two keeps the digit sum `evalX`, `L`'s slots read from `L'`'s
through `splitφ`. -/
theorem evalX_split (h : BoxSplitOK L L' bid bid₂ mb c₁ c₂ B cp cb T) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c₁ + ks c₂) (σ' : ℕ → List ℕ → ℕ) :
    L'.evalX E ks σ' = L.evalX E ks (bndMap L.base L'.base (splitφ L B cp T (ks c₁)) σ') := by
  refine evalX_relabelφ L L' E ks ks σ' (splitφ L B cp T (ks c₁))
    (Diagram.dlabs_nodup _ _ h.legsL) (Diagram.dlabs_nodup _ _ h.legs) h.bnd
    (fun p hp t ht => h.into ks hk p hp t ht) (fun q hq u hu => h.onto ks hk q hq u hu)
    (fun p hp t _ p' hp' t' _ he => h.inj (ks c₁) p hp t p' hp' t' he) ?_ ?_ ?_ ?_
    (fun p hp => (h.bndL p hp).1) ?_ ?_ (fun τ τ' hτ => h.prod E ks hk τ τ' hτ)
  · intro p hp t _
    rcases splitφ_cases (L := L) (B := B) (cp := cp) (T := T) (ks c₁) p t with
      e | ⟨u, j, w, rfl, hu, hs, e⟩
    · rw [e, h.dimAt_eq ks p hp]
    · rw [e]
      by_cases hB : p.node ∈ B
      · obtain rfl : u = [] := List.eq_nil_of_length_eq_zero (by rw [hu, spos_B p hB])
        rw [cq_B p hB]
        exact congrArg E.dimv (h.dimAt_cp ks p hp hB j w)
      · have hT : p ∈ T.map Prod.fst := hs.resolve_left hB
        obtain ⟨d1, -⟩ := h.dimAt_T ks p hp hB hT (u ++ (j + ks c₁) :: w) []
        obtain ⟨-, d2⟩ := h.dimAt_T ks p hp hB hT [] (u ++ j :: w)
        rw [d1, d2, (h.legS p hp hs).2.2.2.2.2]
  · intro x hx
    rcases (mem_dpairsX L ks x).1 hx with hx | ⟨w, hw, t, ht, rfl⟩
    · rcases (mem_dpairs L ks x).1 hx with ⟨w, hw, t, ht, rfl⟩ | ⟨l, hl, hx⟩
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
    · obtain ⟨ha, hb, -, hlt, hcb, -⟩ := h.al w hw
      exact ⟨ha, ht, hb, by rw [hcb]; exact pick_mem ks _ t w.perm ht hlt⟩
  · intro x hx
    rcases (mem_dpairsX L ks x).1 hx with hx | ⟨w, hw, t, ht, rfl⟩
    · rcases h.fwd ks hk x hx with e | e
      · exact Or.inl ((mem_dpairsX L' ks _).2 (Or.inl e))
      · exact Or.inr ((mem_dpairsX L' ks _).2 (Or.inl e))
    · exact Or.inl (h.fwdA ks hk w hw t ht)
  · intro y hy
    rcases (mem_dpairsX L' ks y).1 hy with hy | ⟨w', hw', u, hu, rfl⟩
    · obtain ⟨x, hx, e⟩ := h.bwd ks hk y hy
      exact ⟨x, (mem_dpairsX L ks x).2 (Or.inl hx), e⟩
    · obtain ⟨x, hx, e⟩ := h.bwdA ks hk w' hw' u hu
      exact ⟨x, hx, Or.inl e⟩
  · intro p hp t _
    obtain ⟨-, h2, h3⟩ := h.bndL p hp
    rcases splitφ_cases (L := L) (B := B) (cp := cp) (T := T) (ks c₁) p t with
      e | ⟨u, j, w, -, -, hs, e⟩
    · rw [e]; exact h2
    · rw [e]; exact h3 hs
  · intro q hq u hu
    rcases h.bnd' q hq with hq' | ⟨p, hp, hs, rfl⟩
    · obtain ⟨t, ht, e⟩ := h.onto_same ks hk q (h.bndL q hq').1 u hu
      exact ⟨q, hq', t, ht, e⟩
    · obtain ⟨t, ht, e⟩ := h.onto_cq ks hk p (h.bndL p hp).1 hs u hu
      exact ⟨p, hp, t, ht, e⟩

end split

/-- A digit with its leg renamed by `rn`. -/
def rnφ (rn : Port → Port) (p : Port) (t : List ℕ) : Dig := (rn p, t)

/-- A link with its ports renamed by `rn`. -/
def lmap (rn : Port → Port) (l : DLink) : DLink := ⟨l.box, rn l.l1, rn l.l2, rn l.e1, rn l.e2⟩

/-- An aligned wire with its legs renamed by `rn`. -/
def amap (rn : Port → Port) (w : DAlign) : DAlign := ⟨rn w.a, rn w.b, w.perm⟩

/-- `R` is `A` with the legs of its Z and X spiders renumbered by `rn` within each side, every
other leg kept. -/
structure LegPermOK (A R : LDiagram) (rn : Port → Port) : Prop where
  nodes : R.base.nodes = A.base.nodes
  ids : (A.base.nodes.map (·.id)).Nodup
  legs : A.base.legs.Nodup
  bnd : R.base.bnd.Nodup
  perm : (A.base.legs.map rn).Perm A.base.legs
  side : ∀ p ∈ A.base.legs, (rn p).node = p.node ∧ (rn p).out = p.out ∧
    (rn p ≠ p → ∃ n ∈ A.base.nodes, n.id = p.node ∧ (n.kind = .Z ∨ n.kind = .X))
  chain : ∀ p ∈ A.base.legs, R.base.legChain (rn p) = A.base.legChain p ∧
    R.base.portDim (rn p) = A.base.portDim p
  nchain : ∀ n ∈ A.base.nodes, R.base.nodeChain n.id = A.base.nodeChain n.id
  wires : R.base.wires = A.base.wires.map fun w => (rn w.1, rn w.2)
  outs : R.base.outs = A.base.outs.map rn
  ins : R.base.ins = A.base.ins.map rn
  links : R.links = A.links.map (lmap rn)
  ldep : ∀ l ∈ A.links, R.depth l.box (rn l.l1).node = A.depth l.box l.l1.node ∧
    R.multOf l.box = A.multOf l.box
  lwf : ∀ l ∈ A.links, l.l1 ∈ A.base.legs ∧ l.l2 ∈ A.base.legs ∧ l.e1 ∈ A.base.legs ∧
    l.e2 ∈ A.base.legs ∧ A.base.legChain l.l2 = A.base.legChain l.l1 ∧
    A.depth l.box l.l1.node < (A.base.legChain l.l1).length ∧
    (A.base.legChain l.l1).getD (A.depth l.box l.l1.node) 0 = A.multOf l.box ∧
    A.base.legChain l.e1 = (A.base.legChain l.l1).eraseIdx (A.depth l.box l.l1.node) ∧
    A.base.legChain l.e2 = (A.base.legChain l.l1).eraseIdx (A.depth l.box l.l1.node)
  wwf : ∀ w ∈ A.base.wires, w.1 ∈ A.base.legs ∧ w.2 ∈ A.base.legs ∧
    A.base.legChain w.2 = A.base.legChain w.1
  cph : R.cph = A.cph
  cdep : ∀ q ∈ A.cph, R.depth q.2.1 q.1 = A.depth q.2.1 q.1
  cdim : R.cdim = A.cdim.map fun q => (rn q.1, q.2)
  ddep : ∀ q ∈ A.cdim, q.1 ∈ A.base.legs ∧ R.depth q.2.1 q.1.node = A.depth q.2.1 q.1.node
  aligned : R.aligned = A.aligned.map (amap rn)
  awf : ∀ w ∈ A.aligned, w.a ∈ A.base.legs ∧ w.b ∈ A.base.legs ∧
    (∀ i ∈ w.perm, i < (A.base.legChain w.a).length) ∧
    A.base.legChain w.b = pick (A.base.legChain w.a) w.perm
  bndL : ∀ p ∈ A.base.bnd, p ∈ A.base.legs

theorem lp_lmap (rn : Port → Port) (ks : ℕ → ℕ) (d k : ℕ) (c : List ℕ) (l : DLink) :
    lp ks d k c (lmap rn l) = (lp ks d k c l).map (pmap (rnφ rn)) := by
  unfold lp
  rw [List.map_append, List.map_flatMap]
  congr 1
  · refine List.flatMap_congr fun t _ => ?_
    rw [List.map_append]
    congr 1
    · split_ifs <;> rfl
    · split_ifs <;> rfl
  · split_ifs
    · rw [List.map_map]; rfl
    · rfl

section perm

variable {A R : LDiagram} {rn : Port → Port}

theorem LegPermOK.legsEq (h : LegPermOK A R rn) : R.base.legs = A.base.legs := by
  simp only [Diagram.legs, h.nodes]

theorem LegPermOK.bndEq (h : LegPermOK A R rn) : R.base.bnd = A.base.bnd.map rn := by
  simp only [Diagram.bnd, h.outs, h.ins, List.map_append]

theorem LegPermOK.mem (h : LegPermOK A R rn) (p : Port) (hp : p ∈ A.base.legs) :
    rn p ∈ A.base.legs :=
  h.perm.subset (List.mem_map.2 ⟨p, hp, rfl⟩)

theorem LegPermOK.onto (h : LegPermOK A R rn) (q : Port) (hq : q ∈ A.base.legs) :
    ∃ p ∈ A.base.legs, rn p = q :=
  List.mem_map.1 (h.perm.symm.subset hq)

theorem LegPermOK.inj (h : LegPermOK A R rn) (p : Port) (hp : p ∈ A.base.legs) (q : Port)
    (hq : q ∈ A.base.legs) (e : rn p = rn q) : p = q :=
  List.inj_on_of_nodup_map (h.perm.nodup_iff.2 h.legs) hp hq e

theorem LegPermOK.tail (h : LegPermOK A R rn) (p : Port) (hp : p ∈ A.base.legs) :
    R.base.tail (rn p) = A.base.tail p := by
  obtain ⟨n, hn, en⟩ := node_of_legs _ p hp
  rw [tail_eq, tail_eq, (h.chain p hp).1, (h.side p hp).1, en, h.nchain n hn]

theorem LegPermOK.dpairsX (h : LegPermOK A R rn) (ks : ℕ → ℕ) :
    R.dpairsX ks = (A.dpairsX ks).map (pmap (rnφ rn)) := by
  unfold LDiagram.dpairsX dpairs
  rw [h.wires, h.links, h.aligned, List.map_append, List.map_append, List.flatMap_map,
    List.flatMap_map, List.flatMap_map, List.map_flatMap, List.map_flatMap, List.map_flatMap]
  congr 1
  congr 1
  · refine List.flatMap_congr fun w hw => ?_
    rw [(h.chain w.1 (h.wwf w hw).1).1, List.map_map]
    rfl
  · refine List.flatMap_congr fun l hl => ?_
    obtain ⟨d1, d2⟩ := h.ldep l hl
    unfold linkPairs
    have e1 : (lmap rn l).l1 = rn l.l1 := rfl
    have e2 : (lmap rn l).box = l.box := rfl
    rw [e1, e2, d1, d2, (h.chain l.l1 (h.lwf l hl).1).1, lp_lmap]
  · refine List.flatMap_congr fun w hw => ?_
    unfold alignPairs
    have e1 : (amap rn w).a = rn w.a := rfl
    rw [e1, (h.chain w.a (h.awf w hw).1).1, List.map_map]
    rfl

theorem LegPermOK.dimAt (h : LegPermOK A R rn) (ks : ℕ → ℕ) (p : Port) (hp : p ∈ A.base.legs)
    (t : List ℕ) : R.dimAt ks (rn p) t = A.dimAt ks p t := by
  have hf : R.cdim.find? (fun q => decide (q.1 = rn p)) =
      (A.cdim.find? fun q => decide (q.1 = p)).map fun q => (rn q.1, q.2) := by
    rw [h.cdim, List.find?_map]
    congr 1
    refine find?_congr' fun q hq => ?_
    simp only [Function.comp_apply, decide_eq_decide]
    exact ⟨fun e => h.inj _ (h.ddep q hq).1 _ hp e, fun e => by rw [e]⟩
  unfold LDiagram.dimAt
  rw [hf]
  cases hq : A.cdim.find? (fun q => decide (q.1 = p)) with
  | none => exact (h.chain p hp).2
  | some q =>
    have hm := List.mem_of_find?_eq_some hq
    have he := List.find?_some hq
    simp only [decide_eq_true_eq] at he
    have hd := (h.ddep q hm).2
    rw [he] at hd
    simp only [Option.map_some, (h.chain p hp).2, (h.side p hp).1, hd]

theorem LegPermOK.phAt (h : LegPermOK A R rn) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    R.phAt ks n t = A.phAt ks n t := by
  unfold LDiagram.phAt
  rw [h.cph]
  cases hf : A.cph.find? (fun q => decide (q.1 = n.id)) with
  | none => rfl
  | some q =>
    have hq := List.mem_of_find?_eq_some hf
    have he := List.find?_some hf
    simp only [decide_eq_true_eq] at he
    have hd := h.cdep q hq
    rw [he] at hd
    simp only [hd]

/-- The digits of one side `o` of a node at its copy `t`, renamed by `rn`, are its digits in `R`
up to order. -/
theorem LegPermOK.sidePerm (h : LegPermOK A R rn) (ks : ℕ → ℕ) (n : DNode)
    (hn : n ∈ A.base.nodes) (t : List ℕ) (o : Bool) :
    ((dlegs A.base ks n.id t o (legLen n o)).map (relab (rnφ rn))).Perm
      (dlegs R.base ks n.id t o (legLen n o)) := by
  have hn' : n ∈ R.base.nodes := by rw [h.nodes]; exact hn
  have hid' : (R.base.nodes.map (·.id)).Nodup := by rw [h.nodes]; exact h.ids
  have mem1 : ∀ x, x ∈ dlegs A.base ks n.id t o (legLen n o) ↔ ∃ p ∈ A.base.legs,
      p.node = n.id ∧ p.out = o ∧ ∃ s ∈ cidx ks (A.base.tail p), x = dlab p (t ++ s) := by
    intro x
    rw [mem_dlegs]
    constructor
    · rintro ⟨i, hi, s, hs, rfl⟩
      exact ⟨_, (mem_legs_iff A.base h.ids n hn o i).2 hi, rfl, rfl, s, hs, rfl⟩
    · rintro ⟨⟨x0, o0, i⟩, hp, e1, e2, s, hs, rfl⟩
      simp only at e1 e2
      subst e1 e2
      exact ⟨i, (mem_legs_iff A.base h.ids n hn _ i).1 hp, s, hs, rfl⟩
  have mem2 : ∀ x, x ∈ dlegs R.base ks n.id t o (legLen n o) ↔ ∃ q ∈ A.base.legs,
      q.node = n.id ∧ q.out = o ∧ ∃ s ∈ cidx ks (R.base.tail q), x = dlab q (t ++ s) := by
    intro x
    rw [mem_dlegs, ← h.legsEq]
    constructor
    · rintro ⟨i, hi, s, hs, rfl⟩
      exact ⟨_, (mem_legs_iff R.base hid' n hn' o i).2 hi, rfl, rfl, s, hs, rfl⟩
    · rintro ⟨⟨x0, o0, i⟩, hp, e1, e2, s, hs, rfl⟩
      simp only at e1 e2
      subst e1 e2
      exact ⟨i, (mem_legs_iff R.base hid' n hn' _ i).1 hp, s, hs, rfl⟩
  refine (List.perm_ext_iff_of_nodup ?_ (dlegs_nodup _ _ _ _ _ _)).2 fun x => ?_
  · refine (dlegs_nodup _ _ _ _ _ _).map_on fun a ha b hb e => ?_
    obtain ⟨p, hp, -, -, s, -, rfl⟩ := (mem1 a).1 ha
    obtain ⟨p', hp', -, -, s', -, rfl⟩ := (mem1 b).1 hb
    rw [relab_dlab, relab_dlab] at e
    obtain ⟨e1, e2⟩ := dlab_inj e
    rw [h.inj p hp p' hp' e1]
    exact congrArg (dlab p') e2
  · rw [List.mem_map, mem2]
    constructor
    · rintro ⟨y, hy, rfl⟩
      obtain ⟨p, hp, e1, e2, s, hs, rfl⟩ := (mem1 y).1 hy
      obtain ⟨s1, s2, -⟩ := h.side p hp
      exact ⟨rn p, h.mem p hp, s1.trans e1, s2.trans e2, s, by rw [h.tail p hp]; exact hs,
        by rw [relab_dlab]; rfl⟩
    · rintro ⟨q, hq, e1, e2, s, hs, rfl⟩
      obtain ⟨p, hp, rfl⟩ := h.onto q hq
      obtain ⟨s1, s2, -⟩ := h.side p hp
      exact ⟨dlab p (t ++ s), (mem1 _).2 ⟨p, hp, s1.symm.trans e1, s2.symm.trans e2, s,
        by rw [← h.tail p hp]; exact hs, rfl⟩, by rw [relab_dlab]; rfl⟩

/-- Each node keeps its value at each copy under the renaming `rn`. -/
theorem LegPermOK.node (h : LegPermOK A R rn) (E : Env) (ks : ℕ → ℕ) (τ : ℕ → ℕ) (n : DNode)
    (hn : n ∈ A.base.nodes) (t : List ℕ) :
    val E τ ((A.dnodeL ks n t).rename (relab (rnφ rn))) = val E τ (R.dnodeL ks n t) := by
  by_cases hfix : ∀ o i, i < legLen n o → rn ⟨n.id, o, i⟩ = ⟨n.id, o, i⟩
  · refine congrArg (val E τ) (dnodeL_rename A R ks (rnφ rn) n n t t rfl rfl rfl rfl
      (h.phAt ks n t) fun o i hi => ?_)
    have hp := (mem_legs_iff A.base h.ids n hn o i).2 hi
    refine ⟨?_, fun s => ?_⟩
    · have := h.tail _ hp
      rw [hfix o i hi] at this
      exact this
    · simp only [rnφ, hfix o i hi]
  · push_neg at hfix
    obtain ⟨o, i, hi, hne⟩ := hfix
    have hp := (mem_legs_iff A.base h.ids n hn o i).2 hi
    obtain ⟨m, hm, em, hk⟩ := (h.side _ hp).2.2 hne
    have hmn : m = n := List.inj_on_of_nodup_map h.ids hm hn em
    subst hmn
    have po := h.sidePerm ks m hn t true
    have pi := h.sidePerm ks m hn t false
    refine val_nodeEquiv E τ _ _ ?_
    have a1 : ((A.dnodeL ks m t).rename (relab (rnφ rn))).outs =
        (dlegs A.base ks m.id t true (legLen m true)).map (relab (rnφ rn)) := rfl
    have a2 : ((A.dnodeL ks m t).rename (relab (rnφ rn))).ins =
        (dlegs A.base ks m.id t false (legLen m false)).map (relab (rnφ rn)) := rfl
    have b1 : (R.dnodeL ks m t).outs = dlegs R.base ks m.id t true (legLen m true) := rfl
    have b2 : (R.dnodeL ks m t).ins = dlegs R.base ks m.id t false (legLen m false) := rfl
    have a3 : ((A.dnodeL ks m t).rename (relab (rnφ rn))).kind = m.kind := rfl
    have b3 : (R.dnodeL ks m t).kind = m.kind := rfl
    have a4 : ((A.dnodeL ks m t).rename (relab (rnφ rn))).dim = m.dim := rfl
    have b4 : (R.dnodeL ks m t).dim = m.dim := rfl
    have a5 : ((A.dnodeL ks m t).rename (relab (rnφ rn))).ph = A.phAt ks m t := rfl
    have b5 : (R.dnodeL ks m t).ph = A.phAt ks m t := h.phAt ks m t
    unfold nodeEquiv
    rw [a3, b3, a4, b4, a5, b5]
    rcases hk with hz | hx
    · rw [hz]
      simp only [beq_self_eq_true, Bool.true_and, List.isPerm_iff]
      rw [a1, a2, b1, b2]
      exact po.append pi
    · rw [hx]
      simp only [beq_self_eq_true, Bool.true_and, List.isPerm_iff, Bool.and_eq_true]
      rw [a1, a2, b1, b2]
      exact ⟨po, pi⟩

/-- The node products agree under the renaming `rn`. -/
theorem LegPermOK.prod (h : LegPermOK A R rn) (E : Env) (ks : ℕ → ℕ) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ A.base.dlabs ks, τ' x = τ (relab (rnφ rn) x)) :
    R.prodL E ks τ = A.prodL E ks τ' := by
  have legOf : ∀ n ∈ A.base.nodes, ∀ o i, i < legLen n o →
      (⟨n.id, o, i⟩ : Port) ∈ A.base.legs :=
    fun n hn o i hi => (mem_legs_iff A.base h.ids n hn o i).2 hi
  have hv : ∀ n ∈ A.base.nodes, ∀ t ∈ cidx ks (A.base.nodeChain n.id),
      val E τ' (A.dnodeL ks n t) = val E τ ((A.dnodeL ks n t).rename (relab (rnφ rn))) := by
    intro n hn t ht
    rw [val_rename]
    refine val_reads E _ _ _ fun x hx => ?_
    have a1 : (A.dnodeL ks n t).outs = dlegs A.base ks n.id t true n.outs.length := rfl
    have a2 : (A.dnodeL ks n t).ins = dlegs A.base ks n.id t false n.ins.length := rfl
    simp only [Node.labels, List.mem_append, a1, a2] at hx
    have hmem : x ∈ A.base.dlabs ks := by
      rcases hx with hx | hx
      · obtain ⟨i, hi, s, hs, rfl⟩ := (mem_dlegs _ _ _ _ _ _ _).1 hx
        exact (mem_dlabs _ _ _).2 ⟨_, legOf n hn true i (by simpa [legLen] using hi), _,
          mem_cidx_append ks _ _ t s ht hs, rfl⟩
      · obtain ⟨i, hi, s, hs, rfl⟩ := (mem_dlegs _ _ _ _ _ _ _).1 hx
        exact (mem_dlabs _ _ _).2 ⟨_, legOf n hn false i (by simpa [legLen] using hi), _,
          mem_cidx_append ks _ _ t s ht hs, rfl⟩
    exact hτ x hmem
  unfold prodL
  rw [h.nodes]
  refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
  rw [h.nchain n hn]
  refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
  rw [hv n hn t ht, h.node E ks τ n hn t]

/-- Renumbering the legs of Z and X spiders keeps the digit sum `evalX`, `A`'s slots read from
`R`'s through `rnφ rn`. -/
theorem evalX_legPerm (h : LegPermOK A R rn) (E : Env) (ks : ℕ → ℕ) (σ' : ℕ → List ℕ → ℕ) :
    R.evalX E ks σ' = A.evalX E ks (bndMap A.base R.base (rnφ rn) σ') := by
  have hl := h.legsEq
  refine evalX_relabelφ A R E ks ks σ' (rnφ rn) (Diagram.dlabs_nodup _ _ h.legs)
    (Diagram.dlabs_nodup _ _ (hl ▸ h.legs)) h.bnd (fun p hp t ht => ?_) (fun q hq u hu => ?_)
    (fun p hp t _ p' hp' t' _ e => ⟨h.inj p hp p' hp' (congrArg Prod.fst e), congrArg Prod.snd e⟩)
    (fun p hp t _ => congrArg E.dimv (h.dimAt ks p hp t)) (fun x hx => ?_)
    (fun x hx => Or.inl (by rw [h.dpairsX]; exact List.mem_map.2 ⟨x, hx, rfl⟩))
    (fun y hy => ?_) h.bndL
    (fun p hp t _ => by rw [h.bndEq]; exact List.mem_map.2 ⟨p, hp, rfl⟩) (fun q hq u hu => ?_)
    (fun τ τ' hτ => h.prod E ks τ τ' hτ)
  · refine ⟨by rw [hl]; exact h.mem p hp, ?_⟩
    show t ∈ cidx ks (R.base.legChain (rn p))
    rw [(h.chain p hp).1]
    exact ht
  · rw [hl] at hq
    obtain ⟨p, hp, rfl⟩ := h.onto q hq
    exact ⟨p, hp, u, by rw [← (h.chain p hp).1]; exact hu, rfl⟩
  · rcases (mem_dpairsX A ks x).1 hx with hx | ⟨w, hw, t, ht, rfl⟩
    · rcases (mem_dpairs A ks x).1 hx with ⟨w, hw, t, ht, rfl⟩ | ⟨l, hl', hx⟩
      · obtain ⟨hw1, hw2, hch⟩ := h.wwf w hw
        exact ⟨hw1, ht, hw2, by rw [hch]; exact ht⟩
      · obtain ⟨hl1, hl2, he1, he2, hc2, hdl, hgd, hce1, hce2⟩ := h.lwf l hl'
        unfold linkPairs at hx
        rcases (mem_lp ks _ _ _ l x).1 hx with ⟨t, ht, h1, rfl⟩ | ⟨t, ht, h1, rfl⟩ |
          ⟨t, ht, h1, rfl⟩ | ⟨-, u, hu, rfl⟩
        · refine ⟨hl1, ht, hl2, ?_⟩
          rw [hc2]
          exact cidx_set ks _ t _ _ ht (by rw [hgd]; exact h1) hdl
        · exact ⟨hl1, ht, he1, by rw [hce1]; exact cidx_eraseIdx ks _ t _ ht⟩
        · exact ⟨he2, by rw [hce2]; exact cidx_eraseIdx ks _ t _ ht, hl2, by rw [hc2]; exact ht⟩
        · exact ⟨he2, by rw [hce2]; exact hu, he1, by rw [hce1]; exact hu⟩
    · obtain ⟨ha, hb, hlt, hcb⟩ := h.awf w hw
      exact ⟨ha, ht, hb, by rw [hcb]; exact pick_mem ks _ t w.perm ht hlt⟩
  · rw [h.dpairsX] at hy
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hy
    exact ⟨x, hx, Or.inl rfl⟩
  · rw [h.bndEq] at hq
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hq
    exact ⟨p, hp, u, by rw [← (h.chain p (h.bndL p hp)).1]; exact hu, rfl⟩

end perm

end LDiagram

end ArchytasZX
