/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Split

/-!
# Instantiating a node box

The digits of a diagram and of its leaf node box `b` copied `k` times correspond: digit `u ++ [j]`
of a leg on a node of `b` is digit `u` of the leg on copy `j`; digit `v ++ [j]` of an outside leg
wired into `b` is digit `v` of its `j`-th grown leg.
-/

namespace ArchytasZX

namespace Diagram

theorem lt_fresh (D : Diagram) {x : ℕ} (hx : x ∈ D.nodes.map (·.id)) : x < D.fresh := by
  unfold fresh
  suffices ∀ l : List ℕ, x ∈ l → x ≤ l.foldr max 0 by have := this _ hx; omega
  intro l hl
  induction l with
  | nil => simp at hl
  | cons a l ih =>
    rcases List.mem_cons.1 hl with rfl | hl
    · simp
    · simp only [List.foldr_cons]; exact le_trans (ih hl) (le_max_right _ _)

theorem fresh_le_cpId (D : Diagram) (b : DBox) (j x : ℕ) : D.fresh ≤ D.cpId b j x := by
  unfold cpId; omega

theorem cpId_inj (D : Diagram) (b : DBox) {j j' x x' : ℕ} (hx : x ∈ b.nodes) (hx' : x' ∈ b.nodes)
    (h : D.cpId b j x = D.cpId b j' x') : j = j' ∧ x = x' := by
  unfold cpId at h
  set s := b.nodes.mergeSort (· ≤ ·)
  have hs : s.length = b.nodes.length := List.length_mergeSort _
  have m1 : x ∈ s := (List.mergeSort_perm _ _).mem_iff.2 hx
  have m2 : x' ∈ s := (List.mergeSort_perm _ _).mem_iff.2 hx'
  have i1 := List.idxOf_lt_length_of_mem m1
  have i2 := List.idxOf_lt_length_of_mem m2
  rw [hs] at i1 i2
  have e : j * b.nodes.length + s.idxOf x = j' * b.nodes.length + s.idxOf x' := by omega
  have hj : j = j' := by
    have := congrArg (· / b.nodes.length) e
    rwa [Nat.mul_comm j, Nat.mul_comm j', Nat.mul_add_div (by omega), Nat.mul_add_div (by omega),
      Nat.div_eq_of_lt i1, Nat.div_eq_of_lt i2] at this
  subst hj
  refine ⟨rfl, ?_⟩
  have hi : s.idxOf x = s.idxOf x' := by omega
  rw [← List.getElem_idxOf (List.idxOf_lt_length_of_mem m1),
    ← List.getElem_idxOf (List.idxOf_lt_length_of_mem m2)]
  simp only [hi]

/-- Boxes linked by parents from `start`, each in the diagram. -/
def Linked (D : Diagram) (start : Option ℕ) (L : List DBox) : Prop :=
  (∀ y ∈ L, y ∈ D.boxes) ∧
  ∀ i (h : i < L.length), L[i].parent = (start :: L.map (fun b => some b.id))[i]'(by simp; omega)

theorem Linked.of_chain {D : Diagram} {pr : DBox → Bool} {s : Option ℕ} {L : List DBox}
    (h : ChainOK D pr s L) : Linked D s L := ⟨fun y hy => (h.1 y hy).1, h.2.2.1⟩

theorem Linked.dropLast {D : Diagram} {s : Option ℕ} {L : List DBox} (h : Linked D s L) :
    Linked D s L.dropLast := by
  refine ⟨fun y hy => h.1 y (List.mem_of_mem_dropLast hy), fun i hi => ?_⟩
  have hi' : i < L.length := by simp at hi; omega
  have := h.2 i hi'
  simp only [List.getElem_dropLast]
  rw [this]
  cases i with
  | zero => simp
  | succ i => simp [List.getElem_dropLast]

/-- Two parent-linked chains from `none` ending in the same box are equal. -/
theorem Linked.eq_of_last {D : Diagram} (hb : (D.boxes.map (·.id)).Nodup) :
    ∀ (n : ℕ) (L L' : List DBox), L.length = n → Linked D none L → Linked D none L' →
      L ≠ [] → L.getLast? = L'.getLast? → L = L' := by
  intro n
  induction n with
  | zero => intro L L' hn _ _ hne _; exact absurd (List.length_eq_zero_iff.1 hn) hne
  | succ n ih =>
    intro L L' hn h h' hne hl
    obtain ⟨A, b, rfl⟩ := List.eq_nil_or_concat L |>.resolve_left hne
    have hne' : L' ≠ [] := by rintro rfl; simp at hl
    obtain ⟨B, c, rfl⟩ := List.eq_nil_or_concat L' |>.resolve_left hne'
    simp only [List.concat_eq_append] at *
    rw [List.getLast?_concat, List.getLast?_concat, Option.some.injEq] at hl
    subst hl
    have pa := h.2 A.length (by simp)
    have pb := h'.2 B.length (by simp)
    simp only [List.getElem_concat_length] at pa pb
    have hA := h.dropLast
    have hB := h'.dropLast
    simp only [List.dropLast_concat] at hA hB
    cases A using List.reverseRecOn with
    | nil =>
      cases B using List.reverseRecOn with
      | nil => rfl
      | append_singleton B c' _ =>
        simp at pa pb
        rw [pa] at pb
        simp [List.getElem_append] at pb
    | append_singleton A a _ =>
      cases B using List.reverseRecOn with
      | nil =>
        simp at pa pb
        rw [pb] at pa
        simp [List.getElem_append] at pa
      | append_singleton B c' _ =>
        have ea : (none :: (A ++ [a] ++ [b]).map fun b => some b.id)[(A ++ [a]).length]'(by simp) =
            some a.id := by simp [List.getElem_append]
        have eb : (none :: (B ++ [c'] ++ [b]).map fun b => some b.id)[(B ++ [c']).length]'(by simp) =
            some c'.id := by simp [List.getElem_append]
        rw [ea] at pa
        rw [eb] at pb
        rw [pa] at pb
        have hac : a = c' := List.inj_on_of_nodup_map hb (hA.1 a (by simp)) (hB.1 c' (by simp))
          (by simpa using pb)
        subst hac
        have := ih (A ++ [a]) (B ++ [a]) (by simp at hn ⊢; omega) hA hB (by simp) (by simp)
        rw [this]

section box

variable {D : Diagram} {b : DBox} (hW : WF D) (hb : b ∈ D.boxes)
  (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id)

include hW hb hleaf in
theorem nodeBoxes_last {x : ℕ} (hx : x ∈ b.nodes) : ∃ A, D.nodeBoxes x = A ++ [b] := by
  have hxi := hW.nsub b hb x hx
  have hc := hW.nchain x hxi
  have hbL := hW.nall x b hb hx
  obtain ⟨i, hi, hib⟩ := List.mem_iff_getElem.1 hbL
  have hl := chain_leaf_last b hc hleaf i hi (by simp [hib])
  refine ⟨(D.nodeBoxes x).take i, ?_⟩
  conv_lhs => rw [← List.take_append_drop i (D.nodeBoxes x)]
  rw [List.drop_eq_getElem_cons hi, hib, List.drop_eq_nil_of_le (by omega)]

include hW hb hleaf in
theorem nodeBoxes_same {x z : ℕ} (hx : x ∈ b.nodes) (hz : z ∈ b.nodes) :
    D.nodeBoxes x = D.nodeBoxes z := by
  obtain ⟨A, hA⟩ := nodeBoxes_last hW hb hleaf hx
  obtain ⟨B, hB⟩ := nodeBoxes_last hW hb hleaf hz
  refine Linked.eq_of_last hW.bids _ _ _ rfl (Linked.of_chain (hW.nchain x (hW.nsub b hb x hx)))
    (Linked.of_chain (hW.nchain z (hW.nsub b hb z hz))) (by rw [hA]; simp) (by rw [hA, hB]; simp)

include hW hb hleaf in
theorem box_scope {y : DBox} (hy : y ∈ D.boxes) {x z : ℕ} (hx : x ∈ b.nodes) (hz : z ∈ b.nodes)
    (hzy : z ∈ y.nodes) : x ∈ y.nodes := by
  have h1 := hW.nall z y hy hzy
  rw [← nodeBoxes_same hW hb hleaf hx hz] at h1
  exact of_decide_eq_true ((hW.nchain x (hW.nsub b hb x hx)).1 y h1).2

include hW hb hleaf in
theorem portBoxes_in {p : Port} (hp : p.node ∈ b.nodes) : D.portBoxes p = [] := by
  obtain ⟨A, hA⟩ := nodeBoxes_last hW hb hleaf hp
  unfold portBoxes
  rw [hA, List.getLast?_concat]
  refine walk_eq D _ [] _ _ (by simp) (by simp) (fun y hy _ hpa => ?_) (by simp) (by simp)
  simp only [List.map_nil, Option.map_some, List.mem_singleton] at hpa
  exact absurd hpa (hleaf y hy)

theorem tail_drop (D : Diagram) (p r : Port) (h : D.partner p = some r) :
    D.tail p = (D.nodeChain r.node).drop (D.nodeChain p.node).length := by
  unfold tail
  rw [h]
  simp only
  split_ifs with hl
  · rfl
  · rw [List.drop_eq_nil_of_le (by omega)]

include hW hb hleaf in
theorem tail_in {p : Port} (hpl : p ∈ D.legs) (hp : p.node ∈ b.nodes) : D.tail p = [] := by
  cases hpart : D.partner p with
  | none =>
    unfold tail; rw [hpart]; simp only [portBoxes_in hW hb hleaf hp, List.map_nil]
  | some r =>
    rw [tail_drop D p r hpart]
    obtain ⟨w, hw, hwr⟩ := partner_some hpart
    obtain ⟨A, hA⟩ := nodeBoxes_last hW hb hleaf hp
    refine List.drop_eq_nil_of_le ?_
    unfold nodeChain
    rw [hA]
    have hpre := hW.wpre w hw
    by_cases hr : r.node ∈ b.nodes
    · rw [nodeBoxes_same hW hb hleaf hr hp, hA]
    · have : D.nodeBoxes r.node <+: A ++ [b] := by
        rcases hwr with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
        · rcases hpre with h | h
          · rw [hA] at h
            have hbm : b ∈ D.nodeBoxes w.2.node := h.subset (by simp)
            exact absurd (of_decide_eq_true ((hW.nchain _ (node_of_leg (wire_not_bnd hW hw).2.2.2)).1
              b hbm).2) hr
          · rwa [hA] at h
        · rcases hpre with h | h
          · rwa [hA] at h
          · rw [hA] at h
            have hbm : b ∈ D.nodeBoxes w.1.node := h.subset (by simp)
            exact absurd (of_decide_eq_true ((hW.nchain _ (node_of_leg (wire_not_bnd hW hw).2.2.1)).1
              b hbm).2) hr
      have := this.length_le
      simp only [List.length_map]
      omega

include hW hb hleaf in
/-- An outside node wired into `b` sits in boxes strictly enclosing `b`. -/
theorem outside_prefix {w : Port × Port} (hw : w ∈ D.wires) {y x : ℕ}
    (hends : (w.1.node = y ∧ w.2.node = x) ∨ (w.2.node = y ∧ w.1.node = x))
    (hy : y ∉ b.nodes) (hx : x ∈ b.nodes) {A : List DBox} (hA : D.nodeBoxes x = A ++ [b]) :
    D.nodeBoxes y <+: A := by
  have hyl : y ∈ D.nodes.map (·.id) := by
    obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
    rcases hends with ⟨rfl, -⟩ | ⟨rfl, -⟩
    · exact node_of_leg l1
    · exact node_of_leg l2
  have hpre : D.nodeBoxes y <+: A ++ [b] ∨ A ++ [b] <+: D.nodeBoxes y := by
    rw [← hA]
    rcases hends with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · exact hW.wpre w hw
    · exact (hW.wpre w hw).symm
  have hbn : b ∉ D.nodeBoxes y := fun hm =>
    hy (of_decide_eq_true ((hW.nchain y hyl).1 b hm).2)
  rcases hpre with h | h
  · obtain ⟨c, hc⟩ := h
    cases c using List.reverseRecOn with
    | nil => simp at hc; exact absurd (hc ▸ List.mem_append_right _ (List.mem_singleton_self b)) hbn
    | append_singleton c z _ =>
      rw [← List.append_assoc] at hc
      have := List.append_inj' hc rfl
      exact ⟨c, this.1⟩
  · exact absurd (h.subset (by simp)) hbn

end box

theorem DNode.legs_nodup (n : DNode) : n.legs.Nodup := by
  simp only [DNode.legs]
  refine List.nodup_append.2 ⟨(List.nodup_range).map fun a b e => by simpa using e,
    (List.nodup_range).map fun a b e => by simpa using e, ?_⟩
  simp

theorem DNode.mem_legs {n : DNode} {p : Port} : p ∈ n.legs ↔
    p.node = n.id ∧ p.pos < (if p.out then n.outs.length else n.ins.length) := by
  obtain ⟨x, o, i⟩ := p
  cases o <;> simp [DNode.legs, eq_comm, and_comm]

theorem legs_copy (n : DNode) (c : ℕ) :
    DNode.legs { n with id := c } = n.legs.map fun p => ⟨c, p.out, p.pos⟩ := by
  simp [DNode.legs, List.map_map, Function.comp_def]

theorem node_unique {D : Diagram} (hids : (D.nodes.map (·.id)).Nodup) {n m : DNode}
    (hn : n ∈ D.nodes) (hm : m ∈ D.nodes) (h : n.id = m.id) : n = m :=
  List.inj_on_of_nodup_map hids hn hm h

theorem leg_node {D : Diagram} {p : Port} (hp : p ∈ D.legs) : ∃ n ∈ D.nodes, p ∈ n.legs :=
  List.mem_flatMap.1 hp

/-- An outside node with its legs wired into the node box `b` grown to `k` legs. -/
def ngrow (D : Diagram) (b : DBox) (k : ℕ) (n : DNode) : DNode :=
  { n with outs := growSide (D.crossP b) k n.id true n.outs,
           ins := growSide (D.crossP b) k n.id false n.ins }

/-- The node `n` after copying the node box `b` `k` times. -/
def cnode (D : Diagram) (b : DBox) (k : ℕ) (n : DNode) : List DNode :=
  if n.id ∈ b.nodes then (List.range k).map fun j => { n with id := D.cpId b j n.id }
  else [D.ngrow b k n]

theorem cn_nodes (D : Diagram) (b : DBox) (k : ℕ) :
    (D.copyNodes b k).nodes = D.nodes.flatMap (D.cnode b k) := rfl

theorem cn_wires (D : Diagram) (b : DBox) (k : ℕ) :
    (D.copyNodes b k).wires = D.wires.flatMap fun w => (D.nimg b k w.1).zip (D.nimg b k w.2) := rfl

theorem nimg_in (D : Diagram) (b : DBox) (k : ℕ) {p : Port} (h : inS b p = true) :
    D.nimg b k p = (List.range k).map fun j => ⟨D.cpId b j p.node, p.out, p.pos⟩ := by
  simp [nimg, h]

theorem nimg_out (D : Diagram) (b : DBox) (k : ℕ) {p : Port} (h : inS b p = false) :
    D.nimg b k p = growImg (D.crossP b) k p := by
  simp [nimg, h]

theorem inS_iff (b : DBox) (p : Port) : inS b p = true ↔ p.node ∈ b.nodes := by simp [inS]

theorem cnode_legs (D : Diagram) (b : DBox) (k : ℕ) (n : DNode) (q : Port) :
    (∃ n' ∈ D.cnode b k n, q ∈ n'.legs) ↔ ∃ p ∈ n.legs, q ∈ D.nimg b k p := by
  unfold cnode
  by_cases hn : n.id ∈ b.nodes
  · rw [if_pos hn]
    simp only [List.mem_map, List.mem_range]
    constructor
    · rintro ⟨_, ⟨j, hj, rfl⟩, hq⟩
      rw [legs_copy, List.mem_map] at hq
      obtain ⟨p, hp, rfl⟩ := hq
      have hpn := (DNode.mem_legs.1 hp).1
      refine ⟨p, hp, ?_⟩
      rw [nimg_in D b k ((inS_iff b p).2 (hpn ▸ hn))]
      exact List.mem_map.2 ⟨j, List.mem_range.2 hj, by rw [hpn]⟩
    · rintro ⟨p, hp, hq⟩
      have hpn := (DNode.mem_legs.1 hp).1
      rw [nimg_in D b k ((inS_iff b p).2 (hpn ▸ hn)), List.mem_map] at hq
      obtain ⟨j, hj, rfl⟩ := hq
      refine ⟨_, ⟨j, List.mem_range.1 hj, rfl⟩, ?_⟩
      rw [legs_copy, List.mem_map]
      exact ⟨p, hp, by rw [hpn]⟩
  · rw [if_neg hn]
    simp only [List.mem_singleton, exists_eq_left]
    rw [ngrow, legs_grow (D.crossP b) k n, List.mem_flatMap]
    constructor
    · rintro ⟨p, hp, hq⟩
      have hpn := (DNode.mem_legs.1 hp).1
      refine ⟨p, hp, ?_⟩
      rw [nimg_out D b k (by simpa [inS, hpn] using hn)]; exact hq
    · rintro ⟨p, hp, hq⟩
      have hpn := (DNode.mem_legs.1 hp).1
      rw [nimg_out D b k (by simpa [inS, hpn] using hn)] at hq
      exact ⟨p, hp, hq⟩

theorem cn_memL (D : Diagram) (b : DBox) (k : ℕ) (q : Port) :
    q ∈ (D.copyNodes b k).legs ↔ ∃ p ∈ D.legs, q ∈ D.nimg b k p := by
  simp only [legs, cn_nodes, List.flatMap_assoc, List.mem_flatMap]
  constructor
  · rintro ⟨n, hn, n', hn', hq⟩
    obtain ⟨p, hp, hq⟩ := (cnode_legs D b k n q).1 ⟨n', hn', hq⟩
    exact ⟨p, ⟨n, hn, hp⟩, hq⟩
  · rintro ⟨p, ⟨n, hn, hp⟩, hq⟩
    obtain ⟨n', hn', hq⟩ := (cnode_legs D b k n q).2 ⟨p, hp, hq⟩
    exact ⟨n, hn, n', hn', hq⟩

section copy

variable {D : Diagram} {b : DBox} (k : ℕ) (hW : WF D) (hb : b ∈ D.boxes)

include hW hb in
theorem cn_ids : ((D.copyNodes b k).nodes.map (·.id)).Nodup := by
  have hnd : D.nodes.Nodup := hW.ids.of_map _
  rw [cn_nodes, List.map_flatMap]
  refine List.nodup_flatMap.2 ⟨fun n hn => ?_, hnd.pairwise_of_forall_ne fun n hn m hm hnm => ?_⟩
  · unfold cnode
    split_ifs with h
    · rw [List.map_map]
      refine List.nodup_range.map_on fun j _ j' _ e => ?_
      exact (cpId_inj D b h h e).1
    · simp
  · have hid : n.id ≠ m.id := fun e => hnm (node_unique hW.ids hn hm e)
    have ln := lt_fresh D (List.mem_map_of_mem (f := (·.id)) hn)
    have lm := lt_fresh D (List.mem_map_of_mem (f := (·.id)) hm)
    simp only [Function.onFun, List.disjoint_left, List.mem_map]
    rintro _ ⟨n', hn', rfl⟩ ⟨m', hm', e⟩
    unfold cnode at hn' hm'
    split_ifs at hn' hm' with h1 h2 h2 <;> simp only [List.mem_map, List.mem_range,
      List.mem_singleton] at hn' hm'
    · obtain ⟨j, -, rfl⟩ := hn'; obtain ⟨j', -, rfl⟩ := hm'
      exact hid (cpId_inj D b h2 h1 e).2.symm
    · obtain ⟨j, -, rfl⟩ := hn'; subst hm'
      have := fresh_le_cpId D b j n.id; simp only [ngrow] at e; omega
    · subst hn'; obtain ⟨j', -, rfl⟩ := hm'
      have := fresh_le_cpId D b j' m.id; simp only [ngrow] at e; omega
    · subst hn' hm'; exact hid (by simpa [ngrow] using e.symm)

include hW in
theorem nimg_node_lt {p : Port} (hp : p ∈ D.legs) (hq : inS b p = false) {q : Port}
    (hm : q ∈ D.nimg b k p) : q.node < D.fresh := by
  rw [nimg_out D b k hq] at hm
  rw [(mem_growImg hm).1]
  exact lt_fresh D (node_of_leg hp)

include hW in
theorem cn_disj {p p' : Port} (hp : p ∈ D.legs) (hp' : p' ∈ D.legs) {q : Port}
    (h1 : q ∈ D.nimg b k p) (h2 : q ∈ D.nimg b k p') : p = p' := by
  cases hi : inS b p <;> cases hi' : inS b p'
  · rw [nimg_out D b k hi] at h1
    rw [nimg_out D b k hi'] at h2
    obtain ⟨n, hn, hpn⟩ := leg_node hp
    obtain ⟨n', hn', hpn'⟩ := leg_node hp'
    have e1 := (mem_growImg h1).1
    have e2 := (mem_growImg h2).1
    have : n = n' := node_unique hW.ids hn hn'
      ((DNode.mem_legs.1 hpn).1.symm.trans (e1.symm.trans (e2.trans (DNode.mem_legs.1 hpn').1)))
    subst this
    have hnd := DNode.legs_nodup (D.ngrow b k n)
    rw [ngrow, legs_grow, List.nodup_flatMap] at hnd
    by_contra hne
    have : Std.Symm (Function.onFun List.Disjoint (growImg (D.crossP b) k)) :=
      ⟨fun _ _ h => List.disjoint_comm.1 h⟩
    exact (hnd.2.forall hpn hpn' hne) h1 h2
  · have := nimg_node_lt k hW hp hi h1
    rw [nimg_in D b k hi', List.mem_map] at h2
    obtain ⟨j, -, rfl⟩ := h2
    have := fresh_le_cpId D b j p'.node; simp at *; omega
  · have := nimg_node_lt k hW hp' hi' h2
    rw [nimg_in D b k hi, List.mem_map] at h1
    obtain ⟨j, -, rfl⟩ := h1
    have := fresh_le_cpId D b j p.node; simp at *; omega
  · rw [nimg_in D b k hi, List.mem_map] at h1
    rw [nimg_in D b k hi', List.mem_map] at h2
    obtain ⟨j, -, rfl⟩ := h1
    obtain ⟨j', -, e⟩ := h2
    simp only [Port.mk.injEq] at e
    have := (cpId_inj D b ((inS_iff b p').1 hi') ((inS_iff b p).1 hi) e.1).2
    obtain ⟨x, o, i⟩ := p; obtain ⟨x', o', i'⟩ := p'
    simp_all

theorem nimg_nodup (D : Diagram) (b : DBox) (k : ℕ) (p : Port) : (D.nimg b k p).Nodup := by
  unfold nimg
  split_ifs with h
  · refine List.nodup_range.map_on fun j _ j' _ e => ?_
    have hx := (inS_iff b p).1 h
    exact (cpId_inj D b hx hx (by simpa using e)).1
  · unfold growImg
    split_ifs
    · exact List.nodup_range.map_on fun j _ j' _ e => by simpa using e
    · simp

include hW in
theorem nimg_flat_nodup (l : List Port) (hl : l.Nodup) (hsub : ∀ p ∈ l, p ∈ D.legs) :
    (l.flatMap (D.nimg b k)).Nodup :=
  List.nodup_flatMap.2 ⟨fun p _ => nimg_nodup D b k p, hl.pairwise_of_forall_ne fun p hp p' hp' hne =>
    by
      simp only [Function.onFun, List.disjoint_left]
      exact fun q h1 h2 => hne (cn_disj k hW (hsub p hp) (hsub p' hp') h1 h2)⟩

end copy

theorem flatMap_swap_perm {α β γ : Type*} (l₁ : List α) (l₂ : List β) (f : α → β → γ) :
    (l₁.flatMap fun a => l₂.map (f a)).Perm (l₂.flatMap fun c => l₁.map (f · c)) := by
  induction l₁ with
  | nil => simp
  | cons a l₁ ih =>
    rw [List.flatMap_cons]
    refine (ih.append_left _).trans ?_
    clear ih
    induction l₂ with
    | nil => simp
    | cons c l₂ ih2 =>
      simp only [List.map_cons, List.flatMap_cons, List.cons_append]
      refine List.Perm.cons _ ?_
      rw [← List.append_assoc]
      refine (List.perm_append_comm.append_right _).trans ?_
      rw [List.append_assoc]
      exact ih2.append_left _

theorem spliceN_perm (D : Diagram) (b : DBox) (k : ℕ) (refs : List Port) :
    (D.spliceN b k refs).Perm (refs.flatMap (D.nimg b k)) := by
  unfold spliceN
  simp only
  split_ifs with h
  · exact List.Perm.refl _
  · set at_ := refs.findIdx (inS b)
    have hT : ∀ r ∈ refs.take at_, inS b r = false := fun r hr => by
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hr
      simp only [List.getElem_take]
      have hlt : i < at_ := by simp at hi; omega
      have := List.not_of_lt_findIdx (xs := refs) (p := inS b) hlt
      simpa using this
    have hhere : refs.filter (inS b) = (refs.drop at_).filter (inS b) := by
      conv_lhs => rw [← List.take_append_drop at_ refs]
      rw [List.filter_append, List.filter_eq_nil_iff.2 fun r hr => by simp [hT r hr],
        List.nil_append]
    conv_rhs => rw [← List.take_append_drop at_ refs]
    rw [List.flatMap_append, List.append_assoc]
    refine List.Perm.append_left _ ?_
    refine List.Perm.trans ?_ ((List.filter_append_perm (inS b) (refs.drop at_)).flatMap_right _)
    rw [List.flatMap_append]
    refine List.Perm.append_right _ ?_
    rw [← hhere]
    refine List.Perm.trans (flatMap_swap_perm (refs.filter (inS b)) (List.range k)
      (fun p j => (⟨D.cpId b j p.node, p.out, p.pos⟩ : Port))).symm ?_
    refine List.Perm.of_eq (List.flatMap_congr fun p hp => ?_)
    rw [nimg_in D b k (List.mem_filter.1 hp).2]

theorem cn_bnd_perm (D : Diagram) (b : DBox) (k : ℕ) :
    (D.copyNodes b k).bnd.Perm (D.bnd.flatMap (D.nimg b k)) := by
  simp only [bnd, copyNodes, List.flatMap_append]
  exact (spliceN_perm D b k _).append (spliceN_perm D b k _)

/-- The copies of the node box `b`'s nodes. -/
def copies (D : Diagram) (b : DBox) (k : ℕ) : List ℕ :=
  (List.range k).flatMap fun j => (b.nodes.mergeSort (· ≤ ·)).map (D.cpId b j)

/-- A box after copying the node box `b`: its nodes in `b` replaced by every copy. -/
def nbox (D : Diagram) (b : DBox) (k : ℕ) (y : DBox) : DBox :=
  { y with nodes := y.nodes.filter (· ∉ b.nodes) ++ (if y.nodes.any (· ∈ b.nodes) then D.copies b k else []),
           ports := y.ports.flatMap (D.nimg b k) }

theorem cn_boxes (D : Diagram) (b : DBox) (k : ℕ) : (D.copyNodes b k).boxes =
    ((D.boxes.filter (·.id ≠ b.id)).map (D.nbox b k)).filter fun x => x.nodes ≠ [] ∨ x.ports ≠ [] :=
  rfl

theorem cn_mem_boxes (D : Diagram) (b : DBox) (k : ℕ) (y' : DBox) : y' ∈ (D.copyNodes b k).boxes ↔
    ∃ y ∈ D.boxes, decide (y.id ≠ b.id) = true ∧ y' = D.nbox b k y ∧
      ((D.nbox b k y).nodes ≠ [] ∨ (D.nbox b k y).ports ≠ []) := by
  rw [cn_boxes]
  simp only [List.mem_filter, List.mem_map, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨y, ⟨hy, hne⟩, rfl⟩, h⟩
    exact ⟨y, hy, by simpa using hne, rfl, by simpa using h⟩
  · rintro ⟨y, hy, hne, rfl, h⟩
    exact ⟨⟨y, ⟨hy, by simpa using hne⟩, rfl⟩, by simpa using h⟩

theorem mem_copies (D : Diagram) (b : DBox) (k c : ℕ) :
    c ∈ D.copies b k ↔ ∃ j < k, ∃ x ∈ b.nodes, c = D.cpId b j x := by
  simp only [copies, List.mem_flatMap, List.mem_range, List.mem_map,
    (List.mergeSort_perm _ _).mem_iff]
  constructor
  · rintro ⟨j, hj, x, hx, rfl⟩; exact ⟨j, hj, x, hx, rfl⟩
  · rintro ⟨j, hj, x, hx, rfl⟩; exact ⟨j, hj, x, hx, rfl⟩

section chains

variable {D : Diagram} {b : DBox} (k : ℕ) (hW : WF D) (hb : b ∈ D.boxes)
  (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id)

include hW in
theorem nbox_out {y : DBox} (hy : y ∈ D.boxes) {x : ℕ} (hx : x ∈ D.nodes.map (·.id))
    (hxb : x ∉ b.nodes) : x ∈ (D.nbox b k y).nodes ↔ x ∈ y.nodes := by
  simp only [nbox, List.mem_append, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro (⟨h, -⟩ | h)
    · exact h
    · split_ifs at h
      · obtain ⟨j, -, z, -, rfl⟩ := (mem_copies D b k _).1 h
        have := lt_fresh D hx; have := fresh_le_cpId D b j z; omega
      · simp at h
  · intro h; exact Or.inl ⟨h, hxb⟩

include hW hb hleaf in
theorem nbox_copy {y : DBox} (hy : y ∈ D.boxes) {x j : ℕ} (hx : x ∈ b.nodes) (hj : j < k) :
    D.cpId b j x ∈ (D.nbox b k y).nodes ↔ x ∈ y.nodes := by
  simp only [nbox, List.mem_append, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro (⟨h, -⟩ | h)
    · have := lt_fresh D (hW.nsub y hy _ h); have := fresh_le_cpId D b j x; omega
    · split_ifs at h with hany
      · obtain ⟨z, hz, hzb⟩ := List.any_eq_true.1 hany
        exact box_scope hW hb hleaf hy hx (of_decide_eq_true hzb) hz
      · simp at h
  · intro h
    refine Or.inr ?_
    rw [if_pos (List.any_eq_true.2 ⟨x, h, by simpa using hx⟩)]
    exact (mem_copies D b k _).2 ⟨j, hj, x, hx, rfl⟩

include hW hb in
theorem cn_nchain_out {x : ℕ} (hx : x ∈ D.nodes.map (·.id)) (hxb : x ∉ b.nodes) :
    ChainOK (D.copyNodes b k) (fun y => decide (x ∈ y.nodes)) none
      ((D.nodeBoxes x).map (D.nbox b k)) := by
  have hc := hW.nchain x hx
  have hnb : ∀ y ∈ D.nodeBoxes x, decide (y.id ≠ b.id) = true := fun y hy => by
    have hy' := hc.1 y hy
    simp only [decide_eq_true_eq]
    intro e
    have : y = b := List.inj_on_of_nodup_map hW.bids hy'.1 hb e
    subst this
    exact hxb (of_decide_eq_true hy'.2)
  have h' := hc.transport (D' := D.copyNodes b k) (pr' := fun y => decide (x ∈ y.nodes))
    (D.nbox b k) (fun y => decide (y.id ≠ b.id)) (cn_mem_boxes D b k)
    (fun y hy => ⟨rfl, rfl, by simp only [decide_eq_decide]; exact nbox_out k hW hy hx hxb⟩)
    (fun y hy => Or.inl (List.ne_nil_of_mem (of_decide_eq_true hy)))
    (fun i hi hkp => absurd (hnb _ (List.getElem_mem hi)) (by simp [hkp]))
  rwa [List.filter_eq_self.2 hnb] at h'

include hW hb hleaf in
theorem cn_nchain_copy {x j : ℕ} (hx : x ∈ b.nodes) (hj : j < k) {A : List DBox}
    (hA : D.nodeBoxes x = A ++ [b]) :
    ChainOK (D.copyNodes b k) (fun y => decide (D.cpId b j x ∈ y.nodes)) none (A.map (D.nbox b k)) := by
  have hc := hW.nchain x (hW.nsub b hb x hx)
  have h' := hc.transport (D' := D.copyNodes b k) (pr' := fun y => decide (D.cpId b j x ∈ y.nodes))
    (D.nbox b k) (fun y => decide (y.id ≠ b.id)) (cn_mem_boxes D b k)
    (fun y hy => ⟨rfl, rfl, by simp only [decide_eq_decide]; exact nbox_copy k hW hb hleaf hy hx hj⟩)
    (fun y hy => Or.inl (List.ne_nil_of_mem (of_decide_eq_true hy)))
    (chain_leaf_last b hc hleaf)
  have hf : (D.nodeBoxes x).filter (fun y => decide (y.id ≠ b.id)) = A := by
    rw [hA, List.filter_append]
    have hAb : ∀ y ∈ A, decide (y.id ≠ b.id) = true := fun y hy => by
      have hnd := (List.nodup_cons.1 hc.2.2.2).2
      rw [hA, List.map_append, List.nodup_append] at hnd
      simp only [decide_eq_true_eq]
      intro e
      exact hnd.2.2 (some y.id) (List.mem_map.2 ⟨y, hy, rfl⟩) (some b.id) (by simp) (by rw [e])
    rw [List.filter_eq_self.2 hAb]
    simp
  rwa [hf] at h'

end chains

/-- A leg that splits along the node box `b`: on it, or wired into it. -/
def gN (D : Diagram) (b : DBox) (p : Port) : Bool := inS b p || D.crossP b p

theorem nimg_length (D : Diagram) (b : DBox) (k : ℕ) (p : Port) :
    (D.nimg b k p).length = if D.gN b p then k else 1 := by
  unfold nimg gN
  cases h : inS b p
  · simp only [Bool.false_eq_true, ↓reduceIte, Bool.false_or]
    unfold growImg; split_ifs <;> simp
  · simp

section wires

variable {D : Diagram} {b : DBox} (k : ℕ) (hW : WF D)

include hW in
theorem crossP_iff {p : Port} : D.crossP b p = true ↔
    inS b p = false ∧ ∃ r, D.partner p = some r ∧ inS b r = true := by
  unfold crossP
  simp only [Bool.and_eq_true, Bool.not_eq_true', List.any_eq_true, Bool.or_eq_true,
    beq_iff_eq, decide_eq_true_eq]
  refine and_congr_right fun _ => ⟨?_, ?_⟩
  · rintro ⟨w, hw, (⟨rfl, h⟩ | ⟨rfl, h⟩)⟩
    · exact ⟨w.2, (partner_of_wire hW hw).1, h⟩
    · exact ⟨w.1, (partner_of_wire hW hw).2, h⟩
  · rintro ⟨r, hr, h⟩
    obtain ⟨w, hw, (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)⟩ := partner_some hr
    · exact ⟨w, hw, Or.inl ⟨rfl, h⟩⟩
    · exact ⟨w, hw, Or.inr ⟨rfl, h⟩⟩

include hW in
theorem gN_wire {w : Port × Port} (hw : w ∈ D.wires) : D.gN b w.1 = D.gN b w.2 := by
  obtain ⟨p1, p2⟩ := partner_of_wire hW hw
  unfold gN
  cases h1 : inS b w.1 <;> cases h2 : inS b w.2
  · have c1 : D.crossP b w.1 = false := by
      by_contra hc
      obtain ⟨-, r, hr, hri⟩ := (crossP_iff hW).1 (by simpa using hc)
      rw [p1] at hr; cases hr; simp [h2] at hri
    have c2 : D.crossP b w.2 = false := by
      by_contra hc
      obtain ⟨-, r, hr, hri⟩ := (crossP_iff hW).1 (by simpa using hc)
      rw [p2] at hr; cases hr; simp [h1] at hri
    simp [c1, c2]
  · have : D.crossP b w.1 = true := (crossP_iff hW).2 ⟨h1, w.2, p1, h2⟩
    simp [this]
  · have : D.crossP b w.2 = true := (crossP_iff hW).2 ⟨h2, w.1, p2, h1⟩
    simp [this]
  · simp

include hW in
theorem nimg_wire_len {w : Port × Port} (hw : w ∈ D.wires) :
    (D.nimg b k w.1).length = (D.nimg b k w.2).length := by
  rw [nimg_length, nimg_length, gN_wire hW hw]

include hW in
theorem cn_wires_fst : (D.copyNodes b k).wires.map (·.1) = (D.wires.map (·.1)).flatMap (D.nimg b k) := by
  rw [cn_wires, List.map_flatMap, List.flatMap_map]
  refine List.flatMap_congr fun w hw => ?_
  exact List.map_fst_zip (le_of_eq (nimg_wire_len k hW hw))

include hW in
theorem cn_wires_snd : (D.copyNodes b k).wires.map (·.2) = (D.wires.map (·.2)).flatMap (D.nimg b k) := by
  rw [cn_wires, List.map_flatMap, List.flatMap_map]
  refine List.flatMap_congr fun w hw => ?_
  exact List.map_snd_zip (le_of_eq (nimg_wire_len k hW hw).symm)

end wires

section perm

variable {D : Diagram} {b : DBox} (k : ℕ) (hW : WF D) (hb : b ∈ D.boxes)

include hW hb in
theorem cn_legs_perm : (D.copyNodes b k).legs.Perm (D.legs.flatMap (D.nimg b k)) := by
  refine (List.perm_ext_iff_of_nodup (legs_nodup_of _ (cn_ids k hW hb))
    (nimg_flat_nodup k hW _ (legs_nodup_of D hW.ids) fun _ h => h)).2 fun q => ?_
  rw [cn_memL, List.mem_flatMap]

include hW hb in
theorem cn_perm : ((D.copyNodes b k).wires.map (·.1) ++ ((D.copyNodes b k).wires.map (·.2) ++
    (D.copyNodes b k).bnd)).Perm (D.copyNodes b k).legs := by
  rw [cn_wires_fst k hW, cn_wires_snd k hW]
  refine List.Perm.trans ?_ (cn_legs_perm k hW hb).symm
  refine List.Perm.trans ?_ (hW.perm.flatMap_right _)
  rw [List.flatMap_append, List.flatMap_append]
  exact ((cn_bnd_perm D b k).append_left _).append_left _

include hW hb in
theorem cn_ends_nodup : ((D.copyNodes b k).wires.map (·.1) ++ ((D.copyNodes b k).wires.map (·.2) ++
    (D.copyNodes b k).bnd)).Nodup :=
  (cn_perm k hW hb).nodup_iff.2 (legs_nodup_of _ (cn_ids k hW hb))

include hW hb in
theorem cn_partner_wire {w : Port × Port} (hw : w ∈ D.wires) (j : ℕ)
    (h1 : j < (D.nimg b k w.1).length) (h2 : j < (D.nimg b k w.2).length) :
    (D.copyNodes b k).partner (D.nimg b k w.1)[j] = some (D.nimg b k w.2)[j] ∧
    (D.copyNodes b k).partner (D.nimg b k w.2)[j] = some (D.nimg b k w.1)[j] := by
  have hw' : ((D.nimg b k w.1)[j], (D.nimg b k w.2)[j]) ∈ (D.copyNodes b k).wires := by
    rw [cn_wires]; exact List.mem_flatMap.2 ⟨w, hw, getElem_mem_zip _ _ j h1 h2⟩
  exact partner_of_wire_nd (cn_ends_nodup k hW hb) hw'

include hW hb in
theorem cn_partner_bnd {p : Port} (hp : p ∈ D.bnd) {q : Port} (hq : q ∈ D.nimg b k p) :
    (D.copyNodes b k).partner q = none := by
  cases h : (D.copyNodes b k).partner q with
  | none => rfl
  | some r =>
    exfalso
    obtain ⟨w', hw', hwr⟩ := partner_some h
    have hend : q ∈ (D.wires.map (·.1) ++ D.wires.map (·.2)).flatMap (D.nimg b k) := by
      rw [List.flatMap_append, ← cn_wires_fst k hW, ← cn_wires_snd k hW]
      rcases hwr with ⟨rfl, -⟩ | ⟨rfl, -⟩
      · exact List.mem_append_left _ (List.mem_map.2 ⟨w', hw', rfl⟩)
      · exact List.mem_append_right _ (List.mem_map.2 ⟨w', hw', rfl⟩)
    obtain ⟨p', hp', hqp'⟩ := List.mem_flatMap.1 hend
    have hl' : p' ∈ D.legs := hW.perm.subset (by
      rcases List.mem_append.1 hp' with h | h
      · exact List.mem_append_left _ h
      · exact List.mem_append_right _ (List.mem_append_left _ h))
    have := cn_disj k hW (bnd_legs hW hp) hl' hq hqp'
    subst this
    have hnd := hW.perm.nodup_iff.2 (legs_nodup_of D hW.ids)
    rw [List.nodup_append] at hnd
    rcases List.mem_append.1 hp' with h | h
    · exact hnd.2.2 _ h _ (List.mem_append_right _ hp) rfl
    · exact (List.nodup_append.1 hnd.2.1).2.2 _ h _ hp rfl

end perm

theorem nimg_in_mem (D : Diagram) (b : DBox) (k : ℕ) {p q : Port} (h : inS b p = true)
    (hq : q ∈ D.nimg b k p) : ∃ j < k, q = ⟨D.cpId b j p.node, p.out, p.pos⟩ := by
  rw [nimg_in D b k h, List.mem_map] at hq
  obtain ⟨j, hj, rfl⟩ := hq
  exact ⟨j, List.mem_range.1 hj, rfl⟩

theorem nimg_out_mem (D : Diagram) (b : DBox) (k : ℕ) {p q : Port} (h : inS b p = false)
    (hq : q ∈ D.nimg b k p) : q.node = p.node := by
  rw [nimg_out D b k h] at hq; exact (mem_growImg hq).1

theorem nbox_mult (D : Diagram) (b : DBox) (k : ℕ) : (·.mult) ∘ D.nbox b k = (·.mult) := rfl

section legchains

variable {D : Diagram} {b : DBox} (k : ℕ) (hW : WF D) (hb : b ∈ D.boxes)
  (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (hbp : b.ports = [])

include hW hb hleaf in
theorem cn_chain {p : Port} (hp : p ∈ D.legs) {q : Port} (hq : q ∈ D.nimg b k p) :
    (D.copyNodes b k).nodeBoxes q.node = (if inS b p then (D.nodeBoxes p.node).dropLast
      else D.nodeBoxes p.node).map (D.nbox b k) := by
  cases h : inS b p
  · rw [nimg_out_mem D b k h hq]
    have hx := node_of_leg hp
    have hxb : p.node ∉ b.nodes := by simpa [inS] using h
    have h' := cn_nchain_out k hW hb hx hxb
    simp only [Bool.false_eq_true, ↓reduceIte]
    exact h'.walk _ h'.length_le
  · obtain ⟨j, hj, rfl⟩ := nimg_in_mem D b k h hq
    have hx : p.node ∈ b.nodes := (inS_iff b p).1 h
    obtain ⟨A, hA⟩ := nodeBoxes_last hW hb hleaf hx
    have h' := cn_nchain_copy k hW hb hleaf hx hj hA
    simp only [↓reduceIte, hA, List.dropLast_concat]
    exact h'.walk _ h'.length_le

include hW hb hleaf in
theorem cn_nodeChain {p : Port} (hp : p ∈ D.legs) {q : Port} (hq : q ∈ D.nimg b k p) :
    (D.copyNodes b k).nodeChain q.node =
      if inS b p then (D.nodeChain p.node).dropLast else D.nodeChain p.node := by
  unfold nodeChain
  rw [cn_chain k hW hb hleaf hp hq, List.map_map, nbox_mult]
  split_ifs <;> simp [List.map_dropLast]

include hW hb hbp in
theorem cn_pchain_out {p : Port} (hp : p ∈ D.bnd) (hpi : inS b p = false) {q : Port}
    (hq : q ∈ D.nimg b k p) :
    ChainOK (D.copyNodes b k) (fun y => decide (q ∈ y.ports))
      (((D.copyNodes b k).nodeBoxes q.node).getLast?.map (·.id)) ((D.portBoxes p).map (D.nbox b k)) := by
  have hpl := bnd_legs hW hp
  have hqn := nimg_out_mem D b k hpi hq
  have hx := node_of_leg hpl
  have hxb : p.node ∉ b.nodes := by simpa [inS] using hpi
  have hstart : ((D.copyNodes b k).nodeBoxes q.node).getLast?.map (·.id) =
      (D.nodeBoxes p.node).getLast?.map (·.id) := by
    rw [hqn]
    have h' := cn_nchain_out k hW hb hx hxb
    unfold nodeBoxes
    rw [h'.walk _ h'.length_le, List.getLast?_map, Option.map_map]
    rfl
  have hc := hW.pchain p hp
  have hkeep : ∀ y ∈ D.portBoxes p, decide (y.id ≠ b.id) = true := fun y hy => by
    simp only [decide_eq_true_eq]
    intro e
    have : y = b := List.inj_on_of_nodup_map hW.bids (hc.1 y hy).1 hb e
    subst this
    have := of_decide_eq_true (hc.1 _ hy).2
    rw [hbp] at this; simp at this
  have h' := hc.transport (D' := D.copyNodes b k) (pr' := fun y => decide (q ∈ y.ports))
    (D.nbox b k) (fun y => decide (y.id ≠ b.id)) (cn_mem_boxes D b k) (fun y hy => ⟨rfl, rfl, by
      simp only [nbox, List.mem_flatMap, decide_eq_decide]
      constructor
      · rintro ⟨p'', hp'', hq''⟩
        have := cn_disj k hW hpl (bnd_legs hW (hW.pbnd y hy p'' hp'')) hq hq''
        exact this ▸ hp''
      · exact fun h => ⟨p, h, hq⟩⟩)
    (fun y hy => Or.inr (List.ne_nil_of_mem (of_decide_eq_true hy)))
    (fun i hi hkp => absurd (hkeep _ (List.getElem_mem hi)) (by simp [hkp]))
  rw [List.filter_eq_self.2 hkeep] at h'
  rw [hstart]
  exact h'

include hW hb hbp in
theorem cn_portBoxes_out {p : Port} (hp : p ∈ D.bnd) (hpi : inS b p = false) {q : Port}
    (hq : q ∈ D.nimg b k p) :
    (D.copyNodes b k).portBoxes q = (D.portBoxes p).map (D.nbox b k) := by
  have h' := cn_pchain_out k hW hb hbp hp hpi hq
  exact h'.walk _ h'.length_le

include hW hb in
theorem cn_pchain_in {p : Port} (hp : p ∈ D.bnd) {q : Port}
    (hq : q ∈ D.nimg b k p) (hpb : D.portBoxes p = []) :
    ChainOK (D.copyNodes b k) (fun y => decide (q ∈ y.ports))
      (((D.copyNodes b k).nodeBoxes q.node).getLast?.map (·.id)) [] := by
  refine ⟨by simp, fun y' hy' hpr _ => ?_, by simp, by simp⟩
  exfalso
  obtain ⟨y, hy, -, rfl, -⟩ := (cn_mem_boxes D b k y').1 hy'
  simp only [nbox, List.mem_flatMap, decide_eq_true_eq] at hpr
  obtain ⟨p'', hp'', hq''⟩ := hpr
  have := cn_disj k hW (bnd_legs hW hp) (bnd_legs hW (hW.pbnd y hy p'' hp'')) hq hq''
  subst this
  have := hW.pall p hp y hy hp''
  rw [hpb] at this; simp at this

include hW hb in
theorem cn_portBoxes_in {p : Port} (hp : p ∈ D.bnd) {q : Port}
    (hq : q ∈ D.nimg b k p) (hpb : D.portBoxes p = []) : (D.copyNodes b k).portBoxes q = [] := by
  unfold portBoxes
  refine walk_eq _ _ [] _ _ (by simp) (by simp) (fun y' hy' hpr _ => ?_) (by simp) (by simp)
  exfalso
  obtain ⟨y, hy, -, rfl, -⟩ := (cn_mem_boxes D b k y').1 hy'
  simp only [nbox, List.mem_flatMap, decide_eq_true_eq] at hpr
  obtain ⟨p'', hp'', hq''⟩ := hpr
  have := cn_disj k hW (bnd_legs hW hp) (bnd_legs hW (hW.pbnd y hy p'' hp'')) hq hq''
  subst this
  have := hW.pall p hp y hy hp''
  rw [hpb] at this; simp at this

end legchains

section legchain2

variable {D : Diagram} {b : DBox} (k : ℕ) (hW : WF D) (hb : b ∈ D.boxes)
  (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (hbp : b.ports = [])

include hW hb in
theorem cn_partner_leg {p r : Port} (hr : D.partner p = some r) (j : ℕ)
    (hj : j < (D.nimg b k p).length) : ∃ hjr : j < (D.nimg b k r).length,
      (D.copyNodes b k).partner (D.nimg b k p)[j] = some (D.nimg b k r)[j] := by
  obtain ⟨w, hw, hwr⟩ := partner_some hr
  have hl := nimg_wire_len k hW (b := b) hw
  rcases hwr with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · exact ⟨by omega, (cn_partner_wire k hW hb hw j hj (by omega)).1⟩
  · exact ⟨by omega, (cn_partner_wire k hW hb hw j (by omega) hj).2⟩

theorem partner_leg {D : Diagram} (hW : WF D) {p r : Port} (h : D.partner p = some r) : r ∈ D.legs := by
  obtain ⟨w, hw, hwr⟩ := partner_some h
  obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
  rcases hwr with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> assumption

theorem nodeChain_last {D : Diagram} {b : DBox} (hW : WF D) (hb : b ∈ D.boxes)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) {x : ℕ} (hx : x ∈ b.nodes) :
    ∃ A, D.nodeBoxes x = A ++ [b] ∧ D.nodeChain x = A.map (·.mult) ++ [b.mult] := by
  obtain ⟨A, hA⟩ := nodeBoxes_last hW hb hleaf hx
  exact ⟨A, hA, by simp [nodeChain, hA]⟩

include hW hb hleaf hbp in
theorem cn_legChain_split {p : Port} (hp : p ∈ D.legs) (hg : D.gN b p = true) :
    ∃ c, D.legChain p = c ++ [b.mult] ∧ ∀ q ∈ D.nimg b k p, (D.copyNodes b k).legChain q = c := by
  unfold gN at hg
  cases hi : inS b p
  · -- wired into the box
    rw [hi, Bool.false_or] at hg
    obtain ⟨-, r, hr, hri⟩ := (crossP_iff hW).1 hg
    have hrl := partner_leg hW hr
    have hx : r.node ∈ b.nodes := (inS_iff b r).1 hri
    obtain ⟨A, hA, hcA⟩ := nodeChain_last hW hb hleaf hx
    obtain ⟨w, hw, hwr⟩ := partner_some hr
    have hy : p.node ∉ b.nodes := by simpa [inS] using hi
    have hpre := outside_prefix hW hb hleaf hw (y := p.node) (x := r.node)
      (by rcases hwr with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp) hy hx hA
    have hlen : (D.nodeChain p.node).length ≤ (A.map (·.mult)).length := by
      unfold nodeChain; simpa using hpre.length_le
    refine ⟨D.nodeChain p.node ++ (A.map (·.mult)).drop (D.nodeChain p.node).length, ?_, ?_⟩
    · unfold legChain
      rw [tail_drop D p r hr, hcA, List.drop_append_of_le_length hlen, List.append_assoc]
    · intro q hq
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hq
      obtain ⟨hjr, hpq⟩ := cn_partner_leg k hW hb hr j hj
      unfold legChain
      rw [tail_drop _ _ _ hpq, cn_nodeChain k hW hb hleaf hp (List.getElem_mem hj),
        cn_nodeChain k hW hb hleaf hrl (List.getElem_mem hjr), if_neg (by simp [hi]), if_pos hri,
        hcA, List.dropLast_concat]
  · -- on the box
    have hx : p.node ∈ b.nodes := (inS_iff b p).1 hi
    obtain ⟨A, hA, hcA⟩ := nodeChain_last hW hb hleaf hx
    refine ⟨A.map (·.mult), by rw [legChain, tail_in hW hb hleaf hp hx, hcA, List.append_nil], ?_⟩
    intro q hq
    obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hq
    have hc' := cn_nodeChain k hW hb hleaf hp (List.getElem_mem hj)
    rw [if_pos hi, hcA, List.dropLast_concat] at hc'
    unfold legChain
    rw [hc']
    cases hpart : D.partner p with
    | none =>
      have hpb := bnd_of_partner_none hW hp hpart
      have hnone := cn_partner_bnd k hW hb hpb (List.getElem_mem hj)
      unfold tail
      rw [hnone]
      simp only
      rw [cn_portBoxes_in k hW hb hpb (List.getElem_mem hj) (portBoxes_in hW hb hleaf hx)]
      simp
    | some r =>
      obtain ⟨hjr, hpq⟩ := cn_partner_leg k hW hb hpart j hj
      have hrl := partner_leg hW hpart
      rw [tail_drop _ _ _ hpq, hc', cn_nodeChain k hW hb hleaf hrl (List.getElem_mem hjr)]
      suffices hd : List.drop (A.map (·.mult)).length
          (if inS b r = true then (D.nodeChain r.node).dropLast else D.nodeChain r.node) = [] by
        rw [hd, List.append_nil]
      refine List.drop_eq_nil_of_le ?_
      split_ifs with hri
      · unfold nodeChain
        rw [nodeBoxes_same hW hb hleaf ((inS_iff b r).1 hri) hx, hA]
        simp
      · obtain ⟨w, hw, hwr⟩ := partner_some hpart
        have hpre := outside_prefix hW hb hleaf hw (y := r.node) (x := p.node)
          (by rcases hwr with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp) (by simpa [inS] using hri) hx hA
        unfold nodeChain; simpa using hpre.length_le

include hW hb hleaf hbp in
theorem cn_legChain_keep {p : Port} (hp : p ∈ D.legs) (hg : D.gN b p = false) :
    ∃ q, D.nimg b k p = [q] ∧ (D.copyNodes b k).legChain q = D.legChain p := by
  unfold gN at hg
  simp only [Bool.or_eq_false_iff] at hg
  obtain ⟨hi, hc⟩ := hg
  refine ⟨⟨p.node, p.out, shiftPos (D.crossP b) k p⟩, by
    rw [nimg_out D b k hi, growImg_of_not _ _ _ hc], ?_⟩
  have hq : (⟨p.node, p.out, shiftPos (D.crossP b) k p⟩ : Port) ∈ D.nimg b k p := by
    rw [nimg_out D b k hi, growImg_of_not _ _ _ hc]; simp
  unfold legChain
  rw [cn_nodeChain k hW hb hleaf hp hq, if_neg (by simp [hi])]
  congr 1
  cases hpart : D.partner p with
  | none =>
    have hpb := bnd_of_partner_none hW hp hpart
    unfold tail
    rw [cn_partner_bnd k hW hb hpb hq, hpart]
    simp only
    rw [cn_portBoxes_out k hW hb hbp hpb hi hq, List.map_map, nbox_mult]
  | some r =>
    have hrl := partner_leg hW hpart
    have hri : inS b r = false := by
      by_contra h
      exact absurd ((crossP_iff hW).2 ⟨hi, r, hpart, by simpa using h⟩) (by simp [hc])
    have hlen1 : (D.nimg b k p).length = 1 := by
      rw [nimg_out D b k hi, growImg_of_not _ _ _ hc]; rfl
    obtain ⟨hjr, hpq⟩ := cn_partner_leg k hW hb hpart 0 (by omega)
    have h0 : (D.nimg b k p)[0]'(by omega) = ⟨p.node, p.out, shiftPos (D.crossP b) k p⟩ := by
      simp [nimg_out D b k hi, growImg_of_not _ _ _ hc]
    rw [h0] at hpq
    rw [tail_drop _ _ _ hpq, tail_drop D p r hpart, cn_nodeChain k hW hb hleaf hp hq,
      cn_nodeChain k hW hb hleaf hrl (List.getElem_mem hjr), if_neg (by simp [hi]),
      if_neg (by simp [hri])]

end legchain2

theorem portDim_of (D : Diagram) (hids : (D.nodes.map (·.id)).Nodup) {n : DNode} (hn : n ∈ D.nodes)
    (o : Bool) (i : ℕ) : D.portDim ⟨n.id, o, i⟩ = if o then n.outs.getD i 0 else n.ins.getD i 0 := by
  unfold portDim
  rw [find_node hids hn]

section dims

variable {D : Diagram} {b : DBox} (k : ℕ) (hW : WF D) (hb : b ∈ D.boxes)

include hW hb in
theorem cn_portDim {p : Port} (hp : p ∈ D.legs) {q : Port} (hq : q ∈ D.nimg b k p) :
    (D.copyNodes b k).portDim q = D.portDim p := by
  obtain ⟨n, hn, hpn⟩ := leg_node hp
  obtain ⟨hpid, hpos⟩ := DNode.mem_legs.1 hpn
  have hids' := cn_ids k hW hb
  obtain ⟨x, o, i⟩ := p
  simp only at hpid hpos
  subst hpid
  rw [portDim_of D hW.ids hn]
  cases hi : inS b ⟨n.id, o, i⟩
  · rw [nimg_out D b k hi] at hq
    obtain ⟨j, hj, rfl⟩ := mem_growImg' hq
    have hnb : n.id ∉ b.nodes := by simpa [inS] using hi
    have hm : D.ngrow b k n ∈ (D.copyNodes b k).nodes := by
      rw [cn_nodes]; exact List.mem_flatMap.2 ⟨n, hn, by simp [cnode, hnb]⟩
    have := portDim_of _ hids' hm o (shiftPos (D.crossP b) k ⟨n.id, o, i⟩ + j)
    simp only [ngrow] at this
    rw [this]
    cases o
    · simp only [Bool.false_eq_true, ↓reduceIte] at hpos ⊢
      exact growSide_getD _ _ _ _ _ _ _ hpos hj
    · simp only [↓reduceIte] at hpos ⊢
      exact growSide_getD _ _ _ _ _ _ _ hpos hj
  · obtain ⟨j, hj, rfl⟩ := nimg_in_mem D b k hi hq
    have hnb : n.id ∈ b.nodes := (inS_iff b _).1 hi
    have hm : ({ n with id := D.cpId b j n.id } : DNode) ∈ (D.copyNodes b k).nodes := by
      rw [cn_nodes]
      refine List.mem_flatMap.2 ⟨n, hn, ?_⟩
      simp only [cnode, hnb, ↓reduceIte, List.mem_map, List.mem_range]
      exact ⟨j, hj, rfl⟩
    exact portDim_of _ hids' hm o i

end dims

/-- The node box `b`'s instantiation satisfies the hypotheses of `evalD_split`. -/
theorem cn_split {D : Diagram} {b : DBox} (ks : ℕ → ℕ) (hW : WF D) (hb : b ∈ D.boxes)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (hbp : b.ports = []) :
    SplitOK D (D.copyNodes b (ks b.mult)) (D.nimg b (ks b.mult)) (D.gN b) ks b.mult := by
  set k := ks b.mult
  have hnd := cn_ends_nodup k hW hb
  refine ⟨legs_nodup_of D hW.ids, legs_nodup_of _ (cn_ids k hW hb),
    (List.nodup_append.1 (List.nodup_append.1 hnd).2.1).2.1, cn_memL D b k,
    fun p hp p' hp' q h1 h2 => cn_disj k hW hp hp' h1 h2, fun p hp hg => ?_,
    fun p hp hg => cn_legChain_keep k hW hb hleaf hbp hp hg,
    fun p hp q hq => cn_portDim k hW hb hp hq, fun w hw => ?_, fun w hw => gN_wire hW hw,
    fun w' => by rw [cn_wires, List.mem_flatMap], fun p hp => bnd_legs hW hp, fun q => ?_⟩
  · obtain ⟨c, hc, hq⟩ := cn_legChain_split k hW hb hleaf hbp hp hg
    exact ⟨c, hc, by rw [nimg_length, if_pos hg], nimg_nodup D b k p, hq⟩
  · obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
    exact ⟨l1, l2, legChain_wire hW hw⟩
  · rw [(cn_bnd_perm D b k).mem_iff, List.mem_flatMap]

/-- A grown side's digit labels are the old side's through `φ`. -/
theorem side_relabel (g : Port → Bool) (k y : ℕ) (o : Bool) (len m : ℕ) (ks : ℕ → ℕ)
    (tl tl' : Port → List ℕ) (φ : Port → List ℕ → Port × List ℕ) (t : List ℕ) (hk : ks m = k)
    (hsplit : ∀ i < len, g ⟨y, o, i⟩ = true → ∃ c, tl ⟨y, o, i⟩ = c ++ [m] ∧ ∀ j < k,
      tl' ⟨y, o, shiftPos g k ⟨y, o, i⟩ + j⟩ = c ∧
      ∀ s, φ ⟨y, o, i⟩ (t ++ (s ++ [j])) = (⟨y, o, shiftPos g k ⟨y, o, i⟩ + j⟩, t ++ s))
    (hkeep : ∀ i < len, g ⟨y, o, i⟩ = false → tl' ⟨y, o, shiftPos g k ⟨y, o, i⟩⟩ = tl ⟨y, o, i⟩ ∧
      ∀ s, φ ⟨y, o, i⟩ (t ++ s) = (⟨y, o, shiftPos g k ⟨y, o, i⟩⟩, t ++ s)) :
    (List.range (wsum (fun i' => if g ⟨y, o, i'⟩ then k else 1) len)).flatMap (fun i' =>
      (rcidx ks (tl' ⟨y, o, i'⟩)).map fun s => dlab ⟨y, o, i'⟩ (t ++ s)) =
    ((List.range len).flatMap fun i =>
      (rcidx ks (tl ⟨y, o, i⟩)).map fun s => dlab ⟨y, o, i⟩ (t ++ s)).map (relab φ) := by
  rw [← range_blocks, List.flatMap_assoc, List.map_flatMap]
  refine List.flatMap_congr fun i hi => ?_
  have hi' := List.mem_range.1 hi
  rw [List.flatMap_map, List.map_map]
  by_cases hg : g ⟨y, o, i⟩ = true
  · simp only [hg, ↓reduceIte]
    obtain ⟨c, hc, hj⟩ := hsplit i hi' hg
    rw [hc, rcidx_append_single, hk, List.map_flatMap]
    refine List.flatMap_congr fun j hjr => ?_
    have hj' := List.mem_range.1 hjr
    obtain ⟨htl, hφ⟩ := hj j hj'
    rw [← shiftPos_eq, htl, List.map_map]
    refine List.map_congr_left fun s _ => ?_
    simp only [Function.comp, relab_dlab, hφ]
  · have hg' : g ⟨y, o, i⟩ = false := by simpa using hg
    simp only [hg', Bool.false_eq_true, ↓reduceIte, List.range_one, List.flatMap_singleton,
      add_zero]
    obtain ⟨htl, hφ⟩ := hkeep i hi' hg'
    rw [← shiftPos_eq, htl]
    refine List.map_congr_left fun s _ => ?_
    simp only [Function.comp, relab_dlab, hφ]

theorem cidx_append_single (ks : ℕ → ℕ) (c : List ℕ) (m : ℕ) :
    cidx ks (c ++ [m]) = (cidx ks c).flatMap fun u => (List.range (ks m)).map (u ++ [·]) := by
  induction c with
  | nil => simp [cidx, List.flatMap_singleton', List.map_eq_flatMap]
  | cons a c ih =>
    simp only [List.cons_append, cidx, ih, List.flatMap_assoc, List.map_flatMap, List.flatMap_map,
      List.map_map]
    rfl

theorem prod_flatMap' {α : Type*} (l : List α) (f : α → List ℂ) :
    (l.flatMap f).prod = (l.map fun a => (f a).prod).prod := by
  induction l with
  | nil => simp
  | cons a l ih => simp [List.flatMap_cons, List.prod_append, ih]

theorem rcidx_nil (ks : ℕ → ℕ) : rcidx ks [] = [[]] := by simp [rcidx, cidx]

section prod

variable {D : Diagram} {b : DBox} (ks : ℕ → ℕ) (hW : WF D) (hb : b ∈ D.boxes)
  (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (hbp : b.ports = [])

include hW hb in
theorem cn_nodeChain_out {n : DNode} (hn : n ∈ D.nodes) (hnb : n.id ∉ b.nodes) (k : ℕ) :
    (D.copyNodes b k).nodeChain n.id = D.nodeChain n.id := by
  have h' := cn_nchain_out k hW hb (List.mem_map_of_mem (f := (·.id)) hn) hnb
  unfold nodeChain nodeBoxes
  rw [h'.walk _ h'.length_le, List.map_map, nbox_mult]
  rfl

include hW hb hleaf in
theorem cn_tail_split {p : Port} (hp : p ∈ D.legs) (hpi : inS b p = false) (hg : D.gN b p = true) :
    ∃ c, D.tail p = c ++ [b.mult] ∧ ∀ q ∈ D.nimg b (ks b.mult) p,
      (D.copyNodes b (ks b.mult)).tail q = c := by
  set k := ks b.mult
  unfold gN at hg
  rw [hpi, Bool.false_or] at hg
  obtain ⟨-, r, hr, hri⟩ := (crossP_iff hW).1 hg
  have hrl := partner_leg hW hr
  have hx : r.node ∈ b.nodes := (inS_iff b r).1 hri
  obtain ⟨A, hA, hcA⟩ := nodeChain_last hW hb hleaf hx
  obtain ⟨w, hw, hwr⟩ := partner_some hr
  have hy : p.node ∉ b.nodes := by simpa [inS] using hpi
  have hpre := outside_prefix hW hb hleaf hw (y := p.node) (x := r.node)
    (by rcases hwr with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp) hy hx hA
  have hlen : (D.nodeChain p.node).length ≤ (A.map (·.mult)).length := by
    unfold nodeChain; simpa using hpre.length_le
  refine ⟨(A.map (·.mult)).drop (D.nodeChain p.node).length, ?_, fun q hq => ?_⟩
  · rw [tail_drop D p r hr, hcA, List.drop_append_of_le_length hlen]
  · obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hq
    obtain ⟨hjr, hpq⟩ := cn_partner_leg k hW hb hr j hj
    rw [tail_drop _ _ _ hpq, cn_nodeChain k hW hb hleaf hp (List.getElem_mem hj),
      cn_nodeChain k hW hb hleaf hrl (List.getElem_mem hjr), if_neg (by simp [hpi]), if_pos hri,
      hcA, List.dropLast_concat]

include hW hb hleaf hbp in
theorem cn_tail_keep {p : Port} (hp : p ∈ D.legs) (hpi : inS b p = false) (hg : D.gN b p = false) :
    ∃ q, D.nimg b (ks b.mult) p = [q] ∧ (D.copyNodes b (ks b.mult)).tail q = D.tail p := by
  obtain ⟨q, hq, hc⟩ := cn_legChain_keep (ks b.mult) hW hb hleaf hbp hp hg
  refine ⟨q, hq, ?_⟩
  unfold legChain at hc
  rw [cn_nodeChain _ hW hb hleaf hp (by rw [hq]; simp), if_neg (by simp [hpi])] at hc
  exact List.append_cancel_left hc

include hW hb hleaf hbp in
theorem cn_tail_copy {p : Port} (hp : p ∈ D.legs) (hpi : inS b p = true) {q : Port}
    (hq : q ∈ D.nimg b (ks b.mult) p) : (D.copyNodes b (ks b.mult)).tail q = [] := by
  obtain ⟨c, hc, hq'⟩ := cn_legChain_split (ks b.mult) hW hb hleaf hbp hp (by simp [gN, hpi])
  have e := hq' q hq
  have hx : p.node ∈ b.nodes := (inS_iff b p).1 hpi
  obtain ⟨A, hA, hcA⟩ := nodeChain_last hW hb hleaf hx
  have hch := cn_nodeChain _ hW hb hleaf hp hq
  rw [if_pos hpi, hcA, List.dropLast_concat] at hch
  rw [legChain, tail_in hW hb hleaf hp hx, hcA, List.append_nil] at hc
  have hc' := List.append_inj_left' hc rfl
  unfold legChain at e
  rw [hch, hc'] at e
  simpa using e

include hW hb hleaf in
theorem cn_nodeChain_copy {x j : ℕ} (hx : x ∈ b.nodes) (hj : j < ks b.mult) {A : List DBox}
    (hA : D.nodeBoxes x = A ++ [b]) :
    (D.copyNodes b (ks b.mult)).nodeChain (D.cpId b j x) = A.map (·.mult) := by
  have h' := cn_nchain_copy (ks b.mult) hW hb hleaf hx hj hA
  unfold nodeChain nodeBoxes
  rw [h'.walk _ h'.length_le, List.map_map, nbox_mult]

theorem side_mem {D : Diagram} {n : DNode} (hn : n ∈ D.nodes) (o : Bool) (i : ℕ)
    (hi : i < (if o then n.outs else n.ins).length) : (⟨n.id, o, i⟩ : Port) ∈ D.legs :=
  leg_mem hn o i (by cases o <;> simpa [side] using hi)

include hW hb hleaf hbp in
theorem cn_dnode_out (n : DNode) (hn : n ∈ D.nodes) (hnb : n.id ∉ b.nodes) (t : List ℕ) :
    (D.copyNodes b (ks b.mult)).dnode ks (D.ngrow b (ks b.mult) n) t =
      (D.dnode ks n t).rename (relab (φS (D.nimg b (ks b.mult)) (D.gN b))) := by
  set k := ks b.mult
  have key : ∀ o : Bool, (List.range (growSide (D.crossP b) k n.id o
      (if o then n.outs else n.ins)).length).flatMap (fun i' =>
        (rcidx ks ((D.copyNodes b k).tail ⟨n.id, o, i'⟩)).map fun s => dlab ⟨n.id, o, i'⟩ (t ++ s)) =
      ((List.range (if o then n.outs else n.ins).length).flatMap fun i =>
        (rcidx ks (D.tail ⟨n.id, o, i⟩)).map fun s => dlab ⟨n.id, o, i⟩ (t ++ s)).map
        (relab (φS (D.nimg b k) (D.gN b))) := fun o => by
    rw [growSide_length]
    refine side_relabel (D.crossP b) k n.id o _ b.mult ks _ _ _ t rfl (fun i hi hc => ?_)
      (fun i hi hc => ?_)
    · have hp := side_mem hn o i hi
      have hpi : inS b ⟨n.id, o, i⟩ = false := by simpa [inS] using hnb
      have hg : D.gN b ⟨n.id, o, i⟩ = true := by simp [gN, hc]
      obtain ⟨c, hcp, hq⟩ := cn_tail_split ks hW hb hleaf hp hpi hg
      refine ⟨c, hcp, fun j hj => ⟨hq _ ?_, fun s => ?_⟩⟩
      · rw [nimg_out D b k hpi, growImg_of_pos _ _ _ hc]
        exact List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩
      · simp only [φS, hg, ↓reduceIte, ← List.append_assoc, getLastD_snoc, dropLast_snoc,
          nimg_out D b k hpi, growImg_getD_pos _ _ _ hc j hj]
    · have hp := side_mem hn o i hi
      have hpi : inS b ⟨n.id, o, i⟩ = false := by simpa [inS] using hnb
      have hg : D.gN b ⟨n.id, o, i⟩ = false := by simp [gN, hpi, hc]
      obtain ⟨q, hq, htl⟩ := cn_tail_keep ks hW hb hleaf hbp hp hpi hg
      have hq' : q = ⟨n.id, o, shiftPos (D.crossP b) k ⟨n.id, o, i⟩⟩ := by
        rw [nimg_out D b k hpi, growImg_of_not _ _ _ hc] at hq
        exact (List.cons.inj hq).1.symm
      subst hq'
      refine ⟨htl, fun s => ?_⟩
      simp only [φS, hg, Bool.false_eq_true, ↓reduceIte]
      erw [hq]
      rfl
  have h1 := key true
  have h2 := key false
  simp only [↓reduceIte, Bool.false_eq_true] at h1 h2
  simp only [dnode, Node.rename, ngrow]
  rw [h1, h2]

include hW hb hleaf hbp in
theorem cn_dnode_copy (n : DNode) (hn : n ∈ D.nodes) (hnb : n.id ∈ b.nodes) (j : ℕ)
    (hj : j < ks b.mult) (u : List ℕ) :
    (D.copyNodes b (ks b.mult)).dnode ks { n with id := D.cpId b j n.id } u =
      (D.dnode ks n (u ++ [j])).rename (relab (φS (D.nimg b (ks b.mult)) (D.gN b))) := by
  set k := ks b.mult
  have key : ∀ o : Bool, ((List.range (if o then n.outs else n.ins).length).flatMap fun i =>
        (rcidx ks ((D.copyNodes b k).tail ⟨D.cpId b j n.id, o, i⟩)).map
          fun s => dlab ⟨D.cpId b j n.id, o, i⟩ (u ++ s)) =
      ((List.range (if o then n.outs else n.ins).length).flatMap fun i =>
        (rcidx ks (D.tail ⟨n.id, o, i⟩)).map fun s => dlab ⟨n.id, o, i⟩ (u ++ [j] ++ s)).map
        (relab (φS (D.nimg b k) (D.gN b))) := fun o => by
    rw [List.map_flatMap]
    refine List.flatMap_congr fun i hi => ?_
    have hp := side_mem hn o i (List.mem_range.1 hi)
    have hpi : inS b ⟨n.id, o, i⟩ = true := (inS_iff b _).2 hnb
    have hq : (⟨D.cpId b j n.id, o, i⟩ : Port) ∈ D.nimg b k ⟨n.id, o, i⟩ := by
      rw [nimg_in D b k hpi]; exact List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩
    rw [cn_tail_copy ks hW hb hleaf hbp hp hpi hq, tail_in hW hb hleaf hp hnb, rcidx_nil]
    have hlen : j < (D.nimg b k ⟨n.id, o, i⟩).length := by
      rw [nimg_in D b k hpi]; simpa using hj
    simp only [List.map_cons, List.map_nil, List.append_nil, relab_dlab, φS, gN, hpi,
      Bool.true_or, ↓reduceIte, getLastD_snoc, dropLast_snoc, List.getD_eq_getElem _ _ hlen]
    simp [nimg_in D b k hpi]
  have h1 := key true
  have h2 := key false
  simp only [↓reduceIte, Bool.false_eq_true] at h1 h2
  simp only [dnode, Node.rename]
  rw [h1, h2]

include hW hb hleaf hbp in
theorem cn_prodD (E : Env) (τ τ' : ℕ → ℕ)
    (hl : ∀ x ∈ D.dlabs ks, τ' x = τ (relab (φS (D.nimg b (ks b.mult)) (D.gN b)) x)) :
    (D.copyNodes b (ks b.mult)).prodD E ks τ = D.prodD E ks τ' := by
  set k := ks b.mult
  set ρ := relab (φS (D.nimg b k) (D.gN b))
  have hval : ∀ n ∈ D.nodes, ∀ t ∈ cidx ks (D.nodeChain n.id),
      val E τ ((D.dnode ks n t).rename ρ) = val E τ' (D.dnode ks n t) := fun n hn t ht => by
    rw [val_rename]
    exact val_reads E _ _ _ fun x hx => (hl x (dnode_labels_mem ks hn ht hx)).symm
  unfold prodD
  rw [cn_nodes, List.map_flatMap, prod_flatMap']
  congr 1
  refine List.map_congr_left fun n hn => ?_
  unfold cnode
  by_cases hnb : n.id ∈ b.nodes
  · rw [if_pos hnb]
    obtain ⟨A, hA, hcA⟩ := nodeChain_last hW hb hleaf hnb
    rw [hcA, cidx_append_single, List.map_flatMap, prod_flatMap', List.map_map]
    have hsw := (flatMap_swap_perm (cidx ks (A.map (·.mult))) (List.range k)
      (fun u j => val E τ' (D.dnode ks n (u ++ [j])))).prod_eq
    rw [prod_flatMap', prod_flatMap'] at hsw
    simp only [List.map_map, Function.comp_def] at hsw ⊢
    rw [hsw]
    congr 1
    refine List.map_congr_left fun j hj => ?_
    have hj' := List.mem_range.1 hj
    rw [cn_nodeChain_copy ks hW hb hleaf hnb hj' hA]
    congr 1
    refine List.map_congr_left fun u hu => ?_
    rw [cn_dnode_copy ks hW hb hleaf hbp n hn hnb j hj' u]
    refine hval n hn _ ?_
    rw [hcA, mem_cidx_single]
    exact ⟨u, j, rfl, hu, hj'⟩
  · rw [if_neg hnb]
    simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one]
    rw [show (D.ngrow b k n).id = n.id from rfl, cn_nodeChain_out hW hb hn hnb]
    congr 1
    refine List.map_congr_left fun t ht => ?_
    rw [cn_dnode_out ks hW hb hleaf hbp n hn hnb t]
    exact hval n hn t ht

end prod

/-- Copying a leaf node box keeps the digit sum. -/
theorem evalD_copyNodes (D : Diagram) (b : DBox) (ks : ℕ → ℕ) (hW : WF D) (hb : b ∈ D.boxes)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (hbp : b.ports = []) (E : Env)
    (σ' : ℕ → List ℕ → ℕ) :
    (D.copyNodes b (ks b.mult)).evalD E ks σ' =
      D.evalD E ks (bndMap D (D.copyNodes b (ks b.mult)) (φS (D.nimg b (ks b.mult)) (D.gN b)) σ') :=
  evalD_split (cn_split ks hW hb hleaf hbp) E σ' fun τ τ' hl => cn_prodD ks hW hb hleaf hbp E τ τ' hl

/-- The slot reading of a node box's instantiation at `k`. -/
def nodeMap (D : Diagram) (b : DBox) (k : ℕ) (σ' : ℕ → List ℕ → ℕ) : ℕ → List ℕ → ℕ :=
  let D₀ := if k = 0 then D.gather b else D
  bndMap D D₀ (fun p t => (p, t)) (bndMap D₀ (D₀.copyNodes b k) (φS (D₀.nimg b k) (D₀.gN b)) σ')

/-- Instantiating a leaf node box keeps the digit sum, the slots read through `nodeMap`. -/
theorem evalD_expandNode (D : Diagram) (b : DBox) (ks : ℕ → ℕ) (hW : WF D) (hb : b ∈ D.boxes)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (hbp : b.ports = []) (E : Env)
    (σ' : ℕ → List ℕ → ℕ) :
    (D.expandNode b (ks b.mult)).evalD E ks σ' = D.evalD E ks (nodeMap D b (ks b.mult) σ') := by
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
  have hexp : D.expandNode b (ks b.mult) = D₀.copyNodes b (ks b.mult) := rfl
  rw [hexp, evalD_copyNodes D₀ b ks hW₀ (hb₀ ▸ hb) (hb₀ ▸ hleaf) hbp E σ']
  unfold nodeMap
  simp only [← hD₀]
  rw [hD₀']
  exact evalD_perm_bnd D _ _ ho' hi' hW E ks _

theorem cn_mem_ids (D : Diagram) (b : DBox) (k : ℕ) (x : ℕ) :
    x ∈ (D.copyNodes b k).nodes.map (·.id) ↔
      (x ∈ D.nodes.map (·.id) ∧ x ∉ b.nodes) ∨
      (∃ j < k, ∃ z ∈ b.nodes, z ∈ D.nodes.map (·.id) ∧ x = D.cpId b j z) := by
  simp only [cn_nodes, List.map_flatMap, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨n, hn, n', hn', rfl⟩
    unfold cnode at hn'
    split_ifs at hn' with h
    · simp only [List.mem_map, List.mem_range] at hn'
      obtain ⟨j, hj, rfl⟩ := hn'
      exact Or.inr ⟨j, hj, n.id, h, ⟨n, hn, rfl⟩, rfl⟩
    · simp only [List.mem_singleton] at hn'
      subst hn'
      exact Or.inl ⟨⟨n, hn, rfl⟩, h⟩
  · rintro (⟨⟨n, hn, rfl⟩, h⟩ | ⟨j, hj, z, hz, ⟨n, hn, rfl⟩, rfl⟩)
    · exact ⟨n, hn, D.ngrow b k n, by simp [cnode, h], rfl⟩
    · refine ⟨n, hn, { n with id := D.cpId b j n.id }, ?_, rfl⟩
      simp only [cnode, hz, ↓reduceIte, List.mem_map, List.mem_range]
      exact ⟨j, hj, rfl⟩

/-- Copying a leaf node box keeps the box structure. -/
theorem cn_WF {D : Diagram} {b : DBox} (k : ℕ) (hW : WF D) (hb : b ∈ D.boxes)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (hbp : b.ports = []) :
    WF (D.copyNodes b k) := by
  have hmem := cn_mem_boxes D b k
  refine ⟨cn_ids k hW hb, ?_, cn_perm k hW hb, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [cn_boxes]
    refine (List.Sublist.map _ List.filter_sublist).nodup ?_
    rw [List.map_map]
    exact (List.Sublist.map _ List.filter_sublist).nodup hW.bids
  · intro y' hy'
    obtain ⟨y, hy, -, rfl, -⟩ := (hmem y').1 hy'
    rcases hW.kinds y hy with h | h
    · exact Or.inl (by simp [nbox, h])
    · exact Or.inr (by simp [nbox, h])
  · intro y' hy' x hx
    obtain ⟨y, hy, -, rfl, -⟩ := (hmem y').1 hy'
    rw [cn_mem_ids]
    simp only [nbox, List.mem_append, List.mem_filter, decide_eq_true_eq] at hx
    rcases hx with ⟨h1, h2⟩ | h
    · exact Or.inl ⟨hW.nsub y hy x h1, h2⟩
    · split_ifs at h
      · obtain ⟨j, hj, z, hz, rfl⟩ := (mem_copies D b k _).1 h
        exact Or.inr ⟨j, hj, z, hz, hW.nsub b hb z hz, rfl⟩
      · simp at h
  · intro x hx
    rcases (cn_mem_ids D b k x).1 hx with ⟨hx, hxb⟩ | ⟨j, hj, z, hz, -, rfl⟩
    · have h' := cn_nchain_out k hW hb hx hxb
      unfold nodeBoxes; rw [h'.walk _ h'.length_le]; exact h'
    · obtain ⟨A, hA⟩ := nodeBoxes_last hW hb hleaf hz
      have h' := cn_nchain_copy k hW hb hleaf hz hj hA
      unfold nodeBoxes; rw [h'.walk _ h'.length_le]; exact h'
  · intro x y' hy' hx
    obtain ⟨y, hy, hne, rfl, -⟩ := (hmem y').1 hy'
    have hx' := hx
    simp only [nbox, List.mem_append, List.mem_filter, decide_eq_true_eq] at hx
    rcases hx with ⟨h1, h2⟩ | h
    · have hxi := hW.nsub y hy x h1
      have h' := cn_nchain_out k hW hb hxi h2
      unfold nodeBoxes; rw [h'.walk _ h'.length_le]
      exact List.mem_map.2 ⟨y, hW.nall x y hy h1, rfl⟩
    · split_ifs at h with hany
      · obtain ⟨j, hj, z, hz, rfl⟩ := (mem_copies D b k _).1 h
        obtain ⟨A, hA⟩ := nodeBoxes_last hW hb hleaf hz
        have h' := cn_nchain_copy k hW hb hleaf hz hj hA
        unfold nodeBoxes; rw [h'.walk _ h'.length_le]
        obtain ⟨z', hz', hz'b⟩ := List.any_eq_true.1 hany
        have hzy := box_scope hW hb hleaf hy hz (of_decide_eq_true hz'b) hz'
        have hyL := hW.nall z y hy hzy
        rw [hA] at hyL
        rcases List.mem_append.1 hyL with h | h
        · exact List.mem_map.2 ⟨y, h, rfl⟩
        · rw [List.mem_singleton] at h; subst h; simp at hne
      · simp at h
  · intro q hq
    rw [(cn_bnd_perm D b k).mem_iff, List.mem_flatMap] at hq
    obtain ⟨p, hp, hqp⟩ := hq
    unfold portBoxes
    cases hpi : inS b p
    · have h' := cn_pchain_out k hW hb hbp hp hpi hqp
      rw [h'.walk _ h'.length_le]; exact h'
    · have hpb := portBoxes_in hW hb hleaf ((inS_iff b p).1 hpi)
      have h' := cn_pchain_in k hW hb hp hqp hpb
      rw [h'.walk _ h'.length_le]; exact h'
  · intro q hq y' hy' hqy
    rw [(cn_bnd_perm D b k).mem_iff, List.mem_flatMap] at hq
    obtain ⟨p, hp, hqp⟩ := hq
    obtain ⟨y, hy, -, rfl, -⟩ := (hmem y').1 hy'
    simp only [nbox, List.mem_flatMap] at hqy
    obtain ⟨p'', hp'', hq''⟩ := hqy
    have := cn_disj k hW (bnd_legs hW hp) (bnd_legs hW (hW.pbnd y hy p'' hp'')) hqp hq''
    subst this
    have hyp := hW.pall p hp y hy hp''
    cases hpi : inS b p
    · rw [cn_portBoxes_out k hW hb hbp hp hpi hqp]
      exact List.mem_map.2 ⟨y, hyp, rfl⟩
    · rw [portBoxes_in hW hb hleaf ((inS_iff b p).1 hpi)] at hyp; simp at hyp
  · intro y' hy' q hq
    obtain ⟨y, hy, -, rfl, -⟩ := (hmem y').1 hy'
    simp only [nbox, List.mem_flatMap] at hq
    obtain ⟨p, hp, hqp⟩ := hq
    rw [(cn_bnd_perm D b k).mem_iff]
    exact List.mem_flatMap.2 ⟨p, hW.pbnd y hy p hp, hqp⟩
  · intro w' hw'
    rw [cn_wires, List.mem_flatMap] at hw'
    obtain ⟨w, hw, hz⟩ := hw'
    obtain ⟨j, j1, j2, rfl⟩ := mem_zip_idx hz
    obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
    simp only
    rw [cn_chain k hW hb hleaf l1 (List.getElem_mem j1), cn_chain k hW hb hleaf l2 (List.getElem_mem j2)]
    have hmap : ∀ {L L' : List DBox}, L <+: L' ∨ L' <+: L →
        L.map (D.nbox b k) <+: L'.map (D.nbox b k) ∨ L'.map (D.nbox b k) <+: L.map (D.nbox b k) :=
      fun h => h.elim (fun h => Or.inl (h.map _)) (fun h => Or.inr (h.map _))
    refine hmap ?_
    cases h1 : inS b w.1 <;> cases h2 : inS b w.2 <;> simp only [Bool.false_eq_true, ↓reduceIte]
    · exact hW.wpre w hw
    · have hx := (inS_iff b _).1 h2
      obtain ⟨A, hA⟩ := nodeBoxes_last hW hb hleaf hx
      have := outside_prefix hW hb hleaf hw (y := w.1.node) (x := w.2.node) (Or.inl ⟨rfl, rfl⟩)
        (by simpa [inS] using h1) hx hA
      rw [hA, List.dropLast_concat]; exact Or.inl this
    · have hx := (inS_iff b _).1 h1
      obtain ⟨A, hA⟩ := nodeBoxes_last hW hb hleaf hx
      have := outside_prefix hW hb hleaf hw (y := w.2.node) (x := w.1.node) (Or.inr ⟨rfl, rfl⟩)
        (by simpa [inS] using h2) hx hA
      rw [hA, List.dropLast_concat]; exact Or.inr this
    · rw [nodeBoxes_same hW hb hleaf ((inS_iff b _).1 h1) ((inS_iff b _).1 h2)]
      exact Or.inl (List.prefix_refl _)
  · intro y' hy'
    obtain ⟨y, -, -, rfl, h⟩ := (hmem y').1 hy'
    exact h
  · intro w' hw'
    rw [cn_wires, List.mem_flatMap] at hw'
    obtain ⟨w, hw, hz⟩ := hw'
    obtain ⟨j, j1, j2, rfl⟩ := mem_zip_idx hz
    obtain ⟨-, -, l1, l2⟩ := wire_not_bnd hW hw
    rw [cn_portDim k hW hb l1 (List.getElem_mem j1), cn_portDim k hW hb l2 (List.getElem_mem j2)]
    exact hW.wdim w hw

end Diagram

end ArchytasZX
