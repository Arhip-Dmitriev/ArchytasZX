/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PairBox

/-!
# The expansion of a pair box as a network

`PairSpec` reads `PairOK` as propositions. `netBy_expandPair`: in the labelling `plab`, the
expansion's network is `pairNet` (copy `j` of each template node at the template's labels moved
by `shJ`, its pair legs at `gL` / `gR`, then the pair nodes). `equal_expandPair`: for every
`K`, every offset `o ≥ K m` and every environment reading every label at one dimension,
`(D.expandPair p K).toNet o` equals `D.pairNet p K` (`equal_netBy`).
-/

namespace ArchytasZX

namespace Diagram

variable {D : Diagram} {p : DPair} {b : DBox}

/-- `PairOK` as propositions. -/
structure PairSpec (D : Diagram) (p : DPair) : Prop where
  nodup : (D.nodes.map (·.id)).Nodup
  ins : D.ins = []
  lr : p.left ≠ p.right
  lpos : p.left.pos < D.sideLen p.left.node p.left.out
  rpos : p.right.pos < D.sideLen p.right.node p.right.out
  lastL : D.lastP p p.left = true
  lastR : D.lastP p p.right = true
  lbnd : p.left ∉ D.bnd
  rbnd : p.right ∉ D.bnd
  wires : ∀ w ∈ D.wires, w.1 ≠ p.left ∧ w.1 ≠ p.right ∧ w.2 ≠ p.left ∧ w.2 ≠ p.right ∧
    w.1.pos < D.sideLen w.1.node w.1.out ∧ w.2.pos < D.sideLen w.2.node w.2.out
  outs : ∀ q ∈ D.outs, q.pos < D.sideLen q.node q.out
  legsO : ∀ n ∈ D.nodes, ∀ k < n.outs.length, keepP p ⟨n.id, true, k⟩ = true →
    D.okPort ⟨n.id, true, k⟩ = true
  legsI : ∀ n ∈ D.nodes, ∀ k < n.ins.length, keepP p ⟨n.id, false, k⟩ = true →
    D.okPort ⟨n.id, false, k⟩ = true
  slots : ∀ i < D.outs.length, D.label D.outs.length (D.outs.getD i Port.dflt) = i
  wlab : ∀ i < D.wires.length,
    D.label D.outs.length (D.wires.getD i (Port.dflt, Port.dflt)).1 = D.outs.length + i ∧
    D.label D.outs.length (D.wires.getD i (Port.dflt, Port.dflt)).2 = D.outs.length + i
  fk : p.node.kind = .F
  fo : p.node.outs = [0]
  fi : p.node.ins = [0]
  fd : p.node.dim = 0
  dims : ∀ n ∈ D.nodes, n.dim = 0

theorem PairOK.spec (h : D.PairOK p b = true) : PairSpec D p := by
  simp only [PairOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, Bool.not_eq_true',
    Bool.or_eq_true, List.mem_range, and_assoc] at h
  obtain ⟨-, -, -, -, -, hnd, hins, hlr, hlp, hrp, hll, hlr', hlb, hrb, hw, ho, hlg, hsl, hwl, hfk,
    hfo, hfi, hfd, hdm⟩ := h
  refine ⟨hnd, hins, hlr, hlp, hrp, hll, hlr', ?_, ?_, hw, ho, fun n hn k hk hkp => ?_,
    fun n hn k hk hkp => ?_, hsl, hwl, hfk, hfo, hfi, hfd, hdm⟩
  · simpa using hlb
  · simpa using hrb
  · exact ((hlg n hn).1 k hk).resolve_left (by simp [hkp])
  · exact ((hlg n hn).2 k hk).resolve_left (by simp [hkp])

/-! ## Decoding the expansion's node ids -/

theorem pcp_half (j x : ℕ) : (pcp j x / 2).unpair = (j, x) := by
  simp [pcp, Nat.unpair_pair]

theorem pfid_half (j k : ℕ) : (pfid j k / 2).unpair = (j, k) := by
  have : (2 * Nat.pair j k + 1) / 2 = Nat.pair j k := by omega
  simp [pfid, this, Nat.unpair_pair]

theorem plab_cp (K j x : ℕ) (out : Bool) (pos : ℕ) :
    D.plab p K ⟨pcp j x, out, pos⟩ = D.clab p K j x out pos := by
  have h : pcp j x % 2 = 0 := by simp [pcp]
  simp [plab, h, pcp_half]

theorem plab_fo (K j k pos : ℕ) :
    D.plab p K ⟨pfid j k, true, pos⟩ = gR K D.outs.length D.wires.length j k := by
  have h : pfid j k % 2 = 1 := by simp [pfid]
  simp [plab, h, pfid_half]

theorem plab_fi (K j k pos : ℕ) :
    D.plab p K ⟨pfid j k, false, pos⟩ = gL K D.outs.length D.wires.length j k := by
  have h : pfid j k % 2 = 1 := by simp [pfid]
  simp [plab, h, pfid_half]

theorem port_eq_iff (x : ℕ) (out : Bool) (k : ℕ) (q : Port) :
    (⟨x, out, k⟩ : Port) = q ↔ x = q.node ∧ out = q.out ∧ k = q.pos := by
  cases q; simp

theorem port_ext {q r : Port} (h1 : q.node = r.node) (h2 : q.out = r.out) (h3 : q.pos = r.pos) :
    q = r := by
  cases q; cases r; simp_all

/-! ## Sides -/

theorem find_of_mem (l : List DNode) (hnd : (l.map (·.id)).Nodup) {n : DNode} (hn : n ∈ l) :
    l.find? (fun m => decide (m.id = n.id)) = some n := by
  induction l with
  | nil => simp at hn
  | cons a l ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
    rcases List.mem_cons.1 hn with rfl | hn'
    · simp
    · have hne : a.id ≠ n.id := fun e => hnd.1 ⟨n, hn', e.symm⟩
      simp [hne, ih hnd.2 hn']

theorem sideLen_of (s : PairSpec D p) {n : DNode} (hn : n ∈ D.nodes) (out : Bool) :
    D.sideLen n.id out = (if out then n.outs.length else n.ins.length) := by
  simp [sideLen, find_of_mem D.nodes s.nodup hn]

theorem right_last (s : PairSpec D p) : p.right.pos + 1 = D.sideLen p.right.node p.right.out := by
  have h := s.lastR
  simp only [lastP, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
  rcases h with h | ⟨⟨h1, -⟩, -⟩
  · exact h
  · exact absurd h1.symm s.lr

theorem left_last (s : PairSpec D p) : p.left.pos + 1 = D.sideLen p.left.node p.left.out ∨
    (p.right = ⟨p.left.node, p.left.out, p.left.pos + 1⟩ ∧
      p.left.pos + 2 = D.sideLen p.left.node p.left.out) := by
  have h := s.lastL
  simp only [lastP, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
  rcases h with h | ⟨⟨-, h2⟩, h3⟩
  · exact Or.inl h
  · exact Or.inr ⟨h2, h3⟩

/-- The positions of the pair ports on the side `out` of the node `x`. -/
theorem side_facts (s : PairSpec D p) (x : ℕ) (out : Bool) :
    ((x = p.left.node ∧ out = p.left.out) → (x = p.right.node ∧ out = p.right.out) →
      p.right.pos = p.left.pos + 1 ∧ p.left.pos + 2 = D.sideLen x out) ∧
    ((x = p.left.node ∧ out = p.left.out) → ¬(x = p.right.node ∧ out = p.right.out) →
      p.left.pos + 1 = D.sideLen x out) ∧
    ((x = p.right.node ∧ out = p.right.out) → p.right.pos + 1 = D.sideLen x out) := by
  have hl := left_last s
  have hr := right_last s
  refine ⟨fun ⟨a1, a2⟩ ⟨b1, b2⟩ => ?_, fun ⟨a1, a2⟩ hb => ?_, fun ⟨b1, b2⟩ => by
    subst b1 b2; exact hr⟩
  · subst a1 a2
    rcases hl with hl | ⟨he, hl⟩
    · exfalso
      rw [← b1, ← b2] at hr
      exact s.lr (port_ext b1 b2 (by omega))
    · have := (port_eq_iff _ _ _ _).1 he.symm
      exact ⟨this.2.2.symm, hl⟩
  · subst a1 a2
    rcases hl with hl | ⟨he, -⟩
    · exact hl
    · have := (port_eq_iff _ _ _ _).1 he.symm
      exact absurd ⟨this.1, this.2.1⟩ hb

/-! ## Positions after growing -/

theorem posC_ones (c : Port → ℕ) (q : Port) (h : ∀ i < q.pos, c ⟨q.node, q.out, i⟩ = 1) :
    posC c q = q.pos := by
  unfold posC
  rw [List.map_congr_left (g := fun _ => 1) (fun i hi => h i (List.mem_range.1 hi))]
  simp

theorem pcnt_one (K j : ℕ) (q : Port) (h1 : q ≠ p.left) (h2 : q ≠ p.right) : pcnt p K j q = 1 := by
  simp [pcnt, h1, h2]

/-- Below a kept leg of its side, every leg is kept. -/
theorem below_keep (s : PairSpec D p) (q : Port) (hq : q.pos < D.sideLen q.node q.out)
    (hk : keepP p q = true) (i : ℕ) (hi : i < q.pos) :
    (⟨q.node, q.out, i⟩ : Port) ≠ p.left ∧ (⟨q.node, q.out, i⟩ : Port) ≠ p.right := by
  simp only [keepP, Bool.and_eq_true, decide_eq_true_eq, ne_eq] at hk
  obtain ⟨f1, f2, f3⟩ := side_facts s q.node q.out
  refine ⟨fun e => ?_, fun e => ?_⟩
  · obtain ⟨a1, a2, a3⟩ := (port_eq_iff _ _ _ _).1 e
    by_cases hB : q.node = p.right.node ∧ q.out = p.right.out
    · obtain ⟨r1, r2⟩ := f1 ⟨a1, a2⟩ hB
      exact hk.2 (port_ext hB.1 hB.2 (by omega))
    · have := f2 ⟨a1, a2⟩ hB; omega
  · obtain ⟨a1, a2, a3⟩ := (port_eq_iff _ _ _ _).1 e
    have := f3 ⟨a1, a2⟩; omega

theorem posC_keep (s : PairSpec D p) (K j : ℕ) (q : Port) (hq : q.pos < D.sideLen q.node q.out)
    (hk : keepP p q = true) : posC (pcnt p K j) q = q.pos :=
  posC_ones _ _ fun i hi => pcnt_one K j _ (below_keep s q hq hk i hi).1 (below_keep s q hq hk i hi).2

theorem posC_left (s : PairSpec D p) (K j : ℕ) : posC (pcnt p K j) p.left = p.left.pos := by
  refine posC_ones _ _ fun i hi => pcnt_one K j _ (fun e => ?_) (fun e => ?_)
  · have := ((port_eq_iff _ _ _ _).1 e).2.2; omega
  · obtain ⟨a1, a2, a3⟩ := (port_eq_iff _ _ _ _).1 e
    have := ((side_facts s p.left.node p.left.out).1 ⟨rfl, rfl⟩ ⟨a1, a2⟩).1; omega

theorem posC_right (s : PairSpec D p) (K j : ℕ) :
    posC (pcnt p K j) p.right = if p.right.node = p.left.node ∧ p.right.out = p.left.out then
      p.left.pos + (K - 1 - j) else p.right.pos := by
  split_ifs with hS
  · obtain ⟨r1, r2⟩ := (side_facts s p.left.node p.left.out).1 ⟨rfl, rfl⟩ ⟨hS.1.symm, hS.2.symm⟩
    unfold posC
    rw [r1, List.range_succ, List.map_append, List.sum_append]
    have hL : (⟨p.right.node, p.right.out, p.left.pos⟩ : Port) = p.left :=
      port_ext hS.1 hS.2 rfl
    rw [List.map_congr_left (g := fun _ => 1) (fun i hi => pcnt_one K j _ (fun e => by
        have := ((port_eq_iff _ _ _ _).1 e).2.2; simp at hi; omega) (fun e => by
        have := ((port_eq_iff _ _ _ _).1 e).2.2; simp at hi; omega))]
    simp [pcnt, hL]
  · refine posC_ones _ _ fun i hi => pcnt_one K j _ (fun e => ?_) (fun e => ?_)
    · obtain ⟨a1, a2, -⟩ := (port_eq_iff _ _ _ _).1 e; exact hS ⟨a1, a2⟩
    · have := ((port_eq_iff _ _ _ _).1 e).2.2; omega

theorem length_growC (c : Port → ℕ) (x : ℕ) (out : Bool) (ds : List ℕ) :
    (growC c x out ds).length = ((List.range ds.length).map fun i => c ⟨x, out, i⟩).sum := by
  simp [growC, List.length_flatMap]

/-! ## Copy `j`'s legs in the labelling -/

section clab

variable (K j x : ℕ) (out : Bool)

theorem clab_low (k : ℕ) (h1 : ¬(x = p.left.node ∧ out = p.left.out ∧ p.left.pos ≤ k))
    (h2 : ¬(x = p.right.node ∧ out = p.right.out ∧ p.right.pos ≤ k)) :
    D.clab p K j x out k =
      shJ K D.outs.length D.wires.length j (D.label D.outs.length ⟨x, out, k⟩) := by
  simp only [clab, h1, h2, ite_false]

theorem clab_L (hA : x = p.left.node ∧ out = p.left.out) (t : ℕ) (ht : t < K - 1 - j) :
    D.clab p K j x out (p.left.pos + t) = gL K D.outs.length D.wires.length j (j + 1 + t) := by
  have h : x = p.left.node ∧ out = p.left.out ∧ p.left.pos ≤ p.left.pos + t := ⟨hA.1, hA.2, by omega⟩
  simp only [clab, h, and_self, ite_true, Nat.add_sub_cancel_left, ht]

theorem clab_RA (hA : x = p.left.node ∧ out = p.left.out) (i : ℕ) :
    D.clab p K j x out (p.left.pos + (K - 1 - j) + i) = gR K D.outs.length D.wires.length i j := by
  have h : x = p.left.node ∧ out = p.left.out ∧ p.left.pos ≤ p.left.pos + (K - 1 - j) + i :=
    ⟨hA.1, hA.2, by omega⟩
  have h2 : ¬(p.left.pos + (K - 1 - j) + i - p.left.pos < K - 1 - j) := by omega
  have h3 : p.left.pos + (K - 1 - j) + i - p.left.pos - (K - 1 - j) = i := by omega
  simp only [clab, h, and_self, ite_true, h2, ite_false, h3]

theorem clab_RB (hA : ¬(x = p.left.node ∧ out = p.left.out))
    (hB : x = p.right.node ∧ out = p.right.out) (i : ℕ) :
    D.clab p K j x out (p.right.pos + i) = gR K D.outs.length D.wires.length i j := by
  have h1 : ¬(x = p.left.node ∧ out = p.left.out ∧ p.left.pos ≤ p.right.pos + i) :=
    fun h => hA ⟨h.1, h.2.1⟩
  have h : x = p.right.node ∧ out = p.right.out ∧ p.right.pos ≤ p.right.pos + i :=
    ⟨hB.1, hB.2, by omega⟩
  simp only [clab]
  rw [if_neg h1, if_pos h, Nat.add_sub_cancel_left]

end clab

theorem later_eq (K j : ℕ) : later K j = (List.range (K - 1 - j)).map (j + 1 + ·) := by
  simp [later, List.range'_eq_map_range]

theorem map_low (s : PairSpec D p) (K j x : ℕ) (out : Bool) (bl : ℕ)
    (hbl : bl ≤ D.sideLen x out)
    (hl : (x = p.left.node ∧ out = p.left.out) → bl ≤ p.left.pos)
    (hr : (x = p.right.node ∧ out = p.right.out) → bl ≤ p.right.pos) :
    (List.range bl).map (fun k => D.plab p K ⟨pcp j x, out, k⟩) =
      ((List.range bl).map fun k => D.label D.outs.length ⟨x, out, k⟩).map
        (shJ K D.outs.length D.wires.length j) := by
  rw [List.map_map]
  refine List.map_congr_left fun k hk => ?_
  have hk' := List.mem_range.1 hk
  rw [plab_cp]
  exact clab_low K j x out k (fun h => by have := hl ⟨h.1, h.2.1⟩; omega)
    (fun h => by have := hr ⟨h.1, h.2.1⟩; omega)

theorem keep_low (s : PairSpec D p) (x : ℕ) (out : Bool) (bl k : ℕ)
    (hl : (x = p.left.node ∧ out = p.left.out) → bl ≤ p.left.pos)
    (hr : (x = p.right.node ∧ out = p.right.out) → bl ≤ p.right.pos) (hk : k < bl) :
    keepP p ⟨x, out, k⟩ = true := by
  simp only [keepP, Bool.and_eq_true, decide_eq_true_eq, ne_eq]
  refine ⟨fun e => ?_, fun e => ?_⟩
  · obtain ⟨a1, a2, a3⟩ := (port_eq_iff _ _ _ _).1 e; have := hl ⟨a1, a2⟩; omega
  · obtain ⟨a1, a2, a3⟩ := (port_eq_iff _ _ _ _).1 e; have := hr ⟨a1, a2⟩; omega

theorem sum_ones (bl : ℕ) (c : ℕ → ℕ) (h : ∀ i < bl, c i = 1) : ((List.range bl).map c).sum = bl := by
  rw [List.map_congr_left (g := fun _ => 1) fun i hi => h i (List.mem_range.1 hi)]
  simp

theorem filter_low (x : ℕ) (out : Bool) (bl e : ℕ)
    (hk : ∀ k < bl, keepP p ⟨x, out, k⟩ = true) (hn : ∀ t < e, keepP p ⟨x, out, bl + t⟩ = false) :
    (List.range (bl + e)).filter (fun k => keepP p ⟨x, out, k⟩) = List.range bl := by
  rw [List.range_add, List.filter_append, List.filter_eq_self.2 fun k hk' => hk k (List.mem_range.1 hk'),
    List.filter_eq_nil_iff.2 fun k hk' => ?_, List.append_nil]
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hk'
  simp [hn t (List.mem_range.1 ht)]

theorem keep_left (x : ℕ) (out : Bool) (hA : x = p.left.node ∧ out = p.left.out) :
    keepP p ⟨x, out, p.left.pos⟩ = false := by
  have : (⟨x, out, p.left.pos⟩ : Port) = p.left := port_ext hA.1 hA.2 rfl
  simp [keepP, this]

theorem keep_right (x : ℕ) (out : Bool) (hB : x = p.right.node ∧ out = p.right.out) :
    keepP p ⟨x, out, p.right.pos⟩ = false := by
  have : (⟨x, out, p.right.pos⟩ : Port) = p.right := port_ext hB.1 hB.2 rfl
  simp [keepP, this]

theorem pcnt_left (K j x : ℕ) (out : Bool) (hA : x = p.left.node ∧ out = p.left.out) :
    pcnt p K j ⟨x, out, p.left.pos⟩ = K - 1 - j := by
  have : (⟨x, out, p.left.pos⟩ : Port) = p.left := port_ext hA.1 hA.2 rfl
  simp [pcnt, this]

theorem pcnt_right (s : PairSpec D p) (K j x : ℕ) (out : Bool)
    (hB : x = p.right.node ∧ out = p.right.out) : pcnt p K j ⟨x, out, p.right.pos⟩ = j := by
  have : (⟨x, out, p.right.pos⟩ : Port) = p.right := port_ext hB.1 hB.2 rfl
  simp [pcnt, this, s.lr.symm]

/-- Copy `j`'s legs on a side in the labelling: the template's kept legs moved by `shJ`, then its
pair legs. -/
theorem side_labels (s : PairSpec D p) (K j x : ℕ) (out : Bool) (ds : List ℕ)
    (hds : ds.length = D.sideLen x out) :
    (List.range (growC (pcnt p K j) x out ds).length).map (fun k => D.plab p K ⟨pcp j x, out, k⟩) =
      (((List.range ds.length).filter fun k => keepP p ⟨x, out, k⟩).map
        fun k => D.label D.outs.length ⟨x, out, k⟩).map (shJ K D.outs.length D.wires.length j) ++
      D.exL p K j x out := by
  rw [length_growC, hds]
  obtain ⟨f1, f2, f3⟩ := side_facts s x out
  have cnt1 : ∀ bl, ((x = p.left.node ∧ out = p.left.out) → bl ≤ p.left.pos) →
      ((x = p.right.node ∧ out = p.right.out) → bl ≤ p.right.pos) →
      ∀ i < bl, pcnt p K j ⟨x, out, i⟩ = 1 := fun bl hl hr i hi => by
    have := keep_low s x out bl i hl hr hi
    simp only [keepP, Bool.and_eq_true, decide_eq_true_eq] at this
    exact pcnt_one K j _ this.1 this.2
  by_cases hA : x = p.left.node ∧ out = p.left.out <;>
    by_cases hB : x = p.right.node ∧ out = p.right.out
  · obtain ⟨r1, r2⟩ := f1 hA hB
    have hl : (x = p.left.node ∧ out = p.left.out) → p.left.pos ≤ p.left.pos := fun _ => le_rfl
    have hr : (x = p.right.node ∧ out = p.right.out) → p.left.pos ≤ p.right.pos := fun _ => by omega
    have hsum : ((List.range (D.sideLen x out)).map fun i => pcnt p K j ⟨x, out, i⟩).sum =
        p.left.pos + (K - 1 - j) + j := by
      rw [← r2, List.range_add, List.map_append, List.sum_append, sum_ones _ _ (cnt1 _ hl hr)]
      have e1 : (⟨x, out, p.left.pos + 1⟩ : Port) = ⟨x, out, p.right.pos⟩ := by rw [r1]
      simp [List.range_succ, pcnt_left K j x out hA, e1, pcnt_right s K j x out hB]
      omega
    have hfil : (List.range (D.sideLen x out)).filter (fun k => keepP p ⟨x, out, k⟩) =
        List.range p.left.pos := by
      rw [← r2]
      refine filter_low x out p.left.pos 2 (fun k hk => keep_low s x out _ k hl hr hk)
        fun t ht => ?_
      interval_cases t
      · exact keep_left x out hA
      · rw [show p.left.pos + 1 = p.right.pos by omega]; exact keep_right x out hB
    rw [hsum, hfil, List.range_add, List.range_add, List.map_append, List.map_append,
      map_low s K j x out _ (by omega) hl hr]
    simp only [exL]
    rw [if_pos hA, if_pos hB, List.append_assoc]
    congr 1
    congr 1
    · rw [later_eq, List.map_map, List.map_map]
      refine List.map_congr_left fun t ht => ?_
      simp only [Function.comp, plab_cp]
      exact clab_L K j x out hA t (List.mem_range.1 ht)
    · rw [List.map_map]
      refine List.map_congr_left fun i _ => ?_
      simp only [Function.comp, plab_cp]
      exact clab_RA K j x out hA i
  · have r2 := f2 hA hB
    have hl : (x = p.left.node ∧ out = p.left.out) → p.left.pos ≤ p.left.pos := fun _ => le_rfl
    have hr : (x = p.right.node ∧ out = p.right.out) → p.left.pos ≤ p.right.pos :=
      fun h => absurd h hB
    have hsum : ((List.range (D.sideLen x out)).map fun i => pcnt p K j ⟨x, out, i⟩).sum =
        p.left.pos + (K - 1 - j) := by
      rw [← r2, List.range_succ, List.map_append, List.sum_append, sum_ones _ _ (cnt1 _ hl hr)]
      simp [pcnt_left K j x out hA]
    have hfil : (List.range (D.sideLen x out)).filter (fun k => keepP p ⟨x, out, k⟩) =
        List.range p.left.pos := by
      rw [← r2]
      refine filter_low x out p.left.pos 1 (fun k hk => keep_low s x out _ k hl hr hk)
        fun t ht => ?_
      interval_cases t
      exact keep_left x out hA
    rw [hsum, hfil, List.range_add, List.map_append, map_low s K j x out _ (by omega) hl hr]
    simp only [exL]
    rw [if_pos hA, if_neg hB, List.append_nil]
    congr 1
    rw [later_eq, List.map_map, List.map_map]
    refine List.map_congr_left fun t ht => ?_
    simp only [Function.comp, plab_cp]
    exact clab_L K j x out hA t (List.mem_range.1 ht)
  · have r3 := f3 hB
    have hl : (x = p.left.node ∧ out = p.left.out) → p.right.pos ≤ p.left.pos :=
      fun h => absurd h hA
    have hr : (x = p.right.node ∧ out = p.right.out) → p.right.pos ≤ p.right.pos := fun _ => le_rfl
    have hsum : ((List.range (D.sideLen x out)).map fun i => pcnt p K j ⟨x, out, i⟩).sum =
        p.right.pos + j := by
      rw [← r3, List.range_succ, List.map_append, List.sum_append, sum_ones _ _ (cnt1 _ hl hr)]
      simp [pcnt_right s K j x out hB]
    have hfil : (List.range (D.sideLen x out)).filter (fun k => keepP p ⟨x, out, k⟩) =
        List.range p.right.pos := by
      rw [← r3]
      refine filter_low x out p.right.pos 1 (fun k hk => keep_low s x out _ k hl hr hk)
        fun t ht => ?_
      interval_cases t
      exact keep_right x out hB
    rw [hsum, hfil, List.range_add, List.map_append, map_low s K j x out _ (by omega) hl hr]
    simp only [exL]
    rw [if_neg hA, if_pos hB, List.nil_append]
    congr 1
    rw [List.map_map]
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp, plab_cp]
    exact clab_RB K j x out hA hB i
  · have hl : (x = p.left.node ∧ out = p.left.out) → D.sideLen x out ≤ p.left.pos :=
      fun h => absurd h hA
    have hr : (x = p.right.node ∧ out = p.right.out) → D.sideLen x out ≤ p.right.pos :=
      fun h => absurd h hB
    rw [sum_ones _ _ (cnt1 _ hl hr), map_low s K j x out _ le_rfl hl hr,
      List.filter_eq_self.2 fun k hk => keep_low s x out _ k hl hr (List.mem_range.1 hk)]
    simp only [exL]
    rw [if_neg hA, if_neg hB]
    simp


/-! ## The leg count of a grown side -/

/-- The kept legs of a side: its length less its pair ports. -/
def pbl (D : Diagram) (p : DPair) (x : ℕ) (out : Bool) : ℕ :=
  D.sideLen x out - (if x = p.left.node ∧ out = p.left.out then 1 else 0) -
    (if x = p.right.node ∧ out = p.right.out then 1 else 0)

theorem pbl_facts (s : PairSpec D p) (x : ℕ) (out : Bool) :
    ((x = p.left.node ∧ out = p.left.out) → p.left.pos = D.pbl p x out) ∧
    ((x = p.right.node ∧ out = p.right.out) → p.right.pos =
      D.pbl p x out + (if x = p.left.node ∧ out = p.left.out then 1 else 0)) ∧
    D.pbl p x out ≤ D.sideLen x out := by
  obtain ⟨f1, f2, f3⟩ := side_facts s x out
  unfold pbl
  by_cases hA : x = p.left.node ∧ out = p.left.out <;>
    by_cases hB : x = p.right.node ∧ out = p.right.out
  · obtain ⟨r1, r2⟩ := f1 hA hB
    rw [if_pos hA, if_pos hB]
    exact ⟨fun _ => by omega, fun _ => by omega, by omega⟩
  · have := f2 hA hB
    rw [if_pos hA, if_neg hB]
    exact ⟨fun _ => by omega, fun h => absurd h hB, by omega⟩
  · have := f3 hB
    rw [if_neg hA, if_pos hB]
    exact ⟨fun h => absurd h hA, fun _ => by omega, by omega⟩
  · rw [if_neg hA, if_neg hB]
    exact ⟨fun h => absurd h hA, fun h => absurd h hB, by omega⟩

theorem glen (s : PairSpec D p) (K j x : ℕ) (out : Bool) (ds : List ℕ)
    (hds : ds.length = D.sideLen x out) :
    (growC (pcnt p K j) x out ds).length = D.pbl p x out +
      (if x = p.left.node ∧ out = p.left.out then K - 1 - j else 0) +
      (if x = p.right.node ∧ out = p.right.out then j else 0) := by
  rw [length_growC, hds]
  obtain ⟨f1, f2, f3⟩ := side_facts s x out
  have cnt1 : ∀ bl, ((x = p.left.node ∧ out = p.left.out) → bl ≤ p.left.pos) →
      ((x = p.right.node ∧ out = p.right.out) → bl ≤ p.right.pos) →
      ∀ i < bl, pcnt p K j ⟨x, out, i⟩ = 1 := fun bl hl hr i hi => by
    have := keep_low s x out bl i hl hr hi
    simp only [keepP, Bool.and_eq_true, decide_eq_true_eq] at this
    exact pcnt_one K j _ this.1 this.2
  unfold pbl
  by_cases hA : x = p.left.node ∧ out = p.left.out <;>
    by_cases hB : x = p.right.node ∧ out = p.right.out
  · obtain ⟨r1, r2⟩ := f1 hA hB
    rw [if_pos hA, if_pos hB, if_pos hA, if_pos hB, ← r2, List.range_add, List.map_append,
      List.sum_append, sum_ones _ _ (cnt1 _ (fun _ => le_rfl) (fun _ => by omega))]
    have e1 : (⟨x, out, p.left.pos + 1⟩ : Port) = ⟨x, out, p.right.pos⟩ := by rw [r1]
    simp [List.range_succ, pcnt_left K j x out hA, e1, pcnt_right s K j x out hB]
    omega
  · have r2 := f2 hA hB
    rw [if_pos hA, if_neg hB, if_pos hA, if_neg hB, ← r2, List.range_succ, List.map_append,
      List.sum_append, sum_ones _ _ (cnt1 _ (fun _ => le_rfl) (fun h => absurd h hB))]
    simp [pcnt_left K j x out hA]
  · have r3 := f3 hB
    rw [if_neg hA, if_pos hB, if_neg hA, if_pos hB, ← r3, List.range_succ, List.map_append,
      List.sum_append, sum_ones _ _ (cnt1 _ (fun h => absurd h hA) (fun _ => le_rfl))]
    simp [pcnt_right s K j x out hB]
  · rw [if_neg hA, if_neg hB, if_neg hA, if_neg hB,
      sum_ones _ _ (cnt1 _ (fun h => absurd h hA) (fun h => absurd h hB))]
    simp

/-- A leg of copy `j` of a template node: a kept template leg, a left leg, or a right leg. -/
theorem leg_cases (s : PairSpec D p) (K j x : ℕ) (out : Bool) (ds : List ℕ)
    (hds : ds.length = D.sideLen x out) (k : ℕ) (hk : k < (growC (pcnt p K j) x out ds).length) :
    (k < D.sideLen x out ∧ keepP p ⟨x, out, k⟩ = true ∧ posC (pcnt p K j) ⟨x, out, k⟩ = k) ∨
    (x = p.left.node ∧ out = p.left.out ∧ ∃ t < K - 1 - j, k = p.left.pos + t) ∨
    (x = p.right.node ∧ out = p.right.out ∧ ∃ i < j, k = posC (pcnt p K j) p.right + i) := by
  rw [glen s K j x out ds hds] at hk
  obtain ⟨g1, g2, g3⟩ := pbl_facts s x out
  have hr := posC_right s K j
  by_cases hlow : k < D.pbl p x out
  · have hl : (x = p.left.node ∧ out = p.left.out) → D.pbl p x out ≤ p.left.pos := fun h => by
      rw [g1 h]
    have hr' : (x = p.right.node ∧ out = p.right.out) → D.pbl p x out ≤ p.right.pos := fun h => by
      rw [g2 h]; omega
    have hkeep := keep_low s x out _ k hl hr' hlow
    exact Or.inl ⟨by omega, hkeep, posC_keep s K j ⟨x, out, k⟩ (by simp; omega) hkeep⟩
  · right
    by_cases hA : x = p.left.node ∧ out = p.left.out
    · rw [if_pos hA] at hk
      by_cases hB : x = p.right.node ∧ out = p.right.out
      · rw [if_pos hB] at hk
        by_cases hk2 : k < D.pbl p x out + (K - 1 - j)
        · exact Or.inl ⟨hA.1, hA.2, k - D.pbl p x out, by omega, by rw [g1 hA]; omega⟩
        · refine Or.inr ⟨hB.1, hB.2, k - D.pbl p x out - (K - 1 - j), by omega, ?_⟩
          rw [hr, if_pos ⟨hB.1.symm.trans hA.1, hB.2.symm.trans hA.2⟩, g1 hA]; omega
      · rw [if_neg hB] at hk
        exact Or.inl ⟨hA.1, hA.2, k - D.pbl p x out, by omega, by rw [g1 hA]; omega⟩
    · rw [if_neg hA] at hk
      by_cases hB : x = p.right.node ∧ out = p.right.out
      · rw [if_pos hB] at hk
        refine Or.inr ⟨hB.1, hB.2, k - D.pbl p x out, by omega, ?_⟩
        have hS : ¬(p.right.node = p.left.node ∧ p.right.out = p.left.out) := fun h =>
          hA ⟨hB.1.trans h.1, hB.2.trans h.2⟩
        rw [hr, if_neg hS, g2 hB, if_neg hA]; omega
      · rw [if_neg hB] at hk; omega

/-! ## Labels of single legs -/

theorem clab_keep (s : PairSpec D p) (K j : ℕ) (q : Port) (hq : q.pos < D.sideLen q.node q.out)
    (hk : keepP p q = true) :
    D.clab p K j q.node q.out q.pos =
      shJ K D.outs.length D.wires.length j (D.label D.outs.length q) := by
  simp only [keepP, Bool.and_eq_true, decide_eq_true_eq] at hk
  obtain ⟨f1, f2, f3⟩ := side_facts s q.node q.out
  refine clab_low K j q.node q.out q.pos (fun h => ?_) (fun h => ?_)
  · by_cases hB : q.node = p.right.node ∧ q.out = p.right.out
    · obtain ⟨r1, r2⟩ := f1 ⟨h.1, h.2.1⟩ hB
      by_cases e : q.pos = p.left.pos
      · exact hk.1 (port_ext h.1 h.2.1 e)
      · exact hk.2 (port_ext hB.1 hB.2 (by omega))
    · have := f2 ⟨h.1, h.2.1⟩ hB
      by_cases e : q.pos = p.left.pos
      · exact hk.1 (port_ext h.1 h.2.1 e)
      · omega
  · have := f3 ⟨h.1, h.2.1⟩
    exact hk.2 (port_ext h.1 h.2.1 (by omega))

theorem plab_pimg (s : PairSpec D p) (K j : ℕ) (q : Port) (hq : q.pos < D.sideLen q.node q.out)
    (hk : keepP p q = true) :
    D.plab p K (pimg p K j q) = shJ K D.outs.length D.wires.length j (D.label D.outs.length q) := by
  rw [pimg, posC_keep s K j q hq hk, plab_cp]
  exact clab_keep s K j q hq hk

theorem plab_plw (s : PairSpec D p) (K j k : ℕ) (hk : k ∈ later K j) :
    D.plab p K (plw p K j k).1 = gL K D.outs.length D.wires.length j k := by
  have hk' : j + 1 ≤ k ∧ k < j + 1 + (K - 1 - j) := by simpa [later] using hk
  simp only [plw]
  rw [posC_left s, plab_cp, clab_L K j _ _ ⟨rfl, rfl⟩ (k - j - 1) (by omega)]
  congr 1
  omega

theorem plab_prw (s : PairSpec D p) (K i j : ℕ) :
    D.plab p K (prw p K i j).2 = gR K D.outs.length D.wires.length i j := by
  simp only [prw]
  rw [posC_right s]
  split_ifs with hS
  · rw [plab_cp]; exact clab_RA K j _ _ hS i
  · rw [plab_cp]; exact clab_RB K j _ _ hS ⟨rfl, rfl⟩ i

theorem shJ_wire (K m L j i : ℕ) : shJ K m L j (m + i) = K * m + j * L + i := by
  simp [shJ]

theorem keep_of_wire (s : PairSpec D p) {w : Port × Port} (hw : w ∈ D.wires) :
    keepP p w.1 = true ∧ keepP p w.2 = true := by
  obtain ⟨a1, a2, a3, a4, -, -⟩ := s.wires w hw
  simp [keepP, a1, a2, a3, a4]

theorem keep_of_out (s : PairSpec D p) {q : Port} (hq : q ∈ D.outs) : keepP p q = true := by
  have h1 : q ≠ p.left := fun e => s.lbnd (e ▸ List.mem_append_left _ hq)
  have h2 : q ≠ p.right := fun e => s.rbnd (e ▸ List.mem_append_left _ hq)
  simp [keepP, h1, h2]

/-! ## The expansion's network -/

theorem pnode_lab (s : PairSpec D p) (K j : ℕ) {n : DNode} (hn : n ∈ D.nodes) :
    (pcpNode p K j n).node (D.plab p K) = D.pNode p K j n := by
  simp only [DNode.node, pcpNode, pNode, tNode]
  rw [side_labels s K j n.id true n.outs (by rw [sideLen_of s hn]; rfl),
    side_labels s K j n.id false n.ins (by rw [sideLen_of s hn]; rfl)]

theorem fnode_lab (s : PairSpec D p) (K j k : ℕ) :
    ({ p.node with id := pfid j k } : DNode).node (D.plab p K) = D.pFNode p K j k := by
  simp only [DNode.node, pFNode, s.fo, s.fi, List.length_singleton, List.range_one,
    List.map_singleton, plab_fo, plab_fi]

theorem nodes_eq (s : PairSpec D p) (K : ℕ) :
    (D.expandPair p K).nodes.map (fun n => n.node (D.plab p K)) = (D.pairNet p K).nodes := by
  simp only [expandPair, pairNet, List.map_append, List.map_flatMap, List.map_map]
  congr 1
  · refine List.flatMap_congr fun j _ => List.map_congr_left fun n hn => ?_
    exact pnode_lab s K j hn
  · refine List.flatMap_congr fun j _ => List.map_congr_left fun k _ => ?_
    exact fnode_lab s K j k

theorem int_eq (s : PairSpec D p) (K : ℕ) :
    (D.expandPair p K).wires.map (fun w => D.plab p K w.1) = (List.range K).flatMap (D.pInt K) := by
  simp only [expandPair, List.map_flatMap]
  refine List.flatMap_congr fun j hj => ?_
  simp only [List.map_append, List.map_map, pInt]
  congr 1
  congr 1
  · refine List.ext_getElem (by simp) fun i h1 h2 => ?_
    simp only [List.getElem_map, List.getElem_range, Function.comp]
    have hw := List.getElem_mem (l := D.wires) (n := i) (by simpa using h1)
    have hi : i < D.wires.length := by simpa using h1
    rw [plab_pimg s K j _ (s.wires _ hw).2.2.2.2.1 (keep_of_wire s hw).1]
    have := (s.wlab i hi).1
    rw [List.getD_eq_getElem _ _ hi] at this
    rw [this, shJ_wire]
  · exact List.map_congr_left fun k hk => plab_plw s K j k hk
  · exact List.map_congr_left fun i _ => by simp [prw, plab_fo]

theorem prange_blocks (K m : ℕ) :
    (List.range K).flatMap (fun j => (List.range m).map (j * m + ·)) = List.range (K * m) := by
  induction K with
  | zero => simp
  | succ K ih => rw [List.range_succ, List.flatMap_append, ih, Nat.succ_mul, List.range_add]; simp

theorem bnd_map (s : PairSpec D p) (K : ℕ) :
    (D.expandPair p K).bnd.map (D.plab p K) = List.range (K * D.outs.length) := by
  have h0 : (List.range K).flatMap (fun j => D.ins.map (pimg p K j)) = [] := by simp [s.ins]
  simp only [bnd, expandPair, h0, List.append_nil, List.map_flatMap, List.map_map]
  rw [← prange_blocks]
  refine List.flatMap_congr fun j _ => List.ext_getElem (by simp) fun r h1 h2 => ?_
  simp only [List.getElem_map, List.getElem_range, Function.comp]
  have hr : r < D.outs.length := by simpa using h1
  have hq := List.getElem_mem (l := D.outs) hr
  rw [plab_pimg s K j _ (s.outs _ hq) (keep_of_out s hq)]
  have := s.slots r hr
  rw [List.getD_eq_getElem _ _ hr] at this
  rw [this]
  simp [shJ, hr]

theorem bnd_len (s : PairSpec D p) (K : ℕ) : (D.expandPair p K).bnd.length = K * D.outs.length := by
  have := congrArg List.length (bnd_map s K)
  simpa using this

theorem wire_ends (s : PairSpec D p) (K : ℕ) :
    ∀ w ∈ (D.expandPair p K).wires, D.plab p K w.1 = D.plab p K w.2 := by
  intro w hw
  simp only [expandPair, List.mem_flatMap, List.mem_append, List.mem_map, List.mem_range] at hw
  obtain ⟨j, -, ((⟨a, ha, rfl⟩ | ⟨k, hk, rfl⟩) | ⟨i, -, rfl⟩)⟩ := hw
  · obtain ⟨t, ht, rfl⟩ := List.getElem_of_mem ha
    obtain ⟨-, -, -, -, e1, e2⟩ := s.wires _ ha
    simp only
    rw [plab_pimg s K j _ e1 (keep_of_wire s ha).1, plab_pimg s K j _ e2 (keep_of_wire s ha).2]
    have := s.wlab t ht
    rw [List.getD_eq_getElem _ _ ht] at this
    rw [this.1, this.2]
  · rw [plab_plw s K j k hk]; simp [plw, plab_fi]
  · rw [plab_prw s K i j]; simp [prw, plab_fo]

theorem okPort_iff (D' : Diagram) (q : Port) :
    D'.okPort q = true ↔ q ∈ D'.bnd ∨ ∃ w ∈ D'.wires, w.1 = q ∨ w.2 = q := by
  unfold okPort locate
  rcases h1 : D'.bnd.idxOf? q with _ | i
  · have hq : q ∉ D'.bnd := List.idxOf?_eq_none_iff.1 h1
    rcases h2 : D'.wires.findIdx? (fun w => decide (w.1 = q ∨ w.2 = q)) with _ | j
    · have := List.findIdx?_eq_none_iff.1 h2
      simp only [h2, Option.map_none]
      refine ⟨fun h => absurd h (by simp), fun h => ?_⟩
      rcases h with h | ⟨w, hw, h⟩
      · exact absurd h hq
      · exact absurd h (of_decide_eq_false (this w hw))
    · obtain ⟨hj, hp, -⟩ := List.findIdx?_eq_some_iff_getElem.1 h2
      simp only [h2, Option.map_some, decide_eq_true_eq, hj, true_iff]
      exact Or.inr ⟨_, List.getElem_mem hj, by simpa using hp⟩
  · obtain ⟨hi, he, -⟩ := List.idxOf?_eq_some_iff.1 h1
    simp only [decide_eq_true_eq, hi, true_iff]
    exact Or.inl (he ▸ List.getElem_mem hi)

theorem ports_ok (s : PairSpec D p) (K : ℕ) : (D.expandPair p K).portsOK := by
  have mem_b : ∀ j < K, ∀ q ∈ D.outs, pimg p K j q ∈ (D.expandPair p K).bnd := fun j hj q hq => by
    simp only [bnd, expandPair, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range]
    exact Or.inl ⟨j, hj, q, hq, rfl⟩
  have mem_w : ∀ j < K, ∀ a ∈ D.wires, (pimg p K j a.1, pimg p K j a.2) ∈ (D.expandPair p K).wires :=
    fun j hj a ha => by
      simp only [expandPair, List.mem_flatMap, List.mem_append, List.mem_map, List.mem_range]
      exact ⟨j, hj, Or.inl (Or.inl ⟨a, ha, rfl⟩)⟩
  have mem_l : ∀ j < K, ∀ k ∈ later K j, plw p K j k ∈ (D.expandPair p K).wires := fun j hj k hk => by
    simp only [expandPair, List.mem_flatMap, List.mem_append, List.mem_map, List.mem_range]
    exact ⟨j, hj, Or.inl (Or.inr ⟨k, hk, rfl⟩)⟩
  have mem_r : ∀ j < K, ∀ i < j, prw p K i j ∈ (D.expandPair p K).wires := fun j hj i hi => by
    simp only [expandPair, List.mem_flatMap, List.mem_append, List.mem_map, List.mem_range]
    exact ⟨j, hj, Or.inr ⟨i, hi, rfl⟩⟩
  have copyLeg : ∀ j < K, ∀ n ∈ D.nodes, ∀ (out : Bool) (ds : List ℕ),
      ds.length = D.sideLen n.id out → ∀ k < (growC (pcnt p K j) n.id out ds).length,
      (k < D.sideLen n.id out → keepP p ⟨n.id, out, k⟩ = true → D.okPort ⟨n.id, out, k⟩ = true) →
      (D.expandPair p K).okPort ⟨pcp j n.id, out, k⟩ = true := by
    intro j hj n hn out ds hds k hk hok
    rw [okPort_iff]
    rcases leg_cases s K j n.id out ds hds k hk with ⟨h1, h2, h3⟩ | ⟨a1, a2, t, ht, rfl⟩ |
      ⟨b1, b2, i, hi, rfl⟩
    · have hp : pimg p K j ⟨n.id, out, k⟩ = ⟨pcp j n.id, out, k⟩ := by simp [pimg, h3]
      rcases (okPort_iff D _).1 (hok h1 h2) with hb | ⟨a, ha, e⟩
      · left
        rw [← hp]
        exact mem_b j hj _ (by simpa [bnd, s.ins] using hb)
      · right
        refine ⟨_, mem_w j hj a ha, ?_⟩
        rcases e with e | e
        · left; rw [e, hp]
        · right; rw [e, hp]
    · right
      refine ⟨_, mem_l j hj (j + 1 + t) (by simp [later]; omega), Or.inl ?_⟩
      simp only [plw, posC_left s, a1, a2]
      congr 2
      omega
    · right
      exact ⟨_, mem_r j hj i hi, Or.inr (by simp [prw, b1, b2])⟩
  intro n hn
  simp only [expandPair, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range] at hn
  rcases hn with ⟨j, hj, n0, hn0, rfl⟩ | ⟨j, hj, k, hk, rfl⟩
  · refine ⟨fun k hk => ?_, fun k hk => ?_⟩
    · exact copyLeg j hj n0 hn0 true n0.outs (by rw [sideLen_of s hn0]; rfl) k hk fun h1 h2 =>
        s.legsO n0 hn0 k (by rw [sideLen_of s hn0] at h1; simpa using h1) h2
    · exact copyLeg j hj n0 hn0 false n0.ins (by rw [sideLen_of s hn0]; rfl) k hk fun h1 h2 =>
        s.legsI n0 hn0 k (by rw [sideLen_of s hn0] at h1; simpa using h1) h2
  · have hk' : j + 1 ≤ k ∧ k < j + 1 + (K - 1 - j) := by simpa [later] using hk
    refine ⟨fun t ht => ?_, fun t ht => ?_⟩
    · simp only [s.fo, List.length_singleton, Nat.lt_one_iff] at ht
      subst ht
      rw [okPort_iff]
      exact Or.inr ⟨_, mem_r k (by omega) j (by omega), Or.inl rfl⟩
    · simp only [s.fi, List.length_singleton, Nat.lt_one_iff] at ht
      subst ht
      rw [okPort_iff]
      exact Or.inr ⟨_, mem_l j hj k hk, Or.inr rfl⟩

/-- The block of a summed label of `pairNet`. -/
def blkOf (K m L x : ℕ) : ℕ :=
  if x < pG K m L then (x - K * m) / L
  else if (x - pG K m L) % 2 = 0 then ((x - pG K m L) / 2).unpair.1
  else ((x - pG K m L) / 2).unpair.2

theorem mem_pInt (K j x : ℕ) (hj : j < K) (hx : x ∈ D.pInt K j) :
    K * D.outs.length ≤ x ∧ blkOf K D.outs.length D.wires.length x = j := by
  simp only [pInt, List.mem_append, List.mem_map, List.mem_range] at hx
  rcases hx with (⟨i, hi, rfl⟩ | ⟨k, -, rfl⟩) | ⟨i, -, rfl⟩
  · refine ⟨by omega, ?_⟩
    have h1 : K * D.outs.length + j * D.wires.length + i < pG K D.outs.length D.wires.length := by
      unfold pG
      have : (j + 1) * D.wires.length ≤ K * D.wires.length := Nat.mul_le_mul_right _ hj
      rw [Nat.succ_mul] at this
      omega
    simp only [blkOf, h1, ite_true]
    rw [show K * D.outs.length + j * D.wires.length + i - K * D.outs.length =
      i + D.wires.length * j by ring_nf; omega]
    rw [Nat.add_mul_div_left _ _ (by omega : 0 < D.wires.length), Nat.div_eq_of_lt hi, zero_add]
  · refine ⟨by simp [gL, pG]; omega, ?_⟩
    simp [blkOf, gL, Nat.unpair_pair]
  · refine ⟨by unfold gR pG; omega, ?_⟩
    have h1 : ¬(gR K D.outs.length D.wires.length i j < pG K D.outs.length D.wires.length) := by
      unfold gR; omega
    have h2 : (gR K D.outs.length D.wires.length i j - pG K D.outs.length D.wires.length) % 2 = 1 := by
      simp [gR]; omega
    have h3 : (gR K D.outs.length D.wires.length i j - pG K D.outs.length D.wires.length) / 2 =
        Nat.pair i j := by simp [gR]; omega
    simp only [blkOf, h1, ite_false, h2, h3, Nat.unpair_pair]
    simp

theorem pInt_nodup (K j : ℕ) (hj : j < K) : (D.pInt K j).Nodup := by
  simp only [pInt, later_eq, List.map_map]
  refine List.Nodup.append (List.Nodup.append ?_ ?_ ?_) ?_ ?_
  · exact List.nodup_range.map fun a b h => by simpa using h
  · exact List.nodup_range.map fun a b h => by
      simpa [gL, Nat.pair_eq_pair] using h
  · intro x h1 h2
    simp only [List.mem_map, List.mem_range] at h1 h2
    obtain ⟨i, hi, rfl⟩ := h1
    obtain ⟨k, -, e⟩ := h2
    have : (j + 1) * D.wires.length ≤ K * D.wires.length := Nat.mul_le_mul_right _ hj
    rw [Nat.succ_mul] at this
    simp [gL, pG] at e; omega
  · exact List.nodup_range.map fun a b h => by
      simpa [gR, Nat.pair_eq_pair] using h
  · intro x h1 h2
    simp only [List.mem_append, List.mem_map, List.mem_range] at h1 h2
    obtain ⟨i, -, rfl⟩ := h2
    rcases h1 with ⟨i', hi', e⟩ | ⟨k, -, e⟩
    · have : (j + 1) * D.wires.length ≤ K * D.wires.length := Nat.mul_le_mul_right _ hj
      rw [Nat.succ_mul] at this
      simp [gR, pG] at e; omega
    · simp [gL, gR] at e; omega

theorem int_nodup (K : ℕ) : ((List.range K).flatMap (D.pInt K)).Nodup := by
  refine List.nodup_flatMap.2 ⟨fun j hj => pInt_nodup K j (List.mem_range.1 hj), ?_⟩
  refine List.nodup_range.pairwise_of_forall_ne fun i hi j hj hij => ?_
  simp only [Function.onFun, List.disjoint_left]
  intro x h1 h2
  exact hij ((mem_pInt K i x (List.mem_range.1 hi) h1).2.symm.trans
    (mem_pInt K j x (List.mem_range.1 hj) h2).2)

/-- The expansion's network is `pairNet` in the labelling `plab`. -/
theorem netBy_eq (s : PairSpec D p) (K : ℕ) :
    (D.expandPair p K).netBy (D.plab p K) = D.pairNet p K := by
  simp only [netBy]
  rw [nodes_eq s K, int_eq s K, bnd_len s K]
  rfl

/-- At every `K` and every offset above the slots, the expansion's network equals `pairNet`. -/
theorem equal_expandPair (h : D.PairOK p b = true) (E : Env) (hE : ∀ x y, labDim E x = labDim E y)
    (K o : ℕ) (ho : K * D.outs.length ≤ o) :
    Equal E 1 ((D.expandPair p K).toNet o) 1 (D.pairNet p K) := by
  have s := PairOK.spec h
  have hint := int_eq s K
  have e := equal_netBy (D.expandPair p K) E o (D.plab p K) (by rw [bnd_len s]; exact ho)
    (fun i hi => by
      have := List.getElem_of_eq (bnd_map s K) (by simpa using hi)
      simpa using this)
    (wire_ends s K) (ports_ok s K) (by rw [hint]; exact int_nodup K)
    (fun w hw => by
      rw [bnd_len s]
      have hm : D.plab p K w.1 ∈ (D.expandPair p K).wires.map (fun w => D.plab p K w.1) :=
        List.mem_map.2 ⟨w, hw, rfl⟩
      rw [hint, List.mem_flatMap] at hm
      obtain ⟨j, hj, hx⟩ := hm
      exact (mem_pInt K j _ (List.mem_range.1 hj) hx).1)
    (fun _ _ => hE _ _)
  rwa [netBy_eq s K] at e

end Diagram

end ArchytasZX
