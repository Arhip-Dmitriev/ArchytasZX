/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamFcaWire
import ArchytasZX.ChainLabel
import ArchytasZX.ChainFam
import ArchytasZX.FamEval

/-!
# The boxed Fourier constant adder's expansion is `engL`

`ports0`: every leg of `D0.famExpand n` is on its boundary or a wire. `net0`: over
`engE d n`, `(D0.famExpand n).toNet (2 n)` equals `FcaEng.engL n` (`equal_netBy` with `lab0`,
then `nodes0`, `int0`).
-/

namespace ArchytasZX

namespace FamFca

open List FcaEng

theorem bnd0' (n : ℕ) : (D0.famExpand n).bnd =
    (range n).map (fun j => (⟨cp0 j 9, true, 0⟩ : Port)) ++
      (range n).map (fun j => (⟨cp0 j 0, false, 0⟩ : Port)) := by
  simp only [Diagram.bnd, FamDiagram.famExpand, X0, copyNodes0_outs, copyNodes0_ins, List.map_map]
  congr 1

theorem okX {n j a b : ℕ} (h : ((⟨a, true, 0⟩ : Port), (⟨b, false, 0⟩ : Port)) ∈ D0.base.wires)
    (hj : j < n) : (D0.famExpand n).okPort ⟨cp0 j a, true, 0⟩ = true ∧
      (D0.famExpand n).okPort ⟨cp0 j b, false, 0⟩ = true := by
  have hm : ((⟨cp0 j a, true, 0⟩ : Port), (⟨cp0 j b, false, 0⟩ : Port)) ∈ (D0.famExpand n).wires := by
    rw [wires0]
    exact List.mem_append_left _ (List.mem_flatMap.2 ⟨_, h, List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩⟩)
  exact ⟨Diagram.okPort_of_mem _ _ (Or.inr ⟨_, hm, Or.inl rfl⟩),
    Diagram.okPort_of_mem _ _ (Or.inr ⟨_, hm, Or.inr rfl⟩)⟩

theorem okB {n j : ℕ} (hj : j < n) : (D0.famExpand n).okPort ⟨cp0 j 9, true, 0⟩ = true ∧
    (D0.famExpand n).okPort ⟨cp0 j 0, false, 0⟩ = true := by
  refine ⟨Diagram.okPort_of_mem _ _ (Or.inl ?_), Diagram.okPort_of_mem _ _ (Or.inl ?_)⟩ <;>
    rw [bnd0'] <;> simp only [List.mem_append, List.mem_map, List.mem_range]
  · exact Or.inl ⟨j, hj, rfl⟩
  · exact Or.inr ⟨j, hj, rfl⟩

theorem okP {n a l r : ℕ} (hpw : ∀ q ∈ FamDiagram.jk n, ∀ w ∈ [(⟨cp0 q.1 l, true, 1 + (q.2 - q.1 - 1)⟩,
      ⟨D0.pid n a q.1 q.2, false, 0⟩), (⟨D0.pid n a q.1 q.2, true, 0⟩, ⟨cp0 q.2 r, false, 1 + q.1⟩)],
      w ∈ (D0.famExpand n).wires) {q : ℕ × ℕ} (hq : q ∈ FamDiagram.jk n) :
    (D0.famExpand n).okPort ⟨cp0 q.1 l, true, 1 + (q.2 - q.1 - 1)⟩ = true ∧
    (D0.famExpand n).okPort ⟨D0.pid n a q.1 q.2, false, 0⟩ = true ∧
    (D0.famExpand n).okPort ⟨D0.pid n a q.1 q.2, true, 0⟩ = true ∧
    (D0.famExpand n).okPort ⟨cp0 q.2 r, false, 1 + q.1⟩ = true := by
  have h1 := hpw q hq _ List.mem_cons_self
  have h2 := hpw q hq _ (List.mem_cons_of_mem _ List.mem_cons_self)
  exact ⟨Diagram.okPort_of_mem _ _ (Or.inr ⟨_, h1, Or.inl rfl⟩),
    Diagram.okPort_of_mem _ _ (Or.inr ⟨_, h1, Or.inr rfl⟩),
    Diagram.okPort_of_mem _ _ (Or.inr ⟨_, h2, Or.inl rfl⟩),
    Diagram.okPort_of_mem _ _ (Or.inr ⟨_, h2, Or.inr rfl⟩)⟩

theorem pw0_mem (n : ℕ) : ∀ q ∈ FamDiagram.jk n, ∀ w ∈ [(⟨cp0 q.1 2, true, 1 + (q.2 - q.1 - 1)⟩,
      ⟨D0.pid n 0 q.1 q.2, false, 0⟩), (⟨D0.pid n 0 q.1 q.2, true, 0⟩, ⟨cp0 q.2 0, false, 1 + q.1⟩)],
      w ∈ (D0.famExpand n).wires := fun q hq w hw => by
  rw [wires0]
  exact List.mem_append_right _ (List.mem_append_left _ (List.mem_flatMap.2 ⟨q, hq, hw⟩))

theorem pw1_mem (n : ℕ) : ∀ q ∈ FamDiagram.jk n, ∀ w ∈ [(⟨cp0 q.1 5, true, 1 + (q.2 - q.1 - 1)⟩,
      ⟨D0.pid n 1 q.1 q.2, false, 0⟩), (⟨D0.pid n 1 q.1 q.2, true, 0⟩, ⟨cp0 q.2 9, false, 1 + q.1⟩)],
      w ∈ (D0.famExpand n).wires := fun q hq w hw => by
  rw [wires0]
  exact List.mem_append_right _ (List.mem_append_right _ (List.mem_flatMap.2 ⟨q, hq, hw⟩))

/-- The left leg `i + 1` of copy `j`: the pair `(j, j + 1 + i)`. -/
theorem okL {n j i l r a : ℕ} (hpw : ∀ q ∈ FamDiagram.jk n, ∀ w ∈ [(⟨cp0 q.1 l, true, 1 + (q.2 - q.1 - 1)⟩,
      ⟨D0.pid n a q.1 q.2, false, 0⟩), (⟨D0.pid n a q.1 q.2, true, 0⟩, ⟨cp0 q.2 r, false, 1 + q.1⟩)],
      w ∈ (D0.famExpand n).wires) (h : j + 1 + i < n) :
    (D0.famExpand n).okPort ⟨cp0 j l, true, i + 1⟩ = true := by
  have := (okP hpw (q := (j, j + 1 + i)) (mem_jk.2 ⟨by simp only; omega, h⟩)).1
  simpa [Nat.add_comm 1, show j + 1 + i - j - 1 = i by omega] using this

/-- The right leg `m + 1` of copy `k`: the pair `(m, k)`. -/
theorem okR {n k m l r a : ℕ} (hpw : ∀ q ∈ FamDiagram.jk n, ∀ w ∈ [(⟨cp0 q.1 l, true, 1 + (q.2 - q.1 - 1)⟩,
      ⟨D0.pid n a q.1 q.2, false, 0⟩), (⟨D0.pid n a q.1 q.2, true, 0⟩, ⟨cp0 q.2 r, false, 1 + q.1⟩)],
      w ∈ (D0.famExpand n).wires) (hm : m < k) (hk : k < n) :
    (D0.famExpand n).okPort ⟨cp0 k r, false, m + 1⟩ = true := by
  have := (okP hpw (q := (m, k)) (mem_jk.2 ⟨hm, hk⟩)).2.2.2
  simpa [Nat.add_comm 1] using this

theorem node_eq0 {n j : ℕ} {x : DNode} (hx : x ∈ D0.base.nodes) (hj : j < n) :
    D0.node n { x with id := cp0 j x.id } =
      { x with id := cp0 j x.id, ph := FamDiagram.ixAt n D0.ph x.id j x.ph,
               dim := FamDiagram.ixAt n D0.dim x.id j x.dim,
               outs := Diagram.growC (D0.cnt n j) x.id true x.outs,
               ins := Diagram.growC (D0.cnt n j) x.id false x.ins } := by
  simp only [FamDiagram.node, orgOf0 hx hj, side0]

theorem mem_nodesX {n : ℕ} {y : DNode} (hy : y ∈ (D0.X n).nodes.map (D0.node n)) :
    ∃ x ∈ D0.base.nodes, ∃ j < n, y = D0.node n { x with id := cp0 j x.id } := by
  obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hy
  rw [X0, copyNodes0_nodes] at hz
  obtain ⟨x, hx, hz⟩ := List.mem_flatMap.1 hz
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hz
  exact ⟨x, hx, j, List.mem_range.1 hj, rfl⟩

theorem pairs_flat (n : ℕ) : D0.pairs.zipIdx.flatMap (fun q => D0.pairNodes n q.2 q.1) =
    D0.pairNodes n 0 ⟨1, 0, ⟨2, true, 1⟩, ⟨0, false, 1⟩, ⟨3, .ZP 0 0 1, 0, 0, [0], [0]⟩⟩ ++
    D0.pairNodes n 1 ⟨2, 0, ⟨5, true, 1⟩, ⟨9, false, 1⟩, ⟨10, .ZP 0 0 (-1), 0, 0, [0], [0]⟩⟩ := by
  simp [D0, List.zipIdx]

theorem ports0 (n : ℕ) : (D0.famExpand n).portsOK := by
  intro y hy
  simp only [FamDiagram.famExpand, List.mem_append] at hy
  rcases hy with hy | hy
  · obtain ⟨x, hx, j, hj, rfl⟩ := mem_nodesX hy
    rw [node_eq0 hx hj]
    show (∀ k < (Diagram.growC (D0.cnt n j) x.id true x.outs).length,
        (D0.famExpand n).okPort ⟨cp0 j x.id, true, k⟩ = true) ∧
      ∀ k < (Diagram.growC (D0.cnt n j) x.id false x.ins).length,
        (D0.famExpand n).okPort ⟨cp0 j x.id, false, k⟩ = true
    have hx' := hx
    simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hx'
    have w : ∀ a b : ℕ, ((⟨a, true, 0⟩ : Port), (⟨b, false, 0⟩ : Port)) ∈ D0.base.wires →
        (D0.famExpand n).okPort ⟨cp0 j a, true, 0⟩ = true ∧
          (D0.famExpand n).okPort ⟨cp0 j b, false, 0⟩ = true := fun a b h => okX h hj
    rcases hx' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;>
      simp only [growC_two, growC_one', cnt0, List.length_append, List.length_replicate] at hk <;>
      simp at hk
    · obtain rfl : k = 0 := by omega
      exact (w 0 1 (by decide)).1
    · rcases k with _ | m
      · exact (okB hj).2
      · exact okR (pw0_mem n) (by omega) hj
    · obtain rfl : k = 0 := by omega
      exact (w 1 2 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 0 1 (by decide)).2
    · rcases k with _ | i
      · exact (w 2 4 (by decide)).1
      · exact okL (pw0_mem n) (by omega)
    · obtain rfl : k = 0 := by omega
      exact (w 1 2 (by decide)).2
    · obtain rfl : k = 0 := by omega
      exact (w 4 5 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 2 4 (by decide)).2
    · rcases k with _ | i
      · exact (w 5 6 (by decide)).1
      · exact okL (pw1_mem n) (by omega)
    · obtain rfl : k = 0 := by omega
      exact (w 4 5 (by decide)).2
    · obtain rfl : k = 0 := by omega
      exact (w 6 7 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 5 6 (by decide)).2
    · obtain rfl : k = 0 := by omega
      exact (w 7 8 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 6 7 (by decide)).2
    · obtain rfl : k = 0 := by omega
      exact (w 8 9 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 7 8 (by decide)).2
    · obtain rfl : k = 0 := by omega
      exact (okB hj).1
    · rcases k with _ | m
      · exact (w 8 9 (by decide)).2
      · exact okR (pw1_mem n) (by omega) hj
  · rw [pairs_flat, List.mem_append] at hy
    rcases hy with hy | hy <;>
    · simp only [FamDiagram.pairNodes, List.mem_map] at hy
      obtain ⟨q, hq, rfl⟩ := hy
      refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;> simp at hk <;> subst hk <;>
      first
        | exact (okP (pw0_mem n) hq).2.2.1
        | exact (okP (pw1_mem n) hq).2.2.1
        | exact (okP (pw0_mem n) hq).2.1
        | exact (okP (pw1_mem n) hq).2.1

/-! ## The entry -/

theorem growC_zero (c : Port → ℕ) (x : ℕ) (out : Bool) (ds : List ℕ) (h : ∀ v ∈ ds, v = 0) :
    ∀ v ∈ Diagram.growC c x out ds, v = 0 := by
  intro v hv
  simp only [Diagram.growC, List.mem_flatMap, List.mem_replicate] at hv
  obtain ⟨i, -, -, rfl⟩ := hv
  rcases Nat.lt_or_ge i ds.length with hi | hi
  · rw [List.getD_eq_getElem _ _ hi]; exact h _ (List.getElem_mem hi)
  · rw [List.getD_eq_default _ _ hi]

theorem legs_zero (n : ℕ) : ∀ y ∈ (D0.famExpand n).nodes, (∀ v ∈ y.outs, v = 0) ∧ ∀ v ∈ y.ins, v = 0 := by
  intro y hy
  simp only [FamDiagram.famExpand, List.mem_append] at hy
  rcases hy with hy | hy
  · obtain ⟨x, hx, j, hj, rfl⟩ := mem_nodesX hy
    rw [node_eq0 hx hj]
    have hx' := hx
    simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hx'
    refine ⟨growC_zero _ _ _ _ ?_, growC_zero _ _ _ _ ?_⟩ <;>
      rcases hx' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp
  · rw [pairs_flat, List.mem_append] at hy
    rcases hy with hy | hy <;>
    · simp only [FamDiagram.pairNodes, List.mem_map] at hy
      obtain ⟨q, -, rfl⟩ := hy
      simp

theorem getD_zero {l : List ℕ} (h : ∀ v ∈ l, v = 0) (i : ℕ) : l.getD i 0 = 0 := by
  rcases Nat.lt_or_ge i l.length with hi | hi
  · rw [List.getD_eq_getElem _ _ hi]; exact h _ (List.getElem_mem hi)
  · rw [List.getD_eq_default _ _ hi]

theorem portDim0 (n : ℕ) (p : Port) : (D0.famExpand n).portDim p = 0 := by
  unfold Diagram.portDim
  split
  · next y hy =>
    have := legs_zero n y (List.mem_of_find?_eq_some hy)
    split
    · exact getD_zero this.1 _
    · exact getD_zero this.2 _
  · rfl

theorem bnd_len0 (n : ℕ) : (D0.famExpand n).bnd.length = 2 * n := by
  rw [bnd0']; simp; ring

theorem wires_len0 (n : ℕ) : (D0.famExpand n).wires.length = (engL n).int.length := by
  have := (int0 n).length_eq; simpa using this

theorem lab_int {n : ℕ} {w : Port × Port} (hw : w ∈ (D0.famExpand n).wires) :
    2 * n ≤ lab0 n w.1 ∧ lab0 n w.1 < rb n := by
  have h := (int0 n).subset (List.mem_map_of_mem (f := fun w => lab0 n w.1) hw)
  simp only [engL, List.mem_map, List.mem_range] at h
  obtain ⟨k, hk, e⟩ := h
  rw [← e, rb_eq]; omega

/-- The expanded boxed side's entry is `engL n`'s. -/
theorem evalL0 (d n : ℕ) (σ : ℕ → ℕ) (hσ : (D0.famExpand n).InRange (engE d n) σ) :
    (D0.famExpand n).eval (engE d n) σ = ArchytasZX.eval (engE d n) (engL n) σ := by
  have hbnd : ∀ i (hi : i < (D0.famExpand n).bnd.length), lab0 n (D0.famExpand n).bnd[i] = i := by
    intro i hi
    have hl := bnd_len0 n
    rw [List.getElem_of_eq (bnd0' n)]
    by_cases h : i < n
    · rw [List.getElem_append_left (by simpa using h)]
      simp only [List.getElem_map, List.getElem_range]
      rw [lab0_cp' (by decide) h]; rfl
    · rw [List.getElem_append_right (by simpa using h)]
      simp only [List.getElem_map, List.getElem_range, List.length_map, List.length_range]
      rw [lab0_cp' (by decide) (by omega)]
      simp only [labX]; omega
  have hnd : ((D0.famExpand n).wires.map fun w => lab0 n w.1).Nodup :=
    (int0 n).nodup_iff.2 ((List.nodup_range).map fun a b h => by simpa using h)
  have hdim : ∀ j (hj : j < (D0.famExpand n).wires.length),
      labDim (engE d n) (lab0 n (D0.famExpand n).wires[j].1) =
        (engE d n).dimv ((D0.famExpand n).wireDim j) := by
    intro j hj
    have hn : 0 < n := by
      rcases Nat.eq_zero_or_pos n with rfl | h
      · rw [wires_len0] at hj; simp [engL, tri] at hj
      · exact h
    have hl := lab_int (List.getElem_mem hj)
    rw [labDim_low d hn hl.2]
    simp only [Diagram.wireDim, List.getElem?_eq_getElem hj, Option.map_some, Option.getD_some,
      portDim0]
    exact (engD_zero d hn).symm
  have hbd : ∀ i < (D0.famExpand n).bnd.length,
      labDim (engE d n) i = (engE d n).dimv ((D0.famExpand n).bdims.getD i 0) := by
    intro i hi
    rw [bnd_len0] at hi
    have hn : 0 < n := by omega
    have hb : (D0.famExpand n).bdims.getD i 0 = 0 := by
      exact getD_zero (fun v hv => by
        obtain ⟨a, -, rfl⟩ := List.mem_map.1 hv; exact portDim0 n a) i
    rw [hb, labDim_low d hn (by rw [rb_eq]; omega)]
    exact (engD_zero d hn).symm
  rw [Diagram.eval_netBy _ _ (lab0 n) hbnd (hw0 n) (ports0 n) hnd
    (fun w hw => by rw [bnd_len0]; exact (lab_int hw).1) hdim hbd σ hσ]
  exact eval_perm_net _ _ _ (nodes0 n) (int0 n) σ

end FamFca

end ArchytasZX
