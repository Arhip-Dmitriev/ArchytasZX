/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamDraperWire
import ArchytasZX.FamFcaEq

/-!
# The boxed Draper adder's expansion is `engL`

`ports0`: every leg of `D0.famExpand n` is on its boundary or a wire. `evalL0`: over
`engE d n`, the entry of `D0.famExpand n` is that of `DraperEng.engL n` (`eval_netBy` with
`lab0`, then `nodes0`, `int0`).
-/

namespace ArchytasZX

namespace FamDraper

open List DraperEng
open FcaEng (tri off i2 later nd)

theorem bnd0' (n : ℕ) : (D0.famExpand n).bnd = BO n ++ BI n := by
  have hm : ∀ p ∈ BO n ++ BI n, D0.mv n p = p := by
    intro p hp
    rcases List.mem_append.1 hp with hp | hp <;> obtain ⟨j, hj, hp⟩ := List.mem_flatMap.1 hp <;>
      have hj := List.mem_range.1 hj <;>
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hp <;>
      rcases hp with rfl | rfl <;> rw [mvc (by decide) hj] <;> simp [Diagram.posC]
  have h := List.map_congr_left hm
  simp only [List.map_id'] at h
  rw [← h]
  simp only [Diagram.bnd, FamDiagram.famExpand, X0, copyNodes0_outs, copyNodes0_ins, List.map_append]
  rfl

theorem okX {n j a b p p' : ℕ}
    (h : ((⟨a, true, p⟩ : Port), (⟨b, false, p'⟩ : Port)) ∈ D0.base.wires) (hj : j < n) :
    (D0.famExpand n).okPort ⟨cp0 j a, true, p⟩ = true ∧
      (D0.famExpand n).okPort ⟨cp0 j b, false, p'⟩ = true := by
  have hm : ((⟨cp0 j a, true, p⟩ : Port), (⟨cp0 j b, false, p'⟩ : Port)) ∈ (D0.famExpand n).wires := by
    rw [wires0]
    exact List.mem_append_left _ (List.mem_flatMap.2 ⟨_, h, List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩⟩)
  exact ⟨Diagram.okPort_of_mem _ _ (Or.inr ⟨_, hm, Or.inl rfl⟩),
    Diagram.okPort_of_mem _ _ (Or.inr ⟨_, hm, Or.inr rfl⟩)⟩

theorem okB {n j : ℕ} (hj : j < n) : (D0.famExpand n).okPort ⟨cp0 j 7, true, 0⟩ = true ∧
    (D0.famExpand n).okPort ⟨cp0 j 14, true, 0⟩ = true ∧
    (D0.famExpand n).okPort ⟨cp0 j 4, false, 0⟩ = true ∧
    (D0.famExpand n).okPort ⟨cp0 j 0, false, 0⟩ = true := by
  refine ⟨Diagram.okPort_of_mem _ _ (Or.inl ?_), Diagram.okPort_of_mem _ _ (Or.inl ?_),
    Diagram.okPort_of_mem _ _ (Or.inl ?_), Diagram.okPort_of_mem _ _ (Or.inl ?_)⟩ <;>
    rw [bnd0', List.mem_append]
  · exact Or.inl (List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, by simp⟩)
  · exact Or.inl (List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, by simp⟩)
  · exact Or.inr (List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, by simp⟩)
  · exact Or.inr (List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, by simp⟩)

/-- Both wires of every pair of the pair box `a` with legs `l`, `r`. -/
def PW (n a l r : ℕ) : Prop := ∀ q ∈ FamDiagram.jk n, ∀ w ∈ [(⟨cp0 q.1 l, true, 1 + (q.2 - q.1 - 1)⟩,
      ⟨D0.pid n a q.1 q.2, false, 0⟩), (⟨D0.pid n a q.1 q.2, true, 0⟩, ⟨cp0 q.2 r, false, 1 + q.1⟩)],
      w ∈ (D0.famExpand n).wires

theorem okP {n a l r : ℕ} (hpw : PW n a l r) {q : ℕ × ℕ} (hq : q ∈ FamDiagram.jk n) :
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

theorem pw0_mem (n : ℕ) : PW n 0 2 0 := fun q hq w hw => by
  rw [wires0]
  exact List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _
    (List.mem_flatMap.2 ⟨q, hq, hw⟩)))

theorem pw1_mem (n : ℕ) : PW n 1 8 7 := fun q hq w hw => by
  rw [wires0]
  exact List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _
    (List.mem_flatMap.2 ⟨q, hq, hw⟩)))

theorem pw2_mem (n : ℕ) : PW n 2 10 14 := fun q hq w hw => by
  rw [wires0]
  exact List.mem_append_right _ (List.mem_append_right _ (List.mem_flatMap.2 ⟨q, hq, hw⟩))

theorem okL {n j i l r a : ℕ} (hpw : PW n a l r) (h : j + 1 + i < n) :
    (D0.famExpand n).okPort ⟨cp0 j l, true, i + 1⟩ = true := by
  have := (okP hpw (q := (j, j + 1 + i)) (FamFca.mem_jk.2 ⟨by simp only; omega, h⟩)).1
  simpa [Nat.add_comm 1, show j + 1 + i - j - 1 = i by omega] using this

theorem okR {n k m l r a : ℕ} (hpw : PW n a l r) (hm : m < k) (hk : k < n) :
    (D0.famExpand n).okPort ⟨cp0 k r, false, m + 1⟩ = true := by
  have := (okP hpw (q := (m, k)) (FamFca.mem_jk.2 ⟨hm, hk⟩)).2.2.2
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
    D0.pairNodes n 0 (pb 0) ++ D0.pairNodes n 1 (pb 1) ++ D0.pairNodes n 2 (pb 2) := by
  rw [pairs0]; simp

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
    have w : ∀ a p b p' : ℕ, ((⟨a, true, p⟩ : Port), (⟨b, false, p'⟩ : Port)) ∈ D0.base.wires →
        (D0.famExpand n).okPort ⟨cp0 j a, true, p⟩ = true ∧
          (D0.famExpand n).okPort ⟨cp0 j b, false, p'⟩ = true := fun a p b p' h => okX h hj
    rcases hx' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;>
      simp only [FamFca.growC_two, FamFca.growC_one', cnt0, List.length_append,
        List.length_replicate] at hk <;>
      simp at hk
    -- 0
    · obtain rfl : k = 0 := by omega
      exact (w 0 0 1 0 (by decide)).1
    · rcases k with _ | m
      · exact (okB hj).2.2.2
      · exact okR (pw0_mem n) (by omega) hj
    -- 1
    · obtain rfl : k = 0 := by omega
      exact (w 1 0 2 0 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 0 0 1 0 (by decide)).2
    -- 2
    · rcases k with _ | i
      · exact (w 2 0 5 0 (by decide)).1
      · exact okL (pw0_mem n) (by omega)
    · obtain rfl : k = 0 := by omega
      exact (w 1 0 2 0 (by decide)).2
    -- 4
    · rcases k with _ | _ | k
      · exact (w 4 0 7 0 (by decide)).1
      · exact (w 4 1 6 0 (by decide)).1
      · omega
    · obtain rfl : k = 0 := by omega
      exact (okB hj).2.2.1
    -- 5
    · obtain rfl : k = 0 := by omega
      exact (w 5 0 8 0 (by decide)).1
    · rcases k with _ | _ | k
      · exact (w 2 0 5 0 (by decide)).2
      · exact (w 6 0 5 1 (by decide)).2
      · omega
    -- 6
    · obtain rfl : k = 0 := by omega
      exact (w 6 0 5 1 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 4 1 6 0 (by decide)).2
    -- 7
    · obtain rfl : k = 0 := by omega
      exact (okB hj).1
    · rcases k with _ | m
      · exact (w 4 0 7 0 (by decide)).2
      · exact okR (pw1_mem n) (by omega) hj
    -- 8
    · rcases k with _ | i
      · exact (w 8 0 10 0 (by decide)).1
      · exact okL (pw1_mem n) (by omega)
    · obtain rfl : k = 0 := by omega
      exact (w 5 0 8 0 (by decide)).2
    -- 10
    · rcases k with _ | i
      · exact (w 10 0 11 0 (by decide)).1
      · exact okL (pw2_mem n) (by omega)
    · obtain rfl : k = 0 := by omega
      exact (w 8 0 10 0 (by decide)).2
    -- 11
    · obtain rfl : k = 0 := by omega
      exact (w 11 0 12 0 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 10 0 11 0 (by decide)).2
    -- 12
    · obtain rfl : k = 0 := by omega
      exact (w 12 0 13 0 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 11 0 12 0 (by decide)).2
    -- 13
    · obtain rfl : k = 0 := by omega
      exact (w 13 0 14 0 (by decide)).1
    · obtain rfl : k = 0 := by omega
      exact (w 12 0 13 0 (by decide)).2
    -- 14
    · obtain rfl : k = 0 := by omega
      exact (okB hj).2.1
    · rcases k with _ | m
      · exact (w 13 0 14 0 (by decide)).2
      · exact okR (pw2_mem n) (by omega) hj
  · rw [pairs_flat, List.mem_append, List.mem_append] at hy
    have hP : ∀ a l r, PW n a l r → ∀ y ∈ D0.pairNodes n a (pb a),
        (∀ k < y.outs.length, (D0.famExpand n).okPort ⟨y.id, true, k⟩ = true) ∧
        ∀ k < y.ins.length, (D0.famExpand n).okPort ⟨y.id, false, k⟩ = true := by
      intro a l r hpw y hy
      simp only [FamDiagram.pairNodes, List.mem_map] at hy
      obtain ⟨q, hq, rfl⟩ := hy
      refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;> rcases a with _ | _ | a <;> simp [pb] at hk <;>
        subst hk
      all_goals first | exact (okP hpw hq).2.2.1 | exact (okP hpw hq).2.1
    rcases hy with (hy | hy) | hy
    · exact hP 0 2 0 (pw0_mem n) y hy
    · exact hP 1 8 7 (pw1_mem n) y hy
    · exact hP 2 10 14 (pw2_mem n) y hy

/-! ## The entry -/

theorem legs_zero (n : ℕ) : ∀ y ∈ (D0.famExpand n).nodes, (∀ v ∈ y.outs, v = 0) ∧ ∀ v ∈ y.ins, v = 0 := by
  intro y hy
  simp only [FamDiagram.famExpand, List.mem_append] at hy
  rcases hy with hy | hy
  · obtain ⟨x, hx, j, hj, rfl⟩ := mem_nodesX hy
    rw [node_eq0 hx hj]
    have hx' := hx
    simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hx'
    refine ⟨FamFca.growC_zero _ _ _ _ ?_, FamFca.growC_zero _ _ _ _ ?_⟩ <;>
      rcases hx' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp
  · rw [pairs_flat, List.mem_append, List.mem_append] at hy
    rcases hy with (hy | hy) | hy <;>
    · simp only [FamDiagram.pairNodes, List.mem_map, pb] at hy
      obtain ⟨q, -, rfl⟩ := hy
      simp

theorem portDim0 (n : ℕ) (p : Port) : (D0.famExpand n).portDim p = 0 := by
  unfold Diagram.portDim
  split
  · next y hy =>
    have := legs_zero n y (List.mem_of_find?_eq_some hy)
    split
    · exact FamFca.getD_zero this.1 _
    · exact FamFca.getD_zero this.2 _
  · rfl

theorem flat2_range (c : ℕ) : ∀ n, ((range n).flatMap fun j => [c + 2 * j, c + 2 * j + 1]) =
    (range (2 * n)).map (c + ·)
  | 0 => rfl
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, flat2_range c n,
      show 2 * (n + 1) = 2 * n + 1 + 1 by ring, List.range_succ, List.range_succ]
    simp; ring_nf

theorem bnd_lab0 (n : ℕ) : (D0.famExpand n).bnd.map (lab0 n) = range (4 * n) := by
  rw [bnd0', List.map_append, BO, BI, List.map_flatMap, List.map_flatMap]
  have h1 : ((range n).flatMap fun j => [⟨cp0 j 7, true, 0⟩, ⟨cp0 j 14, true, 0⟩].map (lab0 n)) =
      (range n).flatMap fun j => [0 + 2 * j, 0 + 2 * j + 1] := List.flatMap_congr fun j hj => by
    have hj := List.mem_range.1 hj
    simp only [List.map_cons, List.map_nil]
    rw [lab0_cp' (x := 7) (by decide) hj, lab0_cp' (x := 14) (by decide) hj]
    simp [labX, ao, bo]
  have h2 : ((range n).flatMap fun j => [⟨cp0 j 4, false, 0⟩, ⟨cp0 j 0, false, 0⟩].map (lab0 n)) =
      (range n).flatMap fun j => [2 * n + 2 * j, 2 * n + 2 * j + 1] := List.flatMap_congr fun j hj => by
    have hj := List.mem_range.1 hj
    simp only [List.map_cons, List.map_nil]
    rw [lab0_cp' (x := 4) (by decide) hj, lab0_cp' (x := 0) (by decide) hj]
    simp [labX, ai, bi]
  rw [h1, h2, flat2_range, flat2_range, show 4 * n = 2 * n + 2 * n by ring, List.range_add]
  simp

theorem bnd_len0 (n : ℕ) : (D0.famExpand n).bnd.length = 4 * n := by
  have := congrArg List.length (bnd_lab0 n); simpa using this

theorem wires_len0 (n : ℕ) : (D0.famExpand n).wires.length = (engL n).int.length := by
  have := (int0 n).length_eq; simpa using this

theorem lab_int {n : ℕ} {w : Port × Port} (hw : w ∈ (D0.famExpand n).wires) :
    4 * n ≤ lab0 n w.1 ∧ lab0 n w.1 < rb n := by
  have h := (int0 n).subset (List.mem_map_of_mem (f := fun w => lab0 n w.1) hw)
  simp only [engL, List.mem_map, List.mem_range] at h
  obtain ⟨k, hk, e⟩ := h
  rw [← e, rb]; omega

/-- The expanded boxed side's entry is `engL n`'s. -/
theorem evalL0 (d n : ℕ) (σ : ℕ → ℕ) (hσ : (D0.famExpand n).InRange (engE d n) σ) :
    (D0.famExpand n).eval (engE d n) σ = ArchytasZX.eval (engE d n) (engL n) σ := by
  have hbnd : ∀ i (hi : i < (D0.famExpand n).bnd.length), lab0 n (D0.famExpand n).bnd[i] = i := by
    intro i hi
    have := congrArg (·[i]?) (bnd_lab0 n)
    simp only [List.getElem?_map, List.getElem?_eq_getElem hi, Option.map_some] at this
    rw [List.getElem?_range (by rw [bnd_len0] at hi; exact hi)] at this
    exact Option.some.inj this
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
      exact FamFca.getD_zero (fun v hv => by
        obtain ⟨a, -, rfl⟩ := List.mem_map.1 hv; exact portDim0 n a) i
    rw [hb, labDim_low d hn (by rw [rb]; omega)]
    exact (engD_zero d hn).symm
  rw [Diagram.eval_netBy _ _ (lab0 n) hbnd (hw0 n) (ports0 n) hnd
    (fun w hw => by rw [bnd_len0]; exact (lab_int hw).1) hdim hbd σ hσ]
  exact eval_perm_net _ _ _ (nodes0 n) (int0 n) σ

end FamDraper

end ArchytasZX
