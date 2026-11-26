/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamDraperNet
import ArchytasZX.FamFcaWire

/-!
# The wires of the boxed Draper adder's expansion

`wires0`: the wires of `D0.famExpand n`: each template wire on every copy, then the three pair
boxes' wires over the pairs `(j, k)`. `hw0`: both ends of each wire have one `lab0` label.
`int0`: the first ends' labels are a permutation of `engL n`'s inner labels.
-/

namespace ArchytasZX

namespace FamDraper

open List DraperEng
open FcaEng (tri off i2 later nd)

theorem mem_nodes {x : ℕ} (hx : x ∈ ids0) : ∃ y ∈ D0.base.nodes, y.id = x := by
  rw [← ids0_eq] at hx; simpa using hx

theorem cp_eq0' {n j x : ℕ} (hx : x ∈ ids0) (hj : j < n) : D0.cp n x j = cp0 j x := by
  obtain ⟨y, hy, rfl⟩ := mem_nodes hx; exact cp_eq0 hy hj

theorem lab0_cp' {n j x : ℕ} (hx : x ∈ ids0) (hj : j < n) (out : Bool) (k : ℕ) :
    lab0 n ⟨cp0 j x, out, k⟩ = labX n x j out k := by
  obtain ⟨y, hy, rfl⟩ := mem_nodes hx; exact lab0_cp hy hj out k

theorem at0 {n j x : ℕ} (hx : x ∈ ids0) (hj : j < n) (out : Bool) :
    D0.at_ n j ⟨x, out, 1⟩ = ⟨cp0 j x, out, 1⟩ := by
  simp only [FamDiagram.at_, cp_eq0' hx hj, Diagram.posC, cnt0]
  simp

theorem mvc {n j x : ℕ} (hx : x ∈ ids0) (hj : j < n) (out : Bool) (p : ℕ) :
    D0.mv n ⟨cp0 j x, out, p⟩ = ⟨cp0 j x, out, Diagram.posC (D0.cnt n j) ⟨x, out, p⟩⟩ := by
  obtain ⟨y, hy, rfl⟩ := mem_nodes hx
  simp only [FamDiagram.mv, orgOf0 hy hj]

/-- The pair box wires of `D0` over the pairs, box `a`, legs `l`, `r`. -/
def pw (n a l r : ℕ) : List (Port × Port) := (FamDiagram.jk n).flatMap fun q =>
  [(⟨cp0 q.1 l, true, 1 + (q.2 - q.1 - 1)⟩, ⟨D0.pid n a q.1 q.2, false, 0⟩),
   (⟨D0.pid n a q.1 q.2, true, 0⟩, ⟨cp0 q.2 r, false, 1 + q.1⟩)]

/-- The template wires on every copy. -/
def tw (n : ℕ) : List (Port × Port) := D0.base.wires.flatMap fun w => (range n).map fun j =>
  ((⟨cp0 j w.1.node, w.1.out, w.1.pos⟩ : Port), (⟨cp0 j w.2.node, w.2.out, w.2.pos⟩ : Port))

theorem wires0 (n : ℕ) : (D0.famExpand n).wires = tw n ++ (pw n 0 2 0 ++ pw n 1 8 7 ++ pw n 2 10 14) := by
  simp only [FamDiagram.famExpand, X0, copyNodes0_wires, show D0.links = [] from rfl,
    List.flatMap_nil, List.append_nil, tw]
  congr 1
  · rw [List.map_flatMap]
    refine List.flatMap_congr fun w hw => ?_
    rw [List.map_map]
    refine List.map_congr_left fun j hj => ?_
    have hj := List.mem_range.1 hj
    simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hw
    rcases hw with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp only [Function.comp, Prod.mk.injEq] <;> refine ⟨?_, ?_⟩ <;> rw [mvc ?_ hj] <;>
      first | decide | simp [Diagram.posC, cnt0]
  · rw [pairs0]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, FamDiagram.pairWires, pw, pb,
      List.append_assoc]
    congr 1
    · refine List.flatMap_congr fun q hq => ?_
      have := FamFca.mem_jk.1 hq
      rw [at0 (by decide) (by omega), at0 (by decide) (by omega)]
    congr 1
    · refine List.flatMap_congr fun q hq => ?_
      have := FamFca.mem_jk.1 hq
      rw [at0 (by decide) (by omega), at0 (by decide) (by omega)]
    · refine List.flatMap_congr fun q hq => ?_
      have := FamFca.mem_jk.1 hq
      rw [at0 (by decide) (by omega), at0 (by decide) (by omega)]

/-! ## The wire labels -/

/-- The chain labels of the copies, template wire by template wire. -/
def XL (n : ℕ) : List ℕ :=
  [0, 1, 4, 2, 3, 5, 6].flatMap (fun c => (range n).map fun m => 4 * n + 11 * m + c) ++
    (range n).map (g8o n) ++
    [7, 8, 9, 10].flatMap (fun c => (range n).map fun m => 4 * n + 11 * m + c)

theorem wlabX (n : ℕ) : (tw n).map (fun w => lab0 n w.1) = XL n := by
  have hb : ∀ x p (f : ℕ → ℕ), x ∈ ids0 → (∀ j, labX n x j true p = f j) →
      (range n).map (fun j => lab0 n ⟨cp0 j x, true, p⟩) = (range n).map f :=
    fun x p f hx h => List.map_congr_left fun j hj => by
      rw [lab0_cp' hx (List.mem_range.1 hj), h]
  simp only [tw, D0, List.flatMap_cons, List.flatMap_nil, List.append_nil, XL]
  simp only [List.map_append, List.map_map, Function.comp_def, List.append_assoc]
  have hc : ∀ x p c, x ∈ ids0 → (∀ j, labX n x j true p = cb n j + c) →
      (range n).map (fun j => lab0 n ⟨cp0 j x, true, p⟩) = (range n).map fun m => 4 * n + 11 * m + c :=
    fun x p c hx h => hb x p _ hx fun j => by rw [h]; rfl
  rw [hc 0 0 0 (by decide) (fun j => rfl), hc 1 0 1 (by decide) (fun j => rfl),
    hc 2 0 4 (by decide) (fun j => rfl), hc 4 0 2 (by decide) (fun j => rfl),
    hc 4 1 3 (by decide) (fun j => rfl), hc 6 0 5 (by decide) (fun j => rfl),
    hc 5 0 6 (by decide) (fun j => rfl), hb 8 0 (g8o n) (by decide) (fun j => rfl),
    hc 10 0 7 (by decide) (fun j => rfl), hc 11 0 8 (by decide) (fun j => rfl),
    hc 12 0 9 (by decide) (fun j => rfl), hc 13 0 10 (by decide) (fun j => rfl)]

theorem wlabP (n a l r : ℕ) (hl : l ∈ ids0) :
    (pw n a l r).map (fun w => lab0 n w.1) = (FamDiagram.jk n).flatMap fun q =>
      [labX n l q.1 true (q.2 - q.1 - 1 + 1), labP n a q.1 q.2 true] := by
  simp only [pw, List.map_flatMap]
  refine List.flatMap_congr fun q hq => ?_
  have := FamFca.mem_jk.1 hq
  simp only [List.map_cons, List.map_nil, lab0_pid, lab0_cp' hl (show q.1 < n by omega),
    Nat.add_comm 1]

theorem wlab0 (n : ℕ) : (D0.famExpand n).wires.map (fun w => lab0 n w.1) =
    XL n ++ ((FamDiagram.jk n).flatMap (fun q => [a2 n q.1 q.2, a0 n q.2 q.1]) ++
      (FamDiagram.jk n).flatMap (fun q => [a8 n q.1 q.2, a7 n q.2 q.1]) ++
      (FamDiagram.jk n).flatMap (fun q => [a10 n q.1 q.2, a14 n q.2 q.1])) := by
  rw [wires0, List.map_append, List.map_append, List.map_append, wlabX, wlabP n 0 2 0 (by decide),
    wlabP n 1 8 7 (by decide), wlabP n 2 10 14 (by decide)]
  refine congrArg (XL n ++ ·) (congrArg₂ (· ++ ·) (congrArg₂ (· ++ ·) ?_ ?_) ?_) <;>
    refine List.flatMap_congr fun q hq => ?_ <;> have := FamFca.mem_jk.1 hq <;>
    simp only [labX, labP] <;> rw [show q.1 + 1 + (q.2 - q.1 - 1) = q.2 by omega]

theorem hw0 (n : ℕ) : ∀ w ∈ (D0.famExpand n).wires, lab0 n w.1 = lab0 n w.2 := by
  intro w hw
  rw [wires0, List.mem_append, List.mem_append, List.mem_append] at hw
  rcases hw with hw | (hw | hw) | hw
  · obtain ⟨v, hv, hw⟩ := List.mem_flatMap.1 hw
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hw
    have hj := List.mem_range.1 hj
    have hX : ∀ x y p p', x ∈ ids0 → y ∈ ids0 → labX n x j true p = labX n y j false p' →
        lab0 n ⟨cp0 j x, true, p⟩ = lab0 n ⟨cp0 j y, false, p'⟩ := fun x y p p' hx hy h => by
      rw [lab0_cp' hx hj, lab0_cp' hy hj, h]
    simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      exact hX _ _ _ _ (by decide) (by decide) (by simp [labX])
  all_goals
    obtain ⟨q, hq, hw⟩ := List.mem_flatMap.1 hw
    obtain ⟨h0, h2⟩ := FamFca.mem_jk.1 hq
    have h1 : q.1 < n := by omega
    have e : q.1 + 1 + (q.2 - q.1 - 1) = q.2 := by omega
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hw
  · rcases hw with rfl | rfl
    · rw [lab0_cp' (x := 2) (by decide) h1, lab0_pid, Nat.add_comm 1]; simp only [labX, labP, e]
    · rw [lab0_cp' (x := 0) (by decide) h2, lab0_pid, Nat.add_comm 1]; simp only [labX, labP]
  · rcases hw with rfl | rfl
    · rw [lab0_cp' (x := 8) (by decide) h1, lab0_pid, Nat.add_comm 1]; simp only [labX, labP, e]
    · rw [lab0_cp' (x := 7) (by decide) h2, lab0_pid, Nat.add_comm 1]; simp only [labX, labP]
  · rcases hw with rfl | rfl
    · rw [lab0_cp' (x := 10) (by decide) h1, lab0_pid, Nat.add_comm 1]; simp only [labX, labP, e]
    · rw [lab0_cp' (x := 14) (by decide) h2, lab0_pid, Nat.add_comm 1]; simp only [labX, labP]

/-! ## The labels as `engL n`'s inner labels -/

theorem XL_perm (n : ℕ) : (XL n).Perm (frees n ++ chainP n ++ (range n).map (g8o n)) := by
  have h1 := FamFca.flatMap_map_perm (range n) [1, 8, 9] fun m r => 4 * n + 11 * m + r
  have h2 := FamFca.flatMap_map_perm (range n) [0, 2, 3, 4, 5, 6, 7, 10] fun m r => 4 * n + 11 * m + r
  refine List.Perm.trans ?_ ((h1.append h2).append_right _).symm
  rw [List.perm_iff_count]
  intro a
  simp only [XL, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.count_append]
  omega

theorem g8_perm (n : ℕ) : ((range n).map (g8o n) ++ (FamDiagram.jk n).map (fun q => a8 n q.1 q.2)).Perm
    ((range (tri n + n)).map (15 * n + 3 * tri n + ·)) := by
  set l : List (ℕ × ℕ) := (range n).map (fun m => (m, m)) ++ FamDiagram.jk n
  have hmem : ∀ q ∈ l, q.1 ≤ q.2 ∧ q.2 < n := fun q hq => by
    rcases List.mem_append.1 hq with hq | hq
    · obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hq; exact ⟨le_rfl, List.mem_range.1 hm⟩
    · have := FamFca.mem_jk.1 hq; omega
  have hnd : l.Nodup := by
    refine List.nodup_append.2 ⟨(List.nodup_range).map fun a b h => by simpa using h,
      FamFca.jk_nodup n, fun a ha b hb e => ?_⟩
    obtain ⟨m, -, rfl⟩ := List.mem_map.1 ha
    have := FamFca.mem_jk.1 hb; subst e; exact absurd this.1 (lt_irrefl _)
  have hp := FamFca.perm_range_of hnd (fun q => off2 n q.1 + (q.2 - q.1)) (tri n + n)
    (by simp [l, FamFca.jk_length]; ring)
    (fun a ha b hb e => by
      have h1 := hmem a ha; have h2 := hmem b hb
      have hm : a.1 = b.1 := by
        have := mOf2_eq (n := n) (m := a.1) (t := a.2 - a.1) (by omega) (by omega)
        rw [e, mOf2_eq (by omega) (by omega)] at this; exact this.symm
      rcases a with ⟨a1, a2⟩; rcases b with ⟨b1, b2⟩
      try simp only at hm e h1 h2 ⊢
      subst hm; simp only [Prod.mk.injEq, true_and]; omega)
    (fun a ha => by
      have h1 := hmem a ha
      have := off2_succ (n := n) (m := a.1) (by omega)
      have := off2_mono n (show a.1 + 1 ≤ n by omega)
      have := off2_top n
      omega)
  have := hp.map (15 * n + 3 * tri n + ·)
  refine List.Perm.trans (List.Perm.of_eq ?_) this
  simp only [l, List.map_append, List.map_map]
  congr 1
  · exact List.map_congr_left fun m _ => by simp [Function.comp, g8o_eq]
  · exact List.map_congr_left fun q hq => by
      have := FamFca.mem_jk.1 hq
      simp only [Function.comp, a8_eq this.1]

theorem pairP_eq (n : ℕ) : pairP n = (range (tri n)).map (15 * n + ·) ++
    (range (tri n)).map (15 * n + tri n + ·) ++ (range (tri n)).map (15 * n + 2 * tri n + ·) ++
    (range (tri n + n)).map (15 * n + 3 * tri n + ·) ++
    (range (tri n)).map (15 * n + 4 * tri n + n + ·) ++ (range (tri n)).map (15 * n + 5 * tri n + n + ·) := by
  rw [pairP, show n + 6 * tri n = tri n + tri n + tri n + (tri n + n) + tri n + tri n by ring]
  simp only [List.range_add, List.map_append, List.map_map, List.append_assoc]
  refine congrArg₂ (· ++ ·) ?_ (congrArg₂ (· ++ ·) ?_ (congrArg₂ (· ++ ·) ?_
    (congrArg₂ (· ++ ·) ?_ (congrArg₂ (· ++ ·) ?_ (congrArg₂ (· ++ ·) ?_ ?_))))) <;>
    exact List.map_congr_left fun x _ => by
      first | rfl | (simp only [Function.comp_apply]; ring)

theorem tri_perm (n c : ℕ) : ((FamDiagram.jk n).map fun q => c + tri q.2 + q.1).Perm
    ((range (tri n)).map (c + ·)) := by
  have := FamFca.map_shift_perm n c (tri n) _ (FamFca.a0_perm n)
  simpa [Nat.add_assoc] using this

theorem i2_perm' (n c : ℕ) : ((FamDiagram.jk n).map fun q => c + i2 n q.1 q.2).Perm
    ((range (tri n)).map (c + ·)) :=
  FamFca.map_shift_perm n c (tri n) _ (FamFca.i2_perm n)

theorem int0 (n : ℕ) : ((D0.famExpand n).wires.map (fun w => lab0 n w.1)).Perm (engL n).int := by
  have p0 : ((FamDiagram.jk n).map fun q => a0 n q.2 q.1).Perm
      ((range (tri n)).map (15 * n + ·)) := tri_perm n (15 * n)
  have p2 : ((FamDiagram.jk n).map fun q => a2 n q.1 q.2).Perm
      ((range (tri n)).map (15 * n + tri n + ·)) := i2_perm' n (15 * n + tri n)
  have p7 : ((FamDiagram.jk n).map fun q => a7 n q.2 q.1).Perm
      ((range (tri n)).map (15 * n + 2 * tri n + ·)) := tri_perm n (15 * n + 2 * tri n)
  have p10 : ((FamDiagram.jk n).map fun q => a10 n q.1 q.2).Perm
      ((range (tri n)).map (15 * n + 4 * tri n + n + ·)) := i2_perm' n (15 * n + 4 * tri n + n)
  have p14 : ((FamDiagram.jk n).map fun q => a14 n q.2 q.1).Perm
      ((range (tri n)).map (15 * n + 5 * tri n + n + ·)) := tri_perm n (15 * n + 5 * tri n + n)
  rw [wlab0, List.perm_iff_count]
  intro a
  have c1 := (XL_perm n).count_eq a
  have c2 := (FamFca.pair_split (FamDiagram.jk n) (fun q => a2 n q.1 q.2) (fun q => a0 n q.2 q.1)).count_eq a
  have c3 := (FamFca.pair_split (FamDiagram.jk n) (fun q => a8 n q.1 q.2) (fun q => a7 n q.2 q.1)).count_eq a
  have c4 := (FamFca.pair_split (FamDiagram.jk n) (fun q => a10 n q.1 q.2) (fun q => a14 n q.2 q.1)).count_eq a
  have c5 := p0.count_eq a
  have c6 := p2.count_eq a
  have c7 := p7.count_eq a
  have c8 := p10.count_eq a
  have c9 := p14.count_eq a
  have c10 := (g8_perm n).count_eq a
  have c11 := (int_perm n).count_eq a
  rw [pairP_eq] at c11
  simp only [List.count_append] at c1 c2 c3 c4 c10 c11 ⊢
  omega

end FamDraper

end ArchytasZX
