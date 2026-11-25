/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamFcaNet

/-!
# The wires of the boxed Fourier constant adder's expansion

`wires0`: the wires of `D0.famExpand n`: each template wire on every copy, then the two pair
boxes' wires over the pairs `(j, k)`. `wlabs0`: their first ends' `lab0` labels.
-/

namespace ArchytasZX

namespace FamFca

open List FcaEng

theorem mv_pos0 (n : ℕ) (q : Port) (hq : q.pos = 0) : D0.mv n q = q := by
  rcases q with ⟨a, o, p⟩
  simp only at hq
  subst hq
  simp [FamDiagram.mv, Diagram.posC]

theorem mem_jk {n : ℕ} {q : ℕ × ℕ} : q ∈ FamDiagram.jk n ↔ q.1 < q.2 ∧ q.2 < n := by
  rcases q with ⟨j, k⟩
  simp only [jk_eq, List.mem_flatMap, List.mem_range, List.mem_map, mem_later, Prod.mk.injEq]
  constructor
  · rintro ⟨m, -, k', ⟨h1, h2⟩, rfl, rfl⟩; exact ⟨h1, h2⟩
  · rintro ⟨h1, h2⟩; exact ⟨j, by omega, k, ⟨h1, h2⟩, rfl, rfl⟩

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

/-- The pair box wires of `D0` over the pairs, box `a`, legs `l`, `r`. -/
def pw (n a l r : ℕ) : List (Port × Port) := (FamDiagram.jk n).flatMap fun q =>
  [(⟨cp0 q.1 l, true, 1 + (q.2 - q.1 - 1)⟩, ⟨D0.pid n a q.1 q.2, false, 0⟩),
   (⟨D0.pid n a q.1 q.2, true, 0⟩, ⟨cp0 q.2 r, false, 1 + q.1⟩)]

theorem wires0 (n : ℕ) : (D0.famExpand n).wires =
    D0.base.wires.flatMap (fun w => (range n).map fun j =>
      ((⟨cp0 j w.1.node, w.1.out, w.1.pos⟩ : Port), (⟨cp0 j w.2.node, w.2.out, w.2.pos⟩ : Port))) ++
    (pw n 0 2 0 ++ pw n 1 5 9) := by
  simp only [FamDiagram.famExpand, X0, copyNodes0_wires, show D0.links = [] from rfl,
    List.flatMap_nil, List.append_nil]
  congr 1
  · rw [List.map_flatMap]
    refine List.flatMap_congr fun w hw => ?_
    have h : w.1.pos = 0 ∧ w.2.pos = 0 := by
      simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hw
      rcases hw with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
    rw [List.map_map]
    exact List.map_congr_left fun j _ => by
      exact Prod.ext (mv_pos0 n ⟨cp0 j w.1.node, w.1.out, w.1.pos⟩ h.1)
        (mv_pos0 n ⟨cp0 j w.2.node, w.2.out, w.2.pos⟩ h.2)
  · have hz : D0.pairs.zipIdx = [(⟨1, 0, ⟨2, true, 1⟩, ⟨0, false, 1⟩, ⟨3, .ZP 0 0 1, 0, 0, [0], [0]⟩⟩, 0),
        (⟨2, 0, ⟨5, true, 1⟩, ⟨9, false, 1⟩, ⟨10, .ZP 0 0 (-1), 0, 0, [0], [0]⟩⟩, 1)] := rfl
    rw [hz]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, FamDiagram.pairWires, pw]
    congr 1
    · refine List.flatMap_congr fun q hq => ?_
      have := mem_jk.1 hq
      rw [at0 (by decide) (by omega), at0 (by decide) (by omega)]
    · refine List.flatMap_congr fun q hq => ?_
      have := mem_jk.1 hq
      rw [at0 (by decide) (by omega), at0 (by decide) (by omega)]

/-! ## The wire labels -/

theorem flatMap_map_perm {β : Type} (l : List ℕ) (rs : List ℕ) (g : ℕ → ℕ → β) :
    (l.flatMap fun m => rs.map (g m)).Perm (rs.flatMap fun r => l.map fun m => g m r) := by
  induction rs with
  | nil => simp
  | cons r rs ih =>
    have e : (l.flatMap fun m => (r :: rs).map (g m)) =
        l.flatMap fun m => [g m r] ++ rs.map (g m) := by simp
    rw [e, List.flatMap_cons]
    refine (List.flatMap_append_perm _ _ _).symm.trans ?_
    rw [flatMap_single (fun m => g m r)]
    exact ih.append_left _

theorem jk_len (n : ℕ) : ∀ m, ((range m).flatMap fun m' => (later n m').map (m', ·)).length = off n m
  | 0 => rfl
  | m + 1 => by
    rw [List.range_succ, List.flatMap_append, List.length_append, jk_len n m]
    simp [off, later]; omega

theorem jk_length (n : ℕ) : (FamDiagram.jk n).length = tri n := by
  rw [jk_eq, jk_len]
  have := off_tri n n le_rfl
  simpa [tri] using this

theorem jk_nodup (n : ℕ) : (FamDiagram.jk n).Nodup := by
  rw [jk_eq, List.nodup_flatMap]
  refine ⟨fun m _ => ?_, ?_⟩
  · exact ((List.nodup_range).map fun a b h => by simpa using h).map fun a b h => by simpa using h
  · refine List.nodup_range.pairwise_of_forall_ne fun a _ b _ hab => ?_
    simp only [Function.onFun, List.disjoint_left, List.mem_map]
    rintro _ ⟨x, -, rfl⟩ ⟨y, -, e⟩
    exact hab (Prod.mk.inj e).1.symm

theorem perm_range_of {l : List (ℕ × ℕ)} (hl : l.Nodup) (f : ℕ × ℕ → ℕ) (T : ℕ)
    (hlen : l.length = T) (hinj : ∀ a ∈ l, ∀ b ∈ l, f a = f b → a = b) (hlt : ∀ a ∈ l, f a < T) :
    (l.map f).Perm (range T) := by
  have hnd : (l.map f).Nodup := hl.map_on fun a ha b hb e => hinj a ha b hb e
  refine (hnd.subperm fun x hx => ?_).perm_of_length_le (by simp [hlen])
  obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hx
  exact List.mem_range.2 (hlt a ha)

theorem a0_perm (n : ℕ) : ((FamDiagram.jk n).map fun q => tri q.2 + q.1).Perm (range (tri n)) := by
  refine perm_range_of (jk_nodup n) _ _ (jk_length n) (fun a ha b hb e => ?_) (fun a ha => ?_)
  · have h1 := mem_jk.1 ha; have h2 := mem_jk.1 hb
    have hk : a.2 = b.2 := by
      have := jOf_tri h1.1; rw [e, jOf_tri h2.1] at this; exact this.symm
    rcases a with ⟨a1, a2⟩; rcases b with ⟨b1, b2⟩
    simp only at hk e ⊢; subst hk; simp only [Prod.mk.injEq, and_true]; omega
  · have h1 := mem_jk.1 ha; exact a0_lt h1.1 h1.2

theorem i2_perm (n : ℕ) : ((FamDiagram.jk n).map fun q => i2 n q.1 q.2).Perm (range (tri n)) := by
  refine perm_range_of (jk_nodup n) _ _ (jk_length n) (fun a ha b hb e => ?_) (fun a ha => ?_)
  · have h1 := mem_jk.1 ha; have h2 := mem_jk.1 hb
    have hm : a.1 = b.1 := by
      have := mOf_off h1.1 h1.2; rw [e, mOf_off h2.1 h2.2] at this; exact this.symm
    rcases a with ⟨a1, a2⟩; rcases b with ⟨b1, b2⟩
    simp only [i2] at hm e h1 h2 ⊢; subst hm; simp only [Prod.mk.injEq, true_and]; omega
  · have h1 := mem_jk.1 ha; exact i2_lt h1.1 h1.2

/-- The chain labels of the copies, wire `c` of every copy in turn. -/
def XL (n : ℕ) : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7].flatMap fun c => (range n).map fun m => 2 * n + 8 * m + c

theorem wlabX (n : ℕ) : (D0.base.wires.flatMap (fun w => (range n).map fun j =>
      ((⟨cp0 j w.1.node, w.1.out, w.1.pos⟩ : Port), (⟨cp0 j w.2.node, w.2.out, w.2.pos⟩ : Port)))).map
      (fun w => lab0 n w.1) = XL n := by
  have hb : ∀ x c, x ∈ ids0 → (∀ j, labX n x j true 0 = 2 * n + 8 * j + c) →
      (range n).map (fun j => lab0 n ⟨cp0 j x, true, 0⟩) = (range n).map fun m => 2 * n + 8 * m + c :=
    fun x c hx h => List.map_congr_left fun j hj => by
      rw [lab0_cp' hx (List.mem_range.1 hj), h]
  simp only [D0, List.flatMap_cons, List.flatMap_nil, List.append_nil, XL]
  simp only [List.map_append, List.map_map, Function.comp_def]
  rw [hb 0 0 (by decide) (fun j => by simp [labX, cb]), hb 1 1 (by decide) (fun j => by simp [labX, cb]),
    hb 2 2 (by decide) (fun j => by simp [labX, cb]), hb 4 3 (by decide) (fun j => by simp [labX, cb]),
    hb 5 4 (by decide) (fun j => by simp [labX, cb]), hb 6 5 (by decide) (fun j => by simp [labX, cb]),
    hb 7 6 (by decide) (fun j => by simp [labX, cb]), hb 8 7 (by decide) (fun j => by simp [labX, cb])]

theorem wlabP (n a l r : ℕ) (hl : l ∈ ids0) :
    (pw n a l r).map (fun w => lab0 n w.1) = (FamDiagram.jk n).flatMap fun q =>
      [labX n l q.1 true (q.2 - q.1 - 1 + 1), labP n a q.1 q.2 true] := by
  simp only [pw, List.map_flatMap]
  refine List.flatMap_congr fun q hq => ?_
  have := mem_jk.1 hq
  simp only [List.map_cons, List.map_nil, lab0_pid, lab0_cp' hl (show q.1 < n by omega),
    Nat.add_comm 1]

theorem wlab0 (n : ℕ) : (D0.famExpand n).wires.map (fun w => lab0 n w.1) =
    XL n ++ ((FamDiagram.jk n).flatMap (fun q => [a2 n q.1 q.2, a0 n q.2 q.1]) ++
      (FamDiagram.jk n).flatMap (fun q => [a5 n q.1 q.2, a9 n q.2 q.1])) := by
  rw [wires0, List.map_append, List.map_append, wlabX, wlabP n 0 2 0 (by decide),
    wlabP n 1 5 9 (by decide)]
  congr 2 <;> refine List.flatMap_congr fun q hq => ?_ <;> have := mem_jk.1 hq <;>
    simp only [labX, labP] <;> rw [show q.1 + 1 + (q.2 - q.1 - 1) = q.2 by omega]

theorem hw0 (n : ℕ) : ∀ w ∈ (D0.famExpand n).wires, lab0 n w.1 = lab0 n w.2 := by
  intro w hw
  rw [wires0, List.mem_append, List.mem_append] at hw
  rcases hw with hw | hw | hw
  · obtain ⟨v, hv, hw⟩ := List.mem_flatMap.1 hw
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hw
    have hj := List.mem_range.1 hj
    have hX : ∀ x y, x ∈ ids0 → y ∈ ids0 → labX n x j true 0 = labX n y j false 0 →
        lab0 n ⟨cp0 j x, true, 0⟩ = lab0 n ⟨cp0 j y, false, 0⟩ := fun x y hx hy h => by
      rw [lab0_cp' hx hj, lab0_cp' hy hj, h]
    simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      exact hX _ _ (by decide) (by decide) (by simp [labX])
  all_goals
    obtain ⟨q, hq, hw⟩ := List.mem_flatMap.1 hw
    obtain ⟨h0, h2⟩ := mem_jk.1 hq
    have h1 : q.1 < n := by omega
    have e : q.1 + 1 + (q.2 - q.1 - 1) = q.2 := by omega
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hw
  · rcases hw with rfl | rfl
    · rw [lab0_cp' (x := 2) (by decide) h1, lab0_pid, Nat.add_comm 1]; simp only [labX, labP, e]
    · rw [lab0_cp' (x := 0) (by decide) h2, lab0_pid, Nat.add_comm 1]; simp only [labX, labP]
  · rcases hw with rfl | rfl
    · rw [lab0_cp' (x := 5) (by decide) h1, lab0_pid, Nat.add_comm 1]; simp only [labX, labP, e]
    · rw [lab0_cp' (x := 9) (by decide) h2, lab0_pid, Nat.add_comm 1]; simp only [labX, labP]

theorem pair_split {β : Type} (l : List (ℕ × ℕ)) (f g : ℕ × ℕ → β) :
    (l.flatMap fun q => [f q, g q]).Perm (l.map f ++ l.map g) := by
  have e : (l.flatMap fun q => [f q, g q]) = l.flatMap fun q => [f q] ++ [g q] := rfl
  rw [e, ← flatMap_single f l, ← flatMap_single g l]
  exact (List.flatMap_append_perm _ _ _).symm

theorem map_shift_perm (n c T : ℕ) (h : ℕ × ℕ → ℕ) (hp : ((FamDiagram.jk n).map h).Perm (range T)) :
    ((FamDiagram.jk n).map fun q => c + h q).Perm ((range T).map (c + ·)) := by
  have := hp.map (c + ·)
  simpa [List.map_map, Function.comp_def] using this

theorem pairP_eq (n : ℕ) : pairP n = (range (tri n)).map (10 * n + ·) ++
    (range (tri n)).map (10 * n + tri n + ·) ++ (range (tri n)).map (10 * n + 2 * tri n + ·) ++
    (range (tri n)).map (10 * n + 3 * tri n + ·) := by
  rw [pairP, show 4 * tri n = tri n + tri n + tri n + tri n by ring, List.range_add, List.range_add,
    List.range_add]
  simp only [List.map_append, List.map_map, Function.comp_def, List.append_assoc, two_mul,
    show 3 * tri n = tri n + tri n + tri n by ring, Nat.add_assoc]

theorem XL_perm (n : ℕ) : (XL n).Perm (frees n ++ chainP n) := by
  have h1 := flatMap_map_perm (range n) [1, 5, 6] fun m r => 2 * n + 8 * m + r
  have h2 := flatMap_map_perm (range n) [0, 2, 3, 4, 7] fun m r => 2 * n + 8 * m + r
  refine List.Perm.trans ?_ (h1.append h2).symm
  rw [List.perm_iff_count]
  intro a
  simp only [XL, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.count_append]
  omega

theorem int0 (n : ℕ) : ((D0.famExpand n).wires.map (fun w => lab0 n w.1)).Perm (engL n).int := by
  rw [wlab0]
  have e0 : (FamDiagram.jk n).map (fun q => a0 n q.2 q.1) =
      (FamDiagram.jk n).map (fun q => 10 * n + (tri q.2 + q.1)) :=
    List.map_congr_left fun q _ => by simp only [a0]; ring
  have e9 : (FamDiagram.jk n).map (fun q => a9 n q.2 q.1) =
      (FamDiagram.jk n).map (fun q => 10 * n + 3 * tri n + (tri q.2 + q.1)) :=
    List.map_congr_left fun q _ => by simp only [a9]; ring
  have q2 : ((FamDiagram.jk n).map (fun q => a2 n q.1 q.2) ++
      (FamDiagram.jk n).map (fun q => a0 n q.2 q.1)).Perm
      ((range (tri n)).map (10 * n + tri n + ·) ++ (range (tri n)).map (10 * n + ·)) := by
    rw [e0]
    exact (map_shift_perm n (10 * n + tri n) (tri n) _ (i2_perm n)).append
      (map_shift_perm n (10 * n) (tri n) _ (a0_perm n))
  have q3 : ((FamDiagram.jk n).map (fun q => a5 n q.1 q.2) ++
      (FamDiagram.jk n).map (fun q => a9 n q.2 q.1)).Perm
      ((range (tri n)).map (10 * n + 2 * tri n + ·) ++ (range (tri n)).map (10 * n + 3 * tri n + ·)) := by
    rw [e9]
    exact (map_shift_perm n (10 * n + 2 * tri n) (tri n) _ (i2_perm n)).append
      (map_shift_perm n (10 * n + 3 * tri n) (tri n) _ (a0_perm n))
  have p2 := (pair_split (FamDiagram.jk n) (fun q => a2 n q.1 q.2) (fun q => a0 n q.2 q.1)).trans q2
  have p3 := (pair_split (FamDiagram.jk n) (fun q => a5 n q.1 q.2) (fun q => a9 n q.2 q.1)).trans q3
  refine ((XL_perm n).append (p2.append p3)).trans ?_
  refine List.Perm.trans ?_ (int_perm n).symm
  rw [pairP_eq, List.perm_iff_count]
  intro a
  simp only [List.count_append]
  omega

end FamFca

end ArchytasZX
