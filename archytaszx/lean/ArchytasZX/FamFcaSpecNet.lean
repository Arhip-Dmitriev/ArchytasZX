/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamFcaSpec
import ArchytasZX.AdderFamFcaEng
import ArchytasZX.ChainFam
import ArchytasZX.FamFcaEq

/-!
# The Fourier constant adder spec's expansion as `engR`

`lab1 n`: the label of each port of `D1.famExpand n` in `FcaEng.engR n`, read from the template
node and copy (`orgOf`). `nodeR`: the copy of each template node under `lab1 n`. `nodesR`: for
`n > 0` its nodes under `lab1 n` are a permutation of `engR n`'s.
-/

namespace ArchytasZX

namespace FamFca

open List FcaEng

/-- The `engR` label of the leg `(out, k)` of copy `j` of the template node `x`. -/
def labY (n x j : ℕ) (out : Bool) (k : ℕ) : ℕ :=
  match x, out, k with
  | 0, true, _ => rb n
  | 1, true, _ => bout n j
  | 1, false, 0 => bin n j
  | 1, false, _ => n + j
  | 2, true, _ => if n = 0 then rb n + 1 else rb n + 2
  | 2, false, _ => if n = 0 then rb n else rb n + 1
  | 3, true, 0 => sv n j
  | 3, true, _ => j
  | 3, false, _ => sv n (j + 1)
  | _, _, _ => if n = 0 then rb n + 1 else rb n + 3

/-- The `engR` label of each port of `D1.famExpand n`. -/
def lab1 (n : ℕ) (p : Port) : ℕ := labY n (D1.orgOf n p.node).1 (D1.orgOf n p.node).2 p.out p.pos

theorem lab1_eq {n : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) (out : Bool) (k : ℕ) :
    lab1 n ⟨e.1.id, out, k⟩ = labY n e.2.1 e.2.2 out k := by
  simp only [lab1, orgOf1_eq he rfl]

theorem side1 (n x j : ℕ) (out : Bool) (ds : List ℕ) : (D1.side n x j out ds).length = ds.length := by
  simp [FamDiagram.side, FamDiagram.cnt_plain n (show D1.pairs = [] from rfl), FamDiagram.growC_one]

theorem node1_lab {n : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) :
    (D1.node n e.1).node (lab1 n) =
      ⟨e.1.kind, (range e.1.outs.length).map (labY n e.2.1 e.2.2 true),
        (range e.1.ins.length).map (labY n e.2.1 e.2.2 false),
        FamDiagram.ixAt n D1.dim e.2.1 e.2.2 e.1.dim, FamDiagram.ixAt n D1.ph e.2.1 e.2.2 e.1.ph⟩ := by
  simp only [FamDiagram.node, orgOf1_eq he rfl, DNode.node, side1]
  congr 1
  · exact List.map_congr_left fun k _ => lab1_eq he true k
  · exact List.map_congr_left fun k _ => lab1_eq he false k

theorem nodes1_fam (n : ℕ) : (D1.famExpand n).nodes = (Z1 n).map fun e => D1.node n e.1 := by
  simp only [FamDiagram.famExpand, show D1.pairs = [] from rfl, List.zipIdx_nil, List.flatMap_nil,
    List.append_nil]
  rw [← zip1, X1]
  have hl : (E2n n).nodes.length = (D1.org n).length := by rw [org1]; simp [E2n]
  rw [show (fun e : DNode × ℕ × ℕ => D1.node n e.1) = D1.node n ∘ Prod.fst from rfl, ← List.map_map,
    List.map_fst_zip (le_of_eq hl)]

/-- The `engR n` nodes, `n > 0`, in `D1`'s template order. -/
def engRo (n : ℕ) : List Node := [hN n] ++ bN n ++ [xN n] ++ sN n ++ [eN n]

theorem nodesR_eq {n : ℕ} (hn : 0 < n) :
    ((D1.famExpand n).nodes.map (·.node (lab1 n))) = engRo n := by
  rw [nodes1_fam, List.map_map]
  have hc : ∀ e ∈ Z1 n, ((fun x : DNode => x.node (lab1 n)) ∘ fun e : DNode × ℕ × ℕ => D1.node n e.1) e =
      (⟨e.1.kind, (range e.1.outs.length).map (labY n e.2.1 e.2.2 true),
        (range e.1.ins.length).map (labY n e.2.1 e.2.2 false),
        FamDiagram.ixAt n D1.dim e.2.1 e.2.2 e.1.dim, FamDiagram.ixAt n D1.ph e.2.1 e.2.2 e.1.ph⟩ : Node) :=
    fun e he => node1_lab he
  rw [List.map_congr_left hc]
  have hn' : n ≠ 0 := by omega
  simp [Z1, engRo, hN, bN, xN, sN, eN, n0, n1, n2, n3, n4, labY, FamDiagram.ixAt, D1, hn', dI,
    List.range_succ, Function.comp_def, List.lookup]

theorem nodesR {n : ℕ} (hn : 0 < n) :
    ((D1.famExpand n).nodes.map (·.node (lab1 n))).Perm (engR n).nodes := by
  rw [nodesR_eq hn]
  unfold engR engRo
  rcases Nat.lt_or_ge n 2 with h | h
  · obtain rfl : n = 1 := by omega
    simp
  · have h0 : n ≠ 0 := by omega
    have h1 : n ≠ 1 := by omega
    simp only [h0, h1, ite_false]
    rw [List.perm_iff_count]
    intro a
    simp only [List.count_append, List.count_cons, List.count_nil]
    omega

/-! ## The wires -/

theorem mem_Z1_1 {n j : ℕ} (hj : j < n) : (({ n1 with id := 5 + j } : DNode), (1, j)) ∈ Z1 n := by
  simp only [Z1, List.mem_append, List.mem_map, List.mem_range]
  exact Or.inl (Or.inl (Or.inl (Or.inr ⟨j, hj, rfl⟩)))

theorem mem_Z1_3 {n j : ℕ} (hj : j < n) : (({ n3 with id := n + 5 + j } : DNode), (3, j)) ∈ Z1 n := by
  simp only [Z1, List.mem_append, List.mem_map, List.mem_range]
  exact Or.inl (Or.inr ⟨j, hj, rfl⟩)

theorem cp1_1 {n j : ℕ} (hj : j < n) : D1.cp n 1 j = 5 + j := cp1_eq (mem_Z1_1 hj)

theorem cp1_3 {n j : ℕ} (hj : j < n) : D1.cp n 3 j = n + 5 + j := cp1_eq (mem_Z1_3 hj)

theorem at1 (n j : ℕ) (q : Port) : D1.at_ n j q = ⟨D1.cp n q.node j, q.out, q.pos⟩ := by
  simp [FamDiagram.at_, FamDiagram.cnt_plain n (show D1.pairs = [] from rfl), FamDiagram.posC_one]

theorem mv1 (n : ℕ) (q : Port) : D1.mv n q = q := FamDiagram.mv_plain n rfl q

/-- The binder chain's wires, `n > 0`. -/
def W1 (n : ℕ) : List (Port × Port) :=
  (⟨0, true, 0⟩, ⟨5, false, 0⟩) :: (range (n - 1)).map (fun j => (⟨5 + j, true, 0⟩, ⟨5 + (j + 1), false, 0⟩)) ++
    [(⟨5 + (n - 1), true, 0⟩, ⟨2, false, 0⟩)]

/-- The splitter chain's wires, `n > 0`. -/
def W2 (n : ℕ) : List (Port × Port) :=
  (⟨4, false, 0⟩, ⟨n + 5, true, 0⟩) ::
    (range (n - 1)).map (fun j => (⟨n + 5 + j, false, 0⟩, ⟨n + 5 + (j + 1), true, 0⟩)) ++
    [(⟨n + 5 + (n - 1), false, 0⟩, ⟨2, true, 0⟩)]

theorem wires1 {n : ℕ} (hn : 0 < n) : (D1.famExpand n).wires = W1 n ++ W2 n := by
  simp only [FamDiagram.famExpand, X1, show (E2n n).wires = [] from rfl, List.map_nil,
    show D1.pairs = [] from rfl, List.zipIdx_nil, List.flatMap_nil, List.append_nil, List.nil_append]
  have hL : D1.links = [⟨⟨1, true, 0⟩, ⟨1, false, 0⟩, ⟨2, false, 0⟩, ⟨0, true, 0⟩⟩,
      ⟨⟨3, false, 0⟩, ⟨3, true, 0⟩, ⟨2, true, 0⟩, ⟨4, false, 0⟩⟩] := rfl
  simp only [hL, List.flatMap_cons, List.flatMap_nil, List.append_nil]
  simp only [FamDiagram.linkWires, show n ≠ 0 by omega, ite_false, mv1, at1]
  have h1 : ∀ j < n, (D1.cp n 1 j) = 5 + j := fun j hj => cp1_1 hj
  have h3 : ∀ j < n, (D1.cp n 3 j) = n + 5 + j := fun j hj => cp1_3 hj
  simp only [W1, W2]
  congr 1
  · rw [h1 0 hn, h1 (n - 1) (by omega)]
    congr 2
    refine List.map_congr_left fun j hj => ?_
    have := List.mem_range.1 hj
    rw [h1 j (by omega), h1 (j + 1) (by omega)]
  · rw [h3 0 hn, h3 (n - 1) (by omega)]
    simp only [Nat.add_zero]
    congr 2
    refine List.map_congr_left fun j hj => ?_
    have := List.mem_range.1 hj
    rw [h3 j (by omega), h3 (j + 1) (by omega)]

/-! ## The wire labels -/

theorem lab1_0 (n : ℕ) (out : Bool) (k : ℕ) : lab1 n ⟨0, out, k⟩ = labY n 0 0 out k :=
  lab1_eq (e := (n0, (0, 0))) (by simp [Z1]) out k

theorem lab1_2 (n : ℕ) (out : Bool) (k : ℕ) : lab1 n ⟨2, out, k⟩ = labY n 2 0 out k :=
  lab1_eq (e := (n2, (2, 0))) (by simp [Z1]) out k

theorem lab1_4 (n : ℕ) (out : Bool) (k : ℕ) : lab1 n ⟨4, out, k⟩ = labY n 4 0 out k :=
  lab1_eq (e := (n4, (4, 0))) (by simp [Z1]) out k

theorem lab1_b {n j : ℕ} (hj : j < n) (out : Bool) (k : ℕ) :
    lab1 n ⟨5 + j, out, k⟩ = labY n 1 j out k := lab1_eq (mem_Z1_1 hj) out k

theorem lab1_s {n j : ℕ} (hj : j < n) (out : Bool) (k : ℕ) :
    lab1 n ⟨n + 5 + j, out, k⟩ = labY n 3 j out k := lab1_eq (mem_Z1_3 hj) out k

/-- The offsets above `rb n` of the wire labels, `n > 0`. -/
def offR (n : ℕ) : List ℕ :=
  [0] ++ (range (n - 1)).map (4 + ·) ++ [1, 3] ++ (range (n - 1)).map (n + 3 + ·) ++ [2]

theorem wlabR {n : ℕ} (hn : 0 < n) :
    (D1.famExpand n).wires.map (fun w => lab1 n w.1) = (offR n).map (rb n + ·) := by
  rw [wires1 hn]
  have hn' : n ≠ 0 := by omega
  simp only [W1, W2, offR, List.map_append, List.map_cons, List.map_nil, List.map_map,
    Function.comp_def, List.cons_append, List.nil_append, List.append_assoc]
  rw [lab1_0, lab1_4, lab1_b (by omega), lab1_s (by omega)]
  simp only [labY, hn', ite_false, bout, sv, show n - 1 + 1 = n by omega, Nat.add_zero, ite_true]
  congr 1
  congr 1
  · refine List.map_congr_left fun j hj => ?_
    have := List.mem_range.1 hj
    rw [lab1_b (by omega)]
    simp only [labY, bout, show j + 1 ≠ n by omega, ite_false]; ring
  · congr 2
    congr 1
    refine List.map_congr_left fun j hj => ?_
    have := List.mem_range.1 hj
    rw [lab1_s (by omega)]
    simp only [labY, sv, show j + 1 ≠ 0 by omega, show j + 1 ≠ n by omega, ite_false]; ring

theorem offR_perm {n : ℕ} (hn : 0 < n) : (offR n).Perm (range (2 * n + 2)) := by
  rw [show 2 * n + 2 = 4 + (n - 1) + (n - 1) by omega, List.range_add, List.range_add]
  rw [List.perm_iff_count]; intro a
  simp only [offR, show range 4 = [0, 1, 2, 3] from rfl,
    show ∀ x, 4 + (n - 1) + x = n + 3 + x from fun x => by omega, List.count_append,
    List.count_cons, List.count_nil]
  omega

theorem intR {n : ℕ} (hn : 0 < n) :
    ((D1.famExpand n).wires.map fun w => lab1 n w.1).Perm (engR n).int := by
  rw [wlabR hn]
  exact (offR_perm hn).map _

theorem hwR {n : ℕ} (hn : 0 < n) : ∀ w ∈ (D1.famExpand n).wires, lab1 n w.1 = lab1 n w.2 := by
  have hn' : n ≠ 0 := by omega
  intro w hw
  rw [wires1 hn, List.mem_append] at hw
  rcases hw with hw | hw
  · simp only [W1, List.mem_cons, List.mem_append, List.mem_map, List.mem_range]
      at hw
    rcases hw with (rfl | ⟨j, hj, rfl⟩) | rfl | hf
    · rw [lab1_0, show (5 : ℕ) = 5 + 0 from rfl, lab1_b hn]; simp [labY, bin]
    · rw [lab1_b (by omega), show 5 + (j + 1) = 5 + (j + 1) from rfl, lab1_b (by omega)]
      simp only [labY, bout, bin, show j + 1 ≠ n by omega, show j + 1 ≠ 0 by omega, ite_false]; ring
    · rw [lab1_b (by omega), lab1_2]; simp [labY, bout, hn', show n - 1 + 1 = n by omega]
    · simp at hf
  · simp only [W2, List.mem_cons, List.mem_append, List.mem_map, List.mem_range]
      at hw
    rcases hw with (rfl | ⟨j, hj, rfl⟩) | rfl | hf
    · rw [lab1_4, show n + 5 = n + 5 + 0 from rfl, lab1_s hn]; simp [labY, sv, hn']
    · rw [lab1_s (by omega), lab1_s (by omega)]; simp [labY]
    · rw [lab1_s (by omega), lab1_2]; simp [labY, sv, hn', show n - 1 + 1 = n by omega]
    · simp at hf

/-! ## Every leg on the boundary or a wire -/

theorem bndR (n : ℕ) : (D1.famExpand n).bnd =
    (range n).map (fun j => (⟨n + 5 + j, true, 1⟩ : Port)) ++ (range n).map fun j => ⟨5 + j, false, 1⟩ := by
  have hm : D1.mv n = id := funext (mv1 n)
  simp only [Diagram.bnd, FamDiagram.famExpand, X1, hm, List.map_id]
  rfl

theorem okW {n : ℕ} (hn : 0 < n) {w : Port × Port} (hw : w ∈ W1 n ∨ w ∈ W2 n) :
    (D1.famExpand n).okPort w.1 = true ∧ (D1.famExpand n).okPort w.2 = true := by
  have hm : w ∈ (D1.famExpand n).wires := by
    rw [wires1 hn, List.mem_append]; exact hw
  exact ⟨Diagram.okPort_of_mem _ _ (Or.inr ⟨w, hm, Or.inl rfl⟩),
    Diagram.okPort_of_mem _ _ (Or.inr ⟨w, hm, Or.inr rfl⟩)⟩

theorem okBR {n : ℕ} {p : Port} (hp : p ∈ (D1.famExpand n).bnd) : (D1.famExpand n).okPort p = true :=
  Diagram.okPort_of_mem _ _ (Or.inl hp)

theorem w1h (n : ℕ) : ((⟨0, true, 0⟩ : Port), (⟨5 + 0, false, 0⟩ : Port)) ∈ W1 n := by simp [W1]

theorem w1m {n j : ℕ} (hj : j + 1 < n) :
    ((⟨5 + j, true, 0⟩ : Port), (⟨5 + (j + 1), false, 0⟩ : Port)) ∈ W1 n := by
  exact List.mem_cons_of_mem _ (List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_range.2 (by omega), rfl⟩))

theorem w1l (n : ℕ) : ((⟨5 + (n - 1), true, 0⟩ : Port), (⟨2, false, 0⟩ : Port)) ∈ W1 n := by simp [W1]

theorem w2h (n : ℕ) : ((⟨4, false, 0⟩ : Port), (⟨n + 5 + 0, true, 0⟩ : Port)) ∈ W2 n := by simp [W2]

theorem w2m {n j : ℕ} (hj : j + 1 < n) :
    ((⟨n + 5 + j, false, 0⟩ : Port), (⟨n + 5 + (j + 1), true, 0⟩ : Port)) ∈ W2 n := by
  exact List.mem_cons_of_mem _ (List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_range.2 (by omega), rfl⟩))

theorem w2l (n : ℕ) : ((⟨n + 5 + (n - 1), false, 0⟩ : Port), (⟨2, true, 0⟩ : Port)) ∈ W2 n := by
  simp [W2]

theorem portsR {n : ℕ} (hn : 0 < n) : (D1.famExpand n).portsOK := by
  intro y hy
  rw [nodes1_fam] at hy
  obtain ⟨e, he, rfl⟩ := List.mem_map.1 hy
  simp only [FamDiagram.node, side1]
  rcases mem_Z1 he with rfl | ⟨j, hj, rfl⟩ | rfl | ⟨j, hj, rfl⟩ | rfl
  · refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;> simp [n0] at hk
    subst hk; exact (okW hn (Or.inl (w1h n))).1
  · refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;> simp [n1] at hk
    · subst hk
      by_cases h : j + 1 < n
      · exact (okW hn (Or.inl (w1m h))).1
      · obtain rfl : j = n - 1 := by omega
        exact (okW hn (Or.inl (w1l n))).1
    · rcases k with _ | k
      · rcases j with _ | i
        · exact (okW hn (Or.inl (w1h n))).2
        · exact (okW hn (Or.inl (w1m (j := i) (by omega)))).2
      · obtain rfl : k = 0 := by omega
        exact okBR (by rw [bndR]; simp only [List.mem_append, List.mem_map, List.mem_range]
                       exact Or.inr ⟨j, hj, rfl⟩)
  · refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;> simp [n2] at hk <;> subst hk
    · exact (okW hn (Or.inr (w2l n))).2
    · exact (okW hn (Or.inl (w1l n))).2
  · refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;> simp [n3] at hk
    · rcases k with _ | k
      · rcases j with _ | i
        · exact (okW hn (Or.inr (w2h n))).2
        · exact (okW hn (Or.inr (w2m (j := i) (by omega)))).2
      · obtain rfl : k = 0 := by omega
        exact okBR (by rw [bndR]; simp only [List.mem_append, List.mem_map, List.mem_range]
                       exact Or.inl ⟨j, hj, rfl⟩)
    · subst hk
      by_cases h : j + 1 < n
      · exact (okW hn (Or.inr (w2m h))).1
      · obtain rfl : j = n - 1 := by omega
        exact (okW hn (Or.inr (w2l n))).1
  · refine ⟨fun k hk => ?_, fun k hk => ?_⟩ <;> simp [n4] at hk
    subst hk; exact (okW hn (Or.inr (w2h n))).1

/-! ## Leg dimensions -/

theorem Z1_id_inj {n : ℕ} {a e : DNode × ℕ × ℕ} (ha : a ∈ Z1 n) (he : e ∈ Z1 n)
    (h : a.1.id = e.1.id) : a = e := by
  rcases mem_Z1 ha with rfl | ⟨j, hj, rfl⟩ | rfl | ⟨j, hj, rfl⟩ | rfl <;>
  rcases mem_Z1 he with rfl | ⟨k, hk, rfl⟩ | rfl | ⟨k, hk, rfl⟩ | rfl <;>
  simp only [n0, n1, n2, n3, n4] at h ⊢ <;> first | omega |
    (have : j = k := by omega
     subst this; rfl)

theorem find1 {n : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) :
    (D1.famExpand n).nodes.find? (fun y => y.id = e.1.id) = some (D1.node n e.1) := by
  rw [nodes1_fam]
  refine find?_unique (List.mem_map.2 ⟨e, he, rfl⟩) (by simp [FamDiagram.node]) fun y hy hp => ?_
  obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hy
  simp only [FamDiagram.node] at hp
  rw [Z1_id_inj ha he (of_decide_eq_true hp)]

theorem portDim1 {n : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) (out : Bool) (k : ℕ)
    (hk : k < (if out then e.1.outs else e.1.ins).length) :
    (D1.famExpand n).portDim ⟨e.1.id, out, k⟩ = FamDiagram.ixAt n D1.legs ⟨e.2.1, out, k⟩ e.2.2 0 := by
  have h0 : ∀ v ∈ e.1.outs ++ e.1.ins, v = 0 := by
    rcases mem_Z1 he with rfl | ⟨j, hj, rfl⟩ | rfl | ⟨j, hj, rfl⟩ | rfl <;> simp [n0, n1, n2, n3, n4]
  unfold Diagram.portDim
  rw [find1 he]
  simp only [FamDiagram.node, orgOf1_eq he rfl, FamDiagram.side,
    FamDiagram.cnt_plain n (show D1.pairs = [] from rfl), FamDiagram.growC_one]
  cases out
  · simp only [Bool.false_eq_true, ite_false] at hk ⊢
    rw [List.getD_eq_getElem _ _ (by simpa using hk)]
    simp only [List.getElem_map, List.getElem_range]
    rw [List.getD_eq_getElem _ _ hk, h0 _ (List.mem_append_right _ (List.getElem_mem hk))]
  · simp only [ite_true] at hk ⊢
    rw [List.getD_eq_getElem _ _ (by simpa using hk)]
    simp only [List.getElem_map, List.getElem_range]
    rw [List.getD_eq_getElem _ _ hk, h0 _ (List.mem_append_left _ (List.getElem_mem hk))]

theorem legs1_0 (n j d : ℕ) : FamDiagram.ixAt n D1.legs ⟨0, true, 0⟩ j d = n := rfl
theorem legs1_1 (n j d : ℕ) :
    FamDiagram.ixAt n D1.legs ⟨1, true, 0⟩ j d = if j + 1 = 0 then n else j + 1 - 1 := rfl
theorem legs1_3 (n j d : ℕ) :
    FamDiagram.ixAt n D1.legs ⟨3, false, 0⟩ j d = if j + 1 = 0 then n else j + 1 - 1 := rfl
theorem legs1_4 (n j d : ℕ) : FamDiagram.ixAt n D1.legs ⟨4, false, 0⟩ j d = n := rfl

theorem dimW {n : ℕ} (hn : 0 < n) (d : ℕ) : ∀ w ∈ (D1.famExpand n).wires,
    labDim (engE d n) (lab1 n w.1) = (engE d n).dimv ((D1.famExpand n).portDim w.1) := by
  have hn' : n ≠ 0 := by omega
  have key : ∀ k, k < 2 * n + 2 → ∀ p : Port, lab1 n p = rb n + k →
      (D1.famExpand n).portDim p = dI n (edim n k) →
      labDim (engE d n) (lab1 n p) = (engE d n).dimv ((D1.famExpand n).portDim p) := by
    intro k hk p hl hp
    rw [hl, hp, labDim_eng d hk]
    exact (engD_dI d (edim_le hk)).symm
  intro w hw
  rw [wires1 hn, List.mem_append] at hw
  rcases hw with hw | hw
  · simp only [W1, List.mem_cons, List.mem_append, List.mem_map, List.mem_range] at hw
    rcases hw with (rfl | ⟨j, hj, rfl⟩) | rfl | hf
    · refine key 0 (by omega) _ (by rw [lab1_0]; simp [labY]) ?_
      refine (portDim1 (e := (n0, (0, 0))) (by simp [Z1]) true 0 (by simp [n0])).trans ?_
      simp [legs1_0, edim, dI]
    · refine key (4 + j) (by omega) _ (by
        rw [lab1_b (by omega)]; simp only [labY, bout, show j + 1 ≠ n by omega, ite_false]; ring) ?_
      rw [portDim1 (mem_Z1_1 (j := j) (by omega)) true 0 (by simp [n1])]
      dsimp only
      simp only [legs1_1, edim, dI]
      split_ifs <;> first | omega | contradiction
    · refine key 1 (by omega) _ (by
        rw [lab1_b (by omega)]; simp [labY, bout, show n - 1 + 1 = n by omega]) ?_
      rw [portDim1 (mem_Z1_1 (j := n - 1) (by omega)) true 0 (by simp [n1])]
      dsimp only
      simp only [legs1_1, edim, dI]
      split_ifs <;> first | omega | contradiction
    · simp at hf
  · simp only [W2, List.mem_cons, List.mem_append, List.mem_map, List.mem_range] at hw
    rcases hw with (rfl | ⟨j, hj, rfl⟩) | rfl | hf
    · refine key 3 (by omega) _ (by rw [lab1_4]; simp [labY, hn']) ?_
      refine (portDim1 (e := (n4, (4, 0))) (by simp [Z1]) false 0 (by simp [n4])).trans ?_
      simp [legs1_4, edim, dI]
    · refine key (n + 3 + j) (by omega) _ (by
        rw [lab1_s (by omega)]
        simp only [labY, sv, show j + 1 ≠ 0 by omega, show j + 1 ≠ n by omega, ite_false]; ring) ?_
      rw [portDim1 (mem_Z1_3 (j := j) (by omega)) false 0 (by simp [n3])]
      dsimp only
      simp only [legs1_3, edim, dI]
      split_ifs <;> first | omega | contradiction
    · refine key 2 (by omega) _ (by
        rw [lab1_s (by omega)]; simp [labY, sv, hn', show n - 1 + 1 = n by omega]) ?_
      rw [portDim1 (mem_Z1_3 (j := n - 1) (by omega)) false 0 (by simp [n3])]
      dsimp only
      simp only [legs1_3, edim, dI]
      split_ifs <;> first | omega | contradiction
    · simp at hf

/-! ## The entry -/

theorem legs1_31 (n j d : ℕ) : FamDiagram.ixAt n D1.legs ⟨3, true, 1⟩ j d = d := rfl
theorem legs1_11 (n j d : ℕ) : FamDiagram.ixAt n D1.legs ⟨1, false, 1⟩ j d = d := rfl

theorem bnd_lenR (n : ℕ) : (D1.famExpand n).bnd.length = 2 * n := by
  rw [bndR]; simp; ring

/-- The expanded spec's entry is `engR n`'s, `n > 0`. -/
theorem evalR0 (d : ℕ) {n : ℕ} (hn : 0 < n) (σ : ℕ → ℕ) (hσ : (D1.famExpand n).InRange (engE d n) σ) :
    (D1.famExpand n).eval (engE d n) σ = ArchytasZX.eval (engE d n) (engR n) σ := by
  have hbnd : ∀ i (hi : i < (D1.famExpand n).bnd.length), lab1 n (D1.famExpand n).bnd[i] = i := by
    intro i hi
    have hl := bnd_lenR n
    rw [List.getElem_of_eq (bndR n)]
    by_cases h : i < n
    · rw [List.getElem_append_left (by simpa using h)]
      simp only [List.getElem_map, List.getElem_range]
      rw [lab1_s h]; rfl
    · rw [List.getElem_append_right (by simpa using h)]
      simp only [List.getElem_map, List.getElem_range, List.length_map, List.length_range]
      rw [lab1_b (by omega)]
      simp only [labY]; omega
  have hnd : ((D1.famExpand n).wires.map fun w => lab1 n w.1).Nodup :=
    (intR hn).nodup_iff.2 (engR_int_nodup n)
  have hmem : ∀ w ∈ (D1.famExpand n).wires, rb n ≤ lab1 n w.1 ∧ lab1 n w.1 < rb n + (2 * n + 2) :=
    fun w hw => by
      have h := (intR hn).subset (List.mem_map_of_mem (f := fun w : Port × Port => lab1 n w.1) hw)
      exact mem_engR_int.1 h
  have hdim : ∀ j (hj : j < (D1.famExpand n).wires.length),
      labDim (engE d n) (lab1 n (D1.famExpand n).wires[j].1) =
        (engE d n).dimv ((D1.famExpand n).wireDim j) := by
    intro j hj
    rw [dimW hn d _ (List.getElem_mem hj)]
    simp only [Diagram.wireDim, List.getElem?_eq_getElem hj, Option.map_some, Option.getD_some]
  have hbd : ∀ i < (D1.famExpand n).bnd.length,
      labDim (engE d n) i = (engE d n).dimv ((D1.famExpand n).bdims.getD i 0) := by
    intro i hi
    rw [bnd_lenR] at hi
    have hb : (D1.famExpand n).bdims.getD i 0 = 0 := by
      simp only [Diagram.bdims]
      refine getD_zero (fun v hv => ?_) i
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hv
      rw [bndR, List.mem_append] at hp
      rcases hp with hp | hp <;> obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hp <;>
        have hj := List.mem_range.1 hj
      · exact (portDim1 (mem_Z1_3 hj) true 1 (by simp [n3])).trans (legs1_31 _ _ _)
      · exact (portDim1 (mem_Z1_1 hj) false 1 (by simp [n1])).trans (legs1_11 _ _ _)
    rw [hb, labDim_low d hn (by unfold rb; nlinarith)]
    exact (engD_zero d hn).symm
  rw [Diagram.eval_netBy _ _ (lab1 n) hbnd (hwR hn) (portsR hn) hnd
    (fun w hw => by rw [bnd_lenR]; have := hmem w hw; unfold rb at this; nlinarith) hdim hbd σ hσ]
  exact eval_perm_net _ _ _ (nodesR hn) (intR hn) σ

end FamFca

end ArchytasZX
