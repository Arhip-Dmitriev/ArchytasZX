/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamFcaSpec
import ArchytasZX.AdderFamFcaEng

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

end FamFca

end ArchytasZX
