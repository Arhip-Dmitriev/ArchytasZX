/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamDraperExpand
import ArchytasZX.FamFcaNet
import ArchytasZX.AdderFamDraperEng

/-!
# The boxed Draper adder's expansion as `engL`

`lab0 n`: the label of each port of `D0.famExpand n` in `DraperEng.engL n`, read from the
template node and copy (`orgOf`) of a copied node, from `Nat.unpair` of the id above
`(D0.X n).fresh` of a pair node. `nodes0`: the nodes of `D0.famExpand n` under `lab0 n` are a
permutation of `engL n`'s.
-/

namespace ArchytasZX

namespace FamDraper

open List DraperEng
open FcaEng (tri off i2 later nd)

/-- The label of the leg `(out, k)` of copy `j` of the template node `x`. -/
def labX (n x j : ℕ) (out : Bool) (k : ℕ) : ℕ :=
  match x, out, k with
  | 0, true, _ => cb n j
  | 0, false, 0 => bi n j
  | 0, false, k + 1 => a0 n j k
  | 1, true, _ => cb n j + 1
  | 1, false, _ => cb n j
  | 2, true, 0 => cb n j + 4
  | 2, true, k + 1 => a2 n j (j + 1 + k)
  | 2, false, _ => cb n j + 1
  | 4, true, 0 => cb n j + 2
  | 4, true, _ => cb n j + 3
  | 4, false, _ => ai n j
  | 5, true, _ => cb n j + 6
  | 5, false, 0 => cb n j + 4
  | 5, false, _ => cb n j + 5
  | 6, true, _ => cb n j + 5
  | 6, false, _ => cb n j + 3
  | 7, true, _ => ao j
  | 7, false, 0 => cb n j + 2
  | 7, false, k + 1 => a7 n j k
  | 8, true, 0 => g8o n j
  | 8, true, k + 1 => a8 n j (j + 1 + k)
  | 8, false, _ => cb n j + 6
  | 10, true, 0 => cb n j + 7
  | 10, true, k + 1 => a10 n j (j + 1 + k)
  | 10, false, _ => g8o n j
  | 11, true, _ => cb n j + 8
  | 11, false, _ => cb n j + 7
  | 12, true, _ => cb n j + 9
  | 12, false, _ => cb n j + 8
  | 13, true, _ => cb n j + 10
  | 13, false, _ => cb n j + 9
  | 14, true, _ => bo j
  | 14, false, 0 => cb n j + 10
  | 14, false, k + 1 => a14 n j k
  | _, _, _ => 0

/-- The label of the leg `out` of the pair `(j, k)` of the `a`-th pair box. -/
def labP (n a j k : ℕ) (out : Bool) : ℕ :=
  match a, out with
  | 0, true => a0 n k j
  | 0, false => a2 n j k
  | 1, true => a7 n k j
  | 1, false => a8 n j k
  | _, true => a14 n k j
  | _, false => a10 n j k

/-- The `engL` label of each port of `D0.famExpand n`. -/
def lab0 (n : ℕ) (p : Port) : ℕ :=
  if p.node < (D0.X n).fresh then labX n (D0.orgOf n p.node).1 (D0.orgOf n p.node).2 p.out p.pos
  else
    let u := (p.node - (D0.X n).fresh).unpair
    labP n u.1 u.2.unpair.1 u.2.unpair.2 p.out

/-- The `engL n` node of copy `j` of the template node `x`. -/
def engNode (n x j : ℕ) : Node :=
  match x with
  | 0 => nd .Z [cb n j] (bi n j :: (range j).map (a0 n j))
  | 1 => nd .F [cb n j + 1] [cb n j]
  | 2 => nd .Z ((cb n j + 4) :: (later n j).map (a2 n j)) [cb n j + 1]
  | 4 => nd .Z [cb n j + 2, cb n j + 3] [ai n j]
  | 5 => nd .Z [cb n j + 6] [cb n j + 4, cb n j + 5]
  | 6 => nd .F [cb n j + 5] [cb n j + 3]
  | 7 => nd .Z [ao j] ((cb n j + 2) :: (range j).map (a7 n j))
  | 8 => nd .Z (g8o n j :: (later n j).map (a8 n j)) [cb n j + 6]
  | 10 => nd .Z ((cb n j + 7) :: (later n j).map (a10 n j)) [g8o n j]
  | 11 => nd .F [cb n j + 8] [cb n j + 7]
  | 12 => nd .F [cb n j + 9] [cb n j + 8]
  | 13 => nd .F [cb n j + 10] [cb n j + 9]
  | _ => nd .Z [bo j] ((cb n j + 10) :: (range j).map (a14 n j))

theorem cp0_lt {n j : ℕ} {x : DNode} (hx : x ∈ D0.base.nodes) (hj : j < n) :
    cp0 j x.id < (D0.X n).fresh := by
  refine Diagram.lt_fresh _ ?_
  rw [X0, copyNodes0_nodes, List.map_flatMap]
  exact List.mem_flatMap.2 ⟨x, hx, by simpa using ⟨j, hj, rfl⟩⟩

theorem lab0_cp {n j : ℕ} {x : DNode} (hx : x ∈ D0.base.nodes) (hj : j < n) (out : Bool) (k : ℕ) :
    lab0 n ⟨cp0 j x.id, out, k⟩ = labX n x.id j out k := by
  simp only [lab0, cp0_lt hx hj, ite_true, orgOf0 hx hj]

theorem cnt0 (n j : ℕ) (q : Port) : D0.cnt n j q =
    if q = ⟨2, true, 1⟩ ∨ q = ⟨8, true, 1⟩ ∨ q = ⟨10, true, 1⟩ then n - 1 - j
    else if q = ⟨0, false, 1⟩ ∨ q = ⟨7, false, 1⟩ ∨ q = ⟨14, false, 1⟩ then j else 1 := by
  simp only [FamDiagram.cnt, D0, List.any_cons, List.any_nil, Bool.or_false, Bool.or_eq_true,
    decide_eq_true_eq]
  by_cases h1 : q = ⟨2, true, 1⟩ <;> by_cases h2 : q = ⟨8, true, 1⟩ <;>
    by_cases h3 : q = ⟨10, true, 1⟩ <;> simp [h1, h2, h3, eq_comm]

theorem side0 (n x j : ℕ) (out : Bool) (ds : List ℕ) :
    D0.side n x j out ds = Diagram.growC (D0.cnt n j) x out ds := by
  simp only [FamDiagram.side, show D0.legs = [] from rfl, FamDiagram.ixAt, List.lookup_nil,
    FamDiagram.map_getD]

theorem node_lab0 {n j : ℕ} {x : DNode} (hx : x ∈ D0.base.nodes) (hj : j < n) :
    (D0.node n { x with id := cp0 j x.id }).node (lab0 n) =
      ⟨x.kind, (range (Diagram.growC (D0.cnt n j) x.id true x.outs).length).map (labX n x.id j true),
        (range (Diagram.growC (D0.cnt n j) x.id false x.ins).length).map (labX n x.id j false),
        FamDiagram.ixAt n D0.dim x.id j x.dim, FamDiagram.ixAt n D0.ph x.id j x.ph⟩ := by
  simp only [FamDiagram.node, orgOf0 hx hj, side0, DNode.node]
  congr 1
  · exact List.map_congr_left fun k _ => lab0_cp hx hj true k
  · exact List.map_congr_left fun k _ => lab0_cp hx hj false k

theorem ixDim0 (n x j d : ℕ) (h : (x == 3 || x == 9 || x == 15) = false) :
    FamDiagram.ixAt n D0.dim x j d = d := by
  simp only [Bool.or_eq_false_iff] at h
  simp [FamDiagram.ixAt, D0, List.lookup_cons, h.1.1, h.1.2, h.2]

theorem ixPh0 (n x j d : ℕ) : FamDiagram.ixAt n D0.ph x j d = d := rfl

theorem node0 {n j : ℕ} {x : DNode} (hx : x ∈ D0.base.nodes) (hj : j < n) :
    (D0.node n { x with id := cp0 j x.id }).node (lab0 n) = engNode n x.id j := by
  rw [node_lab0 hx hj]
  have hx' := hx
  simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hx'
  rcases hx' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [FamFca.growC_two, FamFca.growC_one', cnt0, ixPh0] <;>
    simp [engNode, nd, labX, FamFca.range_one_add, FamFca.later_eq, List.map_map,
      Function.comp_def] <;>
    exact ixDim0 _ _ _ _ rfl

/-! ## The node list -/

theorem fresh_le_pid (n a j k : ℕ) : (D0.X n).fresh ≤ D0.pid n a j k := by
  simp only [FamDiagram.pid]; omega

theorem lab0_pid (n a j k : ℕ) (out : Bool) : lab0 n ⟨D0.pid n a j k, out, 0⟩ = labP n a j k out := by
  have h := fresh_le_pid n a j k
  unfold lab0
  split
  · next h' => exact absurd (show D0.pid n a j k < (D0.X n).fresh from h') (by omega)
  · simp only [FamDiagram.pid, Nat.add_sub_cancel_left, Nat.unpair_pair]

theorem ixDimP (n j d : ℕ) : FamDiagram.ixAt n D0.dim 3 j d = j ∧ FamDiagram.ixAt n D0.dim 9 j d = j ∧
    FamDiagram.ixAt n D0.dim 15 j d = j := ⟨rfl, rfl, rfl⟩

/-- The pair box `a` of `D0`. -/
def pb (a : ℕ) : DPair :=
  match a with
  | 0 => ⟨1, 0, ⟨2, true, 1⟩, ⟨0, false, 1⟩, ⟨3, .ZP 0 0 1, 0, 0, [0], [0]⟩⟩
  | 1 => ⟨2, 0, ⟨8, true, 1⟩, ⟨7, false, 1⟩, ⟨9, .ZP 0 0 1, 0, 0, [0], [0]⟩⟩
  | _ => ⟨3, 0, ⟨10, true, 1⟩, ⟨14, false, 1⟩, ⟨15, .ZP 0 0 (-1), 0, 0, [0], [0]⟩⟩

theorem pairs0 : D0.pairs.zipIdx = [(pb 0, 0), (pb 1, 1), (pb 2, 2)] := rfl

theorem pairNodes0 (n : ℕ) : (D0.pairNodes n 0 (pb 0)).map (·.node (lab0 n)) =
    FamFca.pairL n 1 (a0 n) (a2 n) := by
  simp only [FamDiagram.pairNodes, FamFca.jk_eq, List.map_flatMap, List.map_map, FamFca.pairL]
  refine List.flatMap_congr fun m _ => List.map_congr_left fun j _ => ?_
  simp [DNode.node, lab0_pid, labP, ixPh0, ixDimP, pb]

theorem pairNodes1 (n : ℕ) : (D0.pairNodes n 1 (pb 1)).map (·.node (lab0 n)) =
    FamFca.pairL n 1 (a7 n) (a8 n) := by
  simp only [FamDiagram.pairNodes, FamFca.jk_eq, List.map_flatMap, List.map_map, FamFca.pairL]
  refine List.flatMap_congr fun m _ => List.map_congr_left fun j _ => ?_
  simp [DNode.node, lab0_pid, labP, ixPh0, ixDimP, pb]

theorem pairNodes2 (n : ℕ) : (D0.pairNodes n 2 (pb 2)).map (·.node (lab0 n)) =
    FamFca.pairL n (-1) (a14 n) (a10 n) := by
  simp only [FamDiagram.pairNodes, FamFca.jk_eq, List.map_flatMap, List.map_map, FamFca.pairL]
  refine List.flatMap_congr fun m _ => List.map_congr_left fun j _ => ?_
  simp [DNode.node, lab0_pid, labP, ixPh0, ixDimP, pb]

theorem engL_nodes (n : ℕ) : (engL n).nodes =
    (range n).flatMap (chainN n) ++ (range n).map (engNode n 0) ++ (range n).map (engNode n 2) ++
      (range n).map (engNode n 7) ++ (range n).map (engNode n 8) ++ (range n).map (engNode n 10) ++
      (range n).map (engNode n 14) ++ FamFca.pairL n 1 (a0 n) (a2 n) ++
      FamFca.pairL n 1 (a7 n) (a8 n) ++ FamFca.pairL n (-1) (a14 n) (a10 n) := rfl

theorem map_single (n : ℕ) (f : ℕ → Node) : (range n).map f = (range n).flatMap fun j => [f j] := by
  induction (range n) <;> simp_all

theorem chain_perm (n : ℕ) : ((range n).flatMap (chainN n)).Perm
    ((range n).map (engNode n 1) ++ (range n).map (engNode n 4) ++ (range n).map (engNode n 5) ++
      (range n).map (engNode n 6) ++ (range n).map (engNode n 11) ++ (range n).map (engNode n 12) ++
      (range n).map (engNode n 13)) := by
  rw [map_single n (engNode n 1), map_single n (engNode n 4), map_single n (engNode n 5),
    map_single n (engNode n 6), map_single n (engNode n 11), map_single n (engNode n 12),
    map_single n (engNode n 13)]
  refine List.Perm.symm ((((((((List.flatMap_append_perm _ _ _).append_right _).trans
    (List.flatMap_append_perm _ _ _)).append_right _ |>.trans (List.flatMap_append_perm _ _ _)
    |>.append_right _ |>.trans (List.flatMap_append_perm _ _ _)).append_right _).trans
    (List.flatMap_append_perm _ _ _)).append_right _ |>.trans (List.flatMap_append_perm _ _ _)).trans ?_)
  rfl

theorem nodes0 (n : ℕ) : ((D0.famExpand n).nodes.map (·.node (lab0 n))).Perm (engL n).nodes := by
  have hX : ((D0.X n).nodes.map (D0.node n)).map (·.node (lab0 n)) =
      (range n).map (engNode n 0) ++ (range n).map (engNode n 1) ++ (range n).map (engNode n 2) ++
      (range n).map (engNode n 4) ++ (range n).map (engNode n 5) ++ (range n).map (engNode n 6) ++
      (range n).map (engNode n 7) ++ (range n).map (engNode n 8) ++ (range n).map (engNode n 10) ++
      (range n).map (engNode n 11) ++ (range n).map (engNode n 12) ++ (range n).map (engNode n 13) ++
      (range n).map (engNode n 14) := by
    rw [X0, copyNodes0_nodes, List.map_map, List.map_flatMap]
    rw [List.flatMap_congr (g := fun x => (range n).map (engNode n x.id)) fun x hx => by
      rw [List.map_map]; exact List.map_congr_left fun j hj => node0 hx (List.mem_range.1 hj)]
    simp [D0, List.append_assoc]
  have hP : D0.pairs.zipIdx.flatMap (fun q => D0.pairNodes n q.2 q.1) =
      D0.pairNodes n 0 (pb 0) ++ D0.pairNodes n 1 (pb 1) ++ D0.pairNodes n 2 (pb 2) := by
    rw [pairs0]; simp
  simp only [FamDiagram.famExpand, List.map_append]
  rw [hX, hP, List.map_append, List.map_append, pairNodes0, pairNodes1, pairNodes2, engL_nodes]
  refine List.Perm.trans ?_ ((((((((((chain_perm n).append_right _).append_right _).append_right _
    ).append_right _).append_right _).append_right _).append_right _).append_right _
    ).append_right _).symm
  rw [List.perm_iff_count]
  intro a
  simp only [List.count_append]
  omega

end FamDraper

end ArchytasZX
