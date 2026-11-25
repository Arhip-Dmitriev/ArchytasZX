/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamFcaExpand
import ArchytasZX.ChainForm
import ArchytasZX.PortBox

/-!
# The Fourier constant adder spec's expansion

`D1`: the engine's binder / shift / splitter spec as a `FamDiagram`: node boxes `b1` over the
binder `1` and `c1` over the splitter `3`, chain links through each.

`expandK1`: its expansion in key order at `n` copies is `copyNodes c1` after `copyNodes b1`, with
copy `j` of the binder at `5 + j` and of the splitter at `n + 5 + j`; `E1n`, `E2n` give both
stages in closed form.
-/

namespace ArchytasZX

namespace FamFca

open List

/-- The engine's binder / shift / splitter spec. -/
def D1 : FamDiagram :=
  { base :=
      { nodes := [⟨0, .Z, 0, 0, [0], []⟩, ⟨1, .B, 0, 0, [0], [0, 0]⟩, ⟨2, .X, 0, 0, [0], [0]⟩,
          ⟨3, .S, 0, 0, [0, 0], [0]⟩, ⟨4, .Z, 0, 0, [], [0]⟩],
        wires := [],
        outs := [⟨3, true, 1⟩], ins := [⟨1, false, 1⟩],
        boxes := [⟨0, 0, none, [1], []⟩, ⟨1, 0, none, [3], []⟩] },
    links := [⟨⟨1, true, 0⟩, ⟨1, false, 0⟩, ⟨2, false, 0⟩, ⟨0, true, 0⟩⟩,
              ⟨⟨3, false, 0⟩, ⟨3, true, 0⟩, ⟨2, true, 0⟩, ⟨4, false, 0⟩⟩],
    pairs := [],
    ph := [(2, fun n _ => n + 1)],
    dim := [(0, fun n _ => n), (1, fun n j => if j = 0 then n else j - 1),
            (2, fun n _ => if n = 0 then n else n - 1), (3, fun n j => if j = 0 then n else j - 1),
            (4, fun n _ => n)],
    legs := [(⟨0, true, 0⟩, fun n _ => n), (⟨1, true, 0⟩, fun n j => if j + 1 = 0 then n else j + 1 - 1),
             (⟨1, false, 0⟩, fun n j => if j = 0 then n else j - 1),
             (⟨2, true, 0⟩, fun n _ => if n = 0 then n else n - 1),
             (⟨2, false, 0⟩, fun n _ => if n = 0 then n else n - 1),
             (⟨3, true, 0⟩, fun n j => if j = 0 then n else j - 1),
             (⟨3, false, 0⟩, fun n j => if j + 1 = 0 then n else j + 1 - 1), (⟨4, false, 0⟩, fun n _ => n)] }

/-- The binder box. -/
def b1 : DBox := ⟨0, 0, none, [1], []⟩

/-- The splitter box. -/
def c1 : DBox := ⟨1, 0, none, [3], []⟩

/-- The template nodes. -/
def n0 : DNode := ⟨0, .Z, 0, 0, [0], []⟩
/-- The template binder. -/
def n1 : DNode := ⟨1, .B, 0, 0, [0], [0, 0]⟩
/-- The template shift. -/
def n2 : DNode := ⟨2, .X, 0, 0, [0], [0]⟩
/-- The template splitter. -/
def n3 : DNode := ⟨3, .S, 0, 0, [0, 0], [0]⟩
/-- The template end. -/
def n4 : DNode := ⟨4, .Z, 0, 0, [], [0]⟩

theorem nodes1 : D1.base.nodes = [n0, n1, n2, n3, n4] := rfl

theorem expand1 (n : ℕ) :
    D1.base.expand (fun _ => n) = (D1.base.copyNodes b1 n).copyNodes c1 n :=
  Diagram.expand_two _ _ b1 c1 rfl (by decide) rfl rfl rfl rfl (by decide) (by decide)

theorem growSide_false (k x : ℕ) (out : Bool) (ds : List ℕ) :
    Diagram.growSide (fun _ => false) k x out ds = ds :=
  Diagram.growSide_off _ k x out ds fun _ => rfl

/-- The first stage: the binder copied. -/
def E1n (n : ℕ) : Diagram :=
  { nodes := [n0] ++ (range n).map (fun j => { n1 with id := 5 + j }) ++ [n2, n3, n4],
    wires := [], outs := [⟨3, true, 1⟩], ins := (range n).map fun j => ⟨5 + j, false, 1⟩,
    boxes := [c1] }

theorem crossP_nil (D : Diagram) (hw : D.wires = []) (b : DBox) (p : Port) : D.crossP b p = false := by
  simp [Diagram.crossP, hw]

theorem cpId_b1 (j x : ℕ) : D1.base.cpId b1 j x = 5 + j + [1].idxOf x := by
  simp only [Diagram.cpId, show D1.base.fresh = 5 by decide, show b1.nodes.length = 1 from rfl,
    show b1.nodes.mergeSort (· ≤ ·) = [1] from List.mergeSort_eq_self (r := (· ≤ ·)) (by decide)]
  omega

theorem shiftPos_false (k : ℕ) (p : Port) : Diagram.shiftPos (fun _ => false) k p = p.pos := by
  simp [Diagram.shiftPos]

theorem growImg_false (k : ℕ) (p : Port) : Diagram.growImg (fun _ => false) k p = [p] := by
  simp [Diagram.growImg, shiftPos_false]

theorem stage1 (n : ℕ) : D1.base.copyNodes b1 n = E1n n := by
  have hc : D1.base.crossP b1 = fun _ => false := funext (crossP_nil _ rfl b1)
  have hB := Diagram.copyNodes_boxes_two D1.base b1 c1 rfl (by decide) (by decide) (by decide) rfl n
  have h0 : ¬ n0.id ∈ b1.nodes := by decide
  have h1 : n1.id ∈ b1.nodes := by decide
  have h2 : ¬ n2.id ∈ b1.nodes := by decide
  have h3 : ¬ n3.id ∈ b1.nodes := by decide
  have h4 : ¬ n4.id ∈ b1.nodes := by decide
  have hi : [1].idxOf n1.id = 0 := rfl
  have hO : D1.base.outs = [⟨3, true, 1⟩] := rfl
  have hI : D1.base.ins = [⟨1, false, 1⟩] := rfl
  have s3 : Diagram.inS b1 ⟨3, true, 1⟩ = false := by decide
  have s1 : Diagram.inS b1 ⟨1, false, 1⟩ = true := by decide
  unfold E1n
  rw [← hB]
  simp only [Diagram.copyNodes, Diagram.mk.injEq]
  refine ⟨?_, rfl, ?_, ?_, trivial⟩
  · simp only [hc, growSide_false, nodes1, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      h0, h1, h2, h3, h4, ite_true, ite_false, cpId_b1, hi, add_zero]
    rfl
  · simp only [Diagram.spliceN, hO, List.filter_cons, s3, List.filter_nil,
      List.flatMap_cons, List.flatMap_nil, List.append_nil, Diagram.nimg, hc, growImg_false]
    simp
  · simp only [Diagram.spliceN, hI, List.filter_cons, s1, List.filter_nil, ite_true,
      List.findIdx_cons, List.take_zero, List.drop_zero, List.flatMap_nil, List.nil_append,
      Bool.not_true, Bool.false_eq_true, ite_false, List.append_nil, List.map_cons, List.map_nil,
      cpId_b1, reduceCtorEq]
    simp [flatMap_single]

/-- The second stage: the splitter copied. -/
def E2n (n : ℕ) : Diagram :=
  { nodes := [n0] ++ (range n).map (fun j => { n1 with id := 5 + j }) ++ [n2] ++
      (range n).map (fun j => { n3 with id := n + 5 + j }) ++ [n4],
    wires := [], outs := (range n).map (fun j => ⟨n + 5 + j, true, 1⟩),
    ins := (range n).map fun j => ⟨5 + j, false, 1⟩, boxes := [] }

theorem foldr_max_le {l : List ℕ} {m : ℕ} (h : ∀ x ∈ l, x ≤ m) : l.foldr max 0 ≤ m := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.foldr_cons]
    exact max_le (h a List.mem_cons_self) (ih fun x hx => h x (List.mem_cons_of_mem _ hx))

theorem fresh_E1 (n : ℕ) : (E1n n).fresh = n + 5 := by
  have hmem : n + 4 ∈ (E1n n).nodes.map (·.id) := by
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · decide
    · simp only [E1n, List.map_append, List.map_cons, List.map_nil, List.map_map, List.mem_append,
        List.mem_map, List.mem_range, List.mem_cons]
      exact Or.inl (Or.inr ⟨n - 1, by omega, by simp; omega⟩)
  have hle : ∀ x ∈ (E1n n).nodes.map (·.id), x ≤ n + 4 := by
    intro x hx
    simp only [E1n, List.map_append, List.map_cons, List.map_nil, List.map_map, List.mem_append,
      List.mem_map, List.mem_range, List.mem_cons, List.not_mem_nil, or_false, Function.comp_def,
      n0, n1, n2, n3, n4] at hx
    rcases hx with (rfl | ⟨j, hj, rfl⟩) | rfl | rfl | rfl <;> omega
  have := Diagram.lt_fresh _ hmem
  have h2 := foldr_max_le hle
  unfold Diagram.fresh at this ⊢
  omega

theorem cpId_c1 (n j x : ℕ) : (E1n n).cpId c1 j x = n + 5 + j + [3].idxOf x := by
  simp only [Diagram.cpId, fresh_E1, show c1.nodes.length = 1 from rfl,
    show c1.nodes.mergeSort (· ≤ ·) = [3] from List.mergeSort_eq_self (r := (· ≤ ·)) (by decide)]
  omega

theorem stage2 (n : ℕ) : (E1n n).copyNodes c1 n = E2n n := by
  have hc : (E1n n).crossP c1 = fun _ => false := funext (crossP_nil _ rfl c1)
  have h0 : ¬ n0.id ∈ c1.nodes := by decide
  have h1 : ∀ j, ¬ ({ n1 with id := 5 + j } : DNode).id ∈ c1.nodes := fun j => by
    simp [c1]; omega
  have h2 : ¬ n2.id ∈ c1.nodes := by decide
  have h3 : n3.id ∈ c1.nodes := by decide
  have h4 : ¬ n4.id ∈ c1.nodes := by decide
  have hi : [3].idxOf n3.id = 0 := rfl
  have s3 : Diagram.inS c1 ⟨3, true, 1⟩ = true := by decide
  have s5 : ∀ j, Diagram.inS c1 ⟨5 + j, false, 1⟩ = false := fun j => by
    simp [Diagram.inS, c1]; omega
  have hcp : (E1n n).cpId c1 = fun j x => n + 5 + j + [3].idxOf x := funext fun j => funext (cpId_c1 n j)
  simp only [Diagram.copyNodes, E2n, Diagram.mk.injEq]
  refine ⟨?_, rfl, ?_, ?_, ?_⟩
  · simp only [hc, growSide_false, hcp]
    simp only [E1n, List.flatMap_append, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, h0, h2, h3, h4, ite_true, ite_false, hi, add_zero, List.flatMap_map]
    simp only [h1, ite_false, flatMap_single]
    simp [List.append_assoc]
  · simp only [Diagram.spliceN, hcp]
    simp only [E1n, List.filter_cons, s3, List.filter_nil, ite_true,
      List.findIdx_cons, List.take_zero, List.drop_zero, List.flatMap_nil, List.nil_append,
      Bool.not_true, Bool.false_eq_true, ite_false, List.append_nil, List.map_cons, List.map_nil,
      reduceCtorEq]
    simp [flatMap_single]
  · have hf : List.filter (Diagram.inS c1) ((range n).map fun j => (⟨5 + j, false, 1⟩ : Port)) = [] := by
      rw [List.filter_eq_nil_iff]; intro p hp
      obtain ⟨j, -, rfl⟩ := List.mem_map.1 hp
      simp [s5 j]
    have hn : ∀ j, (E1n n).nimg c1 n ⟨5 + j, false, 1⟩ = [⟨5 + j, false, 1⟩] := fun j => by
      simp only [Diagram.nimg, s5 j, hc, growImg_false, Bool.false_eq_true, ite_false]
    simp only [Diagram.spliceN]
    simp only [E1n] at hn ⊢
    simp only [hn, hf, ite_true, List.flatMap_map, flatMap_single]
  · simp [E1n, c1]

/-! ## The boundary in key order -/

theorem keySig1 : D1.base.keySig = [([], [0]), ([], [0])] := by decide

theorem legChain1 : D1.base.bnd.map D1.base.legChain = [[0], [0]] := by decide

theorem SDom1 (n : ℕ) : D1.base.SDom (fun _ => n) =
    (range 2).flatMap fun i => (range n).map fun t => (i, [t]) := by
  unfold Diagram.SDom
  rw [show D1.base.bnd.length = 2 from rfl]
  refine List.flatMap_congr fun i hi => ?_
  have hc : D1.base.legChain (D1.base.bnd.getD i Port.dflt) = [0] := by
    simp only [List.mem_range] at hi
    interval_cases i <;> decide
  rw [hc, cidx0, List.map_map]
  rfl

theorem key1 (i t : ℕ) (hi : i < 2) :
    Diagram.keyOf (D1.base.keySig.getD i ([], [])) i [t] = [i, t] := by
  rw [keySig1]
  interval_cases i <;> rfl

theorem SL1 (n : ℕ) : D1.base.SL (fun _ => n) =
    (range 2).flatMap fun i => (range n).map fun t => (i, [t]) := by
  unfold Diagram.SL
  rw [SDom1]
  refine List.Pairwise.insertionSort_eq ?_
  rw [List.pairwise_flatMap]
  refine ⟨fun i hi => ?_, ?_⟩
  · rw [List.pairwise_map]
    refine (List.pairwise_lt_range (n := n)).imp_of_mem fun {a b} _ _ hab => ?_
    simp only [List.mem_range] at hi
    rw [key1 i a hi, key1 i b hi]
    simp [Diagram.lexLe, hab]
  · refine (List.pairwise_lt_range (n := 2)).imp_of_mem fun {i i'} hi hi' hii => ?_
    simp only [List.mem_range] at hi hi'
    intro x hx y hy
    obtain ⟨a, -, rfl⟩ := List.mem_map.1 hx
    obtain ⟨b, -, rfl⟩ := List.mem_map.1 hy
    rw [key1 i a hi, key1 i' b hi']
    simp [Diagram.lexLe, hii]

theorem bndE1 (n : ℕ) : (E1n n).bnd = ⟨3, true, 1⟩ :: (range n).map fun j => ⟨5 + j, false, 1⟩ := rfl

theorem bndE2 (n : ℕ) : (E2n n).bnd =
    (range n).map (fun j => (⟨n + 5 + j, true, 1⟩ : Port)) ++ (range n).map fun j => ⟨5 + j, false, 1⟩ := rfl

theorem idx_shift5 {n t : ℕ} (ht : t < n) :
    ((range n).map fun j => (⟨5 + j, false, 1⟩ : Port)).idxOf ⟨5 + t, false, 1⟩ = t :=
  idxOf_map_range (f := fun j => (⟨5 + j, false, 1⟩ : Port))
    (fun a b e => by simp only [Port.mk.injEq] at e; omega) ht

theorem bidx1 (n i t : ℕ) (hi : i < 2) (ht : t < n) :
    D1.base.bidx (fun _ => n) (i, [t]) = (i * n + t, []) := by
  unfold Diagram.bidx
  have hB1 := stage1 n
  have hB2 := stage2 n
  have hp1 : D1.base.pick = some b1 := by decide
  have hp2 : (E1n n).pick = some c1 := rfl
  have hp3 : (E2n n).pick = none := pick_nil _ rfl
  have hc1 : D1.base.crossP b1 = fun _ => false := funext (crossP_nil _ rfl b1)
  have hc2 : (E1n n).crossP c1 = fun _ => false := funext (crossP_nil _ rfl c1)
  simp only [show D1.base.boxes.length = 2 from rfl, Diagram.bidxFuel, hp1, Diagram.step,
    show b1.ports = [] from rfl, ite_true, Diagram.expandNode_none _ b1 rfl, hB1, hp2,
    show c1.ports = [] from rfl, Diagram.expandNode_none _ c1 rfl, hB2, hp3]
  simp only [Diagram.stepIx, Diagram.gather_none _ b1 rfl, Diagram.gather_none _ c1 rfl, ite_self,
    show b1.ports = [] from rfl, show c1.ports = [] from rfl, ite_true, hB1, hB2]
  interval_cases i
  · simp only [Diagram.ixMap]
    have e0 : D1.base.bnd.getD 0 Port.dflt = ⟨3, true, 1⟩ := rfl
    have g0 : D1.base.gN b1 ⟨3, true, 1⟩ = false := by simp [Diagram.gN, hc1]; decide
    have m0 : D1.base.nimg b1 n ⟨3, true, 1⟩ = [⟨3, true, 1⟩] := by
      simp only [Diagram.nimg, hc1, growImg_false, show Diagram.inS b1 ⟨3, true, 1⟩ = false by decide,
        Bool.false_eq_true, ite_false]
    have g1 : (E1n n).gN c1 ⟨3, true, 1⟩ = true := by
      simp [Diagram.gN, show Diagram.inS c1 ⟨3, true, 1⟩ = true by decide]
    have m1 : (E1n n).nimg c1 n ⟨3, true, 1⟩ = (range n).map fun j => ⟨n + 5 + j, true, 1⟩ := by
      simp only [Diagram.nimg, show Diagram.inS c1 ⟨3, true, 1⟩ = true by decide, ite_true, cpId_c1]
      rfl
    simp only [e0, show D1.base.bnd.idxOf ⟨3, true, 1⟩ = 0 from rfl, φS, g0, m0,
      Bool.false_eq_true, ite_false, List.headD_cons, bndE1, List.idxOf_cons_self, List.getD_cons_zero,
      g1, ite_true, m1, show [t].getLastD 0 = t from rfl, List.dropLast_singleton]
    rw [List.getD_eq_getElem _ _ (by simpa using ht)]
    simp only [List.getElem_map, List.getElem_range, bndE2]
    have hm : (⟨n + 5 + t, true, 1⟩ : Port) ∈ (range n).map (fun j => (⟨n + 5 + j, true, 1⟩ : Port)) :=
      List.mem_map.2 ⟨t, List.mem_range.2 ht, rfl⟩
    rw [List.idxOf_append_of_mem hm, idxOf_map_range (f := fun j => (⟨n + 5 + j, true, 1⟩ : Port))
      (fun a b e => by simp only [Port.mk.injEq] at e; omega) ht]
    simp
  · have e1 : D1.base.bnd.getD 1 Port.dflt = ⟨1, false, 1⟩ := rfl
    have g0 : D1.base.gN b1 ⟨1, false, 1⟩ = true := by
      simp [Diagram.gN, show Diagram.inS b1 ⟨1, false, 1⟩ = true by decide]
    have m0 : D1.base.nimg b1 n ⟨1, false, 1⟩ = (range n).map fun j => ⟨5 + j, false, 1⟩ := by
      simp only [Diagram.nimg, show Diagram.inS b1 ⟨1, false, 1⟩ = true by decide, ite_true, cpId_b1]
      rfl
    have hI : (E1n n).bnd.idxOf ⟨5 + t, false, 1⟩ = 1 + t := by
      rw [bndE1, List.idxOf_cons_ne _ (by simp), idx_shift5 ht]; omega
    have hG : (E1n n).bnd.getD (1 + t) Port.dflt = ⟨5 + t, false, 1⟩ := by
      rw [bndE1, Nat.add_comm, List.getD_cons_succ, List.getD_eq_getElem _ _ (by simpa using ht)]
      simp
    have s5 : Diagram.inS c1 ⟨5 + t, false, 1⟩ = false := by
      simp only [Diagram.inS, c1, List.mem_singleton, decide_eq_false_iff_not]; omega
    have g1 : (E1n n).gN c1 ⟨5 + t, false, 1⟩ = false := by
      simp only [Diagram.gN, hc2, s5, Bool.or_false]
    have m1 : (E1n n).nimg c1 n ⟨5 + t, false, 1⟩ = [⟨5 + t, false, 1⟩] := by
      simp only [Diagram.nimg, hc2, growImg_false, s5, Bool.false_eq_true, ite_false]
    have hS1 : ArchytasZX.Diagram.ixMap D1.base (E1n n) (φS (D1.base.nimg b1 n) (D1.base.gN b1))
        (ArchytasZX.Diagram.ixMap D1.base D1.base (fun p t => (p, t)) (1, [t])) = (1 + t, []) := by
      simp only [Diagram.ixMap, e1, show D1.base.bnd.idxOf ⟨1, false, 1⟩ = 1 from rfl, φS, g0, m0,
        ite_true, show [t].getLastD 0 = t from rfl, List.dropLast_singleton]
      rw [List.getD_eq_getElem _ _ (by simpa using ht)]
      simp only [List.getElem_map, List.getElem_range, hI]
    have hS2 : ArchytasZX.Diagram.ixMap (E1n n) (E2n n) (φS ((E1n n).nimg c1 n) ((E1n n).gN c1))
        (ArchytasZX.Diagram.ixMap (E1n n) (E1n n) (fun p t => (p, t)) (1 + t, [])) = (n + t, []) := by
      simp only [Diagram.ixMap, hG, hI, φS, g1, m1, Bool.false_eq_true, ite_false, List.headD_cons,
        bndE2]
      rw [List.idxOf_append_of_notMem (fun hm => by
          obtain ⟨a, -, e⟩ := List.mem_map.1 hm
          simp at e), idx_shift5 ht]
      simp
    rw [hS1, hS2]
    simp

/-- The expansion in key order is `E2n n`. -/
theorem expandK1 (n : ℕ) : D1.base.expandK (fun _ => n) = E2n n := by
  unfold Diagram.expandK
  rw [expand1, stage1, stage2, SL1]
  set E := E2n n
  have hO : E.outs = (range n).map (fun j => (⟨n + 5 + j, true, 1⟩ : Port)) := rfl
  have hI : E.ins = (range n).map (fun j => (⟨5 + j, false, 1⟩ : Port)) := rfl
  have hL : ((range 2).flatMap fun i => (range n).map fun t => (i, [t])).map
      (fun x => E.bnd.getD (D1.base.bidx (fun _ => n) x).1 Port.dflt) = E.outs ++ E.ins := by
    rw [show range 2 = [0, 1] from rfl]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.map_append, List.map_map]
    congr 1
    · rw [hO]
      refine List.map_congr_left fun t ht => ?_
      have ht' := List.mem_range.1 ht
      simp only [Function.comp, bidx1 n 0 t (by omega) ht', Diagram.bnd, hO, zero_mul, zero_add]
      rw [List.getD_eq_getElem _ _ (by simp; omega), List.getElem_append_left (by simpa using ht')]
      simp
    · rw [hI]
      refine List.map_congr_left fun t ht => ?_
      have ht' := List.mem_range.1 ht
      simp only [Function.comp, bidx1 n 1 t (by omega) ht', Diagram.bnd, hO, hI, one_mul]
      rw [List.getD_eq_getElem _ _ (by simp; omega), List.getElem_append_right (by simp)]
      simp
  dsimp only
  rw [hL]
  simp [E, E2n]

/-! ## Template node and copy -/

theorem X1 (n : ℕ) : D1.X n = E2n n := expandK1 n

/-- The nodes of `D1.X n` with their template node and copy. -/
def Z1 (n : ℕ) : List (DNode × ℕ × ℕ) :=
  [(n0, (0, 0))] ++ (range n).map (fun j => ({ n1 with id := 5 + j }, (1, j))) ++ [(n2, (2, 0))] ++
    (range n).map (fun j => ({ n3 with id := n + 5 + j }, (3, j))) ++ [(n4, (4, 0))]

theorem org1 (n : ℕ) : D1.org n =
    [(0, 0)] ++ (range n).map (1, ·) ++ [(2, 0)] ++ (range n).map (3, ·) ++ [(4, 0)] := by
  simp [FamDiagram.org, D1]

theorem zip_map_range {α : Type} (n : ℕ) (f : ℕ → α) (g : ℕ → ℕ × ℕ) :
    ((range n).map f).zip ((range n).map g) = (range n).map fun j => (f j, g j) := by
  rw [List.zip_map, zip_range_self, List.map_map]; rfl

theorem zip1 (n : ℕ) : (D1.X n).nodes.zip (D1.org n) = Z1 n := by
  rw [X1, org1]
  simp only [E2n, Z1, List.append_assoc, List.cons_append, List.nil_append, List.zip_cons_cons]
  rw [List.zip_append (by simp), zip_map_range, List.zip_cons_cons,
    List.zip_append (by simp), zip_map_range]
  rfl

theorem mem_Z1 {n : ℕ} {a : DNode × ℕ × ℕ} (h : a ∈ Z1 n) :
    a = (n0, (0, 0)) ∨ (∃ j < n, a = ({ n1 with id := 5 + j }, (1, j))) ∨ a = (n2, (2, 0)) ∨
      (∃ j < n, a = ({ n3 with id := n + 5 + j }, (3, j))) ∨ a = (n4, (4, 0)) := by
  simp only [Z1, List.mem_append, List.mem_singleton, List.mem_map, List.mem_range] at h
  rcases h with (((h | ⟨j, hj, rfl⟩) | h) | ⟨j, hj, rfl⟩) | h
  · exact Or.inl h
  · exact Or.inr (Or.inl ⟨j, hj, rfl⟩)
  · exact Or.inr (Or.inr (Or.inl h))
  · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨j, hj, rfl⟩)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr h)))

theorem orgOf1_eq {n y : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) (hy : e.1.id = y) :
    D1.orgOf n y = e.2 := by
  unfold FamDiagram.orgOf
  rw [zip1, find?_unique (e := e) he (by simpa using hy)]
  · rfl
  · intro a ha hp
    simp only [decide_eq_true_eq] at hp
    rcases mem_Z1 ha with rfl | ⟨j, hj, rfl⟩ | rfl | ⟨j, hj, rfl⟩ | rfl <;>
    rcases mem_Z1 he with rfl | ⟨k, hk, rfl⟩ | rfl | ⟨k, hk, rfl⟩ | rfl <;>
    simp only [n0, n1, n2, n3, n4] at hp hy ⊢ <;> first | omega | (subst hy; simp_all)

theorem cp1_eq {n : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) : D1.cp n e.2.1 e.2.2 = e.1.id := by
  unfold FamDiagram.cp
  rw [zip1, find?_unique (e := e) he (by simp)]
  · rfl
  · intro a ha hp
    simp only [decide_eq_true_eq] at hp
    rcases mem_Z1 ha with rfl | ⟨j, hj, rfl⟩ | rfl | ⟨j, hj, rfl⟩ | rfl <;>
    rcases mem_Z1 he with rfl | ⟨k, hk, rfl⟩ | rfl | ⟨k, hk, rfl⟩ | rfl <;>
    simp_all

end FamFca

end ArchytasZX
