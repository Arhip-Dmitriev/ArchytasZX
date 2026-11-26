/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamDraperExpand
import ArchytasZX.FamFcaSpec

/-!
# The Draper adder spec's expansion

`D1`: the engine's binder / copy / sum / splitter spec as a `FamDiagram`: node boxes `b1` over the
binders `1, 5` and `c1` over the splitters `2, 6`, chain links through each.

`expandK1`: its expansion in key order at `n` copies is `copyNodes c1` after `copyNodes b1`, with
copy `j` of the binders at `10 + 2 j`, `11 + 2 j` and of the splitters at `10 + 2 n + 2 j`,
`11 + 2 n + 2 j`; `E1n`, `E2n` give both stages in closed form.
-/

namespace ArchytasZX

namespace FamDraper

open List

/-- The engine's binder / copy / sum / splitter spec. -/
def D1 : FamDiagram :=
  { base :=
      { nodes := [⟨0, .Z, 0, 0, [0], []⟩, ⟨1, .B, 0, 0, [0], [0, 0]⟩, ⟨2, .S, 0, 0, [0, 0], [0]⟩,
          ⟨3, .Z, 0, 0, [], [0]⟩, ⟨4, .Z, 0, 0, [0], []⟩, ⟨5, .B, 0, 0, [0], [0, 0]⟩,
          ⟨6, .S, 0, 0, [0, 0], [0]⟩, ⟨7, .Z, 0, 0, [], [0]⟩, ⟨8, .Z, 0, 0, [0, 0], [0]⟩,
          ⟨9, .X, 0, 0, [0], [0, 0]⟩],
        wires := [(⟨8, true, 1⟩, ⟨9, false, 1⟩)],
        outs := [⟨2, true, 1⟩, ⟨6, true, 1⟩], ins := [⟨1, false, 1⟩, ⟨5, false, 1⟩],
        boxes := [⟨0, 0, none, [1, 5], []⟩, ⟨1, 0, none, [2, 6], []⟩] },
    links := [⟨⟨1, true, 0⟩, ⟨1, false, 0⟩, ⟨8, false, 0⟩, ⟨0, true, 0⟩⟩,
              ⟨⟨5, true, 0⟩, ⟨5, false, 0⟩, ⟨9, false, 0⟩, ⟨4, true, 0⟩⟩,
              ⟨⟨2, false, 0⟩, ⟨2, true, 0⟩, ⟨8, true, 0⟩, ⟨3, false, 0⟩⟩,
              ⟨⟨6, false, 0⟩, ⟨6, true, 0⟩, ⟨9, true, 0⟩, ⟨7, false, 0⟩⟩],
    pairs := [],
    ph := [],
    dim := [(0, fun n _ => n), (1, fun n j => if j = 0 then n else j - 1),
            (2, fun n j => if j = 0 then n else j - 1), (3, fun n _ => n), (4, fun n _ => n),
            (5, fun n j => if j = 0 then n else j - 1), (6, fun n j => if j = 0 then n else j - 1),
            (7, fun n _ => n), (8, fun n _ => if n = 0 then n else n - 1),
            (9, fun n _ => if n = 0 then n else n - 1)],
    legs := [(⟨0, true, 0⟩, fun n _ => n), (⟨1, true, 0⟩, fun n j => if j + 1 = 0 then n else j + 1 - 1),
             (⟨1, false, 0⟩, fun n j => if j = 0 then n else j - 1),
             (⟨2, true, 0⟩, fun n j => if j = 0 then n else j - 1),
             (⟨2, false, 0⟩, fun n j => if j + 1 = 0 then n else j + 1 - 1), (⟨3, false, 0⟩, fun n _ => n),
             (⟨4, true, 0⟩, fun n _ => n), (⟨5, true, 0⟩, fun n j => if j + 1 = 0 then n else j + 1 - 1),
             (⟨5, false, 0⟩, fun n j => if j = 0 then n else j - 1),
             (⟨6, true, 0⟩, fun n j => if j = 0 then n else j - 1),
             (⟨6, false, 0⟩, fun n j => if j + 1 = 0 then n else j + 1 - 1), (⟨7, false, 0⟩, fun n _ => n),
             (⟨8, true, 0⟩, fun n _ => if n = 0 then n else n - 1),
             (⟨8, true, 1⟩, fun n _ => if n = 0 then n else n - 1),
             (⟨8, false, 0⟩, fun n _ => if n = 0 then n else n - 1),
             (⟨9, true, 0⟩, fun n _ => if n = 0 then n else n - 1),
             (⟨9, false, 0⟩, fun n _ => if n = 0 then n else n - 1),
             (⟨9, false, 1⟩, fun n _ => if n = 0 then n else n - 1)] }

/-- The binder box. -/
def b1 : DBox := ⟨0, 0, none, [1, 5], []⟩

/-- The splitter box. -/
def c1 : DBox := ⟨1, 0, none, [2, 6], []⟩

/-- The template head of chain `a`. -/
def m0 : DNode := ⟨0, .Z, 0, 0, [0], []⟩
/-- The template binder of chain `a`. -/
def m1 : DNode := ⟨1, .B, 0, 0, [0], [0, 0]⟩
/-- The template splitter of chain `a`. -/
def m2 : DNode := ⟨2, .S, 0, 0, [0, 0], [0]⟩
/-- The template end of chain `a`. -/
def m3 : DNode := ⟨3, .Z, 0, 0, [], [0]⟩
/-- The template head of chain `b`. -/
def m4 : DNode := ⟨4, .Z, 0, 0, [0], []⟩
/-- The template binder of chain `b`. -/
def m5 : DNode := ⟨5, .B, 0, 0, [0], [0, 0]⟩
/-- The template splitter of chain `b`. -/
def m6 : DNode := ⟨6, .S, 0, 0, [0, 0], [0]⟩
/-- The template end of chain `b`. -/
def m7 : DNode := ⟨7, .Z, 0, 0, [], [0]⟩
/-- The template copy spider. -/
def m8 : DNode := ⟨8, .Z, 0, 0, [0, 0], [0]⟩
/-- The template sum spider. -/
def m9 : DNode := ⟨9, .X, 0, 0, [0], [0, 0]⟩

theorem nodes1 : D1.base.nodes = [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9] := rfl

theorem expand1 (n : ℕ) :
    D1.base.expand (fun _ => n) = (D1.base.copyNodes b1 n).copyNodes c1 n :=
  Diagram.expand_two _ _ b1 c1 rfl (by decide) rfl rfl rfl rfl (by decide) (by decide)

/-- The first stage: the binders copied. -/
def E1n (n : ℕ) : Diagram :=
  { nodes := [m0] ++ (range n).map (fun j => { m1 with id := 10 + 2 * j }) ++ [m2, m3, m4] ++
      (range n).map (fun j => { m5 with id := 11 + 2 * j }) ++ [m6, m7, m8, m9],
    wires := [(⟨8, true, 1⟩, ⟨9, false, 1⟩)], outs := [⟨2, true, 1⟩, ⟨6, true, 1⟩],
    ins := (range n).flatMap fun j => [⟨10 + 2 * j, false, 1⟩, ⟨11 + 2 * j, false, 1⟩],
    boxes := [c1] }

theorem cpId_b1 (j x : ℕ) : D1.base.cpId b1 j x = 10 + 2 * j + [1, 5].idxOf x := by
  simp only [Diagram.cpId, show D1.base.fresh = 10 by decide, show b1.nodes.length = 2 from rfl,
    show b1.nodes.mergeSort (· ≤ ·) = [1, 5] from List.mergeSort_eq_self (r := (· ≤ ·)) (by decide)]
  omega

theorem crossP_b1 (p : Port) : D1.base.crossP b1 p = false := by
  rcases p with ⟨a, o, q⟩
  simp only [Diagram.crossP, Diagram.inS, D1, b1, List.any_cons, List.any_nil, Bool.or_false]
  by_cases h : a ∈ [1, 5] <;> simp [h]

theorem crossP_c1 (n : ℕ) (p : Port) : (E1n n).crossP c1 p = false := by
  rcases p with ⟨a, o, q⟩
  simp only [Diagram.crossP, Diagram.inS, E1n, c1, List.any_cons, List.any_nil, Bool.or_false]
  by_cases h : a ∈ [2, 6] <;> simp [h]

theorem stage1 (n : ℕ) : D1.base.copyNodes b1 n = E1n n := by
  have hc : D1.base.crossP b1 = fun _ => false := funext crossP_b1
  have hB := Diagram.copyNodes_boxes_two D1.base b1 c1 rfl (by decide) (by decide) (by decide) rfl n
  have hO : D1.base.outs = [⟨2, true, 1⟩, ⟨6, true, 1⟩] := rfl
  have hI : D1.base.ins = [⟨1, false, 1⟩, ⟨5, false, 1⟩] := rfl
  have s2 : Diagram.inS b1 ⟨2, true, 1⟩ = false := by decide
  have s6 : Diagram.inS b1 ⟨6, true, 1⟩ = false := by decide
  have s1 : Diagram.inS b1 ⟨1, false, 1⟩ = true := by decide
  have s5 : Diagram.inS b1 ⟨5, false, 1⟩ = true := by decide
  unfold E1n
  rw [← hB]
  simp only [Diagram.copyNodes, Diagram.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, trivial⟩
  · simp only [hc, FamFca.growSide_false, nodes1, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, cpId_b1]
    simp [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, b1]
    intro a _; omega
  · simp only [Diagram.nimg, hc, FamFca.growImg_false]
    rw [show D1.base.wires = [(⟨8, true, 1⟩, ⟨9, false, 1⟩)] from rfl]
    simp [Diagram.inS, b1]
  · simp only [Diagram.spliceN, hO, List.filter_cons, s2, s6, List.filter_nil,
      List.flatMap_cons, List.flatMap_nil, List.append_nil, Diagram.nimg, hc, FamFca.growImg_false]
    simp
  · simp only [Diagram.spliceN, hI, List.filter_cons, s1, s5, ite_true,
      List.findIdx_cons, List.take_zero, List.drop_zero, List.flatMap_nil, List.nil_append,
      Bool.not_true, Bool.false_eq_true, ite_false, List.append_nil, List.map_cons, List.map_nil,
      cpId_b1, reduceCtorEq, List.filter_nil]
    simp only [List.idxOf_cons_self, List.idxOf_cons_ne _ (show (1 : ℕ) ≠ 5 by decide), add_zero]
    exact List.flatMap_congr fun j _ => by simp; omega

/-- The second stage: the splitters copied. -/
def E2n (n : ℕ) : Diagram :=
  { nodes := [m0] ++ (range n).map (fun j => { m1 with id := 10 + 2 * j }) ++
      (range n).map (fun j => { m2 with id := 10 + 2 * n + 2 * j }) ++ [m3, m4] ++
      (range n).map (fun j => { m5 with id := 11 + 2 * j }) ++
      (range n).map (fun j => { m6 with id := 11 + 2 * n + 2 * j }) ++ [m7, m8, m9],
    wires := [(⟨8, true, 1⟩, ⟨9, false, 1⟩)],
    outs := (range n).flatMap fun j => [⟨10 + 2 * n + 2 * j, true, 1⟩, ⟨11 + 2 * n + 2 * j, true, 1⟩],
    ins := (range n).flatMap fun j => [⟨10 + 2 * j, false, 1⟩, ⟨11 + 2 * j, false, 1⟩],
    boxes := [] }

theorem fresh_E1 (n : ℕ) : (E1n n).fresh = 2 * n + 10 := by
  have hmem : 2 * n + 9 ∈ (E1n n).nodes.map (·.id) := by
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · decide
    · simp only [E1n, List.map_append]
      refine List.mem_append_left _ (List.mem_append_right _ ?_)
      rw [List.map_map]
      exact List.mem_map.2 ⟨n - 1, List.mem_range.2 (by omega), by simp [m5]; omega⟩
  have hle : ∀ x ∈ (E1n n).nodes.map (·.id), x ≤ 2 * n + 9 := by
    intro x hx
    simp only [E1n, List.map_append, List.map_cons, List.map_nil, List.map_map, List.mem_append,
      List.mem_map, List.mem_range, List.mem_cons, List.not_mem_nil, or_false, Function.comp_def,
      m0, m1, m2, m3, m4, m5, m6, m7, m8, m9] at hx
    rcases hx with ((((rfl | ⟨j, hj, rfl⟩) | rfl | rfl | rfl) | ⟨j, hj, rfl⟩) | rfl | rfl | rfl | rfl) <;>
      omega
  have := Diagram.lt_fresh _ hmem
  have h2 := FamFca.foldr_max_le hle
  unfold Diagram.fresh at this ⊢
  omega

theorem cpId_c1 (n j x : ℕ) : (E1n n).cpId c1 j x = 2 * n + 10 + 2 * j + [2, 6].idxOf x := by
  simp only [Diagram.cpId, fresh_E1, show c1.nodes.length = 2 from rfl,
    show c1.nodes.mergeSort (· ≤ ·) = [2, 6] from List.mergeSort_eq_self (r := (· ≤ ·)) (by decide)]
  omega

theorem stage2 (n : ℕ) : (E1n n).copyNodes c1 n = E2n n := by
  have hc : (E1n n).crossP c1 = fun _ => false := funext (crossP_c1 n)
  have h1 : ∀ j, ¬ ({ m1 with id := 10 + 2 * j } : DNode).id ∈ c1.nodes := fun j => by
    simp [c1]; omega
  have h5 : ∀ j, ¬ ({ m5 with id := 11 + 2 * j } : DNode).id ∈ c1.nodes := fun j => by
    simp [c1]; omega
  have hcp : (E1n n).cpId c1 = fun j x => 2 * n + 10 + 2 * j + [2, 6].idxOf x :=
    funext fun j => funext (cpId_c1 n j)
  have s2 : Diagram.inS c1 ⟨2, true, 1⟩ = true := by decide
  have s6 : Diagram.inS c1 ⟨6, true, 1⟩ = true := by decide
  have sI : ∀ j, Diagram.inS c1 ⟨10 + 2 * j, false, 1⟩ = false ∧
      Diagram.inS c1 ⟨11 + 2 * j, false, 1⟩ = false := fun j => by
    simp only [Diagram.inS, c1, List.mem_cons, List.not_mem_nil, or_false, decide_eq_false_iff_not]
    omega
  simp only [Diagram.copyNodes, E2n, Diagram.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp only [hc, FamFca.growSide_false, hcp]
    simp only [E1n, List.flatMap_append, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, List.flatMap_map]
    simp only [h1, h5, ite_false, FamFca.flatMap_single]
    simp [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, c1, List.append_assoc]
    have e1 : ∀ j, 2 * n + 10 + 2 * j = 10 + 2 * n + 2 * j := fun j => by omega
    have e2 : ∀ j, 10 + 2 * n + 2 * j + 1 = 11 + 2 * n + 2 * j := fun j => by omega
    simp only [e1]
    simp only [e2]
  · simp only [Diagram.nimg, hc, FamFca.growImg_false]
    simp [E1n, Diagram.inS, c1]
  · simp only [Diagram.spliceN, hcp]
    simp only [E1n, List.filter_cons, s2, s6, List.filter_nil, ite_true,
      List.findIdx_cons, List.take_zero, List.drop_zero, List.flatMap_nil, List.nil_append,
      Bool.not_true, Bool.false_eq_true, ite_false, List.append_nil, List.map_cons, List.map_nil,
      reduceCtorEq]
    refine List.flatMap_congr fun j _ => ?_
    simp; omega
  · have hf : List.filter (Diagram.inS c1)
        ((range n).flatMap fun j => [(⟨10 + 2 * j, false, 1⟩ : Port), ⟨11 + 2 * j, false, 1⟩]) = [] := by
      rw [List.filter_eq_nil_iff]; intro p hp
      obtain ⟨j, -, hp⟩ := List.mem_flatMap.1 hp
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
      rcases hp with rfl | rfl <;> simp [(sI j).1, (sI j).2]
    have hn : ∀ p : Port, Diagram.inS c1 p = false → (E1n n).nimg c1 n p = [p] := fun p hp => by
      simp only [Diagram.nimg, hp, hc, FamFca.growImg_false, Bool.false_eq_true, ite_false]
    simp only [Diagram.spliceN]
    rw [show (E1n n).ins = (range n).flatMap fun j =>
      [(⟨10 + 2 * j, false, 1⟩ : Port), ⟨11 + 2 * j, false, 1⟩] from rfl, hf]
    simp only [ite_true, List.flatMap_assoc]
    refine List.flatMap_congr fun j _ => ?_
    simp [hn _ (sI j).1, hn _ (sI j).2]
  · simp [E1n, c1]

/-! ## The boundary in key order -/

theorem keySig1 : D1.base.keySig = D0.base.keySig := by decide

theorem SDom1 (n : ℕ) : D1.base.SDom (fun _ => n) = D0.base.SDom (fun _ => n) := by
  rw [SDom0]
  unfold Diagram.SDom
  rw [show D1.base.bnd.length = 4 from rfl]
  refine List.flatMap_congr fun i hi => ?_
  have hc : D1.base.legChain (D1.base.bnd.getD i Port.dflt) = [0] := by
    simp only [List.mem_range] at hi
    interval_cases i <;> decide
  rw [hc, FamFca.cidx0, List.map_map]
  rfl

theorem SL1 (n : ℕ) : D1.base.SL (fun _ => n) = (range (4 * n)).map (sl n) := by
  have h := SL0 n
  unfold Diagram.SL at h ⊢
  rw [SDom1, keySig1]
  exact h

/-- The outputs of the copies, copy by copy. -/
def BOn (n : ℕ) : List Port :=
  (range n).flatMap fun j => [⟨10 + 2 * n + 2 * j, true, 1⟩, ⟨11 + 2 * n + 2 * j, true, 1⟩]

/-- The inputs of the copies, copy by copy. -/
def BIn (n : ℕ) : List Port :=
  (range n).flatMap fun j => [⟨10 + 2 * j, false, 1⟩, ⟨11 + 2 * j, false, 1⟩]

theorem bndE1 (n : ℕ) : (E1n n).bnd = [⟨2, true, 1⟩, ⟨6, true, 1⟩] ++ BIn n := rfl

theorem bndE2 (n : ℕ) : (E2n n).bnd = BOn n ++ BIn n := rfl

theorem idxBIn {n t : ℕ} (ht : t < n) : (BIn n).idxOf ⟨10 + 2 * t, false, 1⟩ = 2 * t ∧
    (BIn n).idxOf ⟨11 + 2 * t, false, 1⟩ = 2 * t + 1 :=
  idx_flat2 (fun j => ⟨10 + 2 * j, false, 1⟩) (fun j => ⟨11 + 2 * j, false, 1⟩)
    (fun a b e => by simp only [Port.mk.injEq] at e; omega) (fun a b e => by simp only [Port.mk.injEq] at e; omega)
    (fun a b e => by simp only [Port.mk.injEq] at e; omega) ht

theorem idxBOn {n t : ℕ} (ht : t < n) : (BOn n).idxOf ⟨10 + 2 * n + 2 * t, true, 1⟩ = 2 * t ∧
    (BOn n).idxOf ⟨11 + 2 * n + 2 * t, true, 1⟩ = 2 * t + 1 :=
  idx_flat2 (fun j => ⟨10 + 2 * n + 2 * j, true, 1⟩) (fun j => ⟨11 + 2 * n + 2 * j, true, 1⟩)
    (fun a b e => by simp only [Port.mk.injEq] at e; omega) (fun a b e => by simp only [Port.mk.injEq] at e; omega)
    (fun a b e => by simp only [Port.mk.injEq] at e; omega) ht

theorem len_BOn (n : ℕ) : (BOn n).length = 2 * n := len_flat2 _ _ _

theorem notin_BOn {n : ℕ} (p : Port) (hp : p.out = false) : p ∉ BOn n := fun hm => by
  obtain ⟨a, -, hm⟩ := List.mem_flatMap.1 hm
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl | rfl <;> simp at hp

theorem φS_g {img : Port → List Port} {g : Port → Bool} {p : Port} (t : ℕ) (hg : g p = true) :
    φS img g p [t] = ((img p).getD t Port.dflt, []) := by
  simp [φS, hg]

theorem φS_n {img : Port → List Port} {g : Port → Bool} {p : Port} (ts : List ℕ) (hg : g p = false)
    (hi : img p = [p]) : φS img g p ts = (p, ts) := by
  simp [φS, hg, hi]

theorem bidx1 (n i t : ℕ) (hi : i < 4) (ht : t < n) :
    D1.base.bidx (fun _ => n) (i, [t]) = (bpos n i t, []) := by
  unfold Diagram.bidx
  have hB1 := stage1 n
  have hB2 := stage2 n
  have hp1 : D1.base.pick = some b1 := by decide
  have hp2 : (E1n n).pick = some c1 := rfl
  have hp3 : (E2n n).pick = none := FamFca.pick_nil _ rfl
  have hc1 : D1.base.crossP b1 = fun _ => false := funext crossP_b1
  have hc2 : (E1n n).crossP c1 = fun _ => false := funext (crossP_c1 n)
  simp only [show D1.base.boxes.length = 2 from rfl, Diagram.bidxFuel, hp1, Diagram.step,
    show b1.ports = [] from rfl, ite_true, Diagram.expandNode_none _ b1 rfl, hB1, hp2,
    show c1.ports = [] from rfl, Diagram.expandNode_none _ c1 rfl, hB2, hp3]
  simp only [Diagram.stepIx, Diagram.gather_none _ b1 rfl, Diagram.gather_none _ c1 rfl, ite_self,
    show b1.ports = [] from rfl, show c1.ports = [] from rfl, ite_true, hB1, hB2]
  have hn1 : ∀ p : Port, Diagram.inS b1 p = false → D1.base.nimg b1 n p = [p] := fun p hp => by
    simp only [Diagram.nimg, hp, hc1, FamFca.growImg_false, Bool.false_eq_true, ite_false]
  have hn2 : ∀ p : Port, Diagram.inS c1 p = false → (E1n n).nimg c1 n p = [p] := fun p hp => by
    simp only [Diagram.nimg, hp, hc2, FamFca.growImg_false, Bool.false_eq_true, ite_false]
  have hg1 : ∀ p : Port, D1.base.gN b1 p = Diagram.inS b1 p := fun p => by simp [Diagram.gN, hc1]
  have hg2 : ∀ p : Port, (E1n n).gN c1 p = Diagram.inS c1 p := fun p => by simp [Diagram.gN, hc2]
  have hm1 : ∀ x, Diagram.inS b1 ⟨x, false, 1⟩ = true →
      D1.base.nimg b1 n ⟨x, false, 1⟩ = (range n).map fun j => ⟨10 + 2 * j + [1, 5].idxOf x, false, 1⟩ :=
    fun x hx => by simp only [Diagram.nimg, hx, ite_true, cpId_b1]
  have hm2 : ∀ x, Diagram.inS c1 ⟨x, true, 1⟩ = true →
      (E1n n).nimg c1 n ⟨x, true, 1⟩ = (range n).map fun j => ⟨2 * n + 10 + 2 * j + [2, 6].idxOf x, true, 1⟩ :=
    fun x hx => by simp only [Diagram.nimg, hx, ite_true, cpId_c1]
  have hgt : ∀ (f : ℕ → Port), ((range n).map f).getD t Port.dflt = f t := fun f => by
    rw [List.getD_eq_getElem _ _ (by simpa using ht)]; simp
  have hBI : ∀ p ∈ BIn n, Diagram.inS c1 p = false := fun p hp => by
    obtain ⟨j, -, hp⟩ := List.mem_flatMap.1 hp
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> simp only [Diagram.inS, c1, List.mem_cons, List.not_mem_nil,
      or_false, decide_eq_false_iff_not] <;> omega
  have hb : ∀ i < 4, D1.base.bnd.idxOf (D1.base.bnd.getD i Port.dflt) = i := by decide
  have hb0 : D1.base.bnd.idxOf ⟨2, true, 1⟩ = 0 := by decide
  have hb1 : D1.base.bnd.idxOf ⟨6, true, 1⟩ = 1 := by decide
  have hb2 : D1.base.bnd.idxOf ⟨1, false, 1⟩ = 2 := by decide
  have hb3 : D1.base.bnd.idxOf ⟨5, false, 1⟩ = 3 := by decide
  have d0 : (D1.base.bnd[0]?).getD Port.dflt = ⟨2, true, 1⟩ := rfl
  have d1 : (D1.base.bnd[1]?).getD Port.dflt = ⟨6, true, 1⟩ := rfl
  have d2 : (D1.base.bnd[2]?).getD Port.dflt = ⟨1, false, 1⟩ := rfl
  have d3 : (D1.base.bnd[3]?).getD Port.dflt = ⟨5, false, 1⟩ := rfl
  have e0 : (E1n n).bnd.getD 0 Port.dflt = ⟨2, true, 1⟩ := rfl
  have e1 : (E1n n).bnd.getD 1 Port.dflt = ⟨6, true, 1⟩ := rfl
  have e0' : ((E1n n).bnd[0]?).getD Port.dflt = ⟨2, true, 1⟩ := rfl
  have e1' : ((E1n n).bnd[1]?).getD Port.dflt = ⟨6, true, 1⟩ := rfl
  have i0 : (E1n n).bnd.idxOf ⟨2, true, 1⟩ = 0 := by rw [bndE1]; simp
  have i1 : (E1n n).bnd.idxOf ⟨6, true, 1⟩ = 1 := by rw [bndE1]; simp
  have iB : ∀ p ∈ BIn n, (E1n n).bnd.idxOf p = 2 + (BIn n).idxOf p := fun p hp => by
    have hf : p.out = false := by
      obtain ⟨j, -, hp⟩ := List.mem_flatMap.1 hp
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
      rcases hp with rfl | rfl <;> rfl
    rw [bndE1, List.idxOf_append_of_notMem (fun h => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at h
      rcases h with rfl | rfl <;> simp at hf)]
    simp
  have gB : ∀ p ∈ BIn n, (E1n n).bnd.getD (2 + (BIn n).idxOf p) Port.dflt = p := fun p hp => by
    rw [bndE1, List.getD_append_right _ _ _ _ (by simp), show 2 + (BIn n).idxOf p -
      [(⟨2, true, 1⟩ : Port), ⟨6, true, 1⟩].length = (BIn n).idxOf p by simp,
      List.getD_eq_getElem _ _ (List.idxOf_lt_length_of_mem hp), List.getElem_idxOf]
  have gB' : ∀ p ∈ BIn n, ((E1n n).bnd[2 + (BIn n).idxOf p]?).getD Port.dflt = p := gB
  have mI0 : (⟨10 + 2 * t, false, 1⟩ : Port) ∈ BIn n := List.mem_flatMap.2 ⟨t, List.mem_range.2 ht, by simp⟩
  have mI1 : (⟨11 + 2 * t, false, 1⟩ : Port) ∈ BIn n := List.mem_flatMap.2 ⟨t, List.mem_range.2 ht, by simp⟩
  have mO0 : (⟨10 + 2 * n + 2 * t, true, 1⟩ : Port) ∈ BOn n :=
    List.mem_flatMap.2 ⟨t, List.mem_range.2 ht, by simp⟩
  have mO1 : (⟨11 + 2 * n + 2 * t, true, 1⟩ : Port) ∈ BOn n :=
    List.mem_flatMap.2 ⟨t, List.mem_range.2 ht, by simp⟩
  have p0 : (⟨2 * n + 10 + 2 * t + [2, 6].idxOf 2, true, 1⟩ : Port) = ⟨10 + 2 * n + 2 * t, true, 1⟩ := by
    simp only [Port.mk.injEq, and_true, show [2, 6].idxOf 2 = 0 from rfl]; omega
  have p1 : (⟨2 * n + 10 + 2 * t + [2, 6].idxOf 6, true, 1⟩ : Port) = ⟨11 + 2 * n + 2 * t, true, 1⟩ := by
    simp only [Port.mk.injEq, and_true, show [2, 6].idxOf 6 = 1 from rfl]; omega
  have q0 : (⟨10 + 2 * t + [1, 5].idxOf 1, false, 1⟩ : Port) = ⟨10 + 2 * t, false, 1⟩ := by
    simp only [Port.mk.injEq, and_true, show [1, 5].idxOf 1 = 0 from rfl]; omega
  have q1 : (⟨10 + 2 * t + [1, 5].idxOf 5, false, 1⟩ : Port) = ⟨11 + 2 * t, false, 1⟩ := by
    simp only [Port.mk.injEq, and_true, show [1, 5].idxOf 5 = 1 from rfl]; omega
  have hBO0 := (idxBOn ht).1
  have hBO1 := (idxBOn ht).2
  have hBI0 := (idxBIn ht).1
  have hBI1 := (idxBIn ht).2
  have gt0 : ((E1n n).bnd[2 + 2 * t]?).getD Port.dflt = ⟨10 + 2 * t, false, 1⟩ := by
    have := gB' _ mI0; rwa [hBI0] at this
  have gt1 : ((E1n n).bnd[2 + (2 * t + 1)]?).getD Port.dflt = ⟨11 + 2 * t, false, 1⟩ := by
    have := gB' _ mI1; rwa [hBI1] at this
  have gt0g : (E1n n).bnd.getD (2 + 2 * t) Port.dflt = ⟨10 + 2 * t, false, 1⟩ := gt0
  have gt1g : (E1n n).bnd.getD (2 + (2 * t + 1)) Port.dflt = ⟨11 + 2 * t, false, 1⟩ := gt1
  interval_cases i
  · have hg : D1.base.bnd.getD 0 Port.dflt = ⟨2, true, 1⟩ := rfl
    simp only [Diagram.ixMap, hb0, hb1, hb2, hb3, d0, d1, d2, d3, e0', e1',hb 0 (by omega), hg,
      φS_n (img := D1.base.nimg b1 n) (g := D1.base.gN b1) (p := ⟨2, true, 1⟩) [t] (by rw [hg1]; decide)
        (hn1 _ (by decide)),
      i0, e0, φS_g (img := (E1n n).nimg c1 n) (g := (E1n n).gN c1) (p := ⟨2, true, 1⟩) t
        (by rw [hg2]; decide),
      hm2 2 (by decide), hgt, p0, bndE2, List.idxOf_append_of_mem mO0, hBO0]
    simp [bpos]
  · have hg : D1.base.bnd.getD 1 Port.dflt = ⟨6, true, 1⟩ := rfl
    simp only [Diagram.ixMap, hb0, hb1, hb2, hb3, d0, d1, d2, d3, e0', e1',hb 1 (by omega), hg,
      φS_n (img := D1.base.nimg b1 n) (g := D1.base.gN b1) (p := ⟨6, true, 1⟩) [t] (by rw [hg1]; decide)
        (hn1 _ (by decide)),
      i1, e1, φS_g (img := (E1n n).nimg c1 n) (g := (E1n n).gN c1) (p := ⟨6, true, 1⟩) t
        (by rw [hg2]; decide),
      hm2 6 (by decide), hgt, p1, bndE2, List.idxOf_append_of_mem mO1, hBO1]
    simp [bpos]
  · have hg : D1.base.bnd.getD 2 Port.dflt = ⟨1, false, 1⟩ := rfl
    simp only [Diagram.ixMap, hb0, hb1, hb2, hb3, d0, d1, d2, d3, e0', e1',hb 2 (by omega), hg,
      φS_g (img := D1.base.nimg b1 n) (g := D1.base.gN b1) (p := ⟨1, false, 1⟩) t (by rw [hg1]; decide),
      hm1 1 (by decide), hgt, q0, iB _ mI0, gB _ mI0, gB' _ mI0, gt0, gt0g,
      φS_n (img := (E1n n).nimg c1 n) (g := (E1n n).gN c1) (p := ⟨10 + 2 * t, false, 1⟩) []
        (by rw [hg2]; exact hBI _ mI0) (hn2 _ (hBI _ mI0)),
      bndE2, List.idxOf_append_of_notMem (notin_BOn (n := n) ⟨10 + 2 * t, false, 1⟩ rfl), len_BOn, hBI0]
    simp [bpos]
  · have hg : D1.base.bnd.getD 3 Port.dflt = ⟨5, false, 1⟩ := rfl
    simp only [Diagram.ixMap, hb0, hb1, hb2, hb3, d0, d1, d2, d3, e0', e1',hb 3 (by omega), hg,
      φS_g (img := D1.base.nimg b1 n) (g := D1.base.gN b1) (p := ⟨5, false, 1⟩) t (by rw [hg1]; decide),
      hm1 5 (by decide), hgt, q1, iB _ mI1, gB _ mI1, gB' _ mI1, gt1, gt1g,
      φS_n (img := (E1n n).nimg c1 n) (g := (E1n n).gN c1) (p := ⟨11 + 2 * t, false, 1⟩) []
        (by rw [hg2]; exact hBI _ mI1) (hn2 _ (hBI _ mI1)),
      bndE2, List.idxOf_append_of_notMem (notin_BOn (n := n) ⟨11 + 2 * t, false, 1⟩ rfl), len_BOn, hBI1]
    simp [bpos]; omega

theorem bidx_sl1 {n s : ℕ} (hs : s < 4 * n) : (D1.base.bidx (fun _ => n) (sl n s)).1 = s := by
  unfold sl
  split_ifs with h
  · rw [bidx1 n _ _ (by omega) (by omega)]; simp only [bpos]; split_ifs <;> omega
  · rw [bidx1 n _ _ (by omega) (by omega)]; simp only [bpos]; split_ifs <;> omega

/-- The expansion in key order is `E2n n`. -/
theorem expandK1 (n : ℕ) : D1.base.expandK (fun _ => n) = E2n n := by
  unfold Diagram.expandK
  rw [expand1, stage1, stage2, SL1]
  set E := E2n n
  have hlen : E.bnd.length = 4 * n := by
    rw [bndE2, List.length_append, BOn, BIn, len_flat2, len_flat2]; ring
  have hL : ((range (4 * n)).map (sl n)).map
      (fun x => E.bnd.getD (D1.base.bidx (fun _ => n) x).1 Port.dflt) = E.outs ++ E.ins := by
    rw [List.map_map]
    have : ((range (4 * n)).map fun s => E.bnd.getD s Port.dflt) = E.bnd := by
      rw [← hlen]; exact Diagram.range_getD _
    rw [← Diagram.bnd]
    refine (List.map_congr_left fun s hs => ?_).trans this
    simp only [Function.comp, bidx_sl1 (List.mem_range.1 hs)]
  dsimp only
  rw [hL]
  simp [E, E2n]

/-! ## Template node and copy -/

theorem X1 (n : ℕ) : D1.X n = E2n n := expandK1 n

/-- The nodes of `D1.X n` with their template node and copy. -/
def Z1 (n : ℕ) : List (DNode × ℕ × ℕ) :=
  [(m0, (0, 0))] ++ (range n).map (fun j => ({ m1 with id := 10 + 2 * j }, (1, j))) ++
    (range n).map (fun j => ({ m2 with id := 10 + 2 * n + 2 * j }, (2, j))) ++ [(m3, (3, 0)), (m4, (4, 0))] ++
    (range n).map (fun j => ({ m5 with id := 11 + 2 * j }, (5, j))) ++
    (range n).map (fun j => ({ m6 with id := 11 + 2 * n + 2 * j }, (6, j))) ++
    [(m7, (7, 0)), (m8, (8, 0)), (m9, (9, 0))]

theorem org1 (n : ℕ) : D1.org n =
    [(0, 0)] ++ (range n).map (1, ·) ++ (range n).map (2, ·) ++ [(3, 0), (4, 0)] ++ (range n).map (5, ·) ++
      (range n).map (6, ·) ++ [(7, 0), (8, 0), (9, 0)] := by
  simp [FamDiagram.org, D1]

theorem zip1 (n : ℕ) : (D1.X n).nodes.zip (D1.org n) = Z1 n := by
  rw [X1, org1]
  simp only [E2n, Z1, List.append_assoc, List.cons_append, List.nil_append, List.zip_cons_cons]
  rw [List.zip_append (by simp), FamFca.zip_map_range, List.zip_append (by simp), FamFca.zip_map_range,
    List.zip_cons_cons, List.zip_cons_cons, List.zip_append (by simp), FamFca.zip_map_range,
    List.zip_append (by simp), FamFca.zip_map_range]
  rfl

/-- The id, template node and copy of an entry of `Z1 n`. -/
def zk (n : ℕ) (e : DNode × ℕ × ℕ) : Prop :=
  e.2.1 ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
  (e.2.1 ∈ [1, 2, 5, 6] → e.2.2 < n) ∧ (e.2.1 ∉ [1, 2, 5, 6] → e.2.2 = 0) ∧
  e.1 = { (D1.base.nodes.getD e.2.1 m0) with id :=
    if e.2.1 = 1 then 10 + 2 * e.2.2 else if e.2.1 = 2 then 10 + 2 * n + 2 * e.2.2
    else if e.2.1 = 5 then 11 + 2 * e.2.2 else if e.2.1 = 6 then 11 + 2 * n + 2 * e.2.2 else e.2.1 }

theorem mem_Z1 {n : ℕ} {a : DNode × ℕ × ℕ} (h : a ∈ Z1 n) : zk n a := by
  simp only [Z1, List.mem_append, List.mem_cons, List.not_mem_nil, or_false, List.mem_map,
    List.mem_range] at h
  rcases h with ((((((rfl | ⟨j, hj, rfl⟩) | ⟨j, hj, rfl⟩) | rfl | rfl) | ⟨j, hj, rfl⟩) | ⟨j, hj, rfl⟩) |
    rfl | rfl | rfl) <;> simp [zk, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, D1] <;> assumption

theorem Z1_inj {n : ℕ} {a e : DNode × ℕ × ℕ} (ha : a ∈ Z1 n) (he : e ∈ Z1 n) (h : a.1.id = e.1.id) :
    a = e := by
  obtain ⟨a1, a2⟩ := a
  obtain ⟨x, j⟩ := a2
  obtain ⟨e1, e2⟩ := e
  obtain ⟨y, k⟩ := e2
  obtain ⟨hx, hj1, hj0, ha1⟩ := mem_Z1 ha
  obtain ⟨hy, hk1, hk0, he1⟩ := mem_Z1 he
  simp only at hx hy hj1 hj0 hk1 hk0 ha1 he1 h
  subst ha1 he1
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hx hy hj1 hj0 hk1 hk0 h ⊢
  rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
  rcases hy with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
  simp only [true_or, or_true, true_implies, not_true, false_implies, forall_const, reduceCtorEq,
    false_or, or_false, OfNat.ofNat_ne_one, one_ne_zero] at hj1 hj0 hk1 hk0 h ⊢ <;>
  simp_all <;> omega

theorem orgOf1_eq {n y : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) (hy : e.1.id = y) :
    D1.orgOf n y = e.2 := by
  unfold FamDiagram.orgOf
  rw [zip1, FamFca.find?_unique (e := e) he (by simpa using hy)]
  · rfl
  · intro a ha hp
    simp only [decide_eq_true_eq] at hp
    exact Z1_inj ha he (hp.trans hy.symm)

theorem cp1_eq {n : ℕ} {e : DNode × ℕ × ℕ} (he : e ∈ Z1 n) : D1.cp n e.2.1 e.2.2 = e.1.id := by
  unfold FamDiagram.cp
  rw [zip1, FamFca.find?_unique (e := e) he (by simp)]
  · rfl
  · intro a ha hp
    simp only [decide_eq_true_eq] at hp
    rcases a with ⟨a1, x, j⟩
    rcases e with ⟨e1, y, k⟩
    obtain ⟨-, -, -, ha1⟩ := mem_Z1 ha
    obtain ⟨-, -, -, he1⟩ := mem_Z1 he
    simp only [Prod.mk.injEq] at hp ha1 he1
    obtain ⟨rfl, rfl⟩ := hp
    rw [ha1, he1]

end FamDraper

end ArchytasZX
