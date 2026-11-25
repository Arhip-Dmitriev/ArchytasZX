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

end FamFca

end ArchytasZX
