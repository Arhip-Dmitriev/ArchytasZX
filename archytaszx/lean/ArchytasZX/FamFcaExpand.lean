/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamExpandEq

/-!
# The boxed Fourier constant adder's expansion

`D0`: the engine's boxed `fourier_constant_adder` as a `FamDiagram`, one node box `b0` over
the nine template nodes, the pair boxes `(2.out 1, 0.in 1)` and `(5.out 1, 9.in 1)`.

`expand0`: its expansion at `n` copies is `copyNodes b0 n`. `cp0 j x = 10 + 9 j + rk x` is copy
`j` of the template node `x`, `rk x` its rank in the box. `copyNodes0_nodes`,
`copyNodes0_wires`, `copyNodes0_outs`, `copyNodes0_ins`: the closed form, node-major and
wire-major.
-/

namespace ArchytasZX

namespace FamFca

open List

/-- The engine's boxed `fourier_constant_adder`. -/
def D0 : FamDiagram :=
  { base :=
      { nodes := [⟨0, .Z, 0, 0, [0], [0, 0]⟩, ⟨1, .F, 0, 0, [0], [0]⟩, ⟨2, .Z, 0, 0, [0, 0], [0]⟩,
          ⟨4, .Z, 0, 0, [0], [0]⟩, ⟨5, .Z, 0, 0, [0, 0], [0]⟩, ⟨6, .F, 0, 0, [0], [0]⟩,
          ⟨7, .F, 0, 0, [0], [0]⟩, ⟨8, .F, 0, 0, [0], [0]⟩, ⟨9, .Z, 0, 0, [0], [0, 0]⟩],
        wires := [(⟨0, true, 0⟩, ⟨1, false, 0⟩), (⟨1, true, 0⟩, ⟨2, false, 0⟩),
          (⟨2, true, 0⟩, ⟨4, false, 0⟩), (⟨4, true, 0⟩, ⟨5, false, 0⟩),
          (⟨5, true, 0⟩, ⟨6, false, 0⟩), (⟨6, true, 0⟩, ⟨7, false, 0⟩),
          (⟨7, true, 0⟩, ⟨8, false, 0⟩), (⟨8, true, 0⟩, ⟨9, false, 0⟩)],
        outs := [⟨9, true, 0⟩], ins := [⟨0, false, 0⟩],
        boxes := [⟨0, 0, none, [0, 1, 2, 4, 5, 6, 7, 8, 9], []⟩] },
    links := [],
    pairs := [⟨1, 0, ⟨2, true, 1⟩, ⟨0, false, 1⟩, ⟨3, .ZP 0 0 1, 0, 0, [0], [0]⟩⟩,
              ⟨2, 0, ⟨5, true, 1⟩, ⟨9, false, 1⟩, ⟨10, .ZP 0 0 (-1), 0, 0, [0], [0]⟩⟩],
    ph := [(4, fun _ j => j + 1)],
    dim := [(3, fun _ c => c), (10, fun _ c => c)],
    legs := [] }

/-- The node box. -/
def b0 : DBox := ⟨0, 0, none, [0, 1, 2, 4, 5, 6, 7, 8, 9], []⟩

/-- The template node `x`'s rank in `b0`. -/
def rk (x : ℕ) : ℕ := [0, 1, 2, 4, 5, 6, 7, 8, 9].idxOf x

/-- Copy `j` of the template node `x`. -/
def cp0 (j x : ℕ) : ℕ := 10 + 9 * j + rk x

theorem boxes0 : D0.base.boxes = [b0] := rfl

theorem expand0 (n : ℕ) : D0.base.expand (fun _ => n) = D0.base.copyNodes b0 n :=
  Diagram.expand_one _ _ b0 boxes0 rfl rfl

theorem fresh0 : D0.base.fresh = 10 := by decide

theorem sort0 : b0.nodes.mergeSort (· ≤ ·) = [0, 1, 2, 4, 5, 6, 7, 8, 9] :=
  List.mergeSort_eq_self (r := (· ≤ ·)) (by decide)

theorem cpId0 (j x : ℕ) : D0.base.cpId b0 j x = cp0 j x := by
  simp only [Diagram.cpId, fresh0, sort0, cp0, rk, show b0.nodes.length = 9 from rfl]
  omega

theorem crossP0 (p : Port) : D0.base.crossP b0 p = false := by
  simp only [Diagram.crossP, Diagram.inS, D0, b0]
  by_cases h : p.node ∈ [0, 1, 2, 4, 5, 6, 7, 8, 9]
  · simp [h]
  · simp only [h, decide_false, Bool.not_false, Bool.true_and]
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    simp only [List.any_cons, List.any_nil]
    rcases p with ⟨a, o, q⟩
    simp only [not_or] at h
    obtain ⟨h0, h1, h2, h4, h5, h6, h7, h8, h9⟩ := h
    simp [Ne.symm h0, Ne.symm h1, Ne.symm h2, Ne.symm h4, Ne.symm h5, Ne.symm h6, Ne.symm h7,
      Ne.symm h8, Ne.symm h9]

theorem mem_b0 (x : DNode) (hx : x ∈ D0.base.nodes) : x.id ∈ b0.nodes := by
  simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem copyNodes0_nodes (n : ℕ) : (D0.base.copyNodes b0 n).nodes =
    D0.base.nodes.flatMap fun x => (range n).map fun j => { x with id := cp0 j x.id } := by
  simp only [Diagram.copyNodes]
  refine List.flatMap_congr fun x hx => ?_
  simp [mem_b0 x hx, cpId0]

theorem nimg0 (p : Port) (hp : p.node ∈ b0.nodes) (n : ℕ) :
    D0.base.nimg b0 n p = (range n).map fun j => ⟨cp0 j p.node, p.out, p.pos⟩ := by
  simp [Diagram.nimg, Diagram.inS, hp, cpId0]

theorem zip_range_self (n : ℕ) : (range n).zip (range n) = (range n).map fun j => (j, j) := by
  refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
  simp

theorem copyNodes0_wires (n : ℕ) : (D0.base.copyNodes b0 n).wires =
    D0.base.wires.flatMap fun w => (range n).map fun j =>
      ((⟨cp0 j w.1.node, w.1.out, w.1.pos⟩ : Port), (⟨cp0 j w.2.node, w.2.out, w.2.pos⟩ : Port)) := by
  simp only [Diagram.copyNodes]
  refine List.flatMap_congr fun w hw => ?_
  have h : w.1.node ∈ b0.nodes ∧ w.2.node ∈ b0.nodes := by
    simp only [D0, List.mem_cons, List.not_mem_nil, or_false] at hw
    rcases hw with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  rw [nimg0 _ h.1, nimg0 _ h.2, List.zip_map, zip_range_self, List.map_map]
  rfl

theorem flatMap_single {α β : Type} (f : α → β) (l : List α) :
    l.flatMap (fun j => [f j]) = l.map f := by
  induction l <;> simp_all

theorem copyNodes0_outs (n : ℕ) : (D0.base.copyNodes b0 n).outs =
    (range n).map fun j => ⟨cp0 j 9, true, 0⟩ := by
  have hO : D0.base.outs = [⟨9, true, 0⟩] := rfl
  have hi : Diagram.inS b0 ⟨9, true, 0⟩ = true := by decide
  simp only [Diagram.copyNodes, Diagram.spliceN, hO, List.filter_cons, hi, ite_true,
    List.filter_nil, List.findIdx_cons, List.take_zero, List.flatMap_nil, List.nil_append,
    List.drop_zero, Bool.not_true, Bool.false_eq_true, ite_false, cpId0, reduceCtorEq,
    List.map_cons, List.map_nil, List.append_nil]
  exact flatMap_single _ _

theorem copyNodes0_ins (n : ℕ) : (D0.base.copyNodes b0 n).ins =
    (range n).map fun j => ⟨cp0 j 0, false, 0⟩ := by
  have hO : D0.base.ins = [⟨0, false, 0⟩] := rfl
  have hi : Diagram.inS b0 ⟨0, false, 0⟩ = true := by decide
  simp only [Diagram.copyNodes, Diagram.spliceN, hO, List.filter_cons, hi, ite_true,
    List.filter_nil, List.findIdx_cons, List.take_zero, List.flatMap_nil, List.nil_append,
    List.drop_zero, Bool.not_true, Bool.false_eq_true, ite_false, cpId0, reduceCtorEq,
    List.map_cons, List.map_nil, List.append_nil]
  exact flatMap_single _ _

end FamFca

end ArchytasZX
