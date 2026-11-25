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

/-! ## The boundary in key order -/

theorem keySig0 : D0.base.keySig = [([], [0]), ([], [0])] := by decide

theorem legChain0 : D0.base.bnd.map D0.base.legChain = [[0], [0]] := by decide

theorem bnd0 : D0.base.bnd = [⟨9, true, 0⟩, ⟨0, false, 0⟩] := rfl

theorem cidx0 (n : ℕ) : cidx (fun _ => n) [0] = (range n).map fun t => [t] := by
  simp [cidx, flatMap_single]

theorem SDom0 (n : ℕ) : D0.base.SDom (fun _ => n) =
    (range 2).flatMap fun i => (range n).map fun t => (i, [t]) := by
  unfold Diagram.SDom
  have hl : D0.base.bnd.length = 2 := rfl
  rw [hl]
  refine List.flatMap_congr fun i hi => ?_
  have hc : D0.base.legChain (D0.base.bnd.getD i Port.dflt) = [0] := by
    have := congrArg (fun l => l.getD i []) legChain0
    simp only [List.mem_range] at hi
    interval_cases i <;> decide
  rw [hc, cidx0, List.map_map]
  rfl

theorem key0 (i t : ℕ) (hi : i < 2) :
    Diagram.keyOf (D0.base.keySig.getD i ([], [])) i [t] = [i, t] := by
  rw [keySig0]
  interval_cases i <;> rfl

theorem SL0 (n : ℕ) : D0.base.SL (fun _ => n) =
    (range 2).flatMap fun i => (range n).map fun t => (i, [t]) := by
  unfold Diagram.SL
  rw [SDom0]
  refine List.Pairwise.insertionSort_eq ?_
  rw [List.pairwise_flatMap]
  refine ⟨fun i hi => ?_, ?_⟩
  · rw [List.pairwise_map]
    refine (List.pairwise_lt_range (n := n)).imp_of_mem fun {a b} _ _ hab => ?_
    simp only [List.mem_range] at hi
    rw [key0 i a hi, key0 i b hi]
    simp [Diagram.lexLe, hab]
  · refine (List.pairwise_lt_range (n := 2)).imp_of_mem fun {i i'} hi hi' hii => ?_
    simp only [List.mem_range] at hi hi'
    intro x hx y hy
    obtain ⟨a, -, rfl⟩ := List.mem_map.1 hx
    obtain ⟨b, -, rfl⟩ := List.mem_map.1 hy
    rw [key0 i a hi, key0 i' b hi']
    simp [Diagram.lexLe, hii]

theorem pick0 : D0.base.pick = some b0 := by decide

theorem pick_nil (D : Diagram) (h : D.boxes = []) : D.pick = none := by
  simp [Diagram.pick, h]

theorem bnd_copyNodes0 (n : ℕ) : (D0.base.copyNodes b0 n).bnd =
    (range n).map (fun j => (⟨cp0 j 9, true, 0⟩ : Port)) ++
      (range n).map (fun j => (⟨cp0 j 0, false, 0⟩ : Port)) := by
  rw [Diagram.bnd, copyNodes0_outs, copyNodes0_ins]

theorem idxOf_map_range {f : ℕ → Port} (hf : ∀ a b, f a = f b → a = b) :
    ∀ {n t : ℕ}, t < n → ((range n).map f).idxOf (f t) = t
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | n + 1, t, h => by
    rw [List.range_succ, List.map_append]
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | rfl
    · rw [List.idxOf_append_of_mem (List.mem_map_of_mem (List.mem_range.2 h)), idxOf_map_range hf h]
    · rw [List.idxOf_append_of_notMem (fun hm => by
        obtain ⟨a, ha, e⟩ := List.mem_map.1 hm
        exact absurd (hf _ _ e) (by simp at ha; omega))]
      simp

theorem cp0_inj {j j' x : ℕ} (h : cp0 j x = cp0 j' x) : j = j' := by
  simp only [cp0] at h; omega

theorem phi0 (n t : ℕ) (ht : t < n) (p : Port) (hp : p.node ∈ b0.nodes) :
    φS (D0.base.nimg b0 n) (D0.base.gN b0) p [t] = (⟨cp0 t p.node, p.out, p.pos⟩, []) := by
  have hin : Diagram.inS b0 p = true := by simpa [Diagram.inS] using hp
  simp only [φS, Diagram.gN, hin, Bool.true_or, ite_true, nimg0 p hp]
  rw [List.getD_eq_getElem _ _ (by simpa using ht)]
  simp

theorem bidx0 (n i t : ℕ) (hi : i < 2) (ht : t < n) :
    D0.base.bidx (fun _ => n) (i, [t]) = (i * n + t, []) := by
  unfold Diagram.bidx
  have hB := Diagram.copyNodes_boxes_one D0.base b0 boxes0 n
  simp only [boxes0, List.length_singleton, Diagram.bidxFuel, pick0, Diagram.step,
    show b0.ports = [] from rfl, ite_true, Diagram.expandNode_none _ b0 rfl]
  rw [pick_nil _ hB]
  simp only [Diagram.stepIx, Diagram.gather_none _ b0 rfl, ite_self, show b0.ports = [] from rfl,
    ite_true, Diagram.ixMap, bnd0]
  rw [bnd_copyNodes0]
  have hf : ∀ (o : Bool) (x : ℕ), ∀ a b, (⟨cp0 a x, o, 0⟩ : Port) = ⟨cp0 b x, o, 0⟩ → a = b :=
    fun o x a b e => cp0_inj (Port.mk.inj e).1
  interval_cases i
  · simp only [List.getD_cons_zero, List.idxOf_cons_self]
    rw [phi0 n t ht _ (by decide)]
    simp only
    have hm : (⟨cp0 t 9, true, 0⟩ : Port) ∈ (range n).map (fun j => (⟨cp0 j 9, true, 0⟩ : Port)) :=
      List.mem_map.2 ⟨t, List.mem_range.2 ht, rfl⟩
    rw [List.idxOf_append_of_mem hm, idxOf_map_range (hf true 9) ht]
    simp
  · simp only [List.getD_cons_succ, List.getD_cons_zero]
    have h1 : D0.base.bnd.idxOf ⟨0, false, 0⟩ = 1 := by decide
    rw [bnd0] at h1
    rw [h1]
    simp only [List.getD_cons_succ, List.getD_cons_zero]
    rw [phi0 n t ht _ (by decide)]
    simp only
    rw [List.idxOf_append_of_notMem (fun hm => by
        obtain ⟨a, -, e⟩ := List.mem_map.1 hm
        exact absurd (Port.mk.inj e).2.1 (by decide)), idxOf_map_range (hf false 0) ht]
    simp
