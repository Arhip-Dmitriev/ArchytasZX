/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import Mathlib

/-!
# Diagrams as labelled tensor networks over ℂ

A `Net` lists generator instances whose legs carry wire labels. A label shared by legs is
one contracted index; `int` lists the labels summed over, `bnd` the open boundary labels.
`eval` is the sum over the internal labels of the product of the generator tensors.
-/

namespace ArchytasZX

open Complex Finset

/-- Generator kinds: spiders, Fourier box, triangle and inverse, W node, dimension binder
and splitter, the bare wire, a Z, X or W whose last legs bundle along the chains of
multiplicities `cs` (outermost first), and copies of a generator along multiplicity `m`. -/
inductive Kind
  | Z | X | F | T | Ti | W | B | S | I
  | ZB (cs : List (List ℕ)) | XB (co ci : List (List ℕ)) | WB (cs : List (List ℕ))
  | Bun (m : ℕ) (base : Kind)
  deriving DecidableEq, Repr

/-- One generator instance: output and input leg labels, dimension index, phase index. -/
structure Node where
  kind : Kind
  outs : List ℕ
  ins : List ℕ
  dim : ℕ
  ph : ℕ
  deriving DecidableEq, Repr

/-- Nodes, summed labels and boundary labels. -/
structure Net where
  nodes : List Node
  int : List ℕ
  bnd : List ℕ
  deriving DecidableEq, Repr

/-- Dimension values, phase vectors in turns, each label's dimension index, and each
multiplicity symbol's value. -/
structure Env where
  dimv : ℕ → ℕ
  phase : ℕ → ℕ → ℝ
  labs : List (ℕ × ℕ)
  mult : ℕ → ℕ := fun _ => 0

/-- `e^{2πiθ}`. -/
noncomputable def ex (θ : ℝ) : ℂ := Complex.exp (2 * Real.pi * Complex.I * θ)

/-- `e^{2πi/d}`. -/
noncomputable def om (d : ℕ) : ℂ := Complex.exp (2 * Real.pi * Complex.I / d)

/-- The unitary DFT entry `ω^{jk}/√d`. -/
noncomputable def fou (d j k : ℕ) : ℂ := om d ^ (j * k) / (Real.sqrt d : ℂ)

/-- The Z spider on leg values `vs`. -/
noncomputable def tZ (d : ℕ) (φ : ℕ → ℝ) (vs : List ℕ) : ℂ :=
  ∑ k ∈ range d, if ∀ v ∈ vs, v = k then ex (φ k) else 0

/-- The X spider: the Z spider with `F` on every output and `conj F` on every input. -/
noncomputable def tX (d : ℕ) (φ : ℕ → ℝ) (os is : List ℕ) : ℂ :=
  ∑ k ∈ range d, ex (φ k) * (os.map fun o => fou d o k).prod *
    (is.map fun i => (starRingEnd ℂ) (fou d i k)).prod

/-- The triangle: `1` iff `o = 0` or `o = i`. -/
def tT (o i : ℕ) : ℂ := if o = 0 ∨ o = i then 1 else 0

/-- The inverse triangle. -/
def tTi (o i : ℕ) : ℂ := if o = i then 1 else if o = 0 ∧ 1 ≤ i then -1 else 0

/-- The W node with input `i`. -/
def tW (d : ℕ) (os : List ℕ) (i : ℕ) : ℂ :=
  if i = 0 then (if ∀ o ∈ os, o = 0 then 1 else 0)
  else if i < d ∧ os.filter (· ≠ 0) = [i] then 1 else 0

/-- The binder entry `[k = a t + b]` over `a < s`, `b < t`. -/
def tB (s t k a b : ℕ) : ℂ := if a < s ∧ b < t ∧ k = a * t + b then 1 else 0

/-- The bare wire on dimension `d`. -/
def tI (d a b : ℕ) : ℂ := if a = b ∧ a < d then 1 else 0

/-- Digit `i` of `v` in base `d`. -/
def digit (d i v : ℕ) : ℕ := v / d ^ i % d

/-- The first `k` digits of `v` in base `d`. -/
def digs (d k v : ℕ) : List ℕ := (List.range k).map fun i => digit d i v

/-- A leg value bundling along the chain `c` (outermost first) as its base-`d` digits. -/
def digsC (mult : ℕ → ℕ) (d : ℕ) : List ℕ → ℕ → List ℕ
  | [], v => [v]
  | m :: c, v => (digs (d ^ (c.map mult).prod) (mult m) v).flatMap (digsC mult d c)

/-- The digits of the bundled legs `vs` along the chains `cs`. -/
def tailVals (mult : ℕ → ℕ) (d : ℕ) (vs : List ℕ) (cs : List (List ℕ)) : List ℕ :=
  (List.zipWith (fun v c => digsC mult d c v) vs cs).flatten

/-- Leg values whose last `cs.length` legs bundle along the chains `cs`. -/
def bundled (mult : ℕ → ℕ) (d : ℕ) (vs : List ℕ) (cs : List (List ℕ)) : List ℕ :=
  vs.take (vs.length - cs.length) ++ tailVals mult d (vs.drop (vs.length - cs.length)) cs

/-- The chains of a copied generator's `n` outputs. -/
def Kind.outCs : Kind → ℕ → List (List ℕ)
  | .XB co _, n => List.replicate (n - co.length) [] ++ co
  | .WB cs, n => List.replicate (n - cs.length) [] ++ cs
  | .Bun m b, n => (b.outCs n).map (m :: ·)
  | _, n => List.replicate n []

/-- The chains of a copied generator's `n` inputs. -/
def Kind.inCs : Kind → ℕ → List (List ℕ)
  | .ZB cs, n => List.replicate (n - cs.length) [] ++ cs
  | .XB _ ci, n => List.replicate (n - ci.length) [] ++ ci
  | .Bun m b, n => (b.inCs n).map (m :: ·)
  | _, n => List.replicate n []

/-- Digit `i` of each leg value, in base `d` to the multiplicities of its chain. -/
def copyLegs (mult : ℕ → ℕ) (d i : ℕ) (cs : List (List ℕ)) (vs : List ℕ) : List ℕ :=
  List.zipWith (fun c v => digit (d ^ (c.map mult).prod) i v) cs vs

/-- Copy `i`'s output values. -/
def copyOuts (E : Env) (base : Kind) (d i : ℕ) (os : List ℕ) : List ℕ :=
  copyLegs E.mult d i (base.outCs os.length) os

/-- Copy `i`'s input values. -/
def copyIns (E : Env) (base : Kind) (d i : ℕ) (is : List ℕ) : List ℕ :=
  copyLegs E.mult d i (base.inCs is.length) is

/-- A generator's tensor entry at the leg values `os`, `is`. -/
noncomputable def tensor (E : Env) : Kind → ℕ → ℕ → List ℕ → List ℕ → ℂ
  | .Z, d, ph, os, is => tZ d (E.phase ph) (os ++ is)
  | .X, d, ph, os, is => tX d (E.phase ph) os is
  | .F, d, _, [o], [i] => fou d o i
  | .T, _, _, [o], [i] => tT o i
  | .Ti, _, _, [o], [i] => tTi o i
  | .W, d, _, os, [i] => tW d os i
  | .I, d, _, [a], [b] => tI d a b
  | .ZB cs, d, ph, os, is => tZ d (E.phase ph) (os ++ bundled E.mult d is cs)
  | .XB co ci, d, ph, os, is => tX d (E.phase ph) (bundled E.mult d os co) (bundled E.mult d is ci)
  | .WB cs, d, _, os, [w] => tW d (bundled E.mult d os cs) w
  | .Bun m b, d, ph, os, is => ((List.range (E.mult m)).map fun i =>
      tensor E b d ph (copyOuts E b d i os) (copyIns E b d i is)).prod
  | _, _, _, _, _ => 0

/-- One node's tensor entry under the label assignment `σ`. -/
noncomputable def val (E : Env) (σ : ℕ → ℕ) (n : Node) : ℂ :=
  let d := E.dimv n.dim
  let os := n.outs.map σ
  let is := n.ins.map σ
  match n.kind, os, is with
  | .Z, os, is => tZ d (E.phase n.ph) (os ++ is)
  | .X, os, is => tX d (E.phase n.ph) os is
  | .F, [o], [i] => fou d o i
  | .T, [o], [i] => tT o i
  | .Ti, [o], [i] => tTi o i
  | .W, os, [i] => tW d os i
  | .B, [k], [a, b] => tB d (E.dimv n.ph) k a b
  | .S, [a, b], [k] => tB d (E.dimv n.ph) k a b
  | .I, [a], [b] => tI d a b
  | .ZB cs, os, is => tZ d (E.phase n.ph) (os ++ bundled E.mult d is cs)
  | .XB co ci, os, is => tX d (E.phase n.ph) (bundled E.mult d os co) (bundled E.mult d is ci)
  | .WB cs, os, [w] => tW d (bundled E.mult d os cs) w
  | .Bun m b, os, is => tensor E (.Bun m b) d n.ph os is
  | _, _, _ => 0

/-- A label's dimension. -/
def labDim (E : Env) (l : ℕ) : ℕ := E.dimv ((E.labs.lookup l).getD 0)

/-- The sum over the labels `ls`, each over its dimension, of `f`. -/
noncomputable def sumOver (D : ℕ → ℕ) : List ℕ → ((ℕ → ℕ) → ℂ) → (ℕ → ℕ) → ℂ
  | [], f, σ => f σ
  | l :: ls, f, σ => ∑ j ∈ range (D l), sumOver D ls f (Function.update σ l j)

/-- The product of the node entries. -/
noncomputable def prodVal (E : Env) (ns : List Node) (σ : ℕ → ℕ) : ℂ :=
  (ns.map (val E σ)).prod

/-- The network's entry at the boundary assignment `σ`. -/
noncomputable def eval (E : Env) (N : Net) (σ : ℕ → ℕ) : ℂ :=
  sumOver (labDim E) N.int (prodVal E N.nodes) σ

/-- Every label in `ls` within its dimension. -/
def InRange (E : Env) (ls : List ℕ) (σ : ℕ → ℕ) : Prop := ∀ l ∈ ls, σ l < labDim E l

/-- `s₁ • N₁` and `s₂ • N₂` are the same tensor. -/
def Equal (E : Env) (s₁ : ℂ) (N₁ : Net) (s₂ : ℂ) (N₂ : Net) : Prop :=
  N₁.bnd.Perm N₂.bnd ∧ ∀ σ, InRange E N₁.bnd σ → s₁ * eval E N₁ σ = s₂ * eval E N₂ σ

theorem InRange.perm {E : Env} {a b : List ℕ} {σ : ℕ → ℕ} (h : a.Perm b) (ha : InRange E a σ) :
    InRange E b σ := fun l hl => ha l (h.mem_iff.2 hl)

theorem Equal.refl (E : Env) (s : ℂ) (N : Net) : Equal E s N s N :=
  ⟨List.Perm.refl _, fun _ _ => rfl⟩

theorem Equal.symm {E : Env} {s₁ s₂ : ℂ} {N₁ N₂ : Net} (h : Equal E s₁ N₁ s₂ N₂) :
    Equal E s₂ N₂ s₁ N₁ :=
  ⟨h.1.symm, fun σ hσ => (h.2 σ (hσ.perm h.1.symm)).symm⟩

theorem Equal.trans {E : Env} {s₁ s₂ s₃ : ℂ} {N₁ N₂ N₃ : Net} (h : Equal E s₁ N₁ s₂ N₂)
    (h' : Equal E s₂ N₂ s₃ N₃) : Equal E s₁ N₁ s₃ N₃ :=
  ⟨h.1.trans h'.1, fun σ hσ => (h.2 σ hσ).trans (h'.2 σ (hσ.perm h.1))⟩

/-- Chaining two scaled equations through their shared network. -/
theorem Equal.mul_trans {E : Env} {a b c e : ℂ} {X Y Z : Net} (h : Equal E a X b Y)
    (h' : Equal E c Y e Z) : Equal E (a * c) X (b * e) Z := by
  refine ⟨h.1.trans h'.1, fun σ hσ => ?_⟩
  have h1 := h.2 σ hσ
  have h2 := h'.2 σ (hσ.perm h.1)
  calc a * c * eval E X σ = c * (a * eval E X σ) := by ring
    _ = c * (b * eval E Y σ) := by rw [h1]
    _ = b * (c * eval E Y σ) := by ring
    _ = b * (e * eval E Z σ) := by rw [h2]
    _ = b * e * eval E Z σ := by ring

/-- The same equation under other scalars, given `s b = t a` and `a ≠ 0`. -/
theorem Equal.rescale {E : Env} {a b s t : ℂ} {X Y : Net} (h : Equal E a X b Y) (ha : a ≠ 0)
    (hst : s * b = t * a) : Equal E s X t Y := by
  refine ⟨h.1, fun σ hσ => ?_⟩
  have h1 := h.2 σ hσ
  calc s * eval E X σ = s / a * (a * eval E X σ) := by field_simp
    _ = s / a * (b * eval E Y σ) := by rw [h1]
    _ = (s * b) / a * eval E Y σ := by ring
    _ = t * eval E Y σ := by rw [hst]; field_simp

/-! ## Sums over labels -/

section sums

variable (D : ℕ → ℕ)

theorem sumOver_congr (ls : List ℕ) (f g : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ)
    (h : ∀ τ, (∀ x ∉ ls, τ x = σ x) → (∀ x ∈ ls, τ x < D x) → f τ = g τ) :
    sumOver D ls f σ = sumOver D ls g σ := by
  induction ls generalizing σ with
  | nil => exact h σ (fun _ _ => rfl) (by simp)
  | cons l ls ih =>
    refine Finset.sum_congr rfl fun j hj => ih _ fun τ hτ hr => h τ ?_ ?_
    · intro x hx
      rw [hτ x (fun h' => hx (List.mem_cons_of_mem _ h'))]
      exact Function.update_of_ne (by rintro rfl; exact hx List.mem_cons_self) _ _
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx'
      · by_cases hm : x ∈ ls
        · exact hr x hm
        · rw [hτ x hm, Function.update_self]; exact Finset.mem_range.1 hj
      · exact hr x hx'

/-- `f` reads `σ` only on the labels `S`. -/
def Reads (S : List ℕ) (f : (ℕ → ℕ) → ℂ) : Prop :=
  ∀ τ τ', (∀ x ∈ S, τ x = τ' x) → f τ = f τ'

theorem sumOver_ext (ls S : List ℕ) (f : (ℕ → ℕ) → ℂ) (hf : Reads S f) (σ σ' : ℕ → ℕ)
    (h : ∀ x ∈ S, x ∉ ls → σ x = σ' x) : sumOver D ls f σ = sumOver D ls f σ' := by
  induction ls generalizing σ σ' with
  | nil => exact hf σ σ' fun x hx => h x hx (by simp)
  | cons l ls ih =>
    refine Finset.sum_congr rfl fun j _ => ih _ _ fun x hx hn => ?_
    by_cases hxl : x = l
    · subst hxl; simp
    · rw [Function.update_of_ne hxl, Function.update_of_ne hxl]
      exact h x hx (by simp [hxl, hn])

theorem sumOver_append (a b : List ℕ) (f : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver D (a ++ b) f σ = sumOver D a (sumOver D b f) σ := by
  induction a generalizing σ with
  | nil => rfl
  | cons l a ih => exact Finset.sum_congr rfl fun j _ => ih _

theorem mul_sumOver (k : ℂ) (ls : List ℕ) (f : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    k * sumOver D ls f σ = sumOver D ls (fun τ => k * f τ) σ := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l ls ih =>
    simp only [sumOver, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => ih _

theorem sumOver_const_mul (ls S : List ℕ) (g f : (ℕ → ℕ) → ℂ) (hg : Reads S g)
    (hS : ∀ x ∈ S, x ∉ ls) (σ : ℕ → ℕ) :
    sumOver D ls (fun τ => g τ * f τ) σ = g σ * sumOver D ls f σ := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l ls ih =>
    simp only [sumOver, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [ih (fun x hx hm => hS x hx (List.mem_cons_of_mem _ hm))]
    congr 1
    refine hg _ _ fun x hx => ?_
    have : x ≠ l := fun h' => hS x hx (h' ▸ List.mem_cons_self)
    exact Function.update_of_ne this _ _

theorem sumOver_perm {a b : List ℕ} (h : a.Perm b) (f : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver D a f σ = sumOver D b f σ := by
  induction h generalizing σ with
  | nil => rfl
  | cons l _ ih => exact Finset.sum_congr rfl fun j _ => ih _
  | swap x y l =>
    simp only [sumOver]
    by_cases hxy : x = y
    · subst hxy; rfl
    · rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
      rw [Function.update_comm hxy]
  | trans _ _ ih₁ ih₂ => exact (ih₁ σ).trans (ih₂ σ)

/-- Renaming the summed labels by a bijection `f` that keeps their dimensions. -/
theorem sumOver_map (f : ℕ → ℕ) (hf : Function.Injective f) (ls : List ℕ)
    (hd : ∀ x ∈ ls, D (f x) = D x) (g : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver D (ls.map f) (fun τ => g (τ ∘ f)) σ = sumOver D ls g (σ ∘ f) := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l ls ih =>
    simp only [List.map_cons, sumOver]
    rw [hd l List.mem_cons_self]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [ih (fun x hx => hd x (List.mem_cons_of_mem _ hx))]
    congr 1
    funext x
    simp only [Function.comp, Function.update]
    by_cases hx : x = l
    · subst hx; simp
    · have : f x ≠ f l := fun h' => hx (hf h')
      simp [hx, this]

end sums

/-! ## Node entries -/

/-- All labels on a node. -/
def Node.labels (n : Node) : List ℕ := n.outs ++ n.ins

/-- All labels on a list of nodes. -/
def labelsOf (ns : List Node) : List ℕ := ns.flatMap Node.labels

theorem val_reads (E : Env) (n : Node) : Reads n.labels (fun σ => val E σ n) := by
  intro τ τ' h
  have ho : n.outs.map τ = n.outs.map τ' :=
    List.map_congr_left fun x hx => h x (List.mem_append_left _ hx)
  have hi : n.ins.map τ = n.ins.map τ' :=
    List.map_congr_left fun x hx => h x (List.mem_append_right _ hx)
  simp only [val, ho, hi]

theorem prodVal_reads (E : Env) (ns : List Node) : Reads (labelsOf ns) (prodVal E ns) := by
  intro τ τ' h
  unfold prodVal
  congr 1
  refine List.map_congr_left fun n hn => val_reads E n τ τ' fun x hx => h x ?_
  exact List.mem_flatMap.2 ⟨n, hn, hx⟩

theorem prodVal_append (E : Env) (a b : List Node) (σ : ℕ → ℕ) :
    prodVal E (a ++ b) σ = prodVal E a σ * prodVal E b σ := by
  simp [prodVal]

/-- Rename every leg label through `f`. -/
def Node.rename (f : ℕ → ℕ) (n : Node) : Node := { n with outs := n.outs.map f, ins := n.ins.map f }

theorem val_rename (E : Env) (f : ℕ → ℕ) (σ : ℕ → ℕ) (n : Node) :
    val E σ (n.rename f) = val E (σ ∘ f) n := by
  simp [val, Node.rename, List.map_map]

theorem prodVal_rename (E : Env) (f : ℕ → ℕ) (σ : ℕ → ℕ) (ns : List Node) :
    prodVal E (ns.map (Node.rename f)) σ = prodVal E ns (σ ∘ f) := by
  simp [prodVal, List.map_map, Function.comp_def, val_rename]

/-! ## Frame: a local equation in any surrounding network -/

/-- `kL • L = kR • R` as local networks, at every assignment in range on `iface`. -/
def LocalEq (E : Env) (L : List Node) (iL : List ℕ) (R : List Node) (iR : List ℕ)
    (iface : List ℕ) (kL kR : ℂ) : Prop :=
  ∀ τ, InRange E iface τ →
    kL * sumOver (labDim E) iL (prodVal E L) τ = kR * sumOver (labDim E) iR (prodVal E R) τ

theorem frame (E : Env) (C L R : List Node) (iC iL iR bnd iface : List ℕ) (kL kR : ℂ)
    (hloc : LocalEq E L iL R iR iface kL kR)
    (hL : ∀ x ∈ labelsOf C, x ∉ iL) (hR : ∀ x ∈ labelsOf C, x ∉ iR)
    (hface : ∀ x ∈ iface, x ∈ iC ∨ x ∈ bnd) (σ : ℕ → ℕ) (hσ : InRange E bnd σ) :
    kL * eval E ⟨C ++ L, iC ++ iL, bnd⟩ σ = kR * eval E ⟨C ++ R, iC ++ iR, bnd⟩ σ := by
  have split : ∀ (X : List Node) (iX : List ℕ), (∀ x ∈ labelsOf C, x ∉ iX) →
      eval E ⟨C ++ X, iC ++ iX, bnd⟩ σ =
        sumOver (labDim E) iC (fun τ => prodVal E C τ *
          sumOver (labDim E) iX (prodVal E X) τ) σ := by
    intro X iX hX
    show sumOver _ (iC ++ iX) _ σ = _
    rw [sumOver_append]
    refine sumOver_congr _ _ _ _ _ fun τ _ _ => ?_
    have : prodVal E (C ++ X) = fun ρ => prodVal E C ρ * prodVal E X ρ :=
      funext (prodVal_append E C X)
    rw [this]
    exact sumOver_const_mul _ _ _ _ _ (prodVal_reads E C) hX τ
  rw [split L iL hL, split R iR hR, mul_sumOver, mul_sumOver]
  refine sumOver_congr _ _ _ _ _ fun τ hτ hr => ?_
  rw [mul_left_comm, hloc τ ?_, mul_left_comm]
  intro x hx
  by_cases hm : x ∈ iC
  · exact hr x hm
  · rw [hτ x hm]
    exact hσ x ((hface x hx).resolve_left hm)

/-- A rewrite in context: `s • (C ++ L) = s' • (C ++ R)` from `L = k • R` and `s' = s k`. -/
theorem rewrite (E : Env) (C L R : List Node) (iC iL iR bnd iface : List ℕ) (s k s' : ℂ)
    (hloc : LocalEq E L iL R iR iface 1 k)
    (hL : ∀ x ∈ labelsOf C, x ∉ iL) (hR : ∀ x ∈ labelsOf C, x ∉ iR)
    (hface : ∀ x ∈ iface, x ∈ iC ∨ x ∈ bnd) (hs : s' = s * k) :
    Equal E s ⟨C ++ L, iC ++ iL, bnd⟩ s' ⟨C ++ R, iC ++ iR, bnd⟩ := by
  refine ⟨List.Perm.refl _, fun σ hσ => ?_⟩
  have := frame E C L R iC iL iR bnd iface 1 k hloc hL hR hface σ hσ
  rw [one_mul] at this
  rw [this, hs, mul_assoc]

end ArchytasZX
