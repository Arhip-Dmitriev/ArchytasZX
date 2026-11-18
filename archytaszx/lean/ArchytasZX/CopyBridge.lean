/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.CopyPin
import ArchytasZX.Linked

/-!
# Copy-indexed phases through pinned gadgets

A linked diagram `L` whose one node `x` reads its copy index in its phase (`cph`) has a gadget
diagram `G` with no reading: `x` phaseless (phase index `z`) with one more input `w`. For a Z
spider (`GadZ`) `w` is wired from the output of a new Fourier box `f` in `x`'s boxes whose input
is the last boundary input `K`; for an X spider (`GadX`) `w` is `K` itself.

`evalL_gadZ`, `evalL_gadX`: `G`'s digit sum with slot `K`'s digit `t` at the reading `u t` of
`x` modulo `d` is `L.evalL` times `(√d)⁻¹` per copy of `x`, when the reading at `u` is the phase
`u k / d` (Z) or `-u k / d` (X). `evalL_eq_of_gad`: equal family networks of two gadget
diagrams give equal linked digit sums at every slot digits in range.
-/

namespace ArchytasZX

open Finset

/-! ## Sums and products -/

/-- Two sums over the same labels from two starts, their integrands agreeing on assignments that
agree on the labels. -/
theorem sumOver_congr2 (D : ℕ → ℕ) (ls : List ℕ) (f g : (ℕ → ℕ) → ℂ) (σ σ' : ℕ → ℕ)
    (h : ∀ τ τ', (∀ x ∈ ls, τ x = τ' x) → (∀ x ∉ ls, τ x = σ x) → (∀ x ∉ ls, τ' x = σ' x) →
      (∀ x ∈ ls, τ x < D x) → f τ = g τ') :
    sumOver D ls f σ = sumOver D ls g σ' := by
  induction ls generalizing σ σ' with
  | nil => exact h σ σ' (by simp) (fun _ _ => rfl) (fun _ _ => rfl) (by simp)
  | cons l ls ih =>
    simp only [sumOver]
    refine Finset.sum_congr rfl fun j hj => ih _ _ fun τ τ' he h1 h2 hr => h τ τ' ?_ ?_ ?_ ?_
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · by_cases hm : x ∈ ls
        · exact he x hm
        · rw [h1 x hm, h2 x hm, Function.update_self, Function.update_self]
      · exact he x hx
    · intro x hx
      rw [h1 x (fun h' => hx (List.mem_cons_of_mem _ h'))]
      exact Function.update_of_ne (fun e => hx (by rw [e]; exact List.mem_cons_self)) _ _
    · intro x hx
      rw [h2 x (fun h' => hx (List.mem_cons_of_mem _ h'))]
      exact Function.update_of_ne (fun e => hx (by rw [e]; exact List.mem_cons_self)) _ _
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · by_cases hm : x ∈ ls
        · exact hr x hm
        · rw [h1 x hm, Function.update_self]; exact Finset.mem_range.1 hj
      · exact hr x hx

/-- A product of factors, each reading its own label and labels `S` off the summed ones, summed
label by label. -/
theorem sumOver_prod_sep (D : ℕ → ℕ) {β : Type*} (ts : List β) (lab : β → ℕ)
    (hnd : (ts.map lab).Nodup) (g : β → (ℕ → ℕ) → ℂ) (S : List ℕ)
    (hS : ∀ x ∈ S, x ∉ ts.map lab) (hg : ∀ t ∈ ts, Reads (lab t :: S) (g t)) (σ : ℕ → ℕ) :
    sumOver D (ts.map lab) (fun τ => (ts.map fun t => g t τ).prod) σ =
      (ts.map fun t => ∑ j ∈ range (D (lab t)), g t (Function.update σ (lab t) j)).prod := by
  induction ts generalizing σ with
  | nil => simp [sumOver]
  | cons t ts ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
    simp only [List.map_cons, List.prod_cons, sumOver]
    have hS' : ∀ x ∈ S, x ∉ ts.map lab := fun x hx h => hS x hx (List.mem_cons_of_mem _ h)
    have hg' : ∀ t' ∈ ts, Reads (lab t' :: S) (g t') := fun t' ht' => hg t' (List.mem_cons_of_mem _ ht')
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [sumOver_const_mul D _ (lab t :: S) (g t) _ (hg t List.mem_cons_self) (fun x hx => ?_),
      ih hnd.2 hS' hg']
    · congr 1
      refine congrArg List.prod (List.map_congr_left fun t' ht' => ?_)
      refine Finset.sum_congr rfl fun j' _ => ?_
      refine hg' t' ht' _ _ fun x hx => ?_
      have hne : lab t' ≠ lab t := fun e => hnd.1 ⟨t', ht', e⟩
      by_cases hxl : x = lab t'
      · subst hxl; simp
      · have hxt : x ≠ lab t := by
          rcases List.mem_cons.1 hx with rfl | hx
          · exact absurd rfl hxl
          · exact fun e => hS x hx (by rw [e]; simp)
        rw [Function.update_of_ne hxl, Function.update_of_ne hxl, Function.update_of_ne hxt]
    · rcases List.mem_cons.1 hx with rfl | hx
      · simpa using hnd.1
      · exact hS' x hx

theorem prod_map_div {β : Type*} (l : List β) (a : β → ℂ) (c : ℂ) :
    (l.map fun t => a t / c).prod = c⁻¹ ^ l.length * (l.map a).prod := by
  induction l with
  | nil => simp
  | cons t l ih => simp only [List.map_cons, List.prod_cons, ih, List.length_cons, pow_succ]; ring

/-- A product over nodes agreeing with another off the one node of id `x`, `c` times it there. -/
theorem prod_single (l : List DNode) (x : ℕ) (hnd : (l.map (·.id)).Nodup)
    (hx : ∃ n ∈ l, n.id = x) (c : ℂ) (f g : DNode → ℂ) (h1 : ∀ n ∈ l, n.id = x → f n = c * g n)
    (h2 : ∀ n ∈ l, n.id ≠ x → f n = g n) : (l.map f).prod = c * (l.map g).prod := by
  induction l with
  | nil => simp at hx
  | cons m l ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
    simp only [List.map_cons, List.prod_cons]
    by_cases hm : m.id = x
    · have hl : ∀ n ∈ l, n.id ≠ x := fun n hn e => hnd.1 ⟨n, hn, e.trans hm.symm⟩
      rw [h1 m List.mem_cons_self hm, List.map_congr_left fun n hn =>
        h2 n (List.mem_cons_of_mem _ hn) (hl n hn)]
      ring
    · obtain ⟨n, hn, e⟩ := hx
      have hn' : n ∈ l := by
        rcases List.mem_cons.1 hn with rfl | h
        · exact absurd e hm
        · exact h
      rw [h2 m List.mem_cons_self hm,
        ih hnd.2 ⟨n, hn', e⟩ (fun n hn => h1 n (List.mem_cons_of_mem _ hn))
          (fun n hn => h2 n (List.mem_cons_of_mem _ hn))]
      ring

/-! ## Environments -/

theorem tensor_envAt (E E' : Env) (hd : E'.dimv = E.dimv) (hc : E'.coef = E.coef)
    (hm : E'.mult = E.mult) (ph : ℕ) (hp : E'.phase ph = E.phase ph) :
    ∀ (k : Kind) (d : ℕ) (os is : List ℕ), tensor E' k d ph os is = tensor E k d ph os is := by
  intro k
  induction k with
  | ZB cs => intro d os is; simp only [tensor, hp, hm]
  | XB co ci => intro d os is; simp only [tensor, hp, hm]
  | WB cs => intro d os is; rcases is with _ | ⟨_, _ | ⟨_, _⟩⟩ <;> simp only [tensor, hm]
  | Bun m b ih =>
    intro d os is
    simp only [tensor, hm]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [copyOuts, copyIns, hm]
    exact ih _ _ _
  | Z => intro d os is; simp only [tensor, hp]
  | X => intro d os is; simp only [tensor, hp]
  | ZW => intro d os is; simp only [tensor, hc]
  | XW => intro d os is; simp only [tensor, hc]
  | _ =>
    intro d os is
    rcases os with _ | ⟨_, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨_, _ | ⟨_, _⟩⟩ <;> rfl

theorem val_envAt (E E' : Env) (hd : E'.dimv = E.dimv) (hc : E'.coef = E.coef)
    (hm : E'.mult = E.mult) (σ : ℕ → ℕ) (n : Node) (hp : E'.phase n.ph = E.phase n.ph) :
    val E' σ n = val E σ n := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  by_cases hk : kind ≠ .B ∧ kind ≠ .S
  · rw [val_tensor E' σ kind hk, val_tensor E σ kind hk, hd]
    exact tensor_envAt E E' hd hc hm ph hp kind _ _ _
  · simp only [not_and_or, not_not] at hk
    rcases hk with rfl | rfl <;> simp only [val, hd] <;>
      generalize os.map σ = A <;> generalize is.map σ = B <;>
      rcases A with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;> rcases B with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;> rfl

theorem tZ_congr (d : ℕ) (φ ψ : ℕ → ℝ) (h : ∀ k, ex (φ k) = ex (ψ k)) (vs : List ℕ) :
    tZ d φ vs = tZ d ψ vs := by
  unfold tZ; simp only [h]

theorem tX_congr (d : ℕ) (φ ψ : ℕ → ℝ) (h : ∀ k, ex (φ k) = ex (ψ k)) (os is : List ℕ) :
    tX d φ os is = tX d ψ os is := by
  unfold tX; simp only [h]

/-! ## Pinned digits -/

namespace Diagram

variable {D : Diagram}

/-- The digits read through the pins: a wire's second end at its first end, a boundary leg at its
slot. -/
def pv (D : Diagram) (ks : ℕ → ℕ) (τ : ℕ → ℕ) (x : ℕ) : ℕ :=
  if x ∈ D.rdl ks then τ (D.pinD x) else τ x

theorem pv_w2 (hW : WF D) (ks : ℕ → ℕ) (τ : ℕ → ℕ) {w : Port × Port} (hw : w ∈ D.wires)
    {t : List ℕ} (ht : t ∈ cidx ks (D.legChain w.2)) : D.pv ks τ (dlab w.2 t) = τ (dlab w.1 t) := by
  have hm : dlab w.2 t ∈ D.rdl ks :=
    (D.mem_rdl ks _).2 ⟨w.2, List.mem_append_left _ (List.mem_map.2 ⟨w, hw, rfl⟩), t, ht, rfl⟩
  unfold pv
  rw [if_pos hm, pinD_wire hW hw]

theorem pv_bnd (hW : WF D) (ks : ℕ → ℕ) (τ : ℕ → ℕ) {i : ℕ} (hi : i < D.bnd.length)
    {t : List ℕ} (ht : t ∈ cidx ks (D.legChain D.bnd[i])) :
    D.pv ks τ (dlab D.bnd[i] t) = τ (slab i t) := by
  have hm : dlab D.bnd[i] t ∈ D.rdl ks :=
    (D.mem_rdl ks _).2 ⟨_, List.mem_append_right _ (List.getElem_mem hi), t, ht, rfl⟩
  unfold pv
  rw [if_pos hm, pinD_bnd hW i hi]

theorem pv_w1 (hW : WF D) (ks : ℕ → ℕ) (τ : ℕ → ℕ) {w : Port × Port} (hw : w ∈ D.wires)
    {t : List ℕ} (ht : t ∈ cidx ks (D.legChain w.1)) : D.pv ks τ (dlab w.1 t) = τ (dlab w.1 t) := by
  have hnd := (dlabs_perm hW ks).nodup_iff.1 (dlabs_nodup D ks (legs_nodup_of D hW.ids))
  have hm : dlab w.1 t ∈ D.wdl ks := (D.mem_wdl ks _).2 ⟨w, hw, t, ht, rfl⟩
  unfold pv
  rw [if_neg fun h => (List.nodup_append.1 hnd).2.2 _ hm _ h rfl]

/-- Pinned digits of the legs of `D` agree in `D'` holding `D`'s wires, its boundary first and
its legs' chains. -/
theorem pv_eq {D' : Diagram} (hW : WF D) (hW' : WF D') (ks : ℕ → ℕ)
    (hws : ∀ w ∈ D.wires, w ∈ D'.wires) (ext : List Port) (hb : D'.bnd = D.bnd ++ ext)
    (hch : ∀ p ∈ D.legs, D'.legChain p = D.legChain p) (τ τ' : ℕ → ℕ)
    (hw1 : ∀ w ∈ D.wires, ∀ s ∈ cidx ks (D.legChain w.1), τ' (dlab w.1 s) = τ (dlab w.1 s))
    (hsl : ∀ i < D.bnd.length, ∀ s, τ' (slab i s) = τ (slab i s))
    {p : Port} (hp : p ∈ D.legs) {t : List ℕ} (ht : t ∈ cidx ks (D.legChain p)) :
    D'.pv ks τ' (dlab p t) = D.pv ks τ (dlab p t) := by
  have ht' : t ∈ cidx ks (D'.legChain p) := by rw [hch p hp]; exact ht
  rcases List.mem_append.1 (hW.perm.symm.subset hp) with h | h
  · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 h
    rw [pv_w1 hW' ks τ' (hws w hw) ht', pv_w1 hW ks τ hw ht]
    exact hw1 w hw t ht
  · rcases List.mem_append.1 h with h | h
    · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 h
      rw [pv_w2 hW' ks τ' (hws w hw) ht', pv_w2 hW ks τ hw ht]
      exact hw1 w hw t (by rw [legChain_wire hW hw]; exact ht)
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 h
      have hi' : i < D'.bnd.length := by rw [hb, List.length_append]; omega
      have he : D'.bnd[i] = D.bnd[i] := by simp only [hb, List.getElem_append_left hi]
      have h3 := pv_bnd hW' ks τ' hi' (t := t) (by rw [he]; exact ht')
      rw [he] at h3
      rw [pv_bnd hW ks τ hi ht, h3]
      exact hsl i hi t

/-- The digit sum over the first ends' digits, `pinsD` read through `pv`, for any integrand. -/
theorem sum_pin (hW : WF D) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < D.bnd.length), ∀ t ∈ cidx ks (D.legChain D.bnd[i]),
      σ i t < E.dimv (D.portDim D.bnd[i])) (P : (ℕ → ℕ) → ℂ) :
    (open Classical in sumOver (fun l => E.dimv (D.portDim (dport l))) (D.dlabs ks)
      (fun ν => if D.pinsD ks ν then P ν else 0) (slotsD σ)) =
    sumOver (fun l => E.dimv (D.portDim (dport l))) (D.wdl ks) (fun τ => P (D.pv ks τ))
      (slotsD σ) := by
  have hnd := (dlabs_perm hW ks).nodup_iff.1 (dlabs_nodup D ks (legs_nodup_of D hW.ids))
  obtain ⟨-, hrnd, hdis⟩ := List.nodup_append.1 hnd
  have hpins : ∀ ν : ℕ → ℕ, D.pinsD ks ν ↔ ∀ l ∈ D.rdl ks, ν l = ν (D.pinD l) := by
    intro ν
    constructor
    · rintro ⟨hw, hb⟩ l hl
      obtain ⟨p, hp, t, ht, rfl⟩ := (D.mem_rdl ks l).1 hl
      rcases List.mem_append.1 hp with hp | hp
      · obtain ⟨w, hwm, rfl⟩ := List.mem_map.1 hp
        rw [pinD_wire hW hwm]
        exact (hw w hwm t (by rw [legChain_wire hW hwm]; exact ht)).symm
      · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
        rw [pinD_bnd hW i hi]
        exact hb (D.bnd[i], i) ((mem_zipIdx' _ _ _).2 ⟨hi, rfl⟩) t ht
    · intro h
      refine ⟨fun w hwm t ht => ?_, fun q hq t ht => ?_⟩
      · have hl : dlab w.2 t ∈ D.rdl ks := (D.mem_rdl ks _).2 ⟨w.2,
          List.mem_append_left _ (List.mem_map.2 ⟨w, hwm, rfl⟩), t,
          by rw [← legChain_wire hW hwm]; exact ht, rfl⟩
        have := h _ hl
        rw [pinD_wire hW hwm] at this
        exact this.symm
      · obtain ⟨p, i⟩ := q
        obtain ⟨hi, rfl⟩ := (mem_zipIdx' _ _ _).1 hq
        have hl : dlab D.bnd[i] t ∈ D.rdl ks := (D.mem_rdl ks _).2 ⟨D.bnd[i],
          List.mem_append_right _ (List.getElem_mem hi), t, ht, rfl⟩
        have := h _ hl
        rw [pinD_bnd hW i hi] at this
        exact this
  rw [sumOver_perm _ (dlabs_perm hW ks), sumOver_append]
  refine sumOver_congr _ _ _ _ _ fun τ hτ hr => ?_
  classical
  rw [show (fun ν => if D.pinsD ks ν then P ν else 0) =
      fun ν => if ∀ l ∈ D.rdl ks, ν l = ν (D.pinD l) then P ν else 0 from
    funext fun ν => if_congr (hpins ν) rfl rfl]
  refine sumOver_pin _ _ hrnd _ (fun l hl => ?_) _ τ (fun l hl => ?_)
  · obtain ⟨p, hp, t, ht, rfl⟩ := (D.mem_rdl ks l).1 hl
    rcases List.mem_append.1 hp with hp | hp
    · obtain ⟨w, hwm, rfl⟩ := List.mem_map.1 hp
      rw [pinD_wire hW hwm]
      exact fun h => hdis _ ((D.mem_wdl ks _).2 ⟨w, hwm, t,
        by rw [legChain_wire hW hwm]; exact ht, rfl⟩) _ h rfl
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
      rw [pinD_bnd hW i hi]
      intro h
      obtain ⟨q, -, u, -, e⟩ := (D.mem_rdl ks _).1 h
      exact dlab_ne_slab q u i t e.symm
  · obtain ⟨p, hp, t, ht, rfl⟩ := (D.mem_rdl ks l).1 hl
    rcases List.mem_append.1 hp with hp | hp
    · obtain ⟨w, hwm, rfl⟩ := List.mem_map.1 hp
      rw [pinD_wire hW hwm]
      have := hr _ ((D.mem_wdl ks _).2 ⟨w, hwm, t, by rw [legChain_wire hW hwm]; exact ht, rfl⟩)
      simpa [dport_dlab, hW.wdim w hwm] using this
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
      rw [pinD_bnd hW i hi, hτ _ fun h => by
        obtain ⟨w, -, u, -, e⟩ := (D.mem_wdl ks _).1 h
        exact dlab_ne_slab _ u i t e.symm, slotsD_slab, dport_dlab]
      exact hσ i hi t ht

/-- `evalD` over the first ends' digits read through `pv`. -/
theorem evalD_pv (hW : WF D) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < D.bnd.length), ∀ t ∈ cidx ks (D.legChain D.bnd[i]),
      σ i t < E.dimv (D.portDim D.bnd[i])) :
    D.evalD E ks σ = sumOver (fun l => E.dimv (D.portDim (dport l))) (D.wdl ks)
      (fun τ => D.prodD E ks (D.pv ks τ)) (slotsD σ) := by
  unfold evalD
  exact sum_pin hW E ks σ hσ _

/-- Copy `t` of `n` alike in two diagrams whose tails agree on `n`'s legs. -/
theorem dnode_of_tail {D' : Diagram} (ks : ℕ → ℕ) (n : DNode) (t : List ℕ)
    (h : ∀ p ∈ n.legs, D'.tail p = D.tail p) : D'.dnode ks n t = D.dnode ks n t := by
  simp only [dnode, Node.mk.injEq, true_and, and_true]
  constructor
  · refine List.flatMap_congr fun i hi => ?_
    rw [h ⟨n.id, true, i⟩ (by simp [DNode.legs]; exact List.mem_range.1 hi)]
  · refine List.flatMap_congr fun i hi => ?_
    rw [h ⟨n.id, false, i⟩ (by simp [DNode.legs]; exact List.mem_range.1 hi)]

end Diagram

/-! ## Gadgets -/

namespace LDiagram

/-- `n` with phase index `z` and one more input of its dimension. -/
def gnode (z : ℕ) (n : DNode) : DNode := { n with ph := z, ins := n.ins ++ [n.dim] }

/-- The nodes of `L` with the node of id `x` through `gnode z`. -/
def gnodes (L : LDiagram) (x z : ℕ) : List DNode :=
  L.base.nodes.map fun n => if n.id = x then gnode z n else n

/-- The reading of the only copy-indexed phase: depth of its box, coefficient, multiplicity
index. -/
def rdOf (L : LDiagram) : ℕ × ℕ × ℕ :=
  match L.cph with
  | [q] => (L.depth q.2.1 q.1, q.2.2.1, q.2.2.2)
  | _ => (0, 0, 0)

/-- The copy index the only copy-indexed phase reads at copy `t`. -/
def rdU (L : LDiagram) (ks : ℕ → ℕ) (t : List ℕ) : ℕ :=
  t.getD L.rdOf.1 0 + L.rdOf.2.1 * ks L.rdOf.2.2

theorem phAt_rd (L : LDiagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ)
    (h : L.cph.map (·.1) = [n.id]) : L.phAt ks n t = Nat.pair n.ph (L.rdU ks t) := by
  obtain ⟨q, hc⟩ : ∃ q, L.cph = [q] := by
    rcases hc : L.cph with _ | ⟨q, _ | ⟨q', r⟩⟩ <;> simp_all
  rw [hc] at h
  simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at h
  simp [phAt, rdU, rdOf, hc, h]

theorem phAt_off (L : LDiagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) (x : ℕ)
    (h : L.cph.map (·.1) = [x]) (hn : n.id ≠ x) : L.phAt ks n t = n.ph := by
  obtain ⟨q, hc⟩ : ∃ q, L.cph = [q] := by
    rcases hc : L.cph with _ | ⟨q, _ | ⟨q', r⟩⟩ <;> simp_all
  rw [hc] at h
  simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at h
  have : ¬ (q.1 = n.id) := fun e => hn (e.symm.trans h)
  simp [phAt, hc, this]

/-- Conditions shared by both gadgets: no links or other readings, one copy-indexed phase, on
`x`; `G` keeps `L`'s outputs, its legs' tails and dimensions and its nodes' chains, and `x`'s
new input has no tail. -/
def GadBase (L : LDiagram) (G : Diagram) (x : DNode) : Prop :=
  L.links = [] ∧ L.cdim = [] ∧ L.aligned = [] ∧ L.cph.map (·.1) = [x.id] ∧ x ∈ L.base.nodes ∧
  G.outs = L.base.outs ∧
  (∀ p ∈ L.base.legs, G.tail p = L.base.tail p ∧ G.portDim p = L.base.portDim p) ∧
  (∀ n ∈ L.base.nodes, G.nodeChain n.id = L.base.nodeChain n.id) ∧
  G.tail ⟨x.id, false, x.ins.length⟩ = [] ∧ G.portDim ⟨x.id, false, x.ins.length⟩ = x.dim

instance (L : LDiagram) (G : Diagram) (x : DNode) : Decidable (GadBase L G x) := by
  unfold GadBase; infer_instance

/-- The Z gadget: the Fourier box `f` from slot `K = ⟨f, false, 0⟩` into `x`'s new input, its
wire at place `i`. -/
def GadZ (L : LDiagram) (G : Diagram) (x : DNode) (f z i : ℕ) : Prop :=
  GadBase L G x ∧ x.kind = .Z ∧
  G.nodes = L.gnodes x.id z ++ [⟨f, .F, x.dim, z, [x.dim], [x.dim]⟩] ∧
  G.wires = L.base.wires.take i ++
    (⟨f, true, 0⟩, ⟨x.id, false, x.ins.length⟩) :: L.base.wires.drop i ∧
  G.ins = L.base.ins ++ [⟨f, false, 0⟩] ∧ G.nodeChain f = L.base.nodeChain x.id ∧
  G.tail ⟨f, true, 0⟩ = [] ∧ G.tail ⟨f, false, 0⟩ = [] ∧
  G.portDim ⟨f, true, 0⟩ = x.dim ∧ G.portDim ⟨f, false, 0⟩ = x.dim

instance (L : LDiagram) (G : Diagram) (x : DNode) (f z i : ℕ) : Decidable (GadZ L G x f z i) := by
  unfold GadZ; infer_instance

/-- The X gadget: `x`'s new input is slot `K`. -/
def GadX (L : LDiagram) (G : Diagram) (x : DNode) (z : ℕ) : Prop :=
  GadBase L G x ∧ x.kind = .X ∧ G.nodes = L.gnodes x.id z ∧ G.wires = L.base.wires ∧
  G.ins = L.base.ins ++ [⟨x.id, false, x.ins.length⟩]

instance (L : LDiagram) (G : Diagram) (x : DNode) (z : ℕ) : Decidable (GadX L G x z) := by
  unfold GadX; infer_instance

/-! ## Facts of the gadgets -/

theorem gnode_id (z : ℕ) (n : DNode) : (gnode z n).id = n.id := rfl

theorem leg_of_node {L : LDiagram} {n : DNode} (hn : n ∈ L.base.nodes) {p : Port}
    (hp : p ∈ n.legs) : p ∈ L.base.legs := List.mem_flatMap.2 ⟨n, hn, hp⟩

theorem GadBase.legChain {L : LDiagram} {G : Diagram} {x : DNode} (h : GadBase L G x)
    (hW : WF L.base) {p : Port} (hp : p ∈ L.base.legs) : G.legChain p = L.base.legChain p := by
  obtain ⟨-, -, -, -, -, -, hl, hnc, -⟩ := h
  have hpn := Diagram.node_of_leg hp
  obtain ⟨n, hn, e⟩ := List.mem_map.1 hpn
  unfold Diagram.legChain
  rw [(hl p hp).1, ← e, hnc n hn]

theorem GadBase.chainK {L : LDiagram} {G : Diagram} {x : DNode} (h : GadBase L G x) :
    G.legChain ⟨x.id, false, x.ins.length⟩ = L.base.nodeChain x.id := by
  obtain ⟨-, -, -, -, hx, -, -, hnc, hnt, -⟩ := h
  unfold Diagram.legChain
  rw [hnt, List.append_nil]
  exact hnc x hx

/-- Copy `t` of the gadget node: `x`'s copy with phase index `z` and the new input's digit. -/
theorem GadBase.dnode {L : LDiagram} {G : Diagram} {x : DNode} (h : GadBase L G x) (z : ℕ)
    (ks : ℕ → ℕ) (t : List ℕ) :
    G.dnode ks (gnode z x) t = ⟨x.kind, (L.base.dnode ks x t).outs,
      (L.base.dnode ks x t).ins ++ [dlab ⟨x.id, false, x.ins.length⟩ t], x.dim, z⟩ := by
  obtain ⟨-, -, -, -, hx, -, hl, -, hnt, -⟩ := h
  have hr : rcidx ks [] = [[]] := by simp [rcidx, cidx]
  simp only [Diagram.dnode, gnode, List.length_append, List.length_singleton, List.range_succ,
    List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, hnt, hr,
    List.map_cons, List.map_nil, Node.mk.injEq, true_and, and_true]
  constructor
  · refine List.flatMap_congr fun i hi => ?_
    rw [(hl _ (Diagram.leg_mem hx true i (by simpa [Diagram.side] using List.mem_range.1 hi))).1]
  · congr 1
    refine List.flatMap_congr fun i hi => ?_
    rw [(hl _ (Diagram.leg_mem hx false i (by simpa [Diagram.side] using List.mem_range.1 hi))).1]

/-- The values of the nodes off `x` in the gadget and in `L`. -/
theorem GadBase.fac_off {L : LDiagram} {G : Diagram} {x : DNode} (h : GadBase L G x)
    (hW : WF L.base) (hWG : WF G) (hws : ∀ w ∈ L.base.wires, w ∈ G.wires) (ext : List Port)
    (hb : G.bnd = L.base.bnd ++ ext) (E E' : Env) (hd : E'.dimv = E.dimv) (hc : E'.coef = E.coef)
    (hm : E'.mult = E.mult) (hph : ∀ n ∈ L.base.nodes, n.id ≠ x.id → E'.phase n.ph = E.phase n.ph)
    (ks : ℕ → ℕ) (τ τ' : ℕ → ℕ)
    (hw1 : ∀ w ∈ L.base.wires, ∀ s ∈ cidx ks (L.base.legChain w.1), τ (dlab w.1 s) = τ' (dlab w.1 s))
    (hsl : ∀ i < L.base.bnd.length, ∀ s, τ (slab i s) = τ' (slab i s))
    (n : DNode) (hn : n ∈ L.base.nodes) (hne : n.id ≠ x.id) :
    ((cidx ks (G.nodeChain n.id)).map fun t => val E (G.pv ks τ) (G.dnode ks n t)).prod =
      ((cidx ks (L.base.nodeChain n.id)).map fun t => val E' (L.base.pv ks τ') (L.dnodeL ks n t)).prod := by
  have h' := h
  obtain ⟨-, -, -, hcp, -, -, hl, hnc, -⟩ := h'
  rw [hnc n hn]
  refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
  have hdl : L.dnodeL ks n t = L.base.dnode ks n t := by
    simp only [dnodeL, phAt_off L ks n t x.id hcp hne]; rfl
  rw [hdl, Diagram.dnode_of_tail ks n t fun p hp => (hl p (leg_of_node hn hp)).1,
    val_envAt E E' hd hc hm _ _ (hph n hn hne)]
  refine val_reads E _ _ _ fun l hl' => ?_
  obtain ⟨p, hp, s, hs, rfl⟩ := (mem_dlabs _ _ _).1 (Diagram.dnode_labels_mem ks hn ht hl')
  exact Diagram.pv_eq hW hWG ks hws ext hb (fun q hq => h.legChain hW hq) τ' τ hw1 hsl hp hs

/-- The gadget's slot digits in range. -/
theorem range_ext {L : LDiagram} {G : Diagram} {x : DNode} (h : GadBase L G x) (hW : WF L.base)
    (K : Port) (hb : G.bnd = L.base.bnd ++ [K]) (hK : G.legChain K = L.base.nodeChain x.id)
    (hKd : G.portDim K = x.dim) (E : Env) (ks : ℕ → ℕ) (σ ρ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t < E.dimv (L.base.portDim L.base.bnd[i]))
    (hρ : ∀ i < L.base.bnd.length, ∀ t, ρ i t = σ i t)
    (hρK : ∀ t ∈ cidx ks (L.base.nodeChain x.id), ρ L.base.bnd.length t < E.dimv x.dim) :
    ∀ i (hi : i < G.bnd.length), ∀ t ∈ cidx ks (G.legChain G.bnd[i]),
      ρ i t < E.dimv (G.portDim G.bnd[i]) := by
  obtain ⟨-, -, -, -, -, -, hl, -⟩ := id h
  intro i hi t ht
  have hlen : G.bnd.length = L.base.bnd.length + 1 := by rw [hb]; simp
  by_cases hi' : i < L.base.bnd.length
  · have he : G.bnd[i] = L.base.bnd[i] := by
      rw [List.getElem_of_eq hb, List.getElem_append_left hi']
    have hp := Diagram.bnd_legs hW (List.getElem_mem hi')
    rw [he, h.legChain hW hp] at ht
    rw [he, (hl _ hp).2, hρ i hi' t]
    exact hσ i hi' t ht
  · have hiK : i = L.base.bnd.length := by omega
    subst hiK
    have he : G.bnd[L.base.bnd.length] = K := by
      rw [List.getElem_of_eq hb, List.getElem_append_right (by omega)]; simp
    rw [he, hK] at ht
    rw [he, hKd]
    exact hρK t ht

/-- `L.evalL` over the first ends' digits read through `pv`. -/
theorem evalL_pv {L : LDiagram} (hl : L.links = []) (hW : WF L.base) (E : Env) (ks : ℕ → ℕ)
    (σ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t < E.dimv (L.base.portDim L.base.bnd[i])) :
    L.evalL E ks σ = sumOver (fun l => E.dimv (L.base.portDim (dport l))) (L.base.wdl ks)
      (fun τ => L.prodL E ks (L.base.pv ks τ)) (slotsD σ) := by
  have hq : ∀ ν, L.pinsL ks ν ↔ L.base.pinsD ks ν := fun ν => by
    simp only [pinsL, Diagram.pinsD, dpairs, hl, List.flatMap_nil, List.append_nil,
      List.mem_flatMap, List.mem_map]
    refine and_congr ⟨fun h w hw t ht => h _ ⟨w, hw, t, ht, rfl⟩, ?_⟩ Iff.rfl
    rintro h _ ⟨w, hw, t, ht, rfl⟩
    exact h w hw t ht
  unfold evalL
  classical
  rw [show (fun ν => if L.pinsL ks ν then L.prodL E ks ν else 0) =
      fun ν => if L.base.pinsD ks ν then L.prodL E ks ν else 0 from
    funext fun ν => if_congr (hq ν) rfl rfl]
  exact Diagram.sum_pin hW E ks σ hσ _

theorem slab_not_wdl (D : Diagram) (ks : ℕ → ℕ) (i : ℕ) (s : List ℕ) : slab i s ∉ D.wdl ks := by
  intro h
  obtain ⟨w, -, u, -, e⟩ := (D.mem_wdl ks _).1 h
  exact dlab_ne_slab _ u i s e.symm

/-! ## The X gadget -/

theorem evalL_gadX {L : LDiagram} {G : Diagram} {x : DNode} {z : ℕ} (h : GadX L G x z)
    (hW : WF L.base) (hWG : WF G) (E E' : Env) (hd : E'.dimv = E.dimv) (hc : E'.coef = E.coef)
    (hm : E'.mult = E.mult) (hph : ∀ n ∈ L.base.nodes, n.id ≠ x.id → E'.phase n.ph = E.phase n.ph)
    (hz : ∀ k, E.phase z k = 0) (hpos : 0 < E.dimv x.dim)
    (hx : ∀ u k, ex (E'.phase (Nat.pair x.ph u) k) = ex (-((u : ℝ) * k / E.dimv x.dim)))
    (ks : ℕ → ℕ) (σ ρ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t < E.dimv (L.base.portDim L.base.bnd[i]))
    (hρ : ∀ i < L.base.bnd.length, ∀ t, ρ i t = σ i t)
    (hK : ∀ t ∈ cidx ks (L.base.nodeChain x.id),
      ρ L.base.bnd.length t = L.rdU ks t % E.dimv x.dim) :
    G.evalD E ks ρ = (rt (E.dimv x.dim))⁻¹ ^ size ks (L.base.nodeChain x.id) * L.evalL E' ks σ := by
  obtain ⟨hb, hk, hn, hw, hi⟩ := h
  have hb' := hb
  obtain ⟨hl0, -, -, hcp, hxm, ho, hlegs, hnc, -, hKd⟩ := hb'
  set d := E.dimv x.dim with hdd
  set K : Port := ⟨x.id, false, x.ins.length⟩
  have hbnd : G.bnd = L.base.bnd ++ [K] := by simp [Diagram.bnd, ho, hi, K]
  have hσ' : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t < E'.dimv (L.base.portDim L.base.bnd[i]) := by rw [hd]; exact hσ
  have hρG := range_ext hb hW K hbnd hb.chainK hKd E ks σ ρ hσ hρ fun t ht => by
    rw [hK t ht]; exact Nat.mod_lt _ hpos
  have hwdl : G.wdl ks = L.base.wdl ks := by
    unfold Diagram.wdl
    rw [hw]
    exact List.flatMap_congr fun w hwm => by rw [hb.legChain hW (Diagram.wire_not_bnd hW hwm).2.2.1]
  have hKi : L.base.bnd.length < G.bnd.length := by rw [hbnd]; simp
  have hKe : G.bnd[L.base.bnd.length] = K := by
    rw [List.getElem_of_eq hbnd, List.getElem_append_right (by omega)]; simp
  rw [Diagram.evalD_pv hWG E ks ρ hρG, evalL_pv hl0 hW E' ks σ hσ', hwdl, mul_sumOver,
    sumOver_dim_congr _ (fun l => E'.dimv (L.base.portDim (dport l))) _ fun l hl => by
      obtain ⟨w, hwm, s, -, rfl⟩ := (L.base.mem_wdl ks _).1 hl
      rw [dport_dlab, hd, (hlegs _ (Diagram.wire_not_bnd hW hwm).2.2.1).2]]
  refine sumOver_congr2 _ _ _ _ _ _ fun τ τ' he h1 h2 _ => ?_
  have hw1 : ∀ w ∈ L.base.wires, ∀ s ∈ cidx ks (L.base.legChain w.1),
      τ (dlab w.1 s) = τ' (dlab w.1 s) := fun w hwm s hs =>
    he _ ((L.base.mem_wdl ks _).2 ⟨w, hwm, s, hs, rfl⟩)
  have hsl : ∀ i < L.base.bnd.length, ∀ s, τ (slab i s) = τ' (slab i s) := fun i hi s => by
    rw [h1 _ (slab_not_wdl _ ks i s), h2 _ (slab_not_wdl _ ks i s), slotsD_slab, slotsD_slab,
      hρ i hi s]
  have hws : ∀ w ∈ L.base.wires, w ∈ G.wires := fun w hwm => hw ▸ hwm
  unfold Diagram.prodD prodL
  rw [hn, gnodes, List.map_map]
  refine prod_single _ x.id hW.ids ⟨x, hxm, rfl⟩ _ _ _ (fun n hnm e => ?_) (fun n hnm hne => ?_)
  · have hnx : n = x := List.inj_on_of_nodup_map hW.ids hnm hxm e
    subst hnx
    simp only [Function.comp, ↓reduceIte, gnode_id]
    rw [hnc n hxm, ← cidx_length ks, ← prod_map_div]
    refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
    rw [hb.dnode z ks t, dnodeL, phAt_rd L ks n t hcp]
    have hKv : G.pv ks τ (dlab K t) = L.rdU ks t % d := by
      have := Diagram.pv_bnd hWG ks τ hKi (t := t) (by rw [hKe, hb.chainK]; exact ht)
      rw [hKe] at this
      rw [this, h1 _ (slab_not_wdl _ ks _ t), slotsD_slab, hK t ht]
    have hlab : ∀ l ∈ (L.base.dnode ks n t).labels, G.pv ks τ l = L.base.pv ks τ' l := by
      intro l hl'
      obtain ⟨p, hp, s, hs, rfl⟩ := (mem_dlabs _ _ _).1 (Diagram.dnode_labels_mem ks hxm ht hl')
      exact Diagram.pv_eq hW hWG ks hws [K] hbnd (fun q hq => hb.legChain hW hq) τ' τ hw1 hsl hp hs
    have ho' : (L.base.dnode ks n t).outs.map (G.pv ks τ) =
        (L.base.dnode ks n t).outs.map (L.base.pv ks τ') :=
      List.map_congr_left fun l hl' => hlab l (List.mem_append_left _ hl')
    have hi' : (L.base.dnode ks n t).ins.map (G.pv ks τ) =
        (L.base.dnode ks n t).ins.map (L.base.pv ks τ') :=
      List.map_congr_left fun l hl' => hlab l (List.mem_append_right _ hl')
    simp only [val, Diagram.dnode, hk] at ho' hi' ⊢
    rw [List.map_append, ho', hi', List.map_singleton, hKv, hd,
      show E.phase z = fun _ => 0 from funext hz, tX_pin]
    congr 1
    refine tX_congr _ _ _ (fun k => ?_) _ _
    rw [hx, ← ex_inv, ← ex_inv, ex_mod d _ k hpos]
  · have e : (if n.id = x.id then gnode z n else n) = n := if_neg hne
    simp only [Function.comp, e]
    exact hb.fac_off hW hWG hws [K] hbnd E E' hd hc hm hph ks τ τ' hw1 hsl n hnm hne

/-! ## The Z gadget -/

theorem gnodes_ids (L : LDiagram) (x z : ℕ) :
    (L.gnodes x z).map (·.id) = L.base.nodes.map (·.id) := by
  unfold gnodes
  rw [List.map_map]
  refine List.map_congr_left fun n _ => ?_
  simp only [Function.comp]
  split <;> rfl

theorem evalL_gadZ {L : LDiagram} {G : Diagram} {x : DNode} {f z i : ℕ} (h : GadZ L G x f z i)
    (hW : WF L.base) (hWG : WF G) (E E' : Env) (hd : E'.dimv = E.dimv) (hc : E'.coef = E.coef)
    (hm : E'.mult = E.mult) (hph : ∀ n ∈ L.base.nodes, n.id ≠ x.id → E'.phase n.ph = E.phase n.ph)
    (hz : ∀ k, E.phase z k = 0) (hpos : 0 < E.dimv x.dim)
    (hx : ∀ u k, ex (E'.phase (Nat.pair x.ph u) k) = ex ((u : ℝ) * k / E.dimv x.dim))
    (ks : ℕ → ℕ) (σ ρ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t < E.dimv (L.base.portDim L.base.bnd[i]))
    (hρ : ∀ i < L.base.bnd.length, ∀ t, ρ i t = σ i t)
    (hK : ∀ t ∈ cidx ks (L.base.nodeChain x.id),
      ρ L.base.bnd.length t = L.rdU ks t % E.dimv x.dim) :
    G.evalD E ks ρ = (rt (E.dimv x.dim))⁻¹ ^ size ks (L.base.nodeChain x.id) * L.evalL E' ks σ := by
  obtain ⟨hb, hk, hn, hw, hi, hfc, hft, hfi, hfd, hfid⟩ := h
  have hb' := hb
  obtain ⟨hl0, -, -, hcp, hxm, ho, hlegs, hnc, hnt, -⟩ := hb'
  set d := E.dimv x.dim with hdd
  set K : Port := ⟨f, false, 0⟩ with hKdef
  set fo : Port := ⟨f, true, 0⟩ with hfodef
  set xw : Port := ⟨x.id, false, x.ins.length⟩ with hxwdef
  set C := L.base.nodeChain x.id with hC
  have hbnd : G.bnd = L.base.bnd ++ [K] := by simp [Diagram.bnd, ho, hi, K]
  have hKch : G.legChain K = C := by
    unfold Diagram.legChain; rw [hfi, List.append_nil]; exact hfc
  have hfoch : G.legChain fo = C := by
    unfold Diagram.legChain; rw [hft, List.append_nil]; exact hfc
  have hσ' : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t < E'.dimv (L.base.portDim L.base.bnd[i]) := by rw [hd]; exact hσ
  have hρG := range_ext hb hW K hbnd hKch hfid E ks σ ρ hσ hρ fun t ht => by
    rw [hK t ht]; exact Nat.mod_lt _ hpos
  have hgw : (fo, xw) ∈ G.wires := by rw [hw]; simp
  have hws : ∀ w ∈ L.base.wires, w ∈ G.wires := fun w hwm => by
    rw [hw]
    rw [← List.take_append_drop i L.base.wires] at hwm
    rcases List.mem_append.1 hwm with h' | h'
    · exact List.mem_append_left _ h'
    · exact List.mem_append_right _ (List.mem_cons_of_mem _ h')
  have hfL : f ∉ L.base.nodes.map (·.id) := by
    have := hWG.ids
    rw [hn, List.map_append, gnodes_ids] at this
    exact fun h' => (List.nodup_append.1 this).2.2 _ h' _ (by simp) rfl
  have hfoL : ∀ p ∈ L.base.legs, p ≠ fo := fun p hp e =>
    hfL (by have := Diagram.node_of_leg hp; rw [e] at this; exact this)
  set fd := (cidx ks C).map (dlab fo) with hfd'
  have hwdl : (G.wdl ks).Perm (L.base.wdl ks ++ fd) := by
    unfold Diagram.wdl
    rw [hw]
    have e1 := (List.perm_middle (a := (fo, xw)) (l₁ := L.base.wires.take i)
      (l₂ := L.base.wires.drop i)).flatMap_right fun w => (cidx ks (G.legChain w.1)).map (dlab w.1)
    rw [List.flatMap_cons, List.take_append_drop] at e1
    refine e1.trans (List.perm_append_comm.trans ?_)
    rw [show (fo, xw).1 = fo from rfl, hfoch]
    refine List.Perm.append_right _ (List.Perm.of_eq (List.flatMap_congr fun w hwm => ?_))
    rw [hb.legChain hW (Diagram.wire_not_bnd hW hwm).2.2.1]
  have hKi : L.base.bnd.length < G.bnd.length := by rw [hbnd]; simp
  have hKe : G.bnd[L.base.bnd.length] = K := by
    rw [List.getElem_of_eq hbnd, List.getElem_append_right (by omega)]; simp
  have hfdnd : fd.Nodup := (cidx_nodup ks C).map fun a b e => (dlab_inj e).2
  rw [Diagram.evalD_pv hWG E ks ρ hρG, sumOver_perm _ hwdl, sumOver_append,
    evalL_pv hl0 hW E' ks σ hσ', mul_sumOver,
    sumOver_dim_congr _ (fun l => E'.dimv (L.base.portDim (dport l))) _ fun l hl => by
      obtain ⟨w, hwm, s, -, rfl⟩ := (L.base.mem_wdl ks _).1 hl
      rw [dport_dlab, hd, (hlegs _ (Diagram.wire_not_bnd hW hwm).2.2.1).2]]
  refine sumOver_congr2 _ _ _ _ _ _ fun τ τ' he h1 h2 _ => ?_
  have hw1 : ∀ w ∈ L.base.wires, ∀ s ∈ cidx ks (L.base.legChain w.1),
      τ (dlab w.1 s) = τ' (dlab w.1 s) := fun w hwm s hs =>
    he _ ((L.base.mem_wdl ks _).2 ⟨w, hwm, s, hs, rfl⟩)
  have hsl : ∀ i < L.base.bnd.length, ∀ s, τ (slab i s) = τ' (slab i s) := fun i hi s => by
    rw [h1 _ (slab_not_wdl _ ks i s), h2 _ (slab_not_wdl _ ks i s), slotsD_slab, slotsD_slab,
      hρ i hi s]
  have hnfd : ∀ p ∈ L.base.legs, ∀ s, dlab p s ∉ fd := fun p hp s hm' => by
    obtain ⟨u, -, e⟩ := List.mem_map.1 hm'
    exact hfoL p hp (dlab_inj e).1.symm
  have hsfd : ∀ j s, slab j s ∉ fd := fun j s hm' => by
    obtain ⟨u, -, e⟩ := List.mem_map.1 hm'
    exact dlab_ne_slab _ _ _ _ e
  obtain ⟨vs, hvs⟩ : ∃ vs : List ℕ → List ℕ, ∀ t, vs t =
      (L.base.dnode ks x t).outs.map (L.base.pv ks τ') ++
        (L.base.dnode ks x t).ins.map (L.base.pv ks τ') := ⟨_, fun _ => rfl⟩
  obtain ⟨Rr, hRr⟩ : ∃ Rr : ℂ, Rr = (L.base.nodes.map fun n => if n.id = x.id then 1 else
      ((cidx ks (L.base.nodeChain n.id)).map fun t =>
        val E' (L.base.pv ks τ') (L.dnodeL ks n t)).prod).prod := ⟨_, rfl⟩
  have hL : L.prodL E' ks (L.base.pv ks τ') =
      ((cidx ks C).map fun t => tZ d (E'.phase (Nat.pair x.ph (L.rdU ks t))) (vs t)).prod * Rr := by
    unfold prodL
    rw [hRr]
    refine prod_single _ x.id hW.ids ⟨x, hxm, rfl⟩ _ _ _ (fun n hnm e => ?_) (fun n hnm hne => ?_)
    · have hnx : n = x := List.inj_on_of_nodup_map hW.ids hnm hxm e
      subst hnx
      rw [if_pos rfl, mul_one]
      refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
      rw [dnodeL, phAt_rd L ks n t hcp, hvs]
      simp only [val, Diagram.dnode, hk, hd]
      rfl
    · rw [if_neg hne]
  have hG : ∀ τ'' : ℕ → ℕ, (∀ y ∉ fd, τ'' y = τ y) → G.prodD E ks (G.pv ks τ'') =
      Rr * ((cidx ks C).map fun t => tZ d (fun _ => 0) (vs t ++ [τ'' (dlab fo t)]) *
        fou d (τ'' (dlab fo t)) (L.rdU ks t % d)).prod := by
    intro τ'' hτ''
    have hw1' : ∀ w ∈ L.base.wires, ∀ s ∈ cidx ks (L.base.legChain w.1),
        τ'' (dlab w.1 s) = τ' (dlab w.1 s) := fun w hwm s hs => by
      rw [hτ'' _ (hnfd _ (Diagram.wire_not_bnd hW hwm).2.2.1 s)]; exact hw1 w hwm s hs
    have hsl' : ∀ i < L.base.bnd.length, ∀ s, τ'' (slab i s) = τ' (slab i s) := fun i hi s => by
      rw [hτ'' _ (hsfd i s)]; exact hsl i hi s
    have hlab : ∀ t ∈ cidx ks C, ∀ l ∈ (L.base.dnode ks x t).labels,
        G.pv ks τ'' l = L.base.pv ks τ' l := by
      intro t ht l hl'
      obtain ⟨p, hp, s, hs, rfl⟩ :=
        (mem_dlabs _ _ _).1 (Diagram.dnode_labels_mem ks hxm ht hl')
      exact Diagram.pv_eq hW hWG ks hws [K] hbnd (fun q hq => hb.legChain hW hq) τ' τ'' hw1' hsl'
        hp hs
    unfold Diagram.prodD
    rw [hn, List.map_append, List.prod_append, gnodes, List.map_map, List.map_singleton,
      List.prod_singleton]
    rw [prod_single _ x.id hW.ids ⟨x, hxm, rfl⟩
      ((cidx ks C).map fun t => tZ d (fun _ => 0) (vs t ++ [τ'' (dlab fo t)])).prod _
      (fun n => if n.id = x.id then 1 else ((cidx ks (L.base.nodeChain n.id)).map fun t =>
        val E' (L.base.pv ks τ') (L.dnodeL ks n t)).prod) (fun n hnm e => ?_) (fun n hnm hne => ?_)]
    · rw [← hRr]
      have hF : ((cidx ks (G.nodeChain f)).map fun t =>
          val E (G.pv ks τ'') (G.dnode ks ⟨f, .F, x.dim, z, [x.dim], [x.dim]⟩ t)).prod =
          ((cidx ks C).map fun t => fou d (τ'' (dlab fo t)) (L.rdU ks t % d)).prod := by
        rw [hfc]
        refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        have hr : rcidx ks [] = [[]] := by simp [rcidx, cidx]
        have hfo' : G.pv ks τ'' (dlab fo t) = τ'' (dlab fo t) :=
          Diagram.pv_w1 hWG ks τ'' hgw (by rw [show (fo, xw).1 = fo from rfl, hfoch]; exact ht)
        have hK' : G.pv ks τ'' (dlab K t) = L.rdU ks t % d := by
          have := Diagram.pv_bnd hWG ks τ'' hKi (t := t) (by rw [hKe, hKch]; exact ht)
          rw [hKe] at this
          rw [this, hτ'' _ (hsfd _ t), h1 _ (slab_not_wdl _ ks _ t), slotsD_slab, hK t ht]
        simp only [Diagram.dnode, List.length_singleton, List.range_one, List.flatMap_cons,
          List.flatMap_nil, List.append_nil]
        rw [← hfodef, ← hKdef, hft, hfi, hr]
        simp only [List.map_cons, List.map_nil, List.append_nil, val, hfo', hK']
        rfl
      rw [hF, mul_comm Rr, mul_assoc, mul_comm Rr, ← mul_assoc, ← List.prod_map_mul]
    · have hnx : n = x := List.inj_on_of_nodup_map hW.ids hnm hxm e
      subst hnx
      simp only [Function.comp, ↓reduceIte, gnode_id, mul_one]
      rw [hnc n hxm]
      refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
      rw [hb.dnode z ks t]
      have hxw' : G.pv ks τ'' (dlab xw t) = τ'' (dlab fo t) :=
        Diagram.pv_w2 hWG ks τ'' hgw (by rw [show (fo, xw).2 = xw from rfl, hb.chainK]; exact ht)
      have ho' : (L.base.dnode ks n t).outs.map (G.pv ks τ'') =
          (L.base.dnode ks n t).outs.map (L.base.pv ks τ') :=
        List.map_congr_left fun l hl' => hlab t ht l (List.mem_append_left _ hl')
      have hi' : (L.base.dnode ks n t).ins.map (G.pv ks τ'') =
          (L.base.dnode ks n t).ins.map (L.base.pv ks τ') :=
        List.map_congr_left fun l hl' => hlab t ht l (List.mem_append_right _ hl')
      have hval : val E (G.pv ks τ'') ⟨n.kind, (L.base.dnode ks n t).outs,
          (L.base.dnode ks n t).ins ++ [dlab xw t], n.dim, z⟩ =
          tZ d (E.phase z) ((L.base.dnode ks n t).outs.map (G.pv ks τ'') ++
            ((L.base.dnode ks n t).ins.map (G.pv ks τ'') ++ [G.pv ks τ'' (dlab xw t)])) := by
        simp only [val, hk, List.map_append, List.map_singleton]
        rfl
      rw [hval, ho', hi', hxw', ← List.append_assoc, ← hvs,
        show E.phase z = fun _ => 0 from funext hz]
    · have e : (if n.id = x.id then gnode z n else n) = n := if_neg hne
      simp only [Function.comp, e, if_neg hne]
      exact hb.fac_off hW hWG hws [K] hbnd E E' hd hc hm hph ks τ'' τ' hw1' hsl' n hnm hne
  rw [sumOver_congr _ _ _ _ _ fun τ'' hτ'' _ => hG τ'' hτ'', ← mul_sumOver,
    sumOver_prod_sep _ (cidx ks C) (dlab fo) hfdnd _ [] (by simp) (fun t _ τ₁ τ₂ he' => by
      simp only [he' _ List.mem_cons_self]), hL, ← cidx_length ks, mul_comm (List.prod _) Rr,
    mul_left_comm, ← prod_map_div]
  congr 1
  refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
  simp only [Function.update_self, dport_dlab, hfd]
  rw [← hdd, tZ_pin]
  congr 1
  refine tZ_congr _ _ _ (fun k => ?_) _
  rw [hx, ex_mod d _ k hpos]

/-! ## Both gadgets -/

/-- `G` is a Z or an X gadget of `L` at `x`. -/
def Gad (L : LDiagram) (G : Diagram) (x : DNode) (f z i : ℕ) : Prop :=
  GadZ L G x f z i ∨ GadX L G x z

instance (L : LDiagram) (G : Diagram) (x : DNode) (f z i : ℕ) : Decidable (Gad L G x f z i) := by
  unfold Gad; infer_instance

/-- The sign of the phase a gadget of kind `k` pins: `1` for a Z spider, `-1` otherwise. -/
def gsg (k : Kind) : ℝ := if k = .Z then 1 else -1

theorem evalL_gad {L : LDiagram} {G : Diagram} {x : DNode} {f z i : ℕ} (h : Gad L G x f z i)
    (hW : WF L.base) (hWG : WF G) (E E' : Env) (hd : E'.dimv = E.dimv) (hc : E'.coef = E.coef)
    (hm : E'.mult = E.mult) (hph : ∀ n ∈ L.base.nodes, n.id ≠ x.id → E'.phase n.ph = E.phase n.ph)
    (hz : ∀ k, E.phase z k = 0) (hpos : 0 < E.dimv x.dim)
    (hx : ∀ u k, ex (E'.phase (Nat.pair x.ph u) k) = ex (gsg x.kind * ((u : ℝ) * k / E.dimv x.dim)))
    (ks : ℕ → ℕ) (σ ρ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t < E.dimv (L.base.portDim L.base.bnd[i]))
    (hρ : ∀ i < L.base.bnd.length, ∀ t, ρ i t = σ i t)
    (hK : ∀ t ∈ cidx ks (L.base.nodeChain x.id),
      ρ L.base.bnd.length t = L.rdU ks t % E.dimv x.dim) :
    G.evalD E ks ρ = (rt (E.dimv x.dim))⁻¹ ^ size ks (L.base.nodeChain x.id) * L.evalL E' ks σ := by
  rcases h with h | h
  · refine evalL_gadZ h hW hWG E E' hd hc hm hph hz hpos (fun u k => ?_) ks σ ρ hσ hρ hK
    rw [hx, gsg, if_pos h.2.1, one_mul]
  · refine evalL_gadX h hW hWG E E' hd hc hm hph hz hpos (fun u k => ?_) ks σ ρ hσ hρ hK
    rw [hx, gsg, if_neg (by rw [h.2.1]; decide)]
    congr 1; ring

/-- The boundary of a gadget: `L`'s, then the slot `K`, of `x`'s chain and dimension. -/
theorem Gad.bnd {L : LDiagram} {G : Diagram} {x : DNode} {f z i : ℕ} (h : Gad L G x f z i) :
    ∃ K, G.bnd = L.base.bnd ++ [K] ∧ G.legChain K = L.base.nodeChain x.id ∧
      G.portDim K = x.dim := by
  rcases h with ⟨hb, -, -, -, hi, hfc, -, hfi, -, hfid⟩ | ⟨hb, -, -, -, hi⟩
  · refine ⟨⟨f, false, 0⟩, by simp [Diagram.bnd, hb.2.2.2.2.2.1, hi], ?_, hfid⟩
    unfold Diagram.legChain; rw [hfi, List.append_nil]; exact hfc
  · exact ⟨_, by simp [Diagram.bnd, hb.2.2.2.2.2.1, hi], hb.chainK, hb.2.2.2.2.2.2.2.2.2⟩

theorem Gad.base {L : LDiagram} {G : Diagram} {x : DNode} {f z i : ℕ} (h : Gad L G x f z i) :
    GadBase L G x := by
  rcases h with h | h
  exacts [h.1, h.1]

/-- Equal family networks of two gadget diagrams give equal linked digit sums at every slot
digits in range. -/
theorem evalL_eq_of_gad {F : Fam} {L₁ L₂ : LDiagram} {G₁ G₂ : Diagram} {o₁ o₂ : ℕ}
    {x₁ x₂ : DNode} {f₁ f₂ z i₁ i₂ : ℕ} (ks : ℕ → ℕ) (s₁ s₂ : ℂ)
    (g₁ : Gad L₁ G₁ x₁ f₁ z i₁) (g₂ : Gad L₂ G₂ x₂ f₂ z i₂) (hW₁ : WF L₁.base) (hW₂ : WF L₂.base)
    (hc₁ : G₁.FamOK F o₁) (hc₂ : G₂.FamOK F o₂) (hbd : G₁.bndData = G₂.bndData)
    (hpos : ∀ i, 0 < F.dims i)
    (hsame : L₁.base.nodeChain x₁.id = L₂.base.nodeChain x₂.id ∧ x₁.dim = x₂.dim ∧
      L₁.rdOf = L₂.rdOf ∧ L₁.base.bndData = L₂.base.bndData)
    (E' : Env) (hd : E'.dimv = (F.env ks).dimv) (hc : E'.coef = (F.env ks).coef)
    (hm : E'.mult = (F.env ks).mult)
    (hph₁ : ∀ n ∈ L₁.base.nodes, n.id ≠ x₁.id → E'.phase n.ph = F.phase n.ph)
    (hph₂ : ∀ n ∈ L₂.base.nodes, n.id ≠ x₂.id → E'.phase n.ph = F.phase n.ph)
    (hz : ∀ k, F.phase z k = 0)
    (hx₁ : ∀ u k, ex (E'.phase (Nat.pair x₁.ph u) k) =
      ex (gsg x₁.kind * ((u : ℝ) * k / (F.env ks).dimv x₁.dim)))
    (hx₂ : ∀ u k, ex (E'.phase (Nat.pair x₂.ph u) k) =
      ex (gsg x₂.kind * ((u : ℝ) * k / (F.env ks).dimv x₂.dim)))
    (hEq : Equal (F.env ks) s₁ (G₁.toNetB o₁) s₂ (G₂.toNetB o₂)) (σ : ℕ → List ℕ → ℕ)
    (hσ : ∀ i (hi : i < L₁.base.bnd.length), ∀ t ∈ cidx ks (L₁.base.legChain L₁.base.bnd[i]),
      σ i t < (F.env ks).dimv (L₁.base.portDim L₁.base.bnd[i])) :
    s₁ * L₁.evalL E' ks σ = s₂ * L₂.evalL E' ks σ := by
  obtain ⟨hch, hdx, hrd, hbdL⟩ := hsame
  obtain ⟨hlen, hdL⟩ := Diagram.bndData_at hbdL
  have hp : ∀ i, 0 < (F.env ks).dimv i := fun i => Fam.env_dimv_pos F ks i (hpos _)
  have hd₂ : (F.env ks).dimv x₂.dim = (F.env ks).dimv x₁.dim := by rw [hdx]
  have hWG₁ : WF G₁ := Diagram.WF_of_WFb hc₁.1
  have hWG₂ : WF G₂ := Diagram.WF_of_WFb hc₂.1
  let ρ : ℕ → List ℕ → ℕ := fun i t =>
    if i < L₁.base.bnd.length then σ i t else L₁.rdU ks t % (F.env ks).dimv x₁.dim
  have hρ₁ : ∀ i < L₁.base.bnd.length, ∀ t, ρ i t = σ i t := fun i hi t => if_pos hi
  have hρ₂ : ∀ i < L₂.base.bnd.length, ∀ t, ρ i t = σ i t := fun i hi t => if_pos (hlen ▸ hi)
  have hK₁ : ∀ t ∈ cidx ks (L₁.base.nodeChain x₁.id),
      ρ L₁.base.bnd.length t = L₁.rdU ks t % (F.env ks).dimv x₁.dim :=
    fun t _ => if_neg (lt_irrefl _)
  have hK₂ : ∀ t ∈ cidx ks (L₂.base.nodeChain x₂.id),
      ρ L₂.base.bnd.length t = L₂.rdU ks t % (F.env ks).dimv x₂.dim := fun t _ => by
    rw [← hlen, hd₂, show L₂.rdU ks t = L₁.rdU ks t by simp only [rdU, hrd]]
    exact if_neg (lt_irrefl _)
  have hσ₂ : ∀ i (hi : i < L₂.base.bnd.length), ∀ t ∈ cidx ks (L₂.base.legChain L₂.base.bnd[i]),
      σ i t < (F.env ks).dimv (L₂.base.portDim L₂.base.bnd[i]) := fun i hi t ht => by
    have h₁ : i < L₁.base.bnd.length := hlen ▸ hi
    obtain ⟨e1, e2⟩ := hdL i h₁ hi
    rw [← e2] at ht
    rw [← e1]
    exact hσ i h₁ t ht
  have e₁ := evalL_gad g₁ hW₁ hWG₁ (F.env ks) E' hd hc hm hph₁ hz (hp _) hx₁ ks σ ρ hσ hρ₁ hK₁
  have e₂ := evalL_gad g₂ hW₂ hWG₂ (F.env ks) E' hd hc hm hph₂ hz (hp _) hx₂ ks σ ρ hσ₂ hρ₂ hK₂
  obtain ⟨K₁, hb₁, hKc₁, hKd₁⟩ := g₁.bnd
  have hin := range_ext g₁.base hW₁ K₁ hb₁ hKc₁ hKd₁ (F.env ks) ks σ ρ hσ hρ₁ fun t ht => by
    rw [hK₁ t ht]; exact Nat.mod_lt _ (hp _)
  have key := Diagram.evalD_eq_of_checks ks s₁ s₂ hc₁ hc₂ hbd hpos hEq ρ hin
  rw [e₁, e₂, hd₂, ← hch] at key
  have hne : (rt ((F.env ks).dimv x₁.dim))⁻¹ ^ size ks (L₁.base.nodeChain x₁.id) ≠ 0 :=
    pow_ne_zero _ (inv_ne_zero (rt_ne _ (hp _)))
  refine mul_left_cancel₀ hne ?_
  linear_combination key

end LDiagram

end ArchytasZX
