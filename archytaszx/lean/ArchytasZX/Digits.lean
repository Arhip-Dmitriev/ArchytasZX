/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Elim

/-!
# Boxed diagrams over digits

A leg whose chain of multiplicities is `c` carries one plain value for every copy index `t` of
`c` (`cidx ks c`, lexicographic, outermost coordinate first): the label `dlab p t`. `evalD` sums
over every such digit the product, over every node and every copy index of the node's own boxes,
of the node's generator at the digits of its legs under that index, where every wire's two legs
agree digit by digit and every boundary leg's digits read its slot's digits `slab i t`. A leg
whose chain is longer than its node's reads the rest of its chain as extra legs, in `rcidx` order.
-/

namespace ArchytasZX

/-- The copy indices along the chain `c`, outermost coordinate first, in lexicographic order. -/
def cidx (ks : ℕ → ℕ) : List ℕ → List (List ℕ)
  | [] => [[]]
  | m :: c => (List.range (ks m)).flatMap fun i => (cidx ks c).map (i :: ·)

/-- The copy indices along `c`, innermost coordinate slowest. -/
def rcidx (ks : ℕ → ℕ) (c : List ℕ) : List (List ℕ) := (cidx ks c.reverse).map List.reverse

/-- Digit `t` of the leg `p`. -/
def dlab (p : Port) (t : List ℕ) : ℕ := 2 * Nat.pair p.code (Encodable.encode t)

/-- Digit `t` of the slot `i`. -/
def slab (i : ℕ) (t : List ℕ) : ℕ := 2 * Nat.pair i (Encodable.encode t) + 1

/-- The port of a digit label. -/
def dport (l : ℕ) : Port := Port.decode (l / 2).unpair.1

/-- The slots' digit values at their labels. -/
def slotsD (σ : ℕ → List ℕ → ℕ) (l : ℕ) : ℕ :=
  if l % 2 = 1 then σ (l / 2).unpair.1 ((Encodable.decode (l / 2).unpair.2).getD []) else 0

theorem mem_cidx (ks : ℕ → ℕ) (c t : List ℕ) :
    t ∈ cidx ks c ↔ List.Forall₂ (fun i m => i < ks m) t c := by
  induction c generalizing t with
  | nil => cases t <;> simp [cidx]
  | cons m c ih =>
    cases t with
    | nil => simp [cidx]
    | cons i t => simp [cidx, ih, and_comm]

theorem cidx_nodup (ks : ℕ → ℕ) (c : List ℕ) : (cidx ks c).Nodup := by
  induction c with
  | nil => simp [cidx]
  | cons m c ih =>
    simp only [cidx]
    refine List.nodup_flatMap.2 ⟨fun i _ => ih.map fun a b h => by simpa using h, ?_⟩
    refine List.nodup_range.pairwise_of_forall_ne fun i _ j _ hij => ?_
    simp only [Function.onFun, List.disjoint_left, List.mem_map]
    rintro _ ⟨a, -, rfl⟩ ⟨b, -, h⟩
    exact hij (List.cons.inj h).1.symm

theorem mem_rcidx (ks : ℕ → ℕ) (c t : List ℕ) : t ∈ rcidx ks c ↔ t ∈ cidx ks c := by
  simp only [rcidx, List.mem_map, mem_cidx]
  constructor
  · rintro ⟨u, hu, rfl⟩
    exact List.forall₂_reverse_iff.1 (by simpa using hu)
  · intro h
    exact ⟨t.reverse, List.forall₂_reverse_iff.2 h, by simp⟩

theorem rcidx_append_single (ks : ℕ → ℕ) (c : List ℕ) (m : ℕ) :
    rcidx ks (c ++ [m]) = (List.range (ks m)).flatMap fun j => (rcidx ks c).map (· ++ [j]) := by
  simp [rcidx, cidx, List.map_flatMap, Function.comp_def]

theorem mem_cidx_append (ks : ℕ → ℕ) (a b t s : List ℕ) (ht : t ∈ cidx ks a) (hs : s ∈ cidx ks b) :
    t ++ s ∈ cidx ks (a ++ b) := by
  rw [mem_cidx] at *
  exact List.rel_append ht hs

theorem mem_cidx_single (ks : ℕ → ℕ) (c : List ℕ) (m : ℕ) (t : List ℕ) :
    t ∈ cidx ks (c ++ [m]) ↔ ∃ u j, t = u ++ [j] ∧ u ∈ cidx ks c ∧ j < ks m := by
  rw [mem_cidx]
  constructor
  · intro h
    have h' := List.forall₂_reverse_iff.2 h
    rw [List.reverse_append, List.reverse_singleton, List.singleton_append] at h'
    obtain ⟨j, u', hj, hu', he⟩ : ∃ j u', j < ks m ∧ List.Forall₂ (fun i m => i < ks m) u' c.reverse ∧
        t.reverse = j :: u' := by
      generalize t.reverse = r at h'
      cases h' with
      | cons hj hu => exact ⟨_, _, hj, hu, rfl⟩
    refine ⟨u'.reverse, j, ?_, (mem_cidx ks c _).2 ?_, hj⟩
    · have := congrArg List.reverse he
      simpa using this
    · exact List.forall₂_reverse_iff.1 (by simpa using hu')
  · rintro ⟨u, j, rfl, hu, hj⟩
    exact List.rel_append ((mem_cidx ks c u).1 hu) (List.Forall₂.cons hj List.Forall₂.nil)

theorem dlab_inj {p q : Port} {t u : List ℕ} (h : dlab p t = dlab q u) : p = q ∧ t = u := by
  simp only [dlab, Nat.mul_left_cancel_iff (by norm_num : 0 < 2)] at h
  have := Nat.pair_eq_pair.1 h
  exact ⟨Port.code_inj this.1, Encodable.encode_injective this.2⟩

theorem dlab_ne_slab (p : Port) (t : List ℕ) (i : ℕ) (u : List ℕ) : dlab p t ≠ slab i u := by
  simp only [dlab, slab]; omega

theorem dport_dlab (p : Port) (t : List ℕ) : dport (dlab p t) = p := by
  simp [dport, dlab, Port.decode_code]

theorem slotsD_slab (σ : ℕ → List ℕ → ℕ) (i : ℕ) (t : List ℕ) : slotsD σ (slab i t) = σ i t := by
  have h1 : slab i t % 2 = 1 := by simp only [slab]; omega
  have h2 : slab i t / 2 = Nat.pair i (Encodable.encode t) := by simp only [slab]; omega
  simp [slotsD, h1, h2, Nat.unpair_pair]

theorem slotsD_dlab (σ : ℕ → List ℕ → ℕ) (p : Port) (t : List ℕ) : slotsD σ (dlab p t) = 0 := by
  have h1 : dlab p t % 2 = 0 := by simp only [dlab]; omega
  simp [slotsD, h1]

namespace Diagram

variable (D : Diagram)

/-- A leg's chain: its node's, then the multiplicities its leg bundles beyond. -/
def legChain (p : Port) : List ℕ := D.nodeChain p.node ++ D.tail p

/-- Every leg's digits. -/
def dlabs (ks : ℕ → ℕ) : List ℕ := D.legs.flatMap fun p => (cidx ks (D.legChain p)).map (dlab p)

/-- Copy `t` of the node `n`, each leg read as its digits under `t`, its tail's in `rcidx` order. -/
def dnode (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) : Node :=
  let legs := fun (out : Bool) (len : ℕ) => (List.range len).flatMap fun i =>
    (rcidx ks (D.tail ⟨n.id, out, i⟩)).map fun s => dlab ⟨n.id, out, i⟩ (t ++ s)
  ⟨n.kind, legs true n.outs.length, legs false n.ins.length, n.dim, n.ph⟩

/-- Every wire's two legs agree digit by digit, every boundary leg's digits are its slot's. -/
def pinsD (ks : ℕ → ℕ) (ν : ℕ → ℕ) : Prop :=
  (∀ w ∈ D.wires, ∀ t ∈ cidx ks (D.legChain w.1), ν (dlab w.1 t) = ν (dlab w.2 t)) ∧
    (∀ q ∈ D.bnd.zipIdx, ∀ t ∈ cidx ks (D.legChain q.1), ν (dlab q.1 t) = ν (slab q.2 t))

/-- The product over the nodes and their copies of the node values at the digits `ν`. -/
noncomputable def prodD (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : ℂ :=
  (D.nodes.map fun n => ((cidx ks (D.nodeChain n.id)).map fun t => val E ν (D.dnode ks n t)).prod).prod

open Classical in
/-- The diagram at the multiplicities `ks`, over digits: slot `i`'s digit `t` is `σ i t`. -/
noncomputable def evalD (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) : ℂ :=
  sumOver (fun l => E.dimv (D.portDim (dport l))) (D.dlabs ks)
    (fun ν => if D.pinsD ks ν then D.prodD E ks ν else 0) (slotsD σ)

end Diagram

theorem mem_dlabs (D : Diagram) (ks : ℕ → ℕ) (x : ℕ) :
    x ∈ D.dlabs ks ↔ ∃ p ∈ D.legs, ∃ t ∈ cidx ks (D.legChain p), x = dlab p t := by
  simp only [Diagram.dlabs, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨p, hp, t, ht, rfl⟩; exact ⟨p, hp, t, ht, rfl⟩
  · rintro ⟨p, hp, t, ht, rfl⟩; exact ⟨p, hp, t, ht, rfl⟩

/-- Two digit sums agree when the digits of one are a relabelling `ρ` of the other's, the slots
read alike, and the integrands agree under the relabelling. -/
theorem evalD_relabel (D D' : Diagram) (E : Env) (ks ks' : ℕ → ℕ) (σ σ' : ℕ → List ℕ → ℕ)
    (ρ : ℕ → ℕ) (hnd : (D.dlabs ks).Nodup) (hnd' : (D'.dlabs ks').Nodup)
    (hmem : ∀ x, x ∈ D'.dlabs ks' ↔ ∃ y ∈ D.dlabs ks, ρ y = x)
    (hinj : ∀ x ∈ D.dlabs ks, ∀ y ∈ D.dlabs ks, ρ x = ρ y → x = y)
    (hdim : ∀ x ∈ D.dlabs ks, E.dimv (D'.portDim (dport (ρ x))) = E.dimv (D.portDim (dport x)))
    (hF : ∀ τ τ' : ℕ → ℕ, (∀ x ∈ D.dlabs ks, τ' x = τ (ρ x)) → (∀ i t, τ (slab i t) = σ' i t) →
      (∀ i t, τ' (slab i t) = σ i t) → (D'.pinsD ks' τ ↔ D.pinsD ks τ') ∧
        D'.prodD E ks' τ = D.prodD E ks τ') :
    D'.evalD E ks' σ' = D.evalD E ks σ := by
  have hperm : (D'.dlabs ks').Perm ((D.dlabs ks).map ρ) := by
    refine (List.perm_ext_iff_of_nodup hnd' (List.Nodup.map_on hinj hnd)).2 fun x => ?_
    rw [hmem x, List.mem_map]
  have hdl : ∀ x ∈ D'.dlabs ks', ∃ p t, x = dlab p t := fun x hx => by
    obtain ⟨p, -, t, -, rfl⟩ := (mem_dlabs D' ks' x).1 hx
    exact ⟨p, t, rfl⟩
  have hdl0 : ∀ x ∈ D.dlabs ks, ∃ p t, x = dlab p t := fun x hx => by
    obtain ⟨p, -, t, -, rfl⟩ := (mem_dlabs D ks x).1 hx
    exact ⟨p, t, rfl⟩
  unfold Diagram.evalD
  rw [sumOver_perm _ hperm]
  refine sumOver_map_on _ _ ρ _ hnd hinj hdim _ _ _ _ (fun τ τ' hτ hτ' hl => ?_) (fun x hx => ?_)
  · have hs : ∀ i t, τ (slab i t) = σ' i t := fun i t => by
      rw [hτ _ fun h => by
        obtain ⟨p, u, e⟩ := hdl _ ((List.Perm.mem_iff hperm).2 h)
        exact dlab_ne_slab p u i t e.symm, slotsD_slab]
    have hs' : ∀ i t, τ' (slab i t) = σ i t := fun i t => by
      rw [hτ' _ fun h => by
        obtain ⟨p, u, e⟩ := hdl0 _ h
        exact dlab_ne_slab p u i t e.symm, slotsD_slab]
    obtain ⟨h1, h2⟩ := hF τ τ' hl hs hs'
    rw [h2]
    classical
    exact if_congr h1 rfl rfl
  · obtain ⟨p, t, rfl⟩ := hdl0 x hx
    obtain ⟨q, u, e⟩ := hdl _ ((hmem _).2 ⟨_, hx, rfl⟩)
    rw [slotsD_dlab, e, slotsD_dlab]

end ArchytasZX
