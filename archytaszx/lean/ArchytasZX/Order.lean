/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamCheck

/-!
# The slot reading of the expansion as an index map

`SDom D ks`: every slot `i` with every copy index `t` of its chain. `bidx D ks` sends each slot
digit of `D` to a slot of `D.expand ks`, one to one and onto, keeping dimensions, and `bmap` reads
`σ'` there (`bmap_read`).
-/

namespace ArchytasZX

namespace Diagram

/-- Every slot with every copy index of its chain. -/
def SDom (D : Diagram) (ks : ℕ → ℕ) : List (ℕ × List ℕ) :=
  (List.range D.bnd.length).flatMap fun i => (cidx ks (D.legChain (D.bnd.getD i Port.dflt))).map (i, ·)

theorem mem_SDom (D : Diagram) (ks : ℕ → ℕ) (x : ℕ × List ℕ) :
    x ∈ D.SDom ks ↔ ∃ h : x.1 < D.bnd.length, x.2 ∈ cidx ks (D.legChain D.bnd[x.1]) := by
  obtain ⟨i, t⟩ := x
  simp only [SDom, List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨j, hj, u, hu, rfl, rfl⟩
    exact ⟨hj, by rwa [List.getD_eq_getElem _ _ hj] at hu⟩
  · rintro ⟨hi, ht⟩
    exact ⟨i, hi, t, by rwa [List.getD_eq_getElem _ _ hi], rfl, rfl⟩

/-- The slot digit `x` of `D` read through `φ` as a slot digit of `D'`. -/
def ixMap (D D' : Diagram) (φ : Port → List ℕ → Port × List ℕ) (x : ℕ × List ℕ) : ℕ × List ℕ :=
  (D'.bnd.idxOf (φ (D.bnd.getD x.1 Port.dflt) x.2).1, (φ (D.bnd.getD x.1 Port.dflt) x.2).2)

theorem bndMap_ix (D D' : Diagram) (φ : Port → List ℕ → Port × List ℕ) (σ' : ℕ → List ℕ → ℕ)
    (i : ℕ) (t : List ℕ) :
    bndMap D D' φ σ' i t = σ' (ixMap D D' φ (i, t)).1 (ixMap D D' φ (i, t)).2 := rfl

/-- `φ` sends the boundary digits of `D` onto those of `D'`, one to one, keeping dimensions. -/
structure BndBij (D D' : Diagram) (ks : ℕ → ℕ) (φ : Port → List ℕ → Port × List ℕ) : Prop where
  into : ∀ p ∈ D.bnd, ∀ t ∈ cidx ks (D.legChain p),
    (φ p t).1 ∈ D'.bnd ∧ (φ p t).2 ∈ cidx ks (D'.legChain (φ p t).1)
  onto : ∀ q ∈ D'.bnd, ∀ u ∈ cidx ks (D'.legChain q),
    ∃ p ∈ D.bnd, ∃ t ∈ cidx ks (D.legChain p), φ p t = (q, u)
  inj : ∀ p ∈ D.bnd, ∀ t ∈ cidx ks (D.legChain p), ∀ p' ∈ D.bnd, ∀ t' ∈ cidx ks (D.legChain p'),
    φ p t = φ p' t' → p = p' ∧ t = t'
  dim : ∀ p ∈ D.bnd, ∀ t ∈ cidx ks (D.legChain p), D'.portDim (φ p t).1 = D.portDim p
  nd : D.bnd.Nodup
  nd' : D'.bnd.Nodup

variable {D D' : Diagram} {ks : ℕ → ℕ} {φ : Port → List ℕ → Port × List ℕ}

theorem idxOf_getElem' {l : List Port} (hnd : l.Nodup) (i : ℕ) (hi : i < l.length) :
    l.idxOf l[i] = i := by
  have := List.get_idxOf hnd ⟨i, hi⟩
  simpa using this

theorem BndBij.mem (h : BndBij D D' ks φ) {x : ℕ × List ℕ} (hx : x ∈ D.SDom ks) :
    ixMap D D' φ x ∈ D'.SDom ks := by
  obtain ⟨hi, ht⟩ := (mem_SDom D ks x).1 hx
  have hg : D.bnd.getD x.1 Port.dflt = D.bnd[x.1] := List.getD_eq_getElem _ _ hi
  obtain ⟨h1, h2⟩ := h.into _ (List.getElem_mem hi) _ ht
  have hlt : D'.bnd.idxOf (φ D.bnd[x.1] x.2).1 < D'.bnd.length := List.idxOf_lt_length_of_mem h1
  refine (mem_SDom D' ks _).2 ⟨by simp only [ixMap, hg]; exact hlt, ?_⟩
  simp only [ixMap, hg]
  rwa [List.getElem_idxOf]

theorem BndBij.inj' (h : BndBij D D' ks φ) {x y : ℕ × List ℕ} (hx : x ∈ D.SDom ks)
    (hy : y ∈ D.SDom ks) (e : ixMap D D' φ x = ixMap D D' φ y) : x = y := by
  obtain ⟨hi, ht⟩ := (mem_SDom D ks x).1 hx
  obtain ⟨hj, hu⟩ := (mem_SDom D ks y).1 hy
  have hgx : D.bnd.getD x.1 Port.dflt = D.bnd[x.1] := List.getD_eq_getElem _ _ hi
  have hgy : D.bnd.getD y.1 Port.dflt = D.bnd[y.1] := List.getD_eq_getElem _ _ hj
  simp only [ixMap, hgx, hgy, Prod.mk.injEq] at e
  have m1 := (h.into _ (List.getElem_mem hi) _ ht).1
  have m2 := (h.into _ (List.getElem_mem hj) _ hu).1
  have e1 : (φ D.bnd[x.1] x.2).1 = (φ D.bnd[y.1] y.2).1 := by
    rw [← List.getElem_idxOf (List.idxOf_lt_length_of_mem m1),
      ← List.getElem_idxOf (List.idxOf_lt_length_of_mem m2)]
    simp only [e.1]
  obtain ⟨hp, htt⟩ := h.inj _ (List.getElem_mem hi) _ ht _ (List.getElem_mem hj) _ hu
    (Prod.ext e1 e.2)
  have := (List.Nodup.getElem_inj_iff h.nd).1 hp
  exact Prod.ext this htt

theorem BndBij.onto' (h : BndBij D D' ks φ) {y : ℕ × List ℕ} (hy : y ∈ D'.SDom ks) :
    ∃ x ∈ D.SDom ks, ixMap D D' φ x = y := by
  obtain ⟨hs, hu⟩ := (mem_SDom D' ks y).1 hy
  obtain ⟨p, hp, t, ht, he⟩ := h.onto _ (List.getElem_mem hs) _ hu
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
  refine ⟨(i, t), (mem_SDom D ks _).2 ⟨hi, ht⟩, ?_⟩
  have hg : D.bnd.getD i Port.dflt = D.bnd[i] := List.getD_eq_getElem _ _ hi
  simp only [ixMap, hg, he]
  rw [idxOf_getElem' h.nd' _ hs]

theorem BndBij.dim' (h : BndBij D D' ks φ) {x : ℕ × List ℕ} (hx : x ∈ D.SDom ks) :
    D'.portDim (D'.bnd.getD (ixMap D D' φ x).1 Port.dflt) = D.portDim (D.bnd.getD x.1 Port.dflt) := by
  obtain ⟨hi, ht⟩ := (mem_SDom D ks x).1 hx
  have hg : D.bnd.getD x.1 Port.dflt = D.bnd[x.1] := List.getD_eq_getElem _ _ hi
  have m1 := (h.into _ (List.getElem_mem hi) _ ht).1
  have hlt := List.idxOf_lt_length_of_mem m1
  simp only [ixMap, hg, List.getD_eq_getElem _ _ hlt, List.getElem_idxOf]
  exact h.dim _ (List.getElem_mem hi) _ ht

theorem WF.bnd_nodup {D : Diagram} (hW : WF D) : D.bnd.Nodup := by
  have := hW.perm.nodup_iff.2 (legs_nodup_of D hW.ids)
  exact (List.nodup_append.1 (List.nodup_append.1 this).2.1).2.1

theorem bij_perm (hW : WF D) (o i : List Port) (ho : o.Perm D.outs) (hi : i.Perm D.ins) :
    BndBij D { D with outs := o, ins := i } ks (fun p t => (p, t)) := by
  have hp := bnd_perm D o i ho hi
  have hc : ∀ q, ({ D with outs := o, ins := i } : Diagram).legChain q = D.legChain q :=
    legChain_congr (D := D) rfl rfl
  have hd : ∀ q, ({ D with outs := o, ins := i } : Diagram).portDim q = D.portDim q :=
    portDim_congr (D := D) rfl
  refine ⟨fun p hp' t ht => ⟨hp.mem_iff.2 hp', by rw [hc]; exact ht⟩,
    fun q hq u hu => ⟨q, hp.mem_iff.1 hq, u, by rw [← hc]; exact hu, rfl⟩,
    fun p _ t _ p' _ t' _ e => by simpa using e, fun p _ t _ => hd p, WF.bnd_nodup hW,
    hp.nodup_iff.2 (WF.bnd_nodup hW)⟩

theorem bij_port {b : DBox} {k : ℕ} (hkm : ks b.mult = k) (hW : WF D) (hb : b ∈ D.boxes)
    (hbn : b.nodes = []) (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) :
    BndBij D (D.growPorts b k) ks (φP b k) := by
  have hbl : ∀ p ∈ D.bnd, p ∈ D.legs := fun p hp => bnd_legs hW hp
  refine ⟨fun p hp t ht => ?_, fun q hq u hu => ?_, fun p hp t ht p' hp' t' ht' e => ?_,
    fun p hp t ht => ?_, WF.bnd_nodup hW, WF.bnd_nodup (ep_WF D b k hW hb hbn hleaf)⟩
  · obtain ⟨h1, h2⟩ := φP_spec D b k hkm hW hb hbn hleaf p (hbl p hp) t ht
    exact ⟨by rw [ep_bnd]; exact List.mem_flatMap.2 ⟨p, hp, h1⟩, h2⟩
  · rw [ep_bnd] at hq
    obtain ⟨p, hp, hqp⟩ := List.mem_flatMap.1 hq
    obtain ⟨t, ht, he⟩ := φP_onto D b k hkm hW hb hbn hleaf p (hbl p hp) q hqp u hu
    exact ⟨p, hp, t, ht, he⟩
  · have h1 := (φP_spec D b k hkm hW hb hbn hleaf p (hbl p hp) t ht).1
    have h2 := (φP_spec D b k hkm hW hb hbn hleaf p' (hbl p' hp') t' ht').1
    rw [e] at h1
    have hpp := ep_disj D b k hW.ids (hbl p' hp') (hbl p hp) h2 h1
    subst hpp
    exact ⟨rfl, φP_inj D b k hkm hW hb hbn hleaf p' (hbl p' hp') t t' ht ht' e⟩
  · exact ep_portDim D b k hW p (hbl p hp) _ (φP_spec D b k hkm hW hb hbn hleaf p (hbl p hp) t ht).1

theorem bij_node {b : DBox} (hW : WF D) (hb : b ∈ D.boxes) (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id)
    (hbp : b.ports = []) :
    BndBij D (D.copyNodes b (ks b.mult)) ks (φS (D.nimg b (ks b.mult)) (D.gN b)) := by
  have h := cn_split ks hW hb hleaf hbp
  refine ⟨fun p hp t ht => ?_, fun q hq u hu => ?_, fun p hp t ht p' hp' t' ht' e => ?_,
    fun p hp t ht => ?_, WF.bnd_nodup hW, h.bnd'⟩
  · obtain ⟨h1, h2⟩ := h.into p (h.bl p hp) t ht
    exact ⟨(h.memB _).2 ⟨p, hp, h1⟩, h2⟩
  · obtain ⟨p, hp, hqp⟩ := (h.memB q).1 hq
    obtain ⟨t, ht, he⟩ := h.onto p (h.bl p hp) q hqp u hu
    exact ⟨p, hp, t, ht, he⟩
  · exact h.inj p (h.bl p hp) t ht p' (h.bl p' hp') t' ht' e
  · exact h.dim p (h.bl p hp) _ (h.into p (h.bl p hp) t ht).1

/-- An index map sending `D`'s slot digits onto `D'`'s, one to one, keeping dimensions. -/
structure IxBij (D D' : Diagram) (ks : ℕ → ℕ) (f : ℕ × List ℕ → ℕ × List ℕ) : Prop where
  mem : ∀ x ∈ D.SDom ks, f x ∈ D'.SDom ks
  inj : ∀ x ∈ D.SDom ks, ∀ y ∈ D.SDom ks, f x = f y → x = y
  onto : ∀ y ∈ D'.SDom ks, ∃ x ∈ D.SDom ks, f x = y
  dim : ∀ x ∈ D.SDom ks,
    D'.portDim (D'.bnd.getD (f x).1 Port.dflt) = D.portDim (D.bnd.getD x.1 Port.dflt)

theorem BndBij.ix (h : BndBij D D' ks φ) : IxBij D D' ks (ixMap D D' φ) :=
  ⟨fun _ hx => h.mem hx, fun _ hx _ hy e => h.inj' hx hy e, fun _ hy => h.onto' hy,
    fun _ hx => h.dim' hx⟩

theorem IxBij.id (D : Diagram) (ks : ℕ → ℕ) : IxBij D D ks id :=
  ⟨fun _ hx => hx, fun _ _ _ _ e => e, fun y hy => ⟨y, hy, rfl⟩, fun _ _ => rfl⟩

theorem IxBij.comp {D₁ D₂ D₃ : Diagram} {f g : ℕ × List ℕ → ℕ × List ℕ} (h₁ : IxBij D₁ D₂ ks f)
    (h₂ : IxBij D₂ D₃ ks g) : IxBij D₁ D₃ ks (g ∘ f) := by
  refine ⟨fun x hx => h₂.mem _ (h₁.mem x hx), fun x hx y hy e => h₁.inj x hx y hy
    (h₂.inj _ (h₁.mem x hx) _ (h₁.mem y hy) e), fun z hz => ?_, fun x hx => ?_⟩
  · obtain ⟨y, hy, rfl⟩ := h₂.onto z hz
    obtain ⟨x, hx, rfl⟩ := h₁.onto y hy
    exact ⟨x, hx, rfl⟩
  · simp only [Function.comp]
    rw [h₂.dim _ (h₁.mem x hx), h₁.dim x hx]

variable (D) (ks)

/-- One instantiation step at the index level. -/
def stepIx (b : DBox) (x : ℕ × List ℕ) : ℕ × List ℕ :=
  let D₀ := if ks b.mult = 0 then D.gather b else D
  if b.ports = [] then
    ixMap D₀ (D₀.copyNodes b (ks b.mult)) (φS (D₀.nimg b (ks b.mult)) (D₀.gN b))
      (ixMap D D₀ (fun p t => (p, t)) x)
  else ixMap D₀ (D₀.growPorts b (ks b.mult)) (φP b (ks b.mult)) (ixMap D D₀ (fun p t => (p, t)) x)

theorem stepMap_ix (b : DBox) (σ' : ℕ → List ℕ → ℕ) (i : ℕ) (t : List ℕ) :
    D.stepMap ks b σ' i t = σ' (D.stepIx ks b (i, t)).1 (D.stepIx ks b (i, t)).2 := by
  unfold stepMap nodeMap portMap stepIx
  split_ifs <;> rfl

variable {D} {ks}

theorem stepIx_bij (hW : WF D) {b : DBox} (hb : b ∈ D.boxes)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) : IxBij D (D.step ks b) ks (D.stepIx ks b) := by
  obtain ⟨o, i, e, ho, hi⟩ := gather_eq D b
  set D₀ := if ks b.mult = 0 then D.gather b else D with hD₀
  set o' := if ks b.mult = 0 then o else D.outs with ho'
  set i' := if ks b.mult = 0 then i else D.ins with hi'
  have hD₀' : D₀ = { D with outs := o', ins := i' } := by
    rw [hD₀, ho', hi']; split_ifs
    · exact e
    · rfl
  have h0 : BndBij D D₀ ks (fun p t => (p, t)) := by
    rw [hD₀']
    exact bij_perm hW _ _ (by rw [ho']; split_ifs <;> simp [ho]) (by rw [hi']; split_ifs <;> simp [hi])
  obtain ⟨hW₀, hb₀⟩ := pre_facts D b (ks b.mult) hW
  rw [← hD₀] at hW₀ hb₀
  have hb' : b ∈ D₀.boxes := hb₀ ▸ hb
  have hleaf' : ∀ y ∈ D₀.boxes, y.parent ≠ some b.id := hb₀ ▸ hleaf
  unfold step stepIx expandNode expandPort
  rw [← hD₀]
  split_ifs with hbp
  · exact h0.ix.comp (bij_node hW₀ hb' hleaf' hbp).ix
  · have hbn : b.nodes = [] := (hW.kinds b hb).resolve_right hbp
    exact h0.ix.comp (bij_port rfl hW₀ hb' hbn hleaf').ix

variable (ks)

/-- The composite index map of `fuel` instantiation steps. -/
def bidxFuel : ℕ → Diagram → ℕ × List ℕ → ℕ × List ℕ
  | 0, _, x => x
  | f + 1, D, x => match D.pick with
    | none => x
    | some b => bidxFuel f (D.step ks b) (D.stepIx ks b x)

theorem bmapFuel_read : ∀ (f : ℕ) (D : Diagram) (σ' : ℕ → List ℕ → ℕ) (i : ℕ) (t : List ℕ),
    bmapFuel ks f D σ' i t = σ' (bidxFuel ks f D (i, t)).1 (bidxFuel ks f D (i, t)).2 := by
  intro f
  induction f with
  | zero => intro D σ' i t; rfl
  | succ f ih =>
    intro D σ' i t
    simp only [bmapFuel, bidxFuel]
    cases h : D.pick with
    | none => rfl
    | some b =>
      show D.stepMap ks b (bmapFuel ks f (D.step ks b) σ') i t = _
      rw [stepMap_ix, ih]

variable {ks}

theorem bidxFuel_bij : ∀ (f : ℕ) (D : Diagram), WF D → D.boxes.length < f →
    IxBij D (expandFuel ks f D) ks (bidxFuel ks f D) := by
  intro f
  induction f with
  | zero => intro D _ h; omega
  | succ f ih =>
    intro D hW hf
    rw [expandFuel_succ]
    simp only [bidxFuel]
    cases hp : D.pick with
    | none => exact IxBij.id D ks
    | some b =>
      obtain ⟨hb, hleaf⟩ := pick_some hp
      obtain ⟨-, hW', hlt⟩ := step_facts D ks hW hb hleaf ⟨fun _ => 0, fun _ _ => 0, [], fun _ => 0, fun _ _ => 1⟩
      exact (stepIx_bij hW hb hleaf).comp (ih _ hW' (by omega))

variable (D) (ks)

/-- Each slot digit's slot in the expansion. -/
def bidx (x : ℕ × List ℕ) : ℕ × List ℕ := bidxFuel ks (D.boxes.length + 1) D x

theorem bmap_read (σ' : ℕ → List ℕ → ℕ) (i : ℕ) (t : List ℕ) :
    D.bmap ks σ' i t = σ' (D.bidx ks (i, t)).1 (D.bidx ks (i, t)).2 :=
  bmapFuel_read ks _ D σ' i t

variable {D} {ks}

theorem bidx_bij (hW : WF D) : IxBij D (D.expand ks) ks (D.bidx ks) :=
  bidxFuel_bij _ D hW (by omega)

end Diagram

end ArchytasZX
