/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Digits

/-!
# Relabelling digits through a map of legs

`evalD_relabelφ`: when `φ` sends every digit `(p, t)` of `D` to a digit of `D'`, one to one and
onto, wire to wire and boundary leg to boundary leg, and the node products agree, the two digit
sums agree, `D`'s slot `i` reading `D'`'s slot of the image of its leg.
-/

namespace ArchytasZX

/-- The copy index of a digit label. -/
def dtup (l : ℕ) : List ℕ := (Encodable.decode (l / 2).unpair.2).getD []

theorem dtup_dlab (p : Port) (t : List ℕ) : dtup (dlab p t) = t := by
  simp [dtup, dlab, Nat.unpair_pair]

/-- A digit label through the map of digits `φ`. -/
def relab (φ : Port → List ℕ → Port × List ℕ) (l : ℕ) : ℕ :=
  dlab (φ (dport l) (dtup l)).1 (φ (dport l) (dtup l)).2

theorem relab_dlab (φ : Port → List ℕ → Port × List ℕ) (p : Port) (t : List ℕ) :
    relab φ (dlab p t) = dlab (φ p t).1 (φ p t).2 := by
  simp [relab, dport_dlab, dtup_dlab]

/-- A port standing for none. -/
def Port.dflt : Port := ⟨0, false, 0⟩

/-- `D`'s slot digits read from `D'`'s through `φ`. -/
def bndMap (D D' : Diagram) (φ : Port → List ℕ → Port × List ℕ) (σ' : ℕ → List ℕ → ℕ)
    (i : ℕ) (t : List ℕ) : ℕ :=
  σ' (D'.bnd.idxOf (φ (D.bnd.getD i Port.dflt) t).1) (φ (D.bnd.getD i Port.dflt) t).2

theorem mem_zipIdx' {α : Type*} (l : List α) (x : α) (i : ℕ) :
    (x, i) ∈ l.zipIdx ↔ ∃ h : i < l.length, l[i] = x := by
  rw [List.mem_zipIdx_iff_getElem?]
  simp only [List.getElem?_eq_some_iff]

theorem evalD_relabelφ (D D' : Diagram) (E : Env) (ks ks' : ℕ → ℕ) (σ' : ℕ → List ℕ → ℕ)
    (φ : Port → List ℕ → Port × List ℕ)
    (hnd : (D.dlabs ks).Nodup) (hnd' : (D'.dlabs ks').Nodup) (hbnd' : D'.bnd.Nodup)
    (hinto : ∀ p ∈ D.legs, ∀ t ∈ cidx ks (D.legChain p),
      (φ p t).1 ∈ D'.legs ∧ (φ p t).2 ∈ cidx ks' (D'.legChain (φ p t).1))
    (honto : ∀ q ∈ D'.legs, ∀ u ∈ cidx ks' (D'.legChain q),
      ∃ p ∈ D.legs, ∃ t ∈ cidx ks (D.legChain p), φ p t = (q, u))
    (hinj : ∀ p ∈ D.legs, ∀ t ∈ cidx ks (D.legChain p), ∀ p' ∈ D.legs,
      ∀ t' ∈ cidx ks (D.legChain p'), φ p t = φ p' t' → p = p' ∧ t = t')
    (hdim : ∀ p ∈ D.legs, ∀ t ∈ cidx ks (D.legChain p),
      E.dimv (D'.portDim (φ p t).1) = E.dimv (D.portDim p))
    (hwl : ∀ w ∈ D.wires, w.1 ∈ D.legs ∧ w.2 ∈ D.legs ∧ D.legChain w.1 = D.legChain w.2)
    (hW : ∀ w ∈ D.wires, ∀ t ∈ cidx ks (D.legChain w.1), ∃ w' ∈ D'.wires, ∃ u,
      u ∈ cidx ks' (D'.legChain w'.1) ∧ φ w.1 t = (w'.1, u) ∧ φ w.2 t = (w'.2, u))
    (hW' : ∀ w' ∈ D'.wires, ∀ u ∈ cidx ks' (D'.legChain w'.1), ∃ w ∈ D.wires,
      ∃ t ∈ cidx ks (D.legChain w.1), φ w.1 t = (w'.1, u) ∧ φ w.2 t = (w'.2, u))
    (hbl : ∀ p ∈ D.bnd, p ∈ D.legs)
    (hB : ∀ p ∈ D.bnd, ∀ t ∈ cidx ks (D.legChain p), (φ p t).1 ∈ D'.bnd)
    (hB' : ∀ q ∈ D'.bnd, ∀ u ∈ cidx ks' (D'.legChain q), ∃ p ∈ D.bnd,
      ∃ t ∈ cidx ks (D.legChain p), φ p t = (q, u))
    (hP : ∀ τ τ' : ℕ → ℕ, (∀ x ∈ D.dlabs ks, τ' x = τ (relab φ x)) →
      D'.prodD E ks' τ = D.prodD E ks τ') :
    D'.evalD E ks' σ' = D.evalD E ks (bndMap D D' φ σ') := by
  have dl : ∀ p ∈ D.legs, ∀ t ∈ cidx ks (D.legChain p), dlab p t ∈ D.dlabs ks := fun p hp t ht =>
    (mem_dlabs D ks _).2 ⟨p, hp, t, ht, rfl⟩
  refine evalD_relabel D D' E ks ks' _ σ' (relab φ) hnd hnd' (fun x => ?_) ?_ ?_ ?_
  · rw [mem_dlabs]
    constructor
    · rintro ⟨q, hq, u, hu, rfl⟩
      obtain ⟨p, hp, t, ht, he⟩ := honto q hq u hu
      exact ⟨_, dl p hp t ht, by rw [relab_dlab, he]⟩
    · rintro ⟨y, hy, rfl⟩
      obtain ⟨p, hp, t, ht, rfl⟩ := (mem_dlabs D ks _).1 hy
      obtain ⟨h1, h2⟩ := hinto p hp t ht
      exact ⟨_, h1, _, h2, relab_dlab φ p t⟩
  · intro x hx y hy he
    obtain ⟨p, hp, t, ht, rfl⟩ := (mem_dlabs D ks _).1 hx
    obtain ⟨p', hp', t', ht', rfl⟩ := (mem_dlabs D ks _).1 hy
    rw [relab_dlab, relab_dlab] at he
    obtain ⟨e1, e2⟩ := dlab_inj he
    obtain ⟨rfl, rfl⟩ := hinj p hp t ht p' hp' t' ht' (Prod.ext e1 e2)
    rfl
  · intro x hx
    obtain ⟨p, hp, t, ht, rfl⟩ := (mem_dlabs D ks _).1 hx
    rw [relab_dlab, dport_dlab, dport_dlab]
    exact hdim p hp t ht
  · intro τ τ' hl hs hs'
    refine ⟨?_, hP τ τ' hl⟩
    have hwv : ∀ w ∈ D.wires, ∀ t ∈ cidx ks (D.legChain w.1),
        τ' (dlab w.1 t) = τ (dlab (φ w.1 t).1 (φ w.1 t).2) ∧
        τ' (dlab w.2 t) = τ (dlab (φ w.2 t).1 (φ w.2 t).2) := fun w hw t ht => by
      obtain ⟨h1, h2, h3⟩ := hwl w hw
      exact ⟨by rw [hl _ (dl _ h1 t ht), relab_dlab],
        by rw [hl _ (dl _ h2 t (h3 ▸ ht)), relab_dlab]⟩
    unfold Diagram.pinsD
    refine and_congr ⟨fun h w hw t ht => ?_, fun h w' hw' u hu => ?_⟩ ⟨fun h q hq t ht => ?_, ?_⟩
    · obtain ⟨w', hw', u, hu, e1, e2⟩ := hW w hw t ht
      rw [(hwv w hw t ht).1, (hwv w hw t ht).2, e1, e2]
      exact h w' hw' u hu
    · obtain ⟨w, hw0, t, ht, e1, e2⟩ := hW' w' hw' u hu
      have := h w hw0 t ht
      rw [(hwv w hw0 t ht).1, (hwv w hw0 t ht).2, e1, e2] at this
      exact this
    · obtain ⟨p, i⟩ := q
      obtain ⟨hi, rfl⟩ := (mem_zipIdx' _ _ _).1 hq
      have hp : D.bnd[i] ∈ D.bnd := List.getElem_mem hi
      have hq' := hB _ hp t ht
      set r := φ D.bnd[i] t with hr
      have hidx := List.idxOf_lt_length_of_mem hq'
      have hmem : (r.1, D'.bnd.idxOf r.1) ∈ D'.bnd.zipIdx :=
        (mem_zipIdx' _ _ _).2 ⟨hidx, List.getElem_idxOf hidx⟩
      have hu : r.2 ∈ cidx ks' (D'.legChain r.1) := (hinto _ (hbl _ hp) t ht).2
      have := h _ hmem r.2 hu
      simp only at this
      rw [hl _ (dl _ (hbl _ hp) t ht), relab_dlab, hs', bndMap, List.getD_eq_getElem _ _ hi]
      rw [hs] at this
      exact this
    · intro h q hq u hu
      obtain ⟨q, i'⟩ := q
      obtain ⟨hi', rfl⟩ := (mem_zipIdx' _ _ _).1 hq
      obtain ⟨p, hp, t, ht, he⟩ := hB' _ (List.getElem_mem hi') u hu
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
      have := h (D.bnd[i], i) ((mem_zipIdx' _ _ _).2 ⟨hi, rfl⟩) t ht
      simp only at this
      rw [hl _ (dl _ (hbl _ hp) t ht), relab_dlab, hs', bndMap, List.getD_eq_getElem _ _ hi,
        he] at this
      simp only at this ⊢
      rw [hs]
      rw [List.Nodup.idxOf_getElem hbnd'] at this
      exact this

end ArchytasZX
